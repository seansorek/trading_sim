#!/usr/bin/env python3
"""
eval_report.py — Honest-evaluation report for the daily_predictor strategy.

Reports:
  * PBO  — probability of backtest overfitting over the (quantile, window) grid.
  * DSR  — deflated Sharpe per symbol, deflating the selected config's Sharpe by
           the number of configs tried and the cross-config Sharpe variance.

Real-data path fetches via the same cache-aware loader as training. Pure helpers
(compute_dsr_for_symbol) accept an in-memory price frame for testing.
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np

from base_strategy import StrategyConfig
from cpcv import cscv_pbo
from deflated_sharpe import deflated_sharpe
from ml_strategies import DailyPredictorStrategy
from oos_guard import enforce_oos_start, get_artifact_train_end
from simulation_pipeline import Backtester, ExecutionConfig
from walk_forward import (
    WalkForwardConfig, build_fold_data, fold_config_ic_matrix,
    _DEFAULT_QUANTILES, _DEFAULT_WINDOWS,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_pbo(symbols, days, db, config=None):
    """Build the fold IC matrix over _DEFAULT_QUANTILES × _DEFAULT_WINDOWS and return cscv_pbo."""
    if config is None:
        config = WalkForwardConfig()
    fold_data = build_fold_data(symbols, days, db, config)
    matrix, configs = fold_config_ic_matrix(fold_data, _DEFAULT_QUANTILES, _DEFAULT_WINDOWS)
    result = cscv_pbo(matrix)
    result["n_configs"] = len(configs)
    result["n_folds"] = int(matrix.shape[0]) if matrix.size else 0
    return result


def _backtest_sharpe(df, strat, daily_feats, pred_ret, q, w):
    """Daily returns array, annualized Sharpe, and per-period Sharpe for one (q, w) config.

    `strat`/`daily_feats`/`pred_ret` are the (q, w)-independent pipeline outputs
    (model load, make_daily_features, model.predict) computed once per symbol
    by the caller and reused across the whole grid -- only the final
    threshold step varies per config (see issue #135).

    Returns:
        (daily_values, annualized_sharpe, per_period_sharpe)

    The annualized_sharpe (sqrt(252) * mean/std) is used for ranking configs.
    The per_period_sharpe (mean/std, no annualization) must be used for
    trial_sharpe_var so it stays consistent with deflated_sharpe's internal sr
    computation which is also per-period.
    """
    sig = strat._signal_from_pred_ret(daily_feats, pred_ret, q, w)
    sig = sig.reindex(df.index).fillna(0).astype(int)
    bt = Backtester(ExecutionConfig())
    res = bt.run(df, df, sig, artifact_paths={})
    daily = res.equity_curve.resample("1D").last().dropna().pct_change().dropna()
    ann_sharpe = res.metrics.get("daily_sharpe", 0.0)
    sd = daily.std(ddof=1)
    per_period_sr = float(daily.mean() / sd) if sd > 0 else 0.0
    return daily.values, ann_sharpe, per_period_sr


def compute_dsr_for_symbol(symbol, df, quantiles=None, windows=None,
                            model_path="models/daily_predictor.pkl"):
    """Run a backtest per (q, w) config on one symbol's price frame.

    `df` is trimmed to rows after the daily_predictor model's train_end
    cutoff (+ embargo) before any backtest runs, so DSR is never computed
    over in-sample rows the model was fit on (issue #115). Pass
    model_path=None to skip this check (e.g. when df is already known to be
    out-of-sample, as in unit tests with a from-scratch fit).

    Returns a dict with keys:
      dsr, sr, sr0, p_value  — from deflated_sharpe on the best config's daily returns
      selected               — (q, w) tuple of the best config (highest Sharpe)
      n_trials               — total number of (q, w) combinations evaluated
    """
    if model_path and os.path.exists(model_path):
        try:
            import pickle
            with open(model_path, "rb") as f:
                artifact = pickle.load(f)
            train_end = get_artifact_train_end(artifact)
            df = enforce_oos_start(df, train_end, label=f"{symbol}/daily_predictor")
        except Exception as exc:
            if isinstance(exc, ValueError) and "in-sample" in str(exc):
                raise
            logger.warning("oos_guard: could not read daily_predictor artifact %s: %s",
                            model_path, exc)
        if len(df) < 300:
            raise ValueError(
                f"{symbol}: fewer than 300 out-of-sample rows remain after trimming to the "
                "daily_predictor model's train cutoff + embargo."
            )

    quantiles = quantiles or _DEFAULT_QUANTILES
    windows = windows or _DEFAULT_WINDOWS
    configs = [(q, w) for q in quantiles for w in windows]

    # Model load, make_daily_features, and model.predict are identical for
    # every (q, w) in the grid -- only the final threshold step depends on
    # the config. Run the shared pipeline once per symbol instead of once
    # per config (issue #135).
    cfg = StrategyConfig(name="daily_predictor")
    strat = DailyPredictorStrategy(cfg)
    daily_feats, pred_ret = strat._predict_returns(df)

    per_config = []       # [(config, daily_returns, annualized_sharpe, per_period_sharpe), ...]
    for (q, w) in configs:
        daily, ann_sr, pp_sr = _backtest_sharpe(df, strat, daily_feats, pred_ret, q, w)
        per_config.append(((q, w), daily, ann_sr, pp_sr))

    # Use per-period Sharpes for trial_var so it stays in the same units as
    # deflated_sharpe's internal sr (mean/std, no sqrt(252) annualization).
    sharpes_pp = np.array([pp for _, _, _, pp in per_config], dtype=float)
    trial_var = float(np.var(sharpes_pp, ddof=1)) if len(sharpes_pp) > 1 else 0.0
    # Rank by annualized Sharpe (same ordering; we just keep units consistent for DSR).
    best_i = int(np.argmax([ann for _, _, ann, _ in per_config]))
    sel_config, sel_daily, _, _ = per_config[best_i]

    dsr = deflated_sharpe(sel_daily, n_trials=len(configs), trial_sharpe_var=trial_var)
    dsr["selected"] = sel_config
    dsr["n_trials"] = len(configs)
    return dsr


def main():
    from datetime import datetime, timedelta
    from db import DB
    from predict_next_day_lite import _load_bars_cached

    parser = argparse.ArgumentParser(description="Honest-eval report for daily_predictor")
    parser.add_argument("--symbols", default="AAPL,MSFT,SPY,QQQ,NVDA")
    parser.add_argument("--days", type=int, default=2500)
    parser.add_argument("--db", default="data/trading_sim.db")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    try:
        db = DB(args.db)
    except Exception:
        db = None

    pbo = compute_pbo(symbols, args.days, db)
    print("\n=== Probability of Backtest Overfitting (q,w grid) ===")
    if pbo.get("reason"):
        print(f"PBO: n/a — {pbo['reason']}")
    else:
        print(f"PBO = {pbo['pbo']:.3f}  (folds={pbo['n_folds']}, configs={pbo['n_configs']}, "
              f"combinations={pbo['n_combinations']})")

    print("\n=== Deflated Sharpe per symbol ===")
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    dsrs = []
    for sym in symbols:
        df = _load_bars_cached(sym, start, end, db=db)
        if df is None or len(df) < 300:
            print(f"  {sym}: insufficient data")
            continue
        try:
            out = compute_dsr_for_symbol(sym, df)
        except ValueError as exc:
            print(f"  {sym}: {exc}")
            continue
        dsrs.append(out["dsr"])
        print(f"  {sym}: DSR={out['dsr']:.3f} SR={out['sr']:+.4f} SR0={out['sr0']:+.4f} "
              f"selected(q,w)={out['selected']}")
    if dsrs:
        print(f"\nMedian DSR across symbols: {np.median(dsrs):.3f}")


if __name__ == "__main__":
    main()
