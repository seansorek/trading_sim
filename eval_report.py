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


def _backtest_sharpe(df, q, w):
    """Daily returns array + annualized Sharpe for one (q, w) config on one price frame.

    Forces the config via the env-var override that DailyPredictorStrategy reads
    at highest priority, so pickle best_* keys do not shadow it.  Saves and
    restores env vars so calls are independent and never leak state.
    """
    prev_q = os.environ.get("PREDICTOR_SIGNAL_QUANTILE")
    prev_w = os.environ.get("PREDICTOR_THRESHOLD_WINDOW")
    os.environ["PREDICTOR_SIGNAL_QUANTILE"] = str(q)
    os.environ["PREDICTOR_THRESHOLD_WINDOW"] = str(w)
    try:
        cfg = StrategyConfig(name="daily_predictor")
        strat = DailyPredictorStrategy(cfg)
        sig = strat.signal(None, df)
        sig = sig.reindex(df.index).fillna(0).astype(int)
        bt = Backtester(ExecutionConfig())
        res = bt.run(df, df, sig, artifact_paths={})
        daily = res.equity_curve.resample("1D").last().dropna().pct_change().dropna()
        return daily.values, res.metrics.get("daily_sharpe", 0.0)
    finally:
        for key, prev in (("PREDICTOR_SIGNAL_QUANTILE", prev_q),
                          ("PREDICTOR_THRESHOLD_WINDOW", prev_w)):
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def compute_dsr_for_symbol(symbol, df, quantiles=None, windows=None):
    """Run a backtest per (q, w) config on one symbol's price frame.

    Returns a dict with keys:
      dsr, sr, sr0, p_value  — from deflated_sharpe on the best config's daily returns
      selected               — (q, w) tuple of the best config (highest Sharpe)
      n_trials               — total number of (q, w) combinations evaluated
    """
    quantiles = quantiles or _DEFAULT_QUANTILES
    windows = windows or _DEFAULT_WINDOWS
    configs = [(q, w) for q in quantiles for w in windows]

    per_config = []       # [(config, daily_returns, annualized_sharpe), ...]
    for (q, w) in configs:
        daily, sharpe = _backtest_sharpe(df, q, w)
        per_config.append(((q, w), daily, sharpe))

    sharpes = np.array([s for _, _, s in per_config], dtype=float)
    trial_var = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    best_i = int(np.argmax(sharpes))
    sel_config, sel_daily, _ = per_config[best_i]

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
        out = compute_dsr_for_symbol(sym, df)
        dsrs.append(out["dsr"])
        print(f"  {sym}: DSR={out['dsr']:.3f} SR={out['sr']:+.4f} SR0={out['sr0']:+.4f} "
              f"selected(q,w)={out['selected']}")
    if dsrs:
        print(f"\nMedian DSR across symbols: {np.median(dsrs):.3f}")


if __name__ == "__main__":
    main()
