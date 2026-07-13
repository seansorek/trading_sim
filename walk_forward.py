#!/usr/bin/env python3
"""
walk_forward.py — Walk-forward validation harness for the daily predictor.

Provides run_walk_forward_on_df (pure, no I/O) for testing and sweep_params
(loads data, tunes signal_quantile + threshold_window) for use by train_predictor.py.

CLI:
    python walk_forward.py --symbol SPY --train 504 --test 63 --step 21
    python walk_forward.py --symbols AAPL,MSFT,SPY --train 504 --test 63 --step 21
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import RobustScaler
from predictors.base import _scale

from daily_features import FEATURE_COLS, FWD_RET_HORIZON_DAYS, make_daily_features
from ml_strategies import compute_predictor_signal
from train_models import _preprocess, _load_symbol

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    train_bars: int = 504
    test_bars: int = 63
    step_bars: int = 21
    min_train_bars: int = 252
    ridge_alpha: float = 1.0
    l1_ratio: float = 0.5
    use_elasticnet: bool = True


def _ic(pred: np.ndarray, actual: np.ndarray) -> float:
    if len(pred) < 2 or np.std(pred) < 1e-12:
        return 0.0
    ic, _ = spearmanr(pred, actual)
    return 0.0 if np.isnan(ic) else float(ic)


def run_walk_forward_on_df(
    df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame],
    config: WalkForwardConfig,
) -> pd.DataFrame:
    """
    Run walk-forward validation on a single symbol's price DataFrame.

    Returns a DataFrame with one row per fold:
      fold, train_start, train_end, test_start, test_end, ic, dir_acc, n_test

    Raises ValueError if there are fewer bars than one complete fold
    (train_bars + FWD_RET_HORIZON_DAYS + test_bars).
    """
    feats = make_daily_features(df, spy_df=spy_df).dropna(subset=["fwd_ret_vol_adj"])
    n = len(feats)
    min_bars = config.train_bars + FWD_RET_HORIZON_DAYS + config.test_bars
    if n < min_bars:
        raise ValueError(
            f"Need at least {min_bars} bars for one fold "
            f"(train={config.train_bars} + embargo={FWD_RET_HORIZON_DAYS} + test={config.test_bars}), "
            f"got {n}."
        )

    X_all = _preprocess(feats[FEATURE_COLS].values.astype(np.float32))
    y_all = feats["fwd_ret_vol_adj"].values.astype(np.float64)
    dates = feats.index

    records = []
    fold = 0
    train_start_idx = 0

    while True:
        train_end_idx = train_start_idx + config.train_bars
        test_start_idx = train_end_idx + FWD_RET_HORIZON_DAYS
        test_end_idx = test_start_idx + config.test_bars
        if test_end_idx > n:
            break

        X_tr = X_all[train_start_idx:train_end_idx]
        y_tr = y_all[train_start_idx:train_end_idx]
        X_te = X_all[test_start_idx:test_end_idx]
        y_te = y_all[test_start_idx:test_end_idx]

        scaler = RobustScaler().fit(X_tr)
        X_tr_s = _scale(scaler, X_tr)
        X_te_s = _scale(scaler, X_te)

        if config.use_elasticnet:
            model = ElasticNet(alpha=config.ridge_alpha, l1_ratio=config.l1_ratio,
                              max_iter=5000, random_state=42)
        else:
            model = Ridge(alpha=config.ridge_alpha)
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_te_s)

        fold_ic = _ic(pred, y_te)
        dir_acc = float(np.mean(np.sign(pred) == np.sign(y_te)))

        records.append({
            "fold": fold,
            "train_start": str(dates[train_start_idx].date()),
            "train_end": str(dates[train_end_idx - 1].date()),
            "test_start": str(dates[test_start_idx].date()),
            "test_end": str(dates[test_end_idx - 1].date()),
            "ic": fold_ic,
            "dir_acc": dir_acc,
            "n_test": config.test_bars,
        })
        fold += 1
        train_start_idx += config.step_bars

    return pd.DataFrame(records)


_DEFAULT_QUANTILES = [0.60, 0.65, 0.70, 0.75, 0.80]
_DEFAULT_WINDOWS = [40, 60, 80, 100]
_FALLBACK_QUANTILE = 0.7
_FALLBACK_WINDOW = 60


def sweep_params(
    symbols: list[str],
    days: int,
    db,
    config: Optional[WalkForwardConfig] = None,
    quantiles: Optional[list[float]] = None,
    windows: Optional[list[int]] = None,
) -> tuple[float, int]:
    """
    Sweep (signal_quantile, threshold_window) over walk-forward folds for all symbols.
    Returns (best_signal_quantile, best_threshold_window) based on median IC across symbols.
    Falls back to (0.7, 60) if no symbols produce valid folds or all IC <= 0.
    """
    if config is None:
        config = WalkForwardConfig()
    if quantiles is None:
        quantiles = _DEFAULT_QUANTILES
    if windows is None:
        windows = _DEFAULT_WINDOWS

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    spy_df = _load_symbol("SPY", start, end, db)

    # Per fold: store (pred_window, y_te, test_offset) so the rolling quantile
    # has history before the test window — matches live production behaviour.
    fold_data: list[tuple[np.ndarray, np.ndarray, int]] = []

    for symbol in symbols:
        df = _load_symbol(symbol, start, end, db)
        if df is None:
            continue
        try:
            spy_arg = spy_df if symbol != "SPY" else None
            feats = make_daily_features(df, spy_df=spy_arg).dropna(subset=["fwd_ret_vol_adj"])
        except Exception as exc:
            logger.warning("sweep_params: feature error for %s: %s", symbol, exc)
            continue

        n = len(feats)
        min_bars = config.train_bars + FWD_RET_HORIZON_DAYS + config.test_bars
        if n < min_bars:
            continue

        X_all = _preprocess(feats[FEATURE_COLS].values.astype(np.float32))
        y_all = feats["fwd_ret_vol_adj"].values.astype(np.float64)
        train_start_idx = 0

        while True:
            train_end_idx = train_start_idx + config.train_bars
            test_start_idx = train_end_idx + FWD_RET_HORIZON_DAYS
            test_end_idx = test_start_idx + config.test_bars
            if test_end_idx > n:
                break

            # Fit scaler only on train, transform full window for causal rolling quantile
            X_window = X_all[train_start_idx:test_end_idx]
            scaler = RobustScaler().fit(X_all[train_start_idx:train_end_idx])
            X_window_s = _scale(scaler, X_window)

            model = Ridge(alpha=config.ridge_alpha)
            if config.use_elasticnet:
                model = ElasticNet(alpha=config.ridge_alpha, l1_ratio=config.l1_ratio,
                                  max_iter=5000, random_state=42)
            model.fit(X_window_s[:config.train_bars], y_all[train_start_idx:train_end_idx])
            pred_window = model.predict(X_window_s)
            test_offset = config.train_bars + FWD_RET_HORIZON_DAYS
            y_te = y_all[test_start_idx:test_end_idx]
            fold_data.append((pred_window, y_te, test_offset))
            train_start_idx += config.step_bars

    if not fold_data:
        logger.warning("sweep_params: no valid folds — returning defaults (%s, %d)",
                       _FALLBACK_QUANTILE, _FALLBACK_WINDOW)
        return _FALLBACK_QUANTILE, _FALLBACK_WINDOW

    best_ic = -np.inf
    best_pair = (_FALLBACK_QUANTILE, _FALLBACK_WINDOW)

    for q in quantiles:
        for w in windows:
            fold_ics = []
            for pred_window, y_te, test_offset in fold_data:
                if len(pred_window) < w:
                    continue
                signals_window = compute_predictor_signal(pred_window, q, w)
                signals_test = signals_window[test_offset:]
                active = signals_test != 0
                if active.sum() < 5:
                    fold_ics.append(0.0)
                    continue
                fold_ics.append(_ic(pred_window[test_offset:][active], y_te[active]))
            if not fold_ics:
                continue
            median_ic = float(np.median(fold_ics))
            if median_ic > best_ic:
                best_ic = median_ic
                best_pair = (q, w)

    if best_ic <= 0:
        logger.warning(
            "sweep_params: all (quantile, window) combinations produced IC <= 0 "
            "— keeping defaults (%.2f, %d)", _FALLBACK_QUANTILE, _FALLBACK_WINDOW
        )
        return _FALLBACK_QUANTILE, _FALLBACK_WINDOW

    logger.info("sweep_params: best q=%.2f w=%d median_IC=%.4f", best_pair[0], best_pair[1], best_ic)
    return best_pair


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Walk-forward validation for daily predictor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", help="Single symbol")
    group.add_argument("--symbols", help="Comma-separated symbols")
    parser.add_argument("--train", type=int, default=504, dest="train_bars")
    parser.add_argument("--test", type=int, default=63, dest="test_bars")
    parser.add_argument("--step", type=int, default=21, dest="step_bars")
    parser.add_argument("--days", type=int, default=2500)
    parser.add_argument("--db", default="data/trading_sim.db")
    args = parser.parse_args()

    from db import DB
    db = DB(args.db)
    config = WalkForwardConfig(train_bars=args.train_bars, test_bars=args.test_bars,
                               step_bars=args.step_bars)

    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    spy_df = _load_symbol("SPY", start, end, db)

    for symbol in symbols:
        df = _load_symbol(symbol, start, end, db)
        if df is None:
            logger.warning("Could not load data for %s", symbol)
            continue
        spy_arg = spy_df if symbol != "SPY" else None
        try:
            result = run_walk_forward_on_df(df, spy_arg, config)
        except ValueError as exc:
            logger.error("%s: %s", symbol, exc)
            continue
        print(f"\n=== {symbol} Walk-Forward Results ===")
        print(result[["fold", "train_start", "test_start", "ic", "dir_acc"]].to_string(index=False))
        print(f"Mean IC: {result['ic'].mean():.4f}  Median IC: {result['ic'].median():.4f}")

    if len(symbols) > 1 or args.symbols:
        print("\n=== Parameter Sweep ===")
        best_q, best_w = sweep_params(symbols, args.days, db, config)
        print(f"Best: signal_quantile={best_q:.2f}  threshold_window={best_w}")


if __name__ == "__main__":
    main()
