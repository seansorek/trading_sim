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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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
    ridge_alpha: float = 10.0


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
    feats = make_daily_features(df, spy_df=spy_df).dropna(subset=["fwd_ret_1d"])
    n = len(feats)
    min_bars = config.train_bars + FWD_RET_HORIZON_DAYS + config.test_bars
    if n < min_bars:
        raise ValueError(
            f"Need at least {min_bars} bars for one fold "
            f"(train={config.train_bars} + embargo={FWD_RET_HORIZON_DAYS} + test={config.test_bars}), "
            f"got {n}."
        )

    X_all = _preprocess(feats[FEATURE_COLS].values.astype(np.float32))
    y_all = feats["fwd_ret_1d"].values.astype(np.float64)
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

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

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
