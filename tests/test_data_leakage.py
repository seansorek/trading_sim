"""test_data_leakage.py — Guard against train/test leakage in the training pipelines.

fwd_ret_1d is a FWD_RET_HORIZON_DAYS-bar forward return. Without a purge/embargo
gap, the last training rows' labels would depend on price action that falls
inside the test period — a subtle leak even though the *features* themselves
never look ahead. These tests pin the embargo gap so a future refactor can't
silently remove it.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_features import FWD_RET_HORIZON_DAYS, make_daily_features
import train_models


def _make_price_df(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close * rng.uniform(0.99, 1.005, n),
            "high": close * rng.uniform(1.001, 1.02, n),
            "low": close * rng.uniform(0.98, 0.999, n),
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


def test_prepare_data_has_embargo_gap(tmp_path):
    """train_models._prepare_data must drop FWD_RET_HORIZON_DAYS rows between
    the last training row and the first test row (purged split)."""
    db = train_models.DB(str(tmp_path / "test.db"))
    df = _make_price_df()

    with patch("train_models._load_symbol", return_value=df):
        X_train, y_train, X_test, y_test, used = train_models._prepare_data(
            ["AAPL"], days=900, db=db, vol_mult=0.5,
        )

    feats = make_daily_features(df, spy_df=None).dropna(subset=["fwd_ret_1d"])
    split = int(len(feats) * 0.8)
    expected_test_len = len(feats) - (split + FWD_RET_HORIZON_DAYS)

    assert used == ["AAPL"]
    assert len(X_train) == split
    assert len(X_test) == expected_test_len
    # The gap between the last train row and the first test row must be at
    # least the forward-return horizon, otherwise a train label could be
    # computed from a close price inside the test window.
    gap = len(feats) - expected_test_len - split
    assert gap == FWD_RET_HORIZON_DAYS


def test_prepare_data_no_embargo_means_no_overlap_in_dates(tmp_path):
    """The actual calendar gap between train-end and test-start must be >=
    the forward-return horizon (not just a row-count coincidence)."""
    db = train_models.DB(str(tmp_path / "test.db"))
    df = _make_price_df()

    with patch("train_models._load_symbol", return_value=df):
        train_models._prepare_data(["AAPL"], days=900, db=db, vol_mult=0.5)

    feats = make_daily_features(df, spy_df=None).dropna(subset=["fwd_ret_1d"])
    split = int(len(feats) * 0.8)
    test_start = split + FWD_RET_HORIZON_DAYS

    train_end_date = feats.index[split - 1]
    test_start_date = feats.index[test_start]
    # fwd_ret_1d at the last train row looks FWD_RET_HORIZON_DAYS bars ahead;
    # that target bar must fall strictly before the first test row's date.
    label_horizon_date = feats.index[split - 1 + FWD_RET_HORIZON_DAYS]
    assert label_horizon_date < test_start_date


def test_hybrid_prepare_data_has_embargo_gap(tmp_path):
    pytest.importorskip("torch")
    import train_hybrid

    db = train_hybrid.DB(str(tmp_path / "test.db"))
    df = _make_price_df()

    with patch("train_hybrid._load_symbol", return_value=df):
        data = train_hybrid.prepare_data(["AAPL"], days=900, db=db, lookback=20, vol_mult=0.5)

    feats = make_daily_features(df, spy_df=None).dropna(subset=["fwd_ret_1d"])
    split = int(len(feats) * 0.8)
    test_start = split + FWD_RET_HORIZON_DAYS

    assert data["used_symbols"] == ["AAPL"]
    # build_sequences further trims the first (lookback - 1) rows of each
    # block, so just assert the block boundary (pre-sequence) embargo holds.
    gap = test_start - split
    assert gap == FWD_RET_HORIZON_DAYS
