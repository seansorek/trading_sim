"""test_walk_forward.py — Walk-forward harness tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from walk_forward import WalkForwardConfig, run_walk_forward_on_df
from daily_features import FWD_RET_HORIZON_DAYS


def _sine_price_df(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Synthetic price series with a weak predictable component."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # Sine trend gives Ridge something to learn
    close = 100 + np.sin(t * 0.05) * 5 + rng.normal(0, 0.5, n).cumsum()
    close = np.abs(close) + 10  # keep positive
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


def test_run_walk_forward_returns_dataframe_with_expected_columns():
    df = _sine_price_df(800)
    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    result = run_walk_forward_on_df(df, spy_df=None, config=config)
    assert isinstance(result, pd.DataFrame)
    for col in ("fold", "train_start", "train_end", "test_start", "test_end", "ic", "dir_acc", "n_test"):
        assert col in result.columns, f"Missing column: {col}"


def test_run_walk_forward_produces_at_least_one_fold():
    df = _sine_price_df(800)
    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    result = run_walk_forward_on_df(df, spy_df=None, config=config)
    assert len(result) >= 1


def test_run_walk_forward_raises_on_insufficient_bars():
    df = _sine_price_df(100)  # too short for any fold
    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    with pytest.raises(ValueError, match="bars"):
        run_walk_forward_on_df(df, spy_df=None, config=config)


def test_embargo_gap_enforced():
    """No training label should depend on price action inside the test window.
    The gap between train_end and test_start must be >= FWD_RET_HORIZON_DAYS."""
    df = _sine_price_df(800)
    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    result = run_walk_forward_on_df(df, spy_df=None, config=config)
    for _, row in result.iterrows():
        train_end = pd.Timestamp(row["train_end"])
        test_start = pd.Timestamp(row["test_start"])
        gap_days = (test_start - train_end).days
        assert gap_days >= FWD_RET_HORIZON_DAYS, (
            f"Embargo gap {gap_days} < FWD_RET_HORIZON_DAYS={FWD_RET_HORIZON_DAYS}"
        )


def test_ic_values_are_finite():
    df = _sine_price_df(800)
    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    result = run_walk_forward_on_df(df, spy_df=None, config=config)
    assert result["ic"].notna().all()
    assert result["dir_acc"].between(0.0, 1.0).all()
