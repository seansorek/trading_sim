"""test_walk_forward.py — Walk-forward harness tests."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from walk_forward import WalkForwardConfig, run_walk_forward_on_df, sweep_params
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


def _mock_load_symbol(symbol, start, end, db):
    return _sine_price_df(800)


def test_sweep_params_returns_valid_pair():
    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    with patch("walk_forward._load_symbol", side_effect=_mock_load_symbol):
        q, w = sweep_params(
            ["AAPL"], days=900, db=None, config=config,
            quantiles=[0.65, 0.70], windows=[40, 60],
        )
    assert q in [0.65, 0.70]
    assert w in [40, 60]


def test_sweep_params_falls_back_when_all_ic_nonpositive():
    """All-constant predictions → IC = 0 for every param pair → fallback to defaults."""
    # Flat price series → Ridge predicts near-constant → IC ≈ 0
    n = 800
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    flat_df = pd.DataFrame({
        "open": np.full(n, 100.0), "high": np.full(n, 100.1),
        "low": np.full(n, 99.9), "close": np.full(n, 100.0),
        "volume": np.full(n, 1_000_000.0),
    }, index=idx)

    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    with patch("walk_forward._load_symbol", return_value=flat_df):
        q, w = sweep_params(
            ["AAPL"], days=900, db=None, config=config,
            quantiles=[0.65, 0.70], windows=[40, 60],
        )
    assert q == 0.7
    assert w == 60


def test_sweep_params_no_valid_symbols_returns_defaults():
    with patch("walk_forward._load_symbol", return_value=None):
        q, w = sweep_params(["AAPL"], days=900, db=None)
    assert q == 0.7
    assert w == 60


def test_build_fold_data_matches_matrix_shape():
    from walk_forward import build_fold_data, fold_config_ic_matrix, WalkForwardConfig
    import numpy as np
    # Synthetic fold_data: 10 folds, each a length-120 prediction window.
    rng = np.random.default_rng(0)
    fold_data = []
    for _ in range(10):
        pred = rng.standard_normal(120)
        y_te = rng.standard_normal(60)      # test slice length
        test_offset = 60
        fold_data.append((pred, y_te, test_offset))
    quantiles = [0.6, 0.7]
    windows = [40, 60]
    matrix, configs = fold_config_ic_matrix(fold_data, quantiles, windows)
    assert matrix.shape == (10, 4)
    assert configs == [(0.6, 40), (0.6, 60), (0.7, 40), (0.7, 60)]
    assert np.isfinite(matrix).all()
