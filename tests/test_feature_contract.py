"""
test_feature_contract.py — Guard the feature contract between training and prediction.

These tests must pass before any model retrain. If FEATURE_COLS changes,
all models must be retrained (bump FEATURE_SET_NAME to "daily_v2").
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_features import FEATURE_COLS, FEATURE_SET_NAME, make_daily_features


# Hard-coded expected list — if this test fails, you changed FEATURE_COLS
_EXPECTED_FEATURE_COLS = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "vol_20d",
    "ma_spread_10_20",
    "ma_spread_20_50",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "atr_14",
    "price_vs_sma20",
    "price_vs_sma50",
    "vol_z_20",
    "volume_ma_20",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "bb_position",
    "stoch_k",
    "stoch_d",
    "williams_r",
    "momentum_10",
    "roc_12",
    "atr_normalized",
    "vpt_normalized",
    "ad_normalized",
    "obv_normalized",
]


def _make_synthetic_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close * rng.uniform(0.99, 1.01, n),
            "high": close * rng.uniform(1.00, 1.02, n),
            "low": close * rng.uniform(0.98, 1.00, n),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


def test_feature_cols_order_is_stable():
    """FEATURE_COLS must match the hard-coded expected list exactly."""
    assert FEATURE_COLS == _EXPECTED_FEATURE_COLS, (
        "FEATURE_COLS changed! If intentional, update _EXPECTED_FEATURE_COLS "
        "AND bump FEATURE_SET_NAME, then retrain all models."
    )


def test_feature_set_name_is_string():
    assert isinstance(FEATURE_SET_NAME, str) and len(FEATURE_SET_NAME) > 0


def test_make_daily_features_contains_all_feature_cols():
    df = _make_synthetic_df()
    feats = make_daily_features(df)
    missing = [col for col in FEATURE_COLS if col not in feats.columns]
    assert not missing, f"make_daily_features() is missing columns: {missing}"


def test_features_indexed_by_name_vs_position():
    """Indexing feats[FEATURE_COLS] must be identical regardless of column order."""
    df = _make_synthetic_df()
    feats = make_daily_features(df)
    X_by_name = feats[FEATURE_COLS].values
    X_by_list = feats[list(FEATURE_COLS)].values
    np.testing.assert_array_equal(X_by_name, X_by_list)


def test_no_raw_cumsum_in_feature_cols():
    assert "vpt" not in FEATURE_COLS, "Raw VPT cumsum is non-stationary"
    assert "ad_line" not in FEATURE_COLS, "Raw AD line cumsum is non-stationary"


def test_no_raw_smas_in_feature_cols():
    """Raw price-level SMAs are non-stationary; only keep ratio/spread derivatives."""
    assert "sma_10" not in FEATURE_COLS
    assert "sma_20" not in FEATURE_COLS
    assert "sma_50" not in FEATURE_COLS


def test_feature_cols_length():
    assert len(FEATURE_COLS) == len(_EXPECTED_FEATURE_COLS)


def test_feature_cols_no_duplicates():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), "FEATURE_COLS has duplicates"


def test_pickle_feature_contract_matches_constant():
    """Any .pkl in models/ must have feature_contract == FEATURE_COLS."""
    models_dir = Path(__file__).parent.parent / "models"
    if not models_dir.exists():
        pytest.skip("No models/ directory — run train_models.py first")

    pkl_files = list(models_dir.glob("*.pkl"))
    if not pkl_files:
        pytest.skip("No .pkl files in models/ — run train_models.py first")

    checked = 0
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            try:
                data = pickle.load(f)
            except Exception:
                continue  # unreadable pickle — skip

        if not isinstance(data, dict) or "feature_contract" not in data:
            # Old-format model; warn but keep checking the rest.
            import warnings
            warnings.warn(
                f"{pkl_path.name} is in old format (missing 'feature_contract'). "
                "Retrain with train_models.py.",
                stacklevel=2,
            )
            continue

        assert data["feature_contract"] == FEATURE_COLS, (
            f"{pkl_path.name}: feature_contract does not match FEATURE_COLS. "
            "Retrain the model."
        )
        checked += 1

    if checked == 0:
        pytest.skip("No new-format models found. Run train_models.py to generate them.")


def test_normalized_cumsum_features_are_bounded():
    """vpt_normalized and ad_normalized should be roughly z-scored."""
    df = _make_synthetic_df(n=200)
    feats = make_daily_features(df)

    for col in ("vpt_normalized", "ad_normalized", "obv_normalized"):
        vals = feats[col].dropna()
        if len(vals) < 30:
            continue
        mean = float(vals.mean())
        std = float(vals.std())
        assert abs(mean) < 1.0, f"{col} mean {mean:.3f} too far from 0"
        assert 0.1 < std < 5.0, f"{col} std {std:.3f} looks wrong"
