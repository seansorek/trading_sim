"""test_features.py — Feature engineering and label encoding tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_features import FEATURE_COLS, discretize_labels, make_daily_features


def _synthetic_df(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close * rng.uniform(0.99, 1.005, n),
            "high": close * rng.uniform(1.001, 1.02, n),
            "low": close * rng.uniform(0.98, 0.999, n),
            "close": close,
            "volume": rng.integers(500_000, 10_000_000, n).astype(float),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# make_daily_features
# ---------------------------------------------------------------------------

def test_make_daily_features_shape(capsys):
    df = _synthetic_df(200)
    feats = make_daily_features(df)
    # Warmup rows (up to ~50 for sma_50) are dropped, so output is shorter than input
    assert len(feats) < len(df)
    assert len(feats) > 0


def test_make_daily_features_no_inf():
    df = _synthetic_df(100)
    feats = make_daily_features(df)
    assert not np.isinf(feats[FEATURE_COLS].values).any(), "Features contain inf values"


def test_make_daily_features_has_fwd_ret():
    df = _synthetic_df(200)
    feats = make_daily_features(df)
    assert "fwd_ret_1d" in feats.columns
    # Last 3 rows have no 3-day forward close, so fwd_ret_1d is NaN (not zero-filled)
    assert pd.isna(feats["fwd_ret_1d"].iloc[-1])
    assert pd.isna(feats["fwd_ret_1d"].iloc[-2])
    assert pd.isna(feats["fwd_ret_1d"].iloc[-3])


def test_feature_cols_indexing_gives_correct_shape():
    df = _synthetic_df(80)
    feats = make_daily_features(df)
    X = feats[FEATURE_COLS].values
    # Row count is < 80 due to warmup-row removal; column count must be exact
    assert X.shape[0] < 80
    assert X.shape[1] == len(FEATURE_COLS)


def test_no_nan_after_fillna():
    df = _synthetic_df(120)
    feats = make_daily_features(df)
    assert not feats[FEATURE_COLS].isna().any().any(), "Features still contain NaN"


# ---------------------------------------------------------------------------
# discretize_labels
# ---------------------------------------------------------------------------

def test_discretize_labels_correct_classes():
    returns = np.array([-0.01, -0.002, -0.001, 0.0, 0.001, 0.002, 0.01])
    labels = discretize_labels(returns)
    # -0.01  → 0 (SELL, < -0.002)
    # -0.002 → 0 (SELL, at boundary; strict <)
    # -0.001 → 1 (HOLD)
    #  0.0   → 1 (HOLD)
    #  0.001 → 1 (HOLD)
    #  0.002 → 1 (HOLD, at boundary; strict >)
    #  0.01  → 2 (BUY, > 0.002)
    expected = np.array([0, 1, 1, 1, 1, 1, 2])
    np.testing.assert_array_equal(labels, expected)


def test_discretize_labels_only_3_classes():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 500)
    labels = discretize_labels(returns)
    unique = np.unique(labels)
    assert set(unique).issubset({0, 1, 2})


def test_discretize_labels_custom_thresholds():
    returns = np.array([-0.05, 0.0, 0.05])
    labels = discretize_labels(returns, pos_thr=0.04, neg_thr=-0.04)
    assert labels[0] == 0   # SELL
    assert labels[1] == 1   # HOLD
    assert labels[2] == 2   # BUY


def test_discretize_labels_default_hold_majority():
    """For normal returns, HOLD should be the most common class."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.005, 1000)  # low volatility → mostly HOLD
    labels = discretize_labels(returns)
    counts = np.bincount(labels, minlength=3)
    assert counts[1] > counts[0] and counts[1] > counts[2]


# ---------------------------------------------------------------------------
# Stationarity checks (normalized cumsum features)
# ---------------------------------------------------------------------------

def test_spy_relative_features_default_to_zero():
    df = _synthetic_df(100)
    feats = make_daily_features(df)  # no spy_df
    assert (feats["ret_1d_vs_spy"] == 0.0).all()
    assert (feats["ret_5d_vs_spy"] == 0.0).all()


def test_spy_relative_features_nonzero_when_spy_provided():
    rng = np.random.default_rng(99)
    idx = pd.date_range("2023-01-02", periods=200, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 200))
    spy_close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.008, 200))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": np.ones(200) * 1e6}, index=idx)
    spy_df = pd.DataFrame({"open": spy_close, "high": spy_close * 1.01,
                           "low": spy_close * 0.99, "close": spy_close,
                           "volume": np.ones(200) * 1e6}, index=idx)
    feats = make_daily_features(df, spy_df=spy_df)
    # With different price paths, relative returns should be non-zero for most rows
    assert not (feats["ret_1d_vs_spy"] == 0.0).all()


def test_vpt_normalized_mean_near_zero():
    df = _synthetic_df(200)
    feats = make_daily_features(df)
    vals = feats["vpt_normalized"].dropna()
    assert abs(float(vals.mean())) < 1.0, f"vpt_normalized mean too far from 0: {vals.mean():.3f}"


def test_ad_normalized_bounded_std():
    df = _synthetic_df(200)
    feats = make_daily_features(df)
    vals = feats["ad_normalized"].dropna()
    std = float(vals.std())
    assert 0.1 < std < 5.0, f"ad_normalized std {std:.3f} looks wrong"


# ---------------------------------------------------------------------------
# Feature correlation guard
# ---------------------------------------------------------------------------

def test_no_pairwise_feature_correlation_above_098():
    import numpy as np
    import pandas as pd
    from daily_features import FEATURE_COLS, make_daily_features

    rng = np.random.default_rng(7)
    n = 500
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "open": close * rng.uniform(0.99, 1.01, n),
        "high": close * rng.uniform(1.00, 1.02, n),
        "low": close * rng.uniform(0.98, 1.00, n),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)
    spy = df.copy()  # non-constant SPY so ret_*_vs_spy has variance

    feats = make_daily_features(df, spy_df=spy)[FEATURE_COLS]
    # Drop near-constant columns (correlation undefined)
    keep = [c for c in FEATURE_COLS if feats[c].std() > 1e-9]
    corr = feats[keep].corr().abs()
    # Use a writable copy of the underlying array for the off-diagonal mask
    off_diag = np.where(~np.eye(len(keep), dtype=bool), corr.values, 0.0)
    worst = off_diag.max()
    offenders = [(keep[i], keep[j], off_diag[i, j])
                 for i in range(len(keep)) for j in range(i + 1, len(keep))
                 if off_diag[i, j] > 0.98]
    assert worst <= 0.98, f"Feature pairs with |corr|>0.98: {offenders}"
