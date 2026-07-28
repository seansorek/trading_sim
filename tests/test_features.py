"""test_features.py — Feature engineering and label encoding tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_features import (
    FEATURE_COLS,
    FWD_RET_HORIZON_DAYS,
    MIN_CS_NAMES,
    cross_sectional_normalize,
    discretize_labels,
    make_daily_features,
)


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
    # Output is shorter than input due to warmup (z-score, sma_50, etc.)
    assert len(feats) < len(df)
    # Warmup-drift lower bound: at most ~123 bars lost to z-scored vol_regime
    # plus the forward-return horizon. Adjust this if the bottleneck column changes.
    assert len(feats) >= len(df) - FWD_RET_HORIZON_DAYS - 130


def test_make_daily_features_no_inf():
    df = _synthetic_df(200)
    feats = make_daily_features(df)
    assert len(feats) > 0, "Empty features DataFrame — test would be vacuous"
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
    df = _synthetic_df(200)
    feats = make_daily_features(df)
    X = feats[FEATURE_COLS].values
    assert X.shape[0] > 0, "Empty feature array — test would be vacuous"
    # Column count must be exact
    assert X.shape[1] == len(FEATURE_COLS)


def test_no_nan_after_fillna():
    df = _synthetic_df(200)
    feats = make_daily_features(df)
    assert len(feats) > 0, "Empty features DataFrame — test would be vacuous"
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
    df = _synthetic_df(200)
    feats = make_daily_features(df)  # no spy_df
    assert len(feats) > 0, "Empty features DataFrame — test would be vacuous"
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


# --- cross_sectional_normalize ---------------------------------------------

def _long_panel(n_dates=3, n_syms=8, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rows = []
    for d in dates:
        for i in range(n_syms):
            row = {c: float(rng.normal()) for c in FEATURE_COLS}
            row["_date"] = d
            row["_symbol"] = f"S{i}"
            rows.append(row)
    return pd.DataFrame(rows)


def test_cross_sectional_normalize_ranks_within_date_to_pm_one():
    out = cross_sectional_normalize(_long_panel())
    assert out[FEATURE_COLS].min().min() > -1.0
    assert out[FEATURE_COLS].max().max() == 1.0
    for _, grp in out.groupby("_date"):
        # 8 names -> pct ranks 1/8..8/8 -> -0.75..+1.0, evenly spaced
        vals = np.sort(grp["ret_1d"].values)
        np.testing.assert_allclose(vals, np.linspace(-0.75, 1.0, 8))


def test_cross_sectional_normalize_is_invariant_to_a_market_wide_move():
    """The whole point: a shift common to every name on a date changes nothing."""
    panel = _long_panel(n_dates=2)
    shifted = panel.copy()
    day2 = shifted["_date"] == shifted["_date"].max()
    shifted.loc[day2, FEATURE_COLS] += 10.0   # everyone gapped up together

    base = cross_sectional_normalize(panel)
    moved = cross_sectional_normalize(shifted)
    pd.testing.assert_frame_equal(base[FEATURE_COLS], moved[FEATURE_COLS])


def test_cross_sectional_normalize_compares_names_not_a_symbols_own_history():
    """A name that is always the universe's best stays at +1 even as it decays."""
    panel = _long_panel(n_dates=3, n_syms=6)
    for i, (_, grp) in enumerate(panel.groupby("_date")):
        panel.loc[grp.index, "ret_1d"] = np.arange(len(grp)) * 0.01 - i * 5.0
    out = cross_sectional_normalize(panel)
    best = out.loc[out.groupby("_date")["ret_1d"].idxmax()]
    assert (best["ret_1d"] == 1.0).all()
    assert (best["_symbol"] == "S5").all()


def test_cross_sectional_normalize_nans_dates_with_too_few_names():
    panel = _long_panel(n_dates=2, n_syms=MIN_CS_NAMES)
    thin_date = panel["_date"].min()
    panel = panel.drop(panel.index[panel["_date"] == thin_date][:1])  # one short

    out = cross_sectional_normalize(panel)
    assert out.loc[out["_date"] == thin_date, FEATURE_COLS].isna().all().all()
    assert out.loc[out["_date"] != thin_date, FEATURE_COLS].notna().all().all()
