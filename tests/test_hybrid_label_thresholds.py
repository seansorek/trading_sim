"""
Regression test for issue #111: train_hybrid.py must derive BUY/SELL
thresholds from raw 20-day return volatility, not the z-scored vol_20d
feature (which can be negative and would make pos_thr/neg_thr overlap).
"""
from unittest.mock import patch

import numpy as np
import pandas as pd

from daily_features import discretize_labels, make_daily_features


def _synthetic_ohlcv(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, size=n))
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 10_000, size=n).astype(float),
        },
        index=idx,
    )
    return df


def test_zscored_vol_20d_can_go_negative():
    # Precondition motivating the bug: the vol_20d feature column is
    # rolling-z-scored in place, so it is not usable as a raw volatility.
    df = _synthetic_ohlcv()
    feats = make_daily_features(df)
    feats = feats.dropna(subset=["fwd_ret_1d"])
    assert (feats["vol_20d"].values < 0).any()


def test_raw_volatility_thresholds_are_nonnegative_and_labels_ordered():
    df = _synthetic_ohlcv()
    feats = make_daily_features(df)
    feats = feats.dropna(subset=["fwd_ret_1d"])

    vol_mult = 1.0
    raw_vol = df["close"].pct_change().rolling(20).std().reindex(feats.index).values
    pos_thr = raw_vol * np.sqrt(3) * vol_mult
    neg_thr = -pos_thr

    # Raw volatility (and thus pos_thr) is never negative, so the SELL/BUY
    # predicates (returns < neg_thr, returns > pos_thr) can never overlap.
    assert np.all(pos_thr >= 0)
    assert np.all(neg_thr <= 0)

    y = discretize_labels(feats["fwd_ret_1d"].values, pos_thr=pos_thr, neg_thr=neg_thr)
    returns = feats["fwd_ret_1d"].values

    assert np.all(returns[y == 2] > pos_thr[y == 2])
    assert np.all(returns[y == 0] < neg_thr[y == 0])


def test_prepare_data_labels_are_not_zscore_derived():
    """Exercise train_hybrid.prepare_data (the real production path) so a
    regression that reintroduces feats["vol_20d"] as the threshold source
    is caught here, not just in the standalone raw_vol calculation above.
    """
    import train_hybrid

    df = _synthetic_ohlcv(n=400)

    with patch.object(train_hybrid, "_load_symbol", return_value=df):
        data = train_hybrid.prepare_data(
            symbols=["AAPL"], days=1000, db=None, lookback=10, vol_mult=1.0,
        )

    assert data["used_symbols"] == ["AAPL"]
    # discretize_labels(neg_thr=pos_thr's negation) can only overlap into a
    # malformed (empty-middle-class) distribution if pos_thr went negative,
    # which only happens when the z-scored vol_20d feature leaks in. All
    # three classes should be represented across train+val+test.
    all_labels = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])
    assert set(np.unique(all_labels)) == {0, 1, 2}
