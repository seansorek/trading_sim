"""Guard: no two FEATURE_COLS may be near-perfect linear duplicates."""
import numpy as np
import pandas as pd

from daily_features import FEATURE_COLS, make_daily_features


def _synthetic_ohlcv(n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    openp = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": openp, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def test_no_feature_pair_exceeds_098_corr():
    feats = make_daily_features(_synthetic_ohlcv())[FEATURE_COLS]
    corr = feats.corr().abs()
    arr = corr.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    corr = pd.DataFrame(arr, index=corr.index, columns=corr.columns)
    worst = corr.stack().sort_values(ascending=False)
    top_pair, top_val = worst.index[0], float(worst.iloc[0])
    assert top_val <= 0.98, f"{top_pair} corr={top_val:.3f} > 0.98 — near-duplicate feature"
