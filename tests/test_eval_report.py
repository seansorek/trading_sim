import numpy as np
import pandas as pd
from eval_report import compute_dsr_for_symbol


def _synth_prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + 0.0005 + 0.01 * rng.standard_normal(n))
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    high = close * (1 + 0.005)
    low = close * (1 - 0.005)
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def test_compute_dsr_returns_expected_keys():
    df = _synth_prices()
    out = compute_dsr_for_symbol("SYNTH", df, quantiles=[0.6, 0.7], windows=[40, 60])
    for k in ("dsr", "sr", "sr0", "p_value", "selected", "n_trials"):
        assert k in out
    assert out["n_trials"] == 4
    assert 0.0 <= out["dsr"] <= 1.0
