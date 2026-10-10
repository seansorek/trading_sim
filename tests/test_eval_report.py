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


def test_compute_dsr_runs_predict_pipeline_once_per_symbol_not_per_config():
    """Issue #135: model load / make_daily_features / model.predict must run
    once per symbol and be reused across the whole (q, w) grid, not once per
    config -- only the final threshold step is genuinely per-config."""
    import ml_strategies

    df = _synth_prices()
    calls = []
    real_predict_returns = ml_strategies.DailyPredictorStrategy._predict_returns

    def spy(self, df_arg):
        calls.append(1)
        return real_predict_returns(self, df_arg)

    ml_strategies.DailyPredictorStrategy._predict_returns = spy
    try:
        compute_dsr_for_symbol("SYNTH", df, quantiles=[0.6, 0.7, 0.8], windows=[40, 60])
    finally:
        ml_strategies.DailyPredictorStrategy._predict_returns = real_predict_returns

    assert len(calls) == 1
