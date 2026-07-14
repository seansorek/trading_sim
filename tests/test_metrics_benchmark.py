import numpy as np
import pandas as pd
from simulation_pipeline import compute_metrics


def _series(vals, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="D")
    return pd.Series(vals, index=idx)


def test_benchmark_keys_absent_when_no_benchmark():
    equity = _series([100.0, 101.0, 102.0, 103.0])
    m = compute_metrics(equity, pd.DataFrame())
    assert "benchmark_return_pct" not in m
    assert "alpha_pct" not in m


def test_benchmark_alpha_positive_when_strategy_beats_hold():
    # Strategy doubles; benchmark price rises 10%.
    equity = _series([100.0, 120.0, 150.0, 200.0])
    bench = _series([50.0, 52.0, 54.0, 55.0])
    m = compute_metrics(equity, pd.DataFrame(), benchmark_close=bench)
    assert m["benchmark_return_pct"] == pytest_approx(10.0)
    assert m["alpha_pct"] == pytest_approx(m["total_return_pct"] - 10.0)
    assert "information_ratio" in m


def test_benchmark_alpha_negative_when_strategy_lags_hold():
    equity = _series([100.0, 100.5, 101.0, 101.0])   # +1%
    bench = _series([50.0, 55.0, 60.0, 65.0])          # +30%
    m = compute_metrics(equity, pd.DataFrame(), benchmark_close=bench)
    assert m["alpha_pct"] < 0


# local helper to avoid a pytest.approx import line at top
def pytest_approx(x, tol=1e-6):
    import pytest
    return pytest.approx(x, abs=tol)
