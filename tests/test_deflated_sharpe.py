import numpy as np
from deflated_sharpe import deflated_sharpe, expected_max_sharpe


def test_expected_max_sharpe_grows_with_trials():
    lo = expected_max_sharpe(trial_sharpe_var=0.01, n_trials=5)
    hi = expected_max_sharpe(trial_sharpe_var=0.01, n_trials=100)
    assert hi > lo > 0


def test_expected_max_sharpe_zero_for_single_trial():
    assert expected_max_sharpe(trial_sharpe_var=0.01, n_trials=1) == 0.0


def test_dsr_high_when_sharpe_far_exceeds_null():
    rng = np.random.default_rng(0)
    # Strong positive per-period Sharpe (~0.3/step)
    returns = 0.003 + 0.01 * rng.standard_normal(750)
    out = deflated_sharpe(returns, n_trials=20, trial_sharpe_var=1e-4)
    assert out["sr"] > 0
    assert out["dsr"] > 0.9


def test_dsr_near_half_when_sharpe_matches_null():
    rng = np.random.default_rng(1)
    returns = 0.0005 + 0.01 * rng.standard_normal(500)
    sr = returns.mean() / returns.std(ddof=1)
    # Set the null equal to the observed SR by choosing trial variance so that
    # sr0 == sr, i.e. the observed result is exactly the expected max under null.
    from deflated_sharpe import expected_max_sharpe as ems
    # binary-search trial_sharpe_var so ems(var, 20) == sr
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ems(mid, 20) < sr:
            lo = mid
        else:
            hi = mid
    out = deflated_sharpe(returns, n_trials=20, trial_sharpe_var=(lo + hi) / 2)
    assert 0.35 < out["dsr"] < 0.65


def test_dsr_degenerate_short_series():
    out = deflated_sharpe(np.array([0.01, 0.02]), n_trials=10, trial_sharpe_var=0.01)
    assert out["dsr"] == 0.0
    assert out["p_value"] == 1.0
