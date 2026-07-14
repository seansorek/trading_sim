"""
deflated_sharpe.py — Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

Adjusts an observed Sharpe for (a) the number of trials that produced it
(selection bias), (b) non-normal return skew/kurtosis, and (c) sample length.
All Sharpes here are PER-PERIOD (not annualized) and must be used consistently.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EULER_MASCHERONI = 0.5772156649015329


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def expected_max_sharpe(trial_sharpe_var: float, n_trials: int) -> float:
    """Expected maximum Sharpe under the null of zero true skill across
    n_trials independent strategy configurations (the SR0 deflation target)."""
    if n_trials < 2 or trial_sharpe_var <= 0:
        return 0.0
    g = _EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(trial_sharpe_var) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe(returns: np.ndarray, n_trials: int,
                    trial_sharpe_var: float) -> dict:
    r = np.asarray(returns, dtype=float)
    T = len(r)
    degenerate = {"dsr": 0.0, "sr": 0.0, "sr0": 0.0, "p_value": 1.0}
    if T < 4:
        return degenerate
    sr = _sharpe(r)
    sr0 = expected_max_sharpe(trial_sharpe_var, n_trials)
    sd = r.std(ddof=0)
    if sd == 0:
        return {"dsr": 0.0, "sr": sr, "sr0": sr0, "p_value": 1.0}
    rm = r - r.mean()
    skew = float(np.mean(rm ** 3) / sd ** 3)
    kurt = float(np.mean(rm ** 4) / sd ** 4)   # non-excess kurtosis (gamma_4)
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2)
    if denom <= 0:
        return {"dsr": 0.0, "sr": sr, "sr0": sr0, "p_value": 1.0}
    stat = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(denom)
    dsr = float(norm.cdf(stat))
    return {"dsr": dsr, "sr": sr, "sr0": sr0, "p_value": 1.0 - dsr}


if __name__ == "__main__":
    # ponytail: runnable self-check — DSR high when SR >> SR0, ~0.5 when SR ~ SR0.
    rng = np.random.default_rng(0)
    strong = 0.003 + 0.01 * rng.standard_normal(750)
    assert deflated_sharpe(strong, 20, 1e-4)["dsr"] > 0.9
    assert deflated_sharpe(np.array([0.01, 0.02]), 10, 0.01)["dsr"] == 0.0
    print("deflated_sharpe self-check OK")
