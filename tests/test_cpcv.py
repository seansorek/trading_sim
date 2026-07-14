import numpy as np
from cpcv import cscv_pbo


def test_pbo_near_half_for_pure_noise():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((40, 20))     # 40 obs, 20 configs, no real edge
    out = cscv_pbo(M, n_splits=10)
    assert 0.30 < out["pbo"] < 0.70


def test_pbo_near_zero_for_one_genuine_edge():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((40, 20)) * 0.1
    M[:, 3] += 1.0                         # config 3 is consistently best
    out = cscv_pbo(M, n_splits=10)
    assert out["pbo"] < 0.10


def test_pbo_insufficient_observations_returns_reason():
    M = np.random.default_rng(2).standard_normal((4, 5))
    out = cscv_pbo(M, n_splits=16)
    assert np.isnan(out["pbo"])
    assert "insufficient" in out["reason"].lower()


def test_pbo_needs_two_configs():
    M = np.random.default_rng(3).standard_normal((40, 1))
    out = cscv_pbo(M, n_splits=10)
    assert np.isnan(out["pbo"])
