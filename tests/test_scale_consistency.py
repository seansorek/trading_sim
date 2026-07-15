import numpy as np
from sklearn.preprocessing import RobustScaler
from predictors.base import _scale, CLIP


def test_scale_is_frozen_and_clipped():
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, (200, 5))
    scaler = RobustScaler().fit(X_train)
    X_new = rng.normal(0, 1, (10, 5))
    out = _scale(scaler, X_new)
    # frozen transform: same input -> same output regardless of batch
    assert np.allclose(out, _scale(scaler, X_new))
    assert out.max() <= CLIP + 1e-9 and out.min() >= -CLIP - 1e-9
