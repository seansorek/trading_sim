"""Tests for the predictor layer — no real models, no file I/O."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from predictors.base import BasePredictor, _preprocess


class TestBasePredictor:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BasePredictor()

    def test_preprocess_replaces_inf_with_zero(self):
        X = np.array([[1.0, np.inf, -np.inf]])
        out = _preprocess(X)
        assert np.isfinite(out).all()

    def test_preprocess_replaces_nan_with_zero(self):
        X = np.array([[np.nan, 1.0, 2.0]])
        out = _preprocess(X)
        assert not np.isnan(out).any()

    def test_preprocess_clips_outliers_per_column(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        X[0, 0] = 1_000.0  # extreme outlier in col 0
        out = _preprocess(X.copy())
        assert out[0, 0] < 1_000.0

    def test_preprocess_preserves_shape(self):
        X = np.ones((50, 10))
        assert _preprocess(X).shape == (50, 10)
