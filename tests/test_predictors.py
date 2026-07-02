"""Tests for the predictor layer — no real models, no file I/O."""
import sys
from pathlib import Path
import pickle
import tempfile
from unittest import mock as unittest_mock
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_features import FEATURE_COLS
from predictors.base import BasePredictor, _preprocess
from predictors.logistic import LogisticPredictor
from predictors.xgboost_pred import XGBPredictor
from predictors.ridge import RidgePredictor


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


def _make_fake_pickle(path: str, feature_contract=None):
    """Write a minimal valid pickle for LogisticPredictor.load().

    Patches _load_validated_pickle to inject mocks after loading.
    """
    if feature_contract is None:
        feature_contract = list(FEATURE_COLS)

    # Pickle only serializable data (str/list/np.array), mocks attached in tests
    artifact = {
        "model": None,  # Placeholder, will be mocked in tests
        "scaler": None,  # Placeholder, will be mocked in tests
        "feature_contract": feature_contract,
        "confidence_threshold": 0.55,
    }
    with open(path, "wb") as f:
        pickle.dump(artifact, f)


class TestLogisticPredictor:
    def _create_mock_predictor(self):
        """Helper to create a LogisticPredictor with mocks."""
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x  # identity
        clf = MagicMock()
        clf.classes_ = np.array([0, 1, 2])
        clf.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        return LogisticPredictor(model=clf, scaler=scaler, confidence_threshold=0.55)

    def test_load_returns_predictor_instance(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            _make_fake_pickle(f.name)
            # Patch _load_validated_pickle to inject mocks
            with unittest_mock.patch("predictors.base._load_validated_pickle") as mock_load:
                scaler = MagicMock()
                scaler.transform.side_effect = lambda x: x
                clf = MagicMock()
                clf.classes_ = np.array([0, 1, 2])
                clf.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
                mock_load.return_value = {
                    "model": clf,
                    "scaler": scaler,
                    "feature_contract": list(FEATURE_COLS),
                    "confidence_threshold": 0.55,
                }
                pred = LogisticPredictor.load(f.name)
            assert isinstance(pred, LogisticPredictor)

    def test_load_missing_file_raises(self):
        with pytest.raises(RuntimeError, match="not found"):
            LogisticPredictor.load("/nonexistent/model.pkl")

    def test_load_wrong_feature_contract_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            _make_fake_pickle(f.name, feature_contract=["bad_col"])
            with pytest.raises(RuntimeError, match="Feature contract mismatch"):
                LogisticPredictor.load(f.name)

    def test_predict_returns_tuple(self):
        pred = self._create_mock_predictor()
        n = 5
        X = np.zeros((n, len(FEATURE_COLS)), dtype=np.float32)
        # mock returns same shape regardless of input length
        pred.model.predict_proba.return_value = np.tile([0.1, 0.2, 0.7], (n, 1))
        scores, proba = pred.predict(X)
        assert scores.shape == (n,)
        assert proba.shape[0] == n

    def test_predict_scores_in_minus1_0_1(self):
        pred = self._create_mock_predictor()
        n = 3
        pred.model.classes_ = np.array([0, 1, 2])
        pred.model.predict_proba.return_value = np.array([
            [0.7, 0.2, 0.1],  # SELL → -1
            [0.1, 0.7, 0.2],  # HOLD → 0
            [0.1, 0.2, 0.7],  # BUY  → +1
        ])
        X = np.zeros((n, len(FEATURE_COLS)), dtype=np.float32)
        scores, _ = pred.predict(X)
        np.testing.assert_array_equal(scores, [-1.0, 0.0, 1.0])

    def test_predict_maps_through_classes_not_column_index(self):
        """classes_=[0, 2] (no HOLD class): col 1 → class 2 → BUY (+1)."""
        pred = self._create_mock_predictor()
        pred.model.classes_ = np.array([0, 2])  # missing class 1 (HOLD)
        pred.model.predict_proba.return_value = np.array([[0.3, 0.7]])
        X = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
        scores, _ = pred.predict(X)
        assert scores[0] == 1.0  # argmax=1 → classes_[1]=2 → 2-1=1 (BUY)

    def test_confidence_threshold_loaded_from_pickle(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            _make_fake_pickle(f.name)
            with unittest_mock.patch("predictors.base._load_validated_pickle") as mock_load:
                scaler = MagicMock()
                scaler.transform.side_effect = lambda x: x
                clf = MagicMock()
                clf.classes_ = np.array([0, 1, 2])
                mock_load.return_value = {
                    "model": clf,
                    "scaler": scaler,
                    "feature_contract": list(FEATURE_COLS),
                    "confidence_threshold": 0.55,
                }
                pred = LogisticPredictor.load(f.name)
            assert pred.confidence_threshold == 0.55


class TestXGBPredictor:
    def _create_mock_predictor(self):
        """Helper to create an XGBPredictor with mocks."""
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x  # identity
        clf = MagicMock()
        clf.classes_ = np.array([0, 1, 2])
        clf.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        return XGBPredictor(model=clf, scaler=scaler, confidence_threshold=0.55)

    def test_load_returns_predictor_instance(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            _make_fake_pickle(f.name)
            with unittest_mock.patch("predictors.base._load_validated_pickle") as mock_load:
                scaler = MagicMock()
                scaler.transform.side_effect = lambda x: x
                clf = MagicMock()
                clf.classes_ = np.array([0, 1, 2])
                clf.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
                mock_load.return_value = {
                    "model": clf,
                    "scaler": scaler,
                    "feature_contract": list(FEATURE_COLS),
                    "confidence_threshold": 0.55,
                }
                pred = XGBPredictor.load(f.name)
            assert isinstance(pred, XGBPredictor)

    def test_predict_scores_shape_matches_input(self):
        pred = self._create_mock_predictor()
        n = 7
        pred.model.predict_proba.return_value = np.tile([0.1, 0.2, 0.7], (n, 1))
        X = np.zeros((n, len(FEATURE_COLS)), dtype=np.float32)
        scores, proba = pred.predict(X)
        assert scores.shape == (n,)
        assert proba.shape == (n, 3)


class TestRidgePredictor:
    def _create_mock_predictor(self):
        """Helper to create a RidgePredictor with mocks."""
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x  # identity
        model = MagicMock()
        model.predict.return_value = np.array([0.002])
        return RidgePredictor(model=model, scaler=scaler, train_ic=0.06, train_r2=0.012)

    def test_load_returns_predictor_instance(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            _make_fake_pickle(f.name)
            with unittest_mock.patch("predictors.base._load_validated_pickle") as mock_load:
                scaler = MagicMock()
                scaler.transform.side_effect = lambda x: x
                model = MagicMock()
                model.predict.return_value = np.array([0.002, -0.001, 0.003])
                mock_load.return_value = {
                    "model": model,
                    "scaler": scaler,
                    "feature_contract": list(FEATURE_COLS),
                    "train_ic": 0.06,
                    "train_r2": 0.012,
                }
                pred = RidgePredictor.load(f.name)
            assert isinstance(pred, RidgePredictor)

    def test_predict_returns_continuous_scores_and_none_proba(self):
        pred = self._create_mock_predictor()
        n = 3
        pred.model.predict.return_value = np.array([0.002, -0.001, 0.003])
        X = np.zeros((n, len(FEATURE_COLS)), dtype=np.float32)
        scores, proba = pred.predict(X)
        assert proba is None
        assert scores.shape == (n,)
        assert scores.dtype == float

    def test_predict_scores_are_signed_floats(self):
        pred = self._create_mock_predictor()
        pred.model.predict.return_value = np.array([0.005, -0.003])
        X = np.zeros((2, len(FEATURE_COLS)), dtype=np.float32)
        scores, _ = pred.predict(X)
        assert scores[0] > 0
        assert scores[1] < 0

    def test_metadata_attributes_available(self):
        pred = self._create_mock_predictor()
        assert pred.train_ic == pytest.approx(0.06)
        assert pred.train_r2 == pytest.approx(0.012)


from unittest.mock import patch
import torch
from predictors.dqn import DQNPredictor


def _make_mock_agent(q_values=(0.5, 5.0, 0.2)):
    """Build a mock DQNAgent returning constant Q-values."""
    q_tensor = torch.tensor([list(q_values)], dtype=torch.float32)
    q_network = MagicMock()
    q_network.return_value = q_tensor
    agent = MagicMock()
    agent.q = q_network
    return agent


class TestDQNPredictor:
    def test_predict_returns_tuple(self):
        agent = _make_mock_agent()
        pred = DQNPredictor(agent, window=5)
        X = np.zeros((20, len(FEATURE_COLS)), dtype=np.float32)
        scores, proba = pred.predict(X)
        assert scores.shape == (20,)
        assert proba is not None
        assert proba.shape == (20, 3)

    def test_warmup_bars_are_zero(self):
        agent = _make_mock_agent()
        pred = DQNPredictor(agent, window=5)
        X = np.ones((20, len(FEATURE_COLS)), dtype=np.float32)
        scores, proba = pred.predict(X)
        # First 5 bars are warmup — Q-matrix rows should be all-zero
        np.testing.assert_array_equal(proba[:5], 0.0)

    def test_active_bars_have_nonzero_q_values(self):
        agent = _make_mock_agent(q_values=(0.5, 5.0, 0.2))
        pred = DQNPredictor(agent, window=5)
        X = np.ones((20, len(FEATURE_COLS)), dtype=np.float32)
        _, proba = pred.predict(X)
        # Bars after warmup (index >= window) should have non-zero Q-values
        assert not np.all(proba[5:] == 0.0)

    def test_scores_equal_q_long_minus_q_short(self):
        # Q = [Hold=0.5, Long=5.0, Short=0.2]
        agent = _make_mock_agent(q_values=(0.5, 5.0, 0.2))
        pred = DQNPredictor(agent, window=5)
        X = np.ones((20, len(FEATURE_COLS)), dtype=np.float32)
        scores, proba = pred.predict(X)
        # For active bars, score = Q(Long) - Q(Short) = 5.0 - 0.2 = 4.8
        active_scores = scores[5:]
        active_proba = proba[5:]
        np.testing.assert_allclose(
            active_scores, active_proba[:, 1] - active_proba[:, 2], atol=1e-5
        )

    def test_nan_input_does_not_crash(self):
        agent = _make_mock_agent()
        pred = DQNPredictor(agent, window=5)
        X = np.full((20, len(FEATURE_COLS)), np.nan, dtype=np.float32)
        scores, proba = pred.predict(X)  # should not raise
        assert scores.shape == (20,)

    def test_load_delegates_to_dqn_agent(self):
        with patch("predictors.dqn.DQNAgent") as MockDQN:
            MockDQN.load.return_value = _make_mock_agent()
            pred = DQNPredictor.load("models/fake_dqn.pt", window=10)
        MockDQN.load.assert_called_once_with("models/fake_dqn.pt")
        assert pred.window == 10
