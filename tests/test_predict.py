"""test_predict.py — Tests for predict_next_day_lite.py (no network, no real models)."""
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_features import FEATURE_COLS
from dqn_signal import gate_dqn_signal
from predict_next_day_lite import _load_pkl, predict_symbol, send_discord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(500_000, 10_000_000, n).astype(float),
        },
        index=idx,
    )


def _make_features_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = {col: rng.standard_normal(n) for col in FEATURE_COLS}
    data["fwd_ret_1d"] = rng.standard_normal(n)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame(data, index=idx)


class _PicklableStub:
    """Minimal picklable stand-in for model/scaler in pickle-roundtrip tests."""
    pass


def _make_artifact(feature_contract=None, include_keys=None) -> dict:
    """Build a minimal valid model artifact dict (picklable)."""
    if feature_contract is None:
        feature_contract = FEATURE_COLS
    artifact = {
        "model": _PicklableStub(),
        "scaler": _PicklableStub(),
        "feature_contract": list(feature_contract),
        "confidence_threshold": 0.55,
    }
    if include_keys is not None:
        artifact = {k: v for k, v in artifact.items() if k in include_keys}
    return artifact


# ---------------------------------------------------------------------------
# _load_pkl
# ---------------------------------------------------------------------------

class TestLoadPkl:
    def test_valid_artifact_loads(self):
        artifact = _make_artifact()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(artifact, f)
            path = f.name
        loaded = _load_pkl(path, "daily_logistic")
        assert "model" in loaded
        assert "scaler" in loaded
        assert loaded["feature_contract"] == list(FEATURE_COLS)

    def test_missing_file_raises(self):
        with pytest.raises(RuntimeError, match="not found"):
            _load_pkl("/nonexistent/model.pkl", "daily_logistic")

    def test_missing_model_key_raises(self):
        artifact = _make_artifact(include_keys={"scaler", "feature_contract"})
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(artifact, f)
            path = f.name
        with pytest.raises(RuntimeError, match="missing key 'model'"):
            _load_pkl(path, "daily_logistic")

    def test_wrong_feature_contract_raises(self):
        artifact = _make_artifact(feature_contract=["wrong_feat_a", "wrong_feat_b"])
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(artifact, f)
            path = f.name
        with pytest.raises(RuntimeError, match="Feature contract mismatch"):
            _load_pkl(path, "daily_logistic")


# ---------------------------------------------------------------------------
# predict_symbol
# ---------------------------------------------------------------------------

class TestPredictSymbol:
    def _build_models(self, proba: list) -> dict:
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([proba])
        return {
            "daily_logistic": {
                "model": clf,
                "scaler": scaler,
                "feature_contract": list(FEATURE_COLS),
                "confidence_threshold": 0.55,
            }
        }

    def test_high_confidence_buy(self):
        models = self._build_models([0.1, 0.2, 0.7])  # BUY at 70%
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        assert "predictions" in result
        assert "error" not in result
        p = result["predictions"]["daily_logistic"]
        assert p["signal"] == "BUY"
        assert abs(p["confidence"] - 0.7) < 1e-6

    def test_low_confidence_collapses_to_hold(self):
        # All probabilities below threshold — argmax is BUY (idx 2) but confidence < 0.55
        models = self._build_models([0.25, 0.30, 0.45])
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        p = result["predictions"]["daily_logistic"]
        assert p["signal"] == "HOLD"

    def test_data_fetch_failure_returns_error(self):
        models = self._build_models([0.1, 0.2, 0.7])

        with patch("predict_next_day_lite.load_yfinance", side_effect=RuntimeError("timeout")):
            result = predict_symbol("AAPL", models)

        assert "error" in result
        assert "timeout" in result["error"]

    def test_insufficient_data_returns_error(self):
        models = self._build_models([0.1, 0.2, 0.7])
        tiny_df = _make_ohlcv(10)  # only 10 bars, < 50 minimum

        with patch("predict_next_day_lite.load_yfinance", return_value=tiny_df):
            result = predict_symbol("AAPL", models)

        assert "error" in result

    def test_history_days_param_is_used(self):
        """Verify history_days flows into the date range (not silently ignored)."""
        models = self._build_models([0.1, 0.2, 0.7])
        feats_df = _make_features_df(100)
        captured = {}

        def fake_load(symbol, start, end, interval):
            captured["start"] = start
            return _make_ohlcv(100)

        with patch("predict_next_day_lite.load_yfinance", side_effect=fake_load), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            predict_symbol("AAPL", models, history_days=500)

        from datetime import datetime, timedelta
        expected_start = datetime.now() - timedelta(days=500)
        actual_start = datetime.strptime(captured["start"], "%Y-%m-%d")
        assert abs((actual_start - expected_start).days) <= 1


# ---------------------------------------------------------------------------
# send_discord
# ---------------------------------------------------------------------------

class TestSendDiscord:
    def _minimal_predictions(self, signal="BUY", confidence=0.8):
        return [
            {
                "symbol": "AAPL",
                "price": 150.0,
                "predictions": {
                    "daily_logistic": {"signal": signal, "confidence": confidence}
                },
            }
        ]

    def test_sends_payload_and_returns_true(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 204

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = send_discord(self._minimal_predictions(), "https://discord.example/webhook")

        assert result is True
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        assert "embeds" in payload
        assert len(payload["embeds"]) >= 1
        assert "BUY" in payload["embeds"][0]["title"]

    def test_connection_error_returns_false(self):
        with patch("requests.post", side_effect=ConnectionError("refused")):
            result = send_discord(self._minimal_predictions(), "https://discord.example/webhook")

        assert result is False

    def test_empty_webhook_returns_false(self):
        result = send_discord(self._minimal_predictions(), "")
        assert result is False

    def test_hold_signals_included_in_embeds(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        preds = self._minimal_predictions(signal="HOLD", confidence=0.6)

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = send_discord(preds, "https://discord.example/webhook")

        assert result is True
        payload = mock_post.call_args.kwargs["json"]
        titles = [e["title"] for e in payload["embeds"]]
        assert any("HOLD" in t for t in titles)

    def test_error_predictions_are_skipped(self):
        preds = [{"symbol": "AAPL", "error": "fetch failed"}]
        mock_resp = MagicMock()
        mock_resp.status_code = 204

        with patch("requests.post", return_value=mock_resp):
            result = send_discord(preds, "https://discord.example/webhook")

        # No valid predictions → no embeds → returns False
        assert result is False


# ---------------------------------------------------------------------------
# gate_dqn_signal  (shared helper)
# ---------------------------------------------------------------------------

class TestGateDqnSignal:
    """Tests for the shared DQN signal gating function."""

    def test_low_advantage_returns_hold_even_if_argmax_is_buy(self):
        """When Q-values are close together (below thresholds), signal must be HOLD."""
        # q_vals: [Hold=1.0, Long=1.3, Short=0.9]
        # q_max - q_min = 0.4, below confidence_threshold=2.0
        q_vals = np.array([1.0, 1.3, 0.9])
        signal, confidence = gate_dqn_signal(
            q_vals, confidence_threshold=2.0, q_advantage_threshold=1.0
        )
        assert signal == "HOLD"
        assert abs(confidence - 0.4) < 1e-6

    def test_low_advantage_returns_hold_even_if_argmax_is_sell(self):
        """When Q-values are close together, SELL argmax still produces HOLD."""
        # q_vals: [Hold=1.0, Long=0.8, Short=1.2]
        q_vals = np.array([1.0, 0.8, 1.2])
        signal, confidence = gate_dqn_signal(
            q_vals, confidence_threshold=2.0, q_advantage_threshold=1.0
        )
        assert signal == "HOLD"

    def test_high_advantage_buy_signal(self):
        """When Q-Long clearly dominates, signal should be BUY."""
        # q_vals: [Hold=0.5, Long=5.0, Short=0.2]
        # confidence = 5.0 - 0.2 = 4.8 >= 2.0
        # q_long - q_hold = 4.5 > 1.0
        q_vals = np.array([0.5, 5.0, 0.2])
        signal, confidence = gate_dqn_signal(
            q_vals, confidence_threshold=2.0, q_advantage_threshold=1.0
        )
        assert signal == "BUY"
        assert abs(confidence - 4.8) < 1e-6

    def test_high_advantage_sell_signal(self):
        """When Q-Short clearly dominates, signal should be SELL."""
        # q_vals: [Hold=0.5, Long=0.2, Short=5.0]
        q_vals = np.array([0.5, 0.2, 5.0])
        signal, confidence = gate_dqn_signal(
            q_vals, confidence_threshold=2.0, q_advantage_threshold=1.0
        )
        assert signal == "SELL"

    def test_confidence_met_but_advantage_over_hold_too_small(self):
        """Overall spread is large but best action barely beats Hold => HOLD."""
        # q_vals: [Hold=3.0, Long=3.5, Short=0.0]
        # confidence = 3.5 - 0.0 = 3.5 >= 2.0
        # q_long - q_hold = 0.5 < 1.0  (below q_advantage_threshold)
        q_vals = np.array([3.0, 3.5, 0.0])
        signal, confidence = gate_dqn_signal(
            q_vals, confidence_threshold=2.0, q_advantage_threshold=1.0
        )
        assert signal == "HOLD"
        assert abs(confidence - 3.5) < 1e-6

    def test_hold_is_best_action(self):
        """When Hold has the highest Q-value, signal is HOLD regardless."""
        q_vals = np.array([10.0, 3.0, 2.0])
        signal, confidence = gate_dqn_signal(
            q_vals, confidence_threshold=2.0, q_advantage_threshold=1.0
        )
        assert signal == "HOLD"


# ---------------------------------------------------------------------------
# predict_symbol with DQN model
# ---------------------------------------------------------------------------

class TestPredictSymbolDQN:
    """Test that predict_symbol applies DQN gating correctly end-to-end."""

    def _make_dqn_models(self, q_values):
        """Build a models dict with a mock DQN agent returning given Q-values."""
        import torch

        q_tensor = torch.tensor([q_values], dtype=torch.float32)
        q_network = MagicMock()
        q_network.return_value = q_tensor

        agent = MagicMock()
        agent.q = q_network
        return {"daily_dqn": agent}

    def test_dqn_low_advantage_produces_hold(self):
        """DQN prediction with low Q-advantage should produce HOLD."""
        # Q-values close together: Hold=1.0, Long=1.3, Short=0.9
        models = self._make_dqn_models([1.0, 1.3, 0.9])
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        p = result["predictions"]["daily_dqn"]
        assert p["signal"] == "HOLD"

    def test_dqn_high_advantage_produces_buy(self):
        """DQN prediction with high Q-advantage for Long should produce BUY."""
        # Q-values with clear Long advantage: Hold=0.5, Long=10.0, Short=0.2
        models = self._make_dqn_models([0.5, 10.0, 0.2])
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        p = result["predictions"]["daily_dqn"]
        assert p["signal"] == "BUY"

    def test_dqn_high_advantage_produces_sell(self):
        """DQN prediction with high Q-advantage for Short should produce SELL."""
        # Q-values with clear Short advantage: Hold=0.5, Long=0.2, Short=10.0
        models = self._make_dqn_models([0.5, 0.2, 10.0])
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        p = result["predictions"]["daily_dqn"]
        assert p["signal"] == "SELL"

    def test_dqn_confidence_is_normalized_to_unit_interval(self):
        """DQN confidence in prediction output must be in [0, 1], not raw Q-spread."""
        # Large Q-spread: raw confidence = 10.0 - 0.2 = 9.8 (unbounded)
        # Normalized: clip((9.8 + 1.0) / 2.0, 0, 1) = 1.0
        models = self._make_dqn_models([0.5, 10.0, 0.2])
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        p = result["predictions"]["daily_dqn"]
        assert 0.0 <= p["confidence"] <= 1.0, (
            f"DQN confidence should be in [0, 1], got {p['confidence']}"
        )
        assert abs(p["confidence"] - 1.0) < 1e-6

    def test_dqn_confidence_small_spread_normalized(self):
        """Small Q-spread normalizes to a value strictly between 0 and 1."""
        # Q-values: Hold=1.0, Long=1.3, Short=0.9
        # raw confidence = 0.4, normalized = clip((0.4 + 1.0) / 2.0, 0, 1) = 0.7
        models = self._make_dqn_models([1.0, 1.3, 0.9])
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        p = result["predictions"]["daily_dqn"]
        assert 0.0 <= p["confidence"] <= 1.0
        assert abs(p["confidence"] - 0.7) < 1e-6
