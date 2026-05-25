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
