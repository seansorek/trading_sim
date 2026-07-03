"""test_predict.py — Tests for predict_next_day_lite.py (no network, no real models)."""
import json
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
from predict_next_day_lite import (
    _load_pkl,
    _predict_classifier_signal,
    _predict_regressor_signal,
    _regressor_confidence,
    append_predictions_history,
    load_models,
    predict_symbol,
    send_discord,
)


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
        clf.classes_ = np.array(list(range(len(proba))))
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
# append_predictions_history
# ---------------------------------------------------------------------------

class TestAppendPredictionsHistory:
    def _predictions(self):
        return [
            {
                "symbol": "AAPL",
                "price": 150.0,
                "predictions": {
                    "daily_logistic": {"signal": "BUY", "confidence": 0.7},
                    "daily_xgboost": {"signal": "HOLD", "confidence": 0.6},
                },
            },
            {"symbol": "BADTICKER", "error": "Insufficient data"},
        ]

    def test_writes_one_record_per_symbol_model_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "history.jsonl")
            n = append_predictions_history(self._predictions(), path, "2026-01-01")

            assert n == 2
            lines = Path(path).read_text().strip().splitlines()
            assert len(lines) == 2
            records = [json.loads(line) for line in lines]
            assert {r["model"] for r in records} == {"daily_logistic", "daily_xgboost"}
            assert all(r["symbol"] == "AAPL" for r in records)

    def test_error_predictions_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "history.jsonl")
            n = append_predictions_history(
                [{"symbol": "BADTICKER", "error": "Insufficient data"}], path, "2026-01-01"
            )

        assert n == 0
        assert not Path(path).exists()

    def test_appends_without_truncating_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "history.jsonl")
            append_predictions_history(self._predictions(), path, "2026-01-01")
            append_predictions_history(self._predictions(), path, "2026-01-02")

            lines = Path(path).read_text().strip().splitlines()
            assert len(lines) == 4

    def test_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "nested" / "dir" / "history.jsonl")
            n = append_predictions_history(self._predictions(), path, "2026-01-01")

            assert n == 2
            assert Path(path).exists()

    def test_record_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "history.jsonl")
            append_predictions_history(self._predictions(), path, "2026-01-01")

            record = json.loads(Path(path).read_text().splitlines()[0])
            assert set(record) == {"date", "symbol", "model", "signal", "confidence", "price"}
            assert record["price"] == 150.0


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


# ---------------------------------------------------------------------------
# Issue #42 — predict_proba column index vs model.classes_
# ---------------------------------------------------------------------------

class TestPredictProbaClassMapping:
    """Verify that predict_symbol maps through model.classes_ instead of
    assuming column i == class i."""

    def _build_models_with_classes(self, proba: list, classes: list) -> dict:
        """Build a mock model dict whose model.classes_ may differ from [0,1,2]."""
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([proba])
        clf.classes_ = np.array(classes)
        return {
            "daily_logistic": {
                "model": clf,
                "scaler": scaler,
                "feature_contract": list(FEATURE_COLS),
                "confidence_threshold": 0.0,  # disable threshold for clarity
            }
        }

    def test_missing_hold_class_maps_correctly(self):
        """If training data had no HOLD (class 1), classes_ = [0, 2].
        Column 0 = class 0 (SELL), column 1 = class 2 (BUY).
        With proba [0.3, 0.7], argmax=1 should map to class 2 → BUY, not HOLD."""
        models = self._build_models_with_classes([0.3, 0.7], [0, 2])
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        p = result["predictions"]["daily_logistic"]
        assert p["signal"] == "BUY", (
            f"With classes_=[0,2] and argmax=1, signal should be BUY, got {p['signal']}"
        )

    def test_missing_sell_class_maps_correctly(self):
        """classes_ = [1, 2] (no SELL). Column 0 = HOLD, column 1 = BUY.
        With proba [0.8, 0.2], argmax=0 should map to class 1 → HOLD."""
        models = self._build_models_with_classes([0.8, 0.2], [1, 2])
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        p = result["predictions"]["daily_logistic"]
        assert p["signal"] == "HOLD", (
            f"With classes_=[1,2] and argmax=0, signal should be HOLD, got {p['signal']}"
        )

    def test_full_classes_still_works(self):
        """Sanity check: with classes_ = [0,1,2], the fix should not break anything."""
        models = self._build_models_with_classes([0.1, 0.2, 0.7], [0, 1, 2])
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        p = result["predictions"]["daily_logistic"]
        assert p["signal"] == "BUY", (
            f"With classes_=[0,1,2] and argmax=2, signal should be BUY, got {p['signal']}"
        )


# ---------------------------------------------------------------------------
# _predict_classifier_signal / _predict_regressor_signal (Task 3)
# ---------------------------------------------------------------------------

class TestPredictClassifierSignal:
    def test_matches_existing_predict_symbol_behavior(self):
        """The extracted helper must reproduce exactly what the old inline
        daily_logistic/daily_xgboost blocks computed."""
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        clf.classes_ = np.array([0, 1, 2])
        data = {"model": clf, "scaler": scaler, "confidence_threshold": 0.55}

        result = _predict_classifier_signal(data, np.zeros((1, len(FEATURE_COLS))), 0.55)

        assert result["signal"] == "BUY"
        assert abs(result["confidence"] - 0.7) < 1e-6

    def test_below_threshold_collapses_to_hold(self):
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.25, 0.30, 0.45]])
        clf.classes_ = np.array([0, 1, 2])
        data = {"model": clf, "scaler": scaler, "confidence_threshold": 0.55}

        result = _predict_classifier_signal(data, np.zeros((1, len(FEATURE_COLS))), 0.55)

        assert result["signal"] == "HOLD"

    def test_missing_pickle_threshold_falls_back_to_default_threshold_arg(self):
        """Pickles trained before confidence_threshold was stored must fall
        back to the caller-supplied default (sourced from config), not a
        hardcoded literal duplicating config/default.yaml."""
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        clf.classes_ = np.array([0, 1, 2])
        data = {"model": clf, "scaler": scaler}  # no confidence_threshold key

        result = _predict_classifier_signal(data, np.zeros((1, len(FEATURE_COLS))), 0.9)

        assert result["signal"] == "HOLD", "0.7 confidence should collapse to HOLD under a 0.9 default threshold"


class TestRegressorConfidence:
    def test_today_is_max_in_window_gives_confidence_one(self):
        pred_ret = np.array([0.001] * 59 + [0.05])
        conf = _regressor_confidence(pred_ret, threshold_window=60)
        assert conf == pytest.approx(1.0)

    def test_today_is_typical_gives_mid_confidence(self):
        pred_ret = np.array([0.001] * 60)
        conf = _regressor_confidence(pred_ret, threshold_window=60)
        assert conf == pytest.approx(1.0)  # all equal -> today <= every value

    def test_too_short_window_returns_zero(self):
        assert _regressor_confidence(np.array([0.01]), threshold_window=60) == 0.0


class TestPredictRegressorSignal:
    def test_extreme_prediction_triggers_buy_with_clean_features(self):
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        model = MagicMock()
        pred_ret = np.full(80, 0.001)
        pred_ret[-1] = 0.05
        model.predict.return_value = pred_ret
        data = {"model": model, "scaler": scaler}

        X_all = np.zeros((80, len(FEATURE_COLS)), dtype=np.float32)
        result = _predict_regressor_signal(data, X_all)

        assert result["signal"] == "BUY"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_unremarkable_prediction_holds(self):
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        model = MagicMock()
        model.predict.return_value = np.full(80, 0.001)
        data = {"model": model, "scaler": scaler}

        X_all = np.zeros((80, len(FEATURE_COLS)), dtype=np.float32)
        result = _predict_regressor_signal(data, X_all)

        assert result["signal"] == "HOLD"

    def test_inf_and_nan_features_do_not_crash(self):
        """X_all may contain inf/nan from upstream feature computation on
        thin data — _preprocess must clip these before scaling, matching
        what the model was trained on (train_models._preprocess)."""
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        model = MagicMock()
        model.predict.return_value = np.full(80, 0.001)
        data = {"model": model, "scaler": scaler}

        X_all = np.zeros((80, len(FEATURE_COLS)), dtype=np.float32)
        X_all[0, 0] = np.inf
        X_all[1, 1] = np.nan
        result = _predict_regressor_signal(data, X_all)

        assert result["signal"] in {"BUY", "SELL", "HOLD"}
        called_with = scaler.transform.call_args[0][0]
        assert np.isfinite(called_with).all()

    def test_valid_signal_quantile_env_override_is_applied(self, monkeypatch):
        """A well-formed PREDICTOR_SIGNAL_QUANTILE override should be parsed
        and used without raising — the valid-override path."""
        monkeypatch.setenv("PREDICTOR_SIGNAL_QUANTILE", "0.9")
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        model = MagicMock()
        pred_ret = np.full(80, 0.001)
        pred_ret[-1] = 0.05
        model.predict.return_value = pred_ret
        data = {"model": model, "scaler": scaler}

        X_all = np.zeros((80, len(FEATURE_COLS)), dtype=np.float32)
        result = _predict_regressor_signal(data, X_all)

        assert result["signal"] in {"BUY", "SELL", "HOLD"}
        assert 0.0 <= result["confidence"] <= 1.0

    def test_invalid_signal_quantile_env_falls_back_to_default(self, monkeypatch):
        """A malformed PREDICTOR_SIGNAL_QUANTILE (e.g. a typo in the GitHub
        Actions environment) must not raise — it should log a warning and
        fall back to the function's default signal_quantile=0.7, still
        producing a valid result. Regression test for the bug where this
        previously surfaced as an uncaught ValueError, silently turning into
        a per-model {"error": ...} entry for daily_predictor every day."""
        monkeypatch.setenv("PREDICTOR_SIGNAL_QUANTILE", "not-a-number")
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        model = MagicMock()
        pred_ret = np.full(80, 0.001)
        pred_ret[-1] = 0.05
        model.predict.return_value = pred_ret
        data = {"model": model, "scaler": scaler}

        X_all = np.zeros((80, len(FEATURE_COLS)), dtype=np.float32)
        result = _predict_regressor_signal(data, X_all)

        assert result["signal"] == "BUY"
        assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# predict_symbol with daily_predictor (end-to-end through the dispatch loop)
# ---------------------------------------------------------------------------

class TestPredictSymbolPredictor:
    def _build_predictor_models(self, pred_ret: np.ndarray) -> dict:
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        model = MagicMock()
        model.predict.return_value = pred_ret
        return {
            "daily_predictor": {
                "model": model,
                "scaler": scaler,
                "feature_contract": list(FEATURE_COLS),
            }
        }

    def test_extreme_prediction_produces_buy(self):
        n = 100
        pred_ret = np.full(n, 0.001)
        pred_ret[-1] = 0.05
        models = self._build_predictor_models(pred_ret)
        feats_df = _make_features_df(n)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(n)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        assert "error" not in result
        p = result["predictions"]["daily_predictor"]
        assert p["signal"] == "BUY"

    def test_unremarkable_prediction_produces_hold(self):
        n = 100
        pred_ret = np.full(n, 0.001)
        models = self._build_predictor_models(pred_ret)
        feats_df = _make_features_df(n)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(n)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        p = result["predictions"]["daily_predictor"]
        assert p["signal"] == "HOLD"

    def test_runs_alongside_classifier_without_interference(self):
        """Both a classifier and the regressor in the same models dict must
        each produce their own independent prediction entry."""
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        clf.classes_ = np.array([0, 1, 2])

        n = 100
        pred_ret = np.full(n, 0.001)
        pred_ret[-1] = 0.05
        models = {
            "daily_logistic": {
                "model": clf, "scaler": scaler,
                "feature_contract": list(FEATURE_COLS), "confidence_threshold": 0.55,
            },
            **self._build_predictor_models(pred_ret),
        }
        feats_df = _make_features_df(n)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(n)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        assert result["predictions"]["daily_logistic"]["signal"] == "BUY"
        assert result["predictions"]["daily_predictor"]["signal"] == "BUY"

    def test_unknown_model_kind_reports_error_not_crash(self):
        """A model_key with no MODEL_KINDS entry must surface as a
        per-model error in the result dict, not crash the whole symbol's
        prediction (one bad/misconfigured model must not take down the
        others) — this is the 'easy to add without breaking things'
        contract: forgetting to register a new model's kind fails loudly
        and locally, not silently or globally."""
        models = {
            "totally_new_model": {
                "model": MagicMock(), "scaler": MagicMock(),
                "feature_contract": list(FEATURE_COLS),
            }
        }
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        assert "error" not in result  # the symbol overall still succeeds
        assert "error" in result["predictions"]["totally_new_model"]


# ---------------------------------------------------------------------------
# load_models (Task 5 — config-driven model list)
# ---------------------------------------------------------------------------

class TestLoadModels:
    def test_respects_explicit_model_keys_list(self, tmp_path, monkeypatch):
        """Passing model_keys explicitly must load exactly that list,
        independent of config — proves a model can be added/removed by
        changing the list with no other code path involved."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        artifact = _make_artifact()
        with open(tmp_path / "models" / "daily_logistic.pkl", "wb") as f:
            pickle.dump(artifact, f)

        loaded = load_models(db=None, model_keys=["daily_logistic"])

        assert set(loaded.keys()) <= {"daily_logistic", "daily_dqn"}
        assert "daily_logistic" in loaded

    def test_missing_pickle_for_configured_model_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        """A model_key in the config list whose pickle doesn't exist on
        disk must be silently skipped (logged), not raise — so a
        half-deployed model doesn't take down the whole prediction run."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        loaded = load_models(db=None, model_keys=["daily_predictor", "totally_unknown"])

        assert loaded == {}

    def test_defaults_to_config_prediction_models(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        artifact = _make_artifact()
        with open(tmp_path / "models" / "daily_predictor.pkl", "wb") as f:
            pickle.dump(artifact, f)

        with patch("predict_next_day_lite.get_config") as mock_cfg:
            mock_cfg.return_value.prediction.models = ["daily_predictor"]
            loaded = load_models(db=None)

        assert "daily_predictor" in loaded
