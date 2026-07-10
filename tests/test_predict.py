"""test_predict.py — Tests for predict_next_day_lite.py (no network, no real models)."""
import json
import logging
import os
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
# _load_bars_cached / predict_symbol DB-cache reuse (#93)
# ---------------------------------------------------------------------------

def _make_ohlcv_ending_today(n: int = 100) -> pd.DataFrame:
    """Like _make_ohlcv, but the index ends at today's date so the cache
    lines up with predict_symbol's datetime.now()-based date range."""
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq="B")
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


class TestLoadBarsCached:
    """predict_symbol should reuse the DB bar cache instead of re-fetching
    the full history_days window from yfinance on every run (#93)."""

    def test_no_db_always_fetches_full_window(self):
        """With db=None, behavior is unchanged: always fetch directly."""
        from predict_next_day_lite import _load_bars_cached

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)) as mock_fetch:
            result = _load_bars_cached("AAPL", "2023-01-01", "2023-06-01", db=None)

        mock_fetch.assert_called_once()
        assert result is not None and len(result) == 100

    def test_fresh_complete_cache_skips_yfinance_fetch(self, tmp_path):
        """When the DB cache is fresh and covers the requested start, no
        network fetch should happen — this is the core optimization."""
        from db import DB
        from predict_next_day_lite import _load_bars_cached

        db = DB(str(tmp_path / "test.db"))
        data = _make_ohlcv(100)
        db.upsert_bars("AAPL", "1d", data)

        latest_bar = data.index.max()
        end_date = (latest_bar + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
        start_date = data.index.min().strftime("%Y-%m-%d")

        with patch("predict_next_day_lite.load_yfinance") as mock_fetch:
            result = _load_bars_cached("AAPL", start_date, end_date, db=db)

        mock_fetch.assert_not_called()
        assert result is not None
        assert len(result) == 100

    def test_stale_cache_triggers_fetch_and_upsert(self, tmp_path):
        """When the cache is stale (latest bar too old), fall back to a full
        yfinance fetch and write the fresh data back into the DB."""
        from db import DB
        from predict_next_day_lite import _load_bars_cached

        db = DB(str(tmp_path / "test.db"))
        old_data = _make_ohlcv(100)
        db.upsert_bars("AAPL", "1d", old_data)

        # Request an end date far past the cache's latest bar.
        latest_bar = old_data.index.max()
        end_date = (latest_bar + pd.tseries.offsets.BDay(30)).strftime("%Y-%m-%d")
        start_date = old_data.index.min().strftime("%Y-%m-%d")

        fresh_data = _make_ohlcv(130)
        with patch("predict_next_day_lite.load_yfinance", return_value=fresh_data) as mock_fetch:
            result = _load_bars_cached("AAPL", start_date, end_date, db=db)

        mock_fetch.assert_called_once()
        assert result is not None and len(result) == 130

        # The fresh fetch should have been persisted back into the DB cache
        # (query a wide-open range to avoid an off-by-one at the exact
        # boundary date).
        recached = db.load_bars("AAPL", "1d", "1990-01-01", "2100-01-01")
        assert recached is not None and len(recached) == 130

    def test_cache_missing_older_history_triggers_fetch(self, tmp_path):
        """A cache that is fresh but doesn't reach back to the requested
        start must not be silently reused (regression class from #25,
        applied here to the prediction path)."""
        from db import DB
        from predict_next_day_lite import _load_bars_cached

        db = DB(str(tmp_path / "test.db"))
        short_data = _make_ohlcv(100)
        db.upsert_bars("AAPL", "1d", short_data)

        latest_bar = short_data.index.max()
        end_date = (latest_bar + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
        # Ask for history starting well before the cache's earliest bar.
        much_earlier_start = (short_data.index.min() - pd.tseries.offsets.BDay(500)).strftime("%Y-%m-%d")

        long_data = _make_ohlcv(600)
        with patch("predict_next_day_lite.load_yfinance", return_value=long_data) as mock_fetch:
            result = _load_bars_cached("AAPL", much_earlier_start, end_date, db=db)

        mock_fetch.assert_called_once()
        assert result is not None and len(result) == 600

    def test_predict_symbol_uses_cache_and_skips_fetch(self, tmp_path):
        """End-to-end: predict_symbol with a warm, fresh DB cache should not
        call yfinance at all."""
        from db import DB

        db = DB(str(tmp_path / "test.db"))
        data = _make_ohlcv_ending_today(200)
        db.upsert_bars("AAPL", "1d", data)

        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        clf.classes_ = np.array([0, 1, 2])
        models = {
            "daily_logistic": {
                "model": clf,
                "scaler": scaler,
                "feature_contract": list(FEATURE_COLS),
                "confidence_threshold": 0.55,
            }
        }
        feats_df = _make_features_df(200)

        # history_days must be short enough that the requested start falls
        # within the cached window's coverage tolerance.
        with patch("predict_next_day_lite.load_yfinance") as mock_fetch, \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol(
                "AAPL", models, db=db, history_days=200,
            )

        mock_fetch.assert_not_called()
        assert "error" not in result
        assert result["predictions"]["daily_logistic"]["signal"] == "BUY"

    def test_predict_symbol_falls_back_to_fetch_when_cache_empty(self, tmp_path):
        """With an empty DB, predict_symbol must still fetch from yfinance
        (no regression to 'cache-only, no data' behavior)."""
        from db import DB

        db = DB(str(tmp_path / "test.db"))
        models = {
            "daily_logistic": {
                "model": MagicMock(predict_proba=MagicMock(return_value=np.array([[0.1, 0.2, 0.7]])),
                                    classes_=np.array([0, 1, 2])),
                "scaler": MagicMock(transform=MagicMock(side_effect=lambda x: x)),
                "feature_contract": list(FEATURE_COLS),
                "confidence_threshold": 0.55,
            }
        }
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)) as mock_fetch, \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models, db=db)

        mock_fetch.assert_called_once()
        assert "error" not in result


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

    def test_pure_hold_day_sends_no_embeds(self):
        """When every signal for a strategy is HOLD, suppress the HOLD embed
        — a pure-HOLD day produces no strategy embeds and returns False."""
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        preds = self._minimal_predictions(signal="HOLD", confidence=0.6)

        with patch("requests.post", return_value=mock_resp):
            result = send_discord(preds, "https://discord.example/webhook")

        assert result is False

    def test_hold_included_alongside_buy_signals(self):
        """HOLD embed must appear when the same strategy also has BUY records."""
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        preds = [
            {"symbol": "AAPL", "price": 150.0,
             "predictions": {"daily_logistic": {"signal": "BUY", "confidence": 0.8}}},
            {"symbol": "MSFT", "price": 420.0,
             "predictions": {"daily_logistic": {"signal": "HOLD", "confidence": 0.6}}},
        ]

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = send_discord(preds, "https://discord.example/webhook")

        assert result is True
        payload = mock_post.call_args.kwargs["json"]
        titles = [e["title"] for e in payload["embeds"]]
        assert any("BUY" in t for t in titles)
        assert any("HOLD" in t for t in titles)

    def test_consensus_buy_embed_appears_when_two_models_agree(self):
        """★ Consensus — BUY embed must appear when ≥2 models signal BUY for the same symbol."""
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        preds = [
            {"symbol": "AAPL", "price": 150.0, "predictions": {
                "daily_logistic": {"signal": "BUY", "confidence": 0.8},
                "daily_xgboost": {"signal": "BUY", "confidence": 0.75},
            }},
        ]

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = send_discord(preds, "https://discord.example/webhook")

        assert result is True
        payload = mock_post.call_args.kwargs["json"]
        titles = [e["title"] for e in payload["embeds"]]
        assert any("Consensus" in t and "BUY" in t for t in titles), (
            f"Expected consensus BUY embed, got titles: {titles}"
        )
        # Consensus embed must come first
        assert "Consensus" in payload["embeds"][0]["title"]

    def test_consensus_absent_when_models_disagree(self):
        """No consensus embed when each symbol has only one model voting BUY."""
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        preds = [
            {"symbol": "AAPL", "price": 150.0, "predictions": {
                "daily_logistic": {"signal": "BUY", "confidence": 0.8},
                "daily_xgboost": {"signal": "SELL", "confidence": 0.75},
            }},
        ]

        with patch("requests.post", return_value=mock_resp) as mock_post:
            send_discord(preds, "https://discord.example/webhook")

        payload = mock_post.call_args.kwargs["json"]
        titles = [e["title"] for e in payload["embeds"]]
        assert not any("Consensus" in t for t in titles)

    def test_consensus_sell_embed_appears_when_two_models_agree(self):
        """★ Consensus — SELL embed must appear when ≥2 models signal SELL for a symbol."""
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        preds = [
            {"symbol": "TSLA", "price": 250.0, "predictions": {
                "daily_logistic": {"signal": "SELL", "confidence": 0.8},
                "daily_predictor": {"signal": "SELL", "confidence": 0.7},
            }},
        ]

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = send_discord(preds, "https://discord.example/webhook")

        assert result is True
        payload = mock_post.call_args.kwargs["json"]
        titles = [e["title"] for e in payload["embeds"]]
        assert any("Consensus" in t and "SELL" in t for t in titles)

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
# daily_hybrid (issue #64) — XGBoost-transformer blend
# ---------------------------------------------------------------------------

import torch as _torch

from hybrid_model import TransformerCfg, TransformerEncoder
from predict_next_day_lite import _load_hybrid_pkl, _predict_hybrid_signal


def _make_hybrid_artifact(lookback: int = 10, blend_alpha: float = 0.4,
                           use_embeddings: bool = False,
                           feature_contract=None) -> dict:
    """Build a real (tiny) TransformerEncoder + picklable XGBoost stub so
    _load_hybrid_pkl's reconstruction path can be exercised end-to-end,
    following train_hybrid.py's artifact schema exactly."""
    if feature_contract is None:
        feature_contract = FEATURE_COLS
    tcfg = TransformerCfg(
        lookback=lookback, d_model=8, nhead=2, num_layers=1,
        dim_feedforward=16, dropout=0.0, embed_dim=4, num_classes=3,
    )
    model = TransformerEncoder(n_features=len(feature_contract), cfg=tcfg)
    return {
        "model": _PicklableStub(),
        "scaler": _PicklableStub(),
        "transformer_state": model.state_dict(),
        "transformer_cfg": tcfg.__dict__,
        "lookback": lookback,
        "feature_contract": list(feature_contract),
        "blend_alpha": blend_alpha,
        "use_embeddings": use_embeddings,
        "confidence_threshold": 0.4,
    }


class TestLoadHybridPkl:
    def test_valid_artifact_loads_and_reconstructs_transformer(self):
        artifact = _make_hybrid_artifact()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(artifact, f)
            path = f.name

        loaded = _load_hybrid_pkl(path)

        assert "transformer" in loaded
        assert isinstance(loaded["transformer"], TransformerEncoder)
        assert loaded["lookback"] == 10
        assert loaded["blend_alpha"] == 0.4

    def test_missing_file_raises(self):
        with pytest.raises(RuntimeError, match="not found"):
            _load_hybrid_pkl("/nonexistent/daily_hybrid.pkl")

    def test_missing_required_key_raises(self):
        artifact = _make_hybrid_artifact()
        del artifact["blend_alpha"]
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(artifact, f)
            path = f.name
        with pytest.raises(RuntimeError, match="missing key 'blend_alpha'"):
            _load_hybrid_pkl(path)

    def test_wrong_feature_contract_raises(self):
        artifact = _make_hybrid_artifact(feature_contract=["wrong_a", "wrong_b"])
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(artifact, f)
            path = f.name
        with pytest.raises(RuntimeError, match="Feature contract mismatch"):
            _load_hybrid_pkl(path)


class TestLoadModelsHybrid:
    def test_loads_daily_hybrid_when_pickle_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        artifact = _make_hybrid_artifact()
        with open(tmp_path / "models" / "daily_hybrid.pkl", "wb") as f:
            pickle.dump(artifact, f)

        loaded = load_models(db=None, model_keys=[])

        assert "daily_hybrid" in loaded
        assert "transformer" in loaded["daily_hybrid"]
        assert loaded["daily_hybrid"]["_artifact_path"].endswith("daily_hybrid.pkl")

    def test_missing_pickle_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        loaded = load_models(db=None, model_keys=[])

        assert loaded == {}

    def test_corrupt_pickle_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        with open(tmp_path / "models" / "daily_hybrid.pkl", "wb") as f:
            f.write(b"not a pickle")

        loaded = load_models(db=None, model_keys=[])

        assert "daily_hybrid" not in loaded


class TestPredictSymbolHybrid:
    """predict_symbol's daily_hybrid branch: build the lookback window the
    same way build_sequences does, run the (mocked) transformer + XGBoost,
    and blend argmax(alpha * P_xgb + (1 - alpha) * P_tx)."""

    def _build_hybrid_models(self, logits, xgb_proba, classes=(0, 1, 2),
                              blend_alpha=0.5, lookback=10,
                              use_embeddings=False, confidence_threshold=0.1) -> dict:
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x

        transformer = MagicMock()
        logits_t = _torch.tensor([logits], dtype=_torch.float32)
        embed_t = _torch.zeros((1, 4), dtype=_torch.float32)
        transformer.return_value = (logits_t, embed_t)

        xgb_model = MagicMock()
        xgb_model.predict_proba.return_value = np.array([xgb_proba])
        xgb_model.classes_ = np.array(classes)

        return {
            "daily_hybrid": {
                "model": xgb_model,
                "scaler": scaler,
                "transformer": transformer,
                "lookback": lookback,
                "feature_contract": list(FEATURE_COLS),
                "blend_alpha": blend_alpha,
                "use_embeddings": use_embeddings,
                "confidence_threshold": confidence_threshold,
            }
        }

    def test_blends_xgb_and_transformer_probs(self):
        logits = [0.0, 1.0, 2.0]  # softmax favors BUY (idx 2)
        tx_probs = _torch.softmax(_torch.tensor(logits), dim=-1).numpy()
        xgb_proba = [0.7, 0.2, 0.1]  # favors SELL (idx 0)
        models = self._build_hybrid_models(
            logits, xgb_proba, blend_alpha=0.5, confidence_threshold=0.1
        )
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        assert "error" not in result
        p = result["predictions"]["daily_hybrid"]
        blended = 0.5 * np.array(xgb_proba) + 0.5 * tx_probs
        expected_idx = int(np.argmax(blended))
        expected_signal = ["SELL", "HOLD", "BUY"][expected_idx]
        assert p["signal"] == expected_signal
        assert abs(p["confidence"] - blended[expected_idx]) < 1e-6

    def test_low_confidence_collapses_to_hold(self):
        # Both models weakly favor BUY but blended confidence stays under
        # the configured threshold, so the result should collapse to HOLD.
        logits = [0.33, 0.34, 0.35]
        xgb_proba = [0.30, 0.32, 0.38]
        models = self._build_hybrid_models(
            logits, xgb_proba, blend_alpha=0.5, confidence_threshold=0.9
        )
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        p = result["predictions"]["daily_hybrid"]
        assert p["signal"] == "HOLD"

    def test_xgb_classes_out_of_order_realigned(self):
        """If model.classes_ isn't [0,1,2] in order, predict_proba's columns
        must be realigned before blending — same contract as issue #42 for
        the plain classifiers (_predict_classifier_signal)."""
        logits = [-5.0, -5.0, -5.0]  # ~uniform transformer probs
        # classes_ = [2, 0, 1] means predict_proba columns are ordered
        # BUY, SELL, HOLD instead of SELL, HOLD, BUY
        xgb_proba = [0.9, 0.05, 0.05]  # column 0 -> class 2 (BUY)
        models = self._build_hybrid_models(
            logits, xgb_proba, classes=(2, 0, 1), blend_alpha=1.0,
            confidence_threshold=0.1,
        )
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        p = result["predictions"]["daily_hybrid"]
        # blend_alpha=1.0 -> pure XGBoost; realigned probs put 0.9 mass on
        # class 2 (BUY), so the signal must be BUY, not SELL.
        assert p["signal"] == "BUY"

    def test_insufficient_lookback_history_reports_error_not_crash(self):
        """Fewer bars than the model's lookback window must surface as a
        per-model error (matching the other models' try/except contract),
        not crash predict_symbol or the other models' predictions."""
        models = self._build_hybrid_models(
            [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], lookback=1000,
        )
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        assert "error" not in result  # the symbol overall still succeeds
        assert "error" in result["predictions"]["daily_hybrid"]

    def test_runs_alongside_other_models_without_interference(self):
        """daily_hybrid must produce an independent prediction entry
        alongside a classifier in the same models dict (mirrors
        TestPredictSymbolPredictor.test_runs_alongside_classifier_...)."""
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        clf.classes_ = np.array([0, 1, 2])

        models = {
            "daily_logistic": {
                "model": clf, "scaler": scaler,
                "feature_contract": list(FEATURE_COLS), "confidence_threshold": 0.55,
            },
            **self._build_hybrid_models(
                [0.1, 0.2, 5.0], [0.1, 0.2, 0.7], blend_alpha=0.5, confidence_threshold=0.1
            ),
        }
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        assert result["predictions"]["daily_logistic"]["signal"] == "BUY"
        assert result["predictions"]["daily_hybrid"]["signal"] == "BUY"


# ---------------------------------------------------------------------------
# JSON structured logging (Task 1)
# ---------------------------------------------------------------------------

from predict_next_day_lite import _JsonFormatter, _RunIdFilter, _configure_logging


class TestJsonLogging:
    def test_run_id_filter_injects_attribute(self):
        f = _RunIdFilter("test-run-abc")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        f.filter(record)
        assert record.run_id == "test-run-abc"

    def test_run_id_filter_returns_true(self):
        f = _RunIdFilter("x")
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="", lineno=0,
            msg="m", args=(), exc_info=None,
        )
        assert f.filter(record) is True

    def test_json_formatter_produces_valid_json(self):
        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="mymodule", level=logging.WARNING, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        record.run_id = "abc-123"
        output = fmt.format(record)
        doc = json.loads(output)
        assert doc["msg"] == "test message"
        assert doc["run_id"] == "abc-123"
        assert doc["level"] == "WARNING"
        assert doc["logger"] == "mymodule"
        assert "ts" in doc

    def test_json_formatter_omits_exc_key_when_no_exception(self):
        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="", lineno=0,
            msg="no exc", args=(), exc_info=None,
        )
        record.run_id = ""
        doc = json.loads(fmt.format(record))
        assert "exc" not in doc

    def test_configure_logging_installs_json_formatter(self):
        _configure_logging("run-xyz")
        root = logging.getLogger()
        formatters = [h.formatter for h in root.handlers if h.formatter is not None]
        assert any(isinstance(f, _JsonFormatter) for f in formatters)
        # Restore plain logging so subsequent tests aren't affected
        root.handlers.clear()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


# ---------------------------------------------------------------------------
# load_models (Task 5 — config-driven model list)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model staleness detection (Task 2)
# ---------------------------------------------------------------------------

import time
from predict_next_day_lite import check_model_staleness


class TestCheckModelStaleness:
    def _make_pkl(self, tmp_path: Path, name: str) -> str:
        artifact = _make_artifact()
        p = tmp_path / f"{name}.pkl"
        with open(p, "wb") as f:
            pickle.dump(artifact, f)
        return str(p)

    def test_fresh_model_not_in_result(self, tmp_path):
        path = self._make_pkl(tmp_path, "daily_logistic")
        models = {"daily_logistic": {"_artifact_path": path}}
        stale = check_model_staleness(models, max_age_days=30)
        assert "daily_logistic" not in stale

    def test_old_model_appears_in_result_with_age(self, tmp_path):
        path = self._make_pkl(tmp_path, "daily_logistic")
        # Backdate mtime by 45 days
        old_time = time.time() - 45 * 86400
        os.utime(path, (old_time, old_time))
        models = {"daily_logistic": {"_artifact_path": path}}
        stale = check_model_staleness(models, max_age_days=30)
        assert "daily_logistic" in stale
        assert stale["daily_logistic"] >= 44

    def test_missing_artifact_path_is_skipped(self):
        models = {"daily_dqn": object()}  # no _artifact_path
        stale = check_model_staleness(models, max_age_days=30)
        assert stale == {}

    def test_nonexistent_path_is_skipped(self):
        models = {"daily_predictor": {"_artifact_path": "/nonexistent/model.pkl"}}
        stale = check_model_staleness(models, max_age_days=30)
        assert stale == {}

    def test_mixed_fresh_and_stale(self, tmp_path):
        fresh = self._make_pkl(tmp_path, "fresh")
        stale_path = self._make_pkl(tmp_path, "stale")
        old_time = time.time() - 40 * 86400
        os.utime(stale_path, (old_time, old_time))
        models = {
            "daily_logistic": {"_artifact_path": fresh},
            "daily_predictor": {"_artifact_path": stale_path},
        }
        result = check_model_staleness(models, max_age_days=30)
        assert "daily_logistic" not in result
        assert "daily_predictor" in result


class TestSendDiscordStaleModels:
    def _minimal_predictions(self):
        return [
            {"symbol": "AAPL", "price": 150.0,
             "predictions": {"daily_logistic": {"signal": "BUY", "confidence": 0.8}}}
        ]

    def test_stale_model_warning_embed_prepended(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        with patch("requests.post", return_value=mock_resp) as mock_post:
            send_discord(
                self._minimal_predictions(),
                "https://discord.example/webhook",
                stale_models={"daily_predictor": 45},
            )
        payload = mock_post.call_args.kwargs["json"]
        titles = [e["title"] for e in payload["embeds"]]
        assert any("daily_predictor" in t for t in titles)
        assert any("Stale" in t for t in titles)
        # Warning must be the first embed
        assert "daily_predictor" in payload["embeds"][0]["title"]

    def test_no_stale_models_no_warning_embed(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        with patch("requests.post", return_value=mock_resp) as mock_post:
            send_discord(
                self._minimal_predictions(),
                "https://discord.example/webhook",
                stale_models={},
            )
        payload = mock_post.call_args.kwargs["json"]
        titles = [e["title"] for e in payload["embeds"]]
        assert not any("Stale" in t for t in titles)


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
        assert "_artifact_path" in loaded["daily_logistic"]
        assert loaded["daily_logistic"]["_artifact_path"].endswith("daily_logistic.pkl")

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
