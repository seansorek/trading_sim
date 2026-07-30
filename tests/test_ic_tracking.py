"""test_ic_tracking.py — Tests for score_realized_ic in signal_monitor.py."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_monitor import score_realized_ic, _signal_to_score
from daily_features import FWD_RET_HORIZON_DAYS


def _write_history(tmp_path, records: list[dict]) -> str:
    path = tmp_path / "history.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


def _make_price_df_from_close(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"close": closes}, index=idx)


def test_signal_to_score_buy():
    assert _signal_to_score("BUY", 0.8) == pytest.approx(0.8)


def test_signal_to_score_sell():
    assert _signal_to_score("SELL", 0.8) == pytest.approx(-0.8)


def test_signal_to_score_hold():
    assert _signal_to_score("HOLD", 0.9) == pytest.approx(0.0)


def test_score_realized_ic_returns_none_when_insufficient_history(tmp_path):
    today = date.today().strftime("%Y-%m-%d")
    # Only 5 records — below min_lookback=20
    old_date = (date.today() - timedelta(days=FWD_RET_HORIZON_DAYS + 10)).strftime("%Y-%m-%d")
    records = [
        {"date": old_date, "symbol": "AAPL", "model": "daily_predictor",
         "signal": "BUY", "confidence": 0.7, "price": 100.0}
        for _ in range(5)
    ]
    path = _write_history(tmp_path, records)
    result = score_realized_ic(path, today, fetch_prices_fn=lambda s, st, en: None)
    assert result["daily_predictor"] is None


def test_score_realized_ic_excludes_recent_entries(tmp_path):
    """Entries within FWD_RET_HORIZON_DAYS of today cannot be scored yet."""
    today = date.today()
    records = []
    # 20 records that are too recent
    for i in range(20):
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        records.append({"date": d, "symbol": "AAPL", "model": "daily_predictor",
                        "signal": "BUY", "confidence": 0.7, "price": 100.0})
    path = _write_history(tmp_path, records)
    result = score_realized_ic(path, today.strftime("%Y-%m-%d"),
                               fetch_prices_fn=lambda s, st, en: None)
    assert result.get("daily_predictor") is None


def test_score_realized_ic_computes_correct_ic(tmp_path):
    """When BUY predictions are followed by positive returns, IC should be positive."""
    today = date.today()
    cutoff_date = today - timedelta(days=FWD_RET_HORIZON_DAYS + 1)
    records = []
    # 20 BUY predictions; store indices for lookup
    indices_by_date = {}
    for i in range(20):
        d = (cutoff_date - timedelta(days=i)).strftime("%Y-%m-%d")
        indices_by_date[d] = i
        records.append({"date": d, "symbol": "AAPL", "model": "daily_predictor",
                        "signal": "BUY", "confidence": 0.7 + i * 0.01, "price": 100.0})

    path = _write_history(tmp_path, records)

    # Mock: return prices where realized_return is positive and correlates with date
    def mock_fetch(symbol, start, end):
        # Extract the prediction date from the start parameter (it's the pred_date)
        # Return FWD_RET_HORIZON_DAYS+1 bars; close[FWD_RET_HORIZON_DAYS] > close[0]
        # Realized return varies positively: more recent predictions (higher confidence) get higher returns
        # We'll vary based on the start date to correlate with confidence
        pred_date = start
        idx = indices_by_date.get(pred_date, 10)
        realized_return = 0.01 + (idx * 0.0001)  # Higher idx (older dates) get smaller returns
        closes = [100.0] + [100.0] * (FWD_RET_HORIZON_DAYS - 1) + [100.0 * (1 + realized_return)]
        return _make_price_df_from_close(closes)

    result = score_realized_ic(path, today.strftime("%Y-%m-%d"), fetch_prices_fn=mock_fetch)
    assert result["daily_predictor"] is not None
    assert result["daily_predictor"]["ic"] > 0


def test_score_realized_ic_excludes_near_zero_returns(tmp_path):
    """Rows where |realized_return| < 1e-5 (likely holidays) are excluded."""
    today = date.today()
    cutoff_date = today - timedelta(days=FWD_RET_HORIZON_DAYS + 1)
    records = []
    for i in range(20):
        d = (cutoff_date - timedelta(days=i)).strftime("%Y-%m-%d")
        records.append({"date": d, "symbol": "AAPL", "model": "daily_predictor",
                        "signal": "BUY", "confidence": 0.7, "price": 100.0})
    path = _write_history(tmp_path, records)

    # All fetched prices show zero return (flat)
    def mock_flat_fetch(symbol, start, end):
        return _make_price_df_from_close([100.0] * (FWD_RET_HORIZON_DAYS + 1))

    result = score_realized_ic(path, today.strftime("%Y-%m-%d"), fetch_prices_fn=mock_flat_fetch)
    # All rows excluded → insufficient scored rows → None
    assert result.get("daily_predictor") is None


def test_score_realized_ic_directional_accuracy_ignores_holds(tmp_path):
    """HOLD signals (score=0) must not count as directional misses."""
    today = date.today()
    cutoff_date = today - timedelta(days=FWD_RET_HORIZON_DAYS + 1)
    records = []
    for i in range(20):
        d = (cutoff_date - timedelta(days=i)).strftime("%Y-%m-%d")
        # 10 correct BUYs, 10 HOLDs
        signal = "BUY" if i < 10 else "HOLD"
        records.append({"date": d, "symbol": "AAPL", "model": "daily_predictor",
                        "signal": signal, "confidence": 0.7, "price": 100.0})
    path = _write_history(tmp_path, records)

    def mock_fetch(symbol, start, end):
        closes = [100.0] + [100.0] * (FWD_RET_HORIZON_DAYS - 1) + [101.0]
        return _make_price_df_from_close(closes)

    result = score_realized_ic(path, today.strftime("%Y-%m-%d"), fetch_prices_fn=mock_fetch)
    assert result["daily_predictor"]["directional_accuracy"] == pytest.approx(1.0)
    assert result["daily_predictor"]["n_directional"] == 10


def test_score_realized_ic_handles_missing_file(tmp_path):
    result = score_realized_ic(str(tmp_path / "nonexistent.jsonl"),
                               date.today().strftime("%Y-%m-%d"),
                               fetch_prices_fn=lambda s, st, en: None)
    assert result == {}
