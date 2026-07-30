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
    # 20 BUY predictions; older records (higher i) get higher confidence
    for i in range(20):
        d = (cutoff_date - timedelta(days=i)).strftime("%Y-%m-%d")
        records.append({"date": d, "symbol": "AAPL", "model": "daily_predictor",
                        "signal": "BUY", "confidence": 0.7 + i * 0.01, "price": 100.0})

    path = _write_history(tmp_path, records)

    # Prices rise monotonically further back in time, so older (higher-confidence)
    # predictions realize larger returns — single fetch spans the whole date range.
    def mock_fetch(symbol, start, end):
        idx = pd.bdate_range(start, end)
        closes = [100.0 * (1 + 0.01 + (today - d.date()).days * 0.0001) for d in idx]
        return pd.DataFrame({"close": closes}, index=idx)

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


def test_score_realized_ic_fetches_once_per_symbol(tmp_path):
    """Regardless of model count or lookback size, fetch_prices_fn is called once per symbol."""
    today = date.today()
    cutoff_date = today - timedelta(days=FWD_RET_HORIZON_DAYS + 1)
    records = []
    for model in ("daily_predictor", "hourly_predictor"):
        for i in range(20):
            d = (cutoff_date - timedelta(days=i)).strftime("%Y-%m-%d")
            records.append({"date": d, "symbol": "AAPL", "model": model,
                            "signal": "BUY", "confidence": 0.7, "price": 100.0})
    path = _write_history(tmp_path, records)

    call_count = {"n": 0}

    def mock_fetch(symbol, start, end):
        call_count["n"] += 1
        idx = pd.bdate_range(start, end)
        closes = [100.0 + i * 0.5 for i in range(len(idx))]
        return pd.DataFrame({"close": closes}, index=idx)

    result = score_realized_ic(path, today.strftime("%Y-%m-%d"), fetch_prices_fn=mock_fetch)
    assert call_count["n"] == 1  # one symbol -> one fetch, even across two models
    assert result["daily_predictor"] is not None
    assert result["hourly_predictor"] is not None


def test_score_realized_ic_handles_missing_file(tmp_path):
    result = score_realized_ic(str(tmp_path / "nonexistent.jsonl"),
                               date.today().strftime("%Y-%m-%d"),
                               fetch_prices_fn=lambda s, st, en: None)
    assert result == {}
