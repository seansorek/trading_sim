"""test_drift_detection.py — Tests for check_signal_drift in signal_monitor.py."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_monitor import check_signal_drift


def _write_history(tmp_path, records: list[dict]) -> str:
    path = tmp_path / "history.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


def _build_history(
    tmp_path,
    n_days: int = 35,
    base_score: float = 0.0,
    yesterday_score: float | None = None,
) -> tuple[str, str]:
    """Build history with stable daily mean = base_score for n_days.
    Optionally override yesterday's entry."""
    today = date.today()
    records = []
    for i in range(2, n_days + 2):  # from 2 days ago backwards
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        signal = "BUY" if base_score >= 0 else "SELL"
        records.append({
            "date": d, "symbol": "AAPL", "model": "daily_predictor",
            "signal": signal, "confidence": abs(base_score), "price": 100.0,
        })

    if yesterday_score is not None:
        yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        signal = "BUY" if yesterday_score >= 0 else "SELL"
        records.append({
            "date": yesterday, "symbol": "AAPL", "model": "daily_predictor",
            "signal": signal, "confidence": abs(yesterday_score), "price": 100.0,
        })

    path = _write_history(tmp_path, records)
    return path, today.strftime("%Y-%m-%d")


def test_no_drift_when_today_is_normal(tmp_path):
    path, today = _build_history(tmp_path, base_score=0.3)
    # Today's mean score is the same as historical mean — no drift
    result = check_signal_drift(path, today, {"daily_predictor": 0.3})
    assert result.get("daily_predictor") is False


def test_no_drift_warning_on_single_day_shift(tmp_path):
    """Even if today's shift exceeds threshold, single-day guard prevents warning."""
    path, today = _build_history(tmp_path, base_score=0.1)
    # Today is very bullish but yesterday was normal (no history of yesterday means no warning)
    result = check_signal_drift(path, today, {"daily_predictor": 0.9})
    assert result.get("daily_predictor") is False


def test_drift_warning_on_two_consecutive_days(tmp_path):
    """Two consecutive days of large shift triggers a warning."""
    # Historical mean ~0.1; yesterday was 0.9 (large shift); today also 0.9
    path, today = _build_history(tmp_path, base_score=0.1, yesterday_score=0.9)
    result = check_signal_drift(path, today, {"daily_predictor": 0.9})
    assert result.get("daily_predictor") is True


def test_no_drift_when_abs_shift_below_threshold(tmp_path):
    """Even if z-score is high, abs shift must exceed DRIFT_ABS_THRESHOLD (0.002).

    This test validates that the abs-threshold guard (line 212-214 in signal_monitor.py)
    actually blocks a warning when abs(score - mu) < 0.002, even if z-score > 2.0.

    Setup: alternating BUY/HOLD baseline (29 days) creates genuine std ≈ 0.0001.
    - Even baseline days (i in 2..30, step 2): BUY with confidence 0.0002 → score +0.0002
    - Odd baseline days (i in 3..29, step 2): HOLD with confidence 0.0 → score 0.0
    - Baseline mean ≈ 0.0001, std ≈ 0.0001
    - Yesterday: confidence 0.001 (BUY) → score +0.001 → abs-shift ≈ 0.0009 < 0.002
    - Today: confidence 0.001 (BUY) → score +0.001 → z ≈ 9 (> 2.0) but abs-shift < 0.002
    - Expected: False (abs-threshold blocks both yesterday and today)
    """
    today = date.today()
    records = []

    # Build 29-day baseline with alternating BUY/HOLD
    for i in range(2, 31):
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        if i % 2 == 0:  # Even: BUY 0.0002
            records.append({
                "date": d, "symbol": "AAPL", "model": "daily_predictor",
                "signal": "BUY", "confidence": 0.0002, "price": 100.0,
            })
        else:  # Odd: HOLD 0.0
            records.append({
                "date": d, "symbol": "AAPL", "model": "daily_predictor",
                "signal": "HOLD", "confidence": 0.0, "price": 100.0,
            })

    # Yesterday: score 0.001 (high z-score but low abs-shift)
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    records.append({
        "date": yesterday, "symbol": "AAPL", "model": "daily_predictor",
        "signal": "BUY", "confidence": 0.001, "price": 100.0,
    })

    path = _write_history(tmp_path, records)
    # Today: same as yesterday (0.001) → z ≈ 9, abs-shift ≈ 0.0009
    result = check_signal_drift(path, today.strftime("%Y-%m-%d"), {"daily_predictor": 0.001})
    assert result.get("daily_predictor") is False


def test_no_drift_when_std_is_zero(tmp_path):
    """All-constant historical scores → std=0 → skip drift check, no exception."""
    # All historical entries are identical BUY 0.5
    path, today = _build_history(tmp_path, base_score=0.5, yesterday_score=0.5)
    result = check_signal_drift(path, today, {"daily_predictor": 0.9})
    # std=0 → can't compute z-score → no warning
    assert result.get("daily_predictor") is False


def test_drift_returns_false_when_insufficient_window(tmp_path):
    """Fewer than MIN_DRIFT_WINDOW days of history → no warning."""
    today = date.today()
    records = []
    for i in range(2, 5):  # only 3 historical days
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        records.append({"date": d, "symbol": "AAPL", "model": "daily_predictor",
                        "signal": "BUY", "confidence": 0.1, "price": 100.0})
    path = _write_history(tmp_path, records)
    result = check_signal_drift(path, today.strftime("%Y-%m-%d"), {"daily_predictor": 0.9})
    assert result.get("daily_predictor") is False
