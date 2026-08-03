"""test_ic_tracking.py — Tests for score_realized_ic in signal_monitor.py."""
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_monitor import score_realized_ic, _signal_to_score, _dedupe_scoreable
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


def _business_days_before(cutoff_date, n):
    """
    `n` distinct real trading-session dates at/before `cutoff_date`, most
    recent first. Used instead of raw calendar-day subtraction so each
    synthetic record represents a genuinely independent session (one
    calendar day can otherwise land on a weekend and collapse into its
    neighbor under session-anchored dedup — see #134/#136).
    """
    idx = pd.bdate_range(end=cutoff_date, periods=n)
    return [d.date() for d in reversed(idx)]


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
    dates = _business_days_before(cutoff_date, 20)
    # 20 BUY predictions; older records (higher i) get higher confidence
    for i, day in enumerate(dates):
        d = day.strftime("%Y-%m-%d")
        # Entry price varies per record (as it would for real distinct trading
        # sessions) so these aren't mistaken for weekend/pre-open duplicates.
        records.append({"date": d, "symbol": "AAPL", "model": "daily_predictor",
                        "signal": "BUY", "confidence": 0.7 + i * 0.01, "price": 100.0 + i * 1e-6})

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
    dates = _business_days_before(cutoff_date, 20)
    for model in ("daily_predictor", "hourly_predictor"):
        for i, day in enumerate(dates):
            d = day.strftime("%Y-%m-%d")
            # Distinct entry price per record — see note above.
            records.append({"date": d, "symbol": "AAPL", "model": model,
                            "signal": "BUY", "confidence": 0.7, "price": 100.0 + i * 1e-6})
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


def test_score_realized_ic_directional_accuracy_ignores_holds(tmp_path):
    """HOLD signals (score=0) must not count as directional misses."""
    today = date.today()
    cutoff_date = today - timedelta(days=FWD_RET_HORIZON_DAYS + 1)
    records = []
    dates = _business_days_before(cutoff_date, 20)
    for i, day in enumerate(dates):
        d = day.strftime("%Y-%m-%d")
        # 10 correct BUYs, 10 HOLDs
        signal = "BUY" if i < 10 else "HOLD"
        # Distinct entry price per record — see note above.
        records.append({"date": d, "symbol": "AAPL", "model": "daily_predictor",
                        "signal": signal, "confidence": 0.7, "price": 100.0 + i * 1e-6})
    path = _write_history(tmp_path, records)

    def mock_fetch(symbol, start, end):
        idx = pd.bdate_range(start, end)
        closes = [100.0 + i * 0.5 for i in range(len(idx))]
        return pd.DataFrame({"close": closes}, index=idx)

    result = score_realized_ic(path, today.strftime("%Y-%m-%d"), fetch_prices_fn=mock_fetch)
    assert result["daily_predictor"]["directional_accuracy"] == pytest.approx(1.0)
    assert result["daily_predictor"]["n_directional"] == 10


def test_score_realized_ic_handles_missing_file(tmp_path):
    result = score_realized_ic(str(tmp_path / "nonexistent.jsonl"),
                               date.today().strftime("%Y-%m-%d"),
                               fetch_prices_fn=lambda s, st, en: None)
    assert result == {}


# --- Regression tests for #134: weekend/pre-open duplicate records ---------

def test_dedupe_scoreable_collapses_consecutive_same_price_runs():
    """
    Unit test for the helper directly: a Friday close restated on Saturday,
    Sunday, and Monday (the exact scenario from #134 — cron runs daily but
    the market only produced one new close) must collapse to a single
    record, keeping the earliest (real trading-session) date.
    """
    records = [
        {"date": "2026-07-03", "symbol": "AAPL", "model": "m", "price": 308.63,
         "signal": "HOLD", "confidence": 0.5},   # Friday (real close)
        {"date": "2026-07-04", "symbol": "AAPL", "model": "m", "price": 308.63,
         "signal": "HOLD", "confidence": 0.5},   # Saturday (duplicate)
        {"date": "2026-07-05", "symbol": "AAPL", "model": "m", "price": 308.63,
         "signal": "HOLD", "confidence": 0.5},   # Sunday (duplicate)
        {"date": "2026-07-06", "symbol": "AAPL", "model": "m", "price": 308.65,
         "signal": "HOLD", "confidence": 0.5},   # Monday (genuine new close)
    ]
    # Real trading calendar backing the records above: only Fri/Mon are
    # sessions, so Sat/Sun resolve back to Friday and collapse into it.
    idx = pd.to_datetime(["2026-07-03", "2026-07-06"])
    frames = {"AAPL": pd.DataFrame({"close": [308.63, 308.65]}, index=idx)}
    deduped = _dedupe_scoreable(records, frames)
    assert [r["date"] for r in deduped] == ["2026-07-03", "2026-07-06"]


def test_dedupe_scoreable_keeps_distinct_prices_and_symbols():
    """Records with genuinely different prices, or different symbols/models,
    are independent observations and must not be collapsed."""
    records = [
        {"date": "2026-07-01", "symbol": "AAPL", "model": "m", "price": 100.0,
         "signal": "BUY", "confidence": 0.5},
        {"date": "2026-07-02", "symbol": "AAPL", "model": "m", "price": 101.0,
         "signal": "BUY", "confidence": 0.5},
        {"date": "2026-07-01", "symbol": "MSFT", "model": "m", "price": 100.0,
         "signal": "BUY", "confidence": 0.5},
        {"date": "2026-07-01", "symbol": "AAPL", "model": "other_model", "price": 100.0,
         "signal": "BUY", "confidence": 0.5},
    ]
    # Both dates are trading sessions for both symbols, so nothing here
    # should resolve to a shared date.
    idx = pd.to_datetime(["2026-07-01", "2026-07-02"])
    frames = {
        "AAPL": pd.DataFrame({"close": [100.0, 101.0]}, index=idx),
        "MSFT": pd.DataFrame({"close": [100.0, 100.5]}, index=idx),
    }
    deduped = _dedupe_scoreable(records, frames)
    assert len(deduped) == 4


def _fridays_with_weekend_duplicates(cutoff_date, n=20):
    """
    Build `n` distinct "trading day" records, each with a unique price, and
    a version of the same history where every record is additionally
    restated on the following two calendar days with an identical price
    (simulating Saturday/Sunday cron runs that see no new close).

    Base dates are spaced 7 real days apart and offset 3 days behind the
    cutoff so that even the +2-day duplicate of the most recent record is
    still old enough to be scoreable. The anchor is walked back to the
    nearest Friday so the +1/+2 duplicates are always genuine non-trading
    days (Sat/Sun), regardless of what weekday `cutoff_date` itself falls
    on for a given test run.
    """
    anchor = cutoff_date - timedelta(days=3)
    anchor -= timedelta(days=(anchor.weekday() - 4) % 7)

    clean, messy = [], []
    for i in range(n):
        base_date = anchor - timedelta(days=7 * i)
        price = 100.0 + i * 1.0
        confidence = 0.5 + (i % 10) * 0.02
        rec = {
            "date": base_date.strftime("%Y-%m-%d"), "symbol": "AAPL", "model": "daily_predictor",
            "signal": "BUY", "confidence": confidence, "price": price,
        }
        clean.append(rec)
        messy.append(dict(rec))
        for offset in (1, 2):
            messy.append({
                "date": (base_date + timedelta(days=offset)).strftime("%Y-%m-%d"),
                "symbol": "AAPL", "model": "daily_predictor",
                "signal": "BUY", "confidence": confidence, "price": price,
            })
    return clean, messy


def _linear_mock_fetch(symbol, start, end):
    idx = pd.bdate_range(start, end)
    closes = [100.0 + i * 0.3 for i in range(len(idx))]
    return pd.DataFrame({"close": closes}, index=idx)


def test_score_realized_ic_weekend_duplicates_do_not_change_result(tmp_path):
    """
    The central regression case for #134: history padded with weekend/
    pre-open duplicates (a raw record count ~3x the real trading-session
    count, matching the ~30% duplicate rate reported in the issue) must
    score identically to the deduplicated history — the duplicates must
    not be counted as extra independent observations nor shift which
    forward window a prediction is scored against.
    """
    today = date.today()
    cutoff_date = today - timedelta(days=FWD_RET_HORIZON_DAYS + 1)
    clean, messy = _fridays_with_weekend_duplicates(cutoff_date, n=20)

    assert len(messy) == 60  # 20 real records x 3 (Fri/Sat/Sun) — the ~3x
                              # duplication described in the issue.

    (tmp_path / "clean").mkdir()
    (tmp_path / "messy").mkdir()
    clean_path = _write_history(tmp_path / "clean", clean)
    messy_path = _write_history(tmp_path / "messy", messy)

    clean_result = score_realized_ic(clean_path, today.strftime("%Y-%m-%d"),
                                      fetch_prices_fn=_linear_mock_fetch)
    messy_result = score_realized_ic(messy_path, today.strftime("%Y-%m-%d"),
                                      fetch_prices_fn=_linear_mock_fetch)

    assert clean_result["daily_predictor"] is not None
    # Sample size must reflect real trading sessions only, never the
    # inflated raw record count.
    assert clean_result["daily_predictor"]["lookback_n"] == 20
    assert messy_result["daily_predictor"] == clean_result["daily_predictor"]


def test_score_realized_ic_weekend_duplicates_do_not_inflate_sample_past_min_lookback(tmp_path):
    """
    A more extreme duplicate ratio: only 12 real trading sessions exist,
    each tripled by weekend restatement (36 raw records — comfortably
    above min_lookback=20). Before the fix, [-min_lookback:] would happily
    slice 20 rows out of the padded 36 and report a "healthy" lookback_n
    of 20, even though only 12 independent closes exist. After the fix,
    deduplication happens first, so the true sample (12) is correctly
    judged insufficient.
    """
    today = date.today()
    cutoff_date = today - timedelta(days=FWD_RET_HORIZON_DAYS + 1)
    _, messy = _fridays_with_weekend_duplicates(cutoff_date, n=12)
    assert len(messy) == 36

    path = _write_history(tmp_path, messy)
    result = score_realized_ic(path, today.strftime("%Y-%m-%d"),
                               fetch_prices_fn=_linear_mock_fetch)
    assert result["daily_predictor"] is None


def test_score_realized_ic_manual_numerical_sanity_check(tmp_path):
    """
    Directly recompute the expected IC for the deduplicated (clean) history
    using scipy, independent of score_realized_ic's internals, as a
    numerical sanity check on the realized-return math and the final
    Spearman IC.
    """
    today = date.today()
    cutoff_date = today - timedelta(days=FWD_RET_HORIZON_DAYS + 1)
    clean, _ = _fridays_with_weekend_duplicates(cutoff_date, n=20)
    path = _write_history(tmp_path, clean)

    result = score_realized_ic(path, today.strftime("%Y-%m-%d"),
                               fetch_prices_fn=_linear_mock_fetch)
    assert result["daily_predictor"] is not None

    # Reproduce the expected scores/returns by hand from the same records
    # and the same deterministic price function used by the mock fetch.
    full_span_start = min(datetime.strptime(r["date"], "%Y-%m-%d") for r in clean)
    idx = pd.bdate_range(
        full_span_start.strftime("%Y-%m-%d"),
        (today + timedelta(days=FWD_RET_HORIZON_DAYS * 2 + 10)).strftime("%Y-%m-%d"),
    )
    closes = pd.Series([100.0 + i * 0.3 for i in range(len(idx))], index=idx)

    expected_scores, expected_returns = [], []
    for r in sorted(clean, key=lambda r: r["date"]):
        pred_date = datetime.strptime(r["date"], "%Y-%m-%d")
        window = closes.loc[pred_date:]
        realized_price = float(window.iloc[FWD_RET_HORIZON_DAYS])
        realized_return = (realized_price / r["price"]) - 1.0
        expected_scores.append(_signal_to_score(r["signal"], r["confidence"]))
        expected_returns.append(realized_return)

    expected_ic, _ = spearmanr(np.array(expected_scores), np.array(expected_returns))
    assert result["daily_predictor"]["ic"] == pytest.approx(float(expected_ic), abs=1e-9)
    assert result["daily_predictor"]["lookback_n"] == len(expected_scores) == 20


def test_score_realized_ic_weekend_duplicate_anchors_to_correct_session(tmp_path):
    """
    Regression test for the PR #136 review finding: the record kept after
    deduplication must be anchored to the trading session that actually
    produced the close, not the earliest calendar date it happened to be
    restated under. Retaining "Saturday" as the entry date (the naive
    keep-the-earliest-duplicate behavior) makes `price_df.loc[pred_date:]`
    snap forward past the weekend to the following Monday, measuring the
    forward return one session later than intended. Anchoring to the real
    session (Friday, here) fixes the window.
    """
    records = []
    # Two independent Friday sessions a week apart, each restated on the
    # following Saturday and Sunday exactly as the daily cron would.
    sessions = [
        ("2026-07-03", 100.0, 0.9),
        ("2026-07-10", 200.0, 0.2),
    ]
    for friday, price, confidence in sessions:
        f = datetime.strptime(friday, "%Y-%m-%d").date()
        for offset in (1, 2):  # Sat, Sun restatements
            d = (f + timedelta(days=offset)).strftime("%Y-%m-%d")
            records.append({"date": d, "symbol": "AAPL", "model": "m",
                            "signal": "BUY", "confidence": confidence, "price": price})
    path = _write_history(tmp_path, records)

    idx = pd.bdate_range("2026-06-01", "2026-08-01")
    closes = pd.Series(np.arange(len(idx), dtype=float) + 50.0, index=idx)
    closes.loc["2026-07-03"] = 100.0
    closes.loc["2026-07-10"] = 200.0
    price_df = pd.DataFrame({"close": closes})

    def mock_fetch(symbol, start, end):
        return price_df.loc[start:end]

    today = datetime.strptime("2026-08-01", "%Y-%m-%d").date()
    result = score_realized_ic(path, today.strftime("%Y-%m-%d"),
                               fetch_prices_fn=mock_fetch, min_lookback=2)

    assert result["m"] is not None
    # Both weekend restatement runs must collapse to one session each.
    assert result["m"]["lookback_n"] == 2

    correct_returns, buggy_returns = [], []
    for friday, price, _ in sessions:
        f_ts = pd.Timestamp(friday)
        correct_window = closes.loc[f_ts:]
        correct_returns.append(float(correct_window.iloc[FWD_RET_HORIZON_DAYS]) / price - 1.0)

        monday = f_ts + pd.Timedelta(days=3)  # where the pre-fix bug snapped pred_date to
        buggy_window = closes.loc[monday:]
        buggy_returns.append(float(buggy_window.iloc[FWD_RET_HORIZON_DAYS]) / price - 1.0)

    # Sanity check that this scenario actually distinguishes the two
    # anchor points; otherwise the assertion below wouldn't catch a
    # regression back to the buggy behavior.
    assert correct_returns != buggy_returns

    scores = [_signal_to_score("BUY", c) for _, _, c in sessions]
    expected_ic, _ = spearmanr(np.array(scores), np.array(correct_returns))
    assert result["m"]["ic"] == pytest.approx(float(expected_ic), abs=1e-9)
