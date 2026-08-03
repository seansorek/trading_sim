"""
signal_monitor.py — Realized IC scoring and signal drift detection.

score_realized_ic: scores trailing predictions against realized returns.
check_signal_drift: detects if today's predicted-return distribution has shifted.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.stats import spearmanr

from daily_features import FWD_RET_HORIZON_DAYS

logger = logging.getLogger(__name__)

MIN_LOOKBACK = 20
MIN_DRIFT_WINDOW = 10
DRIFT_SIGMA_THRESHOLD = 2.0
DRIFT_ABS_THRESHOLD = 0.002


def _load_history(history_path: str) -> list[dict]:
    path = Path(history_path)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _spearman_ic(scores: np.ndarray, actuals: np.ndarray) -> float:
    if len(scores) < 2 or np.std(scores) < 1e-12:
        return 0.0
    ic, _ = spearmanr(scores, actuals)
    return 0.0 if np.isnan(ic) else float(ic)


def _prefetch_by_symbol(
    records: list[dict], fetch_prices_fn: Callable, fwd_ret_horizon: int
) -> dict[str, Optional["pd.DataFrame"]]:
    """One fetch per symbol, spanning the full date range records need."""
    by_symbol: dict[str, tuple[datetime, datetime]] = {}
    for r in records:
        d = datetime.strptime(r["date"], "%Y-%m-%d")
        lo, hi = by_symbol.get(r["symbol"], (d, d))
        by_symbol[r["symbol"]] = (min(lo, d), max(hi, d))

    frames: dict[str, Optional["pd.DataFrame"]] = {}
    for symbol, (lo, hi) in by_symbol.items():
        # Pad the start back a bit: a record's true trading session (see
        # `_resolve_session_date`) can fall a few calendar days before its
        # own recorded date, and that session needs to be present in the
        # fetched frame to be resolved correctly.
        start = lo - timedelta(days=10)
        end = hi + timedelta(days=fwd_ret_horizon * 2 + 10)
        try:
            frames[symbol] = fetch_prices_fn(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        except Exception:
            frames[symbol] = None
    return frames


def _resolve_session_date(price_df: Optional["pd.DataFrame"], calendar_date) -> Optional["pd.Timestamp"]:
    """
    The most recent trading session at or before `calendar_date`.

    `calendar_date` is passed through unchanged whenever it is itself a
    trading session (the ordinary case). It only gets remapped backward
    when it isn't one — a weekend or holiday the daily cron still ran
    on — in which case the close it observed belongs to the last real
    session before it. Returns None if `price_df` has no session at or
    before `calendar_date` to resolve to.
    """
    if price_df is None or price_df.empty:
        return None
    eligible = price_df.index[price_df.index <= calendar_date]
    if len(eligible) == 0:
        return None
    return eligible[-1]


def _dedupe_scoreable(records: list[dict], frames: dict[str, Optional["pd.DataFrame"]]) -> list[dict]:
    """
    Collapse records down to one observation per real trading session,
    anchored to the session that actually produced the recorded close —
    not the calendar date the cron happened to run on.

    The daily job runs every day (including weekends) before the US open,
    so Saturday/Sunday/Monday-pre-open runs all observe the same last
    close as the preceding Friday and each appends its own record to the
    history file under its own calendar date. Left alone,
    `score_realized_ic` would treat those calendar dates as independent
    observations and slice each one's forward-return window from a
    different start point, since `price_df.loc[pred_date:]` snaps a
    non-trading `pred_date` forward to the *next* bar rather than back to
    the session that actually produced the close.

    Resolving each record to its actual session (`_resolve_session_date`)
    fixes both problems in one step: restatements of the same session
    collapse naturally, because they resolve to the same date, and the
    forward-return window is anchored to the session that really produced
    the entry price instead of snapping forward past it. Unlike collapsing
    on price equality alone, two genuinely distinct sessions that happen
    to close at the same price are never merged, since they resolve to
    different session dates. If a record's price data isn't available to
    resolve against, it's kept under its original date rather than
    dropped, so a fetch failure degrades to the pre-dedup behavior
    instead of silently discarding history.
    """
    resolved: list[dict] = []
    for r in records:
        calendar_date = datetime.strptime(r["date"], "%Y-%m-%d")
        session_date = _resolve_session_date(frames.get(r["symbol"]), calendar_date)
        if session_date is not None:
            r = {**r, "date": session_date.strftime("%Y-%m-%d")}
        resolved.append(r)

    resolved.sort(key=lambda r: r["date"])
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict] = []
    for r in resolved:
        key = (r["model"], r["symbol"], r["date"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def _signal_to_score(signal: str, confidence: float) -> float:
    """Map signal + confidence to a directional score for IC computation."""
    if signal == "BUY":
        return float(confidence)
    if signal == "SELL":
        return -float(confidence)
    return 0.0


def score_realized_ic(
    history_path: str,
    today_date: str,
    fetch_prices_fn: Callable[[str, str, str], Optional[object]],
    min_lookback: int = MIN_LOOKBACK,
    fwd_ret_horizon: int = FWD_RET_HORIZON_DAYS,
) -> dict[str, Optional[dict]]:
    """
    Score trailing predictions against realized returns.

    fetch_prices_fn(symbol, start_date, end_date) must return a DataFrame
    with a 'close' column (or None on failure).

    Returns {model_key: {ic, directional_accuracy, lookback_n, mean_pred, std_pred} | None}.
    Returns {} if history file does not exist.
    """
    records = _load_history(history_path)
    if not records:
        return {}

    today = datetime.strptime(today_date, "%Y-%m-%d").date()
    cutoff = today - timedelta(days=fwd_ret_horizon + 1)

    scoreable = [
        r for r in records
        if r.get("price") is not None
        and "signal" in r and "confidence" in r and "date" in r
        and datetime.strptime(r["date"], "%Y-%m-%d").date() <= cutoff
    ]

    # Fetched once per symbol up front: needed both to resolve each
    # record's real trading session below and to score the forward
    # return later, and the range already covers every scoreable record.
    frames = _prefetch_by_symbol(scoreable, fetch_prices_fn, fwd_ret_horizon)

    # Collapse weekend/pre-open cron runs that restate the same close as a
    # prior record, anchored to the session that produced it, so each
    # realized bar is scored exactly once against the right window (#134).
    scoreable = _dedupe_scoreable(scoreable, frames)

    # Group by model; take the most recent min_lookback entries
    by_model: dict[str, list[dict]] = {}
    for r in scoreable:
        by_model.setdefault(r["model"], []).append(r)

    truncated_by_model: dict[str, list[dict]] = {
        model: sorted(model_records, key=lambda r: r["date"])[-min_lookback:]
        for model, model_records in by_model.items()
    }

    results: dict[str, Optional[dict]] = {}
    for model, sorted_records in truncated_by_model.items():
        if len(sorted_records) < min_lookback:
            logger.info(
                "score_realized_ic: insufficient history for %s (%d < %d)",
                model, len(sorted_records), min_lookback,
            )
            results[model] = None
            continue

        scores, actuals = [], []
        for r in sorted_records:
            symbol = r["symbol"]
            pred_date = datetime.strptime(r["date"], "%Y-%m-%d")
            price_df = frames.get(symbol)
            if price_df is None:
                continue
            window = price_df.loc[pred_date:]
            if len(window) < fwd_ret_horizon + 1:
                continue
            entry_price = float(r["price"])
            if entry_price <= 0:
                continue
            realized_price = float(window["close"].iloc[fwd_ret_horizon])
            realized_return = (realized_price / entry_price) - 1.0
            if abs(realized_return) < 1e-5:
                continue
            scores.append(_signal_to_score(r["signal"], float(r["confidence"])))
            actuals.append(realized_return)

        if len(scores) < min_lookback // 2:
            logger.info(
                "score_realized_ic: too few valid scored rows for %s (%d)", model, len(scores)
            )
            results[model] = None
            continue

        scores_arr = np.array(scores)
        actuals_arr = np.array(actuals)
        directional = scores_arr != 0.0
        n_dir = int(directional.sum())
        results[model] = {
            "ic": _spearman_ic(scores_arr, actuals_arr),
            "directional_accuracy": (
                float(np.mean(np.sign(scores_arr[directional]) == np.sign(actuals_arr[directional])))
                if n_dir else 0.0
            ),
            "n_directional": n_dir,
            "lookback_n": len(scores),
            "mean_pred": float(scores_arr.mean()),
            "std_pred": float(scores_arr.std()),
        }

    return results


def check_signal_drift(
    history_path: str,
    today_date: str,
    today_mean_scores: dict[str, float],
    window_days: int = 30,
    sigma_threshold: float = DRIFT_SIGMA_THRESHOLD,
    abs_threshold: float = DRIFT_ABS_THRESHOLD,
) -> dict[str, bool]:
    """
    Check if today's predicted-return distribution has drifted vs trailing window.

    today_mean_scores: {model: mean score across all symbols today}
    Returns {model: True} only when shift > sigma_threshold σ AND > abs_threshold
    AND yesterday also showed the same shift (two-consecutive-day guard).
    """
    records = _load_history(history_path)
    today = datetime.strptime(today_date, "%Y-%m-%d").date()
    yesterday = today - timedelta(days=1)
    window_start = today - timedelta(days=window_days)

    # Build daily mean scores from history: {date -> {model -> mean_score}}
    raw: dict = {}
    for r in records:
        try:
            d = datetime.strptime(r["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        if d < window_start or d >= today:
            continue
        model = r.get("model")
        if not model:
            continue
        score = _signal_to_score(r.get("signal", "HOLD"), float(r.get("confidence", 0.0)))
        raw.setdefault(d, {}).setdefault(model, []).append(score)

    daily_means: dict = {d: {m: float(np.mean(s)) for m, s in ms.items()}
                         for d, ms in raw.items()}

    warnings: dict[str, bool] = {}
    all_models = set(today_mean_scores.keys())

    for model in all_models:
        today_score = today_mean_scores.get(model)
        if today_score is None:
            warnings[model] = False
            continue

        # Baseline window: exclude today and yesterday for a clean reference
        baseline_scores = [
            daily_means[d][model]
            for d in daily_means
            if model in daily_means[d] and d < yesterday
        ]
        if len(baseline_scores) < MIN_DRIFT_WINDOW:
            warnings[model] = False
            continue

        baseline = np.array(baseline_scores)
        mu = float(baseline.mean())
        sigma = float(baseline.std())
        if sigma == 0.0:
            warnings[model] = False
            continue

        def _is_shifted(score: float) -> bool:
            z = (score - mu) / sigma
            return abs(z) > sigma_threshold and abs(score - mu) > abs_threshold

        today_shifted = _is_shifted(today_score)
        yesterday_score = daily_means.get(yesterday, {}).get(model)
        yesterday_shifted = yesterday_score is not None and _is_shifted(yesterday_score)

        warnings[model] = today_shifted and yesterday_shifted

    return warnings
