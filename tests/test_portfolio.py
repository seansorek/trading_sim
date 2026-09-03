"""Tests for portfolio.py — the live target book.

The point of these is that the published book equals the backtested book. The
weighting itself is panel_backtester's and is tested in test_panel_backtester.py;
what is new here is the dict->Series->dict plumbing, the sector/beta wiring, and
the rebalance cadence that stops the daily job from trading daily.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from panel_backtester import rank_to_weights
from predict_next_day_lite import build_portfolio, send_discord
from portfolio import (
    Book,
    append_book,
    build_book,
    format_book,
    is_rebalance_due,
    load_last_book,
)


def _scores(n: int = 40) -> dict[str, float]:
    """n symbols with strictly increasing forecasts, so the ranking is known."""
    return {f"S{i:02d}": i / 1000.0 for i in range(n)}


# --- construction ----------------------------------------------------------

def test_book_longs_top_shorts_bottom():
    book = build_book(_scores(40), "2026-07-30", decile=0.1, gross_exposure=1.0,
                      min_names=20)
    assert {s for s, _ in book.longs} == {"S36", "S37", "S38", "S39"}
    assert {s for s, _ in book.shorts} == {"S00", "S01", "S02", "S03"}
    assert all(w > 0 for _, w in book.longs)
    assert all(w < 0 for _, w in book.shorts)


def test_book_is_dollar_neutral_without_betas():
    book = build_book(_scores(40), "2026-07-30", decile=0.1, gross_exposure=1.0,
                      min_names=20)
    assert sum(book.weights.values()) == pytest.approx(0.0, abs=1e-12)
    assert sum(abs(w) for w in book.weights.values()) == pytest.approx(1.0)


def test_betas_size_legs_so_beta_cancels():
    """With a high-beta long leg the book must hold LESS long notional."""
    scores = _scores(40)
    betas = {s: (1.6 if s >= "S36" else 0.8) for s in scores}
    book = build_book(scores, "2026-07-30", decile=0.1, gross_exposure=1.0,
                      min_names=20, betas=betas)

    long_notional = sum(w for w in book.weights.values() if w > 0)
    short_notional = -sum(w for w in book.weights.values() if w < 0)
    assert long_notional < short_notional            # high beta -> smaller leg
    assert book.diagnostics["ex_ante_beta"] == pytest.approx(0.0, abs=1e-9)
    assert book.diagnostics["beta_neutral"] is True


@pytest.mark.parametrize("with_betas", [False, True])
def test_matches_rank_to_weights_exactly(with_betas):
    """The live book IS panel_backtester's book — not a parallel implementation.

    Verified against the real thing too: on 2026-07-30 the live path and
    build_panels -> sector_neutralize -> rank_to_weights produced identical
    62-name books (net -0.716 both, betas equal to machine precision).
    """
    scores = _scores(40)
    betas = {s: 0.5 + (i % 7) / 4 for i, s in enumerate(scores)} if with_betas else None
    book = build_book(scores, "2026-07-30", decile=0.1, gross_exposure=1.0,
                      min_names=20, betas=betas)
    expected = rank_to_weights(
        pd.Series(scores, dtype=float), decile=0.1, gross_exposure=1.0, min_names=20,
        beta_row=None if betas is None else pd.Series(betas, dtype=float),
    )
    for symbol, weight in book.weights.items():
        assert weight == pytest.approx(expected[symbol])
    assert len(book.weights) == int((expected != 0).sum())


def test_extreme_leg_beta_divergence_is_flagged_not_clamped():
    """A book beyond the backtested net-exposure range warns but is NOT altered.

    Clamping would make the live book differ from the one run_panel measured,
    which defeats the point of sharing rank_to_weights.
    """
    scores = _scores(40)
    # Long leg 6x the short leg's beta -> short notional ~6x the long's.
    betas = {s: (3.0 if s >= "S36" else 0.5) for s in scores}
    book = build_book(scores, "2026-07-30", decile=0.1, gross_exposure=1.0,
                      min_names=20, betas=betas)

    assert book.diagnostics["net_exposure_warning"] is True
    assert abs(book.diagnostics["net_exposure"]) > 0.5
    assert book.diagnostics["gross_exposure"] == pytest.approx(1.0)   # unclamped
    assert book.diagnostics["ex_ante_beta"] == pytest.approx(0.0, abs=1e-9)
    assert "net exposure beyond backtested range" in format_book(book)


def test_balanced_betas_do_not_warn():
    scores = _scores(40)
    book = build_book(scores, "2026-07-30", decile=0.1, gross_exposure=1.0,
                      min_names=20, betas={s: 1.0 for s in scores})
    assert "net_exposure_warning" not in book.diagnostics


def test_ex_ante_beta_not_reported_on_partial_beta_coverage():
    """Regression test for #149.

    Before the fix, `ex_ante_beta` was summed only over held names that had
    a finite beta, so a book with mostly-missing betas could read as
    "beta-neutral" (exactly 0.0) while its real beta was large and
    directional. build_book must now report `beta_coverage` and omit
    `ex_ante_beta` whenever coverage is partial, rather than publish a
    number computed over an incomplete set.
    """
    scores = _scores(40)
    # Betas for only 2 of the 8 held names (decile 0.1 -> 4 longs + 4 shorts).
    betas = {"S36": 1.6, "S00": 0.7}
    book = build_book(scores, "2026-07-30", decile=0.1, gross_exposure=1.0,
                      min_names=20, betas=betas)

    assert "ex_ante_beta" not in book.diagnostics
    assert book.diagnostics["beta_coverage"] == pytest.approx(2 / 8)


def test_ex_ante_beta_reported_when_coverage_is_complete():
    scores = _scores(40)
    betas = {s: 1.0 for s in scores}
    book = build_book(scores, "2026-07-30", decile=0.1, gross_exposure=1.0,
                      min_names=20, betas=betas)
    assert book.diagnostics["beta_coverage"] == pytest.approx(1.0)
    assert "ex_ante_beta" in book.diagnostics


def test_sector_neutralization_changes_the_ranking():
    """A sector-wide level must not decide the legs.

    Ten Energy names carry a large positive offset. Un-neutralized, the long
    leg is all Energy; demeaned within sector, Energy competes with itself.
    """
    scores = {f"T{i:02d}": i / 1000.0 for i in range(15)}
    scores.update({f"E{i:02d}": 0.5 + i / 1000.0 for i in range(15)})
    sector_of = {s: ("Energy" if s.startswith("E") else "Tech") for s in scores}

    raw = build_book(scores, "2026-07-30", decile=0.2, gross_exposure=1.0,
                     min_names=20)
    neutral = build_book(scores, "2026-07-30", decile=0.2, gross_exposure=1.0,
                         min_names=20, sector_of=sector_of)

    assert all(s.startswith("E") for s, _ in raw.longs)
    assert {s for s, _ in neutral.longs} != {s for s, _ in raw.longs}
    # Both sectors are represented on both sides once the level is removed.
    assert len({s[0] for s, _ in neutral.longs}) == 2
    assert neutral.diagnostics["n_sector_neutralized"] == 30


# --- refusal to trade a thin or broken cross-section -----------------------

def test_flat_below_min_names():
    book = build_book(_scores(10), "2026-07-30", decile=0.1, gross_exposure=1.0,
                      min_names=20)
    assert book.is_flat
    assert "min_names" in book.diagnostics["reason"]


def test_non_finite_scores_are_dropped_not_ranked():
    scores = _scores(40)
    scores["S39"] = float("nan")     # would otherwise be the top long
    scores["S00"] = float("inf")
    book = build_book(scores, "2026-07-30", decile=0.1, gross_exposure=1.0,
                      min_names=20)
    assert "S39" not in book.weights
    assert "S00" not in book.weights
    assert book.diagnostics["n_ranked"] == 38


def test_empty_scores_is_flat_not_an_exception():
    book = build_book({}, "2026-07-30", decile=0.1, gross_exposure=1.0, min_names=20)
    assert book.is_flat


# --- rebalance cadence -----------------------------------------------------

def test_rebalance_due_when_no_prior_book():
    assert is_rebalance_due(None, "2026-07-30", 10)


def test_holds_inside_the_window_and_rebalances_after():
    # 2026-07-16 -> 2026-07-30 is 10 business days.
    assert not is_rebalance_due("2026-07-16", "2026-07-29", 10)
    assert is_rebalance_due("2026-07-16", "2026-07-30", 10)


def test_daily_rebalance_always_due():
    assert is_rebalance_due("2026-07-30", "2026-07-30", 1)


def test_unparseable_date_rebalances_rather_than_crashing():
    assert is_rebalance_due("not-a-date", "2026-07-30", 10)


# --- durable book log ------------------------------------------------------

def test_book_roundtrips_through_the_log(tmp_path):
    path = str(tmp_path / "portfolio.jsonl")
    book = build_book(_scores(40), "2026-07-30", decile=0.1, gross_exposure=1.0,
                      min_names=20)
    append_book(path, book, "2026-07-30")

    loaded = load_last_book(path)
    assert loaded.as_of == "2026-07-30"
    assert loaded.weights == book.weights
    # A book read back from disk is one being held, never a fresh rebalance.
    assert loaded.rebalanced is False


def test_load_returns_the_last_record(tmp_path):
    path = str(tmp_path / "portfolio.jsonl")
    append_book(path, Book(as_of="2026-07-01", weights={"A": 0.5}), "2026-07-01")
    append_book(path, Book(as_of="2026-07-15", weights={"B": 0.5}), "2026-07-15")
    assert load_last_book(path).as_of == "2026-07-15"


def test_missing_or_corrupt_log_returns_none(tmp_path):
    assert load_last_book(str(tmp_path / "nope.jsonl")) is None
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json\n")
    assert load_last_book(str(path)) is None


def test_log_line_is_valid_json_with_the_weights(tmp_path):
    path = str(tmp_path / "portfolio.jsonl")
    append_book(path, Book(as_of="2026-07-30", weights={"AAPL": 0.25}), "2026-07-30")
    with open(path) as f:
        record = json.loads(f.readline())
    assert record["date"] == "2026-07-30"
    assert record["weights"] == {"AAPL": 0.25}


# --- formatting ------------------------------------------------------------

def _fake_cfg(universe, sectors=None, models=("daily_predictor",),
              decile=0.2, rebalance_days=10):
    """Stand-in for AppConfig with only the fields build_portfolio reads."""
    panel = SimpleNamespace(
        universe=list(universe), decile=decile, rebalance_days=rebalance_days,
        gross_exposure=1.0, min_names=20,
        sector_of=lambda: dict(sectors or {}),
    )
    return SimpleNamespace(
        panel=panel, prediction=SimpleNamespace(models=list(models))
    )


def _fake_predictions(symbols, betas=True):
    return [
        {
            "symbol": s,
            "price": 100.0,
            "beta": 1.0 if betas else None,
            "predictions": {"daily_predictor": {
                "signal": "HOLD", "confidence": 0.5, "predicted_return": i / 1000.0,
            }},
        }
        for i, s in enumerate(symbols)
    ]


# --- build_portfolio: the live wiring --------------------------------------

def test_watchlist_etfs_never_enter_the_cross_section(tmp_path):
    """The regression this whole split exists to prevent.

    SPY is a basket of the ranked names, so ranking a stock against it is not a
    cross-sectional bet. It reaches predict_symbol (features need it) but must
    not reach the book.
    """
    universe = [f"S{i:02d}" for i in range(40)]
    preds = _fake_predictions([*universe, "SPY", "TQQQ"])
    book = build_portfolio(
        preds, universe, _fake_cfg(universe), "2026-07-30",
        book_path=str(tmp_path / "p.jsonl"),
    )
    assert "SPY" not in book.weights and "TQQQ" not in book.weights
    assert book.diagnostics["n_ranked"] == 40


def test_symbols_that_errored_are_skipped(tmp_path):
    universe = [f"S{i:02d}" for i in range(40)]
    preds = _fake_predictions(universe)
    preds[0] = {"symbol": "S00", "error": "Data fetch failed"}
    preds[1]["predictions"] = {}          # model failed for this name only
    book = build_portfolio(
        preds, universe, _fake_cfg(universe), "2026-07-30",
        book_path=str(tmp_path / "p.jsonl"),
    )
    assert book.diagnostics["n_ranked"] == 38
    assert "S00" not in book.weights and "S01" not in book.weights


def test_holds_the_stored_book_inside_the_rebalance_window(tmp_path):
    path = str(tmp_path / "p.jsonl")
    universe = [f"S{i:02d}" for i in range(40)]
    cfg = _fake_cfg(universe)

    first = build_portfolio(_fake_predictions(universe), universe, cfg,
                            "2026-07-16", book_path=path)
    append_book(path, first, "2026-07-16")
    assert first.rebalanced

    # Same day next run, and a run inside the window: hold, don't re-rank.
    held = build_portfolio(_fake_predictions(universe), universe, cfg,
                           "2026-07-29", book_path=path)
    assert held.rebalanced is False
    assert held.weights == first.weights

    fresh = build_portfolio(_fake_predictions(universe), universe, cfg,
                            "2026-07-30", book_path=path)
    assert fresh.rebalanced is True


def test_no_book_when_ranking_model_is_not_live(tmp_path):
    universe = [f"S{i:02d}" for i in range(40)]
    cfg = _fake_cfg(universe, models=("daily_logistic",))
    assert build_portfolio(_fake_predictions(universe), universe, cfg,
                           "2026-07-30", book_path=str(tmp_path / "p.jsonl")) is None


def test_no_book_when_portfolio_disabled(tmp_path):
    assert build_portfolio([], [], _fake_cfg([]), "2026-07-30", book_path="") is None


# --- Discord --------------------------------------------------------------

def test_discord_leads_with_the_portfolio_embed():
    book = build_book(_scores(40), "2026-07-30", decile=0.1, gross_exposure=1.0,
                      min_names=20)
    sent = {}

    class _Resp:
        status_code = 204

    def _post(url, json=None, timeout=None):
        sent.setdefault("payloads", []).append(json)
        return _Resp()

    with mock.patch.dict(sys.modules, {"requests": SimpleNamespace(post=_post)}):
        ok = send_discord(
            [{"symbol": "AAPL", "price": 1.0,
              "predictions": {"daily_logistic": {"signal": "BUY", "confidence": 0.9}}}],
            "https://example.invalid/webhook",
            book=book,
        )

    assert ok
    first_embed = sent["payloads"][0]["embeds"][0]
    assert "Portfolio" in first_embed["title"]
    assert "REBALANCE" in first_embed["title"]
    assert "S39" in first_embed["description"]


def test_discord_marks_a_held_book():
    book = Book(as_of="2026-07-16", weights={"AAPL": 0.5, "MSFT": -0.5},
                rebalanced=False, diagnostics={"n_ranked": 40})
    sent = {}

    class _Resp:
        status_code = 204

    with mock.patch.dict(sys.modules, {"requests": SimpleNamespace(
        post=lambda url, json=None, timeout=None: (
            sent.setdefault("p", []).append(json), _Resp())[1]
    )}):
        send_discord([], "https://example.invalid/webhook", book=book)

    title = sent["p"][0]["embeds"][0]["title"]
    assert "HOLD" in title and "2026-07-16" in title


def test_format_book_shows_both_legs_and_flat_reason():
    book = build_book(_scores(40), "2026-07-30", decile=0.1, gross_exposure=1.0,
                      min_names=20)
    text = format_book(book, prices={"S39": 100.0})
    assert "REBALANCE" in text and "LONG:" in text and "SHORT:" in text
    assert "$100.00" in text

    flat = build_book({}, "2026-07-30", decile=0.1, gross_exposure=1.0, min_names=20)
    assert format_book(flat).startswith("FLAT")
