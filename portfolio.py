"""
portfolio.py — Today's target book from one cross-section of forecasts.

The portfolio analogue of `ml_strategies.compute_predictor_signal`: ONE
implementation of "what do I hold", shared by the panel research path
(`run_panel.py`) and the live daily job (`predict_next_day_lite.py`). The
weighting itself is `panel_backtester.rank_to_weights` — imported, not
reimplemented — so the book that gets published can never drift from the book
that was measured.

Two facts from `models/README.md` shape this module:

- The edge is cross-sectional, not per-name. IC is positive in 5/5 yearly
  walk-forward folds; the per-symbol timing path loses to buy-and-hold. So the
  book is the headline output and the per-symbol signals are the sidecar.
- Rebalancing daily is not viable. At `rebalance_days=1` turnover is ~0.85/day
  and cost removes 1.4-2.7 Sharpe every year. A job that runs daily must
  therefore *hold* on most days, which is what `is_rebalance_due` enforces.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from panel_backtester import rank_to_weights, sector_neutralize

logger = logging.getLogger(__name__)

# Beta-neutral leg sizing buys zero *beta* by giving up zero *net notional*: when
# the long leg's mean beta is k times the short leg's, the book holds k times more
# short notional. Measured per date over 1200 days at decile 0.2, net exposure ran
# mean -0.22, p1 -0.62, min -0.64, and |net| > 0.5 on 5.7% of dates.
#
# Past this the book is a large dollar-directional bet that "market neutral" hides,
# so it gets a warning. Deliberately NOT a clamp: clamping would make the live book
# differ from the one run_panel.py measured, which is the one property this module
# exists to guarantee. Surface it, let the reader decide.
NET_EXPOSURE_WARN = 0.5


@dataclass
class Book:
    """A target book: symbol -> weight as a fraction of equity.

    Longs are positive, shorts negative. Untraded names are absent, not 0.0 —
    a JSON record of 159 zeros helps nobody.
    """

    as_of: str                       # date the weights were formed
    weights: dict[str, float] = field(default_factory=dict)
    rebalanced: bool = True          # False when re-published from an earlier date
    diagnostics: dict = field(default_factory=dict)

    @property
    def longs(self) -> list[tuple[str, float]]:
        """Biggest weight first, ties broken by symbol so output is stable.

        Equal-weighted legs are all ties, so in practice this is alphabetical —
        which is what makes two consecutive runs diffable.
        """
        return sorted(
            ((s, w) for s, w in self.weights.items() if w > 0),
            key=lambda x: (-x[1], x[0]),
        )

    @property
    def shorts(self) -> list[tuple[str, float]]:
        return sorted(
            ((s, w) for s, w in self.weights.items() if w < 0),
            key=lambda x: (x[1], x[0]),
        )

    @property
    def is_flat(self) -> bool:
        return not self.weights


def build_book(
    scores: dict[str, float],
    as_of: str,
    decile: float,
    gross_exposure: float,
    min_names: int,
    sector_of: dict[str, str] | None = None,
    betas: dict[str, float] | None = None,
) -> Book:
    """Rank one date's cross-section into a beta-neutral long/short book.

    `scores` is symbol -> forecast return. Symbols with a non-finite score are
    dropped rather than ranked: a NaN forecast means "not in the cross-section
    today", the same convention `panel_data` uses.

    `betas` is symbol -> trailing beta vs SPY. Supplying it sizes the two legs
    so their beta exposures cancel; omitting it falls back to a dollar-neutral
    book, which measured +0.19 realized beta on this universe and is NOT market
    neutral. Live should always pass betas — see rank_to_weights' docstring.
    """
    clean = {s: float(v) for s, v in scores.items() if v is not None and np.isfinite(v)}
    dropped = len(scores) - len(clean)
    if dropped:
        logger.info("Dropped %d/%d symbols with no usable score", dropped, len(scores))

    if not clean:
        return Book(as_of=as_of, diagnostics={"reason": "no scores", "n_ranked": 0})

    row = pd.Series(clean, dtype=float)

    n_sector_neutralized = 0
    if sector_of:
        # A one-row frame so the panel's own implementation applies, including
        # its MIN_SECTOR_NAMES floor. Reusing it is the point: a second
        # demeaning rule here is a second thing to keep in sync.
        frame = sector_neutralize(row.to_frame().T, sector_of)
        row = frame.iloc[0]
        n_sector_neutralized = sum(1 for s in row.index if s in sector_of)

    beta_row = None
    if betas:
        beta_row = pd.Series(
            {s: betas.get(s, np.nan) for s in row.index}, dtype=float
        )

    weights = rank_to_weights(
        row,
        decile=decile,
        gross_exposure=gross_exposure,
        min_names=min_names,
        beta_row=beta_row,
    )

    held = {s: float(w) for s, w in weights.items() if w != 0.0}
    diagnostics = {
        "n_ranked": int(len(row)),
        "n_held": len(held),
        "n_sector_neutralized": n_sector_neutralized,
        "gross_exposure": float(sum(abs(w) for w in held.values())),
        "net_exposure": float(sum(held.values())),
        "beta_neutral": beta_row is not None,
    }
    if beta_row is not None and held:
        n_covered = sum(1 for s in held if np.isfinite(beta_row.get(s, np.nan)))
        beta_coverage = n_covered / len(held)
        diagnostics["beta_coverage"] = float(beta_coverage)
        if beta_coverage >= 1.0:
            diagnostics["ex_ante_beta"] = float(
                sum(w * beta_row.get(s, 0.0) for s, w in held.items())
            )
        else:
            # Summing only the covered names (the old behavior) understates
            # exposure and can read as beta-neutral (e.g. exactly 0.0) on a
            # book that is anything but — see #149. A diagnostic computed
            # over an incomplete set is worse than no diagnostic, so omit
            # ex_ante_beta entirely and surface the coverage gap instead.
            logger.warning(
                "Book beta coverage %.0f%% (%d/%d held names) — ex_ante_beta "
                "not reported because it would be computed over an "
                "incomplete set of the book's holdings",
                100 * beta_coverage, n_covered, len(held),
            )
    if abs(diagnostics["net_exposure"]) > NET_EXPOSURE_WARN:
        diagnostics["net_exposure_warning"] = True
        logger.warning(
            "Book net exposure %+.2f exceeds %.2f — beta-neutral leg sizing has "
            "made this a large dollar-directional position (leg betas diverged). "
            "Zero ex-ante beta does NOT mean zero directional risk here.",
            diagnostics["net_exposure"], NET_EXPOSURE_WARN,
        )
    if not held:
        diagnostics["reason"] = (
            f"cross-section of {len(row)} names is below min_names={min_names}"
            if len(row) < min_names
            else "ranking produced no tradeable legs"
        )

    return Book(as_of=as_of, weights=held, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Rebalance cadence — the daily job must hold, not re-rank, on most days
# ---------------------------------------------------------------------------

def is_rebalance_due(last_as_of: str | None, today: str, rebalance_days: int) -> bool:
    """True when a fresh book is due, i.e. no prior book or the window elapsed.

    ponytail: counts weekdays via np.busday_count, so market holidays make this
    fire up to ~2 days early over a long window. The backtest counts bars. The
    upgrade path is a holiday calendar, and it is not worth one until the
    cadence is tighter than 10 days.
    """
    if rebalance_days <= 1 or not last_as_of:
        return True
    try:
        elapsed = int(np.busday_count(
            np.datetime64(last_as_of, "D"), np.datetime64(today, "D")
        ))
    except ValueError:
        logger.warning("Unparseable last book date %r — rebalancing", last_as_of)
        return True
    return elapsed >= rebalance_days


def load_last_book(path: str) -> Book | None:
    """Read the most recent book from the append-only JSONL log.

    Returns None when the file is absent or holds no readable record — a fresh
    checkout should rebalance, not crash.
    """
    if not path or not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", path)
    if not last or "as_of" not in last:
        return None
    return Book(
        as_of=last["as_of"],
        weights={s: float(w) for s, w in last.get("weights", {}).items()},
        rebalanced=False,
        diagnostics=last.get("diagnostics", {}),
    )


def append_book(path: str, book: Book, prediction_date: str) -> None:
    """Append the book to a JSONL log meant to be committed back to the repo.

    Same durability rationale as predictions/history.jsonl: CI runners are
    ephemeral, so the repo is the only place the book survives — and without a
    surviving book there is nothing to hold between rebalances.
    """
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    record = {
        "date": prediction_date,
        "as_of": book.as_of,
        "rebalanced": book.rebalanced,
        "weights": book.weights,
        "diagnostics": book.diagnostics,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def format_book(book: Book, prices: dict[str, float] | None = None) -> str:
    """Plain-text book summary for stdout and the Discord payload."""
    if book.is_flat:
        return f"FLAT — {book.diagnostics.get('reason', 'no position')}"

    prices = prices or {}

    def line(sym: str, w: float) -> str:
        px = prices.get(sym)
        px_s = f" ${px:,.2f}" if px else ""
        return f"  {sym:<6}{w:>+7.2%}{px_s}"

    d = book.diagnostics
    head = (
        f"{'REBALANCE' if book.rebalanced else 'HOLD'} — book as of {book.as_of}, "
        f"{len(book.longs)} long / {len(book.shorts)} short "
        f"of {d.get('n_ranked', 0)} ranked"
    )
    stats = (
        f"  gross {d.get('gross_exposure', 0):.2f}  net {d.get('net_exposure', 0):+.2f}"
        + (f"  ex-ante beta {d['ex_ante_beta']:+.3f}" if "ex_ante_beta" in d else "")
        + ("  [!] net exposure beyond backtested range"
           if d.get("net_exposure_warning") else "")
    )
    parts = [head, stats, "LONG:"]
    parts += [line(s, w) for s, w in book.longs]
    parts += ["SHORT:"]
    parts += [line(s, w) for s, w in book.shorts]
    return "\n".join(parts)
