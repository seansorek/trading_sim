"""
panel_backtester.py — Weight-based cross-sectional panel engine.

Research instrument, deliberately NOT a deployability simulation: it models no
share granularity, assumes fills at the close, and ignores per-name market
impact. See docs/superpowers/specs/2026-07-15-step3-panel-portfolio-design.md.

Deliberately omits Backtester's stop_loss_pct / take_profit_pct /
daily_loss_limit_pct / forced-exit cooldown. Those are per-name path-dependent
rules, and a stop-loss that exits one leg breaks the dollar- and beta-neutrality
the book exists to test.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rank_to_weights(
    pred_row: pd.Series,
    decile: float,
    gross_exposure: float,
    min_names: int,
) -> pd.Series:
    """Equal-weight long the top `decile` / short the bottom `decile` of one date.

    Ranking is location-invariant, so this neutralizes market-wide moves in the
    forecast for free: adding a constant to every prediction leaves the ranking
    unchanged. Sector neutrality is NOT provided (deferred increment).

    Returns weights indexed like pred_row, 0.0 for untraded names. Dollar-neutral:
    long notional == short notional == gross_exposure / 2.
    """
    weights = pd.Series(0.0, index=pred_row.index)
    valid = pred_row.dropna()
    if len(valid) < min_names:
        return weights

    k = int(len(valid) * decile)
    k = min(k, len(valid) // 2)   # legs must never overlap
    if k < 1:
        return weights

    ranked = valid.sort_values()
    shorts = ranked.index[:k]
    longs = ranked.index[-k:]

    leg = gross_exposure / 2.0
    weights.loc[longs] = leg / k
    weights.loc[shorts] = -leg / k
    return weights
