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


@dataclass
class PanelConfig:
    decile: float = 0.1
    rebalance_days: int = 1
    gross_exposure: float = 1.0
    cost_bps: float = 5.0              # one-way, on turnover notional (assumption)
    borrow_bps_annual: float = 50.0    # on short notional
    min_names: int = 20
    start_cash: float = 100_000.0


@dataclass
class PanelResult:
    equity: pd.Series
    book_ret: pd.Series
    gross_ret: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    diagnostics: dict = field(default_factory=dict)


def run_panel(pred: pd.DataFrame, ret: pd.DataFrame, cfg: PanelConfig) -> PanelResult:
    """Run the cross-sectional book over the panel.

    LAG CONVENTION: weights on date t come from predictions computed on close[t],
    and earn ret[t], which panel_data defines as close[t] -> close[t+1]. The
    weight and the return it earns therefore share index t. Shifting either side
    reintroduces the #106 look-ahead.
    """
    ret = ret.reindex(index=pred.index, columns=pred.columns)
    dates = pred.index

    weight_rows: list[pd.Series] = []
    turnover_vals: list[float] = []
    prev_w = pd.Series(0.0, index=pred.columns)
    n_flat = 0

    for i, date in enumerate(dates):
        if i % cfg.rebalance_days == 0:
            w = rank_to_weights(
                pred.loc[date], cfg.decile, cfg.gross_exposure, cfg.min_names
            )
        else:
            w = prev_w.copy()
        if (w == 0.0).all():
            n_flat += 1
        weight_rows.append(w)
        turnover_vals.append(float((w - prev_w).abs().sum()))
        prev_w = w

    weights = pd.DataFrame(weight_rows, index=dates)
    turnover = pd.Series(turnover_vals, index=dates)

    cost = turnover * (cfg.cost_bps / 1e4)
    short_notional = weights.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_notional * (cfg.borrow_bps_annual / 1e4 / 252.0)

    # skipna: a name holding weight whose next-day bar is missing earns 0 rather
    # than poisoning the whole day's book return with NaN.
    gross_ret = (weights * ret).sum(axis=1, skipna=True)
    book_ret = gross_ret - cost - borrow
    equity = cfg.start_cash * (1.0 + book_ret).cumprod()

    diagnostics = {
        "n_dates": int(len(dates)),
        "n_flat_days": int(n_flat),
        "mean_turnover": float(turnover.mean()),
        "mean_cost": float(cost.mean()),
        "mean_borrow_cost": float(borrow.mean()),
        "total_cost_drag": float(cost.sum() + borrow.sum()),
        "mean_gross_exposure": float(weights.abs().sum(axis=1).mean()),
        "mean_net_exposure": float(weights.sum(axis=1).mean()),
    }

    return PanelResult(
        equity=equity,
        book_ret=book_ret,
        gross_ret=gross_ret,
        weights=weights,
        turnover=turnover,
        diagnostics=diagnostics,
    )
