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


# Below this, a leg's mean beta is too close to zero to divide by — the scaling
# needed to equalize exposures explodes. The beta-neutral solve is then not
# constructible, so the book stays flat that date (see rank_to_weights).
MIN_LEG_BETA = 0.1

# Fraction of a leg's names that need a beta estimate before the leg's mean
# beta is trusted to represent the whole leg. pandas' .mean() skips NaN, so
# without this floor a leg with betas for e.g. 1 of 4 names is scored on that
# single beta and sized as if it were full coverage — see #149.
MIN_BETA_COVERAGE = 0.8

# A sector needs this many names on a date for its mean to be worth subtracting.
# Below it, demeaning mostly removes the names' own signal: with 2 names the
# demeaned pair is always {+d, -d}, which is pure noise amplification.
MIN_SECTOR_NAMES = 4

# Conviction weights are clipped to [median/CAP, median*CAP] within a leg, so the
# most-convicted name can hold at most CAP**2 = 4x the least-convicted one.
# ponytail: a flat ratio cap, not a risk model. It bounds concentration without
# needing a covariance estimate; if leg-level factor concentration turns out to
# matter, that is the upgrade path, not a larger CAP.
CONVICTION_CAP = 2.0


def _leg_weights(leg_pred: pd.Series, center: float, notional: float,
                 conviction: bool) -> pd.Series:
    """Split `notional` across one leg's names.

    Equal-weight unless `conviction`, in which case weight is proportional to
    each name's capped distance from the date's cross-sectional center — the
    panel analogue of the single-name decision layer sizing by how far a
    prediction clears its own threshold.

    The leg's total notional is preserved exactly either way, so conviction
    redistributes *within* a leg and cannot disturb the beta- or
    dollar-neutrality the two leg notionals encode, nor the gross exposure.
    """
    k = len(leg_pred)
    if not conviction:
        return pd.Series(notional / k, index=leg_pred.index)

    score = (leg_pred - center).abs()
    med = score.median()
    if not (med > 0):
        # Degenerate cross-section (every selected name at the center) — no
        # conviction information to act on.
        return pd.Series(notional / k, index=leg_pred.index)
    score = score.clip(med / CONVICTION_CAP, med * CONVICTION_CAP)
    return notional * score / score.sum()


def sector_neutralize(pred: pd.DataFrame, sector_of: dict[str, str]) -> pd.DataFrame:
    """Subtract each (date, sector) mean from the predictions.

    Ranking alone is neutral to a market-wide shift but not to a sector-wide
    one: if the model likes Energy this week it will fill the long leg with
    Energy and the book becomes a sector bet whose Sharpe is really one
    sector's realized return. Demeaning within sector removes that level, so
    names compete against their own sector rather than across sectors.

    Symbols with no sector, and sectors with fewer than MIN_SECTOR_NAMES live
    names on a date, are passed through unchanged — a noisy correction is worse
    than none. NaNs stay NaN so panel_data's "not in the cross-section today"
    convention survives.
    """
    if not sector_of:
        return pred

    groups = pd.Series(
        [sector_of.get(c) for c in pred.columns], index=pred.columns
    ).dropna()
    if groups.empty:
        return pred

    out = pred.copy()
    for sector, cols in groups.groupby(groups):
        block = pred[list(cols.index)]
        # Per date: subtract the mean, but only where the sector is well populated.
        n_live = block.notna().sum(axis=1)
        means = block.mean(axis=1).where(n_live >= MIN_SECTOR_NAMES, 0.0)
        out[list(cols.index)] = block.sub(means, axis=0)
    return out


def rank_to_weights(
    pred_row: pd.Series,
    decile: float,
    gross_exposure: float,
    min_names: int,
    beta_row: pd.Series | None = None,
    conviction: bool = False,
) -> pd.Series:
    """Equal-weight long the top `decile` / short the bottom `decile` of one date.

    Ranking is location-invariant, so this neutralizes market-wide moves in the
    forecast for free: adding a constant to every prediction leaves the ranking
    unchanged. Sector neutrality is NOT provided (deferred increment).

    Returns weights indexed like pred_row, 0.0 for untraded names.

    `conviction` weights within each leg by distance from the cross-sectional
    center instead of equal-weighting; the leg notionals are unchanged, so every
    neutrality property below still holds. See _leg_weights.

    Without `beta_row` the book is dollar-neutral: long notional == short
    notional == gross_exposure / 2. Dollar neutrality does NOT imply market
    neutrality — measured on the live panel the ranker puts higher-beta names in
    the long leg, giving the equal-notional book a beta of +0.19 and failing the
    panel_eval precondition.

    With `beta_row` the two legs are instead sized so their beta exposures
    cancel: L*bL == S*bS with L + S == gross_exposure. That trades dollar
    neutrality (net notional is no longer 0) for beta neutrality, which is the
    property the book actually needs — net notional was only ever a proxy for
    it. Beta is estimated on a trailing window, so realized beta will not be
    exactly zero; panel_eval's +/-0.1 band still does real work as a check that
    the estimate held up out of sample.

    Each leg's mean beta is only trusted when at least MIN_BETA_COVERAGE of
    its names actually have a beta estimate — otherwise the book stays flat
    for that date (see the unconstructible-solve case below), since a mean
    over 1-2 names in a much larger leg is not representative of the leg's
    real exposure.
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

    long_notional = short_notional = gross_exposure / 2.0
    if beta_row is not None:
        beta_long = beta_row.reindex(longs)
        beta_short = beta_row.reindex(shorts)
        # .mean() skips NaN, so a leg whose beta estimates are mostly missing
        # would otherwise be sized off whichever one or two names happen to
        # have one, with no signal that the rest of the leg was unrepresented
        # (#149). Require a minimum fraction of the leg to actually have a
        # beta estimate before trusting the leg mean at all.
        cov_long = beta_long.notna().mean()
        cov_short = beta_short.notna().mean()
        if cov_long < MIN_BETA_COVERAGE or cov_short < MIN_BETA_COVERAGE:
            # beta_row was supplied, so the caller wants a beta-neutral book —
            # sizing off a handful of covered names would silently trade the
            # dollar-neutral book the module docstring calls out as *not* an
            # acceptable stand-in (measured beta +0.19 on the live panel).
            # Stay flat instead, matching the unconstructible-solve case below
            # (#148); run_panel's n_flat_days counts these dates.
            logger.warning(
                "beta coverage %.0f%%/%.0f%% (long/short) below floor %.0f%% — "
                "staying flat",
                100 * cov_long, 100 * cov_short, 100 * MIN_BETA_COVERAGE,
            )
            return weights

        b_long = beta_long.mean()
        b_short = beta_short.mean()
        # Opposite-signed leg betas would make the equal-exposure solve produce
        # a negative notional; that book is not constructible, so stay flat.
        if pd.notna(b_long) and pd.notna(b_short) and b_long > MIN_LEG_BETA and b_short > MIN_LEG_BETA:
            total_beta = b_long + b_short
            long_notional = gross_exposure * b_short / total_beta
            short_notional = gross_exposure * b_long / total_beta
        else:
            logger.debug(
                "rank_to_weights: beta-neutral solve not constructible "
                "(b_long=%s b_short=%s) — staying flat", b_long, b_short,
            )
            return weights

    # Centre on the full cross-section, not the leg: a leg-local centre would
    # make the boundary name's distance ~0 by construction on every date.
    center = float(valid.median())
    weights.loc[longs] = _leg_weights(valid.loc[longs], center, long_notional, conviction)
    weights.loc[shorts] = -_leg_weights(valid.loc[shorts], center, short_notional, conviction)
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
    conviction: bool = False   # weight within each leg by conviction, not equally


@dataclass
class PanelResult:
    equity: pd.Series
    book_ret: pd.Series
    gross_ret: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    diagnostics: dict = field(default_factory=dict)


def run_panel(
    pred: pd.DataFrame,
    ret: pd.DataFrame,
    cfg: PanelConfig,
    beta: pd.DataFrame | None = None,
) -> PanelResult:
    """Run the cross-sectional book over the panel.

    LAG CONVENTION: weights on date t come from predictions computed on close[t],
    and earn ret[t], which panel_data defines as close[t] -> close[t+1]. The
    weight and the return it earns therefore share index t. Shifting either side
    reintroduces the #106 look-ahead. `beta` follows the same convention: the row
    at t is estimated on bars through t.

    beta=None reproduces the original dollar-neutral book.
    """
    ret = ret.reindex(index=pred.index, columns=pred.columns)
    if beta is not None:
        beta = beta.reindex(index=pred.index, columns=pred.columns)
    dates = pred.index

    weight_rows: list[pd.Series] = []
    turnover_vals: list[float] = []
    prev_w = pd.Series(0.0, index=pred.columns)
    n_flat = 0

    for i, date in enumerate(dates):
        if i % cfg.rebalance_days == 0:
            w = rank_to_weights(
                pred.loc[date], cfg.decile, cfg.gross_exposure, cfg.min_names,
                beta_row=None if beta is None else beta.loc[date],
                conviction=cfg.conviction,
            )
        else:
            # Hold, don't re-peg. prev_w already carries the drift from the
            # position's own returns. Re-pegging to the original weights each
            # day would silently require daily trading to maintain, while
            # turnover — and therefore cost — is only charged on rebalance days.
            # That understates cost in proportion to rebalance_days.
            w = prev_w
        if (w == 0.0).all():
            n_flat += 1
        weight_rows.append(w)
        # prev_w is the drifted book, so this is the trade actually required to
        # reach the new target. Zero on hold days by construction.
        turnover_vals.append(float((w - prev_w).abs().sum()))
        # Carry into the next date: a position earning ret[t] is worth
        # w * (1 + ret[t]) at t+1. Missing bars drift by 0, matching the
        # skipna=True convention used for gross_ret below.
        prev_w = w * (1.0 + ret.loc[date].fillna(0.0))

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
    if beta is not None:
        # Ex-ante book beta from the weights actually held. Compare against
        # panel_eval's realized beta: a gap between them is estimation error in
        # the trailing window, not a construction bug.
        diagnostics["mean_ex_ante_beta"] = float(
            (weights * beta).sum(axis=1, skipna=True).mean()
        )

    return PanelResult(
        equity=equity,
        book_ret=book_ret,
        gross_ret=gross_ret,
        weights=weights,
        turnover=turnover,
        diagnostics=diagnostics,
    )
