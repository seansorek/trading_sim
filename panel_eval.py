"""
panel_eval.py — Gate and diagnostics for the cross-sectional panel book.

Gate (see the spec): deflated Sharpe >= 0.95 on net-of-cost book returns, with
|beta| < 0.1 vs SPY as a PRECONDITION. Beta is a correctness check, not a
performance one — a dollar-neutral book must have ~zero market beta by
construction, so beta outside the band means the neutralization failed and the
Sharpe describes something other than the intended book.

"Alpha vs buy-and-hold" is deliberately NOT the gate: a zero-beta book
underperforming a long benchmark is expected, not informative.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cpcv import cscv_pbo
from daily_features import FWD_RET_HORIZON_DAYS
from deflated_sharpe import deflated_sharpe

logger = logging.getLogger(__name__)

# decile x rebalance_days. cost_bps is an ASSUMPTION and never enters this grid:
# tuning a cost assumption until the gate passes is fiction, and it would make
# PBO measure the wrong thing.
#
# Rebalance periods run out to 10 days because the IC decay curve is flat to
# rising through h=10 (see ic_report) while turnover — and therefore cost — falls
# roughly as 1/rebalance_days. Holding longer is the only lever that acts on the
# cost side without touching the cost assumption.
#
# Widening this grid is NOT free: len(CONFIG_GRID) is n_trials for the deflated
# Sharpe, so every config added raises the sr0 bar the winner must clear. That is
# the correct price of looking at more configurations, and it is why the grid is
# a fixed list rather than something a caller can extend ad hoc.
CONFIG_GRID: list[tuple[float, int]] = [
    (0.1, 1), (0.1, 3), (0.1, 5), (0.1, 10),
    (0.2, 1), (0.2, 3), (0.2, 5), (0.2, 10),
]

DSR_THRESHOLD = 0.95
BETA_LIMIT = 0.1
MIN_BETA_OBS = 30


# ---------------------------------------------------------------------------
# Cross-sectional IC diagnostics
#
# The gate above is end-to-end: a FAIL cannot distinguish a bad ranker from bad
# weighting from costs. These functions measure the ranker alone.
#
# The distinction that matters: train_predictor._forecast_metrics pools every
# (date, symbol) sample into ONE Spearman correlation, which is dominated by
# time-series variation — "everything rose on Tuesday" scores well and carries
# zero ranking skill. rank_to_weights is location-invariant, so it discards
# exactly that component. Only the PER-DATE cross-sectional IC survives into
# book returns.
# ---------------------------------------------------------------------------

IC_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10)
MIN_IC_NAMES = 20  # matches PanelConfig.min_names — below this a date is untradeable


def forward_return(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Cumulative forward return over `horizon` bars, close[t] -> close[t+horizon]."""
    return close.shift(-horizon) / close - 1.0


def cross_sectional_ic(
    pred: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    min_names: int = MIN_IC_NAMES,
) -> pd.Series:
    """One Spearman rank correlation per date, across the universe.

    Returns a Series on pred.index; NaN on dates with fewer than `min_names`
    valid (pred, return) pairs or with a constant row, where the rank
    correlation is undefined. NaN means "not measured", never 0 — a zero would
    be averaged in as evidence of no skill rather than absence of evidence.
    """
    fwd = fwd_ret.reindex(index=pred.index, columns=pred.columns)
    ics = np.full(len(pred), np.nan)
    for i, date in enumerate(pred.index):
        pair = pd.concat([pred.loc[date], fwd.loc[date]], axis=1).dropna()
        if len(pair) < min_names:
            continue
        a, b = pair.iloc[:, 0].values, pair.iloc[:, 1].values
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        ic, _ = spearmanr(a, b)
        ics[i] = ic
    return pd.Series(ics, index=pred.index, name="ic")


def _ic_stats(ic: pd.Series, overlapping: bool) -> dict:
    """Summarize a per-date IC series.

    overlapping=True means the horizon exceeds the 1-day sampling interval, so
    consecutive observations share return windows. The autocorrelation that
    creates inflates t_stat and ann_ic_ir — they are reported for shape
    comparison across horizons, NOT as significance tests.
    """
    v = ic.dropna()
    n = len(v)
    if n < 2:
        return {"n_dates": n, "mean_ic": float("nan"), "ic_ir": float("nan")}
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    ic_ir = mean / sd if sd > 0 else float("nan")
    return {
        "n_dates": n,
        "mean_ic": mean,
        "std_ic": sd,
        "ic_ir": ic_ir,
        # Book Sharpe tracks roughly ic_ir * sqrt(252) at a daily rebalance.
        "ann_ic_ir": ic_ir * float(np.sqrt(252)),
        "t_stat": mean / (sd / np.sqrt(n)) if sd > 0 else float("nan"),
        "pct_positive": float((v > 0).mean()),
        "overlapping": overlapping,
    }


def _pooled_ic(pred: pd.DataFrame, fwd_ret: pd.DataFrame) -> float:
    """Spearman over every (date, symbol) sample stacked together.

    This is the metric train_predictor reports. Included only as the contrast
    to mean_ic: pooled >> per-date means the model forecasts the market, not
    the cross-section.
    """
    pair = pd.concat(
        [pred.stack(), fwd_ret.reindex(index=pred.index, columns=pred.columns).stack()],
        axis=1,
    ).dropna()
    if len(pair) < 2:
        return float("nan")
    a, b = pair.iloc[:, 0].values, pair.iloc[:, 1].values
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    ic, _ = spearmanr(a, b)
    return float(ic)


def ic_report(
    pred: pd.DataFrame,
    close: pd.DataFrame,
    horizons: tuple[int, ...] = IC_HORIZONS,
    min_names: int = MIN_IC_NAMES,
) -> dict:
    """Per-date IC, IC-IR, and IC decay across horizons, plus the pooled contrast.

    `primary_horizon` is the model's training target (FWD_RET_HORIZON_DAYS), so
    per_date_ic and pooled_ic there are measured against the same thing the
    model was fit on and are directly comparable.
    """
    per_horizon = {}
    for h in horizons:
        fwd = forward_return(close, h)
        ic = cross_sectional_ic(pred, fwd, min_names=min_names)
        per_horizon[h] = _ic_stats(ic, overlapping=h > 1)

    primary = FWD_RET_HORIZON_DAYS
    pooled = _pooled_ic(pred, forward_return(close, primary))
    n_names = pred.notna().sum(axis=1)

    return {
        "horizons": {str(h): s for h, s in per_horizon.items()},
        "primary_horizon": primary,
        "per_date_ic_at_primary": per_horizon.get(primary, {}).get("mean_ic", float("nan")),
        "pooled_ic_at_primary": pooled,
        "n_dates": int(len(pred)),
        "mean_n_names": float(n_names.mean()),
        "n_dates_below_min_names": int((n_names < min_names).sum()),
    }


def book_beta(book_ret: pd.Series, spy_ret: pd.Series) -> float:
    """OLS beta of the book against SPY. NaN if too few overlapping observations."""
    aligned = pd.concat(
        [book_ret.rename("book"), spy_ret.rename("spy")], axis=1
    ).dropna()
    if len(aligned) < MIN_BETA_OBS:
        return float("nan")
    var = aligned["spy"].var()
    if not var or var <= 0:
        return float("nan")
    return float(aligned["spy"].cov(aligned["book"]) / var)


def _per_period_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def evaluate_grid(
    results: dict[tuple[float, int], "object"],
    spy_ret: pd.Series,
) -> dict:
    """Score the config grid, deflate the winner's Sharpe, and apply the gate.

    results maps (decile, rebalance_days) -> PanelResult.
    """
    if not results:
        raise ValueError("evaluate_grid needs at least one config result")

    # Per-period (NOT annualized) Sharpe. Passing an annualized Sharpe variance
    # to deflated_sharpe made sr0 ~15.9x too large and drove DSR to 0 for every
    # symbol — the unit bug fixed in #106.
    pp_sharpes = {cfg: _per_period_sharpe(res.book_ret) for cfg, res in results.items()}
    best_config = max(pp_sharpes, key=pp_sharpes.get)
    best = results[best_config]

    trial_var = (
        float(np.var(list(pp_sharpes.values()), ddof=1)) if len(pp_sharpes) > 1 else 0.0
    )
    dsr_out = deflated_sharpe(
        best.book_ret.dropna().values,
        n_trials=len(results),
        trial_sharpe_var=trial_var,
    )

    beta = book_beta(best.book_ret, spy_ret)

    # PBO across the grid: observations x configs matrix of daily book returns.
    perf = pd.DataFrame({str(cfg): res.book_ret for cfg, res in results.items()}).dropna()
    pbo_out = cscv_pbo(perf.values) if perf.shape[1] >= 2 else {"pbo": float("nan")}

    if np.isnan(beta):
        passed, verdict = False, "beta undefined — too few overlapping observations with SPY"
    elif abs(beta) >= BETA_LIMIT:
        passed, verdict = False, (
            f"beta {beta:+.3f} outside +/-{BETA_LIMIT} — neutralization failed. "
            "Diagnose the book before reading its performance."
        )
    elif dsr_out["dsr"] < DSR_THRESHOLD:
        passed, verdict = False, (
            f"DSR {dsr_out['dsr']:.3f} < {DSR_THRESHOLD} — no statistically "
            "significant edge after multiple-testing correction. Null result: "
            "do not build Phase B."
        )
    else:
        passed, verdict = True, (
            f"DSR {dsr_out['dsr']:.3f} >= {DSR_THRESHOLD} with beta {beta:+.3f} — "
            "significant neutral edge. Phase B (vol-targeting) is justified."
        )

    return {
        "best_config": best_config,
        "dsr": float(dsr_out["dsr"]),
        "sr": float(dsr_out["sr"]),
        "sr0": float(dsr_out["sr0"]),
        "beta": beta,
        "pbo": float(pbo_out.get("pbo", float("nan"))),
        "passed": passed,
        "verdict": verdict,
        "per_config": {
            str(cfg): {
                "pp_sharpe": pp_sharpes[cfg],
                "ann_sharpe": pp_sharpes[cfg] * float(np.sqrt(252)),
                **results[cfg].diagnostics,
            }
            for cfg in results
        },
    }
