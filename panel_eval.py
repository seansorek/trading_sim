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

from cpcv import cscv_pbo
from deflated_sharpe import deflated_sharpe

logger = logging.getLogger(__name__)

# decile x rebalance_days. cost_bps is an ASSUMPTION and never enters this grid:
# tuning a cost assumption until the gate passes is fiction, and it would make
# PBO measure the wrong thing.
CONFIG_GRID: list[tuple[float, int]] = [(0.1, 1), (0.1, 3), (0.2, 1), (0.2, 3)]

DSR_THRESHOLD = 0.95
BETA_LIMIT = 0.1
MIN_BETA_OBS = 30


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
