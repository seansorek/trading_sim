#!/usr/bin/env python3
"""
run_panel.py — Entry point for the cross-sectional panel backtest (Step 3 Phase A).

Research-only. Touches no live prediction path.

Usage:
    python run_panel.py --days 2500
    python run_panel.py --days 2500 --cost-bps 10   # cost SENSITIVITY, not tuning

NOTE on --cost-bps: this flag is for *sensitivity reporting* only. cost_bps is a
modelling assumption (round-trip transaction cost on turned-over notional). Tuning
it until the gate passes is not valid — the gate is only meaningful when the cost
assumption is fixed in advance. The default of 5 bps matches config/default.yaml.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import get_config
from panel_backtester import PanelConfig, run_panel, sector_neutralize
from panel_data import build_panels
from panel_eval import CONFIG_GRID, evaluate_grid, ic_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def run_grid(panel_data, base_cfg: PanelConfig, spy_ret: pd.Series) -> dict:
    """Run every (decile, rebalance_days) config and evaluate the grid."""
    results = {}
    for decile, rebal in CONFIG_GRID:
        cfg = dataclasses.replace(base_cfg, decile=decile, rebalance_days=rebal)
        results[(decile, rebal)] = run_panel(
            panel_data.pred, panel_data.ret, cfg, beta=panel_data.beta
        )
        logger.info("ran config decile=%.2f rebalance_days=%d", decile, rebal)
    return evaluate_grid(results, spy_ret)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-sectional panel backtest (research-only, Phase A gate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "NOTE: --cost-bps is for SENSITIVITY REPORTING only — it is a modelling\n"
            "assumption, not a parameter to tune until the gate passes. The default\n"
            "of 5 bps matches config/default.yaml and must not be changed to chase\n"
            "a PASS verdict."
        ),
    )
    parser.add_argument("--days", type=int, default=2500)
    parser.add_argument("--db", default="data/trading_sim.db")
    parser.add_argument("--model", default="models/daily_predictor.pkl")
    parser.add_argument(
        "--cost-bps", type=float, default=None,
        help=(
            "Override cost_bps for SENSITIVITY reporting. This is a modelling "
            "assumption (one-way cost on turned-over notional), not a parameter to "
            "tune until the gate passes. Default: config/default.yaml → panel.cost_bps (5)."
        ),
    )
    parser.add_argument(
        "--no-sector-neutral", action="store_true",
        help="Rank raw predictions instead of sector-demeaned ones (A/B comparison).",
    )
    parser.add_argument("--output", default="results/panel_summary.json")
    args = parser.parse_args()

    app_cfg = get_config()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    from db import DB
    db = DB(args.db)

    from train_models import _load_symbol
    spy_df = _load_symbol("SPY", start, end, db)
    if spy_df is None:
        raise RuntimeError("SPY failed to load — needed for features and the beta gate.")
    spy_ret = spy_df["close"].astype(float).pct_change().shift(-1)

    logger.info("Building panels for %d symbols...", len(app_cfg.panel.universe))
    panel_data = build_panels(
        app_cfg.panel.universe, start, end, db,
        model_path=args.model, spy_df=spy_df,
    )
    logger.info(
        "Panel: %d symbols x %d dates (%d dropped)",
        len(panel_data.symbols), len(panel_data.pred), len(panel_data.dropped),
    )

    if args.no_sector_neutral:
        logger.info("Sector neutralization DISABLED by flag")
    else:
        sector_of = app_cfg.panel.sector_of()
        covered = sum(1 for s in panel_data.symbols if s in sector_of)
        logger.info(
            "Sector-neutralizing predictions: %d/%d symbols across %d sectors",
            covered, len(panel_data.symbols), len(app_cfg.panel.sectors),
        )
        # Applied before BOTH the IC diagnostic and the book, so the diagnostic
        # scores the same predictions the book actually ranks.
        panel_data = dataclasses.replace(
            panel_data, pred=sector_neutralize(panel_data.pred, sector_of)
        )

    base_cfg = PanelConfig(
        gross_exposure=app_cfg.panel.gross_exposure,
        cost_bps=args.cost_bps if args.cost_bps is not None else app_cfg.panel.cost_bps,
        borrow_bps_annual=app_cfg.panel.borrow_bps_annual,
        min_names=app_cfg.panel.min_names,
    )
    out = run_grid(panel_data, base_cfg, spy_ret)
    out["ic"] = ic_report(panel_data.pred, panel_data.close, min_names=base_cfg.min_names)
    out["universe_size"] = len(panel_data.symbols)
    out["dropped"] = panel_data.dropped
    out["cost_bps"] = base_cfg.cost_bps

    Path("results").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)

    ic = out["ic"]
    print("\n" + "=" * 72)
    print("RANKER DIAGNOSTIC — cross-sectional IC")
    print("=" * 72)
    print(f"  {len(panel_data.symbols)} symbols x {ic['n_dates']} dates, "
          f"mean {ic['mean_n_names']:.0f} names/date, "
          f"{ic['n_dates_below_min_names']} dates below min_names")
    print(f"  {'horizon':<9}{'mean_IC':>10}{'IC-IR':>9}{'ann':>8}{'t':>8}{'%pos':>8}{'n':>7}")
    for h, s in ic["horizons"].items():
        mark = "" if h == "1" else " *"
        print(
            f"  {h + 'd':<9}{s['mean_ic']:>+10.4f}{s['ic_ir']:>+9.3f}"
            f"{s['ann_ic_ir']:>+8.2f}{s['t_stat']:>+8.2f}"
            f"{s['pct_positive']:>8.1%}{s['n_dates']:>7d}{mark}"
        )
    print("  * overlapping windows — t and ann inflated; read the shape, not the significance")
    print("-" * 72)
    print(f"  at the {ic['primary_horizon']}d training horizon:")
    print(f"    per-date IC : {ic['per_date_ic_at_primary']:+.4f}   <- what the book can trade")
    print(f"    pooled IC   : {ic['pooled_ic_at_primary']:+.4f}   <- what training reports")
    print("=" * 72)

    print("\n" + "=" * 72)
    print("PANEL BACKTEST — STEP 3 PHASE A")
    print("=" * 72)
    for cfg_key, stats in out["per_config"].items():
        print(
            f"  {cfg_key:<20} ann_sharpe={stats['ann_sharpe']:+.2f}  "
            f"turnover={stats.get('mean_turnover', 0):.3f}  "
            f"ex_ante_beta={stats.get('mean_ex_ante_beta', float('nan')):+.3f}  "
            f"flat_days={stats.get('n_flat_days', 0)}"
        )
    print("-" * 72)
    print(f"  best config : {out['best_config']}")
    print(f"  DSR         : {out['dsr']:.3f}  (SR={out['sr']:.4f} SR0={out['sr0']:.4f})")
    print(f"  beta        : {out['beta']:+.3f}")
    print(f"  PBO         : {out['pbo']:.3f}")
    print(f"  GATE        : {'PASS' if out['passed'] else 'FAIL'}")
    print(f"  {out['verdict']}")
    print("=" * 72)
    logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
