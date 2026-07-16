# Step-3 Results: Cross-Sectional Panel Backtest (Phase A)

**Date:** 2026-07-15  
**Run command:** `python run_panel.py --days 2500`  
**Model:** `models/daily_predictor.pkl` (Ridge, `daily_v6` features, 30 cols)  
**History window:** 2500 calendar days (~1597 trading days, 2019-09-10 to 2026-07-15)

---

## Universe

| Metric | Value |
|--------|-------|
| Configured | 157 symbols (stocks only — no ETFs) |
| Loaded successfully | 156 |
| Dropped | 1 (HES — insufficient bars: 0 < 250) |
| Sectors | Technology (30), Financials (20), Healthcare (20), Consumer (20), Industrials (20), Energy (14), Communications (10), Utilities & Real Estate (14), Materials (9) |

HES returned zero bars from yfinance, likely a delisting or symbol-change issue.

---

## Per-Config Results (5 bps, 1597 dates)

| config (decile, rebalance_days) | ann_sharpe | mean_turnover | total_cost_drag | flat_days |
|--------------------------------|------------|---------------|-----------------|-----------|
| (0.1, 1) — daily, narrow | -0.42 | 1.058 | 0.861 | 0 |
| (0.1, 3) — 3-day, narrow | +0.34 | 0.427 | 0.357 | 0 |
| (0.2, 1) — daily, wide | -0.42 | 0.883 | 0.721 | 0 |
| **(0.2, 3) — 3-day, wide** | **+0.48** | **0.359** | **0.302** | **0** |

Best config by per-period Sharpe: **(0.2, 3)**

---

## Gate Metrics (5 bps baseline)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| beta vs SPY | **+0.186** | abs < 0.10 | **PRECONDITION FAIL** |
| DSR | 0.472 | >= 0.95 | FAIL (secondary) |
| SR (best config) | 0.0305 | — | — |
| SR0 (expected max SR) | 0.0322 | — | — |
| PBO | 0.000 | — | — |

**Gate verdict: FAIL**

The beta precondition (`|beta| < 0.10`) is checked first; a beta of +0.186 means the Sharpe and DSR describe something other than the intended dollar-neutral book, and the gate stops there.

---

## Cost Sensitivity: 5 bps vs 10 bps

| config | ann_sharpe @ 5 bps | ann_sharpe @ 10 bps | verdict |
|--------|--------------------|---------------------|---------|
| (0.1, 1) | -0.42 | -1.60 | FAIL/FAIL |
| (0.1, 3) | +0.34 | -0.13 | FAIL/FAIL |
| (0.2, 1) | -0.42 | -1.69 | FAIL/FAIL |
| (0.2, 3) | +0.48 | -0.04 | FAIL/FAIL |

| Gate metric | 5 bps | 10 bps |
|-------------|-------|--------|
| beta | +0.186 | +0.186 |
| DSR | 0.472 | 0.006 |
| Verdict | FAIL | FAIL |

The gate fails at both cost levels. The 3-day rebalance configs flip from positive to near-zero Sharpe when costs double — the book is turnover-heavy enough that the cost assumption matters, but not the deciding factor here. The beta precondition fails regardless.

---

## Verdict

**GATE: FAIL**

Primary reason: beta +0.186 vs SPY violates the |beta| < 0.10 precondition. The book is dollar-neutral (mean_net_exposure ≈ 0 at floating-point precision), but dollar-neutral is not the same as beta-neutral. The Ridge predictor's cross-sectional rankings are positively correlated with market beta: it systematically places higher-beta names on the long side and lower-beta names on the short side (or vice versa). This creates ~0.19 units of market exposure for every dollar gross.

Secondary reason (would fail independently): DSR 0.472 < 0.95. Even if the beta precondition were waived, there is no statistically significant neutral edge after multiple-testing correction across the 4-config grid.

PBO = 0.000 is favorable (the best config dominates in all permutations) but is irrelevant while the primary gates are failing.

**Phase B (vol-targeting overlay) should not be built on this result.** A dollar-neutral book with 0.19 beta is not testing the cross-sectional signal in isolation — it is testing the cross-sectional signal plus a market-directional bet. Phase B would inherit and potentially amplify that bias.

---

## Recommended next investigation (not a commitment to build)

The beta finding is diagnostic, not terminal:

1. **Compute per-name beta at ranking time** and weight-adjust positions to neutralize. This is the standard extension from dollar-neutral to beta-neutral portfolio construction.
2. **Examine which names are persistently long vs short**: if high-beta names (NVDA, TSLA, META) cluster on the long side and low-beta names (utilities, healthcare staples) cluster on the short, the predictor has learned a beta factor, not a cross-sectional momentum edge.
3. **Re-run after beta-neutralization** to see if DSR crosses the 0.95 threshold on the residual signal.

This is a Phase A increment, not a new research direction.

---

## Known limitations

1. **Survivorship bias**: the 156-symbol universe is drawn from 2026 index constituents. Symbols that existed in 2019 but were delisted, merged, or dropped from the index before 2026 are absent. Historical returns for surviving names are therefore biased upward — they are the names that did not go bust.

2. **Cost assumption**: 5 bps one-way on turned-over notional is a plausible but unvalidated estimate for a ~$100k simulated book. Real fill costs for a large institutional book executing 15-name lots daily would be higher; for a retail backtest with no market impact, they might be lower.

3. **Fill model**: assumes fills at close of day t with weights also computed from close[t] prices. Real-world execution at the next-day open or VWAP would add noise, a timing lag, and additional slippage not captured here.

4. **Borrow costs**: 50 bps/year flat on short notional is the assumption. Hard-to-borrow names in the universe (e.g., TSLA during high short-interest periods) carry multiples of this rate; GC rate fluctuates.

5. **No sector neutrality**: the ranker operates across the full cross-section. If the predictor has a sector tilt (e.g., systematically bullish on Tech), the book carries sector exposure that is not hedged.

6. **Feature staleness**: the Ridge predictor was trained on the same symbols used in the panel. Spearman IC of ~0.04 on 10-symbol walk-forward (Step 2) does not guarantee cross-sectional rank IC on a 156-symbol universe drawn from different sectors and market-cap tiers.

---

## Files

| File | Description |
|------|-------------|
| `results/panel_summary.json` | Full grid results at 5 bps (canonical run) |
| `results/panel_summary_cost10.json` | Sensitivity run at 10 bps |
| `run_panel.py` | CLI entry point |
| `panel_backtester.py` | Engine + `PanelConfig` |
| `panel_data.py` | Data loading and alignment |
| `panel_eval.py` | Gate: DSR, beta, PBO |
