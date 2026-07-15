# Step 2 — Land & honestly re-validate `daily_v5`, then a disciplined orthogonal-feature increment

**Date:** 2026-07-14
**Status:** Approved for planning
**Feature-set version shipped:** `daily_v6`
**Base branch:** `feat/step2-daily-v6-orthogonal`, branched off `main` at `1444abb` (#106)

## Context

This is "sequence step 2" from the pipeline-improvement analysis: originally
**#3 (ensemble)** + **#5 (orthogonal features)**. Two decisions during
brainstorming reshaped it:

1. **Ensemble (#3) deferred.** Per the #106 re-baseline, the two classifiers
   (`daily_logistic`, `daily_xgboost`) sit at IC ≈ 0 — no signal. Only
   `daily_predictor` (Ridge) carries signal (OOS IC +0.06). Averaging one
   skilled model with two unskilled ones dilutes IC. An ensemble is revisited
   only once ≥2 individually-skilled, decorrelated signals exist.

2. **`daily_v5` is the vehicle for #5.** An unmerged 28-commit
   `feat/daily-v5-preprocessing` branch (merge-base `b115b21`, ~2026-07-09)
   already does most of #5's feature work: drops redundant features, adds
   several new ones, adds per-symbol rolling z-score + RobustScaler, fixes the
   mis-scaled regression target, and retrains all models. **But** it was built
   and "validated" *before* #106 landed the 1-bar look-ahead fix and the
   DSR/PBO referee, so its numbers are suspect in exactly the way the old
   `daily_predictor` numbers were (pre-fix Sharpe +0.27 → post-fix +0.16). It
   has no open PR and conflicts with `main` today (same files as #106).

So step 2 = **port `daily_v5` onto `main`, honestly re-validate it under the
referee it never saw, then add a small hypothesis-driven orthogonal increment.**

## Honest baseline (from #106, `models/README.md`)

- `daily_predictor` (Ridge, `daily_v3`): OOS IC **+0.06**, PBO **0.513**
  (high-overfit zone), alpha **−8.94%**, IR **−0.62**, median DSR 0.80.
  README verdict: *not a proven deployable edge.*
- Classifiers: at majority-class baseline, negative Sharpe.

The real value of step 2 is landing v5's overhaul and validating it honestly —
not the new-feature increment, which is deliberately tiny and may return null.

## Success criteria & keep/drop gate

- **Primary gate (blocking):** a feature (or the ported v5 set as a whole) is
  kept only if it **raises median OOS walk-forward IC** *and* **does not worsen
  PBO** (`cpcv.cscv_pbo`) by more than a **+0.05** tolerance versus the
  ported-v5 re-baseline. (The baseline PBO is already ~0.51, so the rule is
  "don't make overfitting worse," not "reach a low absolute PBO.")
- **Secondary (reported, non-blocking):** per-symbol DSR and alpha/IR vs
  buy-and-hold from `eval_report.py`. Recorded so nothing regresses silently,
  but not used to reject a signal — portfolio alpha is step 3 / #1's job, not
  step 2's.
- **A null result is an acceptable, documented outcome.** If the ported v5 set
  and the new features do not beat the `daily_v3` baseline on OOS IC + PBO, we
  ship the honest finding and the validated baseline, not a worse model.

## Scope

**In scope**

- Port `daily_v5`'s feature + preprocessing overhaul onto `main` (Approach B,
  below).
- Re-validate the ported feature set under `eval_report` (walk-forward IC, PBO,
  per-symbol DSR).
- Add two genuinely-new orthogonal features (§ New features).
- Retrain all live models + `daily_hybrid` on the final `daily_v6` contract,
  re-tune `(signal_quantile, threshold_window)`, record honest numbers.
- Update tests and `models/README.md`.

**Out of scope (named explicitly)**

- The ensemble (#3) — deferred until ≥2 skilled signals exist.
- `mean_reversion_strategy.py` (a strategy, not a feature) — v5 added it; step 2
  is features-only.
- Earnings-proximity feature — real economic prior, but needs a reliable
  point-in-time earnings-date source (yfinance is sparse/unreliable and
  PIT-messy). Its own mini-project.
- Cross-sectional / portfolio construction (#1) and vol-targeting (#2).
- DQN retraining — its state dimension breaks on a feature-count change; it is
  not live, so its stale pickle is simply skipped at load. Optional cleanup
  only.

## Approach B — squash-port the net diff

The mandated re-validation means we re-baseline from scratch regardless, so
replaying 28 commits through the #106 conflict zone buys nothing.

1. Branch off `origin/main` (done: `feat/step2-daily-v6-orthogonal` at
   `1444abb`).
2. Apply `daily_v5`'s **feature + preprocessing** changes as a few logical
   commits:
   - `daily_features.py` rewrite: the 29-feature `daily_v5` contract, the
     `_rolling_zscore` helper (window 252, min_periods 60, unbounded features
     only), dropped exact duplicates (`williams_r`, `macd_hist`, `vol_z_20`).
   - RobustScaler in `predictors/base.py`.
   - The vol-adjusted **target fix** in `train_predictor.py` (divide by raw
     volatility, not z-scored `vol_20d`).
   - `config/default.yaml` deltas that accompany the feature change.
3. **Hand-reconcile** the three files both branches touched —
   `walk_forward.py`, `ml_strategies.py`, `simulation_pipeline.py` — so
   **#106's look-ahead shift-fix and eval refactor
   (`build_fold_data` / `fold_config_ic_matrix`) are preserved**, not
   overwritten by v5's older versions. This is the delicate part of the port
   and gets its own verification (diff the reconciled files against both
   parents; confirm the shift-fix line and the `build_fold_data` split both
   survive).
4. **Do not** bring v5's committed pickles — they were trained pre-fix. Retrain
   fresh at the end.

## New features (small, hypothesis-driven)

After redundancy-checking against v5's 29 features, two survive as genuinely
new orthogonal sources:

- **Implied-vol regime (VIX):** `^VIX` rolling-z level + 5-day change. Fetched
  once and reindexed like the existing `spy_df` market-relative features.
  Orthogonal to v5's `vol_regime` (realized vol) via the variance risk premium
  (implied ≠ realized). Two columns.
- **Amihud illiquidity:** `|ret_1d| / dollar_volume`, per-symbol rolling-z.
  Distinct from `turnover_z` (volume *level* vs return *impact*). One column.

**Dropped as redundant:** a raw short-term "reversal" feature — v5's rolling-z
`ret_5d` / `ret_10d` already expose recent-return information to the linear
model, which learns the sign. Adding a reversal column mainly inflates the
multiple-testing surface.

Both new features are **causal** (current + past bars only): VIX rolling-z uses
its own past window; Amihud uses same-bar return and volume, then a trailing
rolling z-score. No look-ahead.

## Validation methodology

1. **Re-baseline the ported v5 set first**, before any new feature: run
   `eval_report.py` (walk-forward median IC, PBO over the `(q,w)` grid,
   per-symbol DSR). This number replaces v5's pre-fix claims and is the honest
   reference point.
2. **Per-feature evaluation:** add a candidate → retrain the Ridge predictor →
   recompute OOS median walk-forward IC and PBO. Keep per the gate above.
3. **Multiple-testing discipline:** the two candidates are pre-registered by
   economic hypothesis, not greedy-searched over dozens; the trial count is
   recorded; the final selected `daily_v6` set is evaluated once via CPCV PBO +
   DSR.
4. The `|corr| > 0.98` feature-pair guard (from v5) stays and must cover the
   new columns.

## Retrain / re-tune / artifacts

- New contract → bump `FEATURE_SET_NAME` to **`daily_v6`** (disambiguates from
  the abandoned v5 branch tree).
- Retrain all live models (`daily_logistic`, `daily_xgboost`,
  `daily_predictor`) + `daily_hybrid` on the final contract; re-run
  `walk_forward.sweep_params` for `(signal_quantile, threshold_window)`; write
  pickles + `.sha256`; update `model_registry`.
- Rewrite the `models/README.md` model cards with honest post-referee numbers
  (OOS IC, PBO, DSR, alpha/IR) and the `daily_v6` feature-set description.

## Testing

- `test_feature_contract.py` — updated `daily_v6` contract; train/predict
  consistency.
- `test_features.py` — no NaN/inf, warmup `dropna`, new columns present and
  finite.
- `test_data_leakage.py` — assert VIX and Amihud are causal (no future bars).
- Correlation guard (`|corr| > 0.98`) extended to cover the new columns.
- Record an `eval_report` run (the honest step-2 result) in `models/README.md`.

## Risks

- **Port reconciliation** of `walk_forward.py` / `ml_strategies.py` /
  `simulation_pipeline.py` is where a silent regression could sneak in
  (dropping the #106 look-ahead fix would re-inflate the backtest). Mitigation:
  explicit post-reconcile diff check against both parents (§ Approach B step 3).
- **Target-fix interaction:** v5 changed the regression target scaling. This
  changes what "IC" measures, so the re-baseline must be run *after* the target
  fix is in place, on the same footing for baseline and new-feature runs.
- **Null result likely.** Given the honest baseline, the increment may not beat
  `daily_v3`. That is a valid outcome; the deliverable is the validated
  baseline + honest numbers, not a forced improvement.
