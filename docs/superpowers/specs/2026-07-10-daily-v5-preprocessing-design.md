# daily_v5 Preprocessing Overhaul — Design

**Date:** 2026-07-10
**Status:** Approved for planning
**Scope:** One combined change shipping as feature-set version `daily_v5`. Six
issues in the data-preprocessing workflow, fixed together, followed by a single
retrain and a single walk-forward threshold re-tune.

## Motivation

Analysis of the preprocessing pipeline (`make_daily_features` →
`_preprocess` → `StandardScaler` → model) surfaced six issues spanning
redundant features, missing normalization, a train/serve consistency bug, an
outlier-sensitive scaler, a mis-scaled regression target, and doc/dead-code
drift. They are coupled through the feature contract, so they ship as one
version bump rather than piecemeal.

## Design decisions (settled during brainstorming)

- **Scope:** all six issues in one `daily_v5` change; retrain + re-tune once.
- **Normalization approach (#2):** per-symbol time-series rolling z-score
  (not per-date cross-sectional), because the training universe is only ~10
  symbols and cross-sectional ranks depend on universe composition, creating a
  train/serve mismatch when `prediction.symbols` is wider.
- **Z-score scope (#2):** unbounded / scale-varying features only. Bounded
  oscillators keep their native scale so absolute thresholds (RSI 70 =
  overbought, ADX 25 = trending) survive.

## The six changes

### #1 — Drop provably-redundant features

Remove the two **exact** linear dependencies from `FEATURE_COLS`:

- `williams_r` — equals `stoch_k` − 100 (identical 14-bar HH/LL windows), so
  `corr = −1.0`.
- `macd_hist` — equals `macd` − `macd_signal` by construction
  (`daily_features.py:146`).

Keep `stoch_d` (≈0.9 correlation with `stoch_k`; carries 3-bar smoothing, below
the guard threshold). Feature count: **32 → 30**, then **→ 29** after the guard
below exposed a third near-duplicate (see note).

Stop computing the two dropped series entirely (remove their assignments).

**Guard:** add a regression test asserting no feature pair in `FEATURE_COLS`
exceeds `|corr| > 0.98` on a representative sample, so future additions cannot
silently reintroduce an exact duplicate.

**Implementation note (added during execution):** the `|corr| > 0.98` guard
exposed a real near-duplicate the design missed — `vol_z_20` ~ `turnover_z`
(r = 0.997), since share-volume and dollar-volume z-scores track almost
perfectly. Per decision, `vol_z_20` was dropped (kept `turnover_z`, the richer
dollar-volume measure), so the final feature count is **29**, not 30. All
"30-feature" references elsewhere in this doc should read **29**.

### #2 — Per-symbol rolling z-score (unbounded features only)

New helper in `daily_features.py`:

```python
def _rolling_zscore(s: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    return (s - mean) / (std + 1e-12)
```

Applied to **unbounded** features:
`ret_1d`, `ret_5d`, `ret_10d`, `ret_21d`, `vol_20d`, `macd`, `macd_signal`,
`ma_spread_10_20`, `ma_spread_20_50`, `price_vs_sma20`, `price_vs_sma50`,
`roc_12`, `gap`, `hl_ratio`, `vol_regime`, `rel_volume`, `ret_1d_vs_spy`,
`ret_5d_vs_spy`.

**Left at native scale** (already bounded oscillators): `rsi_14`, `stoch_k`,
`stoch_d`, `adx_14`, `bb_position`, `atr_normalized`.

**Left unchanged** (already per-symbol z-scores): `turnover_z`,
`vpt_normalized`, `ad_normalized`, `obv_normalized`, `bb_width`.
(`vol_z_20` was in this set in the original design but was dropped — see the #1
implementation note.)

Rationale for rolling (not expanding): adapts to regime shifts. Uses only
current + past bars → **no look-ahead**. Extends the warmup period; absorbed by
the existing `dropna(subset=FEATURE_COLS)`.

### #3 + #4 — Reorder preprocessing (fused)

**Current problems:**
- `_preprocess` recomputes clip bounds (mean/std) from *whatever batch it is
  handed* (`predictors/base.py:18`, `train_models.py:87`). Clip bounds differ
  between training slices, backtest batches, and single-row live calls →
  train/serve skew (#3).
- `StandardScaler` (mean/std) is outlier-sensitive on fat-tailed return
  features (#4).

**New order**, shared across all trainers and predictors:

1. `inf/nan → 0` — `_preprocess` keeps *only* this step (drop the per-batch clip).
2. `RobustScaler.transform` — fit on train, frozen in the artifact (#4).
3. `clip` to fixed `±5` on robust-scaled units (#3).

Because the clip now runs on **frozen-scaler output**, there are no per-batch
statistics; the skew bug is gone and the transform is identical at train and
serve time.

New shared helper (in `predictors/base.py`):

```python
CLIP = 5.0

def _scale(scaler, X: np.ndarray) -> np.ndarray:
    X = _preprocess(X)                 # inf/nan -> 0 only
    X = scaler.transform(X)
    return np.clip(X, -CLIP, CLIP)
```

The four predictors (`ridge`, `logistic`, `xgboost`, and the DQN path if
applicable) call `_scale` instead of repeating `_preprocess` + `transform`.

**XGBoost:** keeps a fitted `RobustScaler` too (harmless no-op — XGBoost is
scale-invariant) rather than adding a `None`-branch to the shared path.
Special-casing saves microseconds and costs a branch, so it is not worth it.

### #5 — Vol-normalized regression target

Add an auxiliary column in `daily_features.py`:

```python
feats["fwd_ret_vol_adj"] = feats["fwd_ret_1d"] / (feats["vol_20d"] + 1e-6)
```

- The **regressor** (`train_predictor.py`) trains on `fwd_ret_vol_adj`.
- The **classifier** (`train_models.py`) keeps raw `fwd_ret_1d` + its existing
  vol-scaled discretization thresholds — unchanged.

Single source of truth: `prepare_data` **and** the walk-forward IC sweep
(`_sweep_model_params`, `walk_forward.sweep_params`) must both read
`fwd_ret_vol_adj`, so the swept alpha matches the trained target.

**Consumer impact:** regressor predictions become vol-standardized scores, not
raw returns. The quantile decision layer only *ranks* predictions → signals
unaffected. For display (`predict_next_day_lite.py`, Discord), multiply the
prediction back by current `vol_20d` to recover an expected-return estimate,
preserving displayed semantics.

### #6 — Cleanup

- Fix the stale "25-feature" wording → "29-feature" in `daily_features.py`
  module docstring, `CLAUDE.md`, and `models/README.md`.
- Delete the dead line `feats[FEATURE_COLS] = feats[FEATURE_COLS].fillna(0.0)`
  (`daily_features.py:238`) — `dropna(subset=FEATURE_COLS)` already guarantees
  no NaN in those columns. Keep the `replace([inf, -inf], nan)` before it.

## Cross-cutting concerns

### Version

Bump `FEATURE_SET_NAME` → `daily_v5`. Old pickles become explicitly
incompatible via the existing feature-contract check
(`_load_validated_pickle`, `predictors/base.py:78`).

### Files touched

- `daily_features.py` — drop 2 features, add `_rolling_zscore` + apply, add
  `fwd_ret_vol_adj`, remove dead `fillna`, fix docstring, bump version.
- `predictors/base.py` — `_preprocess` keeps inf/nan only; add `_scale` + `CLIP`.
- `predictors/ridge.py`, `predictors/logistic.py`, `predictors/xgboost_pred.py`,
  `predictors/dqn.py` — use `_scale`.
- `train_models.py` — `_preprocess` drops clip; `StandardScaler` → `RobustScaler`;
  clip after scale.
- `train_predictor.py` — `RobustScaler`; train on `fwd_ret_vol_adj`; sweep
  target consistency; XGBoost scaler note.
- `walk_forward.py` — `RobustScaler`; clip; vol-adj target consistency.
- `predict_next_day_lite.py` — new scale/clip path; recover return for display.
- `train_hybrid.py` — its own `_preprocess`/scaler updated for consistency.
- `ml_strategies.py` — scaler path.
- Tests + docs (`CLAUDE.md`, `models/README.md`).

### Migration sequence

`FEATURE_SET_NAME` bumping invalidates every committed model. Between merging
code and finishing the retrain, the live GitHub Actions prediction would fail
the contract check. Sequence the work so code, retrain, and recommit land
together:

1. Land code changes.
2. Retrain all models (`train_models.py`, `train_predictor.py`, `train_dqn.py`,
   `train_hybrid.py`) on the configured universe.
3. Re-run the walk-forward threshold sweep; record new
   `best_signal_quantile` / `best_threshold_window`.
4. Regenerate `.sha256` integrity files.
5. Commit new pickles + hashes alongside the code so `main` is never in a
   half-migrated state.

## Testing

- **No look-ahead (#2):** for a sample symbol, the rolling z-score value at row
  `i` computed on the truncated series `[:i+1]` equals the value at row `i`
  computed on the full series (within tolerance). Extend
  `tests/test_data_leakage.py`.
- **Correlation guard (#1):** no pair in `FEATURE_COLS` has `|corr| > 0.98` on a
  representative sample; `williams_r` and `macd_hist` are absent.
- **Frozen-scaler + clip (#3/#4):** scaler fit on train, transform of a new
  batch uses train medians (not batch medians); values beyond `±CLIP` are
  bounded.
- **Vol target (#5):** `fwd_ret_vol_adj ≈ fwd_ret_1d / vol_20d`; the regressor
  is fit on this column, and the sweep reads the same column.
- **Feature contract:** `len(FEATURE_COLS) == 29`; `FEATURE_SET_NAME ==
  "daily_v5"`. Update `tests/test_feature_contract.py`, `tests/test_features.py`,
  `tests/test_config.py` as needed.
- **No NaN/inf** in output features (existing check still passes).

## Out of scope

- Per-date cross-sectional normalization (rejected: thin universe +
  universe-composition dependency).
- PCA / learned dimension reduction (dropping exact dups + a correlation guard
  is sufficient; revisit only if collinearity still hurts IC after retrain).
- Adding categorical / calendar encodings (no current need).
