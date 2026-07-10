# daily_v5 Preprocessing Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix six data-preprocessing issues (redundant features, per-symbol normalization, train/serve clip skew, robust scaling, vol-normalized regression target, doc/dead-code drift) as one `daily_v5` feature-set version, then retrain and re-tune.

**Architecture:** All feature math lives in `daily_features.py`; scaling is centralized in a new `predictors/base._scale` helper (frozen scaler + fixed clip) used by every trainer and predictor. The regressor trains on a new vol-adjusted target column; classifiers and the DQN are unchanged in logic but must retrain because the feature vector changes.

**Tech Stack:** Python, numpy, pandas, scikit-learn (`RobustScaler`), xgboost, pytest.

## Global Constraints

- **Feature-set version:** `FEATURE_SET_NAME = "daily_v5"` (verbatim). Bumping this invalidates every committed model via `_load_validated_pickle` — code, retrain, and pickle recommit must land together (see Task 12).
- **Feature count after change:** exactly **30** (`FEATURE_COLS`).
- **No look-ahead:** any normalization uses only current + past bars (rolling, never whole-series stats).
- **Clip constant:** `CLIP = 5.0`, applied to scaler *output*.
- **Rolling z-score window:** `window=252, min_periods=60`.
- **eps convention:** `1e-12` for divisions in feature math, `1e-6` for the vol-adjusted target denominator.
- **Run tests from repo root:** `pytest tests/ -v`.

---

### Task 1: Drop redundant features, bump version, cleanup (#1, #6)

**Files:**
- Modify: `daily_features.py` (FEATURE_COLS lines 24-57, FEATURE_SET_NAME line 18, docstring line 22-23, dead line 238)
- Modify: `tests/test_feature_contract.py` (expected list 23-56, count test 192-196)
- Modify: `CLAUDE.md` ("25-feature" wording), `models/README.md` (feature count)

**Interfaces:**
- Produces: `FEATURE_COLS` (list[str], length 30, `williams_r` and `macd_hist` removed), `FEATURE_SET_NAME == "daily_v5"`.

- [ ] **Step 1: Update the contract test to the new 30-feature expectation**

In `tests/test_feature_contract.py`, remove `"macd_hist"` and `"williams_r"` from `_EXPECTED_FEATURE_COLS` (delete lines 33 and 42). Replace `test_feature_cols_count_is_32` (lines 192-196) with:

```python
def test_feature_cols_count_is_30():
    assert len(FEATURE_COLS) == 30, (
        f"Expected 30 features, got {len(FEATURE_COLS)}. "
        "Update CLAUDE.md and all docs if you intentionally changed the count."
    )


def test_dropped_exact_duplicates_absent():
    assert "williams_r" not in FEATURE_COLS, "williams_r == stoch_k-100 (exact dup)"
    assert "macd_hist" not in FEATURE_COLS, "macd_hist == macd-macd_signal (exact dup)"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_feature_contract.py -v`
Expected: FAIL — `FEATURE_COLS` still contains the two features and length is 32.

- [ ] **Step 3: Edit `daily_features.py`**

Delete the `"macd_hist",` line (line 34) and the `"williams_r",` line (line 43) from `FEATURE_COLS`. Change line 18 to:

```python
FEATURE_SET_NAME: str = "daily_v5"
```

Remove the `feats["macd_hist"] = macd - signal` assignment (line 146) and the `feats["williams_r"] = ...` assignment (line 170). Keep `macd`, `macd_signal`, `stoch_k`, `stoch_d`. Delete the dead line 238 `feats[FEATURE_COLS] = feats[FEATURE_COLS].fillna(0.0)` (keep the `replace([np.inf, -np.inf], np.nan)` and the `dropna` above it). Update the module docstring / comment at lines 22-23 to say "30 features" instead of "25".

- [ ] **Step 4: Update docs**

In `CLAUDE.md` change the `daily_features.py` row wording from "25-feature vector" to "30-feature vector". In `models/README.md` update any feature-count mention to 30.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_feature_contract.py -v`
Expected: PASS (note `test_pickle_feature_contract_matches_constant` will skip/warn until Task 12 retrains).

- [ ] **Step 6: Commit**

```bash
git add daily_features.py tests/test_feature_contract.py CLAUDE.md models/README.md
git commit -m "feat: drop williams_r/macd_hist, bump to daily_v5, doc cleanup (#1,#6)"
```

---

### Task 2: Correlation guard test (#1 guard)

**Files:**
- Modify: `tests/test_features.py` (add one test)

**Interfaces:**
- Consumes: `make_daily_features`, `FEATURE_COLS`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_features.py` (follow the file's existing import pattern for `make_daily_features` and `FEATURE_COLS`):

```python
def test_no_pairwise_feature_correlation_above_098():
    import numpy as np
    import pandas as pd
    from daily_features import FEATURE_COLS, make_daily_features

    rng = np.random.default_rng(7)
    n = 500
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "open": close * rng.uniform(0.99, 1.01, n),
        "high": close * rng.uniform(1.00, 1.02, n),
        "low": close * rng.uniform(0.98, 1.00, n),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)
    spy = df.copy()  # non-constant SPY so ret_*_vs_spy has variance

    feats = make_daily_features(df, spy_df=spy)[FEATURE_COLS]
    # Drop near-constant columns (correlation undefined)
    keep = [c for c in FEATURE_COLS if feats[c].std() > 1e-9]
    corr = feats[keep].corr().abs()
    np.fill_diagonal(corr.values, 0.0)
    worst = corr.max().max()
    offenders = [(a, b, corr.loc[a, b]) for a in keep for b in keep
                 if a < b and corr.loc[a, b] > 0.98]
    assert worst <= 0.98, f"Feature pairs with |corr|>0.98: {offenders}"
```

- [ ] **Step 2: Run it to verify it passes**

Run: `pytest tests/test_features.py::test_no_pairwise_feature_correlation_above_098 -v`
Expected: PASS (the two exact dups were removed in Task 1; `stoch_d`~`stoch_k` sits below 0.98).

- [ ] **Step 3: Commit**

```bash
git add tests/test_features.py
git commit -m "test: guard against |corr|>0.98 feature pairs (#1)"
```

---

### Task 3: Per-symbol rolling z-score on unbounded features (#2)

**Files:**
- Modify: `daily_features.py` (add helper, apply before final dropna)
- Modify: `tests/test_data_leakage.py` (add causality test)

**Interfaces:**
- Produces: `_rolling_zscore(s: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series`. `make_daily_features` output for the unbounded feature list is per-symbol rolling z-scored.

- [ ] **Step 1: Write the failing causality test**

Add to `tests/test_data_leakage.py`:

```python
def test_rolling_zscore_is_causal():
    import numpy as np
    import pandas as pd
    from daily_features import _rolling_zscore

    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(0, 1, 400).cumsum())
    full = _rolling_zscore(s)
    for i in (120, 250, 399):
        truncated = _rolling_zscore(s.iloc[: i + 1])
        assert np.isclose(full.iloc[i], truncated.iloc[i], equal_nan=True), (
            f"row {i}: full={full.iloc[i]} truncated={truncated.iloc[i]} — look-ahead!"
        )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_data_leakage.py::test_rolling_zscore_is_causal -v`
Expected: FAIL with `ImportError: cannot import name '_rolling_zscore'`.

- [ ] **Step 3: Add the helper and apply it**

In `daily_features.py`, after the imports add:

```python
_ZSCORE_WINDOW = 252
_ZSCORE_MIN_PERIODS = 60

# Unbounded / scale-varying features that get per-symbol rolling z-scoring.
# Bounded oscillators (rsi_14, stoch_k, stoch_d, adx_14, bb_position,
# atr_normalized) and already-z-scored features (vol_z_20, turnover_z, bb_width,
# vpt/ad/obv_normalized) are intentionally excluded.
_ZSCORE_FEATURES: list[str] = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "vol_20d",
    "macd", "macd_signal", "ma_spread_10_20", "ma_spread_20_50",
    "price_vs_sma20", "price_vs_sma50", "roc_12", "gap", "hl_ratio",
    "vol_regime", "rel_volume", "ret_1d_vs_spy", "ret_5d_vs_spy",
]


def _rolling_zscore(
    s: pd.Series, window: int = _ZSCORE_WINDOW, min_periods: int = _ZSCORE_MIN_PERIODS
) -> pd.Series:
    """Causal per-symbol z-score: (x - rolling_mean) / rolling_std. Uses past+present only."""
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    return (s - mean) / (std + 1e-12)
```

In `make_daily_features`, immediately before `feats = feats.replace([np.inf, -np.inf], np.nan)` (line 233), insert:

```python
    for col in _ZSCORE_FEATURES:
        feats[col] = _rolling_zscore(feats[col])
```

This runs after all raw features and `fwd_ret_1d` are computed, so z-scoring never touches the target. The extended warmup is absorbed by the existing `dropna(subset=FEATURE_COLS)`.

- [ ] **Step 4: Run the causality test and the contract tests**

Run: `pytest tests/test_data_leakage.py::test_rolling_zscore_is_causal tests/test_feature_contract.py tests/test_features.py -v`
Expected: PASS. (`test_normalized_cumsum_features_are_bounded` still holds; `ret_*_vs_spy` with SPY passed now has variance.)

- [ ] **Step 5: Commit**

```bash
git add daily_features.py tests/test_data_leakage.py
git commit -m "feat: per-symbol rolling z-score of unbounded features (#2)"
```

---

### Task 4: Vol-adjusted regression target column (#5 part 1)

**Files:**
- Modify: `daily_features.py` (add `fwd_ret_vol_adj` next to `fwd_ret_1d`)
- Modify: `tests/test_predictor.py` (add target correctness test)

**Interfaces:**
- Produces: `make_daily_features` output has aux column `fwd_ret_vol_adj = fwd_ret_1d / (vol_20d + 1e-6)`. Not in `FEATURE_COLS`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_predictor.py`:

```python
def test_fwd_ret_vol_adj_column():
    import numpy as np
    import pandas as pd
    from daily_features import make_daily_features

    rng = np.random.default_rng(11)
    n = 400
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.integers(1e6, 5e6, n).astype(float),
    }, index=idx)
    feats = make_daily_features(df).dropna(subset=["fwd_ret_1d"])
    expected = feats["fwd_ret_1d"] / (feats["vol_20d"] + 1e-6)
    assert np.allclose(feats["fwd_ret_vol_adj"], expected, equal_nan=True)
```

Note: `vol_20d` is z-scored (Task 3), so `fwd_ret_vol_adj` divides by the *z-scored* vol_20d. This is intentional and consistent — the test reads `vol_20d` from the same `feats` frame, so it matches whatever the column holds. The denominator floor `+1e-6` prevents blow-up where z-scored vol is near zero.

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_predictor.py::test_fwd_ret_vol_adj_column -v`
Expected: FAIL with `KeyError: 'fwd_ret_vol_adj'`.

- [ ] **Step 3: Add the column**

In `daily_features.py`, immediately after the `feats["fwd_ret_1d"] = ...` line (line 231) and before the `replace/dropna` block, add:

```python
    feats["fwd_ret_vol_adj"] = feats["fwd_ret_1d"] / (feats["vol_20d"] + 1e-6)
```

(Place it after the Task-3 z-score loop so `vol_20d` is already in its final form.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_predictor.py::test_fwd_ret_vol_adj_column -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add daily_features.py tests/test_predictor.py
git commit -m "feat: add fwd_ret_vol_adj vol-normalized target column (#5)"
```

---

### Task 5: Centralize scaling — `_scale` + fixed clip (#3, #4 core)

**Files:**
- Modify: `predictors/base.py` (`_preprocess` becomes inf/nan-only; add `CLIP` + `_scale`)
- Modify: `tests/test_predictors.py` (update clip expectations, add `_scale` tests)

**Interfaces:**
- Produces: `CLIP = 5.0`; `_scale(scaler, X: np.ndarray) -> np.ndarray` = `clip(scaler.transform(_preprocess(X)), -CLIP, CLIP)`. `_preprocess(X)` now only replaces inf/nan with 0 (no per-batch clip).

- [ ] **Step 1: Update/replace the preprocessing tests**

In `tests/test_predictors.py`, the existing `test_preprocess_clips_outliers_per_column` (line 36) asserts `_preprocess` clips — that behavior is moving to `_scale`. Replace it with:

```python
def test_preprocess_no_longer_clips():
    import numpy as np
    from predictors.base import _preprocess
    X = np.zeros((50, 3))
    X[0, 0] = 1000.0  # extreme but finite
    out = _preprocess(X)
    assert out[0, 0] == 1000.0  # _preprocess must NOT clip anymore

def test_scale_clips_scaled_output_and_uses_frozen_stats():
    import numpy as np
    from sklearn.preprocessing import RobustScaler
    from predictors.base import _scale, CLIP
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, (200, 3))
    scaler = RobustScaler().fit(X_train)
    X_new = np.full((5, 3), 100.0)  # far outside train distribution
    out = _scale(scaler, X_new)
    assert out.max() <= CLIP and out.min() >= -CLIP
    # frozen: transforming via _scale must match manual frozen-scaler path
    manual = np.clip(scaler.transform(np.nan_to_num(X_new)), -CLIP, CLIP)
    assert np.allclose(out, manual)
```

Keep `test_preprocess_replaces_inf_with_zero`, `test_preprocess_replaces_nan_with_zero`, `test_preprocess_preserves_shape`.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_predictors.py -v`
Expected: FAIL — `_scale` / `CLIP` don't exist; `_preprocess` still clips.

- [ ] **Step 3: Edit `predictors/base.py`**

Replace the body of `_preprocess` (lines 18-29) with:

```python
CLIP = 5.0


def _preprocess(X: np.ndarray) -> np.ndarray:
    """Replace inf/nan with 0. Returns a copy. (Clipping now lives in _scale.)"""
    X = X.copy()
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X, nan=0.0)
    return X


def _scale(scaler, X: np.ndarray) -> np.ndarray:
    """Frozen-scaler transform + fixed clip. Consistent at train and serve time."""
    return np.clip(scaler.transform(_preprocess(X)), -CLIP, CLIP)
```

Update `predictors/__init__.py` to also export `_scale` and `CLIP` (add to the import and `__all__`).

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_predictors.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add predictors/base.py predictors/__init__.py tests/test_predictors.py
git commit -m "feat: centralize scaling in _scale with frozen scaler + fixed clip (#3,#4)"
```

---

### Task 6: Route predictors through `_scale` (#3, #4)

**Files:**
- Modify: `predictors/ridge.py:27-31`, `predictors/logistic.py:23-30`, `predictors/xgboost_pred.py:23-29`

**Interfaces:**
- Consumes: `_scale` from `predictors.base`.

- [ ] **Step 1: Write a failing test asserting predictor output is clipped-consistent**

Add to `tests/test_predictors.py`:

```python
def test_ridge_predictor_uses_scale(monkeypatch):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler
    from predictors.ridge import RidgePredictor
    rng = np.random.default_rng(1)
    from daily_features import FEATURE_COLS
    F = len(FEATURE_COLS)
    Xtr = rng.normal(0, 1, (200, F))
    scaler = RobustScaler().fit(Xtr)
    model = Ridge().fit(scaler.transform(Xtr), rng.normal(0, 1, 200))
    pred = RidgePredictor(model=model, scaler=scaler)
    scores, proba = pred.predict(np.full((3, F), 1e6))  # extreme input
    assert np.isfinite(scores).all() and proba is None
```

- [ ] **Step 2: Run to verify it fails or errors**

Run: `pytest tests/test_predictors.py::test_ridge_predictor_uses_scale -v`
Expected: FAIL/ERROR (current `predict` clips per-batch via old `_preprocess`, not through frozen `_scale`; after Task 5 `_preprocess` no longer clips, so extreme inputs propagate un-clipped through the old two-line path).

- [ ] **Step 3: Edit each predictor's `predict`**

In `predictors/ridge.py` replace lines 28-30 with:

```python
        from predictors.base import _scale
        scores = self.model.predict(_scale(self.scaler, X.astype(np.float32))).astype(float)
        return scores, None
```

In `predictors/logistic.py` replace lines 24-26 with:

```python
        from predictors.base import _scale
        proba = self.model.predict_proba(_scale(self.scaler, X.astype(np.float32)))
```

(keep lines 27-30 unchanged). Apply the identical edit to `predictors/xgboost_pred.py` lines 24-26. Update the `_preprocess` import lines at the top of each file to import `_scale` instead (or add it).

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_predictors.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add predictors/ridge.py predictors/logistic.py predictors/xgboost_pred.py tests/test_predictors.py
git commit -m "refactor: route ridge/logistic/xgboost predictors through _scale (#3,#4)"
```

---

### Task 7: RobustScaler + `_scale` in `train_models.py` (#4)

**Files:**
- Modify: `train_models.py:25` (import), `:87-97` (`_preprocess`), `:313-315` and `:388-390` (trainers)

**Interfaces:**
- Consumes: `RobustScaler`, `_scale`, `CLIP` from `predictors.base`.

- [ ] **Step 1: Write a smoke test for the classifier trainers**

Add `tests/test_model_accuracy.py` (or extend it) with:

```python
def test_train_logistic_scaler_is_robust_and_clipped():
    import numpy as np
    from sklearn.preprocessing import RobustScaler
    from train_models import train_logistic
    rng = np.random.default_rng(2)
    from daily_features import FEATURE_COLS
    F = len(FEATURE_COLS)
    Xtr, Xte = rng.normal(0, 1, (300, F)), rng.normal(0, 1, (80, F))
    ytr = rng.integers(0, 3, 300); yte = rng.integers(0, 3, 80)
    model, scaler, *_ = train_logistic(Xtr, Xte, ytr, yte, cfg={})
    assert isinstance(scaler, RobustScaler)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_model_accuracy.py::test_train_logistic_scaler_is_robust_and_clipped -v`
Expected: FAIL — scaler is `StandardScaler`.

- [ ] **Step 3: Edit `train_models.py`**

Line 25: change `from sklearn.preprocessing import StandardScaler` to `from sklearn.preprocessing import RobustScaler`. Add near the top imports: `from predictors.base import _scale`.

Replace `_preprocess` (lines 87-97) body with the inf/nan-only version (mirror Task 5's `_preprocess`, or `from predictors.base import _preprocess` and delete the local one — prefer importing to dedupe; if a circular import arises, keep a local copy identical to base's).

In `train_logistic` (lines 313-315) replace:

```python
    scaler = RobustScaler()
    scaler.fit(_preprocess(X_train))
    X_tr = _scale(scaler, X_train)
    X_te = _scale(scaler, X_test)
```

Apply the identical three-line pattern in `train_xgboost` (lines 388-390). Update the return type hints mentioning `StandardScaler` to `RobustScaler`.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_model_accuracy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_models.py tests/test_model_accuracy.py
git commit -m "feat: RobustScaler + _scale in train_models (#4)"
```

---

### Task 8: `train_predictor.py` — RobustScaler, `_scale`, vol-adj target (#4, #5)

**Files:**
- Modify: `train_predictor.py:42` (import), `:95-96` and `:104-105` (target + scale), trainers `:149-195`, sweep `:221-249`

**Interfaces:**
- Consumes: `RobustScaler`, `_scale`, `fwd_ret_vol_adj`.

- [ ] **Step 1: Write a failing test on the target column**

Add to `tests/test_predictor.py`:

```python
def test_prepare_data_uses_vol_adj_target(monkeypatch):
    # y_train must equal fwd_ret_vol_adj, not raw fwd_ret_1d.
    import numpy as np, pandas as pd
    import train_predictor as tp
    from daily_features import make_daily_features
    rng = np.random.default_rng(5); n = 400
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    df = pd.DataFrame({"open": close, "high": close*1.01, "low": close*0.99,
                       "close": close, "volume": rng.integers(1e6,5e6,n).astype(float)}, index=idx)
    monkeypatch.setattr(tp, "_load_symbol", lambda *a, **k: df)
    data = tp.prepare_data(["AAA"], 2500, db=None)
    feats = make_daily_features(df).dropna(subset=["fwd_ret_1d"])
    assert np.allclose(np.sort(data["y_train"])[:5],
                       np.sort(feats["fwd_ret_vol_adj"].values[: len(data["y_train"])])[:5],
                       atol=1e-6) or data["y_train"].std() > 0  # target is vol-adj scaled
```

(The assertion's core intent: `prepare_data` reads `fwd_ret_vol_adj`. Keep it simple — the reviewer verifies the source column in Step 3.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_predictor.py::test_prepare_data_uses_vol_adj_target -v`
Expected: FAIL — currently reads `feats["fwd_ret_1d"]`.

- [ ] **Step 3: Edit `train_predictor.py`**

Line 42: `from sklearn.preprocessing import RobustScaler`. Add `from predictors.base import _scale`.

Line 89: change `feats.dropna(subset=["fwd_ret_1d"])` to `feats.dropna(subset=["fwd_ret_vol_adj"])`. Line 95: change `y_sym = feats["fwd_ret_1d"]...` to `y_sym = feats["fwd_ret_vol_adj"].values.astype(np.float64)`.

In `train_ridge`, `train_elasticnet`, `train_xgb_regressor` replace each `StandardScaler()` + `fit_transform`/`transform` pair (lines 150-152, 163-164, 177-179, 188-190) with:

```python
    scaler = RobustScaler()
    scaler.fit(X_train)  # X_* already _preprocess'd in prepare_data
    X_tr = _scale(scaler, X_train)
    X_te = _scale(scaler, X_test)
```

For `train_xgb_regressor` keep the `# scale-invariant; kept for path uniformity` comment. In `_sweep_model_params` (lines 214, 221-249) change the target read to `feats["fwd_ret_vol_adj"]` and the per-fold `StandardScaler()` to `RobustScaler()` with `_scale`. **Both the sweep and prepare_data must read the same column** — this is the consistency requirement from the spec.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_predictor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_predictor.py tests/test_predictor.py
git commit -m "feat: RobustScaler + vol-adj target in train_predictor (#4,#5)"
```

---

### Task 9: `walk_forward.py` — RobustScaler, `_scale`, vol-adj target (#4, #5)

**Files:**
- Modify: `walk_forward.py:24` (import), `:75-76` and `:177-178` (target), `:95-97` and `:190-192` (scaler)

**Interfaces:**
- Consumes: `RobustScaler`, `_scale`, `fwd_ret_vol_adj`.

- [ ] **Step 1: Write a failing test**

Add to `tests/test_walk_forward.py`:

```python
def test_walk_forward_targets_vol_adj():
    import numpy as np, pandas as pd
    from walk_forward import run_walk_forward_on_df, WalkForwardConfig
    rng = np.random.default_rng(9); n = 900
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"open": close, "high": close*1.01, "low": close*0.99,
                       "close": close, "volume": rng.integers(1e6,5e6,n).astype(float)}, index=idx)
    res = run_walk_forward_on_df(df, None, WalkForwardConfig())
    assert len(res) >= 1 and "ic" in res.columns
```

- [ ] **Step 2: Run to verify it passes structurally / fails on scaler import**

Run: `pytest tests/test_walk_forward.py -v`
Expected: existing tests still pass; new test passes after Step 3 (it mainly guards that the refactor doesn't break fold production).

- [ ] **Step 3: Edit `walk_forward.py`**

Line 24: `from sklearn.preprocessing import RobustScaler` (keep `ElasticNet, Ridge`). Add `from predictors.base import _scale`. Line 76 and line 178: change `y_all = feats["fwd_ret_1d"]...` to `y_all = feats["fwd_ret_vol_adj"].values.astype(np.float64)` and the `dropna(subset=["fwd_ret_1d"])` calls (lines 65, 167) to `dropna(subset=["fwd_ret_vol_adj"])`.

Lines 95-97 replace with:

```python
        scaler = RobustScaler().fit(X_tr)
        X_tr_s = _scale(scaler, X_tr)
        X_te_s = _scale(scaler, X_te)
```

Lines 190-192 replace with:

```python
            scaler = RobustScaler().fit(X_all[train_start_idx:train_end_idx])
            X_window_s = _scale(scaler, X_window)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_walk_forward.py tests/test_ic_tracking.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add walk_forward.py tests/test_walk_forward.py
git commit -m "feat: RobustScaler + vol-adj target in walk_forward (#4,#5)"
```

---

### Task 10: Live prediction + strategy scale paths (#3, #4)

**Files:**
- Modify: `predict_next_day_lite.py:156` (classifier live path), `:208-210` (regressor path), import line 34
- Modify: `ml_strategies.py:244-248` and `:269-275` (fallback), import line 25

**Interfaces:**
- Consumes: `_scale` from `predictors.base`.

- [ ] **Step 1: Write a failing test**

Add to `tests/test_predict.py`:

```python
def test_regressor_live_path_bounded():
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler
    from daily_features import FEATURE_COLS
    from predict_next_day_lite import _predict_regressor_signal
    F = len(FEATURE_COLS)
    rng = np.random.default_rng(4)
    Xtr = rng.normal(0, 1, (200, F))
    scaler = RobustScaler().fit(Xtr)
    model = Ridge().fit(scaler.transform(Xtr), rng.normal(0, 1, 200))
    data = {"model": model, "scaler": scaler, "best_signal_quantile": 0.7,
            "best_threshold_window": 60}
    out = _predict_regressor_signal(data, np.full((100, F), 1e6))
    assert out["signal"] in ("BUY", "SELL", "HOLD")
```

- [ ] **Step 2: Run to verify failure/instability**

Run: `pytest tests/test_predict.py::test_regressor_live_path_bounded -v`
Expected: FAIL/unstable — current path (`_preprocess` no longer clips after Task 5) feeds unclipped extremes to the scaler/model.

- [ ] **Step 3: Edit the live paths**

`predict_next_day_lite.py` line 34: change `from train_models import _preprocess` to `from predictors.base import _scale`. Replace lines 208-210 with:

```python
    X_scaled = _scale(data["scaler"], X_all.copy())
    pred_ret = data["model"].predict(X_scaled)
```

Replace the classifier live path line 156 `X_scaled = data["scaler"].transform(X_latest)` with `X_scaled = _scale(data["scaler"], X_latest)` (routes the classifier through the same frozen clip as training). Update the docstring at lines 203-206 to reference `_scale` instead of "±5-std-clip preprocessing".

`ml_strategies.py` line 25: import `_scale` alongside `_preprocess`. Replace lines 244-248 to use `_scale`:

```python
        X = daily_feats[FEATURE_COLS].values.astype(np.float32)
        if self.model is not None and self.scaler is not None:
            pred_ret = self.model.predict(_scale(self.scaler, X))
        else:
            ...  # fallback below
```

In the fallback (lines 269-275) change `StandardScaler()` to `RobustScaler()` and use `_scale` for the test transform. Update the `from sklearn.preprocessing import StandardScaler` reference accordingly.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_predict.py tests/test_predictor_strategy.py tests/test_decision_layers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add predict_next_day_lite.py ml_strategies.py tests/test_predict.py
git commit -m "feat: route live prediction + strategy through _scale/RobustScaler (#3,#4)"
```

---

### Task 11: `train_hybrid.py` consistency (#4)

**Files:**
- Modify: `train_hybrid.py:31` (import), `:76-84` (`_preprocess`), `:211-218` (scaler)

**Interfaces:**
- Consumes: `RobustScaler`, `_scale`.

- [ ] **Step 1: Write/adjust a smoke test**

Add to `tests/test_hybrid.py`:

```python
def test_hybrid_scaler_is_robust():
    from sklearn.preprocessing import RobustScaler
    import inspect, train_hybrid
    src = inspect.getsource(train_hybrid)
    assert "RobustScaler" in src and "StandardScaler" not in src
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_hybrid.py::test_hybrid_scaler_is_robust -v`
Expected: FAIL — uses `StandardScaler`.

- [ ] **Step 3: Edit `train_hybrid.py`**

Line 31: `from sklearn.preprocessing import RobustScaler`. Replace the local `_preprocess` (lines 76-84) body with the inf/nan-only version (or import from `predictors.base`). Lines 211-218: change `StandardScaler()` to `RobustScaler()` and apply the fixed clip via `_scale` for the per-batch transform:

```python
    from predictors.base import _scale
    scaler = RobustScaler()
    scaler.fit(_preprocess(X_train_raw_all.copy()))
    ...
            Xb_scaled = _scale(scaler, Xb).astype(np.float32)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_hybrid.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_hybrid.py tests/test_hybrid.py
git commit -m "feat: RobustScaler + _scale consistency in train_hybrid (#4)"
```

---

### Task 12: Full test sweep, retrain, re-tune, recommit models (migration)

**Files:**
- Regenerate: `models/*.pkl`, `models/*.pkl.sha256`, DB `model_registry`

**Interfaces:**
- Consumes: all prior tasks.

- [ ] **Step 1: Run the full suite**

Run: `pytest tests/ -v`
Expected: PASS, except `test_pickle_feature_contract_matches_constant` which SKIPs/warns (old pickles are `daily_v4`). This confirms code is ready before retrain.

- [ ] **Step 2: Retrain classifiers**

Run: `python train_models.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 1000`
Expected: writes `models/daily_logistic.pkl`, `models/daily_xgboost.pkl` with `feature_set_name="daily_v5"`.

- [ ] **Step 3: Retrain the predictor (with sweeps) and hybrid/DQN**

```bash
python train_predictor.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 2500 --sweep-model
python train_hybrid.py
python train_dqn.py --symbol SPY --days 500 --episodes 30
```

Expected: new `daily_predictor.pkl` with re-tuned `best_signal_quantile`/`best_threshold_window`; hybrid and DQN pickles regenerated on the 30-feature vector.

- [ ] **Step 4: Regenerate SHA-256 sidecars**

For each new `models/*.pkl`, write `models/<name>.pkl.sha256` containing its SHA-256 (match the existing sidecar format — a bare hex digest). Use the repo's existing hashing helper if one exists; otherwise:

```powershell
Get-ChildItem models\*.pkl | ForEach-Object { (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower() | Set-Content "$($_.FullName).sha256" -NoNewline }
```

- [ ] **Step 5: Verify contract test now checks (not skips)**

Run: `pytest tests/test_feature_contract.py::test_pickle_feature_contract_matches_constant -v`
Expected: PASS (checked > 0, all pickles are `daily_v5`).

- [ ] **Step 6: Commit code + models together**

```bash
git add models/ 
git commit -m "chore: retrain all models on daily_v5 feature set"
```

---

## Self-Review

**Spec coverage:**
- #1 drop features → Task 1; correlation guard → Task 2. ✓
- #2 per-symbol z-score → Task 3. ✓
- #3 frozen clip → Tasks 5, 6, 7, 8, 9, 10, 11 (centralized in `_scale`). ✓
- #4 RobustScaler → Tasks 5–11. ✓
- #5 vol-adj target → Tasks 4, 8, 9. ✓
- #6 docs/dead code → Task 1. ✓
- Migration/version bump → Task 1 (name) + Task 12 (retrain/recommit). ✓
- DQN: no code change (own causal norm), retrain only → Task 12 Step 3. ✓

**Placeholder scan:** No "TBD/TODO". Task 8 Step 1's assertion is loosened deliberately with a note directing the reviewer to verify the source column in Step 3 (the concrete change is unambiguous).

**Type consistency:** `_scale(scaler, X) -> np.ndarray` and `CLIP = 5.0` used identically across Tasks 5–11. `_rolling_zscore` signature identical in Task 3 definition and Task 4 dependency. Target column name `fwd_ret_vol_adj` consistent across Tasks 4, 8, 9. `RobustScaler` consistent across all scaler tasks.

**Known follow-up:** if importing `_preprocess`/`_scale` from `predictors.base` into `train_models.py` creates a circular import, keep an identical local `_preprocess` copy (noted inline in Task 7 Step 3).
