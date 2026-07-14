# Step 2 — daily_v6 Orthogonal Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `daily_v5`'s feature set onto `main` as `daily_v6`, add two orthogonal features (Amihud illiquidity, VIX regime), and keep each change only if it raises out-of-sample IC without worsening PBO.

**Architecture:** Change *one thing at a time*. Hold the model (Ridge), target (`fwd_ret_1d`), and decision layer (#106 quantile + 1-bar shift) fixed so every task's effect on OOS IC is attributable to the feature/preprocessing change it makes. Each task is validated with the #106 referee (`walk_forward` IC, `cpcv` PBO, `eval_report` DSR) before it's kept.

**Tech Stack:** Python, pandas/numpy, scikit-learn (Ridge, StandardScaler/RobustScaler), scipy.stats (spearmanr), xgboost, pytest, yfinance.

## Global Constraints

- Base branch: `feat/step2-daily-v6-orthogonal`, off `origin/main` at `1444abb` (#106). Already checked out.
- `FEATURE_SET_NAME` becomes `daily_v6` (was `daily_v3` on main; the `daily_v5` branch name is deliberately skipped to disambiguate).
- **Held fixed (do NOT port from v5):** ElasticNet (keep Ridge), `fwd_ret_vol_adj` target (keep `fwd_ret_1d`), `compute_predictor_signal_raw_sign` / decision-layer rework (keep #106's quantile gating + the 1-bar `shift(1)` look-ahead fix in `ml_strategies.py:290`), `mean_reversion_strategy.py`, and the `config/default.yaml` execution/risk-knob edits. These are out of feature-scope and/or carry a known live/backtest divergence.
- **Keep-a-change gate:** median OOS walk-forward IC **rises** vs the prior task's baseline **and** PBO (`cpcv.cscv_pbo`) does not rise by more than **+0.05**. DSR + alpha/IR (`eval_report.py`) are recorded but non-blocking.
- **A null result is a valid outcome.** If `daily_v6` does not beat `daily_v3` on IC + PBO, ship the honest finding (Task 5's off-ramp), do not force an improvement.
- Fixed eval invocation used for every baseline/validation in this plan (keep identical so numbers are comparable):
  - IC: `python walk_forward.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 2500`
  - PBO + DSR: `python eval_report.py --symbols AAPL,MSFT,SPY,QQQ,NVDA --days 2500`
- Record every task's numbers in `docs/superpowers/plans/2026-07-14-step2-results.md` (created in Task 0) so the keep/drop decisions are auditable.

---

### Task 0: Capture the `daily_v3` baseline (reference numbers)

No code change. Establishes the numbers every later task must beat. The repo is on `daily_v3` (25 features) at this point.

**Files:**
- Create: `docs/superpowers/plans/2026-07-14-step2-results.md`

- [ ] **Step 1: Confirm a clean, green starting point**

Run: `pytest -q`
Expected: all pass (this is `main` at #106).

- [ ] **Step 2: Record baseline IC**

Run: `python walk_forward.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 2500`
Capture the printed per-symbol `Mean IC` / `Median IC` and the final `Parameter Sweep` line.

- [ ] **Step 3: Record baseline PBO + DSR**

Run: `python eval_report.py --symbols AAPL,MSFT,SPY,QQQ,NVDA --days 2500`
Capture `PBO = ...` and the per-symbol + median DSR lines.

- [ ] **Step 4: Write the results file**

Create `docs/superpowers/plans/2026-07-14-step2-results.md` with a table whose first row is `daily_v3 baseline` and columns: `stage | median walk-forward IC | PBO | median DSR | notes`. Paste the Step 2/3 numbers into that row.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-07-14-step2-results.md
git commit -m "docs: record daily_v3 baseline IC/PBO/DSR for step-2 comparison"
```

---

### Task 1: Port the `daily_v6` feature contract (v5 features + rolling-z)

Replace `daily_features.py` with the `daily_v5` branch version, minus the target change we are not adopting, and rename the version to `daily_v6`. Retrain every model on the new 29-feature contract (the repo is invalid until pickles match the contract). Scaler stays `StandardScaler`, model stays Ridge, target stays `fwd_ret_1d`.

**Files:**
- Modify: `daily_features.py` (whole-file replace + 2 edits)
- Modify: `tests/test_feature_contract.py`
- Modify: `tests/test_data_leakage.py`
- Create: `tests/test_feature_correlation_guard.py`
- Modify: `CLAUDE.md` (feature-count / version references)
- Modify (regenerate): `models/daily_logistic.pkl`, `models/daily_xgboost.pkl`, `models/daily_predictor.pkl`, `models/daily_hybrid.pkl` (+ their `.sha256`)

**Interfaces:**
- Produces: `daily_features.FEATURE_COLS` (29 cols, listed below), `FEATURE_SET_NAME == "daily_v6"`, `daily_features._rolling_zscore(s, window=252, min_periods=60)`, `daily_features._ZSCORE_FEATURES`. `make_daily_features(df, spy_df=None)` unchanged in signature; still returns a frame with `close` + `fwd_ret_1d` aux columns.

- [ ] **Step 1: Update the feature-contract test to the daily_v6 list (write it failing first)**

Replace `_EXPECTED_FEATURE_COLS`, the count test, and add the dropped-duplicates test in `tests/test_feature_contract.py`:

```python
_EXPECTED_FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "vol_20d",
    "ma_spread_10_20", "ma_spread_20_50", "macd", "macd_signal",
    "rsi_14", "price_vs_sma20", "price_vs_sma50", "bb_width", "bb_position",
    "stoch_k", "stoch_d", "roc_12", "atr_normalized", "adx_14",
    "vol_regime", "rel_volume", "hl_ratio", "turnover_z", "gap",
    "vpt_normalized", "ad_normalized", "obv_normalized",
    "ret_1d_vs_spy", "ret_5d_vs_spy",
]


def test_feature_cols_count_is_29():
    assert len(FEATURE_COLS) == 29, (
        f"Expected 29 features, got {len(FEATURE_COLS)}. Update CLAUDE.md and docs."
    )


def test_dropped_exact_duplicates_absent():
    assert "williams_r" not in FEATURE_COLS, "williams_r == stoch_k-100 (exact dup)"
    assert "macd_hist" not in FEATURE_COLS, "macd_hist == macd-macd_signal (exact dup)"
```

Also add, in the pickle-vs-constant test, the `feature_set_name` skip guard so `daily_v3` pickles are warned-and-skipped rather than hard-failing during the retrain window (copy the block from the `daily_v5` branch version of this test — it inserts a `pickle_fsn = data.get("feature_set_name")` check before the `assert data["feature_contract"] == FEATURE_COLS`).

- [ ] **Step 2: Run it to confirm it fails**

Run: `pytest tests/test_feature_contract.py::test_feature_cols_count_is_29 -v`
Expected: FAIL (repo still has 25 `daily_v3` features).

- [ ] **Step 3: Replace `daily_features.py` with the v5 version**

```bash
git show origin/feat/daily-v5-preprocessing:daily_features.py > daily_features.py
```

- [ ] **Step 4: Apply the two scope edits to `daily_features.py`**

Edit A — bump the version:
```python
# from:
FEATURE_SET_NAME: str = "daily_v5"
# to:
FEATURE_SET_NAME: str = "daily_v6"
```

Edit B — drop the vol-adjusted target we are NOT adopting. Delete these lines (keep `fwd_ret_1d` as the only target):
```python
    # Capture raw vol_20d before the z-score loop (the z-scored vol_20d is
    # still a valid model feature; only the target denominator needs raw vol).
    raw_vol_20d = feats["vol_20d"].copy()
```
and
```python
    # ponytail: divides by RAW vol_20d, not the z-scored column (which
    # crosses zero). The z-scored vol_20d remains a model feature — only
    # the target denominator changes. See task-12-fix-brief.
    feats["fwd_ret_vol_adj"] = feats["fwd_ret_1d"] / (raw_vol_20d + 1e-6)
```
Leave the `for col in _ZSCORE_FEATURES: feats[col] = _rolling_zscore(feats[col])` loop in place — it must still run.

- [ ] **Step 5: Verify contract tests pass**

Run: `pytest tests/test_feature_contract.py -v`
Expected: PASS (29 features, `daily_v6`, duplicates absent).

- [ ] **Step 6: Port the causal rolling-zscore leakage test**

Add to `tests/test_data_leakage.py` (verbatim from the v5 branch):

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

Run: `pytest tests/test_data_leakage.py::test_rolling_zscore_is_causal -v`
Expected: PASS.

- [ ] **Step 7: Add the correlation guard test (spec requirement)**

Create `tests/test_feature_correlation_guard.py`:

```python
"""Guard: no two FEATURE_COLS may be near-perfect linear duplicates."""
import numpy as np
import pandas as pd

from daily_features import FEATURE_COLS, make_daily_features


def _synthetic_ohlcv(n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    openp = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": openp, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def test_no_feature_pair_exceeds_098_corr():
    feats = make_daily_features(_synthetic_ohlcv())[FEATURE_COLS]
    corr = feats.corr().abs()
    np.fill_diagonal(corr.values, 0.0)
    worst = corr.stack().sort_values(ascending=False)
    top_pair, top_val = worst.index[0], float(worst.iloc[0])
    assert top_val <= 0.98, f"{top_pair} corr={top_val:.3f} > 0.98 — near-duplicate feature"
```

Run: `pytest tests/test_feature_correlation_guard.py -v`
Expected: PASS. (If it fails, the offending pair is a real redundancy — stop and report; do not silently drop.)

- [ ] **Step 8: Update `tests/test_features.py` count/NaN expectations**

If `test_features.py` asserts a specific feature count or lists column names, update them to the 29-name `daily_v6` list from Step 1. Run: `pytest tests/test_features.py -v` → PASS.

- [ ] **Step 9: Retrain all models on `daily_v6`**

```bash
python train_models.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 1000
python train_predictor.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 2500
python train_hybrid.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 1000
```
Note: `models/dqn_agent.pt` is intentionally NOT retrained — its 500-dim state (20 bars × 25 feats) no longer matches 29 feats, so it will fail its load check and be skipped by `predict_next_day_lite`. It is not in `prediction.models`, so this is acceptable; delete the stale file if preferred.

- [ ] **Step 10: Full test suite green**

Run: `pytest -q`
Expected: all pass (retrained pickles now match the `daily_v6` contract).

- [ ] **Step 11: Update `CLAUDE.md`**

Change "25-feature vector" / `daily_v3` references to "29-feature vector" / `daily_v6` in `CLAUDE.md` (the `daily_features.py` row of the Key files table and the `test_feature_contract.py` line).

- [ ] **Step 12: Commit**

```bash
git add daily_features.py tests/ CLAUDE.md models/
git commit -m "feat: port daily_v5 feature set as daily_v6 (Ridge/StandardScaler/fwd_ret_1d held fixed)"
```

- [ ] **Step 13: Validate against the Task 0 baseline**

Run the two Global-Constraints eval commands. Append a `daily_v6 (features only)` row to `2026-07-14-step2-results.md` with median IC / PBO / median DSR. Apply the gate vs the `daily_v3` row:
- IC up and PBO not worse by >0.05 → keep, proceed to Task 2.
- Otherwise → record the null finding; proceed to Task 5's off-ramp evaluation before deciding to keep or revert.
Commit the results update: `git add docs/ && git commit -m "docs: record daily_v6 features-only IC/PBO/DSR"`.

---

### Task 2: Add Amihud illiquidity feature (self-contained)

Cheapest orthogonal add — no new data source. `|ret_1d| / dollar_volume`, then per-symbol rolling z-score. Feature count 29 → 30.

**Files:**
- Modify: `daily_features.py`
- Modify: `tests/test_feature_contract.py`, `tests/test_data_leakage.py`, `tests/test_feature_correlation_guard.py` (count 29 → 30)
- Modify (regenerate): all four model pickles

**Interfaces:**
- Produces: new `FEATURE_COLS` entry `"amihud_illiq"` (rolling-z'd), count 30.

- [ ] **Step 1: Write the failing contract test**

In `tests/test_feature_contract.py`: append `"amihud_illiq"` to `_EXPECTED_FEATURE_COLS` (after `"turnover_z"`) and change `test_feature_cols_count_is_29` → `test_feature_cols_count_is_30` asserting `== 30`.

Run: `pytest tests/test_feature_contract.py::test_feature_cols_count_is_30 -v` → FAIL.

- [ ] **Step 2: Add the feature to `daily_features.py`**

In `FEATURE_COLS`, add `"amihud_illiq",` immediately after `"turnover_z",`. In `_ZSCORE_FEATURES`, add `"amihud_illiq"`. In `make_daily_features`, right after the turnover block (which already computes `dollar_vol = df["close"] * df["volume"]`), insert:

```python
    # --- Amihud illiquidity: |return| per dollar traded (price impact) ---
    # Raw magnitude is tiny/heavy-tailed; the _ZSCORE_FEATURES loop normalizes it.
    feats["amihud_illiq"] = feats["ret_1d"].abs() / (dollar_vol + 1e-12)
```

(Placed before the `for col in _ZSCORE_FEATURES` loop, so `ret_1d` is still raw here and `amihud_illiq` gets z-scored by the loop.)

- [ ] **Step 3: Contract + guard tests pass**

Run: `pytest tests/test_feature_contract.py tests/test_feature_correlation_guard.py -v`
Expected: PASS (count 30; `amihud_illiq` not >0.98 correlated with `turnover_z` — if it is, report and stop).

- [ ] **Step 4: Add a causality assertion**

In `tests/test_data_leakage.py`, add:

```python
def test_amihud_is_causal():
    import numpy as np
    from daily_features import make_daily_features
    from tests.test_feature_correlation_guard import _synthetic_ohlcv

    df = _synthetic_ohlcv(n=500, seed=7)
    full = make_daily_features(df)["amihud_illiq"]
    trunc = make_daily_features(df.iloc[:300])["amihud_illiq"]
    common = full.index.intersection(trunc.index)[-1]
    assert np.isclose(full.loc[common], trunc.loc[common], equal_nan=True)
```

Run: `pytest tests/test_data_leakage.py::test_amihud_is_causal -v` → PASS.

- [ ] **Step 5: Retrain + full suite**

Re-run the three training commands from Task 1 Step 9, then `pytest -q` → all pass.

- [ ] **Step 6: Commit + validate against the Task 1 baseline**

```bash
git add daily_features.py tests/ models/
git commit -m "feat: add Amihud illiquidity feature (daily_v6, 30 cols)"
```
Run the two eval commands; append an `+amihud` row to the results file. Apply the gate vs the `daily_v6 (features only)` row: keep the feature if IC up and PBO not worse by >0.05; otherwise revert this task's `FEATURE_COLS`/feature additions (git revert the commit), retrain, and record the null. Commit the results note.

---

### Task 3: Add VIX implied-vol regime feature (needs `^VIX` data)

Two columns: z-scored VIX level and z-scored 5-day VIX change. Market-wide series, loaded once and threaded through callers exactly like the existing `spy_df`. Feature count 30 → 32.

**Files:**
- Modify: `daily_features.py` (signature `make_daily_features(df, spy_df=None, vix_df=None)`)
- Modify (thread `vix_df`): `train_predictor.py` (`prepare_data`), `train_models.py` (feature-build call site), `walk_forward.py` (`run_walk_forward_on_df`, `build_fold_data`, `sweep_params`, CLI `main`), `predict_next_day_lite.py` (`main` loads VIX, `predict_symbol` passes it)
- Modify: `tests/test_feature_contract.py`, `tests/test_data_leakage.py` (count 30 → 32)
- Modify (regenerate): all four model pickles

**Interfaces:**
- Consumes: `data_loader.load_yfinance("^VIX", ...)` / the existing `_load_symbol` / `_load_bars_cached` helpers.
- Produces: `make_daily_features(df, spy_df=None, vix_df=None)`; new `FEATURE_COLS` entries `"vix_z"`, `"vix_chg_5d"` (rolling-z'd), count 32. When `vix_df is None`, both columns are `0.0` (same contract-satisfying fallback as the `_vs_spy` features).

- [ ] **Step 1: Failing contract test**

In `tests/test_feature_contract.py`: append `"vix_z", "vix_chg_5d"` to `_EXPECTED_FEATURE_COLS` (end of list) and rename the count test to `test_feature_cols_count_is_32` asserting `== 32`.

Run: `pytest tests/test_feature_contract.py::test_feature_cols_count_is_32 -v` → FAIL.

- [ ] **Step 2: Add the feature + fallback to `daily_features.py`**

Signature: `def make_daily_features(df, spy_df=None, vix_df=None):`. Add `"vix_z", "vix_chg_5d"` to the end of `FEATURE_COLS` and both to `_ZSCORE_FEATURES`. Near the `spy_df` block, add:

```python
    if vix_df is not None:
        vix_close = vix_df["close"].reindex(df.index).ffill()   # ffill = past-only, causal
        feats["vix_z"] = vix_close
        feats["vix_chg_5d"] = vix_close.pct_change(5)
    else:
        feats["vix_z"] = 0.0
        feats["vix_chg_5d"] = 0.0
```

(The `_ZSCORE_FEATURES` loop turns `vix_z` into a rolling z-score of the VIX level and `vix_chg_5d` into a z-score of its 5-day change. The `0.0` fallback z-scores to `0.0`, satisfying the contract when VIX is unavailable.)

- [ ] **Step 3: Thread `vix_df` through the training/eval callers**

Canonical pattern (from `train_predictor.prepare_data`) — load once near the `spy_df` load, pass at the feature-build call:

```python
    spy_df = _load_symbol("SPY", start, end, db)
    vix_df = _load_symbol("^VIX", start, end, db)     # NEW
    ...
        spy_arg = spy_df if symbol != "SPY" else None
        feats = make_daily_features(df, spy_df=spy_arg, vix_df=vix_df)   # add vix_df
```

Apply the same two-line change (load `^VIX` once; pass `vix_df=` into every `make_daily_features(...)` that already passes `spy_df=`) at these exact sites:
- `train_models.py` — the pooled feature-build loop (mirror of `prepare_data`).
- `walk_forward.py::run_walk_forward_on_df` — accept an optional `vix_df` param and pass it; `build_fold_data` and `sweep_params` load `^VIX` alongside their existing `spy_df = _load_symbol("SPY", ...)` and pass it; `main` loads `^VIX` and passes it to `run_walk_forward_on_df`.
- `predict_next_day_lite.py::main` — load VIX once via `_load_bars_cached("^VIX", start, end, db=db)` next to the existing `spy_df` load, pass it into `predict_symbol(...)`, which forwards `vix_df=` into its `make_daily_features(df, spy_df=spy_arg, vix_df=vix_df)` call.

Note: `DailyPredictorStrategy.signal` (backtest, used by `eval_report` DSR) calls `make_daily_features(df)` with no market series — `vix_z`/`vix_chg_5d` fall back to `0.0` there, exactly as `spy` already does. This is consistent; the IC gate is measured via `walk_forward`, which does pass the series.

- [ ] **Step 4: Contract + causality tests**

Run: `pytest tests/test_feature_contract.py tests/test_feature_correlation_guard.py -v` → PASS (32 cols).
Add to `tests/test_data_leakage.py`:

```python
def test_vix_features_are_causal():
    import numpy as np, pandas as pd
    from daily_features import make_daily_features
    from tests.test_feature_correlation_guard import _synthetic_ohlcv

    df = _synthetic_ohlcv(n=500, seed=11)
    vix = _synthetic_ohlcv(n=500, seed=12).rename(columns=str)  # any close series
    full = make_daily_features(df, vix_df=vix)[["vix_z", "vix_chg_5d"]]
    trunc = make_daily_features(df.iloc[:300], vix_df=vix.iloc[:300])[["vix_z", "vix_chg_5d"]]
    common = full.index.intersection(trunc.index)[-1]
    assert np.allclose(full.loc[common].values, trunc.loc[common].values, equal_nan=True)
```

Run: `pytest tests/test_data_leakage.py::test_vix_features_are_causal -v` → PASS.

- [ ] **Step 5: Retrain + full suite**

Re-run the three training commands, then `pytest -q` → all pass.

- [ ] **Step 6: Commit + validate against the Task 2 baseline**

```bash
git add daily_features.py train_predictor.py train_models.py walk_forward.py predict_next_day_lite.py tests/ models/
git commit -m "feat: add VIX implied-vol regime features (daily_v6, 32 cols)"
```
Run the two eval commands; append a `+vix` row. Apply the gate vs the prior kept row. If null, revert this commit + retrain and record it. Commit the results note.

---

### Task 4: (Separable) Adopt RobustScaler + fixed-clip preprocessing

The `daily_v5` "preprocessing" change, isolated so its marginal effect is measured on its own. Moves clipping to *after* a frozen-scaler transform (train/serve consistent) and swaps StandardScaler → RobustScaler. Kept only if it does not hurt the gate.

**Files:**
- Modify: `predictors/base.py` (add `CLIP`, `_scale`; slim `_preprocess`)
- Modify (StandardScaler→RobustScaler, `_preprocess`+`transform` → `_scale`): `train_models.py`, `train_predictor.py`, `walk_forward.py`, `ml_strategies.py` (`DailyPredictorStrategy.signal` predict branch only — NOT the decision layer), `predict_next_day_lite.py` (`_predict_classifier_signal`, `_predict_regressor_signal`, `_predict_hybrid_signal`), `predictors/logistic.py`, `predictors/ridge.py`, `predictors/xgboost_pred.py`
- Modify (regenerate): all four model pickles

**Interfaces:**
- Produces: `predictors.base._scale(scaler, X) -> np.ndarray` = `np.clip(scaler.transform(_preprocess(X)), -CLIP, CLIP)`; `_preprocess` now only does inf/nan → 0 (no clip).

- [ ] **Step 1: Add `_scale` + `CLIP`, slim `_preprocess` in `predictors/base.py`**

Copy the `CLIP`, `_preprocess`, and `_scale` definitions from `git show origin/feat/daily-v5-preprocessing:predictors/base.py` (also adds the `feature_set_name` soft-mismatch branch in `_load_validated_pickle` — take that too).

- [ ] **Step 2: Write a `_scale` train/serve-consistency test (failing first)**

Create `tests/test_scale_consistency.py`:

```python
import numpy as np
from sklearn.preprocessing import RobustScaler
from predictors.base import _scale, CLIP


def test_scale_is_frozen_and_clipped():
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, (200, 5))
    scaler = RobustScaler().fit(X_train)
    X_new = rng.normal(0, 1, (10, 5))
    out = _scale(scaler, X_new)
    # frozen transform: same input -> same output regardless of batch
    assert np.allclose(out, _scale(scaler, X_new))
    assert out.max() <= CLIP + 1e-9 and out.min() >= -CLIP - 1e-9
```

Run: `pytest tests/test_scale_consistency.py -v` → FAIL until Step 1 is imported correctly, then PASS.

- [ ] **Step 3: Switch call sites to RobustScaler + `_scale`**

At each site listed in **Files**, replace `StandardScaler()` with `RobustScaler()` and replace the `_preprocess(X)` + `scaler.transform(...)` pair with `_scale(scaler, X)`. Use the `daily_v5` branch versions of `predictors/logistic.py`, `predictors/ridge.py`, `predictors/xgboost_pred.py`, `train_models.py`, `train_predictor.py` as the reference for the exact per-file edits, but **do not** take their `walk_forward.py`/`ml_strategies.py` wholesale (those carry the ElasticNet + raw-sign changes we are excluding) — hand-edit only the scaler/`_scale` lines in `walk_forward.py` and in `DailyPredictorStrategy.signal`'s predict branch.

- [ ] **Step 4: Retrain + full suite**

Re-run the three training commands, then `pytest -q` → all pass.

- [ ] **Step 5: Commit + validate**

```bash
git add predictors/ train_models.py train_predictor.py walk_forward.py ml_strategies.py predict_next_day_lite.py tests/ models/
git commit -m "feat: RobustScaler + fixed-clip train/serve-consistent preprocessing"
```
Run the two eval commands; append a `+robustscaler` row. Gate vs the prior kept row. **This task is the most droppable** — if it doesn't help, `git revert` it and keep StandardScaler. Record the decision.

---

### Task 5: Finalize — re-tune, record honest numbers, docs, off-ramp

**Files:**
- Modify (regenerate with tuned `(q,w)`): `models/daily_predictor.pkl` (+ `.sha256`)
- Modify: `models/README.md`, `CLAUDE.md`, `docs/superpowers/plans/2026-07-14-step2-results.md`

- [ ] **Step 1: Re-tune the decision-layer `(q, w)` on the final feature set**

The `train_predictor.py` run already calls `walk_forward.sweep_params` and stores `best_signal_quantile` / `best_threshold_window` in the pickle. Confirm the final `train_predictor.py` run's log shows the sweep result; if you retrained before the last feature landed, re-run it once more so the stored `(q,w)` matches the final contract.

- [ ] **Step 2: Produce the final honest eval**

Run both Global-Constraints eval commands one last time on the final kept feature set. These are the numbers that go in the README.

- [ ] **Step 3: Rewrite the model cards**

In `models/README.md`, update the `daily_logistic` / `daily_xgboost` / `daily_predictor` / `daily_hybrid` cards: feature set `daily_v6`, feature count, and the final OOS IC / PBO / DSR / alpha-IR from Step 2. Keep the honest-caveat framing (state plainly whether `daily_v6` beat `daily_v3` or not).

- [ ] **Step 4: Finalize the results table + verdict**

In `2026-07-14-step2-results.md`, add a final `verdict` section stating, per the gate, which features were kept and whether the `daily_v6` signal beats the `daily_v3` baseline.

- [ ] **Step 5: Off-ramp if null**

If the final kept set does **not** beat `daily_v3` on IC + PBO: keep `daily_v6` on the branch (the feature hygiene + tests are still worth landing) but state explicitly in the README that live `prediction.models` performance is not improved, and recommend against widening capital reliance. Do **not** fabricate an improvement. If `daily_v6` is *worse*, recommend the branch not be merged to `main` and record why.

- [ ] **Step 6: Commit**

```bash
git add models/ docs/ CLAUDE.md
git commit -m "docs: finalize daily_v6 honest eval (IC/PBO/DSR) and model cards"
```

---

## Self-Review

**Spec coverage:**
- Port v5 feature+preprocessing onto main → Task 1 (features + rolling-z) + Task 4 (RobustScaler, isolated). ✓
- Re-validate under #106 referee → Task 0 baseline + per-task `walk_forward`/`eval_report` gates. ✓
- Two new orthogonal features (VIX, Amihud) → Task 2 (Amihud), Task 3 (VIX). ✓
- Reversal dropped / earnings deferred → not in plan. ✓
- Gate = OOS IC up AND PBO not worse by >0.05; DSR/alpha non-blocking → Global Constraints + every validate step. ✓
- `daily_v6` bump + retrain all models → Task 1 Step 4/9, repeated on each feature task. ✓
- Hold model/target/decision fixed → Global Constraints; #106 `ml_strategies.py` shift-fix explicitly preserved. ✓
- DQN state-dim break → Task 1 Step 9 note. ✓
- Honest null off-ramp → Task 5 Step 5. ✓
- Tests: contract, no-NaN, leakage/causality, corr guard → Tasks 1–4. ✓

**Placeholder scan:** No TBD/TODO; every code step shows code; every run step shows command + expected result. ✓

**Type/name consistency:** `make_daily_features(df, spy_df=None, vix_df=None)` introduced in Task 3 and used consistently thereafter; `_scale(scaler, X)` defined in Task 4 Step 1 and used in Steps 3; `FEATURE_COLS` count walked 29 → 30 → 32 across Tasks 1–3 with matching test names (`test_feature_cols_count_is_29/30/32`). ✓
