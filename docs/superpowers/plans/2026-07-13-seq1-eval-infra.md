# Seq-1 Honest-Eval Infra + Look-Ahead Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the daily_predictor backtest look-ahead and add overfitting-aware evaluation (deflated Sharpe, CPCV/PBO, buy-and-hold benchmark), then re-baseline the honest numbers.

**Architecture:** Three standalone pure-function modules (`deflated_sharpe.py`, `cpcv.py`, plus a matrix-builder refactor in `walk_forward.py`) feed a thin `eval_report.py` CLI. The look-ahead fix is a one-line change in the backtest strategy; the benchmark is an optional arg threaded into `compute_metrics`. No new dependencies.

**Tech Stack:** Python 3, NumPy, SciPy (`scipy.stats.norm`, already used), pandas, pytest. All already in the project.

## Global Constraints

- Feature ordering always via `daily_features.FEATURE_COLS`; never by column position.
- The shared decision function `ml_strategies.compute_predictor_signal` is the single source of truth for the daily_predictor signal — do NOT fork its logic.
- Backtest execution lag (`.shift(1)`) belongs in the backtest strategy wrapper, never in `compute_predictor_signal` (the live path must stay unshifted).
- No network access in the test suite — unit tests use synthetic/fixture data only. Real-data fetch happens only in the final re-baseline task, run manually.
- Deflated Sharpe uses **per-period (non-annualized)** Sharpe internally and consistently.
- `compute_metrics` changes must be backward-compatible: the new arg defaults to `None` and omits the new keys when absent.
- Preserve `sweep_params`' existing selection and fallback semantics exactly (pinned by `tests/test_walk_forward.py`).
- Run `pytest tests/ -q` green before every commit.

---

### Task 1: Fix daily_predictor backtest look-ahead

**Files:**
- Modify: `ml_strategies.py:290` (`DailyPredictorStrategy.signal` return)
- Test: `tests/test_predictor.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `DailyPredictorStrategy.signal` now returns a signal lagged one bar (matches `PredictorStrategy.signal`). No signature change.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_predictor.py`:

```python
def test_predictor_strategy_applies_one_bar_execution_lag():
    """Backtest signal must be lagged one bar vs the raw decision, matching
    PredictorStrategy (predictor_strategy.py:45). Guards the close[t]->close[t]
    look-ahead fix. Live path (_predict_regressor_signal) is intentionally NOT
    shifted and is not exercised here."""
    import numpy as np
    from base_strategy import StrategyConfig
    from ml_strategies import DailyPredictorStrategy

    df = _make_price_df(n=200)
    cfg = StrategyConfig(name="daily_predictor", holding_period=0)
    strat = DailyPredictorStrategy(cfg, use_pretrained=False, threshold_window=20)

    # First non-HOLD signal index must be strictly greater than it would be
    # without the shift: the shift guarantees element 0 is always HOLD (0).
    sig = strat.signal(None, df)
    assert sig.iloc[0] == 0, "shift(1) must force the first bar to HOLD"
    assert len(sig) == len(make_daily_features(df))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_predictor.py::test_predictor_strategy_applies_one_bar_execution_lag -v`
Expected: FAIL — currently the first bar can be non-zero (no shift), or the assertion on `iloc[0] == 0` is not guaranteed.

- [ ] **Step 3: Apply the fix**

In `ml_strategies.py`, `DailyPredictorStrategy.signal`, change the final return from:

```python
        return self._apply_holding_period(pd.Series(signals, index=daily_feats.index))
```

to:

```python
        raw = self._apply_holding_period(pd.Series(signals, index=daily_feats.index))
        # Execution lag: decide on close[t], trade at close[t+1]. Matches
        # PredictorStrategy.signal (predictor_strategy.py:45). Backtest-only —
        # the live path (_predict_regressor_signal) takes signals[-1] and trades
        # the next session, so it is already correct and must NOT be shifted.
        return raw.shift(1).fillna(0).astype(int)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_predictor.py -v`
Expected: PASS — the new lag test passes, and the pre-existing predictor tests (value-set `{-1,0,1}` and all-HOLD prefix) still pass (both are shift-invariant).

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add ml_strategies.py tests/test_predictor.py
git commit -m "fix: apply 1-bar execution lag to DailyPredictorStrategy backtest signal"
```

---

### Task 2: Buy-and-hold benchmark row in compute_metrics

**Files:**
- Modify: `simulation_pipeline.py` (`compute_metrics` signature + body; `Backtester.run` call site)
- Test: `tests/test_metrics_benchmark.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `compute_metrics(equity, trades, benchmark_close=None)`. When `benchmark_close` (a `pd.Series` indexed like `equity`) is given, the returned dict gains `benchmark_return_pct: float`, `alpha_pct: float`, `information_ratio: float`. When `None`, output is unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics_benchmark.py`:

```python
import numpy as np
import pandas as pd
from simulation_pipeline import compute_metrics


def _series(vals, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="D")
    return pd.Series(vals, index=idx)


def test_benchmark_keys_absent_when_no_benchmark():
    equity = _series([100.0, 101.0, 102.0, 103.0])
    m = compute_metrics(equity, pd.DataFrame())
    assert "benchmark_return_pct" not in m
    assert "alpha_pct" not in m


def test_benchmark_alpha_positive_when_strategy_beats_hold():
    # Strategy doubles; benchmark price rises 10%.
    equity = _series([100.0, 120.0, 150.0, 200.0])
    bench = _series([50.0, 52.0, 54.0, 55.0])
    m = compute_metrics(equity, pd.DataFrame(), benchmark_close=bench)
    assert m["benchmark_return_pct"] == pytest_approx(10.0)
    assert m["alpha_pct"] == pytest_approx(m["total_return_pct"] - 10.0)
    assert "information_ratio" in m


def test_benchmark_alpha_negative_when_strategy_lags_hold():
    equity = _series([100.0, 100.5, 101.0, 101.0])   # +1%
    bench = _series([50.0, 55.0, 60.0, 65.0])          # +30%
    m = compute_metrics(equity, pd.DataFrame(), benchmark_close=bench)
    assert m["alpha_pct"] < 0


# local helper to avoid a pytest.approx import line at top
def pytest_approx(x, tol=1e-6):
    import pytest
    return pytest.approx(x, abs=tol)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics_benchmark.py -v`
Expected: FAIL — `compute_metrics` does not accept `benchmark_close`.

- [ ] **Step 3: Modify compute_metrics**

In `simulation_pipeline.py`, change the signature:

```python
def compute_metrics(equity: pd.Series, trades: pd.DataFrame,
                    benchmark_close: Optional[pd.Series] = None) -> Dict[str, Any]:
```

At the end of `compute_metrics`, build the base `result` dict as today, then before `return result` insert:

```python
    if benchmark_close is not None and len(benchmark_close) > 1:
        bench = benchmark_close.reindex(equity.index).dropna()
        if len(bench) > 1:
            bench_daily = bench.resample("1D").last().dropna()
            if len(bench_daily) > 1:
                bench_total = float((bench_daily.iloc[-1] / bench_daily.iloc[0] - 1) * 100)
                bench_ret = bench_daily.pct_change().dropna()
                excess = (daily_ret.reindex(bench_ret.index).fillna(0.0) - bench_ret).dropna()
                ir = (float(np.sqrt(252) * excess.mean() / excess.std())
                      if excess.std() and excess.std() != 0 else 0.0)
                result["benchmark_return_pct"] = bench_total
                result["alpha_pct"] = result["total_return_pct"] - bench_total
                result["information_ratio"] = ir
    return result
```

(Assign the current `return {...}` literal to a local `result` variable first, so the block above can add keys. `daily_ret` is already computed at the top of the function.)

- [ ] **Step 4: Thread the benchmark through Backtester.run**

In `simulation_pipeline.py`, `Backtester.run`, change the metrics call:

```python
        metrics = compute_metrics(equity_series, trades_df)
```

to:

```python
        # Benchmark = buy-and-hold the traded symbol itself over the same dates.
        metrics = compute_metrics(equity_series, trades_df, benchmark_close=df["close"])
```

(`df` here is already `df.loc[signal.index]`, so `df["close"]` aligns with `equity_series.index`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_metrics_benchmark.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `pytest tests/ -q`
Expected: all green (existing `test_backtester.py` unaffected — the new keys are additive).

- [ ] **Step 7: Commit**

```bash
git add simulation_pipeline.py tests/test_metrics_benchmark.py
git commit -m "feat: add buy-and-hold benchmark, alpha, and information ratio to metrics"
```

---

### Task 3: Deflated Sharpe ratio module

**Files:**
- Create: `deflated_sharpe.py`
- Test: `tests/test_deflated_sharpe.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `expected_max_sharpe(trial_sharpe_var: float, n_trials: int) -> float`
  - `deflated_sharpe(returns: np.ndarray, n_trials: int, trial_sharpe_var: float) -> dict` with keys `{"dsr", "sr", "sr0", "p_value"}`. `sr`/`sr0` are per-period; `dsr` ∈ [0,1] is the probability the true Sharpe exceeds the expected-max-under-null; `p_value = 1 - dsr`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_deflated_sharpe.py`:

```python
import numpy as np
from deflated_sharpe import deflated_sharpe, expected_max_sharpe


def test_expected_max_sharpe_grows_with_trials():
    lo = expected_max_sharpe(trial_sharpe_var=0.01, n_trials=5)
    hi = expected_max_sharpe(trial_sharpe_var=0.01, n_trials=100)
    assert hi > lo > 0


def test_expected_max_sharpe_zero_for_single_trial():
    assert expected_max_sharpe(trial_sharpe_var=0.01, n_trials=1) == 0.0


def test_dsr_high_when_sharpe_far_exceeds_null():
    rng = np.random.default_rng(0)
    # Strong positive per-period Sharpe (~0.3/step)
    returns = 0.003 + 0.01 * rng.standard_normal(750)
    out = deflated_sharpe(returns, n_trials=20, trial_sharpe_var=1e-4)
    assert out["sr"] > 0
    assert out["dsr"] > 0.9


def test_dsr_near_half_when_sharpe_matches_null():
    rng = np.random.default_rng(1)
    returns = 0.0005 + 0.01 * rng.standard_normal(500)
    sr = returns.mean() / returns.std(ddof=1)
    # Set the null equal to the observed SR by choosing trial variance so that
    # sr0 == sr, i.e. the observed result is exactly the expected max under null.
    from deflated_sharpe import expected_max_sharpe as ems
    # binary-search trial_sharpe_var so ems(var, 20) == sr
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ems(mid, 20) < sr:
            lo = mid
        else:
            hi = mid
    out = deflated_sharpe(returns, n_trials=20, trial_sharpe_var=(lo + hi) / 2)
    assert 0.35 < out["dsr"] < 0.65


def test_dsr_degenerate_short_series():
    out = deflated_sharpe(np.array([0.01, 0.02]), n_trials=10, trial_sharpe_var=0.01)
    assert out["dsr"] == 0.0
    assert out["p_value"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_deflated_sharpe.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the implementation**

Create `deflated_sharpe.py`:

```python
"""
deflated_sharpe.py — Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

Adjusts an observed Sharpe for (a) the number of trials that produced it
(selection bias), (b) non-normal return skew/kurtosis, and (c) sample length.
All Sharpes here are PER-PERIOD (not annualized) and must be used consistently.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EULER_MASCHERONI = 0.5772156649015329


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def expected_max_sharpe(trial_sharpe_var: float, n_trials: int) -> float:
    """Expected maximum Sharpe under the null of zero true skill across
    n_trials independent strategy configurations (the SR0 deflation target)."""
    if n_trials < 2 or trial_sharpe_var <= 0:
        return 0.0
    g = _EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(trial_sharpe_var) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe(returns: np.ndarray, n_trials: int,
                    trial_sharpe_var: float) -> dict:
    r = np.asarray(returns, dtype=float)
    T = len(r)
    degenerate = {"dsr": 0.0, "sr": 0.0, "sr0": 0.0, "p_value": 1.0}
    if T < 4:
        return degenerate
    sr = _sharpe(r)
    sr0 = expected_max_sharpe(trial_sharpe_var, n_trials)
    sd = r.std(ddof=0)
    if sd == 0:
        return {"dsr": 0.0, "sr": sr, "sr0": sr0, "p_value": 1.0}
    rm = r - r.mean()
    skew = float(np.mean(rm ** 3) / sd ** 3)
    kurt = float(np.mean(rm ** 4) / sd ** 4)   # non-excess kurtosis (gamma_4)
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2)
    if denom <= 0:
        return {"dsr": 0.0, "sr": sr, "sr0": sr0, "p_value": 1.0}
    stat = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(denom)
    dsr = float(norm.cdf(stat))
    return {"dsr": dsr, "sr": sr, "sr0": sr0, "p_value": 1.0 - dsr}


if __name__ == "__main__":
    # ponytail: runnable self-check — DSR high when SR >> SR0, ~0.5 when SR ~ SR0.
    rng = np.random.default_rng(0)
    strong = 0.003 + 0.01 * rng.standard_normal(750)
    assert deflated_sharpe(strong, 20, 1e-4)["dsr"] > 0.9
    assert deflated_sharpe(np.array([0.01, 0.02]), 10, 0.01)["dsr"] == 0.0
    print("deflated_sharpe self-check OK")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_deflated_sharpe.py -v && python deflated_sharpe.py`
Expected: all tests PASS; script prints `deflated_sharpe self-check OK`.

- [ ] **Step 5: Commit**

```bash
git add deflated_sharpe.py tests/test_deflated_sharpe.py
git commit -m "feat: add deflated Sharpe ratio (Bailey-Lopez de Prado 2014)"
```

---

### Task 4: Refactor walk_forward fold-data + IC-matrix builders

**Files:**
- Modify: `walk_forward.py` (extract two helpers; refactor `sweep_params` to use them)
- Test: `tests/test_walk_forward.py` (add matrix-builder tests; existing tests must still pass unchanged)

**Interfaces:**
- Consumes: `compute_predictor_signal`, `_ic`, `_preprocess`, `_load_symbol`, `make_daily_features` (all already imported in `walk_forward.py`).
- Produces:
  - `build_fold_data(symbols: list[str], days: int, db, config: WalkForwardConfig) -> list[tuple[np.ndarray, np.ndarray, int]]` — one `(pred_window, y_te, test_offset)` per walk-forward fold, pooled across symbols. Identical to the tuples `sweep_params` builds today.
  - `fold_config_ic_matrix(fold_data, quantiles: list[float], windows: list[int]) -> tuple[np.ndarray, list[tuple[float, int]]]` — returns `(matrix, configs)` where `matrix` is `(n_folds, n_configs)` of per-fold OOS IC (higher = better) and `configs[j]` is the `(quantile, window)` for column `j`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_walk_forward.py`:

```python
def test_build_fold_data_matches_matrix_shape():
    from walk_forward import build_fold_data, fold_config_ic_matrix, WalkForwardConfig
    import numpy as np
    # Synthetic fold_data: 10 folds, each a length-120 prediction window.
    rng = np.random.default_rng(0)
    fold_data = []
    for _ in range(10):
        pred = rng.standard_normal(120)
        y_te = rng.standard_normal(60)      # test slice length
        test_offset = 60
        fold_data.append((pred, y_te, test_offset))
    quantiles = [0.6, 0.7]
    windows = [40, 60]
    matrix, configs = fold_config_ic_matrix(fold_data, quantiles, windows)
    assert matrix.shape == (10, 4)
    assert configs == [(0.6, 40), (0.6, 60), (0.7, 40), (0.7, 60)]
    assert np.isfinite(matrix).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_walk_forward.py::test_build_fold_data_matches_matrix_shape -v`
Expected: FAIL — `build_fold_data` / `fold_config_ic_matrix` not defined.

- [ ] **Step 3: Extract the helpers**

In `walk_forward.py`, add these two functions above `sweep_params`:

```python
def build_fold_data(symbols, days, db, config=None):
    """Pool per-fold (pred_window, y_te, test_offset) tuples across symbols.

    Extracted verbatim from sweep_params' inner loop so sweep_params and the
    CPCV matrix builder derive folds identically (single source of truth).
    """
    if config is None:
        config = WalkForwardConfig()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    spy_df = _load_symbol("SPY", start, end, db)

    fold_data = []
    for symbol in symbols:
        df = _load_symbol(symbol, start, end, db)
        if df is None:
            continue
        try:
            spy_arg = spy_df if symbol != "SPY" else None
            feats = make_daily_features(df, spy_df=spy_arg).dropna(subset=["fwd_ret_1d"])
        except Exception as exc:
            logger.warning("build_fold_data: feature error for %s: %s", symbol, exc)
            continue
        n = len(feats)
        min_bars = config.train_bars + FWD_RET_HORIZON_DAYS + config.test_bars
        if n < min_bars:
            continue
        X_all = _preprocess(feats[FEATURE_COLS].values.astype(np.float32))
        y_all = feats["fwd_ret_1d"].values.astype(np.float64)
        train_start_idx = 0
        while True:
            train_end_idx = train_start_idx + config.train_bars
            test_start_idx = train_end_idx + FWD_RET_HORIZON_DAYS
            test_end_idx = test_start_idx + config.test_bars
            if test_end_idx > n:
                break
            X_window = X_all[train_start_idx:test_end_idx]
            scaler = StandardScaler()
            scaler.fit(X_all[train_start_idx:train_end_idx])
            X_window_s = scaler.transform(X_window)
            model = Ridge(alpha=config.ridge_alpha)
            model.fit(X_window_s[:config.train_bars], y_all[train_start_idx:train_end_idx])
            pred_window = model.predict(X_window_s)
            test_offset = config.train_bars + FWD_RET_HORIZON_DAYS
            y_te = y_all[test_start_idx:test_end_idx]
            fold_data.append((pred_window, y_te, test_offset))
            train_start_idx += config.step_bars
    return fold_data


def fold_config_ic_matrix(fold_data, quantiles, windows):
    """(n_folds x n_configs) matrix of per-fold OOS IC; columns = (q, w) configs."""
    configs = [(q, w) for q in quantiles for w in windows]
    rows = []
    for pred_window, y_te, test_offset in fold_data:
        row = []
        for (q, w) in configs:
            if len(pred_window) < w:
                row.append(0.0)
                continue
            signals_window = compute_predictor_signal(pred_window, q, w)
            signals_test = signals_window[test_offset:]
            active = signals_test != 0
            if active.sum() < 5:
                row.append(0.0)
            else:
                row.append(_ic(pred_window[test_offset:][active], y_te[active]))
        rows.append(row)
    return np.array(rows, dtype=float), configs
```

- [ ] **Step 4: Refactor sweep_params to use them**

Replace the body of `sweep_params` (the per-symbol fold-building loop and the `for q ... for w ...` selection) with a call to the new helpers, preserving the exact fallback semantics:

```python
def sweep_params(symbols, days, db, config=None, quantiles=None, windows=None):
    if config is None:
        config = WalkForwardConfig()
    if quantiles is None:
        quantiles = _DEFAULT_QUANTILES
    if windows is None:
        windows = _DEFAULT_WINDOWS

    fold_data = build_fold_data(symbols, days, db, config)
    if not fold_data:
        logger.warning("sweep_params: no valid folds — returning defaults (%s, %d)",
                       _FALLBACK_QUANTILE, _FALLBACK_WINDOW)
        return _FALLBACK_QUANTILE, _FALLBACK_WINDOW

    matrix, configs = fold_config_ic_matrix(fold_data, quantiles, windows)
    median_ic = np.median(matrix, axis=0)          # per-config median across folds
    best_j = int(np.argmax(median_ic))
    best_ic = float(median_ic[best_j])
    if best_ic <= 0:
        logger.warning(
            "sweep_params: all (quantile, window) combinations produced IC <= 0 "
            "— keeping defaults (%.2f, %d)", _FALLBACK_QUANTILE, _FALLBACK_WINDOW)
        return _FALLBACK_QUANTILE, _FALLBACK_WINDOW

    best_q, best_w = configs[best_j]
    logger.info("sweep_params: best q=%.2f w=%d median_IC=%.4f", best_q, best_w, best_ic)
    return best_q, best_w
```

Note: the old code set unmatched folds' IC to `0.0` when `active.sum() < 5`; `fold_config_ic_matrix` preserves that, so the median selection is equivalent.

- [ ] **Step 5: Run tests to verify no regression**

Run: `pytest tests/test_walk_forward.py -v`
Expected: PASS — the new matrix test passes AND the pinned `test_sweep_params_returns_valid_pair`, `test_sweep_params_falls_back_when_all_ic_nonpositive`, `test_sweep_params_no_valid_symbols_returns_defaults` still pass (selection + fallback preserved).

- [ ] **Step 6: Run the full suite**

Run: `pytest tests/ -q`
Expected: all green. (`train_predictor.py` imports `sweep_params` — confirm it still imports and runs.)

- [ ] **Step 7: Commit**

```bash
git add walk_forward.py tests/test_walk_forward.py
git commit -m "refactor: extract build_fold_data + fold_config_ic_matrix, reuse in sweep_params"
```

---

### Task 5: CPCV / PBO engine

**Files:**
- Create: `cpcv.py`
- Test: `tests/test_cpcv.py`

**Interfaces:**
- Consumes: nothing (pure — takes a performance matrix).
- Produces: `cscv_pbo(perf_matrix: np.ndarray, n_splits: int = 16) -> dict` with keys `{"pbo", "logits", "n_combinations", "reason"}`. `perf_matrix` is `(n_observations, n_configs)`, higher = better (feed it `fold_config_ic_matrix`'s output). `pbo` ∈ [0,1] is the fraction of combinatorial splits where the in-sample-best config lands below the out-of-sample median; `NaN` with a `reason` when inputs are too small.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cpcv.py`:

```python
import numpy as np
from cpcv import cscv_pbo


def test_pbo_near_half_for_pure_noise():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((40, 20))     # 40 obs, 20 configs, no real edge
    out = cscv_pbo(M, n_splits=10)
    assert 0.30 < out["pbo"] < 0.70


def test_pbo_near_zero_for_one_genuine_edge():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((40, 20)) * 0.1
    M[:, 3] += 1.0                         # config 3 is consistently best
    out = cscv_pbo(M, n_splits=10)
    assert out["pbo"] < 0.10


def test_pbo_insufficient_observations_returns_reason():
    M = np.random.default_rng(2).standard_normal((4, 5))
    out = cscv_pbo(M, n_splits=16)
    assert np.isnan(out["pbo"])
    assert "insufficient" in out["reason"].lower()


def test_pbo_needs_two_configs():
    M = np.random.default_rng(3).standard_normal((40, 1))
    out = cscv_pbo(M, n_splits=10)
    assert np.isnan(out["pbo"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cpcv.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the implementation**

Create `cpcv.py`:

```python
"""
cpcv.py — Probability of Backtest Overfitting via Combinatorially Symmetric
Cross-Validation (Bailey, Borwein, Lopez de Prado & Zhu, 2015).

Given a performance matrix (observations x strategy-configs), estimate the
probability that the configuration chosen as best in-sample underperforms the
median configuration out-of-sample. Targets the (signal_quantile,
threshold_window) grid selection in walk_forward.sweep_params.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def cscv_pbo(perf_matrix: np.ndarray, n_splits: int = 16) -> dict:
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return {"pbo": float("nan"), "logits": [], "n_combinations": 0,
                "reason": "need >= 2 configs"}
    T, N = M.shape

    # Clamp n_splits to an even number <= T; require a meaningful minimum.
    S = min(n_splits, T - (T % 2))
    if S % 2 == 1:
        S -= 1
    if T < 8 or S < 4:
        return {"pbo": float("nan"), "logits": [], "n_combinations": 0,
                "reason": f"insufficient observations for PBO (got {T}, need >= 8)"}

    blocks = np.array_split(np.arange(T), S)
    half = S // 2
    logits = []
    for is_blocks in combinations(range(S), half):
        is_set = set(is_blocks)
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in range(S) if b not in is_set])

        is_perf = M[is_rows].mean(axis=0)
        oos_perf = M[oos_rows].mean(axis=0)
        n_star = int(np.argmax(is_perf))

        # OOS relative rank of the IS-best config (higher perf -> higher rank).
        order = oos_perf.argsort()
        ranks = np.empty(N)
        ranks[order] = np.arange(1, N + 1)
        w = ranks[n_star] / (N + 1)                 # in (0, 1)
        w = min(max(w, 1e-6), 1.0 - 1e-6)
        logits.append(float(np.log(w / (1.0 - w))))

    logits_arr = np.array(logits)
    pbo = float(np.mean(logits_arr < 0.0))
    return {"pbo": pbo, "logits": logits, "n_combinations": len(logits),
            "reason": ""}


if __name__ == "__main__":
    # ponytail: runnable self-check — noise -> ~0.5, single genuine edge -> ~0.
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((40, 20))
    assert 0.3 < cscv_pbo(noise, 10)["pbo"] < 0.7
    edged = rng.standard_normal((40, 20)) * 0.1
    edged[:, 3] += 1.0
    assert cscv_pbo(edged, 10)["pbo"] < 0.1
    print("cpcv self-check OK")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cpcv.py -v && python cpcv.py`
Expected: all tests PASS; script prints `cpcv self-check OK`.

- [ ] **Step 5: Commit**

```bash
git add cpcv.py tests/test_cpcv.py
git commit -m "feat: add CSCV probability-of-backtest-overfitting engine"
```

---

### Task 6: eval_report.py CLI

**Files:**
- Create: `eval_report.py`
- Test: `tests/test_eval_report.py`

**Interfaces:**
- Consumes: `build_fold_data`, `fold_config_ic_matrix`, `WalkForwardConfig` (Task 4); `cscv_pbo` (Task 5); `deflated_sharpe` (Task 3); `DailyPredictorStrategy`, `Backtester`, `ExecutionConfig`, `StrategyConfig`, `_DEFAULT_QUANTILES`, `_DEFAULT_WINDOWS`.
- Produces:
  - `compute_pbo(symbols, days, db, config=None) -> dict` — builds the fold IC matrix over `_DEFAULT_QUANTILES` × `_DEFAULT_WINDOWS` and returns `cscv_pbo(...)`.
  - `compute_dsr_for_symbol(symbol, df, quantiles, windows) -> dict` — runs a backtest per `(q,w)` config on one symbol's price frame, returns `{"dsr", "sr", "sr0", "p_value", "selected": (q, w), "n_trials"}`, where the selected config's daily returns feed `deflated_sharpe` and the variance of all configs' daily Sharpes is `trial_sharpe_var`.
  - `main()` — CLI printing a PBO line plus a per-symbol DSR table.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_report.py` (uses synthetic prices; no network):

```python
import numpy as np
import pandas as pd
from eval_report import compute_dsr_for_symbol


def _synth_prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + 0.0005 + 0.01 * rng.standard_normal(n))
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    high = close * (1 + 0.005)
    low = close * (1 - 0.005)
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def test_compute_dsr_returns_expected_keys():
    df = _synth_prices()
    out = compute_dsr_for_symbol("SYNTH", df, quantiles=[0.6, 0.7], windows=[40, 60])
    for k in ("dsr", "sr", "sr0", "p_value", "selected", "n_trials"):
        assert k in out
    assert out["n_trials"] == 4
    assert 0.0 <= out["dsr"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_report.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the implementation**

Create `eval_report.py`:

```python
#!/usr/bin/env python3
"""
eval_report.py — Honest-evaluation report for the daily_predictor strategy.

Reports:
  * PBO  — probability of backtest overfitting over the (quantile, window) grid.
  * DSR  — deflated Sharpe per symbol, deflating the selected config's Sharpe by
           the number of configs tried and the cross-config Sharpe variance.

Real-data path fetches via the same cache-aware loader as training. Pure helpers
(compute_dsr_for_symbol) accept an in-memory price frame for testing.
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np

from base_strategy import StrategyConfig
from cpcv import cscv_pbo
from deflated_sharpe import deflated_sharpe
from ml_strategies import DailyPredictorStrategy
from simulation_pipeline import Backtester, ExecutionConfig, compute_metrics
from walk_forward import (
    WalkForwardConfig, build_fold_data, fold_config_ic_matrix,
    _DEFAULT_QUANTILES, _DEFAULT_WINDOWS,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_pbo(symbols, days, db, config=None):
    if config is None:
        config = WalkForwardConfig()
    fold_data = build_fold_data(symbols, days, db, config)
    matrix, configs = fold_config_ic_matrix(fold_data, _DEFAULT_QUANTILES, _DEFAULT_WINDOWS)
    result = cscv_pbo(matrix)
    result["n_configs"] = len(configs)
    result["n_folds"] = int(matrix.shape[0]) if matrix.size else 0
    return result


def _backtest_sharpe(df, q, w):
    """Daily returns + Sharpe for one (q, w) config on one price frame.

    Forces the config via the env-var override that DailyPredictorStrategy reads
    at highest priority, so pickle best_* keys do not shadow it."""
    prev_q = os.environ.get("PREDICTOR_SIGNAL_QUANTILE")
    prev_w = os.environ.get("PREDICTOR_THRESHOLD_WINDOW")
    os.environ["PREDICTOR_SIGNAL_QUANTILE"] = str(q)
    os.environ["PREDICTOR_THRESHOLD_WINDOW"] = str(w)
    try:
        cfg = StrategyConfig(name="daily_predictor")
        strat = DailyPredictorStrategy(cfg)
        sig = strat.signal(None, df)
        sig = sig.reindex(df.index).fillna(0).astype(int)
        bt = Backtester(ExecutionConfig())
        res = bt.run(df, df, sig, artifact_paths={})
        daily = res.equity_curve.resample("1D").last().dropna().pct_change().dropna()
        return daily.values, res.metrics.get("daily_sharpe", 0.0)
    finally:
        for key, prev in (("PREDICTOR_SIGNAL_QUANTILE", prev_q),
                          ("PREDICTOR_THRESHOLD_WINDOW", prev_w)):
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def compute_dsr_for_symbol(symbol, df, quantiles=None, windows=None):
    quantiles = quantiles or _DEFAULT_QUANTILES
    windows = windows or _DEFAULT_WINDOWS
    configs = [(q, w) for q in quantiles for w in windows]

    per_config = []       # (config, daily_returns, sharpe)
    for (q, w) in configs:
        daily, sharpe = _backtest_sharpe(df, q, w)
        per_config.append(((q, w), daily, sharpe))

    sharpes = np.array([s for _, _, s in per_config], dtype=float)
    trial_var = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    best_i = int(np.argmax(sharpes))
    sel_config, sel_daily, _ = per_config[best_i]

    dsr = deflated_sharpe(sel_daily, n_trials=len(configs), trial_sharpe_var=trial_var)
    dsr["selected"] = sel_config
    dsr["n_trials"] = len(configs)
    return dsr


def main():
    from datetime import datetime, timedelta
    from db import DB
    from predict_next_day_lite import _load_bars_cached

    parser = argparse.ArgumentParser(description="Honest-eval report for daily_predictor")
    parser.add_argument("--symbols", default="AAPL,MSFT,SPY,QQQ,NVDA")
    parser.add_argument("--days", type=int, default=2500)
    parser.add_argument("--db", default="data/trading_sim.db")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    try:
        db = DB(args.db)
    except Exception:
        db = None

    pbo = compute_pbo(symbols, args.days, db)
    print("\n=== Probability of Backtest Overfitting (q,w grid) ===")
    if pbo.get("reason"):
        print(f"PBO: n/a — {pbo['reason']}")
    else:
        print(f"PBO = {pbo['pbo']:.3f}  (folds={pbo['n_folds']}, configs={pbo['n_configs']}, "
              f"combinations={pbo['n_combinations']})")

    print("\n=== Deflated Sharpe per symbol ===")
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    dsrs = []
    for sym in symbols:
        df = _load_bars_cached(sym, start, end, db=db)
        if df is None or len(df) < 300:
            print(f"  {sym}: insufficient data")
            continue
        out = compute_dsr_for_symbol(sym, df)
        dsrs.append(out["dsr"])
        print(f"  {sym}: DSR={out['dsr']:.3f} SR={out['sr']:+.4f} SR0={out['sr0']:+.4f} "
              f"selected(q,w)={out['selected']}")
    if dsrs:
        print(f"\nMedian DSR across symbols: {np.median(dsrs):.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_report.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add eval_report.py tests/test_eval_report.py
git commit -m "feat: add eval_report CLI (PBO + per-symbol deflated Sharpe)"
```

---

### Task 7: Re-baseline and record honest numbers

**Files:**
- Modify: `models/README.md` ("Prediction vs. strategy" section)
- Modify: `C:\Users\sssor\.claude\projects\C--Users-sssor-Documents-trading-sim\memory\trading_sim_honest_accuracy_2026_06.md` (and `MEMORY.md` pointer if the hook changes)

**Interfaces:** none (operational task).

> **Depends on:** Tasks 1–6 merged. **Requires network** (yfinance) — `data/` has no `trading_sim.db`, so the first run fetches and caches bars. If yfinance is unavailable, stop here; code from Tasks 1–6 still stands.

- [ ] **Step 1: Re-run the daily_predictor backtest (post-shift-fix)**

Run:
```bash
python simulate_multi.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --strategies daily_predictor --days 365
```
Expected: writes `results/multi_summary.json` and per-run metrics. Note the new `total_return_pct`, `daily_sharpe`, and the new `alpha_pct` / `information_ratio` keys (Task 2). These should be **lower** than the pre-fix figures — that is the point.

- [ ] **Step 2: Run the honest-eval report**

Run:
```bash
python eval_report.py --symbols AAPL,MSFT,SPY,QQQ,NVDA --days 2500
```
Expected: a PBO value and a per-symbol DSR table. Record the PBO and median DSR.

- [ ] **Step 3: Update models/README.md**

In the "Prediction vs. strategy" section, replace the pre-fix backtest numbers with the Step 1/2 results. Add a line stating the look-ahead fix (1-bar execution lag) was applied on 2026-07-13 and that the quoted figures are post-fix, alongside the PBO and median DSR. Keep the existing honest caveats.

- [ ] **Step 4: Update the memory file**

Edit `memory/trading_sim_honest_accuracy_2026_06.md`: note that the daily_predictor backtest previously carried a 1-bar look-ahead (fixed 2026-07-13), record the corrected Sharpe/alpha and the PBO/DSR, and cross-link `[[project_rebuild_state]]` if relevant. Update the `MEMORY.md` one-line hook only if its summary changed.

- [ ] **Step 5: Commit**

```bash
git add models/README.md results/multi_summary.json
git commit -m "chore: re-baseline daily_predictor after look-ahead fix; record PBO/DSR"
```
(The memory files live outside the repo — they are saved via the memory tooling, not committed here.)

---

## Self-Review

**Spec coverage:**
- Look-ahead shift fix → Task 1. ✓
- Deflated Sharpe module → Task 3. ✓
- CPCV/PBO engine → Task 5 (matrix builder → Task 4). ✓
- Benchmark row (traded symbol, alpha, IR) → Task 2. ✓
- eval_report.py CLI → Task 6. ✓
- Re-baseline + doc/memory update → Task 7. ✓
- PBO targets the (q,w) sweep specifically → Task 4 matrix + Task 5, wired in Task 6 `compute_pbo`. ✓
- Backward-compatible metrics, no-network tests, live path unshifted → Global Constraints + Tasks 1/2. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. Task 7 is operational (its "output" is data-dependent), with concrete commands and explicit expected shapes. ✓

**Type consistency:**
- `deflated_sharpe(returns, n_trials, trial_sharpe_var) -> {dsr,sr,sr0,p_value}` — defined Task 3, consumed Task 6 with matching args. ✓
- `cscv_pbo(perf_matrix, n_splits=16) -> {pbo,logits,n_combinations,reason}` — defined Task 5, consumed Task 6. ✓
- `build_fold_data(symbols, days, db, config)` / `fold_config_ic_matrix(fold_data, quantiles, windows) -> (matrix, configs)` — defined Task 4, consumed Task 6. ✓
- `compute_metrics(equity, trades, benchmark_close=None)` — Task 2; call site updated same task. ✓
- Env-var override names `PREDICTOR_SIGNAL_QUANTILE` / `PREDICTOR_THRESHOLD_WINDOW` match those `DailyPredictorStrategy.__init__` reads (ml_strategies.py). ✓

No gaps found.
