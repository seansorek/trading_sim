# Sub-project 2: Code Quality — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove dead intraday-strategy code, add missing type annotations to public functions, and fix the incorrect `BacktestResult.metrics` type annotation — all without changing runtime behavior.

**Architecture:** Three independent PRs, each targeting one logical group. PR 1 deletes dead code from `ml_strategies.py` and `simulation_pipeline.py` and updates callers. PR 2 fills in missing return-type annotations. PR 3 corrects the `BaseStrategy.signal()` contract and the `BacktestResult.metrics` type.

**Tech Stack:** Python 3.10+, pytest, existing test suite. No new dependencies.

## Global Constraints

- Do NOT change any runtime behavior — only delete unreachable code and add/fix annotations.
- Do NOT touch `FEATURE_COLS`, `compute_predictor_signal`, or `predictions/history.jsonl`.
- Do NOT mix changes across PRs — dead code removal, type annotations, and interface fixes go in separate commits.
- The three-level param priority (env var → pickle → hardcoded default) in `DailyPredictorStrategy` must be preserved untouched.
- Run `pytest tests/ -v` after every task and confirm all tests pass before committing.

---

## File Structure

**Task 1 — Dead code removal:**
- Modify: `ml_strategies.py` — delete `_apply_confidence_filter`, `OrdinalLogisticStrategy`, `XGBoostStrategy`; remove `import pickle` and the `LogisticRegression` line from the try block
- Modify: `simulation_pipeline.py` — delete `rsi`, `make_features`, `_DailyRidgeQuantileStrategy` conditional block, `walk_forward_backtest`
- Modify: `simulate_multi.py` — remove `make_features` and `walk_forward_backtest` from imports; simplify `run_symbol_strategy`
- Modify: `tests/test_backtester.py` — delete three test functions that test the removed code
- Create: `tests/test_dead_code.py` — regression assertions that the removed names are truly gone

**Task 2 — Type annotations:**
- Modify: `data_loader.py` — add `-> None` to `save_to_csv`
- Modify: `train_models.py` — narrow `train_xgboost` return from bare `tuple` to `tuple[Any, Any, float, float, float]`
- Modify: `predict_next_day_lite.py` — add `from typing import Any`; change `load_models` and `predict_symbol` return types from `dict` to `dict[str, Any]`; annotate `spy_df` param
- Create: `tests/test_code_quality.py` — lightweight checks that annotations exist

**Task 3 — Interface consistency:**
- Modify: `base_strategy.py` — add `-> pd.Series` to `BaseStrategy.signal()`
- Modify: `simulation_pipeline.py` — change `BacktestResult.metrics` from `Dict[str, float]` to `Dict[str, Any]`; update `compute_metrics` return annotation to match
- Extend: `tests/test_code_quality.py` — add interface consistency checks

---

## Task 1: Dead Code Removal

**Files:**
- Modify: `ml_strategies.py`
- Modify: `simulation_pipeline.py`
- Modify: `simulate_multi.py`
- Modify: `tests/test_backtester.py`
- Create: `tests/test_dead_code.py`

**Interfaces:**
- Consumes: nothing new — this task only removes code
- Produces: a leaner `STRATEGY_REGISTRY` (no `daily_ridge_q`), `simulate_multi.run_symbol_strategy` that always sets `feats = df`

- [ ] **Step 1: Write the failing regression tests**

Create `tests/test_dead_code.py`:

```python
"""Regression tests: verify removed dead code is truly gone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_ordinal_logistic_removed():
    import ml_strategies
    assert not hasattr(ml_strategies, "OrdinalLogisticStrategy"), (
        "OrdinalLogisticStrategy (intraday legacy) must be removed from ml_strategies"
    )


def test_xgboost_intraday_removed():
    import ml_strategies
    assert not hasattr(ml_strategies, "XGBoostStrategy"), (
        "XGBoostStrategy (intraday legacy) must be removed from ml_strategies"
    )


def test_apply_confidence_filter_removed():
    import ml_strategies
    assert not hasattr(ml_strategies, "_apply_confidence_filter"), (
        "_apply_confidence_filter (used only by intraday classes) must be removed"
    )


def test_daily_ridge_q_not_in_registry():
    from simulation_pipeline import STRATEGY_REGISTRY
    assert "daily_ridge_q" not in STRATEGY_REGISTRY, (
        "daily_ridge_q (_DailyRidgeQuantileStrategy) must not appear in STRATEGY_REGISTRY"
    )


def test_walk_forward_backtest_removed():
    import simulation_pipeline
    assert not hasattr(simulation_pipeline, "walk_forward_backtest"), (
        "walk_forward_backtest (intraday-only) must be removed from simulation_pipeline"
    )


def test_make_features_removed():
    import simulation_pipeline
    assert not hasattr(simulation_pipeline, "make_features"), (
        "make_features (intraday feature builder) must be removed from simulation_pipeline"
    )


def test_rsi_removed():
    import simulation_pipeline
    assert not hasattr(simulation_pipeline, "rsi"), (
        "rsi helper (used only by make_features) must be removed from simulation_pipeline"
    )


def test_run_symbol_strategy_wf_always_skipped():
    """After removal, run_symbol_strategy must always return wf_metrics with skipped=True."""
    import pandas as pd
    import numpy as np
    from unittest.mock import patch
    from simulation_pipeline import ExecutionConfig, StrategyConfig
    import simulate_multi

    n = 60
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    prices = 100.0 + np.arange(n, dtype=float) * 0.1
    df = pd.DataFrame(
        {
            "open": prices - 0.2,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )
    exec_cfg = ExecutionConfig(
        start_cash=100_000.0,
        commission_per_share=0.0,
        slippage_bps=0.0,
        stop_loss_pct=0.5,
        take_profit_pct=0.5,
        daily_loss_limit_pct=0.99,
        max_position_pct=0.05,
    )
    flat_signal = pd.Series(0, index=df.index)
    with patch("simulate_multi.build_strategy_signal", return_value=flat_signal), \
         patch("simulate_multi.monte_carlo_stress", return_value=pd.DataFrame()):
        result = simulate_multi.run_symbol_strategy(
            symbol="TEST",
            strategy_name="daily_logistic",
            df=df,
            cfg=StrategyConfig(name="daily_logistic", lookback=20, holding_period=5),
            exec_cfg=exec_cfg,
            run_id="test-run-001",
            n_mc_runs=0,
        )
    assert result["wf_metrics"].get("skipped") is True, (
        f"Expected wf_metrics['skipped']=True after walk_forward removal, got: {result['wf_metrics']}"
    )
```

- [ ] **Step 2: Run the new tests — confirm they all FAIL**

```
pytest tests/test_dead_code.py -v
```

Expected: 8 failures — all the names still exist in the modules.

- [ ] **Step 3: Edit `ml_strategies.py` — remove dead imports and classes**

**Remove `import pickle` (line 17).**

Old:
```python
import logging
import os
import pickle
from typing import Optional
```

New:
```python
import logging
import os
from typing import Optional
```

**Remove the `LogisticRegression` import from the sklearn try block (line 36). Keep `StandardScaler` and `HAS_SKLEARN`.**

Old:
```python
try:
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
```

New:
```python
try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
```

**Remove `_apply_confidence_filter` (lines 49–55) entirely:**

```python
# DELETE this entire function:
def _apply_confidence_filter(
    preds: np.ndarray, probs: np.ndarray, threshold: float
) -> np.ndarray:
    """Zero out predictions whose max probability is below threshold."""
    confidence = probs.max(axis=1)
    result = preds.copy()
    result[confidence < threshold] = 0
    return result
```

**Remove the entire intraday legacy section (from the separator comment through `XGBoostStrategy`):**

```python
# DELETE everything from this comment through the end of the file:

# ---------------------------------------------------------------------------
# Intraday strategies (legacy — not used in daily pipeline)
# ---------------------------------------------------------------------------

class OrdinalLogisticStrategy(BaseStrategy):
    ... (entire class, ~43 lines)

class XGBoostStrategy(BaseStrategy):
    ... (entire class, ~41 lines)
```

- [ ] **Step 4: Edit `simulation_pipeline.py` — remove three dead blocks**

**Block A: Remove the intraday feature builder section (lines 38–76).**

Delete the section header comment and both functions:

```python
# DELETE from the comment through the end of make_features:

# ---------------------------------------------------------------------------
# Intraday feature builder (kept for walk-forward; not used by daily strategies)
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    ... (entire function)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    ... (entire function)
```

**Block B: Remove the `_DailyRidgeQuantileStrategy` conditional block (lines 136–151).**

Delete the entire block (variable assignment, imports, class definition, registry insertion):

```python
# DELETE this entire block:

# Register Ridge + QuantileDecision if daily_predictor.pkl exists
_RIDGE_PATH = os.environ.get("RIDGE_MODEL", "models/daily_predictor.pkl")
if os.path.exists(_RIDGE_PATH):
    from predictors.ridge import RidgePredictor
    from decision_layers.quantile import QuantileDecision

    class _DailyRidgeQuantileStrategy(BaseStrategy):
        def __init__(self, cfg: StrategyConfig, spy_df=None, **_kwargs):
            super().__init__(cfg)
            predictor = RidgePredictor.load(_RIDGE_PATH)
            decision = QuantileDecision(signal_quantile=0.7, threshold_window=63)
            self._inner = PredictorStrategy(cfg, predictor, decision, spy_df=spy_df)

        def signal(self, feats, df):
            return self._inner.signal(feats, df)

    STRATEGY_REGISTRY["daily_ridge_q"] = _DailyRidgeQuantileStrategy
```

**Block C: Remove `walk_forward_backtest` (lines 543–580).**

Delete the section header comment and entire function:

```python
# DELETE from the comment through the end of walk_forward_backtest:

# ---------------------------------------------------------------------------
# Walk-forward backtest (linear model on intraday features)
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    train_days: int = 5,
    test_days: int = 1,
    artifact_paths: Optional[dict] = None,
) -> BacktestResult:
    ... (entire function body)
```

- [ ] **Step 5: Edit `simulate_multi.py` — remove dead imports and simplify `run_symbol_strategy`**

**Remove `make_features` and `walk_forward_backtest` from the import block (lines 27–28).**

Old:
```python
from simulation_pipeline import (
    Backtester,
    ExecutionConfig,
    StrategyConfig,
    STRATEGY_REGISTRY,
    build_strategy_signal,
    make_features,
    monte_carlo_stress,
    walk_forward_backtest,
)
```

New:
```python
from simulation_pipeline import (
    Backtester,
    ExecutionConfig,
    StrategyConfig,
    STRATEGY_REGISTRY,
    build_strategy_signal,
    monte_carlo_stress,
)
```

**Simplify `run_symbol_strategy` — replace the ternary feats line and the if/else wf block.**

Old (lines 67–93 inside the function):
```python
    # Daily strategies build their own features internally; pass empty feats
    feats = make_features(df) if not strategy_name.startswith("daily_") else df

    signal = build_strategy_signal(strategy_name, cfg, feats, df, spy_df=spy_df)

    bt = Backtester(exec_cfg)
    res = bt.run(
        df, feats, signal,
        artifact_paths=artifacts,
        run_id=run_id,
        symbol=symbol,
        strategy=strategy_name,
    )

    # walk_forward_backtest uses intraday features (_COLS) not present in daily
    # OHLCV data, so it produces an all-flat signal for daily strategies.
    # Skip it for daily_ strategies to avoid silently returning meaningless metrics.
    if strategy_name.startswith("daily_"):
        wf_metrics = {"skipped": True, "reason": "daily strategy"}
    else:
        wf = walk_forward_backtest(
            df, feats, train_days=3, test_days=1,
            artifact_paths={
                "equity_curve_csv": f"{artifact_base}_wf_equity_curve.csv",
                "trade_log_csv": f"{artifact_base}_wf_trade_log.csv",
                "metrics_json": f"{artifact_base}_wf_metrics.json",
            },
        )
        wf_metrics = wf.metrics
```

New:
```python
    feats = df  # all registered strategies are daily_ and build features internally

    signal = build_strategy_signal(strategy_name, cfg, feats, df, spy_df=spy_df)

    bt = Backtester(exec_cfg)
    res = bt.run(
        df, feats, signal,
        artifact_paths=artifacts,
        run_id=run_id,
        symbol=symbol,
        strategy=strategy_name,
    )

    wf_metrics: dict = {"skipped": True}
```

- [ ] **Step 6: Remove three dead test functions from `tests/test_backtester.py`**

Delete these three complete test functions (including their docstrings):

1. `test_walk_forward_skipped_for_daily_strategy` — lines 298–329 (the one with `from simulation_pipeline import walk_forward_backtest`)
2. `test_run_symbol_strategy_wf_skipped_flag` — lines 332–387 (the one that patches `simulate_multi.walk_forward_backtest`)
3. `test_walk_forward_artifact_paths_are_scoped` — lines 657–668 (the one with `walk_forward_backtest(df, df, train_days=5, ...)`)

After deleting, the section comment block starting with `# Issue #20 — walk_forward_backtest skip for daily strategies` (line 294–296) has no tests under it and should also be deleted.

- [ ] **Step 7: Run the full test suite**

```
pytest tests/ -v
```

Expected: All pass. `tests/test_dead_code.py` should now show 8 passing tests. No tests in `test_backtester.py` should reference `walk_forward_backtest` or `make_features`.

- [ ] **Step 8: Commit**

```bash
git add ml_strategies.py simulation_pipeline.py simulate_multi.py \
        tests/test_backtester.py tests/test_dead_code.py
git commit -m "refactor: remove intraday legacy dead code (OrdinalLogistic, walk_forward, daily_ridge_q)"
```

---

## Task 2: Type Annotations

**Files:**
- Modify: `data_loader.py`
- Modify: `train_models.py`
- Modify: `predict_next_day_lite.py`
- Create: `tests/test_code_quality.py`

**Interfaces:**
- Consumes: nothing — pure annotation additions
- Produces: `load_models` and `predict_symbol` return `dict[str, Any]`; `save_to_csv` returns `None`; `train_xgboost` returns `tuple[Any, Any, float, float, float]`

- [ ] **Step 1: Write the failing annotation tests**

Create `tests/test_code_quality.py`:

```python
"""Code quality checks: annotation presence and accuracy."""
import sys
from pathlib import Path
import typing

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_save_to_csv_has_return_annotation():
    from data_loader import save_to_csv
    assert "return" in save_to_csv.__annotations__, (
        "save_to_csv must have a -> None return annotation"
    )


def test_train_xgboost_return_is_parameterized():
    """train_xgboost should return tuple[Any, Any, float, float, float], not bare tuple."""
    from train_models import train_xgboost
    ret = train_xgboost.__annotations__.get("return")
    assert ret is not None, "train_xgboost must have a return annotation"
    assert hasattr(ret, "__args__"), (
        f"Expected parameterized tuple (e.g. tuple[Any, Any, float, float, float]), got {ret}"
    )


def test_load_models_return_annotation():
    from predict_next_day_lite import load_models
    ret = load_models.__annotations__.get("return")
    assert ret is not None, "load_models must have a return annotation"


def test_predict_symbol_spy_df_annotated():
    from predict_next_day_lite import predict_symbol
    assert "spy_df" in predict_symbol.__annotations__, (
        "predict_symbol must annotate its spy_df parameter"
    )


def test_predict_symbol_return_annotation():
    from predict_next_day_lite import predict_symbol
    ret = predict_symbol.__annotations__.get("return")
    assert ret is not None, "predict_symbol must have a return annotation"
```

- [ ] **Step 2: Run the new tests — confirm they FAIL**

```
pytest tests/test_code_quality.py -v
```

Expected: FAIL on `test_train_xgboost_return_is_parameterized`, `test_predict_symbol_spy_df_annotated`, and the others that are currently unannotated.

- [ ] **Step 3: Edit `data_loader.py` — add `-> None` to `save_to_csv`**

Old (line 298):
```python
def save_to_csv(df: pd.DataFrame, path: str):
```

New:
```python
def save_to_csv(df: pd.DataFrame, path: str) -> None:
```

- [ ] **Step 4: Edit `train_models.py` — narrow `train_xgboost` return type**

Add `from typing import Any` to the imports at the top of the file (after the stdlib imports, before third-party):

Old (near line 14):
```python
import argparse
import json
import hashlib
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
```

New (add `Any` to the existing imports — no separate line needed since `Any` comes from `typing`):
```python
import argparse
import json
import hashlib
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
```

Old `train_xgboost` signature (line 379–387):
```python
def train_xgboost(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
    optimize: bool = False,
    opt_cfg=None,
) -> tuple:
```

New:
```python
def train_xgboost(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
    optimize: bool = False,
    opt_cfg: Any = None,
) -> tuple[Any, Any, float, float, float]:
```

- [ ] **Step 5: Edit `predict_next_day_lite.py` — annotate `load_models` and `predict_symbol`**

Add `Any` to the existing `from typing import Optional` import:

Old (line 20):
```python
from typing import Optional
```

New:
```python
from typing import Any, Optional
```

Change `load_models` return type (line 176):

Old:
```python
def load_models(
    db: Optional[DB] = None, model_keys: Optional[list] = None
) -> dict:
```

New:
```python
def load_models(
    db: Optional[DB] = None, model_keys: Optional[list[str]] = None
) -> dict[str, Any]:
```

Change `predict_symbol` signature (line 228):

Old:
```python
def predict_symbol(
    symbol: str,
    models: dict,
    db: Optional[DB] = None,
    dqn_window: int = 20,
    history_days: int = 1000,
    spy_df=None,
) -> dict:
```

New:
```python
def predict_symbol(
    symbol: str,
    models: dict[str, Any],
    db: Optional[DB] = None,
    dqn_window: int = 20,
    history_days: int = 1000,
    spy_df: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
```

- [ ] **Step 6: Run the test suite**

```
pytest tests/ -v
```

Expected: All pass, including all tests in `tests/test_code_quality.py`.

- [ ] **Step 7: Commit**

```bash
git add data_loader.py train_models.py predict_next_day_lite.py tests/test_code_quality.py
git commit -m "feat: add missing type annotations to public functions"
```

---

## Task 3: Interface Consistency

**Files:**
- Modify: `base_strategy.py`
- Modify: `simulation_pipeline.py`
- Extend: `tests/test_code_quality.py`

**Interfaces:**
- Consumes: Task 2's `tests/test_code_quality.py`
- Produces: `BaseStrategy.signal()` is annotated `-> pd.Series`; `BacktestResult.metrics` is `Dict[str, Any]`; `compute_metrics` return annotation matches

- [ ] **Step 1: Add interface consistency tests to `tests/test_code_quality.py`**

Append to the existing file:

```python
# ---------------------------------------------------------------------------
# Task 3 — Interface consistency
# ---------------------------------------------------------------------------

def test_base_strategy_signal_return_annotated():
    import pandas as pd
    from base_strategy import BaseStrategy
    hints = typing.get_type_hints(BaseStrategy.signal)
    assert hints.get("return") is pd.Series, (
        f"BaseStrategy.signal must be annotated '-> pd.Series', got {hints.get('return')}"
    )


def test_backtest_result_metrics_allows_none():
    """BacktestResult.metrics annotation must reflect that profit_factor can be None."""
    import dataclasses
    from simulation_pipeline import BacktestResult
    field_types = {f.name: f.type for f in dataclasses.fields(BacktestResult)}
    metrics_type = str(field_types.get("metrics", ""))
    assert "float" not in metrics_type or "Any" in metrics_type, (
        f"BacktestResult.metrics should be Dict[str, Any] (profit_factor is Optional), "
        f"but annotation is: {metrics_type}"
    )


def test_compute_metrics_profit_factor_none_when_no_losses():
    """compute_metrics must return None for profit_factor when gross_loss == 0."""
    import pandas as pd
    import numpy as np
    from simulation_pipeline import compute_metrics

    equity = pd.Series(
        [100_000.0, 100_100.0, 100_200.0],
        index=pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC"),
    )
    result = compute_metrics(equity, pd.DataFrame())
    assert result.get("profit_factor") is None, (
        "profit_factor should be None when there are no realized losses"
    )
```

- [ ] **Step 2: Run the new tests — confirm `test_backtest_result_metrics_allows_none` FAILS**

```
pytest tests/test_code_quality.py::test_base_strategy_signal_return_annotated \
       tests/test_code_quality.py::test_backtest_result_metrics_allows_none \
       tests/test_code_quality.py::test_compute_metrics_profit_factor_none_when_no_losses -v
```

Expected:
- `test_base_strategy_signal_return_annotated` — **PASS** (the annotation already exists in `base_strategy.py` line 27)
- `test_backtest_result_metrics_allows_none` — **FAIL** (`Dict[str, float]` annotation does not include `Any`)
- `test_compute_metrics_profit_factor_none_when_no_losses` — **PASS** (runtime behavior already correct)

- [ ] **Step 3: Confirm `base_strategy.py` — `signal()` already has `-> pd.Series` (no edit needed)**

Read `base_strategy.py` line 27. The current signature should read:
```python
    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
```

If the `-> pd.Series` annotation is present, no change is needed for this file. If it is absent, add it.

- [ ] **Step 4: Edit `simulation_pipeline.py` — fix `BacktestResult.metrics` type**

Add `Any` to the `Dict` import. The file currently has:
```python
from typing import Dict, Optional
```

Change to:
```python
from typing import Any, Dict, Optional
```

Old `BacktestResult` dataclass (lines 193–197):
```python
@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
```

New:
```python
@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, Any]
```

Old `compute_metrics` signature (line 456):
```python
def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
```

New:
```python
def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> Dict[str, Any]:
```

- [ ] **Step 5: Run the full test suite**

```
pytest tests/ -v
```

Expected: All tests pass. The three new tests in `tests/test_code_quality.py` (interface section) should now pass, and no existing tests should have broken.

- [ ] **Step 6: Commit**

```bash
git add base_strategy.py simulation_pipeline.py tests/test_code_quality.py
git commit -m "feat: fix interface contracts — BaseStrategy.signal return type and BacktestResult.metrics"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Covered by |
|---|---|
| Remove `OrdinalLogisticStrategy` and `XGBoostStrategy` | Task 1, Step 3 |
| Remove `_DailyRidgeQuantileStrategy` | Task 1, Step 4 Block B |
| Remove `walk_forward_backtest` | Task 1, Step 4 Block C |
| Remove `rsi` and `make_features` (implied collateral) | Task 1, Step 4 Block A |
| Type annotations on `data_loader.py` (save_to_csv gap) | Task 2, Step 3 |
| Type annotations on `train_models.py` (train_xgboost) | Task 2, Step 4 |
| Annotate `load_models` and `predict_symbol` | Task 2, Step 5 |
| Annotate `BaseStrategy.signal()` return | Task 3, Step 3 |
| Fix `BacktestResult.metrics` type | Task 3, Step 4 |
| One PR per logical group | Tasks 1/2/3 produce separate commits |

### Placeholder scan

None — all steps contain full code.

### Type consistency

- `dict[str, Any]` used consistently for `load_models` return, `predict_symbol` params and return, and `BacktestResult.metrics`.
- `tuple[Any, Any, float, float, float]` used for `train_xgboost` return — matches the 5 values returned: `(model, scaler, train_acc, test_acc, test_f1)`.
- `Dict[str, Any]` used in `BacktestResult.metrics` and `compute_metrics` return (both use `typing.Dict` since that import already exists).
