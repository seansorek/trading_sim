# Signal Integrity & Walk-Forward Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add walk-forward validation, tuned signal parameters, realized IC tracking, and drift detection to the daily predictor pipeline.

**Architecture:** A new `walk_forward.py` module provides a pure, testable fold-iteration engine and parameter sweep. A new `signal_monitor.py` module handles realized IC scoring and drift detection. Both are wired into `predict_next_day_lite.py`; best parameters flow from `walk_forward.py` → `train_predictor.py` → pickle → live pipeline.

**Tech Stack:** Python 3.10, scikit-learn (Ridge, StandardScaler), scipy (spearmanr), pandas, numpy, sqlite3, existing `daily_features.FEATURE_COLS`, `ml_strategies.compute_predictor_signal`, `train_models._preprocess/_load_symbol`.

## Global Constraints

- Python 3.10 — no 3.11+ syntax
- All DB access via the `DB` class in `db.py` — never raw sqlite3 outside it
- Feature indexing must use `FEATURE_COLS` from `daily_features.py` — never column position
- Embargo gap must equal `FWD_RET_HORIZON_DAYS` (currently 3) — import, never hardcode
- Three-level param priority must be preserved everywhere: `env var → pickle key → hardcoded default`
- `compute_predictor_signal` in `ml_strategies.py` is the single source of truth for the decision layer — never duplicate it
- Tests use `tmp_path` (pytest fixture) for DB; never write to `data/trading_sim.db` in tests
- No network calls in tests — mock `load_yfinance` / `_load_symbol` with `unittest.mock.patch`
- Run tests with: `.venv/Scripts/python.exe -m pytest tests/ -v` (Windows venv)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `db.py` | Modify | Add `ic_history` table, `upsert_ic()`, `get_ic_history()` |
| `walk_forward.py` | Create | `WalkForwardConfig`, `run_walk_forward_on_df()`, `sweep_params()`, CLI |
| `signal_monitor.py` | Create | `score_realized_ic()`, `check_signal_drift()` |
| `train_predictor.py` | Modify | Call `sweep_params` after training; pass best params to `_save_and_register` |
| `ml_strategies.py` | Modify | `DailyPredictorStrategy` reads best params from pickle (three-level priority) |
| `predict_next_day_lite.py` | Modify | Wire IC scorer + drift detector; pass best params from pickle to regressor; update Discord |
| `tests/test_db.py` | Modify | Add `upsert_ic` / `get_ic_history` tests |
| `tests/test_walk_forward.py` | Create | Fold iteration, embargo gap, sweep, edge cases |
| `tests/test_ic_tracking.py` | Create | `score_realized_ic` with mocked price fetch |
| `tests/test_drift_detection.py` | Create | `check_signal_drift` — normal, single-day, two-day, degenerate |
| `tests/test_predictor.py` | Modify | Two backward-compat cases for best-params pickle keys |

---

## Task 1: DB — `ic_history` table, `upsert_ic`, `get_ic_history`

**Files:**
- Modify: `db.py`
- Modify: `tests/test_db.py`

**Interfaces:**
- Produces:
  - `DB.upsert_ic(model: str, computed_at: str, lookback_n: int, ic: float, directional_accuracy: float, mean_pred: float | None, std_pred: float | None) -> None`
  - `DB.get_ic_history(model: str, limit: int = 30) -> pd.DataFrame` — columns: `model, computed_at, lookback_n, ic, directional_accuracy, mean_pred, std_pred`

- [ ] **Step 1: Write failing tests**

In `tests/test_db.py`, add after the existing tests:

```python
def test_upsert_and_get_ic_history(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    db.upsert_ic("daily_predictor", "2026-07-01", 20, 0.07, 0.54, 0.012, 0.008)
    df = db.get_ic_history("daily_predictor")
    assert len(df) == 1
    assert df.iloc[0]["ic"] == pytest.approx(0.07)
    assert df.iloc[0]["directional_accuracy"] == pytest.approx(0.54)
    assert df.iloc[0]["lookback_n"] == 20


def test_upsert_ic_is_idempotent(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    db.upsert_ic("daily_predictor", "2026-07-01", 20, 0.07, 0.54, None, None)
    db.upsert_ic("daily_predictor", "2026-07-01", 20, 0.09, 0.56, None, None)  # update
    df = db.get_ic_history("daily_predictor")
    assert len(df) == 1
    assert df.iloc[0]["ic"] == pytest.approx(0.09)


def test_get_ic_history_empty(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    df = db.get_ic_history("no_such_model")
    assert df.empty


def test_get_ic_history_respects_limit(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    for i in range(5):
        db.upsert_ic("daily_predictor", f"2026-07-0{i+1}", 20, 0.01 * i, 0.5, None, None)
    df = db.get_ic_history("daily_predictor", limit=3)
    assert len(df) == 3
```

- [ ] **Step 2: Run tests to confirm they fail**

```
.venv/Scripts/python.exe -m pytest tests/test_db.py::test_upsert_and_get_ic_history tests/test_db.py::test_upsert_ic_is_idempotent tests/test_db.py::test_get_ic_history_empty tests/test_db.py::test_get_ic_history_respects_limit -v
```

Expected: `AttributeError: 'DB' object has no attribute 'upsert_ic'`

- [ ] **Step 3: Add `ic_history` table to `_SCHEMA` in `db.py`**

Append to the end of the `_SCHEMA` string, before the closing `"""`:

```python
CREATE TABLE IF NOT EXISTS ic_history (
    id                   INTEGER PRIMARY KEY,
    model                TEXT    NOT NULL,
    computed_at          TEXT    NOT NULL,
    lookback_n           INTEGER NOT NULL,
    ic                   REAL    NOT NULL,
    directional_accuracy REAL    NOT NULL,
    mean_pred            REAL,
    std_pred             REAL,
    UNIQUE(model, computed_at)
);
CREATE INDEX IF NOT EXISTS ic_history_model_date ON ic_history(model, computed_at);
```

- [ ] **Step 4: Add `upsert_ic` and `get_ic_history` methods to the `DB` class in `db.py`**

Add after the `get_predictions` method (before the `# Utilities` section):

```python
# ------------------------------------------------------------------
# IC history
# ------------------------------------------------------------------

def upsert_ic(
    self,
    model: str,
    computed_at: str,
    lookback_n: int,
    ic: float,
    directional_accuracy: float,
    mean_pred: float | None = None,
    std_pred: float | None = None,
) -> None:
    sql = """
        INSERT INTO ic_history
          (model, computed_at, lookback_n, ic, directional_accuracy, mean_pred, std_pred)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(model, computed_at) DO UPDATE SET
          lookback_n=excluded.lookback_n,
          ic=excluded.ic,
          directional_accuracy=excluded.directional_accuracy,
          mean_pred=excluded.mean_pred,
          std_pred=excluded.std_pred
    """
    with self._lock, self._connect() as con:
        con.execute(sql, (model, computed_at, lookback_n, ic,
                          directional_accuracy, mean_pred, std_pred))

def get_ic_history(self, model: str, limit: int = 30) -> pd.DataFrame:
    sql = """
        SELECT * FROM ic_history WHERE model=?
        ORDER BY computed_at DESC LIMIT ?
    """
    with self._lock, self._connect() as con:
        rows = con.execute(sql, (model, limit)).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
```

- [ ] **Step 5: Run tests to confirm they pass**

```
.venv/Scripts/python.exe -m pytest tests/test_db.py -v
```

Expected: all existing + 4 new tests PASS.

- [ ] **Step 6: Commit**

```
git add db.py tests/test_db.py
git commit -m "feat: add ic_history table with upsert_ic and get_ic_history"
```

---

## Task 2: `walk_forward.py` — core fold engine

**Files:**
- Create: `walk_forward.py`
- Create: `tests/test_walk_forward.py` (partial — fold tests only)

**Interfaces:**
- Consumes: `daily_features.{FEATURE_COLS, FWD_RET_HORIZON_DAYS, make_daily_features}`, `train_models._preprocess`
- Produces:
  - `WalkForwardConfig` dataclass with fields: `train_bars: int = 504`, `test_bars: int = 63`, `step_bars: int = 21`, `min_train_bars: int = 252`, `ridge_alpha: float = 10.0`
  - `run_walk_forward_on_df(df: pd.DataFrame, spy_df: pd.DataFrame | None, config: WalkForwardConfig) -> pd.DataFrame` — columns: `fold, train_start, train_end, test_start, test_end, ic, dir_acc, n_test`. Raises `ValueError` if fewer bars than one fold.

- [ ] **Step 1: Write failing tests**

Create `tests/test_walk_forward.py`:

```python
"""test_walk_forward.py — Walk-forward harness tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from walk_forward import WalkForwardConfig, run_walk_forward_on_df
from daily_features import FWD_RET_HORIZON_DAYS


def _sine_price_df(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Synthetic price series with a weak predictable component."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # Sine trend gives Ridge something to learn
    close = 100 + np.sin(t * 0.05) * 5 + rng.normal(0, 0.5, n).cumsum()
    close = np.abs(close) + 10  # keep positive
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


def test_run_walk_forward_returns_dataframe_with_expected_columns():
    df = _sine_price_df(800)
    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    result = run_walk_forward_on_df(df, spy_df=None, config=config)
    assert isinstance(result, pd.DataFrame)
    for col in ("fold", "train_start", "train_end", "test_start", "test_end", "ic", "dir_acc", "n_test"):
        assert col in result.columns, f"Missing column: {col}"


def test_run_walk_forward_produces_at_least_one_fold():
    df = _sine_price_df(800)
    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    result = run_walk_forward_on_df(df, spy_df=None, config=config)
    assert len(result) >= 1


def test_run_walk_forward_raises_on_insufficient_bars():
    df = _sine_price_df(100)  # too short for any fold
    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    with pytest.raises(ValueError, match="bars"):
        run_walk_forward_on_df(df, spy_df=None, config=config)


def test_embargo_gap_enforced():
    """No training label should depend on price action inside the test window.
    The gap between train_end and test_start must be >= FWD_RET_HORIZON_DAYS."""
    df = _sine_price_df(800)
    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    result = run_walk_forward_on_df(df, spy_df=None, config=config)
    for _, row in result.iterrows():
        train_end = pd.Timestamp(row["train_end"])
        test_start = pd.Timestamp(row["test_start"])
        gap_days = (test_start - train_end).days
        assert gap_days >= FWD_RET_HORIZON_DAYS, (
            f"Embargo gap {gap_days} < FWD_RET_HORIZON_DAYS={FWD_RET_HORIZON_DAYS}"
        )


def test_ic_values_are_finite():
    df = _sine_price_df(800)
    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    result = run_walk_forward_on_df(df, spy_df=None, config=config)
    assert result["ic"].notna().all()
    assert result["dir_acc"].between(0.0, 1.0).all()
```

- [ ] **Step 2: Run tests to confirm they fail**

```
.venv/Scripts/python.exe -m pytest tests/test_walk_forward.py -v
```

Expected: `ModuleNotFoundError: No module named 'walk_forward'`

- [ ] **Step 3: Create `walk_forward.py`**

```python
#!/usr/bin/env python3
"""
walk_forward.py — Walk-forward validation harness for the daily predictor.

Provides run_walk_forward_on_df (pure, no I/O) for testing and sweep_params
(loads data, tunes signal_quantile + threshold_window) for use by train_predictor.py.

CLI:
    python walk_forward.py --symbol SPY --train 504 --test 63 --step 21
    python walk_forward.py --symbols AAPL,MSFT,SPY --train 504 --test 63 --step 21
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from daily_features import FEATURE_COLS, FWD_RET_HORIZON_DAYS, make_daily_features
from ml_strategies import compute_predictor_signal
from train_models import _preprocess, _load_symbol

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    train_bars: int = 504
    test_bars: int = 63
    step_bars: int = 21
    min_train_bars: int = 252
    ridge_alpha: float = 10.0


def _ic(pred: np.ndarray, actual: np.ndarray) -> float:
    if len(pred) < 2 or np.std(pred) < 1e-12:
        return 0.0
    ic, _ = spearmanr(pred, actual)
    return 0.0 if np.isnan(ic) else float(ic)


def run_walk_forward_on_df(
    df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame],
    config: WalkForwardConfig,
) -> pd.DataFrame:
    """
    Run walk-forward validation on a single symbol's price DataFrame.

    Returns a DataFrame with one row per fold:
      fold, train_start, train_end, test_start, test_end, ic, dir_acc, n_test

    Raises ValueError if there are fewer bars than one complete fold
    (train_bars + FWD_RET_HORIZON_DAYS + test_bars).
    """
    feats = make_daily_features(df, spy_df=spy_df).dropna(subset=["fwd_ret_1d"])
    n = len(feats)
    min_bars = config.train_bars + FWD_RET_HORIZON_DAYS + config.test_bars
    if n < min_bars:
        raise ValueError(
            f"Need at least {min_bars} bars for one fold "
            f"(train={config.train_bars} + embargo={FWD_RET_HORIZON_DAYS} + test={config.test_bars}), "
            f"got {n}."
        )

    X_all = _preprocess(feats[FEATURE_COLS].values.astype(np.float32))
    y_all = feats["fwd_ret_1d"].values.astype(np.float64)
    dates = feats.index

    records = []
    fold = 0
    train_start_idx = 0

    while True:
        train_end_idx = train_start_idx + config.train_bars
        test_start_idx = train_end_idx + FWD_RET_HORIZON_DAYS
        test_end_idx = test_start_idx + config.test_bars
        if test_end_idx > n:
            break

        X_tr = X_all[train_start_idx:train_end_idx]
        y_tr = y_all[train_start_idx:train_end_idx]
        X_te = X_all[test_start_idx:test_end_idx]
        y_te = y_all[test_start_idx:test_end_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = Ridge(alpha=config.ridge_alpha)
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_te_s)

        fold_ic = _ic(pred, y_te)
        dir_acc = float(np.mean(np.sign(pred) == np.sign(y_te)))

        records.append({
            "fold": fold,
            "train_start": str(dates[train_start_idx].date()),
            "train_end": str(dates[train_end_idx - 1].date()),
            "test_start": str(dates[test_start_idx].date()),
            "test_end": str(dates[test_end_idx - 1].date()),
            "ic": fold_ic,
            "dir_acc": dir_acc,
            "n_test": config.test_bars,
        })
        fold += 1
        train_start_idx += config.step_bars

    return pd.DataFrame(records)
```

- [ ] **Step 4: Run tests to confirm they pass**

```
.venv/Scripts/python.exe -m pytest tests/test_walk_forward.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```
git add walk_forward.py tests/test_walk_forward.py
git commit -m "feat: add walk_forward.py core fold engine with embargo gap"
```

---

## Task 3: `walk_forward.py` — parameter sweep + CLI

**Files:**
- Modify: `walk_forward.py`
- Modify: `tests/test_walk_forward.py`

**Interfaces:**
- Consumes: `WalkForwardConfig`, `run_walk_forward_on_df`, `_load_symbol`, `compute_predictor_signal`
- Produces:
  - `sweep_params(symbols: list[str], days: int, db, config: WalkForwardConfig | None, quantiles: list[float] | None, windows: list[int] | None) -> tuple[float, int]` — returns `(best_signal_quantile, best_threshold_window)`, falls back to `(0.7, 60)` if all IC ≤ 0

- [ ] **Step 1: Write failing tests**

Add to `tests/test_walk_forward.py`:

```python
from unittest.mock import patch
from walk_forward import sweep_params, WalkForwardConfig


def _mock_load_symbol(symbol, start, end, db):
    return _sine_price_df(800)


def test_sweep_params_returns_valid_pair():
    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    with patch("walk_forward._load_symbol", side_effect=_mock_load_symbol):
        q, w = sweep_params(
            ["AAPL"], days=900, db=None, config=config,
            quantiles=[0.65, 0.70], windows=[40, 60],
        )
    assert q in [0.65, 0.70]
    assert w in [40, 60]


def test_sweep_params_falls_back_when_all_ic_nonpositive():
    """All-constant predictions → IC = 0 for every param pair → fallback to defaults."""
    # Flat price series → Ridge predicts near-constant → IC ≈ 0
    n = 800
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    flat_df = pd.DataFrame({
        "open": np.full(n, 100.0), "high": np.full(n, 100.1),
        "low": np.full(n, 99.9), "close": np.full(n, 100.0),
        "volume": np.full(n, 1_000_000.0),
    }, index=idx)

    config = WalkForwardConfig(train_bars=400, test_bars=63, step_bars=63)
    with patch("walk_forward._load_symbol", return_value=flat_df):
        q, w = sweep_params(
            ["AAPL"], days=900, db=None, config=config,
            quantiles=[0.65, 0.70], windows=[40, 60],
        )
    assert q == 0.7
    assert w == 60


def test_sweep_params_no_valid_symbols_returns_defaults():
    with patch("walk_forward._load_symbol", return_value=None):
        q, w = sweep_params(["AAPL"], days=900, db=None)
    assert q == 0.7
    assert w == 60
```

- [ ] **Step 2: Run tests to confirm they fail**

```
.venv/Scripts/python.exe -m pytest tests/test_walk_forward.py::test_sweep_params_returns_valid_pair tests/test_walk_forward.py::test_sweep_params_falls_back_when_all_ic_nonpositive tests/test_walk_forward.py::test_sweep_params_no_valid_symbols_returns_defaults -v
```

Expected: `ImportError: cannot import name 'sweep_params'`

- [ ] **Step 3: Add `sweep_params` and CLI to `walk_forward.py`**

Append to `walk_forward.py` after `run_walk_forward_on_df`:

```python
_DEFAULT_QUANTILES = [0.60, 0.65, 0.70, 0.75, 0.80]
_DEFAULT_WINDOWS = [40, 60, 80, 100]
_FALLBACK_QUANTILE = 0.7
_FALLBACK_WINDOW = 60


def sweep_params(
    symbols: list[str],
    days: int,
    db,
    config: Optional[WalkForwardConfig] = None,
    quantiles: Optional[list[float]] = None,
    windows: Optional[list[int]] = None,
) -> tuple[float, int]:
    """
    Sweep (signal_quantile, threshold_window) over walk-forward folds for all symbols.
    Returns (best_signal_quantile, best_threshold_window) based on median IC across symbols.
    Falls back to (0.7, 60) if no symbols produce valid folds or all IC <= 0.
    """
    if config is None:
        config = WalkForwardConfig()
    if quantiles is None:
        quantiles = _DEFAULT_QUANTILES
    if windows is None:
        windows = _DEFAULT_WINDOWS

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    spy_df = _load_symbol("SPY", start, end, db)

    # Per fold: store (pred_window, y_te, test_offset) so the rolling quantile
    # has history before the test window — matches live production behaviour.
    fold_data: list[tuple[np.ndarray, np.ndarray, int]] = []

    for symbol in symbols:
        df = _load_symbol(symbol, start, end, db)
        if df is None:
            continue
        try:
            spy_arg = spy_df if symbol != "SPY" else None
            feats = make_daily_features(df, spy_df=spy_arg).dropna(subset=["fwd_ret_1d"])
        except Exception as exc:
            logger.warning("sweep_params: feature error for %s: %s", symbol, exc)
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

            # Fit scaler only on train, transform full window for causal rolling quantile
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

    if not fold_data:
        logger.warning("sweep_params: no valid folds — returning defaults (%s, %d)",
                       _FALLBACK_QUANTILE, _FALLBACK_WINDOW)
        return _FALLBACK_QUANTILE, _FALLBACK_WINDOW

    best_ic = -np.inf
    best_pair = (_FALLBACK_QUANTILE, _FALLBACK_WINDOW)

    for q in quantiles:
        for w in windows:
            fold_ics = []
            for pred_window, y_te, test_offset in fold_data:
                if len(pred_window) < w:
                    continue
                signals_window = compute_predictor_signal(pred_window, q, w)
                signals_test = signals_window[test_offset:]
                active = signals_test != 0
                if active.sum() < 5:
                    fold_ics.append(0.0)
                    continue
                fold_ics.append(_ic(pred_window[test_offset:][active], y_te[active]))
            if not fold_ics:
                continue
            median_ic = float(np.median(fold_ics))
            if median_ic > best_ic:
                best_ic = median_ic
                best_pair = (q, w)

    if best_ic <= 0:
        logger.warning(
            "sweep_params: all (quantile, window) combinations produced IC <= 0 "
            "— keeping defaults (%.2f, %d)", _FALLBACK_QUANTILE, _FALLBACK_WINDOW
        )
        return _FALLBACK_QUANTILE, _FALLBACK_WINDOW

    logger.info("sweep_params: best q=%.2f w=%d median_IC=%.4f", best_pair[0], best_pair[1], best_ic)
    return best_pair


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Walk-forward validation for daily predictor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", help="Single symbol")
    group.add_argument("--symbols", help="Comma-separated symbols")
    parser.add_argument("--train", type=int, default=504, dest="train_bars")
    parser.add_argument("--test", type=int, default=63, dest="test_bars")
    parser.add_argument("--step", type=int, default=21, dest="step_bars")
    parser.add_argument("--days", type=int, default=2500)
    parser.add_argument("--db", default="data/trading_sim.db")
    args = parser.parse_args()

    from db import DB
    db = DB(args.db)
    config = WalkForwardConfig(train_bars=args.train_bars, test_bars=args.test_bars,
                               step_bars=args.step_bars)

    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    spy_df = _load_symbol("SPY", start, end, db)

    for symbol in symbols:
        df = _load_symbol(symbol, start, end, db)
        if df is None:
            logger.warning("Could not load data for %s", symbol)
            continue
        spy_arg = spy_df if symbol != "SPY" else None
        try:
            result = run_walk_forward_on_df(df, spy_arg, config)
        except ValueError as exc:
            logger.error("%s: %s", symbol, exc)
            continue
        print(f"\n=== {symbol} Walk-Forward Results ===")
        print(result[["fold", "train_start", "test_start", "ic", "dir_acc"]].to_string(index=False))
        print(f"Mean IC: {result['ic'].mean():.4f}  Median IC: {result['ic'].median():.4f}")

    if len(symbols) > 1 or args.symbols:
        print("\n=== Parameter Sweep ===")
        best_q, best_w = sweep_params(symbols, args.days, db, config)
        print(f"Best: signal_quantile={best_q:.2f}  threshold_window={best_w}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all walk-forward tests**

```
.venv/Scripts/python.exe -m pytest tests/test_walk_forward.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```
git add walk_forward.py tests/test_walk_forward.py
git commit -m "feat: add sweep_params and CLI to walk_forward.py"
```

---

## Task 4: Save best params into pickle; read them with three-level priority

**Files:**
- Modify: `train_predictor.py`
- Modify: `ml_strategies.py`
- Modify: `predict_next_day_lite.py`
- Modify: `tests/test_predictor.py`

**Interfaces:**
- Consumes: `sweep_params`, `WalkForwardConfig` from `walk_forward.py`
- Produces: pickle artifact gains optional keys `best_signal_quantile: float` and `best_threshold_window: int`. Both `DailyPredictorStrategy` and `_predict_regressor_signal` read them at load time with three-level priority.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_predictor.py`:

```python
def test_predictor_strategy_reads_best_params_from_pickle(tmp_path):
    """DailyPredictorStrategy must use best_signal_quantile/best_threshold_window
    from the pickle when env vars are not set — second priority level."""
    import pickle, os
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from daily_features import FEATURE_COLS

    # Build a minimal valid pickle with best params
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, len(FEATURE_COLS))).astype(np.float32)
    y = rng.normal(size=100)
    scaler = StandardScaler()
    model = Ridge().fit(scaler.fit_transform(X), y)
    artifact = {
        "model": model, "scaler": scaler,
        "feature_contract": FEATURE_COLS,
        "best_signal_quantile": 0.65,
        "best_threshold_window": 40,
    }
    pkl_path = str(tmp_path / "predictor.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artifact, f)

    env_backup = {k: os.environ.pop(k, None)
                  for k in ("PREDICTOR_SIGNAL_QUANTILE", "PREDICTOR_THRESHOLD_WINDOW")}
    try:
        cfg = StrategyConfig(name="daily_predictor")
        strat = DailyPredictorStrategy(cfg, use_pretrained=True, model_path=pkl_path)
        assert strat.signal_quantile == pytest.approx(0.65)
        assert strat.threshold_window == 40
    finally:
        for k, v in env_backup.items():
            if v is not None:
                os.environ[k] = v


def test_predictor_strategy_old_pickle_falls_back_to_defaults(tmp_path):
    """Pickle without best_signal_quantile/best_threshold_window keys must
    fall back to hardcoded defaults without raising."""
    import pickle
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from daily_features import FEATURE_COLS

    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, len(FEATURE_COLS))).astype(np.float32)
    y = rng.normal(size=100)
    scaler = StandardScaler()
    model = Ridge().fit(scaler.fit_transform(X), y)
    # Old-format pickle — no best_* keys
    artifact = {"model": model, "scaler": scaler, "feature_contract": FEATURE_COLS}
    pkl_path = str(tmp_path / "old_predictor.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artifact, f)

    import os
    env_backup = {k: os.environ.pop(k, None)
                  for k in ("PREDICTOR_SIGNAL_QUANTILE", "PREDICTOR_THRESHOLD_WINDOW")}
    try:
        cfg = StrategyConfig(name="daily_predictor")
        strat = DailyPredictorStrategy(cfg, use_pretrained=True, model_path=pkl_path)
        assert strat.signal_quantile == pytest.approx(0.7)
        assert strat.threshold_window == 60
    finally:
        for k, v in env_backup.items():
            if v is not None:
                os.environ[k] = v
```

- [ ] **Step 2: Run tests to confirm they fail**

```
.venv/Scripts/python.exe -m pytest tests/test_predictor.py::test_predictor_strategy_reads_best_params_from_pickle tests/test_predictor.py::test_predictor_strategy_old_pickle_falls_back_to_defaults -v
```

Expected: both FAIL (strategy still uses only env var / constructor default).

- [ ] **Step 3: Update `DailyPredictorStrategy.__init__` in `ml_strategies.py`**

Replace the current param-loading block in `DailyPredictorStrategy.__init__`:

```python
# BEFORE (lines ~189-190):
self.signal_quantile = float(os.environ.get("PREDICTOR_SIGNAL_QUANTILE", signal_quantile))
self.threshold_window = int(os.environ.get("PREDICTOR_THRESHOLD_WINDOW", threshold_window))
```

with:

```python
# Params set here before pickle load; updated below if pickle has tuned values.
self.signal_quantile = signal_quantile
self.threshold_window = threshold_window
```

Then after `self.scaler = data["scaler"]` add:

```python
    # Three-level priority: env var → pickle best_* keys → constructor default
    sq_env = os.environ.get("PREDICTOR_SIGNAL_QUANTILE")
    if sq_env is not None:
        try:
            self.signal_quantile = float(sq_env)
        except ValueError:
            pass
    elif "best_signal_quantile" in data:
        self.signal_quantile = float(data["best_signal_quantile"])

    tw_env = os.environ.get("PREDICTOR_THRESHOLD_WINDOW")
    if tw_env is not None:
        try:
            self.threshold_window = int(tw_env)
        except ValueError:
            pass
    elif "best_threshold_window" in data:
        self.threshold_window = int(data["best_threshold_window"])
```

Also update the `else` branch (no pretrained model) to still apply env vars:

```python
else:
    sq_env = os.environ.get("PREDICTOR_SIGNAL_QUANTILE")
    if sq_env is not None:
        try:
            self.signal_quantile = float(sq_env)
        except ValueError:
            pass
    tw_env = os.environ.get("PREDICTOR_THRESHOLD_WINDOW")
    if tw_env is not None:
        try:
            self.threshold_window = int(tw_env)
        except ValueError:
            pass
```

- [ ] **Step 4: Update `_predict_regressor_signal` in `predict_next_day_lite.py`**

Replace the existing env-var reading block (lines ~150-175) with:

```python
sq_env = os.environ.get("PREDICTOR_SIGNAL_QUANTILE")
if sq_env is not None:
    try:
        sq = float(sq_env)
    except ValueError:
        logger.warning("Invalid PREDICTOR_SIGNAL_QUANTILE=%r — ignoring", sq_env)
        sq = float(data.get("best_signal_quantile", signal_quantile))
else:
    sq = float(data.get("best_signal_quantile", signal_quantile))

tw_env = os.environ.get("PREDICTOR_THRESHOLD_WINDOW")
if tw_env is not None:
    try:
        tw = int(tw_env)
    except ValueError:
        logger.warning("Invalid PREDICTOR_THRESHOLD_WINDOW=%r — ignoring", tw_env)
        tw = int(data.get("best_threshold_window", threshold_window))
else:
    tw = int(data.get("best_threshold_window", threshold_window))
```

- [ ] **Step 5: Update `_save_and_register` in `train_predictor.py` to accept and store best params**

Change the function signature from:

```python
def _save_and_register(
    model_obj, scaler, model_type: str, train_metrics: dict, test_metrics: dict,
    train_symbols: list[str], days: int, db: DB, train_samples: int, test_samples: int,
) -> int:
```

to:

```python
def _save_and_register(
    model_obj, scaler, model_type: str, train_metrics: dict, test_metrics: dict,
    train_symbols: list[str], days: int, db: DB, train_samples: int, test_samples: int,
    best_signal_quantile: float = 0.7,
    best_threshold_window: int = 60,
) -> int:
```

Then add to the `artifact` dict (after the existing keys):

```python
"best_signal_quantile": best_signal_quantile,
"best_threshold_window": best_threshold_window,
```

- [ ] **Step 6: Call `sweep_params` in `train_predictor.main()` and pass results to `_save_and_register`**

In `train_predictor.main()`, after the `prepare_data` call and before the training loop, add:

```python
from walk_forward import sweep_params, WalkForwardConfig
logger.info("Running walk-forward parameter sweep...")
try:
    wf_config = WalkForwardConfig()
    best_q, best_w = sweep_params(symbols, args.days, db, wf_config)
    logger.info("Parameter sweep complete: signal_quantile=%.2f threshold_window=%d", best_q, best_w)
except Exception as exc:
    logger.warning("Parameter sweep failed: %s — using defaults (0.7, 60)", exc)
    best_q, best_w = 0.7, 60
```

Then in the training loop, pass them to `_save_and_register`:

```python
_save_and_register(
    model, scaler, model_type, train_m, test_m,
    data["used_symbols"], args.days, db,
    train_samples=len(X_train), test_samples=len(X_test),
    best_signal_quantile=best_q,
    best_threshold_window=best_w,
)
```

- [ ] **Step 7: Run all predictor tests**

```
.venv/Scripts/python.exe -m pytest tests/test_predictor.py -v
```

Expected: all existing + 2 new tests PASS.

- [ ] **Step 8: Commit**

```
git add train_predictor.py ml_strategies.py predict_next_day_lite.py tests/test_predictor.py
git commit -m "feat: save best params from sweep into pickle; three-level priority in predictor loading"
```

---

## Task 5: `signal_monitor.py` — realized IC scorer

**Files:**
- Create: `signal_monitor.py`
- Create: `tests/test_ic_tracking.py`

**Interfaces:**
- Produces:
  - `score_realized_ic(history_path: str, today_date: str, fetch_prices_fn: callable, min_lookback: int = 20, fwd_ret_horizon: int = FWD_RET_HORIZON_DAYS) -> dict[str, dict | None]`
    - Returns `{model_key: {"ic": float, "directional_accuracy": float, "lookback_n": int, "mean_pred": float, "std_pred": float} | None}`
    - `fetch_prices_fn(symbol: str, start: str, end: str) -> pd.DataFrame | None` — `close` column required
  - `_signal_to_score(signal: str, confidence: float) -> float` — BUY→+conf, SELL→-conf, HOLD→0

- [ ] **Step 1: Write failing tests**

Create `tests/test_ic_tracking.py`:

```python
"""test_ic_tracking.py — Tests for score_realized_ic in signal_monitor.py."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_monitor import score_realized_ic, _signal_to_score
from daily_features import FWD_RET_HORIZON_DAYS


def _write_history(tmp_path, records: list[dict]) -> str:
    path = tmp_path / "history.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


def _make_price_df_from_close(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"close": closes}, index=idx)


def test_signal_to_score_buy():
    assert _signal_to_score("BUY", 0.8) == pytest.approx(0.8)


def test_signal_to_score_sell():
    assert _signal_to_score("SELL", 0.8) == pytest.approx(-0.8)


def test_signal_to_score_hold():
    assert _signal_to_score("HOLD", 0.9) == pytest.approx(0.0)


def test_score_realized_ic_returns_none_when_insufficient_history(tmp_path):
    today = date.today().strftime("%Y-%m-%d")
    # Only 5 records — below min_lookback=20
    old_date = (date.today() - timedelta(days=FWD_RET_HORIZON_DAYS + 10)).strftime("%Y-%m-%d")
    records = [
        {"date": old_date, "symbol": "AAPL", "model": "daily_predictor",
         "signal": "BUY", "confidence": 0.7, "price": 100.0}
        for _ in range(5)
    ]
    path = _write_history(tmp_path, records)
    result = score_realized_ic(path, today, fetch_prices_fn=lambda s, st, en: None)
    assert result["daily_predictor"] is None


def test_score_realized_ic_excludes_recent_entries(tmp_path):
    """Entries within FWD_RET_HORIZON_DAYS of today cannot be scored yet."""
    today = date.today()
    records = []
    # 20 records that are too recent
    for i in range(20):
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        records.append({"date": d, "symbol": "AAPL", "model": "daily_predictor",
                        "signal": "BUY", "confidence": 0.7, "price": 100.0})
    path = _write_history(tmp_path, records)
    result = score_realized_ic(path, today.strftime("%Y-%m-%d"),
                               fetch_prices_fn=lambda s, st, en: None)
    assert result.get("daily_predictor") is None


def test_score_realized_ic_computes_correct_ic(tmp_path):
    """When BUY predictions are followed by positive returns, IC should be positive."""
    today = date.today()
    cutoff_date = today - timedelta(days=FWD_RET_HORIZON_DAYS + 1)
    records = []
    # 20 BUY predictions on days where price actually went up
    for i in range(20):
        d = (cutoff_date - timedelta(days=i)).strftime("%Y-%m-%d")
        records.append({"date": d, "symbol": "AAPL", "model": "daily_predictor",
                        "signal": "BUY", "confidence": 0.7 + i * 0.01, "price": 100.0})

    path = _write_history(tmp_path, records)

    # Mock: each fetch returns prices where realized_return is positive
    def mock_fetch(symbol, start, end):
        # Return FWD_RET_HORIZON_DAYS+1 bars; close[FWD_RET_HORIZON_DAYS] > close[0]
        closes = [100.0] + [100.0] * (FWD_RET_HORIZON_DAYS - 1) + [102.0]
        return _make_price_df_from_close(closes)

    result = score_realized_ic(path, today.strftime("%Y-%m-%d"), fetch_prices_fn=mock_fetch)
    assert result["daily_predictor"] is not None
    assert result["daily_predictor"]["ic"] > 0


def test_score_realized_ic_excludes_near_zero_returns(tmp_path):
    """Rows where |realized_return| < 1e-5 (likely holidays) are excluded."""
    today = date.today()
    cutoff_date = today - timedelta(days=FWD_RET_HORIZON_DAYS + 1)
    records = []
    for i in range(20):
        d = (cutoff_date - timedelta(days=i)).strftime("%Y-%m-%d")
        records.append({"date": d, "symbol": "AAPL", "model": "daily_predictor",
                        "signal": "BUY", "confidence": 0.7, "price": 100.0})
    path = _write_history(tmp_path, records)

    # All fetched prices show zero return (flat)
    def mock_flat_fetch(symbol, start, end):
        return _make_price_df_from_close([100.0] * (FWD_RET_HORIZON_DAYS + 1))

    result = score_realized_ic(path, today.strftime("%Y-%m-%d"), fetch_prices_fn=mock_flat_fetch)
    # All rows excluded → insufficient scored rows → None
    assert result.get("daily_predictor") is None


def test_score_realized_ic_handles_missing_file(tmp_path):
    result = score_realized_ic(str(tmp_path / "nonexistent.jsonl"),
                               date.today().strftime("%Y-%m-%d"),
                               fetch_prices_fn=lambda s, st, en: None)
    assert result == {}
```

- [ ] **Step 2: Run tests to confirm they fail**

```
.venv/Scripts/python.exe -m pytest tests/test_ic_tracking.py -v
```

Expected: `ModuleNotFoundError: No module named 'signal_monitor'`

- [ ] **Step 3: Create `signal_monitor.py`**

```python
"""
signal_monitor.py — Realized IC scoring and signal drift detection.

score_realized_ic: scores trailing predictions against realized returns.
check_signal_drift: detects if today's predicted-return distribution has shifted.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.stats import spearmanr

from daily_features import FWD_RET_HORIZON_DAYS

logger = logging.getLogger(__name__)

MIN_LOOKBACK = 20
MIN_DRIFT_WINDOW = 10
DRIFT_SIGMA_THRESHOLD = 2.0
DRIFT_ABS_THRESHOLD = 0.002


def _load_history(history_path: str) -> list[dict]:
    path = Path(history_path)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _spearman_ic(scores: np.ndarray, actuals: np.ndarray) -> float:
    if len(scores) < 2 or np.std(scores) < 1e-12:
        return 0.0
    ic, _ = spearmanr(scores, actuals)
    return 0.0 if np.isnan(ic) else float(ic)


def _signal_to_score(signal: str, confidence: float) -> float:
    """Map signal + confidence to a directional score for IC computation."""
    if signal == "BUY":
        return float(confidence)
    if signal == "SELL":
        return -float(confidence)
    return 0.0


def score_realized_ic(
    history_path: str,
    today_date: str,
    fetch_prices_fn: Callable[[str, str, str], Optional[object]],
    min_lookback: int = MIN_LOOKBACK,
    fwd_ret_horizon: int = FWD_RET_HORIZON_DAYS,
) -> dict[str, Optional[dict]]:
    """
    Score trailing predictions against realized returns.

    fetch_prices_fn(symbol, start_date, end_date) must return a DataFrame
    with a 'close' column (or None on failure).

    Returns {model_key: {ic, directional_accuracy, lookback_n, mean_pred, std_pred} | None}.
    Returns {} if history file does not exist.
    """
    records = _load_history(history_path)
    if not records:
        return {}

    today = datetime.strptime(today_date, "%Y-%m-%d").date()
    cutoff = today - timedelta(days=fwd_ret_horizon + 1)

    scoreable = [
        r for r in records
        if r.get("price") is not None
        and "signal" in r and "confidence" in r and "date" in r
        and datetime.strptime(r["date"], "%Y-%m-%d").date() <= cutoff
    ]

    # Group by model; take the most recent min_lookback entries
    by_model: dict[str, list[dict]] = {}
    for r in scoreable:
        by_model.setdefault(r["model"], []).append(r)

    results: dict[str, Optional[dict]] = {}
    for model, model_records in by_model.items():
        sorted_records = sorted(model_records, key=lambda r: r["date"])[-min_lookback:]
        if len(sorted_records) < min_lookback:
            logger.info(
                "score_realized_ic: insufficient history for %s (%d < %d)",
                model, len(sorted_records), min_lookback,
            )
            results[model] = None
            continue

        scores, actuals = [], []
        for r in sorted_records:
            symbol = r["symbol"]
            pred_date = datetime.strptime(r["date"], "%Y-%m-%d")
            fetch_start = pred_date.strftime("%Y-%m-%d")
            fetch_end = (pred_date + timedelta(days=fwd_ret_horizon * 2 + 10)).strftime("%Y-%m-%d")
            try:
                price_df = fetch_prices_fn(symbol, fetch_start, fetch_end)
            except Exception:
                continue
            if price_df is None or len(price_df) < fwd_ret_horizon + 1:
                continue
            entry_price = float(r["price"])
            if entry_price <= 0:
                continue
            realized_price = float(price_df["close"].iloc[fwd_ret_horizon])
            realized_return = (realized_price / entry_price) - 1.0
            if abs(realized_return) < 1e-5:
                continue
            scores.append(_signal_to_score(r["signal"], float(r["confidence"])))
            actuals.append(realized_return)

        if len(scores) < min_lookback // 2:
            logger.info(
                "score_realized_ic: too few valid scored rows for %s (%d)", model, len(scores)
            )
            results[model] = None
            continue

        scores_arr = np.array(scores)
        actuals_arr = np.array(actuals)
        results[model] = {
            "ic": _spearman_ic(scores_arr, actuals_arr),
            "directional_accuracy": float(np.mean(np.sign(scores_arr) == np.sign(actuals_arr))),
            "lookback_n": len(scores),
            "mean_pred": float(scores_arr.mean()),
            "std_pred": float(scores_arr.std()),
        }

    return results
```

- [ ] **Step 4: Run tests to confirm they pass**

```
.venv/Scripts/python.exe -m pytest tests/test_ic_tracking.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```
git add signal_monitor.py tests/test_ic_tracking.py
git commit -m "feat: add signal_monitor.py with score_realized_ic"
```

---

## Task 6: `signal_monitor.py` — drift detector

**Files:**
- Modify: `signal_monitor.py`
- Create: `tests/test_drift_detection.py`

**Interfaces:**
- Produces:
  - `check_signal_drift(history_path: str, today_date: str, today_mean_scores: dict[str, float], window_days: int = 30, sigma_threshold: float = 2.0, abs_threshold: float = 0.002) -> dict[str, bool]`
    - `today_mean_scores`: `{model: mean(_signal_to_score(s, c) for all symbols today)}`
    - Returns `{model: True}` only when shift > sigma_threshold σ AND > abs_threshold AND yesterday also shifted

- [ ] **Step 1: Write failing tests**

Create `tests/test_drift_detection.py`:

```python
"""test_drift_detection.py — Tests for check_signal_drift in signal_monitor.py."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_monitor import check_signal_drift


def _write_history(tmp_path, records: list[dict]) -> str:
    path = tmp_path / "history.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


def _build_history(
    tmp_path,
    n_days: int = 35,
    base_score: float = 0.0,
    yesterday_score: float | None = None,
) -> tuple[str, str]:
    """Build history with stable daily mean = base_score for n_days.
    Optionally override yesterday's entry."""
    today = date.today()
    records = []
    for i in range(2, n_days + 2):  # from 2 days ago backwards
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        signal = "BUY" if base_score >= 0 else "SELL"
        records.append({
            "date": d, "symbol": "AAPL", "model": "daily_predictor",
            "signal": signal, "confidence": abs(base_score), "price": 100.0,
        })

    if yesterday_score is not None:
        yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        signal = "BUY" if yesterday_score >= 0 else "SELL"
        records.append({
            "date": yesterday, "symbol": "AAPL", "model": "daily_predictor",
            "signal": signal, "confidence": abs(yesterday_score), "price": 100.0,
        })

    path = _write_history(tmp_path, records)
    return path, today.strftime("%Y-%m-%d")


def test_no_drift_when_today_is_normal(tmp_path):
    path, today = _build_history(tmp_path, base_score=0.3)
    # Today's mean score is the same as historical mean — no drift
    result = check_signal_drift(path, today, {"daily_predictor": 0.3})
    assert result.get("daily_predictor") is False


def test_no_drift_warning_on_single_day_shift(tmp_path):
    """Even if today's shift exceeds threshold, single-day guard prevents warning."""
    path, today = _build_history(tmp_path, base_score=0.1)
    # Today is very bullish but yesterday was normal (no history of yesterday means no warning)
    result = check_signal_drift(path, today, {"daily_predictor": 0.9})
    assert result.get("daily_predictor") is False


def test_drift_warning_on_two_consecutive_days(tmp_path):
    """Two consecutive days of large shift triggers a warning."""
    # Historical mean ~0.1; yesterday was 0.9 (large shift); today also 0.9
    path, today = _build_history(tmp_path, base_score=0.1, yesterday_score=0.9)
    result = check_signal_drift(path, today, {"daily_predictor": 0.9})
    assert result.get("daily_predictor") is True


def test_no_drift_when_abs_shift_below_threshold(tmp_path):
    """Even if z-score is high, abs shift must exceed DRIFT_ABS_THRESHOLD (0.002)."""
    # Base score near zero; tiny absolute shift even if sigma-ratio is large
    path, today = _build_history(tmp_path, base_score=0.0001, yesterday_score=0.003)
    # today shift of 0.003 - 0.0001 = 0.0029 > 0.002, so this WILL warn
    # Let's use an even smaller shift: 0.001
    result = check_signal_drift(path, today, {"daily_predictor": 0.001})
    assert result.get("daily_predictor") is False


def test_no_drift_when_std_is_zero(tmp_path):
    """All-constant historical scores → std=0 → skip drift check, no exception."""
    # All historical entries are identical BUY 0.5
    path, today = _build_history(tmp_path, base_score=0.5, yesterday_score=0.5)
    result = check_signal_drift(path, today, {"daily_predictor": 0.9})
    # std=0 → can't compute z-score → no warning
    assert result.get("daily_predictor") is False


def test_drift_returns_false_when_insufficient_window(tmp_path):
    """Fewer than MIN_DRIFT_WINDOW days of history → no warning."""
    today = date.today()
    records = []
    for i in range(2, 5):  # only 3 historical days
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        records.append({"date": d, "symbol": "AAPL", "model": "daily_predictor",
                        "signal": "BUY", "confidence": 0.1, "price": 100.0})
    path = _write_history(tmp_path, records)
    result = check_signal_drift(path, today.strftime("%Y-%m-%d"), {"daily_predictor": 0.9})
    assert result.get("daily_predictor") is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```
.venv/Scripts/python.exe -m pytest tests/test_drift_detection.py -v
```

Expected: `ImportError: cannot import name 'check_signal_drift'`

- [ ] **Step 3: Add `check_signal_drift` to `signal_monitor.py`**

Append to `signal_monitor.py`:

```python
def check_signal_drift(
    history_path: str,
    today_date: str,
    today_mean_scores: dict[str, float],
    window_days: int = 30,
    sigma_threshold: float = DRIFT_SIGMA_THRESHOLD,
    abs_threshold: float = DRIFT_ABS_THRESHOLD,
) -> dict[str, bool]:
    """
    Check if today's predicted-return distribution has drifted vs trailing window.

    today_mean_scores: {model: mean score across all symbols today}
    Returns {model: True} only when shift > sigma_threshold σ AND > abs_threshold
    AND yesterday also showed the same shift (two-consecutive-day guard).
    """
    records = _load_history(history_path)
    today = datetime.strptime(today_date, "%Y-%m-%d").date()
    yesterday = today - timedelta(days=1)
    window_start = today - timedelta(days=window_days)

    # Build daily mean scores from history: {date -> {model -> mean_score}}
    raw: dict = {}
    for r in records:
        try:
            d = datetime.strptime(r["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        if d < window_start or d >= today:
            continue
        model = r.get("model")
        if not model:
            continue
        score = _signal_to_score(r.get("signal", "HOLD"), float(r.get("confidence", 0.0)))
        raw.setdefault(d, {}).setdefault(model, []).append(score)

    daily_means: dict = {d: {m: float(np.mean(s)) for m, s in ms.items()}
                         for d, ms in raw.items()}

    warnings: dict[str, bool] = {}
    all_models = set(today_mean_scores.keys())

    for model in all_models:
        today_score = today_mean_scores.get(model)
        if today_score is None:
            warnings[model] = False
            continue

        # Baseline window: exclude today and yesterday for a clean reference
        baseline_scores = [
            daily_means[d][model]
            for d in daily_means
            if model in daily_means[d] and d < yesterday
        ]
        if len(baseline_scores) < MIN_DRIFT_WINDOW:
            warnings[model] = False
            continue

        baseline = np.array(baseline_scores)
        mu = float(baseline.mean())
        sigma = float(baseline.std())
        if sigma == 0.0:
            warnings[model] = False
            continue

        def _is_shifted(score: float) -> bool:
            z = (score - mu) / sigma
            return abs(z) > sigma_threshold and abs(score - mu) > abs_threshold

        today_shifted = _is_shifted(today_score)
        yesterday_score = daily_means.get(yesterday, {}).get(model)
        yesterday_shifted = yesterday_score is not None and _is_shifted(yesterday_score)

        warnings[model] = today_shifted and yesterday_shifted

    return warnings
```

- [ ] **Step 4: Run all drift tests**

```
.venv/Scripts/python.exe -m pytest tests/test_drift_detection.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```
git add signal_monitor.py tests/test_drift_detection.py
git commit -m "feat: add check_signal_drift to signal_monitor.py"
```

---

## Task 7: Wire everything into `predict_next_day_lite.py` + Discord IC summary

**Files:**
- Modify: `predict_next_day_lite.py`

**Interfaces:**
- Consumes: `signal_monitor.score_realized_ic`, `signal_monitor.check_signal_drift`, `DB.upsert_ic`, `DB.get_ic_history`
- `send_discord` gains optional `ic_results: dict | None = None` parameter — if provided, appends a one-line IC summary to the header content and adds a yellow warning embed for any drifting models

- [ ] **Step 1: Add IC scorer + drift detector call in `main()` in `predict_next_day_lite.py`**

In `main()`, after `models = load_models(db=db)` and before the per-symbol prediction loop, add:

```python
from signal_monitor import score_realized_ic, check_signal_drift

# --- Realized IC scoring ---
ic_results: dict = {}
prediction_date = datetime.utcnow().strftime("%Y-%m-%d")
if args.history:
    try:
        from data_loader import load_yfinance as _fetch

        def _fetch_prices(symbol: str, start: str, end: str):
            return _fetch(symbol, start=start, end=end, interval="1d")

        ic_results = score_realized_ic(
            args.history, prediction_date, fetch_prices_fn=_fetch_prices
        )
        for model, res in ic_results.items():
            if res is not None:
                logger.info(
                    "Trailing IC [%s]: ic=%.4f dir_acc=%.2f n=%d",
                    model, res["ic"], res["directional_accuracy"], res["lookback_n"],
                )
                if db is not None:
                    db.upsert_ic(
                        model, prediction_date,
                        res["lookback_n"], res["ic"], res["directional_accuracy"],
                        res["mean_pred"], res["std_pred"],
                    )
    except Exception as exc:
        logger.warning("IC scoring failed: %s", exc)
```

- [ ] **Step 2: Compute `today_mean_scores` after the prediction loop and run drift check**

After the prediction loop (after `predictions = [predict_symbol(...) for s in symbols]`), add:

```python
# --- Drift detection ---
drift_warnings: dict[str, bool] = {}
if args.history:
    try:
        today_mean_scores: dict[str, float] = {}
        for pred in predictions:
            if "error" in pred:
                continue
            for model_key, model_pred in pred.get("predictions", {}).items():
                if "signal" not in model_pred:
                    continue
                from signal_monitor import _signal_to_score
                score = _signal_to_score(model_pred["signal"], model_pred.get("confidence", 0.0))
                today_mean_scores.setdefault(model_key, [])
                today_mean_scores[model_key].append(score)
        today_mean_scores = {m: float(np.mean(s)) for m, s in today_mean_scores.items()}
        drift_warnings = check_signal_drift(args.history, prediction_date, today_mean_scores)
        for model, warned in drift_warnings.items():
            if warned:
                logger.warning("Signal drift detected for %s", model)
    except Exception as exc:
        logger.warning("Drift detection failed: %s", exc)
```

- [ ] **Step 3: Update `send_discord` to accept and render IC results and drift warnings**

Change the `send_discord` signature from:

```python
def send_discord(predictions: list, webhook_url: str) -> bool:
```

to:

```python
def send_discord(
    predictions: list,
    webhook_url: str,
    ic_results: dict | None = None,
    drift_warnings: dict | None = None,
) -> bool:
```

Replace the `header` line inside `send_discord`:

```python
# BEFORE:
header = (
    f"**Daily Trading Predictions** — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    "Results from daily ML models"
)
```

with:

```python
ic_lines = []
if ic_results:
    for model, res in sorted(ic_results.items()):
        if res is not None:
            ic_lines.append(
                f"`{model}` IC={res['ic']:+.3f} dir-acc={res['directional_accuracy']:.0%} "
                f"(n={res['lookback_n']})"
            )
ic_summary = ("  |  ".join(ic_lines)) if ic_lines else "IC: not yet available"
header = (
    f"**Daily Trading Predictions** — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    f"Trailing-20 {ic_summary}"
)
```

Add drift warning embeds before `if not embeds:`. Insert after the existing `embeds = []` block:

```python
if drift_warnings:
    for model, warned in sorted(drift_warnings.items()):
        if warned:
            embeds.insert(0, {
                "title": f"⚠️ Signal Drift Detected — {model}",
                "description": (
                    "Today's predicted-return distribution has shifted more than 2σ "
                    "from the trailing 30-day baseline for the second consecutive day. "
                    "Consider reviewing model freshness or market regime."
                ),
                "color": 0xFFFF00,
            })
```

- [ ] **Step 4: Update `send_discord` call in `main()` to pass new args**

Find the existing call:

```python
if send_discord(predictions, webhook):
```

Replace with:

```python
if send_discord(predictions, webhook, ic_results=ic_results, drift_warnings=drift_warnings):
```

- [ ] **Step 5: Run existing predict tests to ensure no regressions**

```
.venv/Scripts/python.exe -m pytest tests/test_predict.py -v
```

Expected: all existing tests PASS. (Tests mock the webhook and don't call Discord.)

- [ ] **Step 6: Run full test suite**

```
.venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: all tests PASS with no regressions.

- [ ] **Step 7: Commit**

```
git add predict_next_day_lite.py
git commit -m "feat: wire IC scorer and drift detector into daily prediction pipeline"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Walk-forward harness producing IC time series → Task 2
- [x] Parameter sweep via walk-forward IC → Task 3
- [x] Best params saved into pickle → Task 4 (`_save_and_register`)
- [x] Three-level param priority (env var → pickle → default) → Task 4 (`ml_strategies.py`, `predict_next_day_lite.py`)
- [x] `ic_history` DB table + `upsert_ic` → Task 1
- [x] Realized IC scorer (trailing 20 pooled (date, symbol) pairs) → Task 5
- [x] IC written to DB → Task 7 (`db.upsert_ic` call)
- [x] IC summary in Discord header → Task 7 (`send_discord`)
- [x] Drift detector (2σ shift, 2-consecutive-day guard, abs threshold) → Task 6
- [x] Drift warning embed → Task 7 (`send_discord`)
- [x] `test_walk_forward.py` → Tasks 2 + 3
- [x] `test_ic_tracking.py` → Task 5
- [x] `test_drift_detection.py` → Task 6
- [x] `test_predictor.py` backward-compat cases → Task 4

**Placeholder scan:** No TBD/TODO present. All steps show exact code.

**Type consistency:**
- `sweep_params` returns `tuple[float, int]` → consumed by `train_predictor._save_and_register(best_signal_quantile: float, best_threshold_window: int)` ✓
- `score_realized_ic` returns `dict[str, dict | None]` → `ic_results` in `main()` iterates `.items()` on it ✓
- `check_signal_drift` accepts `dict[str, float]` for `today_mean_scores` → built as `{m: float(np.mean(s))}` ✓
- `DB.upsert_ic` signature matches call in Task 7 ✓
- `send_discord(predictions, webhook_url, ic_results, drift_warnings)` matches call in `main()` ✓
