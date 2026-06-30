# Deploy Modularized Prediction/Strategy Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy `daily_predictor` (the Ridge regression forecaster + `DailyPredictorStrategy` rolling-quantile decision layer) into the live `predict_next_day_lite.py` / Discord production pipeline, and refactor that pipeline so adding or removing a model is a config/registry change, not a copy-pasted code block.

**Architecture:** `predict_next_day_lite.py` currently hardcodes three near-identical prediction blocks (one each for `daily_logistic`, `daily_xgboost`, `daily_dqn`) inside `predict_symbol()`, plus a hardcoded two-tuple list in `load_models()`. This plan extracts the duplicated classifier logic into one shared function, adds a new regressor-handling function for `daily_predictor` that reuses the exact same rolling-quantile decision logic already proven in backtesting (`ml_strategies.DailyPredictorStrategy`), and makes the active model list config-driven (`prediction.models` in `config/default.yaml`). DQN keeps its existing presence-gated `.pt` loading path unchanged — it's a structurally different artifact (no `feature_contract` pickle, different state shape) and forcing it into the same config list would not reduce risk, only add a misleading abstraction.

**Tech Stack:** Python, scikit-learn (Ridge — already loaded for `daily_predictor.pkl`), pandas, existing `db.DB` SQLite model registry (no schema changes needed — `daily_predictions` and `model_registry` tables are already model-key-generic), pytest.

## Global Constraints

- Use the project venv interpreter for every test/run command: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe` (the bare `python` on PATH resolves to the Windows Store stub and lacks `xgboost`/`torch`/`sklearn`). Run commands from the worktree root.
- Do not change behavior of the already-deployed `daily_logistic`/`daily_xgboost` classifier prediction math — the refactor in Task 4 must be a behavior-preserving extraction (verified by keeping all existing `tests/test_predict.py` assertions passing unmodified).
- `daily_predictor`'s live inference must apply the exact same feature preprocessing the model was trained on (`train_models._preprocess`, ±5 std clip + inf/nan handling) and the exact same decision logic as backtesting (`ml_strategies.DailyPredictorStrategy`'s rolling-quantile threshold) — these must come from one shared function, not be reimplemented, so backtest and live predictions can never silently diverge.
- No new third-party dependencies — `scikit-learn` (for `Ridge`) is already in `requirements-predict.txt`; nothing else is needed for `daily_predictor` inference.
- `models/daily_predictor.pkl` already exists (Ridge, test_ic=0.0600, test_dir_acc=0.5347) and is committed — this plan deploys it, it does not retrain it.
- Per `models/README.md`, `daily_predictor`'s backtest edge is a single-window, untuned, unvalidated lead, not a proven edge — Discord output must not overstate it (no special "winner" styling, just another model row like the existing two).
- Follow CLAUDE.md: only commit when explicitly asked. This plan's steps include `git commit` commands per the writing-plans template — when executing, hold off on actually running them unless the user has authorized commits for this session, or ask first.

---

### Task 1: Extract shared decision-layer function in `ml_strategies.py`

**Files:**
- Modify: `ml_strategies.py:249-328` (the `DailyPredictorStrategy` class — add a module-level function before it, refactor `signal()` to call it)
- Test: `tests/test_predictor.py`

**Interfaces:**
- Produces: `compute_predictor_signal(pred_ret: np.ndarray, signal_quantile: float, threshold_window: int) -> np.ndarray` — pure function, no pandas Series/index required on input or output, returns an `int` array of `{-1, 0, 1}` the same length as `pred_ret`. This becomes the single source of truth for the rolling-quantile decision logic, called by both `DailyPredictorStrategy.signal()` (Task 1) and the live-prediction path in `predict_next_day_lite.py` (Task 3).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_predictor.py`, after the existing imports (after line 26, before `_make_price_df`):

```python
from ml_strategies import compute_predictor_signal
```

Add these new test functions at the end of `tests/test_predictor.py` (after `test_predictor_strategy_raises_without_artifact`):

```python
def test_compute_predictor_signal_buy_on_extreme_positive():
    """A predicted-return spike well above the trailing window's quantile
    must trigger BUY on the day it occurs."""
    pred_ret = np.full(80, 0.001)
    pred_ret[-1] = 0.05
    signals = compute_predictor_signal(pred_ret, signal_quantile=0.7, threshold_window=60)
    assert signals[-1] == 1


def test_compute_predictor_signal_sell_on_extreme_negative():
    pred_ret = np.full(80, 0.001)
    pred_ret[-1] = -0.05
    signals = compute_predictor_signal(pred_ret, signal_quantile=0.7, threshold_window=60)
    assert signals[-1] == -1


def test_compute_predictor_signal_hold_when_unremarkable():
    """A prediction with the same magnitude as the entire trailing window
    is exactly at the boundary, not above it — must not trigger a trade."""
    pred_ret = np.full(80, 0.001)
    signals = compute_predictor_signal(pred_ret, signal_quantile=0.7, threshold_window=60)
    assert signals[-1] == 0


def test_compute_predictor_signal_early_bars_are_hold():
    """Before min_periods=20 trailing predictions exist, the rolling
    threshold is undefined (NaN) — must default to HOLD, never trade on
    an undefined threshold."""
    pred_ret = np.array([0.05, -0.05, 0.03])
    signals = compute_predictor_signal(pred_ret, signal_quantile=0.7, threshold_window=60)
    assert (signals == 0).all()


def test_compute_predictor_signal_output_length_matches_input():
    pred_ret = np.full(100, 0.002)
    signals = compute_predictor_signal(pred_ret, signal_quantile=0.7, threshold_window=60)
    assert len(signals) == 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/test_predictor.py -v -k compute_predictor_signal`
Expected: FAIL with `ImportError: cannot import name 'compute_predictor_signal' from 'ml_strategies'`

- [ ] **Step 3: Add the shared function and refactor `DailyPredictorStrategy.signal()` to use it**

In `ml_strategies.py`, insert this function immediately before `class DailyPredictorStrategy(BaseStrategy):` (i.e. right before current line 249):

```python
def compute_predictor_signal(
    pred_ret: np.ndarray, signal_quantile: float, threshold_window: int
) -> np.ndarray:
    """
    Causal rolling-quantile decision layer for a continuous return forecast.

    Single source of truth for the daily_predictor decision logic — called
    by both DailyPredictorStrategy.signal() (backtest) and
    predict_next_day_lite.py's live-prediction path, so the two can never
    silently diverge. A fixed vol-scaled band does not work here because a
    regularized regressor's predictions are shrunk toward zero on a
    different scale than raw returns (empirically ~6x smaller) — this
    adapts to whatever scale a given prediction model produces by trading
    only the top `1 - signal_quantile` fraction of the trailing
    `threshold_window` bars' |prediction| magnitudes. The threshold is
    shifted by one bar so it never sees today's own prediction.

    Returns an int array of {-1, 0, 1} (SELL/HOLD/BUY) the same length as
    pred_ret. Bars before `threshold_window`'s min_periods is satisfied
    default to HOLD (0), since the threshold is undefined (NaN) and any
    NaN comparison is False.
    """
    pred_series = pd.Series(pred_ret)
    abs_pred = pred_series.abs()
    rolling_thr = (
        abs_pred.rolling(threshold_window, min_periods=20)
        .quantile(signal_quantile)
        .shift(1)
    )
    trigger = (abs_pred > rolling_thr).values
    actions = np.ones(len(pred_series), dtype=int)  # default: HOLD
    actions[trigger & (pred_ret > 0)] = 2  # BUY
    actions[trigger & (pred_ret < 0)] = 0  # SELL
    return actions - 1  # {0,1,2} -> {-1,0,1}
```

Then replace the body of `DailyPredictorStrategy.signal()` (current lines 297-328) with:

```python
    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        daily_feats = make_daily_features(df)
        X = _preprocess(daily_feats[FEATURE_COLS].values.astype(np.float32))

        if self.model is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            pred_ret = self.model.predict(X_scaled)
        else:
            # In-session training fallback (no pre-trained model)
            from sklearn.linear_model import Ridge
            y_raw = daily_feats["fwd_ret_1d"].values
            mask = ~np.isnan(y_raw)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = Ridge(alpha=10.0)
            model.fit(X_scaled[mask], y_raw[mask])
            pred_ret = model.predict(X_scaled)

        signals = compute_predictor_signal(
            pred_ret, self.signal_quantile, self.threshold_window
        )
        return self._apply_holding_period(pd.Series(signals, index=daily_feats.index))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/test_predictor.py -v`
Expected: All tests in the file PASS, including the 5 new ones and the pre-existing `test_predictor_strategy_inline_training_produces_valid_signals` / `test_predictor_artifact_contract` (confirms the refactor didn't change `DailyPredictorStrategy`'s observable behavior).

- [ ] **Step 5: Commit**

```bash
git add ml_strategies.py tests/test_predictor.py
git commit -m "feat: extract compute_predictor_signal as shared decision-layer function"
```

---

### Task 2: Make the live prediction model list config-driven

**Files:**
- Modify: `config.py:126-128` (`PredictionCfg`), `config.py:171-176` (`load_config`'s prediction construction)
- Modify: `config/default.yaml:16-49` (`prediction:` section)
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `PredictionCfg.models: list[str]` (default `["daily_logistic", "daily_xgboost", "daily_predictor"]`), readable via `get_config().prediction.models`. Consumed by `load_models()` in Task 5.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
"""test_config.py — Tests for config.py's YAML loading, focused on the
prediction.models field that drives which models predict_next_day_lite.py
attempts to load (see Task 5 in docs/superpowers/plans/2026-06-30-predictor-production-deploy.md)."""
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PredictionCfg, load_config


def test_prediction_cfg_default_models():
    """With no config file, PredictionCfg.models defaults to the three
    currently-deployed daily models."""
    cfg = PredictionCfg()
    assert cfg.models == ["daily_logistic", "daily_xgboost", "daily_predictor"]


def test_load_config_reads_prediction_models_from_yaml():
    yaml_content = dedent("""
        prediction:
          models:
            - daily_logistic
            - daily_predictor
          symbols:
            - AAPL
    """)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        path = f.name

    cfg = load_config(path)
    assert cfg.prediction.models == ["daily_logistic", "daily_predictor"]
    assert cfg.prediction.symbols == ["AAPL"]


def test_load_config_missing_prediction_models_uses_default():
    """A config file that sets prediction.symbols but omits models must
    still fall back to the default model list, not an empty list."""
    yaml_content = dedent("""
        prediction:
          symbols:
            - AAPL
    """)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        path = f.name

    cfg = load_config(path)
    assert cfg.prediction.models == ["daily_logistic", "daily_xgboost", "daily_predictor"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: FAIL with `AttributeError: 'PredictionCfg' object has no attribute 'models'`

- [ ] **Step 3: Add the `models` field to `PredictionCfg` and wire it through `load_config`**

In `config.py`, replace the `PredictionCfg` dataclass (current lines 126-128):

```python
@dataclass
class PredictionCfg:
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "SPY", "MSFT", "GOOGL", "NVDA"])
    models: list[str] = field(
        default_factory=lambda: ["daily_logistic", "daily_xgboost", "daily_predictor"]
    )
```

In `config.py`'s `load_config()`, replace the `prediction=PredictionCfg(...)` block (current lines 171-176):

```python
        prediction=PredictionCfg(
            symbols=prediction_raw.get(
                "symbols",
                PredictionCfg.__dataclass_fields__["symbols"].default_factory(),
            ),
            models=prediction_raw.get(
                "models",
                PredictionCfg.__dataclass_fields__["models"].default_factory(),
            ),
        ),
```

In `config/default.yaml`, replace the `prediction:` section header (current lines 16-19, immediately before the `symbols:` list) with:

```yaml
# Symbol universe for the daily prediction job and Discord notifications.
# Edit `symbols` to change what gets sent to Discord — no workflow YAML edit needed.
# Edit `models` to add/remove a model from the live pipeline — no code change needed
# (predict_next_day_lite.py loads exactly this list; a model missing its pickle is
# logged and skipped, not a hard failure). See models/README.md for what each does.
prediction:
  models:
    - daily_logistic
    - daily_xgboost
    - daily_predictor
  symbols:
```

(The existing symbol list lines that currently follow `symbols:` at lines 20-49 stay exactly as they are — only the header comment block and the new `models:` key are added above them.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: All 3 tests PASS.

Then run the full suite to confirm nothing else reads `config/default.yaml`'s `prediction:` section in a way that breaks on the new key:
Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: All tests PASS (same count as before plus the 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add config.py config/default.yaml tests/test_config.py
git commit -m "feat: make the live prediction model list config-driven (prediction.models)"
```

---

### Task 3: Add classifier/regressor prediction helpers to `predict_next_day_lite.py`

**Files:**
- Modify: `predict_next_day_lite.py:25-29` (imports), add new module-level functions after `_load_pkl` (current lines 44-66, before `load_models`)
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `ml_strategies.compute_predictor_signal(pred_ret, signal_quantile, threshold_window)` from Task 1; `train_models._preprocess(X)` (existing, unchanged).
- Produces: `_predict_classifier_signal(data: dict, X_latest: np.ndarray) -> dict` (keys `signal: str`, `confidence: float`) — extraction of the existing daily_logistic/daily_xgboost logic, behavior-identical. `_predict_regressor_signal(data: dict, X_all: np.ndarray, signal_quantile: float = 0.7, threshold_window: int = 60) -> dict` (same return shape) — new, for `daily_predictor`. `_regressor_confidence(pred_ret: np.ndarray, threshold_window: int) -> float`. These are written and unit-tested in isolation in this task; Task 4 wires them into `predict_symbol`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_predict.py`, after the existing imports (after line 22):

```python
from predict_next_day_lite import (
    _predict_classifier_signal,
    _predict_regressor_signal,
    _regressor_confidence,
)
```

Add these new test classes at the end of `tests/test_predict.py` (after `TestPredictProbaClassMapping`):

```python
# ---------------------------------------------------------------------------
# _predict_classifier_signal / _predict_regressor_signal (Task 3)
# ---------------------------------------------------------------------------

class TestPredictClassifierSignal:
    def test_matches_existing_predict_symbol_behavior(self):
        """The extracted helper must reproduce exactly what the old inline
        daily_logistic/daily_xgboost blocks computed."""
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        clf.classes_ = np.array([0, 1, 2])
        data = {"model": clf, "scaler": scaler, "confidence_threshold": 0.55}

        result = _predict_classifier_signal(data, np.zeros((1, len(FEATURE_COLS))))

        assert result["signal"] == "BUY"
        assert abs(result["confidence"] - 0.7) < 1e-6

    def test_below_threshold_collapses_to_hold(self):
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.25, 0.30, 0.45]])
        clf.classes_ = np.array([0, 1, 2])
        data = {"model": clf, "scaler": scaler, "confidence_threshold": 0.55}

        result = _predict_classifier_signal(data, np.zeros((1, len(FEATURE_COLS))))

        assert result["signal"] == "HOLD"


class TestRegressorConfidence:
    def test_today_is_max_in_window_gives_confidence_one(self):
        pred_ret = np.array([0.001] * 59 + [0.05])
        conf = _regressor_confidence(pred_ret, threshold_window=60)
        assert conf == pytest.approx(1.0)

    def test_today_is_typical_gives_mid_confidence(self):
        pred_ret = np.array([0.001] * 60)
        conf = _regressor_confidence(pred_ret, threshold_window=60)
        assert conf == pytest.approx(1.0)  # all equal -> today <= every value

    def test_too_short_window_returns_zero(self):
        assert _regressor_confidence(np.array([0.01]), threshold_window=60) == 0.0


class TestPredictRegressorSignal:
    def test_extreme_prediction_triggers_buy_with_clean_features(self):
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        model = MagicMock()
        pred_ret = np.full(80, 0.001)
        pred_ret[-1] = 0.05
        model.predict.return_value = pred_ret
        data = {"model": model, "scaler": scaler}

        X_all = np.zeros((80, len(FEATURE_COLS)), dtype=np.float32)
        result = _predict_regressor_signal(data, X_all)

        assert result["signal"] == "BUY"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_unremarkable_prediction_holds(self):
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        model = MagicMock()
        model.predict.return_value = np.full(80, 0.001)
        data = {"model": model, "scaler": scaler}

        X_all = np.zeros((80, len(FEATURE_COLS)), dtype=np.float32)
        result = _predict_regressor_signal(data, X_all)

        assert result["signal"] == "HOLD"

    def test_inf_and_nan_features_do_not_crash(self):
        """X_all may contain inf/nan from upstream feature computation on
        thin data — _preprocess must clip these before scaling, matching
        what the model was trained on (train_models._preprocess)."""
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        model = MagicMock()
        model.predict.return_value = np.full(80, 0.001)
        data = {"model": model, "scaler": scaler}

        X_all = np.zeros((80, len(FEATURE_COLS)), dtype=np.float32)
        X_all[0, 0] = np.inf
        X_all[1, 1] = np.nan
        result = _predict_regressor_signal(data, X_all)

        assert result["signal"] in {"BUY", "SELL", "HOLD"}
        called_with = scaler.transform.call_args[0][0]
        assert np.isfinite(called_with).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/test_predict.py -v -k "ClassifierSignal or RegressorConfidence or RegressorSignal"`
Expected: FAIL with `ImportError: cannot import name '_predict_classifier_signal' from 'predict_next_day_lite'`

- [ ] **Step 3: Implement the helper functions**

In `predict_next_day_lite.py`, update the import block (current lines 25-29) to:

```python
from config import get_config
from data_loader import load_yfinance
from daily_features import FEATURE_COLS, make_daily_features
from db import DB
from dqn_signal import gate_dqn_signal
from ml_strategies import compute_predictor_signal
from train_models import _preprocess
```

Then insert these three functions after `_load_pkl` (current lines 44-66) and before `def load_models(...)`:

```python
def _predict_classifier_signal(data: dict, X_latest: np.ndarray) -> dict:
    """
    Predict today's signal for a 3-class classifier model (daily_logistic,
    daily_xgboost). Both share identical prediction logic — only the
    trained model object differs — so this one function serves both,
    instead of two copy-pasted blocks.
    """
    X_scaled = data["scaler"].transform(X_latest)
    prob = data["model"].predict_proba(X_scaled)[0]
    pred_idx = int(np.argmax(prob))
    confidence = float(prob[pred_idx])
    threshold = data.get("confidence_threshold", 0.55)
    pred_class = int(data["model"].classes_[pred_idx])
    signal = _SIGNAL_NAMES[pred_class]
    if signal != "HOLD" and confidence < threshold:
        signal = "HOLD"
    return {"signal": signal, "confidence": confidence}


def _regressor_confidence(pred_ret: np.ndarray, threshold_window: int) -> float:
    """
    Percentile rank of today's |predicted return| within the trailing
    threshold_window predictions, in [0, 1].

    This is NOT a calibrated probability like the classifiers' confidence
    (Ridge has no predict_proba) — it is the fraction of the trailing
    window whose magnitude today's prediction equals or exceeds. Higher
    means today's forecast is more extreme relative to its recent history,
    which is also what compute_predictor_signal's rolling quantile uses to
    decide whether to trade.
    """
    window = pred_ret[-threshold_window:]
    if len(window) < 2:
        return 0.0
    today_abs = abs(pred_ret[-1])
    return float(np.mean(np.abs(window) <= today_abs))


def _predict_regressor_signal(
    data: dict,
    X_all: np.ndarray,
    signal_quantile: float = 0.7,
    threshold_window: int = 60,
) -> dict:
    """
    Predict today's signal for a regression-style model (daily_predictor).

    Unlike the classifiers, this needs the *trailing window* of
    predictions (X_all), not just the latest bar, because
    compute_predictor_signal's causal rolling quantile needs history to
    decide whether today's forecast is extreme enough to trade — the same
    decision logic used in backtesting (ml_strategies.DailyPredictorStrategy),
    so live and backtest predictions can't silently diverge.

    Applies the same ±5-std-clip preprocessing (train_models._preprocess)
    daily_predictor was trained on (see train_predictor.prepare_data) —
    skipping this would feed the model out-of-distribution inputs it never
    saw in training.
    """
    X_clean = _preprocess(X_all.copy())
    X_scaled = data["scaler"].transform(X_clean)
    pred_ret = data["model"].predict(X_scaled)

    sq = float(os.environ.get("PREDICTOR_SIGNAL_QUANTILE", signal_quantile))
    tw = int(os.environ.get("PREDICTOR_THRESHOLD_WINDOW", threshold_window))
    signals = compute_predictor_signal(pred_ret, sq, tw)

    last_signal = int(signals[-1])
    signal_name = _SIGNAL_NAMES[last_signal + 1]
    confidence = _regressor_confidence(pred_ret, tw)
    return {"signal": signal_name, "confidence": confidence}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/test_predict.py -v -k "ClassifierSignal or RegressorConfidence or RegressorSignal"`
Expected: All new tests PASS.

Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/test_predict.py -v`
Expected: All pre-existing tests in the file still PASS (these new functions are not yet called from `predict_symbol`, so nothing else should change).

- [ ] **Step 5: Commit**

```bash
git add predict_next_day_lite.py tests/test_predict.py
git commit -m "feat: add classifier/regressor prediction helpers to predict_next_day_lite.py"
```

---

### Task 4: Wire the helpers into `predict_symbol` via a model-kind dispatch

**Files:**
- Modify: `predict_next_day_lite.py:114-264` (`predict_symbol`)
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `_predict_classifier_signal`, `_predict_regressor_signal` from Task 3.
- Produces: `MODEL_KINDS: dict[str, str]` module-level constant mapping `model_key -> "classifier" | "regressor"`. Adding a future classifier or regressor model means adding one entry here plus one entry to `prediction.models` in config (Task 2) — no changes to `predict_symbol` itself.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_predict.py`, inside a new test class after `TestPredictRegressorSignal`:

```python
# ---------------------------------------------------------------------------
# predict_symbol with daily_predictor (end-to-end through the dispatch loop)
# ---------------------------------------------------------------------------

class TestPredictSymbolPredictor:
    def _build_predictor_models(self, pred_ret: np.ndarray) -> dict:
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        model = MagicMock()
        model.predict.return_value = pred_ret
        return {
            "daily_predictor": {
                "model": model,
                "scaler": scaler,
                "feature_contract": list(FEATURE_COLS),
            }
        }

    def test_extreme_prediction_produces_buy(self):
        n = 100
        pred_ret = np.full(n, 0.001)
        pred_ret[-1] = 0.05
        models = self._build_predictor_models(pred_ret)
        feats_df = _make_features_df(n)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(n)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        assert "error" not in result
        p = result["predictions"]["daily_predictor"]
        assert p["signal"] == "BUY"

    def test_unremarkable_prediction_produces_hold(self):
        n = 100
        pred_ret = np.full(n, 0.001)
        models = self._build_predictor_models(pred_ret)
        feats_df = _make_features_df(n)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(n)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        p = result["predictions"]["daily_predictor"]
        assert p["signal"] == "HOLD"

    def test_runs_alongside_classifier_without_interference(self):
        """Both a classifier and the regressor in the same models dict must
        each produce their own independent prediction entry."""
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        clf.classes_ = np.array([0, 1, 2])

        n = 100
        pred_ret = np.full(n, 0.001)
        pred_ret[-1] = 0.05
        models = {
            "daily_logistic": {
                "model": clf, "scaler": scaler,
                "feature_contract": list(FEATURE_COLS), "confidence_threshold": 0.55,
            },
            **self._build_predictor_models(pred_ret),
        }
        feats_df = _make_features_df(n)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(n)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        assert result["predictions"]["daily_logistic"]["signal"] == "BUY"
        assert result["predictions"]["daily_predictor"]["signal"] == "BUY"

    def test_unknown_model_kind_reports_error_not_crash(self):
        """A model_key with no MODEL_KINDS entry must surface as a
        per-model error in the result dict, not crash the whole symbol's
        prediction (one bad/misconfigured model must not take down the
        others) — this is the 'easy to add without breaking things'
        contract: forgetting to register a new model's kind fails loudly
        and locally, not silently or globally."""
        models = {
            "totally_new_model": {
                "model": MagicMock(), "scaler": MagicMock(),
                "feature_contract": list(FEATURE_COLS),
            }
        }
        feats_df = _make_features_df(100)

        with patch("predict_next_day_lite.load_yfinance", return_value=_make_ohlcv(100)), \
             patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
            result = predict_symbol("AAPL", models)

        assert "error" not in result  # the symbol overall still succeeds
        assert "error" in result["predictions"]["totally_new_model"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/test_predict.py -v -k TestPredictSymbolPredictor`
Expected: FAIL — `result["predictions"]` has no `"daily_predictor"` key (the dispatch loop doesn't know about it yet).

- [ ] **Step 3: Replace the three hardcoded blocks in `predict_symbol` with the dispatch loop**

In `predict_next_day_lite.py`, add this module-level constant right after the `_SIGNAL_NAMES` line (current line 37):

```python
_SIGNAL_NAMES = ["SELL", "HOLD", "BUY"]  # index matches label {0,1,2}

# Maps a loaded model_key to how predict_symbol should run it. Adding a new
# classifier or regressor model means adding one entry here and one entry
# to prediction.models in config/default.yaml — predict_symbol, the DB
# upsert, and the Discord payload all pick it up automatically with no
# further code changes. DQN is handled separately below (different
# artifact format and a windowed state, not a single-row prediction).
MODEL_KINDS = {
    "daily_logistic": "classifier",
    "daily_xgboost": "classifier",
    "daily_predictor": "regressor",
}
```

Then replace the entire body from the `# --- DailyLogistic ---` comment through the end of the `# --- DailyXGBoost ---` block (current lines 158-211) with:

```python
    for model_key, kind in MODEL_KINDS.items():
        if model_key not in models:
            continue
        try:
            data = models[model_key]
            if kind == "classifier":
                pred = _predict_classifier_signal(data, X_latest)
            elif kind == "regressor":
                pred = _predict_regressor_signal(data, X_all)
            else:
                raise RuntimeError(
                    f"Unknown model kind '{kind}' for '{model_key}' — "
                    "register it in MODEL_KINDS."
                )
            result["predictions"][model_key] = pred
            if db is not None:
                meta = db.get_active_model(model_key)
                version = meta["version"] if meta else 0
                db.upsert_prediction(
                    symbol, model_key, version, prediction_date,
                    pred["signal"], pred["confidence"], result["price"],
                )
        except Exception as exc:
            result["predictions"][model_key] = {"error": str(exc)}
```

The `# --- DailyDQN ---` block (current lines 212-262) stays exactly as-is, immediately after this loop.

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/test_predict.py -v`
Expected: All tests PASS — both the new `TestPredictSymbolPredictor` class and every pre-existing test in the file (`TestPredictSymbol`, `TestPredictProbaClassMapping`, `TestPredictSymbolDQN`, etc.), confirming the refactor preserved existing classifier/DQN behavior exactly.

- [ ] **Step 5: Commit**

```bash
git add predict_next_day_lite.py tests/test_predict.py
git commit -m "feat: dispatch predict_symbol's classifier/regressor models via MODEL_KINDS"
```

---

### Task 5: Make `load_models` config-driven

**Files:**
- Modify: `predict_next_day_lite.py:69-107` (`load_models`)
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `get_config().prediction.models` from Task 2.
- Produces: `load_models(db=None, model_keys=None)` — `model_keys` defaults to `get_config().prediction.models` when not passed; existing callers (`main()`) are unaffected since they already call `load_models(db=db)` with no `model_keys` arg.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_predict.py`, after the imports add `load_models` to the existing `from predict_next_day_lite import (...)` block:

```python
from predict_next_day_lite import (
    _load_pkl,
    _predict_classifier_signal,
    _predict_regressor_signal,
    _regressor_confidence,
    append_predictions_history,
    load_models,
    predict_symbol,
    send_discord,
)
```

Add this new test class at the end of `tests/test_predict.py`:

```python
# ---------------------------------------------------------------------------
# load_models (Task 5 — config-driven model list)
# ---------------------------------------------------------------------------

class TestLoadModels:
    def test_respects_explicit_model_keys_list(self, tmp_path, monkeypatch):
        """Passing model_keys explicitly must load exactly that list,
        independent of config — proves a model can be added/removed by
        changing the list with no other code path involved."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        artifact = _make_artifact()
        with open(tmp_path / "models" / "daily_logistic.pkl", "wb") as f:
            pickle.dump(artifact, f)

        loaded = load_models(db=None, model_keys=["daily_logistic"])

        assert set(loaded.keys()) <= {"daily_logistic", "daily_dqn"}
        assert "daily_logistic" in loaded

    def test_missing_pickle_for_configured_model_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        """A model_key in the config list whose pickle doesn't exist on
        disk must be silently skipped (logged), not raise — so a
        half-deployed model doesn't take down the whole prediction run."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        loaded = load_models(db=None, model_keys=["daily_predictor", "totally_unknown"])

        assert loaded == {}

    def test_defaults_to_config_prediction_models(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        artifact = _make_artifact()
        with open(tmp_path / "models" / "daily_predictor.pkl", "wb") as f:
            pickle.dump(artifact, f)

        with patch("predict_next_day_lite.get_config") as mock_cfg:
            mock_cfg.return_value.prediction.models = ["daily_predictor"]
            loaded = load_models(db=None)

        assert "daily_predictor" in loaded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/test_predict.py -v -k TestLoadModels`
Expected: FAIL — `TypeError: load_models() got an unexpected keyword argument 'model_keys'`

- [ ] **Step 3: Implement**

In `predict_next_day_lite.py`, replace `load_models` (current lines 69-95, the function body up through the classifier-loading `for` loop — the DQN block below it is unchanged) with:

```python
def load_models(
    db: Optional[DB] = None, model_keys: Optional[list] = None
) -> dict:
    """
    Load all available daily model pickles.

    model_keys defaults to config.prediction.models (config-driven, so a
    model can be added to or removed from the live pipeline by editing
    config/default.yaml — no code change). Uses DB model_registry to
    resolve artifact_path when available, falling back to the canonical
    models/<model_key>.pkl path otherwise. A configured model whose
    artifact is missing or invalid is logged and skipped, not fatal —
    other configured models still load and predict.
    """
    models = {}

    def _resolve_path(model_key: str, canonical: str) -> str:
        if db is not None:
            meta = db.get_active_model(model_key)
            if meta and os.path.exists(meta["artifact_path"]):
                return meta["artifact_path"]
        return canonical

    if model_keys is None:
        model_keys = get_config().prediction.models

    for model_key in model_keys:
        canonical = f"models/{model_key}.pkl"
        path = _resolve_path(model_key, canonical)
        try:
            models[model_key] = _load_pkl(path, model_key)
            logger.info("Loaded %s from %s", model_key, path)
        except RuntimeError as exc:
            logger.warning("%s", exc)
```

(The blank line and the `# DQN agent (separate format)` comment plus its `if os.path.exists("models/dqn_agent.pt"):` block and the final `return models` — current lines 96-107 — stay exactly as they are, unchanged, directly after this.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/test_predict.py -v`
Expected: All tests PASS, including the new `TestLoadModels` class.

Run full suite: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add predict_next_day_lite.py tests/test_predict.py
git commit -m "feat: make load_models read its model list from prediction.models config"
```

---

### Task 6: Discord display name + documentation updates

**Files:**
- Modify: `predict_next_day_lite.py:325-329` (`send_discord`'s `strategy_display` dict)
- Modify: `CLAUDE.md` (pipeline description, key files table)
- Modify: `models/README.md` ("Prediction vs. strategy" section)

**Interfaces:** None new — this task is documentation/display-only, no behavior change to test.

- [ ] **Step 1: Add the Discord display name**

In `predict_next_day_lite.py`, update the `strategy_display` dict inside `send_discord` (current lines 325-329):

```python
    strategy_display = {
        "daily_logistic": "Daily Logistic",
        "daily_xgboost": "Daily XGBoost",
        "daily_predictor": "Daily Predictor",
        "daily_dqn": "Daily DQN",
    }
```

- [ ] **Step 2: Verify the existing Discord test still passes**

Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/test_predict.py -v -k TestSendDiscord`
Expected: All PASS (these tests only ever use `daily_logistic`, unaffected by adding a new dict entry).

- [ ] **Step 3: Update `CLAUDE.md`**

In `CLAUDE.md`, find the "**Experimental: prediction/strategy split.**" paragraph (currently describes `train_predictor.py`/`DailyPredictorStrategy` as "Not wired into the production Discord pipeline yet"). Replace that paragraph with:

```markdown
**Prediction/strategy split.** `train_models.py`'s Logistic/XGBoost/hybrid models classify the
discretized SELL/HOLD/BUY action directly — discretization bakes a decision threshold into the
training target. `train_predictor.py` instead trains a Ridge regression on the continuous
forward-return target (evaluated by Spearman IC, not accuracy), and
`ml_strategies.DailyPredictorStrategy` is a separate, independently-tunable decision layer that
converts those forecasts into trade signals via a rolling-quantile threshold
(`ml_strategies.compute_predictor_signal` — the single shared implementation used by both
backtesting and the live pipeline). `daily_predictor` is wired into `predict_next_day_lite.py`
and Discord alongside `daily_logistic`/`daily_xgboost` — see `models/README.md` → "Prediction vs.
strategy" for the backtest comparison and honest caveats.
```bash
python train_predictor.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 2500
```
```

Find the "**To change what symbols get predicted:**" paragraph under "Deployment (GitHub Actions)" and add a new paragraph immediately after it:

```markdown
**To change which models predict (add/remove a model from the live pipeline):**
Edit `prediction.models` in `config/default.yaml` and push. `predict_next_day_lite.py` reads this
list at startup — a model removed from the list is simply not loaded or predicted; a model added
to the list whose pickle is missing is logged and skipped, not a hard failure. No workflow YAML
edit needed.
```

In the "Key files" table, update the `ml_strategies.py` row (already mentions `DailyPredictorStrategy`) to also note the shared function:

```markdown
| `ml_strategies.py` | `DailyLogisticStrategy`, `DailyXGBoostStrategy`, `DailyPredictorStrategy` wrappers; `compute_predictor_signal` is the shared decision-layer function used by both backtest and live prediction |
```

Add `test_config.py` to the "Running tests" list:

```markdown
- `test_config.py` — prediction.models / prediction.symbols YAML loading
```

- [ ] **Step 4: Update `models/README.md`**

In the "Prediction vs. strategy" section, replace the final paragraph (the one starting "Backtested head-to-head against...") by appending this new paragraph immediately after it:

```markdown
**Deployment status:** `daily_predictor` is wired into `predict_next_day_lite.py` and the live
Discord pipeline (`prediction.models` in `config/default.yaml`), alongside `daily_logistic` and
`daily_xgboost`. Live inference uses the exact same decision function as backtesting
(`ml_strategies.compute_predictor_signal`) so the two can't silently diverge. Its Discord
"confidence" field is not a calibrated probability like the classifiers' — it's the percentile
rank of today's |predicted return| within its trailing window (see
`predict_next_day_lite._regressor_confidence`). Given the caveats above, treat its live signals
with the same skepticism as the backtest: a promising lead under active validation, not a
proven edge.
```

- [ ] **Step 5: Commit**

```bash
git add predict_next_day_lite.py CLAUDE.md models/README.md
git commit -m "docs: document daily_predictor's production deployment and config-driven model list"
```

---

### Task 7: Full regression run and live smoke test

**Files:** None modified — verification only.

**Interfaces:** None new.

- [ ] **Step 1: Run the full automated test suite**

Run: `C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: Every test passes — the pre-existing suite (143 tests as of the last full run) plus all tests added in Tasks 1-5. Note the new total count.

- [ ] **Step 2: Smoke-test the live pipeline locally against real (committed) models**

Run from the worktree root (no `DISCORD_WEBHOOK_URL` set, so this only writes `tomorrow_trades.json` and logs to stdout — it will not post anywhere):

```bash
C:/Users/sssor/Documents/trading_sim/.venv/Scripts/python.exe predict_next_day_lite.py --symbols AAPL,SPY --history ""
```

Expected: stdout's `DAILY PREDICTIONS` summary block shows a `daily_predictor=<SIGNAL> (<NN>%)` entry alongside `daily_logistic=` and `daily_xgboost=` for both AAPL and SPY, with no `ERROR` for either symbol. Confirm by reading the printed summary lines.

- [ ] **Step 3: Inspect the written artifact**

Read `tomorrow_trades.json` (Read tool) and confirm each symbol's `predictions` object has a `daily_predictor` key with `signal` in `{"BUY","SELL","HOLD"}` and `confidence` in `[0,1]`, with no `error` key under it.

- [ ] **Step 4: Confirm git status is clean of anything unexpected**

Run: `git status --short`
Expected: only the files touched across Tasks 1-6 are modified/new, plus the freshly-generated `tomorrow_trades.json` (gitignored or harmless to leave untracked — do not commit it). No unrelated changes.

- [ ] **Step 5: Report completion**

No commit in this task (verification only). Summarize for the user: test count before/after, confirmation that `daily_predictor` now appears in live prediction output end-to-end, and a reminder that `predictions/history.jsonl` and Discord posting only happen on the scheduled/manual GitHub Actions run (per `.github/workflows/simulation.yaml`), not on this local smoke test.

---

## Self-Review Notes

- **Spec coverage:** "Deploy the modularized architecture" → Tasks 3-5 wire `daily_predictor` into `predict_next_day_lite.py` end-to-end (load, predict, DB upsert, Discord). "Easy to add/remove models without things breaking" → Task 2 (config-driven model list), Task 4 (`MODEL_KINDS` registry replacing copy-pasted blocks, with a per-model try/except so one bad model doesn't crash a symbol's whole prediction, verified by `test_unknown_model_kind_reports_error_not_crash`), Task 5 (`load_models` skips missing artifacts per-model rather than failing the run).
- **Placeholder scan:** No TBD/TODO; every step has complete code or an exact command with stated expected output.
- **Type consistency:** `_predict_classifier_signal`/`_predict_regressor_signal`/`_regressor_confidence` signatures introduced in Task 3 are used unchanged in Task 4's dispatch loop and Task 5's tests. `compute_predictor_signal`'s signature from Task 1 (`pred_ret, signal_quantile, threshold_window`) matches its call site in Task 3's `_predict_regressor_signal`. `MODEL_KINDS` keys (`"daily_logistic"`, `"daily_xgboost"`, `"daily_predictor"`) match `PredictionCfg.models`'s default list from Task 2 and the canonical pickle naming convention `models/<model_key>.pkl` already used by `load_models`.
- **Backward compatibility:** Every pre-existing test in `tests/test_predict.py` and `tests/test_predictor.py` is preserved unmodified — Tasks 1, 4, and 5 are explicitly behavior-preserving extractions for the already-deployed models, verified by re-running the full pre-existing test set after each refactor step.
