# Step 3 Phase A — Cross-sectional panel backtester: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a weight-based cross-sectional panel backtester that ranks ~150 large-cap US stocks daily, holds an equal-weight dollar-neutral decile long-short book, and reports whether that book clears a DSR ≥ 0.95 gate on net-of-cost returns.

**Architecture:** A new standalone engine, separate from the single-symbol `Backtester`. Three aligned `date × symbol` frames (predictions, 1-day returns, closes) feed a pure ranking function, which feeds a loop that accrues costs and produces an equity curve. Evaluation reuses the existing `deflated_sharpe` and `cpcv.cscv_pbo` modules from #106.

**Tech Stack:** Python 3, numpy, pandas, scikit-learn (existing `RidgePredictor`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-15-step3-panel-portfolio-design.md`

## Global Constraints

Every task's requirements implicitly include these.

- **Touch no live path.** Do not modify `predict_next_day_lite.py`, `prediction.models`, `.github/workflows/simulation.yaml`, or any Discord code. Phase A is research-only.
- **`ret_panel` is built from `close.pct_change().shift(-1)` — never from `daily_features.fwd_ret_1d`.** Despite its name, `fwd_ret_1d` is a `FWD_RET_HORIZON_DAYS`-bar (3-day) cumulative return. Paying it per daily bar triples the book's apparent return via double-counted overlapping windows.
- **Never forward-fill prices.** A symbol missing on date `t` leaves that date's cross-section. Commit `3255f87` fixed exactly this leak in `_standardize`.
- **The lag convention:** weights derived from predictions on `close[t]` earn the return `close[t] → close[t+1]`. Never `t-1 → t`.
- **`cost_bps` is a fixed assumption, not a tunable parameter.** It never enters the PBO config grid. The grid is `decile ∈ {0.1, 0.2} × rebalance_days ∈ {1, 3}` — four configs, nothing else.
- **Deflated Sharpe takes per-period Sharpe, never annualized.** Passing an annualized Sharpe variance made `sr0` ~15.9× too large and drove DSR to 0 for every symbol (bug fixed in #106).
- **Universe is stocks only.** No SPY/QQQ/IWM/XL* in the tradeable cross-section. SPY loads for `ret_*_vs_spy` features and the beta regression only.
- Feature set is `daily_v6`, 30 columns. Index features via `FEATURE_COLS`, never by column position.
- Run tests with `pytest tests/ -v`. All existing tests (332) must stay green.

---

### Task 1: `panel:` configuration block

**Files:**
- Modify: `config/default.yaml` (append a `panel:` block)
- Modify: `config.py:126-145` (add `PanelCfg`, wire into `AppConfig`), `config.py:158-205` (wire into `load_config`)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `config.PanelCfg` dataclass with fields `universe: list[str]`, `decile: float`, `rebalance_days: int`, `gross_exposure: float`, `cost_bps: float`, `borrow_bps_annual: float`, `min_names: int`. Reachable as `get_config().panel`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_panel_config_loads_from_yaml():
    cfg = load_config("config/default.yaml")
    assert cfg.panel.decile == 0.1
    assert cfg.panel.rebalance_days == 1
    assert cfg.panel.gross_exposure == 1.0
    assert cfg.panel.cost_bps == 5.0
    assert cfg.panel.borrow_bps_annual == 50.0
    assert cfg.panel.min_names == 20


def test_panel_universe_is_stocks_only():
    cfg = load_config("config/default.yaml")
    banned = {"SPY", "QQQ", "IWM", "DIA", "GLD", "USO",
              "XLF", "XLV", "XLE", "XLY", "XLI", "XLK", "XLP", "XLU", "XLB", "XLRE"}
    overlap = banned & set(cfg.panel.universe)
    assert not overlap, f"index/sector ETFs must not be in the panel cross-section: {overlap}"


def test_panel_universe_is_wide_and_unique():
    cfg = load_config("config/default.yaml")
    assert len(cfg.panel.universe) >= 100, "panel needs breadth to be worth running"
    assert len(cfg.panel.universe) == len(set(cfg.panel.universe)), "duplicate symbols"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -k panel -v`
Expected: FAIL with `AttributeError: 'AppConfig' object has no attribute 'panel'`

- [ ] **Step 3: Add `PanelCfg` to `config.py`**

Insert after the `PredictionCfg` dataclass (around `config.py:133`):

```python
@dataclass
class PanelCfg:
    """Cross-sectional panel backtester (research-only; see docs/superpowers/specs/2026-07-15-step3-panel-portfolio-design.md).

    universe is stocks only — index/sector ETFs are baskets of the same names
    and must not be ranked against their own constituents.
    """
    universe: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    decile: float = 0.1
    rebalance_days: int = 1
    gross_exposure: float = 1.0
    cost_bps: float = 5.0
    borrow_bps_annual: float = 50.0
    min_names: int = 20
```

Add the field to `AppConfig` (after `prediction`, around `config.py:138`):

```python
    panel: PanelCfg = field(default_factory=PanelCfg)
```

In `load_config`, add near the other `*_raw` lines (around `config.py:171`):

```python
    panel_raw = raw.get("panel", {})
```

And add to the `AppConfig(...)` return (after the `prediction=...` block):

```python
        panel=PanelCfg(
            universe=panel_raw.get(
                "universe", PanelCfg.__dataclass_fields__["universe"].default_factory()
            ),
            **{
                k: v for k, v in panel_raw.items()
                if k in PanelCfg.__dataclass_fields__ and k != "universe"
            },
        ),
```

- [ ] **Step 4: Add the `panel:` block to `config/default.yaml`**

Append to `config/default.yaml`:

```yaml
# Cross-sectional panel backtester (research-only — not part of the live
# prediction path). See docs/superpowers/specs/2026-07-15-step3-panel-portfolio-design.md
#
# universe is STOCKS ONLY. Index/sector ETFs (SPY, QQQ, XL*) are baskets of
# these same names — ranking a stock against a fund that holds it is not a
# cross-sectional bet. SPY still loads for ret_*_vs_spy features and the
# gate's beta regression, but is never ranked or held.
panel:
  decile: 0.1               # top/bottom fraction of the cross-section
  rebalance_days: 1         # daily
  gross_exposure: 1.0       # 0.5 long + 0.5 short; net exposure 0
  cost_bps: 5.0             # one-way cost on turnover notional (ASSUMPTION, not a tunable)
  borrow_bps_annual: 50.0   # flat GC-style rate on short notional
  min_names: 20             # below this, hold no position that day
  universe:
    # Technology (30)
    - AAPL
    - MSFT
    - GOOGL
    - AMZN
    - NVDA
    - META
    - TSLA
    - AVGO
    - ORCL
    - CRM
    - ADBE
    - AMD
    - INTC
    - CSCO
    - QCOM
    - TXN
    - INTU
    - IBM
    - NOW
    - AMAT
    - MU
    - LRCX
    - ADI
    - KLAC
    - SNPS
    - CDNS
    - PANW
    - ANET
    - MRVL
    - ACN
    # Financials (20)
    - BRK-B
    - JPM
    - V
    - MA
    - BAC
    - WFC
    - GS
    - MS
    - SCHW
    - BLK
    - AXP
    - C
    - SPGI
    - CB
    - PGR
    - USB
    - PNC
    - TFC
    - COF
    - MET
    # Healthcare (20)
    - UNH
    - JNJ
    - LLY
    - ABBV
    - MRK
    - PFE
    - TMO
    - ABT
    - DHR
    - BMY
    - AMGN
    - CVS
    - MDT
    - GILD
    - ISRG
    - VRTX
    - REGN
    - ZTS
    - BSX
    - HCA
    # Consumer (20)
    - WMT
    - PG
    - KO
    - PEP
    - COST
    - MCD
    - NKE
    - SBUX
    - TGT
    - LOW
    - HD
    - TJX
    - CL
    - MDLZ
    - MO
    - PM
    - KMB
    - GIS
    - EL
    - YUM
    # Industrials (20)
    - CAT
    - BA
    - HON
    - UNP
    - GE
    - LMT
    - RTX
    - DE
    - UPS
    - MMM
    - ADP
    - NOC
    - GD
    - CSX
    - NSC
    - EMR
    - ETN
    - ITW
    - PH
    - FDX
    # Energy (14)
    - XOM
    - CVX
    - COP
    - SLB
    - EOG
    - MPC
    - PSX
    - VLO
    - OXY
    - WMB
    - KMI
    - HAL
    - DVN
    - HES
    # Communications (10)
    - DIS
    - NFLX
    - CMCSA
    - T
    - VZ
    - TMUS
    - CHTR
    - EA
    - WBD
    - OMC
    # Utilities & Real Estate (14)
    - NEE
    - DUK
    - SO
    - D
    - AEP
    - EXC
    - SRE
    - XEL
    - AMT
    - PLD
    - CCI
    - EQIX
    - SPG
    - PSA
    # Materials (9)
    - LIN
    - APD
    - SHW
    - ECL
    - NEM
    - FCX
    - DOW
    - DD
    - PPG
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (all config tests, including the three new panel tests)

- [ ] **Step 6: Commit**

```bash
git add config.py config/default.yaml tests/test_config.py
git commit -m "feat: add panel config block (157-stock universe, stocks only)"
```

---

### Task 2: `panel_data.py` — aligned prediction/return/close panels

**Files:**
- Create: `panel_data.py`
- Test: `tests/test_panel_data.py`

**Interfaces:**
- Consumes: `config.PanelCfg` (Task 1). `train_models._load_symbol(symbol: str, start: str, end: str, db: DB) -> pd.DataFrame | None`. `daily_features.make_daily_features(df, spy_df=None) -> pd.DataFrame`. `predictors.ridge.RidgePredictor.load(path) -> RidgePredictor` with `.predict(X: np.ndarray) -> tuple[np.ndarray, None]`.
- Produces:
  - `PanelData` dataclass: `.pred: pd.DataFrame`, `.ret: pd.DataFrame`, `.close: pd.DataFrame` (all `date × symbol`), `.symbols: list[str]`, `.dropped: dict[str, str]`.
  - `build_panels(symbols: list[str], start: str, end: str, db, model_path: str = "models/daily_predictor.pkl", spy_df=None, min_bars: int = 250, predictor=None, load_fn=None) -> PanelData`
  - `MAX_LOAD_FAILURE_FRAC: float = 0.10`

`predictor` and `load_fn` are injection seams for tests — production callers omit both.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_panel_data.py`:

```python
import numpy as np
import pandas as pd
import pytest

from panel_data import build_panels, MAX_LOAD_FAILURE_FRAC


class FakePredictor:
    """Returns the first feature column as the score. Deterministic, no pickle."""

    def predict(self, X):
        return X[:, 0].astype(float), None


def _make_bars(n=400, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


def _loader(bars_by_symbol):
    def load_fn(symbol, start, end, db):
        return bars_by_symbol.get(symbol)
    return load_fn


def test_ret_panel_is_one_day_return_not_the_three_day_target():
    """ret must be close[t]->close[t+1], NOT daily_features.fwd_ret_1d (3-bar cumulative)."""
    bars = {"AAA": _make_bars(seed=1)}
    pd_out = build_panels(
        ["AAA"], "2020-01-01", "2021-06-01", db=None,
        predictor=FakePredictor(), load_fn=_loader(bars),
    )
    close = pd_out.close["AAA"]
    expected = close.pct_change().shift(-1)
    pd.testing.assert_series_equal(
        pd_out.ret["AAA"], expected, check_names=False,
    )
    # A 3-day cumulative return would be ~3x larger in magnitude on average.
    three_day = (close.shift(-3) / close) - 1
    assert not np.allclose(
        pd_out.ret["AAA"].dropna().values[:50],
        three_day.dropna().values[:50],
    ), "ret_panel is using the 3-day horizon target — see Global Constraints"


def test_ragged_histories_align_outer_without_forward_fill():
    bars = {
        "AAA": _make_bars(n=400, seed=1, start="2020-01-01"),
        "BBB": _make_bars(n=300, seed=2, start="2020-06-01"),
    }
    out = build_panels(
        ["AAA", "BBB"], "2020-01-01", "2021-12-31", db=None,
        predictor=FakePredictor(), load_fn=_loader(bars),
    )
    assert set(out.pred.columns) == {"AAA", "BBB"}
    # BBB starts later: its earliest rows must be NaN, not back/forward-filled.
    assert out.pred["BBB"].isna().any()
    first_bbb = out.pred["BBB"].first_valid_index()
    assert out.pred.loc[out.pred.index < first_bbb, "BBB"].isna().all()


def test_symbol_with_insufficient_history_is_dropped_and_recorded():
    bars = {"AAA": _make_bars(n=400, seed=1), "TINY": _make_bars(n=60, seed=3)}
    out = build_panels(
        ["AAA", "TINY"], "2020-01-01", "2021-12-31", db=None,
        predictor=FakePredictor(), load_fn=_loader(bars), min_bars=250,
    )
    assert "TINY" not in out.pred.columns
    assert "TINY" in out.dropped
    assert "insufficient" in out.dropped["TINY"]


def test_raises_when_too_much_of_the_universe_fails_to_load():
    bars = {"AAA": _make_bars(n=400, seed=1)}
    symbols = ["AAA"] + [f"MISSING{i}" for i in range(9)]
    with pytest.raises(RuntimeError, match="Refusing to backtest"):
        build_panels(
            symbols, "2020-01-01", "2021-12-31", db=None,
            predictor=FakePredictor(), load_fn=_loader(bars),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_panel_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'panel_data'`

- [ ] **Step 3: Write `panel_data.py`**

```python
"""
panel_data.py — Build aligned cross-sectional panels for panel_backtester.

Research-only: nothing here touches the live prediction path.

Produces three date x symbol frames — predictions, 1-day forward returns, and
closes — outer-joined across symbols with ragged histories left as NaN. A NaN
means "this symbol has no cross-section entry today", which is the correct
signal for the ranker to skip it. Never forward-fill: commit 3255f87 fixed
exactly that leak in data_loader._standardize.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from daily_features import FEATURE_COLS, make_daily_features
from predictors.ridge import RidgePredictor

logger = logging.getLogger(__name__)

# Above this fraction of the configured universe failing to load, raise rather
# than silently backtest a truncated universe — a 40-name result reported as a
# 150-name result describes nothing.
MAX_LOAD_FAILURE_FRAC = 0.10


@dataclass
class PanelData:
    pred: pd.DataFrame    # date x symbol — Ridge forecast (3-day cumulative target)
    ret: pd.DataFrame     # date x symbol — 1-day simple return, close[t] -> close[t+1]
    close: pd.DataFrame   # date x symbol
    symbols: list[str]
    dropped: dict[str, str]


def build_panels(
    symbols: list[str],
    start: str,
    end: str,
    db,
    model_path: str = "models/daily_predictor.pkl",
    spy_df: pd.DataFrame | None = None,
    min_bars: int = 250,
    predictor=None,
    load_fn=None,
) -> PanelData:
    """Load bars, featurize, predict, and assemble aligned panels.

    predictor / load_fn are injection seams for tests; production callers omit
    both and get RidgePredictor.load(model_path) and train_models._load_symbol.
    """
    if predictor is None:
        predictor = RidgePredictor.load(model_path)
    if load_fn is None:
        from train_models import _load_symbol  # deferred: heavy import chain
        load_fn = _load_symbol

    preds: dict[str, pd.Series] = {}
    rets: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    dropped: dict[str, str] = {}

    for symbol in symbols:
        try:
            df = load_fn(symbol, start, end, db)
        except Exception as exc:
            dropped[symbol] = f"load failed: {exc}"
            continue

        n_bars = 0 if df is None else len(df)
        if df is None or n_bars < min_bars:
            dropped[symbol] = f"insufficient bars ({n_bars} < {min_bars})"
            continue

        try:
            spy_arg = spy_df if symbol != "SPY" else None
            feats = make_daily_features(df, spy_df=spy_arg)
        except Exception as exc:
            dropped[symbol] = f"feature error: {exc}"
            continue

        if feats.empty:
            dropped[symbol] = "no rows survived feature warmup"
            continue

        X = feats[FEATURE_COLS].values.astype(np.float32)
        scores, _ = predictor.predict(X)

        close = feats["close"].astype(float)
        preds[symbol] = pd.Series(scores, index=feats.index)
        closes[symbol] = close
        # 1-day simple forward return. NOT feats["fwd_ret_1d"], which is a
        # FWD_RET_HORIZON_DAYS-bar cumulative return despite its name — paying
        # that per daily bar would triple the book's return.
        rets[symbol] = close.pct_change().shift(-1)

    if not preds:
        raise RuntimeError(f"No symbols produced predictions. Dropped: {dropped}")

    fail_frac = len(dropped) / len(symbols)
    if fail_frac > MAX_LOAD_FAILURE_FRAC:
        raise RuntimeError(
            f"Refusing to backtest a truncated universe: {len(dropped)}/{len(symbols)} "
            f"symbols failed ({fail_frac:.0%} > {MAX_LOAD_FAILURE_FRAC:.0%}). "
            f"Dropped: {dropped}"
        )

    if dropped:
        logger.warning("Dropped %d/%d symbols: %s", len(dropped), len(symbols), dropped)

    # DataFrame(dict_of_series) outer-joins on index. NaN stays NaN.
    pred_panel = pd.DataFrame(preds).sort_index()
    ret_panel = pd.DataFrame(rets).reindex(pred_panel.index)
    close_panel = pd.DataFrame(closes).reindex(pred_panel.index)

    return PanelData(
        pred=pred_panel,
        ret=ret_panel,
        close=close_panel,
        symbols=sorted(preds),
        dropped=dropped,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_panel_data.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add panel_data.py tests/test_panel_data.py
git commit -m "feat: add panel_data — aligned pred/ret/close panels

ret_panel is built from close.pct_change().shift(-1), never from
fwd_ret_1d (a 3-bar cumulative return despite its name). Ragged
histories outer-join to NaN and are never forward-filled."
```

---

### Task 3: `rank_to_weights` — the pure ranking function

**Files:**
- Create: `panel_backtester.py`
- Test: `tests/test_panel_backtester.py`

**Interfaces:**
- Consumes: nothing (pure function over a `pd.Series`).
- Produces: `rank_to_weights(pred_row: pd.Series, decile: float, gross_exposure: float, min_names: int) -> pd.Series` — returns weights indexed like `pred_row`, `0.0` for untraded names, summing to 0.0 (dollar-neutral) with `abs().sum() == gross_exposure` when it trades.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_panel_backtester.py`:

```python
import numpy as np
import pandas as pd
import pytest

from panel_backtester import rank_to_weights


def test_equal_weight_long_top_short_bottom():
    pred = pd.Series({f"S{i}": float(i) for i in range(10)})
    w = rank_to_weights(pred, decile=0.2, gross_exposure=1.0, min_names=5)
    # k = int(10 * 0.2) = 2. Longs = two highest (S8, S9), shorts = two lowest (S0, S1).
    assert w["S9"] == pytest.approx(0.25)
    assert w["S8"] == pytest.approx(0.25)
    assert w["S0"] == pytest.approx(-0.25)
    assert w["S1"] == pytest.approx(-0.25)
    assert w["S5"] == pytest.approx(0.0)


def test_book_is_dollar_neutral_and_hits_gross_exposure():
    pred = pd.Series({f"S{i}": float(i) for i in range(20)})
    w = rank_to_weights(pred, decile=0.1, gross_exposure=1.0, min_names=5)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert w.abs().sum() == pytest.approx(1.0)


def test_gross_exposure_scales_weights():
    pred = pd.Series({f"S{i}": float(i) for i in range(20)})
    w = rank_to_weights(pred, decile=0.1, gross_exposure=2.0, min_names=5)
    assert w.abs().sum() == pytest.approx(2.0)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)


def test_below_min_names_holds_nothing():
    pred = pd.Series({f"S{i}": float(i) for i in range(5)})
    w = rank_to_weights(pred, decile=0.2, gross_exposure=1.0, min_names=20)
    assert (w == 0.0).all()


def test_nan_predictions_are_excluded_from_the_cross_section():
    pred = pd.Series({f"S{i}": float(i) for i in range(10)})
    pred["S9"] = np.nan   # would otherwise be the top long
    w = rank_to_weights(pred, decile=0.2, gross_exposure=1.0, min_names=5)
    assert w["S9"] == 0.0
    # k = int(9 * 0.2) = 1 -> single long is now S8, single short is S0.
    assert w["S8"] == pytest.approx(0.5)
    assert w["S0"] == pytest.approx(-0.5)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)


def test_legs_never_overlap_at_large_decile():
    pred = pd.Series({f"S{i}": float(i) for i in range(4)})
    w = rank_to_weights(pred, decile=0.9, gross_exposure=1.0, min_names=2)
    # k is capped at len//2 so a name is never both long and short.
    assert (w > 0).sum() == (w < 0).sum() == 2
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_panel_backtester.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'panel_backtester'`

- [ ] **Step 3: Write `rank_to_weights` in `panel_backtester.py`**

```python
"""
panel_backtester.py — Weight-based cross-sectional panel engine.

Research instrument, deliberately NOT a deployability simulation: it models no
share granularity, assumes fills at the close, and ignores per-name market
impact. See docs/superpowers/specs/2026-07-15-step3-panel-portfolio-design.md.

Deliberately omits Backtester's stop_loss_pct / take_profit_pct /
daily_loss_limit_pct / forced-exit cooldown. Those are per-name path-dependent
rules, and a stop-loss that exits one leg breaks the dollar- and beta-neutrality
the book exists to test.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rank_to_weights(
    pred_row: pd.Series,
    decile: float,
    gross_exposure: float,
    min_names: int,
) -> pd.Series:
    """Equal-weight long the top `decile` / short the bottom `decile` of one date.

    Ranking is location-invariant, so this neutralizes market-wide moves in the
    forecast for free: adding a constant to every prediction leaves the ranking
    unchanged. Sector neutrality is NOT provided (deferred increment).

    Returns weights indexed like pred_row, 0.0 for untraded names. Dollar-neutral:
    long notional == short notional == gross_exposure / 2.
    """
    weights = pd.Series(0.0, index=pred_row.index)
    valid = pred_row.dropna()
    if len(valid) < min_names:
        return weights

    k = int(len(valid) * decile)
    k = min(k, len(valid) // 2)   # legs must never overlap
    if k < 1:
        return weights

    ranked = valid.sort_values()
    shorts = ranked.index[:k]
    longs = ranked.index[-k:]

    leg = gross_exposure / 2.0
    weights.loc[longs] = leg / k
    weights.loc[shorts] = -leg / k
    return weights
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_panel_backtester.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add panel_backtester.py tests/test_panel_backtester.py
git commit -m "feat: add rank_to_weights — equal-weight dollar-neutral decile ranking"
```

---

### Task 4: `run_panel` — the engine loop

**Files:**
- Modify: `panel_backtester.py` (append `PanelConfig`, `PanelResult`, `run_panel`)
- Test: `tests/test_panel_backtester.py` (append)

**Interfaces:**
- Consumes: `rank_to_weights` (Task 3). `PanelData.pred` / `.ret` (Task 2).
- Produces:
  - `PanelConfig` dataclass: `decile: float = 0.1`, `rebalance_days: int = 1`, `gross_exposure: float = 1.0`, `cost_bps: float = 5.0`, `borrow_bps_annual: float = 50.0`, `min_names: int = 20`, `start_cash: float = 100_000.0`
  - `PanelResult` dataclass: `.equity: pd.Series`, `.book_ret: pd.Series`, `.gross_ret: pd.Series`, `.weights: pd.DataFrame`, `.turnover: pd.Series`, `.diagnostics: dict`
  - `run_panel(pred: pd.DataFrame, ret: pd.DataFrame, cfg: PanelConfig) -> PanelResult`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_panel_backtester.py`:

```python
from panel_backtester import PanelConfig, run_panel

# Loose by design — see test_stale_prediction_earns_nothing_lookahead_guard.
# Separates "no edge" from "reading the future", not a tight estimate of zero.
NO_EDGE_SHARPE = 2.0


def _synthetic_panel(n_dates=500, n_symbols=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_dates)
    cols = [f"S{i}" for i in range(n_symbols)]
    ret = pd.DataFrame(
        rng.normal(0.0, 0.02, (n_dates, n_symbols)), index=idx, columns=cols
    )
    return ret


def _sharpe(book_ret):
    r = book_ret.dropna()
    return float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else 0.0


def test_perfect_foresight_produces_large_sharpe():
    """If the engine cannot profit from tomorrow's actual returns, it is broken."""
    ret = _synthetic_panel()
    cfg = PanelConfig(min_names=5, cost_bps=0.0, borrow_bps_annual=0.0)
    res = run_panel(pred=ret, ret=ret, cfg=cfg)   # pred[t] == ret[t] == tomorrow
    assert _sharpe(res.book_ret) > 3.0


def test_stale_prediction_earns_nothing_lookahead_guard():
    """pred[t] = ret[t-1] is knowable at close[t] and predicts nothing.

    If this scores well, the engine is paying ret[t-1] instead of ret[t] — i.e.
    weights are earning the return they were derived from. This is the #106 bug
    class (close[t] decision filled at close[t]) in panel form.

    NO_EDGE_SHARPE is deliberately loose. This test separates "no edge" from
    "reading the future", which differ by an order of magnitude (perfect
    foresight scores >10 on this panel) — it is not a tight estimate of zero.
    Over 1500 days the standard error of an annualized Sharpe under the null is
    ~sqrt(252/1500) ~= 0.41, so a 2.0 bound is ~5 SE: robust to reseeding, and
    still nowhere near a look-ahead's score.
    """
    ret = _synthetic_panel(n_dates=1500)
    stale = ret.shift(1)
    cfg = PanelConfig(min_names=5, cost_bps=0.0, borrow_bps_annual=0.0)
    res = run_panel(pred=stale, ret=ret, cfg=cfg)
    assert abs(_sharpe(res.book_ret)) < NO_EDGE_SHARPE


def test_random_predictions_produce_no_edge():
    ret = _synthetic_panel(n_dates=1500, seed=1)
    rng = np.random.default_rng(99)
    noise = pd.DataFrame(
        rng.normal(0, 1, ret.shape), index=ret.index, columns=ret.columns
    )
    cfg = PanelConfig(min_names=5, cost_bps=0.0, borrow_bps_annual=0.0)
    res = run_panel(pred=noise, ret=ret, cfg=cfg)
    assert abs(_sharpe(res.book_ret)) < NO_EDGE_SHARPE


def test_book_is_dollar_neutral_every_traded_day():
    ret = _synthetic_panel()
    rng = np.random.default_rng(7)
    noise = pd.DataFrame(
        rng.normal(0, 1, ret.shape), index=ret.index, columns=ret.columns
    )
    res = run_panel(pred=noise, ret=ret, cfg=PanelConfig(min_names=5))
    assert res.weights.sum(axis=1).abs().max() < 1e-12


def test_zero_turnover_charges_zero_cost():
    """Constant predictions -> same weights every day -> no turnover after day 1."""
    ret = _synthetic_panel(n_dates=100, n_symbols=20)
    const = pd.DataFrame(
        np.tile(np.arange(20, dtype=float), (100, 1)),
        index=ret.index, columns=ret.columns,
    )
    cfg = PanelConfig(min_names=5, cost_bps=5.0, borrow_bps_annual=0.0)
    res = run_panel(pred=const, ret=ret, cfg=cfg)
    # Day 1 establishes the book (turnover == gross); every later day is flat.
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    assert res.turnover.iloc[1:].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_borrow_accrues_on_short_notional_only():
    ret = _synthetic_panel(n_dates=100, n_symbols=20)
    const = pd.DataFrame(
        np.tile(np.arange(20, dtype=float), (100, 1)),
        index=ret.index, columns=ret.columns,
    )
    cfg = PanelConfig(min_names=5, cost_bps=0.0, borrow_bps_annual=50.0,
                      gross_exposure=1.0)
    res = run_panel(pred=const, ret=ret, cfg=cfg)
    # Short notional is gross/2 = 0.5; daily borrow = 0.5 * 50/1e4/252.
    expected = 0.5 * (50.0 / 1e4 / 252.0)
    assert res.diagnostics["mean_borrow_cost"] == pytest.approx(expected, rel=1e-6)


def test_costs_reduce_returns_versus_gross():
    ret = _synthetic_panel(seed=3)
    rng = np.random.default_rng(11)
    noise = pd.DataFrame(
        rng.normal(0, 1, ret.shape), index=ret.index, columns=ret.columns
    )
    free = run_panel(noise, ret, PanelConfig(min_names=5, cost_bps=0.0,
                                            borrow_bps_annual=0.0))
    charged = run_panel(noise, ret, PanelConfig(min_names=5, cost_bps=20.0,
                                                borrow_bps_annual=50.0))
    assert charged.book_ret.mean() < free.book_ret.mean()
    pd.testing.assert_series_equal(free.gross_ret, charged.gross_ret)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_panel_backtester.py -k "foresight or lookahead or random or neutral or turnover or borrow or costs" -v`
Expected: FAIL with `ImportError: cannot import name 'PanelConfig' from 'panel_backtester'`

- [ ] **Step 3: Append the engine to `panel_backtester.py`**

```python
@dataclass
class PanelConfig:
    decile: float = 0.1
    rebalance_days: int = 1
    gross_exposure: float = 1.0
    cost_bps: float = 5.0              # one-way, on turnover notional (assumption)
    borrow_bps_annual: float = 50.0    # on short notional
    min_names: int = 20
    start_cash: float = 100_000.0


@dataclass
class PanelResult:
    equity: pd.Series
    book_ret: pd.Series
    gross_ret: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    diagnostics: dict = field(default_factory=dict)


def run_panel(pred: pd.DataFrame, ret: pd.DataFrame, cfg: PanelConfig) -> PanelResult:
    """Run the cross-sectional book over the panel.

    LAG CONVENTION: weights on date t come from predictions computed on close[t],
    and earn ret[t], which panel_data defines as close[t] -> close[t+1]. The
    weight and the return it earns therefore share index t. Shifting either side
    reintroduces the #106 look-ahead.
    """
    ret = ret.reindex(index=pred.index, columns=pred.columns)
    dates = pred.index

    weight_rows: list[pd.Series] = []
    turnover_vals: list[float] = []
    prev_w = pd.Series(0.0, index=pred.columns)
    n_flat = 0

    for i, date in enumerate(dates):
        if i % cfg.rebalance_days == 0:
            w = rank_to_weights(
                pred.loc[date], cfg.decile, cfg.gross_exposure, cfg.min_names
            )
        else:
            w = prev_w.copy()
        if (w == 0.0).all():
            n_flat += 1
        weight_rows.append(w)
        turnover_vals.append(float((w - prev_w).abs().sum()))
        prev_w = w

    weights = pd.DataFrame(weight_rows, index=dates)
    turnover = pd.Series(turnover_vals, index=dates)

    cost = turnover * (cfg.cost_bps / 1e4)
    short_notional = weights.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_notional * (cfg.borrow_bps_annual / 1e4 / 252.0)

    # skipna: a name holding weight whose next-day bar is missing earns 0 rather
    # than poisoning the whole day's book return with NaN.
    gross_ret = (weights * ret).sum(axis=1, skipna=True)
    book_ret = gross_ret - cost - borrow
    equity = cfg.start_cash * (1.0 + book_ret).cumprod()

    diagnostics = {
        "n_dates": int(len(dates)),
        "n_flat_days": int(n_flat),
        "mean_turnover": float(turnover.mean()),
        "mean_cost": float(cost.mean()),
        "mean_borrow_cost": float(borrow.mean()),
        "total_cost_drag": float(cost.sum() + borrow.sum()),
        "mean_gross_exposure": float(weights.abs().sum(axis=1).mean()),
        "mean_net_exposure": float(weights.sum(axis=1).mean()),
    }

    return PanelResult(
        equity=equity,
        book_ret=book_ret,
        gross_ret=gross_ret,
        weights=weights,
        turnover=turnover,
        diagnostics=diagnostics,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_panel_backtester.py -v`
Expected: PASS (13 tests total)

- [ ] **Step 5: Run the full suite for regressions**

Run: `pytest tests/ -q`
Expected: PASS — 332 existing tests plus the new ones, no failures

- [ ] **Step 6: Commit**

```bash
git add panel_backtester.py tests/test_panel_backtester.py
git commit -m "feat: add run_panel engine — costs, borrow, lag-correct book returns

Includes a stale-prediction look-ahead guard: pred[t]=ret[t-1] is
knowable at close[t] and must score ~0. If it scores well, the engine
is paying the return the weights were derived from (the #106 bug in
panel form)."
```

---

### Task 5: `panel_eval.py` — beta, DSR gate, PBO grid

**Files:**
- Create: `panel_eval.py`
- Test: `tests/test_panel_eval.py`

**Interfaces:**
- Consumes: `PanelResult` (Task 4). `deflated_sharpe.deflated_sharpe(returns: np.ndarray, n_trials: int, trial_sharpe_var: float) -> dict` with keys `dsr`, `sr`, `sr0`, `p_value`. `cpcv.cscv_pbo(perf_matrix: np.ndarray, n_splits: int = 16) -> dict` with keys `pbo`, `logits`, `n_combinations`, `reason`.
- Produces:
  - `CONFIG_GRID: list[tuple[float, int]]` — `[(0.1, 1), (0.1, 3), (0.2, 1), (0.2, 3)]`
  - `book_beta(book_ret: pd.Series, spy_ret: pd.Series) -> float`
  - `evaluate_grid(results: dict[tuple[float, int], PanelResult], spy_ret: pd.Series) -> dict` with keys `best_config`, `dsr`, `sr`, `sr0`, `beta`, `pbo`, `passed`, `verdict`, `per_config`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_panel_eval.py`:

```python
import numpy as np
import pandas as pd
import pytest

from panel_backtester import PanelResult
from panel_eval import CONFIG_GRID, book_beta, evaluate_grid


def _result(book_ret):
    idx = book_ret.index
    return PanelResult(
        equity=(1 + book_ret).cumprod() * 100_000,
        book_ret=book_ret,
        gross_ret=book_ret,
        weights=pd.DataFrame(0.0, index=idx, columns=["A"]),
        turnover=pd.Series(0.0, index=idx),
        diagnostics={},
    )


def _idx(n=750):
    return pd.bdate_range("2020-01-01", periods=n)


def test_book_beta_recovers_a_known_beta():
    idx = _idx()
    rng = np.random.default_rng(0)
    spy = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    book = 0.5 * spy + pd.Series(rng.normal(0, 0.0001, len(idx)), index=idx)
    assert book_beta(book, spy) == pytest.approx(0.5, abs=0.02)


def test_book_beta_of_uncorrelated_book_is_near_zero():
    idx = _idx()
    rng = np.random.default_rng(1)
    spy = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    book = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    assert abs(book_beta(book, spy)) < 0.1


def test_grid_has_four_configs_and_excludes_cost_bps():
    assert len(CONFIG_GRID) == 4
    assert set(CONFIG_GRID) == {(0.1, 1), (0.1, 3), (0.2, 1), (0.2, 3)}


def test_high_beta_fails_the_gate_before_performance_is_read():
    idx = _idx()
    rng = np.random.default_rng(2)
    spy = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    # Strong returns, but the book is just levered SPY -> neutralization failed.
    book = 0.9 * spy + 0.002
    results = {cfg: _result(book) for cfg in CONFIG_GRID}
    out = evaluate_grid(results, spy)
    assert out["passed"] is False
    assert "beta" in out["verdict"].lower()


def test_no_edge_fails_the_dsr_gate():
    idx = _idx()
    rng = np.random.default_rng(3)
    spy = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    results = {
        cfg: _result(pd.Series(rng.normal(0, 0.01, len(idx)), index=idx))
        for cfg in CONFIG_GRID
    }
    out = evaluate_grid(results, spy)
    assert out["passed"] is False
    assert out["dsr"] < 0.95


def test_strong_neutral_book_passes_the_gate():
    idx = _idx()
    rng = np.random.default_rng(4)
    spy = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    results = {
        cfg: _result(pd.Series(0.0016 + rng.normal(0, 0.004, len(idx)), index=idx))
        for cfg in CONFIG_GRID
    }
    out = evaluate_grid(results, spy)
    assert abs(out["beta"]) < 0.1
    assert out["dsr"] > 0.95
    assert out["passed"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_panel_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'panel_eval'`

- [ ] **Step 3: Write `panel_eval.py`**

```python
"""
panel_eval.py — Gate and diagnostics for the cross-sectional panel book.

Gate (see the spec): deflated Sharpe >= 0.95 on net-of-cost book returns, with
|beta| < 0.1 vs SPY as a PRECONDITION. Beta is a correctness check, not a
performance one — a dollar-neutral book must have ~zero market beta by
construction, so beta outside the band means the neutralization failed and the
Sharpe describes something other than the intended book.

"Alpha vs buy-and-hold" is deliberately NOT the gate: a zero-beta book
underperforming a long benchmark is expected, not informative.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cpcv import cscv_pbo
from deflated_sharpe import deflated_sharpe

logger = logging.getLogger(__name__)

# decile x rebalance_days. cost_bps is an ASSUMPTION and never enters this grid:
# tuning a cost assumption until the gate passes is fiction, and it would make
# PBO measure the wrong thing.
CONFIG_GRID: list[tuple[float, int]] = [(0.1, 1), (0.1, 3), (0.2, 1), (0.2, 3)]

DSR_THRESHOLD = 0.95
BETA_LIMIT = 0.1
MIN_BETA_OBS = 30


def book_beta(book_ret: pd.Series, spy_ret: pd.Series) -> float:
    """OLS beta of the book against SPY. NaN if too few overlapping observations."""
    aligned = pd.concat(
        [book_ret.rename("book"), spy_ret.rename("spy")], axis=1
    ).dropna()
    if len(aligned) < MIN_BETA_OBS:
        return float("nan")
    var = aligned["spy"].var()
    if not var or var <= 0:
        return float("nan")
    return float(aligned["spy"].cov(aligned["book"]) / var)


def _per_period_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def evaluate_grid(
    results: dict[tuple[float, int], "object"],
    spy_ret: pd.Series,
) -> dict:
    """Score the config grid, deflate the winner's Sharpe, and apply the gate.

    results maps (decile, rebalance_days) -> PanelResult.
    """
    if not results:
        raise ValueError("evaluate_grid needs at least one config result")

    # Per-period (NOT annualized) Sharpe. Passing an annualized Sharpe variance
    # to deflated_sharpe made sr0 ~15.9x too large and drove DSR to 0 for every
    # symbol — the unit bug fixed in #106.
    pp_sharpes = {cfg: _per_period_sharpe(res.book_ret) for cfg, res in results.items()}
    best_config = max(pp_sharpes, key=pp_sharpes.get)
    best = results[best_config]

    trial_var = (
        float(np.var(list(pp_sharpes.values()), ddof=1)) if len(pp_sharpes) > 1 else 0.0
    )
    dsr_out = deflated_sharpe(
        best.book_ret.dropna().values,
        n_trials=len(results),
        trial_sharpe_var=trial_var,
    )

    beta = book_beta(best.book_ret, spy_ret)

    # PBO across the grid: observations x configs matrix of daily book returns.
    perf = pd.DataFrame({str(cfg): res.book_ret for cfg, res in results.items()}).dropna()
    pbo_out = cscv_pbo(perf.values) if perf.shape[1] >= 2 else {"pbo": float("nan")}

    if np.isnan(beta):
        passed, verdict = False, "beta undefined — too few overlapping observations with SPY"
    elif abs(beta) >= BETA_LIMIT:
        passed, verdict = False, (
            f"beta {beta:+.3f} outside +/-{BETA_LIMIT} — neutralization failed. "
            "Diagnose the book before reading its performance."
        )
    elif dsr_out["dsr"] < DSR_THRESHOLD:
        passed, verdict = False, (
            f"DSR {dsr_out['dsr']:.3f} < {DSR_THRESHOLD} — no statistically "
            "significant edge after multiple-testing correction. Null result: "
            "do not build Phase B."
        )
    else:
        passed, verdict = True, (
            f"DSR {dsr_out['dsr']:.3f} >= {DSR_THRESHOLD} with beta {beta:+.3f} — "
            "significant neutral edge. Phase B (vol-targeting) is justified."
        )

    return {
        "best_config": best_config,
        "dsr": float(dsr_out["dsr"]),
        "sr": float(dsr_out["sr"]),
        "sr0": float(dsr_out["sr0"]),
        "beta": beta,
        "pbo": float(pbo_out.get("pbo", float("nan"))),
        "passed": passed,
        "verdict": verdict,
        "per_config": {
            str(cfg): {
                "pp_sharpe": pp_sharpes[cfg],
                "ann_sharpe": pp_sharpes[cfg] * float(np.sqrt(252)),
                **results[cfg].diagnostics,
            }
            for cfg in results
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_panel_eval.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add panel_eval.py tests/test_panel_eval.py
git commit -m "feat: add panel_eval — DSR gate, beta precondition, PBO grid

cost_bps is excluded from CONFIG_GRID by design: it is a modelling
assumption, not a tunable. Tuning it until the gate passes would make
PBO measure the wrong thing."
```

---

### Task 6: `run_panel.py` CLI + results document

**Files:**
- Create: `run_panel.py`
- Create: `docs/superpowers/plans/2026-07-15-step3-results.md`
- Test: `tests/test_run_panel.py`

**Interfaces:**
- Consumes: `build_panels` (Task 2), `PanelConfig`/`run_panel` (Task 4), `CONFIG_GRID`/`evaluate_grid` (Task 5), `get_config().panel` (Task 1).
- Produces: `run_grid(panel_data, cfg_panel, spy_ret) -> dict` and a `main()` CLI entry point. Writes `results/panel_summary.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_panel.py`:

```python
import numpy as np
import pandas as pd

from panel_backtester import PanelConfig
from run_panel import run_grid


class _FakePanel:
    def __init__(self, pred, ret):
        self.pred = pred
        self.ret = ret
        self.close = pred
        self.symbols = list(pred.columns)
        self.dropped = {}


def test_run_grid_runs_every_config_and_returns_a_verdict():
    idx = pd.bdate_range("2020-01-01", periods=400)
    cols = [f"S{i}" for i in range(30)]
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(rng.normal(0, 0.02, (len(idx), 30)), index=idx, columns=cols)
    pred = pd.DataFrame(rng.normal(0, 1, (len(idx), 30)), index=idx, columns=cols)
    spy_ret = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)

    out = run_grid(_FakePanel(pred, ret), PanelConfig(min_names=5), spy_ret)

    assert len(out["per_config"]) == 4
    assert "verdict" in out
    assert isinstance(out["passed"], bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_panel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_panel'`

- [ ] **Step 3: Write `run_panel.py`**

```python
#!/usr/bin/env python3
"""
run_panel.py — Entry point for the cross-sectional panel backtest (Step 3 Phase A).

Research-only. Touches no live prediction path.

Usage:
    python run_panel.py --days 2500
    python run_panel.py --days 2500 --cost-bps 10   # cost SENSITIVITY, not tuning
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
from panel_backtester import PanelConfig, run_panel
from panel_data import build_panels
from panel_eval import CONFIG_GRID, evaluate_grid

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def run_grid(panel_data, base_cfg: PanelConfig, spy_ret: pd.Series) -> dict:
    """Run every (decile, rebalance_days) config and evaluate the grid."""
    results = {}
    for decile, rebal in CONFIG_GRID:
        cfg = dataclasses.replace(base_cfg, decile=decile, rebalance_days=rebal)
        results[(decile, rebal)] = run_panel(panel_data.pred, panel_data.ret, cfg)
        logger.info("ran config decile=%.2f rebalance_days=%d", decile, rebal)
    return evaluate_grid(results, spy_ret)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-sectional panel backtest")
    parser.add_argument("--days", type=int, default=2500)
    parser.add_argument("--db", default="data/trading_sim.db")
    parser.add_argument("--model", default="models/daily_predictor.pkl")
    parser.add_argument(
        "--cost-bps", type=float, default=None,
        help="Override cost_bps for SENSITIVITY reporting. This is a modelling "
             "assumption, not a parameter to tune until the gate passes.",
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

    base_cfg = PanelConfig(
        gross_exposure=app_cfg.panel.gross_exposure,
        cost_bps=args.cost_bps if args.cost_bps is not None else app_cfg.panel.cost_bps,
        borrow_bps_annual=app_cfg.panel.borrow_bps_annual,
        min_names=app_cfg.panel.min_names,
    )
    out = run_grid(panel_data, base_cfg, spy_ret)
    out["universe_size"] = len(panel_data.symbols)
    out["dropped"] = panel_data.dropped
    out["cost_bps"] = base_cfg.cost_bps

    Path("results").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("PANEL BACKTEST — STEP 3 PHASE A")
    print("=" * 72)
    for cfg_key, stats in out["per_config"].items():
        print(
            f"  {cfg_key:<12} ann_sharpe={stats['ann_sharpe']:+.2f}  "
            f"turnover={stats.get('mean_turnover', 0):.3f}  "
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


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_panel.py -v`
Expected: PASS

- [ ] **Step 5: Run the real backtest**

Run: `python run_panel.py --days 2500`
Expected: first run fetches ~157 symbols into the SQLite cache (slow, several minutes); prints the per-config table and the gate verdict; writes `results/panel_summary.json`.

If more than 10% of the universe fails to load, `build_panels` raises by design — fix the symbol list rather than lowering `MAX_LOAD_FAILURE_FRAC`.

- [ ] **Step 6: Run the cost sensitivity**

Run: `python run_panel.py --days 2500 --cost-bps 10 --output results/panel_summary_cost10.json`
Expected: a second summary. Record both. A daily-rebalanced decile book is turnover-heavy, so if the verdict flips between 5bps and 10bps, the gate is being decided by the cost assumption and the result must be reported as such.

- [ ] **Step 7: Write the results document**

Create `docs/superpowers/plans/2026-07-15-step3-results.md` following the structure of `docs/superpowers/plans/2026-07-14-step2-results.md`: a metrics table (config, ann_sharpe, turnover, DSR, beta, PBO), a verdict section, and an honest-caveat section. Record the outcome whether it passes or fails — **a null result is a shippable deliverable**, and it means Phase B does not get built.

Include: universe size actually used, symbols dropped, cost sensitivity at 5 vs 10 bps, and the survivorship-bias caveat from the spec's "Known limitations".

- [ ] **Step 8: Run the full suite**

Run: `pytest tests/ -q`
Expected: PASS — all existing tests plus ~26 new ones

- [ ] **Step 9: Commit**

```bash
git add run_panel.py tests/test_run_panel.py docs/superpowers/plans/2026-07-15-step3-results.md results/panel_summary.json results/panel_summary_cost10.json
git commit -m "feat: add run_panel CLI and record Step 3 Phase A results"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| `panel:` config block | Task 1 |
| Universe ~150 stocks, no ETFs | Task 1 (`test_panel_universe_is_stocks_only`) |
| Data loading reuses `_load_symbol` | Task 2 |
| `panel_data.py` — three aligned frames | Task 2 |
| `ret_panel` not `fwd_ret_1d` | Task 2 (`test_ret_panel_is_one_day_return_not_the_three_day_target`) |
| No forward-fill | Task 2 (`test_ragged_histories_align_outer_without_forward_fill`) |
| Fail loudly above 10% load failure | Task 2 (`test_raises_when_too_much_of_the_universe_fails_to_load`) |
| `panel_backtester.py` — ranking | Task 3 |
| Equal-weight, dollar-neutral, gross exposure | Task 3 |
| `min_names` floor | Task 3 |
| Engine loop, costs, borrow | Task 4 |
| Lag convention | Task 4 (`test_stale_prediction_earns_nothing_lookahead_guard`) |
| Perfect-foresight test | Task 4 |
| Random-prediction test | Task 4 |
| Neutrality test | Task 4 |
| Cost test | Task 4 |
| Alignment test | Task 2 |
| DSR ≥ 0.95 gate | Task 5 |
| |beta| < 0.1 precondition | Task 5 |
| PBO over 4-config grid | Task 5 |
| `cost_bps` out of the grid | Task 5 (`test_grid_has_four_configs_and_excludes_cost_bps`) |
| Turnover / cost drag reported | Task 4 diagnostics, Task 6 output |
| Results doc, null acceptable | Task 6 Step 7 |
| Touch no live path | Global Constraints; no task modifies live files |

No gaps.

**Type consistency:** `PanelData.pred/.ret/.close` (Task 2) are consumed by `run_panel(pred, ret, cfg)` (Task 4) and `run_grid` (Task 6) — consistent. `PanelResult.book_ret` (Task 4) is consumed by `evaluate_grid` and `book_beta` (Task 5) — consistent. `PanelConfig` field names match `PanelCfg` YAML keys (Task 1) where `run_panel.py` maps them (Task 6); `PanelConfig` adds `start_cash` and omits `universe`, which is intentional and handled explicitly in `main()`. `rank_to_weights` signature is identical in Task 3's definition and Task 4's call.

**Placeholder scan:** none. Every code step contains complete, runnable code.
