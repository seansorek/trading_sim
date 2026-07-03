# Signal Integrity & Walk-Forward Validation — Design Spec

**Date:** 2026-07-02  
**Sub-project:** 1 of 4 (Professional Quality upgrade)  
**Status:** Approved

---

## 1. Problem

`daily_predictor` (Ridge regression) shows Spearman IC = +0.06 on a single held-out test window.
`signal_quantile` (0.7) and `threshold_window` (60) were chosen by intuition.
There is no way to know whether the edge is stable across time, and the live Discord signals have
no feedback loop — we never measure whether past predictions were correct.

---

## 2. Goals

1. A walk-forward harness that produces a **time series** of out-of-sample IC, not a single point estimate.
2. Parameter tuning (`signal_quantile`, `threshold_window`) via walk-forward IC rather than guesswork.
3. Realized IC tracking — after each day's predictions age `FWD_RET_HORIZON_DAYS` bars, compute IC vs actual returns and log it to DB + Discord.
4. Signal drift detection — warn if today's predicted-return distribution looks abnormal vs the trailing 30-prediction window.

**Out of scope for this sub-project:** new model families, feature engineering, code cleanup, Discord overhaul, documentation.

---

## 3. Architecture

### 3.1 `walk_forward.py` (new, ~200 lines)

Standalone module and CLI. Rolls a train/test window across historical data with a
`FWD_RET_HORIZON_DAYS`-bar purge/embargo gap (matching `train_predictor.py`).

Per fold:
- Trains a fresh Ridge model on the training window.
- Computes Spearman IC and directional accuracy on the test window.
- Runs `compute_predictor_signal` for each `(signal_quantile, threshold_window)` pair in the sweep grid.

Returns a `WalkForwardResult` dataframe with one row per fold.

Also runs a parameter sweep: for each `(signal_quantile, threshold_window)` pair, computes mean
IC across all folds. Best-performing pair (highest mean IC, tie-broken by directional accuracy)
gets written alongside the model pickle as `best_signal_quantile` and `best_threshold_window`.

When called with `--symbols` (multiple symbols), the sweep selects the pair with the highest
**median** IC across all symbols — a single pair is chosen for the shared model, not one per
symbol. `train_predictor.py` uses this pooled-median selection when it calls walk_forward.

**CLI:**
```
python walk_forward.py --symbol SPY --train 504 --test 63 --step 21
python walk_forward.py --symbols AAPL,MSFT,SPY --train 504 --test 63 --step 21  # multi-symbol
```

**Sweep grid (defaults):**
- `signal_quantile`: [0.60, 0.65, 0.70, 0.75, 0.80]
- `threshold_window`: [40, 60, 80, 100]

### 3.2 `train_predictor.py` (extend, ~50 lines)

After training, automatically runs walk-forward validation on each trained symbol and prints a
per-fold IC summary table. Saves `best_signal_quantile` and `best_threshold_window` into the
pickle so both `DailyPredictorStrategy` and `predict_next_day_lite.py` read them at load time.

### 3.3 `predict_next_day_lite.py` (extend, ~80 lines)

Two additions at the start of `main()`, before signal generation:

**Realized IC scorer:**
- Loads `predictions/history.jsonl`.
- Filters to entries at least `FWD_RET_HORIZON_DAYS + 1` days old.
- Fetches realized close prices for the scoring window.
- Computes Spearman IC and directional accuracy over the trailing 20 scoreable (date, symbol) pairs
  per model, pooled across all symbols (not per-symbol), sorted by date descending.
- Writes results to the new `ic_history` DB table.
- Includes a one-line IC summary in the Discord message:
  `Trailing-20 IC: +0.07 | dir-acc: 54% (daily_predictor)`

**Drift detector:**
- Computes mean, std, and skew of today's predicted returns (across all symbols).
- Compares against the trailing 30-calendar-day window of daily distribution means (one mean per
  day, not one per (date, symbol) pair — so it tracks day-level regime shifts, not symbol outliers).
- If the mean shift > 2σ **and** the absolute shift > 0.002 **and** this is the second consecutive
  day of such a shift → posts a yellow warning embed to Discord.

### 3.4 `db.py` (extend schema, ~20 lines)

New table `ic_history`:

```sql
CREATE TABLE IF NOT EXISTS ic_history (
    id                  INTEGER PRIMARY KEY,
    model               TEXT    NOT NULL,
    computed_at         TEXT    NOT NULL,
    lookback_n          INTEGER NOT NULL,
    ic                  REAL    NOT NULL,
    directional_accuracy REAL   NOT NULL,
    mean_pred           REAL,
    std_pred            REAL,
    UNIQUE(model, computed_at)
);
CREATE INDEX IF NOT EXISTS ic_history_model_date ON ic_history(model, computed_at);
```

---

## 4. Data Flow

```
train_predictor.py
  → walk_forward.py  (tune signal_quantile + threshold_window per symbol)
  → saves best params into models/daily_predictor.pkl

predict_next_day_lite.py  (daily, 06:00 UTC)
  → loads params from pickle (env var → pickle → hardcoded default)
  → scores trailing predictions → ic_history DB table
  → drift check → Discord warning embed if conditions met
  → generates today's signals → Discord
  → appends to predictions/history.jsonl
```

---

## 5. Error Handling

### Realized IC scorer
| Condition | Behavior |
|-----------|----------|
| `history.jsonl` has < 20 scoreable entries | Skip IC computation; log INFO; no DB write |
| Price fetch fails for a symbol | Exclude that symbol; score the rest |
| Realized return is near-zero (`abs < 1e-5`) | Exclude row (likely holiday/data gap) |
| All entries within `FWD_RET_HORIZON_DAYS` of today | Same as < 20 entries |

### Walk-forward
| Condition | Behavior |
|-----------|----------|
| Fewer total bars than one full fold | Raise `ValueError` with message |
| Fold produces < `threshold_window` predictions | Exclude fold from IC average; emit warning |
| All sweep combinations produce IC ≤ 0 | Keep hardcoded defaults (0.7, 60); emit warning |

### Drift detector
| Condition | Behavior |
|-----------|----------|
| Shift > 2σ but only one day | No warning (single-day guard) |
| `std_pred == 0` (all-HOLD predictor) | Skip drift check; no divide-by-zero |
| < 10 predictions in trailing window | Skip drift check |

### Backward compatibility
`DailyPredictorStrategy` and `predict_next_day_lite.py` read params with three-level priority:

```
env var (PREDICTOR_SIGNAL_QUANTILE / PREDICTOR_THRESHOLD_WINDOW)
  → pickle keys (best_signal_quantile / best_threshold_window)
    → hardcoded default (0.7 / 60)
```

Old pickles without the new keys fall back cleanly to hardcoded defaults.

---

## 6. Testing

### `tests/test_walk_forward.py` (new)
- Synthetic sine+noise data produces ≥1 fold with IC in a sensible range
- Parameter sweep returns correct shape, no NaN IC values
- Purge/embargo gap enforced — no label overlap between train and test windows
- Fewer bars than one fold → `ValueError`

### `tests/test_ic_tracking.py` (new)
- Mocked `history.jsonl` + mocked price fetch → hand-calculated IC matches output
- < 20 entries → returns `None`, no DB write
- Entries within `FWD_RET_HORIZON_DAYS` of today → excluded
- Near-zero return rows → excluded (holiday filtering)

### `tests/test_drift_detection.py` (new)
- Normal distribution input → no warning
- Single-day shift > 2σ → no warning
- Two-day shift > 2σ with absolute shift > 0.002 → warning triggered
- `std_pred == 0` → no exception

### `tests/test_predictor.py` (extend)
- Pickle with `best_signal_quantile`/`best_threshold_window` → strategy uses those values
- Old pickle without those keys → falls back to hardcoded defaults without error

**Coverage target:** ≥85% line coverage on `walk_forward.py` and the new predict extensions.  
**DB testing pattern:** in-memory SQLite (same as `test_db.py`), no mocking.

---

## 7. Files Changed

| File | Change |
|------|--------|
| `walk_forward.py` | **New** — walk-forward harness + param sweep CLI |
| `train_predictor.py` | **Extend** — run walk-forward after training, save best params to pickle |
| `predict_next_day_lite.py` | **Extend** — realized IC scorer + drift detector |
| `db.py` | **Extend** — `ic_history` table + `upsert_ic` method |
| `tests/test_walk_forward.py` | **New** |
| `tests/test_ic_tracking.py` | **New** |
| `tests/test_drift_detection.py` | **New** |
| `tests/test_predictor.py` | **Extend** — two new backward-compat cases |
