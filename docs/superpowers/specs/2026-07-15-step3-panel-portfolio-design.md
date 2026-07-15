# Step 3 (Phase A) — Cross-sectional panel backtester

**Date:** 2026-07-15
**Status:** Approved for planning
**Base branch:** `feat/step3-panel-backtester`, branched off `main` at `996d33d` (#107)

## Context

This is "sequence step 3" from the pipeline-improvement analysis: originally
**#1 (cross-sectional + neutralize)** + **#4 (widen universe)** + **#2
(vol-target + conviction sizing)**. Four brainstorming decisions reshaped it.

1. **Phase A / Phase B split.** Vol-targeting improves the Sharpe of an edge
   that already exists; it cannot create one. Building it before establishing
   whether cross-sectional alpha exists risks tuning a dead signal — and adds
   knobs (target vol, conviction curve) that inflate PBO before there is a
   baseline to inflate against. **This spec covers Phase A only:** the panel
   engine, the wide universe, and an equal-weight decile long-short book.
   Phase B (#2) gets its own spec, gated on Phase A's result.

2. **Research instrument, not a deployable system.** A dollar-neutral
   long-short book over ~150 names rebalanced daily is not tradeable by hand
   from a Discord alert, and Phase A does not pretend otherwise. The question
   Phase A answers is narrow: *does breadth plus neutralization produce real
   risk-adjusted alpha?* `models/README.md` currently answers "not a proven
   deployable edge" for the per-symbol path. Optimizing execution for an
   unproven edge inverts the order of work.

3. **Ranking neutralizes the market for free.** Cross-sectional ranking is
   location-invariant: adding a constant to every prediction on date `t`
   leaves the ranking unchanged. Market-wide moves in the forecast therefore
   neutralize themselves, and the explicit market-demeaning proposed in #1 is
   unnecessary for the *decision*. Sector-neutrality still requires real
   demeaning and a residual target still helps *training*, so both become
   gated increments rather than core.

4. **Equal weights remove the current PBO source.** `compute_predictor_signal`
   does not appear in the panel path. Phase A drops `signal_quantile` and
   `threshold_window` — the two parameters whose selection instability drives
   PBO 0.228 — and replaces twenty configs with two knobs: decile fraction and
   rebalance frequency.

## Honest baseline (from #106/#107, `models/README.md`)

The per-symbol `daily_predictor` (Ridge, `daily_v6`) after the look-ahead fix:

| Metric | Value |
|---|---|
| Median OOS walk-forward IC | **+0.0407** (up from 0.0266 on `daily_v3`) |
| PBO | **0.228** (down from 0.514) — moderate-overfit zone |
| Median DSR | 0.776 across 5 symbols — below the 0.95 bar |
| Backtest Sharpe | +0.16 (10-symbol, 365-day) |
| **Alpha vs buy-and-hold** | **−8.94%** |
| **Information ratio** | **−0.62** |

Step 2 improved the signal and cut overfitting, both real. The strategy still
loses to buy-and-hold. That is the structural problem Phase A tests: a
long-biased per-name timing strategy underperforming its own benchmark is the
disease cross-sectional long-short treats.

Two facts support the cross-sectional thesis. Per-symbol IC disperses widely
(GOOGL +0.131, IWM +0.093, TSLA +0.084 against AMZN −0.007, NVDA +0.008) —
dispersion is raw material for ranking but poison for per-name timing. And PBO
drifted 0.067 → 0.228 when a re-run selected different `(q, w)` params, which
is the parameter instability showing itself.

## Success criteria & gate

**Blocking gate:** deflated Sharpe **≥ 0.95** on net-of-cost book returns
(`deflated_sharpe.deflated_sharpe`, per-period Sharpe — see the unit bug fixed
in #106), with realized **|beta| < 0.1** against SPY as a precondition.

Beta is a *correctness* check, not a performance check. A dollar-neutral book
must have near-zero market beta by construction. Beta outside the band means
the neutralization failed and the Sharpe number describes something other than
the intended book — diagnose before reading performance.

**Why DSR and not alpha.** "Alpha vs SPY buy-and-hold," the current headline
metric, is the wrong gate for a neutral book: zero beta makes underperforming a
long benchmark expected rather than informative. `compute_metrics`'
`benchmark_close` parameter takes the traded symbol's close and has no meaning
for a panel; the panel path passes no benchmark and judges the book on its own
risk-adjusted return.

**Reported, non-blocking:** Sharpe (gross and net), turnover, cost drag,
borrow drag, PBO over the config grid, realized beta, and the count of days
the cross-section fell below `min_names`.

**A null result is an acceptable, documented outcome.** If the equal-weight
decile book shows no significant alpha, Phase A ships the finding and the
validated engine, and Phase B does not get built. Step 2 set this precedent
by reverting VIX when it failed the gate.

## Scope

**In scope**

- A `panel:` block in `config/default.yaml` holding the universe and engine
  parameters.
- `panel_data.py` — builds aligned prediction/return/close panels.
- `panel_backtester.py` — the weight-based engine.
- Panel evaluation: DSR, beta, turnover, cost drag, PBO over the config grid.
- Tests per the Testing section.
- A results document recording the gate outcome, including a null.

**Out of scope**

- **The live path.** Phase A changes no `predict_next_day_lite.py`, no
  `prediction.models`, no workflow YAML, no Discord payload. The panel is a
  research instrument and touches nothing that trades.
- Vol-targeting and conviction sizing — Phase B, gated on this result.
- Retraining on the wide universe, residual target, sector-neutralization —
  increments #1–3 below.
- Market-impact modelling, capacity analysis, point-in-time constituents.

## Architecture

Five pieces. One new engine. Nothing existing changes.

### Universe → `config/default.yaml`

A new `panel:` block holds ~150 liquid US symbols. The universe is data, not
code, and `config/default.yaml` is already the single source of truth
(CLAUDE.md). No `universe.py` module.

**No sector map.** Sector-neutrality is increment #3, so building the map now
is speculative work.

**Selection criteria:** large-cap US **stocks only**, chosen for dollar-volume
and history depth (≥ 2500 calendar days), spread across sectors so no single
sector dominates a decile.

**Index and sector ETFs stay out of the tradeable cross-section.** SPY, QQQ,
IWM, and the XL* sector funds are baskets of the same names the panel ranks.
Ranking AAPL against XLK means potentially shorting a fund that holds the long
— the two are not independent bets, and the "cross-section" stops being one.
ETFs would also rank persistently mid-pack, since a diversified basket's
idiosyncratic forecast is structurally damped relative to a single name.

SPY still loads, for two jobs it keeps: the `ret_1d_vs_spy` / `ret_5d_vs_spy`
features in `make_daily_features`, and the beta regression in the gate. It is
never ranked and never held.

This narrows the existing `prediction.symbols` set (which contains SPY, QQQ,
IWM, GLD, USO, and six XL* funds) rather than extending it.

### Data loading → reuse

`train_models._load_symbol` already performs DB-cached bar loading, and
`walk_forward` already imports it. The panel uses it unchanged. The first run
populates the SQLite cache for ~150 symbols (slow once); later runs read the
cache. No prefetch script and no third cached-loader — the repo has two
already (`train_models._load_symbol`, `predict_next_day_lite._load_bars_cached`)
and does not need another.

### `panel_data.py`

Builds three DataFrames indexed by date, columns by symbol:

- `pred_panel` — Ridge forecasts from the existing `models/daily_predictor.pkl`
  via `predictors.ridge` and `predictors.base._scale`, preserving the frozen
  train/serve preprocessing contract.
- `ret_panel` — **1-day simple returns**, `close[t] → close[t+1]`.
- `close_panel` — closes, for the beta regression and diagnostics.

Symbols join outer on dates and carry a validity mask.

**`ret_panel` must not use `daily_features.fwd_ret_1d`.** Despite its name,
that column holds a `FWD_RET_HORIZON_DAYS`-bar (3-day) cumulative return — the
docstring in `daily_features.py` warns that consumers paying it as per-bar PnL
must divide by the horizon, and `rl_env` already had to. Paying a 3-day
cumulative return once per daily bar would triple the book's apparent return by
double-counting overlapping windows. Build `ret_panel` from
`close.pct_change().shift(-1)` directly.

**Horizon mismatch is expected and acceptable.** The model forecasts a 3-day
cumulative return while the book earns a 1-day return, so daily rebalancing
re-enters an overlapping 3-day view each day. That is a rolling ensemble of
overlapping forecasts, not an error — provided the book pays the 1-day return.
`rebalance_days` ∈ {1, 3} enters the PBO config grid so the horizon-matched
variant gets measured rather than assumed.

### `panel_backtester.py`

The engine, roughly 100 lines. On each rebalance date it takes that date's
prediction cross-section, drops NaNs, ranks, longs the top decile and shorts
the bottom decile at equal weight, and scales to gross exposure. Weights hold
until the next rebalance.

```
book_ret[t] = Σ w[t] · ret[t → t+1] − turnover_cost[t] − borrow_cost[t]
```

Costs use a flat one-way rate on turnover notional:

```
turnover_cost[t] = Σ |w[t] − w[t-1]| · cost_bps / 1e4
borrow_cost[t]   = Σ |w[t] where w < 0| · borrow_bps_annual / 1e4 / 252
```

The equity curve feeds the existing `compute_metrics`.

**The panel does not reuse `ExecutionConfig`.** Its cost model is per-share
(`commission_per_share`) and its spread proxy reads `high`/`low` off a single
symbol's bar — neither maps onto a weight vector. A flat `cost_bps` on turnover
is the standard, honest model at this altitude, and it keeps the net-vs-gross
gap explicit and easy to stress. Sensitivity to `cost_bps` is reported, since a
daily-rebalanced decile book is turnover-heavy and cost assumptions can decide
the gate on their own.

**The engine deliberately omits** `stop_loss_pct`, `take_profit_pct`,
`daily_loss_limit_pct`, and the forced-exit cooldown. These are per-name,
path-dependent rules from `Backtester`, and a stop-loss that exits one leg
breaks the dollar- and beta-neutrality the book depends on.

### Why a second engine

`Backtester` is single-symbol to its foundations: it iterates `df.iterrows()`,
holds a scalar `position` and a scalar `avg_entry_price`, and applies
per-name exit rules. Retrofitting it to hold a symbol-keyed book would be major
surgery on tested code, and its exit rules actively fight neutrality. The two
engines answer different questions: `Backtester` simulates one tradeable name;
`panel_backtester` is a portfolio research instrument.

**The honest cost:** a weight-based backtest is optimistic. It models no share
granularity, assumes fills at the close, and ignores per-name market impact.
For "does this alpha exist," that is the correct trade. It is not a
deployability simulation, and no result from it should be read as one.

## Data flow

```
config.panel.universe (~150 symbols)
  → train_models._load_symbol            (DB-cached bars)
  → make_daily_features(df, spy_df)      (daily_v6, 30 features)
  → predictors.ridge + _scale            (existing daily_predictor.pkl)
  → pred_panel[date, symbol]
  → rank cross-sectionally per date
  → equal-weight top/bottom decile → weights[date, symbol]
  → panel_backtester: book_ret[t] = Σ w[t]·ret[t→t+1] − costs
  → compute_metrics + deflated_sharpe + beta check
```

## The lag convention

Weights derived from predictions on `close[t]` apply to the return
`close[t] → close[t+1]`. Never `t-1 → t`.

This is the bug class #106 just fixed in `DailyPredictorStrategy`, where a
missing `.shift(1)` let the backtest decide on `close[t]` and fill at
`close[t]`, inflating Sharpe from +0.16 to +0.27. A panel engine makes the same
error easier to commit, because the lag lives in a join between two frames
rather than in a visible `.shift()`. It gets a dedicated test.

## Error handling

- **A symbol missing on date `t` leaves that date's cross-section.** Never
  forward-fill prices: commit `3255f87` fixed exactly that leak in
  `_standardize`.
- Insufficient history → drop the symbol from the panel; log and count it.
- Cross-section below `min_names` → hold no position that day; log and count.
- More than 10% of the universe failing to load → raise. Quietly backtesting
  40 names while the config says 150 produces a number that describes nothing.

## Configuration

```yaml
panel:
  universe: [~150 liquid symbols]
  decile: 0.1               # top/bottom fraction of the cross-section
  rebalance_days: 1         # daily
  gross_exposure: 1.0       # 0.5 long + 0.5 short; net exposure 0
  cost_bps: 5.0             # one-way cost on turnover notional
  borrow_bps_annual: 50.0   # flat GC-style rate on short notional
  min_names: 20             # below this, hold no position
```

`gross_exposure` denotes total gross as a fraction of equity, split evenly
between legs: `1.0` means 50% long and 50% short, netting to zero.

`min_names` is a data-sparsity floor, not a target. At the full ~150-name
universe, `decile: 0.1` fills each leg with ~15 names; `min_names: 20` only
binds early in the history or after mass load failures, where a 2-name leg
would produce noise rather than a cross-section.

**Config grid for PBO:** `decile` ∈ {0.1, 0.2} × `rebalance_days` ∈ {1, 3} —
four configs. Phase A reports PBO across this grid, against the twenty configs
the per-symbol path sweeps today.

## Testing

`tests/test_panel_backtester.py`:

- **Perfect foresight.** Feed tomorrow's actual returns as predictions; Sharpe
  must be large. This proves the engine *can* detect signal when signal exists
  — if it fails, the wiring is broken and every other result is meaningless.
- **Random predictions.** Shuffled predictions must produce Sharpe ≈ 0 and
  beta ≈ 0. This catches look-ahead: random predictions that make money mean
  the lag is wrong.
- **Neutrality.** Long and short notional match; book beta ≈ 0.
- **Costs.** Zero turnover charges zero cost; a known turnover charges a
  hand-computed cost. Borrow accrues on short notional only.
- **Alignment.** Ragged histories align correctly, and a symbol missing on
  date `t` leaves that day's ranking rather than being filled.

`tests/test_panel_data.py`: panel construction, NaN handling, validity mask.

Both reuse the existing `conftest.py` fixtures.

## Increments (deferred, each gated separately)

Test each alone against the gate, as step 2 tested and reverted VIX:

1. **Retrain on the wide universe.** Phase A core reuses the current
   `daily_predictor.pkl`, trained on 10 mega-caps and applied to ~150 names.
   Per-symbol rolling z-scored features are cross-sectionally comparable, so it
   should transfer. Retraining and going cross-sectional simultaneously would
   confound the engine test with a model change.
2. **Residual target.** Demean `fwd_ret_1d` cross-sectionally per date in
   `train_predictor.prepare_data`, so the model stops spending capacity on the
   market component.
3. **Sector-neutralization.** Demean predictions within sector buckets. Needs
   the sector map that Phase A omits.

## Known limitations

- **Survivorship bias.** The universe lists names liquid *today*, so it
  excludes delisted and removed names and the results run optimistic. Fixing
  this needs point-in-time index membership, which yfinance does not provide
  and which is its own project. Documented, not solved.
- **Optimistic execution.** No share granularity, close fills, no market
  impact (see "Why a second engine").
- **Flat borrow.** 50bps annual approximates general-collateral rates for
  liquid large caps. Hard-to-borrow names cost far more; the panel does not
  model them.
- **One data vendor.** yfinance, unaudited for corporate actions.
