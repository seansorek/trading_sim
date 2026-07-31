# Trading Sim

Ranks a ~159-name US equity universe daily and publishes a **beta-neutral
long/short target book** to Discord, alongside per-symbol BUY/SELL/HOLD signals
for a smaller watchlist. GitHub Actions runs the job every morning.

**This is a portfolio system, not a per-symbol signal system.** That distinction
is load-bearing, and it is the repo's own measurement that forces it:

- The cross-sectional ranker has a small but **sign-stable** edge — IC positive
  in 5/5 yearly walk-forward folds, net Sharpe positive in 4/5.
- The per-symbol timing path **loses to buy-and-hold** (alpha −8.94%, IR −0.62),
  and the classifiers sit at the majority-class baseline.

So the book is the product and the per-symbol signals are a sidecar kept for
their IC/drift telemetry. See `models/README.md` for both sets of numbers,
including the caveats — a *small* edge measured on one vendor's unaudited data
with survivorship bias is not a licence to trade it.

---

## Three-step pipeline

### 1. Train models

Fetches historical data, engineers features, and fits Logistic Regression + XGBoost classifiers. Saves pickles to `models/` and registers them in the SQLite DB.

```bash
python train_models.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 1000
```

Optional flags:
- `--models logistic,xgboost` — which models to train (default: both)
- `--optimize` — run Bayesian hyperparameter search for XGBoost (slow, requires `scikit-optimize`)
- `--confidence 0.55` — confidence threshold baked into the saved pickle

DQN is trained separately:
```bash
python train_dqn.py --symbol SPY --days 500 --episodes 30
```

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
python train_predictor.py --days 2500 --model enet
```

`--symbols` defaults to a 10-name list for backwards compatibility; the deployed model is
trained on the full 159-name `symbols` universe from `config/default.yaml`. Widening it is what
made per-year out-of-sample IC stable (see `models/README.md` → "Universe width"), so pass the
config universe when retraining for production:

```bash
python train_predictor.py --symbols "$(python -c 'from config import get_config; print(",".join(get_config().symbols))')" --days 2500 --model enet
```

`--train-end YYYY-MM-DD` pins the train/test split to a calendar date instead of the first 80%
of history. Use it to hold out a fixed window — a fraction split slides whenever `--days` moves
and therefore holds nothing out.

### 2. Backtest the portfolio

The primary backtest. Ranks the whole `panel.universe` cross-section per date,
builds a beta-neutral decile long/short book, and reports DSR, realized beta,
turnover, cost drag and PBO over `panel_eval.CONFIG_GRID`.

```bash
python run_panel.py --days 2500
```

Optional flags:
- `--cost-bps 10` — cost **sensitivity reporting**, not a knob to tune until the gate passes
- `--no-sector-neutral` / `--conviction` — A/B comparisons against the base book

Output: `results/panel_summary.json`.

**Per-symbol backtest (secondary).** `simulate_multi.py` runs the single-name
`Backtester` per (symbol, strategy). Keep it for per-name diagnostics; do not
read its Sharpe as the system's performance — a long-biased per-name timing
strategy is the thing the portfolio exists to replace.

```bash
python simulate_multi.py --symbols AAPL,MSFT,SPY --strategies daily_logistic,daily_xgboost,daily_predictor
```

- `--start 2023-01-01 --end 2024-01-01` — explicit date range (default: last `--days` days)
- `--days 365` / `--workers 4`
- Outputs: `results/multi_summary.json`, `results/<SYMBOL>_<STRATEGY>_{metrics.json,equity_curve.csv}`

### 3. Publish the daily book

Ranks `panel.universe`, builds today's target book, and posts it to Discord
ahead of the per-symbol watchlist signals.

```bash
python predict_next_day_lite.py
```

- `--symbols AAPL,SPY` — override the per-symbol **watchlist** only. The ranked
  universe always comes from `panel.universe`; a hand-picked list is not a
  cross-section.
- `--book ""` — skip the portfolio entirely (per-symbol signals only)
- `--rebalance-days 1` — force a fresh book instead of holding

Outputs `tomorrow_trades.json` (`{date, portfolio, predictions}`), appends
`predictions/portfolio.jsonl` and `predictions/history.jsonl`, and logs to
stdout. `DISCORD_WEBHOOK_URL` enables notifications.

#### The book holds; it does not re-rank daily

`panel.rebalance_days` (10) is enforced live, not just in the backtest. On most
days the job re-publishes the stored book marked `HOLD`; only when the window
elapses does it re-rank and mark `REBALANCE`. This is not a nicety — at
`rebalance_days: 1` the same book turns over ~0.85/day and cost removes 1.4–2.7
Sharpe every year.

That makes `predictions/portfolio.jsonl` **state, not a log**. The CI job commits
it back to the repo; delete it and the next run rebalances from scratch.

---

## Deployment (GitHub Actions)

`.github/workflows/simulation.yaml` runs step 3 daily at **06:00 UTC** on every push to `main` and on manual dispatch. It uses pre-committed model files from `models/`.

**To update the models deployed in production:**
1. Run step 1 locally with the symbols you want.
2. Commit the new `models/daily_logistic.pkl` and `models/daily_xgboost.pkl`.
3. Push to `main`. The next GitHub Actions run will use the new models.

**To change what the portfolio ranks:**
Edit `panel.sectors` in `config/default.yaml` and push — membership and sector
identity come from the same block, so they cannot drift apart. Stocks only.

**To change the per-symbol watchlist:**
Edit `prediction.symbols` in `config/default.yaml` and push. No workflow YAML edit needed.

**To change the book's shape or cadence:**
Edit `panel.decile` / `panel.rebalance_days`. These configure the *live* book
only — `run_panel.py` overrides both from `panel_eval.CONFIG_GRID`. Re-run
`run_panel.py` before changing them: the grid is what the DSR is deflated
against, and picking a cell by eye outside it is unmeasured.

**To change which models predict (add/remove a model from the live pipeline):**
Edit `prediction.models` in `config/default.yaml` and push. `predict_next_day_lite.py` reads this
list at startup — a model removed from the list is simply not loaded or predicted; a model added
to the list whose pickle is missing is logged and skipped, not a hard failure. No workflow YAML
edit needed.

**Required GitHub secret:** `DISCORD_WEBHOOK_URL` — set in repo Settings → Secrets and variables → Actions.

---

## Symbol universes

Three lists in `config/default.yaml`, with three different jobs. They are not
interchangeable.

| Config key | Size | Purpose | Where used |
|---|---|---|---|
| `panel.sectors` → `panel.universe` | 156 | **The portfolio cross-section.** Stocks only. Flattened from `sectors` at load time, so a symbol cannot be ranked without a sector and the two can never drift | `run_panel.py`, `portfolio.py`, `predict_next_day_lite.py` |
| `symbols` | 159 | Training + per-symbol backtest — `panel.universe` plus SPY/QQQ/IWM | `train_models.py`, `train_predictor.py`, `simulate_multi.py` |
| `prediction.symbols` | 30 | Per-symbol watchlist for the BUY/SELL/HOLD sidecar. May contain ETFs | `predict_next_day_lite.py`, GitHub Actions |

**ETFs must never enter `panel.universe`.** SPY, QQQ, IWM and the XL\* funds are
baskets of the same names being ranked — shorting a fund that holds your long is
not a cross-sectional bet, and a diversified basket ranks structurally mid-pack.
SPY still loads, for the `ret_*_vs_spy` features and the beta estimate, but is
never ranked or held. `tests/test_config.py::test_panel_universe_is_stocks_only`
enforces this.

The daily job predicts the *union* of `panel.universe` and
`prediction.symbols` (~174 names, ~15 overlap) and fetches each symbol once.

---

## Running tests

```bash
pytest tests/ -v
```

Test coverage:
- `test_features.py` — feature engineering, no NaN/inf, label discretization
- `test_backtester.py` — position sizing, commission, equity curve
- `test_data_loader.py` — yfinance fetching, data standardization
- `test_db.py` — SQLite schema, inserts, queries
- `test_feature_contract.py` — FEATURE_COLS (30 features, `daily_v6`) consistency between training and prediction
- `test_predict.py` — model loading validation, signal generation, Discord formatting
- `test_data_leakage.py` — purged/embargo-gap regression tests for the train/test split
- `test_predictor.py` — regression prediction model + `DailyPredictorStrategy` decision layer
- `test_config.py` — prediction.models / prediction.symbols / panel YAML loading
- `test_portfolio.py` — target book construction, beta-neutral leg sizing, sector neutralization, rebalance cadence, and that the live book equals `rank_to_weights`
- `test_panel_backtester.py` / `test_panel_data.py` / `test_panel_eval.py` — the portfolio backtest engine, panel alignment, and the DSR/beta/PBO gate

---

## Key files

| File | Role |
|---|---|
| `config/default.yaml` | Single source of truth for all parameters |
| `config.py` | Dataclass definitions; `get_config()` returns a cached `AppConfig` |
| `daily_features.py` | Computes the 30-feature vector (`daily_v6`); `FEATURE_COLS` is the canonical feature contract |
| `train_models.py` | **Entry point:** train and save Logistic + XGBoost models |
| `train_predictor.py` | **Entry point:** train the Ridge return-prediction model (experimental prediction/strategy split) |
| `train_dqn.py` | **Entry point:** train the DQN agent |
| `run_panel.py` | **Entry point:** portfolio backtest + DSR/beta/PBO gate |
| `simulate_multi.py` | **Entry point:** per-symbol backtest runner (secondary) |
| `predict_next_day_lite.py` | **Entry point:** daily book + per-symbol signals + Discord webhook |
| `portfolio.py` | Live target book: ranks one cross-section, holds it between rebalances. Delegates weighting to `panel_backtester` so live and backtest cannot diverge |
| `panel_data.py` | Builds aligned date×symbol prediction/return/close/beta panels |
| `panel_backtester.py` | Portfolio engine — `rank_to_weights` and `sector_neutralize` are the shared decision layer for both backtest and live |
| `panel_eval.py` | Portfolio gate: deflated Sharpe, realized beta, PBO over `CONFIG_GRID` |
| `simulation_pipeline.py` | Single-symbol backtester, metrics (Sharpe/Sortino/drawdown), walk-forward |
| `db.py` | SQLite layer — bars, features, model registry, predictions, backtest runs |
| `data_loader.py` | yfinance wrapper with DB caching |
| `ml_strategies.py` | `DailyLogisticStrategy`, `DailyXGBoostStrategy`, `DailyPredictorStrategy` wrappers; `compute_predictor_signal` is the shared decision-layer function used by both backtest and live prediction |
| `dqn_agent.py` | PyTorch DQN network and agent |
| `rl_env.py` | Gym-style environment for DQN training |
