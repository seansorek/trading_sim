# Trading Sim

Ranks a 156-name US equity cross-section every morning and publishes a
**beta-neutral long/short target book** to Discord, plus per-symbol
BUY/SELL/HOLD signals for a smaller watchlist. GitHub Actions runs the job at
06:00 UTC.

This is a **portfolio** system. The repo's own measurements are what force that:
the cross-sectional ranker's IC is positive in 5/5 yearly walk-forward folds,
while the per-symbol timing path loses to buy-and-hold (alpha −8.94%, IR −0.62).
So the book is the product; the per-symbol signals are a sidecar retained for
their IC and drift telemetry. The honest numbers, caveats included, are in
[`models/README.md`](models/README.md).

> **Not investment advice, and not a deployable trading system.** The edge is
> small, measured on one unaudited data vendor, over a survivorship-biased
> universe, with a weight-based backtest that assumes close fills and no market
> impact. Read `models/README.md` before drawing any conclusion from a signal.

## How it works

Train the ranker locally, backtest the book, then let the daily job run in CI.

```
Historical data  →  train_predictor.py  →  models/daily_predictor.pkl
                                                   ↓
                              run_panel.py   (portfolio backtest + DSR gate)
                                                   ↓
                        predict_next_day_lite.py  →  target book  →  Discord
                        (GitHub Actions, every morning)
```

The book is built by `portfolio.py`, which delegates the actual weighting to
`panel_backtester.rank_to_weights` — the same function `run_panel.py` measures.
One implementation, so the published book cannot drift from the tested one.

## Quick start

**Prerequisites:** Python 3.10+, a Discord webhook URL (optional but recommended).

```bash
git clone <this-repo>
cd trading_sim
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 1. Train models

```bash
python train_models.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 1000
```

Fetches 1000 days of daily bars via yfinance, engineers 25 features, fits Logistic Regression and XGBoost classifiers with a temporal train/test split, and writes pickles to `models/`.

Optional flags:

| Flag | Description |
|---|---|
| `--models logistic,xgboost` | Which models to train (default: both) |
| `--optimize` | Bayesian hyperparameter search for XGBoost (slow; requires `scikit-optimize`) |
| `--confidence 0.55` | Minimum confidence threshold baked into the saved pickle |
| `--days N` | Days of history to fetch (default: 1000) |

DQN is trained separately:

```bash
python train_dqn.py --symbol SPY --days 500 --episodes 30
```

### 2. Backtest the portfolio

```bash
python run_panel.py --days 2500
```

Ranks the whole cross-section per date, builds the beta-neutral decile
long/short book, and reports the gate: deflated Sharpe, realized beta, turnover,
cost drag and PBO across `panel_eval.CONFIG_GRID`. Writes
`results/panel_summary.json`.

| Flag | Description |
|---|---|
| `--cost-bps 10` | Cost **sensitivity reporting**. Not a knob to tune until the gate passes |
| `--no-sector-neutral` | Rank raw predictions instead of sector-demeaned ones (A/B) |
| `--conviction` | Weight within each leg by distance from the cross-sectional centre (A/B) |

**Per-symbol backtest (secondary).** Keep for per-name diagnostics; its Sharpe is
not the system's performance.

```bash
python simulate_multi.py --symbols AAPL,MSFT,SPY --strategies daily_logistic,daily_xgboost
```

| Flag | Description |
|---|---|
| `--start 2023-01-01 --end 2024-01-01` | Explicit date range |
| `--days 365` | Days back from today (default: 365) |
| `--workers 4` | Parallel processes (default: CPU count, capped at 4) |

Outputs: `results/multi_summary.json`, `results/<SYMBOL>_<STRATEGY>_metrics.json`,
`results/<SYMBOL>_<STRATEGY>_equity_curve.csv`.

### 3. Publish the daily book

```bash
export DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
python predict_next_day_lite.py
```

Ranks `panel.universe` into today's target book, then computes per-symbol signals
for the `prediction.symbols` watchlist. Posts the book first. Without the webhook,
everything still lands in `tomorrow_trades.json` and stdout.

| Flag | Description |
|---|---|
| `--symbols AAPL,SPY` | Override the per-symbol **watchlist** only — the ranked universe always comes from `panel.universe` |
| `--book ""` | Skip the portfolio; per-symbol signals only |
| `--rebalance-days 1` | Force a fresh book instead of holding the stored one |

**The book holds between rebalances.** `panel.rebalance_days` (10) is enforced
live, not just in the backtest: most mornings the job re-publishes the stored
book marked `HOLD` and only re-ranks when the window elapses. At
`rebalance_days: 1` the same book turns over ~0.85/day and cost removes 1.4–2.7
Sharpe a year. That makes `predictions/portfolio.jsonl` **state, not a log** —
CI commits it back to the repo, and deleting it forces a rebalance from scratch.

**Zero beta is not zero directional risk.** The two legs are sized so their beta
exposures cancel, which means net *notional* is not zero. Measured per date at
decile 0.2 it ran mean −0.22 and reached −0.64; past |0.5| the book is flagged in
the output. It is flagged, never clamped — clamping would make the published book
differ from the backtested one.

## Configuration

All parameters live in [`config/default.yaml`](config/default.yaml). Common things to change:

**What the portfolio ranks** (`panel.sectors`): stocks only. Membership and sector
identity come from the same block so they cannot drift; the list is flattened into
`panel.universe` at load time.

**The per-symbol watchlist** (`prediction.symbols`): edit and push — no workflow
YAML change needed. ETFs are fine here because they never enter a ranking.

**Book shape and cadence**:
```yaml
panel:
  decile: 0.2               # top/bottom fraction of the cross-section
  rebalance_days: 10        # business days held between re-rankings
  gross_exposure: 1.0       # total gross as a fraction of equity
  min_names: 20             # below this, hold nothing
```

**Backtest execution model** (single-symbol path only):
```yaml
execution:
  start_cash: 100000.0
  commission_per_share: 0.00005
  max_position_pct: 0.05      # 5% of equity per trade
  stop_loss_pct: 0.05
  take_profit_pct: 0.10
  holding_period_days: 5
```

**Strategy thresholds**:
```yaml
strategies:
  logistic:
    confidence_threshold: 0.55
  xgboost:
    confidence_threshold: 0.55
```

## Deployment (GitHub Actions)

[`.github/workflows/simulation.yaml`](.github/workflows/simulation.yaml) runs step 3 daily at 06:00 UTC, on every push to `main`, and on manual dispatch.

**To update the deployed models:**
1. Train locally: `python train_predictor.py --symbols ... ` (the ranker) and/or
   `python train_models.py --symbols ...` (the watchlist classifiers)
2. Commit the updated pickles in `models/`
3. Push to `main` — the next Actions run picks up the new models automatically

Retraining the ranker changes the ranking, so the held book becomes stale. Run
`run_panel.py` against the new pickle before deploying it, and expect the next
scheduled run to rebalance.

**Required GitHub secret:** `DISCORD_WEBHOOK_URL` — set in repo Settings → Secrets and variables → Actions.

## Feature engineering

25 normalized features are computed in [`daily_features.py`](daily_features.py) from daily OHLCV data. All features are dimensionless so they're comparable across symbols at different price levels.

| Category | Features |
|---|---|
| Returns | `ret_1d`, `ret_5d`, `ret_10d` |
| Volatility | `vol_20d`, `atr_normalized`, `bb_width`, `bb_position` |
| Trend | `ma_spread_10_20`, `ma_spread_20_50`, `price_vs_sma20`, `price_vs_sma50` |
| Momentum | `macd`, `macd_signal`, `macd_hist`, `rsi_14`, `stoch_k`, `stoch_d`, `williams_r`, `roc_12` |
| Volume | `vol_z_20`, `vpt_normalized`, `ad_normalized`, `obv_normalized` |
| Market-relative | `ret_1d_vs_spy`, `ret_5d_vs_spy` |

Labels are discretized from 3-day forward returns: BUY (>+0.2%), SELL (<-0.2%), HOLD otherwise.

`FEATURE_COLS` in `daily_features.py` is the strict contract between training and prediction — both sides index features by this list, never by DataFrame column order.

## Running tests

```bash
pytest tests/ -v
```

| Test file | What it covers |
|---|---|
| `test_features.py` | Feature engineering, no NaN/inf, label discretization |
| `test_backtester.py` | Position sizing, commission, equity curve |
| `test_data_loader.py` | yfinance fetching, data standardization |
| `test_db.py` | SQLite schema, inserts, queries |
| `test_feature_contract.py` | `FEATURE_COLS` consistency between training and prediction |
| `test_predict.py` | Model loading, signal generation, Discord formatting |
| `test_portfolio.py` | Target book construction, beta-neutral leg sizing, sector neutralization, rebalance cadence, live-equals-backtest |
| `test_panel_backtester.py` | Portfolio engine — perfect-foresight, random-signal, neutrality, costs, alignment |
| `test_panel_data.py` | Panel construction, NaN handling, the lag convention |
| `test_panel_eval.py` | DSR / beta / PBO gate |

## Project layout

```
config/
  default.yaml          # Single source of truth for all parameters
config.py               # Dataclass definitions; get_config() returns cached AppConfig
daily_features.py       # 30-feature vector (daily_v6); FEATURE_COLS is the canonical contract

# --- portfolio (primary) ---
portfolio.py            # Live target book: rank one cross-section, hold it between rebalances
panel_data.py           # Aligned date x symbol prediction/return/close/beta panels
panel_backtester.py     # Portfolio engine; rank_to_weights + sector_neutralize are shared
                        #   by the backtest AND the live book
panel_eval.py           # Gate: deflated Sharpe, realized beta, PBO over CONFIG_GRID
run_panel.py            # Entry point: portfolio backtest
predict_next_day_lite.py  # Entry point: daily book + watchlist signals + Discord

# --- per-symbol (secondary) ---
train_models.py         # Train and save Logistic + XGBoost classifiers
train_predictor.py      # Train the ElasticNet return predictor the ranking uses
train_dqn.py            # Train the DQN agent
simulate_multi.py       # Per-symbol backtest runner
simulation_pipeline.py  # Single-symbol backtester, Sharpe/Sortino/drawdown, walk-forward
ml_strategies.py        # DailyLogistic / DailyXGBoost / DailyPredictor strategy wrappers
dqn_agent.py            # PyTorch DQN network and agent
rl_env.py               # Gym-style environment for DQN training

db.py                   # SQLite layer — bars, features, model registry, predictions
data_loader.py          # yfinance wrapper with DB caching
models/                 # Committed model pickles (used by GitHub Actions)
predictions/            # history.jsonl (telemetry) + portfolio.jsonl (held book — state!)
results/                # Backtest output (gitignored)
tests/                  # pytest suite
```

## Dependencies

Full training stack: `requirements.txt`

```
numpy, pandas, scikit-learn, xgboost, scikit-optimize, torch, yfinance, pyyaml, requests
```

Prediction-only (used by GitHub Actions): `requirements-predict.txt` — same minus `scikit-optimize`.
