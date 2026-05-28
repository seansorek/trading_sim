# Trading Sim

Generates daily ML trading signals (BUY / SELL / HOLD) for a configurable symbol universe and delivers them to Discord. GitHub Actions runs the prediction job automatically every morning at 06:00 UTC.

## How it works

Three steps: train models locally, backtest them, then let the daily job run in CI forever.

```
Historical data  →  train_models.py  →  models/*.pkl
                                              ↓
                     simulate_multi.py  (backtest, optional)
                                              ↓
                  predict_next_day_lite.py  →  Discord
                  (runs via GitHub Actions every morning)
```

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

### 2. Backtest (optional)

```bash
python simulate_multi.py --symbols AAPL,MSFT,SPY --strategies daily_logistic,daily_xgboost
```

Runs simulated trading in parallel, writes equity curves and metrics to `results/`, and records runs in the SQLite DB.

| Flag | Description |
|---|---|
| `--start 2023-01-01 --end 2024-01-01` | Explicit date range |
| `--days 365` | Days back from today (default: 365) |
| `--workers 4` | Parallel processes (default: CPU count, capped at 4) |

Key outputs:
- `results/multi_summary.json` — all metrics in one file
- `results/<SYMBOL>_<STRATEGY>_metrics.json` — per-run detail
- `results/<SYMBOL>_<STRATEGY>_equity_curve.csv`

### 3. Generate predictions

```bash
export DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
python predict_next_day_lite.py
```

Loads the most recent model pickles, fetches the latest bar for each symbol listed in `config/default.yaml → prediction.symbols`, and posts signals to Discord. Without `DISCORD_WEBHOOK_URL`, predictions are still written to `tomorrow_trades.json` and stdout.

Test with a subset:

```bash
python predict_next_day_lite.py --symbols AAPL,SPY
```

## Configuration

All parameters live in [`config/default.yaml`](config/default.yaml). Common things to change:

**Symbols predicted each morning** (`prediction.symbols`): edit the list and push — no workflow YAML change needed.

**Backtest execution model**:
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
1. Train locally: `python train_models.py --symbols ...`
2. Commit `models/daily_logistic.pkl` and `models/daily_xgboost.pkl`
3. Push to `main` — the next Actions run picks up the new models automatically

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

## Project layout

```
config/
  default.yaml          # Single source of truth for all parameters
config.py               # Dataclass definitions; get_config() returns cached AppConfig
daily_features.py       # 25-feature vector; FEATURE_COLS is the canonical feature contract
train_models.py         # Train and save Logistic + XGBoost models
train_dqn.py            # Train the DQN agent
simulate_multi.py       # Parallel backtest runner
predict_next_day_lite.py  # Daily prediction + Discord webhook
simulation_pipeline.py  # Backtester engine, Sharpe/Sortino/drawdown, walk-forward
db.py                   # SQLite layer — bars, features, model registry, predictions
data_loader.py          # yfinance wrapper with DB caching
ml_strategies.py        # DailyLogisticStrategy and DailyXGBoostStrategy wrappers
dqn_agent.py            # PyTorch DQN network and agent
rl_env.py               # Gym-style environment for DQN training
models/                 # Committed model pickles (used by GitHub Actions)
results/                # Backtest output (gitignored)
tests/                  # pytest suite
```

## Dependencies

Full training stack: `requirements.txt`

```
numpy, pandas, scikit-learn, xgboost, scikit-optimize, torch, yfinance, pyyaml, requests
```

Prediction-only (used by GitHub Actions): `requirements-predict.txt` — same minus `scikit-optimize`.
