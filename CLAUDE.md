# Trading Sim

Generates daily ML trading signals (BUY/SELL/HOLD) for a configurable symbol universe and sends them to Discord. GitHub Actions runs the prediction job automatically every morning.

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

### 2. Backtest models

Runs simulated trading across symbols and strategies in parallel, writes equity curves and metrics to `results/`, and records runs in the DB.

```bash
python simulate_multi.py --symbols AAPL,MSFT,SPY --strategies daily_logistic,daily_xgboost
```

Optional flags:
- `--start 2023-01-01 --end 2024-01-01` — explicit date range (default: last `--days` days)
- `--days 365` — how many days back to run (default: 365)
- `--workers 4` — parallel processes (default: CPU count capped at 4)

Key outputs:
- `results/multi_summary.json` — all metrics in one file
- `results/<SYMBOL>_<STRATEGY>_metrics.json` — per-run detail
- `results/<SYMBOL>_<STRATEGY>_equity_curve.csv`

### 3. Generate daily predictions

Loads the most recent trained models, fetches the latest bar for each symbol, and sends predictions to Discord.

```bash
python predict_next_day_lite.py
```

Reads symbols from `config/default.yaml → prediction.symbols` by default. Override for a quick test:
```bash
python predict_next_day_lite.py --symbols AAPL,SPY
```

Requires `DISCORD_WEBHOOK_URL` env var for notifications. Without it, predictions are still written to `tomorrow_trades.json` and logged to stdout.

---

## Deployment (GitHub Actions)

`.github/workflows/simulation.yaml` runs step 3 daily at **06:00 UTC** on every push to `main` and on manual dispatch. It uses pre-committed model files from `models/`.

**To update the models deployed in production:**
1. Run step 1 locally with the symbols you want.
2. Commit the new `models/daily_logistic.pkl` and `models/daily_xgboost.pkl`.
3. Push to `main`. The next GitHub Actions run will use the new models.

**To change what symbols get predicted:**
Edit `prediction.symbols` in `config/default.yaml` and push. No workflow YAML edit needed.

**Required GitHub secret:** `DISCORD_WEBHOOK_URL` — set in repo Settings → Secrets and variables → Actions.

---

## Symbol universes

| Config key | Purpose | Where used |
|---|---|---|
| `symbols` | Training + backtest universe (keep focused, ~10–20 liquid names) | `train_models.py`, `simulate_multi.py` |
| `prediction.symbols` | Daily prediction + Discord output (can be wider) | `predict_next_day_lite.py`, GitHub Actions |

Both live in `config/default.yaml`.

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
- `test_feature_contract.py` — FEATURE_COLS consistency between training and prediction
- `test_predict.py` — model loading validation, signal generation, Discord formatting

---

## Key files

| File | Role |
|---|---|
| `config/default.yaml` | Single source of truth for all parameters |
| `config.py` | Dataclass definitions; `get_config()` returns a cached `AppConfig` |
| `daily_features.py` | Computes the 28-feature vector; `FEATURE_COLS` is the canonical feature contract |
| `train_models.py` | **Entry point:** train and save Logistic + XGBoost models |
| `train_dqn.py` | **Entry point:** train the DQN agent |
| `simulate_multi.py` | **Entry point:** parallel backtest runner |
| `predict_next_day_lite.py` | **Entry point:** daily prediction + Discord webhook |
| `simulation_pipeline.py` | Backtester engine, metrics (Sharpe/Sortino/drawdown), walk-forward |
| `db.py` | SQLite layer — bars, features, model registry, predictions, backtest runs |
| `data_loader.py` | yfinance wrapper with DB caching |
| `ml_strategies.py` | `DailyLogisticStrategy` and `DailyXGBoostStrategy` wrappers |
| `dqn_agent.py` | PyTorch DQN network and agent |
| `rl_env.py` | Gym-style environment for DQN training |
