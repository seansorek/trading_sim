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

### 2. Backtest models

Runs simulated trading across symbols and strategies in parallel, writes equity curves and metrics to `results/`, and records runs in the DB.

```bash
python simulate_multi.py --symbols AAPL,MSFT,SPY --strategies daily_logistic,daily_xgboost,daily_predictor
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

**To change which models predict (add/remove a model from the live pipeline):**
Edit `prediction.models` in `config/default.yaml` and push. `predict_next_day_lite.py` reads this
list at startup — a model removed from the list is simply not loaded or predicted; a model added
to the list whose pickle is missing is logged and skipped, not a hard failure. No workflow YAML
edit needed.

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
- `test_feature_contract.py` — FEATURE_COLS (29 features, `daily_v6`) consistency between training and prediction
- `test_predict.py` — model loading validation, signal generation, Discord formatting
- `test_data_leakage.py` — purged/embargo-gap regression tests for the train/test split
- `test_predictor.py` — regression prediction model + `DailyPredictorStrategy` decision layer
- `test_config.py` — prediction.models / prediction.symbols YAML loading

---

## Key files

| File | Role |
|---|---|
| `config/default.yaml` | Single source of truth for all parameters |
| `config.py` | Dataclass definitions; `get_config()` returns a cached `AppConfig` |
| `daily_features.py` | Computes the 29-feature vector (`daily_v6`); `FEATURE_COLS` is the canonical feature contract |
| `train_models.py` | **Entry point:** train and save Logistic + XGBoost models |
| `train_predictor.py` | **Entry point:** train the Ridge return-prediction model (experimental prediction/strategy split) |
| `train_dqn.py` | **Entry point:** train the DQN agent |
| `simulate_multi.py` | **Entry point:** parallel backtest runner |
| `predict_next_day_lite.py` | **Entry point:** daily prediction + Discord webhook |
| `simulation_pipeline.py` | Backtester engine, metrics (Sharpe/Sortino/drawdown), walk-forward |
| `db.py` | SQLite layer — bars, features, model registry, predictions, backtest runs |
| `data_loader.py` | yfinance wrapper with DB caching |
| `ml_strategies.py` | `DailyLogisticStrategy`, `DailyXGBoostStrategy`, `DailyPredictorStrategy` wrappers; `compute_predictor_signal` is the shared decision-layer function used by both backtest and live prediction |
| `dqn_agent.py` | PyTorch DQN network and agent |
| `rl_env.py` | Gym-style environment for DQN training |
