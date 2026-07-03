# Architecture

Trading Sim is a three-phase pipeline: **train → backtest → predict**. The diagram below shows
the data-flow for the daily live prediction path (Phase 3) and how training artifacts feed into it.
The SQLite database is a side-channel cache — it is not authoritative for predictions (that role
belongs to `predictions/history.jsonl`, which is committed to the repo and survives ephemeral CI runners).

```mermaid
flowchart LR
    subgraph External
        YF[/"yfinance API\n(OHLCV)"/]
    end

    subgraph DataLayer["Data Layer"]
        DL["data_loader.py\nload_yfinance()"]
        DF["daily_features.py\nmake_daily_features()\n→ 25 features (daily_v3)"]
    end

    subgraph Training["Training (run locally, commit artifacts)"]
        TM["train_models.py\nLogisticRegression\nXGBoostClassifier"]
        TP["train_predictor.py\nRidge regression"]
        TH["train_hybrid.py\nXGBoost + transformer\n(not deployed)"]
        TD["train_dqn.py\nPyTorch DQN\n(not deployed)"]
    end

    subgraph Artifacts["models/ (committed to repo)"]
        ML[daily_logistic.pkl]
        MX[daily_xgboost.pkl]
        MP[daily_predictor.pkl]
        MH["daily_hybrid.pkl (not in pipeline)"]
        MD["dqn_agent.pt (not in pipeline)"]
    end

    subgraph Prediction["predict_next_day_lite.py\n(GitHub Actions — daily 06:00 UTC)"]
        LM["load_models()\nreads prediction.models\nfrom config/default.yaml"]
        PS["predict_symbol() × N\nclassifier: softmax → SELL/HOLD/BUY\nregressor: rolling-quantile → SELL/HOLD/BUY"]
        IC["signal_monitor.py\nTrailing-20 Spearman IC\nDrift detection (2σ / 2 consecutive days)"]
    end

    subgraph Output
        DI[/"Discord\nwebhook"/]
        TJ["tomorrow_trades.json\n(GitHub Actions artifact)"]
        HJ["predictions/history.jsonl\n(append-only, committed to repo)"]
    end

    subgraph DB["data/trading_sim.db (SQLite — ephemeral on CI)"]
        BD[bar_data\nOHLCV cache]
        MR[model_registry\nversion + artifact path]
        DP[daily_predictions\nper-symbol per-model]
        ICT[ic_history\ntrailing IC per model]
    end

    YF -->|HTTP fetch| DL
    DL -->|pd.DataFrame| DF
    DL <-->|cache read/write| BD

    DF -->|feature matrix| TM
    DF -->|feature matrix| TP
    DF -->|feature matrix| TH
    DF -->|feature matrix| TD

    TM -->|saves pickle| ML
    TM -->|saves pickle| MX
    TM -->|registers| MR
    TP -->|saves pickle| MP
    TP -->|registers| MR
    TH -->|saves pickle| MH
    TD -->|saves .pt| MD

    ML -->|loaded by| LM
    MX -->|loaded by| LM
    MP -->|loaded by| LM
    MR -->|resolves artifact path| LM

    LM --> PS
    DL -->|live features| DF
    DF -->|live feature matrix| PS

    PS -->|predictions| IC
    HJ -->|history| IC
    IC -->|IC metrics| ICT
    IC -->|IC + drift to Discord| DI

    PS -->|upsert| DP
    PS --> DI
    PS --> TJ
    PS --> HJ
```

## Component Descriptions

| Component | File | Role |
|---|---|---|
| **Data loader** | `data_loader.py` | `load_yfinance()` fetches OHLCV from yfinance; caches bars in SQLite `bar_data` table to avoid redundant fetches |
| **Feature engineering** | `daily_features.py` | `make_daily_features()` computes the 25-feature `daily_v3` vector. `FEATURE_COLS` is the canonical feature contract — changing it requires retraining all models and bumping `FEATURE_SET_NAME` |
| **Logistic/XGBoost training** | `train_models.py` | Trains 3-class classifiers on discretized SELL/HOLD/BUY targets; saves `daily_logistic.pkl` and `daily_xgboost.pkl` |
| **Predictor training** | `train_predictor.py` | Trains Ridge regression on continuous forward return target; saves `daily_predictor.pkl`; runs walk-forward param sweep via `walk_forward.sweep_params` |
| **Daily prediction** | `predict_next_day_lite.py` | Entry point for GitHub Actions; loads models configured in `prediction.models`, calls `predict_symbol()` per symbol, scores trailing IC, sends to Discord |
| **Decision layer (predictor)** | `ml_strategies.compute_predictor_signal` | Single source of truth for the rolling-quantile threshold used by both `DailyPredictorStrategy` (backtest) and live prediction — the two can never silently diverge |
| **IC scoring** | `signal_monitor.py` | Computes trailing Spearman IC against realized returns from `predictions/history.jsonl`; checks for distribution drift |
| **Database** | `db.py` | SQLite layer; schema is idempotent (`CREATE TABLE IF NOT EXISTS`) so a fresh DB is safe. Ephemeral on CI — `predictions/history.jsonl` is the durable prediction record |
| **Config** | `config/default.yaml` | Single source of truth for symbol universes, model list, hyperparameters, thresholds. `prediction.models` controls which models are loaded at runtime — no code change needed to add/remove a model |

## Key Invariants

- **`FEATURE_COLS` is the feature contract.** All pickles store it under `feature_contract`; `predict_next_day_lite._load_pkl` raises `RuntimeError` on mismatch. Never change `FEATURE_COLS` without bumping `FEATURE_SET_NAME` and retraining all models.
- **`compute_predictor_signal` is the single decision layer.** It is called identically in backtest (`DailyPredictorStrategy.signal`) and live prediction (`predict_next_day_lite._predict_regressor_signal`). Do not duplicate it.
- **`predictions/history.jsonl` is append-only.** It is the only durable record of what the live models predicted (the SQLite `daily_predictions` table is ephemeral on CI runners). Committed to the repo after every scheduled run by `simulation.yaml`.
