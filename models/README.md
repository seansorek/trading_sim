# Pre-trained ML Models

This directory contains pre-trained models committed to the repo so GitHub Actions can load them without retraining on every run.

## Files

- `daily_logistic.pkl` — Trained LogisticRegression model + StandardScaler + feature contract
- `daily_logistic_v<N>.pkl` — Versioned snapshots (canonical path is always `daily_logistic.pkl`)
- `daily_xgboost.pkl` — Trained XGBoost classifier + StandardScaler + feature contract
- `daily_xgboost_v<N>.pkl` — Versioned snapshots
- `dqn_agent.pt` — PyTorch DQN agent (optional; trained separately via `train_dqn.py`)

Each pickle contains: `model`, `scaler`, `feature_contract`, `confidence_threshold`, `label_map`, `trained_at`, `train_symbols`, and accuracy metrics.

## Model details

### Daily Logistic (`daily_logistic.pkl`)
- **Algorithm**: `sklearn.linear_model.LogisticRegression` (multinomial, balanced class weights)
- **Features**: 28 daily features defined in `daily_features.FEATURE_COLS`
- **Labels**: `SELL=0, HOLD=1, BUY=2` (forward 1-day return thresholds: ±0.2%)
- **Normalization**: `StandardScaler` (fit on training data, stored in pickle)
- **Confidence threshold**: Default 0.55; stored in pickle, read by `predict_next_day_lite.py`

### Daily XGBoost (`daily_xgboost.pkl`)
- **Algorithm**: `xgboost.XGBClassifier` (`multi:softprob`, 3-class)
- **Features**: Same 28 features as logistic
- **Labels**: Same as logistic — `SELL=0, HOLD=1, BUY=2`
- **Hyperparameters**: Set in `config/default.yaml → strategies.xgboost`
- **Class weighting**: Sample weights computed per run to counteract imbalance

### DQN (`dqn_agent.pt`)
- **Algorithm**: PyTorch DQN with target network and experience replay
- **Actions**: `HOLD=0, LONG=1, SHORT=2` (mapped to `HOLD/BUY/SELL` in predictions)
- **State**: Rolling window of last 20 days × 28 features (flattened)
- **Trained via**: `train_dqn.py`

## Updating models

```bash
# Retrain locally (uses symbols + hyperparameters from config/default.yaml)
python train_models.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 1000

# Commit the updated canonical pickles
git add models/daily_logistic.pkl models/daily_xgboost.pkl
git commit -m "Retrain models"
git push
```

The next GitHub Actions run will pick up the new files automatically.
