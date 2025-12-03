# Pre-trained ML Models

This directory contains pre-trained machine learning models for the trading simulator.

## Files

- `ordinal_logistic.pkl` - Trained LogisticRegression model with StandardScaler
- `xgboost.pkl` - Trained XGBoost classifier
- `metadata.json` - Model training metadata (date, symbols, features, accuracy)

## Usage

### Training Models

To train models on historical data:

```bash
# Train on default symbols (SPY, QQQ, AAPL, MSFT, etc.) with 60 days of data
python train_models.py

# Train on specific symbols with custom timeframe
python train_models.py --symbols AAPL,MSFT,NVDA,TSLA --days 90

# Train on different interval
python train_models.py --interval 1m --days 14
```

### Using Pre-trained Models

Models are automatically loaded when running simulations:

```bash
# Pre-trained models used by default (no retraining)
python simulate_multi.py --strategies ordinal_logistic,xgboost

# Force retraining from scratch (disable pre-trained models)
# Requires code modification: set use_pretrained=False in strategy init
```

### Model Deployment

1. Train models locally with sufficient data:
   ```bash
   python train_models.py --days 60
   ```

2. Commit trained models to git:
   ```bash
   git add models/
   git commit -m "Update pre-trained ML models"
   git push
   ```

3. GitHub Actions will load these models instead of retraining

## Model Details

### Ordinal Logistic Regression
- **Algorithm**: sklearn.linear_model.LogisticRegression
- **Features**: ret_1m, ma_spread, vol_10, rsi_14, vol_z, momentum_5, momentum_20, vp_ratio, vol_regime, price_position
- **Classes**: 3-class (BUY=1, HOLD=0, SELL=-1)
- **Normalization**: StandardScaler
- **Label Threshold**: ±0.5% for 5m bars

### XGBoost
- **Algorithm**: xgboost.XGBClassifier
- **Features**: Same as ordinal logistic
- **Classes**: 3-class (BUY=1, HOLD=0, SELL=-1)
- **Hyperparameters**: max_depth=3, lr=0.05, n_estimators=50
- **Confidence Threshold**: 50% minimum probability to trade
- **Label Threshold**: ±0.5% for 5m bars

## Retraining Schedule

Retrain models periodically as market conditions change:
- **Weekly**: For active trading with frequent market shifts
- **Monthly**: For stable market conditions
- **After major events**: Market crashes, regime changes, etc.

Check model performance in `results/` to determine if retraining is needed.
