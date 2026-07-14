# Step-2 Results: Baseline and Improvements

## Evaluation Metrics Table

| stage | median walk-forward IC | PBO | median DSR | notes |
|-------|------------------------|-----|-----------|-------|
| daily_v3 baseline | 0.0266 | 0.514 | 0.800 | 25-feature baseline, signal_quantile=0.80, threshold_window=40. Per-symbol IC: AAPL=0.0421, MSFT=-0.0037, GOOGL=0.0196, AMZN=0.0261, NVDA=0.0120, META=0.0899, TSLA=0.0023, SPY=0.0836, QQQ=0.0271, IWM=0.1044. Per-symbol DSR: AAPL=0.943, MSFT=0.290, SPY=0.539, QQQ=0.800, NVDA=0.859. |

## Baseline Summary

- **Walk-forward IC**: Computed over 10 symbols (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, SPY, QQQ, IWM) with 2500 days of history
- **PBO (Parameter Efficiency)**: 0.514 indicates moderate parameter overfitting across the hyperparameter sweep (folds=265, configs=20, combinations=12870)
- **DSR (Deflated Sharpe Ratio)**: Median of 0.800 across 5-symbol subset (AAPL, MSFT, SPY, QQQ, NVDA) suggests modest statistical significance after multiple testing adjustment
- **Baseline model**: Daily logistic regression / XGBoost / predictor with 25 engineered features

## Next Steps

Rows below will track improvements from orthogonal feature engineering, preprocessing, and model refinement:
