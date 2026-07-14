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

| daily_v6 (features only) | 0.0380 | 0.206 | 0.848 | 29-feature v6 (dropped williams_r/macd_hist/vol_z_20; added ret_21d, adx_14, vol_regime, rel_volume, hl_ratio, turnover_z, gap; rolling z-score on 18 unbounded features). signal_quantile=0.75, threshold_window=40. Per-symbol IC: AAPL=0.0304, MSFT=0.0217, GOOGL=0.1026, AMZN=-0.0113, NVDA=-0.0103, META=0.0456, TSLA=0.0805, SPY=0.0840, QQQ=0.0266, IWM=0.0901. Gate: KEPT (IC 0.0380 > 0.0266 AND PBO 0.206 ≤ 0.564). |
| +amihud | 0.0526 | 0.163 | 0.783 | 30-feature daily_v6 with Amihud illiquidity added. amihud_illiq = \|ret_1d\| / dollar_volume (price impact proxy, z-scored). signal_quantile=0.80, threshold_window=60. Per-symbol IC: AAPL=0.0496, MSFT=0.0247, GOOGL=0.1158, AMZN=-0.0075, NVDA=-0.0108, META=0.0556, TSLA=0.0834, SPY=0.0729, QQQ=0.0327, IWM=0.0972. Gate: KEPT (IC 0.0526 > 0.0380 AND PBO 0.163 ≤ 0.256). +38% IC lift, -21% PBO improvement, -8% DSR cost (acceptable orthogonal trade). Retrained all 4 models (daily_logistic_v2, daily_xgboost_v2, daily_predictor_v2, daily_hybrid_v2). All 331 tests pass. |

| +vix (REVERTED) | 0.0423 | 0.037 | 0.627 | 32-feature: added vix_z (rolling z-score of VIX level) and vix_chg_5d (rolling z-score of 5-day VIX pct change). Threaded through train_models, train_predictor, train_hybrid, walk_forward, predict_next_day_lite. Index normalization required (VIX from yfinance has 05:00 UTC vs DB-cached symbols at 04:00 UTC). Per-symbol IC: AAPL=0.0098, MSFT=0.0290, GOOGL=0.0879, AMZN=-0.0227, NVDA=-0.0277, META=0.0396, TSLA=0.0959, SPY=0.0896, QQQ=0.0449, IWM=0.0854. Per-symbol DSR: AAPL=0.965, MSFT=0.503, SPY=0.627, QQQ=0.967, NVDA=0.337. Gate: REVERTED (IC 0.0423 < baseline 0.0526; VIX noise dominated signal on AMZN/NVDA; PBO improved to 0.037 but IC gate failed). |

## Next Steps

Rows below will track improvements from orthogonal feature engineering, preprocessing, and model refinement:
