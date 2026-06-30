# Pre-trained ML Models

This directory contains pre-trained models committed to the repo so GitHub Actions can load them without retraining on every run.

## Files

- `daily_logistic.pkl` — Trained LogisticRegression model + StandardScaler + feature contract
- `daily_logistic_v<N>.pkl` — Versioned snapshots (canonical path is always `daily_logistic.pkl`)
- `daily_xgboost.pkl` — Trained XGBoost classifier + StandardScaler + feature contract
- `daily_xgboost_v<N>.pkl` — Versioned snapshots
- `daily_hybrid.pkl` / `daily_hybrid_v<N>.pkl` — XGBoost-transformer hybrid, trained via `train_hybrid.py` (see [hybrid_model.py](../hybrid_model.py))
- `daily_predictor.pkl` / `daily_predictor_v<N>.pkl` — Regression forecaster, trained via `train_predictor.py`. Pairs with `ml_strategies.DailyPredictorStrategy` as the decision layer — see "Prediction vs. strategy" below.
- `dqn_agent.pt` — PyTorch DQN agent (optional; trained separately via `train_dqn.py`)

Each classifier pickle contains: `model`, `scaler`, `feature_contract`, `confidence_threshold`, `label_map`, `trained_at`, `train_symbols`, and accuracy metrics. `daily_predictor.pkl` has the same shape minus the classification-specific fields — see its own section below.

## Model details

### Daily Logistic (`daily_logistic.pkl`)
- **Algorithm**: `sklearn.linear_model.LogisticRegression` (multinomial)
- **Features**: 25 daily features defined in `daily_features.FEATURE_COLS`
- **Labels**: `SELL=0, HOLD=1, BUY=2` — 3-day forward return, thresholded by volatility-scaled bands (`--vol-mult`)
- **Normalization**: `StandardScaler` (fit on training data, stored in pickle)
- **Confidence threshold**: Default 0.55; stored in pickle, read by `predict_next_day_lite.py`
- **Class weighting**: `--logistic-class-weight none` (default) or `balanced`. Default `class_weight="balanced"` tanks test accuracy to ~35% by fighting the natural HOLD-majority prior — see "Accuracy" below.

### Daily XGBoost (`daily_xgboost.pkl`)
- **Algorithm**: `xgboost.XGBClassifier` (`multi:softprob`, 3-class)
- **Features**: Same 25 features as logistic
- **Labels**: Same as logistic — `SELL=0, HOLD=1, BUY=2`
- **Hyperparameters**: Set in `config/default.yaml → strategies.xgboost`, overridable via CLI flags
- **Class weighting**: `--xgb-class-weight none` (default), `sqrt`, or `inverse`. `none` performs best for the same reason as logistic.

### Accuracy — read before trusting the numbers

Both `daily_logistic` and `daily_xgboost` are trained and evaluated with a purged/embargoed
train/test split (a `FWD_RET_HORIZON_DAYS`-row gap at the split boundary, since the label is
a 3-day forward return — see `tests/test_data_leakage.py`). With leakage correctly excluded,
test accuracy for both models tracks the trivial "always predict HOLD" majority-class baseline
within ~2 percentage points across every vol_mult / class-weight / lookback-window /
training-history-length configuration tested (current canonical models are trained on ~6.8
years / 1718 bars per symbol — see "Prediction vs. strategy" below for why more data didn't
change this). There is no consistent, genuine lift over that baseline with the current
25-feature single-bar technical-indicator set — `daily_hybrid`'s transformer component shows
the same pattern. Raising `--vol-mult` pushes the nominal accuracy number up, but only by
shrinking the SELL/BUY classes until the model approaches a constant HOLD predictor; that's not
a meaningful accuracy claim. `tests/test_model_accuracy.py` and
`tests/test_hybrid.py::test_hybrid_artifact_contract` pin a 0.50 floor for this reason, not a
higher target reached by gaming class imbalance.

### Prediction vs. strategy (`daily_predictor.pkl` + `DailyPredictorStrategy`)

`daily_logistic`/`daily_xgboost`/`daily_hybrid` all directly classify the *discretized* action
(SELL/HOLD/BUY), which bakes a decision threshold (`vol_mult`) into the training target itself —
conflating "what will the price do" (a forecasting problem) with "what should I do about it" (a
policy/risk decision). Two things were tested to explain the lack of signal above:

1. **Not enough data?** Tested directly: a data-loader caching bug (`check_cache_freshness` only
   checked recency, never coverage of the requested `start` date — fixed in `data_loader.py`,
   see `tests/test_train_cache.py::test_short_but_fresh_cache_triggers_refetch_for_more_history`)
   meant earlier "more data" experiments were silently reusing the same ~700-bar cache. With the
   fix, retraining on the genuine ~1718-bar (6.8yr) history still landed exactly on the
   majority-class baseline (XGBoost test_acc 0.612 vs. baseline 0.612). **More data alone does
   not produce genuine classification lift on this feature set.**

2. **Bad architecture (action-classification vs. prediction+strategy split)?** Tested via
   `train_predictor.py`: a Ridge regression trained on the *continuous* `fwd_ret_1d` target
   (same purged split, same 25 features) recovers a small but real out-of-sample signal —
   **Spearman IC = +0.06, R² = +0.012** — that none of the 3-class classifiers could detect at
   all. An XGBoost regressor on the same target collapses to a near-constant predictor (IC ≈ 0),
   consistent with the classifiers' XGBoost variant also underperforming Logistic throughout this
   investigation — tree ensembles appear to over-regularize away this weak a signal on this
   feature set; the simpler linear model picks it up.

`daily_predictor.pkl` is that Ridge model. `ml_strategies.DailyPredictorStrategy` is the
decoupled decision layer: it converts the continuous forecast into SELL/HOLD/BUY via a *causal
rolling quantile* of `|predicted return|` (not a fixed vol-scaled band — Ridge's shrunk
predictions are ~6x smaller in magnitude than raw returns, so a fixed band never fires) —
trade only the most extreme `1 - signal_quantile` fraction of the trailing `threshold_window`
bars' predictions. Both `signal_quantile` and `threshold_window` are strategy parameters,
independently tunable from the prediction model.

Backtested head-to-head against `daily_logistic`/`daily_xgboost` over the same 10-symbol,
700-day window (`simulate_multi.py --strategies daily_logistic,daily_xgboost,daily_predictor`),
`daily_predictor` was the only one of the three with a **positive average return (+0.18%) and
positive average Sharpe (+0.27)** — both classifiers were solidly negative. This is a single,
untuned backtest window, not a validated edge: Sharpe 0.27 is still weak in absolute terms, the
default `signal_quantile=0.7`/`threshold_window=60` were not optimized, and there's been no
walk-forward or out-of-sample-of-out-of-sample validation yet. Treat this as a promising lead
that the prediction/strategy split surfaces real signal the classification framing was
discarding, not as a finished, deployable edge.

**Deployment status:** `daily_predictor` is wired into `predict_next_day_lite.py` and the live
Discord pipeline (`prediction.models` in `config/default.yaml`), alongside `daily_logistic` and
`daily_xgboost`. Live inference uses the exact same decision function as backtesting
(`ml_strategies.compute_predictor_signal`) so the two can't silently diverge. Its Discord
"confidence" field is not a calibrated probability like the classifiers' — it's the percentile
rank of today's |predicted return| within its trailing window (see
`predict_next_day_lite._regressor_confidence`). Given the caveats above, treat its live signals
with the same skepticism as the backtest: a promising lead under active validation, not a
proven edge.

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
