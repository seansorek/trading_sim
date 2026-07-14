# Pre-trained ML Models

This directory contains pre-trained models committed to the repo so GitHub Actions can load them without retraining on every run.

## Model Cards

One entry per model that is trained or deployed in this project. "Live" means the model
key appears in `prediction.models` in `config/default.yaml` and is loaded by
`predict_next_day_lite.py` on every GitHub Actions run.

---

### daily_logistic — Live

| Field | Value |
|---|---|
| **Status** | Live — in `prediction.models` |
| **Algorithm** | `sklearn.linear_model.LogisticRegression` (multinomial, 3-class) |
| **Task** | Classify next-day direction: `SELL=0 / HOLD=1 / BUY=2` |
| **Feature set** | `daily_v5` — 29 normalized daily technical features (see `daily_features.FEATURE_COLS`) |
| **Label construction** | 3-day cumulative forward return (`FWD_RET_HORIZON_DAYS=3`) thresholded by volatility-scaled bands (`--vol-mult`) |
| **Training window** | ~1 000 calendar days (≈ 670 trading bars) per symbol, 10 symbols pooled |
| **Train/test split** | Purged + embargoed: 3-bar gap at split boundary to prevent label leakage |
| **Normalization** | `RobustScaler` fit on training data, stored in pickle alongside the model |
| **Confidence threshold** | 0.55 (softmax probability of the predicted class); non-HOLD signals below this threshold are converted to HOLD |
| **OOS test accuracy** | At or within ~2 pp of the majority-class (HOLD) baseline — no consistent genuine lift (see "Accuracy" section below for full investigation) |
| **OOS directional accuracy** | Not meaningfully above 50% |
| **Backtest Sharpe** | Negative across tested windows (10-symbol, 700-day) |
| **Known limitations** | Single-bar technical features do not produce genuine classification lift once leakage is correctly excluded. Hyperparameter sweeps (Bayesian via `--optimize`, extensive manual search) confirmed no configuration escapes the baseline. |

**Honest caveat:** The accuracy number printed during training (`~60%`) reflects the HOLD-majority class prior, not predictive skill. Raising `--vol-mult` inflates the number by shrinking SELL/BUY classes toward a constant HOLD predictor — that is not a meaningful accuracy improvement. See the "Accuracy" section for the full investigation.

---

### daily_xgboost — Live

| Field | Value |
|---|---|
| **Status** | Live — in `prediction.models` |
| **Algorithm** | `xgboost.XGBClassifier` (`multi:softprob`, 3-class) |
| **Task** | Same 3-class direction classification as `daily_logistic` |
| **Feature set** | `daily_v5` — same 29 features |
| **Label construction** | Same as `daily_logistic` |
| **Training window** | Same as `daily_logistic` |
| **Train/test split** | Same purged + embargoed split |
| **Normalization** | `RobustScaler` (same pattern) |
| **Confidence threshold** | 0.55 (softmax probability) |
| **OOS test accuracy** | At majority-class baseline (~61%); no consistent lift over `daily_logistic` |
| **Backtest Sharpe** | Negative across tested windows |
| **Hyperparameters** | `n_estimators=200`, `max_depth=4`, `lr=0.03`, `subsample=0.85`, `colsample_bytree=0.85`, `min_child_weight=2`, `gamma=1.0` (from `config/default.yaml:strategies.xgboost`) |
| **Known limitations** | Same as `daily_logistic`. XGBoost appears to over-regularize away the weak signal that Ridge can detect on this feature set — XGBoost regressor on the same continuous target also collapses to a near-constant predictor (IC ≈ 0). |

---

### daily_predictor — Live (Preferred)

| Field | Value |
|---|---|
| **Status** | Live — in `prediction.models`. Most promising signal of the three live models. |
| **Algorithm** | `sklearn.linear_model.Ridge` (regression) |
| **Task** | Forecast continuous 3-day forward return (`fwd_ret_1d` in code — the label horizon is `FWD_RET_HORIZON_DAYS=3` trading bars) |
| **Feature set** | `daily_v5` — same 29 features |
| **Label construction** | Continuous cumulative forward return (not discretized) |
| **Training window** | ~2 500 calendar days (~1 718 trading bars) per symbol, 10 symbols pooled (use `--days 2500` for retraining) |
| **Train/test split** | Same purged + embargoed split (3-bar gap) |
| **Normalization** | `RobustScaler` + `±5-std clip` preprocessing (`train_models._preprocess`) |
| **OOS Spearman IC** | +0.06 (positive, statistically meaningful given n; classifiers show IC ≈ 0) |
| **OOS R²** | +0.012 (small but real; confirms the regression picks up a signal the classifiers discarded) |
| **Decision layer** | `DailyPredictorStrategy` (`ml_strategies.py`) — causal rolling-quantile threshold on `|predicted return|`: trade only when today's forecast magnitude exceeds the top `1 - signal_quantile` fraction of the trailing `threshold_window` bars' predictions. Default: `signal_quantile=0.7`, `threshold_window=60` |
| **Backtest Sharpe** | +0.16 avg (10-symbol, 365-day window, post-fix 2026-07-13; pre-fix was +0.27 on 700-day window) — the only positive Sharpe among the three live models; see PBO/DSR below |
| **Average return** | +0.07% avg per round-trip (10-symbol, 365-day window, post-fix 2026-07-13; pre-fix was +0.18%) |
| **Alpha vs. benchmark** | -8.94% avg; avg information ratio -0.62 (strategy underperforms buy-and-hold on most symbols over the 365-day post-fix window) |
| **PBO** | 0.513 (CPCV, 16 folds, 20 configs) — high-overfitting zone: IS-selected params expected to underperform OOS in >50% of paths |
| **Median DSR** | 0.800 across AAPL, MSFT, SPY, QQQ, NVDA (2500-day eval, corrected 2026-07-14) — per-symbol DSR range 0.290–0.943; prior 0.000 was a unit-mismatch bug (annualized variance used in per-period deflation) |
| **Walk-forward** | Param sweep run at training time via `walk_forward.sweep_params`; best `(signal_quantile, threshold_window)` pair stored in pickle |
| **Live confidence** | Percentile rank of today's `|predicted return|` within the trailing `threshold_window` — NOT a calibrated probability |
| **Known limitations** | Sharpe 0.16 (post-fix) is weak in absolute terms. PBO=0.513 (high-overfitting zone) remains a concern. Corrected DSR median=0.800 suggests genuine per-symbol signal after unit-mismatch fix, but PBO indicates the selected (q,w) pair is still unreliable OOS. Not a proven deployable edge. |

**Honest caveat (updated 2026-07-14 post DSR unit-mismatch fix):** Treat `daily_predictor` as a *promising lead under active validation*, not a finished edge. The prediction/strategy split surfaces real signal the classification framing was discarding (IC=+0.06). PBO=0.513 confirms the selected (q,w) pair is unreliable OOS. Corrected median DSR=0.800 (range 0.290–0.943 across 5 symbols) is more encouraging than the prior erroneous DSR=0.000, which resulted from a unit-mismatch bug where annualized Sharpe variance was passed to the per-period deflation formula — fixed 2026-07-14. A 1-bar execution look-ahead was fixed in the backtest pipeline on 2026-07-13. A single untuned backtest is not sufficient validation for capital deployment.

---

### daily_hybrid — Built, Not Deployed

| Field | Value |
|---|---|
| **Status** | Pickle exists (`models/daily_hybrid.pkl`), **not** in `prediction.models` — not loaded by live pipeline |
| **Algorithm** | XGBoost + transformer hybrid (see `hybrid_model.py`, trained via `train_hybrid.py`) |
| **Task** | Same 3-class direction classification |
| **Feature set** | `daily_v5` — same 29 features |
| **OOS accuracy** | At majority-class baseline — transformer component does not improve over `daily_xgboost` alone on this feature set |
| **Known limitations** | An earlier claim of ~58% accuracy was a data-leakage artifact (pre-leakage-fix). After correcting the train/test split, results converge to the classifier baseline. The transformer adds complexity without signal benefit on single-bar daily features. |

To deploy: add `daily_hybrid` to `prediction.models` in `config/default.yaml` and register `"daily_hybrid": "classifier"` in `predict_next_day_lite.MODEL_KINDS`. Not recommended until an independent leakage-free validation shows consistent lift.

---

### daily_dqn — Built, Not Deployed

| Field | Value |
|---|---|
| **Status** | `models/dqn_agent.pt` loaded opportunistically at runtime (if file exists), **not** in `prediction.models` config list |
| **Algorithm** | PyTorch DQN with target network and experience replay |
| **Actions** | `HOLD=0, LONG=1, SHORT=2` mapped to `HOLD/BUY/SELL` |
| **State** | Rolling 20-day window × 29 features (flattened to 580-dim vector) |
| **Trained via** | `train_dqn.py --symbol SPY --days 500 --episodes 30` |
| **Known limitations** | RL on noisy daily financial data is extremely sample-inefficient. Not validated OOS; included for research purposes only. |

---

### When to Retrain

Retrain when **any** of the following trigger conditions is met:

| Condition | Details |
|---|---|
| **Age** | Any model's pickle `mtime` exceeds `prediction.max_model_age_days` (currently 30 days). `predict_next_day_lite.py` checks this at startup and posts a Discord warning embed. |
| **IC degradation** | `daily_predictor` trailing-20 Spearman IC (tracked in `ic_history` DB table and displayed in each Discord header) drops below 0.0 for **10 consecutive trading days**. The classifiers are already at baseline — IC monitoring is most meaningful for `daily_predictor`. |
| **Feature contract bump** | `daily_features.FEATURE_SET_NAME` changes (e.g. `daily_v3` → `daily_v4`). All pickles are immediately incompatible — retrain everything before the next prediction run. Check: `predict_next_day_lite.py` raises `RuntimeError` with "Feature contract mismatch" on the stale model and skips it. |
| **New symbol universe** | If `prediction.symbols` grows to include securities with very different return characteristics (e.g. crypto assets on a daily equity model), consider retraining on the expanded symbol set. |

See `docs/runbook.md` for the exact retraining commands.

## Files

- `daily_logistic.pkl` — Trained LogisticRegression model + RobustScaler + feature contract
- `daily_logistic_v<N>.pkl` — Versioned snapshots (canonical path is always `daily_logistic.pkl`)
- `daily_xgboost.pkl` — Trained XGBoost classifier + RobustScaler + feature contract
- `daily_xgboost_v<N>.pkl` — Versioned snapshots
- `daily_hybrid.pkl` / `daily_hybrid_v<N>.pkl` — XGBoost-transformer hybrid, trained via `train_hybrid.py` (see [hybrid_model.py](../hybrid_model.py))
- `daily_predictor.pkl` / `daily_predictor_v<N>.pkl` — Regression forecaster, trained via `train_predictor.py`. Pairs with `ml_strategies.DailyPredictorStrategy` as the decision layer — see "Prediction vs. strategy" below.
- `dqn_agent.pt` — PyTorch DQN agent (optional; trained separately via `train_dqn.py`)

Each classifier pickle contains: `model`, `scaler`, `feature_contract`, `confidence_threshold`, `label_map`, `trained_at`, `train_symbols`, and accuracy metrics. `daily_predictor.pkl` has the same shape minus the classification-specific fields — see its own section below.

## Model details

### Daily Logistic (`daily_logistic.pkl`)
- **Algorithm**: `sklearn.linear_model.LogisticRegression` (multinomial)
- **Features**: 29 daily features defined in `daily_features.FEATURE_COLS`
- **Labels**: `SELL=0, HOLD=1, BUY=2` — 3-day forward return, thresholded by volatility-scaled bands (`--vol-mult`)
- **Normalization**: `RobustScaler` (fit on training data, stored in pickle)
- **Confidence threshold**: Default 0.55; stored in pickle, read by `predict_next_day_lite.py`
- **Class weighting**: `--logistic-class-weight none` (default) or `balanced`. Default `class_weight="balanced"` tanks test accuracy to ~35% by fighting the natural HOLD-majority prior — see "Accuracy" below.

### Daily XGBoost (`daily_xgboost.pkl`)
- **Algorithm**: `xgboost.XGBClassifier` (`multi:softprob`, 3-class)
- **Features**: Same 29 features as logistic
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
29-feature single-bar technical-indicator set — `daily_hybrid`'s transformer component shows
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
    (same purged split, same 29 features) recovers a small but real out-of-sample signal —
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

**[Look-ahead fix applied 2026-07-13]** The original backtest results below carried a 1-bar
execution look-ahead: the strategy was inadvertently using the same-bar signal for execution
rather than the prior bar's signal, inflating the pre-fix figures (pre-fix: avg return +0.18%,
avg Sharpe +0.27 on a 700-day window). The fix was applied to `simulate_multi.py` /
`ml_strategies.py` and the figures below are the corrected post-fix baseline.

Backtested over a 10-symbol, 365-day window (`simulate_multi.py --strategies daily_predictor
--symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 365`, run 2026-07-13),
`daily_predictor` still had a **positive average return (+0.07%) and positive average Sharpe
(+0.16)** post-fix — a reduction from the pre-fix figures, as expected. Average alpha vs.
buy-and-hold benchmark: **-8.94%** (average information ratio: **-0.62**), confirming the
strategy underperforms passive ownership on most symbols over this window.

**Honest eval results (2026-07-13):**
- **PBO = 0.513** (CPCV, 16 folds, 20 `(signal_quantile, threshold_window)` configs, 12 870
  IS/OOS combinations) — the IS-selected parameter pair is expected to underperform in >50% of
  out-of-sample paths. This is firmly in the high-overfitting zone.
- **Median DSR = 0.800** across AAPL, MSFT, SPY, QQQ, NVDA (2500-day eval window, 5 symbols,
  corrected 2026-07-14 after unit-mismatch fix) — per-symbol: AAPL 0.943, MSFT 0.290, SPY
  0.539, QQQ 0.800, NVDA 0.859. Prior value of 0.000 was an artifact of passing annualized
  Sharpe variance (scale ~252×) into the per-period deflation formula, driving sr0 ~15.9× too
  large and stat deeply negative.

These results confirm that while the prediction/strategy split surfaces a real IC signal
(+0.06, R² = +0.012) that the classification framing discards, the decision-layer parameter
selection is unreliable at current data volumes. The positive Sharpe (+0.16) is encouraging as
a direction but is not a proven edge. Treat this as a promising lead under active validation,
not a finished, deployable strategy.

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
- **State**: Rolling window of last 20 days × 29 features (flattened to 580-dim vector)
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
