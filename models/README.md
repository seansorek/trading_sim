# Pre-trained ML Models

This directory contains pre-trained models committed to the repo so GitHub Actions can load them without retraining on every run.

**Storage strategy:** artifacts stay in git directly (no LFS, no release assets). At ~2.8MB total (largest file `daily_hybrid.pkl` ~1.2MB) the repo-size cost of retrain-and-commit cycles is negligible; LFS would add a quota/billing dependency and an extra checkout step to every workflow for no real benefit at this scale. Revisit if the models directory grows into the tens of MB.

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
| **Feature set** | `daily_v6` — 30 normalized daily technical features (see `daily_features.FEATURE_COLS`; adds Amihud illiquidity and 6 v5 features over the prior daily_v3 set) |
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
| **Feature set** | `daily_v6` — same 30 features as `daily_logistic` |
| **Label construction** | Same as `daily_logistic` |
| **Training window** | Same as `daily_logistic` |
| **Train/test split** | Same purged + embargoed split |
| **Normalization** | `RobustScaler` (same pattern) |
| **Confidence threshold** | 0.55 (softmax probability) |
| **OOS test accuracy** | At majority-class baseline (~61%); no consistent lift over `daily_logistic` |
| **Backtest Sharpe** | Negative across tested windows |
| **Hyperparameters** | `n_estimators=200`, `max_depth=4`, `lr=0.03`, `subsample=0.85`, `colsample_bytree=0.85`, `min_child_weight=2`, `gamma=1.0` (from `config/default.yaml:strategies.xgboost`) |
| **Known limitations** | Same as `daily_logistic`. The regression-side claim that "XGBoost over-regularizes away the signal" was **wrong and has been retracted** — see "XGBoost's IC ≈ 0 was a gamma bug" below. It has not been re-tested for the 3-class classifier, whose `gamma=1.0` is measured against log-loss and is not obviously mis-scaled. |

---

### daily_predictor — Live (Preferred)

| Field | Value |
|---|---|
| **Status** | Live — in `prediction.models`. Most promising signal of the three live models. |
| **Algorithm** | `sklearn.linear_model.ElasticNetCV` (regression) — swapped from `Ridge` 2026-07-28; alpha/l1_ratio chosen by `TimeSeriesSplit` CV inside the training window |
| **Task** | Forecast continuous 3-day forward return (`fwd_ret_1d` in code — the label horizon is `FWD_RET_HORIZON_DAYS=3` trading bars) |
| **Feature set** | `daily_v6` — 30 normalized daily features: the 29 v5 features (dropped williams_r/macd_hist/vol_z_20; added ret_21d, adx_14, vol_regime, rel_volume, hl_ratio, turnover_z, gap; rolling z-score on 18 unbounded features) plus Amihud illiquidity ratio (`|ret_1d| / dollar_volume`, z-scored). VIX features (vix_z, vix_chg_5d) were tested and reverted — they degraded IC on AMZN/NVDA and failed the gate. |
| **Label construction** | Continuous cumulative forward return (not discretized) |
| **Training window** | ~2 500 calendar days (~1 718 trading bars) per symbol, **159 symbols pooled** (was 10 until 2026-07-28 — see "Universe width", below). 202 725 training rows. Retrain with `--days 2500`; symbols come from `config/default.yaml → symbols`. |
| **Train/test split** | Same purged + embargoed split (3-bar gap) |
| **Normalization** | `RobustScaler` + `±5 clip` (via `predictors.base._scale`); consistent between training and live inference. Replaced `StandardScaler` in Task 4 — RobustScaler is more robust to the fat-tailed return distribution. |
| **OOS Spearman IC (final, 2026-07-15)** | Median **+0.0407** across 10 symbols (AAPL=0.0254, MSFT=0.0183, GOOGL=0.1310, AMZN=-0.0073, NVDA=0.0075, META=0.0559, TSLA=0.0840, SPY=0.0595, QQQ=0.0191, IWM=0.0934). Beats `daily_v3` baseline of 0.0266. |
| **Decision layer** | `DailyPredictorStrategy` (`ml_strategies.py`) — causal rolling-quantile threshold on `|predicted return|`: trade only when today's forecast magnitude exceeds the top `1 - signal_quantile` fraction of the trailing `threshold_window` bars' predictions. Tuned to `signal_quantile=0.80`, `threshold_window=40` (sweep run at train time, stored in pickle). The threshold is a percentile of the model's own trailing predictions, so it is scale-invariant — swapping the regressor does not require retuning it by hand. |

**Universe width and the Ridge → ElasticNet swap (2026-07-28).** Both changes were picked by
an expanding walk-forward — each calendar year scored by a fit whose training window stops at
that year's Jan 1 — not by a single split.

*Width.* On 10 symbols, per-year OOS IC ranges from **−0.0143 to +0.0962**; on 159 it ranges
from **+0.0173 to +0.0571**, positive every year for both regressors. Most of that spread was
estimation noise, and 10 pooled names could not tell two models apart at all.

*Model class.* ElasticNet wins 3 of 5 years at the wider universe (mean IC **+0.0400** vs
Ridge's **+0.0385**), and its two losing years lose by 0.0001 and 0.0005 while its wins are
0.0014–0.0047. That is a small, consistent edge, not a decisive one — the honest summary is
"no worse, marginally better, and better motivated": Ridge cannot zero a coefficient, so
correlated near-duplicate features split weight between them, while L1 picks one. The same
ordering held on the 157-name demeaned panel across two separate evaluation windows.

Do not compare the new `test_ic` (+0.0436) against the old pickle's (+0.0740) — different
universes pool different numbers of rows into the test statistic, so those two figures are not
on the same scale. The walk-forward above is the like-for-like comparison.
| **Backtest Sharpe** | **Stale — measured under the one-bar-hold bug (see below).** Was +0.16 avg (10-symbol, 365-day window, post look-ahead fix 2026-07-13). Needs a rerun. |
| **Average return** | **Stale — same reason.** Was +0.07% avg per round-trip. |
| **Alpha vs. benchmark** | -8.94% avg; avg information ratio -0.62 (strategy underperforms buy-and-hold on most symbols over the 365-day post-fix window) |
| **PBO (final, 2026-07-15)** | **0.228** (CPCV, 245 folds, 20 configs, 12870 combinations) — moderate-overfitting zone; better than `daily_v3` baseline of 0.514. IS-selected params expected to underperform OOS in ~23% of paths. |
| **Median DSR (final, 2026-07-15)** | **0.776** across AAPL, MSFT, SPY, QQQ, NVDA — per-symbol DSR: AAPL=0.859, MSFT=0.776, SPY=0.580, QQQ=0.819, NVDA=0.724. Selected (q,w) per symbol: AAPL=(0.75,40), MSFT=(0.80,60), SPY=(0.80,100), QQQ=(0.75,60), NVDA=(0.75,40). |
| **Walk-forward** | Param sweep run at training time via `walk_forward.sweep_params`; best `(signal_quantile, threshold_window)` pair stored in pickle as `best_signal_quantile=0.75`, `best_threshold_window=40`. |
| **Live confidence** | Percentile rank of today's `|predicted return|` within the trailing `threshold_window` — NOT a calibrated probability |
| **Known limitations** | Backtest Sharpe +0.16 is weak in absolute terms. PBO=0.228 is improved vs. the prior baseline (0.514) but still indicates moderate parameter-selection risk. DSR median=0.776 confirms genuine per-symbol signal, but not all symbols are positive (AMZN, NVDA show negative IC). Not a proven deployable edge without portfolio construction and live forward testing. |

**Honest caveat (updated 2026-07-15 — daily_v6 final eval):** Treat `daily_predictor` as a *promising lead under active validation*, not a finished edge. `daily_v6` (30 features, RobustScaler) beats the `daily_v3` baseline on both primary metrics: IC 0.0407 > 0.0266 and PBO 0.228 < 0.514. The Amihud illiquidity feature and the v5 feature cleanup both contributed real signal lift. However, PBO=0.228 still indicates moderate overfitting risk in (q,w) parameter selection, and several individual symbols (AMZN, NVDA, MSFT, QQQ) show near-zero or negative IC folds in recent periods. DSR median=0.776 confirms statistical significance above the multiple-testing-adjusted threshold across the 5-symbol eval set. A 1-bar execution look-ahead was fixed 2026-07-13; the corrected Sharpe (+0.16) is lower than the pre-fix figure (+0.27). A prior DSR=0.000 artifact was a unit-mismatch bug (annualized Sharpe variance passed to per-period deflation) fixed 2026-07-14. A single untuned backtest is not sufficient validation for capital deployment.

---

### daily_hybrid — Built, Not Deployed

| Field | Value |
|---|---|
| **Status** | Pickle exists (`models/daily_hybrid.pkl`), **not** in `prediction.models` — not loaded by live pipeline |
| **Algorithm** | XGBoost + transformer hybrid (see `hybrid_model.py`, trained via `train_hybrid.py`) |
| **Task** | Same 3-class direction classification |
| **Feature set** | `daily_v6` — same 30 features as `daily_logistic`/`daily_xgboost` |
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
| **State** | State-dim = bars × feature count at train time — must retrain if FEATURE_COLS changes (stale; not in prediction.models) |
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
- `daily_predictor_holdout.pkl` — **Research only, never deployed.** Same recipe as `daily_predictor` but `--train-end 2025-07-28`, carving a 12-month holdout. Exists so execution changes can be scored on data the model provably never saw; do not add to `prediction.models`, it deliberately ignores the last year.
- `daily_predictor_holdoutB.pkl` — **Research only, never deployed.** `--train-end 2024-04-16`, so both the flat "window B" year and the favorable "window C" stretch are out of sample for it. This is the fit that regime-tests a result rather than just date-tests it — see "Window-B re-cut" below.
- `dqn_agent.pt` — PyTorch DQN agent (optional; trained separately via `train_dqn.py`)

Each classifier pickle contains: `model`, `scaler`, `feature_contract`, `confidence_threshold`, `label_map`, `trained_at`, `train_symbols`, and accuracy metrics. `daily_predictor.pkl` has the same shape minus the classification-specific fields — see its own section below.

## Model details

### Daily Logistic (`daily_logistic.pkl`)
- **Algorithm**: `sklearn.linear_model.LogisticRegression` (multinomial)
- **Features**: 30 daily features defined in `daily_features.FEATURE_COLS` (`daily_v6`)
- **Labels**: `SELL=0, HOLD=1, BUY=2` — 3-day forward return, thresholded by volatility-scaled bands (`--vol-mult`)
- **Normalization**: `RobustScaler` (fit on training data, stored in pickle)
- **Confidence threshold**: Default 0.55; stored in pickle, read by `predict_next_day_lite.py`
- **Class weighting**: `--logistic-class-weight none` (default) or `balanced`. Default `class_weight="balanced"` tanks test accuracy to ~35% by fighting the natural HOLD-majority prior — see "Accuracy" below.

### Daily XGBoost (`daily_xgboost.pkl`)
- **Algorithm**: `xgboost.XGBClassifier` (`multi:softprob`, 3-class)
- **Features**: Same 30 features as logistic (`daily_v6`)
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
30-feature single-bar technical-indicator set (`daily_v6`) — `daily_hybrid`'s transformer component shows
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
   (same purged split, upgraded to 30 `daily_v6` features) recovers a real out-of-sample signal —
   **median Spearman IC = +0.0407 across 10 symbols (final eval 2026-07-15)** — that none of the
   3-class classifiers could detect at all.

   **Retracted 2026-07-28 — XGBoost's IC ≈ 0 was a gamma bug, not a finding.** This section
   previously concluded that "tree ensembles over-regularize away this weak a signal on this
   feature set". They do not. `train_xgb_regressor` had inherited `gamma=1.0` from the 3-class
   classifier config. `gamma` is a minimum split gain *in units of the loss*: against 3-class
   log-loss a gain of 1.0 is a modest bar, but against squared error on a ~1e-2 return the best
   available split gains ~1e-4, so every split was rejected and the model returned the training
   mean. With `gamma=0.0` the same regressor on the same features reaches test IC +0.029 and
   per-date cross-sectional IC-IR +0.192, against Ridge's +0.170 — comparable, not absent.
   Nothing about tree ensembles on weak signals was ever measured here.

`daily_predictor.pkl` is that regression model (Ridge until 2026-07-28, ElasticNet since — see
the model card above). `ml_strategies.DailyPredictorStrategy` is the
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

**Honest eval results (final, 2026-07-15 — `daily_v6`):**
- **Median walk-forward IC = 0.0407** across 10 symbols (AAPL=0.0254, MSFT=0.0183,
  GOOGL=0.1310, AMZN=-0.0073, NVDA=0.0075, META=0.0559, TSLA=0.0840, SPY=0.0595,
  QQQ=0.0191, IWM=0.0934). Beats `daily_v3` baseline of 0.0266 (+53%).
- **PBO = 0.228** (CPCV, 245 folds, 20 `(signal_quantile, threshold_window)` configs, 12 870
  IS/OOS combinations) — moderate-overfitting zone; substantially better than `daily_v3`
  baseline of 0.514. IS-selected params expected to underperform OOS in ~23% of paths.
- **Median DSR = 0.776** across AAPL, MSFT, SPY, QQQ, NVDA (2500-day eval window, 5 symbols)
  — per-symbol: AAPL 0.859, MSFT 0.776, SPY 0.580, QQQ 0.819, NVDA 0.724.
  Prior DSR=0.000 was a unit-mismatch bug (annualized Sharpe variance passed to per-period
  deflation); fixed 2026-07-14.

These results confirm that `daily_v6` surfaces a real IC signal that beats the `daily_v3`
baseline on both IC and PBO. The decision-layer parameter selection is in the moderate-overfitting
zone (PBO=0.228) but no longer in the high-overfitting zone (PBO=0.514) of the original
classifier baseline. The positive Sharpe (+0.16) is encouraging as a direction but is not a
proven edge without portfolio construction and live forward testing.

**Deployment status:** `daily_predictor` is wired into `predict_next_day_lite.py` and the live
Discord pipeline (`prediction.models` in `config/default.yaml`), alongside `daily_logistic` and
`daily_xgboost`. Live inference uses the exact same decision function as backtesting
(`ml_strategies.compute_predictor_signal`) so the two can't silently diverge. Its Discord
"confidence" field is not a calibrated probability like the classifiers' — it's the percentile
rank of today's |predicted return| within its trailing window (see
`predict_next_day_lite._regressor_confidence`). Given the caveats above, treat its live signals
with the same skepticism as the backtest: a promising lead under active validation, not a
proven edge.

### Every single-name backtest before 2026-07-28 held each position for exactly one bar

`BaseStrategy._apply_holding_period` zeroed suppressed bars. Its output is read by
`Backtester` as a *target position*, not a per-bar trade instruction, so a zeroed
continuation bar did not mean "no new trade" — it meant **go flat**. The method named for
enforcing a *minimum* hold was enforcing a one-bar *maximum* hold followed by a
`holding_period`-bar lockout.

Measured on `daily_predictor`, 10 symbols, 365 days: every position, every symbol, ran for
exactly 1 bar. With the filter disabled entirely, the raw decision layer produces runs of up
to 15–16 bars. So the deployed strategy took a model fit on **3-day** forward returns, held
**1 day**, paid a full round trip, and sat out the next four. Stop-loss and take-profit were
unreachable by construction — across all 10 symbols, zero barrier exits ever fired.

Fixed 2026-07-28: a suppressed bar now carries the position forward. The dwell clock restarts
only on entries and reversals, not on exits — a flat state is not a position being held, and
imposing a cooldown there would discard signal for no reason. Same signal model, same window,
old rule vs. new:

| | mean hold (bars) | mean Sharpe | mean return | mean IR | barrier exits |
|---|---|---|---|---|---|
| old (zero-out) | 1.0 | −1.31 | −0.230% | −0.515 | 0 |
| new (carry-forward) | 6.6 | −0.53 | −0.081% | −0.506 | 11 |

Better on Sharpe for 7 of 10 symbols. **Read this as a bug fix, not an edge.** Both columns
are still negative, information ratio is unchanged to two decimals (the strategy remains worse
than buy-and-hold), and this is a single 365-day window of ~129 usable bars per symbol on
which `oos_guard` could not confirm out-of-sample boundaries. It says the previous numbers
measured a broken execution path, not that the fixed path makes money.

**Every single-name backtest figure recorded above this line predates the fix and was produced
under the one-bar-hold behaviour.** The `daily_predictor` card's +0.16 Sharpe / +0.07% return
included. The panel results are unaffected — `panel_backtester` never used this code path.

Mean hold is now 6.6 bars against a 3-day label horizon, so the next question is a vertical
barrier at the horizon rather than a longer dwell.

### Volatility-scaled stop-loss / take-profit (2026-07-28)

`ExecutionConfig.stop_loss_atr_mult` / `take_profit_atr_mult` replace the fixed 5%/10%
barriers with `mult × ATR-14%` measured at entry (`config/default.yaml` sets 2×/4×, keeping
the same 1:2 risk/reward). ATR is shifted one bar so a wide bar cannot widen the stop meant to
catch it, clipped to a 0.5–10% sanity band, and pinned for the life of the position so the
stop cannot run away from a losing trade. Setting both multipliers to `0.0` restores the fixed
pcts and is the A/B baseline; the fixed pcts also apply during the ATR-14 warmup.

Motivation: a flat 5% stop is a two-day move on TSLA and a quarterly event on SPY, which made
the barrier a de facto per-symbol random exit rule. Measured against the *old* holding rule it
changed nothing at all (identical Sharpe to four decimals, zero barrier exits either way) —
no position lived long enough to reach any barrier. It only became live once that was fixed.

### Vertical barrier (2026-07-28)

`ExecutionConfig.max_holding_bars` forces flat after N bars in a position, completing the
triple barrier: stop, target, clock. `config/default.yaml` sets 3, matching
`daily_features.FWD_RET_HORIZON_DAYS` — holding past the horizon the model was fit on is a bet
on nothing it forecast. Checked after the price barriers so a stop landing on the final bar is
still logged as a stop, and it sets the same re-entry cooldown as the other forced exits;
without that a still-live signal would re-enter next bar and the barrier would only churn
commission. `0` disables it.

### Execution decomposition (2026-07-28)

Each change isolated as one increment, same signal model, 10 symbols, 365 days, means across
symbols. `signal_run` is mean bars per non-zero signal run; `vert` counts vertical-barrier
exits.

| cell | Sharpe | return | IR | max DD | hit | trades | signal_run | stops | TPs | vert |
|---|---|---|---|---|---|---|---|---|---|---|
| A baseline (pre-fix) | −1.311 | −0.230% | −0.515 | −0.383 | 0.359 | 17.5 | 1.00 | 0 | 0 | 0 |
| B +holding fix | −0.654 | −0.131% | −0.511 | −0.666 | 0.487 | 19.9 | 6.61 | 16 | 4 | 0 |
| C +ATR barriers | −0.534 | −0.081% | −0.506 | −0.644 | 0.502 | 20.1 | 6.61 | 10 | 1 | 0 |
| D +vertical barrier | **−0.326** | −0.014% | −0.498 | −0.426 | 0.533 | 17.3 | 6.61 | 4 | 0 | 59 |
| E D but holding_period=0 | −0.896 | −0.208% | −0.511 | −0.483 | 0.450 | 20.7 | 3.05 | 4 | 0 | 18 |

Reading it:

- **The holding fix is the dominant term** (A→B, +0.66 Sharpe), and the vertical barrier is
  second (C→D, +0.21). ATR scaling is the smallest (B→C, +0.12) and works mostly by *widening*
  barriers: stop exits fall 16→10 and take-profits 4→1, i.e. the fixed 5%/10% was cutting
  trades that had not actually moved much in their own name's terms.
- **B and C nearly double max drawdown** (−0.38 → −0.66). Holding a position for six bars
  instead of one is more exposure, and the price barriers alone did not contain it. The
  vertical barrier is what brings drawdown back (−0.43) — the clock, not the stop, is doing
  the risk control.
- **`holding_period` is NOT made redundant by the vertical barrier** (D→E, −0.57 Sharpe). The
  prediction that the clock could replace the dwell filter was wrong: dropping it halves mean
  signal-run to 3.05 and gives back more than the vertical barrier gained. The two rules do
  different jobs — the dwell filter suppresses rapid reversals, the clock caps total exposure.

**Caveat on this table:** it was run against a model whose training window covered the same
dates, because `oos_guard` could not enforce a boundary (see the two bugs below). It is
superseded by the holdout rerun in the next section.

### A real holdout — `daily_predictor_holdout` (2026-07-28)

Two bugs made every previous "out-of-sample" claim in this file unverifiable:

1. **`_save_and_register` recorded `train_end = datetime.now()`**, ignoring `--train-end`
   entirely. `oos_guard` reads exactly that field to refuse in-sample backtest rows, so a model
   split at a fixed calendar date still advertised "trained through today". Fixed: the artifact
   now records the split boundary returned by `prepare_data` (specifically the day before the
   first test date — `oos_guard`'s embargo is in calendar days while the split purges
   `FWD_RET_HORIZON_DAYS` *trading* dates, and a weekend could otherwise let rows inside the
   last training label's horizon back in).
2. **The `(signal_quantile, threshold_window)` sweep ran over all history.**
   `walk_forward.build_fold_data` hardcoded `end = now()`, so the decision layer was tuned on
   the holdout even when the prediction model was not. `build_fold_data` and `sweep_params`
   now take an `end` bound, and `train_predictor` passes `--train-end` through.

`models/daily_predictor_holdout.pkl` — ElasticNet, 159 symbols, `--days 2500
--train-end 2025-07-28`. Train 214 014 rows through 2025-07-28, embargo 3 dates, test 38 955
rows from 2025-07-31. The sweep, now bounded, still picks (0.80, 40).

| | value |
|---|---|
| Holdout pooled IC | **+0.0280** |
| Holdout cross-sectional IC | **+0.0298** (IR +0.162 over 245 dates) |
| Holdout directional accuracy | 0.5190 |
| Holdout R² | −0.0057 |

Do not compare +0.0280 against the deployed model's +0.0436 — different test windows pool
different rows, the same trap flagged for the Ridge→ElasticNet swap. This is simply the first
IC figure for this model measured on data it provably never saw.

The same five-cell decomposition, rerun on that holdout (10 symbols, 245 dates). Features are
built on full history so the `daily_v6` z-scores get their ~110-bar warmup, then `df` and
signal are sliced to the holdout — features at `t` use only data ≤ `t`, so the longer history
cannot leak, whereas trimming *before* feature construction (what `_oos_trim_for_strategy`
does) hands the backtester ~110 bars of half-warmed features.

| cell | Sharpe | return | IR | max DD | hit | trades | signal_run | stops | TPs | vert | beats A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A baseline (pre-fix) | −0.193 | −0.101% | −0.756 | −0.520 | 0.447 | 45.6 | 1.00 | 0 | 0 | 0 | — |
| B +holding fix | +0.028 | +0.008% | −0.752 | −0.884 | 0.520 | 53.0 | 5.61 | 24 | 4 | 0 | 4/10 |
| C +ATR barriers | +0.211 | +0.186% | −0.747 | −0.792 | 0.522 | 53.7 | 5.61 | 20 | 2 | 0 | 6/10 |
| D +vertical barrier | **+0.510** | +0.192% | −0.741 | −0.636 | 0.583 | 44.0 | 5.61 | 4 | 1 | 166 | **8/10** |
| E D but holding_period=0 | −0.188 | −0.108% | −0.758 | −0.694 | 0.517 | 55.5 | 2.58 | 4 | 1 | 52 | 5/10 |

The A < B < C < D ordering replicates on clean data, and holds up better than it did in the
window above: D crosses into positive Sharpe and beats baseline on 8 of 10 symbols rather than
6. E again confirms the vertical barrier does not replace `holding_period`.

**What is still not established.** Information ratio is ≈ −0.75 in every cell and barely moves
across the ladder — against buy-and-hold the strategy loses, and none of these execution
changes touch that. Max drawdown is *worse* than baseline everywhere (−0.52 → −0.64 at D);
holding six bars instead of one is simply more exposure, and the clock only partly contains it.
And this is one 245-date holdout in one regime. The yearly walk-forward below shows 2024 with
positive IC and negative gross Sharpe, so a favorable stretch proves little; the honest read is
that the execution path is now coherent and the ladder is real, not that the strategy makes
money.

### Conviction sizing vs volatility targeting (2026-07-28)

`build_strategy_signal` used to wrap every strategy's output in `np.sign()`, so a fractional
target position was flattened to full size and sizing was unreachable no matter what a decision
layer emitted. It now clips to [-1, 1] instead (identity for ternary strategies), and:

- `compute_predictor_signal(..., conviction=True)` returns the *same triggers and signs* as a
  float scaled by how far the prediction clears its own rolling threshold
  (`|pred| / (2 x thr)`, capped at 1). Only magnitude differs, so the modes are directly
  comparable. The live path uses the default int mode and is untouched.
- `ExecutionConfig.vol_target_annual` scales notional by `target / realized_vol` (20-bar,
  shifted one bar, multiplier clipped to [0.25, 3.0]) — equal risk rather than equal notional.
  Deliberately independent of conviction so each can be measured alone.
- `_apply_holding_period` compares *direction* rather than value, so a conviction size is
  pinned at entry instead of resizing (and paying spread) every bar.

Measured on the holdout, on top of cell D. **Every sizing variant also cuts exposure, and
cutting exposure raises Sharpe by itself** whenever losses are fatter-tailed than gains — so
each is shown against a flat multiplier matched to its realized mean position size: same
average exposure, zero information.

| cell | Sharpe | return | IR | max DD | hit | mean shares |
|---|---|---|---|---|---|---|
| D ternary | +0.510 | +0.192% | −0.741 | −0.636 | 0.583 | 11.18 |
| D +conviction | **+0.598** | +0.159% | −0.743 | −0.369 | 0.589 | 7.58 |
| ⤷ flat control, same exposure | +0.519 | +0.124% | −0.745 | −0.406 | 0.577 | 7.68 |
| D +vol target (15% annual) | +0.389 | +0.135% | −0.743 | −0.381 | 0.575 | 6.66 |
| ⤷ flat control, same exposure | +0.522 | +0.110% | −0.745 | −0.356 | 0.563 | 6.94 |

- **Conviction is doing real work, not de-levering.** The flat control at the same exposure
  gains almost nothing over ternary (+0.519 vs +0.510), so the +0.088 is not an artifact of
  holding less. Against its own control: paired diff **+0.079, t = +1.77, 7/10 symbols**.
  Suggestive, not established — ten symbols in one market are far from ten independent
  observations, so the effective t is weaker than the nominal one.
- **Volatility targeting is worse than simply holding a smaller flat position** (−0.133,
  t = −1.38, 4/10). It destroys more signal than it saves in risk on this strategy; NVDA
  (+0.594 → +0.029) and QQQ (+0.497 → +0.109) carry most of the damage, consistent with the
  edge living partly in the high-vol names that vol targeting shrinks hardest.
- This was **pre-registered the other way**: the prediction was that with IC ≈ 0.03, `|pred|`
  would be too noisy to size on and the vol-target term would carry any gain. Both halves were
  wrong.
- Conviction buys risk-adjusted quality, not return: total return *falls* (+0.192% → +0.159%)
  while max drawdown nearly halves (−0.636 → −0.369).
- IR is ≈ −0.74 in every cell. None of this closes the gap to buy-and-hold.

**Both default to off** (`conviction=False`, `vol_target_annual=0.0`). A t of 1.77 on a single
holdout is not grounds for flipping a default — that is precisely the parameter-selection
behaviour the PBO figures above exist to police.

**Update:** the holdout used here sits entirely inside the favorable 2025+ regime, and the
window-B re-cut of the panel version (below) shows conviction weighting reversing sign in the
flat regime. Treat this single-name t = 1.77 as regime-contaminated too; it has not been
re-cut over window B, and should be before it is quoted again.

### Conviction weighting on the cross-sectional book (2026-07-28)

`rank_to_weights` equal-weighted every name in a leg. With `PanelConfig.conviction`
(`run_panel.py --conviction`) it instead weights by each name's distance from the date's
cross-sectional median, clipped to `[median/2, median*2]` so the most-convicted name holds at
most 4x the least. **Each leg's total notional is preserved exactly**, so gross exposure,
dollar neutrality and beta neutrality are untouched by construction — measured gross exposure
matches to four decimals (1.0033 vs 1.0032), which is why no exposure-matched control is needed
here, unlike the single-name test.

Holdout model, sector-neutralized, 156 names, across the standard config grid:

| window | config | gross SR flat → conv | net SR flat → conv | turnover flat → conv |
|---|---|---|---|---|
| full (1597 d) | d=0.1 r=10 | 0.967 → 0.934 | 0.728 → 0.708 | 0.139 → 0.140 |
| full | d=0.2 r=10 | 0.803 → 0.918 | 0.527 → 0.662 | 0.118 → 0.123 |
| full | d=0.2 r=5 | 1.007 → 1.216 | 0.519 → 0.758 | 0.219 → 0.227 |
| full | d=0.1 r=5 | 1.348 → 1.361 | 0.912 → 0.951 | 0.256 → 0.258 |
| **holdout (249 d)** | d=0.1 r=10 | 1.118 → 1.188 | 0.838 → 0.915 | 0.145 → 0.147 |
| **holdout** | d=0.2 r=10 | 0.569 → 0.847 | 0.264 → **0.547** | 0.128 → 0.132 |
| **holdout** | d=0.2 r=5 | 0.891 → 1.077 | 0.400 → 0.597 | 0.230 → 0.238 |
| **holdout** | d=0.1 r=5 | 0.819 → 1.048 | 0.362 → 0.607 | 0.274 → 0.276 |

Mean net Sharpe: full **+0.672 → +0.770** (3/4 configs), holdout **+0.466 → +0.666** (4/4).
Gross improves in 7 of 8 cells too, so this is not a cost effect — turnover rises by only
1–3%, since re-weighting inside a leg trades far less than changing which names are in it.

The one loss is `d=0.1 r=10` on the full window, and it is the cell where you would expect it:
at decile 0.1 a leg holds ~15 of 156 names, so there is little within-leg dispersion left to
exploit and tilting mostly concentrates.

On its face this looked like the same idea working far better on the book than on single names
(+0.20 mean net Sharpe, 4/4, versus t = 1.77 on 10 symbols). **The window-B re-cut below shows
that reading was wrong** — the 12-month holdout above sits *entirely inside* "window C"
(2025-04-16 →), the favorable stretch identified by the second-holdout analysis, so it tested
the model out-of-sample but never tested the regime.

#### Window-B re-cut — it does not replicate (2026-07-28)

`daily_predictor_holdoutB` (`--train-end 2024-04-16`, `models/daily_predictor_holdoutB.pkl`)
never saw either regime, so one fit scores both:

| window | dates | gross SR flat → conv | net SR flat → conv | net diff | conv wins |
|---|---|---|---|---|---|
| **B** 2024-04-19 → 2025-04-15 | 247 | −0.453 → −0.572 | −0.913 → **−1.009** | **−0.096** | 1/4 |
| **C** 2025-04-16 → 2026-07-28 | 320 | +1.834 → +1.953 | +1.457 → +1.593 | +0.136 | 2/4 |

Conviction *loses* in the flat regime, on gross as well as net, and its window-C win drops from
4/4 to 2/4 once a different fit produces the rankings — with a per-config spread from −0.18 to
+0.43. The pre-registered claim from the previous section fails.

The coherent reading is that conviction weighting is **leverage on signal quality, not a source
of signal**: it concentrates capital in the names the ranker is most confident about, which
amplifies whatever edge exists and equally amplifies its absence. Window B's book is deeply
negative before conviction touches it (net −0.91 flat), and conviction makes it worse. That is
the mechanism behaving exactly as designed, and it is precisely why it cannot be switched on —
the regime is not knowable in advance.

This is the third time a promising lead in this file has turned out to be one regime: the
augment-axis result, the +0.16 single-name Sharpe, and now conviction weighting. The pattern is
consistent enough to treat "measured on 2025+ data only" as disqualifying on its own.

`conviction` defaults to **off** in both the panel and the single-name path, and should stay
off. Reviving it needs a mechanism that is positive in window B, not a better window-C number.

**The deployed `daily_predictor.pkl` is unchanged** — it is trained through the present, which
is correct for live prediction, and was deliberately not replaced by a model that ignores the
last 12 months. Now that `train_end` is recorded properly, a production retrain also makes its
boundary enforceable:

```bash
python train_predictor.py --symbols "$(python -c 'from config import get_config; print(",".join(get_config().symbols))')" --days 2500 --model enet
```

#### Single-name conviction, re-cut the same way (2026-07-28)

The panel re-cut above left the single-name claim (+0.079 Sharpe over its exposure-matched
control, t=+1.77, n=10) untested. Same `holdoutB` fit, 60 names sampled across the config
universe instead of 10, flat control recalibrated **per window** from that window's realised
mean position size:

| window | ternary | conviction | control | conviction − control | t | wins |
|---|---|---|---|---|---|---|
| **B** 2024-04-19 → 2025-04-15 | +0.316 | +0.336 | +0.312 (×0.719) | **+0.024** | +0.90 | 29/60 |
| **C** 2025-04-16 → 2026-07-28 | +0.008 | +0.052 | +0.026 (×0.714) | **+0.027** | +1.22 | 35/60 |

Unlike the panel, single-name conviction does *not* flip sign — but the effect is a third of
what the 10-name holdout reported and is not significant in either window. The original +0.079
was small-sample noise around a true effect of roughly +0.025. Conviction also cuts mean
drawdown (−0.61% → −0.41% in B), but so does the flat control (−0.43%), so that is de-levering,
not conviction. Verdict unchanged: **stays off.**

### The signal has real directional information; the single-name structure cannot harvest it (2026-07-28)

Decomposing the single-name book by side, same 60 names, same two out-of-sample windows:

| window | side | Sharpe | mean ret | hit rate | trades |
|---|---|---|---|---|---|
| **B** | both | +0.316 | +0.225% | 0.546 | 55 |
| | long | +0.399 | +0.217% | 0.677 | 36 |
| | short | +0.072 | +0.036% | 0.382 | 25 |
| **C** | both | +0.008 | +0.087% | 0.559 | 67 |
| | long | +0.170 | +0.177% | 0.656 | 52 |
| | short | −0.308 | −0.110% | 0.290 | 20 |

A 0.29 short-side hit rate reads as "the model is good long and bad short", but that is the
wrong statistic — with P(up) well above 0.5 over a 3-day horizon, a *coin-flip* signal produces
this exact pattern. The discriminating quantity is whether P(up) differs between long-signal and
short-signal bars:

| window | P(up) uncond. | P(up \| long) | P(up \| short) | directional edge | t |
|---|---|---|---|---|---|
| **B** | 0.526 | 0.559 | 0.505 | **+0.056** | **+2.35** |
| **C** | 0.540 | 0.553 | 0.500 | **+0.052** | **+2.35** |

Same magnitude, same t-statistic, two independent out-of-sample regimes. **This is the most
stable result in this file** — and it is the first one that survived the window-B test that
killed the augment axis, the +0.16 single-name Sharpe, and conviction weighting.

The problem is not the signal, it is the structure it is being traded through. On the return
scale:

| window | uncond. 3d ret | long-signal | short-signal | short excess |
|---|---|---|---|---|
| **B** | +0.036% | +0.297% | +0.034% | −0.00% |
| **C** | +0.293% | +0.403% | −0.028% | −0.32% |

The short leg is *informed and still loses*: "below average" in a universe that drifts up is not
a short. A symmetric single-name long/short spends real information fighting drift. Neither
obvious repair works:

- **Drop the shorts.** Long-only Sharpe is +0.399 (B) and +0.170 (C) against buy-and-hold at
  +0.250 and **+0.818**. In the good regime it is a strictly worse way to be long.
- **Net the market out.** Pre-registered: subtracting SPY's own model forecast per bar should
  push short excess negative in both windows and help C more than B, +0.05 to +0.15 Sharpe.
  Result: **−0.44 (t=−2.76) in B and −0.05 in C** — wrong sign in both. The diagnostic says why:
  SPY's same-horizon forecast is noise, not a drift estimate, and subtracting one shared series
  from every name degraded long-side selection from +0.278% to +0.169% excess. Window B's short
  excess *did* reach the predicted −0.30%, and it did not matter, because the long side was
  carrying the book. Reverted, not kept behind a flag.

What this points at is the panel, which nets drift out by construction (long the top decile,
short the bottom) rather than by subtracting an estimate of it. The single-name backtester runs
one symbol per process and cannot hold a hedge, so it structurally cannot express the trade the
evidence supports. **Further single-name decision-layer work is not where the remaining value
is** — the +0.05 directional edge is real and belongs in a market-neutral book.

### Cross-sectional axis and model class — second-holdout result (2026-07-28)

Six cells, {per-symbol z-score axis, both axes} x {Ridge, ElasticNet, XGBoost}, all on the
demeaned target, sector-neutralized, 157-name panel. Selected on 2025-04-16+ ("window C"),
then refit with the split pulled back to 2024-04-16 (`train_predictor --train-end`) and scored
on **2024-04-19 .. 2025-04-15 ("window B"), which had never been used to evaluate anything** —
it sat inside the training data of every earlier fit, so it carries no selection bias.

Pre-registered before looking: augment/ENet beats off/Ridge on paired per-date cross-sectional
IC. It does not. Paired diff on B: **-0.0013 (t = -0.21)** at 1d, -0.0000 at 3d, against
+0.0070 (t = +1.30) and +0.0078 (t = +1.49) on window C with the same refit models. The
augment-axis lead is specific to window C, not a property of the axis. It is not a
training-set-size artifact — the window-C numbers above come from the *short* fits.

The larger fact the holdout exposed is regime, not model choice. Every cell earns roughly
nothing on B and something on C:

| window | dates | book Sharpe range across the 6 cells | mean IC (1d) range |
|---|---|---|---|
| B (2024-04-19 .. 2025-04-15) | 247 | -0.55 .. +0.64 | +0.0014 .. +0.0116 |
| C (2025-04-16 .. 2026-07-28) | 319 | +0.73 .. +1.59 | +0.0165 .. +0.0242 |

Book config held fixed at (decile 0.2, rebalance 10) for both windows — no per-window grid
search. So the encouraging Sharpes recorded elsewhere in this file come from one 15-month
stretch, and the preceding year was flat regardless of which model produced the ranking.

**Yearly walk-forward (2026-07-28).** The two-window split above is coarse, so each calendar
year was re-scored by its own fit whose training window stops at that year's Jan 1 (slicing
years out of one fit is meaningless — the early years are inside its training data). Book
config still fixed at (0.2, 10), Ridge:

| year | mean IC 1d | gross SR | net SR | daily turnover | SPY | median cross-sec. dispersion |
|---|---|---|---|---|---|---|
| 2022 | +0.0027 | +0.83 | +0.41 | 0.145 | −18.6% | 0.0167 |
| 2023 | +0.0160 | +1.07 | +0.56 | 0.141 | +26.7% | 0.0136 |
| 2024 | +0.0063 | −0.96 | −1.51 | 0.146 | +26.0% | 0.0141 |
| 2025 | +0.0103 | +1.59 | +1.23 | 0.144 | +18.9% | 0.0161 |
| 2026 YTD | +0.0173 | +1.49 | +1.25 | 0.133 | +8.7% | 0.0207 |

IC is positive in **5/5 years for both Ridge and ElasticNet** — the ranker has a small but
sign-stable edge across regimes, which the coarse two-window comparison hid. Net Sharpe is
positive in 4/5. 2024 is negative *gross*, so it is not a cost artifact; a positive IC that
year did not convert into book P&L.

Two structural facts from the same decomposition. Rebalancing daily is not viable at all: at
`rebalance_days=1` turnover is ~0.85/day and cost removes 1.4–2.7 Sharpe every year, which is
why the config grid always lands on 10. And the long/short leg split is dominated by market
direction (long leg positive in every up year, negative in 2022; short leg the mirror), so leg
P&L says nothing about ranker skill — only the beta-neutralized combination does.

The only effect with a consistent sign in both windows is ElasticNet over Ridge on the
per-symbol axis (+0.0046/+0.0055 on B, +0.0030/+0.0050 on C at 1d/3d), each |t| between 0.7
and 1.3; pooling the two windows still lands near t ~ 1.3. Suggestive, unproven, and the
cheaper of the two changes to keep.

### DQN (`dqn_agent.pt`)
- **Algorithm**: PyTorch DQN with target network and experience replay
- **Actions**: `HOLD=0, LONG=1, SHORT=2` (mapped to `HOLD/BUY/SELL` in predictions)
- **State**: Rolling window of last 20 days × features (flattened vector; state-dim depends on feature count at train time — must retrain if `FEATURE_COLS` changes)
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
