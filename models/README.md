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
| **Backtest Sharpe** | +0.16 avg (10-symbol, 365-day window, post look-ahead fix 2026-07-13; pre-fix was +0.27 on 700-day window) — the only positive Sharpe among the three live models; see PBO/DSR below |
| **Average return** | +0.07% avg per round-trip (10-symbol, 365-day window, post-fix 2026-07-13; pre-fix was +0.18%) |
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
