# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

## [0.4.0] — 2026-07-03 — Documentation & Rigor (Sub-project 4)

### Added
- `models/README.md`: formal model cards for all five models (`daily_logistic`, `daily_xgboost`, `daily_predictor`, `daily_hybrid`, `daily_dqn`) with honest OOS metrics, training details, known limitations, and trigger conditions for retraining
- `docs/runbook.md`: step-by-step operational procedures for retraining models, adding/removing symbols, adding a new live model, Discord webhook failure recovery, and corrupt DB recovery
- `docs/architecture.md`: Mermaid flowchart of the full prediction pipeline (data sources → feature engineering → models → Discord), with component descriptions and key invariants
- `CHANGELOG.md`: this file

---

## [0.3.0] — 2026-07-03 — Operational Quality (Sub-project 3, PR #88)

### Added
- Structured JSON logging in `predict_next_day_lite.py`: every log line includes `ts`, `level`, `logger`, `msg`, and `run_id` (UUID per invocation) — logs are grep-able and traceable end-to-end across a single GitHub Actions run
- Model staleness detection: at startup, `predict_next_day_lite.py` checks each pickle's `mtime` against `prediction.max_model_age_days` (30 days, configurable in `config/default.yaml`) and posts a yellow Discord warning embed for stale models
- Discord consensus block: symbols where ≥2 models agree on BUY or SELL are listed first with a "★ Consensus" label
- Discord IC summary: trailing-20 Spearman IC and directional accuracy per model shown in every Discord header embed
- Discord HOLD suppression: HOLD embeds are suppressed on days when no non-HOLD signals exist for the same strategy (reduces channel noise)
- `tests.yaml` GitHub Actions job: runs `pytest tests/ -v` and `ruff check/format` on every PR and push to main
- pip caching (`cache: 'pip'`) in both GitHub Actions workflows

---

## [0.2.0] — 2026-07-03 — Signal Integrity & Code Quality (Sub-projects 1+2, PR #83, PR #84)

### Added (Signal Integrity)
- `walk_forward.py`: rolling train/test harness with purge/embargo gap, Spearman IC time series, and parameter sweep over `signal_quantile` × `threshold_window`
- Walk-forward param sweep integrated into `train_predictor.py`: best `(signal_quantile, threshold_window)` pair stored in the `daily_predictor` pickle
- Realized IC scorer in `predict_next_day_lite.py`: trailing-20 longitudinal Spearman IC pooled across symbols, computed from `predictions/history.jsonl` via `signal_monitor.score_realized_ic`
- Signal drift detector: 2σ shift sustained for ≥2 consecutive days triggers a Discord warning embed (`signal_monitor.check_signal_drift`)
- `ic_history` table in `db.py`: persists IC scores and directional accuracy per model per date
- Three-level parameter priority for `signal_quantile` / `threshold_window`: env var → pickle → hardcoded default (0.7 / 60)

### Changed (Code Quality)
- Removed dead code: `OrdinalLogisticStrategy`, `XGBoostStrategy` (intraday, not exercised in daily pipeline), `_DailyRidgeQuantileStrategy` (superseded by `DailyPredictorStrategy`), `walk_forward_backtest` (intraday feature set, not called in daily workflow)
- Added type annotations to public functions in `db.py`, `data_loader.py`, `simulation_pipeline.py`, `train_models.py`, `predict_next_day_lite.py`
- `BaseStrategy.signal()` return type annotated as `pd.Series`
- `BacktestResult.profit_factor` type corrected to `Optional[float]`

---

## [0.1.0] — 2026-06 — Leakage Fix & Prediction/Strategy Split

### Fixed
- **Critical:** Train/test split was not purging the `FWD_RET_HORIZON_DAYS`-bar overlap between training labels and test features. After the fix, both `daily_logistic` and `daily_xgboost` accuracy correctly tracks the majority-class baseline — no genuine classification lift exists on this feature set.
- `data_loader.py` cache bug: `check_cache_freshness` checked recency but not coverage of the requested `start` date — earlier "more data" experiments were silently reusing a shorter cache. Retraining on the full requested history still landed at baseline.

### Added
- `train_predictor.py`: Ridge regression on continuous `fwd_ret_1d` target (not discretized SELL/HOLD/BUY) — recovers Spearman IC ≈ +0.06 that classifiers could not detect
- `ml_strategies.DailyPredictorStrategy`: decision layer over the regression model using a causal rolling-quantile threshold on `|predicted return|`
- `ml_strategies.compute_predictor_signal`: shared implementation used identically in backtest and live prediction — the two cannot silently diverge
- `daily_predictor` wired into `predict_next_day_lite.py` and Discord alongside the classifiers
- `predictions/history.jsonl`: append-only JSONL file committed to the repo after every scheduled CI run — the only durable prediction record (the SQLite `daily_predictions` table is ephemeral on CI runners)

---

## [0.0.1] — 2026-05 — Initial Rebuild

### Added
- Full pipeline rebuilt: `data_loader.py` (yfinance + SQLite cache), `daily_features.py` (25-feature `daily_v3` vector), `train_models.py` (logistic + XGBoost), `simulate_multi.py` (parallel backtester), `predict_next_day_lite.py` (daily prediction + Discord), `db.py` (SQLite schema)
- GitHub Actions `simulation.yaml`: daily prediction job at 06:00 UTC
- `config/default.yaml`: single source of truth for all parameters
- DQN agent (`train_dqn.py`, `dqn_agent.py`, `rl_env.py`) — research artifact, not in live pipeline
- XGBoost-transformer hybrid (`train_hybrid.py`, `hybrid_model.py`) — research artifact, not deployed
