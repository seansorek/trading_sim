---
title: Trading Sim — Professional Quality Roadmap
date: 2026-07-02
tags:
  - trading-sim
  - roadmap
  - architecture
status: in-progress
project: trading-sim
---

# Trading Sim — Professional Quality Roadmap

Session on 2026-07-02 kicked off a four-part upgrade to make this bot production-grade. Sub-project 1 is specced and ready to implement. Sub-projects 2–4 are explicitly deferred — documented here so the next session can pick up without re-deriving context.

---

## What Was Decided

The four dimensions of "professional quality" agreed upon:

1. **Signal quality / alpha** — walk-forward validation, IC tracking, tuned parameters
2. **Code quality / architecture** — dead code removal, type annotations, cleaner interfaces
3. **Operational quality** — structured logging, model staleness detection, Discord improvements
4. **Documentation** — model cards, runbooks, architecture diagrams

**Sequencing decision:** Signal quality first (Option A), because the bot's core purpose is generating good signals — validating the signal validates everything else.

---

## Sub-project 1 — Signal Integrity & Walk-Forward Validation

> [!success] Status: Specced — ready for implementation
> Spec: `docs/superpowers/specs/2026-07-02-signal-integrity-design.md`

**What it builds:**
- `walk_forward.py` — rolling train/test harness with purge/embargo gap, IC time series, parameter sweep over `signal_quantile` × `threshold_window`
- `train_predictor.py` extended — runs walk-forward after training, saves best params into pickle
- `predict_next_day_lite.py` extended — realized IC scorer (trailing-20, pooled across symbols) + drift detector (2σ shift for 2 consecutive days)
- `db.py` extended — `ic_history` table
- 3 new test files + extensions to `test_predictor.py`

**Key design choices:**
- Param selection uses median IC across symbols (not per-symbol) — single shared model, single best pair
- Three-level param priority: env var → pickle → hardcoded default (0.7 / 60)
- IC computed longitudinally (trailing 20 (date, symbol) pairs pooled), not cross-sectionally — too few symbols per day for cross-sectional power
- Drift detection uses day-level distribution means (one mean per day), not per-(date, symbol) records

**To start implementation in a new session:**
> Invoke `superpowers:writing-plans` skill with the spec file as context.

---

## Sub-project 2 — Code Quality

> [!todo] Status: Deferred — start after Sub-project 1 ships

**Problem:** The codebase has accumulated dead code and inconsistent patterns that make changes riskier than they need to be.

**Specific targets identified:**

*Dead code to remove:*
- `OrdinalLogisticStrategy` and `XGBoostStrategy` in `ml_strategies.py` — intraday strategies, explicitly not in the daily pipeline, no tests exercise them in live context
- `_DailyRidgeQuantileStrategy` in `simulation_pipeline.py` — defined inline inside a conditional block (`if os.path.exists(_RIDGE_PATH)`), registered as `daily_ridge_q`; superseded by `DailyPredictorStrategy`; the conditional registration pattern is fragile
- `walk_forward_backtest` in `simulation_pipeline.py` — uses a different (intraday) feature set than the daily pipeline; not called anywhere in the daily workflow

*Type annotations:*
- `db.py`, `data_loader.py`, `simulation_pipeline.py`, `train_models.py` have no type annotations on public functions
- `predict_next_day_lite.py` has partial annotations; `load_models` and `predict_symbol` should be fully annotated

*Interface consistency:*
- `BaseStrategy.signal()` return type is `pd.Series` but not annotated — downstream code has to infer the `{-1, 0, 1}` contract
- `BacktestResult` is a dataclass but `metrics` is typed as `Dict[str, float]` while `profit_factor` can be `None` — fix the type

**Approach:** One PR per logical group (dead code removal, type annotations, interface cleanup). Do not mix with signal-quality or operational changes.

---

## Sub-project 3 — Operational Quality

> [!todo] Status: Deferred — start after Sub-project 2

**Problem:** The live pipeline has no observability beyond raw log lines and Discord messages. Model staleness and runtime errors are invisible until Discord goes silent.

**Targets:**

*Structured logging:*
- Replace `logging.basicConfig` with a JSON formatter in `predict_next_day_lite.py` — structured logs are grep-able and can be shipped to a log aggregator later
- Add `run_id` (UUID per invocation) to every log line so a single GitHub Actions run is traceable end-to-end

*Model staleness detection:*
- At startup, `predict_next_day_lite.py` should check the `mtime` of each loaded pickle against a configurable `max_model_age_days` (suggested: 30)
- If any model is stale → post a yellow warning embed to Discord ("daily_predictor model is 45 days old — consider retraining")
- Register `max_model_age_days` in `config/default.yaml`

*Discord output improvements:*
- Currently organizes by strategy → signal type (BUY/SELL/HOLD). Add a top-level **consensus block**: symbols where ≥2 models agree on BUY or SELL, listed first with a "★ Consensus" label
- Include trailing-20 IC per model (from Sub-project 1's `ic_history`) in the header embed so readers can judge signal quality at a glance
- Cap HOLD embeds: only send HOLD embed if non-HOLD signals exist for the same strategy (suppress pure-HOLD days from cluttering the channel)

*CI/CD:*
- Add dependency caching (`pip cache`) to `simulation.yaml` — the `pip install` step is the slowest part of the GitHub Actions run
- Add a separate `tests.yaml` job that runs `pytest` on every PR (currently only the prediction job runs on push)

**Note:** The Discord consensus block depends on Sub-project 1's IC summary being in the message. Build Sub-project 3 after Sub-project 1 is deployed.

---

## Sub-project 4 — Documentation & Rigor

> [!todo] Status: Deferred — start after Sub-project 3

**Problem:** There are no model cards, no honest performance summary for external readers, and no runbook for common operations.

**Targets:**

*Model cards (update `models/README.md`):*
- One section per live model: what it predicts, how it was trained (feature set, training window, label construction), honest OOS metrics (IC, directional accuracy, backtest Sharpe with caveats), known limitations
- Add a "when to retrain" section: trigger conditions (model age > 30 days, IC drops below 0 for 10 consecutive days, feature contract version bump)

*Runbook (`docs/runbook.md`):*
- Re-training procedure (step-by-step, including the `--days` flag rationale)
- Adding/removing a symbol from the prediction universe
- Adding a new model to the live pipeline (the `MODEL_KINDS` registry + `config/default.yaml` dance)
- Responding to a Discord webhook failure
- Recovering from a corrupt DB (`data/trading_sim.db`)

*Architecture diagram:*
- Mermaid diagram in `docs/architecture.md` showing: data sources → feature engineering → models → prediction pipeline → Discord. Include the DB as a side channel.

*CHANGELOG:*
- Start a `CHANGELOG.md` from the current state, with one entry per major milestone (rebuild May 2026, leakage fix Jun 2026, signal integrity Jul 2026)

---

## Context for Future Sessions

> [!info] Read before starting any sub-project
> The `daily_predictor` (Ridge regression) is the most promising model. The classifiers (`daily_logistic`, `daily_xgboost`) are at or near the majority-class baseline — do NOT try to improve them by sweeping hyperparameters; that space was exhaustively covered in June 2026 with no genuine lift. See memory file `trading_sim_honest_accuracy_2026_06.md` for the full investigation.

**Key invariants to preserve:**
- `FEATURE_COLS` in `daily_features.py` is the feature contract — changing it requires retraining all models and bumping `FEATURE_SET_NAME`
- `compute_predictor_signal` in `ml_strategies.py` is the single source of truth for the predictor decision layer — used by both backtest and live pipeline; never duplicate it
- `predictions/history.jsonl` is append-only and committed to the repo — it's the only durable record of what the live models predicted (the SQLite DB is ephemeral on CI runners)
- The three-level param priority (`env var → pickle → hardcoded default`) must be preserved in any code that reads `signal_quantile` or `threshold_window`

**Running the project locally:**
```bash
.venv/Scripts/python.exe train_predictor.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 2500
.venv/Scripts/python.exe simulate_multi.py --symbols AAPL,SPY --strategies daily_predictor
.venv/Scripts/python.exe predict_next_day_lite.py --symbols AAPL,SPY
pytest tests/ -v
```
