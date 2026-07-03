# Trading Sim — Operations Runbook

Operational procedures for common maintenance tasks. Each procedure is self-contained.

---

## 1. Retraining Models

### When to retrain

See `models/README.md` → "When to Retrain" for trigger conditions. The short version:
- Any model's pickle `mtime` exceeds 30 days (Discord will warn you)
- `daily_predictor` trailing-20 IC drops below 0.0 for 10 consecutive trading days
- `daily_features.FEATURE_SET_NAME` changes (you will see a `RuntimeError` in logs: "Feature contract mismatch")

### Retraining `daily_logistic` and `daily_xgboost`

```bash
# From the project root. Adjust --symbols if your universe changed.
python train_models.py \
  --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM \
  --days 1000

# On Windows:
.venv\Scripts\python.exe train_models.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 1000
```

This writes `models/daily_logistic.pkl` and `models/daily_xgboost.pkl` and registers them in
`data/trading_sim.db`.

Optional flags:
- `--models logistic` or `--models xgboost` — train only one of the two
- `--optimize` — Bayesian hyperparameter search (slow; requires `scikit-optimize`)
- `--confidence 0.55` — bakes this threshold into the saved pickle

### Retraining `daily_predictor`

```bash
python train_predictor.py \
  --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM \
  --days 2500
```

`--days 2500` is intentionally longer (~6.8 years of trading data) because the regression model
benefits from more history; the classifiers don't.

### Deploying updated models

```bash
# Stage the canonical pickles (not the versioned snapshots)
git add models/daily_logistic.pkl models/daily_xgboost.pkl models/daily_predictor.pkl

# Verify what you're committing — never commit dqn_agent.pt or hybrid.pkl accidentally
git diff --cached --stat

git commit -m "retrain: update models $(date -u +%Y-%m-%d)"
git push
```

The next GitHub Actions run (or the next 06:00 UTC scheduled run) will pick up the new pickles.

### Verifying after deploy

Run the prediction job locally against a small symbol set to confirm no load errors:

```bash
python predict_next_day_lite.py --symbols AAPL,SPY
```

Expected stdout: a JSON-structured prediction log with no `RuntimeError`, followed by the
`DAILY PREDICTIONS` summary block with signals for both symbols.

---

## 2. Adding or Removing a Symbol from the Prediction Universe

### Adding a symbol

1. Add the ticker to `prediction.symbols` in `config/default.yaml`:
   ```yaml
   prediction:
     symbols:
       - AAPL
       - NEWTKR   # ← add here
   ```
2. No code change needed. `predict_next_day_lite.py` reads `prediction.symbols` at startup.
3. *(Optional but recommended)* Backtest the new symbol before adding it to Discord output:
   ```bash
   python simulate_multi.py --symbols NEWTKR --strategies daily_predictor --days 365
   ```
4. Commit and push:
   ```bash
   git add config/default.yaml
   git commit -m "feat: add NEWTKR to prediction universe"
   git push
   ```

### Removing a symbol

1. Remove the ticker from `prediction.symbols` in `config/default.yaml`.
2. Commit and push — no other changes needed.

**Note:** Removing a symbol does not delete its historical predictions from `predictions/history.jsonl`. That file is append-only; old records are harmless.

---

## 3. Adding a New Model to the Live Pipeline

This procedure adds a trained model so it generates daily Discord signals.

### Step A — Train and save the model

The pickle must contain at minimum:
```python
{
    "model": <fitted sklearn/xgboost estimator>,
    "scaler": <fitted StandardScaler>,
    "feature_contract": FEATURE_COLS,   # from daily_features.py — must match exactly
    # Classifier pickles also need:
    "confidence_threshold": 0.55,
    "label_map": {"0": "SELL", "1": "HOLD", "2": "BUY"},
    # Regressor pickles also need:
    "best_signal_quantile": 0.7,
    "best_threshold_window": 60,
}
```

Save it to `models/daily_mymodel.pkl`.

### Step B — Register the model kind in `predict_next_day_lite.py`

Open `predict_next_day_lite.py` and add one entry to `MODEL_KINDS`:

```python
MODEL_KINDS = {
    "daily_logistic": "classifier",
    "daily_xgboost": "classifier",
    "daily_predictor": "regressor",
    "daily_mymodel": "classifier",   # ← add this line ("classifier" or "regressor")
}
```

A model whose key is not in `MODEL_KINDS` raises `RuntimeError("Unknown model kind")` and is
skipped — this is intentional to catch misconfiguration early.

### Step C — Enable the model in `config/default.yaml`

```yaml
prediction:
  models:
    - daily_logistic
    - daily_xgboost
    - daily_predictor
    - daily_mymodel   # ← add here
```

### Step D — Add a display name for Discord (optional)

In `predict_next_day_lite.send_discord`, add:
```python
strategy_display = {
    ...
    "daily_mymodel": "My Model Name",
}
```

If omitted, the raw model key is used as the display name — acceptable.

### Step E — Commit and push

```bash
git add models/daily_mymodel.pkl predict_next_day_lite.py config/default.yaml
git commit -m "feat: add daily_mymodel to live prediction pipeline"
git push
```

### Step F — Verify

```bash
python predict_next_day_lite.py --symbols AAPL,SPY
```

Expected: log line `Loaded daily_mymodel from models/daily_mymodel.pkl` and model appears in
the `DAILY PREDICTIONS` summary.

### Removing a model from the live pipeline

Remove its entry from `prediction.models` in `config/default.yaml`. No code change needed —
`predict_next_day_lite.py` loads exactly the list in config. The pickle and `MODEL_KINDS`
registration can stay in place (harmless).

---

## 4. Responding to a Discord Webhook Failure

### Symptoms

- No Discord message arrives by 06:10 UTC on a scheduled day
- GitHub Actions log shows `Discord HTTP 4xx` or `Discord send failed`
- `predict_next_day_lite.py` log line: `Discord notification failed.`

### Diagnosis

1. **Check the GitHub Actions run:**
   - Go to repo → Actions → "Daily Trading Predictions" → most recent run
   - Look for the `Generate next-day predictions` step output
   - A `Discord HTTP 401` means the secret is invalid/expired
   - A `Discord HTTP 404` means the webhook URL has been deleted or revoked

2. **Check whether Discord actually received anything:**
   - Even a failed webhook run still writes `tomorrow_trades.json` — check that the predictions were generated correctly
   - The `predictions/history.jsonl` commit (the next step in the workflow) confirms the run completed

3. **Verify the webhook URL manually:**
   ```bash
   curl -X POST -H "Content-Type: application/json" \
     -d '{"content":"test — webhook alive"}' \
     "$DISCORD_WEBHOOK_URL"
   # Expected: HTTP 204 No Content
   ```

### Resolution

- **Invalid/expired webhook URL:** Regenerate the webhook in Discord (channel Settings → Integrations → Webhooks → copy new URL) and update the GitHub secret: repo Settings → Secrets and variables → Actions → `DISCORD_WEBHOOK_URL` → Update.
- **Push-to-main runs don't send Discord:** By design — only `schedule` and `workflow_dispatch` events have the secret injected (see `.github/workflows/simulation.yaml`). This is not a failure.
- **Temporary Discord outage:** Rerun the workflow via GitHub Actions → "Daily Trading Predictions" → "Run workflow". This triggers a `workflow_dispatch` event, which does inject the Discord secret.

---

## 5. Recovering from a Corrupt or Missing Database

### Context

`data/trading_sim.db` is a local SQLite database. On GitHub Actions, it starts fresh every run — CI intentionally does not persist the DB between runs. The durable record of live predictions is `predictions/history.jsonl` (committed to the repo after every scheduled run), not the DB.

### Local recovery

```bash
# Remove the corrupt file
rm data/trading_sim.db

# The DB is recreated idempotently the next time any script runs
python predict_next_day_lite.py --symbols AAPL,SPY
# Expected: DB created at data/trading_sim.db, no errors

# Re-populate the OHLCV bar cache by running a training job
# (the DB bar_data table is just a cache; yfinance is the source of truth)
python train_models.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 1000
```

### What is lost

- **bar_data cache** — refetched from yfinance on next run; no permanent data loss
- **features cache** — recomputed on next run
- **model_registry** — lost; `predict_next_day_lite.py` falls back to canonical `models/<key>.pkl` paths automatically (the `_resolve_path` fallback in `load_models`)
- **daily_predictions table** — historical prediction records in the DB. The durable copy is `predictions/history.jsonl` in the repo
- **ic_history table** — IC tracking data. Will rebuild from `predictions/history.jsonl` on next run

### SQLite integrity check (before deleting)

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/trading_sim.db')
result = conn.execute('PRAGMA integrity_check').fetchone()
print(result)
conn.close()
"
# Expected: ('ok',)
# If not 'ok', the DB is corrupt — safe to delete and recreate
```
