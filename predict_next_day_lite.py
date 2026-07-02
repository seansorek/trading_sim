#!/usr/bin/env python3
"""
predict_next_day_lite.py — Lightweight prediction script for GitHub Actions.

Key design contract:
- Feature ordering comes from daily_features.FEATURE_COLS, never from DataFrame column iteration.
- Scaler is always applied before predict_proba (raises RuntimeError if missing).
- Predictions are written to daily_predictions DB table and tomorrow_trades.json.
- Fails loudly on model load errors instead of silently returning HOLD.
"""
import argparse
import json
import logging
import os
import pickle
import random
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import torch

from config import get_config
from data_loader import load_yfinance
from daily_features import FEATURE_COLS, make_daily_features
from db import DB
from dqn_signal import gate_dqn_signal
from ml_strategies import compute_predictor_signal
from signal_monitor import score_realized_ic, check_signal_drift, _signal_to_score
from train_models import _preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_SIGNAL_NAMES = ["SELL", "HOLD", "BUY"]  # index matches label {0,1,2}

# Maps a loaded model_key to how predict_symbol should run it. Adding a new
# classifier or regressor model means adding one entry here and one entry
# to prediction.models in config/default.yaml — predict_symbol, the DB
# upsert, and the Discord payload all pick it up automatically with no
# further code changes. DQN is handled separately below (different
# artifact format and a windowed state, not a single-row prediction).
MODEL_KINDS = {
    "daily_logistic": "classifier",
    "daily_xgboost": "classifier",
    "daily_predictor": "regressor",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_pkl(path: str, model_key: str) -> dict:
    """Load and validate a model pickle. Raises RuntimeError on any failure."""
    if not os.path.exists(path):
        raise RuntimeError(
            f"[{model_key}] Model file not found: {path}. Run train_models.py first."
        )
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(f"[{model_key}] Cannot unpickle {path}: {exc}") from exc

    for key in ("model", "scaler", "feature_contract"):
        if key not in data:
            raise RuntimeError(
                f"[{model_key}] Pickle {path} is missing key '{key}'. Retrain."
            )
    if data["feature_contract"] != FEATURE_COLS:
        raise RuntimeError(
            f"[{model_key}] Feature contract mismatch — expected {len(FEATURE_COLS)} "
            f"features, got {len(data['feature_contract'])}. Retrain."
        )
    return data


def _predict_classifier_signal(data: dict, X_latest: np.ndarray) -> dict:
    """
    Predict today's signal for a 3-class classifier model (daily_logistic,
    daily_xgboost). Both share identical prediction logic — only the
    trained model object differs — so this one function serves both,
    instead of two copy-pasted blocks.
    """
    X_scaled = data["scaler"].transform(X_latest)
    prob = data["model"].predict_proba(X_scaled)[0]
    pred_idx = int(np.argmax(prob))
    confidence = float(prob[pred_idx])
    threshold = data.get("confidence_threshold", 0.55)
    pred_class = int(data["model"].classes_[pred_idx])
    signal = _SIGNAL_NAMES[pred_class]
    if signal != "HOLD" and confidence < threshold:
        signal = "HOLD"
    return {"signal": signal, "confidence": confidence}


def _regressor_confidence(pred_ret: np.ndarray, threshold_window: int) -> float:
    """
    Percentile rank of today's |predicted return| within the trailing
    threshold_window predictions, in [0, 1].

    This is NOT a calibrated probability like the classifiers' confidence
    (Ridge has no predict_proba) — it is the fraction of the trailing
    window whose magnitude today's prediction equals or exceeds. Higher
    means today's forecast is more extreme relative to its recent history,
    which is also what compute_predictor_signal's rolling quantile uses to
    decide whether to trade.
    """
    window = pred_ret[-threshold_window:]
    if len(window) < 2:
        return 0.0
    today_abs = abs(pred_ret[-1])
    return float(np.mean(np.abs(window) <= today_abs))


def _predict_regressor_signal(
    data: dict,
    X_all: np.ndarray,
    signal_quantile: float = 0.7,
    threshold_window: int = 60,
) -> dict:
    """
    Predict today's signal for a regression-style model (daily_predictor).

    Unlike the classifiers, this needs the *trailing window* of
    predictions (X_all), not just the latest bar, because
    compute_predictor_signal's causal rolling quantile needs history to
    decide whether today's forecast is extreme enough to trade — the same
    decision logic used in backtesting (ml_strategies.DailyPredictorStrategy),
    so live and backtest predictions can't silently diverge.

    Applies the same ±5-std-clip preprocessing (train_models._preprocess)
    daily_predictor was trained on (see train_predictor.prepare_data) —
    skipping this would feed the model out-of-distribution inputs it never
    saw in training.
    """
    X_clean = _preprocess(X_all.copy())
    X_scaled = data["scaler"].transform(X_clean)
    pred_ret = data["model"].predict(X_scaled)

    sq_env = os.environ.get("PREDICTOR_SIGNAL_QUANTILE")
    if sq_env is not None:
        try:
            sq = float(sq_env)
        except ValueError:
            logger.warning("Invalid PREDICTOR_SIGNAL_QUANTILE=%r — ignoring", sq_env)
            sq = float(data.get("best_signal_quantile", signal_quantile))
    else:
        sq = float(data.get("best_signal_quantile", signal_quantile))

    tw_env = os.environ.get("PREDICTOR_THRESHOLD_WINDOW")
    if tw_env is not None:
        try:
            tw = int(tw_env)
        except ValueError:
            logger.warning("Invalid PREDICTOR_THRESHOLD_WINDOW=%r — ignoring", tw_env)
            tw = int(data.get("best_threshold_window", threshold_window))
    else:
        tw = int(data.get("best_threshold_window", threshold_window))

    signals = compute_predictor_signal(pred_ret, sq, tw)

    last_signal = int(signals[-1])
    signal_name = _SIGNAL_NAMES[last_signal + 1]
    confidence = _regressor_confidence(pred_ret, tw)
    return {"signal": signal_name, "confidence": confidence}


def load_models(
    db: Optional[DB] = None, model_keys: Optional[list] = None
) -> dict:
    """
    Load all available daily model pickles.

    model_keys defaults to config.prediction.models (config-driven, so a
    model can be added to or removed from the live pipeline by editing
    config/default.yaml — no code change). Uses DB model_registry to
    resolve artifact_path when available, falling back to the canonical
    models/<model_key>.pkl path otherwise. A configured model whose
    artifact is missing or invalid is logged and skipped, not fatal —
    other configured models still load and predict.
    """
    models = {}

    def _resolve_path(model_key: str, canonical: str) -> str:
        if db is not None:
            meta = db.get_active_model(model_key)
            if meta and os.path.exists(meta["artifact_path"]):
                return meta["artifact_path"]
        return canonical

    if model_keys is None:
        model_keys = get_config().prediction.models

    for model_key in model_keys:
        canonical = f"models/{model_key}.pkl"
        path = _resolve_path(model_key, canonical)
        try:
            models[model_key] = _load_pkl(path, model_key)
            logger.info("Loaded %s from %s", model_key, path)
        except RuntimeError as exc:
            logger.warning("%s", exc)

    # DQN agent (separate format)
    if os.path.exists("models/dqn_agent.pt"):
        try:
            from dqn_agent import DQNAgent
            agent = DQNAgent.load("models/dqn_agent.pt")
            agent.q.eval()
            models["daily_dqn"] = agent
            logger.info("Loaded daily_dqn from models/dqn_agent.pt")
        except Exception as exc:
            logger.warning("DQN load failed: %s", exc)

    return models


# ---------------------------------------------------------------------------
# Per-symbol prediction
# ---------------------------------------------------------------------------

def predict_symbol(
    symbol: str,
    models: dict,
    db: Optional[DB] = None,
    dqn_window: int = 20,
    history_days: int = 1000,
    spy_df=None,
) -> dict:
    result: dict = {"symbol": symbol, "timestamp": datetime.utcnow().isoformat()}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=history_days)

    try:
        df = load_yfinance(
            symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
        )
    except Exception as exc:
        result["error"] = f"Data fetch failed: {exc}"
        return result

    if df is None or len(df) < 50:
        result["error"] = "Insufficient data"
        return result

    try:
        spy_arg = spy_df if symbol != "SPY" else None
        feats = make_daily_features(df, spy_df=spy_arg)
    except Exception as exc:
        result["error"] = f"Feature computation failed: {exc}"
        return result

    # Feature matrix — always index by FEATURE_COLS, never by column position
    X_all = feats[FEATURE_COLS].values.astype(np.float32)

    # Latest single-day input for sklearn models
    X_latest = X_all[-1:]
    result["price"] = float(df["close"].iloc[-1])
    result["predictions"] = {}
    prediction_date = datetime.utcnow().strftime("%Y-%m-%d")

    for model_key, data in models.items():
        if model_key == "daily_dqn":
            continue  # handled separately below — different artifact format
        try:
            kind = MODEL_KINDS.get(model_key)
            if kind == "classifier":
                pred = _predict_classifier_signal(data, X_latest)
            elif kind == "regressor":
                pred = _predict_regressor_signal(data, X_all)
            else:
                raise RuntimeError(
                    f"Unknown model kind for '{model_key}' — "
                    "register it in MODEL_KINDS."
                )
            result["predictions"][model_key] = pred
            if db is not None:
                meta = db.get_active_model(model_key)
                version = meta["version"] if meta else 0
                db.upsert_prediction(
                    symbol, model_key, version, prediction_date,
                    pred["signal"], pred["confidence"], result["price"],
                )
        except Exception as exc:
            result["predictions"][model_key] = {"error": str(exc)}

    # --- DailyDQN ---
    if "daily_dqn" in models:
        try:
            agent = models["daily_dqn"]
            n_rows, n_cols = X_all.shape

            # Z-score normalize features to match DQN training (rl_env.py).
            # Fit scaler on a warmup window to avoid look-ahead bias.
            fit_end = min(252, max(dqn_window + 1, n_rows // 2))
            X_normed = X_all.copy()
            for ci, col in enumerate(FEATURE_COLS):
                mu = float(X_all[:fit_end, ci].mean())
                sd = float(X_all[:fit_end, ci].std() or 1.0)
                if sd == 0:
                    sd = 1.0
                X_normed[:, ci] = (X_all[:, ci] - mu) / sd

            if n_rows >= dqn_window:
                state = X_normed[-dqn_window:].flatten()
            else:
                pad = np.zeros(dqn_window * n_cols - X_normed.size, dtype=np.float32)
                state = np.concatenate([X_normed.flatten(), pad])

            with torch.no_grad():
                s_t = torch.from_numpy(state).float().unsqueeze(0)
                q_vals = agent.q(s_t).squeeze(0).cpu().numpy()

            # Apply the same gating logic used by DailyDQNStrategy in backtests
            cfg = get_config()
            signal, raw_confidence = gate_dqn_signal(
                q_vals,
                confidence_threshold=cfg.strategies.dqn.confidence_threshold,
                q_advantage_threshold=cfg.strategies.dqn.q_advantage_threshold,
            )

            # Normalize raw Q-spread to [0, 1] for downstream consumers
            # (Discord messages, tomorrow_trades.json).  The old inline DQN
            # code used:  np.clip((q_margin + 1) / 2, 0, 1)
            confidence = float(np.clip((raw_confidence + 1.0) / 2.0, 0.0, 1.0))

            result["predictions"]["daily_dqn"] = {
                "signal": signal,
                "confidence": confidence,
            }
            if db is not None:
                db.upsert_prediction(
                    symbol, "daily_dqn", 0, prediction_date,
                    signal, confidence, result["price"]
                )
        except Exception as exc:
            result["predictions"]["daily_dqn"] = {"error": str(exc)}

    return result


# ---------------------------------------------------------------------------
# Predictions history (append-only, survives across ephemeral CI runs)
# ---------------------------------------------------------------------------

def append_predictions_history(predictions: list, history_path: str) -> int:
    """Append today's predictions to an append-only JSONL history file.

    One record per (symbol, model) prediction. Unlike the daily_predictions
    DB table, this file is meant to be committed back to the repo so it
    survives ephemeral CI runners — it's the only durable record of what
    the live models actually predicted on a given day. Returns the number
    of records written.
    """
    prediction_date = datetime.utcnow().strftime("%Y-%m-%d")
    records = []
    for pred in predictions:
        if "error" in pred:
            continue
        symbol = pred["symbol"]
        price = pred.get("price")
        for model_key, model_pred in pred.get("predictions", {}).items():
            if "signal" not in model_pred:
                continue
            records.append({
                "date": prediction_date,
                "symbol": symbol,
                "model": model_key,
                "signal": model_pred["signal"],
                "confidence": model_pred["confidence"],
                "price": price,
            })

    if not records:
        return 0

    parent = os.path.dirname(history_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(history_path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return len(records)


# ---------------------------------------------------------------------------
# Discord notification
# ---------------------------------------------------------------------------

def send_discord(
    predictions: list,
    webhook_url: str,
    ic_results: dict | None = None,
    drift_warnings: dict | None = None,
) -> bool:
    if not webhook_url:
        logger.warning("No Discord webhook URL — skipping notification")
        return False
    try:
        import requests
    except ImportError:
        logger.error("requests not installed — cannot send Discord notification")
        return False

    strategy_display = {
        "daily_logistic": "Daily Logistic",
        "daily_xgboost": "Daily XGBoost",
        "daily_predictor": "Daily Predictor",
        "daily_dqn": "Daily DQN",
    }

    # Organize by strategy → signal
    by_strategy: dict = {}
    for pred in predictions:
        if "error" in pred:
            continue
        symbol = pred["symbol"]
        price = pred.get("price", 0.0)
        for model_key, model_pred in pred.get("predictions", {}).items():
            if "signal" not in model_pred:
                continue
            signal = model_pred["signal"]
            confidence = model_pred.get("confidence", 0.0)
            by_strategy.setdefault(model_key, {"BUY": [], "SELL": [], "HOLD": []})
            by_strategy[model_key][signal].append(
                {"symbol": symbol, "price": price, "confidence": confidence}
            )

    embeds = []
    color_map = {"BUY": 0x00FF00, "SELL": 0xFF0000, "HOLD": 0xFFFF00}

    for strategy in sorted(by_strategy):
        display = strategy_display.get(strategy, strategy)
        for sig_type in ("BUY", "SELL", "HOLD"):
            recs = sorted(
                by_strategy[strategy][sig_type],
                key=lambda x: x["confidence"],
                reverse=True,
            )
            if not recs:
                continue
            lines = [
                f"**{r['symbol']}** ${r['price']:.2f} | {r['confidence']:.1%}"
                for r in recs
            ]
            description = "\n".join(lines)
            if len(description) > 4096:
                description = description[:4093] + "..."
            embeds.append({
                "title": f"{display} — {sig_type} ({len(recs)})",
                "description": description,
                "color": color_map[sig_type],
            })

    if drift_warnings:
        for model, warned in sorted(drift_warnings.items()):
            if warned:
                embeds.insert(0, {
                    "title": f"⚠️ Signal Drift Detected — {model}",
                    "description": (
                        "Today's predicted-return distribution has shifted more than 2σ "
                        "from the trailing 30-day baseline for the second consecutive day. "
                        "Consider reviewing model freshness or market regime."
                    ),
                    "color": 0xFFFF00,
                })

    if not embeds:
        logger.warning("No embeds to send to Discord")
        return False

    ic_lines = []
    if ic_results:
        for model, res in sorted(ic_results.items()):
            if res is not None:
                ic_lines.append(
                    f"`{model}` IC={res['ic']:+.3f} dir-acc={res['directional_accuracy']:.0%} "
                    f"(n={res['lookback_n']})"
                )
    ic_summary = ("  |  ".join(ic_lines)) if ic_lines else "IC: not yet available"
    header = (
        f"**Daily Trading Predictions** — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Trailing-20 {ic_summary}"
    )
    success = True
    for i in range(0, len(embeds), 10):
        batch = embeds[i : i + 10]
        payload = {
            "username": "Trading Sim",
            "content": header if i == 0 else "",
            "embeds": batch,
        }
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            if resp.status_code not in (200, 204):
                logger.warning("Discord HTTP %s: %s", resp.status_code, resp.text[:200])
                success = False
        except Exception as exc:
            logger.error("Discord send failed: %s", exc)
            success = False

    return success


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    cfg = get_config()

    parser = argparse.ArgumentParser(description="Predict next-day signals")
    parser.add_argument(
        "--symbols", default=None,
        help="Comma-separated symbols (default: prediction.symbols from config/default.yaml)"
    )
    parser.add_argument("--db", default="data/trading_sim.db")
    parser.add_argument("--output", default="tomorrow_trades.json")
    parser.add_argument(
        "--history", default="predictions/history.jsonl",
        help="Append-only JSONL file recording each day's predictions "
             "(tracked in git so it survives ephemeral CI runners; "
             "pass an empty string to skip writing it)"
    )
    args = parser.parse_args()

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else cfg.prediction.symbols
    )

    db: Optional[DB] = None
    try:
        db = DB(args.db)
    except Exception as exc:
        logger.warning("DB not available: %s — predictions will not be stored", exc)

    models = load_models(db=db)
    if not models:
        logger.error("No trained models found. Run train_models.py first.")
        sys.exit(1)

    # --- Realized IC scoring ---
    ic_results: dict = {}
    prediction_date = datetime.utcnow().strftime("%Y-%m-%d")
    if args.history:
        try:
            def _fetch_prices(symbol: str, start: str, end: str):
                return load_yfinance(symbol, start=start, end=end, interval="1d")

            ic_results = score_realized_ic(
                args.history, prediction_date, fetch_prices_fn=_fetch_prices
            )
            for model, res in ic_results.items():
                if res is not None:
                    logger.info(
                        "Trailing IC [%s]: ic=%.4f dir_acc=%.2f n=%d",
                        model, res["ic"], res["directional_accuracy"], res["lookback_n"],
                    )
                    if db is not None:
                        db.upsert_ic(
                            model, prediction_date,
                            res["lookback_n"], res["ic"], res["directional_accuracy"],
                            res["mean_pred"], res["std_pred"],
                        )
        except Exception as exc:
            logger.warning("IC scoring failed: %s", exc)

    # Load SPY once for market-relative features
    spy_df = None
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=cfg.data.history_days)
        spy_df = load_yfinance(
            "SPY",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
        )
    except Exception as exc:
        logger.warning("Could not load SPY data for relative features: %s", exc)

    logger.info("Predicting for %d symbols using %d models...", len(symbols), len(models))
    predictions = [
        predict_symbol(s, models, db=db, history_days=cfg.data.history_days, spy_df=spy_df)
        for s in symbols
    ]

    # --- Drift detection ---
    drift_warnings: dict = {}
    if args.history:
        try:
            today_mean_scores: dict = {}
            for pred in predictions:
                if "error" in pred:
                    continue
                for model_key, model_pred in pred.get("predictions", {}).items():
                    if "signal" not in model_pred:
                        continue
                    score = _signal_to_score(model_pred["signal"], model_pred.get("confidence", 0.0))
                    today_mean_scores.setdefault(model_key, []).append(score)
            today_mean_scores = {m: float(np.mean(s)) for m, s in today_mean_scores.items()}
            drift_warnings = check_signal_drift(args.history, prediction_date, today_mean_scores)
            for model, warned in drift_warnings.items():
                if warned:
                    logger.warning("Signal drift detected for %s", model)
        except Exception as exc:
            logger.warning("Drift detection failed: %s", exc)

    # Summary
    print("\n" + "=" * 50)
    print("DAILY PREDICTIONS")
    print("=" * 50)
    for pred in predictions:
        if "error" in pred:
            print(f"  {pred['symbol']}: ERROR — {pred['error']}")
            continue
        parts = []
        for model_key, mp in pred.get("predictions", {}).items():
            if "signal" in mp:
                parts.append(f"{model_key}={mp['signal']} ({mp['confidence']:.0%})")
            elif "error" in mp:
                parts.append(f"{model_key}=ERROR")
        print(f"  {pred['symbol']} (${pred.get('price', 0):.2f}): {' | '.join(parts)}")

    # Write JSON artifact (for GitHub Actions)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info("Saved predictions to %s", args.output)

    # Append to durable predictions history (separate from the ephemeral DB)
    if args.history:
        n = append_predictions_history(predictions, args.history)
        logger.info("Appended %d records to %s", n, args.history)

    # Discord notification
    webhook = os.environ.get("DISCORD_WEBHOOK_URL") or os.environ.get("WEBHOOK_URL")
    if webhook:
        if send_discord(predictions, webhook, ic_results=ic_results, drift_warnings=drift_warnings):
            logger.info("Discord notification sent.")
        else:
            logger.warning("Discord notification failed.")


if __name__ == "__main__":
    main()
