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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_SIGNAL_NAMES = ["SELL", "HOLD", "BUY"]  # index matches label {0,1,2}


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


def load_models(db: Optional[DB] = None) -> dict:
    """
    Load all available daily model pickles.

    Uses DB model_registry to resolve artifact_path when available.
    Falls back to canonical paths (models/daily_logistic.pkl, etc.) otherwise.
    """
    models = {}

    def _resolve_path(model_key: str, canonical: str) -> str:
        if db is not None:
            meta = db.get_active_model(model_key)
            if meta and os.path.exists(meta["artifact_path"]):
                return meta["artifact_path"]
        return canonical

    for model_key, canonical in [
        ("daily_logistic", "models/daily_logistic.pkl"),
        ("daily_xgboost", "models/daily_xgboost.pkl"),
    ]:
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
        feats = make_daily_features(df)
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

    # --- DailyLogistic ---
    if "daily_logistic" in models:
        try:
            data = models["daily_logistic"]
            X_scaled = data["scaler"].transform(X_latest)
            prob = data["model"].predict_proba(X_scaled)[0]
            pred_idx = int(np.argmax(prob))
            confidence = float(prob[pred_idx])
            threshold = data.get("confidence_threshold", 0.55)
            signal = _SIGNAL_NAMES[pred_idx]
            if signal != "HOLD" and confidence < threshold:
                signal = "HOLD"
            result["predictions"]["daily_logistic"] = {
                "signal": signal,
                "confidence": confidence,
            }
            if db is not None:
                meta = db.get_active_model("daily_logistic")
                version = meta["version"] if meta else 0
                db.upsert_prediction(
                    symbol, "daily_logistic", version, prediction_date,
                    signal, confidence, result["price"]
                )
        except Exception as exc:
            result["predictions"]["daily_logistic"] = {"error": str(exc)}

    # --- DailyXGBoost ---
    if "daily_xgboost" in models:
        try:
            data = models["daily_xgboost"]
            X_scaled = data["scaler"].transform(X_latest)
            prob = data["model"].predict_proba(X_scaled)[0]
            pred_idx = int(np.argmax(prob))
            confidence = float(prob[pred_idx])
            threshold = data.get("confidence_threshold", 0.55)
            signal = _SIGNAL_NAMES[pred_idx]
            if signal != "HOLD" and confidence < threshold:
                signal = "HOLD"
            result["predictions"]["daily_xgboost"] = {
                "signal": signal,
                "confidence": confidence,
            }
            if db is not None:
                meta = db.get_active_model("daily_xgboost")
                version = meta["version"] if meta else 0
                db.upsert_prediction(
                    symbol, "daily_xgboost", version, prediction_date,
                    signal, confidence, result["price"]
                )
        except Exception as exc:
            result["predictions"]["daily_xgboost"] = {"error": str(exc)}

    # --- DailyDQN ---
    if "daily_dqn" in models:
        try:
            agent = models["daily_dqn"]
            n_rows, n_cols = X_all.shape
            if n_rows >= dqn_window:
                state = X_all[-dqn_window:].flatten()
            else:
                pad = np.zeros(dqn_window * n_cols - X_all.size, dtype=np.float32)
                state = np.concatenate([X_all.flatten(), pad])

            with torch.no_grad():
                s_t = torch.from_numpy(state).float().unsqueeze(0)
                q_vals = agent.q(s_t).squeeze(0).cpu().numpy()

            # Action space: 0=Hold, 1=Long, 2=Short
            pred = int(np.argmax(q_vals))
            action_to_signal = {0: "HOLD", 1: "BUY", 2: "SELL"}
            signal = action_to_signal[pred]

            sorted_q = np.sort(q_vals)
            q_margin = float(sorted_q[-1] - sorted_q[-2])
            confidence = float(np.clip((q_margin + 1.0) / 2.0, 0.0, 1.0))

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
# Discord notification
# ---------------------------------------------------------------------------

def send_discord(predictions: list, webhook_url: str) -> bool:
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

    if not embeds:
        logger.warning("No embeds to send to Discord")
        return False

    header = (
        f"**Daily Trading Predictions** — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        "Results from daily ML models"
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

    logger.info("Predicting for %d symbols using %d models...", len(symbols), len(models))
    predictions = [predict_symbol(s, models, db=db, history_days=cfg.data.history_days) for s in symbols]

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

    # Discord notification
    webhook = os.environ.get("DISCORD_WEBHOOK_URL") or os.environ.get("WEBHOOK_URL")
    if webhook:
        if send_discord(predictions, webhook):
            logger.info("Discord notification sent.")
        else:
            logger.warning("Discord notification failed.")


if __name__ == "__main__":
    main()
