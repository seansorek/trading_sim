#!/usr/bin/env python3
"""
train_models.py — Single unified training entry point.

Trains DailyLogistic and/or DailyXGBoost models on historical daily data
and registers them in the SQLite model registry.

Usage:
    python train_models.py --symbols AAPL,MSFT,SPY --days 1000 --models logistic,xgboost
    python train_models.py --symbols AAPL,MSFT,SPY --days 1000 --models xgboost --optimize
"""
import argparse
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

from data_loader import load_yfinance
from daily_features import (
    FEATURE_COLS,
    FEATURE_SET_NAME,
    discretize_labels,
    make_daily_features,
)
from db import DB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train_models.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _preprocess(X: np.ndarray) -> np.ndarray:
    """Replace inf/nan with 0; clip to ±5 std per column."""
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X, nan=0.0)
    for col in range(X.shape[1]):
        col_data = X[:, col]
        std = np.std(col_data)
        if std > 0:
            mean = np.mean(col_data)
            X[:, col] = np.clip(col_data, mean - 5 * std, mean + 5 * std)
    return X


def _load_symbol(
    symbol: str, start: str, end: str, db: DB
) -> pd.DataFrame | None:
    """Fetch daily bars, cache in DB, return DataFrame."""
    cached = db.load_bars(symbol, "1d", start, end)
    if cached is not None and len(cached) >= 50:
        logger.info("  %s: loaded %d bars from DB cache", symbol, len(cached))
        return cached

    logger.info("  %s: fetching from yfinance...", symbol)
    try:
        df = load_yfinance(symbol, start=start, end=end, interval="1d")
    except Exception as exc:
        logger.warning("  %s: fetch failed: %s", symbol, exc)
        return None

    if df is None or len(df) < 50:
        logger.warning("  %s: insufficient data (%s rows)", symbol, len(df) if df is not None else 0)
        return None

    db.upsert_bars(symbol, "1d", df)
    return df


def _prepare_data(
    symbols: list[str], days: int, db: DB
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Collect and pool training data across symbols.

    Returns (X, y, symbol_list_used).
    Uses temporal ordering: data is sorted by (symbol, date) and NOT shuffled,
    preserving the time structure needed for the train/test split.
    """
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    used_symbols: list[str] = []

    for symbol in symbols:
        df = _load_symbol(symbol, start, end, db)
        if df is None:
            continue

        try:
            feats = make_daily_features(df)
        except Exception as exc:
            logger.warning("  %s: feature computation failed: %s", symbol, exc)
            continue

        # Drop rows where forward return is not available
        feats = feats.dropna(subset=["fwd_ret_1d"])
        if len(feats) < 50:
            logger.warning("  %s: too few valid rows (%d) after dropna", symbol, len(feats))
            continue

        X_sym = _preprocess(feats[FEATURE_COLS].values.astype(np.float32))
        y_sym = discretize_labels(feats["fwd_ret_1d"].values)

        all_X.append(X_sym)
        all_y.append(y_sym)
        used_symbols.append(symbol)
        logger.info("  %s: %d samples, class dist %s", symbol, len(y_sym), np.bincount(y_sym).tolist())

    if not all_X:
        raise RuntimeError("No usable training data across all symbols.")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    return X, y, used_symbols


def _temporal_split(
    X: np.ndarray, y: np.ndarray, test_frac: float = 0.20
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Temporal train/test split — first (1-test_frac) rows for train,
    last test_frac rows for test.

    Avoids data leakage from shuffling time-series data.
    """
    split = int(len(X) * (1 - test_frac))
    return X[:split], X[split:], y[:split], y[split:]


def _make_artifact_path(model_key: str, version: int) -> str:
    Path("models").mkdir(exist_ok=True)
    return f"models/{model_key}_v{version}.pkl"


def _save_and_register(
    model_key: str,
    model_obj,
    scaler: StandardScaler,
    train_accuracy: float,
    test_accuracy: float,
    test_f1: float,
    confidence_threshold: float,
    train_symbols: list[str],
    days: int,
    db: DB,
) -> int:
    """Save pickle and register in DB. Returns new version number."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Reserve version number from DB first so we can name the file
    version = db.register_model(
        model_key=model_key,
        artifact_path="PLACEHOLDER",
        feature_contract=FEATURE_COLS,
        trained_on=train_symbols,
        train_start=start,
        train_end=end,
        train_samples=0,
        test_samples=0,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        test_f1=test_f1,
        label_map=LABEL_MAP,
        feature_set_name=FEATURE_SET_NAME,
    )

    artifact_path = _make_artifact_path(model_key, version)
    artifact = {
        "model": model_obj,
        "scaler": scaler,
        "feature_contract": FEATURE_COLS,
        "feature_set_name": FEATURE_SET_NAME,
        "label_map": LABEL_MAP,
        "confidence_threshold": confidence_threshold,
        "trained_at": datetime.now().isoformat(),
        "train_symbols": train_symbols,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
    }
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)

    # Also write to the canonical path (no version suffix) for backward compat
    canonical = f"models/{model_key}.pkl"
    with open(canonical, "wb") as f:
        pickle.dump(artifact, f)

    # Update artifact_path in DB now that we know it
    db.deactivate_old_models(model_key, version)

    logger.info(
        "%s v%d: train_acc=%.3f test_acc=%.3f f1=%.3f -> %s",
        model_key, version, train_accuracy, test_accuracy, test_f1, artifact_path,
    )
    return version


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------

def train_logistic(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
) -> tuple[LogisticRegression, StandardScaler, float, float, float]:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = LogisticRegression(
        C=cfg.get("C", 1.0),
        max_iter=cfg.get("max_iter", 1000),
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        multi_class="multinomial",
    )
    model.fit(X_tr, y_train)

    train_acc = float(accuracy_score(y_train, model.predict(X_tr)))
    test_acc = float(accuracy_score(y_test, model.predict(X_te)))
    test_f1 = float(f1_score(y_test, model.predict(X_te), average="macro", zero_division=0))

    logger.info(
        "Logistic — train_acc=%.3f  test_acc=%.3f  f1=%.3f",
        train_acc, test_acc, test_f1,
    )
    return model, scaler, train_acc, test_acc, test_f1


def train_xgboost(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
    optimize: bool = False,
) -> tuple:
    if not HAS_XGBOOST:
        raise ImportError("xgboost not installed. pip install xgboost")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    if optimize and HAS_SKOPT:
        logger.info("Running Bayesian hyperparameter search for XGBoost...")
        search_space = {
            "n_estimators": Integer(100, 500),
            "max_depth": Integer(2, 6),
            "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
            "subsample": Real(0.6, 1.0),
            "colsample_bytree": Real(0.6, 1.0),
            "gamma": Real(0.0, 5.0),
            "min_child_weight": Integer(1, 10),
        }
        base = xgb.XGBClassifier(
            random_state=42, tree_method="hist", verbosity=0,
            objective="multi:softprob", num_class=3,
        )
        opt = BayesSearchCV(base, search_space, n_iter=20, cv=3, n_jobs=-1,
                            scoring="f1_macro", random_state=42)
        opt.fit(X_tr, y_train)
        model = opt.best_estimator_
        logger.info("Best params: %s", opt.best_params_)
    else:
        model = xgb.XGBClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", 4),
            learning_rate=cfg.get("learning_rate", 0.03),
            subsample=cfg.get("subsample", 0.85),
            colsample_bytree=cfg.get("colsample_bytree", 0.85),
            min_child_weight=cfg.get("min_child_weight", 2),
            gamma=cfg.get("gamma", 1.0),
            random_state=42,
            tree_method="hist",
            verbosity=0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
        )
        # Class weights to counteract class imbalance
        class_counts = np.bincount(y_train, minlength=3)
        total = len(y_train)
        sample_weights = np.array([
            total / (3 * class_counts[y]) for y in y_train
        ])
        model.fit(X_tr, y_train, sample_weight=sample_weights)

    train_acc = float(accuracy_score(y_train, model.predict(X_tr)))
    test_acc = float(accuracy_score(y_test, model.predict(X_te)))
    test_f1 = float(f1_score(y_test, model.predict(X_te), average="macro", zero_division=0))

    logger.info(
        "XGBoost — train_acc=%.3f  test_acc=%.3f  f1=%.3f",
        train_acc, test_acc, test_f1,
    )
    return model, scaler, train_acc, test_acc, test_f1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train daily ML models")
    parser.add_argument(
        "--symbols", default="AAPL,MSFT,GOOGL,AMZN,SPY,QQQ",
        help="Comma-separated list of symbols"
    )
    parser.add_argument("--days", type=int, default=1000, help="Days of history to use")
    parser.add_argument(
        "--models", default="logistic,xgboost",
        help="Comma-separated list of models to train: logistic, xgboost"
    )
    parser.add_argument("--optimize", action="store_true", help="Run Bayesian hyperparameter search")
    parser.add_argument("--db", default="data/trading_sim.db", help="Path to SQLite DB")
    parser.add_argument("--confidence", type=float, default=0.55, help="Confidence threshold")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    models_to_train = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    logger.info("Training symbols: %s", symbols)
    logger.info("Models: %s", models_to_train)
    logger.info("History: %d days", args.days)

    db = DB(args.db)

    logger.info("Collecting and preparing data...")
    X, y, used_symbols = _prepare_data(symbols, args.days, db)
    logger.info("Total samples: %d  Features: %d", len(X), X.shape[1])
    logger.info("Class distribution: %s", np.bincount(y).tolist())

    X_train, X_test, y_train, y_test = _temporal_split(X, y)
    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))

    if "logistic" in models_to_train:
        logger.info("--- Training DailyLogistic ---")
        from config import get_config
        cfg = get_config()
        logistic_cfg = {
            "C": cfg.strategies.logistic.C,
            "max_iter": cfg.strategies.logistic.max_iter,
        }
        model, scaler, train_acc, test_acc, test_f1 = train_logistic(
            X_train, X_test, y_train, y_test, logistic_cfg
        )
        _save_and_register(
            model_key="daily_logistic",
            model_obj=model,
            scaler=scaler,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            test_f1=test_f1,
            confidence_threshold=args.confidence,
            train_symbols=used_symbols,
            days=args.days,
            db=db,
        )

    if "xgboost" in models_to_train:
        logger.info("--- Training DailyXGBoost ---")
        from config import get_config
        cfg = get_config()
        xgb_cfg = {
            "n_estimators": cfg.strategies.xgboost.n_estimators,
            "max_depth": cfg.strategies.xgboost.max_depth,
            "learning_rate": cfg.strategies.xgboost.learning_rate,
            "subsample": cfg.strategies.xgboost.subsample,
            "colsample_bytree": cfg.strategies.xgboost.colsample_bytree,
            "min_child_weight": cfg.strategies.xgboost.min_child_weight,
            "gamma": cfg.strategies.xgboost.gamma,
        }
        model, scaler, train_acc, test_acc, test_f1 = train_xgboost(
            X_train, X_test, y_train, y_test, xgb_cfg, optimize=args.optimize
        )
        _save_and_register(
            model_key="daily_xgboost",
            model_obj=model,
            scaler=scaler,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            test_f1=test_f1,
            confidence_threshold=args.confidence,
            train_symbols=used_symbols,
            days=args.days,
            db=db,
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
