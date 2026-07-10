#!/usr/bin/env python3
"""
train_predictor.py — Train the daily return *prediction* model.

This is the prediction half of the prediction/strategy split: a regression
model that forecasts continuous forward return (fwd_ret_1d, a
FWD_RET_HORIZON_DAYS-bar cumulative return), evaluated by forecast-quality
metrics (Spearman IC, R^2, directional accuracy) — not classification
accuracy.

Why a separate model from daily_logistic/daily_xgboost: those models predict
the *discretized* SELL/HOLD/BUY action directly, which bakes a decision
threshold (vol_mult) into the training target itself. That conflates two
different problems — forecasting price movement (continuous, naturally a
regression task) and deciding what to do about a forecast (a policy that can
trade off conviction, cost, and risk independently of how the forecast was
produced). See models/README.md "Prediction vs. strategy" for the empirical
case for this split: regression on the same 25-feature set recovers a small
positive Spearman IC (~0.05-0.06 with Ridge) that the 3-class classifiers
could not detect at all once the train/test leakage fix was applied.

The actual SELL/HOLD/BUY decision is made by DailyPredictorStrategy
(ml_strategies.py), which loads this model's continuous predictions and
applies an independently-tunable threshold — so the strategy can be retuned
(or replaced entirely) without retraining the prediction model.

Usage:
    python train_predictor.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 2500
"""
from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from daily_features import FEATURE_COLS, FEATURE_SET_NAME, FWD_RET_HORIZON_DAYS, make_daily_features
from db import DB
from train_models import _load_symbol, _preprocess
from walk_forward import sweep_params, WalkForwardConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_KEY = "daily_predictor"


# ---------------------------------------------------------------------------
# Data prep — same purged/embargoed per-symbol split as train_models.py, but
# the target is continuous fwd_ret_1d instead of a discretized class.
# ---------------------------------------------------------------------------

def prepare_data(symbols: list[str], days: int, db: DB) -> dict:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    train_Xs, train_ys, train_vols = [], [], []
    test_Xs, test_ys, test_vols = [], [], []
    used_symbols: list[str] = []

    spy_df = _load_symbol("SPY", start, end, db)
    if spy_df is None:
        logger.warning("Could not load SPY data — ret_*_vs_spy features will be 0")

    for symbol in symbols:
        df = _load_symbol(symbol, start, end, db)
        if df is None:
            continue
        try:
            spy_arg = spy_df if symbol != "SPY" else None
            feats = make_daily_features(df, spy_df=spy_arg)
        except Exception as exc:
            logger.warning("  %s: feature computation failed: %s", symbol, exc)
            continue

        feats = feats.dropna(subset=["fwd_ret_1d"])
        if len(feats) < 50:
            logger.warning("  %s: too few valid rows (%d) after dropna", symbol, len(feats))
            continue

        X_sym = feats[FEATURE_COLS].values.astype(np.float32)
        y_sym = feats["fwd_ret_1d"].values.astype(np.float64)
        vol_sym = feats["vol_20d"].values.astype(np.float64)

        split = int(len(X_sym) * 0.8)
        test_start = split + FWD_RET_HORIZON_DAYS
        if test_start >= len(X_sym):
            logger.warning("  %s: too few rows for embargo gap, skipping", symbol)
            continue

        X_tr = _preprocess(X_sym[:split].copy())
        X_te = _preprocess(X_sym[test_start:].copy())

        train_Xs.append(X_tr); train_ys.append(y_sym[:split]); train_vols.append(vol_sym[:split])
        test_Xs.append(X_te); test_ys.append(y_sym[test_start:]); test_vols.append(vol_sym[test_start:])
        used_symbols.append(symbol)
        logger.info(
            "  %s: %d samples (train=%d embargo=%d test=%d)",
            symbol, len(y_sym), split, FWD_RET_HORIZON_DAYS, len(y_sym) - test_start,
        )

    if not train_Xs:
        raise RuntimeError("No usable training data across all symbols.")

    return {
        "X_train": np.vstack(train_Xs), "y_train": np.concatenate(train_ys),
        "vol_train": np.concatenate(train_vols),
        "X_test": np.vstack(test_Xs), "y_test": np.concatenate(test_ys),
        "vol_test": np.concatenate(test_vols),
        "used_symbols": used_symbols,
    }


# ---------------------------------------------------------------------------
# Forecast-quality metrics — NOT classification accuracy.
# ---------------------------------------------------------------------------

def _forecast_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    if np.std(pred) < 1e-12:
        ic = 0.0  # constant predictions carry no rank information
    else:
        ic, _ = spearmanr(pred, actual)
        ic = 0.0 if np.isnan(ic) else float(ic)
    return {
        "ic": ic,
        "dir_acc": float(np.mean(np.sign(pred) == np.sign(actual))),
        "r2": float(r2_score(actual, pred)),
        "mae": float(mean_absolute_error(actual, pred)),
    }


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------

def train_elasticnet(X_train, X_test, y_train, y_test, alpha: float, l1_ratio: float):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
    model.fit(X_tr, y_train)
    train_metrics = _forecast_metrics(model.predict(X_tr), y_train)
    test_metrics = _forecast_metrics(model.predict(X_te), y_test)
    return model, scaler, train_metrics, test_metrics


def train_xgb_regressor(X_train, X_test, y_train, y_test):
    if not HAS_XGBOOST:
        raise ImportError("xgboost not installed.")
    scaler = StandardScaler()  # kept for artifact-shape consistency; XGBoost is scale-invariant
    scaler.fit(X_train)
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.7, min_child_weight=5,
        gamma=1.0, random_state=42, tree_method="hist", verbosity=0,
    )
    model.fit(X_train, y_train)
    train_metrics = _forecast_metrics(model.predict(X_train), y_train)
    test_metrics = _forecast_metrics(model.predict(X_test), y_test)
    return model, scaler, train_metrics, test_metrics


def train_ridge(X_train, X_test, y_train, y_test, alpha: float):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_train)
    train_metrics = _forecast_metrics(model.predict(X_tr), y_train)
    test_metrics = _forecast_metrics(model.predict(X_te), y_test)
    return model, scaler, train_metrics, test_metrics


def train_elasticnet(X_train, X_test, y_train, y_test, alpha: float, l1_ratio: float):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
    model.fit(X_tr, y_train)
    train_metrics = _forecast_metrics(model.predict(X_tr), y_train)
    test_metrics = _forecast_metrics(model.predict(X_te), y_test)
    return model, scaler, train_metrics, test_metrics


def _sweep_model_params(symbols: list[str], days: int, db) -> tuple[float, float]:
    """Walk-forward sweep of alpha and l1_ratio. Returns (best_alpha, best_l1_ratio)."""
    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    from datetime import datetime, timedelta
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    spy_df = _load_symbol("SPY", start, end, db)

    fold_data = []
    for symbol in symbols:
        df = _load_symbol(symbol, start, end, db)
        if df is None:
            continue
        try:
            spy_arg = spy_df if symbol != "SPY" else None
            feats = make_daily_features(df, spy_df=spy_arg).dropna(subset=["fwd_ret_1d"])
        except Exception:
            continue
        n = len(feats)
        config = WalkForwardConfig()
        if n < config.train_bars + FWD_RET_HORIZON_DAYS + config.test_bars:
            continue
        X_all = _preprocess(feats[FEATURE_COLS].values.astype(np.float32))
        y_all = feats["fwd_ret_1d"].values.astype(np.float64)
        train_start_idx = 0
        while True:
            train_end_idx = train_start_idx + config.train_bars
            test_start_idx = train_end_idx + FWD_RET_HORIZON_DAYS
            test_end_idx = test_start_idx + config.test_bars
            if test_end_idx > n:
                break
            fold_data.append((X_all[train_start_idx:train_end_idx], y_all[train_start_idx:train_end_idx],
                             X_all[test_start_idx:test_end_idx], y_all[test_start_idx:test_end_idx]))
            train_start_idx += config.step_bars

    if not fold_data:
        logger.warning("_sweep_model_params: no folds — returning defaults")
        return 1.0, 0.5

    best_ic = -np.inf
    best_pair = (1.0, 0.5)
    for alpha in alphas:
        for l1r in l1_ratios:
            fold_ics = []
            for X_tr, y_tr, X_te, y_te in fold_data:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                model = ElasticNet(alpha=alpha, l1_ratio=l1r, max_iter=5000, random_state=42)
                model.fit(X_tr_s, y_tr)
                pred = model.predict(X_te_s)
                if np.std(pred) < 1e-12:
                    continue
                ic, _ = spearmanr(pred, y_te)
                if not np.isnan(ic):
                    fold_ics.append(float(ic))
            if not fold_ics:
                continue
            median_ic = float(np.median(fold_ics))
            if median_ic > best_ic:
                best_ic = median_ic
                best_pair = (alpha, l1r)
    logger.info("_sweep_model_params: best alpha=%.4f l1_ratio=%.2f median_IC=%.4f",
                best_pair[0], best_pair[1], best_ic)
    return best_pair


# ---------------------------------------------------------------------------
# Save / register
# ---------------------------------------------------------------------------

def _save_and_register(
    model_obj, scaler, model_type: str, train_metrics: dict, test_metrics: dict,
    train_symbols: list[str], days: int, db: DB, train_samples: int, test_samples: int,
    best_signal_quantile: float = 0.7,
    best_threshold_window: int = 60,
    best_model_alpha: float = 0.0,
    best_l1_ratio: float = 0.0,
) -> int:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # model_registry has columns shaped for classifiers (test_accuracy, test_f1);
    # repurposed here for directional accuracy and IC respectively rather than
    # adding regression-specific columns for a single model type.
    version = db.register_model(
        model_key=MODEL_KEY,
        artifact_path="PLACEHOLDER",
        feature_contract=FEATURE_COLS,
        trained_on=train_symbols,
        train_start=start,
        train_end=end,
        train_samples=train_samples,
        test_samples=test_samples,
        train_accuracy=train_metrics["dir_acc"],
        test_accuracy=test_metrics["dir_acc"],
        test_f1=test_metrics["ic"],
        label_map=None,
        feature_set_name=FEATURE_SET_NAME,
    )

    Path("models").mkdir(exist_ok=True)
    artifact_path = f"models/{MODEL_KEY}_v{version}.pkl"
    artifact = {
        "model": model_obj,
        "scaler": scaler,
        "model_type": model_type,
        "feature_contract": FEATURE_COLS,
        "feature_set_name": FEATURE_SET_NAME,
        "trained_at": datetime.now().isoformat(),
        "train_symbols": train_symbols,
        "train_ic": train_metrics["ic"],
        "train_dir_acc": train_metrics["dir_acc"],
        "train_r2": train_metrics["r2"],
        "train_mae": train_metrics["mae"],
        "test_ic": test_metrics["ic"],
        "test_dir_acc": test_metrics["dir_acc"],
        "test_r2": test_metrics["r2"],
        "test_mae": test_metrics["mae"],
        # Generic aliases so this artifact can be sanity-checked alongside the
        # classifier artifacts without special-casing every consumer.
        "test_accuracy": test_metrics["dir_acc"],
        "test_f1": test_metrics["ic"],
        "best_signal_quantile": best_signal_quantile,
        "best_threshold_window": best_threshold_window,
        "best_model_alpha": best_model_alpha,
        "best_l1_ratio": best_l1_ratio,
    }
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)

    canonical = f"models/{MODEL_KEY}.pkl"
    with open(canonical, "wb") as f:
        pickle.dump(artifact, f)

    db.update_artifact_path(MODEL_KEY, version, artifact_path)
    db.deactivate_old_models(MODEL_KEY, version)

    logger.info(
        "%s v%d [%s]: train_ic=%.4f test_ic=%.4f test_dir_acc=%.4f test_r2=%+.5f -> %s",
        MODEL_KEY, version, model_type, train_metrics["ic"], test_metrics["ic"],
        test_metrics["dir_acc"], test_metrics["r2"], artifact_path,
    )
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the daily return prediction model")
    parser.add_argument(
        "--symbols", default="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM",
        help="Comma-separated list of symbols",
    )
    parser.add_argument("--days", type=int, default=2500, help="Days of history to use")
    parser.add_argument(
        "--model", choices=["ridge", "xgboost", "elasticnet", "both"], default="elasticnet",
        help="Which regressor to train.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Regularization strength")
    parser.add_argument("--l1-ratio", type=float, default=0.5, help="ElasticNet L1 ratio (0=Ridge, 1=Lasso)")
    parser.add_argument("--db", default="data/trading_sim.db", help="Path to SQLite DB")
    parser.add_argument("--sweep-model", action="store_true", help="Walk-forward sweep model hyperparams")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    logger.info("Training symbols: %s", symbols)
    logger.info("History: %d days", args.days)

    db = DB(args.db)
    logger.info("Collecting and preparing data...")
    data = prepare_data(symbols, args.days, db)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    logger.info(
        "Total samples: %d  Features: %d  Train: %d  Test: %d",
        len(X_train) + len(X_test), X_train.shape[1], len(X_train), len(X_test),
    )

    best_alpha = args.alpha
    best_l1_ratio = args.l1_ratio
    if args.sweep_model and args.model in ("elasticnet", "both", "ridge"):
        logger.info("Running walk-forward model-parameter sweep...")
        best_alpha, best_l1_ratio = _sweep_model_params(symbols, args.days, db)
        logger.info("Best model params: alpha=%.4f l1_ratio=%.2f", best_alpha, best_l1_ratio)

    logger.info("Running walk-forward parameter sweep...")
    try:
        wf_config = WalkForwardConfig(ridge_alpha=best_alpha)
        best_q, best_w = sweep_params(symbols, args.days, db, wf_config)
        logger.info("Parameter sweep complete: signal_quantile=%.2f threshold_window=%d", best_q, best_w)
    except Exception as exc:
        logger.warning("Parameter sweep failed: %s — using defaults (0.7, 60)", exc)
        best_q, best_w = 0.7, 60

    models_to_train = ["ridge", "xgboost"] if args.model == "both" else [args.model]

    for model_type in models_to_train:
        logger.info("--- Training %s ---", model_type)
        if model_type == "ridge":
            model, scaler, train_m, test_m = train_ridge(X_train, X_test, y_train, y_test, best_alpha)
            ma, lr = best_alpha, 0.0
        elif model_type == "elasticnet":
            model, scaler, train_m, test_m = train_elasticnet(X_train, X_test, y_train, y_test, best_alpha, best_l1_ratio)
            ma, lr = best_alpha, best_l1_ratio
        else:
            model, scaler, train_m, test_m = train_xgb_regressor(X_train, X_test, y_train, y_test)
            ma, lr = 0.0, 0.0
        _save_and_register(
            model, scaler, model_type, train_m, test_m,
            data["used_symbols"], args.days, db,
            train_samples=len(X_train), test_samples=len(X_test),
            best_signal_quantile=best_q,
            best_threshold_window=best_w,
            best_model_alpha=ma,
            best_l1_ratio=lr,
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
