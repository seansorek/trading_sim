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
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from daily_features import (
    CS_MODES,
    FEATURE_COLS,
    FEATURE_SET_NAME,
    FWD_RET_HORIZON_DAYS,
    cross_sectional_normalize,
    cs_feature_cols,
    make_daily_features,
)
from db import DB
from predictors.base import CLIP, _preprocess
from train_models import _load_symbol, _pickle_and_hash
from walk_forward import sweep_params, WalkForwardConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_KEY = "daily_predictor"


# ---------------------------------------------------------------------------
# Data prep — same purged/embargoed per-symbol split as train_models.py, but
# the target is continuous fwd_ret_1d instead of a discretized class.
# ---------------------------------------------------------------------------

# A date needs at least this many symbols for its cross-sectional mean to be a
# usable estimate of the common component. Thinner dates are dropped when
# demeaning rather than demeaned against 2-3 names of noise.
MIN_DEMEAN_NAMES = 5


def prepare_data(
    symbols: list[str],
    days: int,
    db: DB,
    demean: bool = False,
    cs_mode: str = "off",
    train_frac: float = 0.8,
    train_end: str | None = None,
) -> dict:
    """Assemble the (date, symbol) panel and split it on a GLOBAL date.

    The split is by calendar date, not by each symbol's own row count. A
    per-symbol 80/20 split puts symbol A's test dates inside symbol B's train
    window whenever their histories differ in length - tolerable for a
    per-symbol time-series model, but for a cross-sectional one it leaks
    precisely the common factor the model is supposed to rank against.

    demean=True subtracts each date's cross-sectional mean from the target, so
    the model forecasts return-relative-to-universe instead of absolute return.
    That is what a rank-based book actually trades: rank_to_weights is
    location-invariant and discards the common component, so fitting it spends
    capacity on a signal the book throws away. Demeaning uses only same-date
    rows, so it moves no information across the train/test boundary.

    cs_mode does the same to the *inputs*: "replace" swaps each feature for its
    per-date rank across the universe, "augment" keeps both axes (see
    daily_features.cross_sectional_normalize). Like demeaning these are
    same-date transforms, so applying them before the split leaks nothing.
    """
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    spy_df = _load_symbol("SPY", start, end, db)
    if spy_df is None:
        logger.warning("Could not load SPY data — ret_*_vs_spy features will be 0")

    frames: list[pd.DataFrame] = []
    used_symbols: list[str] = []

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

        block = feats[FEATURE_COLS].copy()
        block["_y"] = feats["fwd_ret_1d"].astype(np.float64)
        block["_vol"] = feats["vol_20d"].astype(np.float64)
        block["_date"] = feats.index
        block["_symbol"] = symbol
        frames.append(block)
        used_symbols.append(symbol)

    if not frames:
        raise RuntimeError("No usable training data across all symbols.")

    panel = pd.concat(frames, ignore_index=True)

    model_cols = cs_feature_cols(cs_mode)
    if cs_mode != "off":
        before = len(panel)
        panel = cross_sectional_normalize(panel, mode=cs_mode).dropna(subset=model_cols)
        logger.info(
            "Feature axis %r: %d columns, %d rows (%d dropped as thin dates)",
            cs_mode, len(model_cols), len(panel), before - len(panel),
        )

    if demean:
        n_names = panel.groupby("_date")["_y"].transform("size")
        thin = n_names < MIN_DEMEAN_NAMES
        if thin.any():
            logger.info(
                "Dropping %d rows on dates with < %d symbols (too thin to demean)",
                int(thin.sum()), MIN_DEMEAN_NAMES,
            )
            panel = panel[~thin].copy()
        panel["_y"] = panel["_y"] - panel.groupby("_date")["_y"].transform("mean")

    dates = pd.DatetimeIndex(np.sort(panel["_date"].unique()))
    if train_end is None:
        cut = int(len(dates) * train_frac)
    else:
        # Explicit cutoff. train_frac slides whenever the history window moves,
        # which is useless for a holdout that has to sit on fixed calendar dates.
        ts = pd.Timestamp(train_end)
        if dates.tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(dates.tz)
        cut = int(dates.searchsorted(ts))
    if cut < 1 or cut + FWD_RET_HORIZON_DAYS >= len(dates):
        raise RuntimeError(f"Only {len(dates)} dates — too few for a split plus embargo.")
    # Half-open bounds matching the original per-symbol split: train is rows
    # [0, cut), test is [cut + H, n), and the H dates between them are dropped.
    # A train label on the last train date looks FWD_RET_HORIZON_DAYS ahead, so
    # test cannot start until past that window.
    split_date = dates[cut]
    test_start_date = dates[cut + FWD_RET_HORIZON_DAYS]

    train = panel[panel["_date"] < split_date]
    test = panel[panel["_date"] >= test_start_date]
    if train.empty or test.empty:
        raise RuntimeError("Global date split produced an empty side.")

    logger.info(
        "Global date split: train < %s (%d rows), embargo %d dates, test >= %s (%d rows)",
        pd.Timestamp(split_date).date(), len(train), FWD_RET_HORIZON_DAYS,
        pd.Timestamp(test_start_date).date(), len(test),
    )

    return {
        "X_train": _preprocess(train[model_cols].values.astype(np.float32)),
        "y_train": train["_y"].values,
        "vol_train": train["_vol"].values,
        "X_test": _preprocess(test[model_cols].values.astype(np.float32)),
        "y_test": test["_y"].values,
        "vol_test": test["_vol"].values,
        "train_dates": train["_date"].values,
        "test_dates": test["_date"].values,
        "test_symbols": test["_symbol"].values,
        "used_symbols": used_symbols,
        "demeaned": demean,
        "cs_mode": cs_mode,
        "feature_cols": model_cols,
        "train_start_date": str(pd.Timestamp(dates[0]).date()),
        # The day before the first test date, not the last training date.
        # oos_guard adds its embargo in *calendar* days while the split above
        # purges FWD_RET_HORIZON_DAYS *trading* dates, so a weekend could let
        # rows inside the last training label's horizon back into a backtest.
        # Recording the conservative boundary makes that impossible; the cost
        # is trimming at most two extra days from a holdout.
        "train_end_date": str((pd.Timestamp(test_start_date) - pd.Timedelta(days=1)).date()),
    }


# ---------------------------------------------------------------------------
# Forecast-quality metrics — NOT classification accuracy.
# ---------------------------------------------------------------------------

def cross_sectional_metrics(
    pred: np.ndarray, actual: np.ndarray, dates: np.ndarray, symbols: np.ndarray
) -> dict:
    """Per-date IC over the test panel — the metric a rank-based book trades.

    The pooled `ic` from _forecast_metrics stacks every (date, symbol) sample
    into one correlation and is dominated by time-series variation: on the live
    panel it read 3.3x the per-date IC, all of it market direction that
    rank_to_weights discards. Reuses panel_eval.cross_sectional_ic so training
    and the panel backtest measure ranking skill the same way.
    """
    from panel_eval import cross_sectional_ic  # deferred: keeps the import graph acyclic

    long = pd.DataFrame({"date": dates, "symbol": symbols, "pred": pred, "actual": actual})
    pred_wide = long.pivot_table(index="date", columns="symbol", values="pred")
    actual_wide = long.pivot_table(index="date", columns="symbol", values="actual")
    # min_names=5 not 20: the training universe may be far smaller than the panel
    # universe, and a metric that silently returns all-NaN is worse than a noisy
    # one whose noise is visible in cs_n_dates.
    ic = cross_sectional_ic(pred_wide, actual_wide, min_names=5)
    v = ic.dropna()
    if len(v) < 2:
        return {"cs_ic": float("nan"), "cs_ic_ir": float("nan"), "cs_n_dates": len(v)}
    sd = float(v.std(ddof=1))
    return {
        "cs_ic": float(v.mean()),
        "cs_ic_ir": float(v.mean() / sd) if sd > 0 else float("nan"),
        "cs_n_dates": int(len(v)),
    }


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

def train_ridge(X_train, X_test, y_train, y_test, alpha: float):
    scaler = RobustScaler()
    X_tr = np.clip(scaler.fit_transform(X_train), -CLIP, CLIP)
    X_te = np.clip(scaler.transform(X_test),  -CLIP, CLIP)
    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_train)
    pred_test = model.predict(X_te)
    train_metrics = _forecast_metrics(model.predict(X_tr), y_train)
    test_metrics = _forecast_metrics(pred_test, y_test)
    return model, scaler, train_metrics, test_metrics, pred_test


def train_enet(X_train, X_test, y_train, y_test, cv_splits: int = 3):
    """ElasticNet with alpha and l1_ratio chosen inside the training window only.

    Ridge cannot zero a coefficient, so with 60 augmented columns — each feature
    present on two correlated axes — it spreads weight across near-duplicates.
    ElasticNet's L1 term can pick an axis per feature. A fixed alpha is not
    usable here the way Ridge's is: the target is a ~1e-2 return, so anything
    near sklearn's default zeroes every coefficient and yields a constant
    forecast. TimeSeriesSplit rather than plain KFold keeps the folds causal.
    """
    scaler = RobustScaler()
    X_tr = np.clip(scaler.fit_transform(X_train), -CLIP, CLIP)
    X_te = np.clip(scaler.transform(X_test), -CLIP, CLIP)
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        n_alphas=20,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        max_iter=5000,
        selection="random",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_train)
    logger.info(
        "  ElasticNetCV chose alpha=%.3g l1_ratio=%.2f (%d/%d nonzero coefficients)",
        model.alpha_, model.l1_ratio_, int((model.coef_ != 0).sum()), len(model.coef_),
    )
    pred_test = model.predict(X_te)
    train_metrics = _forecast_metrics(model.predict(X_tr), y_train)
    test_metrics = _forecast_metrics(pred_test, y_test)
    return model, scaler, train_metrics, test_metrics, pred_test


def train_xgb_regressor(X_train, X_test, y_train, y_test):
    if not HAS_XGBOOST:
        raise ImportError("xgboost not installed.")
    # Fit through the same scaler+clip path the linear models use, so one
    # predictor class can serve every regressor this script produces. Trees are
    # invariant to the affine part; the clip is a real transform but is applied
    # identically at train and serve time.
    scaler = RobustScaler()
    X_tr = np.clip(scaler.fit_transform(X_train), -CLIP, CLIP)
    X_te = np.clip(scaler.transform(X_test), -CLIP, CLIP)
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.7, min_child_weight=5,
        # gamma=0, NOT the 1.0 copied from the classifier config. gamma is a
        # minimum split gain in units of the loss: on 3-class log-loss a gain of
        # 1.0 is a modest bar, but on squared error against a ~1e-2 return the
        # best split in the data gains ~1e-4, so gamma=1.0 rejects every split
        # and the model returns the training mean. That — not "no signal in the
        # features" — is what earlier XGBoost runs here were measuring.
        gamma=0.0, random_state=42, tree_method="hist", verbosity=0,
    )
    model.fit(X_tr, y_train)
    pred_test = model.predict(X_te)
    train_metrics = _forecast_metrics(model.predict(X_tr), y_train)
    test_metrics = _forecast_metrics(pred_test, y_test)
    return model, scaler, train_metrics, test_metrics, pred_test


# ---------------------------------------------------------------------------
# Save / register
# ---------------------------------------------------------------------------

def _save_and_register(
    model_obj, scaler, model_type: str, train_metrics: dict, test_metrics: dict,
    train_symbols: list[str], days: int, db: DB, train_samples: int, test_samples: int,
    best_signal_quantile: float = 0.7,
    best_threshold_window: int = 60,
    model_key: str = MODEL_KEY,
    feature_cols: list[str] | None = None,
    extra: dict | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
) -> int:
    feature_cols = FEATURE_COLS if feature_cols is None else feature_cols
    # `train_end` is what oos_guard reads to refuse in-sample backtest rows, so
    # it must be the split boundary, not the wall clock. It defaulted to
    # datetime.now() until 2026-07-28, which recorded "trained through today"
    # for every model however it was actually split — making a --train-end
    # holdout invisible to the guard, and any real holdout unenforceable.
    end = datetime.now().strftime("%Y-%m-%d") if train_end is None else train_end
    start = (
        (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        if train_start is None else train_start
    )

    # model_registry has columns shaped for classifiers (test_accuracy, test_f1);
    # repurposed here for directional accuracy and IC respectively rather than
    # adding regression-specific columns for a single model type.
    version = db.register_model(
        model_key=model_key,
        artifact_path="PLACEHOLDER",
        feature_contract=feature_cols,
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
    artifact_path = f"models/{model_key}_v{version}.pkl"
    artifact = {
        "model": model_obj,
        "scaler": scaler,
        "model_type": model_type,
        "feature_contract": feature_cols,
        "feature_set_name": FEATURE_SET_NAME,
        "trained_at": datetime.now().isoformat(),
        "train_start": start,
        "train_end": end,
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
        **(extra or {}),
    }
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)

    # Write canonical path with SHA-256 integrity file so _load_validated_pickle
    # can verify it on next load (matches train_models._pickle_and_hash contract).
    canonical = f"models/{model_key}.pkl"
    _pickle_and_hash(artifact, canonical)

    db.update_artifact_path(model_key, version, artifact_path)
    db.deactivate_old_models(model_key, version)

    logger.info(
        "%s v%d [%s]: train_ic=%.4f test_ic=%.4f test_dir_acc=%.4f test_r2=%+.5f -> %s",
        model_key, version, model_type, train_metrics["ic"], test_metrics["ic"],
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
        "--model", choices=["ridge", "enet", "xgboost", "all"], default="ridge",
        help="Which regressor to train. enet picks alpha/l1_ratio by walk-forward CV "
             "inside the training window; ridge uses a fixed --alpha.",
    )
    parser.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization strength")
    parser.add_argument("--db", default="data/trading_sim.db", help="Path to SQLite DB")
    parser.add_argument(
        "--demean", action="store_true",
        help="Subtract each date's cross-sectional mean from the target, so the model "
             "forecasts return-relative-to-universe. This is the target a rank-based "
             "panel book trades; the absolute-return default is not.",
    )
    parser.add_argument(
        "--cs-mode", choices=list(CS_MODES), default="off",
        help="Feature normalization axis. 'off' keeps the per-symbol rolling z-score; "
             "'replace' swaps in each feature's per-date rank across the universe; "
             "'augment' keeps both (60 columns). Models trained on a non-'off' axis are "
             "served the same transform automatically by panel_data, and are NOT usable "
             "on the per-symbol live path, which has no cross-section to rank against.",
    )
    parser.add_argument(
        "--train-end",
        help="Train on dates < this (YYYY-MM-DD) instead of the first 80%% of the "
             "history. Use it to leave a fixed calendar window unseen: a fraction "
             "split moves every time --days does, so it cannot hold one out.",
    )
    parser.add_argument(
        "--model-key", default=MODEL_KEY,
        help="Registry key and pickle basename. Use a distinct key for experiments so "
             f"they do not overwrite the live models/{MODEL_KEY}.pkl.",
    )
    parser.add_argument(
        "--skip-sweep", action="store_true",
        help="Skip the walk-forward signal_quantile/threshold_window sweep. Those "
             "parameters are only read by DailyPredictorStrategy (the per-symbol path); "
             "a model destined for the panel ranker never uses them.",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    logger.info("Training symbols: %d (%s...)", len(symbols), ",".join(symbols[:6]))
    logger.info(
        "History: %d days  demean=%s  cs_mode=%s  model_key=%s",
        args.days, args.demean, args.cs_mode, args.model_key,
    )

    db = DB(args.db)
    logger.info("Collecting and preparing data...")
    data = prepare_data(
        symbols, args.days, db, demean=args.demean, cs_mode=args.cs_mode,
        train_end=args.train_end,
    )
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    logger.info(
        "Total samples: %d  Features: %d  Train: %d  Test: %d",
        len(X_train) + len(X_test), X_train.shape[1], len(X_train), len(X_test),
    )

    if args.skip_sweep:
        best_q, best_w = 0.7, 60
        logger.info("Skipping parameter sweep — using defaults (0.7, 60)")
    else:
        logger.info("Running walk-forward parameter sweep...")
        try:
            wf_config = WalkForwardConfig()
            best_q, best_w = sweep_params(
                symbols, args.days, db, wf_config, end=args.train_end
            )
            logger.info("Parameter sweep complete: signal_quantile=%.2f threshold_window=%d", best_q, best_w)
        except Exception as exc:
            logger.warning("Parameter sweep failed: %s — using defaults (0.7, 60)", exc)
            best_q, best_w = 0.7, 60

    models_to_train = ["ridge", "enet", "xgboost"] if args.model == "all" else [args.model]

    for model_type in models_to_train:
        logger.info("--- Training %s ---", model_type)
        if model_type == "ridge":
            model, scaler, train_m, test_m, pred_test = train_ridge(
                X_train, X_test, y_train, y_test, args.alpha
            )
        elif model_type == "enet":
            model, scaler, train_m, test_m, pred_test = train_enet(
                X_train, X_test, y_train, y_test
            )
        else:
            model, scaler, train_m, test_m, pred_test = train_xgb_regressor(
                X_train, X_test, y_train, y_test
            )

        cs = cross_sectional_metrics(
            pred_test, y_test, data["test_dates"], data["test_symbols"]
        )
        logger.info(
            "  test pooled_ic=%+.4f  cross_sectional_ic=%+.4f (IR %+.3f over %d dates)",
            test_m["ic"], cs["cs_ic"], cs["cs_ic_ir"], cs["cs_n_dates"],
        )

        _save_and_register(
            model, scaler, model_type, train_m, test_m,
            data["used_symbols"], args.days, db,
            train_samples=len(X_train), test_samples=len(X_test),
            best_signal_quantile=best_q,
            best_threshold_window=best_w,
            model_key=args.model_key,
            feature_cols=data["feature_cols"],
            train_start=data["train_start_date"],
            train_end=data["train_end_date"],
            extra={
                "demeaned_target": data["demeaned"],
                "cs_mode": data["cs_mode"],
                **cs,
            },
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
