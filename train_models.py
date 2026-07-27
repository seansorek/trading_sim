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
import hashlib
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import RobustScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

try:
    from rl_env import TradingEnv
    from dqn_agent import DQNAgent, DQNConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from data_loader import check_cache_coverage, check_cache_freshness, load_yfinance
from daily_features import (
    FEATURE_COLS,
    FEATURE_SET_NAME,
    FWD_RET_HORIZON_DAYS,
    discretize_labels,
    make_daily_features,
)
from db import DB
from predictors.base import CLIP, _preprocess

Path("logs").mkdir(exist_ok=True)
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

# Maximum number of business days the cached data's latest bar can lag behind
# the requested end date before we consider the cache stale and re-fetch.
_STALE_TOLERANCE_BDAYS = 4


def _pickle_and_hash(artifact: dict, path: str) -> None:
    """Write artifact as pickle and save its SHA-256 to <path>.sha256."""
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    Path(path + ".sha256").write_text(digest, encoding="ascii")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_symbol(
    symbol: str, start: str, end: str, db: DB
) -> pd.DataFrame | None:
    """Fetch daily bars, cache in DB, return DataFrame.

    A recency check ensures we don't silently train on stale data: if the
    cache's most recent bar is more than ``_STALE_TOLERANCE_BDAYS`` business
    days before the requested *end* date, the cache is considered stale and
    data is re-fetched from yfinance.
    """
    cached = db.load_bars(symbol, "1d", start, end)
    if cached is not None and len(cached) >= 50:
        fresh = check_cache_freshness(cached, end, _STALE_TOLERANCE_BDAYS)
        covers_start = check_cache_coverage(cached, start)
        if fresh and covers_start:
            logger.info("  %s: loaded %d bars from DB cache", symbol, len(cached))
            return cached
        elif not covers_start:
            logger.info(
                "  %s: cache missing older history (earliest bar %s, need <= start %s), re-fetching",
                symbol, cached.index.min().date(), pd.to_datetime(start).date(),
            )
        else:
            latest_bar = cached.index.max()
            latest_date = latest_bar.tz_localize(None) if latest_bar.tzinfo else latest_bar
            end_dt = pd.to_datetime(end)
            end_date = end_dt.tz_localize(None) if end_dt.tzinfo else end_dt
            stale_cutoff = end_date - pd.tseries.offsets.BDay(_STALE_TOLERANCE_BDAYS)
            logger.info(
                "  %s: cache stale (latest bar %s, need >= %s), re-fetching",
                symbol, latest_date.date(), stale_cutoff.date(),
            )

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
    symbols: list[str], days: int, db: DB, vol_mult: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Collect training data across symbols with per-symbol temporal splits.

    Each symbol is split 80/20 by time independently before pooling, so the
    test set contains truly held-out future data for every symbol. An embargo
    gap of FWD_RET_HORIZON_DAYS rows is dropped between train and test so that
    no training label's forward-return horizon overlaps the test period's
    price action (purged split — see Lopez de Prado).

    Returns (X_train, y_train, X_test, y_test, symbol_list_used).
    """
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    train_Xs: list[np.ndarray] = []
    train_ys: list[np.ndarray] = []
    test_Xs: list[np.ndarray] = []
    test_ys: list[np.ndarray] = []
    used_symbols: list[str] = []

    # Load SPY once for market-relative features; warn but continue if unavailable
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

        # Drop the last 3 rows where fwd_ret_1d is NaN (3-day horizon)
        feats = feats.dropna(subset=["fwd_ret_1d"])
        if len(feats) < 50:
            logger.warning("  %s: too few valid rows (%d) after dropna", symbol, len(feats))
            continue

        X_sym = feats[FEATURE_COLS].values.astype(np.float32)

        # Volatility-adjusted thresholds: use raw 20-day return vol (not z-scored vol_20d)
        raw_vol = df["close"].pct_change().rolling(20).std().reindex(feats.index).values
        pos_thr = raw_vol * np.sqrt(3) * vol_mult
        y_sym = discretize_labels(feats["fwd_ret_1d"].values, pos_thr=pos_thr, neg_thr=-pos_thr)

        split = int(len(X_sym) * 0.8)
        # Embargo gap: drop rows whose fwd_ret_1d horizon would reach into the
        # test period, so train labels never depend on test-period prices.
        test_start = split + FWD_RET_HORIZON_DAYS
        if test_start >= len(X_sym):
            logger.warning("  %s: too few rows for embargo gap, skipping", symbol)
            continue
        X_tr = _preprocess(X_sym[:split].copy())
        X_te = _preprocess(X_sym[test_start:].copy())

        train_Xs.append(X_tr); train_ys.append(y_sym[:split])
        test_Xs.append(X_te);  test_ys.append(y_sym[test_start:])
        used_symbols.append(symbol)
        logger.info(
            "  %s: %d samples (train=%d embargo=%d test=%d), class dist %s",
            symbol, len(y_sym), split, FWD_RET_HORIZON_DAYS, len(y_sym) - test_start,
            np.bincount(y_sym).tolist(),
        )

    if not train_Xs:
        raise RuntimeError("No usable training data across all symbols.")

    X_train = np.vstack(train_Xs)
    y_train = np.concatenate(train_ys)
    X_test  = np.vstack(test_Xs)
    y_test  = np.concatenate(test_ys)
    return X_train, y_train, X_test, y_test, used_symbols


def _make_artifact_path(model_key: str, version: int) -> str:
    Path("models").mkdir(exist_ok=True)
    return f"models/{model_key}_v{version}.pkl"


def _save_and_register(
    model_key: str,
    model_obj,
    scaler: RobustScaler,
    train_accuracy: float,
    test_accuracy: float,
    test_f1: float,
    confidence_threshold: float,
    train_symbols: list[str],
    days: int,
    db: DB,
    train_samples: int = 0,
    test_samples: int = 0,
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
        train_samples=train_samples,
        test_samples=test_samples,
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
    _pickle_and_hash(artifact, artifact_path)

    # Also write to the canonical path (no version suffix) for backward compat
    canonical = f"models/{model_key}.pkl"
    _pickle_and_hash(artifact, canonical)

    db.update_artifact_path(model_key, version, artifact_path)
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
    optimize: bool = False,
    opt_cfg=None,
) -> tuple[LogisticRegression, RobustScaler, float, float, float]:
    scaler = RobustScaler()
    X_tr = np.clip(scaler.fit_transform(X_train), -CLIP, CLIP)
    X_te = np.clip(scaler.transform(X_test),  -CLIP, CLIP)

    if optimize and HAS_SKOPT and opt_cfg is not None:
        logger.info("Running Bayesian hyperparameter search for Logistic Regression...")
        ol = opt_cfg.logistic
        search_space = {
            "C": Real(ol.C[0], ol.C[1], prior="log-uniform"),
            "penalty": Categorical(ol.penalty),
        }
        # saga is the only solver that supports both l1 and l2 with multinomial
        base = LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
        opt = BayesSearchCV(
            base, search_space,
            n_iter=opt_cfg.n_iter,
            cv=opt_cfg.cv,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42,
        )
        opt.fit(X_tr, y_train)
        best_params = dict(opt.best_params_)
        logger.info("[Logistic] Best params: %s", best_params)
        model = LogisticRegression(
            **best_params,
            solver="saga",
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
        model.fit(X_tr, y_train)
    else:
        if optimize and not HAS_SKOPT:
            logger.warning("scikit-optimize not installed — falling back to config hyperparams (pip install scikit-optimize)")
        # class_weight=None keeps the natural HOLD-majority prior, which test
        # accuracy is measured against. "balanced" fights that prior and tanks
        # accuracy (see hybrid_model tuning notes).
        model = LogisticRegression(
            C=cfg.get("C", 1.0),
            max_iter=cfg.get("max_iter", 1000),
            solver="lbfgs",
            class_weight=cfg.get("class_weight"),
            random_state=42,
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
    opt_cfg: Any = None,
) -> tuple[Any, Any, float, float, float]:
    if not HAS_XGBOOST:
        raise ImportError("xgboost not installed. pip install xgboost")

    scaler = RobustScaler()
    X_tr = np.clip(scaler.fit_transform(X_train), -CLIP, CLIP)
    X_te = np.clip(scaler.transform(X_test),  -CLIP, CLIP)

    if optimize and not HAS_SKOPT:
        logger.warning("scikit-optimize not installed — falling back to config hyperparams (pip install scikit-optimize)")

    if optimize and HAS_SKOPT and opt_cfg is not None:
        logger.info("Running Bayesian hyperparameter search for XGBoost...")
        ox = opt_cfg.xgboost
        search_space = {
            "n_estimators": Integer(ox.n_estimators[0], ox.n_estimators[1]),
            "max_depth": Integer(ox.max_depth[0], ox.max_depth[1]),
            "learning_rate": Real(ox.learning_rate[0], ox.learning_rate[1], prior="log-uniform"),
            "subsample": Real(ox.subsample[0], ox.subsample[1]),
            "colsample_bytree": Real(ox.colsample_bytree[0], ox.colsample_bytree[1]),
            "gamma": Real(ox.gamma[0], ox.gamma[1]),
            "min_child_weight": Integer(ox.min_child_weight[0], ox.min_child_weight[1]),
            "reg_alpha": Real(ox.reg_alpha[0], ox.reg_alpha[1]),
            "reg_lambda": Real(ox.reg_lambda[0], ox.reg_lambda[1]),
        }
        base = xgb.XGBClassifier(
            random_state=42, tree_method="hist", verbosity=0,
            objective="multi:softprob", num_class=3,
        )
        opt = BayesSearchCV(base, search_space, n_iter=opt_cfg.n_iter, cv=opt_cfg.cv,
                            n_jobs=-1, scoring="f1_macro", random_state=42)
        opt.fit(X_tr, y_train)
        model = opt.best_estimator_
        logger.info("[XGBoost] Best params: %s", dict(opt.best_params_))
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
        # Sample weighting scheme: "none" keeps the natural HOLD-majority prior
        # (best for test accuracy, since the test set has the same prior);
        # "sqrt" is a mild rebalance; "inverse" fully balances classes but was
        # found to tank accuracy to ~34-43% (see hybrid_model tuning notes).
        class_weight = cfg.get("class_weight", "none")
        class_counts = np.bincount(y_train, minlength=3)
        total = len(y_train)
        if class_weight == "inverse":
            sample_weights = np.array([total / (3 * class_counts[y]) for y in y_train])
        elif class_weight == "sqrt":
            sample_weights = np.sqrt(np.array([total / (3 * class_counts[y]) for y in y_train]))
        else:
            sample_weights = None
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
# DQN trainer
# ---------------------------------------------------------------------------

_DQN_DEFAULT_SYMBOLS = (
    "AAPL,SPY,MSFT,GOOGL,NVDA,TSLA,META,NFLX,AMD,INTC,"
    "AVGO,ADBE,CSCO,CRM,DXCM,QQQ,IWM,GLD,USO,XLF,"
    "XLV,XLE,XLY,XLI,COIN,RIOT,MARA,SOXL,TQQQ,UPRO"
)


def train_dqn(symbols, start, end, cfg_dqn, out_path, use_dueling=True, use_per=True):
    """Train DQN agent using config-driven hyperparameters."""
    print("\n" + "=" * 70)
    print("DQN AGENT TRAINING - REAL DATA VERIFICATION")
    print("=" * 70)
    print(f"Training Period: {start} to {end}")
    print(f"Data Source: yfinance (REAL MARKET DATA)")
    print(f"Symbols: {len(symbols)}")
    print(f"Interval: Daily (1d)")
    print(f"Architecture: Dueling DQN with PER")
    print("=" * 70 + "\n")

    # Fetch SPY once and share it across every env so ret_*_vs_spy features
    # are real during training (see issue #123) — matches the spy_df the
    # live prediction path always passes to make_daily_features.
    try:
        spy_df = load_yfinance("SPY", start=start, end=end, interval="1d")
    except Exception as exc:
        logger.warning("Could not load SPY for DQN training features: %s", exc)
        spy_df = None

    # Build each env with an identity scaler first so self.df holds RAW
    # (unscaled) features — we need those to fit ONE shared scaler across
    # all training symbols (see below) instead of each env silently keeping
    # its own per-symbol scaler that inference could never reproduce.
    _identity_scaler = {c: (0.0, 1.0) for c in FEATURE_COLS}
    envs = [
        TradingEnv(
            sym, start=start, end=end, window=cfg_dqn.window,
            feature_scaler=_identity_scaler,
            spy_df=spy_df if sym != "SPY" else None,
        )
        for sym in symbols
    ]
    state_dim = envs[0].observation_space_shape[0]
    action_dim = envs[0].action_space_n

    # Fit one shared z-score scaler across all training symbols' warmup
    # windows, apply it to every env in place, and persist it with the
    # agent so live prediction normalizes on the exact same statistics
    # instead of re-deriving different ones from a daily-drifting window
    # (see issue #123).
    warmup_frames = []
    for env in envs:
        fit_end = min(252, max(env.window + 1, len(env.df) // 2))
        warmup_frames.append(env.df[env.features].iloc[:fit_end])
    combined = pd.concat(warmup_frames)
    dqn_scaler: dict = {}
    for c in FEATURE_COLS:
        mu = float(combined[c].mean())
        sd = float(combined[c].std())
        if not sd or np.isnan(sd):
            sd = 1.0
        dqn_scaler[c] = (mu, sd)
    for env in envs:
        for c in env.features:
            mu, sd = dqn_scaler[c]
            env.df[c] = (env.df[c] - mu) / sd
        env.scaler = dqn_scaler

    print(f"[info] Loaded {len(symbols)} symbols with real yfinance data")
    print("\nData Verification:")
    total_bars = 0
    for env in envs:
        info = env.get_data_info()
        print(f"  [ok] {info['symbol']:6s}: {info['num_bars']:4d} bars from {info['date_range']} ({info['source']})")
        total_bars += info['num_bars']
    print(f"\n[ok] Total bars loaded: {total_bars:,}")
    print(f"\n[info] Training Enhanced DQN with Dueling={use_dueling}, PER={use_per}")
    print(f"[info] State dim: {state_dim}, Action dim: {action_dim}")
    print(f"[info] Episodes: {cfg_dqn.episodes}, Steps/ep: {cfg_dqn.steps_per_episode}\n")

    cfg = DQNConfig(
        gamma=cfg_dqn.gamma,
        lr=cfg_dqn.lr,
        batch_size=cfg_dqn.batch_size,
        buffer_size=cfg_dqn.buffer_size,
        start_epsilon=cfg_dqn.epsilon_start,
        end_epsilon=cfg_dqn.epsilon_end,
        epsilon_decay_steps=cfg_dqn.epsilon_decay_steps,
        target_update_interval=cfg_dqn.target_update_interval,
        hidden=cfg_dqn.hidden,
        device="cpu",
        use_dueling=use_dueling,
        use_per=use_per,
        per_alpha=0.6,
        per_beta=0.4,
    )
    agent = DQNAgent(state_dim, action_dim, cfg)

    global_step = 0
    episode_rewards = []

    for ep in range(cfg_dqn.episodes):
        ep_reward = 0.0
        for env in envs:
            s = env.reset()
            for t in range(cfg_dqn.steps_per_episode):
                a = agent.act(s)
                s2, r, done, info = env.step(a)
                agent.push(s, a, r, s2, done)
                agent.learn()
                s = s2
                ep_reward += r
                global_step += 1
                if done:
                    break

        episode_rewards.append(ep_reward)
        if (ep + 1) % 5 == 0 or ep == 0:
            avg = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else ep_reward
            epsilon = agent._current_epsilon()
            print(f"[ep {ep+1:3d}] Reward: {ep_reward:7.2f} | Avg(5): {avg:7.2f} | Epsilon: {epsilon:.4f} | Buffer: {len(agent.buffer)}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    agent.save(out_path, scaler=dqn_scaler, feature_contract=FEATURE_COLS)
    logger.info("Saved DQN agent to %s (final reward: %.2f)", out_path, episode_rewards[-1])


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
        help="Comma-separated list of models to train: logistic, xgboost, dqn"
    )
    parser.add_argument("--optimize", action="store_true", help="Run Bayesian hyperparameter search")
    parser.add_argument("--db", default="data/trading_sim.db", help="Path to SQLite DB")
    parser.add_argument("--confidence", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument(
        "--vol-mult", type=float, default=0.5,
        help="Volatility multiplier for label thresholds (higher -> larger HOLD class, easier task)",
    )
    parser.add_argument(
        "--logistic-class-weight", choices=["none", "balanced"], default="none",
        help="Logistic class_weight scheme ('balanced' fights the natural HOLD-majority prior)",
    )
    parser.add_argument(
        "--xgb-class-weight", choices=["none", "sqrt", "inverse"], default="none",
        help="XGBoost sample weighting scheme",
    )
    # DQN-specific args
    parser.add_argument("--dqn-symbols", default=_DQN_DEFAULT_SYMBOLS, help="Symbols for DQN training")
    parser.add_argument("--dqn-start", default="2024-01-01", help="DQN training start date")
    parser.add_argument("--dqn-end", default="2025-12-02", help="DQN training end date")
    parser.add_argument("--dqn-out", default="models/dqn_agent.pt", help="DQN model output path")
    parser.add_argument("--no-dueling", action="store_true", help="Disable Dueling architecture")
    parser.add_argument("--no-per", action="store_true", help="Disable Prioritized Experience Replay")
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
    X_train, y_train, X_test, y_test, used_symbols = _prepare_data(
        symbols, args.days, db, vol_mult=args.vol_mult,
    )
    logger.info(
        "Total samples: %d  Features: %d  Train: %d  Test: %d",
        len(X_train) + len(X_test), X_train.shape[1], len(X_train), len(X_test),
    )
    logger.info("Train class distribution: %s", np.bincount(y_train).tolist())
    logger.info("Test class distribution: %s", np.bincount(y_test).tolist())

    if "logistic" in models_to_train:
        logger.info("--- Training DailyLogistic ---")
        from config import get_config
        cfg = get_config()
        logistic_cfg = {
            "C": cfg.strategies.logistic.C,
            "max_iter": cfg.strategies.logistic.max_iter,
            "class_weight": None if args.logistic_class_weight == "none" else args.logistic_class_weight,
        }
        model, scaler, train_acc, test_acc, test_f1 = train_logistic(
            X_train, X_test, y_train, y_test, logistic_cfg,
            optimize=args.optimize, opt_cfg=cfg.optimize,
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
            train_samples=len(X_train),
            test_samples=len(X_test),
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
            "class_weight": args.xgb_class_weight,
        }
        model, scaler, train_acc, test_acc, test_f1 = train_xgboost(
            X_train, X_test, y_train, y_test, xgb_cfg,
            optimize=args.optimize, opt_cfg=cfg.optimize,
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
            train_samples=len(X_train),
            test_samples=len(X_test),
        )

    if "dqn" in models_to_train:
        if not HAS_TORCH:
            logger.error("PyTorch / rl_env not available — skipping DQN (pip install torch)")
        else:
            from config import get_config
            cfg = get_config()
            dqn_symbols = [s.strip().upper() for s in args.dqn_symbols.split(",") if s.strip()]
            train_dqn(
                dqn_symbols,
                args.dqn_start,
                args.dqn_end,
                cfg.strategies.dqn,
                args.dqn_out,
                use_dueling=not args.no_dueling,
                use_per=not args.no_per,
            )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
