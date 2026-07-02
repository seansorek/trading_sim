#!/usr/bin/env python3
"""
train_hybrid.py — Train the XGBoost-transformer hybrid model.

Pipeline:
  1. Fetch daily bars for each symbol; build the 25-feature daily vector.
  2. Construct (lookback, 25) sequences per symbol with temporal 80/20 split.
  3. Pretrain a small transformer encoder on the 3-class target.
  4. Extract transformer embeddings for every (sequence) row.
  5. Train XGBoost on [last-bar features ⊕ embedding] -> {0:SELL, 1:HOLD, 2:BUY}.
  6. Report test accuracy; save the bundle.

Usage:
  python train_hybrid.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM --days 1000
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from daily_features import (
    FEATURE_COLS,
    FEATURE_SET_NAME,
    FWD_RET_HORIZON_DAYS,
    discretize_labels,
    make_daily_features,
)
from data_loader import check_cache_coverage, check_cache_freshness, load_yfinance
from db import DB
from hybrid_model import (
    TransformerCfg,
    TransformerEncoder,
    build_sequences,
    extract_embeddings,
    train_transformer,
)

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train_hybrid.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}
MODEL_KEY = "daily_hybrid"

# Maximum number of business days the cached data's latest bar can lag behind
# the requested end date before we consider the cache stale and re-fetch.
_STALE_TOLERANCE_BDAYS = 4


def _seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _preprocess(X: np.ndarray) -> np.ndarray:
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X, nan=0.0)
    for col in range(X.shape[1]):
        col_data = X[:, col]
        std = np.std(col_data)
        if std > 0:
            mean = np.mean(col_data)
            X[:, col] = np.clip(col_data, mean - 5 * std, mean + 5 * std)
    return X


def _load_symbol(symbol: str, start: str, end: str, db: DB) -> pd.DataFrame | None:
    cached = db.load_bars(symbol, "1d", start, end)
    if cached is not None and len(cached) >= 50:
        if check_cache_freshness(cached, end, _STALE_TOLERANCE_BDAYS) and check_cache_coverage(
            cached, start
        ):
            logger.info("  %s: %d bars from DB cache", symbol, len(cached))
            return cached
        logger.info(
            "  %s: cache stale or missing older history, re-fetching from yfinance...", symbol,
        )
    else:
        logger.info("  %s: fetching from yfinance...", symbol)
    try:
        df = load_yfinance(symbol, start=start, end=end, interval="1d")
    except Exception as exc:
        logger.warning("  %s: fetch failed: %s", symbol, exc)
        return None
    if df is None or len(df) < 50:
        logger.warning("  %s: insufficient data", symbol)
        return None
    db.upsert_bars(symbol, "1d", df)
    return df


def prepare_data(
    symbols: list[str],
    days: int,
    db: DB,
    lookback: int,
    vol_mult: float = 0.5,
) -> dict:
    """
    Build training/test arrays.

    Returns dict with keys:
      X_train_seq, X_test_seq        — (M, lookback, n_feat) sequences (scaled)
      X_train_last, X_test_last      — (M, n_feat) last-bar features (scaled)
      y_train, y_test                — labels
      scaler                         — fitted StandardScaler on raw features
      used_symbols                   — list of symbols that yielded data
    """
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    spy_df = _load_symbol("SPY", start, end, db)
    if spy_df is None:
        logger.warning("SPY missing — ret_*_vs_spy features will be 0")

    # First pass: collect per-symbol raw features + labels, do per-symbol temporal split
    train_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    test_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    used_symbols: list[str] = []

    for sym in symbols:
        df = _load_symbol(sym, start, end, db)
        if df is None:
            continue
        try:
            spy_arg = spy_df if sym != "SPY" else None
            feats = make_daily_features(df, spy_df=spy_arg)
        except Exception as exc:
            logger.warning("  %s: feature failure: %s", sym, exc)
            continue
        feats = feats.dropna(subset=["fwd_ret_1d"])
        if len(feats) < lookback + 50:
            logger.warning("  %s: too few rows (%d)", sym, len(feats))
            continue

        X_sym = feats[FEATURE_COLS].values.astype(np.float32)
        vol = feats["vol_20d"].values
        pos_thr = vol * np.sqrt(3) * vol_mult
        y_sym = discretize_labels(
            feats["fwd_ret_1d"].values, pos_thr=pos_thr, neg_thr=-pos_thr
        )

        split = int(len(X_sym) * 0.8)
        # Embargo gap: drop rows whose fwd_ret_1d horizon would reach into the
        # test period, so train labels never depend on test-period prices.
        test_start = split + FWD_RET_HORIZON_DAYS
        if test_start + lookback >= len(X_sym):
            logger.warning("  %s: too few rows for embargo gap, skipping", sym)
            continue
        train_blocks.append((X_sym[:split], y_sym[:split]))
        test_blocks.append((X_sym[test_start:], y_sym[test_start:]))
        used_symbols.append(sym)
        logger.info(
            "  %s: %d rows (train=%d embargo=%d test=%d) class dist %s",
            sym, len(y_sym), split, FWD_RET_HORIZON_DAYS, len(y_sym) - test_start,
            np.bincount(y_sym).tolist(),
        )

    if not train_blocks:
        raise RuntimeError("No usable training data.")

    # Fit scaler on training features only (concatenated across symbols)
    X_train_raw_all = np.vstack([b[0] for b in train_blocks])
    scaler = StandardScaler()
    scaler.fit(_preprocess(X_train_raw_all.copy()))

    def _scale(blocks):
        out_X, out_y, starts = [], [], [0]
        for Xb, yb in blocks:
            Xb_clip = _preprocess(Xb.copy())
            Xb_scaled = scaler.transform(Xb_clip).astype(np.float32)
            out_X.append(Xb_scaled)
            out_y.append(yb)
            starts.append(starts[-1] + len(Xb_scaled))
        return np.vstack(out_X), np.concatenate(out_y), starts[:-1]

    X_train_flat, y_train_flat, train_starts = _scale(train_blocks)
    X_test_flat, y_test_flat, test_starts = _scale(test_blocks)

    X_train_seq, X_train_last, y_train = build_sequences(
        X_train_flat, y_train_flat, lookback, train_starts
    )
    X_test_seq, X_test_last, y_test = build_sequences(
        X_test_flat, y_test_flat, lookback, test_starts
    )

    logger.info(
        "Train: %d sequences  Test: %d sequences  Features per bar: %d  Lookback: %d",
        len(X_train_seq), len(X_test_seq), X_train_seq.shape[2], lookback,
    )
    logger.info("Train labels: %s   Test labels: %s",
                np.bincount(y_train).tolist(), np.bincount(y_test).tolist())

    return {
        "X_train_seq": X_train_seq,
        "X_test_seq": X_test_seq,
        "X_train_last": X_train_last,
        "X_test_last": X_test_last,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "used_symbols": used_symbols,
    }


def train_hybrid(args) -> dict:
    _seed_all(args.seed)
    db = DB(args.db)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    logger.info("Symbols: %s", symbols)
    logger.info("Days: %d  Lookback: %d", args.days, args.lookback)

    data = prepare_data(symbols, args.days, db, args.lookback, vol_mult=args.vol_mult)

    # ----- Pretrain transformer -----
    tcfg = TransformerCfg(
        lookback=args.lookback,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        embed_dim=args.embed_dim,
        num_classes=3,
    )
    n_feat = data["X_train_seq"].shape[2]
    model = TransformerEncoder(n_feat, tcfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Transformer params: %d", n_params)

    logger.info("Pretraining transformer...")
    tr_stats = train_transformer(
        model,
        data["X_train_seq"], data["y_train"],
        data["X_test_seq"], data["y_test"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )
    logger.info(
        "Transformer best val_loss=%.4f val_acc=%.3f",
        tr_stats["best_val_loss"], tr_stats["best_val_acc"],
    )

    # ----- Extract embeddings AND class probabilities -----
    logger.info("Extracting embeddings + transformer probabilities...")
    emb_train = extract_embeddings(model, data["X_train_seq"], device=args.device)
    emb_test = extract_embeddings(model, data["X_test_seq"], device=args.device)
    logger.info("Embedding shape: %s", emb_train.shape)

    model.eval()
    with torch.no_grad():
        tr_logits, _ = model(torch.from_numpy(data["X_train_seq"]))
        te_logits, _ = model(torch.from_numpy(data["X_test_seq"]))
        tr_probs = torch.softmax(tr_logits, dim=-1).cpu().numpy()
        te_probs = torch.softmax(te_logits, dim=-1).cpu().numpy()
        tx_pred = te_logits.argmax(-1).cpu().numpy()
    tx_acc = float(accuracy_score(data["y_test"], tx_pred))
    tx_f1 = float(f1_score(data["y_test"], tx_pred, average="macro", zero_division=0))
    logger.info("Transformer-only test acc=%.4f f1=%.4f", tx_acc, tx_f1)

    # ----- Train XGBoost on [last-bar features ⊕ transformer probs] -----
    # Using probabilities (3 dims) rather than raw embeddings (64 dims) keeps
    # XGBoost's feature space focused. The embedding adds 64 noisy dims; the
    # probability vector is a distilled, supervised summary.
    if args.use_embeddings:
        Xc_train = np.hstack([data["X_train_last"], tr_probs, emb_train])
        Xc_test = np.hstack([data["X_test_last"], te_probs, emb_test])
    else:
        Xc_train = np.hstack([data["X_train_last"], tr_probs])
        Xc_test = np.hstack([data["X_test_last"], te_probs])
    logger.info("XGBoost feature space: %d", Xc_train.shape[1])

    # Sample weights: 'none' keeps the natural class prior (best for accuracy
    # when the test set is class-imbalanced), 'sqrt' is a mild rebalance,
    # 'inverse' fully balances the classes.
    class_counts = np.bincount(data["y_train"], minlength=3)
    total = len(data["y_train"])
    if args.xgb_class_weight == "inverse":
        sample_w = np.array([total / (3 * class_counts[y]) for y in data["y_train"]])
    elif args.xgb_class_weight == "sqrt":
        sample_w = np.sqrt(np.array([total / (3 * class_counts[y]) for y in data["y_train"]]))
    else:  # "none"
        sample_w = None

    xgb_model = xgb.XGBClassifier(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_lr,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample,
        min_child_weight=args.xgb_min_child,
        gamma=args.xgb_gamma,
        reg_alpha=args.xgb_reg_alpha,
        reg_lambda=args.xgb_reg_lambda,
        random_state=args.seed,
        tree_method="hist",
        verbosity=0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
    )
    xgb_model.fit(Xc_train, data["y_train"], sample_weight=sample_w)

    # XGBoost predictions and probabilities
    xgb_test_probs = xgb_model.predict_proba(Xc_test)
    xgb_test_pred = xgb_test_probs.argmax(-1)
    xgb_test_acc = float(accuracy_score(data["y_test"], xgb_test_pred))
    xgb_test_f1 = float(f1_score(data["y_test"], xgb_test_pred, average="macro", zero_division=0))
    logger.info("XGBoost-only test acc=%.4f f1=%.4f", xgb_test_acc, xgb_test_f1)

    # ----- Hybrid prediction: average XGBoost + transformer probabilities -----
    blended_test_probs = args.blend_alpha * xgb_test_probs + (1 - args.blend_alpha) * te_probs
    test_pred = blended_test_probs.argmax(-1)

    blended_train_probs = args.blend_alpha * xgb_model.predict_proba(Xc_train) + \
                          (1 - args.blend_alpha) * tr_probs
    train_pred = blended_train_probs.argmax(-1)

    train_acc = float(accuracy_score(data["y_train"], train_pred))
    test_acc = float(accuracy_score(data["y_test"], test_pred))
    test_f1 = float(f1_score(data["y_test"], test_pred, average="macro", zero_division=0))

    logger.info(
        "Hybrid (alpha=%.2f XGB + %.2f TX): train_acc=%.4f test_acc=%.4f f1=%.4f",
        args.blend_alpha, 1 - args.blend_alpha, train_acc, test_acc, test_f1,
    )
    logger.info("\n%s", classification_report(
        data["y_test"], test_pred, target_names=["SELL", "HOLD", "BUY"], zero_division=0
    ))

    # ----- Save bundle -----
    Path("models").mkdir(exist_ok=True)
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    version = db.register_model(
        model_key=MODEL_KEY,
        artifact_path="PLACEHOLDER",
        feature_contract=FEATURE_COLS,
        trained_on=data["used_symbols"],
        train_start=start,
        train_end=end,
        train_samples=len(data["y_train"]),
        test_samples=len(data["y_test"]),
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        test_f1=test_f1,
        label_map=LABEL_MAP,
        feature_set_name=FEATURE_SET_NAME,
    )

    artifact_path = f"models/{MODEL_KEY}_v{version}.pkl"
    artifact = {
        "model": xgb_model,
        "scaler": data["scaler"],
        "transformer_state": model.state_dict(),
        "transformer_cfg": tcfg.__dict__,
        "lookback": args.lookback,
        "feature_contract": FEATURE_COLS,
        "feature_set_name": FEATURE_SET_NAME,
        "label_map": LABEL_MAP,
        "confidence_threshold": args.confidence,
        "trained_at": datetime.now().isoformat(),
        "train_symbols": data["used_symbols"],
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "transformer_test_acc": tx_acc,
        "transformer_test_f1": tx_f1,
        "xgboost_test_acc": xgb_test_acc,
        "xgboost_test_f1": xgb_test_f1,
        "blend_alpha": args.blend_alpha,
        "use_embeddings": args.use_embeddings,
    }
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)
    with open(f"models/{MODEL_KEY}.pkl", "wb") as f:
        pickle.dump(artifact, f)

    db.update_artifact_path(MODEL_KEY, version, artifact_path)
    db.deactivate_old_models(MODEL_KEY, version)

    logger.info("Saved hybrid model -> %s (version %d)", artifact_path, version)

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "transformer_test_acc": tx_acc,
        "transformer_test_f1": tx_f1,
        "xgboost_test_acc": xgb_test_acc,
        "xgboost_test_f1": xgb_test_f1,
        "version": version,
        "artifact_path": artifact_path,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train XGBoost-transformer hybrid")
    p.add_argument(
        "--symbols",
        default="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,IWM",
    )
    p.add_argument("--days", type=int, default=1000)
    p.add_argument("--db", default="data/trading_sim.db")
    p.add_argument("--confidence", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")

    # Transformer hyperparams
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dim-feedforward", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12)

    # XGBoost hyperparams
    p.add_argument("--xgb-n-estimators", type=int, default=300)
    p.add_argument("--xgb-max-depth", type=int, default=4)
    p.add_argument("--xgb-lr", type=float, default=0.03)
    p.add_argument("--xgb-subsample", type=float, default=0.85)
    p.add_argument("--xgb-colsample", type=float, default=0.85)
    p.add_argument("--xgb-min-child", type=int, default=2)
    p.add_argument("--xgb-gamma", type=float, default=1.0)
    p.add_argument("--xgb-reg-alpha", type=float, default=0.1)
    p.add_argument("--xgb-reg-lambda", type=float, default=1.0)

    # Hybrid blending
    p.add_argument("--blend-alpha", type=float, default=0.5,
                   help="Weight on XGBoost probs in the blended prediction (1-alpha on transformer)")
    p.add_argument("--use-embeddings", action="store_true",
                   help="Pass the 64-d transformer embedding into XGBoost (default: probs only)")

    # Label noise control
    p.add_argument("--vol-mult", type=float, default=0.5,
                   help="Volatility multiplier for label thresholds (higher -> larger HOLD class)")
    p.add_argument("--xgb-class-weight", choices=["none", "sqrt", "inverse"], default="sqrt",
                   help="XGBoost sample weighting scheme")

    args = p.parse_args()
    stats = train_hybrid(args)
    print("\n" + "=" * 60)
    print("HYBRID TRAINING COMPLETE")
    print("=" * 60)
    print(json.dumps(stats, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
