"""
panel_data.py — Build aligned cross-sectional panels for panel_backtester.

Research-only: nothing here touches the live prediction path.

Produces three date x symbol frames — predictions, 1-day forward returns, and
closes — outer-joined across symbols with ragged histories left as NaN. A NaN
means "this symbol has no cross-section entry today", which is the correct
signal for the ranker to skip it. Never forward-fill: commit 3255f87 fixed
exactly that leak in data_loader._standardize.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from daily_features import (
    FEATURE_COLS,
    cross_sectional_normalize,
    cs_feature_cols,
    make_daily_features,
)
from predictors.ridge import RidgePredictor

logger = logging.getLogger(__name__)

# Above this fraction of the configured universe failing to load, raise rather
# than silently backtest a truncated universe — a 40-name result reported as a
# 150-name result describes nothing.
MAX_LOAD_FAILURE_FRAC = 0.10

# Trailing window for the rolling beta used to neutralize the book. 126 bars
# (~6 months) trades estimation noise against staleness; min_periods leaves
# beta NaN early, which panel_backtester reads as "unknown, don't size on it".
BETA_WINDOW = 126
BETA_MIN_PERIODS = 60


@dataclass
class PanelData:
    pred: pd.DataFrame    # date x symbol — Ridge forecast (3-day cumulative target)
    ret: pd.DataFrame     # date x symbol — 1-day simple return, close[t] -> close[t+1]
    close: pd.DataFrame   # date x symbol
    symbols: list[str]
    dropped: dict[str, str]
    # date x symbol trailing beta vs SPY. None when spy_df was not supplied, in
    # which case the backtester falls back to dollar-neutral-only weighting.
    beta: pd.DataFrame | None = None


def rolling_beta(sym_ret: pd.Series, spy_ret: pd.Series) -> pd.Series:
    """Trailing OLS beta of one symbol vs SPY, using bars up to and including t.

    Causal: rolling() spans past+present only, and weights formed on date t earn
    ret[t] = close[t] -> close[t+1], so a beta through t uses nothing unknown at
    decision time.
    """
    spy = spy_ret.reindex(sym_ret.index)
    cov = sym_ret.rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS).cov(spy)
    var = spy.rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS).var()
    return cov / var.where(var > 0)


def build_panels(
    symbols: list[str],
    start: str,
    end: str,
    db,
    model_path: str = "models/daily_predictor.pkl",
    spy_df: pd.DataFrame | None = None,
    min_bars: int = 250,
    predictor=None,
    load_fn=None,
    cs_mode: str | None = None,
) -> PanelData:
    """Load bars, featurize, predict, and assemble aligned panels.

    cs_mode=None (the default) follows the model's own flag, so a model trained
    on per-date ranked features is always served them. Pass an explicit mode
    only to force the mismatch deliberately, e.g. an A/B.

    predictor / load_fn are injection seams for tests; production callers omit
    both and get RidgePredictor.load(model_path) and train_models._load_symbol.
    """
    if predictor is None:
        predictor = RidgePredictor.load(model_path)
    if load_fn is None:
        from train_models import _load_symbol  # deferred: heavy import chain
        load_fn = _load_symbol

    feat_frames: dict[str, pd.DataFrame] = {}
    rets: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    betas: dict[str, pd.Series] = {}
    dropped: dict[str, str] = {}

    spy_ret = (
        spy_df["close"].astype(float).pct_change() if spy_df is not None else None
    )

    for symbol in symbols:
        try:
            df = load_fn(symbol, start, end, db)
        except Exception as exc:
            dropped[symbol] = f"load failed: {exc}"
            continue

        n_bars = 0 if df is None else len(df)
        if df is None or n_bars < min_bars:
            dropped[symbol] = f"insufficient bars ({n_bars} < {min_bars})"
            continue

        try:
            spy_arg = spy_df if symbol != "SPY" else None
            feats = make_daily_features(df, spy_df=spy_arg)
        except Exception as exc:
            dropped[symbol] = f"feature error: {exc}"
            continue

        if feats.empty:
            dropped[symbol] = "no rows survived feature warmup"
            continue

        close = feats["close"].astype(float)
        feat_frames[symbol] = feats
        closes[symbol] = close
        # 1-day simple forward return. NOT feats["fwd_ret_1d"], which is a
        # FWD_RET_HORIZON_DAYS-bar cumulative return despite its name — paying
        # that per daily bar would triple the book's return.
        rets[symbol] = close.pct_change().shift(-1)
        if spy_ret is not None:
            betas[symbol] = rolling_beta(close.pct_change(), spy_ret)

    if not feat_frames:
        raise RuntimeError(f"No symbols produced features. Dropped: {dropped}")

    fail_frac = len(dropped) / len(symbols)
    if fail_frac > MAX_LOAD_FAILURE_FRAC:
        raise RuntimeError(
            f"Refusing to backtest a truncated universe: {len(dropped)}/{len(symbols)} "
            f"symbols failed ({fail_frac:.0%} > {MAX_LOAD_FAILURE_FRAC:.0%}). "
            f"Dropped: {dropped}"
        )

    if dropped:
        logger.warning("Dropped %d/%d symbols: %s", len(dropped), len(symbols), dropped)

    # Predict on one long (date, symbol) frame rather than symbol by symbol:
    # cross-sectional normalization needs every name on a date at once, and
    # routing both cases through the same call keeps the cs and non-cs paths
    # from drifting apart.
    blocks = []
    for symbol, feats in feat_frames.items():
        block = feats[FEATURE_COLS].copy()
        block["_date"] = feats.index
        block["_symbol"] = symbol
        blocks.append(block)
    long = pd.concat(blocks, ignore_index=True)

    if cs_mode is None:
        cs_mode = getattr(predictor, "cs_mode", "off")
    model_cols = cs_feature_cols(cs_mode)
    if cs_mode != "off":
        before = len(long)
        long = cross_sectional_normalize(long, mode=cs_mode).dropna(subset=model_cols)
        logger.info(
            "Feature axis %r: %d columns, %d rows (%d dropped as thin dates)",
            cs_mode, len(model_cols), len(long), before - len(long),
        )

    scores, _ = predictor.predict(long[model_cols].values.astype(np.float32))
    long["_score"] = scores
    # pivot (not pivot_table): a duplicate (date, symbol) is a data bug worth
    # raising on, not something to silently average.
    pred_panel = long.pivot(index="_date", columns="_symbol", values="_score").sort_index()
    pred_panel.index.name = None
    pred_panel.columns.name = None

    ret_panel = pd.DataFrame(rets).reindex(pred_panel.index)
    close_panel = pd.DataFrame(closes).reindex(pred_panel.index)

    return PanelData(
        pred=pred_panel,
        ret=ret_panel,
        close=close_panel,
        symbols=sorted(pred_panel.columns),
        dropped=dropped,
        beta=pd.DataFrame(betas).reindex(pred_panel.index) if betas else None,
    )
