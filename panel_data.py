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

from daily_features import FEATURE_COLS, make_daily_features
from predictors.ridge import RidgePredictor

logger = logging.getLogger(__name__)

# Above this fraction of the configured universe failing to load, raise rather
# than silently backtest a truncated universe — a 40-name result reported as a
# 150-name result describes nothing.
MAX_LOAD_FAILURE_FRAC = 0.10


@dataclass
class PanelData:
    pred: pd.DataFrame    # date x symbol — Ridge forecast (3-day cumulative target)
    ret: pd.DataFrame     # date x symbol — 1-day simple return, close[t] -> close[t+1]
    close: pd.DataFrame   # date x symbol
    symbols: list[str]
    dropped: dict[str, str]


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
) -> PanelData:
    """Load bars, featurize, predict, and assemble aligned panels.

    predictor / load_fn are injection seams for tests; production callers omit
    both and get RidgePredictor.load(model_path) and train_models._load_symbol.
    """
    if predictor is None:
        predictor = RidgePredictor.load(model_path)
    if load_fn is None:
        from train_models import _load_symbol  # deferred: heavy import chain
        load_fn = _load_symbol

    preds: dict[str, pd.Series] = {}
    rets: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    dropped: dict[str, str] = {}

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

        X = feats[FEATURE_COLS].values.astype(np.float32)
        scores, _ = predictor.predict(X)

        close = feats["close"].astype(float)
        preds[symbol] = pd.Series(scores, index=feats.index)
        closes[symbol] = close
        # 1-day simple forward return. NOT feats["fwd_ret_1d"], which is a
        # FWD_RET_HORIZON_DAYS-bar cumulative return despite its name — paying
        # that per daily bar would triple the book's return.
        rets[symbol] = close.pct_change().shift(-1)

    if not preds:
        raise RuntimeError(f"No symbols produced predictions. Dropped: {dropped}")

    fail_frac = len(dropped) / len(symbols)
    if fail_frac > MAX_LOAD_FAILURE_FRAC:
        raise RuntimeError(
            f"Refusing to backtest a truncated universe: {len(dropped)}/{len(symbols)} "
            f"symbols failed ({fail_frac:.0%} > {MAX_LOAD_FAILURE_FRAC:.0%}). "
            f"Dropped: {dropped}"
        )

    if dropped:
        logger.warning("Dropped %d/%d symbols: %s", len(dropped), len(symbols), dropped)

    # DataFrame(dict_of_series) outer-joins on index. NaN stays NaN.
    pred_panel = pd.DataFrame(preds).sort_index()
    ret_panel = pd.DataFrame(rets).reindex(pred_panel.index)
    close_panel = pd.DataFrame(closes).reindex(pred_panel.index)

    return PanelData(
        pred=pred_panel,
        ret=ret_panel,
        close=close_panel,
        symbols=sorted(preds),
        dropped=dropped,
    )
