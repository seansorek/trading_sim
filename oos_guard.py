"""oos_guard.py — Enforce out-of-sample date boundaries for pretrained-model backtests.

A pretrained model artifact (see train_models.py, train_predictor.py,
train_hybrid.py) records ``train_end`` — the last date of data used to fit
it. Backtesting that model over a date range that dips into
[train_start, train_end] reports in-sample performance dressed up as
out-of-sample evidence (see issue #115).

This module provides a single choke point — ``get_artifact_train_end`` to
read the cutoff from a loaded artifact dict, and ``enforce_oos_start`` to
trim (or reject) a price frame so no row at or before
``train_end + embargo`` can reach the backtester.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from daily_features import FWD_RET_HORIZON_DAYS

logger = logging.getLogger(__name__)


def get_artifact_train_end(artifact: dict) -> Optional[pd.Timestamp]:
    """Extract the training-data cutoff date from a loaded model artifact dict.

    Returns None if the artifact predates the ``train_end`` field (older
    pickles trained before this field was added) — callers should treat
    that as "unknown cutoff, cannot enforce" rather than crash.
    """
    raw = artifact.get("train_end")
    if not raw:
        return None
    try:
        return pd.Timestamp(raw)
    except (ValueError, TypeError):
        return None


def enforce_oos_start(
    df: pd.DataFrame,
    train_end,
    embargo_days: int = FWD_RET_HORIZON_DAYS,
    *,
    label: str = "",
    strict: bool = False,
) -> pd.DataFrame:
    """Trim `df` to rows strictly after `train_end + embargo_days`.

    Parameters
    ----------
    df : DataFrame indexed by date/datetime (as produced by load_yfinance etc).
    train_end : the model's training-data cutoff (str or Timestamp). If None,
        the boundary is unknown and `df` is returned unchanged (with a
        warning) — this only happens for artifacts saved before this field
        existed.
    embargo_days : extra calendar days beyond train_end to also exclude, so
        a purge/embargo gap is preserved the same way it is during training.
    label : optional string (e.g. "AAPL/daily_xgboost") used in log messages.
    strict : if True, raise ValueError instead of returning an empty frame
        when trimming would remove every row (i.e. the entire requested
        range is in-sample).

    Returns
    -------
    The trimmed DataFrame (a view/copy of the rows after the cutoff).
    """
    if train_end is None:
        logger.warning(
            "[oos_guard]%s train_end unknown on model artifact — cannot enforce "
            "out-of-sample boundary; backtest may include in-sample rows.",
            f" {label}:" if label else "",
        )
        return df

    cutoff = pd.Timestamp(train_end) + pd.Timedelta(days=embargo_days)
    # Normalize both sides to naive timestamps for comparison, since df's
    # index may be tz-aware (yfinance) while train_end is a plain date string.
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        cmp_idx = idx.tz_localize(None)
    else:
        cmp_idx = idx
    if cutoff.tz is not None:
        cutoff = cutoff.tz_localize(None)

    mask = cmp_idx > cutoff
    n_dropped = int((~mask).sum())
    trimmed = df.loc[mask]

    if n_dropped:
        logger.warning(
            "[oos_guard]%s dropped %d/%d rows at/before train cutoff %s "
            "(embargo=%dd) to enforce out-of-sample backtesting.",
            f" {label}:" if label else "", n_dropped, len(df), cutoff.date(), embargo_days,
        )

    if trimmed.empty and strict:
        raise ValueError(
            f"[oos_guard]{' ' + label + ':' if label else ''} requested backtest range is "
            f"entirely in-sample (train cutoff {cutoff.date()} + embargo {embargo_days}d "
            "covers the whole range). Choose a later start date or a model trained on "
            "an earlier window."
        )

    return trimmed
