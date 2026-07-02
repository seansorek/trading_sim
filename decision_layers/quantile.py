"""Quantile-based decision layer."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from decision_layers.base import BaseDecisionLayer, DecisionContext


class QuantileDecision(BaseDecisionLayer):
    """Trade only the top fraction of bars by predicted-score magnitude.

    Uses a causal rolling quantile of |score| over the past `threshold_window`
    bars, shifted 1 bar to prevent look-ahead. Bars whose |score| does not
    clear the quantile threshold become HOLD.

    Parameters
    ----------
    signal_quantile : float
        Rolling quantile threshold (0–1). E.g. 0.7 means trade only bars
        in the top 30% by magnitude.
    threshold_window : int
        Look-back window for the rolling quantile.
    """

    def __init__(self, signal_quantile: float = 0.7, threshold_window: int = 63):
        self.signal_quantile = float(signal_quantile)
        self.threshold_window = int(threshold_window)

    def decide(
        self,
        scores: np.ndarray,
        proba: Optional[np.ndarray],
        ctx: DecisionContext,
    ) -> np.ndarray:
        magnitudes = np.abs(scores).astype(float)
        ser = pd.Series(magnitudes, index=ctx.index)
        # shift(1): today's threshold was set by yesterday's history (causal)
        rolling_q = (
            ser
            .rolling(self.threshold_window, min_periods=1)
            .quantile(self.signal_quantile)
            .shift(1)
        )
        # NaN at bar 0 (no prior history) → fill with inf so no trade fires
        threshold = rolling_q.fillna(np.inf).values
        direction = np.sign(scores).astype(int)
        return np.where(magnitudes >= threshold, direction, 0).astype(int)
