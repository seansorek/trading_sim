"""Abstract base for decision layers and the DecisionContext dataclass."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DecisionContext:
    """Contextual information the decision layer may need beyond raw scores."""
    index: pd.DatetimeIndex
    symbol: str


class BaseDecisionLayer(abc.ABC):
    """Converts predictor scores to discrete trading signals {-1, 0, 1}."""

    @abc.abstractmethod
    def decide(
        self,
        scores: np.ndarray,
        proba: Optional[np.ndarray],
        ctx: DecisionContext,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        scores : np.ndarray, shape (N,)
            Signed continuous scores from BasePredictor.predict().
        proba : np.ndarray or None
            Class probabilities or Q-values from BasePredictor.predict().
        ctx : DecisionContext
            Timestamp index and symbol name.

        Returns
        -------
        np.ndarray, shape (N,), dtype int
            Values in {-1 (SELL), 0 (HOLD), 1 (BUY)}.
        """
