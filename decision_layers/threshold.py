"""Threshold-based decision layer."""
from __future__ import annotations

from typing import Optional

import numpy as np

from decision_layers.base import BaseDecisionLayer, DecisionContext


class ThresholdDecision(BaseDecisionLayer):
    """Emit a signal only when predictor confidence exceeds a threshold.

    Works with classifier predictors whose predict_proba returns (N, K)
    probability matrices. When proba is None, passes score sign through directly.
    """

    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = float(confidence_threshold)

    def decide(
        self,
        scores: np.ndarray,
        proba: Optional[np.ndarray],
        ctx: DecisionContext,
    ) -> np.ndarray:
        signals = np.sign(scores).astype(int)
        if proba is not None:
            confidence = proba.max(axis=1)
            signals[confidence < self.confidence_threshold] = 0
        return signals
