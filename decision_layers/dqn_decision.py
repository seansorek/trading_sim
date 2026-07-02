"""DQN-specific decision layer."""
from __future__ import annotations

from typing import Optional

import numpy as np

from decision_layers.base import BaseDecisionLayer, DecisionContext
from dqn_signal import gate_dqn_signal

_SIGNAL_MAP = {"BUY": 1, "SELL": -1, "HOLD": 0}


class DQNDecision(BaseDecisionLayer):
    """Apply DQN-specific Q-value gating via gate_dqn_signal.

    Requires proba to be a (N, 3) Q-value matrix with columns
    [Hold=0, Long=1, Short=2]. Warmup bars (all-zero Q-values) emit HOLD.

    Parameters
    ----------
    confidence_threshold : float
        Minimum Q-spread (q_max - q_min) to act.
    q_advantage_threshold : float
        Minimum advantage of the chosen action over Hold to act.
    """

    def __init__(
        self,
        confidence_threshold: float = 2.0,
        q_advantage_threshold: float = 1.0,
    ):
        self.confidence_threshold = float(confidence_threshold)
        self.q_advantage_threshold = float(q_advantage_threshold)

    def decide(
        self,
        scores: np.ndarray,
        proba: Optional[np.ndarray],
        ctx: DecisionContext,
    ) -> np.ndarray:
        if proba is None:
            return np.zeros(len(scores), dtype=int)

        signals = np.zeros(len(proba), dtype=int)
        for i, q_vals in enumerate(proba):
            if np.all(q_vals == 0.0):  # warmup bars produce no signal
                continue
            sig_str, _ = gate_dqn_signal(
                q_vals,
                confidence_threshold=self.confidence_threshold,
                q_advantage_threshold=self.q_advantage_threshold,
            )
            signals[i] = _SIGNAL_MAP[sig_str]
        return signals
