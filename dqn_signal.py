"""
dqn_signal.py -- Shared DQN signal gating logic.

Both predict_next_day_lite.py and simulation_pipeline.py use this to
decide whether Q-values are decisive enough to emit BUY/SELL (vs. HOLD).
Keeping the logic in one place prevents drift between live prediction
and backtesting.
"""
from __future__ import annotations

import numpy as np


def gate_dqn_signal(
    q_vals: np.ndarray,
    confidence_threshold: float,
    q_advantage_threshold: float,
) -> tuple[str, float]:
    """Apply confidence and Q-advantage gating to raw DQN Q-values.

    Parameters
    ----------
    q_vals : ndarray of shape (3,)
        Q-values for [Hold, Long, Short].
    confidence_threshold : float
        Minimum spread (q_max - q_min) required to act.
    q_advantage_threshold : float
        Minimum advantage of the chosen action over Hold to act.

    Returns
    -------
    signal : str
        One of "BUY", "SELL", "HOLD".
    confidence : float
        The Q-value spread (q_max - q_min), useful for logging.
    """
    q_hold, q_long, q_short = float(q_vals[0]), float(q_vals[1]), float(q_vals[2])
    q_max = float(q_vals.max())
    q_min = float(q_vals.min())
    confidence = q_max - q_min

    signal = "HOLD"
    if confidence >= confidence_threshold:
        if q_long == q_max and q_long - q_hold > q_advantage_threshold:
            signal = "BUY"
        elif q_short == q_max and q_short - q_hold > q_advantage_threshold:
            signal = "SELL"

    return signal, confidence
