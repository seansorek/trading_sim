from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from predictors.base import BasePredictor


class DQNPredictor(BasePredictor):
    """DQN agent predictor returning Q-values per bar.

    Applies online (causal) normalization fitted on the first `fit_window` bars,
    then slides a look-back window across the series to produce Q-values.

    predict() returns:
      scores  : Q(Long) - Q(Short), shape (N,)
      proba   : (N, 3) Q-value matrix, columns [Hold=0, Long=1, Short=2]

    Warmup bars (indices < window) have all-zero Q-values and scores.
    """

    def __init__(
        self,
        agent,
        window: int = 20,
        fit_window: int = 252,
        confidence_threshold: float = 2.0,
        q_advantage_threshold: float = 1.0,
    ):
        self.agent = agent
        self.window = int(window)
        self.fit_window = int(fit_window)
        self.confidence_threshold = float(confidence_threshold)
        self.q_advantage_threshold = float(q_advantage_threshold)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X = np.nan_to_num(X.astype(np.float32), nan=0.0)
        fit_end = min(self.fit_window, max(self.window + 1, len(X) // 2))

        means = X[:fit_end].mean(axis=0)
        stds = X[:fit_end].std(axis=0)
        stds = np.where(stds > 0, stds, 1.0)
        X_norm = (X - means) / stds

        q_matrix = np.zeros((len(X), 3), dtype=np.float32)
        for i in range(self.window, len(X_norm)):
            state = X_norm[i - self.window : i].flatten()
            with torch.no_grad():
                s_t = torch.from_numpy(state).float().unsqueeze(0)
                q_vals = self.agent.q(s_t).squeeze(0).cpu().numpy()
            q_matrix[i] = q_vals

        # scores = Q(Long) - Q(Short): positive → bullish, negative → bearish
        scores = (q_matrix[:, 1] - q_matrix[:, 2]).astype(float)
        return scores, q_matrix

    @classmethod
    def load(cls, path: str, **kwargs) -> "DQNPredictor":
        from dqn_agent import DQNAgent  # lazy import to avoid eager torch load
        agent = DQNAgent.load(path)
        return cls(agent, **kwargs)
