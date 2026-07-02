"""Logistic regression predictor for next-day return class."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from predictors.base import BasePredictor, _load_validated_pickle, _preprocess


class LogisticPredictor(BasePredictor):
    """Logistic regression predictor for next-day return class.

    Loads from a pickle produced by train_models.py.
    Handles its own StandardScaler and _preprocess pass.
    """

    def __init__(self, model, scaler, confidence_threshold: float = 0.55):
        self.model = model
        self.scaler = scaler
        self.confidence_threshold = float(confidence_threshold)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_clean = _preprocess(X.copy().astype(np.float32))
        X_scaled = self.scaler.transform(X_clean)
        proba = self.model.predict_proba(X_scaled)
        pred_idx = np.argmax(proba, axis=1)
        # Map through model.classes_ to handle missing classes correctly
        scores = (self.model.classes_[pred_idx] - 1).astype(float)  # {0,1,2} → {-1,0,1}
        return scores, proba

    @classmethod
    def load(cls, path: str) -> "LogisticPredictor":
        data = _load_validated_pickle(path, "LogisticPredictor")
        return cls(
            model=data["model"],
            scaler=data["scaler"],
            confidence_threshold=data.get("confidence_threshold", 0.55),
        )
