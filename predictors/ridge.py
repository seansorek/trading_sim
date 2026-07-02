"""Ridge regression predictor for continuous forward-return forecasting."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from predictors.base import BasePredictor, _load_validated_pickle, _preprocess


class RidgePredictor(BasePredictor):
    """Ridge regression predictor returning continuous forward-return forecasts.

    Pickle keys: model, scaler, feature_contract, train_ic, train_r2, model_type.
    Produced by train_predictor.py.

    Because Ridge output is continuous (not a class distribution), predict()
    returns proba=None. Pair with QuantileDecision for signal generation.
    """

    def __init__(self, model, scaler, train_ic: float = 0.0, train_r2: float = 0.0):
        self.model = model
        self.scaler = scaler
        self.train_ic = float(train_ic)
        self.train_r2 = float(train_r2)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_clean = _preprocess(X.copy().astype(np.float32))
        X_scaled = self.scaler.transform(X_clean)
        scores = self.model.predict(X_scaled).astype(float)
        return scores, None

    @classmethod
    def load(cls, path: str) -> "RidgePredictor":
        data = _load_validated_pickle(
            path,
            "RidgePredictor",
            required_keys={"model", "scaler", "feature_contract"},
        )
        return cls(
            model=data["model"],
            scaler=data["scaler"],
            train_ic=data.get("train_ic", 0.0),
            train_r2=data.get("train_r2", 0.0),
        )
