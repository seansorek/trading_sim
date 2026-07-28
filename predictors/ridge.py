"""Ridge regression predictor for continuous forward-return forecasting."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from predictors.base import BasePredictor, _load_validated_pickle, _scale


class RidgePredictor(BasePredictor):
    """Continuous forward-return regressor: Ridge, ElasticNet, or XGBRegressor.

    Named for its first and default occupant. Every regressor train_predictor.py
    produces is fit through the same scaler+clip path, so serving them differs
    only in the pickled estimator — a second class would be identical code.

    Pickle keys: model, scaler, feature_contract, train_ic, train_r2, model_type.

    Because the output is continuous (not a class distribution), predict()
    returns proba=None. Pair with QuantileDecision for signal generation.
    """

    def __init__(
        self,
        model,
        scaler,
        train_ic: float = 0.0,
        train_r2: float = 0.0,
        cs_mode: str = "off",
    ):
        self.model = model
        self.scaler = scaler
        self.train_ic = float(train_ic)
        self.train_r2 = float(train_r2)
        # Which feature-normalization axes this model was fit on (see
        # daily_features.CS_MODES). Callers with a full cross-section
        # (panel_data) must build the matrix the same way; feeding the wrong
        # axis is a silent train/serve mismatch, not an error.
        self.cs_mode = str(cs_mode)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        scores = self.model.predict(_scale(self.scaler, X.astype(np.float32))).astype(float)
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
            # cs_normalized is the superseded boolean form of the same field.
            cs_mode=data.get("cs_mode") or ("replace" if data.get("cs_normalized") else "off"),
        )
