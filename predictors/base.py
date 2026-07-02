"""Base class and shared utilities for all predictor implementations."""
from __future__ import annotations

import abc
import hashlib
import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np

from daily_features import FEATURE_COLS

logger = logging.getLogger(__name__)


def _preprocess(X: np.ndarray) -> np.ndarray:
    """Replace inf/nan with 0, clip to ±5 std per column. Returns a copy."""
    X = X.copy()
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X, nan=0.0)
    for col in range(X.shape[1]):
        col_data = X[:, col]
        std = np.std(col_data)
        if std > 0:
            mean = np.mean(col_data)
            X[:, col] = np.clip(col_data, mean - 5 * std, mean + 5 * std)
    return X


def _load_validated_pickle(
    path: str,
    name: str,
    required_keys: Optional[set] = None,
) -> dict:
    """Load a model pickle, verify SHA-256 integrity, and check required keys.

    Raises RuntimeError on any failure — never silently continues.
    """
    if required_keys is None:
        required_keys = {"model", "scaler", "feature_contract"}

    if not os.path.exists(path):
        raise RuntimeError(
            f"[{name}] Model file not found: {path}. Run the appropriate train_*.py first."
        )

    hash_path = path + ".sha256"
    if os.path.exists(hash_path):
        with open(hash_path, encoding="ascii") as fh:
            expected = fh.read().strip()
        with open(path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"[{name}] Integrity check failed for {path}: SHA-256 mismatch. Retrain."
            )
    else:
        logger.warning(
            "[%s] No hash file for %s — skipping integrity check. Retrain to enable it.",
            name,
            path,
        )

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(f"[{name}] Failed to unpickle {path}: {exc}") from exc

    missing = required_keys - data.keys()
    if missing:
        raise RuntimeError(
            f"[{name}] Pickle {path} missing keys: {missing}. Retrain."
        )

    if "feature_contract" in required_keys and data["feature_contract"] != FEATURE_COLS:
        raise RuntimeError(
            f"[{name}] Feature contract mismatch in {path}. "
            f"Expected {len(FEATURE_COLS)} features, got {len(data['feature_contract'])}. "
            "Retrain models."
        )

    return data


class BasePredictor(abc.ABC):
    """Abstract base for all prediction models.

    Owns its own scaler and preprocessing. Receives raw (unscaled) feature
    arrays from PredictorStrategy and returns signed continuous scores.
    """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Map raw features to signal scores.

        Parameters
        ----------
        X : np.ndarray, shape (N, F)
            Raw feature values in FEATURE_COLS order, float32, may contain NaN/inf.

        Returns
        -------
        scores : np.ndarray, shape (N,)
            Signed continuous signal strength.
            Positive → bullish, negative → bearish, 0 → neutral.
        proba : np.ndarray, shape (N, K), or None
            For classifiers: class probability matrix (columns match model.classes_).
            For RL predictors: Q-value matrix (columns [Hold, Long, Short]).
            For regressors: None.
        """

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str) -> "BasePredictor":
        """Load a trained predictor from a pickle file."""
