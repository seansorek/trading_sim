"""
ml_strategies.py — Machine learning trading strategies.

Strategy classes for daily prediction (DailyLogistic, DailyXGBoost).
Intraday strategies (OrdinalLogistic, XGBoost) kept for reference but not
used in the daily pipeline.

Feature contract: always use FEATURE_COLS from daily_features. Never infer
column order from DataFrame iteration — training and prediction must agree.
"""
import hashlib
import logging
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from base_strategy import BaseStrategy, StrategyConfig
from daily_features import FEATURE_COLS, make_daily_features

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _preprocess(X: np.ndarray) -> np.ndarray:
    """Replace inf/nan with 0, clip to ±5 std per column."""
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X, nan=0.0)
    for col in range(X.shape[1]):
        col_data = X[:, col]
        std = np.std(col_data)
        if std > 0:
            mean = np.mean(col_data)
            X[:, col] = np.clip(col_data, mean - 5 * std, mean + 5 * std)
    return X


def _load_pickle(path: str, strategy_name: str) -> dict:
    """
    Load a model pickle and validate it contains the required keys.

    Raises RuntimeError (not a silent HOLD) if the file is missing or
    has an incompatible structure.
    """
    if not os.path.exists(path):
        raise RuntimeError(
            f"[{strategy_name}] Model file not found: {path}. "
            "Run train_models.py first."
        )
    hash_path = path + ".sha256"
    if os.path.exists(hash_path):
        expected = open(hash_path, encoding="ascii").read().strip()
        with open(path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"[{strategy_name}] Integrity check failed for {path}: "
                "SHA-256 mismatch. The file may have been tampered with. Retrain."
            )
    else:
        logger.warning(
            "[%s] No hash file found for %s — skipping integrity check. "
            "Retrain with the current train_models.py to enable verification.",
            strategy_name, path,
        )
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"[{strategy_name}] Failed to unpickle {path}: {exc}"
        ) from exc

    required = {"model", "scaler", "feature_contract"}
    missing = required - data.keys()
    if missing:
        raise RuntimeError(
            f"[{strategy_name}] Pickle {path} is missing keys: {missing}. "
            "Retrain with the current train_models.py."
        )
    if data["feature_contract"] != FEATURE_COLS:
        raise RuntimeError(
            f"[{strategy_name}] Feature contract mismatch in {path}. "
            f"Expected {len(FEATURE_COLS)} features, got {len(data['feature_contract'])}. "
            "Retrain models."
        )
    return data


def _apply_confidence_filter(
    preds: np.ndarray, probs: np.ndarray, threshold: float
) -> np.ndarray:
    """Zero out predictions whose max probability is below threshold."""
    confidence = probs.max(axis=1)
    result = preds.copy()
    result[confidence < threshold] = 0
    return result


# ---------------------------------------------------------------------------
# Daily strategies
# ---------------------------------------------------------------------------

class DailyLogisticStrategy(BaseStrategy):
    """
    Logistic regression predicting next-day return class using daily features.

    Loads from a pickle with structure:
      {'model': LogisticRegression, 'scaler': StandardScaler,
       'feature_contract': FEATURE_COLS, 'confidence_threshold': float, ...}

    Raises RuntimeError on load failure — never silently returns all-HOLD.
    """

    def __init__(
        self,
        cfg: StrategyConfig,
        use_pretrained: bool = True,
        confidence_threshold: float = 0.55,
        model_path: str = "models/daily_logistic.pkl",
        spy_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__(cfg)
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn not installed.")

        self.spy_df = spy_df
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.confidence_threshold = float(
            os.environ.get("LOGISTIC_CONFIDENCE_THRESHOLD", confidence_threshold)
        )

        if use_pretrained:
            data = _load_pickle(model_path, "DailyLogisticStrategy")
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.confidence_threshold = data.get(
                "confidence_threshold", self.confidence_threshold
            )
            logger.info("DailyLogisticStrategy: loaded model from %s", model_path)

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        daily_feats = make_daily_features(df, spy_df=self.spy_df)
        X = _preprocess(daily_feats[FEATURE_COLS].values.astype(np.float32))

        if self.model is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            probs = self.model.predict_proba(X_scaled)
            # Map through model.classes_ instead of assuming column i == class i
            classes = self.model.classes_
            preds = classes[np.argmax(probs, axis=1)]
            preds = _apply_confidence_filter(preds, probs, self.confidence_threshold)
            signals = preds - 1  # {0,1,2} -> {-1,0,1}
        else:
            # In-session training fallback (no pre-trained model)
            y_raw = daily_feats["fwd_ret_1d"].values
            mask = ~np.isnan(y_raw)
            from daily_features import discretize_labels
            y = discretize_labels(y_raw)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LogisticRegression(
                max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=42
            )
            model.fit(X_scaled[mask], y[mask])
            probs = model.predict_proba(X_scaled)
            classes = model.classes_
            preds = classes[np.argmax(probs, axis=1)]
            preds = _apply_confidence_filter(preds, probs, self.confidence_threshold)
            signals = preds - 1

        raw = self._apply_holding_period(pd.Series(signals, index=daily_feats.index))
        # Shift by one bar: decision from close[D] executes on bar D+1
        return raw.shift(1).fillna(0).astype(int)


class DailyXGBoostStrategy(BaseStrategy):
    """
    XGBoost classifier predicting next-day return class using daily features.

    Loads from a pickle with the same structure as DailyLogisticStrategy.
    No randomness in signal generation — results are deterministic given seeds.
    """

    def __init__(
        self,
        cfg: StrategyConfig,
        use_pretrained: bool = True,
        confidence_threshold: float = 0.55,
        model_path: str = "models/daily_xgboost.pkl",
        spy_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__(cfg)
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed.")

        self.spy_df = spy_df
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.confidence_threshold = float(
            os.environ.get("XGB_CONFIDENCE_THRESHOLD", confidence_threshold)
        )

        if use_pretrained:
            data = _load_pickle(model_path, "DailyXGBoostStrategy")
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.confidence_threshold = data.get(
                "confidence_threshold", self.confidence_threshold
            )
            logger.info("DailyXGBoostStrategy: loaded model from %s", model_path)

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        daily_feats = make_daily_features(df, spy_df=self.spy_df)
        X = _preprocess(daily_feats[FEATURE_COLS].values.astype(np.float32))

        if self.model is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            probs = self.model.predict_proba(X_scaled)
            classes = self.model.classes_
            preds = classes[np.argmax(probs, axis=1)]
            preds = _apply_confidence_filter(preds, probs, self.confidence_threshold)
            signals = preds - 1
        else:
            from daily_features import discretize_labels
            y_raw = daily_feats["fwd_ret_1d"].values
            mask = ~np.isnan(y_raw)
            y = discretize_labels(y_raw)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                gamma=1.0,
                random_state=42,
                tree_method="hist",
                verbosity=0,
            )
            model.fit(X_scaled[mask], y[mask])
            probs = model.predict_proba(X_scaled)
            classes = model.classes_
            preds = classes[np.argmax(probs, axis=1)]
            preds = _apply_confidence_filter(preds, probs, self.confidence_threshold)
            signals = preds - 1

        raw = self._apply_holding_period(pd.Series(signals, index=daily_feats.index))
        return raw.shift(1).fillna(0).astype(int)


def compute_predictor_signal(
    pred_ret: np.ndarray, signal_quantile: float, threshold_window: int
) -> np.ndarray:
    """
    Causal rolling-quantile decision layer for a continuous return forecast.

    Single source of truth for the daily_predictor decision logic — called
    by both DailyPredictorStrategy.signal() (backtest) and
    predict_next_day_lite.py's live-prediction path, so the two can never
    silently diverge. A fixed vol-scaled band does not work here because a
    regularized regressor's predictions are shrunk toward zero on a
    different scale than raw returns (empirically ~6x smaller) — this
    adapts to whatever scale a given prediction model produces by trading
    only the top `1 - signal_quantile` fraction of the trailing
    `threshold_window` bars' |prediction| magnitudes. The threshold is
    shifted by one bar so it never sees today's own prediction.

    Returns an int array of {-1, 0, 1} (SELL/HOLD/BUY) the same length as
    pred_ret. Bars before `threshold_window`'s min_periods is satisfied
    default to HOLD (0), since the threshold is undefined (NaN) and any
    NaN comparison is False.
    """
    pred_series = pd.Series(pred_ret)
    abs_pred = pred_series.abs()
    rolling_thr = (
        abs_pred.rolling(threshold_window, min_periods=20)
        .quantile(signal_quantile)
        .shift(1)
    )
    trigger = (abs_pred > rolling_thr).values
    actions = np.ones(len(pred_series), dtype=int)  # default: HOLD
    actions[trigger & (pred_ret > 0)] = 2  # BUY
    actions[trigger & (pred_ret < 0)] = 0  # SELL
    return actions - 1  # {0,1,2} -> {-1,0,1}


class DailyPredictorStrategy(BaseStrategy):
    """
    Decision layer over the daily_predictor regression model.

    This is the "strategy" half of the prediction/strategy split (see
    train_predictor.py for the "prediction" half and the rationale). The
    model forecasts a continuous forward return; this class converts that
    forecast into a SELL/HOLD/BUY decision independently of how the forecast
    was produced, so the decision policy can be re-tuned (or replaced
    entirely) without retraining the prediction model.

    Threshold design: a fixed vol-scaled band (the discretize_labels shape
    used for training-time labels) does NOT work here, because a regularized
    regressor's predictions are shrunk toward zero and have a different scale
    than raw returns — empirically ~6x smaller than the vol*sqrt(h) band at
    the same multiplier. Instead this uses a causal rolling quantile of
    |predicted return|: trade only when today's prediction magnitude exceeds
    the top `1 - signal_quantile` fraction of the trailing `threshold_window`
    bars' predictions (shifted by one bar, so the threshold never sees
    today's own prediction — safe for live use). This adapts automatically to
    whatever scale a given prediction model produces.

    Loads from a pickle with structure:
      {'model': Ridge, 'scaler': StandardScaler, 'feature_contract': FEATURE_COLS, ...}

    Raises RuntimeError on load failure — never silently returns all-HOLD.
    """

    def __init__(
        self,
        cfg: StrategyConfig,
        use_pretrained: bool = True,
        signal_quantile: float = 0.7,
        threshold_window: int = 60,
        model_path: str = "models/daily_predictor.pkl",
    ):
        super().__init__(cfg)
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.signal_quantile = float(os.environ.get("PREDICTOR_SIGNAL_QUANTILE", signal_quantile))
        self.threshold_window = int(os.environ.get("PREDICTOR_THRESHOLD_WINDOW", threshold_window))

        if use_pretrained:
            data = _load_pickle(model_path, "DailyPredictorStrategy")
            self.model = data["model"]
            self.scaler = data["scaler"]
            logger.info("DailyPredictorStrategy: loaded model from %s", model_path)

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        daily_feats = make_daily_features(df)
        X = _preprocess(daily_feats[FEATURE_COLS].values.astype(np.float32))

        if self.model is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            pred_ret = self.model.predict(X_scaled)
        else:
            # In-session training fallback (no pre-trained model)
            from sklearn.linear_model import Ridge
            y_raw = daily_feats["fwd_ret_1d"].values
            mask = ~np.isnan(y_raw)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = Ridge(alpha=10.0)
            model.fit(X_scaled[mask], y_raw[mask])
            pred_ret = model.predict(X_scaled)

        signals = compute_predictor_signal(
            pred_ret, self.signal_quantile, self.threshold_window
        )
        return self._apply_holding_period(pd.Series(signals, index=daily_feats.index))


# ---------------------------------------------------------------------------
# Intraday strategies (legacy — not used in daily pipeline)
# ---------------------------------------------------------------------------

class OrdinalLogisticStrategy(BaseStrategy):
    """Logistic regression on 5-minute intraday features. Legacy; not in daily pipeline."""

    _INTRADAY_COLS = [
        "ret_1m", "ma_spread", "vol_10", "rsi_14", "vol_z",
        "momentum_5", "momentum_20", "vp_ratio", "vol_regime", "price_position",
    ]

    def __init__(self, cfg: StrategyConfig, use_pretrained: bool = True,
                 confidence_threshold: float = 0.65):
        super().__init__(cfg)
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn not installed.")
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.confidence_threshold = confidence_threshold
        self._feature_cols: list[str] = []

        if use_pretrained and os.path.exists("models/ordinal_logistic.pkl"):
            try:
                with open("models/ordinal_logistic.pkl", "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self._feature_cols = data.get("feature_cols", self._INTRADAY_COLS)
                self.confidence_threshold = data.get("confidence_threshold", confidence_threshold)
                logger.info("OrdinalLogisticStrategy: loaded intraday model")
            except Exception as exc:
                logger.warning("OrdinalLogisticStrategy: failed to load model: %s", exc)

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler is None:
            return pd.Series(0, index=feats.index)
        cols = [c for c in self._feature_cols if c in feats.columns]
        if not cols:
            return pd.Series(0, index=feats.index)
        X = _preprocess(feats[cols].values.astype(np.float32))
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        preds = np.argmax(probs, axis=1)
        preds = _apply_confidence_filter(preds, probs, self.confidence_threshold)
        signals = preds - 1
        return self._apply_holding_period(pd.Series(signals, index=feats.index))


class XGBoostStrategy(BaseStrategy):
    """XGBoost on 5-minute intraday features. Legacy; not in daily pipeline."""

    _INTRADAY_COLS = [
        "ret_1m", "ma_spread", "vol_10", "rsi_14", "vol_z",
        "momentum_5", "momentum_20", "vp_ratio", "vol_regime", "price_position",
    ]

    def __init__(self, cfg: StrategyConfig, use_pretrained: bool = True,
                 confidence_threshold: float = 0.60):
        super().__init__(cfg)
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed.")
        self.model = None
        self.confidence_threshold = confidence_threshold
        self._feature_cols: list[str] = []

        if use_pretrained and os.path.exists("models/xgboost.pkl"):
            try:
                with open("models/xgboost.pkl", "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self._feature_cols = data.get("feature_cols", self._INTRADAY_COLS)
                self.confidence_threshold = data.get("confidence_threshold", confidence_threshold)
                logger.info("XGBoostStrategy: loaded intraday model")
            except Exception as exc:
                logger.warning("XGBoostStrategy: failed to load model: %s", exc)

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            return pd.Series(0, index=feats.index)
        cols = [c for c in self._feature_cols if c in feats.columns]
        if not cols:
            return pd.Series(0, index=feats.index)
        X = _preprocess(feats[cols].values.astype(np.float32))
        probs = self.model.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        preds = _apply_confidence_filter(preds, probs, self.confidence_threshold)
        signals = preds - 1
        return self._apply_holding_period(pd.Series(signals, index=feats.index))
