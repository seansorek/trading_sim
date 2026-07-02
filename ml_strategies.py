"""
ml_strategies.py — Machine learning trading strategies.

DailyLogisticStrategy and DailyXGBoostStrategy are thin wrappers that
delegate signal generation to PredictorStrategy, using the appropriate
predictor and ThresholdDecision from the new modular layers.

DailyPredictorStrategy (regression-based rolling-quantile strategy) and the
legacy intraday strategies are preserved as-is from main.

Shared preprocessing (_preprocess) and model loading (_load_validated_pickle)
live in predictors/base.py; _load_pickle is a backward-compatible alias.
"""
import logging
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from base_strategy import BaseStrategy, StrategyConfig
from daily_features import FEATURE_COLS, make_daily_features
from decision_layers.threshold import ThresholdDecision
from predictor_strategy import PredictorStrategy
from predictors.base import _preprocess, _load_validated_pickle
from predictors.logistic import LogisticPredictor
from predictors.xgboost_pred import XGBPredictor

logger = logging.getLogger(__name__)

# Backward-compatible alias used by DailyPredictorStrategy
_load_pickle = _load_validated_pickle

try:
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb  # noqa: F401
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def _apply_confidence_filter(
    preds: np.ndarray, probs: np.ndarray, threshold: float
) -> np.ndarray:
    """Zero out predictions whose max probability is below threshold."""
    confidence = probs.max(axis=1)
    result = preds.copy()
    result[confidence < threshold] = 0
    return result


# ---------------------------------------------------------------------------
# Daily strategies — modular architecture (PredictorStrategy-backed)
# ---------------------------------------------------------------------------

class DailyLogisticStrategy(BaseStrategy):
    """Logistic regression next-day strategy backed by PredictorStrategy.

    Loads the trained LogisticPredictor from a pickle and gates signals with
    ThresholdDecision. All shared orchestration (feature extraction, lag,
    holding period) is handled by PredictorStrategy.
    """

    def __init__(
        self,
        cfg: StrategyConfig,
        model_path: str = "models/daily_logistic.pkl",
        spy_df: Optional[pd.DataFrame] = None,
        **_kwargs,
    ):
        super().__init__(cfg)
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is not installed.")
        predictor = LogisticPredictor.load(model_path)
        decision = ThresholdDecision(predictor.confidence_threshold)
        self._inner = PredictorStrategy(cfg, predictor, decision, spy_df=spy_df)

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        return self._inner.signal(feats, df)


class DailyXGBoostStrategy(BaseStrategy):
    """XGBoost next-day strategy backed by PredictorStrategy."""

    def __init__(
        self,
        cfg: StrategyConfig,
        model_path: str = "models/daily_xgboost.pkl",
        spy_df: Optional[pd.DataFrame] = None,
        **_kwargs,
    ):
        super().__init__(cfg)
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed.")
        predictor = XGBPredictor.load(model_path)
        decision = ThresholdDecision(predictor.confidence_threshold)
        self._inner = PredictorStrategy(cfg, predictor, decision, spy_df=spy_df)

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        return self._inner.signal(feats, df)


# ---------------------------------------------------------------------------
# Regression-based daily strategy
# ---------------------------------------------------------------------------

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
