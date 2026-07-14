"""
ml_strategies.py — Machine learning trading strategies.

DailyLogisticStrategy and DailyXGBoostStrategy are thin wrappers that
delegate signal generation to PredictorStrategy, using the appropriate
predictor and ThresholdDecision from the new modular layers.

DailyPredictorStrategy (regression-based rolling-quantile strategy) is a
standalone strategy wrapper.

Shared preprocessing (_preprocess) and model loading (_load_validated_pickle)
live in predictors/base.py; _load_pickle is a backward-compatible alias.
"""
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from base_strategy import BaseStrategy, StrategyConfig
from daily_features import FEATURE_COLS, FWD_RET_HORIZON_DAYS, make_daily_features
from decision_layers.threshold import ThresholdDecision
from predictor_strategy import PredictorStrategy
from predictors.base import _preprocess, _load_validated_pickle, _scale
from predictors.logistic import LogisticPredictor
from predictors.xgboost_pred import XGBPredictor

logger = logging.getLogger(__name__)

# Backward-compatible alias used by DailyPredictorStrategy
_load_pickle = _load_validated_pickle

try:
    from sklearn.preprocessing import RobustScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb  # noqa: F401
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


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

    Used by predict_next_day_lite.py's live-prediction path.

    NOTE: DailyPredictorStrategy (backtest) now uses
    compute_predictor_signal_raw_sign instead — the two decision layers
    currently diverge. See compute_predictor_signal_raw_sign's docstring.

    Returns an int array of {-1, 0, 1} (SELL/HOLD/BUY) the same length as
    pred_ret.
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


def compute_predictor_signal_raw_sign(
    pred_ret: np.ndarray, smooth_span: int = 1
) -> np.ndarray:
    """
    Maximum-breadth decision layer: trade on sign(prediction) directly,
    optionally EMA-smoothed first to reduce noise. No gating — every
    non-zero (post-smoothing) forecast becomes a trade.

    Used by DailyPredictorStrategy.signal() (backtest) as of the Sharpe
    0.27->0.83 rework.

    KNOWN GAP: predict_next_day_lite.py's live path still calls
    compute_predictor_signal (rolling-quantile gating), not this function —
    live daily_predictor signals no longer match what was backtested here.
    Flagged for a maintainer decision (align live to raw-sign, or revert
    the strategy to quantile gating) rather than silently changed, since
    it changes real trading signals sent to Discord.

    Returns an int array of {-1, 0, 1} (SELL/HOLD/BUY) the same length as
    pred_ret.
    """
    pred_series = pd.Series(pred_ret)
    if smooth_span > 1:
        pred_series = pred_series.ewm(span=smooth_span, adjust=False).mean()
    return np.sign(pred_series.values).astype(int)


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
      {'model': Ridge, 'scaler': RobustScaler, 'feature_contract': FEATURE_COLS, ...}

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
        self.scaler: Optional[RobustScaler] = None
        # Params set here before pickle load; updated below if pickle has tuned values.
        self.signal_quantile = signal_quantile
        self.threshold_window = threshold_window

        if use_pretrained:
            data = _load_pickle(model_path, "DailyPredictorStrategy")
            self.model = data["model"]
            self.scaler = data["scaler"]
            logger.info("DailyPredictorStrategy: loaded model from %s", model_path)

            # Three-level priority: env var -> pickle best_* keys -> constructor default
            sq_env = os.environ.get("PREDICTOR_SIGNAL_QUANTILE")
            if sq_env is not None:
                try:
                    self.signal_quantile = float(sq_env)
                except ValueError:
                    # Invalid env var — fall through to pickle key
                    if "best_signal_quantile" in data:
                        self.signal_quantile = float(data["best_signal_quantile"])
                    # else: stays at constructor default
            else:
                if "best_signal_quantile" in data:
                    self.signal_quantile = float(data["best_signal_quantile"])
                # else: stays at constructor default

            tw_env = os.environ.get("PREDICTOR_THRESHOLD_WINDOW")
            if tw_env is not None:
                try:
                    self.threshold_window = int(tw_env)
                except ValueError:
                    # Invalid env var — fall through to pickle key
                    if "best_threshold_window" in data:
                        self.threshold_window = int(data["best_threshold_window"])
                    # else: stays at constructor default
            else:
                if "best_threshold_window" in data:
                    self.threshold_window = int(data["best_threshold_window"])
                # else: stays at constructor default
        else:
            sq_env = os.environ.get("PREDICTOR_SIGNAL_QUANTILE")
            if sq_env is not None:
                try:
                    self.signal_quantile = float(sq_env)
                except ValueError:
                    pass
            tw_env = os.environ.get("PREDICTOR_THRESHOLD_WINDOW")
            if tw_env is not None:
                try:
                    self.threshold_window = int(tw_env)
                except ValueError:
                    pass

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        daily_feats = make_daily_features(df)

        if self.model is not None and self.scaler is not None:
            pred_ret = self.model.predict(
                _scale(self.scaler, daily_feats[FEATURE_COLS].values.astype(np.float32))
            )
        else:
            X = _preprocess(daily_feats[FEATURE_COLS].values.astype(np.float32))
            logger.warning(
                "DailyPredictorStrategy: no pretrained model — using in-session "
                "training fallback (single train/test split with embargo gap, "
                "NOT walk-forward). This is a toy path for smoke-testing only "
                "and should not be used to judge live strategy performance."
            )
            from sklearn.linear_model import Ridge

            y_raw = daily_feats["fwd_ret_vol_adj"].values
            n = len(daily_feats)
            split = int(n * 0.8)
            test_start = split + FWD_RET_HORIZON_DAYS

            pred_ret = np.zeros(n, dtype=np.float64)
            if test_start < n:
                train_mask = ~np.isnan(y_raw[:split])
                X_train = X[:split][train_mask]
                y_train = y_raw[:split][train_mask]

                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                model = Ridge(alpha=10.0)
                model.fit(X_train_scaled, y_train)

                X_test_scaled = _scale(scaler, X[test_start:])
                pred_ret[test_start:] = model.predict(X_test_scaled)
            else:
                logger.warning(
                    "DailyPredictorStrategy: not enough rows (%d) for an "
                    "embargoed train/test split — fallback will emit all-HOLD.",
                    n,
                )

        smooth_span = int(os.environ.get("PREDICTOR_SMOOTH_SPAN", "1"))
        signals = compute_predictor_signal_raw_sign(pred_ret, smooth_span)
        raw = self._apply_holding_period(pd.Series(signals, index=daily_feats.index))
        # Execution lag: decide on close[t], trade at close[t+1]. Matches
        # PredictorStrategy.signal (predictor_strategy.py:45). Backtest-only —
        # the live path (_predict_regressor_signal) takes signals[-1] and trades
        # the next session, so it is already correct and must NOT be shifted.
        return raw.shift(1).fillna(0).astype(int)

