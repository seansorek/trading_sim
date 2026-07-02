"""PredictorStrategy: wires BasePredictor + BaseDecisionLayer into BaseStrategy."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from base_strategy import BaseStrategy, StrategyConfig
from daily_features import FEATURE_COLS, make_daily_features
from decision_layers.base import BaseDecisionLayer, DecisionContext
from predictors.base import BasePredictor


class PredictorStrategy(BaseStrategy):
    """Compose a predictor and a decision layer into a backtest-compatible strategy.

    Handles shared orchestration:
      1. Feature extraction via make_daily_features
      2. Pass raw features to predictor.predict()
      3. Pass scores + proba to decision.decide()
      4. Apply holding-period filter
      5. Apply 1-bar execution lag (shift(1))
    """

    def __init__(
        self,
        cfg: StrategyConfig,
        predictor: BasePredictor,
        decision: BaseDecisionLayer,
        spy_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__(cfg)
        self.predictor = predictor
        self.decision = decision
        self.spy_df = spy_df

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        daily_feats = make_daily_features(df, spy_df=self.spy_df)
        X = daily_feats[FEATURE_COLS].values.astype(np.float32)
        scores, proba = self.predictor.predict(X)
        ctx = DecisionContext(index=daily_feats.index, symbol=self.cfg.name)
        signals = self.decision.decide(scores, proba, ctx)
        raw = self._apply_holding_period(pd.Series(signals, index=daily_feats.index))
        return raw.shift(1).fillna(0).astype(int)
