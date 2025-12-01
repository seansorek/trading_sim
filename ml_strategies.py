"""
ml_strategies.py - Machine learning-based trading strategies using scikit-learn and XGBoost.

Provides ordinal logistic regression and gradient boosting approaches to predict
3-class signals (-1, 0, +1) based on technical features.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

# Try to import optional ML libraries
try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class MLStrategyBase:
    """Base class for ML strategies with holding period."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.feature_cols = ['ret_1m', 'ma_spread', 'vol_10', 'rsi_14', 'vol_z']
    
    def _discretize_labels(self, returns: pd.Series) -> np.ndarray:
        """Discretize 5-bar forward returns into 3 classes."""
        labels = np.zeros(len(returns), dtype=int)
        labels[returns > 0.005] = 1      # BUY threshold at +0.5%
        labels[returns < -0.005] = -1    # SELL threshold at -0.5%
        return labels
    
    def _apply_holding_period(self, sig: pd.Series) -> pd.Series:
        """Prevent position changes for holding_period bars after each trade."""
        result = sig.copy()
        last_trade_idx = -self.cfg.holding_period
        
        for i in range(len(result)):
            if i - last_trade_idx < self.cfg.holding_period:
                result.iloc[i] = 0
            elif result.iloc[i] != 0:
                last_trade_idx = i
        
        return result


class OrdinalLogisticStrategy(MLStrategyBase):
    """
    Ordinal logistic regression strategy using scikit-learn.
    
    Trains on features [ret_1m, ma_spread, vol_10, rsi_14, vol_z] to predict
    3-class labels (-1: SELL, 0: HOLD, +1: BUY) based on discretized 5-bar future returns.
    """
    
    def __init__(self, cfg, train_size: int = 500):
        """Initialize OrdinalLogisticStrategy."""
        super().__init__(cfg)
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn not installed. Install with: pip install scikit-learn")
        
        self.train_size = train_size
        self.model: Optional[LogisticRegression] = None
        self.cfg = cfg
    
    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using ordinal logistic regression."""
        n = len(feats)
        signals = np.zeros(n, dtype=int)
        
        # Rolling window training
        for i in range(self.train_size, n):
            # Training data: previous train_size bars
            train_start = i - self.train_size
            train_end = i
            
            X_train = feats[self.feature_cols].iloc[train_start:train_end].values
            future_ret_train = df['close'].iloc[train_start:train_end].pct_change(5).shift(-5).fillna(0).values
            y_train = self._discretize_labels(future_ret_train)
            
            # Skip if insufficient training data
            if len(y_train) < 10 or len(np.unique(y_train)) < 2:
                continue
            
            # Train ordinal logistic regression
            self.model = LogisticRegression(multi_class='multinomial', max_iter=500, random_state=42)
            try:
                self.model.fit(X_train, y_train)
            except Exception:
                continue
            
            # Predict for current bar
            X_test = feats[self.feature_cols].iloc[i:i+1].values
            
            # Skip if any NaN in current features
            if np.isnan(X_test).any():
                continue
            
            pred = self.model.predict(X_test)[0]
            signals[i] = pred
        
        sig_series = pd.Series(signals, index=feats.index, dtype=int)
        sig_series = self._apply_holding_period(sig_series)
        
        return sig_series


class XGBoostStrategy(MLStrategyBase):
    """
    XGBoost-based strategy using gradient boosting for 3-class classification.
    
    Trains on features [ret_1m, ma_spread, vol_10, rsi_14, vol_z] to predict
    3-class labels (-1: SELL, 0: HOLD, +1: BUY) based on discretized 5-bar future returns.
    """
    
    def __init__(self, cfg, train_size: int = 500, n_rounds: int = 50):
        """Initialize XGBoostStrategy."""
        super().__init__(cfg)
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")
        
        self.train_size = train_size
        self.n_rounds = n_rounds
        self.model: Optional[xgb.XGBClassifier] = None
        self.cfg = cfg
    
    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using XGBoost classifier."""
        n = len(feats)
        signals = np.zeros(n, dtype=int)
        
        # Rolling window training
        for i in range(self.train_size, n):
            # Training data: previous train_size bars
            train_start = i - self.train_size
            train_end = i
            
            X_train = feats[self.feature_cols].iloc[train_start:train_end].values
            future_ret_train = df['close'].iloc[train_start:train_end].pct_change(5).shift(-5).fillna(0).values
            y_train = self._discretize_labels(future_ret_train)
            
            # Skip if insufficient training data or no class diversity
            if len(y_train) < 10 or len(np.unique(y_train)) < 2:
                continue
            
            # Train XGBoost classifier
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_rounds,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            try:
                self.model.fit(X_train, y_train)
            except Exception:
                continue
            
            # Predict for current bar
            X_test = feats[self.feature_cols].iloc[i:i+1].values
            
            # Skip if any NaN in current features
            if np.isnan(X_test).any():
                continue
            
            pred = self.model.predict(X_test)[0]
            signals[i] = pred
        
        sig_series = pd.Series(signals, index=feats.index, dtype=int)
        sig_series = self._apply_holding_period(sig_series)
        
        return sig_series
