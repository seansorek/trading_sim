"""
ml_strategies.py - Machine learning-based trading strategies using scikit-learn and XGBoost.

Provides ordinal logistic regression and gradient boosting approaches to predict
3-class signals (-1, 0, +1) based on technical features.
"""
from base_strategy import BaseStrategy
import numpy as np
import pandas as pd
from typing import Optional, Tuple

# Try to import optional ML libraries
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

class MLStrategyBase(BaseStrategy):
    """Base class for ML strategies, inheriting holding period logic."""
    # Updated to include new enhanced features
    feature_cols = ['ret_1m', 'ma_spread', 'vol_10', 'rsi_14', 'vol_z', 
                    'momentum_5', 'momentum_20', 'vp_ratio', 'vol_regime', 'price_position']
    
    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """Clean and normalize features for better ML convergence."""
        # Replace inf/-inf with NaN, then fill NaN with 0
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0)
        
        # Clip extreme outliers (beyond 5 std devs)
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            if len(col_data) > 0 and np.std(col_data) > 0:
                mean = np.mean(col_data)
                std = np.std(col_data)
                X[:, col_idx] = np.clip(col_data, mean - 5*std, mean + 5*std)
        
        return X
    
    def _discretize_labels(self, returns: pd.Series) -> np.ndarray:
        """Discretize 5-bar forward returns into 3 classes."""
        labels = np.zeros(len(returns), dtype=int)
        labels[returns > 0.005] = 1      # BUY threshold at +0.5% (realistic for 5m bars)
        labels[returns < -0.005] = -1    # SELL threshold at -0.5% (realistic for 5m bars)
        return labels


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
        
        self.train_size = train_size  # Increased from 100 for more training data
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
            
            # Preprocess features
            X_train = self._preprocess_features(X_train)
            
            # Remap labels from {-1, 0, 1} to {0, 1, 2} for sklearn
            y_train_mapped = y_train + 1  # -1->0, 0->1, 1->2
            
            # Skip if insufficient training data or no class diversity
            if len(y_train) < 20 or len(np.unique(y_train_mapped)) < 2:
                continue
            
            # Normalize features for better convergence
            scaler = StandardScaler()
            try:
                X_train_scaled = scaler.fit_transform(X_train)
            except Exception:
                continue
            
            # Train ordinal logistic regression with class balancing
            self.model = LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                class_weight='balanced',  # Handle imbalanced classes
                random_state=42,
                C=1.0  # Regularization
            )
            try:
                self.model.fit(X_train_scaled, y_train_mapped)
            except Exception:
                continue
            
            # Predict for current bar
            X_test = feats[self.feature_cols].iloc[i:i+1].values
            X_test = self._preprocess_features(X_test)
            
            # Skip if any NaN in current features
            if np.isnan(X_test).any():
                continue
            
            try:
                X_test_scaled = scaler.transform(X_test)
                pred_mapped = self.model.predict(X_test_scaled)[0]
                # Remap back from {0, 1, 2} to {-1, 0, 1}
                signals[i] = pred_mapped - 1
            except Exception:
                continue
        
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
        
        self.train_size = train_size  # Increased from 150 to 500 for more training data
        self.n_rounds = n_rounds  # Increased from 25 for better learning
        self.model: Optional[xgb.XGBClassifier] = None
        self.cfg = cfg
        self.feature_importance = {}  # Track which features matter most
        self.confidence_threshold = 0.50  # Min probability to trade (tunable)
    
    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using XGBoost classifier."""
        n = len(feats)
        signals = np.zeros(n, dtype=int)
        
        # Check if all required features exist, otherwise use subset
        available_features = [col for col in self.feature_cols if col in feats.columns]
        if not available_features:
            # Fallback to basic features if new ones don't exist
            available_features = ['ret_1m', 'ma_spread', 'vol_10', 'rsi_14', 'vol_z']
        
        # Rolling window training
        for i in range(self.train_size, n):
            # Training data: previous train_size bars
            train_start = i - self.train_size
            train_end = i
            
            X_train = feats[available_features].iloc[train_start:train_end].values
            future_ret_train = df['close'].iloc[train_start:train_end].pct_change(5).shift(-5).fillna(0).values
            y_train = self._discretize_labels(future_ret_train)
            
            # Preprocess features
            X_train = self._preprocess_features(X_train)
            
            # Remap labels from {-1, 0, 1} to {0, 1, 2} for XGBoost
            y_train_mapped = y_train + 1  # -1->0, 0->1, 1->2
            
            # Skip if insufficient training data or no class diversity
            if len(y_train) < 25 or len(np.unique(y_train_mapped)) < 2:
                continue
            
            # Train XGBoost classifier with optimized hyperparameters
            # Calculate class weights for imbalanced data
            classes, counts = np.unique(y_train_mapped, return_counts=True)
            total = len(y_train_mapped)
            class_weights = {cls: total / (len(classes) * count) for cls, count in zip(classes, counts)}
            sample_weights = np.array([class_weights[y] for y in y_train_mapped])
            
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_rounds,
                max_depth=3,  # Shallower trees prevent overfitting on small windows
                learning_rate=0.05,  # Lower for gradual learning
                subsample=0.7,  # More aggressive subsampling
                colsample_bytree=0.7,
                colsample_bylevel=0.7,  # Column sampling at each level
                min_child_weight=3,  # Require more samples per leaf
                gamma=0.1,  # Min loss reduction for split
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=42,
                verbosity=0,
                tree_method='hist',
                enable_categorical=False,
                early_stopping_rounds=10  # Stop if no improvement
            )
            try:
                # Split training data for early stopping validation
                split_point = int(len(X_train) * 0.8)
                X_train_fit = X_train[:split_point]
                y_train_fit = y_train_mapped[:split_point]
                X_val = X_train[split_point:]
                y_val = y_train_mapped[split_point:]
                weights_fit = sample_weights[:split_point]
                weights_val = sample_weights[split_point:]
                
                if len(X_val) > 5 and len(np.unique(y_val)) > 1:
                    self.model.fit(
                        X_train_fit, y_train_fit,
                        sample_weight=weights_fit,
                        eval_set=[(X_val, y_val)],
                        sample_weight_eval_set=[weights_val],
                        verbose=False
                    )
                else:
                    # Fall back to no validation if insufficient data
                    self.model.fit(X_train, y_train_mapped, sample_weight=sample_weights)
                
                # Track feature importance (helps debug which features matter)
                if hasattr(self.model, 'feature_importances_'):
                    for idx, importance in enumerate(self.model.feature_importances_):
                        feat_name = available_features[idx] if idx < len(available_features) else f'feat_{idx}'
                        if feat_name not in self.feature_importance:
                            self.feature_importance[feat_name] = []
                        self.feature_importance[feat_name].append(importance)
                        
            except Exception:
                continue
            
            # Predict for current bar with probability thresholding
            X_test = feats[available_features].iloc[i:i+1].values
            X_test = self._preprocess_features(X_test)
            
            # Skip if any NaN in current features
            if np.isnan(X_test).any():
                continue
            
            try:
                # Use probability predictions for confidence filtering
                pred_proba = self.model.predict_proba(X_test)[0]
                max_proba = pred_proba.max()
                
                # Only trade if model is confident (above threshold)
                if max_proba >= self.confidence_threshold:
                    pred_mapped = np.argmax(pred_proba)
                    # Remap back from {0, 1, 2} to {-1, 0, 1}
                    signals[i] = pred_mapped - 1
                # else: stay at 0 (no trade) when confidence is low
                
            except Exception:
                continue
        
        sig_series = pd.Series(signals, index=feats.index, dtype=int)
        sig_series = self._apply_holding_period(sig_series)
        
        return sig_series


class EnsembleXGBoostStrategy(MLStrategyBase):
    """
    An ensemble strategy that uses the signals from other strategies as features
    for an XGBoost model. This is a meta-learning approach.
    """
    def __init__(self, cfg, train_size: int = 100, n_rounds: int = 25):
        super().__init__(cfg)
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")
        
        self.train_size = train_size
        self.n_rounds = n_rounds
        self.model: Optional[xgb.XGBClassifier] = None

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame, base_signals: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using an ensemble of other strategies' signals.
        
        Args:
            feats: Standard features (not used directly, but for compatibility).
            df: The price DataFrame for calculating future returns (the target).
            base_signals: A DataFrame where columns are strategy names and values are their signals.
        """
        n = len(df)
        signals = np.zeros(n, dtype=int)
        
        # Align base_signals with the main dataframe index
        base_signals = base_signals.reindex(df.index, fill_value=0)

        # Rolling window training
        for i in range(self.train_size, n):
            train_start = i - self.train_size
            train_end = i
            
            # Features are the signals from the base strategies
            X_train = base_signals.iloc[train_start:train_end].values
            
            # Target is the discretized future return
            future_ret_train = df['close'].iloc[train_start:train_end].pct_change(5).shift(-5).fillna(0).values
            y_train = self._discretize_labels(future_ret_train)
            
            # Remap labels from {-1, 0, 1} to {0, 1, 2} for XGBoost
            y_train_mapped = y_train + 1  # -1->0, 0->1, 1->2
            
            if len(y_train) < 10 or len(np.unique(y_train_mapped)) < 2:
                continue
            
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_rounds, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
            )
            try:
                self.model.fit(X_train, y_train_mapped)
            except Exception:
                continue
            
            # Predict for the current bar
            X_test = base_signals.iloc[i:i+1].values
            if np.isnan(X_test).any():
                continue
            
            pred_mapped = self.model.predict(X_test)[0]
            # Remap back from {0, 1, 2} to {-1, 0, 1}
            signals[i] = pred_mapped - 1
            
        sig_series = pd.Series(signals, index=df.index, dtype=int)
        sig_series = self._apply_holding_period(sig_series)
        return sig_series
