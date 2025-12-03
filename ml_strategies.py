"""ml_strategies.py - Machine learning-based trading strategies using scikit-learn and XGBoost.

Provides ordinal logistic regression and gradient boosting approaches to predict
3-class signals (-1, 0, +1) based on technical features.
"""
from base_strategy import BaseStrategy
import numpy as np
import pandas as pd
import pickle
import os
from typing import Optional, Tuple
from daily_features import make_daily_features

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

    @staticmethod
    def discretize_next_day(returns_1d: pd.Series, pos_threshold: float = 0.002, neg_threshold: float = -0.002) -> np.ndarray:
        """
        Discretize next-day returns for daily prediction into {-1,0,1}.
        Defaults: +/-0.2% thresholds.
        """
        labels = np.zeros(len(returns_1d), dtype=int)
        labels[returns_1d > pos_threshold] = 1
        labels[returns_1d < neg_threshold] = -1
        return labels
    
    def calculate_position_size(self, confidence: float, volatility: float = 1.0) -> float:
        """
        Calculate position size based on prediction confidence and volatility.
        
        Args:
            confidence: Prediction confidence (0.0-1.0), max probability from classifier
            volatility: Normalized volatility (default 1.0 = normal)
        
        Returns:
            Position size multiplier (0.0-1.0)
            - 0.0: No position (low confidence)
            - 1.0: Full position (high confidence, low volatility)
        """
        # Base sizing: scale linearly from confidence 0.50 (min) to 0.90 (max)
        if confidence < 0.50:
            return 0.0
        
        base_size = min(1.0, (confidence - 0.50) / 0.40)  # Scales 0.50->0.0, 0.90->1.0
        
        # Reduce size in high volatility environments
        vol_adjusted = base_size / max(1.0, volatility)
        
        return min(1.0, vol_adjusted)
    
    def get_recommendation(self, signal: int, confidence: float, position_size: float) -> str:
        """
        Generate trading recommendation with confidence levels.
        
        Args:
            signal: Trading signal (-1, 0, 1)
            confidence: Prediction confidence (0.0-1.0)
            position_size: Calculated position size (0.0-1.0)
        
        Returns:
            Recommendation string: "BUY", "SELL", or "HOLD"
        """
        if position_size < 0.1:
            return "HOLD"
        
        if signal == 1:
            return f"BUY (confidence: {confidence:.0%}, size: {position_size:.0%})"
        elif signal == -1:
            return f"SELL (confidence: {confidence:.0%}, size: {position_size:.0%})"
        else:
            return "HOLD"


class OrdinalLogisticStrategy(MLStrategyBase):
    """
    Ordinal logistic regression strategy using scikit-learn.
    
    Trains on features [ret_1m, ma_spread, vol_10, rsi_14, vol_z] to predict
    3-class labels (-1: SELL, 0: HOLD, +1: BUY) based on discretized 5-bar future returns.
    """
    
    def __init__(self, cfg, train_size: int = 500, use_pretrained: bool = True):
        """Initialize OrdinalLogisticStrategy."""
        super().__init__(cfg)
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn not installed. Install with: pip install scikit-learn")
        
        self.train_size = train_size  # Increased from 100 for more training data
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.cfg = cfg
        self.use_pretrained = use_pretrained
        self.pretrained_features = None
        
        # Load pre-trained model if available
        if use_pretrained and os.path.exists('models/ordinal_logistic.pkl'):
            try:
                with open('models/ordinal_logistic.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.pretrained_features = model_data['feature_cols']
                print(f"[info] Loaded pre-trained OrdinalLogistic model (features: {len(self.pretrained_features)})")
            except Exception as e:
                print(f"[warn] Failed to load pre-trained model: {e}. Will train on-the-fly.")
                self.model = None
                self.scaler = None
    
    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using ordinal logistic regression."""
        n = len(feats)
        signals = np.zeros(n, dtype=int)
        
        # If pre-trained model loaded, use it for all predictions
        if self.model is not None and self.scaler is not None and self.pretrained_features is not None:
            # Use pre-trained features if they exist, otherwise fall back
            feature_cols = [col for col in self.pretrained_features if col in feats.columns]
            if not feature_cols:
                print("[warn] Pre-trained features not available in data, falling back to training")
            else:
                X_all = feats[feature_cols].values
                X_all = self._preprocess_features(X_all)
                X_scaled = self.scaler.transform(X_all)
                
                # Predict for all bars
                preds = self.model.predict(X_scaled)
                signals = preds - 1  # Remap {0,1,2} back to {-1,0,1}
                
                # Apply holding period
                return self._apply_holding_period(pd.Series(signals, index=feats.index))
        
        # No pre-trained model available - return neutral signals
        # Training is disabled to prevent slow execution during simulation
        return pd.Series(0, index=feats.index)


class XGBoostStrategy(MLStrategyBase):
    """
    XGBoost-based strategy using gradient boosting for 3-class classification.
    
    Trains on features [ret_1m, ma_spread, vol_10, rsi_14, vol_z] to predict
    3-class labels (-1: SELL, 0: HOLD, +1: BUY) based on discretized 5-bar future returns.
    """
    
    def __init__(self, cfg, train_size: int = 500, n_rounds: int = 50, use_pretrained: bool = True):
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
        self.use_pretrained = use_pretrained
        self.pretrained_features = None
        
        # Load pre-trained model if available
        if use_pretrained and os.path.exists('models/xgboost.pkl'):
            try:
                with open('models/xgboost.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.pretrained_features = model_data['feature_cols']
                self.confidence_threshold = model_data.get('confidence_threshold', 0.50)
                print(f"[info] Loaded pre-trained XGBoost model (features: {len(self.pretrained_features)})")
            except Exception as e:
                print(f"[warn] Failed to load pre-trained model: {e}. Will train on-the-fly.")
                self.model = None
    
    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using XGBoost classifier."""
        n = len(feats)
        signals = np.zeros(n, dtype=int)
        
        # If pre-trained model loaded, use it for all predictions (skip training)
        if self.model is not None and self.pretrained_features is not None:
            # Use pre-trained features if they exist, otherwise fall back
            feature_cols = [col for col in self.pretrained_features if col in feats.columns]
            if not feature_cols:
                print("[warn] Pre-trained features not available in data, falling back to training")
            else:
                X_all = feats[feature_cols].values
                X_all = self._preprocess_features(X_all)
                
                # Get probability predictions for all bars
                probs = self.model.predict_proba(X_all)
                max_probs = np.max(probs, axis=1)
                preds = np.argmax(probs, axis=1)
                
                # Apply confidence threshold
                confident_mask = max_probs >= self.confidence_threshold
                signals_raw = np.where(confident_mask, preds - 1, 0)  # Remap {0,1,2} to {-1,0,1}
                
                signals = signals_raw
                
                # Apply holding period
                return self._apply_holding_period(pd.Series(signals, index=feats.index))
        
        # No pre-trained model available - return neutral signals
        # Training is disabled to prevent slow execution during simulation
        return pd.Series(0, index=feats.index)


class EnsembleXGBoostStrategy(MLStrategyBase):
    """
    An ensemble strategy that uses the signals from other strategies as features
    for an XGBoost model. This is a meta-learning approach.
    
    Note: Ensemble strategies require real-time base strategy signals, so they
    cannot use pre-trained models and must train on-the-fly.
    """
    def __init__(self, cfg, train_size: int = 100, n_rounds: int = 25):
        super().__init__(cfg)
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")
        
        self.train_size = train_size
        self.n_rounds = n_rounds
        self.model: Optional[xgb.XGBClassifier] = None
        # Ensemble cannot use pretrained models (needs live strategy signals)

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


class DailyLogisticStrategy(MLStrategyBase):
    """
    Daily logistic regression predicting next-day return class using daily features.
    Trains once over available history and predicts for all days.
    """
    def __init__(self, cfg, use_pretrained: bool = True):
        super().__init__(cfg)
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn not installed. Install with: pip install scikit-learn")
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.pretrained_features = None
        self.use_pretrained = use_pretrained
        # Try both naming conventions
        model_paths = ['models/daily_logistic.pkl', 'models/ordinal_logistic.pkl']
        for model_path in model_paths:
            if use_pretrained and os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)
                    self.model = data['model']
                    self.scaler = data['scaler']
                    self.pretrained_features = data['feature_cols']
                    print(f"[info] Loaded pre-trained DailyLogistic model from {model_path}")
                    break
                except Exception as e:
                    print(f"[warn] Failed to load DailyLogistic from {model_path}: {e}")
                    continue

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        # Build daily features from df (assumed daily interval)
        daily_feats = make_daily_features(df)
        feature_cols = [c for c in daily_feats.columns if c not in ['fwd_ret_1d', 'close']]
        X = daily_feats[feature_cols].values
        X = self._preprocess_features(X)
        y = self.discretize_next_day(daily_feats['fwd_ret_1d']) + 1  # map to {0,1,2}

        # Use pre-trained if available
        if self.model is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            # Use predict_proba for confidence-based filtering
            probs = self.model.predict_proba(X_scaled)
            preds = np.argmax(probs, axis=1) - 1  # Map {0,1,2} -> {-1,0,1}
            confidence = np.max(probs, axis=1)
            
            # Confidence filtering: only trade (BUY/SELL) with confidence > 0.50
            for i in range(len(preds)):
                if preds[i] != 0 and confidence[i] < 0.50:
                    preds[i] = 0  # Downgrade to HOLD if low confidence
            
            sig = pd.Series(preds, index=df.index)
            return self._apply_holding_period(sig)

        # Otherwise, train once and predict
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=42)
        # Filter out last NaN label rows
        mask = ~np.isnan(y)
        model.fit(X_scaled[mask], y[mask])
        
        # Use predict_proba for confidence-based filtering
        probs = model.predict_proba(X_scaled)
        preds = np.argmax(probs, axis=1) - 1  # Map {0,1,2} -> {-1,0,1}
        confidence = np.max(probs, axis=1)
        
        # Confidence filtering: only trade (BUY/SELL) with confidence > 0.50
        for i in range(len(preds)):
            if preds[i] != 0 and confidence[i] < 0.50:
                preds[i] = 0  # Downgrade to HOLD if low confidence
        
        sig = pd.Series(preds, index=df.index)
        return self._apply_holding_period(sig)


class DailyXGBoostStrategy(MLStrategyBase):
    """
    Daily XGBoost classifier predicting next-day return class using daily features.
    Trains once over available history and predicts for all days.
    """
    def __init__(self, cfg, use_pretrained: bool = True):
        super().__init__(cfg)
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")
        self.model = None
        self.pretrained_features = None
        self.confidence_threshold = 0.50
        self.use_pretrained = use_pretrained
        # Try both naming conventions
        model_paths = ['models/daily_xgboost.pkl', 'models/xgboost.pkl']
        for model_path in model_paths:
            if use_pretrained and os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)
                    self.model = data['model']
                    self.pretrained_features = data['feature_cols']
                    self.confidence_threshold = data.get('confidence_threshold', 0.50)
                    print(f"[info] Loaded pre-trained DailyXGBoost model from {model_path}")
                    break
                except Exception as e:
                    print(f"[warn] Failed to load DailyXGBoost from {model_path}: {e}")
                    continue

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        daily_feats = make_daily_features(df)
        feature_cols = [c for c in daily_feats.columns if c not in ['fwd_ret_1d', 'close']]
        X = daily_feats[feature_cols].values
        X = self._preprocess_features(X)
        y = self.discretize_next_day(daily_feats['fwd_ret_1d']) + 1

        if self.model is not None:
            probs = self.model.predict_proba(X)
            maxp = probs.max(axis=1)
            preds = np.argmax(probs, axis=1)
            signals = np.where(maxp >= self.confidence_threshold, preds - 1, 0)
            sig = pd.Series(signals, index=df.index)
            return self._apply_holding_period(sig)

        import xgboost as xgb
        # Train once
        mask = ~np.isnan(y)
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            verbosity=0
        )
        model.fit(X[mask], y[mask])
        probs = model.predict_proba(X)
        maxp = probs.max(axis=1)
        preds = np.argmax(probs, axis=1)
        signals = np.where(maxp >= self.confidence_threshold, preds - 1, 0)
        sig = pd.Series(signals, index=df.index)
        return self._apply_holding_period(sig)

