"""
Hybrid strategies combining multiple ML models for improved signal generation.
"""

import numpy as np
import pandas as pd
import os
from base_strategy import BaseStrategy, StrategyConfig
from daily_features import make_daily_features


class HybridDQNXGBoostStrategy(BaseStrategy):
    """
    Combines DQN and XGBoost signals via voting logic:
    - Strong agreement (both signal same direction): Full signal strength
    - Disagreement or low confidence from either: Hold (0)
    - Weighted by model confidence levels
    """
    def __init__(self, cfg: StrategyConfig):
        super().__init__(cfg)
        # Confidence thresholds determined via threshold sweep:
        # 0.01 / 0.33 -> 359 trades, -0.54% (too many false signals)
        # 0.05 / 0.35 -> ~100-150 trades, -0.02% (balanced)
        # 0.1  / 0.4  -> ~350 trades, -0.34% (converges to too many signals)
        # 0.5  / 0.65 -> 4 trades, -0.00% (too selective)
        # Conclusion: Model disagreement is high; voting logic may need refinement
        # For now, using conservative thresholds to reduce false signals
        self.dqn_confidence_threshold = float(os.environ.get('DQN_CONFIDENCE', '0.05'))
        self.xgb_confidence_threshold = float(os.environ.get('XGB_CONFIDENCE', '0.35'))
        
        # Load models lazily
        self.dqn_model = None
        self.xgb_model = None
        self.scaler = None
        
    def _load_dqn(self):
        """Lazy load DQN model."""
        if self.dqn_model is not None:
            return True
        try:
            from dqn_agent import DQNAgent
            import torch
            model_path = os.environ.get('DQN_MODEL', 'models/dqn_agent.pt')
            try:
                # Try loading with original DQNAgent
                self.dqn_model = DQNAgent.load(model_path)
            except Exception as e:
                # If that fails, try with enhanced agent
                try:
                    from dqn_agent_enhanced import DQNAgent as DQNAgentEnhanced
                    self.dqn_model = DQNAgentEnhanced.load(model_path)
                except Exception as e2:
                    print(f"[warn] Failed both standard and enhanced DQN load: {e2}")
                    return False
            self.torch = torch
            return True
        except Exception as e:
            print(f"[warn] Failed to load DQN: {e}")
            return False
    
    def _load_xgb(self):
        """Lazy load XGBoost model."""
        if self.xgb_model is not None:
            return True
        try:
            import pickle
            import xgboost as xgb
            # Try both naming conventions
            model_path = os.environ.get('XGB_MODEL', None)
            if not model_path:
                # Try default names in order
                for p in ['models/daily_xgboost.pkl', 'models/xgboost.pkl']:
                    if os.path.exists(p):
                        model_path = p
                        break
            if not model_path or not os.path.exists(model_path):
                print(f"[warn] XGBoost model not found (checked models/daily_xgboost.pkl, models/xgboost.pkl)")
                return False
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            self.xgb_model = data.get('model')
            self.scaler = data.get('scaler')
            return self.xgb_model is not None
        except Exception as e:
            print(f"[warn] Failed to load XGBoost: {e}")
            return False

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        if not self._load_dqn() or not self._load_xgb():
            print("[warn] Hybrid strategy: Missing models, returning Hold")
            return pd.Series(0, index=df.index)
        
        daily_feats = make_daily_features(df)
        daily_feats = daily_feats.fillna(0.0)
        
        window = int(os.environ.get('DQN_WINDOW', 20))
        feature_cols = [c for c in daily_feats.columns if c not in ["fwd_ret_1d"]]
        
        signals = []
        idxs = []
        
        for i in range(max(window, 10), len(daily_feats)):
            # DQN signal
            frame = daily_feats.iloc[i - window:i]
            state = frame[feature_cols].values.astype(np.float32).flatten()
            
            try:
                with self.torch.no_grad():
                    s_t = self.torch.from_numpy(state).float().unsqueeze(0)
                    q_vals = self.dqn_model.q(s_t).squeeze(0).cpu().numpy()
                
                dqn_max_q = q_vals.max()
                dqn_min_q = q_vals.min()
                dqn_confidence = dqn_max_q - dqn_min_q
                dqn_action = int(np.argmax(q_vals))
                dqn_sig = 1 if dqn_action == 1 else (-1 if dqn_action == 2 else 0)
            except Exception as e:
                # If DQN fails, skip this bar
                continue
            
            # XGBoost signal - be flexible with feature count
            try:
                X_test = daily_feats[feature_cols].iloc[i:i+1].values
                
                # XGBoost model might expect different number of features
                # Try to handle mismatch gracefully
                try:
                    probs = self.xgb_model.predict_proba(X_test)[0]
                except Exception:
                    # If features don't match, try subsetting to expected features
                    n_features = self.xgb_model.n_features_in_
                    if X_test.shape[1] > n_features:
                        X_test = X_test[:, :n_features]
                    elif X_test.shape[1] < n_features:
                        # Pad with zeros
                        pad = np.zeros((X_test.shape[0], n_features - X_test.shape[1]))
                        X_test = np.hstack([X_test, pad])
                    probs = self.xgb_model.predict_proba(X_test)[0]
                
                xgb_max_prob = probs.max()
                xgb_action = np.argmax(probs)
                xgb_sig = 1 if xgb_action == 1 else (-1 if xgb_action == 2 else 0)
            except Exception as e:
                # If XGBoost fails, skip this bar
                continue
            
            # Hybrid voting logic
            if dqn_confidence < self.dqn_confidence_threshold or xgb_max_prob < self.xgb_confidence_threshold:
                # At least one model lacks confidence
                sig = 0
            elif dqn_sig == xgb_sig and dqn_sig != 0:  # Both agree on Long or Short (not Hold)
                # Strong agreement: use weighted combination
                dqn_weight = min(dqn_confidence, 1.0)  # Normalize to [0, 1]
                xgb_weight = xgb_max_prob
                combined_weight = (dqn_weight + xgb_weight) / 2
                sig = dqn_sig if combined_weight > 0.35 else 0
            else:
                # Disagreement or both holding: be cautious
                sig = 0
            
            signals.append(sig)
            idxs.append(daily_feats.index[i])
        
        if not signals:
            return pd.Series(0, index=df.index)
        
        ser = pd.Series(signals, index=pd.DatetimeIndex(idxs))
        ser = ser.reindex(df.index).fillna(0).astype(int)
        return self._apply_holding_period(ser)


class EnsembleWeightedStrategy(BaseStrategy):
    """
    Weighted ensemble of logistic, xgboost, and dqn signals.
    Weights are tunable via environment variables.
    """
    def __init__(self, cfg: StrategyConfig):
        super().__init__(cfg)
        self.logistic_weight = float(os.environ.get('LOGISTIC_WEIGHT', '1.0'))
        self.xgb_weight = float(os.environ.get('XGB_WEIGHT', '2.0'))
        self.dqn_weight = float(os.environ.get('DQN_WEIGHT', '1.0'))
        
        # Normalize weights
        total = self.logistic_weight + self.xgb_weight + self.dqn_weight
        self.logistic_weight /= total
        self.xgb_weight /= total
        self.dqn_weight /= total
        
        self.models_loaded = False
        self.logistic_model = None
        self.xgb_model = None
        self.dqn_model = None
        self.scaler = None
    
    def _load_models(self):
        """Load all three models."""
        if self.models_loaded:
            return True
        try:
            import pickle
            import xgboost as xgb
            from dqn_agent import DQNAgent
            from sklearn.linear_model import LogisticRegression
            
            # Load logistic - try both names
            logistic_path = None
            for p in ['models/daily_logistic.pkl', 'models/ordinal_logistic.pkl']:
                if os.path.exists(p):
                    logistic_path = p
                    break
            if not logistic_path:
                print(f"[warn] Logistic model not found")
                return False
            
            with open(logistic_path, 'rb') as f:
                ldata = pickle.load(f)
            self.logistic_model = ldata.get('model')
            logistic_scaler = ldata.get('scaler')
            
            # Load XGBoost - try both names
            xgb_path = None
            for p in ['models/daily_xgboost.pkl', 'models/xgboost.pkl']:
                if os.path.exists(p):
                    xgb_path = p
                    break
            if not xgb_path:
                print(f"[warn] XGBoost model not found")
                return False
            
            with open(xgb_path, 'rb') as f:
                xdata = pickle.load(f)
            self.xgb_model = xdata.get('model')
            
            # Load DQN - handle both standard and enhanced formats
            try:
                from dqn_agent import DQNAgent
                self.dqn_model = DQNAgent.load('models/dqn_agent.pt')
            except Exception as e:
                try:
                    from dqn_agent_enhanced import DQNAgent as DQNAgentEnhanced
                    self.dqn_model = DQNAgentEnhanced.load('models/dqn_agent.pt')
                except Exception as e2:
                    print(f"[warn] Failed to load DQN model: {e2}")
                    return False
            
            self.scaler = logistic_scaler
            self.models_loaded = True
            return True
        except Exception as e:
            print(f"[warn] Failed to load ensemble models: {e}")
            return False
    
    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        if not self._load_models():
            return pd.Series(0, index=df.index)
        
        from daily_features import make_daily_features
        import torch
        
        daily_feats = make_daily_features(df)
        daily_feats = daily_feats.fillna(0.0)
        
        window = int(os.environ.get('DQN_WINDOW', 20))
        # Logistic and XGBoost expect 18 features (no 'close', no 'fwd_ret_1d')
        feature_cols_ml = [c for c in daily_feats.columns if c not in ["fwd_ret_1d", "close"]]
        # DQN was trained with 19 features (includes 'close', excludes 'fwd_ret_1d')
        feature_cols_dqn = [c for c in daily_feats.columns if c not in ["fwd_ret_1d"]]
        
        signals = []
        idxs = []
        
        for i in range(max(window, 10), len(daily_feats)):
            # Logistic signal - uses 18 features
            X_log = daily_feats[feature_cols_ml].iloc[i:i+1].values
            X_log_scaled = self.scaler.transform(X_log)
            log_pred = self.logistic_model.predict(X_log_scaled)[0] - 1
            
            # XGBoost signal - uses 18 features
            xgb_probs = self.xgb_model.predict_proba(X_log)[0]
            xgb_sig = np.argmax(xgb_probs) - 1
            
            # DQN signal - uses 19 features (with 'close')
            frame = daily_feats.iloc[i - window:i]
            state = frame[feature_cols_dqn].values.astype(np.float32).flatten()
            with torch.no_grad():
                s_t = torch.from_numpy(state).float().unsqueeze(0)
                q_vals = self.dqn_model.q(s_t).squeeze(0).cpu().numpy()
            dqn_sig = np.argmax(q_vals) - 1
            
            # Weighted ensemble
            weighted_sig = (self.logistic_weight * log_pred + 
                          self.xgb_weight * xgb_sig + 
                          self.dqn_weight * dqn_sig)
            
            # Threshold to {-1, 0, 1}
            if weighted_sig > 0.3:
                sig = 1
            elif weighted_sig < -0.3:
                sig = -1
            else:
                sig = 0
            
            signals.append(sig)
            idxs.append(daily_feats.index[i])
        
        if not signals:
            return pd.Series(0, index=df.index)
        
        ser = pd.Series(signals, index=pd.DatetimeIndex(idxs))
        ser = ser.reindex(df.index).fillna(0).astype(int)
        return self._apply_holding_period(ser)
