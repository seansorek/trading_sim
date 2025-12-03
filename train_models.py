#!/usr/bin/env python3
"""
train_models.py

Pre-train ML models on historical data and save them for deployment.
Run this locally before committing to avoid retraining on GitHub Actions.

Features:
- Local data caching to avoid re-downloading
- Bayesian hyperparameter optimization
- Comprehensive evaluation (Precision, Recall, F1, Sharpe, Max Drawdown)
- Multi-month training data support

Usage:
    python train_models.py --symbols AAPL,MSFT,GOOGL --days 180
    python train_models.py --symbols AAPL,MSFT,GOOGL --days 180 --optimize
"""

import os
import argparse
import pickle
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    print("[warn] scikit-optimize not installed. Bayesian optimization disabled.")
    print("       Install with: pip install scikit-optimize")

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from data_loader import load_yfinance
from simulation_pipeline import make_features
from daily_features import make_daily_features
from ml_strategies import OrdinalLogisticStrategy, XGBoostStrategy
from base_strategy import StrategyConfig
from skorch import NeuralNetClassifier
import torch
import torch.nn as nn

# Top-level RNN module to ensure picklability
class RNNModule(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)
        self.hidden_size = hidden_size
    def forward(self, X):
        out, _ = self.rnn(X)
        last = out[:, -1, :]
        return self.fc(last)

def load_or_fetch_data(symbol, start_date, end_date, interval, cache_dir="data/cache"):
    """
    Load data from cache if available, otherwise fetch and cache it.
    
    Args:
        symbol: Stock symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        interval: Data interval
        cache_dir: Directory to store cached data
    
    Returns:
        DataFrame with OHLCV data
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename
    cache_file = os.path.join(
        cache_dir,
        f"{symbol}_{start_date}_{end_date}_{interval}.pkl"
    )
    
    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                df = pickle.load(f)
            print(f"  [cache] Loaded {symbol} from cache")
            return df
        except Exception as e:
            print(f"  [warn] Cache load failed for {symbol}: {e}")
    
    # Fetch fresh data
    print(f"  [fetch] Downloading {symbol}...")
    df = load_yfinance(symbol, start=start_date, end=end_date, interval=interval)
    
    # Cache for next time
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
    except Exception as e:
        print(f"  [warn] Cache save failed for {symbol}: {e}")
    
    return df


def calculate_profit_metrics(y_true, y_pred, returns):
    """
    Calculate profit-based metrics: Sharpe ratio and max drawdown.
    
    Args:
        y_true: True labels {-1, 0, 1}
        y_pred: Predicted labels {-1, 0, 1}
        returns: Actual forward returns for each prediction
    
    Returns:
        dict with sharpe_ratio and max_drawdown
    """
    # Calculate strategy returns (position * forward return)
    strategy_returns = y_pred * returns
    
    # Remove zeros (no position)
    active_returns = strategy_returns[strategy_returns != 0]
    
    if len(active_returns) < 2:
        return {'sharpe_ratio': 0.0, 'max_drawdown_pct': 0.0}
    
    # Sharpe ratio (annualized for 5m bars: sqrt(252*78) = 140)
    sharpe = np.mean(active_returns) / (np.std(active_returns) + 1e-8) * np.sqrt(140)
    
    # Max drawdown
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (running_max + 1e-8)
    max_dd = np.min(drawdown) * 100  # Convert to percentage
    
    return {
        'sharpe_ratio': float(sharpe),
        'max_drawdown_pct': float(max_dd)
    }


def evaluate_model(model, X_test, y_test, returns_test, scaler=None):
    """
    Comprehensive model evaluation with classification and profit metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels {0, 1, 2}
        returns_test: Actual forward returns
        scaler: Optional StandardScaler for preprocessing
    
    Returns:
        dict with all evaluation metrics
    """
    # Preprocess if scaler provided
    if scaler is not None:
        X_test = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Classification metrics
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    accuracy = np.mean(y_test == y_pred)
    
    # Profit metrics (convert back to {-1, 0, 1})
    y_pred_signals = y_pred - 1
    profit_metrics = calculate_profit_metrics(y_test - 1, y_pred_signals, returns_test)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'sharpe_ratio': profit_metrics['sharpe_ratio'],
        'max_drawdown_pct': profit_metrics['max_drawdown_pct']
    }


def train_and_save_models(symbols, days=180, interval="5m", optimize_hyperparams=False):
    """
    Train models on historical data and save to models/ directory.
    
    Args:
        symbols: List of stock symbols to train on
        days: Number of days of historical data to use (default 180 = ~6 months)
        interval: Data interval (default 5m)
        optimize_hyperparams: Use Bayesian optimization for hyperparameter tuning
    
    Note: Yahoo Finance has data limits:
        - 1m: last 7 days only
        - 5m/15m/30m: last 60 days only (will auto-adjust if you request more)
        - 60m/90m: last 2 years
        - 1d and above: many years available
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)
    
    # Warn about data limits for intraday intervals
    if interval in ['1m'] and days > 7:
        print(f"[warn] Requested {days} days but {interval} data limited to last 7 days. Will auto-adjust.")
    elif interval in ['5m', '15m', '30m'] and days > 60:
        print(f"[warn] Requested {days} days but {interval} data limited to last 60 days. Will auto-adjust.")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"Training models on {days} days of data ({start_date.date()} to {end_date.date()})")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Interval: {interval}\n")
    
    # Create dummy config (strategy params don't affect model training)
    cfg = StrategyConfig(name="training", holding_period=0)
    
    model_metadata = {
        "training_date": datetime.now().isoformat(),
        "training_days": days,
        "interval": interval,
        "symbols": symbols,
        "models": {}
    }
    
    # Collect all training data
    all_features = []
    all_labels = []
    all_returns = []
    
    print("Loading and processing data...")
    for symbol in symbols:
        try:
            df = load_or_fetch_data(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                interval
            )
            
            if len(df) < 100:
                print(f"  [warn] {symbol}: Insufficient data ({len(df)} bars), skipping")
                continue
            
            # Generate features and labels depending on interval
            if interval == '1d':
                feats = make_daily_features(df)
                future_returns = feats['fwd_ret_1d']
            else:
                feats = make_features(df)
                future_returns = df['close'].pct_change(5).shift(-5).fillna(0)
            
            # Discretize labels
            if interval == '1d':
                labels = np.zeros(len(future_returns), dtype=int)
                labels[future_returns > 0.002] = 1   # BUY (+0.2% next day)
                labels[future_returns < -0.002] = -1  # SELL (-0.2% next day)
            else:
                labels = np.zeros(len(future_returns), dtype=int)
                labels[future_returns > 0.005] = 1   # BUY
                labels[future_returns < -0.005] = -1  # SELL
            
            # Extract feature columns
            if interval == '1d':
                feature_cols = [
                    'ret_1d','ret_5d','ret_10d','vol_20d','sma_10','sma_20','sma_50',
                    'ma_spread_10_20','ma_spread_20_50','macd','macd_signal','macd_hist',
                    'rsi_14','atr_14','price_vs_sma20','price_vs_sma50','vol_z_20','volume_ma_20'
                ]
            else:
                feature_cols = ['ret_1m', 'ma_spread', 'vol_10', 'rsi_14', 'vol_z', 
                                'momentum_5', 'momentum_20', 'vp_ratio', 'vol_regime', 'price_position']
            
            # Check which features exist
            available_features = [col for col in feature_cols if col in feats.columns]
            if not available_features:
                print(f"  [warn] {symbol}: No features available, skipping")
                continue
            
            X = feats[available_features].values
            y = labels  # labels is already a numpy array
            
            # Remove NaN rows
            valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) < 50:
                print(f"  [warn] {symbol}: Insufficient clean data ({len(X_clean)} samples), skipping")
                continue
            
            # Store actual returns for profit metrics
            future_returns_clean = future_returns.values[valid_mask]
            
            all_features.append(X_clean)
            all_labels.append(y_clean)
            all_returns.append(future_returns_clean)
            
            print(f"  ✓ {symbol}: {len(X_clean)} samples")
            
        except Exception as e:
            print(f"  [error] {symbol}: {e}")
            continue
    
    if not all_features:
        print("\n[error] No training data collected. Exiting.")
        return
    
    # Combine all data
    X_combined = np.vstack(all_features)
    y_combined = np.hstack(all_labels)
    returns_combined = np.hstack(all_returns)
    
    print(f"\nTotal training samples: {len(X_combined)}")
    print(f"Class distribution: BUY={np.sum(y_combined==1)}, HOLD={np.sum(y_combined==0)}, SELL={np.sum(y_combined==-1)}")
    
    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test, returns_train, returns_test = train_test_split(
        X_combined, y_combined, returns_combined,
        test_size=0.2,
        random_state=42,
        stratify=y_combined
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train OrdinalLogistic model
    print("\nTraining OrdinalLogistic model...")
    try:
        ordinal_strategy = OrdinalLogisticStrategy(cfg, train_size=500, use_pretrained=False)
        
        # Preprocess features
        X_train_clean = ordinal_strategy._preprocess_features(X_train)
        X_test_clean = ordinal_strategy._preprocess_features(X_test)
        y_train_mapped = y_train + 1  # Map {-1,0,1} to {0,1,2}
        y_test_mapped = y_test + 1
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_test_scaled = scaler.transform(X_test_clean)
        
        # Hyperparameter optimization or default training
        from sklearn.linear_model import LogisticRegression
        
        if optimize_hyperparams and HAS_SKOPT:
            print("  Running Bayesian hyperparameter optimization...")
            
            search_space = {
                'C': Real(0.001, 10.0, prior='log-uniform'),
                'max_iter': Integer(500, 2000),
                'solver': Categorical(['lbfgs', 'saga'])
            }
            
            base_model = LogisticRegression(
                class_weight='balanced',
                random_state=42
            )
            
            opt = BayesSearchCV(
                base_model,
                search_space,
                n_iter=20,
                cv=3,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            opt.fit(X_train_scaled, y_train_mapped)
            model = opt.best_estimator_
            
            print(f"  Best params: {opt.best_params_}")
        else:
            # Default training
            model = LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                class_weight='balanced',
                random_state=42,
                C=1.0
            )
            model.fit(X_train_scaled, y_train_mapped)
        
        # Evaluate model
        print("  Evaluating model...")
        train_metrics = evaluate_model(model, X_train_clean, y_train_mapped, returns_train, scaler)
        test_metrics = evaluate_model(model, X_test_clean, y_test_mapped, returns_test, scaler)
        
        print(f"  Train - Acc: {train_metrics['accuracy']:.3f}, F1: {train_metrics['f1_score']:.3f}, Sharpe: {train_metrics['sharpe_ratio']:.2f}")
        print(f"  Test  - Acc: {test_metrics['accuracy']:.3f}, F1: {test_metrics['f1_score']:.3f}, Sharpe: {test_metrics['sharpe_ratio']:.2f}")
        
        # Save model and scaler
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_cols': available_features,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        with open('models/ordinal_logistic.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        model_metadata['models']['ordinal_logistic'] = {
            'file': 'ordinal_logistic.pkl',
            'training_samples': len(X_train_scaled),
            'test_samples': len(X_test_scaled),
            'features': available_features,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        print(f"  ✓ Saved to models/ordinal_logistic.pkl")
        
    except Exception as e:
        print(f"  [error] Failed to train OrdinalLogistic: {e}")
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    try:
        xgb_strategy = XGBoostStrategy(cfg, train_size=500, n_rounds=50, use_pretrained=False)
        
        # Preprocess features
        X_train_clean = xgb_strategy._preprocess_features(X_train)
        X_test_clean = xgb_strategy._preprocess_features(X_test)
        y_train_mapped = y_train + 1  # Map {-1,0,1} to {0,1,2}
        y_test_mapped = y_test + 1
        
        # Calculate sample weights
        classes, counts = np.unique(y_train_mapped, return_counts=True)
        total = len(y_train_mapped)
        class_weights = {cls: total / (len(classes) * count) for cls, count in zip(classes, counts)}
        sample_weights = np.array([class_weights[y] for y in y_train_mapped])
        
        # Train model
        import xgboost as xgb
        
        if optimize_hyperparams and HAS_SKOPT:
            print("  Running Bayesian hyperparameter optimization...")
            
            search_space = {
                'n_estimators': Integer(30, 100),
                'max_depth': Integer(2, 5),
                'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
                'subsample': Real(0.5, 1.0),
                'colsample_bytree': Real(0.5, 1.0),
                'min_child_weight': Integer(1, 10),
                'gamma': Real(0.0, 0.5),
                'reg_alpha': Real(0.0, 1.0),
                'reg_lambda': Real(0.5, 2.0)
            }
            
            base_model = xgb.XGBClassifier(
                random_state=42,
                verbosity=0,
                tree_method='hist',
                enable_categorical=False
            )
            
            opt = BayesSearchCV(
                base_model,
                search_space,
                n_iter=30,
                cv=3,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            opt.fit(X_train_clean, y_train_mapped, sample_weight=sample_weights)
            model = opt.best_estimator_
            
            print(f"  Best params: {opt.best_params_}")
        else:
            # Default training
            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                colsample_bylevel=0.7,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                tree_method='hist',
                enable_categorical=False
            )
            
            # Split for early stopping
            split_idx = int(len(X_train_clean) * 0.8)
            X_train_fit = X_train_clean[:split_idx]
            y_train_fit = y_train_mapped[:split_idx]
            X_val = X_train_clean[split_idx:]
            y_val = y_train_mapped[split_idx:]
            weights_fit = sample_weights[:split_idx]
            weights_val = sample_weights[split_idx:]
            
            model.fit(
                X_train_fit, y_train_fit,
                sample_weight=weights_fit,
                eval_set=[(X_val, y_val)],
                sample_weight_eval_set=[weights_val],
                verbose=False
            )
        
        # Evaluate model
        print("  Evaluating model...")
        train_metrics = evaluate_model(model, X_train_clean, y_train_mapped, returns_train)
        test_metrics = evaluate_model(model, X_test_clean, y_test_mapped, returns_test)
        
        print(f"  Train - Acc: {train_metrics['accuracy']:.3f}, F1: {train_metrics['f1_score']:.3f}, Sharpe: {train_metrics['sharpe_ratio']:.2f}")
        print(f"  Test  - Acc: {test_metrics['accuracy']:.3f}, F1: {test_metrics['f1_score']:.3f}, Sharpe: {test_metrics['sharpe_ratio']:.2f}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(available_features, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top features: {', '.join([f'{k}={v:.3f}' for k, v in top_features])}")
        
        # Save model
        model_data = {
            'model': model,
            'feature_cols': available_features,
            'confidence_threshold': 0.50,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        with open('models/xgboost.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        model_metadata['models']['xgboost'] = {
            'file': 'xgboost.pkl',
            'training_samples': len(X_train_clean),
            'test_samples': len(X_test_clean),
            'features': available_features,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        print(f"  ✓ Saved to models/xgboost.pkl")
        
    except Exception as e:
        print(f"  [error] Failed to train XGBoost: {e}")
    
    # If training daily interval, also train and save an RNN on sequences
    try:
        if interval == '1d':
            print("\nTraining Daily RNN (GRU) model...")
            # Build sequences from scaled features
            from sklearn.preprocessing import StandardScaler
            scaler_seq = StandardScaler()
            X_scaled_all = scaler_seq.fit_transform(X_combined)
            seq_len = 30  # Increased from 10 for better temporal context
            X_seq = []
            y_seq = []
            for i in range(seq_len, len(X_scaled_all)):
                X_seq.append(X_scaled_all[i-seq_len:i])
                y_seq.append(y_combined[i])
            X_seq = np.asarray(X_seq, dtype=np.float32)
            y_seq = np.asarray(y_seq) + 1  # map to {0,1,2}
            net = NeuralNetClassifier(
                RNNModule,
                module__input_size=X_seq.shape[-1],
                module__hidden_size=128,  # Increased from 64 for better feature extraction
                max_epochs=40,  # Increased from 25 for better convergence
                lr=5e-4,  # Reduced learning rate for stability
                optimizer=torch.optim.Adam,
                criterion=nn.CrossEntropyLoss,
                batch_size=32,  # Reduced batch size for better gradient updates
                device='cpu'
            )

            if optimize_hyperparams and HAS_SKOPT:
                print("  Running Bayesian hyperparameter optimization for RNN...")
                search_space = {
                    'module__hidden_size': Integer(32, 128),
                    'lr': Real(1e-4, 5e-3, prior='log-uniform')
                }
                opt = BayesSearchCV(net, search_space, n_iter=16, cv=3, n_jobs=1, random_state=42, verbose=1)
                opt.fit(X_seq, y_seq)
                net = opt.best_estimator_
                print(f"  Best params: {opt.best_params_}")
            else:
                net.fit(X_seq, y_seq)

            # Save RNN (save weights separately to avoid pickle deserialization issues in multiprocessing)
            model_weights = {name: param.data.cpu().numpy() for name, param in net.module_.named_parameters()}
            with open('models/daily_rnn.pkl', 'wb') as f:
                pickle.dump({
                    'model': net,  # Keep for backward compatibility
                    'model_weights': model_weights,  # Use this for multiprocessing-safe loading
                    'scaler': scaler_seq,
                    'seq_len': seq_len,
                    'hidden_size': getattr(net.module_, 'hidden_size', 64)
                }, f)
            model_metadata['models']['daily_rnn'] = {
                'file': 'daily_rnn.pkl',
                'seq_len': seq_len,
                'features': X_seq.shape[-1]
            }
            print("  ✓ Saved to models/daily_rnn.pkl")
    except Exception as e:
        print(f"  [error] Failed to train Daily RNN: {e}")

    # Save metadata
    with open('models/metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"\n✓ Training complete! Model metadata saved to models/metadata.json")
    print(f"\nNext steps:")
    print(f"  1. Commit the models/ directory to git")
    print(f"  2. Deploy to GitHub Actions - models will be loaded instead of retrained")


def main():
    parser = argparse.ArgumentParser(description="Pre-train ML models for deployment")
    parser.add_argument(
        "--symbols",
        default="SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,NFLX,AMD,INTC,AVGO,ADBE,CSCO",
        help="Comma-separated list of symbols to train on"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days of historical data (default: 180 = ~6 months)"
    )
    parser.add_argument(
        "--interval",
        default="5m",
        help="Data interval (default: 5m)"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Use Bayesian hyperparameter optimization (slower but better results)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached data before training"
    )
    
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        cache_dir = "data/cache"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"[info] Cleared cache directory: {cache_dir}")
    
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    
    train_and_save_models(
        symbols,
        days=args.days,
        interval=args.interval,
        optimize_hyperparams=args.optimize
    )


if __name__ == "__main__":
    main()
