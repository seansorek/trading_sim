#!/usr/bin/env python3
"""
predict_next_day_lite.py - Lightweight prediction script for GitHub Actions.
Only uses trained models from disk, no training dependencies.
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd

from data_loader import load_yfinance
from daily_features import make_daily_features


def load_models() -> Dict:
    """Load pre-trained models from disk."""
    models = {}
    
    if os.path.exists('models/ordinal_logistic.pkl'):
        with open('models/ordinal_logistic.pkl', 'rb') as f:
            models['ordinal_logistic'] = pickle.load(f)
    
    if os.path.exists('models/xgboost.pkl'):
        with open('models/xgboost.pkl', 'rb') as f:
            models['xgboost'] = pickle.load(f)
    
    # Load DQN agent if available
    if os.path.exists('models/dqn_agent.pt'):
        try:
            from dqn_agent import DQNAgent
            
            # Use the DQNAgent's built-in load method
            agent = DQNAgent.load('models/dqn_agent.pt')
            agent.q.eval()
            models['dqn_agent'] = agent
        except Exception as e:
            print(f'[warn] Failed to load DQN agent: {e}')
    
    return models


def predict_symbol(symbol: str, models: Dict) -> Dict:
    """Generate predictions for a symbol using loaded models."""
    result = {'symbol': symbol, 'timestamp': datetime.now().isoformat()}
    
    try:
        # Load 1000 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1000)
        
        df = load_yfinance(
            symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval='1d'
        )
        
        if df is None or len(df) < 50:
            result['error'] = 'Insufficient data'
            return result
        
        # Generate features
        feats = make_daily_features(df)
        # Use same features as RL environment: all columns except ticker, date, fwd_ret_1d
        feature_cols = [c for c in feats.columns if c not in ['ticker', 'date', 'fwd_ret_1d']]
        X = feats[feature_cols].values
        
        # Remove NaN/inf rows
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X_valid = X[valid_mask]
        
        if len(X_valid) < 30:
            result['error'] = 'Insufficient valid features'
            return result
        
        # For DQN: build 20-day window state (latest 20 days, includes close price)
        window_size = 20
        if len(X_valid) >= window_size:
            X_window = X_valid[-window_size:]
            state_dqn = X_window.flatten().astype(np.float32)
        else:
            state_dqn = np.concatenate([X_valid.flatten(), np.zeros(380 - len(X_valid.flatten()))]).astype(np.float32)
        
        # For other models: use latest day only (exclude close price)
        close_idx = feature_cols.index('close')
        X_latest_with_close = X_valid[-1:].copy()
        X_latest = np.delete(X_latest_with_close, close_idx, axis=1)
        result['price'] = float(df['close'].iloc[-1])
        result['predictions'] = {}
        
        # OrdinalLogistic
        if 'ordinal_logistic' in models:
            try:
                model_data = models['ordinal_logistic']
                model = model_data['model']
                scaler = model_data['scaler']
                
                X_scaled = scaler.transform(X_latest)
                prob = model.predict_proba(X_scaled)[0]
                pred = np.argmax(prob)
                confidence = np.max(prob)
                
                signals = ['SELL', 'HOLD', 'BUY']
                result['predictions']['ordinal_logistic'] = {
                    'signal': signals[pred],
                    'confidence': float(confidence)
                }
            except Exception as e:
                result['predictions']['ordinal_logistic'] = {'error': str(e)}
        
        # XGBoost
        if 'xgboost' in models:
            try:
                model_data = models['xgboost']
                model = model_data['model']
                
                prob = model.predict_proba(X_latest)[0]
                pred = np.argmax(prob)
                confidence = np.max(prob)
                
                signals = ['SELL', 'HOLD', 'BUY']
                result['predictions']['xgboost'] = {
                    'signal': signals[pred],
                    'confidence': float(confidence)
                }
            except Exception as e:
                result['predictions']['xgboost'] = {'error': str(e)}
        
        # DQN Agent
        if 'dqn_agent' in models:
            try:
                import torch
                agent = models['dqn_agent']
                
                # DQN expects 20-day window state (flattened)
                state_tensor = torch.FloatTensor(state_dqn).unsqueeze(0).to('cpu')
                with torch.no_grad():
                    q_values = agent.q(state_tensor)
                
                # Q-values for actions: 0=SELL, 1=HOLD, 2=BUY
                q_vals = q_values[0].cpu().numpy()
                pred = np.argmax(q_vals)
                confidence = float(torch.softmax(q_values[0], dim=0).max().item())
                
                signals = ['SELL', 'HOLD', 'BUY']
                result['predictions']['dqn_agent'] = {
                    'signal': signals[pred],
                    'confidence': confidence
                }
            except Exception as e:
                result['predictions']['dqn_agent'] = {'error': str(e)}
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result


def calculate_position_size(confidence: float, signal: str, account_size: float = 100000.0) -> Dict:
    """Calculate position size using Kelly Criterion."""
    if signal == "HOLD" or confidence < 0.50:
        return {'fraction': 0.0, 'dollar_amount': 0.0, 'shares': 0}
    
    # Kelly criterion: win_rate based on confidence
    win_rate = 0.50 + (confidence * 0.20)  # 0.50 conf = 50% win, 0.90 conf = 70% win
    loss_rate = 1.0 - win_rate
    
    # Assumptions: 1.5% avg win/loss per trade
    avg_win = 0.015
    avg_loss = 0.015
    
    kelly_fraction = (win_rate * avg_win - loss_rate * avg_loss) / avg_win if avg_win > 0 else 0
    kelly_fraction = max(0, min(1.0, kelly_fraction))
    
    # Cap at 25% per trade for safety
    position_fraction = min(kelly_fraction, 0.25)
    dollar_amount = position_fraction * account_size
    
    return {
        'fraction': position_fraction,
        'dollar_amount': dollar_amount,
        'shares': int(dollar_amount / 100)  # Estimate based on $100 share price
    }


def send_discord(predictions: list, webhook_url: str) -> bool:
    """Send predictions to Discord webhook with position sizing. Batches embeds in groups of 10."""
    if not webhook_url:
        print('[warn] No webhook URL provided')
        return False
    
    try:
        import requests
    except ImportError:
        print('[error] requests not installed')
        return False
    
    # Build embeds
    embeds = []
    for pred in predictions:
        if 'error' in pred:
            continue
        
        symbol = pred['symbol']
        price = pred.get('price', 'N/A')
        
        # Get signals and confidence scores
        signals = []
        confidence_scores = []
        for model_name, model_pred in pred.get('predictions', {}).items():
            if 'signal' in model_pred:
                signals.append(model_pred['signal'])
                confidence_scores.append(model_pred.get('confidence', 0))
        
        if not signals:
            continue
        
        # Consensus and average confidence
        consensus = signals[0] if all(s == signals[0] for s in signals) else 'MIXED'
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Calculate position size
        pos_size = calculate_position_size(avg_confidence, consensus, account_size=100000.0)
        
        # Color coding
        color_map = {
            'BUY': 0x00ff00,
            'SELL': 0xff0000,
            'HOLD': 0xffff00,
            'MIXED': 0x808080
        }
        color = color_map.get(consensus, 0x808080)
        
        # Build embed
        embed = {
            'title': f'{symbol}',
            'color': color,
            'fields': [
                {'name': 'Signal', 'value': consensus, 'inline': True},
                {'name': 'Confidence', 'value': f'{avg_confidence:.1%}', 'inline': True},
                {'name': 'Price', 'value': f'${price}', 'inline': True},
            ]
        }
        
        # Add position sizing
        if consensus != 'MIXED' and pos_size['fraction'] > 0:
            embed['fields'].extend([
                {'name': 'Position %', 'value': f'{pos_size["fraction"]*100:.1f}%', 'inline': True},
                {'name': 'Amount', 'value': f'${pos_size["dollar_amount"]:.0f}', 'inline': True},
                {'name': 'Shares', 'value': str(pos_size['shares']), 'inline': True},
            ])
        
        # Add per-model details
        for model_name, model_pred in pred.get('predictions', {}).items():
            if 'signal' in model_pred:
                conf = model_pred.get('confidence', 0)
                embed['fields'].append({
                    'name': model_name,
                    'value': f"{model_pred['signal']} ({conf:.1%})",
                    'inline': True
                })
        
        embeds.append(embed)
    
    if not embeds:
        print('[warn] No predictions to send to Discord')
        return False
    
    # Send embeds in batches of 10 (Discord limit)
    success = True
    batch_size = 10
    for i in range(0, len(embeds), batch_size):
        batch = embeds[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(embeds) + batch_size - 1) // batch_size
        
        payload = {
            'username': 'Trading Bot',
            'content': f'**Daily Predictions** - {datetime.now().strftime("%Y-%m-%d %H:%M UTC")} (Batch {batch_num}/{total_batches})',
            'embeds': batch
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code in [204, 200]:
                print(f'[ok] Discord Batch {batch_num}/{total_batches}: {len(batch)} predictions (HTTP {response.status_code})')
            else:
                print(f'[warn] Discord Batch {batch_num}/{total_batches}: HTTP {response.status_code}')
                if response.text:
                    print(f'      Response: {response.text[:200]}')
                success = False
        except Exception as e:
            print(f'[error] Discord Batch {batch_num}/{total_batches} failed: {e}')
            success = False
    
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', default='AAPL,SPY,MSFT')
    parser.add_argument('--webhook', action='store_true')
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    print(f'[info] Predicting for {len(symbols)} symbols...')
    models = load_models()
    
    if not models:
        print('[error] No trained models found')
        sys.exit(1)
    
    predictions = [predict_symbol(sym, models) for sym in symbols]
    
    # Print results
    for pred in predictions:
        if 'error' not in pred:
            sigs = [p.get('signal', '?') for p in pred.get('predictions', {}).values()]
            print(f"[ok] {pred['symbol']}: {' / '.join(sigs)}")
        else:
            print(f"[warn] {pred['symbol']}: {pred['error']}")
    
    # Generate summary
    buy_signals = []
    sell_signals = []
    hold_count = 0
    
    for pred in predictions:
        if 'error' in pred:
            continue
        symbol = pred['symbol']
        preds = pred.get('predictions', {})
        signals = [p.get('signal') for p in preds.values() if 'signal' in p]
        
        if not signals:
            continue
        
        # Determine consensus
        if all(s == 'BUY' for s in signals):
            buy_signals.append((symbol, signals, preds))
        elif all(s == 'SELL' for s in signals):
            sell_signals.append((symbol, signals, preds))
        else:
            hold_count += 1
    
    # Print summary
    print("\n" + "="*50)
    print("TRADING PREDICTIONS SUMMARY")
    print("="*50)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    print(f"BUY Signals: {len(buy_signals)}")
    print(f"SELL Signals: {len(sell_signals)}")
    print(f"HOLD/Mixed: {hold_count}\n")
    
    if buy_signals:
        print("--- BUY SIGNALS ---")
        for symbol, signals, preds in buy_signals:
            price = next((p.get('price') for p in [predictions[i] for i, s in enumerate(symbols) if s == symbol]), 'N/A')
            print(f"{symbol} (${price})")
            for model_name, pred in preds.items():
                if 'signal' in pred:
                    conf = pred.get('confidence', 0)
                    print(f"  {model_name}: {pred['signal']} ({conf:.1%})")
    
    if sell_signals:
        print("\n--- SELL SIGNALS ---")
        for symbol, signals, preds in sell_signals:
            price = next((p.get('price') for p in [predictions[i] for i, s in enumerate(symbols) if s == symbol]), 'N/A')
            print(f"{symbol} (${price})")
            for model_name, pred in preds.items():
                if 'signal' in pred:
                    conf = pred.get('confidence', 0)
                    print(f"  {model_name}: {pred['signal']} ({conf:.1%})")
    
    print("="*50)
    
    # Summary by strategy
    print("\n--- BY STRATEGY ---")
    strategy_stats = {}
    strategy_symbols = {}
    
    for pred in predictions:
        if 'error' in pred:
            continue
        symbol = pred['symbol']
        for strategy, pred_data in pred.get('predictions', {}).items():
            if 'signal' in pred_data:
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                    strategy_symbols[strategy] = {'BUY': [], 'SELL': [], 'HOLD': []}
                
                signal = pred_data['signal']
                strategy_stats[strategy][signal] += 1
                strategy_symbols[strategy][signal].append(symbol)
    
    for strategy in sorted(strategy_stats.keys()):
        stats = strategy_stats[strategy]
        symbols = strategy_symbols[strategy]
        total = sum(stats.values())
        print(f"\n{strategy}:")
        print(f"  BUY:  {stats['BUY']:3d} ({stats['BUY']/total*100:.1f}%) - {', '.join(symbols['BUY']) if symbols['BUY'] else 'None'}")
        print(f"  SELL: {stats['SELL']:3d} ({stats['SELL']/total*100:.1f}%) - {', '.join(symbols['SELL']) if symbols['SELL'] else 'None'}")
        print(f"  HOLD: {stats['HOLD']:3d} ({stats['HOLD']/total*100:.1f}%) - {', '.join(symbols['HOLD']) if symbols['HOLD'] else 'None'}")
    
    print("="*50 + "\n")
    
    # Send Discord (ALWAYS try if webhook URL exists, regardless of flag)
    webhook = os.environ.get('DISCORD_WEBHOOK_URL') or os.environ.get('WEBHOOK_URL')
    if webhook:
        print('[info] Sending to Discord...')
        if send_discord(predictions, webhook):
            print('[ok] Successfully sent predictions to Discord!')
        else:
            print('[error] Failed to send to Discord')
    
    # Save results
    with open('tomorrow_trades.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print('[ok] Results saved to tomorrow_trades.json')


if __name__ == '__main__':
    main()
