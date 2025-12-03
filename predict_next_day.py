#!/usr/bin/env python3
"""
predict_next_day.py

Generate trading recommendations for the next day, sorted by strategy and confidence.
Sends results via GitHub Actions webhook (WEBHOOK_URL secret).
"""

import os
import json
import sys
import argparse
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from data_loader import load_yfinance
from simulation_pipeline import (
    make_features,
    StrategyConfig,
    build_strategy_signal,
    STRATEGY_REGISTRY,
)
from daily_features import make_daily_features


def get_daily_prediction(symbol: str, predict_only: bool = True) -> Dict:
    """
    Generate next-day trading prediction for a symbol.
    Returns dict with symbol, predictions from each strategy with confidence.
    """
    result = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "strategies": {},  # {strategy: {signal, confidence}}
        "error": None,
    }

    try:
        # Load last 180 days of daily data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        df = load_yfinance(symbol, start=start_date, end=end_date, interval="1d")

        if df is None or len(df) < 20:
            result["error"] = f"Insufficient data for {symbol}"
            return result

        # Generate daily features
        daily_feats = make_daily_features(df)
        feats = make_features(df)

        # Get predictions from each strategy
        # Note: Skip intraday-only strategies (ordinal_logistic, xgboost) since we don't have intraday data
        intraday_only_strategies = {'ordinal_logistic', 'xgboost'}
        
        for strategy_name in sorted(STRATEGY_REGISTRY.keys()):
            if strategy_name in intraday_only_strategies:
                print(f"[debug] Skipping intraday-only strategy: {strategy_name}")
                result["strategies"][strategy_name] = {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "position_size": {'fraction': 0.0, 'dollar_amount': 0.0},
                    "error": "Intraday-only strategy (requires 5m data)",
                }
                continue
            
            try:
                cfg = StrategyConfig(name=strategy_name, holding_period=0)
                signal_series = build_strategy_signal(
                    strategy_name, cfg, feats, df, predict_only=predict_only
                )

                if signal_series is not None and not signal_series.empty:
                    latest_signal = int(signal_series.iloc[-1])

                    # Calculate confidence based on strategy type
                    confidence = calculate_confidence(
                        strategy_name, daily_feats, signal_series
                    )

                    signal_name = "BUY" if latest_signal == 1 else "SELL" if latest_signal == -1 else "HOLD"
                    position_data = calculate_position_size(confidence, signal_name, volatility=1.0, account_size=100.0)
                    position_display = format_position_size(position_data)

                    result["strategies"][strategy_name] = {
                        "signal": signal_name,
                        "confidence": float(confidence),
                        "position_size": position_data,
                        "recommendation": f"{signal_name} ({confidence:.0%} confidence) | Size: {position_display}",
                    }
                else:
                    result["strategies"][strategy_name] = {
                        "signal": "HOLD",
                        "confidence": 0.0,
                        "position_size": {'fraction': 0.0, 'dollar_amount': 0.0},
                        "recommendation": "HOLD (no signal)",
                        "error": "No signal generated",
                    }

            except Exception as e:
                print(f"[debug] Strategy {strategy_name} failed: {e}")
                result["strategies"][strategy_name] = {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "position_size": {'fraction': 0.0, 'dollar_amount': 0.0},
                    "recommendation": "HOLD (error)",
                    "error": str(e),
                }

    except Exception as e:
        result["error"] = f"Failed to process {symbol}: {str(e)}"

    return result


def calculate_confidence(strategy_name: str, daily_feats: pd.DataFrame, signal_series: pd.Series) -> float:
    """
    Calculate confidence for a strategy's signal (0.0 to 1.0).
    Varies by strategy type.
    """
    try:
        if signal_series.empty:
            return 0.0

        # Base confidence on recent signal consistency
        recent_signals = signal_series.tail(5).values
        signal_consistency = len(set(recent_signals)) == 1  # All same signal

        if "xgboost" in strategy_name.lower():
            # XGBoost: medium confidence, boost if consistent
            base_conf = 0.65
            return min(1.0, base_conf + (0.2 if signal_consistency else 0.0))

        elif "logistic" in strategy_name.lower():
            # Logistic: conservative confidence
            base_conf = 0.55
            return min(1.0, base_conf + (0.15 if signal_consistency else 0.0))

        elif "dqn" in strategy_name.lower():
            # DQN: medium-high confidence if consistent
            base_conf = 0.70
            return min(1.0, base_conf + (0.2 if signal_consistency else 0.0))

        elif "rnn" in strategy_name.lower():
            # RNN: medium confidence
            base_conf = 0.60
            return min(1.0, base_conf + (0.2 if signal_consistency else 0.0))

        else:
            # Other strategies: low-medium confidence
            base_conf = 0.50
            return min(1.0, base_conf + (0.15 if signal_consistency else 0.0))

    except Exception:
        return 0.5


def calculate_position_size(confidence: float, signal: str, volatility: float = 1.0, account_size: float = 100.0) -> Dict[str, float]:
    """
    Calculate position size using multiple methods and return sizing recommendations.
    
    Methods:
    1. KELLY CRITERION: Optimal bet sizing for long-term growth
       Kelly% = (win% * avg_win - loss% * avg_loss) / avg_win
    2. VOLATILITY-ADJUSTED: Inverse volatility scaling (lower vol = bigger position)
    3. CONFIDENCE-SCALED: Linear scaling based on prediction confidence
    
    Args:
        confidence: Prediction confidence (0.0-1.0)
        signal: Trading signal ("BUY", "SELL", "HOLD")
        volatility: Normalized volatility (default 1.0 = normal)
        account_size: Account size in dollars (default $100)
    
    Returns:
        Dict with:
            - 'fraction': Position size as % of account (0.0-1.0)
            - 'dollar_amount': Position size in dollars (0.0-account_size)
            - 'kelly': Kelly Criterion recommendation
            - 'conservative': Half-Kelly (safer)
    """
    if signal == "HOLD" or confidence < 0.50:
        return {
            'fraction': 0.0,
            'dollar_amount': 0.0,
            'kelly': 0.0,
            'conservative': 0.0,
            'method': 'HOLD'
        }
    
    # === METHOD 1: KELLY CRITERION ===
    # Assumptions: 55% win rate at confidence level, 1:1 reward/risk
    win_rate = 0.50 + (confidence * 0.20)  # 0.50 confidence = 50% win rate, 0.90 = 70% win rate
    loss_rate = 1.0 - win_rate
    avg_win = 0.015  # 1.5% avg win per trade
    avg_loss = 0.015  # 1.5% avg loss per trade
    
    kelly_fraction = (win_rate * avg_win - loss_rate * avg_loss) / avg_win if avg_win > 0 else 0
    kelly_fraction = max(0, min(1.0, kelly_fraction))  # Clamp to [0, 1]
    
    # === METHOD 2: VOLATILITY-ADJUSTED ===
    # Higher volatility → smaller position
    # Lower volatility → bigger position
    base_size = min(1.0, (confidence - 0.50) / 0.40)  # 0.50→0.0, 0.90→1.0
    vol_adjusted = base_size / max(1.0, volatility)
    vol_adjusted = min(1.0, vol_adjusted)
    
    # === METHOD 3: CONFIDENCE-SCALED (Original) ===
    confidence_scaled = min(1.0, (confidence - 0.50) / 0.40)
    
    # === RECOMMENDATION: Use Kelly with conservative cap ===
    # Kelly can be aggressive, so use Kelly * 0.5 (Half-Kelly) for safety
    position_fraction = min(kelly_fraction, 0.25)  # Cap at 25% of account per trade
    conservative_fraction = position_fraction * 0.5  # Half-Kelly for more conservative approach
    
    return {
        'fraction': position_fraction,
        'dollar_amount': position_fraction * account_size,
        'kelly': kelly_fraction,
        'conservative': conservative_fraction,
        'confidence': confidence,
        'volatility': volatility,
        'method': 'KELLY_CRITERION'
    }


def format_position_size(position_data: Dict[str, float]) -> str:
    """Format position size for display with dollar amounts and percentages."""
    if position_data['fraction'] == 0.0:
        return "HOLD (0%)"
    
    frac_pct = position_data['fraction'] * 100
    dollar = position_data['dollar_amount']
    conservative = position_data['conservative']
    conservative_dollar = conservative * (position_data.get('dollar_amount', 100) / position_data['fraction']) if position_data['fraction'] > 0 else 0
    
    return f"{frac_pct:.0f}% (${dollar:.2f} | Conservative: {conservative*100:.0f}% ${conservative_dollar:.2f})"


def format_webhook_message(predictions: List[Dict]) -> str:
    """
    Format predictions into a detailed Discord embed-compatible message.
    Organized by signal type with actionable sizing recommendations.
    """
    # Aggregate by signal type across all strategies
    buy_signals = []
    sell_signals = []
    hold_signals = []
    
    for pred in predictions:
        if pred.get("error"):
            continue
        
        symbol = pred.get("symbol", "?")
        for strategy, data in pred.get("strategies", {}).items():
            if "error" in data or data.get("signal") == "HOLD":
                continue
            
            signal = data.get("signal", "HOLD")
            confidence = data.get("confidence", 0.0)
            position_data = data.get("position_size", {})
            dollar_amount = position_data.get("dollar_amount", 0.0) if isinstance(position_data, dict) else 0.0
            
            entry = {
                'symbol': symbol,
                'strategy': strategy.upper().replace('_', ' '),
                'confidence': confidence,
                'dollar': dollar_amount,
                'kelly': position_data.get('kelly', 0.0) if isinstance(position_data, dict) else 0.0,
                'recommendation': data.get("recommendation", "")
            }
            
            if signal == "BUY":
                buy_signals.append(entry)
            elif signal == "SELL":
                sell_signals.append(entry)
            else:
                hold_signals.append(entry)
    
    # Sort by confidence (highest first)
    buy_signals.sort(key=lambda x: x['confidence'], reverse=True)
    sell_signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Build Discord message
    lines = []
    lines.append("=== DAILY TRADING PREDICTIONS ===")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")
    
    # Summary stats
    total_signals = len(buy_signals) + len(sell_signals)
    if total_signals > 0:
        lines.append("--- SUMMARY ---")
        lines.append(f"BUY Signals: {len(buy_signals)} [UP]")
        lines.append(f"SELL Signals: {len(sell_signals)} [DOWN]")
        lines.append(f"Total Capital Required (Aggressive): ${sum(s['dollar'] for s in buy_signals) + sum(s['dollar'] for s in sell_signals):.2f}")
        lines.append("")
    
    # BUY signals
    if buy_signals:
        lines.append("--- BUY SIGNALS (Highest Confidence) ---")
        lines.append("")
        for entry in buy_signals[:10]:  # Top 10
            lines.append(f"{entry['symbol']}")
            lines.append(f"  Strategy: {entry['strategy']}")
            lines.append(f"  Confidence: {entry['confidence']:.0%}")
            lines.append(f"  Position Size: ${entry['dollar']:.2f} (Kelly: {entry['kelly']:.1%})")
            lines.append("")
    
    # SELL signals
    if sell_signals:
        lines.append("--- SELL SIGNALS (Highest Confidence) ---")
        lines.append("")
        for entry in sell_signals[:10]:  # Top 10
            lines.append(f"{entry['symbol']}")
            lines.append(f"  Strategy: {entry['strategy']}")
            lines.append(f"  Confidence: {entry['confidence']:.0%}")
            lines.append(f"  Position Size: ${entry['dollar']:.2f} (Kelly: {entry['kelly']:.1%})")
            lines.append("")
    
    if total_signals == 0:
        lines.append("--- No actionable signals today ---")
        lines.append("All strategies recommend HOLD or have low confidence.")
    
    lines.append("---")
    lines.append("Note: Use Conservative sizing (Half-Kelly) for risk-averse approach")
    
    return "\n".join(lines)


def send_webhook_message(message: str):
    """Send message to GitHub Actions webhook via WEBHOOK_URL secret."""
    webhook_url = os.environ.get("WEBHOOK_URL")

    if not webhook_url:
        print("[warn] WEBHOOK_URL not set, printing message instead:")
        print(message)
        return

    import requests

    payload = {
        "text": message,
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code in [200, 204]:
            print(f"[ok] Webhook message sent successfully")
        else:
            print(f"[error] Webhook failed: HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}")
    except Exception as e:
        print(f"[error] Failed to send webhook: {e}")


def main():
    parser = argparse.ArgumentParser(description="Predict next day trades")
    parser.add_argument(
        "--symbols",
        default="AAPL,SPY,MSFT,GOOGL,NVDA,TSLA,META,NFLX,AMD,INTC",
        help="Comma-separated symbols to predict",
    )
    parser.add_argument(
        "--webhook",
        action="store_true",
        help="Send results to webhook",
    )

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    print(f"[info] Generating next-day predictions for {len(symbols)} symbols...")
    predictions = []

    for symbol in symbols:
        print(f"  [{symbol}] Generating predictions...", end=" ")
        pred = get_daily_prediction(symbol)
        if pred.get("error"):
            print(f"ERROR: {pred['error']}")
        else:
            strategies_success = len([s for s in pred.get("strategies", {}).values() if "error" not in s])
            print(f"OK ({strategies_success} strategies)")
            predictions.append(pred)

    # Format and send message
    if predictions:
        message = format_webhook_message(predictions)
        print("\n" + message)
        
        # Save timestamped predictions file
        import os
        os.makedirs("results", exist_ok=True)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_file = f"results/predictions_{run_timestamp}.json"
        
        predictions_data = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "predictions": predictions,
        }
        
        with open(predictions_file, "w") as f:
            json.dump(predictions_data, f, indent=2)
        
        print(f"\n[ok] Predictions saved to {predictions_file}")

        if args.webhook:
            send_webhook_message(message)
    else:
        print("[error] No successful predictions generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
