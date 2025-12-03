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
                    position_size = calculate_position_size(confidence, signal_name)

                    result["strategies"][strategy_name] = {
                        "signal": signal_name,
                        "confidence": float(confidence),
                        "position_size": float(position_size),
                        "recommendation": f"{signal_name} ({confidence:.0%} confidence, {position_size:.0%} size)",
                    }
                else:
                    result["strategies"][strategy_name] = {
                        "signal": "HOLD",
                        "confidence": 0.0,
                        "position_size": 0.0,
                        "recommendation": "HOLD (no signal)",
                        "error": "No signal generated",
                    }

            except Exception as e:
                print(f"[debug] Strategy {strategy_name} failed: {e}")
                result["strategies"][strategy_name] = {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "position_size": 0.0,
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


def calculate_position_size(confidence: float, signal: str, volatility: float = 1.0) -> float:
    """
    Calculate position size based on prediction confidence and signal.
    
    Args:
        confidence: Prediction confidence (0.0-1.0)
        signal: Trading signal ("BUY", "SELL", "HOLD")
        volatility: Normalized volatility (default 1.0 = normal)
    
    Returns:
        Position size multiplier (0.0-1.0)
    """
    if signal == "HOLD" or confidence < 0.50:
        return 0.0
    
    # Base sizing: scale linearly from confidence 0.50 (min) to 0.90 (max)
    base_size = min(1.0, (confidence - 0.50) / 0.40)  # 0.50->0.0, 0.90->1.0
    
    # Reduce size in high volatility
    vol_adjusted = base_size / max(1.0, volatility)
    
    return min(1.0, vol_adjusted)


def format_webhook_message(predictions: List[Dict]) -> str:
    """
    Format predictions into a readable GitHub Actions webhook message.
    Sorted by strategy and confidence. Includes position sizing and recommendations.
    """
    # Aggregate predictions by strategy
    by_strategy = {}

    for pred in predictions:
        if pred.get("error"):
            continue

        for strategy, data in pred.get("strategies", {}).items():
            if "error" in data:
                continue

            if strategy not in by_strategy:
                by_strategy[strategy] = {"BUY": [], "HOLD": [], "SELL": []}

            signal = data.get("signal", "HOLD")
            confidence = data.get("confidence", 0.0)
            position_size = data.get("position_size", 0.0)
            recommendation = data.get("recommendation", signal)
            
            by_strategy[strategy][signal].append(
                (pred["symbol"], confidence, position_size, recommendation)
            )

    # Build message
    lines = ["# Next Day Trading Recommendations\n"]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    # Sort by strategy
    for strategy in sorted(by_strategy.keys()):
        lines.append(f"\n## {strategy.upper()}\n")

        for signal_type in ["BUY", "SELL", "HOLD"]:
            items = by_strategy[strategy][signal_type]
            if not items:
                continue

            # Sort by confidence (highest first)
            items.sort(key=lambda x: x[1], reverse=True)

            lines.append(f"### {signal_type}")

            for symbol, confidence, position_size, recommendation in items:
                lines.append(f"- **{symbol}**: {recommendation}")

            lines.append("")

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
