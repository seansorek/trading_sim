#!/usr/bin/env python3
"""
trade_history.py

Manages a persistent trade history across simulation runs.
Tracks all trades, calculates performance statistics, and maintains a historical database.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

HISTORY_DIR = "results/trade_history"
HISTORY_FILE = os.path.join(HISTORY_DIR, "all_trades.jsonl")
STATS_FILE = os.path.join(HISTORY_DIR, "statistics.json")


def ensure_history_dir():
    """Create history directory if it doesn't exist."""
    os.makedirs(HISTORY_DIR, exist_ok=True)


def append_trade(symbol: str, strategy: str, trade_log: pd.DataFrame, run_timestamp: str):
    """
    Append trades from a strategy run to the historical trade log.
    """
    ensure_history_dir()
    
    if trade_log.empty:
        return
    
    # Add metadata to each trade
    trades_with_meta = []
    for ts, row in trade_log.iterrows():
        trade = row.to_dict()
        trade['symbol'] = symbol
        trade['strategy'] = strategy
        trade['run_timestamp'] = run_timestamp
        trade['timestamp'] = str(ts)
        trades_with_meta.append(trade)
    
    # Append to JSONL (one JSON object per line)
    with open(HISTORY_FILE, 'a') as f:
        for trade in trades_with_meta:
            f.write(json.dumps(trade) + '\n')


def calculate_trade_stats() -> dict:
    """
    Calculate comprehensive statistics from historical trades.
    Returns dict with performance metrics.
    """
    ensure_history_dir()
    
    if not os.path.exists(HISTORY_FILE):
        return {
            "total_trades": 0,
            "message": "No historical trades yet"
        }
    
    # Read all trades
    trades = []
    with open(HISTORY_FILE, 'r') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[warn] Skipping malformed JSON on line {i} in {HISTORY_FILE}")
    
    if not trades:
        return {"total_trades": 0, "message": "No trades in history"}
    
    df = pd.DataFrame(trades)
    
    # Parse timestamps
    # Parse timestamps; enforce UTC to avoid mixed timezone warnings
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    # Calculate round-trip PnL
    pnl_list = []
    position = {}  # Track open positions by (symbol, strategy)
    
    for idx, row in df.sort_values('timestamp').iterrows():
        key = (row['symbol'], row['strategy'])
        side = 1 if row['side'] == 'BUY' else -1
        fill_price = row['fill_price']
        shares = row['shares']
        commission = row.get('commission', 0)
        
        if key not in position:
            position[key] = {'shares': 0, 'avg_price': 0}
        
        pos = position[key]
        
        # Entry
        if pos['shares'] == 0:
            pos['shares'] = side * shares
            pos['avg_price'] = fill_price
        # Exit
        else:
            if np.sign(pos['shares']) == side:
                # Adding to position
                total_cost = pos['avg_price'] * abs(pos['shares']) + fill_price * shares
                pos['shares'] += side * shares
                pos['avg_price'] = total_cost / abs(pos['shares'])
            else:
                # Closing position
                pnl = (fill_price - pos['avg_price']) * np.sign(pos['shares']) * shares - commission
                pnl_list.append({
                    'symbol': row['symbol'],
                    'strategy': row['strategy'],
                    'pnl': pnl,
                    'exit_price': fill_price,
                    'entry_price': pos['avg_price'],
                    'shares': shares,
                    'timestamp': row['timestamp']
                })
                pos['shares'] -= side * shares
                if pos['shares'] == 0:
                    pos['avg_price'] = 0
    
    pnl_df = pd.DataFrame(pnl_list) if pnl_list else pd.DataFrame()
    
    # Calculate statistics
    stats = {
        "total_trades": len(df),
        "date_range": {
            "start": df['timestamp'].min().isoformat(),
            "end": df['timestamp'].max().isoformat()
        },
        "by_symbol": {},
        "by_strategy": {},
        "overall": {}
    }
    
    # Overall stats
    if not pnl_df.empty:
        pnls = pnl_df['pnl'].dropna()
        stats['overall'] = {
            "total_pnl": float(pnls.sum()),
            "avg_pnl": float(pnls.mean()),
            "std_pnl": float(pnls.std()),
            "min_pnl": float(pnls.min()),
            "max_pnl": float(pnls.max()),
            "win_rate": float((pnls > 0).mean()),
            "num_winning_trades": int((pnls > 0).sum()),
            "num_losing_trades": int((pnls < 0).sum()),
            "profit_factor": float(pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else float('inf'),
            "expectancy": float(pnls.mean())
        }
    
    # By symbol
    for symbol in df['symbol'].unique():
        symbol_trades = df[df['symbol'] == symbol]
        symbol_pnl = pnl_df[pnl_df['symbol'] == symbol]['pnl'] if not pnl_df.empty else pd.Series([])
        
        if not symbol_pnl.empty:
            stats['by_symbol'][symbol] = {
                "trades": len(symbol_trades),
                "round_trips": len(symbol_pnl),
                "total_pnl": float(symbol_pnl.sum()),
                "avg_pnl": float(symbol_pnl.mean()),
                "win_rate": float((symbol_pnl > 0).mean())
            }
    
    # By strategy
    for strategy in df['strategy'].unique():
        strat_trades = df[df['strategy'] == strategy]
        strat_pnl = pnl_df[pnl_df['strategy'] == strategy]['pnl'] if not pnl_df.empty else pd.Series([])
        
        if not strat_pnl.empty:
            stats['by_strategy'][strategy] = {
                "trades": len(strat_trades),
                "round_trips": len(strat_pnl),
                "total_pnl": float(strat_pnl.sum()),
                "avg_pnl": float(strat_pnl.mean()),
                "win_rate": float((strat_pnl > 0).mean())
            }
    
    return stats


def save_stats():
    """Calculate and save statistics to file."""
    ensure_history_dir()
    stats = calculate_trade_stats()
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    return stats


def get_stats() -> dict:
    """Get cached statistics."""
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    return {}


def export_trades_csv():
    """Export all trades to CSV for analysis."""
    ensure_history_dir()
    
    if not os.path.exists(HISTORY_FILE):
        return None
    
    trades = []
    with open(HISTORY_FILE, 'r') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[warn] Skipping malformed JSON on line {i} in {HISTORY_FILE}")
    
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    csv_path = os.path.join(HISTORY_DIR, "all_trades.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def get_recent_trades(limit: int = 50) -> list:
    """Get the most recent trades."""
    ensure_history_dir()
    
    if not os.path.exists(HISTORY_FILE):
        return []
    
    trades = []
    with open(HISTORY_FILE, 'r') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[warn] Skipping malformed JSON on line {i} in {HISTORY_FILE}")
    
    return trades[-limit:] if trades else []


if __name__ == "__main__":
    stats = save_stats()
    print(json.dumps(stats, indent=2))
