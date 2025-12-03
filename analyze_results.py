#!/usr/bin/env python3
"""Analyze backtest results from multi_summary.json"""

import json

with open('results/multi_summary.json') as f:
    data = json.load(f)

# Calculate strategy totals
strategies = ['daily_logistic', 'daily_xgboost', 'daily_rnn', 'daily_dqn', 'ensemble_weighted']
strategy_stats = {}

for strat in strategies:
    pnl_dollars = []
    pnl_pcts = []
    trades = 0
    errors = 0
    
    for symbol in data:
        strat_data = data[symbol].get(strat, {})
        if 'error' not in strat_data:
            metrics = strat_data.get('metrics', {})
            pnl_pct = metrics.get('total_return_pct', 0.0)
            pnl_dollars.append((pnl_pct / 100.0) * 100000)
            pnl_pcts.append(pnl_pct)
            trades += metrics.get('n_round_trades', 0)
        else:
            errors += 1
    
    if pnl_dollars:
        strategy_stats[strat] = {
            'avg_pnl_dollar': sum(pnl_dollars) / len(pnl_dollars),
            'total_pnl_dollar': sum(pnl_dollars),
            'avg_pnl_pct': sum(pnl_pcts) / len(pnl_pcts),
            'total_trades': trades,
            'symbols': len(pnl_dollars),
            'errors': errors
        }

print('='*95)
print('DAILY MODELS & ENSEMBLE - 3 YEAR BACKTEST (2023-2025)')
print(f'Dataset: {len(data)} symbols × 731 bars (3 years of daily data)')
print('='*95)
print(f"{'Strategy':<20} {'Avg PnL ($)':<15} {'Total PnL ($)':<15} {'Avg Return %':<12} {'Trades':<10} {'Symbols':<10}")
print('-'*95)

best_strategy = None
best_avg_pnl = float('-inf')

for strat in strategies:
    if strat in strategy_stats:
        s = strategy_stats[strat]
        print(f"{strat:<20} ${s['avg_pnl_dollar']:>+12,.2f} ${s['total_pnl_dollar']:>+12,.2f} {s['avg_pnl_pct']:>+10.2f}% {s['total_trades']:>8} {s['symbols']:>9}")
        
        if s['avg_pnl_dollar'] > best_avg_pnl:
            best_avg_pnl = s['avg_pnl_dollar']
            best_strategy = strat

print('='*95)
print(f"\nBest Strategy (by avg PnL): {best_strategy} at ${best_avg_pnl:+,.2f} per symbol")
print(f"Total symbols processed: {len(data)}")
print(f"Data range: 2023-01-01 to 2025-12-02 (731 trading days)")
print(f"Total backtests: {len(data)} symbols × 5 strategies = {len(data) * 5} runs")
