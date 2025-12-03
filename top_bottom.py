#!/usr/bin/env python3
"""Show top and bottom performing combinations"""

import json

with open('results/multi_summary.json') as f:
    data = json.load(f)

print('\nTOP 10 PERFORMING SYMBOL-STRATEGY COMBINATIONS (3-Year Backtest):')
print('='*80)
print(f"{'Symbol':<10} {'Strategy':<20} {'PnL (USD)':<15} {'Return %':<12} {'Trades':<8} {'Sharpe':<8}")
print('-'*80)

results = []
for symbol in data:
    for strat in ['daily_logistic', 'daily_xgboost', 'daily_rnn', 'daily_dqn', 'ensemble_weighted']:
        strat_data = data[symbol].get(strat, {})
        if 'error' not in strat_data:
            metrics = strat_data.get('metrics', {})
            pnl_pct = metrics.get('total_return_pct', 0.0)
            pnl_dollar = (pnl_pct / 100.0) * 100000
            trades = metrics.get('n_round_trades', 0)
            sharpe = metrics.get('daily_sharpe', 0.0)
            results.append((symbol, strat, pnl_dollar, pnl_pct, trades, sharpe))

# Sort by PnL descending
results.sort(key=lambda x: x[2], reverse=True)

for i, (symbol, strat, pnl_dollar, pnl_pct, trades, sharpe) in enumerate(results[:10], 1):
    print(f"{symbol:<10} {strat:<20} ${pnl_dollar:>+12,.2f} {pnl_pct:>+10.2f}% {trades:>7} {sharpe:>7.2f}")

print('='*80)
print('\nBOTTOM 10 (Worst Performers):')
print('='*80)
print(f"{'Symbol':<10} {'Strategy':<20} {'PnL (USD)':<15} {'Return %':<12} {'Trades':<8} {'Sharpe':<8}")
print('-'*80)

for i, (symbol, strat, pnl_dollar, pnl_pct, trades, sharpe) in enumerate(results[-10:], 1):
    print(f"{symbol:<10} {strat:<20} ${pnl_dollar:>+12,.2f} {pnl_pct:>+10.2f}% {trades:>7} {sharpe:>7.2f}")

print('='*80)

# Group by strategy, show best symbols for each
print('\n\nBEST SYMBOLS PER STRATEGY:')
print('='*80)

by_strat = {}
for symbol, strat, pnl_dollar, pnl_pct, trades, sharpe in results:
    if strat not in by_strat:
        by_strat[strat] = []
    by_strat[strat].append((symbol, pnl_dollar, pnl_pct))

for strat in ['daily_logistic', 'daily_xgboost', 'daily_rnn', 'daily_dqn', 'ensemble_weighted']:
    if strat in by_strat:
        by_strat[strat].sort(key=lambda x: x[1], reverse=True)
        print(f"\n{strat.upper()}:")
        print(f"  Top 5: {', '.join(f'{s[0]} (+${s[1]:,.0f})' for s in by_strat[strat][:5])}")
        print(f"  Worst 3: {', '.join(f'{s[0]} (${s[1]:,.0f})' for s in by_strat[strat][-3:])}")
