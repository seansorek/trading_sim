
#!/usr/bin/env python3
"""
simulate_multi.py

Runs multiple strategies for multiple symbols in a simulation-only pipeline,
saves per-symbol/per-strategy artifacts, builds a multi-symbol summary JSON,
and triggers the HTML dashboard builder (build_multi_report.py).

Artifacts written per symbol+strategy:
- results/{SYMBOL}_{STRATEGY}_equity_curve.png
- results/{SYMBOL}_{STRATEGY}_equity_curve.csv
- results/{SYMBOL}_{STRATEGY}_trade_log.csv
- results/{SYMBOL}_{STRATEGY}_metrics.json

Index:
- results/multi_summary.json
- site/index.html (via build_multi_report.py)
"""

import os
import json
import argparse
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import pandas as pd

# Import your pipeline components
from simulation_pipeline import (
    make_features,
    StrategyConfig,
    ExecutionConfig,
    Backtester,
    walk_forward_backtest,
    monte_carlo_stress,
    build_strategy_signal,
    STRATEGY_REGISTRY,  # so we can use all available strategy names automatically
)
from data_loader import load_yfinance, load_csv, load_alpha_vantage
import yfinance as yf
from datetime import datetime, timedelta

def is_weekly_run():
    """Check if this is the Monday 6am weekly run."""
    now = datetime.now()
    # Check if it's Monday (weekday 0) and between 6am-7am
    return now.weekday() == 0 and 6 <= now.hour < 7

def fetch_expert_opinion(symbol: str) -> dict:
    """
    Fetch expert sentiment and ratings for a stock using yfinance.
    Returns dict with keys: 'consensus', 'score', 'analyst_count'
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # yfinance provides recommendation ratings
        recommendation = info.get('recommendationKey', 'hold').upper()
        analyst_count = info.get('numberOfAnalystOpinions', 0)
        
        # Map recommendation to consensus and score
        if recommendation in ['STRONG_BUY', 'BUY']:
            consensus = 'BUY'
            score = 2.0
        elif recommendation == 'HOLD':
            consensus = 'HOLD'
            score = 0.0
        elif recommendation in ['STRONG_SELL', 'SELL']:
            consensus = 'SELL'
            score = -2.0
        else:
            consensus = 'HOLD'
            score = 0.0
        
        return {
            "consensus": consensus,
            "score": float(score),
            "num_analysts": int(analyst_count),
            "recommendation_key": recommendation
        }
        
    except Exception as e:
        print(f"[warn] Could not fetch expert opinion for {symbol}: {e}")
        return {
            "consensus": "HOLD",
            "score": 0.0,
            "num_analysts": 0,
            "recommendation_key": "unknown"
        }


def generate_recommendation(metrics: dict, wf_metrics: dict, expert_opinion: dict) -> str:
    """
    Generate a trading recommendation based on backtest, walk-forward metrics, and expert opinion.
    Returns: 'BUY', 'HOLD', 'SELL', or 'NO_DATA'
    
    Combines:
    - Backtest performance (30%)
    - Walk-forward validation (20%)
    - Risk metrics (20%)
    - Expert opinion prior (30%)
    """
    if not metrics or not wf_metrics:
        return 'NO_DATA'
    
    # Extract key metrics
    backtest_return = metrics.get('total_return_pct', 0)
    wf_return = wf_metrics.get('total_return_pct', 0)
    sharpe = metrics.get('daily_sharpe', 0)
    max_dd = metrics.get('max_drawdown_pct', 0)
    
    # Scoring logic
    score = 0
    
    # Backtest profitability (30% weight, less strict)
    if backtest_return > 2:
        score += 30
    elif backtest_return > 0:
        score += 20
    elif backtest_return > -3:
        score += 10
    elif backtest_return > -5:
        score += 5
    
    # Walk-forward validation (20% weight, much less strict)
    if wf_return > 1:
        score += 20
    elif wf_return > 0:
        score += 15
    elif wf_return > -1:
        score += 10
    elif wf_return > -2:
        score += 5
    
    # Risk-adjusted returns (20% weight)
    if sharpe > 0.5:
        score += 20
    elif sharpe > 0:
        score += 10
    
    # Drawdown check (10% weight)
    if max_dd > -10:
        score += 10
    elif max_dd > -15:
        score += 5
    
    # Expert opinion prior (30% weight, greatly increased)
    expert_score = expert_opinion.get('score', 0.0)
    expert_consensus = expert_opinion.get('consensus', 'HOLD')
    
    if expert_consensus == 'BUY':
        score += 30
    elif expert_consensus == 'HOLD':
        score += 20
    elif expert_consensus == 'SELL':
        score += 5
    
    # Boost if expert score is positive
    if expert_score > 0.5:
        score += 5
    elif expert_score < -0.5:
        score -= 5
    
    # Recommendations with lower thresholds
    if score >= 65:
        return 'BUY'
    elif score >= 35:
        return 'HOLD'
    else:
        return 'SELL'


def run_symbol_strategy(symbol: str,
                        strategy_name: str,
                        df,
                        feats,
                        cfg: StrategyConfig,
                        exec_cfg: ExecutionConfig,
                        n_mc_runs: int = 10,
                        expert_opinion: dict = None):
    """
    Build signal for (symbol, strategy), run backtest, walk-forward, Monte Carlo,
    save artifacts, and return summary dict for multi_summary.json.
    """
    if expert_opinion is None:
        expert_opinion = {}
    
    # Define unique artifact paths for this specific run
    artifact_base = f"results/{symbol}_{strategy_name}"
    artifacts = {
        "equity_curve_csv": f"{artifact_base}_equity_curve.csv",
        "trade_log_csv":    f"{artifact_base}_trade_log.csv",
        "metrics_json":     f"{artifact_base}_metrics.json",
    }

    # 1) Signal (from registry)
    signal = build_strategy_signal(strategy_name, cfg, feats, df)

    # 2) Backtest (now writes directly to unique, symbol-specific files)
    bt = Backtester(exec_cfg)
    res = bt.run(df, feats, signal, artifact_paths=artifacts)

    # 3) Track historical trades (results saved to results/ folder)

    # 4) Walk-forward (out-of-sample)
    wf = walk_forward_backtest(df, feats, train_days=3, test_days=1)

    # 5) Monte Carlo stress test for execution assumptions
    mc = monte_carlo_stress(df, feats, signal, n_runs=n_mc_runs)

    # 6) Generate recommendation with expert opinion prior
    recommendation = generate_recommendation(res.metrics, wf.metrics, expert_opinion)

    # 7) Summary for dashboard
    summary = {
        "metrics": res.metrics,            # backtest metrics for this strategy
        "wf_metrics": wf.metrics,          # walk-forward (out-of-sample) metrics
        "mc_mean": mc.mean().to_dict(),    # Monte Carlo average metrics
        "mc_std": mc.std().to_dict(),      # Monte Carlo std dev of metrics
        "config": cfg.__dict__,            # strategy config used
        "recommendation": recommendation,  # BUY / HOLD / SELL
        "artifacts": {k: os.path.basename(v) for k, v in artifacts.items()},
        "current_run_trades": len(res.trades),  # Number of trades in this run
    }
    return summary


def process_symbol(symbol: str, strategies: list, args, expert_opinions: dict, 
                   exec_cfg: ExecutionConfig, start: str, end: str, interval: str, api_key: str = None):
    """
    Process a single symbol across all strategies.
    Runs in a separate thread.
    """
    logs = []
    logs.append(f"\n=== Running symbol: {symbol} ===")
    
    symbol_results = {
        "meta": {
            "source": args.source,
            "interval": args.interval,
            "start": args.start,
            "end": args.end,
        },
        "expert_opinion": expert_opinions[symbol],
    }
    
    # Load data
    try:
        if args.source == "yfinance":
            df = load_yfinance(symbol, start=start, end=end, interval=interval)
        elif args.source == "csv":
            df = load_csv(symbol)
        elif args.source == "alphavantage":
            df = load_alpha_vantage(symbol, api_key=api_key, interval=interval)
        else:
            raise ValueError("Unsupported source. Use 'yfinance' or 'csv'.")
    except Exception as e:
        logs.append(f"[error] Failed to load data for {symbol}: {e}")
        symbol_results["meta"]["error"] = f"Failed to load data: {str(e)}"
        return symbol, symbol_results, logs
    
    print(f"\n[info] Processing {symbol} ({len(df)} bars)")
    
    # Features once per symbol
    feats = make_features(df)

    # Run each strategy and collect results
    for strat in strategies:
        cfg = StrategyConfig(
            name=strat,
            lookback=args.lookback,
            threshold=args.threshold,
            rsi_lower=args.rsi_lower,
            rsi_upper=args.rsi_upper,
            holding_period=args.holding_period,
        )
        
        try:
            strat_summary = run_symbol_strategy(
                symbol=symbol,
                strategy_name=strat,
                df=df,
                feats=feats,
                cfg=cfg,
                exec_cfg=exec_cfg,
                n_mc_runs=args.mc_runs,
                expert_opinion=expert_opinions[symbol]
            )
            # Enhance with expert opinion
            strat_summary['expert_opinion'] = expert_opinions[symbol]
            symbol_results[strat] = strat_summary
            
            # Buffer PnL summary for this strategy
            metrics = strat_summary.get('metrics', {})
            pnl_pct = metrics.get('total_return_pct', 0.0)
            pnl_dollar = (pnl_pct / 100.0) * 100000  # Assuming $100k starting capital
            n_trades = metrics.get('n_round_trades', 0)
            logs.append(f"  [{symbol}] {strat}: PnL = ${pnl_dollar:+,.2f} ({pnl_pct:+.2f}%) | Trades: {n_trades}")
        except Exception as e:
            logs.append(f"[error] Failed on {symbol} / {strat}: {e}")
            # Still record the failure so dashboard can show a notice
            symbol_results[strat] = {
                "metrics": {},
                "wf_metrics": {},
                "mc_mean": {},
                "mc_std": {},
                "config": cfg.__dict__,
                "artifacts": {},
                "expert_opinion": expert_opinions[symbol],
                "error": str(e),
            }
    
    return symbol, symbol_results, logs


def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(
        description="Run multi-symbol, multi-strategy simulation-only pipeline and build dashboard."
    )
    parser.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,NFLX,AMD,INTC,AVGO,ADBE,CSCO,CRM,NVSN,IBM,DXCM,SQ,SHOP,ZM,DOCU,CRWD,OKTA,NET,ROKU,COIN,HOOD,LCID,PLTR",
                        help="Comma-separated list of symbols (e.g., 'SPY,AAPL,QQQ').")
    parser.add_argument("--strategies", type=str, default='daily_logistic,daily_xgboost,daily_rnn,daily_dqn',
                        help="Comma-separated list of strategies to run per symbol, or 'all'.")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "csv", "alphavantage"],
                        help="Data source: yfinance, csv, or alphavantage.")
    parser.add_argument("--interval", default="5m", help="Bar interval (e.g., 1m, 5m).")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (optional)")
    parser.add_argument("--av-key", default=None, help="Alpha Vantage API key. Can also be set via AV_API_KEY environment variable.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes for parallel processing (default: number of CPUs).")

    # Shared strategy hyperparameters (applied to all strategies where relevant)
    parser.add_argument("--lookback", type=int, default=20, help="Generic lookback for MR/breakout.")
    parser.add_argument("--threshold", type=float, default=1.2, help="Z-score threshold for MR (higher = fewer trades).")
    parser.add_argument("--rsi-lower", type=int, default=20, help="RSI lower band (lower = fewer buys).")
    parser.add_argument("--rsi-upper", type=int, default=80, help="RSI upper band (higher = fewer sells).")
    parser.add_argument("--holding-period", type=int, default=78, help="Minimum bars to hold between position changes (78 bars = 1 trading day for 5m intervals).")
    parser.add_argument("--mc-runs", type=int, default=10, help="Monte Carlo stress test runs.")

    args = parser.parse_args()

    # --- Date Handling ---
    # Set default date range to the last 14 days
    if args.end is None:
        end_date = datetime.now()
        args.end = end_date.strftime("%Y-%m-%d")
    
    if args.start is None:
        start_date = datetime.strptime(args.end, "%Y-%m-%d") - timedelta(days=14)
        args.start = start_date.strftime("%Y-%m-%d")

    # Determine number of workers (default: min(available CPUs, 8) for stability)
    if args.workers is None:
        args.workers = min(multiprocessing.cpu_count(), 4)
    print(f"[info] Using {args.workers} worker processes for parallel processing")

    # Handle Alpha Vantage API key
    api_key = args.av_key or os.environ.get("AV_API_KEY")
    if args.source == "alphavantage" and not api_key:
        raise ValueError(
            "Alpha Vantage source requires an API key. "
            "Provide it with --av-key or set the AV_API_KEY environment variable.")

    # Parse lists
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.strategies.lower() == "all":
        strategies = sorted(STRATEGY_REGISTRY.keys())
    else:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    if not symbols or not strategies:
        raise ValueError("Provide at least one symbol and one strategy (or 'all').")

    os.makedirs("results", exist_ok=True)

    # Combined summary for dropdown dashboard
    multi_summary = {}

    # Common execution config for all runs (tweak as needed)
    exec_cfg = ExecutionConfig(
        commission_per_share=0.00005,  # Reduced from 0.0001 for better profitability
        slippage_bps=1.0,              # Reduced from 2.0 - use limit orders
        max_position=2000,
        stop_loss_pct=0.05,            # Widened from 0.03 to avoid premature exits
        daily_loss_limit_pct=0.05,     # Increased from 0.02
    )

    # Fetch expert opinions for all symbols upfront (sequentially)
    expert_opinions = {}
    print("\n[info] Fetching expert opinions from Yahoo Finance...")
    for symbol in symbols:
        expert_opinions[symbol] = fetch_expert_opinion(symbol)
        print(f"  {symbol}: {expert_opinions[symbol]['consensus']} (score: {expert_opinions[symbol]['score']:.2f})")

    # Process symbols in parallel using ProcessPoolExecutor
    print(f"\n[info] Processing {len(symbols)} symbols using {args.workers} processes...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for symbol in symbols:
            future = executor.submit(
                process_symbol,
                symbol=symbol,
                strategies=strategies,
                args=args,
                expert_opinions=expert_opinions,
                exec_cfg=exec_cfg,
                start=args.start,
                end=args.end,
                interval=args.interval,
                api_key=api_key,
            )
            futures[future] = symbol

        # Collect results without printing, then print in input order
        completed = 0
        results_by_symbol = {}
        logs_by_symbol = {}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                s, res, logs = future.result()
                results_by_symbol[s] = res
                logs_by_symbol[s] = logs
                completed += 1
            except Exception as e:
                logs_by_symbol[symbol] = [f"[error] Process failed for {symbol}: {e}"]

        # Now assign multi_summary and print logs deterministically
        for idx, symbol in enumerate(symbols, start=1):
            if symbol in results_by_symbol:
                multi_summary[symbol] = results_by_symbol[symbol]
            if symbol in logs_by_symbol:
                for line in logs_by_symbol[symbol]:
                    print(line)
                print(f"[ok] Completed {symbol} ({idx}/{len(symbols)})")

    # Write combined multi-summary used by the dropdown dashboard
    with open("results/multi_summary.json", "w") as f:
        json.dump(multi_summary, f, indent=2)

    # Create timestamped run results file
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "symbols": symbols,
            "strategies": strategies,
            "interval": args.interval,
            "start": args.start,
            "end": args.end,
            "workers": args.workers,
        },
        "results": multi_summary,
    }
    
    with open(f"results/run_{run_timestamp}.json", "w") as f:
        json.dump(run_results, f, indent=2)

    # Build a sorted recommendations summary for Discord
    recommendations_sorted = {}
    for symbol in sorted(multi_summary.keys()):
        recommendations_sorted[symbol] = {}
        for strat in strategies:
            strat_data = multi_summary[symbol].get(strat, {})
            rec = strat_data.get("recommendation", "NO_DATA")
            recommendations_sorted[symbol][strat] = rec
    
    # Write recommendations summary
    with open("results/recommendations_summary.json", "w") as f:
        json.dump(recommendations_sorted, f, indent=2)

    # Print PnL summary table (averages only)
    print("\n" + "="*100)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("="*100)
    print(f"{'Strategy':<20} {'Avg PnL ($)':<15} {'Avg PnL (%)':<12} {'Total PnL ($)':<15} {'Total Trades':<15} {'Symbols':<10}")
    print("-"*100)
    
    strategy_totals = {strat: {'pnl_dollar': 0, 'pnl_pct': 0, 'trades': 0, 'count': 0} for strat in strategies}
    
    for strat in strategies:
        for symbol in sorted(multi_summary.keys()):
            strat_data = multi_summary[symbol].get(strat, {})
            metrics = strat_data.get('metrics', {})
            pnl_pct = metrics.get('total_return_pct', 0.0)
            pnl_dollar = (pnl_pct / 100.0) * 100000  # Assuming $100k starting capital
            n_trades = metrics.get('n_round_trades', 0)
            
            if 'error' not in strat_data:
                strategy_totals[strat]['pnl_dollar'] += pnl_dollar
                strategy_totals[strat]['pnl_pct'] += pnl_pct
                strategy_totals[strat]['trades'] += n_trades
                strategy_totals[strat]['count'] += 1

    for strat in strategies:
        totals = strategy_totals[strat]
        if totals['count'] > 0:
            avg_pnl_dollar = totals['pnl_dollar'] / totals['count']
            avg_pnl_pct = totals['pnl_pct'] / totals['count']
            total_trades = totals['trades']
            symbol_count = totals['count']
            total_pnl_dollar = totals['pnl_dollar']
            print(f"{strat:<20} ${avg_pnl_dollar:>+12,.2f} {avg_pnl_pct:>+10.2f}% ${total_pnl_dollar:>+12,.2f} {total_trades:>13} {symbol_count:>9}")
    print("="*100 + "\n")

    # Trade statistics generated during backtest runs
    
    # Calculate current run statistics
    current_run_trades = sum(
        multi_summary.get(symbol, {}).get(strat, {}).get('current_run_trades', 0)
        for symbol in multi_summary.keys()
        for strat in strategies
    )
    
    # Calculate current run PnL from the multi_summary
    current_run_pnl = sum(
        multi_summary.get(symbol, {}).get(strat, {}).get('metrics', {}).get('total_return_pct', 0) / 100.0 * 100000
        for symbol in multi_summary.keys()
        for strat in strategies
        if 'error' not in multi_summary.get(symbol, {}).get(strat, {})
    )
    
    # Show current run stats or full history based on whether it's the weekly run
    weekly_run = is_weekly_run()
    if weekly_run:
        print(f"[ok] Weekly Run - Full Trade History:")
        print(f"     Total trades in this run: {current_run_trades}")
        print(f"     P&L for this run: ${current_run_pnl:+,.2f}")
    else:
        print(f"[ok] Current Run Statistics:")
        print(f"     Trades in this run: {current_run_trades}")
        print(f"     P&L for this run: ${current_run_pnl:+,.2f}")
        print(f"     (Run on Monday 6-7am for full historical stats)")

    # Build the dashboard (site/index.html)
    try:
        subprocess.run(
            [sys.executable, "build_multi_report.py"],
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
    except subprocess.CalledProcessError as e:
        print(f"[warn] Dashboard build failed: {e.stderr}")
    except Exception as e:
        print(f"[warn] Dashboard build error: {e}")
    
    # Calculate and save runtime statistics
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60
    
    runtime_stats = {
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "elapsed_minutes": round(elapsed_minutes, 2),
        "symbols_processed": len(symbols),
        "strategies_per_symbol": len(strategies),
        "total_runs": len(symbols) * len(strategies),
        "workers_used": args.workers,
        "data_source": args.source,
        "interval": args.interval,
    }
    
    with open("results/runtime_stats.json", "w") as f:
        json.dump(runtime_stats, f, indent=2)
    
    print(f"\n[Completed in {elapsed_minutes:.2f} min | {runtime_stats['total_runs']} backtests]")

if __name__ == "__main__":
    main()
