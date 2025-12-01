
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
import shutil
import subprocess
import requests
from bs4 import BeautifulSoup

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
from data_loader import load_yfinance, load_csv
from datetime import datetime, timedelta

# Calculate last month's dates for more historical context
end_date = datetime.now()
start_date = end_date - timedelta(days=8)

# Format as strings
start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")
interval = "1m"

def safe_copy(src: str, dst: str):
    """Copy file if it exists; create destination folder as needed."""
    if not os.path.exists(src):
        print(f"[warn] Source missing, skip: {src}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[ok] Copied {src} -> {dst}")


def fetch_expert_opinion(symbol: str) -> dict:
    """
    Fetch expert sentiment and ratings for a stock from multiple sources.
    Returns dict with keys: 'consensus', 'target_price', 'upside_pct', 'num_analysts'
    """
    try:
        # Try Yahoo Finance API for analyst ratings
        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=recommendationTrend"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        rec_trend = data.get('quoteSummary', {}).get('result', [{}])[0].get('recommendationTrend', {})
        trend = rec_trend.get('trend', [])
        
        if trend:
            # Most recent recommendation
            latest = trend[0]
            strong_buy = latest.get('strongBuy', 0)
            buy = latest.get('buy', 0)
            hold = latest.get('hold', 0)
            sell = latest.get('sell', 0)
            strong_sell = latest.get('strongSell', 0)
            total = strong_buy + buy + hold + sell + strong_sell
            
            if total > 0:
                # Calculate weighted score: strong_buy=2, buy=1, hold=0, sell=-1, strong_sell=-2
                score = (strong_buy * 2 + buy * 1 - sell * 1 - strong_sell * 2) / total
                
                if score > 0.5:
                    consensus = "BUY"
                elif score < -0.5:
                    consensus = "SELL"
                else:
                    consensus = "HOLD"
                
                return {
                    "consensus": consensus,
                    "score": float(score),
                    "num_analysts": total,
                    "breakdown": {
                        "strong_buy": strong_buy,
                        "buy": buy,
                        "hold": hold,
                        "sell": sell,
                        "strong_sell": strong_sell
                    }
                }
    except Exception as e:
        print(f"[warn] Could not fetch expert opinion for {symbol}: {e}")
    
    return {
        "consensus": "HOLD",
        "score": 0.0,
        "num_analysts": 0,
        "breakdown": {}
    }


def generate_recommendation(metrics: dict, wf_metrics: dict, expert_opinion: dict) -> str:
    """
    Generate a trading recommendation based on backtest, walk-forward metrics, and expert opinion.
    Returns: 'BUY', 'HOLD', 'SELL', or 'NO_DATA'
    
    Combines:
    - Backtest performance (40%)
    - Walk-forward validation (30%)
    - Risk metrics (20%)
    - Expert opinion prior (10%)
    """
    if not metrics or not wf_metrics:
        return 'NO_DATA'
    
    # Extract key metrics
    backtest_return = metrics.get('total_return_pct', 0)
    wf_return = wf_metrics.get('total_return_pct', 0)
    sharpe = metrics.get('daily_sharpe', 0)
    max_dd = metrics.get('max_drawdown_pct', 0)
    hit_rate = metrics.get('hit_rate', 0)
    
    # Scoring logic
    score = 0
    
    # Profitability (40% weight)
    if backtest_return > 5:
        score += 40
    elif backtest_return > 0:
        score += 20
    elif backtest_return > -5:
        score += 10
    
    # Walk-forward validation (30% weight)
    if wf_return > 2:
        score += 30
    elif wf_return > 0:
        score += 15
    elif wf_return > -2:
        score += 5
    
    # Risk-adjusted returns (20% weight)
    if sharpe > 1.0:
        score += 20
    elif sharpe > 0.5:
        score += 10
    
    # Drawdown check (10% weight)
    if max_dd > -10:
        score += 10
    
    # Expert opinion prior (10% weight, normalized)
    expert_score = expert_opinion.get('score', 0.0)  # Range: [-2, 2]
    if expert_score > 0.5:
        score += 10
    elif expert_score < -0.5:
        score -= 5
    
    # Recommendations
    if score >= 80:
        return 'BUY'
    elif score >= 50:
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
    
    print(f"→ [{symbol}] Strategy: {strategy_name} | Config: {cfg.__dict__}")

    # 1) Signal (from registry)
    signal = build_strategy_signal(strategy_name, cfg, feats, df)

    # 2) Backtest (writes results/equity_curve.png, trade_log.csv, metrics.json)
    bt = Backtester(exec_cfg)
    res = bt.run(df, feats, signal)

    # 3) Walk-forward (out-of-sample)
    wf = walk_forward_backtest(df, feats, train_days=3, test_days=1)

    # 4) Monte Carlo stress test for execution assumptions
    mc = monte_carlo_stress(df, feats, signal, n_runs=n_mc_runs)

    # 5) Persist per-symbol, per-strategy artifacts (copy from the last backtest run)
    base_png   = "results/equity_curve.png"
    base_csv   = "results/equity_curve.csv"
    base_log   = "results/trade_log.csv"
    base_mjson = "results/metrics.json"

    safe_copy(base_png,   f"results/{symbol}_{strategy_name}_equity_curve.png")
    safe_copy(base_csv,   f"results/{symbol}_{strategy_name}_equity_curve.csv")
    safe_copy(base_log,   f"results/{symbol}_{strategy_name}_trade_log.csv")
    safe_copy(base_mjson, f"results/{symbol}_{strategy_name}_metrics.json")

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
        "artifacts": {
            "equity_curve_png": f"{symbol}_{strategy_name}_equity_curve.png",
            "equity_curve_csv": f"{symbol}_{strategy_name}_equity_curve.csv",
            "trade_log_csv":    f"{symbol}_{strategy_name}_trade_log.csv",
            "metrics_json":     f"{symbol}_{strategy_name}_metrics.json",
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-symbol, multi-strategy simulation-only pipeline and build dashboard."
    )
    parser.add_argument("--symbols", default="SPY,AAPL,QQQ",
                        help="Comma-separated list of symbols (e.g., 'SPY,AAPL,QQQ').")
    parser.add_argument("--strategies", default="all",
                        help="Comma-separated list of strategies to run per symbol, or 'all'.")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "csv"],
                        help="Data source: yfinance or csv.")
    parser.add_argument("--interval", default="1m", help="Bar interval (e.g., 1m, 5m).")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (optional)")

    # Shared strategy hyperparameters (applied to all strategies where relevant)
    parser.add_argument("--lookback", type=int, default=20, help="Generic lookback for MR/breakout.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Z-score threshold for MR.")
    parser.add_argument("--rsi-lower", type=int, default=30, help="RSI lower band.")
    parser.add_argument("--rsi-upper", type=int, default=70, help="RSI upper band.")
    parser.add_argument("--mc-runs", type=int, default=10, help="Monte Carlo stress test runs.")

    args = parser.parse_args()

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
        commission_per_share=0.0005,
        slippage_bps=2.0,
        max_position=2000,
        stop_loss_pct=0.03,
        daily_loss_limit_pct=0.02,
    )

    # Fetch expert opinions for all symbols upfront
    expert_opinions = {}
    print("\n[info] Fetching expert opinions from Yahoo Finance...")
    for symbol in symbols:
        expert_opinions[symbol] = fetch_expert_opinion(symbol)
        print(f"  {symbol}: {expert_opinions[symbol]['consensus']} (score: {expert_opinions[symbol]['score']:.2f})")

    for symbol in symbols:
        print(f"\n=== Running symbol: {symbol} ===")

        # Load data
        if args.source == "yfinance":
            df = load_yfinance(symbol, start=start, end=end, interval=interval)
        elif args.source == "csv":
            # For CSV source, 'symbol' should be a file path; multi-symbol via CSV only if you pass multiple file paths.
            df = load_csv(symbol)
        else:
            raise ValueError("Unsupported source. Use 'yfinance' or 'csv'.")

        # Features once per symbol
        feats = make_features(df)

        # Prepare symbol entry for multi_summary
        multi_summary[symbol] = {
            "strategies": strategies,  # the dashboard dropdown reads this list
            "meta": {
                "source": args.source,
                "interval": args.interval,
                "start": args.start,
                "end": args.end,
            },
            "expert_opinion": expert_opinions[symbol],
        }

        # Run each strategy and collect results
        for strat in strategies:
            cfg = StrategyConfig(
                name=strat,
                lookback=args.lookback,
                threshold=args.threshold,
                rsi_lower=args.rsi_lower,
                rsi_upper=args.rsi_upper,
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
                    expert_opinion=expert_opinions[symbol],
                )
                # Enhance with expert opinion
                strat_summary['expert_opinion'] = expert_opinions[symbol]
                multi_summary[symbol][strat] = strat_summary
            except Exception as e:
                print(f"[error] Failed on {symbol} / {strat}: {e}")
                # Still record the failure so dashboard can show a notice
                multi_summary[symbol][strat] = {
                    "metrics": {},
                    "wf_metrics": {},
                    "mc_mean": {},
                    "mc_std": {},
                    "config": cfg.__dict__,
                    "artifacts": {},
                    "expert_opinion": expert_opinions[symbol],
                    "error": str(e),
                }

    # Write combined multi-summary used by the dropdown dashboard
    with open("results/multi_summary.json", "w") as f:
        json.dump(multi_summary, f, indent=2)
    print("\n[ok] Wrote results/multi_summary.json")

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
    print("[ok] Wrote results/recommendations_summary.json")

    # Build the dashboard (site/index.html)
    print("[info] Building multi-symbol strategy dashboard...")
    subprocess.check_call(["python", "build_multi_report.py"])
    print("[ok] Dashboard generated → site/index.html")


if __name__ == "__main__":
    main()
