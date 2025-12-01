
# simulate_multi.py
import os, json, subprocess
from simulation_pipeline import (
    make_features, MeanReversionStrategy, StrategyConfig,
    Backtester, ExecutionConfig, walk_forward_backtest, monte_carlo_stress
)
from data_loader import load_yfinance

SYMBOLS = ["SPY", "AAPL", "QQQ"]   # Adjust as you like
INTERVAL = "1m"

os.makedirs("results", exist_ok=True)
all_results = {}

for symbol in SYMBOLS:
    print(f"Running simulation for {symbol}...")
    df = load_yfinance(symbol, start=None, end=None, interval=INTERVAL)
    feats = make_features(df)
    mr_cfg = StrategyConfig(name="mean_reversion", lookback=20, threshold=0.8)
    mr = MeanReversionStrategy(mr_cfg)
    signal_mr = mr.signal(feats)

    bt = Backtester(ExecutionConfig())
    result = bt.run(df, feats, signal_mr)
    wf_result = walk_forward_backtest(df, feats, train_days=3, test_days=1)
    mc_stats = monte_carlo_stress(df, feats, signal_mr, n_runs=10)

    summary = {
        "symbol": symbol,
        "mean_reversion_metrics": result.metrics,
        "walk_forward_metrics": wf_result.metrics,
        "monte_carlo_metrics_mean": mc_stats.mean().to_dict(),
        "monte_carlo_metrics_std": mc_stats.std().to_dict()
    }
    all_results[symbol] = summary

with open("results/multi_summary.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Build multi-symbol site
subprocess.check_call(["python", "build_multi_report.py"])
