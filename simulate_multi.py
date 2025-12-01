
# simulate_multi.py
import os, json, subprocess, shutil
from simulation_pipeline import (
    make_features, MeanReversionStrategy, StrategyConfig,
    Backtester, ExecutionConfig, walk_forward_backtest, monte_carlo_stress
)
from data_loader import load_yfinance
from datetime import datetime, timedelta

# Calculate last week's dates
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

# Format as strings
start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")


SYMBOLS = ["SPY", "AAPL", "QQQ"]   # Adjust as you like
INTERVAL = "1m"

os.makedirs("results", exist_ok=True)
all_results = {}

for symbol in SYMBOLS:

    print(f"Running simulation for {symbol}...")

    # 1) Load data (yfinance minute data often covers ~7 days)
    df = load_yfinance(symbol, start=start, end=end, interval=INTERVAL)

    # 2) Features + signals
    feats = make_features(df)
    mr_cfg = StrategyConfig(name="mean_reversion", lookback=20, threshold=0.8)
    mr = MeanReversionStrategy(mr_cfg)
    signal_mr = mr.signal(feats)

    # 3) Backtest
    bt = Backtester(ExecutionConfig())
    result = bt.run(df, feats, signal_mr)

    # 4) Walk-forward (train 3 days → test 1 day)
    wf_result = walk_forward_backtest(df, feats, train_days=3, test_days=1)

    # 5) Monte Carlo stress test on execution assumptions
    mc_stats = monte_carlo_stress(df, feats, signal_mr, n_runs=10)

    # 6) Collect metrics
    summary = {
        "symbol": symbol,
        "mean_reversion_metrics": result.metrics,
        "walk_forward_metrics": wf_result.metrics,
        "monte_carlo_metrics_mean": mc_stats.mean().to_dict(),
        "monte_carlo_metrics_std": mc_stats.std().to_dict()
    }
    all_results[symbol] = summary

    # 7) Save symbol-specific artifacts (copy/rename last-run outputs)
    # NOTE: Backtester saved general files; we preserve a per-symbol copy.
    def _copy(src, dst):
        if os.path.exists(src):
            shutil.copy2(src, dst)

    _copy("results/equity_curve.png", f"results/{symbol}_equity_curve.png")
    _copy("results/equity_curve.csv", f"results/{symbol}_equity_curve.csv")
    _copy("results/trade_log.csv",   f"results/{symbol}_trade_log.csv")
    _copy("results/metrics.json",    f"results/{symbol}_metrics.json")


with open("results/multi_summary.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Build multi-symbol site
subprocess.check_call(["python", "build_multi_report.py"])
print("Multi-symbol run complete. See site/index.html after build.")