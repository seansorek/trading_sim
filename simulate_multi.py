#!/usr/bin/env python3
"""
simulate_multi.py — Multi-symbol, multi-strategy backtest runner.

Runs backtests in parallel across symbols and strategies, writes results
to the SQLite DB, and saves per-run JSON/CSV artifacts.
"""
import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from data_loader import load_yfinance
from oos_guard import enforce_oos_start, get_artifact_train_end
from simulation_pipeline import (
    Backtester,
    ExecutionConfig,
    StrategyConfig,
    STRATEGY_REGISTRY,
    build_strategy_signal,
    monte_carlo_stress,
)

# Default pickle path for each pretrained-model strategy, used to read the
# artifact's train_end cutoff (see issue #115). daily_dqn is intentionally
# excluded — its torch checkpoint doesn't carry the same metadata dict and
# is covered separately.
_STRATEGY_MODEL_PATHS = {
    "daily_logistic": "models/daily_logistic.pkl",
    "daily_xgboost": "models/daily_xgboost.pkl",
    "daily_predictor": "models/daily_predictor.pkl",
    "daily_hybrid": "models/daily_hybrid.pkl",
}


def _oos_trim_for_strategy(df, strategy_name: str, symbol: str):
    """Trim `df` to rows after the strategy's pretrained-model train cutoff.

    Returns `df` unchanged if the strategy has no known model path, the
    model file doesn't exist, or the artifact predates the train_end field
    (nothing to enforce in that case beyond a logged warning).
    """
    model_path = _STRATEGY_MODEL_PATHS.get(strategy_name)
    if model_path is None or not os.path.exists(model_path):
        return df
    try:
        import pickle
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
    except Exception as exc:
        logger.warning(
            "oos_guard: could not read %s artifact %s: %s", strategy_name, model_path, exc
        )
        return df
    train_end = get_artifact_train_end(artifact)
    return enforce_oos_start(df, train_end, label=f"{symbol}/{strategy_name}")

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/simulate_multi.log"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-symbol-strategy runner (executes in worker process)
# ---------------------------------------------------------------------------

def run_symbol_strategy(
    symbol: str,
    strategy_name: str,
    df,
    cfg: StrategyConfig,
    exec_cfg: ExecutionConfig,
    run_id: str,
    n_mc_runs: int = 10,
    spy_df=None,
) -> dict:
    """Run backtest + Monte Carlo for one (symbol, strategy) pair."""
    artifact_base = f"results/{symbol}_{strategy_name}"
    artifacts = {
        "equity_curve_csv": f"{artifact_base}_equity_curve.csv",
        "trade_log_csv": f"{artifact_base}_trade_log.csv",
        "metrics_json": f"{artifact_base}_metrics.json",
    }

    # Enforce out-of-sample date boundaries: never backtest a pretrained
    # model over rows that were part of its training window (issue #115).
    df = _oos_trim_for_strategy(df, strategy_name, symbol)
    if df is None or len(df) < 30:
        raise ValueError(
            f"{symbol}/{strategy_name}: requested backtest range has no (or too few) "
            "out-of-sample rows after the model's training cutoff + embargo."
        )

    feats = df  # all registered strategies are daily_ and build features internally

    signal = build_strategy_signal(strategy_name, cfg, feats, df, spy_df=spy_df)

    bt = Backtester(exec_cfg)
    res = bt.run(
        df, feats, signal,
        artifact_paths=artifacts,
        run_id=run_id,
        symbol=symbol,
        strategy=strategy_name,
    )

    wf_metrics: dict = {"skipped": True}
    mc = monte_carlo_stress(
        df, feats, signal, n_runs=n_mc_runs,
        base_exec_cfg=exec_cfg,
        out_csv=f"{artifact_base}_monte_carlo_stats.csv",
    )

    return {
        "metrics": res.metrics,
        "wf_metrics": wf_metrics,
        "mc_mean": mc.mean().to_dict() if not mc.empty else {},
        "mc_std": mc.std().to_dict() if not mc.empty else {},
        "config": cfg.__dict__,
        "artifacts": {k: os.path.basename(v) for k, v in artifacts.items()},
        "n_trades": len(res.trades),
    }


def process_symbol(
    symbol: str,
    strategies: list[str],
    exec_cfg: ExecutionConfig,
    start: str,
    end: str,
    run_id: str,
    holding_period: int,
    lookback: int,
    n_mc_runs: int,
) -> tuple[str, dict, list[str]]:
    """Load data for one symbol and run all strategies. Returns (symbol, results, log_lines)."""
    logs: list[str] = [f"\n=== {symbol} ==="]

    try:
        df = load_yfinance(symbol, start=start, end=end, interval="1d")
    except Exception as exc:
        logs.append(f"[error] Data load failed for {symbol}: {exc}")
        return symbol, {"error": str(exc)}, logs

    if df is None or len(df) < 30:
        msg = f"Insufficient data ({len(df) if df is not None else 0} bars)"
        logs.append(f"[warn] {symbol}: {msg}")
        return symbol, {"error": msg}, logs

    logger.info("Processing %s (%d bars)", symbol, len(df))

    spy_df = None
    if symbol != "SPY":
        try:
            spy_df = load_yfinance("SPY", start=start, end=end, interval="1d")
        except Exception as exc:
            logger.warning("Could not load SPY data for %s relative features: %s", symbol, exc)

    results: dict = {}
    for strat in strategies:
        cfg = StrategyConfig(
            name=strat,
            lookback=lookback,
            holding_period=holding_period,
        )
        try:
            summary = run_symbol_strategy(
                symbol=symbol,
                strategy_name=strat,
                df=df,
                cfg=cfg,
                exec_cfg=exec_cfg,
                run_id=run_id,
                n_mc_runs=n_mc_runs,
                spy_df=spy_df,
            )
            results[strat] = summary
            pnl_pct = summary["metrics"].get("total_return_pct", 0.0)
            pnl_dollar = (pnl_pct / 100.0) * exec_cfg.start_cash
            logs.append(
                f"  {symbol} / {strat}: PnL ${pnl_dollar:+,.0f} ({pnl_pct:+.2f}%), "
                f"trades={summary['n_trades']}"
            )
        except Exception as exc:
            logs.append(f"  [error] {symbol} / {strat}: {exc}")
            results[strat] = {"metrics": {}, "error": str(exc)}

    return symbol, results, logs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="Multi-symbol backtest runner")
    parser.add_argument(
        "--symbols",
        default="SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA",
        help="Comma-separated symbols",
    )
    parser.add_argument(
        "--strategies",
        default="daily_logistic,daily_xgboost",
        help="Comma-separated strategy names, or 'all'",
    )
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=365, help="Days of history (if --start not set)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--mc-runs", type=int, default=10)
    parser.add_argument("--holding-period", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--db", default="data/trading_sim.db")
    args = parser.parse_args()

    end_str = args.end or datetime.now().strftime("%Y-%m-%d")
    start_str = args.start or (
        datetime.strptime(end_str, "%Y-%m-%d") - timedelta(days=args.days)
    ).strftime("%Y-%m-%d")

    workers = args.workers or min(multiprocessing.cpu_count(), 4)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.strategies.lower() == "all":
        strategies = sorted(STRATEGY_REGISTRY.keys())
    else:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    unknown = [s for s in strategies if s not in STRATEGY_REGISTRY]
    if unknown:
        logger.error("Unknown strategies: %s. Available: %s", unknown, list(STRATEGY_REGISTRY))
        sys.exit(1)

    from config import get_config
    cfg = get_config()
    exec_cfg = ExecutionConfig(
        start_cash=cfg.execution.start_cash,
        commission_per_share=cfg.execution.commission_per_share,
        slippage_bps=cfg.execution.slippage_bps,
        stop_loss_pct=cfg.execution.stop_loss_pct,
        take_profit_pct=cfg.execution.take_profit_pct,
        daily_loss_limit_pct=cfg.execution.daily_loss_limit_pct,
        max_position_pct=cfg.execution.max_position_pct,
    )

    from db import DB
    db = DB(args.db)
    run_id = DB.new_run_id()

    logger.info(
        "Run %s: %d symbols × %d strategies, %s→%s, %d workers",
        run_id[:8], len(symbols), len(strategies), start_str, end_str, workers,
    )

    start_time = time.time()
    multi_summary: dict = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_symbol,
                symbol=sym,
                strategies=strategies,
                exec_cfg=exec_cfg,
                start=start_str,
                end=end_str,
                run_id=run_id,
                holding_period=args.holding_period,
                lookback=args.lookback,
                n_mc_runs=args.mc_runs,
            ): sym
            for sym in symbols
        }

        completed_results: dict = {}
        completed_logs: dict = {}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                s, res, logs = future.result(timeout=300)
                completed_results[s] = res
                completed_logs[s] = logs
            except Exception as exc:
                completed_logs[sym] = [f"[error] {sym}: worker crashed: {exc}"]

    # Print logs in deterministic order and populate summary
    for sym in symbols:
        for line in completed_logs.get(sym, []):
            print(line)
        if sym in completed_results:
            multi_summary[sym] = completed_results[sym]

    # Write DB backtest run records
    for sym, sym_results in multi_summary.items():
        for strat, strat_data in sym_results.items():
            if strat in ("error",) or not isinstance(strat_data, dict):
                continue
            metrics = strat_data.get("metrics", {})
            if not metrics:
                continue
            try:
                db.insert_backtest_run(
                    run_id=run_id,
                    symbol=sym,
                    strategy=strat,
                    data_start=start_str,
                    data_end=end_str,
                    start_cash=exec_cfg.start_cash,
                    final_equity=metrics.get("final_equity"),
                    total_return_pct=metrics.get("total_return_pct"),
                    daily_sharpe=metrics.get("daily_sharpe"),
                    daily_sortino=metrics.get("daily_sortino"),
                    max_drawdown_pct=metrics.get("max_drawdown_pct"),
                    n_round_trades=metrics.get("n_round_trades"),
                    hit_rate=metrics.get("hit_rate"),
                    profit_factor=metrics.get("profit_factor"),
                    commission_per_share=exec_cfg.commission_per_share,
                    slippage_bps=exec_cfg.slippage_bps,
                    stop_loss_pct=exec_cfg.stop_loss_pct,
                    holding_period=args.holding_period,
                )
            except Exception as exc:
                logger.warning("DB insert failed for %s/%s: %s", sym, strat, exc)

    # Write JSON artifacts
    with open("results/multi_summary.json", "w") as f:
        json.dump(multi_summary, f, indent=2)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "symbols": symbols,
            "strategies": strategies,
            "start": start_str,
            "end": end_str,
            "workers": workers,
        },
        "results": multi_summary,
    }
    with open(f"results/run_{run_ts}.json", "w") as f:
        json.dump(run_record, f, indent=2)

    # Performance summary
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print("STRATEGY PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<22} {'Avg Return':<12} {'Avg Sharpe':<12} {'Avg Trades':<12} {'Symbols'}")
    print(f"{'-'*80}")

    for strat in strategies:
        returns, sharpes, trades, count = [], [], [], 0
        for sym in multi_summary:
            strat_data = multi_summary[sym].get(strat, {})
            if "error" in strat_data or not strat_data.get("metrics"):
                continue
            m = strat_data["metrics"]
            returns.append(m.get("total_return_pct", 0))
            sharpes.append(m.get("daily_sharpe", 0))
            trades.append(m.get("n_round_trades", 0))
            count += 1
        if count:
            import numpy as np
            print(
                f"{strat:<22} {np.mean(returns):>+8.2f}%     "
                f"{np.mean(sharpes):>+8.2f}     "
                f"{int(np.mean(trades)):>8}      {count}"
            )

    print(f"{'='*80}")
    print(f"Completed {len(symbols)} symbols in {elapsed:.1f}s (run_id={run_id[:8]})")

    # Optional dashboard rebuild
    if os.path.exists("build_multi_report.py"):
        try:
            subprocess.run(
                [sys.executable, "build_multi_report.py"],
                check=True,
                capture_output=True,
                timeout=60,
            )
            logger.info("Dashboard rebuilt.")
        except Exception as exc:
            logger.warning("Dashboard build failed: %s", exc)


if __name__ == "__main__":
    main()
