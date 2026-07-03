"""Regression tests: verify removed dead code is truly gone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_ordinal_logistic_removed():
    import ml_strategies
    assert not hasattr(ml_strategies, "OrdinalLogisticStrategy"), (
        "OrdinalLogisticStrategy (intraday legacy) must be removed from ml_strategies"
    )


def test_xgboost_intraday_removed():
    import ml_strategies
    assert not hasattr(ml_strategies, "XGBoostStrategy"), (
        "XGBoostStrategy (intraday legacy) must be removed from ml_strategies"
    )


def test_apply_confidence_filter_removed():
    import ml_strategies
    assert not hasattr(ml_strategies, "_apply_confidence_filter"), (
        "_apply_confidence_filter (used only by intraday classes) must be removed"
    )


def test_daily_ridge_q_not_in_registry():
    from simulation_pipeline import STRATEGY_REGISTRY
    assert "daily_ridge_q" not in STRATEGY_REGISTRY, (
        "daily_ridge_q (_DailyRidgeQuantileStrategy) must not appear in STRATEGY_REGISTRY"
    )


def test_walk_forward_backtest_removed():
    import simulation_pipeline
    assert not hasattr(simulation_pipeline, "walk_forward_backtest"), (
        "walk_forward_backtest (intraday-only) must be removed from simulation_pipeline"
    )


def test_make_features_removed():
    import simulation_pipeline
    assert not hasattr(simulation_pipeline, "make_features"), (
        "make_features (intraday feature builder) must be removed from simulation_pipeline"
    )


def test_rsi_removed():
    import simulation_pipeline
    assert not hasattr(simulation_pipeline, "rsi"), (
        "rsi helper (used only by make_features) must be removed from simulation_pipeline"
    )


def test_run_symbol_strategy_wf_always_skipped():
    """After removal, run_symbol_strategy must always return wf_metrics with skipped=True."""
    import pandas as pd
    import numpy as np
    from unittest.mock import patch
    from simulation_pipeline import ExecutionConfig, StrategyConfig
    import simulate_multi

    n = 60
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    prices = 100.0 + np.arange(n, dtype=float) * 0.1
    df = pd.DataFrame(
        {
            "open": prices - 0.2,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )
    exec_cfg = ExecutionConfig(
        start_cash=100_000.0,
        commission_per_share=0.0,
        slippage_bps=0.0,
        stop_loss_pct=0.5,
        take_profit_pct=0.5,
        daily_loss_limit_pct=0.99,
        max_position_pct=0.05,
    )
    flat_signal = pd.Series(0, index=df.index)
    with patch("simulate_multi.build_strategy_signal", return_value=flat_signal), \
         patch("simulate_multi.monte_carlo_stress", return_value=pd.DataFrame()):
        result = simulate_multi.run_symbol_strategy(
            symbol="TEST",
            strategy_name="daily_logistic",
            df=df,
            cfg=StrategyConfig(name="daily_logistic", lookback=20, holding_period=5),
            exec_cfg=exec_cfg,
            run_id="test-run-001",
            n_mc_runs=0,
        )
    assert result["wf_metrics"].get("skipped") is True, (
        f"Expected wf_metrics['skipped']=True after walk_forward removal, got: {result['wf_metrics']}"
    )
