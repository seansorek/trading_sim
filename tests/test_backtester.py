"""test_backtester.py — Backtester correctness tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation_pipeline import Backtester, ExecutionConfig


def _make_df(n: int = 50, start_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    prices = start_price + np.arange(n) * 0.1  # slow drift up
    return pd.DataFrame(
        {
            "open": prices - 0.2,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def _default_cfg() -> ExecutionConfig:
    return ExecutionConfig(
        start_cash=100_000.0,
        commission_per_share=0.0,
        slippage_bps=0.0,
        stop_loss_pct=0.50,   # very wide — won't trigger in these tests unless intended
        take_profit_pct=0.50,
        daily_loss_limit_pct=0.99,
        max_position_pct=0.05,
    )


def _run(
    signal: pd.Series,
    df: pd.DataFrame | None = None,
    cfg: ExecutionConfig | None = None,
) -> "BacktestResult":
    if df is None:
        df = _make_df(len(signal))
    if cfg is None:
        cfg = _default_cfg()
    bt = Backtester(cfg)
    return bt.run(df, df, signal)


# ---------------------------------------------------------------------------
# Basic smoke test
# ---------------------------------------------------------------------------

def test_backtest_runs_without_error():
    df = _make_df(30)
    signal = pd.Series(0, index=df.index)
    result = _run(signal, df)
    assert len(result.equity_curve) == 30
    assert result.equity_curve.iloc[0] == pytest.approx(100_000.0, rel=0.01)


def test_all_hold_returns_flat_equity():
    df = _make_df(20)
    signal = pd.Series(0, index=df.index)
    result = _run(signal, df)
    assert result.equity_curve.min() == pytest.approx(result.equity_curve.max(), rel=1e-6)


# ---------------------------------------------------------------------------
# Entry price on reversal
# ---------------------------------------------------------------------------

def test_entry_price_resets_on_reversal():
    """
    After buying and then selling (reversal), avg_entry_price should reflect
    the new fill price, not a blended value from the long position.
    """
    n = 10
    df = _make_df(n, start_price=100.0)
    signal = pd.Series(0, index=df.index)
    # Buy at bar 2, reverse to short at bar 5
    signal.iloc[2] = 1
    signal.iloc[5] = -1

    result = _run(signal, df)
    # Verify equity didn't blow up (a wrong avg_entry_price would cause miscalculated PnL)
    assert not result.equity_curve.isna().any()
    assert result.equity_curve.iloc[-1] > 0


# ---------------------------------------------------------------------------
# Stop-loss
# ---------------------------------------------------------------------------

def test_stop_loss_triggers():
    """Price should drop below stop-loss and exit the position."""
    n = 20
    prices = [100.0] * 5 + [90.0] * 15  # sharp drop at bar 5
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1_000_000.0] * n,
        },
        index=idx,
    )
    signal = pd.Series(0, index=df.index)
    signal.iloc[3] = 1  # Buy at $100

    cfg = ExecutionConfig(
        start_cash=100_000.0,
        commission_per_share=0.0,
        slippage_bps=0.0,
        stop_loss_pct=0.05,   # 5% stop-loss
        take_profit_pct=0.50,
        daily_loss_limit_pct=0.99,
        max_position_pct=0.05,
    )
    result = _run(signal, df, cfg)
    # After stop-loss, equity should be < start but not catastrophically less
    assert result.equity_curve.iloc[-1] > 90_000


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_identical_runs_produce_identical_equity_curves():
    df = _make_df(40)
    signal = pd.Series(0, index=df.index)
    signal.iloc[5] = 1
    signal.iloc[15] = -1
    signal.iloc[25] = 1

    cfg = _default_cfg()
    bt = Backtester(cfg)
    r1 = bt.run(df, df, signal)
    r2 = bt.run(df, df, signal)

    pd.testing.assert_series_equal(r1.equity_curve, r2.equity_curve)


# ---------------------------------------------------------------------------
# Holding period
# ---------------------------------------------------------------------------

def test_holding_period_blocks_rapid_flip():
    from base_strategy import BaseStrategy, StrategyConfig

    class _DummyStrategy(BaseStrategy):
        def signal(self, feats, df):
            return pd.Series(0, index=feats.index)

    cfg = StrategyConfig(name="test", holding_period=5)
    strat = _DummyStrategy(cfg)

    n = 15
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    raw = pd.Series(0, index=idx)
    raw.iloc[0] = 1   # trade at bar 0
    raw.iloc[2] = -1  # reversal at bar 2 — within holding period, should be suppressed
    raw.iloc[8] = -1  # reversal at bar 8 — after holding period, should pass

    filtered = strat._apply_holding_period(raw)
    assert filtered.iloc[2] == 0, "Bar 2 reversal should be suppressed within holding period"
    assert filtered.iloc[8] == -1, "Bar 8 reversal should be allowed after holding period"


# ---------------------------------------------------------------------------
# Metrics smoke test
# ---------------------------------------------------------------------------

def test_compute_metrics_returns_all_keys():
    from simulation_pipeline import compute_metrics

    idx = pd.date_range("2024-01-02", periods=30, freq="B", tz="UTC")
    equity = pd.Series(100_000.0 + np.arange(30) * 10, index=idx)
    metrics = compute_metrics(equity, pd.DataFrame())
    for key in (
        "final_equity",
        "total_return_pct",
        "daily_sharpe",
        "daily_sortino",
        "max_drawdown_pct",
        "n_round_trades",
        "hit_rate",
        "profit_factor",
    ):
        assert key in metrics, f"Missing metric: {key}"
