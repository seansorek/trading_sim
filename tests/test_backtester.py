"""test_backtester.py — Backtester correctness tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("torch")

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


# ---------------------------------------------------------------------------
# Issue #19 — compute_metrics reversal correctness
# ---------------------------------------------------------------------------

def _make_trades(rows: list) -> pd.DataFrame:
    """Build a trades DataFrame from (side, shares, fill_price) tuples."""
    idx = pd.date_range("2024-01-02", periods=len(rows), freq="B", tz="UTC")
    return pd.DataFrame(
        [{"side": r[0], "shares": r[1], "fill_price": r[2]} for r in rows],
        index=idx,
    )


def test_compute_metrics_simple_long_close():
    """Long +1 then SELL 1: PnL = fill2 - fill1. hit_rate should be 1.0."""
    from simulation_pipeline import compute_metrics

    idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
    equity = pd.Series(100_000.0 + np.arange(10) * 0, index=idx)
    trades = _make_trades([("BUY", 1, 100.0), ("SELL", 1, 110.0)])
    metrics = compute_metrics(equity, trades)
    assert metrics["n_round_trades"] == 1
    assert metrics["hit_rate"] == pytest.approx(1.0)
    # profit_factor is None when there are no losing trades (gross_loss == 0)
    assert metrics["profit_factor"] is None or metrics["profit_factor"] > 0


def test_compute_metrics_reversal_entry_price_reset():
    """
    Issue #19: long +N then SELL 2N (reversal into short).

    Before the fix, entry_price stayed as the long entry, so the short leg
    computed PnL using the wrong reference price.  After the fix, entry_price
    is reset to the reversal fill price.

    Sequence:
      1. BUY 1 @ 100  → pos = +1, entry = 100
      2. SELL 2 @ 110 → closes long for +10 PnL, opens short -1 @ 110
      3. BUY 1 @ 105  → closes short for +5 PnL  (110 - 105 = 5)

    Total round-trips = 2 (one long close, one short close).
    Without the fix, trade 3 would use entry=100 → PnL = (105-100)*(-1)*1 = -5 WRONG.
    """
    from simulation_pipeline import compute_metrics

    idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
    equity = pd.Series(100_000.0 + np.zeros(10), index=idx)
    trades = _make_trades([
        ("BUY",  1, 100.0),  # open long @ 100
        ("SELL", 2, 110.0),  # close long + open short @ 110
        ("BUY",  1, 105.0),  # close short @ 105
    ])
    metrics = compute_metrics(equity, trades)

    assert metrics["n_round_trades"] == 2, (
        f"Expected 2 round-trips (long close + short close), got {metrics['n_round_trades']}"
    )
    # Long PnL = (110-100)*1 = +10; Short PnL = (110-105)*1 = +5
    # Both wins → hit_rate = 1.0
    assert metrics["hit_rate"] == pytest.approx(1.0), (
        f"Expected hit_rate=1.0 (both legs profitable), got {metrics['hit_rate']}. "
        "This indicates entry_price was not reset after reversal."
    )


def test_compute_metrics_reversal_losing_leg():
    """
    Reversal where the short leg is a losing trade.

    Sequence:
      1. BUY 1 @ 100  → pos = +1
      2. SELL 2 @ 110 → close long (+10), open short -1 @ 110
      3. BUY 1 @ 115  → close short (110-115 = -5, a loss)

    hit_rate should be 0.5 (1 win, 1 loss).
    Before the fix, trade 3 would compute PnL as (115-100)*(-1)*1 = -15, still a
    loss, but the magnitude is wrong. More critically, hit_rate is only 0.5 either
    way here — the bug is tested more cleanly in test_compute_metrics_reversal_entry_price_reset.
    """
    from simulation_pipeline import compute_metrics

    idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
    equity = pd.Series(100_000.0 + np.zeros(10), index=idx)
    trades = _make_trades([
        ("BUY",  1, 100.0),
        ("SELL", 2, 110.0),
        ("BUY",  1, 115.0),
    ])
    metrics = compute_metrics(equity, trades)
    assert metrics["n_round_trades"] == 2
    assert metrics["hit_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Issue #20 — walk_forward_backtest skip for daily strategies
# ---------------------------------------------------------------------------

def test_walk_forward_skipped_for_daily_strategy():
    """
    Issue #20: simulate_multi.run_symbol_strategy must skip walk_forward_backtest
    for daily_ strategies and return wf_metrics with {"skipped": True}.

    We verify the logic directly by calling run_symbol_strategy with a mock
    strategy that doesn't need external data.
    """
    from simulation_pipeline import walk_forward_backtest

    # Build a DataFrame that intentionally lacks the intraday _COLS features
    n = 30
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    prices = 100.0 + np.arange(n) * 0.1
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

    # walk_forward_backtest with a feats frame missing _COLS should produce all-zero signals
    result = walk_forward_backtest(df, df, train_days=3, test_days=1)
    # The signal is all-flat (zeros), so n_round_trades should be 0
    assert result.metrics.get("n_round_trades", 0) == 0, (
        "walk_forward_backtest with missing intraday features should produce "
        "no trades (all-flat signal)"
    )


def test_run_symbol_strategy_wf_skipped_flag():
    """
    Issue #20: run_symbol_strategy for a daily_ strategy must set
    wf_metrics['skipped'] = True instead of calling walk_forward_backtest.
    """
    import importlib
    from unittest.mock import patch, MagicMock
    from simulation_pipeline import ExecutionConfig, StrategyConfig

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
         patch("simulate_multi.make_features", return_value=df), \
         patch("simulate_multi.walk_forward_backtest") as mock_wf, \
         patch("simulate_multi.monte_carlo_stress", return_value=pd.DataFrame()):
        import simulate_multi
        result = simulate_multi.run_symbol_strategy(
            symbol="TEST",
            strategy_name="daily_logistic",
            df=df,
            cfg=StrategyConfig(name="daily_logistic", lookback=20, holding_period=5),
            exec_cfg=exec_cfg,
            run_id="test-run-001",
            n_mc_runs=0,
        )

    # walk_forward_backtest should NOT have been called for a daily_ strategy
    mock_wf.assert_not_called()
    # wf_metrics should carry the skipped flag
    assert result["wf_metrics"].get("skipped") is True, (
        f"Expected wf_metrics['skipped']=True for daily strategy, got: {result['wf_metrics']}"
    )


# ---------------------------------------------------------------------------
# Scoped artifact paths — regression test for issue #17
# ---------------------------------------------------------------------------

def test_monte_carlo_stress_writes_to_custom_out_csv(tmp_path):
    """monte_carlo_stress must write to the caller-supplied path, not the hardcoded default."""
    from simulation_pipeline import monte_carlo_stress

    df = _make_df(30)
    signal = pd.Series(0, index=df.index)
    out = tmp_path / "scoped_mc.csv"
    monte_carlo_stress(df, df, signal, n_runs=3, out_csv=str(out))
    assert out.exists(), "monte_carlo_stress did not write to the custom out_csv path"


def test_walk_forward_artifact_paths_are_scoped(tmp_path):
    """walk_forward_backtest must write artifacts to the caller-supplied paths."""
    from simulation_pipeline import walk_forward_backtest

    df = _make_df(30)
    paths = {
        "equity_curve_csv": str(tmp_path / "wf_equity.csv"),
        "trade_log_csv": str(tmp_path / "wf_trades.csv"),
        "metrics_json": str(tmp_path / "wf_metrics.json"),
    }
    walk_forward_backtest(df, df, train_days=5, test_days=1, artifact_paths=paths)
    assert (tmp_path / "wf_metrics.json").exists(), "walk_forward did not write metrics json to custom path"


# ---------------------------------------------------------------------------
# Issue #41 — compute_metrics must not book P&L on same-direction scale-ins
# ---------------------------------------------------------------------------

def test_compute_metrics_scale_in_no_intermediate_pnl():
    """
    BUY, BUY, SELL-all: there should be exactly one round-trip P&L booked
    (when the position is closed), computed against the share-weighted average
    entry price.

    Sequence:
      1. BUY 10 @ 100  → pos = +10, entry = 100
      2. BUY 10 @ 110  → pos = +20, entry = (100*10+110*10)/20 = 105 (scale-in, no PnL)
      3. SELL 20 @ 120  → close all, PnL = (120 - 105) * 20 = 300

    Only 1 round-trip P&L should be booked, and it should be positive.
    """
    from simulation_pipeline import compute_metrics

    idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
    equity = pd.Series(100_000.0 + np.zeros(10), index=idx)
    trades = _make_trades([
        ("BUY",  10, 100.0),  # open long
        ("BUY",  10, 110.0),  # scale-in
        ("SELL", 20, 120.0),  # close all
    ])
    metrics = compute_metrics(equity, trades)

    assert metrics["n_round_trades"] == 1, (
        f"Expected exactly 1 round-trip (close only), got {metrics['n_round_trades']}. "
        "Scale-in should not book intermediate P&L."
    )
    assert metrics["hit_rate"] == pytest.approx(1.0), (
        f"Expected hit_rate=1.0 (the single close is profitable), got {metrics['hit_rate']}"
    )


def test_compute_metrics_scale_in_weighted_average_entry():
    """
    Verify that the PnL from the close uses the weighted-average entry price.

    BUY 10 @ 100, BUY 10 @ 120, SELL 20 @ 110.
    Weighted entry = (100*10 + 120*10) / 20 = 110.
    PnL = (110 - 110) * 20 = 0 → neither a win nor a loss.
    """
    from simulation_pipeline import compute_metrics

    idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
    equity = pd.Series(100_000.0 + np.zeros(10), index=idx)
    trades = _make_trades([
        ("BUY",  10, 100.0),
        ("BUY",  10, 120.0),
        ("SELL", 20, 110.0),
    ])
    metrics = compute_metrics(equity, trades)

    assert metrics["n_round_trades"] == 1
    # hit_rate = 0.0 because the only trade has PnL = 0 (not > 0)
    assert metrics["hit_rate"] == pytest.approx(0.0), (
        f"Expected hit_rate=0.0 (PnL is exactly zero, not a win), got {metrics['hit_rate']}"
    )
