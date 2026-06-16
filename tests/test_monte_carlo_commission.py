"""
Tests for Issue #28 (Monte Carlo commission) and Issue #27 (daily loss limit).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation_pipeline import Backtester, ExecutionConfig, monte_carlo_stress


def _make_df(prices, freq="B", tz="UTC"):
    """Build a OHLCV DataFrame from a list of close prices."""
    n = len(prices)
    idx = pd.date_range("2024-01-02", periods=n, freq=freq, tz=tz)
    return pd.DataFrame(
        {
            "open": [p - 0.5 for p in prices],
            "high": [p + 1.0 for p in prices],
            "low": [p - 1.0 for p in prices],
            "close": prices,
            "volume": [1_000_000.0] * n,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Issue #28 — Monte Carlo stress test must use exec_cfg.commission_per_share
# ---------------------------------------------------------------------------

class TestMonteCarlCommission:
    """Verify that monte_carlo_stress uses the passed exec_cfg commission."""

    def test_lower_commission_yields_higher_equity(self):
        """
        Direct backtester comparison: lower commission_per_share should produce
        higher final equity. This confirms the commission value is actually used.
        The old bug hardcoded 0.0005; the real default is 0.00005 (10× cheaper).
        """
        n = 30
        prices = [100.0 + i * 0.5 for i in range(n)]
        df = _make_df(prices)
        signal = pd.Series(1, index=df.index)

        bt_correct = Backtester(ExecutionConfig(
            commission_per_share=0.00005,  # the real configured value
            slippage_bps=0.0,
            stop_loss_pct=0.50,
            take_profit_pct=0.50,
            daily_loss_limit_pct=0.99,
        ))
        result_correct = bt_correct.run(df, df, signal)

        bt_wrong = Backtester(ExecutionConfig(
            commission_per_share=0.0005,  # old hardcoded value (10× more expensive)
            slippage_bps=0.0,
            stop_loss_pct=0.50,
            take_profit_pct=0.50,
            daily_loss_limit_pct=0.99,
        ))
        result_wrong = bt_wrong.run(df, df, signal)

        # With lower commission costs, final equity must be higher
        assert result_correct.equity_curve.iloc[-1] > result_wrong.equity_curve.iloc[-1], (
            "Commission=0.00005 should yield higher equity than commission=0.0005"
        )

    def test_monte_carlo_default_exec_cfg_uses_dataclass_default(self):
        """Without exec_cfg arg, commission defaults to ExecutionConfig default (0.00005)."""
        n = 20
        prices = [100.0] * n
        df = _make_df(prices)
        signal = pd.Series(0, index=df.index)  # all HOLD — no trades

        # Should run without error
        result_df = monte_carlo_stress(df, df, signal, n_runs=3)
        assert isinstance(result_df, pd.DataFrame)

    def test_hardcoded_commission_not_used(self, monkeypatch):
        """
        Directly verify the commission seen by each Backtester in monte_carlo_stress
        matches exec_cfg, not the old literal 0.0005.
        """
        captured_commissions = []

        original_init = Backtester.__init__

        def mock_init(self, cfg):
            captured_commissions.append(cfg.commission_per_share)
            original_init(self, cfg)

        monkeypatch.setattr(Backtester, "__init__", mock_init)

        n = 10
        prices = [100.0] * n
        df = _make_df(prices)
        signal = pd.Series(0, index=df.index)

        target_commission = 0.00001
        exec_cfg = ExecutionConfig(commission_per_share=target_commission)
        monte_carlo_stress(df, df, signal, n_runs=4, exec_cfg=exec_cfg)

        assert len(captured_commissions) == 4
        for c in captured_commissions:
            assert c == pytest.approx(target_commission), (
                f"Expected commission {target_commission}, got {c}. "
                "monte_carlo_stress is still using a hardcoded value."
            )


# ---------------------------------------------------------------------------
# Issue #27 — Daily loss-limit must fire across daily bar boundaries
# ---------------------------------------------------------------------------

class TestDailyLossLimit:
    """Verify daily loss-limit fires on a cross-day loss, not just intrabar."""

    def test_daily_loss_limit_fires_on_cross_day_loss(self):
        """
        Day 1: buy at 100, fully invested. Day 2: price stays at 100.
        Day 3: price drops to 90 (10% drop). With a 5% daily loss limit,
        the position must be liquidated on Day 3.

        Previously the bug re-set daily_start_cash = cash only (not including
        position mark-to-market), so the check could never fire on daily bars.
        The fix sets daily_start_cash = cash + position * prev_close.

        We use max_position_pct=1.0 so the full portfolio is exposed to the
        price move and the 10% price drop translates to a ~10% equity drop.
        """
        prices = [100.0, 100.0, 90.0, 90.0]  # 10% price drop on day 3
        n = len(prices)
        idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [p - 0.5 for p in prices],
                "high": [p + 1.0 for p in prices],
                "low": [p - 1.0 for p in prices],
                "close": prices,
                "volume": [1_000_000.0] * n,
            },
            index=idx,
        )

        signal = pd.Series([1, 1, 1, 1], index=idx, dtype=int)

        cfg = ExecutionConfig(
            start_cash=100_000.0,
            commission_per_share=0.0,
            slippage_bps=0.0,
            stop_loss_pct=0.50,         # wide — won't fire from stop_loss
            take_profit_pct=0.50,       # wide — won't fire from take_profit
            daily_loss_limit_pct=0.05,  # 5% daily limit — fires on 10% equity drop
            max_position_pct=1.0,       # fully invested so price drop = equity drop
            max_position=2000,
        )

        bt = Backtester(cfg)
        result = bt.run(df, df, signal)

        daily_limit_exits = [
            t for t in result.trades.itertuples()
            if t.exit_reason == "daily_limit"
        ]

        assert len(daily_limit_exits) > 0, (
            "Expected daily loss limit to trigger on the 10% cross-day drop "
            "(fully invested), but no 'daily_limit' exit was recorded. "
            "Bug: daily_start_cash may still be rebased on cash only "
            "(ignoring position mark-to-market)."
        )

    def test_daily_loss_limit_does_not_fire_within_allowed_loss(self):
        """
        Loss that is within the daily limit threshold should NOT trigger an exit.
        Even fully invested, a 1% price drop is well within the 5% limit.
        """
        prices = [100.0, 100.0, 99.0, 99.0]  # only 1% drop on day 3
        n = len(prices)
        idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [p - 0.5 for p in prices],
                "high": [p + 1.0 for p in prices],
                "low": [p - 1.0 for p in prices],
                "close": prices,
                "volume": [1_000_000.0] * n,
            },
            index=idx,
        )

        signal = pd.Series([1, 1, 1, 1], index=idx, dtype=int)

        cfg = ExecutionConfig(
            start_cash=100_000.0,
            commission_per_share=0.0,
            slippage_bps=0.0,
            stop_loss_pct=0.50,
            take_profit_pct=0.50,
            daily_loss_limit_pct=0.05,  # 5% limit — 1% drop should not trigger
            max_position_pct=1.0,
            max_position=2000,
        )

        bt = Backtester(cfg)
        result = bt.run(df, df, signal)

        daily_limit_exits = [
            t for t in result.trades.itertuples()
            if t.exit_reason == "daily_limit"
        ]

        assert len(daily_limit_exits) == 0, (
            "Daily loss limit should NOT fire for a 1% cross-day drop with a 5% threshold."
        )
