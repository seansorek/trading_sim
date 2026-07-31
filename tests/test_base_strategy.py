"""test_base_strategy.py — Tests for BaseStrategy._apply_holding_period (Issue #34)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_strategy import BaseStrategy, StrategyConfig


class _DummyStrategy(BaseStrategy):
    """Concrete subclass so we can instantiate BaseStrategy."""

    def signal(self, feats, df):
        return pd.Series(0, index=feats.index)


class TestApplyHoldingPeriod:
    def test_persistent_signal_is_held_not_chopped(self):
        """
        A continuously-BUY signal is one position held throughout, not a trade
        every `holding_period` bars with flat gaps between.

        This is the regression test for the inverted rule: the output is a
        target position, so zeroing a suppressed bar told the engine to go
        flat, capping every hold at one bar.
        """
        cfg = StrategyConfig(name="test", holding_period=5)
        strat = _DummyStrategy(cfg)

        idx = pd.date_range("2024-01-02", periods=30, freq="B")
        raw = pd.Series(1, index=idx)

        filtered = strat._apply_holding_period(raw)
        assert (filtered == 1).all(), (
            "A persistent BUY must stay in position; any 0 here means the "
            "backtester would be told to flatten and re-enter."
        )

    def test_position_is_held_for_exactly_holding_period_after_signal_ends(self):
        """One BUY bar then silence: hold 5 bars (0-4), flat from bar 5."""
        cfg = StrategyConfig(name="test", holding_period=5)
        strat = _DummyStrategy(cfg)

        idx = pd.date_range("2024-01-02", periods=10, freq="B")
        raw = pd.Series(0, index=idx)
        raw.iloc[0] = 1

        filtered = strat._apply_holding_period(raw)
        assert list(filtered.iloc[:5]) == [1, 1, 1, 1, 1]
        assert list(filtered.iloc[5:]) == [0] * 5

    def test_reversal_exactly_at_holding_period_is_allowed(self):
        """BUY at bar 0, SELL at bar 5 — the reversal lands on the first free bar."""
        cfg = StrategyConfig(name="test", holding_period=5)
        strat = _DummyStrategy(cfg)

        idx = pd.date_range("2024-01-02", periods=10, freq="B")
        raw = pd.Series(0, index=idx)
        raw.iloc[0] = 1
        raw.iloc[5] = -1  # exactly holding_period bars later

        filtered = strat._apply_holding_period(raw)
        assert (filtered.iloc[:5] == 1).all(), "Long position held bars 0-4"
        assert filtered.iloc[5] == -1, "Reversal accepted at exactly holding_period"
        assert (filtered.iloc[5:] == -1).all(), "Short then held its own 5 bars"

    def test_reversal_one_bar_early_is_suppressed_and_position_carried(self):
        """A SELL at bar 4 is too early: the long is carried, not flattened."""
        cfg = StrategyConfig(name="test", holding_period=5)
        strat = _DummyStrategy(cfg)

        idx = pd.date_range("2024-01-02", periods=10, freq="B")
        raw = pd.Series(0, index=idx)
        raw.iloc[0] = 1
        raw.iloc[4] = -1  # one bar too early

        filtered = strat._apply_holding_period(raw)
        assert filtered.iloc[4] == 1, (
            "Suppressed reversal must carry the existing long forward — "
            "emitting 0 would execute as an exit."
        )
        assert filtered.iloc[5] == 0, "Bar 5 is free and the raw signal is flat"

    def test_holding_period_zero_passes_all_signals(self):
        cfg = StrategyConfig(name="test", holding_period=0)
        strat = _DummyStrategy(cfg)

        idx = pd.date_range("2024-01-02", periods=5, freq="B")
        raw = pd.Series([1, -1, 1, -1, 1], index=idx)

        filtered = strat._apply_holding_period(raw)
        pd.testing.assert_series_equal(filtered, raw)

    def test_all_zero_signals_returns_all_zero(self):
        cfg = StrategyConfig(name="test", holding_period=5)
        strat = _DummyStrategy(cfg)

        idx = pd.date_range("2024-01-02", periods=10, freq="B")
        raw = pd.Series(0, index=idx)

        filtered = strat._apply_holding_period(raw)
        assert (filtered == 0).all()
