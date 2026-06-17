"""test_base_strategy.py — Tests for BaseStrategy._apply_holding_period (Issue #34)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_strategy import BaseStrategy, StrategyConfig


class _DummyStrategy(BaseStrategy):
    """Concrete subclass so we can instantiate BaseStrategy."""

    def signal(self, feats, df):
        return pd.Series(0, index=feats.index)


class TestApplyHoldingPeriod:
    def test_spacing_is_exactly_holding_period(self):
        """
        With holding_period=5, a trade at bar 0 should allow
        the next trade at bar 5, not bar 6 (off-by-one bug).
        """
        cfg = StrategyConfig(name="test", holding_period=5)
        strat = _DummyStrategy(cfg)

        n = 30
        idx = pd.date_range("2024-01-02", periods=n, freq="B")
        # Signal at every bar — only some should survive
        raw = pd.Series(1, index=idx)

        filtered = strat._apply_holding_period(raw)

        # Collect indices of executed trades
        trade_locs = [i for i in range(n) if filtered.iloc[i] != 0]

        # There should be trades, and consecutive trades should be
        # spaced exactly holding_period bars apart.
        assert len(trade_locs) >= 2, "Expected at least 2 trades"
        for i in range(1, len(trade_locs)):
            spacing = trade_locs[i] - trade_locs[i - 1]
            assert spacing == 5, (
                f"Expected spacing of 5 between trade at bar {trade_locs[i - 1]} "
                f"and bar {trade_locs[i]}, got {spacing}"
            )

    def test_first_trade_at_bar_zero_is_accepted(self):
        cfg = StrategyConfig(name="test", holding_period=5)
        strat = _DummyStrategy(cfg)

        idx = pd.date_range("2024-01-02", periods=10, freq="B")
        raw = pd.Series(0, index=idx)
        raw.iloc[0] = 1

        filtered = strat._apply_holding_period(raw)
        assert filtered.iloc[0] == 1

    def test_trade_exactly_at_holding_period_is_allowed(self):
        """A trade at bar 0 and another at bar 5 should both execute (holding_period=5)."""
        cfg = StrategyConfig(name="test", holding_period=5)
        strat = _DummyStrategy(cfg)

        idx = pd.date_range("2024-01-02", periods=10, freq="B")
        raw = pd.Series(0, index=idx)
        raw.iloc[0] = 1
        raw.iloc[5] = -1  # exactly holding_period bars later

        filtered = strat._apply_holding_period(raw)
        assert filtered.iloc[0] == 1, "Bar 0 trade should be accepted"
        assert filtered.iloc[5] == -1, "Bar 5 trade should be accepted (exactly holding_period)"

    def test_trade_one_bar_before_holding_period_is_suppressed(self):
        """A trade at bar 0 and another at bar 4 should suppress bar 4 (holding_period=5)."""
        cfg = StrategyConfig(name="test", holding_period=5)
        strat = _DummyStrategy(cfg)

        idx = pd.date_range("2024-01-02", periods=10, freq="B")
        raw = pd.Series(0, index=idx)
        raw.iloc[0] = 1
        raw.iloc[4] = -1  # one bar too early

        filtered = strat._apply_holding_period(raw)
        assert filtered.iloc[0] == 1
        assert filtered.iloc[4] == 0, "Bar 4 should be suppressed (within holding period)"

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
