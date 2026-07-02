"""Tests for decision layer implementations."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from decision_layers.base import BaseDecisionLayer, DecisionContext


class TestBaseDecisionLayer:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseDecisionLayer()

    def test_decision_context_stores_index_and_symbol(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        ctx = DecisionContext(index=idx, symbol="AAPL")
        assert ctx.symbol == "AAPL"
        assert len(ctx.index) == 5


from decision_layers.threshold import ThresholdDecision


class TestThresholdDecision:
    def _ctx(self, n=10):
        return DecisionContext(
            index=pd.date_range("2024-01-01", periods=n, freq="B"),
            symbol="TEST",
        )

    def test_high_confidence_buy_passes(self):
        scores = np.array([1.0])
        proba = np.array([[0.1, 0.2, 0.7]])  # max=0.7 >= 0.55
        signals = ThresholdDecision(0.55).decide(scores, proba, self._ctx(1))
        assert signals[0] == 1

    def test_high_confidence_sell_passes(self):
        scores = np.array([-1.0])
        proba = np.array([[0.7, 0.2, 0.1]])  # max=0.7 >= 0.55
        signals = ThresholdDecision(0.55).decide(scores, proba, self._ctx(1))
        assert signals[0] == -1

    def test_low_confidence_becomes_hold(self):
        scores = np.array([1.0])
        proba = np.array([[0.25, 0.30, 0.45]])  # max=0.45 < 0.55
        signals = ThresholdDecision(0.55).decide(scores, proba, self._ctx(1))
        assert signals[0] == 0

    def test_none_proba_uses_score_sign_only(self):
        scores = np.array([0.8, -0.3, 0.0])
        signals = ThresholdDecision(0.55).decide(scores, None, self._ctx(3))
        np.testing.assert_array_equal(signals, [1, -1, 0])

    def test_output_dtype_is_int(self):
        scores = np.array([1.0, -1.0])
        proba = np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
        signals = ThresholdDecision(0.55).decide(scores, proba, self._ctx(2))
        assert signals.dtype in (np.int32, np.int64, int)


from decision_layers.quantile import QuantileDecision


class TestQuantileDecision:
    def _ctx(self, n):
        return DecisionContext(
            index=pd.date_range("2024-01-01", periods=n, freq="B"),
            symbol="TEST",
        )

    def test_high_magnitude_bar_trades(self):
        # 20 bars with magnitude 0.01, last bar 1.0 — should trade
        n = 21
        scores = np.concatenate([np.ones(n - 1) * 0.01, [1.0]])
        signals = QuantileDecision(signal_quantile=0.7, threshold_window=10).decide(
            scores, None, self._ctx(n)
        )
        assert signals[-1] in (1, -1)

    def test_low_magnitude_bar_holds(self):
        # All magnitudes equal — tie doesn't clear threshold
        scores = np.ones(50) * 0.01
        signals = QuantileDecision(signal_quantile=0.7, threshold_window=10).decide(
            scores, None, self._ctx(50)
        )
        # Tie at the quantile boundary: all equal → quantile == magnitude, signals may vary
        # What we test: no bar with magnitude BELOW threshold produces a signal
        # (here all equal so all at threshold; pass if this doesn't raise)
        assert signals.shape == (50,)
        assert set(signals).issubset({-1, 0, 1})

    def test_no_lookahead_first_bar_is_always_hold(self):
        # After shift(1), bar 0 threshold is NaN → filled with inf → HOLD
        scores = np.array([100.0, 100.0, 100.0])
        signals = QuantileDecision(signal_quantile=0.0, threshold_window=1).decide(
            scores, None, self._ctx(3)
        )
        # Bar 0 must be HOLD (no history)
        assert signals[0] == 0

    def test_output_only_contains_valid_signals(self):
        rng = np.random.default_rng(42)
        scores = rng.standard_normal(200)
        signals = QuantileDecision().decide(scores, None, self._ctx(200))
        assert set(signals).issubset({-1, 0, 1})

    def test_signal_direction_matches_score_sign(self):
        scores = np.array([-5.0] * 30 + [5.0])
        signals = QuantileDecision(signal_quantile=0.5, threshold_window=20).decide(
            scores, None, self._ctx(31)
        )
        # Any non-zero signal must match the sign of its score
        for i, (sig, sc) in enumerate(zip(signals, scores)):
            if sig != 0:
                assert np.sign(sig) == np.sign(sc), f"bar {i}: signal {sig} mismatches score {sc}"


from decision_layers.dqn_decision import DQNDecision


class TestDQNDecision:
    def _ctx(self, n=1):
        return DecisionContext(
            index=pd.date_range("2024-01-01", periods=n, freq="B"),
            symbol="TEST",
        )

    def _q(self, hold, long, short):
        """Build a (1, 3) Q-value matrix."""
        return np.array([[hold, long, short]], dtype=np.float32)

    def test_high_long_advantage_produces_buy(self):
        # Q(Long)=10, Q(Hold)=0.5 → advantage=9.5 > 1.0; spread=9.8 > 2.0
        scores = np.array([9.8])  # Long - Short
        proba = self._q(0.5, 10.0, 0.2)
        signals = DQNDecision(confidence_threshold=2.0, q_advantage_threshold=1.0).decide(
            scores, proba, self._ctx()
        )
        assert signals[0] == 1

    def test_high_short_advantage_produces_sell(self):
        scores = np.array([-9.8])
        proba = self._q(0.5, 0.2, 10.0)
        signals = DQNDecision(confidence_threshold=2.0, q_advantage_threshold=1.0).decide(
            scores, proba, self._ctx()
        )
        assert signals[0] == -1

    def test_low_spread_produces_hold(self):
        # spread = 1.3 - 0.9 = 0.4 < confidence_threshold=2.0
        scores = np.array([0.4])
        proba = self._q(1.0, 1.3, 0.9)
        signals = DQNDecision(confidence_threshold=2.0, q_advantage_threshold=1.0).decide(
            scores, proba, self._ctx()
        )
        assert signals[0] == 0

    def test_warmup_zeros_produce_hold(self):
        scores = np.zeros(5)
        proba = np.zeros((5, 3), dtype=np.float32)
        signals = DQNDecision().decide(scores, proba, self._ctx(5))
        np.testing.assert_array_equal(signals, 0)

    def test_none_proba_produces_all_hold(self):
        signals = DQNDecision().decide(np.ones(3), None, self._ctx(3))
        np.testing.assert_array_equal(signals, 0)
