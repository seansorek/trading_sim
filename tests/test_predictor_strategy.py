"""Integration tests for PredictorStrategy — no file I/O, mocked predictor/decision."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_strategy import StrategyConfig
from daily_features import FEATURE_COLS
from predictor_strategy import PredictorStrategy


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(500_000, 10_000_000, n).astype(float),
        },
        index=idx,
    )


def _mock_predictor(scores: np.ndarray, proba=None):
    """Create a predictor mock that returns fixed (scores, proba).

    If the caller passes a full-length array (size=len(df)), we trim/zero-pad
    to match the actual X shape so the index alignment in PredictorStrategy works.
    """
    fixed_scores = scores
    fixed_proba = proba

    def _predict(X):
        m = X.shape[0]
        s = fixed_scores[:m] if len(fixed_scores) >= m else np.zeros(m)
        p = None
        if fixed_proba is not None:
            p = fixed_proba[:m] if len(fixed_proba) >= m else np.ones((m, 3)) / 3
        return (s, p)

    pred = MagicMock()
    pred.predict.side_effect = _predict
    return pred


def _mock_decision(signals: np.ndarray):
    """Create a decision mock that returns fixed signals trimmed to match scores length."""
    fixed_signals = signals

    def _decide(scores, proba, ctx):
        m = len(scores)
        return fixed_signals[:m] if len(fixed_signals) >= m else np.zeros(m, dtype=int)

    dec = MagicMock()
    dec.decide.side_effect = _decide
    return dec


def _cfg():
    return StrategyConfig(name="test", holding_period=0)


class TestPredictorStrategy:
    def test_signal_returns_series_with_df_index(self):
        df = _make_ohlcv(50)
        n = len(df)
        pred = _mock_predictor(np.zeros(n))
        dec = _mock_decision(np.zeros(n, dtype=int))
        strat = PredictorStrategy(_cfg(), pred, dec)
        sig = strat.signal(pd.DataFrame(), df)
        assert isinstance(sig, pd.Series)

    def test_signal_values_are_in_minus1_0_1(self):
        df = _make_ohlcv(50)
        n = len(df)
        signals = np.array([1, -1, 0] * (n // 3) + [0] * (n % 3))
        pred = _mock_predictor(signals.astype(float))
        dec = _mock_decision(signals)
        strat = PredictorStrategy(_cfg(), pred, dec)
        sig = strat.signal(pd.DataFrame(), df)
        assert set(sig.dropna().unique()).issubset({-1, 0, 1})

    def test_signal_applies_one_bar_execution_lag(self):
        """Signal at bar i should appear at bar i+1 in the output."""
        df = _make_ohlcv(100)
        n = len(df)
        # Decision emits BUY (1) at bar index 5 (relative to daily_feats), HOLD elsewhere
        raw = np.zeros(n, dtype=int)
        raw[5] = 1
        pred = _mock_predictor(raw.astype(float))
        dec = _mock_decision(raw)
        strat = PredictorStrategy(_cfg(), pred, dec)
        sig = strat.signal(pd.DataFrame(), df)
        # The 1 at position 5 must appear at position 6 due to shift(1)
        assert sig.iloc[6] == 1
        assert sig.iloc[5] == 0

    def test_predictor_receives_correct_feature_shape(self):
        df = _make_ohlcv(100)
        n = len(df)
        pred = _mock_predictor(np.zeros(n))
        dec = _mock_decision(np.zeros(n, dtype=int))
        strat = PredictorStrategy(_cfg(), pred, dec)
        strat.signal(pd.DataFrame(), df)
        call_args = pred.predict.call_args
        X = call_args[0][0]
        assert X.shape[1] == len(FEATURE_COLS)
        assert X.dtype == np.float32

    def test_decision_receives_scores_and_proba(self):
        df = _make_ohlcv(100)
        n = len(df)
        # We need proba sized to the actual daily_feats length; use a dynamic mock
        captured_proba = {}

        def _predict(X):
            m = X.shape[0]
            p = np.ones((m, 3)) / 3
            captured_proba["value"] = p
            return (np.zeros(m), p)

        def _decide(scores, proba, ctx):
            return np.zeros(len(scores), dtype=int)

        pred = MagicMock()
        pred.predict.side_effect = _predict
        dec = MagicMock()
        dec.decide.side_effect = _decide

        strat = PredictorStrategy(_cfg(), pred, dec)
        strat.signal(pd.DataFrame(), df)

        call_kwargs = dec.decide.call_args
        _, proba_arg, _ = call_kwargs[0]
        np.testing.assert_array_equal(proba_arg, captured_proba["value"])

    def test_holding_period_is_enforced(self):
        df = _make_ohlcv(100)
        n = len(df)
        # BUY signals at bars 0, 1, 2 — with holding_period=5, only first should fire
        raw = np.array([1, 1, 1] + [0] * (n - 3), dtype=int)
        pred = _mock_predictor(raw.astype(float))
        dec = _mock_decision(raw)
        cfg = StrategyConfig(name="test", holding_period=5)
        strat = PredictorStrategy(cfg, pred, dec)
        sig = strat.signal(pd.DataFrame(), df)
        # After shift(1): only bar 1 should be 1; bars 2 and 3 should be 0
        assert sig.iloc[1] == 1
        assert sig.iloc[2] == 0
        assert sig.iloc[3] == 0
