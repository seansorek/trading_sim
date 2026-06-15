"""test_rl_env.py — Tests for TradingEnv (Issue #24: DQN state dimension fix)."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_features import FEATURE_COLS


def _make_raw_ohlcv(n: int = 250, start_price: float = 100.0) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame large enough for TradingEnv."""
    idx = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    prices = start_price + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n))
    prices = np.maximum(prices, 1.0)
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.full(n, 1_000_000.0),
            "ticker": "TEST",
        },
        index=idx,
    )


def _make_feats_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a feature DataFrame that includes all FEATURE_COLS plus extras."""
    from daily_features import make_daily_features

    # make_daily_features may require SPY data; patch it to return a minimal frame
    n = len(raw)
    idx = raw.index
    data = {col: np.random.default_rng(0).normal(0, 1, n) for col in FEATURE_COLS}
    data["close"] = raw["close"].values
    data["fwd_ret_1d"] = np.random.default_rng(1).normal(0, 0.01, n)
    data["ticker"] = "TEST"
    data["date"] = idx.date
    return pd.DataFrame(data, index=idx)


class TestTradingEnvFeatureDimension:
    """Issue #24: TradingEnv.features must equal FEATURE_COLS (25 cols), not 26."""

    def _build_env(self, window: int = 5):
        """Build a TradingEnv with mocked data, bypassing yfinance."""
        raw = _make_raw_ohlcv(n=250)
        feats = _make_feats_df(raw)

        with patch("rl_env.load_yfinance", return_value=raw), \
             patch("rl_env.make_daily_features", return_value=feats):
            from rl_env import TradingEnv
            env = TradingEnv(symbol="TEST", window=window)
        return env

    def test_features_equals_FEATURE_COLS(self):
        """TradingEnv.features must be exactly FEATURE_COLS (not column-exclusion list)."""
        env = self._build_env()
        assert env.features == FEATURE_COLS, (
            f"Expected features == FEATURE_COLS ({len(FEATURE_COLS)} cols), "
            f"got {len(env.features)} cols: {env.features}"
        )

    def test_observation_space_matches_feature_cols(self):
        """observation_space_shape must be (window * len(FEATURE_COLS),)."""
        window = 5
        env = self._build_env(window=window)
        expected = (window * len(FEATURE_COLS),)
        assert env.observation_space_shape == expected, (
            f"Expected observation_space_shape={expected}, "
            f"got {env.observation_space_shape}"
        )

    def test_observation_shape_at_reset(self):
        """State returned by reset() must have shape (window * len(FEATURE_COLS),)."""
        window = 5
        env = self._build_env(window=window)
        obs = env.reset()
        assert obs.shape == (window * len(FEATURE_COLS),), (
            f"Expected obs shape ({window * len(FEATURE_COLS)},), got {obs.shape}"
        )

    def test_close_not_in_features(self):
        """'close' must NOT be in env.features (it was before the fix)."""
        env = self._build_env()
        assert "close" not in env.features, (
            "close should not be in env.features — it was included before the fix "
            "causing a 26-column observation vs 25-column FEATURE_COLS mismatch."
        )
