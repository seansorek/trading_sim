"""test_rl_env.py — Tests for TradingEnv (Issue #24: DQN state dimension fix)."""
import sys
from pathlib import Path
from unittest.mock import patch

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


class TestTradingEnvScalerNoLookahead:
    """Issue #45: scaler must be fit on a warmup window only, not the full df.

    If the scaler were fit over the entire DataFrame, perturbing the tail of
    the feature series would change the (mu, sd) used to normalise rows at the
    head — leaking future statistics into past observations.
    """

    def _build_env_with_feats(self, feats, window=5):
        raw = _make_raw_ohlcv(n=len(feats))
        # Align raw index to feats so make_daily_features patch is consistent.
        raw = raw.iloc[: len(feats)]
        with patch("rl_env.load_yfinance", return_value=raw), \
             patch("rl_env.make_daily_features", return_value=feats):
            from rl_env import TradingEnv
            env = TradingEnv(symbol="TEST", window=window)
        return env

    def test_scaler_independent_of_future_rows(self):
        """Mutating the tail of a feature must not change the scaler's (mu, sd)."""
        raw = _make_raw_ohlcv(n=600)
        feats_a = _make_feats_df(raw)
        feats_b = feats_a.copy()
        # Inject a huge spike in the back half of every feature column.
        tail_start = len(feats_b) // 2 + 50
        for c in FEATURE_COLS:
            feats_b.iloc[tail_start:, feats_b.columns.get_loc(c)] += 1e6

        env_a = self._build_env_with_feats(feats_a)
        env_b = self._build_env_with_feats(feats_b)

        for c in FEATURE_COLS:
            mu_a, sd_a = env_a.scaler[c]
            mu_b, sd_b = env_b.scaler[c]
            assert mu_a == pytest.approx(mu_b, rel=1e-6, abs=1e-6), (
                f"Scaler mean for {c} leaked future stats: "
                f"{mu_a} vs {mu_b} when only tail rows changed"
            )
            assert sd_a == pytest.approx(sd_b, rel=1e-6, abs=1e-6), (
                f"Scaler std for {c} leaked future stats: "
                f"{sd_a} vs {sd_b} when only tail rows changed"
            )


class TestTradingEnvRewardTransitions:
    """Issue #113: reward must use the real one-day return, and reversal
    penalties must key off the position immediately before the action."""

    def _build_env(self, n=30, window=5, transaction_cost_bps=0.0, daily_ret=0.01):
        raw = _make_raw_ohlcv(n=n, start_price=100.0)
        # Deterministic close path with a known, constant daily return.
        closes = 100.0 * (1.0 + daily_ret) ** np.arange(n)
        raw["close"] = closes
        raw["open"] = closes
        raw["high"] = closes * 1.001
        raw["low"] = closes * 0.999

        feats = _make_feats_df(raw)
        feats["close"] = closes
        # fwd_ret_1d deliberately set far from the real 1-day return, so a
        # test that still reads it (bug) would produce a very different pnl.
        feats["fwd_ret_1d"] = 5.0

        with patch("rl_env.load_yfinance", return_value=raw), \
             patch("rl_env.make_daily_features", return_value=feats):
            from rl_env import TradingEnv
            env = TradingEnv(
                symbol="TEST", window=window, transaction_cost_bps=transaction_cost_bps
            )
        return env, closes

    def test_reward_uses_actual_one_day_return(self):
        env, closes = self._build_env(daily_ret=0.01)
        env.reset()
        i0 = env.idx  # == window
        _, reward, _, info = env.step(1)  # go long

        expected_ret = closes[i0] / closes[i0 - 1] - 1.0
        expected_pnl = 1 * expected_ret * closes[i0 - 1]
        assert info["pnl"] == pytest.approx(expected_pnl, rel=1e-9)
        assert reward == pytest.approx(expected_pnl / (closes[0] * 0.005), rel=1e-9)

    def test_reversal_penalty_applies_at_reversal_not_next_step(self):
        env, closes = self._build_env(daily_ret=0.0)  # flat prices -> pnl-neutral rewards
        env.reset()

        def raw_reward(target_pos, price_prev):
            ret = 0.0  # flat prices
            pnl = target_pos * ret * price_prev
            return pnl / (closes[0] * 0.005)

        # hold -> long -> short (reversal here) -> short (unchanged)
        _, r0, _, _ = env.step(0)
        _, r1, _, _ = env.step(1)
        _, r2, _, _ = env.step(2)
        _, r3, _, _ = env.step(2)

        assert r0 == pytest.approx(raw_reward(0, closes[env.window - 1]), abs=1e-9)
        assert r1 == pytest.approx(raw_reward(1, closes[env.window]), abs=1e-9)
        # Reversal happens on this step (long -> short): penalty applied here.
        assert r2 == pytest.approx(raw_reward(-1, closes[env.window + 1]) - 0.02, abs=1e-9)
        # Unchanged action (short -> short): no penalty.
        assert r3 == pytest.approx(raw_reward(-1, closes[env.window + 2]), abs=1e-9)

    def test_exit_to_flat_not_penalized(self):
        env, closes = self._build_env(daily_ret=0.0)  # flat prices -> pnl-neutral rewards
        env.reset()

        def raw_reward(target_pos, price_prev):
            ret = 0.0  # flat prices
            pnl = target_pos * ret * price_prev
            return pnl / (closes[0] * 0.005)

        # long -> flat (ordinary exit, not a direction reversal): no penalty.
        env.step(1)
        _, r1, _, _ = env.step(0)
        assert r1 == pytest.approx(raw_reward(0, closes[env.window]), abs=1e-9)
