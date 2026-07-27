"""tests/test_dqn_scaler_persistence.py — Regression tests for issue #123.

The DQN's feature scaler must be persisted with the trained agent and
applied verbatim at inference, instead of the model being trained on one
z-scored distribution and re-normalized on a different, daily-drifting
window at predict time.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_features import FEATURE_COLS
from dqn_agent import DQNAgent, DQNConfig


# ---------------------------------------------------------------------------
# DQNAgent.save/load round trip
# ---------------------------------------------------------------------------

def _tiny_agent(state_dim=8, action_dim=3):
    cfg = DQNConfig(hidden=4, buffer_size=10, batch_size=2, use_per=False)
    return DQNAgent(state_dim, action_dim, cfg)


def test_save_load_round_trips_scaler_and_feature_contract(tmp_path):
    agent = _tiny_agent()
    scaler = {c: (float(i), float(i + 1)) for i, c in enumerate(FEATURE_COLS)}
    path = str(tmp_path / "dqn_agent.pt")

    agent.save(path, scaler=scaler, feature_contract=FEATURE_COLS)
    loaded = DQNAgent.load(path)

    assert loaded.scaler == scaler
    assert list(loaded.feature_contract) == list(FEATURE_COLS)


def test_load_of_legacy_checkpoint_without_scaler_yields_none(tmp_path):
    """Checkpoints saved before this fix (no scaler/feature_contract keys)
    must load without crashing, exposing scaler=None so callers can detect
    and refuse to serve from them rather than silently mis-normalizing."""
    agent = _tiny_agent()
    path = str(tmp_path / "legacy_dqn_agent.pt")
    agent.save(path)  # no scaler/feature_contract args -> None

    loaded = DQNAgent.load(path)
    assert loaded.scaler is None
    assert loaded.feature_contract is None


# ---------------------------------------------------------------------------
# predict_next_day_lite.predict_symbol must use the persisted scaler
# ---------------------------------------------------------------------------

def _make_ohlcv(n=100):
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, n))
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": np.full(n, 1_000_000.0)},
        index=idx,
    )


def _make_features_df(n=100, feature_value=3.0):
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    data = {c: np.full(n, feature_value) for c in FEATURE_COLS}
    data["close"] = np.linspace(100, 110, n)
    data["fwd_ret_1d"] = np.zeros(n)
    return pd.DataFrame(data, index=idx)


def test_predict_symbol_raises_clear_error_when_scaler_missing():
    from predict_next_day_lite import predict_symbol

    agent = MagicMock()
    agent.q = MagicMock(return_value=__import__("torch").tensor([[0.0, 0.0, 0.0]]))
    agent.scaler = None
    agent.feature_contract = None
    models = {"daily_dqn": agent}
    feats_df = _make_features_df()

    with patch("predict_next_day_lite.fetch_bars_with_fallback", return_value=_make_ohlcv()), \
         patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
        result = predict_symbol("AAPL", models)

    pred = result["predictions"]["daily_dqn"]
    assert "error" in pred
    assert "scaler" in pred["error"].lower()


def test_predict_symbol_normalizes_with_persisted_scaler_not_rederived_stats():
    """Feed a feature frame whose warmup-window mean/std differ sharply from
    the persisted scaler's (mu, sd). The Q-network must see the state
    normalized with the PERSISTED scaler — capture the tensor passed to
    agent.q and check it against the persisted-scaler-normalized value,
    not the value a warmup-window re-derivation would produce."""
    import torch
    from predict_next_day_lite import predict_symbol

    n = 100
    feature_value = 3.0
    feats_df = _make_features_df(n=n, feature_value=feature_value)

    captured = {}

    def fake_q(state_tensor):
        captured["state"] = state_tensor.clone()
        return torch.tensor([[0.0, 5.0, 0.0]])  # clear BUY

    agent = MagicMock()
    agent.q = fake_q
    # Persisted scaler deliberately different from "re-derive from X_all"
    # (which would give mu=feature_value, sd=0 -> sd clamped to 1, so
    # (value-mu)/sd == 0 for every feature). Use a scaler with mu=0, sd=2
    # so the persisted-scaler-normalized value is feature_value/2 != 0.
    agent.scaler = {c: (0.0, 2.0) for c in FEATURE_COLS}
    agent.feature_contract = list(FEATURE_COLS)
    models = {"daily_dqn": agent}

    with patch("predict_next_day_lite.fetch_bars_with_fallback", return_value=_make_ohlcv(n)), \
         patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
        result = predict_symbol("AAPL", models)

    assert "error" not in result["predictions"]["daily_dqn"]
    state = captured["state"].numpy().flatten()
    expected = feature_value / 2.0
    assert np.allclose(state, expected), (
        f"Expected every state entry == {expected} (persisted-scaler "
        f"normalization), got distinct values {np.unique(state)} — DQN is "
        "not using the persisted scaler verbatim."
    )


def test_predict_symbol_raises_on_feature_contract_mismatch():
    from predict_next_day_lite import predict_symbol

    agent = MagicMock()
    agent.q = MagicMock()
    agent.scaler = {c: (0.0, 1.0) for c in FEATURE_COLS}
    agent.feature_contract = ["some_other_feature"]  # mismatched contract
    models = {"daily_dqn": agent}
    feats_df = _make_features_df()

    with patch("predict_next_day_lite.fetch_bars_with_fallback", return_value=_make_ohlcv()), \
         patch("predict_next_day_lite.make_daily_features", return_value=feats_df):
        result = predict_symbol("AAPL", models)

    pred = result["predictions"]["daily_dqn"]
    assert "error" in pred
    assert "contract" in pred["error"].lower()


# ---------------------------------------------------------------------------
# TradingEnv must feed real SPY-relative features to make_daily_features
# ---------------------------------------------------------------------------

def test_trading_env_passes_spy_df_to_make_daily_features():
    """Training must not silently zero out ret_*_vs_spy while live prediction
    uses real values — make_daily_features must be called with a non-None
    spy_df for a non-SPY symbol."""
    from daily_features import FEATURE_COLS as FC

    n = 250
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    raw = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
         "volume": 1_000_000.0}, index=idx,
    )
    spy_raw = raw.copy()
    feats = pd.DataFrame(
        {**{c: np.zeros(n) for c in FC}, "close": np.full(n, 100.0),
         "fwd_ret_1d": np.zeros(n)},
        index=idx,
    )

    def fake_load_yfinance(symbol, start=None, end=None, interval=None):
        return spy_raw if symbol == "SPY" else raw

    with patch("rl_env.load_yfinance", side_effect=fake_load_yfinance), \
         patch("rl_env.make_daily_features", return_value=feats) as mock_feats:
        from rl_env import TradingEnv
        TradingEnv(symbol="AAPL", window=5)

    _, kwargs = mock_feats.call_args
    passed_spy = kwargs.get("spy_df", None) if kwargs else mock_feats.call_args[0][1] if len(mock_feats.call_args[0]) > 1 else None
    assert passed_spy is not None, (
        "TradingEnv must pass a real spy_df into make_daily_features for a "
        "non-SPY symbol, otherwise ret_*_vs_spy train at a constant 0.0 "
        "while inference sees real values (issue #123)."
    )


# ---------------------------------------------------------------------------
# train_models.train_dqn must fit and persist ONE scaler shared across all
# training symbols, not let each env keep its own private one.
# ---------------------------------------------------------------------------

def test_train_dqn_fits_one_shared_scaler_and_persists_it(tmp_path):
    """AAPL and MSFT have deliberately different raw feature distributions
    (offset +100 vs -50). A per-symbol-fit scaler would recover mu~=+100 for
    AAPL and mu~=-50 for MSFT; the shared scaler train_dqn fits must average
    across both symbols' warmup windows and be identical for every env, and
    that exact scaler must be the one written to the saved artifact."""
    import torch
    import train_models
    from config import DQNCfg

    n = 60

    class _FakeEnv:
        window = 5
        features = list(FEATURE_COLS)

        def __init__(self, symbol, start=None, end=None, window=20,
                     transaction_cost_bps=10.0, feature_scaler=None, spy_df=None):
            self.symbol = symbol
            self.window = window
            offset = 100.0 if symbol == "AAPL" else -50.0
            idx = pd.date_range("2023-01-01", periods=n, freq="B")
            df = pd.DataFrame({c: np.full(n, offset) for c in FEATURE_COLS}, index=idx)
            df["close"] = 100.0 + np.arange(n) * 0.1
            df["fwd_ret_1d"] = 0.001
            self.scaler = feature_scaler
            if feature_scaler is not None:
                for c in self.features:
                    mu, sd = feature_scaler[c]
                    df[c] = (df[c] - mu) / (sd if sd else 1.0)
            self.df = df
            self.prices = df["close"].values.astype(float)
            self.returns = df["fwd_ret_1d"].values.astype(float)
            self.idx = self.window
            self.position = 0
            self.equity = 0.0

        def get_data_info(self):
            return {
                "symbol": self.symbol, "source": "fake", "num_bars": len(self.df),
                "date_range": "n/a", "features": len(self.features),
                "feature_list": self.features,
            }

        @property
        def observation_space_shape(self):
            return (self.window * len(self.features),)

        @property
        def action_space_n(self):
            return 3

        def _get_state(self):
            frame = self.df.iloc[self.idx - self.window: self.idx]
            return frame[self.features].values.astype(np.float32).flatten()

        def reset(self):
            self.idx = self.window
            self.position = 0
            self.equity = 0.0
            return self._get_state()

        def step(self, action):
            self.idx += 1
            done = self.idx >= len(self.df)
            next_state = (
                np.zeros(self.observation_space_shape, dtype=np.float32)
                if done else self._get_state()
            )
            return next_state, 0.0, done, {"pnl": 0.0, "equity": 0.0, "price": 100.0}

    symbols = ["AAPL", "MSFT"]
    out_path = str(tmp_path / "dqn_agent.pt")
    cfg_dqn = DQNCfg(
        window=5, hidden=4, episodes=1, steps_per_episode=2,
        buffer_size=50, batch_size=2, target_update_interval=1000,
        epsilon_decay_steps=10,
    )

    with patch.object(train_models, "TradingEnv", _FakeEnv), \
         patch.object(train_models, "load_yfinance", return_value=pd.DataFrame(
             {"close": [1.0]}, index=pd.date_range("2023-01-01", periods=1))):
        train_models.train_dqn(symbols, "2023-01-01", "2023-06-01", cfg_dqn, out_path)

    blob = torch.load(out_path, map_location="cpu", weights_only=True)
    scaler = blob["scaler"]
    assert scaler is not None
    assert list(blob["feature_contract"]) == list(FEATURE_COLS)

    # Shared scaler's mean must sit between the two symbols' raw offsets
    # (100 and -50) — a per-symbol scaler would instead read exactly one
    # of those two values.
    any_feature = FEATURE_COLS[0]
    mu, _sd = scaler[any_feature]
    assert -50.0 < mu < 100.0, (
        f"Expected shared scaler mean between the two symbols' raw offsets, "
        f"got {mu} — looks like a per-symbol scaler was persisted instead."
    )
