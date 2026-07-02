"""test_predictor.py — Tests for the regression prediction model (train_predictor.py)
and its decoupled decision layer (ml_strategies.DailyPredictorStrategy).

The honesty bar here is different from the classifier tests: there is no
"floor accuracy" to clear, because accuracy isn't the right metric for a
forecast. Instead test_predictor_artifact_contract pins a floor on Spearman
IC (rank correlation between predicted and actual forward return) — IC > 0
means the model carries *some* detectable signal, which is the bar that
matters for a forecasting model, not a specific hit-rate.
"""
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_features import FWD_RET_HORIZON_DAYS, make_daily_features
import train_predictor
from base_strategy import StrategyConfig
from ml_strategies import DailyPredictorStrategy, compute_predictor_signal


def _make_price_df(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close * rng.uniform(0.99, 1.005, n),
            "high": close * rng.uniform(1.001, 1.02, n),
            "low": close * rng.uniform(0.98, 0.999, n),
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


def test_prepare_data_has_embargo_gap(tmp_path):
    """train_predictor.prepare_data must use the same purged split as
    train_models._prepare_data — a continuous target doesn't change the
    leakage math, fwd_ret_1d still looks FWD_RET_HORIZON_DAYS bars ahead."""
    db = train_predictor.DB(str(tmp_path / "test.db"))
    df = _make_price_df()

    with patch("train_predictor._load_symbol", return_value=df):
        data = train_predictor.prepare_data(["AAPL"], days=900, db=db)

    feats = make_daily_features(df, spy_df=None).dropna(subset=["fwd_ret_1d"])
    split = int(len(feats) * 0.8)
    expected_test_len = len(feats) - (split + FWD_RET_HORIZON_DAYS)

    assert data["used_symbols"] == ["AAPL"]
    assert len(data["X_train"]) == split
    assert len(data["X_test"]) == expected_test_len


def test_forecast_metrics_constant_prediction_has_zero_ic():
    """A model that collapses to a constant predictor (e.g. an over-regularized
    XGBRegressor on a weak-signal target — observed empirically with this
    feature set) must report IC=0, not NaN or a misleading spurious value."""
    pred = np.zeros(50)
    actual = np.random.default_rng(0).normal(size=50)
    m = train_predictor._forecast_metrics(pred, actual)
    assert m["ic"] == 0.0


def test_forecast_metrics_perfect_rank_correlation():
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = actual * 2 + 1  # monotonic transform -> IC should be 1.0
    m = train_predictor._forecast_metrics(pred, actual)
    assert m["ic"] == pytest.approx(1.0)


def test_predictor_artifact_contract():
    """If a trained predictor pickle exists, it must contain the canonical
    keys and have positive test-set IC — i.e. detectable rank signal, not a
    hit-rate floor (hit-rate/accuracy is the wrong metric for a regression
    forecast). See models/README.md for why this model exists at all."""
    path = "models/daily_predictor.pkl"
    if not os.path.exists(path):
        pytest.skip("No daily_predictor.pkl yet — run train_predictor.py first")
    with open(path, "rb") as f:
        art = pickle.load(f)
    required = {
        "model", "scaler", "feature_contract", "model_type",
        "test_ic", "test_dir_acc", "test_r2", "test_mae",
    }
    missing = required - art.keys()
    assert not missing, f"Predictor artifact missing keys: {missing}"
    assert art["test_ic"] > 0.0, (
        f"daily_predictor test IC {art['test_ic']:.4f} is not positive — "
        "no detectable rank signal."
    )


def test_predictor_strategy_inline_training_produces_valid_signals():
    """Without a pretrained artifact, DailyPredictorStrategy must fall back to
    in-session training and still emit a valid {-1,0,1} signal series — same
    contract as DailyLogisticStrategy / DailyXGBoostStrategy's fallback path."""
    df = _make_price_df(n=200)
    cfg = StrategyConfig(name="daily_predictor")
    strat = DailyPredictorStrategy(cfg, use_pretrained=False, threshold_window=20)
    sig = strat.signal(None, df)
    assert set(sig.unique()).issubset({-1, 0, 1})
    assert len(sig) > 0


def test_predictor_strategy_raises_without_artifact():
    """Pretrained mode with a missing artifact must raise, not silently
    return all-HOLD — same non-silent-failure contract as the other
    pretrained strategies."""
    cfg = StrategyConfig(name="daily_predictor")
    with pytest.raises(RuntimeError):
        DailyPredictorStrategy(cfg, use_pretrained=True, model_path="models/does_not_exist.pkl")


def test_compute_predictor_signal_buy_on_extreme_positive():
    """A predicted-return spike well above the trailing window's quantile
    must trigger BUY on the day it occurs."""
    pred_ret = np.full(80, 0.001)
    pred_ret[-1] = 0.05
    signals = compute_predictor_signal(pred_ret, signal_quantile=0.7, threshold_window=60)
    assert signals[-1] == 1


def test_compute_predictor_signal_sell_on_extreme_negative():
    pred_ret = np.full(80, 0.001)
    pred_ret[-1] = -0.05
    signals = compute_predictor_signal(pred_ret, signal_quantile=0.7, threshold_window=60)
    assert signals[-1] == -1


def test_compute_predictor_signal_hold_when_unremarkable():
    """A prediction with the same magnitude as the entire trailing window
    is exactly at the boundary, not above it — must not trigger a trade."""
    pred_ret = np.full(80, 0.001)
    signals = compute_predictor_signal(pred_ret, signal_quantile=0.7, threshold_window=60)
    assert signals[-1] == 0


def test_compute_predictor_signal_early_bars_are_hold():
    """Before min_periods=20 trailing predictions exist, the rolling
    threshold is undefined (NaN) — must default to HOLD, never trade on
    an undefined threshold."""
    pred_ret = np.array([0.05, -0.05, 0.03])
    signals = compute_predictor_signal(pred_ret, signal_quantile=0.7, threshold_window=60)
    assert (signals == 0).all()


def test_compute_predictor_signal_output_length_matches_input():
    pred_ret = np.full(100, 0.002)
    signals = compute_predictor_signal(pred_ret, signal_quantile=0.7, threshold_window=60)
    assert len(signals) == 100


def test_predictor_strategy_reads_best_params_from_pickle(tmp_path):
    """DailyPredictorStrategy must use best_signal_quantile/best_threshold_window
    from the pickle when env vars are not set — second priority level."""
    import pickle, os
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from daily_features import FEATURE_COLS

    # Build a minimal valid pickle with best params
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, len(FEATURE_COLS))).astype(np.float32)
    y = rng.normal(size=100)
    scaler = StandardScaler()
    model = Ridge().fit(scaler.fit_transform(X), y)
    artifact = {
        "model": model, "scaler": scaler,
        "feature_contract": FEATURE_COLS,
        "best_signal_quantile": 0.65,
        "best_threshold_window": 40,
    }
    pkl_path = str(tmp_path / "predictor.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artifact, f)

    env_backup = {k: os.environ.pop(k, None)
                  for k in ("PREDICTOR_SIGNAL_QUANTILE", "PREDICTOR_THRESHOLD_WINDOW")}
    try:
        cfg = StrategyConfig(name="daily_predictor")
        strat = DailyPredictorStrategy(cfg, use_pretrained=True, model_path=pkl_path)
        assert strat.signal_quantile == pytest.approx(0.65)
        assert strat.threshold_window == 40
    finally:
        for k, v in env_backup.items():
            if v is not None:
                os.environ[k] = v


def test_predictor_strategy_old_pickle_falls_back_to_defaults(tmp_path):
    """Pickle without best_signal_quantile/best_threshold_window keys must
    fall back to hardcoded defaults without raising."""
    import pickle
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from daily_features import FEATURE_COLS

    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, len(FEATURE_COLS))).astype(np.float32)
    y = rng.normal(size=100)
    scaler = StandardScaler()
    model = Ridge().fit(scaler.fit_transform(X), y)
    # Old-format pickle — no best_* keys
    artifact = {"model": model, "scaler": scaler, "feature_contract": FEATURE_COLS}
    pkl_path = str(tmp_path / "old_predictor.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artifact, f)

    import os
    env_backup = {k: os.environ.pop(k, None)
                  for k in ("PREDICTOR_SIGNAL_QUANTILE", "PREDICTOR_THRESHOLD_WINDOW")}
    try:
        cfg = StrategyConfig(name="daily_predictor")
        strat = DailyPredictorStrategy(cfg, use_pretrained=True, model_path=pkl_path)
        assert strat.signal_quantile == pytest.approx(0.7)
        assert strat.threshold_window == 60
    finally:
        for k, v in env_backup.items():
            if v is not None:
                os.environ[k] = v
