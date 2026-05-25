"""test_db.py — SQLite data layer tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import DB


def _make_bar_df(n: int = 10, symbol: str = "AAPL") -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    close = 150.0 + rng.normal(0, 1, n).cumsum()
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Bar data
# ---------------------------------------------------------------------------

def test_upsert_and_load_bars(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    df = _make_bar_df(10)
    written = db.upsert_bars("AAPL", "1d", df)
    assert written == 10

    loaded = db.load_bars("AAPL", "1d", "2000-01-01", "2099-01-01")
    assert loaded is not None
    assert len(loaded) == 10
    assert "close" in loaded.columns


def test_upsert_bars_idempotent(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    df = _make_bar_df(5)
    db.upsert_bars("SPY", "1d", df)
    db.upsert_bars("SPY", "1d", df)  # second upsert should not duplicate
    loaded = db.load_bars("SPY", "1d", "2000-01-01", "2099-01-01")
    assert len(loaded) == 5


def test_load_bars_returns_none_when_empty(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    result = db.load_bars("NONEXISTENT", "1d", "2024-01-01", "2024-12-31")
    assert result is None


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _register(db: DB, model_key: str = "daily_logistic") -> int:
    from daily_features import FEATURE_COLS
    return db.register_model(
        model_key=model_key,
        artifact_path=f"models/{model_key}.pkl",
        feature_contract=FEATURE_COLS,
        trained_on=["AAPL", "SPY"],
        train_start="2023-01-01",
        train_end="2024-01-01",
        train_samples=500,
        test_samples=100,
        train_accuracy=0.55,
        test_accuracy=0.41,
        test_f1=0.38,
        label_map={0: "SELL", 1: "HOLD", 2: "BUY"},
    )


def test_register_model_returns_version(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    v = _register(db)
    assert v == 1


def test_second_registration_increments_version(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    v1 = _register(db)
    v2 = _register(db)
    assert v2 == v1 + 1


def test_get_active_model_returns_latest(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    _register(db)
    v2 = _register(db)
    db.deactivate_old_models("daily_logistic", keep_version=v2)

    active = db.get_active_model("daily_logistic")
    assert active is not None
    assert active["version"] == v2


def test_deactivate_old_models(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    _register(db)
    v2 = _register(db)
    db.deactivate_old_models("daily_logistic", keep_version=v2)

    # v1 should be inactive; only v2 active
    active = db.get_active_model("daily_logistic")
    assert active["version"] == v2


def test_get_active_model_returns_none_when_missing(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    assert db.get_active_model("nonexistent_model") is None


def test_feature_contract_roundtrips_as_list(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    _register(db)
    active = db.get_active_model("daily_logistic")
    assert isinstance(active["feature_contract"], list)
    from daily_features import FEATURE_COLS
    assert active["feature_contract"] == FEATURE_COLS


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def test_upsert_prediction_roundtrip(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    db.upsert_prediction("AAPL", "daily_logistic", 1, "2026-05-23", "BUY", 0.72, 180.0)
    preds = db.get_predictions("2026-05-23")
    assert len(preds) == 1
    assert preds.iloc[0]["signal"] == "BUY"
    assert preds.iloc[0]["symbol"] == "AAPL"


def test_upsert_prediction_deduplicates(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    db.upsert_prediction("AAPL", "daily_logistic", 1, "2026-05-23", "BUY", 0.72, 180.0)
    db.upsert_prediction("AAPL", "daily_logistic", 1, "2026-05-23", "SELL", 0.68, 181.0)
    preds = db.get_predictions("2026-05-23")
    assert len(preds) == 1
    assert preds.iloc[0]["signal"] == "SELL"  # second upsert wins


def test_get_predictions_returns_empty_df_for_missing_date(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    preds = db.get_predictions("1990-01-01")
    assert preds.empty


# ---------------------------------------------------------------------------
# Backtest runs
# ---------------------------------------------------------------------------

def test_insert_backtest_run_returns_id(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    row_id = db.insert_backtest_run(
        run_id="test-run-1",
        symbol="AAPL",
        strategy="daily_logistic",
        data_start="2024-01-01",
        data_end="2024-12-31",
        start_cash=100_000.0,
        final_equity=105_000.0,
        total_return_pct=5.0,
        daily_sharpe=0.8,
    )
    assert isinstance(row_id, int) and row_id > 0


def test_get_backtest_runs(tmp_path):
    db = DB(str(tmp_path / "test.db"))
    db.insert_backtest_run(
        run_id="r1", symbol="AAPL", strategy="daily_logistic",
        data_start="2024-01-01", data_end="2024-12-31", start_cash=100_000.0,
    )
    db.insert_backtest_run(
        run_id="r1", symbol="MSFT", strategy="daily_logistic",
        data_start="2024-01-01", data_end="2024-12-31", start_cash=100_000.0,
    )
    df = db.get_backtest_runs(strategy="daily_logistic")
    assert len(df) == 2
