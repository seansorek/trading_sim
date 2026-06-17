"""test_train_cache.py — Tests for DB cache freshness in _load_symbol (#25)."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import DB
from train_models import _load_symbol, _STALE_TOLERANCE_BDAYS


def _make_bar_df(start: str, n: int = 100) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
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


def test_stale_cache_triggers_refetch(tmp_path):
    """When cached data's latest bar is too old relative to the requested end
    date, _load_symbol should re-fetch from yfinance instead of returning
    stale cached data."""
    db = DB(str(tmp_path / "test.db"))

    # Seed cache with data up to 2023-06-01 (about 100 business days from 2023-01-02)
    old_data = _make_bar_df("2023-01-02", n=100)
    db.upsert_bars("AAPL", "1d", old_data)

    # Now request data up to 2024-06-01 — the cache is very stale
    fresh_data = _make_bar_df("2023-01-02", n=350)

    with patch("train_models.load_yfinance", return_value=fresh_data) as mock_fetch:
        result = _load_symbol("AAPL", "2023-01-02", "2024-06-01", db)

    # load_yfinance should have been called because cache was stale
    mock_fetch.assert_called_once()
    assert result is not None
    assert len(result) == 350


def test_fresh_cache_skips_fetch(tmp_path):
    """When cached data is recent enough, _load_symbol should use the cache
    and NOT call yfinance."""
    db = DB(str(tmp_path / "test.db"))

    # Seed cache with data ending near the end date
    data = _make_bar_df("2024-01-02", n=100)
    db.upsert_bars("AAPL", "1d", data)

    latest_bar = data.index.max()
    # Request end date only 1 business day after latest bar — well within tolerance
    end_date = (latest_bar + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")

    with patch("train_models.load_yfinance") as mock_fetch:
        result = _load_symbol("AAPL", "2024-01-02", end_date, db)

    # yfinance should NOT have been called — cache is fresh enough
    mock_fetch.assert_not_called()
    assert result is not None
    assert len(result) == 100


def test_stale_tolerance_boundary(tmp_path):
    """Cache exactly at the stale boundary should still be considered fresh."""
    db = DB(str(tmp_path / "test.db"))

    data = _make_bar_df("2024-01-02", n=100)
    db.upsert_bars("AAPL", "1d", data)

    latest_bar = data.index.max()
    # End date exactly _STALE_TOLERANCE_BDAYS business days after latest bar
    end_date = (latest_bar + pd.tseries.offsets.BDay(_STALE_TOLERANCE_BDAYS)).strftime("%Y-%m-%d")

    with patch("train_models.load_yfinance") as mock_fetch:
        result = _load_symbol("AAPL", "2024-01-02", end_date, db)

    # At the boundary, cache should still be accepted
    mock_fetch.assert_not_called()
    assert result is not None


def test_one_past_stale_tolerance_triggers_refetch(tmp_path):
    """Cache one business day past the tolerance should trigger a refetch."""
    db = DB(str(tmp_path / "test.db"))

    data = _make_bar_df("2024-01-02", n=100)
    db.upsert_bars("AAPL", "1d", data)

    latest_bar = data.index.max()
    # End date one business day past the stale tolerance
    end_date = (latest_bar + pd.tseries.offsets.BDay(_STALE_TOLERANCE_BDAYS + 1)).strftime("%Y-%m-%d")

    fresh_data = _make_bar_df("2024-01-02", n=110)

    with patch("train_models.load_yfinance", return_value=fresh_data) as mock_fetch:
        result = _load_symbol("AAPL", "2024-01-02", end_date, db)

    mock_fetch.assert_called_once()
    assert result is not None
