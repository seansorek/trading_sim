
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime
import os

# Make sure the script can find the data_loader module
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import (
    load_yfinance,
    load_csv,
    _standardize,
    _ensure_cols,
    _filter_us_hours,
    _add_spread
)

@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    start_date = datetime(2025, 10, 1, 9, 30)
    end_date = datetime(2025, 10, 1, 16, 0)
    index = pd.date_range(start_date, end_date, freq='1min', inclusive='left')
    data = {
        'open': np.random.uniform(99, 101, size=len(index)),
        'high': np.random.uniform(100, 102, size=len(index)),
        'low': np.random.uniform(98, 100, size=len(index)),
        'close': np.random.uniform(99, 101, size=len(index)),
        'volume': np.random.randint(1000, 5000, size=len(index))
    }
    df = pd.DataFrame(data, index=index)
    df.index.name = 'timestamp'
    return df

@pytest.fixture
def dataframe_with_nan():
    """Create a dataframe with some NaN values."""
    start_date = datetime(2025, 10, 1, 9, 30)
    end_date = datetime(2025, 10, 1, 16, 0)
    index = pd.date_range(start_date, end_date, freq='1min', inclusive='left')
    data = {
        'open': np.random.uniform(99, 101, size=len(index)),
        'high': np.random.uniform(100, 102, size=len(index)),
        'low': np.random.uniform(98, 100, size=len(index)),
        'close': np.random.uniform(99, 101, size=len(index)),
        'volume': np.random.randint(1000, 5000, size=len(index))
    }
    df = pd.DataFrame(data, index=index)
    df.iloc[5:10, 1:3] = np.nan
    df.index.name = 'timestamp'
    return df

def test_ensure_cols(sample_dataframe):
    """Test that _ensure_cols raises error for missing columns."""
    with pytest.raises(ValueError):
        _ensure_cols(sample_dataframe.drop(columns=['open']))
    
    df = _ensure_cols(sample_dataframe)
    assert 'open' in df.columns

def test_filter_us_hours(sample_dataframe):
    """Test that _filter_us_hours correctly filters time."""
    df = sample_dataframe.copy()
    # Add some data outside of US hours
    df_early = df.copy().set_index(df.index - pd.Timedelta(hours=1))
    df_late = df.copy().set_index(df.index + pd.Timedelta(hours=8))
    df_extended = pd.concat([df, df_early, df_late])
    
    df_filtered = _filter_us_hours(df_extended)
    
    assert df_filtered.index.min().hour == 9
    assert df_filtered.index.max().hour < 16
    assert len(df_filtered) < len(df_extended)

def test_add_spread(sample_dataframe):
    """Test that _add_spread adds a spread column."""
    df = _add_spread(sample_dataframe)
    assert 'spread' in df.columns
    assert (df['spread'] >= 0.01).all()

def test_standardize_clean(sample_dataframe):
    """Test _standardize with a clean dataframe."""
    df = _standardize(sample_dataframe)
    assert not df.isnull().values.any()
    assert 'spread' in df.columns
    assert df.index.name == 'timestamp'

def test_standardize_with_nans(dataframe_with_nan):
    """Test that _standardize correctly fills NaNs."""
    df_dirty = dataframe_with_nan
    assert df_dirty.isnull().values.any()
    
    df_clean = _standardize(df_dirty)
    assert not df_clean.isnull().values.any()

@patch('yfinance.Ticker')
def test_load_yfinance(mock_ticker, sample_dataframe):
    """Test load_yfinance with a mocked yfinance call."""
    mock_instance = MagicMock()
    mock_instance.history.return_value = sample_dataframe
    mock_ticker.return_value = mock_instance

    # Use '1d' interval: 3650-day history limit means 2025 dates are always valid.
    df = load_yfinance('SPY', '2025-10-01', '2025-10-02', interval='1d')

    mock_ticker.assert_called_with('SPY')
    mock_instance.history.assert_called_with(start='2025-10-01', end='2025-10-02', interval='1d', actions=False, prepost=False)
    assert not df.isnull().values.any()
    assert 'spread' in df.columns

def test_load_csv(sample_dataframe, tmpdir):
    """Test loading data from a CSV file."""
    csv_path = tmpdir.join('test_data.csv')
    sample_dataframe.to_csv(csv_path)

    df = load_csv(str(csv_path))

    assert not df.empty
    assert not df.isnull().values.any()
    assert 'spread' in df.columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        assert col in df.columns
    assert df.index.tz is not None  # load_csv always produces tz-aware index
    assert (df['spread'] >= 0.01).all()


# ---------------------------------------------------------------------------
# Issue #40 — load_csv must not drop daily bars with midnight timestamps
# ---------------------------------------------------------------------------

def test_load_csv_daily_bars_not_dropped(tmpdir):
    """
    Daily bars have midnight-UTC timestamps that fall outside 09:30-16:00 ET.
    With the default intraday=True, _standardize's _filter_us_hours drops every
    row. load_csv must detect daily intervals and skip the intraday filter.
    """
    n = 20
    dates = pd.date_range("2024-01-02", periods=n, freq="B")  # business days, midnight
    data = {
        "timestamp": dates,
        "open": np.arange(100.0, 100.0 + n),
        "high": np.arange(101.0, 101.0 + n),
        "low": np.arange(99.0, 99.0 + n),
        "close": np.arange(100.5, 100.5 + n),
        "volume": np.full(n, 1_000_000),
    }
    csv_path = tmpdir.join("daily_bars.csv")
    pd.DataFrame(data).to_csv(str(csv_path), index=False)

    df = load_csv(str(csv_path), interval="1d")

    assert len(df) == n, (
        f"Expected {n} daily bars, got {len(df)}. "
        "load_csv likely applied intraday US-hours filter to daily data."
    )
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns


def test_load_csv_intraday_still_filters(tmpdir):
    """
    When interval indicates intraday (e.g. '5m'), load_csv should still apply
    the US-hours filter via _standardize(intraday=True).
    """
    # Create bars with timestamps outside US hours (e.g., 03:00 UTC)
    n = 10
    dates = pd.date_range("2024-01-02 03:00:00", periods=n, freq="5min")
    data = {
        "timestamp": dates,
        "open": np.full(n, 100.0),
        "high": np.full(n, 101.0),
        "low": np.full(n, 99.0),
        "close": np.full(n, 100.5),
        "volume": np.full(n, 1_000_000),
    }
    csv_path = tmpdir.join("intraday_bars.csv")
    pd.DataFrame(data).to_csv(str(csv_path), index=False)

    df = load_csv(str(csv_path), interval="5m")

    # 03:00 UTC is outside 09:30-16:00 ET, so all bars should be filtered out
    assert len(df) == 0, (
        f"Expected 0 bars after US-hours filter for 03:00 UTC data, got {len(df)}"
    )
