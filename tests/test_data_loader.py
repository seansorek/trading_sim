
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
