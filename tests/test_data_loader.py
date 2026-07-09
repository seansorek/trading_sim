
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
    _add_spread,
    _check_trading_day_gaps,
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


def test_standardize_drops_leading_nan_rows_instead_of_backfilling(sample_dataframe):
    """Regression test for #43: leading NaN rows must be dropped, not
    back-filled with a later (future) price -- back-filling leaks look-ahead
    information into backtests/training."""
    df_dirty = sample_dataframe.copy()
    future_close = 999.0
    price_cols = ["open", "high", "low", "close"]
    df_dirty.iloc[0, [df_dirty.columns.get_loc(c) for c in price_cols]] = np.nan
    df_dirty.iloc[1, df_dirty.columns.get_loc("close")] = future_close

    df_clean = _standardize(df_dirty)

    # The leading all-NaN row should be dropped entirely rather than
    # fabricated via back-fill from the next (future) close price.
    assert len(df_clean) == len(df_dirty) - 1
    assert not df_clean.isnull().values.any()


def test_standardize_raises_when_fill_fraction_exceeds_threshold(sample_dataframe):
    """Regression test for #43: heavily-missing data should fail loudly
    rather than being silently fabricated via fill."""
    df_dirty = sample_dataframe.copy()
    n = len(df_dirty)
    # Blank out most rows (after the first, which anchors first_valid).
    df_dirty.iloc[1: int(n * 0.9), df_dirty.columns.get_loc("close")] = np.nan

    with pytest.raises(ValueError, match="forward-fill"):
        _standardize(df_dirty)

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
# Gap detection in chunked fetches (#32)
# ---------------------------------------------------------------------------

def test_check_trading_day_gaps_raises_on_large_gap():
    """_check_trading_day_gaps should raise ValueError when a gap exceeds the threshold."""
    # Create a DataFrame with a 10-day gap (way over the 5-day default)
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04",
                            "2024-01-18", "2024-01-19"])  # 14-day gap
    df = pd.DataFrame({"close": [100.0] * 5}, index=dates)

    with pytest.raises(ValueError, match="gap"):
        _check_trading_day_gaps(df, max_gap_days=5, symbol="TEST")


def test_check_trading_day_gaps_passes_on_small_gap():
    """Normal weekend-sized gaps (2-3 days) should not raise."""
    dates = pd.bdate_range("2024-01-02", periods=20)  # business days only
    df = pd.DataFrame({"close": [100.0] * 20}, index=dates)

    # Should not raise — weekends create 2-day gaps at most
    _check_trading_day_gaps(df, max_gap_days=5, symbol="TEST")


@patch('yfinance.Ticker')
def test_load_yfinance_rejects_gapped_data_from_failed_chunks(mock_ticker):
    """When a middle chunk fails, load_yfinance should raise due to the gap."""
    # Use 1d interval (chunk_days=730). Total range must exceed 730 days
    # to trigger chunking: ~3 years.

    # Build chunk 1 data: first 2 years
    dates_1 = pd.bdate_range("2022-06-01", "2023-06-01", freq="B")
    chunk1 = pd.DataFrame({
        "Open": [100.0] * len(dates_1),
        "High": [101.0] * len(dates_1),
        "Low": [99.0] * len(dates_1),
        "Close": [100.0] * len(dates_1),
        "Volume": [1000000] * len(dates_1),
    }, index=dates_1)

    # Build chunk 3 data: last year (big gap from chunk 1 because chunk 2 fails)
    dates_3 = pd.bdate_range("2025-06-01", "2026-06-01", freq="B")
    chunk3 = pd.DataFrame({
        "Open": [100.0] * len(dates_3),
        "High": [101.0] * len(dates_3),
        "Low": [99.0] * len(dates_3),
        "Close": [100.0] * len(dates_3),
        "Volume": [1000000] * len(dates_3),
    }, index=dates_3)

    call_count = [0]

    def mock_history(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return chunk1
        elif call_count[0] == 2:
            raise ConnectionError("simulated failure")
        else:
            return chunk3

    mock_instance = MagicMock()
    mock_instance.history.side_effect = mock_history
    mock_ticker.return_value = mock_instance

    with pytest.raises(ValueError, match="gap"):
        load_yfinance("FAKE", "2022-06-01", "2026-06-15", interval="1d")


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
