
# data_loader.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# --- Utilities ---
US_TZ = "America/New_York"

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def _filter_us_hours(df: pd.DataFrame) -> pd.DataFrame:
    # Filter for US regular trading hours: 09:30–16:00 local time
    idx_local = df.index.tz_convert(US_TZ) if df.index.tz is not None else df.index.tz_localize(US_TZ)
    df = df.copy()
    df.index = idx_local
    df = df.between_time("09:30", "16:00", inclusive="left")
    return df

def _add_spread(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Conservative spread proxy (~2 bps of close). Ensure minimum absolute value.
    df["spread"] = np.maximum(df["close"] * 0.0002, 0.01)
    return df

def _standardize(df: pd.DataFrame, intraday: bool = True) -> pd.DataFrame:
    """Ensure correct columns, sort, drop duplicates, optional US hours, add spread."""
    df = df.sort_index().loc[~df.index.duplicated(keep="last")]
    df = _ensure_cols(df)
    if intraday:
        df = _filter_us_hours(df)
    df = df.ffill().bfill()  # Fill any gaps
    df = _add_spread(df)
    df.index.name = "timestamp"
    return df

def _check_trading_day_gaps(
    df: pd.DataFrame, max_gap_days: int = 5, symbol: str = ""
) -> None:
    """Raise ValueError if the date index has any gap exceeding *max_gap_days* trading days."""
    if df.empty or len(df) < 2:
        return
    dates = pd.Series(df.index.normalize().unique()).sort_values()
    gaps = dates.diff().dt.days.dropna()
    worst = gaps.max()
    if worst > max_gap_days:
        gap_idx = gaps.idxmax()
        gap_start = dates.iloc[gap_idx - 1]
        gap_end = dates.iloc[gap_idx]
        raise ValueError(
            f"Data for {symbol} has a {int(worst)}-day gap "
            f"({gap_start.date()} to {gap_end.date()}), "
            f"exceeding the {max_gap_days}-day tolerance. "
            f"One or more chunk fetches failed."
        )


# --- Cache freshness helper ---
def check_cache_freshness(
    cached: pd.DataFrame, end: str, stale_tolerance_bdays: int = 4
) -> bool:
    """Return True if *cached* data is fresh enough relative to *end*.

    The cache is considered fresh when its latest bar is within
    *stale_tolerance_bdays* business days of the requested *end* date.
    """
    end_dt = pd.to_datetime(end)
    latest_bar = cached.index.max()
    # Normalize both to tz-naive timestamps for comparison
    latest_date = latest_bar.tz_localize(None) if latest_bar.tzinfo else latest_bar
    end_date = end_dt.tz_localize(None) if end_dt.tzinfo else end_dt
    stale_cutoff = end_date - pd.tseries.offsets.BDay(stale_tolerance_bdays)
    return latest_date >= stale_cutoff


# --- Loaders: YFinance and CSV ---
def load_yfinance(symbol: str, start: str, end: str, interval: str = "5m") -> pd.DataFrame:
    """
    Fetch intraday bars from Yahoo Finance via yfinance.
    Automatically chunks large requests to work around yfinance API limits.
    
    Limits per interval (from current date, not end date):
    - 1m: last 7 days only
    - 5m, 15m, 30m: last 60 days only
    - 60m, 90m: last 730 days
    - 1d and above: years of history
    
    Raises ValueError if no data is returned or if data is insufficient.
    """
    import yfinance as yf
    
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    now = datetime.now()
    
    # Determine maximum historical range based on interval
    if interval == "1m":
        max_history_days = 7
        chunk_days = 6  # Fetch in 6-day chunks
    elif interval in ["5m", "15m", "30m"]:
        max_history_days = 60
        chunk_days = 30  # Fetch in 30-day chunks
    elif interval in ["60m", "90m"]:
        max_history_days = 730
        chunk_days = 180  # Fetch in 180-day chunks
    else:
        # Daily/weekly intervals - no practical limit
        max_history_days = 3650
        chunk_days = 730
    
    # Adjust start date if it's beyond the available history
    # Yahoo's limit is from TODAY, not from end_dt
    earliest_available = now - timedelta(days=max_history_days - 1)  # -1 for safety margin
    if start_dt < earliest_available:
        original_start = start_dt
        start_dt = earliest_available
        print(f"[warn] {interval} data for {symbol} only available for last {max_history_days} days from today")
        print(f"      Adjusted start date: {original_start.date()} -> {start_dt.date()}")
    
    # Ensure end date is not in the future
    if end_dt > now:
        end_dt = now
    
    # Ensure we're not requesting future dates
    if start_dt > now:
        raise ValueError(f"Start date {start_dt.date()} is in the future")
    
    total_days = (end_dt - start_dt).days
    
    # For very recent data, don't chunk - just fetch directly
    if total_days <= 0:
        raise ValueError(f"Invalid date range: start {start_dt.date()} >= end {end_dt.date()}")
    
    # Use chunking if request is large OR if we want to be safe with API limits
    if total_days > chunk_days:
        print(f"[info] Fetching {interval} data for {symbol} in {chunk_days}-day chunks ({total_days} days total)...")
        all_dfs = []
        current_start = start_dt
        chunk_num = 0
        
        failed_chunks = []
        while current_start < end_dt:
            chunk_end = min(current_start + timedelta(days=chunk_days), end_dt)
            chunk_num += 1

            try:
                ticker = yf.Ticker(symbol)
                chunk_df = ticker.history(
                    start=current_start.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"),
                    interval=interval,
                    actions=False,
                    prepost=False
                )

                if not chunk_df.empty:
                    all_dfs.append(chunk_df)
                    print(f"  Chunk {chunk_num}: {current_start.date()} to {chunk_end.date()} - {len(chunk_df)} bars")
                else:
                    print(f"  [warn] Chunk {chunk_num}: No data returned for {current_start.date()} to {chunk_end.date()}")
                    failed_chunks.append((chunk_num, current_start.date(), chunk_end.date()))

            except Exception as e:
                print(f"  [error] Chunk {chunk_num} failed ({current_start.date()} to {chunk_end.date()}): {e}")
                failed_chunks.append((chunk_num, current_start.date(), chunk_end.date()))

            current_start = chunk_end

        if not all_dfs:
            df = pd.DataFrame()
        else:
            # Combine all chunks and remove duplicates
            df = pd.concat(all_dfs)
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            print(f"  Combined {len(all_dfs)} chunks into {len(df)} total bars")

            # Reject result if failed chunks left gaps in coverage
            if failed_chunks:
                _check_trading_day_gaps(df, max_gap_days=5, symbol=symbol)
    else:
        # Single request for small ranges
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval, actions=False, prepost=False)
    
    # Check if empty
    if df.empty:
        raise ValueError(f"No data returned by yfinance for {symbol} in {start}–{end} ({interval}). Symbol may be invalid, data unavailable, or API limit exceeded.")
    
    # Check if we have sufficient data (at least 10 candles)
    if len(df) < 10:
        raise ValueError(f"Insufficient data for {symbol}: only {len(df)} candles returned (need at least 10). Try a different date range or interval.")
    
    # Rename columns
    df = df.rename(columns={"Open":"open", "High":"high", "Low":"low", "Close":"close", "Volume":"volume"})
    
    # Ensure required columns exist
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {symbol}: {missing}")
    
    # Standardize timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    
    # For daily/weekly/monthly intervals, do not filter intraday hours
    is_intraday = interval.endswith('m') or interval.endswith('h')
    return _standardize(df, intraday=is_intraday)

def load_csv(path: str, interval: str = "1d") -> pd.DataFrame:
    """
    Load bars from a local CSV with columns:
    timestamp, open, high, low, close, volume

    Args:
        path: Path to the CSV file.
        interval: Bar interval (e.g. "1d", "5m", "1h"). Used to determine
                  whether to apply intraday US-hours filtering. Daily and
                  weekly bars skip the filter to avoid dropping rows with
                  midnight timestamps.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    # Assume timestamp is UTC if naive; localize to UTC then convert
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    is_intraday = interval.endswith("m") or interval.endswith("h")
    return _standardize(df, intraday=is_intraday)

# --- Optional: Alpha Vantage (intraday 1min/5min/15min); requires API key ---
def load_alpha_vantage(symbol: str, api_key: str, interval: str = "5min", outputsize: str = "full") -> pd.DataFrame:
    """
    Fetch intraday bars using Alpha Vantage TIME_SERIES_INTRADAY.
    NOTE: Free tier has strict rate limits and may have limited historical coverage.
    """
    import requests
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "datatype": "json",
        "apikey": api_key
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Key pattern: "Time Series (1min)" etc.
    ts_key = f"Time Series ({interval})"
    if ts_key not in data:
        raise ValueError(f"Unexpected Alpha Vantage response keys: {list(data.keys())}")
    ts = data[ts_key]
    records = []
    for ts_str, row in ts.items():
        records.append({
            "timestamp": pd.to_datetime(ts_str, utc=True),
            "open": float(row["1. open"]),
            "high": float(row["2. high"]),
            "low": float(row["3. low"]),
            "close": float(row["4. close"]),
            "volume": float(row["5. volume"])
        })
    df = pd.DataFrame.from_records(records).set_index("timestamp").sort_index()
    return _standardize(df)

# --- Save helper ---
def save_to_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
