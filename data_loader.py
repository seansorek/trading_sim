
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
    df = df.between_time("09:30", "16:00", include_end=False)
    return df

def _add_spread(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Conservative spread proxy (~2 bps of close). Ensure minimum absolute value.
    df["spread"] = np.maximum(df["close"] * 0.0002, 0.01)
    return df

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct columns, sort, drop duplicates, US hours, add spread."""
    df = df.sort_index().loc[~df.index.duplicated(keep="last")]
    df = _ensure_cols(df)
    df = _filter_us_hours(df)
    df = _add_spread(df)
    df.index.name = "timestamp"
    return df

# --- Loaders: YFinance and CSV ---
def load_yfinance(symbol: str, start: str, end: str, interval: str = "1m") -> pd.DataFrame:
    """
    Fetch intraday bars from Yahoo Finance via yfinance.
    NOTE: 1m data is often limited to ~7 days for free access and may be delayed or incomplete.
    """
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval, actions=False, prepost=False)
    if df.empty:
        raise ValueError(f"No data returned by yfinance for {symbol} in {start}–{end} ({interval}).")
    df = df.rename(columns={"Open":"open", "High":"high", "Low":"low", "Close":"close", "Volume":"volume"})
    # yfinance timestamps are tz-aware (UTC or local depending on source). Standardize:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return _standardize(df)

def load_csv(path: str) -> pd.DataFrame:
    """
    Load intraday bars from a local CSV with columns:
    timestamp, open, high, low, close, volume
    """
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    # Assume timestamp is UTC if naive; localize to UTC then convert
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return _standardize(df)

# --- Optional: Alpha Vantage (intraday 1min/5min/15min); requires API key ---
def load_alpha_vantage(symbol: str, api_key: str, interval: str = "1min", outputsize: str = "full") -> pd.DataFrame:
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
