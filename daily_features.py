"""
daily_features.py

Generate daily OHLCV features and rolling technical indicators for next-day
return prediction. Designed for 1d interval data from yfinance.
"""
import pandas as pd
import numpy as np


def _safe_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(window).mean()
    roll_down = pd.Series(down, index=series.index).rolling(window).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def make_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from daily OHLCV data.
    Returns a DataFrame aligned with df.index containing feature columns.
    """
    feats = pd.DataFrame(index=df.index)
    # Keep raw close to compute PnL in RL env
    feats['close'] = df['close'].astype(float)

    # Basic returns and volatility
    feats['ret_1d'] = df['close'].pct_change(1)
    feats['ret_5d'] = df['close'].pct_change(5)
    feats['ret_10d'] = df['close'].pct_change(10)
    feats['vol_20d'] = df['close'].pct_change().rolling(20).std()

    # Moving averages and spreads
    sma_10 = df['close'].rolling(10).mean()
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()
    ema_12 = _safe_ema(df['close'], 12)
    ema_26 = _safe_ema(df['close'], 26)
    feats['sma_10'] = sma_10
    feats['sma_20'] = sma_20
    feats['sma_50'] = sma_50
    feats['ma_spread_10_20'] = sma_10 - sma_20
    feats['ma_spread_20_50'] = sma_20 - sma_50

    # MACD
    macd = ema_12 - ema_26
    signal = _safe_ema(macd, 9)
    feats['macd'] = macd
    feats['macd_signal'] = signal
    feats['macd_hist'] = macd - signal

    # RSI and ATR
    feats['rsi_14'] = _rsi(df['close'], 14)
    feats['atr_14'] = _atr(df, 14)

    # Price position vs moving averages
    feats['price_vs_sma20'] = (df['close'] - sma_20) / (sma_20 + 1e-12)
    feats['price_vs_sma50'] = (df['close'] - sma_50) / (sma_50 + 1e-12)

    # Volume features
    feats['vol_z_20'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + 1e-12)
    feats['volume_ma_20'] = df['volume'].rolling(20).mean()

    # Forward target: next-day return (not part of features but useful downstream)
    feats['fwd_ret_1d'] = df['close'].pct_change(1).shift(-1)

    # Clean up
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.fillna(0.0)
    return feats
