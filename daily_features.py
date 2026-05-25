"""
daily_features.py — Daily OHLCV feature engineering.

FEATURE_COLS is the canonical ordered list of model input columns.
All training and prediction code must index features using this list,
never by DataFrame column iteration order.
"""
import numpy as np
import pandas as pd


# Version string stored alongside model pickles. Bump when FEATURE_COLS changes;
# old models become explicitly incompatible.
FEATURE_SET_NAME: str = "daily_v1"

# Canonical feature order — the contract between training and prediction.
# Raw cumsum columns (vpt, ad_line) are excluded: they are non-stationary.
# Raw SMAs (sma_10, sma_20, sma_50) excluded; keep ratio/spread derivatives only.
FEATURE_COLS: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "vol_20d",
    "ma_spread_10_20",
    "ma_spread_20_50",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "atr_14",
    "price_vs_sma20",
    "price_vs_sma50",
    "vol_z_20",
    "volume_ma_20",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "bb_position",
    "stoch_k",
    "stoch_d",
    "williams_r",
    "momentum_10",
    "roc_12",
    "atr_normalized",
    "vpt_normalized",
    "ad_normalized",
    "obv_normalized",
]


def discretize_labels(
    returns: np.ndarray,
    pos_thr: float = 0.002,
    neg_thr: float = -0.002,
) -> np.ndarray:
    """
    Map forward returns to {0:SELL, 1:HOLD, 2:BUY}.

    This encoding is used consistently across all training scripts and the
    backtester. Do not use ad-hoc label assignments elsewhere.
    """
    y = np.ones(len(returns), dtype=int)  # default: HOLD
    y[returns > pos_thr] = 2              # BUY
    y[returns < neg_thr] = 0              # SELL
    return y


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
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def make_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from daily OHLCV data.

    Returns a DataFrame with all columns in FEATURE_COLS plus auxiliary
    columns 'close' and 'fwd_ret_1d' (not part of the model input).
    Always index model inputs via feats[FEATURE_COLS], never by position.
    """
    feats = pd.DataFrame(index=df.index)
    feats["close"] = df["close"].astype(float)

    feats["ret_1d"] = df["close"].pct_change(1)
    feats["ret_5d"] = df["close"].pct_change(5)
    feats["ret_10d"] = df["close"].pct_change(10)
    feats["vol_20d"] = df["close"].pct_change().rolling(20).std()

    sma_10 = df["close"].rolling(10).mean()
    sma_20 = df["close"].rolling(20).mean()
    sma_50 = df["close"].rolling(50).mean()
    ema_12 = _safe_ema(df["close"], 12)
    ema_26 = _safe_ema(df["close"], 26)

    feats["ma_spread_10_20"] = sma_10 - sma_20
    feats["ma_spread_20_50"] = sma_20 - sma_50

    macd = ema_12 - ema_26
    signal = _safe_ema(macd, 9)
    feats["macd"] = macd
    feats["macd_signal"] = signal
    feats["macd_hist"] = macd - signal

    feats["rsi_14"] = _rsi(df["close"], 14)
    feats["atr_14"] = _atr(df, 14)

    feats["price_vs_sma20"] = (df["close"] - sma_20) / (sma_20 + 1e-12)
    feats["price_vs_sma50"] = (df["close"] - sma_50) / (sma_50 + 1e-12)

    feats["vol_z_20"] = (df["volume"] - df["volume"].rolling(20).mean()) / (
        df["volume"].rolling(20).std() + 1e-12
    )
    feats["volume_ma_20"] = df["volume"].rolling(20).mean()

    bb_mid = df["close"].rolling(20).mean()
    bb_std_dev = df["close"].rolling(20).std()
    feats["bb_upper"] = bb_mid + (bb_std_dev * 2)
    feats["bb_lower"] = bb_mid - (bb_std_dev * 2)
    feats["bb_width"] = feats["bb_upper"] - feats["bb_lower"]
    feats["bb_position"] = (df["close"] - feats["bb_lower"]) / (feats["bb_width"] + 1e-12)

    lowest_low = df["low"].rolling(14).min()
    highest_high = df["high"].rolling(14).max()
    feats["stoch_k"] = 100.0 * (df["close"] - lowest_low) / (highest_high - lowest_low + 1e-12)
    feats["stoch_d"] = feats["stoch_k"].rolling(3).mean()

    feats["williams_r"] = -100.0 * (highest_high - df["close"]) / (highest_high - lowest_low + 1e-12)

    feats["momentum_10"] = df["close"].diff(10)
    feats["roc_12"] = (df["close"] - df["close"].shift(12)) / (df["close"].shift(12) + 1e-12)
    feats["atr_normalized"] = _atr(df, 14) / (df["close"] + 1e-12)

    # Cumsum-based indicators kept for internal computation but normalized before
    # being included in FEATURE_COLS (raw cumsum is non-stationary)
    vpt_raw = (df["volume"] * df["close"].pct_change()).cumsum()
    feats["vpt_normalized"] = (vpt_raw - vpt_raw.rolling(20).mean()) / (
        vpt_raw.rolling(20).std() + 1e-12
    )

    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"] + 1e-12
    )
    ad_raw = (clv * df["volume"]).cumsum()
    feats["ad_normalized"] = (ad_raw - ad_raw.rolling(20).mean()) / (
        ad_raw.rolling(20).std() + 1e-12
    )

    obv_raw = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    feats["obv_normalized"] = (obv_raw - obv_raw.rolling(20).mean()) / (
        obv_raw.rolling(20).std() + 1e-12
    )

    feats["fwd_ret_1d"] = df["close"].pct_change(1).shift(-1)

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.fillna(0.0)
    return feats
