"""
daily_features.py — Daily OHLCV feature engineering.

FEATURE_COLS is the canonical ordered list of model input columns (30 features).
All training and prediction code must index features using this list,
never by DataFrame column iteration order.
"""
import numpy as np
import pandas as pd


# Horizon (in trading days) of fwd_ret_1d. See note where it is computed.
FWD_RET_HORIZON_DAYS = 3


# Version string stored alongside model pickles. Bump when FEATURE_COLS changes;
# old models become explicitly incompatible.
FEATURE_SET_NAME: str = "daily_v6"

# Canonical feature order — the contract between training and prediction.
# All features are dimensionless/normalized so they are comparable across symbols
# at different price levels.  Raw price-unit columns (bb_upper, bb_lower, atr_14,
# momentum_10, volume_ma_20) and the raw SMA levels are excluded.
FEATURE_COLS: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_21d",            # 1-month return
    "vol_20d",
    "ma_spread_10_20",   # (sma10 - sma20) / close — normalized
    "ma_spread_20_50",   # (sma20 - sma50) / close — normalized
    "macd",
    "macd_signal",
    "rsi_14",
    "price_vs_sma20",
    "price_vs_sma50",
    "bb_width",          # (bb_upper - bb_lower) / close — normalized
    "bb_position",
    "stoch_k",
    "stoch_d",
    "roc_12",
    "atr_normalized",
    "adx_14",             # trend strength (0-100)
    "vol_regime",         # 20d vol / 63d vol
    "rel_volume",         # 5d avg vol / 20d avg vol
    "hl_ratio",           # (high - low) / close
    "turnover_z",         # z-score of close * volume
    "amihud_illiq",       # |return| / dollar_volume (price impact proxy), z-scored
    "gap",                # (open - prev_close) / prev_close
    "vpt_normalized",
    "ad_normalized",
    "obv_normalized",
    "ret_1d_vs_spy",     # symbol ret_1d minus SPY ret_1d (market-relative alpha, 1d)
    "ret_5d_vs_spy",     # symbol ret_5d minus SPY ret_5d (market-relative alpha, 5d)
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


_ZSCORE_WINDOW = 252
_ZSCORE_MIN_PERIODS = 60

# Unbounded / scale-varying features that get per-symbol rolling z-scoring.
# Bounded oscillators (rsi_14, stoch_k, stoch_d, adx_14, bb_position,
# atr_normalized) and already-z-scored features (turnover_z, bb_width,
# vpt/ad/obv_normalized) are intentionally excluded.
_ZSCORE_FEATURES: list[str] = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "vol_20d",
    "macd", "macd_signal", "ma_spread_10_20", "ma_spread_20_50",
    "price_vs_sma20", "price_vs_sma50", "roc_12", "gap", "hl_ratio",
    "vol_regime", "rel_volume", "amihud_illiq", "ret_1d_vs_spy", "ret_5d_vs_spy",
]


def _rolling_zscore(
    s: pd.Series, window: int = _ZSCORE_WINDOW, min_periods: int = _ZSCORE_MIN_PERIODS
) -> pd.Series:
    """Causal per-symbol z-score: (x - rolling_mean) / rolling_std. Uses past+present only."""
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    return (s - mean) / (std + 1e-12)


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


def make_daily_features(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build features from daily OHLCV data.

    spy_df: optional SPY DataFrame used to compute market-relative features
    (ret_1d_vs_spy, ret_5d_vs_spy).  Pass None for backtesting fallback — those
    features will be 0.0 so the column contract is always satisfied.

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

    if spy_df is not None:
        spy_ret_1d = spy_df["close"].pct_change(1).reindex(df.index)
        spy_ret_5d = spy_df["close"].pct_change(5).reindex(df.index)
        feats["ret_1d_vs_spy"] = feats["ret_1d"] - spy_ret_1d
        feats["ret_5d_vs_spy"] = feats["ret_5d"] - spy_ret_5d
    else:
        feats["ret_1d_vs_spy"] = 0.0
        feats["ret_5d_vs_spy"] = 0.0

    sma_10 = df["close"].rolling(10).mean()
    sma_20 = df["close"].rolling(20).mean()
    sma_50 = df["close"].rolling(50).mean()
    ema_12 = _safe_ema(df["close"], 12)
    ema_26 = _safe_ema(df["close"], 26)

    feats["ma_spread_10_20"] = (sma_10 - sma_20) / (df["close"] + 1e-12)
    feats["ma_spread_20_50"] = (sma_20 - sma_50) / (df["close"] + 1e-12)

    macd = ema_12 - ema_26
    signal = _safe_ema(macd, 9)
    feats["macd"] = macd
    feats["macd_signal"] = signal

    feats["rsi_14"] = _rsi(df["close"], 14)

    feats["price_vs_sma20"] = (df["close"] - sma_20) / (sma_20 + 1e-12)
    feats["price_vs_sma50"] = (df["close"] - sma_50) / (sma_50 + 1e-12)

    bb_mid = df["close"].rolling(20).mean()
    bb_std_dev = df["close"].rolling(20).std()
    _bb_upper = bb_mid + (bb_std_dev * 2)
    _bb_lower = bb_mid - (bb_std_dev * 2)
    _bb_width = _bb_upper - _bb_lower
    feats["bb_width"] = _bb_width / (df["close"] + 1e-12)
    feats["bb_position"] = (df["close"] - _bb_lower) / (_bb_width + 1e-12)

    lowest_low = df["low"].rolling(14).min()
    highest_high = df["high"].rolling(14).max()
    feats["stoch_k"] = 100.0 * (df["close"] - lowest_low) / (highest_high - lowest_low + 1e-12)
    feats["stoch_d"] = feats["stoch_k"].rolling(3).mean()

    feats["roc_12"] = (df["close"] - df["close"].shift(12)) / (df["close"].shift(12) + 1e-12)
    feats["atr_normalized"] = _atr(df, 14) / (df["close"] + 1e-12)

    # --- ADX (trend strength, 0-100) ---
    # Wilder's smoothing (ewm alpha=1/14) applied once to raw TR/DM, matching
    # the standard formula — _atr() already applies its own smoothing, so
    # reusing it here would double-smooth the DI+/DI- denominator.
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]), (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr_14_raw = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / (atr_14_raw + 1e-12))
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / (atr_14_raw + 1e-12))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    feats["adx_14"] = dx.ewm(alpha=1/14, adjust=False).mean()

    # --- Volatility regime ---
    vol_63d = df["close"].pct_change().rolling(63).std()
    feats["vol_regime"] = feats["vol_20d"] / (vol_63d + 1e-12)

    # --- Relative volume ---
    feats["rel_volume"] = df["volume"].rolling(5).mean() / (df["volume"].rolling(20).mean() + 1e-12)

    # --- High-low ratio ---
    feats["hl_ratio"] = (df["high"] - df["low"]) / (df["close"] + 1e-12)

    # --- Turnover z-score ---
    dollar_vol = df["close"] * df["volume"]
    feats["turnover_z"] = (dollar_vol - dollar_vol.rolling(20).mean()) / (dollar_vol.rolling(20).std() + 1e-12)

    # --- Amihud illiquidity: |return| per dollar traded (price impact) ---
    # Raw magnitude is tiny/heavy-tailed; the _ZSCORE_FEATURES loop normalizes it.
    feats["amihud_illiq"] = feats["ret_1d"].abs() / (dollar_vol + 1e-12)

    # --- Overnight gap ---
    feats["gap"] = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-12)

    feats["ret_21d"] = df["close"].pct_change(21)

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

    # The column name is fwd_ret_1d for backwards compatibility, but it is a
    # FWD_RET_HORIZON_DAYS-bar cumulative forward return — less noisy than a true
    # 1-day return. Consumers that pay this as per-bar PnL (e.g. rl_env, which
    # advances idx by 1 per step) must divide by FWD_RET_HORIZON_DAYS to avoid
    # inflating reward magnitude and double-counting overlapping windows.
    feats["fwd_ret_1d"] = (df["close"].shift(-FWD_RET_HORIZON_DAYS) / df["close"]) - 1

    for col in _ZSCORE_FEATURES:
        feats[col] = _rolling_zscore(feats[col])

    feats = feats.replace([np.inf, -np.inf], np.nan)
    # Drop warmup rows where rolling indicators are still NaN.
    # fwd_ret_1d is intentionally kept NaN for the last row so training code
    # can remove it with dropna(subset=["fwd_ret_1d"]).
    feats = feats.dropna(subset=FEATURE_COLS)
    return feats
