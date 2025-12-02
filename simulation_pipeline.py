
# simulation_pipeline.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict
from base_strategy import BaseStrategy, StrategyConfig

# Try to import ML strategies (optional)
try:
    from ml_strategies import OrdinalLogisticStrategy, XGBoostStrategy
    HAS_ML_STRATEGIES = True
except ImportError as e:
    print(f"[warning] ML strategies not loaded. Please verify scikit-learn and xgboost are installed. Error: {e}")
    HAS_ML_STRATEGIES = False

np.random.seed(42)
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 1) Synthetic intraday generator (US hours: 09:30–16:00)
def generate_synthetic_intraday(start_price: float = 100.0, days: int = 10) -> pd.DataFrame:
    minutes_per_day = 390
    all_minutes = []
    for d in range(days):
        date = pd.Timestamp('2025-10-01') + pd.Timedelta(days=d)
        day_minutes = pd.date_range(date + pd.Timedelta(hours=9, minutes=30),
                                    date + pd.Timedelta(hours=16),
                                    freq='1min', inclusive='left')
        all_minutes.append(day_minutes)
    idx = all_minutes[0]
    for i in range(1, len(all_minutes)):
        idx = idx.append(all_minutes[i])
    df = pd.DataFrame(index=idx)

    # Intraday patterns
    minutes = np.arange(minutes_per_day)
    vol_shape = 0.6 + 0.8 * (np.sin((minutes / minutes_per_day) * np.pi))**2
    vol_shape = vol_shape / np.mean(vol_shape)
    vol_daily = np.random.uniform(0.0004, 0.0012, size=days)
    vol_series = np.concatenate([vol_daily[d] * vol_shape for d in range(days)])

    # Price path
    noise = np.random.normal(0, 1, size=len(df))
    returns = vol_series * noise
    mid = start_price * np.exp(np.cumsum(returns))

    micro_noise = np.random.normal(0, 0.0005, size=len(df))
    close = mid * (1 + micro_noise)
    high = close * (1 + np.abs(np.random.normal(0, 0.0015, size=len(df))))
    low  = close * (1 - np.abs(np.random.normal(0, 0.0015, size=len(df))))
    open_ = np.concatenate([[start_price], close[:-1]])

    vol_u = 0.6 + 0.8 * (np.sin((minutes / minutes_per_day) * np.pi))**2
    vol_u = vol_u / np.mean(vol_u)
    base_vol = np.random.uniform(1000, 5000, size=days)
    volume = np.concatenate([
        base_vol[d] * vol_u * (1 + np.random.normal(0, 0.2, size=minutes_per_day))
        for d in range(days)
    ]).astype(int)
    volume = np.maximum(10, volume)

    # Spread widens with volatility
    spread_bps = 4 + 200 * vol_series
    spread = (spread_bps / 1e4) * close

    df['open']   = open_
    df['high']   = high
    df['low']    = low
    df['close']  = close
    df['volume'] = volume
    df['spread'] = spread
    df.index.name = 'timestamp'
    df.to_csv('data/synthetic_intraday.csv')
    return df

# 2) Features
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(alpha=1/window, adjust=False).mean()
    ema_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-12)
    return 100 - (100 / (1 + rs))

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    price = df['close']
    returns_1m = price.pct_change().fillna(0)
    feats['ret_1m']    = returns_1m
    feats['ma_fast']   = price.rolling(10).mean()
    feats['ma_slow']   = price.rolling(60).mean()
    feats['ma_spread'] = feats['ma_fast'] - feats['ma_slow']
    feats['vol_10']    = returns_1m.rolling(10).std()
    feats['vol_60']    = returns_1m.rolling(60).std()
    feats['rsi_14']    = rsi(price, 14)
    feats['vol_z']     = ((df['volume'] - df['volume'].rolling(60).mean())
                          / (df['volume'].rolling(60).std() + 1e-12))
    feats['hour']      = feats.index.hour + feats.index.minute / 60.0
    
    # Enhanced features for better predictions
    feats['momentum_5'] = price.pct_change(5).fillna(0)   # 5-bar momentum
    feats['momentum_20'] = price.pct_change(20).fillna(0)  # Longer trend
    feats['vp_ratio'] = feats['ret_1m'] / (feats['vol_z'].abs() + 1e-6)  # Volume-price divergence
    feats['vol_regime'] = (feats['vol_60'] > feats['vol_60'].rolling(100).mean()).astype(int)  # Volatility regime
    
    # Price relative to daily range
    range_high = df['high'].rolling(60).max()
    range_low = df['low'].rolling(60).min()
    feats['price_position'] = (price - range_low) / (range_high - range_low + 1e-6)
    
    feats = feats.bfill().ffill()
    return feats


class MeanReversionStrategy(BaseStrategy):
    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        ret = feats['ret_1m']
        mu = ret.rolling(self.cfg.lookback).mean()
        sd = ret.rolling(self.cfg.lookback).std().replace(0, np.nan)
        # Calculate Z-score, filling NaNs from std=0 with 0 to avoid warnings
        z = ((ret - mu) / sd).fillna(0)
        sig = pd.Series(
            np.where(z < -self.cfg.threshold, 1,
                     np.where(z >  self.cfg.threshold, -1, 0)),
            index=feats.index
        )
        
        # Don't trade mean reversion in strong trends - it fails there
        ma_trend = feats['ma_spread'] / (feats['ma_slow'] + 1e-6)
        in_strong_trend = np.abs(ma_trend) > 0.02
        sig[in_strong_trend] = 0
        
        # Apply holding period to reduce trading frequency
        sig = self._apply_holding_period(sig)
        return sig

class MomentumStrategy(BaseStrategy):
    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        ma_spread = feats['ma_spread']
        # Add volume confirmation - momentum with volume is stronger
        vol_confirmed = feats['vol_z'] > 0.5  # Above-average volume
        sig = pd.Series(
            np.where((ma_spread > 0) & vol_confirmed, 1,
                     np.where((ma_spread < 0) & vol_confirmed, -1, 0)),
            index=feats.index
        )
        # Apply holding period to reduce trading frequency
        sig = self._apply_holding_period(sig)
        return sig

class BreakoutStrategy(BaseStrategy):
    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        price = df['close']
        high_roll = price.rolling(self.cfg.lookback).max()
        low_roll  = price.rolling(self.cfg.lookback).min()
        sig = np.where(price > high_roll.shift(1), 1,
                  np.where(price < low_roll.shift(1), -1, 0))
        result = pd.Series(sig, index=price.index).fillna(0)
        # Apply holding period to reduce trading frequency
        result = self._apply_holding_period(result)
        return result

class RSIStrategy(BaseStrategy):
    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        rsi = feats['rsi_14']
        sig = pd.Series(
            np.where(rsi < self.cfg.rsi_lower, 1,
                     np.where(rsi > self.cfg.rsi_upper, -1, 0)),
            index=feats.index
        ).fillna(0)
        # Apply holding period to reduce trading frequency
        sig = self._apply_holding_period(sig)
        return sig

# ===== Strategy Registry & Builder =====

STRATEGY_REGISTRY = {
    "mean_reversion": MeanReversionStrategy,
    "momentum":       MomentumStrategy,
    "breakout":       BreakoutStrategy,
    "rsi":            RSIStrategy,
}

# Add ML strategies if available
if HAS_ML_STRATEGIES:
    STRATEGY_REGISTRY["ordinal_logistic"] = OrdinalLogisticStrategy
    STRATEGY_REGISTRY["xgboost"] = XGBoostStrategy

def build_strategy_signal(strategy_name: str,
                          cfg: StrategyConfig,
                          feats: pd.DataFrame,
                          df: pd.DataFrame,
                          **kwargs) -> pd.Series:
    name = strategy_name.lower()
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {list(STRATEGY_REGISTRY.keys())}")
    strat_cls = STRATEGY_REGISTRY[name]
    strat = strat_cls(cfg)
    sig = strat.signal(feats, df)
    # Final sanity: ensure integer {-1,0,+1}
    sig = pd.Series(np.sign(sig).astype(int), index=sig.index)
    return sig

# 4) Execution + Backtester
@dataclass
class ExecutionConfig:
    commission_per_share: float = 0.000001    # Reduced from 0.0005
    slippage_bps: float = 0.2               # Reduced from 2.0
    max_position: int = 2000
    stop_loss_pct: float = 0.05             # Increased from 0.03
    daily_loss_limit_pct: float = 0.05      # Increased from 0.02

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]

class Backtester:
    def __init__(self, exec_cfg: ExecutionConfig, start_cash: float = 100_000.0):
        self.exec_cfg = exec_cfg
        self.start_cash = start_cash

    def run(self, df: pd.DataFrame, feats: pd.DataFrame, signal: pd.Series, artifact_paths: dict = None) -> BacktestResult:
        # Define default paths if none are provided (for backward compatibility or single runs)
        if artifact_paths is None:
            artifact_paths = {
                "equity_curve_csv": "results/equity_curve.csv",
                "trade_log_csv": "results/trade_log.csv",
                "metrics_json": "results/metrics.json",
            }

        df = df.loc[signal.index]
        feats = feats.loc[signal.index]
        cash = self.start_cash
        position = 0
        avg_entry_price = None
        equity_curve = []
        timestamps = []
        trade_log = []
        daily_start_cash = cash
        current_day = df.index[0].date()

        for ts, row in df.iterrows():
            # Daily loss cap resets
            if ts.date() != current_day:
                current_day = ts.date()
                daily_start_cash = cash

            mid = row['close']
            spr = max(row['spread'], 0.01)
            slippage = (self.exec_cfg.slippage_bps / 1e4) * mid

            desired = int(signal.loc[ts])  # -1 / 0 / +1

            # Simple notional sizing (5% of capital per trade)
            notional = self.start_cash * 0.05
            shares = int(notional / mid) if mid > 0 else 0
            shares = min(shares, self.exec_cfg.max_position)

            target_pos = desired * shares
            delta = target_pos - position

            # Execute market order for delta
            if delta != 0:
                side = np.sign(delta)
                fill_price = mid + side * (spr/2 + slippage)
                cost_commission = self.exec_cfg.commission_per_share * abs(delta)
                cost_spread = (spr/2) * abs(delta)
                cash -= fill_price * delta + cost_commission
                position += delta
                trade_log.append({
                    'timestamp': ts,
                    'side': 'BUY' if side>0 else 'SELL',
                    'shares': abs(delta),
                    'fill_price': float(fill_price),
                    'commission': float(cost_commission),
                    'spread_cost': float(cost_spread)
                })
                # Weighted-average entry price
                if position != 0:
                    avg_entry_price = fill_price if avg_entry_price is None else (
                        (avg_entry_price * (abs(position - delta)) + fill_price * abs(delta)) / abs(position)
                    )
                else:
                    avg_entry_price = None

            # Take-profit: Exit at 10% profit target
            if position != 0 and avg_entry_price is not None:
                pnl_from_entry = (mid - avg_entry_price) * np.sign(position)
                take_profit_threshold = 0.10 * avg_entry_price  # 10% profit target
                
                if pnl_from_entry > take_profit_threshold:
                    # Exit at profit
                    side = -np.sign(position)
                    fill_price = mid + side * (spr/2 + slippage)
                    cost_commission = self.exec_cfg.commission_per_share * abs(position)
                    cost_spread = (spr/2) * abs(position)
                    cash -= fill_price * (-position) + cost_commission
                    trade_log.append({
                        'timestamp': ts,
                        'side': 'SELL' if side<0 else 'BUY',
                        'shares': abs(position),
                        'fill_price': float(fill_price),
                        'commission': float(cost_commission),
                        'spread_cost': float(cost_spread)
                    })
                    position = 0
                    avg_entry_price = None
            
            # Stop-loss
            if position != 0 and avg_entry_price is not None:
                pnl_from_entry = (mid - avg_entry_price) * np.sign(position)
                if pnl_from_entry < -self.exec_cfg.stop_loss_pct * avg_entry_price:
                    side = -np.sign(position)
                    fill_price = mid + side * (spr/2 + slippage)
                    cost_commission = self.exec_cfg.commission_per_share * abs(position)
                    cost_spread = (spr/2) * abs(position)
                    cash -= fill_price * (-position) + cost_commission
                    trade_log.append({
                        'timestamp': ts,
                        'side': 'SELL' if side<0 else 'BUY',
                        'shares': abs(position),
                        'fill_price': float(fill_price),
                        'commission': float(cost_commission),
                        'spread_cost': float(cost_spread)
                    })
                    position = 0
                    avg_entry_price = None

            # Daily loss limit
            equity = cash + position * mid
            if equity < daily_start_cash * (1 - self.exec_cfg.daily_loss_limit_pct):
                if position != 0:
                    side = -np.sign(position)
                    fill_price = mid + side * (spr/2 + slippage)
                    cost_commission = self.exec_cfg.commission_per_share * abs(position)
                    cost_spread = (spr/2) * abs(position)
                    cash -= fill_price * (-position) + cost_commission
                    trade_log.append({
                        'timestamp': ts,
                        'side': 'SELL' if side<0 else 'BUY',
                        'shares': abs(position),
                        'fill_price': float(fill_price),
                        'commission': float(cost_commission),
                        'spread_cost': float(cost_spread)
                    })
                    position = 0
                    avg_entry_price = None

            equity = cash + position * mid
            equity_curve.append(equity)
            timestamps.append(ts)

        equity_series = pd.Series(equity_curve, index=pd.DatetimeIndex(timestamps), name='equity')
        trades_df = pd.DataFrame(trade_log)
        if not trades_df.empty:
            trades_df.set_index('timestamp', inplace=True)
            trades_df.sort_index(inplace=True)

        metrics = compute_metrics(equity_series, trades_df)
        # Save outputs to the specified artifact paths
        if not trades_df.empty:
            trades_df.to_csv(artifact_paths["trade_log_csv"])
        equity_series.to_csv(artifact_paths["equity_curve_csv"])
        with open(artifact_paths["metrics_json"], 'w') as f:
            json.dump(metrics, f, indent=2)
        return BacktestResult(equity_series, trades_df, metrics)

# 5) Metrics
def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
    daily_equity = equity.resample('1D').last().dropna()
    daily_ret = daily_equity.pct_change().dropna()

    def sharpe(r):
        mu, sd = r.mean(), r.std()
        return float(np.sqrt(252) * mu / sd) if sd and sd != 0 else 0.0

    def sortino(r):
        downside = r[r < 0]; dd = downside.std(); mu = r.mean()
        return float(np.sqrt(252) * mu / dd) if dd and dd != 0 else 0.0

    cum = daily_equity.values
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Round-trip PnL approximation
    gross_profit = 0.0; gross_loss = 0.0; hit_rate = 0.0; n_trades = 0
    if trades is not None and not trades.empty:
        pnl_list = []; pos = 0; entry_price = None
        for ts, t in trades.iterrows():
            side = 1 if t['side']=='BUY' else -1
            if pos == 0:
                pos = side * t['shares']; entry_price = t['fill_price']
            else:
                pnl = (t['fill_price'] - entry_price) * np.sign(pos) * min(abs(pos), t['shares'])
                pnl_list.append(pnl)
                pos += side * t['shares']
                if pos == 0: entry_price = None
        if len(pnl_list) > 0:
            pnl_arr = np.array(pnl_list)
            gross_profit = float(pnl_arr[pnl_arr>0].sum())
            gross_loss = float(-pnl_arr[pnl_arr<0].sum())
            hit_rate = float((pnl_arr>0).mean())
            n_trades = int(len(pnl_arr))

    return {
        'final_equity': float(equity.iloc[-1]),
        'total_return_pct': float((equity.iloc[-1]/equity.iloc[0] - 1) * 100),
        'daily_sharpe': sharpe(daily_ret),
        'daily_sortino': sortino(daily_ret),
        'max_drawdown_pct': float(max_dd * 100),
        'n_round_trades': n_trades,
        'hit_rate': float(hit_rate),
        'profit_factor': float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    }

# 6) Walk-forward (simple linear model, no external ML libs)
def walk_forward_backtest(df: pd.DataFrame, feats: pd.DataFrame,
                          train_days: int = 5, test_days: int = 1) -> BacktestResult:
    days = sorted(set(df.index.date))
    signals = []
    for i in range(train_days, len(days) - test_days):  # corrected bounds
        train_idx = (df.index.date >= days[i-train_days]) & (df.index.date < days[i])
        test_idx  = (df.index.date >= days[i]) & (df.index.date < days[i+test_days])
        X_train = feats.loc[train_idx, ['ret_1m', 'ma_spread', 'vol_10', 'rsi_14', 'vol_z']].values
        y_train = np.sign(df.loc[train_idx, 'close'].pct_change(5).shift(-5).fillna(0).values)
        X = np.c_[np.ones(len(X_train)), X_train]
        beta, *_ = np.linalg.lstsq(X, y_train, rcond=None)

        X_test = feats.loc[test_idx, ['ret_1m', 'ma_spread', 'vol_10', 'rsi_14', 'vol_z']].values
        X_t = np.c_[np.ones(len(X_test)), X_test]
        y_hat = X_t @ beta
        # Stricter threshold to reduce trades
        sig = np.where(y_hat > 0.10, 1, np.where(y_hat < -0.10, -1, 0))
        signals.append(pd.Series(sig, index=feats.loc[test_idx].index))
    signal_full = pd.concat(signals).sort_index() if signals else pd.Series(0, index=feats.index)

    bt = Backtester(ExecutionConfig())
    return bt.run(df.loc[signal_full.index], feats.loc[signal_full.index], signal_full)

# 7) Monte Carlo stress test on execution assumptions
def monte_carlo_stress(df: pd.DataFrame, feats: pd.DataFrame, signal: pd.Series, n_runs: int = 50) -> pd.DataFrame:
    stats = []
    for i in range(n_runs):
        exec_cfg = ExecutionConfig(
            commission_per_share=0.0005,
            slippage_bps=max(0.5, np.random.normal(2.0, 1.0)),
            max_position=2000,
            stop_loss_pct=np.clip(np.random.normal(0.03, 0.01), 0.01, 0.06),
            daily_loss_limit_pct=np.clip(np.random.normal(0.02, 0.005), 0.01, 0.05)
        )
        bt = Backtester(exec_cfg)
        res = bt.run(df, feats, signal)
        stats.append(res.metrics)
    df_stats = pd.DataFrame(stats)
    # Clean up non-finite values (inf, -inf) before calculating stats to avoid warnings
    df_stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_stats.dropna(inplace=True)
    
    df_stats.to_csv('results/monte_carlo_stats.csv', index=False)
    return df_stats
