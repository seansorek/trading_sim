
# simulation_pipeline.py
import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict
import torch
from base_strategy import BaseStrategy, StrategyConfig
from daily_features import make_daily_features
from dqn_agent import DQNAgent
from data_loader import load_yfinance

# Try to import ML strategies (optional)
try:
    from ml_strategies import OrdinalLogisticStrategy, XGBoostStrategy, DailyLogisticStrategy, DailyXGBoostStrategy, DailyRNNStrategy
    HAS_ML_STRATEGIES = True
except ImportError as e:
    print(f"[warning] ML strategies not loaded. Please verify required ML libraries are installed. Error: {e}")
    HAS_ML_STRATEGIES = False

np.random.seed(42)
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

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


"""ML-only strategy registry and builder (non-ML strategies removed)."""

# Import hybrid strategies
try:
    from hybrid_strategy import HybridDQNXGBoostStrategy, EnsembleWeightedStrategy
    HAS_HYBRID = True
except ImportError as e:
    print(f"[warning] Hybrid strategies not loaded: {e}")
    HAS_HYBRID = False

STRATEGY_REGISTRY = {}
if HAS_ML_STRATEGIES:
    STRATEGY_REGISTRY["ordinal_logistic"] = OrdinalLogisticStrategy
    STRATEGY_REGISTRY["xgboost"] = XGBoostStrategy
    STRATEGY_REGISTRY["daily_logistic"] = DailyLogisticStrategy
    STRATEGY_REGISTRY["daily_xgboost"] = DailyXGBoostStrategy
    STRATEGY_REGISTRY["daily_rnn"] = DailyRNNStrategy
    # DQN strategy will be added below as a local class

if HAS_HYBRID:
    STRATEGY_REGISTRY["hybrid_dqn_xgboost"] = HybridDQNXGBoostStrategy
    STRATEGY_REGISTRY["ensemble_weighted"] = EnsembleWeightedStrategy

class DailyDQNStrategy(BaseStrategy):
    """
    Uses a pre-trained DQN agent to choose daily actions (Hold/Long/Short)
    based on a window of daily indicators. Includes confidence thresholding
    to reduce overtrading and improve quality.
    
    Improvements for profitability:
    - Higher confidence threshold to avoid weak signals
    - Feature scaling to match training distribution
    - Reduced trading frequency with stronger signal filtering
    """
    def __init__(self, cfg: StrategyConfig):
        super().__init__(cfg)
        self.model_path = os.environ.get('DQN_MODEL', getattr(cfg, 'model_path', 'models/dqn_agent.pt'))
        self.window = int(os.environ.get('DQN_WINDOW', getattr(cfg, 'window', 20)))
        # Confidence threshold - balance between signal quality and trading frequency
        self.confidence_threshold = float(os.environ.get('DQN_CONFIDENCE', '8.0'))
        # Q-value advantage threshold: best action must be this much better than hold
        self.q_advantage_threshold = float(os.environ.get('DQN_Q_ADVANTAGE', '1.5'))
        self.feature_scaler = {}  # Will be populated from training data statistics

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        try:
            agent = DQNAgent.load(self.model_path)
        except Exception as e:
            print(f"[warn] Failed to load DQN agent from {self.model_path}: {e}. Will default to Hold.")
            return pd.Series(0, index=df.index)

        # Build daily features aligned to df
        daily_feats = make_daily_features(df)
        daily_feats = daily_feats.fillna(0.0)

        # DQN was trained with ALL features including 'close' and 'fwd_ret_1d' excluded
        feature_cols = [c for c in daily_feats.columns if c not in ["fwd_ret_1d"]]
        
        # Normalize features using training statistics if available
        # This ensures the DQN sees features in the same distribution as training
        if not self.feature_scaler:
            for col in feature_cols:
                mu = float(daily_feats[col].mean())
                sigma = float(daily_feats[col].std() or 1.0)
                self.feature_scaler[col] = (mu, sigma)
        
        # Apply normalization
        daily_feats_normalized = daily_feats.copy()
        for col in feature_cols:
            if col in self.feature_scaler:
                mu, sigma = self.feature_scaler[col]
                daily_feats_normalized[col] = (daily_feats[col] - mu) / (sigma if sigma != 0 else 1.0)
        
        # Create rolling window states
        signals = []
        idxs = []
        for i in range(self.window, len(daily_feats_normalized)):
            frame = daily_feats_normalized.iloc[i-self.window:i]
            state = frame[feature_cols].values.astype(np.float32).flatten()
            
            # Get Q-values for all actions to assess confidence
            with torch.no_grad():
                s_t = torch.from_numpy(state).float().unsqueeze(0)
                q_vals = agent.q(s_t).squeeze(0).cpu().numpy()
            
            # Get Q-values for all actions to assess confidence
            with torch.no_grad():
                s_t = torch.from_numpy(state).float().unsqueeze(0)
                q_vals = agent.q(s_t).squeeze(0).cpu().numpy()
            
            # Action-space: 0=Hold, 1=Long, 2=Short
            q_hold = q_vals[0]
            q_long = q_vals[1]
            q_short = q_vals[2]
            q_max = q_vals.max()
            q_min = q_vals.min()
            confidence = q_max - q_min
            
            sig = 0
            
            # Only trade if confident AND best action is significantly better than hold
            if confidence >= self.confidence_threshold:
                # Long signal: if long is best AND significantly better than hold
                if q_long == q_max and q_long - q_hold > self.q_advantage_threshold:
                    sig = 1
                # Short signal: if short is best AND significantly better than hold
                elif q_short == q_max and q_short - q_hold > self.q_advantage_threshold:
                    sig = -1
                # Otherwise hold (even if max confidence, protect against weak signals)
                else:
                    sig = 0
            
            signals.append(sig)
            idxs.append(daily_feats.index[i])

        if not signals:
            return pd.Series(0, index=df.index)
        ser = pd.Series(signals, index=pd.DatetimeIndex(idxs))
        # Reindex to df and fill missing with 0 (Hold)
        ser = ser.reindex(df.index).fillna(0).astype(int)
        # Apply holding period to reduce reversals and transaction costs
        return self._apply_holding_period(ser)

def build_strategy_signal(strategy_name: str,
                          cfg: StrategyConfig,
                          feats: pd.DataFrame,
                          df: pd.DataFrame,
                          predict_only: bool = True,
                          **kwargs) -> pd.Series:
    name = strategy_name.lower()
    if name not in STRATEGY_REGISTRY:
        # Allow DQN strategy even if ML_STRATEGIES missing
        if name == "daily_dqn":
            STRATEGY_REGISTRY["daily_dqn"] = DailyDQNStrategy
        else:
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {list(STRATEGY_REGISTRY.keys())}")
    strat_cls = STRATEGY_REGISTRY[name]
    # For daily strategies, we don't rely on intraday features passed in
    # Pass predict_only to ML strategies that support it
    try:
        strat = strat_cls(cfg, predict_only=predict_only)
    except TypeError:
        # Fallback for strategies that don't accept predict_only
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
