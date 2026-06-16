"""
simulation_pipeline.py — Backtesting engine for daily trading strategies.
"""
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from base_strategy import BaseStrategy, StrategyConfig
from daily_features import FEATURE_COLS, make_daily_features
from dqn_agent import DQNAgent
from data_loader import load_yfinance

logger = logging.getLogger(__name__)

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Import ML strategies
# ---------------------------------------------------------------------------
try:
    from ml_strategies import (
        OrdinalLogisticStrategy,
        XGBoostStrategy,
        DailyLogisticStrategy,
        DailyXGBoostStrategy,
    )
    HAS_ML_STRATEGIES = True
except ImportError as exc:
    logger.warning("ML strategies not loaded: %s", exc)
    HAS_ML_STRATEGIES = False

# ---------------------------------------------------------------------------
# Intraday feature builder (kept for walk-forward; not used by daily strategies)
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(alpha=1 / window, adjust=False).mean()
    ema_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-12)
    return 100 - (100 / (1 + rs))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Intraday feature builder. Used by walk_forward_backtest and DQN training."""
    feats = pd.DataFrame(index=df.index)
    price = df["close"]
    returns_1m = price.pct_change().fillna(0)
    feats["ret_1m"] = returns_1m
    feats["ma_fast"] = price.rolling(10).mean()
    feats["ma_slow"] = price.rolling(60).mean()
    feats["ma_spread"] = feats["ma_fast"] - feats["ma_slow"]
    feats["vol_10"] = returns_1m.rolling(10).std()
    feats["vol_60"] = returns_1m.rolling(60).std()
    feats["rsi_14"] = rsi(price, 14)
    feats["vol_z"] = (
        (df["volume"] - df["volume"].rolling(60).mean())
        / (df["volume"].rolling(60).std() + 1e-12)
    )
    feats["momentum_5"] = price.pct_change(5).fillna(0)
    feats["momentum_20"] = price.pct_change(20).fillna(0)
    feats["vp_ratio"] = feats["ret_1m"] / (feats["vol_z"].abs() + 1e-6)
    feats["vol_regime"] = (feats["vol_60"] > feats["vol_60"].rolling(100).mean()).astype(int)
    range_high = df["high"].rolling(60).max()
    range_low = df["low"].rolling(60).min()
    feats["price_position"] = (price - range_low) / (range_high - range_low + 1e-6)
    feats = feats.bfill().ffill()
    return feats


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict = {}
if HAS_ML_STRATEGIES:
    STRATEGY_REGISTRY["daily_logistic"] = DailyLogisticStrategy
    STRATEGY_REGISTRY["daily_xgboost"] = DailyXGBoostStrategy


class DailyDQNStrategy(BaseStrategy):
    """DQN agent generating daily long/short/hold signals."""

    def __init__(self, cfg: StrategyConfig):
        super().__init__(cfg)
        self.model_path = os.environ.get(
            "DQN_MODEL", getattr(cfg, "model_path", "models/dqn_agent.pt")
        )
        self.window = int(os.environ.get("DQN_WINDOW", getattr(cfg, "window", 20)))
        self.confidence_threshold = float(os.environ.get("DQN_CONFIDENCE", "8.0"))
        self.q_advantage_threshold = float(os.environ.get("DQN_Q_ADVANTAGE", "1.5"))

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        try:
            agent = DQNAgent.load(self.model_path)
        except Exception as exc:
            logger.warning("DQN: failed to load agent from %s: %s", self.model_path, exc)
            return pd.Series(0, index=df.index)

        daily_feats = make_daily_features(df).fillna(0.0)
        feature_cols = FEATURE_COLS

        # Fit per-run normalizer from available data
        scaler_stats: dict = {}
        for col in feature_cols:
            mu = float(daily_feats[col].mean())
            sigma = float(daily_feats[col].std() or 1.0)
            scaler_stats[col] = (mu, sigma)

        normalized = daily_feats.copy()
        for col in feature_cols:
            mu, sigma = scaler_stats[col]
            normalized[col] = (daily_feats[col] - mu) / (sigma if sigma != 0 else 1.0)

        signals, idxs = [], []
        for i in range(self.window, len(normalized)):
            state = (
                normalized.iloc[i - self.window : i][feature_cols]
                .values.astype(np.float32)
                .flatten()
            )

            with torch.no_grad():
                s_t = torch.from_numpy(state).float().unsqueeze(0)
                q_vals = agent.q(s_t).squeeze(0).cpu().numpy()

            q_hold, q_long, q_short = q_vals[0], q_vals[1], q_vals[2]
            q_max, q_min = q_vals.max(), q_vals.min()
            confidence = q_max - q_min

            sig = 0
            if confidence >= self.confidence_threshold:
                if q_long == q_max and q_long - q_hold > self.q_advantage_threshold:
                    sig = 1
                elif q_short == q_max and q_short - q_hold > self.q_advantage_threshold:
                    sig = -1

            signals.append(sig)
            idxs.append(daily_feats.index[i])

        if not signals:
            return pd.Series(0, index=df.index)

        ser = pd.Series(signals, index=pd.DatetimeIndex(idxs))
        ser = ser.reindex(df.index).fillna(0).astype(int)
        return self._apply_holding_period(ser)


STRATEGY_REGISTRY["daily_dqn"] = DailyDQNStrategy


def build_strategy_signal(
    strategy_name: str,
    cfg: StrategyConfig,
    feats: pd.DataFrame,
    df: pd.DataFrame,
    **kwargs,
) -> pd.Series:
    name = strategy_name.lower()
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. Available: {list(STRATEGY_REGISTRY)}"
        )
    strat_cls = STRATEGY_REGISTRY[name]
    try:
        strat = strat_cls(cfg, **kwargs)
    except TypeError:
        strat = strat_cls(cfg)
    sig = strat.signal(feats, df)
    return pd.Series(np.sign(sig).astype(int), index=sig.index)


# ---------------------------------------------------------------------------
# Execution model
# ---------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    start_cash: float = 100_000.0
    commission_per_share: float = 0.00005
    slippage_bps: float = 1.0
    max_position: int = 2000
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    daily_loss_limit_pct: float = 0.05
    max_position_pct: float = 0.05
    holding_period_days: int = 0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]


def _exit_position(
    position: int,
    avg_entry_price: float,
    mid: float,
    spr: float,
    slippage: float,
    exec_cfg: ExecutionConfig,
) -> Optional[tuple]:
    """
    Check stop-loss and take-profit exits.

    Returns (fill_price, cost_commission, cost_spread, reason) if exit triggered,
    else None.
    """
    if position == 0 or avg_entry_price is None:
        return None

    pnl_pct = (mid - avg_entry_price) / avg_entry_price * np.sign(position)

    if pnl_pct > exec_cfg.take_profit_pct:
        reason = "take_profit"
    elif pnl_pct < -exec_cfg.stop_loss_pct:
        reason = "stop_loss"
    else:
        return None

    side = -np.sign(position)
    fill_price = mid + side * (spr / 2 + slippage)
    cost_commission = exec_cfg.commission_per_share * abs(position)
    cost_spread = (spr / 2) * abs(position)
    return fill_price, cost_commission, cost_spread, reason


class Backtester:
    def __init__(self, exec_cfg: ExecutionConfig):
        self.exec_cfg = exec_cfg

    def run(
        self,
        df: pd.DataFrame,
        feats: pd.DataFrame,
        signal: pd.Series,
        artifact_paths: Optional[dict] = None,
        db=None,
        run_id: Optional[str] = None,
        symbol: str = "",
        strategy: str = "",
    ) -> BacktestResult:
        # Deterministic execution
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)

        if artifact_paths is None:
            artifact_paths = {
                "equity_curve_csv": "results/equity_curve.csv",
                "trade_log_csv": "results/trade_log.csv",
                "metrics_json": "results/metrics.json",
            }

        df = df.loc[signal.index]
        feats = feats.reindex(signal.index)

        cash = self.exec_cfg.start_cash
        position = 0
        avg_entry_price: Optional[float] = None
        equity_curve: list[float] = []
        timestamps: list = []
        trade_log: list[dict] = []
        daily_start_cash = cash
        current_day = df.index[0].date() if len(df) > 0 else None
        prev_close: Optional[float] = None

        for ts, row in df.iterrows():
            if ts.date() != current_day:
                current_day = ts.date()
                # Use closing equity from the previous bar (cash + position * prior close)
                # so the daily loss limit measures a full day's loss, not just intrabar.
                if prev_close is not None:
                    daily_start_cash = cash + position * prev_close
                else:
                    daily_start_cash = cash

            mid = float(row["close"])
            if mid <= 0:
                equity_curve.append(cash + position * mid)
                timestamps.append(ts)
                continue

            # Spread: use row value if present (intraday), else estimate from high-low
            raw_spr = row.get("spread", (row.get("high", mid) - row.get("low", mid)) * 0.05)
            spr = max(float(raw_spr), 0.01)
            slippage = (self.exec_cfg.slippage_bps / 1e4) * mid

            desired = int(signal.loc[ts])

            notional = self.exec_cfg.start_cash * self.exec_cfg.max_position_pct
            shares = int(notional / mid) if mid > 0 else 0
            shares = min(shares, self.exec_cfg.max_position)

            target_pos = desired * shares
            delta = target_pos - position

            if delta != 0:
                side = int(np.sign(delta))
                fill_price = mid + side * (spr / 2 + slippage)
                cost_commission = self.exec_cfg.commission_per_share * abs(delta)
                cost_spread = (spr / 2) * abs(delta)
                cash -= fill_price * delta + cost_commission
                old_position = position
                position += delta

                # Weighted-average entry price — branch on trade type
                if position == 0:
                    avg_entry_price = None
                elif old_position == 0 or np.sign(old_position) != np.sign(position):
                    # New position or full reversal: reset to current fill
                    avg_entry_price = fill_price
                else:
                    # Adding to existing position: weighted average
                    old_shares = abs(old_position)
                    new_shares = abs(delta)
                    total_shares = abs(position)
                    avg_entry_price = (
                        (avg_entry_price * old_shares + fill_price * new_shares) / total_shares
                    )

                trade_log.append({
                    "ts": ts,
                    "side": "BUY" if side > 0 else "SELL",
                    "shares": abs(delta),
                    "fill_price": float(fill_price),
                    "commission": float(cost_commission),
                    "spread_cost": float(cost_spread),
                    "exit_reason": "signal",
                })

            # Stop-loss / take-profit check
            exit_result = _exit_position(
                position, avg_entry_price, mid, spr, slippage, self.exec_cfg
            )
            if exit_result is not None:
                fill_price, cost_commission, cost_spread, reason = exit_result
                side = int(-np.sign(position))
                cash -= fill_price * (-position) + cost_commission
                trade_log.append({
                    "ts": ts,
                    "side": "SELL" if side < 0 else "BUY",
                    "shares": abs(position),
                    "fill_price": float(fill_price),
                    "commission": float(cost_commission),
                    "spread_cost": float(cost_spread),
                    "exit_reason": reason,
                })
                position = 0
                avg_entry_price = None

            # Daily loss limit
            equity = cash + position * mid
            if equity < daily_start_cash * (1 - self.exec_cfg.daily_loss_limit_pct):
                if position != 0:
                    side = int(-np.sign(position))
                    fill_price = mid + side * (spr / 2 + slippage)
                    cost_commission = self.exec_cfg.commission_per_share * abs(position)
                    cost_spread = (spr / 2) * abs(position)
                    cash -= fill_price * (-position) + cost_commission
                    trade_log.append({
                        "ts": ts,
                        "side": "SELL" if side < 0 else "BUY",
                        "shares": abs(position),
                        "fill_price": float(fill_price),
                        "commission": float(cost_commission),
                        "spread_cost": float(cost_spread),
                        "exit_reason": "daily_limit",
                    })
                    position = 0
                    avg_entry_price = None

            equity = cash + position * mid
            equity_curve.append(equity)
            timestamps.append(ts)
            prev_close = mid

        equity_series = pd.Series(
            equity_curve, index=pd.DatetimeIndex(timestamps), name="equity"
        )
        trades_df = pd.DataFrame(trade_log)
        if not trades_df.empty and "ts" in trades_df.columns:
            trades_df = trades_df.set_index("ts").sort_index()

        metrics = compute_metrics(equity_series, trades_df)

        # Persist artifacts
        try:
            os.makedirs(os.path.dirname(artifact_paths["trade_log_csv"]), exist_ok=True)
            if not trades_df.empty:
                trades_df.to_csv(artifact_paths["trade_log_csv"])
            equity_series.to_csv(artifact_paths["equity_curve_csv"])
            with open(artifact_paths["metrics_json"], "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to write artifacts: %s", exc)

        # DB logging (optional)
        if db is not None and run_id is not None:
            try:
                data_start = str(df.index[0].date()) if len(df) > 0 else ""
                data_end = str(df.index[-1].date()) if len(df) > 0 else ""
                run_row_id = db.insert_backtest_run(
                    run_id=run_id,
                    symbol=symbol,
                    strategy=strategy,
                    data_start=data_start,
                    data_end=data_end,
                    start_cash=self.exec_cfg.start_cash,
                    final_equity=metrics.get("final_equity"),
                    total_return_pct=metrics.get("total_return_pct"),
                    daily_sharpe=metrics.get("daily_sharpe"),
                    daily_sortino=metrics.get("daily_sortino"),
                    max_drawdown_pct=metrics.get("max_drawdown_pct"),
                    n_round_trades=metrics.get("n_round_trades"),
                    hit_rate=metrics.get("hit_rate"),
                    profit_factor=metrics.get("profit_factor"),
                    commission_per_share=self.exec_cfg.commission_per_share,
                    slippage_bps=self.exec_cfg.slippage_bps,
                    stop_loss_pct=self.exec_cfg.stop_loss_pct,
                    holding_period=self.exec_cfg.holding_period_days,
                )
                if not trades_df.empty:
                    trades_for_db = trades_df.reset_index().rename(columns={"ts": "ts"})
                    trades_for_db["symbol"] = symbol
                    trades_for_db["strategy"] = strategy
                    db.insert_trades(run_row_id, trades_for_db)
            except Exception as exc:
                logger.warning("DB logging failed: %s", exc)

        return BacktestResult(equity_series, trades_df, metrics)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
    daily_equity = equity.resample("1D").last().dropna()
    daily_ret = daily_equity.pct_change().dropna()

    def sharpe(r: pd.Series) -> float:
        mu, sd = r.mean(), r.std()
        return float(np.sqrt(252) * mu / sd) if sd and sd != 0 else 0.0

    def sortino(r: pd.Series) -> float:
        downside = r[r < 0]
        dd = downside.std()
        mu = r.mean()
        return float(np.sqrt(252) * mu / dd) if dd and dd != 0 else 0.0

    cum = daily_equity.values
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / (peak + 1e-12)
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    gross_profit = 0.0
    gross_loss = 0.0
    hit_rate = 0.0
    n_trades = 0

    if trades is not None and not trades.empty:
        pnl_list: list[float] = []
        pos = 0
        entry_price: Optional[float] = None
        for ts, t in trades.iterrows():
            side = 1 if t["side"] == "BUY" else -1
            if pos == 0:
                pos = side * t["shares"]
                entry_price = t["fill_price"]
            else:
                pnl = (
                    (t["fill_price"] - entry_price)
                    * np.sign(pos)
                    * min(abs(pos), t["shares"])
                )
                pnl_list.append(float(pnl))
                pos += side * t["shares"]
                if pos == 0:
                    entry_price = None
        if pnl_list:
            pnl_arr = np.array(pnl_list)
            gross_profit = float(pnl_arr[pnl_arr > 0].sum())
            gross_loss = float(-pnl_arr[pnl_arr < 0].sum())
            hit_rate = float((pnl_arr > 0).mean())
            n_trades = int(len(pnl_arr))

    start_eq = float(equity.iloc[0]) if len(equity) > 0 else 1.0
    end_eq = float(equity.iloc[-1]) if len(equity) > 0 else start_eq

    return {
        "final_equity": end_eq,
        "total_return_pct": float((end_eq / start_eq - 1) * 100),
        "daily_sharpe": sharpe(daily_ret),
        "daily_sortino": sortino(daily_ret),
        "max_drawdown_pct": float(max_dd * 100),
        "n_round_trades": n_trades,
        "hit_rate": float(hit_rate),
        "profit_factor": float(gross_profit / gross_loss)
        if gross_loss > 0
        else None,
    }


# ---------------------------------------------------------------------------
# Walk-forward backtest (linear model on intraday features)
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    train_days: int = 5,
    test_days: int = 1,
) -> BacktestResult:
    _COLS = ["ret_1m", "ma_spread", "vol_10", "rsi_14", "vol_z"]
    days = sorted(set(df.index.date))
    signals = []

    for i in range(train_days, len(days)):
        train_idx = (df.index.date >= days[i - train_days]) & (df.index.date < days[i])
        test_idx = df.index.date == days[i]

        available = [c for c in _COLS if c in feats.columns]
        if not available:
            continue

        X_train = feats.loc[train_idx, available].values
        y_train = np.sign(
            df.loc[train_idx, "close"].pct_change(5).shift(-5).fillna(0).values
        )
        X = np.c_[np.ones(len(X_train)), X_train]
        beta, *_ = np.linalg.lstsq(X, y_train, rcond=None)

        X_test = feats.loc[test_idx, available].values
        X_t = np.c_[np.ones(len(X_test)), X_test]
        y_hat = X_t @ beta
        sig = np.where(y_hat > 0.10, 1, np.where(y_hat < -0.10, -1, 0))
        signals.append(pd.Series(sig, index=feats.loc[test_idx].index))

    signal_full = (
        pd.concat(signals).sort_index() if signals else pd.Series(0, index=feats.index)
    )
    bt = Backtester(ExecutionConfig())
    return bt.run(df.loc[signal_full.index], feats.loc[signal_full.index], signal_full)


# ---------------------------------------------------------------------------
# Monte Carlo stress test
# ---------------------------------------------------------------------------

def monte_carlo_stress(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    signal: pd.Series,
    n_runs: int = 50,
    exec_cfg: Optional[ExecutionConfig] = None,
) -> pd.DataFrame:
    if exec_cfg is None:
        exec_cfg = ExecutionConfig()
    rng = np.random.default_rng(42)
    stats = []
    for _ in range(n_runs):
        sim_cfg = ExecutionConfig(
            commission_per_share=exec_cfg.commission_per_share,
            slippage_bps=max(0.5, float(rng.normal(2.0, 1.0))),
            stop_loss_pct=float(np.clip(rng.normal(0.03, 0.01), 0.01, 0.06)),
            daily_loss_limit_pct=float(np.clip(rng.normal(0.02, 0.005), 0.01, 0.05)),
        )
        bt = Backtester(sim_cfg)
        res = bt.run(df, feats, signal)
        stats.append(res.metrics)

    df_stats = pd.DataFrame(stats).replace([np.inf, -np.inf], np.nan).dropna()
    df_stats.to_csv("results/monte_carlo_stats.csv", index=False)
    return df_stats
