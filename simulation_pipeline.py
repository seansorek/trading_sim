"""
simulation_pipeline.py — Backtesting engine for daily trading strategies.
"""
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from base_strategy import BaseStrategy, StrategyConfig
from predictor_strategy import PredictorStrategy

logger = logging.getLogger(__name__)

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Import ML strategies
# ---------------------------------------------------------------------------
try:
    from ml_strategies import (
        DailyLogisticStrategy,
        DailyXGBoostStrategy,
        DailyPredictorStrategy,
    )
    HAS_ML_STRATEGIES = True
except ImportError as exc:
    logger.warning("ML strategies not loaded: %s", exc)
    HAS_ML_STRATEGIES = False

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict = {}
if HAS_ML_STRATEGIES:
    STRATEGY_REGISTRY["daily_logistic"] = DailyLogisticStrategy
    STRATEGY_REGISTRY["daily_xgboost"] = DailyXGBoostStrategy
    STRATEGY_REGISTRY["daily_predictor"] = DailyPredictorStrategy


class DailyDQNStrategy(BaseStrategy):
    """DQN agent next-day strategy backed by PredictorStrategy + DQNDecision."""

    def __init__(
        self,
        cfg: StrategyConfig,
        spy_df=None,
        model_path: str = "models/dqn_agent.pt",
        window: int = 20,
        confidence_threshold: float = 2.0,
        q_advantage_threshold: float = 1.0,
    ):
        super().__init__(cfg)
        _model_path = os.environ.get("DQN_MODEL", getattr(cfg, "model_path", model_path))
        _window = int(os.environ.get("DQN_WINDOW", getattr(cfg, "window", window)))
        _ct = float(os.environ.get("DQN_CONFIDENCE", getattr(cfg, "confidence_threshold", confidence_threshold)))
        _qt = float(os.environ.get("DQN_Q_ADVANTAGE", getattr(cfg, "q_advantage_threshold", q_advantage_threshold)))

        # Expose as instance attrs so tests and callers can inspect them
        self.confidence_threshold = _ct
        self.q_advantage_threshold = _qt

        from predictors.dqn import DQNPredictor
        from decision_layers.dqn_decision import DQNDecision
        try:
            predictor = DQNPredictor.load(_model_path, window=_window,
                                          confidence_threshold=_ct,
                                          q_advantage_threshold=_qt)
        except Exception as exc:
            logger.warning("DQNPredictor: failed to load from %s: %s", _model_path, exc)
            predictor = None

        self._inner = None
        if predictor is not None:
            decision = DQNDecision(confidence_threshold=_ct, q_advantage_threshold=_qt)
            self._inner = PredictorStrategy(cfg, predictor, decision, spy_df=spy_df)

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        if self._inner is None:
            return pd.Series(0, index=df.index)
        return self._inner.signal(feats, df)


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
    metrics: Dict[str, Any]


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
        seed: Optional[int] = None,
    ) -> BacktestResult:
        # Seed only when the caller explicitly requests reproducibility.
        # Previously this hard-coded seed=42 on every call, which defeated
        # Monte Carlo stress tests (see issue #55).
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

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
        # Cooldown: after a forced exit, suppress re-entry until signal returns to flat
        forced_exit_active = False

        for ts, row in df.iterrows():
            if ts.date() != current_day:
                current_day = ts.date()
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

            # Cooldown logic: after forced exit, wait for signal to return to flat
            if forced_exit_active:
                if desired == 0:
                    forced_exit_active = False
                else:
                    desired = 0  # suppress re-entry

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
                forced_exit_active = True

            # Daily loss limit (baseline = mark-to-market equity at day boundary)
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
                    forced_exit_active = True

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

        # Persist artifacts (skip when caller passes an empty dict to suppress writes)
        if artifact_paths:
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

def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> Dict[str, Any]:
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
            elif np.sign(side) == np.sign(pos):
                # Same-direction add (scale-in): update entry to weighted avg, no PnL
                old_shares = abs(pos)
                new_shares = t["shares"]
                total_shares = old_shares + new_shares
                entry_price = (
                    (entry_price * old_shares + t["fill_price"] * new_shares)
                    / total_shares
                )
                pos += side * t["shares"]
            else:
                # Opposite-direction: realize PnL on the portion reduced/closed
                closed_shares = min(abs(pos), t["shares"])
                pnl = (
                    (t["fill_price"] - entry_price)
                    * np.sign(pos)
                    * closed_shares
                )
                pnl_list.append(float(pnl))
                prev_pos = pos
                pos += side * t["shares"]
                if pos == 0:
                    entry_price = None
                elif np.sign(pos) != np.sign(prev_pos):
                    # Position crossed zero — start tracking the new leg
                    entry_price = t["fill_price"]
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
# Monte Carlo stress test
# ---------------------------------------------------------------------------

def monte_carlo_stress(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    signal: pd.Series,
    n_runs: int = 50,
    base_exec_cfg: Optional[ExecutionConfig] = None,
    out_csv: str = "results/monte_carlo_stats.csv",
) -> pd.DataFrame:
    if base_exec_cfg is None:
        base_exec_cfg = ExecutionConfig()
    rng = np.random.default_rng(42)
    stats = []
    for _ in range(n_runs):
        exec_cfg = ExecutionConfig(
            commission_per_share=base_exec_cfg.commission_per_share,
            slippage_bps=max(0.5, float(rng.normal(2.0, 1.0))),
            stop_loss_pct=float(np.clip(rng.normal(0.03, 0.01), 0.01, 0.06)),
            daily_loss_limit_pct=float(np.clip(rng.normal(0.02, 0.005), 0.01, 0.05)),
        )
        bt = Backtester(exec_cfg)
        res = bt.run(df, feats, signal, artifact_paths={})
        stats.append(res.metrics)

    df_stats = pd.DataFrame(stats).replace([np.inf, -np.inf], np.nan).dropna()
    df_stats.to_csv(out_csv, index=False)
    return df_stats
