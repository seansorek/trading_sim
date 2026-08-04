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
    # Clip, don't np.sign. A strategy may return a fractional target position in
    # [-1, 1] to express conviction; np.sign() here silently flattened every
    # such signal to full size, so sizing was unreachable no matter what the
    # decision layer emitted. Ternary strategies are unaffected — clipping
    # {-1,0,1} is the identity.
    return pd.Series(np.clip(sig.values.astype(float), -1.0, 1.0), index=sig.index)


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
    # Volatility-scaled barriers. When > 0 these REPLACE the fixed pcts above
    # with `mult * ATR%_at_entry`, so a 5% stop stops being a two-day move on
    # TSLA and a quarterly event on SPY. 0.0 = off (fixed pct), which is what
    # the dataclass defaults to so existing callers/tests are unchanged;
    # config/default.yaml turns it on for real backtests. Keep the off path —
    # it is the A/B baseline for judging whether the scaling helped.
    stop_loss_atr_mult: float = 0.0
    take_profit_atr_mult: float = 0.0
    # Vertical barrier: force flat after this many bars in a position, 0 = off.
    # The third leg of the triple barrier — with stop and target it bounds every
    # trade in price and in time. Set it to the label horizon the predictor was
    # fit on (daily_features.FWD_RET_HORIZON_DAYS); holding past that is a bet
    # on nothing the model forecast.
    max_holding_bars: int = 0
    # Annualized volatility target. >0 scales each position by
    # vol_target / realized_vol, so equal *risk* rather than equal *notional*
    # goes into SPY and TSLA. 0.0 = off (flat max_position_pct notional).
    # Deliberately separate from the strategy's conviction: they multiply, and
    # keeping them independent is what lets each be measured on its own.
    vol_target_annual: float = 0.0
    daily_loss_limit_pct: float = 0.05
    max_position_pct: float = 0.05
    holding_period_days: int = 0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, Any]


# Sanity band on the ATR% estimate before it is multiplied into a barrier.
# A stale/degenerate bar can produce an ATR% of ~0 (stop fires on the entry
# spread) or of 0.5 (no stop at all); neither is a volatility measurement.
ATR_PCT_MIN = 0.005
ATR_PCT_MAX = 0.10


# Bounds on the vol-target multiplier. Unbounded, a quiet stretch levers the
# book up arbitrarily on a realized-vol estimate that is about to be wrong.
VOL_SCALE_MIN = 0.25
VOL_SCALE_MAX = 3.0


def _vol_scale_series(df: pd.DataFrame, target_annual: float) -> pd.Series:
    """Causal position multiplier targeting a constant annualized volatility.

    Realized vol from a 20-bar close-to-close window, shifted one bar so the
    size traded on bar t is set by information available at t-1. NaN during
    warmup — callers fall back to unscaled size there.
    """
    daily = df["close"].pct_change().rolling(20).std().shift(1)
    annual = daily * np.sqrt(252.0)
    return (target_annual / annual).clip(VOL_SCALE_MIN, VOL_SCALE_MAX)


def _atr_pct_series(df: pd.DataFrame) -> pd.Series:
    """Causal ATR-14 as a fraction of price, for volatility-scaled barriers.

    Shifted one bar: the barrier active on bar t must not be sized by bar t's
    own high/low, or a wide bar widens the stop that is supposed to catch it.
    NaN during the warmup window — callers fall back to the fixed pct there.
    """
    from daily_features import _atr

    prev_close = df["close"].shift(1)
    return (_atr(df, 14).shift(1) / prev_close).clip(ATR_PCT_MIN, ATR_PCT_MAX)


def _exit_position(
    position: int,
    avg_entry_price: float,
    mid: float,
    spr: float,
    slippage: float,
    exec_cfg: ExecutionConfig,
    stop_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
) -> Optional[tuple]:
    """
    Check stop-loss and take-profit exits.

    `stop_pct` / `take_profit_pct` override the fixed ExecutionConfig values
    (used to pass in volatility-scaled barriers); None means use the config.

    Returns (fill_price, cost_commission, cost_spread, reason) if exit triggered,
    else None.
    """
    if position == 0 or avg_entry_price is None:
        return None

    stop = exec_cfg.stop_loss_pct if stop_pct is None else stop_pct
    target = exec_cfg.take_profit_pct if take_profit_pct is None else take_profit_pct

    pnl_pct = (mid - avg_entry_price) / avg_entry_price * np.sign(position)

    if pnl_pct > target:
        reason = "take_profit"
    elif pnl_pct < -stop:
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

        # Volatility-scaled barriers are sized once, at entry, and held for the
        # life of the position — a stop that re-widens as vol rises is a stop
        # that runs away from a losing trade.
        vol_scaled = (
            self.exec_cfg.stop_loss_atr_mult > 0
            or self.exec_cfg.take_profit_atr_mult > 0
        )
        atr_pct = _atr_pct_series(df) if vol_scaled else None
        vol_scale = (
            _vol_scale_series(df, self.exec_cfg.vol_target_annual)
            if self.exec_cfg.vol_target_annual > 0 else None
        )

        cash = self.exec_cfg.start_cash
        position = 0
        avg_entry_price: Optional[float] = None
        entry_atr_pct: Optional[float] = None
        entry_bar: Optional[int] = None
        equity_curve: list[float] = []
        timestamps: list = []
        trade_log: list[dict] = []
        daily_start_cash = cash
        current_day = df.index[0].date() if len(df) > 0 else None
        prev_close: Optional[float] = None
        # Cooldown: after a forced exit, suppress re-entry until signal returns to flat
        forced_exit_active = False

        for bar_i, (ts, row) in enumerate(df.iterrows()):
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

            # Fractional target position in [-1, 1] — 1.0 for ternary strategies,
            # a conviction weight for strategies that size their signal.
            desired = float(signal.loc[ts])

            # Cooldown logic: after forced exit, wait for signal to return to flat
            if forced_exit_active:
                if desired == 0:
                    forced_exit_active = False
                else:
                    desired = 0.0  # suppress re-entry

            current_equity = cash + position * mid
            notional = current_equity * self.exec_cfg.max_position_pct
            if vol_scale is not None:
                s = vol_scale.get(ts, np.nan)
                if not pd.isna(s):
                    notional *= float(s)
            shares = int(notional / mid) if mid > 0 else 0
            shares = min(shares, self.exec_cfg.max_position)

            target_pos = int(desired * shares)
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
                    entry_atr_pct = None
                    entry_bar = None
                elif old_position == 0 or np.sign(old_position) != np.sign(position):
                    # New position or full reversal: reset to current fill
                    avg_entry_price = fill_price
                    entry_bar = bar_i
                    # ponytail: barrier width is pinned at open and left alone
                    # when adding to the position. Re-blending it the way
                    # avg_entry_price is blended would be more consistent, but
                    # adds are rare here (signal is ternary) — revisit with
                    # conviction sizing, where partial adds become the norm.
                    if atr_pct is not None:
                        a = atr_pct.get(ts, np.nan)
                        entry_atr_pct = None if pd.isna(a) else float(a)
                elif abs(position) > abs(old_position):
                    # Adding to existing position: weighted average
                    old_shares = abs(old_position)
                    new_shares = abs(delta)
                    total_shares = abs(position)
                    avg_entry_price = (
                        (avg_entry_price * old_shares + fill_price * new_shares) / total_shares
                    )
                # else: same-direction reduction — avg_entry_price is unchanged

                trade_log.append({
                    "ts": ts,
                    "side": "BUY" if side > 0 else "SELL",
                    "shares": abs(delta),
                    "fill_price": float(fill_price),
                    "commission": float(cost_commission),
                    "spread_cost": float(cost_spread),
                    "exit_reason": "signal",
                })

            # Stop-loss / take-profit check. entry_atr_pct is None both when
            # vol scaling is off and during the ATR warmup — either way the
            # fixed pct applies.
            stop_pct = tp_pct = None
            if entry_atr_pct is not None:
                if self.exec_cfg.stop_loss_atr_mult > 0:
                    stop_pct = self.exec_cfg.stop_loss_atr_mult * entry_atr_pct
                if self.exec_cfg.take_profit_atr_mult > 0:
                    tp_pct = self.exec_cfg.take_profit_atr_mult * entry_atr_pct

            exit_result = _exit_position(
                position, avg_entry_price, mid, spr, slippage, self.exec_cfg,
                stop_pct=stop_pct, take_profit_pct=tp_pct,
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
                entry_atr_pct = None
                entry_bar = None
                forced_exit_active = True

            # Vertical barrier: flat after max_holding_bars. Checked after the
            # price barriers so a stop that lands on the final bar is still
            # logged as a stop, and it sets the same re-entry cooldown — without
            # that, a still-live signal would just re-enter next bar and the
            # barrier would only churn commission.
            if (
                self.exec_cfg.max_holding_bars > 0
                and position != 0
                and entry_bar is not None
                and bar_i - entry_bar >= self.exec_cfg.max_holding_bars
            ):
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
                    "exit_reason": "max_holding",
                })
                position = 0
                avg_entry_price = None
                entry_atr_pct = None
                entry_bar = None
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
                    entry_atr_pct = None
                    entry_bar = None
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

        # Benchmark = buy-and-hold the traded symbol itself over the same dates.
        metrics = compute_metrics(equity_series, trades_df, benchmark_close=df["close"])

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

def compute_metrics(equity: pd.Series, trades: pd.DataFrame,
                    benchmark_close: Optional[pd.Series] = None) -> Dict[str, Any]:
    daily_equity = equity.resample("1D").last().dropna()
    daily_ret = daily_equity.pct_change().dropna()

    def sharpe(r: pd.Series) -> float:
        mu, sd = r.mean(), r.std()
        return float(np.sqrt(252) * mu / sd) if sd and sd != 0 else 0.0

    def sortino(r: pd.Series, target: float = 0.0) -> float:
        """Standard downside deviation: root-mean-square of shortfalls below
        `target`, taken over ALL observations (not just the losing ones).
        Using every observation in the denominator — rather than the sample
        std of only the negative subset — keeps this finite even with a
        single loss (a 1-sample std has an undefined ddof=1 denominator and
        returns NaN); it also matches the standard Sortino-ratio definition."""
        if len(r) == 0:
            return 0.0
        shortfall = np.minimum(r.values - target, 0.0)
        dd = float(np.sqrt(np.mean(shortfall ** 2)))
        mu = float(r.mean())
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
        # Weighted-average commission-per-share for the currently open leg,
        # tracked the same way entry_price is — so realized PnL can be
        # netted of the entry-side commission as well as the exit-side
        # commission (issue #114: hit_rate/profit_factor must be net of
        # commission, not just the raw fill-price delta).
        #
        # `spread_cost` is deliberately excluded here: Backtester.run already
        # bakes half the spread into `fill_price` itself
        # (`fill_price = mid + side * (spr / 2 + slippage)`), so the
        # fill-to-fill gross_pnl below already reflects the round-trip
        # spread cost. `spread_cost` is logged for diagnostics only — netting
        # it out again here would charge the spread twice (PR #126 review,
        # discussion_r3655948346).
        entry_cost_per_share: float = 0.0
        for ts, t in trades.iterrows():
            side = 1 if t["side"] == "BUY" else -1
            t_shares = t["shares"]
            # .get(...) rather than [...]: tolerate older/synthetic trade
            # frames that predate this column, treating missing cost data
            # as zero rather than raising.
            t_cost_per_share = (
                float(t.get("commission", 0.0) or 0.0) / t_shares
                if t_shares else 0.0
            )
            if pos == 0:
                pos = side * t_shares
                entry_price = t["fill_price"]
                entry_cost_per_share = t_cost_per_share
            elif np.sign(side) == np.sign(pos):
                # Same-direction add (scale-in): update entry to weighted avg, no PnL
                old_shares = abs(pos)
                new_shares = t_shares
                total_shares = old_shares + new_shares
                entry_price = (
                    (entry_price * old_shares + t["fill_price"] * new_shares)
                    / total_shares
                )
                entry_cost_per_share = (
                    (entry_cost_per_share * old_shares + t_cost_per_share * new_shares)
                    / total_shares
                )
                pos += side * t_shares
            else:
                # Opposite-direction: realize PnL on the portion reduced/closed,
                # net of both legs' commission attributable to the closed
                # shares (spread is already priced into fill_price, see above).
                closed_shares = min(abs(pos), t_shares)
                gross_pnl = (
                    (t["fill_price"] - entry_price)
                    * np.sign(pos)
                    * closed_shares
                )
                net_cost = (entry_cost_per_share + t_cost_per_share) * closed_shares
                pnl_list.append(float(gross_pnl - net_cost))
                prev_pos = pos
                pos += side * t_shares
                if pos == 0:
                    entry_price = None
                    entry_cost_per_share = 0.0
                elif np.sign(pos) != np.sign(prev_pos):
                    # Position crossed zero — start tracking the new leg,
                    # whose cost basis is this same trade's exit-side cost.
                    entry_price = t["fill_price"]
                    entry_cost_per_share = t_cost_per_share
        if pnl_list:
            pnl_arr = np.array(pnl_list)
            gross_profit = float(pnl_arr[pnl_arr > 0].sum())
            gross_loss = float(-pnl_arr[pnl_arr < 0].sum())
            hit_rate = float((pnl_arr > 0).mean())
            n_trades = int(len(pnl_arr))

    start_eq = float(equity.iloc[0]) if len(equity) > 0 else 1.0
    end_eq = float(equity.iloc[-1]) if len(equity) > 0 else start_eq

    result = {
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

    if benchmark_close is not None and len(benchmark_close) > 1:
        bench = benchmark_close.reindex(equity.index).dropna()
        if len(bench) > 1:
            bench_daily = bench.resample("1D").last().dropna()
            if len(bench_daily) > 1:
                bench_total = float((bench_daily.iloc[-1] / bench_daily.iloc[0] - 1) * 100)
                bench_ret = bench_daily.pct_change().dropna()
                excess = (daily_ret.reindex(bench_ret.index).fillna(0.0) - bench_ret).dropna()
                ir = (float(np.sqrt(252) * excess.mean() / excess.std())
                      if excess.std() and excess.std() != 0 else 0.0)
                result["benchmark_return_pct"] = bench_total
                result["alpha_pct"] = result["total_return_pct"] - bench_total
                result["information_ratio"] = ir

    return result


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
