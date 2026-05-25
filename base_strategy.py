"""
base_strategy.py — Base classes for trading strategies.

Contains the core BaseStrategy and StrategyConfig used by all strategies.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    name: str
    lookback: int = 20
    threshold: float = 0.8
    rsi_lower: int = 30
    rsi_upper: int = 70
    holding_period: int = 5  # minimum bars between position changes


class BaseStrategy:
    """Unified interface: implement signal(feats, df) -> pd.Series of {-1, 0, 1}."""

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def _apply_holding_period(self, sig: pd.Series) -> pd.Series:
        """
        Enforce a minimum holding period between position changes.

        Suppresses any signal (including direction reversals) that arrives
        within `holding_period` bars of the last executed trade.
        """
        if self.cfg.holding_period <= 0:
            return sig

        sig_array = sig.values.copy().astype(int)
        result = np.zeros_like(sig_array)
        last_trade_loc = -self.cfg.holding_period - 1
        current_position = 0

        for loc in range(len(sig_array)):
            new_signal = sig_array[loc]
            if new_signal == 0:
                continue
            if loc > last_trade_loc + self.cfg.holding_period:
                result[loc] = new_signal
                last_trade_loc = loc
                current_position = new_signal

        return pd.Series(result, index=sig.index)
