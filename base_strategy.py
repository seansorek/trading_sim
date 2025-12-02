"""
base_strategy.py - Base classes for trading strategies.

Contains the core BaseStrategy and StrategyConfig classes that are shared
across all strategy implementations to avoid circular imports.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    name: str
    # Common params used by various strategies
    lookback: int = 20
    threshold: float = 0.8
    rsi_lower: int = 30
    rsi_upper: int = 70
    holding_period: int = 5  # Minimum bars between position changes


class BaseStrategy:
    """Unified interface: implement signal(feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series."""
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def signal(self, feats: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def _apply_holding_period(self, sig: pd.Series) -> pd.Series:
        """
        Vectorized implementation to enforce a minimum holding period.
        Suppresses new trade signals for `holding_period` bars after a trade.
        Also filters out trades during low-liquidity periods (first/last 30 min).
        """
        if self.cfg.holding_period <= 0:
            return sig
        
        # Convert to numpy array for easier manipulation
        sig_array = sig.values.copy()
        
        # Filter out first/last 30 minutes (low liquidity, wide spreads)
        hour = sig.index.hour + sig.index.minute / 60.0
        market_hours_filter = (hour >= 10.0) & (hour <= 15.5)  # 10am-3:30pm EST
        sig_array[~market_hours_filter] = 0
        
        # Find the integer locations of trades (non-zero signals)
        trade_locs = np.where(sig_array != 0)[0]
        
        if len(trade_locs) == 0:
            return sig
        
        # Keep track of when we can trade again
        last_trade_loc = -self.cfg.holding_period - 1
        filtered_signals = np.zeros_like(sig_array)
        
        for loc in trade_locs:
            # Only allow this trade if enough bars have passed since the last trade
            if loc > last_trade_loc + self.cfg.holding_period:
                filtered_signals[loc] = sig_array[loc]
                last_trade_loc = loc
        
        return pd.Series(filtered_signals, index=sig.index)
