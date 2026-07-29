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

        The output is a *target position* per bar, not a per-bar trade
        instruction — that is how the backtester reads it. So a suppressed bar
        carries the position forward rather than emitting 0, which the engine
        would execute as "go flat".

        Until 2026-07-28 this zeroed suppressed bars instead. Against a
        target-position engine that inverted the rule it is named for: a
        continuously-BUY signal became a one-bar hold followed by a
        `holding_period`-bar lockout, so no position ever survived long enough
        for the 3-day forecast horizon or the stop/take-profit barriers to
        matter. Measured on daily_predictor over 365 days, every single
        position was held exactly one bar.

        A live position is held for at least `holding_period` bars; after that
        the incoming signal takes effect, and an unchanged signal extends the
        hold indefinitely without re-trading.
        """
        if self.cfg.holding_period <= 0:
            return sig

        sig_array = sig.values.copy().astype(float)
        result = np.zeros_like(sig_array)
        position = 0.0
        last_entry_loc = -self.cfg.holding_period - 1

        for loc in range(len(sig_array)):
            new_signal = sig_array[loc]
            # Direction, not magnitude, defines a position change: a
            # conviction-sized strategy varies |signal| every bar, and treating
            # each wobble as a new position would resize daily and pay the
            # spread for it. Size is therefore pinned at entry.
            if np.sign(new_signal) != np.sign(position) and loc >= last_entry_loc + self.cfg.holding_period:
                position = new_signal
                # Only entries and reversals restart the clock. Flat is not a
                # position being held, so exiting does not impose a cooldown
                # on the next entry — that would just discard signal, which is
                # the failure mode this method used to have.
                if position != 0:
                    last_entry_loc = loc
            result[loc] = position

        return pd.Series(result, index=sig.index)
