import numpy as np
from typing import Tuple, Dict, Optional

from daily_features import make_daily_features, FEATURE_COLS, FWD_RET_HORIZON_DAYS
from data_loader import load_yfinance


class TradingEnv:
    """
    Minimal Gym-like environment for daily trading.
    - Observation: stacked features for last `window` days
    - Actions: 0=Hold, 1=Long, 2=Short
    - Reward: PnL for the day minus transaction cost
    - Episode: iterates each day in the requested range
    """

    def __init__(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        window: int = 20,
        transaction_cost_bps: float = 10.0,
        feature_scaler: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.window = window
        self.transaction_cost_bps = transaction_cost_bps
        self.scaler = feature_scaler

        # Load REAL data from yfinance
        raw = load_yfinance(symbol=symbol, start=start or "2024-01-01", end=end or "2025-12-02", interval="1d")
        feats = make_daily_features(raw)
        # Drop rows needing forward return to avoid peeking
        feats = feats.dropna(subset=["fwd_ret_1d"]).copy()
        self.df = feats
        self.features = FEATURE_COLS

        if len(self.df) < self.window + 2:
            raise ValueError(f"Insufficient data for env: {symbol} has {len(self.df)} bars")
        
        # Log data source verification
        self._data_source = "yfinance"
        self._num_bars = len(self.df)
        self._date_range = f"{self.df.index[0].date()} to {self.df.index[-1].date()}"

        # Optional simple scaler: z-score per feature.
        # Fit the scaler on a strict warmup window only — fitting over the
        # entire DataFrame leaks future means/stdevs into observations at
        # earlier steps (look-ahead bias). See issue #45.
        if self.scaler is None:
            fit_end = min(252, max(self.window + 1, len(self.df) // 2))
            self.scaler = {}
            for c in self.features:
                mu = float(self.df[c].iloc[:fit_end].mean())
                sd = float(self.df[c].iloc[:fit_end].std() or 1.0)
                self.scaler[c] = (mu, sd)
        for c in self.features:
            mu, sd = self.scaler[c]
            self.df[c] = (self.df[c] - mu) / (sd if sd != 0 else 1.0)

        # Build price for PnL
        self.prices = self.df["close"].values.astype(float)
        self.returns = self.df["fwd_ret_1d"].values.astype(float)

        self.reset()

    @property
    def action_space_n(self) -> int:
        return 3

    @property
    def observation_space_shape(self) -> Tuple[int]:
        return (self.window * len(self.features),)
    
    def get_data_info(self) -> Dict:
        """Get information about the data source and quality."""
        return {
            "symbol": self.symbol,
            "source": self._data_source,
            "num_bars": self._num_bars,
            "date_range": self._date_range,
            "features": len(self.features),
            "feature_list": self.features
        }

    def _get_state(self) -> np.ndarray:
        w = self.window
        i0 = self.idx - w
        frame = self.df.iloc[i0:self.idx]
        obs = frame[self.features].values.astype(np.float32)
        return obs.flatten()

    def reset(self) -> np.ndarray:
        self.idx = self.window
        self.position = 0  # -1 short, 0 flat, +1 long
        self.equity = 0.0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # Action -> target position
        target_pos = 0
        if action == 1:
            target_pos = 1
        elif action == 2:
            target_pos = -1

        # Transaction cost if position changes
        cost = 0.0
        if target_pos != self.position:
            cost = (abs(target_pos - self.position)) * (self.transaction_cost_bps * 1e-4) * self.prices[self.idx - 1]

        # Actual one-day close-to-close return for the bar this position is
        # held over (today -> next bar), not the FWD_RET_HORIZON_DAYS-bar
        # cumulative fwd_ret_1d, which would overlap across consecutive steps.
        ret = self.prices[self.idx] / self.prices[self.idx - 1] - 1.0
        gross_pnl = target_pos * ret * self.prices[self.idx - 1]
        pnl = gross_pnl - cost

        # Reward: simple scaled PnL
        reward = pnl / (self.prices[0] * 0.005)  # Normalize by 0.5% of initial price

        # Penalty for reversals (changing direction), based on the position
        # immediately before this action (not a one-step-stale value).
        if target_pos != self.position and self.position != 0:
            reward -= 0.02

        self.equity += pnl
        self.position = target_pos

        done = False
        info = {"pnl": pnl, "equity": self.equity, "price": float(self.prices[self.idx - 1])}

        # advance
        self.idx += 1
        if self.idx >= len(self.df):
            done = True
            next_state = np.zeros(self.observation_space_shape, dtype=np.float32)
        else:
            next_state = self._get_state()
        return next_state, float(reward), done, info
