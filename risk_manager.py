"""
risk_manager.py - Advanced risk management for trading strategies.

Provides position sizing, stop loss, take profit, and trailing stop logic.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


class PositionSizer:
    """Calculate position size based on confidence, volatility, and risk budget."""
    
    def __init__(self, risk_per_trade: float = 0.02, max_position: float = 1.0):
        """
        Args:
            risk_per_trade: Max risk per trade as % of account (default 2%)
            max_position: Maximum position size as % of account (default 100%)
        """
        self.risk_per_trade = risk_per_trade
        self.max_position = max_position
    
    def calculate_size(
        self,
        confidence: float,
        volatility: float = 1.0,
        equity: float = 100000.0,
        stop_loss_pct: float = 0.02
    ) -> float:
        """
        Calculate position size using Kelly criterion and confidence.
        
        Args:
            confidence: Model confidence (0.0-1.0)
            volatility: Volatility multiplier (1.0 = normal)
            equity: Current account equity
            stop_loss_pct: Stop loss distance as %
        
        Returns:
            Position size in dollars
        
        Formula:
            f = (b*p - q) / b  where:
            - f: fraction of bankroll
            - p: probability of win (confidence)
            - q: probability of loss (1 - confidence)
            - b: odds (risk/reward ratio)
        """
        if confidence < 0.50:
            return 0.0  # No trade
        
        # Fractional Kelly (reduce by 50% for safety)
        p = confidence
        q = 1.0 - confidence
        
        # Assume 2:1 reward/risk (standard)
        reward_pct = stop_loss_pct * 2.0  # If risk 2%, target 4% gain
        b = reward_pct / stop_loss_pct
        
        kelly_fraction = max(0, (b * p - q) / b) * 0.5  # Half Kelly for safety
        
        # Adjust for volatility (reduce in high vol)
        vol_adjustment = 1.0 / max(1.0, volatility)
        
        # Calculate position size
        position_fraction = kelly_fraction * vol_adjustment
        position_fraction = min(position_fraction, self.max_position)
        
        # Convert to dollars
        position_size = position_fraction * equity
        
        return position_size


class RiskManager:
    """Track positions, stops, and profit targets."""
    
    def __init__(
        self,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03,
        trailing_stop_pct: float = 0.01
    ):
        """
        Args:
            stop_loss_pct: Hard stop loss at this % below entry
            take_profit_pct: Take profit at this % above entry
            trailing_stop_pct: Trailing stop at this % below highest price
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        
        # Track open positions
        self.positions: Dict[str, Dict] = {}
    
    def open_position(self, symbol: str, entry_price: float, position_size: int, direction: int):
        """
        Record a new position (long=1, short=-1).
        
        Args:
            symbol: Stock ticker
            entry_price: Entry price
            position_size: Number of shares
            direction: 1 for long, -1 for short
        """
        self.positions[symbol] = {
            'entry_price': entry_price,
            'position_size': position_size,
            'direction': direction,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'entry_time': pd.Timestamp.now()
        }
    
    def check_exit(
        self,
        symbol: str,
        current_price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if position should be exited based on stops.
        
        Returns:
            (should_exit, exit_reason)
            exit_reason: 'stop_loss', 'take_profit', 'trailing_stop', None
        """
        if symbol not in self.positions:
            return False, None
        
        pos = self.positions[symbol]
        entry_price = pos['entry_price']
        direction = pos['direction']
        
        # For longs
        if direction == 1:
            # Update highest price (for trailing stop)
            pos['highest_price'] = max(pos['highest_price'], current_price)
            
            # Calculate returns
            entry_return = (current_price - entry_price) / entry_price
            highest_return = (pos['highest_price'] - entry_price) / entry_price
            
            # Hard stop loss
            if entry_return < -self.stop_loss_pct:
                return True, 'stop_loss'
            
            # Take profit
            if entry_return > self.take_profit_pct:
                return True, 'take_profit'
            
            # Trailing stop (only after profit)
            if highest_return > self.take_profit_pct * 0.5:  # After 1.5% gain
                if entry_return < highest_return - self.trailing_stop_pct:
                    return True, 'trailing_stop'
        
        # For shorts (reverse logic)
        elif direction == -1:
            # Update lowest price (for trailing stop)
            pos['lowest_price'] = min(pos['lowest_price'], current_price)
            
            # Calculate returns
            entry_return = (entry_price - current_price) / entry_price
            lowest_return = (entry_price - pos['lowest_price']) / entry_price
            
            # Hard stop loss
            if entry_return < -self.stop_loss_pct:
                return True, 'stop_loss'
            
            # Take profit
            if entry_return > self.take_profit_pct:
                return True, 'take_profit'
            
            # Trailing stop (only after profit)
            if lowest_return > self.take_profit_pct * 0.5:
                if entry_return < lowest_return - self.trailing_stop_pct:
                    return True, 'trailing_stop'
        
        return False, None
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[Dict]:
        """
        Close a position and return P&L information.
        
        Returns:
            dict with 'entry_price', 'exit_price', 'pnl_pct', 'exit_reason'
        """
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        entry_price = pos['entry_price']
        position_size = pos['position_size']
        direction = pos['direction']
        
        # Calculate P&L
        if direction == 1:  # Long
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl = (exit_price - entry_price) * position_size
        else:  # Short
            pnl_pct = (entry_price - exit_price) / entry_price
            pnl = (entry_price - exit_price) * position_size
        
        result = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl': pnl,
            'position_size': position_size,
            'direction': direction,
            'duration': pd.Timestamp.now() - pos['entry_time']
        }
        
        # Remove position
        del self.positions[symbol]
        
        return result


class VolumeAdjustedExecutionPrice:
    """Calculate execution price adjusting for market impact."""
    
    @staticmethod
    def estimate_slippage(position_size: float, avg_volume: float, volume_pct_cap: float = 0.10) -> float:
        """
        Estimate slippage as % of price.
        
        Args:
            position_size: Shares to trade
            avg_volume: Average daily volume
            volume_pct_cap: Max % of daily volume to trade (default 10%)
        
        Returns:
            Slippage as decimal (e.g., 0.002 = 0.2%)
        """
        volume_pct = position_size / avg_volume
        
        if volume_pct > volume_pct_cap:
            # Higher impact for larger orders
            impact_bps = 10 + (volume_pct - volume_pct_cap) * 100
            return impact_bps / 10000
        else:
            # Minimal impact for normal orders
            return 0.001  # 1 bp minimum
    
    @staticmethod
    def apply_slippage(price: float, slippage_pct: float, direction: int) -> float:
        """
        Apply slippage to execution price.
        
        Args:
            price: Reference price
            slippage_pct: Slippage as decimal
            direction: 1 for buy (pay more), -1 for sell (receive less)
        
        Returns:
            Adjusted price
        """
        return price * (1 + direction * slippage_pct)


# Example usage:
if __name__ == '__main__':
    # Test position sizing
    sizer = PositionSizer(risk_per_trade=0.02, max_position=1.0)
    
    # High confidence, normal volatility
    size1 = sizer.calculate_size(confidence=0.70, volatility=1.0, equity=100000, stop_loss_pct=0.02)
    print(f"Position size (70% confidence): ${size1:,.0f}")
    
    # High confidence, high volatility
    size2 = sizer.calculate_size(confidence=0.70, volatility=2.0, equity=100000, stop_loss_pct=0.02)
    print(f"Position size (70% conf, 2x vol): ${size2:,.0f}")
    
    # Risk manager test
    rm = RiskManager(stop_loss_pct=0.02, take_profit_pct=0.03)
    rm.open_position('AAPL', entry_price=150.0, position_size=100, direction=1)
    
    # Check various prices
    should_exit, reason = rm.check_exit('AAPL', 147.0)  # 2% loss
    print(f"Exit at $147: {should_exit}, {reason}")
    
    should_exit, reason = rm.check_exit('AAPL', 154.5)  # 3% gain
    print(f"Exit at $154.5: {should_exit}, {reason}")
