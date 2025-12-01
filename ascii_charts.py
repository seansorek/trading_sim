#!/usr/bin/env python3
"""
ascii_charts.py

Generate cute ASCII charts from equity curve data for Discord messages.
"""

import pandas as pd
import numpy as np


def equity_curve_to_ascii(csv_path: str, height: int = 8, width: int = 30) -> str:
    """
    Convert an equity curve CSV to an ASCII chart.
    Returns a formatted string suitable for Discord.
    """
    try:
        # Read equity curve
        df = pd.read_csv(csv_path, index_col=0)
        equity = df.iloc[:, 0].values
        
        if len(equity) == 0:
            return "No data"
        
        # Resample to width
        if len(equity) > width:
            indices = np.linspace(0, len(equity) - 1, width, dtype=int)
            equity = equity[indices]
        
        # Normalize to height
        min_val = equity.min()
        max_val = equity.max()
        
        if max_val == min_val:
            normalized = np.full_like(equity, height // 2, dtype=float)
        else:
            normalized = ((equity - min_val) / (max_val - min_val)) * (height - 1)
        
        # Build chart
        chart = []
        for row in range(height - 1, -1, -1):
            line = ""
            for col in range(width):
                if normalized[col] >= row:
                    line += "█"
                else:
                    line += " "
            chart.append(line)
        
        # Add axis labels
        chart_str = "```\n"
        for line in chart:
            chart_str += line + "\n"
        chart_str += "└" + "─" * width + "\n"
        
        # Add value labels
        start_val = equity[0]
        end_val = equity[-1]
        change_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
        
        chart_str += f"${start_val:.0f} → ${end_val:.0f} ({change_pct:+.1f}%)\n```"
        
        return chart_str
        
    except Exception as e:
        return f"Error: {str(e)}"


def simple_metric_chart(metrics: dict) -> str:
    """
    Create a simple ASCII representation of key metrics.
    """
    try:
        total_return = metrics.get('total_return_pct', 0)
        sharpe = metrics.get('daily_sharpe', 0)
        max_dd = metrics.get('max_drawdown_pct', 0)
        win_rate = metrics.get('hit_rate', 0)
        
        # Create mini bar chart for win rate
        win_bars = int(win_rate * 20)
        win_chart = "█" * win_bars + "░" * (20 - win_bars)
        
        # Color coding for return
        return_emoji = "📈" if total_return > 0 else "📉"
        
        chart = (
            f"{return_emoji} Return: {total_return:+.2f}%\n"
            f"Win Rate: [{win_chart}] {win_rate*100:.0f}%\n"
            f"Sharpe: {sharpe:.2f}\n"
            f"Max DD: {max_dd:.2f}%"
        )
        
        return chart
        
    except Exception as e:
        return f"Error: {str(e)}"
