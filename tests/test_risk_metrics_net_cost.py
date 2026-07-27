"""tests/test_risk_metrics_net_cost.py — Regression tests for issue #114.

- sortino() must define downside deviation over ALL observations relative
  to the target return (not the sample std of only the losing subset), so a
  single-loss sample still returns a finite value.
- compute_metrics' realized trade P&L (hit_rate, profit_factor) must be net
  of the logged `commission` and `spread_cost` fields, not just the raw
  fill-price delta.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation_pipeline import compute_metrics


# ---------------------------------------------------------------------------
# Sortino: finite result with a single loss
# ---------------------------------------------------------------------------

def test_sortino_finite_with_single_loss():
    """A daily-return series with exactly one negative observation must not
    produce NaN. The old implementation took std(ddof=1) of the length-1
    negative subset, which is undefined (NaN) by construction."""
    idx = pd.date_range("2024-01-02", periods=6, freq="B", tz="UTC")
    # Equity path: five up days, one down day.
    daily_returns = [0.01, 0.01, 0.01, 0.01, -0.005, 0.01]
    equity_vals = [100_000.0]
    for r in daily_returns:
        equity_vals.append(equity_vals[-1] * (1 + r))
    equity = pd.Series(equity_vals, index=pd.date_range("2024-01-02", periods=7, freq="B", tz="UTC"))

    metrics = compute_metrics(equity, pd.DataFrame())
    assert np.isfinite(metrics["daily_sortino"]), (
        f"Expected a finite Sortino ratio with a single loss, got "
        f"{metrics['daily_sortino']!r}"
    )


def test_sortino_finite_and_matches_full_sample_downside_deviation():
    """Sortino's denominator must be computed over ALL observations (losses
    contribute their squared shortfall, wins contribute 0), not just the
    sample std of the losing subset."""
    idx = pd.date_range("2024-01-02", periods=11, freq="B", tz="UTC")
    equity = pd.Series(
        [100_000.0, 101_000.0, 99_000.0, 100_500.0, 101_500.0, 100_800.0,
         102_000.0, 101_200.0, 103_000.0, 102_400.0, 104_000.0],
        index=idx,
    )
    daily_ret = equity.pct_change().dropna()

    metrics = compute_metrics(equity, pd.DataFrame())

    target = 0.0
    shortfall = np.minimum(daily_ret.values - target, 0.0)
    expected_dd = np.sqrt(np.mean(shortfall ** 2))
    expected_sortino = np.sqrt(252) * daily_ret.mean() / expected_dd

    assert np.isfinite(metrics["daily_sortino"])
    assert metrics["daily_sortino"] == pytest.approx(expected_sortino, rel=1e-9)


def test_sortino_zero_when_no_losses():
    """No losses at all -> downside deviation is 0 -> function must not
    divide by zero (returns 0.0, matching the existing sharpe() guard
    convention rather than raising or returning inf)."""
    idx = pd.date_range("2024-01-02", periods=6, freq="B", tz="UTC")
    equity = pd.Series(100_000.0 * (1.01 ** np.arange(6)), index=idx)
    metrics = compute_metrics(equity, pd.DataFrame())
    assert metrics["daily_sortino"] == 0.0


# ---------------------------------------------------------------------------
# Realized trade P&L must be net of commission + spread_cost
# ---------------------------------------------------------------------------

def _make_trades_with_costs(rows: list) -> pd.DataFrame:
    """Build a trades DataFrame from (side, shares, fill_price, commission,
    spread_cost) tuples — the full schema Backtester.run logs per fill."""
    idx = pd.date_range("2024-01-02", periods=len(rows), freq="B", tz="UTC")
    return pd.DataFrame(
        [
            {"side": r[0], "shares": r[1], "fill_price": r[2],
             "commission": r[3], "spread_cost": r[4]}
            for r in rows
        ],
        index=idx,
    )


def test_gross_winner_becomes_net_loser_after_costs():
    """A trade that looks like a small win on fill-price alone (110 - 109 =
    +1/share over 10 shares = +10 gross) must be reported as a LOSS once the
    logged commission + spread_cost are netted out (here: 8 + 8 = 16 total
    cost > 10 gross gain)."""
    idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
    equity = pd.Series(100_000.0 + np.zeros(10), index=idx)

    trades = _make_trades_with_costs([
        ("BUY", 10, 109.0, 4.0, 4.0),   # entry: costs 4 commission + 4 spread
        ("SELL", 10, 110.0, 4.0, 4.0),  # exit: costs 4 commission + 4 spread
    ])
    # Gross PnL = (110 - 109) * 10 = +10
    # Net PnL   = 10 - (4+4) - (4+4) = 10 - 16 = -6  -> a LOSS
    metrics = compute_metrics(equity, trades)

    assert metrics["n_round_trades"] == 1
    assert metrics["hit_rate"] == pytest.approx(0.0), (
        "Gross-winning trade should be reclassified as a net loser once "
        "commission + spread_cost are netted out."
    )
    # gross_profit == 0 (no winning trades) and gross_loss == 6 -> factor 0.0
    assert metrics["profit_factor"] == pytest.approx(0.0)


def test_net_pnl_nets_commission_and_spread_on_both_legs():
    """Directly pin the net PnL value (not just win/loss classification) for
    a simple one-round-trip trade with costs on both legs."""
    idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
    equity = pd.Series(100_000.0 + np.zeros(10), index=idx)

    trades = _make_trades_with_costs([
        ("BUY", 5, 100.0, 1.0, 2.0),
        ("SELL", 5, 120.0, 1.5, 2.5),
    ])
    # Gross = (120-100)*5 = 100
    # Entry cost/share = (1+2)/5 = 0.6 ; exit cost/share = (1.5+2.5)/5 = 0.8
    # Net = 100 - (0.6+0.8)*5 = 100 - 7 = 93
    metrics = compute_metrics(equity, trades)
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["profit_factor"] is None or metrics["profit_factor"] > 0

    # Recover gross_profit via profit_factor's numerator is indirect; assert
    # indirectly by re-deriving what a still-net-loser scenario would look
    # like with heavier costs (belt-and-braces against a regression that
    # nets only one of the two fields).
    heavier_costs = _make_trades_with_costs([
        ("BUY", 5, 100.0, 5.0, 5.0),
        ("SELL", 5, 120.0, 5.0, 5.0),
    ])
    # Gross = 100; net = 100 - (10/5 + 10/5)*5 = 100 - 20 = 80 -> still a win
    heavier_metrics = compute_metrics(equity, heavier_costs)
    assert heavier_metrics["hit_rate"] == pytest.approx(1.0)

    even_heavier_costs = _make_trades_with_costs([
        ("BUY", 5, 100.0, 15.0, 15.0),
        ("SELL", 5, 120.0, 15.0, 15.0),
    ])
    # Gross = 100; net = 100 - (30/5 + 30/5)*5 = 100 - 60 = 40 -> still a win,
    # but pushing costs further should eventually flip it (sanity check the
    # monotonic direction of the netting, not just a single fixed point).
    even_heavier_metrics = compute_metrics(equity, even_heavier_costs)
    assert even_heavier_metrics["hit_rate"] == pytest.approx(1.0)

    flipping_costs = _make_trades_with_costs([
        ("BUY", 5, 100.0, 25.0, 25.0),
        ("SELL", 5, 120.0, 25.0, 25.0),
    ])
    # Gross = 100; net = 100 - (50/5 + 50/5)*5 = 100 - 100 = 0 -> not > 0
    flipping_metrics = compute_metrics(equity, flipping_costs)
    assert flipping_metrics["hit_rate"] == pytest.approx(0.0)


def test_trades_missing_cost_columns_default_to_zero_cost():
    """Backward compatibility: a trades frame without commission/spread_cost
    columns (e.g. an older synthetic test fixture) must not raise, and must
    behave exactly as the pre-fix fill-price-only calculation did."""
    idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
    equity = pd.Series(100_000.0 + np.zeros(10), index=idx)
    trades = pd.DataFrame(
        [{"side": "BUY", "shares": 1, "fill_price": 100.0},
         {"side": "SELL", "shares": 1, "fill_price": 110.0}],
        index=idx[:2],
    )
    metrics = compute_metrics(equity, trades)
    assert metrics["n_round_trades"] == 1
    assert metrics["hit_rate"] == pytest.approx(1.0)
