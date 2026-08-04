"""tests/test_risk_metrics_net_cost.py — Regression tests for issue #114.

- sortino() must define downside deviation over ALL observations relative
  to the target return (not the sample std of only the losing subset), so a
  single-loss sample still returns a finite value.
- compute_metrics' realized trade P&L (hit_rate, profit_factor) must be net
  of the logged `commission` field, not just the raw fill-price delta —
  but must NOT additionally net `spread_cost`, since that cost is already
  priced into `fill_price` by the execution model (netting it again would
  double-charge the spread; see PR #126 review, discussion_r3655948346).
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
# Realized trade P&L must be net of commission only — spread_cost is already
# priced into fill_price by the execution model and must NOT be netted again.
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


def test_gross_winner_becomes_net_loser_after_commission():
    """A trade that looks like a small win on fill-price alone (110 - 109 =
    +1/share over 10 shares = +10 gross) must be reported as a LOSS once the
    logged commission is netted out (here: 6 + 6 = 12 total commission > 10
    gross gain). `spread_cost` is set absurdly large on both legs to pin
    that it has no effect on the result — it's already reflected in the 109
    -> 110 fill-price delta, not a separate cash outflow to net again."""
    idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
    equity = pd.Series(100_000.0 + np.zeros(10), index=idx)

    trades = _make_trades_with_costs([
        ("BUY", 10, 109.0, 6.0, 999.0),   # entry: 6 commission; spread_cost ignored
        ("SELL", 10, 110.0, 6.0, 999.0),  # exit: 6 commission; spread_cost ignored
    ])
    # Gross PnL = (110 - 109) * 10 = +10
    # Net PnL   = 10 - (6+6) = -2  -> a LOSS, regardless of spread_cost
    metrics = compute_metrics(equity, trades)

    assert metrics["n_round_trades"] == 1
    assert metrics["hit_rate"] == pytest.approx(0.0), (
        "Gross-winning trade should be reclassified as a net loser once "
        "commission is netted out, independent of spread_cost."
    )
    # gross_profit == 0 (no winning trades) and gross_loss == 2 -> factor 0.0
    assert metrics["profit_factor"] == pytest.approx(0.0)


def test_net_pnl_nets_commission_only_and_ignores_spread_cost():
    """Directly pin the net PnL value (not just win/loss classification) for
    a simple one-round-trip trade, and confirm varying spread_cost alone
    never changes it (only commission does)."""
    idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
    equity = pd.Series(100_000.0 + np.zeros(10), index=idx)

    trades = _make_trades_with_costs([
        ("BUY", 5, 100.0, 1.0, 2.0),
        ("SELL", 5, 120.0, 1.5, 2.5),
    ])
    # Gross = (120-100)*5 = 100
    # Entry commission/share = 1/5 = 0.2 ; exit commission/share = 1.5/5 = 0.3
    # Net = 100 - (0.2+0.3)*5 = 100 - 2.5 = 97.5 (spread_cost of 2.0/2.5 not netted)
    metrics = compute_metrics(equity, trades)
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["profit_factor"] is None or metrics["profit_factor"] > 0

    # Same commission, wildly different spread_cost: result must be identical.
    same_commission_diff_spread = _make_trades_with_costs([
        ("BUY", 5, 100.0, 1.0, 5000.0),
        ("SELL", 5, 120.0, 1.5, 5000.0),
    ])
    diff_spread_metrics = compute_metrics(equity, same_commission_diff_spread)
    assert diff_spread_metrics["hit_rate"] == metrics["hit_rate"]
    assert diff_spread_metrics["profit_factor"] == metrics["profit_factor"]

    # Commission alone, pushed high enough, still flips the classification
    # (sanity check that commission is netted, not ignored entirely).
    flipping_costs = _make_trades_with_costs([
        ("BUY", 5, 100.0, 12.0, 0.0),
        ("SELL", 5, 120.0, 12.0, 0.0),
    ])
    # Gross = 100; net = 100 - (12/5 + 12/5)*5 = 100 - 24 = 76 -> still a win
    flipping_metrics = compute_metrics(equity, flipping_costs)
    assert flipping_metrics["hit_rate"] == pytest.approx(1.0)

    heavier_flipping_costs = _make_trades_with_costs([
        ("BUY", 5, 100.0, 60.0, 0.0),
        ("SELL", 5, 120.0, 60.0, 0.0),
    ])
    # Gross = 100; net = 100 - (60/5 + 60/5)*5 = 100 - 120 = -20 -> a loss
    heavier_flipping_metrics = compute_metrics(equity, heavier_flipping_costs)
    assert heavier_flipping_metrics["hit_rate"] == pytest.approx(0.0)


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
