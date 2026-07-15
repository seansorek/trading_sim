import numpy as np
import pandas as pd
import pytest

from panel_backtester import rank_to_weights, PanelConfig, run_panel


def test_equal_weight_long_top_short_bottom():
    pred = pd.Series({f"S{i}": float(i) for i in range(10)})
    w = rank_to_weights(pred, decile=0.2, gross_exposure=1.0, min_names=5)
    # k = int(10 * 0.2) = 2. Longs = two highest (S8, S9), shorts = two lowest (S0, S1).
    assert w["S9"] == pytest.approx(0.25)
    assert w["S8"] == pytest.approx(0.25)
    assert w["S0"] == pytest.approx(-0.25)
    assert w["S1"] == pytest.approx(-0.25)
    assert w["S5"] == pytest.approx(0.0)


def test_book_is_dollar_neutral_and_hits_gross_exposure():
    pred = pd.Series({f"S{i}": float(i) for i in range(20)})
    w = rank_to_weights(pred, decile=0.1, gross_exposure=1.0, min_names=5)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert w.abs().sum() == pytest.approx(1.0)


def test_gross_exposure_scales_weights():
    pred = pd.Series({f"S{i}": float(i) for i in range(20)})
    w = rank_to_weights(pred, decile=0.1, gross_exposure=2.0, min_names=5)
    assert w.abs().sum() == pytest.approx(2.0)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)


def test_below_min_names_holds_nothing():
    pred = pd.Series({f"S{i}": float(i) for i in range(5)})
    w = rank_to_weights(pred, decile=0.2, gross_exposure=1.0, min_names=20)
    assert (w == 0.0).all()


def test_nan_predictions_are_excluded_from_the_cross_section():
    pred = pd.Series({f"S{i}": float(i) for i in range(10)})
    pred["S9"] = np.nan   # would otherwise be the top long
    w = rank_to_weights(pred, decile=0.2, gross_exposure=1.0, min_names=5)
    assert w["S9"] == 0.0
    # k = int(9 * 0.2) = 1 -> single long is now S8, single short is S0.
    assert w["S8"] == pytest.approx(0.5)
    assert w["S0"] == pytest.approx(-0.5)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)


def test_legs_never_overlap_at_large_decile():
    pred = pd.Series({f"S{i}": float(i) for i in range(4)})
    w = rank_to_weights(pred, decile=0.9, gross_exposure=1.0, min_names=2)
    # k is capped at len//2 so a name is never both long and short.
    assert (w > 0).sum() == (w < 0).sum() == 2
    assert w.sum() == pytest.approx(0.0, abs=1e-12)


# Loose by design — see test_stale_prediction_earns_nothing_lookahead_guard.
# Separates "no edge" from "reading the future", not a tight estimate of zero.
NO_EDGE_SHARPE = 2.0


def _synthetic_panel(n_dates=500, n_symbols=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_dates)
    cols = [f"S{i}" for i in range(n_symbols)]
    ret = pd.DataFrame(
        rng.normal(0.0, 0.02, (n_dates, n_symbols)), index=idx, columns=cols
    )
    return ret


def _sharpe(book_ret):
    r = book_ret.dropna()
    return float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else 0.0


def test_perfect_foresight_produces_large_sharpe():
    """If the engine cannot profit from tomorrow's actual returns, it is broken."""
    ret = _synthetic_panel()
    cfg = PanelConfig(min_names=5, cost_bps=0.0, borrow_bps_annual=0.0)
    res = run_panel(pred=ret, ret=ret, cfg=cfg)   # pred[t] == ret[t] == tomorrow
    assert _sharpe(res.book_ret) > 3.0


def test_stale_prediction_earns_nothing_lookahead_guard():
    """pred[t] = ret[t-1] is knowable at close[t] and predicts nothing.

    If this scores well, the engine is paying ret[t-1] instead of ret[t] — i.e.
    weights are earning the return they were derived from. This is the #106 bug
    class (close[t] decision filled at close[t]) in panel form.

    NO_EDGE_SHARPE is deliberately loose. This test separates "no edge" from
    "reading the future", which differ by an order of magnitude (perfect
    foresight scores >10 on this panel) — it is not a tight estimate of zero.
    Over 1500 days the standard error of an annualized Sharpe under the null is
    ~sqrt(252/1500) ~= 0.41, so a 2.0 bound is ~5 SE: robust to reseeding, and
    still nowhere near a look-ahead's score.
    """
    ret = _synthetic_panel(n_dates=1500)
    stale = ret.shift(1)
    cfg = PanelConfig(min_names=5, cost_bps=0.0, borrow_bps_annual=0.0)
    res = run_panel(pred=stale, ret=ret, cfg=cfg)
    assert abs(_sharpe(res.book_ret)) < NO_EDGE_SHARPE


def test_random_predictions_produce_no_edge():
    ret = _synthetic_panel(n_dates=1500, seed=1)
    rng = np.random.default_rng(99)
    noise = pd.DataFrame(
        rng.normal(0, 1, ret.shape), index=ret.index, columns=ret.columns
    )
    cfg = PanelConfig(min_names=5, cost_bps=0.0, borrow_bps_annual=0.0)
    res = run_panel(pred=noise, ret=ret, cfg=cfg)
    assert abs(_sharpe(res.book_ret)) < NO_EDGE_SHARPE


def test_book_is_dollar_neutral_every_traded_day():
    ret = _synthetic_panel()
    rng = np.random.default_rng(7)
    noise = pd.DataFrame(
        rng.normal(0, 1, ret.shape), index=ret.index, columns=ret.columns
    )
    res = run_panel(pred=noise, ret=ret, cfg=PanelConfig(min_names=5))
    assert res.weights.sum(axis=1).abs().max() < 1e-12


def test_zero_turnover_charges_zero_cost():
    """Constant predictions -> same weights every day -> no turnover after day 1."""
    ret = _synthetic_panel(n_dates=100, n_symbols=20)
    const = pd.DataFrame(
        np.tile(np.arange(20, dtype=float), (100, 1)),
        index=ret.index, columns=ret.columns,
    )
    cfg = PanelConfig(min_names=5, cost_bps=5.0, borrow_bps_annual=0.0)
    res = run_panel(pred=const, ret=ret, cfg=cfg)
    # Day 1 establishes the book (turnover == gross); every later day is flat.
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    assert res.turnover.iloc[1:].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_borrow_accrues_on_short_notional_only():
    ret = _synthetic_panel(n_dates=100, n_symbols=20)
    const = pd.DataFrame(
        np.tile(np.arange(20, dtype=float), (100, 1)),
        index=ret.index, columns=ret.columns,
    )
    cfg = PanelConfig(min_names=5, cost_bps=0.0, borrow_bps_annual=50.0,
                      gross_exposure=1.0)
    res = run_panel(pred=const, ret=ret, cfg=cfg)
    # Short notional is gross/2 = 0.5; daily borrow = 0.5 * 50/1e4/252.
    expected = 0.5 * (50.0 / 1e4 / 252.0)
    assert res.diagnostics["mean_borrow_cost"] == pytest.approx(expected, rel=1e-6)


def test_costs_reduce_returns_versus_gross():
    ret = _synthetic_panel(seed=3)
    rng = np.random.default_rng(11)
    noise = pd.DataFrame(
        rng.normal(0, 1, ret.shape), index=ret.index, columns=ret.columns
    )
    free = run_panel(noise, ret, PanelConfig(min_names=5, cost_bps=0.0,
                                            borrow_bps_annual=0.0))
    charged = run_panel(noise, ret, PanelConfig(min_names=5, cost_bps=20.0,
                                                borrow_bps_annual=50.0))
    assert charged.book_ret.mean() < free.book_ret.mean()
    pd.testing.assert_series_equal(free.gross_ret, charged.gross_ret)
