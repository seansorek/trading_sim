import numpy as np
import pandas as pd
import pytest

from panel_backtester import rank_to_weights, sector_neutralize, PanelConfig, run_panel


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
    """Constant predictions AND flat returns -> nothing to trade after day 1.

    Returns must be flat for this to hold: once positions drift, restoring the
    target weights is a real trade even when the rankings never change. That
    rebalancing turnover is asserted separately below.
    """
    ret = _synthetic_panel(n_dates=100, n_symbols=20) * 0.0
    const = pd.DataFrame(
        np.tile(np.arange(20, dtype=float), (100, 1)),
        index=ret.index, columns=ret.columns,
    )
    cfg = PanelConfig(min_names=5, cost_bps=5.0, borrow_bps_annual=0.0)
    res = run_panel(pred=const, ret=ret, cfg=cfg)
    # Day 1 establishes the book (turnover == gross); every later day is flat.
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    assert res.turnover.iloc[1:].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert res.diagnostics["mean_cost"] == pytest.approx(
        (1.0 * 5.0 / 1e4) / len(ret), rel=1e-9
    )


def test_drifted_positions_cost_turnover_to_restore():
    """Same names, same ranks — but the weights moved, so the trade is real."""
    ret = _synthetic_panel(n_dates=100, n_symbols=20)
    const = pd.DataFrame(
        np.tile(np.arange(20, dtype=float), (100, 1)),
        index=ret.index, columns=ret.columns,
    )
    res = run_panel(pred=const, ret=ret, cfg=PanelConfig(min_names=5))
    later = res.turnover.iloc[1:]
    assert later.max() > 0            # drift has to be traded back
    assert later.mean() < 0.2         # but it is far cheaper than a full re-book


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


# --- beta neutralization ---------------------------------------------------

def _pred_row(n=40):
    return pd.Series(np.arange(n, dtype=float), index=[f"S{i:02d}" for i in range(n)])


def test_beta_row_equalizes_leg_beta_exposure():
    """The long leg holds the high-beta names; sizing must cancel the exposure."""
    pred = _pred_row()
    beta = pd.Series(np.linspace(0.5, 2.0, len(pred)), index=pred.index)
    w = rank_to_weights(pred, decile=0.25, gross_exposure=1.0, min_names=20, beta_row=beta)

    assert float((w * beta).sum()) == pytest.approx(0.0, abs=1e-12)
    assert w.abs().sum() == pytest.approx(1.0)      # gross preserved
    assert w.sum() < 0                              # dollar neutrality traded away


def test_conviction_tilts_within_leg_but_preserves_every_invariant():
    """Conviction redistributes inside a leg — it must not move the leg totals.

    Gross exposure, dollar neutrality and beta neutrality are all encoded in the
    two leg notionals, so conviction weighting is only safe if those are exactly
    preserved.
    """
    pred = _pred_row()
    beta = pd.Series(np.linspace(0.5, 2.0, len(pred)), index=pred.index)
    kw = dict(decile=0.25, gross_exposure=1.0, min_names=20)

    flat = rank_to_weights(pred, **kw)
    conv = rank_to_weights(pred, **kw, conviction=True)

    assert conv.abs().sum() == pytest.approx(1.0)                    # gross
    assert conv.sum() == pytest.approx(0.0)                          # dollar neutral
    assert conv[conv > 0].sum() == pytest.approx(flat[flat > 0].sum())
    assert conv[conv < 0].sum() == pytest.approx(flat[flat < 0].sum())
    assert not np.allclose(conv.values, flat.values), "conviction did not tilt anything"

    # Beta neutrality survives too.
    cb = rank_to_weights(pred, **kw, beta_row=beta, conviction=True)
    assert cb.abs().sum() == pytest.approx(1.0)


def test_conviction_weights_the_most_extreme_name_highest():
    """Monotone in |pred - centre|, and bounded by the concentration cap."""
    pred = _pred_row()
    w = rank_to_weights(pred, decile=0.25, gross_exposure=1.0, min_names=20,
                        conviction=True)
    longs = w[w > 0].sort_index()
    # S39 is the most extreme prediction, S30 the leg boundary.
    assert longs["S39"] > longs["S30"]
    assert longs.max() / longs.min() <= 4.0 + 1e-9   # CONVICTION_CAP ** 2


def test_conviction_falls_back_to_equal_weight_on_a_flat_cross_section():
    """No dispersion means no conviction information — do not divide by zero."""
    pred = pd.Series(3.0, index=[f"S{i:02d}" for i in range(40)])
    w = rank_to_weights(pred, decile=0.25, gross_exposure=1.0, min_names=20,
                        conviction=True)
    nz = w[w != 0].abs()
    assert nz.nunique() == 1
    assert w.abs().sum() == pytest.approx(1.0)


def test_without_beta_row_the_book_stays_dollar_neutral():
    pred = _pred_row()
    w = rank_to_weights(pred, decile=0.25, gross_exposure=1.0, min_names=20)
    assert w.sum() == pytest.approx(0.0)
    assert w.abs().sum() == pytest.approx(1.0)


def test_unusable_leg_beta_falls_back_to_dollar_neutral():
    """A near-zero or negative leg beta must not blow up the notional solve."""
    pred = _pred_row()
    beta = pd.Series(0.0, index=pred.index)
    w = rank_to_weights(pred, decile=0.25, gross_exposure=1.0, min_names=20, beta_row=beta)
    assert w.sum() == pytest.approx(0.0)
    assert w.abs().sum() == pytest.approx(1.0)

    nan_beta = pd.Series(np.nan, index=pred.index)
    w2 = rank_to_weights(pred, decile=0.25, gross_exposure=1.0, min_names=20, beta_row=nan_beta)
    assert w2.sum() == pytest.approx(0.0)


def test_run_panel_with_beta_reports_near_zero_ex_ante_beta():
    idx = pd.bdate_range("2021-01-01", periods=120)
    cols = [f"S{i:02d}" for i in range(40)]
    rng = np.random.default_rng(7)
    pred = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
    ret = pd.DataFrame(rng.normal(0, 0.01, (len(idx), len(cols))), index=idx, columns=cols)
    # Beta correlated with the prediction — the exact bias found on the live panel.
    beta = 1.0 + 0.4 * pred

    cfg = PanelConfig(decile=0.25, min_names=20)
    with_beta = run_panel(pred, ret, cfg, beta=beta)
    without = run_panel(pred, ret, cfg)

    assert abs(with_beta.diagnostics["mean_ex_ante_beta"]) < 1e-9
    assert "mean_ex_ante_beta" not in without.diagnostics


# --- weight drift between rebalances ---------------------------------------

def _flat_panel(n_dates=12, n_syms=40, ret_value=0.0):
    idx = pd.bdate_range("2022-01-03", periods=n_dates)
    cols = [f"S{i:02d}" for i in range(n_syms)]
    rng = np.random.default_rng(3)
    pred = pd.DataFrame(rng.normal(size=(n_dates, n_syms)), index=idx, columns=cols)
    ret = pd.DataFrame(ret_value, index=idx, columns=cols)
    return pred, ret


def test_no_turnover_charged_on_hold_days():
    pred, ret = _flat_panel()
    res = run_panel(pred, ret, PanelConfig(decile=0.25, rebalance_days=5, min_names=20))
    # Rebalance on i = 0, 5, 10; every other day is a hold.
    held = [i for i in range(len(pred)) if i % 5 != 0]
    assert res.turnover.iloc[held].abs().max() == pytest.approx(0.0)
    assert res.turnover.iloc[0] > 0


def test_weights_drift_with_returns_between_rebalances():
    """A held long position that gains 10% is a bigger position tomorrow.

    Re-pegging it to the original weight instead would require an untracked,
    uncharged daily trade.
    """
    pred, ret = _flat_panel(ret_value=0.10)
    res = run_panel(pred, ret, PanelConfig(decile=0.25, rebalance_days=5, min_names=20))

    day0 = res.weights.iloc[0]
    day1 = res.weights.iloc[1]
    longs = day0[day0 > 0].index
    np.testing.assert_allclose(day1[longs].values, day0[longs].values * 1.10, rtol=1e-12)


def test_daily_rebalance_is_unaffected_by_the_drift_path():
    """rebalance_days=1 never takes the hold branch, so prior results stand."""
    pred, ret = _flat_panel(n_dates=40)
    ret = ret + 0.001
    cfg = PanelConfig(decile=0.25, rebalance_days=1, min_names=20)
    res = run_panel(pred, ret, cfg)
    for i in range(len(pred)):
        target = rank_to_weights(pred.iloc[i], 0.25, cfg.gross_exposure, 20)
        np.testing.assert_allclose(res.weights.iloc[i].values, target.values, atol=1e-12)


# --- sector neutralization -------------------------------------------------

def _sector_panel():
    idx = pd.bdate_range("2023-01-02", periods=8)
    cols = [f"T{i}" for i in range(5)] + [f"F{i}" for i in range(5)] + ["X0", "X1"]
    rng = np.random.default_rng(21)
    pred = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
    # A strong sector-level tilt: every Tech name is shifted up together.
    pred[[c for c in cols if c.startswith("T")]] += 5.0
    sector_of = {c: ("Tech" if c.startswith("T") else "Fin") for c in cols if not c.startswith("X")}
    return pred, sector_of


def test_sector_neutralize_zeroes_each_sector_mean_per_date():
    pred, sector_of = _sector_panel()
    out = sector_neutralize(pred, sector_of)
    for sector in ("Tech", "Fin"):
        cols = [c for c, s in sector_of.items() if s == sector]
        np.testing.assert_allclose(out[cols].mean(axis=1).values, 0.0, atol=1e-12)


def test_sector_tilt_no_longer_dominates_the_long_leg():
    """The point of the whole exercise: a +5 tilt on Tech must stop buying Tech."""
    pred, sector_of = _sector_panel()
    tech = [c for c in pred.columns if c.startswith("T")]

    raw_w = rank_to_weights(pred.iloc[0], decile=0.4, gross_exposure=1.0, min_names=5)
    neu_w = rank_to_weights(
        sector_neutralize(pred, sector_of).iloc[0], decile=0.4, gross_exposure=1.0, min_names=5
    )
    raw_long = raw_w[raw_w > 0]
    neu_long = neu_w[neu_w > 0]
    assert raw_long.index.isin(tech).all()        # every long is Tech
    assert not neu_long.index.isin(tech).all()    # no longer


def test_unmapped_symbols_and_thin_sectors_pass_through():
    pred, sector_of = _sector_panel()
    out = sector_neutralize(pred, sector_of)
    # X0/X1 have no sector at all.
    pd.testing.assert_frame_equal(out[["X0", "X1"]], pred[["X0", "X1"]])

    # A 2-name sector is below MIN_SECTOR_NAMES and must be left alone.
    thin = {"T0": "Tiny", "T1": "Tiny"}
    pd.testing.assert_frame_equal(sector_neutralize(pred, thin), pred)


def test_sector_neutralize_preserves_nan_and_is_a_noop_without_sectors():
    pred, sector_of = _sector_panel()
    pred.iloc[0, 0] = np.nan
    out = sector_neutralize(pred, sector_of)
    assert np.isnan(out.iloc[0, 0])
    pd.testing.assert_frame_equal(sector_neutralize(pred, {}), pred)
