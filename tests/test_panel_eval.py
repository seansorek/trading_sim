import numpy as np
import pandas as pd
import pytest

from panel_backtester import PanelResult
from panel_eval import (
    CONFIG_GRID,
    book_beta,
    cross_sectional_ic,
    evaluate_grid,
    forward_return,
    ic_report,
)


def _result(book_ret):
    idx = book_ret.index
    return PanelResult(
        equity=(1 + book_ret).cumprod() * 100_000,
        book_ret=book_ret,
        gross_ret=book_ret,
        weights=pd.DataFrame(0.0, index=idx, columns=["A"]),
        turnover=pd.Series(0.0, index=idx),
        diagnostics={},
    )


def _idx(n=750):
    return pd.bdate_range("2020-01-01", periods=n)


def test_book_beta_recovers_a_known_beta():
    idx = _idx()
    rng = np.random.default_rng(0)
    spy = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    book = 0.5 * spy + pd.Series(rng.normal(0, 0.0001, len(idx)), index=idx)
    assert book_beta(book, spy) == pytest.approx(0.5, abs=0.02)


def test_book_beta_of_uncorrelated_book_is_near_zero():
    idx = _idx()
    rng = np.random.default_rng(1)
    spy = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    book = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    assert abs(book_beta(book, spy)) < 0.1


def test_grid_spans_both_deciles_and_excludes_cost_bps():
    assert len(CONFIG_GRID) == 8
    assert set(CONFIG_GRID) == {
        (0.1, 1), (0.1, 3), (0.1, 5), (0.1, 10),
        (0.2, 1), (0.2, 3), (0.2, 5), (0.2, 10),
    }
    # cost_bps must never become a grid axis — see the note on CONFIG_GRID.
    assert all(len(cfg) == 2 for cfg in CONFIG_GRID)


def test_high_beta_fails_the_gate_before_performance_is_read():
    idx = _idx()
    rng = np.random.default_rng(2)
    spy = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    # Strong returns, but the book is just levered SPY -> neutralization failed.
    book = 0.9 * spy + 0.002
    results = {cfg: _result(book) for cfg in CONFIG_GRID}
    out = evaluate_grid(results, spy)
    assert out["passed"] is False
    assert "beta" in out["verdict"].lower()


def test_no_edge_fails_the_dsr_gate():
    idx = _idx()
    rng = np.random.default_rng(3)
    spy = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    results = {
        cfg: _result(pd.Series(rng.normal(0, 0.01, len(idx)), index=idx))
        for cfg in CONFIG_GRID
    }
    out = evaluate_grid(results, spy)
    assert out["passed"] is False
    assert out["dsr"] < 0.95


def test_strong_neutral_book_passes_the_gate():
    idx = _idx()
    rng = np.random.default_rng(4)
    spy = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    results = {
        cfg: _result(pd.Series(0.0016 + rng.normal(0, 0.004, len(idx)), index=idx))
        for cfg in CONFIG_GRID
    }
    out = evaluate_grid(results, spy)
    assert abs(out["beta"]) < 0.1
    assert out["dsr"] > 0.95
    assert out["passed"] is True


# --- cross-sectional IC diagnostics ---------------------------------------

_SYMS = [f"S{i:03d}" for i in range(60)]


def _panel(values, idx):
    return pd.DataFrame(values, index=idx, columns=_SYMS)


def test_cross_sectional_ic_recovers_real_ranking_skill():
    idx, rng = _idx(300), np.random.default_rng(10)
    fwd = _panel(rng.normal(0, 0.02, (300, len(_SYMS))), idx)
    pred = fwd + rng.normal(0, 0.02, fwd.shape)  # ~half signal, half noise
    ic = cross_sectional_ic(pred, fwd)
    assert ic.notna().all()
    assert ic.mean() > 0.4


def test_market_only_forecast_scores_pooled_but_not_per_date():
    """The failure mode this diagnostic exists to catch.

    pred forecasts the market move perfectly and carries no information about
    which name beats which. Pooled IC looks strong; the per-date IC — the only
    part a rank-based book can trade — is zero.
    """
    idx, rng = _idx(400), np.random.default_rng(11)
    market = rng.normal(0, 0.012, len(idx))
    idio = rng.normal(0, 0.02, (len(idx), len(_SYMS)))
    ret = _panel(market[:, None] + idio, idx)
    close = (1 + ret).cumprod() * 100.0
    # Perfect knowledge of tomorrow's market, zero cross-sectional view. The
    # jitter only breaks rank ties; it is independent of every return.
    fwd1 = forward_return(close, 1)
    pred = _panel(
        np.roll(market, -1)[:, None] + rng.normal(0, 1e-6, (len(idx), len(_SYMS))), idx
    )

    per_date = cross_sectional_ic(pred, fwd1).mean()
    out = ic_report(pred, close, horizons=(1,))

    assert abs(per_date) < 0.05
    assert out["pooled_ic_at_primary"] > 0.2
    assert abs(out["horizons"]["1"]["ic_ir"]) < 0.2


def test_dates_below_min_names_are_nan_not_zero():
    idx, rng = _idx(50), np.random.default_rng(12)
    fwd = _panel(rng.normal(0, 0.02, (50, len(_SYMS))), idx)
    pred = fwd.copy()
    pred.iloc[0, 5:] = np.nan  # only 5 usable names on day 0
    ic = cross_sectional_ic(pred, fwd, min_names=20)
    assert np.isnan(ic.iloc[0])
    assert ic.iloc[1:].notna().all()
