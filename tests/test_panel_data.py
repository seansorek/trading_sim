import numpy as np
import pandas as pd
import pytest

from panel_data import build_panels, MAX_LOAD_FAILURE_FRAC


class FakePredictor:
    """Returns the first feature column as the score. Deterministic, no pickle."""

    def predict(self, X):
        return X[:, 0].astype(float), None


def _make_bars(n=400, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


def _loader(bars_by_symbol):
    def load_fn(symbol, start, end, db):
        return bars_by_symbol.get(symbol)
    return load_fn


def test_ret_panel_is_one_day_return_not_the_three_day_target():
    """ret must be close[t]->close[t+1], NOT daily_features.fwd_ret_1d (3-bar cumulative)."""
    bars = {"AAA": _make_bars(seed=1)}
    pd_out = build_panels(
        ["AAA"], "2020-01-01", "2021-06-01", db=None,
        predictor=FakePredictor(), load_fn=_loader(bars),
    )
    close = pd_out.close["AAA"]
    expected = close.pct_change().shift(-1)
    pd.testing.assert_series_equal(
        pd_out.ret["AAA"], expected, check_names=False,
    )
    # A 3-day cumulative return would be ~3x larger in magnitude on average.
    three_day = (close.shift(-3) / close) - 1
    assert not np.allclose(
        pd_out.ret["AAA"].dropna().values[:50],
        three_day.dropna().values[:50],
    ), "ret_panel is using the 3-day horizon target — see Global Constraints"


def test_ragged_histories_align_outer_without_forward_fill():
    bars = {
        "AAA": _make_bars(n=400, seed=1, start="2020-01-01"),
        "BBB": _make_bars(n=300, seed=2, start="2020-06-01"),
    }
    out = build_panels(
        ["AAA", "BBB"], "2020-01-01", "2021-12-31", db=None,
        predictor=FakePredictor(), load_fn=_loader(bars),
    )
    assert set(out.pred.columns) == {"AAA", "BBB"}
    # BBB starts later: its earliest rows must be NaN, not back/forward-filled.
    assert out.pred["BBB"].isna().any()
    first_bbb = out.pred["BBB"].first_valid_index()
    assert out.pred.loc[out.pred.index < first_bbb, "BBB"].isna().all()


def test_symbol_with_insufficient_history_is_dropped_and_recorded():
    bars = {
        "AAA": _make_bars(n=400, seed=1),
        "TINY": _make_bars(n=60, seed=3),
        "BBB": _make_bars(n=400, seed=4),
        "CCC": _make_bars(n=400, seed=5),
        "DDD": _make_bars(n=400, seed=6),
        "EEE": _make_bars(n=400, seed=7),
        "FFF": _make_bars(n=400, seed=8),
        "GGG": _make_bars(n=400, seed=9),
        "HHH": _make_bars(n=400, seed=10),
        "III": _make_bars(n=400, seed=11),
        "JJJ": _make_bars(n=400, seed=12),
    }
    out = build_panels(
        list(bars.keys()), "2020-01-01", "2021-12-31", db=None,
        predictor=FakePredictor(), load_fn=_loader(bars), min_bars=250,
    )
    assert "TINY" not in out.pred.columns
    assert "TINY" in out.dropped
    assert "insufficient" in out.dropped["TINY"]


def test_raises_when_too_much_of_the_universe_fails_to_load():
    bars = {"AAA": _make_bars(n=400, seed=1)}
    symbols = ["AAA"] + [f"MISSING{i}" for i in range(9)]
    with pytest.raises(RuntimeError, match="Refusing to backtest"):
        build_panels(
            symbols, "2020-01-01", "2021-12-31", db=None,
            predictor=FakePredictor(), load_fn=_loader(bars),
        )
