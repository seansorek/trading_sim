import numpy as np
import pandas as pd
import pytest

from daily_features import FEATURE_COLS
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


class CSFakePredictor(FakePredictor):
    """Same scorer, but flagged as trained on cross-sectionally ranked features."""

    cs_mode = "replace"


class AugFakePredictor:
    """Scores the first per-date-rank column, i.e. the start of the second axis.

    Not the last column: without spy_df the ret_*_vs_spy features are constant
    0.0, so their ranks are all ties and would prove nothing.
    """

    cs_mode = "augment"

    def predict(self, X):
        return X[:, len(FEATURE_COLS)].astype(float), None


def _six_symbol_bars():
    return {f"S{i}": _make_bars(n=400, seed=20 + i) for i in range(6)}


def test_cs_normalized_model_gets_per_date_ranks_not_raw_features():
    bars = _six_symbol_bars()
    out = build_panels(
        list(bars), "2020-01-01", "2021-12-31", db=None,
        predictor=CSFakePredictor(), load_fn=_loader(bars),
    )
    # FakePredictor scores X[:, 0], so every score IS the rank of ret_1d.
    full = out.pred.dropna()
    assert len(full) > 100
    np.testing.assert_allclose(full.max(axis=1).values, 1.0)
    # 6 names -> ranks evenly spaced from 2*(1/6)-1 to 1.0
    expected = np.linspace(2.0 / 6 - 1.0, 1.0, 6)
    for _, row in full.iterrows():
        np.testing.assert_allclose(np.sort(row.values), expected)


def test_predictor_without_the_flag_still_sees_raw_features():
    bars = _six_symbol_bars()
    raw = build_panels(
        list(bars), "2020-01-01", "2021-12-31", db=None,
        predictor=FakePredictor(), load_fn=_loader(bars),
    )
    ranked = build_panels(
        list(bars), "2020-01-01", "2021-12-31", db=None,
        predictor=CSFakePredictor(), load_fn=_loader(bars),
    )
    assert raw.pred.shape == ranked.pred.shape
    assert not np.allclose(raw.pred.dropna().values, ranked.pred.dropna().values)
    # Raw z-scores are unbounded; ranks are not.
    assert raw.pred.abs().max().max() > 1.0


def test_cs_mode_argument_overrides_the_models_own_setting():
    bars = _six_symbol_bars()
    forced = build_panels(
        list(bars), "2020-01-01", "2021-12-31", db=None,
        predictor=CSFakePredictor(), load_fn=_loader(bars), cs_mode="off",
    )
    assert forced.pred.abs().max().max() > 1.0


def test_augment_mode_serves_both_axes_in_contract_order():
    """60 columns: the 30 per-symbol z-scores, then their 30 per-date ranks."""
    bars = _six_symbol_bars()
    seen = {}

    class Recorder(AugFakePredictor):
        def predict(self, X):
            seen["shape"] = X.shape
            return super().predict(X)

    out = build_panels(
        list(bars), "2020-01-01", "2021-12-31", db=None,
        predictor=Recorder(), load_fn=_loader(bars),
    )
    assert seen["shape"][1] == 2 * len(FEATURE_COLS)
    # Column 30 is a rank, so scores are bounded; the raw axis is not.
    assert out.pred.abs().max().max() == 1.0
    raw = build_panels(
        list(bars), "2020-01-01", "2021-12-31", db=None,
        predictor=FakePredictor(), load_fn=_loader(bars),
    )
    assert raw.pred.abs().max().max() > 1.0
