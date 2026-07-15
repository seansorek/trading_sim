import numpy as np
import pandas as pd
import pytest

from panel_backtester import rank_to_weights


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
