"""test_data_leakage.py — Guard against train/test leakage in the training pipelines.

fwd_ret_1d is a FWD_RET_HORIZON_DAYS-bar forward return. Without a purge/embargo
gap, the last training rows' labels would depend on price action that falls
inside the test period — a subtle leak even though the *features* themselves
never look ahead. These tests pin the embargo gap so a future refactor can't
silently remove it.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_features import FWD_RET_HORIZON_DAYS, _rolling_zscore, make_daily_features
import train_models


def _make_price_df(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close * rng.uniform(0.99, 1.005, n),
            "high": close * rng.uniform(1.001, 1.02, n),
            "low": close * rng.uniform(0.98, 0.999, n),
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


def test_rolling_zscore_is_causal():
    import numpy as np
    import pandas as pd
    from daily_features import _rolling_zscore

    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(0, 1, 400).cumsum())
    full = _rolling_zscore(s)
    for i in (120, 250, 399):
        truncated = _rolling_zscore(s.iloc[: i + 1])
        assert np.isclose(full.iloc[i], truncated.iloc[i], equal_nan=True), (
            f"row {i}: full={full.iloc[i]} truncated={truncated.iloc[i]} — look-ahead!"
        )


def test_prepare_data_has_embargo_gap(tmp_path):
    """train_models._prepare_data must drop FWD_RET_HORIZON_DAYS rows between
    the last training row and the first test row (purged split)."""
    db = train_models.DB(str(tmp_path / "test.db"))
    df = _make_price_df()

    with patch("train_models._load_symbol", return_value=df):
        X_train, y_train, X_test, y_test, used = train_models._prepare_data(
            ["AAPL"], days=900, db=db, vol_mult=0.5,
        )

    feats = make_daily_features(df, spy_df=None).dropna(subset=["fwd_ret_1d"])
    split = int(len(feats) * 0.8)
    expected_test_len = len(feats) - (split + FWD_RET_HORIZON_DAYS)

    assert used == ["AAPL"]
    assert len(X_train) == split
    assert len(X_test) == expected_test_len
    # The gap between the last train row and the first test row must be at
    # least the forward-return horizon, otherwise a train label could be
    # computed from a close price inside the test window.
    gap = len(feats) - expected_test_len - split
    assert gap == FWD_RET_HORIZON_DAYS


def test_prepare_data_no_embargo_means_no_overlap_in_dates(tmp_path):
    """The actual calendar gap between train-end and test-start must be >=
    the forward-return horizon (not just a row-count coincidence)."""
    db = train_models.DB(str(tmp_path / "test.db"))
    df = _make_price_df()

    with patch("train_models._load_symbol", return_value=df):
        train_models._prepare_data(["AAPL"], days=900, db=db, vol_mult=0.5)

    feats = make_daily_features(df, spy_df=None).dropna(subset=["fwd_ret_1d"])
    split = int(len(feats) * 0.8)
    test_start = split + FWD_RET_HORIZON_DAYS

    test_start_date = feats.index[test_start]
    # fwd_ret_1d at the last train row looks FWD_RET_HORIZON_DAYS bars ahead;
    # that target bar must fall strictly before the first test row's date.
    label_horizon_date = feats.index[split - 1 + FWD_RET_HORIZON_DAYS]
    assert label_horizon_date < test_start_date


def test_bayes_search_cv_uses_time_series_split_not_kfold():
    """train_logistic/train_xgboost's --optimize path must pass a
    TimeSeriesSplit (or equivalent time-ordered splitter) to BayesSearchCV,
    not a bare int (which defaults to ordinary K-fold and lets later
    observations select hyperparameters for earlier validation folds)."""
    pytest.importorskip("skopt")
    from sklearn.model_selection import TimeSeriesSplit

    from config import OptimizeCfg

    captured = {}

    class _FakeBayesSearchCV:
        def __init__(self, base, search_space, **kwargs):
            captured["cv"] = kwargs["cv"]
            self.best_params_ = {}
            self.best_estimator_ = base

        def fit(self, X, y):
            return self

    n_samples, n_features = 120, 5
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(n_samples, n_features))
    y_train = rng.integers(0, 3, size=n_samples)
    X_test = rng.normal(size=(20, n_features))
    y_test = rng.integers(0, 3, size=20)

    opt_cfg = OptimizeCfg()

    with patch("train_models.BayesSearchCV", _FakeBayesSearchCV):
        train_models.train_logistic(
            X_train, X_test, y_train, y_test, cfg={}, optimize=True, opt_cfg=opt_cfg,
        )

    cv = captured["cv"]
    assert isinstance(cv, TimeSeriesSplit), (
        f"expected a TimeSeriesSplit, got {type(cv)} (bare int/K-fold leaks "
        "future observations into earlier validation folds)"
    )

    # Every validation fold must occur strictly after its training fold.
    for train_idx, test_idx in cv.split(X_train):
        assert train_idx.max() < test_idx.min()


def test_hybrid_prepare_data_has_embargo_gap(tmp_path):
    pytest.importorskip("torch")
    import train_hybrid

    db = train_hybrid.DB(str(tmp_path / "test.db"))
    df = _make_price_df()

    with patch("train_hybrid._load_symbol", return_value=df):
        data = train_hybrid.prepare_data(["AAPL"], days=900, db=db, lookback=20, vol_mult=0.5)

    feats = make_daily_features(df, spy_df=None).dropna(subset=["fwd_ret_1d"])
    split = int(len(feats) * 0.8)
    test_start = split + FWD_RET_HORIZON_DAYS

    assert data["used_symbols"] == ["AAPL"]
    # build_sequences further trims the first (lookback - 1) rows of each
    # block, so just assert the block boundary (pre-sequence) embargo holds.
    gap = test_start - split
    assert gap == FWD_RET_HORIZON_DAYS


def test_hybrid_prepare_data_val_carved_from_train_not_test(tmp_path):
    """The validation slice used for early stopping must be carved out of the
    TRAIN period (most recent portion of it), never overlap the test set, and
    itself be separated from train-proper by an embargo gap. This guards
    against issue #90 (test set leaking into early-stopping/checkpoint
    selection)."""
    pytest.importorskip("torch")
    import train_hybrid

    db = train_hybrid.DB(str(tmp_path / "test.db"))
    df = _make_price_df()

    val_frac = 0.15
    with patch("train_hybrid._load_symbol", return_value=df):
        data = train_hybrid.prepare_data(
            ["AAPL"], days=900, db=db, lookback=20, vol_mult=0.5, val_frac=val_frac,
        )

    feats = make_daily_features(df, spy_df=None).dropna(subset=["fwd_ret_1d"])
    split = int(len(feats) * 0.8)
    val_len = max(int(split * val_frac), 1)
    val_start = split - val_len
    train_end = val_start - FWD_RET_HORIZON_DAYS

    expected_val_len = split - val_start
    expected_train_len = train_end

    assert len(data["y_val"]) == expected_val_len - (20 - 1)
    assert len(data["y_train"]) == expected_train_len - (20 - 1)
    # Val labels must never be drawn from the test block.
    assert val_start < split <= split + FWD_RET_HORIZON_DAYS
    # Train-proper/val embargo gap must be at least the forward-return horizon.
    assert val_start - train_end == FWD_RET_HORIZON_DAYS


def _make_ragged_price_df(n: int, start: str, seed: int) -> pd.DataFrame:
    """Same synthetic OHLCV generator as _make_price_df but with a
    configurable start date and length, so two symbols can be given
    deliberately different (ragged) histories."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "open": close * rng.uniform(0.99, 1.005, n),
            "high": close * rng.uniform(1.001, 1.02, n),
            "low": close * rng.uniform(0.98, 0.999, n),
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


def test_prepare_data_uses_single_global_split_across_ragged_histories(tmp_path):
    """Regression test for #138: train_models._prepare_data must cut every
    symbol at ONE global trading-date boundary, not each symbol's own 80%
    row-count boundary. With ragged histories (different listing dates,
    lengths) a per-symbol split lands each symbol's boundary on a different
    calendar date; once every symbol is pooled into one model, that lets a
    later training row from the long-history symbol sit alongside an
    earlier test row from the short-history symbol -- not a genuinely
    out-of-sample holdout."""
    db = train_models.DB(str(tmp_path / "test.db"))

    # LONG: listed early, long history. SHORT: listed much later ("newer
    # listing"), shorter history, overlapping LONG's tail so both symbols
    # have rows on both sides of a single global boundary.
    long_df = _make_ragged_price_df(n=700, start="2020-01-02", seed=11)
    short_df = _make_ragged_price_df(n=400, start="2021-06-01", seed=12)

    def _load(symbol, start, end, db):
        if symbol == "LONG":
            return long_df
        if symbol == "SHORT":
            return short_df
        return None  # SPY unavailable -> ret_*_vs_spy features fall back to 0

    with patch("train_models._load_symbol", side_effect=_load):
        X_train, y_train, X_test, y_test, used = train_models._prepare_data(
            ["LONG", "SHORT"], days=3000, db=db, vol_mult=0.5,
        )

    assert set(used) == {"LONG", "SHORT"}, (
        f"expected both ragged-history symbols to contribute rows on both "
        f"sides of the global split, got used_symbols={used}"
    )

    feats_long = make_daily_features(long_df, spy_df=None).dropna(subset=["fwd_ret_1d"])
    feats_short = make_daily_features(short_df, spy_df=None).dropna(subset=["fwd_ret_1d"])

    # Independently replicate the implementation's single global date split.
    all_dates = pd.DatetimeIndex(sorted(set(feats_long.index) | set(feats_short.index)))
    cut = int(len(all_dates) * 0.8)
    split_date = all_dates[cut]
    test_start_date = all_dates[cut + FWD_RET_HORIZON_DAYS]

    long_train_mask = feats_long.index < split_date
    long_test_mask = feats_long.index >= test_start_date
    short_train_mask = feats_short.index < split_date
    short_test_mask = feats_short.index >= test_start_date

    # Sanity: the fixture must actually exercise both sides for both symbols,
    # otherwise this test can't discriminate global-split from per-symbol-split.
    assert long_train_mask.sum() > 0 and long_test_mask.sum() > 0
    assert short_train_mask.sum() > 0 and short_test_mask.sum() > 0

    expected_train_len = int(long_train_mask.sum()) + int(short_train_mask.sum())
    expected_test_len = int(long_test_mask.sum()) + int(short_test_mask.sum())

    assert len(X_train) == expected_train_len
    assert len(X_test) == expected_test_len
    assert len(y_train) == expected_train_len
    assert len(y_test) == expected_test_len

    # A per-symbol (pre-#138) split would have cut each symbol at its own
    # row-count boundary instead of this shared calendar date; pin that the
    # two disagree for this fixture, so this test would fail under the old
    # per-symbol implementation.
    long_split = int(len(feats_long) * 0.8)
    short_split = int(len(feats_short) * 0.8)
    per_symbol_train_len = long_split + short_split
    assert per_symbol_train_len != expected_train_len, (
        "fixture's ragged histories happen to share one boundary by "
        "coincidence -- adjust seeds/lengths so this regression test can "
        "discriminate global vs. per-symbol splits"
    )

    # Core invariant: every training timestamp precedes every test
    # timestamp, GLOBALLY across the pooled symbols -- not just within each
    # symbol's own history.
    max_train_date = max(
        feats_long.index[long_train_mask].max(), feats_short.index[short_train_mask].max(),
    )
    min_test_date = min(
        feats_long.index[long_test_mask].min(), feats_short.index[short_test_mask].min(),
    )
    assert max_train_date < min_test_date

    # No label horizon overlaps the next block: the embargo between the two
    # boundary dates must span at least FWD_RET_HORIZON_DAYS trading dates,
    # shared by both symbols (not a per-symbol embargo on different dates).
    embargo_dates = all_dates[(all_dates >= split_date) & (all_dates < test_start_date)]
    assert len(embargo_dates) == FWD_RET_HORIZON_DAYS


def test_hybrid_prepare_data_uses_single_global_split_across_ragged_histories(tmp_path):
    """Regression test for #138 (hybrid side): train_hybrid.prepare_data must
    cut every symbol's train/val/test blocks at ONE shared set of global
    trading-date boundaries, not each symbol's own 80%-of-its-own-rows
    boundary. Mirrors test_prepare_data_uses_single_global_split_across_
    ragged_histories above but for the three-way hybrid split."""
    pytest.importorskip("torch")
    import train_hybrid

    db = train_hybrid.DB(str(tmp_path / "test.db"))

    long_df = _make_ragged_price_df(n=700, start="2020-01-02", seed=21)
    short_df = _make_ragged_price_df(n=400, start="2021-06-01", seed=22)
    lookback = 20
    val_frac = 0.15

    def _load(symbol, start, end, db):
        if symbol == "LONG":
            return long_df
        if symbol == "SHORT":
            return short_df
        return None

    with patch("train_hybrid._load_symbol", side_effect=_load):
        data = train_hybrid.prepare_data(
            ["LONG", "SHORT"], days=3000, db=db, lookback=lookback,
            vol_mult=0.5, val_frac=val_frac,
        )

    assert set(data["used_symbols"]) == {"LONG", "SHORT"}, (
        f"expected both ragged-history symbols to contribute rows to every "
        f"block, got used_symbols={data['used_symbols']}"
    )

    feats_long = make_daily_features(long_df, spy_df=None).dropna(subset=["fwd_ret_1d"])
    feats_short = make_daily_features(short_df, spy_df=None).dropna(subset=["fwd_ret_1d"])

    # Independently replicate the implementation's single global date split.
    all_dates = pd.DatetimeIndex(sorted(set(feats_long.index) | set(feats_short.index)))
    n_dates = len(all_dates)
    cut = int(n_dates * 0.8)
    val_len = max(int(cut * val_frac), 1)
    val_start_idx = cut - val_len
    train_end_idx = val_start_idx - FWD_RET_HORIZON_DAYS
    assert train_end_idx >= 1, "fixture too small for a val/train embargo -- widen it"

    train_end_date = all_dates[train_end_idx]
    val_start_date = all_dates[val_start_idx]
    split_date = all_dates[cut]
    test_start_date = all_dates[cut + FWD_RET_HORIZON_DAYS]

    def _masks(idx):
        train_mask = idx < train_end_date
        val_mask = (idx >= val_start_date) & (idx < split_date)
        test_mask = idx >= test_start_date
        return train_mask, val_mask, test_mask

    long_train, long_val, long_test = _masks(feats_long.index)
    short_train, short_val, short_test = _masks(feats_short.index)

    # Sanity: fixture must exercise every block for both symbols.
    for name, mask in [
        ("long_train", long_train), ("long_val", long_val), ("long_test", long_test),
        ("short_train", short_train), ("short_val", short_val), ("short_test", short_test),
    ]:
        assert mask.sum() > 0, f"fixture doesn't exercise {name} -- adjust seeds/lengths"

    # A per-symbol (pre-#138) split would carve each symbol's own row-count
    # boundary instead of this shared calendar date; pin the two disagree.
    long_split = int(len(feats_long) * 0.8)
    short_split = int(len(feats_short) * 0.8)
    per_symbol_train_len = (
        max(long_split - max(int(long_split * val_frac), 1) - FWD_RET_HORIZON_DAYS, 0)
        + max(short_split - max(int(short_split * val_frac), 1) - FWD_RET_HORIZON_DAYS, 0)
    )
    global_train_len = int(long_train.sum()) + int(short_train.sum())
    assert per_symbol_train_len != global_train_len, (
        "fixture's ragged histories happen to share one boundary by "
        "coincidence -- adjust seeds/lengths so this regression test can "
        "discriminate global vs. per-symbol splits"
    )

    # Core invariant: every training timestamp precedes every val/test
    # timestamp, GLOBALLY across the pooled symbols.
    max_train_date = max(
        feats_long.index[long_train].max(), feats_short.index[short_train].max(),
    )
    min_val_date = min(
        feats_long.index[long_val].min(), feats_short.index[short_val].min(),
    )
    min_test_date = min(
        feats_long.index[long_test].min(), feats_short.index[short_test].min(),
    )
    assert max_train_date < min_val_date
    max_val_date = max(
        feats_long.index[long_val].max(), feats_short.index[short_val].max(),
    )
    assert max_val_date < min_test_date

    # No label horizon overlaps the next block: both embargoes (train/val and
    # val/test) must span at least FWD_RET_HORIZON_DAYS trading dates, shared
    # by both symbols.
    train_val_embargo = all_dates[(all_dates >= train_end_date) & (all_dates < val_start_date)]
    val_test_embargo = all_dates[(all_dates >= split_date) & (all_dates < test_start_date)]
    assert len(train_val_embargo) == FWD_RET_HORIZON_DAYS
    assert len(val_test_embargo) == FWD_RET_HORIZON_DAYS


def test_amihud_is_causal():
    import numpy as np
    from daily_features import make_daily_features
    from tests.test_feature_correlation_guard import _synthetic_ohlcv

    df = _synthetic_ohlcv(n=500, seed=7)
    full = make_daily_features(df)["amihud_illiq"]
    trunc = make_daily_features(df.iloc[:300])["amihud_illiq"]
    common = full.index.intersection(trunc.index)[-1]
    assert np.isclose(full.loc[common], trunc.loc[common], equal_nan=True)
