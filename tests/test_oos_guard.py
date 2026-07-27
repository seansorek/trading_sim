"""tests/test_oos_guard.py — Regression tests for issue #115.

Pretrained-model backtests (simulate_multi.py, eval_report.py) must never
let rows at or before a model's training cutoff contribute to reported
performance. These tests pin the enforcement in oos_guard.py and its two
call sites.
"""
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_features import FWD_RET_HORIZON_DAYS
from oos_guard import enforce_oos_start, get_artifact_train_end


def _synth_prices(n=400, seed=0, start="2022-01-01"):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + 0.0005 + 0.01 * rng.standard_normal(n))
    idx = pd.date_range(start, periods=n, freq="B")
    high = close * 1.005
    low = close * 0.995
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


# ---------------------------------------------------------------------------
# get_artifact_train_end
# ---------------------------------------------------------------------------

def test_get_artifact_train_end_reads_field():
    assert get_artifact_train_end({"train_end": "2023-06-01"}) == pd.Timestamp("2023-06-01")


def test_get_artifact_train_end_missing_returns_none():
    assert get_artifact_train_end({}) is None


def test_get_artifact_train_end_bad_value_returns_none():
    assert get_artifact_train_end({"train_end": object()}) is None


# ---------------------------------------------------------------------------
# enforce_oos_start
# ---------------------------------------------------------------------------

def test_enforce_oos_start_drops_pre_cutoff_rows():
    df = _synth_prices(n=100)
    train_end = df.index[49]
    trimmed = enforce_oos_start(df, train_end, embargo_days=0)
    assert len(trimmed) < len(df)
    assert (trimmed.index > train_end).all()


def test_enforce_oos_start_applies_embargo_gap():
    df = _synth_prices(n=100)
    train_end = df.index[49]
    embargo = FWD_RET_HORIZON_DAYS
    trimmed = enforce_oos_start(df, train_end, embargo_days=embargo)
    cutoff = train_end + pd.Timedelta(days=embargo)
    assert (trimmed.index > cutoff).all()
    # Rows strictly between train_end and train_end+embargo must be excluded too.
    embargoed_rows = df.loc[(df.index > train_end) & (df.index <= cutoff)]
    assert len(embargoed_rows) >= 1
    assert not trimmed.index.isin(embargoed_rows.index).any()


def test_enforce_oos_start_unknown_cutoff_returns_unchanged():
    df = _synth_prices(n=50)
    trimmed = enforce_oos_start(df, None)
    assert len(trimmed) == len(df)


def test_enforce_oos_start_strict_raises_when_fully_in_sample():
    df = _synth_prices(n=50)
    train_end = df.index[-1]  # cutoff after every row in df
    with pytest.raises(ValueError, match="in-sample"):
        enforce_oos_start(df, train_end, strict=True)


def test_enforce_oos_start_non_strict_returns_empty_when_fully_in_sample():
    df = _synth_prices(n=50)
    train_end = df.index[-1]
    trimmed = enforce_oos_start(df, train_end, strict=False)
    assert trimmed.empty


# ---------------------------------------------------------------------------
# eval_report.compute_dsr_for_symbol integration
# ---------------------------------------------------------------------------

def test_compute_dsr_for_symbol_trims_pre_cutoff_rows(tmp_path):
    """The DSR backtest loop must only ever see rows after the model's
    train_end + embargo — pre-cutoff (in-sample) rows must never reach
    the backtester, regardless of how favorable their synthetic returns are."""
    from eval_report import compute_dsr_for_symbol
    import eval_report

    df = _synth_prices(n=800)
    train_end_ts = df.index[199]
    model_path = tmp_path / "fake_daily_predictor.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"train_end": train_end_ts.strftime("%Y-%m-%d")}, f)

    seen_lengths = []
    real_backtest_sharpe = eval_report._backtest_sharpe

    def spy(df_arg, q, w):
        seen_lengths.append(len(df_arg))
        return real_backtest_sharpe(df_arg, q, w)

    with patch("eval_report._backtest_sharpe", side_effect=spy):
        compute_dsr_for_symbol(
            "SYNTH", df, quantiles=[0.7], windows=[40], model_path=str(model_path),
        )

    cutoff = train_end_ts + pd.Timedelta(days=FWD_RET_HORIZON_DAYS)
    expected_len = int((df.index > cutoff).sum())

    assert seen_lengths, "expected _backtest_sharpe to be called"
    assert all(n == expected_len for n in seen_lengths)
    assert expected_len < len(df)


def test_compute_dsr_for_symbol_raises_when_all_in_sample(tmp_path):
    from eval_report import compute_dsr_for_symbol

    df = _synth_prices(n=400)
    model_path = tmp_path / "fake_daily_predictor.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"train_end": df.index[-1].strftime("%Y-%m-%d")}, f)

    with pytest.raises(ValueError):
        compute_dsr_for_symbol(
            "SYNTH", df, quantiles=[0.7], windows=[40], model_path=str(model_path),
        )


def test_compute_dsr_for_symbol_no_model_path_is_noop():
    """model_path=None must skip the boundary check entirely (used by callers
    that already know their df is out-of-sample, e.g. unit tests fitting a
    fresh in-session model)."""
    from eval_report import compute_dsr_for_symbol

    df = _synth_prices(n=400)
    out = compute_dsr_for_symbol(
        "SYNTH", df, quantiles=[0.6, 0.7], windows=[40, 60], model_path=None,
    )
    assert "dsr" in out


# ---------------------------------------------------------------------------
# simulate_multi.run_symbol_strategy integration
# ---------------------------------------------------------------------------

def test_run_symbol_strategy_trims_and_rejects_fully_in_sample_range(tmp_path, monkeypatch):
    """run_symbol_strategy must reject a backtest whose entire requested date
    range predates the model's train_end + embargo."""
    import simulate_multi
    from simulation_pipeline import ExecutionConfig, StrategyConfig

    df = _synth_prices(n=60)
    model_path = tmp_path / "daily_logistic.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"train_end": df.index[-1].strftime("%Y-%m-%d")}, f)

    monkeypatch.setitem(
        simulate_multi._STRATEGY_MODEL_PATHS, "daily_logistic", str(model_path)
    )

    cfg = StrategyConfig(name="daily_logistic")
    with pytest.raises(ValueError, match="out-of-sample"):
        simulate_multi.run_symbol_strategy(
            symbol="SYNTH",
            strategy_name="daily_logistic",
            df=df,
            cfg=cfg,
            exec_cfg=ExecutionConfig(),
            run_id="test-run",
            n_mc_runs=0,
        )
