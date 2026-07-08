"""Code quality checks: annotation presence and accuracy."""
import sys
from pathlib import Path
import typing

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_xgboost_return_is_parameterized():
    """train_xgboost should return tuple[Any, Any, float, float, float], not bare tuple."""
    from train_models import train_xgboost
    ret = train_xgboost.__annotations__.get("return")
    assert ret is not None, "train_xgboost must have a return annotation"
    assert hasattr(ret, "__args__"), (
        f"Expected parameterized tuple (e.g. tuple[Any, Any, float, float, float]), got {ret}"
    )


def test_load_models_return_annotation():
    from predict_next_day_lite import load_models
    ret = load_models.__annotations__.get("return")
    assert ret is not None, "load_models must have a return annotation"


def test_predict_symbol_spy_df_annotated():
    from predict_next_day_lite import predict_symbol
    assert "spy_df" in predict_symbol.__annotations__, (
        "predict_symbol must annotate its spy_df parameter"
    )


def test_predict_symbol_return_annotation():
    from predict_next_day_lite import predict_symbol
    ret = predict_symbol.__annotations__.get("return")
    assert ret is not None, "predict_symbol must have a return annotation"


# ---------------------------------------------------------------------------
# Task 3 — Interface consistency
# ---------------------------------------------------------------------------

def test_base_strategy_signal_return_annotated():
    import pandas as pd
    from base_strategy import BaseStrategy
    hints = typing.get_type_hints(BaseStrategy.signal)
    assert hints.get("return") is pd.Series, (
        f"BaseStrategy.signal must be annotated '-> pd.Series', got {hints.get('return')}"
    )


def test_backtest_result_metrics_allows_none():
    """BacktestResult.metrics annotation must reflect that profit_factor can be None."""
    import dataclasses
    from simulation_pipeline import BacktestResult
    field_types = {f.name: f.type for f in dataclasses.fields(BacktestResult)}
    metrics_type = str(field_types.get("metrics", ""))
    assert "Any" in metrics_type, (
        f"BacktestResult.metrics must be Dict[str, Any] (profit_factor is Optional), "
        f"but annotation is: {metrics_type}"
    )


def test_compute_metrics_profit_factor_none_when_no_losses():
    """compute_metrics must return None for profit_factor when gross_loss == 0."""
    import pandas as pd
    import numpy as np
    from simulation_pipeline import compute_metrics

    equity = pd.Series(
        [100_000.0, 100_100.0, 100_200.0],
        index=pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC"),
    )
    result = compute_metrics(equity, pd.DataFrame())
    assert result.get("profit_factor") is None, (
        "profit_factor should be None when there are no realized losses"
    )
