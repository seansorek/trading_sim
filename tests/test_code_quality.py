"""Code quality checks: annotation presence and accuracy."""
import sys
from pathlib import Path
import typing

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_save_to_csv_has_return_annotation():
    from data_loader import save_to_csv
    assert "return" in save_to_csv.__annotations__, (
        "save_to_csv must have a -> None return annotation"
    )


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
