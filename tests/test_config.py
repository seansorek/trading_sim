"""test_config.py — Tests for config.py's YAML loading, focused on the
prediction.models field that drives which models predict_next_day_lite.py
attempts to load (see Task 5 in docs/superpowers/plans/2026-06-30-predictor-production-deploy.md)."""
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PredictionCfg, load_config


def test_prediction_cfg_default_models():
    """With no config file, PredictionCfg.models defaults to the three
    currently-deployed daily models."""
    cfg = PredictionCfg()
    assert cfg.models == ["daily_logistic", "daily_xgboost", "daily_predictor"]


def test_load_config_reads_prediction_models_from_yaml():
    yaml_content = dedent("""
        prediction:
          models:
            - daily_logistic
            - daily_predictor
          symbols:
            - AAPL
    """)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        path = f.name

    cfg = load_config(path)
    assert cfg.prediction.models == ["daily_logistic", "daily_predictor"]
    assert cfg.prediction.symbols == ["AAPL"]


def test_load_config_missing_prediction_models_uses_default():
    """A config file that sets prediction.symbols but omits models must
    still fall back to the default model list, not an empty list."""
    yaml_content = dedent("""
        prediction:
          symbols:
            - AAPL
    """)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        path = f.name

    cfg = load_config(path)
    assert cfg.prediction.models == ["daily_logistic", "daily_xgboost", "daily_predictor"]
