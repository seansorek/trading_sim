"""test_config.py — Tests for config.py's YAML loading, focused on the
prediction.models field that drives which models predict_next_day_lite.py
attempts to load (see Task 5 in docs/superpowers/plans/2026-06-30-predictor-production-deploy.md).

Also covers the broader load_config()/get_config() contract: default
structure/types, override-merge precedence (unrelated defaults must survive
a partial override), missing/malformed override files, and get_config()'s
singleton caching (issue #75)."""
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as config_module
from config import (
    AppConfig,
    DataCfg,
    ExecutionCfg,
    PredictionCfg,
    get_config,
    load_config,
)


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """get_config() caches into a module-level global; reset it around each
    test so tests don't leak state into each other."""
    config_module._config = None
    yield
    config_module._config = None


def _write_yaml(content: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(dedent(content))
        return f.name


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


def test_load_config_missing_file_returns_defaults():
    """A nonexistent override path must not raise -- load_config() falls
    back to the dataclass defaults for every section."""
    cfg = load_config("does/not/exist.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.symbols == AppConfig.__dataclass_fields__["symbols"].default_factory()
    assert cfg.execution.start_cash == ExecutionCfg().start_cash
    assert cfg.data.source == DataCfg().source


def test_load_config_default_structure_and_types():
    """Default (no override file) config has the expected nested structure
    and types across every top-level section."""
    cfg = load_config("does/not/exist.yaml")
    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg.symbols, list) and all(isinstance(s, str) for s in cfg.symbols)
    assert isinstance(cfg.prediction, PredictionCfg)
    assert isinstance(cfg.data, DataCfg)
    assert isinstance(cfg.execution, ExecutionCfg)
    assert isinstance(cfg.execution.start_cash, float)
    assert isinstance(cfg.execution.holding_period_days, int)
    assert isinstance(cfg.strategies.logistic.confidence_threshold, float)
    assert isinstance(cfg.strategies.xgboost.n_estimators, int)
    assert isinstance(cfg.strategies.dqn.buffer_size, int)
    assert isinstance(cfg.optimize.n_iter, int)
    assert isinstance(cfg.optimize.logistic.penalty, list)
    assert isinstance(cfg.paths.models_dir, str)
    assert isinstance(cfg.discord.enabled, bool)


def test_load_config_partial_override_preserves_unrelated_defaults():
    """Overriding one nested key (execution.start_cash) must not drop
    sibling defaults in the same section (execution.stop_loss_pct) or in
    unrelated sections (strategies.xgboost.n_estimators)."""
    path = _write_yaml("""
        execution:
          start_cash: 250000.0
    """)
    cfg = load_config(path)
    assert cfg.execution.start_cash == 250000.0
    assert cfg.execution.stop_loss_pct == ExecutionCfg().stop_loss_pct
    assert cfg.execution.commission_per_share == ExecutionCfg().commission_per_share
    assert cfg.strategies.xgboost.n_estimators == 200
    assert cfg.data.source == "yfinance"


def test_load_config_unknown_keys_in_override_are_ignored():
    """An override section containing a key that isn't a dataclass field
    must not raise -- load_config() filters to known fields per section."""
    path = _write_yaml("""
        execution:
          start_cash: 5000.0
          not_a_real_field: 123
    """)
    cfg = load_config(path)
    assert cfg.execution.start_cash == 5000.0


def test_load_config_empty_file_returns_defaults():
    """An override file that exists but is empty (or all-comments) must
    yield the same defaults as no file at all -- yaml.safe_load returns
    None for an empty document, and load_config() must handle that."""
    path = _write_yaml("""
        # just a comment, no actual config
    """)
    cfg = load_config(path)
    assert cfg.execution.start_cash == ExecutionCfg().start_cash
    assert cfg.symbols == AppConfig.__dataclass_fields__["symbols"].default_factory()


def test_load_config_malformed_yaml_raises():
    """A syntactically invalid YAML override file propagates a YAMLError
    rather than silently falling back to defaults -- a malformed file
    should be a loud failure, not a silent no-op."""
    path = _write_yaml("""
        execution:
          start_cash: [unterminated
    """)
    with pytest.raises(yaml.YAMLError):
        load_config(path)


def test_get_config_returns_singleton_across_calls():
    """get_config() must cache: repeated calls return the exact same
    AppConfig instance, not a fresh reload each time."""
    first = get_config()
    second = get_config()
    assert first is second


def test_get_config_caches_first_path_ignores_later_path_argument():
    """Once cached, get_config() must not reload even if called again with
    a different path -- the global singleton wins over the argument."""
    path = _write_yaml("""
        execution:
          start_cash: 999.0
    """)
    first = get_config(path)
    assert first.execution.start_cash == 999.0

    second = get_config("does/not/exist.yaml")
    assert second is first
    assert second.execution.start_cash == 999.0
