"""test_dqn_strategy.py — Tests for DailyDQNStrategy config fallback (Issue #33).

The root conftest.py installs a torch mock if torch is unavailable, so
simulation_pipeline can be imported without a real torch installation.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_strategy import StrategyConfig
from simulation_pipeline import DailyDQNStrategy


class TestDailyDQNStrategyThresholds:
    """Verify DailyDQNStrategy reads thresholds from config, not hardcoded defaults."""

    def _make_cfg(self, **overrides) -> StrategyConfig:
        """Build a StrategyConfig with DQN-specific attributes attached."""
        cfg = StrategyConfig(name="daily_dqn")
        # Attach DQN-specific attrs that getattr(cfg, ...) should find
        cfg.confidence_threshold = overrides.get("confidence_threshold", 2.0)
        cfg.q_advantage_threshold = overrides.get("q_advantage_threshold", 1.0)
        cfg.model_path = overrides.get("model_path", "models/dqn_agent.pt")
        cfg.window = overrides.get("window", 20)
        return cfg

    def test_uses_config_defaults_when_env_unset(self):
        """When DQN_CONFIDENCE and DQN_Q_ADVANTAGE env vars are absent,
        thresholds should come from the config object (2.0 and 1.0)."""
        # Ensure env vars are not set
        env_backup = {}
        for key in ("DQN_CONFIDENCE", "DQN_Q_ADVANTAGE"):
            env_backup[key] = os.environ.pop(key, None)

        try:
            cfg = self._make_cfg(confidence_threshold=2.0, q_advantage_threshold=1.0)
            strategy = DailyDQNStrategy(cfg)

            assert strategy.confidence_threshold == pytest.approx(2.0), (
                f"Expected confidence_threshold=2.0 from config, got {strategy.confidence_threshold}"
            )
            assert strategy.q_advantage_threshold == pytest.approx(1.0), (
                f"Expected q_advantage_threshold=1.0 from config, got {strategy.q_advantage_threshold}"
            )
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_env_vars_override_config(self):
        """When DQN_CONFIDENCE and DQN_Q_ADVANTAGE are set, they take priority."""
        env_backup = {}
        for key in ("DQN_CONFIDENCE", "DQN_Q_ADVANTAGE"):
            env_backup[key] = os.environ.get(key)

        try:
            os.environ["DQN_CONFIDENCE"] = "5.0"
            os.environ["DQN_Q_ADVANTAGE"] = "3.0"

            cfg = self._make_cfg(confidence_threshold=2.0, q_advantage_threshold=1.0)
            strategy = DailyDQNStrategy(cfg)

            assert strategy.confidence_threshold == pytest.approx(5.0)
            assert strategy.q_advantage_threshold == pytest.approx(3.0)
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                else:
                    os.environ.pop(key, None)

    def test_config_with_custom_thresholds(self):
        """Config with non-default thresholds should be respected."""
        env_backup = {}
        for key in ("DQN_CONFIDENCE", "DQN_Q_ADVANTAGE"):
            env_backup[key] = os.environ.pop(key, None)

        try:
            cfg = self._make_cfg(confidence_threshold=3.5, q_advantage_threshold=2.5)
            strategy = DailyDQNStrategy(cfg)

            assert strategy.confidence_threshold == pytest.approx(3.5)
            assert strategy.q_advantage_threshold == pytest.approx(2.5)
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_fallback_defaults_when_config_lacks_attributes(self):
        """If config has no threshold attributes, fall back to 2.0 and 1.0."""
        env_backup = {}
        for key in ("DQN_CONFIDENCE", "DQN_Q_ADVANTAGE"):
            env_backup[key] = os.environ.pop(key, None)

        try:
            # Plain StrategyConfig without DQN-specific attrs
            cfg = StrategyConfig(name="daily_dqn")
            strategy = DailyDQNStrategy(cfg)

            assert strategy.confidence_threshold == pytest.approx(2.0), (
                "Should fall back to 2.0 when config has no confidence_threshold"
            )
            assert strategy.q_advantage_threshold == pytest.approx(1.0), (
                "Should fall back to 1.0 when config has no q_advantage_threshold"
            )
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
