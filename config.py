"""
config.py — Single source of truth for all configuration.

All modules import from here. No hardcoded values elsewhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExecutionCfg:
    start_cash: float = 100_000.0
    commission_per_share: float = 0.00005
    slippage_bps: float = 1.0
    max_position_pct: float = 0.05
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    daily_loss_limit_pct: float = 0.05
    holding_period_days: int = 5


@dataclass
class LogisticCfg:
    confidence_threshold: float = 0.55
    pos_threshold: float = 0.002
    neg_threshold: float = -0.002
    C: float = 1.0
    max_iter: int = 1000


@dataclass
class XGBoostCfg:
    confidence_threshold: float = 0.55
    pos_threshold: float = 0.002
    neg_threshold: float = -0.002
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    min_child_weight: int = 2
    gamma: float = 1.0


@dataclass
class DQNCfg:
    window: int = 20
    hidden: int = 256
    gamma: float = 0.99
    lr: float = 0.0005
    batch_size: int = 64
    buffer_size: int = 500_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000
    target_update_interval: int = 5000
    episodes: int = 30
    steps_per_episode: int = 500
    q_advantage_threshold: float = 1.0
    confidence_threshold: float = 2.0
    transaction_cost_bps: float = 5.0


@dataclass
class StrategiesCfg:
    logistic: LogisticCfg = field(default_factory=LogisticCfg)
    xgboost: XGBoostCfg = field(default_factory=XGBoostCfg)
    dqn: DQNCfg = field(default_factory=DQNCfg)


@dataclass
class DataCfg:
    source: str = "yfinance"
    interval: str = "1d"
    history_days: int = 1000
    cache_dir: str = "data/cache"
    database_path: str = "data/trading_sim.db"


@dataclass
class PathsCfg:
    models_dir: str = "models"
    results_dir: str = "results"


@dataclass
class DiscordCfg:
    enabled: bool = True
    username: str = "Trading Sim"
    webhook_url: str = field(default_factory=lambda: os.environ.get("DISCORD_WEBHOOK_URL", ""))


@dataclass
class PredictionCfg:
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "SPY", "MSFT", "GOOGL", "NVDA"])


@dataclass
class AppConfig:
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"])
    prediction: PredictionCfg = field(default_factory=PredictionCfg)
    data: DataCfg = field(default_factory=DataCfg)
    execution: ExecutionCfg = field(default_factory=ExecutionCfg)
    strategies: StrategiesCfg = field(default_factory=StrategiesCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    discord: DiscordCfg = field(default_factory=DiscordCfg)


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str = "config/default.yaml") -> AppConfig:
    """Load AppConfig from YAML, falling back to defaults if file not found."""
    raw: dict[str, Any] = {}
    if Path(path).exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    data_raw = raw.get("data", {})
    exec_raw = raw.get("execution", {})
    strat_raw = raw.get("strategies", {})
    paths_raw = raw.get("paths", {})
    discord_raw = raw.get("discord", {})
    prediction_raw = raw.get("prediction", {})

    return AppConfig(
        symbols=raw.get("symbols", AppConfig.__dataclass_fields__["symbols"].default_factory()),
        prediction=PredictionCfg(
            symbols=prediction_raw.get(
                "symbols",
                PredictionCfg.__dataclass_fields__["symbols"].default_factory(),
            )
        ),
        data=DataCfg(**{k: v for k, v in data_raw.items() if k in DataCfg.__dataclass_fields__}),
        execution=ExecutionCfg(**{k: v for k, v in exec_raw.items() if k in ExecutionCfg.__dataclass_fields__}),
        strategies=StrategiesCfg(
            logistic=LogisticCfg(**{k: v for k, v in strat_raw.get("logistic", {}).items() if k in LogisticCfg.__dataclass_fields__}),
            xgboost=XGBoostCfg(**{k: v for k, v in strat_raw.get("xgboost", {}).items() if k in XGBoostCfg.__dataclass_fields__}),
            dqn=DQNCfg(**{k: v for k, v in strat_raw.get("dqn", {}).items() if k in DQNCfg.__dataclass_fields__}),
        ),
        paths=PathsCfg(**{k: v for k, v in paths_raw.items() if k in PathsCfg.__dataclass_fields__}),
        discord=DiscordCfg(
            enabled=discord_raw.get("enabled", True),
            username=discord_raw.get("username", "Trading Sim"),
            webhook_url=os.environ.get("DISCORD_WEBHOOK_URL", ""),
        ),
    )


# Module-level singleton — import this in other modules
_config: AppConfig | None = None


def get_config(path: str = "config/default.yaml") -> AppConfig:
    """Return cached AppConfig, loading from YAML on first call."""
    global _config
    if _config is None:
        _config = load_config(path)
    return _config
