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
class OptimizeLogisticBounds:
    C: list = field(default_factory=lambda: [0.001, 100.0])
    penalty: list = field(default_factory=lambda: ["l1", "l2"])


@dataclass
class OptimizeXGBoostBounds:
    n_estimators: list = field(default_factory=lambda: [100, 600])
    max_depth: list = field(default_factory=lambda: [2, 7])
    learning_rate: list = field(default_factory=lambda: [0.005, 0.3])
    subsample: list = field(default_factory=lambda: [0.5, 1.0])
    colsample_bytree: list = field(default_factory=lambda: [0.5, 1.0])
    gamma: list = field(default_factory=lambda: [0.0, 5.0])
    min_child_weight: list = field(default_factory=lambda: [1, 10])
    reg_alpha: list = field(default_factory=lambda: [0.0, 2.0])
    reg_lambda: list = field(default_factory=lambda: [0.5, 10.0])


@dataclass
class OptimizeCfg:
    n_iter: int = 25
    cv: int = 3
    # Embargo gap (in samples) between each TimeSeriesSplit train/validation
    # fold, to reduce leakage from rolling-window feature autocorrelation.
    cv_gap: int = 0
    logistic: OptimizeLogisticBounds = field(default_factory=OptimizeLogisticBounds)
    xgboost: OptimizeXGBoostBounds = field(default_factory=OptimizeXGBoostBounds)


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
    models: list[str] = field(
        default_factory=lambda: ["daily_logistic", "daily_xgboost", "daily_predictor"]
    )
    max_model_age_days: int = 30


@dataclass
class PanelCfg:
    """Cross-sectional panel backtester (research-only; see docs/superpowers/specs/2026-07-15-step3-panel-portfolio-design.md).

    universe is stocks only — index/sector ETFs are baskets of the same names
    and must not be ranked against their own constituents.
    """

    universe: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    # sector name -> symbols. Flattened into `universe` at load time, so this is
    # the single source of truth for both membership and sector identity. Empty
    # means no sector data, and the panel falls back to un-neutralized ranking.
    sectors: dict[str, list[str]] = field(default_factory=dict)
    decile: float = 0.1
    rebalance_days: int = 1
    gross_exposure: float = 1.0
    cost_bps: float = 5.0
    borrow_bps_annual: float = 50.0
    min_names: int = 20

    def sector_of(self) -> dict[str, str]:
        """symbol -> sector name. Empty when no sectors are configured."""
        return {sym: name for name, syms in self.sectors.items() for sym in syms}


@dataclass
class AppConfig:
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"])
    prediction: PredictionCfg = field(default_factory=PredictionCfg)
    panel: PanelCfg = field(default_factory=PanelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    execution: ExecutionCfg = field(default_factory=ExecutionCfg)
    strategies: StrategiesCfg = field(default_factory=StrategiesCfg)
    optimize: OptimizeCfg = field(default_factory=OptimizeCfg)
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


def _panel_universe(panel_raw: dict) -> list[str]:
    """Flatten panel.sectors into the traded universe.

    `sectors` wins when present so membership and sector identity cannot drift;
    an explicit `universe` list is still honoured for configs that predate
    sectors. A symbol appearing in two sectors is a config error, not something
    to silently de-duplicate — its sector-neutral weight would be ambiguous.
    """
    sectors = panel_raw.get("sectors") or {}
    if not sectors:
        return panel_raw.get(
            "universe", PanelCfg.__dataclass_fields__["universe"].default_factory()
        )
    flat = [sym for syms in sectors.values() for sym in syms]
    dupes = {s for s in flat if flat.count(s) > 1}
    if dupes:
        raise ValueError(f"panel.sectors lists these symbols more than once: {sorted(dupes)}")
    return flat


def load_config(path: str = "config/default.yaml") -> AppConfig:
    """Load AppConfig from YAML, falling back to defaults if file not found."""
    raw: dict[str, Any] = {}
    if Path(path).exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    data_raw = raw.get("data", {})
    exec_raw = raw.get("execution", {})
    strat_raw = raw.get("strategies", {})
    opt_raw = raw.get("optimize", {})
    paths_raw = raw.get("paths", {})
    discord_raw = raw.get("discord", {})
    prediction_raw = raw.get("prediction", {})
    panel_raw = raw.get("panel", {})

    return AppConfig(
        symbols=raw.get("symbols", AppConfig.__dataclass_fields__["symbols"].default_factory()),
        prediction=PredictionCfg(
            symbols=prediction_raw.get(
                "symbols",
                PredictionCfg.__dataclass_fields__["symbols"].default_factory(),
            ),
            models=prediction_raw.get(
                "models",
                PredictionCfg.__dataclass_fields__["models"].default_factory(),
            ),
            max_model_age_days=prediction_raw.get("max_model_age_days", 30),
        ),
        panel=PanelCfg(
            universe=_panel_universe(panel_raw),
            **{
                k: v for k, v in panel_raw.items()
                if k in PanelCfg.__dataclass_fields__ and k != "universe"
            },
        ),
        data=DataCfg(**{k: v for k, v in data_raw.items() if k in DataCfg.__dataclass_fields__}),
        execution=ExecutionCfg(**{k: v for k, v in exec_raw.items() if k in ExecutionCfg.__dataclass_fields__}),
        strategies=StrategiesCfg(
            logistic=LogisticCfg(**{k: v for k, v in strat_raw.get("logistic", {}).items() if k in LogisticCfg.__dataclass_fields__}),
            xgboost=XGBoostCfg(**{k: v for k, v in strat_raw.get("xgboost", {}).items() if k in XGBoostCfg.__dataclass_fields__}),
            dqn=DQNCfg(**{k: v for k, v in strat_raw.get("dqn", {}).items() if k in DQNCfg.__dataclass_fields__}),
        ),
        optimize=OptimizeCfg(
            n_iter=opt_raw.get("n_iter", 25),
            cv=opt_raw.get("cv", 3),
            logistic=OptimizeLogisticBounds(**{k: v for k, v in opt_raw.get("logistic", {}).items() if k in OptimizeLogisticBounds.__dataclass_fields__}),
            xgboost=OptimizeXGBoostBounds(**{k: v for k, v in opt_raw.get("xgboost", {}).items() if k in OptimizeXGBoostBounds.__dataclass_fields__}),
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
