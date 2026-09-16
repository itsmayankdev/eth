from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PatternConfig:
    lookbacks: list[int] = field(default_factory=lambda: [3, 5, 8, 10, 15, 20, 30, 50])
    minimum_similarity: float = 0.85
    top_k: int = 10


@dataclass(frozen=True)
class TurningPointConfig:
    method: str = "zigzag"
    thresholds_pct: list[float] = field(default_factory=lambda: [0.10, 0.25, 0.50, 1.0, 2.0, 5.0])


@dataclass(frozen=True)
class OutcomeConfig:
    horizons: list[int] = field(default_factory=lambda: [5, 10, 20, 50])
    target_pct: float = 1.0
    stop_pct: float = 0.5


@dataclass(frozen=True)
class StatisticsConfig:
    confidence_level: float = 0.95


@dataclass(frozen=True)
class LiveConfig:
    websocket_base: str = "wss://stream.binance.com:9443/ws"
    reconnect_seconds: int = 5


@dataclass(frozen=True)
class Config:
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    historical_days: int = 365
    database: str = "data/eth_market.db"
    timeframes: list[str] = field(default_factory=lambda: ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"])
    pattern: PatternConfig = field(default_factory=PatternConfig)
    turning_points: TurningPointConfig = field(default_factory=TurningPointConfig)
    outcomes: OutcomeConfig = field(default_factory=OutcomeConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    live: LiveConfig = field(default_factory=LiveConfig)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def load_config(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config(
        symbol=str(raw.get("symbol", "ETHUSDT")),
        exchange=str(raw.get("exchange", "binance")),
        historical_days=int(raw.get("historical_days", 365)),
        database=str(raw.get("database", "data/eth_market.db")),
        timeframes=list(raw.get("timeframes", Config().timeframes)),
        pattern=PatternConfig(**_section(raw, "pattern")),
        turning_points=TurningPointConfig(**_section(raw, "turning_points")),
        outcomes=OutcomeConfig(**_section(raw, "outcomes")),
        statistics=StatisticsConfig(**_section(raw, "statistics")),
        live=LiveConfig(**_section(raw, "live")),
    )
