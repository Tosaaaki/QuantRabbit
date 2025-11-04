"""Configuration values for the H1 trend worker."""

from __future__ import annotations

import os
from typing import FrozenSet


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _float(key: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def _int(key: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def _choices(key: str, default: str) -> FrozenSet[str]:
    raw = os.getenv(key, default)
    tokens = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        tokens.append(token.title())
    return frozenset(tokens)


def _parse_float_tuple(key: str, default: str) -> tuple[float, ...]:
    raw = os.getenv(key, default)
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = float(token)
        except ValueError:
            continue
        if val <= 0.0:
            continue
        values.append(val)
    if not values:
        values = [1.0]
    return tuple(values)


LOG_PREFIX = "[TREND-H1]"
POCKET = "macro"

ENABLED: bool = _bool("TREND_H1_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(5.0, _float("TREND_H1_LOOP_INTERVAL_SEC", 45.0))
MIN_CANDLES: int = max(60, _int("TREND_H1_MIN_CANDLES", 180))
DATA_STALE_SECONDS: float = max(0.0, _float("TREND_H1_DATA_STALE_SECONDS", 180.0))

MIN_CONFIDENCE: int = max(40, _int("TREND_H1_MIN_CONFIDENCE", 55))
CONFIDENCE_FLOOR: int = max(30, _int("TREND_H1_CONFIDENCE_FLOOR", 50))
CONFIDENCE_CEIL: int = max(MIN_CONFIDENCE + 5, _int("TREND_H1_CONFIDENCE_CEIL", 85))
MIN_CONFIDENCE_SCALE: float = max(0.1, _float("TREND_H1_MIN_CONF_SCALE", 0.25))
MAX_CONFIDENCE_SCALE: float = max(
    MIN_CONFIDENCE_SCALE, _float("TREND_H1_MAX_CONF_SCALE", 1.0)
)

MIN_LOT: float = max(0.0, _float("TREND_H1_MIN_LOT", 0.0006))
MAX_LOT: float = max(MIN_LOT, _float("TREND_H1_MAX_LOT", 0.0055))
RISK_PCT: float = max(0.0005, _float("TREND_H1_RISK_PCT", 0.015))
MIN_UNITS: int = max(1000, _int("TREND_H1_MIN_UNITS", 8000))

MAX_ACTIVE_TRADES: int = max(1, _int("TREND_H1_MAX_ACTIVE_TRADES", 2))
MAX_DIRECTIONAL_TRADES: int = max(1, _int("TREND_H1_MAX_DIRECTIONAL_TRADES", 2))
MAX_DIRECTIONAL_UNITS: int = max(MIN_UNITS, _int("TREND_H1_MAX_DIRECTIONAL_UNITS", 120000))
STAGE_RATIOS: tuple[float, ...] = _parse_float_tuple(
    "TREND_H1_STAGE_RATIOS", "0.35,0.25"
)

ENTRY_COOLDOWN_SEC: float = max(0.0, _float("TREND_H1_ENTRY_COOLDOWN_SEC", 180.0))
REENTRY_COOLDOWN_SEC: float = max(0.0, _float("TREND_H1_REENTRY_COOLDOWN_SEC", 300.0))
REPEAT_BLOCK_SEC: float = max(0.0, _float("TREND_H1_REPEAT_BLOCK_SEC", 720.0))

NEWS_BLOCK_MINUTES: float = max(0.0, _float("TREND_H1_NEWS_BLOCK_MINUTES", 25.0))
NEWS_BLOCK_MIN_IMPACT: int = max(1, _int("TREND_H1_NEWS_BLOCK_MIN_IMPACT", 3))

LOSS_STREAK_MAX: int = max(0, _int("TREND_H1_LOSS_STREAK_MAX", 3))
LOSS_STREAK_COOLDOWN_MIN: float = max(
    0.0, _float("TREND_H1_LOSS_STREAK_COOLDOWN_MIN", 150.0)
)

MIN_ATR_PIPS: float = max(0.0, _float("TREND_H1_MIN_ATR_PIPS", 6.0))
MAX_ATR_PIPS: float = max(MIN_ATR_PIPS, _float("TREND_H1_MAX_ATR_PIPS", 110.0))
MIN_FAST_GAP_PIPS: float = max(0.0, _float("TREND_H1_MIN_FAST_GAP_PIPS", 42.0))
H1_OVERRIDE_GAP_PIPS: float = max(0.0, _float("TREND_H1_OVERRIDE_GAP_PIPS", 30.0))
H1_OVERRIDE_ATR_PIPS: float = max(0.0, _float("TREND_H1_OVERRIDE_ATR_PIPS", 32.0))

SPREAD_MAX_PIPS: float = max(0.0, _float("TREND_H1_SPREAD_MAX_PIPS", 2.1))

REQUIRE_REGIME: FrozenSet[str] = _choices(
    "TREND_H1_REQUIRE_REGIME", "Trend,Breakout,Mixed"
)
BLOCK_REGIME: FrozenSet[str] = _choices("TREND_H1_BLOCK_REGIME", "Range")
ALLOWED_DIRECTIONS: FrozenSet[str] = _choices(
    "TREND_H1_ALLOWED_DIRECTIONS", "Long,Short"
)

LOG_SKIP_REASON: bool = _bool("TREND_H1_LOG_SKIP_REASON", True)
