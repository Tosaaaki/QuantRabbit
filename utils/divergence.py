"""Helpers to interpret divergence factors for strategy biasing."""

from __future__ import annotations

import os
from typing import Mapping


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


DIV_BIAS_MAX_AGE_BARS = _env_int("DIV_BIAS_MAX_AGE_BARS", 24)
DIV_RSI_WEIGHT = _env_float("DIV_BIAS_RSI_WEIGHT", 0.65)
DIV_MACD_WEIGHT = _env_float("DIV_BIAS_MACD_WEIGHT", 0.45)
DIV_REVERSION_HIDDEN_WEIGHT = _env_float("DIV_BIAS_REVERSION_HIDDEN_WEIGHT", 0.5)
DIV_TREND_REG_WEIGHT = _env_float("DIV_BIAS_TREND_REG_WEIGHT", -0.7)
DIV_TREND_HIDDEN_WEIGHT = _env_float("DIV_BIAS_TREND_HIDDEN_WEIGHT", 0.7)
DIV_NEUTRAL_REG_WEIGHT = _env_float("DIV_BIAS_NEUTRAL_REG_WEIGHT", 0.7)
DIV_NEUTRAL_HIDDEN_WEIGHT = _env_float("DIV_BIAS_NEUTRAL_HIDDEN_WEIGHT", 0.4)


def _side_dir(side: str) -> int:
    text = (side or "").lower()
    if "short" in text or "sell" in text:
        return -1
    return 1


def _score_from_kind(kind: int, strength: float, mode: str) -> float:
    if kind == 0:
        return 0.0
    sign = 1.0 if kind > 0 else -1.0
    is_hidden = abs(kind) == 2
    mode = (mode or "neutral").lower()
    if mode == "trend":
        weight = DIV_TREND_HIDDEN_WEIGHT if is_hidden else DIV_TREND_REG_WEIGHT
    elif mode == "reversion":
        weight = DIV_REVERSION_HIDDEN_WEIGHT if is_hidden else 1.0
    else:
        weight = DIV_NEUTRAL_HIDDEN_WEIGHT if is_hidden else DIV_NEUTRAL_REG_WEIGHT
    strength = max(0.0, min(1.0, strength))
    return sign * weight * strength


def divergence_bias(
    fac: Mapping[str, object],
    side: str,
    *,
    mode: str = "neutral",
    max_age_bars: int | None = None,
) -> float:
    if not fac:
        return 0.0
    max_age = DIV_BIAS_MAX_AGE_BARS if max_age_bars is None else max(0, max_age_bars)

    def _extract(prefix: str) -> float:
        kind_raw = fac.get(f"div_{prefix}_kind")
        age_raw = fac.get(f"div_{prefix}_age")
        strength_raw = fac.get(f"div_{prefix}_strength")
        try:
            kind = int(round(float(kind_raw))) if kind_raw is not None else 0
        except (TypeError, ValueError):
            kind = 0
        try:
            age = int(round(float(age_raw))) if age_raw is not None else 10**9
        except (TypeError, ValueError):
            age = 10**9
        try:
            strength = float(strength_raw) if strength_raw is not None else 1.0
        except (TypeError, ValueError):
            strength = 1.0
        if age > max_age:
            return 0.0
        return _score_from_kind(kind, strength, mode)

    rsi_score = _extract("rsi") * DIV_RSI_WEIGHT
    macd_score = _extract("macd") * DIV_MACD_WEIGHT
    raw = rsi_score + macd_score
    raw = max(-1.0, min(1.0, raw))
    if _side_dir(side) < 0:
        raw = -raw
    return max(-1.0, min(1.0, raw))


def apply_divergence_confidence(
    confidence: float | int,
    bias: float,
    *,
    max_bonus: float = 8.0,
    max_penalty: float = 10.0,
    floor: float = 0.0,
    ceil: float = 100.0,
) -> int:
    try:
        base = float(confidence)
    except (TypeError, ValueError):
        base = 0.0
    bias = max(-1.0, min(1.0, float(bias)))
    if bias >= 0:
        adj = max_bonus * bias
    else:
        adj = max_penalty * bias
    value = base + adj
    value = max(floor, min(ceil, value))
    return int(round(value))

