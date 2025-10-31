"""
Strategy profile selection for the FastScalp worker.

Profiles encapsulate target/stop/timeout preferences so we can switch
behaviour numerically based on the observed micro-structure (momentum,
reversal, range mode, volatility bands).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import config
from .signal import SignalFeatures


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    tp_adjust: float = 0.0
    tp_margin_multiplier: float = 1.0
    sl_pips: Optional[float] = None
    drawdown_close_pips: float = config.MAX_DRAWDOWN_CLOSE_PIPS
    timeout_sec: Optional[float] = config.TIMEOUT_SEC_BASE
    timeout_min_gain_pips: float = config.TIMEOUT_MIN_GAIN_PIPS
    low_vol_timeout_override: Optional[float] = None
    high_vol_timeout_override: Optional[float] = None


_BASE_TIMEOUT = config.TIMEOUT_SEC_BASE

DEFAULT_PROFILE = StrategyProfile(
    name="momentum_core",
    tp_adjust=0.0,
    tp_margin_multiplier=1.0,
    drawdown_close_pips=4.2,
    timeout_sec=_BASE_TIMEOUT,
    timeout_min_gain_pips=config.TIMEOUT_MIN_GAIN_PIPS,
    high_vol_timeout_override=_BASE_TIMEOUT * config.TIMEOUT_HIGH_VOL_MULT,
)

REVERSAL_PROFILE = StrategyProfile(
    name="reversal_snapback",
    tp_adjust=-0.15,
    tp_margin_multiplier=0.75,
    drawdown_close_pips=2.6,
    timeout_sec=_BASE_TIMEOUT * 0.8,
    timeout_min_gain_pips=0.25,
    low_vol_timeout_override=_BASE_TIMEOUT,
    high_vol_timeout_override=_BASE_TIMEOUT * max(0.6, config.TIMEOUT_HIGH_VOL_MULT * 0.8),
)

IMPULSE_PROFILE = StrategyProfile(
    name="impulse_follow",
    tp_adjust=0.35,
    tp_margin_multiplier=1.15,
    drawdown_close_pips=5.5,
    timeout_sec=_BASE_TIMEOUT * 0.9,
    timeout_min_gain_pips=0.45,
    high_vol_timeout_override=_BASE_TIMEOUT * config.TIMEOUT_HIGH_VOL_MULT,
)

RANGE_PROFILE = StrategyProfile(
    name="range_reversion",
    tp_adjust=-0.05,
    tp_margin_multiplier=0.65,
    drawdown_close_pips=1.1,
    timeout_sec=_BASE_TIMEOUT * 1.3,
    timeout_min_gain_pips=0.2,
    low_vol_timeout_override=_BASE_TIMEOUT * 1.6,
    high_vol_timeout_override=_BASE_TIMEOUT * max(0.8, config.TIMEOUT_HIGH_VOL_MULT * 0.9),
)

LOW_VOL_PROFILE = StrategyProfile(
    name="low_vol_hold",
    tp_adjust=0.2,
    tp_margin_multiplier=0.9,
    drawdown_close_pips=2.2,
    timeout_sec=_BASE_TIMEOUT * 1.4,
    timeout_min_gain_pips=0.15,
    low_vol_timeout_override=_BASE_TIMEOUT * 1.8,
    high_vol_timeout_override=_BASE_TIMEOUT * max(1.0, config.TIMEOUT_HIGH_VOL_MULT),
)


def select_profile(
    action: str,
    features: SignalFeatures,
    *,
    range_active: bool,
) -> StrategyProfile:
    pattern = (features.pattern_tag or "").lower()
    atr = features.atr_pips or 0.0

    if range_active:
        return RANGE_PROFILE
    if action.startswith("REVERSAL") or pattern.startswith("spike_reversal"):
        return REVERSAL_PROFILE
    if atr <= config.ATR_LOW_VOL_PIPS * 0.95:
        return LOW_VOL_PROFILE
    if pattern in {"impulse_up", "impulse_down"} or (
        abs(features.momentum_pips) >= 1.25 and features.range_pips >= 0.6
    ):
        return IMPULSE_PROFILE
    return DEFAULT_PROFILE


def resolve_timeout(profile: StrategyProfile, atr_pips: Optional[float]) -> Optional[float]:
    base = profile.timeout_sec
    if atr_pips is None:
        return base

    if atr_pips <= config.ATR_LOW_VOL_PIPS:
        if profile.low_vol_timeout_override is not None:
            return profile.low_vol_timeout_override
        if config.TIMEOUT_LOW_VOL_MULT <= 0:
            return None
        if base is None:
            return _BASE_TIMEOUT * config.TIMEOUT_LOW_VOL_MULT
        return max(base, _BASE_TIMEOUT * config.TIMEOUT_LOW_VOL_MULT)

    if atr_pips >= config.ATR_HIGH_VOL_PIPS:
        if profile.high_vol_timeout_override is not None:
            return profile.high_vol_timeout_override
        if base is None:
            return _BASE_TIMEOUT * config.TIMEOUT_HIGH_VOL_MULT
        return min(base, _BASE_TIMEOUT * config.TIMEOUT_HIGH_VOL_MULT)

    return base


PROFILE_BY_NAME = {
    profile.name: profile
    for profile in (
        DEFAULT_PROFILE,
        REVERSAL_PROFILE,
        IMPULSE_PROFILE,
        RANGE_PROFILE,
        LOW_VOL_PROFILE,
    )
}


def get_profile(name: str) -> StrategyProfile:
    return PROFILE_BY_NAME.get(name, DEFAULT_PROFILE)
