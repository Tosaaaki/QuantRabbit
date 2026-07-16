"""One finite market-state vocabulary shared by every chart timeframe.

This module normalizes the existing regime, indicator-family, structure, and
liquidity observations.  It does not create a new signal and never grants live
permission; it makes each observed state choose a playbook instead of becoming
ambiguous ``TRANSITION``/``UNCLEAR`` prose.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from quant_rabbit.analysis.regime import (
    ADX_RANGE_THRESHOLD,
    ADX_TREND_THRESHOLD,
    CHOP_RANGE_THRESHOLD,
    CHOP_TREND_THRESHOLD,
)


PHASES = ("PRE_RANGE", "RANGE", "PRE_TREND", "TREND")
STRUCTURE_ACTIVE_LOOKBACK_BARS = 12
SWEEP_ACTIVE_LOOKBACK_BARS = 6
TAXONOMY_SECTIONS: dict[str, tuple[str, ...]] = {
    "phase": PHASES,
    "direction": ("UP", "DOWN", "EITHER"),
    "direction_quality": ("NEUTRAL", "LEAN", "ALIGNED", "CONFLICT"),
    "trend_strength": ("UNKNOWN", "FLAT", "DEVELOPING", "STRONG", "EXTREME"),
    "volatility": ("COMPRESSION", "NORMAL", "EXPANSION", "EXHAUSTION"),
    "momentum": ("ACCELERATING", "STEADY", "DECELERATING", "DIVERGENT"),
    "noise": ("UNKNOWN", "ORDERLY", "MIXED", "CHOPPY"),
    "structure": (
        "INTACT",
        "LIQUIDITY_TEST_ACTIVE",
        "BREAKOUT_ACTIVE",
        "REVERSAL_ACTIVE",
        "LIQUIDITY_TEST_STALE",
        "BREAKOUT_STALE",
        "REVERSAL_STALE",
    ),
    "trigger": ("NONE", "WICK_TEST", "BREAKOUT_CLOSE", "REVERSAL_CLOSE", "LIQUIDITY_SWEEP"),
    "location": ("UNBOUNDED", "LOWER_THIRD", "MIDDLE_THIRD", "UPPER_THIRD"),
    "value_zone": (
        "UNDEFINED",
        "DEEP_DISCOUNT",
        "DISCOUNT",
        "FAIR_VALUE",
        "PREMIUM",
        "DEEP_PREMIUM",
    ),
    "extension": ("OVERSOLD", "STRETCHED_DOWN", "BALANCED", "STRETCHED_UP", "OVERBOUGHT"),
    "mean_reversion_speed": ("UNKNOWN", "FAST", "NORMAL", "SLOW", "UNSTABLE"),
    "liquidity": ("CLEAR", "POOL_PRESENT", "SWEEP_ACTIVE", "SWEEP_STALE"),
    "trend_maturity": ("FORMING", "CONFIRMED", "MATURE", "EXHAUSTING", "DECAYING", "RANGE_MATURE"),
    "readiness": ("FORMING", "ARMED", "TRIGGERED", "ACTIVE"),
    "strategy_family": ("BREAKOUT", "TREND", "REVERSAL", "MEAN_REVERSION"),
    "entry_mode": (
        "STOP_ENTRY_AFTER_CONFIRMATION",
        "PULLBACK_OR_CONTINUATION",
        "FADE_AFTER_REVERSAL_CONFIRMATION",
        "LIMIT_AT_RANGE_EDGE",
    ),
    "invalidation_phase": PHASES,
}


@dataclass(frozen=True)
class MarketStateReading:
    phase: str
    direction: str
    direction_quality: str
    trend_strength: str
    volatility: str
    momentum: str
    noise: str
    structure: str
    trigger: str
    location: str
    value_zone: str
    extension: str
    mean_reversion_speed: str
    liquidity: str
    trend_maturity: str
    readiness: str
    strategy_family: str
    entry_mode: str
    invalidation_phase: str
    confidence: float
    evidence_complete: bool
    basis: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["basis"] = list(self.basis)
        return value


def classify_market_state(
    *,
    indicators: Any,
    regime_reading: Any = None,
    family_scores: Any = None,
    structure: Any = None,
    smc: Any = None,
    legacy_regime: str | None = None,
) -> MarketStateReading:
    """Normalize one timeframe's already-computed technical observations."""

    statistical_state = str(
        getattr(regime_reading, "state", "") or "UNKNOWN"
    ).upper()
    legacy = str(legacy_regime or "").upper()
    trend_score = _finite(getattr(family_scores, "trend_score", None))
    mean_rev_score = _finite(getattr(family_scores, "mean_rev_score", None))
    breakout_score = _finite(getattr(family_scores, "breakout_score", None))
    direction = _direction(
        trend_score=trend_score,
        indicators=indicators,
        legacy_regime=legacy,
    )
    direction_quality = _direction_quality(
        trend_score=trend_score,
        indicators=indicators,
    )
    trend_strength = _trend_strength(indicators)
    momentum = _momentum(indicators)
    noise = _noise(indicators)
    volatility, volatility_complete = _volatility(
        indicators,
        regime_reading=regime_reading,
        momentum=momentum,
    )
    phase, phase_basis = _phase(
        statistical_state=statistical_state,
        legacy_regime=legacy,
        trend_score=trend_score,
        mean_rev_score=mean_rev_score,
        breakout_score=breakout_score,
        volatility=volatility,
    )
    structure_state = _structure(
        structure,
        candles_count=_candles_count(indicators),
    )
    liquidity = _liquidity(
        structure=structure,
        smc=smc,
        candles_count=_candles_count(indicators),
    )
    trigger = _trigger(structure=structure_state, liquidity=liquidity)
    location = _location(indicators)
    value_zone = _value_zone(indicators=indicators, smc=smc)
    extension = _extension(indicators)
    mean_reversion_speed = _mean_reversion_speed(indicators)
    trend_maturity = _trend_maturity(
        phase=phase,
        momentum=momentum,
        volatility=volatility,
        extension=extension,
        trigger=trigger,
    )
    readiness = _readiness(
        phase=phase,
        structure=structure_state,
        location=location,
        liquidity=liquidity,
    )
    strategy_family, entry_mode, invalidation_phase = _playbook(phase)
    regime_confidence = _finite(getattr(regime_reading, "confidence", None))
    confidence = min(
        1.0,
        max(0.0, regime_confidence if regime_confidence is not None else 0.0),
    )
    evidence_complete = bool(
        getattr(indicators, "valid", False)
        and statistical_state != "UNKNOWN"
        and trend_score is not None
        and mean_rev_score is not None
        and breakout_score is not None
        and volatility_complete
    )
    if not evidence_complete:
        # Keep routing possible but disclose incomplete evidence. UNKNOWN is
        # reserved for the input, not overloaded as a fifth market phase.
        confidence = min(confidence, 0.25)
    reading = MarketStateReading(
        phase=phase,
        direction=direction,
        direction_quality=direction_quality,
        trend_strength=trend_strength,
        volatility=volatility,
        momentum=momentum,
        noise=noise,
        structure=structure_state,
        trigger=trigger,
        location=location,
        value_zone=value_zone,
        extension=extension,
        mean_reversion_speed=mean_reversion_speed,
        liquidity=liquidity,
        trend_maturity=trend_maturity,
        readiness=readiness,
        strategy_family=strategy_family,
        entry_mode=entry_mode,
        invalidation_phase=invalidation_phase,
        confidence=round(confidence, 6),
        evidence_complete=evidence_complete,
        basis=(
            f"REGIME={statistical_state}",
            f"LEGACY={legacy or 'NONE'}",
            phase_basis,
            f"VOLATILITY={volatility}",
            f"MOMENTUM={momentum}",
            f"NOISE={noise}",
            f"STRUCTURE={structure_state}",
            f"TRIGGER={trigger}",
            f"LIQUIDITY={liquidity}",
        ),
    )
    _validate_reading(reading)
    return reading


def market_state_taxonomy_contract() -> dict[str, Any]:
    """Return the single machine-readable vocabulary and seed thresholds.

    Thresholds describe classification only.  They are replay dimensions and
    never order permission or a profitability assertion.
    """

    return {
        "contract": "QR_MARKET_STATE_TAXONOMY_V2",
        "sections": {key: list(values) for key, values in TAXONOMY_SECTIONS.items()},
        "thresholds": {
            "structure_active_lookback_bars": STRUCTURE_ACTIVE_LOOKBACK_BARS,
            "sweep_active_lookback_bars": SWEEP_ACTIVE_LOOKBACK_BARS,
            "atr_percentile": {"compression_below": 20.0, "expansion_above": 80.0},
            "adx": {
                "flat_below": ADX_RANGE_THRESHOLD,
                "strong_from": ADX_TREND_THRESHOLD,
                "extreme_from": 40.0,
            },
            "choppiness": {
                "orderly_below": CHOP_TREND_THRESHOLD,
                "choppy_above": CHOP_RANGE_THRESHOLD,
            },
            "value_zone_percentile_edges": [0.20, 0.45, 0.55, 0.80],
            "extension": {
                "rsi_edges": [30.0, 40.0, 60.0, 70.0],
                "z_score_edges": [-2.0, -1.0, 1.0, 2.0],
            },
            "mean_reversion_half_life_bar_edges": [5.0, 20.0, 60.0],
        },
        "threshold_role": "REPLAY_SEED_NOT_LIVE_PERMISSION",
        "grants_live_permission": False,
    }


def summarize_market_states(views: list[Any]) -> dict[str, Any]:
    """Build pair-level phase/direction alignment without hiding disagreement."""

    by_timeframe: dict[str, dict[str, Any]] = {}
    for view in views:
        timeframe = str(getattr(view, "granularity", "") or "").upper()
        reading = getattr(view, "market_state", None)
        if not timeframe or not isinstance(reading, MarketStateReading):
            continue
        by_timeframe[timeframe] = {
            "phase": reading.phase,
            "direction": reading.direction,
            "direction_quality": reading.direction_quality,
            "trend_strength": reading.trend_strength,
            "volatility": reading.volatility,
            "momentum": reading.momentum,
            "noise": reading.noise,
            "structure": reading.structure,
            "trigger": reading.trigger,
            "location": reading.location,
            "value_zone": reading.value_zone,
            "extension": reading.extension,
            "mean_reversion_speed": reading.mean_reversion_speed,
            "liquidity": reading.liquidity,
            "trend_maturity": reading.trend_maturity,
            "readiness": reading.readiness,
            "strategy_family": reading.strategy_family,
            "entry_mode": reading.entry_mode,
            "invalidation_phase": reading.invalidation_phase,
            "confidence": reading.confidence,
            "evidence_complete": reading.evidence_complete,
        }
    short = [by_timeframe[tf] for tf in ("M1", "M5", "M15") if tf in by_timeframe]
    long = [by_timeframe[tf] for tf in ("H1", "H4", "D") if tf in by_timeframe]
    phase_alignment = _phase_alignment(short=short, long=long)
    direction_alignment = _direction_alignment(list(by_timeframe.values()))
    complete_count = sum(
        row["evidence_complete"] is True for row in by_timeframe.values()
    )
    primary = _primary_state(by_timeframe)
    return {
        "contract": "QR_MARKET_STATE_TAXONOMY_V2",
        "by_timeframe": by_timeframe,
        "phase_alignment": phase_alignment,
        "direction_alignment": direction_alignment,
        "opportunity_stage": _opportunity_stage(
            rows=list(by_timeframe.values()),
            phase_alignment=phase_alignment,
            direction_alignment=direction_alignment,
            complete_count=complete_count,
        ),
        "primary_timeframe": primary[0],
        "primary_phase": primary[1].get("phase") if primary[1] else None,
        "primary_strategy_family": (
            primary[1].get("strategy_family") if primary[1] else None
        ),
        "complete_timeframe_count": complete_count,
        "classified_timeframe_count": len(by_timeframe),
        "grants_live_permission": False,
    }


def _phase(
    *,
    statistical_state: str,
    legacy_regime: str,
    trend_score: float | None,
    mean_rev_score: float | None,
    breakout_score: float | None,
    volatility: str,
) -> tuple[str, str]:
    if statistical_state == "TREND_STRONG":
        return "TREND", "STRONG_TREND_ENSEMBLE"
    if statistical_state == "RANGE":
        return "RANGE", "RANGE_ENSEMBLE"
    if statistical_state == "BREAKOUT_PENDING":
        return "PRE_TREND", "COMPRESSION_BREAKOUT_PENDING"
    if statistical_state == "TREND_WEAK":
        if "FAILURE_RISK" in legacy_regime or _dominates(mean_rev_score, trend_score):
            return "PRE_RANGE", "WEAK_TREND_DECAY"
        return "PRE_TREND", "WEAK_TREND_FORMING"
    if statistical_state == "TRANSITION":
        if "FAILURE_RISK" in legacy_regime or _dominates(
            mean_rev_score, breakout_score, trend_score
        ):
            return "PRE_RANGE", "TRANSITION_TOWARD_RANGE"
        if volatility == "COMPRESSION" or _dominates(breakout_score, mean_rev_score):
            return "PRE_TREND", "TRANSITION_TOWARD_EXPANSION"
        return "PRE_RANGE", "TRANSITION_DECELERATION_DEFAULT"
    if "TREND" in legacy_regime or legacy_regime.startswith("IMPULSE_"):
        return "TREND", "LEGACY_DIRECTIONAL_FALLBACK"
    if "RANGE" in legacy_regime:
        return "RANGE", "LEGACY_RANGE_FALLBACK"
    if volatility == "COMPRESSION":
        return "PRE_TREND", "COMPRESSION_FALLBACK"
    return "PRE_RANGE", "INCOMPLETE_EVIDENCE_CONSERVATIVE_PHASE"


def _direction(*, trend_score: float | None, indicators: Any, legacy_regime: str) -> str:
    if trend_score is not None and trend_score != 0.0:
        return "UP" if trend_score > 0.0 else "DOWN"
    plus_di = _finite(getattr(indicators, "plus_di_14", None))
    minus_di = _finite(getattr(indicators, "minus_di_14", None))
    if plus_di is not None and minus_di is not None and plus_di != minus_di:
        return "UP" if plus_di > minus_di else "DOWN"
    if legacy_regime.endswith("_UP"):
        return "UP"
    if legacy_regime.endswith("_DOWN"):
        return "DOWN"
    return "EITHER"


def _direction_quality(*, trend_score: float | None, indicators: Any) -> str:
    votes: list[str] = []
    if trend_score is not None and trend_score != 0.0:
        votes.append("UP" if trend_score > 0.0 else "DOWN")
    plus_di = _finite(getattr(indicators, "plus_di_14", None))
    minus_di = _finite(getattr(indicators, "minus_di_14", None))
    if plus_di is not None and minus_di is not None and plus_di != minus_di:
        votes.append("UP" if plus_di > minus_di else "DOWN")
    for attribute in ("supertrend_dir", "psar_dir", "ichimoku_cloud_pos"):
        value = _finite(getattr(indicators, attribute, None))
        if value is not None and value != 0.0:
            votes.append("UP" if value > 0.0 else "DOWN")
    if not votes:
        return "NEUTRAL"
    up = votes.count("UP")
    down = votes.count("DOWN")
    if up and down:
        return "CONFLICT"
    if len(votes) >= 3:
        return "ALIGNED"
    return "LEAN"


def _trend_strength(indicators: Any) -> str:
    adx = _finite(getattr(indicators, "adx_14", None))
    if adx is None:
        return "UNKNOWN"
    if adx < ADX_RANGE_THRESHOLD:
        return "FLAT"
    if adx < ADX_TREND_THRESHOLD:
        return "DEVELOPING"
    if adx < 40.0:
        return "STRONG"
    return "EXTREME"


def _noise(indicators: Any) -> str:
    choppiness = _finite(getattr(indicators, "choppiness_14", None))
    if choppiness is None:
        return "UNKNOWN"
    if choppiness < CHOP_TREND_THRESHOLD:
        return "ORDERLY"
    if choppiness > CHOP_RANGE_THRESHOLD:
        return "CHOPPY"
    return "MIXED"


def _volatility(indicators: Any, *, regime_reading: Any, momentum: str) -> tuple[str, bool]:
    raw = _finite(getattr(regime_reading, "atr_percentile", None))
    if raw is None:
        raw = _finite(getattr(indicators, "atr_percentile_100", None))
        if raw is not None and 0.0 <= raw <= 1.0:
            raw *= 100.0
    if raw is None:
        if getattr(indicators, "bb_squeeze", None) == 1:
            return "COMPRESSION", False
        return "NORMAL", False
    # 20 is the existing breakout-compression percentile. 80 is its exact
    # complementary high-volatility partition, not a new fitted threshold.
    if raw < 20.0:
        return "COMPRESSION", True
    if raw > 80.0:
        return ("EXHAUSTION" if momentum == "DECELERATING" else "EXPANSION"), True
    return "NORMAL", True


def _momentum(indicators: Any) -> str:
    fast = _finite(getattr(indicators, "ema_slope_5", None))
    slow = _finite(getattr(indicators, "ema_slope_20", None))
    if fast is None or slow is None:
        return "STEADY"
    if fast * slow < 0.0:
        return "DIVERGENT"
    if abs(fast) > abs(slow):
        return "ACCELERATING"
    if abs(fast) < abs(slow):
        return "DECELERATING"
    return "STEADY"


def _structure(structure: Any, *, candles_count: int | None) -> str:
    event = getattr(structure, "last_event", None)
    if event is None:
        return "INTACT"
    active = _recent_index(
        getattr(event, "index", None),
        candles_count=candles_count,
        lookback_bars=STRUCTURE_ACTIVE_LOOKBACK_BARS,
    )
    if getattr(event, "close_confirmed", True) is not True:
        return "LIQUIDITY_TEST_ACTIVE" if active else "LIQUIDITY_TEST_STALE"
    kind = str(getattr(event, "kind", "") or "").upper()
    if kind.startswith("CHOCH_"):
        return "REVERSAL_ACTIVE" if active else "REVERSAL_STALE"
    if kind.startswith("BOS_"):
        return "BREAKOUT_ACTIVE" if active else "BREAKOUT_STALE"
    return "INTACT"


def _location(indicators: Any) -> str:
    close = _finite(getattr(indicators, "close", None))
    low = _finite(getattr(indicators, "donchian_low", None))
    high = _finite(getattr(indicators, "donchian_high", None))
    if close is None or low is None or high is None or high <= low:
        return "UNBOUNDED"
    percentile = (close - low) / (high - low)
    if percentile < 1.0 / 3.0:
        return "LOWER_THIRD"
    if percentile > 2.0 / 3.0:
        return "UPPER_THIRD"
    return "MIDDLE_THIRD"


def _value_zone(*, indicators: Any, smc: Any) -> str:
    close = _finite(getattr(indicators, "close", None))
    dealing_range = getattr(smc, "dealing_range", None)
    high_pivot = getattr(dealing_range, "swing_high", None)
    low_pivot = getattr(dealing_range, "swing_low", None)
    high = _finite(getattr(high_pivot, "price", None))
    low = _finite(getattr(low_pivot, "price", None))
    if close is None or high is None or low is None or high <= low:
        return "UNDEFINED"
    percentile = (close - low) / (high - low)
    if percentile < 0.20:
        return "DEEP_DISCOUNT"
    if percentile < 0.45:
        return "DISCOUNT"
    if percentile <= 0.55:
        return "FAIR_VALUE"
    if percentile <= 0.80:
        return "PREMIUM"
    return "DEEP_PREMIUM"


def _extension(indicators: Any) -> str:
    rsi = _finite(getattr(indicators, "rsi_14", None))
    z_score = _finite(getattr(indicators, "z_score_20", None))
    if (z_score is not None and z_score >= 2.0) or (rsi is not None and rsi >= 70.0):
        return "OVERBOUGHT"
    if (z_score is not None and z_score <= -2.0) or (rsi is not None and rsi <= 30.0):
        return "OVERSOLD"
    if (z_score is not None and z_score >= 1.0) or (rsi is not None and rsi >= 60.0):
        return "STRETCHED_UP"
    if (z_score is not None and z_score <= -1.0) or (rsi is not None and rsi <= 40.0):
        return "STRETCHED_DOWN"
    return "BALANCED"


def _mean_reversion_speed(indicators: Any) -> str:
    half_life = _finite(getattr(indicators, "half_life_60", None))
    if half_life is None:
        return "UNKNOWN"
    if half_life <= 0.0 or half_life > 60.0:
        return "UNSTABLE"
    if half_life <= 5.0:
        return "FAST"
    if half_life <= 20.0:
        return "NORMAL"
    return "SLOW"


def _liquidity(*, structure: Any, smc: Any, candles_count: int | None) -> str:
    sweeps = tuple(getattr(smc, "sweeps", ()) or ())
    if any(
        _recent_index(
            getattr(sweep, "index", None),
            candles_count=candles_count,
            lookback_bars=SWEEP_ACTIVE_LOOKBACK_BARS,
        )
        for sweep in sweeps
    ):
        return "SWEEP_ACTIVE"
    if tuple(getattr(structure, "liquidity", ()) or ()) or tuple(
        getattr(getattr(smc, "structure", None), "liquidity_clusters", ()) or ()
    ):
        return "POOL_PRESENT"
    if sweeps:
        return "SWEEP_STALE"
    return "CLEAR"


def _trigger(*, structure: str, liquidity: str) -> str:
    if liquidity == "SWEEP_ACTIVE":
        return "LIQUIDITY_SWEEP"
    if structure == "REVERSAL_ACTIVE":
        return "REVERSAL_CLOSE"
    if structure == "BREAKOUT_ACTIVE":
        return "BREAKOUT_CLOSE"
    if structure == "LIQUIDITY_TEST_ACTIVE":
        return "WICK_TEST"
    return "NONE"


def _trend_maturity(
    *,
    phase: str,
    momentum: str,
    volatility: str,
    extension: str,
    trigger: str,
) -> str:
    if phase == "PRE_TREND":
        return "FORMING"
    if phase == "PRE_RANGE":
        return "DECAYING"
    if phase == "RANGE":
        return "RANGE_MATURE"
    if (
        momentum in {"DECELERATING", "DIVERGENT"}
        or volatility == "EXHAUSTION"
        or extension in {"OVERBOUGHT", "OVERSOLD"}
    ):
        return "EXHAUSTING"
    if trigger == "BREAKOUT_CLOSE" and momentum == "ACCELERATING":
        return "CONFIRMED"
    return "MATURE"


def _readiness(*, phase: str, structure: str, location: str, liquidity: str) -> str:
    if phase == "TREND":
        return "ACTIVE"
    if structure in {"BREAKOUT_ACTIVE", "REVERSAL_ACTIVE"}:
        return "TRIGGERED"
    if phase == "PRE_TREND" or (phase == "RANGE" and location != "MIDDLE_THIRD"):
        return "ARMED"
    if liquidity == "SWEEP_ACTIVE":
        return "ARMED"
    return "FORMING"


def _playbook(phase: str) -> tuple[str, str, str]:
    return {
        "PRE_TREND": ("BREAKOUT", "STOP_ENTRY_AFTER_CONFIRMATION", "RANGE"),
        "TREND": ("TREND", "PULLBACK_OR_CONTINUATION", "PRE_RANGE"),
        "PRE_RANGE": ("REVERSAL", "FADE_AFTER_REVERSAL_CONFIRMATION", "TREND"),
        "RANGE": ("MEAN_REVERSION", "LIMIT_AT_RANGE_EDGE", "PRE_TREND"),
    }[phase]


def _phase_alignment(*, short: list[dict[str, Any]], long: list[dict[str, Any]]) -> str:
    phases = [row["phase"] for row in short + long]
    if not phases:
        return "NO_DATA"
    if len(set(phases)) == 1:
        return "ALIGNED"
    short_phases = {row["phase"] for row in short}
    long_phases = {row["phase"] for row in long}
    if short_phases & {"PRE_TREND", "TREND"} and long_phases & {"PRE_RANGE", "RANGE"}:
        return "LOWER_TF_LEADING"
    if short_phases & {"PRE_RANGE", "RANGE"} and long_phases & {"PRE_TREND", "TREND"}:
        return "LOWER_TF_DECELERATING"
    return "MIXED"


def _direction_alignment(rows: list[dict[str, Any]]) -> str:
    directions = {row["direction"] for row in rows if row["direction"] != "EITHER"}
    if not directions:
        return "NEUTRAL"
    if len(directions) == 1:
        return "ALIGNED"
    return "CONFLICT"


def _primary_state(
    by_timeframe: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    for timeframe in ("M15", "H1", "M5", "M30", "H4", "M1", "D"):
        if timeframe in by_timeframe:
            return timeframe, by_timeframe[timeframe]
    return None, None


def _opportunity_stage(
    *,
    rows: list[dict[str, Any]],
    phase_alignment: str,
    direction_alignment: str,
    complete_count: int,
) -> str:
    if not rows or complete_count == 0:
        return "EVIDENCE_INCOMPLETE"
    if direction_alignment == "CONFLICT" or phase_alignment == "MIXED":
        return "CROSS_TF_CONFLICT"
    if any(row.get("trigger") != "NONE" for row in rows):
        return "TRIGGER_PRESENT"
    if any(row.get("readiness") in {"ARMED", "ACTIVE"} for row in rows):
        return "SETUP_PRESENT"
    return "FORMING"


def _candles_count(indicators: Any) -> int | None:
    value = getattr(indicators, "candles_count", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _recent_index(
    value: Any,
    *,
    candles_count: int | None,
    lookback_bars: int,
) -> bool:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or candles_count is None
        or value < 0
        or value >= candles_count
    ):
        return False
    return (candles_count - 1 - value) < lookback_bars


def _validate_reading(reading: MarketStateReading) -> None:
    for field_name, allowed in TAXONOMY_SECTIONS.items():
        value = getattr(reading, field_name)
        if value not in allowed:
            raise ValueError(
                f"market-state field {field_name}={value!r} is outside the V2 taxonomy"
            )


def _dominates(value: float | None, *others: float | None) -> bool:
    if value is None:
        return False
    other_values = [abs(item) for item in others if item is not None]
    return bool(other_values and abs(value) > max(other_values))


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


__all__ = [
    "PHASES",
    "STRUCTURE_ACTIVE_LOOKBACK_BARS",
    "SWEEP_ACTIVE_LOOKBACK_BARS",
    "TAXONOMY_SECTIONS",
    "MarketStateReading",
    "classify_market_state",
    "market_state_taxonomy_contract",
    "summarize_market_states",
]
