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


PHASES = ("PRE_RANGE", "RANGE", "PRE_TREND", "TREND")


@dataclass(frozen=True)
class MarketStateReading:
    phase: str
    direction: str
    volatility: str
    momentum: str
    structure: str
    location: str
    liquidity: str
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
    momentum = _momentum(indicators)
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
    structure_state = _structure(structure)
    location = _location(indicators)
    liquidity = _liquidity(structure=structure, smc=smc)
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
    return MarketStateReading(
        phase=phase,
        direction=direction,
        volatility=volatility,
        momentum=momentum,
        structure=structure_state,
        location=location,
        liquidity=liquidity,
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
            f"STRUCTURE={structure_state}",
        ),
    )


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
            "readiness": reading.readiness,
            "confidence": reading.confidence,
            "evidence_complete": reading.evidence_complete,
        }
    short = [by_timeframe[tf] for tf in ("M1", "M5", "M15") if tf in by_timeframe]
    long = [by_timeframe[tf] for tf in ("H1", "H4", "D") if tf in by_timeframe]
    return {
        "contract": "QR_MARKET_STATE_TAXONOMY_V1",
        "by_timeframe": by_timeframe,
        "phase_alignment": _phase_alignment(short=short, long=long),
        "direction_alignment": _direction_alignment(list(by_timeframe.values())),
        "complete_timeframe_count": sum(
            row["evidence_complete"] is True for row in by_timeframe.values()
        ),
        "classified_timeframe_count": len(by_timeframe),
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


def _structure(structure: Any) -> str:
    event = getattr(structure, "last_event", None)
    if event is None:
        return "INTACT"
    if getattr(event, "close_confirmed", True) is not True:
        return "LIQUIDITY_TEST"
    kind = str(getattr(event, "kind", "") or "").upper()
    if kind.startswith("CHOCH_"):
        return "REVERSAL_CONFIRMED"
    if kind.startswith("BOS_"):
        return "BREAKOUT_CONFIRMED"
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


def _liquidity(*, structure: Any, smc: Any) -> str:
    if tuple(getattr(smc, "sweeps", ()) or ()):
        return "SWEPT"
    if tuple(getattr(structure, "liquidity", ()) or ()) or tuple(
        getattr(smc, "liquidity_clusters", ()) or ()
    ):
        return "POOL_PRESENT"
    return "CLEAR"


def _readiness(*, phase: str, structure: str, location: str, liquidity: str) -> str:
    if phase == "TREND":
        return "ACTIVE"
    if structure in {"BREAKOUT_CONFIRMED", "REVERSAL_CONFIRMED"}:
        return "TRIGGERED"
    if phase == "PRE_TREND" or (phase == "RANGE" and location != "MIDDLE_THIRD"):
        return "ARMED"
    if liquidity == "SWEPT":
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
    "MarketStateReading",
    "classify_market_state",
    "summarize_market_states",
]
