"""Single directional forecast per pair — the focal predictor.

User insight 2026-05-15:
  「論理や計算、経験や統計で仮説や予測をたてる。
   そして、その方向にベットする。その方向性が間違いすぎなんでしょ？
   間違えたら損切りとか、あってるならTP伸ばすとか。
   レンジとかもあるから、それも予測段階」

The 17 detectors (pattern_signals, forward_projection,
correlation_predictor, path_projection, reversal_signal, ...) all
emit individual scores. The trader_brain.run() ended up summing
those into a single lane score, but the SCORE is not the same thing
as a DIRECTIONAL FORECAST. The same +30 score can come from:
  - 5 weak detectors all leaning UP (noisy LONG)
  - 1 strong RSI divergence + 2 confluences (high-conviction LONG)
  - mixed signals that average to UP (false LONG)

These should NOT all trigger an entry the same way. This module
synthesizes a single `DirectionalForecast` per pair from all
detector outputs:

  direction: "UP" | "DOWN" | "RANGE" | "UNCLEAR"
  confidence: 0.0-1.0 (probability the direction is correct)
  invalidation_price: price level where forecast is proven wrong
  target_price: price level where forecast is proven right
  horizon_min: expected time to play out

Trader_brain consumes this single forecast:
  - LONG entry iff direction == "UP" AND confidence ≥ ENTRY_CONFIDENCE_MIN
  - SHORT entry iff direction == "DOWN" AND confidence ≥ ENTRY_CONFIDENCE_MIN
  - RANGE / UNCLEAR → NO TRADE (critical — current system enters
    even when ranging, that's the leak)

Position management consumes forecast EVOLUTION:
  - Position UP + forecast still UP → HOLD or EXTEND TP
  - Position UP + forecast flipped to DOWN for ≥ FLIP_PERSISTENCE_CYCLES
    → CLOSE (this is 「間違えた」 → 損切り)
  - Position UP + forecast went to RANGE/UNCLEAR for ≥ persistence
    → CLOSE (lost edge, recycle capital)

Forecast quality is tracked via projection_ledger (record each
forecast, verify after horizon_min, compute pair × regime hit rate).
Confidence is then BAYESIAN-CALIBRATED per past accuracy — same
mechanism the individual detectors use, but at the forecast level.

Kill switch: `QR_DISABLE_DIRECTIONAL_FORECASTER=1`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.strategy.price_action import structural_tp_target


# Minimum confidence to enable entry. Below this → no trade.
ENTRY_CONFIDENCE_MIN = float(os.environ.get("QR_FORECAST_ENTRY_CONFIDENCE_MIN", "0.55"))
# Cycles a flipped forecast must persist before triggering CLOSE.
FLIP_PERSISTENCE_CYCLES = int(os.environ.get("QR_FORECAST_FLIP_PERSISTENCE", "3"))
# Cycles a RANGE/UNCLEAR forecast must persist before triggering CLOSE.
RANGE_PERSISTENCE_CYCLES = int(os.environ.get("QR_FORECAST_RANGE_PERSISTENCE", "5"))

# Pair-chart prior used by the pair-level forecast, not by risk validation.
# A large MTF score gap means the indicator panel is not "just noise"; it
# should be a prior that reversal/short-term patterns must overcome.
FORECAST_SCORE_GAP_PRIOR_GAIN = float(os.environ.get("QR_FORECAST_SCORE_GAP_PRIOR_GAIN", "35.0"))
FORECAST_SCORE_GAP_PRIOR_CAP = float(os.environ.get("QR_FORECAST_SCORE_GAP_PRIOR_CAP", "30.0"))
FORECAST_SCORE_MOMENTUM_MIN_DELTA = float(os.environ.get("QR_FORECAST_SCORE_MOMENTUM_MIN_DELTA", "0.08"))
FORECAST_SCORE_MOMENTUM_PRIOR_GAIN = float(os.environ.get("QR_FORECAST_SCORE_MOMENTUM_PRIOR_GAIN", "50.0"))
FORECAST_SCORE_MOMENTUM_PRIOR_CAP = float(os.environ.get("QR_FORECAST_SCORE_MOMENTUM_PRIOR_CAP", "24.0"))
FORECAST_MARKET_LOCATION_EXTREME = float(os.environ.get("QR_FORECAST_MARKET_LOCATION_EXTREME", "0.15"))
FORECAST_MARKET_LOCATION_PRIOR_BASE = float(os.environ.get("QR_FORECAST_MARKET_LOCATION_PRIOR_BASE", "5.0"))
FORECAST_MARKET_LOCATION_PRIOR_CAP = float(os.environ.get("QR_FORECAST_MARKET_LOCATION_PRIOR_CAP", "12.0"))
FORECAST_TECH_FAMILY_MIN_SCORE = float(os.environ.get("QR_FORECAST_TECH_FAMILY_MIN_SCORE", "0.35"))
FORECAST_TECH_TREND_PRIOR_GAIN = float(os.environ.get("QR_FORECAST_TECH_TREND_PRIOR_GAIN", "14.0"))
FORECAST_TECH_TREND_PRIOR_CAP = float(os.environ.get("QR_FORECAST_TECH_TREND_PRIOR_CAP", "20.0"))
FORECAST_TECH_MEAN_REV_MIN_SCORE = float(os.environ.get("QR_FORECAST_TECH_MEAN_REV_MIN_SCORE", "0.55"))
FORECAST_TECH_MEAN_REV_PRIOR_GAIN = float(os.environ.get("QR_FORECAST_TECH_MEAN_REV_PRIOR_GAIN", "10.0"))
FORECAST_TECH_MEAN_REV_PRIOR_CAP = float(os.environ.get("QR_FORECAST_TECH_MEAN_REV_PRIOR_CAP", "16.0"))
FORECAST_TECH_BREAKOUT_MIN_SCORE = float(os.environ.get("QR_FORECAST_TECH_BREAKOUT_MIN_SCORE", "0.60"))
FORECAST_TECH_BREAKOUT_PRIOR_GAIN = float(os.environ.get("QR_FORECAST_TECH_BREAKOUT_PRIOR_GAIN", "8.0"))
FORECAST_TECH_BREAKOUT_PRIOR_CAP = float(os.environ.get("QR_FORECAST_TECH_BREAKOUT_PRIOR_CAP", "12.0"))
FORECAST_TECH_DISAGREEMENT_RANGE_MIN = float(os.environ.get("QR_FORECAST_TECH_DISAGREEMENT_RANGE_MIN", "0.75"))
FORECAST_TECH_DISAGREEMENT_RANGE_PRIOR = float(os.environ.get("QR_FORECAST_TECH_DISAGREEMENT_RANGE_PRIOR", "10.0"))
# Range-phase priors are forecast-score magnitudes, not risk limits. They
# encode market structure reality: a stable low-ADX/high-Chop box is tradeable
# only as rail rotation; a low-volatility squeeze is a two-sided breakout risk;
# a close on the box rail with trend/breakout family support is a directional
# break. Fixed defaults keep behavior reproducible across pairs until the
# projection ledger has enough samples to learn pair/session-specific priors;
# replace these env-tunable values with calibrated hit-rate priors when that
# sample is available.
FORECAST_RANGE_INSIDE_PRIOR = float(os.environ.get("QR_FORECAST_RANGE_INSIDE_PRIOR", "26.0"))
FORECAST_RANGE_FORMING_PRIOR = float(os.environ.get("QR_FORECAST_RANGE_FORMING_PRIOR", "14.0"))
FORECAST_RANGE_BREAKOUT_PENDING_PRIOR = float(os.environ.get("QR_FORECAST_RANGE_BREAKOUT_PENDING_PRIOR", "18.0"))
FORECAST_RANGE_BREAKOUT_DIRECTION_PRIOR = float(os.environ.get("QR_FORECAST_RANGE_BREAKOUT_DIRECTION_PRIOR", "18.0"))
FORECAST_RANGE_PHASE_MIN_EVIDENCE = float(os.environ.get("QR_FORECAST_RANGE_PHASE_MIN_EVIDENCE", "2.0"))
FORECAST_RANGE_FORMING_MIN_EVIDENCE = float(os.environ.get("QR_FORECAST_RANGE_FORMING_MIN_EVIDENCE", "1.5"))
FORECAST_RANGE_BREAKOUT_MIN_EVIDENCE = float(os.environ.get("QR_FORECAST_RANGE_BREAKOUT_MIN_EVIDENCE", "1.3"))
FORECAST_HTF_TREND_PRIOR_PER_TF = float(os.environ.get("QR_FORECAST_HTF_TREND_PRIOR_PER_TF", "12.0"))
FORECAST_STRONG_ADX = float(os.environ.get("QR_FORECAST_STRONG_ADX", "25.0"))
FORECAST_COUNTERTREND_UNCONFIRMED_MULT = float(
    os.environ.get("QR_FORECAST_COUNTERTREND_UNCONFIRMED_MULT", "0.55")
)

RANGE_PHASE_TIMEFRAMES = {"M5", "M15", "M30", "H1"}


@dataclass(frozen=True)
class DirectionalForecast:
    pair: str
    direction: str  # "UP" | "DOWN" | "RANGE" | "UNCLEAR"
    confidence: float  # 0.0-1.0
    invalidation_price: Optional[float]  # where forecast is wrong
    target_price: Optional[float]  # where forecast is right
    horizon_min: int  # expected play-out time
    drivers_for: tuple[str, ...]  # rationale lines supporting
    drivers_against: tuple[str, ...]  # rationale lines opposing
    rationale_summary: str
    current_price: Optional[float] = None
    up_score: float = 0.0
    down_score: float = 0.0
    range_score: float = 0.0
    raw_confidence: float = 0.0
    calibration_multiplier: float = 1.0
    component_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "current_price": self.current_price,
            "invalidation_price": self.invalidation_price,
            "target_price": self.target_price,
            "horizon_min": self.horizon_min,
            "drivers_for": list(self.drivers_for),
            "drivers_against": list(self.drivers_against),
            "rationale_summary": self.rationale_summary,
            "up_score": round(self.up_score, 3),
            "down_score": round(self.down_score, 3),
            "range_score": round(self.range_score, 3),
            "raw_confidence": round(self.raw_confidence, 3),
            "calibration_multiplier": round(self.calibration_multiplier, 3),
            "component_scores": {k: round(v, 3) for k, v in self.component_scores.items()},
        }


@dataclass(frozen=True)
class _RangePhase:
    phase: str
    confidence: float
    direction: str | None
    rationale: str
    evidence: tuple[str, ...]


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_DIRECTIONAL_FORECASTER", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pip_factor(pair: str, pair_chart: Dict[str, Any]) -> float:
    for view in pair_chart.get("views") or []:
        if not isinstance(view, dict):
            continue
        indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else {}
        pip_size = _to_float((indicators or {}).get("pip_size"))
        if pip_size is not None and pip_size > 0:
            return 1.0 / pip_size
    return float(instrument_pip_factor(pair))


def _round_price(pair: str, price: Optional[float]) -> Optional[float]:
    if price is None:
        return None
    return round(price, 3 if instrument_pip_factor(pair) == 100 else 5)


def _collect_structural_levels(pair_chart: Dict[str, Any], *, side: str) -> list[float]:
    """Collect same-side price anchors from all chart views.

    The forecast target must be a market level, not a fixed pip guess. Pull
    from liquidity pools, swing pivots, dealing range rails, and indicator
    channel rails already derived by the chart reader.
    """
    side = side.upper()
    prices: list[float] = []
    for view in pair_chart.get("views") or []:
        if not isinstance(view, dict):
            continue
        structure = view.get("structure") if isinstance(view.get("structure"), dict) else {}
        for pool in (structure or {}).get("liquidity") or []:
            if not isinstance(pool, dict):
                continue
            pool_side = str(pool.get("side") or "").upper()
            if side == "HIGH" and "HIGH" not in pool_side:
                continue
            if side == "LOW" and "LOW" not in pool_side:
                continue
            price = _to_float(pool.get("price"))
            if price is not None and price > 0:
                prices.append(price)
        for swing in (structure or {}).get("swings") or []:
            if not isinstance(swing, dict):
                continue
            if str(swing.get("side") or "").upper() != side:
                continue
            price = _to_float(swing.get("price"))
            if price is not None and price > 0:
                prices.append(price)

        smc = view.get("smc") if isinstance(view.get("smc"), dict) else {}
        dealing_range = (smc or {}).get("dealing_range") if isinstance((smc or {}).get("dealing_range"), dict) else {}
        dr_key = "swing_high" if side == "HIGH" else "swing_low"
        price = _to_float((dealing_range or {}).get(dr_key))
        if price is not None and price > 0:
            prices.append(price)

        indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else {}
        indicator_keys = (
            ("donchian_high", "linreg_channel_upper", "bb_upper", "avwap_upper_1sd", "avwap_upper_2sd", "avwap_swing_high")
            if side == "HIGH"
            else ("donchian_low", "linreg_channel_lower", "bb_lower", "avwap_lower_1sd", "avwap_lower_2sd", "avwap_swing_low")
        )
        for key in indicator_keys:
            price = _to_float((indicators or {}).get(key))
            if price is not None and price > 0:
                prices.append(price)
    return prices


def _nearest_level(prices: list[float], *, above: Optional[float] = None, below: Optional[float] = None) -> Optional[float]:
    if above is not None:
        candidates = [p for p in prices if p > above]
        return min(candidates) if candidates else None
    if below is not None:
        candidates = [p for p in prices if p < below]
        return max(candidates) if candidates else None
    return None


def _geometry_valid(direction: str, *, current_price: float, target_price: Optional[float], invalidation_price: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    if direction == "UP":
        target = target_price if target_price is not None and target_price > current_price else None
        invalidation = invalidation_price if invalidation_price is not None and invalidation_price < current_price else None
        return target, invalidation
    if direction == "DOWN":
        target = target_price if target_price is not None and target_price < current_price else None
        invalidation = invalidation_price if invalidation_price is not None and invalidation_price > current_price else None
        return target, invalidation
    return None, None


def _forecast_geometry(
    *,
    pair: str,
    pair_chart: Dict[str, Any],
    direction: str,
    current_price: float,
) -> tuple[Optional[float], Optional[float]]:
    """Return target/invalidation levels that are on the correct side.

    This is deliberately strict: a DOWN forecast with a target above current
    price is not a target, it is bad evidence. Invalid geometry is nulled
    instead of being persisted into forecast_history.
    """
    if direction not in {"UP", "DOWN"}:
        return None, None

    pip_factor = _pip_factor(pair, pair_chart)
    side = "LONG" if direction == "UP" else "SHORT"
    structural_target, _reason = structural_tp_target(
        pair_chart,
        side=side,
        current_price=current_price,
        pip_factor=pip_factor,
        intent="HARVEST",
    )

    high_levels = _collect_structural_levels(pair_chart, side="HIGH")
    low_levels = _collect_structural_levels(pair_chart, side="LOW")
    if direction == "UP":
        target_price = structural_target or _nearest_level(high_levels, above=current_price)
        invalidation_price = _nearest_level(low_levels, below=current_price)
    else:
        target_price = structural_target or _nearest_level(low_levels, below=current_price)
        invalidation_price = _nearest_level(high_levels, above=current_price)

    target_price, invalidation_price = _geometry_valid(
        direction,
        current_price=current_price,
        target_price=target_price,
        invalidation_price=invalidation_price,
    )
    return _round_price(pair, target_price), _round_price(pair, invalidation_price)


def _view_by_tf(pair_chart: Dict[str, Any], timeframe: str) -> Dict[str, Any] | None:
    for view in pair_chart.get("views") or []:
        if isinstance(view, dict) and str(view.get("granularity") or "").upper() == timeframe:
            return view
    return None


def _view_adx(view: Dict[str, Any] | None) -> Optional[float]:
    if not isinstance(view, dict):
        return None
    indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else {}
    return _to_float((indicators or {}).get("adx_14") or (indicators or {}).get("adx"))


def _regime_direction(regime: object) -> str | None:
    text = str(regime or "").upper()
    if text.startswith("TREND_UP") or text.startswith("IMPULSE_UP"):
        return "UP"
    if text.startswith("TREND_DOWN") or text.startswith("IMPULSE_DOWN"):
        return "DOWN"
    return None


def _tf_weight(timeframe: str) -> float:
    return {
        "M1": 0.5,
        "M5": 1.0,
        "M15": 1.5,
        "M30": 1.5,
        "H1": 2.0,
        "H4": 1.7,
        "D": 1.0,
    }.get(timeframe.upper(), 1.0)


def _market_location_supports_direction(confluence: Dict[str, Any], direction: str) -> bool:
    p24 = _to_float(confluence.get("price_percentile_24h"))
    p7 = _to_float(confluence.get("price_percentile_7d"))
    if direction == "UP":
        return (
            (p24 is not None and p24 <= FORECAST_MARKET_LOCATION_EXTREME)
            or (p7 is not None and p7 <= FORECAST_MARKET_LOCATION_EXTREME)
        )
    if direction == "DOWN":
        upper = 1.0 - FORECAST_MARKET_LOCATION_EXTREME
        return (
            (p24 is not None and p24 >= upper)
            or (p7 is not None and p7 >= upper)
        )
    return False


def _has_range_context(pair_chart: Dict[str, Any]) -> bool:
    confluence = pair_chart.get("confluence") if isinstance(pair_chart.get("confluence"), dict) else {}
    if "RANGE" in str((confluence or {}).get("dominant_regime") or "").upper():
        return True
    for view in pair_chart.get("views") or []:
        if not isinstance(view, dict):
            continue
        if "RANGE" in str(view.get("regime") or "").upper():
            return True
        reading = view.get("regime_reading") if isinstance(view.get("regime_reading"), dict) else {}
        if str((reading or {}).get("state") or "").upper() == "RANGE":
            return True
        indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else {}
        adx = _to_float((indicators or {}).get("adx_14"))
        chop = _to_float((indicators or {}).get("choppiness_14"))
        if adx is not None and chop is not None and adx < 20.0 and chop > 61.8:
            return True
    return False


def _percent_0_100(value: Any) -> Optional[float]:
    pct = _to_float(value)
    if pct is None:
        return None
    return pct * 100.0 if 0.0 <= pct <= 1.0 else pct


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    numeric = _to_float(value)
    return numeric is not None and numeric > 0


def _indicator_first(indicators: Dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        value = _to_float((indicators or {}).get(key))
        if value is not None and value > 0:
            return value
    return None


def _range_position(indicators: Dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    close = _to_float((indicators or {}).get("close"))
    low = _indicator_first(
        indicators,
        ("donchian_low", "bb_lower", "linreg_channel_lower", "avwap_lower_2sd", "avwap_swing_low"),
    )
    high = _indicator_first(
        indicators,
        ("donchian_high", "bb_upper", "linreg_channel_upper", "avwap_upper_2sd", "avwap_swing_high"),
    )
    if close is None or low is None or high is None or high <= low:
        return close, None, None
    position = (close - low) / (high - low)
    return close, low, max(0.0, min(1.0, position))


def _view_regime_state(view: Dict[str, Any]) -> str:
    reading = view.get("regime_reading") if isinstance(view.get("regime_reading"), dict) else {}
    state = str((reading or {}).get("state") or "").upper()
    if state:
        return state
    return str(view.get("regime") or "").upper()


def _range_phase_analysis(pair_chart: Dict[str, Any]) -> _RangePhase:
    """Classify range context before the forecast winner is selected.

    A stable range and a compressed box that is about to break are opposite
    trading problems. The first can support RANGE_ROTATION rails; the second
    should block fade entries until direction is confirmed.
    """
    stable = 0.0
    forming = 0.0
    pending = 0.0
    break_up = 0.0
    break_down = 0.0
    evidence: list[str] = []

    for view in pair_chart.get("views") or []:
        if not isinstance(view, dict):
            continue
        tf = str(view.get("granularity") or "").upper()
        if tf not in RANGE_PHASE_TIMEFRAMES:
            continue
        weight = _tf_weight(tf)
        state = _view_regime_state(view)
        indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else {}
        families = view.get("family_scores") if isinstance(view.get("family_scores"), dict) else {}
        reading = view.get("regime_reading") if isinstance(view.get("regime_reading"), dict) else {}
        reading_conf = max(0.35, min(1.0, _to_float((reading or {}).get("confidence")) or 0.65))

        adx = _to_float((indicators or {}).get("adx_14") or (indicators or {}).get("adx"))
        chop = _to_float((indicators or {}).get("choppiness_14"))
        atr_pct = _percent_0_100((reading or {}).get("atr_percentile"))
        if atr_pct is None:
            atr_pct = _percent_0_100((indicators or {}).get("atr_percentile_100"))
        bb_width_pct = _percent_0_100((indicators or {}).get("bb_width_percentile_100"))
        trend_score = _to_float((families or {}).get("trend_score")) or 0.0
        breakout_score = _to_float((families or {}).get("breakout_score")) or 0.0
        slope = _to_float((indicators or {}).get("linreg_slope_20")) or 0.0
        close, lower, position = _range_position(indicators)

        if state == "RANGE":
            contribution = weight * reading_conf
            stable += contribution
            evidence.append(f"{tf} RANGE state +{contribution:.1f}")
        elif "RANGE" in str(view.get("regime") or "").upper():
            contribution = weight * 0.75
            stable += contribution
            evidence.append(f"{tf} legacy RANGE +{contribution:.1f}")

        if adx is not None and chop is not None and adx < 20.0 and chop > 61.8:
            contribution = weight * 0.65
            stable += contribution
            evidence.append(f"{tf} ADX={adx:.1f}/Chop={chop:.1f} choppy +{contribution:.1f}")

        if state == "BREAKOUT_PENDING":
            contribution = weight * reading_conf * 1.2
            pending += contribution
            evidence.append(f"{tf} BREAKOUT_PENDING +{contribution:.1f}")

        squeeze = _truthy_flag((indicators or {}).get("bb_squeeze"))
        if squeeze and (
            (atr_pct is not None and atr_pct <= 25.0)
            or (bb_width_pct is not None and bb_width_pct <= 25.0)
        ):
            contribution = weight * 0.9
            pending += contribution
            evidence.append(f"{tf} squeeze atr_pct={atr_pct} bb_width_pct={bb_width_pct} +{contribution:.1f}")

        if state in {"TREND_WEAK", "TRANSITION"} and (
            (adx is not None and adx < 25.0)
            or (chop is not None and chop >= 45.0)
            or (bb_width_pct is not None and bb_width_pct <= 35.0)
        ):
            contribution = weight * 0.55
            forming += contribution
            evidence.append(f"{tf} {state} soft trend/chop +{contribution:.1f}")

        if close is not None and lower is not None and position is not None:
            if position >= 0.985 and (breakout_score > 0.20 or trend_score > 0.25 or slope > 0.0):
                contribution = weight * min(1.3, 0.7 + max(breakout_score, trend_score, 0.0))
                break_up += contribution
                evidence.append(f"{tf} close at upper rail pos={position:.2f} trend={trend_score:+.2f} breakout={breakout_score:+.2f} +{contribution:.1f}")
            elif position <= 0.015 and (breakout_score < -0.20 or trend_score < -0.25 or slope < 0.0):
                contribution = weight * min(1.3, 0.7 + max(abs(breakout_score), abs(trend_score), 0.0))
                break_down += contribution
                evidence.append(f"{tf} close at lower rail pos={position:.2f} trend={trend_score:+.2f} breakout={breakout_score:+.2f} +{contribution:.1f}")

    total_evidence = stable + forming + pending + break_up + break_down
    if total_evidence <= 0:
        return _RangePhase("NONE", 0.0, None, "no range-phase evidence", ())

    top_break = max(break_up, break_down)
    if top_break >= FORECAST_RANGE_BREAKOUT_MIN_EVIDENCE and top_break >= pending * 0.7:
        direction = "UP" if break_up >= break_down else "DOWN"
        confidence = min(1.0, top_break / max(FORECAST_RANGE_PHASE_MIN_EVIDENCE * 2.0, 1.0))
        return _RangePhase(
            f"BREAKOUT_{direction}",
            confidence,
            direction,
            f"range breakout confirmed {direction}: up={break_up:.1f} down={break_down:.1f} pending={pending:.1f}",
            tuple(evidence[:6]),
        )

    if pending >= FORECAST_RANGE_PHASE_MIN_EVIDENCE and pending >= stable * 0.6:
        confidence = min(1.0, pending / max(FORECAST_RANGE_PHASE_MIN_EVIDENCE * 2.0, 1.0))
        return _RangePhase(
            "BREAKOUT_PENDING",
            confidence,
            None,
            f"range breakout pending: pending={pending:.1f} stable={stable:.1f} forming={forming:.1f}",
            tuple(evidence[:6]),
        )

    if stable >= FORECAST_RANGE_PHASE_MIN_EVIDENCE:
        confidence = min(1.0, stable / max(FORECAST_RANGE_PHASE_MIN_EVIDENCE * 2.0, 1.0))
        return _RangePhase(
            "IN_RANGE",
            confidence,
            None,
            f"inside stable range: stable={stable:.1f} forming={forming:.1f}",
            tuple(evidence[:6]),
        )

    if forming >= FORECAST_RANGE_FORMING_MIN_EVIDENCE:
        confidence = min(1.0, forming / max(FORECAST_RANGE_FORMING_MIN_EVIDENCE * 2.0, 1.0))
        return _RangePhase(
            "RANGE_FORMING",
            confidence,
            None,
            f"range forming: forming={forming:.1f} stable={stable:.1f}",
            tuple(evidence[:6]),
        )

    return _RangePhase("NONE", 0.0, None, "range-phase evidence below threshold", tuple(evidence[:6]))


def _range_phase_priors(phase: _RangePhase) -> list[tuple[str, float, str]]:
    if phase.phase == "IN_RANGE":
        magnitude = FORECAST_RANGE_INSIDE_PRIOR * max(phase.confidence, 0.35)
        return [("RANGE", magnitude, f"{phase.rationale} → RANGE rotation prior {magnitude:.1f}")]
    if phase.phase == "RANGE_FORMING":
        magnitude = FORECAST_RANGE_FORMING_PRIOR * max(phase.confidence, 0.35)
        return [("RANGE", magnitude, f"{phase.rationale} → RANGE-forming prior {magnitude:.1f}")]
    if phase.phase == "BREAKOUT_PENDING":
        magnitude = FORECAST_RANGE_BREAKOUT_PENDING_PRIOR * max(phase.confidence, 0.35)
        # Add symmetric directional pressure so the forecast becomes UNCLEAR,
        # not RANGE. That blocks rail fades until an actual break chooses side.
        return [
            ("UP", magnitude, f"{phase.rationale} → two-sided breakout risk UP {magnitude:.1f}"),
            ("DOWN", magnitude, f"{phase.rationale} → two-sided breakout risk DOWN {magnitude:.1f}"),
        ]
    if phase.phase in {"BREAKOUT_UP", "BREAKOUT_DOWN"} and phase.direction is not None:
        magnitude = FORECAST_RANGE_BREAKOUT_DIRECTION_PRIOR * max(phase.confidence, 0.35)
        return [(phase.direction, magnitude, f"{phase.rationale} → breakout {phase.direction} prior {magnitude:.1f}")]
    return []


def _family_average(
    pair_chart: Dict[str, Any],
    key: str,
    *,
    allowed_tfs: set[str] | None = None,
) -> tuple[Optional[float], tuple[str, ...], Optional[float]]:
    numerator = 0.0
    denominator = 0.0
    disagreement_num = 0.0
    evidence: list[tuple[float, str, float]] = []
    for view in pair_chart.get("views") or []:
        if not isinstance(view, dict):
            continue
        tf = str(view.get("granularity") or "").upper()
        if allowed_tfs is not None and tf not in allowed_tfs:
            continue
        families = view.get("family_scores") if isinstance(view.get("family_scores"), dict) else {}
        score = _to_float((families or {}).get(key))
        if score is None:
            continue
        disagreement = _to_float((families or {}).get("disagreement")) or 0.0
        quality = max(0.45, 1.0 - min(abs(disagreement), 1.0) * 0.35)
        weight = _tf_weight(tf) * quality
        numerator += score * weight
        denominator += weight
        disagreement_num += disagreement * weight
        evidence.append((abs(score) * weight, tf, score))
    if denominator <= 0:
        return None, (), None
    evidence.sort(key=lambda item: item[0], reverse=True)
    labels = tuple(f"{tf}={score:+.2f}" for _rank, tf, score in evidence[:3])
    return numerator / denominator, labels, disagreement_num / denominator


def _technical_family_priors(pair_chart: Dict[str, Any]) -> list[tuple[str, float, str]]:
    """Use grouped technical composites without letting collinear indicators over-vote.

    The chart reader already computes RSI/Stoch/MACD/EMA/ADX/Aroon/SuperTrend/
    BB/VWAP/ATR/Donchian as family scores. Forecasts should consume those
    families directly instead of reducing the whole chart to one current
    long/short number.
    """
    confluence = pair_chart.get("confluence") if isinstance(pair_chart.get("confluence"), dict) else {}
    priors: list[tuple[str, float, str]] = []

    trend_avg, trend_labels, trend_disagreement = _family_average(pair_chart, "trend_score")
    if trend_avg is not None and abs(trend_avg) >= FORECAST_TECH_FAMILY_MIN_SCORE:
        direction = "UP" if trend_avg > 0 else "DOWN"
        magnitude = min(FORECAST_TECH_TREND_PRIOR_CAP, abs(trend_avg) * FORECAST_TECH_TREND_PRIOR_GAIN)
        priors.append(
            (
                direction,
                magnitude,
                f"technical trend family avg={trend_avg:+.2f} {' '.join(trend_labels)} → {direction} prior {magnitude:.1f}",
            )
        )

    mean_rev_avg, mean_rev_labels, mean_rev_disagreement = _family_average(
        pair_chart,
        "mean_rev_score",
        allowed_tfs={"M1", "M5", "M15", "M30", "H1"},
    )
    if mean_rev_avg is not None and abs(mean_rev_avg) >= FORECAST_TECH_MEAN_REV_MIN_SCORE:
        direction = "UP" if mean_rev_avg > 0 else "DOWN"
        location_ok = _market_location_supports_direction(confluence, direction)
        range_ok = _has_range_context(pair_chart)
        if location_ok or range_ok or abs(mean_rev_avg) >= 0.9:
            magnitude = min(
                FORECAST_TECH_MEAN_REV_PRIOR_CAP,
                abs(mean_rev_avg) * FORECAST_TECH_MEAN_REV_PRIOR_GAIN,
            )
            if not location_ok and range_ok:
                magnitude *= 0.75
            priors.append(
                (
                    direction,
                    magnitude,
                    f"technical mean-reversion family avg={mean_rev_avg:+.2f} {' '.join(mean_rev_labels)} → {direction} prior {magnitude:.1f}",
                )
            )

    breakout_avg, breakout_labels, breakout_disagreement = _family_average(pair_chart, "breakout_score")
    if (
        breakout_avg is not None
        and trend_avg is not None
        and abs(breakout_avg) >= FORECAST_TECH_BREAKOUT_MIN_SCORE
        and abs(trend_avg) >= FORECAST_TECH_FAMILY_MIN_SCORE
    ):
        direction = "UP" if trend_avg > 0 else "DOWN"
        magnitude = min(
            FORECAST_TECH_BREAKOUT_PRIOR_CAP,
            abs(breakout_avg) * FORECAST_TECH_BREAKOUT_PRIOR_GAIN,
        )
        priors.append(
            (
                direction,
                magnitude,
                f"technical breakout family avg={breakout_avg:+.2f} with trend_avg={trend_avg:+.2f} {' '.join(breakout_labels)} → {direction} prior {magnitude:.1f}",
            )
        )

    disagreements = [d for d in (trend_disagreement, mean_rev_disagreement, breakout_disagreement) if d is not None]
    avg_disagreement = sum(disagreements) / len(disagreements) if disagreements else None
    if (
        avg_disagreement is not None
        and avg_disagreement >= FORECAST_TECH_DISAGREEMENT_RANGE_MIN
        and (trend_avg is None or abs(trend_avg) < 0.65)
    ):
        magnitude = min(
            FORECAST_TECH_DISAGREEMENT_RANGE_PRIOR,
            FORECAST_TECH_DISAGREEMENT_RANGE_PRIOR * avg_disagreement,
        )
        priors.append(
            (
                "RANGE",
                magnitude,
                f"technical family disagreement={avg_disagreement:.2f} → RANGE/stand-aside prior {magnitude:.1f}",
            )
        )

    return priors


def _score_momentum_prior(confluence: Dict[str, Any]) -> tuple[str, float, str] | None:
    momentum = confluence.get("score_momentum")
    if not isinstance(momentum, dict):
        return None
    gap_delta = _to_float(momentum.get("score_gap_delta"))
    if gap_delta is None or abs(gap_delta) < FORECAST_SCORE_MOMENTUM_MIN_DELTA:
        return None
    direction = "UP" if gap_delta > 0 else "DOWN"
    magnitude = min(FORECAST_SCORE_MOMENTUM_PRIOR_CAP, abs(gap_delta) * FORECAST_SCORE_MOMENTUM_PRIOR_GAIN)
    if magnitude <= 0:
        return None
    long_delta = _to_float(momentum.get("long_score_delta"))
    short_delta = _to_float(momentum.get("short_score_delta"))
    slope = _to_float(momentum.get("score_gap_slope_per_hour"))
    elapsed = _to_float(momentum.get("elapsed_min"))
    details: list[str] = [f"gapΔ={gap_delta:+.3f}"]
    if long_delta is not None and short_delta is not None:
        details.append(f"longΔ={long_delta:+.3f} shortΔ={short_delta:+.3f}")
    if slope is not None:
        details.append(f"slope/h={slope:+.3f}")
    if elapsed is not None:
        details.append(f"{elapsed:.0f}m")
    return (
        direction,
        magnitude,
        f"pair_chart score momentum {' '.join(details)} → {direction} turn prior {magnitude:.1f}",
    )


def _market_location_priors(confluence: Dict[str, Any]) -> list[tuple[str, float, str]]:
    """Convert broad chart location into a small directional prior.

    Location is not a standalone entry trigger. It answers "where are we in the
    story?" so a late short at the 24h/7d floor or a late long at the ceiling is
    penalized by the opposing side's forecast evidence.
    """
    p24 = _to_float(confluence.get("price_percentile_24h"))
    p7 = _to_float(confluence.get("price_percentile_7d"))
    sigma = _to_float(confluence.get("range_24h_sigma_multiple"))
    priors: list[tuple[str, float, str]] = []

    def _location_prior(direction: str, strength: float, labels: list[str]) -> None:
        if strength <= 0:
            return
        sigma_boost = 0.0
        if sigma is not None and sigma >= 3.0:
            sigma_boost = 0.2
            labels.append(f"24h_range={sigma:.2f}x")
        strength = min(1.0, strength + sigma_boost)
        magnitude = min(
            FORECAST_MARKET_LOCATION_PRIOR_CAP,
            FORECAST_MARKET_LOCATION_PRIOR_BASE
            + strength * (FORECAST_MARKET_LOCATION_PRIOR_CAP - FORECAST_MARKET_LOCATION_PRIOR_BASE),
        )
        priors.append(
            (
                direction,
                magnitude,
                f"market location {' '.join(labels)} → {direction} location prior {magnitude:.1f}",
            )
        )

    lower_strength = 0.0
    lower_labels: list[str] = []
    if p24 is not None and p24 <= FORECAST_MARKET_LOCATION_EXTREME:
        lower_strength = max(lower_strength, (FORECAST_MARKET_LOCATION_EXTREME - p24) / FORECAST_MARKET_LOCATION_EXTREME)
        lower_labels.append(f"24h_pct={p24:.2f}")
    if p7 is not None and p7 <= FORECAST_MARKET_LOCATION_EXTREME:
        lower_strength = max(lower_strength, (FORECAST_MARKET_LOCATION_EXTREME - p7) / FORECAST_MARKET_LOCATION_EXTREME)
        lower_labels.append(f"7d_pct={p7:.2f}")
    _location_prior("UP", lower_strength, lower_labels)

    upper_strength = 0.0
    upper_labels: list[str] = []
    upper_cutoff = 1.0 - FORECAST_MARKET_LOCATION_EXTREME
    if p24 is not None and p24 >= upper_cutoff:
        upper_strength = max(upper_strength, (p24 - upper_cutoff) / FORECAST_MARKET_LOCATION_EXTREME)
        upper_labels.append(f"24h_pct={p24:.2f}")
    if p7 is not None and p7 >= upper_cutoff:
        upper_strength = max(upper_strength, (p7 - upper_cutoff) / FORECAST_MARKET_LOCATION_EXTREME)
        upper_labels.append(f"7d_pct={p7:.2f}")
    _location_prior("DOWN", upper_strength, upper_labels)

    return priors


def _pair_chart_prior(pair_chart: Dict[str, Any]) -> list[tuple[str, float, str]]:
    """Turn the multi-timeframe indicator panel into forecast priors.

    This does not add a new trading gate. It makes the existing indicator stack
    count as first-class forecast evidence, so a few micro reversal patterns do
    not overpower H1/H4/D agreement without confirmation.
    """
    confluence = pair_chart.get("confluence") or {}
    priors: list[tuple[str, float, str]] = []

    score_gap = _to_float(confluence.get("score_gap"))
    balance = str(confluence.get("score_balance") or "").upper()
    if score_gap is not None and balance in {"LONG_LEAN", "SHORT_LEAN"}:
        direction = "UP" if balance == "LONG_LEAN" else "DOWN"
        magnitude = min(FORECAST_SCORE_GAP_PRIOR_CAP, abs(score_gap) * FORECAST_SCORE_GAP_PRIOR_GAIN)
        if magnitude > 0:
            priors.append(
                (
                    direction,
                    magnitude,
                    f"pair_chart {balance} score_gap={score_gap:.3f} → {direction} prior {magnitude:.1f}",
                )
            )

    momentum_prior = _score_momentum_prior(confluence)
    if momentum_prior is not None:
        priors.append(momentum_prior)

    priors.extend(_market_location_priors(confluence))
    priors.extend(_technical_family_priors(pair_chart))

    for tf in ("H1", "H4"):
        view = _view_by_tf(pair_chart, tf)
        direction = _regime_direction((view or {}).get("regime") if view else None)
        adx = _view_adx(view)
        if direction is not None and adx is not None and adx >= FORECAST_STRONG_ADX:
            priors.append(
                (
                    direction,
                    FORECAST_HTF_TREND_PRIOR_PER_TF,
                    f"{tf} {direction} trend ADX={adx:.1f} → HTF prior {FORECAST_HTF_TREND_PRIOR_PER_TF:.1f}",
                )
            )
    return priors


def _has_close_confirmed_structure(pair_chart: Dict[str, Any], direction: str) -> bool:
    expected = "UP" if direction == "UP" else "DOWN"
    # Reversal against a strong higher-TF move must be current structure, not
    # any stale historical BOS/CHOCH in the retained event list. H1 dominates
    # M15 when available; otherwise M15 is the fallback confirmation frame.
    h1 = _latest_close_confirmed_structure_direction(_view_by_tf(pair_chart, "H1"))
    if h1 is not None:
        return h1 == expected
    m15 = _latest_close_confirmed_structure_direction(_view_by_tf(pair_chart, "M15"))
    return m15 == expected if m15 is not None else False


def _latest_close_confirmed_structure_direction(view: Dict[str, Any] | None) -> str | None:
    if not isinstance(view, dict):
        return None
    structure = view.get("structure") if isinstance(view.get("structure"), dict) else {}
    latest: tuple[float, str] | None = None
    for order, event in enumerate((structure or {}).get("structure_events") or []):
        if not isinstance(event, dict):
            continue
        if not bool(event.get("close_confirmed")):
            continue
        kind = str(event.get("kind") or "").upper().split(":", 1)[0]
        if kind.endswith("_UP"):
            event_dir = "UP"
        elif kind.endswith("_DOWN"):
            event_dir = "DOWN"
        else:
            continue
        index = _to_float(event.get("index"))
        sort_key = index if index is not None else float(order)
        if latest is None or sort_key >= latest[0]:
            latest = (sort_key, event_dir)
    return latest[1] if latest is not None else None


def _score_reversal_location_confirmation(pair_chart: Dict[str, Any], direction: str) -> str | None:
    confluence = pair_chart.get("confluence") if isinstance(pair_chart.get("confluence"), dict) else {}
    momentum = (confluence or {}).get("score_momentum") if isinstance((confluence or {}).get("score_momentum"), dict) else {}
    gap_delta = _to_float((momentum or {}).get("score_gap_delta"))
    if gap_delta is None:
        return None

    p24 = _to_float((confluence or {}).get("price_percentile_24h"))
    p7 = _to_float((confluence or {}).get("price_percentile_7d"))
    lower_extreme = (
        (p24 is not None and p24 <= FORECAST_MARKET_LOCATION_EXTREME)
        or (p7 is not None and p7 <= FORECAST_MARKET_LOCATION_EXTREME)
    )
    upper_extreme = (
        (p24 is not None and p24 >= 1.0 - FORECAST_MARKET_LOCATION_EXTREME)
        or (p7 is not None and p7 >= 1.0 - FORECAST_MARKET_LOCATION_EXTREME)
    )
    min_confirmation_delta = max(FORECAST_SCORE_MOMENTUM_MIN_DELTA * 2.5, 0.2)
    if direction == "UP" and gap_delta >= min_confirmation_delta and lower_extreme:
        return (
            f"countertrend UP allowed: score momentum gapΔ={gap_delta:+.3f} from lower market location "
            f"24h_pct={p24} 7d_pct={p7}"
        )
    if direction == "DOWN" and gap_delta <= -min_confirmation_delta and upper_extreme:
        return (
            f"countertrend DOWN allowed: score momentum gapΔ={gap_delta:+.3f} from upper market location "
            f"24h_pct={p24} 7d_pct={p7}"
        )
    return None


def _countertrend_adjustment(
    pair_chart: Dict[str, Any],
    winner: str,
    winner_score: float,
) -> tuple[float, str | None]:
    if winner not in {"UP", "DOWN"}:
        return winner_score, None
    confluence = pair_chart.get("confluence") or {}
    score_gap = _to_float(confluence.get("score_gap")) or 0.0
    balance = str(confluence.get("score_balance") or "").upper()
    tf_agree = _to_float(confluence.get("tf_agreement_score"))
    prior_dir = "UP" if balance == "LONG_LEAN" else "DOWN" if balance == "SHORT_LEAN" else None
    if prior_dir is None or prior_dir == winner:
        return winner_score, None
    if abs(score_gap) < 0.2:
        return winner_score, None
    if tf_agree is not None and tf_agree < (2.0 / 3.0):
        return winner_score, None
    if _has_close_confirmed_structure(pair_chart, winner):
        return winner_score, (
            f"countertrend {winner} allowed: M15/H1 close-confirmed structure offsets "
            f"{balance} score_gap={score_gap:.3f}"
        )
    score_reversal_reason = _score_reversal_location_confirmation(pair_chart, winner)
    if score_reversal_reason:
        return winner_score, score_reversal_reason
    adjusted = winner_score * FORECAST_COUNTERTREND_UNCONFIRMED_MULT
    return adjusted, (
        f"countertrend {winner} damped {winner_score:.1f}→{adjusted:.1f}: "
        f"{balance} score_gap={score_gap:.3f}, tf_agreement={tf_agree}, "
        "no M15/H1 close-confirmed reversal"
    )


def _top_reasons(
    contributions: list[tuple[str, float, str]],
    *,
    direction: str,
    limit: int = 5,
) -> tuple[str, ...]:
    items = [
        (mag, rationale)
        for sig_dir, mag, rationale in contributions
        if sig_dir == direction and rationale
    ]
    items.sort(key=lambda item: item[0], reverse=True)
    return tuple(f"{rationale} ({mag:.1f})" for mag, rationale in items[:limit])


def synthesize_forecast(
    *,
    pair: str,
    pair_chart: Dict[str, Any],
    current_price: float,
    pattern_signals: List[Any],
    projection_signals: List[Any],
    correlation_signals: List[Any],
    paths: List[Any],
    reversal_long: Optional[Any] = None,
    reversal_short: Optional[Any] = None,
    hit_rates: Optional[Dict[str, Dict[str, Any]]] = None,
    regime: Optional[str] = None,
) -> DirectionalForecast:
    """Combine all detector outputs into ONE directional forecast.

    Algorithm:
    1. Compute UP_score = sum of detector contributions favoring UP
    2. Compute DOWN_score = sum of detector contributions favoring DOWN
    3. RANGE_score = signals from detectors that explicitly flag range
       (regime_classifier RANGING, dealing_range_top/bottom, etc.)
    4. Apply Bayesian calibration from past forecast hit-rate
    5. Decision:
       - If max(UP, DOWN) - second_max > MARGIN AND > 0 → directional
       - If RANGE_score dominant → RANGE
       - Else UNCLEAR

    Invalidation_price: nearest opposite-direction structural level
    (swing high for LONG forecasts → invalidates if price exceeds).
    Target_price: nearest same-direction liquidity / structural target.

    Confidence: computed as |UP - DOWN| / total_score, clamped 0-1,
    then multiplied by historical accuracy multiplier.
    """
    if _is_disabled():
        return DirectionalForecast(
            pair=pair, direction="UNCLEAR", confidence=0.0,
            invalidation_price=None, target_price=None, horizon_min=0,
            drivers_for=(), drivers_against=(),
            rationale_summary="forecaster disabled",
            current_price=current_price,
        )

    up_score = 0.0
    down_score = 0.0
    range_score = 0.0
    either_score = 0.0
    contributions: list[tuple[str, float, str]] = []
    range_phase = _range_phase_analysis(pair_chart)

    def _add(sig_direction: str, magnitude: float, rationale: str) -> None:
        nonlocal up_score, down_score, range_score, either_score
        if magnitude <= 0:
            return
        if sig_direction == "UP":
            up_score += magnitude
        elif sig_direction == "DOWN":
            down_score += magnitude
        elif sig_direction == "RANGE":
            range_score += magnitude
            contributions.append(("RANGE", magnitude, rationale))
            return
        elif sig_direction == "EITHER":
            # EITHER means expansion/uncertainty, not range rotation. It can
            # reduce confidence once a side exists, but must not select RANGE.
            either_score += magnitude
            contributions.append(("EITHER", magnitude, rationale))
            return
        if sig_direction in {"UP", "DOWN"}:
            contributions.append((sig_direction, magnitude, rationale))

    for s in pattern_signals or []:
        _add(getattr(s, "direction", "UNCLEAR"),
             getattr(s, "bonus_magnitude", 0) * getattr(s, "confidence", 0),
             getattr(s, "rationale", ""))
    for s in projection_signals or []:
        # News-catalyst signals have NEGATIVE bonus → treat as RANGE
        # (don't bet directionally before high-impact event)
        mag = getattr(s, "bonus_magnitude", 0) * getattr(s, "confidence", 0)
        if mag < 0:
            range_score += abs(mag)
            continue
        _add(getattr(s, "direction", "UNCLEAR"), mag, getattr(s, "rationale", ""))
    for s in correlation_signals or []:
        _add(getattr(s, "direction", "UNCLEAR"),
             getattr(s, "bonus_magnitude", 0) * getattr(s, "confidence", 0),
             getattr(s, "rationale", ""))
    for p in paths or []:
        _add(getattr(p, "direction", "UNCLEAR"),
             getattr(p, "bonus_magnitude", 0) * getattr(p, "confidence", 0),
             getattr(p, "rationale", ""))
    if reversal_long is not None:
        _add("UP", getattr(reversal_long, "bonus", 0), getattr(reversal_long, "rationale", ""))
    if reversal_short is not None:
        _add("DOWN", getattr(reversal_short, "bonus", 0), getattr(reversal_short, "rationale", ""))

    for prior_dir, prior_mag, prior_reason in _pair_chart_prior(pair_chart):
        _add(prior_dir, prior_mag, prior_reason)
    for prior_dir, prior_mag, prior_reason in _range_phase_priors(range_phase):
        _add(prior_dir, prior_mag, prior_reason)

    # Regime label can push toward RANGE explicitly.
    confluence = pair_chart.get("confluence") or {}
    regime_label = str(confluence.get("dominant_regime") or "").upper()
    if "RANGE" in regime_label:
        range_score += 15.0
        contributions.append(("RANGE", 15.0, "dominant regime RANGE → range forecast prior"))
    sigma_24h = _to_float(confluence.get("range_24h_sigma_multiple"))
    if sigma_24h is not None and sigma_24h < 1.0:
        range_score += 10.0  # very tight range
        contributions.append(("RANGE", 10.0, f"24h range sigma {sigma_24h:.2f}<1.0 → tight-range prior"))

    directional_total = up_score + down_score + range_score
    total = directional_total + either_score
    if directional_total <= 0:
        return DirectionalForecast(
            pair=pair, direction="UNCLEAR", confidence=0.0,
            invalidation_price=None, target_price=None, horizon_min=0,
            drivers_for=(), drivers_against=(),
            rationale_summary=(
                "no directional/range detector evidence"
                if either_score > 0
                else "no detector evidence"
            ),
            current_price=current_price,
            up_score=up_score,
            down_score=down_score,
            range_score=range_score,
            component_scores={"UP": up_score, "DOWN": down_score, "RANGE": range_score, "EITHER": either_score},
        )

    # Pick winner
    candidates = [("UP", up_score), ("DOWN", down_score), ("RANGE", range_score)]
    candidates.sort(key=lambda x: -x[1])
    winner, winner_score = candidates[0]
    adjusted_winner_score, adjustment_reason = _countertrend_adjustment(pair_chart, winner, winner_score)
    if adjusted_winner_score != winner_score:
        if winner == "UP":
            up_score = adjusted_winner_score
        elif winner == "DOWN":
            down_score = adjusted_winner_score
        contributions.append(("RANGE", max(winner_score - adjusted_winner_score, 0.0), adjustment_reason or "countertrend dampening"))
        total = up_score + down_score + range_score + either_score
        candidates = [("UP", up_score), ("DOWN", down_score), ("RANGE", range_score)]
        candidates.sort(key=lambda x: -x[1])
        winner, winner_score = candidates[0]
    elif adjustment_reason:
        contributions.append((winner, 0.1, adjustment_reason))
    runner_up_score = candidates[1][1]

    if range_phase.phase == "BREAKOUT_PENDING" and winner == "RANGE":
        evidence = "; ".join(range_phase.evidence[:3])
        return DirectionalForecast(
            pair=pair, direction="UNCLEAR",
            confidence=0.0,
            invalidation_price=None, target_price=None, horizon_min=0,
            drivers_for=(range_phase.rationale,) + ((evidence,) if evidence else ()),
            drivers_against=_top_reasons(contributions, direction=candidates[1][0]),
            rationale_summary=(
                f"range breakout pending blocks RANGE rotation: "
                f"UP={up_score:.1f} DOWN={down_score:.1f} RANGE={range_score:.1f} EITHER={either_score:.1f}"
            ),
            current_price=_round_price(pair, current_price),
            up_score=up_score,
            down_score=down_score,
            range_score=range_score,
            raw_confidence=0.0,
            component_scores={"UP": up_score, "DOWN": down_score, "RANGE": range_score, "EITHER": either_score},
        )

    # Decisiveness: need clear winner, not close call
    margin = winner_score - runner_up_score
    if margin < winner_score * 0.25:
        # Too contested → UNCLEAR
        return DirectionalForecast(
            pair=pair, direction="UNCLEAR",
            confidence=margin / max(winner_score, 1.0),
            invalidation_price=None, target_price=None, horizon_min=0,
            drivers_for=_top_reasons(contributions, direction=winner),
            drivers_against=_top_reasons(contributions, direction=candidates[1][0]),
            rationale_summary=f"contested: {candidates[0][0]}={winner_score:.1f} vs {candidates[1][0]}={runner_up_score:.1f} (margin {margin:.1f} < 25%)",
            current_price=current_price,
            up_score=up_score,
            down_score=down_score,
            range_score=range_score,
            raw_confidence=margin / max(winner_score, 1.0),
            component_scores={"UP": up_score, "DOWN": down_score, "RANGE": range_score, "EITHER": either_score},
        )

    raw_confidence = min(1.0, (margin / total) + 0.3)

    # Bayesian calibration at forecast level
    if hit_rates is not None:
        from quant_rabbit.strategy.projection_ledger import (
            confidence_calibration,
            select_calibration_signal_name,
        )
        cal_signal_name = select_calibration_signal_name(
            "directional_forecast",
            winner,
            pair,
            hit_rates=hit_rates,
            regime=regime,
        )
        cal_mult = confidence_calibration(
            cal_signal_name, pair, hit_rates=hit_rates, regime=regime,
        )
        calibrated_confidence = min(1.0, raw_confidence * cal_mult)
    else:
        cal_mult = 1.0
        calibrated_confidence = raw_confidence

    target_price, invalidation_price = _forecast_geometry(
        pair=pair,
        pair_chart=pair_chart,
        direction=winner,
        current_price=current_price,
    )

    horizon_min = 60 if winner != "RANGE" else 120

    rationale_summary = (
        f"UP={up_score:.1f} DOWN={down_score:.1f} RANGE={range_score:.1f} EITHER={either_score:.1f} → "
        f"{winner} conf {raw_confidence:.2f} × cal {cal_mult:.2f} = {calibrated_confidence:.2f}"
    )
    if adjustment_reason:
        rationale_summary = f"{rationale_summary}; {adjustment_reason}"
    opposite = "DOWN" if winner == "UP" else "UP" if winner == "DOWN" else candidates[1][0]

    return DirectionalForecast(
        pair=pair, direction=winner, confidence=calibrated_confidence,
        invalidation_price=invalidation_price, target_price=target_price,
        horizon_min=horizon_min,
        drivers_for=_top_reasons(contributions, direction=winner),
        drivers_against=_top_reasons(contributions, direction=opposite),
        rationale_summary=rationale_summary,
        current_price=_round_price(pair, current_price),
        up_score=up_score,
        down_score=down_score,
        range_score=range_score,
        raw_confidence=raw_confidence,
        calibration_multiplier=cal_mult,
        component_scores={"UP": up_score, "DOWN": down_score, "RANGE": range_score, "EITHER": either_score},
    )
