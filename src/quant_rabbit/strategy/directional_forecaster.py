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

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "invalidation_price": self.invalidation_price,
            "target_price": self.target_price,
            "horizon_min": self.horizon_min,
            "drivers_for": list(self.drivers_for),
            "drivers_against": list(self.drivers_against),
            "rationale_summary": self.rationale_summary,
        }


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
        )

    up_score = 0.0
    down_score = 0.0
    range_score = 0.0
    drivers_for: List[str] = []
    drivers_against: List[str] = []

    def _add(sig_direction: str, magnitude: float, rationale: str) -> None:
        nonlocal up_score, down_score, range_score
        if magnitude <= 0:
            return
        if sig_direction == "UP":
            up_score += magnitude
        elif sig_direction == "DOWN":
            down_score += magnitude
        elif sig_direction in ("RANGE", "EITHER"):
            # EITHER signals (volatility expansion, etc) don't pick a
            # side but indicate uncertainty — add to range pool with
            # half weight.
            range_score += magnitude * 0.5

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

    # Regime label can push toward RANGE explicitly.
    confluence = pair_chart.get("confluence") or {}
    regime_label = str(confluence.get("dominant_regime") or "").upper()
    if "RANGE" in regime_label:
        range_score += 15.0
    sigma_24h = _to_float(confluence.get("range_24h_sigma_multiple"))
    if sigma_24h is not None and sigma_24h < 1.0:
        range_score += 10.0  # very tight range

    total = up_score + down_score + range_score
    if total <= 0:
        return DirectionalForecast(
            pair=pair, direction="UNCLEAR", confidence=0.0,
            invalidation_price=None, target_price=None, horizon_min=0,
            drivers_for=(), drivers_against=(),
            rationale_summary="no detector evidence",
        )

    # Pick winner
    candidates = [("UP", up_score), ("DOWN", down_score), ("RANGE", range_score)]
    candidates.sort(key=lambda x: -x[1])
    winner, winner_score = candidates[0]
    runner_up_score = candidates[1][1]

    # Decisiveness: need clear winner, not close call
    margin = winner_score - runner_up_score
    if margin < winner_score * 0.25:
        # Too contested → UNCLEAR
        return DirectionalForecast(
            pair=pair, direction="UNCLEAR",
            confidence=margin / max(winner_score, 1.0),
            invalidation_price=None, target_price=None, horizon_min=0,
            drivers_for=(), drivers_against=(),
            rationale_summary=f"contested: {candidates[0][0]}={winner_score:.1f} vs {candidates[1][0]}={runner_up_score:.1f} (margin {margin:.1f} < 25%)",
        )

    raw_confidence = min(1.0, (margin / total) + 0.3)

    # Bayesian calibration at forecast level
    if hit_rates is not None:
        from quant_rabbit.strategy.projection_ledger import confidence_calibration
        cal_mult = confidence_calibration(
            "directional_forecast", pair, hit_rates=hit_rates, regime=regime,
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
        f"UP={up_score:.1f} DOWN={down_score:.1f} RANGE={range_score:.1f} → "
        f"{winner} conf {raw_confidence:.2f} × cal {cal_mult:.2f} = {calibrated_confidence:.2f}"
    )

    return DirectionalForecast(
        pair=pair, direction=winner, confidence=calibrated_confidence,
        invalidation_price=invalidation_price, target_price=target_price,
        horizon_min=horizon_min,
        drivers_for=tuple(drivers_for[:5]),
        drivers_against=tuple(drivers_against[:5]),
        rationale_summary=rationale_summary,
    )
