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
FORECAST_HTF_TREND_PRIOR_PER_TF = float(os.environ.get("QR_FORECAST_HTF_TREND_PRIOR_PER_TF", "12.0"))
FORECAST_STRONG_ADX = float(os.environ.get("QR_FORECAST_STRONG_ADX", "25.0"))
FORECAST_COUNTERTREND_UNCONFIRMED_MULT = float(
    os.environ.get("QR_FORECAST_COUNTERTREND_UNCONFIRMED_MULT", "0.55")
)


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


def _has_calibration_samples(
    hit_rates: Dict[str, Dict[str, Any]],
    signal_name: str,
    pair: str,
    *,
    regime: Optional[str] = None,
) -> bool:
    try:
        from quant_rabbit.strategy.projection_ledger import CONFIDENCE_MIN_SAMPLES
    except Exception:
        return False
    by_key = hit_rates.get(signal_name) or {}
    candidates: list[Optional[Dict[str, Any]]] = []
    if regime is not None:
        candidates.append(by_key.get(f"{pair}:{regime}"))
    candidates.append(by_key.get(f"{pair}:_all_regimes"))
    if regime is not None:
        candidates.append(by_key.get(f"_all_pairs:{regime}"))
    candidates.append(by_key.get("_all_pairs:_all_regimes"))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            samples = int(candidate.get("samples", 0) or 0)
        except (TypeError, ValueError):
            continue
        if samples >= CONFIDENCE_MIN_SAMPLES:
            return True
    return False


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
    contributions: list[tuple[str, float, str]] = []

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
            contributions.append(("RANGE", magnitude * 0.5, rationale))
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

    total = up_score + down_score + range_score
    if total <= 0:
        return DirectionalForecast(
            pair=pair, direction="UNCLEAR", confidence=0.0,
            invalidation_price=None, target_price=None, horizon_min=0,
            drivers_for=(), drivers_against=(),
            rationale_summary="no detector evidence",
            current_price=current_price,
            up_score=up_score,
            down_score=down_score,
            range_score=range_score,
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
        total = up_score + down_score + range_score
        candidates = [("UP", up_score), ("DOWN", down_score), ("RANGE", range_score)]
        candidates.sort(key=lambda x: -x[1])
        winner, winner_score = candidates[0]
    elif adjustment_reason:
        contributions.append((winner, 0.1, adjustment_reason))
    runner_up_score = candidates[1][1]

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
            component_scores={"UP": up_score, "DOWN": down_score, "RANGE": range_score},
        )

    raw_confidence = min(1.0, (margin / total) + 0.3)

    # Bayesian calibration at forecast level
    if hit_rates is not None:
        from quant_rabbit.strategy.projection_ledger import confidence_calibration
        cal_signal_name = "directional_forecast"
        if winner in {"UP", "DOWN"}:
            directional_cal_signal_name = f"directional_forecast_{winner.lower()}"
            if _has_calibration_samples(
                hit_rates,
                directional_cal_signal_name,
                pair,
                regime=regime,
            ):
                cal_signal_name = directional_cal_signal_name
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
        f"UP={up_score:.1f} DOWN={down_score:.1f} RANGE={range_score:.1f} → "
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
        component_scores={"UP": up_score, "DOWN": down_score, "RANGE": range_score},
    )
