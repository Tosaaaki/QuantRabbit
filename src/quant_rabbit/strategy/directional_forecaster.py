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
  confidence: 0.0-1.0 calibrated direction score (not a win probability)
  invalidation_price: price level where forecast is proven wrong
  target_price: price level where forecast is proven right
  horizon_min: expected time to play out

Trader_brain consumes this single forecast:
  - LONG entry iff direction == "UP" AND confidence ≥ ENTRY_CONFIDENCE_MIN
  - SHORT entry iff direction == "DOWN" AND confidence ≥ ENTRY_CONFIDENCE_MIN
  - RANGE supports only RANGE_ROTATION with executable rail geometry.
    UNCLEAR remains NO TRADE. This keeps "no direction" from becoming
    a trend chase while still allowing a predicted box to rotate rails.

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

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Optional

from quant_rabbit.analysis.candles import (
    PAIR_TECHNICAL_CANDLE_INTEGRITY_SCHEMA,
    SUPPORTED_GRANULARITIES,
    TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT,
    TECHNICAL_CANDLE_INTEGRITY_SCHEMA,
    TECHNICAL_CANDLE_PROVENANCE_INVALID,
    TECHNICAL_CANDLE_QUARANTINE_DETAIL_LIMIT,
    TECHNICAL_CANDLE_QUARANTINE_DETAILS_ORDER,
    TECHNICAL_CANDLE_QUARANTINE_DETAILS_SELECTION,
    TECHNICAL_CANDLE_SPREAD_EXECUTION_MODE,
    TECHNICAL_CANDLE_SPREAD_EXECUTION_TIMEFRAMES,
    TECHNICAL_CANDLE_SPREAD_PROVENANCE_ONLY_MODE,
    TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
    _technical_candle_cadence_valid,
)
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.paths import ROOT
from quant_rabbit.strategy.forecast_technical_context import build_forecast_technical_context
from quant_rabbit.strategy.regime_family_weighting import forecast_direction_consistency
from quant_rabbit.strategy.price_action import structural_tp_target


# Minimum confidence to enable entry. Below this → no trade.
ENTRY_CONFIDENCE_MIN = float(os.environ.get("QR_FORECAST_ENTRY_CONFIDENCE_MIN", "0.55"))
# Cycles a flipped forecast must persist before triggering CLOSE.
FLIP_PERSISTENCE_CYCLES = int(os.environ.get("QR_FORECAST_FLIP_PERSISTENCE", "3"))
# Cycles a RANGE/UNCLEAR forecast must persist before triggering CLOSE.
RANGE_PERSISTENCE_CYCLES = int(os.environ.get("QR_FORECAST_RANGE_PERSISTENCE", "5"))

# Pair-chart indicator constants are retained for diagnostic/risk-context
# helpers. Smoothed indicator panels are lagging evidence: they may damp an
# unconfirmed countertrend story, but they must not create the forecast side.
FORECAST_SCORE_GAP_PRIOR_GAIN = float(os.environ.get("QR_FORECAST_SCORE_GAP_PRIOR_GAIN", "35.0"))
FORECAST_SCORE_GAP_PRIOR_CAP = float(os.environ.get("QR_FORECAST_SCORE_GAP_PRIOR_CAP", "30.0"))
FORECAST_MARKET_LOCATION_EXTREME = float(os.environ.get("QR_FORECAST_MARKET_LOCATION_EXTREME", "0.15"))
FORECAST_MARKET_LOCATION_PRIOR_BASE = float(os.environ.get("QR_FORECAST_MARKET_LOCATION_PRIOR_BASE", "5.0"))
FORECAST_MARKET_LOCATION_PRIOR_CAP = float(os.environ.get("QR_FORECAST_MARKET_LOCATION_PRIOR_CAP", "12.0"))
FORECAST_TECH_FAMILY_MIN_SCORE = float(os.environ.get("QR_FORECAST_TECH_FAMILY_MIN_SCORE", "0.35"))
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
FORECAST_STRONG_ADX = float(os.environ.get("QR_FORECAST_STRONG_ADX", "25.0"))
FORECAST_COUNTERTREND_UNCONFIRMED_MULT = float(
    os.environ.get("QR_FORECAST_COUNTERTREND_UNCONFIRMED_MULT", "0.55")
)
# Forecast geometry must prove a move beyond the current operating noise, not
# the next tick. One M1/M5 ATR is the observed single-window noise envelope, so
# target/invalidation levels inside that distance are treated as non-structural
# unless a future calibrated pair/session model replaces this floor.
FORECAST_GEOMETRY_NOISE_ATR_MULT = float(os.environ.get("QR_FORECAST_GEOMETRY_NOISE_ATR_MULT", "1.0"))
# Execution spread belongs to the order gateway, not to future-price geometry.
# A transient rollover spread must not discard nearby structure and replace it
# with a target tens or hundreds of pips away. Forecast levels therefore use
# closed-chart ATR/structure only; live spread, slippage and RR remain mandatory
# execution gates downstream.
# Missing robust target or invalidation makes a directional forecast less
# actionable, but not useless for advisory bias. This confidence haircut is an
# evidence-quality penalty, not a trade gate; ENTRY_CONFIDENCE_MIN remains the
# execution gate.
FORECAST_INCOMPLETE_GEOMETRY_CONFIDENCE_MULT = float(
    os.environ.get("QR_FORECAST_INCOMPLETE_GEOMETRY_CONFIDENCE_MULT", "0.75")
)
# A detector with a large, audited sub-random hit rate is not just "lower
# confidence"; it is a bad direction voter. Calibration still handles ordinary
# noise and small samples. This hard exclusion is reserved for enough resolved
# HIT/MISS samples that a poor detector should not create the final forecast
# side at all.
FORECAST_PROJECTION_WEAK_BLOCK_HIT_RATE = float(
    os.environ.get("QR_FORECAST_PROJECTION_WEAK_BLOCK_HIT_RATE", "0.45")
)
FORECAST_PROJECTION_WEAK_BLOCK_MIN_SAMPLES = max(
    1,
    int(os.environ.get("QR_FORECAST_PROJECTION_WEAK_BLOCK_MIN_SAMPLES", "30")),
)

# Forecast horizons are timeframe definitions, not entry gates. They let the
# trader distinguish an execution scalp from an operating swing or H4/D anchor
# while the same risk/profile/gateway checks still decide whether any order can
# be sent. These values map directly to chart bar durations: one hour for
# M5/M15 execution follow-through, three hours for M15/M30/H1 operating swings,
# four hours for H4 anchors, and one day for D+H4 aligned anchors. If live
# evaluation shows a different play-out window per pair/session, replace these
# constants with horizon-specific projection-ledger calibration.
FORECAST_EXECUTION_HORIZON_MIN = int(os.environ.get("QR_FORECAST_EXECUTION_HORIZON_MIN", "60"))
FORECAST_OPERATING_SWING_HORIZON_MIN = int(os.environ.get("QR_FORECAST_OPERATING_SWING_HORIZON_MIN", "180"))
FORECAST_H4_ANCHOR_HORIZON_MIN = int(os.environ.get("QR_FORECAST_H4_ANCHOR_HORIZON_MIN", "240"))
FORECAST_D_ANCHOR_HORIZON_MIN = int(os.environ.get("QR_FORECAST_D_ANCHOR_HORIZON_MIN", "1440"))
FORECAST_RANGE_HORIZON_MIN = int(os.environ.get("QR_FORECAST_RANGE_HORIZON_MIN", "120"))

RANGE_PHASE_TIMEFRAMES = {"M5", "M15", "M30", "H1"}

# Full-cycle pair charts are rebuilt on the five-minute trader cadence. The
# observed chart-to-final-snapshot gap is only a few minutes, while the
# guardian's established freshness budget is 12 minutes; reuse that bounded
# operational window so a skipped refresh cannot authorize a later forecast.
TECHNICAL_CANDLE_FORECAST_MAX_AGE_SECONDS = 12 * 60
# These timestamps are written on the same host. Permit only tiny
# serialization/clock-order jitter, never a future-dated freshness lease.
TECHNICAL_CANDLE_FORECAST_MAX_FUTURE_SKEW_SECONDS = 5
TECHNICAL_CANDLE_LATEST_COMPLETE_MAX_AGE_SECONDS: dict[str, int] = {
    # One missed M1 close plus bounded cycle/serialization allowance.
    "M1": 3 * 60,
    # One missed M5 close plus one full bar of bounded cycle allowance.
    "M5": 10 * 60,
}

# A transient fast-timeframe spread quarantine must not silently become a
# full-size trade path.  This receipt only preserves a bounded M15/HTF forecast
# and geometry candidate while M1/M5 rebuild their normal 30-clean-bar tail.
# RiskEngine and the live gateway must independently revalidate the receipt
# before a later change may grant live permission.
M15_RECOVERY_MICRO_CONTRACT = "QR_M15_RECOVERY_MICRO_V1"
M15_RECOVERY_MICRO_MODE = "M15_RECOVERY_MICRO"
M15_RECOVERY_MICRO_MAX_UNITS = 999
M15_RECOVERY_SOURCE_TIMEFRAME = "M15"
M15_RECOVERY_M15_MAX_AGE_SECONDS = 30 * 60
M15_RECOVERY_FORECAST_CONTRACT = "QR_M15_RECOVERY_FORECAST_V1"
DEFAULT_FORECAST_ORIENTATION_MODEL = (
    ROOT / "config" / "forecast_orientation_model_v1.json"
)


@dataclass(frozen=True)
class DirectionalForecast:
    pair: str
    direction: str  # "UP" | "DOWN" | "RANGE" | "UNCLEAR"
    confidence: float  # 0.0-1.0 calibrated score, not a win probability
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
    range_low_price: Optional[float] = None
    range_high_price: Optional[float] = None
    range_width_pips: Optional[float] = None
    technical_context_v1: dict[str, Any] = field(default_factory=dict)
    forecast_learning_v1: dict[str, Any] = field(default_factory=dict)
    m15_recovery_receipt: dict[str, Any] = field(default_factory=dict)
    m15_recovery_forecast_evidence: dict[str, Any] = field(default_factory=dict)
    # Evaluation-only counterfactual evidence.  This field is deliberately
    # omitted from ``to_dict`` so it cannot enter intents, GPT allocation,
    # RiskEngine, or the live gateway.  The forecast persistence boundary may
    # bind it to a dedicated read-only shadow ledger instead.
    regime_family_contradiction_shadow: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
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
        if self.range_low_price is not None:
            payload["range_low_price"] = self.range_low_price
        if self.range_high_price is not None:
            payload["range_high_price"] = self.range_high_price
        if self.range_width_pips is not None:
            payload["range_width_pips"] = round(self.range_width_pips, 3)
        if self.technical_context_v1:
            payload["technical_context_v1"] = self.technical_context_v1
        if self.forecast_learning_v1:
            payload["forecast_learning_v1"] = self.forecast_learning_v1
        if self.m15_recovery_receipt:
            payload["m15_recovery_receipt"] = self.m15_recovery_receipt
        if self.m15_recovery_forecast_evidence:
            payload["m15_recovery_forecast_evidence"] = (
                self.m15_recovery_forecast_evidence
            )
        return payload


@dataclass(frozen=True)
class _RangePhase:
    phase: str
    confidence: float
    direction: str | None
    rationale: str
    evidence: tuple[str, ...]
    range_low_price: Optional[float] = None
    range_high_price: Optional[float] = None
    range_width_pips: Optional[float] = None


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


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _technical_candle_integrity_forecast_blockers(
    pair_chart: Dict[str, Any],
    *,
    expected_pair: str,
    require_technical_candle_integrity: bool,
    now_utc: datetime | None,
    _validated_evidence_out: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Fail closed on blocked or self-contradictory production MBA evidence."""

    if _validated_evidence_out is not None:
        _validated_evidence_out.clear()
    if not isinstance(require_technical_candle_integrity, bool):
        return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
    forecast_now = _aware_utc_datetime(now_utc)
    if require_technical_candle_integrity and forecast_now is None:
        return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
    generated_present = "generated_at_utc" in pair_chart
    generated_at = (
        _strict_aware_iso_datetime(pair_chart.get("generated_at_utc"))
        if generated_present
        else None
    )
    if (generated_present and generated_at is None) or (
        require_technical_candle_integrity and generated_at is None
    ):
        return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
    if require_technical_candle_integrity:
        assert forecast_now is not None and generated_at is not None
        chart_age_seconds = (forecast_now - generated_at).total_seconds()
        if (
            chart_age_seconds < -TECHNICAL_CANDLE_FORECAST_MAX_FUTURE_SKEW_SECONDS
            or chart_age_seconds > TECHNICAL_CANDLE_FORECAST_MAX_AGE_SECONDS
        ):
            return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
    raw = pair_chart.get("technical_candle_integrity")
    if raw is None:
        return (
            (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
            if require_technical_candle_integrity or generated_present
            else ()
        )
    if not isinstance(raw, dict) or raw.get("schema") != PAIR_TECHNICAL_CANDLE_INTEGRITY_SCHEMA:
        return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)

    from quant_rabbit.instruments import NORMAL_SPREAD_PIPS

    if (
        expected_pair.__class__ is not str
        or expected_pair != expected_pair.upper()
        or expected_pair not in NORMAL_SPREAD_PIPS
        or pair_chart.get("pair") != expected_pair
        or raw.get("pair") != expected_pair
    ):
        return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)

    source = raw.get("source")
    status = raw.get("evaluation_status")
    forecast_blocking = raw.get("forecast_blocking")
    codes = raw.get("codes")
    blocking_codes = raw.get("blocking_codes")
    requested_timeframes = raw.get("requested_timeframes")
    timeframes = raw.get("timeframes")
    if (
        not isinstance(source, str)
        or status not in {"PASS", "DEGRADED", "BLOCKED", "NOT_EVALUATED"}
        or not isinstance(forecast_blocking, bool)
        or not _valid_integrity_code_list(codes)
        or not _valid_integrity_code_list(blocking_codes)
        or not _is_unique_nonempty_string_list(requested_timeframes)
        or not requested_timeframes
        or any(tf not in SUPPORTED_GRANULARITIES for tf in requested_timeframes)
        or not isinstance(timeframes, dict)
        or set(timeframes) != set(requested_timeframes)
    ):
        return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
    if (
        require_technical_candle_integrity
        and not TECHNICAL_CANDLE_SPREAD_EXECUTION_TIMEFRAMES.issubset(
            set(requested_timeframes)
        )
    ):
        return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
    if (require_technical_candle_integrity or generated_present) and source != "OANDA_MBA":
        return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)

    views = pair_chart.get("views")
    if not isinstance(views, list) or len(views) != len(requested_timeframes):
        return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
    view_by_tf: dict[str, dict[str, Any]] = {}
    for view in views:
        if not isinstance(view, dict):
            return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
        tf = view.get("granularity")
        if tf not in SUPPORTED_GRANULARITIES or tf in view_by_tf:
            return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
        view_by_tf[tf] = view
    if set(view_by_tf) != set(requested_timeframes):
        return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)

    derived_sources: list[str] = []
    derived_codes: list[str] = []
    derived_blocking_codes: list[str] = []
    evaluated_count = 0
    for tf in requested_timeframes:
        item = timeframes.get(tf)
        if not isinstance(item, dict) or item.get("schema") != TECHNICAL_CANDLE_INTEGRITY_SCHEMA:
            return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
        view = view_by_tf[tf]
        if view.get("candle_integrity") != item:
            return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
        item_source = item.get("source")
        item_status = item.get("evaluation_status")
        item_blocking = item.get("forecast_blocking")
        item_codes = item.get("codes")
        item_blocking_codes = item.get("blocking_codes")
        if (
            item.get("pair") != expected_pair
            or item.get("granularity") != tf
            or item.get("payload_instrument") != expected_pair
            or item.get("payload_granularity") != tf
            or not isinstance(item_source, str)
            or item_status not in {"PASS", "DEGRADED", "BLOCKED", "NOT_EVALUATED"}
            or not isinstance(item_blocking, bool)
            or not _valid_integrity_code_list(item_codes)
            or not _valid_integrity_code_list(item_blocking_codes)
            or item_blocking != bool(item_blocking_codes)
        ):
            return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
        if (require_technical_candle_integrity or generated_present) and item_source != "OANDA_MBA":
            return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
        if item_source == "OANDA_MBA" and not _valid_mba_integrity(
            item,
            chart_generated_at=generated_at,
            view=view,
            forecast_now=(forecast_now if require_technical_candle_integrity else None),
        ):
            return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
        if item_status != "NOT_EVALUATED":
            evaluated_count += 1
        derived_sources.append(item_source)
        derived_codes.extend(item_codes)
        derived_blocking_codes.extend(item_blocking_codes)

    derived_sources = list(dict.fromkeys(derived_sources))
    expected_source = derived_sources[0] if len(derived_sources) == 1 else "MIXED"
    derived_codes = list(dict.fromkeys(derived_codes))
    derived_blocking_codes = list(dict.fromkeys(derived_blocking_codes))
    if derived_blocking_codes:
        expected_status = "BLOCKED"
    elif evaluated_count == 0:
        expected_status = "NOT_EVALUATED"
    elif derived_codes:
        expected_status = "DEGRADED"
    else:
        expected_status = "PASS"
    aggregate_evaluated_count = raw.get("evaluated_timeframe_count")
    if (
        not _is_exact_nonnegative_int(aggregate_evaluated_count)
        or source != expected_source
        or status != expected_status
        or aggregate_evaluated_count != evaluated_count
        or list(codes) != derived_codes
        or list(blocking_codes) != derived_blocking_codes
        or forecast_blocking != bool(derived_blocking_codes)
    ):
        return (TECHNICAL_CANDLE_PROVENANCE_INVALID,)
    if _validated_evidence_out is not None:
        _validated_evidence_out.update(
            {
                "generated_at": generated_at,
                "aggregate": raw,
                "requested_timeframes": tuple(requested_timeframes),
                "timeframes": timeframes,
                "view_by_tf": view_by_tf,
            }
        )
    return tuple(blocking_codes) if forecast_blocking else ()


def _m15_recovery_atr_pips(view: dict[str, Any]) -> float | None:
    indicators = view.get("indicators")
    if not isinstance(indicators, dict) or indicators.get("valid") is not True:
        return None
    atr_pips = _strict_finite_number(indicators.get("atr_pips"))
    if atr_pips is None:
        atr = _strict_finite_number(indicators.get("atr_14") or indicators.get("atr"))
        pip_size = _strict_finite_number(indicators.get("pip_size"))
        if atr is not None and pip_size is not None and pip_size > 0.0:
            atr_pips = atr / pip_size
    return atr_pips if atr_pips is not None and atr_pips > 0.0 else None


def _m15_recovery_receipt_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_m15_recovery_micro_receipt(
    pair_chart: Dict[str, Any],
    *,
    expected_pair: str,
    now_utc: datetime | None,
    current_spread_pips: object,
    receipt_validated_at: datetime,
) -> dict[str, Any] | None:
    """Build a content-addressed, non-live M15 recovery candidate receipt."""

    forecast_now = _aware_utc_datetime(now_utc)
    validated_at = _aware_utc_datetime(receipt_validated_at)
    if forecast_now is None or validated_at is None:
        return None
    evidence: dict[str, Any] = {}
    blockers = _technical_candle_integrity_forecast_blockers(
        pair_chart,
        expected_pair=expected_pair,
        require_technical_candle_integrity=True,
        now_utc=forecast_now,
        _validated_evidence_out=evidence,
    )
    # The only admissible aggregate block is historical fast-timeframe spread
    # quarantine plus an incomplete clean-tail rebuild. Any structural receipt
    # failure leaves `evidence` empty and therefore cannot enter this branch.
    if blockers != (TECHNICAL_CANDLE_PROVENANCE_INVALID,) or not evidence:
        return None
    generated_at = evidence.get("generated_at")
    if not isinstance(generated_at, datetime):
        return None
    validation_age = (forecast_now - validated_at).total_seconds()
    if (
        validated_at < generated_at
        or validation_age < -TECHNICAL_CANDLE_FORECAST_MAX_FUTURE_SKEW_SECONDS
        or validation_age > TECHNICAL_CANDLE_FORECAST_MAX_AGE_SECONDS
    ):
        return None

    timeframes = evidence["timeframes"]
    view_by_tf = evidence["view_by_tf"]
    fast_receipts: dict[str, dict[str, Any]] = {}
    spread_caps: list[float] = []
    recovering_fast_timeframes = 0
    for tf in sorted(TECHNICAL_CANDLE_SPREAD_EXECUTION_TIMEFRAMES):
        item = timeframes.get(tf)
        if not isinstance(item, dict):
            return None
        cap = _strict_finite_number(item.get("spread_cap_pips"))
        tail = item.get("recent_clean_tail_count")
        common_valid = bool(
            item.get("recent_tail_state") == "CLEAN"
            and item.get("malformed_count") == 0
            and item.get("coverage_complete") is True
            and item.get("provenance_complete") is True
            and item.get("indicator_warmup_complete") is True
            and _is_exact_nonnegative_int(tail)
            and item.get("latest_complete_timestamp_utc")
            == item.get("latest_clean_timestamp_utc")
            and cap is not None
            and cap > 0.0
        )
        recovering = bool(
            common_valid
            and item.get("evaluation_status") == "BLOCKED"
            and item.get("forecast_blocking") is True
            and item.get("codes")
            == [
                TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
                TECHNICAL_CANDLE_PROVENANCE_INVALID,
            ]
            and item.get("blocking_codes")
            == [TECHNICAL_CANDLE_PROVENANCE_INVALID]
            and item.get("contaminated_count", 0) > 0
            and item.get("recent_clean_coverage_complete") is False
            and 1 <= tail < TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT
        )
        recovered_codes = item.get("codes")
        recovered_status = item.get("evaluation_status")
        recovered_contaminated = item.get("contaminated_count", 0)
        recovered = bool(
            common_valid
            and item.get("forecast_blocking") is False
            and item.get("blocking_codes") == []
            and item.get("recent_clean_coverage_complete") is True
            and tail >= TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT
            and (
                (
                    recovered_status == "PASS"
                    and recovered_codes == []
                    and recovered_contaminated == 0
                )
                or (
                    recovered_status == "DEGRADED"
                    and recovered_codes == [TECHNICAL_CANDLE_SPREAD_CONTAMINATED]
                    and recovered_contaminated > 0
                )
            )
        )
        if not recovering and not recovered:
            return None
        if recovering:
            recovering_fast_timeframes += 1
        spread_caps.append(cap)
        fast_receipts[tf] = {
            "recovery_state": "RECOVERING" if recovering else "RECOVERED",
            "evaluation_status": item["evaluation_status"],
            "forecast_blocking": item["forecast_blocking"],
            "codes": list(item["codes"]),
            "blocking_codes": list(item["blocking_codes"]),
            "latest_clean_timestamp_utc": item["latest_clean_timestamp_utc"],
            "recent_clean_tail_count": tail,
            "required_clean_tail_count": TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT,
            "contaminated_count": item["contaminated_count"],
            "malformed_count": 0,
            "spread_cap_pips": cap,
        }
    # The normal forecast path owns the fully recovered state.  This receipt
    # exists only while at least one fast timeframe is still rebuilding its
    # clean tail; the other may already have crossed the normal threshold.
    if recovering_fast_timeframes == 0:
        return None

    # Every operating/higher timeframe must remain structurally valid and
    # non-blocking. M15 is additionally the sole recovery ATR/geometry source.
    for tf in evidence["requested_timeframes"]:
        if tf in TECHNICAL_CANDLE_SPREAD_EXECUTION_TIMEFRAMES:
            continue
        item = timeframes.get(tf)
        if (
            not isinstance(item, dict)
            or item.get("forecast_blocking") is not False
            or item.get("blocking_codes") != []
        ):
            return None
    m15_item = timeframes.get(M15_RECOVERY_SOURCE_TIMEFRAME)
    m15_view = view_by_tf.get(M15_RECOVERY_SOURCE_TIMEFRAME)
    if not isinstance(m15_item, dict) or not isinstance(m15_view, dict):
        return None
    m15_latest = _strict_aware_iso_datetime(
        m15_item.get("latest_complete_timestamp_utc")
    )
    m15_atr_pips = _m15_recovery_atr_pips(m15_view)
    m15_candles = (m15_view.get("indicators") or {}).get("candles_count")
    if (
        m15_item.get("evaluation_status") != "PASS"
        or m15_item.get("codes") != []
        or m15_item.get("malformed_count") != 0
        or m15_item.get("recent_tail_state") != "CLEAN"
        or m15_item.get("latest_complete_timestamp_utc")
        != m15_item.get("latest_clean_timestamp_utc")
        or m15_latest is None
        or (forecast_now - m15_latest).total_seconds()
        < -TECHNICAL_CANDLE_FORECAST_MAX_FUTURE_SKEW_SECONDS
        or (forecast_now - m15_latest).total_seconds()
        > M15_RECOVERY_M15_MAX_AGE_SECONDS
        or not _is_exact_nonnegative_int(m15_candles)
        or m15_candles < TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT
        or m15_atr_pips is None
    ):
        return None

    current_spread = _strict_finite_number(current_spread_pips)
    spread_cap = min(spread_caps) if spread_caps else None
    if (
        current_spread is None
        or current_spread <= 0.0
        or spread_cap is None
        or current_spread > spread_cap
    ):
        return None
    body: dict[str, Any] = {
        "contract": M15_RECOVERY_MICRO_CONTRACT,
        "mode": M15_RECOVERY_MICRO_MODE,
        "status": "ELIGIBLE_FOR_MICRO_REVALIDATION",
        "pair": expected_pair,
        "chart_generated_at_utc": generated_at.isoformat(),
        "m15_plus_scoring_input_sha256": (
            m15_recovery_scoring_input_sha256(pair_chart)
        ),
        "validated_at_utc": validated_at.isoformat(),
        "fast_timeframe_receipts": fast_receipts,
        "geometry_source": {
            "timeframe": M15_RECOVERY_SOURCE_TIMEFRAME,
            "latest_clean_timestamp_utc": m15_item[
                "latest_clean_timestamp_utc"
            ],
            "atr_pips": m15_atr_pips,
            "candles_count": m15_candles,
        },
        "current_spread_pips": current_spread,
        "spread_cap_pips": spread_cap,
        "sizing": {
            "max_units": M15_RECOVERY_MICRO_MAX_UNITS,
            "full_size_allowed": False,
            "minimum_units_override": False,
        },
        "live_permission": False,
        "requires_risk_gateway_revalidation": True,
        "manual_position_mutation_allowed": False,
    }
    return {**body, "receipt_sha256": _m15_recovery_receipt_hash(body)}


def build_m15_recovery_micro_receipt(
    pair_chart: Dict[str, Any],
    *,
    expected_pair: str,
    now_utc: datetime | None,
    current_spread_pips: object,
) -> dict[str, Any] | None:
    """Return a bounded recovery candidate, never a live permission."""

    validated_at = _aware_utc_datetime(now_utc)
    if validated_at is None:
        return None
    return _build_m15_recovery_micro_receipt(
        pair_chart,
        expected_pair=expected_pair,
        now_utc=validated_at,
        current_spread_pips=current_spread_pips,
        receipt_validated_at=validated_at,
    )


def validate_m15_recovery_micro_receipt(
    receipt: object,
    *,
    pair_chart: Dict[str, Any],
    expected_pair: str,
    now_utc: datetime | None,
    current_spread_pips: object,
) -> bool:
    """Recompute the receipt from source evidence and recheck current spread."""

    if not isinstance(receipt, dict):
        return False
    receipt_body = dict(receipt)
    digest = receipt_body.pop("receipt_sha256", None)
    if (
        digest.__class__ is not str
        or len(digest) != 64
        or _m15_recovery_receipt_hash(receipt_body) != digest
    ):
        return False
    validated_at = _strict_aware_iso_datetime(receipt.get("validated_at_utc"))
    receipt_spread = _strict_finite_number(receipt.get("current_spread_pips"))
    current_spread = _strict_finite_number(current_spread_pips)
    cap = _strict_finite_number(receipt.get("spread_cap_pips"))
    if (
        validated_at is None
        or receipt_spread is None
        or current_spread is None
        or current_spread <= 0.0
        or cap is None
        or current_spread > cap
    ):
        return False
    expected = _build_m15_recovery_micro_receipt(
        pair_chart,
        expected_pair=expected_pair,
        now_utc=now_utc,
        current_spread_pips=receipt_spread,
        receipt_validated_at=validated_at,
    )
    return expected == receipt


def _m15_recovery_forecast_chart(pair_chart: Dict[str, Any]) -> Dict[str, Any]:
    """Remove fast-timeframe and mixed-TF aggregates from recovery scoring."""

    allowed = {"M15", "M30", "H1", "H4", "D"}
    chart = dict(pair_chart)
    chart["views"] = [
        view
        for view in pair_chart.get("views") or []
        if isinstance(view, dict)
        and str(view.get("granularity") or "").upper() in allowed
    ]
    # `confluence` and the top-level scores aggregate M1/M5 evidence. Keep
    # them out of a contract that explicitly claims M15+ forecast evidence.
    chart["confluence"] = {}
    chart["long_score"] = 0.0
    chart["short_score"] = 0.0
    return chart


def m15_recovery_scoring_input_sha256(pair_chart: Dict[str, Any]) -> str:
    """Bind every chart field visible to the M15+ recovery forecaster.

    The ordinary candle-integrity receipt proves candle provenance and tail
    state, but does not content-address derived regime/family/structure fields.
    Hash the complete sanitized forecast input so Risk/Gateway revalidation
    detects any change to those scoring values as well as the root clock.
    """

    return _m15_recovery_receipt_hash(
        _m15_recovery_forecast_chart(pair_chart)
    )


def _m15_or_higher_signal(signal: object) -> bool:
    """Accept only detector evidence with explicit M15+ provenance."""

    allowed = {"M15", "M30", "H1", "H4", "D"}
    for key in ("timeframe", "granularity", "source_timeframe"):
        value = signal.get(key) if isinstance(signal, dict) else getattr(signal, key, None)
        if value is not None:
            return value.__class__ is str and value.upper() in allowed
    values = signal.get("timeframes") if isinstance(signal, dict) else getattr(signal, "timeframes", None)
    return (
        isinstance(values, (list, tuple))
        and bool(values)
        and all(value.__class__ is str and value.upper() in allowed for value in values)
    )


def m15_or_higher_signal(signal: object) -> bool:
    """Public contract helper shared by intent support/telemetry filtering."""

    return _m15_or_higher_signal(signal)


def m15_recovery_calibration_regime(
    regime: str | None,
    recovery_receipt: object,
) -> str | None:
    """Exclude mixed-fast confluence from every recovery calibration read."""

    return None if isinstance(recovery_receipt, dict) and recovery_receipt else regime


def _m15_recovery_signal_material(signal: object) -> dict[str, Any]:
    """Freeze only bounded forecast inputs with explicit M15+ provenance."""

    def value(key: str) -> object:
        return signal.get(key) if isinstance(signal, dict) else getattr(signal, key, None)

    material: dict[str, Any] = {
        "type": type(signal).__name__,
    }
    for key in (
        "name",
        "direction",
        "timeframe",
        "granularity",
        "source_timeframe",
        "rationale",
    ):
        raw = value(key)
        if raw.__class__ is str:
            material[key] = raw[:512]
    raw_timeframes = value("timeframes")
    if isinstance(raw_timeframes, (list, tuple)):
        material["timeframes"] = [
            raw for raw in raw_timeframes if raw.__class__ is str
        ][:8]
    for key in (
        "bonus",
        "bonus_magnitude",
        "confidence",
        "lead_time_min",
        "probability",
        "target_price",
        "invalidation_price",
    ):
        raw = _strict_finite_number(value(key))
        if raw is not None:
            material[key] = raw
    return material


def _m15_recovery_filtered_inputs_sha256(
    *,
    pattern_signals: list[Any],
    projection_signals: list[Any],
    correlation_signals: list[Any],
    paths: list[Any],
    reversal_long: object,
    reversal_short: object,
) -> str:
    payload = {
        "pattern_signals": [
            _m15_recovery_signal_material(signal) for signal in pattern_signals
        ],
        "projection_signals": [
            _m15_recovery_signal_material(signal) for signal in projection_signals
        ],
        "correlation_signals": [
            _m15_recovery_signal_material(signal) for signal in correlation_signals
        ],
        "paths": [_m15_recovery_signal_material(signal) for signal in paths],
        "reversal_long": (
            _m15_recovery_signal_material(reversal_long)
            if reversal_long is not None
            else None
        ),
        "reversal_short": (
            _m15_recovery_signal_material(reversal_short)
            if reversal_short is not None
            else None
        ),
    }
    return _m15_recovery_receipt_hash(payload)


def _m15_recovery_forecast_evidence(
    *,
    receipt: dict[str, Any],
    pair: str,
    current_price: float,
    spread_pips: float | None,
    filtered_input_sha256: str,
    direction: str,
    confidence: float,
    raw_confidence: float,
    calibration_multiplier: float,
    component_scores: dict[str, float],
    target_price: float | None,
    invalidation_price: float | None,
    horizon_min: int,
) -> dict[str, Any]:
    body = {
        "contract": M15_RECOVERY_FORECAST_CONTRACT,
        "source_recovery_receipt_sha256": receipt.get("receipt_sha256"),
        "pair": pair,
        "chart_generated_at_utc": receipt.get("chart_generated_at_utc"),
        "forecast_current_price": current_price,
        "forecast_spread_pips": spread_pips,
        "filtered_input_sha256": filtered_input_sha256,
        "raw_winner": direction,
        "component_scores": {
            key: float(value) for key, value in sorted(component_scores.items())
        },
        "final_direction": direction,
        "raw_confidence": raw_confidence,
        "calibration_multiplier": calibration_multiplier,
        "calibration_scope": "M15_RECOVERY_CONSERVATIVE_DIRECTIONAL_PRIOR",
        "confidence": confidence,
        "target_price": target_price,
        "invalidation_price": invalidation_price,
        "horizon_min": horizon_min,
        "geometry_source_timeframe": M15_RECOVERY_SOURCE_TIMEFRAME,
        "live_permission": False,
    }
    return {**body, "evidence_sha256": _m15_recovery_receipt_hash(body)}


def _is_unique_nonempty_string_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and item for item in value)
        and len(value) == len(set(value))
    )


def _valid_integrity_code_list(value: object) -> bool:
    allowed = {
        TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
        TECHNICAL_CANDLE_PROVENANCE_INVALID,
    }
    return (
        _is_unique_nonempty_string_list(value)
        and all(code in allowed for code in value)
    ) or value == []


def _valid_mba_integrity(
    item: dict[str, Any],
    *,
    chart_generated_at: datetime | None,
    view: dict[str, Any],
    forecast_now: datetime | None,
) -> bool:
    granularity = item.get("granularity")
    expected_spread_mode = (
        TECHNICAL_CANDLE_SPREAD_EXECUTION_MODE
        if granularity in TECHNICAL_CANDLE_SPREAD_EXECUTION_TIMEFRAMES
        else TECHNICAL_CANDLE_SPREAD_PROVENANCE_ONLY_MODE
    )
    if item.get("spread_evaluation_mode") != expected_spread_mode:
        return False
    counts: dict[str, int] = {}
    for key in (
        "requested_count",
        "raw_entry_count",
        "complete_entry_count",
        "clean_count",
        "recent_clean_tail_count",
        "contaminated_count",
        "malformed_count",
        "quarantined_count",
    ):
        value = item.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return False
        counts[key] = value
    coverage_complete = (
        counts["requested_count"] > 0
        and counts["raw_entry_count"] == counts["requested_count"]
    )
    if (
        not coverage_complete
        or item.get("coverage_complete") is not coverage_complete
    ):
        return False
    if (
        counts["quarantined_count"] != counts["contaminated_count"] + counts["malformed_count"]
        or counts["clean_count"] + counts["contaminated_count"] > counts["complete_entry_count"]
        or counts["recent_clean_tail_count"] > counts["clean_count"]
        or counts["complete_entry_count"] > counts["raw_entry_count"]
        or (
            counts["malformed_count"] == 0
            and counts["clean_count"] + counts["contaminated_count"]
            != counts["complete_entry_count"]
        )
    ):
        return False
    indicator_warmup_min_clean_count = (
        TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT
        if granularity in TECHNICAL_CANDLE_SPREAD_EXECUTION_TIMEFRAMES
        else 1
    )
    warmup_complete = counts["clean_count"] >= indicator_warmup_min_clean_count
    recent_clean_coverage_complete = (
        counts["recent_clean_tail_count"]
        >= indicator_warmup_min_clean_count
    )
    if (
        item.get("indicator_warmup_min_clean_count")
        != indicator_warmup_min_clean_count
        or item.get("indicator_warmup_complete") is not warmup_complete
        or item.get("recent_clean_coverage_complete")
        is not recent_clean_coverage_complete
        or (
            counts["contaminated_count"] == 0
            and counts["malformed_count"] == 0
            and counts["recent_clean_tail_count"] != counts["clean_count"]
        )
    ):
        return False
    if (
        expected_spread_mode == TECHNICAL_CANDLE_SPREAD_PROVENANCE_ONLY_MODE
        and counts["contaminated_count"] != 0
    ):
        return False

    normal = _strict_finite_number(item.get("normal_spread_pips"))
    multiple = _strict_finite_number(item.get("max_spread_multiple"))
    execution_cap = _strict_finite_number(item.get("execution_spread_cap_pips"))
    cap = _strict_finite_number(item.get("spread_cap_pips"))
    if (
        item.get("requested_price") != "MBA"
        or item.get("policy_source")
        != "OANDA_SPREAD_CALIBRATION_V1.pairs.max_pips"
        or normal is None
        or multiple is None
        or execution_cap is None
        or cap is None
        or not all(
            value > 0.0 for value in (normal, multiple, execution_cap, cap)
        )
    ):
        return False
    from quant_rabbit.instruments import (
        NORMAL_SPREAD_PIPS,
        OANDA_SPREAD_CALIBRATION_V1,
    )
    from quant_rabbit.risk import RiskPolicy

    canonical_normal = _strict_finite_number(NORMAL_SPREAD_PIPS.get(item.get("pair")))
    canonical_multiple = _strict_finite_number(RiskPolicy().max_spread_multiple)
    if canonical_normal is None or canonical_multiple is None:
        return False
    calibration = OANDA_SPREAD_CALIBRATION_V1.pairs.get(str(item.get("pair") or ""))
    calibration_sha256 = str(item.get("spread_calibration_sha256") or "")
    canonical_execution_cap = round(canonical_normal * canonical_multiple, 6)
    canonical_cap = (
        round(float(calibration.max_pips), 6)
        if calibration is not None
        else None
    )
    if (
        normal != canonical_normal
        or multiple != canonical_multiple
        or execution_cap != canonical_execution_cap
        or canonical_cap is None
        or cap != canonical_cap
        or calibration_sha256 != OANDA_SPREAD_CALIBRATION_V1.calibration_sha256
    ):
        return False

    state = item.get("recent_tail_state")
    if state not in {"CLEAN", "SPREAD_CONTAMINATED", "PROVENANCE_INVALID"}:
        return False
    if item.get("recent_tail_contaminated") is not (state == "SPREAD_CONTAMINATED"):
        return False
    if item.get("recent_tail_invalid") is not (state == "PROVENANCE_INVALID"):
        return False
    if (
        (state == "CLEAN" and counts["recent_clean_tail_count"] <= 0)
        or (state == "SPREAD_CONTAMINATED" and counts["recent_clean_tail_count"] != 0)
    ):
        return False
    provenance_complete = counts["malformed_count"] == 0 and counts["complete_entry_count"] > 0
    if item.get("provenance_complete") is not provenance_complete:
        return False

    latest_complete_raw = item.get("latest_complete_timestamp_utc")
    latest_clean_raw = item.get("latest_clean_timestamp_utc")
    latest_complete = _strict_aware_iso_datetime(latest_complete_raw)
    latest_clean = _strict_aware_iso_datetime(latest_clean_raw)
    if counts["complete_entry_count"] > 0:
        if latest_complete is None:
            return False
    elif latest_complete_raw is not None:
        return False
    if counts["clean_count"] > 0:
        if latest_clean is None:
            return False
    elif latest_clean_raw is not None:
        return False
    if chart_generated_at is not None and (
        (latest_complete is not None and latest_complete > chart_generated_at)
        or (latest_clean is not None and latest_clean > chart_generated_at)
    ):
        return False
    latest_complete_max_age = TECHNICAL_CANDLE_LATEST_COMPLETE_MAX_AGE_SECONDS.get(
        str(granularity)
    )
    if forecast_now is not None and latest_complete_max_age is not None:
        if latest_complete is None:
            return False
        latest_complete_age_seconds = (forecast_now - latest_complete).total_seconds()
        if (
            latest_complete_age_seconds
            < -TECHNICAL_CANDLE_FORECAST_MAX_FUTURE_SKEW_SECONDS
            or latest_complete_age_seconds > latest_complete_max_age
        ):
            return False
    if state == "CLEAN" and latest_complete != latest_clean:
        return False
    if state == "SPREAD_CONTAMINATED":
        if latest_complete is None:
            return False
        if counts["clean_count"] > 0:
            if latest_clean is None or latest_clean >= latest_complete:
                return False
        elif latest_clean is not None:
            return False

    indicators = view.get("indicators")
    if not isinstance(indicators, dict):
        return False
    view_candle_count = indicators.get("candles_count")
    if (
        not _is_exact_nonnegative_int(view_candle_count)
        or view_candle_count != counts["recent_clean_tail_count"]
    ):
        return False
    recent_candles = view.get("recent_candles")
    if not isinstance(recent_candles, list):
        return False
    first_recent: datetime | None = None
    if counts["recent_clean_tail_count"] > 0:
        expected_recent_count = min(
            counts["recent_clean_tail_count"],
            TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT,
        )
        if (
            len(recent_candles) != expected_recent_count
        ):
            return False
        prior_recent: datetime | None = None
        for recent in recent_candles:
            if not isinstance(recent, dict):
                return False
            recent_time = _strict_aware_iso_datetime(recent.get("t"))
            if recent_time is None or not _technical_candle_cadence_valid(
                granularity=str(granularity),
                timestamp=recent_time,
                prior_timestamp=prior_recent,
            ):
                return False
            if first_recent is None:
                first_recent = recent_time
            prior_recent = recent_time
        recent_tail_raw = recent_candles[-1].get("t")
        if (
            recent_tail_raw != latest_clean_raw
            or _strict_aware_iso_datetime(recent_tail_raw) != latest_clean
        ):
            return False
    elif recent_candles:
        return False

    quarantine_details = item.get("quarantine_details")
    truncated = item.get("quarantine_details_truncated")
    quarantine_window = item.get("quarantine_details_window")
    quarantine_codes = (
        TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
        TECHNICAL_CANDLE_PROVENANCE_INVALID,
    )
    expected_total_code_counts = {
        TECHNICAL_CANDLE_SPREAD_CONTAMINATED: counts["contaminated_count"],
        TECHNICAL_CANDLE_PROVENANCE_INVALID: counts["malformed_count"],
    }
    expected_window_start = max(
        0,
        counts["quarantined_count"] - TECHNICAL_CANDLE_QUARANTINE_DETAIL_LIMIT,
    )
    expected_published_count = min(
        counts["quarantined_count"],
        TECHNICAL_CANDLE_QUARANTINE_DETAIL_LIMIT,
    )
    if (
        not isinstance(quarantine_details, list)
        or len(quarantine_details) != expected_published_count
        or not _is_exact_nonnegative_int(truncated)
        or truncated != expected_window_start
        or not isinstance(quarantine_window, dict)
        or set(quarantine_window) != {
            "selection",
            "order",
            "limit",
            "start_index",
            "end_index_exclusive",
            "total_count",
            "total_code_counts",
            "published_code_counts",
            "omitted_code_counts",
            "total_timestamped_count",
            "published_timestamped_count",
            "omitted_timestamped_count",
            "latest_timestamp_utc",
        }
        or quarantine_window.get("selection")
        != TECHNICAL_CANDLE_QUARANTINE_DETAILS_SELECTION
        or quarantine_window.get("order") != TECHNICAL_CANDLE_QUARANTINE_DETAILS_ORDER
        or not _is_exact_nonnegative_int(quarantine_window.get("limit"))
        or quarantine_window.get("limit") != TECHNICAL_CANDLE_QUARANTINE_DETAIL_LIMIT
        or not _is_exact_nonnegative_int(quarantine_window.get("start_index"))
        or quarantine_window.get("start_index") != expected_window_start
        or not _is_exact_nonnegative_int(
            quarantine_window.get("end_index_exclusive")
        )
        or quarantine_window.get("end_index_exclusive") != counts["quarantined_count"]
        or not _is_exact_nonnegative_int(quarantine_window.get("total_count"))
        or quarantine_window.get("total_count") != counts["quarantined_count"]
    ):
        return False
    total_code_counts_raw = quarantine_window.get("total_code_counts")
    published_code_counts_raw = quarantine_window.get("published_code_counts")
    omitted_code_counts_raw = quarantine_window.get("omitted_code_counts")
    total_timestamped_count = quarantine_window.get("total_timestamped_count")
    published_timestamped_count = quarantine_window.get(
        "published_timestamped_count"
    )
    omitted_timestamped_count = quarantine_window.get("omitted_timestamped_count")
    if (
        not isinstance(total_code_counts_raw, dict)
        or set(total_code_counts_raw) != set(quarantine_codes)
        or not isinstance(published_code_counts_raw, dict)
        or set(published_code_counts_raw) != set(quarantine_codes)
        or not isinstance(omitted_code_counts_raw, dict)
        or set(omitted_code_counts_raw) != set(quarantine_codes)
        or any(
            not _is_exact_nonnegative_int(total_code_counts_raw.get(code))
            or total_code_counts_raw.get(code) != expected_total_code_counts[code]
            or not _is_exact_nonnegative_int(published_code_counts_raw.get(code))
            or not _is_exact_nonnegative_int(omitted_code_counts_raw.get(code))
            for code in quarantine_codes
        )
        or sum(published_code_counts_raw.values()) != expected_published_count
        or sum(omitted_code_counts_raw.values()) != truncated
        or any(
            published_code_counts_raw[code] + omitted_code_counts_raw[code]
            != total_code_counts_raw[code]
            for code in quarantine_codes
        )
        or not _is_exact_nonnegative_int(total_timestamped_count)
        or not _is_exact_nonnegative_int(published_timestamped_count)
        or not _is_exact_nonnegative_int(omitted_timestamped_count)
        or total_timestamped_count > counts["quarantined_count"]
        or total_timestamped_count < counts["contaminated_count"]
        or published_timestamped_count
        < published_code_counts_raw[TECHNICAL_CANDLE_SPREAD_CONTAMINATED]
        or omitted_timestamped_count
        < omitted_code_counts_raw[TECHNICAL_CANDLE_SPREAD_CONTAMINATED]
        or published_timestamped_count
        != min(total_timestamped_count, expected_published_count)
        or omitted_timestamped_count
        != total_timestamped_count - published_timestamped_count
    ):
        return False
    parsed_quarantine: list[tuple[str, datetime | None]] = []
    actual_published_code_counts = dict.fromkeys(quarantine_codes, 0)
    prior_quarantine_time: datetime | None = None
    timestamped_quarantine_seen = False
    quarantine_detail_identities: set[tuple[tuple[str, str], ...]] = set()
    quarantine_detail_sort_keys: list[tuple[object, ...]] = []
    minimum_quarantine_time = datetime.min.replace(tzinfo=timezone.utc)
    for detail in quarantine_details:
        if not isinstance(detail, dict):
            return False
        code = detail.get("code")
        if code not in {
            TECHNICAL_CANDLE_SPREAD_CONTAMINATED,
            TECHNICAL_CANDLE_PROVENANCE_INVALID,
        }:
            return False
        expected_detail_fields = (
            {
                "timestamp_utc",
                "code",
                "max_spread_pips",
                "spread_cap_pips",
            }
            if code == TECHNICAL_CANDLE_SPREAD_CONTAMINATED
            else {"timestamp_utc", "code", "reason"}
        )
        if set(detail) != expected_detail_fields:
            return False
        detail_identity = tuple(
            (key, repr(detail[key])) for key in sorted(detail)
        )
        if detail_identity in quarantine_detail_identities:
            return False
        quarantine_detail_identities.add(detail_identity)
        actual_published_code_counts[code] += 1
        if code == TECHNICAL_CANDLE_SPREAD_CONTAMINATED:
            detail_max_spread = _strict_finite_number(detail.get("max_spread_pips"))
            detail_spread_cap = _strict_finite_number(detail.get("spread_cap_pips"))
            if (
                detail_max_spread is None
                or detail_spread_cap is None
                or detail_spread_cap != cap
                or detail_max_spread <= detail_spread_cap
            ):
                return False
        else:
            reason = detail.get("reason")
            if not isinstance(reason, str) or not reason or len(reason) > 256:
                return False
        detail_time_raw = detail.get("timestamp_utc")
        detail_time = (
            _strict_aware_iso_datetime(detail_time_raw)
            if detail_time_raw is not None
            else None
        )
        if detail_time_raw is not None and detail_time is None:
            return False
        if code == TECHNICAL_CANDLE_SPREAD_CONTAMINATED and detail_time is None:
            return False
        if (
            code == TECHNICAL_CANDLE_SPREAD_CONTAMINATED
            and detail_time_raw != detail_time.isoformat()
        ):
            return False
        if detail_time is None:
            if timestamped_quarantine_seen:
                return False
        else:
            if prior_quarantine_time is not None and (
                detail_time < prior_quarantine_time
                or (
                    counts["malformed_count"] == 0
                    and detail_time == prior_quarantine_time
                )
            ):
                return False
            timestamped_quarantine_seen = True
            prior_quarantine_time = detail_time
        if detail_time is not None and chart_generated_at is not None and detail_time > chart_generated_at:
            return False
        quarantine_detail_sort_keys.append(
            (
                0 if detail_time is None else 1,
                detail_time or minimum_quarantine_time,
                detail_identity,
            )
        )
        parsed_quarantine.append((code, detail_time))
    if quarantine_detail_sort_keys != sorted(quarantine_detail_sort_keys):
        return False
    if actual_published_code_counts != published_code_counts_raw:
        return False
    if (
        sum(timestamp is not None for _, timestamp in parsed_quarantine)
        != published_timestamped_count
    ):
        return False
    latest_quarantine_raw = quarantine_window.get("latest_timestamp_utc")
    latest_quarantine = (
        _strict_aware_iso_datetime(latest_quarantine_raw)
        if latest_quarantine_raw is not None
        else None
    )
    if (
        (latest_quarantine_raw is not None and latest_quarantine is None)
        or (
            latest_quarantine is not None
            and latest_quarantine_raw != latest_quarantine.isoformat()
        )
        or latest_quarantine != prior_quarantine_time
        or (
            first_recent is not None
            and any(
                timestamp is not None and timestamp >= first_recent
                for _, timestamp in parsed_quarantine
            )
        )
    ):
        return False
    if state == "SPREAD_CONTAMINATED":
        if (
            not parsed_quarantine
            or parsed_quarantine[-1][0] != TECHNICAL_CANDLE_SPREAD_CONTAMINATED
            or parsed_quarantine[-1][1] != latest_complete
        ):
            return False
    if state == "CLEAN" and counts["contaminated_count"] > 0 and counts["malformed_count"] == 0:
        if (
            not parsed_quarantine
            or parsed_quarantine[-1][0] != TECHNICAL_CANDLE_SPREAD_CONTAMINATED
            or parsed_quarantine[-1][1] is None
            or latest_complete is None
            or parsed_quarantine[-1][1] >= latest_complete
        ):
            return False

    expected_codes: list[str] = []
    expected_blocking_codes: list[str] = []
    clean_coverage_blocked = (
        state == "CLEAN"
        and (not warmup_complete or not recent_clean_coverage_complete)
    )
    if counts["contaminated_count"] > 0:
        expected_codes.append(TECHNICAL_CANDLE_SPREAD_CONTAMINATED)
    if (
        counts["malformed_count"] > 0
        or counts["complete_entry_count"] <= 0
        or counts["clean_count"] <= 0
        or clean_coverage_blocked
    ):
        expected_codes.append(TECHNICAL_CANDLE_PROVENANCE_INVALID)
    if state == "SPREAD_CONTAMINATED":
        expected_blocking_codes.append(TECHNICAL_CANDLE_SPREAD_CONTAMINATED)
    if (
        counts["malformed_count"] > 0
        or state == "PROVENANCE_INVALID"
        or counts["complete_entry_count"] <= 0
        or counts["clean_count"] <= 0
        or clean_coverage_blocked
    ):
        expected_blocking_codes.append(TECHNICAL_CANDLE_PROVENANCE_INVALID)
    expected_status = "BLOCKED" if expected_blocking_codes else "DEGRADED" if expected_codes else "PASS"
    return (
        item.get("codes") == expected_codes
        and item.get("blocking_codes") == expected_blocking_codes
        and item.get("forecast_blocking") is bool(expected_blocking_codes)
        and item.get("evaluation_status") == expected_status
    )


def validate_mba_integrity_receipt(
    item: dict[str, Any],
    *,
    chart_generated_at: datetime | None,
    view: dict[str, Any],
    now_utc: datetime | None,
) -> bool:
    """Validate one producer-owned OANDA MBA integrity receipt exactly.

    Fast monitoring needs to distinguish a current, self-consistent
    pair-local quarantine from a malformed chart packet.  Reuse the same
    validator as the directional forecaster so a wrapper cannot weaken the
    count, provenance, cadence, spread-policy, or bounded-quarantine contract.
    """

    return _valid_mba_integrity(
        item,
        chart_generated_at=chart_generated_at,
        view=view,
        forecast_now=_aware_utc_datetime(now_utc),
    )


def _strict_finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _is_exact_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _strict_aware_iso_datetime(value: object) -> datetime | None:
    if value.__class__ is not str or not value or value != value.strip() or "T" not in value:
        return None
    text = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _aware_utc_datetime(value: object) -> datetime | None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


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


def _forecast_noise_floor_pips(
    pair_chart: Dict[str, Any],
    *,
    source_timeframes: frozenset[str] = frozenset({"M1", "M5"}),
) -> Optional[float]:
    floors: list[float] = []
    for view in pair_chart.get("views") or []:
        if not isinstance(view, dict):
            continue
        tf = str(view.get("granularity") or "").upper()
        if tf not in source_timeframes:
            continue
        indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else {}
        atr_pips = _to_float((indicators or {}).get("atr_pips"))
        if atr_pips is None:
            atr = _to_float((indicators or {}).get("atr_14") or (indicators or {}).get("atr"))
            pip_size = _to_float((indicators or {}).get("pip_size"))
            if atr is not None and pip_size is not None and pip_size > 0:
                atr_pips = atr / pip_size
        if atr_pips is not None and atr_pips > 0:
            floors.append(atr_pips * FORECAST_GEOMETRY_NOISE_ATR_MULT)
    return max(floors) if floors else None


def _clears_noise(
    price: Optional[float],
    *,
    current_price: float,
    pip_factor: float,
    noise_floor_pips: Optional[float],
) -> bool:
    if price is None:
        return False
    if noise_floor_pips is None:
        return True
    return abs(price - current_price) * pip_factor >= noise_floor_pips


def _nearest_level(
    prices: list[float],
    *,
    above: Optional[float] = None,
    below: Optional[float] = None,
    pip_factor: Optional[float] = None,
    noise_floor_pips: Optional[float] = None,
) -> Optional[float]:
    if above is not None:
        candidates = [
            p
            for p in prices
            if p > above
            and (
                pip_factor is None
                or _clears_noise(
                    p,
                    current_price=above,
                    pip_factor=pip_factor,
                    noise_floor_pips=noise_floor_pips,
                )
            )
        ]
        return min(candidates) if candidates else None
    if below is not None:
        candidates = [
            p
            for p in prices
            if p < below
            and (
                pip_factor is None
                or _clears_noise(
                    p,
                    current_price=below,
                    pip_factor=pip_factor,
                    noise_floor_pips=noise_floor_pips,
                )
            )
        ]
        return max(candidates) if candidates else None
    return None


def _nearest_forecast_target(
    *,
    direction: str,
    current_price: float,
    pip_factor: float,
    noise_floor_pips: Optional[float],
    structural_target: Optional[float],
    levels: list[float],
) -> Optional[float]:
    candidates: list[float] = []
    if direction == "UP":
        nearest_level = _nearest_level(
            levels,
            above=current_price,
            pip_factor=pip_factor,
            noise_floor_pips=noise_floor_pips,
        )
        for price in (nearest_level, structural_target):
            if (
                price is not None
                and price > current_price
                and _clears_noise(
                    price,
                    current_price=current_price,
                    pip_factor=pip_factor,
                    noise_floor_pips=noise_floor_pips,
                )
            ):
                candidates.append(price)
        return min(candidates) if candidates else None
    if direction == "DOWN":
        nearest_level = _nearest_level(
            levels,
            below=current_price,
            pip_factor=pip_factor,
            noise_floor_pips=noise_floor_pips,
        )
        for price in (nearest_level, structural_target):
            if (
                price is not None
                and price < current_price
                and _clears_noise(
                    price,
                    current_price=current_price,
                    pip_factor=pip_factor,
                    noise_floor_pips=noise_floor_pips,
                )
            ):
                candidates.append(price)
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
    spread_pips: Optional[float] = None,
    noise_source_timeframes: frozenset[str] = frozenset({"M1", "M5"}),
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
    # Keep `spread_pips` in the public call contract for callers that already
    # pass the live quote, but never let it alter prediction levels. The order
    # gateway owns spread feasibility and can block execution independently.
    _ = spread_pips
    noise_floor_pips = _forecast_noise_floor_pips(
        pair_chart,
        source_timeframes=noise_source_timeframes,
    )

    high_levels = _collect_structural_levels(pair_chart, side="HIGH")
    low_levels = _collect_structural_levels(pair_chart, side="LOW")
    if direction == "UP":
        target_price = _nearest_forecast_target(
            direction=direction,
            current_price=current_price,
            pip_factor=pip_factor,
            noise_floor_pips=noise_floor_pips,
            structural_target=structural_target,
            levels=high_levels,
        )
        invalidation_price = _nearest_level(
            low_levels,
            below=current_price,
            pip_factor=pip_factor,
            noise_floor_pips=noise_floor_pips,
        )
    else:
        target_price = _nearest_forecast_target(
            direction=direction,
            current_price=current_price,
            pip_factor=pip_factor,
            noise_floor_pips=noise_floor_pips,
            structural_target=structural_target,
            levels=low_levels,
        )
        invalidation_price = _nearest_level(
            high_levels,
            above=current_price,
            pip_factor=pip_factor,
            noise_floor_pips=noise_floor_pips,
        )

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


def _forecast_horizon_for_direction(pair_chart: Dict[str, Any], direction: str) -> tuple[int, str | None]:
    """Return forecast play-out horizon from aligned timeframe anchors.

    This is not an entry permission. It labels the same forecast with the
    timeframe stack that supports it so downstream ledgers can evaluate short,
    medium, and long-horizon forecasts separately.
    """
    if direction not in {"UP", "DOWN"}:
        return FORECAST_EXECUTION_HORIZON_MIN, None
    m15 = _directional_anchor_reason(pair_chart, "M15", direction)
    m30 = _directional_anchor_reason(pair_chart, "M30", direction)
    h1 = _directional_anchor_reason(pair_chart, "H1", direction)
    h4 = _directional_anchor_reason(pair_chart, "H4", direction)
    daily = _directional_anchor_reason(pair_chart, "D", direction)
    if daily and h4:
        return FORECAST_D_ANCHOR_HORIZON_MIN, f"D/H4 {direction} anchor horizon={FORECAST_D_ANCHOR_HORIZON_MIN}m ({daily}; {h4})"
    if h4 and (h1 or daily):
        companion = h1 or daily
        return FORECAST_H4_ANCHOR_HORIZON_MIN, f"H4 {direction} anchor horizon={FORECAST_H4_ANCHOR_HORIZON_MIN}m ({h4}; {companion})"
    if h1 and (m30 or m15 or h4):
        companion = m30 or m15 or h4
        return (
            FORECAST_OPERATING_SWING_HORIZON_MIN,
            f"H1 operating-swing {direction} horizon={FORECAST_OPERATING_SWING_HORIZON_MIN}m ({h1}; {companion})",
        )
    return FORECAST_EXECUTION_HORIZON_MIN, None


def _directional_anchor_reason(pair_chart: Dict[str, Any], timeframe: str, direction: str) -> str | None:
    view = _view_by_tf(pair_chart, timeframe)
    if not isinstance(view, dict):
        return None
    regime_direction = _regime_direction(view.get("regime"))
    reading = view.get("regime_reading") if isinstance(view.get("regime_reading"), dict) else {}
    if regime_direction is None:
        regime_direction = _regime_direction((reading or {}).get("state"))
    if regime_direction != direction:
        return None
    evidence: list[str] = []
    adx = _view_adx(view)
    if adx is not None and adx >= FORECAST_STRONG_ADX:
        evidence.append(f"ADX={adx:.1f}")
    family_scores = view.get("family_scores") if isinstance(view.get("family_scores"), dict) else {}
    trend_score = _to_float((family_scores or {}).get("trend_score"))
    if (
        trend_score is not None
        and (
            (direction == "UP" and trend_score >= FORECAST_TECH_FAMILY_MIN_SCORE)
            or (direction == "DOWN" and trend_score <= -FORECAST_TECH_FAMILY_MIN_SCORE)
        )
    ):
        evidence.append(f"trend_score={trend_score:.2f}")
    if not evidence:
        return None
    return f"{timeframe} {regime_direction} {'/'.join(evidence)}"


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
    """Convert an IndicatorSet 0..1 quantile rank to percentage points."""

    pct = _to_float(value)
    if pct is None or not 0.0 <= pct <= 1.0:
        return None
    return pct * 100.0


def _bounded_percent_0_100(value: Any) -> Optional[float]:
    """Validate chart-reader regime percentiles, already on a 0..100 scale."""

    pct = _to_float(value)
    if pct is None or not 0.0 <= pct <= 100.0:
        return None
    return pct


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


def _range_bounds(
    indicators: Dict[str, Any],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
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
        return close, low, high, None, None
    position = (close - low) / (high - low)
    pip_size = _to_float((indicators or {}).get("pip_size"))
    width_pips = (high - low) / pip_size if pip_size is not None and pip_size > 0 else None
    return close, low, high, max(0.0, min(1.0, position)), width_pips


def _range_position(indicators: Dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    close, low, _high, position, _width_pips = _range_bounds(indicators)
    return close, low, position


def _range_phase_bounds(
    candidates: list[tuple[int, float, float, float]],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not candidates:
        return None, None, None
    _priority, low, high, width_pips = sorted(candidates, key=lambda item: (item[0], item[3]))[0]
    return low, high, width_pips


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
    bound_candidates: list[tuple[int, float, float, float]] = []
    bound_priority = {"M15": 0, "M5": 1, "M30": 2, "H1": 3}

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
        atr_pct = _bounded_percent_0_100(
            (reading or {}).get("atr_percentile")
        )
        if atr_pct is None:
            atr_pct = _percent_0_100((indicators or {}).get("atr_percentile_100"))
        bb_width_pct = _percent_0_100((indicators or {}).get("bb_width_percentile_100"))
        trend_score = _to_float((families or {}).get("trend_score")) or 0.0
        breakout_score = _to_float((families or {}).get("breakout_score")) or 0.0
        slope = _to_float((indicators or {}).get("linreg_slope_20")) or 0.0
        close, lower, upper, position, width_pips = _range_bounds(indicators)
        if (
            lower is not None
            and upper is not None
            and width_pips is not None
            and width_pips > 0
            and (
                state in {"RANGE", "TREND_WEAK", "TRANSITION", "BREAKOUT_PENDING"}
                or "RANGE" in str(view.get("regime") or "").upper()
            )
        ):
            bound_candidates.append((bound_priority.get(tf, 9), lower, upper, width_pips))

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

    range_low, range_high, range_width_pips = _range_phase_bounds(bound_candidates)
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
            range_low,
            range_high,
            range_width_pips,
        )

    if pending >= FORECAST_RANGE_PHASE_MIN_EVIDENCE and pending >= stable * 0.6:
        confidence = min(1.0, pending / max(FORECAST_RANGE_PHASE_MIN_EVIDENCE * 2.0, 1.0))
        return _RangePhase(
            "BREAKOUT_PENDING",
            confidence,
            None,
            f"range breakout pending: pending={pending:.1f} stable={stable:.1f} forming={forming:.1f}",
            tuple(evidence[:6]),
            range_low,
            range_high,
            range_width_pips,
        )

    if stable >= FORECAST_RANGE_PHASE_MIN_EVIDENCE:
        confidence = min(1.0, stable / max(FORECAST_RANGE_PHASE_MIN_EVIDENCE * 2.0, 1.0))
        return _RangePhase(
            "IN_RANGE",
            confidence,
            None,
            f"inside stable range: stable={stable:.1f} forming={forming:.1f}",
            tuple(evidence[:6]),
            range_low,
            range_high,
            range_width_pips,
        )

    if forming >= FORECAST_RANGE_FORMING_MIN_EVIDENCE:
        confidence = min(1.0, forming / max(FORECAST_RANGE_FORMING_MIN_EVIDENCE * 2.0, 1.0))
        return _RangePhase(
            "RANGE_FORMING",
            confidence,
            None,
            f"range forming: forming={forming:.1f} stable={stable:.1f}",
            tuple(evidence[:6]),
            range_low,
            range_high,
            range_width_pips,
        )

    return _RangePhase(
        "NONE",
        0.0,
        None,
        "range-phase evidence below threshold",
        tuple(evidence[:6]),
        range_low,
        range_high,
        range_width_pips,
    )


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


def _range_rail_reversion_priors(pair_chart: Dict[str, Any], phase: _RangePhase) -> list[tuple[str, float, str]]:
    """Add path-first pressure inside a range box.

    When an active box holds, the midpoint is the geometric separator between
    support-side and resistance-side path risk. Price in the lower half has
    bounce/retest risk before a fresh downside forecast is proved; price in the
    upper half has fade risk before a fresh upside forecast is proved. This is
    not an entry gate and not a tuned threshold: it is the same support/resistance
    box geometry used by range entries and failed-break timing.
    """
    if phase.phase not in {"IN_RANGE", "RANGE_FORMING"}:
        return []
    up_strength = 0.0
    down_strength = 0.0
    evidence: list[str] = []
    for view in pair_chart.get("views") or []:
        if not isinstance(view, dict):
            continue
        tf = str(view.get("granularity") or "").upper()
        if tf not in {"M5", "M15", "M30", "H1"}:
            continue
        state = _view_regime_state(view)
        if state not in {"RANGE", "TREND_WEAK", "TRANSITION"} and "RANGE" not in str(view.get("regime") or "").upper():
            continue
        indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else {}
        _close, _lower, position = _range_position(indicators)
        if position is None:
            continue
        if abs(position - 0.5) <= 1e-9:
            continue
        weight = _tf_weight(tf)
        if position < 0.5:
            strength = (0.5 - position) * 2.0 * weight
            up_strength += strength
            evidence.append(f"{tf} lower-half pos={position:.2f}")
        elif position > 0.5:
            strength = (position - 0.5) * 2.0 * weight
            down_strength += strength
            evidence.append(f"{tf} upper-half pos={position:.2f}")
    total = up_strength + down_strength
    if total <= 0:
        return []
    direction = "UP" if up_strength >= down_strength else "DOWN"
    dominant = max(up_strength, down_strength) / total
    magnitude = FORECAST_RANGE_INSIDE_PRIOR * max(phase.confidence, 0.35) * dominant
    return [
        (
            direction,
            magnitude,
            f"{phase.phase} rail-location {'; '.join(evidence[:4])} → {direction} range-reversion path prior {magnitude:.1f}",
        )
    ]


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
    """Return path/location priors that are not lagging-indicator direction.

    RSI/MACD/ADX/EMA/BB-family composites describe the tape that already
    printed. They are useful context for noise, exhaustion, and countertrend
    skepticism, but letting them add UP/DOWN forecast score turns a lagging
    panel into a leading predictor. Directional forecast score must therefore
    come from detector/projection/path/structure evidence, while this helper
    keeps only broad market-location priors.
    """
    confluence = pair_chart.get("confluence") or {}
    priors: list[tuple[str, float, str]] = []

    priors.extend(_market_location_priors(confluence))
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
    runner_up_score: float,
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
    # Low TF agreement means the tape is mixed; it is not evidence that a
    # short-term countertrend forecast should be trusted. A decisive pair-chart
    # score gap still has to damp an unconfirmed fade, otherwise mixed tapes
    # are where micro reversal signals most often become reverse-first entries.
    if _has_close_confirmed_structure(pair_chart, winner):
        return winner_score, (
            f"countertrend {winner} allowed: M15/H1 close-confirmed structure offsets "
            f"{balance} score_gap={score_gap:.3f}"
        )
    adjusted = min(winner_score * FORECAST_COUNTERTREND_UNCONFIRMED_MULT, max(runner_up_score, 0.0))
    return adjusted, (
        f"countertrend {winner} damped {winner_score:.1f}→{adjusted:.1f}: "
        f"{balance} score_gap={score_gap:.3f}, tf_agreement={tf_agree}, "
        "lagging indicator bias requires M15/H1 close-confirmed reversal"
    )


def _regime_family_adjustment(
    technical_context: Dict[str, Any],
    winner: str,
    winner_score: float,
    runner_up_score: float,
) -> tuple[float, str | None]:
    """Use the shared receipt only as a contradiction gate.

    Same-direction family evidence never boosts a detector forecast. An exact
    opposite aggregate removes the winning margin, forcing the existing
    contested/UNCLEAR logic to decide conservatively without inventing a new
    confidence multiplier.
    """

    if winner not in {"UP", "DOWN"}:
        return winner_score, None
    receipt = technical_context.get("regime_family_weighting")
    consistent, reason = forecast_direction_consistency(
        receipt,
        forecast_direction=winner,
    )
    if consistent:
        # Aligned/non-directional family context is not an additional detector
        # and must never surface as a support driver or live permission.
        return winner_score, None
    return min(winner_score, runner_up_score), (
        f"{reason}: {winner} winning margin removed by content-addressed "
        "regime-selected family evidence"
    )


def _build_regime_family_contradiction_shadow(
    *,
    pair: str,
    current_price: float,
    detector_direction: str,
    detector_scores: dict[str, float],
    technical_context_v1: dict[str, Any],
    entry_bid: float | None,
    entry_ask: float | None,
) -> dict[str, Any]:
    """Build isolated evaluation evidence without affecting the forecast."""

    try:
        from quant_rabbit.strategy.regime_family_contradiction_shadow import (
            build_regime_family_contradiction_shadow,
        )

        return build_regime_family_contradiction_shadow(
            pair=pair,
            current_price=current_price,
            detector_direction=detector_direction,
            detector_scores=detector_scores,
            technical_context_v1=technical_context_v1,
            entry_bid=entry_bid,
            entry_ask=entry_ask,
        )
    except (ImportError, TypeError, ValueError, OverflowError):
        # Shadow telemetry is deliberately non-authoritative.  A malformed or
        # unavailable evaluator must never change the fail-closed live result.
        return {}


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


def _contested_range_raw_confidence(
    *,
    phase: _RangePhase,
    winner_score: float,
    runner_up_score: float,
    range_score: float,
    margin: float,
) -> Optional[float]:
    """Return RANGE confidence when direction is contested inside a box.

    This does not introduce a separate market threshold. It reuses the same
    decisiveness margin that would otherwise produce UNCLEAR: if the active
    range evidence is at least as large as the UP/DOWN separation, the correct
    forecast is "trade the box only", not "no executable forecast metadata".
    Confidence blends two normalized score facts equally: how indecisive the
    directional book is, and how material range evidence is versus the weaker
    directional side.
    """
    if phase.phase not in {"IN_RANGE", "RANGE_FORMING"}:
        return None
    if range_score <= 0 or winner_score <= 0 or runner_up_score <= 0:
        return None
    directional_uncertainty = 1.0 - min(1.0, margin / max(winner_score, 1.0))
    if range_score < margin:
        # The aggregate range-prior echo lost to directional signal noise,
        # but the phase detector's DIRECT box measurement can still qualify.
        # 2026-06-11 live funnel: 19/28 pairs sat in IN_RANGE/RANGE_FORMING
        # boxes (phase confidence 0.66-1.00) while the production signal
        # book inflated up/down scores enough that margin exceeded the
        # small range_score echo — so 60+ rotation lanes per cycle died as
        # "UNCLEAR conf=0.1" despite a measured, tradeable box. The bar is
        # phase confidence >= 0.5, which by the detector's own formula
        # (confidence = evidence / (2 x FORECAST_RANGE_PHASE_MIN_EVIDENCE))
        # means its evidence reached the module's documented minimum-evidence
        # constant — not a new tuned threshold. BREAKOUT_PENDING is already
        # excluded above, so squeeze boxes still refuse rotation.
        phase_confidence = max(0.0, min(1.0, float(phase.confidence or 0.0)))
        if phase_confidence < 0.5:
            return None
        return min(1.0, max(0.0, (directional_uncertainty + phase_confidence) / 2.0))
    range_materiality = min(1.0, range_score / max(runner_up_score, 1.0))
    return min(1.0, max(0.0, (directional_uncertainty + range_materiality) / 2.0))


def _weak_directional_range_raw_confidence(
    *,
    phase: _RangePhase,
    winner: str,
    directional_confidence: float,
    winner_score: float,
    range_score: float,
) -> Optional[float]:
    """Return RANGE confidence when a calibrated direction is weak inside a box.

    A decisive raw UP/DOWN score inside a measured box is not enough for live
    direction if the audited directional bucket has already damped it below the
    entry floor. In that case, keep the box thesis executable as RANGE_ROTATION
    and let the range rail / spread / RR gates decide whether a small rotation
    can actually trade.
    """
    if winner not in {"UP", "DOWN"}:
        return None
    if phase.phase not in {"IN_RANGE", "RANGE_FORMING"}:
        return None
    if phase.range_low_price is None or phase.range_high_price is None:
        return None
    if directional_confidence >= ENTRY_CONFIDENCE_MIN:
        return None
    phase_confidence = max(0.0, min(1.0, float(phase.confidence or 0.0)))
    if phase_confidence < 0.5:
        return None
    if range_score <= 0:
        return None
    range_materiality = min(1.0, range_score / max(winner_score * 0.25, 1.0))
    return min(1.0, max(phase_confidence, (phase_confidence + range_materiality) / 2.0))


def _calibrated_confidence(
    *,
    pair: str,
    direction: str,
    raw_confidence: float,
    hit_rates: Dict[str, Dict[str, Any]] | None,
    regime: Optional[str],
) -> tuple[float, float]:
    if hit_rates is None:
        return raw_confidence, 1.0
    from quant_rabbit.strategy.projection_ledger import (
        confidence_calibration,
        select_calibration_signal_name,
    )

    cal_signal_name = select_calibration_signal_name(
        "directional_forecast",
        direction,
        pair,
        hit_rates=hit_rates,
        regime=regime,
    )
    cal_mult = confidence_calibration(
        cal_signal_name, pair, hit_rates=hit_rates, regime=regime,
    )
    return min(1.0, raw_confidence * cal_mult), cal_mult


def _forecast_learning_receipt(
    *,
    pair: str,
    direction: str,
    confidence: float,
    raw_confidence: float,
    up_score: float,
    down_score: float,
    range_score: float,
    horizon_min: float,
    technical_context_v1: dict[str, Any],
    now_utc: datetime | None,
    drivers_for: tuple[str, ...],
    drivers_against: tuple[str, ...],
) -> dict[str, Any]:
    """Return one content-addressed keep/invert prediction, never no-trade."""

    if direction not in {"UP", "DOWN"}:
        return {}
    from quant_rabbit.forecast_learning import (
        forecast_feature_values,
        forecast_orientation_decision,
        load_forecast_orientation_model,
    )

    model = load_forecast_orientation_model(DEFAULT_FORECAST_ORIENTATION_MODEL)
    if not model:
        return {}
    timestamp = _aware_utc_datetime(now_utc) or datetime.now(timezone.utc)
    features = forecast_feature_values(
        pair=pair,
        direction=direction,
        confidence=confidence,
        raw_confidence=raw_confidence,
        up_score=up_score,
        down_score=down_score,
        range_score=range_score,
        horizon_min=horizon_min,
        technical_context=technical_context_v1,
        timestamp_utc=timestamp,
        drivers_for=drivers_for,
        drivers_against=drivers_against,
    )
    decision = forecast_orientation_decision(
        model,
        original_direction=direction,
        features=features,
    )
    feature_sha256 = hashlib.sha256(
        json.dumps(
            features,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    body = {
        "schema": "QR_FORECAST_LEARNING_DECISION_V1",
        "model_sha256": decision.get("model_sha256"),
        "model_status": decision.get("model_status"),
        "original_direction": direction,
        "forecast_direction": decision.get("direction"),
        "rank_direction": decision.get("rank_direction"),
        "orientation": decision.get("orientation"),
        "direct_orientation_probability": decision.get("direct_probability"),
        "selected_orientation_probability": decision.get("selected_probability"),
        "ordinary_correction_applied": decision.get(
            "ordinary_correction_applied"
        )
        is True,
        "ranking_horizon_min": int(horizon_min),
        "always_returns_direction": True,
        "probability_semantics": "ORIENTATION_RANK_NOT_TRADE_WIN_PROBABILITY",
        "features": features,
        "feature_sha256": feature_sha256,
    }
    return {
        **body,
        "decision_sha256": hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _projection_signal_calibration_multiplier(
    *,
    signal: Any,
    pair: str,
    hit_rates: Dict[str, Dict[str, Any]] | None,
    regime: Optional[str],
) -> float:
    """Return detector-specific calibration before forecast aggregation."""
    if hit_rates is None:
        return 1.0
    name = str(getattr(signal, "name", "") or "").strip()
    if not name:
        return 1.0
    from quant_rabbit.strategy.projection_ledger import (
        confidence_calibration,
        select_calibration_signal_name,
    )

    direction = str(getattr(signal, "direction", "UNCLEAR") or "UNCLEAR")
    cal_signal_name = select_calibration_signal_name(
        name,
        direction,
        pair,
        hit_rates=hit_rates,
        regime=regime,
    )
    return confidence_calibration(
        cal_signal_name,
        pair,
        hit_rates=hit_rates,
        regime=regime,
    )


def _projection_signal_known_weak_reason(
    *,
    signal: Any,
    pair: str,
    hit_rates: Dict[str, Dict[str, Any]] | None,
    regime: Optional[str],
) -> str | None:
    """Return a rejection reason for audited sub-random directional signals."""
    if hit_rates is None:
        return None
    name = str(getattr(signal, "name", "") or "").strip()
    direction = str(getattr(signal, "direction", "UNCLEAR") or "UNCLEAR").upper()
    if not name or direction not in {"UP", "DOWN"}:
        return None
    try:
        from quant_rabbit.strategy.projection_ledger import select_calibration_signal_name
    except Exception:
        return None

    pair_key = pair.upper()
    cal_signal_name = select_calibration_signal_name(
        name,
        direction,
        pair_key,
        hit_rates=hit_rates,
        regime=regime,
    )
    by_key = hit_rates.get(cal_signal_name) or {}
    if not isinstance(by_key, dict):
        return None

    candidates: list[tuple[str, Any]] = []
    if regime is not None:
        candidates.append((f"{pair_key}:{regime}", by_key.get(f"{pair_key}:{regime}")))
    candidates.append((f"{pair_key}:_all_regimes", by_key.get(f"{pair_key}:_all_regimes")))
    if regime is not None:
        candidates.append((f"_all_pairs:{regime}", by_key.get(f"_all_pairs:{regime}")))
    candidates.append(("_all_pairs:_all_regimes", by_key.get("_all_pairs:_all_regimes")))
    candidates.append((pair_key, by_key.get(pair_key)))
    candidates.append(("_all_pairs", by_key.get("_all_pairs")))

    for key, bucket in candidates:
        if not isinstance(bucket, dict):
            continue
        samples = _to_int(bucket.get("samples"))
        hit_rate = _to_float(bucket.get("hit_rate"))
        if samples is None or hit_rate is None:
            continue
        if samples < FORECAST_PROJECTION_WEAK_BLOCK_MIN_SAMPLES:
            continue
        if hit_rate < FORECAST_PROJECTION_WEAK_BLOCK_HIT_RATE:
            return (
                f"ignored known-weak projection {cal_signal_name} via {key}: "
                f"hit_rate={hit_rate:.2f} samples={samples}"
            )
        return None
    return None


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
    spread_pips: Optional[float] = None,
    entry_bid: Optional[float] = None,
    entry_ask: Optional[float] = None,
    calendar_path: Path | None = None,
    strategy_profile_path: Path | None = None,
    now_utc: datetime | None = None,
    require_technical_candle_integrity: bool = True,
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
    integrity_blockers = _technical_candle_integrity_forecast_blockers(
        pair_chart,
        expected_pair=pair,
        require_technical_candle_integrity=require_technical_candle_integrity,
        now_utc=now_utc,
    )
    m15_recovery_receipt: dict[str, Any] = {}
    m15_recovery_filtered_input_sha256 = ""
    if integrity_blockers:
        candidate = build_m15_recovery_micro_receipt(
            pair_chart,
            expected_pair=pair,
            now_utc=now_utc,
            current_spread_pips=spread_pips,
        )
        if candidate is None:
            return DirectionalForecast(
                pair=pair, direction="UNCLEAR", confidence=0.0,
                invalidation_price=None, target_price=None, horizon_min=0,
                drivers_for=(), drivers_against=integrity_blockers,
                rationale_summary=(
                    f"{','.join(integrity_blockers)}: technical candle provenance blocks directional forecast"
                ),
                current_price=current_price,
                raw_confidence=0.0,
            )
        m15_recovery_receipt = candidate
    technical_context_v1 = build_forecast_technical_context(
        pair_chart,
        pair=pair,
        current_price=current_price,
        spread_pips=spread_pips,
        calendar_path=calendar_path,
        strategy_profile_path=strategy_profile_path,
        now_utc=now_utc,
    )
    if m15_recovery_receipt:
        pair_chart = _m15_recovery_forecast_chart(pair_chart)
        pattern_signals = [
            signal for signal in pattern_signals or [] if _m15_or_higher_signal(signal)
        ]
        projection_signals = [
            signal for signal in projection_signals or [] if _m15_or_higher_signal(signal)
        ]
        correlation_signals = [
            signal for signal in correlation_signals or [] if _m15_or_higher_signal(signal)
        ]
        paths = [signal for signal in paths or [] if _m15_or_higher_signal(signal)]
        reversal_long = (
            reversal_long if _m15_or_higher_signal(reversal_long) else None
        )
        reversal_short = (
            reversal_short if _m15_or_higher_signal(reversal_short) else None
        )
        m15_recovery_filtered_input_sha256 = (
            _m15_recovery_filtered_inputs_sha256(
                pattern_signals=list(pattern_signals),
                projection_signals=list(projection_signals),
                correlation_signals=list(correlation_signals),
                paths=list(paths),
                reversal_long=reversal_long,
                reversal_short=reversal_short,
            )
        )
    # The caller's regime label comes from the raw top-level confluence,
    # which aggregates M1/M5.  A recovery forecast explicitly excludes those
    # inputs, so regime-specific calibration must also fall back to the
    # all-regimes evidence rather than smuggling the fast tails back in.
    calibration_regime = m15_recovery_calibration_regime(
        regime,
        m15_recovery_receipt,
    )
    if _is_disabled():
        return DirectionalForecast(
            pair=pair, direction="UNCLEAR", confidence=0.0,
            invalidation_price=None, target_price=None, horizon_min=0,
            drivers_for=(), drivers_against=(),
            rationale_summary="forecaster disabled",
            current_price=current_price,
            technical_context_v1=technical_context_v1,
            m15_recovery_receipt=m15_recovery_receipt,
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
        known_weak_reason = _projection_signal_known_weak_reason(
            signal=s,
            pair=pair,
            hit_rates=hit_rates,
            regime=calibration_regime,
        )
        if known_weak_reason is not None:
            contributions.append(("IGNORED", 0.0, known_weak_reason))
            continue
        cal_mult = _projection_signal_calibration_multiplier(
            signal=s,
            pair=pair,
            hit_rates=hit_rates,
            regime=calibration_regime,
        )
        mag = getattr(s, "bonus_magnitude", 0) * getattr(s, "confidence", 0) * cal_mult
        rationale = getattr(s, "rationale", "")
        if cal_mult != 1.0 and rationale:
            rationale = f"{rationale} [cal×{cal_mult:.2f}]"
        if mag < 0:
            range_score += abs(mag)
            if rationale:
                contributions.append(("RANGE", abs(mag), rationale))
            continue
        _add(getattr(s, "direction", "UNCLEAR"), mag, rationale)
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
    for prior_dir, prior_mag, prior_reason in _range_rail_reversion_priors(pair_chart, range_phase):
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
            technical_context_v1=technical_context_v1,
            m15_recovery_receipt=m15_recovery_receipt,
        )

    # Pick winner
    candidates = [("UP", up_score), ("DOWN", down_score), ("RANGE", range_score)]
    candidates.sort(key=lambda x: -x[1])
    winner, winner_score = candidates[0]
    runner_up_score = candidates[1][1]
    pre_family_detector_scores = {
        "UP": up_score,
        "DOWN": down_score,
        "RANGE": range_score,
        "EITHER": either_score,
    }
    if m15_recovery_receipt:
        # The normal technical-context receipt includes M1/M5 families. The
        # recovery forecast may publish it diagnostically, but must not use it
        # to choose or veto the M15+ direction.
        family_adjusted_score, family_adjustment_reason = winner_score, None
    else:
        family_adjusted_score, family_adjustment_reason = _regime_family_adjustment(
            technical_context_v1,
            winner,
            winner_score,
            runner_up_score,
        )
    if family_adjusted_score != winner_score:
        contradicted_winner = winner
        contradiction_shadow = _build_regime_family_contradiction_shadow(
            pair=pair,
            current_price=current_price,
            detector_direction=contradicted_winner,
            detector_scores=pre_family_detector_scores,
            technical_context_v1=technical_context_v1,
            entry_bid=entry_bid,
            entry_ask=entry_ask,
        )
        contradiction_learning_receipt: dict[str, Any] = {}
        if require_technical_candle_integrity and not m15_recovery_receipt:
            # The ordinary forecast remains fail-closed below, but the ranked
            # forward-SCOUT learner must still see the point-in-time detector
            # thesis that the regime-family selector contradicted.  Returning
            # before building this receipt made the most informative conflict
            # cases invisible to the historical-tick learner and could erase
            # every bounded fallback candidate in a cycle.
            detector_candidates = sorted(
                (
                    ("UP", float(pre_family_detector_scores["UP"])),
                    ("DOWN", float(pre_family_detector_scores["DOWN"])),
                    ("RANGE", float(pre_family_detector_scores["RANGE"])),
                ),
                key=lambda item: -item[1],
            )
            detector_winner_score = detector_candidates[0][1]
            detector_runner_up_score = detector_candidates[1][1]
            detector_total = sum(pre_family_detector_scores.values())
            detector_raw_confidence = min(
                1.0,
                (
                    max(detector_winner_score - detector_runner_up_score, 0.0)
                    / max(detector_total, 1.0)
                )
                + 0.3,
            )
            detector_confidence, _ = _calibrated_confidence(
                pair=pair,
                direction=contradicted_winner,
                raw_confidence=detector_raw_confidence,
                hit_rates=hit_rates,
                regime=calibration_regime,
            )
            detector_horizon_min, _ = _forecast_horizon_for_direction(
                pair_chart,
                contradicted_winner,
            )
            contradiction_learning_receipt = _forecast_learning_receipt(
                pair=pair,
                direction=contradicted_winner,
                confidence=detector_confidence,
                raw_confidence=detector_raw_confidence,
                up_score=float(pre_family_detector_scores["UP"]),
                down_score=float(pre_family_detector_scores["DOWN"]),
                range_score=float(pre_family_detector_scores["RANGE"]),
                horizon_min=detector_horizon_min,
                technical_context_v1=technical_context_v1,
                now_utc=now_utc,
                drivers_for=_top_reasons(
                    contributions,
                    direction=contradicted_winner,
                ),
                drivers_against=(
                    family_adjustment_reason or "regime-family contradiction",
                ),
            )
        if winner == "UP":
            up_score = family_adjusted_score
        elif winner == "DOWN":
            down_score = family_adjusted_score
        total = up_score + down_score + range_score + either_score
        candidates = [("UP", up_score), ("DOWN", down_score), ("RANGE", range_score)]
        candidates.sort(key=lambda x: -x[1])
        winner, winner_score = candidates[0]
        runner_up_score = candidates[1][1]
        # Do not let stable-sort order turn a vetoed DOWN winner into an UP
        # forecast (or vice versa) when the scores are forced level.  An exact
        # family-direction contradiction is a fail-closed forecast fact.
        return DirectionalForecast(
            pair=pair,
            direction="UNCLEAR",
            confidence=0.0,
            invalidation_price=None,
            target_price=None,
            horizon_min=0,
            drivers_for=_top_reasons(
                contributions,
                direction=contradicted_winner,
            ),
            drivers_against=(
                family_adjustment_reason or "regime-family contradiction",
            ),
            rationale_summary=(
                f"regime-family contradiction vetoed {contradicted_winner}: "
                f"UP={up_score:.1f} DOWN={down_score:.1f} "
                f"RANGE={range_score:.1f} EITHER={either_score:.1f}"
            ),
            current_price=current_price,
            up_score=up_score,
            down_score=down_score,
            range_score=range_score,
            raw_confidence=0.0,
            component_scores={
                "UP": up_score,
                "DOWN": down_score,
                "RANGE": range_score,
                "EITHER": either_score,
            },
            technical_context_v1=technical_context_v1,
            forecast_learning_v1=contradiction_learning_receipt,
            m15_recovery_receipt=m15_recovery_receipt,
            regime_family_contradiction_shadow=contradiction_shadow,
        )
    adjusted_winner_score, adjustment_reason = _countertrend_adjustment(
        pair_chart,
        winner,
        winner_score,
        runner_up_score,
    )
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
    if up_score + down_score + range_score <= 0:
        return DirectionalForecast(
            pair=pair,
            direction="UNCLEAR",
            confidence=0.0,
            invalidation_price=None,
            target_price=None,
            horizon_min=0,
            drivers_for=_top_reasons(contributions, direction=winner),
            drivers_against=((adjustment_reason,) if adjustment_reason else ()),
            rationale_summary=adjustment_reason or "no forward forecast evidence after lagging-indicator dampening",
            current_price=current_price,
            up_score=up_score,
            down_score=down_score,
            range_score=range_score,
            raw_confidence=0.0,
            component_scores={"UP": up_score, "DOWN": down_score, "RANGE": range_score, "EITHER": either_score},
            technical_context_v1=technical_context_v1,
            m15_recovery_receipt=m15_recovery_receipt,
        )
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
            technical_context_v1=technical_context_v1,
            m15_recovery_receipt=m15_recovery_receipt,
        )

    # Decisiveness: need clear winner, not close call
    margin = winner_score - runner_up_score
    if margin < winner_score * 0.25:
        contested_range_confidence = _contested_range_raw_confidence(
            phase=range_phase,
            winner_score=winner_score,
            runner_up_score=runner_up_score,
            range_score=range_score,
            margin=margin,
        )
        if contested_range_confidence is not None:
            calibrated_confidence, cal_mult = _calibrated_confidence(
                pair=pair,
                direction="RANGE",
                raw_confidence=contested_range_confidence,
                hit_rates=hit_rates,
                regime=calibration_regime,
            )
            evidence = "; ".join(range_phase.evidence[:3])
            rationale_summary = (
                f"contested direction inside {range_phase.phase}: "
                f"UP={up_score:.1f} DOWN={down_score:.1f} RANGE={range_score:.1f} "
                f"EITHER={either_score:.1f}, margin={margin:.1f} ≤ range evidence "
                f"→ RANGE conf {contested_range_confidence:.2f} × cal {cal_mult:.2f} = "
                f"{calibrated_confidence:.2f}"
            )
            return DirectionalForecast(
                pair=pair,
                direction="RANGE",
                confidence=calibrated_confidence,
                invalidation_price=None,
                target_price=None,
                horizon_min=FORECAST_RANGE_HORIZON_MIN,
                drivers_for=(
                    (range_phase.rationale,)
                    + ((evidence,) if evidence else ())
                    + _top_reasons(contributions, direction="RANGE", limit=3)
                ),
                drivers_against=_top_reasons(contributions, direction=winner),
                rationale_summary=rationale_summary,
                current_price=_round_price(pair, current_price),
                up_score=up_score,
                down_score=down_score,
                range_score=range_score,
                raw_confidence=contested_range_confidence,
                calibration_multiplier=cal_mult,
                component_scores={"UP": up_score, "DOWN": down_score, "RANGE": range_score, "EITHER": either_score},
                range_low_price=range_phase.range_low_price,
                range_high_price=range_phase.range_high_price,
                range_width_pips=range_phase.range_width_pips,
                technical_context_v1=technical_context_v1,
                m15_recovery_receipt=m15_recovery_receipt,
            )
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
            technical_context_v1=technical_context_v1,
            m15_recovery_receipt=m15_recovery_receipt,
        )

    raw_confidence = min(1.0, (margin / total) + 0.3)

    # Bayesian calibration at forecast level
    calibrated_confidence, cal_mult = _calibrated_confidence(
        pair=pair,
        direction=winner,
        raw_confidence=raw_confidence,
        hit_rates=hit_rates,
        regime=calibration_regime,
    )
    if winner == "RANGE":
        horizon_min = FORECAST_RANGE_HORIZON_MIN
        horizon_reason = None
    else:
        horizon_min, horizon_reason = _forecast_horizon_for_direction(
            pair_chart,
            winner,
        )
    forecast_learning_v1: dict[str, Any] = {}
    if require_technical_candle_integrity and not m15_recovery_receipt:
        original_winner = winner
        forecast_learning_v1 = _forecast_learning_receipt(
            pair=pair,
            direction=original_winner,
            confidence=calibrated_confidence,
            raw_confidence=raw_confidence,
            up_score=up_score,
            down_score=down_score,
            range_score=range_score,
            horizon_min=horizon_min,
            technical_context_v1=technical_context_v1,
            now_utc=now_utc,
            drivers_for=_top_reasons(contributions, direction=original_winner),
            drivers_against=_top_reasons(
                contributions,
                direction="DOWN" if original_winner == "UP" else "UP",
            ),
        )
        learned_direction = str(
            forecast_learning_v1.get("forecast_direction") or ""
        ).upper()
        if (
            forecast_learning_v1.get("ordinary_correction_applied") is True
            and learned_direction in {"UP", "DOWN"}
            and learned_direction != original_winner
        ):
            winner = learned_direction
            horizon_min, horizon_reason = _forecast_horizon_for_direction(
                pair_chart,
                winner,
            )
            learning_reason = (
                f"S5 bid/ask walk-forward model inverted {original_winner} to "
                f"{winner} (orientation rank="
                f"{forecast_learning_v1.get('selected_orientation_probability')})"
            )
            adjustment_reason = (
                f"{adjustment_reason}; {learning_reason}"
                if adjustment_reason
                else learning_reason
            )

    weak_directional_range_confidence = (
        None
        if m15_recovery_receipt
        else _weak_directional_range_raw_confidence(
            phase=range_phase,
            winner=winner,
            directional_confidence=calibrated_confidence,
            winner_score=winner_score,
            range_score=range_score,
        )
    )
    if weak_directional_range_confidence is not None:
        range_confidence, range_cal_mult = _calibrated_confidence(
            pair=pair,
            direction="RANGE",
            raw_confidence=weak_directional_range_confidence,
            hit_rates=hit_rates,
            regime=calibration_regime,
        )
        evidence = "; ".join(range_phase.evidence[:3])
        rationale_summary = (
            f"weak calibrated {winner} forecast inside {range_phase.phase}: "
            f"UP={up_score:.1f} DOWN={down_score:.1f} RANGE={range_score:.1f} "
            f"EITHER={either_score:.1f}; {winner} conf {raw_confidence:.2f} × cal "
            f"{cal_mult:.2f} = {calibrated_confidence:.2f} < {ENTRY_CONFIDENCE_MIN:.2f} "
            f"→ RANGE conf {weak_directional_range_confidence:.2f} × cal "
            f"{range_cal_mult:.2f} = {range_confidence:.2f}"
        )
        return DirectionalForecast(
            pair=pair,
            direction="RANGE",
            confidence=range_confidence,
            invalidation_price=None,
            target_price=None,
            horizon_min=FORECAST_RANGE_HORIZON_MIN,
            drivers_for=(
                (range_phase.rationale,)
                + ((evidence,) if evidence else ())
                + _top_reasons(contributions, direction="RANGE", limit=3)
            ),
            drivers_against=_top_reasons(contributions, direction=winner),
            rationale_summary=rationale_summary,
            current_price=_round_price(pair, current_price),
            up_score=up_score,
            down_score=down_score,
            range_score=range_score,
            raw_confidence=weak_directional_range_confidence,
            calibration_multiplier=range_cal_mult,
            component_scores={"UP": up_score, "DOWN": down_score, "RANGE": range_score, "EITHER": either_score},
            range_low_price=range_phase.range_low_price,
            range_high_price=range_phase.range_high_price,
            range_width_pips=range_phase.range_width_pips,
            technical_context_v1=technical_context_v1,
            m15_recovery_receipt=m15_recovery_receipt,
        )

    target_price, invalidation_price = _forecast_geometry(
        pair=pair,
        pair_chart=pair_chart,
        direction=winner,
        current_price=current_price,
        spread_pips=spread_pips,
        noise_source_timeframes=(
            frozenset({M15_RECOVERY_SOURCE_TIMEFRAME})
            if m15_recovery_receipt
            else frozenset({"M1", "M5"})
        ),
    )
    geometry_reason = ""
    if winner in {"UP", "DOWN"} and (target_price is None or invalidation_price is None):
        calibrated_confidence *= FORECAST_INCOMPLETE_GEOMETRY_CONFIDENCE_MULT
        missing = []
        if target_price is None:
            missing.append("target")
        if invalidation_price is None:
            missing.append("invalidation")
        geometry_reason = (
            f"; robust forecast geometry missing {'/'.join(missing)} "
            f"→ confidence ×{FORECAST_INCOMPLETE_GEOMETRY_CONFIDENCE_MULT:.2f}"
        )

    rationale_summary = (
        f"UP={up_score:.1f} DOWN={down_score:.1f} RANGE={range_score:.1f} EITHER={either_score:.1f} → "
        f"{winner} conf {raw_confidence:.2f} × cal {cal_mult:.2f} = {calibrated_confidence:.2f}"
    )
    if adjustment_reason:
        rationale_summary = f"{rationale_summary}; {adjustment_reason}"
    if geometry_reason:
        rationale_summary = f"{rationale_summary}{geometry_reason}"
    if horizon_reason:
        rationale_summary = f"{rationale_summary}; {horizon_reason}"
    opposite = "DOWN" if winner == "UP" else "UP" if winner == "DOWN" else candidates[1][0]
    drivers_for = _top_reasons(contributions, direction=winner)
    if horizon_reason:
        drivers_for = (*drivers_for, horizon_reason)

    component_scores = {
        "UP": up_score,
        "DOWN": down_score,
        "RANGE": range_score,
        "EITHER": either_score,
    }
    recovery_forecast_evidence = (
        _m15_recovery_forecast_evidence(
            receipt=m15_recovery_receipt,
            pair=pair,
            current_price=_round_price(pair, current_price),
            spread_pips=spread_pips,
            filtered_input_sha256=m15_recovery_filtered_input_sha256,
            direction=winner,
            confidence=calibrated_confidence,
            raw_confidence=raw_confidence,
            calibration_multiplier=cal_mult,
            component_scores=component_scores,
            target_price=target_price,
            invalidation_price=invalidation_price,
            horizon_min=horizon_min,
        )
        if m15_recovery_receipt
        and winner in {"UP", "DOWN"}
        and m15_recovery_filtered_input_sha256
        else {}
    )
    return DirectionalForecast(
        pair=pair, direction=winner, confidence=calibrated_confidence,
        invalidation_price=invalidation_price, target_price=target_price,
        horizon_min=horizon_min,
        drivers_for=drivers_for,
        drivers_against=_top_reasons(contributions, direction=opposite),
        rationale_summary=rationale_summary,
        current_price=_round_price(pair, current_price),
        up_score=up_score,
        down_score=down_score,
        range_score=range_score,
        raw_confidence=raw_confidence,
        calibration_multiplier=cal_mult,
        component_scores=component_scores,
        range_low_price=range_phase.range_low_price if winner == "RANGE" else None,
        range_high_price=range_phase.range_high_price if winner == "RANGE" else None,
        range_width_pips=range_phase.range_width_pips if winner == "RANGE" else None,
        technical_context_v1=technical_context_v1,
        forecast_learning_v1=forecast_learning_v1,
        m15_recovery_receipt=m15_recovery_receipt,
        m15_recovery_forecast_evidence=recovery_forecast_evidence,
    )
