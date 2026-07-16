"""Content-addressed regime-family selection with situation-aware TF weights.

This is the single structured handoff between chart interpretation and the
directional forecast.  It deliberately does not create entry permission:
aligned family evidence is diagnostic, while contradictory evidence may only
veto or damp an independently produced forecast.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS
from quant_rabbit.strategy.failed_break_evidence import (
    build_m5_failed_break_evidence,
    failed_break_direction,
    verify_m5_failed_break_evidence,
)
from quant_rabbit.strategy.tf_weights import (
    BASELINE_WEIGHTS,
    DYNAMIC_TF_POLICY_CONTRACT,
    DYNAMIC_TF_POLICY_FIELDS,
    derive_dynamic_tf_policy_from_evidence,
    dynamic_tf_weights,
    dynamic_tf_weights_from_policy_inputs,
    verify_dynamic_tf_policy_evidence,
)


CONTRACT = "QR_REGIME_FAMILY_WEIGHTING_V1"
SCHEMA_VERSION = "regime_family_weighting_v1"
POLICY_VERSION = "regime_family_dynamic_tf_v1"
TIMEFRAMES = tuple(BASELINE_WEIGHTS)
ACTIVE_FAMILIES = {"TREND", "MEAN_REVERSION", "BREAKOUT", "NONE"}
DIRECTIONS = {"UP", "DOWN", "EITHER", "NONE"}
REGIME_TO_FAMILY = {
    "TREND_STRONG": "TREND",
    "TREND_WEAK": "TREND",
    "RANGE": "MEAN_REVERSION",
    "BREAKOUT_PENDING": "BREAKOUT",
    "TRANSITION": "NONE",
    "UNKNOWN": "NONE",
}
FAMILY_TO_METHOD = {
    "TREND": "TREND_CONTINUATION",
    "MEAN_REVERSION": "RANGE_ROTATION",
    # BREAKOUT_PENDING is an observed situation, not an executable thesis.
    # A close-confirmed continuation and a failed-break fade require opposite
    # methods/directions, so v1 waits until one of those method-specific facts
    # exists instead of guessing from the breakout score or Donchian sign.
    "BREAKOUT": None,
    "NONE": None,
}
MAX_SCORE_ABS = 100.0
MAX_GENERATED_AT_CHARS = 64
MAX_SESSION_CHARS = 32
MAX_SITUATION_CHARS = 256

_TOP_LEVEL_FIELDS = {
    "contract",
    "schema_version",
    "policy_version",
    "source_identity",
    "source_identity_sha256",
    "policy_inputs",
    "m5_failed_break_evidence",
    "weights",
    "by_timeframe",
    "aggregate",
    "receipt_sha256",
}
_SOURCE_FIELDS = {
    "pair",
    "chart_generated_at_utc",
    "session",
    "dominant_regime",
    "primary_regime",
    "family_selected_method",
    "effective_weight_method",
    "news_event_active",
    "selected_method",
    "situation_label",
    "situation",
    "failed_break_direction",
    "failed_break_evidence_sha256",
}
_ROW_FIELDS = {
    "regime_state",
    "regime_confidence",
    "selected_family",
    "selected_score",
    "selected_direction",
    "direction_basis",
    "tf_weight",
    "weighted_directional_score",
}
_AGGREGATE_FIELDS = {
    "direction",
    "weighted_directional_score",
    "directional_coverage_weight",
    "selected_family_coverage_weight",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _without_field(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    material = deepcopy(dict(value))
    material.pop(field, None)
    return material


def _finite_number(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed):
        return None
    if minimum is not None and parsed < minimum:
        return None
    if maximum is not None and parsed > maximum:
        return None
    return parsed


def _round(value: float, digits: int = 12) -> float:
    return round(float(value), digits)


def _normalized_regime(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in REGIME_TO_FAMILY:
        return text
    if "BREAKOUT_PENDING" in text or "SQUEEZE" in text:
        return "BREAKOUT_PENDING"
    if "TREND" in text or text.startswith("IMPULSE_"):
        return "TREND_WEAK"
    if "RANGE" in text:
        return "RANGE"
    if text in {"TRANSITION", "FAILURE_RISK", "UNCLEAR"}:
        return "TRANSITION"
    return "UNKNOWN"


def _session_label(chart: Mapping[str, Any]) -> str:
    session = chart.get("session")
    if isinstance(session, Mapping):
        return str(
            session.get("current_tag")
            or session.get("bucket")
            or session.get("name")
            or ""
        ).strip()[:MAX_SESSION_CHARS]
    return str(session or "").strip()[:MAX_SESSION_CHARS]


def _views_by_timeframe(chart: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    views = chart.get("views")
    for view in views if isinstance(views, list) else []:
        if not isinstance(view, Mapping):
            continue
        timeframe = str(view.get("granularity") or "").strip().upper()
        if timeframe in TIMEFRAMES and timeframe not in result:
            result[timeframe] = view
    return result


def _view_regime(view: Mapping[str, Any]) -> tuple[str, float | None]:
    market_state = (
        view.get("market_state")
        if isinstance(view.get("market_state"), Mapping)
        else {}
    )
    phase = str((market_state or {}).get("phase") or "").strip().upper()
    readiness = str((market_state or {}).get("readiness") or "").strip().upper()
    # Only complete producer-owned taxonomy evidence changes method routing.
    # PRE states remain observable rather than becoming a blanket stop:
    # confirmation promotes them to the method capable of trading that
    # transition; an unconfirmed state remains non-directional evidence.
    if (market_state or {}).get("evidence_complete") is True:
        phase_regime = {
            "TREND": "TREND_STRONG",
            "RANGE": "RANGE",
            "PRE_TREND": (
                "TREND_WEAK"
                if readiness in {"TRIGGERED", "ACTIVE"}
                else "BREAKOUT_PENDING"
            ),
            "PRE_RANGE": (
                "RANGE"
                if readiness in {"TRIGGERED", "ACTIVE"}
                else "TRANSITION"
            ),
        }.get(phase)
        if phase_regime is not None:
            confidence = _finite_number(
                (market_state or {}).get("confidence"),
                minimum=0.0,
                maximum=1.0,
            )
            return phase_regime, confidence
    reading = view.get("regime_reading") if isinstance(view.get("regime_reading"), Mapping) else {}
    state = _normalized_regime((reading or {}).get("state") or view.get("regime"))
    confidence = _finite_number((reading or {}).get("confidence"), minimum=0.0, maximum=1.0)
    return state, confidence


def regime_family_state_from_view(
    view: Mapping[str, Any],
) -> tuple[str, float | None]:
    """Return the canonical phase-aware regime used by family weighting.

    Forecast technical context stores the same per-timeframe regime as the
    weighting receipt.  Keeping this raw-view derivation in one function
    prevents the context sibling and receipt from silently diverging when the
    market-state phase model evolves.
    """

    return _view_regime(view)


def _primary_regime(
    views: Mapping[str, Mapping[str, Any]],
    *,
    dominant_regime: Any,
) -> str:
    # Keep the same primary-frame contract as forecast_technical_context:
    # M15 first, then H1, then dominant.  TRANSITION is meaningful and must
    # remain a fail-closed NONE family rather than being skipped in search of a
    # more permissive regime on another timeframe.
    for timeframe in ("M15", "H1"):
        view = views.get(timeframe)
        if not isinstance(view, Mapping):
            continue
        state, _confidence = _view_regime(view)
        if state != "UNKNOWN":
            return state
    fallback = _normalized_regime(dominant_regime)
    return fallback


def _latest_close_confirmed_structure_direction(view: Mapping[str, Any]) -> tuple[str | None, str | None]:
    structure = view.get("structure") if isinstance(view.get("structure"), Mapping) else {}
    latest: tuple[float, str, str] | None = None
    events = (structure or {}).get("structure_events")
    for order, event in enumerate(events if isinstance(events, list) else []):
        if not isinstance(event, Mapping) or event.get("close_confirmed") is not True:
            continue
        kind = str(event.get("kind") or "").strip().upper().split(":", 1)[0]
        if kind.endswith("_UP"):
            direction = "UP"
        elif kind.endswith("_DOWN"):
            direction = "DOWN"
        else:
            continue
        index = _finite_number(event.get("index"))
        sort_key = index if index is not None else float(order)
        if latest is None or sort_key >= latest[0]:
            latest = (sort_key, direction, kind)
    if latest is not None:
        return latest[1], f"CLOSE_CONFIRMED_{latest[2]}"

    families = view.get("family_scores") if isinstance(view.get("family_scores"), Mapping) else {}
    components = (
        families.get("breakout_components")
        if isinstance(families.get("breakout_components"), Mapping)
        else {}
    )
    donchian = _finite_number((components or {}).get("donchian_break"), minimum=-1.0, maximum=1.0)
    if donchian is not None and donchian != 0.0:
        return ("UP" if donchian > 0.0 else "DOWN"), "CLOSE_DONCHIAN_BREAK"
    return None, None


def _selected_family_row(view: Mapping[str, Any], *, weight: float) -> dict[str, Any]:
    state, confidence = _view_regime(view)
    family = REGIME_TO_FAMILY[state]
    families = view.get("family_scores") if isinstance(view.get("family_scores"), Mapping) else {}
    score_key = {
        "TREND": "trend_score",
        "MEAN_REVERSION": "mean_rev_score",
        "BREAKOUT": "breakout_score",
    }.get(family)
    score = (
        _finite_number((families or {}).get(score_key), minimum=-MAX_SCORE_ABS, maximum=MAX_SCORE_ABS)
        if score_key is not None
        else None
    )
    # Direction and weighted arithmetic must use the exact score serialized in
    # the receipt.  Computing from higher precision and only then rounding can
    # make the builder emit an artifact its own verifier rejects (or retain a
    # direction after the stored score rounded to zero).
    stored_score = _round(score, 6) if score is not None else None
    direction = "NONE"
    direction_basis = "REGIME_SELECTS_NONE"
    directional_score = 0.0
    if family in {"TREND", "MEAN_REVERSION"} and stored_score is not None:
        if stored_score > 0.0:
            direction = "UP"
            directional_score = stored_score
        elif stored_score < 0.0:
            direction = "DOWN"
            directional_score = stored_score
        else:
            direction = "EITHER"
        direction_basis = f"{family}_SCORE_SIGN"
    elif family == "BREAKOUT":
        # A pending breakout does not say whether to follow a confirmed break
        # or fade a failed one.  Even a close-confirmed Donchian/BOS direction
        # cannot authorize BREAKOUT_FAILURE in the same direction.  Keep this
        # observation explicitly non-directional until a method-specific
        # continuation/failure detector supplies its own evidence.
        direction = "EITHER"
        direction_basis = "BREAKOUT_METHOD_EVIDENCE_REQUIRED"
    elif family != "NONE":
        direction_basis = "SELECTED_FAMILY_SCORE_MISSING"
    return {
        "regime_state": state,
        "regime_confidence": _round(confidence, 6) if confidence is not None else None,
        "selected_family": family,
        "selected_score": stored_score,
        "selected_direction": direction,
        "direction_basis": direction_basis,
        "tf_weight": _round(weight),
        "weighted_directional_score": _round(weight * directional_score),
    }


def build_regime_family_weighting_receipt(
    pair_chart: Mapping[str, Any] | None,
    *,
    pair: str,
    calendar_path: Path | None = None,
    strategy_profile_path: Path | None = None,
    now_utc: datetime | None = None,
    m5_failed_break_evidence: Mapping[str, Any] | None = None,
    include_policy_evidence: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    chart = pair_chart if isinstance(pair_chart, Mapping) else {}
    pair_text = str(pair or "").strip().upper()
    views = _views_by_timeframe(chart)
    confluence = chart.get("confluence") if isinstance(chart.get("confluence"), Mapping) else {}
    dominant_regime = _normalized_regime(
        (confluence or {}).get("dominant_regime")
        or chart.get("dominant_regime")
        or "UNKNOWN"
    )
    primary_regime = _primary_regime(views, dominant_regime=dominant_regime)
    selected_family = REGIME_TO_FAMILY[primary_regime]
    failed_break_evidence = (
        deepcopy(dict(m5_failed_break_evidence))
        if isinstance(m5_failed_break_evidence, Mapping)
        else build_m5_failed_break_evidence(chart)
    )
    failed_break_valid, _failed_break_error = verify_m5_failed_break_evidence(
        failed_break_evidence
    )
    proof_direction = (
        failed_break_direction(failed_break_evidence)
        if failed_break_valid
        else None
    )
    family_selected_method = (
        "BREAKOUT_FAILURE"
        if proof_direction is not None
        else FAMILY_TO_METHOD[selected_family]
    )
    session = _session_label(chart)
    dynamic_result = dynamic_tf_weights(
        session=session,
        chart_story=str(chart.get("chart_story") or ""),
        dominant_regime=dominant_regime,
        method=family_selected_method,
        pair=pair_text,
        pair_chart=dict(chart),
        calendar_path=calendar_path,
        strategy_profile_path=strategy_profile_path,
        now_utc=now_utc,
        include_trace=True,
        include_evidence=True,
    )
    weights, situation_label, weighting_trace, policy_evidence = dynamic_result
    effective_weight_method = weighting_trace.get("effective_weight_method")
    selected_method = (
        effective_weight_method if family_selected_method is not None else None
    )
    normalized_weights = {
        timeframe: _round(float(weights.get(timeframe, 0.0)))
        for timeframe in TIMEFRAMES
    }
    by_timeframe = {
        timeframe: _selected_family_row(
            views.get(timeframe) or {},
            weight=normalized_weights[timeframe],
        )
        for timeframe in TIMEFRAMES
    }
    weighted_directional_score = sum(
        float(row["weighted_directional_score"])
        for row in by_timeframe.values()
    )
    directional_coverage_weight = sum(
        float(row["tf_weight"])
        for row in by_timeframe.values()
        if row["selected_direction"] in {"UP", "DOWN"}
    )
    selected_family_coverage_weight = sum(
        float(row["tf_weight"])
        for row in by_timeframe.values()
        if row["selected_family"] != "NONE" and row["selected_score"] is not None
    )
    aggregate_direction = (
        "UP"
        if weighted_directional_score > 1e-12
        else "DOWN"
        if weighted_directional_score < -1e-12
        else "EITHER"
        if selected_family_coverage_weight > 0.0
        else "NONE"
    )
    source_identity = {
        "pair": pair_text,
        "chart_generated_at_utc": (
            str(chart.get("generated_at_utc") or "").strip()[:MAX_GENERATED_AT_CHARS]
            or None
        ),
        "session": session or None,
        "dominant_regime": dominant_regime,
        "primary_regime": primary_regime,
        "family_selected_method": family_selected_method,
        "effective_weight_method": effective_weight_method,
        "news_event_active": weighting_trace.get("news_event_active") is True,
        "selected_method": selected_method,
        "situation_label": str(situation_label or "")[:MAX_SITUATION_CHARS],
        "situation": weighting_trace.get("situation"),
        "failed_break_direction": proof_direction,
        "failed_break_evidence_sha256": failed_break_evidence.get(
            "evidence_sha256"
        ),
    }
    receipt: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "source_identity": source_identity,
        "source_identity_sha256": _sha256(source_identity),
        "policy_inputs": weighting_trace,
        "m5_failed_break_evidence": failed_break_evidence,
        "weights": normalized_weights,
        "by_timeframe": by_timeframe,
        "aggregate": {
            "direction": aggregate_direction,
            "weighted_directional_score": _round(weighted_directional_score),
            "directional_coverage_weight": _round(
                min(1.0, max(0.0, directional_coverage_weight))
            ),
            "selected_family_coverage_weight": _round(
                min(1.0, max(0.0, selected_family_coverage_weight))
            ),
        },
    }
    receipt["receipt_sha256"] = _sha256(receipt)
    if include_policy_evidence:
        return receipt, policy_evidence
    return receipt


def verify_regime_family_weighting_receipt(
    value: object,
    *,
    pair: str | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(value, Mapping):
        return False, "REGIME_FAMILY_WEIGHTING_MISSING"
    receipt = dict(value)
    if set(receipt) != _TOP_LEVEL_FIELDS:
        return False, "REGIME_FAMILY_WEIGHTING_SCHEMA_INVALID"
    if (
        receipt.get("contract") != CONTRACT
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("policy_version") != POLICY_VERSION
    ):
        return False, "REGIME_FAMILY_WEIGHTING_SCHEMA_INVALID"
    source = receipt.get("source_identity")
    if not isinstance(source, Mapping) or set(source) != _SOURCE_FIELDS:
        return False, "REGIME_FAMILY_WEIGHTING_SOURCE_INVALID"
    pair_text = str(source.get("pair") or "")
    if pair_text != pair_text.strip().upper() or pair_text not in DEFAULT_TRADER_PAIRS:
        return False, "REGIME_FAMILY_WEIGHTING_PAIR_INVALID"
    if pair is not None and pair_text != str(pair or "").strip().upper():
        return False, "REGIME_FAMILY_WEIGHTING_PAIR_MISMATCH"
    primary_regime = str(source.get("primary_regime") or "")
    if primary_regime not in REGIME_TO_FAMILY:
        return False, "REGIME_FAMILY_WEIGHTING_REGIME_INVALID"
    failed_break_evidence = receipt.get("m5_failed_break_evidence")
    failed_break_valid, failed_break_error = verify_m5_failed_break_evidence(
        failed_break_evidence
    )
    if not failed_break_valid:
        return False, failed_break_error or "M5_FAILED_BREAK_EVIDENCE_INVALID"
    proof_direction = failed_break_direction(failed_break_evidence)
    expected_family_method = (
        "BREAKOUT_FAILURE"
        if proof_direction is not None
        else FAMILY_TO_METHOD[REGIME_TO_FAMILY[primary_regime]]
    )
    if source.get("family_selected_method") != expected_family_method:
        return False, "REGIME_FAMILY_WEIGHTING_METHOD_INVALID"
    if (
        source.get("failed_break_direction") != proof_direction
        or source.get("failed_break_evidence_sha256")
        != failed_break_evidence.get("evidence_sha256")
    ):
        return False, "REGIME_FAMILY_WEIGHTING_FAILED_BREAK_MISMATCH"
    news_event_active = source.get("news_event_active")
    if not isinstance(news_event_active, bool):
        return False, "REGIME_FAMILY_WEIGHTING_SOURCE_INVALID"
    expected_effective_method = (
        "EVENT_RISK" if news_event_active else expected_family_method
    )
    if source.get("effective_weight_method") != expected_effective_method:
        return False, "REGIME_FAMILY_WEIGHTING_METHOD_INVALID"
    expected_selected_method = (
        expected_effective_method if expected_family_method is not None else None
    )
    if source.get("selected_method") != expected_selected_method:
        return False, "REGIME_FAMILY_WEIGHTING_METHOD_INVALID"
    dominant_regime = source.get("dominant_regime")
    if not isinstance(dominant_regime, str) or dominant_regime not in REGIME_TO_FAMILY:
        return False, "REGIME_FAMILY_WEIGHTING_REGIME_INVALID"
    generated_at = source.get("chart_generated_at_utc")
    if generated_at is not None and (
        not isinstance(generated_at, str)
        or not generated_at.strip()
        or generated_at != generated_at.strip()
        or len(generated_at) > MAX_GENERATED_AT_CHARS
    ):
        return False, "REGIME_FAMILY_WEIGHTING_SOURCE_INVALID"
    session = source.get("session")
    if session is not None and (
        not isinstance(session, str)
        or not session.strip()
        or session != session.strip()
        or len(session) > MAX_SESSION_CHARS
    ):
        return False, "REGIME_FAMILY_WEIGHTING_SOURCE_INVALID"
    situation = source.get("situation_label")
    if (
        not isinstance(situation, str)
        or not situation.strip()
        or situation != situation.strip()
        or len(situation) > MAX_SITUATION_CHARS
    ):
        return False, "REGIME_FAMILY_WEIGHTING_SOURCE_INVALID"
    policy_inputs = receipt.get("policy_inputs")
    if not isinstance(policy_inputs, Mapping) or set(policy_inputs) != DYNAMIC_TF_POLICY_FIELDS:
        return False, "REGIME_FAMILY_WEIGHTING_POLICY_INPUTS_INVALID"
    try:
        expected_weights = dynamic_tf_weights_from_policy_inputs(dict(policy_inputs))
    except (TypeError, ValueError, OverflowError):
        return False, "REGIME_FAMILY_WEIGHTING_POLICY_INPUTS_INVALID"
    if (
        policy_inputs.get("contract") != DYNAMIC_TF_POLICY_CONTRACT
        or policy_inputs.get("pair") != pair_text
        or policy_inputs.get("requested_method")
        != source.get("family_selected_method")
        or policy_inputs.get("effective_weight_method")
        != source.get("effective_weight_method")
        or policy_inputs.get("news_event_active")
        is not source.get("news_event_active")
        or policy_inputs.get("situation") != source.get("situation")
    ):
        return False, "REGIME_FAMILY_WEIGHTING_POLICY_SOURCE_MISMATCH"
    try:
        if receipt.get("source_identity_sha256") != _sha256(dict(source)):
            return False, "REGIME_FAMILY_WEIGHTING_SOURCE_SHA_MISMATCH"
        if receipt.get("receipt_sha256") != _sha256(_without_field(receipt, "receipt_sha256")):
            return False, "REGIME_FAMILY_WEIGHTING_SHA_MISMATCH"
    except (TypeError, ValueError, OverflowError):
        return False, "REGIME_FAMILY_WEIGHTING_SHA_MISMATCH"

    weights = receipt.get("weights")
    if not isinstance(weights, Mapping) or set(weights) != set(TIMEFRAMES):
        return False, "REGIME_FAMILY_WEIGHTING_WEIGHTS_INVALID"
    parsed_weights: dict[str, float] = {}
    for timeframe in TIMEFRAMES:
        weight = _finite_number(weights.get(timeframe), minimum=0.0, maximum=1.0)
        if weight is None or weight <= 0.0:
            return False, "REGIME_FAMILY_WEIGHTING_WEIGHTS_INVALID"
        parsed_weights[timeframe] = weight
        if not math.isclose(
            weight,
            _round(expected_weights[timeframe]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False, "REGIME_FAMILY_WEIGHTING_POLICY_WEIGHT_MISMATCH"
    if not math.isclose(sum(parsed_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        return False, "REGIME_FAMILY_WEIGHTING_WEIGHT_SUM_INVALID"

    rows = receipt.get("by_timeframe")
    if not isinstance(rows, Mapping) or set(rows) != set(TIMEFRAMES):
        return False, "REGIME_FAMILY_WEIGHTING_ROWS_INVALID"
    expected_total = 0.0
    directional_coverage = 0.0
    selected_coverage = 0.0
    for timeframe in TIMEFRAMES:
        row = rows.get(timeframe)
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            return False, "REGIME_FAMILY_WEIGHTING_ROW_INVALID"
        state = str(row.get("regime_state") or "")
        family = str(row.get("selected_family") or "")
        direction = str(row.get("selected_direction") or "")
        if state not in REGIME_TO_FAMILY or family != REGIME_TO_FAMILY[state]:
            return False, "REGIME_FAMILY_WEIGHTING_FAMILY_MISMATCH"
        if family not in ACTIVE_FAMILIES or direction not in DIRECTIONS:
            return False, "REGIME_FAMILY_WEIGHTING_ROW_INVALID"
        confidence = row.get("regime_confidence")
        if confidence is not None and _finite_number(confidence, minimum=0.0, maximum=1.0) is None:
            return False, "REGIME_FAMILY_WEIGHTING_ROW_INVALID"
        score = row.get("selected_score")
        parsed_score = (
            _finite_number(score, minimum=-MAX_SCORE_ABS, maximum=MAX_SCORE_ABS)
            if score is not None
            else None
        )
        if score is not None and parsed_score is None:
            return False, "REGIME_FAMILY_WEIGHTING_SCORE_INVALID"
        weight = _finite_number(row.get("tf_weight"), minimum=0.0, maximum=1.0)
        weighted = _finite_number(
            row.get("weighted_directional_score"),
            minimum=-MAX_SCORE_ABS,
            maximum=MAX_SCORE_ABS,
        )
        if weight is None or weighted is None or not math.isclose(
            weight,
            parsed_weights[timeframe],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False, "REGIME_FAMILY_WEIGHTING_WEIGHT_BINDING_INVALID"
        if family in {"TREND", "MEAN_REVERSION"}:
            expected_direction = (
                "UP" if parsed_score is not None and parsed_score > 0.0
                else "DOWN" if parsed_score is not None and parsed_score < 0.0
                else "EITHER" if parsed_score is not None
                else "NONE"
            )
            expected_directional_score = parsed_score or 0.0
        elif family == "BREAKOUT":
            if row.get("direction_basis") != "BREAKOUT_METHOD_EVIDENCE_REQUIRED":
                return False, "REGIME_FAMILY_WEIGHTING_BREAKOUT_DIRECTION_UNPROVEN"
            expected_direction = "EITHER"
            expected_directional_score = 0.0
        else:
            expected_direction = "NONE"
            expected_directional_score = 0.0
            if parsed_score is not None:
                return False, "REGIME_FAMILY_WEIGHTING_SCORE_INVALID"
        if direction != expected_direction or not math.isclose(
            weighted,
            weight * expected_directional_score,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            return False, "REGIME_FAMILY_WEIGHTING_DIRECTION_INVALID"
        expected_total += weighted
        if direction in {"UP", "DOWN"}:
            directional_coverage += weight
        if family != "NONE" and parsed_score is not None:
            selected_coverage += weight

    aggregate = receipt.get("aggregate")
    if not isinstance(aggregate, Mapping) or set(aggregate) != _AGGREGATE_FIELDS:
        return False, "REGIME_FAMILY_WEIGHTING_AGGREGATE_INVALID"
    aggregate_score = _finite_number(
        aggregate.get("weighted_directional_score"),
        minimum=-MAX_SCORE_ABS,
        maximum=MAX_SCORE_ABS,
    )
    aggregate_direction = str(aggregate.get("direction") or "")
    stored_directional_coverage = _finite_number(
        aggregate.get("directional_coverage_weight"),
        minimum=0.0,
        maximum=1.0 + 1e-9,
    )
    stored_selected_coverage = _finite_number(
        aggregate.get("selected_family_coverage_weight"),
        minimum=0.0,
        maximum=1.0 + 1e-9,
    )
    expected_direction = (
        "UP" if expected_total > 1e-12
        else "DOWN" if expected_total < -1e-12
        else "EITHER" if selected_coverage > 0.0
        else "NONE"
    )
    if (
        aggregate_score is None
        or stored_directional_coverage is None
        or stored_selected_coverage is None
        or aggregate_direction != expected_direction
        or not math.isclose(aggregate_score, expected_total, rel_tol=0.0, abs_tol=1e-9)
        or not math.isclose(
            stored_directional_coverage,
            directional_coverage,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            stored_selected_coverage,
            selected_coverage,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        return False, "REGIME_FAMILY_WEIGHTING_AGGREGATE_INVALID"
    return True, None


def verify_regime_family_weighting_context_binding(
    value: object,
    *,
    technical_context: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Bind a receipt to independently stored fields in its parent context.

    A valid internal SHA only proves that a receipt was re-hashed consistently.
    This second check recomputes the regime-selected family row from the
    surrounding technical-context regime/family values, then requires those
    independently hashed siblings to agree with the receipt.  The shared
    weight map is also bound to every row.  Session/situation and external
    source identities are duplicated in the parent context and must match the
    verified raw policy evidence; fresh allocation adds a full rebuild from
    packet-authoritative sources.
    """

    valid, error = verify_regime_family_weighting_receipt(value)
    if not valid or not isinstance(value, Mapping):
        return False, error or "REGIME_FAMILY_WEIGHTING_INVALID"

    identity = technical_context.get("identity")
    regime = technical_context.get("regime")
    families = technical_context.get("families")
    volatility = technical_context.get("volatility")
    context_failed_break = technical_context.get("m5_failed_break_evidence")
    context_policy_evidence = technical_context.get("dynamic_tf_policy_evidence")
    context_policy_source = technical_context.get(
        "dynamic_tf_policy_source_context"
    )
    if not all(
        isinstance(item, Mapping)
        for item in (identity, regime, families, volatility)
    ):
        return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_BINDING_INVALID"
    regime_by_tf = regime.get("by_timeframe")
    family_by_tf = families.get("by_timeframe")
    if not isinstance(regime_by_tf, Mapping) or not isinstance(family_by_tf, Mapping):
        return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_BINDING_INVALID"

    source = value.get("source_identity")
    rows = value.get("by_timeframe")
    weights = value.get("weights")
    policy_inputs = value.get("policy_inputs")
    if not all(
        isinstance(item, Mapping)
        for item in (source, rows, weights, policy_inputs)
    ):
        return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_BINDING_INVALID"
    context_atr = volatility.get("atr_percentile_by_timeframe")
    policy_atr = policy_inputs.get("atr_percentile_by_timeframe")
    if not isinstance(context_atr, Mapping) or not isinstance(policy_atr, Mapping):
        return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_ATR_MISMATCH"
    pair = str(identity.get("pair") or "").strip().upper()
    if source.get("pair") != pair:
        return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_PAIR_MISMATCH"
    if source.get("dominant_regime") != _normalized_regime(regime.get("dominant")):
        return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_REGIME_MISMATCH"
    if source.get("primary_regime") != _normalized_regime(regime.get("primary")):
        return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_PRIMARY_MISMATCH"
    receipt_failed_break = value.get("m5_failed_break_evidence")
    if context_failed_break is not None:
        context_failed_valid, _context_failed_error = verify_m5_failed_break_evidence(
            context_failed_break
        )
        if not context_failed_valid or receipt_failed_break != context_failed_break:
            return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_FAILED_BREAK_MISMATCH"

    if context_policy_evidence is not None:
        policy_evidence_valid, policy_evidence_error = (
            verify_dynamic_tf_policy_evidence(context_policy_evidence)
        )
        if not policy_evidence_valid:
            return False, policy_evidence_error or "DYNAMIC_TF_EVIDENCE_INVALID"
        try:
            expected_policy_inputs, expected_weights, expected_label = (
                derive_dynamic_tf_policy_from_evidence(
                    dict(context_policy_evidence)
                )
            )
        except (TypeError, ValueError, OverflowError):
            return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_POLICY_INVALID"
        classifier_inputs = context_policy_evidence.get("classifier_inputs")
        evidence_atr = context_policy_evidence.get(
            "atr_percentile_by_timeframe"
        )
        if (
            not isinstance(classifier_inputs, Mapping)
            or not isinstance(evidence_atr, Mapping)
            or context_policy_evidence.get("pair") != pair
            or context_policy_evidence.get("requested_method")
            != source.get("family_selected_method")
            or classifier_inputs.get("session") != source.get("session")
            or classifier_inputs.get("dominant_regime")
            != source.get("dominant_regime")
            or dict(policy_inputs) != expected_policy_inputs
            or source.get("situation_label")
            != str(expected_label)[:MAX_SITUATION_CHARS]
        ):
            return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_POLICY_MISMATCH"
        for timeframe in TIMEFRAMES:
            if (
                evidence_atr.get(timeframe) != context_atr.get(timeframe)
                or policy_atr.get(timeframe) != evidence_atr.get(timeframe)
                or weights.get(timeframe)
                != _round(float(expected_weights[timeframe]))
            ):
                return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_POLICY_MISMATCH"

        if context_policy_source is not None:
            news_evidence = context_policy_evidence.get("news_evidence")
            strategy_evidence = context_policy_evidence.get(
                "strategy_profile_evidence"
            )
            if (
                not isinstance(context_policy_source, Mapping)
                or set(context_policy_source)
                != {
                    "classifier_inputs",
                    "news_source",
                    "strategy_profile_source",
                    "pair_chart_row_sha256",
                    "evaluated_at_utc",
                }
                or not isinstance(news_evidence, Mapping)
                or not isinstance(strategy_evidence, Mapping)
                or dict(
                    context_policy_source.get("classifier_inputs") or {}
                )
                != dict(classifier_inputs)
                or dict(context_policy_source.get("news_source") or {})
                != dict(news_evidence.get("source") or {})
                or dict(
                    context_policy_source.get("strategy_profile_source") or {}
                )
                != dict(strategy_evidence.get("source") or {})
                or context_policy_source.get("evaluated_at_utc")
                != news_evidence.get("evaluated_at_utc")
            ):
                return (
                    False,
                    "REGIME_FAMILY_WEIGHTING_CONTEXT_POLICY_SOURCE_MISMATCH",
                )

    score_fields = {
        "TREND": "trend_score",
        "MEAN_REVERSION": "mean_reversion_score",
        "BREAKOUT": "breakout_score",
    }
    parent_timeframes = set(regime_by_tf)
    required_timeframes = (
        TIMEFRAMES
        if parent_timeframes == set(TIMEFRAMES)
        else ("M5", "M15", "H1", "H4")
    )
    for timeframe in required_timeframes:
        row = rows.get(timeframe)
        scores = family_by_tf.get(timeframe)
        if not isinstance(row, Mapping) or not isinstance(scores, Mapping):
            return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_ROW_MISSING"
        if policy_atr.get(timeframe) != context_atr.get(timeframe):
            return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_ATR_MISMATCH"
        expected_state = _normalized_regime(regime_by_tf.get(timeframe))
        expected_family = REGIME_TO_FAMILY[expected_state]
        score_field = score_fields.get(expected_family)
        expected_score = (
            _finite_number(
                scores.get(score_field),
                minimum=-MAX_SCORE_ABS,
                maximum=MAX_SCORE_ABS,
            )
            if score_field is not None
            else None
        )
        expected_score = _round(expected_score, 6) if expected_score is not None else None
        weight = _finite_number(weights.get(timeframe), minimum=0.0, maximum=1.0)
        if weight is None:
            return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_WEIGHT_MISMATCH"
        pseudo_view = {
            "regime_reading": {"state": expected_state},
            "family_scores": {
                "trend_score": scores.get("trend_score"),
                "mean_rev_score": scores.get("mean_reversion_score"),
                "breakout_score": scores.get("breakout_score"),
            },
        }
        expected_row = _selected_family_row(pseudo_view, weight=weight)
        if (
            row.get("regime_state") != expected_state
            or row.get("selected_family") != expected_family
            or row.get("selected_score") != expected_score
            or row.get("selected_direction") != expected_row["selected_direction"]
            or row.get("direction_basis") != expected_row["direction_basis"]
            or row.get("tf_weight") != expected_row["tf_weight"]
            or row.get("weighted_directional_score")
            != expected_row["weighted_directional_score"]
        ):
            return False, "REGIME_FAMILY_WEIGHTING_CONTEXT_ROW_MISMATCH"
    return True, None


def regime_family_weighting_sha256(value: object) -> str | None:
    valid, _error = verify_regime_family_weighting_receipt(value)
    if not valid or not isinstance(value, Mapping):
        return None
    return str(value.get("receipt_sha256") or "") or None


def forecast_direction_consistency(
    receipt: object,
    *,
    forecast_direction: Any,
) -> tuple[bool, str]:
    valid, error = verify_regime_family_weighting_receipt(receipt)
    if not valid or not isinstance(receipt, Mapping):
        return False, error or "REGIME_FAMILY_WEIGHTING_INVALID"
    direction = str(forecast_direction or "").strip().upper()
    aggregate = receipt.get("aggregate") if isinstance(receipt.get("aggregate"), Mapping) else {}
    family_direction = str((aggregate or {}).get("direction") or "")
    if direction in {"UP", "DOWN"} and family_direction in {"UP", "DOWN"}:
        return direction == family_direction, (
            "REGIME_FAMILY_DIRECTION_ALIGNED"
            if direction == family_direction
            else "REGIME_FAMILY_DIRECTION_CONTRADICTS_FORECAST"
        )
    return True, "REGIME_FAMILY_DIRECTION_NON_DIRECTIONAL"
