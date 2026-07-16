"""Content-addressed forecast/lane bindings for the bounded M15 recovery path.

The ordinary technical-context and lane contracts intentionally keep their
existing semantics.  This module supplies a separate, narrow evidence chain
for a recovery candidate that excludes M1/M5 directional inputs while their
clean indicator tails rebuild.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


FORECAST_CONTRACT = "QR_M15_RECOVERY_FORECAST_V1"
LANE_CONTRACT = "QR_M15_RECOVERY_LANE_V1"
GEOMETRY_CONTRACT = "QR_M15_RECOVERY_GEOMETRY_V1"
RECOVERY_RECEIPT_CONTRACT = "QR_M15_RECOVERY_MICRO_V1"
RECOVERY_RECEIPT_MODE = "M15_RECOVERY_MICRO"
RECOVERY_ORDER_TYPE = "STOP-ENTRY"

_RECOVERY_GEOMETRY_MARKER_KEYS = frozenset(
    {
        "geometry_atr_source_timeframe",
        "geometry_atr_pips",
        "geometry_source_recovery_receipt_sha256",
        "geometry_forecast_binding_sha256",
        "geometry_forecast_target_price",
        "geometry_forecast_invalidation_price",
        "geometry_tp_within_forecast_target",
        "geometry_sl_at_or_beyond_forecast_invalidation",
        "geometry_generic_overwrite_forbidden",
        "geometry_forbidden_overwrite_sources",
        "geometry_current_spread_pips",
    }
)

_HISTORY_FIELDS = (
    "cycle_id",
    "pair",
    "direction",
    "confidence",
    "current_price",
    "invalidation_price",
    "target_price",
    "horizon_min",
    "raw_confidence",
    "calibration_multiplier",
    "up_score",
    "down_score",
    "range_score",
)

_PROOF_FIELDS = (
    "positive_rotation_mode",
    "positive_rotation_confidence_method",
    "positive_rotation_confidence_z",
    "positive_rotation_tp_wins",
    "positive_rotation_tp_trades",
    "positive_rotation_tp_win_rate_lower",
    "positive_rotation_loss_proxy_jpy",
    "positive_rotation_pessimistic_expectancy_jpy",
    "positive_rotation_proof_collection_ready",
    "positive_rotation_proof_collection_mode",
    "positive_rotation_proof_collection_min_trades",
    "positive_rotation_proof_collection_target_trades",
    "positive_rotation_proof_collection_gap_trades",
    "positive_rotation_proof_collection_bootstrap_contract",
    "positive_rotation_proof_collection_evidence_status",
    "positive_rotation_proof_collection_existing_net_trades",
    "positive_rotation_proof_collection_existing_net_wins",
    "positive_rotation_proof_collection_existing_net_losses",
    "positive_rotation_proof_collection_existing_net_jpy",
    "positive_rotation_proof_collection_existing_net_expectancy_jpy",
    "positive_rotation_live_ready",
    "loss_asymmetry_guard_active",
    "loss_asymmetry_guard_mode",
    "loss_asymmetry_guard_relaxed",
    "loss_asymmetry_guard_loss_cap_jpy",
    "loss_asymmetry_guard_base_max_loss_jpy",
    "loss_asymmetry_guard_effective_max_loss_jpy",
    "capture_economics_status",
    "capture_take_profit_scope",
    "capture_take_profit_scope_key",
    "capture_take_profit_exact_vehicle_required",
    "capture_take_profit_vehicle",
    "capture_take_profit_metrics_source",
    "capture_take_profit_trades",
    "capture_take_profit_wins",
    "capture_take_profit_losses",
    "capture_take_profit_expectancy_jpy",
    "capture_take_profit_net_jpy",
    "capture_take_profit_avg_win_jpy",
    "capture_take_profit_avg_loss_jpy",
    "capture_exact_vehicle_net_scope",
    "capture_exact_vehicle_net_scope_key",
    "capture_exact_vehicle_net_vehicle",
    "capture_exact_vehicle_net_metrics_source",
    "capture_exact_vehicle_net_trades",
    "capture_exact_vehicle_net_wins",
    "capture_exact_vehicle_net_losses",
    "capture_exact_vehicle_net_jpy",
    "capture_exact_vehicle_net_expectancy_jpy",
    "capture_exact_vehicle_net_unresolved_realized_trades",
    "capture_avg_win_jpy",
    "capture_avg_loss_jpy",
    "capture_market_close_expectancy_jpy",
    "max_loss_jpy",
    "attach_take_profit_on_fill",
    "tp_execution_mode",
    "tp_target_intent",
    "opportunity_mode",
)


def recovery_claimed(metadata: Mapping[str, Any]) -> bool:
    """Detect any material belonging to the bounded recovery contract.

    Ordinary forecasts intentionally contain empty
    ``forecast_m15_recovery_*`` placeholders, so those values claim recovery
    only when non-empty.  Every ``m15_recovery_*`` field and the recovery-only
    geometry markers are producer material: their presence remains a claim
    even if an upstream mutation changed the value to ``None`` or ``{}``.

    Keeping this predicate next to the content-addressed schemas gives Risk,
    TraderBrain, GPT, and the gateway one downgrade-resistant definition.  A
    partial deletion therefore reaches the recovery validators and fails
    closed instead of silently becoming an ordinary lane.
    """

    if not isinstance(metadata, Mapping):
        return False
    for key in metadata:
        normalized = str(key)
        if normalized.startswith("m15_recovery_"):
            return True
        if normalized in _RECOVERY_GEOMETRY_MARKER_KEYS:
            return True
    if metadata.get("geometry_model") == GEOMETRY_CONTRACT:
        return True
    if metadata.get("tp_target_source") == "M15_RECOVERY_FORECAST_BOUND":
        return True
    for key, value in metadata.items():
        if not str(key).startswith("forecast_m15_recovery_"):
            continue
        if value is None or value == "":
            continue
        if isinstance(value, Mapping) and not value:
            continue
        return True
    return False


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_digest_valid(payload: object, *, digest_key: str) -> bool:
    if not isinstance(payload, Mapping):
        return False
    body = dict(payload)
    digest = body.pop(digest_key, None)
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    try:
        return canonical_sha256(body) == digest
    except (TypeError, ValueError):
        return False


def _finite_number(value: object) -> bool:
    return value.__class__ in {int, float} and math.isfinite(float(value))


def _finite_positive_price(value: object) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _recovery_receipt_body_valid(receipt: object) -> bool:
    """Validate the immutable recovery receipt before trusting its ATR.

    Source replay against current chart/spread remains the responsibility of
    Risk/Gateway.  This local check closes the producer/binding hole where a
    caller could retain the old receipt digest while changing geometry_source.
    """

    if not _content_digest_valid(receipt, digest_key="receipt_sha256"):
        return False
    assert isinstance(receipt, Mapping)
    geometry_source = receipt.get("geometry_source")
    sizing = receipt.get("sizing")
    return bool(
        receipt.get("contract") == RECOVERY_RECEIPT_CONTRACT
        and receipt.get("mode") == RECOVERY_RECEIPT_MODE
        and receipt.get("pair").__class__ is str
        and receipt.get("pair")
        and isinstance(geometry_source, Mapping)
        and geometry_source.get("timeframe") == "M15"
        and _finite_number(geometry_source.get("atr_pips"))
        and float(geometry_source.get("atr_pips")) > 0.0
        and isinstance(sizing, Mapping)
        and sizing.get("max_units") == 999
        and sizing.get("full_size_allowed") is False
        and sizing.get("minimum_units_override") is False
        and receipt.get("live_permission") is False
        and receipt.get("requires_risk_gateway_revalidation") is True
        and receipt.get("manual_position_mutation_allowed") is False
    )


def _geometry_is_forecast_contained(
    *,
    side: str,
    entry: float,
    tp: float,
    sl: float,
    target: float,
    invalidation: float,
) -> bool:
    if side == "LONG":
        return invalidation < entry < tp <= target and sl <= invalidation
    if side == "SHORT":
        return target <= tp < entry < invalidation and sl >= invalidation
    return False


def build_geometry_binding(
    *,
    forecast_binding: Mapping[str, Any],
    recovery_receipt: Mapping[str, Any],
    pair: str,
    side: str,
    entry: float | None,
    tp: float | None,
    sl: float | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Bind final order geometry to the sole M15 ATR and forecast bounds."""

    if (
        not _content_digest_valid(forecast_binding, digest_key="binding_sha256")
        or forecast_binding.get("contract") != FORECAST_CONTRACT
        or not _recovery_receipt_body_valid(recovery_receipt)
        or forecast_binding.get("source_recovery_receipt_sha256")
        != recovery_receipt.get("receipt_sha256")
        or forecast_binding.get("pair") != pair
        or recovery_receipt.get("pair") != pair
        or side not in {"LONG", "SHORT"}
        or not all(_finite_positive_price(value) for value in (entry, tp, sl))
    ):
        return None
    assert entry is not None and tp is not None and sl is not None
    geometry_source = recovery_receipt.get("geometry_source")
    assert isinstance(geometry_source, Mapping)
    atr_pips = float(geometry_source["atr_pips"])
    target = forecast_binding.get("target_price")
    invalidation = forecast_binding.get("invalidation_price")
    expected_direction = "UP" if side == "LONG" else "DOWN"
    if (
        not _finite_positive_price(target)
        or not _finite_positive_price(invalidation)
        or forecast_binding.get("final_direction") != expected_direction
        or forecast_binding.get("geometry_source_timeframe") != "M15"
        or not _geometry_is_forecast_contained(
            side=side,
            entry=float(entry),
            tp=float(tp),
            sl=float(sl),
            target=float(target),
            invalidation=float(invalidation),
        )
        or metadata.get("geometry_model") != GEOMETRY_CONTRACT
        or metadata.get("geometry_atr_source_timeframe") != "M15"
        or metadata.get("geometry_atr_pips") != atr_pips
        or metadata.get("geometry_source_recovery_receipt_sha256")
        != recovery_receipt.get("receipt_sha256")
        or metadata.get("geometry_forecast_binding_sha256")
        != forecast_binding.get("binding_sha256")
        or metadata.get("geometry_forecast_target_price") != target
        or metadata.get("geometry_forecast_invalidation_price") != invalidation
        or metadata.get("geometry_tp_within_forecast_target") is not True
        or metadata.get("geometry_sl_at_or_beyond_forecast_invalidation") is not True
        or metadata.get("geometry_generic_overwrite_forbidden") is not True
    ):
        return None
    body = {
        "contract": GEOMETRY_CONTRACT,
        "source_recovery_receipt_sha256": recovery_receipt.get("receipt_sha256"),
        "forecast_binding_sha256": forecast_binding.get("binding_sha256"),
        "pair": pair,
        "side": side,
        "entry": float(entry),
        "tp": float(tp),
        "sl": float(sl),
        "atr_source_timeframe": "M15",
        "atr_pips": atr_pips,
        "forecast_direction": expected_direction,
        "forecast_target_price": float(target),
        "forecast_invalidation_price": float(invalidation),
        "tp_within_forecast_target": True,
        "sl_at_or_beyond_forecast_invalidation": True,
        "generic_overwrite_forbidden": True,
    }
    try:
        return {**body, "binding_sha256": canonical_sha256(body)}
    except (TypeError, ValueError):
        return None


def forecast_history_identity(source: Mapping[str, Any]) -> dict[str, Any] | None:
    body = {key: source.get(key) for key in _HISTORY_FIELDS}
    if (
        body["cycle_id"].__class__ is not str
        or not body["cycle_id"]
        or body["pair"].__class__ is not str
        or not body["pair"]
        or body["direction"] not in {"UP", "DOWN"}
        or body["horizon_min"].__class__ is not int
        or body["horizon_min"] <= 0
    ):
        return None
    for key in (
        "confidence",
        "current_price",
        "invalidation_price",
        "target_price",
        "raw_confidence",
        "calibration_multiplier",
        "up_score",
        "down_score",
        "range_score",
    ):
        value = body[key]
        if not _finite_number(value):
            return None
        body[key] = float(value)
    return {**body, "identity_sha256": canonical_sha256(body)}


def build_forecast_binding(
    evidence: Mapping[str, Any],
    *,
    cycle_id: str,
) -> dict[str, Any] | None:
    if (
        not _content_digest_valid(evidence, digest_key="evidence_sha256")
        or evidence.get("contract") != FORECAST_CONTRACT
        or evidence.get("final_direction") not in {"UP", "DOWN"}
        or evidence.get("raw_winner") != evidence.get("final_direction")
        or evidence.get("geometry_source_timeframe") != "M15"
        or evidence.get("live_permission") is not False
        or cycle_id.__class__ is not str
        or not cycle_id
    ):
        return None
    scores = evidence.get("component_scores")
    if (
        not isinstance(scores, Mapping)
        or any(key not in scores for key in ("UP", "DOWN", "RANGE"))
        or any(not _finite_number(value) for value in scores.values())
    ):
        return None
    history = forecast_history_identity(
        {
            "cycle_id": cycle_id,
            "pair": evidence.get("pair"),
            "direction": evidence.get("final_direction"),
            "confidence": evidence.get("confidence"),
            "current_price": evidence.get("forecast_current_price"),
            "invalidation_price": evidence.get("invalidation_price"),
            "target_price": evidence.get("target_price"),
            "horizon_min": evidence.get("horizon_min"),
            "raw_confidence": evidence.get("raw_confidence"),
            "calibration_multiplier": evidence.get("calibration_multiplier"),
            "up_score": scores.get("UP"),
            "down_score": scores.get("DOWN"),
            "range_score": scores.get("RANGE"),
        }
    )
    if history is None:
        return None
    body = {
        "contract": FORECAST_CONTRACT,
        "source_recovery_receipt_sha256": evidence.get(
            "source_recovery_receipt_sha256"
        ),
        "producer_evidence_sha256": evidence.get("evidence_sha256"),
        "pair": evidence.get("pair"),
        "chart_generated_at_utc": evidence.get("chart_generated_at_utc"),
        "forecast_cycle_id": cycle_id,
        "forecast_history_identity": history,
        "forecast_current_price": evidence.get("forecast_current_price"),
        "forecast_spread_pips": evidence.get("forecast_spread_pips"),
        "filtered_input_sha256": evidence.get("filtered_input_sha256"),
        "raw_winner": evidence.get("raw_winner"),
        "component_scores": dict(scores),
        "final_direction": evidence.get("final_direction"),
        "raw_confidence": evidence.get("raw_confidence"),
        "calibration_multiplier": evidence.get("calibration_multiplier"),
        "calibration_scope": evidence.get("calibration_scope"),
        "confidence": evidence.get("confidence"),
        "target_price": evidence.get("target_price"),
        "invalidation_price": evidence.get("invalidation_price"),
        "horizon_min": evidence.get("horizon_min"),
        "geometry_source_timeframe": "M15",
        "live_permission": False,
    }
    try:
        return {**body, "binding_sha256": canonical_sha256(body)}
    except (TypeError, ValueError):
        return None


def proof_source_receipt(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    material = {key: metadata.get(key) for key in _PROOF_FIELDS}
    try:
        return {**material, "proof_source_sha256": canonical_sha256(material)}
    except (TypeError, ValueError):
        return None


def build_lane_binding(
    *,
    forecast_binding: Mapping[str, Any],
    pair: str,
    side: str,
    method: str,
    order_type: str,
    entry: float | None,
    tp: float | None,
    sl: float | None,
    producer_units: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    if (
        not _content_digest_valid(forecast_binding, digest_key="binding_sha256")
        or forecast_binding.get("contract") != FORECAST_CONTRACT
        or producer_units.__class__ is not int
        or not 1 <= producer_units <= 999
        or side not in {"LONG", "SHORT"}
        or method != "BREAKOUT_FAILURE"
        or order_type != RECOVERY_ORDER_TYPE
        or not all(_finite_positive_price(value) for value in (entry, tp, sl))
    ):
        return None
    assert entry is not None and tp is not None and sl is not None
    if (side == "LONG" and not sl < entry < tp) or (
        side == "SHORT" and not tp < entry < sl
    ):
        return None
    recovery_receipt = metadata.get("m15_recovery_micro_receipt")
    if not isinstance(recovery_receipt, Mapping):
        return None
    geometry_binding = build_geometry_binding(
        forecast_binding=forecast_binding,
        recovery_receipt=recovery_receipt,
        pair=pair,
        side=side,
        entry=entry,
        tp=tp,
        sl=sl,
        metadata=metadata,
    )
    if geometry_binding is None:
        return None
    proof = proof_source_receipt(metadata)
    if proof is None:
        return None
    body = {
        "contract": LANE_CONTRACT,
        "forecast_binding_sha256": forecast_binding.get("binding_sha256"),
        "pair": pair,
        "side": side,
        "method": method,
        "order_type": order_type,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "producer_units": producer_units,
        "positive_rotation_mode": metadata.get("positive_rotation_mode"),
        "capture_take_profit_scope_key": metadata.get(
            "capture_take_profit_scope_key"
        ),
        "capture_take_profit_vehicle": metadata.get(
            "capture_take_profit_vehicle"
        ),
        "tp_execution_mode": metadata.get("tp_execution_mode"),
        "geometry_binding": geometry_binding,
        "proof_source_receipt": proof,
        "manual_position_mutation_allowed": False,
        "live_permission": False,
    }
    try:
        return {**body, "binding_sha256": canonical_sha256(body)}
    except (TypeError, ValueError):
        return None


def validate_forecast_binding(
    binding: object,
    *,
    recovery_receipt: Mapping[str, Any],
    metadata: Mapping[str, Any],
    history_row: Mapping[str, Any] | None = None,
) -> tuple[bool, str | None]:
    if not _recovery_receipt_body_valid(recovery_receipt):
        return False, "M15_RECOVERY_RECEIPT_BODY_DIGEST_INVALID"
    if not _content_digest_valid(binding, digest_key="binding_sha256"):
        return False, "M15_RECOVERY_FORECAST_BINDING_DIGEST_INVALID"
    assert isinstance(binding, Mapping)
    history = binding.get("forecast_history_identity")
    scores = binding.get("component_scores")
    evidence = metadata.get("forecast_m15_recovery_evidence")
    if (
        binding.get("contract") != FORECAST_CONTRACT
        or binding.get("source_recovery_receipt_sha256")
        != recovery_receipt.get("receipt_sha256")
        or binding.get("pair") != recovery_receipt.get("pair")
        or binding.get("chart_generated_at_utc")
        != recovery_receipt.get("chart_generated_at_utc")
        or binding.get("geometry_source_timeframe") != "M15"
        or binding.get("live_permission") is not False
        or binding.get("raw_winner") != binding.get("final_direction")
        or binding.get("final_direction") not in {"UP", "DOWN"}
        or not isinstance(history, Mapping)
        or not _content_digest_valid(history, digest_key="identity_sha256")
        or not isinstance(scores, Mapping)
        or any(key not in scores for key in ("UP", "DOWN", "RANGE"))
        or any(not _finite_number(value) for value in scores.values())
        or not isinstance(evidence, Mapping)
        or not _content_digest_valid(evidence, digest_key="evidence_sha256")
        or binding.get("producer_evidence_sha256")
        != evidence.get("evidence_sha256")
    ):
        return False, "M15_RECOVERY_FORECAST_BINDING_CONTRACT_INVALID"
    rebuilt = build_forecast_binding(
        evidence,
        cycle_id=str(binding.get("forecast_cycle_id") or ""),
    )
    if rebuilt is None or dict(binding) != rebuilt:
        return False, "M15_RECOVERY_FORECAST_BINDING_REBUILD_MISMATCH"
    rebound_history = forecast_history_identity(
        {
            "cycle_id": binding.get("forecast_cycle_id"),
            "pair": binding.get("pair"),
            "direction": binding.get("final_direction"),
            "confidence": binding.get("confidence"),
            "current_price": binding.get("forecast_current_price"),
            "invalidation_price": binding.get("invalidation_price"),
            "target_price": binding.get("target_price"),
            "horizon_min": binding.get("horizon_min"),
            "raw_confidence": binding.get("raw_confidence"),
            "calibration_multiplier": binding.get("calibration_multiplier"),
            "up_score": scores.get("UP"),
            "down_score": scores.get("DOWN"),
            "range_score": scores.get("RANGE"),
        }
    )
    if rebound_history is None or rebound_history != dict(history):
        return False, "M15_RECOVERY_FORECAST_HISTORY_BINDING_MISMATCH"
    comparisons = {
        "forecast_cycle_id": binding.get("forecast_cycle_id"),
        "forecast_direction": binding.get("final_direction"),
        "forecast_raw_confidence": binding.get("raw_confidence"),
        "forecast_calibration_multiplier": binding.get("calibration_multiplier"),
        "forecast_current_price": binding.get("forecast_current_price"),
        "forecast_target_price": binding.get("target_price"),
        "forecast_invalidation_price": binding.get("invalidation_price"),
        "forecast_horizon_min": binding.get("horizon_min"),
    }
    if any(metadata.get(key) != value for key, value in comparisons.items()):
        return False, "M15_RECOVERY_FORECAST_METADATA_MISMATCH"
    confidence = metadata.get("forecast_confidence")
    binding_confidence = binding.get("confidence")
    if (
        not _finite_number(confidence)
        or not _finite_number(binding_confidence)
        or round(float(binding_confidence), 4) != float(confidence)
    ):
        return False, "M15_RECOVERY_FORECAST_CONFIDENCE_MISMATCH"
    published_scores = metadata.get("forecast_component_scores")
    if not isinstance(published_scores, Mapping) or any(
        not _finite_number(published_scores.get(key))
        or published_scores.get(key) != round(float(value), 4)
        for key, value in scores.items()
    ):
        return False, "M15_RECOVERY_FORECAST_SCORE_MISMATCH"
    if history_row is not None:
        current_history = forecast_history_identity(history_row)
        if current_history is None or current_history != dict(history):
            return False, "M15_RECOVERY_FORECAST_HISTORY_MISMATCH"
    return True, None


def validate_lane_binding(
    binding: object,
    *,
    forecast_binding: Mapping[str, Any],
    pair: str,
    side: str,
    method: str,
    order_type: str,
    entry: float | None,
    tp: float | None,
    sl: float | None,
    current_units: int,
    metadata: Mapping[str, Any],
) -> tuple[bool, str | None]:
    if not _content_digest_valid(binding, digest_key="binding_sha256"):
        return False, "M15_RECOVERY_LANE_BINDING_DIGEST_INVALID"
    assert isinstance(binding, Mapping)
    producer_units = binding.get("producer_units")
    proof = binding.get("proof_source_receipt")
    current_proof = proof_source_receipt(metadata)
    recovery_receipt = metadata.get("m15_recovery_micro_receipt")
    current_geometry = (
        build_geometry_binding(
            forecast_binding=forecast_binding,
            recovery_receipt=recovery_receipt,
            pair=pair,
            side=side,
            entry=entry,
            tp=tp,
            sl=sl,
            metadata=metadata,
        )
        if isinstance(recovery_receipt, Mapping)
        else None
    )
    geometry_binding = binding.get("geometry_binding")
    if (
        binding.get("contract") != LANE_CONTRACT
        or binding.get("forecast_binding_sha256")
        != forecast_binding.get("binding_sha256")
        or binding.get("pair") != pair
        or binding.get("side") != side
        or binding.get("method") != method
        or binding.get("order_type") != order_type
        or method != "BREAKOUT_FAILURE"
        or order_type != RECOVERY_ORDER_TYPE
        or not all(_finite_positive_price(value) for value in (entry, tp, sl))
        or binding.get("entry") != entry
        or binding.get("tp") != tp
        or binding.get("sl") != sl
        or producer_units.__class__ is not int
        or current_units.__class__ is not int
        or not 1 <= current_units <= producer_units <= 999
        or binding.get("positive_rotation_mode")
        != metadata.get("positive_rotation_mode")
        or binding.get("capture_take_profit_scope_key")
        != metadata.get("capture_take_profit_scope_key")
        or binding.get("capture_take_profit_vehicle")
        != metadata.get("capture_take_profit_vehicle")
        or binding.get("tp_execution_mode") != metadata.get("tp_execution_mode")
        or not isinstance(geometry_binding, Mapping)
        or current_geometry is None
        or dict(geometry_binding) != current_geometry
        or not isinstance(proof, Mapping)
        or current_proof is None
        or dict(proof) != current_proof
        or binding.get("manual_position_mutation_allowed") is not False
        or binding.get("live_permission") is not False
    ):
        return False, "M15_RECOVERY_LANE_BINDING_CONTRACT_INVALID"
    assert entry is not None and tp is not None and sl is not None
    if (side == "LONG" and not sl < entry < tp) or (
        side == "SHORT" and not tp < entry < sl
    ):
        return False, "M15_RECOVERY_LANE_GEOMETRY_INVALID"
    return True, None
