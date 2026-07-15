"""Append-only receipts for exact RANGE_RAIL_LIMIT candidate geometry.

``order_intents.json`` is intentionally replaced every cycle.  That makes it
the right current-state packet, but the wrong evidence source for replay: an
old range candidate can no longer be reconstructed without silently borrowing
new ATR, spread, rail, or gateway settings.  This ledger freezes the candidate
that actually existed at intent time.

The receipt is still a *shadow candidate*, not a broker send receipt.  It binds
the forecast, intent, projected dependent orders, and source-artifact hashes,
while keeping activation/latency and the adaptive post-fill lifecycle explicitly
unproven.  A gateway/transaction receipt must be joined before any row can be
called an exact live vehicle.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_rabbit.predictive_scout import (
    predictive_scout_geometry_claimed,
    predictive_scout_metadata_supported,
)
from quant_rabbit.strategy.forecast_technical_context import (
    build_forecast_technical_context_evidence,
)


LEDGER_FILENAME = "range_vehicle_candidate_ledger.jsonl"
SCHEMA_VERSION = "QR_RANGE_VEHICLE_CANDIDATE_RECEIPT_V1"
GATEWAY_PROJECTION_VERSION = "QR_RANGE_GATEWAY_CONTRACT_PROJECTION_V1"
IDENTITY_METADATA_FIELDS = {
    "range_vehicle_candidate_id",
    "range_vehicle_shape_sha256",
    "range_vehicle_gateway_projection_sha256",
    "range_vehicle_candidate_generated_at_utc",
    "range_vehicle_candidate_live_permission",
}
BOUND_LINEAGE_CAVEAT = (
    "the intent geometry is frozen exactly and the forecast row is independently bound, "
    "but mutable source files are post-generation observations rather than proof of the "
    "exact bytes consumed by every upstream builder"
)
UNBOUND_LINEAGE_CAVEAT = (
    "forecast lineage is not independently bound; this non-LIVE_READY shadow is "
    "diagnostic only and must never authorize send or replay promotion"
)


def bind_range_vehicle_candidate_ids(
    results: Sequence[Mapping[str, Any]], *, generated_at_utc: str
) -> int:
    """Embed a stable candidate/shape identity into generated intent metadata.

    The identity excludes itself but includes the generation timestamp and the
    complete pre-binding intent digest.  Retrying the same serialized event is
    idempotent; a later generation can never alias an older candidate merely
    because its price geometry happened to match.  The gateway receives these
    fields inside the selected intent and can bind its final order request back
    to the immutable candidate row.
    """

    bound = 0
    for result in results:
        intent = result.get("intent")
        if not isinstance(intent, dict):
            continue
        metadata = intent.get("metadata")
        if not isinstance(metadata, dict) or not _is_range_rail_candidate(intent):
            continue
        projection = _project_gateway_contract(intent, metadata)
        shape = _candidate_shape(result, intent, metadata, projection)
        shape_sha = _canonical_sha256(shape)
        candidate_id = _canonical_sha256(
            {
                "generated_at_utc": generated_at_utc,
                "forecast_cycle_id": metadata.get("forecast_cycle_id"),
                "lane_id": result.get("lane_id"),
                "vehicle_shape_sha256": shape_sha,
            }
        )
        metadata.update(
            {
                "range_vehicle_candidate_id": candidate_id,
                "range_vehicle_shape_sha256": shape_sha,
                "range_vehicle_gateway_projection_sha256": _canonical_sha256(
                    projection
                ),
                "range_vehicle_candidate_generated_at_utc": generated_at_utc,
                "range_vehicle_candidate_live_permission": False,
            }
        )
        bound += 1
    return bound


def record_range_vehicle_candidates(
    results: Sequence[Mapping[str, Any]],
    *,
    generated_at_utc: str,
    order_intents_path: Path,
    order_intents_serialized: bytes,
    snapshot_path: Path | None,
    pair_charts_path: Path,
    market_context_matrix_path: Path,
    campaign_plan_path: Path,
    strategy_profile_path: Path,
) -> int:
    """Freeze all priced forecast RANGE rail LIMIT candidates once per shape.

    The function validates the complete existing hash chain under an exclusive
    file lock before appending.  A malformed/tampered ledger therefore stops
    the enclosing intent-generation command instead of quietly starting a new
    unverifiable chain.
    """

    _validate_serialized_order_intents(
        results,
        generated_at_utc=generated_at_utc,
        serialized=order_intents_serialized,
    )
    candidate_results = [
        result
        for result in results
        if isinstance(result.get("intent"), Mapping)
        and _is_range_rail_candidate(result["intent"])
    ]
    if not candidate_results:
        return 0
    wanted_forecasts = {
        (
            str((result["intent"].get("metadata") or {}).get("forecast_cycle_id") or ""),
            str(result["intent"].get("pair") or "").upper(),
        )
        for result in candidate_results
    }
    try:
        forecast_lineage_index = _build_forecast_lineage_index(
            data_dir=order_intents_path.parent,
            wanted_forecasts=wanted_forecasts,
            generated_at_utc=generated_at_utc,
        )
    except OSError:
        forecast_lineage_index = {
            key: {"status": "FORECAST_LINEAGE_ARTIFACT_READ_ERROR"}
            for key in wanted_forecasts
        }
    except ValueError as exc:
        error_code = str(exc)
        if error_code == "RANGE_VEHICLE_GENERATED_AT_INVALID":
            raise
        status = {
            "RANGE_VEHICLE_FORECAST_HISTORY_INVALID": (
                "FORECAST_HISTORY_INTEGRITY_INVALID"
            ),
            "RANGE_VEHICLE_FORECAST_RECEIPT_CHAIN_INVALID": (
                "FORECAST_RECEIPT_CHAIN_INVALID"
            ),
        }.get(error_code, "FORECAST_LINEAGE_VALIDATION_INVALID")
        forecast_lineage_index = {
            key: {"status": status} for key in wanted_forecasts
        }
    source_artifacts = _source_artifact_receipts(
        order_intents_path=order_intents_path,
        order_intents_serialized=order_intents_serialized,
        snapshot_path=snapshot_path,
        pair_charts_path=pair_charts_path,
        market_context_matrix_path=market_context_matrix_path,
        campaign_plan_path=campaign_plan_path,
        strategy_profile_path=strategy_profile_path,
    )
    candidate_payloads = [
        _candidate_payload(
            result,
            generated_at_utc=generated_at_utc,
            forecast_lineage_index=forecast_lineage_index,
            source_artifacts=source_artifacts,
        )
        for result in candidate_results
    ]
    if not all(_candidate_payload_integrity_valid(payload) for payload in candidate_payloads):
        raise ValueError("RANGE_VEHICLE_CANDIDATE_PAYLOAD_INVALID")

    ledger_path = order_intents_path.with_name(LEDGER_FILENAME)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with ledger_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            previous_sha, sequence, seen_payload_shas = _validate_existing_chain(handle)
            handle.seek(0, os.SEEK_END)
            for payload in candidate_payloads:
                payload_sha = _canonical_sha256(payload)
                candidate_key = str(payload["candidate"]["candidate_id"])
                existing_payload_sha = seen_payload_shas.get(candidate_key)
                if existing_payload_sha is not None:
                    if existing_payload_sha != payload_sha:
                        raise ValueError(
                            "RANGE_VEHICLE_CANDIDATE_IDENTITY_COLLISION"
                        )
                    continue
                sequence += 1
                body = {
                    "schema_version": SCHEMA_VERSION,
                    "sequence": sequence,
                    "recorded_at_utc": generated_at_utc,
                    "candidate_key": candidate_key,
                    "payload_sha256": payload_sha,
                    "previous_receipt_sha256": previous_sha,
                    "payload": payload,
                }
                receipt = {**body, "receipt_sha256": _canonical_sha256(body)}
                handle.write(_canonical_json(receipt) + "\n")
                previous_sha = receipt["receipt_sha256"]
                seen_payload_shas[candidate_key] = payload_sha
                appended += 1
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return appended


def _validate_serialized_order_intents(
    results: Sequence[Mapping[str, Any]],
    *,
    generated_at_utc: str,
    serialized: bytes,
) -> None:
    try:
        payload = json.loads(serialized)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("RANGE_VEHICLE_ORDER_INTENTS_SERIALIZATION_INVALID") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("generated_at_utc") != generated_at_utc
        or not isinstance(payload.get("results"), list)
        or _canonical_sha256(payload["results"])
        != _canonical_sha256(list(results))
    ):
        raise ValueError("RANGE_VEHICLE_ORDER_INTENTS_SERIALIZATION_MISMATCH")


def validate_range_vehicle_candidate_ledger(path: Path) -> dict[str, Any]:
    """Validate a receipt ledger without mutating it (used by replay/tests)."""

    if not path.exists():
        return {
            "status": "MISSING",
            "rows": 0,
            "last_receipt_sha256": None,
        }
    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            last_sha, sequence, _seen = _validate_existing_chain(handle)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {
        "status": "VALID",
        "rows": sequence,
        "last_receipt_sha256": last_sha,
    }


def _candidate_payload(
    result: Mapping[str, Any],
    *,
    generated_at_utc: str,
    forecast_lineage_index: Mapping[tuple[str, str], Mapping[str, Any]],
    source_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    intent = result.get("intent")
    if not isinstance(intent, Mapping):
        raise ValueError("RANGE_VEHICLE_CANDIDATE_INTENT_INVALID")
    metadata = intent.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("RANGE_VEHICLE_CANDIDATE_METADATA_INVALID")
    if not _is_range_rail_candidate(intent):
        raise ValueError("RANGE_VEHICLE_CANDIDATE_CONTRACT_INVALID")

    pair = str(intent.get("pair") or "").upper()
    side = str(intent.get("side") or "").upper()
    entry = _finite_positive(intent.get("entry"), label="entry")
    take_profit = _finite_positive(intent.get("tp"), label="tp")
    intent_stop = _finite_positive(intent.get("sl"), label="sl")
    support = _finite_positive(metadata.get("range_support"), label="range_support")
    resistance = _finite_positive(
        metadata.get("range_resistance"), label="range_resistance"
    )
    if not pair or side not in {"LONG", "SHORT"} or resistance <= support:
        raise ValueError("RANGE_VEHICLE_CANDIDATE_GEOMETRY_INVALID")
    candidate_status = str(result.get("status") or "")
    live_blocker_codes = list(result.get("live_blocker_codes") or [])
    units = intent.get("units")
    if (
        not isinstance(units, int)
        or isinstance(units, bool)
        or units < 0
        or (candidate_status == "LIVE_READY" and units == 0)
    ):
        raise ValueError("RANGE_VEHICLE_CANDIDATE_UNITS_INVALID")
    if candidate_status == "LIVE_READY" and (
        result.get("risk_allowed") is not True or live_blocker_codes
    ):
        raise ValueError("RANGE_VEHICLE_CANDIDATE_LIVE_READY_CONTRACT_INVALID")
    geometry_ok = (
        intent_stop < support <= entry < take_profit < resistance
        if side == "LONG"
        else intent_stop > resistance >= entry > take_profit > support
    )
    if not geometry_ok:
        raise ValueError("RANGE_VEHICLE_CANDIDATE_GEOMETRY_INVALID")

    prebinding_intent = _intent_without_candidate_identity(intent)
    intent_sha = _canonical_sha256(prebinding_intent)
    forecast_lineage = _forecast_lineage(
        lineage_index=forecast_lineage_index,
        pair=pair,
        cycle_id=str(metadata.get("forecast_cycle_id") or ""),
        metadata=metadata,
    )
    forecast_binding_complete = bool(
        forecast_lineage.get("status") == "BOUND_TO_EMISSION_RECEIPT"
        and forecast_lineage.get("metadata_binding_status")
        == "EXACT_GENERATOR_PROJECTION_MATCH"
    )
    if not forecast_binding_complete and candidate_status == "LIVE_READY":
        status = str(forecast_lineage.get("status") or "UNKNOWN")
        raise ValueError(f"RANGE_VEHICLE_FORECAST_BINDING_INCOMPLETE:{status}")
    artifacts = {key: dict(value) for key, value in source_artifacts.items()}
    gateway_contract = _project_gateway_contract(intent, metadata)
    gateway_sha = _canonical_sha256(gateway_contract)
    shape = _candidate_shape(result, intent, metadata, gateway_contract)
    shape_sha = _canonical_sha256(shape)
    candidate_identity_preimage = {
        "generated_at_utc": generated_at_utc,
        "forecast_cycle_id": metadata.get("forecast_cycle_id"),
        "lane_id": result.get("lane_id"),
        "vehicle_shape_sha256": shape_sha,
    }
    candidate_id = _canonical_sha256(candidate_identity_preimage)
    if (
        metadata.get("range_vehicle_candidate_id") != candidate_id
        or metadata.get("range_vehicle_shape_sha256") != shape_sha
        or metadata.get("range_vehicle_gateway_projection_sha256") != gateway_sha
        or metadata.get("range_vehicle_candidate_generated_at_utc")
        != generated_at_utc
        or metadata.get("range_vehicle_candidate_live_permission") is not False
    ):
        raise ValueError("RANGE_VEHICLE_CANDIDATE_IDENTITY_MISMATCH")
    source_artifact_snapshot_complete = all(
        value.get("status") in {"PRESENT", "SERIALIZED_FOR_ATOMIC_PUBLISH"}
        for value in artifacts.values()
    )
    bound_identity_metadata = {
        key: metadata.get(key) for key in sorted(IDENTITY_METADATA_FIELDS)
    }
    return {
        "kind": "range_vehicle_candidate",
        "read_only": True,
        "live_permission_allowed": False,
        "candidate_contract_status": (
            "SHADOW_NOT_SENT"
            if forecast_binding_complete
            else "UNBOUND_NON_LIVE_SHADOW"
        ),
        "generated_at_utc": generated_at_utc,
        "candidate": {
            "candidate_id": candidate_id,
            "vehicle_shape_sha256": shape_sha,
            "lane_id": str(result.get("lane_id") or ""),
            "status": str(result.get("status") or ""),
            "risk_allowed": result.get("risk_allowed"),
            "live_blocker_codes": list(result.get("live_blocker_codes") or []),
        },
        "vehicle": {
            "pair": pair,
            "side": side,
            "method": "RANGE_ROTATION",
            "order_type": "LIMIT",
            "units": intent.get("units"),
            "entry": entry,
            "take_profit": take_profit,
            "intent_stop_loss": intent_stop,
            "disaster_stop_loss": _optional_finite_positive(
                metadata.get("disaster_sl")
            ),
            "range_support": support,
            "range_resistance": resistance,
            "range_entry_side": metadata.get("range_entry_side"),
            "range_indicator_source": metadata.get("range_indicator_source"),
            "forecast_horizon_min": metadata.get("forecast_horizon_min"),
            "forecast_confidence": metadata.get("forecast_confidence"),
            "m1_atr_pips": metadata.get("m1_atr_pips"),
            "m5_atr_pips": metadata.get("m5_atr_pips"),
            "h4_atr_pips": metadata.get("h4_atr_pips"),
            "target_reward_risk": metadata.get("target_reward_risk"),
            "intent_reward_risk": metadata.get("opportunity_mode_reward_risk"),
        },
        "vehicle_shape": shape,
        "candidate_identity_preimage": candidate_identity_preimage,
        "bound_identity_metadata": bound_identity_metadata,
        "gateway_contract_projection": gateway_contract,
        "gateway_contract_projection_sha256": gateway_sha,
        "gateway_contract_caveat": (
            "projection only; activation latency, final pre-POST repricing, broker acceptance, "
            "fill, dependent-order replacement, adaptive close, financing, and slippage require "
            "gateway/transaction receipts"
        ),
        "prebinding_intent_sha256": intent_sha,
        "forecast_lineage": forecast_lineage,
        "source_artifacts": artifacts,
        "candidate_geometry_frozen": True,
        "forecast_binding_complete": forecast_binding_complete,
        "source_artifact_snapshot_complete": source_artifact_snapshot_complete,
        "exact_generation_input_bytes_proved": False,
        "exact_candidate_lineage_complete": False,
        "lineage_caveat": (
            BOUND_LINEAGE_CAVEAT
            if forecast_binding_complete
            else UNBOUND_LINEAGE_CAVEAT
        ),
    }


def _is_range_rail_candidate(intent: Mapping[str, Any]) -> bool:
    context = intent.get("market_context")
    metadata = intent.get("metadata")
    return bool(
        isinstance(context, Mapping)
        and isinstance(metadata, Mapping)
        and str(context.get("method") or "").upper() == "RANGE_ROTATION"
        and str(intent.get("order_type") or "").upper() == "LIMIT"
        and str(metadata.get("geometry_model") or "").upper()
        == "RANGE_RAIL_LIMIT"
        and str(metadata.get("forecast_direction") or "").upper() == "RANGE"
    )


def _candidate_shape(
    result: Mapping[str, Any],
    intent: Mapping[str, Any],
    metadata: Mapping[str, Any],
    gateway_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "prebinding_intent_sha256": _canonical_sha256(
            _intent_without_candidate_identity(intent)
        ),
        "lane_id": str(result.get("lane_id") or ""),
        "forecast_cycle_id": str(metadata.get("forecast_cycle_id") or ""),
        "pair": str(intent.get("pair") or "").upper(),
        "side": str(intent.get("side") or "").upper(),
        "order_type": str(intent.get("order_type") or "").upper(),
        "units": intent.get("units"),
        "entry": intent.get("entry"),
        "take_profit": intent.get("tp"),
        "intent_stop_loss": intent.get("sl"),
        "disaster_stop_loss": metadata.get("disaster_sl"),
        "range_support": metadata.get("range_support"),
        "range_resistance": metadata.get("range_resistance"),
        "geometry_model": metadata.get("geometry_model"),
        "gateway_contract_projection": dict(gateway_contract),
    }


def _intent_without_candidate_identity(intent: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(intent)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        payload["metadata"] = {
            key: value
            for key, value in metadata.items()
            if key not in IDENTITY_METADATA_FIELDS
        }
    return payload


def _project_gateway_contract(
    intent: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    context = intent.get("market_context")
    method = context.get("method") if isinstance(context, Mapping) else None
    metadata_dict = dict(metadata)
    predictive_scout = predictive_scout_geometry_claimed(
        metadata_dict,
        pair=str(intent.get("pair") or "").upper(),
        side=str(intent.get("side") or "").upper(),
        order_type=str(intent.get("order_type") or "").upper(),
        method=str(method) if method is not None else None,
    )
    predictive_scout_supported = predictive_scout_metadata_supported(metadata_dict)
    initial_sl_on = _env_flag("QR_NEW_ENTRY_INITIAL_SL")
    sl_repair_disabled = _env_flag("QR_TRADER_DISABLE_SL_REPAIR")
    intent_stop_required = bool(
        predictive_scout_supported
        or metadata.get("broker_stop_loss_mode") == "INTENT_SL"
        or (
            metadata.get("campaign_role") == "OANDA_FIREPOWER_ROUTE"
            and metadata.get(
                "positive_rotation_oanda_campaign_firepower_vehicle_match"
            )
            is True
        )
    )
    if initial_sl_on or not sl_repair_disabled or intent_stop_required:
        attached_stop = _optional_finite_positive(intent.get("sl"))
        stop_basis = "INTENT_SL"
    else:
        attached_stop = _optional_finite_positive(metadata.get("disaster_sl"))
        stop_basis = "DISASTER_SL" if attached_stop is not None else "NONE"
    disable_auto_tp = _env_flag("QR_DISABLE_AUTO_TP")
    attach_tp_raw = metadata.get("attach_take_profit_on_fill")
    if isinstance(attach_tp_raw, bool):
        attach_tp_requested = attach_tp_raw
    elif isinstance(attach_tp_raw, str):
        attach_tp_requested = attach_tp_raw.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    else:
        attach_tp_requested = True
    attached_tp = (
        _optional_finite_positive(intent.get("tp"))
        if attach_tp_requested and not disable_auto_tp
        else None
    )
    expiry = (
        str(metadata.get("predictive_scout_expires_at_utc") or "") or None
        if predictive_scout
        else None
    )
    if predictive_scout:
        position_fill = "OPEN_ONLY"
    else:
        raw_position_fill = str(metadata.get("position_fill") or "").upper()
        if raw_position_fill in {
            "DEFAULT",
            "OPEN_ONLY",
            "REDUCE_FIRST",
            "REDUCE_ONLY",
        }:
            position_fill = raw_position_fill
        elif str(metadata.get("position_intent") or "").upper() in {
            "HEDGE",
            "PYRAMID",
        }:
            position_fill = "OPEN_ONLY"
        else:
            position_fill = "DEFAULT"
    return {
        "contract_version": GATEWAY_PROJECTION_VERSION,
        "projection_only": True,
        "activation_at_utc": None,
        "activation_latency_bound": False,
        "time_in_force": "GTD" if predictive_scout else "GTC",
        "gtd_time_utc": expiry,
        "position_fill": position_fill,
        "take_profit_on_fill": attached_tp,
        "take_profit_basis": (
            "INTENT_TP" if attached_tp is not None else "OMITTED_BY_RUNTIME_CONTRACT"
        ),
        "stop_loss_on_fill": attached_stop,
        "stop_loss_basis": stop_basis,
        "filled_position_lifecycle": (
            "STATIC_BROKER_BARRIERS"
            if predictive_scout
            else "ADAPTIVE_TRADER_GUARDIAN_AND_BROKER_BARRIERS"
        ),
        "predictive_scout_contract_supported": predictive_scout_supported,
        "runtime_flags": {
            "QR_DISABLE_AUTO_TP": disable_auto_tp,
            "QR_NEW_ENTRY_INITIAL_SL": initial_sl_on,
            "QR_TRADER_DISABLE_SL_REPAIR": sl_repair_disabled,
        },
    }


def _forecast_lineage(
    *,
    lineage_index: Mapping[tuple[str, str], Mapping[str, Any]],
    pair: str,
    cycle_id: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    technical = metadata.get("forecast_technical_context")
    if not isinstance(technical, Mapping):
        technical = {}
    base = {
        "forecast_cycle_id": cycle_id or None,
        "pair": pair,
        "technical_context_sha256": technical.get("context_sha256"),
        "technical_evidence_sha256": technical.get("evidence_sha256"),
    }
    if not cycle_id:
        return {**base, "status": "MISSING_FORECAST_CYCLE_ID"}
    indexed = dict(lineage_index.get((cycle_id, pair)) or {})
    if not indexed:
        return {**base, "status": "FORECAST_LINEAGE_NOT_INDEXED"}
    row = indexed.pop("forecast_row", None)
    if indexed.get("status") != "BOUND_TO_EMISSION_RECEIPT" or not isinstance(
        row, Mapping
    ):
        return {**base, **indexed}
    mismatches = _forecast_metadata_mismatches(row, metadata)
    row_summary = {
        "timestamp_utc": row.get("timestamp_utc"),
        "direction": row.get("direction"),
        "confidence": row.get("confidence"),
        "current_price": row.get("current_price"),
        "range_low_price": row.get("range_low_price"),
        "range_high_price": row.get("range_high_price"),
        "horizon_min": row.get("horizon_min"),
        "technical_context_sha256": (
            (row.get("technical_context_v1") or {}).get("context_sha256")
            if isinstance(row.get("technical_context_v1"), Mapping)
            else None
        ),
    }
    if mismatches:
        return {
            **base,
            **indexed,
            "status": "FORECAST_METADATA_MISMATCH",
            "metadata_mismatch_fields": mismatches,
            "forecast_row_summary": row_summary,
        }
    return {
        **base,
        **indexed,
        "metadata_binding_status": "EXACT_GENERATOR_PROJECTION_MATCH",
        "forecast_row_summary": row_summary,
    }


def _build_forecast_lineage_index(
    *,
    data_dir: Path,
    wanted_forecasts: set[tuple[str, str]],
    generated_at_utc: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate forecast history/receipt truth once for the whole cycle.

    ``forecast_history.jsonl`` is large in production.  Candidate-by-candidate
    rescans delayed fresh quote use by seconds per RANGE lane.  This streaming
    index preserves full-file validation while retaining only the current
    cycle/pair rows required by the generated packet.
    """

    history_path = data_dir / "forecast_history.jsonl"
    receipt_path = data_dir / "forecast_emission_receipts.jsonl"
    if not history_path.exists() or not receipt_path.exists():
        return {
            key: {"status": "MISSING_FORECAST_LINEAGE_ARTIFACT"}
            for key in wanted_forecasts
        }
    generated_at = _strict_utc_datetime(
        generated_at_utc,
        error_code="RANGE_VEHICLE_GENERATED_AT_INVALID",
    )
    normalized_wanted = {
        (str(cycle_id or ""), str(pair or "").upper())
        for cycle_id, pair in wanted_forecasts
        if cycle_id and pair
    }
    index: dict[tuple[str, str], dict[str, Any]] = {}
    eligible_rows: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    latest_receipts: dict[tuple[str, str], dict[str, Any]] = {}
    # The producer always takes the history lock before the receipt lock.  Use
    # the same order with shared locks so a same-cycle APPEND -> REPLACE cannot
    # be observed halfway through and bound to stale forecast bytes.
    with history_path.open(encoding="utf-8") as history_handle:
        fcntl.flock(history_handle.fileno(), fcntl.LOCK_SH)
        try:
            with receipt_path.open(encoding="utf-8") as receipt_handle:
                fcntl.flock(receipt_handle.fileno(), fcntl.LOCK_SH)
                try:
                    match_counts = {key: 0 for key in normalized_wanted}
                    matching_rows: dict[tuple[str, str], dict[str, Any]] = {}
                    for raw in history_handle:
                        if not raw.strip():
                            continue
                        try:
                            value = json.loads(raw)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                "RANGE_VEHICLE_FORECAST_HISTORY_INVALID"
                            ) from exc
                        if not isinstance(value, dict):
                            raise ValueError("RANGE_VEHICLE_FORECAST_HISTORY_INVALID")
                        key = (
                            str(value.get("cycle_id") or ""),
                            str(value.get("pair") or "").upper(),
                        )
                        if key not in normalized_wanted:
                            continue
                        match_counts[key] += 1
                        if match_counts[key] == 1:
                            matching_rows[key] = value

                    for key in normalized_wanted:
                        count = match_counts[key]
                        if count != 1:
                            index[key] = {
                                "status": "FORECAST_ROW_NOT_UNIQUE",
                                "matching_forecast_rows": count,
                            }
                            continue
                        row = matching_rows[key]
                        row_timestamp = _strict_utc_datetime(
                            row.get("timestamp_utc"),
                            error_code="RANGE_VEHICLE_FORECAST_HISTORY_INVALID",
                        )
                        if row_timestamp > generated_at:
                            raise ValueError("RANGE_VEHICLE_FORECAST_HISTORY_INVALID")
                        eligible_rows[key] = (row, _canonical_sha256(row))

                    previous_sha: str | None = None
                    expected_sequence = 0
                    receipt_count_by_key: dict[tuple[str, str], int] = {}
                    for raw in receipt_handle:
                        if not raw.strip():
                            continue
                        try:
                            receipt = json.loads(raw)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                "RANGE_VEHICLE_FORECAST_RECEIPT_CHAIN_INVALID"
                            ) from exc
                        if not isinstance(receipt, dict):
                            raise ValueError(
                                "RANGE_VEHICLE_FORECAST_RECEIPT_CHAIN_INVALID"
                            )
                        expected_sequence += 1
                        actual_sha = str(receipt.get("receipt_sha256") or "")
                        body = {
                            key: value
                            for key, value in receipt.items()
                            if key != "receipt_sha256"
                        }
                        forecast_at = _strict_utc_datetime(
                            receipt.get("forecast_timestamp_utc"),
                            error_code="RANGE_VEHICLE_FORECAST_RECEIPT_CHAIN_INVALID",
                        )
                        recorded_at = _strict_utc_datetime(
                            receipt.get("recorded_at_utc"),
                            error_code="RANGE_VEHICLE_FORECAST_RECEIPT_CHAIN_INVALID",
                        )
                        key = (
                            str(receipt.get("cycle_id") or ""),
                            str(receipt.get("pair") or "").upper(),
                        )
                        prior_key_receipts = receipt_count_by_key.get(key, 0)
                        operation = receipt.get("operation")
                        if (
                            receipt.get("schema_version")
                            != "QR_FORECAST_EMISSION_RECEIPT_V1"
                            or operation not in {"APPEND", "REPLACE"}
                            or (prior_key_receipts == 0 and operation != "APPEND")
                            or (prior_key_receipts > 0 and operation != "REPLACE")
                            or not key[0]
                            or not key[1]
                            or not _is_sha256(receipt.get("forecast_row_sha256"))
                            or receipt.get("sequence") != expected_sequence
                            or receipt.get("previous_receipt_sha256") != previous_sha
                            or actual_sha != _canonical_sha256(body)
                            or recorded_at < forecast_at
                            or recorded_at > generated_at
                        ):
                            raise ValueError(
                                "RANGE_VEHICLE_FORECAST_RECEIPT_CHAIN_INVALID"
                            )
                        previous_sha = actual_sha
                        receipt_count_by_key[key] = prior_key_receipts + 1
                        if key in normalized_wanted:
                            latest_receipts[key] = receipt
                finally:
                    fcntl.flock(receipt_handle.fileno(), fcntl.LOCK_UN)
        finally:
            fcntl.flock(history_handle.fileno(), fcntl.LOCK_UN)

    for key, (row, forecast_sha) in eligible_rows.items():
        receipt = latest_receipts.get(key)
        if receipt is None:
            index[key] = {
                "status": "FORECAST_EMISSION_RECEIPT_NOT_FOUND",
                "forecast_row_sha256": forecast_sha,
            }
            continue
        if (
            receipt.get("forecast_row_sha256") != forecast_sha
            or receipt.get("forecast_timestamp_utc") != row.get("timestamp_utc")
        ):
            index[key] = {
                "status": "FORECAST_LATEST_EMISSION_RECEIPT_ROW_MISMATCH",
                "forecast_row_sha256": forecast_sha,
                "latest_receipt_row_sha256": receipt.get("forecast_row_sha256"),
                "latest_receipt_sequence": receipt.get("sequence"),
                "latest_receipt_operation": receipt.get("operation"),
            }
            continue
        index[key] = {
            "status": "BOUND_TO_EMISSION_RECEIPT",
            "forecast_row": row,
            "forecast_row_sha256": forecast_sha,
            "forecast_emission_receipt_sha256": receipt.get("receipt_sha256"),
            "forecast_emission_receipt_sequence": receipt.get("sequence"),
            "forecast_emission_operation": receipt.get("operation"),
            "forecast_emission_recorded_at_utc": receipt.get("recorded_at_utc"),
        }
    for key in wanted_forecasts:
        index.setdefault(key, {"status": "MISSING_FORECAST_CYCLE_ID"})
    return index


def _forecast_metadata_mismatches(
    row: Mapping[str, Any], metadata: Mapping[str, Any]
) -> list[str]:
    mismatches: list[str] = []
    if str(metadata.get("forecast_direction") or "").upper() != str(
        row.get("direction") or ""
    ).upper():
        mismatches.append("forecast_direction")
    expected_confidence = _optional_finite_number(row.get("confidence"))
    actual_confidence = _optional_finite_number(metadata.get("forecast_confidence"))
    if (
        expected_confidence is None
        or actual_confidence is None
        or not _numbers_match(actual_confidence, round(expected_confidence, 4))
    ):
        mismatches.append("forecast_confidence")
    for metadata_key, row_key in (
        ("forecast_current_price", "current_price"),
        ("forecast_range_low_price", "range_low_price"),
        ("forecast_range_high_price", "range_high_price"),
        ("forecast_horizon_min", "horizon_min"),
        ("range_support", "range_low_price"),
        ("range_resistance", "range_high_price"),
    ):
        if not _numbers_match(metadata.get(metadata_key), row.get(row_key)):
            mismatches.append(metadata_key)
    technical_context = row.get("technical_context_v1")
    expected_technical = build_forecast_technical_context_evidence(
        technical_context if isinstance(technical_context, dict) else None,
        pair=str(row.get("pair") or "").upper() or None,
        current_price=_optional_finite_number(row.get("current_price")),
    )
    actual_technical = metadata.get("forecast_technical_context")
    if not isinstance(actual_technical, Mapping) or _canonical_sha256(
        dict(actual_technical)
    ) != _canonical_sha256(expected_technical):
        mismatches.append("forecast_technical_context")
    return mismatches


def _candidate_payload_integrity_valid(payload: Mapping[str, Any]) -> bool:
    """Rebuild the immutable candidate identity from ledger-contained bytes."""

    try:
        candidate = payload.get("candidate")
        vehicle = payload.get("vehicle")
        shape = payload.get("vehicle_shape")
        identity_preimage = payload.get("candidate_identity_preimage")
        bound_identity = payload.get("bound_identity_metadata")
        gateway_projection = payload.get("gateway_contract_projection")
        forecast_lineage = payload.get("forecast_lineage")
        if not all(
            isinstance(value, Mapping)
            for value in (
                candidate,
                vehicle,
                shape,
                identity_preimage,
                bound_identity,
                gateway_projection,
                forecast_lineage,
            )
        ):
            return False
        assert isinstance(candidate, Mapping)
        assert isinstance(vehicle, Mapping)
        assert isinstance(shape, Mapping)
        assert isinstance(identity_preimage, Mapping)
        assert isinstance(bound_identity, Mapping)
        assert isinstance(gateway_projection, Mapping)
        assert isinstance(forecast_lineage, Mapping)

        intent_sha = payload.get("prebinding_intent_sha256")
        if not _is_sha256(intent_sha):
            return False
        shape_sha = _canonical_sha256(dict(shape))
        gateway_sha = _canonical_sha256(dict(gateway_projection))
        candidate_id = str(candidate.get("candidate_id") or "")
        generated_at_utc = payload.get("generated_at_utc")
        _strict_utc_datetime(
            generated_at_utc,
            error_code="RANGE_VEHICLE_CANDIDATE_LEDGER_INVALID",
        )
        expected_identity_preimage = {
            "generated_at_utc": generated_at_utc,
            "forecast_cycle_id": shape.get("forecast_cycle_id"),
            "lane_id": shape.get("lane_id"),
            "vehicle_shape_sha256": shape_sha,
        }
        expected_bound_identity = {
            "range_vehicle_candidate_generated_at_utc": generated_at_utc,
            "range_vehicle_candidate_id": candidate_id,
            "range_vehicle_candidate_live_permission": False,
            "range_vehicle_gateway_projection_sha256": gateway_sha,
            "range_vehicle_shape_sha256": shape_sha,
        }
        units = shape.get("units")
        candidate_status = candidate.get("status")
        side = str(shape.get("side") or "")
        entry = _finite_positive(shape.get("entry"), label="entry")
        take_profit = _finite_positive(
            shape.get("take_profit"), label="take_profit"
        )
        intent_stop = _finite_positive(
            shape.get("intent_stop_loss"), label="intent_stop_loss"
        )
        support = _finite_positive(shape.get("range_support"), label="range_support")
        resistance = _finite_positive(
            shape.get("range_resistance"), label="range_resistance"
        )
        geometry_ok = (
            intent_stop < support <= entry < take_profit < resistance
            if side == "LONG"
            else intent_stop > resistance >= entry > take_profit > support
            if side == "SHORT"
            else False
        )
        runtime_flags = gateway_projection.get("runtime_flags")
        if not isinstance(runtime_flags, Mapping):
            return False
        take_profit_basis = gateway_projection.get("take_profit_basis")
        projected_take_profit = gateway_projection.get("take_profit_on_fill")
        take_profit_projection_ok = (
            projected_take_profit == take_profit
            if take_profit_basis == "INTENT_TP"
            else projected_take_profit is None
            and take_profit_basis == "OMITTED_BY_RUNTIME_CONTRACT"
        )
        stop_basis = gateway_projection.get("stop_loss_basis")
        projected_stop = gateway_projection.get("stop_loss_on_fill")
        disaster_stop = shape.get("disaster_stop_loss")
        stop_projection_ok = (
            projected_stop == intent_stop
            if stop_basis == "INTENT_SL"
            else projected_stop == disaster_stop
            if stop_basis == "DISASTER_SL"
            else projected_stop is None and stop_basis == "NONE"
        )
        forecast_binding_value = payload.get("forecast_binding_complete")
        if not isinstance(forecast_binding_value, bool):
            return False
        forecast_binding_complete = forecast_binding_value
        bound_lineage = bool(
            forecast_lineage.get("status") == "BOUND_TO_EMISSION_RECEIPT"
            and forecast_lineage.get("metadata_binding_status")
            == "EXACT_GENERATOR_PROJECTION_MATCH"
        )
        binding_state_valid = bool(
            (
                forecast_binding_complete
                and bound_lineage
                and payload.get("candidate_contract_status") == "SHADOW_NOT_SENT"
            )
            or (
                not forecast_binding_complete
                and not bound_lineage
                and payload.get("candidate_contract_status")
                == "UNBOUND_NON_LIVE_SHADOW"
                and candidate.get("status") != "LIVE_READY"
            )
        )
        expected_lineage_caveat = (
            BOUND_LINEAGE_CAVEAT
            if forecast_binding_complete
            else UNBOUND_LINEAGE_CAVEAT
        )
        candidate_blockers = candidate.get("live_blocker_codes")
        live_ready_state_valid = bool(
            isinstance(candidate_blockers, list)
            and (
                candidate.get("status") != "LIVE_READY"
                or (
                    candidate.get("risk_allowed") is True
                    and not candidate_blockers
                )
            )
        )
        return bool(
            payload.get("kind") == "range_vehicle_candidate"
            and payload.get("read_only") is True
            and payload.get("live_permission_allowed") is False
            and payload.get("candidate_geometry_frozen") is True
            and binding_state_valid
            and live_ready_state_valid
            and payload.get("exact_generation_input_bytes_proved") is False
            and payload.get("exact_candidate_lineage_complete") is False
            and payload.get("lineage_caveat") == expected_lineage_caveat
            and shape.get("prebinding_intent_sha256") == intent_sha
            and _is_sha256(shape_sha)
            and _is_sha256(gateway_sha)
            and _is_sha256(candidate_id)
            and candidate.get("vehicle_shape_sha256") == shape_sha
            and payload.get("gateway_contract_projection_sha256") == gateway_sha
            and dict(identity_preimage) == expected_identity_preimage
            and candidate_id == _canonical_sha256(expected_identity_preimage)
            and dict(bound_identity) == expected_bound_identity
            and candidate.get("lane_id") == shape.get("lane_id")
            and shape.get("gateway_contract_projection") == gateway_projection
            and isinstance(units, int)
            and not isinstance(units, bool)
            and units >= 0
            and (candidate_status != "LIVE_READY" or units > 0)
            and vehicle.get("units") == shape.get("units")
            and vehicle.get("pair") == shape.get("pair")
            and vehicle.get("side") == shape.get("side")
            and vehicle.get("order_type") == shape.get("order_type")
            and vehicle.get("method") == "RANGE_ROTATION"
            and shape.get("geometry_model") == "RANGE_RAIL_LIMIT"
            and isinstance(shape.get("forecast_cycle_id"), str)
            and bool(shape.get("forecast_cycle_id"))
            and isinstance(shape.get("lane_id"), str)
            and bool(shape.get("lane_id"))
            and isinstance(shape.get("pair"), str)
            and shape.get("pair") == str(shape.get("pair")).upper()
            and shape.get("order_type") == "LIMIT"
            and geometry_ok
            and vehicle.get("entry") == shape.get("entry")
            and vehicle.get("take_profit") == shape.get("take_profit")
            and vehicle.get("intent_stop_loss") == shape.get("intent_stop_loss")
            and vehicle.get("disaster_stop_loss") == shape.get("disaster_stop_loss")
            and vehicle.get("range_support") == shape.get("range_support")
            and vehicle.get("range_resistance") == shape.get("range_resistance")
            and forecast_lineage.get("forecast_cycle_id")
            == shape.get("forecast_cycle_id")
            and forecast_lineage.get("pair") == shape.get("pair")
            and gateway_projection.get("contract_version")
            == GATEWAY_PROJECTION_VERSION
            and gateway_projection.get("projection_only") is True
            and gateway_projection.get("activation_at_utc") is None
            and gateway_projection.get("activation_latency_bound") is False
            and gateway_projection.get("time_in_force") in {"GTC", "GTD"}
            and gateway_projection.get("position_fill")
            in {"DEFAULT", "OPEN_ONLY", "REDUCE_FIRST", "REDUCE_ONLY"}
            and take_profit_projection_ok
            and stop_projection_ok
            and gateway_projection.get("filled_position_lifecycle")
            in {
                "STATIC_BROKER_BARRIERS",
                "ADAPTIVE_TRADER_GUARDIAN_AND_BROKER_BARRIERS",
            }
            and set(runtime_flags)
            == {
                "QR_DISABLE_AUTO_TP",
                "QR_NEW_ENTRY_INITIAL_SL",
                "QR_TRADER_DISABLE_SL_REPAIR",
            }
            and all(isinstance(value, bool) for value in runtime_flags.values())
        )
    except (TypeError, ValueError):
        return False


def _validate_existing_chain(handle) -> tuple[str | None, int, dict[str, str]]:
    handle.seek(0)
    previous_sha: str | None = None
    sequence = 0
    seen_payload_shas: dict[str, str] = {}
    for raw in handle:
        if not raw.strip():
            continue
        try:
            receipt = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("RANGE_VEHICLE_CANDIDATE_LEDGER_INVALID") from exc
        if not isinstance(receipt, dict):
            raise ValueError("RANGE_VEHICLE_CANDIDATE_LEDGER_INVALID")
        sequence += 1
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        payload = receipt.get("payload")
        candidate_key = str(receipt.get("candidate_key") or "")
        payload_candidate = (
            payload.get("candidate") if isinstance(payload, dict) else None
        )
        payload_candidate_id = (
            str(payload_candidate.get("candidate_id") or "")
            if isinstance(payload_candidate, dict)
            else ""
        )
        payload_sha = _canonical_sha256(payload) if isinstance(payload, dict) else None
        if (
            receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("sequence") != sequence
            or receipt.get("previous_receipt_sha256") != previous_sha
            or not isinstance(payload, dict)
            or receipt.get("payload_sha256") != payload_sha
            or receipt.get("receipt_sha256") != _canonical_sha256(body)
            or not candidate_key
            or candidate_key != payload_candidate_id
            or not _candidate_payload_integrity_valid(payload)
            or candidate_key in seen_payload_shas
        ):
            raise ValueError("RANGE_VEHICLE_CANDIDATE_LEDGER_INVALID")
        previous_sha = str(receipt["receipt_sha256"])
        seen_payload_shas[candidate_key] = str(payload_sha)
    return previous_sha, sequence, seen_payload_shas


def _source_artifact_receipts(
    *,
    order_intents_path: Path,
    order_intents_serialized: bytes,
    snapshot_path: Path | None,
    pair_charts_path: Path,
    market_context_matrix_path: Path,
    campaign_plan_path: Path,
    strategy_profile_path: Path,
) -> dict[str, dict[str, Any]]:
    return {
        "order_intents": {
            "path": str(order_intents_path),
            "status": "SERIALIZED_FOR_ATOMIC_PUBLISH",
            "sha256": hashlib.sha256(order_intents_serialized).hexdigest(),
            "byte_length": len(order_intents_serialized),
            "capture_basis": "IN_MEMORY_SERIALIZATION_BEFORE_ATOMIC_PUBLISH",
        },
        "broker_snapshot": _artifact_receipt(snapshot_path),
        "pair_charts": _artifact_receipt(pair_charts_path),
        "market_context_matrix": _artifact_receipt(market_context_matrix_path),
        "campaign_plan": _artifact_receipt(campaign_plan_path),
        "strategy_profile": _artifact_receipt(strategy_profile_path),
        "intent_generator_source": _artifact_receipt(
            Path(__file__).with_name("intent_generator.py")
        ),
        "gateway_execution_source": _artifact_receipt(
            Path(__file__).parents[1] / "broker" / "execution.py"
        ),
        "candidate_ledger_source": _artifact_receipt(Path(__file__)),
    }


def _artifact_receipt(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "status": "MISSING", "sha256": None}
    if not path.is_file():
        return {"path": str(path), "status": "MISSING", "sha256": None}
    digest = _file_sha256(path)
    return {
        "path": str(path),
        "status": "PRESENT",
        "sha256": digest,
        "byte_length": path.stat().st_size,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_utc_datetime(value: Any, *, error_code: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error_code)
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(error_code) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(error_code)
    return parsed.astimezone(timezone.utc)


def _optional_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _numbers_match(left: Any, right: Any) -> bool:
    left_number = _optional_finite_number(left)
    right_number = _optional_finite_number(right)
    if left_number is None or right_number is None:
        return False
    tolerance = max(1e-12, abs(right_number) * 1e-12)
    return abs(left_number - right_number) <= tolerance


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }


def _finite_positive(value: Any, *, label: str) -> float:
    parsed = _optional_finite_positive(value)
    if parsed is None:
        raise ValueError(f"RANGE_VEHICLE_{label.upper()}_INVALID")
    return parsed


def _optional_finite_positive(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
