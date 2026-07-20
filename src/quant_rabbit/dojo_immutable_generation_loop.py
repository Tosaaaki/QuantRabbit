"""Read-only policy for immutable DOJO AI trainer generations.

The economic replay, AI proposal, lineage, and disk/resource contracts live in
separate modules.  This module composes their *state transitions* without
starting a model or runner and without accepting partial economics.

One generation is immutable after its run dispatch is reserved.  While it is
running, the only admissible input is a small health record.  A later
generation can be proposed only after the prior trainer result is terminal and
bound to the candidate-lineage registry.  ``MANUAL_START_ELIGIBLE`` remains a
recommendation: this module never grants process, filesystem, broker, or live
authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_ai_tuning_state import (
    MAX_ATTEMPTS,
    MAX_PROPOSAL_SLOTS,
    verify_state_store,
)
from quant_rabbit.dojo_candidate_lineage_registry import (
    CandidateLineageSnapshot,
    verify_registry,
)
from quant_rabbit.dojo_conveyor_resource_gate import (
    BACKPRESSURE,
    REVIEW_REQUIRED,
    START_ALLOWED,
    plan_conveyor_start,
)


HEALTH_CONTRACT: Final = "QR_DOJO_IMMUTABLE_GENERATION_HEALTH_V1"
MODEL_INVOCATION_MANIFEST_CONTRACT: Final = (
    "QR_DOJO_MODEL_INVOCATION_PRECOMMIT_MANIFEST_V1"
)
DECISION_CONTRACT: Final = "QR_DOJO_IMMUTABLE_GENERATION_DECISION_V1"
SCHEMA_VERSION: Final = 1
MAX_HEALTH_AGE_SECONDS: Final = 300
MAX_MODEL_MANIFEST_AGE_SECONDS: Final = 300
MAX_PROMPT_BYTES: Final = 1_000_000

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}\Z")
_PROCESS_STATES: Final = frozenset(
    {"IDLE", "RESERVED", "RUNNING", "EXITED", "MISSING"}
)
_HEALTH_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "generation_ordinal",
        "dispatch_id",
        "process_state",
        "expected_replay_count",
        "completed_replay_count",
        "failure_artifact_present",
        "terminal_bundle_present",
        "run_dir_bytes",
        "free_bytes",
        "observed_at_utc",
        "economics_fields_present",
    }
)
_MODEL_INVOCATION_MANIFEST_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "registry_id",
        "lineage_prefix",
        "tuning_state_store_tip_sha256",
        "tuning_state_sha256",
        "prior_result_binding_sha256",
        "fixed_envelope_sha256",
        "source_bundle_sha256",
        "scorer_contract",
        "prompt_bytes_sha256",
        "prompt_size_bytes",
        "provider_id",
        "model_id",
        "attempt_ordinal",
        "invocation_ordinal",
        "attempt_budget_remaining",
        "proposal_slot_budget_remaining",
        "prepared_at_utc",
        "manifest_sha256",
    }
)

_NO_AUTHORITY: Final = {
    "model_invocation_allowed": False,
    "runner_invocation_allowed": False,
    "automatic_runner_start_allowed": False,
    "filesystem_write_allowed": False,
    "filesystem_delete_allowed": False,
    "broker_mutation_allowed": False,
    "live_permission": False,
    "order_authority": "NONE",
}
_INVARIANTS: Final = {
    "partial_economics_inspection_allowed": False,
    "same_attempt_retry_allowed": False,
    "candidate_mutation_after_dispatch_allowed": False,
    "next_generation_requires_terminal_bound_result": True,
    "window_cost_denominator_scorer_risk_envelope_mutable": False,
    "duplicate_executable_candidate_allowed": False,
    "large_run_requires_resource_gate": True,
    "large_run_start_is_automatic": False,
}


class DojoImmutableGenerationLoopError(ValueError):
    """The generation state, health record, or resource binding is unsafe."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict canonical JSON bytes."""

    _validate_json(value, "value")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoImmutableGenerationLoopError(
            "value is not canonical JSON"
        ) from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_health_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a deliberately economics-free run-health observation.

    Exact keys are important here.  A caller cannot add balance, P/L, win rate,
    candidate score, or ledger excerpts to a running-generation observation and
    still pass this boundary.
    """

    row = _mapping(value, "health")
    _exact_keys(row, _HEALTH_KEYS, "health")
    if row["contract"] != HEALTH_CONTRACT or row["schema_version"] != 1:
        raise DojoImmutableGenerationLoopError(
            "health contract/version is unsupported"
        )
    ordinal = _optional_positive_integer(
        row["generation_ordinal"], "health.generation_ordinal"
    )
    dispatch_id = _optional_identifier(row["dispatch_id"], "health.dispatch_id")
    process_state = row["process_state"]
    if not isinstance(process_state, str) or process_state not in _PROCESS_STATES:
        raise DojoImmutableGenerationLoopError("health process_state is unsupported")
    expected = _nonnegative_integer(
        row["expected_replay_count"], "health.expected_replay_count"
    )
    completed = _nonnegative_integer(
        row["completed_replay_count"], "health.completed_replay_count"
    )
    if completed > expected:
        raise DojoImmutableGenerationLoopError(
            "health completed replay count exceeds its fixed denominator"
        )
    failure = _boolean(
        row["failure_artifact_present"], "health.failure_artifact_present"
    )
    terminal = _boolean(
        row["terminal_bundle_present"], "health.terminal_bundle_present"
    )
    _nonnegative_integer(row["run_dir_bytes"], "health.run_dir_bytes")
    _nonnegative_integer(row["free_bytes"], "health.free_bytes")
    observed = _utc_text(row["observed_at_utc"], "health.observed_at_utc")
    if row["economics_fields_present"] is not False:
        raise DojoImmutableGenerationLoopError(
            "running-generation health must not expose economics"
        )
    if (ordinal is None) != (dispatch_id is None):
        raise DojoImmutableGenerationLoopError(
            "health generation ordinal and dispatch id must be present together"
        )
    if process_state == "IDLE":
        if ordinal is not None or expected != 0 or completed != 0 or failure or terminal:
            raise DojoImmutableGenerationLoopError(
                "idle health cannot retain a generation, progress, or terminal artifact"
            )
    else:
        if ordinal is None or expected <= 0:
            raise DojoImmutableGenerationLoopError(
                "active health requires a generation and positive fixed denominator"
            )
    if process_state in {"RESERVED", "RUNNING"} and (failure or terminal):
        raise DojoImmutableGenerationLoopError(
            "an active process cannot claim a terminal artifact"
        )
    if terminal and failure:
        raise DojoImmutableGenerationLoopError(
            "success and failure terminal artifacts are mutually exclusive"
        )
    if terminal and completed != expected:
        raise DojoImmutableGenerationLoopError(
            "terminal bundle requires the complete fixed replay denominator"
        )
    normalized = {
        "contract": HEALTH_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "generation_ordinal": ordinal,
        "dispatch_id": dispatch_id,
        "process_state": process_state,
        "expected_replay_count": expected,
        "completed_replay_count": completed,
        "failure_artifact_present": failure,
        "terminal_bundle_present": terminal,
        "run_dir_bytes": row["run_dir_bytes"],
        "free_bytes": row["free_bytes"],
        "observed_at_utc": observed,
        "economics_fields_present": False,
    }
    return normalized


def seal_model_invocation_manifest(
    *,
    tuning_state_events_dir: Path,
    prompt_bytes: bytes,
    provider_id: str,
    model_id: str,
    prepared_at_utc: datetime | str,
) -> dict[str, Any]:
    """Precommit one exact model request against the append-only state tip.

    Returning this seal does not call a model.  The caller must persist it
    before asking :func:`plan_immutable_generation` whether reservation is
    admissible, and the resulting manifest SHA is the model request identity.
    """

    store = _load_verified_state_store(Path(tuning_state_events_dir))
    state = store["latest_state"]
    if state["phase"] != "READY_FOR_MODEL" or _active_attempt(state) is not None:
        raise DojoImmutableGenerationLoopError(
            "model invocation manifest requires a terminal READY_FOR_MODEL state"
        )
    prompt = _prompt_bytes(prompt_bytes)
    provider = _identifier(provider_id, "provider_id")
    model = _identifier(model_id, "model_id")
    prepared = _utc_text(prepared_at_utc, "prepared_at_utc")
    attempts_used, proposal_slots_used = _used_budgets(state)
    body = {
        "contract": MODEL_INVOCATION_MANIFEST_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "registry_id": state["registry_id"],
        "lineage_prefix": state["lineage_prefix"],
        "tuning_state_store_tip_sha256": store["latest_event_sha256"],
        "tuning_state_sha256": state["state_sha256"],
        "prior_result_binding_sha256": canonical_sha256(
            state["last_terminal_result_binding"]
        ),
        "fixed_envelope_sha256": state["fixed_envelope_sha256"],
        "source_bundle_sha256": state["fixed_envelope"][
            "source_bundle_sha256"
        ],
        "scorer_contract": state["fixed_envelope"]["scorer_contract"],
        "prompt_bytes_sha256": hashlib.sha256(prompt).hexdigest(),
        "prompt_size_bytes": len(prompt),
        "provider_id": provider,
        "model_id": model,
        "attempt_ordinal": attempts_used + 1,
        "invocation_ordinal": _model_invocation_count(state) + 1,
        "attempt_budget_remaining": MAX_ATTEMPTS - attempts_used,
        "proposal_slot_budget_remaining": MAX_PROPOSAL_SLOTS
        - proposal_slots_used,
        "prepared_at_utc": prepared,
    }
    if (
        body["attempt_budget_remaining"] <= 0
        or body["proposal_slot_budget_remaining"] <= 0
    ):
        raise DojoImmutableGenerationLoopError(
            "model invocation manifest has no remaining search budget"
        )
    return {**body, "manifest_sha256": canonical_sha256(body)}


def plan_immutable_generation(
    *,
    tuning_state_events_dir: Path,
    lineage_events_dir: Path,
    artifact_root: Path,
    health: Mapping[str, Any],
    decision_at_utc: datetime | str,
    model_invocation_manifest: Mapping[str, Any] | None = None,
    model_prompt_bytes: bytes | None = None,
    resource_gate_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan the next safe inter-run action without performing it."""

    try:
        store = _load_verified_state_store(Path(tuning_state_events_dir))
        state = store["latest_state"]
        lineage = verify_registry(
            Path(lineage_events_dir), artifact_root=Path(artifact_root)
        )
    except ValueError as exc:
        raise DojoImmutableGenerationLoopError(
            "tuning state or lineage registry is invalid"
        ) from exc
    health_row = validate_health_record(health)
    decision_at = _utc_datetime(decision_at_utc, "decision_at_utc")
    _require_fresh_timestamp(
        health_row["observed_at_utc"],
        decision_at=decision_at,
        not_before=state["last_transition_at_utc"],
        max_age_seconds=MAX_HEALTH_AGE_SECONDS,
        field="health.observed_at_utc",
    )
    relation = _lineage_relation(state, lineage)
    attempt = _active_attempt(state)
    _bind_health_to_attempt(health_row, attempt)

    phase = state["phase"]
    resource_decision: dict[str, Any] | None = None
    reasons: list[str] = []
    next_generation_allowed = False
    manual_start_eligible = False

    if phase == "READY_FOR_MODEL":
        if relation != "TERMINAL_RESULT_BOUND":
            raise DojoImmutableGenerationLoopError(
                "next generation requires the exact terminal bound lineage tip"
            )
        manifest = _verify_model_invocation_manifest(
            model_invocation_manifest,
            prompt_bytes=model_prompt_bytes,
            state=state,
            store=store,
            lineage=lineage,
            decision_at=decision_at,
        )
        action = "RESERVE_NEXT_MODEL_INVOCATION"
        next_generation_allowed = True
        model_manifest_sha256: str | None = manifest["manifest_sha256"]
    elif phase == "MODEL_INVOCATION_RESERVED":
        _reject_late_model_manifest(model_invocation_manifest, model_prompt_bytes)
        _require_relation(relation, "TERMINAL_RESULT_BOUND")
        action = "WAIT_FOR_RESERVED_MODEL_RESPONSE"
        model_manifest_sha256 = None
    elif phase == "COLLECTING_PROPOSALS":
        _reject_late_model_manifest(model_invocation_manifest, model_prompt_bytes)
        model_manifest_sha256 = None
        if relation == "TERMINAL_RESULT_BOUND":
            action = "MATERIALIZE_AND_SEAL_NEXT_STUDY"
        elif relation == "NEXT_STUDY_SEALED":
            action = "RESERVE_EXACT_RUN_DISPATCH"
        else:
            raise DojoImmutableGenerationLoopError(
                "proposal phase has an invalid lineage relation"
            )
    elif phase == "RUN_DISPATCH_RESERVED":
        _reject_late_model_manifest(model_invocation_manifest, model_prompt_bytes)
        model_manifest_sha256 = None
        _require_relation(relation, "RUN_AWAITING_RESULT")
        _require_reserved_not_started_health(health_row)
        if resource_gate_request is None:
            action = "WAIT_FOR_RESOURCE_GATE"
            reasons.append("RESOURCE_GATE_REQUEST_REQUIRED")
        else:
            _verify_resource_request_binding(
                state,
                resource_gate_request,
                observed_free_bytes=health_row["free_bytes"],
            )
            resource_decision = plan_conveyor_start(resource_gate_request)
            if resource_decision["status"] == START_ALLOWED:
                action = "MANUAL_START_ELIGIBLE"
                manual_start_eligible = True
            elif resource_decision["status"] == BACKPRESSURE:
                action = "WAIT_FOR_RESOURCES"
                reasons.extend(resource_decision["backpressure_reasons"])
            elif resource_decision["status"] == REVIEW_REQUIRED:
                action = "REVIEW_RESOURCE_EVIDENCE"
                reasons.extend(resource_decision["review_required_reasons"])
            else:  # pragma: no cover - resource gate is already strict
                raise DojoImmutableGenerationLoopError(
                    "resource gate returned an unknown status"
                )
    elif phase == "AWAITING_LINEAGE_RESULT":
        _reject_late_model_manifest(model_invocation_manifest, model_prompt_bytes)
        model_manifest_sha256 = None
        _require_dispatched_health(health_row)
        if relation == "DISPATCHED_RESULT_BOUND":
            action = "ADVANCE_BOUND_TERMINAL_RESULT"
        elif relation != "RUN_AWAITING_RESULT":
            raise DojoImmutableGenerationLoopError(
                "running attempt has an invalid lineage relation"
            )
        elif health_row["terminal_bundle_present"]:
            action = "VALIDATE_AND_BIND_TERMINAL_BUNDLE"
        elif health_row["failure_artifact_present"] or health_row[
            "process_state"
        ] in {"EXITED", "MISSING"}:
            action = "MARK_INCOMPLETE_REVIEW_REQUIRED"
            reasons.append("NO_FREE_RETRY_AFTER_INCOMPLETE_RUN")
        else:
            action = "MONITOR_HEALTH_ONLY"
    elif phase == "REVIEW_REQUIRED":
        _reject_late_model_manifest(model_invocation_manifest, model_prompt_bytes)
        model_manifest_sha256 = None
        action = "HUMAN_REVIEW_REQUIRED"
        if state["terminal_reason"] is not None:
            reasons.append(state["terminal_reason"])
    elif phase in {"EXHAUSTED", "TERMINATED"}:
        _reject_late_model_manifest(model_invocation_manifest, model_prompt_bytes)
        model_manifest_sha256 = None
        action = "STOP"
        if state["terminal_reason"] is not None:
            reasons.append(state["terminal_reason"])
    else:
        raise DojoImmutableGenerationLoopError(
            f"unsupported tuning phase: {phase}"
        )

    attempts_used, proposal_slots_used = _used_budgets(state)
    body = {
        "contract": DECISION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": "WORN_HISTORICAL_TRAIN_ORCHESTRATION_ONLY",
        "decision_at_utc": _utc_text(decision_at, "decision_at_utc"),
        "tuning_state_store_tip_sha256": store["latest_event_sha256"],
        "tuning_state_store_event_count": store["event_count"],
        "tuning_state_sha256": state["state_sha256"],
        "lineage_tip_sha256": lineage.latest_event_sha256,
        "health_sha256": canonical_sha256(health_row),
        "fixed_envelope_sha256": state["fixed_envelope_sha256"],
        "phase": phase,
        "lineage_relation": relation,
        "action": action,
        "model_invocation_manifest_sha256": model_manifest_sha256,
        "reasons": reasons,
        "budget": {
            "attempts_used": attempts_used,
            "attempts_remaining": max(0, MAX_ATTEMPTS - attempts_used),
            "proposal_slots_used": proposal_slots_used,
            "proposal_slots_remaining": max(
                0, MAX_PROPOSAL_SLOTS - proposal_slots_used
            ),
        },
        "permissions": {
            "next_generation_creation_allowed": next_generation_allowed,
            "terminal_bundle_validation_allowed": action
            == "VALIDATE_AND_BIND_TERMINAL_BUNDLE",
            "manual_runner_start_eligible": manual_start_eligible,
            **_NO_AUTHORITY,
        },
        "invariants": dict(_INVARIANTS),
        "resource_gate_decision": resource_decision,
        "limitations": [
            "READ_ONLY_POLICY_NO_PROCESS_START",
            "HEALTH_INPUT_EXCLUDES_PARTIAL_ECONOMICS",
            "TERMINAL_BUNDLE_STILL_REQUIRES_FULL_HANDOFF_AND_LINEAGE_BINDING",
            "LOCAL_LINEAGE_HAS_NO_EXTERNAL_MONOTONIC_WITNESS",
            "LOCAL_STATE_STORE_HAS_NO_EXTERNAL_MONOTONIC_WITNESS",
            "HEALTH_TIMESTAMP_IS_CALLER_SUPPLIED_NOT_ATTESTED",
            "DECISION_TIMESTAMP_IS_CALLER_SUPPLIED_NOT_ATTESTED",
            "MODEL_PROVIDER_ID_IS_REQUESTED_NOT_PROVIDER_ATTESTED",
            "NO_PROOF_PROMOTION_OR_LIVE_AUTHORITY",
        ],
    }
    return {**body, "decision_sha256": canonical_sha256(body)}


def _load_verified_state_store(events_dir: Path) -> dict[str, Any]:
    try:
        snapshot = _mapping(verify_state_store(events_dir), "tuning state store")
        state = _mapping(snapshot["latest_state"], "tuning state store latest state")
        _sha_field(
            snapshot["latest_event_sha256"],
            "tuning state store latest event sha256",
        )
        _positive_integer(snapshot["event_count"], "tuning state store event count")
    except (KeyError, TypeError, ValueError) as exc:
        raise DojoImmutableGenerationLoopError(
            "append-only tuning state store proof is invalid"
        ) from exc
    required_boundary = {
        "automation_ready": True,
        "research_train_only": True,
        "holdout_access_allowed": False,
        "forward_access_allowed": False,
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    if any(snapshot.get(key) != expected for key, expected in required_boundary.items()):
        raise DojoImmutableGenerationLoopError(
            "tuning state store does not prove the research-only boundary"
        )
    return {**snapshot, "latest_state": state}


def _verify_model_invocation_manifest(
    value: Mapping[str, Any] | None,
    *,
    prompt_bytes: bytes | None,
    state: Mapping[str, Any],
    store: Mapping[str, Any],
    lineage: CandidateLineageSnapshot,
    decision_at: datetime,
) -> dict[str, Any]:
    if value is None or prompt_bytes is None:
        raise DojoImmutableGenerationLoopError(
            "READY_FOR_MODEL requires a sealed exact model invocation manifest and prompt bytes"
        )
    row = _mapping(value, "model invocation manifest")
    _exact_keys(
        row,
        _MODEL_INVOCATION_MANIFEST_KEYS,
        "model invocation manifest",
    )
    if (
        row["contract"] != MODEL_INVOCATION_MANIFEST_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
    ):
        raise DojoImmutableGenerationLoopError(
            "model invocation manifest contract/version is unsupported"
        )
    for key in (
        "tuning_state_store_tip_sha256",
        "tuning_state_sha256",
        "prior_result_binding_sha256",
        "fixed_envelope_sha256",
        "source_bundle_sha256",
        "prompt_bytes_sha256",
        "manifest_sha256",
    ):
        _sha_field(row[key], f"model invocation manifest {key}")
    _identifier(row["registry_id"], "model invocation manifest registry_id")
    _identifier(row["lineage_prefix"], "model invocation manifest lineage_prefix")
    _identifier(row["provider_id"], "model invocation manifest provider_id")
    _identifier(row["model_id"], "model invocation manifest model_id")
    _bounded_text(
        row["scorer_contract"], "model invocation manifest scorer_contract"
    )
    prompt = _prompt_bytes(prompt_bytes)
    attempts_used, proposal_slots_used = _used_budgets(state)
    expected = {
        "registry_id": state["registry_id"],
        "lineage_prefix": state["lineage_prefix"],
        "tuning_state_store_tip_sha256": store["latest_event_sha256"],
        "tuning_state_sha256": state["state_sha256"],
        "prior_result_binding_sha256": canonical_sha256(
            state["last_terminal_result_binding"]
        ),
        "fixed_envelope_sha256": state["fixed_envelope_sha256"],
        "source_bundle_sha256": state["fixed_envelope"][
            "source_bundle_sha256"
        ],
        "scorer_contract": state["fixed_envelope"]["scorer_contract"],
        "prompt_bytes_sha256": hashlib.sha256(prompt).hexdigest(),
        "prompt_size_bytes": len(prompt),
        "attempt_ordinal": attempts_used + 1,
        "invocation_ordinal": _model_invocation_count(state) + 1,
        "attempt_budget_remaining": MAX_ATTEMPTS - attempts_used,
        "proposal_slot_budget_remaining": MAX_PROPOSAL_SLOTS
        - proposal_slots_used,
    }
    if any(row.get(key) != expected_value for key, expected_value in expected.items()):
        raise DojoImmutableGenerationLoopError(
            "model invocation manifest drifted from prompt, lineage, store, scorer, envelope, ordinal, or budget"
        )
    for key in (
        "prompt_size_bytes",
        "attempt_ordinal",
        "invocation_ordinal",
        "attempt_budget_remaining",
        "proposal_slot_budget_remaining",
    ):
        _positive_integer(row[key], f"model invocation manifest {key}")
    body = {key: item for key, item in row.items() if key != "manifest_sha256"}
    if row["manifest_sha256"] != canonical_sha256(body):
        raise DojoImmutableGenerationLoopError(
            "model invocation manifest seal mismatch"
        )
    not_before = max(
        _utc_datetime(state["last_transition_at_utc"], "state transition time"),
        _utc_datetime(lineage.latest_event_at_utc, "lineage latest event time"),
    )
    _require_fresh_timestamp(
        row["prepared_at_utc"],
        decision_at=decision_at,
        not_before=not_before,
        max_age_seconds=MAX_MODEL_MANIFEST_AGE_SECONDS,
        field="model invocation manifest prepared_at_utc",
    )
    return dict(row)


def _reject_late_model_manifest(
    manifest: Mapping[str, Any] | None, prompt_bytes: bytes | None
) -> None:
    if manifest is not None or prompt_bytes is not None:
        raise DojoImmutableGenerationLoopError(
            "model invocation manifest is admissible only before reservation"
        )


def _lineage_relation(
    state: Mapping[str, Any], lineage: CandidateLineageSnapshot
) -> str:
    if (
        lineage.registry_id != state["registry_id"]
        or lineage.lineage_prefix != state["lineage_prefix"]
    ):
        raise DojoImmutableGenerationLoopError(
            "tuning state and lineage registry identity diverged"
        )
    studies = lineage.studies
    results = lineage.results
    if not studies:
        raise DojoImmutableGenerationLoopError("lineage has no sealed generation")
    attempt = _active_attempt(state)
    dispatch = attempt["dispatch"] if attempt is not None else None

    if len(studies) == len(results):
        binding = _latest_result_binding(lineage)
        if dispatch is not None and binding["attempt_ordinal"] == attempt[
            "attempt_ordinal"
        ]:
            if (
                dispatch["status"] != "DISPATCHED"
                or binding["study_sha256"] != dispatch["study_sha256"]
            ):
                raise DojoImmutableGenerationLoopError(
                    "bound result does not match the dispatched generation"
                )
            return "DISPATCHED_RESULT_BOUND"
        if binding != state["last_terminal_result_binding"]:
            raise DojoImmutableGenerationLoopError(
                "tuning state is stale relative to terminal lineage result"
            )
        return "TERMINAL_RESULT_BOUND"

    if len(studies) != len(results) + 1:
        raise DojoImmutableGenerationLoopError(
            "lineage contains a forked or incomplete study/result grammar"
        )
    if attempt is None:
        raise DojoImmutableGenerationLoopError(
            "pending lineage study lacks an active tuning attempt"
        )
    latest = studies[-1]
    if latest["attempt_ordinal"] != attempt["attempt_ordinal"]:
        raise DojoImmutableGenerationLoopError(
            "pending lineage study belongs to another generation"
        )
    if dispatch is None:
        return "NEXT_STUDY_SEALED"
    if latest["study_sha256"] != dispatch["study_sha256"]:
        raise DojoImmutableGenerationLoopError(
            "reserved dispatch does not match the pending lineage study"
        )
    return "RUN_AWAITING_RESULT"


def _latest_result_binding(lineage: CandidateLineageSnapshot) -> dict[str, Any]:
    if not lineage.results or not lineage.events:
        raise DojoImmutableGenerationLoopError("lineage lacks a terminal result")
    event = lineage.events[-1]
    if event["event_type"] != "RESULT_BOUND":
        raise DojoImmutableGenerationLoopError(
            "terminal lineage tip is not a result binding"
        )
    result = lineage.results[-1]
    return {
        "registry_id": lineage.registry_id,
        "lineage_prefix": lineage.lineage_prefix,
        "attempt_ordinal": result["attempt_ordinal"],
        "study_sha256": result["study_sha256"],
        "evaluation_sha256": result["evaluation_sha256"],
        "evaluation_artifact_sha256": result["evaluation_artifact_sha256"],
        "evaluation_artifact_size_bytes": result[
            "evaluation_artifact_size_bytes"
        ],
        "result_event_sha256": event["event_sha256"],
        "result_event_sequence": event["sequence"],
        "lineage_tip_sha256": lineage.latest_event_sha256,
    }


def _bind_health_to_attempt(
    health: Mapping[str, Any], attempt: Mapping[str, Any] | None
) -> None:
    dispatch = attempt["dispatch"] if attempt is not None else None
    if dispatch is None:
        if health["process_state"] != "IDLE":
            raise DojoImmutableGenerationLoopError(
                "health names an unreserved generation"
            )
        return
    assert attempt is not None
    if health["generation_ordinal"] != attempt["attempt_ordinal"]:
        raise DojoImmutableGenerationLoopError(
            "health does not bind the reserved generation ordinal"
        )
    if health["dispatch_id"] != dispatch["dispatch_id"]:
        raise DojoImmutableGenerationLoopError(
            "health does not bind the reserved dispatch id"
        )


def _verify_resource_request_binding(
    state: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    observed_free_bytes: int,
) -> None:
    attempt = _active_attempt(state)
    if attempt is None or attempt["dispatch"] is None:
        raise DojoImmutableGenerationLoopError(
            "resource request lacks a reserved dispatch"
        )
    dispatch = attempt["dispatch"]
    prior = state["last_terminal_result_binding"]
    expected_result_binding_sha = canonical_sha256(prior)
    try:
        request_prior_value = request["prior_result"]
        bindings_value = request["fixed_bindings"]
        authority_value = request["authority"]
        resources_value = request["resources"]
    except (KeyError, TypeError) as exc:
        raise DojoImmutableGenerationLoopError(
            "resource request is structurally incomplete"
        ) from exc
    request_prior = _mapping(request_prior_value, "resource request prior result")
    bindings = _mapping(bindings_value, "resource request fixed bindings")
    authority = _mapping(authority_value, "resource request authority")
    resources = _mapping(resources_value, "resource request resources")
    expected = {
        "result_binding_sha256": expected_result_binding_sha,
        "lineage_tip_sha256": prior["lineage_tip_sha256"],
    }
    if any(request_prior.get(key) != value for key, value in expected.items()):
        raise DojoImmutableGenerationLoopError(
            "resource request prior-result binding drifted"
        )
    envelope = state["fixed_envelope"]
    expected_bindings = {
        "corpus_sha256": envelope["window"]["corpus_sha256"],
        # The source-bundle digest covers the scorer and its sealed dependency
        # closure; it is stronger than trusting one Python file in isolation.
        "scorer_sha256": envelope["source_bundle_sha256"],
        "study_sha256": dispatch["study_sha256"],
    }
    if any(bindings.get(key) != value for key, value in expected_bindings.items()):
        raise DojoImmutableGenerationLoopError(
            "resource request changed corpus, scorer closure, or study"
        )
    if resources.get("free_bytes") != observed_free_bytes:
        raise DojoImmutableGenerationLoopError(
            "resource request free bytes are not bound to the health observation"
        )
    if any(
        value is not False
        for key, value in authority.items()
        if key != "order_authority"
    ) or authority.get("order_authority") != "NONE":
        raise DojoImmutableGenerationLoopError(
            "resource request attempts to grant execution authority"
        )


def _active_attempt(state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    attempts = state["attempts"]
    if not attempts:
        return None
    latest = attempts[-1]
    terminal = latest["terminal"]
    if terminal is not None and terminal["status"] == "LINEAGE_RESULT_BOUND":
        return None
    return latest


def _used_budgets(state: Mapping[str, Any]) -> tuple[int, int]:
    attempts_used = int(state["initial_attempts_consumed"]) + len(
        state["attempts"]
    )
    proposal_slots_used = int(state["initial_proposal_slots_consumed"]) + sum(
        invocation["proposal_slot_charge"]
        for item in state["attempts"]
        for invocation in item["invocations"]
    )
    return attempts_used, proposal_slots_used


def _model_invocation_count(state: Mapping[str, Any]) -> int:
    return sum(len(item["invocations"]) for item in state["attempts"])


def _require_reserved_not_started_health(health: Mapping[str, Any]) -> None:
    if (
        health["process_state"] != "RESERVED"
        or health["completed_replay_count"] != 0
        or health["failure_artifact_present"]
        or health["terminal_bundle_present"]
    ):
        raise DojoImmutableGenerationLoopError(
            "reserved dispatch health indicates a start or prior run artifact"
        )


def _require_dispatched_health(health: Mapping[str, Any]) -> None:
    if health["process_state"] not in {"RUNNING", "EXITED", "MISSING"}:
        raise DojoImmutableGenerationLoopError(
            "dispatched generation health does not indicate a started process"
        )


def _require_relation(actual: str, expected: str) -> None:
    if actual != expected:
        raise DojoImmutableGenerationLoopError(
            f"lineage relation must be {expected}, got {actual}"
        )


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoImmutableGenerationLoopError(f"{field} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], field: str) -> None:
    if set(value) != set(expected):
        raise DojoImmutableGenerationLoopError(f"{field} schema mismatch")


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise DojoImmutableGenerationLoopError(f"{field} must be boolean")
    return value


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DojoImmutableGenerationLoopError(
            f"{field} must be a non-negative integer"
        )
    return value


def _positive_integer(value: Any, field: str) -> int:
    result = _nonnegative_integer(value, field)
    if result <= 0:
        raise DojoImmutableGenerationLoopError(f"{field} must be positive")
    return result


def _optional_positive_integer(value: Any, field: str) -> int | None:
    return None if value is None else _positive_integer(value, field)


def _optional_identifier(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise DojoImmutableGenerationLoopError(f"{field} is invalid")
    return value


def _identifier(value: Any, field: str) -> str:
    result = _optional_identifier(value, field)
    if result is None:
        raise DojoImmutableGenerationLoopError(f"{field} is required")
    return result


def _sha_field(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DojoImmutableGenerationLoopError(f"{field} is not SHA-256")
    return value


def _prompt_bytes(value: Any) -> bytes:
    if not isinstance(value, bytes) or not value or len(value) > MAX_PROMPT_BYTES:
        raise DojoImmutableGenerationLoopError(
            "model prompt must be non-empty immutable bytes within the size limit"
        )
    return value


def _bounded_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise DojoImmutableGenerationLoopError(f"{field} is invalid")
    return value


def _utc_text(value: datetime | str, field: str) -> str:
    parsed = _utc_datetime(value, field)
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _utc_datetime(value: datetime | str, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.endswith("Z") and "T" in value:
        try:
            parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
        except ValueError as exc:
            raise DojoImmutableGenerationLoopError(
                f"{field} must be a valid UTC timestamp"
            ) from exc
    else:
        raise DojoImmutableGenerationLoopError(
            f"{field} must be a valid UTC timestamp ending in Z"
        )
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DojoImmutableGenerationLoopError(f"{field} must use UTC")
    return parsed.astimezone(timezone.utc)


def _require_fresh_timestamp(
    observed_at: datetime | str,
    *,
    decision_at: datetime,
    not_before: datetime | str,
    max_age_seconds: int,
    field: str,
) -> None:
    observed = _utc_datetime(observed_at, field)
    lower_bound = _utc_datetime(not_before, f"{field} lower bound")
    if observed < lower_bound:
        raise DojoImmutableGenerationLoopError(
            f"{field} predates the state or lineage it claims to observe"
        )
    if observed > decision_at:
        raise DojoImmutableGenerationLoopError(f"{field} is in the future")
    if (decision_at - observed).total_seconds() > max_age_seconds:
        raise DojoImmutableGenerationLoopError(f"{field} is stale")


def _validate_json(value: Any, field: str, *, depth: int = 0) -> None:
    if depth > 64:
        raise DojoImmutableGenerationLoopError(f"{field} exceeds JSON depth")
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DojoImmutableGenerationLoopError(f"{field} contains non-finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{field}[{index}]", depth=depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DojoImmutableGenerationLoopError(
                    f"{field} contains a non-string key"
                )
            _validate_json(item, f"{field}.{key}", depth=depth + 1)
        return
    raise DojoImmutableGenerationLoopError(f"{field} is not JSON")


__all__ = [
    "DECISION_CONTRACT",
    "HEALTH_CONTRACT",
    "MAX_HEALTH_AGE_SECONDS",
    "MAX_MODEL_MANIFEST_AGE_SECONDS",
    "MODEL_INVOCATION_MANIFEST_CONTRACT",
    "DojoImmutableGenerationLoopError",
    "canonical_json_bytes",
    "canonical_sha256",
    "plan_immutable_generation",
    "seal_model_invocation_manifest",
    "validate_health_record",
]
