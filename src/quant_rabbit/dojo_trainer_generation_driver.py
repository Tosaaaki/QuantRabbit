"""One-way, read-only driver for the next immutable DOJO trainer generation.

The driver deliberately stops at sealed hand-off boundaries.  It can derive a
model reservation transition and, after an external caller supplies exact
response bytes, derive the next sealed TRAIN study.  It never appends the
transition, calls a model, writes the study, starts replay, or touches a broker.

Only a complete trainer packet made from the latest terminal, lineage-bound
evaluation is admitted.  Running/partial economics are therefore outside this
API.  The exact prompt artifact, requested provider/model, scorer, corpus,
source-code closure, fixed envelope, fixed cell denominator, lineage tip, and
remaining search budgets are sealed before a reservation can be derived.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_ai_trainer_packet import (
    canonical_packet_bytes,
    verify_trainer_packet,
)
from quant_rabbit.dojo_ai_tuning_state import (
    MAX_ATTEMPTS,
    MAX_PROPOSAL_SLOTS,
    record_model_response,
    reserve_model_invocation,
    status_artifact,
    verify_state_store,
    verify_tuning_state,
)
from quant_rabbit.dojo_bot_catalog import catalog_manifest
from quant_rabbit.dojo_candidate_lineage_registry import (
    CandidateLineageSnapshot,
    verify_registry,
)
from quant_rabbit.dojo_immutable_generation_loop import (
    canonical_sha256,
    plan_immutable_generation,
    seal_model_invocation_manifest,
)
from quant_rabbit.dojo_next_study_builder import (
    PROMPT_ARTIFACT_CONTRACT,
    UNVERIFIED_MODEL_CLAIM,
    UNVERIFIED_PROVIDER_ATTESTATION,
    materialize_next_study,
)


DRIVER_CONTRACT: Final = "QR_DOJO_TRAINER_GENERATION_DRIVER_V1"
RESPONSE_PLAN_CONTRACT: Final = "QR_DOJO_TRAINER_RESPONSE_STUDY_PLAN_V1"
SCHEMA_VERSION: Final = 1
MAX_PROMPT_TEXT_BYTES: Final = 256 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_BINDING_KEYS: Final = frozenset(
    {
        "registry_id",
        "lineage_prefix",
        "attempt_ordinal",
        "study_sha256",
        "run_sha256",
        "evaluation_sha256",
        "evaluation_artifact_sha256",
        "lineage_result_event_sha256",
        "lineage_tip_sha256",
        "tuning_state_sha256",
        "fixed_envelope_sha256",
        "external_witness_status",
        "exact_result_binding_verified",
    }
)
_FIXED_ENVIRONMENT_KEYS: Final = frozenset(
    {
        "window_role",
        "window",
        "initial_balance_jpy",
        "trade_pairs",
        "feed_pairs",
        "intrabar_paths",
        "cost_arms",
        "thresholds",
        "catalog",
    }
)
_SEARCH_BUDGET_KEYS: Final = frozenset(
    {
        "phase",
        "attempts_consumed",
        "attempts_remaining",
        "max_attempts",
        "proposal_slots_consumed",
        "proposal_slots_remaining",
        "max_proposal_slots",
        "invalid_proposal_count",
        "duplicate_proposal_count",
    }
)
_NO_AUTHORITY: Final = {
    "model_invocation_allowed": False,
    "runner_invocation_allowed": False,
    "state_store_write_allowed": False,
    "filesystem_write_allowed": False,
    "filesystem_delete_allowed": False,
    "broker_mutation_allowed": False,
    "live_permission": False,
    "order_authority": "NONE",
}
_REQUIRED_STORE_BOUNDARY: Final = {
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
_PROMPT_AUTHORITY: Final = {
    "proof_eligible": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
}


class DojoTrainerGenerationDriverError(ValueError):
    """The requested trainer hand-off crosses an immutable boundary."""


@dataclass(frozen=True)
class TrainerGenerationPlan:
    """Sealed artifacts derived without performing an external side effect."""

    receipt: dict[str, Any]
    model_request_bytes: bytes
    prompt_artifact_bytes: bytes
    reservation_state: dict[str, Any] | None


@dataclass(frozen=True)
class TrainerResponseStudyPlan:
    """Pure response reduction; the returned state/study are not persisted."""

    receipt: dict[str, Any]
    response_state: dict[str, Any]
    sealed_study: dict[str, Any] | None


def plan_terminal_to_next_generation(
    *,
    tuning_state_events_dir: Path,
    lineage_events_dir: Path,
    artifact_root: Path,
    trainer_packet: Mapping[str, Any],
    prompt_text: str,
    provider_id: str,
    model_id: str,
    health: Mapping[str, Any],
    prepared_at_utc: datetime | str,
    model_adapter_configured: bool = False,
) -> TrainerGenerationPlan:
    """Seal a terminal-result-to-model hand-off without calling the model.

    With no adapter, the result remains ``READY_FOR_MODEL`` and contains no
    reservation transition.  With an adapter configured, a *candidate* next
    state is derived with the attempt/model-call budget charged, but it must be
    CAS-appended and re-read by a separate orchestrator before any model call.
    """

    if not isinstance(model_adapter_configured, bool):
        raise DojoTrainerGenerationDriverError(
            "model_adapter_configured must be boolean"
        )
    store, state, lineage, packet = _load_terminal_context(
        tuning_state_events_dir=Path(tuning_state_events_dir),
        lineage_events_dir=Path(lineage_events_dir),
        artifact_root=Path(artifact_root),
        trainer_packet=trainer_packet,
    )
    prepared = _utc(prepared_at_utc, "prepared_at_utc")
    model_request = canonical_packet_bytes(packet)
    request_sha256 = hashlib.sha256(model_request).hexdigest()
    prompt_artifact = _build_prompt_artifact(
        prompt_text=prompt_text,
        trainer_packet_sha256=packet["packet_sha256"],
        model_request_sha256=request_sha256,
    )
    manifest = seal_model_invocation_manifest(
        tuning_state_events_dir=Path(tuning_state_events_dir),
        prompt_bytes=model_request,
        provider_id=provider_id,
        model_id=model_id,
        prepared_at_utc=prepared,
    )
    immutable_decision = plan_immutable_generation(
        tuning_state_events_dir=Path(tuning_state_events_dir),
        lineage_events_dir=Path(lineage_events_dir),
        artifact_root=Path(artifact_root),
        health=health,
        decision_at_utc=prepared,
        model_invocation_manifest=manifest,
        model_prompt_bytes=model_request,
    )
    if immutable_decision["action"] != "RESERVE_NEXT_MODEL_INVOCATION":
        raise DojoTrainerGenerationDriverError(
            "terminal state is not eligible for an immutable model reservation"
        )

    invocation_id = (
        f"dojo-a{manifest['attempt_ordinal']}-"
        f"{manifest['manifest_sha256'][:20]}"
    )
    reservation_state: dict[str, Any] | None = None
    if model_adapter_configured:
        try:
            reservation_state = reserve_model_invocation(
                state,
                lineage_events_dir=Path(lineage_events_dir),
                artifact_root=Path(artifact_root),
                expected_parent_state_sha256=state["state_sha256"],
                invocation_id=invocation_id,
                request_sha256=request_sha256,
                event_at_utc=prepared,
            )
        except ValueError as exc:
            raise DojoTrainerGenerationDriverError(
                "model reservation transition cannot be derived"
            ) from exc

    status = (
        "RESERVATION_TRANSITION_SEALED"
        if reservation_state is not None
        else "READY_FOR_MODEL"
    )
    blockers = (
        ["RESERVATION_STATE_NOT_DURABLY_APPENDED"]
        if reservation_state is not None
        else ["MODEL_ADAPTER_NOT_CONFIGURED"]
    )
    fixed_denominator = _fixed_denominator_binding(packet)
    source_digests = state["fixed_envelope"]["source_digests"]
    status_before = status_artifact(state)
    body = {
        "contract": DRIVER_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": "WORN_HISTORICAL_TRAIN_ORCHESTRATION_ONLY",
        "prepared_at_utc": prepared,
        "status": status,
        "blockers": blockers,
        "invocation_id": invocation_id,
        "bindings": {
            "registry_id": lineage.registry_id,
            "lineage_prefix": lineage.lineage_prefix,
            "lineage_tip_sha256": lineage.latest_event_sha256,
            "terminal_result_binding_sha256": canonical_sha256(
                state["last_terminal_result_binding"]
            ),
            "tuning_state_store_tip_sha256": store["latest_event_sha256"],
            "tuning_state_store_event_count": store["event_count"],
            "tuning_state_sha256": state["state_sha256"],
            "trainer_packet_sha256": packet["packet_sha256"],
            "model_request_sha256": request_sha256,
            "model_request_size_bytes": len(model_request),
            "prompt_artifact_sha256": hashlib.sha256(prompt_artifact).hexdigest(),
            "prompt_artifact_size_bytes": len(prompt_artifact),
            "model_invocation_manifest_sha256": manifest["manifest_sha256"],
            "provider_id": manifest["provider_id"],
            "model_id": manifest["model_id"],
            "scorer_contract": state["fixed_envelope"]["scorer_contract"],
            "corpus_sha256": state["fixed_envelope"]["window"][
                "corpus_sha256"
            ],
            "code_source_bundle_sha256": state["fixed_envelope"][
                "source_bundle_sha256"
            ],
            "source_digests_sha256": canonical_sha256(source_digests),
            "fixed_envelope_sha256": state["fixed_envelope_sha256"],
            "terminal_fixed_denominator_sha256": canonical_sha256(
                fixed_denominator
            ),
            "immutable_decision_sha256": immutable_decision["decision_sha256"],
            "health_sha256": immutable_decision["health_sha256"],
            "reservation_state_sha256": (
                reservation_state["state_sha256"]
                if reservation_state is not None
                else None
            ),
        },
        "budget": {
            "attempts_consumed_before": status_before["attempts_consumed"],
            "attempts_remaining_before": MAX_ATTEMPTS
            - status_before["attempts_consumed"],
            "model_invocations_consumed_before": status_before[
                "model_invocation_count"
            ],
            "proposal_slots_consumed_before": status_before[
                "proposal_slots_consumed"
            ],
            "proposal_slots_remaining_before": MAX_PROPOSAL_SLOTS
            - status_before["proposal_slots_consumed"],
            "reservation_attempt_charge": 1,
            "reservation_model_invocation_charge": 1,
            "response_proposal_slot_charge_rule": "max(1, emitted_submission_count)",
            "reservation_charge_derived": reservation_state is not None,
            "reservation_charge_durable": False,
        },
        "capacity_gate": {
            "current_generation_idle": health.get("process_state") == "IDLE",
            "next_run_resource_gate_required": True,
            "next_run_resource_gate_status": (
                "BLOCKED_UNTIL_NEXT_STUDY_LINEAGE_AND_DISPATCH_ARE_SEALED"
            ),
            "runner_start_eligible": False,
        },
        "invariants": {
            "terminal_result_bound_required": True,
            "partial_economics_allowed": False,
            "exact_model_request_required": True,
            "duplicate_executable_candidate_allowed": False,
            "fixed_denominator_mutation_allowed": False,
            "model_call_before_durable_reservation_allowed": False,
            "runner_start_before_resource_gate_allowed": False,
        },
        "permissions": dict(_NO_AUTHORITY),
        "limitations": [
            "PURE_PLAN_AND_SEAL_NO_SIDE_EFFECTS",
            "MODEL_ADAPTER_CONFIGURED_IS_CALLER_ASSERTED_NOT_ATTESTED",
            "MODEL_PROVIDER_AND_MODEL_ARE_REQUESTED_NOT_ATTESTED",
            "RESERVATION_STATE_REQUIRES_EXTERNAL_CAS_APPEND_AND_READBACK",
            "MODEL_RESPONSE_REQUIRES_EXACT_BYTES_AND_STATE_REDUCTION",
            "NEXT_STUDY_REQUIRES_LINEAGE_SEAL_AND_RESOURCE_GATE",
            "NO_HOLDOUT_FORWARD_PROOF_PROMOTION_OR_LIVE_AUTHORITY",
        ],
    }
    receipt = {**body, "driver_sha256": canonical_sha256(body)}
    return TrainerGenerationPlan(
        receipt=receipt,
        model_request_bytes=model_request,
        prompt_artifact_bytes=prompt_artifact,
        reservation_state=reservation_state,
    )


def reduce_model_response_to_next_study(
    plan: TrainerGenerationPlan,
    *,
    response_bytes: bytes,
    submissions: Sequence[Mapping[str, Any]],
    recorded_at_utc: datetime | str,
) -> TrainerResponseStudyPlan:
    """Reduce exact externally supplied response bytes into a sealed study.

    This function does not call the external model and does not persist the
    returned state or study.  ``record_model_response`` charges every emitted
    row (or one slot for an empty/malformed response), rejects executable
    duplicates, and ``materialize_next_study`` admits only accepted rows.
    """

    receipt = _verify_generation_plan(plan)
    if plan.reservation_state is None:
        raise DojoTrainerGenerationDriverError(
            "MODEL_ADAPTER_NOT_CONFIGURED: no reservation transition exists"
        )
    raw_response = _response_bytes(response_bytes)
    recorded = _utc(recorded_at_utc, "recorded_at_utc")
    if not isinstance(submissions, Sequence) or isinstance(
        submissions, (str, bytes, bytearray)
    ):
        raise DojoTrainerGenerationDriverError("submissions must be a sequence")
    parsed_response, response_parse_status = _strict_model_response(raw_response)
    response_is_array = isinstance(parsed_response, list)
    raw_response_rows = parsed_response if response_is_array else raw_response
    caller_submissions_match = response_is_array and _json_equal(
        list(submissions), parsed_response
    )
    try:
        response_state = record_model_response(
            plan.reservation_state,
            expected_parent_state_sha256=plan.reservation_state["state_sha256"],
            invocation_id=receipt["invocation_id"],
            response_sha256=hashlib.sha256(raw_response).hexdigest(),
            # Raw response bytes are the sole proposal denominator.  The
            # caller-supplied parsed view is compatibility/audit evidence only
            # and can neither omit a losing row nor add a candidate.
            submissions=raw_response_rows,
            event_at_utc=recorded,
        )
    except ValueError as exc:
        raise DojoTrainerGenerationDriverError(
            "external model response cannot be charged to its reservation"
        ) from exc

    try:
        if response_parse_status != "STRICT_JSON_ARRAY" or not caller_submissions_match:
            raise ValueError("raw response and caller submission denominator differ")
        sealed_study: dict[str, Any] | None = materialize_next_study(
            response_state,
            request_artifacts={
                receipt["invocation_id"]: plan.model_request_bytes,
            },
            response_artifacts={receipt["invocation_id"]: raw_response},
            prompt_artifact=plan.prompt_artifact_bytes,
        )
    except ValueError:
        # The response transition remains the authoritative consumed budget.
        # Returning it is essential: throwing it away would make an invalid or
        # duplicate-only response look like a free retry opportunity.
        sealed_study = None

    invocation = response_state["attempts"][-1]["invocations"][-1]
    accepted = [
        row for row in invocation["submissions"] if row["status"] == "ACCEPTED"
    ]
    duplicate = [
        row for row in invocation["submissions"] if row["status"] == "DUPLICATE"
    ]
    rejected = [
        row for row in invocation["submissions"] if row["status"] != "ACCEPTED"
    ]
    status_after = status_artifact(response_state)
    budget_before = receipt["budget"]
    expected_charge = max(1, len(parsed_response)) if response_is_array else 1
    if (
        status_after["attempts_consumed"]
        != budget_before["attempts_consumed_before"] + 1
        or status_after["model_invocation_count"]
        != budget_before["model_invocations_consumed_before"] + 1
        or status_after["proposal_slots_consumed"]
        != budget_before["proposal_slots_consumed_before"]
        + invocation["proposal_slot_charge"]
        or invocation["proposal_slot_charge"] != expected_charge
    ):
        raise DojoTrainerGenerationDriverError(
            "response state did not consume the sealed attempt, invocation, and proposal budgets"
        )
    study_sealed = sealed_study is not None
    body = {
        "contract": RESPONSE_PLAN_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": "WORN_HISTORICAL_TRAIN_ORCHESTRATION_ONLY",
        "recorded_at_utc": recorded,
        "driver_sha256": receipt["driver_sha256"],
        "invocation_id": receipt["invocation_id"],
        "model_request_sha256": hashlib.sha256(
            plan.model_request_bytes
        ).hexdigest(),
        "prompt_artifact_sha256": hashlib.sha256(
            plan.prompt_artifact_bytes
        ).hexdigest(),
        "response_sha256": hashlib.sha256(raw_response).hexdigest(),
        "response_size_bytes": len(raw_response),
        "response_parse_status": response_parse_status,
        "raw_response_row_count": len(parsed_response) if response_is_array else None,
        "submissions_source": "STRICT_RAW_RESPONSE_BYTES",
        "caller_submissions_match_raw_response": caller_submissions_match,
        "response_state_sha256": response_state["state_sha256"],
        "status": "NEXT_STUDY_SEALED" if study_sealed else "REVIEW_REQUIRED",
        "blockers": (
            [] if study_sealed else ["NEXT_STUDY_MATERIALIZATION_REJECTED"]
        ),
        "sealed_study_sha256": (
            sealed_study["study_sha256"] if sealed_study is not None else None
        ),
        "proposal_slot_charge": invocation["proposal_slot_charge"],
        "budget_after": {
            "attempts_consumed": status_after["attempts_consumed"],
            "attempts_remaining": MAX_ATTEMPTS
            - status_after["attempts_consumed"],
            "model_invocation_count": status_after["model_invocation_count"],
            "proposal_slots_consumed": status_after[
                "proposal_slots_consumed"
            ],
            "proposal_slots_remaining": MAX_PROPOSAL_SLOTS
            - status_after["proposal_slots_consumed"],
            "invalid_proposal_count": status_after["invalid_proposal_count"],
            "duplicate_proposal_count": status_after[
                "duplicate_proposal_count"
            ],
        },
        "accepted_candidate_ids": [row["candidate_id"] for row in accepted],
        "duplicate_submission_ids": [row["submission_id"] for row in duplicate],
        "rejected_submission_ids": [row["submission_id"] for row in rejected],
        "fixed_candidate_denominator": len(accepted) if study_sealed else None,
        "next_action": (
            "PERSIST_STATE_AND_STUDY_THEN_SEAL_LINEAGE"
            if study_sealed
            else "PERSIST_CHARGED_RESPONSE_STATE_THEN_REVIEW"
        ),
        "capacity_gate": {
            "required": True,
            "status": "BLOCKED_UNTIL_LINEAGE_SEAL_AND_DISPATCH_RESERVATION",
            "runner_start_eligible": False,
        },
        "permissions": dict(_NO_AUTHORITY),
        "limitations": [
            "PURE_RESPONSE_REDUCTION_NO_SIDE_EFFECTS",
            "MODEL_PROVIDER_OUTPUT_NOT_PROVIDER_ATTESTED",
            "STATE_AND_STUDY_ARE_NOT_DURABLE",
            "RESOURCE_GATE_NOT_YET_EVALUABLE",
            "NO_PROOF_PROMOTION_OR_LIVE_AUTHORITY",
        ],
    }
    response_receipt = {**body, "response_plan_sha256": canonical_sha256(body)}
    return TrainerResponseStudyPlan(
        receipt=response_receipt,
        response_state=response_state,
        sealed_study=sealed_study,
    )


def plan_reserved_generation_capacity(
    *,
    tuning_state_events_dir: Path,
    lineage_events_dir: Path,
    artifact_root: Path,
    health: Mapping[str, Any],
    resource_gate_request: Mapping[str, Any],
    decision_at_utc: datetime | str,
) -> dict[str, Any]:
    """Evaluate the real capacity gate only after dispatch reservation exists."""

    try:
        decision = plan_immutable_generation(
            tuning_state_events_dir=Path(tuning_state_events_dir),
            lineage_events_dir=Path(lineage_events_dir),
            artifact_root=Path(artifact_root),
            health=health,
            decision_at_utc=decision_at_utc,
            resource_gate_request=resource_gate_request,
        )
    except ValueError as exc:
        raise DojoTrainerGenerationDriverError(
            "reserved generation capacity cannot be evaluated"
        ) from exc
    if decision["action"] not in {
        "MANUAL_START_ELIGIBLE",
        "WAIT_FOR_RESOURCES",
        "REVIEW_RESOURCE_EVIDENCE",
    }:
        raise DojoTrainerGenerationDriverError(
            "capacity gate requires a lineage-sealed reserved run dispatch"
        )
    permissions = decision["permissions"]
    if (
        permissions["runner_invocation_allowed"] is not False
        or permissions["automatic_runner_start_allowed"] is not False
        or permissions["live_permission"] is not False
        or permissions["order_authority"] != "NONE"
    ):
        raise DojoTrainerGenerationDriverError(
            "capacity decision unexpectedly gained execution authority"
        )
    return decision


def _load_terminal_context(
    *,
    tuning_state_events_dir: Path,
    lineage_events_dir: Path,
    artifact_root: Path,
    trainer_packet: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    CandidateLineageSnapshot,
    dict[str, Any],
]:
    try:
        store = dict(verify_state_store(tuning_state_events_dir))
        state = verify_tuning_state(store["latest_state"])
        lineage = verify_registry(lineage_events_dir, artifact_root=artifact_root)
        packet = verify_trainer_packet(trainer_packet)
    except (KeyError, TypeError, ValueError) as exc:
        raise DojoTrainerGenerationDriverError(
            "terminal trainer context is invalid"
        ) from exc
    if any(store.get(key) != expected for key, expected in _REQUIRED_STORE_BOUNDARY.items()):
        raise DojoTrainerGenerationDriverError(
            "tuning state store is not the research-only automation boundary"
        )
    if state["phase"] != "READY_FOR_MODEL":
        raise DojoTrainerGenerationDriverError(
            "only a READY_FOR_MODEL terminal state may enter the driver"
        )
    if not lineage.results or len(lineage.studies) != len(lineage.results):
        raise DojoTrainerGenerationDriverError(
            "lineage does not have a complete terminal result denominator"
        )
    if not lineage.events or lineage.events[-1]["event_type"] != "RESULT_BOUND":
        raise DojoTrainerGenerationDriverError(
            "lineage tip is not an exact terminal result binding"
        )
    expected_binding = _latest_result_binding(lineage)
    if state["last_terminal_result_binding"] != expected_binding:
        raise DojoTrainerGenerationDriverError(
            "tuning state is not bound to the latest terminal lineage result"
        )
    _verify_packet_context(packet, state=state, lineage=lineage)
    return store, state, lineage, packet


def _verify_packet_context(
    packet: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    lineage: CandidateLineageSnapshot,
) -> None:
    source = packet["source_bindings"]
    if set(source) != _SOURCE_BINDING_KEYS:
        raise DojoTrainerGenerationDriverError(
            "trainer packet source binding schema drifted"
        )
    _sha(source["run_sha256"], "trainer packet run_sha256")
    binding = state["last_terminal_result_binding"]
    expected_source = {
        "registry_id": state["registry_id"],
        "lineage_prefix": state["lineage_prefix"],
        "attempt_ordinal": binding["attempt_ordinal"],
        "study_sha256": binding["study_sha256"],
        "evaluation_sha256": binding["evaluation_sha256"],
        "evaluation_artifact_sha256": binding["evaluation_artifact_sha256"],
        "lineage_result_event_sha256": binding["result_event_sha256"],
        "lineage_tip_sha256": binding["lineage_tip_sha256"],
        "tuning_state_sha256": state["state_sha256"],
        "fixed_envelope_sha256": state["fixed_envelope_sha256"],
        "external_witness_status": lineage.external_witness_status,
        "exact_result_binding_verified": True,
    }
    if any(source.get(key) != expected for key, expected in expected_source.items()):
        raise DojoTrainerGenerationDriverError(
            "trainer packet is not the exact latest terminal result"
        )
    if packet["current_run"]["status"] not in {
        "COMPLETE",
        "COMPLETE_WITH_FAILED_CELLS",
    }:
        raise DojoTrainerGenerationDriverError(
            "partial or non-terminal run economics are forbidden"
        )
    envelope = state["fixed_envelope"]
    fixed = packet["fixed_environment"]
    if set(fixed) != _FIXED_ENVIRONMENT_KEYS:
        raise DojoTrainerGenerationDriverError(
            "trainer packet fixed environment schema drifted"
        )
    for key in (
        "window_role",
        "window",
        "initial_balance_jpy",
        "trade_pairs",
        "feed_pairs",
        "intrabar_paths",
        "cost_arms",
        "thresholds",
    ):
        if fixed.get(key) != envelope[key]:
            raise DojoTrainerGenerationDriverError(
                f"trainer packet fixed environment drifted: {key}"
            )
    if fixed["catalog"] != catalog_manifest():
        raise DojoTrainerGenerationDriverError(
            "trainer packet catalog/code binding drifted"
        )
    tuning_status = status_artifact(state)
    budget = packet["search_budget"]
    if set(budget) != _SEARCH_BUDGET_KEYS:
        raise DojoTrainerGenerationDriverError(
            "trainer packet search budget schema drifted"
        )
    expected_budget = {
        "phase": "READY_FOR_MODEL",
        "attempts_consumed": tuning_status["attempts_consumed"],
        "attempts_remaining": MAX_ATTEMPTS - tuning_status["attempts_consumed"],
        "max_attempts": MAX_ATTEMPTS,
        "proposal_slots_consumed": tuning_status["proposal_slots_consumed"],
        "proposal_slots_remaining": MAX_PROPOSAL_SLOTS
        - tuning_status["proposal_slots_consumed"],
        "max_proposal_slots": MAX_PROPOSAL_SLOTS,
        "invalid_proposal_count": tuning_status["invalid_proposal_count"],
        "duplicate_proposal_count": tuning_status["duplicate_proposal_count"],
    }
    if any(budget.get(key) != expected for key, expected in expected_budget.items()):
        raise DojoTrainerGenerationDriverError(
            "trainer packet search budget is stale or incomplete"
        )
    if (
        expected_budget["attempts_remaining"] <= 0
        or expected_budget["proposal_slots_remaining"] <= 0
    ):
        raise DojoTrainerGenerationDriverError(
            "trainer search budget is exhausted"
        )


def _latest_result_binding(lineage: CandidateLineageSnapshot) -> dict[str, Any]:
    result = lineage.results[-1]
    event = lineage.events[-1]
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


def _fixed_denominator_binding(packet: Mapping[str, Any]) -> dict[str, Any]:
    coordinates = sorted(
        (
            {
                "candidate_id": row["candidate_id"],
                "intrabar": row["intrabar"],
                "cost_arm": row["cost_arm"],
            }
            for row in packet["cells"]
        ),
        key=lambda row: (row["candidate_id"], row["intrabar"], row["cost_arm"]),
    )
    return {
        "fixed_denominator": copy.deepcopy(
            packet["current_run"]["fixed_denominator"]
        ),
        "candidate_ids": list(packet["current_run"]["candidate_ids"]),
        "coordinates": coordinates,
        "failed_coordinates": list(packet["failed_coordinates"]),
    }


def _build_prompt_artifact(
    *,
    prompt_text: str,
    trainer_packet_sha256: str,
    model_request_sha256: str,
) -> bytes:
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise DojoTrainerGenerationDriverError("prompt_text must be non-empty")
    raw_text = prompt_text.encode("utf-8")
    if len(raw_text) > MAX_PROMPT_TEXT_BYTES or "\x00" in prompt_text:
        raise DojoTrainerGenerationDriverError("prompt_text exceeds its safe bound")
    _sha(trainer_packet_sha256, "trainer_packet_sha256")
    _sha(model_request_sha256, "model_request_sha256")
    body = {
        "contract": PROMPT_ARTIFACT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": "UNVERIFIED_CALLER_ARTIFACT",
        "trainer_packet_sha256": trainer_packet_sha256,
        "model_request_sha256": model_request_sha256,
        "prompt_text": prompt_text,
        "model_claim": UNVERIFIED_MODEL_CLAIM,
        "provider_attestation": UNVERIFIED_PROVIDER_ATTESTATION,
        **_PROMPT_AUTHORITY,
    }
    return _canonical_bytes(body) + b"\n"


def _verify_generation_plan(plan: TrainerGenerationPlan) -> dict[str, Any]:
    if not isinstance(plan, TrainerGenerationPlan):
        raise DojoTrainerGenerationDriverError(
            "generation plan must be the typed driver output"
        )
    receipt = copy.deepcopy(plan.receipt)
    if receipt.get("contract") != DRIVER_CONTRACT:
        raise DojoTrainerGenerationDriverError("generation plan contract drifted")
    claimed = _sha(receipt.get("driver_sha256"), "driver_sha256")
    body = {key: value for key, value in receipt.items() if key != "driver_sha256"}
    if claimed != canonical_sha256(body):
        raise DojoTrainerGenerationDriverError("generation plan seal mismatch")
    bindings = receipt.get("bindings")
    if not isinstance(bindings, Mapping):
        raise DojoTrainerGenerationDriverError("generation plan bindings are absent")
    if (
        hashlib.sha256(plan.model_request_bytes).hexdigest()
        != bindings.get("model_request_sha256")
        or len(plan.model_request_bytes) != bindings.get("model_request_size_bytes")
        or hashlib.sha256(plan.prompt_artifact_bytes).hexdigest()
        != bindings.get("prompt_artifact_sha256")
        or len(plan.prompt_artifact_bytes)
        != bindings.get("prompt_artifact_size_bytes")
    ):
        raise DojoTrainerGenerationDriverError(
            "generation plan artifact bytes drifted"
        )
    if plan.reservation_state is None:
        if bindings.get("reservation_state_sha256") is not None:
            raise DojoTrainerGenerationDriverError(
                "generation plan reservation binding is inconsistent"
            )
    else:
        try:
            state = verify_tuning_state(plan.reservation_state)
        except ValueError as exc:
            raise DojoTrainerGenerationDriverError(
                "generation plan reservation state is invalid"
            ) from exc
        if state["state_sha256"] != bindings.get("reservation_state_sha256"):
            raise DojoTrainerGenerationDriverError(
                "generation plan reservation state drifted"
            )
    if receipt.get("permissions") != _NO_AUTHORITY:
        raise DojoTrainerGenerationDriverError(
            "generation plan unexpectedly gained authority"
        )
    return receipt


def _canonical_bytes(value: Any) -> bytes:
    _validate_json(value, "value")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoTrainerGenerationDriverError(
            "value cannot be encoded as canonical JSON"
        ) from exc


def _strict_model_response(raw: bytes) -> tuple[Any, str]:
    """Parse model bytes once without turning malformed output into a free retry."""

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON token is forbidden: {token}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key is forbidden")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
        _validate_json(value, "model_response")
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        RecursionError,
        DojoTrainerGenerationDriverError,
    ):
        return None, "INVALID_STRICT_JSON"
    if not isinstance(value, list):
        return value, "STRICT_JSON_NON_ARRAY"
    return value, "STRICT_JSON_ARRAY"


def _json_equal(left: Any, right: Any) -> bool:
    try:
        return _canonical_bytes(left) == _canonical_bytes(right)
    except DojoTrainerGenerationDriverError:
        return False


def _bytes(value: Any, field: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise DojoTrainerGenerationDriverError(f"{field} must be non-empty bytes")
    return value


def _response_bytes(value: Any) -> bytes:
    if not isinstance(value, bytes):
        raise DojoTrainerGenerationDriverError("response_bytes must be bytes")
    return value


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DojoTrainerGenerationDriverError(f"{field} is not SHA-256")
    return value


def _utc(value: datetime | str, field: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.endswith("Z") and "T" in value:
        try:
            parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
        except ValueError as exc:
            raise DojoTrainerGenerationDriverError(
                f"{field} must be a valid UTC timestamp"
            ) from exc
    else:
        raise DojoTrainerGenerationDriverError(
            f"{field} must be a valid UTC timestamp"
        )
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DojoTrainerGenerationDriverError(f"{field} must use UTC")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _validate_json(value: Any, field: str, *, depth: int = 0) -> None:
    if depth > 64:
        raise DojoTrainerGenerationDriverError(f"{field} exceeds JSON depth")
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DojoTrainerGenerationDriverError(
                f"{field} contains a non-finite value"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{field}[{index}]", depth=depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DojoTrainerGenerationDriverError(
                    f"{field} contains a non-string key"
                )
            _validate_json(item, f"{field}.{key}", depth=depth + 1)
        return
    raise DojoTrainerGenerationDriverError(f"{field} is not JSON")


__all__ = [
    "DRIVER_CONTRACT",
    "RESPONSE_PLAN_CONTRACT",
    "DojoTrainerGenerationDriverError",
    "TrainerGenerationPlan",
    "TrainerResponseStudyPlan",
    "plan_reserved_generation_capacity",
    "plan_terminal_to_next_generation",
    "reduce_model_response_to_next_study",
]
