"""Fail-closed inter-run state for the DOJO AI bot trainer.

This module owns no model client and no replay runner.  It records reservations
which an external orchestrator must persist *before* calling either service.
That separation is intentional: a crash can be retried with the same operation
identifier without silently creating another model search or replay run.

The state is research-only.  It accepts only the worn ``TRAIN`` contract used by
the deterministic bot trainer, never reads a holdout or forward window, and
cannot grant proof, promotion, live-order, or broker-mutation authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_bot_catalog import (
    RISK_POLICY_CONTRACT,
    bot_risk_policy_manifest,
)
from quant_rabbit.dojo_bot_trainer import (
    EVALUATION_CONTRACT,
    REQUIRED_INTRABAR_PATHS,
    DojoBotTrainerError,
    seal_candidate_proposal,
    verify_sealed_study,
)
from quant_rabbit.dojo_candidate_lineage_registry import (
    CandidateLineageError,
    CandidateLineageSnapshot,
    verify_registry,
)


STATE_CONTRACT: Final = "QR_DOJO_AI_TUNING_STATE_V1"
ENVELOPE_CONTRACT: Final = "QR_DOJO_AI_TUNING_FIXED_ENVELOPE_V1"
STATUS_CONTRACT: Final = "QR_DOJO_AI_TUNING_STATUS_V1"
STORE_EVENT_CONTRACT: Final = "QR_DOJO_AI_TUNING_STATE_STORE_EVENT_V1"
SCHEMA_VERSION: Final = 1

# These are the preregistered search-budget ceilings shared with the candidate
# lineage registry.  They limit researcher degrees of freedom; they are not
# market, sizing, or production-risk constants.
MAX_ATTEMPTS: Final = 3
MAX_PROPOSAL_SLOTS: Final = 14
# Persistence bounds protect the local verifier from unbounded files.  They are
# engineering limits only, never search-budget or market-risk parameters.
MAX_STATE_STORE_EVENTS: Final = 64
MAX_STATE_STORE_EVENT_BYTES: Final = 8 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}\Z")
_STORE_EVENT_NAME = re.compile(r"[0-9]{6}\.json\Z")
_AUTHORITY = {
    "automation_ready": False,
    "research_train_only": True,
    "holdout_access_allowed": False,
    "forward_access_allowed": False,
    "proof_eligible": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
}
_STATE_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "registry_id",
        "lineage_prefix",
        "fixed_envelope",
        "fixed_envelope_sha256",
        "revision",
        "previous_state_sha256",
        "initialized_at_utc",
        "last_transition_at_utc",
        "initial_attempts_consumed",
        "initial_proposal_slots_consumed",
        "attempts",
        "max_attempts",
        "max_proposal_slots",
        "global_executable_identity_sha256s",
        "last_terminal_result_binding",
        "phase",
        "terminal_reason",
        *_AUTHORITY,
        "state_sha256",
    }
)
_STORE_EVENT_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "sequence",
        "previous_event_sha256",
        "parent_state_sha256",
        "state",
        "event_sha256",
    }
)
_ATTEMPT_KEYS = frozenset(
    {
        "attempt_ordinal",
        "prior_result_binding",
        "envelope_sha256",
        "phase",
        "invocations",
        "dispatch",
        "terminal",
    }
)
_INVOCATION_KEYS = frozenset(
    {
        "invocation_id",
        "request_sha256",
        "reserved_at_utc",
        "response_sha256",
        "response_input_sha256",
        "response_recorded_at_utc",
        "proposal_slot_charge",
        "submissions",
    }
)
_SUBMISSION_KEYS = frozenset(
    {
        "submission_id",
        "raw_proposal_sha256",
        "status",
        "validation_errors",
        "candidate_id",
        "proposal_sha256",
        "executable_identity_sha256",
        "sealed_proposal",
    }
)
_DISPATCH_KEYS = frozenset(
    {
        "dispatch_id",
        "status",
        "reserved_at_utc",
        "dispatched_at_utc",
        "study_sha256",
        "study_event_sha256",
        "study_event_sequence",
        "candidate_ids",
        "proposal_sha256s",
        "executable_identity_sha256s",
        "prior_result_binding",
        "envelope_sha256",
    }
)
_RESULT_BINDING_KEYS = frozenset(
    {
        "registry_id",
        "lineage_prefix",
        "attempt_ordinal",
        "study_sha256",
        "evaluation_sha256",
        "evaluation_artifact_sha256",
        "evaluation_artifact_size_bytes",
        "result_event_sha256",
        "result_event_sequence",
        "lineage_tip_sha256",
    }
)
_TERMINAL_KEYS = frozenset(
    {
        "status",
        "recorded_at_utc",
        "result_binding",
        "reason_code",
        "review_id",
        "review_rationale",
    }
)
_ENVELOPE_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "window_role",
        "window",
        "initial_balance_jpy",
        "trade_pairs",
        "feed_pairs",
        "cost_arms",
        "thresholds",
        "intrabar_paths",
        "risk_policy_contract",
        "risk_policy_sha256",
        "scorer_contract",
        "source_digests",
        "source_bundle_sha256",
        "classification",
        "holdout_access_allowed",
        "forward_access_allowed",
        "live_permission",
        "order_authority",
        "broker_mutation_allowed",
        "envelope_sha256",
    }
)


class DojoAITuningStateError(ValueError):
    """The tuning state, transition, or lineage binding is invalid."""


def fixed_envelope_from_sealed_study(
    sealed_study: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the immutable risk/window/cost/scorer envelope from a study.

    Candidate proposals and attempt identifiers are deliberately excluded.  All
    fields which may change economics or scoring are retained byte-for-byte in
    canonical JSON form, including every source digest used by the scorer.
    """

    try:
        source_digests = sealed_study.get("source_digests")
        if not isinstance(source_digests, Mapping):
            raise DojoAITuningStateError("sealed study lacks source_digests")
        sealed = verify_sealed_study(sealed_study, source_digests)
    except DojoBotTrainerError as exc:
        raise DojoAITuningStateError(
            f"sealed study cannot define a tuning envelope: {exc}"
        ) from exc
    study = sealed["study"]
    risk_manifest = bot_risk_policy_manifest()
    source_bundle = _canonical_sha(source_digests)
    body = {
        "contract": ENVELOPE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "window_role": study["window_role"],
        "window": _clone(study["window"]),
        "initial_balance_jpy": study["initial_balance_jpy"],
        "trade_pairs": list(study["trade_pairs"]),
        "feed_pairs": list(study["feed_pairs"]),
        "cost_arms": _clone(study["cost_arms"]),
        "thresholds": _clone(study["thresholds"]),
        "intrabar_paths": list(REQUIRED_INTRABAR_PATHS),
        "risk_policy_contract": RISK_POLICY_CONTRACT,
        "risk_policy_sha256": _canonical_sha(risk_manifest),
        "scorer_contract": EVALUATION_CONTRACT,
        "source_digests": _clone(source_digests),
        "source_bundle_sha256": source_bundle,
        "classification": "WORN_HISTORICAL_TRAIN_ONLY",
        "holdout_access_allowed": False,
        "forward_access_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return {**body, "envelope_sha256": _canonical_sha(body)}


def initialize_tuning_state(
    lineage_events_dir: Path,
    *,
    artifact_root: Path,
    sealed_study: Mapping[str, Any],
) -> dict[str, Any]:
    """Initialize after a terminal trainer evaluation is lineage-bound.

    Existing sealed candidates consume both the attempt and proposal budgets.
    This prevents adding a fresh fourteen-proposal AI search on top of an
    already-used baseline lineage.
    """

    lineage = _load_verified_lineage(lineage_events_dir, artifact_root)
    if not lineage.studies:
        raise DojoAITuningStateError("AI tuning requires a completed baseline study")
    if len(lineage.studies) != len(lineage.results):
        raise DojoAITuningStateError(
            "incomplete lineage run requires review before AI dispatch"
        )
    if len(lineage.studies) > MAX_ATTEMPTS:
        raise DojoAITuningStateError("lineage already exceeds the attempt budget")
    last_study = lineage.studies[-1]
    envelope = fixed_envelope_from_sealed_study(sealed_study)
    try:
        sealed = verify_sealed_study(sealed_study, sealed_study["source_digests"])
    except (KeyError, DojoBotTrainerError) as exc:
        raise DojoAITuningStateError("baseline sealed study is invalid") from exc
    if sealed["study_sha256"] != last_study["study_sha256"]:
        raise DojoAITuningStateError(
            "fixed envelope study is not the latest lineage-bound study"
        )
    initial_slots = sum(int(study["candidate_count"]) for study in lineage.studies)
    if initial_slots > MAX_PROPOSAL_SLOTS:
        raise DojoAITuningStateError("lineage already exceeds proposal budget")
    identities = sorted(lineage.cumulative_unique_config_sha256s)
    if len(identities) != initial_slots or len(set(identities)) != len(identities):
        raise DojoAITuningStateError(
            "lineage executable identities do not match consumed proposal slots"
        )
    state = {
        "contract": STATE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "registry_id": lineage.registry_id,
        "lineage_prefix": lineage.lineage_prefix,
        "fixed_envelope": envelope,
        "fixed_envelope_sha256": envelope["envelope_sha256"],
        "revision": 0,
        "previous_state_sha256": None,
        "initialized_at_utc": _utc(
            lineage.latest_event_at_utc, "lineage.latest_event_at_utc"
        ),
        "last_transition_at_utc": _utc(
            lineage.latest_event_at_utc, "lineage.latest_event_at_utc"
        ),
        "initial_attempts_consumed": len(lineage.studies),
        "initial_proposal_slots_consumed": initial_slots,
        "attempts": [],
        "max_attempts": MAX_ATTEMPTS,
        "max_proposal_slots": MAX_PROPOSAL_SLOTS,
        "global_executable_identity_sha256s": identities,
        "last_terminal_result_binding": _terminal_result_binding(lineage),
        "phase": (
            "EXHAUSTED"
            if len(lineage.studies) == MAX_ATTEMPTS
            or initial_slots == MAX_PROPOSAL_SLOTS
            else "READY_FOR_MODEL"
        ),
        "terminal_reason": None,
        **_AUTHORITY,
    }
    return _seal_state(state)


def reserve_model_invocation(
    state: Mapping[str, Any],
    *,
    lineage_events_dir: Path,
    artifact_root: Path,
    expected_parent_state_sha256: str,
    invocation_id: str,
    request_sha256: str,
    event_at_utc: datetime | str,
) -> dict[str, Any]:
    """Reserve one model call before external dispatch.

    Each tuning attempt permits exactly one model invocation.  Replaying that
    first reservation from its exact parent with the same invocation id and
    request is an idempotent no-op.  Once a response is recorded, callers must
    dispatch its accepted candidates or send the attempt to review; they cannot
    buy another model search inside the same attempt.
    """

    value = verify_tuning_state(state)
    cas_mode = _cas_mode(value, expected_parent_state_sha256)
    invocation_key = _identifier(invocation_id, "invocation_id")
    request_digest = _sha(request_sha256, "request_sha256")
    event_time = _utc(event_at_utc, "event_at_utc")
    lineage = _load_verified_lineage(lineage_events_dir, artifact_root)
    _require_terminal_lineage(value, lineage)

    existing = _find_invocation(value, invocation_key)
    if existing is not None:
        if existing["request_sha256"] != request_digest:
            raise DojoAITuningStateError(
                "invocation id replay changed the model request"
            )
        if (
            cas_mode != "REPLAY_LAST"
            or value["phase"] != "MODEL_INVOCATION_RESERVED"
            or existing["response_sha256"] is not None
        ):
            raise DojoAITuningStateError(
                "model reservation replay is not the exact immediate first invocation"
            )
        return value
    _require_current_cas(cas_mode)
    if value["phase"] != "READY_FOR_MODEL":
        raise DojoAITuningStateError(
            f"model invocation is not allowed from phase {value['phase']}"
        )
    if _current_attempt(value) is not None:
        raise DojoAITuningStateError(
            "model invocation requires no active AI tuning attempt"
        )
    if _proposal_slots_used(value) >= MAX_PROPOSAL_SLOTS:
        raise DojoAITuningStateError("proposal budget is exhausted")

    updated = _clone(value)
    ordinal = _attempts_used(updated) + 1
    if ordinal > MAX_ATTEMPTS:
        raise DojoAITuningStateError("attempt budget is exhausted")
    attempt = {
        "attempt_ordinal": ordinal,
        "prior_result_binding": _clone(updated["last_terminal_result_binding"]),
        "envelope_sha256": updated["fixed_envelope_sha256"],
        "phase": "MODEL_INVOCATION_RESERVED",
        "invocations": [],
        "dispatch": None,
        "terminal": None,
    }
    updated["attempts"].append(attempt)
    attempt["invocations"].append(
        {
            "invocation_id": invocation_key,
            "request_sha256": request_digest,
            "reserved_at_utc": event_time,
            "response_sha256": None,
            "response_input_sha256": None,
            "response_recorded_at_utc": None,
            "proposal_slot_charge": 0,
            "submissions": [],
        }
    )
    attempt["phase"] = "MODEL_INVOCATION_RESERVED"
    updated["phase"] = "MODEL_INVOCATION_RESERVED"
    return _seal_next_state(updated, parent=value, transition_at_utc=event_time)


def record_model_response(
    state: Mapping[str, Any],
    *,
    expected_parent_state_sha256: str,
    invocation_id: str,
    response_sha256: str,
    submissions: Any,
    event_at_utc: datetime | str,
) -> dict[str, Any]:
    """Record every proposal emitted by one reserved model invocation.

    Invalid and executable-duplicate proposals consume slots.  A response with
    no parseable proposal also consumes one slot, so empty or malformed model
    calls cannot create an unbounded free search loop.
    """

    value = verify_tuning_state(state)
    cas_mode = _cas_mode(value, expected_parent_state_sha256)
    invocation_key = _identifier(invocation_id, "invocation_id")
    response_digest = _sha(response_sha256, "response_sha256")
    event_time = _utc(event_at_utc, "event_at_utc")
    updated = _clone(value)
    invocation = _find_invocation(updated, invocation_key)
    if invocation is None:
        raise DojoAITuningStateError("model response lacks a reserved invocation")
    prior_submission_ids = {
        row["submission_id"]
        for attempt in updated["attempts"]
        for prior_invocation in attempt["invocations"]
        if prior_invocation["invocation_id"] != invocation_key
        for row in prior_invocation["submissions"]
    }
    raw_submissions = normalize_model_response_submissions(
        submissions,
        response_sha256=response_digest,
        invocation_id=invocation_key,
        forbidden_submission_ids=prior_submission_ids,
    )
    result_fingerprint = _canonical_sha(
        {
            "response_sha256": response_digest,
            "submissions": raw_submissions,
        }
    )
    if invocation["response_sha256"] is not None:
        if invocation["response_input_sha256"] != result_fingerprint:
            raise DojoAITuningStateError(
                "invocation replay changed the model response or proposals"
            )
        return value
    _require_current_cas(cas_mode)
    if updated["phase"] != "MODEL_INVOCATION_RESERVED":
        raise DojoAITuningStateError("model response is outside its reserved phase")

    # One empty/malformed response is one consumed proposal opportunity.  A
    # multi-candidate response consumes the number of candidate rows returned.
    charge = max(1, len(raw_submissions))
    remaining = MAX_PROPOSAL_SLOTS - _proposal_slots_used(updated)
    budget_breach = charge > remaining
    known_identities = set(updated["global_executable_identity_sha256s"])
    normalized_rows: list[dict[str, Any]] = []
    for submission in raw_submissions:
        row = _evaluate_submission(
            submission,
            lineage_prefix=updated["lineage_prefix"],
            known_identities=known_identities,
            force_budget_reject=budget_breach,
        )
        normalized_rows.append(row)
        if row["status"] == "ACCEPTED":
            identity = row["executable_identity_sha256"]
            assert identity is not None
            known_identities.add(identity)

    invocation["response_sha256"] = response_digest
    invocation["response_input_sha256"] = result_fingerprint
    invocation["response_recorded_at_utc"] = event_time
    invocation["proposal_slot_charge"] = charge
    invocation["submissions"] = normalized_rows
    updated["global_executable_identity_sha256s"] = sorted(known_identities)
    attempt = _current_attempt(updated)
    assert attempt is not None
    if budget_breach:
        attempt["phase"] = "REVIEW_REQUIRED"
        updated["phase"] = "REVIEW_REQUIRED"
        updated["terminal_reason"] = "MODEL_RESPONSE_EXCEEDED_PROPOSAL_BUDGET"
    else:
        attempt["phase"] = "COLLECTING_PROPOSALS"
        updated["phase"] = "COLLECTING_PROPOSALS"
        if _proposal_slots_used(updated) >= MAX_PROPOSAL_SLOTS and not _accepted_rows(
            attempt
        ):
            attempt["phase"] = "REVIEW_REQUIRED"
            updated["phase"] = "REVIEW_REQUIRED"
            updated["terminal_reason"] = "PROPOSAL_BUDGET_EXHAUSTED_WITHOUT_CANDIDATE"
    return _seal_next_state(updated, parent=value, transition_at_utc=event_time)


def normalize_model_response_submissions(
    submissions: Any,
    *,
    response_sha256: str,
    invocation_id: str,
    forbidden_submission_ids: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Derive the complete budget-consuming denominator for one model response.

    Downstream study materialization uses the same function to recompute the
    denominator from exact raw response bytes.  A caller-provided winner subset
    therefore cannot retain a matching response-input digest.
    """

    response_digest = _sha(response_sha256, "response_sha256")
    invocation_key = _identifier(invocation_id, "invocation_id")
    if not isinstance(forbidden_submission_ids, Collection) or isinstance(
        forbidden_submission_ids, (str, bytes, bytearray)
    ):
        raise DojoAITuningStateError("forbidden_submission_ids must be a collection")
    forbidden = {
        _identifier(value, "forbidden_submission_id")
        for value in forbidden_submission_ids
    }
    return _chargeable_submission_inputs(
        submissions,
        response_sha256=response_digest,
        invocation_id=invocation_key,
        forbidden_submission_ids=forbidden,
    )


def reserve_run_dispatch(
    state: Mapping[str, Any],
    *,
    lineage_events_dir: Path,
    artifact_root: Path,
    expected_parent_state_sha256: str,
    sealed_study: Mapping[str, Any],
    dispatch_id: str,
    event_at_utc: datetime | str,
) -> dict[str, Any]:
    """Reserve the exact lineage-sealed TRAIN run before launching replay."""

    value = verify_tuning_state(state)
    cas_mode = _cas_mode(value, expected_parent_state_sha256)
    dispatch_key = _identifier(dispatch_id, "dispatch_id")
    event_time = _utc(event_at_utc, "event_at_utc")
    lineage = _load_verified_lineage(lineage_events_dir, artifact_root)
    _require_event_not_before_lineage(event_time, lineage)
    attempt = _current_attempt(value)
    if attempt is None:
        raise DojoAITuningStateError("run dispatch requires an active AI attempt")
    if attempt["dispatch"] is not None:
        existing = attempt["dispatch"]
        if existing["dispatch_id"] != dispatch_key:
            raise DojoAITuningStateError("attempt already has another dispatch")
        _verify_dispatch_inputs(value, lineage, sealed_study, existing)
        return value
    _require_current_cas(cas_mode)
    if value["phase"] != "COLLECTING_PROPOSALS":
        raise DojoAITuningStateError(
            f"run dispatch is not allowed from phase {value['phase']}"
        )
    if any(
        invocation["response_sha256"] is None for invocation in attempt["invocations"]
    ):
        raise DojoAITuningStateError("reserved model invocation is incomplete")

    dispatch = _derive_dispatch(
        value,
        lineage,
        sealed_study,
        dispatch_id=dispatch_key,
        reserved_at_utc=event_time,
    )
    updated = _clone(value)
    current = _current_attempt(updated)
    assert current is not None
    current["dispatch"] = dispatch
    current["phase"] = "RUN_DISPATCH_RESERVED"
    updated["phase"] = "RUN_DISPATCH_RESERVED"
    return _seal_next_state(updated, parent=value, transition_at_utc=event_time)


def mark_run_dispatched(
    state: Mapping[str, Any],
    *,
    lineage_events_dir: Path,
    artifact_root: Path,
    expected_parent_state_sha256: str,
    dispatch_id: str,
    event_at_utc: datetime | str,
) -> dict[str, Any]:
    """Record that the reserved replay process was actually started."""

    value = verify_tuning_state(state)
    cas_mode = _cas_mode(value, expected_parent_state_sha256)
    dispatch_key = _identifier(dispatch_id, "dispatch_id")
    event_time = _utc(event_at_utc, "event_at_utc")
    lineage = _load_verified_lineage(lineage_events_dir, artifact_root)
    _require_event_not_before_lineage(event_time, lineage)
    attempt = _current_attempt(value)
    if attempt is None or attempt["dispatch"] is None:
        raise DojoAITuningStateError("run start lacks a dispatch reservation")
    dispatch = attempt["dispatch"]
    if dispatch["dispatch_id"] != dispatch_key:
        raise DojoAITuningStateError("run start does not match reserved dispatch")
    if dispatch["status"] == "DISPATCHED":
        return value
    _require_current_cas(cas_mode)
    if value["phase"] != "RUN_DISPATCH_RESERVED":
        raise DojoAITuningStateError("run start is outside dispatch reservation")
    _require_reserved_dispatch_lineage(value, lineage)
    updated = _clone(value)
    current = _current_attempt(updated)
    assert current is not None and current["dispatch"] is not None
    current["dispatch"]["status"] = "DISPATCHED"
    current["dispatch"]["dispatched_at_utc"] = event_time
    current["phase"] = "AWAITING_LINEAGE_RESULT"
    updated["phase"] = "AWAITING_LINEAGE_RESULT"
    return _seal_next_state(updated, parent=value, transition_at_utc=event_time)


def bind_terminal_evaluation(
    state: Mapping[str, Any],
    *,
    lineage_events_dir: Path,
    artifact_root: Path,
    expected_parent_state_sha256: str,
    event_at_utc: datetime | str,
) -> dict[str, Any]:
    """Bind the exact registry RESULT_BOUND event for the completed attempt."""

    value = verify_tuning_state(state)
    cas_mode = _cas_mode(value, expected_parent_state_sha256)
    event_time = _utc(event_at_utc, "event_at_utc")
    lineage = _load_verified_lineage(lineage_events_dir, artifact_root)
    _require_event_not_before_lineage(event_time, lineage)
    latest_attempt = value["attempts"][-1] if value["attempts"] else None
    if (
        latest_attempt is not None
        and latest_attempt["terminal"] is not None
        and latest_attempt["terminal"]["status"] == "LINEAGE_RESULT_BOUND"
    ):
        existing_binding = latest_attempt["terminal"]["result_binding"]
        if existing_binding == _terminal_result_binding(lineage):
            return value
        raise DojoAITuningStateError("terminal bind replay changed its result")
    _require_current_cas(cas_mode)
    attempt = _current_attempt(value)
    if attempt is None or attempt["dispatch"] is None:
        raise DojoAITuningStateError("terminal result lacks a dispatched attempt")
    if (
        attempt["dispatch"]["status"] != "DISPATCHED"
        or value["phase"] != "AWAITING_LINEAGE_RESULT"
    ):
        raise DojoAITuningStateError(
            "terminal evaluation cannot bind before replay dispatch"
        )
    binding = _terminal_result_binding(lineage)
    if binding["attempt_ordinal"] != attempt["attempt_ordinal"]:
        raise DojoAITuningStateError("terminal result attempt ordinal drifted")
    if binding["study_sha256"] != attempt["dispatch"]["study_sha256"]:
        raise DojoAITuningStateError("terminal result belongs to another study")
    _require_registry_identity(value, lineage)
    if len(lineage.studies) != len(lineage.results):
        raise DojoAITuningStateError("terminal evaluation is not lineage-bound")

    updated = _clone(value)
    current = _current_attempt(updated)
    assert current is not None
    current["terminal"] = {
        "status": "LINEAGE_RESULT_BOUND",
        "recorded_at_utc": event_time,
        "result_binding": binding,
        "reason_code": None,
        "review_id": None,
        "review_rationale": None,
    }
    current["phase"] = "LINEAGE_RESULT_BOUND"
    updated["last_terminal_result_binding"] = binding
    updated["terminal_reason"] = None
    updated["phase"] = (
        "EXHAUSTED"
        if _attempts_used(updated) >= MAX_ATTEMPTS
        or _proposal_slots_used(updated) >= MAX_PROPOSAL_SLOTS
        else "READY_FOR_MODEL"
    )
    return _seal_next_state(updated, parent=value, transition_at_utc=event_time)


def mark_incomplete_run(
    state: Mapping[str, Any],
    *,
    expected_parent_state_sha256: str,
    reason_code: str,
    event_at_utc: datetime | str,
) -> dict[str, Any]:
    """Fail closed after a crash or missing terminal artifact.

    The attempt was consumed when its first model call was reserved.  It cannot
    be silently retried.  Review may later bind a genuine lineage result, or
    abandon the lineage; there is no same-attempt reset operation.
    """

    value = verify_tuning_state(state)
    cas_mode = _cas_mode(value, expected_parent_state_sha256)
    reason = _identifier(reason_code, "reason_code")
    event_time = _utc(event_at_utc, "event_at_utc")
    attempt = _current_attempt(value)
    if attempt is None:
        raise DojoAITuningStateError("no active attempt can be marked incomplete")
    if attempt["terminal"] is not None:
        terminal = attempt["terminal"]
        if (
            terminal["status"] == "INCOMPLETE_REVIEW_REQUIRED"
            and terminal["reason_code"] == reason
        ):
            return value
        raise DojoAITuningStateError("attempt already has a different terminal state")
    _require_current_cas(cas_mode)
    updated = _clone(value)
    current = _current_attempt(updated)
    assert current is not None
    current["terminal"] = {
        "status": "INCOMPLETE_REVIEW_REQUIRED",
        "recorded_at_utc": event_time,
        "result_binding": None,
        "reason_code": reason,
        "review_id": None,
        "review_rationale": None,
    }
    current["phase"] = "REVIEW_REQUIRED"
    updated["phase"] = "REVIEW_REQUIRED"
    updated["terminal_reason"] = reason
    return _seal_next_state(updated, parent=value, transition_at_utc=event_time)


def abandon_incomplete_lineage(
    state: Mapping[str, Any],
    *,
    expected_parent_state_sha256: str,
    review_id: str,
    rationale: str,
    event_at_utc: datetime | str,
) -> dict[str, Any]:
    """Review an incomplete attempt and terminate, without recycling budget."""

    value = verify_tuning_state(state)
    cas_mode = _cas_mode(value, expected_parent_state_sha256)
    review_key = _identifier(review_id, "review_id")
    review_rationale = _text(rationale, "rationale")
    event_time = _utc(event_at_utc, "event_at_utc")
    attempt = _current_attempt(value)
    if attempt is None or attempt["terminal"] is None:
        raise DojoAITuningStateError("abandonment requires incomplete-run review")
    terminal = attempt["terminal"]
    if terminal["status"] == "ABANDONED_AFTER_REVIEW":
        if (
            terminal["review_id"] == review_key
            and terminal["review_rationale"] == review_rationale
        ):
            return value
        raise DojoAITuningStateError("review replay changed abandonment evidence")
    _require_current_cas(cas_mode)
    if terminal["status"] != "INCOMPLETE_REVIEW_REQUIRED":
        raise DojoAITuningStateError("only an incomplete attempt can be abandoned")
    updated = _clone(value)
    current = _current_attempt(updated)
    assert current is not None and current["terminal"] is not None
    current["terminal"].update(
        {
            "status": "ABANDONED_AFTER_REVIEW",
            "recorded_at_utc": event_time,
            "review_id": review_key,
            "review_rationale": review_rationale,
        }
    )
    current["phase"] = "ABANDONED_AFTER_REVIEW"
    updated["phase"] = "TERMINATED"
    updated["terminal_reason"] = terminal["reason_code"]
    return _seal_next_state(updated, parent=value, transition_at_utc=event_time)


def verify_tuning_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the state digest, budgets, phases, and zero-authority boundary."""

    row = _exact_mapping(state, _STATE_KEYS, "AI tuning state")
    if row["contract"] != STATE_CONTRACT or row["schema_version"] != SCHEMA_VERSION:
        raise DojoAITuningStateError("tuning state contract/version drifted")
    claimed = _sha(row["state_sha256"], "state_sha256")
    body = {key: value for key, value in row.items() if key != "state_sha256"}
    if _canonical_sha(body) != claimed:
        raise DojoAITuningStateError("tuning state digest mismatch")
    if (
        row["max_attempts"] != MAX_ATTEMPTS
        or row["max_proposal_slots"] != MAX_PROPOSAL_SLOTS
    ):
        raise DojoAITuningStateError("tuning search budget drifted")
    for key, expected in _AUTHORITY.items():
        if row[key] != expected:
            raise DojoAITuningStateError("tuning state exceeds research authority")
    _identifier(row["registry_id"], "registry_id")
    _identifier(row["lineage_prefix"], "lineage_prefix")
    envelope = _verify_envelope(row["fixed_envelope"])
    if row["fixed_envelope_sha256"] != envelope["envelope_sha256"]:
        raise DojoAITuningStateError("fixed envelope binding drifted")
    revision = _integer(row["revision"], "revision")
    if revision == 0:
        if row["previous_state_sha256"] is not None:
            raise DojoAITuningStateError("genesis state claims a parent")
    else:
        _sha(row["previous_state_sha256"], "previous_state_sha256")
    initialized_at = _utc(row["initialized_at_utc"], "initialized_at_utc")
    last_transition_at = _utc(row["last_transition_at_utc"], "last_transition_at_utc")
    if _parse_utc_value(last_transition_at) < _parse_utc_value(initialized_at):
        raise DojoAITuningStateError("state transition clock predates initialization")
    initial_attempts = _integer(
        row["initial_attempts_consumed"], "initial_attempts_consumed"
    )
    initial_slots = _integer(
        row["initial_proposal_slots_consumed"],
        "initial_proposal_slots_consumed",
    )
    if not 1 <= initial_attempts <= MAX_ATTEMPTS:
        raise DojoAITuningStateError("initial attempt count is out of range")
    if not 1 <= initial_slots <= MAX_PROPOSAL_SLOTS:
        raise DojoAITuningStateError("initial proposal count is out of range")
    binding = _verify_result_binding(row["last_terminal_result_binding"])
    if (
        binding["registry_id"] != row["registry_id"]
        or binding["lineage_prefix"] != row["lineage_prefix"]
    ):
        raise DojoAITuningStateError("terminal result registry identity drifted")

    attempts_raw = row["attempts"]
    if not isinstance(attempts_raw, list):
        raise DojoAITuningStateError("attempts must be a JSON array")
    if len(attempts_raw) > MAX_ATTEMPTS:
        raise DojoAITuningStateError("too many AI attempt records")
    expected_ordinal = initial_attempts + 1
    seen_invocations: set[str] = set()
    seen_submissions: set[str] = set()
    accepted_identities: set[str] = set()
    expected_prior_binding: dict[str, Any] | None = None
    for index, item in enumerate(attempts_raw):
        attempt = _verify_attempt(item)
        if attempt["attempt_ordinal"] != expected_ordinal + index:
            raise DojoAITuningStateError("AI attempt ordinal has a gap or fork")
        if attempt["envelope_sha256"] != envelope["envelope_sha256"]:
            raise DojoAITuningStateError("attempt changed the fixed envelope")
        if index == 0:
            expected_prior_binding = attempt["prior_result_binding"]
            if expected_prior_binding["attempt_ordinal"] != initial_attempts:
                raise DojoAITuningStateError(
                    "first AI attempt does not follow the initial lineage result"
                )
        if attempt["prior_result_binding"] != expected_prior_binding:
            raise DojoAITuningStateError("AI attempt prior-result chain forked")
        for invocation in attempt["invocations"]:
            if invocation["invocation_id"] in seen_invocations:
                raise DojoAITuningStateError("invocation id is not globally unique")
            seen_invocations.add(invocation["invocation_id"])
            for submission in invocation["submissions"]:
                if submission["submission_id"] in seen_submissions:
                    raise DojoAITuningStateError("submission id is not globally unique")
                seen_submissions.add(submission["submission_id"])
                if submission["status"] == "ACCEPTED":
                    identity = submission["executable_identity_sha256"]
                    assert identity is not None
                    if identity in accepted_identities:
                        raise DojoAITuningStateError(
                            "accepted executable identity repeats across attempts"
                        )
                    accepted_identities.add(identity)
        _verify_attempt_dispatch_crosscheck(
            attempt, envelope_sha256=envelope["envelope_sha256"]
        )
        terminal = attempt["terminal"]
        if terminal is not None and terminal["status"] == "LINEAGE_RESULT_BOUND":
            result_binding = terminal["result_binding"]
            assert result_binding is not None
            if (
                result_binding["registry_id"] != row["registry_id"]
                or result_binding["lineage_prefix"] != row["lineage_prefix"]
                or result_binding["attempt_ordinal"] != attempt["attempt_ordinal"]
                or attempt["dispatch"] is None
                or attempt["dispatch"]["status"] != "DISPATCHED"
                or result_binding["study_sha256"] != attempt["dispatch"]["study_sha256"]
            ):
                raise DojoAITuningStateError(
                    "attempt terminal result does not bind its dispatched study"
                )
            expected_prior_binding = result_binding
        elif index != len(attempts_raw) - 1:
            raise DojoAITuningStateError(
                "a non-terminal attempt cannot precede another attempt"
            )
    if expected_prior_binding is not None and binding != expected_prior_binding:
        raise DojoAITuningStateError(
            "top-level last result diverges from the attempt result chain"
        )
    if not attempts_raw and binding["attempt_ordinal"] != initial_attempts:
        raise DojoAITuningStateError(
            "initial result binding does not match consumed attempts"
        )
    if revision != _expected_revision(row):
        raise DojoAITuningStateError(
            "state revision diverges from recorded transitions"
        )
    if _attempts_used(row) > MAX_ATTEMPTS:
        raise DojoAITuningStateError("attempt budget exceeded")
    global_identities = _sha_list(
        row["global_executable_identity_sha256s"],
        "global_executable_identity_sha256s",
    )
    if global_identities != sorted(global_identities):
        raise DojoAITuningStateError("global executable identities are not sorted")
    if not accepted_identities.issubset(set(global_identities)):
        raise DojoAITuningStateError(
            "accepted proposal identity is not globally burned"
        )
    if len(global_identities) != initial_slots + len(accepted_identities):
        raise DojoAITuningStateError(
            "accepted executable identity repeats a pre-existing lineage identity"
        )
    used_slots = _proposal_slots_used(row)
    if used_slots > MAX_PROPOSAL_SLOTS and row["phase"] != "REVIEW_REQUIRED":
        raise DojoAITuningStateError("proposal budget exceeded outside review")
    _verify_top_phase(row)
    _verify_transition_timestamps(row)
    return _clone(row)


def status_artifact(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact research-only status for orchestration and reports."""

    value = verify_tuning_state(state)
    attempt = _current_attempt(value)
    accepted = _accepted_rows(attempt) if attempt is not None else []
    invalid = 0
    duplicate = 0
    invocations = 0
    for item in value["attempts"]:
        invocations += len(item["invocations"])
        for invocation in item["invocations"]:
            invalid += sum(
                submission["status"] in {"INVALID", "BUDGET_REJECTED"}
                for submission in invocation["submissions"]
            )
            duplicate += sum(
                submission["status"] == "DUPLICATE"
                for submission in invocation["submissions"]
            )
    return {
        "contract": STATUS_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "registry_id": value["registry_id"],
        "phase": value["phase"],
        "attempts_consumed": _attempts_used(value),
        "max_attempts": MAX_ATTEMPTS,
        "proposal_slots_consumed": _proposal_slots_used(value),
        "max_proposal_slots": MAX_PROPOSAL_SLOTS,
        "model_invocation_count": invocations,
        "invalid_proposal_count": invalid,
        "duplicate_proposal_count": duplicate,
        "current_accepted_candidate_ids": [row["candidate_id"] for row in accepted],
        "fixed_envelope_sha256": value["fixed_envelope_sha256"],
        "terminal_reason": value["terminal_reason"],
        **_AUTHORITY,
    }


def initialize_state_store(
    events_dir: Path, state: Mapping[str, Any]
) -> dict[str, Any]:
    """Persist the revision-zero state in a new append-only local store.

    A bare in-memory state deliberately reports ``automation_ready=false``.
    Only a fully re-read store snapshot reports that its local CAS discipline is
    ready for an orchestrator.  This still grants no trading or broker authority.
    """

    value = verify_tuning_state(state)
    if value["revision"] != 0 or value["previous_state_sha256"] is not None:
        raise DojoAITuningStateError(
            "state store must start from the revision-zero tuning state"
        )
    directory_fd = _open_state_store_directory(Path(events_dir), create=True)
    try:
        if _state_store_event_names(directory_fd):
            raise DojoAITuningStateError("AI tuning state store is already initialized")
        event = _new_store_event(
            sequence=0,
            previous_event_sha256=None,
            parent_state_sha256=None,
            state=value,
        )
        _write_store_event_exclusive(directory_fd, "000000.json", event)
    finally:
        os.close(directory_fd)
    return verify_state_store(Path(events_dir))


def append_state_transition(
    events_dir: Path,
    state: Mapping[str, Any],
    *,
    expected_tip_event_sha256: str,
    expected_parent_state_sha256: str,
) -> dict[str, Any]:
    """CAS-append exactly one verified state transition.

    Competing writers may derive states from the same parent, but the exclusive
    ordinal slot permits only one child.  Retrying the already-written identical
    child is idempotent; a different child must reload and derive a new valid
    transition from the current tip.
    """

    value = verify_tuning_state(state)
    expected_tip = _sha(expected_tip_event_sha256, "expected_tip_event_sha256")
    expected_parent = _sha(expected_parent_state_sha256, "expected_parent_state_sha256")
    snapshot = verify_state_store(Path(events_dir))
    latest_event = snapshot["latest_event"]
    latest_state = snapshot["latest_state"]

    # A caller may crash after the durable fsync and retry with its former CAS
    # tokens.  Only the byte-identical immediately appended child is a no-op.
    if value["state_sha256"] == latest_state["state_sha256"]:
        if (
            latest_event["previous_event_sha256"] == expected_tip
            and latest_event["parent_state_sha256"] == expected_parent
        ):
            return snapshot
        raise DojoAITuningStateError("state-store replay does not name its parent")

    if snapshot["latest_event_sha256"] != expected_tip:
        raise DojoAITuningStateError("stale or forked state-store event tip")
    if latest_state["state_sha256"] != expected_parent:
        raise DojoAITuningStateError("stale or forked state-store parent state")
    if (
        value["previous_state_sha256"] != expected_parent
        or value["revision"] != latest_state["revision"] + 1
    ):
        raise DojoAITuningStateError(
            "appended state is not the immediate child of the store tip"
        )
    if _parse_utc_value(value["last_transition_at_utc"]) < _parse_utc_value(
        latest_state["last_transition_at_utc"]
    ):
        raise DojoAITuningStateError("appended state timestamp moved backward")

    sequence = snapshot["event_count"]
    if sequence >= MAX_STATE_STORE_EVENTS:
        raise DojoAITuningStateError("AI tuning state-store event limit exceeded")
    event = _new_store_event(
        sequence=sequence,
        previous_event_sha256=expected_tip,
        parent_state_sha256=expected_parent,
        state=value,
    )
    directory_fd = _open_state_store_directory(Path(events_dir), create=False)
    try:
        _write_store_event_exclusive(directory_fd, f"{sequence:06d}.json", event)
    finally:
        os.close(directory_fd)
    return verify_state_store(Path(events_dir))


def verify_state_store(events_dir: Path) -> dict[str, Any]:
    """Rebuild and verify every append-only tuning-state event from disk."""

    directory_fd = _open_state_store_directory(Path(events_dir), create=False)
    try:
        names = _state_store_event_names(directory_fd)
        if not names:
            raise DojoAITuningStateError("AI tuning state store lacks genesis")
        if len(names) > MAX_STATE_STORE_EVENTS:
            raise DojoAITuningStateError("AI tuning state-store event limit exceeded")
        expected_names = [f"{sequence:06d}.json" for sequence in range(len(names))]
        if names != expected_names:
            raise DojoAITuningStateError("AI tuning state store has a gap or fork")
        events = [_read_store_event(directory_fd, name) for name in names]
        if _state_store_event_names(directory_fd) != names:
            raise DojoAITuningStateError(
                "AI tuning state store changed while being read"
            )
    finally:
        os.close(directory_fd)

    previous_event_sha: str | None = None
    previous_state: dict[str, Any] | None = None
    verified_events: list[dict[str, Any]] = []
    for sequence, raw_event in enumerate(events):
        event = _verify_store_event(raw_event)
        if event["sequence"] != sequence:
            raise DojoAITuningStateError("state-store event sequence drifted")
        if event["previous_event_sha256"] != previous_event_sha:
            raise DojoAITuningStateError("state-store event SHA chain drifted")
        current_state = event["state"]
        if previous_state is None:
            if (
                sequence != 0
                or event["parent_state_sha256"] is not None
                or current_state["revision"] != 0
                or current_state["previous_state_sha256"] is not None
            ):
                raise DojoAITuningStateError("state-store genesis is not revision zero")
        else:
            parent_sha = previous_state["state_sha256"]
            if (
                event["parent_state_sha256"] != parent_sha
                or current_state["previous_state_sha256"] != parent_sha
                or current_state["revision"] != previous_state["revision"] + 1
            ):
                raise DojoAITuningStateError("state-store state chain forked")
            if _parse_utc_value(
                current_state["last_transition_at_utc"]
            ) < _parse_utc_value(previous_state["last_transition_at_utc"]):
                raise DojoAITuningStateError(
                    "state-store transition clock moved backward"
                )
        verified_events.append(event)
        previous_event_sha = event["event_sha256"]
        previous_state = current_state

    assert previous_state is not None and previous_event_sha is not None
    return {
        "event_count": len(verified_events),
        "latest_sequence": len(verified_events) - 1,
        "latest_event_sha256": previous_event_sha,
        "latest_event": _clone(verified_events[-1]),
        "latest_state": _clone(previous_state),
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


def _derive_dispatch(
    state: Mapping[str, Any],
    lineage: CandidateLineageSnapshot,
    sealed_study: Mapping[str, Any],
    *,
    dispatch_id: str,
    reserved_at_utc: str,
) -> dict[str, Any]:
    _require_registry_identity(state, lineage)
    attempt = _current_attempt(state)
    assert attempt is not None
    ordinal = attempt["attempt_ordinal"]
    if len(lineage.studies) != ordinal or len(lineage.results) != ordinal - 1:
        raise DojoAITuningStateError(
            "run dispatch requires exactly one pending lineage-sealed study"
        )
    if lineage.events[-1]["event_type"] not in {"STUDY_SEALED", "NEXT_STUDY_SEALED"}:
        raise DojoAITuningStateError("latest lineage event is not the pending study")
    latest_study = lineage.studies[-1]
    if lineage.events[-1]["body"] != latest_study:
        raise DojoAITuningStateError("pending study event body is not exact")
    if latest_study["attempt_ordinal"] != ordinal:
        raise DojoAITuningStateError("pending study attempt ordinal drifted")
    expected_previous = _previous_triple(attempt["prior_result_binding"])
    if latest_study["previous_attempt_evaluation_binding"] != expected_previous:
        raise DojoAITuningStateError(
            "pending study lacks the exact prior evaluation triple"
        )
    try:
        sealed = verify_sealed_study(sealed_study, sealed_study["source_digests"])
    except (KeyError, DojoBotTrainerError) as exc:
        raise DojoAITuningStateError("dispatch study seal is invalid") from exc
    if sealed["study_sha256"] != latest_study["study_sha256"]:
        raise DojoAITuningStateError("dispatch study is not lineage-bound")
    observed_envelope = fixed_envelope_from_sealed_study(sealed)
    if observed_envelope != state["fixed_envelope"]:
        raise DojoAITuningStateError(
            "dispatch changed risk, window, cost, or scorer envelope"
        )
    accepted = _accepted_rows(attempt)
    if not accepted:
        raise DojoAITuningStateError("dispatch has no valid unique AI proposal")
    by_candidate = {row["candidate_id"]: row for row in accepted}
    candidate_ids = list(latest_study["candidate_ids"])
    if set(candidate_ids) != set(by_candidate) or len(candidate_ids) != len(
        by_candidate
    ):
        raise DojoAITuningStateError(
            "lineage study candidate denominator differs from accepted AI proposals"
        )
    proposal_sha256s = [
        by_candidate[candidate]["proposal_sha256"] for candidate in candidate_ids
    ]
    identities = [
        by_candidate[candidate]["executable_identity_sha256"]
        for candidate in candidate_ids
    ]
    if sorted(proposal_sha256s) != list(latest_study["proposal_sha256s"]):
        raise DojoAITuningStateError("lineage study proposal seals drifted")
    if sorted(identities) != list(latest_study["config_sha256s"]):
        raise DojoAITuningStateError("lineage study executable identities drifted")
    return {
        "dispatch_id": dispatch_id,
        "status": "RESERVED",
        "reserved_at_utc": reserved_at_utc,
        "dispatched_at_utc": None,
        "study_sha256": sealed["study_sha256"],
        "study_event_sha256": lineage.events[-1]["event_sha256"],
        "study_event_sequence": lineage.events[-1]["sequence"],
        "candidate_ids": candidate_ids,
        "proposal_sha256s": proposal_sha256s,
        "executable_identity_sha256s": identities,
        "prior_result_binding": _clone(attempt["prior_result_binding"]),
        "envelope_sha256": state["fixed_envelope_sha256"],
    }


def _verify_dispatch_inputs(
    state: Mapping[str, Any],
    lineage: CandidateLineageSnapshot,
    sealed_study: Mapping[str, Any],
    existing: Mapping[str, Any],
) -> None:
    derived = _derive_dispatch(
        state,
        lineage,
        sealed_study,
        dispatch_id=existing["dispatch_id"],
        reserved_at_utc=existing["reserved_at_utc"],
    )
    if existing["status"] == "DISPATCHED":
        derived["status"] = "DISPATCHED"
        derived["dispatched_at_utc"] = existing["dispatched_at_utc"]
    if derived != existing:
        raise DojoAITuningStateError("dispatch replay changed its sealed inputs")


def _require_reserved_dispatch_lineage(
    state: Mapping[str, Any], lineage: CandidateLineageSnapshot
) -> None:
    """Revalidate the reserved run against the current registry and artifacts."""

    _require_registry_identity(state, lineage)
    attempt = _current_attempt(state)
    if attempt is None or attempt["dispatch"] is None:
        raise DojoAITuningStateError("active dispatch binding is absent")
    dispatch = attempt["dispatch"]
    ordinal = attempt["attempt_ordinal"]
    if len(lineage.studies) != ordinal or len(lineage.results) != ordinal - 1:
        raise DojoAITuningStateError(
            "reserved dispatch no longer has one pending lineage study"
        )
    study = lineage.studies[-1]
    event = lineage.events[-1]
    if (
        study["attempt_ordinal"] != ordinal
        or event["event_type"] not in {"STUDY_SEALED", "NEXT_STUDY_SEALED"}
        or event["body"] != study
        or dispatch["study_sha256"] != study["study_sha256"]
        or dispatch["study_event_sha256"] != event["event_sha256"]
        or dispatch["study_event_sequence"] != event["sequence"]
        or study["previous_attempt_evaluation_binding"]
        != _previous_triple(attempt["prior_result_binding"])
        or dispatch["candidate_ids"] != list(study["candidate_ids"])
        or sorted(dispatch["proposal_sha256s"]) != list(study["proposal_sha256s"])
        or sorted(dispatch["executable_identity_sha256s"])
        != list(study["config_sha256s"])
    ):
        raise DojoAITuningStateError(
            "reserved dispatch diverges from the verified lineage study"
        )


def _verify_attempt_dispatch_crosscheck(
    attempt: Mapping[str, Any], *, envelope_sha256: str
) -> None:
    """Bind dispatch fields to accepted model submissions, not self-claims."""

    dispatch = attempt["dispatch"]
    if dispatch is None:
        if attempt["phase"] in {
            "RUN_DISPATCH_RESERVED",
            "AWAITING_LINEAGE_RESULT",
            "LINEAGE_RESULT_BOUND",
        }:
            raise DojoAITuningStateError("attempt phase claims a missing dispatch")
        return
    if (
        dispatch["prior_result_binding"] != attempt["prior_result_binding"]
        or dispatch["envelope_sha256"] != attempt["envelope_sha256"]
        or dispatch["envelope_sha256"] != envelope_sha256
    ):
        raise DojoAITuningStateError(
            "dispatch prior-result or envelope binding drifted"
        )
    accepted = _accepted_rows(attempt)
    candidate_ids = [row["candidate_id"] for row in accepted]
    if any(candidate is None for candidate in candidate_ids) or len(
        set(candidate_ids)
    ) != len(candidate_ids):
        raise DojoAITuningStateError("accepted candidate identity is ambiguous")
    by_candidate = {row["candidate_id"]: row for row in accepted}
    if len(dispatch["candidate_ids"]) != len(by_candidate) or set(
        dispatch["candidate_ids"]
    ) != set(by_candidate):
        raise DojoAITuningStateError(
            "dispatch candidate set differs from accepted submissions"
        )
    expected_proposals = [
        by_candidate[candidate]["proposal_sha256"]
        for candidate in dispatch["candidate_ids"]
    ]
    expected_configs = [
        by_candidate[candidate]["executable_identity_sha256"]
        for candidate in dispatch["candidate_ids"]
    ]
    if (
        dispatch["proposal_sha256s"] != expected_proposals
        or dispatch["executable_identity_sha256s"] != expected_configs
    ):
        raise DojoAITuningStateError(
            "dispatch proposal/config set differs from accepted submissions"
        )
    if dispatch["status"] == "RESERVED" and attempt["phase"] not in {
        "RUN_DISPATCH_RESERVED",
        "REVIEW_REQUIRED",
    }:
        raise DojoAITuningStateError("reserved dispatch phase drifted")
    if dispatch["status"] == "DISPATCHED" and attempt["phase"] not in {
        "AWAITING_LINEAGE_RESULT",
        "LINEAGE_RESULT_BOUND",
        "REVIEW_REQUIRED",
        "ABANDONED_AFTER_REVIEW",
    }:
        raise DojoAITuningStateError("dispatched run phase drifted")


def _evaluate_submission(
    submission: Mapping[str, Any],
    *,
    lineage_prefix: str,
    known_identities: set[str],
    force_budget_reject: bool,
) -> dict[str, Any]:
    raw = submission["proposal"]
    declared_errors = list(submission["validation_errors"])
    sealed: dict[str, Any] | None = None
    errors: list[str] = []
    if raw is None:
        errors = declared_errors or ["MODEL_PROPOSAL_UNPARSEABLE"]
    elif declared_errors:
        errors = declared_errors
    else:
        try:
            sealed = seal_candidate_proposal(raw)
        except DojoBotTrainerError:
            errors = ["CATALOG_OR_PROPOSAL_CONTRACT_INVALID"]
    candidate_id = sealed["candidate_id"] if sealed is not None else None
    proposal_sha = sealed["proposal_sha256"] if sealed is not None else None
    identity = sealed["config_sha256"] if sealed is not None else None
    if sealed is not None and not candidate_id.startswith(lineage_prefix):
        errors = ["CANDIDATE_ID_OUTSIDE_LINEAGE_PREFIX"]
        sealed = None
        proposal_sha = None
        identity = None
    if force_budget_reject:
        status = "BUDGET_REJECTED"
        errors = ["MODEL_RESPONSE_EXCEEDED_PROPOSAL_BUDGET"]
        sealed = None
        proposal_sha = None
        identity = None
    elif errors:
        status = "INVALID"
        sealed = None
        proposal_sha = None
        identity = None
    elif identity in known_identities:
        status = "DUPLICATE"
        errors = ["GLOBAL_EXECUTABLE_IDENTITY_ALREADY_USED"]
    else:
        status = "ACCEPTED"
    return {
        "submission_id": submission["submission_id"],
        "raw_proposal_sha256": submission["raw_proposal_sha256"],
        "status": status,
        "validation_errors": errors,
        "candidate_id": candidate_id,
        "proposal_sha256": proposal_sha,
        "executable_identity_sha256": identity,
        "sealed_proposal": sealed,
    }


def _normalize_submission_input(value: Any) -> dict[str, Any]:
    row = _exact_mapping(
        value,
        {
            "submission_id",
            "raw_proposal_sha256",
            "proposal",
            "validation_errors",
        },
        "proposal submission",
    )
    errors = _text_list(row["validation_errors"], "validation_errors")
    proposal = _json_value(row["proposal"], "proposal")
    return {
        "submission_id": _identifier(row["submission_id"], "submission_id"),
        "raw_proposal_sha256": _sha(row["raw_proposal_sha256"], "raw_proposal_sha256"),
        "proposal": proposal,
        "validation_errors": errors,
    }


def _chargeable_submission_inputs(
    value: Any,
    *,
    response_sha256: str,
    invocation_id: str,
    forbidden_submission_ids: set[str],
) -> list[dict[str, Any]]:
    """Turn malformed model output into durable, budget-consuming records."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [
            _malformed_submission(
                value,
                response_sha256=response_sha256,
                invocation_id=invocation_id,
                index=0,
                error="MODEL_RESPONSE_SUBMISSIONS_NOT_ARRAY",
            )
        ]
    if not value:
        return [
            _malformed_submission(
                value,
                response_sha256=response_sha256,
                invocation_id=invocation_id,
                index=0,
                error="MODEL_RESPONSE_NO_PROPOSALS",
            )
        ]
    rows: list[dict[str, Any]] = []
    seen = set(forbidden_submission_ids)
    for index, item in enumerate(value):
        try:
            normalized = _normalize_submission_input(item)
        except DojoAITuningStateError:
            normalized = _malformed_submission(
                item,
                response_sha256=response_sha256,
                invocation_id=invocation_id,
                index=index,
                error="MODEL_PROPOSAL_RECORD_MALFORMED",
            )
        if normalized["submission_id"] in seen:
            normalized = _malformed_submission(
                item,
                response_sha256=response_sha256,
                invocation_id=invocation_id,
                index=index,
                error="SUBMISSION_ID_REUSED",
            )
        if normalized["submission_id"] in seen:
            base = normalized["submission_id"]
            suffix = 1
            while f"{base}-{suffix}" in seen:
                suffix += 1
            normalized["submission_id"] = f"{base}-{suffix}"
        seen.add(normalized["submission_id"])
        rows.append(normalized)
    return rows


def _malformed_submission(
    value: Any,
    *,
    response_sha256: str,
    invocation_id: str,
    index: int,
    error: str,
) -> dict[str, Any]:
    try:
        raw_sha = _canonical_sha(value)
    except DojoAITuningStateError:
        raw_sha = response_sha256
    return {
        "submission_id": (
            f"invalid-{_canonical_sha({'response': response_sha256, 'invocation': invocation_id})[:16]}-{index}"
        ),
        "raw_proposal_sha256": raw_sha,
        "proposal": None,
        "validation_errors": [error],
    }


def _terminal_result_binding(
    lineage: CandidateLineageSnapshot,
) -> dict[str, Any]:
    _verified_lineage_identity(lineage)
    if not lineage.results or len(lineage.studies) != len(lineage.results):
        raise DojoAITuningStateError("latest evaluation is not lineage-bound")
    event = lineage.events[-1]
    result = lineage.results[-1]
    if event["event_type"] != "RESULT_BOUND" or event["body"] != result:
        raise DojoAITuningStateError("latest lineage result event is not exact")
    return {
        "registry_id": lineage.registry_id,
        "lineage_prefix": lineage.lineage_prefix,
        "attempt_ordinal": result["attempt_ordinal"],
        "study_sha256": result["study_sha256"],
        "evaluation_sha256": result["evaluation_sha256"],
        "evaluation_artifact_sha256": result["evaluation_artifact_sha256"],
        "evaluation_artifact_size_bytes": result["evaluation_artifact_size_bytes"],
        "result_event_sha256": event["event_sha256"],
        "result_event_sequence": event["sequence"],
        "lineage_tip_sha256": lineage.latest_event_sha256,
    }


def _require_terminal_lineage(
    state: Mapping[str, Any], lineage: CandidateLineageSnapshot
) -> None:
    _require_registry_identity(state, lineage)
    binding = _terminal_result_binding(lineage)
    if binding != state["last_terminal_result_binding"]:
        raise DojoAITuningStateError(
            "model dispatch requires the exact last lineage-bound evaluation triple "
            "and RESULT_BOUND event"
        )


def _require_registry_identity(
    state: Mapping[str, Any], lineage: CandidateLineageSnapshot
) -> None:
    _verified_lineage_identity(lineage)
    if (
        lineage.registry_id != state["registry_id"]
        or lineage.lineage_prefix != state["lineage_prefix"]
    ):
        raise DojoAITuningStateError("candidate lineage identity changed")


def _require_event_not_before_lineage(
    event_at_utc: str, lineage: CandidateLineageSnapshot
) -> None:
    if _parse_utc_value(event_at_utc) < _parse_utc_value(
        _utc(lineage.latest_event_at_utc, "lineage.latest_event_at_utc")
    ):
        raise DojoAITuningStateError(
            "tuning transition timestamp predates its verified lineage event"
        )


def _load_verified_lineage(
    events_dir: Path, artifact_root: Path
) -> CandidateLineageSnapshot:
    """Load lineage only through the artifact-revalidating registry verifier."""

    if isinstance(events_dir, CandidateLineageSnapshot) or isinstance(
        artifact_root, CandidateLineageSnapshot
    ):
        raise DojoAITuningStateError(
            "caller-constructed lineage snapshots are not accepted"
        )
    try:
        return verify_registry(Path(events_dir), artifact_root=Path(artifact_root))
    except (CandidateLineageError, OSError, TypeError, ValueError) as exc:
        raise DojoAITuningStateError(
            f"candidate lineage verification failed: {exc}"
        ) from exc


def _verified_lineage_identity(lineage: CandidateLineageSnapshot) -> None:
    if not isinstance(lineage, CandidateLineageSnapshot):
        raise DojoAITuningStateError("a verified CandidateLineageSnapshot is required")
    _identifier(lineage.registry_id, "registry_id")
    _identifier(lineage.lineage_prefix, "lineage_prefix")
    _sha(lineage.latest_event_sha256, "latest_event_sha256")
    if not lineage.events or lineage.event_count != len(lineage.events):
        raise DojoAITuningStateError("candidate lineage event count drifted")
    latest = lineage.events[-1]
    if (
        latest.get("event_sha256") != lineage.latest_event_sha256
        or latest.get("sequence") != lineage.latest_sequence
    ):
        raise DojoAITuningStateError("candidate lineage tip drifted")


def _previous_triple(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attempt_ordinal": binding["attempt_ordinal"],
        "evaluation_sha256": binding["evaluation_sha256"],
        "evaluation_artifact_sha256": binding["evaluation_artifact_sha256"],
        "evaluation_artifact_size_bytes": binding["evaluation_artifact_size_bytes"],
    }


def _verify_envelope(value: Any) -> dict[str, Any]:
    row = _exact_mapping(value, _ENVELOPE_KEYS, "fixed envelope")
    claimed = _sha(row["envelope_sha256"], "envelope_sha256")
    body = {key: item for key, item in row.items() if key != "envelope_sha256"}
    if _canonical_sha(body) != claimed:
        raise DojoAITuningStateError("fixed envelope digest mismatch")
    if (
        row["contract"] != ENVELOPE_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
        or row["window_role"] != "TRAIN"
        or row["classification"] != "WORN_HISTORICAL_TRAIN_ONLY"
        or row["window"].get("evidence_tier") != "WORN_TRAIN"
        or row["holdout_access_allowed"] is not False
        or row["forward_access_allowed"] is not False
        or row["live_permission"] is not False
        or row["order_authority"] != "NONE"
        or row["broker_mutation_allowed"] is not False
    ):
        raise DojoAITuningStateError("fixed envelope exceeds worn TRAIN authority")
    if list(row["intrabar_paths"]) != list(REQUIRED_INTRABAR_PATHS):
        raise DojoAITuningStateError("fixed intrabar denominator drifted")
    if row["risk_policy_contract"] != RISK_POLICY_CONTRACT:
        raise DojoAITuningStateError("risk policy contract drifted")
    if row["risk_policy_sha256"] != _canonical_sha(bot_risk_policy_manifest()):
        raise DojoAITuningStateError("risk policy digest drifted")
    if row["scorer_contract"] != EVALUATION_CONTRACT:
        raise DojoAITuningStateError("scorer contract drifted")
    if row["source_bundle_sha256"] != _canonical_sha(row["source_digests"]):
        raise DojoAITuningStateError("scorer source bundle digest drifted")
    return _clone(row)


def _verify_result_binding(value: Any) -> dict[str, Any]:
    row = _exact_mapping(value, _RESULT_BINDING_KEYS, "result binding")
    normalized = {
        "registry_id": _identifier(row["registry_id"], "registry_id"),
        "lineage_prefix": _identifier(row["lineage_prefix"], "lineage_prefix"),
        "attempt_ordinal": _positive_integer(row["attempt_ordinal"], "attempt_ordinal"),
        "study_sha256": _sha(row["study_sha256"], "study_sha256"),
        "evaluation_sha256": _sha(row["evaluation_sha256"], "evaluation_sha256"),
        "evaluation_artifact_sha256": _sha(
            row["evaluation_artifact_sha256"],
            "evaluation_artifact_sha256",
        ),
        "evaluation_artifact_size_bytes": _positive_integer(
            row["evaluation_artifact_size_bytes"],
            "evaluation_artifact_size_bytes",
        ),
        "result_event_sha256": _sha(row["result_event_sha256"], "result_event_sha256"),
        "result_event_sequence": _integer(
            row["result_event_sequence"], "result_event_sequence"
        ),
        "lineage_tip_sha256": _sha(row["lineage_tip_sha256"], "lineage_tip_sha256"),
    }
    if normalized != row:
        raise DojoAITuningStateError("result binding is not canonical")
    return normalized


def _verify_attempt(value: Any) -> dict[str, Any]:
    row = _exact_mapping(value, _ATTEMPT_KEYS, "AI attempt")
    _positive_integer(row["attempt_ordinal"], "attempt_ordinal")
    _verify_result_binding(row["prior_result_binding"])
    _sha(row["envelope_sha256"], "envelope_sha256")
    if not isinstance(row["invocations"], list) or len(row["invocations"]) != 1:
        raise DojoAITuningStateError(
            "AI attempt must contain exactly one model invocation"
        )
    for invocation in row["invocations"]:
        _verify_invocation(invocation)
    if row["dispatch"] is not None:
        _verify_dispatch(row["dispatch"])
    if row["terminal"] is not None:
        _verify_terminal(row["terminal"])
    _verify_attempt_phase(row)
    return _clone(row)


def _verify_attempt_phase(attempt: Mapping[str, Any]) -> None:
    phase = attempt["phase"]
    dispatch = attempt["dispatch"]
    terminal = attempt["terminal"]
    incomplete_invocation = any(
        invocation["response_sha256"] is None for invocation in attempt["invocations"]
    )
    if phase == "MODEL_INVOCATION_RESERVED":
        valid = incomplete_invocation and dispatch is None and terminal is None
    elif phase == "COLLECTING_PROPOSALS":
        valid = not incomplete_invocation and dispatch is None and terminal is None
    elif phase == "RUN_DISPATCH_RESERVED":
        valid = (
            not incomplete_invocation
            and dispatch is not None
            and dispatch["status"] == "RESERVED"
            and terminal is None
        )
    elif phase == "AWAITING_LINEAGE_RESULT":
        valid = (
            not incomplete_invocation
            and dispatch is not None
            and dispatch["status"] == "DISPATCHED"
            and terminal is None
        )
    elif phase == "LINEAGE_RESULT_BOUND":
        valid = terminal is not None and terminal["status"] == "LINEAGE_RESULT_BOUND"
    elif phase == "REVIEW_REQUIRED":
        valid = (
            terminal is not None and terminal["status"] == "INCOMPLETE_REVIEW_REQUIRED"
        ) or (terminal is None and not incomplete_invocation and dispatch is None)
    elif phase == "ABANDONED_AFTER_REVIEW":
        valid = terminal is not None and terminal["status"] == "ABANDONED_AFTER_REVIEW"
    else:
        raise DojoAITuningStateError("attempt phase is unsupported")
    if not valid:
        raise DojoAITuningStateError("attempt phase internals are inconsistent")


def _verify_invocation(value: Any) -> dict[str, Any]:
    row = _exact_mapping(value, _INVOCATION_KEYS, "model invocation")
    _identifier(row["invocation_id"], "invocation_id")
    _sha(row["request_sha256"], "request_sha256")
    _utc(row["reserved_at_utc"], "reserved_at_utc")
    if row["response_sha256"] is None:
        if (
            row["response_input_sha256"] is not None
            or row["response_recorded_at_utc"] is not None
            or row["proposal_slot_charge"] != 0
            or row["submissions"] != []
        ):
            raise DojoAITuningStateError("reserved invocation carries response data")
        return _clone(row)
    _sha(row["response_sha256"], "response_sha256")
    _sha(row["response_input_sha256"], "response_input_sha256")
    _utc(row["response_recorded_at_utc"], "response_recorded_at_utc")
    if not isinstance(row["submissions"], list):
        raise DojoAITuningStateError("invocation submissions must be an array")
    expected_charge = max(1, len(row["submissions"]))
    if row["proposal_slot_charge"] != expected_charge:
        raise DojoAITuningStateError("invocation proposal charge drifted")
    for submission in row["submissions"]:
        _verify_submission(submission)
    return _clone(row)


def _verify_submission(value: Any) -> dict[str, Any]:
    row = _exact_mapping(value, _SUBMISSION_KEYS, "proposal record")
    _identifier(row["submission_id"], "submission_id")
    _sha(row["raw_proposal_sha256"], "raw_proposal_sha256")
    status = row["status"]
    if status not in {"ACCEPTED", "INVALID", "DUPLICATE", "BUDGET_REJECTED"}:
        raise DojoAITuningStateError("proposal status is unsupported")
    errors = _text_list(row["validation_errors"], "validation_errors")
    if status == "ACCEPTED" and errors:
        raise DojoAITuningStateError("accepted proposal carries errors")
    if status != "ACCEPTED" and not errors:
        raise DojoAITuningStateError("rejected proposal lacks an error")
    if status in {"ACCEPTED", "DUPLICATE"}:
        if row["sealed_proposal"] is None:
            raise DojoAITuningStateError("catalog-valid proposal lacks its seal")
        try:
            sealed = seal_candidate_proposal(row["sealed_proposal"])
        except DojoBotTrainerError as exc:
            raise DojoAITuningStateError("stored proposal seal is invalid") from exc
        if (
            row["candidate_id"] != sealed["candidate_id"]
            or row["proposal_sha256"] != sealed["proposal_sha256"]
            or row["executable_identity_sha256"] != sealed["config_sha256"]
        ):
            raise DojoAITuningStateError("proposal identity binding drifted")
    elif any(
        row[key] is not None
        for key in (
            "proposal_sha256",
            "executable_identity_sha256",
            "sealed_proposal",
        )
    ):
        raise DojoAITuningStateError("invalid proposal retained executable material")
    return _clone(row)


def _verify_dispatch(value: Any) -> dict[str, Any]:
    row = _exact_mapping(value, _DISPATCH_KEYS, "run dispatch")
    _identifier(row["dispatch_id"], "dispatch_id")
    if row["status"] not in {"RESERVED", "DISPATCHED"}:
        raise DojoAITuningStateError("run dispatch status is unsupported")
    _utc(row["reserved_at_utc"], "reserved_at_utc")
    if row["status"] == "DISPATCHED":
        _utc(row["dispatched_at_utc"], "dispatched_at_utc")
    elif row["dispatched_at_utc"] is not None:
        raise DojoAITuningStateError("reserved dispatch has a start timestamp")
    _sha(row["study_sha256"], "study_sha256")
    _sha(row["study_event_sha256"], "study_event_sha256")
    _integer(row["study_event_sequence"], "study_event_sequence")
    _identifier_list(row["candidate_ids"], "candidate_ids")
    _sha_list(row["proposal_sha256s"], "proposal_sha256s")
    _sha_list(row["executable_identity_sha256s"], "executable identities")
    _verify_result_binding(row["prior_result_binding"])
    _sha(row["envelope_sha256"], "envelope_sha256")
    return _clone(row)


def _verify_terminal(value: Any) -> dict[str, Any]:
    row = _exact_mapping(value, _TERMINAL_KEYS, "attempt terminal")
    if row["status"] not in {
        "LINEAGE_RESULT_BOUND",
        "INCOMPLETE_REVIEW_REQUIRED",
        "ABANDONED_AFTER_REVIEW",
    }:
        raise DojoAITuningStateError("attempt terminal status is unsupported")
    _utc(row["recorded_at_utc"], "recorded_at_utc")
    if row["status"] == "LINEAGE_RESULT_BOUND":
        _verify_result_binding(row["result_binding"])
        if any(
            row[key] is not None
            for key in ("reason_code", "review_id", "review_rationale")
        ):
            raise DojoAITuningStateError("bound result carries review fields")
    else:
        if row["result_binding"] is not None:
            raise DojoAITuningStateError("incomplete terminal claims a result")
        _identifier(row["reason_code"], "reason_code")
        if row["status"] == "ABANDONED_AFTER_REVIEW":
            _identifier(row["review_id"], "review_id")
            _text(row["review_rationale"], "review_rationale")
    return _clone(row)


def _verify_top_phase(state: Mapping[str, Any]) -> None:
    allowed = {
        "READY_FOR_MODEL",
        "MODEL_INVOCATION_RESERVED",
        "COLLECTING_PROPOSALS",
        "RUN_DISPATCH_RESERVED",
        "AWAITING_LINEAGE_RESULT",
        "REVIEW_REQUIRED",
        "EXHAUSTED",
        "TERMINATED",
    }
    if state["phase"] not in allowed:
        raise DojoAITuningStateError("top-level tuning phase is unsupported")
    attempt = _current_attempt(state)
    if state["phase"] == "READY_FOR_MODEL":
        if attempt is not None:
            raise DojoAITuningStateError("ready state retains an active attempt")
    elif state["phase"] == "EXHAUSTED":
        if attempt is not None:
            raise DojoAITuningStateError("exhausted state retains an active attempt")
        if (
            _attempts_used(state) < MAX_ATTEMPTS
            and _proposal_slots_used(state) < MAX_PROPOSAL_SLOTS
        ):
            raise DojoAITuningStateError("state claims exhaustion before a budget cap")
    elif attempt is None:
        raise DojoAITuningStateError("active phase lacks an active attempt")
    elif state["phase"] != attempt["phase"] and not (
        state["phase"] == "TERMINATED" and attempt["phase"] == "ABANDONED_AFTER_REVIEW"
    ):
        raise DojoAITuningStateError("top and attempt phases diverged")
    if (
        state["phase"] not in {"REVIEW_REQUIRED", "TERMINATED"}
        and state["terminal_reason"] is not None
    ):
        raise DojoAITuningStateError("non-review state carries a terminal reason")
    if state["phase"] == "REVIEW_REQUIRED":
        assert attempt is not None
        terminal = attempt["terminal"]
        if terminal is not None:
            if state["terminal_reason"] != terminal["reason_code"]:
                raise DojoAITuningStateError(
                    "review reason diverges from the incomplete terminal"
                )
        elif state["terminal_reason"] == "MODEL_RESPONSE_EXCEEDED_PROPOSAL_BUDGET":
            if _proposal_slots_used(state) <= MAX_PROPOSAL_SLOTS:
                raise DojoAITuningStateError(
                    "proposal-budget review lacks an actual budget breach"
                )
        elif state["terminal_reason"] == "PROPOSAL_BUDGET_EXHAUSTED_WITHOUT_CANDIDATE":
            if _proposal_slots_used(state) < MAX_PROPOSAL_SLOTS or _accepted_rows(
                attempt
            ):
                raise DojoAITuningStateError(
                    "exhaustion review does not match proposal outcomes"
                )
        else:
            raise DojoAITuningStateError("review state has an unsupported reason")
    elif state["phase"] == "TERMINATED":
        assert attempt is not None and attempt["terminal"] is not None
        if state["terminal_reason"] != attempt["terminal"]["reason_code"]:
            raise DojoAITuningStateError(
                "termination reason diverges from reviewed attempt"
            )


def _verify_transition_timestamps(state: Mapping[str, Any]) -> None:
    """Require every recorded phase timestamp to be monotonic."""

    cursor = _parse_utc_value(state["initialized_at_utc"])
    for attempt in state["attempts"]:
        for invocation in attempt["invocations"]:
            reserved = _parse_utc_value(invocation["reserved_at_utc"])
            if reserved < cursor:
                raise DojoAITuningStateError(
                    "model reservation timestamp moved backward"
                )
            cursor = reserved
            if invocation["response_recorded_at_utc"] is not None:
                response = _parse_utc_value(invocation["response_recorded_at_utc"])
                if response < cursor:
                    raise DojoAITuningStateError(
                        "model response timestamp moved backward"
                    )
                cursor = response
        dispatch = attempt["dispatch"]
        if dispatch is not None:
            reserved = _parse_utc_value(dispatch["reserved_at_utc"])
            if reserved < cursor:
                raise DojoAITuningStateError("run reservation timestamp moved backward")
            cursor = reserved
            if dispatch["dispatched_at_utc"] is not None:
                dispatched = _parse_utc_value(dispatch["dispatched_at_utc"])
                if dispatched < cursor:
                    raise DojoAITuningStateError(
                        "run dispatch timestamp moved backward"
                    )
                cursor = dispatched
        terminal = attempt["terminal"]
        if terminal is not None:
            recorded = _parse_utc_value(terminal["recorded_at_utc"])
            if recorded < cursor:
                raise DojoAITuningStateError(
                    "attempt terminal timestamp moved backward"
                )
            cursor = recorded
    if cursor != _parse_utc_value(state["last_transition_at_utc"]):
        raise DojoAITuningStateError(
            "top-level transition timestamp does not match the latest phase"
        )


def _current_attempt(state: Mapping[str, Any]) -> dict[str, Any] | None:
    attempts = state["attempts"]
    if not attempts:
        return None
    latest = attempts[-1]
    terminal = latest["terminal"]
    if terminal is not None and terminal["status"] == "LINEAGE_RESULT_BOUND":
        return None
    return latest


def _accepted_rows(attempt: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if attempt is None:
        return []
    return [
        submission
        for invocation in attempt["invocations"]
        for submission in invocation["submissions"]
        if submission["status"] == "ACCEPTED"
    ]


def _find_invocation(
    state: Mapping[str, Any], invocation_id: str
) -> dict[str, Any] | None:
    for attempt in state["attempts"]:
        for invocation in attempt["invocations"]:
            if invocation["invocation_id"] == invocation_id:
                return invocation
    return None


def _attempts_used(state: Mapping[str, Any]) -> int:
    return int(state["initial_attempts_consumed"]) + len(state["attempts"])


def _proposal_slots_used(state: Mapping[str, Any]) -> int:
    return int(state["initial_proposal_slots_consumed"]) + sum(
        invocation["proposal_slot_charge"]
        for attempt in state["attempts"]
        for invocation in attempt["invocations"]
    )


def _expected_revision(state: Mapping[str, Any]) -> int:
    revision = 0
    for attempt in state["attempts"]:
        for invocation in attempt["invocations"]:
            revision += 1  # durable reservation before the model call
            if invocation["response_sha256"] is not None:
                revision += 1
        dispatch = attempt["dispatch"]
        if dispatch is not None:
            revision += 1  # run reservation
            if dispatch["status"] == "DISPATCHED":
                revision += 1
        terminal = attempt["terminal"]
        if terminal is not None:
            revision += 1
            if terminal["status"] == "ABANDONED_AFTER_REVIEW":
                revision += 1
    return revision


def _seal_state(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: _clone(item) for key, item in value.items() if key != "state_sha256"}
    return {**body, "state_sha256": _canonical_sha(body)}


def _seal_next_state(
    value: Mapping[str, Any],
    *,
    parent: Mapping[str, Any],
    transition_at_utc: str,
) -> dict[str, Any]:
    if _parse_utc_value(transition_at_utc) < _parse_utc_value(
        parent["last_transition_at_utc"]
    ):
        raise DojoAITuningStateError("state transition timestamp moved backward")
    updated = _clone(value)
    updated["revision"] = parent["revision"] + 1
    updated["previous_state_sha256"] = parent["state_sha256"]
    updated["last_transition_at_utc"] = transition_at_utc
    return _seal_state(updated)


def _cas_mode(state: Mapping[str, Any], expected_parent_state_sha256: Any) -> str:
    expected = _sha(expected_parent_state_sha256, "expected_parent_state_sha256")
    if expected == state["state_sha256"]:
        return "CURRENT"
    if expected == state["previous_state_sha256"]:
        return "REPLAY_LAST"
    raise DojoAITuningStateError("stale or forked parent state SHA")


def _require_current_cas(mode: str) -> None:
    if mode != "CURRENT":
        raise DojoAITuningStateError(
            "previous-parent replay is not the same idempotent transition"
        )


def _new_store_event(
    *,
    sequence: int,
    previous_event_sha256: str | None,
    parent_state_sha256: str | None,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "contract": STORE_EVENT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "previous_event_sha256": previous_event_sha256,
        "parent_state_sha256": parent_state_sha256,
        "state": _clone(state),
    }
    return {**body, "event_sha256": _canonical_sha(body)}


def _verify_store_event(value: Any) -> dict[str, Any]:
    row = _exact_mapping(value, _STORE_EVENT_KEYS, "AI tuning state-store event")
    if (
        row["contract"] != STORE_EVENT_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
    ):
        raise DojoAITuningStateError("state-store event contract/version drifted")
    _integer(row["sequence"], "state-store event sequence")
    if row["sequence"] == 0:
        if (
            row["previous_event_sha256"] is not None
            or row["parent_state_sha256"] is not None
        ):
            raise DojoAITuningStateError("state-store genesis claims a parent")
    else:
        _sha(row["previous_event_sha256"], "previous_event_sha256")
        _sha(row["parent_state_sha256"], "parent_state_sha256")
    row["state"] = verify_tuning_state(row["state"])
    claimed = _sha(row["event_sha256"], "state-store event_sha256")
    body = {key: item for key, item in row.items() if key != "event_sha256"}
    if _canonical_sha(body) != claimed:
        raise DojoAITuningStateError("state-store event digest mismatch")
    return row


def _open_state_store_directory(path: Path, *, create: bool) -> int:
    candidate = path.absolute()
    if candidate == candidate.parent:
        raise DojoAITuningStateError(
            "state-store directory cannot be a filesystem root"
        )
    try:
        state = candidate.lstat()
    except FileNotFoundError:
        if not create:
            raise DojoAITuningStateError("AI tuning state-store directory is absent")
        parent_fd = _open_real_directory(candidate.parent, "state-store parent")
        try:
            try:
                os.mkdir(candidate.name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            os.fsync(parent_fd)
        except OSError as exc:
            raise DojoAITuningStateError(
                f"cannot create AI tuning state-store directory: {exc}"
            ) from exc
        finally:
            os.close(parent_fd)
        try:
            state = candidate.lstat()
        except OSError as exc:
            raise DojoAITuningStateError(
                f"cannot stat AI tuning state-store directory: {exc}"
            ) from exc
    except OSError as exc:
        raise DojoAITuningStateError(
            f"cannot stat AI tuning state-store directory: {exc}"
        ) from exc
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise DojoAITuningStateError("state-store directory must be a real directory")
    return _open_real_directory(candidate, "state-store directory")


def _open_real_directory(path: Path, label: str) -> int:
    try:
        expected = path.lstat()
    except OSError as exc:
        raise DojoAITuningStateError(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISDIR(expected.st_mode) or stat.S_ISLNK(expected.st_mode):
        raise DojoAITuningStateError(f"{label} must be a real directory")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        actual = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise DojoAITuningStateError(f"cannot open {label}: {exc}") from exc
    assert descriptor is not None
    if (
        not stat.S_ISDIR(actual.st_mode)
        or actual.st_dev != expected.st_dev
        or actual.st_ino != expected.st_ino
    ):
        os.close(descriptor)
        raise DojoAITuningStateError(f"{label} changed while being opened")
    return descriptor


def _state_store_event_names(directory_fd: int) -> list[str]:
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as exc:
        raise DojoAITuningStateError(
            f"cannot list AI tuning state-store directory: {exc}"
        ) from exc
    if any(_STORE_EVENT_NAME.fullmatch(name) is None for name in names):
        raise DojoAITuningStateError(
            "AI tuning state-store directory contains an unexpected file"
        )
    return names


def _read_store_event(directory_fd: int, name: str) -> dict[str, Any]:
    try:
        state = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise DojoAITuningStateError(
            f"cannot stat state-store event {name}: {exc}"
        ) from exc
    if not stat.S_ISREG(state.st_mode) or state.st_size <= 0:
        raise DojoAITuningStateError(
            "state-store event must be a nonempty regular file"
        )
    if state.st_size > MAX_STATE_STORE_EVENT_BYTES:
        raise DojoAITuningStateError("state-store event byte limit exceeded")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            raw = handle.read(MAX_STATE_STORE_EVENT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise DojoAITuningStateError(
            f"cannot read state-store event {name}: {exc}"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_dev != state.st_dev
        or before.st_ino != state.st_ino
        or len(raw) != before.st_size
    ):
        raise DojoAITuningStateError("state-store event changed while being read")
    value = _strict_store_json(raw)
    if raw != _canonical_bytes(value) + b"\n":
        raise DojoAITuningStateError("state-store event bytes are not canonical JSON")
    return value


def _write_store_event_exclusive(
    directory_fd: int, name: str, event: Mapping[str, Any]
) -> None:
    payload = _canonical_bytes(event) + b"\n"
    if len(payload) > MAX_STATE_STORE_EVENT_BYTES:
        raise DojoAITuningStateError("state-store event byte limit exceeded")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
        handle = os.fdopen(descriptor, "wb", closefd=True)
        descriptor = None
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(directory_fd)
    except FileExistsError as exc:
        raise DojoAITuningStateError(
            "state-store event slot already exists; reload from the current tip"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _strict_store_json(raw: bytes) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise DojoAITuningStateError(
            f"non-finite state-store JSON is forbidden: {token}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DojoAITuningStateError(
                    f"duplicate state-store JSON key is forbidden: {key}"
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoAITuningStateError("state-store event is not strict JSON") from exc
    if not isinstance(value, dict):
        raise DojoAITuningStateError("state-store event must be a JSON object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoAITuningStateError("value is not canonical JSON") from exc


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _clone(value: Any) -> Any:
    return copy.deepcopy(value)


def _json_value(value: Any, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise DojoAITuningStateError(f"{label} is not strict JSON") from exc


def _exact_mapping(
    value: Any, keys: set[str] | frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoAITuningStateError(f"{label} must be a JSON object")
    if set(value) != set(keys):
        raise DojoAITuningStateError(
            f"{label} schema mismatch; missing={sorted(set(keys) - set(value))}, "
            f"unknown={sorted(set(value) - set(keys))}"
        )
    return _clone(dict(value))


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise DojoAITuningStateError(f"{label} must be a bounded identifier")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DojoAITuningStateError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DojoAITuningStateError(f"{label} must be a non-negative integer")
    return value


def _positive_integer(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result <= 0:
        raise DojoAITuningStateError(f"{label} must be positive")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2_000:
        raise DojoAITuningStateError(f"{label} must be bounded non-empty text")
    return value


def _text_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise DojoAITuningStateError(f"{label} must be a JSON array")
    result = [_text(item, f"{label} item") for item in value]
    if len(set(result)) != len(result):
        raise DojoAITuningStateError(f"{label} contains duplicates")
    return result


def _sha_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise DojoAITuningStateError(f"{label} must be a JSON array")
    result = [_sha(item, f"{label} item") for item in value]
    if len(set(result)) != len(result):
        raise DojoAITuningStateError(f"{label} contains duplicates")
    return result


def _identifier_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise DojoAITuningStateError(f"{label} must be a JSON array")
    result = [_identifier(item, f"{label} item") for item in value]
    if len(set(result)) != len(result):
        raise DojoAITuningStateError(f"{label} contains duplicates")
    return result


def _utc(value: datetime | str, label: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DojoAITuningStateError(f"{label} must be ISO-8601 UTC") from exc
    else:
        raise DojoAITuningStateError(f"{label} must be ISO-8601 UTC")
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DojoAITuningStateError(f"{label} must be UTC")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc_value(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


__all__ = [
    "DojoAITuningStateError",
    "ENVELOPE_CONTRACT",
    "MAX_ATTEMPTS",
    "MAX_PROPOSAL_SLOTS",
    "MAX_STATE_STORE_EVENT_BYTES",
    "MAX_STATE_STORE_EVENTS",
    "SCHEMA_VERSION",
    "STATE_CONTRACT",
    "STATUS_CONTRACT",
    "STORE_EVENT_CONTRACT",
    "abandon_incomplete_lineage",
    "append_state_transition",
    "bind_terminal_evaluation",
    "fixed_envelope_from_sealed_study",
    "initialize_state_store",
    "initialize_tuning_state",
    "mark_incomplete_run",
    "mark_run_dispatched",
    "normalize_model_response_submissions",
    "record_model_response",
    "reserve_model_invocation",
    "reserve_run_dispatch",
    "status_artifact",
    "verify_state_store",
    "verify_tuning_state",
]
