from __future__ import annotations

import copy
from pathlib import Path

import pytest

import quant_rabbit.dojo_immutable_generation_loop as loop
from quant_rabbit.dojo_candidate_lineage_registry import CandidateLineageSnapshot
from quant_rabbit.dojo_conveyor_resource_gate import (
    DISK_RESERVE_BYTES,
    REQUEST_CONTRACT,
)

PROMPT_BYTES = b"fixed DOJO trainer prompt bytes\n"
NOW = "2026-07-20T00:00:00Z"


def _result_body(*, attempt: int = 1, study_sha: str = "1" * 64) -> dict:
    return {
        "attempt_ordinal": attempt,
        "study_sha256": study_sha,
        "evaluation_sha256": "2" * 64,
        "evaluation_artifact_relpath": f"artifacts/evaluation-{attempt}.json",
        "evaluation_artifact_sha256": "3" * 64,
        "evaluation_artifact_size_bytes": 1234,
        "verified_trainer_evaluation": True,
    }


def _lineage(
    *,
    pending: bool = False,
    attempt: int = 1,
    study_sha: str = "1" * 64,
    result_bound_for_pending: bool = False,
) -> CandidateLineageSnapshot:
    studies = [
        {
            "attempt_ordinal": 1,
            "study_sha256": "1" * 64,
            "candidate_count": 4,
        }
    ]
    results = [_result_body()]
    events = [
        {
            "event_type": "RESULT_BOUND",
            "event_sha256": "4" * 64,
            "sequence": 2,
        }
    ]
    if pending:
        studies.append(
            {
                "attempt_ordinal": attempt,
                "study_sha256": study_sha,
                "candidate_count": 2,
            }
        )
        if result_bound_for_pending:
            results.append(_result_body(attempt=attempt, study_sha=study_sha))
            events.append(
                {
                    "event_type": "RESULT_BOUND",
                    "event_sha256": "5" * 64,
                    "sequence": 4,
                }
            )
    tip = events[-1]["event_sha256"]
    return CandidateLineageSnapshot(
        registry_id="qr-lineage",
        lineage_prefix="qr-",
        event_count=len(events),
        latest_sequence=events[-1]["sequence"],
        latest_event_sha256=tip,
        latest_event_at_utc="2026-07-20T00:00:00Z",
        studies=tuple(studies),
        results=tuple(results),
        cumulative_unique_config_sha256s=tuple("a" * 64 for _ in studies),
        cumulative_unique_proposal_sha256s=tuple("b" * 64 for _ in studies),
        events=tuple(events),
    )


def _binding(lineage: CandidateLineageSnapshot) -> dict:
    result = lineage.results[-1]
    event = lineage.events[-1]
    return {
        "registry_id": "qr-lineage",
        "lineage_prefix": "qr-",
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


def _state(lineage: CandidateLineageSnapshot, *, phase: str) -> dict:
    return {
        "registry_id": "qr-lineage",
        "lineage_prefix": "qr-",
        "state_sha256": "6" * 64,
        "fixed_envelope_sha256": "7" * 64,
        "fixed_envelope": {
            "window": {"corpus_sha256": "8" * 64},
            "source_bundle_sha256": "9" * 64,
            "scorer_contract": "QR_DOJO_BOT_TRAINER_EVALUATION_V1",
        },
        "initial_attempts_consumed": 1,
        "initial_proposal_slots_consumed": 4,
        "attempts": [],
        "last_terminal_result_binding": _binding(lineage),
        "phase": phase,
        "terminal_reason": None,
        "last_transition_at_utc": NOW,
    }


def _attempt(*, phase: str, dispatched: bool = True) -> dict:
    dispatch = None
    if phase in {"RUN_DISPATCH_RESERVED", "AWAITING_LINEAGE_RESULT"}:
        dispatch = {
            "dispatch_id": "dispatch-2",
            "status": "DISPATCHED" if dispatched else "RESERVED",
            "study_sha256": "c" * 64,
        }
    return {
        "attempt_ordinal": 2,
        "phase": phase,
        "invocations": [
            {
                "proposal_slot_charge": 2,
                "submissions": [],
            }
        ],
        "dispatch": dispatch,
        "terminal": None,
    }


def _idle_health() -> dict:
    return {
        "contract": loop.HEALTH_CONTRACT,
        "schema_version": 1,
        "generation_ordinal": None,
        "dispatch_id": None,
        "process_state": "IDLE",
        "expected_replay_count": 0,
        "completed_replay_count": 0,
        "failure_artifact_present": False,
        "terminal_bundle_present": False,
        "run_dir_bytes": 0,
        "free_bytes": 80 * 1024**3,
        "observed_at_utc": "2026-07-20T00:00:00Z",
        "economics_fields_present": False,
    }


def _active_health(
    *,
    process: str = "RUNNING",
    completed: int = 17,
    terminal: bool = False,
    failure: bool = False,
    free_bytes: int = 80 * 1024**3,
) -> dict:
    row = _idle_health()
    row.update(
        {
            "generation_ordinal": 2,
            "dispatch_id": "dispatch-2",
            "process_state": process,
            "expected_replay_count": 96,
            "completed_replay_count": completed,
            "terminal_bundle_present": terminal,
            "failure_artifact_present": failure,
            "run_dir_bytes": 8 * 1024**3,
            "free_bytes": free_bytes,
        }
    )
    return row


def _install(monkeypatch, state: dict, lineage: CandidateLineageSnapshot) -> None:
    monkeypatch.setattr(
        loop,
        "verify_state_store",
        lambda events: {
            "event_count": 1,
            "latest_event_sha256": "f" * 64,
            "latest_state": copy.deepcopy(state),
            "automation_ready": True,
            "research_train_only": True,
            "holdout_access_allowed": False,
            "forward_access_allowed": False,
            "proof_eligible": False,
            "promotion_eligible": False,
            "live_permission": False,
            "order_authority": "NONE",
            "broker_mutation_allowed": False,
        },
    )
    monkeypatch.setattr(
        loop,
        "verify_registry",
        lambda events, artifact_root: lineage,
    )


def _plan(
    state: dict,
    health: dict,
    *,
    request=None,
    manifest=None,
    prompt_bytes=None,
    decision_at_utc: str = NOW,
) -> dict:
    return loop.plan_immutable_generation(
        tuning_state_events_dir=Path("state-events"),
        lineage_events_dir=Path("events"),
        artifact_root=Path("artifacts"),
        health=health,
        decision_at_utc=decision_at_utc,
        model_invocation_manifest=manifest,
        model_prompt_bytes=prompt_bytes,
        resource_gate_request=request,
    )


def _manifest() -> dict:
    return loop.seal_model_invocation_manifest(
        tuning_state_events_dir=Path("state-events"),
        prompt_bytes=PROMPT_BYTES,
        provider_id="openai-api",
        model_id="gpt-fixed",
        prepared_at_utc=NOW,
    )


def test_health_schema_rejects_partial_economics_and_unknown_fields() -> None:
    health = _active_health()
    health["terminal_net_jpy"] = 123.0
    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="schema"):
        loop.validate_health_record(health)

    health = _active_health()
    health["economics_fields_present"] = True
    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="economics"):
        loop.validate_health_record(health)


def test_terminal_bundle_requires_complete_fixed_denominator() -> None:
    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="complete"):
        loop.validate_health_record(
            _active_health(process="EXITED", completed=95, terminal=True)
        )


def test_running_generation_can_only_emit_health_monitor_action(monkeypatch) -> None:
    lineage = _lineage(pending=True, attempt=2, study_sha="c" * 64)
    state = _state(_lineage(), phase="AWAITING_LINEAGE_RESULT")
    state["attempts"] = [_attempt(phase="AWAITING_LINEAGE_RESULT")]
    _install(monkeypatch, state, lineage)

    decision = _plan(state, _active_health())

    assert decision["action"] == "MONITOR_HEALTH_ONLY"
    assert decision["permissions"]["next_generation_creation_allowed"] is False
    assert decision["permissions"]["runner_invocation_allowed"] is False
    assert decision["invariants"]["partial_economics_inspection_allowed"] is False
    assert decision["invariants"]["candidate_mutation_after_dispatch_allowed"] is False


def test_terminal_unbound_result_cannot_create_next_generation(monkeypatch) -> None:
    lineage = _lineage(pending=True, attempt=2, study_sha="c" * 64)
    state = _state(_lineage(), phase="AWAITING_LINEAGE_RESULT")
    state["attempts"] = [_attempt(phase="AWAITING_LINEAGE_RESULT")]
    _install(monkeypatch, state, lineage)

    decision = _plan(
        state,
        _active_health(process="EXITED", completed=96, terminal=True),
    )

    assert decision["action"] == "VALIDATE_AND_BIND_TERMINAL_BUNDLE"
    assert decision["permissions"]["terminal_bundle_validation_allowed"] is True
    assert decision["permissions"]["next_generation_creation_allowed"] is False


def test_only_terminal_bound_state_may_reserve_next_model(monkeypatch) -> None:
    lineage = _lineage()
    state = _state(lineage, phase="READY_FOR_MODEL")
    _install(monkeypatch, state, lineage)

    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="manifest"):
        _plan(state, _idle_health())

    manifest = _manifest()
    decision = _plan(
        state,
        _idle_health(),
        manifest=manifest,
        prompt_bytes=PROMPT_BYTES,
    )

    assert decision["action"] == "RESERVE_NEXT_MODEL_INVOCATION"
    assert decision["permissions"]["next_generation_creation_allowed"] is True
    assert decision["model_invocation_manifest_sha256"] == manifest["manifest_sha256"]
    assert decision["tuning_state_store_tip_sha256"] == "f" * 64
    assert "LOCAL_STATE_STORE_HAS_NO_EXTERNAL_MONOTONIC_WITNESS" in decision[
        "limitations"
    ]

    pending = _lineage(pending=True, attempt=2, study_sha="c" * 64)
    _install(monkeypatch, state, pending)
    with pytest.raises(loop.DojoImmutableGenerationLoopError):
        _plan(
            state,
            _idle_health(),
            manifest=manifest,
            prompt_bytes=PROMPT_BYTES,
        )


def test_model_manifest_binds_exact_prompt_and_store_tip(monkeypatch) -> None:
    lineage = _lineage()
    state = _state(lineage, phase="READY_FOR_MODEL")
    _install(monkeypatch, state, lineage)
    manifest = _manifest()

    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="drifted"):
        _plan(
            state,
            _idle_health(),
            manifest=manifest,
            prompt_bytes=b"different prompt bytes",
        )

    forged = copy.deepcopy(manifest)
    forged["tuning_state_store_tip_sha256"] = "e" * 64
    body = {key: value for key, value in forged.items() if key != "manifest_sha256"}
    forged["manifest_sha256"] = loop.canonical_sha256(body)
    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="drifted"):
        _plan(
            state,
            _idle_health(),
            manifest=forged,
            prompt_bytes=PROMPT_BYTES,
        )


def test_model_manifest_must_be_preregistered_fresh_and_after_lineage(monkeypatch) -> None:
    lineage = _lineage()
    state = _state(lineage, phase="READY_FOR_MODEL")
    _install(monkeypatch, state, lineage)
    manifest = _manifest()
    health = _idle_health()
    health["observed_at_utc"] = "2026-07-20T00:05:01Z"

    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="stale"):
        _plan(
            state,
            health,
            manifest=manifest,
            prompt_bytes=PROMPT_BYTES,
            decision_at_utc="2026-07-20T00:05:01Z",
        )


def test_append_only_state_store_proof_is_required(monkeypatch) -> None:
    lineage = _lineage()
    monkeypatch.setattr(
        loop,
        "verify_state_store",
        lambda events: (_ for _ in ()).throw(ValueError("bare state rejected")),
    )
    monkeypatch.setattr(
        loop,
        "verify_registry",
        lambda events, artifact_root: lineage,
    )

    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="state"):
        loop.plan_immutable_generation(
            tuning_state_events_dir=Path("state-events"),
            lineage_events_dir=Path("events"),
            artifact_root=Path("artifacts"),
            health=_idle_health(),
            decision_at_utc=NOW,
        )


def test_health_timestamp_must_be_valid_fresh_and_monotonic(monkeypatch) -> None:
    lineage = _lineage(pending=True, attempt=2, study_sha="c" * 64)
    state = _state(_lineage(), phase="AWAITING_LINEAGE_RESULT")
    state["attempts"] = [_attempt(phase="AWAITING_LINEAGE_RESULT")]
    _install(monkeypatch, state, lineage)

    malformed = _active_health()
    malformed["observed_at_utc"] = "today"
    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="UTC"):
        _plan(state, malformed)

    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="stale"):
        _plan(
            state,
            _active_health(),
            decision_at_utc="2026-07-20T00:05:01Z",
        )

    state["last_transition_at_utc"] = "2026-07-20T00:00:01Z"
    _install(monkeypatch, state, lineage)
    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="predates"):
        _plan(state, _active_health())


def test_incomplete_exit_consumes_attempt_and_forbids_free_retry(monkeypatch) -> None:
    lineage = _lineage(pending=True, attempt=2, study_sha="c" * 64)
    state = _state(_lineage(), phase="AWAITING_LINEAGE_RESULT")
    state["attempts"] = [_attempt(phase="AWAITING_LINEAGE_RESULT")]
    _install(monkeypatch, state, lineage)

    decision = _plan(state, _active_health(process="EXITED", completed=61))

    assert decision["action"] == "MARK_INCOMPLETE_REVIEW_REQUIRED"
    assert decision["reasons"] == ["NO_FREE_RETRY_AFTER_INCOMPLETE_RUN"]
    assert decision["invariants"]["same_attempt_retry_allowed"] is False


def _resource_request(state: dict, *, free_bytes: int) -> dict:
    prior = state["last_terminal_result_binding"]
    return {
        "contract": REQUEST_CONTRACT,
        "schema_version": 1,
        "prior_result": {
            "terminal": True,
            "result_bound": True,
            "result_binding_sha256": loop.canonical_sha256(prior),
            "lineage_tip_sha256": prior["lineage_tip_sha256"],
        },
        "drive_compact_report": {
            "remote_verified": True,
            "report_sha256": "d" * 64,
            "metadata_receipt_sha256": "e" * 64,
        },
        "resources": {
            "active_trainer_count": 0,
            "remote_unverified_generation_count": 0,
            "compression_upload_active_count": 0,
            "free_bytes": free_bytes,
            "predicted_next_output_bytes": 10 * 1024**3,
            "archive_temp_bytes": 0,
        },
        "fixed_bindings": {
            "corpus_sha256": "8" * 64,
            "scorer_sha256": "9" * 64,
            "study_sha256": "c" * 64,
        },
        "authority": {
            "model_invocation_allowed": False,
            "runner_invocation_allowed": False,
            "filesystem_write_allowed": False,
            "filesystem_delete_allowed": False,
            "broker_mutation_allowed": False,
            "live_permission": False,
            "order_authority": "NONE",
        },
    }


def test_low_disk_applies_backpressure_before_large_run(monkeypatch) -> None:
    lineage = _lineage(pending=True, attempt=2, study_sha="c" * 64)
    state = _state(_lineage(), phase="RUN_DISPATCH_RESERVED")
    state["attempts"] = [
        _attempt(phase="RUN_DISPATCH_RESERVED", dispatched=False)
    ]
    _install(monkeypatch, state, lineage)

    decision = _plan(
        state,
        _active_health(
            process="RESERVED", completed=0, free_bytes=DISK_RESERVE_BYTES
        ),
        request=_resource_request(state, free_bytes=DISK_RESERVE_BYTES),
    )

    assert decision["action"] == "WAIT_FOR_RESOURCES"
    assert "INSUFFICIENT_FREE_BYTES" in decision["reasons"]
    assert decision["permissions"]["manual_runner_start_eligible"] is False
    assert decision["permissions"]["automatic_runner_start_allowed"] is False


def test_sufficient_disk_is_manual_eligibility_not_start_authority(monkeypatch) -> None:
    lineage = _lineage(pending=True, attempt=2, study_sha="c" * 64)
    state = _state(_lineage(), phase="RUN_DISPATCH_RESERVED")
    state["attempts"] = [
        _attempt(phase="RUN_DISPATCH_RESERVED", dispatched=False)
    ]
    _install(monkeypatch, state, lineage)

    decision = _plan(
        state,
        _active_health(process="RESERVED", completed=0, free_bytes=100 * 1024**3),
        request=_resource_request(state, free_bytes=100 * 1024**3),
    )

    assert decision["action"] == "MANUAL_START_ELIGIBLE"
    assert decision["permissions"]["manual_runner_start_eligible"] is True
    assert decision["permissions"]["runner_invocation_allowed"] is False
    assert decision["permissions"]["automatic_runner_start_allowed"] is False


def test_resource_request_cannot_change_fixed_scorer_closure(monkeypatch) -> None:
    lineage = _lineage(pending=True, attempt=2, study_sha="c" * 64)
    state = _state(_lineage(), phase="RUN_DISPATCH_RESERVED")
    state["attempts"] = [
        _attempt(phase="RUN_DISPATCH_RESERVED", dispatched=False)
    ]
    _install(monkeypatch, state, lineage)
    request = _resource_request(state, free_bytes=100 * 1024**3)
    request["fixed_bindings"]["scorer_sha256"] = "f" * 64

    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="scorer"):
        _plan(
            state,
            _active_health(process="RESERVED", completed=0),
            request=request,
        )


def test_reserved_dispatch_cannot_be_started_twice(monkeypatch) -> None:
    lineage = _lineage(pending=True, attempt=2, study_sha="c" * 64)
    state = _state(_lineage(), phase="RUN_DISPATCH_RESERVED")
    state["attempts"] = [
        _attempt(phase="RUN_DISPATCH_RESERVED", dispatched=False)
    ]
    _install(monkeypatch, state, lineage)

    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="start"):
        _plan(
            state,
            _active_health(process="RUNNING", free_bytes=100 * 1024**3),
            request=_resource_request(state, free_bytes=100 * 1024**3),
        )


def test_resource_free_bytes_must_match_health_observation(monkeypatch) -> None:
    lineage = _lineage(pending=True, attempt=2, study_sha="c" * 64)
    state = _state(_lineage(), phase="RUN_DISPATCH_RESERVED")
    state["attempts"] = [
        _attempt(phase="RUN_DISPATCH_RESERVED", dispatched=False)
    ]
    _install(monkeypatch, state, lineage)

    with pytest.raises(loop.DojoImmutableGenerationLoopError, match="free bytes"):
        _plan(
            state,
            _active_health(process="RESERVED", completed=0, free_bytes=1024),
            request=_resource_request(state, free_bytes=100 * 1024**3),
        )
