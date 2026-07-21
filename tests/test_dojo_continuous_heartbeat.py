from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from quant_rabbit.dojo_continuous_heartbeat import (
    GENESIS_SHA256,
    OBSERVATION_CONTRACT,
    DojoContinuousHeartbeatError,
    apply_observation,
    build_local_observation,
    build_event,
    canonical_sha256,
    complete_reserved_work,
    initial_state,
    plan_heartbeat,
    reserve_decision,
    seal_local_run_status,
    seal_observation,
    verify_local_probe_manifest,
    verify_event,
    verify_policy,
    verify_state,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config" / "dojo_continuous_heartbeat_policy_v1.json"
PROBE_PATH = ROOT / "config" / "dojo_continuous_heartbeat_local_probe_v1.json"
STATUS_PATH = ROOT / "config" / "dojo_continuous_heartbeat_idle_run_status_v1.json"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _policy() -> dict:
    return verify_policy(json.loads(POLICY_PATH.read_text()))


def _authority() -> dict:
    return copy.deepcopy(_policy()["authority"])


def _raw_observation(
    *,
    process_state: str = "IDLE",
    terminal_sha: str | None = None,
    failure_sha: str | None = None,
    free_bytes: int = 100 * 1024**3,
) -> dict:
    active = process_state != "IDLE"
    completed = 10 if terminal_sha is not None else (3 if active else 0)
    expected = 10 if active else 0
    return {
        "contract": OBSERVATION_CONTRACT,
        "schema_version": 1,
        "observed_at_utc": "2026-07-22T00:01:00Z",
        "run": {
            "process_state": process_state,
            "run_id": "run-1" if active else None,
            "process_identity_sha256": SHA_A if active else None,
            "expected_coordinate_count": expected,
            "completed_coordinate_count": completed,
            "checkpoint_sha256": SHA_B if completed else None,
            "terminal_bundle_sha256": terminal_sha,
            "failure_artifact_sha256": failure_sha,
            "economics_fields_present": False,
        },
        "resources": {
            "active_trainer_count": 0,
            "remote_unverified_generation_count": 0,
            "compression_upload_active_count": 0,
            "free_bytes": free_bytes,
            "predicted_next_output_bytes": 1024,
            "archive_temp_bytes": 0,
        },
        "authority": _authority(),
    }


def _observed_state(raw: dict) -> dict:
    policy = _policy()
    state = initial_state(policy=policy, initialized_at_utc="2026-07-22T00:00:00Z")
    observation = seal_observation(raw, policy=policy)
    state, changed = apply_observation(
        state,
        observation,
        policy=policy,
        event_at_utc="2026-07-22T00:01:00Z",
    )
    assert changed
    return state


def _reserve_and_complete(
    state: dict,
    *,
    result_sha: str,
    outcome: str = "SUCCESS",
    minute: int,
) -> dict:
    policy = _policy()
    decision = plan_heartbeat(state, policy=policy)
    state, changed = reserve_decision(
        state,
        decision,
        policy=policy,
        reserved_at_utc=f"2026-07-22T00:{minute:02d}:00Z",
    )
    assert changed
    return complete_reserved_work(
        state,
        policy=policy,
        operation_id=decision["operation_id"],
        result_sha256=result_sha,
        outcome=outcome,
        completed_at_utc=f"2026-07-22T00:{minute + 1:02d}:00Z",
    )


def test_policy_and_local_probe_are_content_addressed_and_no_authority() -> None:
    policy = _policy()
    probe = verify_local_probe_manifest(
        json.loads(PROBE_PATH.read_text()), policy=policy
    )
    assert probe["probe_sha256"] == policy["local_probe_manifest_sha256"]
    assert policy["authority"]["broker_mutation_allowed"] is False
    assert policy["authority"]["live_permission"] is False
    assert policy["authority"]["order_authority"] == "NONE"


def test_resource_boundary_active_health_and_semantic_noop() -> None:
    policy = _policy()
    required = 20 * 1024**3 + 1280
    blocked = _raw_observation(free_bytes=required - 1)
    blocked_state = _observed_state(blocked)
    assert plan_heartbeat(blocked_state, policy=policy)["action"] == (
        "RESOURCE_BACKPRESSURE"
    )

    running = _raw_observation(process_state="RUNNING")
    state = _observed_state(running)
    assert plan_heartbeat(state, policy=policy)["action"] == "MONITOR_HEALTH_ONLY"
    changed_raw = copy.deepcopy(running)
    changed_raw["observed_at_utc"] = "2026-07-22T00:02:00Z"
    changed_raw["resources"]["free_bytes"] -= 1
    observation = seal_observation(changed_raw, policy=policy)
    same_state, changed = apply_observation(
        state,
        observation,
        policy=policy,
        event_at_utc="2026-07-22T00:02:00Z",
    )
    assert not changed
    assert same_state == state

    leaked = copy.deepcopy(running)
    leaked["run"]["pnl_jpy"] = 123
    with pytest.raises(DojoContinuousHeartbeatError):
        seal_observation(leaked, policy=policy)
    running["run"]["economics_fields_present"] = True
    with pytest.raises(DojoContinuousHeartbeatError):
        seal_observation(running, policy=policy)


def test_terminal_validation_and_review_are_each_recorded_exactly_once() -> None:
    policy = _policy()
    state = _observed_state(
        _raw_observation(process_state="EXITED", terminal_sha=SHA_C)
    )
    first = plan_heartbeat(state, policy=policy)
    assert first["action"] == "VALIDATE_TERMINAL_BUNDLE"
    state = _reserve_and_complete(state, result_sha=SHA_A, minute=2)
    second = plan_heartbeat(state, policy=policy)
    assert second["action"] == "REVIEW_TERMINAL_RESULT"
    state = _reserve_and_complete(state, result_sha=SHA_B, minute=4)
    assert plan_heartbeat(state, policy=policy)["action"] == "TERMINAL_REVIEWED"
    assert len(state["completed_obligations"]) == 2


def test_failed_validation_is_not_retried_or_promoted_to_terminal_review() -> None:
    policy = _policy()
    state = _observed_state(
        _raw_observation(process_state="EXITED", terminal_sha=SHA_C)
    )
    state = _reserve_and_complete(
        state, result_sha=SHA_A, outcome="FAILED", minute=2
    )
    review = plan_heartbeat(state, policy=policy)
    assert review["action"] == "REVIEW_REQUIRED"
    assert review["phase"] == "TERMINAL_VALIDATION_FAILED"
    state = _reserve_and_complete(state, result_sha=SHA_B, minute=4)
    decision = plan_heartbeat(state, policy=policy)
    assert decision["action"] == "REVIEW_RECORDED"
    assert len(state["completed_obligations"]) == 2


def test_single_lease_and_idempotent_completion() -> None:
    policy = _policy()
    state = _observed_state(
        _raw_observation(process_state="MISSING", failure_sha=SHA_C)
    )
    decision = plan_heartbeat(state, policy=policy)
    reserved, changed = reserve_decision(
        state,
        decision,
        policy=policy,
        reserved_at_utc="2026-07-22T00:02:00Z",
    )
    assert changed
    occupied = plan_heartbeat(reserved, policy=policy)
    assert occupied["action"] == "WAIT_FOR_RESERVED_WORK"
    assert occupied["operation_id"] == decision["operation_id"]
    completed = complete_reserved_work(
        reserved,
        policy=policy,
        operation_id=decision["operation_id"],
        result_sha256=SHA_A,
        outcome="SUCCESS",
        completed_at_utc="2026-07-22T00:03:00Z",
    )
    duplicate = complete_reserved_work(
        completed,
        policy=policy,
        operation_id=decision["operation_id"],
        result_sha256=SHA_A,
        outcome="SUCCESS",
        completed_at_utc="2026-07-22T00:04:00Z",
    )
    assert duplicate == completed
    with pytest.raises(DojoContinuousHeartbeatError):
        complete_reserved_work(
            completed,
            policy=policy,
            operation_id=decision["operation_id"],
            result_sha256=SHA_A,
            outcome="FAILED",
            completed_at_utc="2026-07-22T00:04:00Z",
        )


def test_local_observer_builder_binds_status_and_rejects_active_staleness() -> None:
    policy = _policy()
    probe = json.loads(PROBE_PATH.read_text())
    status = json.loads(STATUS_PATH.read_text())
    observation = build_local_observation(
        run_status=status,
        local_probe=probe,
        policy=policy,
        observed_at_utc="2026-07-22T01:00:00Z",
        active_trainer_count=0,
        remote_unverified_generation_count=0,
        compression_upload_active_count=0,
        free_bytes=100 * 1024**3,
    )
    assert observation["run"]["process_state"] == "IDLE"
    assert observation["authority"] == policy["authority"]

    active_raw = copy.deepcopy(status)
    active_raw.pop("status_sha256")
    active_raw["run"] = _raw_observation(process_state="RUNNING")["run"]
    active_raw["updated_at_utc"] = "2026-07-22T00:00:00Z"
    active = seal_local_run_status(active_raw)
    with pytest.raises(DojoContinuousHeartbeatError, match="stale"):
        build_local_observation(
            run_status=active,
            local_probe=probe,
            policy=policy,
            observed_at_utc="2026-07-22T00:03:00Z",
            active_trainer_count=0,
            remote_unverified_generation_count=0,
            compression_upload_active_count=0,
            free_bytes=100 * 1024**3,
        )


def test_state_tampering_is_rejected() -> None:
    policy = _policy()
    state = _observed_state(_raw_observation())
    tampered = copy.deepcopy(state)
    tampered["authority"]["live_permission"] = True
    tampered["state_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "state_sha256"}
    )
    with pytest.raises(DojoContinuousHeartbeatError):
        verify_state(tampered, policy=policy)


def test_event_type_cannot_mislabel_an_observation_transition() -> None:
    policy = _policy()
    initial = initial_state(
        policy=policy, initialized_at_utc="2026-07-22T00:00:00Z"
    )
    genesis = build_event(
        sequence=0,
        previous_event_sha256=GENESIS_SHA256,
        event_type="INITIALIZED",
        prior_state=None,
        state=initial,
        policy=policy,
        event_at_utc="2026-07-22T00:00:00Z",
    )
    observed, changed = apply_observation(
        initial,
        seal_observation(_raw_observation(), policy=policy),
        policy=policy,
        event_at_utc="2026-07-22T00:01:00Z",
    )
    assert changed
    event = build_event(
        sequence=1,
        previous_event_sha256=genesis["event_sha256"],
        event_type="OBSERVATION_CHANGED",
        prior_state=initial,
        state=observed,
        policy=policy,
        event_at_utc="2026-07-22T00:01:00Z",
    )
    event["event_type"] = "WORK_RESERVED"
    event["event_sha256"] = canonical_sha256(
        {key: value for key, value in event.items() if key != "event_sha256"}
    )
    with pytest.raises(DojoContinuousHeartbeatError):
        verify_event(
            event,
            policy=policy,
            expected_sequence=1,
            previous_event_sha256=genesis["event_sha256"],
            prior_state=initial,
        )
