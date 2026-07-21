"""Deterministic, research-only control plane for continuous DOJO health.

The heartbeat records only compact process/progress/checkpoint/resource facts.
It deliberately excludes running P/L and owns no model, Drive, replay runner,
process launcher, broker, or live-order capability.  Expensive work is exposed
as a content-addressed obligation which must be reserved once before an
external worker may act.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Final


POLICY_CONTRACT: Final = "QR_DOJO_CONTINUOUS_HEARTBEAT_POLICY_V1"
OBSERVATION_CONTRACT: Final = "QR_DOJO_CONTINUOUS_HEARTBEAT_OBSERVATION_V1"
STATE_CONTRACT: Final = "QR_DOJO_CONTINUOUS_HEARTBEAT_STATE_V1"
EVENT_CONTRACT: Final = "QR_DOJO_CONTINUOUS_HEARTBEAT_EVENT_V1"
DECISION_CONTRACT: Final = "QR_DOJO_CONTINUOUS_HEARTBEAT_DECISION_V1"
LOCAL_PROBE_CONTRACT: Final = "QR_DOJO_CONTINUOUS_HEARTBEAT_LOCAL_PROBE_V1"
LOCAL_RUN_STATUS_CONTRACT: Final = (
    "QR_DOJO_CONTINUOUS_HEARTBEAT_LOCAL_RUN_STATUS_V1"
)
SCHEMA_VERSION: Final = 1
GENESIS_SHA256: Final = "0" * 64

_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}\Z")
_PROCESS_STATES: Final = frozenset(
    {"IDLE", "RESERVED", "RUNNING", "EXITED", "MISSING"}
)
_ACTIONS: Final = frozenset(
    {
        "NO_OBSERVATION",
        "MONITOR_HEALTH_ONLY",
        "VALIDATE_TERMINAL_BUNDLE",
        "REVIEW_TERMINAL_RESULT",
        "REVIEW_REQUIRED",
        "WAIT_FOR_RESERVED_WORK",
        "RESOURCE_BACKPRESSURE",
        "MANUAL_RUN_START_ELIGIBLE",
        "TERMINAL_REVIEWED",
        "REVIEW_RECORDED",
    }
)
_NO_AUTHORITY: Final = {
    "model_invocation_allowed": False,
    "drive_access_allowed": False,
    "process_start_allowed": False,
    "runner_invocation_allowed": False,
    "filesystem_delete_allowed": False,
    "broker_mutation_allowed": False,
    "live_permission": False,
    "order_authority": "NONE",
}

_POLICY_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "policy_id",
        "health_cadence_seconds",
        "max_observation_age_seconds",
        "local_probe_manifest_sha256",
        "resource_gate",
        "max_completed_obligations",
        "max_event_count",
        "classification",
        "authority",
        "policy_sha256",
    }
)
_LOCAL_PROBE_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "probe_id",
        "state_root",
        "run_status_path",
        "active_trainer_marker_directory",
        "remote_unverified_generation_marker_directory",
        "compression_upload_marker_directory",
        "storage_path",
        "predicted_next_output_bytes",
        "archive_temp_bytes",
        "authority",
        "probe_sha256",
    }
)
_LOCAL_RUN_STATUS_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "updated_at_utc",
        "run",
        "authority",
        "status_sha256",
    }
)
_RESOURCE_POLICY_KEYS = frozenset(
    {
        "disk_reserve_bytes",
        "predicted_output_numerator",
        "predicted_output_denominator",
        "require_active_trainer_zero",
        "require_remote_unverified_generation_zero",
        "require_compression_upload_idle",
    }
)
_OBSERVATION_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "observed_at_utc",
        "run",
        "resources",
        "authority",
        "semantic_sha256",
        "observation_sha256",
    }
)
_RUN_KEYS = frozenset(
    {
        "process_state",
        "run_id",
        "process_identity_sha256",
        "expected_coordinate_count",
        "completed_coordinate_count",
        "checkpoint_sha256",
        "terminal_bundle_sha256",
        "failure_artifact_sha256",
        "economics_fields_present",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "active_trainer_count",
        "remote_unverified_generation_count",
        "compression_upload_active_count",
        "free_bytes",
        "predicted_next_output_bytes",
        "archive_temp_bytes",
    }
)
_STATE_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "policy_sha256",
        "revision",
        "previous_state_sha256",
        "latest_observation",
        "latest_semantic_sha256",
        "active_lease",
        "completed_obligations",
        "last_transition_at_utc",
        "authority",
        "state_sha256",
    }
)
_LEASE_KEYS = frozenset(
    {
        "operation_id",
        "action",
        "obligation_id",
        "input_semantic_sha256",
        "reserved_at_utc",
    }
)
_COMPLETION_KEYS = frozenset(
    {
        "operation_id",
        "action",
        "obligation_id",
        "input_semantic_sha256",
        "result_sha256",
        "outcome",
        "completed_at_utc",
    }
)
_EVENT_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "sequence",
        "previous_event_sha256",
        "event_type",
        "operation_id",
        "input_sha256",
        "event_at_utc",
        "state",
        "event_sha256",
    }
)
_DECISION_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "policy_sha256",
        "state_sha256",
        "observation_semantic_sha256",
        "phase",
        "action",
        "obligation_id",
        "operation_id",
        "reasons",
        "resource_gate",
        "permissions",
        "decision_sha256",
    }
)


class DojoContinuousHeartbeatError(ValueError):
    """The heartbeat policy, state, event, or observation is unsafe."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict canonical JSON bytes for a finite JSON value."""

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
        raise DojoContinuousHeartbeatError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def verify_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(_mapping(value, "policy"))
    _exact(row, _POLICY_KEYS, "policy")
    body = {key: item for key, item in row.items() if key != "policy_sha256"}
    resources = dict(_mapping(row["resource_gate"], "policy.resource_gate"))
    _exact(resources, _RESOURCE_POLICY_KEYS, "policy.resource_gate")
    if (
        row["contract"] != POLICY_CONTRACT
        or _integer(row["schema_version"], "policy.schema_version")
        != SCHEMA_VERSION
        or canonical_sha256(body)
        != _sha(row["policy_sha256"], "policy.policy_sha256")
        or row["classification"] != "RESEARCH_HEALTH_CONTROL_ONLY"
        or row["authority"] != _NO_AUTHORITY
    ):
        raise DojoContinuousHeartbeatError(
            "policy contract, digest, classification, or authority is invalid"
        )
    _identifier(row["policy_id"], "policy.policy_id")
    _sha(
        row["local_probe_manifest_sha256"],
        "policy.local_probe_manifest_sha256",
    )
    cadence = _integer(
        row["health_cadence_seconds"], "policy.health_cadence_seconds", minimum=1
    )
    max_age = _integer(
        row["max_observation_age_seconds"],
        "policy.max_observation_age_seconds",
        minimum=1,
    )
    if max_age < cadence:
        raise DojoContinuousHeartbeatError(
            "observation age must cover at least one health cadence"
        )
    for key in (
        "disk_reserve_bytes",
        "predicted_output_numerator",
        "predicted_output_denominator",
    ):
        _integer(resources[key], f"policy.resource_gate.{key}", minimum=1)
    for key in (
        "require_active_trainer_zero",
        "require_remote_unverified_generation_zero",
        "require_compression_upload_idle",
    ):
        if resources[key] is not True:
            raise DojoContinuousHeartbeatError(
                f"policy.resource_gate.{key} must fail closed"
            )
    _integer(
        row["max_completed_obligations"],
        "policy.max_completed_obligations",
        minimum=1,
    )
    _integer(row["max_event_count"], "policy.max_event_count", minimum=1)
    return row


def verify_local_probe_manifest(
    value: Mapping[str, Any], *, policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the one sealed set of read-only local observation targets."""

    verified_policy = verify_policy(policy)
    row = dict(_mapping(value, "local_probe"))
    _exact(row, _LOCAL_PROBE_KEYS, "local_probe")
    body = {key: item for key, item in row.items() if key != "probe_sha256"}
    if (
        row["contract"] != LOCAL_PROBE_CONTRACT
        or _integer(row["schema_version"], "local_probe.schema_version")
        != SCHEMA_VERSION
        or row["authority"] != _NO_AUTHORITY
        or canonical_sha256(body)
        != _sha(row["probe_sha256"], "local_probe.probe_sha256")
        or row["probe_sha256"]
        != verified_policy["local_probe_manifest_sha256"]
    ):
        raise DojoContinuousHeartbeatError(
            "local probe contract, authority, digest, or policy binding is invalid"
        )
    _identifier(row["probe_id"], "local_probe.probe_id")
    paths = {
        key: _absolute_path(row[key], f"local_probe.{key}")
        for key in (
            "state_root",
            "run_status_path",
            "active_trainer_marker_directory",
            "remote_unverified_generation_marker_directory",
            "compression_upload_marker_directory",
            "storage_path",
        )
    }
    root = paths["state_root"] + "/"
    for key in (
        "run_status_path",
        "active_trainer_marker_directory",
        "remote_unverified_generation_marker_directory",
        "compression_upload_marker_directory",
    ):
        if not paths[key].startswith(root):
            raise DojoContinuousHeartbeatError(
                f"local_probe.{key} must be below state_root"
            )
    _integer(
        row["predicted_next_output_bytes"],
        "local_probe.predicted_next_output_bytes",
        minimum=1,
    )
    _integer(row["archive_temp_bytes"], "local_probe.archive_temp_bytes")
    return row


def seal_local_run_status(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(_mapping(value, "local_run_status"))
    expected = _LOCAL_RUN_STATUS_KEYS - {"status_sha256"}
    _exact(raw, expected, "local_run_status")
    if (
        raw["contract"] != LOCAL_RUN_STATUS_CONTRACT
        or _integer(raw["schema_version"], "local_run_status.schema_version")
        != SCHEMA_VERSION
        or raw["authority"] != _NO_AUTHORITY
    ):
        raise DojoContinuousHeartbeatError(
            "local run status contract or authority boundary is invalid"
        )
    body = {
        "contract": LOCAL_RUN_STATUS_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "updated_at_utc": _utc(
            raw["updated_at_utc"], "local_run_status.updated_at_utc"
        ),
        "run": _validate_run(raw["run"]),
        "authority": dict(_NO_AUTHORITY),
    }
    return {**body, "status_sha256": canonical_sha256(body)}


def verify_local_run_status(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(_mapping(value, "local_run_status"))
    _exact(row, _LOCAL_RUN_STATUS_KEYS, "local_run_status")
    raw = {key: item for key, item in row.items() if key != "status_sha256"}
    rebuilt = seal_local_run_status(raw)
    if row != rebuilt:
        raise DojoContinuousHeartbeatError(
            "local run status content or digest is invalid"
        )
    return rebuilt


def build_local_observation(
    *,
    run_status: Mapping[str, Any],
    local_probe: Mapping[str, Any],
    policy: Mapping[str, Any],
    observed_at_utc: str,
    active_trainer_count: int,
    remote_unverified_generation_count: int,
    compression_upload_active_count: int,
    free_bytes: int,
) -> dict[str, Any]:
    """Build a sealed observation from already collected local probe facts."""

    verified_policy = verify_policy(policy)
    probe = verify_local_probe_manifest(local_probe, policy=verified_policy)
    status = verify_local_run_status(run_status)
    observed = _utc(observed_at_utc, "observed_at_utc")
    if status["run"]["process_state"] in {"RESERVED", "RUNNING"}:
        observed_time = datetime.fromisoformat(observed.replace("Z", "+00:00"))
        updated_time = datetime.fromisoformat(
            status["updated_at_utc"].replace("Z", "+00:00")
        )
        age_seconds = (observed_time - updated_time).total_seconds()
        if age_seconds < 0 or age_seconds > verified_policy[
            "max_observation_age_seconds"
        ]:
            raise DojoContinuousHeartbeatError(
                "active local run status is stale or from the future"
            )
    raw = {
        "contract": OBSERVATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "observed_at_utc": observed,
        "run": status["run"],
        "resources": {
            "active_trainer_count": _integer(
                active_trainer_count, "active_trainer_count"
            ),
            "remote_unverified_generation_count": _integer(
                remote_unverified_generation_count,
                "remote_unverified_generation_count",
            ),
            "compression_upload_active_count": _integer(
                compression_upload_active_count,
                "compression_upload_active_count",
            ),
            "free_bytes": _integer(free_bytes, "free_bytes"),
            "predicted_next_output_bytes": probe["predicted_next_output_bytes"],
            "archive_temp_bytes": probe["archive_temp_bytes"],
        },
        "authority": dict(_NO_AUTHORITY),
    }
    return seal_observation(raw, policy=verified_policy)


def seal_observation(
    value: Mapping[str, Any], *, policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate compact health and seal raw plus semantic digests.

    Exact schemas ensure balance, P/L, win rate, expectancy, or other partial
    economics cannot be smuggled into active-run health.
    """

    verified_policy = verify_policy(policy)
    raw = dict(_mapping(value, "observation"))
    expected_raw_keys = _OBSERVATION_KEYS - {"semantic_sha256", "observation_sha256"}
    _exact(raw, expected_raw_keys, "observation")
    if (
        raw["contract"] != OBSERVATION_CONTRACT
        or _integer(raw["schema_version"], "observation.schema_version")
        != SCHEMA_VERSION
        or raw["authority"] != _NO_AUTHORITY
    ):
        raise DojoContinuousHeartbeatError(
            "observation contract or authority boundary is invalid"
        )
    observed = _utc(raw["observed_at_utc"], "observation.observed_at_utc")
    run = _validate_run(raw["run"])
    resources = _validate_resources(raw["resources"])
    gate = resource_start_gate(resources, policy=verified_policy)
    semantic_body = {
        "run": run,
        # Exact free bytes fluctuate without changing schedulability.  The
        # semantic trigger binds the derived gate, required bytes, and reasons.
        "resource_gate": gate,
        "authority": dict(_NO_AUTHORITY),
    }
    semantic_sha = canonical_sha256(semantic_body)
    body = {
        "contract": OBSERVATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "observed_at_utc": observed,
        "run": run,
        "resources": resources,
        "authority": dict(_NO_AUTHORITY),
        "semantic_sha256": semantic_sha,
    }
    return {**body, "observation_sha256": canonical_sha256(body)}


def verify_observation(
    value: Mapping[str, Any], *, policy: Mapping[str, Any]
) -> dict[str, Any]:
    row = dict(_mapping(value, "observation"))
    _exact(row, _OBSERVATION_KEYS, "observation")
    raw = {key: item for key, item in row.items() if key not in {"semantic_sha256", "observation_sha256"}}
    rebuilt = seal_observation(raw, policy=policy)
    if rebuilt != row:
        raise DojoContinuousHeartbeatError("observation content or digest is invalid")
    return rebuilt


def resource_start_gate(
    resources: Mapping[str, Any], *, policy: Mapping[str, Any]
) -> dict[str, Any]:
    verified_policy = verify_policy(policy)
    row = _validate_resources(resources)
    resource_policy = verified_policy["resource_gate"]
    numerator = int(resource_policy["predicted_output_numerator"])
    denominator = int(resource_policy["predicted_output_denominator"])
    predicted = int(row["predicted_next_output_bytes"])
    scaled = (numerator * predicted + denominator - 1) // denominator
    required = (
        scaled
        + int(row["archive_temp_bytes"])
        + int(resource_policy["disk_reserve_bytes"])
    )
    reasons: list[str] = []
    if row["active_trainer_count"] != 0:
        reasons.append("ACTIVE_TRAINER_PRESENT")
    if row["remote_unverified_generation_count"] != 0:
        reasons.append("REMOTE_UNVERIFIED_GENERATION_PRESENT")
    if row["compression_upload_active_count"] != 0:
        reasons.append("COMPRESSION_UPLOAD_ACTIVE")
    if row["free_bytes"] < required:
        reasons.append("INSUFFICIENT_FREE_BYTES")
    body = {
        "status": "START_ALLOWED" if not reasons else "BACKPRESSURE",
        "reasons": reasons,
        "required_free_bytes": required,
        "start_allowed": not reasons,
    }
    return {**body, "gate_sha256": canonical_sha256(body)}


def initial_state(
    *, policy: Mapping[str, Any], initialized_at_utc: str
) -> dict[str, Any]:
    verified_policy = verify_policy(policy)
    body: dict[str, Any] = {
        "contract": STATE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "policy_sha256": verified_policy["policy_sha256"],
        "revision": 0,
        "previous_state_sha256": None,
        "latest_observation": None,
        "latest_semantic_sha256": None,
        "active_lease": None,
        "completed_obligations": [],
        "last_transition_at_utc": _utc(initialized_at_utc, "initialized_at_utc"),
        "authority": dict(_NO_AUTHORITY),
    }
    body["state_sha256"] = canonical_sha256(body)
    return body


def verify_state(
    value: Mapping[str, Any], *, policy: Mapping[str, Any]
) -> dict[str, Any]:
    verified_policy = verify_policy(policy)
    row = dict(_mapping(value, "state"))
    _exact(row, _STATE_KEYS, "state")
    unsigned = {key: item for key, item in row.items() if key != "state_sha256"}
    if (
        row["contract"] != STATE_CONTRACT
        or _integer(row["schema_version"], "state.schema_version") != SCHEMA_VERSION
        or row["policy_sha256"] != verified_policy["policy_sha256"]
        or canonical_sha256(unsigned) != _sha(row["state_sha256"], "state.state_sha256")
        or row["authority"] != _NO_AUTHORITY
    ):
        raise DojoContinuousHeartbeatError(
            "state contract, policy, digest, or authority is invalid"
        )
    _integer(row["revision"], "state.revision")
    previous = row["previous_state_sha256"]
    if previous is not None:
        _sha(previous, "state.previous_state_sha256")
    observation = row["latest_observation"]
    if observation is None:
        if row["latest_semantic_sha256"] is not None:
            raise DojoContinuousHeartbeatError("state semantic exists without observation")
    else:
        verified_observation = verify_observation(observation, policy=verified_policy)
        if row["latest_semantic_sha256"] != verified_observation["semantic_sha256"]:
            raise DojoContinuousHeartbeatError("state observation semantic binding drifted")
    lease = row["active_lease"]
    if lease is not None:
        _validate_lease(lease)
    completed = _sequence(row["completed_obligations"], "state.completed_obligations")
    if len(completed) > verified_policy["max_completed_obligations"]:
        raise DojoContinuousHeartbeatError("completed obligation bound exceeded")
    obligation_ids: set[str] = set()
    operation_ids: set[str] = set()
    for item in completed:
        completion = _validate_completion(item)
        if completion["obligation_id"] in obligation_ids:
            raise DojoContinuousHeartbeatError("obligation completed more than once")
        if completion["operation_id"] in operation_ids:
            raise DojoContinuousHeartbeatError("operation completed more than once")
        obligation_ids.add(completion["obligation_id"])
        operation_ids.add(completion["operation_id"])
    _utc(row["last_transition_at_utc"], "state.last_transition_at_utc")
    return row


def apply_observation(
    state: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    event_at_utc: str,
) -> tuple[dict[str, Any], bool]:
    """Apply only a semantic change; unchanged health is a write-free no-op."""

    current = verify_state(state, policy=policy)
    observed = verify_observation(observation, policy=policy)
    transition = _utc(event_at_utc, "event_at_utc")
    transition_time = datetime.fromisoformat(transition.replace("Z", "+00:00"))
    observed_time = datetime.fromisoformat(
        observed["observed_at_utc"].replace("Z", "+00:00")
    )
    age_seconds = (transition_time - observed_time).total_seconds()
    if age_seconds < 0 or age_seconds > verify_policy(policy)[
        "max_observation_age_seconds"
    ]:
        raise DojoContinuousHeartbeatError(
            "observation is stale or later than its transition"
        )
    if observed["semantic_sha256"] == current["latest_semantic_sha256"]:
        return current, False
    body = {
        **{key: _copy(item) for key, item in current.items() if key != "state_sha256"},
        "revision": current["revision"] + 1,
        "previous_state_sha256": current["state_sha256"],
        "latest_observation": observed,
        "latest_semantic_sha256": observed["semantic_sha256"],
        "last_transition_at_utc": transition,
    }
    body["state_sha256"] = canonical_sha256(body)
    return body, True


def plan_heartbeat(
    state: Mapping[str, Any], *, policy: Mapping[str, Any]
) -> dict[str, Any]:
    current = verify_state(state, policy=policy)
    observation = current["latest_observation"]
    if observation is None:
        return _decision(
            current,
            action="NO_OBSERVATION",
            phase="BOOTSTRAP",
            obligation_id=None,
            reasons=["HEALTH_OBSERVATION_REQUIRED"],
            resource_gate=None,
        )
    gate = resource_start_gate(observation["resources"], policy=policy)
    if current["active_lease"] is not None:
        lease = current["active_lease"]
        return _decision(
            current,
            action="WAIT_FOR_RESERVED_WORK",
            phase="WORK_RESERVED",
            obligation_id=lease["obligation_id"],
            reasons=["SINGLE_LEASE_OCCUPIED"],
            resource_gate=gate,
            operation_id=lease["operation_id"],
        )
    run = observation["run"]
    process_state = run["process_state"]
    if process_state in {"RESERVED", "RUNNING"}:
        return _decision(
            current,
            action="MONITOR_HEALTH_ONLY",
            phase="RUN_ACTIVE",
            obligation_id=None,
            reasons=[],
            resource_gate=gate,
        )
    terminal_sha = run["terminal_bundle_sha256"]
    failure_sha = run["failure_artifact_sha256"]
    if terminal_sha is not None:
        validate_id = f"validate-terminal:{terminal_sha}"
        validation = _recorded(current, validate_id)
        if validation is None:
            return _decision(
                current,
                action="VALIDATE_TERMINAL_BUNDLE",
                phase="TERMINAL_VALIDATION_PENDING",
                obligation_id=validate_id,
                reasons=[],
                resource_gate=gate,
            )
        if validation["outcome"] == "FAILED":
            review_id = (
                f"review-work-failure:{validation['operation_id']}:"
                f"{validation['result_sha256']}"
            )
            review = _recorded(current, review_id)
            if review is None:
                return _decision(
                    current,
                    action="REVIEW_REQUIRED",
                    phase="TERMINAL_VALIDATION_FAILED",
                    obligation_id=review_id,
                    reasons=["TERMINAL_VALIDATION_WORKER_FAILED"],
                    resource_gate=gate,
                )
            return _decision(
                current,
                action="REVIEW_RECORDED",
                phase="TERMINAL_VALIDATION_FAILURE_REVIEWED",
                obligation_id=None,
                reasons=(
                    []
                    if review["outcome"] == "SUCCESS"
                    else ["FAILURE_REVIEW_WORKER_FAILED_NO_AUTORETRY"]
                ),
                resource_gate=gate,
            )
        review_id = (
            f"review-terminal:{terminal_sha}:{validation['result_sha256']}"
        )
        review = _recorded(current, review_id)
        if review is None:
            return _decision(
                current,
                action="REVIEW_TERMINAL_RESULT",
                phase="TERMINAL_VALIDATED",
                obligation_id=review_id,
                reasons=[],
                resource_gate=gate,
            )
        if review["outcome"] == "FAILED":
            return _decision(
                current,
                action="REVIEW_RECORDED",
                phase="TERMINAL_REVIEW_FAILED_NO_AUTORETRY",
                obligation_id=None,
                reasons=["TERMINAL_REVIEW_WORKER_FAILED_NO_AUTORETRY"],
                resource_gate=gate,
            )
        return _decision(
            current,
            action="TERMINAL_REVIEWED",
            phase="RESULT_BOUNDARY_COMPLETE",
            obligation_id=None,
            reasons=[],
            resource_gate=gate,
        )
    if failure_sha is not None or process_state in {"EXITED", "MISSING"}:
        identity = failure_sha or canonical_sha256(
            {
                "run_id": run["run_id"],
                "process_state": process_state,
                "checkpoint_sha256": run["checkpoint_sha256"],
            }
        )
        review_id = f"review-failure:{identity}"
        review = _recorded(current, review_id)
        if review is None:
            reason = (
                "FAILURE_ARTIFACT_PRESENT"
                if failure_sha is not None
                else "PROCESS_EXITED_WITHOUT_TERMINAL"
            )
            return _decision(
                current,
                action="REVIEW_REQUIRED",
                phase="REVIEW_REQUIRED",
                obligation_id=review_id,
                reasons=[reason],
                resource_gate=gate,
            )
        return _decision(
            current,
            action="REVIEW_RECORDED",
            phase="REVIEW_REQUIRED_RECORDED",
            obligation_id=None,
            reasons=(
                []
                if review["outcome"] == "SUCCESS"
                else ["FAILURE_REVIEW_WORKER_FAILED_NO_AUTORETRY"]
            ),
            resource_gate=gate,
        )
    if not gate["start_allowed"]:
        return _decision(
            current,
            action="RESOURCE_BACKPRESSURE",
            phase="RESOURCE_BACKPRESSURE",
            obligation_id=None,
            reasons=gate["reasons"],
            resource_gate=gate,
        )
    return _decision(
        current,
        action="MANUAL_RUN_START_ELIGIBLE",
        phase="IDLE_READY",
        obligation_id="manual-run-start-eligibility",
        reasons=["PLANNER_ONLY_NO_PROCESS_START_AUTHORITY"],
        resource_gate=gate,
    )


def reserve_decision(
    state: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    reserved_at_utc: str,
) -> tuple[dict[str, Any], bool]:
    current = verify_state(state, policy=policy)
    planned = verify_decision(decision, state=current, policy=policy)
    if planned["action"] not in {
        "VALIDATE_TERMINAL_BUNDLE",
        "REVIEW_TERMINAL_RESULT",
        "REVIEW_REQUIRED",
    }:
        raise DojoContinuousHeartbeatError("decision is not a reservable obligation")
    obligation_id = planned["obligation_id"]
    assert obligation_id is not None
    if _recorded(current, obligation_id) is not None:
        raise DojoContinuousHeartbeatError("obligation is already recorded")
    operation_id = _operation_id(current, planned)
    active = current["active_lease"]
    if active is not None:
        if active["operation_id"] == operation_id:
            return current, False
        raise DojoContinuousHeartbeatError("single heartbeat lease is occupied")
    lease = {
        "operation_id": operation_id,
        "action": planned["action"],
        "obligation_id": obligation_id,
        "input_semantic_sha256": current["latest_semantic_sha256"],
        "reserved_at_utc": _utc(reserved_at_utc, "reserved_at_utc"),
    }
    body = _next_state(current, transition_at_utc=reserved_at_utc)
    body["active_lease"] = lease
    body["state_sha256"] = canonical_sha256(body)
    return body, True


def complete_reserved_work(
    state: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    operation_id: str,
    result_sha256: str,
    outcome: str,
    completed_at_utc: str,
) -> dict[str, Any]:
    current = verify_state(state, policy=policy)
    operation = _sha(operation_id, "operation_id")
    result = _sha(result_sha256, "result_sha256")
    lease = current["active_lease"]
    if lease is None or lease["operation_id"] != operation:
        completed = next(
            (
                row
                for row in current["completed_obligations"]
                if row["operation_id"] == operation
            ),
            None,
        )
        if (
            completed is not None
            and completed["result_sha256"] == result
            and completed["outcome"] == outcome
        ):
            return current
        raise DojoContinuousHeartbeatError("operation has no matching active lease")
    if outcome not in {"SUCCESS", "FAILED"}:
        raise DojoContinuousHeartbeatError("outcome must be SUCCESS or FAILED")
    body = _next_state(current, transition_at_utc=completed_at_utc)
    body["active_lease"] = None
    if len(body["completed_obligations"]) >= verify_policy(policy)[
        "max_completed_obligations"
    ]:
        raise DojoContinuousHeartbeatError("completed obligation registry is full")
    completion = {
        "operation_id": operation,
        "action": lease["action"],
        "obligation_id": lease["obligation_id"],
        "input_semantic_sha256": lease["input_semantic_sha256"],
        "result_sha256": result,
        "outcome": outcome,
        "completed_at_utc": _utc(completed_at_utc, "completed_at_utc"),
    }
    body["completed_obligations"] = [
        *body["completed_obligations"],
        completion,
    ]
    body["state_sha256"] = canonical_sha256(body)
    return body


def verify_decision(
    value: Mapping[str, Any], *, state: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    current = verify_state(state, policy=policy)
    row = dict(_mapping(value, "decision"))
    _exact(row, _DECISION_KEYS, "decision")
    expected = plan_heartbeat(current, policy=policy)
    if row != expected:
        raise DojoContinuousHeartbeatError("decision is stale or non-canonical")
    return row


def build_event(
    *,
    sequence: int,
    previous_event_sha256: str,
    event_type: str,
    prior_state: Mapping[str, Any] | None,
    state: Mapping[str, Any],
    policy: Mapping[str, Any],
    event_at_utc: str,
) -> dict[str, Any]:
    verified_state = verify_state(state, policy=policy)
    if event_type not in {
        "INITIALIZED",
        "OBSERVATION_CHANGED",
        "WORK_RESERVED",
        "WORK_COMPLETED",
        "WORK_FAILED",
    }:
        raise DojoContinuousHeartbeatError("event_type is unsupported")
    expected_sequence = _integer(sequence, "event.sequence")
    previous_event = _sha(
        previous_event_sha256,
        "event.previous_event_sha256",
        allow_zero=True,
    )
    if expected_sequence == 0:
        if event_type != "INITIALIZED" or prior_state is not None or previous_event != GENESIS_SHA256:
            raise DojoContinuousHeartbeatError("genesis event shape is invalid")
        operation_id = None
        input_sha = verified_state["policy_sha256"]
        if (
            verified_state["revision"] != 0
            or verified_state["previous_state_sha256"] is not None
        ):
            raise DojoContinuousHeartbeatError("genesis state is not initial")
    else:
        if prior_state is None:
            raise DojoContinuousHeartbeatError("non-genesis event requires prior state")
        prior = verify_state(prior_state, policy=policy)
        if verified_state["previous_state_sha256"] != prior["state_sha256"]:
            raise DojoContinuousHeartbeatError("event state transition is not contiguous")
        lease = verified_state["active_lease"] or prior["active_lease"]
        operation_id = lease["operation_id"] if lease is not None else None
        input_sha = prior["state_sha256"]
        if verified_state["last_transition_at_utc"] != _utc(
            event_at_utc, "event_at_utc"
        ):
            raise DojoContinuousHeartbeatError(
                "event timestamp does not bind its state transition"
            )
        if event_type == "OBSERVATION_CHANGED":
            if (
                verified_state["latest_semantic_sha256"]
                == prior["latest_semantic_sha256"]
                or verified_state["active_lease"] != prior["active_lease"]
                or verified_state["completed_obligations"]
                != prior["completed_obligations"]
            ):
                raise DojoContinuousHeartbeatError(
                    "observation event carries a non-observation transition"
                )
        elif event_type == "WORK_RESERVED":
            if prior["active_lease"] is not None or verified_state[
                "active_lease"
            ] is None:
                raise DojoContinuousHeartbeatError(
                    "reservation event does not acquire the single lease"
                )
        elif event_type in {"WORK_COMPLETED", "WORK_FAILED"}:
            if (
                prior["active_lease"] is None
                or verified_state["active_lease"] is not None
                or len(verified_state["completed_obligations"])
                != len(prior["completed_obligations"]) + 1
            ):
                raise DojoContinuousHeartbeatError(
                    "completion event does not consume the single lease once"
                )
            completion = verified_state["completed_obligations"][-1]
            expected_outcome = (
                "SUCCESS" if event_type == "WORK_COMPLETED" else "FAILED"
            )
            if (
                completion["operation_id"]
                != prior["active_lease"]["operation_id"]
                or completion["outcome"] != expected_outcome
            ):
                raise DojoContinuousHeartbeatError(
                    "completion event outcome or operation is inconsistent"
                )
    body = {
        "contract": EVENT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "sequence": expected_sequence,
        "previous_event_sha256": previous_event,
        "event_type": event_type,
        "operation_id": operation_id,
        "input_sha256": input_sha,
        "event_at_utc": _utc(event_at_utc, "event_at_utc"),
        "state": verified_state,
    }
    return {**body, "event_sha256": canonical_sha256(body)}


def verify_event(
    value: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    expected_sequence: int,
    previous_event_sha256: str,
    prior_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row = dict(_mapping(value, "event"))
    _exact(row, _EVENT_KEYS, "event")
    rebuilt = build_event(
        sequence=expected_sequence,
        previous_event_sha256=previous_event_sha256,
        event_type=row["event_type"],
        prior_state=prior_state,
        state=row["state"],
        policy=policy,
        event_at_utc=row["event_at_utc"],
    )
    if row != rebuilt:
        raise DojoContinuousHeartbeatError("event chain or digest is invalid")
    return rebuilt


def _decision(
    state: Mapping[str, Any],
    *,
    action: str,
    phase: str,
    obligation_id: str | None,
    reasons: Sequence[str],
    resource_gate: Mapping[str, Any] | None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    if action not in _ACTIONS:
        raise DojoContinuousHeartbeatError("decision action is unsupported")
    if obligation_id is not None:
        _identifier(obligation_id, "decision.obligation_id")
    body = {
        "contract": DECISION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "policy_sha256": state["policy_sha256"],
        "state_sha256": state["state_sha256"],
        "observation_semantic_sha256": state["latest_semantic_sha256"],
        "phase": _identifier(phase, "decision.phase"),
        "action": action,
        "obligation_id": obligation_id,
        "operation_id": operation_id,
        "reasons": [_identifier(reason, "decision.reason") for reason in reasons],
        "resource_gate": None if resource_gate is None else _copy(resource_gate),
        "permissions": {
            "work_reservation_allowed": action
            in {
                "VALIDATE_TERMINAL_BUNDLE",
                "REVIEW_TERMINAL_RESULT",
                "REVIEW_REQUIRED",
            },
            "manual_run_start_eligible": action == "MANUAL_RUN_START_ELIGIBLE",
            **_NO_AUTHORITY,
        },
    }
    if operation_id is None and obligation_id is not None and body[
        "permissions"
    ]["work_reservation_allowed"]:
        operation_id = canonical_sha256(
            {
                "policy_sha256": state["policy_sha256"],
                "state_sha256": state["state_sha256"],
                "action": action,
                "obligation_id": obligation_id,
                "input_semantic_sha256": state["latest_semantic_sha256"],
            }
        )
        body["operation_id"] = operation_id
    body["decision_sha256"] = canonical_sha256(body)
    return body


def _operation_id(state: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    expected = canonical_sha256(
        {
            "policy_sha256": state["policy_sha256"],
            "state_sha256": state["state_sha256"],
            "action": decision["action"],
            "obligation_id": decision["obligation_id"],
            "input_semantic_sha256": state["latest_semantic_sha256"],
        }
    )
    if decision["operation_id"] != expected:
        raise DojoContinuousHeartbeatError("decision operation identity drifted")
    return expected


def _validate_run(value: Any) -> dict[str, Any]:
    row = dict(_mapping(value, "observation.run"))
    _exact(row, _RUN_KEYS, "observation.run")
    process_state = row["process_state"]
    if process_state not in _PROCESS_STATES:
        raise DojoContinuousHeartbeatError("run process_state is unsupported")
    expected = _integer(
        row["expected_coordinate_count"], "run.expected_coordinate_count"
    )
    completed = _integer(
        row["completed_coordinate_count"], "run.completed_coordinate_count"
    )
    if completed > expected:
        raise DojoContinuousHeartbeatError("run progress exceeds its denominator")
    run_id = row["run_id"]
    process_identity = row["process_identity_sha256"]
    checkpoint = row["checkpoint_sha256"]
    terminal = row["terminal_bundle_sha256"]
    failure = row["failure_artifact_sha256"]
    for field, item in (
        ("process_identity_sha256", process_identity),
        ("checkpoint_sha256", checkpoint),
        ("terminal_bundle_sha256", terminal),
        ("failure_artifact_sha256", failure),
    ):
        if item is not None:
            _sha(item, f"run.{field}")
    if row["economics_fields_present"] is not False:
        raise DojoContinuousHeartbeatError("active-run health must exclude economics")
    if terminal is not None and failure is not None:
        raise DojoContinuousHeartbeatError("terminal and failure are mutually exclusive")
    if process_state == "IDLE":
        if any(
            item is not None
            for item in (run_id, process_identity, checkpoint, terminal, failure)
        ) or expected != 0 or completed != 0:
            raise DojoContinuousHeartbeatError("idle run retains active evidence")
    else:
        _identifier(run_id, "run.run_id")
        if expected <= 0:
            raise DojoContinuousHeartbeatError("non-idle run needs a denominator")
        if process_state != "RESERVED":
            _sha(process_identity, "run.process_identity_sha256")
    if process_state in {"RESERVED", "RUNNING"} and (terminal or failure):
        raise DojoContinuousHeartbeatError("active run cannot claim terminal evidence")
    if completed > 0 and checkpoint is None:
        raise DojoContinuousHeartbeatError("run progress requires a checkpoint digest")
    if terminal is not None and completed != expected:
        raise DojoContinuousHeartbeatError("terminal run denominator is incomplete")
    return row


def _validate_resources(value: Any) -> dict[str, int]:
    row = dict(_mapping(value, "observation.resources"))
    _exact(row, _RESOURCE_KEYS, "observation.resources")
    return {
        key: _integer(item, f"resources.{key}") for key, item in row.items()
    }


def _validate_lease(value: Any) -> dict[str, Any]:
    row = dict(_mapping(value, "lease"))
    _exact(row, _LEASE_KEYS, "lease")
    _sha(row["operation_id"], "lease.operation_id")
    if row["action"] not in _ACTIONS:
        raise DojoContinuousHeartbeatError("lease action is unsupported")
    _identifier(row["obligation_id"], "lease.obligation_id")
    _sha(row["input_semantic_sha256"], "lease.input_semantic_sha256")
    _utc(row["reserved_at_utc"], "lease.reserved_at_utc")
    return row


def _validate_completion(value: Any) -> dict[str, Any]:
    row = dict(_mapping(value, "completion"))
    _exact(row, _COMPLETION_KEYS, "completion")
    _sha(row["operation_id"], "completion.operation_id")
    if row["action"] not in _ACTIONS:
        raise DojoContinuousHeartbeatError("completion action is unsupported")
    _identifier(row["obligation_id"], "completion.obligation_id")
    _sha(row["input_semantic_sha256"], "completion.input_semantic_sha256")
    _sha(row["result_sha256"], "completion.result_sha256")
    if row["outcome"] not in {"SUCCESS", "FAILED"}:
        raise DojoContinuousHeartbeatError("completion outcome is unsupported")
    _utc(row["completed_at_utc"], "completion.completed_at_utc")
    return row


def _recorded(state: Mapping[str, Any], obligation_id: str) -> Mapping[str, Any] | None:
    return next(
        (
            row
            for row in state["completed_obligations"]
            if row["obligation_id"] == obligation_id
        ),
        None,
    )


def _next_state(state: Mapping[str, Any], *, transition_at_utc: str) -> dict[str, Any]:
    return {
        **{key: _copy(item) for key, item in state.items() if key != "state_sha256"},
        "revision": state["revision"] + 1,
        "previous_state_sha256": state["state_sha256"],
        "last_transition_at_utc": _utc(transition_at_utc, "transition_at_utc"),
    }


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoContinuousHeartbeatError(f"{field} must be a string-keyed object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DojoContinuousHeartbeatError(f"{field} must be a sequence")
    return value


def _exact(value: Mapping[str, Any], keys: frozenset[str], field: str) -> None:
    actual = frozenset(value)
    if actual != keys:
        raise DojoContinuousHeartbeatError(
            f"{field} schema mismatch: missing={sorted(keys-actual)}, "
            f"extra={sorted(actual-keys)}"
        )


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise DojoContinuousHeartbeatError(f"{field} must be a bounded identifier")
    return value


def _absolute_path(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or len(value) > 1024
        or "\x00" in value
        or "//" in value
        or value.endswith("/")
        or any(component in {".", ".."} for component in value.split("/"))
    ):
        raise DojoContinuousHeartbeatError(
            f"{field} must be a normalized bounded absolute path"
        )
    return value


def _sha(value: Any, field: str, *, allow_zero: bool = False) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoContinuousHeartbeatError(f"{field} must be a lowercase SHA-256")
    if not allow_zero and value == GENESIS_SHA256:
        raise DojoContinuousHeartbeatError(f"{field} must not be the zero digest")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoContinuousHeartbeatError(f"{field} must be an integer >= {minimum}")
    return value


def _utc(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DojoContinuousHeartbeatError(f"{field} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoContinuousHeartbeatError(f"{field} must be a UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DojoContinuousHeartbeatError(f"{field} must be a UTC timestamp")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_json(value: Any, field: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DojoContinuousHeartbeatError(f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{field}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DojoContinuousHeartbeatError(f"{field} has a non-string key")
            _validate_json(item, f"{field}.{key}")
        return
    raise DojoContinuousHeartbeatError(f"{field} contains a non-JSON value")


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


__all__ = [
    "DECISION_CONTRACT",
    "DojoContinuousHeartbeatError",
    "EVENT_CONTRACT",
    "GENESIS_SHA256",
    "LOCAL_PROBE_CONTRACT",
    "LOCAL_RUN_STATUS_CONTRACT",
    "OBSERVATION_CONTRACT",
    "POLICY_CONTRACT",
    "SCHEMA_VERSION",
    "STATE_CONTRACT",
    "apply_observation",
    "build_local_observation",
    "build_event",
    "canonical_json_bytes",
    "canonical_sha256",
    "complete_reserved_work",
    "initial_state",
    "plan_heartbeat",
    "reserve_decision",
    "resource_start_gate",
    "seal_observation",
    "seal_local_run_status",
    "verify_decision",
    "verify_event",
    "verify_observation",
    "verify_local_probe_manifest",
    "verify_local_run_status",
    "verify_policy",
    "verify_state",
]
