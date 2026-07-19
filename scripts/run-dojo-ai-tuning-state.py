#!/usr/bin/env python3
"""Operate the append-only, research-only DOJO AI tuning state machine.

This command only records and verifies orchestration state.  It has no model
client, replay runner, broker client, order path, live permission, or promotion
authority.  Every mutating command uses both the caller's expected state-store
event tip and expected parent-state SHA-256 as compare-and-swap tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_ai_tuning_state import (  # noqa: E402
    DojoAITuningStateError,
    abandon_incomplete_lineage,
    append_state_transition,
    bind_terminal_evaluation,
    initialize_state_store,
    initialize_tuning_state,
    mark_incomplete_run,
    mark_run_dispatched,
    record_model_response,
    reserve_model_invocation,
    reserve_run_dispatch,
    status_artifact,
    verify_state_store,
)
from quant_rabbit.dojo_ai_trainer_packet import (  # noqa: E402
    DojoAITrainerPacketError,
    canonical_packet_bytes,
    verify_trainer_packet,
)
from quant_rabbit.dojo_candidate_lineage_registry import (  # noqa: E402
    CandidateLineageError,
    CandidateLineageSnapshot,
    verify_registry,
)


CLI_STATUS_CONTRACT = "QR_DOJO_AI_TUNING_CLI_STATUS_V1"
SCHEMA_VERSION = 1
# The state store already bounds one event to 8 MiB.  Matching that engineering
# bound prevents a model/request artifact from turning this local verifier into
# an unbounded-memory reader; it is unrelated to trading or risk decisions.
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
REQUEST_ARTIFACT_DIRECTORY = "ai-trainer-model-requests"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}\Z")
_SOURCE_BINDING_KEYS = frozenset(
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


class DojoAITuningCliError(ValueError):
    """A CLI input or local artifact operation failed closed."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="initialize a new append-only store")
    init.add_argument("--state-events", type=Path, required=True)
    init.add_argument("--lineage-events", type=Path, required=True)
    init.add_argument("--artifact-root", type=Path, required=True)
    init.add_argument("--sealed-study", type=Path, required=True)
    init.add_argument("--expected-lineage-tip-sha256", required=True)

    status = commands.add_parser("status", help="verify and print compact status")
    status.add_argument("--state-events", type=Path, required=True)

    reserve_model = commands.add_parser(
        "reserve-model", help="reserve one model invocation before dispatch"
    )
    _add_transition_cas(reserve_model)
    _add_lineage_paths(reserve_model)
    reserve_model.add_argument("--invocation-id", required=True)
    reserve_model.add_argument("--request-artifact", type=Path, required=True)
    reserve_model.add_argument("--event-at-utc", required=True)

    response = commands.add_parser(
        "record-response", help="persist raw model bytes, then charge the response"
    )
    _add_transition_cas(response)
    response.add_argument("--artifact-root", type=Path, required=True)
    response.add_argument("--invocation-id", required=True)
    response.add_argument("--response-input", type=Path, required=True)
    response.add_argument("--response-artifact", type=Path, required=True)
    response.add_argument("--event-at-utc", required=True)

    reserve_run = commands.add_parser(
        "reserve-run", help="reserve an exact lineage-sealed TRAIN replay"
    )
    _add_transition_cas(reserve_run)
    _add_lineage_paths(reserve_run)
    reserve_run.add_argument("--sealed-study", type=Path, required=True)
    reserve_run.add_argument("--dispatch-id", required=True)
    reserve_run.add_argument("--event-at-utc", required=True)

    dispatched = commands.add_parser(
        "mark-dispatched", help="record that the reserved replay was started"
    )
    _add_transition_cas(dispatched)
    _add_lineage_paths(dispatched)
    dispatched.add_argument("--dispatch-id", required=True)
    dispatched.add_argument("--event-at-utc", required=True)

    bind = commands.add_parser(
        "bind-result", help="bind the registry's exact terminal RESULT_BOUND event"
    )
    _add_transition_cas(bind)
    _add_lineage_paths(bind)
    bind.add_argument("--event-at-utc", required=True)

    incomplete = commands.add_parser(
        "mark-incomplete", help="burn the active attempt into review-required state"
    )
    _add_transition_cas(incomplete)
    incomplete.add_argument("--reason-code", required=True)
    incomplete.add_argument("--event-at-utc", required=True)

    abandon = commands.add_parser(
        "abandon", help="terminate an incomplete lineage after explicit review"
    )
    _add_transition_cas(abandon)
    abandon.add_argument("--review-id", required=True)
    abandon.add_argument("--rationale", required=True)
    abandon.add_argument("--event-at-utc", required=True)
    return parser


def _add_transition_cas(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-events", type=Path, required=True)
    parser.add_argument("--expected-tip-event-sha256", required=True)
    parser.add_argument("--expected-parent-state-sha256", required=True)


def _add_lineage_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lineage-events", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)


def _reject_constant(token: str) -> None:
    raise DojoAITuningCliError("artifact JSON contains a non-finite number")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DojoAITuningCliError("artifact JSON contains a duplicate key")
        result[key] = value
    return result


def _strict_json(raw: bytes) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except DojoAITuningCliError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DojoAITuningCliError("artifact is not strict JSON") from exc


def _read_artifact(path: Path, *, label: str, allow_empty: bool = False) -> bytes:
    candidate = path.absolute()
    try:
        expected = candidate.lstat()
    except OSError as exc:
        raise DojoAITuningCliError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(expected.st_mode)
        or stat.S_ISLNK(expected.st_mode)
        or (expected.st_size <= 0 and not allow_empty)
    ):
        raise DojoAITuningCliError(f"{label} must be a nonempty regular file")
    if expected.st_size > MAX_ARTIFACT_BYTES:
        raise DojoAITuningCliError(f"{label} exceeds the artifact size limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(candidate, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            before = os.fstat(handle.fileno())
            raw = handle.read(MAX_ARTIFACT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise DojoAITuningCliError(f"{label} could not be read safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_dev != expected.st_dev
        or before.st_ino != expected.st_ino
        or len(raw) != before.st_size
    ):
        raise DojoAITuningCliError(f"{label} changed while being read")
    return raw


def _read_json_artifact(path: Path, *, label: str) -> Any:
    return _strict_json(_read_artifact(path, label=label))


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise DojoAITuningCliError(f"{label} is invalid")
    return value


def _lower_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DojoAITuningCliError(f"{label} must be a lowercase SHA-256")
    return value


def _utc(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DojoAITuningCliError(f"{label} must be ISO-8601 UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoAITuningCliError(f"{label} must be ISO-8601 UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DojoAITuningCliError(f"{label} must be UTC")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _open_real_directory(path: Path, *, label: str) -> int:
    candidate = path.absolute()
    try:
        expected = candidate.lstat()
    except OSError as exc:
        raise DojoAITuningCliError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(expected.st_mode) or stat.S_ISLNK(expected.st_mode):
        raise DojoAITuningCliError(f"{label} must be a real directory")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(candidate, flags)
        actual = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise DojoAITuningCliError(f"{label} could not be opened safely") from exc
    assert descriptor is not None
    if (
        not stat.S_ISDIR(actual.st_mode)
        or actual.st_dev != expected.st_dev
        or actual.st_ino != expected.st_ino
    ):
        os.close(descriptor)
        raise DojoAITuningCliError(f"{label} changed while being opened")
    return descriptor


def _dedicated_request_target(artifact_root: Path, *, invocation_id: str) -> Path:
    """Return a content-isolated request path under a real artifact root.

    The caller does not choose the durable destination.  One hashed filename is
    reserved per invocation id, so a crash retry can only reuse the same slot.
    """

    invocation_key = _identifier(invocation_id, label="invocation_id")
    root = artifact_root.absolute()
    try:
        root_resolved = root.resolve(strict=True)
    except OSError as exc:
        raise DojoAITuningCliError("artifact root is unavailable") from exc
    if root != root_resolved:
        raise DojoAITuningCliError("artifact root must not traverse a symlink")
    root_fd = _open_real_directory(root, label="artifact root")
    created = False
    child_fd: int | None = None
    try:
        try:
            os.mkdir(REQUEST_ARTIFACT_DIRECTORY, 0o700, dir_fd=root_fd)
            created = True
            os.fsync(root_fd)
        except FileExistsError:
            pass
        try:
            child_state = os.stat(
                REQUEST_ARTIFACT_DIRECTORY,
                dir_fd=root_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise DojoAITuningCliError(
                "dedicated request artifact directory is unavailable"
            ) from exc
        if not stat.S_ISDIR(child_state.st_mode) or stat.S_ISLNK(child_state.st_mode):
            raise DojoAITuningCliError(
                "dedicated request artifact directory must be a real directory"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        child_fd = os.open(REQUEST_ARTIFACT_DIRECTORY, flags, dir_fd=root_fd)
        actual = os.fstat(child_fd)
        if (
            not stat.S_ISDIR(actual.st_mode)
            or actual.st_dev != child_state.st_dev
            or actual.st_ino != child_state.st_ino
        ):
            raise DojoAITuningCliError(
                "dedicated request artifact directory changed while being opened"
            )
    except BaseException:
        if created:
            try:
                os.rmdir(REQUEST_ARTIFACT_DIRECTORY, dir_fd=root_fd)
                os.fsync(root_fd)
            except OSError:
                pass
        raise
    finally:
        if child_fd is not None:
            os.close(child_fd)
        os.close(root_fd)
    invocation_digest = _sha256(invocation_key.encode("utf-8"))
    return (
        root / REQUEST_ARTIFACT_DIRECTORY / (f"{invocation_digest}.trainer-packet.json")
    )


def _persist_raw_request(
    raw: bytes, *, artifact_root: Path, invocation_id: str
) -> dict[str, Any]:
    """Durably bind exact trainer-packet bytes before state reservation."""

    target = _dedicated_request_target(artifact_root, invocation_id=invocation_id)
    root = artifact_root.absolute()
    try:
        root_resolved = root.resolve(strict=True)
        parent_resolved = target.parent.resolve(strict=True)
        parent_resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise DojoAITuningCliError(
            "durable request artifact must remain inside artifact root"
        ) from exc
    if target.parent.absolute() != parent_resolved:
        raise DojoAITuningCliError(
            "durable request artifact parent must not traverse a symlink"
        )
    parent_fd = _open_real_directory(parent_resolved, label="request artifact parent")
    created = False
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(target.name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            existing = _read_existing_request(parent_fd, target.name)
            if existing != raw:
                raise DojoAITuningCliError(
                    "durable request artifact already exists with different bytes"
                )
        else:
            created = True
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = None
                written = handle.write(raw)
                if written != len(raw):
                    raise DojoAITuningCliError(
                        "durable request artifact write was incomplete"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.fsync(parent_fd)
    except BaseException:
        if created:
            try:
                os.unlink(target.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    return {
        "artifact_kind": "TRAINER_PACKET_V1_MODEL_REQUEST",
        "path": str(target),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
        "created_exclusively": created,
    }


def _read_existing_request(parent_fd: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            before = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size <= 0
                or before.st_size > MAX_ARTIFACT_BYTES
            ):
                raise DojoAITuningCliError(
                    "durable request artifact must be a bounded singly-linked regular file"
                )
            raw = handle.read(MAX_ARTIFACT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except DojoAITuningCliError:
        raise
    except OSError as exc:
        raise DojoAITuningCliError(
            "durable request artifact could not be verified"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(raw) != before.st_size
    ):
        raise DojoAITuningCliError("durable request artifact changed while being read")
    return raw


def _reservation_cas_mode(
    snapshot: dict[str, Any],
    *,
    expected_tip_event_sha256: str,
    expected_parent_state_sha256: str,
    invocation_id: str,
    request_sha256: str,
    event_at_utc: str,
) -> str:
    """Validate a current reservation or its exact immediate crash replay."""

    latest_event = snapshot["latest_event"]
    latest_state = snapshot["latest_state"]
    if (
        snapshot["latest_event_sha256"] == expected_tip_event_sha256
        and latest_state["state_sha256"] == expected_parent_state_sha256
    ):
        active_attempts = [
            attempt
            for attempt in latest_state["attempts"]
            if attempt["terminal"] is None
        ]
        if latest_state["phase"] != "READY_FOR_MODEL" or active_attempts:
            raise DojoAITuningCliError(
                "current model reservation requires READY_FOR_MODEL with no active "
                "AI attempt"
            )
        return "CURRENT"
    if (
        latest_event["previous_event_sha256"] != expected_tip_event_sha256
        or latest_event["parent_state_sha256"] != expected_parent_state_sha256
        or latest_state["previous_state_sha256"] != expected_parent_state_sha256
        or latest_state["phase"] != "MODEL_INVOCATION_RESERVED"
        or not latest_state["attempts"]
    ):
        raise DojoAITuningCliError("stale or forked model-reservation CAS tokens")
    attempt = latest_state["attempts"][-1]
    invocations = attempt["invocations"]
    if not invocations:
        raise DojoAITuningCliError("model reservation replay lacks an invocation")
    invocation = invocations[-1]
    if (
        invocation["invocation_id"] != invocation_id
        or invocation["request_sha256"] != request_sha256
        or invocation["reserved_at_utc"] != event_at_utc
        or invocation["response_sha256"] is not None
    ):
        raise DojoAITuningCliError(
            "model reservation replay is not the exact immediate transition"
        )
    return "REPLAY_LAST"


def _verify_packet_source_bindings(
    packet: dict[str, Any],
    *,
    state: dict[str, Any],
    lineage: CandidateLineageSnapshot,
    expected_tuning_state_sha256: str,
    cas_mode: str,
) -> None:
    """Bind the verified packet to the exact current result boundary."""

    bindings = packet.get("source_bindings")
    if not isinstance(bindings, dict) or set(bindings) != _SOURCE_BINDING_KEYS:
        raise DojoAITuningCliError("trainer packet source_bindings schema drifted")
    if (
        not isinstance(bindings["run_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", bindings["run_sha256"]) is None
    ):
        raise DojoAITuningCliError(
            "trainer packet source binding run_sha256 is invalid"
        )
    if not lineage.events or lineage.events[-1]["event_type"] != "RESULT_BOUND":
        raise DojoAITuningCliError(
            "model reservation requires the latest lineage RESULT_BOUND event"
        )
    latest_event = lineage.events[-1]
    latest_result = lineage.results[-1] if lineage.results else None
    if latest_result is None or latest_event["body"] != latest_result:
        raise DojoAITuningCliError("latest lineage result event is not exact")
    result_binding = state["last_terminal_result_binding"]
    expected = {
        "registry_id": lineage.registry_id,
        "lineage_prefix": lineage.lineage_prefix,
        "attempt_ordinal": latest_result["attempt_ordinal"],
        "study_sha256": latest_result["study_sha256"],
        "evaluation_sha256": latest_result["evaluation_sha256"],
        "evaluation_artifact_sha256": latest_result["evaluation_artifact_sha256"],
        "lineage_result_event_sha256": latest_event["event_sha256"],
        "lineage_tip_sha256": lineage.latest_event_sha256,
        "tuning_state_sha256": expected_tuning_state_sha256,
        "fixed_envelope_sha256": state["fixed_envelope_sha256"],
        "external_witness_status": lineage.external_witness_status,
        "exact_result_binding_verified": True,
    }
    for key, expected_value in expected.items():
        if bindings[key] != expected_value:
            raise DojoAITuningCliError(
                f"trainer packet source binding {key} is stale or foreign"
            )
    envelope = state["fixed_envelope"]
    fixed_environment = packet.get("fixed_environment")
    if not isinstance(fixed_environment, dict):
        raise DojoAITuningCliError("trainer packet fixed environment is invalid")
    expected_environment_fields = {
        "window_role": envelope["window_role"],
        "window": envelope["window"],
        "initial_balance_jpy": envelope["initial_balance_jpy"],
        "trade_pairs": envelope["trade_pairs"],
        "feed_pairs": envelope["feed_pairs"],
        "intrabar_paths": envelope["intrabar_paths"],
        "cost_arms": envelope["cost_arms"],
        "thresholds": envelope["thresholds"],
    }
    for key, expected_value in expected_environment_fields.items():
        if fixed_environment.get(key) != expected_value:
            raise DojoAITuningCliError(
                f"trainer packet fixed environment {key} diverges from the envelope"
            )
    if cas_mode == "CURRENT":
        tuning_status = status_artifact(state)
        expected_budget = {
            "phase": tuning_status["phase"],
            "attempts_consumed": tuning_status["attempts_consumed"],
            "attempts_remaining": state["max_attempts"]
            - tuning_status["attempts_consumed"],
            "max_attempts": state["max_attempts"],
            "proposal_slots_consumed": tuning_status["proposal_slots_consumed"],
            "proposal_slots_remaining": state["max_proposal_slots"]
            - tuning_status["proposal_slots_consumed"],
            "max_proposal_slots": state["max_proposal_slots"],
            "invalid_proposal_count": tuning_status["invalid_proposal_count"],
            "duplicate_proposal_count": tuning_status["duplicate_proposal_count"],
        }
        if packet.get("search_budget") != expected_budget:
            raise DojoAITuningCliError(
                "trainer packet search budget diverges from the tuning state"
            )
    if (
        result_binding["registry_id"] != bindings["registry_id"]
        or result_binding["lineage_prefix"] != bindings["lineage_prefix"]
        or result_binding["attempt_ordinal"] != bindings["attempt_ordinal"]
        or result_binding["study_sha256"] != bindings["study_sha256"]
        or result_binding["evaluation_sha256"] != bindings["evaluation_sha256"]
        or result_binding["evaluation_artifact_sha256"]
        != bindings["evaluation_artifact_sha256"]
        or result_binding["result_event_sha256"]
        != bindings["lineage_result_event_sha256"]
        or result_binding["lineage_tip_sha256"] != bindings["lineage_tip_sha256"]
    ):
        raise DojoAITuningCliError(
            "trainer packet diverges from the tuning state's terminal result binding"
        )


def _persist_raw_response(
    source: Path, destination: Path, *, artifact_root: Path
) -> tuple[bytes, dict[str, Any]]:
    """Persist model bytes before state mutation, with crash-safe exact replay."""

    raw = _read_artifact(source, label="raw response input", allow_empty=True)
    digest = _sha256(raw)
    root = artifact_root.absolute()
    target = destination.absolute()
    if source.absolute() == target:
        raise DojoAITuningCliError(
            "raw response input and durable response artifact must differ"
        )
    try:
        root_resolved = root.resolve(strict=True)
        parent_resolved = target.parent.resolve(strict=True)
        parent_resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise DojoAITuningCliError(
            "durable response artifact must have a real parent inside artifact root"
        ) from exc
    if target.parent.absolute() != parent_resolved:
        raise DojoAITuningCliError(
            "durable response artifact parent must not traverse a symlink"
        )
    parent_fd = _open_real_directory(parent_resolved, label="response artifact parent")
    created = False
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(target.name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            try:
                target_state = os.stat(
                    target.name, dir_fd=parent_fd, follow_symlinks=False
                )
            except OSError as exc:
                raise DojoAITuningCliError(
                    "durable response artifact could not be verified"
                ) from exc
            if target_state.st_nlink != 1:
                raise DojoAITuningCliError(
                    "durable response artifact must not have hard links"
                )
            existing = _read_artifact(
                target, label="durable response artifact", allow_empty=True
            )
            if existing != raw:
                raise DojoAITuningCliError(
                    "durable response artifact already exists with different bytes"
                )
        else:
            created = True
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = None
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.fsync(parent_fd)
    except BaseException:
        if created:
            try:
                os.unlink(target.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    return raw, {
        "path": str(target),
        "sha256": digest,
        "size_bytes": len(raw),
        "created_exclusively": created,
    }


def _store_status(
    snapshot: dict[str, Any], *, artifact_receipt: dict[str, Any] | None = None
) -> dict[str, Any]:
    compact = status_artifact(snapshot["latest_state"])
    result = {
        "contract": CLI_STATUS_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "state_store": {
            "event_count": snapshot["event_count"],
            "latest_sequence": snapshot["latest_sequence"],
            "latest_event_sha256": snapshot["latest_event_sha256"],
            "latest_state_sha256": snapshot["latest_state"]["state_sha256"],
            "automation_ready": snapshot["automation_ready"],
        },
        "tuning_status": compact,
        "artifact_receipt": artifact_receipt,
        "research_train_only": True,
        "holdout_access_allowed": False,
        "forward_access_allowed": False,
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return result


def _transition(
    args: argparse.Namespace,
    derive: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    snapshot = verify_state_store(args.state_events)
    child = derive(snapshot["latest_state"])
    return append_state_transition(
        args.state_events,
        child,
        expected_tip_event_sha256=args.expected_tip_event_sha256,
        expected_parent_state_sha256=args.expected_parent_state_sha256,
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "status":
        return _store_status(verify_state_store(args.state_events))
    if args.command == "init":
        sealed_study = _read_json_artifact(args.sealed_study, label="sealed study")
        state = initialize_tuning_state(
            args.lineage_events,
            artifact_root=args.artifact_root,
            sealed_study=sealed_study,
        )
        observed_tip = state["last_terminal_result_binding"]["lineage_tip_sha256"]
        if observed_tip != args.expected_lineage_tip_sha256:
            raise DojoAITuningCliError("stale or forked lineage tip at initialization")
        return _store_status(initialize_state_store(args.state_events, state))

    artifact_receipt: dict[str, Any] | None = None
    if args.command == "reserve-model":
        request_raw = _read_artifact(args.request_artifact, label="model request")
        request_value = _strict_json(request_raw)
        if not isinstance(request_value, dict):
            raise DojoAITuningCliError("model request must be TRAINER_PACKET_V1")
        packet = verify_trainer_packet(request_value)
        expected_raw = canonical_packet_bytes(packet) + b"\n"
        if request_raw != expected_raw:
            raise DojoAITuningCliError("trainer packet request bytes are not canonical")
        invocation_id = _identifier(args.invocation_id, label="invocation_id")
        request_sha = _sha256(request_raw)
        expected_tip = _lower_sha(
            args.expected_tip_event_sha256,
            label="expected_tip_event_sha256",
        )
        expected_parent = _lower_sha(
            args.expected_parent_state_sha256,
            label="expected_parent_state_sha256",
        )
        event_at_utc = _utc(args.event_at_utc, label="event_at_utc")
        before_store = verify_state_store(args.state_events)
        cas_mode = _reservation_cas_mode(
            before_store,
            expected_tip_event_sha256=expected_tip,
            expected_parent_state_sha256=expected_parent,
            invocation_id=invocation_id,
            request_sha256=request_sha,
            event_at_utc=event_at_utc,
        )
        lineage = verify_registry(args.lineage_events, artifact_root=args.artifact_root)
        _verify_packet_source_bindings(
            packet,
            state=before_store["latest_state"],
            lineage=lineage,
            expected_tuning_state_sha256=expected_parent,
            cas_mode=cas_mode,
        )
        artifact_receipt = _persist_raw_request(
            request_raw,
            artifact_root=args.artifact_root,
            invocation_id=invocation_id,
        )
        artifact_receipt["packet_sha256"] = packet["packet_sha256"]
        artifact_receipt["reservation_cas_mode"] = cas_mode
        child = reserve_model_invocation(
            before_store["latest_state"],
            lineage_events_dir=args.lineage_events,
            artifact_root=args.artifact_root,
            expected_parent_state_sha256=expected_parent,
            invocation_id=invocation_id,
            request_sha256=request_sha,
            event_at_utc=event_at_utc,
        )
        snapshot = append_state_transition(
            args.state_events,
            child,
            expected_tip_event_sha256=expected_tip,
            expected_parent_state_sha256=expected_parent,
        )
    elif args.command == "record-response":
        response_raw, artifact_receipt = _persist_raw_response(
            args.response_input,
            args.response_artifact,
            artifact_root=args.artifact_root,
        )
        try:
            submissions = _strict_json(response_raw)
            artifact_receipt["strict_json"] = True
        except DojoAITuningCliError:
            # Invalid model output is still a paid/consumed proposal opportunity.
            # The exact raw bytes remain durable and their digest is bound below.
            submissions = None
            artifact_receipt["strict_json"] = False
        snapshot = _transition(
            args,
            lambda state: record_model_response(
                state,
                expected_parent_state_sha256=args.expected_parent_state_sha256,
                invocation_id=args.invocation_id,
                response_sha256=artifact_receipt["sha256"],
                submissions=submissions,
                event_at_utc=args.event_at_utc,
            ),
        )
    elif args.command == "reserve-run":
        sealed_study = _read_json_artifact(args.sealed_study, label="sealed study")
        snapshot = _transition(
            args,
            lambda state: reserve_run_dispatch(
                state,
                lineage_events_dir=args.lineage_events,
                artifact_root=args.artifact_root,
                expected_parent_state_sha256=args.expected_parent_state_sha256,
                sealed_study=sealed_study,
                dispatch_id=args.dispatch_id,
                event_at_utc=args.event_at_utc,
            ),
        )
    elif args.command == "mark-dispatched":
        snapshot = _transition(
            args,
            lambda state: mark_run_dispatched(
                state,
                lineage_events_dir=args.lineage_events,
                artifact_root=args.artifact_root,
                expected_parent_state_sha256=args.expected_parent_state_sha256,
                dispatch_id=args.dispatch_id,
                event_at_utc=args.event_at_utc,
            ),
        )
    elif args.command == "bind-result":
        snapshot = _transition(
            args,
            lambda state: bind_terminal_evaluation(
                state,
                lineage_events_dir=args.lineage_events,
                artifact_root=args.artifact_root,
                expected_parent_state_sha256=args.expected_parent_state_sha256,
                event_at_utc=args.event_at_utc,
            ),
        )
    elif args.command == "mark-incomplete":
        snapshot = _transition(
            args,
            lambda state: mark_incomplete_run(
                state,
                expected_parent_state_sha256=args.expected_parent_state_sha256,
                reason_code=args.reason_code,
                event_at_utc=args.event_at_utc,
            ),
        )
    elif args.command == "abandon":
        snapshot = _transition(
            args,
            lambda state: abandon_incomplete_lineage(
                state,
                expected_parent_state_sha256=args.expected_parent_state_sha256,
                review_id=args.review_id,
                rationale=args.rationale,
                event_at_utc=args.event_at_utc,
            ),
        )
    else:  # pragma: no cover - argparse owns the closed command set.
        raise DojoAITuningCliError("unsupported command")
    return _store_status(snapshot, artifact_receipt=artifact_receipt)


def _canonical_line(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _run(args)
    except (
        CandidateLineageError,
        DojoAITrainerPacketError,
        DojoAITuningStateError,
        DojoAITuningCliError,
    ) as exc:
        print(
            _canonical_line(
                {
                    "status": "REJECTED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "live_permission": False,
                    "order_authority": "NONE",
                    "broker_mutation_allowed": False,
                }
            ),
            file=sys.stderr,
        )
        return 2
    except OSError:
        print(
            _canonical_line(
                {
                    "status": "REJECTED",
                    "error_type": "LOCAL_IO_ERROR",
                    "error": "local artifact operation failed",
                    "live_permission": False,
                    "order_authority": "NONE",
                    "broker_mutation_allowed": False,
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(_canonical_line(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
