#!/usr/bin/env python3
"""Operate the deterministic, research-only DOJO continuous heartbeat."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import stat
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_continuous_heartbeat import (
    GENESIS_SHA256,
    DojoContinuousHeartbeatError,
    apply_observation,
    build_local_observation,
    build_event,
    canonical_json_bytes,
    complete_reserved_work,
    initial_state,
    plan_heartbeat,
    reserve_decision,
    seal_observation,
    verify_event,
    verify_local_probe_manifest,
    verify_local_run_status,
    verify_observation,
    verify_policy,
)


RESULT_CONTRACT: Final = "QR_DOJO_CONTINUOUS_HEARTBEAT_CLI_RESULT_V1"
MAX_JSON_BYTES: Final = 8 * 1024 * 1024
EXIT_ERROR: Final = 2
EXIT_LEASE_BUSY: Final = 75
_EVENT_NAME_WIDTH: Final = 20


class HeartbeatLeaseBusyError(RuntimeError):
    """Another heartbeat owns the one non-blocking filesystem lease."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("init", "tick", "tick-local", "reserve", "complete", "status"):
        command = commands.add_parser(name)
        command.add_argument("--policy", type=Path, required=True)
        command.add_argument("--state-dir", type=Path, required=True)
        if name in {"init", "tick", "tick-local", "reserve", "complete"}:
            command.add_argument("--event-at-utc", required=True)
        if name == "tick":
            command.add_argument("--observation", type=Path, required=True)
        if name == "tick-local":
            command.add_argument("--probe", type=Path, required=True)
        if name == "reserve":
            command.add_argument("--expected-operation-id", required=True)
        if name == "complete":
            command.add_argument("--operation-id", required=True)
            command.add_argument("--result-sha256", required=True)
            command.add_argument(
                "--outcome", choices=("SUCCESS", "FAILED"), required=True
            )
    observe = commands.add_parser(
        "observe-local",
        help="emit one sealed observation from policy-bound read-only probes",
    )
    observe.add_argument("--policy", type=Path, required=True)
    observe.add_argument("--probe", type=Path, required=True)
    observe.add_argument("--observed-at-utc", required=True)
    return parser


def _load_json(path: Path, *, field: str, require_canonical: bool) -> Any:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise DojoContinuousHeartbeatError(f"cannot open {field}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_JSON_BYTES:
            raise DojoContinuousHeartbeatError(
                f"{field} must be a bounded regular file"
            )
        raw = b""
        while len(raw) <= MAX_JSON_BYTES:
            chunk = os.read(descriptor, min(65536, MAX_JSON_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(descriptor)
    if len(raw) > MAX_JSON_BYTES:
        raise DojoContinuousHeartbeatError(f"{field} exceeds its byte bound")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoContinuousHeartbeatError(f"{field} is not one JSON value") from exc
    if require_canonical and raw != canonical_json_bytes(value) + b"\n":
        raise DojoContinuousHeartbeatError(f"{field} is not canonical JSON")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DojoContinuousHeartbeatError("JSON object contains a duplicate key")
        result[key] = value
    return result


def _regular_directory(path: Path, *, field: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DojoContinuousHeartbeatError(f"{field} does not exist") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise DojoContinuousHeartbeatError(f"{field} must be a real directory")


@contextlib.contextmanager
def _exclusive_lease(state_dir: Path) -> Iterator[None]:
    _regular_directory(state_dir, field="state directory")
    lock_path = state_dir / ".heartbeat.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise DojoContinuousHeartbeatError("cannot open heartbeat lease") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise DojoContinuousHeartbeatError(
                "heartbeat lease must be a regular file"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HeartbeatLeaseBusyError from exc
        yield
    finally:
        os.close(descriptor)


def _event_files(state_dir: Path) -> list[Path]:
    events_dir = state_dir / "events"
    _regular_directory(events_dir, field="event directory")
    files = sorted(events_dir.iterdir())
    for index, path in enumerate(files):
        expected = f"{index:0{_EVENT_NAME_WIDTH}d}.json"
        if path.name != expected or path.is_symlink() or not path.is_file():
            raise DojoContinuousHeartbeatError(
                "event directory is not a contiguous canonical ledger"
            )
    return files


def _marker_count(path: Path, *, field: str) -> int:
    _regular_directory(path, field=field)
    count = 0
    for marker in path.iterdir():
        try:
            metadata = marker.lstat()
        except OSError as exc:
            raise DojoContinuousHeartbeatError(
                f"cannot inspect {field} marker"
            ) from exc
        if marker.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise DojoContinuousHeartbeatError(
                f"{field} may contain only regular marker files"
            )
        count += 1
    return count


def _load_chain(
    state_dir: Path, *, policy: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], int]:
    files = _event_files(state_dir)
    if not files:
        raise DojoContinuousHeartbeatError("heartbeat is not initialized")
    if len(files) > int(policy["max_event_count"]):
        raise DojoContinuousHeartbeatError("event count exceeds sealed policy")
    previous_event_sha = GENESIS_SHA256
    prior_state: Mapping[str, Any] | None = None
    latest: dict[str, Any] | None = None
    for sequence, path in enumerate(files):
        value = _load_json(
            path,
            field=f"event[{sequence}]",
            require_canonical=True,
        )
        latest = verify_event(
            value,
            policy=policy,
            expected_sequence=sequence,
            previous_event_sha256=previous_event_sha,
            prior_state=prior_state,
        )
        previous_event_sha = latest["event_sha256"]
        prior_state = latest["state"]
    assert latest is not None
    return latest["state"], latest, len(files)


def _append_event(
    state_dir: Path, event: Mapping[str, Any], *, policy: Mapping[str, Any]
) -> None:
    sequence = int(event["sequence"])
    if sequence >= int(policy["max_event_count"]):
        raise DojoContinuousHeartbeatError("event registry is full")
    events_dir = state_dir / "events"
    path = events_dir / f"{sequence:0{_EVENT_NAME_WIDTH}d}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise DojoContinuousHeartbeatError("event append target already exists") from exc
    payload = canonical_json_bytes(event) + b"\n"
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(events_dir, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _result(
    *,
    command: str,
    status: str,
    event_appended: bool,
    state: Mapping[str, Any] | None = None,
    event: Mapping[str, Any] | None = None,
    decision: Mapping[str, Any] | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    return {
        "contract": RESULT_CONTRACT,
        "schema_version": 1,
        "command": command,
        "status": status,
        "event_appended": event_appended,
        "event_sequence": None if event is None else event["sequence"],
        "event_sha256": None if event is None else event["event_sha256"],
        "state_revision": None if state is None else state["revision"],
        "state_sha256": None if state is None else state["state_sha256"],
        "decision": decision,
        "error_code": error_code,
        "authority": {
            "broker_mutation_allowed": False,
            "live_permission": False,
            "order_authority": "NONE",
        },
    }


def _initialize(args: argparse.Namespace, policy: Mapping[str, Any]) -> dict[str, Any]:
    state_dir: Path = args.state_dir
    if state_dir.exists():
        _regular_directory(state_dir, field="state directory")
    else:
        state_dir.mkdir(parents=True, mode=0o700)
    events_dir = state_dir / "events"
    try:
        events_dir.mkdir(mode=0o700)
    except FileExistsError:
        _regular_directory(events_dir, field="event directory")
    with _exclusive_lease(state_dir):
        if _event_files(state_dir):
            raise DojoContinuousHeartbeatError("heartbeat is already initialized")
        state = initial_state(
            policy=policy,
            initialized_at_utc=args.event_at_utc,
        )
        event = build_event(
            sequence=0,
            previous_event_sha256=GENESIS_SHA256,
            event_type="INITIALIZED",
            prior_state=None,
            state=state,
            policy=policy,
            event_at_utc=args.event_at_utc,
        )
        _append_event(state_dir, event, policy=policy)
        decision = plan_heartbeat(state, policy=policy)
        return _result(
            command="init",
            status="INITIALIZED",
            event_appended=True,
            state=state,
            event=event,
            decision=decision,
        )


def _tick(args: argparse.Namespace, policy: Mapping[str, Any]) -> dict[str, Any]:
    raw = _load_json(
        args.observation,
        field="observation",
        require_canonical=False,
    )
    if isinstance(raw, Mapping) and "observation_sha256" in raw:
        observation = verify_observation(raw, policy=policy)
    else:
        observation = seal_observation(raw, policy=policy)
    return _tick_with_observation(
        args,
        policy,
        observation=observation,
        command="tick",
    )


def _tick_with_observation(
    args: argparse.Namespace,
    policy: Mapping[str, Any],
    *,
    observation: Mapping[str, Any],
    command: str,
) -> dict[str, Any]:
    with _exclusive_lease(args.state_dir):
        state, prior_event, event_count = _load_chain(args.state_dir, policy=policy)
        next_state, changed = apply_observation(
            state,
            observation,
            policy=policy,
            event_at_utc=args.event_at_utc,
        )
        event = prior_event
        if changed:
            event = build_event(
                sequence=event_count,
                previous_event_sha256=prior_event["event_sha256"],
                event_type="OBSERVATION_CHANGED",
                prior_state=state,
                state=next_state,
                policy=policy,
                event_at_utc=args.event_at_utc,
            )
            _append_event(args.state_dir, event, policy=policy)
        decision = plan_heartbeat(next_state, policy=policy)
        return _result(
            command=command,
            status="UPDATED" if changed else "NO_CHANGE",
            event_appended=changed,
            state=next_state,
            event=event,
            decision=decision,
        )


def _tick_local(
    args: argparse.Namespace, policy: Mapping[str, Any]
) -> dict[str, Any]:
    observation = _observe_local(
        argparse.Namespace(
            probe=args.probe,
            observed_at_utc=args.event_at_utc,
        ),
        policy,
    )
    return _tick_with_observation(
        args,
        policy,
        observation=observation,
        command="tick-local",
    )


def _reserve(args: argparse.Namespace, policy: Mapping[str, Any]) -> dict[str, Any]:
    with _exclusive_lease(args.state_dir):
        state, prior_event, event_count = _load_chain(args.state_dir, policy=policy)
        decision = plan_heartbeat(state, policy=policy)
        active = state["active_lease"]
        if (
            active is not None
            and active["operation_id"] == args.expected_operation_id
        ):
            return _result(
                command="reserve",
                status="ALREADY_RESERVED",
                event_appended=False,
                state=state,
                event=prior_event,
                decision=decision,
            )
        if decision["operation_id"] != args.expected_operation_id:
            raise DojoContinuousHeartbeatError(
                "expected operation does not match the current decision"
            )
        next_state, changed = reserve_decision(
            state,
            decision,
            policy=policy,
            reserved_at_utc=args.event_at_utc,
        )
        event = prior_event
        if changed:
            event = build_event(
                sequence=event_count,
                previous_event_sha256=prior_event["event_sha256"],
                event_type="WORK_RESERVED",
                prior_state=state,
                state=next_state,
                policy=policy,
                event_at_utc=args.event_at_utc,
            )
            _append_event(args.state_dir, event, policy=policy)
        return _result(
            command="reserve",
            status="RESERVED" if changed else "ALREADY_RESERVED",
            event_appended=changed,
            state=next_state,
            event=event,
            decision=plan_heartbeat(next_state, policy=policy),
        )


def _complete(args: argparse.Namespace, policy: Mapping[str, Any]) -> dict[str, Any]:
    with _exclusive_lease(args.state_dir):
        state, prior_event, event_count = _load_chain(args.state_dir, policy=policy)
        next_state = complete_reserved_work(
            state,
            policy=policy,
            operation_id=args.operation_id,
            result_sha256=args.result_sha256,
            outcome=args.outcome,
            completed_at_utc=args.event_at_utc,
        )
        changed = next_state["state_sha256"] != state["state_sha256"]
        event = prior_event
        if changed:
            event = build_event(
                sequence=event_count,
                previous_event_sha256=prior_event["event_sha256"],
                event_type=(
                    "WORK_COMPLETED" if args.outcome == "SUCCESS" else "WORK_FAILED"
                ),
                prior_state=state,
                state=next_state,
                policy=policy,
                event_at_utc=args.event_at_utc,
            )
            _append_event(args.state_dir, event, policy=policy)
        return _result(
            command="complete",
            status="COMPLETED" if changed else "ALREADY_COMPLETED",
            event_appended=changed,
            state=next_state,
            event=event,
            decision=plan_heartbeat(next_state, policy=policy),
        )


def _status(args: argparse.Namespace, policy: Mapping[str, Any]) -> dict[str, Any]:
    with _exclusive_lease(args.state_dir):
        state, event, _ = _load_chain(args.state_dir, policy=policy)
        return _result(
            command="status",
            status="OK",
            event_appended=False,
            state=state,
            event=event,
            decision=plan_heartbeat(state, policy=policy),
        )


def _observe_local(
    args: argparse.Namespace, policy: Mapping[str, Any]
) -> dict[str, Any]:
    raw_probe = _load_json(args.probe, field="local probe", require_canonical=False)
    probe = verify_local_probe_manifest(raw_probe, policy=policy)
    raw_status = _load_json(
        Path(probe["run_status_path"]),
        field="local run status",
        require_canonical=False,
    )
    status = verify_local_run_status(raw_status)
    storage_path = Path(probe["storage_path"])
    _regular_directory(storage_path, field="probe storage path")
    try:
        storage = os.statvfs(storage_path)
    except OSError as exc:
        raise DojoContinuousHeartbeatError("cannot inspect probe storage") from exc
    return build_local_observation(
        run_status=status,
        local_probe=probe,
        policy=policy,
        observed_at_utc=args.observed_at_utc,
        active_trainer_count=_marker_count(
            Path(probe["active_trainer_marker_directory"]),
            field="active trainer marker directory",
        ),
        remote_unverified_generation_count=_marker_count(
            Path(probe["remote_unverified_generation_marker_directory"]),
            field="remote unverified generation marker directory",
        ),
        compression_upload_active_count=_marker_count(
            Path(probe["compression_upload_marker_directory"]),
            field="compression/upload marker directory",
        ),
        free_bytes=storage.f_bavail * storage.f_frsize,
    )


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    raw_policy = _load_json(args.policy, field="policy", require_canonical=False)
    policy = verify_policy(raw_policy)
    if args.command == "observe-local":
        return _observe_local(args, policy)
    if args.command == "init":
        return _initialize(args, policy)
    if args.command == "tick":
        return _tick(args, policy)
    if args.command == "tick-local":
        return _tick_local(args, policy)
    if args.command == "reserve":
        return _reserve(args, policy)
    if args.command == "complete":
        return _complete(args, policy)
    if args.command == "status":
        return _status(args, policy)
    raise DojoContinuousHeartbeatError("command is unsupported")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = _dispatch(args)
    except HeartbeatLeaseBusyError:
        output = _result(
            command=args.command,
            status="LEASE_BUSY",
            event_appended=False,
            error_code="SINGLE_HEARTBEAT_LEASE_BUSY",
        )
        print(json.dumps(output, sort_keys=True, separators=(",", ":")))
        return EXIT_LEASE_BUSY
    except (DojoContinuousHeartbeatError, OSError) as exc:
        output = _result(
            command=args.command,
            status="ERROR",
            event_appended=False,
            error_code=type(exc).__name__,
        )
        print(json.dumps(output, sort_keys=True, separators=(",", ":")))
        return EXIT_ERROR
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
