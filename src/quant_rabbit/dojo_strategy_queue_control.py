"""Durable CAS control plane for the immutable DOJO research queue.

The queue module deliberately exposes pure transitions only.  This module
adds the missing local custody boundary: one OS-locked append-only event chain,
exact parent/tip compare-and-swap, and a byte binding to the TRAIN result that
caused a reservation decision.  It does not execute a replay, call a model,
open a holdout, or grant any trading authority.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_strategy_research_queue import (
    DojoStrategyResearchQueueError,
    build_initial_reservation_state,
    canonical_sha256,
    load_research_queue,
    plan_reservation,
    validate_research_queue,
    validate_reservation_state,
    validate_trigger,
)


STORE_EVENT_CONTRACT: Final = "QR_DOJO_STRATEGY_QUEUE_STORE_EVENT_V1"
STORE_STATUS_CONTRACT: Final = "QR_DOJO_STRATEGY_QUEUE_STORE_STATUS_V1"
SCHEMA_VERSION: Final = 1
MAX_STORE_EVENTS: Final = 64
MAX_EVENT_BYTES: Final = 16 * 1024 * 1024
MAX_RESULT_ARTIFACT_BYTES: Final = 256 * 1024 * 1024
GENESIS_SHA256: Final = "0" * 64

_EVENT_NAME_RE: Final = re.compile(r"[0-9]{6}\.json\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_AUTHORITY: Final = {
    "research_train_only": True,
    "reservation_is_execution_permission": False,
    "proof_eligible": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
    "automatic_deployment_allowed": False,
}
_EVENT_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "event_kind",
        "sequence",
        "previous_event_sha256",
        "parent_state_sha256",
        "queue_artifact_sha256",
        "trigger",
        "result_artifact",
        "decision",
        "state",
        "authority",
        "event_sha256",
    }
)


class DojoStrategyQueueControlError(ValueError):
    """The persistent queue store is altered, forked, stale, or unsafe."""


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
        raise DojoStrategyQueueControlError(
            "value is not strict canonical JSON"
        ) from exc


def _copy(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _sha(value: Any, *, field: str, allow_genesis: bool = False) -> str:
    if (
        not isinstance(value, str)
        or _SHA256_RE.fullmatch(value) is None
        or (not allow_genesis and value == GENESIS_SHA256)
    ):
        raise DojoStrategyQueueControlError(f"{field} must be a valid SHA-256")
    return value


def _strict_json(raw: bytes, *, field: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise DojoStrategyQueueControlError(
            f"{field} contains forbidden non-finite JSON: {token}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DojoStrategyQueueControlError(
                    f"{field} contains duplicate key: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoStrategyQueueControlError(f"{field} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise DojoStrategyQueueControlError(f"{field} must be a JSON object")
    return value


def _open_store_directory(path: Path, *, create: bool) -> int:
    directory = Path(path)
    if create:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        state = directory.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoStrategyQueueControlError("queue store is unavailable") from exc
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise DojoStrategyQueueControlError(
            "queue store must be a real directory"
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(directory, flags)
    except OSError as exc:
        raise DojoStrategyQueueControlError("cannot open queue store") from exc


def _open_lock(directory_fd: int) -> int:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(".queue-control.lock", flags, 0o600, dir_fd=directory_fd)
        state = os.fstat(descriptor)
        if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
            raise DojoStrategyQueueControlError(
                "queue control lock must be a single-link regular file"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor
    except OSError as exc:
        raise DojoStrategyQueueControlError("cannot lock queue store") from exc


def _event_names(directory_fd: int) -> list[str]:
    try:
        names = os.listdir(directory_fd)
    except OSError as exc:
        raise DojoStrategyQueueControlError("cannot list queue store") from exc
    unexpected = [
        name
        for name in names
        if name != ".queue-control.lock" and _EVENT_NAME_RE.fullmatch(name) is None
    ]
    if unexpected:
        raise DojoStrategyQueueControlError(
            "queue store contains unexpected entries"
        )
    return sorted(name for name in names if _EVENT_NAME_RE.fullmatch(name))


def _read_event(directory_fd: int, name: str) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            raw = handle.read(MAX_EVENT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise DojoStrategyQueueControlError(f"cannot read queue event {name}") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(raw) != before.st_size
        or len(raw) > MAX_EVENT_BYTES
    ):
        raise DojoStrategyQueueControlError("queue event changed while reading")
    value = _strict_json(raw, field=f"queue event {name}")
    if raw != _canonical_bytes(value) + b"\n":
        raise DojoStrategyQueueControlError("queue event bytes are not canonical")
    return value


def _write_event(directory_fd: int, name: str, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value) + b"\n"
    if len(raw) > MAX_EVENT_BYTES:
        raise DojoStrategyQueueControlError("queue event exceeds byte bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(directory_fd)
    except FileExistsError as exc:
        raise DojoStrategyQueueControlError(
            "queue event slot already exists; reload the current CAS tip"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _result_artifact_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = _sha(expected_sha256, field="trigger.result_artifact_sha256")
    artifact = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = artifact.stat(follow_symlinks=False)
        descriptor = os.open(artifact, flags)
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            while block := handle.read(1024 * 1024):
                size += len(block)
                if size > MAX_RESULT_ARTIFACT_BYTES:
                    raise DojoStrategyQueueControlError(
                        "TRAIN result artifact exceeds byte bound"
                    )
                digest.update(block)
            after = os.fstat(handle.fileno())
        current = artifact.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoStrategyQueueControlError(
            "cannot read TRAIN result artifact"
        ) from exc
    identities = {
        (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)
        for state in (before, opened, after, current)
    }
    if (
        len(identities) != 1
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or size != before.st_size
        or size <= 0
        or digest.hexdigest() != expected
    ):
        raise DojoStrategyQueueControlError(
            "TRAIN result artifact identity or SHA-256 mismatch"
        )
    return {
        "file_sha256": expected,
        "file_size_bytes": size,
        "path_binding": "CALLER_PATH_READ_ONCE_NOFOLLOW_CONTENT_ADDRESS_ONLY",
    }


def _seal_event(body: Mapping[str, Any]) -> dict[str, Any]:
    copied = _copy(body)
    return {**copied, "event_sha256": canonical_sha256(copied)}


def _verify_event(
    value: Mapping[str, Any],
    *,
    queue: Mapping[str, Any],
    sequence: int,
    previous_event_sha256: str,
    previous_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    event = _copy(value)
    if set(event) != set(_EVENT_KEYS):
        raise DojoStrategyQueueControlError("queue event schema mismatch")
    digest = _sha(event["event_sha256"], field="event_sha256")
    body = {key: item for key, item in event.items() if key != "event_sha256"}
    if digest != canonical_sha256(body):
        raise DojoStrategyQueueControlError("queue event SHA-256 mismatch")
    if (
        event["contract"] != STORE_EVENT_CONTRACT
        or event["schema_version"] != SCHEMA_VERSION
        or event["sequence"] != sequence
        or event["previous_event_sha256"] != previous_event_sha256
        or event["queue_artifact_sha256"] != queue["artifact_sha256"]
        or event["authority"] != _AUTHORITY
    ):
        raise DojoStrategyQueueControlError("queue event identity drifted")
    state = validate_reservation_state(event["state"], queue)
    if sequence == 0:
        if (
            event["event_kind"] != "GENESIS"
            or event["parent_state_sha256"] is not None
            or event["trigger"] is not None
            or event["result_artifact"] is not None
            or event["decision"] is not None
            or state != build_initial_reservation_state(queue)
        ):
            raise DojoStrategyQueueControlError("queue store genesis is invalid")
    else:
        if previous_state is None or event["event_kind"] != "TRANSITION":
            raise DojoStrategyQueueControlError("queue transition has no parent")
        trigger = validate_trigger(event["trigger"])
        receipt = event["result_artifact"]
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != {"file_sha256", "file_size_bytes", "path_binding"}
            or receipt["file_sha256"] != trigger["result_artifact_sha256"]
            or isinstance(receipt["file_size_bytes"], bool)
            or not isinstance(receipt["file_size_bytes"], int)
            or receipt["file_size_bytes"] <= 0
            or receipt["path_binding"]
            != "CALLER_PATH_READ_ONCE_NOFOLLOW_CONTENT_ADDRESS_ONLY"
            or event["parent_state_sha256"] != previous_state["state_sha256"]
        ):
            raise DojoStrategyQueueControlError(
                "queue transition artifact or parent binding is invalid"
            )
        decision = plan_reservation(
            queue=queue,
            trigger=trigger,
            previous_state=previous_state,
        )
        if event["decision"] != decision or state != decision["next_state"]:
            raise DojoStrategyQueueControlError(
                "queue transition differs from canonical recomputation"
            )
    return event


def _verify_store_locked(directory_fd: int, queue: Mapping[str, Any]) -> dict[str, Any]:
    names = _event_names(directory_fd)
    if not names:
        raise DojoStrategyQueueControlError("queue store lacks genesis")
    if len(names) > MAX_STORE_EVENTS:
        raise DojoStrategyQueueControlError("queue store event bound exceeded")
    expected_names = [f"{index:06d}.json" for index in range(len(names))]
    if names != expected_names:
        raise DojoStrategyQueueControlError("queue store has a gap or fork")
    raw_events = [_read_event(directory_fd, name) for name in names]
    if _event_names(directory_fd) != names:
        raise DojoStrategyQueueControlError("queue store changed during verification")
    events: list[dict[str, Any]] = []
    previous_event_sha = GENESIS_SHA256
    previous_state: dict[str, Any] | None = None
    for sequence, raw in enumerate(raw_events):
        event = _verify_event(
            raw,
            queue=queue,
            sequence=sequence,
            previous_event_sha256=previous_event_sha,
            previous_state=previous_state,
        )
        events.append(event)
        previous_event_sha = event["event_sha256"]
        previous_state = event["state"]
    assert previous_state is not None
    return {
        "contract": STORE_STATUS_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "event_count": len(events),
        "latest_sequence": len(events) - 1,
        "latest_event_sha256": previous_event_sha,
        "latest_event": _copy(events[-1]),
        "latest_state": _copy(previous_state),
        "queue_artifact_sha256": queue["artifact_sha256"],
        "cas_ready": True,
        "authority": _copy(_AUTHORITY),
    }


def initialize_queue_store(
    events_dir: Path, queue: Mapping[str, Any]
) -> dict[str, Any]:
    """Create one immutable genesis under the store-wide OS lock."""

    validated_queue = validate_research_queue(queue)
    directory_fd = _open_store_directory(Path(events_dir), create=True)
    lock_fd: int | None = None
    try:
        lock_fd = _open_lock(directory_fd)
        if _event_names(directory_fd):
            raise DojoStrategyQueueControlError("queue store is already initialized")
        state = build_initial_reservation_state(validated_queue)
        event = _seal_event(
            {
                "contract": STORE_EVENT_CONTRACT,
                "schema_version": SCHEMA_VERSION,
                "event_kind": "GENESIS",
                "sequence": 0,
                "previous_event_sha256": GENESIS_SHA256,
                "parent_state_sha256": None,
                "queue_artifact_sha256": validated_queue["artifact_sha256"],
                "trigger": None,
                "result_artifact": None,
                "decision": None,
                "state": state,
                "authority": _copy(_AUTHORITY),
            }
        )
        _write_event(directory_fd, "000000.json", event)
        return _verify_store_locked(directory_fd, validated_queue)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(directory_fd)


def verify_queue_store(
    events_dir: Path, queue: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild the full event and state chain while holding a shared boundary."""

    validated_queue = validate_research_queue(queue)
    directory_fd = _open_store_directory(Path(events_dir), create=False)
    lock_fd: int | None = None
    try:
        lock_fd = _open_lock(directory_fd)
        return _verify_store_locked(directory_fd, validated_queue)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(directory_fd)


def commit_queue_transition(
    events_dir: Path,
    *,
    queue: Mapping[str, Any],
    trigger: Mapping[str, Any],
    result_artifact_path: Path,
    expected_tip_event_sha256: str,
    expected_parent_state_sha256: str,
) -> dict[str, Any]:
    """CAS-commit one canonical reservation/completion decision.

    A retry naming the former parent returns the already-written identical
    child.  A competing or semantically different child must reload the store.
    """

    validated_queue = validate_research_queue(queue)
    validated_trigger = validate_trigger(trigger)
    expected_tip = _sha(expected_tip_event_sha256, field="expected tip")
    expected_parent = _sha(expected_parent_state_sha256, field="expected parent")
    artifact = _result_artifact_receipt(
        Path(result_artifact_path), validated_trigger["result_artifact_sha256"]
    )
    directory_fd = _open_store_directory(Path(events_dir), create=False)
    lock_fd: int | None = None
    try:
        lock_fd = _open_lock(directory_fd)
        snapshot = _verify_store_locked(directory_fd, validated_queue)
        latest = snapshot["latest_event"]
        if (
            latest["event_kind"] == "TRANSITION"
            and latest["previous_event_sha256"] == expected_tip
            and latest["parent_state_sha256"] == expected_parent
        ):
            if latest["trigger"] == validated_trigger and latest["result_artifact"] == artifact:
                return {**snapshot, "idempotent_replay": True}
            raise DojoStrategyQueueControlError(
                "CAS parent already has a different committed child"
            )
        if snapshot["latest_event_sha256"] != expected_tip:
            raise DojoStrategyQueueControlError("stale or forked queue event tip")
        parent_state = snapshot["latest_state"]
        if parent_state["state_sha256"] != expected_parent:
            raise DojoStrategyQueueControlError("stale or forked queue parent state")
        decision = plan_reservation(
            queue=validated_queue,
            trigger=validated_trigger,
            previous_state=parent_state,
        )
        sequence = snapshot["event_count"]
        if sequence >= MAX_STORE_EVENTS:
            raise DojoStrategyQueueControlError("queue store event bound exceeded")
        event = _seal_event(
            {
                "contract": STORE_EVENT_CONTRACT,
                "schema_version": SCHEMA_VERSION,
                "event_kind": "TRANSITION",
                "sequence": sequence,
                "previous_event_sha256": expected_tip,
                "parent_state_sha256": expected_parent,
                "queue_artifact_sha256": validated_queue["artifact_sha256"],
                "trigger": validated_trigger,
                "result_artifact": artifact,
                "decision": decision,
                "state": decision["next_state"],
                "authority": _copy(_AUTHORITY),
            }
        )
        _write_event(directory_fd, f"{sequence:06d}.json", event)
        committed = _verify_store_locked(directory_fd, validated_queue)
        return {**committed, "idempotent_replay": False}
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(directory_fd)


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise DojoStrategyQueueControlError(f"cannot read {field}") from exc
    return _strict_json(raw, field=field)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "status"):
        command = sub.add_parser(name)
        command.add_argument("--queue", type=Path, required=True)
        command.add_argument("--events-dir", type=Path, required=True)
    commit = sub.add_parser("commit")
    commit.add_argument("--queue", type=Path, required=True)
    commit.add_argument("--events-dir", type=Path, required=True)
    commit.add_argument("--trigger", type=Path, required=True)
    commit.add_argument("--result-artifact", type=Path, required=True)
    commit.add_argument("--expected-tip", required=True)
    commit.add_argument("--expected-parent", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        queue = load_research_queue(args.queue)
        if args.command == "init":
            result = initialize_queue_store(args.events_dir, queue)
        elif args.command == "status":
            result = verify_queue_store(args.events_dir, queue)
        else:
            result = commit_queue_transition(
                args.events_dir,
                queue=queue,
                trigger=_read_json(args.trigger, field="trigger"),
                result_artifact_path=args.result_artifact,
                expected_tip_event_sha256=args.expected_tip,
                expected_parent_state_sha256=args.expected_parent,
            )
    except (DojoStrategyQueueControlError, DojoStrategyResearchQueueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(_canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DojoStrategyQueueControlError",
    "STORE_EVENT_CONTRACT",
    "STORE_STATUS_CONTRACT",
    "commit_queue_transition",
    "initialize_queue_store",
    "verify_queue_store",
]
