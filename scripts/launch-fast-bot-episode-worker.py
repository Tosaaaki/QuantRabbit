#!/usr/bin/env python3
"""Publish sealed episode handoffs and launch one detached spool worker."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import time_ns
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot import load_fast_bot_episode_handoff  # noqa: E402
from quant_rabbit.fast_bot_episode import run_fast_bot_episode_shadow  # noqa: E402


MAX_SPOOL_HANDOFFS = 64
MAX_SPOOL_BYTES = 512 * 1024 * 1024
MAX_HANDOFF_BYTES = 64 * 1024 * 1024
MAX_WORKER_LOG_BYTES = 16 * 1024 * 1024
MAX_OWNER_BYTES = 16 * 1024
SPOOL_OWNER_CONTRACT = "QR_FAST_BOT_EPISODE_SPOOL_OWNER_V1"
SUCCESS_STATUSES = {"UPDATED", "NO_NEW_EVENT"}
_OUTER_TEMP = re.compile(
    r"^\.handoff-(?P<pid>[1-9][0-9]*)-"
    r"(?P<owner>[0-9a-f]{64})-[A-Za-z0-9]+\.tmp$"
)
_ATOMIC_TEMP = re.compile(r"^\..*\.(?P<pid>[1-9][0-9]*)\.tmp$")
_OWNER_TEMP = re.compile(
    r"^\.\.owner\.json-(?P<owner>[0-9a-f]{64})-"
    r"(?P<token>[0-9a-f]{16})\.(?P<pid>[1-9][0-9]*)\.tmp$"
)
_FINAL_HANDOFF = re.compile(
    r"^handoff-[0-9]{1,24}-(?P<owner>[0-9a-f]{64})-"
    r"[1-9][0-9]*-[0-9a-f]{16}\.json$"
)


class SpoolFullError(RuntimeError):
    pass


class MetadataLockBusyError(RuntimeError):
    pass


def _aware_cycle(value: object) -> datetime:
    text = str(value or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("episode handoff cycle clock must be aware")
    return parsed.astimezone(timezone.utc)


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _directory_fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_unlink(path: Path, *, spool: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _directory_fsync(spool)


def _open_bounded_worker_log(path: Path) -> int:
    """Open a regular append-only log without following or blocking on links."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("episode worker log directory is invalid")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)

    def open_checked() -> tuple[int, os.stat_result]:
        descriptor = os.open(path, flags, 0o600)
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            os.close(descriptor)
            raise ValueError("episode worker log must be a regular file")
        return descriptor, file_stat

    descriptor, file_stat = open_checked()
    if file_stat.st_size < MAX_WORKER_LOG_BYTES:
        return descriptor
    os.close(descriptor)
    rotated = path.with_name(f"{path.name}.1")
    if rotated.exists() or rotated.is_symlink():
        rotated_stat = rotated.lstat()
        if not stat.S_ISREG(rotated_stat.st_mode):
            raise ValueError("episode worker rotated log must be regular")
        rotated.unlink()
    os.replace(path, rotated)
    _directory_fsync(path.parent)
    descriptor, _ = open_checked()
    return descriptor


@contextmanager
def _metadata_lock(spool: Path):
    """Serialize spool accounting, publication, enumeration, and deletion."""

    spool.mkdir(parents=True, exist_ok=True)
    lock_path = spool / ".metadata.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode):
            raise ValueError("episode spool metadata lock must be regular")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MetadataLockBusyError("episode spool metadata lock is busy") from error
        yield
    finally:
        os.close(descriptor)


def _regular_file_stat(path: Path) -> os.stat_result:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"spool entry must be a regular file: {path.name}")
    return value


def _owner_pid(pattern: re.Pattern[str], name: str) -> int | None:
    matched = pattern.match(name)
    return int(matched.group("pid")) if matched is not None else None


def _handoff_owner(pattern: re.Pattern[str], name: str) -> str | None:
    matched = pattern.match(name)
    return str(matched.group("owner")) if matched is not None else None


def _file_fingerprint(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _cleanup_stale_atomic_temps_unlocked(spool: Path) -> int:
    """Remove only abandoned private atomic-writer temps, never handoffs."""

    removed = 0
    for path in spool.iterdir():
        owner_pid = _owner_pid(_ATOMIC_TEMP, path.name)
        if owner_pid is None or _pid_is_running(owner_pid):
            continue
        try:
            file_stat = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(file_stat.st_mode):
            continue
        _durable_unlink(path, spool=spool)
        removed += 1
    return removed


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _spool_owner_contract(
    *,
    spool: Path,
    output: Path,
    ledger: Path,
    source_archive: Path,
) -> dict[str, Any]:
    destinations = {
        "spool_path": str(spool.resolve(strict=False)),
        "output_path": str(output.resolve(strict=False)),
        "ledger_path": str(ledger.resolve(strict=False)),
        "source_archive_path": str(source_archive.resolve(strict=False)),
    }
    owner_id = hashlib.sha256(_canonical_json_bytes(destinations)).hexdigest()
    body = {
        "contract": SPOOL_OWNER_CONTRACT,
        "schema_version": 1,
        "owner_id": owner_id,
        **destinations,
        "diagnostic_only": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return {
        **body,
        "contract_sha256": hashlib.sha256(
            _canonical_json_bytes(body)
        ).hexdigest(),
    }


def _read_spool_owner(path: Path) -> dict[str, Any] | None:
    try:
        initial = path.lstat()
    except FileNotFoundError:
        return None
    if (
        not stat.S_ISREG(initial.st_mode)
        or initial.st_size <= 0
        or initial.st_size > MAX_OWNER_BYTES
    ):
        raise ValueError("episode spool owner file is invalid")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or _file_fingerprint(before) != _file_fingerprint(initial)
        ):
            raise ValueError("episode spool owner changed before read")
        chunks: list[bytes] = []
        remaining = MAX_OWNER_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(16 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(raw) > MAX_OWNER_BYTES
        or len(raw) != before.st_size
        or _file_fingerprint(before) != _file_fingerprint(after)
    ):
        raise ValueError("episode spool owner changed during read")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("episode spool owner JSON is invalid") from error
    expected_fields = {
        "contract",
        "schema_version",
        "owner_id",
        "spool_path",
        "output_path",
        "ledger_path",
        "source_archive_path",
        "diagnostic_only",
        "shadow_only",
        "order_authority",
        "live_permission",
        "broker_mutation_allowed",
        "contract_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("episode spool owner shape is invalid")
    body = {
        key: item for key, item in value.items() if key != "contract_sha256"
    }
    if (
        raw != _canonical_json_bytes(value) + b"\n"
        or value.get("contract") != SPOOL_OWNER_CONTRACT
        or isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != 1
        or value.get("diagnostic_only") is not True
        or value.get("shadow_only") is not True
        or value.get("order_authority") != "NONE"
        or value.get("live_permission") is not False
        or value.get("broker_mutation_allowed") is not False
        or value.get("contract_sha256")
        != hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    ):
        raise ValueError("episode spool owner seal is invalid")
    return value


def _write_spool_owner_atomic(spool: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_json_bytes(dict(value)) + b"\n"
    if len(raw) > MAX_OWNER_BYTES:
        raise ValueError("episode spool owner exceeds its byte cap")
    owner_id = str(value.get("owner_id") or "")
    if re.fullmatch(r"[0-9a-f]{64}", owner_id) is None:
        raise ValueError("episode spool owner id is invalid")
    temp_name = (
        f"..owner.json-{owner_id}-{secrets.token_hex(8)}."
        f"{os.getpid()}.tmp"
    )
    directory_fd = os.open(
        spool,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        descriptor = os.open(
            temp_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temp_name,
            ".owner.json",
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temp_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def _recover_spool_owner_temp_unlocked(
    spool: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Adopt only an exact, dead-writer owner temp and touch nothing else."""

    entries = [
        path
        for path in spool.iterdir()
        if path.name not in {".metadata.lock", ".worker.lock"}
    ]
    if not entries:
        return None
    owner_id = str(expected.get("owner_id") or "")
    candidates: list[Path] = []
    invalid_candidates: list[Path] = []
    # First establish that every payload is a dead private writer for this
    # exact destination owner.  Until then, recovery must be non-mutating.
    for path in entries:
        matched = _OWNER_TEMP.fullmatch(path.name)
        if (
            matched is None
            or matched.group("owner") != owner_id
            or _pid_is_running(int(matched.group("pid")))
        ):
            return None
        try:
            file_stat = path.lstat()
        except FileNotFoundError:
            return None
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size > MAX_OWNER_BYTES
        ):
            return None
    for path in entries:
        try:
            candidate = _read_spool_owner(path)
        except (OSError, ValueError):
            candidate = None
        if candidate == expected:
            candidates.append(path)
        else:
            invalid_candidates.append(path)
    # A crash before the private temp's write/fsync completes leaves no
    # authoritative bytes.  Once exact owner and dead PID are established,
    # those torn private temps are safe to remove and must not self-brick the
    # empty spool forever.
    for path in invalid_candidates:
        _durable_unlink(path, spool=spool)
    if not candidates:
        return None
    selected = sorted(candidates, key=lambda path: path.name)[0]
    os.replace(selected, spool / ".owner.json")
    _directory_fsync(spool)
    return _read_spool_owner(spool / ".owner.json")


def _ensure_spool_owner(
    *,
    spool: Path,
    output: Path,
    ledger: Path,
    source_archive: Path,
) -> str:
    spool.mkdir(parents=True, exist_ok=True)
    if spool.is_symlink() or not spool.is_dir():
        raise ValueError("episode spool root is invalid")
    expected = _spool_owner_contract(
        spool=spool,
        output=output,
        ledger=ledger,
        source_archive=source_archive,
    )
    with _metadata_lock(spool):
        owner_path = spool / ".owner.json"
        owner = _read_spool_owner(owner_path)
        if owner is None:
            owner = _recover_spool_owner_temp_unlocked(spool, expected)
        if owner is None:
            non_owner_entries = [
                path.name
                for path in spool.iterdir()
                if path.name not in {".metadata.lock", ".worker.lock"}
            ]
            if non_owner_entries:
                raise ValueError(
                    "episode spool owner is missing while payload exists"
                )
            _write_spool_owner_atomic(spool, expected)
            owner = _read_spool_owner(owner_path)
        if owner != expected:
            raise ValueError("episode spool destination owner mismatch")
        # Cleanup is owner-scoped.  A caller with mismatched destinations must
        # have no write or deletion side effect in a foreign spool.
        _cleanup_stale_atomic_temps_unlocked(spool)
    return str(expected["owner_id"])


def cleanup_stale_temps(spool: Path) -> int:
    with _metadata_lock(spool):
        return _cleanup_stale_atomic_temps_unlocked(spool)


def _stale_outer_snapshots_unlocked(
    spool: Path,
    *,
    owner_id: str,
) -> list[tuple[Path, tuple[int, int, int, int]]]:
    snapshots: list[tuple[Path, tuple[int, int, int, int]]] = []
    for path in spool.iterdir():
        owner_pid = _owner_pid(_OUTER_TEMP, path.name)
        if owner_pid is None or _pid_is_running(owner_pid):
            continue
        if _handoff_owner(_OUTER_TEMP, path.name) != owner_id:
            raise ValueError("episode outer handoff owner mismatch")
        try:
            file_stat = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(
                f"episode outer handoff must be regular: {path.name}"
            )
        snapshots.append((path, _file_fingerprint(file_stat)))
    return snapshots


def _same_regular_fingerprint(
    path: Path,
    expected: tuple[int, int, int, int],
) -> bool:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(current.st_mode) and _file_fingerprint(current) == expected


def recover_stale_outer_handoffs(
    spool: Path,
    *,
    owner_id: str,
) -> tuple[int, int]:
    """Validate dead-primary outer temps, then promote or durably discard."""

    with _metadata_lock(spool):
        _cleanup_stale_atomic_temps_unlocked(spool)
        snapshots = _stale_outer_snapshots_unlocked(
            spool,
            owner_id=owner_id,
        )
    promoted = 0
    discarded = 0
    for path, fingerprint in snapshots:
        try:
            load_fast_bot_episode_handoff(path)
        except (OSError, TypeError, ValueError):
            with _metadata_lock(spool):
                if _same_regular_fingerprint(path, fingerprint):
                    _durable_unlink(path, spool=spool)
                    discarded += 1
            continue
        with _metadata_lock(spool):
            if not _same_regular_fingerprint(path, fingerprint):
                continue
            # Promotion is a no-growth rename of bytes already included in the
            # spool.  It must remain available specifically to drain an
            # over-cap spool after a crash or older concurrent producer.
            final_path = spool / (
                f"handoff-{time_ns()}-{owner_id}-"
                f"{os.getpid()}-{secrets.token_hex(8)}.json"
            )
            os.replace(path, final_path)
            _directory_fsync(spool)
            promoted += 1
            print(
                f"fast-bot episode spool recovered {path.name} as {final_path.name}",
                file=sys.stderr,
            )
    return promoted, discarded


def _final_handoffs_unlocked(
    spool: Path,
    *,
    owner_id: str,
) -> list[Path]:
    paths: list[Path] = []
    for path in spool.glob("handoff-*.json"):
        if _handoff_owner(_FINAL_HANDOFF, path.name) != owner_id:
            raise ValueError(f"spool handoff owner is invalid: {path.name}")
        file_stat = _regular_file_stat(path)
        if file_stat.st_size <= 0 or file_stat.st_size > MAX_HANDOFF_BYTES:
            raise ValueError(f"spool handoff size is invalid: {path.name}")
        paths.append(path)
    return paths


def _spool_usage_unlocked(
    spool: Path,
    *,
    owner_id: str,
) -> tuple[int, int]:
    paths = _final_handoffs_unlocked(spool, owner_id=owner_id)
    return len(paths), sum(_regular_file_stat(path).st_size for path in paths)


def _outer_handoff_usage_unlocked(
    spool: Path,
    *,
    owner_id: str,
) -> tuple[int, int]:
    count = 0
    size = 0
    for path in spool.iterdir():
        if _owner_pid(_OUTER_TEMP, path.name) is None:
            continue
        if _handoff_owner(_OUTER_TEMP, path.name) != owner_id:
            raise ValueError(f"outer handoff owner is invalid: {path.name}")
        file_stat = _regular_file_stat(path)
        if file_stat.st_size < 0 or file_stat.st_size > MAX_HANDOFF_BYTES:
            raise ValueError(f"outer handoff size is invalid: {path.name}")
        count += 1
        # An outer file may still be the zero-byte reservation while its
        # producer builds a bounded inner atomic temp.  Charge the full future
        # size so concurrent reservations cannot overcommit the byte cap.
        size += MAX_HANDOFF_BYTES
    return count, size


def _inventory_usage_unlocked(
    spool: Path,
    *,
    owner_id: str,
) -> tuple[int, int]:
    final_count, final_size = _spool_usage_unlocked(
        spool,
        owner_id=owner_id,
    )
    outer_count, outer_size = _outer_handoff_usage_unlocked(
        spool,
        owner_id=owner_id,
    )
    return final_count + outer_count, final_size + outer_size


def spool_usage(spool: Path, *, owner_id: str) -> tuple[int, int]:
    with _metadata_lock(spool):
        return _spool_usage_unlocked(spool, owner_id=owner_id)


def spool_accepts_handoff(
    spool: Path,
    *,
    output: Path,
    ledger: Path,
    source_archive: Path,
) -> tuple[bool, str]:
    owner_id = _ensure_spool_owner(
        spool=spool,
        output=output,
        ledger=ledger,
        source_archive=source_archive,
    )
    with _metadata_lock(spool):
        _cleanup_stale_atomic_temps_unlocked(spool)
        count, size = _inventory_usage_unlocked(
            spool,
            owner_id=owner_id,
        )
        available = (
            count < MAX_SPOOL_HANDOFFS
            and size + MAX_HANDOFF_BYTES <= MAX_SPOOL_BYTES
        )
    return available, owner_id


def reserve_handoff(
    spool: Path,
    *,
    output: Path,
    ledger: Path,
    source_archive: Path,
    producer_pid: int,
) -> Path:
    """Atomically reserve bounded spool capacity for one primary handoff."""

    if (
        isinstance(producer_pid, bool)
        or producer_pid <= 0
        or not _pid_is_running(producer_pid)
    ):
        raise ValueError("episode handoff producer pid is invalid")
    owner_id = _ensure_spool_owner(
        spool=spool,
        output=output,
        ledger=ledger,
        source_archive=source_archive,
    )
    resolved_spool = spool.resolve()
    with _metadata_lock(resolved_spool):
        _cleanup_stale_atomic_temps_unlocked(resolved_spool)
        count, size = _inventory_usage_unlocked(
            resolved_spool,
            owner_id=owner_id,
        )
        if (
            count >= MAX_SPOOL_HANDOFFS
            or size + MAX_HANDOFF_BYTES > MAX_SPOOL_BYTES
        ):
            raise SpoolFullError("episode handoff spool capacity reached")
        path = resolved_spool / (
            f".handoff-{producer_pid}-{owner_id}-{secrets.token_hex(8)}.tmp"
        )
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.close(descriptor)
        _directory_fsync(resolved_spool)
        return path


def publish_handoff(
    spool: Path,
    temp_path: Path,
    *,
    output: Path,
    ledger: Path,
    source_archive: Path,
) -> Path:
    owner_id = _ensure_spool_owner(
        spool=spool,
        output=output,
        ledger=ledger,
        source_archive=source_archive,
    )
    spool.mkdir(parents=True, exist_ok=True)
    resolved_spool = spool.resolve()
    if temp_path.parent.resolve() != resolved_spool:
        raise ValueError("episode handoff temp must be inside its spool")
    if (
        _owner_pid(_OUTER_TEMP, temp_path.name) is None
        or _handoff_owner(_OUTER_TEMP, temp_path.name) != owner_id
    ):
        raise ValueError("episode handoff temp name is invalid")
    temp_stat = _regular_file_stat(temp_path)
    if temp_stat.st_size <= 0 or temp_stat.st_size > MAX_HANDOFF_BYTES:
        raise ValueError("episode handoff temp size is invalid")
    temp_fingerprint = _file_fingerprint(temp_stat)
    # Sealed-contract validation is intentionally deferred to the worker.  The
    # Guardian's synchronous publication path performs only bounded metadata
    # accounting and one atomic rename.
    with _metadata_lock(resolved_spool):
        _cleanup_stale_atomic_temps_unlocked(resolved_spool)
        if not _same_regular_fingerprint(temp_path, temp_fingerprint):
            raise ValueError("episode handoff temp changed before publication")
        count, size = _inventory_usage_unlocked(
            resolved_spool,
            owner_id=owner_id,
        )
        if count > MAX_SPOOL_HANDOFFS or size > MAX_SPOOL_BYTES:
            raise SpoolFullError("episode handoff spool capacity reached")
        final_path = resolved_spool / (
            f"handoff-{time_ns()}-{owner_id}-"
            f"{os.getpid()}-{secrets.token_hex(8)}.json"
        )
        os.replace(temp_path, final_path)
        _directory_fsync(resolved_spool)
        return final_path


def _ordered_handoffs(
    paths: list[Path],
) -> list[tuple[datetime, str, Path]]:
    # Parse one bounded handoff at a time.  Keeping all decoded JSON objects
    # would multiply the 512 MiB spool cap by Python object overhead.
    ordered: list[tuple[datetime, str, Path]] = []
    for path in paths:
        handoff = load_fast_bot_episode_handoff(path)
        cycle = _aware_cycle(handoff["cycle_generated_at_utc"])
        seal = str(handoff.get("contract_sha256") or "")
        ordered.append((cycle, seal, path))
    ordered.sort(key=lambda item: (item[0], item[1], item[2].name))
    return ordered


def _run_episode_truth_cycle(
    *,
    handoffs: list[Mapping[str, Any]],
    episode_ledger_path: Path,
    source_archive_dir: Path,
) -> dict[str, Any]:
    """Load the truth adapter lazily after the detached-worker checks."""

    resolver_path = ROOT / "scripts" / "resolve-fast-bot-episode-outcomes.py"
    spec = importlib.util.spec_from_file_location(
        "quant_rabbit_fast_bot_episode_outcome_resolver",
        resolver_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("fast-bot episode outcome resolver is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_episode_outcome_resolution(
        handoffs=tuple(handoffs),
        episode_ledger_path=episode_ledger_path,
        source_archive_dir=source_archive_dir,
    )
    if not isinstance(result, Mapping):
        raise ValueError("episode truth cycle returned an invalid result")
    return dict(result)


def run_worker(
    *,
    spool: Path,
    output: Path,
    ledger: Path,
    source_archive: Path,
    outcome_enabled: bool = False,
) -> int:
    if os.environ.get("QR_LIVE_ENABLED", "0") != "0":
        print("fast-bot episode spool worker requires QR_LIVE_ENABLED=0", file=sys.stderr)
        return 2
    if os.environ.get("QR_AUTOTRADE_LOCK_HELD", "0") != "0":
        print("fast-bot episode spool worker refuses the shared live lock", file=sys.stderr)
        return 2
    if os.environ.get("QR_AUTOTRADE_LOCK_OWNER_TOKEN"):
        print("fast-bot episode spool worker refuses a live-lock owner token", file=sys.stderr)
        return 2
    owner_id = _ensure_spool_owner(
        spool=spool,
        output=output,
        ledger=ledger,
        source_archive=source_archive,
    )
    spool.mkdir(parents=True, exist_ok=True)
    lock_path = spool / ".worker.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode):
            raise ValueError("episode spool worker lock must be regular")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("fast-bot episode spool worker already active", file=sys.stderr)
            return 75
        try:
            recover_stale_outer_handoffs(spool, owner_id=owner_id)
            with _metadata_lock(spool):
                _cleanup_stale_atomic_temps_unlocked(spool)
                handoff_paths = _final_handoffs_unlocked(
                    spool,
                    owner_id=owner_id,
                )
            ordered = _ordered_handoffs(handoff_paths)
        except (OSError, TypeError, ValueError) as error:
            print(
                f"fast-bot episode spool retained invalid handoff: {type(error).__name__}",
                file=sys.stderr,
            )
            return 1
        pending_v2: list[tuple[Path, str, str]] = []
        pending_v2_handoffs: list[Mapping[str, Any]] = []
        for cycle, ordered_seal, path in ordered:
            try:
                handoff = load_fast_bot_episode_handoff(path)
                if (
                    _aware_cycle(handoff["cycle_generated_at_utc"]) != cycle
                    or str(handoff.get("contract_sha256") or "") != ordered_seal
                ):
                    raise ValueError("episode handoff changed after ordering")
            except (OSError, TypeError, ValueError) as error:
                print(
                    f"fast-bot episode spool retained {path.name}: "
                    f"{type(error).__name__}",
                    file=sys.stderr,
                )
                return 1
            processed_at = datetime.now(timezone.utc)
            delay_seconds = max(
                0.0,
                (processed_at - cycle).total_seconds(),
            )
            print(
                "fast-bot episode spool processing "
                f"{path.name}: spool_delay_seconds={delay_seconds:.6f}",
                file=sys.stderr,
            )
            status = ""
            for recovery_attempt in range(2):
                try:
                    result = run_fast_bot_episode_shadow(
                        regime_contract=handoff["regime_contract"],
                        fast_pair_charts=handoff["fast_pair_charts"],
                        slow_pair_charts=handoff["slow_pair_charts"],
                        output_path=output,
                        ledger_path=ledger,
                        source_archive_dir=source_archive,
                        # The sealed cycle is the knowledge-acquisition clock
                        # and binds every event to that information set.
                        now_utc=cycle,
                        processed_at_utc=processed_at,
                    )
                except Exception as error:  # preserve the exact retry input
                    print(
                        f"fast-bot episode spool retained {path.name}: "
                        f"{type(error).__name__}",
                        file=sys.stderr,
                    )
                    return 1
                status = (
                    str(result.get("status") or "")
                    if isinstance(result, Mapping)
                    else ""
                )
                if status == "RECOVERED_PENDING_BATCH" and recovery_attempt == 0:
                    print(
                        "fast-bot episode spool recovered pending batch; "
                        f"replaying {path.name} before deletion",
                        file=sys.stderr,
                    )
                    continue
                break
            if status not in SUCCESS_STATUSES:
                print(
                    f"fast-bot episode spool retained {path.name}: status={status or 'INVALID'}",
                    file=sys.stderr,
                )
                return 75 if status == "LOCK_BUSY" else 1
            if handoff.get("schema_version") == 2:
                # V2 contains the same-cycle quote/geometry shadow.  Keep its
                # durable spool bytes until the truth cycle confirms that all
                # matching CONFIRMED episodes were projected idempotently into
                # the vehicle ledger.
                pending_v2.append((path, ordered_seal, status))
                pending_v2_handoffs.append(handoff)
            else:
                # V1 has no causal quote binding and remains drain-only.  It is
                # never backfilled from a later market snapshot.
                with _metadata_lock(spool):
                    _durable_unlink(path, spool=spool)
                print(
                    f"fast-bot episode spool consumed {path.name}: status={status}",
                    file=sys.stderr,
                )

        if pending_v2_handoffs or outcome_enabled:
            try:
                truth_result = _run_episode_truth_cycle(
                    handoffs=pending_v2_handoffs,
                    episode_ledger_path=ledger,
                    source_archive_dir=source_archive,
                )
            except Exception as error:
                print(
                    "fast-bot episode truth cycle retained V2 handoffs: "
                    f"{type(error).__name__}",
                    file=sys.stderr,
                )
                return 1
            truth_status = str(truth_result.get("status") or "INVALID")
            projection_verified = (
                truth_result.get("vehicle_projection_status") == "VERIFIED"
            )
            unscored_count = int(
                truth_result.get("handoff_confirmed_unscored_count") or 0
            )
            if unscored_count:
                print(
                    "fast-bot episode truth left confirmed events explicitly "
                    f"unscored: count={unscored_count} reasons="
                    f"{json.dumps(truth_result.get('handoff_confirmed_unscored_reason_counts') or {}, sort_keys=True)}",
                    file=sys.stderr,
                )
            if pending_v2 and projection_verified:
                # Vehicle durability is independent of later outcome fetches.
                # Once verified, an outcome-side conflict must not resurrect
                # or replay the same handoff on the next Guardian cycle.
                for path, expected_seal, episode_status in pending_v2:
                    try:
                        current = load_fast_bot_episode_handoff(path)
                    except (OSError, TypeError, ValueError) as error:
                        print(
                            f"fast-bot episode spool retained {path.name} after "
                            f"vehicle projection: {type(error).__name__}",
                            file=sys.stderr,
                        )
                        return 1
                    if str(current.get("contract_sha256") or "") != expected_seal:
                        print(
                            f"fast-bot episode spool retained {path.name}: "
                            "handoff changed after vehicle projection",
                            file=sys.stderr,
                        )
                        return 1
                    with _metadata_lock(spool):
                        _durable_unlink(path, spool=spool)
                    print(
                        "fast-bot episode spool consumed "
                        f"{path.name}: status={episode_status} "
                        f"vehicle_projection_status=VERIFIED",
                        file=sys.stderr,
                    )
                pending_v2.clear()
            if pending_v2:
                truth_error = str(truth_result.get("error") or "").replace(
                    "\n", " "
                )[:320]
                print(
                    "fast-bot episode truth cycle retained V2 handoffs: "
                    f"status={truth_status} vehicle_projection_status="
                    f"{truth_result.get('vehicle_projection_status') or 'INVALID'}"
                    f" error={truth_error or 'NONE'}",
                    file=sys.stderr,
                )
            if truth_status == "LOCK_BUSY":
                return 75
            if truth_status not in {
                "PROJECTED_NO_DUE",
                "NO_DUE_VEHICLES",
                "RESOLVED",
                "RESOLVED_WITH_ERRORS",
            }:
                return 1
            if pending_v2:
                return 1
        return 0
    finally:
        os.close(descriptor)


def launch_worker(
    *,
    spool: Path,
    output: Path,
    ledger: Path,
    source_archive: Path,
    log_path: Path,
    outcome_enabled: bool = False,
) -> int:
    if os.environ.get("QR_LIVE_ENABLED", "0") != "0":
        print("fast-bot episode launcher requires QR_LIVE_ENABLED=0", file=sys.stderr)
        return 2
    if os.environ.get("QR_AUTOTRADE_LOCK_HELD", "0") != "0":
        print("fast-bot episode launcher refuses the shared live lock", file=sys.stderr)
        return 2
    if os.environ.get("QR_AUTOTRADE_LOCK_OWNER_TOKEN"):
        print("fast-bot episode launcher refuses a live-lock owner token", file=sys.stderr)
        return 2
    owner_id = _ensure_spool_owner(
        spool=spool,
        output=output,
        ledger=ledger,
        source_archive=source_archive,
    )
    spool.mkdir(parents=True, exist_ok=True)
    with _metadata_lock(spool):
        _cleanup_stale_atomic_temps_unlocked(spool)
        has_final = bool(
            _final_handoffs_unlocked(spool, owner_id=owner_id)
        )
        has_stale_outer = bool(
            _stale_outer_snapshots_unlocked(
                spool,
                owner_id=owner_id,
            )
        )
        if not has_final and not has_stale_outer and not outcome_enabled:
            return 0
    environment = os.environ.copy()
    environment["QR_LIVE_ENABLED"] = "0"
    environment["QR_AUTOTRADE_LOCK_HELD"] = "0"
    environment.pop("QR_AUTOTRADE_LOCK_OWNER_TOKEN", None)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--spool",
        str(spool),
        "--output",
        str(output),
        "--ledger",
        str(ledger),
        "--source-archive",
        str(source_archive),
        "--log",
        str(log_path),
    ]
    if outcome_enabled:
        command.append("--outcome-enabled")
    with _metadata_lock(spool):
        log_descriptor = _open_bounded_worker_log(log_path)
    with os.fdopen(log_descriptor, "ab", buffering=0) as log_handle:
        subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-capacity", action="store_true")
    action.add_argument("--reserve", action="store_true")
    action.add_argument("--publish", type=Path)
    action.add_argument("--launch", action="store_true")
    action.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--outcome-enabled",
        action="store_true",
        help="resolve mature episode vehicles even when the handoff spool is empty",
    )
    parser.add_argument("--spool", type=Path, default=ROOT / "data" / "fast_bot_episode_handoffs")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "fast_bot_episode_state.json")
    parser.add_argument("--ledger", type=Path, default=ROOT / "data" / "fast_bot_episode_ledger.jsonl")
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=ROOT / "data" / "fast_bot_episode_sources",
    )
    parser.add_argument("--log", type=Path, default=ROOT / "logs" / "fast_bot_episode_worker.log")
    parser.add_argument("--producer-pid", type=int)
    args = parser.parse_args()

    try:
        if args.check_capacity:
            available, owner_id = spool_accepts_handoff(
                args.spool,
                output=args.output,
                ledger=args.ledger,
                source_archive=args.source_archive,
            )
            if available:
                print(owner_id)
                return 0
            return 75
        if args.reserve:
            print(
                reserve_handoff(
                    args.spool,
                    output=args.output,
                    ledger=args.ledger,
                    source_archive=args.source_archive,
                    producer_pid=(
                        args.producer_pid
                        if args.producer_pid is not None
                        else os.getppid()
                    ),
                )
            )
            return 0
        if args.publish is not None:
            print(
                publish_handoff(
                    args.spool,
                    args.publish,
                    output=args.output,
                    ledger=args.ledger,
                    source_archive=args.source_archive,
                )
            )
            return 0
        if args.launch:
            return launch_worker(
                spool=args.spool,
                output=args.output,
                ledger=args.ledger,
                source_archive=args.source_archive,
                log_path=args.log,
                outcome_enabled=args.outcome_enabled,
            )
        return run_worker(
            spool=args.spool,
            output=args.output,
            ledger=args.ledger,
            source_archive=args.source_archive,
            outcome_enabled=args.outcome_enabled,
        )
    except (SpoolFullError, MetadataLockBusyError) as error:
        print(str(error), file=sys.stderr)
        return 75
    except (OSError, TypeError, ValueError) as error:
        print(
            f"fast-bot episode spool operation failed: {type(error).__name__}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
