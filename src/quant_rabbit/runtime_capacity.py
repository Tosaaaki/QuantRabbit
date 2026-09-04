"""Fail-closed, read-only capacity checks for QuantRabbit runtimes.

This module deliberately does not prune files.  It measures filesystem free
space and explicitly configured runtime roots, emits a bounded latest-state
receipt, and provides helpers that let artifact producers avoid unchanged
rewrites.  Directory measurement uses metadata only: file contents are never
opened and symbolic links are never followed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


# This is a host-contract schema generation, not a market parameter.  It is a
# constant so readers can reject incompatible receipts; replace it with a new
# schema version only when the serialized receipt contract changes.
RECEIPT_SCHEMA_VERSION = 1

# One MiB is a host-I/O memory bound, unrelated to trading or market behavior.
# It is constant to keep hashing memory bounded; replace it only with a measured
# host-specific streaming chunk policy if profiling shows this size is harmful.
HASH_CHUNK_BYTES = 1024 * 1024


class CapacityStatus(str, Enum):
    """Ordered runtime capacity states."""

    OK = "OK"
    PRESSURE = "PRESSURE"
    BLOCK = "BLOCK"


class CapacityReceiptError(RuntimeError):
    """A receipt could not be trusted."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RootQuota:
    """Two-stage quota for one explicitly authorized runtime root."""

    name: str
    path: Path
    pressure_bytes: int
    block_bytes: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("quota name must be non-empty")
        if self.pressure_bytes < 0 or self.block_bytes <= self.pressure_bytes:
            raise ValueError("quota requires 0 <= pressure_bytes < block_bytes")


@dataclass(frozen=True)
class CapacityPolicy:
    """Filesystem free-space watermarks and optional per-root quotas."""

    filesystem_path: Path
    low_free_bytes: int
    high_free_bytes: int
    root_quotas: tuple[RootQuota, ...] = ()

    def __post_init__(self) -> None:
        if self.low_free_bytes < 0 or self.high_free_bytes <= self.low_free_bytes:
            raise ValueError("policy requires 0 <= low_free_bytes < high_free_bytes")
        names = [quota.name for quota in self.root_quotas]
        if len(names) != len(set(names)):
            raise ValueError("quota names must be unique")


@dataclass(frozen=True)
class CapacityAssessment:
    """One immutable capacity observation."""

    status: CapacityStatus
    filesystem_path: str
    total_bytes: int | None
    used_bytes: int | None
    free_bytes: int | None
    low_free_bytes: int
    high_free_bytes: int
    roots: tuple[Mapping[str, Any], ...]
    issues: tuple[str, ...]

    def material(self) -> dict[str, Any]:
        """Return the stable, timestamp-free receipt material."""

        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": self.status.value,
            "filesystem": {
                "path": self.filesystem_path,
                "total_bytes": self.total_bytes,
                "used_bytes": self.used_bytes,
                "free_bytes": self.free_bytes,
                "low_free_bytes": self.low_free_bytes,
                "high_free_bytes": self.high_free_bytes,
            },
            "roots": [dict(root) for root in self.roots],
            "issues": list(self.issues),
        }


DiskUsageReader = Callable[[Path], Any]
RootSizeReader = Callable[[Path], int]


def measure_tree_size(path: Path) -> int:
    """Measure regular-file bytes under *path* without reading file contents.

    Symlinks and non-regular filesystem objects are ignored.  Any metadata or
    traversal error propagates so callers can fail closed rather than reporting
    a misleading partial total.
    """

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise OSError(f"capacity root is not a readable directory: {root}")
    total = 0
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += int(entry.stat(follow_symlinks=False).st_size)
    return total


def evaluate_capacity(
    policy: CapacityPolicy,
    *,
    disk_usage_reader: DiskUsageReader = shutil.disk_usage,
    root_size_reader: RootSizeReader = measure_tree_size,
) -> CapacityAssessment:
    """Evaluate a policy without deleting or changing any filesystem object."""

    issues: list[str] = []
    total_bytes: int | None = None
    used_bytes: int | None = None
    free_bytes: int | None = None
    disk_status = CapacityStatus.BLOCK

    try:
        usage = disk_usage_reader(policy.filesystem_path)
        total_bytes = _nonnegative_int(usage.total, "filesystem total")
        used_bytes = _nonnegative_int(usage.used, "filesystem used")
        free_bytes = _nonnegative_int(usage.free, "filesystem free")
        if used_bytes + free_bytes > total_bytes:
            raise ValueError("filesystem usage fields are inconsistent")
        if free_bytes < policy.low_free_bytes:
            disk_status = CapacityStatus.BLOCK
            issues.append("FILESYSTEM_FREE_BELOW_LOW_WATERMARK")
        elif free_bytes < policy.high_free_bytes:
            disk_status = CapacityStatus.PRESSURE
            issues.append("FILESYSTEM_FREE_BELOW_HIGH_WATERMARK")
        else:
            disk_status = CapacityStatus.OK
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        issues.append(f"FILESYSTEM_STATS_UNAVAILABLE:{type(exc).__name__}")

    roots: list[Mapping[str, Any]] = []
    root_statuses: list[CapacityStatus] = []
    for quota in sorted(policy.root_quotas, key=lambda item: item.name):
        used: int | None = None
        state = CapacityStatus.BLOCK
        root_issues: list[str] = []
        try:
            used = _nonnegative_int(root_size_reader(quota.path), "root used")
            if used >= quota.block_bytes:
                state = CapacityStatus.BLOCK
                root_issues.append("ROOT_QUOTA_REACHED")
            elif used >= quota.pressure_bytes:
                state = CapacityStatus.PRESSURE
                root_issues.append("ROOT_QUOTA_PRESSURE")
            else:
                state = CapacityStatus.OK
        except (OSError, TypeError, ValueError) as exc:
            root_issues.append(f"ROOT_STATS_UNAVAILABLE:{type(exc).__name__}")
        roots.append(
            {
                "name": quota.name,
                "path": str(Path(quota.path).resolve(strict=False)),
                "used_bytes": used,
                "pressure_bytes": quota.pressure_bytes,
                "block_bytes": quota.block_bytes,
                "status": state.value,
                "issues": root_issues,
            }
        )
        root_statuses.append(state)
        issues.extend(f"{quota.name}:{issue}" for issue in root_issues)

    statuses = [disk_status, *root_statuses]
    status = _worst_status(statuses)
    return CapacityAssessment(
        status=status,
        filesystem_path=str(Path(policy.filesystem_path).resolve(strict=False)),
        total_bytes=total_bytes,
        used_bytes=used_bytes,
        free_bytes=free_bytes,
        low_free_bytes=policy.low_free_bytes,
        high_free_bytes=policy.high_free_bytes,
        roots=tuple(roots),
        issues=tuple(issues),
    )


def capture_size_snapshot(
    roots: Mapping[str, Path],
    *,
    root_size_reader: RootSizeReader = measure_tree_size,
) -> dict[str, int]:
    """Capture metadata-only sizes for a caller-supplied set of roots."""

    if not roots:
        raise ValueError("at least one root is required")
    snapshot: dict[str, int] = {}
    for name, path in sorted(roots.items()):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("snapshot root names must be non-empty strings")
        snapshot[name] = _nonnegative_int(root_size_reader(Path(path)), "root used")
    return snapshot


def measure_cycle_size_delta(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, Any]:
    """Return exact per-root and total byte deltas for one runtime cycle."""

    if not before or set(before) != set(after):
        raise ValueError("before and after snapshots must have identical non-empty roots")
    rows: list[dict[str, Any]] = []
    total_before = 0
    total_after = 0
    for name in sorted(before):
        start = _nonnegative_int(before[name], "before size")
        end = _nonnegative_int(after[name], "after size")
        total_before += start
        total_after += end
        rows.append(
            {
                "name": name,
                "before_bytes": start,
                "after_bytes": end,
                "delta_bytes": end - start,
            }
        )
    return {
        "roots": rows,
        "total_before_bytes": total_before,
        "total_after_bytes": total_after,
        "total_delta_bytes": total_after - total_before,
    }


def canonical_json_digest(value: Any) -> str:
    """Return SHA-256 of deterministic JSON bytes."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def content_digest_unchanged(path: Path, candidate: bytes) -> bool:
    """Tell a producer whether exact candidate bytes already exist at *path*."""

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        return False
    try:
        if target.stat().st_size != len(candidate):
            return False
        expected = hashlib.sha256(candidate).digest()
        actual = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                actual.update(chunk)
        return actual.digest() == expected
    except OSError:
        return False


def build_capacity_receipt(
    assessment: CapacityAssessment,
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a self-verifying latest-state receipt."""

    observed = _utc_now(observed_at)
    material = assessment.material()
    content_digest = canonical_json_digest(material)
    envelope = {
        **material,
        "observed_at_utc": observed.isoformat().replace("+00:00", "Z"),
        "content_sha256": content_digest,
    }
    receipt_digest = canonical_json_digest(envelope)
    return {
        **envelope,
        "receipt_sha256": receipt_digest,
        "receipt_id": f"qrcap:{receipt_digest}",
    }


def write_capacity_receipt(
    path: Path,
    assessment: CapacityAssessment,
    *,
    observed_at: datetime | None = None,
) -> bool:
    """Atomically replace the latest receipt, skipping unchanged material.

    Returns ``True`` only when bytes were published.  A pre-existing malformed
    or tampered receipt is not overwritten silently; callers must surface that
    fail-closed state before deciding how to recover it.
    """

    target = Path(path)
    receipt = build_capacity_receipt(assessment, observed_at=observed_at)
    if target.is_symlink():
        raise CapacityReceiptError("RECEIPT_PATH_SYMLINK", "receipt path is a symlink")
    if target.exists():
        current = read_capacity_receipt(target)
        if current["content_sha256"] == receipt["content_sha256"]:
            return False
    _atomic_write_json(target, receipt)
    return True


def read_capacity_receipt(path: Path) -> dict[str, Any]:
    """Read and verify a capacity receipt or raise ``CapacityReceiptError``."""

    target = Path(path)
    if target.is_symlink():
        raise CapacityReceiptError("RECEIPT_PATH_SYMLINK", "receipt path is a symlink")
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CapacityReceiptError("RECEIPT_MISSING", "capacity receipt is missing") from exc
    except OSError as exc:
        raise CapacityReceiptError("RECEIPT_UNREADABLE", "capacity receipt is unreadable") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CapacityReceiptError("RECEIPT_MALFORMED", "capacity receipt is not JSON") from exc
    if not isinstance(payload, dict):
        raise CapacityReceiptError("RECEIPT_MALFORMED", "capacity receipt must be an object")
    if payload.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise CapacityReceiptError("RECEIPT_SCHEMA_UNSUPPORTED", "receipt schema is unsupported")
    claimed = payload.get("content_sha256")
    receipt_digest = payload.get("receipt_sha256")
    receipt_id = payload.get("receipt_id")
    if (
        not isinstance(claimed, str)
        or not isinstance(receipt_digest, str)
        or receipt_id != f"qrcap:{receipt_digest}"
    ):
        raise CapacityReceiptError("RECEIPT_TAMPERED", "receipt identity is invalid")
    material = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "observed_at_utc",
            "content_sha256",
            "receipt_sha256",
            "receipt_id",
        }
    }
    if canonical_json_digest(material) != claimed:
        raise CapacityReceiptError("RECEIPT_TAMPERED", "receipt content digest mismatches")
    envelope = {
        key: value
        for key, value in payload.items()
        if key not in {"receipt_sha256", "receipt_id"}
    }
    if canonical_json_digest(envelope) != receipt_digest:
        raise CapacityReceiptError("RECEIPT_TAMPERED", "receipt envelope digest mismatches")
    if payload.get("status") not in {item.value for item in CapacityStatus}:
        raise CapacityReceiptError("RECEIPT_MALFORMED", "receipt status is invalid")
    return payload


def read_capacity_receipt_fail_closed(path: Path) -> dict[str, Any]:
    """Read a receipt, returning a BLOCK object on any missing/untrusted state."""

    try:
        return read_capacity_receipt(path)
    except CapacityReceiptError as exc:
        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": CapacityStatus.BLOCK.value,
            "issues": [exc.code],
            "trusted": False,
        }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_flags = getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, os.O_RDONLY | directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _worst_status(statuses: list[CapacityStatus]) -> CapacityStatus:
    rank = {
        CapacityStatus.OK: 0,
        CapacityStatus.PRESSURE: 1,
        CapacityStatus.BLOCK: 2,
    }
    return max(statuses, key=rank.__getitem__)


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("observed_at must include a timezone")
    return current.astimezone(timezone.utc)
