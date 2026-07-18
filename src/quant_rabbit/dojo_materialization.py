"""Deterministic receipt for materializing external evidence symlinks.

The original Fable handoff contains regular files plus absolute symlinks into
its scratchpad.  A durable archive must preserve the bytes, the original link
literal, and the path mapping while making the canonical copy regular-file
only.  This module builds and verifies that mapping without granting any
research validity or live authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final


CONTRACT: Final = "QR_DOJO_MATERIALIZATION_RECEIPT_V2"
SCHEMA_VERSION: Final = 2
MAX_RECEIPT_BYTES: Final = 8 * 1024 * 1024
HASH_CHUNK_BYTES: Final = 1024 * 1024


class MaterializationError(ValueError):
    """Raised when a materialized archive cannot be proved exact."""


def canonical_sha256(value: Any) -> str:
    """Return strict canonical-JSON SHA-256."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MaterializationError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _real_directory(path: Path, *, label: str) -> Path:
    raw = path.expanduser()
    try:
        raw_state = raw.lstat()
    except OSError as exc:
        raise MaterializationError(f"{label} is unavailable: {raw}: {exc}") from exc
    if stat.S_ISLNK(raw_state.st_mode) or not stat.S_ISDIR(raw_state.st_mode):
        raise MaterializationError(f"{label} must be one real directory: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MaterializationError(f"cannot resolve {label}: {raw}: {exc}") from exc
    if not stat.S_ISDIR(resolved.lstat().st_mode):
        raise MaterializationError(f"{label} must resolve to a directory: {resolved}")
    return resolved


def _walk_entries(root: Path, *, source: bool) -> dict[str, Path]:
    entries: dict[str, Path] = {}
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise MaterializationError(f"cannot scan {directory}: {exc}") from exc
        child_directories: list[Path] = []
        for child in children:
            child_path = Path(child.path)
            relative = child_path.relative_to(root).as_posix()
            try:
                if child.is_symlink():
                    if not source:
                        raise MaterializationError(
                            f"materialized tree contains a symlink: {relative}"
                        )
                    entries[relative] = child_path
                elif child.is_file(follow_symlinks=False):
                    entries[relative] = child_path
                elif child.is_dir(follow_symlinks=False):
                    child_directories.append(child_path)
                else:
                    raise MaterializationError(
                        f"tree contains a non-regular special entry: {relative}"
                    )
            except OSError as exc:
                raise MaterializationError(
                    f"cannot inspect tree entry {relative}: {exc}"
                ) from exc
        stack.extend(reversed(child_directories))
    return entries


def _stable_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_regular_file(path: Path) -> tuple[str, int]:
    try:
        before_path = path.lstat()
    except OSError as exc:
        raise MaterializationError(f"cannot stat evidence file {path}: {exc}") from exc
    if not stat.S_ISREG(before_path.st_mode):
        raise MaterializationError(f"evidence target is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    observed = 0
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before_fd = os.fstat(handle.fileno())
            if _stable_identity(before_fd) != _stable_identity(before_path):
                raise MaterializationError(f"evidence file changed before read: {path}")
            while chunk := handle.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
                observed += len(chunk)
            after_fd = os.fstat(handle.fileno())
    except MaterializationError:
        raise
    except OSError as exc:
        raise MaterializationError(f"cannot read evidence file {path}: {exc}") from exc
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise MaterializationError(f"evidence file changed after read: {path}") from exc
    if (
        _stable_identity(before_fd) != _stable_identity(after_fd)
        or _stable_identity(before_path) != _stable_identity(after_path)
        or observed != before_fd.st_size
    ):
        raise MaterializationError(f"evidence file changed while hashing: {path}")
    return digest.hexdigest(), observed


def _source_entry(path: Path) -> tuple[str, str | None, str, int]:
    state_before = path.lstat()
    if stat.S_ISREG(state_before.st_mode):
        digest, size = _hash_regular_file(path)
        return "REGULAR", None, digest, size
    if not stat.S_ISLNK(state_before.st_mode):
        raise MaterializationError(f"source entry is not regular or symlink: {path}")
    try:
        link_literal_before = os.readlink(path)
        resolved_target = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MaterializationError(
            f"source symlink is broken or cyclic: {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(resolved_target.lstat().st_mode):
        raise MaterializationError(f"source symlink must resolve to a file: {path}")
    digest, size = _hash_regular_file(resolved_target)
    try:
        state_after = path.lstat()
        link_literal_after = os.readlink(path)
    except OSError as exc:
        raise MaterializationError(
            f"source symlink changed during read: {path}"
        ) from exc
    if (
        _stable_identity(state_before) != _stable_identity(state_after)
        or link_literal_before != link_literal_after
    ):
        raise MaterializationError(f"source symlink changed during read: {path}")
    return "SYMLINK", link_literal_before, digest, size


def _build_receipt_body(*, source: Path, materialized: Path) -> dict[str, Any]:
    source_entries = _walk_entries(source, source=True)
    materialized_entries = _walk_entries(materialized, source=False)
    source_paths = set(source_entries)
    materialized_paths = set(materialized_entries)
    if source_paths != materialized_paths:
        missing = sorted(source_paths - materialized_paths)[:20]
        extra = sorted(materialized_paths - source_paths)[:20]
        raise MaterializationError(
            f"materialized path set mismatch: missing={missing}, extra={extra}"
        )

    rows: list[dict[str, Any]] = []
    regular_count = 0
    symlink_count = 0
    total_bytes = 0
    for relative in sorted(source_entries):
        entry_type, link_target, source_sha, source_size = _source_entry(
            source_entries[relative]
        )
        materialized_sha, materialized_size = _hash_regular_file(
            materialized_entries[relative]
        )
        if source_sha != materialized_sha or source_size != materialized_size:
            raise MaterializationError(
                f"materialized bytes differ from source target: {relative}"
            )
        if entry_type == "REGULAR":
            regular_count += 1
        else:
            symlink_count += 1
        total_bytes += materialized_size
        rows.append(
            {
                "path": relative,
                "source_entry_type": entry_type,
                "source_link_target": link_target,
                "source_resolved_target_sha256": source_sha,
                "source_resolved_target_size": source_size,
                "materialized_sha256": materialized_sha,
                "materialized_size": materialized_size,
            }
        )

    return {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "source_root": os.fspath(source),
        "materialized_root": os.fspath(materialized),
        "file_count": len(rows),
        "source_regular_count": regular_count,
        "source_symlink_count": symlink_count,
        "total_materialized_bytes": total_bytes,
        "files": rows,
        "materialized_regular_only": True,
        "live_permission": False,
        "order_authority": "NONE",
    }


def build_materialization_receipt(
    *, source_root: Path, materialized_root: Path
) -> dict[str, Any]:
    """Build a deterministic byte/path/link mapping receipt.

    Two complete scans must agree.  The duplicated read is intentional: the
    handoff is external evidence, so a path, link target, or file that changes
    while the receipt is being assembled must fail closed instead of producing
    a mixed-epoch inventory.
    """

    source = _real_directory(source_root, label="source root")
    materialized = _real_directory(materialized_root, label="materialized root")
    if source == materialized:
        raise MaterializationError("source and materialized roots must differ")
    first_body = _build_receipt_body(source=source, materialized=materialized)
    second_body = _build_receipt_body(source=source, materialized=materialized)
    if first_body != second_body:
        raise MaterializationError(
            "source or materialized tree changed between complete scans"
        )
    return {**first_body, "receipt_sha256": canonical_sha256(first_body)}


def _reject_constant(value: str) -> None:
    raise MaterializationError(f"non-finite JSON constant is forbidden: {value}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise MaterializationError(f"duplicate receipt key is forbidden: {key}")
        value[key] = item
    return value


def load_materialization_receipt(path: Path) -> dict[str, Any]:
    """Load and self-verify one bounded receipt."""

    try:
        state = path.lstat()
    except OSError as exc:
        raise MaterializationError(f"cannot stat receipt: {path}: {exc}") from exc
    if not stat.S_ISREG(state.st_mode) or state.st_size <= 0:
        raise MaterializationError("receipt must be one non-empty regular file")
    if state.st_size > MAX_RECEIPT_BYTES:
        raise MaterializationError("receipt exceeds the bounded JSON size")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read(MAX_RECEIPT_BYTES + 1)
            after = os.fstat(handle.fileno())
        if (
            _stable_identity(before) != _stable_identity(after)
            or len(payload) != before.st_size
            or before.st_size != state.st_size
        ):
            raise MaterializationError("receipt changed while reading")
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except MaterializationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot load receipt: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterializationError("receipt must contain one JSON object")
    expected_keys = {
        "contract",
        "schema_version",
        "source_root",
        "materialized_root",
        "file_count",
        "source_regular_count",
        "source_symlink_count",
        "total_materialized_bytes",
        "files",
        "materialized_regular_only",
        "live_permission",
        "order_authority",
        "receipt_sha256",
    }
    if set(value) != expected_keys:
        raise MaterializationError(
            "receipt has invalid keys; "
            f"missing={sorted(expected_keys - set(value))}, "
            f"unknown={sorted(set(value) - expected_keys)}"
        )
    if value["contract"] != CONTRACT or value["schema_version"] != SCHEMA_VERSION:
        raise MaterializationError("receipt contract or schema version is unsupported")
    body = dict(value)
    declared_sha = body.pop("receipt_sha256")
    if not isinstance(declared_sha, str) or canonical_sha256(body) != declared_sha:
        raise MaterializationError("receipt canonical SHA-256 does not verify")
    if (
        value["materialized_regular_only"] is not True
        or value["live_permission"] is not False
        or value["order_authority"] != "NONE"
    ):
        raise MaterializationError("receipt safety declarations are invalid")
    count_fields = (
        "file_count",
        "source_regular_count",
        "source_symlink_count",
        "total_materialized_bytes",
    )
    if any(
        isinstance(value[field], bool)
        or not isinstance(value[field], int)
        or value[field] < 0
        for field in count_fields
    ):
        raise MaterializationError("receipt count fields must be non-negative integers")
    rows = value["files"]
    if not isinstance(rows, list) or len(rows) != value["file_count"]:
        raise MaterializationError("receipt file_count does not match files")
    if (
        value["source_regular_count"] + value["source_symlink_count"]
        != value["file_count"]
    ):
        raise MaterializationError("receipt source type counts do not add up")
    row_keys = {
        "path",
        "source_entry_type",
        "source_link_target",
        "source_resolved_target_sha256",
        "source_resolved_target_size",
        "materialized_sha256",
        "materialized_size",
    }
    observed_paths: list[str] = []
    observed_bytes = 0
    observed_regular = 0
    observed_symlink = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != row_keys:
            raise MaterializationError(f"receipt files[{index}] has invalid keys")
        relative = row["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            raise MaterializationError(f"receipt files[{index}].path is invalid")
        observed_paths.append(relative)
        entry_type = row["source_entry_type"]
        link_target = row["source_link_target"]
        if entry_type == "REGULAR":
            observed_regular += 1
            if link_target is not None:
                raise MaterializationError(
                    "regular source row cannot have a link target"
                )
        elif entry_type == "SYMLINK":
            observed_symlink += 1
            if not isinstance(link_target, str) or not link_target:
                raise MaterializationError(
                    "symlink source row requires its link literal"
                )
        else:
            raise MaterializationError("source_entry_type is unsupported")
        for sha_field in (
            "source_resolved_target_sha256",
            "materialized_sha256",
        ):
            sha = row[sha_field]
            if (
                not isinstance(sha, str)
                or len(sha) != 64
                or any(character not in "0123456789abcdef" for character in sha)
            ):
                raise MaterializationError(f"receipt {sha_field} is malformed")
        for size_field in ("source_resolved_target_size", "materialized_size"):
            size = row[size_field]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise MaterializationError(f"receipt {size_field} is invalid")
        if (
            row["source_resolved_target_sha256"] != row["materialized_sha256"]
            or row["source_resolved_target_size"] != row["materialized_size"]
        ):
            raise MaterializationError(
                "receipt row does not prove exact materialization"
            )
        observed_bytes += row["materialized_size"]
    if observed_paths != sorted(set(observed_paths)):
        raise MaterializationError("receipt file paths must be unique and sorted")
    if (
        observed_regular != value["source_regular_count"]
        or observed_symlink != value["source_symlink_count"]
        or observed_bytes != value["total_materialized_bytes"]
    ):
        raise MaterializationError("receipt aggregate counts do not match file rows")
    return value


def verify_materialization_receipt(
    *, receipt_path: Path, source_root: Path, materialized_root: Path
) -> dict[str, Any]:
    """Rebuild current truth and require exact equality with a sealed receipt."""

    sealed = verify_materialized_archive(
        receipt_path=receipt_path,
        materialized_root=materialized_root,
    )
    current = build_materialization_receipt(
        source_root=source_root,
        materialized_root=materialized_root,
    )
    if sealed != current:
        raise MaterializationError(
            "materialization receipt does not match current source/materialized trees"
        )
    return current


def verify_materialized_archive(
    *, receipt_path: Path, materialized_root: Path
) -> dict[str, Any]:
    """Verify the durable regular-file archive without reopening link targets.

    Full source verification is required when the receipt is created.  This
    narrower verifier intentionally remains usable after the original carrier
    or its external symlink targets have been retired.
    """

    sealed = load_materialization_receipt(receipt_path)
    materialized = _real_directory(materialized_root, label="materialized root")
    if os.fspath(materialized) != sealed["materialized_root"]:
        raise MaterializationError(
            "materialized root does not match the sealed receipt identity"
        )
    actual = _walk_entries(materialized, source=False)
    expected_rows = {row["path"]: row for row in sealed["files"]}
    if set(actual) != set(expected_rows):
        missing = sorted(set(expected_rows) - set(actual))[:20]
        extra = sorted(set(actual) - set(expected_rows))[:20]
        raise MaterializationError(
            f"materialized path set mismatch: missing={missing}, extra={extra}"
        )
    observed_bytes = 0
    for relative, path in sorted(actual.items()):
        digest, size = _hash_regular_file(path)
        row = expected_rows[relative]
        if digest != row["materialized_sha256"] or size != row["materialized_size"]:
            raise MaterializationError(
                f"materialized bytes differ from sealed receipt: {relative}"
            )
        observed_bytes += size
    if observed_bytes != sealed["total_materialized_bytes"]:
        raise MaterializationError(
            "materialized byte count differs from sealed receipt"
        )
    return sealed


def publish_receipt_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Durably create a receipt once; never replace an existing evidence file."""

    parent = _real_directory(path.parent, label="receipt parent")
    output = parent / path.name
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    parent_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    parent_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    file_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = os.open(parent, parent_flags)
    try:
        try:
            descriptor = os.open(
                output.name, file_flags, 0o600, dir_fd=parent_descriptor
            )
        except FileExistsError as exc:
            raise MaterializationError(f"receipt already exists: {output}") from exc
        created = True
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.fsync(parent_descriptor)
        except BaseException:
            if created:
                try:
                    os.unlink(output.name, dir_fd=parent_descriptor)
                except OSError:
                    pass
            raise
    finally:
        os.close(parent_descriptor)
