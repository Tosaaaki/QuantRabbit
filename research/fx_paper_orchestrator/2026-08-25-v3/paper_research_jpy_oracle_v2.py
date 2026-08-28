#!/usr/bin/env python3
"""Independent accounting-only JPY oracle for future paper research cycles.

This module deliberately does not import a strategy runner, the legacy JPY
accounting implementation, the shadow runtime, or a result validator.  It is a
file/FD capability process: exact source bytes and frozen ex-ante artifacts are
read below a launcher-owned directory FD and immutable evidence is published
below a different launcher-owned directory FD.

The oracle verifies causal proposal *provenance* but does not recreate the
detector direction.  Its output is therefore ACCOUNTING_ONLY and can never, by
itself, admit a strategy.  V3 must attach a separate as-of detector replay
receipt before admission.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ORACLE_NAME = "INDEPENDENT_JPY_ORACLE_V2"
CONTRACT_NAME = "PAPER_RESEARCH_JPY_ORACLE_CONTRACT_V2.json"
SCHEMA_NAME = "paper_research_jpy_oracle_schema_v2.json"
ARMS = ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")
ZERO_SHA = "0" * 64
DAY_NS = 86_400_000_000_000
HOUR_NS = 3_600_000_000_000
JPY_MICROS_PER_YEN = 1_000_000
BASE_MICROUNITS_PER_UNIT = 1_000_000
RATIO_DECIMAL_SCALE = 10**18
PRICE_SUBPIP_SCALE = 1_000_000
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_SOURCE_ROWS = 5_000_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
DECIMAL_TEXT_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
FORBIDDEN_PROPOSAL_TOKENS = {
    "signalid", "fill", "fillprice", "path", "mfe", "mae", "pnl", "cost",
    "equity", "drawdown", "dd", "cvar", "profit", "return",
}
AUTHORITY = {
    "paper_only": True,
    "live_authority": False,
    "broker_account_access": False,
    "credential_access": False,
    "order_endpoint": False,
    "external_orders": 0,
    "deploy": False,
    "external_config_mutation": False,
}
CLASSIFICATION = "FUTURE_ONLY_ACCOUNTING_ONLY_LOCAL_UNANCHORED_NOT_ADMISSIBLE"
ANCHOR_STATUS = "LOCAL_UNANCHORED"
EXECUTION_PROVENANCE_SCOPE = (
    "LOCAL_CALLER_ASSERTED_CONTENT_BINDING_NOT_EXECUTION_ATTESTATION_"
    "NOT_EXTERNALLY_ANCHORED"
)


class OracleError(RuntimeError):
    """Fail-closed oracle contract violation."""


class LockIdentityError(OracleError):
    """The live lock pathname no longer names the held locked inode."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _snapshot_regular_file(path: Path) -> bytes:
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        descriptor = os.open(
            path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise OracleError(f"immutable runtime artifact is not regular: {path.name}")
            if before.st_size > MAX_JSON_BYTES:
                raise OracleError(f"immutable runtime artifact exceeds fixed byte limit: {path.name}")
            chunks: list[bytes] = []
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor, min(1024 * 1024, before.st_size - offset), offset
                )
                if not chunk:
                    raise OracleError(f"immutable runtime artifact truncated: {path.name}")
                chunks.append(chunk)
                offset += len(chunk)
            if os.pread(descriptor, 1, before.st_size):
                raise OracleError(f"immutable runtime artifact grew: {path.name}")
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_nlink,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
            ):
                raise OracleError(f"immutable runtime artifact changed: {path.name}")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


if "_SEALED_RUNTIME_CODE_BYTES" in globals():
    _MODULE_PATH = Path("<sealed-fd-oracle-v2>")
    _MODULE_CODE_BYTES = globals()["_SEALED_RUNTIME_CODE_BYTES"]
    _CONTRACT_BYTES = globals()["_SEALED_CONTRACT_BYTES"]
    _SCHEMA_BYTES = globals()["_SEALED_SCHEMA_BYTES"]
    _LAUNCHER_SHA256 = globals()["_SEALED_LAUNCHER_SHA256"]
    _RENAME_EXCLUSIVE = globals().get("_SEALED_RENAME_EXCLUSIVE")
    if not all(type(value) is bytes for value in (
        _MODULE_CODE_BYTES, _CONTRACT_BYTES, _SCHEMA_BYTES,
    )) or type(_LAUNCHER_SHA256) is not str or SHA256_RE.fullmatch(_LAUNCHER_SHA256) is None \
            or not callable(_RENAME_EXCLUSIVE):
        raise OracleError("sealed runtime injection is malformed")
    _EXECUTION_SNAPSHOT_MODE = "SEALED_FD_COMPILE_EXEC_V2"
else:
    _MODULE_PATH = Path(__file__).resolve()
    _MODULE_CODE_BYTES = _snapshot_regular_file(_MODULE_PATH)
    _CONTRACT_BYTES = _snapshot_regular_file(_MODULE_PATH.parent / CONTRACT_NAME)
    _SCHEMA_BYTES = _snapshot_regular_file(_MODULE_PATH.parent / SCHEMA_NAME)
    _LAUNCHER_SHA256 = None
    _RENAME_EXCLUSIVE = None
    _EXECUTION_SNAPSHOT_MODE = "PATH_LOADED_TEST_ADAPTER_NOT_RELEASE_EVIDENCE"
_MODULE_CODE_SHA256 = sha256_bytes(_MODULE_CODE_BYTES)
_CONTRACT_SHA256 = sha256_bytes(_CONTRACT_BYTES)
_SCHEMA_SHA256 = sha256_bytes(_SCHEMA_BYTES)


def _assert_canonical_value(value: Any, *, location: str = "root") -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_canonical_value(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise OracleError(f"non-text JSON key at {location}")
            _assert_canonical_value(item, location=f"{location}.{key}")
        return
    raise OracleError(f"non-canonical JSON type at {location}: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    _assert_canonical_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def embedded_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return sha256_bytes(canonical_bytes(unsigned))


def _reject_float(_: str) -> Any:
    raise OracleError("floating-point JSON number forbidden")


def _strict_int(token: str) -> int:
    if token == "-0":
        raise OracleError("negative zero forbidden")
    return int(token)


def _pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OracleError(f"duplicate JSON key forbidden: {key}")
        result[key] = value
    return result


def strict_json(data: bytes, label: str, *, require_lf: bool = True) -> dict[str, Any]:
    if len(data) > MAX_JSON_BYTES:
        raise OracleError(f"{label} exceeds fixed byte limit")
    if data.startswith(b"\xef\xbb\xbf"):
        raise OracleError(f"{label} UTF-8 BOM forbidden")
    body = data
    if require_lf:
        if not data.endswith(b"\n") or data.endswith(b"\n\n"):
            raise OracleError(f"{label} must have exactly one terminal LF")
        body = data[:-1]
    try:
        text = body.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_pairs_object,
            parse_int=_strict_int,
            parse_float=_reject_float,
            parse_constant=_reject_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OracleError(f"invalid {label} JSON") from error
    if type(value) is not dict:
        raise OracleError(f"{label} must be an object")
    if canonical_bytes(value) != body:
        raise OracleError(f"{label} is not canonical JSON")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise OracleError(f"{label} schema mismatch missing={missing} extra={extra}")


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise OracleError(f"{label} must be an integer, bool is forbidden")
    if minimum is not None and value < minimum:
        raise OracleError(f"{label} below minimum")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise OracleError(f"{label} must be boolean")
    return value


def _validate_authority_exact(value: Any, label: str) -> None:
    if type(value) is not dict:
        raise OracleError(f"{label} must be object")
    _exact_keys(value, set(AUTHORITY), label)
    for key, expected in AUTHORITY.items():
        actual = value[key]
        if type(expected) is bool:
            if type(actual) is not bool or actual is not expected:
                raise OracleError(f"{label}.{key} exact boolean mismatch")
        elif type(expected) is int:
            if type(actual) is not int or actual != expected:
                raise OracleError(f"{label}.{key} exact integer mismatch")
        else:  # pragma: no cover - fixed constant schema guard
            raise OracleError("unsupported authority contract type")


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise OracleError(f"{label} must be lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise OracleError(f"{label} must be nonempty text")
    return value


def _validate_embedded(payload: Mapping[str, Any], field: str, label: str) -> None:
    digest = _digest(payload.get(field), f"{label}.{field}")
    if digest != embedded_hash(payload, field):
        raise OracleError(f"{label} embedded hash mismatch")


def _relative_parts(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not str or not value or len(value) > 512:
        raise OracleError(f"{label} relative path missing or too long")
    if value.startswith("/") or "//" in value or value.endswith("/"):
        raise OracleError(f"{label} must be canonical relative path")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} or SAFE_COMPONENT_RE.fullmatch(part) is None for part in parts):
        raise OracleError(f"{label} contains unsafe component")
    return parts


def _validate_dirfd(directory_fd: int, label: str) -> os.stat_result:
    info = os.fstat(directory_fd)
    if not stat.S_ISDIR(info.st_mode):
        raise OracleError(f"trusted {label} FD is not a directory")
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise OracleError(f"trusted {label} directory ownership/mode invalid")
    return info


def _assert_named_lock_identity(
    output_root_fd: int,
    lock_name: str,
    lock_fd: int,
) -> None:
    try:
        held = os.fstat(lock_fd)
        named = os.stat(lock_name, dir_fd=output_root_fd, follow_symlinks=False)
        access_mode = fcntl.fcntl(lock_fd, fcntl.F_GETFL) & os.O_ACCMODE
    except OSError as error:
        raise LockIdentityError("oracle lock pathname identity changed") from error
    for info in (held, named):
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() \
                or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600:
            raise LockIdentityError("oracle lock pathname identity changed")
    if access_mode != os.O_RDWR or (held.st_dev, held.st_ino) != (
        named.st_dev,
        named.st_ino,
    ):
        raise LockIdentityError("oracle lock pathname identity changed")


def _read_fd_snapshot(
    descriptor: int,
    label: str,
    *,
    allow_unlinked_sealed_runtime: bool = False,
) -> bytes:
    # Only launcher-pinned code/contract/schema FDs may become unlinked after
    # they were opened.  Their bytes are compared below with the already
    # compiled/injected snapshot.  Request and economic input artifacts keep
    # the stricter single-link rule so an unlinked clone cannot enter evidence.
    before = os.fstat(descriptor)
    allowed_link_counts = {0, 1} if allow_unlinked_sealed_runtime else {1}
    if not stat.S_ISREG(before.st_mode) or before.st_nlink not in allowed_link_counts:
        raise OracleError(f"{label} FD must be regular")
    if before.st_size > MAX_ARTIFACT_BYTES:
        raise OracleError(f"{label} FD exceeds fixed byte limit")
    if fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
        raise OracleError(f"{label} FD must be read-only")
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(
            descriptor, min(1024 * 1024, before.st_size - offset), offset
        )
        if not chunk:
            raise OracleError(f"{label} FD truncated during snapshot")
        chunks.append(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, before.st_size):
        raise OracleError(f"{label} FD grew during snapshot")
    after = os.fstat(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    ):
        raise OracleError(f"{label} FD changed during snapshot")
    return b"".join(chunks)


def _read_relative(
    root_fd: int,
    relative_path: str,
    label: str,
    *,
    expected_size: int | None = None,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> bytes:
    _validate_dirfd(root_fd, "input root")
    parts = _relative_parts(relative_path, label)
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
            next_info = os.fstat(next_fd)
            if not stat.S_ISDIR(next_info.st_mode) or next_info.st_uid != os.geteuid() \
                    or next_info.st_mode & 0o022:
                os.close(next_fd)
                raise OracleError(f"{label} parent ownership/mode invalid")
            os.close(current)
            current = next_fd
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=current,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid() \
                    or before.st_mode & 0o022 or before.st_nlink != 1:
                raise OracleError(f"{label} artifact ownership/type/mode invalid")
            if before.st_size > max_bytes:
                raise OracleError(f"{label} artifact exceeds fixed byte limit")
            if expected_size is not None and before.st_size != expected_size:
                raise OracleError(f"{label} declared size differs before read")
            chunks: list[bytes] = []
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor, min(1024 * 1024, before.st_size - offset), offset
                )
                if not chunk:
                    raise OracleError(f"{label} artifact truncated during read")
                chunks.append(chunk)
                offset += len(chunk)
            if os.pread(descriptor, 1, before.st_size):
                raise OracleError(f"{label} artifact grew during read")
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_nlink,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
            ):
                raise OracleError(f"{label} artifact changed during read")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(current)


def _artifact_bytes(spec: Mapping[str, Any], label: str, input_root_fd: int) -> bytes:
    _exact_keys(spec, {"artifact_id", "relative_path", "sha256", "size_bytes"}, label)
    if spec.get("artifact_id") != label:
        raise OracleError(f"{label} artifact identity mismatch")
    expected_size = _integer(spec.get("size_bytes"), f"{label}.size_bytes", minimum=0)
    expected_hash = _digest(spec.get("sha256"), f"{label}.sha256")
    data = _read_relative(
        input_root_fd,
        spec.get("relative_path"),
        label,
        expected_size=expected_size,
        max_bytes=(MAX_ARTIFACT_BYTES if label == "source_blob" else MAX_JSON_BYTES),
    )
    if len(data) != expected_size or sha256_bytes(data) != expected_hash:
        raise OracleError(f"{label} exact-byte binding mismatch")
    return data


def _pair(instrument: Any) -> tuple[str, str]:
    if type(instrument) is not str or PAIR_RE.fullmatch(instrument) is None:
        raise OracleError(f"invalid FX instrument: {instrument!r}")
    base, quote = instrument.split("_", 1)
    if base == quote:
        raise OracleError("FX base and quote currencies must differ")
    return base, quote


def _validate_instrument_registry(payload: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    _exact_keys(payload, {"schema_version", "registry_id", "instruments", "registry_sha256"}, "instrument registry")
    if _integer(payload["schema_version"], "instrument registry schema") != 1 \
            or payload["registry_id"] != "FROZEN_FX_INSTRUMENT_REGISTRY_V1":
        raise OracleError("instrument registry identity mismatch")
    _validate_embedded(payload, "registry_sha256", "instrument registry")
    raw = payload["instruments"]
    if type(raw) is not dict or not raw:
        raise OracleError("instrument registry empty")
    result: dict[str, dict[str, int]] = {}
    economic_pairs: set[tuple[str, str]] = set()
    for instrument, spec in raw.items():
        base, quote = _pair(instrument)
        economic_pair = tuple(sorted((base, quote)))
        if economic_pair in economic_pairs:
            raise OracleError("instrument registry contains inverse duplicate pair")
        economic_pairs.add(economic_pair)
        if type(spec) is not dict:
            raise OracleError("instrument unit spec must be object")
        _exact_keys(spec, {"price_scale", "pip_ticks"}, f"instrument {instrument}")
        price_scale = _integer(spec["price_scale"], f"{instrument}.price_scale", minimum=1)
        pip_ticks = _integer(spec["pip_ticks"], f"{instrument}.pip_ticks", minimum=1)
        if pip_ticks >= price_scale:
            raise OracleError(f"{instrument} pip convention invalid")
        result[instrument] = {"price_scale": price_scale, "pip_ticks": pip_ticks}
    if list(raw) != sorted(raw):
        raise OracleError("instrument registry keys must be sorted")
    return result


def _parse_source(
    blob: bytes,
    manifest: Mapping[str, Any],
    registry_payload: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    _exact_keys(
        manifest,
        {
            "schema_version", "source_bytes_sha256", "source_size_bytes", "event_count",
            "first_source_ts_ns", "last_source_ts_ns", "provider_allowlist",
            "instrument_registry_sha256", "stream_policies", "lossless", "manifest_sha256",
        },
        "source manifest",
    )
    if _integer(manifest["schema_version"], "source manifest schema") != 2:
        raise OracleError("source manifest schema version mismatch")
    _validate_embedded(manifest, "manifest_sha256", "source manifest")
    if _digest(manifest["source_bytes_sha256"], "source bytes hash") != sha256_bytes(blob) \
            or _integer(manifest["source_size_bytes"], "source size", minimum=0) != len(blob):
        raise OracleError("source manifest does not bind exact BBO bytes")
    if manifest["instrument_registry_sha256"] != registry_payload["registry_sha256"]:
        raise OracleError("source manifest instrument registry mismatch")
    providers = manifest["provider_allowlist"]
    if type(providers) is not list or not providers or providers != sorted(set(providers)) \
            or any(type(item) is not str or not item for item in providers):
        raise OracleError("provider allowlist invalid")
    if _boolean(manifest["lossless"], "source lossless") is not True:
        raise OracleError("oracle source must be lossless")
    policies_raw = manifest["stream_policies"]
    if type(policies_raw) is not list or not policies_raw:
        raise OracleError("stream policies missing")
    policies: dict[tuple[str, str], dict[str, Any]] = {}
    policy_order: list[tuple[str, str]] = []
    for spec in policies_raw:
        if type(spec) is not dict:
            raise OracleError("stream policy must be object")
        _exact_keys(
            spec,
            {
                "provider_id", "instrument", "sequence_required", "first_sequence",
                "last_sequence", "event_count", "max_source_gap_ns", "max_arrival_gap_ns",
            },
            "stream policy",
        )
        provider = _text(spec["provider_id"], "stream provider")
        instrument = _text(spec["instrument"], "stream instrument")
        _pair(instrument)
        if provider not in providers or instrument not in registry:
            raise OracleError("stream policy outside frozen provider/instrument registry")
        if _boolean(spec["sequence_required"], "sequence_required") is not True:
            raise OracleError("lossless stream requires sequence")
        for field in ("first_sequence", "last_sequence", "event_count", "max_source_gap_ns", "max_arrival_gap_ns"):
            _integer(spec[field], f"stream policy {field}", minimum=1)
        key = (provider, instrument)
        if key in policies:
            raise OracleError("duplicate stream policy")
        policies[key] = dict(spec)
        policy_order.append(key)
    if policy_order != sorted(policy_order):
        raise OracleError("stream policies must be sorted")
    if not blob or not blob.endswith(b"\n"):
        raise OracleError("empty or truncated source BBO blob")
    raw_lines = blob.splitlines(keepends=True)
    if len(raw_lines) > MAX_SOURCE_ROWS:
        raise OracleError("source row limit exceeded")
    expected_event_keys = {
        "schema_version", "provider_id", "instrument", "bid_ticks", "ask_ticks",
        "tick_scale", "source_ts_ns", "arrival_ts_ns", "provider_event_id", "sequence",
        "heartbeat", "quality_flags",
    }
    rows: list[dict[str, Any]] = []
    provider_event_ids: set[tuple[str, str, str]] = set()
    last_global: tuple[int, int, str, str, int] | None = None
    last_stream: dict[tuple[str, str], tuple[int, int, int]] = {}
    stream_counts: Counter[tuple[str, str]] = Counter()
    prefix = ZERO_SHA
    for line in raw_lines:
        row = strict_json(line, "source BBO record")
        _exact_keys(row, expected_event_keys, "source BBO record")
        if _integer(row["schema_version"], "source schema") != 1:
            raise OracleError("source event schema mismatch")
        provider = _text(row["provider_id"], "source provider")
        instrument = _text(row["instrument"], "source instrument")
        _pair(instrument)
        key = (provider, instrument)
        if key not in policies:
            raise OracleError("source stream not allowlisted")
        for field in ("bid_ticks", "ask_ticks", "tick_scale", "source_ts_ns", "arrival_ts_ns", "sequence"):
            _integer(row[field], f"source.{field}", minimum=1)
        if row["ask_ticks"] <= row["bid_ticks"] or row["arrival_ts_ns"] < row["source_ts_ns"]:
            raise OracleError("invalid executable BBO")
        if row["tick_scale"] != registry[instrument]["price_scale"]:
            raise OracleError("event price scale differs from frozen instrument registry")
        if row["provider_event_id"] is not None and type(row["provider_event_id"]) is not str:
            raise OracleError("provider_event_id type invalid")
        if row["provider_event_id"] is not None:
            provider_event_identity = (
                provider,
                instrument,
                row["provider_event_id"],
            )
            if provider_event_identity in provider_event_ids:
                raise OracleError("duplicate provider event identity")
            provider_event_ids.add(provider_event_identity)
        if _boolean(row["heartbeat"], "source heartbeat") is not False:
            raise OracleError("priced event required, heartbeat rows are not executable")
        if type(row["quality_flags"]) is not list or row["quality_flags"]:
            raise OracleError("quality-flagged event unavailable")
        order = (row["arrival_ts_ns"], row["source_ts_ns"], provider, instrument, row["sequence"])
        if last_global is not None and order <= last_global:
            raise OracleError("global source input order is not strictly arrival-monotonic")
        last_global = order
        prior = last_stream.get(key)
        if prior is not None:
            if row["source_ts_ns"] <= prior[0] or row["arrival_ts_ns"] <= prior[1] \
                    or row["sequence"] != prior[2] + 1:
                raise OracleError("stream chronology/sequence violation")
            policy = policies[key]
            if row["source_ts_ns"] - prior[0] > policy["max_source_gap_ns"] \
                    or row["arrival_ts_ns"] - prior[1] > policy["max_arrival_gap_ns"]:
                raise OracleError("stream gap exceeds frozen policy")
        last_stream[key] = (row["source_ts_ns"], row["arrival_ts_ns"], row["sequence"])
        stream_counts[key] += 1
        event_hash = sha256_bytes(line)
        prefix = sha256_bytes(canonical_bytes({"previous_hash": prefix, "source_event_sha256": event_hash}))
        enriched = dict(row)
        enriched["source_event_sha256"] = event_hash
        enriched["source_prefix_root_sha256"] = prefix
        rows.append(enriched)
    if _integer(manifest["event_count"], "manifest event_count", minimum=1) != len(rows):
        raise OracleError("source event count mismatch")
    if _integer(manifest["first_source_ts_ns"], "manifest first source", minimum=1) != min(row["source_ts_ns"] for row in rows) \
            or _integer(manifest["last_source_ts_ns"], "manifest last source", minimum=1) != max(row["source_ts_ns"] for row in rows):
        raise OracleError("source time boundary mismatch")
    if set(policies) != set(stream_counts):
        raise OracleError("source manifest has missing or extra stream policy")
    providers_by_instrument: defaultdict[str, set[str]] = defaultdict(set)
    for provider, instrument in policies:
        providers_by_instrument[instrument].add(provider)
    if any(len(items) != 1 for items in providers_by_instrument.values()):
        raise OracleError("multiple providers for one instrument are ambiguous")
    for key, policy in policies.items():
        first = next(row for row in rows if (row["provider_id"], row["instrument"]) == key)
        last = next(row for row in reversed(rows) if (row["provider_id"], row["instrument"]) == key)
        if policy["first_sequence"] != first["sequence"] \
                or policy["last_sequence"] != last["sequence"] \
                or policy["event_count"] != stream_counts[key]:
            raise OracleError("stream policy semantic count/sequence mismatch")
    books: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        books[row["instrument"]].append(row)
    return rows, dict(books)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _reject_producer_fields(value: Any, location: str = "proposal") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            token = _normalize_key(key)
            if token in FORBIDDEN_PROPOSAL_TOKENS:
                raise OracleError(f"proposal outcome/identifier forbidden at {location}.{key}")
            _reject_producer_fields(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_producer_fields(item, f"{location}[{index}]")


def _validate_proposal(
    payload: Mapping[str, Any], source_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    _reject_producer_fields(payload)
    _exact_keys(payload, {"schema_version", "candidate_key", "provenance", "rows", "proposal_sha256"}, "proposal")
    if _integer(payload["schema_version"], "proposal schema") != 2:
        raise OracleError("proposal schema mismatch")
    _validate_embedded(payload, "proposal_sha256", "proposal")
    _text(payload["candidate_key"], "candidate_key")
    provenance = payload["provenance"]
    if type(provenance) is not dict:
        raise OracleError("proposal provenance must be object")
    _exact_keys(
        provenance,
        {"detector_code_sha256", "detector_policy_sha256", "generator_policy_sha256", "source_acquisition_contract_sha256"},
        "proposal provenance",
    )
    for key, value in provenance.items():
        _digest(value, f"proposal provenance {key}")
    raw_rows = payload["rows"]
    if type(raw_rows) is not list or not raw_rows:
        raise OracleError("proposal rows missing")
    expected_keys = {
        "proposal_ordinal", "decision_source_ts_ns", "decision_arrival_ts_ns", "available_at_ns",
        "decision_source_event_sha256", "completed_data_watermark_source_ts_ns",
        "completed_data_prefix_root_sha256", "instrument", "direction", "notional_jpy_micros",
        "max_age_ns", "worker_key", "action",
    }
    source_by_hash = {row["source_event_sha256"]: row for row in source_rows}
    validated: list[dict[str, Any]] = []
    economic_lot_keys: set[str] = set()
    last_decision: tuple[int, int, int] | None = None
    for expected_ordinal, row in enumerate(raw_rows, 1):
        if type(row) is not dict:
            raise OracleError("proposal row must be object")
        _exact_keys(row, expected_keys, "proposal row")
        for field in (
            "proposal_ordinal", "decision_source_ts_ns", "decision_arrival_ts_ns", "available_at_ns",
            "completed_data_watermark_source_ts_ns", "direction", "notional_jpy_micros", "max_age_ns",
        ):
            _integer(row[field], f"proposal.{field}")
        if row["proposal_ordinal"] != expected_ordinal or row["direction"] not in {-1, 1} \
                or row["notional_jpy_micros"] <= 0 or row["max_age_ns"] <= 0 \
                or row["action"] != "ENTER":
            raise OracleError("proposal ordinal/direction/size/action invalid")
        instrument = _text(row["instrument"], "proposal instrument")
        _pair(instrument)
        _text(row["worker_key"], "proposal worker_key")
        decision_hash = _digest(row["decision_source_event_sha256"], "decision source event hash")
        prefix_hash = _digest(row["completed_data_prefix_root_sha256"], "completed prefix root")
        if row["available_at_ns"] != row["decision_arrival_ts_ns"] \
                or row["decision_arrival_ts_ns"] < row["decision_source_ts_ns"]:
            raise OracleError("proposal availability chronology invalid")
        available = [event for event in source_rows if event["arrival_ts_ns"] <= row["decision_arrival_ts_ns"]]
        if not available:
            raise OracleError("proposal has no completed-data prefix")
        watermark = max(event["source_ts_ns"] for event in available)
        if row["completed_data_watermark_source_ts_ns"] != watermark \
                or prefix_hash != available[-1]["source_prefix_root_sha256"]:
            raise OracleError("proposal completed-data prefix binding mismatch")
        decision_event = source_by_hash.get(decision_hash)
        if decision_event is None or decision_event not in available \
                or decision_event["source_ts_ns"] != row["decision_source_ts_ns"] \
                or decision_event["instrument"] != instrument \
                or row["decision_source_ts_ns"] > watermark:
            raise OracleError("proposal decision source-event binding mismatch")
        order = (row["decision_arrival_ts_ns"], row["decision_source_ts_ns"], row["proposal_ordinal"])
        if last_decision is not None and order <= last_decision:
            raise OracleError("proposal input order is not strictly monotonic")
        last_decision = order
        economic_lot_key = sha256_bytes(canonical_bytes({
            key: row[key]
            for key in sorted(expected_keys - {"proposal_ordinal"})
        }))
        if economic_lot_key in economic_lot_keys:
            raise OracleError("duplicate economic-lot ticket partition forbidden")
        economic_lot_keys.add(economic_lot_key)
        validated.append(dict(row))
    return {**dict(payload), "rows": validated}


def _validate_policy(payload: Mapping[str, Any], policy_id: str, hash_field: str) -> None:
    if _integer(payload.get("schema_version"), f"{policy_id} schema") != 2 \
            or payload.get("policy_id") != policy_id:
        raise OracleError(f"{policy_id} identity mismatch")
    _validate_embedded(payload, hash_field, policy_id)


def _month_bounds_ns(month_id: str) -> tuple[int, int]:
    if type(month_id) is not str or re.fullmatch(r"[0-9]{4}-(?:0[1-9]|1[0-2])", month_id) is None:
        raise OracleError("month identifier invalid")
    start = datetime(
        int(month_id[:4]), int(month_id[5:]), 1, tzinfo=timezone.utc
    )
    end = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
    return int(start.timestamp()) * 1_000_000_000, int(end.timestamp()) * 1_000_000_000


def _all_intersecting_months(start_ns: int, end_ns: int) -> list[str]:
    if start_ns >= end_ns:
        return []
    current = datetime.fromtimestamp(start_ns // 1_000_000_000, tz=timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    result: list[str] = []
    while int(current.timestamp()) * 1_000_000_000 < end_ns:
        result.append(f"{current.year:04d}-{current.month:02d}")
        current = current.replace(year=current.year + 1, month=1) if current.month == 12 else current.replace(month=current.month + 1)
    return result


def _complete_months(start_ns: int, end_ns: int) -> list[str]:
    return [
        month for month in _all_intersecting_months(start_ns, end_ns)
        if _month_bounds_ns(month)[0] >= start_ns and _month_bounds_ns(month)[1] <= end_ns
    ]


def _validate_policies(
    execution: Mapping[str, Any],
    inventory: Mapping[str, Any],
    accounting: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    _validate_policy(execution, "FROZEN_EXECUTION_POLICY_V2", "execution_policy_sha256")
    _exact_keys(execution, {"schema_version", "policy_id", "arms", "max_trade_quote_staleness_ns", "execution_policy_sha256"}, "execution policy")
    _integer(execution["max_trade_quote_staleness_ns"], "trade quote staleness", minimum=1)
    arms = execution["arms"]
    if type(arms) is not dict or set(arms) != set(ARMS):
        raise OracleError("execution arm set mismatch")
    cost_fields = ("latency_ns", "slippage_micropips_per_side", "commission_ppm_per_side", "financing_ppm_per_day")
    for arm in ARMS:
        spec = arms[arm]
        if type(spec) is not dict:
            raise OracleError("execution arm must be object")
        _exact_keys(spec, set(cost_fields) | {"raw_mid"}, f"execution arm {arm}")
        for field in cost_fields:
            _integer(spec[field], f"{arm}.{field}", minimum=0)
        _boolean(spec["raw_mid"], f"{arm}.raw_mid")
    raw = arms["RAW_SIGNAL"]
    if raw["raw_mid"] is not True or any(raw[field] != 0 for field in cost_fields):
        raise OracleError("RAW arm must be zero-cost midpoint")
    base = arms["EXECUTABLE_BASE"]
    adverse = arms["ADVERSE_STRESS"]
    if base["raw_mid"] is not False or adverse["raw_mid"] is not False \
            or any(adverse[field] < base[field] for field in cost_fields) \
            or not any(adverse[field] > base[field] for field in cost_fields):
        raise OracleError("ADVERSE must be weakly worse in all costs and strictly worse in one")
    _validate_policy(inventory, "FROZEN_INVENTORY_POLICY_V2", "inventory_policy_sha256")
    _exact_keys(
        inventory,
        {
            "schema_version", "policy_id", "max_gross_notional_jpy_micros",
            "max_currency_notional_jpy_micros", "max_open_positions", "same_pair_collision",
            "terminal_liquidation", "inventory_policy_sha256",
        },
        "inventory policy",
    )
    for field in ("max_gross_notional_jpy_micros", "max_currency_notional_jpy_micros", "max_open_positions"):
        _integer(inventory[field], f"inventory.{field}", minimum=1)
    if inventory["same_pair_collision"] != "REJECT_NEW" \
            or _boolean(inventory["terminal_liquidation"], "terminal liquidation") is not True:
        raise OracleError("inventory collision/terminal policy invalid")
    _validate_policy(accounting, "FROZEN_ACCOUNTING_POLICY_V2", "accounting_policy_sha256")
    _exact_keys(
        accounting,
        {
            "schema_version", "policy_id", "jpy_micros_per_yen", "base_microunits_per_unit",
            "max_conversion_staleness_ns", "supported_quote_currencies", "asset_conversion_side",
            "liability_conversion_side", "positive_cost_rounding", "accounting_policy_sha256",
        },
        "accounting policy",
    )
    if _integer(accounting["jpy_micros_per_yen"], "JPY micros") != JPY_MICROS_PER_YEN \
            or _integer(accounting["base_microunits_per_unit"], "base microunits") != BASE_MICROUNITS_PER_UNIT \
            or _integer(accounting["max_conversion_staleness_ns"], "conversion staleness", minimum=1) <= 0 \
            or accounting["supported_quote_currencies"] != ["CAD", "CHF", "JPY", "USD"] \
            or accounting["asset_conversion_side"] != "BID" \
            or accounting["liability_conversion_side"] != "ASK" \
            or accounting["positive_cost_rounding"] != "CEILING":
        raise OracleError("accounting policy invalid")
    _validate_policy(evaluation, "FROZEN_EVALUATION_POLICY_V2", "evaluation_policy_sha256")
    _exact_keys(
        evaluation,
        {
            "schema_version", "policy_id", "period_start_ts_ns", "period_end_ts_ns",
            "initial_equity_jpy_micros", "margin_notional_cap_jpy_micros", "margin_rate_bps",
            "max_gross_to_equity_bps", "cvar_tail_bps", "cluster_window_ns", "full_month_ids",
            "holdout_state", "evaluation_policy_sha256",
        },
        "evaluation policy",
    )
    start = _integer(evaluation["period_start_ts_ns"], "period start", minimum=1)
    end = _integer(evaluation["period_end_ts_ns"], "period end", minimum=1)
    if start >= end:
        raise OracleError("evaluation period invalid")
    for field in ("initial_equity_jpy_micros", "margin_notional_cap_jpy_micros", "margin_rate_bps", "max_gross_to_equity_bps", "cvar_tail_bps", "cluster_window_ns"):
        _integer(evaluation[field], f"evaluation.{field}", minimum=1)
    if evaluation["margin_rate_bps"] > 10_000 or evaluation["cvar_tail_bps"] > 10_000 \
            or evaluation["holdout_state"] != "UNOPENED":
        raise OracleError("evaluation risk/holdout policy invalid")
    full_months = evaluation["full_month_ids"]
    if type(full_months) is not list or any(type(item) is not str for item in full_months) \
            or full_months != _complete_months(start, end):
        raise OracleError("full_month_ids is not the exact complete UTC month set")
    _validate_policy(authority, "FROZEN_PAPER_AUTHORITY_V1", "authority_policy_sha256")
    _exact_keys(authority, {"schema_version", "policy_id", *AUTHORITY.keys(), "authority_policy_sha256"}, "authority policy")
    _validate_authority_exact(
        {key: authority[key] for key in AUTHORITY}, "paper authority"
    )


def _validate_request(
    request: Mapping[str, Any], input_root_fd: int
) -> tuple[
    list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any],
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, int]],
    dict[str, str], str,
]:
    required = {
        "schema_version", "source_blob", "source_manifest", "proposal", "execution_policy",
        "inventory_policy", "accounting_policy", "evaluation_policy", "instrument_registry",
        "authority_policy", "output_directory",
    }
    _exact_keys(request, required, "oracle request")
    if _integer(request["schema_version"], "request schema") != 2:
        raise OracleError("oracle request version mismatch")
    artifacts: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for label in (
        "source_blob", "source_manifest", "proposal", "execution_policy", "inventory_policy",
        "accounting_policy", "evaluation_policy", "instrument_registry", "authority_policy",
    ):
        spec = request[label]
        if type(spec) is not dict:
            raise OracleError(f"{label} artifact spec must be object")
        artifacts[label] = _artifact_bytes(spec, label, input_root_fd)
        hashes[label] = spec["sha256"]
    registry_payload = strict_json(artifacts["instrument_registry"], "instrument registry")
    registry = _validate_instrument_registry(registry_payload)
    source_manifest = strict_json(artifacts["source_manifest"], "source manifest")
    source_rows, books = _parse_source(artifacts["source_blob"], source_manifest, registry_payload, registry)
    proposal = _validate_proposal(strict_json(artifacts["proposal"], "proposal"), source_rows)
    execution = strict_json(artifacts["execution_policy"], "execution policy")
    inventory = strict_json(artifacts["inventory_policy"], "inventory policy")
    accounting = strict_json(artifacts["accounting_policy"], "accounting policy")
    evaluation = strict_json(artifacts["evaluation_policy"], "evaluation policy")
    authority = strict_json(artifacts["authority_policy"], "authority policy")
    _validate_policies(execution, inventory, accounting, evaluation, authority)
    if any(row["instrument"] not in registry for row in proposal["rows"]):
        raise OracleError("proposal instrument outside frozen registry")
    if any(
        row["decision_arrival_ts_ns"] < evaluation["period_start_ts_ns"]
        or row["decision_arrival_ts_ns"] >= evaluation["period_end_ts_ns"]
        for row in proposal["rows"]
    ):
        raise OracleError("proposal decision outside evaluation period")
    output_name = request["output_directory"]
    if type(output_name) is not str or SAFE_COMPONENT_RE.fullmatch(output_name) is None:
        raise OracleError("output directory must be one safe relative component")
    return (
        source_rows, books, proposal, execution, inventory, accounting, evaluation, authority,
        registry, hashes, output_name,
    )


def _market_price(event: Mapping[str, Any], side: str) -> Fraction:
    scale = event["tick_scale"]
    if side == "bid":
        return Fraction(event["bid_ticks"], scale)
    if side == "ask":
        return Fraction(event["ask_ticks"], scale)
    if side == "mid":
        return Fraction(event["bid_ticks"] + event["ask_ticks"], scale * 2)
    raise OracleError("unknown BBO side")


def _execution_price_parts(
    event: Mapping[str, Any],
    direction: int,
    *,
    opening: bool,
    policy: Mapping[str, Any],
    instrument_spec: Mapping[str, int],
) -> tuple[Fraction, int, int]:
    if policy["raw_mid"] is True:
        numerator = (event["bid_ticks"] + event["ask_ticks"]) * PRICE_SUBPIP_SCALE
        denominator = 2 * event["tick_scale"] * PRICE_SUBPIP_SCALE
    else:
        buy = (opening and direction > 0) or (not opening and direction < 0)
        ticks = event["ask_ticks"] if buy else event["bid_ticks"]
        slippage = policy["slippage_micropips_per_side"] * instrument_spec["pip_ticks"]
        numerator = ticks * PRICE_SUBPIP_SCALE + (slippage if buy else -slippage)
        denominator = event["tick_scale"] * PRICE_SUBPIP_SCALE
    if numerator <= 0:
        raise OracleError("execution price became nonpositive")
    return Fraction(numerator, denominator), numerator, denominator


def _latest_causal(
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    instrument: str,
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
    max_staleness_ns: int,
) -> Mapping[str, Any]:
    candidates = [
        event for event in books.get(instrument, ())
        if event["source_ts_ns"] <= source_watermark_ns and event["arrival_ts_ns"] <= arrival_cutoff_ns
    ]
    if not candidates:
        raise OracleError(f"missing causal BBO: {instrument}")
    event = candidates[-1]
    if source_watermark_ns - event["source_ts_ns"] > max_staleness_ns \
            or arrival_cutoff_ns - event["arrival_ts_ns"] > max_staleness_ns \
            or arrival_cutoff_ns - event["source_ts_ns"] > max_staleness_ns:
        raise OracleError(f"stale causal BBO: {instrument}")
    return event


def _arrival_watermark_from_books(
    books: Mapping[str, Sequence[Mapping[str, Any]]], arrival_cutoff_ns: int
) -> int:
    available = [
        event["source_ts_ns"]
        for stream in books.values()
        for event in stream
        if event["arrival_ts_ns"] <= arrival_cutoff_ns
    ]
    if not available:
        raise OracleError("no causal BBO at valuation arrival")
    return max(available)


def _causal_jpy_conversion_paths(
    start_currency: str,
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    registry: Mapping[str, Mapping[str, int]],
    max_staleness_ns: int,
) -> list[tuple[tuple[str, str, str, Mapping[str, Any]], ...]]:
    """Enumerate every simple, fresh registry path from one currency to JPY."""
    if any(instrument not in registry for instrument in books):
        raise OracleError("conversion books outside frozen instrument registry")
    adjacency: defaultdict[
        str, list[tuple[str, str, str, Mapping[str, Any]]]
    ] = defaultdict(list)
    for instrument in sorted(registry):
        if instrument not in books:
            continue
        try:
            event = _latest_causal(
                books,
                instrument,
                source_watermark_ns,
                arrival_cutoff_ns,
                max_staleness_ns,
            )
        except OracleError:
            # A missing, future, or stale quote is not a causal edge.  If its
            # removal leaves zero or multiple routes, the uniqueness gate
            # below still fails closed.
            continue
        base, quote = _pair(instrument)
        adjacency[base].append(
            (quote, "BASE_TO_QUOTE", instrument, event)
        )
        adjacency[quote].append(
            (base, "QUOTE_TO_BASE", instrument, event)
        )

    paths: list[tuple[tuple[str, str, str, Mapping[str, Any]], ...]] = []

    def visit(
        currency: str,
        visited: frozenset[str],
        path: tuple[tuple[str, str, str, Mapping[str, Any]], ...],
    ) -> None:
        if currency == "JPY":
            paths.append(path)
            return
        for edge in sorted(
            adjacency.get(currency, ()),
            key=lambda item: (item[0], item[1], item[2]),
        ):
            destination = edge[0]
            if destination in visited:
                continue
            visit(destination, visited | {destination}, path + (edge,))

    visit(start_currency, frozenset({start_currency}), ())
    return paths


def _convert_currency_node_to_jpy(
    amount: Fraction,
    currency: str,
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    max_staleness_ns: int,
    *,
    registry: Mapping[str, Mapping[str, int]],
) -> Fraction:
    if amount == 0 or currency == "JPY":
        return amount
    paths = _causal_jpy_conversion_paths(
        currency,
        source_watermark_ns,
        arrival_cutoff_ns,
        books,
        registry,
        max_staleness_ns,
    )
    if len(paths) != 1:
        raise OracleError("JPY conversion path must be uniquely causal")
    value = amount
    for _, orientation, _, quote in paths[0]:
        if orientation == "BASE_TO_QUOTE":
            value *= _market_price(quote, "bid" if value > 0 else "ask")
        else:
            value /= _market_price(quote, "ask" if value > 0 else "bid")
    return value


def _convert_to_jpy(
    amount: Fraction,
    currency: str,
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    max_staleness_ns: int,
    *,
    registry: Mapping[str, Mapping[str, int]],
) -> Fraction:
    if amount != 0 and currency != "JPY" \
            and currency not in {"CAD", "CHF", "USD"}:
        raise OracleError(
            f"unsupported quote currency for JPY conversion: {currency}"
        )
    return _convert_currency_node_to_jpy(
        amount,
        currency,
        source_watermark_ns,
        arrival_cutoff_ns,
        books,
        max_staleness_ns,
        registry=registry,
    )


def _asset_micros(value_yen: Fraction) -> int:
    scaled = value_yen * JPY_MICROS_PER_YEN
    return scaled.numerator // scaled.denominator


def _positive_cost_micros(value_micros: Fraction) -> int:
    if value_micros < 0:
        raise OracleError("cost cannot be negative")
    return (value_micros.numerator + value_micros.denominator - 1) // value_micros.denominator


def _outward_signed_currency_micros(value_yen: Fraction) -> int:
    scaled = value_yen * JPY_MICROS_PER_YEN
    if scaled >= 0:
        return (
            scaled.numerator + scaled.denominator - 1
        ) // scaled.denominator
    return scaled.numerator // scaled.denominator


def _scaled_ratio_text(scaled: int) -> str:
    sign = "-" if scaled < 0 else ""
    magnitude = abs(scaled)
    return (
        f"{sign}{magnitude // RATIO_DECIMAL_SCALE}."
        f"{magnitude % RATIO_DECIMAL_SCALE:018d}"
    )


def _ratio_text(numerator: int, denominator: int) -> str:
    """Floor a descriptive/performance ratio to exactly 18 decimals."""
    if denominator <= 0:
        raise OracleError("ratio denominator must be positive")
    return _scaled_ratio_text((numerator * RATIO_DECIMAL_SCALE) // denominator)


def _signed_ratio_text(numerator: int, denominator: int) -> str:
    """Floor a signed return to exactly 18 decimals, including negatives."""
    if denominator <= 0:
        raise OracleError("signed ratio denominator must be positive")
    return _scaled_ratio_text((numerator * RATIO_DECIMAL_SCALE) // denominator)


def _nonnegative_ratio_ceiling_text(numerator: int, denominator: int) -> str:
    """Round a nonnegative risk ratio outward to exactly 18 decimals."""
    if numerator < 0 or denominator <= 0:
        raise OracleError("nonnegative ratio inputs invalid")
    scaled = (
        numerator * RATIO_DECIMAL_SCALE + denominator - 1
    ) // denominator
    return _scaled_ratio_text(scaled)


def _fresh_trade_event(event: Mapping[str, Any], due_arrival_ns: int, max_staleness_ns: int) -> bool:
    return event["arrival_ts_ns"] >= due_arrival_ns \
        and event["arrival_ts_ns"] - event["source_ts_ns"] <= max_staleness_ns \
        and event["arrival_ts_ns"] - due_arrival_ns <= max_staleness_ns


def _first_entry(
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    proposal: Mapping[str, Any],
    latency_ns: int,
    period_end_ns: int,
    max_staleness_ns: int,
) -> Mapping[str, Any] | None:
    due = proposal["decision_arrival_ts_ns"] + latency_ns
    for event in books.get(proposal["instrument"], ()):
        if event["source_ts_ns"] <= proposal["decision_source_ts_ns"] \
                or event["arrival_ts_ns"] < due:
            continue
        if event["arrival_ts_ns"] >= period_end_ns:
            return None
        if not _fresh_trade_event(event, due, max_staleness_ns):
            raise OracleError("first eligible entry quote is stale")
        return event
    return None


def _first_exit(
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    instrument: str,
    entry: Mapping[str, Any],
    due_arrival_ns: int,
    period_end_ns: int,
    max_staleness_ns: int,
) -> Mapping[str, Any] | None:
    for event in books.get(instrument, ()):
        if event["source_ts_ns"] < entry["source_ts_ns"] or event["arrival_ts_ns"] < due_arrival_ns:
            continue
        if event["arrival_ts_ns"] >= period_end_ns:
            return None
        if not _fresh_trade_event(event, due_arrival_ns, max_staleness_ns):
            raise OracleError("first eligible exit quote is stale")
        return event
    return None


def _terminal_event(
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    instrument: str,
    period_end_ns: int,
    max_staleness_ns: int,
) -> Mapping[str, Any]:
    candidates = [
        event for event in books.get(instrument, ())
        if event["source_ts_ns"] < period_end_ns and event["arrival_ts_ns"] < period_end_ns
    ]
    if not candidates:
        raise OracleError(f"terminal BBO missing: {instrument}")
    event = candidates[-1]
    cutoff = period_end_ns - 1
    if cutoff - event["source_ts_ns"] > max_staleness_ns \
            or cutoff - event["arrival_ts_ns"] > max_staleness_ns:
        raise OracleError(f"terminal BBO stale: {instrument}")
    return event


def _signal_id(candidate_key: str, proposal: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes({
        "candidate_key": candidate_key,
        "proposal_ordinal": proposal["proposal_ordinal"],
        "decision_source_ts_ns": proposal["decision_source_ts_ns"],
        "decision_arrival_ts_ns": proposal["decision_arrival_ts_ns"],
        "decision_source_event_sha256": proposal["decision_source_event_sha256"],
        "completed_data_prefix_root_sha256": proposal["completed_data_prefix_root_sha256"],
        "instrument": proposal["instrument"],
        "direction": proposal["direction"],
        "notional_jpy_micros": proposal["notional_jpy_micros"],
        "max_age_ns": proposal["max_age_ns"],
        "worker_key": proposal["worker_key"],
        "detector_code_sha256": provenance["detector_code_sha256"],
        "detector_policy_sha256": provenance["detector_policy_sha256"],
        "generator_policy_sha256": provenance["generator_policy_sha256"],
    }))


def _economic_lot_id(
    candidate_key: str,
    proposal: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    """Identify one economic proposal independently of row/ticket partitioning."""
    return sha256_bytes(canonical_bytes({
        "candidate_key": candidate_key,
        "decision_source_ts_ns": proposal["decision_source_ts_ns"],
        "decision_arrival_ts_ns": proposal["decision_arrival_ts_ns"],
        "decision_source_event_sha256": proposal["decision_source_event_sha256"],
        "completed_data_prefix_root_sha256": proposal["completed_data_prefix_root_sha256"],
        "instrument": proposal["instrument"],
        "direction": proposal["direction"],
        "target_notional_jpy_micros": proposal["notional_jpy_micros"],
        "max_age_ns": proposal["max_age_ns"],
        "worker_key": proposal["worker_key"],
        "detector_code_sha256": provenance["detector_code_sha256"],
        "detector_policy_sha256": provenance["detector_policy_sha256"],
        "generator_policy_sha256": provenance["generator_policy_sha256"],
    }))


def _position_notional_exact_jpy_micros(
    *,
    direction: int,
    units_micros: int,
    price: Fraction,
    quote_currency: str,
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
    opening: bool,
) -> Fraction:
    """Return sign-aware executable position magnitude in exact JPY micros."""
    cash_sign = -direction if opening else direction
    signed_quote_value = (
        Fraction(cash_sign * units_micros, BASE_MICROUNITS_PER_UNIT) * price
    )
    signed_jpy_value = _convert_to_jpy(
        signed_quote_value,
        quote_currency,
        source_watermark_ns,
        arrival_cutoff_ns,
        books,
        accounting["max_conversion_staleness_ns"],
        registry=registry,
    )
    return abs(signed_jpy_value * JPY_MICROS_PER_YEN)


def _risk_notional_micros(exact_jpy_micros: Fraction) -> int:
    """Round a positive risk magnitude outward so caps cannot be understated."""
    return _positive_cost_micros(exact_jpy_micros)


def _units_for_actual_entry(
    proposal: Mapping[str, Any],
    entry: Mapping[str, Any],
    entry_price: Fraction,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
) -> int:
    """Floor units at each arm's actual fill to the common target JPY risk."""
    _, quote_currency = _pair(proposal["instrument"])
    entry_watermark = _arrival_watermark_from_books(books, entry["arrival_ts_ns"])
    per_base_unit_micros = _position_notional_exact_jpy_micros(
        direction=proposal["direction"],
        units_micros=BASE_MICROUNITS_PER_UNIT,
        price=entry_price,
        quote_currency=quote_currency,
        source_watermark_ns=entry_watermark,
        arrival_cutoff_ns=entry["arrival_ts_ns"],
        books=books,
        accounting=accounting,
        registry=registry,
        opening=True,
    )
    if per_base_unit_micros <= 0:
        raise OracleError("position sizing conversion nonpositive")
    exact_units_micros = (
        Fraction(proposal["notional_jpy_micros"], 1)
        * BASE_MICROUNITS_PER_UNIT
        / per_base_unit_micros
    )
    return max(0, exact_units_micros.numerator // exact_units_micros.denominator)


def _units_from_common_entry(
    proposal: Mapping[str, Any],
    entry: Mapping[str, Any],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
) -> int:
    _, quote_currency = _pair(proposal["instrument"])
    common_price = _market_price(entry, "mid")
    entry_watermark = _arrival_watermark_from_books(books, entry["arrival_ts_ns"])
    jpy_per_base_micros = _position_notional_exact_jpy_micros(
        direction=proposal["direction"],
        units_micros=BASE_MICROUNITS_PER_UNIT,
        price=common_price,
        quote_currency=quote_currency,
        source_watermark_ns=entry_watermark,
        arrival_cutoff_ns=entry["arrival_ts_ns"],
        books=books,
        accounting=accounting,
        registry=registry,
        opening=True,
    )
    if jpy_per_base_micros <= 0:
        raise OracleError("position sizing conversion nonpositive")
    exact_units = (
        Fraction(proposal["notional_jpy_micros"], 1)
        * BASE_MICROUNITS_PER_UNIT
        / jpy_per_base_micros
    )
    units = exact_units.numerator // exact_units.denominator
    if units <= 0:
        return 0
    return units


def _common_gross_for_units(
    proposal: Mapping[str, Any],
    common: Mapping[str, Any],
    units_micros: int,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
) -> int:
    """Value the frozen common mid path using a specified integer unit size."""
    if units_micros < 0:
        raise OracleError("common reference units cannot be negative")
    if units_micros == 0:
        return 0
    valuation_arrival_ns = common.get("exit_valuation_arrival_ns")
    if type(valuation_arrival_ns) is not int:
        raise OracleError("common reference valuation clock missing")
    _, quote_currency = _pair(proposal["instrument"])
    quote_pnl = (
        Fraction(
            proposal["direction"] * units_micros,
            BASE_MICROUNITS_PER_UNIT,
        )
        * (
            _market_price(common["exit"], "mid")
            - _market_price(common["entry"], "mid")
        )
    )
    source_watermark = _arrival_watermark_from_books(
        books, valuation_arrival_ns
    )
    return _asset_micros(
        _convert_to_jpy(
            quote_pnl,
            quote_currency,
            source_watermark,
            valuation_arrival_ns,
            books,
            accounting["max_conversion_staleness_ns"],
            registry=registry,
        )
    )


def _common_reference(
    proposal: Mapping[str, Any],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    execution: Mapping[str, Any],
    accounting: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
) -> dict[str, Any] | None:
    entry = _first_entry(
        books, proposal, 0, evaluation["period_end_ts_ns"], execution["max_trade_quote_staleness_ns"]
    )
    if entry is None:
        return None
    due = entry["arrival_ts_ns"] + proposal["max_age_ns"]
    exit_event = _first_exit(
        books, proposal["instrument"], entry, due, evaluation["period_end_ts_ns"],
        execution["max_trade_quote_staleness_ns"],
    )
    exit_reason = "FINITE_MAX_AGE"
    exit_valuation_arrival_ns = exit_event["arrival_ts_ns"] if exit_event is not None else None
    if exit_event is None:
        exit_event = _terminal_event(
            books, proposal["instrument"], evaluation["period_end_ts_ns"],
            execution["max_trade_quote_staleness_ns"],
        )
        if exit_event["arrival_ts_ns"] < entry["arrival_ts_ns"]:
            raise OracleError("terminal reference precedes common entry")
        exit_reason = "TERMINAL_LIQUIDATION"
        exit_valuation_arrival_ns = evaluation["period_end_ts_ns"] - 1
    units = _units_from_common_entry(
        proposal, entry, books, accounting, registry
    )
    if units == 0:
        return {
            "entry": entry,
            "exit": exit_event,
            "exit_reason": exit_reason,
            "exit_valuation_arrival_ns": exit_valuation_arrival_ns,
            "units_micros": 0,
            "gross_pnl_jpy_micros": 0,
        }
    _, quote = _pair(proposal["instrument"])
    quote_pnl = Fraction(proposal["direction"] * units, BASE_MICROUNITS_PER_UNIT) \
        * (_market_price(exit_event, "mid") - _market_price(entry, "mid"))
    if exit_valuation_arrival_ns is None:
        raise OracleError("common exit valuation clock missing")
    exit_watermark = _arrival_watermark_from_books(books, exit_valuation_arrival_ns)
    gross = _asset_micros(_convert_to_jpy(
        quote_pnl,
        quote,
        exit_watermark,
        exit_valuation_arrival_ns,
        books,
        accounting["max_conversion_staleness_ns"],
        registry=registry,
    ))
    return {
        "entry": entry,
        "exit": exit_event,
        "exit_reason": exit_reason,
        "exit_valuation_arrival_ns": exit_valuation_arrival_ns,
        "units_micros": units,
        "gross_pnl_jpy_micros": gross,
    }


def _signed_exposure(
    positions: Sequence[Mapping[str, Any]],
    marked: Sequence[Mapping[str, Any]],
    source_watermark_ns: int,
    arrival_cutoff_ns: int,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    """Value each native currency node independently, then net exact JPY risk."""
    if len(positions) != len(marked):
        raise OracleError("signed exposure position/mark cardinality mismatch")
    exact_yen: defaultdict[str, Fraction] = defaultdict(Fraction)
    for position, mark in zip(positions, marked):
        base, quote = _pair(position["proposal"]["instrument"])
        units_micros = position.get("units_micros")
        if type(units_micros) is not int or units_micros < 0:
            raise OracleError("signed exposure requires actual nonnegative units")
        signed_base_units = Fraction(
            position["proposal"]["direction"] * units_micros,
            BASE_MICROUNITS_PER_UNIT,
        )
        mark_price = mark.get("mark_price")
        if not isinstance(mark_price, Fraction) or mark_price <= 0:
            raise OracleError("signed exposure requires exact positive mark price")
        signed_quote_cash = -signed_base_units * mark_price
        for currency, native_amount in (
            (base, signed_base_units),
            (quote, signed_quote_cash),
        ):
            exact_yen[currency] += _convert_currency_node_to_jpy(
                native_amount,
                currency,
                source_watermark_ns,
                arrival_cutoff_ns,
                books,
                accounting["max_conversion_staleness_ns"],
                registry=registry,
            )
    rounded = {
        currency: _outward_signed_currency_micros(amount_yen)
        for currency, amount_yen in sorted(exact_yen.items())
    }
    return {currency: amount for currency, amount in rounded.items() if amount != 0}


def _position_value(
    position: Mapping[str, Any],
    mark_event: Mapping[str, Any],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
    *,
    valuation_source_watermark_ns: int | None = None,
    valuation_arrival_ns: int | None = None,
) -> dict[str, Any]:
    proposal = position["proposal"]
    policy = position["policy"]
    entry_price = position["entry_price"]
    exit_price, _, _ = _execution_price_parts(
        mark_event,
        proposal["direction"],
        opening=False,
        policy=policy,
        instrument_spec=registry[proposal["instrument"]],
    )
    units = Fraction(position["units_micros"], BASE_MICROUNITS_PER_UNIT)
    _, quote = _pair(proposal["instrument"])
    quote_pnl = proposal["direction"] * units * (exit_price - entry_price)
    arrival_cutoff = (
        mark_event["arrival_ts_ns"]
        if valuation_arrival_ns is None
        else valuation_arrival_ns
    )
    source_watermark = (
        _arrival_watermark_from_books(books, arrival_cutoff)
        if valuation_source_watermark_ns is None
        else valuation_source_watermark_ns
    )
    executable_exact_micros = _convert_to_jpy(
        quote_pnl,
        quote,
        source_watermark,
        arrival_cutoff,
        books,
        accounting["max_conversion_staleness_ns"],
        registry=registry,
    ) * JPY_MICROS_PER_YEN
    executable = (
        executable_exact_micros.numerator
        // executable_exact_micros.denominator
    )
    elapsed = arrival_cutoff - position["entry"]["arrival_ts_ns"]
    if elapsed < 0:
        raise OracleError("mark arrival precedes entry")
    marked_notional_exact = _position_notional_exact_jpy_micros(
        direction=proposal["direction"],
        units_micros=position["units_micros"],
        price=exit_price,
        quote_currency=quote,
        source_watermark_ns=source_watermark,
        arrival_cutoff_ns=arrival_cutoff,
        books=books,
        accounting=accounting,
        registry=registry,
        opening=False,
    )
    entry_notional_exact = position["entry_notional_exact_jpy_micros"]
    # Each executable side is charged from that side's actual causal notional.
    entry_commission_exact = (
        entry_notional_exact * policy["commission_ppm_per_side"] / 1_000_000
    )
    exit_commission_exact = (
        marked_notional_exact * policy["commission_ppm_per_side"] / 1_000_000
    )
    entry_commission = _positive_cost_micros(entry_commission_exact)
    exit_commission = _positive_cost_micros(exit_commission_exact)
    commission = entry_commission + exit_commission
    financing_exact = (
        entry_notional_exact
        * policy["financing_ppm_per_day"]
        * elapsed
        / (DAY_NS * 1_000_000)
    )
    financing = _positive_cost_micros(financing_exact)
    economic_net_exact = (
        executable_exact_micros
        - entry_commission_exact
        - exit_commission_exact
        - financing_exact
    )
    return {
        "mark_price": exit_price,
        "executable_pnl_jpy_micros": executable,
        "commission_jpy_micros": commission,
        "financing_jpy_micros": financing,
        "net_pnl_jpy_micros": executable - commission - financing,
        "elapsed_ns": elapsed,
        "marked_notional_jpy_micros": _risk_notional_micros(marked_notional_exact),
        "financing_basis_notional_jpy_micros": _risk_notional_micros(
            entry_notional_exact
        ),
        "economic_net_pnl_jpy_micros_numerator": economic_net_exact.numerator,
        "economic_net_pnl_jpy_micros_denominator": economic_net_exact.denominator,
    }


def _close_position(
    position: dict[str, Any],
    exit_event: Mapping[str, Any],
    exit_reason: str,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
    execution_hash: str,
    *,
    valuation_source_watermark_ns: int | None = None,
    valuation_arrival_ns: int | None = None,
) -> dict[str, Any]:
    values = _position_value(
        position,
        exit_event,
        books,
        accounting,
        registry,
        valuation_source_watermark_ns=valuation_source_watermark_ns,
        valuation_arrival_ns=valuation_arrival_ns,
    )
    proposal = position["proposal"]
    common = position["common"]
    exit_price, exit_num, exit_den = _execution_price_parts(
        exit_event,
        proposal["direction"],
        opening=False,
        policy=position["policy"],
        instrument_spec=registry[proposal["instrument"]],
    )
    del exit_price
    common_gross = common["gross_pnl_jpy_micros"]
    net = values["net_pnl_jpy_micros"]
    executable = values["executable_pnl_jpy_micros"]
    arm_units_common_gross = _common_gross_for_units(
        proposal,
        common,
        position["units_micros"],
        books,
        accounting,
        registry,
    )
    fill_sizing_drag = common_gross - arm_units_common_gross
    execution_drag = arm_units_common_gross - executable
    return {
        "record_type": "ORACLE_DISPOSITION",
        "arm": position["arm"],
        "signal_id": position["signal_id"],
        "proposal_ordinal": proposal["proposal_ordinal"],
        "instrument": proposal["instrument"],
        "direction": proposal["direction"],
        "status": "FILLED_CLOSED",
        "entry_disposition": "FILLED",
        "exit_disposition": exit_reason,
        "action_transitions": ["ENTER", "EXIT"],
        "notional_jpy_micros": proposal["notional_jpy_micros"],
        "target_notional_jpy_micros": proposal["notional_jpy_micros"],
        "filled_notional_jpy_micros": position["entry_notional_jpy_micros"],
        "financing_basis_notional_jpy_micros": values[
            "financing_basis_notional_jpy_micros"
        ],
        "marked_or_exit_notional_jpy_micros": values[
            "marked_notional_jpy_micros"
        ],
        "exit_notional_jpy_micros": values["marked_notional_jpy_micros"],
        "units_micros": position["units_micros"],
        "economic_lot_id": position["economic_lot_id"],
        "common_entry_source_event_sha256": common["entry"]["source_event_sha256"],
        "common_exit_source_event_sha256": common["exit"]["source_event_sha256"],
        "common_gross_pnl_jpy_micros": common_gross,
        "arm_units_common_gross_pnl_jpy_micros": arm_units_common_gross,
        "entry_price_numerator": position["entry_price_numerator"],
        "entry_price_denominator": position["entry_price_denominator"],
        "exit_price_numerator": exit_num,
        "exit_price_denominator": exit_den,
        "entry_source_event_sha256": position["entry"]["source_event_sha256"],
        "entry_source_ts_ns": position["entry"]["source_ts_ns"],
        "entry_arrival_ts_ns": position["entry"]["arrival_ts_ns"],
        "exit_source_event_sha256": exit_event["source_event_sha256"],
        "exit_source_ts_ns": exit_event["source_ts_ns"],
        # The executable quote provenance may predate a margin/terminal
        # valuation clock.  Record the actual close clock here and retain the
        # source quote's arrival separately in exit_source_reference.
        "exit_arrival_ts_ns": (
            exit_event["arrival_ts_ns"]
            if valuation_arrival_ns is None
            else valuation_arrival_ns
        ),
        "elapsed_ns": values["elapsed_ns"],
        "executable_pnl_before_direct_cost_jpy_micros": executable,
        "fill_sizing_drag_jpy_micros": fill_sizing_drag,
        "latency_spread_slippage_drag_jpy_micros": execution_drag,
        "commission_jpy_micros": values["commission_jpy_micros"],
        "financing_jpy_micros": values["financing_jpy_micros"],
        "realized_cost_jpy_micros": common_gross - net,
        "admission_opportunity_drag_jpy_micros": 0,
        "net_pnl_jpy_micros": net,
        "economic_net_pnl_jpy_micros_numerator": values[
            "economic_net_pnl_jpy_micros_numerator"
        ],
        "economic_net_pnl_jpy_micros_denominator": values[
            "economic_net_pnl_jpy_micros_denominator"
        ],
        "signed_currency_exposure_after_entry_jpy_micros": position["signed_exposure_after_entry"],
        "gross_open_notional_after_entry_jpy_micros": position["gross_after_entry"],
        "marked_equity_after_entry_jpy_micros": position["marked_equity_after_entry"],
        "required_margin_after_entry_jpy_micros": position["required_margin_after_entry"],
        "free_margin_after_entry_jpy_micros": position["free_margin_after_entry"],
        "entry_source_reference": {
            "provider_id": position["entry"]["provider_id"],
            "source_event_sha256": position["entry"]["source_event_sha256"],
            "source_ts_ns": position["entry"]["source_ts_ns"],
            "arrival_ts_ns": position["entry"]["arrival_ts_ns"],
            "execution_policy_sha256": execution_hash,
        },
        "exit_source_reference": {
            "provider_id": exit_event["provider_id"],
            "source_event_sha256": exit_event["source_event_sha256"],
            "source_ts_ns": exit_event["source_ts_ns"],
            "arrival_ts_ns": exit_event["arrival_ts_ns"],
            "execution_policy_sha256": execution_hash,
        },
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_order_count": 0,
    }


def _rejected(
    proposal: Mapping[str, Any],
    signal_id: str,
    economic_lot_id: str,
    arm: str,
    reason: str,
    common: Mapping[str, Any] | None,
) -> dict[str, Any]:
    gross = 0 if common is None else common["gross_pnl_jpy_micros"]
    sizing_drag = gross if reason == "SIZE_ROUNDED_TO_ZERO" else 0
    latency_drag = gross if reason == "NO_CAUSAL_FILL" else 0
    admission_drag = gross if reason in {
        "SAME_PAIR_COLLISION_REJECTED",
        "GROSS_CAP_REJECTED",
        "POSITION_CAP_REJECTED",
        "CURRENCY_CAP_REJECTED",
        "MARGIN_ENTRY_REJECTED",
        "ACCOUNT_HALTED",
    } else 0
    return {
        "record_type": "ORACLE_DISPOSITION",
        "arm": arm,
        "signal_id": signal_id,
        "proposal_ordinal": proposal["proposal_ordinal"],
        "instrument": proposal["instrument"],
        "direction": proposal["direction"],
        "status": reason,
        "entry_disposition": reason,
        "exit_disposition": "NOT_APPLICABLE",
        "action_transitions": ["NO_ENTRY"],
        "notional_jpy_micros": proposal["notional_jpy_micros"],
        "target_notional_jpy_micros": proposal["notional_jpy_micros"],
        "filled_notional_jpy_micros": 0,
        "financing_basis_notional_jpy_micros": 0,
        "marked_or_exit_notional_jpy_micros": 0,
        "exit_notional_jpy_micros": 0,
        "units_micros": 0,
        "economic_lot_id": economic_lot_id,
        "common_entry_source_event_sha256": None if common is None else common["entry"]["source_event_sha256"],
        "common_exit_source_event_sha256": None if common is None else common["exit"]["source_event_sha256"],
        "common_gross_pnl_jpy_micros": gross,
        "arm_units_common_gross_pnl_jpy_micros": (
            0 if reason in {"NO_CAUSAL_FILL", "SIZE_ROUNDED_TO_ZERO"} else gross
        ),
        "executable_pnl_before_direct_cost_jpy_micros": 0,
        "fill_sizing_drag_jpy_micros": sizing_drag,
        "latency_spread_slippage_drag_jpy_micros": latency_drag,
        "commission_jpy_micros": 0,
        "financing_jpy_micros": 0,
        "realized_cost_jpy_micros": 0,
        "admission_opportunity_drag_jpy_micros": admission_drag,
        "net_pnl_jpy_micros": 0,
        "economic_net_pnl_jpy_micros_numerator": 0,
        "economic_net_pnl_jpy_micros_denominator": 1,
        "signed_currency_exposure_after_entry_jpy_micros": {},
        "gross_open_notional_after_entry_jpy_micros": 0,
        "marked_equity_after_entry_jpy_micros": None,
        "required_margin_after_entry_jpy_micros": 0,
        "free_margin_after_entry_jpy_micros": None,
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_order_count": 0,
    }


def _watermark_for_arrival(source_rows: Sequence[Mapping[str, Any]], arrival_ns: int) -> int:
    available = [row["source_ts_ns"] for row in source_rows if row["arrival_ts_ns"] <= arrival_ns]
    if not available:
        raise OracleError("no causal source watermark")
    return max(available)


def _mark_state(
    active: Sequence[Mapping[str, Any]],
    closed: Sequence[Mapping[str, Any]],
    arrival_ns: int,
    source_rows: Sequence[Mapping[str, Any]],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
    evaluation: Mapping[str, Any],
    trade_staleness_ns: int,
) -> dict[str, Any]:
    watermark = _watermark_for_arrival(source_rows, arrival_ns)
    realized = sum(record["net_pnl_jpy_micros"] for record in closed)
    unrealized = 0
    marked_positions: list[dict[str, Any]] = []
    for position in active:
        mark = _latest_causal(
            books,
            position["proposal"]["instrument"],
            watermark,
            arrival_ns,
            trade_staleness_ns,
        )
        values = _position_value(
            position,
            mark,
            books,
            accounting,
            registry,
            valuation_source_watermark_ns=watermark,
            valuation_arrival_ns=arrival_ns,
        )
        unrealized += values["net_pnl_jpy_micros"]
        marked_positions.append({
            "risk_notional_jpy_micros": values["marked_notional_jpy_micros"],
            "mark_price": values["mark_price"],
        })
    equity = evaluation["initial_equity_jpy_micros"] + realized + unrealized
    gross = sum(
        position["risk_notional_jpy_micros"] for position in marked_positions
    )
    required = _positive_cost_micros(
        Fraction(gross * evaluation["margin_rate_bps"], 10_000)
    )
    free = equity - required
    ratio_ok = equity > 0 and gross * 10_000 <= equity * evaluation["max_gross_to_equity_bps"]
    return {
        "arrival_ts_ns": arrival_ns,
        "source_watermark_ts_ns": watermark,
        "marked_equity_jpy_micros": equity,
        "gross_notional_jpy_micros": gross,
        "required_margin_jpy_micros": required,
        "free_margin_jpy_micros": free,
        "signed_currency_exposure_jpy_micros": _signed_exposure(
            active,
            marked_positions,
            watermark,
            arrival_ns,
            books,
            accounting,
            registry,
        ),
        "margin_ratio_pass": ratio_ok,
    }


def _risk_closeout_reason(
    mark: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> str | None:
    if mark["marked_equity_jpy_micros"] <= 0 \
            or mark["free_margin_jpy_micros"] < 0 \
            or mark["margin_ratio_pass"] is not True \
            or mark["gross_notional_jpy_micros"] \
                > evaluation["margin_notional_cap_jpy_micros"]:
        return "MARGIN_CLOSEOUT"
    if mark["gross_notional_jpy_micros"] \
            > inventory["max_gross_notional_jpy_micros"] \
            or max((
                abs(value)
                for value in mark["signed_currency_exposure_jpy_micros"].values()
            ), default=0) > inventory["max_currency_notional_jpy_micros"]:
        return "INVENTORY_CAP_CLOSEOUT"
    return None


def _simulate_arm(
    arm: str,
    source_rows: Sequence[Mapping[str, Any]],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    proposal_root: Mapping[str, Any],
    common: Mapping[int, Mapping[str, Any] | None],
    execution: Mapping[str, Any],
    inventory: Mapping[str, Any],
    accounting: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    policy = execution["arms"][arm]
    max_trade_stale = execution["max_trade_quote_staleness_ns"]
    plans: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    results: dict[int, dict[str, Any]] = {}
    signals: dict[int, str] = {}
    economic_lots: dict[int, str] = {}
    for proposal in proposal_root["rows"]:
        ordinal = proposal["proposal_ordinal"]
        signal_id = _signal_id(proposal_root["candidate_key"], proposal, proposal_root["provenance"])
        economic_lot_id = _economic_lot_id(
            proposal_root["candidate_key"], proposal, proposal_root["provenance"]
        )
        signals[ordinal] = signal_id
        economic_lots[ordinal] = economic_lot_id
        common_item = common[ordinal]
        if common_item is None:
            results[ordinal] = _rejected(
                proposal, signal_id, economic_lot_id, arm,
                "NO_COMMON_CAUSAL_PATH", None,
            )
            continue
        entry = _first_entry(
            books,
            proposal,
            policy["latency_ns"],
            evaluation["period_end_ts_ns"],
            max_trade_stale,
        )
        if entry is None:
            results[ordinal] = _rejected(
                proposal, signal_id, economic_lot_id, arm,
                "NO_CAUSAL_FILL", common_item,
            )
            continue
        plans[entry["source_event_sha256"]].append({
            "proposal": proposal,
            "signal_id": signal_id,
            "economic_lot_id": economic_lot_id,
            "common": common_item,
            "entry": entry,
        })
    active: list[dict[str, Any]] = []
    closed_records: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    risk_timeline: list[dict[str, Any]] = []
    halted = False

    def close_due(event: Mapping[str, Any]) -> None:
        nonlocal active
        due_positions = [
            position for position in active
            if position["proposal"]["instrument"] == event["instrument"]
            and event["arrival_ts_ns"] >= position["due_arrival_ns"]
            and event["source_ts_ns"] >= position["entry"]["source_ts_ns"]
        ]
        for position in sorted(due_positions, key=lambda item: item["proposal"]["proposal_ordinal"]):
            if not _fresh_trade_event(event, position["due_arrival_ns"], max_trade_stale):
                raise OracleError("scheduled exit quote stale")
            record = _close_position(
                position, event, "FINITE_MAX_AGE", books, accounting, registry,
                execution["execution_policy_sha256"],
            )
            position["closed_record"] = record
            results[position["proposal"]["proposal_ordinal"]] = record
            closed_records.append(record)
            active.remove(position)

    def closeout_all(arrival_ns: int, reason: str) -> None:
        nonlocal active
        watermark = _watermark_for_arrival(source_rows, arrival_ns)
        for position in sorted(active, key=lambda item: item["proposal"]["proposal_ordinal"]):
            quote = _latest_causal(
                books,
                position["proposal"]["instrument"],
                watermark,
                arrival_ns,
                max_trade_stale,
            )
            record = _close_position(
                position, quote, reason, books, accounting, registry,
                execution["execution_policy_sha256"],
                valuation_source_watermark_ns=watermark,
                valuation_arrival_ns=arrival_ns,
            )
            position["closed_record"] = record
            results[position["proposal"]["proposal_ordinal"]] = record
            closed_records.append(record)
        active = []

    period_start = evaluation["period_start_ts_ns"]
    period_end = evaluation["period_end_ts_ns"]
    terminal_arrival_ns = period_end - 1
    period_events = tuple(
        event for event in source_rows
        if period_start <= event["arrival_ts_ns"] < period_end
    )
    events_by_arrival: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in period_events:
        events_by_arrival[event["arrival_ts_ns"]].append(event)
    # Risk is defined at every causal market event and at every accounting
    # boundary, even when no quote arrives at that exact clock.  In particular,
    # accrued financing can make free margin breach at a UTC month boundary or
    # immediately before terminal liquidation.  Omitting those clocks used to
    # overstate the minimum free margin by one JPY micro in the 28B fixture and
    # could miss a real boundary closeout entirely.
    boundary_clocks = {terminal_arrival_ns}
    for month in _all_intersecting_months(period_start, period_end):
        _, month_end = _month_bounds_ns(month)
        checkpoint = min(period_end, month_end) - 1
        if checkpoint >= period_start:
            boundary_clocks.add(checkpoint)
    attribution_clocks = {
        item["exit_valuation_arrival_ns"]
        for item in common.values()
        if item is not None
        and period_start <= item["exit_valuation_arrival_ns"] < period_end
    }
    event_clocks = sorted({*events_by_arrival, *boundary_clocks, *attribution_clocks})
    for arrival_ns in event_clocks:
        batch = sorted(
            events_by_arrival.get(arrival_ns, ()),
            key=lambda event: (
                event["source_ts_ns"], event["provider_id"], event["sequence"]
            ),
        )
        # Fixed tie-break: every close eligible at this arrival is processed
        # before any entry at this arrival.  Provider/instrument lexical order
        # therefore cannot change capacity release.
        for event in batch:
            close_due(event)
        mark = _mark_state(
            active, closed_records, arrival_ns, source_rows, books,
            accounting, registry, evaluation, max_trade_stale,
        )
        risk_timeline.append(mark)
        closeout_reason = _risk_closeout_reason(mark, evaluation, inventory)
        if closeout_reason is not None:
            if active:
                closeout_all(arrival_ns, closeout_reason)
            halted = True
            risk_timeline.append(_mark_state(
                active, closed_records, arrival_ns, source_rows, books,
                accounting, registry, evaluation, max_trade_stale,
            ))
        batch_plans = [
            plan
            for event in batch
            for plan in plans.get(event["source_event_sha256"], ())
        ]
        for plan in sorted(batch_plans, key=lambda item: item["proposal"]["proposal_ordinal"]):
            proposal = plan["proposal"]
            ordinal = proposal["proposal_ordinal"]
            event = plan["entry"]
            if halted:
                results[ordinal] = _rejected(
                    proposal, plan["signal_id"], plan["economic_lot_id"], arm,
                    "ACCOUNT_HALTED", plan["common"],
                )
                continue
            if any(position["proposal"]["instrument"] == proposal["instrument"] for position in active):
                results[ordinal] = _rejected(
                    proposal, plan["signal_id"], plan["economic_lot_id"], arm,
                    "SAME_PAIR_COLLISION_REJECTED", plan["common"],
                )
                continue
            entry_price, entry_num, entry_den = _execution_price_parts(
                event,
                proposal["direction"],
                opening=True,
                policy=policy,
                instrument_spec=registry[proposal["instrument"]],
            )
            units_micros = _units_for_actual_entry(
                proposal, event, entry_price, books, accounting, registry
            )
            if units_micros == 0:
                results[ordinal] = _rejected(
                    proposal, plan["signal_id"], plan["economic_lot_id"], arm,
                    "SIZE_ROUNDED_TO_ZERO", plan["common"],
                )
                continue
            _, quote_currency = _pair(proposal["instrument"])
            entry_watermark = _arrival_watermark_from_books(
                books, event["arrival_ts_ns"]
            )
            entry_notional_exact = _position_notional_exact_jpy_micros(
                direction=proposal["direction"],
                units_micros=units_micros,
                price=entry_price,
                quote_currency=quote_currency,
                source_watermark_ns=entry_watermark,
                arrival_cutoff_ns=event["arrival_ts_ns"],
                books=books,
                accounting=accounting,
                registry=registry,
                opening=True,
            )
            position = {
                "arm": arm,
                "proposal": proposal,
                "signal_id": plan["signal_id"],
                "common": plan["common"],
                "entry": event,
                "entry_price": entry_price,
                "entry_price_numerator": entry_num,
                "entry_price_denominator": entry_den,
                "units_micros": units_micros,
                "economic_lot_id": plan["economic_lot_id"],
                "entry_notional_exact_jpy_micros": entry_notional_exact,
                "entry_notional_jpy_micros": _risk_notional_micros(
                    entry_notional_exact
                ),
                "policy": policy,
                # Finite max-age is measured from the actual fill-arrival
                # clock. Entry latency has already elapsed and must not be
                # charged a second time to the holding horizon.
                "due_arrival_ns": event["arrival_ts_ns"] + proposal["max_age_ns"],
            }
            tentative = [*active, position]
            if len(tentative) > inventory["max_open_positions"]:
                results[ordinal] = _rejected(
                    proposal, plan["signal_id"], plan["economic_lot_id"], arm,
                    "POSITION_CAP_REJECTED", plan["common"],
                )
                continue
            mark = _mark_state(
                tentative, closed_records, arrival_ns, source_rows,
                books, accounting, registry, evaluation, max_trade_stale,
            )
            gross = mark["gross_notional_jpy_micros"]
            exposure = mark["signed_currency_exposure_jpy_micros"]
            if gross > inventory["max_gross_notional_jpy_micros"]:
                results[ordinal] = _rejected(
                    proposal, plan["signal_id"], plan["economic_lot_id"], arm,
                    "GROSS_CAP_REJECTED", plan["common"],
                )
                continue
            if max((abs(value) for value in exposure.values()), default=0) > inventory["max_currency_notional_jpy_micros"]:
                results[ordinal] = _rejected(
                    proposal, plan["signal_id"], plan["economic_lot_id"], arm,
                    "CURRENCY_CAP_REJECTED", plan["common"],
                )
                continue
            if _risk_closeout_reason(mark, evaluation, inventory) \
                    == "MARGIN_CLOSEOUT":
                results[ordinal] = _rejected(
                    proposal, plan["signal_id"], plan["economic_lot_id"], arm,
                    "MARGIN_ENTRY_REJECTED", plan["common"],
                )
                continue
            position["signed_exposure_after_entry"] = exposure
            position["gross_after_entry"] = gross
            position["marked_equity_after_entry"] = mark["marked_equity_jpy_micros"]
            position["required_margin_after_entry"] = mark["required_margin_jpy_micros"]
            position["free_margin_after_entry"] = mark["free_margin_jpy_micros"]
            active.append(position)
            positions.append(position)
            risk_timeline.append(mark)
    for ordinal, signal_id in signals.items():
        if ordinal not in results and not any(position["proposal"]["proposal_ordinal"] == ordinal for position in active):
            proposal = proposal_root["rows"][ordinal - 1]
            results[ordinal] = _rejected(
                proposal, signal_id, economic_lots[ordinal], arm,
                "NO_CAUSAL_FILL", common[ordinal],
            )
    if active:
        terminal_watermark = _watermark_for_arrival(source_rows, terminal_arrival_ns)
        frozen_terminal = [
            (position, _terminal_event(
                books,
                position["proposal"]["instrument"],
                evaluation["period_end_ts_ns"],
                max_trade_stale,
            ))
            for position in sorted(
                active, key=lambda item: item["proposal"]["proposal_ordinal"]
            )
        ]
        for position, terminal in frozen_terminal:
            if terminal["arrival_ts_ns"] < position["entry"]["arrival_ts_ns"]:
                raise OracleError("terminal quote precedes entry")
        pre_liquidation = _mark_state(
            active, closed_records, terminal_arrival_ns, source_rows, books,
            accounting, registry, evaluation, max_trade_stale,
        )
        risk_timeline.append(pre_liquidation)
        terminal_reason = _risk_closeout_reason(
            pre_liquidation, evaluation, inventory
        ) or "TERMINAL_LIQUIDATION"
        halted = halted or terminal_reason in {
            "MARGIN_CLOSEOUT", "INVENTORY_CAP_CLOSEOUT"
        }
        for position, terminal in frozen_terminal:
            record = _close_position(
                position, terminal, terminal_reason, books, accounting, registry,
                execution["execution_policy_sha256"],
                valuation_source_watermark_ns=terminal_watermark,
                valuation_arrival_ns=terminal_arrival_ns,
            )
            position["closed_record"] = record
            results[position["proposal"]["proposal_ordinal"]] = record
            closed_records.append(record)
            active.remove(position)
        post_liquidation = _mark_state(
            [], closed_records, terminal_arrival_ns, source_rows, books, accounting,
            registry, evaluation, max_trade_stale,
        )
        if pre_liquidation["marked_equity_jpy_micros"] \
                != post_liquidation["marked_equity_jpy_micros"]:
            raise OracleError("terminal valuation changed while liquidating")
        risk_timeline.append(post_liquidation)
    if set(results) != {row["proposal_ordinal"] for row in proposal_root["rows"]}:
        raise OracleError("not every proposal received exactly one arm disposition")
    return [results[index] for index in sorted(results)], positions, risk_timeline


def _equity_at(
    positions: Sequence[Mapping[str, Any]],
    cutoff_ns: int,
    source_rows: Sequence[Mapping[str, Any]],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
    evaluation: Mapping[str, Any],
    trade_staleness_ns: int,
) -> int:
    equity = evaluation["initial_equity_jpy_micros"]
    if not positions:
        return equity
    available = [row for row in source_rows if row["arrival_ts_ns"] <= cutoff_ns]
    if not available:
        return equity
    watermark = max(row["source_ts_ns"] for row in available)
    for position in positions:
        if position["entry"]["arrival_ts_ns"] > cutoff_ns:
            continue
        record = position.get("closed_record")
        if record is not None and record["exit_arrival_ts_ns"] <= cutoff_ns:
            equity += record["net_pnl_jpy_micros"]
            continue
        mark = _latest_causal(
            books,
            position["proposal"]["instrument"],
            watermark,
            cutoff_ns,
            trade_staleness_ns,
        )
        equity += _position_value(
            position,
            mark,
            books,
            accounting,
            registry,
            valuation_source_watermark_ns=watermark,
            valuation_arrival_ns=cutoff_ns,
        )["net_pnl_jpy_micros"]
    return equity


def _cluster_metrics(
    records: Sequence[Mapping[str, Any]], evaluation: Mapping[str, Any]
) -> tuple[int, int, str, list[dict[str, Any]]]:
    """Aggregate simultaneous currency-connected tickets before tail statistics."""
    by_bucket: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] == "FILLED_CLOSED":
            economic_lot_id = record.get("economic_lot_id")
            if type(economic_lot_id) is not str or SHA256_RE.fullmatch(economic_lot_id) is None:
                raise OracleError("cluster record economic lot identity invalid")
            numerator = record.get("economic_net_pnl_jpy_micros_numerator")
            denominator = record.get("economic_net_pnl_jpy_micros_denominator")
            if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
                raise OracleError("cluster exact economic return fraction invalid")
            by_bucket[record["entry_arrival_ts_ns"] // evaluation["cluster_window_ns"]].append(record)
    observations: list[dict[str, Any]] = []
    exact_observations: list[tuple[Fraction, str]] = []
    initial_equity = evaluation["initial_equity_jpy_micros"]
    for bucket, bucket_records in sorted(by_bucket.items()):
        parent: dict[str, str] = {}

        def find(node: str) -> str:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for record in bucket_records:
            base, quote = _pair(record["instrument"])
            union(base, quote)
        grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in bucket_records:
            grouped[find(_pair(record["instrument"])[0])].append(record)
        for _, component in sorted(grouped.items()):
            nodes = sorted({node for record in component for node in _pair(record["instrument"])})
            exact_pnl = sum(
                (
                    Fraction(
                        record["economic_net_pnl_jpy_micros_numerator"],
                        record["economic_net_pnl_jpy_micros_denominator"],
                    )
                    for record in component
                ),
                Fraction(0, 1),
            )
            risk_pnl = exact_pnl.numerator // exact_pnl.denominator
            ledger_pnl = sum(
                record["net_pnl_jpy_micros"] for record in component
            )
            identity = {"time_bucket": bucket, "currency_nodes": nodes}
            cluster_id = sha256_bytes(canonical_bytes(identity))
            observations.append({
                "cluster_id": cluster_id,
                "time_bucket": bucket,
                "currency_nodes": nodes,
                "source_signal_set_sha256": sha256_bytes(canonical_bytes(sorted({record["economic_lot_id"] for record in component}))),
                "ledger_net_pnl_jpy_micros": ledger_pnl,
                "cluster_risk_net_pnl_jpy_micros": risk_pnl,
                "signed_return": _signed_ratio_text(risk_pnl, initial_equity),
            })
            exact_observations.append((exact_pnl, cluster_id))
    ordered = sorted(exact_observations, key=lambda item: (item[0], item[1]))
    tail_count = max(1, (len(ordered) * evaluation["cvar_tail_bps"] + 9_999) // 10_000) if ordered else 0
    tail = ordered[:tail_count]
    tail_exact = sum((item[0] for item in tail), Fraction(0, 1))
    cvar_exact = tail_exact / tail_count if tail_count else Fraction(0, 1)
    cvar_jpy = cvar_exact.numerator // cvar_exact.denominator if tail_count else 0
    cvar_return = _signed_ratio_text(
        cvar_exact.numerator,
        cvar_exact.denominator * initial_equity,
    ) if tail_count else "0.000000000000000000"
    observations.sort(key=lambda row: row["cluster_id"])
    return len(observations), cvar_jpy, cvar_return, observations


def _arm_metrics(
    records: Sequence[Mapping[str, Any]],
    positions: Sequence[Mapping[str, Any]],
    risk_timeline: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, int]],
    evaluation: Mapping[str, Any],
    trade_staleness_ns: int,
) -> dict[str, Any]:
    initial = evaluation["initial_equity_jpy_micros"]
    filled = [record for record in records if record["status"] == "FILLED_CLOSED"]
    rejected = [record for record in records if record["status"] != "FILLED_CLOSED"]
    net = sum(record["net_pnl_jpy_micros"] for record in filled)
    gross = sum(record["common_gross_pnl_jpy_micros"] for record in records)
    cost = sum(record["realized_cost_jpy_micros"] for record in filled)
    sizing_drag = sum(record["fill_sizing_drag_jpy_micros"] for record in records)
    latency_drag = sum(record["latency_spread_slippage_drag_jpy_micros"] for record in records)
    direct_cost = sum(
        record["commission_jpy_micros"] + record["financing_jpy_micros"]
        for record in filled
    )
    admission_drag = sum(record["admission_opportunity_drag_jpy_micros"] for record in rejected)
    decomposed_drag = sizing_drag + latency_drag + direct_cost + admission_drag
    if decomposed_drag != gross - net:
        raise OracleError("arm drag attribution does not reconcile to common gross minus net")
    monthly: list[dict[str, Any]] = []
    boundary_equities: list[tuple[int, int]] = []
    for month in _all_intersecting_months(evaluation["period_start_ts_ns"], evaluation["period_end_ts_ns"]):
        month_start, month_end = _month_bounds_ns(month)
        segment_start = max(month_start, evaluation["period_start_ts_ns"])
        segment_end = min(month_end, evaluation["period_end_ts_ns"])
        start_equity = _equity_at(
            positions, segment_start - 1, source_rows, books, accounting,
            registry, evaluation, trade_staleness_ns,
        )
        end_equity = _equity_at(
            positions, segment_end - 1, source_rows, books, accounting,
            registry, evaluation, trade_staleness_ns,
        )
        multiple_defined = start_equity > 0
        boundary_equities.extend((
            (segment_start - 1, start_equity),
            (segment_end - 1, end_equity),
        ))
        monthly.append({
            "month_id": month,
            "comparable_full_month": month_start >= evaluation["period_start_ts_ns"] and month_end <= evaluation["period_end_ts_ns"],
            "segment_start_ts_ns": segment_start,
            "segment_end_ts_ns": segment_end,
            "start_equity_jpy_micros": start_equity,
            "end_equity_jpy_micros": end_equity,
            "equity_multiple": _ratio_text(end_equity, start_equity) if multiple_defined else None,
            "equity_multiple_status": (
                "DEFINED" if multiple_defined else "UNDEFINED_NONPOSITIVE_START_EQUITY"
            ),
            "ruin_observed": start_equity <= 0 or end_equity <= 0,
        })
    peak = initial
    max_dd = 0
    max_dd_ratio = Fraction(0, 1)
    drawdown_observations = [
        (mark["arrival_ts_ns"], index, mark["marked_equity_jpy_micros"])
        for index, mark in enumerate(risk_timeline)
    ]
    drawdown_observations.extend(
        (timestamp, len(risk_timeline) + index, equity)
        for index, (timestamp, equity) in enumerate(boundary_equities)
    )
    for _, _, equity in sorted(drawdown_observations):
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        ratio = Fraction(drawdown, peak) if peak > 0 else Fraction(1, 1)
        if drawdown > max_dd:
            max_dd = drawdown
        if ratio > max_dd_ratio:
            max_dd_ratio = ratio
    n_eff, cvar, cvar_return, clusters = _cluster_metrics(records, evaluation)
    max_gross = max((mark["gross_notional_jpy_micros"] for mark in risk_timeline), default=0)
    min_equity = min(
        [initial, *(mark["marked_equity_jpy_micros"] for mark in risk_timeline),
         *(equity for _, equity in boundary_equities)]
    )
    max_required = max((mark["required_margin_jpy_micros"] for mark in risk_timeline), default=0)
    min_free = min((mark["free_margin_jpy_micros"] for mark in risk_timeline), default=initial)
    return {
        "proposal_count": len(records),
        "executed_count": len(filled),
        "disposition_counts": dict(sorted(Counter(record["status"] for record in records).items())),
        "signal_id_set_sha256": sha256_bytes(canonical_bytes(sorted(record["signal_id"] for record in records))),
        "common_gross_pnl_jpy_micros": gross,
        "realized_cost_jpy_micros": cost,
        "fill_sizing_drag_jpy_micros": sizing_drag,
        "latency_spread_slippage_drag_jpy_micros": latency_drag,
        "direct_commission_financing_cost_jpy_micros": direct_cost,
        "admission_opportunity_drag_jpy_micros": admission_drag,
        "total_execution_and_admission_drag_jpy_micros": decomposed_drag,
        "net_pnl_jpy_micros": net,
        "ending_equity_jpy_micros": initial + net,
        "ending_equity_multiple": _ratio_text(initial + net, initial),
        "direction_accuracy": _ratio_text(sum(record["common_gross_pnl_jpy_micros"] > 0 for record in filled), len(filled)) if filled else "0.000000000000000000",
        "max_drawdown_jpy_micros": max_dd,
        "max_drawdown_ratio": _nonnegative_ratio_ceiling_text(
            max_dd_ratio.numerator, max_dd_ratio.denominator
        ),
        "cvar_tail_bps": evaluation["cvar_tail_bps"],
        "cluster_cvar_jpy_micros": cvar,
        "cluster_cvar_return": cvar_return,
        "currency_time_cluster_n_eff": n_eff,
        "currency_time_cluster_observations": clusters,
        "monthly": monthly,
        "max_gross_notional_jpy_micros": max_gross,
        "minimum_marked_equity_jpy_micros": min_equity,
        "maximum_required_margin_jpy_micros": max_required,
        "minimum_free_margin_jpy_micros": min_free,
        "margin_guard_pass": (
            min_equity > 0
            and min_free >= 0
            and max_gross <= evaluation["margin_notional_cap_jpy_micros"]
            and all(mark["margin_ratio_pass"] is True for mark in risk_timeline)
            and all(
                record.get("exit_disposition")
                not in {"MARGIN_CLOSEOUT", "INVENTORY_CAP_CLOSEOUT"}
                for record in records
            )
        ),
        "terminal_open_positions": 0,
        "terminal_inventory_mtm_jpy_micros": 0,
    }


def _hash_chain(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    chained: list[dict[str, Any]] = []
    previous = ZERO_SHA
    for sequence, row in enumerate(rows, 1):
        payload = {"ledger_schema_version": 2, "ledger_sequence": sequence, "previous_hash": previous, **dict(row)}
        payload["record_hash"] = embedded_hash(payload, "record_hash")
        chained.append(payload)
        previous = payload["record_hash"]
    return chained


def _build_evidence(
    validated: tuple[
        list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any],
        dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, int]],
        dict[str, str], str,
    ],
    request_sha256: str,
) -> tuple[bytes, dict[str, Any], str]:
    source_rows, books, proposal, execution, inventory, accounting, evaluation, authority, registry, input_hashes, output_name = validated
    common = {
        row["proposal_ordinal"]: _common_reference(
            row, books, execution, accounting, evaluation, registry
        )
        for row in proposal["rows"]
    }
    all_records: list[dict[str, Any]] = []
    metrics_arms: dict[str, Any] = {}
    signal_sets: dict[str, list[str]] = {}
    for arm in ARMS:
        records, positions, risk = _simulate_arm(
            arm, source_rows, books, proposal, common, execution, inventory, accounting,
            evaluation, registry,
        )
        all_records.extend(records)
        signal_sets[arm] = sorted(record["signal_id"] for record in records)
        metrics_arms[arm] = _arm_metrics(
            records, positions, risk, source_rows, books, accounting, registry,
            evaluation, execution["max_trade_quote_staleness_ns"],
        )
    if len({tuple(signal_sets[arm]) for arm in ARMS}) != 1:
        raise OracleError("arm signal-ID sets diverged")
    all_records.sort(key=lambda row: (row["proposal_ordinal"], ARMS.index(row["arm"])))
    ledger_rows = _hash_chain(all_records)
    ledger_bytes = b"".join(canonical_bytes(row) + b"\n" for row in ledger_rows)
    metrics: dict[str, Any] = {
        "schema_version": 2,
        "initial_equity_jpy_micros": evaluation["initial_equity_jpy_micros"],
        "same_signal_ids_all_arms": True,
        "all_proposals_have_all_arm_dispositions": True,
        "common_gross_reference_shared": all(
            len({
                record["common_gross_pnl_jpy_micros"]
                for record in all_records if record["proposal_ordinal"] == ordinal
            }) == 1
            for ordinal in range(1, len(proposal["rows"]) + 1)
        ),
        "arms": metrics_arms,
        "external_orders": 0,
        "terminal_inventory_mtm_jpy_micros": 0,
    }
    metrics["metrics_sha256"] = embedded_hash(metrics, "metrics_sha256")
    provenance_root = sha256_bytes(canonical_bytes({
        "provenance": proposal["provenance"],
        "rows": [
            {
                "proposal_ordinal": row["proposal_ordinal"],
                "decision_source_event_sha256": row["decision_source_event_sha256"],
                "completed_data_watermark_source_ts_ns": row["completed_data_watermark_source_ts_ns"],
                "completed_data_prefix_root_sha256": row["completed_data_prefix_root_sha256"],
            }
            for row in proposal["rows"]
        ],
    }))
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "oracle_implementation": ORACLE_NAME,
        "status": "COMPLETE",
        "classification": CLASSIFICATION,
        "causal_signal_admission": False,
        # Exact-FD execution is host-local evidence.  The outer Python ``-c``
        # bootstrap cannot self-prove its own bytes, so release eligibility
        # remains false until a separately anchored launch intent exists.
        "release_evidence_eligible": False,
        "detector_replay_receipt_required": True,
        "authority": dict(AUTHORITY),
        "oracle_release_content_binding": {
            "code_sha256": _MODULE_CODE_SHA256,
            "contract_sha256": _CONTRACT_SHA256,
            "schema_sha256": _SCHEMA_SHA256,
            "launcher_sha256": _LAUNCHER_SHA256,
            "snapshot_mode": _EXECUTION_SNAPSHOT_MODE,
        },
        "oracle_execution_provenance_scope": EXECUTION_PROVENANCE_SCOPE,
        "request_sha256": request_sha256,
        "input_artifact_sha256": dict(sorted(input_hashes.items())),
        "raw_source_manifest_sha256": input_hashes["source_manifest"],
        "proposal_provenance_root_sha256": provenance_root,
        "producer_result_or_metrics_used": False,
        "proposal_identity_generated_by_oracle": True,
        "oracle_ledger_file": "oracle_ledger.jsonl",
        "oracle_ledger_sha256": sha256_bytes(ledger_bytes),
        "oracle_ledger_size_bytes": len(ledger_bytes),
        "oracle_ledger_row_count": len(ledger_rows),
        "oracle_ledger_terminal_hash": ledger_rows[-1]["record_hash"] if ledger_rows else ZERO_SHA,
        "oracle_metrics": metrics,
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_orders": 0,
        "anchor_status": ANCHOR_STATUS,
    }
    manifest["oracle_root_sha256"] = embedded_hash(manifest, "oracle_root_sha256")
    return ledger_bytes, manifest, output_name


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise OracleError("short immutable output write")
        view = view[written:]


def _write_file_at(directory_fd: int, name: str, data: bytes, mode: int = 0o600) -> None:
    if SAFE_COMPONENT_RE.fullmatch(name) is None:
        raise OracleError("unsafe output filename")
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
        dir_fd=directory_fd,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() \
                or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600:
            raise OracleError("output file type/owner invalid")
        if len(data) > MAX_ARTIFACT_BYTES:
            raise OracleError("output file exceeds fixed byte limit")
        _write_all(descriptor, data)
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        if not stat.S_ISREG(final.st_mode) or final.st_uid != os.geteuid() \
                or final.st_nlink != 1 or stat.S_IMODE(final.st_mode) != 0o600 \
                or final.st_size != len(data):
            raise OracleError("output file changed while writing")
    finally:
        os.close(descriptor)


def _lstat_at(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _read_child_file(directory_fd: int, child: str, filename: str) -> bytes:
    child_fd = os.open(
        child,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        return _read_relative(child_fd, filename, f"output {filename}")
    finally:
        os.close(child_fd)


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
        info.st_uid,
    )


def _open_bound_file_set(
    directory_fd: int,
    expected_names: set[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    """Open and retain every child FD for collective race detection."""
    if set(os.listdir(directory_fd)) != expected_names:
        raise OracleError(f"{label} file set mismatch")
    held: dict[str, dict[str, Any]] = {}
    try:
        for name in sorted(expected_names):
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_fd,
            )
            info = os.fstat(descriptor)
            path_info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() \
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1 \
                    or _stat_identity(info) != _stat_identity(path_info):
                os.close(descriptor)
                raise OracleError(f"{label} child is not a private bound regular file")
            data = _read_fd_snapshot(descriptor, f"{label} {name}")
            held[name] = {
                "fd": descriptor,
                "identity": _stat_identity(info),
                "sha256": sha256_bytes(data),
                "bytes": data,
            }
        _revalidate_bound_file_set(directory_fd, expected_names, held, label)
        return held
    except BaseException:
        for item in held.values():
            os.close(item["fd"])
        raise


def _revalidate_bound_file_set(
    directory_fd: int,
    expected_names: set[str],
    held: Mapping[str, Mapping[str, Any]],
    label: str,
) -> None:
    if set(os.listdir(directory_fd)) != expected_names or set(held) != expected_names:
        raise OracleError(f"{label} file set changed")
    for name in sorted(expected_names):
        item = held[name]
        descriptor = item["fd"]
        fd_info = os.fstat(descriptor)
        path_info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _stat_identity(fd_info) != item["identity"] \
                or _stat_identity(path_info) != item["identity"] \
                or fd_info.st_nlink != 1:
            raise OracleError(f"{label} child identity changed: {name}")
        data = _read_fd_snapshot(descriptor, f"{label} {name} revalidation")
        if sha256_bytes(data) != item["sha256"] or data != item["bytes"]:
            raise OracleError(f"{label} child bytes changed: {name}")


def _close_bound_file_set(held: Mapping[str, Mapping[str, Any]]) -> None:
    for name in sorted(held, reverse=True):
        os.close(held[name]["fd"])


def _child_file_set(directory_fd: int, child: str) -> set[str]:
    child_fd = os.open(
        child,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        return set(os.listdir(child_fd))
    finally:
        os.close(child_fd)


def _transaction_intent(request_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "transaction_id": sha256_bytes(canonical_bytes({
            "request_sha256": request_sha256,
            "code_sha256": _MODULE_CODE_SHA256,
            "contract_sha256": _CONTRACT_SHA256,
            "schema_sha256": _SCHEMA_SHA256,
        })),
        "request_sha256": request_sha256,
        "code_sha256": _MODULE_CODE_SHA256,
        "contract_sha256": _CONTRACT_SHA256,
        "schema_sha256": _SCHEMA_SHA256,
    }


def _transaction_commit(
    intent_bytes: bytes,
    request_sha256: str,
    ledger_bytes: bytes,
    manifest_bytes: bytes,
    terminal_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "transaction_id": _transaction_intent(request_sha256)["transaction_id"],
        "request_sha256": request_sha256,
        "intent_sha256": sha256_bytes(intent_bytes),
        "ledger_sha256": sha256_bytes(ledger_bytes),
        "ledger_size_bytes": len(ledger_bytes),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "manifest_size_bytes": len(manifest_bytes),
        "terminal_hash": terminal_hash,
    }


def _validate_complete_output(
    output_root_fd: int,
    output_name: str,
    request_sha256: str,
    expected_ledger: bytes | None = None,
    expected_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    info = _lstat_at(output_root_fd, output_name)
    if info is None or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise OracleError("complete output directory missing or non-directory")
    child_fd = os.open(
        output_name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=output_root_fd,
    )
    try:
        return _validate_complete_output_fd(
            child_fd,
            request_sha256,
            expected_ledger,
            expected_manifest,
        )
    finally:
        os.close(child_fd)


def _validate_complete_output_fd(
    child_fd: int,
    request_sha256: str,
    expected_ledger: bytes | None = None,
    expected_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    info = os.fstat(child_fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise OracleError("complete output FD is not a trusted directory")
    expected_names = {
        "intent.json", "oracle_ledger.jsonl", "oracle_manifest.json", "COMMIT.json",
    }
    held = _open_bound_file_set(child_fd, expected_names, "complete output")
    try:
        intent_bytes = held["intent.json"]["bytes"]
        ledger = held["oracle_ledger.jsonl"]["bytes"]
        manifest_bytes = held["oracle_manifest.json"]["bytes"]
        commit_bytes = held["COMMIT.json"]["bytes"]
        manifest = strict_json(manifest_bytes, "oracle manifest")
        strict_json(intent_bytes, "oracle intent")
        commit = strict_json(commit_bytes, "oracle commit")
        expected_intent = _transaction_intent(request_sha256)
        if intent_bytes != canonical_bytes(expected_intent) + b"\n":
            raise OracleError("output intent binding mismatch")
        _exact_keys(
            commit,
            {
                "schema_version", "transaction_id", "request_sha256", "ledger_sha256",
                "intent_sha256", "ledger_size_bytes", "manifest_sha256", "manifest_size_bytes", "terminal_hash",
            },
            "oracle commit",
        )
        if _integer(commit["schema_version"], "commit schema") != 1 \
                or commit["request_sha256"] != request_sha256 \
                or commit["transaction_id"] != expected_intent["transaction_id"] \
                or commit["intent_sha256"] != sha256_bytes(intent_bytes) \
                or commit["ledger_sha256"] != sha256_bytes(ledger) \
                or commit["ledger_size_bytes"] != len(ledger) \
                or commit["manifest_sha256"] != sha256_bytes(manifest_bytes) \
                or commit["manifest_size_bytes"] != len(manifest_bytes) \
                or commit["terminal_hash"] != manifest.get("oracle_ledger_terminal_hash"):
            raise OracleError("output COMMIT binding mismatch")
        if manifest.get("request_sha256") != request_sha256 \
                or manifest.get("oracle_ledger_sha256") != sha256_bytes(ledger):
            raise OracleError("existing output request/ledger binding mismatch")
        if expected_ledger is not None and ledger != expected_ledger:
            raise OracleError("existing canonical ledger differs from deterministic replay")
        if expected_manifest is not None and manifest_bytes != canonical_bytes(expected_manifest) + b"\n":
            raise OracleError("existing canonical manifest differs from deterministic replay")
        _revalidate_bound_file_set(child_fd, expected_names, held, "complete output")
        return {"manifest": manifest, "ledger_bytes": ledger}
    finally:
        _close_bound_file_set(held)


def _materialize_final_output(
    output_root_fd: int,
    output_name: str,
    request_sha256: str,
    ledger_bytes: bytes,
    manifest: Mapping[str, Any],
    assert_lock: Callable[[], None],
    *,
    existing: bool,
) -> dict[str, Any]:
    final_fd = os.open(
        output_name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=output_root_fd,
    )
    try:
        info = os.fstat(final_fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise OracleError("final output directory ownership/mode invalid")
        intent_bytes = canonical_bytes(_transaction_intent(request_sha256)) + b"\n"
        manifest_bytes = canonical_bytes(manifest) + b"\n"
        commit_bytes = canonical_bytes(_transaction_commit(
            intent_bytes,
            request_sha256,
            ledger_bytes,
            manifest_bytes,
            manifest["oracle_ledger_terminal_hash"],
        )) + b"\n"
        expected = (
            ("intent.json", intent_bytes),
            ("oracle_ledger.jsonl", ledger_bytes),
            ("oracle_manifest.json", manifest_bytes),
            ("COMMIT.json", commit_bytes),
        )
        present = set(os.listdir(final_fd))
        allowed = {name for name, _ in expected}
        if not present <= allowed:
            raise OracleError("final output file set mismatch")
        if existing and "intent.json" not in present:
            raise OracleError("FAILED_VISIBLE_FINAL_WITHOUT_INTENT")
        if "COMMIT.json" in present and present != allowed:
            raise OracleError("final COMMIT exists with incomplete artifact set")
        for filename, data in expected:
            if filename in present:
                if _read_relative(final_fd, filename, f"final {filename}") != data:
                    raise OracleError(f"final {filename} differs from deterministic bytes")
            else:
                assert_lock()
                _write_file_at(final_fd, filename, data)
                assert_lock()
        assert_lock()
        os.fsync(final_fd)
        assert_lock()
        result = _validate_complete_output_fd(
            final_fd,
            request_sha256,
            ledger_bytes,
            manifest,
        )
        assert_lock()
        return result
    finally:
        os.close(final_fd)


def _complete_or_recover_stage(
    output_root_fd: int,
    stage_name: str,
    request_sha256: str,
    ledger_bytes: bytes,
    manifest: Mapping[str, Any],
    assert_lock: Callable[[], None],
) -> tuple[int, os.stat_result]:
    stage_fd = os.open(
        stage_name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=output_root_fd,
    )
    try:
        assert_lock()
        stage_info = os.fstat(stage_fd)
        if not stat.S_ISDIR(stage_info.st_mode) or stage_info.st_uid != os.geteuid() or stage_info.st_mode & 0o022:
            raise OracleError("staging directory ownership/mode invalid")
        present = set(os.listdir(stage_fd))
        allowed = {"intent.json", "oracle_ledger.jsonl", "oracle_manifest.json", "COMMIT.json"}
        if not present <= allowed:
            raise OracleError("staging contains unexpected file")
        intent_bytes = canonical_bytes(_transaction_intent(request_sha256)) + b"\n"
        manifest_bytes = canonical_bytes(manifest) + b"\n"
        commit_bytes = canonical_bytes(_transaction_commit(
            intent_bytes,
            request_sha256,
            ledger_bytes,
            manifest_bytes,
            manifest["oracle_ledger_terminal_hash"],
        )) + b"\n"
        expected = (
            ("intent.json", intent_bytes),
            ("oracle_ledger.jsonl", ledger_bytes),
            ("oracle_manifest.json", manifest_bytes),
            ("COMMIT.json", commit_bytes),
        )
        for filename, data in expected:
            if filename in present:
                try:
                    assert_lock()
                    existing = _read_relative(
                        stage_fd, filename, f"staging {filename}"
                    )
                    assert_lock()
                except LockIdentityError:
                    raise
                except OSError as error:
                    raise OracleError(
                        f"staging {filename} cannot be opened safely"
                    ) from error
                if existing != data:
                    raise OracleError(f"staging {filename} is partial or mismatched")
            else:
                assert_lock()
                _write_file_at(stage_fd, filename, data)
                assert_lock()
        assert_lock()
        os.fsync(stage_fd)
        assert_lock()
        return stage_fd, stage_info
    except BaseException:
        os.close(stage_fd)
        raise


def _commit_completed_stage(
    output_root_fd: int,
    stage_fd: int,
    stage_info: os.stat_result,
    stage_name: str,
    output_name: str,
    request_sha256: str,
    ledger_bytes: bytes,
    manifest: Mapping[str, Any],
    assert_lock: Callable[[], None],
) -> dict[str, Any]:
    expected_names = {
        "intent.json", "oracle_ledger.jsonl", "oracle_manifest.json", "COMMIT.json",
    }
    held: dict[str, dict[str, Any]] | None = None
    try:
        assert_lock()
        _validate_complete_output_fd(
            stage_fd, request_sha256, ledger_bytes, manifest
        )
        held = _open_bound_file_set(stage_fd, expected_names, "publish stage")
        _revalidate_bound_file_set(
            stage_fd, expected_names, held, "publish stage before pathname check"
        )
        stage_dirent = _lstat_at(output_root_fd, stage_name)
        if stage_dirent is None or (stage_dirent.st_dev, stage_dirent.st_ino) != (
            stage_info.st_dev, stage_info.st_ino
        ):
            raise OracleError("staging pathname no longer names held directory FD")
        if _RENAME_EXCLUSIVE is None:
            _close_bound_file_set(held)
            held = None
            try:
                assert_lock()
                os.mkdir(output_name, 0o700, dir_fd=output_root_fd)
            except FileExistsError as error:
                raise OracleError("output leaf appeared during exclusive commit") from error
            os.fsync(output_root_fd)
            assert_lock()
            result = _materialize_final_output(
                output_root_fd,
                output_name,
                request_sha256,
                ledger_bytes,
                manifest,
                assert_lock,
                existing=False,
            )
            # PATH_LOADED_TEST_ADAPTER uses a COMMIT-last visibility protocol.
            # Release evidence never takes this branch: the sealed launcher
            # injects native renameatx_np(RENAME_EXCL).
            for filename in (
                "intent.json", "oracle_ledger.jsonl", "oracle_manifest.json", "COMMIT.json",
            ):
                assert_lock()
                os.unlink(filename, dir_fd=stage_fd)
                assert_lock()
            assert_lock()
            os.fsync(stage_fd)
            assert_lock()
            os.rmdir(stage_name, dir_fd=output_root_fd)
            os.fsync(output_root_fd)
            assert_lock()
            return result
        _revalidate_bound_file_set(
            stage_fd, expected_names, held, "publish stage immediately before rename"
        )
        assert_lock()
        _RENAME_EXCLUSIVE(output_root_fd, stage_name, output_name)
        os.fsync(output_root_fd)
        assert_lock()
        final_fd = os.open(
            output_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=output_root_fd,
        )
        try:
            final_info = os.fstat(final_fd)
            if (final_info.st_dev, final_info.st_ino) != (stage_info.st_dev, stage_info.st_ino):
                raise OracleError("atomically published output inode mismatch")
            _revalidate_bound_file_set(
                final_fd, expected_names, held, "published output held-file fence"
            )
            result = _validate_complete_output_fd(
                final_fd, request_sha256, ledger_bytes, manifest
            )
            final_dirent = _lstat_at(output_root_fd, output_name)
            if final_dirent is None or (final_dirent.st_dev, final_dirent.st_ino) != (
                final_info.st_dev, final_info.st_ino
            ):
                raise OracleError("published output pathname changed during validation")
            assert_lock()
            return result
        finally:
            os.close(final_fd)
    finally:
        if held is not None:
            _close_bound_file_set(held)
        os.close(stage_fd)


def _publish_output(
    output_root_fd: int,
    output_name: str,
    request_sha256: str,
    ledger_bytes: bytes,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_dirfd(output_root_fd, "output root")
    transaction_id = _transaction_intent(request_sha256)["transaction_id"]
    stage_name = f".{output_name}.{transaction_id[:16]}.stage"
    lock_name = f".{output_name}.lock"
    failed_name = f".{output_name}.{transaction_id[:16]}.failed"
    lock_fd = os.open(
        lock_name,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=output_root_fd,
    )
    locked = False

    def assert_lock() -> None:
        _assert_named_lock_identity(output_root_fd, lock_name, lock_fd)

    try:
        lock_info = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_info.st_mode) or lock_info.st_uid != os.geteuid() \
                or stat.S_IMODE(lock_info.st_mode) != 0o600 or lock_info.st_nlink != 1:
            raise OracleError("oracle lock file ownership/mode invalid")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise OracleError("concurrent oracle publication") from error
        locked = True
        assert_lock()
        os.ftruncate(lock_fd, 0)
        os.lseek(lock_fd, 0, os.SEEK_SET)
        _write_all(lock_fd, canonical_bytes({"pid": os.getpid(), "transaction_id": transaction_id}) + b"\n")
        os.fsync(lock_fd)
        os.fsync(output_root_fd)
        assert_lock()
        existing = _lstat_at(output_root_fd, output_name)
        if existing is not None:
            if not stat.S_ISDIR(existing.st_mode):
                raise OracleError("output leaf already exists and is not a directory")
            assert_lock()
            result = _validate_complete_output(
                output_root_fd, output_name, request_sha256, ledger_bytes, manifest
            )
            assert_lock()
            return result
        stage_info = _lstat_at(output_root_fd, stage_name)
        if stage_info is not None:
            if not stat.S_ISDIR(stage_info.st_mode):
                raise OracleError("staging leaf exists and is not a directory")
            try:
                stage_fd, completed_stage_info = _complete_or_recover_stage(
                    output_root_fd,
                    stage_name,
                    request_sha256,
                    ledger_bytes,
                    manifest,
                    assert_lock,
                )
                return _commit_completed_stage(
                    output_root_fd,
                    stage_fd,
                    completed_stage_info,
                    stage_name,
                    output_name,
                    request_sha256,
                    ledger_bytes,
                    manifest,
                    assert_lock,
                )
            except LockIdentityError:
                raise
            except OracleError:
                assert_lock()
                current_stage = _lstat_at(output_root_fd, stage_name)
                if current_stage is None or (current_stage.st_dev, current_stage.st_ino) != (
                    stage_info.st_dev, stage_info.st_ino
                ):
                    raise OracleError("STAGE_PATH_SUBSTITUTED")
                if _lstat_at(output_root_fd, failed_name) is not None:
                    raise OracleError("incomplete staging and failure evidence already exist")
                assert_lock()
                if _RENAME_EXCLUSIVE is not None:
                    _RENAME_EXCLUSIVE(output_root_fd, stage_name, failed_name)
                else:
                    os.rename(
                        stage_name, failed_name,
                        src_dir_fd=output_root_fd, dst_dir_fd=output_root_fd,
                    )
                os.fsync(output_root_fd)
                assert_lock()
                raise OracleError("FAILED_VISIBLE_PARTIAL_OUTPUT")
        if _lstat_at(output_root_fd, failed_name) is not None:
            raise OracleError("prior partial output failure is preserved")
        assert_lock()
        os.mkdir(stage_name, 0o700, dir_fd=output_root_fd)
        os.fsync(output_root_fd)
        assert_lock()
        try:
            stage_fd, completed_stage_info = _complete_or_recover_stage(
                output_root_fd,
                stage_name,
                request_sha256,
                ledger_bytes,
                manifest,
                assert_lock,
            )
        except BaseException:
            # Keep the private stage visible for deterministic verify-only
            # recovery.  A mismatched partial file will be quarantined on the
            # next invocation instead of being reported complete.
            os.fsync(output_root_fd)
            raise
        return _commit_completed_stage(
            output_root_fd,
            stage_fd,
            completed_stage_info,
            stage_name,
            output_name,
            request_sha256,
            ledger_bytes,
            manifest,
            assert_lock,
        )
    finally:
        lock_error: BaseException | None = None
        if locked:
            try:
                assert_lock()
            except BaseException as error:
                lock_error = error
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
        if lock_error is not None:
            raise lock_error


def execute_from_fds(
    request_bytes: bytes,
    *,
    input_root_fd: int,
    output_root_fd: int,
    code_fd: int | None = None,
    contract_fd: int | None = None,
    schema_fd: int | None = None,
) -> dict[str, Any]:
    input_root = _validate_dirfd(input_root_fd, "input root")
    output_root = _validate_dirfd(output_root_fd, "output root")
    if (input_root.st_dev, input_root.st_ino) == (output_root.st_dev, output_root.st_ino):
        raise OracleError("input and output roots must be distinct directory inodes")
    if code_fd is None or _read_fd_snapshot(
        code_fd,
        "oracle code",
        allow_unlinked_sealed_runtime=True,
    ) != _MODULE_CODE_BYTES:
        raise OracleError("launcher code FD differs from executed load-time bytes or is missing")
    if contract_fd is None or _read_fd_snapshot(
        contract_fd,
        "oracle contract",
        allow_unlinked_sealed_runtime=True,
    ) != _CONTRACT_BYTES:
        raise OracleError("launcher contract FD differs from frozen contract bytes or is missing")
    if schema_fd is None or _read_fd_snapshot(
        schema_fd,
        "oracle schema",
        allow_unlinked_sealed_runtime=True,
    ) != _SCHEMA_BYTES:
        raise OracleError("launcher schema FD differs from frozen schema bytes or is missing")
    request = strict_json(request_bytes, "oracle request")
    request_sha = sha256_bytes(request_bytes)
    validated = _validate_request(request, input_root_fd)
    ledger_bytes, manifest, output_name = _build_evidence(validated, request_sha)
    published = _publish_output(output_root_fd, output_name, request_sha, ledger_bytes, manifest)
    return {
        "output_directory": output_name,
        "manifest_relative_path": f"{output_name}/oracle_manifest.json",
        "ledger_relative_path": f"{output_name}/oracle_ledger.jsonl",
        "manifest": published["manifest"],
    }


def _open_trusted_directory(path: Path, label: str) -> int:
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise OracleError(f"{label} must be a non-symlink directory")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        os.close(descriptor)
        raise OracleError(f"{label} changed while opening")
    _validate_dirfd(descriptor, label)
    return descriptor


def execute(
    request: Mapping[str, Any], *, trusted_input_root: Path, trusted_output_root: Path
) -> dict[str, Any]:
    """Test/library adapter; production uses inherited launcher FDs."""
    request_bytes = canonical_bytes(dict(request)) + b"\n"
    input_fd = _open_trusted_directory(trusted_input_root, "input root")
    output_fd = _open_trusted_directory(trusted_output_root, "output root")
    code_fd = os.open(_MODULE_PATH, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    contract_fd = os.open(_MODULE_PATH.parent / CONTRACT_NAME, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    schema_fd = os.open(_MODULE_PATH.parent / SCHEMA_NAME, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        return execute_from_fds(
            request_bytes,
            input_root_fd=input_fd,
            output_root_fd=output_fd,
            code_fd=code_fd,
            contract_fd=contract_fd,
            schema_fd=schema_fd,
        )
    finally:
        os.close(schema_fd)
        os.close(contract_fd)
        os.close(code_fd)
        os.close(input_fd)
        os.close(output_fd)


def _audit_hook(event: str, _: tuple[Any, ...]) -> None:
    if event.startswith(("socket.", "subprocess.")) or event in {
        "import",
        "os.system", "os.posix_spawn", "os.exec", "os.spawn",
    }:
        raise OracleError(f"runtime capability denied: {event}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-fd", type=int, required=True)
    parser.add_argument("--input-root-fd", type=int, required=True)
    parser.add_argument("--output-root-fd", type=int, required=True)
    parser.add_argument("--code-fd", type=int, required=True)
    parser.add_argument("--contract-fd", type=int, required=True)
    parser.add_argument("--schema-fd", type=int, required=True)
    args = parser.parse_args()
    request_bytes = _read_fd_snapshot(args.request_fd, "oracle request")
    sys.addaudithook(_audit_hook)
    result = execute_from_fds(
        request_bytes,
        input_root_fd=args.input_root_fd,
        output_root_fd=args.output_root_fd,
        code_fd=args.code_fd,
        contract_fd=args.contract_fd,
        schema_fd=args.schema_fd,
    )
    print(json.dumps({
        "ok": True,
        "classification": result["manifest"]["classification"],
        "oracle_root_sha256": result["manifest"]["oracle_root_sha256"],
        "output_directory": result["output_directory"],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OracleError as error:
        print(json.dumps({
            "ok": False,
            "error_code": "ORACLE_FAIL_CLOSED",
            "error_sha256": sha256_bytes(str(error).encode("utf-8")),
        }, sort_keys=True, separators=(",", ":")))
        raise SystemExit(2)
