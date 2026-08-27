#!/usr/bin/env python3
"""Credential-free, file-only forward-shadow plumbing, corrective V2.

The module accepts local append-only JSONL/CSV BBO batches.  It deliberately
contains no network transport, secret source, broker client, or external
dispatch path.  Expected orders and fills are internal paper records only.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

import shadow_jpy_accounting_v1 as accounting


SCHEMA_VERSION = 1
ZERO_HASH = "0" * 64
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
INSTRUMENT_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
ALLOWED_SUFFIXES = {".jsonl": "JSONL", ".csv": "CSV"}
TIMEFRAMES_SECONDS = {"M5": 300, "M15": 900, "H1": 3600, "H4": 14400}
MAX_SOURCE_GAP_NS = 90 * 1_000_000_000
MAX_ARRIVAL_GAP_NS = 90 * 1_000_000_000
HEARTBEAT_EXPIRY_NS = 90 * 1_000_000_000
BURN_IN_M5_BARS = 48
INITIAL_EQUITY_JPY = 200_000.0
FIXED_GROSS_CAP = 1.0
WORKER_ARMS = ("BOT_ONLY", "ACTUAL_LLM_INVENTORY_POLICY")
EXECUTION_SCENARIOS = {
    "EXECUTABLE_BASE": {
        "latency_ns": 500_000_000,
        "accounting": accounting.EXECUTABLE_BASE,
    },
    "ADVERSE_STRESS": {
        "latency_ns": 1_500_000_000,
        "accounting": accounting.ADVERSE_STRESS,
    },
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


class ShadowCoreError(RuntimeError):
    """Fail-closed file, schema, chronology, or state violation."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def embedded_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return sha256_bytes(canonical_bytes(unsigned))


def _validate_owned_regular(stat_result: os.stat_result, label: str) -> None:
    if not stat.S_ISREG(stat_result.st_mode):
        raise ShadowCoreError("SECURE_REGULAR_FILE_REQUIRED", label)
    if stat_result.st_uid != os.getuid():
        raise ShadowCoreError("FILE_OWNER_MISMATCH", label)
    if stat_result.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ShadowCoreError("INSECURE_FILE_MODE", label)


def _validate_owned_directory(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    try:
        result = os.lstat(path)
    except FileNotFoundError as error:
        raise ShadowCoreError("SECURE_DIRECTORY_REQUIRED", path.name) from error
    if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
        raise ShadowCoreError("SECURE_DIRECTORY_REQUIRED", path.name)
    if result.st_uid != os.getuid() or result.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ShadowCoreError("INSECURE_DIRECTORY_MODE", path.name)


def secure_read_bytes(path: Path, *, missing_ok: bool = False) -> bytes | None:
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise ShadowCoreError("SECURE_REGULAR_FILE_REQUIRED", path.name)
    if stat.S_ISLNK(before.st_mode):
        raise ShadowCoreError("SYMLINK_FORBIDDEN", path.name)
    _validate_owned_regular(before, path.name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ShadowCoreError("SECURE_FILE_OPEN_FAILED", path.name) from error
    try:
        opened = os.fstat(descriptor)
        _validate_owned_regular(opened, path.name)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ShadowCoreError("FILE_IDENTITY_CHANGED", path.name)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise ShadowCoreError("SOURCE_CHANGED_DURING_READ", path.name)
        data = b"".join(chunks)
        if len(data) != opened.st_size:
            raise ShadowCoreError("SOURCE_CHANGED_DURING_READ", path.name)
        return data
    finally:
        os.close(descriptor)


def secure_append_line(path: Path, line: bytes) -> None:
    _validate_owned_directory(path.parent, create=True)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise ShadowCoreError("SECURE_FILE_OPEN_FAILED", path.name) from error
    try:
        result = os.fstat(descriptor)
        _validate_owned_regular(result, path.name)
        view = memoryview(line)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ShadowCoreError("LEDGER_APPEND_FAILED", path.name)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def secure_exclusive_bytes(path: Path, data: bytes) -> None:
    _validate_owned_directory(path.parent, create=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        current = secure_read_bytes(path)
        if current != data:
            raise ShadowCoreError("CONTENT_ADDRESS_MISMATCH", path.name)
        return
    except OSError as error:
        raise ShadowCoreError("SECURE_FILE_OPEN_FAILED", path.name) from error
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ShadowCoreError("CONTENT_ADDRESS_WRITE_FAILED", path.name)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _validate_owned_directory(path.parent, create=True)
    if path.exists() or path.is_symlink():
        secure_read_bytes(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _validate_owned_regular(os.lstat(path), path.name)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_timestamp_ns(value: Any) -> int:
    if isinstance(value, bool):
        raise ShadowCoreError("INVALID_TIMESTAMP", "boolean timestamp")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.isdigit():
        result = int(value)
    else:
        raise ShadowCoreError("INVALID_TIMESTAMP", repr(value))
    if result < 0:
        raise ShadowCoreError("INVALID_TIMESTAMP", "negative epoch nanoseconds")
    return result


def format_timestamp_ns(value: int) -> str:
    seconds, nanoseconds = divmod(parse_timestamp_ns(value), 1_000_000_000)
    base = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{nanoseconds:09d}Z"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ShadowCoreError("INVALID_BOOLEAN", repr(value))


def parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ShadowCoreError("INVALID_SEQUENCE", repr(value))
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ShadowCoreError("INVALID_SEQUENCE", repr(value)) from error
    if result < 0:
        raise ShadowCoreError("INVALID_SEQUENCE", "negative")
    return result


def parse_flags(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ShadowCoreError("INVALID_QUALITY_FLAGS", stripped) from error
        else:
            value = [item for item in stripped.split("|") if item]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ShadowCoreError("INVALID_QUALITY_FLAGS", repr(value))
    return tuple(sorted({item.strip().upper() for item in value if item.strip()}))


def decimal_text(value: Any, *, allow_empty: bool = False) -> str | None:
    if value in (None, "") and allow_empty:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ShadowCoreError("INVALID_PRICE", repr(value)) from error
    if not result.is_finite():
        raise ShadowCoreError("INVALID_PRICE", "non-finite")
    return format(result, "f")


@dataclass(frozen=True)
class BBOEvent:
    schema_version: int
    provider_id: str
    instrument: str
    bid: str | None
    ask: str | None
    liquidity_optional: str | None
    source_ts_ns: int
    arrival_ts_ns: int
    provider_event_id: str | None
    sequence: int | None
    heartbeat: bool
    raw_payload_sha256: str
    quality_flags: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], raw_payload_sha256: str
    ) -> "BBOEvent":
        if set(value).issuperset({"schema_version"}) is False:
            raise ShadowCoreError("UNKNOWN_SCHEMA", "schema_version missing")
        try:
            version = int(value["schema_version"])
        except (TypeError, ValueError) as error:
            raise ShadowCoreError("UNKNOWN_SCHEMA", repr(value["schema_version"])) from error
        if version != SCHEMA_VERSION:
            raise ShadowCoreError("UNKNOWN_SCHEMA", str(version))
        required = {
            "provider_id", "instrument", "source_ts_ns", "arrival_ts_ns", "heartbeat",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ShadowCoreError("SCHEMA_FIELD_MISSING", ",".join(missing))
        provider = str(value["provider_id"]).strip()
        instrument = str(value["instrument"]).strip().upper()
        if not provider:
            raise ShadowCoreError("INVALID_PROVIDER_ID")
        if INSTRUMENT_RE.fullmatch(instrument) is None:
            raise ShadowCoreError("INVALID_INSTRUMENT", instrument)
        heartbeat = parse_bool(value["heartbeat"])
        bid = decimal_text(value.get("bid"), allow_empty=heartbeat)
        ask = decimal_text(value.get("ask"), allow_empty=heartbeat)
        if (bid is None) != (ask is None):
            raise ShadowCoreError("INVALID_BBO", "one side missing")
        if bid is not None and ask is not None:
            bid_value, ask_value = Decimal(bid), Decimal(ask)
            if bid_value <= 0 or ask_value <= 0:
                raise ShadowCoreError("NONPOSITIVE_PRICE", instrument)
            if bid_value >= ask_value:
                raise ShadowCoreError("SPREAD_INVERSION", instrument)
        liquidity = decimal_text(value.get("liquidity_optional"), allow_empty=True)
        if liquidity is not None and Decimal(liquidity) < 0:
            raise ShadowCoreError("INVALID_LIQUIDITY", instrument)
        source = parse_timestamp_ns(value["source_ts_ns"])
        arrival = parse_timestamp_ns(value["arrival_ts_ns"])
        if arrival < source:
            raise ShadowCoreError("CLOCK_REVERSAL", "arrival before source")
        event_id_value = value.get("provider_event_id")
        event_id = None if event_id_value in (None, "") else str(event_id_value)
        if not SHA256_RE.fullmatch(raw_payload_sha256):
            raise ShadowCoreError("INVALID_RAW_PAYLOAD_HASH")
        return cls(
            version, provider, instrument, bid, ask, liquidity, source, arrival,
            event_id, parse_optional_int(value.get("sequence")), heartbeat,
            raw_payload_sha256, parse_flags(value.get("quality_flags")),
        )

    @property
    def has_price(self) -> bool:
        return self.bid is not None and self.ask is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "instrument": self.instrument,
            "bid": self.bid,
            "ask": self.ask,
            "liquidity_optional": self.liquidity_optional,
            "source_ts_ns": self.source_ts_ns,
            "arrival_ts_ns": self.arrival_ts_ns,
            "provider_event_id": self.provider_event_id,
            "sequence": self.sequence,
            "heartbeat": self.heartbeat,
            "raw_payload_sha256": self.raw_payload_sha256,
            "quality_flags": list(self.quality_flags),
        }


@dataclass(frozen=True)
class RawInput:
    mapping: dict[str, Any]
    raw_record: bytes


@dataclass(frozen=True)
class SourceSnapshot:
    source_name: str
    source_identity_sha256: str
    format: str
    data: bytes
    source_bytes_sha256: str
    source_size_bytes: int
    source_mtime_ns: int
    source_device: int
    source_inode: int


class LocalFileEventSource(Protocol):
    def snapshot(self) -> SourceSnapshot: ...

    def raw_inputs(self, snapshot: SourceSnapshot) -> Iterator[RawInput]: ...


class OfflineBBOFile:
    """Strict local JSONL/CSV adapter; it never mutates the source file."""

    def __init__(self, path: Path | str) -> None:
        raw_path = str(path)
        if "://" in raw_path:
            raise ShadowCoreError("LOCAL_FILE_REQUIRED")
        self.path = Path(path)
        self._last_snapshot: SourceSnapshot | None = None

    def _snapshot(self, *, require_complete_record: bool) -> SourceSnapshot:
        suffix = self.path.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ShadowCoreError("UNSUPPORTED_FILE_FORMAT", suffix)
        try:
            before = os.lstat(self.path)
        except FileNotFoundError as error:
            raise ShadowCoreError("LOCAL_FILE_REQUIRED", self.path.name) from error
        if stat.S_ISLNK(before.st_mode):
            raise ShadowCoreError("SYMLINK_FORBIDDEN", self.path.name)
        _validate_owned_regular(before, self.path.name)
        data = secure_read_bytes(self.path)
        assert data is not None
        after = os.lstat(self.path)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ShadowCoreError("FILE_IDENTITY_CHANGED", self.path.name)
        identity = sha256_bytes(canonical_bytes({
            "source_name": self.path.name,
            "format": ALLOWED_SUFFIXES[suffix],
        }))
        snapshot = SourceSnapshot(
            self.path.name, identity, ALLOWED_SUFFIXES[suffix], data,
            sha256_bytes(data), len(data), before.st_mtime_ns,
            before.st_dev, before.st_ino,
        )
        self._last_snapshot = snapshot
        if require_complete_record and (not data or not data.endswith(b"\n")):
            raise ShadowCoreError("TRUNCATED_SOURCE_RECORD", self.path.name)
        return snapshot

    def snapshot(self) -> SourceSnapshot:
        return self._snapshot(require_complete_record=True)

    def failure_snapshot(self, error: ShadowCoreError) -> SourceSnapshot:
        if self._last_snapshot is not None:
            return self._last_snapshot
        try:
            result = os.lstat(self.path)
            size, mtime, device, inode = (
                result.st_size, result.st_mtime_ns, result.st_dev, result.st_ino
            )
        except OSError:
            size = mtime = device = inode = 0
        fingerprint = canonical_bytes({
            "source_name": self.path.name,
            "error_code": error.code,
            "size": size,
            "mtime_ns": mtime,
            "device": device,
            "inode": inode,
        })
        suffix = self.path.suffix.lower()
        return SourceSnapshot(
            self.path.name,
            sha256_bytes(fingerprint),
            ALLOWED_SUFFIXES.get(suffix, "INVALID"),
            b"",
            sha256_bytes(fingerprint),
            size,
            mtime,
            device,
            inode,
        )

    def raw_inputs(self, snapshot: SourceSnapshot) -> Iterator[RawInput]:
        if snapshot.format == "JSONL":
            for raw_line in snapshot.data.splitlines(keepends=True):
                payload = raw_line.rstrip(b"\r\n")
                if not payload:
                    raise ShadowCoreError("EMPTY_SOURCE_RECORD")
                try:
                    decoded = payload.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ShadowCoreError("INVALID_UTF8_RECORD") from error
                try:
                    value = json.loads(decoded)
                except json.JSONDecodeError as error:
                    raise ShadowCoreError("INVALID_JSON_RECORD") from error
                if not isinstance(value, dict):
                    raise ShadowCoreError("INVALID_JSON_RECORD", "object required")
                yield RawInput(value, payload)
            return
        try:
            text = snapshot.data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ShadowCoreError("INVALID_UTF8_RECORD") from error
        lines = text.splitlines(keepends=True)
        if len(lines) < 2:
            raise ShadowCoreError("CSV_DATA_ROW_REQUIRED")
        header = lines[0]
        for line in lines[1:]:
            if not line.strip():
                raise ShadowCoreError("EMPTY_SOURCE_RECORD")
            reader = csv.DictReader(io.StringIO(header + line))
            try:
                value = next(reader)
            except StopIteration as error:
                raise ShadowCoreError("INVALID_CSV_RECORD") from error
            if None in value:
                raise ShadowCoreError("INVALID_CSV_RECORD", "column overflow")
            yield RawInput(dict(value), line.rstrip("\r\n").encode("utf-8"))

    def events(self, snapshot: SourceSnapshot) -> Iterator[BBOEvent]:
        for item in self.raw_inputs(snapshot):
            yield BBOEvent.from_mapping(item.mapping, sha256_bytes(item.raw_record))


class AppendOnlyHashLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows = self._read_verified()

    def _read_verified(self) -> list[dict[str, Any]]:
        data = secure_read_bytes(self.path, missing_ok=True)
        if data is None:
            return []
        if data and not data.endswith(b"\n"):
            raise ShadowCoreError("PARTIAL_LEDGER_RECORD", self.path.name)
        rows: list[dict[str, Any]] = []
        previous = ZERO_HASH
        for index, raw in enumerate(data.splitlines(), 1):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ShadowCoreError("PARTIAL_LEDGER_RECORD", self.path.name) from error
            if not isinstance(row, dict):
                raise ShadowCoreError("LEDGER_RECORD_INVALID", self.path.name)
            record_hash = row.get("record_hash")
            unsigned = dict(row)
            unsigned.pop("record_hash", None)
            if row.get("ledger_sequence") != index or row.get("previous_hash") != previous:
                raise ShadowCoreError("LEDGER_CHAIN_MISMATCH", self.path.name)
            if record_hash != sha256_bytes(canonical_bytes(unsigned)):
                raise ShadowCoreError("LEDGER_HASH_MISMATCH", self.path.name)
            previous = record_hash
            rows.append(row)
        return rows

    @property
    def terminal_hash(self) -> str:
        return self.rows[-1]["record_hash"] if self.rows else ZERO_HASH

    def hash_at(self, sequence: int) -> str:
        if sequence == 0:
            return ZERO_HASH
        if sequence < 0 or sequence > len(self.rows):
            raise ShadowCoreError("CHECKPOINT_AHEAD_OF_LEDGER", self.path.name)
        return self.rows[sequence - 1]["record_hash"]

    def append(self, value: Mapping[str, Any]) -> dict[str, Any]:
        unsigned = {
            "ledger_schema_version": SCHEMA_VERSION,
            "ledger_sequence": len(self.rows) + 1,
            "previous_hash": self.terminal_hash,
            **dict(value),
        }
        row = {**unsigned, "record_hash": sha256_bytes(canonical_bytes(unsigned))}
        secure_append_line(self.path, canonical_bytes(row) + b"\n")
        self.rows.append(row)
        return row


class ShadowStore:
    def __init__(self, state_dir: Path | str) -> None:
        self.state_dir = Path(state_dir)
        _validate_owned_directory(self.state_dir, create=True)
        self.manifest_dir = self.state_dir / "batch_manifests"
        _validate_owned_directory(self.manifest_dir, create=True)
        self.source_blob_dir = self.state_dir / "source_blobs"
        _validate_owned_directory(self.source_blob_dir, create=True)
        self.raw_ledger = AppendOnlyHashLedger(self.state_dir / "raw_bbo_ledger.jsonl")
        self.proposal_ledger = AppendOnlyHashLedger(
            self.state_dir / "proposal_stream_ledger.jsonl"
        )
        self.virtual_ledger = AppendOnlyHashLedger(
            self.state_dir / "virtual_execution_ledger.jsonl"
        )
        self.checkpoint_path = self.state_dir / "restart_checkpoint.json"
        self.manifests = self._load_manifests()
        self._replay_raw_state()
        self._validate_manifest_ledger_bindings()
        self._recover_checkpoint()

    def _load_manifests(self) -> dict[str, dict[str, Any]]:
        result = {}
        entries = sorted(self.manifest_dir.iterdir())
        if any(path.suffix != ".json" or path.is_symlink() for path in entries):
            raise ShadowCoreError("MANIFEST_MISMATCH", "unexpected manifest entry")
        for path in entries:
            try:
                raw = secure_read_bytes(path)
                assert raw is not None
                payload = json.loads(raw)
            except (OSError, json.JSONDecodeError) as error:
                raise ShadowCoreError("MANIFEST_MISMATCH", path.name) from error
            if (
                payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("manifest_sha256")
                != embedded_hash(payload, "manifest_sha256")
            ):
                raise ShadowCoreError("MANIFEST_MISMATCH", path.name)
            if path.stem != payload.get("batch_sha256"):
                raise ShadowCoreError("MANIFEST_MISMATCH", path.name)
            result[path.stem] = payload
        return result

    def _manifest_hashes(self) -> dict[str, str]:
        return {
            key: value["manifest_sha256"]
            for key, value in sorted(self.manifests.items())
        }

    def _validate_manifest_ledger_bindings(self) -> None:
        rows_by_batch: dict[str, list[dict[str, Any]]] = {}
        for row in self.raw_ledger.rows:
            batch = row.get("batch_sha256")
            if not isinstance(batch, str):
                raise ShadowCoreError("MANIFEST_LEDGER_BINDING_MISMATCH", "missing batch")
            rows_by_batch.setdefault(batch, []).append(row)
        if set(rows_by_batch) != set(self.manifests):
            raise ShadowCoreError("MANIFEST_LEDGER_BINDING_MISMATCH", "one-sided state")
        blob_entries = sorted(self.source_blob_dir.iterdir())
        if any(path.suffix != ".blob" or path.is_symlink() for path in blob_entries):
            raise ShadowCoreError("SOURCE_BLOB_BINDING_MISMATCH", "unexpected blob entry")
        expected_blobs = {
            f"{manifest['source_bytes_sha256']}.blob"
            for manifest in self.manifests.values()
            if manifest.get("source_bytes_available") is True
        }
        if {path.name for path in blob_entries} != expected_blobs:
            raise ShadowCoreError("SOURCE_BLOB_BINDING_MISMATCH", "one-sided blob state")
        for batch, manifest in self.manifests.items():
            rows = rows_by_batch[batch]
            start = manifest.get("raw_ledger_start_sequence")
            end = manifest.get("raw_ledger_end_sequence")
            if (
                not isinstance(start, int) or not isinstance(end, int)
                or start < 1 or end < start
                or [row["ledger_sequence"] for row in rows] != list(range(start, end + 1))
                or self.raw_ledger.hash_at(end) != manifest.get("raw_ledger_terminal_hash")
            ):
                raise ShadowCoreError("MANIFEST_LEDGER_BINDING_MISMATCH", batch)
            event_rows = [row for row in rows if row.get("record_type") == "BBO_EVENT"]
            failure_rows = [row for row in rows if row.get("record_type") == "BATCH_FAILURE"]
            if manifest.get("status") == "COMPLETED":
                if failure_rows or len(event_rows) != manifest.get("accepted_event_count"):
                    raise ShadowCoreError("MANIFEST_LEDGER_BINDING_MISMATCH", batch)
                times = [row["event"]["source_ts_ns"] for row in event_rows]
                if (
                    (min(times) if times else None) != manifest.get("first_source_ts_ns")
                    or (max(times) if times else None) != manifest.get("last_source_ts_ns")
                    or all(row["event"]["sequence"] is not None for row in event_rows)
                    is not manifest.get("lossless")
                ):
                    raise ShadowCoreError("MANIFEST_LEDGER_BINDING_MISMATCH", batch)
            elif len(failure_rows) != 1:
                raise ShadowCoreError("MANIFEST_LEDGER_BINDING_MISMATCH", batch)
            record_roots = [row["record_hash"] for row in rows]
            if sha256_bytes(canonical_bytes(record_roots)) != manifest.get(
                "raw_ledger_batch_record_roots_sha256"
            ):
                raise ShadowCoreError("MANIFEST_LEDGER_BINDING_MISMATCH", batch)
            if manifest.get("source_bytes_available") is True:
                blob = self.source_blob_dir / f"{manifest['source_bytes_sha256']}.blob"
                data = secure_read_bytes(blob)
                assert data is not None
                if (
                    sha256_bytes(data) != manifest["source_bytes_sha256"]
                    or len(data) != manifest["source_size_bytes"]
                ):
                    raise ShadowCoreError("SOURCE_BLOB_BINDING_MISMATCH", batch)
                if manifest.get("status") == "COMPLETED":
                    snapshot = SourceSnapshot(
                        manifest["source_name"], manifest["source_identity_sha256"],
                        manifest["source_format"], data, manifest["source_bytes_sha256"],
                        manifest["source_size_bytes"], manifest["source_mtime_ns"],
                        manifest["source_device"], manifest["source_inode"],
                    )
                    adapter = OfflineBBOFile(Path(manifest["source_name"]))
                    source_events = list(adapter.events(snapshot))
                    if (
                        len(source_events) != manifest.get("event_count")
                        or manifest.get("event_count")
                        != manifest.get("accepted_event_count") + manifest.get("exact_duplicate_count")
                        or all(event.sequence is not None for event in source_events)
                        is not manifest.get("lossless")
                    ):
                        raise ShadowCoreError("SOURCE_MANIFEST_COUNT_MISMATCH", batch)
            elif manifest.get("status") == "COMPLETED":
                raise ShadowCoreError("SOURCE_BLOB_BINDING_MISMATCH", batch)

    def _replay_raw_state(self) -> None:
        self.streams: dict[str, dict[str, int | None]] = {}
        self.seen_identities: dict[str, str] = {}
        self.halt_new_actions = False
        self.invalid_intervals: list[dict[str, Any]] = []
        for row in self.raw_ledger.rows:
            if row.get("record_type") == "BATCH_FAILURE":
                self.halt_new_actions = True
                continue
            if row.get("record_type") != "BBO_EVENT":
                raise ShadowCoreError("LEDGER_RECORD_INVALID", "raw record type")
            event = row["event"]
            identity = row["event_identity_sha256"]
            raw_hash = event["raw_payload_sha256"]
            existing = self.seen_identities.get(identity)
            if existing is not None and existing != raw_hash:
                raise ShadowCoreError("CONFLICTING_DUPLICATE", identity)
            self.seen_identities[identity] = raw_hash
            key = f'{event["provider_id"]}|{event["instrument"]}'
            state = self.streams.setdefault(key, {
                "source_ts_ns": None, "arrival_ts_ns": None, "sequence": None,
            })
            state["source_ts_ns"] = max(
                item for item in (state["source_ts_ns"], event["source_ts_ns"])
                if item is not None
            )
            state["arrival_ts_ns"] = max(
                item for item in (state["arrival_ts_ns"], event["arrival_ts_ns"])
                if item is not None
            )
            if event["sequence"] is not None:
                state["sequence"] = max(
                    item for item in (state["sequence"], event["sequence"])
                    if item is not None
                )
            if row.get("quality_reasons"):
                self.halt_new_actions = True
                self.invalid_intervals.append(row["invalid_interval"])
        self.source_files: dict[str, dict[str, Any]] = {}
        for manifest in self.manifests.values():
            if manifest.get("status") != "COMPLETED":
                self.halt_new_actions = True
                continue
            identity = manifest["source_identity_sha256"]
            prior = self.source_files.get(identity)
            if prior is None or manifest["source_size_bytes"] > prior["source_size_bytes"]:
                self.source_files[identity] = {
                    "source_name": manifest["source_name"],
                    "source_size_bytes": manifest["source_size_bytes"],
                    "source_bytes_sha256": manifest["source_bytes_sha256"],
                    "source_mtime_ns": manifest["source_mtime_ns"],
                    "batch_sha256": manifest["batch_sha256"],
                    "source_blob_sha256": manifest["source_bytes_sha256"],
                    "source_device": manifest["source_device"],
                    "source_inode": manifest["source_inode"],
                }

    def _checkpoint_core(self) -> dict[str, Any]:
        state = {
            "raw_ledger_sequence": len(self.raw_ledger.rows),
            "raw_ledger_terminal_hash": self.raw_ledger.terminal_hash,
            "proposal_ledger_sequence": len(self.proposal_ledger.rows),
            "proposal_ledger_terminal_hash": self.proposal_ledger.terminal_hash,
            "virtual_ledger_sequence": len(self.virtual_ledger.rows),
            "virtual_ledger_terminal_hash": self.virtual_ledger.terminal_hash,
            "halt_new_actions": self.halt_new_actions,
            "invalid_interval_count": len(self.invalid_intervals),
            "stream_states": self.streams,
            "source_files": self.source_files,
            "manifest_hashes": self._manifest_hashes(),
        }
        return {
            "schema_version": SCHEMA_VERSION,
            **state,
            "state_sha256": sha256_bytes(canonical_bytes(state)),
        }

    def write_checkpoint(self) -> dict[str, Any]:
        payload = self._checkpoint_core()
        payload["checkpoint_sha256"] = embedded_hash(payload, "checkpoint_sha256")
        atomic_json(self.checkpoint_path, payload)
        return payload

    def _recover_checkpoint(self) -> None:
        raw_checkpoint = secure_read_bytes(self.checkpoint_path, missing_ok=True)
        if raw_checkpoint is None:
            if self.raw_ledger.rows or self.proposal_ledger.rows or self.virtual_ledger.rows or self.manifests:
                raise ShadowCoreError("CHECKPOINT_MISSING_FOR_EXISTING_STATE")
            self.write_checkpoint()
            return
        try:
            current = json.loads(raw_checkpoint)
        except (OSError, json.JSONDecodeError) as error:
            raise ShadowCoreError("CHECKPOINT_MISMATCH") from error
        if (
            current.get("schema_version") != SCHEMA_VERSION
            or current.get("checkpoint_sha256")
            != embedded_hash(current, "checkpoint_sha256")
        ):
            raise ShadowCoreError("CHECKPOINT_MISMATCH")
        bindings = (
            ("raw_ledger_sequence", "raw_ledger_terminal_hash", self.raw_ledger),
            ("proposal_ledger_sequence", "proposal_ledger_terminal_hash", self.proposal_ledger),
            ("virtual_ledger_sequence", "virtual_ledger_terminal_hash", self.virtual_ledger),
        )
        for sequence_key, hash_key, ledger in bindings:
            sequence = current.get(sequence_key)
            if not isinstance(sequence, int) or sequence > len(ledger.rows):
                raise ShadowCoreError("CHECKPOINT_AHEAD_OF_LEDGER", sequence_key)
            if current.get(hash_key) != ledger.hash_at(sequence):
                raise ShadowCoreError("CHECKPOINT_MISMATCH", hash_key)
        expected = self._checkpoint_core()
        comparable = dict(current)
        comparable.pop("checkpoint_sha256", None)
        if comparable != expected:
            current_sources = comparable.get("source_files")
            current_manifests = comparable.get("manifest_hashes")
            if not isinstance(current_sources, dict) or not isinstance(current_manifests, dict):
                raise ShadowCoreError("CHECKPOINT_MISMATCH", "source anchors missing")
            if any(expected["source_files"].get(key) != value for key, value in current_sources.items()):
                raise ShadowCoreError("CHECKPOINT_MISMATCH", "source anchor changed")
            if any(expected["manifest_hashes"].get(key) != value for key, value in current_manifests.items()):
                raise ShadowCoreError("CHECKPOINT_MISMATCH", "manifest anchor changed")
            self.write_checkpoint()

    def _assert_mutable(self) -> None:
        if any(row.get("record_type") == "PERIOD_FINALIZED" for row in self.virtual_ledger.rows):
            raise ShadowCoreError("PERIOD_ALREADY_FINALIZED")

    def _store_source_blob(self, snapshot: SourceSnapshot, *, available: bool) -> None:
        if not available:
            return
        path = self.source_blob_dir / f"{snapshot.source_bytes_sha256}.blob"
        secure_exclusive_bytes(path, snapshot.data)

    def _event_identity(self, event: BBOEvent) -> str:
        return sha256_bytes(canonical_bytes({
            "provider_id": event.provider_id,
            "instrument": event.instrument,
            "provider_event_id": event.provider_event_id,
            "sequence": event.sequence,
            "source_ts_ns": (
                event.source_ts_ns
                if event.provider_event_id is None and event.sequence is None else None
            ),
            "heartbeat": event.heartbeat,
        }))

    def _quality(self, event: BBOEvent) -> tuple[list[str], dict[str, Any]]:
        key = f"{event.provider_id}|{event.instrument}"
        previous = self.streams.get(key)
        reasons = set()
        start = event.source_ts_ns
        if previous is not None:
            previous_source = int(previous["source_ts_ns"])
            previous_arrival = int(previous["arrival_ts_ns"])
            start = previous_source
            source_gap = event.source_ts_ns - previous_source
            arrival_gap = event.arrival_ts_ns - previous_arrival
            if source_gap < 0:
                reasons.add("OUT_OF_ORDER_EVENT")
            if arrival_gap < 0:
                reasons.add("CLOCK_REVERSAL")
            if source_gap > MAX_SOURCE_GAP_NS or arrival_gap > MAX_ARRIVAL_GAP_NS:
                reasons.add("SOURCE_OR_ARRIVAL_GAP")
            if arrival_gap > HEARTBEAT_EXPIRY_NS:
                reasons.add("HEARTBEAT_FAILURE")
            previous_sequence = previous.get("sequence")
            if event.sequence is not None and previous_sequence is not None:
                if event.sequence <= int(previous_sequence):
                    reasons.add("OUT_OF_ORDER_EVENT")
                elif event.sequence > int(previous_sequence) + 1:
                    reasons.add("SOURCE_OR_ARRIVAL_GAP")
        flag_map = {
            "RECONNECT": "RECONNECT_BOUNDARY",
            "RECONNECT_BOUNDARY": "RECONNECT_BOUNDARY",
            "CLOCK_REVERSAL": "CLOCK_REVERSAL",
            "OUT_OF_ORDER": "OUT_OF_ORDER_EVENT",
            "GAP": "SOURCE_OR_ARRIVAL_GAP",
            "HEARTBEAT_EXPIRED": "HEARTBEAT_FAILURE",
        }
        for flag in event.quality_flags:
            if flag in flag_map:
                reasons.add(flag_map[flag])
        interval = {
            "provider_id": event.provider_id,
            "instrument": event.instrument,
            "start_source_ts_ns": start,
            "end_source_ts_ns": event.source_ts_ns,
            "quality_reasons": sorted(reasons),
        }
        return sorted(reasons), interval

    def _advance_stream(self, event: BBOEvent) -> None:
        key = f"{event.provider_id}|{event.instrument}"
        state = self.streams.setdefault(key, {
            "source_ts_ns": None, "arrival_ts_ns": None, "sequence": None,
        })
        for field in ("source_ts_ns", "arrival_ts_ns"):
            value = getattr(event, field)
            prior = state[field]
            state[field] = value if prior is None else max(int(prior), value)
        if event.sequence is not None:
            prior_sequence = state["sequence"]
            state["sequence"] = (
                event.sequence if prior_sequence is None
                else max(int(prior_sequence), event.sequence)
            )

    def _write_manifest(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["manifest_sha256"] = embedded_hash(payload, "manifest_sha256")
        path = self.manifest_dir / f'{payload["batch_sha256"]}.json'
        current_raw = secure_read_bytes(path, missing_ok=True)
        if current_raw is not None:
            current = json.loads(current_raw)
            if current != payload:
                raise ShadowCoreError("MANIFEST_MISMATCH", path.name)
        else:
            atomic_json(path, payload)
        self.manifests[payload["batch_sha256"]] = payload
        return payload

    def _batch_failure(
        self,
        snapshot: SourceSnapshot,
        error: ShadowCoreError,
        event_count: int,
        accepted_count: int,
        duplicate_count: int,
        *,
        source_bytes_available: bool = True,
    ) -> None:
        self.halt_new_actions = True
        self._store_source_blob(snapshot, available=source_bytes_available)
        self.raw_ledger.append({
            "record_type": "BATCH_FAILURE",
            "batch_sha256": snapshot.source_bytes_sha256,
            "failure_code": error.code,
            "source_record_sha256": sha256_bytes(str(error).encode("utf-8")),
            "halt_new_actions_after": True,
        })
        payload = {
            "schema_version": SCHEMA_VERSION,
            "batch_sha256": snapshot.source_bytes_sha256,
            "status": "FAILED",
            "failure_code": error.code,
            "source_name": snapshot.source_name,
            "source_format": snapshot.format,
            "source_identity_sha256": snapshot.source_identity_sha256,
            "source_bytes_sha256": snapshot.source_bytes_sha256,
            "source_size_bytes": snapshot.source_size_bytes,
            "source_mtime_ns": snapshot.source_mtime_ns,
            "source_device": snapshot.source_device,
            "source_inode": snapshot.source_inode,
            "source_bytes_available": source_bytes_available,
            "event_count": event_count,
            "accepted_event_count": accepted_count,
            "exact_duplicate_count": duplicate_count,
            "first_source_ts_ns": None,
            "last_source_ts_ns": None,
            "lossless": False,
            "invalid_interval_count": 1,
            "raw_ledger_terminal_hash": self.raw_ledger.terminal_hash,
            "raw_ledger_start_sequence": min(
                row["ledger_sequence"] for row in self.raw_ledger.rows
                if row.get("batch_sha256") == snapshot.source_bytes_sha256
            ),
            "raw_ledger_end_sequence": max(
                row["ledger_sequence"] for row in self.raw_ledger.rows
                if row.get("batch_sha256") == snapshot.source_bytes_sha256
            ),
            "raw_ledger_batch_record_roots_sha256": sha256_bytes(canonical_bytes([
                row["record_hash"] for row in self.raw_ledger.rows
                if row.get("batch_sha256") == snapshot.source_bytes_sha256
            ])),
            "external_order_count": 0,
        }
        self._write_manifest(payload)
        self.write_checkpoint()

    def ingest(self, source: LocalFileEventSource) -> dict[str, Any]:
        self._assert_mutable()
        try:
            snapshot = source.snapshot()
        except ShadowCoreError as error:
            if not isinstance(source, OfflineBBOFile):
                raise
            snapshot = source.failure_snapshot(error)
            available = source._last_snapshot is not None
            self._batch_failure(
                snapshot, error, 0, 0, 0, source_bytes_available=available
            )
            raise
        self._store_source_blob(snapshot, available=True)
        existing_manifest = self.manifests.get(snapshot.source_bytes_sha256)
        if existing_manifest is not None:
            if (
                existing_manifest["source_size_bytes"] != snapshot.source_size_bytes
                or existing_manifest["source_bytes_sha256"] != snapshot.source_bytes_sha256
                or existing_manifest["source_device"] != snapshot.source_device
                or existing_manifest["source_inode"] != snapshot.source_inode
            ):
                raise ShadowCoreError("MANIFEST_MISMATCH", snapshot.source_name)
            if existing_manifest["status"] != "COMPLETED":
                raise ShadowCoreError("BATCH_PREVIOUSLY_FAILED", snapshot.source_name)
            blob = self.source_blob_dir / f"{snapshot.source_bytes_sha256}.blob"
            if secure_read_bytes(blob) != snapshot.data:
                raise ShadowCoreError("SOURCE_BLOB_BINDING_MISMATCH", snapshot.source_name)
            self._validate_manifest_ledger_bindings()
            return {"manifest": existing_manifest, "idempotent_reingest": True}
        previous = self.source_files.get(snapshot.source_identity_sha256)
        if previous is not None:
            if snapshot.source_size_bytes < previous["source_size_bytes"]:
                error = ShadowCoreError("SOURCE_PREFIX_CHANGED", snapshot.source_name)
                self._batch_failure(snapshot, error, 0, 0, 0)
                raise error
            prefix = snapshot.data[: previous["source_size_bytes"]]
            if sha256_bytes(prefix) != previous["source_bytes_sha256"]:
                error = ShadowCoreError("SOURCE_PREFIX_CHANGED", snapshot.source_name)
                self._batch_failure(snapshot, error, 0, 0, 0)
                raise error
        event_count = accepted_count = duplicate_count = 0
        accepted_times: list[int] = []
        batch_lossless = True
        new_invalid_count = 0
        try:
            for event in source.events(snapshot):
                event_count += 1
                identity = self._event_identity(event)
                existing_hash = self.seen_identities.get(identity)
                if existing_hash is not None:
                    if existing_hash == event.raw_payload_sha256:
                        duplicate_count += 1
                        continue
                    raise ShadowCoreError("CONFLICTING_DUPLICATE", identity)
                reasons, interval = self._quality(event)
                halt_before = self.halt_new_actions
                if reasons:
                    self.halt_new_actions = True
                    self.invalid_intervals.append(interval)
                    new_invalid_count += 1
                row = self.raw_ledger.append({
                    "record_type": "BBO_EVENT",
                    "batch_sha256": snapshot.source_bytes_sha256,
                    "event_identity_sha256": identity,
                    "event": event.as_dict(),
                    "quality_reasons": reasons,
                    "invalid_interval": interval if reasons else None,
                    "halt_new_actions_before": halt_before,
                    "halt_new_actions_after": self.halt_new_actions,
                    "new_decision_or_fill_allowed": not self.halt_new_actions,
                })
                self.seen_identities[identity] = event.raw_payload_sha256
                self._advance_stream(event)
                accepted_count += 1
                accepted_times.append(event.source_ts_ns)
                batch_lossless = batch_lossless and event.sequence is not None
                self.write_checkpoint()
        except ShadowCoreError as error:
            self._batch_failure(
                snapshot, error, event_count, accepted_count, duplicate_count
            )
            raise
        payload = {
            "schema_version": SCHEMA_VERSION,
            "batch_sha256": snapshot.source_bytes_sha256,
            "status": "COMPLETED",
            "source_name": snapshot.source_name,
            "source_format": snapshot.format,
            "source_identity_sha256": snapshot.source_identity_sha256,
            "source_bytes_sha256": snapshot.source_bytes_sha256,
            "source_size_bytes": snapshot.source_size_bytes,
            "source_mtime_ns": snapshot.source_mtime_ns,
            "source_device": snapshot.source_device,
            "source_inode": snapshot.source_inode,
            "source_bytes_available": True,
            "event_count": event_count,
            "accepted_event_count": accepted_count,
            "exact_duplicate_count": duplicate_count,
            "first_source_ts_ns": min(accepted_times) if accepted_times else None,
            "last_source_ts_ns": max(accepted_times) if accepted_times else None,
            "lossless": batch_lossless,
            "invalid_interval_count": new_invalid_count,
            "raw_ledger_terminal_hash": self.raw_ledger.terminal_hash,
            "raw_ledger_start_sequence": min(
                row["ledger_sequence"] for row in self.raw_ledger.rows
                if row.get("batch_sha256") == snapshot.source_bytes_sha256
            ),
            "raw_ledger_end_sequence": max(
                row["ledger_sequence"] for row in self.raw_ledger.rows
                if row.get("batch_sha256") == snapshot.source_bytes_sha256
            ),
            "raw_ledger_batch_record_roots_sha256": sha256_bytes(canonical_bytes([
                row["record_hash"] for row in self.raw_ledger.rows
                if row.get("batch_sha256") == snapshot.source_bytes_sha256
            ])),
            "halt_new_actions": self.halt_new_actions,
            "external_order_count": 0,
        }
        self._write_manifest(payload)
        self.source_files[snapshot.source_identity_sha256] = {
            "source_name": snapshot.source_name,
            "source_size_bytes": snapshot.source_size_bytes,
            "source_bytes_sha256": snapshot.source_bytes_sha256,
            "source_mtime_ns": snapshot.source_mtime_ns,
            "batch_sha256": snapshot.source_bytes_sha256,
            "source_blob_sha256": snapshot.source_bytes_sha256,
            "source_device": snapshot.source_device,
            "source_inode": snapshot.source_inode,
        }
        self.write_checkpoint()
        return {"manifest": payload, "idempotent_reingest": False}

    def status(self) -> dict[str, Any]:
        checkpoint = self._checkpoint_core()
        finalized = [
            row for row in self.virtual_ledger.rows
            if row.get("record_type") == "PERIOD_FINALIZED"
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "authority": AUTHORITY,
            "raw_ledger_records": len(self.raw_ledger.rows),
            "accepted_bbo_events": sum(
                row.get("record_type") == "BBO_EVENT" for row in self.raw_ledger.rows
            ),
            "proposal_stream_records": len(self.proposal_ledger.rows),
            "virtual_execution_records": len(self.virtual_ledger.rows),
            "batch_manifest_count": len(self.manifests),
            "halt_new_actions": self.halt_new_actions,
            "invalid_interval_count": len(self.invalid_intervals),
            "lossless": bool(self.manifests) and all(
                item.get("status") == "COMPLETED" and item.get("lossless") is True
                for item in self.manifests.values()
            ),
            "external_order_count": 0,
            "period_finalized": bool(finalized),
            "finalized_period_end_ts_ns": (
                finalized[-1]["period_end_ts_ns"] if finalized else None
            ),
            "state_sha256": checkpoint["state_sha256"],
            "raw_ledger_terminal_hash": self.raw_ledger.terminal_hash,
            "proposal_ledger_terminal_hash": self.proposal_ledger.terminal_hash,
            "virtual_ledger_terminal_hash": self.virtual_ledger.terminal_hash,
        }


def _accepted_event_rows(store: ShadowStore) -> list[dict[str, Any]]:
    return [
        row for row in store.raw_ledger.rows if row.get("record_type") == "BBO_EVENT"
    ]


def _bar_hash(row: dict[str, Any]) -> dict[str, Any]:
    row["bar_sha256"] = embedded_hash(row, "bar_sha256")
    return row


def completed_bars(store: ShadowStore) -> list[dict[str, Any]]:
    event_rows = _accepted_event_rows(store)
    streams: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in event_rows:
        event = row["event"]
        streams.setdefault(
            (event["provider_id"], event["instrument"]), []
        ).append(row)
    all_bars: list[dict[str, Any]] = []
    for (provider, instrument), rows in sorted(streams.items()):
        rows = sorted(rows, key=lambda item: item["ledger_sequence"])
        earliest = min(item["event"]["source_ts_ns"] for item in rows)
        latest = max(item["event"]["source_ts_ns"] for item in rows)
        first_bucket = earliest // (300 * 1_000_000_000) * (300 * 1_000_000_000)
        end_bucket = latest // (300 * 1_000_000_000) * (300 * 1_000_000_000)
        valid_count = 0
        m5: list[dict[str, Any]] = []
        for start in range(first_bucket, end_bucket, 300 * 1_000_000_000):
            end = start + 300 * 1_000_000_000
            bucket_rows = [
                item for item in rows
                if start <= item["event"]["source_ts_ns"] < end
            ]
            price_rows = [item for item in bucket_rows if item["event"]["bid"] is not None]
            reasons = {
                reason for item in bucket_rows for reason in item["quality_reasons"]
            }
            for interval in store.invalid_intervals:
                if (
                    interval["provider_id"] == provider
                    and interval["instrument"] == instrument
                    and interval["start_source_ts_ns"] < end
                    and interval["end_source_ts_ns"] >= start
                ):
                    reasons.update(interval["quality_reasons"])
            if not price_rows:
                reasons.add("MISSING_PRICE_EVENT")
            if start == first_bucket and (
                not price_rows or price_rows[0]["event"]["source_ts_ns"] != start
            ):
                reasons.add("PARTIAL_START_BAR")
            valid = not reasons
            if valid:
                valid_count += 1
            bids = [Decimal(item["event"]["bid"]) for item in price_rows]
            asks = [Decimal(item["event"]["ask"]) for item in price_rows]
            halt_at_end = any(
                item["halt_new_actions_after"]
                for item in rows if item["event"]["source_ts_ns"] < end
            )
            bar = _bar_hash({
                "schema_version": SCHEMA_VERSION,
                "provider_id": provider,
                "instrument": instrument,
                "timeframe": "M5",
                "start_ts_ns": start,
                "end_ts_ns": end,
                "completed": True,
                "valid": valid,
                "invalid_reasons": sorted(reasons),
                "event_count": len(price_rows),
                "completed_at_arrival_ts_ns": max(
                    (item["event"]["arrival_ts_ns"] for item in bucket_rows),
                    default=None,
                ),
                "bid": {
                    "open": format(bids[0], "f") if bids else None,
                    "high": format(max(bids), "f") if bids else None,
                    "low": format(min(bids), "f") if bids else None,
                    "close": format(bids[-1], "f") if bids else None,
                },
                "ask": {
                    "open": format(asks[0], "f") if asks else None,
                    "high": format(max(asks), "f") if asks else None,
                    "low": format(min(asks), "f") if asks else None,
                    "close": format(asks[-1], "f") if asks else None,
                },
                "completed_valid_m5_count_since_burn_in_start": valid_count,
                "burn_in_required_m5_bars": BURN_IN_M5_BARS,
                "burn_in_complete": valid_count >= BURN_IN_M5_BARS,
                "new_decision_or_fill_allowed": (
                    valid and valid_count >= BURN_IN_M5_BARS and not halt_at_end
                ),
                "input_event_hashes_sha256": sha256_bytes(canonical_bytes(
                    [
                        {
                            "event_identity_sha256": item["event_identity_sha256"],
                            "raw_payload_sha256": item["event"]["raw_payload_sha256"],
                        }
                        for item in bucket_rows
                    ]
                )),
            })
            m5.append(bar)
            all_bars.append(bar)
        for timeframe, seconds in TIMEFRAMES_SECONDS.items():
            if timeframe == "M5":
                continue
            size_ns = seconds * 1_000_000_000
            required = seconds // 300
            group_starts = sorted({bar["start_ts_ns"] // size_ns * size_ns for bar in m5})
            for start in group_starts:
                end = start + size_ns
                if end > latest:
                    continue
                bundle = [bar for bar in m5 if start <= bar["start_ts_ns"] < end]
                expected_starts = [
                    start + index * 300 * 1_000_000_000 for index in range(required)
                ]
                actual_starts = [bar["start_ts_ns"] for bar in bundle]
                complete_bundle = actual_starts == expected_starts
                valid = complete_bundle and all(bar["valid"] for bar in bundle)
                reasons = set()
                if not complete_bundle:
                    reasons.add("INCOMPLETE_M5_BUNDLE")
                for bar in bundle:
                    reasons.update(bar["invalid_reasons"])
                bids = [Decimal(bar["bid"]["open"]) for bar in bundle if bar["bid"]["open"]]
                bid_highs = [Decimal(bar["bid"]["high"]) for bar in bundle if bar["bid"]["high"]]
                bid_lows = [Decimal(bar["bid"]["low"]) for bar in bundle if bar["bid"]["low"]]
                asks = [Decimal(bar["ask"]["open"]) for bar in bundle if bar["ask"]["open"]]
                ask_highs = [Decimal(bar["ask"]["high"]) for bar in bundle if bar["ask"]["high"]]
                ask_lows = [Decimal(bar["ask"]["low"]) for bar in bundle if bar["ask"]["low"]]
                bar = _bar_hash({
                    "schema_version": SCHEMA_VERSION,
                    "provider_id": provider,
                    "instrument": instrument,
                    "timeframe": timeframe,
                    "start_ts_ns": start,
                    "end_ts_ns": end,
                    "completed": True,
                    "valid": valid,
                    "invalid_reasons": sorted(reasons),
                    "m5_bundle_count": len(bundle),
                    "required_m5_bundle_count": required,
                    "completed_at_arrival_ts_ns": max(
                        (item["completed_at_arrival_ts_ns"] for item in bundle
                         if item["completed_at_arrival_ts_ns"] is not None),
                        default=None,
                    ),
                    "bid": {
                        "open": format(bids[0], "f") if bids else None,
                        "high": format(max(bid_highs), "f") if bid_highs else None,
                        "low": format(min(bid_lows), "f") if bid_lows else None,
                        "close": bundle[-1]["bid"]["close"] if bundle else None,
                    },
                    "ask": {
                        "open": format(asks[0], "f") if asks else None,
                        "high": format(max(ask_highs), "f") if ask_highs else None,
                        "low": format(min(ask_lows), "f") if ask_lows else None,
                        "close": bundle[-1]["ask"]["close"] if bundle else None,
                    },
                    "burn_in_complete": bool(bundle) and bundle[-1]["burn_in_complete"],
                    "new_decision_or_fill_allowed": (
                        valid and bool(bundle)
                        and bundle[-1]["new_decision_or_fill_allowed"]
                    ),
                    "input_m5_hashes_sha256": sha256_bytes(canonical_bytes(
                        [item["bar_sha256"] for item in bundle]
                    )),
                })
                all_bars.append(bar)
    return sorted(
        all_bars,
        key=lambda item: (
            item["provider_id"], item["instrument"],
            TIMEFRAMES_SECONDS[item["timeframe"]], item["start_ts_ns"],
        ),
    )


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    signal_id: str
    decision_ts_ns: int
    instrument: str
    direction: int
    notional_jpy: float
    max_age_seconds: int
    strategy_version_sha256: str

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.signal_id:
            raise ShadowCoreError("INVALID_PROPOSAL_ID")
        parse_timestamp_ns(self.decision_ts_ns)
        if INSTRUMENT_RE.fullmatch(self.instrument) is None:
            raise ShadowCoreError("INVALID_INSTRUMENT", self.instrument)
        if self.direction not in (-1, 1):
            raise ShadowCoreError("INVALID_DIRECTION")
        if not math.isfinite(self.notional_jpy) or self.notional_jpy <= 0:
            raise ShadowCoreError("INVALID_NOTIONAL")
        if self.max_age_seconds <= 0:
            raise ShadowCoreError("FINITE_MAX_AGE_REQUIRED")
        if SHA256_RE.fullmatch(self.strategy_version_sha256) is None:
            raise ShadowCoreError("INVALID_STRATEGY_VERSION_HASH")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "signal_id": self.signal_id,
            "decision_ts_ns": self.decision_ts_ns,
            "instrument": self.instrument,
            "direction": self.direction,
            "notional_jpy": self.notional_jpy,
            "max_age_seconds": self.max_age_seconds,
            "strategy_version_sha256": self.strategy_version_sha256,
        }

    @property
    def proposal_sha256(self) -> str:
        return sha256_bytes(canonical_bytes(self.as_dict()))


def _valid_price_rows(store: ShadowStore, instrument: str) -> list[dict[str, Any]]:
    return sorted([
        row for row in _accepted_event_rows(store)
        if row["event"]["instrument"] == instrument
        and row["event"]["bid"] is not None
        and not row["quality_reasons"]
        and not row["halt_new_actions_before"]
    ], key=lambda item: (item["event"]["source_ts_ns"], item["ledger_sequence"]))


def _accounting_bbo(row: Mapping[str, Any]) -> accounting.BBO:
    event = row["event"]
    return accounting.BBO(
        event["instrument"], format_timestamp_ns(event["source_ts_ns"]),
        float(event["bid"]), float(event["ask"]),
    )


def _conversion_book(store: ShadowStore) -> accounting.ConversionBook:
    conversion_pairs = {"USD_JPY", "USD_CAD", "USD_CHF"}
    quotes = [
        _accounting_bbo(row) for row in _accepted_event_rows(store)
        if row["event"]["instrument"] in conversion_pairs
        and row["event"]["bid"] is not None
        and not row["quality_reasons"]
    ]
    return accounting.ConversionBook(quotes)


def _open_virtual_positions(store: ShadowStore) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in store.virtual_ledger.rows:
        if row.get("record_type") == "VIRTUAL_FILL":
            result[row["position_id"]] = row
        elif row.get("record_type") == "VIRTUAL_CLOSE":
            result.pop(row["position_id"], None)
    return result


def _portfolio_currency_inventory(
    positions: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    inventory: dict[str, float] = {}
    for position in positions:
        per_position = position.get("position_currency_inventory", {})
        if not isinstance(per_position, Mapping):
            raise ShadowCoreError("VIRTUAL_LEDGER_INVENTORY_INVALID")
        for currency, amount in per_position.items():
            inventory[str(currency)] = inventory.get(str(currency), 0.0) + float(amount)
    return {
        currency: amount
        for currency, amount in sorted(inventory.items())
        if abs(amount) > 1e-12
    }


def _execution_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("worker_arm")), str(row.get("cost_arm"))


def route_shared_proposal(
    store: ShadowStore,
    proposal: Proposal,
    arm_decisions: Mapping[str, Mapping[str, Any]],
    *,
    actual_llm_called: bool = False,
) -> dict[str, Any]:
    store._assert_mutable()
    if actual_llm_called:
        raise ShadowCoreError("ACTUAL_LLM_CALL_NOT_AUTHORIZED_IN_CHECKPOINT")
    if store.halt_new_actions:
        raise ShadowCoreError("DATA_QUALITY_HALT")
    if set(arm_decisions) != set(WORKER_ARMS):
        raise ShadowCoreError("WORKER_ARM_MISMATCH")
    decision_stream = {
        arm: dict(arm_decisions[arm]) for arm in sorted(arm_decisions)
    }
    decision_stream_sha256 = sha256_bytes(canonical_bytes(decision_stream))
    allowed = {"action", "pair_cap", "currency_cap", "provenance"}
    forbidden = {"direction", "fill", "order", "tp", "sl", "leverage", "hard_guard"}
    for arm, decision in arm_decisions.items():
        if forbidden & set(decision) or set(decision) - allowed:
            raise ShadowCoreError("LLM_POLICY_SCOPE_VIOLATION", arm)
        if decision.get("action") not in {"ENABLE", "FREEZE", "UNWIND"}:
            raise ShadowCoreError("LLM_POLICY_SCOPE_VIOLATION", arm)
        if not isinstance(decision.get("provenance"), str) or not decision["provenance"]:
            raise ShadowCoreError("LLM_POLICY_SCOPE_VIOLATION", "provenance")
        for cap in ("pair_cap", "currency_cap"):
            value = decision.get(cap, 1.0)
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise ShadowCoreError("LLM_POLICY_SCOPE_VIOLATION", cap)
    eligible = [
        bar for bar in completed_bars(store)
        if bar["instrument"] == proposal.instrument
        and bar["timeframe"] == "M5"
        and bar["end_ts_ns"] == proposal.decision_ts_ns
        and bar["new_decision_or_fill_allowed"]
    ]
    if not eligible:
        raise ShadowCoreError("DECISION_BAR_NOT_CAUSALLY_ELIGIBLE")
    if len(eligible) != 1 or eligible[0].get("completed_at_arrival_ts_ns") is None:
        raise ShadowCoreError("DECISION_ARRIVAL_CHRONOLOGY_MISSING")
    decision_bar = eligible[0]
    decision_arrival_ts_ns = int(decision_bar["completed_at_arrival_ts_ns"])
    existing = [
        row for row in store.proposal_ledger.rows
        if row.get("proposal_id") == proposal.proposal_id
    ]
    if len(existing) > 1:
        raise ShadowCoreError("DUPLICATE_PROPOSAL_RECORD")
    proposal_preexisted = bool(existing)
    if proposal_preexisted:
        if (
            existing[0]["proposal_sha256"] != proposal.proposal_sha256
            or existing[0].get("decision_stream_sha256")
            != decision_stream_sha256
            or existing[0].get("arm_decisions") != decision_stream
        ):
            raise ShadowCoreError("CONFLICTING_PROPOSAL_ID")
    else:
        store.proposal_ledger.append({
            "record_type": "PROPOSAL",
            "proposal_id": proposal.proposal_id,
            "proposal_sha256": proposal.proposal_sha256,
            "proposal": proposal.as_dict(),
            "worker_arms": list(WORKER_ARMS),
            "arm_decisions": decision_stream,
            "decision_stream_sha256": decision_stream_sha256,
            "actual_llm_called": False,
        })
    expected_keys = {
        (worker_arm, scenario_name)
        for worker_arm in WORKER_ARMS
        if arm_decisions[worker_arm]["action"] == "ENABLE"
        for scenario_name in EXECUTION_SCENARIOS
    }
    proposal_rows = [
        row for row in store.virtual_ledger.rows
        if row.get("proposal_sha256") == proposal.proposal_sha256
    ]
    existing_orders = [
        row for row in proposal_rows if row.get("record_type") == "EXPECTED_ORDER"
    ]
    existing_fills = [
        row for row in proposal_rows if row.get("record_type") == "VIRTUAL_FILL"
    ]
    for label, rows in (("EXPECTED_ORDER", existing_orders), ("VIRTUAL_FILL", existing_fills)):
        keys = [_execution_key(row) for row in rows]
        if len(keys) != len(set(keys)) or set(keys) - expected_keys:
            raise ShadowCoreError("VIRTUAL_EXECUTION_STREAM_MISMATCH", label)
    order_by_key = {_execution_key(row): row for row in existing_orders}
    fill_by_key = {_execution_key(row): row for row in existing_fills}
    if set(fill_by_key) - set(order_by_key):
        raise ShadowCoreError("VIRTUAL_EXECUTION_STREAM_MISMATCH", "fill without order")
    missing_keys = expected_keys - set(fill_by_key)
    if not missing_keys:
        store.write_checkpoint()
        return {
            "proposal_sha256": proposal.proposal_sha256,
            "decision_stream_sha256": decision_stream_sha256,
            "idempotent": proposal_preexisted,
            "resumed_partial_execution": False,
            "worker_arms": list(WORKER_ARMS),
            "cost_arms": sorted(EXECUTION_SCENARIOS),
            "virtual_fill_count": len(fill_by_key),
            "same_content_addressed_proposal_all_arms": True,
            "external_order_count": 0,
        }
    try:
        conversion_book = _conversion_book(store)
    except accounting.AccountingError as error:
        raise ShadowCoreError("JPY_ACCOUNTING_FAILURE", str(error)) from error
    pair_rows = _valid_price_rows(store, proposal.instrument)
    opens = _open_virtual_positions(store)
    newly_created_fills = []
    for worker_arm in WORKER_ARMS:
        decision = arm_decisions[worker_arm]
        if decision["action"] != "ENABLE":
            continue
        for scenario_name, scenario_spec in EXECUTION_SCENARIOS.items():
            execution_key = (worker_arm, scenario_name)
            if execution_key not in missing_keys:
                continue
            latency_ns = int(scenario_spec["latency_ns"])
            target_arrival = decision_arrival_ts_ns + latency_ns
            fill_row = next(
                (
                    row for row in pair_rows
                    if row["event"]["source_ts_ns"] > proposal.decision_ts_ns
                    and row["event"]["arrival_ts_ns"] >= target_arrival
                ),
                None,
            )
            if fill_row is None:
                raise ShadowCoreError("NO_CAUSAL_EXECUTABLE_FILL", scenario_name)
            key_prefix = f"{worker_arm}|{scenario_name}|"
            matching_opens = [
                row for row in opens.values()
                if row["worker_arm"] == worker_arm and row["cost_arm"] == scenario_name
            ]
            open_notional = sum(row["notional_jpy"] for row in matching_opens)
            if open_notional + proposal.notional_jpy > INITIAL_EQUITY_JPY * FIXED_GROSS_CAP:
                raise ShadowCoreError("MARGIN_GUARD")
            pair_open_notional = sum(
                row["notional_jpy"] for row in matching_opens
                if row["instrument"] == proposal.instrument
            )
            if pair_open_notional + proposal.notional_jpy > (
                INITIAL_EQUITY_JPY * float(decision.get("pair_cap", 1.0))
            ):
                raise ShadowCoreError("PAIR_CAP_GUARD")
            proposal_currencies = set(accounting.pair_currencies(proposal.instrument))
            for currency in proposal_currencies:
                currency_notional = sum(
                    row["notional_jpy"] for row in matching_opens
                    if currency in accounting.pair_currencies(row["instrument"])
                )
                if currency_notional + proposal.notional_jpy > (
                    INITIAL_EQUITY_JPY * float(decision.get("currency_cap", 1.0))
                ):
                    raise ShadowCoreError("CURRENCY_CAP_GUARD", currency)
            position_id = sha256_bytes(canonical_bytes({
                "proposal_sha256": proposal.proposal_sha256,
                "worker_arm": worker_arm,
                "cost_arm": scenario_name,
            }))
            event = fill_row["event"]
            event_time = format_timestamp_ns(event["source_ts_ns"])
            bbo = _accounting_bbo(fill_row)
            try:
                position = accounting.size_position(
                    position_id, proposal.instrument, proposal.direction,
                    proposal.notional_jpy, event_time, bbo, conversion_book,
                )
            except accounting.AccountingError as error:
                raise ShadowCoreError("JPY_ACCOUNTING_FAILURE", str(error)) from error
            expected_order = {
                "record_type": "EXPECTED_ORDER",
                "proposal_id": proposal.proposal_id,
                "proposal_sha256": proposal.proposal_sha256,
                "worker_arm": worker_arm,
                "cost_arm": scenario_name,
                "direction": proposal.direction,
                "notional_jpy": proposal.notional_jpy,
                "decision_ts_ns": proposal.decision_ts_ns,
                "decision_arrival_ts_ns": decision_arrival_ts_ns,
                "minimum_fill_arrival_ts_ns": target_arrival,
                "latency_ns": latency_ns,
                "external_dispatch_allowed": False,
                "external_order_count": 0,
            }
            prior_order = order_by_key.get(execution_key)
            if prior_order is None:
                prior_order = store.virtual_ledger.append(expected_order)
                order_by_key[execution_key] = prior_order
            elif any(prior_order.get(key) != value for key, value in expected_order.items()):
                raise ShadowCoreError("VIRTUAL_EXECUTION_STREAM_MISMATCH", "order changed")
            position_inventory = position.currency_inventory()
            portfolio_inventory_after = _portfolio_currency_inventory([
                *matching_opens,
                {"position_currency_inventory": position_inventory},
            ])
            fill = store.virtual_ledger.append({
                "record_type": "VIRTUAL_FILL",
                "proposal_id": proposal.proposal_id,
                "proposal_sha256": proposal.proposal_sha256,
                "position_id": position_id,
                "worker_arm": worker_arm,
                "cost_arm": scenario_name,
                "instrument": proposal.instrument,
                "direction": proposal.direction,
                "notional_jpy": proposal.notional_jpy,
                "units": position.units,
                "entry_ts_ns": event["source_ts_ns"],
                "entry_arrival_ts_ns": event["arrival_ts_ns"],
                "decision_arrival_ts_ns": decision_arrival_ts_ns,
                "minimum_fill_arrival_ts_ns": target_arrival,
                "latency_ns": latency_ns,
                "entry_bid": event["bid"],
                "entry_ask": event["ask"],
                "source_provider_id": event["provider_id"],
                "source_event_identity_sha256": fill_row["event_identity_sha256"],
                "source_raw_payload_sha256": event["raw_payload_sha256"],
                "sizing_quote_cashflow_per_unit": position.sizing_quote_cashflow_per_unit,
                "sizing_jpy_cashflow_per_unit": position.sizing_jpy_cashflow_per_unit,
                "max_age_seconds": proposal.max_age_seconds,
                "position_currency_inventory": position_inventory,
                "portfolio_currency_inventory_after": portfolio_inventory_after,
                "margin_guard_passed": True,
                "external_dispatch_allowed": False,
                "external_order_count": 0,
                "actual_llm_called": False,
                "structured_policy_sha256": sha256_bytes(canonical_bytes(decision)),
                "position_key_prefix": key_prefix,
            })
            opens[position_id] = fill
            fill_by_key[execution_key] = fill
            newly_created_fills.append(fill)
    store.write_checkpoint()
    return {
        "proposal_sha256": proposal.proposal_sha256,
        "decision_stream_sha256": decision_stream_sha256,
        "idempotent": False,
        "resumed_partial_execution": proposal_preexisted,
        "worker_arms": list(WORKER_ARMS),
        "cost_arms": sorted(EXECUTION_SCENARIOS),
        "virtual_fill_count": len(fill_by_key),
        "new_virtual_fill_count": len(newly_created_fills),
        "same_content_addressed_proposal_all_arms": (
            {row["proposal_sha256"] for row in fill_by_key.values()}
            in ({proposal.proposal_sha256}, set())
        ),
        "external_order_count": 0,
    }


def _position_from_fill(fill: Mapping[str, Any]) -> accounting.Position:
    entry_bbo = accounting.BBO(
        fill["instrument"], format_timestamp_ns(fill["entry_ts_ns"]),
        float(fill["entry_bid"]), float(fill["entry_ask"]),
    )
    return accounting.Position(
        fill["position_id"], fill["instrument"], int(fill["direction"]),
        float(fill["units"]), float(fill["notional_jpy"]),
        format_timestamp_ns(fill["entry_ts_ns"]), entry_bbo,
        float(fill["sizing_quote_cashflow_per_unit"]),
        float(fill["sizing_jpy_cashflow_per_unit"]),
    )


def _fresh_price_rows_at_cutoff(
    store: ShadowStore, instruments: set[str], cutoff_ts_ns: int
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for instrument in sorted(instruments):
        candidates = [
            row for row in _valid_price_rows(store, instrument)
            if row["event"]["source_ts_ns"] <= cutoff_ts_ns
            and row["event"]["arrival_ts_ns"] <= cutoff_ts_ns
        ]
        if not candidates:
            raise ShadowCoreError("TERMINAL_BBO_MISSING", instrument)
        row = candidates[-1]
        source_age = cutoff_ts_ns - row["event"]["source_ts_ns"]
        arrival_age = cutoff_ts_ns - row["event"]["arrival_ts_ns"]
        if (
            source_age > MAX_SOURCE_GAP_NS
            or arrival_age > MAX_ARRIVAL_GAP_NS
            or arrival_age > HEARTBEAT_EXPIRY_NS
        ):
            raise ShadowCoreError("TERMINAL_DATA_STALE", instrument)
        result[instrument] = row
    return result


def finalize_period(store: ShadowStore, period_end_ts_ns: int) -> dict[str, Any]:
    period_end_ts_ns = parse_timestamp_ns(period_end_ts_ns)
    all_finalizations = [
        row for row in store.virtual_ledger.rows
        if row.get("record_type") == "PERIOD_FINALIZED"
    ]
    prior = [
        row for row in all_finalizations
        if row.get("period_end_ts_ns") == period_end_ts_ns
    ]
    if prior:
        if len(prior) != 1 or store.virtual_ledger.rows[-1] != prior[-1]:
            raise ShadowCoreError("FINALIZATION_STATE_MISMATCH")
        if _open_virtual_positions(store):
            raise ShadowCoreError("FINALIZED_SUMMARY_OPEN_INVENTORY_MISMATCH")
        summary = prior[-1]["summary"]
        if (
            summary.get("terminal_inventory_mtm_jpy") != 0.0
            or summary.get("terminal_currency_inventory") != {}
            or summary.get("pre_finalization_virtual_ledger_terminal_hash")
            != prior[-1]["previous_hash"]
            or summary.get("summary_sha256") != embedded_hash(summary, "summary_sha256")
        ):
            raise ShadowCoreError("FINALIZATION_STATE_MISMATCH")
        return summary
    if all_finalizations:
        raise ShadowCoreError("PERIOD_ALREADY_FINALIZED")
    future_proposals = [
        row for row in store.proposal_ledger.rows
        if row.get("record_type") == "PROPOSAL"
        and row["proposal"]["decision_ts_ns"] > period_end_ts_ns
    ]
    future_fills = [
        row for row in store.virtual_ledger.rows
        if row.get("record_type") == "VIRTUAL_FILL"
        and row["entry_ts_ns"] > period_end_ts_ns
    ]
    if future_proposals or future_fills:
        raise ShadowCoreError("FUTURE_ACTIVITY_BEYOND_FINALIZATION_CUTOFF")
    opens = _open_virtual_positions(store)
    required_instruments = {fill["instrument"] for fill in opens.values()}
    if any(not instrument.endswith("_JPY") for instrument in required_instruments):
        required_instruments.add("USD_JPY")
    _fresh_price_rows_at_cutoff(store, required_instruments, period_end_ts_ns)
    try:
        conversion_book = _conversion_book(store)
    except accounting.AccountingError as error:
        raise ShadowCoreError("JPY_ACCOUNTING_FAILURE", str(error)) from error
    planned = []
    for position_id, fill in opens.items():
        rows = _valid_price_rows(store, fill["instrument"])
        expiry = fill["entry_ts_ns"] + fill["max_age_seconds"] * 1_000_000_000
        max_age_row = next(
            (row for row in rows
             if expiry <= row["event"]["source_ts_ns"] <= period_end_ts_ns
             and row["event"]["arrival_ts_ns"] <= period_end_ts_ns),
            None,
        )
        if max_age_row is not None:
            close_row, reason = max_age_row, "MAX_AGE"
        else:
            candidates = [
                row for row in rows
                if fill["entry_ts_ns"] < row["event"]["source_ts_ns"] <= period_end_ts_ns
                and row["event"]["arrival_ts_ns"] <= period_end_ts_ns
            ]
            if not candidates:
                raise ShadowCoreError("TERMINAL_BBO_MISSING", fill["instrument"])
            close_row, reason = candidates[-1], "TERMINAL_LIQUIDATION"
        planned.append((close_row["event"]["source_ts_ns"], position_id, fill, close_row, reason))
    remaining_positions = dict(opens)
    for _, position_id, fill, close_row, reason in sorted(planned):
        position = _position_from_fill(fill)
        scenario = EXECUTION_SCENARIOS[fill["cost_arm"]]["accounting"]
        try:
            evaluation = accounting.evaluate_position(
                position, format_timestamp_ns(close_row["event"]["source_ts_ns"]),
                _accounting_bbo(close_row), conversion_book, scenario,
            )
        except accounting.AccountingError as error:
            raise ShadowCoreError("JPY_ACCOUNTING_FAILURE", str(error)) from error
        inventory_before = _portfolio_currency_inventory(
            list(remaining_positions.values())
        )
        remaining_positions.pop(position_id)
        inventory_after = _portfolio_currency_inventory(
            list(remaining_positions.values())
        )
        store.virtual_ledger.append({
            "record_type": "VIRTUAL_CLOSE",
            "proposal_id": fill["proposal_id"],
            "proposal_sha256": fill["proposal_sha256"],
            "position_id": position_id,
            "worker_arm": fill["worker_arm"],
            "cost_arm": fill["cost_arm"],
            "close_ts_ns": close_row["event"]["source_ts_ns"],
            "reason": reason,
            "realized_net_jpy": evaluation["net_jpy"],
            "gross_jpy": evaluation["gross_jpy"],
            "total_realized_cost_jpy": evaluation["total_realized_cost_jpy"],
            "financing_cost_jpy": evaluation["financing_cost_jpy"],
            "position_currency_inventory_before_close": fill[
                "position_currency_inventory"
            ],
            "portfolio_currency_inventory_before_close": inventory_before,
            "portfolio_currency_inventory_after_close": inventory_after,
            "terminal_inventory_mtm_jpy": 0.0,
            "evaluation_sha256": evaluation["evaluation_sha256"],
            "external_order_count": 0,
        })
    remaining = _open_virtual_positions(store)
    if remaining:
        raise ShadowCoreError("TERMINAL_INVENTORY_REMAINS")
    combinations = {}
    for worker_arm in WORKER_ARMS:
        for scenario_name in sorted(EXECUTION_SCENARIOS):
            net = sum(
                row["realized_net_jpy"] for row in store.virtual_ledger.rows
                if row.get("record_type") == "VIRTUAL_CLOSE"
                and row["worker_arm"] == worker_arm
                and row["cost_arm"] == scenario_name
            )
            combinations[f"{worker_arm}|{scenario_name}"] = {
                "ending_equity_jpy": INITIAL_EQUITY_JPY + net,
                "realized_net_jpy": net,
                "terminal_inventory_mtm_jpy": 0.0,
                "terminal_open_positions": 0,
                "margin_guard_violations": 0,
            }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "period_end_ts_ns": period_end_ts_ns,
        "worker_cost_arms": combinations,
        "max_age_close_count": sum(
            row.get("record_type") == "VIRTUAL_CLOSE" and row.get("reason") == "MAX_AGE"
            for row in store.virtual_ledger.rows
        ),
        "terminal_liquidation_count": sum(
            row.get("record_type") == "VIRTUAL_CLOSE"
            and row.get("reason") == "TERMINAL_LIQUIDATION"
            for row in store.virtual_ledger.rows
        ),
        "terminal_inventory_mtm_jpy": 0.0,
        "terminal_currency_inventory": {},
        "external_order_count": 0,
        "live_authority": False,
        "pre_finalization_virtual_ledger_terminal_hash": store.virtual_ledger.terminal_hash,
        "finalized_mutation_forbidden": True,
    }
    summary["summary_sha256"] = embedded_hash(summary, "summary_sha256")
    store.virtual_ledger.append({
        "record_type": "PERIOD_FINALIZED",
        "period_end_ts_ns": period_end_ts_ns,
        "summary": summary,
    })
    store.write_checkpoint()
    return summary


def validate_schema(path: Path) -> dict[str, Any]:
    source = OfflineBBOFile(path)
    snapshot = source.snapshot()
    events = list(source.events(snapshot))
    if not events:
        raise ShadowCoreError("EMPTY_SOURCE_BATCH")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_name": snapshot.source_name,
        "source_bytes_sha256": snapshot.source_bytes_sha256,
        "source_size_bytes": snapshot.source_size_bytes,
        "source_mtime_ns": snapshot.source_mtime_ns,
        "event_count": len(events),
        "first_source_ts_ns": min(event.source_ts_ns for event in events),
        "last_source_ts_ns": max(event.source_ts_ns for event in events),
        "lossless": all(event.sequence is not None for event in events),
        "source_mutated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate-schema")
    validate_parser.add_argument("source", type=Path)
    ingest_parser = subparsers.add_parser("ingest-batch")
    ingest_parser.add_argument("source", type=Path)
    ingest_parser.add_argument("--state-dir", type=Path, required=True)
    for command in ("resume", "status"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--state-dir", type=Path, required=True)
    finalize_parser = subparsers.add_parser("finalize-period")
    finalize_parser.add_argument("--state-dir", type=Path, required=True)
    finalize_parser.add_argument("--period-end-ts-ns", required=True)
    args = parser.parse_args()
    if args.command == "validate-schema":
        result = validate_schema(args.source)
    else:
        store = ShadowStore(args.state_dir)
        if args.command == "ingest-batch":
            result = store.ingest(OfflineBBOFile(args.source))
        elif args.command in {"resume", "status"}:
            result = store.status()
        else:
            result = finalize_period(store, parse_timestamp_ns(args.period_end_ts_ns))
    print(json.dumps({"ok": True, "result": result}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ShadowCoreError as error:
        print(json.dumps({
            "ok": False,
            "error_code": error.code,
            "error_sha256": sha256_bytes(str(error).encode("utf-8")),
        }, sort_keys=True))
        raise SystemExit(2)
