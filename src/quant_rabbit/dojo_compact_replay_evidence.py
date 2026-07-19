"""Compact append-only evidence for deterministic DOJO replay segments.

The compact ledger records market *events*, not every replayed candle.  Its
first event binds the immutable source, replay plan, bot configuration, and
cost configuration; every later event is chained to it with canonical JSON
SHA-256.  A completed segment can be closed by one atomically published
manifest that binds the exact event bytes.

This is local research evidence.  It neither independently proves the market
source nor grants promotion, live permission, order authority, or broker-write
authority.  A filesystem owner can still delete the complete directory, so
external custody is required for durable monotonic proof.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


EVENT_CONTRACT: Final = "QR_DOJO_COMPACT_REPLAY_EVENT_V1"
MANIFEST_CONTRACT: Final = "QR_DOJO_COMPACT_REPLAY_MANIFEST_V1"
SCHEMA_VERSION: Final = 1
EVENTS_NAME: Final = "events.jsonl"
LOCK_NAME: Final = "writer.lock"
MANIFEST_NAME: Final = "final-manifest.json"
MAX_EVENT_LINE_BYTES: Final = 64 * 1024
MAX_EVENTS: Final = 5_000_000
MAX_LEDGER_BYTES: Final = 16 * 1024 * 1024 * 1024
ZERO_SHA256: Final = "0" * 64

_HEX64: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}\Z")
_PAIR: Final = re.compile(r"[A-Z]{3}_[A-Z]{3}\Z")
_CURRENCY: Final = re.compile(r"[A-Z]{3}\Z")
_REASON_CODE: Final = re.compile(r"[A-Z][A-Z0-9_]{0,95}\Z")
_UTC_TIMESTAMP: Final = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z\Z"
)
_EVENT_TYPES: Final = (
    "SEGMENT_START",
    "BOT",
    "ORDER",
    "FILL",
    "EXIT",
    "MARGIN",
    "CHECKPOINT",
    "SEGMENT_STOP",
)
_EVENT_TYPE_SET: Final = frozenset(_EVENT_TYPES)
_BINDING_KEYS: Final = frozenset(
    {"source_sha256", "plan_sha256", "config_sha256", "cost_sha256"}
)
_EVENT_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "evidence_id",
        "sequence",
        "event_id",
        "event_type",
        "event_at_utc",
        "previous_event_sha256",
        "bindings_sha256",
        "payload",
        "classification",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "order_authority",
        "broker_mutation_allowed",
        "event_sha256",
    }
)
_PAYLOAD_KEYS: Final = {
    "SEGMENT_START": frozenset(
        {
            "segment_id",
            "replay_start_utc",
            "replay_end_utc",
            "initial_balance_jpy",
            "account_currency",
            "bindings",
        }
    ),
    "BOT": frozenset(
        {
            "bot_id",
            "decision_id",
            "pair",
            "decision",
            "reason_code",
            "signal_sha256",
            "related_order_or_trade_id",
        }
    ),
    "ORDER": frozenset(
        {
            "order_id",
            "pair",
            "side",
            "order_type",
            "status",
            "units",
            "requested_price",
            "stop_loss_price",
            "take_profit_price",
        }
    ),
    "FILL": frozenset(
        {
            "fill_id",
            "order_id",
            "trade_id",
            "pair",
            "side",
            "units",
            "fill_price",
            "spread_pips",
            "slippage_pips",
            "fee_jpy",
        }
    ),
    "EXIT": frozenset(
        {
            "exit_id",
            "trade_id",
            "pair",
            "reason",
            "units",
            "exit_price",
            "quote_to_jpy_rate",
            "realized_pnl_jpy",
            "financing_jpy",
        }
    ),
    "MARGIN": frozenset(
        {
            "balance_jpy",
            "equity_jpy",
            "used_margin_jpy",
            "free_margin_jpy",
        }
    ),
    "CHECKPOINT": frozenset(
        {
            "checkpoint_id",
            "source_cursor_sha256",
            "state_sha256",
            "open_trade_count",
            "pending_order_count",
            "balance_jpy",
            "equity_jpy",
        }
    ),
    "SEGMENT_STOP": frozenset(
        {
            "status",
            "reason_code",
            "source_cursor_sha256",
            "terminal_balance_jpy",
            "terminal_equity_jpy",
            "open_trade_count",
            "pending_order_count",
        }
    ),
}
_MANIFEST_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "evidence_id",
        "classification",
        "bindings",
        "bindings_sha256",
        "event_count",
        "event_type_counts",
        "events_sha256",
        "first_event_sha256",
        "final_event_sha256",
        "started_at_utc",
        "stopped_at_utc",
        "total_event_bytes",
        "finalized_at_utc",
        "external_witness_status",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "order_authority",
        "broker_mutation_allowed",
        "manifest_sha256",
    }
)


class CompactReplayEvidenceError(ValueError):
    """The compact replay evidence is malformed, unsafe, or contradictory."""


@dataclass(frozen=True)
class CompactReplayEvidenceSnapshot:
    """Verified, bounded summary of a compact replay event chain."""

    evidence_id: str
    bindings: Mapping[str, str]
    event_count: int
    event_type_counts: Mapping[str, int]
    total_event_bytes: int
    events_sha256: str
    first_event_sha256: str
    final_event_sha256: str
    started_at_utc: str
    stopped_at_utc: str | None
    finalized: bool
    manifest: Mapping[str, Any] | None
    proof_eligible: bool = False
    promotion_eligible: bool = False
    live_permission: bool = False
    order_authority: str = "NONE"
    broker_mutation_allowed: bool = False


@dataclass
class _ReplayState:
    evidence_id: str | None = None
    bindings: dict[str, str] | None = None
    bindings_sha256: str | None = None
    event_count: int = 0
    event_type_counts: Counter[str] | None = None
    event_ids: set[str] | None = None
    previous_event_sha256: str = ZERO_SHA256
    first_event_sha256: str | None = None
    first_timestamp: str | None = None
    previous_timestamp: datetime | None = None
    replay_start: datetime | None = None
    replay_end: datetime | None = None
    stop_timestamp: str | None = None
    stopped: bool = False
    initial_balance_jpy: float | None = None
    calculated_balance_jpy: float | None = None
    orders: dict[
        str,
        tuple[
            str,
            str,
            str,
            float,
            str,
            float | None,
            float | None,
            float | None,
        ],
    ] | None = None
    open_trades: dict[str, tuple[str, str, float, float]] | None = None
    closed_trade_ids: set[str] | None = None
    decision_ids: set[str] | None = None
    fill_ids: set[str] | None = None
    exit_ids: set[str] | None = None
    checkpoint_ids: set[str] | None = None

    def __post_init__(self) -> None:
        self.event_type_counts = Counter()
        self.event_ids = set()
        self.orders = {}
        self.open_trades = {}
        self.closed_trade_ids = set()
        self.decision_ids = set()
        self.fill_ids = set()
        self.exit_ids = set()
        self.checkpoint_ids = set()


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of strict canonical JSON."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class CompactReplayEvidenceWriter:
    """Single-process writer for one new compact replay evidence directory."""

    def __init__(
        self,
        directory: Path,
        directory_fd: int,
        lock_fd: int,
        events_fd: int,
        state: _ReplayState,
        total_event_bytes: int,
    ) -> None:
        self._directory = directory
        self._directory_fd = directory_fd
        self._lock_fd = lock_fd
        self._events_fd = events_fd
        self._state = state
        self._total_event_bytes = total_event_bytes
        self._closed = False

    @classmethod
    def create(
        cls,
        directory: Path,
        *,
        evidence_id: str,
        segment_id: str,
        replay_start_utc: str,
        replay_end_utc: str,
        initial_balance_jpy: int | float,
        bindings: Mapping[str, str],
        account_currency: str = "JPY",
    ) -> "CompactReplayEvidenceWriter":
        """Create a new directory and durably append its first segment event."""

        evidence_path = _create_evidence_directory(Path(directory))
        directory_fd = _open_directory_fd(evidence_path)
        lock_fd = -1
        events_fd = -1
        try:
            lock_fd = _create_regular_file(directory_fd, LOCK_NAME, mode=0o600)
            events_fd = _create_regular_file(directory_fd, EVENTS_NAME, mode=0o600)
            os.close(events_fd)
            events_fd = _open_regular_file(
                directory_fd, EVENTS_NAME, os.O_WRONLY | os.O_APPEND
            )
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            state = _ReplayState()
            writer = cls(
                evidence_path,
                directory_fd,
                lock_fd,
                events_fd,
                state,
                0,
            )
            writer.append(
                "SEGMENT_START",
                event_id=f"{_identifier(segment_id, 'segment_id')}:start",
                event_at_utc=replay_start_utc,
                payload={
                    "segment_id": segment_id,
                    "replay_start_utc": replay_start_utc,
                    "replay_end_utc": replay_end_utc,
                    "initial_balance_jpy": initial_balance_jpy,
                    "account_currency": account_currency,
                    "bindings": dict(bindings),
                },
                evidence_id=evidence_id,
            )
            return writer
        except Exception:
            for descriptor in (events_fd, lock_fd, directory_fd):
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            raise

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def event_count(self) -> int:
        return self._state.event_count

    def append(
        self,
        event_type: str,
        *,
        event_id: str,
        event_at_utc: str,
        payload: Mapping[str, Any],
        evidence_id: str | None = None,
    ) -> str:
        """Validate, hash-chain, append, and fsync one exact-schema event."""

        self._require_open()
        _ensure_directory_unchanged(self._directory, self._directory_fd)
        if _name_exists(self._directory_fd, MANIFEST_NAME):
            raise CompactReplayEvidenceError("finalized evidence cannot be appended")
        current_size = os.fstat(self._events_fd).st_size
        if current_size != self._total_event_bytes:
            raise CompactReplayEvidenceError("event ledger changed outside this writer")
        if self._state.stopped:
            raise CompactReplayEvidenceError("no event may follow SEGMENT_STOP")
        normalized_type = _event_type(event_type)
        chosen_evidence_id = _identifier(
            evidence_id if self._state.evidence_id is None else self._state.evidence_id,
            "evidence_id",
        )
        if self._state.evidence_id is not None and evidence_id not in (
            None,
            self._state.evidence_id,
        ):
            raise CompactReplayEvidenceError("evidence_id changed within one ledger")
        normalized_payload = _validate_payload(normalized_type, payload)
        if normalized_type == "SEGMENT_START":
            bindings = normalized_payload["bindings"]
            bindings_sha256 = canonical_sha256(bindings)
        else:
            if self._state.bindings_sha256 is None:
                raise CompactReplayEvidenceError("SEGMENT_START must be first")
            bindings_sha256 = self._state.bindings_sha256
        body = {
            "contract": EVENT_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "evidence_id": chosen_evidence_id,
            "sequence": self._state.event_count,
            "event_id": _identifier(event_id, "event_id"),
            "event_type": normalized_type,
            "event_at_utc": _timestamp(event_at_utc, "event_at_utc"),
            "previous_event_sha256": self._state.previous_event_sha256,
            "bindings_sha256": bindings_sha256,
            "payload": normalized_payload,
            "classification": "HISTORICAL_REPLAY_DIAGNOSTIC_ONLY",
            "proof_eligible": False,
            "promotion_eligible": False,
            "live_permission": False,
            "order_authority": "NONE",
            "broker_mutation_allowed": False,
        }
        event = {**body, "event_sha256": canonical_sha256(body)}
        # Apply to a cloned state before writing, so a semantic failure never
        # consumes bytes or a sequence number.
        candidate_state = _copy_state(self._state)
        _apply_event(candidate_state, event)
        line = _canonical_bytes(event) + b"\n"
        if len(line) > MAX_EVENT_LINE_BYTES:
            raise CompactReplayEvidenceError("event exceeds compact line limit")
        if self._total_event_bytes + len(line) > MAX_LEDGER_BYTES:
            raise CompactReplayEvidenceError("compact event ledger size limit reached")
        try:
            written = os.write(self._events_fd, line)
            if written != len(line):
                raise CompactReplayEvidenceError("short append to event ledger")
            os.fsync(self._events_fd)
        except CompactReplayEvidenceError:
            raise
        except OSError as exc:
            raise CompactReplayEvidenceError(f"cannot append event ledger: {exc}") from exc
        self._state = candidate_state
        self._total_event_bytes += len(line)
        return event["event_sha256"]

    def stop(
        self,
        *,
        event_id: str,
        stopped_at_utc: str,
        status: str,
        reason_code: str,
        source_cursor_sha256: str,
        terminal_balance_jpy: int | float,
        terminal_equity_jpy: int | float,
        open_trade_count: int,
        pending_order_count: int,
    ) -> str:
        """Append the sole terminal event; completed segments must be flat."""

        return self.append(
            "SEGMENT_STOP",
            event_id=event_id,
            event_at_utc=stopped_at_utc,
            payload={
                "status": status,
                "reason_code": reason_code,
                "source_cursor_sha256": source_cursor_sha256,
                "terminal_balance_jpy": terminal_balance_jpy,
                "terminal_equity_jpy": terminal_equity_jpy,
                "open_trade_count": open_trade_count,
                "pending_order_count": pending_order_count,
            },
        )

    def finalize(self, *, finalized_at_utc: str) -> Mapping[str, Any]:
        """Reverify the chain and atomically publish its immutable manifest."""

        self._require_open()
        if not self._state.stopped:
            raise CompactReplayEvidenceError("SEGMENT_STOP is required before finalize")
        if _name_exists(self._directory_fd, MANIFEST_NAME):
            raise CompactReplayEvidenceError("final manifest already exists")
        os.fsync(self._events_fd)
        verified = _verify_from_open_directory(
            self._directory,
            self._directory_fd,
            expect_manifest=False,
            lock_already_held=True,
        )
        if verified.event_count != self._state.event_count:
            raise CompactReplayEvidenceError("writer state differs from verified ledger")
        body = {
            "contract": MANIFEST_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "evidence_id": verified.evidence_id,
            "classification": "HISTORICAL_REPLAY_DIAGNOSTIC_ONLY",
            "bindings": dict(verified.bindings),
            "bindings_sha256": canonical_sha256(verified.bindings),
            "event_count": verified.event_count,
            "event_type_counts": dict(verified.event_type_counts),
            "events_sha256": verified.events_sha256,
            "first_event_sha256": verified.first_event_sha256,
            "final_event_sha256": verified.final_event_sha256,
            "started_at_utc": verified.started_at_utc,
            "stopped_at_utc": verified.stopped_at_utc,
            "total_event_bytes": verified.total_event_bytes,
            "finalized_at_utc": _timestamp(finalized_at_utc, "finalized_at_utc"),
            "external_witness_status": "ABSENT",
            "proof_eligible": False,
            "promotion_eligible": False,
            "live_permission": False,
            "order_authority": "NONE",
            "broker_mutation_allowed": False,
        }
        if _parse_timestamp(body["finalized_at_utc"]) < _parse_timestamp(
            verified.stopped_at_utc or ""
        ):
            raise CompactReplayEvidenceError(
                "finalized_at_utc precedes SEGMENT_STOP"
            )
        manifest = {**body, "manifest_sha256": canonical_sha256(body)}
        _atomic_write_new_json(self._directory_fd, MANIFEST_NAME, manifest)
        return manifest

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in (self._events_fd, self._lock_fd, self._directory_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _require_open(self) -> None:
        if self._closed:
            raise CompactReplayEvidenceError("compact replay writer is closed")

    def __enter__(self) -> "CompactReplayEvidenceWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def verify_compact_replay_evidence(
    directory: Path,
) -> CompactReplayEvidenceSnapshot:
    """Stream-verify exact bytes, schema, chain, lifecycle, and final manifest."""

    evidence_path = _require_safe_absolute_directory(Path(directory))
    directory_fd = _open_directory_fd(evidence_path)
    try:
        return _verify_from_open_directory(evidence_path, directory_fd)
    finally:
        os.close(directory_fd)


def _verify_from_open_directory(
    directory: Path,
    directory_fd: int,
    *,
    expect_manifest: bool | None = None,
    lock_already_held: bool = False,
) -> CompactReplayEvidenceSnapshot:
    _ensure_directory_unchanged(directory, directory_fd)
    names = set(os.listdir(directory_fd))
    allowed = {EVENTS_NAME, LOCK_NAME, MANIFEST_NAME}
    unknown = sorted(names - allowed)
    if unknown:
        raise CompactReplayEvidenceError(f"unknown evidence path: {unknown[0]}")
    if EVENTS_NAME not in names or LOCK_NAME not in names:
        raise CompactReplayEvidenceError("compact evidence files are incomplete")
    has_manifest = MANIFEST_NAME in names
    if expect_manifest is True and not has_manifest:
        raise CompactReplayEvidenceError("final manifest is missing")
    if expect_manifest is False and has_manifest:
        raise CompactReplayEvidenceError("unexpected final manifest")

    lock_fd = -1
    events_fd = -1
    try:
        if not lock_already_held:
            lock_fd = _open_regular_file(directory_fd, LOCK_NAME, os.O_RDWR)
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
        events_fd = _open_regular_file(directory_fd, EVENTS_NAME, os.O_RDONLY)
        snapshot = _read_event_chain(events_fd)
        manifest: Mapping[str, Any] | None = None
        if has_manifest:
            manifest = _read_manifest(directory_fd)
            _validate_manifest(manifest, snapshot)
        if snapshot.stopped_at_utc is not None and not has_manifest and expect_manifest is None:
            # A stopped-but-unsealed directory is valid crash-recovery evidence,
            # but it remains explicitly non-finalized.
            pass
        return CompactReplayEvidenceSnapshot(
            evidence_id=snapshot.evidence_id,
            bindings=snapshot.bindings,
            event_count=snapshot.event_count,
            event_type_counts=snapshot.event_type_counts,
            total_event_bytes=snapshot.total_event_bytes,
            events_sha256=snapshot.events_sha256,
            first_event_sha256=snapshot.first_event_sha256,
            final_event_sha256=snapshot.final_event_sha256,
            started_at_utc=snapshot.started_at_utc,
            stopped_at_utc=snapshot.stopped_at_utc,
            finalized=has_manifest,
            manifest=manifest,
        )
    finally:
        if events_fd >= 0:
            os.close(events_fd)
        if lock_fd >= 0:
            os.close(lock_fd)


def _read_event_chain(events_fd: int) -> CompactReplayEvidenceSnapshot:
    state = _ReplayState()
    byte_digest = hashlib.sha256()
    total_bytes = 0
    before = os.fstat(events_fd)
    with os.fdopen(os.dup(events_fd), "rb", closefd=True) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            total_bytes += len(raw_line)
            if total_bytes > MAX_LEDGER_BYTES:
                raise CompactReplayEvidenceError("compact event ledger is too large")
            if len(raw_line) > MAX_EVENT_LINE_BYTES:
                raise CompactReplayEvidenceError(
                    f"event line {line_number} exceeds size limit"
                )
            if not raw_line.endswith(b"\n"):
                raise CompactReplayEvidenceError(
                    f"event line {line_number} is incomplete"
                )
            byte_digest.update(raw_line)
            try:
                event = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CompactReplayEvidenceError(
                    f"event line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(event, dict):
                raise CompactReplayEvidenceError(
                    f"event line {line_number} must be an object"
                )
            if _canonical_bytes(event) + b"\n" != raw_line:
                raise CompactReplayEvidenceError(
                    f"event line {line_number} is not canonical JSON"
                )
            _apply_event(state, event)
    after = os.fstat(events_fd)
    if _file_identity(before) != _file_identity(after) or total_bytes != before.st_size:
        raise CompactReplayEvidenceError("event ledger changed while verifying")
    if state.event_count == 0 or state.first_event_sha256 is None:
        raise CompactReplayEvidenceError("compact event ledger is empty")
    if state.bindings is None or state.evidence_id is None or state.first_timestamp is None:
        raise CompactReplayEvidenceError("SEGMENT_START evidence is incomplete")
    return CompactReplayEvidenceSnapshot(
        evidence_id=state.evidence_id,
        bindings=state.bindings,
        event_count=state.event_count,
        event_type_counts={name: state.event_type_counts[name] for name in _EVENT_TYPES},
        total_event_bytes=total_bytes,
        events_sha256=byte_digest.hexdigest(),
        first_event_sha256=state.first_event_sha256,
        final_event_sha256=state.previous_event_sha256,
        started_at_utc=state.first_timestamp,
        stopped_at_utc=state.stop_timestamp,
        finalized=False,
        manifest=None,
    )


def _apply_event(state: _ReplayState, event: Mapping[str, Any]) -> None:
    _exact_keys(event, _EVENT_KEYS, "event")
    if event["contract"] != EVENT_CONTRACT or event["schema_version"] != SCHEMA_VERSION:
        raise CompactReplayEvidenceError("event contract/schema mismatch")
    if event["classification"] != "HISTORICAL_REPLAY_DIAGNOSTIC_ONLY":
        raise CompactReplayEvidenceError("event classification mismatch")
    _require_no_authority(event, "event")
    evidence_id = _identifier(event["evidence_id"], "event.evidence_id")
    sequence = _integer(event["sequence"], "event.sequence", minimum=0)
    if sequence != state.event_count:
        raise CompactReplayEvidenceError("event sequence is not contiguous")
    if state.event_count >= MAX_EVENTS:
        raise CompactReplayEvidenceError("compact event count limit reached")
    event_id = _identifier(event["event_id"], "event.event_id")
    if event_id in state.event_ids:
        raise CompactReplayEvidenceError("duplicate event_id")
    event_type = _event_type(event["event_type"])
    timestamp_text = _timestamp(event["event_at_utc"], "event.event_at_utc")
    timestamp = _parse_timestamp(timestamp_text)
    previous = _sha(event["previous_event_sha256"], "previous_event_sha256", zero=True)
    if previous != state.previous_event_sha256:
        raise CompactReplayEvidenceError("event hash chain is discontinuous")
    body = {key: value for key, value in event.items() if key != "event_sha256"}
    claimed_sha = _sha(event["event_sha256"], "event.event_sha256")
    if claimed_sha != canonical_sha256(body):
        raise CompactReplayEvidenceError("event SHA-256 mismatch")
    payload = _validate_payload(event_type, event["payload"])

    if state.event_count == 0:
        if event_type != "SEGMENT_START":
            raise CompactReplayEvidenceError("SEGMENT_START must be first")
        bindings = payload["bindings"]
        bindings_sha = canonical_sha256(bindings)
        if event["bindings_sha256"] != bindings_sha:
            raise CompactReplayEvidenceError("start binding SHA-256 mismatch")
        state.evidence_id = evidence_id
        state.bindings = dict(bindings)
        state.bindings_sha256 = bindings_sha
        state.replay_start = _parse_timestamp(payload["replay_start_utc"])
        state.replay_end = _parse_timestamp(payload["replay_end_utc"])
        state.initial_balance_jpy = float(payload["initial_balance_jpy"])
        state.calculated_balance_jpy = float(payload["initial_balance_jpy"])
        if state.replay_start >= state.replay_end:
            raise CompactReplayEvidenceError("replay window must be positive")
        if timestamp != state.replay_start:
            raise CompactReplayEvidenceError("SEGMENT_START timestamp/window mismatch")
        state.first_event_sha256 = claimed_sha
        state.first_timestamp = timestamp_text
    else:
        if event_type == "SEGMENT_START":
            raise CompactReplayEvidenceError("duplicate SEGMENT_START")
        if evidence_id != state.evidence_id:
            raise CompactReplayEvidenceError("evidence_id changed within ledger")
        if event["bindings_sha256"] != state.bindings_sha256:
            raise CompactReplayEvidenceError("event binding SHA-256 mismatch")
    if state.stopped:
        raise CompactReplayEvidenceError("event found after SEGMENT_STOP")
    if state.previous_timestamp is not None and timestamp < state.previous_timestamp:
        raise CompactReplayEvidenceError("event timestamps are not monotonic")
    if state.replay_start is not None and not (
        state.replay_start <= timestamp <= state.replay_end
    ):
        raise CompactReplayEvidenceError("event timestamp is outside replay window")
    if (
        event_type == "SEGMENT_STOP"
        and payload["status"] == "COMPLETED"
        and timestamp != state.replay_end
    ):
        raise CompactReplayEvidenceError(
            "COMPLETED SEGMENT_STOP must equal replay_end_utc"
        )

    _apply_semantics(state, event_type, payload)
    if event_type == "SEGMENT_STOP":
        state.stopped = True
        state.stop_timestamp = timestamp_text
    state.event_ids.add(event_id)
    state.event_type_counts[event_type] += 1
    state.previous_event_sha256 = claimed_sha
    state.previous_timestamp = timestamp
    state.event_count += 1


def _apply_semantics(
    state: _ReplayState, event_type: str, payload: Mapping[str, Any]
) -> None:
    if event_type == "SEGMENT_START":
        return
    if event_type == "BOT":
        decision_id = payload["decision_id"]
        if decision_id in state.decision_ids:
            raise CompactReplayEvidenceError("duplicate decision_id")
        state.decision_ids.add(decision_id)
        return
    if event_type == "ORDER":
        order_id = payload["order_id"]
        status_value = payload["status"]
        current = state.orders.get(order_id)
        if status_value in {"SUBMITTED", "REJECTED"}:
            if current is not None:
                raise CompactReplayEvidenceError("duplicate order_id")
            state.orders[order_id] = (
                status_value,
                payload["pair"],
                payload["side"],
                float(payload["units"]),
                payload["order_type"],
                None
                if payload["requested_price"] is None
                else float(payload["requested_price"]),
                None
                if payload["stop_loss_price"] is None
                else float(payload["stop_loss_price"]),
                None
                if payload["take_profit_price"] is None
                else float(payload["take_profit_price"]),
            )
        else:
            if current is None or current[0] != "SUBMITTED":
                raise CompactReplayEvidenceError("cannot cancel a non-pending order")
            cancellation_terms = (
                payload["pair"],
                payload["side"],
                float(payload["units"]),
                payload["order_type"],
                None
                if payload["requested_price"] is None
                else float(payload["requested_price"]),
                None
                if payload["stop_loss_price"] is None
                else float(payload["stop_loss_price"]),
                None
                if payload["take_profit_price"] is None
                else float(payload["take_profit_price"]),
            )
            if current[1:] != cancellation_terms:
                raise CompactReplayEvidenceError(
                    "cancelled order terms differ from submitted order"
                )
            state.orders[order_id] = ("CANCELLED", *current[1:])
        return
    if event_type == "FILL":
        fill_id = payload["fill_id"]
        if fill_id in state.fill_ids:
            raise CompactReplayEvidenceError("duplicate fill_id")
        order_id = payload["order_id"]
        order = state.orders.get(order_id)
        if order is None or order[0] != "SUBMITTED":
            raise CompactReplayEvidenceError("fill does not reference a pending order")
        if order[1] != payload["pair"] or order[2] != payload["side"]:
            raise CompactReplayEvidenceError("fill pair/side differs from order")
        if not math.isclose(
            order[3], float(payload["units"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise CompactReplayEvidenceError("fill units differ from order")
        _validate_fill_against_order(order, float(payload["fill_price"]))
        trade_id = payload["trade_id"]
        if trade_id in state.open_trades or trade_id in state.closed_trade_ids:
            raise CompactReplayEvidenceError("duplicate trade_id")
        state.orders[order_id] = ("FILLED", *order[1:])
        state.fill_ids.add(fill_id)
        state.open_trades[trade_id] = (
            payload["pair"],
            payload["side"],
            payload["units"],
            payload["fill_price"],
        )
        if state.calculated_balance_jpy is None:
            raise CompactReplayEvidenceError("initial balance is unavailable")
        state.calculated_balance_jpy -= float(payload["fee_jpy"])
        return
    if event_type == "EXIT":
        exit_id = payload["exit_id"]
        if exit_id in state.exit_ids:
            raise CompactReplayEvidenceError("duplicate exit_id")
        trade_id = payload["trade_id"]
        trade = state.open_trades.get(trade_id)
        if trade is None:
            raise CompactReplayEvidenceError("exit does not reference an open trade")
        if trade[0] != payload["pair"] or not math.isclose(
            trade[2], payload["units"], rel_tol=0.0, abs_tol=1e-12
        ):
            raise CompactReplayEvidenceError("exit pair/units differs from fill")
        quote_to_jpy = float(payload["quote_to_jpy_rate"])
        if payload["pair"].endswith("_JPY") and not math.isclose(
            quote_to_jpy, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise CompactReplayEvidenceError(
                "JPY-quoted exit requires quote_to_jpy_rate=1"
            )
        direction = 1.0 if trade[1] == "LONG" else -1.0
        expected_realized = (
            (float(payload["exit_price"]) - float(trade[3]))
            * float(payload["units"])
            * direction
            * quote_to_jpy
        )
        if not math.isclose(
            expected_realized,
            float(payload["realized_pnl_jpy"]),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise CompactReplayEvidenceError(
                "exit realized_pnl_jpy differs from independent price calculation"
            )
        del state.open_trades[trade_id]
        state.closed_trade_ids.add(trade_id)
        state.exit_ids.add(exit_id)
        if state.calculated_balance_jpy is None:
            raise CompactReplayEvidenceError("initial balance is unavailable")
        state.calculated_balance_jpy += float(payload["realized_pnl_jpy"])
        state.calculated_balance_jpy += float(payload["financing_jpy"])
        return
    if event_type == "MARGIN":
        _validate_account_balance(
            state, float(payload["balance_jpy"]), field="margin balance_jpy"
        )
        if not math.isclose(
            float(payload["equity_jpy"]),
            float(payload["used_margin_jpy"]) + float(payload["free_margin_jpy"]),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise CompactReplayEvidenceError(
                "margin equity must equal used_margin plus free_margin"
            )
        if not state.open_trades and not math.isclose(
            float(payload["used_margin_jpy"]), 0.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise CompactReplayEvidenceError("flat account cannot use margin")
        return
    pending_count = sum(1 for value in state.orders.values() if value[0] == "SUBMITTED")
    if event_type == "CHECKPOINT":
        checkpoint_id = payload["checkpoint_id"]
        if checkpoint_id in state.checkpoint_ids:
            raise CompactReplayEvidenceError("duplicate checkpoint_id")
        if payload["open_trade_count"] != len(state.open_trades):
            raise CompactReplayEvidenceError("checkpoint open-trade count mismatch")
        if payload["pending_order_count"] != pending_count:
            raise CompactReplayEvidenceError("checkpoint pending-order count mismatch")
        _validate_account_balance(
            state, float(payload["balance_jpy"]), field="checkpoint balance_jpy"
        )
        state.checkpoint_ids.add(checkpoint_id)
        return
    if event_type == "SEGMENT_STOP":
        if payload["open_trade_count"] != len(state.open_trades):
            raise CompactReplayEvidenceError("stop open-trade count mismatch")
        if payload["pending_order_count"] != pending_count:
            raise CompactReplayEvidenceError("stop pending-order count mismatch")
        if payload["status"] == "COMPLETED" and (
            payload["open_trade_count"] != 0 or payload["pending_order_count"] != 0
        ):
            raise CompactReplayEvidenceError("completed segment must be flat")
        _validate_account_balance(
            state,
            float(payload["terminal_balance_jpy"]),
            field="terminal_balance_jpy",
        )
        if payload["status"] == "COMPLETED" and not math.isclose(
            float(payload["terminal_equity_jpy"]),
            float(payload["terminal_balance_jpy"]),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise CompactReplayEvidenceError(
                "completed flat segment terminal equity/balance mismatch"
            )


def _validate_fill_against_order(
    order: tuple[
        str,
        str,
        str,
        float,
        str,
        float | None,
        float | None,
        float | None,
    ],
    fill_price: float,
) -> None:
    side = order[2]
    order_type = order[4]
    requested_price = order[5]
    stop_loss_price = order[6]
    take_profit_price = order[7]
    if order_type == "LIMIT" and requested_price is not None:
        if (side == "LONG" and fill_price > requested_price) or (
            side == "SHORT" and fill_price < requested_price
        ):
            raise CompactReplayEvidenceError("LIMIT fill violates requested price")
    if order_type == "STOP" and requested_price is not None:
        if (side == "LONG" and fill_price < requested_price) or (
            side == "SHORT" and fill_price > requested_price
        ):
            raise CompactReplayEvidenceError("STOP fill violates requested price")
    if stop_loss_price is not None:
        invalid_stop = (side == "LONG" and stop_loss_price >= fill_price) or (
            side == "SHORT" and stop_loss_price <= fill_price
        )
        if invalid_stop:
            raise CompactReplayEvidenceError("fill violates order stop-loss geometry")
    if take_profit_price is not None:
        invalid_take_profit = (
            side == "LONG" and take_profit_price <= fill_price
        ) or (side == "SHORT" and take_profit_price >= fill_price)
        if invalid_take_profit:
            raise CompactReplayEvidenceError("fill violates order take-profit geometry")


def _validate_account_balance(
    state: _ReplayState, observed: float, *, field: str
) -> None:
    if state.calculated_balance_jpy is None:
        raise CompactReplayEvidenceError("initial balance is unavailable")
    if not math.isclose(
        observed,
        state.calculated_balance_jpy,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise CompactReplayEvidenceError(
            f"{field} differs from independently accumulated balance"
        )


def _validate_payload(event_type: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or isinstance(payload, (str, bytes)):
        raise CompactReplayEvidenceError(f"{event_type} payload must be an object")
    value = dict(payload)
    _exact_keys(value, _PAYLOAD_KEYS[event_type], f"{event_type} payload")
    if event_type == "SEGMENT_START":
        result = {
            "segment_id": _identifier(value["segment_id"], "segment_id"),
            "replay_start_utc": _timestamp(value["replay_start_utc"], "replay_start_utc"),
            "replay_end_utc": _timestamp(value["replay_end_utc"], "replay_end_utc"),
            "initial_balance_jpy": _number(
                value["initial_balance_jpy"], "initial_balance_jpy", positive=True
            ),
            "account_currency": _currency(value["account_currency"]),
            "bindings": _bindings(value["bindings"]),
        }
        return result
    if event_type == "BOT":
        decision = _enum(
            value["decision"],
            {"HOLD", "SUBMIT_ORDER", "CANCEL_ORDER", "CLOSE_TRADE"},
            "decision",
        )
        related = _optional_identifier(
            value["related_order_or_trade_id"], "related_order_or_trade_id"
        )
        if (decision == "HOLD") != (related is None):
            raise CompactReplayEvidenceError(
                "HOLD requires no related identity; action decisions require one"
            )
        return {
            "bot_id": _identifier(value["bot_id"], "bot_id"),
            "decision_id": _identifier(value["decision_id"], "decision_id"),
            "pair": _pair(value["pair"]),
            "decision": decision,
            "reason_code": _reason(value["reason_code"]),
            "signal_sha256": _sha(value["signal_sha256"], "signal_sha256"),
            "related_order_or_trade_id": related,
        }
    if event_type == "ORDER":
        order_type = _enum(value["order_type"], {"MARKET", "LIMIT", "STOP"}, "order_type")
        requested = _optional_number(value["requested_price"], "requested_price", positive=True)
        if order_type == "MARKET" and requested is not None:
            raise CompactReplayEvidenceError(
                "MARKET order must not carry requested_price"
            )
        if order_type != "MARKET" and requested is None:
            raise CompactReplayEvidenceError("LIMIT/STOP order requires requested_price")
        return {
            "order_id": _identifier(value["order_id"], "order_id"),
            "pair": _pair(value["pair"]),
            "side": _enum(value["side"], {"LONG", "SHORT"}, "side"),
            "order_type": order_type,
            "status": _enum(value["status"], {"SUBMITTED", "REJECTED", "CANCELLED"}, "status"),
            "units": _number(value["units"], "units", positive=True),
            "requested_price": requested,
            "stop_loss_price": _optional_number(value["stop_loss_price"], "stop_loss_price", positive=True),
            "take_profit_price": _optional_number(value["take_profit_price"], "take_profit_price", positive=True),
        }
    if event_type == "FILL":
        return {
            "fill_id": _identifier(value["fill_id"], "fill_id"),
            "order_id": _identifier(value["order_id"], "order_id"),
            "trade_id": _identifier(value["trade_id"], "trade_id"),
            "pair": _pair(value["pair"]),
            "side": _enum(value["side"], {"LONG", "SHORT"}, "side"),
            "units": _number(value["units"], "units", positive=True),
            "fill_price": _number(value["fill_price"], "fill_price", positive=True),
            "spread_pips": _number(value["spread_pips"], "spread_pips", minimum=0.0),
            "slippage_pips": _number(value["slippage_pips"], "slippage_pips", minimum=0.0),
            "fee_jpy": _number(value["fee_jpy"], "fee_jpy", minimum=0.0),
        }
    if event_type == "EXIT":
        result = {
            "exit_id": _identifier(value["exit_id"], "exit_id"),
            "trade_id": _identifier(value["trade_id"], "trade_id"),
            "pair": _pair(value["pair"]),
            "reason": _enum(
                value["reason"],
                {"TAKE_PROFIT", "STOP_LOSS", "TIME", "SIGNAL", "PERIOD_END", "MARGIN_CLOSEOUT"},
                "reason",
            ),
            "units": _number(value["units"], "units", positive=True),
            "exit_price": _number(value["exit_price"], "exit_price", positive=True),
            "quote_to_jpy_rate": _number(
                value["quote_to_jpy_rate"], "quote_to_jpy_rate", positive=True
            ),
            "realized_pnl_jpy": _number(value["realized_pnl_jpy"], "realized_pnl_jpy"),
            "financing_jpy": _number(value["financing_jpy"], "financing_jpy"),
        }
        if result["financing_jpy"] > 0:
            raise CompactReplayEvidenceError(
                "financing_jpy must be zero or an adverse replay cost"
            )
        return result
    if event_type == "MARGIN":
        return {
            "balance_jpy": _number(value["balance_jpy"], "balance_jpy"),
            "equity_jpy": _number(value["equity_jpy"], "equity_jpy"),
            "used_margin_jpy": _number(value["used_margin_jpy"], "used_margin_jpy", minimum=0.0),
            "free_margin_jpy": _number(value["free_margin_jpy"], "free_margin_jpy"),
        }
    if event_type == "CHECKPOINT":
        return {
            "checkpoint_id": _identifier(value["checkpoint_id"], "checkpoint_id"),
            "source_cursor_sha256": _sha(value["source_cursor_sha256"], "source_cursor_sha256"),
            "state_sha256": _sha(value["state_sha256"], "state_sha256"),
            "open_trade_count": _integer(value["open_trade_count"], "open_trade_count", minimum=0),
            "pending_order_count": _integer(value["pending_order_count"], "pending_order_count", minimum=0),
            "balance_jpy": _number(value["balance_jpy"], "balance_jpy"),
            "equity_jpy": _number(value["equity_jpy"], "equity_jpy"),
        }
    return {
        "status": _enum(value["status"], {"COMPLETED", "FAILED"}, "status"),
        "reason_code": _reason(value["reason_code"]),
        "source_cursor_sha256": _sha(value["source_cursor_sha256"], "source_cursor_sha256"),
        "terminal_balance_jpy": _number(value["terminal_balance_jpy"], "terminal_balance_jpy"),
        "terminal_equity_jpy": _number(value["terminal_equity_jpy"], "terminal_equity_jpy"),
        "open_trade_count": _integer(value["open_trade_count"], "open_trade_count", minimum=0),
        "pending_order_count": _integer(value["pending_order_count"], "pending_order_count", minimum=0),
    }


def _read_manifest(directory_fd: int) -> Mapping[str, Any]:
    fd = _open_regular_file(directory_fd, MANIFEST_NAME, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size <= 0 or size > MAX_EVENT_LINE_BYTES:
            raise CompactReplayEvidenceError("final manifest size is invalid")
        raw = _read_exact(fd, size)
    finally:
        os.close(fd)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompactReplayEvidenceError("final manifest is invalid JSON") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) + b"\n" != raw:
        raise CompactReplayEvidenceError("final manifest is not canonical JSON")
    return value


def _validate_manifest(
    manifest: Mapping[str, Any], snapshot: CompactReplayEvidenceSnapshot
) -> None:
    _exact_keys(manifest, _MANIFEST_KEYS, "manifest")
    if manifest["contract"] != MANIFEST_CONTRACT or manifest["schema_version"] != SCHEMA_VERSION:
        raise CompactReplayEvidenceError("manifest contract/schema mismatch")
    if manifest["classification"] != "HISTORICAL_REPLAY_DIAGNOSTIC_ONLY":
        raise CompactReplayEvidenceError("manifest classification mismatch")
    _require_no_authority(manifest, "manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if _sha(manifest["manifest_sha256"], "manifest_sha256") != canonical_sha256(body):
        raise CompactReplayEvidenceError("manifest SHA-256 mismatch")
    expected_counts = {name: snapshot.event_type_counts[name] for name in _EVENT_TYPES}
    comparisons = {
        "evidence_id": snapshot.evidence_id,
        "bindings": dict(snapshot.bindings),
        "bindings_sha256": canonical_sha256(snapshot.bindings),
        "event_count": snapshot.event_count,
        "event_type_counts": expected_counts,
        "events_sha256": snapshot.events_sha256,
        "first_event_sha256": snapshot.first_event_sha256,
        "final_event_sha256": snapshot.final_event_sha256,
        "started_at_utc": snapshot.started_at_utc,
        "stopped_at_utc": snapshot.stopped_at_utc,
        "total_event_bytes": snapshot.total_event_bytes,
    }
    for key, expected in comparisons.items():
        if manifest[key] != expected:
            raise CompactReplayEvidenceError(f"manifest {key} mismatch")
    _timestamp(manifest["finalized_at_utc"], "manifest.finalized_at_utc")
    if manifest["external_witness_status"] != "ABSENT":
        raise CompactReplayEvidenceError("manifest external witness claim is invalid")
    if snapshot.stopped_at_utc is None:
        raise CompactReplayEvidenceError("finalized ledger lacks SEGMENT_STOP")


def _atomic_write_new_json(directory_fd: int, name: str, value: Any) -> None:
    if _name_exists(directory_fd, name):
        raise CompactReplayEvidenceError(f"artifact already exists: {name}")
    raw = _canonical_bytes(value) + b"\n"
    temp_name = f".{name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    fd = -1
    try:
        fd = _create_regular_file(directory_fd, temp_name, mode=0o600)
        written = os.write(fd, raw)
        if written != len(raw):
            raise CompactReplayEvidenceError("short write while publishing manifest")
        os.fsync(fd)
        os.close(fd)
        fd = -1
        # A hard-link publication is an atomic no-replace operation: unlike
        # rename(2), it cannot overwrite a manifest created by another writer.
        os.link(
            temp_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.unlink(temp_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except CompactReplayEvidenceError:
        raise
    except OSError as exc:
        raise CompactReplayEvidenceError(f"cannot publish final manifest: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _create_evidence_directory(path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise CompactReplayEvidenceError("evidence path must be absolute without '..'")
    parent = _require_safe_absolute_directory(path.parent)
    if not path.name or path.name in {".", ".."}:
        raise CompactReplayEvidenceError("evidence directory name is invalid")
    parent_fd = _open_directory_fd(parent)
    try:
        try:
            os.mkdir(path.name, 0o700, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise CompactReplayEvidenceError("evidence directory already exists") from exc
        except OSError as exc:
            raise CompactReplayEvidenceError(f"cannot create evidence directory: {exc}") from exc
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return _require_safe_absolute_directory(path)


def _require_safe_absolute_directory(path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise CompactReplayEvidenceError("evidence path must be absolute without '..'")
    try:
        state = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CompactReplayEvidenceError(f"evidence directory is unavailable: {exc}") from exc
    if stat.S_ISLNK(state.st_mode):
        raise CompactReplayEvidenceError("evidence path may not be a symlink")
    if not stat.S_ISDIR(state.st_mode):
        raise CompactReplayEvidenceError("evidence path must be a real directory")
    if resolved != path:
        raise CompactReplayEvidenceError("evidence path may not traverse symlinks")
    return path


def _open_directory_fd(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CompactReplayEvidenceError(f"cannot open evidence directory: {exc}") from exc
    if not stat.S_ISDIR(os.fstat(fd).st_mode):
        os.close(fd)
        raise CompactReplayEvidenceError("evidence path is not a directory")
    return fd


def _ensure_directory_unchanged(path: Path, directory_fd: int) -> None:
    try:
        path_state = path.lstat()
        fd_state = os.fstat(directory_fd)
    except OSError as exc:
        raise CompactReplayEvidenceError(f"evidence directory changed: {exc}") from exc
    if stat.S_ISLNK(path_state.st_mode) or (
        path_state.st_dev,
        path_state.st_ino,
    ) != (fd_state.st_dev, fd_state.st_ino):
        raise CompactReplayEvidenceError("evidence directory changed during use")


def _create_regular_file(directory_fd: int, name: str, *, mode: int) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, mode, dir_fd=directory_fd)
    except OSError as exc:
        raise CompactReplayEvidenceError(f"cannot create evidence file {name}: {exc}") from exc


def _open_regular_file(directory_fd: int, name: str, flags: int) -> int:
    safe_flags = flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        state = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(state.st_mode):
            raise CompactReplayEvidenceError(f"evidence path is not regular: {name}")
        fd = os.open(name, safe_flags, dir_fd=directory_fd)
        opened = os.fstat(fd)
    except CompactReplayEvidenceError:
        raise
    except OSError as exc:
        raise CompactReplayEvidenceError(f"cannot open evidence file {name}: {exc}") from exc
    if (state.st_dev, state.st_ino) != (opened.st_dev, opened.st_ino):
        os.close(fd)
        raise CompactReplayEvidenceError(f"evidence file changed before open: {name}")
    return fd


def _name_exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CompactReplayEvidenceError(f"cannot inspect evidence path {name}: {exc}") from exc


def _read_exact(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(fd, min(remaining, 1024 * 1024))
        if not chunk:
            raise CompactReplayEvidenceError("evidence file ended during read")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(fd, 1):
        raise CompactReplayEvidenceError("evidence file grew during read")
    return b"".join(chunks)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _copy_state(state: _ReplayState) -> _ReplayState:
    clone = _ReplayState()
    clone.evidence_id = state.evidence_id
    clone.bindings = None if state.bindings is None else dict(state.bindings)
    clone.bindings_sha256 = state.bindings_sha256
    clone.event_count = state.event_count
    clone.event_type_counts = Counter(state.event_type_counts)
    clone.event_ids = set(state.event_ids)
    clone.previous_event_sha256 = state.previous_event_sha256
    clone.first_event_sha256 = state.first_event_sha256
    clone.first_timestamp = state.first_timestamp
    clone.previous_timestamp = state.previous_timestamp
    clone.replay_start = state.replay_start
    clone.replay_end = state.replay_end
    clone.initial_balance_jpy = state.initial_balance_jpy
    clone.calculated_balance_jpy = state.calculated_balance_jpy
    clone.stop_timestamp = state.stop_timestamp
    clone.stopped = state.stopped
    clone.orders = dict(state.orders)
    clone.open_trades = dict(state.open_trades)
    clone.closed_trade_ids = set(state.closed_trade_ids)
    clone.decision_ids = set(state.decision_ids)
    clone.fill_ids = set(state.fill_ids)
    clone.exit_ids = set(state.exit_ids)
    clone.checkpoint_ids = set(state.checkpoint_ids)
    return clone


def _bindings(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        raise CompactReplayEvidenceError("bindings must be an object")
    mapping = dict(value)
    _exact_keys(mapping, _BINDING_KEYS, "bindings")
    return {key: _sha(mapping[key], f"bindings.{key}") for key in sorted(_BINDING_KEYS)}


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
        raise CompactReplayEvidenceError(f"value is not canonical JSON: {exc}") from exc


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], field: str) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        raise CompactReplayEvidenceError(
            f"{field} schema mismatch: missing={missing}, unknown={unknown}"
        )


def _require_no_authority(value: Mapping[str, Any], field: str) -> None:
    expected = {
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    for key, required in expected.items():
        if value[key] != required or type(value[key]) is not type(required):
            raise CompactReplayEvidenceError(f"{field} authority boundary mismatch")


def _event_type(value: Any) -> str:
    if not isinstance(value, str) or value not in _EVENT_TYPE_SET:
        raise CompactReplayEvidenceError("unknown event_type")
    return value


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise CompactReplayEvidenceError(f"{field} is invalid")
    return value


def _optional_identifier(value: Any, field: str) -> str | None:
    return None if value is None else _identifier(value, field)


def _pair(value: Any) -> str:
    if not isinstance(value, str) or not _PAIR.fullmatch(value):
        raise CompactReplayEvidenceError("pair is invalid")
    return value


def _currency(value: Any) -> str:
    if not isinstance(value, str) or not _CURRENCY.fullmatch(value):
        raise CompactReplayEvidenceError("account_currency is invalid")
    return value


def _reason(value: Any) -> str:
    if not isinstance(value, str) or not _REASON_CODE.fullmatch(value):
        raise CompactReplayEvidenceError("reason_code is invalid")
    return value


def _sha(value: Any, field: str, *, zero: bool = False) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise CompactReplayEvidenceError(f"{field} is not lowercase SHA-256")
    if not zero and value == ZERO_SHA256:
        raise CompactReplayEvidenceError(f"{field} may not be the zero digest")
    return value


def _timestamp(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _UTC_TIMESTAMP.fullmatch(value):
        raise CompactReplayEvidenceError(f"{field} must be a strict UTC timestamp")
    _parse_timestamp(value)
    return value


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CompactReplayEvidenceError("timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise CompactReplayEvidenceError("timestamp must be UTC-aware")
    return parsed


def _enum(value: Any, allowed: set[str], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise CompactReplayEvidenceError(f"{field} is invalid")
    return value


def _number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    positive: bool = False,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompactReplayEvidenceError(f"{field} must be numeric")
    if not math.isfinite(float(value)):
        raise CompactReplayEvidenceError(f"{field} must be finite")
    if positive and value <= 0:
        raise CompactReplayEvidenceError(f"{field} must be positive")
    if minimum is not None and value < minimum:
        raise CompactReplayEvidenceError(f"{field} is below its minimum")
    return value


def _optional_number(
    value: Any, field: str, *, positive: bool = False
) -> int | float | None:
    return None if value is None else _number(value, field, positive=positive)


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CompactReplayEvidenceError(f"{field} must be an integer >= {minimum}")
    return value


__all__ = [
    "CompactReplayEvidenceError",
    "CompactReplayEvidenceSnapshot",
    "CompactReplayEvidenceWriter",
    "EVENT_CONTRACT",
    "MANIFEST_CONTRACT",
    "canonical_sha256",
    "verify_compact_replay_evidence",
]
