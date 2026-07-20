"""Append-only economic transcript and independent DOJO reexecution.

The long-horizon runner and the independent auditor must not share a compact
economic claim.  This module defines the narrow adapter between them: the
producer records the exact quote coordinate, reducer-owned post-exit snapshot,
all-worker proposal batch (including empty/HOLD proposals), and reducer-owned
allocation receipt.  A terminal success additionally records the complete
portfolio result and carry state already embedded in that result.

The reexecutor opens the transcript read-only, validates the canonical JSONL
and hash chain, constructs a fresh :class:`PortfolioReplaySession`, and repeats
every transition.  It accepts a success only when every snapshot, allocation
receipt, terminal result, carry digest, and source-batch chain is identical.
Failure transcripts intentionally expose no partial result through their
attestation.  This module has no broker, model, order, or live capability.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol

from quant_rabbit.dojo_portfolio_replay_reducer import (
    MONTH_END_FLAT_SETTLEMENT,
    MONTH_END_MTM_WITH_STATE_HANDOFF,
    PORTFOLIO_COORDINATE_RECEIPT_CONTRACT,
    PortfolioReplaySession,
    canonical_portfolio_sha256,
    quote_batch_sha256,
    validate_portfolio_replay_result,
    verify_portfolio_policy,
)
from quant_rabbit.dojo_shared_worker_protocol import (
    ProtocolViolation,
    verify_post_exit_snapshot,
    verify_worker_proposal_batch,
)


TRANSCRIPT_HEADER_CONTRACT: Final = "QR_DOJO_ECONOMIC_TRANSCRIPT_HEADER_V1"
TRANSCRIPT_RECORD_CONTRACT: Final = "QR_DOJO_ECONOMIC_TRANSCRIPT_RECORD_V1"
REEXECUTION_ATTESTATION_CONTRACT: Final = (
    "QR_DOJO_ECONOMIC_TRANSCRIPT_REEXECUTION_ATTESTATION_V1"
)
REEXECUTION_DENOMINATOR_CONTRACT: Final = (
    "QR_DOJO_ECONOMIC_REEXECUTION_FIXED_DENOMINATOR_V1"
)
SCHEMA_VERSION: Final = 1
GENESIS_SHA256: Final = "0" * 64
MAX_RECORD_BYTES: Final = 32 * 1024 * 1024
MAX_RECORD_COUNT: Final = 50_000_000
MAX_TRANSCRIPT_BYTES: Final = 512 * 1024 * 1024 * 1024

_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_EVENT_TYPES = frozenset(
    {
        "HEADER",
        "QUOTE_BATCH",
        "POST_EXIT_SNAPSHOT",
        "WORKER_PROPOSAL_BATCH",
        "ALLOCATION_RECEIPT",
        "TERMINAL_SUCCESS",
        "TERMINAL_FAILURE",
    }
)
_RECORD_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "transcript_id",
        "record_index",
        "previous_record_sha256",
        "event_type",
        "payload",
        "record_sha256",
    }
)
_INPUT_BINDING_KEYS = frozenset(
    {
        "job_sha256",
        "claim_sha256",
        "source_slice_receipt_sha256",
        "worker_runtime_binding_sha256",
        "cost_policy_sha256",
        "risk_policy_sha256",
        "replay_engine_sha256",
        "portfolio_policy_binding_sha256",
        "predecessor_state_sha256",
        "predecessor_portfolio_carry_state_sha256",
        "predecessor_source_batch_chain_sha256",
    }
)
_HEADER_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "transcript_id",
        "coordinate_id",
        "terminal_policy",
        "expected_quote_batch_count",
        "portfolio_policy",
        "policy_sha256",
        "initial_balance_jpy",
        "predecessor_portfolio_carry_state",
        "input_bindings",
        "recording_scope",
        "append_only",
        "canonical_jsonl",
        "live_permission",
        "broker_mutation_allowed",
        "order_authority",
        "header_sha256",
    }
)
_QUOTE_KEYS = frozenset(
    {
        "coordinate_id",
        "epoch",
        "phase",
        "intrabar",
        "quote_watermark",
        "quotes",
        "quote_batch_sha256",
        "source_batch_chain_sha256",
    }
)
_ALLOCATION_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "policy_sha256",
        "snapshot_sha256",
        "proposal_batch_sha256",
        "quote_batch_sha256",
        "epoch",
        "phase",
        "quote_watermark",
        "ending_balance_jpy",
        "ending_equity_jpy",
        "reserved_margin_jpy",
        "free_margin_jpy",
        "attempt_delta",
        "entry_fill_delta",
        "rejection_delta",
        "event_count",
        "event_chain_sha256",
        "live_permission",
        "broker_mutation_allowed",
        "coordinate_receipt_sha256",
    }
)
_SUCCESS_KEYS = frozenset(
    {
        "terminal_status",
        "terminal_policy",
        "completed_coordinate_count",
        "source_batch_chain_sha256",
        "portfolio_result",
        "portfolio_result_sha256",
        "portfolio_carry_state_sha256",
    }
)
_FAILURE_KEYS = frozenset(
    {
        "terminal_status",
        "failure_code",
        "failure_stage",
        "failure_evidence_sha256",
        "source_batch_chain_sha256",
        "completed_coordinate_count",
        "partial_economics_reported",
    }
)


class DojoEconomicTranscriptError(ValueError):
    """The transcript is incomplete, non-canonical, altered, or inconsistent."""


class EconomicTranscriptSink(Protocol):
    """Runner-facing adapter; it deliberately exposes no execution capability."""

    def record_quote_batch(
        self,
        *,
        coordinate_id: str,
        epoch: int,
        phase: str,
        intrabar: str,
        quote_watermark: int,
        quotes: Sequence[Mapping[str, Any]],
        quote_batch_sha256_value: str,
        source_batch_chain_sha256: str,
    ) -> dict[str, Any]: ...

    def record_post_exit_snapshot(
        self, snapshot: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    def record_worker_proposal_batch(
        self, proposal_batch: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    def record_allocation_receipt(
        self, receipt: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    def seal_success(
        self,
        *,
        terminal_policy: str,
        portfolio_result: Mapping[str, Any],
        source_batch_chain_sha256: str,
    ) -> dict[str, Any]: ...

    def seal_failure(
        self,
        *,
        failure_code: str,
        failure_stage: str,
        failure_evidence_sha256: str,
    ) -> dict[str, Any]: ...


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
        raise DojoEconomicTranscriptError("value is not strict canonical JSON") from exc


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoEconomicTranscriptError(f"{field} must be a string-keyed object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DojoEconomicTranscriptError(f"{field} must be a sequence")
    return value


def _exact(value: Mapping[str, Any], keys: frozenset[str], field: str) -> None:
    actual = frozenset(value)
    if actual != keys:
        raise DojoEconomicTranscriptError(
            f"{field} schema mismatch: missing={sorted(keys-actual)}, "
            f"extra={sorted(actual-keys)}"
        )


def _identifier(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(char) < 32 for char in value)
    ):
        raise DojoEconomicTranscriptError(f"{field} must be a trimmed identifier")
    return value


def _sha(value: Any, field: str, *, allow_zero: bool = False) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoEconomicTranscriptError(f"{field} must be a lowercase SHA-256")
    if not allow_zero and value == GENESIS_SHA256:
        raise DojoEconomicTranscriptError(f"{field} must not be the zero digest")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoEconomicTranscriptError(f"{field} must be an integer >= {minimum}")
    return value


def _number(
    value: Any, field: str, *, positive: bool = False, minimum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoEconomicTranscriptError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DojoEconomicTranscriptError(f"{field} must be a finite number")
    if positive and result <= 0:
        raise DojoEconomicTranscriptError(f"{field} must be > 0")
    if minimum is not None and result < minimum:
        raise DojoEconomicTranscriptError(f"{field} must be >= {minimum}")
    return result


def _strict_json_line(raw: bytes, *, line_number: int) -> Mapping[str, Any]:
    if not raw.endswith(b"\n") or len(raw) > MAX_RECORD_BYTES:
        raise DojoEconomicTranscriptError(
            f"transcript line {line_number} is truncated or oversized"
        )

    def reject_constant(token: str) -> None:
        raise DojoEconomicTranscriptError(
            f"non-finite JSON token is forbidden: {token}"
        )

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DojoEconomicTranscriptError(
                    f"duplicate JSON key on transcript line {line_number}: {key}"
                )
            result[key] = item
        return result

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except DojoEconomicTranscriptError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoEconomicTranscriptError(
            f"transcript line {line_number} is not strict JSON"
        ) from exc
    row = _mapping(parsed, f"transcript line {line_number}")
    if raw != _canonical_bytes(row) + b"\n":
        raise DojoEconomicTranscriptError(
            f"transcript line {line_number} is not canonical JSONL"
        )
    return row


def _source_batch_chain(
    previous: str,
    *,
    quote_digest: str,
    epoch: int,
    phase: str,
    watermark: int,
) -> str:
    return canonical_portfolio_sha256(
        {
            "previous_batch_chain_sha256": previous,
            "quote_batch_sha256": quote_digest,
            "epoch": epoch,
            "phase": phase,
            "quote_watermark": watermark,
        }
    )


def build_economic_transcript_header(
    *,
    transcript_id: str,
    coordinate_id: str,
    portfolio_policy: Mapping[str, Any],
    input_bindings: Mapping[str, Any],
    terminal_policy: str,
    expected_quote_batch_count: int,
    initial_balance_jpy: int | float | None = None,
    predecessor_portfolio_carry_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal the immutable replay origin and provenance needed by a fresh process."""

    if terminal_policy not in {
        MONTH_END_FLAT_SETTLEMENT,
        MONTH_END_MTM_WITH_STATE_HANDOFF,
    }:
        raise DojoEconomicTranscriptError("terminal policy is unsupported")
    policy = verify_portfolio_policy(portfolio_policy)
    bindings = dict(_mapping(input_bindings, "input_bindings"))
    _exact(bindings, _INPUT_BINDING_KEYS, "input_bindings")
    for key in (
        "job_sha256",
        "claim_sha256",
        "source_slice_receipt_sha256",
        "worker_runtime_binding_sha256",
        "cost_policy_sha256",
        "risk_policy_sha256",
        "replay_engine_sha256",
        "portfolio_policy_binding_sha256",
    ):
        bindings[key] = _sha(bindings[key], f"input_bindings.{key}")
    predecessor_state = bindings["predecessor_state_sha256"]
    predecessor_portfolio_carry_sha = bindings[
        "predecessor_portfolio_carry_state_sha256"
    ]
    predecessor_batch = _sha(
        bindings["predecessor_source_batch_chain_sha256"],
        "input_bindings.predecessor_source_batch_chain_sha256",
        allow_zero=True,
    )
    carry: Mapping[str, Any] | None = predecessor_portfolio_carry_state
    if carry is None:
        balance: float | None = _number(
            initial_balance_jpy, "initial_balance_jpy", positive=True
        )
        if (
            predecessor_state is not None
            or predecessor_portfolio_carry_sha is not None
            or predecessor_batch != GENESIS_SHA256
        ):
            raise DojoEconomicTranscriptError(
                "fresh replay must have null predecessor state and genesis batch chain"
            )
    else:
        if initial_balance_jpy is not None:
            raise DojoEconomicTranscriptError(
                "initial balance and predecessor carry are mutually exclusive"
            )
        balance = None
        bindings["predecessor_state_sha256"] = _sha(
            predecessor_state, "input_bindings.predecessor_state_sha256"
        )
        carry_row = _mapping(carry, "predecessor_portfolio_carry_state")
        carry_sha = _sha(
            carry_row.get("carry_state_sha256"),
            "predecessor_portfolio_carry_state.carry_state_sha256",
        )
        if carry_sha != _sha(
            predecessor_portfolio_carry_sha,
            "input_bindings.predecessor_portfolio_carry_state_sha256",
        ):
            raise DojoEconomicTranscriptError(
                "predecessor portfolio carry digest binding is invalid"
            )
        if predecessor_batch == GENESIS_SHA256:
            raise DojoEconomicTranscriptError(
                "carried replay requires a non-genesis predecessor batch chain"
            )
        # PortfolioReplaySession is the canonical deep carry validator.  This
        # constructor performs no economic transition and grants no authority.
        PortfolioReplaySession(policy=policy, carry_state=carry)
    bindings["predecessor_source_batch_chain_sha256"] = predecessor_batch
    body: dict[str, Any] = {
        "contract": TRANSCRIPT_HEADER_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "transcript_id": _identifier(transcript_id, "transcript_id"),
        "coordinate_id": _identifier(coordinate_id, "coordinate_id"),
        "terminal_policy": terminal_policy,
        "expected_quote_batch_count": _integer(
            expected_quote_batch_count,
            "expected_quote_batch_count",
            minimum=1,
        ),
        "portfolio_policy": policy,
        "policy_sha256": policy["policy_sha256"],
        "initial_balance_jpy": balance,
        "predecessor_portfolio_carry_state": None if carry is None else _copy(carry),
        "input_bindings": bindings,
        "recording_scope": (
            "ORDERED_QUOTES_POST_EXIT_SNAPSHOTS_ALL_WORKER_PROPOSALS_"
            "ALLOCATION_RECEIPTS_TERMINAL_CARRY_OR_FAILURE"
        ),
        "append_only": True,
        "canonical_jsonl": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["header_sha256"] = canonical_portfolio_sha256(body)
    return body


def verify_economic_transcript_header(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(_mapping(value, "header"))
    _exact(row, _HEADER_KEYS, "header")
    unsigned = {key: value for key, value in row.items() if key != "header_sha256"}
    if (
        row["contract"] != TRANSCRIPT_HEADER_CONTRACT
        or _integer(row["schema_version"], "header.schema_version") != SCHEMA_VERSION
        or canonical_portfolio_sha256(unsigned)
        != _sha(row["header_sha256"], "header.header_sha256")
    ):
        raise DojoEconomicTranscriptError("header contract or digest is invalid")
    rebuilt = build_economic_transcript_header(
        transcript_id=row["transcript_id"],
        coordinate_id=row["coordinate_id"],
        portfolio_policy=row["portfolio_policy"],
        input_bindings=row["input_bindings"],
        terminal_policy=row["terminal_policy"],
        expected_quote_batch_count=row["expected_quote_batch_count"],
        initial_balance_jpy=row["initial_balance_jpy"],
        predecessor_portfolio_carry_state=row["predecessor_portfolio_carry_state"],
    )
    if rebuilt != row:
        raise DojoEconomicTranscriptError("header content is not canonical")
    return rebuilt


def _record(
    *,
    transcript_id: str,
    record_index: int,
    previous_record_sha256: str,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if event_type not in _EVENT_TYPES:
        raise DojoEconomicTranscriptError("unsupported transcript event type")
    body = {
        "contract": TRANSCRIPT_RECORD_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "transcript_id": _identifier(transcript_id, "record.transcript_id"),
        "record_index": _integer(record_index, "record.record_index"),
        "previous_record_sha256": _sha(
            previous_record_sha256,
            "record.previous_record_sha256",
            allow_zero=True,
        ),
        "event_type": event_type,
        "payload": _copy(_mapping(payload, "record.payload")),
    }
    body["record_sha256"] = canonical_portfolio_sha256(body)
    return body


def _verify_record(
    value: Mapping[str, Any],
    *,
    transcript_id: str,
    expected_index: int,
    previous_record_sha256: str,
) -> dict[str, Any]:
    row = dict(_mapping(value, "record"))
    _exact(row, _RECORD_KEYS, "record")
    unsigned = {key: value for key, value in row.items() if key != "record_sha256"}
    if (
        row["contract"] != TRANSCRIPT_RECORD_CONTRACT
        or _integer(row["schema_version"], "record.schema_version") != SCHEMA_VERSION
        or _identifier(row["transcript_id"], "record.transcript_id") != transcript_id
        or _integer(row["record_index"], "record.record_index") != expected_index
        or row["previous_record_sha256"] != previous_record_sha256
        or row["event_type"] not in _EVENT_TYPES
        or canonical_portfolio_sha256(unsigned)
        != _sha(row["record_sha256"], "record.record_sha256")
    ):
        raise DojoEconomicTranscriptError(
            "record contract, sequence, chain, or digest is invalid"
        )
    return row


def _verify_quote_payload(
    value: Mapping[str, Any],
    *,
    header: Mapping[str, Any],
    previous_source_chain: str,
) -> dict[str, Any]:
    row = dict(_mapping(value, "quote payload"))
    _exact(row, _QUOTE_KEYS, "quote payload")
    if row["coordinate_id"] != header["coordinate_id"]:
        raise DojoEconomicTranscriptError("quote coordinate differs from header")
    epoch = _integer(row["epoch"], "quote payload.epoch")
    watermark = _integer(row["quote_watermark"], "quote payload.quote_watermark")
    phase = _identifier(row["phase"], "quote payload.phase")
    intrabar = _identifier(row["intrabar"], "quote payload.intrabar")
    quotes = _sequence(row["quotes"], "quote payload.quotes")
    digest = quote_batch_sha256(
        epoch=epoch,
        phase=phase,
        intrabar=intrabar,
        quote_watermark=watermark,
        quotes=quotes,
    )
    if digest != _sha(row["quote_batch_sha256"], "quote payload digest"):
        raise DojoEconomicTranscriptError("quote batch digest is invalid")
    expected_chain = _source_batch_chain(
        previous_source_chain,
        quote_digest=digest,
        epoch=epoch,
        phase=phase,
        watermark=watermark,
    )
    if expected_chain != _sha(
        row["source_batch_chain_sha256"], "quote source batch chain"
    ):
        raise DojoEconomicTranscriptError("source batch chain is invalid")
    return row


def _verify_allocation_receipt(
    value: Mapping[str, Any],
    *,
    policy_sha256: str,
    snapshot: Mapping[str, Any],
    proposal_batch: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(_mapping(value, "allocation receipt"))
    _exact(row, _ALLOCATION_KEYS, "allocation receipt")
    unsigned = {
        key: item for key, item in row.items() if key != "coordinate_receipt_sha256"
    }
    if (
        row["contract"] != PORTFOLIO_COORDINATE_RECEIPT_CONTRACT
        or _integer(row["schema_version"], "allocation receipt.schema_version")
        != SCHEMA_VERSION
        or row["policy_sha256"] != policy_sha256
        or row["snapshot_sha256"] != snapshot["snapshot_sha256"]
        or row["proposal_batch_sha256"] != proposal_batch["batch_sha256"]
        or row["quote_batch_sha256"] != snapshot["quote_batch_sha256"]
        or row["epoch"] != snapshot["epoch"]
        or row["phase"] != snapshot["phase"]
        or row["quote_watermark"] != snapshot["quote_watermark"]
        or row["live_permission"] is not False
        or row["broker_mutation_allowed"] is not False
        or canonical_portfolio_sha256(unsigned)
        != _sha(
            row["coordinate_receipt_sha256"],
            "allocation receipt.coordinate_receipt_sha256",
        )
    ):
        raise DojoEconomicTranscriptError("allocation receipt binding is invalid")
    for key in (
        "ending_balance_jpy",
        "ending_equity_jpy",
        "reserved_margin_jpy",
        "free_margin_jpy",
    ):
        _number(row[key], f"allocation receipt.{key}")
    for key in (
        "attempt_delta",
        "entry_fill_delta",
        "rejection_delta",
        "event_count",
    ):
        _integer(row[key], f"allocation receipt.{key}")
    _sha(row["event_chain_sha256"], "allocation receipt.event_chain_sha256")
    return row


class EconomicTranscriptRecorder:
    """Durable append-only canonical JSONL recorder for one account coordinate."""

    def __init__(self, path: Path, header: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.header = verify_economic_transcript_header(header)
        self._record_index = 0
        self._previous_record_sha256 = GENESIS_SHA256
        self._source_batch_chain_sha256 = self.header["input_bindings"][
            "predecessor_source_batch_chain_sha256"
        ]
        self._stage = "EXPECT_QUOTE_OR_TERMINAL"
        self._coordinate_count = 0
        self._current_quote: dict[str, Any] | None = None
        self._current_snapshot: dict[str, Any] | None = None
        self._current_proposal_batch: dict[str, Any] | None = None
        self._last_allocation_event_count = 0
        self._terminal = False
        self._closed = False
        parent = self.path.parent
        if not parent.exists() or not parent.is_dir() or parent.is_symlink():
            raise DojoEconomicTranscriptError(
                "transcript parent must be an existing non-symlink directory"
            )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        self._fd: int | None = None
        try:
            self._fd = os.open(self.path, flags, 0o600)
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            opened = os.fstat(self._fd)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise DojoEconomicTranscriptError(
                    "transcript target must be a single-link regular file"
                )
            self._device_inode = (opened.st_dev, opened.st_ino)
            self._expected_size = 0
            self._append("HEADER", self.header)
            directory_fd = os.open(
                parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            self.close()
            try:
                current = self.path.stat(follow_symlinks=False)
                if (current.st_dev, current.st_ino) == getattr(
                    self, "_device_inode", (-1, -1)
                ):
                    self.path.unlink()
            except OSError:
                pass
            raise

    def __enter__(self) -> EconomicTranscriptRecorder:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "_fd", None) is not None:
            assert self._fd is not None
            os.close(self._fd)
            self._fd = None
        self._closed = True

    def _append(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed or self._fd is None:
            raise DojoEconomicTranscriptError("transcript recorder is closed")
        if self._record_index >= MAX_RECORD_COUNT:
            raise DojoEconomicTranscriptError("transcript record bound exceeded")
        current_fd = os.fstat(self._fd)
        current_path = self.path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(current_fd.st_mode)
            or current_fd.st_nlink != 1
            or (current_fd.st_dev, current_fd.st_ino) != self._device_inode
            or (current_path.st_dev, current_path.st_ino) != self._device_inode
            or current_fd.st_size != self._expected_size
            or current_path.st_size != self._expected_size
        ):
            raise DojoEconomicTranscriptError(
                "transcript file identity or append position changed"
            )
        row = _record(
            transcript_id=self.header["transcript_id"],
            record_index=self._record_index,
            previous_record_sha256=self._previous_record_sha256,
            event_type=event_type,
            payload=payload,
        )
        raw = _canonical_bytes(row) + b"\n"
        if len(raw) > MAX_RECORD_BYTES:
            raise DojoEconomicTranscriptError("transcript record exceeds byte bound")
        view = memoryview(raw)
        written = 0
        while written < len(raw):
            count = os.write(self._fd, view[written:])
            if count <= 0:
                raise OSError("short transcript append")
            written += count
        os.fsync(self._fd)
        after = os.fstat(self._fd)
        if after.st_size != self._expected_size + len(raw):
            raise DojoEconomicTranscriptError("concurrent transcript append detected")
        self._expected_size = after.st_size
        self._previous_record_sha256 = row["record_sha256"]
        self._record_index += 1
        return _copy(row)

    def record_quote_batch(
        self,
        *,
        coordinate_id: str,
        epoch: int,
        phase: str,
        intrabar: str,
        quote_watermark: int,
        quotes: Sequence[Mapping[str, Any]],
        quote_batch_sha256_value: str,
        source_batch_chain_sha256: str,
    ) -> dict[str, Any]:
        if self._stage != "EXPECT_QUOTE_OR_TERMINAL" or self._terminal:
            raise DojoEconomicTranscriptError("quote event is out of order")
        payload = {
            "coordinate_id": coordinate_id,
            "epoch": epoch,
            "phase": phase,
            "intrabar": intrabar,
            "quote_watermark": quote_watermark,
            "quotes": _copy(list(quotes)),
            "quote_batch_sha256": quote_batch_sha256_value,
            "source_batch_chain_sha256": source_batch_chain_sha256,
        }
        verified = _verify_quote_payload(
            payload,
            header=self.header,
            previous_source_chain=self._source_batch_chain_sha256,
        )
        row = self._append("QUOTE_BATCH", verified)
        self._current_quote = verified
        self._source_batch_chain_sha256 = verified["source_batch_chain_sha256"]
        self._stage = "EXPECT_SNAPSHOT"
        return row

    def record_post_exit_snapshot(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        if self._stage != "EXPECT_SNAPSHOT" or self._current_quote is None:
            raise DojoEconomicTranscriptError("post-exit snapshot is out of order")
        try:
            verified = verify_post_exit_snapshot(snapshot)
        except ProtocolViolation as exc:
            raise DojoEconomicTranscriptError("post-exit snapshot is invalid") from exc
        quote = self._current_quote
        if any(
            verified[key] != quote[key]
            for key in (
                "coordinate_id",
                "epoch",
                "phase",
                "intrabar",
                "quote_watermark",
                "quote_batch_sha256",
                "quotes",
            )
        ):
            raise DojoEconomicTranscriptError(
                "post-exit snapshot differs from the recorded quote coordinate"
            )
        if (
            verified["expected_quote_pairs"]
            != self.header["portfolio_policy"]["expected_quote_pairs"]
            or verified["active_worker_bindings"]
            != self.header["portfolio_policy"]["active_worker_bindings"]
        ):
            raise DojoEconomicTranscriptError(
                "post-exit snapshot policy denominator drifted"
            )
        row = self._append("POST_EXIT_SNAPSHOT", {"snapshot": verified})
        self._current_snapshot = verified
        self._stage = "EXPECT_PROPOSALS"
        return row

    def record_worker_proposal_batch(
        self, proposal_batch: Mapping[str, Any]
    ) -> dict[str, Any]:
        if self._stage != "EXPECT_PROPOSALS" or self._current_snapshot is None:
            raise DojoEconomicTranscriptError("worker proposal batch is out of order")
        try:
            verified = verify_worker_proposal_batch(
                self._current_snapshot, proposal_batch
            )
        except ProtocolViolation as exc:
            raise DojoEconomicTranscriptError(
                "all-worker proposal batch is invalid"
            ) from exc
        row = self._append("WORKER_PROPOSAL_BATCH", {"proposal_batch": verified})
        self._current_proposal_batch = verified
        self._stage = "EXPECT_ALLOCATION"
        return row

    def record_allocation_receipt(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        if (
            self._stage != "EXPECT_ALLOCATION"
            or self._current_snapshot is None
            or self._current_proposal_batch is None
        ):
            raise DojoEconomicTranscriptError("allocation receipt is out of order")
        verified = _verify_allocation_receipt(
            receipt,
            policy_sha256=self.header["policy_sha256"],
            snapshot=self._current_snapshot,
            proposal_batch=self._current_proposal_batch,
        )
        if verified["event_count"] <= self._last_allocation_event_count:
            raise DojoEconomicTranscriptError(
                "allocation event count must advance monotonically"
            )
        row = self._append("ALLOCATION_RECEIPT", {"receipt": verified})
        self._last_allocation_event_count = verified["event_count"]
        self._coordinate_count += 1
        self._current_quote = None
        self._current_snapshot = None
        self._current_proposal_batch = None
        self._stage = "EXPECT_QUOTE_OR_TERMINAL"
        return row

    def seal_success(
        self,
        *,
        terminal_policy: str,
        portfolio_result: Mapping[str, Any],
        source_batch_chain_sha256: str,
    ) -> dict[str, Any]:
        if (
            self._terminal
            or self._stage != "EXPECT_QUOTE_OR_TERMINAL"
            or self._coordinate_count == 0
        ):
            raise DojoEconomicTranscriptError(
                "terminal success requires complete coordinate transactions"
            )
        if terminal_policy not in {
            MONTH_END_FLAT_SETTLEMENT,
            MONTH_END_MTM_WITH_STATE_HANDOFF,
        }:
            raise DojoEconomicTranscriptError("terminal policy is unsupported")
        try:
            result = validate_portfolio_replay_result(portfolio_result)
        except Exception as exc:
            raise DojoEconomicTranscriptError("portfolio result is invalid") from exc
        source_chain = _sha(source_batch_chain_sha256, "terminal source batch chain")
        if (
            terminal_policy != self.header["terminal_policy"]
            or self._coordinate_count != self.header["expected_quote_batch_count"]
            or result["terminal_policy"] != terminal_policy
            or result["policy_sha256"] != self.header["policy_sha256"]
            or result["processed_coordinate_count"] != self._coordinate_count
            or source_chain != self._source_batch_chain_sha256
        ):
            raise DojoEconomicTranscriptError("terminal result binding drifted")
        payload = {
            "terminal_status": "COMPLETE",
            "terminal_policy": terminal_policy,
            "completed_coordinate_count": self._coordinate_count,
            "source_batch_chain_sha256": source_chain,
            "portfolio_result": result,
            "portfolio_result_sha256": result["result_sha256"],
            "portfolio_carry_state_sha256": result["carry_state_sha256"],
        }
        row = self._append("TERMINAL_SUCCESS", payload)
        self._terminal = True
        self.close()
        return row

    def seal_failure(
        self,
        *,
        failure_code: str,
        failure_stage: str,
        failure_evidence_sha256: str,
    ) -> dict[str, Any]:
        if self._terminal:
            raise DojoEconomicTranscriptError("transcript is already terminal")
        payload = {
            "terminal_status": "FAILED",
            "failure_code": _identifier(failure_code, "failure_code"),
            "failure_stage": _identifier(failure_stage, "failure_stage"),
            "failure_evidence_sha256": _sha(
                failure_evidence_sha256, "failure_evidence_sha256"
            ),
            "source_batch_chain_sha256": self._source_batch_chain_sha256,
            "completed_coordinate_count": self._coordinate_count,
            "partial_economics_reported": False,
        }
        row = self._append("TERMINAL_FAILURE", payload)
        self._terminal = True
        self.close()
        return row


def _success_attestation(
    *,
    header: Mapping[str, Any],
    terminal_record: Mapping[str, Any],
    transcript_file_sha256: str,
) -> dict[str, Any]:
    payload = terminal_record["payload"]
    body = {
        "contract": REEXECUTION_ATTESTATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "VERIFIED_COMPLETE",
        "transcript_id": header["transcript_id"],
        "coordinate_id": header["coordinate_id"],
        "header_sha256": header["header_sha256"],
        "terminal_record_sha256": terminal_record["record_sha256"],
        "transcript_file_sha256": transcript_file_sha256,
        "completed_coordinate_count": payload["completed_coordinate_count"],
        "source_batch_chain_sha256": payload["source_batch_chain_sha256"],
        "portfolio_result_sha256": payload["portfolio_result_sha256"],
        "portfolio_carry_state_sha256": payload["portfolio_carry_state_sha256"],
        "transcript_integrity_passed": True,
        "independent_economic_reexecution_passed": True,
        "partial_economics_reported": False,
        "official_evidence_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["reexecution_attestation_sha256"] = canonical_portfolio_sha256(body)
    return body


def _failure_attestation(
    *,
    header: Mapping[str, Any],
    terminal_record: Mapping[str, Any],
    transcript_file_sha256: str,
) -> dict[str, Any]:
    # Deliberately omit every balance, equity, P/L, result, and carry field.
    payload = terminal_record["payload"]
    body = {
        "contract": REEXECUTION_ATTESTATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "VERIFIED_FAILED_TRANSCRIPT",
        "transcript_id": header["transcript_id"],
        "coordinate_id": header["coordinate_id"],
        "header_sha256": header["header_sha256"],
        "terminal_record_sha256": terminal_record["record_sha256"],
        "transcript_file_sha256": transcript_file_sha256,
        "failure_code": payload["failure_code"],
        "failure_stage": payload["failure_stage"],
        "failure_evidence_sha256": payload["failure_evidence_sha256"],
        "completed_coordinate_count": payload["completed_coordinate_count"],
        "transcript_integrity_passed": True,
        "independent_economic_reexecution_passed": False,
        "partial_economics_reported": False,
        "official_evidence_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["reexecution_attestation_sha256"] = canonical_portfolio_sha256(body)
    return body


def reexecute_economic_transcript(path: Path) -> dict[str, Any]:
    """Stream and independently reexecute one immutable transcript file."""

    transcript_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(transcript_path, flags)
    digest = hashlib.sha256()
    header: dict[str, Any] | None = None
    session: PortfolioReplaySession | None = None
    prepared_snapshot: dict[str, Any] | None = None
    recorded_snapshot: dict[str, Any] | None = None
    proposal_batch: dict[str, Any] | None = None
    terminal_record: dict[str, Any] | None = None
    stage = "EXPECT_HEADER"
    source_chain = GENESIS_SHA256
    coordinate_count = 0
    prior_allocation_event_count = 0
    previous_record_sha = GENESIS_SHA256
    record_index = 0
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > MAX_TRANSCRIPT_BYTES
        ):
            raise DojoEconomicTranscriptError(
                "transcript must be a bounded single-link regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while raw := handle.readline(MAX_RECORD_BYTES + 1):
                if terminal_record is not None:
                    raise DojoEconomicTranscriptError(
                        "records are forbidden after the terminal event"
                    )
                if record_index >= MAX_RECORD_COUNT:
                    raise DojoEconomicTranscriptError(
                        "transcript record bound exceeded"
                    )
                digest.update(raw)
                parsed = _strict_json_line(raw, line_number=record_index + 1)
                transcript_id = (
                    parsed.get("transcript_id")
                    if header is None
                    else header["transcript_id"]
                )
                if not isinstance(transcript_id, str):
                    raise DojoEconomicTranscriptError(
                        "first record has no transcript identity"
                    )
                record = _verify_record(
                    parsed,
                    transcript_id=transcript_id,
                    expected_index=record_index,
                    previous_record_sha256=previous_record_sha,
                )
                event_type = record["event_type"]
                payload = _mapping(record["payload"], "record.payload")
                if event_type == "HEADER":
                    if stage != "EXPECT_HEADER" or record_index != 0:
                        raise DojoEconomicTranscriptError("header is out of order")
                    header = verify_economic_transcript_header(payload)
                    if header["transcript_id"] != record["transcript_id"]:
                        raise DojoEconomicTranscriptError(
                            "header and record transcript identities differ"
                        )
                    source_chain = header["input_bindings"][
                        "predecessor_source_batch_chain_sha256"
                    ]
                    session = PortfolioReplaySession(
                        policy=header["portfolio_policy"],
                        initial_balance_jpy=header["initial_balance_jpy"],
                        carry_state=header["predecessor_portfolio_carry_state"],
                    )
                    stage = "EXPECT_QUOTE_OR_TERMINAL"
                elif event_type == "QUOTE_BATCH":
                    if (
                        stage != "EXPECT_QUOTE_OR_TERMINAL"
                        or header is None
                        or session is None
                    ):
                        raise DojoEconomicTranscriptError("quote event is out of order")
                    quote = _verify_quote_payload(
                        payload,
                        header=header,
                        previous_source_chain=source_chain,
                    )
                    prepared_snapshot = session.prepare_coordinate(
                        coordinate_id=quote["coordinate_id"],
                        epoch=quote["epoch"],
                        phase=quote["phase"],
                        intrabar=quote["intrabar"],
                        quote_watermark=quote["quote_watermark"],
                        quotes=quote["quotes"],
                        quote_batch_sha256_value=quote["quote_batch_sha256"],
                    )
                    source_chain = quote["source_batch_chain_sha256"]
                    stage = "EXPECT_SNAPSHOT"
                elif event_type == "POST_EXIT_SNAPSHOT":
                    if stage != "EXPECT_SNAPSHOT" or prepared_snapshot is None:
                        raise DojoEconomicTranscriptError(
                            "snapshot event is out of order"
                        )
                    _exact(payload, frozenset({"snapshot"}), "snapshot payload")
                    try:
                        recorded_snapshot = verify_post_exit_snapshot(
                            payload["snapshot"]
                        )
                    except ProtocolViolation as exc:
                        raise DojoEconomicTranscriptError(
                            "recorded post-exit snapshot is invalid"
                        ) from exc
                    if recorded_snapshot != prepared_snapshot:
                        raise DojoEconomicTranscriptError(
                            "independent post-exit snapshot mismatch"
                        )
                    stage = "EXPECT_PROPOSALS"
                elif event_type == "WORKER_PROPOSAL_BATCH":
                    if stage != "EXPECT_PROPOSALS" or recorded_snapshot is None:
                        raise DojoEconomicTranscriptError(
                            "proposal batch event is out of order"
                        )
                    _exact(payload, frozenset({"proposal_batch"}), "proposal payload")
                    try:
                        proposal_batch = verify_worker_proposal_batch(
                            recorded_snapshot, payload["proposal_batch"]
                        )
                    except ProtocolViolation as exc:
                        raise DojoEconomicTranscriptError(
                            "recorded all-worker proposal batch is invalid"
                        ) from exc
                    stage = "EXPECT_ALLOCATION"
                elif event_type == "ALLOCATION_RECEIPT":
                    if (
                        stage != "EXPECT_ALLOCATION"
                        or header is None
                        or session is None
                        or recorded_snapshot is None
                        or proposal_batch is None
                    ):
                        raise DojoEconomicTranscriptError(
                            "allocation receipt event is out of order"
                        )
                    _exact(payload, frozenset({"receipt"}), "allocation payload")
                    recorded_receipt = _verify_allocation_receipt(
                        payload["receipt"],
                        policy_sha256=header["policy_sha256"],
                        snapshot=recorded_snapshot,
                        proposal_batch=proposal_batch,
                    )
                    if recorded_receipt["event_count"] <= prior_allocation_event_count:
                        raise DojoEconomicTranscriptError(
                            "allocation event count did not advance"
                        )
                    independently_reduced = session.consume_proposal_batch(
                        proposal_batch
                    )
                    if independently_reduced != recorded_receipt:
                        raise DojoEconomicTranscriptError(
                            "independent allocation/admission receipt mismatch"
                        )
                    prior_allocation_event_count = recorded_receipt["event_count"]
                    coordinate_count += 1
                    prepared_snapshot = None
                    recorded_snapshot = None
                    proposal_batch = None
                    stage = "EXPECT_QUOTE_OR_TERMINAL"
                elif event_type == "TERMINAL_SUCCESS":
                    if (
                        stage != "EXPECT_QUOTE_OR_TERMINAL"
                        or header is None
                        or session is None
                        or coordinate_count == 0
                    ):
                        raise DojoEconomicTranscriptError(
                            "terminal success is out of order"
                        )
                    _exact(payload, _SUCCESS_KEYS, "terminal success payload")
                    if payload["terminal_status"] != "COMPLETE":
                        raise DojoEconomicTranscriptError(
                            "terminal success status is invalid"
                        )
                    terminal_policy = payload["terminal_policy"]
                    if terminal_policy not in {
                        MONTH_END_FLAT_SETTLEMENT,
                        MONTH_END_MTM_WITH_STATE_HANDOFF,
                    }:
                        raise DojoEconomicTranscriptError(
                            "terminal success policy is unsupported"
                        )
                    if (
                        terminal_policy != header["terminal_policy"]
                        or coordinate_count != header["expected_quote_batch_count"]
                        or payload["completed_coordinate_count"] != coordinate_count
                        or payload["source_batch_chain_sha256"] != source_chain
                    ):
                        raise DojoEconomicTranscriptError(
                            "terminal success denominator or source chain drifted"
                        )
                    try:
                        recorded_result = validate_portfolio_replay_result(
                            payload["portfolio_result"]
                        )
                    except Exception as exc:
                        raise DojoEconomicTranscriptError(
                            "terminal portfolio result is invalid"
                        ) from exc
                    independent_result = validate_portfolio_replay_result(
                        session.finalize(terminal_policy=terminal_policy)
                    )
                    if (
                        independent_result != recorded_result
                        or payload["portfolio_result_sha256"]
                        != independent_result["result_sha256"]
                        or payload["portfolio_carry_state_sha256"]
                        != independent_result["carry_state_sha256"]
                    ):
                        raise DojoEconomicTranscriptError(
                            "independent terminal result or carry mismatch"
                        )
                    terminal_record = record
                    stage = "TERMINAL"
                elif event_type == "TERMINAL_FAILURE":
                    if stage == "EXPECT_HEADER" or header is None:
                        raise DojoEconomicTranscriptError(
                            "terminal failure cannot precede the header"
                        )
                    _exact(payload, _FAILURE_KEYS, "terminal failure payload")
                    if (
                        payload["terminal_status"] != "FAILED"
                        or payload["completed_coordinate_count"] != coordinate_count
                        or payload["source_batch_chain_sha256"] != source_chain
                        or payload["partial_economics_reported"] is not False
                    ):
                        raise DojoEconomicTranscriptError(
                            "terminal failure leaks or misbinds partial economics"
                        )
                    _identifier(payload["failure_code"], "failure code")
                    _identifier(payload["failure_stage"], "failure stage")
                    _sha(payload["failure_evidence_sha256"], "failure evidence")
                    terminal_record = record
                    stage = "TERMINAL"
                else:  # pragma: no cover - enum check above is exhaustive
                    raise DojoEconomicTranscriptError("unsupported transcript event")
                previous_record_sha = record["record_sha256"]
                record_index += 1
            after = os.fstat(handle.fileno())
        current = transcript_path.stat(follow_symlinks=False)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise DojoEconomicTranscriptError(
                "transcript file changed during independent reexecution"
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if header is None or terminal_record is None or stage != "TERMINAL":
        raise DojoEconomicTranscriptError(
            "transcript is missing one immutable terminal event"
        )
    file_sha = digest.hexdigest()
    if terminal_record["event_type"] == "TERMINAL_SUCCESS":
        return _success_attestation(
            header=header,
            terminal_record=terminal_record,
            transcript_file_sha256=file_sha,
        )
    return _failure_attestation(
        header=header,
        terminal_record=terminal_record,
        transcript_file_sha256=file_sha,
    )


def build_fixed_denominator_reexecution_attestation(
    *,
    expected_coordinate_ids: Sequence[str],
    attestations_by_coordinate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Suppress every success result hash when one fixed-denominator member failed."""

    expected = [
        _identifier(item, "expected coordinate id") for item in expected_coordinate_ids
    ]
    if (
        not expected
        or len(set(expected)) != len(expected)
        or expected != sorted(expected)
    ):
        raise DojoEconomicTranscriptError(
            "expected coordinate ids must be non-empty, unique, and sorted"
        )
    supplied = dict(_mapping(attestations_by_coordinate, "attestations"))
    if set(supplied) != set(expected):
        raise DojoEconomicTranscriptError(
            "reexecution attestations do not equal the fixed denominator"
        )
    verified: dict[str, dict[str, Any]] = {}
    for coordinate_id in expected:
        row = dict(_mapping(supplied[coordinate_id], f"attestation {coordinate_id}"))
        claimed = row.pop("reexecution_attestation_sha256", None)
        if (
            row.get("contract") != REEXECUTION_ATTESTATION_CONTRACT
            or _integer(
                row.get("schema_version"),
                f"attestation {coordinate_id}.schema_version",
            )
            != SCHEMA_VERSION
            or row.get("coordinate_id") != coordinate_id
            or canonical_portfolio_sha256(row)
            != _sha(claimed, f"attestation {coordinate_id} digest")
        ):
            raise DojoEconomicTranscriptError(f"attestation {coordinate_id} is invalid")
        verified[coordinate_id] = {**row, "reexecution_attestation_sha256": claimed}
    failed = [
        coordinate_id
        for coordinate_id in expected
        if verified[coordinate_id]["status"] != "VERIFIED_COMPLETE"
    ]
    body: dict[str, Any] = {
        "contract": REEXECUTION_DENOMINATOR_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "expected_coordinate_ids": expected,
        "coordinate_count": len(expected),
        "verified_complete_count": len(expected) - len(failed),
        "verified_failed_count": len(failed),
        "status": "VERIFIED_COMPLETE" if not failed else "INCOMPLETE_FAILED",
        "failed_coordinate_ids": failed,
        "attestation_sha256_by_coordinate": {
            coordinate_id: verified[coordinate_id]["reexecution_attestation_sha256"]
            for coordinate_id in expected
        },
        "portfolio_result_sha256_by_coordinate": (
            {
                coordinate_id: verified[coordinate_id]["portfolio_result_sha256"]
                for coordinate_id in expected
            }
            if not failed
            else {}
        ),
        "downstream_terminal_reduction_allowed": not failed,
        "partial_economics_reported": False,
        "official_evidence_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["fixed_denominator_attestation_sha256"] = canonical_portfolio_sha256(body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    """Separate-process verifier entrypoint; prints only the bounded attestation."""

    parser = argparse.ArgumentParser(
        description="Independently reexecute one canonical DOJO economic transcript."
    )
    parser.add_argument("transcript", type=Path)
    args = parser.parse_args(argv)
    attestation = reexecute_economic_transcript(args.transcript)
    print(_canonical_bytes(attestation).decode("utf-8"))
    return 0


__all__ = [
    "DojoEconomicTranscriptError",
    "EconomicTranscriptRecorder",
    "EconomicTranscriptSink",
    "REEXECUTION_ATTESTATION_CONTRACT",
    "REEXECUTION_DENOMINATOR_CONTRACT",
    "TRANSCRIPT_HEADER_CONTRACT",
    "TRANSCRIPT_RECORD_CONTRACT",
    "build_economic_transcript_header",
    "build_fixed_denominator_reexecution_attestation",
    "main",
    "reexecute_economic_transcript",
    "verify_economic_transcript_header",
]


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())
