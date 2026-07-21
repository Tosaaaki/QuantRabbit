"""Compact batch-major DOJO economic transcript segments.

V1 writes four durable records for every replay coordinate.  This module began
as the V2 compact foundation and now publishes V3 artifacts: V3 adds an exact
checkpoint-origin denominator and requires a fresh-balance chain genesis.
It keeps the same independent reducer truth while publishing one immutable,
canonical document for a bounded account-local coordinate batch.  HOLD is
never stored as a second representation: only proposals containing an intent
appear in ``non_hold_proposals`` and every omitted sealed active worker is
expanded to the one canonical empty proposal during verification.  V3 does not
yet share quote/snapshot source material across separate account coordinates;
callers must expose that remaining replication rather than describe this local
grouping as cross-account source deduplication.

Segments are research-only.  They expose no broker, model, allocation, or live
authority.  The long-horizon runner has an explicit fresh-month compact opt-in;
continuous carry remains fail-closed until a predecessor-chain binding exists.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_portfolio_replay_reducer import (
    PortfolioReplaySession,
    canonical_portfolio_sha256,
    quote_batch_sha256,
    verify_portfolio_policy,
    verify_portfolio_replay_checkpoint,
)
from quant_rabbit.dojo_shared_worker_protocol import (
    ProtocolViolation,
    seal_worker_proposal,
    seal_worker_proposal_batch,
    verify_post_exit_snapshot,
    verify_worker_proposal,
)


ECONOMIC_SEGMENT_CONTRACT: Final = "QR_DOJO_ECONOMIC_TRANSCRIPT_SEGMENT_V3"
ECONOMIC_SEGMENT_CHAIN_ATTESTATION_CONTRACT: Final = (
    "QR_DOJO_ECONOMIC_TRANSCRIPT_SEGMENT_CHAIN_ATTESTATION_V3"
)
SCHEMA_VERSION: Final = 3
GENESIS_SHA256: Final = "0" * 64
IMPLICIT_NO_INTENT_SEMANTICS: Final = (
    "OMITTED_ACTIVE_WORKER_IS_CANONICAL_EMPTY_PROPOSAL"
)
# One file-data fsync is the V3 durability boundary for an already complete
# in-memory segment.  A future atomic-directory publication protocol should
# replace this constant together with a schema revision, never silently.
PUBLICATION_FSYNC_COUNT: Final = 1
# Research segments may contain strategy evidence, so owner read/write only is
# the fixed local artifact permission.  A managed evidence store should replace
# this mode if publication moves outside a single-user local worktree.
PRIVATE_FILE_MODE: Final = 0o600
# A segment is an I/O checkpoint, not a market parameter.  4,096 coordinates
# bounds verification memory while amortizing publication overhead; measured
# replay memory/throughput should replace it if this transport envelope changes.
MAX_SEGMENT_COORDINATES: Final = 4_096
# This is a defensive artifact transport limit, not trading geometry.  It keeps
# a malformed segment from forcing unbounded allocation; measured production
# segment size should replace it if the batch envelope is deliberately widened.
MAX_SEGMENT_BYTES: Final = 512 * 1024 * 1024

_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_SEGMENT_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "transcript_id",
        "job_sha256",
        "segment_index",
        "prior_segment_sha256",
        "portfolio_policy",
        "policy_sha256",
        "source_range",
        "prior_source_batch_chain_sha256",
        "terminal_source_batch_chain_sha256",
        "coordinate_denominator",
        "start_checkpoint",
        "start_checkpoint_sha256",
        "terminal_checkpoint",
        "terminal_checkpoint_sha256",
        "batches",
        "terminal_segment",
        "append_only",
        "canonical_json",
        "publication_fsyncs_per_segment",
        "external_monotonic_anchor_configured",
        "fork_absence_proven",
        "official_evidence_eligible",
        "live_permission",
        "broker_mutation_allowed",
        "order_authority",
        "segment_sha256",
    }
)
_SOURCE_RANGE_KEYS = frozenset(
    {
        "source_slice_receipt_sha256",
        "offset_start",
        "offset_end_exclusive",
    }
)
_DENOMINATOR_KEYS = frozenset(
    {
        "expected_job_coordinate_count",
        "segment_coordinate_start",
        "segment_coordinate_end_exclusive",
        "segment_coordinate_count",
        "coordinate_keys_sha256",
        "stored_non_hold_proposal_count",
        "expanded_active_worker_proposal_count",
        "implicit_no_intent_semantics",
        "denominator_sha256",
    }
)
_BATCH_KEYS = frozenset(
    {
        "coordinate_ordinal",
        "source_offset_start",
        "source_offset_end_exclusive",
        "quote",
        "post_exit_snapshot",
        "non_hold_proposals",
        "allocation_receipt",
        "batch_sha256",
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


class DojoEconomicSegmentError(ValueError):
    """A V2 segment or exact resume boundary is incomplete or inconsistent."""


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


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
        raise DojoEconomicSegmentError("value is not strict canonical JSON") from exc


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoEconomicSegmentError(f"{field} must be a string-keyed object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DojoEconomicSegmentError(f"{field} must be a sequence")
    return value


def _exact(value: Mapping[str, Any], keys: frozenset[str], field: str) -> None:
    actual = frozenset(value)
    if actual != keys:
        raise DojoEconomicSegmentError(
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
        raise DojoEconomicSegmentError(f"{field} must be a trimmed identifier")
    return value


def _sha(value: Any, field: str, *, allow_zero: bool = False) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoEconomicSegmentError(f"{field} must be a lowercase SHA-256")
    if not allow_zero and value == GENESIS_SHA256:
        raise DojoEconomicSegmentError(f"{field} must not be the zero digest")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoEconomicSegmentError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_json_document(raw: bytes) -> Mapping[str, Any]:
    if not raw.endswith(b"\n") or len(raw) > MAX_SEGMENT_BYTES:
        raise DojoEconomicSegmentError("segment is truncated or exceeds its byte bound")

    def reject_constant(token: str) -> None:
        raise DojoEconomicSegmentError(f"non-finite JSON token is forbidden: {token}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DojoEconomicSegmentError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except DojoEconomicSegmentError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoEconomicSegmentError("segment is not strict JSON") from exc
    row = _mapping(parsed, "segment")
    if raw != _canonical_bytes(row) + b"\n":
        raise DojoEconomicSegmentError("segment is not canonical JSON")
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


def _verify_quote(
    value: Mapping[str, Any], *, previous_source_chain: str
) -> dict[str, Any]:
    row = dict(_mapping(value, "batch.quote"))
    _exact(row, _QUOTE_KEYS, "batch.quote")
    _identifier(row["coordinate_id"], "batch.quote.coordinate_id")
    epoch = _integer(row["epoch"], "batch.quote.epoch")
    phase = row["phase"]
    if phase not in {"O", "H", "L", "C"}:
        raise DojoEconomicSegmentError("batch.quote.phase is unsupported")
    intrabar = row["intrabar"]
    if intrabar not in {"OHLC", "OLHC"}:
        raise DojoEconomicSegmentError("batch.quote.intrabar is unsupported")
    watermark = _integer(row["quote_watermark"], "batch.quote.quote_watermark")
    quotes = _sequence(row["quotes"], "batch.quote.quotes")
    digest = quote_batch_sha256(
        epoch=epoch,
        phase=phase,
        intrabar=intrabar,
        quote_watermark=watermark,
        quotes=quotes,
    )
    if digest != _sha(row["quote_batch_sha256"], "batch.quote.quote_batch_sha256"):
        raise DojoEconomicSegmentError("batch quote digest is invalid")
    expected_chain = _source_batch_chain(
        previous_source_chain,
        quote_digest=digest,
        epoch=epoch,
        phase=phase,
        watermark=watermark,
    )
    if expected_chain != _sha(
        row["source_batch_chain_sha256"], "batch.quote.source_batch_chain_sha256"
    ):
        raise DojoEconomicSegmentError("batch source chain is invalid")
    return row


def _expand_proposal_batch(
    *,
    snapshot: Mapping[str, Any],
    non_hold_proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bindings = {row["worker_id"]: row for row in snapshot["active_worker_bindings"]}
    non_hold: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(non_hold_proposals):
        try:
            proposal = verify_worker_proposal(snapshot, raw)
        except ProtocolViolation as exc:
            raise DojoEconomicSegmentError(
                f"non_hold_proposals[{index}] is invalid"
            ) from exc
        worker_id = proposal["worker_id"]
        if worker_id in non_hold:
            raise DojoEconomicSegmentError("duplicate non-HOLD worker proposal")
        if (
            proposal["intent_counts"]["risk_reducing"] == 0
            and proposal["intent_counts"]["new_risk"] == 0
        ):
            raise DojoEconomicSegmentError(
                "explicit HOLD/NO_INTENT proposal is forbidden in V2"
            )
        non_hold[worker_id] = proposal
    if not set(non_hold).issubset(bindings):
        raise DojoEconomicSegmentError("non-HOLD proposal is outside worker denominator")
    proposals: list[dict[str, Any]] = []
    for worker_id in sorted(bindings):
        proposal = non_hold.get(worker_id)
        if proposal is None:
            binding = bindings[worker_id]
            proposal = seal_worker_proposal(
                snapshot,
                {
                    **binding,
                    "snapshot_sha256": snapshot["snapshot_sha256"],
                    "risk_reducing_intents": [],
                    "new_risk_intents": [],
                },
            )
        proposals.append(proposal)
    try:
        return seal_worker_proposal_batch(snapshot, proposals)
    except ProtocolViolation as exc:  # pragma: no cover - guarded above
        raise DojoEconomicSegmentError("expanded proposal batch is invalid") from exc


def _verify_batch(
    value: Mapping[str, Any],
    *,
    expected_ordinal: int,
    expected_source_offset: int,
    previous_source_chain: str,
    session: PortfolioReplaySession,
) -> tuple[dict[str, Any], int, str, int, int]:
    row = dict(_mapping(value, "batch"))
    _exact(row, _BATCH_KEYS, "batch")
    unsigned = {key: item for key, item in row.items() if key != "batch_sha256"}
    if canonical_portfolio_sha256(unsigned) != _sha(
        row["batch_sha256"], "batch.batch_sha256"
    ):
        raise DojoEconomicSegmentError("batch digest is invalid")
    if _integer(row["coordinate_ordinal"], "batch.coordinate_ordinal") != expected_ordinal:
        raise DojoEconomicSegmentError("batch coordinate ordinal is not contiguous")
    start = _integer(row["source_offset_start"], "batch.source_offset_start")
    end = _integer(
        row["source_offset_end_exclusive"], "batch.source_offset_end_exclusive"
    )
    if start != expected_source_offset or end <= start:
        raise DojoEconomicSegmentError("batch source range is not exact and contiguous")
    quote = _verify_quote(row["quote"], previous_source_chain=previous_source_chain)
    computed_snapshot = session.prepare_coordinate(
        coordinate_id=quote["coordinate_id"],
        epoch=quote["epoch"],
        phase=quote["phase"],
        intrabar=quote["intrabar"],
        quote_watermark=quote["quote_watermark"],
        quotes=quote["quotes"],
        quote_batch_sha256_value=quote["quote_batch_sha256"],
    )
    try:
        recorded_snapshot = verify_post_exit_snapshot(row["post_exit_snapshot"])
    except ProtocolViolation as exc:
        raise DojoEconomicSegmentError("batch post-exit snapshot is invalid") from exc
    if computed_snapshot != recorded_snapshot:
        raise DojoEconomicSegmentError("independent post-exit snapshot mismatch")
    raw_non_hold = _sequence(row["non_hold_proposals"], "batch.non_hold_proposals")
    expanded = _expand_proposal_batch(
        snapshot=recorded_snapshot,
        non_hold_proposals=raw_non_hold,
    )
    independently_reduced = session.consume_proposal_batch(expanded)
    if independently_reduced != row["allocation_receipt"]:
        raise DojoEconomicSegmentError("independent allocation receipt mismatch")
    return (
        row,
        end,
        quote["source_batch_chain_sha256"],
        len(raw_non_hold),
        len(recorded_snapshot["active_worker_bindings"]),
    )


def build_economic_segment(
    *,
    transcript_id: str,
    job_sha256: str,
    segment_index: int,
    prior_segment_sha256: str,
    portfolio_policy: Mapping[str, Any],
    source_slice_receipt_sha256: str,
    source_offset_start: int,
    source_offset_end_exclusive: int,
    prior_source_batch_chain_sha256: str,
    expected_job_coordinate_count: int,
    segment_coordinate_start: int,
    start_checkpoint: Mapping[str, Any],
    terminal_checkpoint: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    terminal_segment: bool,
) -> dict[str, Any]:
    """Build and independently verify one immutable V2 segment value."""

    policy = verify_portfolio_policy(portfolio_policy)
    if not isinstance(terminal_segment, bool):
        raise DojoEconomicSegmentError("terminal_segment must be a boolean")
    batch_rows = [_copy(row) for row in _sequence(batches, "batches")]
    if not batch_rows or len(batch_rows) > MAX_SEGMENT_COORDINATES:
        raise DojoEconomicSegmentError("segment coordinate count is outside its bound")
    start = verify_portfolio_replay_checkpoint(
        policy=policy,
        checkpoint=start_checkpoint,
    )
    terminal = verify_portfolio_replay_checkpoint(
        policy=policy,
        checkpoint=terminal_checkpoint,
    )
    coordinate_start = _integer(
        segment_coordinate_start, "segment_coordinate_start"
    )
    expected_count = _integer(
        expected_job_coordinate_count,
        "expected_job_coordinate_count",
        minimum=1,
    )
    coordinate_end = coordinate_start + len(batch_rows)
    if (
        start["processed_coordinate_count"] != coordinate_start
        or terminal["processed_coordinate_count"] != coordinate_end
        or coordinate_end > expected_count
        or (terminal_segment and coordinate_end != expected_count)
        or (not terminal_segment and coordinate_end >= expected_count)
    ):
        raise DojoEconomicSegmentError(
            "segment checkpoint and coordinate denominator binding is invalid"
        )
    source_start = _integer(source_offset_start, "source_offset_start")
    source_end = _integer(source_offset_end_exclusive, "source_offset_end_exclusive")
    if source_end <= source_start:
        raise DojoEconomicSegmentError("segment source range must be non-empty")
    source_range = {
        "source_slice_receipt_sha256": _sha(
            source_slice_receipt_sha256, "source_slice_receipt_sha256"
        ),
        "offset_start": source_start,
        "offset_end_exclusive": source_end,
    }
    previous_source_chain = _sha(
        prior_source_batch_chain_sha256,
        "prior_source_batch_chain_sha256",
        allow_zero=True,
    )
    session = PortfolioReplaySession.restore_checkpoint(
        policy=policy,
        checkpoint=start,
    )
    verified_batches: list[dict[str, Any]] = []
    next_offset = source_start
    non_hold_count = 0
    expanded_count = 0
    coordinate_keys: list[dict[str, Any]] = []
    for offset, raw in enumerate(batch_rows):
        verified, next_offset, previous_source_chain, non_hold, expanded = _verify_batch(
            raw,
            expected_ordinal=coordinate_start + offset,
            expected_source_offset=next_offset,
            previous_source_chain=previous_source_chain,
            session=session,
        )
        verified_batches.append(verified)
        non_hold_count += non_hold
        expanded_count += expanded
        quote = verified["quote"]
        coordinate_keys.append(
            {
                "coordinate_ordinal": verified["coordinate_ordinal"],
                "coordinate_id": quote["coordinate_id"],
                "epoch": quote["epoch"],
                "phase": quote["phase"],
                "intrabar": quote["intrabar"],
                "quote_watermark": quote["quote_watermark"],
                "source_offset_start": verified["source_offset_start"],
                "source_offset_end_exclusive": verified[
                    "source_offset_end_exclusive"
                ],
                "quote_batch_sha256": quote["quote_batch_sha256"],
            }
        )
    if next_offset != source_end:
        raise DojoEconomicSegmentError("batch ranges do not exhaust source range")
    restored_terminal = session.export_checkpoint()
    if restored_terminal != terminal:
        raise DojoEconomicSegmentError("terminal checkpoint is not deterministic")
    denominator: dict[str, Any] = {
        "expected_job_coordinate_count": expected_count,
        "segment_coordinate_start": coordinate_start,
        "segment_coordinate_end_exclusive": coordinate_end,
        "segment_coordinate_count": len(verified_batches),
        "coordinate_keys_sha256": canonical_portfolio_sha256(coordinate_keys),
        "stored_non_hold_proposal_count": non_hold_count,
        "expanded_active_worker_proposal_count": expanded_count,
        "implicit_no_intent_semantics": IMPLICIT_NO_INTENT_SEMANTICS,
    }
    denominator["denominator_sha256"] = canonical_portfolio_sha256(denominator)
    body: dict[str, Any] = {
        "contract": ECONOMIC_SEGMENT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "transcript_id": _identifier(transcript_id, "transcript_id"),
        "job_sha256": _sha(job_sha256, "job_sha256"),
        "segment_index": _integer(segment_index, "segment_index"),
        "prior_segment_sha256": _sha(
            prior_segment_sha256,
            "prior_segment_sha256",
            allow_zero=True,
        ),
        "portfolio_policy": policy,
        "policy_sha256": policy["policy_sha256"],
        "source_range": source_range,
        "prior_source_batch_chain_sha256": _sha(
            prior_source_batch_chain_sha256,
            "prior_source_batch_chain_sha256",
            allow_zero=True,
        ),
        "terminal_source_batch_chain_sha256": previous_source_chain,
        "coordinate_denominator": denominator,
        "start_checkpoint": start,
        "start_checkpoint_sha256": start["checkpoint_sha256"],
        "terminal_checkpoint": terminal,
        "terminal_checkpoint_sha256": terminal["checkpoint_sha256"],
        "batches": verified_batches,
        "terminal_segment": terminal_segment,
        "append_only": True,
        "canonical_json": True,
        "publication_fsyncs_per_segment": PUBLICATION_FSYNC_COUNT,
        "external_monotonic_anchor_configured": False,
        "fork_absence_proven": False,
        "official_evidence_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["segment_sha256"] = canonical_portfolio_sha256(body)
    return body


def verify_economic_segment_value(value: Mapping[str, Any]) -> dict[str, Any]:
    """Independently rebuild one in-memory V2 segment and its reducer state."""

    row = dict(_mapping(value, "segment"))
    _exact(row, _SEGMENT_KEYS, "segment")
    unsigned = {key: item for key, item in row.items() if key != "segment_sha256"}
    if (
        row["contract"] != ECONOMIC_SEGMENT_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
        or canonical_portfolio_sha256(unsigned)
        != _sha(row["segment_sha256"], "segment.segment_sha256")
        or row["append_only"] is not True
        or row["canonical_json"] is not True
        or row["publication_fsyncs_per_segment"] != PUBLICATION_FSYNC_COUNT
        or row["external_monotonic_anchor_configured"] is not False
        or row["fork_absence_proven"] is not False
        or row["official_evidence_eligible"] is not False
        or row["live_permission"] is not False
        or row["broker_mutation_allowed"] is not False
        or row["order_authority"] != "NONE"
    ):
        raise DojoEconomicSegmentError(
            "segment contract, digest, durability, or authority boundary is invalid"
        )
    source_range = dict(_mapping(row["source_range"], "segment.source_range"))
    _exact(source_range, _SOURCE_RANGE_KEYS, "segment.source_range")
    denominator = dict(
        _mapping(row["coordinate_denominator"], "segment.coordinate_denominator")
    )
    _exact(denominator, _DENOMINATOR_KEYS, "segment.coordinate_denominator")
    claimed_denominator_sha = denominator.pop("denominator_sha256")
    if canonical_portfolio_sha256(denominator) != _sha(
        claimed_denominator_sha, "coordinate_denominator.denominator_sha256"
    ):
        raise DojoEconomicSegmentError("coordinate denominator digest is invalid")
    denominator["denominator_sha256"] = claimed_denominator_sha
    rebuilt = build_economic_segment(
        transcript_id=row["transcript_id"],
        job_sha256=row["job_sha256"],
        segment_index=row["segment_index"],
        prior_segment_sha256=row["prior_segment_sha256"],
        portfolio_policy=row["portfolio_policy"],
        source_slice_receipt_sha256=source_range["source_slice_receipt_sha256"],
        source_offset_start=source_range["offset_start"],
        source_offset_end_exclusive=source_range["offset_end_exclusive"],
        prior_source_batch_chain_sha256=row["prior_source_batch_chain_sha256"],
        expected_job_coordinate_count=denominator["expected_job_coordinate_count"],
        segment_coordinate_start=denominator["segment_coordinate_start"],
        start_checkpoint=row["start_checkpoint"],
        terminal_checkpoint=row["terminal_checkpoint"],
        batches=row["batches"],
        terminal_segment=row["terminal_segment"],
    )
    if rebuilt != row:
        raise DojoEconomicSegmentError("segment is not the canonical V2 value")
    return rebuilt


class EconomicSegmentWriter:
    """Publish one immutable V2 segment with one data fsync.

    The whole segment is verified and encoded before the exclusive file is
    created.  Readers take a non-blocking shared lock, so they never accept a
    concurrently written prefix.  A crash before the sole file fsync leaves a
    truncated artifact which verification rejects; it is never resumed.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def publish(self, **segment_inputs: Any) -> dict[str, Any]:
        segment = build_economic_segment(**segment_inputs)
        raw = _canonical_bytes(segment) + b"\n"
        if len(raw) > MAX_SEGMENT_BYTES:
            raise DojoEconomicSegmentError("segment exceeds its byte bound")
        parent = self.path.parent
        if not parent.exists() or not parent.is_dir() or parent.is_symlink():
            raise DojoEconomicSegmentError(
                "segment parent must be an existing non-symlink directory"
            )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor: int | None = None
        device_inode: tuple[int, int] | None = None
        try:
            descriptor = os.open(self.path, flags, PRIVATE_FILE_MODE)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise DojoEconomicSegmentError(
                    "segment target must be a single-link regular file"
                )
            device_inode = (opened.st_dev, opened.st_ino)
            view = memoryview(raw)
            written = 0
            while written < len(raw):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("short segment publication")
                written += count
            os.fsync(descriptor)
            after = os.fstat(descriptor)
            current = self.path.stat(follow_symlinks=False)
            if (
                after.st_size != len(raw)
                or (after.st_dev, after.st_ino) != device_inode
                or (current.st_dev, current.st_ino) != device_inode
                or current.st_size != len(raw)
            ):
                raise DojoEconomicSegmentError(
                    "segment publication identity or size changed"
                )
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
                descriptor = None
            if device_inode is not None:
                try:
                    current = self.path.stat(follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == device_inode:
                        self.path.unlink()
                except OSError:
                    pass
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return _copy(segment)


def verify_economic_segment(path: Path) -> dict[str, Any]:
    """Read one immutable segment and independently replay its batches."""

    segment_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(segment_path, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > MAX_SEGMENT_BYTES
        ):
            raise DojoEconomicSegmentError(
                "segment must be a bounded single-link regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            raw = handle.read(MAX_SEGMENT_BYTES + 1)
            after = os.fstat(handle.fileno())
        current = segment_path.stat(follow_symlinks=False)
        identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ) != identity:
            raise DojoEconomicSegmentError("segment changed during verification")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return verify_economic_segment_value(_strict_json_document(raw))


def verify_economic_segment_chain(
    paths: Sequence[Path], *, require_terminal: bool = False
) -> dict[str, Any]:
    """Verify a linear chain while retaining at most one decoded segment.

    The first segment must start at the reducer's canonical fresh-balance
    checkpoint.  V3 does not define an externally anchored predecessor
    contract, so accepting an active checkpoint as genesis would let a caller
    discard an economic prefix and reseal its local coordinate denominator.
    """

    segment_paths = list(_sequence(paths, "segment paths"))
    if not segment_paths:
        raise DojoEconomicSegmentError("segment chain must not be empty")
    expected_prior = GENESIS_SHA256
    expected_index = 0
    expected_coordinate_start = 0
    expected_source_offset: int | None = None
    previous_terminal_checkpoint: dict[str, Any] | None = None
    previous_source_chain: str | None = None
    seen: set[str] = set()
    first_bindings: dict[str, Any] | None = None
    last_bindings: dict[str, Any] | None = None
    for index, path in enumerate(segment_paths):
        # Decode, independently replay, and release each full transcript
        # segment before opening the next one.  Only the prior checkpoint and
        # scalar chain bindings survive the loop boundary.
        segment = verify_economic_segment(Path(path))
        denominator = segment["coordinate_denominator"]
        source_range = segment["source_range"]
        if segment["segment_sha256"] in seen:
            raise DojoEconomicSegmentError("duplicate segment digest in chain")
        seen.add(segment["segment_sha256"])
        if (
            segment["segment_index"] != expected_index
            or segment["prior_segment_sha256"] != expected_prior
            or denominator["segment_coordinate_start"] != expected_coordinate_start
        ):
            raise DojoEconomicSegmentError("segment chain is reordered, forked, or gapped")
        if index == 0:
            start_checkpoint = segment["start_checkpoint"]
            if (
                start_checkpoint["state_kind"] != "FRESH_INITIAL_BALANCE"
                or start_checkpoint["origin_coordinate_seq"] != 0
                or start_checkpoint["processed_coordinate_count"] != 0
            ):
                raise DojoEconomicSegmentError(
                    "segment chain genesis is not a fresh initial balance"
                )
            first_bindings = {
                "transcript_id": segment["transcript_id"],
                "job_sha256": segment["job_sha256"],
                "policy_sha256": segment["policy_sha256"],
                "source_slice_receipt_sha256": source_range[
                    "source_slice_receipt_sha256"
                ],
                "expected_job_coordinate_count": denominator[
                    "expected_job_coordinate_count"
                ],
            }
        elif first_bindings is None:  # pragma: no cover - loop invariant
            raise DojoEconomicSegmentError("segment chain genesis is missing")
        else:
            if (
                segment["transcript_id"] != first_bindings["transcript_id"]
                or segment["job_sha256"] != first_bindings["job_sha256"]
                or segment["policy_sha256"] != first_bindings["policy_sha256"]
                or source_range["source_slice_receipt_sha256"]
                != first_bindings["source_slice_receipt_sha256"]
                or denominator["expected_job_coordinate_count"]
                != first_bindings["expected_job_coordinate_count"]
            ):
                raise DojoEconomicSegmentError(
                    "segment chain immutable binding drifted"
                )
        if index == 0:
            expected_source_offset = source_range["offset_start"]
        if source_range["offset_start"] != expected_source_offset:
            raise DojoEconomicSegmentError("segment source ranges are not contiguous")
        if previous_terminal_checkpoint is not None and (
            segment["start_checkpoint"] != previous_terminal_checkpoint
            or segment["prior_source_batch_chain_sha256"] != previous_source_chain
        ):
            raise DojoEconomicSegmentError(
                "segment checkpoint or source-chain handoff differs"
            )
        if index < len(segment_paths) - 1 and segment["terminal_segment"]:
            raise DojoEconomicSegmentError("records follow a terminal segment")
        expected_prior = segment["segment_sha256"]
        expected_index += 1
        expected_coordinate_start = denominator["segment_coordinate_end_exclusive"]
        expected_source_offset = source_range["offset_end_exclusive"]
        previous_terminal_checkpoint = segment["terminal_checkpoint"]
        previous_source_chain = segment["terminal_source_batch_chain_sha256"]
        last_bindings = {
            "segment_sha256": segment["segment_sha256"],
            "terminal_checkpoint_sha256": segment["terminal_checkpoint_sha256"],
            "terminal_source_batch_chain_sha256": segment[
                "terminal_source_batch_chain_sha256"
            ],
            "terminal_segment": segment["terminal_segment"],
        }
        # Avoid retaining the previous decoded transcript while the next file
        # is read.  The exact checkpoint handoff is the only structured state
        # intentionally kept across iterations.
        del denominator, source_range, segment
    if first_bindings is None or last_bindings is None:  # pragma: no cover
        raise DojoEconomicSegmentError("segment chain is empty")
    terminal = last_bindings["terminal_segment"]
    if require_terminal and not terminal:
        raise DojoEconomicSegmentError("segment chain is not terminal")
    body: dict[str, Any] = {
        "contract": ECONOMIC_SEGMENT_CHAIN_ATTESTATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "VERIFIED_TERMINAL" if terminal else "VERIFIED_RESUMABLE_PREFIX",
        "transcript_id": first_bindings["transcript_id"],
        "job_sha256": first_bindings["job_sha256"],
        "policy_sha256": first_bindings["policy_sha256"],
        "segment_count": len(segment_paths),
        "segment_chain_tip_sha256": last_bindings["segment_sha256"],
        "completed_coordinate_count": expected_coordinate_start,
        "expected_job_coordinate_count": first_bindings[
            "expected_job_coordinate_count"
        ],
        "terminal_checkpoint_sha256": last_bindings[
            "terminal_checkpoint_sha256"
        ],
        "terminal_source_batch_chain_sha256": last_bindings[
            "terminal_source_batch_chain_sha256"
        ],
        "terminal_segment": terminal,
        "deterministic_restore_verified": True,
        "implicit_no_intent_semantics": IMPLICIT_NO_INTENT_SEMANTICS,
        # A supplied branch set is checked below, but a local path list cannot
        # prove that no unlisted sibling or hostile rollback exists.
        "external_monotonic_anchor_configured": False,
        "fork_absence_proven": False,
        "official_evidence_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["chain_attestation_sha256"] = canonical_portfolio_sha256(body)
    return body


__all__ = [
    "DojoEconomicSegmentError",
    "ECONOMIC_SEGMENT_CHAIN_ATTESTATION_CONTRACT",
    "ECONOMIC_SEGMENT_CONTRACT",
    "EconomicSegmentWriter",
    "IMPLICIT_NO_INTENT_SEMANTICS",
    "build_economic_segment",
    "verify_economic_segment",
    "verify_economic_segment_chain",
    "verify_economic_segment_value",
]
