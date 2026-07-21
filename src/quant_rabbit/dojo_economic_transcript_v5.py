"""Sparse-event DOJO economic evidence vertical slice.

V5 reuses one V4 shared-source artifact for every account coordinate while
storing account events only at source ordinals that contain at least one
non-HOLD worker intent.  Event payloads deliberately omit snapshot hashes,
allocation receipts, quote/source-chain references, and full snapshots.  The
auditor scans every shared-source ordinal, reconstructs the post-exit snapshot,
seals event proposal payloads against that snapshot, supplies canonical HOLD
for every omitted active worker, and lets ``PortfolioReplaySession`` recompute
allocation and economics.

Storage is O(E) per account, where E is the number of non-HOLD source ordinals;
verification remains O(N) in the shared-source batch denominator.  The current
vertical slice supports one bounded V4 source segment and a fresh initial
balance only.  Multi-source segment chaining, predecessor/carry continuation,
crash-resume worker state, and an external monotonic anchor remain separate
blockers.  Every artifact is research-only and has ``order_authority=NONE``.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_economic_transcript_v4 import (
    DojoEconomicTranscriptV4Error,
    verify_shared_source_segment_value,
)
from quant_rabbit.dojo_portfolio_replay_reducer import (
    MONTH_END_FLAT_SETTLEMENT,
    MONTH_END_MTM_WITH_STATE_HANDOFF,
    DojoPortfolioReplayError,
    PortfolioReplaySession,
    canonical_portfolio_sha256,
    validate_portfolio_replay_result,
    verify_portfolio_policy,
    verify_portfolio_replay_checkpoint,
)
from quant_rabbit.dojo_shared_worker_protocol import (
    ProtocolViolation,
    seal_worker_proposal,
    seal_worker_proposal_batch,
)


ACCOUNT_DELTA_CONTRACT: Final = "QR_DOJO_ECONOMIC_SPARSE_ACCOUNT_DELTA_V5"
ACCOUNT_ATTESTATION_CONTRACT: Final = "QR_DOJO_ECONOMIC_SPARSE_ACCOUNT_REEXECUTION_V5"
SUBSET_REEXECUTION_ATTESTATION_CONTRACT: Final = (
    "QR_DOJO_ECONOMIC_SPARSE_SUBSET_REEXECUTION_V5"
)
SCHEMA_VERSION: Final = 5
MAX_ARTIFACT_BYTES: Final = 512 * 1024 * 1024

_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_EVENT_KEYS = frozenset(
    {
        "source_ordinal",
        "non_hold_proposal_payloads",
        "event_sha256",
    }
)
_PROPOSAL_PAYLOAD_KEYS = frozenset(
    {
        "worker_id",
        "owner_id",
        "family_id",
        "config_sha256",
        "risk_reducing_intents",
        "new_risk_intents",
    }
)
_DELTA_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "transcript_id",
        "job_sha256",
        "coordinate_id",
        "source_segment_sha256",
        "source_slice_receipt_sha256",
        "portfolio_policy",
        "policy_sha256",
        "expected_source_batch_count",
        "start_checkpoint",
        "start_checkpoint_sha256",
        "terminal_checkpoint",
        "terminal_checkpoint_sha256",
        "terminal_policy",
        "producer_portfolio_result_sha256",
        "producer_portfolio_carry_state_sha256",
        "events",
        "events_sha256",
        "event_count",
        "account_storage_complexity",
        "auditor_scan_complexity",
        "omitted_active_worker_semantics",
        "quote_arrays_embedded",
        "full_snapshots_embedded",
        "post_exit_snapshot_hashes_embedded",
        "allocation_receipts_embedded",
        "source_chain_references_embedded",
        "single_source_segment_vertical_slice",
        "multi_source_segment_chaining_configured",
        "external_monotonic_anchor_configured",
        "fork_absence_proven",
        "official_evidence_eligible",
        "live_permission",
        "broker_mutation_allowed",
        "order_authority",
        "account_delta_sha256",
    }
)


class DojoEconomicTranscriptV5Error(ValueError):
    """Sparse account evidence is incomplete or economically inconsistent."""


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
        raise DojoEconomicTranscriptV5Error(
            "value is not strict canonical JSON"
        ) from exc


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoEconomicTranscriptV5Error(f"{field} must be a string-keyed object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DojoEconomicTranscriptV5Error(f"{field} must be a sequence")
    return value


def _exact(value: Mapping[str, Any], keys: frozenset[str], field: str) -> None:
    actual = frozenset(value)
    if actual != keys:
        raise DojoEconomicTranscriptV5Error(
            f"{field} schema mismatch: missing={sorted(keys-actual)}, "
            f"extra={sorted(actual-keys)}"
        )


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise DojoEconomicTranscriptV5Error(
            f"{field} must be a non-zero lowercase SHA-256"
        )
    return value


def _identifier(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(char) < 32 for char in value)
    ):
        raise DojoEconomicTranscriptV5Error(f"{field} must be a trimmed identifier")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoEconomicTranscriptV5Error(f"{field} must be an integer >= {minimum}")
    return value


def _verified_source(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return verify_shared_source_segment_value(value)
    except DojoEconomicTranscriptV4Error as exc:
        raise DojoEconomicTranscriptV5Error("shared source is invalid") from exc


def _proposal_payload_from_sealed(proposal: Mapping[str, Any]) -> dict[str, Any]:
    new_risk = _copy(proposal["new_risk_intents"])
    for intent in new_risk:
        intent["parameters"].pop("activation_policy", None)
    return {
        "worker_id": proposal["worker_id"],
        "owner_id": proposal["owner_id"],
        "family_id": proposal["family_id"],
        "config_sha256": proposal["config_sha256"],
        "risk_reducing_intents": _copy(proposal["risk_reducing_intents"]),
        "new_risk_intents": new_risk,
    }


def _proposal_batch_for_event(
    *,
    snapshot: Mapping[str, Any],
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bindings = {row["worker_id"]: row for row in snapshot["active_worker_bindings"]}
    supplied: dict[str, dict[str, Any]] = {}
    canonical_payloads: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(payloads):
        payload = dict(_mapping(raw, f"event proposal payloads[{index}]"))
        _exact(
            payload,
            _PROPOSAL_PAYLOAD_KEYS,
            f"event proposal payloads[{index}]",
        )
        try:
            proposal = seal_worker_proposal(
                snapshot,
                {**payload, "snapshot_sha256": snapshot["snapshot_sha256"]},
            )
        except ProtocolViolation as exc:
            raise DojoEconomicTranscriptV5Error(
                f"event proposal payloads[{index}] is invalid"
            ) from exc
        if not (
            proposal["intent_counts"]["risk_reducing"]
            or proposal["intent_counts"]["new_risk"]
        ):
            raise DojoEconomicTranscriptV5Error(
                "sparse event cannot contain an explicit HOLD proposal"
            )
        worker_id = proposal["worker_id"]
        if worker_id in supplied:
            raise DojoEconomicTranscriptV5Error(
                "sparse event contains a duplicate worker proposal"
            )
        supplied[worker_id] = proposal
        canonical_payloads[worker_id] = _proposal_payload_from_sealed(proposal)
    if not supplied:
        raise DojoEconomicTranscriptV5Error(
            "sparse event must contain at least one non-HOLD proposal"
        )
    if not set(supplied).issubset(bindings):
        raise DojoEconomicTranscriptV5Error(
            "sparse event proposal is outside the active worker denominator"
        )
    proposals: list[dict[str, Any]] = []
    for worker_id in sorted(bindings):
        proposal = supplied.get(worker_id)
        if proposal is None:
            try:
                proposal = seal_worker_proposal(
                    snapshot,
                    {
                        **bindings[worker_id],
                        "snapshot_sha256": snapshot["snapshot_sha256"],
                        "risk_reducing_intents": [],
                        "new_risk_intents": [],
                    },
                )
            except ProtocolViolation as exc:  # pragma: no cover - sealed policy
                raise DojoEconomicTranscriptV5Error(
                    "canonical omitted-worker HOLD is invalid"
                ) from exc
        proposals.append(proposal)
    try:
        batch = seal_worker_proposal_batch(snapshot, proposals)
    except ProtocolViolation as exc:  # pragma: no cover - guarded above
        raise DojoEconomicTranscriptV5Error(
            "expanded sparse proposal batch is invalid"
        ) from exc
    return batch, [canonical_payloads[key] for key in sorted(canonical_payloads)]


def _canonical_hold_batch(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    bindings = sorted(
        snapshot["active_worker_bindings"], key=lambda row: row["worker_id"]
    )
    try:
        proposals = [
            seal_worker_proposal(
                snapshot,
                {
                    **binding,
                    "snapshot_sha256": snapshot["snapshot_sha256"],
                    "risk_reducing_intents": [],
                    "new_risk_intents": [],
                },
            )
            for binding in bindings
        ]
        return seal_worker_proposal_batch(snapshot, proposals)
    except ProtocolViolation as exc:  # pragma: no cover - sealed policy/snapshot
        raise DojoEconomicTranscriptV5Error(
            "canonical all-worker HOLD batch is invalid"
        ) from exc


def _replay_sparse_events(
    *,
    source: Mapping[str, Any],
    coordinate_id: str,
    policy: Mapping[str, Any],
    start_checkpoint: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> tuple[PortfolioReplaySession, list[dict[str, Any]]]:
    raw_events = list(events)
    prior_ordinal = -1
    for index, raw in enumerate(raw_events):
        event = _mapping(raw, f"events[{index}]")
        _exact(
            event,
            _EVENT_KEYS - {"event_sha256"},
            f"events[{index}]",
        )
        ordinal = _integer(event["source_ordinal"], f"events[{index}].source_ordinal")
        if ordinal <= prior_ordinal:
            raise DojoEconomicTranscriptV5Error(
                "sparse event ordinals must be strictly increasing and unique"
            )
        if ordinal >= source["expected_source_batch_count"]:
            raise DojoEconomicTranscriptV5Error(
                "sparse event ordinal is outside the shared-source denominator"
            )
        prior_ordinal = ordinal
    try:
        session = PortfolioReplaySession.restore_checkpoint(
            policy=policy,
            checkpoint=start_checkpoint,
        )
    except DojoPortfolioReplayError as exc:
        raise DojoEconomicTranscriptV5Error(
            "account start checkpoint cannot be restored"
        ) from exc
    sealed_events: list[dict[str, Any]] = []
    event_index = 0
    for ordinal, source_batch in enumerate(source["batches"]):
        try:
            snapshot = session.prepare_coordinate(
                coordinate_id=coordinate_id,
                epoch=source_batch["epoch"],
                phase=source_batch["phase"],
                intrabar=source_batch["intrabar"],
                quote_watermark=source_batch["quote_watermark"],
                quotes=source_batch["quotes"],
                quote_batch_sha256_value=source_batch["quote_batch_sha256"],
            )
            if (
                event_index < len(raw_events)
                and raw_events[event_index]["source_ordinal"] == ordinal
            ):
                proposal_batch, canonical_payloads = _proposal_batch_for_event(
                    snapshot=snapshot,
                    payloads=_sequence(
                        raw_events[event_index]["non_hold_proposal_payloads"],
                        f"events[{event_index}].non_hold_proposal_payloads",
                    ),
                )
                body = {
                    "source_ordinal": ordinal,
                    "non_hold_proposal_payloads": canonical_payloads,
                }
                body["event_sha256"] = canonical_portfolio_sha256(body)
                sealed_events.append(body)
                event_index += 1
            else:
                proposal_batch = _canonical_hold_batch(snapshot)
            session.consume_proposal_batch(proposal_batch)
        except (DojoPortfolioReplayError, ProtocolViolation) as exc:
            raise DojoEconomicTranscriptV5Error(
                f"independent replay failed at source ordinal {ordinal}"
            ) from exc
    if event_index != len(raw_events):  # pragma: no cover - denominator guard above
        raise DojoEconomicTranscriptV5Error("not every sparse event was replayed")
    return session, sealed_events


def _fresh_checkpoint(
    *, policy: Mapping[str, Any], checkpoint: Mapping[str, Any], field: str
) -> dict[str, Any]:
    try:
        sealed = verify_portfolio_replay_checkpoint(
            policy=policy,
            checkpoint=checkpoint,
        )
    except DojoPortfolioReplayError as exc:
        raise DojoEconomicTranscriptV5Error(f"{field} is invalid") from exc
    return sealed


def build_sparse_account_delta(
    *,
    transcript_id: str,
    coordinate_id: str,
    shared_source_segment: Mapping[str, Any],
    portfolio_policy: Mapping[str, Any],
    start_checkpoint: Mapping[str, Any],
    terminal_checkpoint: Mapping[str, Any],
    terminal_policy: str,
    producer_portfolio_result: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build an O(E) account delta and independently replay all N source rows."""

    source = _verified_source(shared_source_segment)
    try:
        policy = verify_portfolio_policy(portfolio_policy)
    except DojoPortfolioReplayError as exc:
        raise DojoEconomicTranscriptV5Error("portfolio policy is invalid") from exc
    start = _fresh_checkpoint(
        policy=policy,
        checkpoint=start_checkpoint,
        field="start checkpoint",
    )
    terminal = _fresh_checkpoint(
        policy=policy,
        checkpoint=terminal_checkpoint,
        field="terminal checkpoint",
    )
    if (
        start["state_kind"] != "FRESH_INITIAL_BALANCE"
        or start["origin_coordinate_seq"] != 0
        or start["processed_coordinate_count"] != 0
    ):
        raise DojoEconomicTranscriptV5Error(
            "V5 vertical-slice account genesis is not a fresh initial balance"
        )
    if terminal_policy not in {
        MONTH_END_FLAT_SETTLEMENT,
        MONTH_END_MTM_WITH_STATE_HANDOFF,
    }:
        raise DojoEconomicTranscriptV5Error("terminal policy is unsupported")
    session, sealed_events = _replay_sparse_events(
        source=source,
        coordinate_id=_identifier(coordinate_id, "coordinate_id"),
        policy=policy,
        start_checkpoint=start,
        events=list(_sequence(events, "events")),
    )
    if session.export_checkpoint() != terminal:
        raise DojoEconomicTranscriptV5Error(
            "producer terminal checkpoint differs from independent sparse replay"
        )
    try:
        result = validate_portfolio_replay_result(
            session.finalize(terminal_policy=terminal_policy)
        )
        producer = validate_portfolio_replay_result(producer_portfolio_result)
    except DojoPortfolioReplayError as exc:
        raise DojoEconomicTranscriptV5Error("portfolio result is invalid") from exc
    if result != producer:
        raise DojoEconomicTranscriptV5Error(
            "producer result differs from independent sparse replay"
        )
    body = {
        "contract": ACCOUNT_DELTA_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "transcript_id": _identifier(transcript_id, "transcript_id"),
        "job_sha256": source["job_sha256"],
        "coordinate_id": _identifier(coordinate_id, "coordinate_id"),
        "source_segment_sha256": source["source_segment_sha256"],
        "source_slice_receipt_sha256": source["source_slice_receipt_sha256"],
        "portfolio_policy": policy,
        "policy_sha256": policy["policy_sha256"],
        "expected_source_batch_count": source["expected_source_batch_count"],
        "start_checkpoint": start,
        "start_checkpoint_sha256": start["checkpoint_sha256"],
        "terminal_checkpoint": terminal,
        "terminal_checkpoint_sha256": terminal["checkpoint_sha256"],
        "terminal_policy": terminal_policy,
        "producer_portfolio_result_sha256": result["result_sha256"],
        "producer_portfolio_carry_state_sha256": result["carry_state_sha256"],
        "events": sealed_events,
        "events_sha256": canonical_portfolio_sha256(sealed_events),
        "event_count": len(sealed_events),
        "account_storage_complexity": "O(NON_HOLD_SOURCE_ORDINALS)",
        "auditor_scan_complexity": "O(SHARED_SOURCE_BATCHES)",
        "omitted_active_worker_semantics": "CANONICAL_HOLD",
        "quote_arrays_embedded": False,
        "full_snapshots_embedded": False,
        "post_exit_snapshot_hashes_embedded": False,
        "allocation_receipts_embedded": False,
        "source_chain_references_embedded": False,
        "single_source_segment_vertical_slice": True,
        "multi_source_segment_chaining_configured": False,
        "external_monotonic_anchor_configured": False,
        "fork_absence_proven": False,
        "official_evidence_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["account_delta_sha256"] = canonical_portfolio_sha256(body)
    return body


def verify_sparse_account_delta_value(
    value: Mapping[str, Any], *, shared_source_segment: Mapping[str, Any]
) -> dict[str, Any]:
    source = _verified_source(shared_source_segment)
    row = dict(_mapping(value, "sparse account delta"))
    _exact(row, _DELTA_KEYS, "sparse account delta")
    claimed = row.pop("account_delta_sha256")
    if canonical_portfolio_sha256(row) != _sha(claimed, "account_delta_sha256"):
        raise DojoEconomicTranscriptV5Error("sparse account delta digest is invalid")
    if (
        row["contract"] != ACCOUNT_DELTA_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
        or _identifier(row["transcript_id"], "transcript_id") != row["transcript_id"]
        or _identifier(row["coordinate_id"], "coordinate_id") != row["coordinate_id"]
        or row["job_sha256"] != source["job_sha256"]
        or row["source_segment_sha256"] != source["source_segment_sha256"]
        or row["source_slice_receipt_sha256"] != source["source_slice_receipt_sha256"]
        or _integer(
            row["expected_source_batch_count"],
            "expected_source_batch_count",
            minimum=1,
        )
        != source["expected_source_batch_count"]
        or row["account_storage_complexity"] != "O(NON_HOLD_SOURCE_ORDINALS)"
        or row["auditor_scan_complexity"] != "O(SHARED_SOURCE_BATCHES)"
        or row["omitted_active_worker_semantics"] != "CANONICAL_HOLD"
        or row["quote_arrays_embedded"] is not False
        or row["full_snapshots_embedded"] is not False
        or row["post_exit_snapshot_hashes_embedded"] is not False
        or row["allocation_receipts_embedded"] is not False
        or row["source_chain_references_embedded"] is not False
        or row["single_source_segment_vertical_slice"] is not True
        or row["multi_source_segment_chaining_configured"] is not False
        or row["external_monotonic_anchor_configured"] is not False
        or row["fork_absence_proven"] is not False
        or row["official_evidence_eligible"] is not False
        or row["live_permission"] is not False
        or row["broker_mutation_allowed"] is not False
        or row["order_authority"] != "NONE"
    ):
        raise DojoEconomicTranscriptV5Error(
            "sparse account contract, source, or authority boundary is invalid"
        )
    raw_events: list[dict[str, Any]] = []
    stored_events = list(_sequence(row["events"], "events"))
    for index, raw in enumerate(stored_events):
        event = dict(_mapping(raw, f"events[{index}]"))
        _exact(event, _EVENT_KEYS, f"events[{index}]")
        claimed_event_sha = event.pop("event_sha256")
        if canonical_portfolio_sha256(event) != _sha(
            claimed_event_sha, f"events[{index}].event_sha256"
        ):
            raise DojoEconomicTranscriptV5Error("sparse event digest is invalid")
        raw_events.append(event)
    if canonical_portfolio_sha256(stored_events) != _sha(
        row["events_sha256"], "events_sha256"
    ) or _integer(row["event_count"], "event_count") != len(stored_events):
        raise DojoEconomicTranscriptV5Error("sparse event denominator is invalid")
    try:
        policy = verify_portfolio_policy(row["portfolio_policy"])
    except DojoPortfolioReplayError as exc:
        raise DojoEconomicTranscriptV5Error("portfolio policy is invalid") from exc
    start = _fresh_checkpoint(
        policy=policy,
        checkpoint=row["start_checkpoint"],
        field="start checkpoint",
    )
    terminal = _fresh_checkpoint(
        policy=policy,
        checkpoint=row["terminal_checkpoint"],
        field="terminal checkpoint",
    )
    if (
        start["state_kind"] != "FRESH_INITIAL_BALANCE"
        or start["origin_coordinate_seq"] != 0
        or start["processed_coordinate_count"] != 0
    ):
        raise DojoEconomicTranscriptV5Error(
            "V5 vertical-slice account genesis is not a fresh initial balance"
        )
    if row["terminal_policy"] not in {
        MONTH_END_FLAT_SETTLEMENT,
        MONTH_END_MTM_WITH_STATE_HANDOFF,
    }:
        raise DojoEconomicTranscriptV5Error("terminal policy is unsupported")
    session, sealed_events = _replay_sparse_events(
        source=source,
        coordinate_id=_identifier(row["coordinate_id"], "coordinate_id"),
        policy=policy,
        start_checkpoint=start,
        events=raw_events,
    )
    if session.export_checkpoint() != terminal:
        raise DojoEconomicTranscriptV5Error(
            "producer terminal checkpoint differs from independent sparse replay"
        )
    try:
        result = validate_portfolio_replay_result(
            session.finalize(terminal_policy=row["terminal_policy"])
        )
    except DojoPortfolioReplayError as exc:
        raise DojoEconomicTranscriptV5Error(
            "independently replayed result is invalid"
        ) from exc
    if (
        sealed_events != stored_events
        or row["policy_sha256"] != policy["policy_sha256"]
        or row["start_checkpoint_sha256"] != start["checkpoint_sha256"]
        or row["terminal_checkpoint_sha256"] != terminal["checkpoint_sha256"]
        or row["producer_portfolio_result_sha256"] != result["result_sha256"]
        or row["producer_portfolio_carry_state_sha256"] != result["carry_state_sha256"]
    ):
        raise DojoEconomicTranscriptV5Error(
            "sparse account independent economics or commitments are invalid"
        )
    return {**row, "account_delta_sha256": claimed}


def _strict_document(raw: bytes, field: str) -> Mapping[str, Any]:
    if not raw.endswith(b"\n") or len(raw) > MAX_ARTIFACT_BYTES:
        raise DojoEconomicTranscriptV5Error(f"{field} is truncated or oversized")

    def reject_constant(token: str) -> None:
        raise DojoEconomicTranscriptV5Error(f"non-finite token is forbidden: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DojoEconomicTranscriptV5Error(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except DojoEconomicTranscriptV5Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoEconomicTranscriptV5Error(f"{field} is not strict JSON") from exc
    row = _mapping(parsed, field)
    if raw != _canonical_bytes(row) + b"\n":
        raise DojoEconomicTranscriptV5Error(f"{field} is not canonical JSON")
    return row


def _read_immutable(path: Path, field: str) -> tuple[Mapping[str, Any], str, int]:
    artifact_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(artifact_path, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > MAX_ARTIFACT_BYTES
        ):
            raise DojoEconomicTranscriptV5Error(
                f"{field} must be a bounded single-link regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            raw = handle.read(MAX_ARTIFACT_BYTES + 1)
            after = os.fstat(handle.fileno())
        current = artifact_path.stat(follow_symlinks=False)
        identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        if (
            identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            )
            != identity
        ):
            raise DojoEconomicTranscriptV5Error(f"{field} changed during verification")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return _strict_document(raw, field), hashlib.sha256(raw).hexdigest(), len(raw)


def verify_v5_account_files(
    source_path: Path, account_delta_path: Path
) -> dict[str, Any]:
    source_raw, source_file_sha, source_file_bytes = _read_immutable(
        source_path, "shared source file"
    )
    source = _verified_source(source_raw)
    delta_raw, delta_file_sha, delta_file_bytes = _read_immutable(
        account_delta_path, "sparse account delta file"
    )
    delta = verify_sparse_account_delta_value(
        delta_raw,
        shared_source_segment=source,
    )
    body = {
        "contract": ACCOUNT_ATTESTATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "ACCOUNT_REEXECUTED_UNANCHORED",
        "job_sha256": source["job_sha256"],
        "transcript_id": delta["transcript_id"],
        "coordinate_id": delta["coordinate_id"],
        "source_segment_sha256": source["source_segment_sha256"],
        "source_file_sha256": source_file_sha,
        "source_file_bytes": source_file_bytes,
        "account_delta_sha256": delta["account_delta_sha256"],
        "account_delta_file_sha256": delta_file_sha,
        "account_delta_file_bytes": delta_file_bytes,
        "completed_source_batch_count": source["expected_source_batch_count"],
        "sparse_event_count": delta["event_count"],
        "portfolio_result_sha256": delta["producer_portfolio_result_sha256"],
        "portfolio_carry_state_sha256": delta["producer_portfolio_carry_state_sha256"],
        "independent_full_source_scan_passed": True,
        "implicit_hold_reexecution_passed": True,
        "partial_economics_reported": False,
        "complete_job_coordinate_denominator_proven": False,
        "multi_source_segment_chaining_configured": False,
        "external_monotonic_anchor_configured": False,
        "fork_absence_proven": False,
        "official_evidence_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["account_attestation_sha256"] = canonical_portfolio_sha256(body)
    return body


def verify_v5_coordinate_subset(
    *,
    source_path: Path,
    account_delta_paths: Sequence[Path],
    caller_declared_coordinate_ids: Sequence[str],
) -> dict[str, Any]:
    """Reexecute a caller-declared subset without claiming the job denominator."""

    expected = [
        _identifier(item, "caller-declared coordinate id")
        for item in caller_declared_coordinate_ids
    ]
    if not expected or expected != sorted(set(expected)):
        raise DojoEconomicTranscriptV5Error(
            "caller-declared coordinate ids must be non-empty, sorted, and unique"
        )
    paths = [Path(path) for path in account_delta_paths]
    if len(paths) != len(expected):
        raise DojoEconomicTranscriptV5Error(
            "sparse account path count differs from the caller-declared subset"
        )
    attestations: dict[str, dict[str, Any]] = {}
    for path in paths:
        attestation = verify_v5_account_files(source_path, path)
        coordinate_id = attestation["coordinate_id"]
        if coordinate_id in attestations:
            raise DojoEconomicTranscriptV5Error(
                "duplicate/forked sparse account coordinate in declared subset"
            )
        attestations[coordinate_id] = attestation
    if set(attestations) != set(expected):
        raise DojoEconomicTranscriptV5Error(
            "sparse accounts do not equal the caller-declared coordinate subset"
        )
    source_hashes = {row["source_segment_sha256"] for row in attestations.values()}
    source_file_hashes = {row["source_file_sha256"] for row in attestations.values()}
    if len(source_hashes) != 1 or len(source_file_hashes) != 1:
        raise DojoEconomicTranscriptV5Error(
            "sparse accounts do not share one exact source artifact"
        )
    body = {
        "contract": SUBSET_REEXECUTION_ATTESTATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "SUBSET_REEXECUTED_UNANCHORED",
        "caller_declared_coordinate_ids": expected,
        "reexecuted_coordinate_count": len(expected),
        "source_segment_sha256": next(iter(source_hashes)),
        "source_file_sha256": next(iter(source_file_hashes)),
        "source_file_bytes": next(iter(attestations.values()))["source_file_bytes"],
        "account_attestation_sha256_by_coordinate": {
            coordinate_id: attestations[coordinate_id]["account_attestation_sha256"]
            for coordinate_id in expected
        },
        "portfolio_result_sha256_by_coordinate": {
            coordinate_id: attestations[coordinate_id]["portfolio_result_sha256"]
            for coordinate_id in expected
        },
        "account_delta_file_bytes": sum(
            attestations[coordinate_id]["account_delta_file_bytes"]
            for coordinate_id in expected
        ),
        "source_bytes_scale_with_coordinate_count": False,
        "independent_full_source_scan_passed": True,
        "account_economics_complete_for_reexecuted_subset": True,
        "complete_job_coordinate_denominator_proven": False,
        "fixed_coordinate_denominator_proven": False,
        "job_economics_complete": False,
        "partial_economics_reported": True,
        "multi_source_segment_chaining_configured": False,
        "external_monotonic_anchor_configured": False,
        "fork_absence_proven": False,
        "official_evidence_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["subset_reexecution_attestation_sha256"] = canonical_portfolio_sha256(body)
    return body


__all__ = [
    "ACCOUNT_ATTESTATION_CONTRACT",
    "ACCOUNT_DELTA_CONTRACT",
    "DojoEconomicTranscriptV5Error",
    "SUBSET_REEXECUTION_ATTESTATION_CONTRACT",
    "build_sparse_account_delta",
    "verify_sparse_account_delta_value",
    "verify_v5_account_files",
    "verify_v5_coordinate_subset",
]
