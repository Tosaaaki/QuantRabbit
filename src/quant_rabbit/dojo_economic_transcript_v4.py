"""Shared-source/account-delta DOJO economic evidence vertical slice.

V4 stores one immutable quote/source-chain segment for a job and one compact
account delta per portfolio coordinate.  Account deltas contain no quote arrays
and no full post-exit snapshots: they bind source ordinals/digests, the
independently reproducible snapshot digest, non-HOLD proposals, allocation
receipts, reducer checkpoints, and opaque worker-state checkpoints.

The current vertical slice deliberately supports one bounded fresh-balance
source segment.  It does not yet define multi-segment source continuation,
month-to-month predecessor binding, crash-resume worker restoration, or an
external monotonic anchor.  Every artifact is research-only and has
``order_authority=NONE``.
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

from quant_rabbit.dojo_portfolio_replay_reducer import (
    MONTH_END_FLAT_SETTLEMENT,
    MONTH_END_MTM_WITH_STATE_HANDOFF,
    PortfolioReplaySession,
    canonical_portfolio_sha256,
    quote_batch_sha256,
    validate_portfolio_replay_result,
    verify_portfolio_policy,
    verify_portfolio_replay_checkpoint,
)
from quant_rabbit.dojo_shared_worker_protocol import (
    ProtocolViolation,
    seal_worker_proposal,
    seal_worker_proposal_batch,
    verify_worker_proposal,
)


SHARED_SOURCE_CONTRACT: Final = "QR_DOJO_ECONOMIC_SHARED_SOURCE_SEGMENT_V4"
ACCOUNT_DELTA_CONTRACT: Final = "QR_DOJO_ECONOMIC_ACCOUNT_DELTA_V4"
ACCOUNT_ATTESTATION_CONTRACT: Final = (
    "QR_DOJO_ECONOMIC_SHARED_SOURCE_ACCOUNT_REEXECUTION_V4"
)
FIXED_DENOMINATOR_ATTESTATION_CONTRACT: Final = (
    "QR_DOJO_ECONOMIC_SHARED_SOURCE_FIXED_DENOMINATOR_V4"
)
SCHEMA_VERSION: Final = 4
GENESIS_SHA256: Final = "0" * 64
MAX_SOURCE_BATCHES: Final = 4_096
MAX_ARTIFACT_BYTES: Final = 512 * 1024 * 1024
MAX_WORKER_STATE_BYTES: Final = 8 * 1024 * 1024
PRIVATE_FILE_MODE: Final = 0o600

_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "job_sha256",
        "source_slice_receipt_sha256",
        "expected_source_batch_count",
        "source_batch_start",
        "source_batch_end_exclusive",
        "prior_source_batch_chain_sha256",
        "terminal_source_batch_chain_sha256",
        "batches",
        "source_batches_sha256",
        "source_range_unit",
        "single_segment_vertical_slice",
        "external_monotonic_anchor_configured",
        "fork_absence_proven",
        "official_evidence_eligible",
        "live_permission",
        "broker_mutation_allowed",
        "order_authority",
        "source_segment_sha256",
    }
)
_SOURCE_BATCH_KEYS = frozenset(
    {
        "source_ordinal",
        "epoch",
        "phase",
        "intrabar",
        "quote_watermark",
        "quotes",
        "quote_batch_sha256",
        "source_batch_chain_sha256",
        "source_batch_sha256",
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
        "start_worker_state",
        "start_worker_state_sha256",
        "terminal_worker_state",
        "terminal_worker_state_sha256",
        "terminal_policy",
        "producer_portfolio_result_sha256",
        "producer_portfolio_carry_state_sha256",
        "delta_rows",
        "delta_rows_sha256",
        "quote_arrays_embedded",
        "full_snapshots_embedded",
        "implicit_no_intent_semantics",
        "single_segment_vertical_slice",
        "external_monotonic_anchor_configured",
        "fork_absence_proven",
        "official_evidence_eligible",
        "live_permission",
        "broker_mutation_allowed",
        "order_authority",
        "account_delta_sha256",
    }
)
_DELTA_ROW_KEYS = frozenset(
    {
        "source_ordinal",
        "source_quote_batch_sha256",
        "source_batch_chain_sha256",
        "post_exit_snapshot_sha256",
        "non_hold_proposals",
        "allocation_receipt",
        "delta_row_sha256",
    }
)


class DojoEconomicTranscriptV4Error(ValueError):
    """Shared-source/account-delta evidence is incomplete or inconsistent."""


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
        raise DojoEconomicTranscriptV4Error("value is not strict canonical JSON") from exc


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoEconomicTranscriptV4Error(f"{field} must be a string-keyed object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DojoEconomicTranscriptV4Error(f"{field} must be a sequence")
    return value


def _exact(value: Mapping[str, Any], keys: frozenset[str], field: str) -> None:
    actual = frozenset(value)
    if actual != keys:
        raise DojoEconomicTranscriptV4Error(
            f"{field} schema mismatch: missing={sorted(keys-actual)}, "
            f"extra={sorted(actual-keys)}"
        )


def _sha(value: Any, field: str, *, allow_zero: bool = False) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoEconomicTranscriptV4Error(f"{field} must be a lowercase SHA-256")
    if not allow_zero and value == GENESIS_SHA256:
        raise DojoEconomicTranscriptV4Error(f"{field} must not be the zero digest")
    return value


def _identifier(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(char) < 32 for char in value)
    ):
        raise DojoEconomicTranscriptV4Error(f"{field} must be a trimmed identifier")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoEconomicTranscriptV4Error(f"{field} must be an integer >= {minimum}")
    return value


def _source_chain(
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


def _worker_state(value: Any, field: str) -> tuple[Any, str]:
    raw = _canonical_bytes(value)
    if len(raw) > MAX_WORKER_STATE_BYTES:
        raise DojoEconomicTranscriptV4Error(f"{field} exceeds its byte bound")
    return _copy(value), hashlib.sha256(raw).hexdigest()


def _expand_proposal_batch(
    *,
    snapshot: Mapping[str, Any],
    non_hold_proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bindings = {row["worker_id"]: row for row in snapshot["active_worker_bindings"]}
    supplied: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(non_hold_proposals):
        try:
            proposal = verify_worker_proposal(snapshot, raw)
        except ProtocolViolation as exc:
            raise DojoEconomicTranscriptV4Error(
                f"non_hold_proposals[{index}] is invalid"
            ) from exc
        if not (
            proposal["intent_counts"]["risk_reducing"]
            or proposal["intent_counts"]["new_risk"]
        ):
            raise DojoEconomicTranscriptV4Error(
                "explicit HOLD/NO_INTENT proposal is forbidden"
            )
        worker_id = proposal["worker_id"]
        if worker_id in supplied:
            raise DojoEconomicTranscriptV4Error("duplicate non-HOLD proposal")
        supplied[worker_id] = proposal
    if not set(supplied).issubset(bindings):
        raise DojoEconomicTranscriptV4Error(
            "non-HOLD proposal is outside the worker denominator"
        )
    proposals: list[dict[str, Any]] = []
    for worker_id in sorted(bindings):
        proposal = supplied.get(worker_id)
        if proposal is None:
            proposal = seal_worker_proposal(
                snapshot,
                {
                    **bindings[worker_id],
                    "snapshot_sha256": snapshot["snapshot_sha256"],
                    "risk_reducing_intents": [],
                    "new_risk_intents": [],
                },
            )
        proposals.append(proposal)
    try:
        return seal_worker_proposal_batch(snapshot, proposals)
    except ProtocolViolation as exc:  # pragma: no cover - guarded above
        raise DojoEconomicTranscriptV4Error(
            "expanded proposal batch is invalid"
        ) from exc


def build_shared_source_segment(
    *,
    job_sha256: str,
    source_slice_receipt_sha256: str,
    batches: Sequence[Mapping[str, Any]],
    prior_source_batch_chain_sha256: str = GENESIS_SHA256,
) -> dict[str, Any]:
    """Build one bounded quote/source-chain document shared by all accounts."""

    raw_batches = list(_sequence(batches, "source batches"))
    if not raw_batches or len(raw_batches) > MAX_SOURCE_BATCHES:
        raise DojoEconomicTranscriptV4Error(
            "source batch denominator is outside the vertical-slice bound"
        )
    prior = _sha(
        prior_source_batch_chain_sha256,
        "prior_source_batch_chain_sha256",
        allow_zero=True,
    )
    sealed_batches: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_batches):
        row = dict(_mapping(raw, f"source batches[{ordinal}]"))
        expected = frozenset(
            {"epoch", "phase", "intrabar", "quote_watermark", "quotes"}
        )
        _exact(row, expected, f"source batches[{ordinal}]")
        epoch = _integer(row["epoch"], f"source batches[{ordinal}].epoch")
        phase = row["phase"]
        intrabar = row["intrabar"]
        if phase not in {"O", "H", "L", "C"} or intrabar not in {"OHLC", "OLHC"}:
            raise DojoEconomicTranscriptV4Error("source phase/intrabar is unsupported")
        watermark = _integer(
            row["quote_watermark"],
            f"source batches[{ordinal}].quote_watermark",
            minimum=1,
        )
        quotes = list(_sequence(row["quotes"], f"source batches[{ordinal}].quotes"))
        quote_digest = quote_batch_sha256(
            epoch=epoch,
            phase=phase,
            intrabar=intrabar,
            quote_watermark=watermark,
            quotes=quotes,
        )
        prior = _source_chain(
            prior,
            quote_digest=quote_digest,
            epoch=epoch,
            phase=phase,
            watermark=watermark,
        )
        body = {
            "source_ordinal": ordinal,
            "epoch": epoch,
            "phase": phase,
            "intrabar": intrabar,
            "quote_watermark": watermark,
            "quotes": _copy(quotes),
            "quote_batch_sha256": quote_digest,
            "source_batch_chain_sha256": prior,
        }
        body["source_batch_sha256"] = canonical_portfolio_sha256(body)
        sealed_batches.append(body)
    body = {
        "contract": SHARED_SOURCE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "job_sha256": _sha(job_sha256, "job_sha256"),
        "source_slice_receipt_sha256": _sha(
            source_slice_receipt_sha256, "source_slice_receipt_sha256"
        ),
        "expected_source_batch_count": len(sealed_batches),
        "source_batch_start": 0,
        "source_batch_end_exclusive": len(sealed_batches),
        "prior_source_batch_chain_sha256": _sha(
            prior_source_batch_chain_sha256,
            "prior_source_batch_chain_sha256",
            allow_zero=True,
        ),
        "terminal_source_batch_chain_sha256": prior,
        "batches": sealed_batches,
        "source_batches_sha256": canonical_portfolio_sha256(sealed_batches),
        "source_range_unit": "NORMALIZED_QUOTE_BATCH_ORDINAL",
        "single_segment_vertical_slice": True,
        "external_monotonic_anchor_configured": False,
        "fork_absence_proven": False,
        "official_evidence_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["source_segment_sha256"] = canonical_portfolio_sha256(body)
    return body


def verify_shared_source_segment_value(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(_mapping(value, "shared source segment"))
    _exact(row, _SOURCE_KEYS, "shared source segment")
    claimed = row.pop("source_segment_sha256")
    if canonical_portfolio_sha256(row) != _sha(claimed, "source_segment_sha256"):
        raise DojoEconomicTranscriptV4Error("shared source segment digest is invalid")
    if (
        row["contract"] != SHARED_SOURCE_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
        or row["source_batch_start"] != 0
        or row["source_range_unit"] != "NORMALIZED_QUOTE_BATCH_ORDINAL"
        or row["single_segment_vertical_slice"] is not True
        or row["external_monotonic_anchor_configured"] is not False
        or row["fork_absence_proven"] is not False
        or row["official_evidence_eligible"] is not False
        or row["live_permission"] is not False
        or row["broker_mutation_allowed"] is not False
        or row["order_authority"] != "NONE"
    ):
        raise DojoEconomicTranscriptV4Error(
            "shared source contract or authority boundary is invalid"
        )
    raw_batches = list(_sequence(row["batches"], "shared source batches"))
    expected_count = _integer(
        row["expected_source_batch_count"], "expected_source_batch_count", minimum=1
    )
    if (
        len(raw_batches) != expected_count
        or row["source_batch_end_exclusive"] != expected_count
        or canonical_portfolio_sha256(raw_batches) != row["source_batches_sha256"]
    ):
        raise DojoEconomicTranscriptV4Error("shared source denominator is invalid")
    rebuild_inputs: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_batches):
        batch = dict(_mapping(raw, f"shared source batches[{ordinal}]"))
        _exact(batch, _SOURCE_BATCH_KEYS, f"shared source batches[{ordinal}]")
        if batch["source_ordinal"] != ordinal:
            raise DojoEconomicTranscriptV4Error("shared source ordinal is not contiguous")
        rebuild_inputs.append(
            {
                "epoch": batch["epoch"],
                "phase": batch["phase"],
                "intrabar": batch["intrabar"],
                "quote_watermark": batch["quote_watermark"],
                "quotes": batch["quotes"],
            }
        )
    rebuilt = build_shared_source_segment(
        job_sha256=row["job_sha256"],
        source_slice_receipt_sha256=row["source_slice_receipt_sha256"],
        batches=rebuild_inputs,
        prior_source_batch_chain_sha256=row["prior_source_batch_chain_sha256"],
    )
    candidate = {**row, "source_segment_sha256": claimed}
    if rebuilt != candidate:
        raise DojoEconomicTranscriptV4Error("shared source segment is not canonical")
    return rebuilt


def _replay_account_delta(
    *,
    source: Mapping[str, Any],
    coordinate_id: str,
    policy: Mapping[str, Any],
    start_checkpoint: Mapping[str, Any],
    delta_rows: Sequence[Mapping[str, Any]],
) -> tuple[PortfolioReplaySession, list[dict[str, Any]]]:
    session = PortfolioReplaySession.restore_checkpoint(
        policy=policy,
        checkpoint=start_checkpoint,
    )
    sealed_rows: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(delta_rows):
        row = dict(_mapping(raw, f"delta rows[{ordinal}]"))
        expected = _DELTA_ROW_KEYS - {"delta_row_sha256"}
        _exact(row, frozenset(expected), f"delta rows[{ordinal}]")
        source_batch = source["batches"][ordinal]
        if (
            row["source_ordinal"] != ordinal
            or row["source_quote_batch_sha256"]
            != source_batch["quote_batch_sha256"]
            or row["source_batch_chain_sha256"]
            != source_batch["source_batch_chain_sha256"]
        ):
            raise DojoEconomicTranscriptV4Error(
                "account delta source reference is invalid"
            )
        snapshot = session.prepare_coordinate(
            coordinate_id=coordinate_id,
            epoch=source_batch["epoch"],
            phase=source_batch["phase"],
            intrabar=source_batch["intrabar"],
            quote_watermark=source_batch["quote_watermark"],
            quotes=source_batch["quotes"],
            quote_batch_sha256_value=source_batch["quote_batch_sha256"],
        )
        if row["post_exit_snapshot_sha256"] != snapshot["snapshot_sha256"]:
            raise DojoEconomicTranscriptV4Error(
                "account delta snapshot digest differs from independent replay"
            )
        non_hold = list(
            _sequence(row["non_hold_proposals"], f"delta rows[{ordinal}].proposals")
        )
        proposal_batch = _expand_proposal_batch(
            snapshot=snapshot,
            non_hold_proposals=non_hold,
        )
        receipt = session.consume_proposal_batch(proposal_batch)
        if receipt != row["allocation_receipt"]:
            raise DojoEconomicTranscriptV4Error(
                "account delta allocation differs from independent replay"
            )
        body = {
            "source_ordinal": ordinal,
            "source_quote_batch_sha256": source_batch["quote_batch_sha256"],
            "source_batch_chain_sha256": source_batch[
                "source_batch_chain_sha256"
            ],
            "post_exit_snapshot_sha256": snapshot["snapshot_sha256"],
            "non_hold_proposals": _copy(non_hold),
            "allocation_receipt": receipt,
        }
        body["delta_row_sha256"] = canonical_portfolio_sha256(body)
        sealed_rows.append(body)
    return session, sealed_rows


def build_account_delta(
    *,
    transcript_id: str,
    coordinate_id: str,
    shared_source_segment: Mapping[str, Any],
    portfolio_policy: Mapping[str, Any],
    start_checkpoint: Mapping[str, Any],
    terminal_checkpoint: Mapping[str, Any],
    start_worker_state: Any,
    terminal_worker_state: Any,
    terminal_policy: str,
    producer_portfolio_result: Mapping[str, Any],
    delta_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one quote-free account delta and replay it independently."""

    source = verify_shared_source_segment_value(shared_source_segment)
    policy = verify_portfolio_policy(portfolio_policy)
    start = verify_portfolio_replay_checkpoint(
        policy=policy,
        checkpoint=start_checkpoint,
    )
    terminal = verify_portfolio_replay_checkpoint(
        policy=policy,
        checkpoint=terminal_checkpoint,
    )
    if (
        start["state_kind"] != "FRESH_INITIAL_BALANCE"
        or start["origin_coordinate_seq"] != 0
        or start["processed_coordinate_count"] != 0
    ):
        raise DojoEconomicTranscriptV4Error(
            "V4 vertical-slice account genesis is not a fresh initial balance"
        )
    raw_rows = list(_sequence(delta_rows, "delta rows"))
    if len(raw_rows) != source["expected_source_batch_count"]:
        raise DojoEconomicTranscriptV4Error(
            "account delta denominator differs from shared source"
        )
    session, sealed_rows = _replay_account_delta(
        source=source,
        coordinate_id=_identifier(coordinate_id, "coordinate_id"),
        policy=policy,
        start_checkpoint=start,
        delta_rows=raw_rows,
    )
    replay_terminal = session.export_checkpoint()
    if replay_terminal != terminal:
        raise DojoEconomicTranscriptV4Error(
            "account terminal checkpoint differs from independent replay"
        )
    if terminal_policy not in {
        MONTH_END_FLAT_SETTLEMENT,
        MONTH_END_MTM_WITH_STATE_HANDOFF,
    }:
        raise DojoEconomicTranscriptV4Error("terminal policy is unsupported")
    result = validate_portfolio_replay_result(
        session.finalize(terminal_policy=terminal_policy)
    )
    producer = validate_portfolio_replay_result(producer_portfolio_result)
    if result != producer:
        raise DojoEconomicTranscriptV4Error(
            "producer portfolio result differs from independent V4 replay"
        )
    start_worker, start_worker_sha = _worker_state(
        start_worker_state, "start_worker_state"
    )
    terminal_worker, terminal_worker_sha = _worker_state(
        terminal_worker_state, "terminal_worker_state"
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
        "start_worker_state": start_worker,
        "start_worker_state_sha256": start_worker_sha,
        "terminal_worker_state": terminal_worker,
        "terminal_worker_state_sha256": terminal_worker_sha,
        "terminal_policy": terminal_policy,
        "producer_portfolio_result_sha256": result["result_sha256"],
        "producer_portfolio_carry_state_sha256": result["carry_state_sha256"],
        "delta_rows": sealed_rows,
        "delta_rows_sha256": canonical_portfolio_sha256(sealed_rows),
        "quote_arrays_embedded": False,
        "full_snapshots_embedded": False,
        "implicit_no_intent_semantics": (
            "OMITTED_ACTIVE_WORKER_IS_CANONICAL_EMPTY_PROPOSAL"
        ),
        "single_segment_vertical_slice": True,
        "external_monotonic_anchor_configured": False,
        "fork_absence_proven": False,
        "official_evidence_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["account_delta_sha256"] = canonical_portfolio_sha256(body)
    return body


def verify_account_delta_value(
    value: Mapping[str, Any], *, shared_source_segment: Mapping[str, Any]
) -> dict[str, Any]:
    source = verify_shared_source_segment_value(shared_source_segment)
    row = dict(_mapping(value, "account delta"))
    _exact(row, _DELTA_KEYS, "account delta")
    claimed = row.pop("account_delta_sha256")
    if canonical_portfolio_sha256(row) != _sha(claimed, "account_delta_sha256"):
        raise DojoEconomicTranscriptV4Error("account delta digest is invalid")
    if (
        row["contract"] != ACCOUNT_DELTA_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
        or row["source_segment_sha256"] != source["source_segment_sha256"]
        or row["source_slice_receipt_sha256"]
        != source["source_slice_receipt_sha256"]
        or row["job_sha256"] != source["job_sha256"]
        or row["quote_arrays_embedded"] is not False
        or row["full_snapshots_embedded"] is not False
        or row["single_segment_vertical_slice"] is not True
        or row["external_monotonic_anchor_configured"] is not False
        or row["fork_absence_proven"] is not False
        or row["official_evidence_eligible"] is not False
        or row["live_permission"] is not False
        or row["broker_mutation_allowed"] is not False
        or row["order_authority"] != "NONE"
    ):
        raise DojoEconomicTranscriptV4Error(
            "account delta contract, source, or authority boundary is invalid"
        )
    raw_delta_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(_sequence(row["delta_rows"], "delta rows")):
        delta = dict(_mapping(raw, f"delta rows[{index}]"))
        _exact(delta, _DELTA_ROW_KEYS, f"delta rows[{index}]")
        claimed_row_sha = delta.pop("delta_row_sha256")
        if canonical_portfolio_sha256(delta) != _sha(
            claimed_row_sha, f"delta rows[{index}].delta_row_sha256"
        ):
            raise DojoEconomicTranscriptV4Error("account delta row digest is invalid")
        raw_delta_rows.append(delta)
    if canonical_portfolio_sha256(row["delta_rows"]) != row["delta_rows_sha256"]:
        raise DojoEconomicTranscriptV4Error("account delta rows digest is invalid")
    start_worker, start_worker_sha = _worker_state(
        row["start_worker_state"], "start_worker_state"
    )
    terminal_worker, terminal_worker_sha = _worker_state(
        row["terminal_worker_state"], "terminal_worker_state"
    )
    if (
        start_worker_sha != row["start_worker_state_sha256"]
        or terminal_worker_sha != row["terminal_worker_state_sha256"]
    ):
        raise DojoEconomicTranscriptV4Error("worker-state checkpoint digest is invalid")
    policy = verify_portfolio_policy(row["portfolio_policy"])
    start = verify_portfolio_replay_checkpoint(
        policy=policy,
        checkpoint=row["start_checkpoint"],
    )
    terminal = verify_portfolio_replay_checkpoint(
        policy=policy,
        checkpoint=row["terminal_checkpoint"],
    )
    if (
        start["state_kind"] != "FRESH_INITIAL_BALANCE"
        or start["origin_coordinate_seq"] != 0
        or start["processed_coordinate_count"] != 0
    ):
        raise DojoEconomicTranscriptV4Error(
            "V4 vertical-slice account genesis is not a fresh initial balance"
        )
    session, sealed_rows = _replay_account_delta(
        source=source,
        coordinate_id=row["coordinate_id"],
        policy=policy,
        start_checkpoint=start,
        delta_rows=raw_delta_rows,
    )
    if session.export_checkpoint() != terminal:
        raise DojoEconomicTranscriptV4Error(
            "account terminal checkpoint differs from independent replay"
        )
    result = validate_portfolio_replay_result(
        session.finalize(terminal_policy=row["terminal_policy"])
    )
    if (
        result["result_sha256"] != row["producer_portfolio_result_sha256"]
        or result["carry_state_sha256"]
        != row["producer_portfolio_carry_state_sha256"]
        or sealed_rows != row["delta_rows"]
        or row["expected_source_batch_count"]
        != source["expected_source_batch_count"]
        or row["policy_sha256"] != policy["policy_sha256"]
        or row["start_checkpoint_sha256"] != start["checkpoint_sha256"]
        or row["terminal_checkpoint_sha256"] != terminal["checkpoint_sha256"]
        or start_worker != row["start_worker_state"]
        or terminal_worker != row["terminal_worker_state"]
    ):
        raise DojoEconomicTranscriptV4Error(
            "account delta independent economics or denominator is invalid"
        )
    return {**row, "account_delta_sha256": claimed}


def _strict_document(raw: bytes, field: str) -> Mapping[str, Any]:
    if not raw.endswith(b"\n") or len(raw) > MAX_ARTIFACT_BYTES:
        raise DojoEconomicTranscriptV4Error(f"{field} is truncated or oversized")

    def reject_constant(token: str) -> None:
        raise DojoEconomicTranscriptV4Error(f"non-finite token is forbidden: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DojoEconomicTranscriptV4Error(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except DojoEconomicTranscriptV4Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoEconomicTranscriptV4Error(f"{field} is not strict JSON") from exc
    row = _mapping(parsed, field)
    if raw != _canonical_bytes(row) + b"\n":
        raise DojoEconomicTranscriptV4Error(f"{field} is not canonical JSON")
    return row


def publish_immutable_artifact(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one canonical single-link V4 artifact with one data fsync."""

    artifact_path = Path(path)
    raw = _canonical_bytes(value) + b"\n"
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise DojoEconomicTranscriptV4Error("artifact exceeds its byte bound")
    parent = artifact_path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise DojoEconomicTranscriptV4Error(
            "artifact parent must be an existing non-symlink directory"
        )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(artifact_path, flags, PRIVATE_FILE_MODE)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        view = memoryview(raw)
        written = 0
        while written < len(raw):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short V4 artifact publication")
            written += count
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        current = artifact_path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != len(raw)
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (current.st_dev, current.st_ino, current.st_size)
        ):
            raise DojoEconomicTranscriptV4Error(
                "artifact publication identity or size changed"
            )
    finally:
        os.close(descriptor)


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
            raise DojoEconomicTranscriptV4Error(
                f"{field} must be a bounded single-link regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            raw = handle.read(MAX_ARTIFACT_BYTES + 1)
            after = os.fstat(handle.fileno())
        current = artifact_path.stat(follow_symlinks=False)
        identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ) != identity:
            raise DojoEconomicTranscriptV4Error(f"{field} changed during verification")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return _strict_document(raw, field), hashlib.sha256(raw).hexdigest(), len(raw)


def verify_v4_account_files(
    source_path: Path, account_delta_path: Path
) -> dict[str, Any]:
    source_raw, source_file_sha, source_file_bytes = _read_immutable(
        source_path, "shared source file"
    )
    source = verify_shared_source_segment_value(source_raw)
    delta_raw, delta_file_sha, delta_file_bytes = _read_immutable(
        account_delta_path, "account delta file"
    )
    delta = verify_account_delta_value(
        delta_raw,
        shared_source_segment=source,
    )
    body = {
        "contract": ACCOUNT_ATTESTATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "VERIFIED_COMPLETE",
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
        "terminal_source_batch_chain_sha256": source[
            "terminal_source_batch_chain_sha256"
        ],
        "portfolio_result_sha256": delta["producer_portfolio_result_sha256"],
        "portfolio_carry_state_sha256": delta[
            "producer_portfolio_carry_state_sha256"
        ],
        "quote_arrays_in_account_delta": False,
        "full_snapshots_in_account_delta": False,
        "independent_economic_reexecution_passed": True,
        "partial_economics_reported": False,
        "external_monotonic_anchor_configured": False,
        "fork_absence_proven": False,
        "official_evidence_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["account_attestation_sha256"] = canonical_portfolio_sha256(body)
    return body


def verify_v4_fixed_denominator(
    *,
    source_path: Path,
    account_delta_paths: Sequence[Path],
    expected_coordinate_ids: Sequence[str],
) -> dict[str, Any]:
    """Reject omitted, duplicated, or forked account deltas for one source."""

    expected = [_identifier(item, "expected coordinate id") for item in expected_coordinate_ids]
    if not expected or expected != sorted(set(expected)):
        raise DojoEconomicTranscriptV4Error(
            "expected coordinate ids must be non-empty, sorted, and unique"
        )
    paths = [Path(path) for path in account_delta_paths]
    if len(paths) != len(expected):
        raise DojoEconomicTranscriptV4Error(
            "account delta path denominator differs from expected coordinates"
        )
    attestations: dict[str, dict[str, Any]] = {}
    for path in paths:
        attestation = verify_v4_account_files(source_path, path)
        coordinate_id = attestation["coordinate_id"]
        if coordinate_id in attestations:
            raise DojoEconomicTranscriptV4Error(
                "duplicate/forked account coordinate in fixed denominator"
            )
        attestations[coordinate_id] = attestation
    if set(attestations) != set(expected):
        raise DojoEconomicTranscriptV4Error(
            "account deltas do not equal the fixed coordinate denominator"
        )
    source_hashes = {row["source_segment_sha256"] for row in attestations.values()}
    source_file_hashes = {row["source_file_sha256"] for row in attestations.values()}
    if len(source_hashes) != 1 or len(source_file_hashes) != 1:
        raise DojoEconomicTranscriptV4Error(
            "account deltas do not share one exact source artifact"
        )
    body = {
        "contract": FIXED_DENOMINATOR_ATTESTATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "VERIFIED_COMPLETE_UNANCHORED",
        "expected_coordinate_ids": expected,
        "coordinate_count": len(expected),
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
        "independent_economic_reexecution_passed": True,
        "partial_economics_reported": False,
        "external_monotonic_anchor_configured": False,
        "fork_absence_proven": False,
        "official_evidence_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    body["fixed_denominator_attestation_sha256"] = canonical_portfolio_sha256(body)
    return body


__all__ = [
    "ACCOUNT_ATTESTATION_CONTRACT",
    "ACCOUNT_DELTA_CONTRACT",
    "DojoEconomicTranscriptV4Error",
    "FIXED_DENOMINATOR_ATTESTATION_CONTRACT",
    "SHARED_SOURCE_CONTRACT",
    "build_account_delta",
    "build_shared_source_segment",
    "publish_immutable_artifact",
    "verify_account_delta_value",
    "verify_shared_source_segment_value",
    "verify_v4_account_files",
    "verify_v4_fixed_denominator",
]
