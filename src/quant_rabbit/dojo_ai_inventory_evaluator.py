"""Prospective outcome evaluator for future DOJO paper-AI inventory rooms.

This module records research scores only.  It cannot open, close, reduce,
cancel, or otherwise mutate a virtual or real broker, and it deliberately has
no broker, runner, OANDA, or live-gateway import.

The trusted evaluator API opens the canonical room's durable files directly
and proves that the supplied decision digest identifies:

* one ``QR_DOJO_AI_INVENTORY_DECISION_V2`` row;
* its immutable producer receipt; and
* its ``AI_INVENTORY_ACTION_APPLIED`` virtual-broker receipt.

It validates each full hash chain, binds the position and point-in-time quote,
and recomputes settlement or fixed-horizon P/L, MFE, and MAE from those source
files.  A digest or prose assertion by itself is never action authority.  The
low-level writer requires a one-use trusted-evidence token minted only by that
direct-validation path.

New evaluations may be appended only while the deterministic FX week is open.
An exact retry of an already durable score remains idempotent while the market
is closed.  The score clock is generated internally, so callers cannot
backdate a weekend evaluation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_ai_inventory import validate_inventory_decision
from quant_rabbit.dojo_ai_evidence_packet import (
    DEDICATED_EVIDENCE_ROOT,
    verify_ai_inventory_evidence_packet,
)
from quant_rabbit.dojo_ai_inventory_producer import (
    verify_ai_inventory_producer_receipt,
)
from quant_rabbit.dojo_replay_lifecycle import canonical_paper_ai_rooms_root


DOJO_AI_INVENTORY_OUTCOME_CONTRACT = "QR_DOJO_AI_INVENTORY_OUTCOME_V1"
DOJO_AI_INVENTORY_EVALUATION_CONTRACT = "QR_DOJO_AI_INVENTORY_EVALUATION_V1"
REQUIRED_DECISION_CONTRACT = "QR_DOJO_AI_INVENTORY_DECISION_V2"
REQUIRED_PRODUCER_RECEIPT_CONTRACT = "QR_DOJO_AI_INVENTORY_PRODUCER_RECEIPT_V1"
REQUIRED_APPLIED_RECEIPT_EVENT = "AI_INVENTORY_ACTION_APPLIED"
GENESIS_EVALUATION_SHA256 = "0" * 64

MAX_LEDGER_LINE_BYTES = 256 * 1024
MAX_LEDGER_BYTES = 256 * 1024 * 1024
MAX_LEDGER_ROWS = 1_000_000
MAX_SOURCE_WATERMARKS = 64
MAX_ID_CHARS = 256
DECISION_LEDGER_NAME = "ai_inventory_decisions.jsonl"
BROKER_LEDGER_NAME = "broker_ledger.jsonl"
QUOTE_WATERMARK_LEDGER_NAME = "quote_watermarks.jsonl"
EVALUATION_LEDGER_NAME = "evaluations.jsonl"
MAX_SOURCE_LEDGER_BYTES = 256 * 1024 * 1024
SIGNED_ASSESSMENT_CONTRACT = "QR_DOJO_AI_INVENTORY_ASSESSMENT_V1"

ALLOWED_OUTCOME_KINDS = frozenset({"SETTLEMENT", "FIXED_HORIZON"})
ALLOWED_OUTCOMES = frozenset({"WIN", "LOSS", "FLAT"})
ALLOWED_REGIMES = frozenset({"TREND", "RANGE", "SQUEEZE", "EVENT", "UNCLEAR"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_REASON_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,95}$")
_UTC_RE = re.compile(
    r"^(?P<seconds>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)

_OUTCOME_KEYS = frozenset(
    {
        "contract",
        "decision_contract",
        "producer_receipt_contract",
        "applied_receipt_event",
        "decision_sha256",
        "producer_receipt_sha256",
        "applied_receipt_sha256",
        "position_identity",
        "signal_identity",
        "decision_cutoff_at_utc",
        "horizon_end_at_utc",
        "outcome_observed_at_utc",
        "outcome_kind",
        "realized_outcome",
        "settlement_reason",
        "realized_pl_jpy",
        "mfe_jpy",
        "mae_jpy",
        "review_time_executable_exit_pl_jpy",
        "actual_exit_pl_jpy",
        "counterfactual_delta_jpy",
        "declared_assessment",
        "declared_assessment_sha256",
        "declared_regime",
        "realized_regime",
        "regime_correct",
        "regime_confidence",
        "regime_brier_score",
        "source_watermarks",
        "paper_only",
        "order_authority",
        "live_permission",
        "external_broker_mutation_allowed",
        "evaluation_is_not_action",
    }
)
_RECORD_KEYS = _OUTCOME_KEYS | frozenset(
    {
        "evaluation_contract",
        "sequence",
        "previous_evaluation_sha256",
        "scored_at_utc",
        "evaluation_identity_sha256",
        "evaluation_sha256",
    }
)
_POSITION_KEYS = frozenset(
    {
        "position_id",
        "pair",
        "side",
        "strategy_tag",
        "entry_context_sha256",
    }
)
_SIGNAL_KEYS = frozenset(
    {
        "signal_identity_sha256",
        "pair",
        "side",
        "strategy_tag",
        "entry_context_sha256",
    }
)
_WATERMARK_KEYS = frozenset({"source_id", "sha256", "watermark_at_utc"})
_BROKER_ROW_KEYS = frozenset({"ts_utc", "event", "payload", "prev_sha", "sha"})
_QUOTE_ROW_KEYS = frozenset(
    {
        "contract",
        "sequence",
        "recorded_at_utc",
        "timestamp_utc",
        "pair",
        "bid",
        "ask",
        "source_sha256",
        "capture_source_sha256",
        "acquisition_receipt_sha256",
        "slippage_pips_per_fill",
        "financing_pips_per_day",
        "previous_quote_sha256",
        "quote_sha256",
        "paper_only",
        "order_authority",
        "live_permission",
    }
)
_SIGNED_ASSESSMENT_KEYS = frozenset(
    {
        "contract",
        "declared_regime",
        "assessment",
        "primary_path",
        "alternative_path",
        "falsifier",
    }
)


class AiInventoryEvaluationError(RuntimeError):
    """Base class for fail-closed outcome-evaluation failures."""


class AiInventoryEvaluationIntegrityError(AiInventoryEvaluationError):
    """The existing evaluation ledger is not trustworthy."""


class AiInventoryEvaluationConflictError(AiInventoryEvaluationError):
    """The same decision and horizon already have a different score."""


class AiInventoryEvaluationMarketClosedError(AiInventoryEvaluationError):
    """A new score was attempted outside the open FX week."""


@dataclass(frozen=True)
class EvaluationAppendResult:
    """Result of one outcome-score append attempt."""

    record: dict[str, Any]
    appended: bool


@dataclass(frozen=True)
class _TrustedEvidenceToken:
    """One-call capability minted only after canonical source verification."""

    packet_sha256: str
    room_root: str
    decision_sha256: str
    applied_receipt_sha256: str
    producer_receipt_sha256: str


_ACTIVE_TRUSTED_TOKENS: set[int] = set()


def evaluate_ai_inventory_outcome(
    room_root: Path,
    *,
    decision_sha256: str,
    horizon_end_at_utc: str,
    outcome_kind: str,
) -> EvaluationAppendResult:
    """Reconstruct and score one outcome from canonical immutable room files.

    The source paths are not caller-selectable.  They are derived as
    ``ai_inventory_decisions.jsonl``,
    ``producer_receipts/<sha>.json``, ``broker_ledger.jsonl``, and
    ``quote_watermarks.jsonl`` directly beneath one canonical isolated
    paper-AI room.  All files are opened with ``O_NOFOLLOW`` and fully
    validated before any evaluation row is appended.
    """

    # The weekend gate deliberately precedes path resolution and every source
    # read.  A closed-market call therefore performs no reconstruction or
    # evaluation work, including an otherwise-idempotent retry.
    evaluation_started_at = _utc_now().astimezone(timezone.utc)
    _require_market_open(evaluation_started_at)
    root = _require_canonical_room_root(room_root)
    if not _is_sha256(decision_sha256):
        raise AiInventoryEvaluationIntegrityError("trusted decision_sha256 is invalid")
    horizon_ns = _parse_utc_nanoseconds(horizon_end_at_utc)
    if horizon_ns is None:
        raise AiInventoryEvaluationIntegrityError(
            "trusted horizon_end_at_utc is invalid"
        )
    if outcome_kind not in ALLOWED_OUTCOME_KINDS:
        raise AiInventoryEvaluationIntegrityError("trusted outcome_kind is invalid")

    packet = _build_trusted_outcome_packet(
        root,
        decision_sha256=decision_sha256,
        horizon_end_at_utc=horizon_end_at_utc,
        horizon_ns=horizon_ns,
        outcome_kind=outcome_kind,
    )
    token = _TrustedEvidenceToken(
        packet_sha256=_sha256(_canonical_json(packet).encode("utf-8")),
        room_root=str(root),
        decision_sha256=packet["decision_sha256"],
        applied_receipt_sha256=packet["applied_receipt_sha256"],
        producer_receipt_sha256=packet["producer_receipt_sha256"],
    )
    _ACTIVE_TRUSTED_TOKENS.add(id(token))
    try:
        return append_ai_inventory_evaluation(
            root / EVALUATION_LEDGER_NAME,
            packet,
            _trusted_evidence_token=token,
        )
    finally:
        _ACTIVE_TRUSTED_TOKENS.discard(id(token))


def append_ai_inventory_evaluation(
    path: Path | str,
    outcome_packet: Mapping[str, Any],
    *,
    _trusted_evidence_token: _TrustedEvidenceToken | None = None,
) -> EvaluationAppendResult:
    """Append one prospective score to a fully validated hash-chain ledger.

    ``scored_at_utc`` is intentionally absent from ``outcome_packet`` and is
    authored inside this function.  Exact retries return the existing record;
    contradictory retries fail without rewriting history.
    """

    packet = _seal_outcome_packet(outcome_packet)
    packet_sha256 = _sha256(_canonical_json(packet).encode("utf-8"))
    token = _trusted_evidence_token
    if (
        token is None
        or token.__class__ is not _TrustedEvidenceToken
        or id(token) not in _ACTIVE_TRUSTED_TOKENS
        or token.packet_sha256 != packet_sha256
        or token.decision_sha256 != packet["decision_sha256"]
        or token.applied_receipt_sha256 != packet["applied_receipt_sha256"]
        or token.producer_receipt_sha256 != packet["producer_receipt_sha256"]
    ):
        raise AiInventoryEvaluationIntegrityError(
            "raw evaluation append requires a fresh trusted evidence token"
        )
    _ACTIVE_TRUSTED_TOKENS.remove(id(token))
    ledger_path = Path(path)
    if ledger_path.name != EVALUATION_LEDGER_NAME:
        raise AiInventoryEvaluationIntegrityError(
            "evaluation ledger must use its canonical room-local name"
        )
    try:
        if ledger_path.parent.resolve(strict=True) != Path(token.room_root).resolve(
            strict=True
        ):
            raise AiInventoryEvaluationIntegrityError(
                "evaluation ledger is outside the trusted room root"
            )
    except OSError as exc:
        raise AiInventoryEvaluationIntegrityError(
            "trusted room root is unavailable"
        ) from exc
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    handle = _open_locked_ledger(ledger_path, exclusive=True, create=True)
    try:
        raw = _read_locked(handle)
        rows, decode_issues = _decode_ledger_bytes(raw)
        validation = _validate_ledger_rows(rows, initial_issues=decode_issues)
        if not validation["valid"]:
            raise AiInventoryEvaluationIntegrityError(
                "AI inventory evaluation ledger failed validation: "
                + "; ".join(validation["issues"])
            )

        identity = _evaluation_identity_sha256(packet)
        existing = [
            row for row in rows if row.get("evaluation_identity_sha256") == identity
        ]
        if existing:
            if len(existing) != 1:
                raise AiInventoryEvaluationIntegrityError(
                    "evaluation identity is duplicated"
                )
            if _outcome_projection(existing[0]) != packet:
                raise AiInventoryEvaluationConflictError(
                    "same decision and horizon already have a different score"
                )
            return EvaluationAppendResult(record=dict(existing[0]), appended=False)

        scored_at = _utc_now().astimezone(timezone.utc)
        _require_market_open(scored_at)
        scored_at_utc = _datetime_to_canonical(scored_at)
        scored_at_ns = _parse_utc_nanoseconds(scored_at_utc)
        assert scored_at_ns is not None
        _validate_packet_at_score_time(packet, scored_at_ns)

        record = {
            **packet,
            "evaluation_contract": DOJO_AI_INVENTORY_EVALUATION_CONTRACT,
            "sequence": len(rows) + 1,
            "previous_evaluation_sha256": (
                rows[-1]["evaluation_sha256"] if rows else GENESIS_EVALUATION_SHA256
            ),
            "scored_at_utc": scored_at_utc,
            "evaluation_identity_sha256": identity,
        }
        record["evaluation_sha256"] = _evaluation_sha256(record)
        issues = _record_issues(record)
        if issues:
            raise AiInventoryEvaluationError(
                "new evaluation record failed validation: " + "; ".join(issues)
            )

        encoded = (_canonical_json(record) + "\n").encode("utf-8")
        if len(encoded) > MAX_LEDGER_LINE_BYTES:
            raise AiInventoryEvaluationError(
                "evaluation record exceeds the line-size limit"
            )
        handle.seek(0, os.SEEK_END)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())

        handle.seek(0)
        persisted = handle.read()
        persisted_rows, persisted_decode_issues = _decode_ledger_bytes(persisted)
        persisted_validation = _validate_ledger_rows(
            persisted_rows, initial_issues=persisted_decode_issues
        )
        if (
            not persisted_validation["valid"]
            or persisted_validation["terminal_evaluation_sha256"]
            != record["evaluation_sha256"]
        ):
            raise AiInventoryEvaluationIntegrityError(
                "persisted evaluation ledger did not validate"
            )
        return EvaluationAppendResult(record=record, appended=True)
    finally:
        _unlock_close(handle)


def validate_ai_inventory_evaluation_ledger(
    path: Path | str,
) -> dict[str, Any]:
    """Validate the complete append-only evaluation ledger without mutation."""

    ledger_path = Path(path)
    if not ledger_path.exists():
        return _validation_result((), (), None)
    try:
        handle = _open_locked_ledger(ledger_path, exclusive=False, create=False)
    except OSError as exc:
        return _validation_result(
            (f"LEDGER_READ_FAILED:{exc.__class__.__name__}",), (), None
        )
    try:
        raw = _read_locked(handle)
    except OSError as exc:
        return _validation_result(
            (f"LEDGER_READ_FAILED:{exc.__class__.__name__}",), (), None
        )
    finally:
        _unlock_close(handle)
    rows, decode_issues = _decode_ledger_bytes(raw)
    return _validate_ledger_rows(rows, initial_issues=decode_issues)


def quote_watermark_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical digest for one quote-watermark row."""

    snapshot = _snapshot_mapping(value, "quote watermark")
    body = {key: item for key, item in snapshot.items() if key != "quote_sha256"}
    return _sha256(_canonical_json(body).encode("utf-8"))


def _build_trusted_outcome_packet(
    room_root: Path,
    *,
    decision_sha256: str,
    horizon_end_at_utc: str,
    horizon_ns: int,
    outcome_kind: str,
) -> dict[str, Any]:
    assembled_at = _utc_now().astimezone(timezone.utc)
    assembled_ns = _datetime_to_nanoseconds(assembled_at)
    if horizon_ns > assembled_ns:
        raise AiInventoryEvaluationIntegrityError("trusted horizon has not ended")

    decisions = _read_validate_decision_ledger(room_root / DECISION_LEDGER_NAME)
    matching_decisions = [
        row for row in decisions if row.get("decision_sha256") == decision_sha256
    ]
    if len(matching_decisions) != 1:
        raise AiInventoryEvaluationIntegrityError(
            "exact V2 decision row is missing or duplicated"
        )
    decision = matching_decisions[0]
    session = _require_mapping_field(decision, "session_binding")
    if (
        session.get("room_id") != room_root.name
        or session.get("experiment_id") != room_root.parent.name
    ):
        raise AiInventoryEvaluationIntegrityError(
            "decision scope does not match the canonical room root"
        )
    cutoff_ns = _require_timestamp_ns(decision.get("cutoff_at_utc"), "decision cutoff")
    if cutoff_ns >= horizon_ns:
        raise AiInventoryEvaluationIntegrityError(
            "decision cutoff must precede the completed horizon"
        )

    ai_binding = _require_mapping_field(decision, "ai_decision_binding")
    producer_sha = ai_binding.get("producer_receipt_sha256")
    if not _is_sha256(producer_sha):
        raise AiInventoryEvaluationIntegrityError(
            "decision lacks a valid producer receipt binding"
        )
    receipt_path = room_root / "producer_receipts" / f"{producer_sha}.json"
    try:
        producer = verify_ai_inventory_producer_receipt(room_root, receipt_path)
    except Exception as exc:
        raise AiInventoryEvaluationIntegrityError(
            "producer receipt verification failed"
        ) from exc
    _validate_producer_decision_binding(producer, decision)
    declared_assessment = _signed_assessment_from_producer(producer, decision)

    broker_rows = _read_validate_broker_ledger(
        room_root / BROKER_LEDGER_NAME,
        observed_no_later_than_ns=assembled_ns,
    )
    ledger_binding = _require_mapping_field(decision, "ledger_binding")
    pre_cutoff_broker_rows = []
    for broker_row in broker_rows:
        broker_row_ns = _parse_any_utc_nanoseconds(broker_row.get("ts_utc"))
        assert broker_row_ns is not None
        if broker_row_ns <= cutoff_ns:
            pre_cutoff_broker_rows.append(broker_row)
    cutoff_tip = (
        pre_cutoff_broker_rows[-1]["sha"]
        if pre_cutoff_broker_rows
        else GENESIS_EVALUATION_SHA256
    )
    if cutoff_tip != ledger_binding.get("sha256"):
        raise AiInventoryEvaluationIntegrityError(
            "decision broker-ledger cutoff binding is not the exact prefix tip"
        )
    applied = _find_applied_receipt(
        broker_rows,
        decision=decision,
        producer=producer,
    )

    position = _require_mapping_field(decision, "position_binding")
    if position.get("side") not in {"LONG", "SHORT"}:
        return _build_flat_entry_gate_outcome_packet(
            room_root=room_root,
            decision=decision,
            producer=producer,
            applied=applied,
            position=position,
            declared_assessment=declared_assessment,
            horizon_end_at_utc=horizon_end_at_utc,
            horizon_ns=horizon_ns,
            cutoff_ns=cutoff_ns,
            assembled_ns=assembled_ns,
            outcome_kind=outcome_kind,
        )
    fill = _find_position_fill(
        broker_rows,
        position=position,
        cutoff_ns=cutoff_ns,
    )
    entry_price = _fill_entry_price(fill)
    position_units = _required_positive_number(position.get("units"), "position units")

    quote_rows = _read_validate_quote_ledger(
        room_root / QUOTE_WATERMARK_LEDGER_NAME,
        observed_no_later_than_ns=assembled_ns,
    )
    _validate_fill_entry_cost(fill, position=position, quote_rows=quote_rows)
    pair = str(position["pair"])
    decision_quote = _require_mapping_field(decision, "quote_binding")
    _validate_decision_quote_source(
        decision_quote,
        quote_rows=quote_rows,
    )
    interval_quotes = [
        row
        for row in quote_rows
        if row["pair"] == pair
        and cutoff_ns
        <= _require_timestamp_ns(row["timestamp_utc"], "quote timestamp")
        <= horizon_ns
    ]
    if not interval_quotes:
        raise AiInventoryEvaluationIntegrityError(
            "quote watermark has no bounded outcome interval"
        )
    endpoint = interval_quotes[-1]
    if _require_timestamp_ns(endpoint["timestamp_utc"], "endpoint quote") != horizon_ns:
        raise AiInventoryEvaluationIntegrityError(
            "horizon lacks an exact executable quote watermark"
        )

    settlement_reason: str | None
    settlement_row: dict[str, Any] | None
    if outcome_kind == "SETTLEMENT":
        settlement_row = _find_exact_settlement(
            broker_rows,
            position_id=str(position["position_id"]),
            horizon_ns=horizon_ns,
            applied=applied,
        )
        _validate_settlement_source(
            settlement_row,
            endpoint=endpoint,
            position=position,
        )
        evaluation_units = _settlement_evaluation_units(
            decision=decision,
            position=position,
            settlement=settlement_row,
        )
        settlement_reason = _settlement_reason(settlement_row["event"])
    else:
        settlement_row = None
        evaluation_units = position_units
        _reject_pre_horizon_settlement(
            broker_rows,
            position_id=str(position["position_id"]),
            cutoff_ns=cutoff_ns,
            horizon_ns=horizon_ns,
        )
        settlement_reason = None

    _validate_quote_cost_consistency(quote_rows, pair=pair)
    review_pl = _executable_pl_jpy(
        pair=pair,
        side=str(position["side"]),
        units=evaluation_units,
        entry_price=entry_price,
        bid=_required_positive_number(decision_quote.get("bid"), "review bid"),
        ask=_required_positive_number(decision_quote.get("ask"), "review ask"),
        timestamp_utc=str(decision_quote["observed_at_utc"]),
        quote_rows=quote_rows,
    )
    path_pl = [
        _executable_pl_jpy(
            pair=pair,
            side=str(position["side"]),
            units=evaluation_units,
            entry_price=entry_price,
            bid=_required_positive_number(row.get("bid"), "path bid"),
            ask=_required_positive_number(row.get("ask"), "path ask"),
            timestamp_utc=str(row["timestamp_utc"]),
            quote_rows=quote_rows,
        )
        for row in interval_quotes
    ]
    mfe = max(0.0, *path_pl)
    mae = min(0.0, *path_pl)
    if settlement_row is not None:
        actual_pl = _recompute_settlement_pl_jpy(
            fill=fill,
            settlement=settlement_row,
            position=position,
            units=evaluation_units,
            endpoint=endpoint,
            quote_rows=quote_rows,
        )
    else:
        actual_pl = path_pl[-1]

    relevant_broker_row = _later_broker_row(applied, settlement_row)
    relevant_broker_ns = _parse_any_utc_nanoseconds(relevant_broker_row["ts_utc"])
    if relevant_broker_ns is None:
        raise AiInventoryEvaluationIntegrityError(
            "broker evidence timestamp is invalid"
        )
    observed_ns = max(horizon_ns, relevant_broker_ns)
    if observed_ns > assembled_ns:
        raise AiInventoryEvaluationIntegrityError(
            "trusted outcome evidence is future-dated"
        )

    outcome = "WIN" if actual_pl > 0 else "LOSS" if actual_pl < 0 else "FLAT"
    realized_regime = _realized_regime(interval_quotes, position["side"])
    declared_regime = str(declared_assessment["declared_regime"])
    regime_correct = declared_regime == realized_regime
    confidence = _required_unit_interval(
        producer.get("confidence"), "producer confidence"
    )
    brier = (confidence - (1.0 if regime_correct else 0.0)) ** 2
    signal_identity = _signal_identity_from_decision(decision, position)
    position_identity = {
        "position_id": position["position_id"],
        "pair": position["pair"],
        "side": position["side"],
        "strategy_tag": position["strategy_tag"],
        "entry_context_sha256": position["entry_context_sha256"],
    }
    source_watermarks = sorted(
        [
            {
                "source_id": "broker:ledger",
                "sha256": relevant_broker_row["sha"],
                "watermark_at_utc": _canonical_utc(relevant_broker_ns),
            },
            {
                "source_id": "decision:ledger",
                "sha256": decision["decision_sha256"],
                "watermark_at_utc": decision["recorded_at_utc"],
            },
            {
                "source_id": "producer:receipt",
                "sha256": producer["receipt_sha256"],
                "watermark_at_utc": producer["produced_at_utc"],
            },
            {
                "source_id": "quote:watermark",
                "sha256": endpoint["quote_sha256"],
                "watermark_at_utc": endpoint["timestamp_utc"],
            },
        ],
        key=lambda row: row["source_id"],
    )
    packet = {
        "contract": DOJO_AI_INVENTORY_OUTCOME_CONTRACT,
        "decision_contract": REQUIRED_DECISION_CONTRACT,
        "producer_receipt_contract": REQUIRED_PRODUCER_RECEIPT_CONTRACT,
        "applied_receipt_event": REQUIRED_APPLIED_RECEIPT_EVENT,
        "decision_sha256": decision["decision_sha256"],
        "producer_receipt_sha256": producer["receipt_sha256"],
        "applied_receipt_sha256": applied["sha"],
        "position_identity": position_identity,
        "signal_identity": signal_identity,
        "decision_cutoff_at_utc": decision["cutoff_at_utc"],
        "horizon_end_at_utc": horizon_end_at_utc,
        "outcome_observed_at_utc": _canonical_utc(observed_ns),
        "outcome_kind": outcome_kind,
        "realized_outcome": outcome,
        "settlement_reason": settlement_reason,
        "realized_pl_jpy": _normalized_number(actual_pl),
        "mfe_jpy": _normalized_number(mfe),
        "mae_jpy": _normalized_number(mae),
        "review_time_executable_exit_pl_jpy": _normalized_number(review_pl),
        "actual_exit_pl_jpy": _normalized_number(actual_pl),
        "counterfactual_delta_jpy": _normalized_number(review_pl - actual_pl),
        "declared_assessment": declared_assessment,
        "declared_assessment_sha256": _sha256(
            _canonical_json(declared_assessment).encode("utf-8")
        ),
        "declared_regime": declared_regime,
        "realized_regime": realized_regime,
        "regime_correct": regime_correct,
        "regime_confidence": _normalized_number(confidence),
        "regime_brier_score": _normalized_number(brier),
        "source_watermarks": source_watermarks,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "external_broker_mutation_allowed": False,
        "evaluation_is_not_action": True,
    }
    try:
        return _seal_outcome_packet(packet)
    except (TypeError, ValueError) as exc:
        raise AiInventoryEvaluationIntegrityError(
            "recomputed trusted outcome packet is invalid"
        ) from exc


def _build_flat_entry_gate_outcome_packet(
    *,
    room_root: Path,
    decision: Mapping[str, Any],
    producer: Mapping[str, Any],
    applied: Mapping[str, Any],
    position: Mapping[str, Any],
    declared_assessment: Mapping[str, Any],
    horizon_end_at_utc: str,
    horizon_ns: int,
    cutoff_ns: int,
    assembled_ns: int,
    outcome_kind: str,
) -> dict[str, Any]:
    if position.get("side") != "FLAT" or outcome_kind != "FIXED_HORIZON":
        raise AiInventoryEvaluationIntegrityError(
            "flat entry-gate decisions require a fixed-horizon evaluation"
        )
    action = decision.get("action")
    if action not in {"ALLOW_NEW_VIRTUAL", "BLOCK_NEW"}:
        raise AiInventoryEvaluationIntegrityError(
            "flat decision is not an ALLOW/BLOCK entry gate"
        )
    signal = _signed_entry_signal(room_root, decision=decision, producer=producer)
    if (
        signal.get("pair") != position.get("pair")
        or signal.get("strategy_tag") != position.get("strategy_tag")
        or signal.get("entry_context_sha256")
        != position.get("entry_context_sha256")
    ):
        raise AiInventoryEvaluationIntegrityError(
            "signed entry signal does not match the flat position scope"
        )
    quote_rows = _read_validate_quote_ledger(
        room_root / QUOTE_WATERMARK_LEDGER_NAME,
        observed_no_later_than_ns=assembled_ns,
    )
    pair = str(signal["pair"])
    _validate_quote_cost_consistency(quote_rows, pair=pair)
    decision_quote = _require_mapping_field(decision, "quote_binding")
    _validate_decision_quote_source(decision_quote, quote_rows=quote_rows)
    interval_quotes = [
        row
        for row in quote_rows
        if row["pair"] == pair
        and cutoff_ns
        <= _require_timestamp_ns(row["timestamp_utc"], "quote timestamp")
        <= horizon_ns
    ]
    if not interval_quotes:
        raise AiInventoryEvaluationIntegrityError(
            "entry-gate horizon has no bounded quote watermark"
        )
    endpoint = interval_quotes[-1]
    if _require_timestamp_ns(endpoint["timestamp_utc"], "endpoint quote") != horizon_ns:
        raise AiInventoryEvaluationIntegrityError(
            "entry-gate horizon lacks an exact executable quote watermark"
        )
    entry = _counterfactual_signal_entry(signal, interval_quotes)
    if entry is None:
        signal_path = [0.0]
        review_pl = 0.0
        signal_pl = 0.0
    else:
        entry_price, entry_index = entry
        units = _required_positive_number(signal.get("units"), "signal units")
        signal_path = [
            _executable_pl_jpy(
                pair=pair,
                side=str(signal["side"]),
                units=units,
                entry_price=entry_price,
                bid=_required_positive_number(row.get("bid"), "path bid"),
                ask=_required_positive_number(row.get("ask"), "path ask"),
                timestamp_utc=str(row["timestamp_utc"]),
                quote_rows=quote_rows,
            )
            for row in interval_quotes[entry_index:]
        ]
        review_pl = signal_path[0]
        signal_pl = signal_path[-1]
    decision_value_path = (
        signal_path if action == "ALLOW_NEW_VIRTUAL" else [-value for value in signal_path]
    )
    decision_value = (
        signal_pl if action == "ALLOW_NEW_VIRTUAL" else -signal_pl
    )
    actual_pl = signal_pl if action == "ALLOW_NEW_VIRTUAL" else 0.0
    outcome = (
        "WIN" if decision_value > 0 else "LOSS" if decision_value < 0 else "FLAT"
    )
    realized_regime = _realized_regime(interval_quotes, signal["side"])
    declared_regime = str(declared_assessment["declared_regime"])
    regime_correct = declared_regime == realized_regime
    confidence = _required_unit_interval(
        producer.get("confidence"), "producer confidence"
    )
    brier = (confidence - (1.0 if regime_correct else 0.0)) ** 2
    applied_ns = _parse_any_utc_nanoseconds(applied.get("ts_utc"))
    if applied_ns is None or applied_ns > assembled_ns:
        raise AiInventoryEvaluationIntegrityError(
            "entry-gate applied receipt timestamp is invalid"
        )
    observed_ns = max(horizon_ns, applied_ns)
    assessment = _validate_signed_assessment(declared_assessment)
    packet = {
        "contract": DOJO_AI_INVENTORY_OUTCOME_CONTRACT,
        "decision_contract": REQUIRED_DECISION_CONTRACT,
        "producer_receipt_contract": REQUIRED_PRODUCER_RECEIPT_CONTRACT,
        "applied_receipt_event": REQUIRED_APPLIED_RECEIPT_EVENT,
        "decision_sha256": decision["decision_sha256"],
        "producer_receipt_sha256": producer["receipt_sha256"],
        "applied_receipt_sha256": applied["sha"],
        "position_identity": None,
        "signal_identity": {
            "signal_identity_sha256": signal["signal_identity_sha256"],
            "pair": signal["pair"],
            "side": signal["side"],
            "strategy_tag": signal["strategy_tag"],
            "entry_context_sha256": signal["entry_context_sha256"],
        },
        "decision_cutoff_at_utc": decision["cutoff_at_utc"],
        "horizon_end_at_utc": horizon_end_at_utc,
        "outcome_observed_at_utc": _canonical_utc(observed_ns),
        "outcome_kind": "FIXED_HORIZON",
        "realized_outcome": outcome,
        "settlement_reason": None,
        "realized_pl_jpy": _normalized_number(decision_value),
        "mfe_jpy": _normalized_number(max(0.0, *decision_value_path)),
        "mae_jpy": _normalized_number(min(0.0, *decision_value_path)),
        "review_time_executable_exit_pl_jpy": _normalized_number(review_pl),
        "actual_exit_pl_jpy": _normalized_number(actual_pl),
        "counterfactual_delta_jpy": _normalized_number(review_pl - actual_pl),
        "declared_assessment": assessment,
        "declared_assessment_sha256": _sha256(
            _canonical_json(assessment).encode("utf-8")
        ),
        "declared_regime": declared_regime,
        "realized_regime": realized_regime,
        "regime_correct": regime_correct,
        "regime_confidence": _normalized_number(confidence),
        "regime_brier_score": _normalized_number(brier),
        "source_watermarks": sorted(
            [
                {
                    "source_id": "broker:ledger",
                    "sha256": applied["sha"],
                    "watermark_at_utc": applied["ts_utc"],
                },
                {
                    "source_id": "decision:ledger",
                    "sha256": decision["decision_sha256"],
                    "watermark_at_utc": decision["recorded_at_utc"],
                },
                {
                    "source_id": "producer:receipt",
                    "sha256": producer["receipt_sha256"],
                    "watermark_at_utc": producer["produced_at_utc"],
                },
                {
                    "source_id": "quote:watermark",
                    "sha256": endpoint["quote_sha256"],
                    "watermark_at_utc": endpoint["timestamp_utc"],
                },
            ],
            key=lambda row: row["source_id"],
        ),
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "external_broker_mutation_allowed": False,
        "evaluation_is_not_action": True,
    }
    try:
        return _seal_outcome_packet(packet)
    except (TypeError, ValueError) as exc:
        raise AiInventoryEvaluationIntegrityError(
            "recomputed entry-gate outcome packet is invalid"
        ) from exc


def _signed_entry_signal(
    room_root: Path,
    *,
    decision: Mapping[str, Any],
    producer: Mapping[str, Any],
) -> dict[str, Any]:
    admission = decision.get("admission_binding")
    if isinstance(admission, Mapping):
        signal = admission.get("entry_signal")
    else:
        evidence_sha = producer.get("evidence_packet_sha256")
        if not _is_sha256(evidence_sha):
            raise AiInventoryEvaluationIntegrityError(
                "producer lacks an evidence packet for the blocked signal"
            )
        packet_path = (
            _trusted_repository_root()
            / DEDICATED_EVIDENCE_ROOT
            / f"{evidence_sha}.json"
        )
        try:
            packet = verify_ai_inventory_evidence_packet(
                _trusted_repository_root(), packet_path
            )
        except Exception as exc:
            raise AiInventoryEvaluationIntegrityError(
                "blocked entry signal evidence packet verification failed"
            ) from exc
        bindings = packet.get("bindings")
        if (
            not isinstance(bindings, Mapping)
            or bindings.get("experiment_id") != room_root.parent.name
            or bindings.get("room_id") != room_root.name
        ):
            raise AiInventoryEvaluationIntegrityError(
                "blocked entry signal evidence scope mismatch"
            )
        signal = packet.get("entry_signal")
    if not isinstance(signal, Mapping):
        raise AiInventoryEvaluationIntegrityError(
            "entry-gate decision lacks its signed entry signal"
        )
    normalized = _snapshot_mapping(signal, "signed entry signal")
    if producer.get("entry_signal_identity_sha256") != normalized.get(
        "signal_identity_sha256"
    ):
        raise AiInventoryEvaluationIntegrityError(
            "producer receipt does not bind the entry signal identity"
        )
    return normalized


def _counterfactual_signal_entry(
    signal: Mapping[str, Any],
    quotes: Sequence[Mapping[str, Any]],
) -> tuple[float, int] | None:
    side = signal.get("side")
    order_type = signal.get("order_type")
    pair = str(signal.get("pair"))
    if side not in {"LONG", "SHORT"} or order_type not in {"MARKET", "LIMIT", "STOP"}:
        raise AiInventoryEvaluationIntegrityError(
            "entry signal geometry is invalid"
        )
    slippage, _ = _validate_quote_cost_consistency(quotes, pair=pair)
    slip = slippage * _pip_size(pair)
    level = signal.get("price")
    for index, row in enumerate(quotes):
        bid = _required_positive_number(row.get("bid"), "signal bid")
        ask = _required_positive_number(row.get("ask"), "signal ask")
        base: float | None = None
        if order_type == "MARKET":
            base = ask if side == "LONG" else bid
        elif order_type == "LIMIT":
            price = _required_positive_number(level, "limit signal price")
            if side == "LONG" and ask <= price:
                base = min(price, ask)
            elif side == "SHORT" and bid >= price:
                base = max(price, bid)
        else:
            price = _required_positive_number(level, "stop signal price")
            if side == "LONG" and ask >= price:
                base = max(price, ask)
            elif side == "SHORT" and bid <= price:
                base = min(price, bid)
        if base is not None:
            precision = 3 if pair.endswith("_JPY") else 5
            entry = round(base + slip if side == "LONG" else base - slip, precision)
            return entry, index
    return None


def _require_canonical_room_root(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise AiInventoryEvaluationIntegrityError("room_root must be an absolute Path")
    repository_root = _trusted_repository_root()
    try:
        rooms_root = canonical_paper_ai_rooms_root(repository_root).resolve(strict=True)
    except OSError as exc:
        raise AiInventoryEvaluationIntegrityError(
            "canonical paper-AI rooms root is unavailable"
        ) from exc
    try:
        room_info = value.lstat()
        parent_info = value.parent.lstat()
    except OSError as exc:
        raise AiInventoryEvaluationIntegrityError(
            "canonical room root is unavailable"
        ) from exc
    for label, info in (
        ("room", room_info),
        ("experiment", parent_info),
    ):
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise AiInventoryEvaluationIntegrityError(
                f"canonical {label} directory is unsafe"
            )
    root = value.resolve(strict=True)
    try:
        relative = root.relative_to(rooms_root)
    except ValueError as exc:
        raise AiInventoryEvaluationIntegrityError(
            "room_root is outside the canonical paper-AI rooms root"
        ) from exc
    if (
        root != value
        or len(relative.parts) != 2
        or any(
            not part.startswith("paper-ai-inventory-") for part in relative.parts
        )
    ):
        raise AiInventoryEvaluationIntegrityError(
            "room_root is not a canonical paper-ai-inventory room"
        )
    return root


def _trusted_repository_root() -> Path:
    try:
        return Path(__file__).resolve(strict=True).parents[2].resolve(strict=True)
    except (IndexError, OSError) as exc:
        raise AiInventoryEvaluationIntegrityError(
            "package-derived repository root is unavailable"
        ) from exc


def _read_validate_decision_ledger(path: Path) -> list[dict[str, Any]]:
    rows = _read_strict_jsonl(path)
    previous = GENESIS_EVALUATION_SHA256
    for index, row in enumerate(rows, start=1):
        issues = validate_inventory_decision(row)
        if issues:
            raise AiInventoryEvaluationIntegrityError(
                f"decision ledger row {index} failed V2 validation: "
                + "; ".join(issues)
            )
        if (
            row.get("sequence") != index
            or row.get("previous_decision_sha256") != previous
        ):
            raise AiInventoryEvaluationIntegrityError(
                f"decision ledger chain mismatch at row {index}"
            )
        previous = row["decision_sha256"]
    if not rows:
        raise AiInventoryEvaluationIntegrityError("decision ledger is empty")
    return rows


def _read_validate_broker_ledger(
    path: Path, *, observed_no_later_than_ns: int
) -> list[dict[str, Any]]:
    rows = _read_strict_jsonl(path)
    previous = GENESIS_EVALUATION_SHA256
    previous_ts: int | None = None
    for index, row in enumerate(rows, start=1):
        if set(row) != _BROKER_ROW_KEYS:
            raise AiInventoryEvaluationIntegrityError(
                f"broker ledger schema mismatch at row {index}"
            )
        if (
            row.get("prev_sha") != previous
            or not _is_sha256(row.get("sha"))
            or not isinstance(row.get("payload"), dict)
            or not isinstance(row.get("event"), str)
        ):
            raise AiInventoryEvaluationIntegrityError(
                f"broker ledger chain fields are invalid at row {index}"
            )
        body = {key: row[key] for key in ("ts_utc", "event", "payload", "prev_sha")}
        if _sha256(_canonical_json(body).encode("utf-8")) != row["sha"]:
            raise AiInventoryEvaluationIntegrityError(
                f"broker ledger digest mismatch at row {index}"
            )
        timestamp = _parse_any_utc_nanoseconds(row.get("ts_utc"))
        if (
            timestamp is None
            or timestamp > observed_no_later_than_ns
            or (previous_ts is not None and timestamp < previous_ts)
        ):
            raise AiInventoryEvaluationIntegrityError(
                f"broker ledger timestamp is invalid at row {index}"
            )
        previous_ts = timestamp
        previous = row["sha"]
    if not rows:
        raise AiInventoryEvaluationIntegrityError("broker ledger is empty")
    return rows


def _read_validate_quote_ledger(
    path: Path, *, observed_no_later_than_ns: int
) -> list[dict[str, Any]]:
    rows = _read_strict_jsonl(path)
    previous = GENESIS_EVALUATION_SHA256
    previous_ts: int | None = None
    for index, row in enumerate(rows, start=1):
        if set(row) != _QUOTE_ROW_KEYS or row.get("sequence") != index:
            raise AiInventoryEvaluationIntegrityError(
                f"quote watermark schema/sequence mismatch at row {index}"
            )
        timestamp = _require_timestamp_ns(
            row.get("timestamp_utc"), f"quote watermark row {index}"
        )
        recorded = _require_timestamp_ns(
            row.get("recorded_at_utc"), f"quote watermark recorded row {index}"
        )
        bid = _required_positive_number(row.get("bid"), "quote bid")
        ask = _required_positive_number(row.get("ask"), "quote ask")
        _required_nonnegative_number(
            row.get("slippage_pips_per_fill"), "quote slippage cost"
        )
        _required_nonnegative_number(
            row.get("financing_pips_per_day"), "quote financing cost"
        )
        if (
            row.get("contract") != "QR_DOJO_AI_INVENTORY_QUOTE_WATERMARK_V1"
            or ask < bid
            or not _is_pair(row.get("pair"))
            or not _is_sha256(row.get("source_sha256"))
            or not _is_sha256(row.get("capture_source_sha256"))
            or row.get("capture_source_sha256")
            == GENESIS_EVALUATION_SHA256
            or not _is_sha256(row.get("acquisition_receipt_sha256"))
            or row.get("acquisition_receipt_sha256")
            == GENESIS_EVALUATION_SHA256
            or row.get("previous_quote_sha256") != previous
            or row.get("quote_sha256") != quote_watermark_sha256(row)
            or row.get("paper_only") is not True
            or row.get("order_authority") != "NONE"
            or row.get("live_permission") is not False
            or recorded < timestamp
            or timestamp > observed_no_later_than_ns
            or (previous_ts is not None and timestamp < previous_ts)
        ):
            raise AiInventoryEvaluationIntegrityError(
                f"quote watermark integrity mismatch at row {index}"
            )
        source_path = (
            path.parent / "quote_sources" / f"{row['source_sha256']}.json"
        )
        source_raw = _read_regular_nofollow(source_path, MAX_LEDGER_LINE_BYTES)
        if _sha256(source_raw) != row["source_sha256"]:
            raise AiInventoryEvaluationIntegrityError(
                f"quote source digest mismatch at row {index}"
            )
        try:
            source = json.loads(
                source_raw,
                object_pairs_hook=_strict_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise AiInventoryEvaluationIntegrityError(
                f"quote source JSON mismatch at row {index}"
            ) from exc
        if source != {
            "contract": "QR_DOJO_AI_INVENTORY_QUOTE_SOURCE_V1",
            "timestamp_utc": row["timestamp_utc"],
            "pair": row["pair"],
            "bid": row["bid"],
            "ask": row["ask"],
            "capture_source_sha256": row["capture_source_sha256"],
            "acquisition_receipt_sha256": row[
                "acquisition_receipt_sha256"
            ],
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }:
            raise AiInventoryEvaluationIntegrityError(
                f"quote source content mismatch at row {index}"
            )
        previous = row["quote_sha256"]
        previous_ts = timestamp
    if not rows:
        raise AiInventoryEvaluationIntegrityError("quote watermark ledger is empty")
    return rows


def _read_strict_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = _read_regular_nofollow(path, MAX_SOURCE_LEDGER_BYTES)
    if not raw or not raw.endswith(b"\n"):
        raise AiInventoryEvaluationIntegrityError(f"{path.name} is empty or truncated")
    lines = raw.splitlines()
    if len(lines) > MAX_LEDGER_ROWS:
        raise AiInventoryEvaluationIntegrityError(f"{path.name} exceeds the row limit")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line or len(line) > MAX_LEDGER_LINE_BYTES:
            raise AiInventoryEvaluationIntegrityError(
                f"{path.name} row {index} has invalid size"
            )
        try:
            row = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_strict_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise AiInventoryEvaluationIntegrityError(
                f"{path.name} row {index} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise AiInventoryEvaluationIntegrityError(
                f"{path.name} row {index} is not an object"
            )
        rows.append(row)
    return rows


def _read_regular_nofollow(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AiInventoryEvaluationIntegrityError(
            f"canonical source {path.name} is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise AiInventoryEvaluationIntegrityError(
                f"canonical source {path.name} is not a bounded regular file"
            )
        parts: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            parts.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise AiInventoryEvaluationIntegrityError(
                f"canonical source {path.name} changed while reading"
            )
    finally:
        os.close(descriptor)
    raw = b"".join(parts)
    if len(raw) > maximum_bytes or len(raw) != before.st_size:
        raise AiInventoryEvaluationIntegrityError(
            f"canonical source {path.name} changed or exceeds its bound"
        )
    return raw


def _validate_producer_decision_binding(
    producer: Mapping[str, Any], decision: Mapping[str, Any]
) -> None:
    ai_binding = _require_mapping_field(decision, "ai_decision_binding")
    exact = (
        ("producer_id", "producer_id"),
        ("model_id", "model_id"),
        ("evidence_packet_sha256", "evidence_packet_sha256"),
        ("request_sha256", "request_sha256"),
        ("response_sha256", "response_sha256"),
        ("producer_receipt_sha256", "receipt_sha256"),
        ("produced_at_utc", "produced_at_utc"),
    )
    for decision_field, receipt_field in exact:
        if ai_binding.get(decision_field) != producer.get(receipt_field):
            raise AiInventoryEvaluationIntegrityError(
                f"producer/decision mismatch: {decision_field}"
            )
    for field in ("action", "reason_code", "reason", "virtual_units", "confidence"):
        if decision.get(field) != producer.get(field):
            raise AiInventoryEvaluationIntegrityError(
                f"producer/decision semantic mismatch: {field}"
            )


def _signed_assessment_from_producer(
    producer: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, Any]:
    reason = producer.get("reason")
    if reason != decision.get("reason") or not isinstance(reason, str):
        raise AiInventoryEvaluationIntegrityError(
            "signed assessment is not bound identically to producer and decision"
        )
    try:
        value = json.loads(
            reason,
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AiInventoryEvaluationIntegrityError(
            "signed producer reason is not a prospective assessment"
        ) from exc
    assessment = _validate_signed_assessment(value)
    if reason != _canonical_json(assessment):
        raise AiInventoryEvaluationIntegrityError(
            "signed assessment is not canonical JSON"
        )
    return assessment


def _validate_signed_assessment(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AiInventoryEvaluationIntegrityError(
            "prospective assessment is not an object"
        )
    assessment = _snapshot_mapping(value, "prospective assessment")
    if set(assessment) != _SIGNED_ASSESSMENT_KEYS:
        raise AiInventoryEvaluationIntegrityError(
            "prospective assessment schema is invalid"
        )
    if assessment.get("contract") != SIGNED_ASSESSMENT_CONTRACT:
        raise AiInventoryEvaluationIntegrityError(
            "prospective assessment contract is invalid"
        )
    if assessment.get("declared_regime") not in ALLOWED_REGIMES:
        raise AiInventoryEvaluationIntegrityError(
            "prospective assessment regime is invalid"
        )
    for field in ("assessment", "primary_path", "alternative_path", "falsifier"):
        text = assessment.get(field)
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > 1_000
            or "\x00" in text
        ):
            raise AiInventoryEvaluationIntegrityError(
                f"prospective assessment {field} is invalid"
            )
    return assessment


def _find_applied_receipt(
    rows: Sequence[dict[str, Any]],
    *,
    decision: Mapping[str, Any],
    producer: Mapping[str, Any],
) -> dict[str, Any]:
    decision_sha = decision["decision_sha256"]
    applied = [
        row
        for row in rows
        if row.get("event") == REQUIRED_APPLIED_RECEIPT_EVENT
        and isinstance(row.get("payload"), Mapping)
        and row["payload"].get("decision_sha256") == decision_sha
    ]
    if len(applied) != 1:
        raise AiInventoryEvaluationIntegrityError(
            "decision lacks one exact APPLIED receipt"
        )
    row = applied[0]
    payload = _require_mapping_field(row, "payload")
    reservation_sha = payload.get("reservation_sha256")
    reservations = [
        candidate
        for candidate in rows
        if candidate.get("event") == "AI_INVENTORY_ACTION_RESERVED"
        and candidate.get("sha") == reservation_sha
    ]
    if len(reservations) != 1:
        raise AiInventoryEvaluationIntegrityError(
            "APPLIED receipt lacks its exact reservation"
        )
    reservation = _require_mapping_field(reservations[0], "payload")
    bindings = {
        "decision_sha256": decision_sha,
        "action": decision["action"],
        "confidence": decision["confidence"],
        "room_id": _require_mapping_field(decision, "session_binding")["room_id"],
        "position_id": _require_mapping_field(decision, "position_binding")[
            "position_id"
        ],
        "pair": _require_mapping_field(decision, "position_binding")["pair"],
        "strategy_tag": _require_mapping_field(decision, "position_binding")[
            "strategy_tag"
        ],
        "ai_producer_receipt_sha256": producer["receipt_sha256"],
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "external_broker_mutation_allowed": False,
    }
    for field, expected in bindings.items():
        if payload.get(field) != expected or reservation.get(field) != expected:
            raise AiInventoryEvaluationIntegrityError(
                f"APPLIED/reservation binding mismatch: {field}"
            )
    if payload.get("status") != "APPLIED":
        raise AiInventoryEvaluationIntegrityError("APPLIED receipt status is invalid")
    return row


def _find_position_fill(
    rows: Sequence[dict[str, Any]],
    *,
    position: Mapping[str, Any],
    cutoff_ns: int,
) -> dict[str, Any]:
    fills = []
    for row in rows:
        if row.get("event") not in {"FILL_MARKET", "FILL_LIMIT"}:
            continue
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if payload.get("trade_id") != position.get("position_id"):
            continue
        quote = payload.get("quote")
        if (
            not isinstance(quote, Mapping)
            or _require_timestamp_ns(quote.get("ts"), "position fill quote") > cutoff_ns
        ):
            raise AiInventoryEvaluationIntegrityError(
                "position fill is not before the decision cutoff"
            )
        fills.append(row)
    if len(fills) != 1:
        raise AiInventoryEvaluationIntegrityError(
            "position fill is missing or duplicated"
        )
    payload = _require_mapping_field(fills[0], "payload")
    exact = (
        ("pair", position.get("pair")),
        ("side", position.get("side")),
        ("units", position.get("units")),
        ("strategy_tag", position.get("strategy_tag")),
        ("entry_context_sha256", position.get("entry_context_sha256")),
    )
    for field, expected in exact:
        if payload.get(field) != expected:
            raise AiInventoryEvaluationIntegrityError(
                f"position fill mismatch: {field}"
            )
    return fills[0]


def _fill_entry_price(row: Mapping[str, Any]) -> float:
    payload = _require_mapping_field(row, "payload")
    raw = (
        payload.get("entry")
        if row.get("event") == "FILL_MARKET"
        else payload.get("price")
    )
    return _required_positive_number(raw, "position entry price")


def _validate_fill_entry_cost(
    fill: Mapping[str, Any],
    *,
    position: Mapping[str, Any],
    quote_rows: Sequence[Mapping[str, Any]],
) -> None:
    payload = _require_mapping_field(fill, "payload")
    quote = _require_mapping_field(payload, "quote")
    pair = str(position["pair"])
    matches = [
        row
        for row in quote_rows
        if row.get("pair") == pair
        and row.get("timestamp_utc") == quote.get("ts")
        and row.get("bid") == quote.get("bid")
        and row.get("ask") == quote.get("ask")
    ]
    if len(matches) != 1:
        raise AiInventoryEvaluationIntegrityError(
            "position fill quote lacks one immutable watermark"
        )
    slippage, _ = _validate_quote_cost_consistency(quote_rows, pair=pair)
    side = str(position["side"])
    executable = _required_positive_number(
        quote.get("ask") if side == "LONG" else quote.get("bid"),
        "fill executable quote",
    )
    expected = executable + slippage * _pip_size(pair)
    if side == "SHORT":
        expected = executable - slippage * _pip_size(pair)
    expected = round(expected, 3 if pair.endswith("_JPY") else 5)
    if _fill_entry_price(fill) != expected:
        raise AiInventoryEvaluationIntegrityError(
            "position entry does not match executable quote plus bound slippage"
        )


def _validate_decision_quote_source(
    decision_quote: Mapping[str, Any],
    *,
    quote_rows: Sequence[dict[str, Any]],
) -> None:
    matches = [
        row
        for row in quote_rows
        if row.get("pair") == decision_quote.get("pair")
        and row.get("bid") == decision_quote.get("bid")
        and row.get("ask") == decision_quote.get("ask")
        and row.get("timestamp_utc") == decision_quote.get("observed_at_utc")
        and row.get("source_sha256") == decision_quote.get("sha256")
    ]
    if len(matches) != 1:
        raise AiInventoryEvaluationIntegrityError(
            "decision quote is absent or ambiguous in the canonical watermark"
        )


def _find_exact_settlement(
    rows: Sequence[dict[str, Any]],
    *,
    position_id: str,
    horizon_ns: int,
    applied: Mapping[str, Any],
) -> dict[str, Any]:
    events = {"CLOSE", "EXIT_TP", "EXIT_SL", "MARGIN_CLOSEOUT"}
    matches: list[dict[str, Any]] = []
    for row in rows:
        if row.get("event") not in events:
            continue
        payload = row.get("payload")
        if not isinstance(payload, Mapping) or payload.get("trade_id") != position_id:
            continue
        quote = payload.get("quote")
        if not isinstance(quote, Mapping):
            raise AiInventoryEvaluationIntegrityError(
                "settlement lacks an executable quote watermark"
            )
        if _require_timestamp_ns(quote.get("ts"), "settlement quote") == horizon_ns:
            matches.append(row)
    if len(matches) != 1:
        raise AiInventoryEvaluationIntegrityError(
            "exact horizon settlement is missing or ambiguous"
        )
    applied_payload = _require_mapping_field(applied, "payload")
    close_sha = applied_payload.get("close_sha256")
    if close_sha is not None and matches[0].get("sha") != close_sha:
        raise AiInventoryEvaluationIntegrityError(
            "APPLIED receipt close binding does not match settlement"
        )
    return matches[0]


def _validate_settlement_source(
    settlement: Mapping[str, Any],
    *,
    endpoint: Mapping[str, Any],
    position: Mapping[str, Any],
) -> None:
    payload = _require_mapping_field(settlement, "payload")
    quote = _require_mapping_field(payload, "quote")
    exact_quote = (
        ("bid", endpoint.get("bid")),
        ("ask", endpoint.get("ask")),
        ("ts", endpoint.get("timestamp_utc")),
    )
    for field, expected in exact_quote:
        if quote.get(field) != expected:
            raise AiInventoryEvaluationIntegrityError(
                f"settlement/quote-watermark mismatch: {field}"
            )
    for field in ("strategy_tag", "entry_context_sha256"):
        if payload.get(field) != position.get(field):
            raise AiInventoryEvaluationIntegrityError(
                f"settlement/position mismatch: {field}"
            )
    price = _required_positive_number(payload.get("price"), "settlement price")
    bid = _required_positive_number(endpoint.get("bid"), "settlement bid")
    ask = _required_positive_number(endpoint.get("ask"), "settlement ask")
    if position.get("side") == "LONG" and price > bid:
        # A TP limit may fill at its target below a gapped executable bid, but
        # never above that bid.  The inverse applies to a short.
        raise AiInventoryEvaluationIntegrityError(
            "long settlement price exceeds its executable bid"
        )
    if position.get("side") == "SHORT" and price < ask:
        raise AiInventoryEvaluationIntegrityError(
            "short settlement price is below its executable ask"
        )


def _settlement_evaluation_units(
    *,
    decision: Mapping[str, Any],
    position: Mapping[str, Any],
    settlement: Mapping[str, Any],
) -> float:
    position_units = _required_positive_number(position.get("units"), "position units")
    action = decision.get("action")
    payload = _require_mapping_field(settlement, "payload")
    if action == "REDUCE_VIRTUAL":
        if settlement.get("event") != "CLOSE":
            raise AiInventoryEvaluationIntegrityError(
                "REDUCE_VIRTUAL must bind one virtual CLOSE settlement"
            )
        units = _required_positive_number(payload.get("units"), "partial close units")
        requested = _required_positive_number(
            decision.get("virtual_units"), "REDUCE_VIRTUAL units"
        )
        if units != requested or units > position_units:
            raise AiInventoryEvaluationIntegrityError(
                "partial settlement units do not match the signed decision"
            )
        return units
    if action == "CLOSE_VIRTUAL":
        if settlement.get("event") != "CLOSE":
            raise AiInventoryEvaluationIntegrityError(
                "CLOSE_VIRTUAL must bind one virtual CLOSE settlement"
            )
        units = _required_positive_number(payload.get("units"), "close units")
        if units != position_units:
            raise AiInventoryEvaluationIntegrityError(
                "full close settlement units do not match the signed position"
            )
        return units
    if action != "HOLD":
        raise AiInventoryEvaluationIntegrityError(
            "open-position settlement action is not evaluable"
        )
    if settlement.get("event") == "CLOSE":
        units = _required_positive_number(payload.get("units"), "close units")
        if units != position_units:
            raise AiInventoryEvaluationIntegrityError(
                "HOLD close units do not match the signed position"
            )
    return position_units


def _validate_quote_cost_consistency(
    quote_rows: Sequence[Mapping[str, Any]], *, pair: str
) -> tuple[float, float]:
    pair_rows = [row for row in quote_rows if row.get("pair") == pair]
    if not pair_rows:
        raise AiInventoryEvaluationIntegrityError(
            "quote watermark lacks the evaluated pair cost binding"
        )
    costs = {
        (
            _required_nonnegative_number(
                row.get("slippage_pips_per_fill"), "slippage cost"
            ),
            _required_nonnegative_number(
                row.get("financing_pips_per_day"), "financing cost"
            ),
        )
        for row in pair_rows
    }
    if len(costs) != 1:
        raise AiInventoryEvaluationIntegrityError(
            "quote watermark cost binding changed within one room"
        )
    return next(iter(costs))


def _recompute_settlement_pl_jpy(
    *,
    fill: Mapping[str, Any],
    settlement: Mapping[str, Any],
    position: Mapping[str, Any],
    units: float,
    endpoint: Mapping[str, Any],
    quote_rows: Sequence[Mapping[str, Any]],
) -> float:
    payload = _require_mapping_field(settlement, "payload")
    entry = _fill_entry_price(fill)
    exit_price = _required_positive_number(payload.get("price"), "settlement price")
    side = str(position["side"])
    pair = str(position["pair"])
    difference = exit_price - entry if side == "LONG" else entry - exit_price
    conversion = _jpy_per_quote_unit(
        pair,
        timestamp_utc=str(endpoint["timestamp_utc"]),
        quote_rows=quote_rows,
    )
    gross = difference * units * conversion
    _, financing_pips_per_day = _validate_quote_cost_consistency(
        quote_rows, pair=pair
    )
    fill_payload = _require_mapping_field(fill, "payload")
    fill_quote = _require_mapping_field(fill_payload, "quote")
    opened_ns = _require_timestamp_ns(fill_quote.get("ts"), "position opened quote")
    closed_ns = _require_timestamp_ns(
        _require_mapping_field(payload, "quote").get("ts"),
        "position settlement quote",
    )
    if closed_ns < opened_ns:
        raise AiInventoryEvaluationIntegrityError(
            "settlement precedes the position fill"
        )
    held_days = (closed_ns - opened_ns) / (86_400 * 1_000_000_000)
    financing = (
        financing_pips_per_day
        * _pip_size(pair)
        * units
        * conversion
        * held_days
    )
    recomputed = gross - financing
    declared = _required_number(payload.get("pl_jpy"), "settlement P/L")
    if not math.isclose(
        round(recomputed, 2),
        declared,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise AiInventoryEvaluationIntegrityError(
            "settlement P/L does not match entry/exit/units/cost recomputation"
        )
    return declared


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def _reject_pre_horizon_settlement(
    rows: Sequence[dict[str, Any]],
    *,
    position_id: str,
    cutoff_ns: int,
    horizon_ns: int,
) -> None:
    for row in rows:
        if row.get("event") not in {
            "CLOSE",
            "EXIT_TP",
            "EXIT_SL",
            "MARGIN_CLOSEOUT",
        }:
            continue
        payload = row.get("payload")
        if not isinstance(payload, Mapping) or payload.get("trade_id") != position_id:
            continue
        quote = payload.get("quote")
        if not isinstance(quote, Mapping):
            raise AiInventoryEvaluationIntegrityError(
                "settlement lacks a quote timestamp"
            )
        settled_ns = _require_timestamp_ns(quote.get("ts"), "settlement quote")
        if cutoff_ns < settled_ns <= horizon_ns:
            raise AiInventoryEvaluationIntegrityError(
                "fixed horizon follows an already settled position"
            )


def _executable_pl_jpy(
    *,
    pair: str,
    side: str,
    units: float,
    entry_price: float,
    bid: float,
    ask: float,
    timestamp_utc: str,
    quote_rows: Sequence[dict[str, Any]],
) -> float:
    executable = bid if side == "LONG" else ask
    difference = (
        executable - entry_price if side == "LONG" else entry_price - executable
    )
    conversion = _jpy_per_quote_unit(
        pair, timestamp_utc=timestamp_utc, quote_rows=quote_rows
    )
    result = difference * units * conversion
    if not math.isfinite(result):
        raise AiInventoryEvaluationIntegrityError(
            "recomputed executable P/L is non-finite"
        )
    return round(result, 9)


def _jpy_per_quote_unit(
    pair: str,
    *,
    timestamp_utc: str,
    quote_rows: Sequence[dict[str, Any]],
) -> float:
    quote_currency = pair.split("_", 1)[1]
    if quote_currency == "JPY":
        return 1.0
    if quote_currency != "USD":
        raise AiInventoryEvaluationIntegrityError(
            "trusted evaluator lacks a canonical JPY conversion path"
        )
    matches = [
        row
        for row in quote_rows
        if row.get("pair") == "USD_JPY" and row.get("timestamp_utc") == timestamp_utc
    ]
    if len(matches) != 1:
        raise AiInventoryEvaluationIntegrityError(
            "exact USD_JPY conversion watermark is missing or ambiguous"
        )
    bid = _required_positive_number(matches[0].get("bid"), "conversion bid")
    ask = _required_positive_number(matches[0].get("ask"), "conversion ask")
    return (bid + ask) / 2.0


def _realized_regime(quotes: Sequence[Mapping[str, Any]], side: object) -> str:
    del side
    mids = [
        (
            _required_positive_number(row.get("bid"), "regime bid")
            + _required_positive_number(row.get("ask"), "regime ask")
        )
        / 2.0
        for row in quotes
    ]
    if len(mids) < 3:
        return "UNCLEAR"
    spread = max(mids) - min(mids)
    if spread <= 1e-12:
        return "SQUEEZE"
    efficiency = abs(mids[-1] - mids[0]) / spread
    return "TREND" if efficiency >= 0.6 else "RANGE"


def _signal_identity_from_decision(
    decision: Mapping[str, Any],
    position: Mapping[str, Any],
) -> dict[str, Any] | None:
    admission = decision.get("admission_binding")
    if not isinstance(admission, Mapping):
        return None
    signal = admission.get("entry_signal")
    if not isinstance(signal, Mapping):
        return None
    return {
        "signal_identity_sha256": signal["signal_identity_sha256"],
        "pair": signal["pair"],
        "side": signal["side"],
        "strategy_tag": signal["strategy_tag"],
        "entry_context_sha256": signal["entry_context_sha256"],
    }


def _later_broker_row(
    left: Mapping[str, Any],
    right: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if right is None:
        return dict(left)
    left_ns = _parse_any_utc_nanoseconds(left.get("ts_utc"))
    right_ns = _parse_any_utc_nanoseconds(right.get("ts_utc"))
    if left_ns is None or right_ns is None:
        raise AiInventoryEvaluationIntegrityError("broker row timestamp is invalid")
    return dict(right if right_ns > left_ns else left)


def _settlement_reason(event: object) -> str:
    reasons = {
        "CLOSE": "VIRTUAL_CLOSE",
        "EXIT_TP": "TAKE_PROFIT",
        "EXIT_SL": "STOP_LOSS",
        "MARGIN_CLOSEOUT": "MARGIN_CLOSEOUT",
    }
    try:
        return reasons[str(event)]
    except KeyError as exc:
        raise AiInventoryEvaluationIntegrityError(
            "settlement event is unsupported"
        ) from exc


def _require_mapping_field(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    nested = value.get(field)
    if not isinstance(nested, Mapping):
        raise AiInventoryEvaluationIntegrityError(
            f"trusted source field {field} is not an object"
        )
    return nested


def _required_number(value: object, label: str) -> float:
    parsed = _finite_number(value)
    if parsed is None:
        raise AiInventoryEvaluationIntegrityError(f"{label} is not finite")
    return parsed


def _required_positive_number(value: object, label: str) -> float:
    parsed = _required_number(value, label)
    if parsed <= 0:
        raise AiInventoryEvaluationIntegrityError(f"{label} is not positive")
    return parsed


def _required_nonnegative_number(value: object, label: str) -> float:
    parsed = _required_number(value, label)
    if parsed < 0:
        raise AiInventoryEvaluationIntegrityError(f"{label} is negative")
    return parsed


def _required_unit_interval(value: object, label: str) -> float:
    parsed = _required_number(value, label)
    if not 0 <= parsed <= 1:
        raise AiInventoryEvaluationIntegrityError(f"{label} is outside [0,1]")
    return parsed


def _require_timestamp_ns(value: object, label: str) -> int:
    parsed = _parse_utc_nanoseconds(value)
    if parsed is None:
        raise AiInventoryEvaluationIntegrityError(f"{label} is not canonical UTC")
    return parsed


def _seal_outcome_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    packet = _snapshot_mapping(value, "outcome packet")
    issues = _packet_issues(packet)
    if issues:
        raise ValueError("invalid AI inventory outcome packet: " + "; ".join(issues))
    normalized = dict(packet)
    for field in (
        "realized_pl_jpy",
        "mfe_jpy",
        "mae_jpy",
        "review_time_executable_exit_pl_jpy",
        "actual_exit_pl_jpy",
        "counterfactual_delta_jpy",
        "regime_confidence",
        "regime_brier_score",
    ):
        normalized[field] = _normalized_number(float(normalized[field]))
    return normalized


def _packet_issues(packet: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    keys = _exact_string_keys(packet)
    for key in sorted(_OUTCOME_KEYS - keys):
        issues.append(f"MISSING_OUTCOME_FIELD:{key}")
    for key in sorted(keys - _OUTCOME_KEYS):
        issues.append(f"UNKNOWN_OUTCOME_FIELD:{key}")
    if keys != _OUTCOME_KEYS:
        return tuple(issues)

    exact_values = {
        "contract": DOJO_AI_INVENTORY_OUTCOME_CONTRACT,
        "decision_contract": REQUIRED_DECISION_CONTRACT,
        "producer_receipt_contract": REQUIRED_PRODUCER_RECEIPT_CONTRACT,
        "applied_receipt_event": REQUIRED_APPLIED_RECEIPT_EVENT,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "external_broker_mutation_allowed": False,
        "evaluation_is_not_action": True,
    }
    for field, expected in exact_values.items():
        if packet.get(field) != expected or type(packet.get(field)) is not type(
            expected
        ):
            issues.append(f"INVALID_{field.upper()}")

    for field in (
        "decision_sha256",
        "producer_receipt_sha256",
        "applied_receipt_sha256",
        "declared_assessment_sha256",
    ):
        if not _is_sha256(packet.get(field)):
            issues.append(f"INVALID_{field.upper()}")
        elif packet.get(field) == GENESIS_EVALUATION_SHA256:
            issues.append(f"GENESIS_{field.upper()}_FORBIDDEN")

    assessment = packet.get("declared_assessment")
    if not isinstance(assessment, Mapping):
        issues.append("INVALID_DECLARED_ASSESSMENT")
    else:
        try:
            normalized_assessment = _validate_signed_assessment(assessment)
        except AiInventoryEvaluationIntegrityError:
            issues.append("INVALID_DECLARED_ASSESSMENT")
        else:
            if packet.get("declared_assessment_sha256") != _sha256(
                _canonical_json(normalized_assessment).encode("utf-8")
            ):
                issues.append("DECLARED_ASSESSMENT_DIGEST_MISMATCH")

    cutoff = _parse_utc_nanoseconds(packet.get("decision_cutoff_at_utc"))
    horizon = _parse_utc_nanoseconds(packet.get("horizon_end_at_utc"))
    observed = _parse_utc_nanoseconds(packet.get("outcome_observed_at_utc"))
    for field, parsed in (
        ("DECISION_CUTOFF_AT_UTC", cutoff),
        ("HORIZON_END_AT_UTC", horizon),
        ("OUTCOME_OBSERVED_AT_UTC", observed),
    ):
        if parsed is None:
            issues.append(f"INVALID_{field}")
    if cutoff is not None and horizon is not None and cutoff >= horizon:
        issues.append("DECISION_CUTOFF_NOT_BEFORE_HORIZON")
    if horizon is not None and observed is not None and horizon > observed:
        issues.append("HORIZON_AFTER_OUTCOME_OBSERVATION")

    position = packet.get("position_identity")
    signal = packet.get("signal_identity")
    if position is None and signal is None:
        issues.append("MISSING_POSITION_OR_SIGNAL_IDENTITY")
    if position is not None:
        issues.extend(_identity_issues(position, position_identity=True))
    if signal is not None:
        issues.extend(_identity_issues(signal, position_identity=False))
    if isinstance(position, Mapping) and isinstance(signal, Mapping):
        for field in ("pair", "side", "strategy_tag", "entry_context_sha256"):
            if position.get(field) != signal.get(field):
                issues.append(f"POSITION_SIGNAL_MISMATCH:{field}")

    outcome_kind = packet.get("outcome_kind")
    if outcome_kind.__class__ is not str or outcome_kind not in ALLOWED_OUTCOME_KINDS:
        issues.append("INVALID_OUTCOME_KIND")
    settlement_reason = packet.get("settlement_reason")
    if outcome_kind == "SETTLEMENT":
        if (
            settlement_reason.__class__ is not str
            or _REASON_RE.fullmatch(settlement_reason) is None
        ):
            issues.append("INVALID_SETTLEMENT_REASON")
    elif settlement_reason is not None:
        issues.append("FIXED_HORIZON_SETTLEMENT_REASON_MUST_BE_NULL")

    outcome = packet.get("realized_outcome")
    if outcome.__class__ is not str or outcome not in ALLOWED_OUTCOMES:
        issues.append("INVALID_REALIZED_OUTCOME")

    numbers: dict[str, float] = {}
    for field in (
        "realized_pl_jpy",
        "mfe_jpy",
        "mae_jpy",
        "review_time_executable_exit_pl_jpy",
        "actual_exit_pl_jpy",
        "counterfactual_delta_jpy",
        "regime_confidence",
        "regime_brier_score",
    ):
        number = _finite_number(packet.get(field))
        if number is None:
            issues.append(f"INVALID_{field.upper()}")
        else:
            numbers[field] = number
    if numbers.get("mfe_jpy", 0) < 0:
        issues.append("MFE_MUST_BE_NONNEGATIVE")
    if numbers.get("mae_jpy", 0) > 0:
        issues.append("MAE_MUST_BE_NONPOSITIVE")
    if outcome == "WIN" and numbers.get("realized_pl_jpy", 0) <= 0:
        issues.append("WIN_REQUIRES_POSITIVE_REALIZED_PL")
    if outcome == "LOSS" and numbers.get("realized_pl_jpy", 0) >= 0:
        issues.append("LOSS_REQUIRES_NEGATIVE_REALIZED_PL")
    if outcome == "FLAT" and numbers.get("realized_pl_jpy", 1) != 0:
        issues.append("FLAT_REQUIRES_ZERO_REALIZED_PL")
    if {
        "review_time_executable_exit_pl_jpy",
        "actual_exit_pl_jpy",
        "counterfactual_delta_jpy",
    } <= numbers.keys():
        expected_delta = (
            numbers["review_time_executable_exit_pl_jpy"]
            - numbers["actual_exit_pl_jpy"]
        )
        if not math.isclose(
            numbers["counterfactual_delta_jpy"],
            expected_delta,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            issues.append("COUNTERFACTUAL_DELTA_MISMATCH")

    declared = packet.get("declared_regime")
    realized = packet.get("realized_regime")
    if declared.__class__ is not str or declared not in ALLOWED_REGIMES:
        issues.append("INVALID_DECLARED_REGIME")
    if realized.__class__ is not str or realized not in ALLOWED_REGIMES:
        issues.append("INVALID_REALIZED_REGIME")
    regime_correct = packet.get("regime_correct")
    if regime_correct.__class__ is not bool:
        issues.append("INVALID_REGIME_CORRECT")
    elif (
        declared.__class__ is str
        and realized.__class__ is str
        and regime_correct != (declared == realized)
    ):
        issues.append("REGIME_CORRECTNESS_MISMATCH")
    confidence = numbers.get("regime_confidence")
    brier = numbers.get("regime_brier_score")
    if confidence is not None and not 0 <= confidence <= 1:
        issues.append("REGIME_CONFIDENCE_OUT_OF_RANGE")
    if brier is not None and not 0 <= brier <= 1:
        issues.append("REGIME_BRIER_SCORE_OUT_OF_RANGE")
    if (
        confidence is not None
        and brier is not None
        and regime_correct.__class__ is bool
    ):
        expected_brier = (confidence - (1.0 if regime_correct else 0.0)) ** 2
        if not math.isclose(brier, expected_brier, rel_tol=0.0, abs_tol=1e-12):
            issues.append("REGIME_BRIER_SCORE_MISMATCH")

    issues.extend(_source_watermark_issues(packet.get("source_watermarks")))
    return tuple(_dedupe(issues))


def _identity_issues(value: object, *, position_identity: bool) -> list[str]:
    label = "POSITION_IDENTITY" if position_identity else "SIGNAL_IDENTITY"
    if not isinstance(value, Mapping):
        return [f"INVALID_{label}"]
    try:
        identity = _snapshot_mapping(value, label.lower())
    except ValueError:
        return [f"INVALID_{label}"]
    expected_keys = _POSITION_KEYS if position_identity else _SIGNAL_KEYS
    if _exact_string_keys(identity) != expected_keys:
        return [f"INVALID_{label}_KEYS"]
    issues: list[str] = []
    id_field = "position_id" if position_identity else "signal_identity_sha256"
    identifier = identity.get(id_field)
    if position_identity:
        if not _is_identifier(identifier):
            issues.append("INVALID_POSITION_ID")
    elif not _is_sha256(identifier):
        issues.append("INVALID_SIGNAL_IDENTITY_SHA256")
    if not _is_pair(identity.get("pair")):
        issues.append(f"INVALID_{label}_PAIR")
    if identity.get("side") not in {"LONG", "SHORT"}:
        issues.append(f"INVALID_{label}_SIDE")
    if not _is_identifier(identity.get("strategy_tag")):
        issues.append(f"INVALID_{label}_STRATEGY_TAG")
    if not _is_sha256(identity.get("entry_context_sha256")):
        issues.append(f"INVALID_{label}_ENTRY_CONTEXT_SHA256")
    return issues


def _source_watermark_issues(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ["INVALID_SOURCE_WATERMARKS"]
    if not 1 <= len(value) <= MAX_SOURCE_WATERMARKS:
        return ["INVALID_SOURCE_WATERMARK_COUNT"]
    issues: list[str] = []
    source_ids: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            issues.append(f"INVALID_SOURCE_WATERMARK:{index}")
            continue
        try:
            watermark = _snapshot_mapping(item, "source watermark")
        except ValueError:
            issues.append(f"INVALID_SOURCE_WATERMARK:{index}")
            continue
        if _exact_string_keys(watermark) != _WATERMARK_KEYS:
            issues.append(f"INVALID_SOURCE_WATERMARK_KEYS:{index}")
            continue
        source_id = watermark.get("source_id")
        if not _is_identifier(source_id):
            issues.append(f"INVALID_SOURCE_ID:{index}")
        else:
            source_ids.append(source_id)
        if not _is_sha256(watermark.get("sha256")):
            issues.append(f"INVALID_SOURCE_SHA256:{index}")
        if _parse_utc_nanoseconds(watermark.get("watermark_at_utc")) is None:
            issues.append(f"INVALID_SOURCE_TIMESTAMP:{index}")
    if len(source_ids) != len(set(source_ids)):
        issues.append("DUPLICATE_SOURCE_ID")
    if source_ids != sorted(source_ids):
        issues.append("SOURCE_WATERMARKS_NOT_SORTED")
    return issues


def _validate_packet_at_score_time(
    packet: Mapping[str, Any], scored_at_ns: int
) -> None:
    timestamp_fields = (
        "horizon_end_at_utc",
        "outcome_observed_at_utc",
    )
    for field in timestamp_fields:
        parsed = _parse_utc_nanoseconds(packet.get(field))
        assert parsed is not None
        if parsed > scored_at_ns:
            raise AiInventoryEvaluationError(
                f"{field} is after the internally authored score time"
            )
    for watermark in packet["source_watermarks"]:
        parsed = _parse_utc_nanoseconds(watermark["watermark_at_utc"])
        assert parsed is not None
        if parsed > scored_at_ns:
            raise AiInventoryEvaluationError(
                "source watermark is after the internally authored score time"
            )


def _record_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    # _packet_issues sees record chain fields as unknown, so validate a strict
    # projection instead of passing the complete record directly.
    issues = list(_packet_issues(_outcome_projection(record)))
    if _exact_string_keys(record) != _RECORD_KEYS:
        issues.append("INVALID_RECORD_KEYS")
    if record.get("evaluation_contract") != DOJO_AI_INVENTORY_EVALUATION_CONTRACT:
        issues.append("INVALID_EVALUATION_CONTRACT")
    sequence = record.get("sequence")
    if sequence.__class__ is not int or sequence < 1:
        issues.append("INVALID_SEQUENCE")
    if not _is_sha256(record.get("previous_evaluation_sha256")):
        issues.append("INVALID_PREVIOUS_EVALUATION_SHA256")
    scored_at = _parse_utc_nanoseconds(record.get("scored_at_utc"))
    if scored_at is None:
        issues.append("INVALID_SCORED_AT_UTC")
    else:
        try:
            if not compute_market_status(
                _datetime_from_nanoseconds(scored_at)
            ).is_fx_open:
                issues.append("SCORED_WHILE_FX_CLOSED")
        except Exception:
            issues.append("SCORED_AT_MARKET_STATUS_UNAVAILABLE")
        try:
            _validate_packet_at_score_time(record, scored_at)
        except AiInventoryEvaluationError as exc:
            issues.append(f"SCORE_TIME_BINDING:{exc}")
    expected_identity = _evaluation_identity_sha256(record)
    if record.get("evaluation_identity_sha256") != expected_identity:
        issues.append("EVALUATION_IDENTITY_SHA256_MISMATCH")
    expected_sha = _evaluation_sha256(record)
    if record.get("evaluation_sha256") != expected_sha:
        issues.append("EVALUATION_SHA256_MISMATCH")
    return tuple(_dedupe(issues))


def _validate_ledger_rows(
    rows: Sequence[dict[str, Any]],
    *,
    initial_issues: Sequence[str] = (),
) -> dict[str, Any]:
    issues = list(initial_issues)
    previous = GENESIS_EVALUATION_SHA256
    identities: set[str] = set()
    for index, row in enumerate(rows, start=1):
        row_issues = _record_issues(row)
        issues.extend(f"ROW_{index}:{issue}" for issue in row_issues)
        if row.get("sequence") != index:
            issues.append(f"ROW_{index}:SEQUENCE_MISMATCH")
        if row.get("previous_evaluation_sha256") != previous:
            issues.append(f"ROW_{index}:CHAIN_PREVIOUS_MISMATCH")
        identity = row.get("evaluation_identity_sha256")
        if isinstance(identity, str):
            if identity in identities:
                issues.append(f"ROW_{index}:DUPLICATE_EVALUATION_IDENTITY")
            identities.add(identity)
        stored = row.get("evaluation_sha256")
        if _is_sha256(stored):
            previous = stored
    terminal = rows[-1].get("evaluation_sha256") if rows else None
    return _validation_result(tuple(_dedupe(issues)), tuple(rows), terminal)


def _decode_ledger_bytes(
    raw: bytes,
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    rows: list[dict[str, Any]] = []
    if len(raw) > MAX_LEDGER_BYTES:
        return rows, ["LEDGER_SIZE_LIMIT_EXCEEDED"]
    if not raw:
        return rows, issues
    if not raw.endswith(b"\n"):
        issues.append("TRUNCATED_FINAL_ROW")
    lines = raw.splitlines()
    if len(lines) > MAX_LEDGER_ROWS:
        return rows, ["LEDGER_ROW_LIMIT_EXCEEDED"]
    for index, line in enumerate(lines, start=1):
        if not line:
            issues.append(f"ROW_{index}:EMPTY_ROW")
            continue
        if len(line) + 1 > MAX_LEDGER_LINE_BYTES:
            issues.append(f"ROW_{index}:LINE_SIZE_LIMIT_EXCEEDED")
            continue
        try:
            decoded = line.decode("utf-8")
            value = json.loads(
                decoded,
                object_pairs_hook=_strict_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            issues.append(f"ROW_{index}:INVALID_JSON:{exc.__class__.__name__}")
            continue
        if not isinstance(value, dict):
            issues.append(f"ROW_{index}:NOT_OBJECT")
            continue
        rows.append(value)
    return rows, issues


def _evaluation_identity_sha256(value: Mapping[str, Any]) -> str:
    body = {
        "decision_sha256": value.get("decision_sha256"),
        "horizon_end_at_utc": value.get("horizon_end_at_utc"),
    }
    return _sha256(_canonical_json(body).encode("utf-8"))


def _evaluation_sha256(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "evaluation_sha256"}
    return _sha256(_canonical_json(body).encode("utf-8"))


def _outcome_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.get(key) for key in _OUTCOME_KEYS}


def _validation_result(
    issues: Sequence[str],
    rows: Sequence[dict[str, Any]],
    terminal_sha256: object,
) -> dict[str, Any]:
    valid_terminal = terminal_sha256 if _is_sha256(terminal_sha256) else None
    return {
        "valid": not issues,
        "issues": list(issues),
        "row_count": len(rows),
        "terminal_evaluation_sha256": valid_terminal,
        "rows": [dict(row) for row in rows],
    }


def _open_locked_ledger(path: Path, *, exclusive: bool, create: bool) -> Any:
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise OSError("evaluation ledger must be a regular non-symlink file")
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if create:
        flags |= os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    handle = os.fdopen(descriptor, "r+b")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    return handle


def _read_locked(handle: Any) -> bytes:
    handle.seek(0)
    return handle.read(MAX_LEDGER_BYTES + 1)


def _unlock_close(handle: Any) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _snapshot_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if not all(key.__class__ is str for key in value):
        raise ValueError(f"{label} keys must be exact strings")
    try:
        encoded = _canonical_json(dict(value))
        snapshot = json.loads(
            encoded,
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict finite JSON") from exc
    if not isinstance(snapshot, dict):
        raise ValueError(f"{label} must snapshot to an object")
    return snapshot


def _strict_unique_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _parse_utc_nanoseconds(value: object) -> int | None:
    if value.__class__ is not str:
        return None
    matched = _UTC_RE.fullmatch(value)
    if matched is None:
        return None
    fraction = matched.group("fraction") or ""
    try:
        second = datetime.strptime(
            matched.group("seconds"), "%Y-%m-%dT%H:%M:%S"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(second.timestamp()) * 1_000_000_000 + int(fraction.ljust(9, "0") or 0)


def _datetime_to_canonical(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("writer clock must be timezone-aware")
    utc_value = value.astimezone(timezone.utc)
    base = utc_value.strftime("%Y-%m-%dT%H:%M:%S")
    if utc_value.microsecond:
        return f"{base}.{utc_value.microsecond:06d}Z"
    return f"{base}Z"


def _datetime_to_nanoseconds(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("writer clock must be timezone-aware")
    utc_value = value.astimezone(timezone.utc)
    return int(utc_value.timestamp()) * 1_000_000_000 + utc_value.microsecond * 1_000


def _parse_any_utc_nanoseconds(value: object) -> int | None:
    strict = _parse_utc_nanoseconds(value)
    if strict is not None:
        return strict
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return None
    return _datetime_to_nanoseconds(parsed)


def _canonical_utc(epoch_nanoseconds: int) -> str:
    seconds, nanoseconds = divmod(epoch_nanoseconds, 1_000_000_000)
    base = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    if nanoseconds:
        return f"{base}.{nanoseconds:09d}Z"
    return f"{base}Z"


def _datetime_from_nanoseconds(value: int) -> datetime:
    seconds, nanoseconds = divmod(value, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
        microsecond=nanoseconds // 1_000
    )


def _require_market_open(value: datetime) -> None:
    try:
        is_open = compute_market_status(value).is_fx_open
    except Exception as exc:
        raise AiInventoryEvaluationError(
            "FX market status is unavailable; evaluation failed closed"
        ) from exc
    if not is_open:
        raise AiInventoryEvaluationMarketClosedError(
            "new AI inventory evaluations are disabled while FX is closed"
        )


def _finite_number(value: object) -> float | None:
    if type(value) not in {int, float}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _normalized_number(value: float) -> int | float:
    if value == 0:
        return 0
    if value.is_integer():
        return int(value)
    return value


def _exact_string_keys(value: Mapping[object, object]) -> set[str]:
    return {key for key in value if key.__class__ is str}


def _is_sha256(value: object) -> bool:
    return value.__class__ is str and _SHA256_RE.fullmatch(value) is not None


def _is_identifier(value: object) -> bool:
    return (
        value.__class__ is str
        and len(value) <= MAX_ID_CHARS
        and _ID_RE.fullmatch(value) is not None
    )


def _is_pair(value: object) -> bool:
    return value.__class__ is str and _PAIR_RE.fullmatch(value) is not None


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
