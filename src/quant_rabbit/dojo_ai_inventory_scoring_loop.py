"""Prospective, fail-closed scoring loop for isolated paper-AI inventory rooms.

The loop is deliberately separate from the virtual broker and controller.  It
registers every decision before its outcome is observable, waits for the first
bound settlement or the first executable quote at/after the fixed one-hour
horizon, delegates the actual calculation to the trusted evaluator, and keeps
an append-only checkpoint chain.

Scored outcomes can be handed to a candidate-feedback sink through an
idempotency-keyed request/receipt contract.  The sink is not imported here and
cannot gain broker authority through this module.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_ai_inventory import (
    GENESIS_DECISION_SHA256,
    validate_inventory_decision,
)
from quant_rabbit.dojo_ai_inventory_evaluator import (
    BROKER_LEDGER_NAME,
    DECISION_LEDGER_NAME,
    EVALUATION_LEDGER_NAME,
    QUOTE_WATERMARK_LEDGER_NAME,
    AiInventoryEvaluationError,
    evaluate_ai_inventory_outcome,
)
from quant_rabbit.dojo_ai_inventory_quote_watermark import (
    GENESIS_QUOTE_SHA256,
    QUOTE_SOURCE_CONTRACT,
    QUOTE_SOURCE_DIRECTORY,
    QUOTE_WATERMARK_CONTRACT,
    quote_watermark_sha256,
)
from quant_rabbit.dojo_replay_lifecycle import (
    canonical_paper_ai_rooms_root,
    verify_paper_ai_inventory_launch_preflight,
)


SCORING_CHECKPOINT_CONTRACT = "QR_DOJO_AI_INVENTORY_SCORING_CHECKPOINT_V1"
SCORING_FEEDBACK_CONTRACT = "QR_DOJO_AI_INVENTORY_SCORE_FEEDBACK_V1"
SCORING_FEEDBACK_RECEIPT_CONTRACT = (
    "QR_DOJO_AI_INVENTORY_SCORE_FEEDBACK_RECEIPT_V1"
)
SCORING_CHECKPOINT_LEDGER_NAME = "ai_inventory_scoring_checkpoints.jsonl"
FIXED_HORIZON_SECONDS = 60 * 60
GENESIS_CHECKPOINT_SHA256 = "0" * 64
MAX_LEDGER_BYTES = 256 * 1024 * 1024
MAX_LEDGER_ROWS = 1_000_000
MAX_LINE_BYTES = 512 * 1024
MAX_SOURCE_BYTES = 64 * 1024

EVENT_DECISION_PENDING = "DECISION_PENDING"
EVENT_SCORED = "AI_SHADOW_SCORED"
EVENT_UNSCORED = "AI_SHADOW_UNSCORED"
EVENT_FEEDBACK_PENDING = "FEEDBACK_PENDING"
EVENT_FEEDBACK_ACKNOWLEDGED = "FEEDBACK_ACKNOWLEDGED"
EVENT_TYPES = frozenset(
    {
        EVENT_DECISION_PENDING,
        EVENT_SCORED,
        EVENT_UNSCORED,
        EVENT_FEEDBACK_PENDING,
        EVENT_FEEDBACK_ACKNOWLEDGED,
    }
)
SETTLEMENT_EVENTS = frozenset({"CLOSE", "EXIT_TP", "EXIT_SL", "MARGIN_CLOSEOUT"})
UNSCORED_REASON_CODES = frozenset(
    {
        "SOURCE_INTEGRITY_DEFECT",
        "MISSED_PROSPECTIVE_REGISTRATION",
        "OUTCOME_PRECEDES_PENDING_REGISTRATION",
        "EVALUATION_INTEGRITY_DEFECT",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_CHECKPOINT_KEYS = frozenset(
    {
        "contract",
        "sequence",
        "event_type",
        "checkpoint_identity_sha256",
        "decision_sha256",
        "recorded_at_utc",
        "previous_checkpoint_sha256",
        "payload_sha256",
        "payload",
        "paper_only",
        "order_authority",
        "live_permission",
        "checkpoint_sha256",
    }
)
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
_FEEDBACK_RECEIPT_KEYS = frozenset(
    {
        "contract",
        "feedback_identity_sha256",
        "candidate_id",
        "decision_sha256",
        "evaluation_sha256",
        "sink_id",
        "sink_event_sha256",
        "sink_ledger_tip_sha256",
        "accepted_at_utc",
        "paper_only",
        "order_authority",
        "live_permission",
        "receipt_sha256",
    }
)

FeedbackCallback = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class AiInventoryScoringLoopError(RuntimeError):
    """The prospective scoring loop could not safely continue."""


class AiInventoryScoringMarketClosedError(AiInventoryScoringLoopError):
    """A scoring operation was attempted while the FX week was closed."""


class AiInventoryScoringRegistrationError(AiInventoryScoringLoopError):
    """The room is not a registered paper-AI inventory room."""


class AiInventoryScoringIntegrityError(AiInventoryScoringLoopError):
    """A source or checkpoint chain failed validation."""


@dataclass(frozen=True)
class ScoringCycleResult:
    """One bounded scan of a single registered room."""

    checkpoint_events: tuple[dict[str, Any], ...]
    feedback_requests: tuple[dict[str, Any], ...]
    pending_decisions: tuple[str, ...]
    scored_decisions: tuple[str, ...]
    unscored_decisions: tuple[str, ...]


def run_ai_inventory_scoring_cycle(
    room_root: Path,
    *,
    feedback_callback: FeedbackCallback | None = None,
) -> ScoringCycleResult:
    """Register, mature, and score all decisions visible in one room snapshot.

    The weekend gate is intentionally the first operation.  It precedes room
    resolution, launch-preflight verification, and every filesystem read.
    """

    cycle_now = _utc_now().astimezone(timezone.utc)
    _require_market_open(cycle_now)
    root, preflight = _require_registered_room(room_root)
    decisions = _read_validate_decisions(
        root / DECISION_LEDGER_NAME,
        observed_no_later_than=cycle_now,
    )
    checkpoints = _read_validate_checkpoints(
        root / SCORING_CHECKPOINT_LEDGER_NAME,
        missing_ok=True,
    )
    _require_checkpoints_observed(checkpoints, no_later_than=cycle_now)
    _validate_checkpoint_state(checkpoints)

    source_error: Exception | None = None
    try:
        broker_rows = _read_validate_broker_rows(
            root / BROKER_LEDGER_NAME,
            observed_no_later_than=cycle_now,
            missing_ok=True,
        )
        quote_rows = _read_validate_quote_rows(
            root / QUOTE_WATERMARK_LEDGER_NAME,
            observed_no_later_than=cycle_now,
            missing_ok=True,
        )
    except AiInventoryScoringIntegrityError as exc:
        broker_rows = []
        quote_rows = []
        source_error = exc

    appended_events: list[dict[str, Any]] = []
    for decision in decisions:
        decision_sha = str(decision["decision_sha256"])
        state = _state_for_decision(checkpoints, decision_sha)
        target_horizon = _fixed_horizon(decision)
        maturity = (
            None
            if source_error is not None
            else _select_first_maturity(
                decision,
                broker_rows=broker_rows,
                quote_rows=quote_rows,
                target_horizon=target_horizon,
            )
        )
        if state["pending"] is None:
            prospective = (
                source_error is None
                and maturity is None
                and cycle_now < target_horizon
            )
            pending_payload = {
                "status": "AI_SHADOW_UNSCORED",
                "prospective_registration": prospective,
                "decision_cutoff_at_utc": decision["cutoff_at_utc"],
                "target_horizon_at_utc": _canonical_utc(target_horizon),
                "registered_source_tips": _source_tips(
                    decisions, broker_rows, quote_rows
                ),
                "launch_preflight_token_sha256": preflight[
                    "launch_preflight_token_sha256"
                ],
            }
            pending, appended = _append_checkpoint(
                root,
                event_type=EVENT_DECISION_PENDING,
                decision_sha256=decision_sha,
                payload=pending_payload,
                recorded_at=cycle_now,
            )
            if appended:
                appended_events.append(pending)
                checkpoints.append(pending)
            state = _state_for_decision(checkpoints, decision_sha)
            if not prospective:
                reason = (
                    "SOURCE_INTEGRITY_DEFECT"
                    if source_error is not None
                    else "MISSED_PROSPECTIVE_REGISTRATION"
                )
                defect, appended = _append_unscored(
                    root,
                    decision=decision,
                    pending=state["pending"],
                    reason_code=reason,
                    defect=source_error,
                    source_tips=_source_tips(decisions, broker_rows, quote_rows),
                    recorded_at=cycle_now,
                )
                if appended:
                    appended_events.append(defect)
                    checkpoints.append(defect)
                continue

        state = _state_for_decision(checkpoints, decision_sha)
        if state["terminal"] is not None:
            continue
        pending = state["pending"]
        assert pending is not None
        if source_error is not None:
            defect, appended = _append_unscored(
                root,
                decision=decision,
                pending=pending,
                reason_code="SOURCE_INTEGRITY_DEFECT",
                defect=source_error,
                source_tips=_source_tips(decisions, broker_rows, quote_rows),
                recorded_at=cycle_now,
            )
            if appended:
                appended_events.append(defect)
                checkpoints.append(defect)
            continue
        if maturity is None:
            continue
        maturity_at = _parse_utc(maturity["horizon_end_at_utc"], "maturity horizon")
        pending_at = _parse_utc(pending["recorded_at_utc"], "pending recorded_at")
        if maturity_at <= pending_at:
            defect, appended = _append_unscored(
                root,
                decision=decision,
                pending=pending,
                reason_code="OUTCOME_PRECEDES_PENDING_REGISTRATION",
                defect=None,
                source_tips=_source_tips(decisions, broker_rows, quote_rows),
                recorded_at=cycle_now,
            )
            if appended:
                appended_events.append(defect)
                checkpoints.append(defect)
            continue
        try:
            evaluated = evaluate_ai_inventory_outcome(
                root,
                decision_sha256=decision_sha,
                horizon_end_at_utc=maturity["horizon_end_at_utc"],
                outcome_kind=maturity["outcome_kind"],
            )
        except AiInventoryEvaluationError as exc:
            defect, appended = _append_unscored(
                root,
                decision=decision,
                pending=pending,
                reason_code="EVALUATION_INTEGRITY_DEFECT",
                defect=exc,
                source_tips=_source_tips(decisions, broker_rows, quote_rows),
                recorded_at=cycle_now,
            )
            if appended:
                appended_events.append(defect)
                checkpoints.append(defect)
            continue
        scored_payload = {
            "status": "AI_SHADOW_SCORED",
            "pending_checkpoint_sha256": pending["checkpoint_sha256"],
            "evaluation_identity_sha256": evaluated.record[
                "evaluation_identity_sha256"
            ],
            "evaluation_sha256": evaluated.record["evaluation_sha256"],
            "outcome_kind": evaluated.record["outcome_kind"],
            "horizon_end_at_utc": evaluated.record["horizon_end_at_utc"],
            "source_watermarks": evaluated.record["source_watermarks"],
        }
        scored, appended = _append_checkpoint(
            root,
            event_type=EVENT_SCORED,
            decision_sha256=decision_sha,
            payload=scored_payload,
            recorded_at=cycle_now,
        )
        if appended:
            appended_events.append(scored)
            checkpoints.append(scored)

    feedback_requests: list[dict[str, Any]] = []
    evaluation_rows = _read_evaluations(
        root / EVALUATION_LEDGER_NAME,
        observed_no_later_than=cycle_now,
    )
    for decision in decisions:
        decision_sha = str(decision["decision_sha256"])
        state = _state_for_decision(checkpoints, decision_sha)
        scored = state["scored"]
        if scored is None or state["feedback_ack"] is not None:
            continue
        evaluation = _evaluation_for_scored_checkpoint(evaluation_rows, scored)
        request = _feedback_request(decision, scored, evaluation)
        feedback_pending, appended = _append_checkpoint(
            root,
            event_type=EVENT_FEEDBACK_PENDING,
            decision_sha256=decision_sha,
            payload=request,
            recorded_at=cycle_now,
        )
        if appended:
            appended_events.append(feedback_pending)
            checkpoints.append(feedback_pending)
        feedback_requests.append(request)
        if feedback_callback is not None:
            before = _canonical_json(request)
            raw_receipt = feedback_callback(json.loads(before))
            if _canonical_json(request) != before:
                raise AiInventoryScoringIntegrityError(
                    "feedback callback mutated the sealed request"
                )
            callback_completed_at = _utc_now().astimezone(timezone.utc)
            _require_market_open(callback_completed_at)
            acknowledged, appended = _acknowledge_feedback(
                root,
                request=request,
                receipt=raw_receipt,
                recorded_at=callback_completed_at,
            )
            if appended:
                appended_events.append(acknowledged)
                checkpoints.append(acknowledged)

    final_states = {
        str(decision["decision_sha256"]): _state_for_decision(
            checkpoints, str(decision["decision_sha256"])
        )
        for decision in decisions
    }
    return ScoringCycleResult(
        checkpoint_events=tuple(appended_events),
        feedback_requests=tuple(feedback_requests),
        pending_decisions=tuple(
            decision_sha
            for decision_sha, state in final_states.items()
            if state["terminal"] is None
        ),
        scored_decisions=tuple(
            decision_sha
            for decision_sha, state in final_states.items()
            if state["scored"] is not None
        ),
        unscored_decisions=tuple(
            decision_sha
            for decision_sha, state in final_states.items()
            if state["unscored"] is not None
        ),
    )


def acknowledge_ai_inventory_scoring_feedback(
    room_root: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Acknowledge one externally persisted, idempotency-keyed feedback event."""

    now = _utc_now().astimezone(timezone.utc)
    _require_market_open(now)
    root, _preflight = _require_registered_room(room_root)
    checkpoints = _read_validate_checkpoints(
        root / SCORING_CHECKPOINT_LEDGER_NAME,
        missing_ok=False,
    )
    _require_checkpoints_observed(checkpoints, no_later_than=now)
    _validate_checkpoint_state(checkpoints)
    normalized = _validate_feedback_receipt(receipt, no_later_than=now)
    pending_matches = [
        row
        for row in checkpoints
        if row["event_type"] == EVENT_FEEDBACK_PENDING
        and row["payload"]["feedback_identity_sha256"]
        == normalized["feedback_identity_sha256"]
    ]
    if len(pending_matches) != 1:
        raise AiInventoryScoringIntegrityError(
            "feedback receipt lacks one pending request"
        )
    request = pending_matches[0]["payload"]
    acknowledged, _appended = _acknowledge_feedback(
        root,
        request=request,
        receipt=normalized,
        recorded_at=now,
    )
    return acknowledged


def validate_ai_inventory_scoring_checkpoint_ledger(
    path: Path,
) -> dict[str, Any]:
    """Validate a complete checkpoint chain without running the scorer."""

    try:
        rows = _read_validate_checkpoints(path, missing_ok=True)
        _validate_checkpoint_state(rows)
    except AiInventoryScoringLoopError as exc:
        return {
            "valid": False,
            "row_count": 0,
            "terminal_checkpoint_sha256": None,
            "issues": (exc.__class__.__name__,),
        }
    return {
        "valid": True,
        "row_count": len(rows),
        "terminal_checkpoint_sha256": (
            rows[-1]["checkpoint_sha256"] if rows else GENESIS_CHECKPOINT_SHA256
        ),
        "issues": (),
    }


def feedback_receipt_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("receipt_sha256", None)
    return _sha256(_canonical_json(body).encode("utf-8"))


def _require_registered_room(room_root: Path) -> tuple[Path, dict[str, Any]]:
    root = _require_canonical_room_root(room_root)
    try:
        token = verify_paper_ai_inventory_launch_preflight(
            _trusted_repository_root(),
            experiment_id=root.parent.name,
            room_id=root.name,
        )
    except Exception as exc:
        raise AiInventoryScoringRegistrationError(
            "paper-AI inventory room is not registered"
        ) from exc
    if (
        token.get("experiment_id") != root.parent.name
        or token.get("room_id") != root.name
        or not _is_sha256(token.get("launch_preflight_token_sha256"))
        or token.get("paper_only") is not True
        or token.get("order_authority") != "NONE"
        or token.get("live_permission") is not False
    ):
        raise AiInventoryScoringRegistrationError(
            "launch preflight does not bind this paper-only room"
        )
    return root, dict(token)


def _require_canonical_room_root(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise AiInventoryScoringRegistrationError(
            "room_root must be an absolute Path"
        )
    repository_root = _trusted_repository_root()
    try:
        rooms_root = canonical_paper_ai_rooms_root(repository_root).resolve(strict=True)
        info = value.lstat()
        parent_info = value.parent.lstat()
        root = value.resolve(strict=True)
        relative = root.relative_to(rooms_root)
    except (OSError, ValueError) as exc:
        raise AiInventoryScoringRegistrationError(
            "room_root is outside the canonical paper-AI rooms root"
        ) from exc
    if (
        root != value
        or stat.S_ISLNK(info.st_mode)
        or stat.S_ISLNK(parent_info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or not stat.S_ISDIR(parent_info.st_mode)
        or len(relative.parts) != 2
        or any(
            not part.startswith("paper-ai-inventory-") for part in relative.parts
        )
    ):
        raise AiInventoryScoringRegistrationError(
            "room_root is not an isolated paper-AI inventory room"
        )
    return root


def _trusted_repository_root() -> Path:
    try:
        return Path(__file__).resolve(strict=True).parents[2].resolve(strict=True)
    except (IndexError, OSError) as exc:
        raise AiInventoryScoringRegistrationError(
            "package-derived repository root is unavailable"
        ) from exc


def _read_validate_decisions(
    path: Path,
    *,
    observed_no_later_than: datetime,
) -> list[dict[str, Any]]:
    rows = _read_jsonl_locked(path, missing_ok=False)
    previous = GENESIS_DECISION_SHA256
    for index, row in enumerate(rows, 1):
        issues = validate_inventory_decision(row)
        cutoff = _parse_utc(row.get("cutoff_at_utc"), "decision cutoff")
        recorded = _parse_utc(row.get("recorded_at_utc"), "decision recorded_at")
        if (
            issues
            or row.get("sequence") != index
            or row.get("previous_decision_sha256") != previous
            or cutoff > observed_no_later_than
            or recorded > observed_no_later_than
        ):
            raise AiInventoryScoringIntegrityError(
                f"decision ledger failed validation at row {index}"
            )
        previous = str(row["decision_sha256"])
    if not rows:
        raise AiInventoryScoringIntegrityError("decision ledger is empty")
    return rows


def _read_validate_broker_rows(
    path: Path,
    *,
    observed_no_later_than: datetime,
    missing_ok: bool,
) -> list[dict[str, Any]]:
    rows = _read_jsonl_locked(path, missing_ok=missing_ok)
    previous = "0" * 64
    previous_at: datetime | None = None
    for index, row in enumerate(rows, 1):
        body = {
            key: row.get(key) for key in ("ts_utc", "event", "payload", "prev_sha")
        }
        timestamp = _parse_utc(row.get("ts_utc"), "broker timestamp")
        if (
            set(row) != _BROKER_ROW_KEYS
            or row.get("prev_sha") != previous
            or row.get("sha") != _sha256(_canonical_json(body).encode("utf-8"))
            or not isinstance(row.get("event"), str)
            or not isinstance(row.get("payload"), dict)
            or timestamp > observed_no_later_than
            or (previous_at is not None and timestamp < previous_at)
        ):
            raise AiInventoryScoringIntegrityError(
                f"broker ledger failed validation at row {index}"
            )
        previous = str(row["sha"])
        previous_at = timestamp
    return rows


def _read_validate_quote_rows(
    path: Path,
    *,
    observed_no_later_than: datetime,
    missing_ok: bool,
) -> list[dict[str, Any]]:
    rows = _read_jsonl_locked(path, missing_ok=missing_ok)
    previous = GENESIS_QUOTE_SHA256
    previous_at: datetime | None = None
    for index, row in enumerate(rows, 1):
        timestamp = _parse_utc(row.get("timestamp_utc"), "quote timestamp")
        recorded = _parse_utc(row.get("recorded_at_utc"), "quote recorded_at")
        if (
            set(row) != _QUOTE_ROW_KEYS
            or row.get("contract") != QUOTE_WATERMARK_CONTRACT
            or row.get("sequence") != index
            or row.get("previous_quote_sha256") != previous
            or row.get("quote_sha256") != quote_watermark_sha256(row)
            or not _is_sha256(row.get("source_sha256"))
            or not _is_sha256(row.get("capture_source_sha256"))
            or row.get("capture_source_sha256") == GENESIS_QUOTE_SHA256
            or not _is_sha256(row.get("acquisition_receipt_sha256"))
            or row.get("acquisition_receipt_sha256") == GENESIS_QUOTE_SHA256
            or not _is_pair(row.get("pair"))
            or not _positive_number(row.get("bid"))
            or not _positive_number(row.get("ask"))
            or float(row["ask"]) < float(row["bid"])
            or not _nonnegative_number(row.get("slippage_pips_per_fill"))
            or not _nonnegative_number(row.get("financing_pips_per_day"))
            or row.get("paper_only") is not True
            or row.get("order_authority") != "NONE"
            or row.get("live_permission") is not False
            or recorded < timestamp
            or timestamp > observed_no_later_than
            or recorded > observed_no_later_than
            or (previous_at is not None and timestamp < previous_at)
        ):
            raise AiInventoryScoringIntegrityError(
                f"quote ledger failed validation at row {index}"
            )
        source_path = path.parent / QUOTE_SOURCE_DIRECTORY / (
            f"{row['source_sha256']}.json"
        )
        source_raw = _read_regular_nofollow(source_path, MAX_SOURCE_BYTES)
        if _sha256(source_raw) != row["source_sha256"]:
            raise AiInventoryScoringIntegrityError(
                f"quote source digest mismatch at row {index}"
            )
        source = _parse_json(source_raw, f"quote source row {index}")
        if source != {
            "contract": QUOTE_SOURCE_CONTRACT,
            "timestamp_utc": row["timestamp_utc"],
            "pair": row["pair"],
            "bid": row["bid"],
            "ask": row["ask"],
            "capture_source_sha256": row["capture_source_sha256"],
            "acquisition_receipt_sha256": row["acquisition_receipt_sha256"],
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }:
            raise AiInventoryScoringIntegrityError(
                f"quote source content mismatch at row {index}"
            )
        previous = str(row["quote_sha256"])
        previous_at = timestamp
    return rows


def _fixed_horizon(decision: Mapping[str, Any]) -> datetime:
    cutoff = _parse_utc(decision.get("cutoff_at_utc"), "decision cutoff")
    return cutoff + timedelta(seconds=FIXED_HORIZON_SECONDS)


def _select_first_maturity(
    decision: Mapping[str, Any],
    *,
    broker_rows: Sequence[Mapping[str, Any]],
    quote_rows: Sequence[Mapping[str, Any]],
    target_horizon: datetime,
) -> dict[str, str] | None:
    position = decision.get("position_binding")
    if not isinstance(position, Mapping):
        raise AiInventoryScoringIntegrityError("decision position binding is invalid")
    cutoff = _parse_utc(decision.get("cutoff_at_utc"), "decision cutoff")
    pair = str(position.get("pair"))
    settlement: tuple[datetime, Mapping[str, Any]] | None = None
    if position.get("side") in {"LONG", "SHORT"}:
        candidates: list[tuple[datetime, Mapping[str, Any]]] = []
        for row in broker_rows:
            payload = row.get("payload")
            if (
                row.get("event") not in SETTLEMENT_EVENTS
                or not isinstance(payload, Mapping)
                or payload.get("trade_id") != position.get("position_id")
            ):
                continue
            quote = payload.get("quote")
            if not isinstance(quote, Mapping):
                raise AiInventoryScoringIntegrityError(
                    "bound settlement lacks a quote timestamp"
                )
            settled_at = _parse_utc(quote.get("ts"), "settlement quote timestamp")
            if settled_at > cutoff:
                candidates.append((settled_at, row))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            settlement = candidates[0]
    fixed_quotes = sorted(
        (
            _parse_utc(row.get("timestamp_utc"), "quote timestamp"),
            row,
        )
        for row in quote_rows
        if row.get("pair") == pair
        and _parse_utc(row.get("timestamp_utc"), "quote timestamp")
        >= target_horizon
    )
    fixed = fixed_quotes[0] if fixed_quotes else None
    if settlement is not None and (fixed is None or settlement[0] <= fixed[0]):
        return {
            "outcome_kind": "SETTLEMENT",
            "horizon_end_at_utc": _canonical_utc(settlement[0]),
        }
    if fixed is not None:
        return {
            "outcome_kind": "FIXED_HORIZON",
            "horizon_end_at_utc": _canonical_utc(fixed[0]),
        }
    return None


def _append_unscored(
    root: Path,
    *,
    decision: Mapping[str, Any],
    pending: Mapping[str, Any],
    reason_code: str,
    defect: Exception | None,
    source_tips: Mapping[str, Any],
    recorded_at: datetime,
) -> tuple[dict[str, Any], bool]:
    defect_body = {
        "error_class": defect.__class__.__name__ if defect is not None else None,
        "message_sha256": (
            _sha256(str(defect).encode("utf-8")) if defect is not None else None
        ),
    }
    return _append_checkpoint(
        root,
        event_type=EVENT_UNSCORED,
        decision_sha256=str(decision["decision_sha256"]),
        payload={
            "status": "AI_SHADOW_UNSCORED",
            "reason_code": reason_code,
            "pending_checkpoint_sha256": pending["checkpoint_sha256"],
            "defect": defect_body,
            "source_tips": dict(source_tips),
        },
        recorded_at=recorded_at,
    )


def _append_checkpoint(
    root: Path,
    *,
    event_type: str,
    decision_sha256: str,
    payload: Mapping[str, Any],
    recorded_at: datetime,
) -> tuple[dict[str, Any], bool]:
    if event_type not in EVENT_TYPES or not _is_sha256(decision_sha256):
        raise AiInventoryScoringIntegrityError("checkpoint identity is invalid")
    normalized_payload = _snapshot_mapping(payload, "checkpoint payload")
    identity = _sha256(
        _canonical_json(
            {
                "event_type": event_type,
                "decision_sha256": decision_sha256,
                "payload": normalized_payload,
            }
        ).encode("utf-8")
    )
    path = root / SCORING_CHECKPOINT_LEDGER_NAME
    handle = _open_locked_ledger(path, create=True)
    try:
        rows = _decode_checkpoint_bytes(handle.read())
        _validate_checkpoint_rows(rows)
        _require_checkpoints_observed(rows, no_later_than=recorded_at)
        _validate_checkpoint_state(rows)
        existing = [
            row for row in rows if row["checkpoint_identity_sha256"] == identity
        ]
        if existing:
            if len(existing) != 1:
                raise AiInventoryScoringIntegrityError(
                    "checkpoint identity is duplicated"
                )
            return dict(existing[0]), False
        body = {
            "contract": SCORING_CHECKPOINT_CONTRACT,
            "sequence": len(rows) + 1,
            "event_type": event_type,
            "checkpoint_identity_sha256": identity,
            "decision_sha256": decision_sha256,
            "recorded_at_utc": _canonical_utc(recorded_at),
            "previous_checkpoint_sha256": (
                rows[-1]["checkpoint_sha256"]
                if rows
                else GENESIS_CHECKPOINT_SHA256
            ),
            "payload_sha256": _sha256(
                _canonical_json(normalized_payload).encode("utf-8")
            ),
            "payload": normalized_payload,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }
        record = {
            **body,
            "checkpoint_sha256": _sha256(
                _canonical_json(body).encode("utf-8")
            ),
        }
        _validate_checkpoint_rows([*rows, record])
        _validate_checkpoint_state([*rows, record])
        encoded = (_canonical_json(record) + "\n").encode("utf-8")
        if len(encoded) > MAX_LINE_BYTES:
            raise AiInventoryScoringIntegrityError(
                "checkpoint row exceeds the line limit"
            )
        handle.seek(0, os.SEEK_END)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        return record, True
    finally:
        _unlock_close(handle)


def _read_validate_checkpoints(path: Path, *, missing_ok: bool) -> list[dict[str, Any]]:
    if missing_ok and not path.exists():
        return []
    handle = _open_locked_ledger(path, create=False)
    try:
        rows = _decode_checkpoint_bytes(handle.read())
    finally:
        _unlock_close(handle)
    _validate_checkpoint_rows(rows)
    return rows


def _validate_checkpoint_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    previous = GENESIS_CHECKPOINT_SHA256
    previous_at: datetime | None = None
    for index, row in enumerate(rows, 1):
        body = {
            key: row[key] for key in row if key != "checkpoint_sha256"
        }
        payload = row.get("payload")
        recorded = _parse_utc(row.get("recorded_at_utc"), "checkpoint recorded_at")
        if (
            set(row) != _CHECKPOINT_KEYS
            or row.get("contract") != SCORING_CHECKPOINT_CONTRACT
            or row.get("sequence") != index
            or row.get("event_type") not in EVENT_TYPES
            or not _is_sha256(row.get("checkpoint_identity_sha256"))
            or not _is_sha256(row.get("decision_sha256"))
            or row.get("previous_checkpoint_sha256") != previous
            or not isinstance(payload, Mapping)
            or row.get("payload_sha256")
            != _sha256(_canonical_json(payload).encode("utf-8"))
            or row.get("checkpoint_sha256")
            != _sha256(_canonical_json(body).encode("utf-8"))
            or row.get("paper_only") is not True
            or row.get("order_authority") != "NONE"
            or row.get("live_permission") is not False
            or (previous_at is not None and recorded < previous_at)
        ):
            raise AiInventoryScoringIntegrityError(
                f"checkpoint ledger failed validation at row {index}"
            )
        identity = _sha256(
            _canonical_json(
                {
                    "event_type": row["event_type"],
                    "decision_sha256": row["decision_sha256"],
                    "payload": payload,
                }
            ).encode("utf-8")
        )
        if row["checkpoint_identity_sha256"] != identity:
            raise AiInventoryScoringIntegrityError(
                f"checkpoint identity mismatch at row {index}"
            )
        _validate_checkpoint_payload(row)
        previous = str(row["checkpoint_sha256"])
        previous_at = recorded


def _require_checkpoints_observed(
    rows: Sequence[Mapping[str, Any]],
    *,
    no_later_than: datetime,
) -> None:
    if any(
        _parse_utc(row.get("recorded_at_utc"), "checkpoint recorded_at")
        > no_later_than
        for row in rows
    ):
        raise AiInventoryScoringIntegrityError(
            "checkpoint ledger contains a future-authored row"
        )


def _validate_checkpoint_payload(row: Mapping[str, Any]) -> None:
    event = row["event_type"]
    payload = row["payload"]
    recorded = _parse_utc(row["recorded_at_utc"], "checkpoint recorded_at")
    if event == EVENT_DECISION_PENDING:
        if set(payload) != {
            "status",
            "prospective_registration",
            "decision_cutoff_at_utc",
            "target_horizon_at_utc",
            "registered_source_tips",
            "launch_preflight_token_sha256",
        }:
            raise AiInventoryScoringIntegrityError(
                "pending checkpoint payload schema is invalid"
            )
        cutoff = _parse_utc(
            payload.get("decision_cutoff_at_utc"), "pending decision cutoff"
        )
        target = _parse_utc(
            payload.get("target_horizon_at_utc"), "pending target horizon"
        )
        if (
            payload.get("status") != "AI_SHADOW_UNSCORED"
            or not isinstance(
                payload.get("prospective_registration"), bool
            )
            or target <= cutoff
            or (
                payload["prospective_registration"] is True
                and recorded >= target
            )
            or not _is_sha256(payload.get("launch_preflight_token_sha256"))
        ):
            raise AiInventoryScoringIntegrityError(
                "pending checkpoint payload is invalid"
            )
        _validate_source_tips(payload.get("registered_source_tips"))
        return
    if event == EVENT_UNSCORED:
        if set(payload) != {
            "status",
            "reason_code",
            "pending_checkpoint_sha256",
            "defect",
            "source_tips",
        }:
            raise AiInventoryScoringIntegrityError(
                "unscored checkpoint payload schema is invalid"
            )
        defect = payload.get("defect")
        if (
            payload.get("status") != "AI_SHADOW_UNSCORED"
            or payload.get("reason_code") not in UNSCORED_REASON_CODES
            or not _is_sha256(payload.get("pending_checkpoint_sha256"))
            or not isinstance(defect, Mapping)
            or set(defect) != {"error_class", "message_sha256"}
            or (
                defect.get("error_class") is None
                and defect.get("message_sha256") is not None
            )
            or (
                defect.get("error_class") is not None
                and (
                    not isinstance(defect.get("error_class"), str)
                    or not defect["error_class"]
                    or not _is_sha256(defect.get("message_sha256"))
                )
            )
        ):
            raise AiInventoryScoringIntegrityError(
                "unscored checkpoint payload is invalid"
            )
        _validate_source_tips(payload.get("source_tips"))
        return
    if event == EVENT_SCORED:
        if set(payload) != {
            "status",
            "pending_checkpoint_sha256",
            "evaluation_identity_sha256",
            "evaluation_sha256",
            "outcome_kind",
            "horizon_end_at_utc",
            "source_watermarks",
        }:
            raise AiInventoryScoringIntegrityError(
                "scored checkpoint payload schema is invalid"
            )
        horizon = _parse_utc(
            payload.get("horizon_end_at_utc"), "scored horizon"
        )
        if (
            payload.get("status") != "AI_SHADOW_SCORED"
            or any(
                not _is_sha256(payload.get(field))
                for field in (
                    "pending_checkpoint_sha256",
                    "evaluation_identity_sha256",
                    "evaluation_sha256",
                )
            )
            or payload.get("outcome_kind") not in {"SETTLEMENT", "FIXED_HORIZON"}
            or horizon > recorded
        ):
            raise AiInventoryScoringIntegrityError(
                "scored checkpoint payload is invalid"
            )
        _validate_source_watermarks(
            payload.get("source_watermarks"),
            no_later_than=horizon,
        )
        return
    if event == EVENT_FEEDBACK_PENDING:
        expected = {
            "contract",
            "candidate_id",
            "decision_sha256",
            "evaluation_sha256",
            "scored_checkpoint_sha256",
            "feedback_not_before_utc",
            "realized_outcome",
            "realized_pl_jpy",
            "regime_correct",
            "regime_brier_score",
            "source_watermarks",
            "paper_only",
            "order_authority",
            "live_permission",
            "feedback_identity_sha256",
        }
        body = {
            key: value
            for key, value in payload.items()
            if key != "feedback_identity_sha256"
        }
        not_before = _parse_utc(
            payload.get("feedback_not_before_utc"), "feedback not-before"
        )
        if (
            set(payload) != expected
            or payload.get("contract") != SCORING_FEEDBACK_CONTRACT
            or payload.get("decision_sha256") != row["decision_sha256"]
            or any(
                not _is_sha256(payload.get(field))
                for field in (
                    "candidate_id",
                    "decision_sha256",
                    "evaluation_sha256",
                    "scored_checkpoint_sha256",
                )
            )
            or payload.get("realized_outcome") not in {"WIN", "LOSS", "FLAT"}
            or not _finite_number(payload.get("realized_pl_jpy"))
            or not isinstance(payload.get("regime_correct"), bool)
            or not _unit_interval(payload.get("regime_brier_score"))
            or not_before > recorded
            or payload.get("paper_only") is not True
            or payload.get("order_authority") != "NONE"
            or payload.get("live_permission") is not False
            or payload.get("feedback_identity_sha256")
            != _sha256(_canonical_json(body).encode("utf-8"))
        ):
            raise AiInventoryScoringIntegrityError(
                "feedback request payload is invalid"
            )
        _validate_source_watermarks(
            payload.get("source_watermarks"),
            no_later_than=recorded,
        )
        return
    if event == EVENT_FEEDBACK_ACKNOWLEDGED:
        if set(payload) != {
            "status",
            "feedback_identity_sha256",
            "receipt",
        }:
            raise AiInventoryScoringIntegrityError(
                "feedback acknowledgement payload schema is invalid"
            )
        receipt = payload.get("receipt")
        if (
            payload.get("status") != "ACKNOWLEDGED"
            or not _is_sha256(payload.get("feedback_identity_sha256"))
            or not isinstance(receipt, Mapping)
        ):
            raise AiInventoryScoringIntegrityError(
                "feedback acknowledgement payload is invalid"
            )
        normalized = _validate_feedback_receipt(
            receipt,
            no_later_than=recorded,
        )
        if (
            normalized["feedback_identity_sha256"]
            != payload["feedback_identity_sha256"]
            or normalized["decision_sha256"] != row["decision_sha256"]
        ):
            raise AiInventoryScoringIntegrityError(
                "feedback acknowledgement receipt binding is invalid"
            )
        return
    raise AiInventoryScoringIntegrityError("unknown checkpoint event type")


def _validate_source_tips(value: object) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "decision_ledger_tip_sha256",
            "broker_ledger_tip_sha256",
            "quote_ledger_tip_sha256",
        }
        or any(not _is_sha256(item) for item in value.values())
    ):
        raise AiInventoryScoringIntegrityError("checkpoint source tips are invalid")


def _validate_source_watermarks(
    value: object,
    *,
    no_later_than: datetime,
) -> None:
    if not isinstance(value, list) or not value:
        raise AiInventoryScoringIntegrityError("source watermarks are missing")
    identities: set[str] = set()
    for item in value:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"source_id", "sha256", "watermark_at_utc"}
            or not isinstance(item.get("source_id"), str)
            or not item["source_id"]
            or item["source_id"] in identities
            or not _is_sha256(item.get("sha256"))
            or _parse_utc(
                item.get("watermark_at_utc"), "source watermark"
            )
            > no_later_than
        ):
            raise AiInventoryScoringIntegrityError(
                "source watermark binding is invalid"
            )
        identities.add(str(item["source_id"]))


def _validate_checkpoint_state(rows: Sequence[Mapping[str, Any]]) -> None:
    seen_identities: set[str] = set()
    for row in rows:
        identity = str(row["checkpoint_identity_sha256"])
        if identity in seen_identities:
            raise AiInventoryScoringIntegrityError(
                "checkpoint identity appears more than once"
            )
        seen_identities.add(identity)
    decisions = {str(row["decision_sha256"]) for row in rows}
    for decision_sha in decisions:
        state = _state_for_decision(rows, decision_sha)
        if state["pending_count"] != 1:
            raise AiInventoryScoringIntegrityError(
                "decision must have exactly one pending registration"
            )
        if state["terminal_count"] > 1:
            raise AiInventoryScoringIntegrityError(
                "decision has multiple terminal scoring outcomes"
            )
        if state["feedback_pending_count"] > 1 or state["feedback_ack_count"] > 1:
            raise AiInventoryScoringIntegrityError(
                "decision has duplicate feedback lifecycle rows"
            )
        if state["feedback_pending"] is not None and state["scored"] is None:
            raise AiInventoryScoringIntegrityError(
                "feedback was requested before scoring"
            )
        if state["feedback_ack"] is not None and state["feedback_pending"] is None:
            raise AiInventoryScoringIntegrityError(
                "feedback acknowledgement lacks its request"
            )
        pending = state["pending"]
        terminal = state["terminal"]
        if terminal is not None and terminal["payload"].get(
            "pending_checkpoint_sha256"
        ) != pending["checkpoint_sha256"]:
            raise AiInventoryScoringIntegrityError(
                "terminal scoring row does not bind its pending checkpoint"
            )
        feedback_pending = state["feedback_pending"]
        if feedback_pending is not None:
            scored = state["scored"]
            assert scored is not None
            if (
                feedback_pending["payload"].get("scored_checkpoint_sha256")
                != scored["checkpoint_sha256"]
            ):
                raise AiInventoryScoringIntegrityError(
                    "feedback request does not bind its scored checkpoint"
                )
        feedback_ack = state["feedback_ack"]
        if feedback_ack is not None:
            assert feedback_pending is not None
            request = feedback_pending["payload"]
            receipt = feedback_ack["payload"]["receipt"]
            for field in (
                "feedback_identity_sha256",
                "candidate_id",
                "decision_sha256",
                "evaluation_sha256",
            ):
                if receipt.get(field) != request.get(field):
                    raise AiInventoryScoringIntegrityError(
                        "feedback acknowledgement does not bind its request"
                    )


def _state_for_decision(
    rows: Sequence[Mapping[str, Any]], decision_sha: str
) -> dict[str, Any]:
    selected = [row for row in rows if row["decision_sha256"] == decision_sha]
    pending = [row for row in selected if row["event_type"] == EVENT_DECISION_PENDING]
    scored = [row for row in selected if row["event_type"] == EVENT_SCORED]
    unscored = [row for row in selected if row["event_type"] == EVENT_UNSCORED]
    feedback_pending = [
        row for row in selected if row["event_type"] == EVENT_FEEDBACK_PENDING
    ]
    feedback_ack = [
        row for row in selected if row["event_type"] == EVENT_FEEDBACK_ACKNOWLEDGED
    ]
    return {
        "pending": pending[0] if len(pending) == 1 else None,
        "pending_count": len(pending),
        "scored": scored[0] if len(scored) == 1 else None,
        "unscored": unscored[0] if len(unscored) == 1 else None,
        "terminal": (
            scored[0]
            if len(scored) == 1
            else unscored[0]
            if len(unscored) == 1
            else None
        ),
        "terminal_count": len(scored) + len(unscored),
        "feedback_pending": (
            feedback_pending[0] if len(feedback_pending) == 1 else None
        ),
        "feedback_pending_count": len(feedback_pending),
        "feedback_ack": feedback_ack[0] if len(feedback_ack) == 1 else None,
        "feedback_ack_count": len(feedback_ack),
    }


def _read_evaluations(
    path: Path,
    *,
    observed_no_later_than: datetime,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    from quant_rabbit.dojo_ai_inventory_evaluator import (
        validate_ai_inventory_evaluation_ledger,
    )

    validation = validate_ai_inventory_evaluation_ledger(path)
    if not validation.get("valid"):
        raise AiInventoryScoringIntegrityError(
            "evaluation ledger failed full validation"
        )
    rows = _read_jsonl_locked(path, missing_ok=False)
    if (
        len(rows) != validation["row_count"]
        or (
            rows[-1]["evaluation_sha256"] if rows else None
        )
        != validation["terminal_evaluation_sha256"]
        or any(
            _parse_utc(row.get("scored_at_utc"), "evaluation scored_at")
            > observed_no_later_than
            for row in rows
        )
    ):
        raise AiInventoryScoringIntegrityError(
            "evaluation ledger changed during snapshot"
        )
    return rows


def _evaluation_for_scored_checkpoint(
    evaluations: Sequence[Mapping[str, Any]],
    scored: Mapping[str, Any],
) -> dict[str, Any]:
    matches = [
        row
        for row in evaluations
        if row.get("evaluation_sha256") == scored["payload"]["evaluation_sha256"]
        and row.get("evaluation_identity_sha256")
        == scored["payload"]["evaluation_identity_sha256"]
    ]
    if len(matches) != 1:
        raise AiInventoryScoringIntegrityError(
            "scored checkpoint lacks one evaluation record"
        )
    return dict(matches[0])


def _feedback_request(
    decision: Mapping[str, Any],
    scored: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = decision.get("candidate_binding")
    if not isinstance(candidate, Mapping) or not _is_sha256(
        candidate.get("candidate_id")
    ):
        raise AiInventoryScoringIntegrityError(
            "decision candidate binding is invalid"
        )
    body = {
        "contract": SCORING_FEEDBACK_CONTRACT,
        "candidate_id": candidate["candidate_id"],
        "decision_sha256": decision["decision_sha256"],
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "scored_checkpoint_sha256": scored["checkpoint_sha256"],
        "feedback_not_before_utc": scored["recorded_at_utc"],
        "realized_outcome": evaluation["realized_outcome"],
        "realized_pl_jpy": evaluation["realized_pl_jpy"],
        "regime_correct": evaluation["regime_correct"],
        "regime_brier_score": evaluation["regime_brier_score"],
        "source_watermarks": evaluation["source_watermarks"],
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    return {
        **body,
        "feedback_identity_sha256": _sha256(
            _canonical_json(body).encode("utf-8")
        ),
    }


def _acknowledge_feedback(
    root: Path,
    *,
    request: Mapping[str, Any],
    receipt: Mapping[str, Any],
    recorded_at: datetime,
) -> tuple[dict[str, Any], bool]:
    normalized = _validate_feedback_receipt(receipt, no_later_than=recorded_at)
    for field in (
        "feedback_identity_sha256",
        "candidate_id",
        "decision_sha256",
        "evaluation_sha256",
    ):
        if normalized[field] != request[field]:
            raise AiInventoryScoringIntegrityError(
                f"feedback receipt {field} mismatch"
            )
    if _parse_utc(
        normalized["accepted_at_utc"], "feedback accepted_at"
    ) < _parse_utc(request["feedback_not_before_utc"], "feedback not-before"):
        raise AiInventoryScoringIntegrityError(
            "feedback receipt predates the durable scored checkpoint"
        )
    return _append_checkpoint(
        root,
        event_type=EVENT_FEEDBACK_ACKNOWLEDGED,
        decision_sha256=str(request["decision_sha256"]),
        payload={
            "status": "ACKNOWLEDGED",
            "feedback_identity_sha256": request["feedback_identity_sha256"],
            "receipt": normalized,
        },
        recorded_at=recorded_at,
    )


def _validate_feedback_receipt(
    value: Mapping[str, Any],
    *,
    no_later_than: datetime,
) -> dict[str, Any]:
    receipt = _snapshot_mapping(value, "feedback receipt")
    accepted_at = _parse_utc(receipt.get("accepted_at_utc"), "feedback accepted_at")
    if (
        set(receipt) != _FEEDBACK_RECEIPT_KEYS
        or receipt.get("contract") != SCORING_FEEDBACK_RECEIPT_CONTRACT
        or any(
            not _is_sha256(receipt.get(field))
            for field in (
                "feedback_identity_sha256",
                "candidate_id",
                "decision_sha256",
                "evaluation_sha256",
                "sink_event_sha256",
                "sink_ledger_tip_sha256",
            )
        )
        or not isinstance(receipt.get("sink_id"), str)
        or not receipt["sink_id"]
        or receipt.get("paper_only") is not True
        or receipt.get("order_authority") != "NONE"
        or receipt.get("live_permission") is not False
        or receipt.get("receipt_sha256") != feedback_receipt_sha256(receipt)
        or accepted_at > no_later_than
        or not _market_is_open(accepted_at)
    ):
        raise AiInventoryScoringIntegrityError("feedback receipt is invalid")
    return receipt


def _source_tips(
    decisions: Sequence[Mapping[str, Any]],
    broker_rows: Sequence[Mapping[str, Any]],
    quote_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    return {
        "decision_ledger_tip_sha256": (
            str(decisions[-1]["decision_sha256"])
            if decisions
            else GENESIS_DECISION_SHA256
        ),
        "broker_ledger_tip_sha256": (
            str(broker_rows[-1]["sha"]) if broker_rows else "0" * 64
        ),
        "quote_ledger_tip_sha256": (
            str(quote_rows[-1]["quote_sha256"])
            if quote_rows
            else GENESIS_QUOTE_SHA256
        ),
    }


def _read_jsonl_locked(path: Path, *, missing_ok: bool) -> list[dict[str, Any]]:
    if missing_ok and not path.exists():
        return []
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AiInventoryScoringIntegrityError(
            f"{path.name} cannot be opened"
        ) from exc
    with os.fdopen(descriptor, "rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            raw = handle.read(MAX_LEDGER_BYTES + 1)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return _decode_jsonl(raw, path.name)


def _open_locked_ledger(path: Path, *, create: bool) -> Any:
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        if not create:
            raise AiInventoryScoringIntegrityError(
                "scoring checkpoint ledger is missing"
            )
        create_flags = flags | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(path, create_flags, 0o600)
        except FileExistsError:
            descriptor = os.open(path, flags)
    except OSError as exc:
        raise AiInventoryScoringIntegrityError(
            "scoring checkpoint ledger cannot be opened"
        ) from exc
    handle = os.fdopen(descriptor, "r+b")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    info = os.fstat(handle.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_LEDGER_BYTES:
        _unlock_close(handle)
        raise AiInventoryScoringIntegrityError(
            "scoring checkpoint ledger is unsafe"
        )
    return handle


def _unlock_close(handle: Any) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _decode_checkpoint_bytes(raw: bytes) -> list[dict[str, Any]]:
    return _decode_jsonl(raw, SCORING_CHECKPOINT_LEDGER_NAME)


def _decode_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    if len(raw) > MAX_LEDGER_BYTES or (raw and not raw.endswith(b"\n")):
        raise AiInventoryScoringIntegrityError(f"{label} is oversized or truncated")
    lines = raw.splitlines()
    if len(lines) > MAX_LEDGER_ROWS:
        raise AiInventoryScoringIntegrityError(f"{label} exceeds the row limit")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, 1):
        if not line or len(line) > MAX_LINE_BYTES:
            raise AiInventoryScoringIntegrityError(
                f"{label} row {index} has invalid size"
            )
        row = _parse_json(line, f"{label} row {index}")
        if not isinstance(row, dict):
            raise AiInventoryScoringIntegrityError(
                f"{label} row {index} is not an object"
            )
        rows.append(row)
    return rows


def _read_regular_nofollow(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AiInventoryScoringIntegrityError(
            f"{path.name} cannot be opened"
        ) from exc
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum_bytes:
            raise AiInventoryScoringIntegrityError(
                f"{path.name} is not a bounded regular file"
            )
        raw = handle.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        raise AiInventoryScoringIntegrityError(f"{path.name} is oversized")
    return raw


def _parse_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AiInventoryScoringIntegrityError(f"{label} is invalid JSON") from exc


def _snapshot_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        raw = _canonical_json(value).encode("utf-8")
        snapshot = _parse_json(raw, label)
    except (TypeError, ValueError) as exc:
        raise AiInventoryScoringIntegrityError(
            f"{label} is not canonical JSON"
        ) from exc
    if not isinstance(snapshot, dict):
        raise AiInventoryScoringIntegrityError(f"{label} is not an object")
    return snapshot


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AiInventoryScoringIntegrityError(f"{label} is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AiInventoryScoringIntegrityError(
            f"{label} is not canonical UTC"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise AiInventoryScoringIntegrityError(f"{label} is not UTC")
    return parsed.astimezone(timezone.utc)


def _canonical_utc(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc)
    if normalized.microsecond:
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return normalized.isoformat(timespec="seconds").replace("+00:00", "Z")


def _require_market_open(value: datetime) -> None:
    if not _market_is_open(value):
        raise AiInventoryScoringMarketClosedError(
            "AI inventory scoring is disabled while FX is closed"
        )


def _market_is_open(value: datetime) -> bool:
    try:
        return compute_market_status(value).is_fx_open
    except Exception as exc:
        raise AiInventoryScoringLoopError(
            "FX market status is unavailable; scoring stopped"
        ) from exc


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_pair(value: object) -> bool:
    return isinstance(value, str) and _PAIR_RE.fullmatch(value) is not None


def _positive_number(value: object) -> bool:
    return _finite_number(value) and float(value) > 0


def _nonnegative_number(value: object) -> bool:
    return _positive_number(value) or value == 0


def _finite_number(value: object) -> bool:
    return (
        type(value) in {int, float}
        and value == value
        and float("-inf") < float(value) < float("inf")
    )


def _unit_interval(value: object) -> bool:
    return _finite_number(value) and 0.0 <= float(value) <= 1.0


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
