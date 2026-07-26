"""Fail-closed consumer for a future isolated DOJO paper-AI inventory room.

This adapter is deliberately not wired into any existing paper runner.  It
accepts one already validated, durably written AI decision and may apply it
only to the exact VirtualBroker implementation.  There are no live-broker
imports and no fallback path.

For a virtual close/reduction, the broker ledger sequence is:

``AI_INVENTORY_ACTION_RESERVED -> CLOSE -> AI_INVENTORY_ACTION_APPLIED``.

The reservation makes a crash conservative.  Recovery accepts only the exact
contiguous suffix bound to the decision's broker-ledger tip:

* ``RESERVED`` resumes the virtual close once;
* ``RESERVED -> CLOSE`` appends only the missing applied receipt; and
* an existing ``APPLIED`` receipt is returned without another mutation.

Every other suffix fails closed.

``BLOCK_NEW`` writes a durable gate receipt and cancels every same-room,
same-strategy pending virtual entry before APPLIED. It never closes an open
position. ``ALLOW_NEW_VIRTUAL`` writes a short-lived single-use permit receipt
for a later entry proxy; this consumer never opens a position itself.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_ai_inventory import (
    DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_ROLE,
    validate_inventory_decision,
    validate_inventory_decision_ledger,
)
from quant_rabbit.dojo_ai_inventory_producer import (
    verify_ai_inventory_producer_receipt,
)
from quant_rabbit.dojo_replay_lifecycle import (
    canonical_paper_ai_rooms_root,
    verify_paper_ai_inventory_launch_preflight,
)
from quant_rabbit.virtual_broker import VirtualBroker


ALLOWED_ACTIONS = frozenset(
    {
        "HOLD",
        "BLOCK_NEW",
        "ALLOW_NEW_VIRTUAL",
        "REDUCE_VIRTUAL",
        "CLOSE_VIRTUAL",
    }
)
GENESIS_SHA256 = "0" * 64

# Ninety seconds is the paper-AI execution-time quote freshness ceiling fixed
# by the inventory contract.  It is an integrity boundary for binding an AI
# decision to the same executable quote, not a strategy or profit parameter.
# A different allowance requires a versioned consumer contract.
MAX_CONSUME_QUOTE_AGE_SECONDS = 90


class InventoryConsumerError(RuntimeError):
    """A paper-AI decision cannot be consumed safely."""


class InventoryConsumerIntegrityError(InventoryConsumerError):
    """The decision, runtime evidence, or broker ledger is inconsistent."""


class InventoryReservationOutstandingError(InventoryConsumerError):
    """A prior reservation exists and must never be retried automatically."""


class InventoryDecisionAlreadyAppliedError(InventoryConsumerError):
    """The exact decision already has a durable applied receipt."""


def reconcile_inventory_checkpoint_suffix(
    broker: VirtualBroker,
    decision: Mapping[str, Any],
    broker_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replay one fully validated AI suffix onto an authenticated checkpoint.

    This is called only by the isolated broker owner during restart, after the
    checkpoint snapshot and quotes have been authenticated and restored.  It
    never appends a ledger row.  A durable ``CLOSE`` is re-applied to memory
    from its exact payload and decision-bound pre-state; it is never executed
    through :meth:`VirtualBroker.close_trade` a second time.
    """

    if type(broker) is not VirtualBroker:
        raise InventoryConsumerIntegrityError(
            "recovery requires an exact VirtualBroker instance"
        )
    row = _snapshot_mapping(decision, "recovery decision")
    _validate_decision_digest(row)
    runtime_guard = {
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
    }
    _validate_safety(row, runtime_guard)
    bound = _recovery_bound_from_decision(row)
    lifecycle = _validate_exact_decision_suffix(broker_rows, bound)
    if lifecycle["status"] == "NONE":
        raise InventoryConsumerIntegrityError(
            "checkpoint recovery has no AI inventory suffix"
        )
    _validate_live_position_before_action(broker, bound)
    close_row = lifecycle.get("close")
    if close_row is not None:
        _replay_close_payload(broker, close_row, bound)
        _validate_live_position_after_close(broker, bound)
    for cancel_row in lifecycle.get("cancels", []):
        _replay_cancel_payload(broker, cancel_row, bound)
    return lifecycle


def consume_inventory_decision(
    decision: Mapping[str, Any],
    broker: VirtualBroker,
    runtime_evidence: Mapping[str, Any],
    *,
    decision_ledger_path: Path,
    producer_receipt_path: Path,
    repository_root: Path | None = None,
    candidate_lifecycle_ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Apply one validated decision to an isolated virtual broker.

    ``runtime_evidence`` must bind the exact room, session, candidate, policy,
    specification, broker ledger, state, snapshot, position, and executable
    quote observed by the decision writer.  This function intentionally
    requires explicit fields rather than treating a digest as a substitute for
    missing ownership or strategy identity.

    ``HOLD`` and ``ALLOW_NEW_VIRTUAL`` produce a durable applied receipt
    without a broker mutation. ``BLOCK_NEW`` durably cancels every matching
    pending virtual entry, but never closes an open position. A later isolated
    entry proxy may consume an ``ALLOW_NEW_VIRTUAL`` receipt exactly once.
    """

    if type(broker) is not VirtualBroker:
        raise InventoryConsumerIntegrityError(
            "consumer requires an exact VirtualBroker instance"
        )
    decision_row = _snapshot_mapping(decision, "decision")
    runtime = _snapshot_runtime_evidence(runtime_evidence)
    canonical_repository_root = _canonical_repository_root(
        repository_root=repository_root,
        candidate_lifecycle_ledger_path=candidate_lifecycle_ledger_path,
    )
    # The consumer authors this clock.  A caller-supplied time could backdate a
    # weekend call into an open interval and defeat the market-closed guard.
    consume_at = _utc_now()

    _validate_safety(decision_row, runtime)
    _validate_decision_digest(decision_row)
    _validate_market_open(
        _parse_utc(decision_row.get("cutoff_at_utc")),
        context="decision cutoff",
    )
    _validate_market_open(consume_at)
    _validate_room_isolation(
        decision_row,
        runtime,
        broker,
        repository_root=canonical_repository_root,
        decision_ledger_path=decision_ledger_path,
    )
    _verify_producer_receipt_binding(
        decision_row,
        runtime,
        producer_receipt_path=producer_receipt_path,
    )
    ledger_path = _broker_ledger_path(broker)
    observed_rows = _read_broker_ledger(ledger_path)
    recovering = any(
        row["event"] == "AI_INVENTORY_ACTION_RESERVED"
        and row["payload"].get("decision_sha256")
        == decision_row["decision_sha256"]
        for row in observed_rows
    )
    bound = _validate_bindings(
        decision_row,
        runtime,
        broker,
        consume_at,
        recovery=recovering,
    )

    with _validated_source_ledgers(
        decision_row,
        repository_root=canonical_repository_root,
        decision_ledger_path=decision_ledger_path,
    ), _consumer_lock(ledger_path):
        before_rows = _read_broker_ledger(ledger_path)
        _validate_broker_tip(broker, before_rows)
        lifecycle = _validate_exact_decision_suffix(before_rows, bound)
        if lifecycle["status"] == "APPLIED":
            applied = lifecycle["applied"]
            return {
                **applied["payload"],
                "applied_receipt_sha256": applied["sha"],
                "broker_ledger_terminal_sha256": applied["sha"],
            }

        reservation = lifecycle.get("reservation")
        close_row = lifecycle.get("close")
        cancel_rows = list(lifecycle.get("cancels", []))
        if reservation is None:
            _validate_unconsumed(
                before_rows,
                decision_sha256=bound["decision_sha256"],
                room_id=bound["room_id"],
                position_id=bound["position_id"],
            )
            _validate_live_position_before_action(broker, bound)
            reservation_payload = _reservation_payload(
                bound, consume_at_utc=_canonical_utc(consume_at)
            )
            _broker_log(
                broker, "AI_INVENTORY_ACTION_RESERVED", reservation_payload
            )
            reserved_rows = _read_broker_ledger(ledger_path)
            _validate_broker_tip(broker, reserved_rows)
            if len(reserved_rows) != len(before_rows) + 1:
                raise InventoryConsumerIntegrityError(
                    "reservation was not the only broker-ledger append"
                )
            reservation = reserved_rows[-1]
            if (
                reservation["event"] != "AI_INVENTORY_ACTION_RESERVED"
                or reservation["payload"] != reservation_payload
            ):
                raise InventoryConsumerIntegrityError(
                    "broker did not append the exact reservation"
                )
        else:
            reservation_payload = dict(reservation["payload"])

        if close_row is None and bound["action"] in {
            "REDUCE_VIRTUAL",
            "CLOSE_VIRTUAL",
        }:
            _validate_live_position_before_action(broker, bound)
            close_units = (
                bound["virtual_units"] if bound["action"] == "REDUCE_VIRTUAL" else None
            )
            broker.close_trade(bound["position_id"], units=close_units)
            closed_rows = _read_broker_ledger(ledger_path)
            _validate_broker_tip(broker, closed_rows)
            if closed_rows[-2]["sha"] != reservation["sha"]:
                raise InventoryConsumerIntegrityError(
                    "virtual close was not adjacent to its reservation"
                )
            close_row = closed_rows[-1]
            _validate_close_row(close_row, reservation, bound)
        elif close_row is not None:
            _validate_live_position_after_close(broker, bound)

        if bound["action"] == "BLOCK_NEW":
            durable_ids = {
                str(row["payload"]["order_id"]) for row in cancel_rows
            }
            matching_order_ids = sorted(
                order_id
                for order_id, order in broker.orders.items()
                if order.strategy_tag == bound["strategy_tag"]
                and order_id not in durable_ids
            )
            for order_id in matching_order_ids:
                broker.cancel_order(order_id)
                cancelled_rows = _read_broker_ledger(ledger_path)
                _validate_broker_tip(broker, cancelled_rows)
                cancel_row = cancelled_rows[-1]
                expected_previous = (
                    cancel_rows[-1]["sha"]
                    if cancel_rows
                    else reservation["sha"]
                )
                _validate_cancel_row(
                    cancel_row,
                    previous_sha256=expected_previous,
                    bound=bound,
                )
                cancel_rows.append(cancel_row)

        applied_payload = {
            **reservation_payload,
            "reservation_sha256": reservation["sha"],
            "close_sha256": close_row["sha"] if close_row is not None else None,
            "realized_pl_jpy": (
                float(close_row["payload"]["pl_jpy"])
                if close_row is not None
                else None
            ),
            "cancelled_order_ids": [
                row["payload"]["order_id"] for row in cancel_rows
            ],
            "cancel_sha256s": [row["sha"] for row in cancel_rows],
            "block_new": bound["action"] == "BLOCK_NEW",
            "allow_new_virtual": bound["action"] == "ALLOW_NEW_VIRTUAL",
            "single_use_entry_permit": bound["action"] == "ALLOW_NEW_VIRTUAL",
            "entry_proxy_consumed": (
                False if bound["action"] == "ALLOW_NEW_VIRTUAL" else None
            ),
            "status": "APPLIED",
        }
        _broker_log(broker, "AI_INVENTORY_ACTION_APPLIED", applied_payload)
        final_rows = _read_broker_ledger(ledger_path)
        _validate_broker_tip(broker, final_rows)
        expected_previous = (
            close_row["sha"]
            if close_row is not None
            else (
                cancel_rows[-1]["sha"]
                if cancel_rows
                else reservation["sha"]
            )
        )
        if final_rows[-1]["prev_sha"] != expected_previous:
            raise InventoryConsumerIntegrityError(
                "applied receipt was not the next broker-ledger append"
            )
        applied = final_rows[-1]
        if (
            applied["event"] != "AI_INVENTORY_ACTION_APPLIED"
            or applied["payload"] != applied_payload
        ):
            raise InventoryConsumerIntegrityError(
                "broker did not append the exact applied receipt"
            )

        return {
            **applied_payload,
            "applied_receipt_sha256": applied["sha"],
            "broker_ledger_terminal_sha256": applied["sha"],
        }


def _snapshot_runtime_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InventoryConsumerIntegrityError("runtime evidence must be a mapping")
    dedicated_root = value.get("dedicated_root")
    if not isinstance(dedicated_root, Path):
        raise InventoryConsumerIntegrityError(
            "runtime dedicated_root must be an explicit Path"
        )
    serializable = dict(value)
    serializable.pop("dedicated_root", None)
    snapshot = _snapshot_mapping(serializable, "runtime evidence")
    snapshot["dedicated_root"] = dedicated_root
    return snapshot


def _canonical_repository_root(
    *,
    repository_root: Path | None,
    candidate_lifecycle_ledger_path: Path | None,
) -> Path:
    """Resolve only the canonical research-root relationship.

    ``candidate_lifecycle_ledger_path`` remains a compatibility input for the
    isolated broker service.  Its contents are never used as authorization.
    """

    if repository_root is not None and not isinstance(repository_root, Path):
        raise InventoryConsumerIntegrityError(
            "repository_root must be an explicit Path"
        )
    if candidate_lifecycle_ledger_path is not None and not isinstance(
        candidate_lifecycle_ledger_path, Path
    ):
        raise InventoryConsumerIntegrityError(
            "candidate_lifecycle_ledger_path must be an explicit Path"
        )
    inferred: Path | None = None
    if candidate_lifecycle_ledger_path is not None:
        try:
            supplied = candidate_lifecycle_ledger_path.resolve(strict=True)
            inferred = supplied.parents[3]
            expected = (
                inferred
                / "research/data/dojo_autonomous_improvement_v1"
                / "candidate_ledger.jsonl"
            ).resolve(strict=True)
        except (IndexError, OSError) as exc:
            raise InventoryConsumerIntegrityError(
                "candidate lifecycle ledger is not canonical"
            ) from exc
        if supplied != expected:
            raise InventoryConsumerIntegrityError(
                "candidate lifecycle ledger is not canonical"
            )
    chosen = repository_root or inferred
    if chosen is None:
        raise InventoryConsumerIntegrityError("canonical repository root is required")
    try:
        resolved = chosen.resolve(strict=True)
    except OSError as exc:
        raise InventoryConsumerIntegrityError(
            "canonical repository root is unavailable"
        ) from exc
    if inferred is not None and resolved != inferred:
        raise InventoryConsumerIntegrityError(
            "repository and candidate lifecycle roots disagree"
        )
    return resolved


def _validate_room_isolation(
    decision: Mapping[str, Any],
    runtime: Mapping[str, Any],
    broker: VirtualBroker,
    *,
    repository_root: Path,
    decision_ledger_path: Path,
) -> None:
    if runtime.get("room_kind") != "paper-ai-inventory":
        raise InventoryConsumerIntegrityError(
            "consumer is restricted to paper-ai-inventory rooms"
        )
    session = _require_mapping(decision, "session_binding")
    experiment_id = session.get("experiment_id")
    room_id = session.get("room_id")
    for label, identifier in (
        ("experiment_id", experiment_id),
        ("room_id", room_id),
    ):
        if (
            not isinstance(identifier, str)
            or not identifier.startswith("paper-ai-inventory-")
            or Path(identifier).name != identifier
        ):
            raise InventoryConsumerIntegrityError(
                f"non-isolated AI room identifier: {label}"
            )

    dedicated_root = runtime["dedicated_root"]
    if not isinstance(repository_root, Path):
        raise InventoryConsumerIntegrityError(
            "repository_root must be an explicit Path"
        )
    if not isinstance(decision_ledger_path, Path):
        raise InventoryConsumerIntegrityError(
            "decision_ledger_path must be an explicit Path"
        )
    try:
        resolved_root = dedicated_root.resolve(strict=True)
        canonical_root = canonical_paper_ai_rooms_root(repository_root).resolve(
            strict=True
        )
        if resolved_root != canonical_root:
            raise ValueError("non-canonical paper-AI room root")
        resolved_room_root = (
            resolved_root / str(experiment_id) / str(room_id)
        ).resolve(strict=True)
        resolved_room_root.relative_to(resolved_root)
        resolved_ledger = _broker_ledger_path(broker).resolve(strict=True)
        resolved_decisions = decision_ledger_path.resolve(strict=True)
        relative_ledger = resolved_ledger.relative_to(resolved_room_root)
        relative_decisions = resolved_decisions.relative_to(resolved_room_root)
    except (OSError, ValueError) as exc:
        raise InventoryConsumerIntegrityError(
            "broker or decision ledger is outside the dedicated paper-AI room"
        ) from exc
    if not resolved_root.is_dir():
        raise InventoryConsumerIntegrityError(
            "dedicated paper-AI root is not a directory"
        )
    for label, relative_path in (
        ("broker ledger", relative_ledger),
        ("decision ledger", relative_decisions),
    ):
        if not relative_path.parts:
            raise InventoryConsumerIntegrityError(
                f"{label} path does not identify an isolated file"
            )


def _verify_producer_receipt_binding(
    decision: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    producer_receipt_path: Path,
) -> None:
    """Authenticate the exact durable AI output bound by the decision."""

    if not isinstance(producer_receipt_path, Path):
        raise InventoryConsumerIntegrityError(
            "producer_receipt_path must be an explicit Path"
        )
    session = _require_mapping(decision, "session_binding")
    ai_decision = _require_mapping(decision, "ai_decision_binding")
    dedicated_root = runtime.get("dedicated_root")
    if not isinstance(dedicated_root, Path):
        raise InventoryConsumerIntegrityError(
            "runtime dedicated_root must be an explicit Path"
        )
    try:
        room_root = (
            dedicated_root.resolve(strict=True)
            / str(session["experiment_id"])
            / str(session["room_id"])
        ).resolve(strict=True)
        receipt = verify_ai_inventory_producer_receipt(room_root, producer_receipt_path)
    except Exception as exc:
        raise InventoryConsumerIntegrityError(
            "producer receipt failed durable verification"
        ) from exc

    exact_fields = (
        (
            "receipt_sha256",
            receipt.get("receipt_sha256"),
            ai_decision.get("producer_receipt_sha256"),
        ),
        (
            "producer_id",
            receipt.get("producer_id"),
            ai_decision.get("producer_id"),
        ),
        ("model_id", receipt.get("model_id"), ai_decision.get("model_id")),
        (
            "evidence_packet_sha256",
            receipt.get("evidence_packet_sha256"),
            ai_decision.get("evidence_packet_sha256"),
        ),
        (
            "request_sha256",
            receipt.get("request_sha256"),
            ai_decision.get("request_sha256"),
        ),
        (
            "response_sha256",
            receipt.get("response_sha256"),
            ai_decision.get("response_sha256"),
        ),
        ("action", receipt.get("action"), decision.get("action")),
        (
            "reason_code",
            receipt.get("reason_code"),
            decision.get("reason_code"),
        ),
        ("reason", receipt.get("reason"), decision.get("reason")),
        (
            "virtual_units",
            receipt.get("virtual_units"),
            decision.get("virtual_units"),
        ),
        (
            "confidence",
            receipt.get("confidence"),
            decision.get("confidence"),
        ),
        (
            "produced_at_utc",
            receipt.get("produced_at_utc"),
            ai_decision.get("produced_at_utc"),
        ),
    )
    for name, receipted, decided in exact_fields:
        if receipted != decided or type(receipted) is not type(decided):
            raise InventoryConsumerIntegrityError(
                f"producer receipt/decision mismatch: {name}"
            )


@contextmanager
def _validated_source_ledgers(
    decision: Mapping[str, Any],
    *,
    repository_root: Path,
    decision_ledger_path: Path,
) -> Iterator[None]:
    if not isinstance(decision_ledger_path, Path):
        raise InventoryConsumerIntegrityError(
            "decision_ledger_path must be an explicit Path"
        )
    if not isinstance(repository_root, Path):
        raise InventoryConsumerIntegrityError(
            "repository_root must be an explicit Path"
        )
    try:
        decision_handle = decision_ledger_path.open("rb")
    except OSError as exc:
        raise InventoryConsumerIntegrityError(
            "decision ledger is absent or unreadable"
        ) from exc
    try:
        fcntl.flock(decision_handle.fileno(), fcntl.LOCK_SH)
        validation = validate_inventory_decision_ledger(decision_ledger_path)
        if not validation.get("valid") or validation.get("row_count", 0) < 1:
            raise InventoryConsumerIntegrityError(
                "decision ledger failed full validation"
            )
        decision_handle.seek(0)
        try:
            decision_rows = [
                json.loads(line)
                for line in decision_handle.read().decode("utf-8").splitlines()
                if line.strip()
            ]
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise InventoryConsumerIntegrityError(
                "decision ledger rows cannot be reconstructed"
            ) from exc
        if (
            decision_rows[-1] != dict(decision)
            or decision_rows[-1].get("decision_sha256")
            != decision.get("decision_sha256")
            or validation.get("terminal_decision_sha256")
            != decision.get("decision_sha256")
            or validation.get("row_count") != decision.get("sequence")
        ):
            raise InventoryConsumerIntegrityError(
                "decision is not the exact terminal ledger row"
            )

        session = _require_mapping(decision, "session_binding")
        try:
            launch_preflight = verify_paper_ai_inventory_launch_preflight(
                repository_root,
                experiment_id=str(session.get("experiment_id") or ""),
                room_id=str(session.get("room_id") or ""),
            )
        except Exception as exc:
            raise InventoryConsumerIntegrityError(
                "canonical launch preflight failed full provenance validation"
            ) from exc
        _validate_launch_preflight(decision, launch_preflight)
        yield
    finally:
        fcntl.flock(decision_handle.fileno(), fcntl.LOCK_UN)
        decision_handle.close()


def _validate_launch_preflight(
    decision: Mapping[str, Any],
    token: Mapping[str, Any],
) -> None:
    session = _require_mapping(decision, "session_binding")
    candidate = _require_mapping(decision, "candidate_binding")
    policy = _require_mapping(decision, "policy_binding")
    spec = _require_mapping(decision, "spec_binding")
    lifecycle = _require_mapping(decision, "lifecycle_binding")
    candidate_id = candidate.get("candidate_id")
    if candidate.get("candidate_sha256") != candidate_id:
        raise InventoryConsumerIntegrityError(
            "candidate digest is not its canonical candidate id"
        )
    exact = (
        ("candidate_id", token.get("candidate_id"), candidate_id),
        ("spec_sha256", token.get("spec_sha256"), spec.get("spec_sha256")),
        (
            "policy_sha256",
            token.get("policy_sha256"),
            policy.get("policy_sha256"),
        ),
        (
            "experiment_id",
            token.get("experiment_id"),
            session.get("experiment_id"),
        ),
        ("room_id", token.get("room_id"), session.get("room_id")),
        (
            "paper_eligible_event_sha256",
            token.get("paper_eligible_event_sha256"),
            lifecycle.get("paper_eligible_event_sha256"),
        ),
        (
            "candidate_lifecycle_ledger_tip_sha256",
            token.get("candidate_lifecycle_ledger_tip_sha256"),
            lifecycle.get("candidate_lifecycle_ledger_tip_sha256"),
        ),
    )
    for field, proven, decided in exact:
        if proven != decided:
            raise InventoryConsumerIntegrityError(
                f"launch preflight/decision mismatch: {field}"
            )
    future_window = token.get("future_window")
    if not isinstance(future_window, Mapping):
        raise InventoryConsumerIntegrityError(
            "launch preflight future window is missing"
        )
    cutoff = _parse_utc(decision.get("cutoff_at_utc"))
    start = _parse_utc(future_window.get("start_utc"))
    end = _parse_utc(future_window.get("end_utc"))
    if not start <= cutoff < end:
        raise InventoryConsumerIntegrityError(
            "decision cutoff is outside the authorized future window"
        )


def _validate_safety(decision: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
    required_decision = {
        "contract": DOJO_AI_INVENTORY_DECISION_CONTRACT,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
    }
    required_runtime = {
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
    }
    for key, expected in required_decision.items():
        if decision.get(key) != expected or type(decision.get(key)) is not type(
            expected
        ):
            raise InventoryConsumerIntegrityError(f"unsafe decision invariant: {key}")
    for key, expected in required_runtime.items():
        if runtime.get(key) != expected or type(runtime.get(key)) is not type(expected):
            raise InventoryConsumerIntegrityError(f"unsafe runtime invariant: {key}")


def _validate_decision_digest(decision: Mapping[str, Any]) -> None:
    issues = validate_inventory_decision(decision)
    if issues:
        raise InventoryConsumerIntegrityError(
            "decision record failed V1 validation: " + "; ".join(issues)
        )
    stored = decision.get("decision_sha256")
    if not _is_sha256(stored):
        raise InventoryConsumerIntegrityError("invalid decision_sha256")
    body = {key: value for key, value in decision.items() if key != "decision_sha256"}
    if _sha256(body) != stored:
        raise InventoryConsumerIntegrityError("decision_sha256 mismatch")
    if not _is_sha256(decision.get("decision_identity_sha256")):
        raise InventoryConsumerIntegrityError("invalid decision_identity_sha256")
    recorded_at = _parse_utc(decision.get("recorded_at_utc"))
    cutoff = _parse_utc(decision.get("cutoff_at_utc"))
    if recorded_at < cutoff:
        raise InventoryConsumerIntegrityError("decision was recorded before its cutoff")


def _validate_market_open(
    consume_at: datetime, *, context: str = "consume time"
) -> None:
    try:
        status = compute_market_status(consume_at)
    except Exception as exc:
        raise InventoryConsumerIntegrityError(
            f"FX market status unavailable at {context}"
        ) from exc
    if not status.is_fx_open:
        raise InventoryConsumerIntegrityError(
            f"AI inventory {context} is disabled while FX is closed"
        )


def _validate_bindings(
    decision: Mapping[str, Any],
    runtime: Mapping[str, Any],
    broker: VirtualBroker,
    consume_at: datetime,
    *,
    recovery: bool = False,
) -> dict[str, Any]:
    action = decision.get("action")
    if action not in ALLOWED_ACTIONS:
        raise InventoryConsumerIntegrityError("unsupported inventory action")

    session = _require_mapping(decision, "session_binding")
    candidate = _require_mapping(decision, "candidate_binding")
    policy = _require_mapping(decision, "policy_binding")
    spec = _require_mapping(decision, "spec_binding")
    lifecycle = _require_mapping(decision, "lifecycle_binding")
    ai_decision = _require_mapping(decision, "ai_decision_binding")
    ledger = _require_mapping(decision, "ledger_binding")
    state = _require_mapping(decision, "state_binding")
    snapshot = _require_mapping(decision, "snapshot_binding")
    position_binding = _require_mapping(decision, "position_binding")
    quote_binding = _require_mapping(decision, "quote_binding")
    runtime_position = _require_mapping(runtime, "position")
    runtime_quote = _require_mapping(runtime, "quote")
    runtime_ai_decision = _require_mapping(runtime, "ai_decision_binding")
    admission = decision.get("admission_binding")
    runtime_admission = runtime.get("admission_binding")
    if action == "ALLOW_NEW_VIRTUAL":
        if not isinstance(admission, Mapping) or not isinstance(
            runtime_admission, Mapping
        ):
            raise InventoryConsumerIntegrityError(
                "ALLOW_NEW_VIRTUAL requires an exact admission binding"
            )
        admission = _snapshot_mapping(admission, "admission binding")
        runtime_admission = _snapshot_mapping(
            runtime_admission, "runtime admission binding"
        )
        if admission != runtime_admission:
            raise InventoryConsumerIntegrityError("binding mismatch: admission_binding")
    elif admission is not None or runtime_admission is not None:
        raise InventoryConsumerIntegrityError(
            "non-admission action must bind null admission evidence"
        )

    exact_bindings = (
        ("room_id", session.get("room_id"), runtime.get("room_id")),
        (
            "experiment_id",
            session.get("experiment_id"),
            runtime.get("experiment_id"),
        ),
        (
            "session_contract_sha256",
            session.get("session_contract_sha256"),
            runtime.get("session_contract_sha256"),
        ),
        (
            "candidate_id",
            candidate.get("candidate_id"),
            runtime.get("candidate_id"),
        ),
        (
            "candidate_sha256",
            candidate.get("candidate_sha256"),
            runtime.get("candidate_sha256"),
        ),
        ("policy_id", policy.get("policy_id"), runtime.get("policy_id")),
        (
            "policy_sha256",
            policy.get("policy_sha256"),
            runtime.get("policy_sha256"),
        ),
        ("spec_id", spec.get("spec_id"), runtime.get("spec_id")),
        (
            "spec_sha256",
            spec.get("spec_sha256"),
            runtime.get("spec_sha256"),
        ),
        (
            "paper_eligible_event_sha256",
            lifecycle.get("paper_eligible_event_sha256"),
            runtime.get("paper_eligible_event_sha256"),
        ),
        (
            "candidate_lifecycle_ledger_tip_sha256",
            lifecycle.get("candidate_lifecycle_ledger_tip_sha256"),
            runtime.get("candidate_lifecycle_ledger_tip_sha256"),
        ),
        (
            "ai_decision_binding",
            ai_decision,
            runtime_ai_decision,
        ),
        ("ledger_sha256", ledger.get("sha256"), runtime.get("ledger_sha256")),
        ("state_sha256", state.get("sha256"), runtime.get("state_sha256")),
        (
            "snapshot_sha256",
            snapshot.get("sha256"),
            runtime.get("snapshot_sha256"),
        ),
        (
            "position_id",
            position_binding.get("position_id"),
            runtime_position.get("position_id"),
        ),
        ("pair", position_binding.get("pair"), runtime_position.get("pair")),
        ("side", position_binding.get("side"), runtime_position.get("side")),
        ("units", position_binding.get("units"), runtime_position.get("units")),
        (
            "strategy_tag",
            position_binding.get("strategy_tag"),
            runtime_position.get("strategy_tag"),
        ),
        (
            "entry_context_sha256",
            position_binding.get("entry_context_sha256"),
            runtime_position.get("entry_context_sha256"),
        ),
        (
            "position_sha256",
            position_binding.get("sha256"),
            runtime_position.get("sha256"),
        ),
        ("quote_pair", quote_binding.get("pair"), runtime_quote.get("pair")),
        ("quote_bid", quote_binding.get("bid"), runtime_quote.get("bid")),
        ("quote_ask", quote_binding.get("ask"), runtime_quote.get("ask")),
        (
            "quote_observed_at_utc",
            quote_binding.get("observed_at_utc"),
            runtime_quote.get("observed_at_utc"),
        ),
        (
            "quote_sha256",
            quote_binding.get("sha256"),
            runtime_quote.get("sha256"),
        ),
    )
    for name, decided, current in exact_bindings:
        if decided is None or current is None or decided != current:
            raise InventoryConsumerIntegrityError(f"binding mismatch: {name}")

    for digest_name, digest in (
        (
            "session_contract_sha256",
            session.get("session_contract_sha256"),
        ),
        ("candidate_sha256", candidate.get("candidate_sha256")),
        ("policy_sha256", policy.get("policy_sha256")),
        ("spec_sha256", spec.get("spec_sha256")),
        (
            "paper_eligible_event_sha256",
            lifecycle.get("paper_eligible_event_sha256"),
        ),
        (
            "candidate_lifecycle_ledger_tip_sha256",
            lifecycle.get("candidate_lifecycle_ledger_tip_sha256"),
        ),
        ("ai_request_sha256", ai_decision.get("request_sha256")),
        ("ai_response_sha256", ai_decision.get("response_sha256")),
        (
            "ai_producer_receipt_sha256",
            ai_decision.get("producer_receipt_sha256"),
        ),
        (
            "ai_evidence_packet_sha256",
            ai_decision.get("evidence_packet_sha256"),
        ),
        ("ledger_sha256", ledger.get("sha256")),
        ("state_sha256", state.get("sha256")),
        ("snapshot_sha256", snapshot.get("sha256")),
        ("position_sha256", position_binding.get("sha256")),
        ("entry_context_sha256", position_binding.get("entry_context_sha256")),
        ("quote_sha256", quote_binding.get("sha256")),
    ):
        if not _is_sha256(digest):
            raise InventoryConsumerIntegrityError(f"invalid {digest_name}")

    pair = position_binding["pair"]
    if quote_binding["pair"] != pair:
        raise InventoryConsumerIntegrityError("position/quote pair mismatch")
    side = position_binding["side"]
    units = position_binding["units"]
    flat_gate = (
        action in {"BLOCK_NEW", "ALLOW_NEW_VIRTUAL"} and side == "FLAT" and units == 0.0
    )
    if flat_gate:
        if position_binding["position_id"] != f"FLAT:{pair}":
            raise InventoryConsumerIntegrityError(
                "flat entry gate has an invalid position identity"
            )
    else:
        if action in {"BLOCK_NEW", "ALLOW_NEW_VIRTUAL"}:
            raise InventoryConsumerIntegrityError(
                "entry gate requires a flat position binding"
            )
        if side not in {"LONG", "SHORT"}:
            raise InventoryConsumerIntegrityError("invalid position side")
        units = _positive_finite(units, "position units")
    virtual_units = decision.get("virtual_units")
    if action in {"HOLD", "BLOCK_NEW", "ALLOW_NEW_VIRTUAL"}:
        if virtual_units is not None:
            raise InventoryConsumerIntegrityError(
                "non-close action must not carry virtual units"
            )
    elif action == "CLOSE_VIRTUAL":
        if _positive_integral_virtual_units(virtual_units) != units:
            raise InventoryConsumerIntegrityError(
                "CLOSE_VIRTUAL must bind the full position"
            )
    else:
        reduce_units = _positive_integral_virtual_units(virtual_units)
        if not reduce_units < units:
            raise InventoryConsumerIntegrityError(
                "REDUCE_VIRTUAL must be smaller than the position"
            )

    recorded_at = _parse_utc(decision.get("recorded_at_utc"))
    expires_at = _parse_utc(decision.get("expires_at_utc"))
    quote_at = _parse_utc(quote_binding.get("observed_at_utc"))
    if consume_at <= recorded_at:
        raise InventoryConsumerIntegrityError(
            "consume time must be after durable decision time"
        )
    if consume_at >= expires_at and not recovery:
        raise InventoryConsumerIntegrityError(
            "decision expired before virtual consumption"
        )
    quote_age = (consume_at - quote_at).total_seconds()
    if quote_age < 0:
        raise InventoryConsumerIntegrityError("quote is future-dated")
    if quote_age > MAX_CONSUME_QUOTE_AGE_SECONDS and not recovery:
        raise InventoryConsumerIntegrityError("quote is stale at consume time")

    last_quotes = getattr(broker, "last_quotes", None)
    if not isinstance(last_quotes, Mapping):
        raise InventoryConsumerIntegrityError("broker does not expose virtual quotes")
    broker_quote = last_quotes.get(pair)
    expected_quote = (
        quote_binding["bid"],
        quote_binding["ask"],
        quote_binding["observed_at_utc"],
    )
    if broker_quote != expected_quote:
        raise InventoryConsumerIntegrityError(
            "broker quote does not match the bound executable quote"
        )

    bound = {
        "action": action,
        "virtual_units": virtual_units,
        "confidence": decision["confidence"],
        "decision_sha256": decision["decision_sha256"],
        "decision_identity_sha256": decision["decision_identity_sha256"],
        "room_id": session["room_id"],
        "session_id": session["experiment_id"],
        "candidate_id": candidate["candidate_id"],
        "policy_id": policy["policy_id"],
        "spec_id": spec["spec_id"],
        "ai_producer_id": ai_decision["producer_id"],
        "ai_model_id": ai_decision["model_id"],
        "ai_request_sha256": ai_decision["request_sha256"],
        "ai_response_sha256": ai_decision["response_sha256"],
        "ai_evidence_packet_sha256": ai_decision["evidence_packet_sha256"],
        "ai_producer_receipt_sha256": ai_decision["producer_receipt_sha256"],
        "ai_produced_at_utc": ai_decision["produced_at_utc"],
        "admission_binding": (
            _snapshot_mapping(admission, "admission binding")
            if isinstance(admission, Mapping)
            else None
        ),
        "ledger_sha256": ledger["sha256"],
        "position_id": position_binding["position_id"],
        "pair": pair,
        "side": side,
        "units": units,
        "strategy_tag": position_binding["strategy_tag"],
        "entry_context_sha256": position_binding["entry_context_sha256"],
        "quote": {
            "bid": quote_binding["bid"],
            "ask": quote_binding["ask"],
            "ts": quote_binding["observed_at_utc"],
        },
    }
    if not recovery:
        _validate_live_position_before_action(broker, bound)
    return bound


def _reservation_payload(
    bound: Mapping[str, Any], *, consume_at_utc: str
) -> dict[str, Any]:
    return {
        "decision_sha256": bound["decision_sha256"],
        "decision_identity_sha256": bound["decision_identity_sha256"],
        "action": bound["action"],
        "virtual_units": bound["virtual_units"],
        "confidence": bound["confidence"],
        "room_id": bound["room_id"],
        "session_id": bound["session_id"],
        "candidate_id": bound["candidate_id"],
        "policy_id": bound["policy_id"],
        "spec_id": bound["spec_id"],
        "ai_producer_id": bound["ai_producer_id"],
        "ai_model_id": bound["ai_model_id"],
        "ai_request_sha256": bound["ai_request_sha256"],
        "ai_response_sha256": bound["ai_response_sha256"],
        "ai_evidence_packet_sha256": bound["ai_evidence_packet_sha256"],
        "ai_producer_receipt_sha256": bound["ai_producer_receipt_sha256"],
        "ai_produced_at_utc": bound["ai_produced_at_utc"],
        "position_id": bound["position_id"],
        "pair": bound["pair"],
        "strategy_tag": bound["strategy_tag"],
        "admission_binding": bound["admission_binding"],
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "decision_contract": DOJO_AI_INVENTORY_DECISION_CONTRACT,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
        "consume_at_utc": consume_at_utc,
    }


def _validate_exact_decision_suffix(
    rows: list[dict[str, Any]], bound: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the only broker suffix that this decision may own."""

    bound_tip = bound["ledger_sha256"]
    if bound_tip == GENESIS_SHA256:
        start = 0
    else:
        matches = [index for index, row in enumerate(rows) if row["sha"] == bound_tip]
        if len(matches) != 1:
            raise InventoryConsumerIntegrityError(
                "decision-bound broker ledger tip is absent or ambiguous"
            )
        start = matches[0] + 1
    suffix = rows[start:]
    if not suffix:
        return {"status": "NONE"}
    if len(suffix) > 10_002:
        raise InventoryConsumerIntegrityError(
            "broker ledger has an unknown post-decision suffix"
        )

    reservation = suffix[0]
    if (
        reservation["event"] != "AI_INVENTORY_ACTION_RESERVED"
        or reservation["payload"].get("decision_sha256") != bound["decision_sha256"]
    ):
        raise InventoryConsumerIntegrityError(
            "broker ledger advanced outside the exact decision lifecycle"
        )
    consume_at = reservation["payload"].get("consume_at_utc")
    parsed_consume_at = _parse_utc(consume_at)
    if parsed_consume_at < _parse_utc(bound["ai_produced_at_utc"]):
        raise InventoryConsumerIntegrityError(
            "reservation predates the authenticated AI output"
        )
    if reservation["payload"] != _reservation_payload(
        bound, consume_at_utc=str(consume_at)
    ):
        raise InventoryConsumerIntegrityError(
            "reservation does not exactly bind the decision"
        )

    close_action = bound["action"] in {"REDUCE_VIRTUAL", "CLOSE_VIRTUAL"}
    close_row: dict[str, Any] | None = None
    cancel_rows: list[dict[str, Any]] = []
    applied: dict[str, Any] | None = None
    if close_action and len(suffix) >= 2:
        close_row = suffix[1]
        _validate_close_row(close_row, reservation, bound)
        if len(suffix) == 3:
            applied = suffix[2]
        elif len(suffix) > 3:
            raise InventoryConsumerIntegrityError(
                "close inventory action has an unknown ledger suffix"
            )
    elif bound["action"] == "BLOCK_NEW":
        previous = reservation["sha"]
        seen_order_ids: set[str] = set()
        for index, row in enumerate(suffix[1:], 1):
            if row["event"] == "AI_INVENTORY_ACTION_APPLIED":
                if index != len(suffix) - 1:
                    raise InventoryConsumerIntegrityError(
                        "BLOCK_NEW applied receipt is not terminal"
                    )
                applied = row
                break
            _validate_cancel_row(
                row,
                previous_sha256=previous,
                bound=bound,
            )
            order_id = str(row["payload"]["order_id"])
            if order_id in seen_order_ids:
                raise InventoryConsumerIntegrityError(
                    "BLOCK_NEW contains a duplicate order cancellation"
                )
            seen_order_ids.add(order_id)
            cancel_rows.append(row)
            previous = row["sha"]
    elif len(suffix) == 2:
        applied = suffix[1]
    elif len(suffix) > 2:
        raise InventoryConsumerIntegrityError(
            "non-close inventory action has an unknown ledger suffix"
        )

    if applied is not None:
        _validate_applied_row(
            applied,
            reservation,
            close_row,
            cancel_rows,
            bound,
        )
        return {
            "status": "APPLIED",
            "reservation": reservation,
            "close": close_row,
            "cancels": cancel_rows,
            "applied": applied,
        }
    return {
        "status": (
            "CLOSE_DURABLE"
            if close_row is not None
            else ("CANCELS_DURABLE" if cancel_rows else "RESERVED")
        ),
        "reservation": reservation,
        "close": close_row,
        "cancels": cancel_rows,
    }


def _validate_applied_row(
    applied: Mapping[str, Any],
    reservation: Mapping[str, Any],
    close_row: Mapping[str, Any] | None,
    cancel_rows: list[dict[str, Any]],
    bound: Mapping[str, Any],
) -> None:
    expected_previous = (
        close_row["sha"]
        if close_row is not None
        else (
            cancel_rows[-1]["sha"]
            if cancel_rows
            else reservation["sha"]
        )
    )
    if (
        applied.get("event") != "AI_INVENTORY_ACTION_APPLIED"
        or applied.get("prev_sha") != expected_previous
    ):
        raise InventoryConsumerIntegrityError(
            "applied receipt is not adjacent to its inventory action"
        )
    expected = {
        **reservation["payload"],
        "reservation_sha256": reservation["sha"],
        "close_sha256": close_row["sha"] if close_row is not None else None,
        "realized_pl_jpy": (
            float(close_row["payload"]["pl_jpy"])
            if close_row is not None
            else None
        ),
        "cancelled_order_ids": [
            row["payload"]["order_id"] for row in cancel_rows
        ],
        "cancel_sha256s": [row["sha"] for row in cancel_rows],
        "block_new": bound["action"] == "BLOCK_NEW",
        "allow_new_virtual": bound["action"] == "ALLOW_NEW_VIRTUAL",
        "single_use_entry_permit": bound["action"] == "ALLOW_NEW_VIRTUAL",
        "entry_proxy_consumed": (
            False if bound["action"] == "ALLOW_NEW_VIRTUAL" else None
        ),
        "status": "APPLIED",
    }
    if applied.get("payload") != expected:
        raise InventoryConsumerIntegrityError(
            "applied receipt does not exactly bind its reservation"
        )


def _validate_live_position_before_action(
    broker: VirtualBroker, bound: Mapping[str, Any]
) -> None:
    positions = getattr(broker, "positions", None)
    if not isinstance(positions, Mapping):
        raise InventoryConsumerIntegrityError(
            "broker does not expose virtual positions"
        )
    broker_position = positions.get(bound["position_id"])
    flat_gate = (
        bound["action"] in {"BLOCK_NEW", "ALLOW_NEW_VIRTUAL"}
        and bound["side"] == "FLAT"
        and bound["units"] == 0.0
    )
    if flat_gate:
        if broker_position is not None:
            raise InventoryConsumerIntegrityError(
                "flat entry gate unexpectedly resolves to a broker position"
            )
        return
    if broker_position is None:
        raise InventoryConsumerIntegrityError("bound virtual position is absent")
    fields = (
        ("pair", bound["pair"]),
        ("side", bound["side"]),
        ("units", bound["units"]),
        ("strategy_tag", bound["strategy_tag"]),
        ("entry_context_sha256", bound["entry_context_sha256"]),
    )
    for name, expected in fields:
        actual = getattr(broker_position, name, None)
        if name == "units":
            _validate_virtual_broker_units(actual, expected)
        elif actual != expected:
            raise InventoryConsumerIntegrityError(
                f"broker position mismatch: {name}"
            )


def _validate_live_position_after_close(
    broker: VirtualBroker, bound: Mapping[str, Any]
) -> None:
    positions = getattr(broker, "positions", None)
    if not isinstance(positions, Mapping):
        raise InventoryConsumerIntegrityError(
            "broker does not expose virtual positions"
        )
    position = positions.get(bound["position_id"])
    if bound["action"] == "CLOSE_VIRTUAL":
        if position is not None:
            raise InventoryConsumerIntegrityError(
                "recovered full close still has an open virtual position"
            )
        return
    if bound["action"] != "REDUCE_VIRTUAL" or position is None:
        raise InventoryConsumerIntegrityError(
            "recovered reduction has an invalid virtual position"
        )
    expected_units = bound["units"] - bound["virtual_units"]
    _validate_virtual_broker_units(position.units, expected_units)
    for field in ("pair", "side", "strategy_tag", "entry_context_sha256"):
        if getattr(position, field, None) != bound[field]:
            raise InventoryConsumerIntegrityError(
                f"recovered position mismatch: {field}"
            )


def _validate_close_row(
    close_row: Mapping[str, Any],
    reservation: Mapping[str, Any],
    bound: Mapping[str, Any],
) -> None:
    if close_row.get("event") != "CLOSE":
        raise InventoryConsumerIntegrityError(
            "virtual close did not append a CLOSE event"
        )
    if close_row.get("prev_sha") != reservation.get("sha"):
        raise InventoryConsumerIntegrityError(
            "CLOSE is not adjacent to its reservation"
        )
    payload = close_row.get("payload")
    if not isinstance(payload, Mapping):
        raise InventoryConsumerIntegrityError("CLOSE payload is invalid")
    expected_units = (
        bound["units"] if bound["action"] == "CLOSE_VIRTUAL" else bound["virtual_units"]
    )
    exact_fields = (
        ("trade_id", bound["position_id"]),
        ("units", expected_units),
        ("strategy_tag", bound["strategy_tag"]),
        ("entry_context_sha256", bound["entry_context_sha256"]),
        ("quote", bound["quote"]),
    )
    for field, expected in exact_fields:
        if payload.get(field) != expected:
            raise InventoryConsumerIntegrityError(f"CLOSE binding mismatch: {field}")


def _validate_cancel_row(
    cancel_row: Mapping[str, Any],
    *,
    previous_sha256: str,
    bound: Mapping[str, Any],
) -> None:
    if (
        cancel_row.get("event") != "ORDER_CANCEL"
        or cancel_row.get("prev_sha") != previous_sha256
    ):
        raise InventoryConsumerIntegrityError(
            "BLOCK_NEW cancellation is not adjacent to its reservation"
        )
    payload = cancel_row.get("payload")
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"order_id", "strategy_tag"}
        or not isinstance(payload.get("order_id"), str)
        or not payload["order_id"]
        or payload.get("strategy_tag") != bound["strategy_tag"]
    ):
        raise InventoryConsumerIntegrityError(
            "BLOCK_NEW cancellation binding is invalid"
        )


def _replay_cancel_payload(
    broker: VirtualBroker,
    cancel_row: Mapping[str, Any],
    bound: Mapping[str, Any],
) -> None:
    payload = cancel_row["payload"]
    order_id = payload["order_id"]
    order = broker.orders.get(order_id)
    if order is None:
        raise InventoryConsumerIntegrityError(
            "durable ORDER_CANCEL is absent from checkpoint inventory"
        )
    if order.strategy_tag != bound["strategy_tag"]:
        raise InventoryConsumerIntegrityError(
            "durable ORDER_CANCEL strategy binding mismatch"
        )
    del broker.orders[order_id]


def _recovery_bound_from_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    session = _require_mapping(decision, "session_binding")
    candidate = _require_mapping(decision, "candidate_binding")
    policy = _require_mapping(decision, "policy_binding")
    spec = _require_mapping(decision, "spec_binding")
    ai = _require_mapping(decision, "ai_decision_binding")
    ledger = _require_mapping(decision, "ledger_binding")
    position = _require_mapping(decision, "position_binding")
    quote = _require_mapping(decision, "quote_binding")
    return {
        "action": decision["action"],
        "virtual_units": decision["virtual_units"],
        "confidence": decision["confidence"],
        "decision_sha256": decision["decision_sha256"],
        "decision_identity_sha256": decision["decision_identity_sha256"],
        "room_id": session["room_id"],
        "session_id": session["experiment_id"],
        "candidate_id": candidate["candidate_id"],
        "policy_id": policy["policy_id"],
        "spec_id": spec["spec_id"],
        "ai_producer_id": ai["producer_id"],
        "ai_model_id": ai["model_id"],
        "ai_request_sha256": ai["request_sha256"],
        "ai_response_sha256": ai["response_sha256"],
        "ai_evidence_packet_sha256": ai["evidence_packet_sha256"],
        "ai_producer_receipt_sha256": ai["producer_receipt_sha256"],
        "ai_produced_at_utc": ai["produced_at_utc"],
        "admission_binding": (
            _snapshot_mapping(decision["admission_binding"], "admission binding")
            if isinstance(decision.get("admission_binding"), Mapping)
            else None
        ),
        "ledger_sha256": ledger["sha256"],
        "position_id": position["position_id"],
        "pair": position["pair"],
        "side": position["side"],
        "units": position["units"],
        "strategy_tag": position["strategy_tag"],
        "entry_context_sha256": position["entry_context_sha256"],
        "quote": {
            "bid": quote["bid"],
            "ask": quote["ask"],
            "ts": quote["observed_at_utc"],
        },
    }


def _replay_close_payload(
    broker: VirtualBroker,
    close_row: Mapping[str, Any],
    bound: Mapping[str, Any],
) -> None:
    """Rebuild the exact post-close state without writing another CLOSE."""

    payload = _require_mapping(close_row, "payload")
    position = broker.positions.get(bound["position_id"])
    if position is None:
        raise InventoryConsumerIntegrityError(
            "checkpoint position is absent before durable CLOSE replay"
        )
    close_units = _positive_finite(payload.get("units"), "CLOSE units")
    if close_units > position.units:
        raise InventoryConsumerIntegrityError(
            "durable CLOSE exceeds checkpoint position"
        )
    quote = payload.get("quote")
    if not isinstance(quote, Mapping):
        raise InventoryConsumerIntegrityError("durable CLOSE quote is invalid")
    current_quote = broker.last_quotes.get(position.pair)
    expected_quote = (quote.get("bid"), quote.get("ask"), quote.get("ts"))
    if current_quote != expected_quote:
        raise InventoryConsumerIntegrityError(
            "durable CLOSE quote differs from checkpoint"
        )
    price = quote["bid"] if position.side == "LONG" else quote["ask"]
    if payload.get("price") != price:
        raise InventoryConsumerIntegrityError(
            "durable CLOSE executable price mismatch"
        )
    diff = (
        price - position.entry_price
        if position.side == "LONG"
        else position.entry_price - price
    )
    realized = diff * close_units * broker._jpy_per_quote_unit(position.pair)
    realized -= broker._financing_jpy(position, str(quote["ts"])) * (
        close_units / position.units
    )
    recorded_pl = payload.get("pl_jpy")
    if (
        isinstance(recorded_pl, bool)
        or not isinstance(recorded_pl, (int, float))
        or not math.isfinite(float(recorded_pl))
        or float(recorded_pl) != round(realized, 2)
    ):
        raise InventoryConsumerIntegrityError(
            "durable CLOSE realized P/L mismatch"
        )
    broker.balance_jpy += realized
    if close_units >= position.units:
        del broker.positions[position.trade_id]
    else:
        position.units -= close_units


def _validate_unconsumed(
    rows: list[dict[str, Any]],
    *,
    decision_sha256: str,
    room_id: str,
    position_id: str,
) -> None:
    reservations = [
        row for row in rows if row["event"] == "AI_INVENTORY_ACTION_RESERVED"
    ]
    applied_rows = [
        row for row in rows if row["event"] == "AI_INVENTORY_ACTION_APPLIED"
    ]
    reservations_by_sha = {row["sha"]: row for row in reservations}
    applied_by_reservation: dict[str, dict[str, Any]] = {}
    for applied in applied_rows:
        reservation_sha256 = applied["payload"].get("reservation_sha256")
        reservation = reservations_by_sha.get(reservation_sha256)
        if reservation is None:
            raise InventoryConsumerIntegrityError(
                "applied receipt exists without its reservation"
            )
        if reservation_sha256 in applied_by_reservation:
            raise InventoryConsumerIntegrityError(
                "reservation has multiple applied receipts"
            )
        for binding in ("decision_sha256", "room_id", "position_id"):
            if applied["payload"].get(binding) != reservation["payload"].get(binding):
                raise InventoryConsumerIntegrityError(
                    f"applied receipt/reservation mismatch: {binding}"
                )
        applied_by_reservation[reservation_sha256] = applied

    exact_reservations = [
        row
        for row in reservations
        if row["payload"].get("decision_sha256") == decision_sha256
    ]
    exact_applied = [
        row
        for row in applied_rows
        if row["payload"].get("decision_sha256") == decision_sha256
    ]
    if exact_applied:
        if (
            len(exact_reservations) != 1
            or len(exact_applied) != 1
            or exact_applied[0]["payload"].get("reservation_sha256")
            != exact_reservations[0]["sha"]
        ):
            raise InventoryConsumerIntegrityError(
                "exact decision consumption history is ambiguous"
            )
        raise InventoryDecisionAlreadyAppliedError(
            "ALREADY_APPLIED: exact decision will not be executed again"
        )

    unresolved = [
        row
        for row in reservations
        if row["sha"] not in applied_by_reservation
        and row["payload"].get("room_id") == room_id
        and row["payload"].get("position_id") == position_id
    ]
    if unresolved:
        raise InventoryReservationOutstandingError(
            "room position already has a reservation; automatic retry is forbidden"
        )


def _broker_log(broker: VirtualBroker, event: str, payload: dict[str, Any]) -> None:
    logger = getattr(broker, "_log", None)
    if not callable(logger):
        raise InventoryConsumerIntegrityError(
            "broker lacks the virtual append-only logger"
        )
    logger(event, payload)


def _broker_ledger_path(broker: VirtualBroker) -> Path:
    path = getattr(broker, "ledger_path", None)
    if not isinstance(path, (str, os.PathLike)):
        raise InventoryConsumerIntegrityError("broker lacks an isolated ledger path")
    return Path(path)


def _validate_broker_tip(broker: VirtualBroker, rows: list[dict[str, Any]]) -> None:
    expected = _ledger_tip(rows)
    if getattr(broker, "_prev_sha", None) != expected:
        raise InventoryConsumerIntegrityError(
            "broker in-memory tip does not match its ledger"
        )


def _read_broker_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    expected_previous = GENESIS_SHA256
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.endswith("\n"):
                    raise InventoryConsumerIntegrityError(
                        f"truncated broker ledger at line {line_number}"
                    )
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise InventoryConsumerIntegrityError(
                        f"invalid broker ledger row at line {line_number}"
                    )
                if set(row) != {"ts_utc", "event", "payload", "prev_sha", "sha"}:
                    raise InventoryConsumerIntegrityError(
                        f"invalid broker ledger shape at line {line_number}"
                    )
                if (
                    not isinstance(row["event"], str)
                    or not isinstance(row["payload"], dict)
                    or row["prev_sha"] != expected_previous
                ):
                    raise InventoryConsumerIntegrityError(
                        f"invalid broker ledger chain at line {line_number}"
                    )
                body = {
                    key: row[key] for key in ("ts_utc", "event", "payload", "prev_sha")
                }
                if row["sha"] != _sha256(body):
                    raise InventoryConsumerIntegrityError(
                        f"invalid broker ledger digest at line {line_number}"
                    )
                expected_previous = row["sha"]
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InventoryConsumerIntegrityError(
            "broker ledger cannot be validated"
        ) from exc
    return rows


def _ledger_tip(rows: list[dict[str, Any]]) -> str:
    return rows[-1]["sha"] if rows else GENESIS_SHA256


@contextmanager
def _consumer_lock(ledger_path: Path) -> Iterator[None]:
    lock_path = ledger_path.with_name(ledger_path.name + ".ai-inventory.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _require_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise InventoryConsumerIntegrityError(f"missing mapping: {key}")
    return _snapshot_mapping(value, key)


def _snapshot_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InventoryConsumerIntegrityError(f"{label} must be a mapping")
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        snapshot = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise InventoryConsumerIntegrityError(f"{label} is not canonical JSON") from exc
    if not isinstance(snapshot, dict):
        raise InventoryConsumerIntegrityError(f"{label} must be an object")
    return snapshot


def _parse_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise InventoryConsumerIntegrityError("invalid UTC timestamp") from exc
    else:
        raise InventoryConsumerIntegrityError("invalid UTC timestamp")
    if parsed.tzinfo is None:
        raise InventoryConsumerIntegrityError("UTC timestamp must be aware")
    return parsed.astimezone(timezone.utc)


def _canonical_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _positive_finite(value: Any, label: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InventoryConsumerIntegrityError(f"invalid {label}")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise InventoryConsumerIntegrityError(f"invalid {label}")
    return value


def _validate_virtual_broker_units(actual: Any, expected: Any) -> None:
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(float(actual))
        or float(actual) <= 0.0
    ):
        raise InventoryConsumerIntegrityError(
            "virtual broker position units are invalid"
        )
    if isinstance(actual, float) and not actual.is_integer():
        raise InventoryConsumerIntegrityError(
            "fractional VirtualBroker units are incompatible with decision V1"
        )
    if int(actual) != expected:
        raise InventoryConsumerIntegrityError("broker position mismatch: units")


def _positive_integral_virtual_units(value: Any) -> float | int:
    units = _positive_finite(value, "virtual units")
    if not float(units).is_integer():
        raise InventoryConsumerIntegrityError(
            "fractional VirtualBroker units are incompatible with decision V1"
        )
    return units


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
