"""Broker-owner admission internals for a future isolated paper-AI DOJO room.

The fixed/control paper rooms do not import or use this module.  A future
``paper-ai-inventory`` broker-service process owns the exact
:class:`VirtualBroker` and uses the private controller below.  A bot must never
receive this controller or a broker object; its only supported surface is the
authenticated minimal RPC client in :mod:`dojo_ai_inventory_broker_service`.

Entry is default-deny.  One entry is admitted only when the virtual broker's
fully validated hash-chain contains a V2 ``ALLOW_NEW_VIRTUAL`` applied receipt
whose immutable entry signal exactly matches the call.  The permit is reserved
durably before the broker method runs, so a crash burns the permit rather than
risk a duplicate virtual order.  A successful call then appends a consumed
receipt.  ``BLOCK_NEW`` is restored as a persistent scope gate; it never
cancels an existing resting virtual order.  Cancellation is a separate future
AI-inventory consumer responsibility.

There is deliberately no inbox, OANDA, live gateway, active-room runner, or
registry import in this module.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_ai_evidence_packet import entry_signal_identity_sha256
from quant_rabbit.dojo_ai_inventory import (
    DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_ROLE,
    MAX_ENTRY_GATE_VALIDITY_SECONDS,
)
from quant_rabbit.virtual_broker import VirtualBroker


AI_ACTION_RESERVED_EVENT = "AI_INVENTORY_ACTION_RESERVED"
AI_ACTION_APPLIED_EVENT = "AI_INVENTORY_ACTION_APPLIED"
ENTRY_PERMIT_RESERVED_EVENT = "AI_ENTRY_PERMIT_RESERVED"
ENTRY_PERMIT_CONSUMED_EVENT = "AI_ENTRY_PERMIT_CONSUMED"
ENTRY_ADMISSION_REFERENCE_CONTRACT = "QR_DOJO_AI_ENTRY_ADMISSION_REFERENCE_V1"
ENTRY_PERMIT_CONSUMER_CONTRACT = "QR_DOJO_AI_ENTRY_PERMIT_CONSUMER_V1"
GENESIS_SHA256 = "0" * 64

# Ninety seconds is inherited from the V2 AI entry-gate decision contract.  It
# is a point-in-time evidence freshness boundary, not a tuned trading value.
# A longer permit must use a new versioned decision and proxy contract.
MAX_ENTRY_PERMIT_TTL_SECONDS = MAX_ENTRY_GATE_VALIDITY_SECONDS

# These bounds protect the local integrity validator from corrupted or hostile
# files.  They are serialization limits, not market or strategy parameters.
MAX_LEDGER_BYTES = 256 * 1024 * 1024
MAX_LEDGER_LINE_BYTES = 256 * 1024
MAX_LEDGER_ROWS = 1_000_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")
_LEDGER_KEYS = frozenset({"ts_utc", "event", "payload", "prev_sha", "sha"})
_SIGNAL_KEYS = frozenset(
    {
        "signal_identity_sha256",
        "pair",
        "side",
        "order_type",
        "units",
        "price",
        "strategy_tag",
        "entry_context_sha256",
        "tp_pips",
        "sl_pips",
        "observed_at_utc",
    }
)
_ADMISSION_KEYS = frozenset(
    {
        "evidence_packet_sha256",
        "permit_expires_at_utc",
        "entry_signal",
    }
)
_ADMISSION_REFERENCE_KEYS = frozenset(
    {
        "contract",
        "applied_receipt_sha256",
        "decision_sha256",
        "room_id",
        "candidate_id",
        "signal_identity_sha256",
    }
)


class AIInventoryAdmissionError(RuntimeError):
    """Base class for a fail-closed paper-AI entry rejection."""


class AIInventoryAdmissionIntegrityError(AIInventoryAdmissionError):
    """The receipt chain or its paper-only bindings cannot be trusted."""


class AIInventoryEntryDeniedError(AIInventoryAdmissionError):
    """No exact, current, single-use permit admits the requested entry."""


@dataclass(frozen=True)
class AIEntryPermit:
    """One durable single-use virtual-entry permit."""

    applied_receipt_sha256: str
    decision_sha256: str
    room_id: str
    candidate_id: str
    pair: str
    strategy_tag: str
    signal_identity_sha256: str
    signal_observed_at_utc: str
    permit_expires_at_utc: str
    evidence_packet_sha256: str
    entry_signal: Mapping[str, Any]
    applied_sequence: int


@dataclass(frozen=True)
class AIInventoryAdmissionState:
    """Pure state reconstructed from the complete virtual-broker ledger."""

    terminal_sha256: str
    row_count: int
    blocked_scopes: frozenset[tuple[str, str]]
    available_permits: tuple[AIEntryPermit, ...]
    reserved_permit_receipt_sha256s: frozenset[str]

    @property
    def default_deny(self) -> bool:
        """The proxy never admits an entry without one exact permit."""

        return True


def build_ai_inventory_admission_state(
    receipt_ledger_path: Path,
    *,
    room_id: str,
    candidate_id: str,
    as_of_utc: datetime,
) -> AIInventoryAdmissionState:
    """Validate the full hash chain and purely rebuild admission state.

    No file is written and no broker method is called.  Unknown ordinary
    virtual-broker events remain part of the validated chain but do not change
    admission state.
    """

    if not isinstance(receipt_ledger_path, Path):
        raise AIInventoryAdmissionIntegrityError(
            "receipt_ledger_path must be an explicit Path"
        )
    _require_room_scope(room_id, candidate_id)
    as_of = _require_aware_utc(as_of_utc, "as_of_utc")
    rows = _read_and_validate_ledger(receipt_ledger_path, as_of_utc=as_of)

    action_reservations: dict[str, Mapping[str, Any]] = {}
    applied_action_reservations: set[str] = set()
    permits: dict[str, AIEntryPermit] = {}
    reserved_permits: set[str] = set()
    consumed_reservations: set[str] = set()
    blocked_scopes: set[tuple[str, str]] = set()

    for sequence, row in enumerate(rows, start=1):
        event = row["event"]
        payload = row["payload"]
        if event == AI_ACTION_RESERVED_EVENT:
            _validate_action_reservation(
                payload,
                room_id=room_id,
                candidate_id=candidate_id,
            )
            action_reservations[row["sha"]] = payload
            continue
        if event == AI_ACTION_APPLIED_EVENT:
            reservation_sha = payload.get("reservation_sha256")
            reservation = action_reservations.get(reservation_sha)
            if reservation is None:
                raise AIInventoryAdmissionIntegrityError(
                    "AI APPLIED receipt has no reservation"
                )
            if reservation_sha in applied_action_reservations:
                raise AIInventoryAdmissionIntegrityError(
                    "AI action reservation has multiple APPLIED receipts"
                )
            _validate_receipt_safety(payload)
            if (
                payload.get("room_id") != room_id
                or payload.get("candidate_id") != candidate_id
            ):
                raise AIInventoryAdmissionIntegrityError(
                    "AI APPLIED receipt is outside the proxy scope"
                )
            applied_action_reservations.add(str(reservation_sha))
            action = payload.get("action")
            if action == "CLOSE_VIRTUAL":
                _validate_applied_close_receipt(
                    row,
                    payload,
                    reservation=reservation,
                    rows=rows,
                    current_sequence=sequence,
                    room_id=room_id,
                    candidate_id=candidate_id,
                )
                # A full virtual close must not let an older same-scope
                # resting order silently recreate the inventory.  The broker
                # owner enforces this durable scope as fill rejection.
                blocked_scopes.add((payload["pair"], payload["strategy_tag"]))
                continue
            if action not in {"BLOCK_NEW", "ALLOW_NEW_VIRTUAL"}:
                continue
            _validate_applied_gate_receipt(
                row,
                payload,
                reservation=reservation,
                rows=rows,
                current_sequence=sequence,
                room_id=room_id,
                candidate_id=candidate_id,
            )
            pair = payload["pair"]
            strategy_tag = payload["strategy_tag"]
            scope = (pair, strategy_tag)
            if action == "BLOCK_NEW":
                # BLOCK_NEW is a durable scope gate.  A later ALLOW receipt is
                # only an exact one-signal exception and does not erase this
                # restored block.  The inventory consumer has already
                # cancelled same-scope resting orders before APPLIED.
                blocked_scopes.add(scope)
                for receipt_sha, permit in tuple(permits.items()):
                    if permit.pair == pair and permit.strategy_tag == strategy_tag:
                        del permits[receipt_sha]
                continue
            permit = _permit_from_applied(
                row,
                payload,
                applied_sequence=sequence,
            )
            if permit.applied_receipt_sha256 in permits:
                raise AIInventoryAdmissionIntegrityError(
                    "duplicate applied permit receipt"
                )
            permits[permit.applied_receipt_sha256] = permit
            continue
        if event == ENTRY_PERMIT_RESERVED_EVENT:
            receipt_sha = _validate_entry_permit_reservation(
                payload,
                permits=permits,
                room_id=room_id,
                candidate_id=candidate_id,
            )
            if receipt_sha in reserved_permits:
                raise AIInventoryAdmissionIntegrityError(
                    "entry permit has multiple reservations"
                )
            reserved_permits.add(receipt_sha)
            continue
        if event == ENTRY_PERMIT_CONSUMED_EVENT:
            _validate_entry_permit_consumed(
                payload,
                rows=rows,
                current_sequence=sequence,
                reserved_permits=reserved_permits,
                consumed_reservations=consumed_reservations,
            )

    unresolved_actions = set(action_reservations) - applied_action_reservations
    if unresolved_actions:
        raise AIInventoryAdmissionIntegrityError(
            "AI action reservation is unresolved; entry admission is blocked"
        )
    available = tuple(
        permit
        for receipt_sha, permit in permits.items()
        if receipt_sha not in reserved_permits
        and _parse_utc(permit.permit_expires_at_utc) > as_of
    )
    return AIInventoryAdmissionState(
        terminal_sha256=rows[-1]["sha"] if rows else GENESIS_SHA256,
        row_count=len(rows),
        blocked_scopes=frozenset(blocked_scopes),
        available_permits=available,
        reserved_permit_receipt_sha256s=frozenset(reserved_permits),
    )


class _BrokerOwnedAdmissionController:
    """Default-deny admission controller used only by the broker owner.

    This is intentionally private.  It is not a bot-facing capability and
    must never cross the broker-service process boundary.
    """

    __slots__ = ("__broker", "__room_id", "__candidate_id", "__ledger_path")

    def __init__(
        self,
        broker: VirtualBroker,
        *,
        room_id: str,
        candidate_id: str,
    ) -> None:
        if type(broker) is not VirtualBroker:
            raise AIInventoryAdmissionIntegrityError(
                "admission proxy requires an exact VirtualBroker instance"
            )
        if broker.fast_ledger is not False:
            raise AIInventoryAdmissionIntegrityError(
                "admission proxy requires fsync-backed VirtualBroker receipts"
            )
        _require_room_scope(room_id, candidate_id)
        ledger_path = getattr(broker, "ledger_path", None)
        if not isinstance(ledger_path, Path):
            raise AIInventoryAdmissionIntegrityError(
                "VirtualBroker ledger path is unavailable"
            )
        try:
            resolved = ledger_path.resolve(strict=True)
        except OSError as exc:
            raise AIInventoryAdmissionIntegrityError(
                "VirtualBroker ledger path is unavailable"
            ) from exc
        if resolved.parent.name != room_id or "paper-ai-inventory" not in str(
            resolved.parent
        ):
            raise AIInventoryAdmissionIntegrityError(
                "broker ledger is outside the dedicated paper-ai-inventory room"
            )
        self.__broker = broker
        self.__room_id = room_id
        self.__candidate_id = candidate_id
        self.__ledger_path = resolved

    @property
    def positions(self) -> Mapping[str, Any]:
        """Return a detached position snapshot; callers cannot mutate broker state."""

        return MappingProxyType(copy.deepcopy(self.__broker.positions))

    @property
    def orders(self) -> Mapping[str, Any]:
        """Return a detached resting-order snapshot."""

        return MappingProxyType(copy.deepcopy(self.__broker.orders))

    def account(self) -> dict[str, Any]:
        """Return a detached read-only account snapshot."""

        return copy.deepcopy(self.__broker.account())

    def market_order(
        self,
        pair: str,
        side: str,
        units: float,
        tp_pips: float | None = None,
        sl_pips: float | None = None,
        strategy_tag: str | None = None,
        entry_context: dict[str, Any] | None = None,
        *,
        ai_admission: Mapping[str, Any] | None = None,
    ) -> str:
        return self._admit_and_call(
            order_type="MARKET",
            pair=pair,
            side=side,
            units=units,
            price=None,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            strategy_tag=strategy_tag,
            entry_context=entry_context,
            ai_admission=ai_admission,
            broker_call=lambda: self.__broker.market_order(
                pair,
                side,
                units,
                tp_pips=tp_pips,
                sl_pips=sl_pips,
                strategy_tag=strategy_tag,
                entry_context=entry_context,
            ),
        )

    def limit_order(
        self,
        pair: str,
        side: str,
        units: float,
        price: float,
        tp_pips: float | None = None,
        sl_pips: float | None = None,
        strategy_tag: str | None = None,
        entry_context: dict[str, Any] | None = None,
        *,
        ai_admission: Mapping[str, Any] | None = None,
    ) -> str:
        return self._admit_and_call(
            order_type="LIMIT",
            pair=pair,
            side=side,
            units=units,
            price=price,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            strategy_tag=strategy_tag,
            entry_context=entry_context,
            ai_admission=ai_admission,
            broker_call=lambda: self.__broker.limit_order(
                pair,
                side,
                units,
                price,
                tp_pips=tp_pips,
                sl_pips=sl_pips,
                strategy_tag=strategy_tag,
                entry_context=entry_context,
            ),
        )

    def stop_order(
        self,
        pair: str,
        side: str,
        units: float,
        price: float,
        tp_pips: float | None = None,
        sl_pips: float | None = None,
        strategy_tag: str | None = None,
        entry_context: dict[str, Any] | None = None,
        *,
        ai_admission: Mapping[str, Any] | None = None,
    ) -> str:
        return self._admit_and_call(
            order_type="STOP",
            pair=pair,
            side=side,
            units=units,
            price=price,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            strategy_tag=strategy_tag,
            entry_context=entry_context,
            ai_admission=ai_admission,
            broker_call=lambda: self.__broker.stop_order(
                pair,
                side,
                units,
                price,
                tp_pips=tp_pips,
                sl_pips=sl_pips,
                strategy_tag=strategy_tag,
                entry_context=entry_context,
            ),
        )

    def _admit_and_call(
        self,
        *,
        order_type: str,
        pair: str,
        side: str,
        units: float,
        price: float | None,
        tp_pips: float | None,
        sl_pips: float | None,
        strategy_tag: str | None,
        entry_context: dict[str, Any] | None,
        ai_admission: Mapping[str, Any] | None,
        broker_call: Callable[[], str],
    ) -> str:
        now = _utc_now()
        _require_market_open(now)
        reference = _validate_admission_reference(
            ai_admission,
            room_id=self.__room_id,
            candidate_id=self.__candidate_id,
        )
        context_sha = _entry_context_sha256(entry_context, strategy_tag)

        with _admission_lock(self.__ledger_path):
            state = build_ai_inventory_admission_state(
                self.__ledger_path,
                room_id=self.__room_id,
                candidate_id=self.__candidate_id,
                as_of_utc=now,
            )
            rows = _read_and_validate_ledger(
                self.__ledger_path,
                as_of_utc=now,
            )
            lifecycle = _entry_lifecycle_for_reference(rows, reference)
            permit = (
                _permit_for_reference_from_rows(rows, reference)
                if lifecycle is not None
                else _select_exact_permit(state, reference)
            )
            actual_signal = {
                "pair": pair,
                "side": side,
                "order_type": order_type,
                "units": _positive_float(units, "units"),
                "price": _optional_positive_float(price, "price"),
                "strategy_tag": strategy_tag,
                "entry_context_sha256": context_sha,
                "tp_pips": _optional_positive_float(tp_pips, "tp_pips"),
                "sl_pips": _optional_positive_float(sl_pips, "sl_pips"),
                "observed_at_utc": permit.signal_observed_at_utc,
            }
            try:
                actual_identity = entry_signal_identity_sha256(actual_signal)
            except (TypeError, ValueError) as exc:
                raise AIInventoryEntryDeniedError(
                    "actual entry signal is not canonical"
                ) from exc
            if actual_identity != permit.signal_identity_sha256 or dict(
                permit.entry_signal
            ) != {**actual_signal, "signal_identity_sha256": actual_identity}:
                raise AIInventoryEntryDeniedError(
                    "actual entry arguments do not match the permitted signal"
                )

            if lifecycle is None:
                reservation_payload = _entry_reservation_payload(
                    permit, entry_method=order_type, reserved_at_utc=_format_utc(now)
                )
                _broker_log(
                    self.__broker,
                    ENTRY_PERMIT_RESERVED_EVENT,
                    reservation_payload,
                )
                reserved_rows = _read_and_validate_ledger(
                    self.__ledger_path,
                    as_of_utc=_utc_now(),
                )
                reservation = reserved_rows[-1]
                if (
                    reservation["event"] != ENTRY_PERMIT_RESERVED_EVENT
                    or reservation["payload"] != reservation_payload
                ):
                    raise AIInventoryAdmissionIntegrityError(
                        "exact entry reservation was not persisted"
                    )
                lifecycle = {
                    "status": "RESERVED",
                    "reservation": reservation,
                    "mutation": None,
                    "consumed": None,
                }
            return _complete_entry_lifecycle(
                self.__broker,
                self.__ledger_path,
                lifecycle,
                broker_call=broker_call,
            )


def _permit_from_applied(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    applied_sequence: int,
) -> AIEntryPermit:
    admission = _require_mapping(payload.get("admission_binding"), "admission_binding")
    signal = _require_mapping(admission.get("entry_signal"), "entry_signal")
    if set(admission) != _ADMISSION_KEYS or set(signal) != _SIGNAL_KEYS:
        raise AIInventoryAdmissionIntegrityError(
            "ALLOW_NEW_VIRTUAL admission schema is invalid"
        )
    _validate_entry_signal_shape(signal)
    expected_identity = entry_signal_identity_sha256(signal)
    if signal.get("signal_identity_sha256") != expected_identity:
        raise AIInventoryAdmissionIntegrityError(
            "ALLOW_NEW_VIRTUAL signal identity mismatch"
        )
    exact = (
        ("pair", payload.get("pair"), signal.get("pair")),
        (
            "strategy_tag receipt",
            payload.get("strategy_tag"),
            signal.get("strategy_tag"),
        ),
        (
            "evidence_packet_sha256",
            admission.get("evidence_packet_sha256"),
            payload.get("ai_evidence_packet_sha256"),
        ),
    )
    for label, left, right in exact:
        if left is None or left != right:
            raise AIInventoryAdmissionIntegrityError(
                f"ALLOW_NEW_VIRTUAL binding mismatch: {label}"
            )
    signal_observed_at = _parse_utc(signal["observed_at_utc"])
    if _format_utc(signal_observed_at) != signal["observed_at_utc"]:
        raise AIInventoryAdmissionIntegrityError(
            "ALLOW_NEW_VIRTUAL signal timestamp is not canonical"
        )
    permit_expires_at = _parse_utc(admission["permit_expires_at_utc"])
    if _format_utc(permit_expires_at) != admission["permit_expires_at_utc"]:
        raise AIInventoryAdmissionIntegrityError(
            "ALLOW_NEW_VIRTUAL permit expiry is not canonical"
        )
    applied_at = _parse_utc(row["ts_utc"])
    if signal_observed_at > applied_at:
        raise AIInventoryAdmissionIntegrityError(
            "ALLOW_NEW_VIRTUAL signal is future-dated at application"
        )
    _require_market_open(signal_observed_at)
    if not _is_sha256(admission.get("evidence_packet_sha256")):
        raise AIInventoryAdmissionIntegrityError(
            "ALLOW_NEW_VIRTUAL evidence packet digest is invalid"
        )
    return AIEntryPermit(
        applied_receipt_sha256=str(row["sha"]),
        decision_sha256=str(payload["decision_sha256"]),
        room_id=str(payload["room_id"]),
        candidate_id=str(payload["candidate_id"]),
        pair=str(payload["pair"]),
        strategy_tag=str(payload["strategy_tag"]),
        signal_identity_sha256=expected_identity,
        signal_observed_at_utc=str(signal["observed_at_utc"]),
        permit_expires_at_utc=str(admission["permit_expires_at_utc"]),
        evidence_packet_sha256=str(admission["evidence_packet_sha256"]),
        entry_signal=MappingProxyType(copy.deepcopy(dict(signal))),
        applied_sequence=applied_sequence,
    )


def _entry_reservation_payload(
    permit: AIEntryPermit, *, entry_method: str, reserved_at_utc: str
) -> dict[str, Any]:
    return {
        "contract": ENTRY_PERMIT_CONSUMER_CONTRACT,
        "permit_applied_receipt_sha256": permit.applied_receipt_sha256,
        "decision_sha256": permit.decision_sha256,
        "room_id": permit.room_id,
        "candidate_id": permit.candidate_id,
        "pair": permit.pair,
        "strategy_tag": permit.strategy_tag,
        "signal_identity_sha256": permit.signal_identity_sha256,
        "entry_method": entry_method,
        "reserved_at_utc": reserved_at_utc,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "decision_contract": DOJO_AI_INVENTORY_DECISION_CONTRACT,
    }


def _permit_for_reference_from_rows(
    rows: list[dict[str, Any]], reference: Mapping[str, Any]
) -> AIEntryPermit:
    receipt_sha = reference["applied_receipt_sha256"]
    matches = [
        (sequence, row)
        for sequence, row in enumerate(rows, start=1)
        if row["sha"] == receipt_sha and row["event"] == AI_ACTION_APPLIED_EVENT
    ]
    if len(matches) != 1:
        raise AIInventoryAdmissionIntegrityError(
            "reserved entry permit APPLIED receipt is absent or ambiguous"
        )
    sequence, row = matches[0]
    permit = _permit_from_applied(
        row,
        row["payload"],
        applied_sequence=sequence,
    )
    exact = (
        ("decision_sha256", permit.decision_sha256),
        ("room_id", permit.room_id),
        ("candidate_id", permit.candidate_id),
        ("signal_identity_sha256", permit.signal_identity_sha256),
    )
    for field, expected in exact:
        if reference.get(field) != expected:
            raise AIInventoryAdmissionIntegrityError(
                f"reserved entry reference mismatch: {field}"
            )
    return permit


def _entry_lifecycle_for_reference(
    rows: list[dict[str, Any]], reference: Mapping[str, Any]
) -> dict[str, Any] | None:
    reservations = [
        (index, row)
        for index, row in enumerate(rows)
        if row["event"] == ENTRY_PERMIT_RESERVED_EVENT
        and row["payload"].get("permit_applied_receipt_sha256")
        == reference["applied_receipt_sha256"]
    ]
    if not reservations:
        return None
    if len(reservations) != 1:
        raise AIInventoryAdmissionIntegrityError(
            "entry permit has duplicate reservations"
        )
    index, reservation = reservations[0]
    suffix = rows[index:]
    if len(suffix) > 3:
        raise AIInventoryAdmissionIntegrityError(
            "entry reservation has an unknown ledger suffix"
        )
    method = reservation["payload"].get("entry_method")
    expected_event = {
        "MARKET": "FILL_MARKET",
        "LIMIT": "ORDER_LIMIT",
        "STOP": "ORDER_STOP",
    }.get(method)
    if expected_event is None:
        raise AIInventoryAdmissionIntegrityError(
            "entry reservation method is invalid"
        )
    mutation: dict[str, Any] | None = None
    consumed: dict[str, Any] | None = None
    if len(suffix) >= 2:
        mutation = suffix[1]
        if (
            mutation["event"] != expected_event
            or mutation["prev_sha"] != reservation["sha"]
        ):
            raise AIInventoryAdmissionIntegrityError(
                "entry mutation is not exact and adjacent to its reservation"
            )
    if len(suffix) == 3:
        consumed = suffix[2]
        if (
            consumed["event"] != ENTRY_PERMIT_CONSUMED_EVENT
            or consumed["prev_sha"] != mutation["sha"]
        ):
            raise AIInventoryAdmissionIntegrityError(
                "entry consumed receipt is not exact and adjacent"
            )
    return {
        "status": (
            "CONSUMED"
            if consumed is not None
            else "ENTRY_DURABLE"
            if mutation is not None
            else "RESERVED"
        ),
        "reservation": reservation,
        "mutation": mutation,
        "consumed": consumed,
    }


def _validate_entry_mutation_row(
    mutation: Mapping[str, Any],
    reservation: Mapping[str, Any],
    permit: AIEntryPermit,
) -> str:
    signal = permit.entry_signal
    method = reservation["payload"]["entry_method"]
    expected_event = {
        "MARKET": "FILL_MARKET",
        "LIMIT": "ORDER_LIMIT",
        "STOP": "ORDER_STOP",
    }[method]
    if (
        mutation.get("event") != expected_event
        or mutation.get("prev_sha") != reservation.get("sha")
    ):
        raise AIInventoryAdmissionIntegrityError(
            "entry mutation is detached from its reservation"
        )
    payload = _require_mapping(mutation.get("payload"), "entry mutation payload")
    common = (
        ("pair", signal["pair"]),
        ("side", signal["side"]),
        ("units", signal["units"]),
        ("strategy_tag", signal["strategy_tag"]),
    )
    for field, expected in common:
        if payload.get(field) != expected:
            raise AIInventoryAdmissionIntegrityError(
                f"entry mutation/signal mismatch: {field}"
            )
    if method == "MARKET":
        created = payload.get("trade_id")
        if payload.get("quote", {}).get("ts") != signal["observed_at_utc"]:
            raise AIInventoryAdmissionIntegrityError(
                "market entry quote timestamp mismatch"
            )
    else:
        created = payload.get("order_id")
        if payload.get("price") != signal["price"]:
            raise AIInventoryAdmissionIntegrityError(
                "pending entry price mismatch"
            )
        for field in ("tp_pips", "sl_pips"):
            if payload.get(field) != signal[field]:
                raise AIInventoryAdmissionIntegrityError(
                    f"pending entry/signal mismatch: {field}"
                )
    if not _is_identifier(created):
        raise AIInventoryAdmissionIntegrityError(
            "entry mutation created id is invalid"
        )
    return str(created)


def _complete_entry_lifecycle(
    broker: VirtualBroker,
    ledger_path: Path,
    lifecycle: Mapping[str, Any],
    *,
    broker_call: Callable[[], str],
) -> str:
    reservation = lifecycle["reservation"]
    mutation = lifecycle.get("mutation")
    consumed = lifecycle.get("consumed")
    if consumed is not None:
        return str(consumed["payload"]["created_id"])

    reference = {
        "applied_receipt_sha256": reservation["payload"][
            "permit_applied_receipt_sha256"
        ],
        "decision_sha256": reservation["payload"]["decision_sha256"],
        "room_id": reservation["payload"]["room_id"],
        "candidate_id": reservation["payload"]["candidate_id"],
        "signal_identity_sha256": reservation["payload"][
            "signal_identity_sha256"
        ],
    }
    rows = _read_and_validate_ledger(ledger_path, as_of_utc=_utc_now())
    permit = _permit_for_reference_from_rows(rows, reference)
    if mutation is None:
        created_id = broker_call()
        rows = _read_and_validate_ledger(ledger_path, as_of_utc=_utc_now())
        recovered = _entry_lifecycle_for_reference(rows, reference)
        if recovered is None or recovered.get("mutation") is None:
            raise AIInventoryAdmissionIntegrityError(
                "broker did not durably append the permitted entry"
            )
        mutation = recovered["mutation"]
    created_id = _validate_entry_mutation_row(mutation, reservation, permit)
    if created_id in broker.positions:
        pass
    elif created_id in broker.orders:
        pass
    else:
        raise AIInventoryAdmissionIntegrityError(
            "durable entry mutation is absent from broker state"
        )
    consumed_payload = {
        **reservation["payload"],
        "reservation_sha256": reservation["sha"],
        "created_id": created_id,
        "consumed_at_utc": _format_utc(_utc_now()),
        "status": "CONSUMED",
    }
    _broker_log(broker, ENTRY_PERMIT_CONSUMED_EVENT, consumed_payload)
    final_rows = _read_and_validate_ledger(ledger_path, as_of_utc=_utc_now())
    if (
        final_rows[-1]["event"] != ENTRY_PERMIT_CONSUMED_EVENT
        or final_rows[-1]["payload"] != consumed_payload
        or final_rows[-1]["prev_sha"] != mutation["sha"]
    ):
        raise AIInventoryAdmissionIntegrityError(
            "exact entry consumption was not persisted"
        )
    return created_id


def reconcile_entry_checkpoint_suffix(
    broker: VirtualBroker,
    ledger_path: Path,
    *,
    room_id: str,
    candidate_id: str,
    checkpoint_tip: str,
    as_of_utc: datetime,
) -> dict[str, Any]:
    """Replay a validated entry mutation suffix onto broker checkpoint state."""

    state = build_ai_inventory_admission_state(
        ledger_path,
        room_id=room_id,
        candidate_id=candidate_id,
        as_of_utc=as_of_utc,
    )
    rows = _read_and_validate_ledger(ledger_path, as_of_utc=as_of_utc)
    start = 0
    if checkpoint_tip != GENESIS_SHA256:
        matches = [
            index for index, row in enumerate(rows) if row["sha"] == checkpoint_tip
        ]
        if len(matches) != 1:
            raise AIInventoryAdmissionIntegrityError(
                "entry checkpoint tip is absent or ambiguous"
            )
        start = matches[0] + 1
    suffix = rows[start:]
    if not suffix or suffix[0]["event"] != ENTRY_PERMIT_RESERVED_EVENT:
        raise AIInventoryAdmissionIntegrityError(
            "unknown entry suffix after checkpoint"
        )
    reservation = suffix[0]
    reference = {
        "applied_receipt_sha256": reservation["payload"].get(
            "permit_applied_receipt_sha256"
        ),
        "decision_sha256": reservation["payload"].get("decision_sha256"),
        "room_id": reservation["payload"].get("room_id"),
        "candidate_id": reservation["payload"].get("candidate_id"),
        "signal_identity_sha256": reservation["payload"].get(
            "signal_identity_sha256"
        ),
    }
    lifecycle = _entry_lifecycle_for_reference(rows, reference)
    if lifecycle is None or lifecycle["reservation"]["sha"] != reservation["sha"]:
        raise AIInventoryAdmissionIntegrityError(
            "entry recovery lifecycle does not begin at checkpoint"
        )
    permit = _permit_for_reference_from_rows(rows, reference)
    mutation = lifecycle.get("mutation")
    if mutation is not None:
        signal = dict(permit.entry_signal)
        payload = mutation["payload"]
        entry_context = payload.get("entry_context")
        observed: list[tuple[str, dict[str, Any]]] = []
        original_log = broker._log

        def validate_log(event: str, value: dict[str, Any]) -> None:
            observed.append((event, copy.deepcopy(value)))
            if len(observed) > 1:
                raise AIInventoryAdmissionIntegrityError(
                    "entry replay produced multiple broker events"
                )

        broker._log = validate_log  # type: ignore[method-assign]
        try:
            common = {
                "pair": signal["pair"],
                "side": signal["side"],
                "units": signal["units"],
                "tp_pips": signal["tp_pips"],
                "sl_pips": signal["sl_pips"],
                "strategy_tag": signal["strategy_tag"],
                "entry_context": entry_context,
            }
            if signal["order_type"] == "MARKET":
                created = broker.market_order(**common)
            elif signal["order_type"] == "LIMIT":
                created = broker.limit_order(price=signal["price"], **common)
            else:
                created = broker.stop_order(price=signal["price"], **common)
        finally:
            broker._log = original_log  # type: ignore[method-assign]
        if (
            len(observed) != 1
            or observed[0][0] != mutation["event"]
            or observed[0][1] != mutation["payload"]
            or created != _validate_entry_mutation_row(
                mutation, reservation, permit
            )
        ):
            raise AIInventoryAdmissionIntegrityError(
                "entry mutation cannot be deterministically replayed"
            )
    if lifecycle["status"] == "RESERVED" and (
        permit.applied_receipt_sha256 not in state.reserved_permit_receipt_sha256s
    ):
        raise AIInventoryAdmissionIntegrityError(
            "entry reservation was not reconstructed as consumed"
        )
    return lifecycle


def _validate_entry_signal_shape(signal: Mapping[str, Any]) -> None:
    pair = signal.get("pair")
    if not isinstance(pair, str) or _PAIR_RE.fullmatch(pair) is None:
        raise AIInventoryAdmissionIntegrityError("entry signal pair is invalid")
    if signal.get("side") not in {"LONG", "SHORT"}:
        raise AIInventoryAdmissionIntegrityError("entry signal side is invalid")
    order_type = signal.get("order_type")
    if order_type not in {"MARKET", "LIMIT", "STOP"}:
        raise AIInventoryAdmissionIntegrityError("entry signal order type is invalid")
    if not _is_identifier(signal.get("strategy_tag")) or not _is_sha256(
        signal.get("entry_context_sha256")
    ):
        raise AIInventoryAdmissionIntegrityError(
            "entry signal strategy or context binding is invalid"
        )
    units = signal.get("units")
    if units.__class__ is not float or not math.isfinite(units) or units <= 0.0:
        raise AIInventoryAdmissionIntegrityError("entry signal units are not canonical")
    price = signal.get("price")
    if order_type == "MARKET":
        if price is not None:
            raise AIInventoryAdmissionIntegrityError(
                "MARKET entry signal price must be null"
            )
    elif price.__class__ is not float or not math.isfinite(price) or price <= 0.0:
        raise AIInventoryAdmissionIntegrityError(
            "pending entry signal price is not canonical"
        )
    for field in ("tp_pips", "sl_pips"):
        value = signal.get(field)
        if value is not None and (
            value.__class__ is not float or not math.isfinite(value) or value <= 0.0
        ):
            raise AIInventoryAdmissionIntegrityError(
                f"entry signal {field} is not canonical"
            )
    observed_at = _parse_utc(signal.get("observed_at_utc"))
    if _format_utc(observed_at) != signal.get("observed_at_utc"):
        raise AIInventoryAdmissionIntegrityError(
            "entry signal observed_at_utc is not canonical"
        )


def _validate_action_reservation(
    payload: Mapping[str, Any],
    *,
    room_id: str,
    candidate_id: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise AIInventoryAdmissionIntegrityError(
            "AI action reservation payload is invalid"
        )
    action = payload.get("action")
    if action not in {
        "HOLD",
        "BLOCK_NEW",
        "ALLOW_NEW_VIRTUAL",
        "REDUCE_VIRTUAL",
        "CLOSE_VIRTUAL",
    }:
        raise AIInventoryAdmissionIntegrityError(
            "AI action reservation has an invalid action"
        )
    _validate_receipt_safety(payload)
    if payload.get("room_id") != room_id or payload.get("candidate_id") != candidate_id:
        raise AIInventoryAdmissionIntegrityError(
            "AI action reservation is outside the proxy scope"
        )


def _validate_applied_gate_receipt(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    reservation: Mapping[str, Any] | None,
    rows: list[dict[str, Any]],
    current_sequence: int,
    room_id: str,
    candidate_id: str,
) -> None:
    if reservation is None:
        raise AIInventoryAdmissionIntegrityError(
            "entry gate APPLIED receipt has no reservation"
        )
    reservation_sha256 = payload.get("reservation_sha256")
    cancelled_order_ids = payload.get("cancelled_order_ids")
    cancel_sha256s = payload.get("cancel_sha256s")
    if (
        not isinstance(cancelled_order_ids, list)
        or not isinstance(cancel_sha256s, list)
        or len(cancelled_order_ids) != len(cancel_sha256s)
        or len(cancelled_order_ids) != len(set(cancelled_order_ids))
        or len(cancel_sha256s) != len(set(cancel_sha256s))
        or any(not _is_identifier(value) for value in cancelled_order_ids)
        or any(not _is_sha256(value) for value in cancel_sha256s)
    ):
        raise AIInventoryAdmissionIntegrityError(
            "entry gate cancellation receipt list is invalid"
        )
    applied_index = current_sequence - 1
    reservation_matches = [
        index
        for index, candidate in enumerate(rows[:applied_index])
        if candidate.get("sha") == reservation_sha256
        and candidate.get("event") == AI_ACTION_RESERVED_EVENT
    ]
    if len(reservation_matches) != 1:
        raise AIInventoryAdmissionIntegrityError(
            "entry gate APPLIED receipt has no unique reservation row"
        )
    reservation_index = reservation_matches[0]
    cancellation_rows = rows[reservation_index + 1 : applied_index]
    if len(cancellation_rows) != len(cancelled_order_ids):
        raise AIInventoryAdmissionIntegrityError(
            "entry gate cancellation chain length mismatch"
        )
    previous_sha256 = reservation_sha256
    for index, cancellation in enumerate(cancellation_rows):
        cancellation_payload = cancellation.get("payload")
        if (
            cancellation.get("event") != "ORDER_CANCEL"
            or cancellation.get("prev_sha") != previous_sha256
            or cancellation.get("sha") != cancel_sha256s[index]
            or not isinstance(cancellation_payload, Mapping)
            or set(cancellation_payload) != {"order_id", "strategy_tag"}
            or cancellation_payload.get("order_id")
            != cancelled_order_ids[index]
            or cancellation_payload.get("strategy_tag")
            != payload.get("strategy_tag")
        ):
            raise AIInventoryAdmissionIntegrityError(
                "entry gate cancellation chain is invalid"
            )
        previous_sha256 = cancellation["sha"]
    if row.get("prev_sha") != previous_sha256:
        raise AIInventoryAdmissionIntegrityError(
            "entry gate APPLIED receipt is detached from its action chain"
        )
    _validate_receipt_safety(payload)
    common = (
        "decision_sha256",
        "decision_identity_sha256",
        "action",
        "room_id",
        "candidate_id",
        "pair",
        "strategy_tag",
        "decision_contract",
        "consumer_contract",
        "decision_role",
    )
    for key in common:
        if payload.get(key) is None or payload.get(key) != reservation.get(key):
            raise AIInventoryAdmissionIntegrityError(
                f"entry gate reservation/APPLIED mismatch: {key}"
            )
    if payload.get("admission_binding") != reservation.get("admission_binding"):
        raise AIInventoryAdmissionIntegrityError(
            "entry gate reservation/APPLIED mismatch: admission_binding"
        )
    if payload.get("room_id") != room_id or payload.get("candidate_id") != candidate_id:
        raise AIInventoryAdmissionIntegrityError(
            "entry gate APPLIED receipt is outside the proxy scope"
        )
    if payload.get("status") != "APPLIED":
        raise AIInventoryAdmissionIntegrityError("entry gate receipt is not APPLIED")
    if not _is_sha256(payload.get("decision_sha256")):
        raise AIInventoryAdmissionIntegrityError(
            "entry gate decision digest is invalid"
        )
    pair = payload.get("pair")
    strategy_tag = payload.get("strategy_tag")
    if not isinstance(pair, str) or _PAIR_RE.fullmatch(pair) is None:
        raise AIInventoryAdmissionIntegrityError("entry gate pair is invalid")
    if not _is_identifier(strategy_tag):
        raise AIInventoryAdmissionIntegrityError("entry gate strategy_tag is invalid")

    action = payload["action"]
    if action == "BLOCK_NEW":
        if (
            payload.get("block_new") is not True
            or payload.get("allow_new_virtual") is not False
            or payload.get("single_use_entry_permit") is not False
            or payload.get("entry_proxy_consumed") is not None
            or payload.get("admission_binding") is not None
            or payload.get("close_sha256") is not None
            or payload.get("realized_pl_jpy") is not None
        ):
            raise AIInventoryAdmissionIntegrityError(
                "BLOCK_NEW receipt contract is invalid"
            )
        return
    if cancelled_order_ids or cancel_sha256s:
        raise AIInventoryAdmissionIntegrityError(
            "ALLOW_NEW_VIRTUAL cannot contain order cancellations"
        )
    if (
        payload.get("allow_new_virtual") is not True
        or payload.get("single_use_entry_permit") is not True
        or payload.get("entry_proxy_consumed") is not False
        or payload.get("block_new") is not False
        or payload.get("close_sha256") is not None
        or payload.get("realized_pl_jpy") is not None
    ):
        raise AIInventoryAdmissionIntegrityError(
            "ALLOW_NEW_VIRTUAL receipt contract is invalid"
        )
    admission = _require_mapping(payload.get("admission_binding"), "admission_binding")
    expires = _parse_utc(admission.get("permit_expires_at_utc"))
    applied_at = _parse_utc(row.get("ts_utc"))
    if expires <= applied_at:
        raise AIInventoryAdmissionIntegrityError(
            "entry permit expired before it was durably applied"
        )
    if (expires - applied_at).total_seconds() > MAX_ENTRY_PERMIT_TTL_SECONDS:
        raise AIInventoryAdmissionIntegrityError("entry permit exceeds the short TTL")
    _require_market_open(applied_at)


def _validate_applied_close_receipt(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    reservation: Mapping[str, Any],
    rows: list[dict[str, Any]],
    current_sequence: int,
    room_id: str,
    candidate_id: str,
) -> None:
    """Validate the consumer's RESERVED -> CLOSE -> APPLIED close sequence."""

    _validate_receipt_safety(payload)
    if (
        payload.get("room_id") != room_id
        or payload.get("candidate_id") != candidate_id
        or payload.get("action") != "CLOSE_VIRTUAL"
        or payload.get("status") != "APPLIED"
    ):
        raise AIInventoryAdmissionIntegrityError(
            "CLOSE_VIRTUAL receipt is outside the admission scope"
        )
    for key in (
        "decision_sha256",
        "decision_identity_sha256",
        "action",
        "room_id",
        "candidate_id",
        "pair",
        "strategy_tag",
        "position_id",
        "decision_contract",
        "consumer_contract",
        "decision_role",
    ):
        if payload.get(key) is None or payload.get(key) != reservation.get(key):
            raise AIInventoryAdmissionIntegrityError(
                f"CLOSE_VIRTUAL reservation/APPLIED mismatch: {key}"
            )
    reservation_sha = payload.get("reservation_sha256")
    close_sha = payload.get("close_sha256")
    if not _is_sha256(reservation_sha) or not _is_sha256(close_sha):
        raise AIInventoryAdmissionIntegrityError(
            "CLOSE_VIRTUAL receipt chain binding is invalid"
        )
    prior = rows[: current_sequence - 1]
    closes = [
        candidate
        for candidate in prior
        if candidate.get("sha") == close_sha and candidate.get("event") == "CLOSE"
    ]
    if len(closes) != 1:
        raise AIInventoryAdmissionIntegrityError(
            "CLOSE_VIRTUAL has no exact virtual close row"
        )
    close = closes[0]
    if (
        close.get("prev_sha") != reservation_sha
        or row.get("prev_sha") != close_sha
        or close["payload"].get("trade_id") != payload.get("position_id")
        or close["payload"].get("strategy_tag") != payload.get("strategy_tag")
    ):
        raise AIInventoryAdmissionIntegrityError(
            "CLOSE_VIRTUAL receipt is not adjacent to its exact close"
        )


def _validate_receipt_safety(payload: Mapping[str, Any]) -> None:
    required = {
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "decision_contract": DOJO_AI_INVENTORY_DECISION_CONTRACT,
        "consumer_contract": DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
        "decision_role": DOJO_AI_INVENTORY_DECISION_ROLE,
    }
    for key, expected in required.items():
        if payload.get(key) != expected or type(payload.get(key)) is not type(expected):
            raise AIInventoryAdmissionIntegrityError(
                f"unsafe AI inventory receipt invariant: {key}"
            )


def _validate_entry_permit_reservation(
    payload: Mapping[str, Any],
    *,
    permits: Mapping[str, AIEntryPermit],
    room_id: str,
    candidate_id: str,
) -> str:
    if not isinstance(payload, Mapping):
        raise AIInventoryAdmissionIntegrityError(
            "entry permit reservation payload is invalid"
        )
    receipt_sha = payload.get("permit_applied_receipt_sha256")
    permit = permits.get(receipt_sha)
    if permit is None:
        raise AIInventoryAdmissionIntegrityError(
            "entry reservation references no applied permit"
        )
    required = {
        "contract": ENTRY_PERMIT_CONSUMER_CONTRACT,
        "decision_sha256": permit.decision_sha256,
        "room_id": room_id,
        "candidate_id": candidate_id,
        "pair": permit.pair,
        "strategy_tag": permit.strategy_tag,
        "signal_identity_sha256": permit.signal_identity_sha256,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "virtual_broker_mutation_allowed": True,
        "external_broker_mutation_allowed": False,
        "decision_contract": DOJO_AI_INVENTORY_DECISION_CONTRACT,
    }
    for key, expected in required.items():
        if payload.get(key) != expected or type(payload.get(key)) is not type(expected):
            raise AIInventoryAdmissionIntegrityError(
                f"entry reservation binding mismatch: {key}"
            )
    if payload.get("entry_method") not in {"MARKET", "LIMIT", "STOP"}:
        raise AIInventoryAdmissionIntegrityError("entry reservation method is invalid")
    reserved_at = _parse_utc(payload.get("reserved_at_utc"))
    if reserved_at >= _parse_utc(permit.permit_expires_at_utc):
        raise AIInventoryAdmissionIntegrityError(
            "entry reservation is not inside permit TTL"
        )
    _require_market_open(reserved_at)
    return str(receipt_sha)


def _validate_entry_permit_consumed(
    payload: Mapping[str, Any],
    *,
    rows: list[dict[str, Any]],
    current_sequence: int,
    reserved_permits: set[str],
    consumed_reservations: set[str],
) -> None:
    if not isinstance(payload, Mapping):
        raise AIInventoryAdmissionIntegrityError(
            "entry permit consumption payload is invalid"
        )
    receipt_sha = payload.get("permit_applied_receipt_sha256")
    if receipt_sha not in reserved_permits:
        raise AIInventoryAdmissionIntegrityError(
            "entry consumption has no prior reservation"
        )
    reservation_sha = payload.get("reservation_sha256")
    if not _is_sha256(reservation_sha):
        raise AIInventoryAdmissionIntegrityError(
            "entry consumption reservation digest is invalid"
        )
    if reservation_sha in consumed_reservations:
        raise AIInventoryAdmissionIntegrityError(
            "entry reservation has multiple consumed receipts"
        )
    prior = rows[: current_sequence - 1]
    matching = [
        (index, row)
        for index, row in enumerate(prior)
        if row["sha"] == reservation_sha and row["event"] == ENTRY_PERMIT_RESERVED_EVENT
    ]
    if len(matching) != 1:
        raise AIInventoryAdmissionIntegrityError(
            "entry consumption reservation is missing or ambiguous"
        )
    reservation_index, reservation_row = matching[0]
    reservation = reservation_row["payload"]
    if current_sequence != reservation_index + 3:
        raise AIInventoryAdmissionIntegrityError(
            "entry consumption is not adjacent to reservation and mutation"
        )
    mutation = rows[reservation_index + 1]
    expected_event = {
        "MARKET": "FILL_MARKET",
        "LIMIT": "ORDER_LIMIT",
        "STOP": "ORDER_STOP",
    }.get(reservation.get("entry_method"))
    if (
        mutation.get("event") != expected_event
        or mutation.get("prev_sha") != reservation_sha
        or rows[current_sequence - 1].get("prev_sha") != mutation.get("sha")
    ):
        raise AIInventoryAdmissionIntegrityError(
            "entry consumption mutation adjacency is invalid"
        )
    for key in (
        "contract",
        "permit_applied_receipt_sha256",
        "decision_sha256",
        "room_id",
        "candidate_id",
        "pair",
        "strategy_tag",
        "signal_identity_sha256",
        "entry_method",
        "reserved_at_utc",
        "paper_only",
        "order_authority",
        "live_permission",
        "virtual_broker_mutation_allowed",
        "external_broker_mutation_allowed",
        "decision_contract",
    ):
        if payload.get(key) != reservation.get(key):
            raise AIInventoryAdmissionIntegrityError(
                f"entry consumption/reservation mismatch: {key}"
            )
    if payload.get("status") != "CONSUMED" or not _is_identifier(
        payload.get("created_id")
    ):
        raise AIInventoryAdmissionIntegrityError(
            "entry consumption status or created id is invalid"
        )
    created_field = "trade_id" if expected_event == "FILL_MARKET" else "order_id"
    if payload.get("created_id") != mutation["payload"].get(created_field):
        raise AIInventoryAdmissionIntegrityError(
            "entry consumption created id differs from mutation"
        )
    consumed_at = _parse_utc(payload.get("consumed_at_utc"))
    if consumed_at < _parse_utc(payload.get("reserved_at_utc")):
        raise AIInventoryAdmissionIntegrityError(
            "entry consumption predates its reservation"
        )
    consumed_reservations.add(str(reservation_sha))


def _select_exact_permit(
    state: AIInventoryAdmissionState,
    reference: Mapping[str, Any],
) -> AIEntryPermit:
    matches = [
        permit
        for permit in state.available_permits
        if permit.applied_receipt_sha256 == reference["applied_receipt_sha256"]
        and permit.decision_sha256 == reference["decision_sha256"]
        and permit.room_id == reference["room_id"]
        and permit.candidate_id == reference["candidate_id"]
        and permit.signal_identity_sha256 == reference["signal_identity_sha256"]
    ]
    if len(matches) != 1:
        raise AIInventoryEntryDeniedError(
            "no exact unconsumed AI entry permit is available"
        )
    return matches[0]


def _validate_admission_reference(
    value: Mapping[str, Any] | None,
    *,
    room_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AIInventoryEntryDeniedError("AI admission reference is required")
    snapshot = _snapshot_mapping(value, "AI admission reference")
    if set(snapshot) != _ADMISSION_REFERENCE_KEYS:
        raise AIInventoryEntryDeniedError("AI admission reference schema is invalid")
    if snapshot.get("contract") != ENTRY_ADMISSION_REFERENCE_CONTRACT:
        raise AIInventoryEntryDeniedError("AI admission reference contract is invalid")
    for key in (
        "applied_receipt_sha256",
        "decision_sha256",
        "signal_identity_sha256",
    ):
        if not _is_sha256(snapshot.get(key)):
            raise AIInventoryEntryDeniedError(
                f"AI admission reference has invalid {key}"
            )
    if (
        snapshot.get("room_id") != room_id
        or snapshot.get("candidate_id") != candidate_id
    ):
        raise AIInventoryEntryDeniedError(
            "AI admission reference room/candidate mismatch"
        )
    return snapshot


def _entry_context_sha256(
    entry_context: dict[str, Any] | None,
    strategy_tag: str | None,
) -> str:
    if not isinstance(entry_context, dict):
        raise AIInventoryEntryDeniedError(
            "AI-permitted entry requires a canonical entry_context"
        )
    if (
        not isinstance(strategy_tag, str)
        or entry_context.get("strategy_tag") != strategy_tag
    ):
        raise AIInventoryEntryDeniedError(
            "entry_context strategy_tag does not match the order"
        )
    try:
        raw = json.dumps(
            entry_context,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AIInventoryEntryDeniedError(
            "entry_context is not canonical JSON"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def _admission_lock(ledger_path: Path) -> Iterator[None]:
    lock_path = ledger_path.with_name(ledger_path.name + ".ai-entry.lock")
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _broker_log(broker: VirtualBroker, event: str, payload: dict[str, Any]) -> None:
    if type(broker) is not VirtualBroker:
        raise AIInventoryAdmissionIntegrityError(
            "entry receipt writer requires exact VirtualBroker"
        )
    logger = getattr(broker, "_log", None)
    if not callable(logger):
        raise AIInventoryAdmissionIntegrityError(
            "VirtualBroker receipt writer is unavailable"
        )
    logger(event, payload)


def _read_and_validate_ledger(
    path: Path, *, as_of_utc: datetime
) -> list[dict[str, Any]]:
    as_of = _require_aware_utc(as_of_utc, "as_of_utc")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AIInventoryAdmissionIntegrityError(
            "receipt ledger is absent or unreadable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AIInventoryAdmissionIntegrityError(
                "receipt ledger must be a regular file"
            )
        if before.st_size > MAX_LEDGER_BYTES:
            raise AIInventoryAdmissionIntegrityError(
                "receipt ledger exceeds the byte limit"
            )
        raw_parts: list[bytes] = []
        remaining = MAX_LEDGER_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            raw_parts.append(chunk)
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
            raise AIInventoryAdmissionIntegrityError(
                "receipt ledger changed during validation"
            )
    finally:
        os.close(descriptor)
    raw = b"".join(raw_parts)
    if len(raw) > MAX_LEDGER_BYTES:
        raise AIInventoryAdmissionIntegrityError(
            "receipt ledger exceeds the byte limit"
        )
    if raw and not raw.endswith(b"\n"):
        raise AIInventoryAdmissionIntegrityError(
            "receipt ledger has a truncated final row"
        )
    lines = raw.splitlines()
    if len(lines) > MAX_LEDGER_ROWS:
        raise AIInventoryAdmissionIntegrityError("receipt ledger exceeds the row limit")

    rows: list[dict[str, Any]] = []
    expected_prev = GENESIS_SHA256
    previous_ts: datetime | None = None
    for line_number, line in enumerate(lines, start=1):
        if not line or len(line) > MAX_LEDGER_LINE_BYTES:
            raise AIInventoryAdmissionIntegrityError(
                f"invalid receipt ledger line size at {line_number}"
            )
        try:
            row = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise AIInventoryAdmissionIntegrityError(
                f"invalid receipt ledger JSON at {line_number}"
            ) from exc
        if not isinstance(row, dict) or set(row) != _LEDGER_KEYS:
            raise AIInventoryAdmissionIntegrityError(
                f"invalid receipt ledger schema at {line_number}"
            )
        if row.get("prev_sha") != expected_prev or not _is_sha256(row.get("sha")):
            raise AIInventoryAdmissionIntegrityError(
                f"receipt ledger chain mismatch at {line_number}"
            )
        body = {key: row[key] for key in ("ts_utc", "event", "payload", "prev_sha")}
        if _sha256(body) != row["sha"]:
            raise AIInventoryAdmissionIntegrityError(
                f"receipt ledger digest mismatch at {line_number}"
            )
        if not isinstance(row.get("event"), str) or not isinstance(
            row.get("payload"), dict
        ):
            raise AIInventoryAdmissionIntegrityError(
                f"receipt ledger event/payload mismatch at {line_number}"
            )
        timestamp = _parse_utc(row.get("ts_utc"))
        if timestamp > as_of:
            raise AIInventoryAdmissionIntegrityError(
                f"future-dated receipt ledger row at {line_number}"
            )
        if previous_ts is not None and timestamp < previous_ts:
            raise AIInventoryAdmissionIntegrityError(
                f"non-monotonic receipt ledger clock at {line_number}"
            )
        previous_ts = timestamp
        expected_prev = row["sha"]
        rows.append(row)
    return rows


def _require_room_scope(room_id: str, candidate_id: str) -> None:
    if (
        not isinstance(room_id, str)
        or not room_id.startswith("paper-ai-inventory-")
        or Path(room_id).name != room_id
    ):
        raise AIInventoryAdmissionIntegrityError(
            "room is not an isolated paper-ai-inventory room"
        )
    if not _is_identifier(candidate_id):
        raise AIInventoryAdmissionIntegrityError("candidate_id is invalid")


def _require_market_open(value: datetime) -> None:
    try:
        is_open = compute_market_status(value).is_fx_open
    except Exception as exc:
        raise AIInventoryAdmissionIntegrityError(
            "FX market status is unavailable"
        ) from exc
    if not is_open:
        raise AIInventoryEntryDeniedError(
            "AI virtual entry is disabled while FX is closed"
        )


def _require_aware_utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AIInventoryAdmissionIntegrityError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise AIInventoryAdmissionIntegrityError("invalid UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AIInventoryAdmissionIntegrityError("invalid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AIInventoryAdmissionIntegrityError("UTC timestamp is naive")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return _require_aware_utc(value, "timestamp").isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _snapshot_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        decoded = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise AIInventoryEntryDeniedError(f"{label} must be canonical JSON") from exc
    if not isinstance(decoded, dict):
        raise AIInventoryEntryDeniedError(f"{label} must be an object")
    return decoded


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AIInventoryAdmissionIntegrityError(f"{label} must be an object")
    return value


def _reject_duplicate_keys(
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


def _sha256(value: Any) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AIInventoryAdmissionIntegrityError(
            "receipt value is not canonical JSON"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_identifier(value: object) -> bool:
    return isinstance(value, str) and _ID_RE.fullmatch(value) is not None


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AIInventoryEntryDeniedError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise AIInventoryEntryDeniedError(f"{label} must be finite and positive")
    return normalized


def _optional_positive_float(value: object, label: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, label)
