"""Fail-closed AI exit decisions and durable one-shot execution receipts.

This module deliberately knows nothing about a broker client.  A caller may
bind a locally supplied POST callable after all validation succeeds, but the
decision, exact position identity, and reservation are durable before that
callable can run.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


EXIT_DECISION_SCHEMA_VERSION = 1
# Five minutes is an execution-evidence freshness window, not a market model.
# Replace it with a broker-snapshot lease if that lease becomes authoritative.
DEFAULT_EXIT_DECISION_TTL_SECONDS = 5 * 60
# Ten minutes is the hard artifact lifetime bound.  It prevents a delayed
# worker from applying an old position decision after multiple monitor cycles.
MAX_EXIT_DECISION_TTL_SECONDS = 10 * 60
# Five seconds permits ordinary host-clock scheduling jitter while still
# rejecting decisions whose creation time is materially in the future.
MAX_CLOCK_SKEW_SECONDS = 5
# These are storage-abuse bounds, not market thresholds.  A managed receipt
# database should replace them if exit evidence ever outgrows local files.
MAX_TEXT_LENGTH = 512
MAX_EVIDENCE_REFS = 32
MAX_EVIDENCE_REF_LENGTH = 512

_DECISION_ID_RE = re.compile(r"^qrx_[0-9a-f]{64}$")


class ExitAction(str, Enum):
    HOLD = "HOLD"
    CLOSE_ALL = "CLOSE_ALL"
    REDUCE = "REDUCE"
    TIGHTEN_SL = "TIGHTEN_SL"
    REPLACE_TP = "REPLACE_TP"
    REQUEST_EVIDENCE = "REQUEST_EVIDENCE"


class ExitExecutionState(str, Enum):
    RESERVE_PRE_POST = "RESERVE_PRE_POST"
    POST_ATTEMPTED = "POST_ATTEMPTED"
    RECONCILING = "RECONCILING"
    TERMINAL = "TERMINAL"
    UNKNOWN_NO_RESEND = "UNKNOWN_NO_RESEND"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ExitDecisionError(RuntimeError):
    """A fail-closed exit-contract violation with a stable machine code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class NoTouchError(ExitDecisionError):
    """The exact position is not eligible for AI-system mutation."""


class StalePositionError(ExitDecisionError):
    """The broker epoch or position revision changed after the decision."""


class InvalidStateTransition(ExitDecisionError):
    """A one-shot receipt attempted an invalid or ambiguous transition."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > MAX_TEXT_LENGTH:
        raise ExitDecisionError("INVALID_IDENTITY", f"{field} must be non-empty and bounded")
    return text


def _utc_datetime(value: str | datetime, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ExitDecisionError("INVALID_TIME", f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ExitDecisionError("INVALID_TIME", f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _utc_text(value: str | datetime, field: str) -> str:
    return _utc_datetime(value, field).isoformat().replace("+00:00", "Z")


def _decimal_text(value: Any, field: str) -> str:
    if isinstance(value, bool):
        raise ExitDecisionError("INVALID_GEOMETRY", f"{field} must be numeric")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ExitDecisionError("INVALID_GEOMETRY", f"{field} must be numeric") from exc
    if not number.is_finite() or number <= 0:
        raise ExitDecisionError("INVALID_GEOMETRY", f"{field} must be positive and finite")
    return format(number, "f")


def _as_decimal(value: str | None) -> Decimal | None:
    return None if value is None else Decimal(value)


@dataclass(frozen=True)
class OwnerBinding:
    """Exact system owner and broker tag identity for one trade."""

    owner_kind: str
    owner_id: str
    client_extension_id: str
    campaign_id: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OwnerBinding":
        return cls(
            owner_kind=str(value.get("owner_kind") or "").strip().upper(),
            owner_id=str(value.get("owner_id") or "").strip(),
            client_extension_id=str(value.get("client_extension_id") or "").strip(),
            campaign_id=str(value.get("campaign_id") or "").strip(),
        )

    @property
    def is_exact_ai_system_owner(self) -> bool:
        return (
            self.owner_kind == "AI_SYSTEM"
            and bool(self.owner_id)
            and bool(self.client_extension_id)
            and bool(self.campaign_id)
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "owner_kind": self.owner_kind,
            "owner_id": self.owner_id,
            "client_extension_id": self.client_extension_id,
            "campaign_id": self.campaign_id,
        }


@dataclass(frozen=True)
class PositionSnapshot:
    """Minimum broker-readback surface needed to authorize one exit action."""

    cycle_id: str
    broker_epoch: str
    position_revision: str
    trade_id: str
    instrument: str
    side: PositionSide
    units: int
    owner_binding: OwnerBinding
    bid: str | None = None
    ask: str | None = None
    stop_loss: str | None = None
    take_profit: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PositionSnapshot":
        owner_raw = value.get("owner_binding")
        owner = OwnerBinding.from_mapping(owner_raw) if isinstance(owner_raw, Mapping) else OwnerBinding("UNKNOWN", "", "", "")
        try:
            side = PositionSide(str(value.get("side") or "").strip().upper())
        except ValueError as exc:
            raise ExitDecisionError("INVALID_POSITION", "position side must be LONG or SHORT") from exc
        units_raw = value.get("units")
        if isinstance(units_raw, bool):
            raise ExitDecisionError("INVALID_POSITION", "position units must be an integer")
        try:
            units = int(units_raw)
        except (TypeError, ValueError) as exc:
            raise ExitDecisionError("INVALID_POSITION", "position units must be an integer") from exc
        return cls(
            cycle_id=str(value.get("cycle_id") or "").strip(),
            broker_epoch=str(value.get("broker_epoch") or "").strip(),
            position_revision=str(value.get("position_revision") or "").strip(),
            trade_id=str(value.get("trade_id") or "").strip(),
            instrument=str(value.get("instrument") or "").strip(),
            side=side,
            units=units,
            owner_binding=owner,
            bid=None if value.get("bid") is None else _decimal_text(value.get("bid"), "bid"),
            ask=None if value.get("ask") is None else _decimal_text(value.get("ask"), "ask"),
            stop_loss=None if value.get("stop_loss") is None else _decimal_text(value.get("stop_loss"), "stop_loss"),
            take_profit=None if value.get("take_profit") is None else _decimal_text(value.get("take_profit"), "take_profit"),
        )


_MUTATING_ACTIONS = frozenset(
    {ExitAction.CLOSE_ALL, ExitAction.REDUCE, ExitAction.TIGHTEN_SL, ExitAction.REPLACE_TP}
)


@dataclass(frozen=True)
class ExitDecision:
    """A complete, immutable, content-addressed exit decision."""

    decision_id: str
    schema_version: int
    action: ExitAction
    cycle_id: str
    broker_epoch: str
    position_revision: str
    trade_id: str
    instrument: str
    owner_binding: OwnerBinding
    created_at_utc: str
    expires_at_utc: str
    emergency_eligible: bool = False
    units: int | None = None
    stop_loss: str | None = None
    take_profit: str | None = None
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()
    resource_claims: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        action: ExitAction | str,
        cycle_id: str,
        broker_epoch: str,
        position_revision: str,
        trade_id: str,
        instrument: str,
        owner_binding: OwnerBinding | Mapping[str, Any],
        created_at_utc: str | datetime,
        expires_at_utc: str | datetime | None = None,
        ttl_seconds: int = DEFAULT_EXIT_DECISION_TTL_SECONDS,
        emergency_eligible: bool = False,
        units: int | None = None,
        stop_loss: Any = None,
        take_profit: Any = None,
        reason: str = "",
        evidence_refs: Sequence[str] = (),
    ) -> "ExitDecision":
        try:
            normalized_action = action if isinstance(action, ExitAction) else ExitAction(str(action).strip().upper())
        except ValueError as exc:
            raise ExitDecisionError("INVALID_ACTION", "unsupported exit action") from exc
        owner = owner_binding if isinstance(owner_binding, OwnerBinding) else OwnerBinding.from_mapping(owner_binding)
        created = _utc_datetime(created_at_utc, "created_at_utc")
        if expires_at_utc is None:
            if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
                raise ExitDecisionError("INVALID_TTL", "ttl_seconds must be a positive integer")
            expires = created + timedelta(seconds=ttl_seconds)
        else:
            expires = _utc_datetime(expires_at_utc, "expires_at_utc")
        refs = tuple(str(ref).strip() for ref in evidence_refs)
        instrument_text = _required_text(instrument, "instrument")
        trade_text = _required_text(trade_id, "trade_id")
        claims = (
            (
                f"position:{trade_text}",
                f"reverse-entry:{_required_text(cycle_id, 'cycle_id')}:{instrument_text}",
            )
            if normalized_action in _MUTATING_ACTIONS
            else ()
        )
        provisional = cls(
            decision_id="",
            schema_version=EXIT_DECISION_SCHEMA_VERSION,
            action=normalized_action,
            cycle_id=_required_text(cycle_id, "cycle_id"),
            broker_epoch=_required_text(broker_epoch, "broker_epoch"),
            position_revision=_required_text(position_revision, "position_revision"),
            trade_id=trade_text,
            instrument=instrument_text,
            owner_binding=owner,
            created_at_utc=_utc_text(created, "created_at_utc"),
            expires_at_utc=_utc_text(expires, "expires_at_utc"),
            emergency_eligible=emergency_eligible,
            units=units,
            stop_loss=None if stop_loss is None else _decimal_text(stop_loss, "stop_loss"),
            take_profit=None if take_profit is None else _decimal_text(take_profit, "take_profit"),
            reason=str(reason or "").strip(),
            evidence_refs=refs,
            resource_claims=claims,
        )
        provisional._validate_shape()
        return replace(provisional, decision_id="qrx_" + _content_digest(provisional.to_dict(include_decision_id=False)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExitDecision":
        try:
            action = ExitAction(str(value.get("action") or "").strip().upper())
        except ValueError as exc:
            raise ExitDecisionError("INVALID_ACTION", "unsupported exit action") from exc
        owner_raw = value.get("owner_binding")
        if not isinstance(owner_raw, Mapping):
            raise ExitDecisionError("INVALID_OWNER", "owner_binding is required")
        decision = cls(
            decision_id=str(value.get("decision_id") or "").strip(),
            schema_version=int(value.get("schema_version") or 0),
            action=action,
            cycle_id=str(value.get("cycle_id") or "").strip(),
            broker_epoch=str(value.get("broker_epoch") or "").strip(),
            position_revision=str(value.get("position_revision") or "").strip(),
            trade_id=str(value.get("trade_id") or "").strip(),
            instrument=str(value.get("instrument") or "").strip(),
            owner_binding=OwnerBinding.from_mapping(owner_raw),
            created_at_utc=str(value.get("created_at_utc") or "").strip(),
            expires_at_utc=str(value.get("expires_at_utc") or "").strip(),
            emergency_eligible=value.get("emergency_eligible", False),
            units=value.get("units"),
            stop_loss=value.get("stop_loss"),
            take_profit=value.get("take_profit"),
            reason=str(value.get("reason") or "").strip(),
            evidence_refs=tuple(value.get("evidence_refs") or ()),
            resource_claims=tuple(value.get("resource_claims") or ()),
        )
        decision._validate_shape()
        decision._validate_content_address()
        return decision

    def to_dict(self, *, include_decision_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "action": self.action.value,
            "cycle_id": self.cycle_id,
            "broker_epoch": self.broker_epoch,
            "position_revision": self.position_revision,
            "trade_id": self.trade_id,
            "instrument": self.instrument,
            "owner_binding": self.owner_binding.to_dict(),
            "created_at_utc": self.created_at_utc,
            "expires_at_utc": self.expires_at_utc,
            "emergency_eligible": self.emergency_eligible,
            "units": self.units,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "resource_claims": list(self.resource_claims),
        }
        if include_decision_id:
            result["decision_id"] = self.decision_id
        return result

    def _validate_shape(self) -> None:
        if self.schema_version != EXIT_DECISION_SCHEMA_VERSION:
            raise ExitDecisionError("INVALID_SCHEMA", "unsupported exit decision schema")
        for field, value in (
            ("cycle_id", self.cycle_id),
            ("broker_epoch", self.broker_epoch),
            ("position_revision", self.position_revision),
            ("trade_id", self.trade_id),
            ("instrument", self.instrument),
        ):
            _required_text(value, field)
        if not self.owner_binding.is_exact_ai_system_owner:
            raise NoTouchError("NO_TOUCH", "manual, operator, tagless, external, or unknown ownership is immutable")
        created = _utc_datetime(self.created_at_utc, "created_at_utc")
        expires = _utc_datetime(self.expires_at_utc, "expires_at_utc")
        ttl = (expires - created).total_seconds()
        if ttl <= 0 or ttl > MAX_EXIT_DECISION_TTL_SECONDS:
            raise ExitDecisionError("INVALID_TTL", "decision TTL is outside the bounded window")
        if not isinstance(self.emergency_eligible, bool):
            raise ExitDecisionError("INVALID_EMERGENCY_FLAG", "emergency_eligible must be boolean")
        if self.emergency_eligible and self.action not in {ExitAction.CLOSE_ALL, ExitAction.REDUCE}:
            raise ExitDecisionError("INVALID_EMERGENCY_FLAG", "only CLOSE_ALL or REDUCE may be emergency eligible")
        if len(self.reason) > MAX_TEXT_LENGTH:
            raise ExitDecisionError("INVALID_REASON", "reason is too large")
        if len(self.evidence_refs) > MAX_EVIDENCE_REFS or any(
            not isinstance(ref, str) or not ref or len(ref) > MAX_EVIDENCE_REF_LENGTH
            for ref in self.evidence_refs
        ):
            raise ExitDecisionError("INVALID_EVIDENCE_REFS", "evidence refs are malformed or unbounded")

        expected_claims = (
            (f"position:{self.trade_id}", f"reverse-entry:{self.cycle_id}:{self.instrument}")
            if self.action in _MUTATING_ACTIONS
            else ()
        )
        if self.resource_claims != expected_claims:
            raise ExitDecisionError("INVALID_RESOURCE_CLAIMS", "resource claims do not match the exact exit identity")

        has_units = self.units is not None
        if has_units and (isinstance(self.units, bool) or not isinstance(self.units, int) or self.units <= 0):
            raise ExitDecisionError("INVALID_GEOMETRY", "units must be a positive integer")
        has_sl = self.stop_loss is not None
        has_tp = self.take_profit is not None
        if has_sl:
            _decimal_text(self.stop_loss, "stop_loss")
        if has_tp:
            _decimal_text(self.take_profit, "take_profit")
        expected = {
            ExitAction.HOLD: (False, False, False),
            ExitAction.CLOSE_ALL: (False, False, False),
            ExitAction.REDUCE: (True, False, False),
            ExitAction.TIGHTEN_SL: (False, True, False),
            ExitAction.REPLACE_TP: (False, False, True),
            ExitAction.REQUEST_EVIDENCE: (False, False, False),
        }[self.action]
        if (has_units, has_sl, has_tp) != expected:
            raise ExitDecisionError("INVALID_GEOMETRY", f"fields do not match {self.action.value}")

    def _validate_content_address(self) -> None:
        if not _DECISION_ID_RE.fullmatch(self.decision_id):
            raise ExitDecisionError("INVALID_DECISION_ID", "exit decision id is malformed")
        expected = "qrx_" + _content_digest(self.to_dict(include_decision_id=False))
        if self.decision_id != expected:
            raise ExitDecisionError("DECISION_TAMPERED", "exit decision content address does not match")

    def validate_for_position(
        self,
        position: PositionSnapshot | Mapping[str, Any],
        *,
        now: str | datetime,
    ) -> PositionSnapshot:
        """Revalidate exact ownership, revision, TTL, and action geometry."""

        self._validate_shape()
        self._validate_content_address()
        snapshot = position if isinstance(position, PositionSnapshot) else PositionSnapshot.from_mapping(position)
        current = _utc_datetime(now, "now")
        created = _utc_datetime(self.created_at_utc, "created_at_utc")
        expires = _utc_datetime(self.expires_at_utc, "expires_at_utc")
        if current < created - timedelta(seconds=MAX_CLOCK_SKEW_SECONDS):
            raise StalePositionError("DECISION_FROM_FUTURE", "exit decision is materially future-dated")
        if current >= expires:
            raise StalePositionError("DECISION_EXPIRED", "exit decision TTL has elapsed")
        if not snapshot.owner_binding.is_exact_ai_system_owner:
            raise NoTouchError("NO_TOUCH", "position is manual, operator-owned, tagless, external, or unknown")
        if snapshot.trade_id != self.trade_id or snapshot.instrument != self.instrument:
            raise NoTouchError("NO_TOUCH", "trade identity does not exactly match the decision")
        if snapshot.owner_binding != self.owner_binding:
            raise NoTouchError("NO_TOUCH", "position owner binding does not exactly match the decision")
        if snapshot.cycle_id != self.cycle_id:
            raise StalePositionError("STALE_CYCLE", "position cycle changed after the decision")
        if snapshot.broker_epoch != self.broker_epoch:
            raise StalePositionError("STALE_BROKER_EPOCH", "broker epoch changed after the decision")
        if snapshot.position_revision != self.position_revision:
            raise StalePositionError("STALE_POSITION_REVISION", "position revision changed after the decision")
        if isinstance(snapshot.units, bool) or not isinstance(snapshot.units, int) or snapshot.units == 0:
            raise ExitDecisionError("INVALID_POSITION", "position units must be a non-zero integer")
        if self.action is ExitAction.REDUCE and (self.units is None or self.units >= abs(snapshot.units)):
            raise ExitDecisionError("INVALID_REDUCE_GEOMETRY", "REDUCE units must be a strict partial position magnitude")
        if self.action is ExitAction.TIGHTEN_SL:
            self._validate_tightened_stop(snapshot)
        if self.action is ExitAction.REPLACE_TP:
            self._validate_replacement_tp(snapshot)
        return snapshot

    def _validate_tightened_stop(self, snapshot: PositionSnapshot) -> None:
        proposed = _as_decimal(self.stop_loss)
        current = _as_decimal(snapshot.stop_loss)
        bid = _as_decimal(snapshot.bid)
        ask = _as_decimal(snapshot.ask)
        if proposed is None or current is None or bid is None or ask is None:
            raise ExitDecisionError("INVALID_SL_GEOMETRY", "tightening requires current SL and bid/ask")
        if snapshot.side is PositionSide.LONG and not (current < proposed < bid):
            raise ExitDecisionError("INVALID_SL_GEOMETRY", "LONG stop must tighten upward and remain below bid")
        if snapshot.side is PositionSide.SHORT and not (ask < proposed < current):
            raise ExitDecisionError("INVALID_SL_GEOMETRY", "SHORT stop must tighten downward and remain above ask")

    def _validate_replacement_tp(self, snapshot: PositionSnapshot) -> None:
        proposed = _as_decimal(self.take_profit)
        bid = _as_decimal(snapshot.bid)
        ask = _as_decimal(snapshot.ask)
        if proposed is None or bid is None or ask is None:
            raise ExitDecisionError("INVALID_TP_GEOMETRY", "TP replacement requires current bid/ask")
        if snapshot.side is PositionSide.LONG and proposed <= ask:
            raise ExitDecisionError("INVALID_TP_GEOMETRY", "LONG TP must remain above ask")
        if snapshot.side is PositionSide.SHORT and proposed >= bid:
            raise ExitDecisionError("INVALID_TP_GEOMETRY", "SHORT TP must remain below bid")


@dataclass(frozen=True)
class ExitExecutionReceipt:
    decision: ExitDecision
    state: ExitExecutionState
    reserved_at_utc: str
    updated_at_utc: str
    broker_result_digest: str | None = None
    terminal_outcome: str | None = None
    unknown_reason: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExitExecutionReceipt":
        decision_raw = value.get("decision")
        if not isinstance(decision_raw, Mapping):
            raise ExitDecisionError("CORRUPT_RECEIPT", "receipt decision is missing")
        try:
            state = ExitExecutionState(str(value.get("state") or ""))
        except ValueError as exc:
            raise ExitDecisionError("CORRUPT_RECEIPT", "receipt state is invalid") from exc
        receipt = cls(
            decision=ExitDecision.from_mapping(decision_raw),
            state=state,
            reserved_at_utc=_utc_text(str(value.get("reserved_at_utc") or ""), "reserved_at_utc"),
            updated_at_utc=_utc_text(str(value.get("updated_at_utc") or ""), "updated_at_utc"),
            broker_result_digest=value.get("broker_result_digest"),
            terminal_outcome=value.get("terminal_outcome"),
            unknown_reason=value.get("unknown_reason"),
        )
        if receipt.state is ExitExecutionState.UNKNOWN_NO_RESEND and not receipt.unknown_reason:
            raise ExitDecisionError("CORRUPT_RECEIPT", "unknown receipt lacks its no-resend reason")
        if receipt.state is ExitExecutionState.TERMINAL and not receipt.terminal_outcome:
            raise ExitDecisionError("CORRUPT_RECEIPT", "terminal receipt lacks an outcome")
        return receipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "quant_rabbit.exit_execution_receipt.v1",
            "decision": self.decision.to_dict(),
            "state": self.state.value,
            "reserved_at_utc": self.reserved_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "broker_result_digest": self.broker_result_digest,
            "terminal_outcome": self.terminal_outcome,
            "unknown_reason": self.unknown_reason,
        }


@dataclass(frozen=True)
class ReservationOutcome:
    receipt: ExitExecutionReceipt
    may_post: bool
    reason: str


class ExitExecutionStore:
    """Flock-serialized, atomic, durable one-shot exit reservation store."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.receipts_dir = self.root / "receipts"
        self.claims_dir = self.root / "claims"
        self.lock_path = self.root / ".exit-execution.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _receipt_path(self, decision_id: str) -> Path:
        if not _DECISION_ID_RE.fullmatch(decision_id):
            raise ExitDecisionError("INVALID_DECISION_ID", "unsafe decision receipt path")
        return self.receipts_dir / f"{decision_id}.json"

    def _claim_path(self, claim: str) -> Path:
        return self.claims_dir / f"{hashlib.sha256(claim.encode('utf-8')).hexdigest()}.json"

    @staticmethod
    def _read_json(path: Path) -> Mapping[str, Any] | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExitDecisionError("CORRUPT_RECEIPT", f"cannot read durable receipt {path.name}") from exc
        if not isinstance(value, Mapping):
            raise ExitDecisionError("CORRUPT_RECEIPT", f"durable receipt {path.name} is not an object")
        return value

    @staticmethod
    def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(_canonical_json(value) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()

    def read(self, decision_id: str) -> ExitExecutionReceipt | None:
        with self._locked():
            raw = self._read_json(self._receipt_path(decision_id))
            return None if raw is None else ExitExecutionReceipt.from_mapping(raw)

    def reserve(
        self,
        decision: ExitDecision,
        position: PositionSnapshot | Mapping[str, Any],
        *,
        now: str | datetime,
    ) -> ReservationOutcome:
        decision.validate_for_position(position, now=now)
        now_text = _utc_text(now, "now")
        with self._locked():
            receipt_path = self._receipt_path(decision.decision_id)
            existing_raw = self._read_json(receipt_path)
            if existing_raw is not None:
                existing = ExitExecutionReceipt.from_mapping(existing_raw)
                return ReservationOutcome(existing, False, "EXISTING_RESERVATION_NO_RESEND")

            conflict: str | None = None
            for claim in decision.resource_claims:
                claim_raw = self._read_json(self._claim_path(claim))
                if claim_raw is not None:
                    claimed_by = str(claim_raw.get("decision_id") or "")
                    conflict = claimed_by or "UNKNOWN_PRIOR_CLAIM"
                    break
            if conflict is not None:
                blocked = ExitExecutionReceipt(
                    decision=decision,
                    state=ExitExecutionState.UNKNOWN_NO_RESEND,
                    reserved_at_utc=now_text,
                    updated_at_utc=now_text,
                    unknown_reason=f"RESOURCE_ALREADY_RESERVED:{conflict}",
                )
                self._atomic_write(receipt_path, blocked.to_dict())
                return ReservationOutcome(blocked, False, "RESOURCE_ALREADY_RESERVED_NO_RESEND")

            # Claims are committed before the executable receipt.  If the
            # process dies between files, the orphan claim fails closed and a
            # later attempt cannot duplicate an ambiguous execution.
            for claim in decision.resource_claims:
                self._atomic_write(
                    self._claim_path(claim),
                    {"claim": claim, "decision_id": decision.decision_id, "reserved_at_utc": now_text},
                )
            receipt = ExitExecutionReceipt(
                decision=decision,
                state=ExitExecutionState.RESERVE_PRE_POST,
                reserved_at_utc=now_text,
                updated_at_utc=now_text,
            )
            self._atomic_write(receipt_path, receipt.to_dict())
            return ReservationOutcome(receipt, True, "FRESH_RESERVATION")

    def _transition(
        self,
        decision_id: str,
        *,
        expected: frozenset[ExitExecutionState],
        target: ExitExecutionState,
        now: str | datetime,
        broker_result_digest: str | None = None,
        terminal_outcome: str | None = None,
        unknown_reason: str | None = None,
    ) -> ExitExecutionReceipt:
        with self._locked():
            path = self._receipt_path(decision_id)
            raw = self._read_json(path)
            if raw is None:
                raise InvalidStateTransition("MISSING_RESERVATION", "exit decision has no durable reservation")
            receipt = ExitExecutionReceipt.from_mapping(raw)
            if receipt.state is target:
                return receipt
            if receipt.state not in expected:
                raise InvalidStateTransition(
                    "INVALID_STATE_TRANSITION",
                    f"cannot move {receipt.state.value} to {target.value}",
                )
            updated = replace(
                receipt,
                state=target,
                updated_at_utc=_utc_text(now, "now"),
                broker_result_digest=broker_result_digest or receipt.broker_result_digest,
                terminal_outcome=terminal_outcome,
                unknown_reason=unknown_reason,
            )
            self._atomic_write(path, updated.to_dict())
            return updated

    def mark_post_attempted(self, decision_id: str, *, now: str | datetime) -> ExitExecutionReceipt:
        return self._transition(
            decision_id,
            expected=frozenset({ExitExecutionState.RESERVE_PRE_POST}),
            target=ExitExecutionState.POST_ATTEMPTED,
            now=now,
        )

    def mark_reconciling(
        self,
        decision_id: str,
        *,
        broker_result: Any,
        now: str | datetime,
    ) -> ExitExecutionReceipt:
        return self._transition(
            decision_id,
            expected=frozenset({ExitExecutionState.POST_ATTEMPTED}),
            target=ExitExecutionState.RECONCILING,
            now=now,
            broker_result_digest=_content_digest(broker_result),
        )

    def mark_terminal(
        self,
        decision_id: str,
        *,
        outcome: str,
        now: str | datetime,
    ) -> ExitExecutionReceipt:
        outcome_text = _required_text(outcome, "outcome")
        return self._transition(
            decision_id,
            expected=frozenset({ExitExecutionState.RECONCILING}),
            target=ExitExecutionState.TERMINAL,
            now=now,
            terminal_outcome=outcome_text,
        )

    def mark_unknown_no_resend(
        self,
        decision_id: str,
        *,
        reason: str,
        now: str | datetime,
    ) -> ExitExecutionReceipt:
        reason_text = _required_text(reason, "unknown_reason")
        return self._transition(
            decision_id,
            expected=frozenset({ExitExecutionState.POST_ATTEMPTED, ExitExecutionState.RECONCILING}),
            target=ExitExecutionState.UNKNOWN_NO_RESEND,
            now=now,
            unknown_reason=reason_text,
        )

    def run_post_once(
        self,
        decision: ExitDecision,
        position: PositionSnapshot | Mapping[str, Any],
        *,
        post: Callable[[], Any],
        now: str | datetime,
    ) -> ReservationOutcome:
        """Run at most one POST callback; ordinary exceptions become UNKNOWN.

        ``BaseException`` is intentionally not caught.  A simulated process
        crash therefore leaves ``POST_ATTEMPTED`` on disk, and a restart sees
        the existing reservation and never calls ``post`` again.
        """

        reserved = self.reserve(decision, position, now=now)
        if not reserved.may_post:
            return reserved
        self.mark_post_attempted(decision.decision_id, now=now)
        try:
            result = post()
        except Exception as exc:
            receipt = self.mark_unknown_no_resend(
                decision.decision_id,
                reason=f"TRANSPORT_EXCEPTION:{type(exc).__name__}",
                now=now,
            )
            return ReservationOutcome(receipt, False, "TRANSPORT_EXCEPTION_NO_RESEND")
        receipt = self.mark_reconciling(
            decision.decision_id,
            broker_result=result,
            now=now,
        )
        return ReservationOutcome(receipt, False, "POST_ATTEMPTED_RECONCILIATION_REQUIRED")
