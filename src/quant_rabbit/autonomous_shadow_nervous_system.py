"""Autonomous, append-only coordination for the shadow trading nervous system.

The module coordinates bounded workers without putting a human approval step in
the normal path.  It deliberately cannot express an order, grant live
permission, or mutate broker state.  Human input is evidence-only; a human may
also engage the kill switch, but is not required for ordinary shadow state
transitions.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT = "QR_AUTONOMOUS_SHADOW_NERVOUS_SYSTEM_V1"
SCHEMA_VERSION = 1
EXECUTION_AUTHORITY = "NONE"
MANUAL_TAGLESS_POLICY = "NO_TOUCH"
HUMAN_ROLE = "ASSIST"
DEFAULT_MIN_NET_CONFIDENCE = 0.60
MAX_FUTURE_SKEW_SECONDS = 5

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STATES = {
    "IDLE",
    "SIGNAL",
    "HYPOTHESIS",
    "CHALLENGED",
    "ADMITTED",
    "FILLED",
    "UNFILLED",
    "OPEN",
    "EXITED",
    "LEARNED",
    "EXPIRED",
    "BLOCKED",
}
_TERMINAL_STATES = {"LEARNED", "EXPIRED", "BLOCKED"}
_RESTARTABLE_STATES = {"IDLE", "LEARNED", "EXPIRED"}
_EXPECTED_WORKER = {
    "IDLE": "perception",
    "LEARNED": "perception",
    "EXPIRED": "perception",
    "SIGNAL": "hypothesis",
    "HYPOTHESIS": "critic",
    "CHALLENGED": "admission",
    "ADMITTED": "fill_truth",
    "FILLED": "lifecycle",
    "UNFILLED": "learning",
    "OPEN": "exit",
    "EXITED": "learning",
}
_ADVANCE_STATE = {
    "IDLE": "SIGNAL",
    "LEARNED": "SIGNAL",
    "EXPIRED": "SIGNAL",
    "SIGNAL": "HYPOTHESIS",
    "HYPOTHESIS": "CHALLENGED",
    "CHALLENGED": "ADMITTED",
    "ADMITTED": "FILLED",
    "FILLED": "OPEN",
    "UNFILLED": "LEARNED",
    "OPEN": "EXITED",
    "EXITED": "LEARNED",
}
_VERDICTS = {"ADVANCE", "WAIT", "BLOCK", "EXPIRE", "NO_FILL"}
_PACKET_KEYS = {"cycle_id", "decisions", "human_assist", "kill_switch"}
_DECISION_KEYS = {
    "decision_id",
    "worker",
    "verdict",
    "reason",
    "observed_at_utc",
    "expires_at_utc",
    "supporting_evidence",
    "contradicting_evidence",
    "counterevidence_reviewed",
    "confidence",
    "uncertainty",
}
_ASSIST_KEYS = {"note", "evidence_refs", "observed_at_utc"}
_FORBIDDEN_ACTIONS = (
    "AUTHOR_ORDER_ACTION",
    "SELECT_ORDER_SIDE",
    "SELECT_ENTRY_OR_EXIT_PRICE",
    "SET_TP_OR_SL",
    "ALLOCATE_CAPITAL",
    "GRANT_LIVE_PERMISSION",
    "INVOKE_BROKER_GATEWAY",
    "MUTATE_BROKER_STATE",
    "TOUCH_MANUAL_OR_TAGLESS_POSITION",
)


@dataclass(frozen=True)
class WorkerDecision:
    decision_id: str
    worker: str
    verdict: str
    reason: str
    observed_at_utc: datetime
    expires_at_utc: datetime
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    counterevidence_reviewed: bool
    confidence: float
    uncertainty: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkerDecision":
        _require_exact_keys(value, _DECISION_KEYS, "worker decision")
        worker = _bounded_id(value.get("worker"), "worker")
        verdict = str(value.get("verdict") or "").strip().upper()
        if verdict not in _VERDICTS:
            raise ValueError(f"unsupported verdict for {worker}: {verdict!r}")
        reason = str(value.get("reason") or "").strip()
        if not reason or len(reason) > 2_000:
            raise ValueError(f"{worker} reason must contain 1..2000 characters")
        confidence = _probability(value.get("confidence"), f"{worker} confidence")
        uncertainty = _probability(value.get("uncertainty"), f"{worker} uncertainty")
        support = _evidence_refs(value.get("supporting_evidence"), f"{worker} supporting_evidence")
        contradict = _evidence_refs(
            value.get("contradicting_evidence"),
            f"{worker} contradicting_evidence",
        )
        if verdict == "ADVANCE" and not support:
            raise ValueError(f"{worker} ADVANCE requires supporting evidence")
        return cls(
            decision_id=_bounded_id(value.get("decision_id"), f"{worker} decision_id"),
            worker=worker,
            verdict=verdict,
            reason=reason,
            observed_at_utc=_parse_utc(value.get("observed_at_utc"), f"{worker} observed_at_utc"),
            expires_at_utc=_parse_utc(value.get("expires_at_utc"), f"{worker} expires_at_utc"),
            supporting_evidence=support,
            contradicting_evidence=contradict,
            counterevidence_reviewed=value.get("counterevidence_reviewed") is True,
            confidence=confidence,
            uncertainty=uncertainty,
        )


@dataclass(frozen=True)
class HumanAssist:
    note: str
    evidence_refs: tuple[str, ...]
    observed_at_utc: datetime

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HumanAssist":
        _require_exact_keys(value, _ASSIST_KEYS, "human assist")
        note = str(value.get("note") or "").strip()
        if not note or len(note) > 2_000:
            raise ValueError("human assist note must contain 1..2000 characters")
        return cls(
            note=note,
            evidence_refs=_evidence_refs(value.get("evidence_refs"), "human assist evidence_refs"),
            observed_at_utc=_parse_utc(value.get("observed_at_utc"), "human assist observed_at_utc"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": HUMAN_ROLE,
            "can_approve_transition": False,
            "can_grant_live_permission": False,
            "note": self.note,
            "evidence_refs": list(self.evidence_refs),
            "observed_at_utc": _format_utc(self.observed_at_utc),
        }


@dataclass(frozen=True)
class AutonomousShadowCycleSummary:
    status: str
    state: str
    cycle_id: str
    expected_worker: str | None
    events_appended: int
    ledger_path: Path
    output_path: Path
    report_path: Path
    human_approval_required: bool = False
    live_permission_allowed: bool = False


class AutonomousShadowNervousSystem:
    """Advance a shadow episode through independently owned worker joints."""

    def __init__(
        self,
        *,
        ledger_path: Path,
        output_path: Path,
        report_path: Path,
        min_net_confidence: float = DEFAULT_MIN_NET_CONFIDENCE,
    ) -> None:
        resolved_paths = {
            ledger_path.resolve(),
            output_path.resolve(),
            report_path.resolve(),
        }
        if len(resolved_paths) != 3:
            raise ValueError("ledger, output, and report paths must be distinct")
        self.ledger_path = ledger_path
        self.output_path = output_path
        self.report_path = report_path
        self.min_net_confidence = _probability(
            min_net_confidence,
            "min_net_confidence",
        )

    def run(
        self,
        packet: Mapping[str, Any],
        *,
        now_utc: datetime | None = None,
    ) -> AutonomousShadowCycleSummary:
        _require_exact_keys(packet, _PACKET_KEYS, "cycle packet")
        cycle_id = _bounded_id(packet.get("cycle_id"), "cycle_id")
        now = _aware(now_utc or datetime.now(timezone.utc))
        kill_switch = packet.get("kill_switch", False)
        if not isinstance(kill_switch, bool):
            raise ValueError("kill_switch must be boolean")

        raw_decisions = packet.get("decisions", [])
        if not isinstance(raw_decisions, Sequence) or isinstance(raw_decisions, (str, bytes)):
            raise ValueError("decisions must be a list")
        decisions: dict[str, WorkerDecision] = {}
        decision_ids: set[str] = set()
        for raw in raw_decisions:
            if not isinstance(raw, Mapping):
                raise ValueError("each worker decision must be an object")
            decision = WorkerDecision.from_mapping(raw)
            if decision.worker in decisions:
                raise ValueError(f"duplicate worker decision: {decision.worker}")
            if decision.decision_id in decision_ids:
                raise ValueError(f"duplicate decision_id: {decision.decision_id}")
            if decision.observed_at_utc > now + timedelta(seconds=MAX_FUTURE_SKEW_SECONDS):
                raise ValueError(f"{decision.worker} observation is in the future")
            if decision.expires_at_utc <= decision.observed_at_utc:
                raise ValueError(f"{decision.worker} expiry must be after observation")
            decisions[decision.worker] = decision
            decision_ids.add(decision.decision_id)

        raw_assists = packet.get("human_assist", [])
        if not isinstance(raw_assists, Sequence) or isinstance(raw_assists, (str, bytes)):
            raise ValueError("human_assist must be a list")
        assists = []
        for raw in raw_assists:
            if not isinstance(raw, Mapping):
                raise ValueError("each human assist item must be an object")
            assist = HumanAssist.from_mapping(raw)
            if assist.observed_at_utc > now + timedelta(seconds=MAX_FUTURE_SKEW_SECONDS):
                raise ValueError("human assist observation is in the future")
            assists.append(assist)

        unexpected_workers = sorted(set(decisions) - set(_EXPECTED_WORKER.values()))
        if unexpected_workers:
            raise ValueError(f"unknown workers: {', '.join(unexpected_workers)}")

        appended: list[dict[str, Any]] = []
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.ledger_path.with_suffix(self.ledger_path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            events = _read_and_verify_ledger(self.ledger_path)
            for decision in decisions.values():
                for event in events:
                    if (
                        event.get("cycle_id") == cycle_id
                        and event.get("source_decision_id") == decision.decision_id
                        and event.get("source_decision_sha256") != _decision_sha(decision)
                    ):
                        raise ValueError(
                            f"decision identity conflict: {decision.decision_id}"
                        )
            prior = events[-1] if events else None
            state = str(prior["to_state"]) if prior else "IDLE"
            prior_cycle = str(prior["cycle_id"]) if prior else None

            if state == "BLOCKED":
                if cycle_id != prior_cycle:
                    raise ValueError("blocked ledger cannot start a new cycle")
            elif state not in _RESTARTABLE_STATES and prior_cycle != cycle_id:
                raise ValueError(
                    f"cycle {prior_cycle} is still {state}; overlapping cycle {cycle_id} is forbidden"
                )

            if prior_cycle == cycle_id and state in _TERMINAL_STATES:
                pass
            elif kill_switch:
                event = _build_event(
                    cycle_id=cycle_id,
                    sequence=len(events) + 1,
                    parent=prior,
                    worker="kill_switch",
                    source_decision_id=f"kill-switch:{cycle_id}",
                    source_decision_sha256=_canonical_sha(
                        {"cycle_id": cycle_id, "kill_switch": True}
                    ),
                    from_state=state,
                    to_state="BLOCKED",
                    requested_verdict="BLOCK",
                    system_outcome="BLOCK",
                    reason="Emergency kill switch engaged; future shadow transitions are halted.",
                    observed_at_utc=now,
                    decided_at_utc=now,
                    expires_at_utc=now + timedelta(days=3650),
                    supporting_evidence=("kill-switch:true",),
                    contradicting_evidence=(),
                    counterevidence_reviewed=True,
                    confidence=1.0,
                    uncertainty=0.0,
                    assists=assists,
                )
                _append_locked(self.ledger_path, event)
                events.append(event)
                appended.append(event)
                state = "BLOCKED"
            else:
                while state != "BLOCKED" and not (
                    state in {"LEARNED", "EXPIRED"}
                    and events
                    and events[-1].get("cycle_id") == cycle_id
                ):
                    expected = _EXPECTED_WORKER[state]
                    decision = decisions.get(expected)
                    if decision is None:
                        break
                    decision_sha256 = _decision_sha(decision)
                    prior_receipts = [
                        event
                        for event in events
                        if event.get("cycle_id") == cycle_id
                        and event.get("source_decision_id") == decision.decision_id
                    ]
                    if prior_receipts:
                        if any(
                            event.get("source_decision_sha256") != decision_sha256
                            for event in prior_receipts
                        ):
                            raise ValueError(
                                f"decision identity conflict: {decision.decision_id}"
                            )
                        break
                    event, state = self._arbitrate(
                        cycle_id=cycle_id,
                        sequence=len(events) + 1,
                        parent=events[-1] if events else None,
                        state=state,
                        decision=decision,
                        assists=assists if not appended else (),
                        now=now,
                    )
                    _append_locked(self.ledger_path, event)
                    events.append(event)
                    appended.append(event)
                    if event["system_outcome"] not in {"ADVANCE", "NO_FILL"}:
                        break
            snapshot = _build_snapshot(
                events=events,
                cycle_id=cycle_id,
                now=now,
                events_appended=len(appended),
                min_net_confidence=self.min_net_confidence,
                ledger_path=self.ledger_path,
            )
            _atomic_write(
                self.output_path,
                json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            _atomic_write(self.report_path, _render_report(snapshot))
        return AutonomousShadowCycleSummary(
            status=str(snapshot["status"]),
            state=str(snapshot["state"]),
            cycle_id=cycle_id,
            expected_worker=snapshot["expected_worker"],
            events_appended=len(appended),
            ledger_path=self.ledger_path,
            output_path=self.output_path,
            report_path=self.report_path,
        )

    def _arbitrate(
        self,
        *,
        cycle_id: str,
        sequence: int,
        parent: Mapping[str, Any] | None,
        state: str,
        decision: WorkerDecision,
        assists: Sequence[HumanAssist],
        now: datetime,
    ) -> tuple[dict[str, Any], str]:
        expected = _EXPECTED_WORKER[state]
        if decision.worker != expected:
            raise ValueError(f"state {state} requires worker {expected}, got {decision.worker}")

        outcome = decision.verdict
        reason = decision.reason
        if now >= decision.expires_at_utc:
            outcome = "EXPIRE"
            reason = f"{decision.worker} receipt expired before arbitration"
        elif decision.verdict == "ADVANCE" and (
            decision.confidence - decision.uncertainty < self.min_net_confidence
        ):
            outcome = "WAIT"
            reason = (
                f"net confidence below floor: {decision.confidence - decision.uncertainty:.6f} "
                f"< {self.min_net_confidence:.6f}"
            )
        elif state in {"HYPOTHESIS", "CHALLENGED"} and decision.verdict == "ADVANCE" and not decision.counterevidence_reviewed:
            outcome = "WAIT"
            reason = f"{decision.worker} cannot advance before counterevidence review"

        if outcome == "NO_FILL":
            if state != "ADMITTED":
                raise ValueError("NO_FILL is valid only for fill_truth at ADMITTED")
            next_state = "UNFILLED"
        elif outcome == "ADVANCE":
            next_state = _ADVANCE_STATE[state]
        elif outcome == "BLOCK":
            next_state = "BLOCKED"
        elif outcome == "EXPIRE":
            next_state = "EXPIRED"
        else:
            next_state = state

        event = _build_event(
            cycle_id=cycle_id,
            sequence=sequence,
            parent=parent,
            worker=decision.worker,
            source_decision_id=decision.decision_id,
            source_decision_sha256=_decision_sha(decision),
            from_state=state,
            to_state=next_state,
            requested_verdict=decision.verdict,
            system_outcome=outcome,
            reason=reason,
            observed_at_utc=decision.observed_at_utc,
            decided_at_utc=now,
            expires_at_utc=decision.expires_at_utc,
            supporting_evidence=decision.supporting_evidence,
            contradicting_evidence=decision.contradicting_evidence,
            counterevidence_reviewed=decision.counterevidence_reviewed,
            confidence=decision.confidence,
            uncertainty=decision.uncertainty,
            assists=assists,
        )
        return event, next_state


def _build_event(
    *,
    cycle_id: str,
    sequence: int,
    parent: Mapping[str, Any] | None,
    worker: str,
    source_decision_id: str,
    source_decision_sha256: str,
    from_state: str,
    to_state: str,
    requested_verdict: str,
    system_outcome: str,
    reason: str,
    observed_at_utc: datetime,
    decided_at_utc: datetime,
    expires_at_utc: datetime,
    supporting_evidence: Sequence[str],
    contradicting_evidence: Sequence[str],
    counterevidence_reviewed: bool,
    confidence: float,
    uncertainty: float,
    assists: Sequence[HumanAssist],
) -> dict[str, Any]:
    body = {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "sequence": sequence,
        "parent_event_id": parent.get("event_id") if parent else None,
        "worker": worker,
        "source_decision_id": source_decision_id,
        "source_decision_sha256": source_decision_sha256,
        "from_state": from_state,
        "to_state": to_state,
        "requested_verdict": requested_verdict,
        "system_outcome": system_outcome,
        "reason": reason,
        "observed_at_utc": _format_utc(observed_at_utc),
        "decided_at_utc": _format_utc(decided_at_utc),
        "expires_at_utc": _format_utc(expires_at_utc),
        "supporting_evidence": list(supporting_evidence),
        "contradicting_evidence": list(contradicting_evidence),
        "counterevidence_reviewed": counterevidence_reviewed,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "net_confidence": confidence - uncertainty,
        "human_assist": [assist.as_dict() for assist in assists],
        "human_role": HUMAN_ROLE,
        "human_approval_required": False,
        "allowed_actions": ["APPEND_SHADOW_EVENT", "ADVANCE_SHADOW_STATE"],
        "forbidden_actions": list(_FORBIDDEN_ACTIONS),
        "next_owner": (
            "SYSTEM_STOP" if to_state in _TERMINAL_STATES else _EXPECTED_WORKER.get(to_state, "SYSTEM_STOP")
        ),
        "execution_authority": EXECUTION_AUTHORITY,
        "manual_tagless_policy": MANUAL_TAGLESS_POLICY,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
    }
    content_sha256 = _canonical_sha(body)
    return {
        **body,
        "content_sha256": content_sha256,
        "event_id": f"asns:{content_sha256[:32]}",
    }


def _read_and_verify_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid ledger JSON at line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"ledger line {line_number} is not an object")
            _verify_event(value, events[-1] if events else None, line_number)
            events.append(value)
    return events


def _verify_event(value: Mapping[str, Any], prior: Mapping[str, Any] | None, line_number: int) -> None:
    content_sha256 = str(value.get("content_sha256") or "")
    body = {key: item for key, item in value.items() if key not in {"event_id", "content_sha256"}}
    if not _SHA256_RE.fullmatch(content_sha256) or _canonical_sha(body) != content_sha256:
        raise ValueError(f"ledger content hash mismatch at line {line_number}")
    if value.get("event_id") != f"asns:{content_sha256[:32]}":
        raise ValueError(f"ledger event identity mismatch at line {line_number}")
    if value.get("contract") != CONTRACT or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"ledger contract mismatch at line {line_number}")
    if value.get("from_state") not in _STATES or value.get("to_state") not in _STATES:
        raise ValueError(f"ledger state mismatch at line {line_number}")
    expected_parent = prior.get("event_id") if prior else None
    expected_from = prior.get("to_state") if prior else "IDLE"
    expected_sequence = int(prior.get("sequence")) + 1 if prior else 1
    if value.get("parent_event_id") != expected_parent:
        raise ValueError(f"ledger parent mismatch at line {line_number}")
    if value.get("from_state") != expected_from:
        raise ValueError(f"ledger transition chain mismatch at line {line_number}")
    if value.get("sequence") != expected_sequence:
        raise ValueError(f"ledger sequence mismatch at line {line_number}")
    from_state = str(value.get("from_state"))
    outcome = str(value.get("system_outcome"))
    worker = str(value.get("worker"))
    if worker == "kill_switch":
        if outcome != "BLOCK" or value.get("to_state") != "BLOCKED":
            raise ValueError(f"ledger kill-switch transition mismatch at line {line_number}")
    else:
        expected_worker = _EXPECTED_WORKER.get(from_state)
        if expected_worker != worker:
            raise ValueError(f"ledger worker ownership mismatch at line {line_number}")
        expected_to_state = {
            "ADVANCE": _ADVANCE_STATE[from_state],
            "NO_FILL": "UNFILLED" if from_state == "ADMITTED" else None,
            "WAIT": from_state,
            "BLOCK": "BLOCKED",
            "EXPIRE": "EXPIRED",
        }.get(outcome)
        if value.get("to_state") != expected_to_state:
            raise ValueError(f"ledger outcome transition mismatch at line {line_number}")
    if not _ID_RE.fullmatch(str(value.get("source_decision_id") or "")):
        raise ValueError(f"ledger decision identity mismatch at line {line_number}")
    if not _SHA256_RE.fullmatch(str(value.get("source_decision_sha256") or "")):
        raise ValueError(f"ledger decision digest mismatch at line {line_number}")
    if (
        value.get("execution_authority") != EXECUTION_AUTHORITY
        or value.get("manual_tagless_policy") != MANUAL_TAGLESS_POLICY
        or value.get("shadow_only") is not True
        or value.get("live_permission") is not False
        or value.get("broker_mutation_allowed") is not False
        or value.get("external_order_attempts") != 0
        or value.get("external_orders") != 0
        or value.get("human_approval_required") is not False
    ):
        raise ValueError(f"ledger authority invariant mismatch at line {line_number}")
    for assist in value.get("human_assist") or []:
        if (
            not isinstance(assist, Mapping)
            or assist.get("role") != HUMAN_ROLE
            or assist.get("can_approve_transition") is not False
            or assist.get("can_grant_live_permission") is not False
        ):
            raise ValueError(f"ledger human-assist invariant mismatch at line {line_number}")


def _append_locked(path: Path, event: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        target.flush()
        os.fsync(target.fileno())


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _render_report(snapshot: Mapping[str, Any]) -> str:
    expected = snapshot.get("expected_worker") or "none"
    return "\n".join(
        (
            "# Autonomous Shadow Nervous System",
            "",
            f"- Cycle: `{snapshot['cycle_id']}`",
            f"- Status: `{snapshot['status']}`",
            f"- State: `{snapshot['state']}`",
            f"- Next worker: `{expected}`",
            f"- Events appended: `{snapshot['events_appended']}`",
            "- Human role: `ASSIST` (evidence-only; approval is not in the normal path)",
            "- Execution authority: `NONE`",
            "- Manual/tagless positions: `NO_TOUCH`",
            "- Broker mutation: `False`",
            "",
        )
    )


def _build_snapshot(
    *,
    events: Sequence[Mapping[str, Any]],
    cycle_id: str,
    now: datetime,
    events_appended: int,
    min_net_confidence: float,
    ledger_path: Path,
) -> dict[str, Any]:
    latest = events[-1] if events else None
    state = str(latest["to_state"]) if latest else "IDLE"
    status = _status_for(state, cycle_id=cycle_id, latest=latest)
    expected_worker = (
        None
        if latest
        and latest.get("cycle_id") == cycle_id
        and state in _TERMINAL_STATES
        else _EXPECTED_WORKER.get(state)
    )
    return {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _format_utc(now),
        "cycle_id": cycle_id,
        "status": status,
        "state": state,
        "expected_worker": expected_worker,
        "events_appended": events_appended,
        "latest_event_id": latest.get("event_id") if latest else None,
        "human_role": HUMAN_ROLE,
        "human_approval_required": False,
        "human_assist_is_evidence_only": True,
        "execution_authority": EXECUTION_AUTHORITY,
        "manual_tagless_policy": MANUAL_TAGLESS_POLICY,
        "shadow_only": True,
        "live_permission_allowed": False,
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "min_net_confidence": min_net_confidence,
        "forbidden_actions": list(_FORBIDDEN_ACTIONS),
        "ledger_path": str(ledger_path),
    }


def _status_for(state: str, *, cycle_id: str, latest: Mapping[str, Any] | None) -> str:
    if state == "LEARNED" and latest and latest.get("cycle_id") == cycle_id:
        return "COMPLETE"
    if state == "EXPIRED" and latest and latest.get("cycle_id") == cycle_id:
        return "EXPIRED"
    if state == "BLOCKED":
        return "BLOCKED"
    return f"WAITING_FOR_{_EXPECTED_WORKER[state].upper()}"


def _require_exact_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {', '.join(unknown)}")


def _bounded_id(value: Any, label: str) -> str:
    identity = str(value or "").strip()
    if not _ID_RE.fullmatch(identity):
        raise ValueError(f"{label} is invalid")
    return identity


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number in [0, 1]") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a number in [0, 1]")
    return number


def _evidence_refs(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a list")
    refs = tuple(str(item).strip() for item in value)
    if any(not item or len(item) > 512 for item in refs):
        raise ValueError(f"{label} contains an invalid reference")
    if len(refs) != len(set(refs)):
        raise ValueError(f"{label} contains duplicate references")
    return refs


def _parse_utc(value: Any, label: str) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    return value.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return _aware(value).isoformat().replace("+00:00", "Z")


def _canonical_sha(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decision_sha(decision: WorkerDecision) -> str:
    return _canonical_sha(
        {
            "decision_id": decision.decision_id,
            "worker": decision.worker,
            "verdict": decision.verdict,
            "reason": decision.reason,
            "observed_at_utc": _format_utc(decision.observed_at_utc),
            "expires_at_utc": _format_utc(decision.expires_at_utc),
            "supporting_evidence": list(decision.supporting_evidence),
            "contradicting_evidence": list(decision.contradicting_evidence),
            "counterevidence_reviewed": decision.counterevidence_reviewed,
            "confidence": decision.confidence,
            "uncertainty": decision.uncertainty,
        }
    )
