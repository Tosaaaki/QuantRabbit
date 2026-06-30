from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.instruments import NORMAL_SPREAD_PIPS, instrument_pip_factor
from quant_rabbit.models import Owner
from quant_rabbit.risk import RiskPolicy


EVENT_TYPES = (
    "FAILED_ACCEPTANCE",
    "ACCEPTANCE_BREAK",
    "SESSION_EXPANSION",
    "RANGE_RAIL_TOUCH",
    "THEME_CONFIRMATION",
    "SQUEEZE_RELEASE",
    "HARVEST_ZONE",
    "THESIS_INVALIDATION",
    "MARGIN_PRESSURE",
    "UNKNOWN_ORDER",
    "STALE_PENDING",
    "BROKER_SNAPSHOT_STALE",
    "SPREAD_ANOMALY",
    "UNEXPECTED_PROTECTION_MISSING",
    "CONTRACT_HARVEST_TRIGGER",
    "CONTRACT_ADD_TRIGGER",
    "CONTRACT_NO_ADD_TRIGGER",
    "CONTRACT_WOUNDED_TRIGGER",
    "CONTRACT_INVALIDATION_TRIGGER",
    "CONTRACT_EMERGENCY_TRIGGER",
    "CONTRACT_STALE",
    "WAKE_PARSE_FAILURE",
)
SEVERITY_RANK = {"P2": 1, "P1": 2, "P0": 3}
THESIS_STATE_RANK = {"UNKNOWN": 0, "ALIVE": 1, "WOUNDED": 2, "INVALIDATED": 3, "EMERGENCY": 4}
GUARDIAN_ACTIONS = ("TRADE", "ADD", "HOLD", "HARVEST", "REDUCE", "CANCEL_PENDING", "NO_ACTION")

# Cadence contract, not market geometry: a repeated identical state should not
# spend a discretionary wake again inside the normal trader cadence window.
DEFAULT_THROTTLE_SECONDS = 30 * 60

# Fresh entries/adds carry live-risk blast radius, so their same-thesis retry
# throttle is deliberately longer than read-only harvest/review prompts.
DEFAULT_FRESH_ACTION_THROTTLE_SECONDS = 60 * 60

# A stale pending order has sat for roughly one hour. This guardian lifecycle
# watchdog is independent of the hourly full-trader cadence; the deterministic
# router still checks frequently.
DEFAULT_STALE_PENDING_SECONDS = 3 * 20 * 60

# Early margin pressure uses the existing gateway cap as its anchor. At 90% of
# the cap, the next 1000u order can easily fail after quote drift; gateway still
# makes the final broker-truth decision.
MARGIN_PRESSURE_WARNING_CAP_FRACTION = 0.90

# Statistical rail touch bands; they describe where price is at the tail of the
# observed 24h distribution, not a fixed pip level or USD/JPY literal.
RANGE_RAIL_LOW_PERCENTILE = 0.08
RANGE_RAIL_HIGH_PERCENTILE = 0.92

# Contract freshness follows the documented hourly trader cadence with a small
# scheduling/sync grace window. It is a runtime cadence guard, not market logic.
DEFAULT_CONTRACT_MAX_AGE_SECONDS = 75 * 60
DEFAULT_CONTRACT_REVIEW_DEADLINE_SECONDS = 60 * 60

# Broker snapshot stale detection mirrors the guardian wake dispatcher freshness
# window: stale quote truth should wake review rather than feed market triggers.
DEFAULT_ROUTER_SNAPSHOT_MAX_AGE_SECONDS = 5 * 60

CONTRACT_TRIGGER_BUCKETS = {
    "harvest_triggers": ("CONTRACT_HARVEST_TRIGGER", "HARVEST", "HARVEST_REVIEW", "P1", None),
    "add_triggers": ("CONTRACT_ADD_TRIGGER", "ADD", "ADD_REVIEW", "P1", "ALIVE"),
    "no_add_triggers": ("CONTRACT_NO_ADD_TRIGGER", "HOLD", "ADD_REVIEW", "P1", "WOUNDED"),
    "wounded_triggers": ("CONTRACT_WOUNDED_TRIGGER", "HOLD", "THESIS_REVIEW", "P1", "WOUNDED"),
    "invalidation_triggers": ("CONTRACT_INVALIDATION_TRIGGER", "REDUCE", "THESIS_REVIEW", "P0", "INVALIDATED"),
    "emergency_triggers": ("CONTRACT_EMERGENCY_TRIGGER", "REDUCE", "EMERGENCY_RISK_REVIEW", "P0", "EMERGENCY"),
}
CONTRACT_OWNERS = {"SYSTEM", "OPERATOR_MANUAL", "UNKNOWN"}


@dataclass(frozen=True)
class GuardianEvent:
    event_id: str
    event_type: str
    pair: str
    direction: str | None
    thesis: str
    price_zone: str
    severity: str
    recommended_review_type: str
    dedupe_key: str
    action_hint: str
    thesis_state: str = "UNKNOWN"
    detected_at_utc: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.direction is None:
            payload["direction"] = None
        return payload


@dataclass(frozen=True)
class GuardianRouterSummary:
    status: str
    events_path: Path
    escalation_path: Path
    state_path: Path
    report_path: Path
    action_receipt_path: Path | None
    action_review_report_path: Path
    event_count: int
    wake_gpt: bool
    wake_reasons: tuple[str, ...]
    action_review_status: str | None = None


def run_guardian_event_router(
    *,
    snapshot_path: Path,
    pair_charts_path: Path,
    order_intents_path: Path,
    position_management_path: Path,
    thesis_evolution_path: Path,
    forecast_persistence_path: Path,
    market_context_matrix_path: Path,
    trigger_contract_path: Path | None = None,
    wake_dispatcher_state_path: Path | None = None,
    state_path: Path,
    events_output_path: Path,
    escalation_output_path: Path,
    report_path: Path,
    action_receipt_input_path: Path | None = None,
    action_receipt_output_path: Path | None = None,
    action_review_report_path: Path,
    now: datetime | None = None,
) -> GuardianRouterSummary:
    now = _utc(now)
    inputs = {
        "snapshot": _load_json(snapshot_path),
        "pair_charts": _load_json(pair_charts_path),
        "order_intents": _load_json(order_intents_path),
        "position_management": _load_json(position_management_path),
        "thesis_evolution": _load_json(thesis_evolution_path),
        "forecast_persistence": _load_json(forecast_persistence_path),
        "market_context_matrix": _load_json(market_context_matrix_path),
        "trigger_contract": _load_json(trigger_contract_path),
        "wake_dispatcher_state": _load_json(wake_dispatcher_state_path),
    }
    previous_state = _load_json(state_path)
    events = detect_guardian_events(inputs=inputs, now=now)
    escalation, next_state = evaluate_guardian_escalation(
        events=events,
        previous_state=previous_state,
        now=now,
    )

    events_payload = {
        "generated_at_utc": now.isoformat(),
        "schema_version": 1,
        "event_types": list(EVENT_TYPES),
        "events": [event.to_payload() for event in events],
        "trigger_contract": validate_guardian_trigger_contract(inputs["trigger_contract"], now=now),
        "inputs": {
            "snapshot": str(snapshot_path),
            "pair_charts": str(pair_charts_path),
            "order_intents": str(order_intents_path),
            "position_management": str(position_management_path),
            "thesis_evolution": str(thesis_evolution_path),
            "forecast_persistence": str(forecast_persistence_path),
            "market_context_matrix": str(market_context_matrix_path),
            "guardian_trigger_contract": str(trigger_contract_path) if trigger_contract_path is not None else None,
            "guardian_wake_dispatcher_state": str(wake_dispatcher_state_path)
            if wake_dispatcher_state_path is not None
            else None,
        },
    }
    _write_json(events_output_path, events_payload)
    _write_json(escalation_output_path, escalation)
    _write_json(state_path, next_state)
    _write_event_report(report_path, events_payload, escalation)

    action_review_status = None
    written_action_receipt_path: Path | None = None
    if action_receipt_input_path is not None and action_receipt_input_path.exists():
        action_payload = _load_json(action_receipt_input_path)
        action_review = review_guardian_action_receipt(
            action_payload,
            events=events,
            previous_state=previous_state,
            now=now,
        )
        if action_receipt_output_path is not None:
            written_action_receipt_path = action_receipt_output_path
            _write_json(action_receipt_output_path, action_review)
        _write_action_review_report(action_review_report_path, action_review)
        action_review_status = str(action_review.get("status") or "UNKNOWN")
    else:
        _write_action_review_report(
            action_review_report_path,
            {
                "generated_at_utc": now.isoformat(),
                "status": "NO_ACTION_RECEIPT",
                "gateway_required": True,
                "issues": [],
                "summary": "No GPT wake action receipt was provided; guardian router stayed read-only.",
            },
        )

    return GuardianRouterSummary(
        status="OK",
        events_path=events_output_path,
        escalation_path=escalation_output_path,
        state_path=state_path,
        report_path=report_path,
        action_receipt_path=written_action_receipt_path,
        action_review_report_path=action_review_report_path,
        event_count=len(events),
        wake_gpt=bool(escalation.get("wake_gpt")),
        wake_reasons=tuple(str(item) for item in escalation.get("wake_reason_codes", []) or []),
        action_review_status=action_review_status,
    )


def detect_guardian_events(*, inputs: dict[str, Any], now: datetime | None = None) -> list[GuardianEvent]:
    now = _utc(now)
    collected: list[GuardianEvent] = []
    snapshot = inputs.get("snapshot") if isinstance(inputs.get("snapshot"), dict) else {}
    pair_charts = inputs.get("pair_charts") if isinstance(inputs.get("pair_charts"), dict) else {}
    order_intents = inputs.get("order_intents") if isinstance(inputs.get("order_intents"), dict) else {}
    position_management = (
        inputs.get("position_management") if isinstance(inputs.get("position_management"), dict) else {}
    )
    thesis_evolution = inputs.get("thesis_evolution") if isinstance(inputs.get("thesis_evolution"), dict) else {}
    forecast_persistence = (
        inputs.get("forecast_persistence") if isinstance(inputs.get("forecast_persistence"), dict) else {}
    )
    market_context_matrix = (
        inputs.get("market_context_matrix") if isinstance(inputs.get("market_context_matrix"), dict) else {}
    )
    trigger_contract = inputs.get("trigger_contract") if isinstance(inputs.get("trigger_contract"), dict) else {}
    wake_dispatcher_state = (
        inputs.get("wake_dispatcher_state") if isinstance(inputs.get("wake_dispatcher_state"), dict) else {}
    )

    collected.extend(_snapshot_events(snapshot, now=now))
    collected.extend(_contract_events(trigger_contract, snapshot=snapshot, now=now))
    collected.extend(_wake_parse_failure_events(wake_dispatcher_state, now=now))
    collected.extend(_pair_chart_events(pair_charts, snapshot=snapshot, now=now))
    collected.extend(_order_intent_events(order_intents, now=now))
    collected.extend(_position_management_events(position_management, now=now))
    collected.extend(_thesis_evolution_events(thesis_evolution, now=now))
    collected.extend(_forecast_persistence_events(forecast_persistence, now=now))
    collected.extend(_market_context_matrix_events(market_context_matrix, now=now))
    return _dedupe_events(collected)


def evaluate_guardian_escalation(
    *,
    events: list[GuardianEvent],
    previous_state: dict[str, Any] | None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = _utc(now)
    previous = previous_state if isinstance(previous_state, dict) else {}
    previous_events = previous.get("events") if isinstance(previous.get("events"), dict) else {}
    review_events: list[dict[str, Any]] = []
    suppressed_events: list[dict[str, Any]] = []
    wake_reason_codes: list[str] = []
    next_events: dict[str, Any] = {}

    for event in events:
        prior = previous_events.get(event.dedupe_key) if isinstance(previous_events, dict) else None
        if not isinstance(prior, dict):
            prior = None
        reasons = _wake_reasons_for_event(event, prior)
        last_wake = _parse_utc((prior or {}).get("last_wake_at_utc"))
        throttle_seconds = _event_throttle_seconds(event)
        bypass_throttle = _event_bypasses_throttle(event, reasons)
        throttled = False
        if reasons and last_wake is not None and not bypass_throttle:
            throttled = (now - last_wake).total_seconds() < throttle_seconds
        event_payload = event.to_payload()
        if reasons and not throttled:
            review_events.append({**event_payload, "wake_reason_codes": reasons})
            wake_reason_codes.extend(reasons)
            last_wake_at = now.isoformat()
        else:
            last_wake_at = (prior or {}).get("last_wake_at_utc")
            if reasons and throttled:
                suppressed_events.append(
                    {
                        **event_payload,
                        "suppressed_reason": "THROTTLED",
                        "wake_reason_codes": reasons,
                        "throttle_seconds": throttle_seconds,
                        "last_wake_at_utc": last_wake_at,
                    }
                )
            elif not reasons:
                repeated_in_throttle = (
                    prior is not None
                    and last_wake is not None
                    and (now - last_wake).total_seconds() < throttle_seconds
                )
                suppressed_events.append(
                    {
                        **event_payload,
                        "suppressed_reason": "THROTTLED" if repeated_in_throttle else "NO_STATE_CHANGE",
                        "throttle_seconds": throttle_seconds,
                        "last_wake_at_utc": last_wake_at,
                    }
                )
        next_events[event.dedupe_key] = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "pair": event.pair,
            "direction": event.direction,
            "thesis": event.thesis,
            "severity": event.severity,
            "recommended_review_type": event.recommended_review_type,
            "action_hint": event.action_hint,
            "thesis_state": event.thesis_state,
            "in_harvest_zone": event.event_type == "HARVEST_ZONE",
            "margin_pressure": event.event_type == "MARGIN_PRESSURE",
            "last_seen_at_utc": now.isoformat(),
            "last_wake_at_utc": last_wake_at,
        }

    escalation = {
        "generated_at_utc": now.isoformat(),
        "wake_gpt": bool(review_events),
        "model_target": "GPT-5.5",
        "wake_policy": "state_change_only",
        "execution_boundary": {
            "guardian_never_trades": True,
            "gpt_wake_never_calls_oanda_directly": True,
            "live_order_gateway_required": True,
        },
        "wake_reason_codes": sorted(set(wake_reason_codes)),
        "events_to_review": review_events,
        "suppressed_events": suppressed_events,
        "throttle": {
            "default_seconds": DEFAULT_THROTTLE_SECONDS,
            "fresh_entry_or_add_seconds": DEFAULT_FRESH_ACTION_THROTTLE_SECONDS,
            "p0_and_harvest_bypass_on_state_change": True,
        },
    }
    next_state = {
        "generated_at_utc": now.isoformat(),
        "events": next_events,
    }
    return escalation, next_state


def review_guardian_action_receipt(
    receipt_payload: dict[str, Any] | None,
    *,
    events: list[GuardianEvent],
    previous_state: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = _utc(now)
    payload = receipt_payload if isinstance(receipt_payload, dict) else {}
    receipt = payload.get("receipt") if isinstance(payload.get("receipt"), dict) else payload
    action = str(receipt.get("action") or "").strip().upper()
    event_id = str(receipt.get("event_id") or "").strip()
    event_by_id = {event.event_id: event for event in events}
    event = event_by_id.get(event_id)
    issues: list[dict[str, str]] = []

    if action not in GUARDIAN_ACTIONS:
        issues.append(_issue("GUARDIAN_ACTION_BAD_ACTION", f"unsupported guardian action {action!r}"))
    for field_name in (
        "new_information",
        "event_id",
        "thesis_state",
        "reason",
        "invalidation",
        "harvest_trigger",
        "gateway_required",
    ):
        if field_name not in receipt:
            issues.append(_issue("GUARDIAN_ACTION_FIELD_MISSING", f"guardian action receipt missing {field_name}"))
    if event_id and event is None:
        issues.append(_issue("GUARDIAN_ACTION_UNKNOWN_EVENT", f"event_id {event_id} is not in current guardian_events"))
    if receipt.get("gateway_required") is not True:
        issues.append(_issue("GUARDIAN_ACTION_GATEWAY_REQUIRED", "guardian action receipt must set gateway_required=true"))

    if action in {"TRADE", "ADD"}:
        issues.extend(_guardian_trade_add_action_issues(receipt, event=event))

    status = "ACCEPTED" if not any(issue["severity"] == "BLOCK" for issue in issues) else "REJECTED"
    reviewed = {
        "generated_at_utc": now.isoformat(),
        "status": status,
        "receipt": receipt,
        "event": event.to_payload() if event is not None else None,
        "issues": issues,
        "gateway_required": True,
        "execution_boundary": {
            "guardian_never_trades": True,
            "gpt_wake_never_calls_oanda_directly": True,
            "only_live_order_gateway_may_send_cancel_close": True,
        },
    }
    return reviewed


def guardian_action_gateway_issues(
    *,
    intent_metadata: dict[str, Any],
    pair: str,
    thesis: str,
    action_receipt_path: Path | None,
) -> list[dict[str, str]]:
    if not _metadata_marks_guardian_wake(intent_metadata):
        return []
    if action_receipt_path is None or not action_receipt_path.exists():
        return [
            _issue(
                "GUARDIAN_ACTION_RECEIPT_REQUIRED",
                "guardian wake intent requires data/guardian_action_receipt.json before LiveOrderGateway can stage TRADE/ADD",
            )
        ]
    payload = _load_json(action_receipt_path)
    receipt = payload.get("receipt") if isinstance(payload.get("receipt"), dict) else payload
    action = str(receipt.get("action") or "").strip().upper()
    issues = []
    if action not in {"TRADE", "ADD"}:
        issues.append(
            _issue(
                "GUARDIAN_ACTION_NOT_TRADE_OR_ADD",
                f"guardian wake intent cannot be staged from action={action or 'missing'}",
            )
        )
    if str(receipt.get("pair") or pair).upper() != pair.upper():
        issues.append(_issue("GUARDIAN_ACTION_PAIR_MISMATCH", "guardian action pair does not match selected intent"))
    receipt_thesis = str(receipt.get("thesis") or thesis).strip()
    if receipt_thesis and _slug(receipt_thesis) != _slug(thesis):
        issues.append(
            _issue("GUARDIAN_ACTION_THESIS_MISMATCH", "guardian action thesis does not match selected intent thesis")
        )
    event_id = str(intent_metadata.get("guardian_event_id") or "").strip()
    if event_id and str(receipt.get("event_id") or "").strip() != event_id:
        issues.append(_issue("GUARDIAN_ACTION_EVENT_MISMATCH", "guardian action event_id does not match intent metadata"))
    issues.extend(_guardian_trade_add_action_issues(receipt, event=None, intent_metadata=intent_metadata))
    return issues


def validate_guardian_trigger_contract(
    payload: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    clock = _utc(now)
    contract = payload if isinstance(payload, dict) else {}
    issues: list[dict[str, str]] = []
    generated_at = _parse_utc(contract.get("generated_at_utc"))
    if not contract:
        issues.append(_issue("CONTRACT_MISSING", "guardian trigger contract is missing"))
    if contract.get("schema_version") != 1:
        issues.append(_issue("CONTRACT_BAD_SCHEMA_VERSION", "guardian trigger contract schema_version must be 1"))
    if generated_at is None:
        issues.append(_issue("CONTRACT_GENERATED_AT_MISSING", "guardian trigger contract needs generated_at_utc"))
    entries = _contract_entries(contract)
    if "entries" not in contract:
        issues.append(_issue("CONTRACT_ENTRIES_MISSING", "guardian trigger contract needs entries[]"))
    elif not isinstance(contract.get("entries"), list):
        issues.append(_issue("CONTRACT_ENTRIES_NOT_LIST", "guardian trigger contract entries must be a list"))

    deadline_expired = False
    for index, entry in enumerate(entries):
        prefix = f"entries[{index}]"
        for field_name in (
            "pair",
            "side",
            "thesis",
            "owner",
            "thesis_state",
            "next_review_reason",
            "next_review_deadline_utc",
        ):
            if field_name not in entry:
                issues.append(_issue("CONTRACT_ENTRY_FIELD_MISSING", f"{prefix} missing {field_name}"))
        pair = _pair(entry.get("pair"))
        if not pair:
            issues.append(_issue("CONTRACT_ENTRY_BAD_PAIR", f"{prefix} pair is empty"))
        side = _direction_from_text(entry.get("side"))
        if side is None:
            issues.append(_issue("CONTRACT_ENTRY_BAD_SIDE", f"{prefix} side must be LONG or SHORT"))
        raw_owner = str(entry.get("owner") or "").strip().upper()
        if raw_owner not in CONTRACT_OWNERS:
            issues.append(_issue("CONTRACT_ENTRY_BAD_OWNER", f"{prefix} owner must be SYSTEM, OPERATOR_MANUAL, or UNKNOWN"))
        thesis_state = _thesis_state(entry.get("thesis_state"))
        if thesis_state not in {"ALIVE", "WOUNDED", "INVALIDATED", "EMERGENCY"}:
            issues.append(_issue("CONTRACT_ENTRY_BAD_THESIS_STATE", f"{prefix} thesis_state is unsupported"))
        deadline = _parse_utc(entry.get("next_review_deadline_utc"))
        if deadline is None:
            issues.append(
                _issue("CONTRACT_ENTRY_BAD_DEADLINE", f"{prefix} next_review_deadline_utc must be ISO UTC")
            )
        elif deadline <= clock:
            deadline_expired = True
            issues.append(_issue("CONTRACT_ENTRY_DEADLINE_EXPIRED", f"{prefix} next_review_deadline_utc is expired"))
        for bucket in CONTRACT_TRIGGER_BUCKETS:
            if bucket not in entry:
                issues.append(_issue("CONTRACT_ENTRY_TRIGGER_FIELD_MISSING", f"{prefix} missing {bucket}"))
            elif not isinstance(entry.get(bucket), list):
                issues.append(_issue("CONTRACT_ENTRY_TRIGGER_FIELD_NOT_LIST", f"{prefix} {bucket} must be a list"))
        if _contract_entry_has_open_exposure(entry):
            for bucket in ("harvest_triggers", "invalidation_triggers", "emergency_triggers"):
                if isinstance(entry.get(bucket), list) and not entry.get(bucket):
                    issues.append(
                        _issue(
                            "CONTRACT_ENTRY_OPEN_TRIGGER_EMPTY",
                            f"{prefix} open exposure needs non-empty {bucket}",
                        )
                    )

    age_seconds = (clock - generated_at).total_seconds() if generated_at is not None else None
    max_age = int(os.environ.get("QR_GUARDIAN_TRIGGER_CONTRACT_MAX_AGE_SECONDS", DEFAULT_CONTRACT_MAX_AGE_SECONDS))
    stale = age_seconds is None or age_seconds > max_age or deadline_expired
    return {
        "status": "VALID" if not any(issue["severity"] == "BLOCK" for issue in issues) else "INVALID",
        "issues": issues,
        "entry_count": len(entries),
        "generated_at_utc": generated_at.isoformat() if generated_at is not None else None,
        "age_seconds": age_seconds,
        "max_age_seconds": max_age,
        "stale": stale,
        "trigger_event_types": {bucket: values[0] for bucket, values in CONTRACT_TRIGGER_BUCKETS.items()},
    }


def build_guardian_trigger_contract(
    *,
    snapshot: dict[str, Any],
    order_intents: dict[str, Any],
    existing_contract: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    clock = _utc(now)
    existing = existing_contract if isinstance(existing_contract, dict) else {}
    preserved: dict[str, dict[str, Any]] = {}
    for entry in _contract_entries(existing):
        preserved[_contract_entry_key(entry)] = entry
        preserved[_contract_entry_legacy_key(entry)] = entry
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    for position in snapshot.get("positions", []) or []:
        if not isinstance(position, dict):
            continue
        pair = _pair(position.get("pair"))
        side = _direction_from_position(position)
        if not pair or not side:
            continue
        thesis = _position_contract_thesis(position)
        entry = _contract_entry_from_seed(
            pair=pair,
            side=side,
            thesis=thesis,
            owner=_contract_owner_from_snapshot_owner(position.get("owner")),
            thesis_state=_position_contract_state(position),
            seed_reason="open broker position requires guardian triggers",
            preserved=preserved,
            now=clock,
            position=position,
            open_exposure=True,
        )
        seen.add(_contract_entry_key(entry))
        entries.append(entry)

    for result in order_intents.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
        pair = _pair(intent.get("pair"))
        side = _direction_from_text(intent.get("side"))
        if not pair or not side:
            continue
        thesis = str(intent.get("thesis") or result.get("lane_id") or "candidate thesis")
        entry = _contract_entry_from_seed(
            pair=pair,
            side=side,
            thesis=thesis,
            owner="SYSTEM",
            thesis_state=str((intent.get("metadata") or {}).get("thesis_state") or "ALIVE").upper()
            if isinstance(intent.get("metadata"), dict)
            else "ALIVE",
            seed_reason=f"candidate {result.get('lane_id') or pair} requires trader-defined triggers",
            preserved=preserved,
            now=clock,
            position={},
            open_exposure=False,
        )
        key = _contract_entry_key(entry)
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)

    return {
        "schema_version": 1,
        "generated_at_utc": clock.isoformat(),
        "contract_owner": "qr-trader",
        "cycle_horizon_minutes": 60,
        "entries": entries,
        "execution_boundary": {
            "guardian_never_trades": True,
            "contract_triggers_do_not_execute": True,
            "live_order_gateway_required": True,
        },
    }


def write_guardian_trigger_contract_report(path: Path, contract: dict[str, Any], validation: dict[str, Any]) -> None:
    lines = [
        "# Guardian Trigger Contract Report",
        "",
        f"- Generated at UTC: `{contract.get('generated_at_utc')}`",
        f"- Validation: `{validation.get('status')}`",
        f"- Entries: `{validation.get('entry_count')}`",
        f"- Stale: `{validation.get('stale')}` age=`{validation.get('age_seconds')}`s",
        "",
        "## Boundary",
        "",
        "- The trigger contract is read-only evidence for guardian-event-router.",
        "- Contract triggers never execute broker writes.",
        "- Plain trigger prose is not inferred; a trigger fires only when the contract marks it fired or gives a machine-readable predicate that evaluates true.",
        "- Live orders, adds, harvests, cancels, and closes remain gateway-only.",
        "",
        "## Issues",
        "",
    ]
    for issue in validation.get("issues", []) or []:
        lines.append(f"- `{issue.get('severity')}` `{issue.get('code')}` {issue.get('message')}")
    if not validation.get("issues"):
        lines.append("- none")
    lines.extend(["", "## Entries", ""])
    for entry in _contract_entries(contract):
        lines.append(
            f"- `{_pair(entry.get('pair')) or 'UNKNOWN'}` `{_direction_from_text(entry.get('side')) or 'UNKNOWN'}` "
            f"owner=`{_contract_owner(entry.get('owner'))}` state=`{_thesis_state(entry.get('thesis_state'))}`"
        )
        lines.append(f"  - thesis: {entry.get('thesis')}")
        lines.append(f"  - next review: {entry.get('next_review_reason')} by `{entry.get('next_review_deadline_utc')}`")
        for bucket in CONTRACT_TRIGGER_BUCKETS:
            triggers = entry.get(bucket) if isinstance(entry.get(bucket), list) else []
            fired_count = sum(1 for trigger in triggers if _trigger_explicitly_fired(trigger))
            lines.append(f"  - {bucket}: `{len(triggers)}` declared, `{fired_count}` explicitly fired")
    if not _contract_entries(contract):
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _contract_events(contract: dict[str, Any], *, snapshot: dict[str, Any], now: datetime) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    validation = validate_guardian_trigger_contract(contract, now=now)
    exposure_exists = _snapshot_has_exposure(snapshot)
    if exposure_exists and (validation["status"] != "VALID" or validation["stale"]):
        events.append(
            _event(
                event_type="CONTRACT_STALE",
                pair="PORTFOLIO",
                direction=None,
                thesis="guardian trigger contract stale or missing while exposure exists",
                price_zone=_contract_stale_reason(validation),
                severity="P0",
                recommended_review_type="EMERGENCY_RISK_REVIEW",
                action_hint="HOLD",
                thesis_state="WOUNDED",
                now=now,
                details={"validation": validation},
            )
        )
        if validation["status"] != "VALID":
            return events

    for entry in _contract_entries(contract):
        pair = _pair(entry.get("pair"))
        side = _direction_from_text(entry.get("side"))
        thesis = str(entry.get("thesis") or "guardian trigger contract thesis")
        if not pair:
            continue
        for bucket, (event_type, action_hint, review_type, severity, forced_state) in CONTRACT_TRIGGER_BUCKETS.items():
            triggers = entry.get(bucket) if isinstance(entry.get(bucket), list) else []
            for trigger in triggers:
                fired, evidence = _contract_trigger_fired(trigger, entry=entry, snapshot=snapshot)
                if not fired:
                    continue
                thesis_state = str(forced_state or entry.get("thesis_state") or "ALIVE").upper()
                events.append(
                    _event(
                        event_type=event_type,
                        pair=pair,
                        direction=side,
                        thesis=thesis,
                        price_zone=evidence,
                        severity=str(_trigger_field(trigger, "severity") or severity),
                        recommended_review_type=str(_trigger_field(trigger, "recommended_review_type") or review_type),
                        action_hint=str(_trigger_field(trigger, "action_hint") or action_hint),
                        thesis_state=thesis_state,
                        now=now,
                        details={
                            "contract_bucket": bucket,
                            "contract_trigger": trigger,
                            "owner": _contract_owner(entry.get("owner")),
                            "next_review_reason": entry.get("next_review_reason"),
                            "next_review_deadline_utc": entry.get("next_review_deadline_utc"),
                        },
                    )
                )
    return events


def _wake_parse_failure_events(dispatcher_state: dict[str, Any], *, now: datetime) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    seen_failure_keys: set[str] = set()
    last_status = str(dispatcher_state.get("last_status") or "").upper()
    last_result = dispatcher_state.get("last_result") if isinstance(dispatcher_state.get("last_result"), dict) else {}
    selected = last_result.get("selected_event") if isinstance(last_result.get("selected_event"), dict) else {}
    parse_failure = last_result.get("parse_failure") if isinstance(last_result.get("parse_failure"), dict) else {}
    if last_status == "PARSE_FAILED" and parse_failure:
        seen_failure_keys.add(_wake_parse_failure_key(parse_failure))
        events.append(_wake_parse_failure_event(parse_failure, selected_event=selected, now=now))

    failures = dispatcher_state.get("parse_failures")
    if isinstance(failures, dict):
        for failure in failures.values():
            if not isinstance(failure, dict):
                continue
            failure_key = _wake_parse_failure_key(failure)
            if failure_key in seen_failure_keys:
                continue
            seen_failure_keys.add(failure_key)
            selected_for_failure = selected if str(selected.get("event_id") or "") == str(failure.get("event_id") or "") else {}
            events.append(_wake_parse_failure_event(failure, selected_event=selected_for_failure, now=now))
    return events


def _wake_parse_failure_key(parse_failure: dict[str, Any]) -> str:
    return str(parse_failure.get("dedupe_key") or parse_failure.get("event_id") or id(parse_failure))


def _wake_parse_failure_event(
    parse_failure: dict[str, Any],
    *,
    selected_event: dict[str, Any],
    now: datetime,
) -> GuardianEvent:
    pair = _pair(selected_event.get("pair") or parse_failure.get("pair")) or "PORTFOLIO"
    severity = "P0" if int(parse_failure.get("consecutive_failures") or 0) > 1 else "P1"
    thesis = str(
        selected_event.get("thesis")
        or parse_failure.get("thesis")
        or parse_failure.get("event_type")
        or "guardian wake parse failure"
    )
    return _event(
        event_type="WAKE_PARSE_FAILURE",
        pair=pair,
        direction=_direction_from_text(selected_event.get("direction") or parse_failure.get("direction")),
        thesis=thesis,
        price_zone=str(parse_failure.get("last_error") or "guardian wake produced no valid JSON receipt"),
        severity=severity,
        recommended_review_type="WAKE_REPAIR_REVIEW",
        action_hint="HOLD",
        thesis_state="WOUNDED",
        now=now,
        details={"selected_event": selected_event, "parse_failure": parse_failure},
    )


def _broker_snapshot_stale_event(snapshot: dict[str, Any], *, now: datetime) -> GuardianEvent | None:
    if not snapshot:
        return None
    fetched = _parse_utc(snapshot.get("fetched_at_utc"))
    max_age = int(os.environ.get("QR_GUARDIAN_ROUTER_SNAPSHOT_MAX_AGE_SECONDS", DEFAULT_ROUTER_SNAPSHOT_MAX_AGE_SECONDS))
    if fetched is None:
        stale_reason = "broker snapshot missing fetched_at_utc"
    elif (now - fetched).total_seconds() <= max_age:
        return None
    else:
        stale_reason = f"broker snapshot age_seconds={(now - fetched).total_seconds():.1f} max={max_age}"
    severity = "P0" if _snapshot_has_exposure(snapshot) else "P1"
    return _event(
        event_type="BROKER_SNAPSHOT_STALE",
        pair="PORTFOLIO",
        direction=None,
        thesis="broker snapshot freshness",
        price_zone=stale_reason,
        severity=severity,
        recommended_review_type="RISK_REVIEW",
        action_hint="HOLD",
        thesis_state="WOUNDED",
        now=now,
        details={"fetched_at_utc": snapshot.get("fetched_at_utc"), "max_age_seconds": max_age},
    )


def _spread_anomaly_events(quotes: dict[str, Any], *, now: datetime) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    max_multiple = RiskPolicy().max_spread_multiple
    for pair_key, quote in quotes.items():
        pair = _pair(pair_key)
        if not pair or not isinstance(quote, dict):
            continue
        bid = _float(quote.get("bid"))
        ask = _float(quote.get("ask"))
        normal = NORMAL_SPREAD_PIPS.get(pair)
        if bid is None or ask is None or normal is None or normal <= 0:
            continue
        spread_pips = max(0.0, (ask - bid) * instrument_pip_factor(pair))
        cap = normal * max_multiple
        if spread_pips <= cap:
            continue
        events.append(
            _event(
                event_type="SPREAD_ANOMALY",
                pair=pair,
                direction=None,
                thesis="spread anomaly safety trigger",
                price_zone=f"spread_pips={spread_pips:.3f} cap={cap:.3f}",
                severity="P1",
                recommended_review_type="RISK_REVIEW",
                action_hint="HOLD",
                thesis_state="WOUNDED",
                now=now,
                details={"bid": bid, "ask": ask, "spread_pips": spread_pips, "normal_spread_pips": normal},
            )
        )
    return events


def _unexpected_protection_missing_event(
    position: dict[str, Any],
    *,
    pair: str,
    side: str | None,
    now: datetime,
) -> GuardianEvent | None:
    if _owner(position.get("owner")) != Owner.TRADER.value:
        return None
    raw = position.get("raw") if isinstance(position.get("raw"), dict) else {}
    expected_tp = _truthy(position.get("expected_take_profit_required")) or _truthy(
        raw.get("expected_take_profit_required")
    )
    expected_sl = _truthy(position.get("expected_stop_loss_required")) or _truthy(
        raw.get("expected_stop_loss_required")
    )
    missing: list[str] = []
    if expected_tp and position.get("take_profit") is None:
        missing.append("take_profit")
    if expected_sl and position.get("stop_loss") is None:
        missing.append("stop_loss")
    if not missing:
        return None
    return _event(
        event_type="UNEXPECTED_PROTECTION_MISSING",
        pair=pair,
        direction=side,
        thesis=str(position.get("thesis") or position.get("trade_id") or "system position protection"),
        price_zone="missing " + ",".join(missing),
        severity="P0",
        recommended_review_type="PROTECTION_REVIEW",
        action_hint="HOLD",
        thesis_state="WOUNDED",
        now=now,
        details={"trade_id": position.get("trade_id"), "missing": missing},
    )


def _snapshot_events(snapshot: dict[str, Any], *, now: datetime) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    stale = _broker_snapshot_stale_event(snapshot, now=now)
    if stale is not None:
        events.append(stale)
    quotes = snapshot.get("quotes") if isinstance(snapshot.get("quotes"), dict) else {}
    for position in snapshot.get("positions", []) or []:
        if not isinstance(position, dict):
            continue
        pair = _pair(position.get("pair"))
        if not pair:
            continue
        owner = _owner(position.get("owner"))
        side = _direction_from_position(position)
        raw = position.get("raw") if isinstance(position.get("raw"), dict) else {}
        operator_packet = raw.get("operator_manual_position") if isinstance(raw.get("operator_manual_position"), dict) else {}
        if owner in {Owner.UNKNOWN.value, Owner.EXTERNAL.value}:
            events.append(
                _event(
                    event_type="UNKNOWN_ORDER",
                    pair=pair,
                    direction=side,
                    thesis="gateway-outside broker position",
                    price_zone=f"entry={position.get('entry_price')} owner={owner}",
                    severity="P0",
                    recommended_review_type="EMERGENCY_RISK_REVIEW",
                    action_hint="REDUCE",
                    now=now,
                    details={"trade_id": position.get("trade_id"), "owner": owner},
                )
            )
        if operator_packet:
            thesis = str(operator_packet.get("thesis") or "operator manual exposure")
            if _truthy(operator_packet.get("accepted_break_above_major_figure")):
                events.append(
                    _event(
                        event_type="THESIS_INVALIDATION",
                        pair=pair,
                        direction=side,
                        thesis=thesis,
                        price_zone=str(operator_packet.get("invalidation") or "operator manual invalidation"),
                        severity="P0",
                        recommended_review_type="OPERATOR_MANUAL_REVIEW",
                        action_hint="HOLD",
                        thesis_state="INVALIDATED",
                        now=now,
                        details={"trade_id": position.get("trade_id"), "operator_manual_position": operator_packet},
                    )
                )
            if _float(position.get("unrealized_pl_jpy")) is not None and _float(position.get("unrealized_pl_jpy")) > 0:
                events.append(
                    _event(
                        event_type="HARVEST_ZONE",
                        pair=pair,
                        direction=side,
                        thesis=thesis,
                        price_zone=str(operator_packet.get("harvest_zone") or "profitable manual/operator harvest zone"),
                        severity="P1",
                        recommended_review_type="HARVEST_REVIEW",
                        action_hint="HARVEST",
                        thesis_state=str(operator_packet.get("thesis_state") or "ALIVE").upper(),
                        now=now,
                        details={"trade_id": position.get("trade_id"), "harvest_trigger": operator_packet.get("harvest_trigger")},
                    )
                )
        protection_event = _unexpected_protection_missing_event(position, pair=pair, side=side, now=now)
        if protection_event is not None:
            events.append(protection_event)
    for order in snapshot.get("orders", []) or []:
        if not isinstance(order, dict):
            continue
        pair = _pair(order.get("pair") or order.get("instrument"))
        if not pair:
            continue
        owner = _owner(order.get("owner"))
        if owner in {Owner.UNKNOWN.value, Owner.EXTERNAL.value}:
            events.append(
                _event(
                    event_type="UNKNOWN_ORDER",
                    pair=pair,
                    direction=_direction_from_units(order.get("units")),
                    thesis="gateway-outside broker pending order",
                    price_zone=f"price={order.get('price')} owner={owner}",
                    severity="P0",
                    recommended_review_type="EMERGENCY_RISK_REVIEW",
                    action_hint="CANCEL_PENDING",
                    now=now,
                    details={"order_id": order.get("order_id"), "owner": owner},
                )
            )
        if owner == Owner.TRADER.value and _pending_order_is_stale(order, now=now):
            events.append(
                _event(
                    event_type="STALE_PENDING",
                    pair=pair,
                    direction=_direction_from_units(order.get("units")),
                    thesis=_pending_order_thesis(order),
                    price_zone=f"pending price={order.get('price')}",
                    severity="P1",
                    recommended_review_type="PENDING_CANCEL_REVIEW",
                    action_hint="CANCEL_PENDING",
                    now=now,
                    details={"order_id": order.get("order_id"), "state": order.get("state")},
                )
            )
    account = snapshot.get("account") if isinstance(snapshot.get("account"), dict) else {}
    margin_event = _margin_pressure_event(account, now=now)
    if margin_event is not None:
        events.append(margin_event)
    events.extend(_spread_anomaly_events(quotes, now=now))
    events.extend(_quote_major_figure_events(quotes, now=now))
    return events


def _pair_chart_events(pair_charts: dict[str, Any], *, snapshot: dict[str, Any], now: datetime) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    for chart in _chart_rows(pair_charts):
        pair = _pair(chart.get("pair"))
        if not pair:
            continue
        for explicit in chart.get("guardian_events", []) or []:
            if isinstance(explicit, dict):
                explicit_event = _explicit_event(explicit, pair=pair, now=now)
                if explicit_event is not None:
                    events.append(explicit_event)
        confluence = chart.get("confluence") if isinstance(chart.get("confluence"), dict) else {}
        chart_story = str(chart.get("chart_story") or "")
        event_text = " ".join(
            str(item)
            for item in (
                chart_story,
                chart.get("dominant_regime"),
                confluence.get("score_balance"),
                confluence.get("score_momentum"),
            )
        )
        mid = _chart_mid(chart, snapshot)
        events.extend(_text_acceptance_events(pair, event_text, mid=mid, now=now))
        rail_event = _range_rail_event(pair, chart, confluence, now=now)
        if rail_event is not None:
            events.append(rail_event)
        session_event = _session_expansion_event(pair, chart, confluence, now=now)
        if session_event is not None:
            events.append(session_event)
        squeeze_event = _squeeze_release_event(pair, chart, now=now)
        if squeeze_event is not None:
            events.append(squeeze_event)
    return events


def _order_intent_events(order_intents: dict[str, Any], *, now: datetime) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    for result in order_intents.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        pair = _pair(intent.get("pair"))
        if not pair:
            continue
        side = _direction_from_text(intent.get("side"))
        thesis = str(intent.get("thesis") or metadata.get("thesis") or result.get("lane_id") or "candidate thesis")
        status = str(result.get("status") or "").upper()
        if _intent_is_harvest(metadata, intent, status=status):
            events.append(
                _event(
                    event_type="HARVEST_ZONE",
                    pair=pair,
                    direction=side,
                    thesis=thesis,
                    price_zone=str(metadata.get("harvest_zone") or metadata.get("tp_target_intent") or "attached TP harvest zone"),
                    severity="P1" if status == "LIVE_READY" else "P2",
                    recommended_review_type="HARVEST_REVIEW",
                    action_hint="HARVEST",
                    thesis_state=str(metadata.get("thesis_state") or "ALIVE").upper(),
                    now=now,
                    details={"lane_id": result.get("lane_id"), "status": result.get("status")},
                )
            )
        if _truthy(metadata.get("failed_acceptance")) or str(intent.get("market_context", {}).get("method") or "") == "BREAKOUT_FAILURE":
            events.append(
                _event(
                    event_type="FAILED_ACCEPTANCE",
                    pair=pair,
                    direction=side,
                    thesis=thesis,
                    price_zone=str(metadata.get("acceptance_zone") or intent.get("entry") or "failed acceptance zone"),
                    severity="P1" if status == "LIVE_READY" else "P2",
                    recommended_review_type="ENTRY_REVIEW",
                    action_hint="TRADE",
                    now=now,
                    details={"lane_id": result.get("lane_id"), "status": result.get("status")},
                )
            )
        if _truthy(metadata.get("squeeze_release")) or "squeeze" in str(metadata.get("timing_signal") or "").lower():
            events.append(
                _event(
                    event_type="SQUEEZE_RELEASE",
                    pair=pair,
                    direction=side,
                    thesis=thesis,
                    price_zone=str(intent.get("entry") or "squeeze trigger zone"),
                    severity="P1",
                    recommended_review_type="ENTRY_REVIEW",
                    action_hint="TRADE",
                    now=now,
                    details={"lane_id": result.get("lane_id"), "status": result.get("status")},
                )
            )
    return events


def _position_management_events(payload: dict[str, Any], *, now: datetime) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    for item in payload.get("positions", []) or []:
        if not isinstance(item, dict):
            continue
        pair = _pair(item.get("pair"))
        if not pair:
            continue
        action = str(item.get("action") or "").upper()
        side = _direction_from_text(item.get("side"))
        thesis = str(item.get("thesis") or item.get("trade_id") or "open position thesis")
        reasons = item.get("reasons") if isinstance(item.get("reasons"), list) else []
        if action in {"TAKE_PROFIT_MARKET", "HARVEST_TP", "NARROW_TP"}:
            events.append(
                _event(
                    event_type="HARVEST_ZONE",
                    pair=pair,
                    direction=side,
                    thesis=thesis,
                    price_zone=str(item.get("recommended_take_profit") or "position-management harvest"),
                    severity="P1",
                    recommended_review_type="HARVEST_REVIEW",
                    action_hint="HARVEST",
                    thesis_state=str(item.get("thesis_state") or "ALIVE").upper(),
                    now=now,
                    details={"trade_id": item.get("trade_id"), "action": action, "reasons": reasons},
                )
            )
        if action in {"REVIEW_EXIT", "RECOMMEND_CLOSE"}:
            severity = "P0" if _contains_any(reasons, ("loss-cut:", "structural", "BROKEN")) else "P1"
            events.append(
                _event(
                    event_type="THESIS_INVALIDATION",
                    pair=pair,
                    direction=side,
                    thesis=thesis,
                    price_zone=str(item.get("invalidation") or item.get("recommended_stop_loss") or "position invalidation review"),
                    severity=severity,
                    recommended_review_type="THESIS_REVIEW",
                    action_hint="REDUCE",
                    thesis_state="INVALIDATED" if severity == "P0" else "WOUNDED",
                    now=now,
                    details={"trade_id": item.get("trade_id"), "action": action, "reasons": reasons},
                )
            )
    return events


def _thesis_evolution_events(payload: dict[str, Any], *, now: datetime) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    for item in payload.get("evolutions", []) or []:
        if not isinstance(item, dict):
            continue
        pair = _pair(item.get("pair"))
        if not pair:
            continue
        status = str(item.get("status") or item.get("verdict") or "").upper()
        if status not in {"WEAKENED", "BROKEN", "INVALIDATED", "EMERGENCY"}:
            continue
        severity = "P0" if status in {"BROKEN", "INVALIDATED", "EMERGENCY"} else "P2"
        events.append(
            _event(
                event_type="THESIS_INVALIDATION",
                pair=pair,
                direction=_direction_from_text(item.get("side")),
                thesis=str(item.get("thesis") or item.get("trade_id") or "entry thesis"),
                price_zone=str(item.get("invalidation") or item.get("reason") or status),
                severity=severity,
                recommended_review_type="THESIS_REVIEW",
                action_hint="REDUCE" if severity == "P0" else "HOLD",
                thesis_state="INVALIDATED" if severity == "P0" else "WOUNDED",
                now=now,
                details={"trade_id": item.get("trade_id"), "status": status, "reason": item.get("reason")},
            )
        )
    return events


def _forecast_persistence_events(payload: dict[str, Any], *, now: datetime) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    for item in payload.get("verdicts", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("verdict") or "").upper() != "RECOMMEND_CLOSE":
            continue
        pair = _pair(item.get("pair"))
        if not pair:
            continue
        events.append(
            _event(
                event_type="THESIS_INVALIDATION",
                pair=pair,
                direction=_direction_from_text(item.get("side")),
                thesis=str(item.get("thesis") or item.get("trade_id") or "forecast persistence thesis"),
                price_zone=str(item.get("reason") or "forecast persistence close review"),
                severity="P1",
                recommended_review_type="THESIS_REVIEW",
                action_hint="REDUCE",
                thesis_state="WOUNDED",
                now=now,
                details={"trade_id": item.get("trade_id"), "verdict": item.get("verdict")},
            )
        )
    return events


def _market_context_matrix_events(payload: dict[str, Any], *, now: datetime) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    pairs = payload.get("pairs") if isinstance(payload.get("pairs"), dict) else {}
    for pair_key, sides in pairs.items():
        pair = _pair(pair_key)
        if not pair or not isinstance(sides, dict):
            continue
        for side, row in sides.items():
            if not isinstance(row, dict):
                continue
            if _truthy(row.get("theme_confirmation")) or (
                _int(row.get("support_count"), default=0) >= 3
                and _int(row.get("reject_count"), default=0) == 0
            ):
                events.append(
                    _event(
                        event_type="THEME_CONFIRMATION",
                        pair=pair,
                        direction=_direction_from_text(side),
                        thesis=str(row.get("strongest_support") or "multi-layer theme confirmation"),
                        price_zone=str(row.get("evidence_ref") or f"matrix:{pair}:{side}"),
                        severity="P2",
                        recommended_review_type="ENTRY_REVIEW",
                        action_hint="TRADE",
                        now=now,
                        details={
                            "support_count": row.get("support_count"),
                            "reject_count": row.get("reject_count"),
                            "evidence_ref": row.get("evidence_ref"),
                        },
                    )
                )
    return events


def _margin_pressure_event(account: dict[str, Any], *, now: datetime) -> GuardianEvent | None:
    nav = _float(account.get("nav_jpy"))
    used = _float(account.get("margin_used_jpy"))
    available = _float(account.get("margin_available_jpy"))
    if nav is None or used is None or available is None or nav <= 0:
        return None
    cap_pct = RiskPolicy().max_margin_utilization_pct
    if cap_pct is None or cap_pct <= 0:
        return None
    utilization = used / nav
    cap_fraction = cap_pct / 100.0
    available_ratio = available / nav
    if utilization >= cap_fraction or available <= 0:
        severity = "P0"
    elif utilization >= cap_fraction * MARGIN_PRESSURE_WARNING_CAP_FRACTION:
        severity = "P1"
    else:
        return None
    return _event(
        event_type="MARGIN_PRESSURE",
        pair="PORTFOLIO",
        direction=None,
        thesis="portfolio margin capacity",
        price_zone=f"margin_used/nav={utilization:.3f}; available/nav={available_ratio:.3f}; cap={cap_fraction:.3f}",
        severity=severity,
        recommended_review_type="RISK_REVIEW",
        action_hint="REDUCE" if severity == "P0" else "HOLD",
        thesis_state="EMERGENCY" if severity == "P0" else "WOUNDED",
        now=now,
        details={
            "nav_jpy": nav,
            "margin_used_jpy": used,
            "margin_available_jpy": available,
            "max_margin_utilization_pct": cap_pct,
        },
    )


def _quote_major_figure_events(quotes: dict[str, Any], *, now: datetime) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    for pair_key, quote in quotes.items():
        pair = _pair(pair_key)
        if not pair or not isinstance(quote, dict):
            continue
        bid = _float(quote.get("bid"))
        ask = _float(quote.get("ask"))
        if bid is None or ask is None:
            continue
        mid = (bid + ask) / 2.0
        zone = _major_figure_zone(pair, mid)
        if zone is None:
            continue
        events.append(
            _event(
                event_type="FAILED_ACCEPTANCE",
                pair=pair,
                direction=None,
                thesis="major figure / failed acceptance watch",
                price_zone=zone,
                severity="P2",
                recommended_review_type="ENTRY_REVIEW",
                action_hint="HOLD",
                now=now,
                details={"bid": bid, "ask": ask},
            )
        )
    return events


def _contract_entries(contract: dict[str, Any]) -> list[dict[str, Any]]:
    entries = contract.get("entries") if isinstance(contract, dict) else []
    return [entry for entry in entries or [] if isinstance(entry, dict)] if isinstance(entries, list) else []


def _contract_entry_key(entry: dict[str, Any]) -> str:
    trade_id = str(entry.get("trade_id") or "").strip()
    return "|".join(
        [
            _pair(entry.get("pair")) or "UNKNOWN",
            _direction_from_text(entry.get("side")) or "UNKNOWN",
            trade_id or "NO_TRADE_ID",
            _slug(str(entry.get("thesis") or "")),
        ]
    )


def _contract_entry_legacy_key(entry: dict[str, Any]) -> str:
    return "|".join(
        [
            _pair(entry.get("pair")) or "UNKNOWN",
            _direction_from_text(entry.get("side")) or "UNKNOWN",
            _slug(str(entry.get("thesis") or "")),
        ]
    )


def _contract_entry_has_open_exposure(entry: dict[str, Any]) -> bool:
    return bool(str(entry.get("trade_id") or "").strip())


def _contract_entry_from_seed(
    *,
    pair: str,
    side: str,
    thesis: str,
    owner: str,
    thesis_state: str,
    seed_reason: str,
    preserved: dict[str, dict[str, Any]],
    now: datetime,
    position: dict[str, Any] | None = None,
    open_exposure: bool = False,
) -> dict[str, Any]:
    position_payload = position if isinstance(position, dict) else {}
    seed = {
        "pair": pair,
        "side": side,
        "thesis": thesis,
        "owner": owner,
        "thesis_state": _thesis_state(thesis_state) if _thesis_state(thesis_state) != "UNKNOWN" else "ALIVE",
        "trade_id": _position_trade_id(position_payload),
    }
    prior = preserved.get(_contract_entry_key(seed)) or preserved.get(_contract_entry_legacy_key(seed))
    deadline = now.timestamp() + DEFAULT_CONTRACT_REVIEW_DEADLINE_SECONDS
    base = {
        **seed,
        "units": _position_units(position_payload),
        "avg_entry": _position_average_entry(position_payload),
        "harvest_triggers": [],
        "add_triggers": [],
        "no_add_triggers": [],
        "wounded_triggers": [],
        "invalidation_triggers": [],
        "emergency_triggers": [],
        "next_review_reason": seed_reason,
        "next_review_deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
    }
    if open_exposure:
        base.update(_default_open_position_triggers(base, position_payload, now=now))
    if not isinstance(prior, dict):
        return {key: value for key, value in base.items() if value is not None}
    merged = dict(base)
    for field_name in CONTRACT_TRIGGER_BUCKETS:
        if isinstance(prior.get(field_name), list) and prior[field_name]:
            merged[field_name] = prior[field_name]
    for field_name in ("owner", "thesis_state", "next_review_reason", "next_review_deadline_utc"):
        if prior.get(field_name) and (field_name != "next_review_deadline_utc" or _deadline_is_future(prior.get(field_name), now)):
            merged[field_name] = prior[field_name]
    return {key: value for key, value in merged.items() if value is not None}


def _position_contract_thesis(position: dict[str, Any]) -> str:
    raw = position.get("raw") if isinstance(position.get("raw"), dict) else {}
    operator_packet = raw.get("operator_manual_position") if isinstance(raw.get("operator_manual_position"), dict) else {}
    return str(
        operator_packet.get("thesis")
        or position.get("thesis")
        or position.get("trade_id")
        or "open broker position"
    )


def _position_contract_state(position: dict[str, Any]) -> str:
    raw = position.get("raw") if isinstance(position.get("raw"), dict) else {}
    operator_packet = raw.get("operator_manual_position") if isinstance(raw.get("operator_manual_position"), dict) else {}
    state = _thesis_state(operator_packet.get("thesis_state") or position.get("thesis_state"))
    return state if state != "UNKNOWN" else "ALIVE"


def _position_trade_id(position: dict[str, Any]) -> str | None:
    for key in ("trade_id", "id", "tradeID"):
        value = position.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _position_units(position: dict[str, Any]) -> float | None:
    for key in ("units", "current_units", "currentUnits"):
        value = _float(position.get(key))
        if value is not None:
            return value
    return None


def _position_average_entry(position: dict[str, Any]) -> float | None:
    for key in ("avg_entry", "average_entry", "average_entry_price", "price", "entry"):
        value = _float(position.get(key))
        if value is not None:
            return value
    raw = position.get("raw") if isinstance(position.get("raw"), dict) else {}
    for key in ("price", "average_entry", "averagePrice"):
        value = _float(raw.get(key))
        if value is not None:
            return value
    return None


def _deadline_is_future(value: Any, now: datetime) -> bool:
    deadline = _parse_utc(value)
    return deadline is not None and deadline > now


def _default_open_position_triggers(
    entry: dict[str, Any],
    position: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, list[dict[str, Any]]]:
    pair = _pair(entry.get("pair")) or "UNKNOWN"
    side = _direction_from_text(entry.get("side")) or "UNKNOWN"
    owner = _contract_owner(entry.get("owner"))
    thesis_state = _thesis_state(entry.get("thesis_state"))
    avg_entry = _position_average_entry(position)
    trade_id = _position_trade_id(position)
    base_ref = {"trade_id": trade_id, "pair": pair, "side": side}
    if _is_usd_jpy_162_manual_fade(pair, side, owner, position):
        return _usd_jpy_manual_fade_triggers(entry, position, now=now)
    harvest_detail = "profit-side TP/harvest review when current quote reaches the broker TP, declared harvest zone, or positive UPL is outside spread/noise"
    invalidation_detail = "accepted market evidence that the position thesis is broken; red P/L alone is not invalidation"
    no_add_detail = "no add while thesis_state is not ALIVE, margin is under pressure, or overlap lacks explicit operator authorization"
    return {
        "harvest_triggers": [
            {
                **base_ref,
                "trigger_id": "open_exposure_profit_harvest_review",
                "status": "PENDING",
                "kind": "profit_harvest_review",
                "evidence_required": harvest_detail,
                "avg_entry": avg_entry,
                "action_hint": "HARVEST",
            }
        ],
        "no_add_triggers": [
            {
                **base_ref,
                "trigger_id": "open_exposure_no_add_guard",
                "status": "PENDING",
                "kind": "no_add_guard",
                "evidence_required": no_add_detail,
                "thesis_state": thesis_state,
                "action_hint": "HOLD",
            }
        ],
        "wounded_triggers": [
            {
                **base_ref,
                "trigger_id": "open_exposure_wounded_review",
                "status": "PENDING",
                "kind": "thesis_wounded_review",
                "evidence_required": "price action materially damages but does not invalidate the thesis",
                "action_hint": "HOLD",
            }
        ],
        "invalidation_triggers": [
            {
                **base_ref,
                "trigger_id": "open_exposure_invalidation_review",
                "status": "PENDING",
                "kind": "thesis_invalidation_review",
                "evidence_required": invalidation_detail,
                "action_hint": "REDUCE",
            }
        ],
        "emergency_triggers": [
            {
                **base_ref,
                "trigger_id": "open_exposure_margin_or_nav_emergency",
                "status": "PENDING",
                "kind": "margin_or_nav_emergency",
                "evidence_required": "margin pressure, NAV shock, missing broker truth, or gateway-outside exposure needs immediate trader review",
                "action_hint": "REDUCE",
            }
        ],
    }


def _is_usd_jpy_162_manual_fade(pair: str, side: str, owner: str, position: dict[str, Any]) -> bool:
    if pair != "USD_JPY" or side != "SHORT" or owner not in {"OPERATOR_MANUAL", "UNKNOWN"}:
        return False
    thesis = _position_contract_thesis(position).lower()
    return "162" in thesis or "manual" in thesis or "operator" in thesis or owner == "OPERATOR_MANUAL"


def _usd_jpy_manual_fade_triggers(
    entry: dict[str, Any],
    position: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, list[dict[str, Any]]]:
    # Operator-specified manual USD_JPY 162 fade contract; these are review
    # triggers only, and omit machine predicates for confirmation-only evidence.
    trade_id = _position_trade_id(position)
    avg_entry = _position_average_entry(position)
    units = _position_units(position)
    ref = {"trade_id": trade_id, "pair": "USD_JPY", "side": "SHORT", "avg_entry": avg_entry, "units": units}
    return {
        "harvest_triggers": [
            {
                **ref,
                "trigger_id": "usd_jpy_162_manual_fade_profit_zone",
                "status": "PENDING",
                "kind": "manual_tp_profit_zone",
                "evidence_required": "USD_JPY trades below the manual fade average entry enough to show positive UPL outside current spread/noise; TP-only profit assistance is allowed",
                "action_hint": "HARVEST",
            }
        ],
        "no_add_triggers": [
            {
                **ref,
                "trigger_id": "usd_jpy_162_manual_fade_no_add_guard",
                "status": "PENDING",
                "kind": "manual_overlap_no_add",
                "evidence_required": "no fresh bot USD_JPY or JPY-cross add while margin is too high or thesis_state is not ALIVE unless explicit operator manual-overlap authorization exists",
                "action_hint": "HOLD",
            }
        ],
        "wounded_triggers": [
            {
                **ref,
                "trigger_id": "usd_jpy_162_manual_fade_wounded_m5_hold_above_figure",
                "status": "PENDING",
                "kind": "major_figure_wounded",
                "zone": "162.00 figure / upper battle zone",
                "confirmation_required": "M5 holds above the 162.00 figure/upper battle zone rather than a wick-only stop run",
                "action_hint": "HOLD",
            }
        ],
        "invalidation_triggers": [
            {
                **ref,
                "trigger_id": "usd_jpy_162_manual_fade_invalidated_accepted_break",
                "status": "PENDING",
                "kind": "major_figure_invalidation",
                "zone": "accepted break above 162.00 figure",
                "confirmation_required": "accepted break above the 162.00 figure with cross-JPY confirmation; red P/L or a wick/touch is not enough",
                "action_hint": "REDUCE",
            }
        ],
        "emergency_triggers": [
            {
                **ref,
                "trigger_id": "usd_jpy_162_manual_fade_margin_nav_emergency",
                "status": "PENDING",
                "kind": "margin_nav_emergency",
                "evidence_required": "margin/NAV pressure crosses the gateway risk cap or broker truth becomes unavailable; trader review only, no direct close",
                "action_hint": "REDUCE",
            }
        ],
    }


def _contract_owner(value: Any) -> str:
    text = str(value or "UNKNOWN").strip().upper()
    return text if text in CONTRACT_OWNERS else "UNKNOWN"


def _contract_owner_from_snapshot_owner(value: Any) -> str:
    owner = _owner(value)
    if owner == Owner.TRADER.value:
        return "SYSTEM"
    if owner in {Owner.MANUAL.value, Owner.OPERATOR_MANUAL.value, Owner.UNKNOWN.value}:
        return "OPERATOR_MANUAL" if owner in {Owner.MANUAL.value, Owner.OPERATOR_MANUAL.value} else "UNKNOWN"
    return "UNKNOWN"


def _snapshot_has_exposure(snapshot: dict[str, Any]) -> bool:
    return any(isinstance(item, dict) for item in snapshot.get("positions", []) or []) or any(
        isinstance(item, dict) for item in snapshot.get("orders", []) or []
    )


def _contract_stale_reason(validation: dict[str, Any]) -> str:
    if validation.get("status") != "VALID":
        codes = ",".join(str(issue.get("code")) for issue in validation.get("issues", []) or [])
        return f"contract invalid: {codes or 'UNKNOWN'}"
    return f"contract stale age_seconds={validation.get('age_seconds')} max={validation.get('max_age_seconds')}"


def _contract_trigger_fired(
    trigger: Any,
    *,
    entry: dict[str, Any],
    snapshot: dict[str, Any],
) -> tuple[bool, str]:
    if _trigger_explicitly_fired(trigger):
        return True, _trigger_evidence(trigger)
    if not isinstance(trigger, dict):
        return False, ""
    condition = trigger.get("condition") if isinstance(trigger.get("condition"), dict) else trigger
    metric = str(condition.get("metric") or condition.get("field") or "").strip()
    operator = str(condition.get("operator") or condition.get("op") or "").strip()
    expected = _float(condition.get("value") if "value" in condition else condition.get("threshold"))
    if not metric or not operator or expected is None:
        return False, ""
    actual = _contract_metric_value(metric, entry=entry, snapshot=snapshot)
    if actual is None:
        return False, ""
    if not _compare_numeric(actual, operator, expected):
        return False, ""
    return True, f"{metric} {operator} {expected} fired with actual={actual}"


def _trigger_explicitly_fired(trigger: Any) -> bool:
    if isinstance(trigger, str):
        text = trigger.strip().upper()
        return text.startswith("FIRED:") or text.startswith("TRIGGERED:") or "[FIRED]" in text or "[TRIGGERED]" in text
    if not isinstance(trigger, dict):
        return False
    if any(_truthy(trigger.get(key)) for key in ("fired", "triggered", "active", "hit")):
        return True
    return str(trigger.get("status") or "").strip().upper() in {"FIRED", "TRIGGERED", "HIT", "ACTIVE"}


def _trigger_field(trigger: Any, field_name: str) -> Any:
    return trigger.get(field_name) if isinstance(trigger, dict) else None


def _trigger_evidence(trigger: Any) -> str:
    if isinstance(trigger, str):
        return trigger
    if not isinstance(trigger, dict):
        return "contract trigger fired"
    return str(
        trigger.get("evidence")
        or trigger.get("reason")
        or trigger.get("label")
        or trigger.get("description")
        or trigger.get("trigger_id")
        or "contract trigger fired"
    )


def _contract_metric_value(metric: str, *, entry: dict[str, Any], snapshot: dict[str, Any]) -> float | None:
    name = metric.strip().lower().replace("quote.", "").replace("account.", "").replace("position.", "")
    pair = _pair(entry.get("pair"))
    side = _direction_from_text(entry.get("side"))
    quotes = snapshot.get("quotes") if isinstance(snapshot.get("quotes"), dict) else {}
    quote = quotes.get(pair) if pair else None
    if isinstance(quote, dict):
        bid = _float(quote.get("bid"))
        ask = _float(quote.get("ask"))
        if name in {"bid", "ask"}:
            return bid if name == "bid" else ask
        if name in {"mid", "price"} and bid is not None and ask is not None:
            return (bid + ask) / 2.0
        if name == "spread_pips" and bid is not None and ask is not None and pair:
            return max(0.0, (ask - bid) * instrument_pip_factor(pair))
    account = snapshot.get("account") if isinstance(snapshot.get("account"), dict) else {}
    if name in {"nav_jpy", "margin_used_jpy", "margin_available_jpy"}:
        return _float(account.get(name))
    matching_positions = []
    for position in snapshot.get("positions", []) or []:
        if not isinstance(position, dict):
            continue
        if pair and _pair(position.get("pair")) != pair:
            continue
        if side and _direction_from_position(position) != side:
            continue
        matching_positions.append(position)
    if name == "unrealized_pl_jpy":
        values = [_float(position.get("unrealized_pl_jpy")) for position in matching_positions]
        return sum(value for value in values if value is not None) if values else None
    if name == "units":
        values = [_float(position.get("units")) for position in matching_positions]
        return sum(value for value in values if value is not None) if values else None
    return None


def _compare_numeric(actual: float, operator: str, expected: float) -> bool:
    op = operator.strip()
    if op in {">", "gt"}:
        return actual > expected
    if op in {">=", "gte"}:
        return actual >= expected
    if op in {"<", "lt"}:
        return actual < expected
    if op in {"<=", "lte"}:
        return actual <= expected
    if op in {"==", "=", "eq"}:
        return actual == expected
    return False


def _text_acceptance_events(pair: str, text: str, *, mid: float | None, now: datetime) -> list[GuardianEvent]:
    lower = text.lower()
    events: list[GuardianEvent] = []
    if "failed acceptance" in lower or "rejected acceptance" in lower or "stop-run" in lower:
        events.append(
            _event(
                event_type="FAILED_ACCEPTANCE",
                pair=pair,
                direction=_direction_from_acceptance_text(lower),
                thesis="major figure / failed acceptance",
                price_zone=_major_figure_zone(pair, mid) or "failed acceptance zone",
                severity="P1",
                recommended_review_type="ENTRY_REVIEW",
                action_hint="TRADE",
                now=now,
                details={"text_excerpt": text[:240]},
            )
        )
    if "accepted trade above" in lower or "accepted trade below" in lower or "acceptance break" in lower:
        events.append(
            _event(
                event_type="ACCEPTANCE_BREAK",
                pair=pair,
                direction=_direction_from_acceptance_text(lower),
                thesis="accepted break from prior zone",
                price_zone=_major_figure_zone(pair, mid) or "accepted break zone",
                severity="P1",
                recommended_review_type="THESIS_REVIEW",
                action_hint="HOLD",
                thesis_state="WOUNDED",
                now=now,
                details={"text_excerpt": text[:240]},
            )
        )
    return events


def _range_rail_event(pair: str, chart: dict[str, Any], confluence: dict[str, Any], *, now: datetime) -> GuardianEvent | None:
    percentile = _float(confluence.get("price_percentile_24h"))
    regime = str(chart.get("dominant_regime") or confluence.get("dominant_regime") or "").upper()
    if percentile is None or "RANGE" not in regime:
        return None
    if percentile <= RANGE_RAIL_LOW_PERCENTILE:
        direction = "LONG"
        zone = f"lower 24h range rail percentile={percentile:.3f}"
    elif percentile >= RANGE_RAIL_HIGH_PERCENTILE:
        direction = "SHORT"
        zone = f"upper 24h range rail percentile={percentile:.3f}"
    else:
        return None
    return _event(
        event_type="RANGE_RAIL_TOUCH",
        pair=pair,
        direction=direction,
        thesis="range rail rotation review",
        price_zone=zone,
        severity="P2",
        recommended_review_type="ENTRY_REVIEW",
        action_hint="TRADE",
        now=now,
        details={"dominant_regime": regime, "price_percentile_24h": percentile},
    )


def _session_expansion_event(pair: str, chart: dict[str, Any], confluence: dict[str, Any], *, now: datetime) -> GuardianEvent | None:
    session = chart.get("session") if isinstance(chart.get("session"), dict) else {}
    tag = str(session.get("current_tag") or "").upper()
    sigma = _float(confluence.get("range_24h_sigma_multiple"))
    if not tag or not any(item in tag for item in ("LONDON", "NY")):
        return None
    if sigma is None or sigma < 2.0:
        return None
    balance = str(confluence.get("score_balance") or "").upper()
    direction = "LONG" if "LONG" in balance else "SHORT" if "SHORT" in balance else None
    return _event(
        event_type="SESSION_EXPANSION",
        pair=pair,
        direction=direction,
        thesis="session expansion timing event",
        price_zone=f"{tag} expansion sigma={sigma:.2f}",
        severity="P2",
        recommended_review_type="ENTRY_REVIEW",
        action_hint="TRADE",
        now=now,
        details={"session_tag": tag, "range_24h_sigma_multiple": sigma},
    )


def _squeeze_release_event(pair: str, chart: dict[str, Any], *, now: datetime) -> GuardianEvent | None:
    for view in chart.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        family = view.get("family_scores") if isinstance(view.get("family_scores"), dict) else {}
        breakout = family.get("breakout_components") if isinstance(family.get("breakout_components"), dict) else {}
        if _truthy(breakout.get("bb_squeeze_release")) or _truthy(breakout.get("squeeze_release")):
            return _event(
                event_type="SQUEEZE_RELEASE",
                pair=pair,
                direction=None,
                thesis="squeeze release timing event",
                price_zone=str(view.get("granularity") or "squeeze release"),
                severity="P1",
                recommended_review_type="ENTRY_REVIEW",
                action_hint="TRADE",
                now=now,
                details={"granularity": view.get("granularity"), "breakout_components": breakout},
            )
    text = str(chart.get("chart_story") or "").lower()
    if "squeeze release" not in text:
        return None
    return _event(
        event_type="SQUEEZE_RELEASE",
        pair=pair,
        direction=None,
        thesis="squeeze release timing event",
        price_zone="chart_story squeeze release",
        severity="P1",
        recommended_review_type="ENTRY_REVIEW",
        action_hint="TRADE",
        now=now,
        details={"text_excerpt": text[:240]},
    )


def _explicit_event(payload: dict[str, Any], *, pair: str, now: datetime) -> GuardianEvent | None:
    event_type = str(payload.get("event_type") or payload.get("type") or "").strip().upper()
    if event_type not in EVENT_TYPES:
        return None
    return _event(
        event_type=event_type,
        pair=pair,
        direction=_direction_from_text(payload.get("direction")),
        thesis=str(payload.get("thesis") or payload.get("reason") or event_type),
        price_zone=str(payload.get("price_zone") or payload.get("zone") or "explicit guardian event"),
        severity=_severity(payload.get("severity")),
        recommended_review_type=str(payload.get("recommended_review_type") or _default_review_type(event_type)),
        action_hint=str(payload.get("action_hint") or _default_action_hint(event_type)).upper(),
        thesis_state=str(payload.get("thesis_state") or "UNKNOWN").upper(),
        now=now,
        details={key: value for key, value in payload.items() if key not in {"event_type", "type"}},
    )


def _event(
    *,
    event_type: str,
    pair: str,
    direction: str | None,
    thesis: str,
    price_zone: str,
    severity: str,
    recommended_review_type: str,
    action_hint: str,
    now: datetime,
    thesis_state: str = "UNKNOWN",
    details: dict[str, Any] | None = None,
) -> GuardianEvent:
    event_type = event_type.upper()
    pair = _pair(pair) or "UNKNOWN"
    direction = _direction_from_text(direction)
    severity = _severity(severity)
    thesis_state = _thesis_state(thesis_state)
    action_hint = str(action_hint or _default_action_hint(event_type)).upper()
    recommended_review_type = str(recommended_review_type or _default_review_type(event_type)).upper()
    dedupe_key = "|".join(
        [
            pair,
            _slug(thesis or event_type),
            event_type,
            action_hint or recommended_review_type,
        ]
    )
    event_id = hashlib.sha256(
        f"{dedupe_key}|{direction or ''}|{price_zone}|{severity}".encode("utf-8")
    ).hexdigest()[:16]
    return GuardianEvent(
        event_id=event_id,
        event_type=event_type,
        pair=pair,
        direction=direction,
        thesis=thesis or event_type,
        price_zone=price_zone or "UNKNOWN",
        severity=severity,
        recommended_review_type=recommended_review_type,
        dedupe_key=dedupe_key,
        action_hint=action_hint,
        thesis_state=thesis_state,
        detected_at_utc=now.isoformat(),
        details=details or {},
    )


def _dedupe_events(events: list[GuardianEvent]) -> list[GuardianEvent]:
    by_key: dict[str, GuardianEvent] = {}
    for event in events:
        prior = by_key.get(event.dedupe_key)
        if prior is None or SEVERITY_RANK[event.severity] > SEVERITY_RANK[prior.severity]:
            by_key[event.dedupe_key] = event
    return sorted(by_key.values(), key=lambda item: (-SEVERITY_RANK[item.severity], item.pair, item.event_type))


def _wake_reasons_for_event(event: GuardianEvent, prior: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    if prior is None:
        reasons.append("NEW_EVENT")
    else:
        previous_severity = _severity(prior.get("severity"))
        if SEVERITY_RANK[event.severity] > SEVERITY_RANK[previous_severity]:
            reasons.append("SEVERITY_INCREASE")
        previous_state = _thesis_state(prior.get("thesis_state"))
        if (
            previous_state == "ALIVE"
            and event.thesis_state in {"WOUNDED", "INVALIDATED", "EMERGENCY"}
        ):
            reasons.append(f"THESIS_{previous_state}_TO_{event.thesis_state}")
    if event.event_type == "HARVEST_ZONE" and not (prior or {}).get("in_harvest_zone"):
        reasons.append("PRICE_ENTERED_HARVEST_ZONE")
    if event.event_type == "UNKNOWN_ORDER" and prior is None:
        reasons.append("UNKNOWN_GATEWAY_OUTSIDE_ORDER")
    if event.event_type == "MARGIN_PRESSURE" and (prior is None or not prior.get("margin_pressure")):
        reasons.append("MARGIN_RISK_THRESHOLD_CROSSED")
    return list(dict.fromkeys(reasons))


def _event_throttle_seconds(event: GuardianEvent) -> int:
    if event.action_hint in {"TRADE", "ADD"} or event.recommended_review_type in {"ENTRY_REVIEW", "ADD_REVIEW"}:
        return int(os.environ.get("QR_GUARDIAN_FRESH_ACTION_THROTTLE_SECONDS", DEFAULT_FRESH_ACTION_THROTTLE_SECONDS))
    return int(os.environ.get("QR_GUARDIAN_EVENT_THROTTLE_SECONDS", DEFAULT_THROTTLE_SECONDS))


def _event_bypasses_throttle(event: GuardianEvent, reasons: list[str]) -> bool:
    return (
        event.event_type == "HARVEST_ZONE"
        or event.severity == "P0"
        or "SEVERITY_INCREASE" in reasons
        or any(reason.startswith("THESIS_ALIVE_TO_") for reason in reasons)
    )


def _guardian_trade_add_action_issues(
    receipt: dict[str, Any],
    *,
    event: GuardianEvent | None,
    intent_metadata: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    intent_metadata = intent_metadata or {}
    issues: list[dict[str, str]] = []
    action = str(receipt.get("action") or "").strip().upper()
    if action not in {"TRADE", "ADD"}:
        return issues
    if receipt.get("new_information") is not True:
        issues.append(_issue("GUARDIAN_ACTION_REQUIRES_NEW_INFORMATION", "TRADE/ADD requires new_information=true"))
    if _truthy(receipt.get("same_pair_thesis_action_recently_sent")) or _truthy(receipt.get("duplicate_recent_action")):
        issues.append(_issue("GUARDIAN_ACTION_RECENT_DUPLICATE", "same pair/thesis/action was recently sent"))
    reason_blob = " ".join(
        str(receipt.get(key) or "") for key in ("reason", "review_reason", "operator_summary")
    ).lower()
    if _truthy(receipt.get("schedule_only")) or "scheduled hour" in reason_blob or "hourly" in reason_blob:
        issues.append(_issue("GUARDIAN_ACTION_SCHEDULE_ONLY", "scheduled hour alone cannot authorize TRADE/ADD"))
    if _truthy(receipt.get("bc_churn_pace_fix")) or "b/c churn" in reason_blob or "bc churn" in reason_blob:
        issues.append(_issue("GUARDIAN_ACTION_BC_CHURN", "B/C churn cannot be used to fix pace"))
    add_type = str(intent_metadata.get("same_pair_add_type") or receipt.get("same_pair_add_type") or "").upper()
    if add_type == "PYRAMID_WITH_MOVE" and not (
        _truthy(intent_metadata.get("independent_fresh_edge")) or _truthy(receipt.get("independent_fresh_edge"))
    ):
        issues.append(
            _issue(
                "GUARDIAN_ACTION_PYRAMID_NEEDS_FRESH_EDGE",
                "with-move pyramid requires an independent fresh edge",
            )
        )
    thesis_state = _thesis_state(receipt.get("thesis_state") or (event.thesis_state if event else None))
    if thesis_state in {"WOUNDED", "INVALIDATED"}:
        issues.append(
            _issue(
                "GUARDIAN_ACTION_THESIS_STATE_BLOCKS_ENTRY",
                f"TRADE/ADD rejected while thesis_state={thesis_state}",
            )
        )
    if receipt.get("gateway_required") is not True:
        issues.append(_issue("GUARDIAN_ACTION_GATEWAY_REQUIRED", "TRADE/ADD requires gateway_required=true"))
    return issues


def _metadata_marks_guardian_wake(metadata: dict[str, Any]) -> bool:
    return any(
        str(metadata.get(key) or "").strip()
        for key in ("guardian_event_id", "guardian_event_dedupe_key", "guardian_wake_id")
    ) or _truthy(metadata.get("guardian_event_wake"))


def _write_event_report(path: Path, events_payload: dict[str, Any], escalation: dict[str, Any]) -> None:
    lines = [
        "# Guardian Event Report",
        "",
        f"- Generated at UTC: `{events_payload.get('generated_at_utc')}`",
        f"- Events detected: `{len(events_payload.get('events') or [])}`",
        f"- Wake GPT-5.5: `{bool(escalation.get('wake_gpt'))}`",
        f"- Wake reasons: `{', '.join(escalation.get('wake_reason_codes') or []) or 'none'}`",
        "",
        "## Execution Boundary",
        "",
        "- Guardian is read-only and never trades.",
        "- GPT wake output is a receipt only and never calls OANDA directly.",
        "- Live sends/cancels/closes still require the existing gateway path.",
        "",
        "## Events",
        "",
    ]
    for event in events_payload.get("events", []) or []:
        lines.extend(
            [
                f"- `{event.get('severity')}` `{event.get('event_type')}` `{event.get('pair')}` "
                f"`{event.get('direction') or 'N/A'}`",
                f"  - thesis: {event.get('thesis')}",
                f"  - price zone: {event.get('price_zone')}",
                f"  - review: `{event.get('recommended_review_type')}` dedupe: `{event.get('dedupe_key')}`",
            ]
        )
    if not events_payload.get("events"):
        lines.append("- none")
    lines.extend(["", "## Escalation", ""])
    for event in escalation.get("events_to_review", []) or []:
        lines.append(
            f"- wake `{event.get('event_type')}` `{event.get('pair')}` because "
            f"`{', '.join(event.get('wake_reason_codes') or [])}`"
        )
    if not escalation.get("events_to_review"):
        lines.append("- no wake; no meaningful state change")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_action_review_report(path: Path, review: dict[str, Any]) -> None:
    receipt = review.get("receipt") if isinstance(review.get("receipt"), dict) else {}
    lines = [
        "# Guardian Action Review",
        "",
        f"- Generated at UTC: `{review.get('generated_at_utc')}`",
        f"- Status: `{review.get('status')}`",
        f"- Action: `{receipt.get('action') or 'none'}`",
        f"- Event id: `{receipt.get('event_id') or 'none'}`",
        f"- Gateway required: `{review.get('gateway_required')}`",
        "",
        "## Issues",
        "",
    ]
    for issue in review.get("issues", []) or []:
        lines.append(f"- `{issue.get('severity')}` `{issue.get('code')}` {issue.get('message')}")
    if not review.get("issues"):
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- The guardian action receipt does not execute broker writes.",
            "- TRADE/ADD must still pass `gpt-trader-decision`/`LiveOrderGateway` risk and broker-truth checks.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _load_json(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _chart_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    charts = payload.get("charts")
    if isinstance(charts, list):
        return [item for item in charts if isinstance(item, dict)]
    if isinstance(charts, dict):
        rows = []
        for pair, item in charts.items():
            if isinstance(item, dict):
                row = dict(item)
                row.setdefault("pair", pair)
                rows.append(row)
        return rows
    return []


def _chart_mid(chart: dict[str, Any], snapshot: dict[str, Any]) -> float | None:
    for key in ("mid", "last", "last_close", "close"):
        value = _float(chart.get(key))
        if value is not None:
            return value
    pair = _pair(chart.get("pair"))
    quotes = snapshot.get("quotes") if isinstance(snapshot.get("quotes"), dict) else {}
    quote = quotes.get(pair) if pair else None
    if isinstance(quote, dict):
        bid = _float(quote.get("bid"))
        ask = _float(quote.get("ask"))
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
    return None


def _major_figure_zone(pair: str, mid: float | None) -> str | None:
    if mid is None:
        return None
    pip_factor = instrument_pip_factor(pair)
    major_step = 100.0 / pip_factor
    nearest = round(mid / major_step) * major_step
    distance_pips = abs(mid - nearest) * pip_factor
    proximity_pips = NORMAL_SPREAD_PIPS.get(pair, 1.0) * 5.0
    if distance_pips <= proximity_pips:
        decimals = 3 if pair.endswith("_JPY") else 5
        return f"major figure {nearest:.{decimals}f} distance_pips={distance_pips:.2f}"
    return None


def _pending_order_is_stale(order: dict[str, Any], *, now: datetime) -> bool:
    state = str(order.get("state") or "").upper()
    if state and state not in {"PENDING", "LIVE", "OPEN"}:
        return False
    raw = order.get("raw") if isinstance(order.get("raw"), dict) else {}
    created = _parse_utc(raw.get("createTime") or raw.get("time") or order.get("created_at_utc"))
    if created is None:
        return False
    max_age = int(os.environ.get("QR_GUARDIAN_STALE_PENDING_SECONDS", DEFAULT_STALE_PENDING_SECONDS))
    return (now - created).total_seconds() >= max_age


def _pending_order_thesis(order: dict[str, Any]) -> str:
    raw = order.get("raw") if isinstance(order.get("raw"), dict) else {}
    extensions = raw.get("clientExtensions") if isinstance(raw.get("clientExtensions"), dict) else {}
    comment = str(extensions.get("comment") or "")
    return comment or f"pending order {order.get('order_id')}"


def _intent_is_harvest(metadata: dict[str, Any], intent: dict[str, Any], *, status: str) -> bool:
    del intent
    explicit_active = any(
        _truthy(metadata.get(key))
        for key in (
            "guardian_harvest_zone",
            "harvest_zone_active",
            "price_in_harvest_zone",
            "tp_progress_harvest_zone",
            "bankable_now",
            "profit_capture_bankable_now",
        )
    )
    if explicit_active:
        return True
    if status == "LIVE_READY" and str(metadata.get("opportunity_mode") or "").upper() == "HARVEST":
        return True
    return status == "LIVE_READY" and _truthy(metadata.get("self_improvement_p0_repair_live_ready"))


def _default_review_type(event_type: str) -> str:
    return {
        "HARVEST_ZONE": "HARVEST_REVIEW",
        "THESIS_INVALIDATION": "THESIS_REVIEW",
        "MARGIN_PRESSURE": "RISK_REVIEW",
        "UNKNOWN_ORDER": "EMERGENCY_RISK_REVIEW",
        "STALE_PENDING": "PENDING_CANCEL_REVIEW",
        "BROKER_SNAPSHOT_STALE": "RISK_REVIEW",
        "SPREAD_ANOMALY": "RISK_REVIEW",
        "UNEXPECTED_PROTECTION_MISSING": "PROTECTION_REVIEW",
        "CONTRACT_HARVEST_TRIGGER": "HARVEST_REVIEW",
        "CONTRACT_ADD_TRIGGER": "ADD_REVIEW",
        "CONTRACT_NO_ADD_TRIGGER": "ADD_REVIEW",
        "CONTRACT_WOUNDED_TRIGGER": "THESIS_REVIEW",
        "CONTRACT_INVALIDATION_TRIGGER": "THESIS_REVIEW",
        "CONTRACT_EMERGENCY_TRIGGER": "EMERGENCY_RISK_REVIEW",
        "CONTRACT_STALE": "EMERGENCY_RISK_REVIEW",
        "WAKE_PARSE_FAILURE": "WAKE_REPAIR_REVIEW",
    }.get(event_type, "ENTRY_REVIEW")


def _default_action_hint(event_type: str) -> str:
    return {
        "HARVEST_ZONE": "HARVEST",
        "THESIS_INVALIDATION": "REDUCE",
        "MARGIN_PRESSURE": "REDUCE",
        "UNKNOWN_ORDER": "REDUCE",
        "STALE_PENDING": "CANCEL_PENDING",
        "BROKER_SNAPSHOT_STALE": "HOLD",
        "SPREAD_ANOMALY": "HOLD",
        "UNEXPECTED_PROTECTION_MISSING": "HOLD",
        "CONTRACT_HARVEST_TRIGGER": "HARVEST",
        "CONTRACT_ADD_TRIGGER": "ADD",
        "CONTRACT_NO_ADD_TRIGGER": "HOLD",
        "CONTRACT_WOUNDED_TRIGGER": "HOLD",
        "CONTRACT_INVALIDATION_TRIGGER": "REDUCE",
        "CONTRACT_EMERGENCY_TRIGGER": "REDUCE",
        "CONTRACT_STALE": "HOLD",
        "WAKE_PARSE_FAILURE": "HOLD",
    }.get(event_type, "TRADE")


def _issue(code: str, message: str, severity: str = "BLOCK") -> dict[str, str]:
    return {"code": code, "message": message, "severity": severity}


def _severity(value: Any) -> str:
    text = str(value or "P2").strip().upper()
    return text if text in SEVERITY_RANK else "P2"


def _thesis_state(value: Any) -> str:
    text = str(value or "UNKNOWN").strip().upper()
    return text if text in THESIS_STATE_RANK else "UNKNOWN"


def _pair(value: Any) -> str | None:
    text = str(value or "").strip().upper().replace("/", "_").replace("-", "_")
    return text if text else None


def _owner(value: Any) -> str:
    text = str(value or Owner.UNKNOWN.value).strip().lower()
    return text if text in {item.value for item in Owner} else Owner.UNKNOWN.value


def _direction_from_position(position: dict[str, Any]) -> str | None:
    side = _direction_from_text(position.get("side"))
    if side:
        return side
    return _direction_from_units(position.get("units"))


def _direction_from_units(value: Any) -> str | None:
    units = _float(value)
    if units is None or units == 0:
        return None
    return "LONG" if units > 0 else "SHORT"


def _direction_from_text(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if text in {"LONG", "BUY", "UP"}:
        return "LONG"
    if text in {"SHORT", "SELL", "DOWN"}:
        return "SHORT"
    return None


def _direction_from_acceptance_text(text: str) -> str | None:
    if "above" in text or "up" in text:
        return "LONG"
    if "below" in text or "down" in text:
        return "SHORT"
    return None


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Z0-9]+", "_", str(value or "").upper()).strip("_")
    return text[:96] or "UNKNOWN"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _contains_any(values: list[Any], needles: tuple[str, ...]) -> bool:
    text = " ".join(str(value) for value in values).upper()
    return any(needle.upper() in text for needle in needles)


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
