from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from quant_rabbit.guardian_events import GuardianEvent, review_guardian_action_receipt
from quant_rabbit.models import (
    AccountSummary,
    BrokerOrder,
    BrokerPosition,
    BrokerSnapshot,
    MarketContext,
    OrderIntent,
    OrderType,
    Owner,
    Quote,
    Side,
    TradeMethod,
)
from quant_rabbit.paths import (
    ROOT,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_EXECUTION_LEDGER_REPORT,
    DEFAULT_GUARDIAN_ACTION_RECEIPT,
    DEFAULT_GUARDIAN_ACTION_REVIEW,
    DEFAULT_GUARDIAN_ESCALATION,
    DEFAULT_GUARDIAN_EVENT_STATE,
    DEFAULT_GUARDIAN_EVENTS,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_LIVE_ORDER_STAGE_REPORT,
    DEFAULT_ORDER_INTENTS,
)
from quant_rabbit.risk import RiskEngine


DEFAULT_ACTION_CYCLE_RESULT = ROOT / "data" / "guardian_action_cycle_result.json"
DEFAULT_ACTION_CYCLE_REPORT = ROOT / "docs" / "guardian_action_cycle_report.md"
DEFAULT_ACTION_CYCLE_LOG = ROOT / "logs" / "guardian_action_cycle.log"
DEFAULT_LIVE_ROOT = Path("/Users/tossaki/App/QuantRabbit-live")
DEFAULT_SNAPSHOT_MAX_AGE_SECONDS = 5 * 60
DEFAULT_EVENT_STATE_MAX_AGE_SECONDS = 5 * 60

GUARDIAN_ACTIONS = {"TRADE", "ADD", "HOLD", "HARVEST", "REDUCE", "CANCEL_PENDING", "NO_ACTION"}
ENTRY_ACTIONS = {"TRADE", "ADD"}
POSITION_ACTIONS = {"HARVEST", "REDUCE", "CANCEL_PENDING"}
ALLOWED_BY_THESIS_STATE = {
    "ALIVE": {"TRADE", "ADD", "HOLD", "HARVEST", "REDUCE", "CANCEL_PENDING", "NO_ACTION"},
    "WOUNDED": {"HOLD", "HARVEST", "REDUCE"},
    "INVALIDATED": {"REDUCE", "CANCEL_PENDING", "NO_ACTION"},
    "EMERGENCY": {"REDUCE", "CANCEL_PENDING", "NO_ACTION"},
}

GatewayRunner = Callable[[str, bool], dict[str, Any]]
CommandRunner = Callable[..., Any]


@dataclass(frozen=True)
class GuardianActionCyclePaths:
    root: Path = ROOT
    action_receipt: Path = DEFAULT_GUARDIAN_ACTION_RECEIPT
    escalation: Path = DEFAULT_GUARDIAN_ESCALATION
    events: Path = DEFAULT_GUARDIAN_EVENTS
    event_state: Path = DEFAULT_GUARDIAN_EVENT_STATE
    broker_snapshot: Path = DEFAULT_BROKER_SNAPSHOT
    daily_target_state: Path = DEFAULT_DAILY_TARGET_STATE
    order_intents: Path = DEFAULT_ORDER_INTENTS
    gpt_decision: Path = DEFAULT_GPT_TRADER_DECISION
    action_review: Path = DEFAULT_GUARDIAN_ACTION_REVIEW
    result: Path = DEFAULT_ACTION_CYCLE_RESULT
    report: Path = DEFAULT_ACTION_CYCLE_REPORT
    log: Path = DEFAULT_ACTION_CYCLE_LOG
    live_root: Path = DEFAULT_LIVE_ROOT

    @property
    def live_lock(self) -> Path:
        return self.live_root / ".quant_rabbit_live.lock"

    @classmethod
    def from_root(cls, root: Path, *, live_root: Path | None = None) -> "GuardianActionCyclePaths":
        return cls(
            root=root,
            action_receipt=root / "data" / "guardian_action_receipt.json",
            escalation=root / "data" / "guardian_escalation.json",
            events=root / "data" / "guardian_events.json",
            event_state=root / "data" / "guardian_event_state.json",
            broker_snapshot=root / "data" / "broker_snapshot.json",
            daily_target_state=root / "data" / "daily_target_state.json",
            order_intents=root / "data" / "order_intents.json",
            gpt_decision=root / "data" / "gpt_trader_decision.json",
            action_review=root / "docs" / "guardian_action_review.md",
            result=root / "data" / "guardian_action_cycle_result.json",
            report=root / "docs" / "guardian_action_cycle_report.md",
            log=root / "logs" / "guardian_action_cycle.log",
            live_root=live_root or Path(os.environ.get("QR_GUARDIAN_ACTION_LIVE_ROOT", str(DEFAULT_LIVE_ROOT))),
        )


def run_guardian_action_cycle(
    *,
    paths: GuardianActionCyclePaths | None = None,
    now: datetime | None = None,
    env: dict[str, str] | None = None,
    command_runner: CommandRunner = subprocess.run,
    gateway_runner: GatewayRunner | None = None,
) -> dict[str, Any]:
    paths = paths or GuardianActionCyclePaths()
    clock = _utc(now)
    environ = env if env is not None else os.environ

    receipt_payload = _load_json(paths.action_receipt)
    escalation_payload = _load_json(paths.escalation)
    events_payload = _load_json(paths.events)
    event_state_payload = _load_json(paths.event_state)
    snapshot_payload = _load_json(paths.broker_snapshot)
    daily_target_state = _load_json(paths.daily_target_state)
    order_intents = _load_json(paths.order_intents)
    action_review_text = _read_text(paths.action_review)
    previous_result = _load_json(paths.result)

    receipt = _receipt_from_payload(receipt_payload)
    action = str(receipt.get("action") or "").upper()
    events = _events_from_payload(events_payload)
    selected_event = {event.event_id: event for event in events}.get(str(receipt.get("event_id") or ""))
    verifier = review_guardian_action_receipt(
        receipt_payload,
        events=events,
        previous_state=event_state_payload,
        selected_event=selected_event,
        now=clock,
    )
    flags = _execution_flags(environ)
    lane_id = _selected_lane_id(receipt, order_intents)

    strict_issues = _strict_receipt_issues(
        receipt_payload=receipt_payload,
        receipt=receipt,
        verifier=verifier,
        selected_event=selected_event,
        events_payload=events_payload,
        event_state_payload=event_state_payload,
        snapshot_payload=snapshot_payload,
        previous_result=previous_result,
        now=clock,
    )
    strict_issues.extend(
        _entry_intent_binding_issues(
            action=action,
            receipt=receipt,
            selected_event=selected_event,
            lane_id=lane_id,
            order_intents_payload=order_intents,
        )
    )
    manual_safety = _manual_exposure_safety(
        receipt=receipt,
        action=action,
        snapshot_payload=snapshot_payload,
        selected_event=selected_event,
    )
    strict_issues.extend(manual_safety["issues"])

    lock = _active_live_lock(paths.live_lock)
    if lock["active"]:
        strict_issues.append(_issue("ACTIVE_TRADER_OR_GATEWAY_LOCK", "active trader/gateway lock conflicts with guardian action cycle"))

    freshness = _snapshot_freshness(snapshot_payload, now=clock, env=environ)
    refresh = {"status": "SKIPPED", "reason": "snapshot fresh or execution flags not fully enabled"}
    material_change = {"status": "NOT_CHECKED", "changed": False}
    refreshed_snapshot = snapshot_payload
    if not freshness["fresh"]:
        if flags["all_enabled"]:
            refresh = _refresh_broker_snapshot(paths=paths, env=environ, command_runner=command_runner)
            refreshed_snapshot = _load_json(paths.broker_snapshot)
            material_change = _material_broker_truth_change(
                before=snapshot_payload,
                after=refreshed_snapshot,
                pair=str(receipt.get("pair") or (selected_event.pair if selected_event else "")),
            )
            freshness = _snapshot_freshness(refreshed_snapshot, now=clock, env=environ)
            if refresh.get("status") != "OK":
                strict_issues.append(_issue("BROKER_SNAPSHOT_REFRESH_FAILED", "stale broker snapshot refresh failed"))
            elif not freshness["fresh"]:
                strict_issues.append(_issue("BROKER_SNAPSHOT_STILL_STALE", "broker snapshot is still stale after refresh"))
            if material_change["changed"]:
                strict_issues.append(_issue("BROKER_TRUTH_CHANGED", "broker truth changed after refresh; require next cycle"))
        else:
            strict_issues.append(_issue("BROKER_SNAPSHOT_STALE", "stale broker snapshot blocks guardian execution"))

    ledger_sync = {"status": "SKIPPED", "reason": "execution not requested or receipt not executable"}
    if flags["all_enabled"] and action in ENTRY_ACTIONS and not strict_issues:
        ledger_sync = _sync_execution_ledger(paths=paths, env=environ, command_runner=command_runner)

    risk_result = _risk_result(
        action=action,
        lane_id=lane_id,
        order_intents_payload=order_intents,
        snapshot_payload=refreshed_snapshot,
        live_enabled=flags["live_enabled"],
        now=clock,
    )
    if flags["all_enabled"] and action in ENTRY_ACTIONS and not (
        str(risk_result.get("status") or "").upper() == "ALLOWED"
        and risk_result.get("allowed") is True
    ):
        strict_issues.append(
            _issue(
                "RISK_ENGINE_NOT_ALLOWED",
                f"entry requires explicit ALLOWED/allowed=true, got {risk_result.get('status') or 'missing'}",
            )
        )

    no_send = []
    if not flags["all_enabled"]:
        no_send.append("LIVE_FLAGS_DISABLED")
    if strict_issues:
        no_send.append("RECEIPT_OR_SAFETY_REJECTED")
    if action in POSITION_ACTIONS:
        no_send.append("NEEDS_TRADER_CONFIRMATION")
    if action in {"HOLD", "NO_ACTION"}:
        no_send.append("NO_EXECUTABLE_ACTION")
    if action in ENTRY_ACTIONS and not lane_id:
        no_send.append("NEEDS_TRADER_CONFIRMATION")
    if action in ENTRY_ACTIONS and not (
        str(risk_result.get("status") or "").upper() == "ALLOWED"
        and risk_result.get("allowed") is True
    ):
        no_send.append("RISK_ENGINE_BLOCKED")

    gateway_result = None
    executed = False
    if action in ENTRY_ACTIONS and lane_id and not no_send:
        gateway_result = (
            gateway_runner(lane_id, True)
            if gateway_runner is not None
            else _run_live_order_gateway(
                lane_id,
                True,
                intents_path=paths.order_intents,
                target_state_path=paths.daily_target_state,
                target_report_path=paths.root / "docs" / "daily_target_report.md",
                execution_ledger_db_path=paths.root / "data" / "execution_ledger.db",
                execution_ledger_report_path=paths.root / "docs" / "execution_ledger_report.md",
                verified_decision_path=paths.gpt_decision,
                live_enabled=flags["live_enabled"],
            )
        )
        executed = bool(gateway_result.get("sent")) or str(gateway_result.get("status") or "").upper() == "SENT"
    elif action in ENTRY_ACTIONS and lane_id and flags["all_enabled"] and not strict_issues:
        gateway_result = {"status": "SKIPPED", "reason": sorted(set(no_send))}

    result = {
        "generated_at_utc": clock.isoformat(),
        "status": _status_for_result(action=action, issues=strict_issues, no_send_reasons=no_send, executed=executed),
        "no_direct_oanda": True,
        "input_receipt": receipt_payload,
        "receipt": receipt,
        "selected_event": selected_event.to_payload() if selected_event else None,
        "verifier_result": verifier,
        "strict_receipt_issues": strict_issues,
        "manual_exposure_safety": manual_safety,
        "execution_flags": flags,
        "lock": lock,
        "broker_snapshot_freshness": freshness,
        "broker_snapshot_refresh": refresh,
        "broker_truth_material_change": material_change,
        "execution_ledger_sync": ledger_sync,
        "risk_engine_result": risk_result,
        "selected_lane_id": lane_id,
        "gateway_required": True,
        "required_gateway": "LiveOrderGateway" if action in ENTRY_ACTIONS else "PositionProtectionGateway",
        "gateway_result": gateway_result,
        "executed": executed,
        "no_send_reason": sorted(set(no_send)),
        "snapshot_before": _compact_snapshot_payload(snapshot_payload),
        "snapshot_after": _compact_snapshot_payload(_load_json(paths.broker_snapshot)) if executed else None,
        "inputs": {
            "guardian_action_receipt": str(paths.action_receipt),
            "guardian_escalation": str(paths.escalation),
            "guardian_events": str(paths.events),
            "guardian_event_state": str(paths.event_state),
            "broker_snapshot": str(paths.broker_snapshot),
            "daily_target_state": str(paths.daily_target_state),
            "order_intents": str(paths.order_intents),
            "guardian_action_review": str(paths.action_review),
        },
        "daily_target_state_mode": daily_target_state.get("mode"),
        "guardian_escalation": escalation_payload,
        "guardian_action_review_excerpt": action_review_text[:2000],
    }
    result["receipt_lifecycle_update"] = _update_receipt_lifecycle_after_cycle(
        paths.action_receipt,
        status=str(result["status"]),
        issues=strict_issues,
        now=clock,
    )
    _write_json(paths.result, result)
    _write_report(paths.report, result)
    _append_log(paths.log, result)
    return result


def _strict_receipt_issues(
    *,
    receipt_payload: dict[str, Any],
    receipt: dict[str, Any],
    verifier: dict[str, Any],
    selected_event: GuardianEvent | None,
    events_payload: dict[str, Any],
    event_state_payload: dict[str, Any],
    snapshot_payload: dict[str, Any],
    previous_result: dict[str, Any],
    now: datetime,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    action = str(receipt.get("action") or "").upper()
    source = str(receipt_payload.get("source") or receipt.get("source") or "")
    model = str(receipt_payload.get("model") or receipt.get("model") or "")
    if source != "guardian_wake_dispatcher":
        issues.append(_issue("GUARDIAN_ACTION_BAD_SOURCE", "receipt source must be guardian_wake_dispatcher"))
    if model != "gpt-5.5":
        issues.append(_issue("GUARDIAN_ACTION_BAD_MODEL", "guardian wake receipt must come from gpt-5.5"))
    receipt_status = str(receipt_payload.get("receipt_status") or receipt_payload.get("status") or "").upper()
    receipt_lifecycle = str(receipt_payload.get("receipt_lifecycle") or ("ACTIVE" if receipt_status == "ACCEPTED" else "")).upper()
    if receipt_status and receipt_status != "ACCEPTED":
        issues.append(_issue("GUARDIAN_ACTION_RECEIPT_NOT_ACCEPTED", "receipt_status must be ACCEPTED"))
    if receipt_lifecycle and receipt_lifecycle != "ACTIVE":
        issues.append(_issue("GUARDIAN_ACTION_RECEIPT_NOT_ACTIVE", f"receipt_lifecycle is {receipt_lifecycle}"))
    expires_at = _parse_utc(receipt_payload.get("expires_at_utc"))
    if expires_at is not None and expires_at <= now:
        issues.append(_issue("GUARDIAN_ACTION_RECEIPT_EXPIRED", "guardian action receipt is past expires_at_utc"))
    if action not in GUARDIAN_ACTIONS:
        issues.append(_issue("GUARDIAN_ACTION_BAD_ACTION", f"unsupported guardian action {action!r}"))
    if receipt.get("gateway_required") is not True:
        issues.append(_issue("GUARDIAN_ACTION_GATEWAY_REQUIRED", "gateway_required=true is required"))
    if receipt.get("no_direct_oanda") is not True and receipt_payload.get("no_direct_oanda") is not True:
        issues.append(_issue("GUARDIAN_ACTION_NO_DIRECT_OANDA_REQUIRED", "no_direct_oanda=true is required"))
    if action in ENTRY_ACTIONS and receipt.get("new_information") is not True:
        issues.append(_issue("GUARDIAN_ACTION_REQUIRES_NEW_INFORMATION", "TRADE/ADD requires new_information=true"))
    canonical_pair = selected_event.pair if selected_event is not None else ""
    if action in ENTRY_ACTIONS and selected_event is not None:
        if selected_event.action_hint.upper() not in ENTRY_ACTIONS:
            issues.append(
                _issue(
                    "GUARDIAN_ACTION_EVENT_DOES_NOT_AUTHORIZE_ENTRY",
                    "TRADE/ADD requires selected event action_hint TRADE or ADD",
                )
            )
        if str(selected_event.direction or "").upper() not in {"LONG", "SHORT"}:
            issues.append(
                _issue(
                    "GUARDIAN_ACTION_EVENT_DIRECTION_REQUIRED",
                    "TRADE/ADD requires selected event direction LONG or SHORT",
                )
            )
    if action in ENTRY_ACTIONS and canonical_pair in _technical_input_blocked_pairs(
        events_payload=events_payload,
        event_state_payload=event_state_payload,
    ):
        issues.append(
            _issue(
                "GUARDIAN_ACTION_TECHNICAL_INPUT_STALE",
                f"{canonical_pair} technical input is fail-closed; TRADE/ADD is forbidden",
            )
        )
    if action in ENTRY_ACTIONS and canonical_pair and not _technical_state_available_for_entry(
        pair=canonical_pair,
        events_payload=events_payload,
        event_state_payload=event_state_payload,
        now=now,
    ):
        issues.append(
            _issue(
                "GUARDIAN_TECHNICAL_STATE_UNAVAILABLE",
                f"{canonical_pair} has no current fail-closed technical baseline",
            )
        )
    if verifier.get("status") != "ACCEPTED":
        issues.append(_issue("GUARDIAN_ACTION_VERIFIER_REJECTED", "guardian receipt verifier did not accept the receipt"))

    if not str(receipt.get("event_id") or "") or selected_event is None:
        issues.append(_issue("GUARDIAN_ACTION_UNKNOWN_EVENT", "receipt event_id must match a current guardian event"))
    dedupe = _receipt_dedupe_key(receipt_payload, receipt)
    if selected_event is not None and dedupe and dedupe != selected_event.dedupe_key:
        issues.append(_issue("GUARDIAN_ACTION_DEDUPE_MISMATCH", "receipt dedupe_key does not match current guardian event"))
    if selected_event is not None and not dedupe:
        issues.append(_issue("GUARDIAN_ACTION_DEDUPE_MISSING", "receipt must carry current guardian event dedupe_key proof"))

    thesis_state = str(receipt.get("thesis_state") or (selected_event.thesis_state if selected_event else "")).upper()
    allowed = ALLOWED_BY_THESIS_STATE.get(thesis_state)
    if action in ENTRY_ACTIONS and thesis_state != "ALIVE":
        issues.append(_issue("GUARDIAN_ACTION_THESIS_STATE_BLOCKS_ENTRY", f"{action} requires thesis_state=ALIVE"))
    if allowed is not None and action not in allowed:
        issues.append(_issue("GUARDIAN_ACTION_THESIS_STATE_ACTION_BLOCKED", f"{thesis_state} can only use {sorted(allowed)}"))

    reason_blob = " ".join(str(receipt.get(key) or "") for key in ("reason", "review_reason", "operator_summary")).lower()
    if _truthy(receipt.get("schedule_only")) or "scheduled hour" in reason_blob or "hourly" in reason_blob:
        issues.append(_issue("GUARDIAN_ACTION_SCHEDULE_ONLY", "scheduled-hour-only reason is forbidden"))
    if _truthy(receipt.get("bc_churn_pace_fix")) or "b/c churn" in reason_blob or "bc churn" in reason_blob:
        issues.append(_issue("GUARDIAN_ACTION_BC_CHURN", "B/C churn cannot be used to fix pace"))
    if str(receipt.get("same_pair_add_type") or "").upper() == "PYRAMID_WITH_MOVE" and not _truthy(
        receipt.get("independent_fresh_edge")
    ):
        issues.append(_issue("GUARDIAN_ACTION_PYRAMID_NEEDS_FRESH_EDGE", "with-move pyramid requires independent fresh edge"))
    if _truthy(receipt.get("same_pair_thesis_action_recently_sent")) or _truthy(receipt.get("duplicate_recent_action")):
        issues.append(_issue("GUARDIAN_ACTION_RECENT_DUPLICATE", "duplicate action throttle failed"))
    if _previous_result_duplicate(previous_result, receipt=receipt, dedupe_key=dedupe):
        issues.append(_issue("GUARDIAN_ACTION_DUPLICATE_RECEIPT", "this guardian receipt/action already executed"))
    if not isinstance(events_payload.get("events"), list):
        issues.append(_issue("GUARDIAN_EVENTS_MISSING", "guardian_events.json must contain current events list"))
    if not snapshot_payload:
        issues.append(_issue("BROKER_SNAPSHOT_MISSING", "broker_snapshot.json is required before guardian action"))
    return issues


def _technical_input_blocked_pairs(
    *,
    events_payload: dict[str, Any],
    event_state_payload: dict[str, Any],
) -> set[str]:
    current_rows = [
        item
        for item in events_payload.get("events", []) or []
        if isinstance(item, dict)
    ]
    fresh_pairs = {
        str(item.get("pair") or "").upper()
        for item in current_rows
        if str(item.get("event_type") or "").upper() == "TECHNICAL_STATE_CHANGE"
        and str(item.get("pair") or "").strip()
    }
    blocked = {
        str(item.get("pair") or "").upper()
        for item in current_rows
        if str(item.get("event_type") or "").upper() == "TECHNICAL_INPUT_STALE"
        and str(item.get("pair") or "").strip()
    }
    state_events = (
        event_state_payload.get("events")
        if isinstance(event_state_payload.get("events"), dict)
        else {}
    )
    for item in state_events.values():
        if not isinstance(item, dict):
            continue
        if str(item.get("event_type") or "").upper() != "TECHNICAL_INPUT_STALE":
            continue
        pair = str(item.get("pair") or "").upper()
        if pair and pair not in fresh_pairs:
            blocked.add(pair)
    return blocked


def _technical_state_available_for_entry(
    *,
    pair: str,
    events_payload: dict[str, Any],
    event_state_payload: dict[str, Any],
    now: datetime,
) -> bool:
    pair = pair.upper()
    del events_payload
    generated = _strict_aware_utc(event_state_payload.get("generated_at_utc"))
    state_events = (
        event_state_payload.get("events")
        if isinstance(event_state_payload.get("events"), dict)
        else None
    )
    if generated is None or state_events is None:
        return False
    age = (now - generated).total_seconds()
    if age < -5 or age > DEFAULT_EVENT_STATE_MAX_AGE_SECONDS:
        return False
    return any(
        isinstance(item, dict)
        and str(item.get("pair") or "").upper() == pair
        and str(item.get("event_type") or "").upper() == "TECHNICAL_STATE_CHANGE"
        and _clock_is_current(
            item.get("last_seen_at_utc"),
            now=now,
            max_age_seconds=DEFAULT_EVENT_STATE_MAX_AGE_SECONDS,
        )
        for item in state_events.values()
    )


def _clock_is_current(
    value: Any,
    *,
    now: datetime,
    max_age_seconds: int,
) -> bool:
    parsed = _strict_aware_utc(value)
    if parsed is None:
        return False
    age = (now - parsed).total_seconds()
    return age >= -5 and age <= max(1, max_age_seconds)


def _entry_intent_binding_issues(
    *,
    action: str,
    receipt: dict[str, Any],
    selected_event: GuardianEvent | None,
    lane_id: str | None,
    order_intents_payload: dict[str, Any],
) -> list[dict[str, str]]:
    if action not in ENTRY_ACTIONS or selected_event is None or not lane_id:
        return []
    selected = _find_intent_result(order_intents_payload, lane_id)
    if selected is None:
        return []
    intent = selected.get("intent") if isinstance(selected.get("intent"), dict) else selected
    intent_pair = str(intent.get("pair") or "").upper()
    intent_side = str(intent.get("side") or "").upper()
    receipt_pair = str(receipt.get("pair") or "").upper()
    receipt_side = str(receipt.get("side") or receipt.get("direction") or "").upper()
    event_pair = selected_event.pair.upper()
    event_side = str(selected_event.direction or "").upper()
    issues: list[dict[str, str]] = []
    if not receipt_pair or receipt_pair != event_pair or intent_pair != event_pair:
        issues.append(
            _issue(
                "GUARDIAN_ACTION_ENTRY_PAIR_MISMATCH",
                "receipt, selected event, and selected intent must bind the same pair",
            )
        )
    if not receipt_side or (event_side and receipt_side != event_side) or intent_side != receipt_side:
        issues.append(
            _issue(
                "GUARDIAN_ACTION_ENTRY_SIDE_MISMATCH",
                "receipt, selected event, and selected intent must bind the same side",
            )
        )
    return issues


def _manual_exposure_safety(
    *,
    receipt: dict[str, Any],
    action: str,
    snapshot_payload: dict[str, Any],
    selected_event: GuardianEvent | None,
) -> dict[str, Any]:
    manual_positions = [
        item
        for item in snapshot_payload.get("positions", []) or []
        if str(item.get("owner") or "").lower() in {"manual", "unknown", "operator_manual"}
    ]
    pair = str(receipt.get("pair") or (selected_event.pair if selected_event else "")).upper()
    target_ids = {str(item) for item in receipt.get("trade_ids", []) or []}
    target_ids |= {str(item) for item in receipt.get("close_trade_ids", []) or []}
    target_ids |= {str(item) for item in receipt.get("reduce_trade_ids", []) or []}
    touched = [
        item
        for item in manual_positions
        if (pair and str(item.get("pair") or "").upper() == pair) or str(item.get("trade_id") or "") in target_ids
    ]
    touches_manual = bool(touched) or _truthy(receipt.get("touches_manual_exposure"))
    issues: list[dict[str, str]] = []
    if touches_manual and _truthy(receipt.get("attach_sl")):
        issues.append(_issue("MANUAL_SL_FORBIDDEN", "guardian action must not attach SL to manual/unknown exposure"))
    if touches_manual and action == "REDUCE":
        loss_side = [
            item
            for item in touched
            if float(item.get("unrealized_pl_jpy") or 0.0) < 0 and str(item.get("trade_id") or "") in target_ids
        ]
        if loss_side and not _truthy(receipt.get("operator_manual_loss_close_authorized")):
            issues.append(_issue("MANUAL_LOSS_CLOSE_FORBIDDEN", "manual/unknown loss-side close requires explicit operator authorization"))
    if touches_manual and action in ENTRY_ACTIONS and not _truthy(receipt.get("operator_manual_overlap_authorized")):
        issues.append(_issue("MANUAL_THEME_ADD_BLOCKED", "system add into same manual theme requires explicit authorization or gateway proof"))
    return {
        "touches_manual_or_unknown_exposure": touches_manual,
        "manual_positions_seen": len(manual_positions),
        "manual_positions_touched": len(touched),
        "manual_pl_counts_as_system_pl": False,
        "issues": issues,
    }


def _risk_result(
    *,
    action: str,
    lane_id: str | None,
    order_intents_payload: dict[str, Any],
    snapshot_payload: dict[str, Any],
    live_enabled: bool,
    now: datetime,
) -> dict[str, Any]:
    if action not in ENTRY_ACTIONS:
        return {"status": "SKIPPED", "reason": f"RiskEngine not applicable to {action}"}
    if not lane_id:
        return {"status": "NEEDS_TRADER_CONFIRMATION", "reason": "receipt did not identify a current order_intents lane_id"}
    selected = _find_intent_result(order_intents_payload, lane_id)
    if selected is None:
        return {"status": "REJECTED", "issues": [_issue("LANE_NOT_FOUND", f"lane_id {lane_id} not found in order_intents")]}
    try:
        intent = _intent_from_json(selected["intent"])
        snapshot = _snapshot_from_json(snapshot_payload)
        decision = RiskEngine(live_enabled=live_enabled, validation_time_utc=now).validate(intent, snapshot, for_live_send=True)
    except Exception as exc:
        return {"status": "REJECTED", "issues": [_issue("RISK_ENGINE_EXCEPTION", str(exc))]}
    return {
        "status": "ALLOWED" if decision.allowed else "BLOCKED",
        "allowed": decision.allowed,
        "metrics": asdict(decision.metrics) if decision.metrics else None,
        "issues": [asdict(issue) for issue in decision.issues],
        "lane_id": lane_id,
    }


def _selected_lane_id(receipt: dict[str, Any], order_intents_payload: dict[str, Any]) -> str | None:
    for key in ("lane_id", "selected_lane_id", "intent_lane_id"):
        value = str(receipt.get(key) or "").strip()
        if value:
            return value
    selected_lane_ids = receipt.get("selected_lane_ids")
    if isinstance(selected_lane_ids, list) and selected_lane_ids:
        value = str(selected_lane_ids[0] or "").strip()
        if value:
            return value
    pair = str(receipt.get("pair") or "").upper()
    side = str(receipt.get("side") or receipt.get("direction") or "").upper()
    matches = []
    for item in order_intents_payload.get("results", []) or order_intents_payload.get("intents", []) or []:
        if not isinstance(item, dict):
            continue
        intent = item.get("intent") if isinstance(item.get("intent"), dict) else item
        if str(intent.get("pair") or "").upper() == pair and str(intent.get("side") or "").upper() == side:
            if str(item.get("status") or "").upper() in {"LIVE_READY", "STAGED", "READY"} or item.get("risk_allowed") is True:
                matches.append(str(item.get("lane_id") or intent.get("lane_id") or ""))
    matches = [item for item in matches if item]
    return matches[0] if len(matches) == 1 else None


def _run_live_order_gateway(
    lane_id: str,
    send: bool,
    *,
    intents_path: Path = DEFAULT_ORDER_INTENTS,
    target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
    target_report_path: Path | None = None,
    execution_ledger_db_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
    execution_ledger_report_path: Path = DEFAULT_EXECUTION_LEDGER_REPORT,
    verified_decision_path: Path = DEFAULT_GPT_TRADER_DECISION,
    live_enabled: bool | None = None,
) -> dict[str, Any]:
    from quant_rabbit.broker.execution import (
        LiveOrderGateway,
        verified_trade_size_multiple,
    )
    from quant_rabbit.broker.oanda import OandaExecutionClient

    client = OandaExecutionClient()
    summary = LiveOrderGateway(
        client=client,
        output_path=DEFAULT_LIVE_ORDER_REQUEST,
        report_path=DEFAULT_LIVE_ORDER_STAGE_REPORT,
        live_enabled=(os.environ.get("QR_LIVE_ENABLED") == "1") if live_enabled is None else live_enabled,
        target_state_path=target_state_path,
        target_report_path=target_report_path,
        execution_ledger_db_path=execution_ledger_db_path,
        execution_ledger_report_path=execution_ledger_report_path,
        verified_decision_path=verified_decision_path,
    ).run(
        intents_path=intents_path,
        lane_id=lane_id,
        size_multiple=(verified_trade_size_multiple(verified_decision_path) or 1.0),
        send=send,
        confirm_live=send,
    )
    return {
        "status": summary.status,
        "lane_id": summary.lane_id,
        "sent": summary.sent,
        "risk_issues": summary.risk_issues,
        "strategy_issues": summary.strategy_issues,
        "output_path": str(summary.output_path),
        "report_path": str(summary.report_path),
    }


def _refresh_broker_snapshot(*, paths: GuardianActionCyclePaths, env: dict[str, str], command_runner: CommandRunner) -> dict[str, Any]:
    cmd = [env.get("QR_PYTHON", os.environ.get("QR_PYTHON", "python3")), "-m", "quant_rabbit.cli", "broker-snapshot", "--output", str(paths.broker_snapshot)]
    run_env = dict(env)
    run_env["PYTHONPATH"] = str(paths.root / "src")
    try:
        proc = command_runner(cmd, cwd=str(paths.root), capture_output=True, text=True, timeout=120, env=run_env)
    except Exception as exc:
        return {"status": "FAILED", "command": cmd, "error": str(exc)}
    return {
        "status": "OK" if getattr(proc, "returncode", 1) == 0 else "FAILED",
        "command": cmd,
        "returncode": getattr(proc, "returncode", None),
        "stdout_tail": str(getattr(proc, "stdout", "") or "")[-1000:],
        "stderr_tail": str(getattr(proc, "stderr", "") or "")[-1000:],
    }


def _sync_execution_ledger(*, paths: GuardianActionCyclePaths, env: dict[str, str], command_runner: CommandRunner) -> dict[str, Any]:
    cmd = [env.get("QR_PYTHON", os.environ.get("QR_PYTHON", "python3")), "-m", "quant_rabbit.cli", "execution-ledger-sync"]
    run_env = dict(env)
    run_env["PYTHONPATH"] = str(paths.root / "src")
    try:
        proc = command_runner(cmd, cwd=str(paths.root), capture_output=True, text=True, timeout=180, env=run_env)
    except Exception as exc:
        return {"status": "FAILED", "command": cmd, "error": str(exc)}
    return {
        "status": "OK" if getattr(proc, "returncode", 1) == 0 else "FAILED",
        "command": cmd,
        "returncode": getattr(proc, "returncode", None),
        "stdout_tail": str(getattr(proc, "stdout", "") or "")[-1000:],
        "stderr_tail": str(getattr(proc, "stderr", "") or "")[-1000:],
    }


def _execution_flags(env: dict[str, str]) -> dict[str, Any]:
    flags = {
        "QR_LIVE_ENABLED": env.get("QR_LIVE_ENABLED", "0"),
        "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": env.get("QR_GUARDIAN_WAKE_GATEWAY_HANDOFF", "0"),
        "QR_GUARDIAN_ACTION_EXECUTE": env.get("QR_GUARDIAN_ACTION_EXECUTE", "0"),
    }
    return {
        **flags,
        "live_enabled": flags["QR_LIVE_ENABLED"] == "1",
        "handoff_enabled": flags["QR_GUARDIAN_WAKE_GATEWAY_HANDOFF"] == "1",
        "action_execute_enabled": flags["QR_GUARDIAN_ACTION_EXECUTE"] == "1",
        "all_enabled": all(value == "1" for value in flags.values()),
    }


def _snapshot_freshness(payload: dict[str, Any], *, now: datetime, env: dict[str, str]) -> dict[str, Any]:
    account = payload.get("account") if isinstance(payload.get("account"), dict) else {}
    fetched = _parse_utc(payload.get("fetched_at_utc") or account.get("fetched_at_utc"))
    max_age = int(env.get("QR_GUARDIAN_ACTION_MAX_SNAPSHOT_AGE_SECONDS", DEFAULT_SNAPSHOT_MAX_AGE_SECONDS))
    if fetched is None:
        return {"fresh": False, "status": "MISSING_FETCHED_AT", "max_age_seconds": max_age}
    age = (now - fetched).total_seconds()
    return {
        "fresh": age <= max_age,
        "status": "FRESH" if age <= max_age else "STALE",
        "age_seconds": age,
        "max_age_seconds": max_age,
        "fetched_at_utc": fetched.isoformat(),
    }


def _material_broker_truth_change(*, before: dict[str, Any], after: dict[str, Any], pair: str) -> dict[str, Any]:
    before_key = _broker_truth_key(before, pair=pair)
    after_key = _broker_truth_key(after, pair=pair)
    return {"status": "CHECKED", "changed": before_key != after_key, "pair": pair, "before": before_key, "after": after_key}


def _broker_truth_key(payload: dict[str, Any], *, pair: str) -> dict[str, Any]:
    pair = pair.upper()
    positions = [
        {
            "trade_id": item.get("trade_id"),
            "pair": item.get("pair"),
            "side": item.get("side"),
            "units": item.get("units"),
            "owner": item.get("owner"),
            "take_profit": item.get("take_profit"),
            "stop_loss": item.get("stop_loss"),
        }
        for item in payload.get("positions", []) or []
        if not pair or str(item.get("pair") or "").upper() == pair
    ]
    orders = [
        {
            "order_id": item.get("order_id"),
            "pair": item.get("pair"),
            "order_type": item.get("order_type"),
            "state": item.get("state"),
            "units": item.get("units"),
            "owner": item.get("owner"),
        }
        for item in payload.get("orders", []) or []
        if not pair or str(item.get("pair") or "").upper() == pair
    ]
    account = payload.get("account") if isinstance(payload.get("account"), dict) else {}
    return {
        "last_transaction_id": account.get("last_transaction_id"),
        "positions": sorted(positions, key=lambda item: str(item.get("trade_id") or "")),
        "orders": sorted(orders, key=lambda item: str(item.get("order_id") or "")),
        "quote": (payload.get("quotes") or {}).get(pair),
    }


def _active_live_lock(lock_dir: Path) -> dict[str, Any]:
    if not lock_dir.exists():
        return {"active": False, "path": str(lock_dir)}
    pid_text = _read_text(lock_dir / "pid").strip()
    active = pid_text.isdigit() and _pid_running(int(pid_text))
    return {
        "active": bool(active),
        "path": str(lock_dir),
        "pid": int(pid_text) if pid_text.isdigit() else None,
        "command": _read_text(lock_dir / "command").strip() or None,
        "started_at_utc": _read_text(lock_dir / "started_at_utc").strip() or None,
    }


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _events_from_payload(payload: dict[str, Any]) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    for item in payload.get("events", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            events.append(
                GuardianEvent(
                    event_id=str(item.get("event_id") or ""),
                    event_type=str(item.get("event_type") or ""),
                    pair=str(item.get("pair") or ""),
                    direction=item.get("direction"),
                    thesis=str(item.get("thesis") or ""),
                    price_zone=str(item.get("price_zone") or ""),
                    severity=str(item.get("severity") or "P2"),
                    recommended_review_type=str(item.get("recommended_review_type") or ""),
                    dedupe_key=str(item.get("dedupe_key") or ""),
                    action_hint=str(item.get("action_hint") or ""),
                    thesis_state=str(item.get("thesis_state") or "UNKNOWN"),
                    detected_at_utc=str(item.get("detected_at_utc") or ""),
                    details=item.get("details") if isinstance(item.get("details"), dict) else {},
                )
            )
        except TypeError:
            continue
    return events


def _intent_from_json(payload: dict[str, Any]) -> OrderIntent:
    return OrderIntent(
        pair=str(payload["pair"]).upper(),
        side=Side.parse(str(payload["side"])),
        order_type=OrderType.parse(str(payload["order_type"])),
        units=int(payload["units"]),
        entry=float(payload["entry"]) if payload.get("entry") is not None else None,
        tp=float(payload["tp"]),
        sl=float(payload["sl"]),
        thesis=str(payload.get("thesis") or ""),
        reason=str(payload.get("reason") or ""),
        owner=Owner(str(payload.get("owner") or Owner.TRADER.value)),
        market_context=_market_context_from_json(payload.get("market_context")),
        metadata=dict(payload.get("metadata") or {}),
    )


def _market_context_from_json(payload: object) -> MarketContext | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("market_context must be an object")
    return MarketContext(
        regime=str(payload.get("regime") or ""),
        narrative=str(payload.get("narrative") or ""),
        chart_story=str(payload.get("chart_story") or ""),
        method=TradeMethod.parse(str(payload.get("method") or "")),
        invalidation=str(payload.get("invalidation") or ""),
        event_risk=str(payload.get("event_risk") or ""),
        session=str(payload.get("session") or ""),
    )


def _snapshot_from_json(payload: dict[str, Any]) -> BrokerSnapshot:
    positions = [
        BrokerPosition(
            trade_id=str(item["trade_id"]),
            pair=str(item["pair"]),
            side=Side.parse(str(item["side"])),
            units=int(item["units"]),
            entry_price=float(item["entry_price"]),
            unrealized_pl_jpy=float(item.get("unrealized_pl_jpy") or 0.0),
            take_profit=float(item["take_profit"]) if item.get("take_profit") is not None else None,
            stop_loss=float(item["stop_loss"]) if item.get("stop_loss") is not None else None,
            owner=Owner(str(item.get("owner") or Owner.UNKNOWN.value)),
            raw=dict(item.get("raw") or {}),
        )
        for item in payload.get("positions", []) or []
    ]
    orders = [
        BrokerOrder(
            order_id=str(item["order_id"]),
            pair=item.get("pair"),
            order_type=str(item.get("order_type") or ""),
            trade_id=item.get("trade_id"),
            price=float(item["price"]) if item.get("price") is not None else None,
            state=item.get("state"),
            units=int(item["units"]) if item.get("units") is not None else None,
            owner=Owner(str(item.get("owner") or Owner.UNKNOWN.value)),
            raw=dict(item.get("raw") or {}),
        )
        for item in payload.get("orders", []) or []
    ]
    quotes = {}
    for pair, item in (payload.get("quotes") or {}).items():
        quotes[str(pair)] = Quote(
            pair=str(pair),
            bid=float(item["bid"]),
            ask=float(item["ask"]),
            timestamp_utc=_parse_utc(item.get("timestamp_utc")) or datetime.now(timezone.utc),
        )
    return BrokerSnapshot(
        fetched_at_utc=_parse_utc(payload.get("fetched_at_utc")) or datetime.now(timezone.utc),
        positions=tuple(positions),
        orders=tuple(orders),
        quotes=quotes,
        account=_account_summary_from_json(payload.get("account")),
        home_conversions={str(k).upper(): float(v) for k, v in (payload.get("home_conversions") or {}).items()},
    )


def _account_summary_from_json(payload: object) -> AccountSummary | None:
    if not isinstance(payload, dict):
        return None
    return AccountSummary(
        nav_jpy=float(payload.get("nav_jpy") or 0.0),
        balance_jpy=float(payload.get("balance_jpy") or 0.0),
        unrealized_pl_jpy=float(payload.get("unrealized_pl_jpy") or 0.0),
        margin_used_jpy=float(payload.get("margin_used_jpy") or 0.0),
        margin_available_jpy=float(payload.get("margin_available_jpy") or 0.0),
        pl_jpy=float(payload.get("pl_jpy") or 0.0),
        financing_jpy=float(payload.get("financing_jpy") or 0.0),
        last_transaction_id=str(payload.get("last_transaction_id") or ""),
        hedging_enabled=bool(payload.get("hedging_enabled") or False),
        fetched_at_utc=_parse_utc(payload.get("fetched_at_utc")) or datetime.now(timezone.utc),
    )


def _find_intent_result(payload: dict[str, Any], lane_id: str) -> dict[str, Any] | None:
    for item in payload.get("results", []) or payload.get("intents", []) or []:
        if not isinstance(item, dict):
            continue
        intent = item.get("intent") if isinstance(item.get("intent"), dict) else item
        if str(item.get("lane_id") or intent.get("lane_id") or "") == lane_id:
            return item
    return None


def _receipt_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    receipt = payload.get("receipt") if isinstance(payload.get("receipt"), dict) else payload
    return dict(receipt) if isinstance(receipt, dict) else {}


def _receipt_dedupe_key(payload: dict[str, Any], receipt: dict[str, Any]) -> str:
    sources = [
        receipt,
        payload,
        payload.get("selected_event") if isinstance(payload.get("selected_event"), dict) else {},
        payload.get("event") if isinstance(payload.get("event"), dict) else {},
    ]
    for source in sources:
        value = str((source or {}).get("dedupe_key") or (source or {}).get("guardian_event_dedupe_key") or "").strip()
        if value:
            return value
    return ""


def _previous_result_duplicate(previous: dict[str, Any], *, receipt: dict[str, Any], dedupe_key: str) -> bool:
    if not previous or not previous.get("executed"):
        return False
    previous_payload = previous.get("input_receipt") if isinstance(previous.get("input_receipt"), dict) else {}
    previous_receipt = _receipt_from_payload(previous_payload)
    return (
        str(previous_receipt.get("event_id") or "") == str(receipt.get("event_id") or "")
        and str(previous_receipt.get("action") or "").upper() == str(receipt.get("action") or "").upper()
        and _receipt_dedupe_key(previous_payload, previous_receipt) == dedupe_key
    )


def _status_for_result(*, action: str, issues: list[dict[str, str]], no_send_reasons: list[str], executed: bool) -> str:
    if issues:
        return "REJECTED"
    if executed:
        return "EXECUTED"
    if "NEEDS_TRADER_CONFIRMATION" in no_send_reasons:
        return "NEEDS_TRADER_CONFIRMATION"
    if action in {"HOLD", "NO_ACTION"}:
        return "VERIFIED_NO_ACTION"
    if no_send_reasons:
        return "VERIFIED_NO_SEND"
    return "VERIFIED"


def _update_receipt_lifecycle_after_cycle(
    path: Path,
    *,
    status: str,
    issues: list[dict[str, str]],
    now: datetime,
) -> dict[str, Any]:
    payload = _load_json(path)
    if not payload:
        return {"status": "SKIPPED", "reason": "receipt missing"}
    current = str(payload.get("receipt_lifecycle") or ("ACTIVE" if str(payload.get("status") or "").upper() == "ACCEPTED" else "")).upper()
    if current and current != "ACTIVE":
        return {"status": "SKIPPED", "reason": f"receipt already {current}", "receipt_lifecycle": current}
    next_lifecycle = None
    consumed = bool(payload.get("consumed_by_trader", False))
    if status in {"EXECUTED", "VERIFIED_NO_ACTION"}:
        next_lifecycle = "CONSUMED"
        consumed = True
    elif status == "REJECTED":
        codes = {str(issue.get("code") or "") for issue in issues if isinstance(issue, dict)}
        next_lifecycle = "REJECTED"
        if "GUARDIAN_ACTION_RECEIPT_EXPIRED" in codes:
            next_lifecycle = "EXPIRED"
    if next_lifecycle is None:
        return {"status": "SKIPPED", "reason": f"cycle status {status} does not consume receipt"}
    updated = {
        **payload,
        "receipt_lifecycle": next_lifecycle,
        "consumed_by_trader": consumed,
        "lifecycle_updated_at_utc": now.isoformat(),
        "lifecycle_updated_by": "guardian-action-cycle",
    }
    if next_lifecycle == "CONSUMED":
        updated["consumed_at_utc"] = now.isoformat()
    elif next_lifecycle == "REJECTED":
        updated["rejected_at_utc"] = now.isoformat()
    elif next_lifecycle == "EXPIRED":
        updated["expired_at_utc"] = now.isoformat()
    _write_json(path, updated)
    return {
        "status": "UPDATED",
        "receipt_lifecycle": next_lifecycle,
        "consumed_by_trader": consumed,
    }


def _compact_snapshot_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "fetched_at_utc": payload.get("fetched_at_utc"),
        "account": payload.get("account"),
        "positions": payload.get("positions", []),
        "orders": payload.get("orders", []),
        "quotes": payload.get("quotes", {}),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Guardian Action Cycle Report",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Action: `{payload.get('receipt', {}).get('action')}`",
        f"- Selected event: `{(payload.get('selected_event') or {}).get('event_id')}`",
        f"- Selected lane: `{payload.get('selected_lane_id') or 'none'}`",
        f"- Executed: `{payload.get('executed')}`",
        f"- No direct OANDA: `{payload.get('no_direct_oanda')}`",
        f"- Required gateway: `{payload.get('required_gateway')}`",
        "",
        "## No-Send Reason",
        "",
    ]
    lines.extend([f"- `{reason}`" for reason in payload.get("no_send_reason", [])] or ["- none"])
    lines.extend(["", "## Verifier", "", f"- Status: `{payload.get('verifier_result', {}).get('status')}`"])
    for issue in payload.get("strict_receipt_issues", []) or []:
        lines.append(f"- `{issue.get('severity')}` `{issue.get('code')}` {issue.get('message')}")
    lines.extend(["", "## RiskEngine", "", f"- Status: `{payload.get('risk_engine_result', {}).get('status')}`"])
    for issue in payload.get("risk_engine_result", {}).get("issues", []) or []:
        lines.append(f"- `{issue.get('severity')}` `{issue.get('code')}` {issue.get('message')}")
    gateway = payload.get("gateway_result")
    lines.extend(["", "## Gateway", ""])
    if isinstance(gateway, dict):
        lines.append(f"- Status: `{gateway.get('status')}`")
        lines.append(f"- Sent: `{gateway.get('sent', False)}`")
    else:
        lines.append("- not executed")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This cycle does not call OANDA directly.",
            "- TRADE / ADD execution must pass RiskEngine and LiveOrderGateway.",
            "- HARVEST / REDUCE / CANCEL_PENDING remain trader-confirmation unless an approved position gateway path exists.",
            "- Manual/unknown exposure is excluded from system P/L and cannot be loss-closed by guardian action without explicit operator authorization.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "generated_at_utc": payload.get("generated_at_utc"),
                    "status": payload.get("status"),
                    "action": payload.get("receipt", {}).get("action"),
                    "selected_event_id": (payload.get("selected_event") or {}).get("event_id"),
                    "selected_lane_id": payload.get("selected_lane_id"),
                    "executed": payload.get("executed"),
                    "no_send_reason": payload.get("no_send_reason"),
                    "no_direct_oanda": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "")
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def _strict_aware_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _utc(value: datetime | None) -> datetime:
    return datetime.now(timezone.utc) if value is None else value.astimezone(timezone.utc)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _issue(code: str, message: str, severity: str = "BLOCK") -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}
