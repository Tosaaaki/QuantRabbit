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

# A stale pending order has missed roughly three scheduled 20-minute decision
# windows. This is a lifecycle watchdog, not a pair-specific price threshold.
DEFAULT_STALE_PENDING_SECONDS = 3 * 20 * 60

# Early margin pressure uses the existing gateway cap as its anchor. At 90% of
# the cap, the next 1000u order can easily fail after quote drift; gateway still
# makes the final broker-truth decision.
MARGIN_PRESSURE_WARNING_CAP_FRACTION = 0.90

# Statistical rail touch bands; they describe where price is at the tail of the
# observed 24h distribution, not a fixed pip level or USD/JPY literal.
RANGE_RAIL_LOW_PERCENTILE = 0.08
RANGE_RAIL_HIGH_PERCENTILE = 0.92


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
        "inputs": {
            "snapshot": str(snapshot_path),
            "pair_charts": str(pair_charts_path),
            "order_intents": str(order_intents_path),
            "position_management": str(position_management_path),
            "thesis_evolution": str(thesis_evolution_path),
            "forecast_persistence": str(forecast_persistence_path),
            "market_context_matrix": str(market_context_matrix_path),
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

    collected.extend(_snapshot_events(snapshot, now=now))
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


def _snapshot_events(snapshot: dict[str, Any], *, now: datetime) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
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
    }.get(event_type, "ENTRY_REVIEW")


def _default_action_hint(event_type: str) -> str:
    return {
        "HARVEST_ZONE": "HARVEST",
        "THESIS_INVALIDATION": "REDUCE",
        "MARGIN_PRESSURE": "REDUCE",
        "UNKNOWN_ORDER": "REDUCE",
        "STALE_PENDING": "CANCEL_PENDING",
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
