from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.guardian_margin_contract import (
    MARGIN_PRESSURE_WARNING_CAP_FRACTION,
    P0_MARGIN_HARD_CAP_CONTRACT,
    P1_MARGIN_WARNING_CONTRACT,
)
from quant_rabbit.instruments import NORMAL_SPREAD_PIPS, instrument_pip_factor
from quant_rabbit.models import Owner
from quant_rabbit.operator_manual import (
    OPERATOR_MANUAL_POSITION_PACKET,
    is_operator_manual_position,
)
from quant_rabbit.risk import RiskPolicy
from quant_rabbit.strategy.directional_forecaster import (
    validate_mba_integrity_receipt,
)


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
    "EVENT_SUPPRESSED_WITH_PRICE_CHANGE",
    "TRIGGER_CONTRACT_EMPTY_FOR_ACTIVE_PAIR",
    "WATCH_ONLY_NO_TRIGGER_CONTRACT",
    "TECHNICAL_STATE_CHANGE",
    "TECHNICAL_INPUT_STALE",
)
SEVERITY_RANK = {"P2": 1, "P1": 2, "P0": 3}
THESIS_STATE_RANK = {"UNKNOWN": 0, "ALIVE": 1, "WOUNDED": 2, "INVALIDATED": 3, "EMERGENCY": 4}
GUARDIAN_ACTIONS = ("TRADE", "ADD", "HOLD", "HARVEST", "REDUCE", "CANCEL_PENDING", "NO_ACTION")
EVENT_ACTION_RECEIPT_EQUIVALENCE = {
    "TRADE": frozenset({"TRADE", "HOLD", "NO_ACTION"}),
    "ADD": frozenset({"ADD", "HOLD", "NO_ACTION"}),
    "HOLD": frozenset({"HOLD", "NO_ACTION"}),
    "HARVEST": frozenset({"HARVEST", "HOLD", "NO_ACTION"}),
    "REDUCE": frozenset({"REDUCE", "HOLD", "NO_ACTION"}),
    "CANCEL_PENDING": frozenset({"CANCEL_PENDING", "HOLD", "NO_ACTION"}),
    "NO_ACTION": frozenset({"NO_ACTION"}),
}
TUNING_ONLY_EVENT_TYPES = {"TECHNICAL_STATE_CHANGE", "TECHNICAL_INPUT_STALE"}
ENTRY_ACTION_HINTS = {"TRADE", "ADD"}
MAX_RETAINED_TECHNICAL_BASELINES = 64

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

# A bounded M1/M5/M15 guardian chart refresh should normally update on every
# closed M1 candle. Two M5 candles plus scheduling grace is the maximum age at
# which its categorical state may wake a discretionary tuning review.
DEFAULT_GUARDIAN_CHART_MAX_AGE_SECONDS = 12 * 60
TECHNICAL_FAMILY_MIN_SCORE = 0.35

CONTRACT_TRIGGER_BUCKETS = {
    "harvest_triggers": ("CONTRACT_HARVEST_TRIGGER", "HARVEST", "HARVEST_REVIEW", "P1", None),
    "add_triggers": ("CONTRACT_ADD_TRIGGER", "ADD", "ADD_REVIEW", "P1", "ALIVE"),
    "no_add_triggers": ("CONTRACT_NO_ADD_TRIGGER", "HOLD", "ADD_REVIEW", "P1", "WOUNDED"),
    "wounded_triggers": ("CONTRACT_WOUNDED_TRIGGER", "HOLD", "THESIS_REVIEW", "P1", "WOUNDED"),
    "invalidation_triggers": ("CONTRACT_INVALIDATION_TRIGGER", "REDUCE", "THESIS_REVIEW", "P0", "INVALIDATED"),
    "emergency_triggers": ("CONTRACT_EMERGENCY_TRIGGER", "REDUCE", "EMERGENCY_RISK_REVIEW", "P0", "EMERGENCY"),
}
CONTRACT_OWNERS = {"SYSTEM", "OPERATOR_MANUAL", "UNKNOWN"}
MARKET_READ_STATE_CHANGE_EVENTS = {
    "FAILED_ACCEPTANCE",
    "ACCEPTANCE_BREAK",
    "SESSION_EXPANSION",
    "RANGE_RAIL_TOUCH",
    "THEME_CONFIRMATION",
    "SQUEEZE_RELEASE",
}
MATERIAL_ACK_EVENT_TYPES = MARKET_READ_STATE_CHANGE_EVENTS | {"TECHNICAL_STATE_CHANGE"}
OPEN_POSITION_IDENTITY_KEYS = {
    "trade_id",
    "tradeid",
    "opentradeid",
    "position_id",
    "positionid",
    "open_trade_id",
    "open_position_id",
    "openpositionid",
}


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


def _event_requests_entry(event: GuardianEvent) -> bool:
    # Review type alone is not an entry request: CONTRACT_NO_ADD_TRIGGER uses
    # ADD_REVIEW/HOLD and must remain visible as safety evidence.  Any GPT
    # attempt to turn such a review into TRADE/ADD is rejected again by the
    # receipt and action-cycle stale-pair gates.
    return event.action_hint.upper() in ENTRY_ACTION_HINTS


def _technical_input_blocked_pairs(
    events: list[GuardianEvent],
    *,
    previous_state: dict[str, Any] | None = None,
) -> set[str]:
    """Return pairs whose latest rotating technical baseline is fail-closed.

    A pair omitted from the current bounded chart window remains blocked while
    its retained TECHNICAL_INPUT_STALE baseline is current.  A newly observed
    TECHNICAL_STATE_CHANGE explicitly clears that retained baseline.
    """

    fresh_pairs = {
        event.pair
        for event in events
        if event.event_type == "TECHNICAL_STATE_CHANGE" and event.pair
    }
    blocked = {
        event.pair
        for event in events
        if event.event_type == "TECHNICAL_INPUT_STALE" and event.pair
    }
    state = previous_state if isinstance(previous_state, dict) else {}
    state_events = state.get("events") if isinstance(state.get("events"), dict) else {}
    for prior in state_events.values():
        if not isinstance(prior, dict):
            continue
        if str(prior.get("event_type") or "").upper() != "TECHNICAL_INPUT_STALE":
            continue
        pair = _pair(prior.get("pair"))
        if pair and pair not in fresh_pairs:
            blocked.add(pair)
    return blocked


def _suppress_entry_events_for_technical_blocks(
    events: list[GuardianEvent],
    *,
    blocked_pairs: set[str],
) -> list[GuardianEvent]:
    if not blocked_pairs:
        return events
    return [
        event
        for event in events
        if not (event.pair in blocked_pairs and _event_requests_entry(event))
    ]


def run_guardian_event_router(
    *,
    snapshot_path: Path,
    pair_charts_path: Path,
    order_intents_path: Path,
    self_improvement_audit_path: Path | None = None,
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
    chart_freshness_path: Path | None = None,
    now: datetime | None = None,
) -> GuardianRouterSummary:
    now = _utc(now)
    pair_charts, pair_charts_sha256 = _load_json_with_sha256(pair_charts_path)
    inputs = {
        "snapshot": _load_json(snapshot_path),
        # Parse and hash one immutable byte snapshot.  Reading the JSON and
        # digest separately would let a concurrent replacement bind the
        # freshness receipt to different chart content than the router used.
        "pair_charts": pair_charts,
        "chart_freshness": _load_json(chart_freshness_path),
        "chart_freshness_required": chart_freshness_path is not None,
        "pair_charts_sha256": pair_charts_sha256,
        "technical_integrity_required": True,
        "order_intents": _load_json(order_intents_path),
        "self_improvement_audit": _load_json(self_improvement_audit_path),
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
        dispatcher_state=inputs["wake_dispatcher_state"],
        now=now,
    )
    blocked_pairs = {
        str(pair)
        for pair in escalation.get("technical_input_blocked_pairs", []) or []
        if str(pair)
    }
    events = _suppress_entry_events_for_technical_blocks(
        events,
        blocked_pairs=blocked_pairs,
    )

    events_payload = {
        "generated_at_utc": now.isoformat(),
        "schema_version": 1,
        "event_types": list(EVENT_TYPES),
        "events": [event.to_payload() for event in events],
        "trigger_contract": validate_guardian_trigger_contract(
            inputs["trigger_contract"],
            now=now,
            snapshot=inputs["snapshot"],
        ),
        "inputs": {
            "snapshot": str(snapshot_path),
            "pair_charts": str(pair_charts_path),
            "chart_freshness": str(chart_freshness_path)
            if chart_freshness_path is not None
            else None,
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
            previous_state=next_state,
            now=now,
        )
        if action_receipt_output_path is not None:
            written_action_receipt_path = action_receipt_output_path
            receipt_status = str(action_payload.get("receipt_status") or action_payload.get("status") or "").upper()
            if receipt_status == "ACCEPTED":
                reviewed_receipt = dict(action_payload)
                reviewed_receipt["router_review"] = action_review
                reviewed_receipt["router_review_status"] = action_review.get("status")
                reviewed_receipt["router_review_generated_at_utc"] = action_review.get("generated_at_utc")
                _write_json(action_receipt_output_path, reviewed_receipt)
            else:
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
    chart_freshness = (
        inputs.get("chart_freshness")
        if isinstance(inputs.get("chart_freshness"), dict)
        else {}
    )
    order_intents = inputs.get("order_intents") if isinstance(inputs.get("order_intents"), dict) else {}
    self_improvement_audit = (
        inputs.get("self_improvement_audit") if isinstance(inputs.get("self_improvement_audit"), dict) else {}
    )
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
    collected.extend(_self_improvement_pending_cancel_events(self_improvement_audit, snapshot=snapshot, now=now))
    collected.extend(_contract_events(trigger_contract, snapshot=snapshot, now=now))
    collected.extend(_wake_parse_failure_events(wake_dispatcher_state, now=now))
    collected.extend(
        _pair_chart_events(
            pair_charts,
            snapshot=snapshot,
            now=now,
            chart_freshness=chart_freshness,
            chart_freshness_required=inputs.get("chart_freshness_required") is True,
            pair_charts_sha256=str(inputs.get("pair_charts_sha256") or ""),
            technical_integrity_required=inputs.get("technical_integrity_required")
            is True,
        )
    )
    collected.extend(_order_intent_events(order_intents, now=now))
    collected.extend(_position_management_events(position_management, now=now))
    collected.extend(_thesis_evolution_events(thesis_evolution, now=now))
    collected.extend(_forecast_persistence_events(forecast_persistence, now=now))
    collected.extend(_market_context_matrix_events(market_context_matrix, now=now))
    deduped = _dedupe_events(collected)
    return _suppress_entry_events_for_technical_blocks(
        deduped,
        blocked_pairs=_technical_input_blocked_pairs(deduped),
    )


def evaluate_guardian_escalation(
    *,
    events: list[GuardianEvent],
    previous_state: dict[str, Any] | None,
    dispatcher_state: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = _utc(now)
    previous = previous_state if isinstance(previous_state, dict) else {}
    market_status = compute_market_status(now)
    previous_market_status = (
        previous.get("market_status")
        if isinstance(previous.get("market_status"), dict)
        else {}
    )
    previous_generated_at = _parse_utc(previous.get("generated_at_utc"))
    most_recent_open = _parse_utc(market_status.most_recent_open_utc)
    previous_has_events = isinstance(previous.get("events"), dict) and bool(
        previous.get("events")
    )
    market_reopened = bool(
        market_status.is_fx_open
        and (
            previous_market_status.get("is_fx_open") is False
            or (
                previous_generated_at is not None
                and most_recent_open is not None
                and previous_generated_at < most_recent_open
            )
            or (previous_has_events and previous_generated_at is None)
        )
    )
    previous_events = previous.get("events") if isinstance(previous.get("events"), dict) else {}
    technical_input_blocked_pairs = _technical_input_blocked_pairs(
        events,
        previous_state=previous,
    )
    events = _suppress_entry_events_for_technical_blocks(
        events,
        blocked_pairs=technical_input_blocked_pairs,
    )
    dispatcher = dispatcher_state if isinstance(dispatcher_state, dict) else {}
    reviewed_events = (
        dispatcher.get("reviewed_events")
        if isinstance(dispatcher.get("reviewed_events"), dict)
        else {}
    )
    review_events: list[dict[str, Any]] = []
    suppressed_events: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    wake_reason_codes: list[str] = []
    next_events: dict[str, Any] = {}

    for event in events:
        prior = previous_events.get(event.dedupe_key) if isinstance(previous_events, dict) else None
        if not isinstance(prior, dict):
            prior = None
        acknowledged = _dispatcher_acknowledged_event(reviewed_events.get(event.dedupe_key))
        material_acknowledged = (
            acknowledged
            if event.event_type in MATERIAL_ACK_EVENT_TYPES
            else None
        )
        reference = (
            _material_reference_event(prior, acknowledged=material_acknowledged)
            if event.event_type in MATERIAL_ACK_EVENT_TYPES
            else prior
        )
        reasons = _wake_reasons_for_event(event, reference)
        market_closed_observation_reasons: list[str] = []
        failed_acceptance_entry_watch = bool(
            event.event_type == "FAILED_ACCEPTANCE"
            and event.recommended_review_type == "ENTRY_REVIEW"
            and event.action_hint in {"TRADE", "HOLD"}
        )
        details = event.details if isinstance(event.details, dict) else {}
        fresh_entry_observation = bool(
            event.recommended_review_type == "ENTRY_REVIEW"
            and event.action_hint in {"TRADE", "ADD"}
            and not _event_details_have_open_position_identity(details)
        )
        candidate_harvest_watch = bool(
            event.event_type == "HARVEST_ZONE"
            and event.recommended_review_type == "HARVEST_REVIEW"
            and event.action_hint == "HARVEST"
            and str(details.get("lane_id") or "").strip()
            and str(details.get("status") or "").strip()
            and not str(details.get("trade_id") or "").strip()
        )
        market_closed_observation_suppressed = bool(
            (
                event.event_type in TUNING_ONLY_EVENT_TYPES
                or failed_acceptance_entry_watch
                or fresh_entry_observation
                or candidate_harvest_watch
            )
            and not market_status.is_fx_open
        )
        if market_closed_observation_suppressed:
            # A closed weekly market cannot produce a new tradable candle or
            # executable failed-acceptance entry observation. Keep the event,
            # state, and report for manual/safety monitoring, but never spend
            # GPT or mint a tuning obligation from regenerated/stale weekend
            # inputs. Safety actions remain outside this observation-only gate.
            market_closed_observation_reasons = list(reasons)
            reasons = []
        elif market_reopened and (
            event.event_type == "TECHNICAL_INPUT_STALE"
            or failed_acceptance_entry_watch
            or fresh_entry_observation
            or candidate_harvest_watch
        ):
            # Weekend staleness and entry watches are expected to be silent
            # while the market is closed, but the same still-present condition
            # becomes actionable once the market reopens. NEW_EVENT keeps the
            # existing dispatcher contract; the market-reopen reason preserves
            # why this otherwise unchanged event is being reviewed again.
            reopen_reason = (
                "MARKET_REOPEN_TECHNICAL_INPUT_STILL_STALE"
                if event.event_type == "TECHNICAL_INPUT_STALE"
                else "MARKET_REOPEN_CANDIDATE_HARVEST_WATCH"
                if candidate_harvest_watch
                else "MARKET_REOPEN_FAILED_ACCEPTANCE_WATCH"
                if failed_acceptance_entry_watch
                else "MARKET_REOPEN_ENTRY_OBSERVATION"
            )
            reasons = list(dict.fromkeys([*reasons, "NEW_EVENT", reopen_reason]))
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
                diagnostics.extend(_suppressed_price_change_diagnostics(event, reference))
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
                diagnostics.extend(_suppressed_price_change_diagnostics(event, reference))
                suppressed_payload = {
                    **event_payload,
                    "suppressed_reason": (
                        "MARKET_CLOSED_FAILED_ACCEPTANCE_WATCH"
                        if (
                            market_closed_observation_suppressed
                            and failed_acceptance_entry_watch
                        )
                        else "MARKET_CLOSED_CANDIDATE_HARVEST_WATCH"
                        if (
                            market_closed_observation_suppressed
                            and candidate_harvest_watch
                        )
                        else "MARKET_CLOSED_ENTRY_OBSERVATION"
                        if (
                            market_closed_observation_suppressed
                            and fresh_entry_observation
                        )
                        else "MARKET_CLOSED_TUNING_OBSERVATION"
                        if market_closed_observation_suppressed
                        else "THROTTLED"
                        if repeated_in_throttle
                        else "NO_STATE_CHANGE"
                    ),
                    "throttle_seconds": throttle_seconds,
                    "last_wake_at_utc": last_wake_at,
                }
                if market_closed_observation_suppressed:
                    suppressed_payload["suppressed_wake_reason_codes"] = (
                        market_closed_observation_reasons
                    )
                suppressed_events.append(suppressed_payload)
        material_reference = (
            material_acknowledged
            or ((prior or {}).get("material_reference") if isinstance((prior or {}).get("material_reference"), dict) else None)
            or _event_state_snapshot(event)
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
            "price_zone": event.price_zone,
            "details": event.details,
            "in_harvest_zone": event.event_type == "HARVEST_ZONE",
            "margin_pressure": event.event_type == "MARGIN_PRESSURE",
            "last_seen_at_utc": now.isoformat(),
            "last_wake_at_utc": last_wake_at,
            "material_reference": material_reference,
            "acknowledged_event": material_acknowledged or (prior or {}).get("acknowledged_event"),
        }

    # The fast guardian deliberately observes only a rotating subset of the
    # 28-pair G8 universe on each M1 window.  Dropping an omitted pair's last
    # technical baseline would make its next visit look like ``NEW_EVENT``
    # even when the closed-candle watermark and derived state did not change.
    # Retain at most one bounded baseline per pair and technical event type;
    # this also preserves a stale-input baseline across rotation. Current
    # events replace their own key normally, while non-technical conditions
    # keep their existing current-cycle lifecycle. The cap also keeps a
    # malformed or historically expanded state from growing without bound.
    retained_technical_keys: set[tuple[str, str]] = set()
    retained_count = 0
    current_technical_types_by_pair: dict[str, set[str]] = {}
    for current in next_events.values():
        if not isinstance(current, dict):
            continue
        current_type = str(current.get("event_type") or "").upper()
        current_pair = _pair(current.get("pair"))
        if current_pair and current_type in TUNING_ONLY_EVENT_TYPES:
            retained_technical_keys.add((current_pair, current_type))
            current_technical_types_by_pair.setdefault(current_pair, set()).add(
                current_type
            )
    ordered_previous = sorted(
        previous_events.items(),
        key=lambda item: (
            _parse_utc((item[1] or {}).get("last_seen_at_utc"))
            if isinstance(item[1], dict)
            else None
        )
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    for dedupe_key, prior in ordered_previous:
        if dedupe_key in next_events or not isinstance(prior, dict):
            continue
        event_type = str(prior.get("event_type") or "").upper()
        if event_type not in TUNING_ONLY_EVENT_TYPES:
            continue
        pair = _pair(prior.get("pair"))
        baseline_key = (pair or "", event_type)
        if not pair or baseline_key in retained_technical_keys:
            continue
        if (
            event_type == "TECHNICAL_INPUT_STALE"
            and "TECHNICAL_STATE_CHANGE"
            in current_technical_types_by_pair.get(pair, set())
        ):
            # This pair was actively observed with complete technical input,
            # so the stale condition genuinely cleared. Do not confuse that
            # lifecycle transition with a pair omitted by round-robin scope.
            continue
        if retained_count >= MAX_RETAINED_TECHNICAL_BASELINES:
            continue
        retained = dict(prior)
        retained["baseline_retained_out_of_current_event_set"] = True
        next_events[str(dedupe_key)] = retained
        retained_technical_keys.add(baseline_key)
        retained_count += 1

    escalation = {
        "generated_at_utc": now.isoformat(),
        "wake_gpt": bool(review_events),
        "model_target": "GPT-5.5",
        "wake_policy": "state_change_only",
        "market_status": market_status.to_dict(),
        "execution_boundary": {
            "guardian_never_trades": True,
            "gpt_wake_never_calls_oanda_directly": True,
            "live_order_gateway_required": True,
        },
        "wake_reason_codes": sorted(set(wake_reason_codes)),
        "technical_input_blocked_pairs": sorted(technical_input_blocked_pairs),
        "events_to_review": review_events,
        "suppressed_events": suppressed_events,
        "diagnostics": diagnostics,
        "throttle": {
            "default_seconds": DEFAULT_THROTTLE_SECONDS,
            "fresh_entry_or_add_seconds": DEFAULT_FRESH_ACTION_THROTTLE_SECONDS,
            "p0_and_harvest_bypass_on_state_change": True,
        },
    }
    next_state = {
        "generated_at_utc": now.isoformat(),
        "market_status": market_status.to_dict(),
        "events": next_events,
    }
    return escalation, next_state


def review_guardian_action_receipt(
    receipt_payload: dict[str, Any] | None,
    *,
    events: list[GuardianEvent],
    previous_state: dict[str, Any] | None = None,
    selected_event: dict[str, Any] | GuardianEvent | None = None,
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

    binding_issues = _selected_event_binding_issues(receipt, selected_event)
    issues.extend(binding_issues)
    if event is not None:
        issues.extend(
            guardian_event_action_binding_issues(
                receipt_action=action,
                event_action_hint=event.action_hint,
            )
        )

    if action in {"TRADE", "ADD"}:
        issues.extend(_guardian_trade_add_action_issues(receipt, event=event))
        canonical_pair = event.pair if event is not None else _pair(receipt.get("pair"))
        if canonical_pair in _technical_input_blocked_pairs(
            events,
            previous_state=previous_state,
        ):
            issues.append(
                _issue(
                    "GUARDIAN_ACTION_TECHNICAL_INPUT_STALE",
                    f"{canonical_pair} technical input is fail-closed; TRADE/ADD is forbidden",
                )
            )
    if event is not None and event.event_type in TUNING_ONLY_EVENT_TYPES and action not in {"HOLD", "NO_ACTION"}:
        issues.append(
            _issue(
                "GUARDIAN_TUNING_EVENT_LIVE_ACTION_FORBIDDEN",
                f"{event.event_type} is monitoring/tuning evidence only; action must be HOLD or NO_ACTION",
            )
        )

    if binding_issues:
        status = "RECEIPT_EVENT_MISMATCH"
    else:
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


def guardian_event_action_binding_issues(
    *,
    receipt_action: Any,
    event_action_hint: Any,
) -> list[dict[str, str]]:
    """Allow a receipt to keep or safely downgrade, never upgrade, an event action."""

    action = str(receipt_action or "").strip().upper()
    hint = str(event_action_hint or "").strip().upper()
    allowed = EVENT_ACTION_RECEIPT_EQUIVALENCE.get(hint)
    if allowed is None or action not in allowed:
        return [
            _issue(
                "GUARDIAN_ACTION_EVENT_ACTION_MISMATCH",
                f"selected event action_hint={hint or 'missing'} allows only "
                f"{sorted(allowed or ())}; receipt action={action or 'missing'}",
            )
        ]
    return []


def _selected_event_binding_issues(
    receipt: dict[str, Any],
    selected_event: dict[str, Any] | GuardianEvent | None,
) -> list[dict[str, str]]:
    selected = _event_payload(selected_event)
    if not selected:
        return []
    issues: list[dict[str, str]] = []
    receipt_event_id = str(receipt.get("event_id") or "").strip()
    selected_event_id = str(selected.get("event_id") or "").strip()
    if selected_event_id and receipt_event_id != selected_event_id:
        issues.append(
            _issue(
                "RECEIPT_EVENT_MISMATCH",
                f"receipt event_id {receipt_event_id or 'missing'} does not match selected_event {selected_event_id}",
            )
        )
    receipt_pair = _pair(receipt.get("pair"))
    selected_pair = _pair(selected.get("pair"))
    if selected_pair and receipt_pair != selected_pair:
        issues.append(
            _issue(
                "RECEIPT_EVENT_MISMATCH",
                f"receipt pair {receipt_pair or 'missing'} does not match selected_event {selected_pair}",
            )
        )
    selected_direction = _direction_from_text(selected.get("direction") or selected.get("side"))
    receipt_side = _direction_from_text(receipt.get("side") or receipt.get("direction"))
    if selected_direction is not None and receipt_side != selected_direction:
        issues.append(
            _issue(
                "RECEIPT_EVENT_MISMATCH",
                f"receipt side {receipt_side or 'missing'} does not match selected_event direction {selected_direction}",
            )
        )
    receipt_dedupe = str(receipt.get("dedupe_key") or "").strip()
    selected_dedupe = str(selected.get("dedupe_key") or "").strip()
    if receipt_dedupe and selected_dedupe and receipt_dedupe != selected_dedupe:
        issues.append(
            _issue(
                "RECEIPT_EVENT_MISMATCH",
                "receipt dedupe_key does not match selected_event dedupe_key",
            )
        )
    return issues


def _event_payload(event: dict[str, Any] | GuardianEvent | None) -> dict[str, Any]:
    if isinstance(event, GuardianEvent):
        return event.to_payload()
    return event if isinstance(event, dict) else {}


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
    receipt_status = str(payload.get("receipt_status") or payload.get("status") or "").upper()
    receipt_lifecycle = str(payload.get("receipt_lifecycle") or ("ACTIVE" if receipt_status == "ACCEPTED" else "")).upper()
    if receipt_status and receipt_status != "ACCEPTED":
        issues.append(_issue("GUARDIAN_ACTION_RECEIPT_NOT_ACCEPTED", "guardian action receipt_status must be ACCEPTED"))
    if receipt_lifecycle and receipt_lifecycle != "ACTIVE":
        issues.append(
            _issue(
                "GUARDIAN_ACTION_RECEIPT_NOT_ACTIVE",
                f"guardian action receipt_lifecycle is {receipt_lifecycle}",
            )
        )
    expires_at = _parse_utc(payload.get("expires_at_utc"))
    if expires_at is not None and expires_at <= datetime.now(timezone.utc):
        issues.append(_issue("GUARDIAN_ACTION_RECEIPT_EXPIRED", "guardian action receipt is past expires_at_utc"))
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
    snapshot: dict[str, Any] | None = None,
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

    stale_relevant_deadline_expired = False
    open_entry_keys = set()
    for index, entry in enumerate(entries):
        prefix = f"entries[{index}]"
        open_exposure = _contract_entry_has_open_exposure(entry)
        live_relevant = _contract_entry_is_live_relevant(entry)
        all_core_trigger_arrays_empty = all(
            isinstance(entry.get(bucket), list) and not entry.get(bucket)
            for bucket in (
                "harvest_triggers",
                "no_add_triggers",
                "wounded_triggers",
                "invalidation_triggers",
                "emergency_triggers",
            )
        )
        watch_only = _contract_entry_watch_only_reason(entry)
        if open_exposure:
            for key in _contract_open_exposure_keys(entry):
                open_entry_keys.add(key)
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
        if thesis_state not in {"ALIVE", "WOUNDED", "INVALIDATED", "EMERGENCY", "UNKNOWN"}:
            issues.append(_issue("CONTRACT_ENTRY_BAD_THESIS_STATE", f"{prefix} thesis_state is unsupported"))
        deadline = _parse_utc(entry.get("next_review_deadline_utc"))
        if deadline is None:
            issues.append(
                _issue("CONTRACT_ENTRY_BAD_DEADLINE", f"{prefix} next_review_deadline_utc must be ISO UTC")
            )
        elif deadline <= clock:
            if live_relevant:
                stale_relevant_deadline_expired = True
                issues.append(_issue("CONTRACT_ENTRY_DEADLINE_EXPIRED", f"{prefix} next_review_deadline_utc is expired"))
            else:
                issues.append(
                    _issue(
                        "CONTRACT_ENTRY_WATCH_DEADLINE_EXPIRED",
                        f"{prefix} watch-only candidate next_review_deadline_utc is expired",
                        severity="WARN",
                    )
                )
        for bucket in CONTRACT_TRIGGER_BUCKETS:
            if bucket not in entry:
                issues.append(_issue("CONTRACT_ENTRY_TRIGGER_FIELD_MISSING", f"{prefix} missing {bucket}"))
            elif not isinstance(entry.get(bucket), list):
                issues.append(_issue("CONTRACT_ENTRY_TRIGGER_FIELD_NOT_LIST", f"{prefix} {bucket} must be a list"))
            else:
                for trigger_index, trigger in enumerate(entry.get(bucket) or []):
                    if isinstance(trigger, dict):
                        issues.extend(
                            _trigger_parent_binding_issues(
                                entry=entry,
                                trigger=trigger,
                                bucket=bucket,
                                entry_index=index,
                                trigger_index=trigger_index,
                                open_exposure=open_exposure,
                            )
                        )
        if open_exposure:
            for bucket in (
                "harvest_triggers",
                "no_add_triggers",
                "wounded_triggers",
                "invalidation_triggers",
                "emergency_triggers",
            ):
                if isinstance(entry.get(bucket), list) and not entry.get(bucket):
                    issues.append(
                        _issue(
                            "CONTRACT_ENTRY_OPEN_TRIGGER_EMPTY",
                            f"{prefix} open exposure needs non-empty {bucket}",
                        )
                    )
        elif all_core_trigger_arrays_empty:
            if _truthy(entry.get("watch_only")):
                if not watch_only:
                    issues.append(
                        _issue(
                            "WATCH_ONLY_NO_TRIGGER_CONTRACT",
                            (
                                f"{prefix} pair={pair or 'UNKNOWN'} side={side or 'UNKNOWN'} watch_only candidate "
                                "needs watch_only_reason when omitting triggers"
                            ),
                            severity="BLOCK" if live_relevant else "WARN",
                        )
                    )
            elif live_relevant:
                issues.append(
                    _issue(
                        "TRIGGER_CONTRACT_EMPTY_FOR_ACTIVE_PAIR",
                        (
                            f"{prefix} pair={pair or 'UNKNOWN'} side={side or 'UNKNOWN'} selected/current candidate "
                            "has no machine-readable guardian triggers and is not watch_only"
                        ),
                    )
                )
            else:
                issues.append(
                    _issue(
                        "TRIGGER_CONTRACT_EMPTY_FOR_ACTIVE_PAIR",
                        (
                            f"{prefix} pair={pair or 'UNKNOWN'} side={side or 'UNKNOWN'} candidate has empty "
                            "guardian triggers and is not explicitly watch_only"
                        ),
                        severity="WARN",
                    )
                )

    snapshot_payload = snapshot if isinstance(snapshot, dict) else {}
    if snapshot_payload:
        for exposure in _snapshot_open_exposure_contract_refs(snapshot_payload):
            if not exposure["keys"].intersection(open_entry_keys):
                issues.append(
                    _issue(
                        "CONTRACT_OPEN_EXPOSURE_MISSING",
                        (
                            "broker open exposure is missing from guardian trigger contract "
                            f"pair={exposure.get('pair')} side={exposure.get('side')} trade_id={exposure.get('trade_id') or 'none'}"
                        ),
                    )
                )

    age_seconds = (clock - generated_at).total_seconds() if generated_at is not None else None
    max_age = int(os.environ.get("QR_GUARDIAN_TRIGGER_CONTRACT_MAX_AGE_SECONDS", DEFAULT_CONTRACT_MAX_AGE_SECONDS))
    stale = age_seconds is None or age_seconds > max_age or stale_relevant_deadline_expired
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


def _trigger_parent_binding_issues(
    *,
    entry: dict[str, Any],
    trigger: dict[str, Any],
    bucket: str,
    entry_index: int,
    trigger_index: int,
    open_exposure: bool,
) -> list[dict[str, str]]:
    prefix = f"entries[{entry_index}].{bucket}[{trigger_index}]"
    severity = "BLOCK" if open_exposure else "WARN"
    issues: list[dict[str, str]] = []
    parent_trade_id = str(entry.get("trade_id") or "").strip()
    trigger_trade_id = str(trigger.get("trade_id") or "").strip()
    if parent_trade_id and trigger_trade_id and trigger_trade_id != parent_trade_id:
        issues.append(
            _issue(
                "CONTRACT_TRIGGER_TRADE_ID_MISMATCH",
                f"{prefix} trade_id {trigger_trade_id} differs from parent {parent_trade_id}",
                severity="BLOCK" if open_exposure else "WARN",
            )
        )
    parent_units = _float(entry.get("units"))
    trigger_units = _float(trigger.get("units"))
    if parent_units is not None and trigger_units is not None and abs(parent_units - trigger_units) > 1e-9:
        issues.append(
            _issue(
                "CONTRACT_TRIGGER_UNITS_MISMATCH",
                f"{prefix} units {trigger_units} differs from parent {parent_units}",
                severity=severity,
            )
        )
    parent_pair = _pair(entry.get("pair"))
    trigger_pair = _pair(trigger.get("pair"))
    if parent_pair and trigger_pair and trigger_pair != parent_pair:
        issues.append(
            _issue(
                "CONTRACT_TRIGGER_PAIR_MISMATCH",
                f"{prefix} pair {trigger_pair} differs from parent {parent_pair}",
                severity=severity,
            )
        )
    parent_side = _direction_from_text(entry.get("side"))
    trigger_side = _direction_from_text(trigger.get("side"))
    if parent_side and trigger_side and trigger_side != parent_side:
        issues.append(
            _issue(
                "CONTRACT_TRIGGER_SIDE_MISMATCH",
                f"{prefix} side {trigger_side} differs from parent {parent_side}",
                severity=severity,
            )
        )
    parent_owner = _contract_owner(entry.get("owner"))
    trigger_owner = _contract_owner(trigger.get("owner")) if str(trigger.get("owner") or "").strip() else ""
    if parent_owner and trigger_owner and trigger_owner != parent_owner:
        issues.append(
            _issue(
                "CONTRACT_TRIGGER_OWNER_MISMATCH",
                f"{prefix} owner {trigger_owner} differs from parent {parent_owner}",
                severity=severity,
            )
        )
    parent_avg = _float(entry.get("avg_entry"))
    trigger_avg = _float(trigger.get("avg_entry"))
    if parent_avg is not None and trigger_avg is not None and _material_float_differs(parent_avg, trigger_avg):
        issues.append(
            _issue(
                "CONTRACT_TRIGGER_AVG_ENTRY_MISMATCH",
                f"{prefix} avg_entry {trigger_avg} differs from parent {parent_avg}",
                severity=severity,
            )
        )
    return issues


def _material_float_differs(left: float, right: float) -> bool:
    tolerance = max(1e-9, abs(left) * 1e-6)
    return abs(left - right) > tolerance


def build_guardian_trigger_contract(
    *,
    snapshot: dict[str, Any],
    order_intents: dict[str, Any],
    existing_contract: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
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
    selected_lane_ids = _selected_order_intent_lane_ids(order_intents)

    for position in snapshot.get("positions", []) or []:
        if not isinstance(position, dict):
            continue
        pair = _pair(position.get("pair"))
        side = _direction_from_position(position)
        if not pair or not side:
            continue
        thesis = _position_contract_thesis(position)
        ownership_audit = _position_ownership_audit(position)
        entry = _contract_entry_from_seed(
            pair=pair,
            side=side,
            thesis=thesis,
            owner=_contract_owner_from_ownership_audit(ownership_audit),
            thesis_state=_position_contract_state(position),
            seed_reason="open broker position requires guardian triggers",
            preserved=preserved,
            now=clock,
            position=position,
            open_exposure=True,
            ownership_audit=ownership_audit,
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
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        entry = _contract_entry_from_seed(
            pair=pair,
            side=side,
            thesis=thesis,
            owner="SYSTEM",
            thesis_state=str(metadata.get("thesis_state") or "ALIVE").upper(),
            seed_reason=f"candidate {result.get('lane_id') or pair} requires trader-defined triggers",
            preserved=preserved,
            now=clock,
            position={},
            open_exposure=False,
            ownership_audit={"status": "SYSTEM", "evidence": ["candidate intent is system-generated"], "unresolved": False},
        )
        lane_id = str(result.get("lane_id") or "").strip()
        status = str(result.get("status") or "").strip().upper()
        if lane_id:
            entry["lane_id"] = lane_id
        if status:
            entry["status"] = status
        if lane_id and lane_id in selected_lane_ids:
            entry["selected"] = True
            entry["current"] = True
        if _truthy(result.get("selected")) or _truthy(result.get("current")) or _truthy(result.get("selected_for_review")):
            entry["selected"] = _truthy(result.get("selected")) or _truthy(result.get("selected_for_review"))
            entry["current"] = _truthy(result.get("current")) or bool(entry.get("current"))
        watch_only_reason = _candidate_watch_only_reason(result, intent, thesis)
        if watch_only_reason:
            entry["watch_only"] = True
            entry["watch_only_reason"] = watch_only_reason
            entry["thesis_state"] = "UNKNOWN"
            entry["next_review_reason"] = f"watch-only candidate: {watch_only_reason}"
        elif _contract_core_trigger_arrays_empty(entry):
            entry.update(_default_candidate_triggers(entry, result, intent, now=clock))
        key = _contract_entry_key(entry)
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)

    _merge_range_rail_watch_entry(
        entries=entries,
        seen=seen,
        watch_entry=_range_rail_watch_contract_entry(range_rail_geometry_repair, now=clock),
    )

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
    unresolved_unknown: list[dict[str, Any]] = []
    for entry in _contract_entries(contract):
        audit = entry.get("ownership_audit") if isinstance(entry.get("ownership_audit"), dict) else {}
        if entry.get("owner") == "UNKNOWN" or audit.get("status") == "UNKNOWN_NEEDS_OPERATOR_CONFIRM":
            unresolved_unknown.append(entry)
        lines.append(
            f"- `{_pair(entry.get('pair')) or 'UNKNOWN'}` `{_direction_from_text(entry.get('side')) or 'UNKNOWN'}` "
            f"owner=`{_contract_owner(entry.get('owner'))}` state=`{_thesis_state(entry.get('thesis_state'))}` "
            f"trade_id=`{entry.get('trade_id') or 'none'}` units=`{entry.get('units')}` avg_entry=`{entry.get('avg_entry')}`"
        )
        lines.append(f"  - thesis: {entry.get('thesis')}")
        if audit:
            lines.append(f"  - ownership audit: `{audit.get('status')}` evidence=`{'; '.join(audit.get('evidence') or [])}`")
        lines.append(f"  - next review: {entry.get('next_review_reason')} by `{entry.get('next_review_deadline_utc')}`")
        for bucket in CONTRACT_TRIGGER_BUCKETS:
            triggers = entry.get(bucket) if isinstance(entry.get(bucket), list) else []
            fired_count = sum(1 for trigger in triggers if _trigger_explicitly_fired(trigger))
            lines.append(f"  - {bucket}: `{len(triggers)}` declared, `{fired_count}` explicitly fired")
    if not _contract_entries(contract):
        lines.append("- none")
    lines.extend(["", "## Ownership Audit", ""])
    if unresolved_unknown:
        lines.append("- Unresolved UNKNOWN exposure requires operator confirmation before being treated as OPERATOR_MANUAL:")
        for entry in unresolved_unknown:
            lines.append(
                f"  - `{entry.get('pair')}` `{entry.get('side')}` trade_id=`{entry.get('trade_id') or 'none'}` "
                f"units=`{entry.get('units')}` avg_entry=`{entry.get('avg_entry')}` thesis=`{entry.get('thesis')}` "
                "status=`UNKNOWN_NEEDS_OPERATOR_CONFIRM`"
            )
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _contract_events(contract: dict[str, Any], *, snapshot: dict[str, Any], now: datetime) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    validation = validate_guardian_trigger_contract(contract, now=now, snapshot=snapshot)
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
    for issue in validation.get("issues", []) or []:
        code = str(issue.get("code") or "").upper()
        if code not in {"TRIGGER_CONTRACT_EMPTY_FOR_ACTIVE_PAIR", "WATCH_ONLY_NO_TRIGGER_CONTRACT"}:
            continue
        events.append(
            _event(
                event_type=code,
                pair=_issue_pair(issue) or "PORTFOLIO",
                direction=None,
                thesis="guardian trigger contract quality",
                price_zone=str(issue.get("message") or code),
                severity="P1" if str(issue.get("severity") or "").upper() == "BLOCK" else "P2",
                recommended_review_type="THESIS_REVIEW",
                action_hint="HOLD",
                thesis_state="WOUNDED",
                now=now,
                details={"validation_issue": issue},
            )
        )

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
    margin_positions = [
        position
        for position in snapshot.get("positions", []) or []
        if isinstance(position, dict)
    ]
    margin_event = _margin_pressure_event(
        account,
        positions=margin_positions,
        now=now,
    )
    if margin_event is not None:
        events.append(margin_event)
    events.extend(_spread_anomaly_events(quotes, now=now))
    events.extend(_quote_major_figure_events(quotes, now=now))
    return events


def _self_improvement_pending_cancel_events(
    self_improvement_audit: dict[str, Any],
    *,
    snapshot: dict[str, Any],
    now: datetime,
) -> list[GuardianEvent]:
    findings = self_improvement_audit.get("findings") if isinstance(self_improvement_audit, dict) else []
    if not isinstance(findings, list):
        return []
    current_pending_by_id = _current_trader_pending_orders_by_id(snapshot)
    if not current_pending_by_id:
        return []

    events: list[GuardianEvent] = []
    for finding in findings:
        if not isinstance(finding, dict) or finding.get("code") != "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED":
            continue
        evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
        audit_orders = {
            str(order.get("order_id") or ""): order
            for order in evidence.get("orders", []) or []
            if isinstance(order, dict) and str(order.get("order_id") or "")
        }
        order_ids = [
            str(order_id)
            for order_id in evidence.get("cancel_review_order_ids", []) or []
            if str(order_id)
        ]
        if not order_ids:
            order_ids = list(audit_orders)
        for order_id in dict.fromkeys(order_ids):
            current_order = current_pending_by_id.get(order_id)
            if current_order is None:
                continue
            audit_order = audit_orders.get(order_id) or {}
            pair = _pair(current_order.get("pair") or current_order.get("instrument") or audit_order.get("pair"))
            if not pair:
                continue
            price = current_order.get("price")
            if price is None:
                price = audit_order.get("price")
            parent_lane_id = audit_order.get("parent_lane_id") or audit_order.get("lane_id")
            thesis = _pending_order_thesis(current_order)
            if thesis.startswith("pending order") and parent_lane_id:
                thesis = str(parent_lane_id)
            events.append(
                _event(
                    event_type="STALE_PENDING",
                    pair=pair,
                    direction=_direction_from_units(current_order.get("units") or audit_order.get("units")),
                    thesis=thesis,
                    price_zone=f"pending price={price} self_improvement_cancel_review",
                    severity="P1",
                    recommended_review_type="PENDING_CANCEL_REVIEW",
                    action_hint="CANCEL_PENDING",
                    now=now,
                    details={
                        "order_id": order_id,
                        "state": current_order.get("state"),
                        "source": "self_improvement_audit",
                        "finding_code": "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                        "parent_lane_id": parent_lane_id,
                        "current_candidate_count": audit_order.get("current_candidate_count"),
                        "current_live_ready_candidate_count": audit_order.get("current_live_ready_candidate_count"),
                        "review_reasons": audit_order.get("review_reasons") or [],
                    },
                )
            )
    return events


def _current_trader_pending_orders_by_id(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    orders: dict[str, dict[str, Any]] = {}
    for order in snapshot.get("orders", []) or []:
        if not isinstance(order, dict):
            continue
        order_id = str(order.get("order_id") or "")
        if not order_id or _owner(order.get("owner")) != Owner.TRADER.value:
            continue
        state = str(order.get("state") or "").upper()
        if state and state not in {"PENDING", "LIVE", "OPEN"}:
            continue
        if not _pair(order.get("pair") or order.get("instrument")):
            continue
        orders[order_id] = order
    return orders


def _pair_chart_events(
    pair_charts: dict[str, Any],
    *,
    snapshot: dict[str, Any],
    now: datetime,
    chart_freshness: dict[str, Any] | None = None,
    chart_freshness_required: bool = False,
    pair_charts_sha256: str = "",
    technical_integrity_required: bool = False,
) -> list[GuardianEvent]:
    events: list[GuardianEvent] = []
    chart_rows = _chart_rows(pair_charts)
    charts_by_pair = {
        pair: chart
        for chart in chart_rows
        if (pair := _pair(chart.get("pair"))) is not None
    }
    monitor_pairs = _guardian_monitor_pairs(pair_charts, snapshot=snapshot)
    freshness = _guardian_chart_freshness(pair_charts, now=now)
    if monitor_pairs and not freshness["fresh"]:
        for pair in monitor_pairs:
            events.append(
                _event(
                    event_type="TECHNICAL_INPUT_STALE",
                    pair=pair,
                    direction=None,
                    thesis="bounded guardian chart input freshness",
                    price_zone="guardian technical input is stale",
                    severity="P1" if _pair_has_open_exposure(snapshot, pair) else "P2",
                    recommended_review_type="TUNING_REVIEW",
                    action_hint="HOLD" if _pair_has_open_exposure(snapshot, pair) else "NO_ACTION",
                    now=now,
                    details={
                        **freshness,
                        "monitor_scope": _guardian_monitor_scope(pair_charts, pair),
                        "live_permission_allowed": False,
                        "no_direct_oanda": True,
                    },
                )
            )
        return events

    for pair in monitor_pairs:
        chart = charts_by_pair.get(pair)
        freshness_issue = _guardian_pair_freshness_issue(
            pair,
            pair_charts=pair_charts,
            chart_freshness=chart_freshness,
            chart_freshness_required=chart_freshness_required,
            pair_charts_sha256=pair_charts_sha256,
            now=now,
        )
        if freshness_issue is not None:
            events.append(
                _event(
                    event_type="TECHNICAL_INPUT_STALE",
                    pair=pair,
                    direction=None,
                    thesis="bounded guardian chart freshness receipt",
                    price_zone="guardian per-pair technical freshness is fail-closed",
                    severity="P1" if _pair_has_open_exposure(snapshot, pair) else "P2",
                    recommended_review_type="TUNING_REVIEW",
                    action_hint="HOLD" if _pair_has_open_exposure(snapshot, pair) else "NO_ACTION",
                    now=now,
                    details={
                        "fresh": False,
                        "status": "GUARDIAN_CHART_FRESHNESS_BLOCKED",
                        "freshness_issue": freshness_issue,
                        "monitor_scope": _guardian_monitor_scope(pair_charts, pair),
                        "live_permission_allowed": False,
                        "no_direct_oanda": True,
                        "preserve_blockers": True,
                    },
                )
            )
            continue
        if chart is None:
            events.append(
                _event(
                    event_type="TECHNICAL_INPUT_STALE",
                    pair=pair,
                    direction=None,
                    thesis="bounded guardian chart input freshness",
                    price_zone="guardian technical input is missing for monitored pair",
                    severity="P1" if _pair_has_open_exposure(snapshot, pair) else "P2",
                    recommended_review_type="TUNING_REVIEW",
                    action_hint="HOLD" if _pair_has_open_exposure(snapshot, pair) else "NO_ACTION",
                    now=now,
                    details={
                        "fresh": False,
                        "status": "PAIR_CHART_MISSING",
                        "monitor_scope": _guardian_monitor_scope(pair_charts, pair),
                        "live_permission_allowed": False,
                        "no_direct_oanda": True,
                    },
                )
            )
            continue
        technical_event = _technical_state_event(
            pair,
            chart,
            pair_charts=pair_charts,
            snapshot=snapshot,
            now=now,
            technical_integrity_required=technical_integrity_required,
        )
        if technical_event is not None:
            events.append(technical_event)

    for chart in chart_rows:
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


def _guardian_pair_freshness_issue(
    pair: str,
    *,
    pair_charts: dict[str, Any],
    chart_freshness: dict[str, Any] | None,
    chart_freshness_required: bool,
    pair_charts_sha256: str,
    now: datetime,
) -> dict[str, Any] | None:
    if not chart_freshness_required:
        return None
    payload = chart_freshness if isinstance(chart_freshness, dict) else {}
    checked = _parse_aware_utc(payload.get("checked_at_utc"))
    source_generated = _parse_aware_utc(payload.get("source_generated_at_utc"))
    chart_generated = _parse_aware_utc(pair_charts.get("generated_at_utc"))
    expected_sha = str(payload.get("source_pair_charts_sha256") or "").lower()
    max_age = int(
        os.environ.get(
            "QR_GUARDIAN_CHART_MAX_AGE_SECONDS",
            DEFAULT_GUARDIAN_CHART_MAX_AGE_SECONDS,
        )
    )
    reasons: list[str] = []
    if checked is None:
        reasons.append("INVALID_CHECKED_AT_UTC")
    else:
        age = (now - checked).total_seconds()
        if age < -5:
            reasons.append("FUTURE_CHECKED_AT_UTC")
        elif age > max_age:
            reasons.append("STALE_CHECKED_AT_UTC")
    if source_generated is None or chart_generated is None or source_generated != chart_generated:
        reasons.append("PAIR_CHART_GENERATED_AT_BINDING_MISMATCH")
    if (
        len(expected_sha) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha)
        or expected_sha != pair_charts_sha256.lower()
    ):
        reasons.append("PAIR_CHART_SHA256_BINDING_MISMATCH")
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else None
    pair_rows: dict[str, list[dict[str, Any]]] = {}
    if rows is None:
        reasons.append("FRESHNESS_ROWS_MISSING")
    else:
        for row in rows:
            if not isinstance(row, dict) or _pair(row.get("pair")) != pair:
                continue
            timeframe = str(row.get("timeframe") or "").upper()
            pair_rows.setdefault(timeframe, []).append(row)
        allowed = {"FRESH", "TECHNICAL_INPUT_BLOCKED_CURRENT"}
        expected_max_ages = {"M1": 120.0, "M5": 600.0, "M15": 1800.0}
        for timeframe in ("M1", "M5", "M15"):
            matches = pair_rows.get(timeframe, [])
            if len(matches) != 1:
                reasons.append(f"{timeframe}_FRESHNESS_ROW_COUNT_{len(matches)}")
                continue
            status = str(matches[0].get("status") or "").upper()
            if status not in allowed:
                reasons.append(f"{timeframe}_FRESHNESS_{status or 'MISSING_STATUS'}")
                continue
            row = matches[0]
            max_age_value = row.get("max_age_seconds")
            expected_max_age = expected_max_ages[timeframe]
            if (
                isinstance(max_age_value, bool)
                or not isinstance(max_age_value, (int, float))
                or float(max_age_value) != expected_max_age
            ):
                reasons.append(f"{timeframe}_MAX_AGE_CONTRACT_INVALID")
                continue
            closed_at = _parse_aware_utc(
                row.get("latest_complete_candle_closed_at_utc")
            )
            if closed_at is None:
                reasons.append(f"{timeframe}_CLOSED_AT_INVALID")
                continue
            row_age = (now - closed_at).total_seconds()
            if row_age < -5:
                reasons.append(f"{timeframe}_CLOSED_AT_FUTURE")
            elif row_age > expected_max_age:
                reasons.append(f"{timeframe}_CLOSED_AT_STALE")
    if not reasons:
        return None
    return {
        "reasons": reasons,
        "checked_at_utc": payload.get("checked_at_utc"),
        "source_generated_at_utc": payload.get("source_generated_at_utc"),
        "source_pair_charts_sha256": payload.get("source_pair_charts_sha256"),
        "pair_charts_sha256": pair_charts_sha256 or None,
        "row_statuses": {
            timeframe: [str(row.get("status") or "") for row in matches]
            for timeframe, matches in sorted(pair_rows.items())
        },
    }


def _guardian_monitor_pairs(pair_charts: dict[str, Any], *, snapshot: dict[str, Any]) -> list[str]:
    pairs: list[str] = []
    explicit = pair_charts.get("guardian_monitor_pairs")
    for raw in explicit if isinstance(explicit, list) else []:
        pair = _pair(raw)
        if pair and pair not in pairs:
            pairs.append(pair)
    # The fast wrapper already ranks and clamps pending/candidate pairs.  When
    # that explicit scope is present, only add missing open exposures here;
    # appending every broker order would exceed the bounded chart set and emit
    # artificial PAIR_CHART_MISSING events for intentionally deferred pairs.
    collections = (snapshot.get("positions", []) or [],) if isinstance(explicit, list) else (
        snapshot.get("positions", []) or [],
        snapshot.get("orders", []) or [],
    )
    for collection in collections:
        for item in collection:
            if not isinstance(item, dict):
                continue
            pair = _pair(item.get("pair") or item.get("instrument"))
            if pair and pair not in pairs:
                pairs.append(pair)
    return pairs


def _guardian_monitor_scope(pair_charts: dict[str, Any], pair: str) -> Any:
    scope = pair_charts.get("guardian_monitor_scope")
    if isinstance(scope, dict):
        return scope.get(pair) or scope.get(pair.replace("_", "/")) or []
    return []


def _guardian_chart_freshness(pair_charts: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    generated = _parse_aware_utc(pair_charts.get("generated_at_utc"))
    max_age = int(os.environ.get("QR_GUARDIAN_CHART_MAX_AGE_SECONDS", DEFAULT_GUARDIAN_CHART_MAX_AGE_SECONDS))
    if generated is None:
        return {"fresh": False, "status": "INVALID_GENERATED_AT", "max_age_seconds": max_age}
    age = (now - generated).total_seconds()
    if age < -5:
        return {
            "fresh": False,
            "status": "FUTURE_GENERATED_AT",
            "generated_at_utc": generated.isoformat(),
            "age_seconds": round(age, 3),
            "max_age_seconds": max_age,
        }
    return {
        "fresh": age <= max_age,
        "status": "FRESH" if age <= max_age else "STALE",
        "generated_at_utc": generated.isoformat(),
        "age_seconds": round(age, 3),
        "max_age_seconds": max_age,
    }


def _pair_has_open_exposure(snapshot: dict[str, Any], pair: str) -> bool:
    return any(
        isinstance(item, dict)
        and _pair(item.get("pair") or item.get("instrument")) == pair
        and abs(_float(item.get("units")) or 0.0) > 0.0
        for item in snapshot.get("positions", []) or []
    )


def _technical_state_event(
    pair: str,
    chart: dict[str, Any],
    *,
    pair_charts: dict[str, Any],
    snapshot: dict[str, Any],
    now: datetime,
    technical_integrity_required: bool = False,
) -> GuardianEvent | None:
    views = {
        str(item.get("granularity") or "").upper(): item
        for item in chart.get("views", []) or []
        if isinstance(item, dict)
    }
    if technical_integrity_required:
        integrity_issues: list[str] = []
        chart_generated_at = _parse_aware_utc(pair_charts.get("generated_at_utc"))
        timeframe_seconds = {"M1": 60, "M5": 300, "M15": 900}
        for timeframe, seconds in timeframe_seconds.items():
            view = views.get(timeframe)
            if not isinstance(view, dict):
                integrity_issues.append(f"{timeframe}_VIEW_MISSING")
                continue
            integrity = (
                view.get("candle_integrity")
                if isinstance(view.get("candle_integrity"), dict)
                else {}
            )
            scope_bound = bool(
                integrity.get("pair") == pair
                and integrity.get("granularity") == timeframe
                and view.get("granularity") == timeframe
            )
            receipt_valid = bool(
                integrity
                and chart_generated_at is not None
                and scope_bound
                and validate_mba_integrity_receipt(
                    integrity,
                    chart_generated_at=chart_generated_at,
                    view=view,
                    now_utc=None,
                )
            )
            if not receipt_valid:
                integrity_issues.append(f"{timeframe}_RECEIPT_INVALID")
                continue
            started_at = _parse_aware_utc(
                integrity.get("latest_complete_timestamp_utc")
            )
            if started_at is None:
                integrity_issues.append(f"{timeframe}_LATEST_COMPLETE_INVALID")
                continue
            closed_at = started_at + timedelta(seconds=seconds)
            age = (now - closed_at).total_seconds()
            if age < -5:
                integrity_issues.append(f"{timeframe}_LATEST_COMPLETE_FUTURE")
            elif age > seconds * 2:
                integrity_issues.append(f"{timeframe}_LATEST_COMPLETE_STALE")
        if integrity_issues:
            open_exposure = _pair_has_open_exposure(snapshot, pair)
            return _event(
                event_type="TECHNICAL_INPUT_STALE",
                pair=pair,
                direction=None,
                thesis="canonical technical candle integrity receipt",
                price_zone="guardian technical receipt is invalid, missing, or stale",
                severity="P1" if open_exposure else "P2",
                recommended_review_type="TUNING_REVIEW",
                action_hint="HOLD" if open_exposure else "NO_ACTION",
                now=now,
                details={
                    "fresh": False,
                    "status": "TECHNICAL_CANDLE_INTEGRITY_RECEIPT_INVALID",
                    "integrity_issues": integrity_issues,
                    "monitor_scope": _guardian_monitor_scope(pair_charts, pair),
                    "open_exposure": open_exposure,
                    "live_permission_allowed": False,
                    "no_direct_oanda": True,
                    "preserve_blockers": True,
                },
            )
    blocked_integrity: dict[str, dict[str, Any]] = {}
    for timeframe in ("M1", "M5", "M15"):
        view = views.get(timeframe)
        integrity = (
            view.get("candle_integrity")
            if isinstance(view, dict)
            and isinstance(view.get("candle_integrity"), dict)
            else {}
        )
        if integrity.get("forecast_blocking") is not True:
            continue
        codes = integrity.get("blocking_codes")
        blocked_integrity[timeframe] = {
            "evaluation_status": str(
                integrity.get("evaluation_status") or "UNKNOWN"
            ).upper(),
            "blocking_codes": [
                str(code)
                for code in codes
                if isinstance(code, str) and code
            ]
            if isinstance(codes, list)
            else [],
            "latest_complete_timestamp_utc": integrity.get(
                "latest_complete_timestamp_utc"
            ),
            "recent_clean_tail_count": integrity.get("recent_clean_tail_count"),
            "recent_tail_state": integrity.get("recent_tail_state"),
        }
    aggregate_integrity = (
        chart.get("technical_candle_integrity")
        if isinstance(chart.get("technical_candle_integrity"), dict)
        else {}
    )
    if aggregate_integrity.get("forecast_blocking") is True and not blocked_integrity:
        codes = aggregate_integrity.get("blocking_codes")
        blocked_integrity["AGGREGATE"] = {
            "evaluation_status": str(
                aggregate_integrity.get("evaluation_status") or "UNKNOWN"
            ).upper(),
            "blocking_codes": [
                str(code)
                for code in codes
                if isinstance(code, str) and code
            ]
            if isinstance(codes, list)
            else [],
            "latest_complete_timestamp_utc": None,
            "recent_clean_tail_count": None,
            "recent_tail_state": None,
        }
    if blocked_integrity:
        open_exposure = _pair_has_open_exposure(snapshot, pair)
        blocking_codes = sorted(
            {
                code
                for item in blocked_integrity.values()
                for code in item["blocking_codes"]
            }
        )
        return _event(
            event_type="TECHNICAL_INPUT_STALE",
            pair=pair,
            direction=None,
            thesis="bounded guardian technical candle integrity",
            price_zone="guardian technical input is quarantined by broker MBA integrity",
            severity="P1" if open_exposure else "P2",
            recommended_review_type="TUNING_REVIEW",
            action_hint="HOLD" if open_exposure else "NO_ACTION",
            now=now,
            details={
                "fresh": False,
                "status": "TECHNICAL_CANDLE_INTEGRITY_BLOCKED",
                "blocked_timeframes": list(blocked_integrity),
                "blocking_codes": blocking_codes,
                "integrity_by_timeframe": blocked_integrity,
                "monitor_scope": _guardian_monitor_scope(pair_charts, pair),
                "open_exposure": open_exposure,
                "live_permission_allowed": False,
                "no_direct_oanda": True,
                "preserve_blockers": True,
            },
        )
    m5 = views.get("M5") or views.get("M1")
    if not isinstance(m5, dict):
        open_exposure = _pair_has_open_exposure(snapshot, pair)
        return _event(
            event_type="TECHNICAL_INPUT_STALE",
            pair=pair,
            direction=None,
            thesis="bounded guardian chart input completeness",
            price_zone="guardian technical input is missing both M1 and M5 views",
            severity="P1" if open_exposure else "P2",
            recommended_review_type="TUNING_REVIEW",
            action_hint="HOLD" if open_exposure else "NO_ACTION",
            now=now,
            details={
                "fresh": False,
                "status": "REQUIRED_FAST_VIEWS_MISSING",
                "required_any_of": ["M1", "M5"],
                "available_views": sorted(views),
                "monitor_scope": _guardian_monitor_scope(pair_charts, pair),
                "open_exposure": open_exposure,
                "live_permission_allowed": False,
                "no_direct_oanda": True,
                "preserve_blockers": True,
            },
        )
    indicators = m5.get("indicators") if isinstance(m5.get("indicators"), dict) else {}
    mid = _chart_mid(chart, snapshot)
    if mid is None:
        mid = _float(indicators.get("close"))
    atr_pips = _float(indicators.get("atr_pips"))
    spread_pips = _live_spread_pips(snapshot, pair)
    family_consensus = _technical_family_consensus(views)
    closed_structure = _latest_closed_structure(views)
    closed_candle_watermarks = _latest_complete_candle_watermarks(views)
    if not closed_candle_watermarks:
        open_exposure = _pair_has_open_exposure(snapshot, pair)
        return _event(
            event_type="TECHNICAL_INPUT_STALE",
            pair=pair,
            direction=None,
            thesis="bounded guardian chart input completeness",
            price_zone="guardian technical input has no provably complete fast candle",
            severity="P1" if open_exposure else "P2",
            recommended_review_type="TUNING_REVIEW",
            action_hint="HOLD" if open_exposure else "NO_ACTION",
            now=now,
            details={
                "fresh": False,
                "status": "COMPLETE_FAST_CANDLE_MISSING",
                "required_any_of": ["M1", "M5", "M15"],
                "available_views": sorted(views),
                "closed_candle_watermarks": {},
                "monitor_scope": _guardian_monitor_scope(pair_charts, pair),
                "open_exposure": open_exposure,
                "live_permission_allowed": False,
                "no_direct_oanda": True,
                "preserve_blockers": True,
            },
        )
    dominant_regime = str(chart.get("dominant_regime") or "UNKNOWN").upper()
    volatility_bucket = str(indicators.get("regime_quantile") or "UNKNOWN").upper()
    fingerprint = {
        "dominant_regime": dominant_regime,
        "volatility_bucket": volatility_bucket,
        "family_consensus": family_consensus,
        "closed_structure": closed_structure,
    }
    direction_votes = [
        str(value).upper()
        for value in family_consensus.values()
        if str(value).upper() in {"UP", "DOWN"}
    ]
    up_votes = direction_votes.count("UP")
    down_votes = direction_votes.count("DOWN")
    direction = "LONG" if up_votes > down_votes else "SHORT" if down_votes > up_votes else None
    open_exposure = _pair_has_open_exposure(snapshot, pair)
    price_zone = (
        f"mid={mid:.8f} spread_pips={spread_pips:.3f} m5_atr_pips={atr_pips:.3f}"
        if mid is not None and spread_pips is not None and atr_pips is not None
        else "bounded closed-candle technical state"
    )
    return _event(
        event_type="TECHNICAL_STATE_CHANGE",
        pair=pair,
        direction=direction,
        thesis="bounded closed-candle technical state",
        price_zone=price_zone,
        severity="P1" if open_exposure else "P2",
        recommended_review_type="THESIS_REVIEW" if open_exposure else "TUNING_REVIEW",
        action_hint="HOLD" if open_exposure else "NO_ACTION",
        now=now,
        details={
            "mid": mid,
            "live_spread_pips": spread_pips,
            "m5_atr_pips": atr_pips,
            "material_threshold_pips": max(
                2.0 * (spread_pips if spread_pips is not None else NORMAL_SPREAD_PIPS.get(pair, 1.0)),
                atr_pips if atr_pips is not None else 0.0,
                1.0,
            ),
            "material_fingerprint": fingerprint,
            # Separate the semantic detector output above from the market-data
            # observation that produced it.  A regenerated weekend chart can
            # otherwise oscillate between equivalent regime labels without a
            # new closed candle and repeatedly wake GPT/tuning work.
            "closed_candle_watermarks": closed_candle_watermarks,
            "chart_generated_at_utc": pair_charts.get("generated_at_utc"),
            "monitor_scope": _guardian_monitor_scope(pair_charts, pair),
            "open_exposure": open_exposure,
            "live_permission_allowed": False,
            "no_direct_oanda": True,
            "preserve_blockers": True,
        },
    )


def _technical_family_consensus(views: dict[str, dict[str, Any]]) -> dict[str, str]:
    votes: dict[str, list[str]] = {"trend": [], "mean_reversion": [], "breakout": []}
    timeframes = _technical_consensus_timeframes(views)
    for timeframe in timeframes:
        view = views.get(timeframe)
        if not isinstance(view, dict):
            continue
        scores = view.get("family_scores") if isinstance(view.get("family_scores"), dict) else {}
        indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else {}
        for family, score_key in (("trend", "trend_score"), ("mean_reversion", "mean_rev_score")):
            score = _float(scores.get(score_key))
            if score is not None and abs(score) >= TECHNICAL_FAMILY_MIN_SCORE:
                votes[family].append("UP" if score > 0 else "DOWN")
        breakout_score = _float(scores.get("breakout_score"))
        if breakout_score is not None and abs(breakout_score) >= TECHNICAL_FAMILY_MIN_SCORE:
            # FamilyScores.breakout_score is primarily expansion/squeeze
            # magnitude.  Its negative sign can mean "already expanded", not
            # bearish direction, so never turn the score sign itself into a
            # directional vote.  Direction must come from an explicit
            # Donchian/BOS observation or an independent trend proxy.
            direction = _breakout_direction(view, scores=scores, indicators=indicators)
            if direction is not None:
                votes["breakout"].append(direction)
    consensus: dict[str, str] = {}
    for family, family_votes in votes.items():
        if family_votes.count("UP") >= 2:
            consensus[family] = "UP"
        elif family_votes.count("DOWN") >= 2:
            consensus[family] = "DOWN"
        else:
            consensus[family] = "MIXED"
    return consensus


def _breakout_direction(
    view: dict[str, Any],
    *,
    scores: dict[str, Any],
    indicators: dict[str, Any],
) -> str | None:
    components = (
        scores.get("breakout_components")
        if isinstance(scores.get("breakout_components"), dict)
        else {}
    )
    donchian_break = _float(components.get("donchian_break"))
    if donchian_break is not None and donchian_break != 0:
        return "UP" if donchian_break > 0 else "DOWN"

    structure = view.get("structure") if isinstance(view.get("structure"), dict) else {}
    structure_events = structure.get("structure_events") or []
    for event in reversed(structure_events if isinstance(structure_events, list) else []):
        if not isinstance(event, dict) or event.get("close_confirmed") is not True:
            continue
        kind = str(event.get("kind") or "").upper()
        if "UP" in kind and ("BOS" in kind or "BREAK" in kind):
            return "UP"
        if "DOWN" in kind and ("BOS" in kind or "BREAK" in kind):
            return "DOWN"

    close = _float(indicators.get("close"))
    donchian_high = _float(indicators.get("donchian_high"))
    donchian_low = _float(indicators.get("donchian_low"))
    if close is not None:
        if donchian_high is not None and close >= donchian_high:
            return "UP"
        if donchian_low is not None and close <= donchian_low:
            return "DOWN"

    for key in ("supertrend_dir", "linreg_slope_20"):
        proxy = _float(indicators.get(key))
        if proxy is not None and proxy != 0:
            return "UP" if proxy > 0 else "DOWN"
    return None


def _latest_closed_structure(views: dict[str, dict[str, Any]]) -> str:
    rows: list[tuple[datetime, str]] = []
    for timeframe in _technical_consensus_timeframes(views):
        view = views.get(timeframe)
        if not isinstance(view, dict):
            continue
        structure = view.get("structure") if isinstance(view.get("structure"), dict) else {}
        for item in structure.get("structure_events", []) or []:
            if not isinstance(item, dict) or item.get("close_confirmed") is not True:
                continue
            timestamp = _parse_utc(item.get("timestamp"))
            if timestamp is None:
                continue
            rows.append((timestamp, f"{timeframe}:{str(item.get('kind') or 'UNKNOWN').upper()}:{timestamp.isoformat()}"))
    return max(rows, key=lambda item: item[0])[1] if rows else "NONE"


def _technical_consensus_timeframes(views: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    higher = tuple(timeframe for timeframe in ("M15", "M30", "H1") if timeframe in views)
    if len(higher) >= 2:
        return higher
    return tuple(timeframe for timeframe in ("M1", "M5", "M15") if timeframe in views)


def _latest_complete_candle_watermarks(
    views: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Return the latest complete source-candle timestamp per fast view.

    ``chart_generated_at_utc`` is an observation timestamp, not new market
    information.  These watermarks let the router distinguish a genuinely new
    closed-candle observation from a recomputation of the same frozen input.
    """

    watermarks: dict[str, str] = {}
    # The wake contract is intentionally tied to the fast monitoring surface,
    # not to whichever higher-timeframe set happens to be used for consensus.
    # Quick (M1/M5/M15) and full (M5/M15/M30/H1) chart packets share this
    # comparable clock, so switching packet shape cannot manufacture market
    # progress while a genuinely new fast close still can.
    for timeframe in (item for item in ("M1", "M5", "M15") if item in views):
        view = views.get(timeframe)
        if not isinstance(view, dict):
            continue
        latest: datetime | None = None
        for candle in view.get("recent_candles", []) or []:
            if not isinstance(candle, dict) or candle.get("complete") is not True:
                continue
            timestamp = _parse_utc(
                candle.get("t")
                or candle.get("timestamp")
                or candle.get("timestamp_utc")
                or candle.get("time")
            )
            if timestamp is not None and (latest is None or timestamp > latest):
                latest = timestamp
        if latest is not None:
            watermarks[timeframe] = latest.isoformat()
    return watermarks


def _live_spread_pips(snapshot: dict[str, Any], pair: str) -> float | None:
    quotes = snapshot.get("quotes") if isinstance(snapshot.get("quotes"), dict) else {}
    quote = quotes.get(pair)
    if not isinstance(quote, dict):
        return None
    bid = _float(quote.get("bid"))
    ask = _float(quote.get("ask"))
    if bid is None or ask is None or ask < bid:
        return None
    return (ask - bid) * instrument_pip_factor(pair)


def _side_matched_failed_acceptance(metadata: dict[str, Any], side: str | None) -> bool:
    """Accept only the explicit M5 failed-break predicate for the intent side."""

    if side not in {"LONG", "SHORT"}:
        return False
    return metadata.get(f"oanda_m5_failed_break_{side.lower()}") is True


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
        if _side_matched_failed_acceptance(metadata, side):
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
    source_generated_at_utc = payload.get("generated_at_utc")
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
                            "source_generated_at_utc": source_generated_at_utc,
                        },
                    )
                )
    return events


def _margin_pressure_event(
    account: dict[str, Any],
    *,
    positions: list[dict[str, Any]],
    now: datetime,
) -> GuardianEvent | None:
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
    reduction_scope = _margin_reduction_scope(positions)
    reduction_actionable = bool(
        severity == "P0"
        and reduction_scope["system_reduction_candidate_trade_ids"]
    )
    executable_reduction_trade_ids = (
        reduction_scope["system_reduction_candidate_trade_ids"]
        if reduction_actionable
        else []
    )
    return _event(
        event_type="MARGIN_PRESSURE",
        pair="PORTFOLIO",
        direction=None,
        thesis="portfolio margin capacity",
        price_zone=f"margin_used/nav={utilization:.3f}; available/nav={available_ratio:.3f}; cap={cap_fraction:.3f}",
        severity=severity,
        recommended_review_type="RISK_REVIEW",
        action_hint="REDUCE" if reduction_actionable else "HOLD",
        thesis_state="EMERGENCY" if severity == "P0" else "WOUNDED",
        now=now,
        details={
            "nav_jpy": nav,
            "margin_used_jpy": used,
            "margin_available_jpy": available,
            "max_margin_utilization_pct": cap_pct,
            **reduction_scope,
            "executable_reduction_target_trade_ids": (
                executable_reduction_trade_ids
            ),
            "executable_reduction_target_count": len(
                executable_reduction_trade_ids
            ),
            # P1 is a capacity warning below the same 95% hard cap enforced
            # from current broker truth by RiskEngine and LiveOrderGateway.
            # P0 (at/over cap or no available margin) remains a universal
            # fresh-entry block.  Keep both states explicit in the event so a
            # downstream receipt cannot collapse P1 back into P0 by guessing
            # from the word "MARGIN" alone.
            "fresh_entry_risk_block_active": severity == "P0",
            "fresh_entry_risk_block_reason": "MARGIN_PRESSURE",
            "fresh_entry_risk_observation_only": severity == "P1",
            "fresh_entry_margin_contract": (
                P1_MARGIN_WARNING_CONTRACT
                if severity == "P1"
                else P0_MARGIN_HARD_CAP_CONTRACT
            ),
        },
    )


def _margin_reduction_scope(
    positions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Expose only explicitly system-owned positions as reduction candidates."""

    open_positions = [
        position
        for position in positions
        if _position_units(position) not in {None, 0.0}
    ]
    operator_manual_positions = [
        position
        for position in open_positions
        if is_operator_manual_position(position)
        or _owner(position.get("owner")) == Owner.MANUAL.value
    ]
    system_positions = [
        position
        for position in open_positions
        if _owner(position.get("owner")) == Owner.TRADER.value
        and not is_operator_manual_position(position)
    ]
    other_positions = [
        position
        for position in open_positions
        if position not in operator_manual_positions
        and position not in system_positions
    ]

    def trade_ids(rows: list[dict[str, Any]]) -> list[str]:
        return sorted(
            {
                str(_position_trade_id(row) or "").strip()
                for row in rows
                if str(_position_trade_id(row) or "").strip()
            }
        )

    return {
        "open_position_count": len(open_positions),
        "operator_manual_only_exposure": bool(open_positions)
        and len(operator_manual_positions) == len(open_positions),
        "operator_manual_trade_ids": trade_ids(operator_manual_positions),
        "system_reduction_candidate_trade_ids": trade_ids(system_positions),
        "non_system_unattributed_trade_ids": trade_ids(other_positions),
    }
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
        normal_spread = NORMAL_SPREAD_PIPS.get(pair)
        max_spread_multiple = RiskPolicy().max_spread_multiple
        spread_pips = max(0.0, (ask - bid) * instrument_pip_factor(pair))
        # A rollover/closed-market ask expansion can move the midpoint across
        # a major figure while the bid and every closed candle remain frozen.
        # SPREAD_ANOMALY already owns that observation.  Do not manufacture a
        # failed-acceptance tuning event from a non-executable midpoint.
        if (
            normal_spread is not None
            and normal_spread > 0.0
            and max_spread_multiple is not None
            and max_spread_multiple > 0.0
            and spread_pips > normal_spread * max_spread_multiple
        ):
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
                details={"bid": bid, "ask": ask, "spread_pips": spread_pips},
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


def _contract_entry_is_live_relevant(entry: dict[str, Any]) -> bool:
    if _contract_entry_has_open_exposure(entry):
        return True
    for key in (
        "selected",
        "current",
        "live_relevant",
        "is_selected",
        "is_current",
        "selected_for_review",
        "active",
        "active_pending_risk",
    ):
        if _truthy(entry.get(key)):
            return True
    for key in ("status", "intent_status", "lane_status", "contract_status"):
        status = str(entry.get(key) or "").strip().upper()
        if status in {"LIVE_READY", "SELECTED", "CURRENT", "ACTIVE", "PENDING", "ARMED", "OPEN"}:
            return True
    for bucket in CONTRACT_TRIGGER_BUCKETS:
        triggers = entry.get(bucket) if isinstance(entry.get(bucket), list) else []
        if any(_trigger_explicitly_fired(trigger) for trigger in triggers):
            return True
    return False


def _contract_core_trigger_arrays_empty(entry: dict[str, Any]) -> bool:
    return all(
        isinstance(entry.get(bucket), list) and not entry.get(bucket)
        for bucket in (
            "harvest_triggers",
            "no_add_triggers",
            "wounded_triggers",
            "invalidation_triggers",
            "emergency_triggers",
        )
    )


def _contract_entry_watch_only_reason(entry: dict[str, Any]) -> str:
    for key in ("watch_only_reason", "reason", "watch_reason"):
        value = str(entry.get(key) or "").strip()
        if value:
            return value
    return ""


def _contract_open_exposure_keys(entry: dict[str, Any]) -> set[str]:
    pair = _pair(entry.get("pair"))
    side = _direction_from_text(entry.get("side"))
    trade_id = str(entry.get("trade_id") or "").strip()
    keys: set[str] = set()
    if trade_id:
        keys.add(f"trade:{trade_id}")
    if pair and side:
        keys.add(f"pair-side:{pair}:{side}")
    return keys


def _snapshot_open_exposure_contract_refs(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for position in snapshot.get("positions", []) or []:
        if not isinstance(position, dict):
            continue
        pair = _pair(position.get("pair"))
        side = _direction_from_position(position)
        trade_id = _position_trade_id(position)
        if not pair or not side:
            continue
        keys = set()
        if trade_id:
            keys.add(f"trade:{trade_id}")
        keys.add(f"pair-side:{pair}:{side}")
        refs.append({"pair": pair, "side": side, "trade_id": trade_id, "keys": keys})
    return refs


def _selected_order_intent_lane_ids(order_intents: dict[str, Any]) -> set[str]:
    selected: set[str] = set()
    for key in ("selected_lane_id", "current_lane_id", "selected_result_lane_id"):
        value = str(order_intents.get(key) or "").strip()
        if value:
            selected.add(value)
    for key in ("selected_lane_ids", "current_lane_ids"):
        values = order_intents.get(key)
        if isinstance(values, list):
            selected.update(str(value).strip() for value in values if str(value).strip())
    selected_result = order_intents.get("selected_result")
    if isinstance(selected_result, dict):
        value = str(selected_result.get("lane_id") or "").strip()
        if value:
            selected.add(value)
    current_result = order_intents.get("current_result")
    if isinstance(current_result, dict):
        value = str(current_result.get("lane_id") or "").strip()
        if value:
            selected.add(value)
    return selected


def _candidate_watch_only_reason(result: dict[str, Any], intent: dict[str, Any], thesis: str) -> str:
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    for key in ("watch_only_reason", "reason"):
        value = str(metadata.get(key) or result.get(key) or "").strip()
        if _truthy(metadata.get("watch_only")) or _truthy(result.get("watch_only")):
            return value or "no_current_thesis"
    status = str(result.get("status") or "").strip().upper()
    if status in {"WATCH_ONLY", "MARKET_READ_FIRST", "DRY_RUN_BLOCKED", "NO_TRADE"}:
        return str(metadata.get("watch_only_reason") or result.get("reason") or "no_current_thesis").strip()
    thesis_text = str(thesis or "").strip()
    if not thesis_text or thesis_text == "candidate thesis":
        return "no_current_thesis"
    return ""


def _default_candidate_triggers(
    entry: dict[str, Any],
    result: dict[str, Any],
    intent: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, list[dict[str, Any]]]:
    del now
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    pair = _pair(entry.get("pair")) or "UNKNOWN"
    side = _direction_from_text(entry.get("side")) or "UNKNOWN"
    lane_id = str(result.get("lane_id") or entry.get("lane_id") or "").strip()
    entry_price = _float(intent.get("entry") or intent.get("entry_price") or metadata.get("entry"))
    take_profit = _float(
        intent.get("take_profit")
        or intent.get("tp")
        or intent.get("target")
        or metadata.get("take_profit")
        or metadata.get("tp")
        or metadata.get("target")
    )
    stop_loss = _float(
        intent.get("stop_loss")
        or intent.get("sl")
        or intent.get("invalidation_price")
        or metadata.get("stop_loss")
        or metadata.get("sl")
        or metadata.get("invalidation_price")
    )
    spread_cap = _float(metadata.get("max_spread_pips") or metadata.get("spread_cap_pips"))
    base_ref = {
        "lane_id": lane_id or None,
        "pair": pair,
        "side": side,
        "owner": _contract_owner(entry.get("owner")),
        "thesis": entry.get("thesis"),
        "thesis_state": _thesis_state(entry.get("thesis_state")),
    }
    harvest_trigger = {
        **base_ref,
        "trigger_id": "candidate_profit_harvest_review",
        "status": "PENDING",
        "kind": "candidate_harvest_review",
        "action_hint": "HARVEST",
        "evidence_required": "review only after candidate target/harvest zone is reached and gateway/risk gates still allow the action",
    }
    if take_profit is not None:
        harvest_trigger["condition"] = {
            "metric": "mid",
            "operator": ">=" if side == "LONG" else "<=",
            "value": take_profit,
        }
    no_add_trigger = {
        **base_ref,
        "trigger_id": "candidate_no_add_guard",
        "status": "PENDING",
        "kind": "candidate_no_add_guard",
        "action_hint": "HOLD",
        "evidence_required": "no add unless the thesis remains ALIVE, guardian/operator-review blockers are clear, and gateway evidence is fresh",
    }
    if spread_cap is not None:
        no_add_trigger["condition"] = {"metric": "spread_pips", "operator": ">", "value": spread_cap}
    wounded_trigger = {
        **base_ref,
        "trigger_id": "candidate_wounded_review",
        "status": "PENDING",
        "kind": "candidate_wounded_review",
        "action_hint": "HOLD",
        "evidence_required": "failed acceptance, range rail change, or market-read evidence materially weakens the candidate thesis",
    }
    if entry_price is not None and stop_loss is not None:
        wounded_level = (entry_price + stop_loss) / 2.0
        wounded_trigger["condition"] = {
            "metric": "mid",
            "operator": "<=" if side == "LONG" else ">=",
            "value": wounded_level,
        }
    invalidation_trigger = {
        **base_ref,
        "trigger_id": "candidate_invalidation_review",
        "status": "PENDING",
        "kind": "candidate_invalidation_review",
        "action_hint": "HOLD",
        "evidence_required": "accepted break through the candidate invalidation level; no broker action is implied by the contract",
    }
    if stop_loss is not None:
        invalidation_trigger["condition"] = {
            "metric": "mid",
            "operator": "<=" if side == "LONG" else ">=",
            "value": stop_loss,
        }
    emergency_trigger = {
        **base_ref,
        "trigger_id": "candidate_margin_or_broker_truth_emergency",
        "status": "PENDING",
        "kind": "candidate_emergency_review",
        "action_hint": "HOLD",
        "evidence_required": "margin pressure, broker snapshot staleness, or gateway-outside exposure blocks fresh routing",
        "condition": {"metric": "margin_available_jpy", "operator": "<=", "value": 0},
    }
    return {
        "harvest_triggers": [{key: value for key, value in harvest_trigger.items() if value is not None}],
        "no_add_triggers": [{key: value for key, value in no_add_trigger.items() if value is not None}],
        "wounded_triggers": [{key: value for key, value in wounded_trigger.items() if value is not None}],
        "invalidation_triggers": [{key: value for key, value in invalidation_trigger.items() if value is not None}],
        "emergency_triggers": [{key: value for key, value in emergency_trigger.items() if value is not None}],
    }


def _range_rail_watch_contract_entry(
    artifact: dict[str, Any] | None,
    *,
    now: datetime,
) -> dict[str, Any] | None:
    payload = artifact if isinstance(artifact, dict) else {}
    if not payload or payload.get("read_only") is not True or payload.get("live_permission_allowed") is not False:
        return None
    lane = payload.get("top_lane") if isinstance(payload.get("top_lane"), dict) else {}
    if not lane:
        lane = payload.get("target_lane") if isinstance(payload.get("target_lane"), dict) else {}
    condition = lane.get("rail_success_condition") if isinstance(lane.get("rail_success_condition"), dict) else {}
    pair = _pair(condition.get("pair") or lane.get("pair"))
    side = _direction_from_text(condition.get("direction") or lane.get("direction"))
    trigger = _range_rail_watch_trigger(lane, condition)
    if not pair or not side or trigger is None:
        return None
    lane_id = str(lane.get("lane_id") or trigger.get("lane_id") or "").strip()
    repair_status = str(payload.get("status") or lane.get("repair_status") or "").strip() or "RANGE_RAIL_GEOMETRY_REPAIR"
    deadline = datetime.fromtimestamp(now.timestamp() + DEFAULT_CONTRACT_REVIEW_DEADLINE_SECONDS, timezone.utc)
    blockers = _unique(
        _string_list(lane.get("blockers"))
        + _string_list(condition.get("must_preserve_blockers"))
        + _range_rail_counterpart_blockers(lane)
    )
    return {
        "pair": pair,
        "side": side,
        "thesis": f"range rail recheck for {lane_id or pair}",
        "owner": "SYSTEM",
        "thesis_state": "ALIVE",
        "lane_id": lane_id or None,
        "strategy_family": lane.get("strategy_family"),
        "vehicle": lane.get("vehicle"),
        "status": repair_status,
        "current": True,
        "selected_for_review": True,
        "watch_only": True,
        "watch_only_reason": "WAIT_FOR_RANGE_RAIL_RECHECK",
        "range_rail_watch": {
            "source_artifact": "data/range_rail_geometry_repair.json",
            "source_status": repair_status,
            "rail_status": (lane.get("range_box") if isinstance(lane.get("range_box"), dict) else {}).get("rail_status"),
            "condition": condition,
            "preserve_blockers": blockers,
            "live_permission_allowed": False,
        },
        "harvest_triggers": [],
        "add_triggers": [trigger],
        "no_add_triggers": [],
        "wounded_triggers": [],
        "invalidation_triggers": [],
        "emergency_triggers": [],
        "next_review_reason": (
            "watch-only range rail recheck; wake GPT when quote reaches the executable rail, "
            "then reprice/prove blockers before any gateway route"
        ),
        "next_review_deadline_utc": deadline.isoformat(),
    }


def _range_rail_watch_trigger(lane: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any] | None:
    pair = _pair(condition.get("pair") or lane.get("pair"))
    side = _direction_from_text(condition.get("direction") or lane.get("direction"))
    low = _float(condition.get("range_low_price"))
    high = _float(condition.get("range_high_price"))
    if not pair or not side or low is None or high is None or high <= low:
        return None
    if side == "LONG":
        box_threshold = _float(condition.get("required_box_position_lte"))
        operator = "<="
    else:
        box_threshold = _float(condition.get("required_box_position_gte"))
        operator = ">="
    if box_threshold is None:
        return None
    threshold_price = low + (high - low) * box_threshold
    lane_id = str(lane.get("lane_id") or "").strip()
    blockers = _unique(
        _string_list(lane.get("blockers"))
        + _string_list(condition.get("must_preserve_blockers"))
        + _range_rail_counterpart_blockers(lane)
    )
    trigger_id = f"range_rail_recheck_ready:{_slug(lane_id or pair)}"
    return {
        "trigger_id": trigger_id,
        "status": "PENDING",
        "kind": "range_rail_recheck",
        "lane_id": lane_id or None,
        "pair": pair,
        "side": side,
        "owner": "SYSTEM",
        "thesis_state": "ALIVE",
        "action_hint": "ADD",
        "recommended_review_type": "ADD_REVIEW",
        "severity": "P1",
        "condition": {
            "metric": "mid",
            "operator": operator,
            "value": threshold_price,
            "range_box_position_threshold": box_threshold,
            "range_low_price": low,
            "range_high_price": high,
            "source": "range_rail_geometry_repair.rail_success_condition",
        },
        "rail_success_condition": condition,
        "evidence_required": (
            "rail reached only wakes GPT; preserve spread, bid/ask, negative-expectancy, "
            "range-location, proof-floor, and gateway blockers before any live route"
        ),
        "preserve_blockers": blockers,
        "live_permission_allowed": False,
        "contract_triggers_do_not_execute": True,
    }


def _range_rail_counterpart_blockers(lane: dict[str, Any]) -> list[str]:
    counterpart = lane.get("range_rotation_counterpart") if isinstance(lane.get("range_rotation_counterpart"), dict) else {}
    geometry = lane.get("counterpart_geometry") if isinstance(lane.get("counterpart_geometry"), dict) else {}
    return _string_list(counterpart.get("blocker_codes")) + _string_list(geometry.get("reasons"))


def _merge_range_rail_watch_entry(
    *,
    entries: list[dict[str, Any]],
    seen: set[str],
    watch_entry: dict[str, Any] | None,
) -> None:
    if not watch_entry:
        return
    lane_id = str(watch_entry.get("lane_id") or "").strip()
    pair = _pair(watch_entry.get("pair"))
    side = _direction_from_text(watch_entry.get("side"))
    for entry in entries:
        if lane_id and str(entry.get("lane_id") or "").strip() == lane_id:
            _merge_range_rail_watch_into_entry(entry, watch_entry)
            return
        if _contract_entry_key(entry) == _contract_entry_key(watch_entry):
            _merge_range_rail_watch_into_entry(entry, watch_entry)
            return
    if pair and side:
        seen.add(_contract_entry_key(watch_entry))
        entries.append({key: value for key, value in watch_entry.items() if value is not None})


def _merge_range_rail_watch_into_entry(entry: dict[str, Any], watch_entry: dict[str, Any]) -> None:
    trigger = (watch_entry.get("add_triggers") or [None])[0]
    if not isinstance(trigger, dict):
        return
    for field_name in (
        "range_rail_watch",
        "watch_only",
        "watch_only_reason",
        "current",
        "selected_for_review",
        "next_review_reason",
        "next_review_deadline_utc",
    ):
        entry[field_name] = watch_entry[field_name]
    for field_name in ("strategy_family", "vehicle", "status"):
        if watch_entry.get(field_name) is not None:
            entry[field_name] = watch_entry[field_name]
    add_triggers = entry.get("add_triggers") if isinstance(entry.get("add_triggers"), list) else []
    entry["add_triggers"] = _replace_trigger(add_triggers, trigger)


def _replace_trigger(triggers: list[Any], replacement: dict[str, Any]) -> list[Any]:
    replacement_id = str(replacement.get("trigger_id") or "").strip()
    kept: list[Any] = []
    for trigger in triggers:
        trigger_id = str(trigger.get("trigger_id") or "").strip() if isinstance(trigger, dict) else ""
        if replacement_id and trigger_id == replacement_id:
            continue
        kept.append(trigger)
    kept.append(replacement)
    return kept


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
    ownership_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    position_payload = position if isinstance(position, dict) else {}
    seed = {
        "pair": pair,
        "side": side,
        "thesis": thesis,
        "owner": owner,
        "thesis_state": _thesis_state(thesis_state),
        "trade_id": _position_trade_id(position_payload),
    }
    prior = preserved.get(_contract_entry_key(seed)) or preserved.get(_contract_entry_legacy_key(seed))
    deadline = now.timestamp() + DEFAULT_CONTRACT_REVIEW_DEADLINE_SECONDS
    base = {
        **seed,
        "units": _position_units(position_payload),
        "avg_entry": _position_average_entry(position_payload),
        "ownership_audit": ownership_audit if isinstance(ownership_audit, dict) else None,
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
    if _contract_owner(base.get("owner")) != "OPERATOR_MANUAL":
        for field_name in CONTRACT_TRIGGER_BUCKETS:
            if isinstance(prior.get(field_name), list) and prior[field_name]:
                merged[field_name] = [
                    _rebind_trigger_to_parent(trigger, parent=base) if isinstance(trigger, dict) else trigger
                    for trigger in prior[field_name]
                ]
    for field_name in ("thesis_state", "next_review_reason", "next_review_deadline_utc"):
        if prior.get(field_name) and (field_name != "next_review_deadline_utc" or _deadline_is_future(prior.get(field_name), now)):
            merged[field_name] = prior[field_name]
    return {key: value for key, value in merged.items() if value is not None}


def _rebind_trigger_to_parent(trigger: dict[str, Any], *, parent: dict[str, Any]) -> dict[str, Any]:
    rebound = dict(trigger)
    for field_name in ("trade_id", "pair", "side", "units", "avg_entry", "owner", "thesis", "thesis_state"):
        if parent.get(field_name) is not None:
            rebound[field_name] = parent[field_name]
    return rebound


def _position_contract_thesis(position: dict[str, Any]) -> str:
    raw = position.get("raw") if isinstance(position.get("raw"), dict) else {}
    operator_packet = raw.get("operator_manual_position") if isinstance(raw.get("operator_manual_position"), dict) else {}
    if (
        not operator_packet
        and not str(position.get("thesis") or "").strip()
        and _owner(position.get("owner")) in {Owner.UNKNOWN.value, Owner.EXTERNAL.value, ""}
    ):
        return "gateway-outside broker position"
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
    if state != "UNKNOWN":
        return state
    if (
        _owner(position.get("owner")) in {Owner.UNKNOWN.value, Owner.EXTERNAL.value, ""}
        and not _position_has_system_lane_or_gateway_receipt(position)
        and not _operator_manual_packet(position)
    ):
        return "UNKNOWN"
    return "ALIVE"


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
    for key in (
        "avg_entry",
        "average_entry",
        "average_entry_price",
        "averagePrice",
        "average_price",
        "price",
        "entry",
        "entry_price",
        "entryPrice",
        "open_price",
    ):
        value = _float(position.get(key))
        if value is not None:
            return value
    raw = position.get("raw") if isinstance(position.get("raw"), dict) else {}
    for key in ("price", "average_entry", "averagePrice", "average_price", "entry_price", "entryPrice", "open_price"):
        value = _float(raw.get(key))
        if value is not None:
            return value
    return None


def _position_ownership_audit(position: dict[str, Any]) -> dict[str, Any]:
    owner = _owner(position.get("owner"))
    evidence: list[str] = []
    raw = position.get("raw") if isinstance(position.get("raw"), dict) else {}
    trade_id = _position_trade_id(position)
    operator_packet = _operator_manual_packet(position)
    if is_operator_manual_position(position) or owner == Owner.MANUAL.value:
        if owner in {Owner.MANUAL.value, Owner.OPERATOR_MANUAL.value}:
            evidence.append(f"snapshot owner={owner}")
        if operator_packet:
            evidence.append("operator_manual_position packet present")
        audit = {
            "status": "OPERATOR_MANUAL",
            "owner_input": owner,
            "trade_id": trade_id,
            "evidence": evidence or ["canonical operator-manual classification"],
            "unresolved": False,
        }
        if operator_packet:
            for key in (
                "operator_decision",
                "management_intent",
                "operator_confirmation_source",
                "no_live_side_effects",
                "system_pl_counted",
                "same_theme_auto_add_allowed",
                "loss_side_auto_close_allowed",
                "auto_sl_attach_allowed",
                "auto_tp_modify_allowed",
            ):
                if key in operator_packet:
                    audit[key] = operator_packet[key]
        return audit
    if _position_has_system_lane_or_gateway_receipt(position):
        return {
            "status": "SYSTEM",
            "owner_input": owner,
            "trade_id": trade_id,
            "evidence": ["gateway receipt/lane/client extension evidence is present"],
            "unresolved": False,
        }
    if owner == Owner.TRADER.value:
        return {
            "status": "SYSTEM",
            "owner_input": owner,
            "trade_id": trade_id,
            "evidence": ["snapshot owner=trader"],
            "unresolved": False,
        }
    return {
        "status": "UNKNOWN_NEEDS_OPERATOR_CONFIRM",
        "owner_input": owner,
        "trade_id": trade_id,
        "evidence": ["no gateway receipt/lane id and no operator manual confirmation packet"],
        "raw_owner": raw.get("owner"),
        "unresolved": True,
        "system_pl_counted": False,
        "loss_side_auto_close_allowed": False,
        "same_theme_auto_add_allowed": False,
        "requires": "operator confirmation or gateway evidence",
    }


def _contract_owner_from_ownership_audit(audit: dict[str, Any]) -> str:
    status = str(audit.get("status") or "").upper()
    if status == "SYSTEM":
        return "SYSTEM"
    if status == "OPERATOR_MANUAL":
        return "OPERATOR_MANUAL"
    return "UNKNOWN"


def _operator_manual_packet(position: dict[str, Any]) -> dict[str, Any]:
    raw = position.get("raw") if isinstance(position.get("raw"), dict) else {}
    packet = raw.get("operator_manual_position") if isinstance(raw.get("operator_manual_position"), dict) else {}
    if not packet and isinstance(position.get("operator_manual_position"), dict):
        packet = position["operator_manual_position"]
    if packet and str(packet.get("packet_type") or "") == OPERATOR_MANUAL_POSITION_PACKET:
        return dict(packet)
    return {}


def _position_has_system_lane_or_gateway_receipt(position: dict[str, Any]) -> bool:
    if _owner(position.get("owner")) == Owner.TRADER.value:
        return True
    for key in (
        "lane_id",
        "entry_lane_id",
        "gateway_lane_id",
        "source_lane_id",
        "gateway_receipt_id",
        "entry_receipt_id",
        "receipt_id",
        "qr_lane_id",
    ):
        if str(position.get(key) or "").strip():
            return True
    raw = position.get("raw") if isinstance(position.get("raw"), dict) else {}
    for key in (
        "lane_id",
        "entry_lane_id",
        "gateway_lane_id",
        "source_lane_id",
        "gateway_receipt_id",
        "entry_receipt_id",
        "receipt_id",
        "qr_lane_id",
    ):
        if str(raw.get(key) or "").strip():
            return True
    for key in ("clientExtensions", "tradeClientExtensions"):
        ext = raw.get(key) if isinstance(raw.get(key), dict) else {}
        tag = str(ext.get("tag") or "").strip().lower()
        if tag == Owner.TRADER.value:
            return True
        comment = str(ext.get("comment") or "").lower()
        cid = str(ext.get("id") or "").lower()
        if any(token in comment for token in ("lane", "gateway", "trader")):
            return True
        if any(token in cid for token in ("lane", "gateway", "trader")):
            return True
    return False


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
    units = _position_units(position)
    base_ref = {
        "trade_id": trade_id,
        "units": units,
        "avg_entry": avg_entry,
        "pair": pair,
        "side": side,
        "owner": owner,
        "thesis": entry.get("thesis"),
        "thesis_state": thesis_state,
    }
    operator_packet = _operator_manual_packet(position)
    if operator_packet:
        for key in (
            "operator_decision",
            "management_intent",
            "operator_confirmation_source",
            "system_pl_counted",
            "same_theme_auto_add_allowed",
            "loss_side_auto_close_allowed",
            "auto_sl_attach_allowed",
            "auto_tp_modify_allowed",
        ):
            if key in operator_packet:
                base_ref[key] = operator_packet[key]
    if _is_usd_jpy_162_manual_fade(pair, side, owner, position):
        return _usd_jpy_manual_fade_triggers(entry, position, now=now)
    unknown_owner = owner == "UNKNOWN"
    if unknown_owner:
        harvest_detail = (
            "UNKNOWN_NEEDS_OPERATOR_CONFIRM: profit-side TP/harvest review only after operator confirmation "
            "or gateway evidence; loss-side automation remains disabled"
        )
        invalidation_detail = (
            "UNKNOWN_NEEDS_OPERATOR_CONFIRM: operator/gateway review only; red P/L or stale routing evidence "
            "does not authorize automatic loss-side close"
        )
        no_add_detail = (
            "UNKNOWN_NEEDS_OPERATOR_CONFIRM: no add into the same pair/theme until operator confirmation "
            "or gateway evidence classifies the exposure"
        )
        emergency_detail = (
            "UNKNOWN_NEEDS_OPERATOR_CONFIRM: margin pressure, NAV shock, missing broker truth, or gateway-outside "
            "exposure needs trader review; do not auto-close loss-side unknown exposure"
        )
        invalidation_action = "HOLD"
        emergency_action = "HOLD"
        harvest_action = "HARVEST"
    elif owner == "OPERATOR_MANUAL":
        harvest_detail = (
            "OPERATOR_MANUAL: read-only monitoring/alerting is allowed; TP modification or "
            "profit action requires explicit operator authorization when auto_tp_modify_allowed=false"
        )
        invalidation_detail = (
            "OPERATOR_MANUAL: operator review only; do not automatically loss-close, attach SL, "
            "or treat red P/L as thesis invalidation"
        )
        no_add_detail = (
            "OPERATOR_MANUAL: do not add into the same pair/theme unless the operator explicitly authorizes overlap"
        )
        emergency_detail = (
            "OPERATOR_MANUAL: margin/NAV/broker-truth emergency is an alert and operator-review condition only; "
            "no automatic loss-side close or SL attach"
        )
        invalidation_action = "HOLD"
        emergency_action = "HOLD"
        harvest_action = "HOLD"
    else:
        harvest_detail = "profit-side TP/harvest review when current quote reaches the broker TP, declared harvest zone, or positive UPL is outside spread/noise"
        invalidation_detail = "accepted market evidence that the position thesis is broken; red P/L alone is not invalidation"
        no_add_detail = "no add while thesis_state is not ALIVE, margin is under pressure, or overlap lacks explicit operator authorization"
        emergency_detail = "margin pressure, NAV shock, missing broker truth, or gateway-outside exposure needs immediate trader review"
        invalidation_action = "REDUCE"
        emergency_action = "REDUCE"
        harvest_action = "HARVEST"
    return {
        "harvest_triggers": [
            {
                **base_ref,
                "trigger_id": "open_exposure_profit_harvest_review",
                "status": "PENDING",
                "kind": "profit_harvest_review",
                "evidence_required": harvest_detail,
                "avg_entry": avg_entry,
                "action_hint": harvest_action,
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
                "action_hint": invalidation_action,
            }
        ],
        "emergency_triggers": [
            {
                **base_ref,
                "trigger_id": "open_exposure_margin_or_nav_emergency",
                "status": "PENDING",
                "kind": "margin_or_nav_emergency",
                "evidence_required": emergency_detail,
                "action_hint": emergency_action,
            }
        ],
    }


def _is_usd_jpy_162_manual_fade(pair: str, side: str, owner: str, position: dict[str, Any]) -> bool:
    if pair != "USD_JPY" or side != "SHORT" or owner != "OPERATOR_MANUAL":
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
    ref = {
        "trade_id": trade_id,
        "units": units,
        "avg_entry": avg_entry,
        "pair": "USD_JPY",
        "side": "SHORT",
        "owner": _contract_owner(entry.get("owner")),
        "thesis": entry.get("thesis"),
        "thesis_state": _thesis_state(entry.get("thesis_state")),
    }
    return {
        "harvest_triggers": [
            {
                **ref,
                "trigger_id": "usd_jpy_162_manual_fade_profit_zone",
                "status": "PENDING",
                "kind": "manual_tp_profit_zone",
                "evidence_required": "USD_JPY trades below the manual fade average entry enough to show positive UPL outside current spread/noise; TP-only profit assistance is allowed",
                "action_hint": "HOLD",
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
                "action_hint": "HOLD",
            }
        ],
        "emergency_triggers": [
            {
                **ref,
                "trigger_id": "usd_jpy_162_manual_fade_margin_nav_emergency",
                "status": "PENDING",
                "kind": "margin_nav_emergency",
                "evidence_required": "margin/NAV pressure crosses the gateway risk cap or broker truth becomes unavailable; trader review only, no direct close",
                "action_hint": "HOLD",
            }
        ],
    }


def _contract_owner(value: Any) -> str:
    text = str(value or "UNKNOWN").strip().upper()
    return text if text in CONTRACT_OWNERS else "UNKNOWN"


def _snapshot_has_exposure(snapshot: dict[str, Any]) -> bool:
    return any(isinstance(item, dict) for item in snapshot.get("positions", []) or []) or any(
        isinstance(item, dict) for item in snapshot.get("orders", []) or []
    )


def _contract_stale_reason(validation: dict[str, Any]) -> str:
    if validation.get("status") != "VALID":
        block_codes = [
            str(issue.get("code"))
            for issue in validation.get("issues", []) or []
            if str(issue.get("severity") or "").upper() == "BLOCK"
        ]
        codes = ",".join(block_codes)
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
    if event.event_type == "UNKNOWN_ORDER":
        if prior is None:
            reasons.append("UNKNOWN_GATEWAY_OUTSIDE_ORDER")
        elif _unknown_order_broker_truth_changed(event, prior):
            reasons.append("UNKNOWN_GATEWAY_OUTSIDE_ORDER_STATE_CHANGE")
    if event.event_type == "MARGIN_PRESSURE" and (prior is None or not prior.get("margin_pressure")):
        reasons.append("MARGIN_RISK_THRESHOLD_CROSSED")
    # Price-zone parsing is defined only for market-read and technical-state
    # observations.  Safety events use descriptive state strings (for example
    # ``margin_used/nav=...``); parsing their first decimal as an FX price
    # turns ordinary risk-ratio drift into a false market displacement wake.
    material_change = (
        _event_price_zone_material_change(event, prior)
        if event.event_type in MATERIAL_ACK_EVENT_TYPES
        else {"changed": False, "material": False}
    )
    if material_change["material"]:
        reasons.append("LARGE_PRICE_DISPLACEMENT_STATE_CHANGE")
        if event.event_type == "FAILED_ACCEPTANCE":
            reasons.append("FAILED_ACCEPTANCE_PRICE_ZONE_CHANGE")
    if event.event_type == "TECHNICAL_STATE_CHANGE" and prior is not None:
        prior_details = prior.get("details") if isinstance(prior.get("details"), dict) else {}
        current_fingerprint = _technical_fingerprint(event.details)
        prior_fingerprint = _technical_fingerprint(prior_details)
        source_advanced = _technical_closed_candle_source_advanced(
            event.details,
            prior_details,
        )
        # ``False`` means both observations expose comparable closed-candle
        # watermarks and none advanced.  Any regime/family/structure drift is
        # recomputation noise, not future market information.  ``None`` keeps
        # backward compatibility with old state rows that predate watermarks.
        if (
            source_advanced is not False
            and current_fingerprint
            and current_fingerprint != prior_fingerprint
        ):
            reasons.append("TECHNICAL_STATE_CHANGE")
            if current_fingerprint.get("dominant_regime") != prior_fingerprint.get("dominant_regime"):
                reasons.append("REGIME_STATE_CHANGE")
            if current_fingerprint.get("volatility_bucket") != prior_fingerprint.get("volatility_bucket"):
                reasons.append("VOLATILITY_BUCKET_CHANGE")
            if current_fingerprint.get("family_consensus") != prior_fingerprint.get("family_consensus"):
                reasons.append("TECHNICAL_FAMILY_STATE_CHANGE")
            if current_fingerprint.get("closed_structure") != prior_fingerprint.get("closed_structure"):
                reasons.append("CLOSED_CANDLE_STRUCTURE_CHANGE")
    if event.event_type == "THEME_CONFIRMATION" and prior is not None:
        current_details = event.details if isinstance(event.details, dict) else {}
        prior_details = prior.get("details") if isinstance(prior.get("details"), dict) else {}
        current_source = _parse_utc(current_details.get("source_generated_at_utc"))
        prior_source = _parse_utc(prior_details.get("source_generated_at_utc"))
        # Theme identity stays stable while direction and thesis persist, but
        # the hourly matrix is a new point-in-time technical observation. Keep
        # the accepted receipt as baseline and reopen review for a newer matrix;
        # the normal one-hour entry throttle still bounds 30-second GPT wakes.
        if current_source is not None and (
            prior_source is None or current_source > prior_source
        ):
            reasons.append("ENTRY_SIGNAL_SOURCE_REFRESH")
    return list(dict.fromkeys(reasons))


def _technical_closed_candle_source_advanced(
    current_details: dict[str, Any],
    prior_details: dict[str, Any],
) -> bool | None:
    current = _technical_closed_candle_watermarks(current_details)
    prior = _technical_closed_candle_watermarks(prior_details)
    if not current or not prior:
        current_contract_present = "closed_candle_watermarks" in current_details
        prior_contract_present = "closed_candle_watermarks" in prior_details
        if current_contract_present and not current:
            # The producer emitted the watermark contract but could not prove
            # even one complete fast candle.  Treat that as missing current
            # market input, never as legacy permission to review derived drift.
            return False
        if current_contract_present and prior_contract_present:
            return False
        return None
    comparable = set(current).intersection(prior)
    if not comparable:
        # Both packets carry the new watermark contract but expose no common
        # fast candle.  That is an input-surface change, not proof that the
        # market advanced.  ``None`` is reserved for legacy state that has no
        # watermarks at all.
        return False
    for timeframe in comparable:
        current_ts = _parse_utc(current.get(timeframe))
        prior_ts = _parse_utc(prior.get(timeframe))
        if current_ts is not None and prior_ts is not None and current_ts > prior_ts:
            return True
    return False


def _technical_closed_candle_watermarks(details: dict[str, Any]) -> dict[str, str]:
    raw = details.get("closed_candle_watermarks") if isinstance(details, dict) else None
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for timeframe, value in raw.items():
        timestamp = _parse_utc(value)
        if timestamp is not None:
            result[str(timeframe).upper()] = timestamp.isoformat()
    return result


def _event_details_have_open_position_identity(details: dict[str, Any]) -> bool:
    """Keep ambiguous/open-exposure events outside closed-market entry suppression."""
    pending: list[Any] = [details]
    seen: set[int] = set()
    inspected_containers = 0
    while pending:
        current = pending.pop()
        if not isinstance(current, (dict, list, tuple)):
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        inspected_containers += 1
        if inspected_containers > 128:
            # Event details are expected to be a small JSON object. An
            # unexpectedly large/ambiguous shape must not silence a possible
            # open-position action while the market is closed.
            return True
        if isinstance(current, dict):
            for key, value in current.items():
                normalized_key = re.sub(
                    r"[^a-z0-9]+",
                    "_",
                    str(key).strip().lower(),
                ).strip("_")
                if normalized_key in {"open_exposure", "has_open_exposure", "hasopenexposure"}:
                    if _truthy(value):
                        return True
                identity_key = normalized_key in OPEN_POSITION_IDENTITY_KEYS or (
                    normalized_key.endswith("id")
                    and ("trade" in normalized_key or "position" in normalized_key)
                )
                if identity_key and (
                    isinstance(value, (str, int, float))
                    and not isinstance(value, bool)
                    and str(value).strip()
                ):
                    return True
                if isinstance(value, (dict, list, tuple)):
                    pending.append(value)
        else:
            pending.extend(current)
    return False


def _event_throttle_seconds(event: GuardianEvent) -> int:
    if event.action_hint in {"TRADE", "ADD"} or event.recommended_review_type in {"ENTRY_REVIEW", "ADD_REVIEW"}:
        return int(os.environ.get("QR_GUARDIAN_FRESH_ACTION_THROTTLE_SECONDS", DEFAULT_FRESH_ACTION_THROTTLE_SECONDS))
    return int(os.environ.get("QR_GUARDIAN_EVENT_THROTTLE_SECONDS", DEFAULT_THROTTLE_SECONDS))


def _event_bypasses_throttle(event: GuardianEvent, reasons: list[str]) -> bool:
    return (
        event.event_type == "HARVEST_ZONE"
        or (event.severity == "P0" and event.event_type != "UNKNOWN_ORDER")
        or "UNKNOWN_GATEWAY_OUTSIDE_ORDER" in reasons
        or "UNKNOWN_GATEWAY_OUTSIDE_ORDER_STATE_CHANGE" in reasons
        or (
            event.event_type != "TECHNICAL_STATE_CHANGE"
            and "LARGE_PRICE_DISPLACEMENT_STATE_CHANGE" in reasons
        )
        or (
            event.event_type != "TECHNICAL_STATE_CHANGE"
            and "FAILED_ACCEPTANCE_PRICE_ZONE_CHANGE" in reasons
        )
        or "SEVERITY_INCREASE" in reasons
        or any(reason.startswith("THESIS_ALIVE_TO_") for reason in reasons)
    )


def _event_state_snapshot(event: GuardianEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "pair": event.pair,
        "direction": event.direction,
        "severity": event.severity,
        "thesis_state": event.thesis_state,
        "price_zone": event.price_zone,
        "details": event.details,
    }


def _dispatcher_acknowledged_event(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, dict) or record.get("receipt_written") is not True:
        return None
    nested = record.get("selected_event") if isinstance(record.get("selected_event"), dict) else {}
    event_id = nested.get("event_id") or record.get("event_id")
    price_zone = nested.get("price_zone") or record.get("price_zone")
    details = nested.get("details") if isinstance(nested.get("details"), dict) else record.get("details")
    if not event_id and not price_zone and not isinstance(details, dict):
        return None
    return {
        "event_id": event_id,
        "event_type": nested.get("event_type") or record.get("event_type"),
        "pair": nested.get("pair") or record.get("pair"),
        "direction": nested.get("direction") or record.get("direction"),
        "severity": nested.get("severity") or record.get("severity"),
        "thesis_state": nested.get("thesis_state") or record.get("thesis_state"),
        "price_zone": price_zone,
        "details": details if isinstance(details, dict) else {},
        "acknowledged_at_utc": record.get("last_reviewed_at_utc"),
    }


def _material_reference_event(
    prior: dict[str, Any] | None,
    *,
    acknowledged: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if acknowledged is not None:
        return acknowledged
    if not isinstance(prior, dict):
        return None
    reference = prior.get("material_reference")
    if isinstance(reference, dict):
        return reference
    return prior


def _unknown_order_broker_truth_changed(event: GuardianEvent, prior: dict[str, Any]) -> bool:
    current = _unknown_order_truth_fingerprint(event.to_payload())
    previous = _unknown_order_truth_fingerprint(prior)
    return bool(current and previous and current != previous)


def _unknown_order_truth_fingerprint(payload: dict[str, Any]) -> str:
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    trade_id = str(details.get("trade_id") or "").strip()
    order_id = str(details.get("order_id") or "").strip()
    owner = str(details.get("owner") or payload.get("owner") or "").strip().lower()
    pair = _pair(payload.get("pair")) or "UNKNOWN"
    direction = _direction_from_text(payload.get("direction") or payload.get("side")) or "UNKNOWN"
    price_zone = str(payload.get("price_zone") or "").strip()
    return "|".join([pair, direction, trade_id or order_id or "NO_ID", owner or "unknown", price_zone])


def _event_price_zone_material_change(event: GuardianEvent, prior: dict[str, Any] | None) -> dict[str, Any]:
    if prior is None:
        return {"changed": False, "material": False}
    dynamic = _technical_price_material_change(event, prior)
    if dynamic is not None:
        return dynamic
    current_zone = str(event.price_zone or "").strip()
    previous_zone = str(prior.get("price_zone") or "").strip()
    if not current_zone or not previous_zone or current_zone == previous_zone:
        return {"changed": False, "material": False, "current_price_zone": current_zone, "previous_price_zone": previous_zone}
    pair = _pair(event.pair) or "UNKNOWN"
    threshold_pips = _price_zone_material_threshold_pips(pair)
    current_distance = _price_zone_distance_pips(current_zone)
    previous_distance = _price_zone_distance_pips(previous_zone)
    if current_distance is not None and previous_distance is not None:
        delta_pips = abs(current_distance - previous_distance)
        return {
            "changed": True,
            "material": delta_pips >= threshold_pips,
            "delta_pips": delta_pips,
            "threshold_pips": threshold_pips,
            "current_price_zone": current_zone,
            "previous_price_zone": previous_zone,
            "basis": "distance_pips",
        }
    current_price = _price_zone_anchor_price(current_zone)
    previous_price = _price_zone_anchor_price(previous_zone)
    if current_price is None or previous_price is None:
        return {
            "changed": True,
            "material": False,
            "threshold_pips": threshold_pips,
            "current_price_zone": current_zone,
            "previous_price_zone": previous_zone,
            "basis": "text_only",
        }
    delta_pips = abs(current_price - previous_price) * instrument_pip_factor(pair)
    return {
        "changed": True,
        "material": delta_pips >= threshold_pips,
        "delta_pips": delta_pips,
        "threshold_pips": threshold_pips,
        "current_price_zone": current_zone,
        "previous_price_zone": previous_zone,
        "basis": "anchor_price",
    }


def _technical_price_material_change(
    event: GuardianEvent,
    prior: dict[str, Any],
) -> dict[str, Any] | None:
    if event.event_type != "TECHNICAL_STATE_CHANGE":
        return None
    current_details = event.details if isinstance(event.details, dict) else {}
    prior_details = prior.get("details") if isinstance(prior.get("details"), dict) else {}
    current_mid = _float(current_details.get("mid"))
    previous_mid = _float(prior_details.get("mid"))
    if current_mid is None or previous_mid is None:
        return {
            "changed": current_details.get("material_fingerprint") != prior_details.get("material_fingerprint"),
            "material": False,
            "basis": "technical_mid_missing",
        }
    spread = _float(current_details.get("live_spread_pips"))
    atr = _float(current_details.get("m5_atr_pips"))
    threshold = max(
        2.0 * (spread if spread is not None else NORMAL_SPREAD_PIPS.get(event.pair, 1.0)),
        atr if atr is not None else 0.0,
        1.0,
    )
    delta = abs(current_mid - previous_mid) * instrument_pip_factor(event.pair)
    return {
        "changed": delta > 0.0,
        "material": delta >= threshold,
        "delta_pips": delta,
        "threshold_pips": threshold,
        "current_mid": current_mid,
        "previous_mid": previous_mid,
        "live_spread_pips": spread,
        "m5_atr_pips": atr,
        "basis": "max_2x_live_spread_m5_atr",
    }


def _technical_fingerprint(details: dict[str, Any]) -> dict[str, Any]:
    raw = details.get("material_fingerprint") if isinstance(details, dict) else None
    if not isinstance(raw, dict):
        return {}
    families = raw.get("family_consensus") if isinstance(raw.get("family_consensus"), dict) else {}
    return {
        "dominant_regime": str(raw.get("dominant_regime") or "UNKNOWN").upper(),
        "volatility_bucket": str(raw.get("volatility_bucket") or "UNKNOWN").upper(),
        "family_consensus": {
            str(key): str(value).upper()
            for key, value in sorted(families.items())
        },
        "closed_structure": str(raw.get("closed_structure") or "NONE"),
    }


def _price_zone_material_threshold_pips(pair: str) -> float:
    normal = NORMAL_SPREAD_PIPS.get(_pair(pair) or pair, 1.0)
    return max(float(normal) * 2.0, 1.0)


def _price_zone_distance_pips(zone: str) -> float | None:
    match = re.search(r"\bdistance_pips\s*=\s*(-?\d+(?:\.\d+)?)", zone)
    if not match:
        return None
    return _float(match.group(1))


def _price_zone_anchor_price(zone: str) -> float | None:
    lower = zone.lower()
    if "percentile=" in lower or "sigma=" in lower or "spread_pips=" in lower:
        return None
    matches = re.findall(r"(?<![A-Za-z_])(-?\d{1,3}(?:\.\d+)?)(?![A-Za-z_])", zone)
    if not matches:
        return None
    return _float(matches[0])


def _suppressed_price_change_diagnostics(event: GuardianEvent, prior: dict[str, Any] | None) -> list[dict[str, Any]]:
    change = _event_price_zone_material_change(event, prior)
    if not change.get("changed"):
        return []
    return [
        {
            "code": "EVENT_SUPPRESSED_WITH_PRICE_CHANGE",
            "event_id": event.event_id,
            "dedupe_key": event.dedupe_key,
            "pair": event.pair,
            "event_type": event.event_type,
            "material": bool(change.get("material")),
            "details": change,
        }
    ]


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
    if event is not None and event.action_hint.upper() not in {"TRADE", "ADD"}:
        issues.append(
            _issue(
                "GUARDIAN_ACTION_EVENT_DOES_NOT_AUTHORIZE_ENTRY",
                "TRADE/ADD requires selected event action_hint TRADE or ADD",
            )
        )
    if event is not None and str(event.direction or "").upper() not in {"LONG", "SHORT"}:
        issues.append(
            _issue(
                "GUARDIAN_ACTION_EVENT_DIRECTION_REQUIRED",
                "TRADE/ADD requires selected event direction LONG or SHORT",
            )
        )
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
    lines.extend(["", "## Suppressed Events", ""])
    for event in escalation.get("suppressed_events", []) or []:
        lines.append(
            f"- suppressed `{event.get('event_type')}` `{event.get('pair')}` "
            f"reason=`{event.get('suppressed_reason')}` price_zone=`{event.get('price_zone')}`"
        )
    if not escalation.get("suppressed_events"):
        lines.append("- none")
    lines.extend(["", "## Diagnostics", ""])
    trigger_validation = events_payload.get("trigger_contract") if isinstance(events_payload.get("trigger_contract"), dict) else {}
    for issue in trigger_validation.get("issues", []) or []:
        if str(issue.get("code") or "").upper() in {
            "TRIGGER_CONTRACT_EMPTY_FOR_ACTIVE_PAIR",
            "WATCH_ONLY_NO_TRIGGER_CONTRACT",
            "CONTRACT_OPEN_EXPOSURE_MISSING",
        }:
            lines.append(f"- `{issue.get('severity')}` `{issue.get('code')}` {issue.get('message')}")
    for diagnostic in escalation.get("diagnostics", []) or []:
        lines.append(
            f"- `{diagnostic.get('code')}` `{diagnostic.get('pair')}` `{diagnostic.get('event_type')}` "
            f"material=`{diagnostic.get('material')}`"
        )
    if not any(
        str(issue.get("code") or "").upper()
        in {"TRIGGER_CONTRACT_EMPTY_FOR_ACTIVE_PAIR", "WATCH_ONLY_NO_TRIGGER_CONTRACT", "CONTRACT_OPEN_EXPOSURE_MISSING"}
        for issue in trigger_validation.get("issues", []) or []
    ) and not escalation.get("diagnostics"):
        lines.append("- none")
    usd_jpy_events = [
        event for event in events_payload.get("events", []) or [] if _pair(event.get("pair")) == "USD_JPY"
    ]
    if usd_jpy_events:
        open_usd_jpy = any(
            event.get("event_type") == "UNKNOWN_ORDER" and _pair(event.get("pair")) == "USD_JPY"
            for event in events_payload.get("events", []) or []
        )
        usd_jpy_trigger_issue = any(
            "USD_JPY" in str(issue.get("message") or "")
            and str(issue.get("code") or "").upper()
            in {"TRIGGER_CONTRACT_EMPTY_FOR_ACTIVE_PAIR", "WATCH_ONLY_NO_TRIGGER_CONTRACT", "CONTRACT_OPEN_EXPOSURE_MISSING"}
            for issue in trigger_validation.get("issues", []) or []
        )
        usd_jpy_suppressed = [
            event
            for event in escalation.get("suppressed_events", []) or []
            if _pair(event.get("pair")) == "USD_JPY"
        ]
        lines.extend(
            [
                "",
                "## USD_JPY Reaction Classification",
                "",
                "- Classification: `PARTIAL_REACTION` (not `LAUNCHD_FAILURE`).",
                "- Launchd/runtime pass ran far enough to generate this guardian event report.",
                "- USD_JPY was detected in guardian events.",
                f"- No USD_JPY open exposure existed in this router pass: `{not open_usd_jpy}`.",
                (
                    "- USD_JPY trigger contract had no actionable machine-readable triggers: "
                    f"`{usd_jpy_trigger_issue}`."
                ),
                (
                    "- USD_JPY events were throttled or classified no-state-change: "
                    f"`{len(usd_jpy_suppressed)}`."
                ),
                "- Normal TRADE/ADD routing remains blocked unless guardian/operator-review gates explicitly clear.",
                "- Therefore no broker-side action is implied by this report.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, "\n".join(lines) + "\n")


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
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _load_json_with_sha256(path: Path | str | None) -> tuple[dict[str, Any], str]:
    if path is None:
        return {}, ""
    try:
        raw = Path(path).read_bytes()
    except (OSError, ValueError):
        return {}, ""
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return {}, digest
    return (payload if isinstance(payload, dict) else {}), digest


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp_path.unlink(missing_ok=True)


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
        "EVENT_SUPPRESSED_WITH_PRICE_CHANGE": "ENTRY_REVIEW",
        "TRIGGER_CONTRACT_EMPTY_FOR_ACTIVE_PAIR": "THESIS_REVIEW",
        "WATCH_ONLY_NO_TRIGGER_CONTRACT": "THESIS_REVIEW",
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
        "EVENT_SUPPRESSED_WITH_PRICE_CHANGE": "HOLD",
        "TRIGGER_CONTRACT_EMPTY_FOR_ACTIVE_PAIR": "HOLD",
        "WATCH_ONLY_NO_TRIGGER_CONTRACT": "HOLD",
    }.get(event_type, "TRADE")


def _issue_pair(issue: dict[str, Any]) -> str | None:
    text = str(issue.get("message") or "")
    match = re.search(r"pair=([A-Z]{3}_[A-Z]{3})", text)
    return match.group(1) if match else None


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


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


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


def _parse_aware_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
