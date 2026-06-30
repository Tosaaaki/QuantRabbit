from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.guardian_events import (
    build_guardian_trigger_contract,
    detect_guardian_events,
    evaluate_guardian_escalation,
    review_guardian_action_receipt,
    run_guardian_event_router,
    validate_guardian_trigger_contract,
    write_guardian_trigger_contract_report,
)


NOW = datetime(2026, 6, 30, 3, 30, tzinfo=timezone.utc)


class GuardianEventRouterTest(unittest.TestCase):
    def test_no_event_no_wake(self) -> None:
        events = detect_guardian_events(inputs={"snapshot": _snapshot()}, now=NOW)
        escalation, _ = evaluate_guardian_escalation(events=events, previous_state={}, now=NOW)

        self.assertEqual(events, [])
        self.assertFalse(escalation["wake_gpt"])

    def test_repeated_same_event_is_throttled(self) -> None:
        events = _events_from_chart(
            {
                "event_type": "FAILED_ACCEPTANCE",
                "thesis": "major figure rejection",
                "price_zone": "EUR_USD 1.1700 rejection",
                "severity": "P1",
                "action_hint": "TRADE",
            }
        )
        first, state = evaluate_guardian_escalation(events=events, previous_state={}, now=NOW)

        second, _ = evaluate_guardian_escalation(
            events=events,
            previous_state=state,
            now=NOW + timedelta(minutes=5),
        )

        self.assertTrue(first["wake_gpt"])
        self.assertFalse(second["wake_gpt"])
        self.assertEqual(second["suppressed_events"][0]["suppressed_reason"], "THROTTLED")

    def test_p0_severity_increase_bypasses_throttle(self) -> None:
        p1_events = _events_from_chart(
            {
                "event_type": "THESIS_INVALIDATION",
                "thesis": "EUR_USD long thesis",
                "price_zone": "M15 support cracked",
                "severity": "P1",
                "action_hint": "REDUCE",
                "thesis_state": "WOUNDED",
            }
        )
        _, state = evaluate_guardian_escalation(events=p1_events, previous_state={}, now=NOW)
        p0_events = _events_from_chart(
            {
                "event_type": "THESIS_INVALIDATION",
                "thesis": "EUR_USD long thesis",
                "price_zone": "H4 close-confirmed break",
                "severity": "P0",
                "action_hint": "REDUCE",
                "thesis_state": "INVALIDATED",
            }
        )

        escalation, _ = evaluate_guardian_escalation(
            events=p0_events,
            previous_state=state,
            now=NOW + timedelta(minutes=5),
        )

        self.assertTrue(escalation["wake_gpt"])
        self.assertIn("SEVERITY_INCREASE", escalation["wake_reason_codes"])

    def test_harvest_event_wakes(self) -> None:
        events = detect_guardian_events(
            inputs={
                "order_intents": {
                    "results": [
                        {
                            "lane_id": "range:EUR_USD:LONG:HARVEST",
                            "status": "LIVE_READY",
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "thesis": "range harvest long",
                                "entry": 1.171,
                                "metadata": {
                                    "opportunity_mode": "HARVEST",
                                    "tp_target_intent": "HARVEST",
                                    "harvest_zone": "upper range rail",
                                },
                            },
                        }
                    ]
                }
            },
            now=NOW,
        )
        escalation, _ = evaluate_guardian_escalation(events=events, previous_state={}, now=NOW)

        self.assertEqual(events[0].event_type, "HARVEST_ZONE")
        self.assertTrue(escalation["wake_gpt"])
        self.assertIn("PRICE_ENTERED_HARVEST_ZONE", escalation["wake_reason_codes"])

    def test_dry_run_harvest_geometry_does_not_wake_as_harvest_zone(self) -> None:
        events = detect_guardian_events(
            inputs={
                "order_intents": {
                    "results": [
                        {
                            "lane_id": "range:EUR_USD:LONG:DRY_RUN",
                            "status": "DRY_RUN_BLOCKED",
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "thesis": "range target geometry",
                                "entry": 1.171,
                                "metadata": {
                                    "opportunity_mode": "HARVEST",
                                    "opportunity_mode_reason": "tp_target_intent=HARVEST",
                                    "tp_target_intent": "HARVEST",
                                },
                            },
                        }
                    ]
                }
            },
            now=NOW,
        )
        escalation, _ = evaluate_guardian_escalation(events=events, previous_state={}, now=NOW)

        self.assertEqual([event.event_type for event in events], [])
        self.assertFalse(escalation["wake_gpt"])

    def test_valid_contract_schema_passes(self) -> None:
        validation = validate_guardian_trigger_contract(_contract(), now=NOW)

        self.assertEqual(validation["status"], "VALID")
        self.assertEqual(validation["entry_count"], 1)

    def test_stale_contract_with_exposure_wakes_gpt(self) -> None:
        contract = _contract(generated_at=NOW - timedelta(hours=2))
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "units": 1000,
                    "owner": "trader",
                    "trade_id": "1",
                }
            ]
        )

        events = detect_guardian_events(inputs={"snapshot": snapshot, "trigger_contract": contract}, now=NOW)
        escalation, _ = evaluate_guardian_escalation(events=events, previous_state={}, now=NOW)

        self.assertIn("CONTRACT_STALE", {event.event_type for event in events})
        self.assertTrue(escalation["wake_gpt"])

    def test_contract_harvest_trigger_wakes_gpt(self) -> None:
        contract = _contract(
            entry_overrides={
                "harvest_triggers": [{"trigger_id": "upper_rail", "status": "FIRED", "evidence": "M5 upper rail touched"}]
            }
        )

        events = detect_guardian_events(inputs={"snapshot": _snapshot(), "trigger_contract": contract}, now=NOW)
        escalation, _ = evaluate_guardian_escalation(events=events, previous_state={}, now=NOW)

        self.assertEqual(events[0].event_type, "CONTRACT_HARVEST_TRIGGER")
        self.assertTrue(escalation["wake_gpt"])

    def test_contract_invalidation_trigger_wakes_gpt(self) -> None:
        contract = _contract(
            entry_overrides={
                "invalidation_triggers": [
                    {"trigger_id": "breakdown", "fired": True, "evidence": "accepted trade below thesis support"}
                ]
            }
        )

        events = detect_guardian_events(inputs={"snapshot": _snapshot(), "trigger_contract": contract}, now=NOW)
        escalation, _ = evaluate_guardian_escalation(events=events, previous_state={}, now=NOW)

        self.assertEqual(events[0].event_type, "CONTRACT_INVALIDATION_TRIGGER")
        self.assertEqual(events[0].thesis_state, "INVALIDATED")
        self.assertTrue(escalation["wake_gpt"])

    def test_fixed_safety_triggers_work_without_contract(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "units": 1000,
                    "owner": "unknown",
                    "trade_id": "manual-1",
                }
            ]
        )

        events = detect_guardian_events(inputs={"snapshot": snapshot}, now=NOW)
        event_types = {event.event_type for event in events}

        self.assertIn("UNKNOWN_ORDER", event_types)
        self.assertIn("CONTRACT_STALE", event_types)

    def test_missing_contract_does_not_invent_market_triggers(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "units": 1000,
                    "owner": "trader",
                    "trade_id": "1",
                }
            ]
        )

        events = detect_guardian_events(inputs={"snapshot": snapshot}, now=NOW)
        event_types = {event.event_type for event in events}

        self.assertEqual(event_types, {"CONTRACT_STALE"})

    def test_trigger_contract_builder_preserves_existing_trader_triggers(self) -> None:
        existing = _contract(
            entry_overrides={
                "harvest_triggers": [{"trigger_id": "keep_me", "metric": "mid", "operator": ">=", "value": 1.18}]
            }
        )
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "units": 1000,
                    "owner": "trader",
                    "thesis": "range long",
                    "trade_id": "1",
                }
            ]
        )

        contract = build_guardian_trigger_contract(snapshot=snapshot, order_intents={}, existing_contract=existing, now=NOW)

        self.assertEqual(contract["entries"][0]["harvest_triggers"], existing["entries"][0]["harvest_triggers"])

    def test_open_exposure_gets_non_empty_trigger_arrays(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "units": 1000,
                    "owner": "trader",
                    "thesis": "range long",
                    "trade_id": "trade-1",
                    "price": 1.171,
                }
            ]
        )

        contract = build_guardian_trigger_contract(snapshot=snapshot, order_intents={}, existing_contract={}, now=NOW)
        entry = contract["entries"][0]

        for bucket in (
            "harvest_triggers",
            "no_add_triggers",
            "wounded_triggers",
            "invalidation_triggers",
            "emergency_triggers",
        ):
            self.assertTrue(entry[bucket], bucket)
        self.assertEqual(entry["trade_id"], "trade-1")
        self.assertEqual(entry["avg_entry"], 1.171)
        self.assertEqual(entry["units"], 1000)

    def test_open_exposure_avg_entry_populates_from_broker_entry_price(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "units": 1000,
                    "owner": "trader",
                    "thesis": "system USD_JPY short",
                    "thesis_state": "ALIVE",
                    "trade_id": "trade-entry",
                    "entry_price": 162.157,
                }
            ]
        )

        contract = build_guardian_trigger_contract(snapshot=snapshot, order_intents={}, existing_contract={}, now=NOW)
        entry = contract["entries"][0]

        self.assertEqual(entry["avg_entry"], 162.157)
        for bucket in (
            "harvest_triggers",
            "no_add_triggers",
            "wounded_triggers",
            "invalidation_triggers",
            "emergency_triggers",
        ):
            trigger = entry[bucket][0]
            self.assertEqual(trigger["trade_id"], "trade-entry")
            self.assertEqual(trigger["units"], 1000)
            self.assertEqual(trigger["avg_entry"], 162.157)
            self.assertEqual(trigger["pair"], "USD_JPY")
            self.assertEqual(trigger["side"], "SHORT")
            self.assertEqual(trigger["owner"], "SYSTEM")
            self.assertEqual(trigger["thesis"], "system USD_JPY short")
            self.assertEqual(trigger["thesis_state"], "ALIVE")

    def test_unknown_usd_jpy_owner_remains_unresolved_without_evidence(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "units": 1000,
                    "owner": "unknown",
                    "thesis": "operator manual USD_JPY 162 fade",
                    "trade_id": "472933",
                    "entry_price": 162.157,
                    "raw": {"currentUnits": "-1000", "price": "162.157"},
                }
            ]
        )

        contract = build_guardian_trigger_contract(snapshot=snapshot, order_intents={}, existing_contract={}, now=NOW)
        entry = contract["entries"][0]

        self.assertEqual(entry["owner"], "UNKNOWN")
        self.assertEqual(entry["avg_entry"], 162.157)
        self.assertEqual(entry["ownership_audit"]["status"], "UNKNOWN_NEEDS_OPERATOR_CONFIRM")
        self.assertTrue(entry["ownership_audit"]["unresolved"])
        self.assertNotIn("usd_jpy_162_manual_fade", json.dumps(entry))

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "guardian_trigger_contract_report.md"
            write_guardian_trigger_contract_report(report_path, contract, validate_guardian_trigger_contract(contract, now=NOW))
            report = report_path.read_text()
        self.assertIn("UNKNOWN_NEEDS_OPERATOR_CONFIRM", report)
        self.assertIn("trade_id=`472933`", report)

    def test_unknown_owner_with_gateway_lane_evidence_maps_to_system(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "units": 1000,
                    "owner": "unknown",
                    "thesis": "system lane long",
                    "trade_id": "system-1",
                    "entry_price": 1.171,
                    "lane_id": "range:EUR_USD:LONG:system-lane",
                }
            ]
        )

        contract = build_guardian_trigger_contract(snapshot=snapshot, order_intents={}, existing_contract={}, now=NOW)
        entry = contract["entries"][0]

        self.assertEqual(entry["owner"], "SYSTEM")
        self.assertEqual(entry["ownership_audit"]["status"], "SYSTEM")
        self.assertFalse(entry["ownership_audit"]["unresolved"])

    def test_manual_exposure_maps_by_trade_id(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "units": 1000,
                    "owner": "operator_manual",
                    "thesis": "operator manual USD_JPY 162 fade",
                    "trade_id": "472909",
                    "price": 162.157,
                },
                {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "units": 5000,
                    "owner": "operator_manual",
                    "thesis": "operator manual USD_JPY 162 fade",
                    "trade_id": "472913",
                    "price": 162.2,
                },
            ]
        )

        contract = build_guardian_trigger_contract(snapshot=snapshot, order_intents={}, existing_contract={}, now=NOW)

        self.assertEqual({entry["trade_id"] for entry in contract["entries"]}, {"472909", "472913"})
        self.assertEqual(len(contract["entries"]), 2)

    def test_operator_manual_usd_jpy_162_fade_gets_concrete_triggers(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "units": 22000,
                    "owner": "operator_manual",
                    "thesis": "operator manual USD_JPY 162 fade",
                    "thesis_state": "WOUNDED",
                    "trade_id": "manual-162",
                    "price": 162.157,
                }
            ]
        )

        contract = build_guardian_trigger_contract(snapshot=snapshot, order_intents={}, existing_contract={}, now=NOW)
        entry = contract["entries"][0]

        self.assertEqual(entry["owner"], "OPERATOR_MANUAL")
        self.assertEqual(entry["trade_id"], "manual-162")
        self.assertEqual(entry["thesis_state"], "WOUNDED")
        joined = json.dumps(entry, ensure_ascii=False)
        self.assertIn("162.00", joined)
        self.assertIn("cross-JPY confirmation", joined)
        self.assertIn("TP-only profit assistance", joined)
        self.assertIn("no fresh bot USD_JPY", joined)
        for bucket in (
            "harvest_triggers",
            "no_add_triggers",
            "wounded_triggers",
            "invalidation_triggers",
            "emergency_triggers",
        ):
            trigger = entry[bucket][0]
            self.assertEqual(trigger["trade_id"], "manual-162")
            self.assertEqual(trigger["units"], 22000)
            self.assertEqual(trigger["avg_entry"], 162.157)
            self.assertEqual(trigger["pair"], "USD_JPY")
            self.assertEqual(trigger["side"], "SHORT")
            self.assertEqual(trigger["owner"], "OPERATOR_MANUAL")
            self.assertEqual(trigger["thesis"], "operator manual USD_JPY 162 fade")
            self.assertEqual(trigger["thesis_state"], "WOUNDED")

    def test_expired_contract_deadline_emits_contract_stale_with_exposure(self) -> None:
        contract = _contract(entry_overrides={"next_review_deadline_utc": (NOW - timedelta(minutes=1)).isoformat()})
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "units": 1000,
                    "owner": "trader",
                    "trade_id": "1",
                }
            ]
        )

        validation = validate_guardian_trigger_contract(contract, now=NOW)
        events = detect_guardian_events(inputs={"snapshot": snapshot, "trigger_contract": contract}, now=NOW)

        self.assertEqual(validation["status"], "INVALID")
        self.assertTrue(validation["stale"])
        self.assertIn("CONTRACT_STALE", {event.event_type for event in events})

    def test_open_exposure_empty_required_triggers_emit_contract_stale(self) -> None:
        contract = _contract(
            entry_overrides={
                "trade_id": "manual-weak",
                "harvest_triggers": [],
                "invalidation_triggers": [],
                "emergency_triggers": [],
            }
        )
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "units": 1000,
                    "owner": "operator_manual",
                    "trade_id": "manual-weak",
                }
            ]
        )

        validation = validate_guardian_trigger_contract(contract, now=NOW)
        events = detect_guardian_events(inputs={"snapshot": snapshot, "trigger_contract": contract}, now=NOW)

        self.assertEqual(validation["status"], "INVALID")
        self.assertIn(
            "CONTRACT_ENTRY_OPEN_TRIGGER_EMPTY",
            {issue["code"] for issue in validation["issues"]},
        )
        self.assertIn("CONTRACT_STALE", {event.event_type for event in events})

    def test_generated_contract_deadline_is_not_expired(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "units": 1000,
                    "owner": "trader",
                    "trade_id": "1",
                }
            ]
        )

        contract = build_guardian_trigger_contract(snapshot=snapshot, order_intents={}, existing_contract={}, now=NOW)
        deadline = datetime.fromisoformat(contract["entries"][0]["next_review_deadline_utc"])

        self.assertGreater(deadline, NOW)

    def test_parse_failure_map_reaches_next_trader_even_after_later_dispatch_status(self) -> None:
        dispatcher_state = {
            "last_status": "BROKER_SNAPSHOT_STALE",
            "last_result": {
                "status": "BROKER_SNAPSHOT_STALE",
                "selected_event": {
                    "event_id": "event-later",
                    "pair": "USD_CAD",
                    "event_type": "SPREAD_ANOMALY",
                    "thesis": "spread anomaly safety trigger",
                    "dedupe_key": "USD_CAD|SPREAD_ANOMALY_SAFETY_TRIGGER|SPREAD_ANOMALY|HOLD",
                },
            },
            "parse_failures": {
                "EUR_JPY|SPREAD_ANOMALY_SAFETY_TRIGGER|SPREAD_ANOMALY|HOLD": {
                    "event_id": "event-failed",
                    "dedupe_key": "EUR_JPY|SPREAD_ANOMALY_SAFETY_TRIGGER|SPREAD_ANOMALY|HOLD",
                    "pair": "EUR_JPY",
                    "direction": None,
                    "event_type": "SPREAD_ANOMALY",
                    "thesis": "spread anomaly safety trigger",
                    "severity": "P1",
                    "last_error": "CODEX_NO_JSON_RECEIPT",
                    "consecutive_failures": 1,
                }
            },
        }

        events = detect_guardian_events(inputs={"wake_dispatcher_state": dispatcher_state}, now=NOW)
        escalation, _ = evaluate_guardian_escalation(events=events, previous_state={}, now=NOW)

        wake_parse_failures = [event for event in events if event.event_type == "WAKE_PARSE_FAILURE"]
        self.assertEqual(len(wake_parse_failures), 1)
        self.assertEqual(wake_parse_failures[0].pair, "EUR_JPY")
        self.assertIn("CODEX_NO_JSON_RECEIPT", wake_parse_failures[0].price_zone)
        self.assertTrue(escalation["wake_gpt"])

    def test_parse_failure_last_result_and_map_do_not_duplicate_event(self) -> None:
        failure = {
            "event_id": "event-failed",
            "dedupe_key": "EUR_JPY|SPREAD_ANOMALY_SAFETY_TRIGGER|SPREAD_ANOMALY|HOLD",
            "pair": "EUR_JPY",
            "event_type": "SPREAD_ANOMALY",
            "thesis": "spread anomaly safety trigger",
            "last_error": "CODEX_EMPTY_OUTPUT",
            "consecutive_failures": 1,
        }
        dispatcher_state = {
            "last_status": "PARSE_FAILED",
            "last_result": {
                "status": "PARSE_FAILED",
                "selected_event": {
                    "event_id": "event-failed",
                    "pair": "EUR_JPY",
                    "event_type": "SPREAD_ANOMALY",
                    "thesis": "spread anomaly safety trigger",
                },
                "parse_failure": failure,
            },
            "parse_failures": {failure["dedupe_key"]: failure},
        }

        events = detect_guardian_events(inputs={"wake_dispatcher_state": dispatcher_state}, now=NOW)

        self.assertEqual(len([event for event in events if event.event_type == "WAKE_PARSE_FAILURE"]), 1)

    def test_failed_acceptance_event_can_wake(self) -> None:
        events = _events_from_chart(
            {
                "event_type": "FAILED_ACCEPTANCE",
                "thesis": "failed acceptance below major figure",
                "price_zone": "GBP_JPY 198.00 rejection",
                "severity": "P1",
                "action_hint": "TRADE",
            },
            pair="GBP_JPY",
        )
        escalation, _ = evaluate_guardian_escalation(events=events, previous_state={}, now=NOW)

        self.assertEqual(events[0].event_type, "FAILED_ACCEPTANCE")
        self.assertEqual(events[0].pair, "GBP_JPY")
        self.assertTrue(escalation["wake_gpt"])

    def test_same_event_framework_handles_multiple_pairs(self) -> None:
        events = detect_guardian_events(
            inputs={
                "pair_charts": {
                    "charts": [
                        {
                            "pair": "EUR_USD",
                            "guardian_events": [
                                {
                                    "event_type": "RANGE_RAIL_TOUCH",
                                    "direction": "LONG",
                                    "thesis": "EUR lower rail",
                                    "price_zone": "EUR_USD lower rail",
                                    "severity": "P2",
                                }
                            ],
                        },
                        {
                            "pair": "GBP_JPY",
                            "guardian_events": [
                                {
                                    "event_type": "SQUEEZE_RELEASE",
                                    "direction": "SHORT",
                                    "thesis": "GBP_JPY squeeze",
                                    "price_zone": "M5 squeeze release",
                                    "severity": "P1",
                                }
                            ],
                        },
                    ]
                }
            },
            now=NOW,
        )

        self.assertEqual({event.pair for event in events}, {"EUR_USD", "GBP_JPY"})
        self.assertEqual({event.event_type for event in events}, {"RANGE_RAIL_TOUCH", "SQUEEZE_RELEASE"})

    def test_gpt_action_without_new_information_is_rejected(self) -> None:
        event = _events_from_chart(
            {
                "event_type": "FAILED_ACCEPTANCE",
                "thesis": "major figure rejection",
                "price_zone": "EUR_USD 1.1700 rejection",
                "severity": "P1",
                "action_hint": "TRADE",
            }
        )[0]

        review = review_guardian_action_receipt(
            {
                "action": "TRADE",
                "new_information": False,
                "event_id": event.event_id,
                "thesis_state": "ALIVE",
                "reason": "same idea as the last cycle",
                "invalidation": "accepted break above zone",
                "harvest_trigger": "upper rail",
                "gateway_required": True,
            },
            events=[event],
            now=NOW,
        )

        self.assertEqual(review["status"], "REJECTED")
        self.assertIn("GUARDIAN_ACTION_REQUIRES_NEW_INFORMATION", _issue_codes(review))

    def test_live_actions_require_gateway_required_true(self) -> None:
        event = _events_from_chart(
            {
                "event_type": "SQUEEZE_RELEASE",
                "thesis": "squeeze release",
                "price_zone": "M5 squeeze release",
                "severity": "P1",
                "action_hint": "ADD",
            }
        )[0]

        review = review_guardian_action_receipt(
            {
                "action": "ADD",
                "new_information": True,
                "event_id": event.event_id,
                "thesis_state": "ALIVE",
                "reason": "fresh squeeze release",
                "invalidation": "squeeze failed back inside range",
                "harvest_trigger": "first rail touch",
                "gateway_required": False,
            },
            events=[event],
            now=NOW,
        )

        self.assertEqual(review["status"], "REJECTED")
        self.assertIn("GUARDIAN_ACTION_GATEWAY_REQUIRED", _issue_codes(review))

    def test_cli_writes_required_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            pair_charts = root / "pair_charts.json"
            order_intents = root / "order_intents.json"
            position_management = root / "position_management.json"
            thesis_evolution = root / "thesis_evolution.json"
            forecast_persistence = root / "forecast_persistence.json"
            market_context = root / "market_context.json"
            snapshot.write_text(json.dumps(_snapshot()))
            pair_charts.write_text(
                json.dumps(
                    {
                        "charts": [
                            {
                                "pair": "EUR_USD",
                                "guardian_events": [
                                    {
                                        "event_type": "FAILED_ACCEPTANCE",
                                        "thesis": "major figure rejection",
                                        "price_zone": "EUR_USD 1.1700 rejection",
                                        "severity": "P1",
                                        "action_hint": "TRADE",
                                    }
                                ],
                            }
                        ]
                    }
                )
            )
            for path in (order_intents, position_management, thesis_evolution, forecast_persistence, market_context):
                path.write_text("{}")

            summary = run_guardian_event_router(
                snapshot_path=snapshot,
                pair_charts_path=pair_charts,
                order_intents_path=order_intents,
                position_management_path=position_management,
                thesis_evolution_path=thesis_evolution,
                forecast_persistence_path=forecast_persistence,
                market_context_matrix_path=market_context,
                state_path=root / "guardian_event_state.json",
                events_output_path=root / "guardian_events.json",
                escalation_output_path=root / "guardian_escalation.json",
                report_path=root / "guardian_event_report.md",
                action_review_report_path=root / "guardian_action_review.md",
            )

            self.assertTrue(summary.wake_gpt)
            self.assertTrue((root / "guardian_events.json").exists())
            self.assertTrue((root / "guardian_escalation.json").exists())
            self.assertTrue((root / "guardian_event_report.md").exists())
            self.assertTrue((root / "guardian_action_review.md").exists())


def _snapshot(*, positions: list[dict] | None = None, orders: list[dict] | None = None) -> dict:
    return {
        "fetched_at_utc": NOW.isoformat(),
        "account": {
            "nav_jpy": 200000.0,
            "margin_used_jpy": 20000.0,
            "margin_available_jpy": 180000.0,
        },
        "positions": positions or [],
        "orders": orders or [],
        "quotes": {
            "EUR_USD": {"bid": 1.1730, "ask": 1.1731},
            "USD_JPY": {"bid": 157.37, "ask": 157.38},
        },
    }


def _contract(
    *,
    generated_at: datetime = NOW,
    entry_overrides: dict | None = None,
) -> dict:
    entry = {
        "pair": "EUR_USD",
        "side": "LONG",
        "thesis": "range long",
        "owner": "SYSTEM",
        "thesis_state": "ALIVE",
        "harvest_triggers": [],
        "add_triggers": [],
        "no_add_triggers": [],
        "wounded_triggers": [],
        "invalidation_triggers": [],
        "emergency_triggers": [],
        "next_review_reason": "hourly market read refresh",
        "next_review_deadline_utc": (generated_at + timedelta(hours=1)).isoformat(),
    }
    if entry_overrides:
        entry.update(entry_overrides)
    return {
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "contract_owner": "qr-trader",
        "cycle_horizon_minutes": 60,
        "entries": [entry],
    }


def _events_from_chart(event: dict, *, pair: str = "EUR_USD"):
    return detect_guardian_events(
        inputs={
            "snapshot": _snapshot(),
            "pair_charts": {"charts": [{"pair": pair, "guardian_events": [event]}]},
        },
        now=NOW,
    )


def _issue_codes(review: dict) -> set[str]:
    return {str(issue.get("code")) for issue in review.get("issues", [])}


if __name__ == "__main__":
    unittest.main()
