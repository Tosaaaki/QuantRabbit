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

    def test_p2_duplicate_suppressed_only_when_price_zone_materially_unchanged(self) -> None:
        first_events = _events_from_chart(
            {
                "event_type": "FAILED_ACCEPTANCE",
                "thesis": "USD_JPY failed acceptance",
                "price_zone": "USD_JPY 162.000 rejection",
                "severity": "P2",
                "action_hint": "TRADE",
            },
            pair="USD_JPY",
        )
        _, state = evaluate_guardian_escalation(events=first_events, previous_state={}, now=NOW)

        unchanged, _ = evaluate_guardian_escalation(
            events=first_events,
            previous_state=state,
            now=NOW + timedelta(minutes=5),
        )

        self.assertFalse(unchanged["wake_gpt"])
        self.assertEqual(unchanged["suppressed_events"][0]["suppressed_reason"], "THROTTLED")

    def test_large_price_zone_move_produces_state_change_wake_evidence(self) -> None:
        first_events = _events_from_chart(
            {
                "event_type": "FAILED_ACCEPTANCE",
                "thesis": "USD_JPY failed acceptance",
                "price_zone": "USD_JPY 162.000 rejection",
                "severity": "P2",
                "action_hint": "TRADE",
            },
            pair="USD_JPY",
        )
        _, state = evaluate_guardian_escalation(events=first_events, previous_state={}, now=NOW)
        moved_events = _events_from_chart(
            {
                "event_type": "FAILED_ACCEPTANCE",
                "thesis": "USD_JPY failed acceptance",
                "price_zone": "USD_JPY 162.050 rejection",
                "severity": "P2",
                "action_hint": "TRADE",
            },
            pair="USD_JPY",
        )

        escalation, _ = evaluate_guardian_escalation(
            events=moved_events,
            previous_state=state,
            now=NOW + timedelta(minutes=5),
        )

        self.assertTrue(escalation["wake_gpt"])
        self.assertIn("LARGE_PRICE_DISPLACEMENT_STATE_CHANGE", escalation["wake_reason_codes"])
        self.assertIn("FAILED_ACCEPTANCE_PRICE_ZONE_CHANGE", escalation["wake_reason_codes"])

    def test_current_p0_unknown_exposure_wakes_once_and_same_truth_is_suppressed(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": 30000,
                    "owner": "unknown",
                    "trade_id": "472987",
                    "entry_price": 1.14048,
                }
            ]
        )
        events = detect_guardian_events(inputs={"snapshot": snapshot}, now=NOW)

        first, state = evaluate_guardian_escalation(events=events, previous_state={}, now=NOW)
        second, _ = evaluate_guardian_escalation(
            events=events,
            previous_state=state,
            now=NOW + timedelta(minutes=5),
        )

        self.assertTrue(first["wake_gpt"])
        self.assertIn("UNKNOWN_GATEWAY_OUTSIDE_ORDER", first["wake_reason_codes"])
        self.assertFalse(second["wake_gpt"])
        unknown_suppressed = [
            event for event in second["suppressed_events"] if event["event_type"] == "UNKNOWN_ORDER"
        ]
        self.assertEqual(unknown_suppressed[0]["suppressed_reason"], "THROTTLED")

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

        trigger = contract["entries"][0]["harvest_triggers"][0]
        self.assertEqual(trigger["trigger_id"], "keep_me")
        self.assertEqual(trigger["metric"], "mid")
        self.assertEqual(trigger["trade_id"], "1")
        self.assertEqual(trigger["units"], 1000)
        self.assertEqual(trigger["pair"], "EUR_USD")

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

    def test_open_exposure_avg_entry_populates_from_raw_average_price(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "units": 1000,
                    "owner": "trader",
                    "thesis": "system USD_JPY short",
                    "thesis_state": "ALIVE",
                    "trade_id": "trade-average-price",
                    "raw": {"averagePrice": "162.157"},
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
            self.assertEqual(entry[bucket][0]["avg_entry"], 162.157)

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "guardian_trigger_contract_report.md"
            write_guardian_trigger_contract_report(report_path, contract, validate_guardian_trigger_contract(contract, now=NOW))
            report = report_path.read_text()
        self.assertIn("avg_entry=`162.157`", report)

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

    def test_aud_usd_472965_unknown_owner_is_hold_only_and_not_system_counted(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "AUD_USD",
                    "side": "LONG",
                    "units": 14000,
                    "owner": "unknown",
                    "thesis": "unknown/manual AUD_USD exposure awaiting operator confirmation",
                    "thesis_state": "UNKNOWN",
                    "trade_id": "472965",
                    "entry_price": 0.68966,
                    "unrealized_pl_jpy": -120.0,
                    "raw": {
                        "id": "472965",
                        "instrument": "AUD_USD",
                        "currentUnits": "14000",
                        "price": "0.68966",
                    },
                }
            ]
        )

        contract = build_guardian_trigger_contract(snapshot=snapshot, order_intents={}, existing_contract={}, now=NOW)
        entry = contract["entries"][0]
        audit = entry["ownership_audit"]

        self.assertEqual(validate_guardian_trigger_contract(contract, now=NOW)["status"], "VALID")
        self.assertEqual(entry["trade_id"], "472965")
        self.assertEqual(entry["units"], 14000)
        self.assertEqual(entry["avg_entry"], 0.68966)
        self.assertEqual(entry["owner"], "UNKNOWN")
        self.assertEqual(entry["thesis_state"], "UNKNOWN")
        self.assertEqual(audit["status"], "UNKNOWN_NEEDS_OPERATOR_CONFIRM")
        self.assertTrue(audit["unresolved"])
        self.assertFalse(audit["system_pl_counted"])
        self.assertFalse(audit["loss_side_auto_close_allowed"])
        self.assertFalse(audit["same_theme_auto_add_allowed"])
        self.assertEqual(audit["requires"], "operator confirmation or gateway evidence")
        self.assertIn("UNKNOWN_NEEDS_OPERATOR_CONFIRM", entry["no_add_triggers"][0]["evidence_required"])
        self.assertEqual(entry["invalidation_triggers"][0]["action_hint"], "HOLD")
        self.assertEqual(entry["emergency_triggers"][0]["action_hint"], "HOLD")
        self.assertIn("do not auto-close loss-side", entry["emergency_triggers"][0]["evidence_required"])

    def test_current_eur_usd_472987_unknown_owner_requires_operator_confirmation(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": 30000,
                    "owner": "unknown",
                    "thesis": "gateway-outside broker position",
                    "thesis_state": "UNKNOWN",
                    "trade_id": "472987",
                    "entry_price": 1.14048,
                    "take_profit": None,
                    "stop_loss": None,
                    "unrealized_pl_jpy": -922.0941,
                    "raw": {
                        "id": "472987",
                        "instrument": "EUR_USD",
                        "currentUnits": "-30000",
                        "initialUnits": "-30000",
                        "price": "1.14048",
                    },
                }
            ]
        )

        contract = build_guardian_trigger_contract(snapshot=snapshot, order_intents={}, existing_contract={}, now=NOW)
        entry = contract["entries"][0]
        audit = entry["ownership_audit"]

        self.assertEqual(validate_guardian_trigger_contract(contract, now=NOW, snapshot=snapshot)["status"], "VALID")
        self.assertEqual(entry["trade_id"], "472987")
        self.assertEqual(entry["pair"], "EUR_USD")
        self.assertEqual(entry["side"], "SHORT")
        self.assertEqual(entry["units"], 30000)
        self.assertEqual(entry["avg_entry"], 1.14048)
        self.assertEqual(entry["owner"], "UNKNOWN")
        self.assertEqual(entry["thesis_state"], "UNKNOWN")
        self.assertEqual(audit["status"], "UNKNOWN_NEEDS_OPERATOR_CONFIRM")
        self.assertTrue(audit["unresolved"])
        self.assertFalse(audit["system_pl_counted"])
        self.assertFalse(audit["loss_side_auto_close_allowed"])
        self.assertFalse(audit["same_theme_auto_add_allowed"])
        self.assertEqual(audit["requires"], "operator confirmation or gateway evidence")
        self.assertEqual(entry["invalidation_triggers"][0]["action_hint"], "HOLD")
        self.assertEqual(entry["emergency_triggers"][0]["action_hint"], "HOLD")

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

    def test_preserved_trigger_metadata_is_rebound_to_current_parent(self) -> None:
        existing = _contract(
            entry_overrides={
                "trade_id": "472909",
                "units": 15000,
                "avg_entry": 162.1,
                "pair": "USD_JPY",
                "side": "SHORT",
                "owner": "OPERATOR_MANUAL",
                "thesis": "operator manual USD_JPY 162 fade",
                "harvest_triggers": [
                    {
                        "trigger_id": "old_trade_profit_zone",
                        "trade_id": "472909",
                        "units": 15000,
                        "avg_entry": 162.1,
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "owner": "OPERATOR_MANUAL",
                        "thesis": "operator manual USD_JPY 162 fade",
                    }
                ],
            }
        )
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "units": 1000,
                    "owner": "operator_manual",
                    "thesis": "operator manual USD_JPY 162 fade",
                    "trade_id": "472931",
                    "price": 162.157,
                }
            ]
        )

        contract = build_guardian_trigger_contract(snapshot=snapshot, order_intents={}, existing_contract=existing, now=NOW)
        entry = contract["entries"][0]
        trigger = entry["harvest_triggers"][0]

        self.assertEqual(entry["trade_id"], "472931")
        self.assertEqual(entry["units"], 1000)
        self.assertEqual(trigger["trade_id"], "472931")
        self.assertEqual(trigger["units"], 1000)
        self.assertEqual(trigger["avg_entry"], 162.157)
        self.assertNotIn("472909", json.dumps(entry))
        self.assertNotIn("15000", json.dumps(entry))
        self.assertEqual(validate_guardian_trigger_contract(contract, now=NOW)["status"], "VALID")

    def test_trigger_trade_id_mismatch_is_block_for_open_exposure(self) -> None:
        contract = _contract(
            entry_overrides={
                **_open_required_trigger_fields(),
                "trade_id": "472931",
                "units": 1000,
                "avg_entry": 162.157,
                "pair": "USD_JPY",
                "side": "SHORT",
                "owner": "OPERATOR_MANUAL",
                "harvest_triggers": [
                    {
                        "trigger_id": "bad_parent",
                        "trade_id": "472909",
                        "units": 15000,
                        "avg_entry": 162.1,
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "owner": "OPERATOR_MANUAL",
                    }
                ],
            }
        )

        validation = validate_guardian_trigger_contract(contract, now=NOW)

        self.assertEqual(validation["status"], "INVALID")
        codes = {issue["code"] for issue in validation["issues"] if issue["severity"] == "BLOCK"}
        self.assertIn("CONTRACT_TRIGGER_TRADE_ID_MISMATCH", codes)
        self.assertIn("CONTRACT_TRIGGER_UNITS_MISMATCH", codes)
        self.assertIn("CONTRACT_TRIGGER_AVG_ENTRY_MISMATCH", codes)

    def test_expired_contract_deadline_emits_contract_stale_with_exposure(self) -> None:
        contract = _contract(
            entry_overrides={
                **_open_required_trigger_fields(),
                "trade_id": "1",
                "units": 1000,
                "avg_entry": 1.171,
                "next_review_deadline_utc": (NOW - timedelta(minutes=1)).isoformat(),
            }
        )
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

    def test_expired_watch_only_candidate_does_not_emit_contract_stale_by_itself(self) -> None:
        contract = _contract(entry_overrides={"next_review_deadline_utc": (NOW - timedelta(minutes=1)).isoformat()})

        validation = validate_guardian_trigger_contract(contract, now=NOW)
        events = detect_guardian_events(inputs={"snapshot": _snapshot(), "trigger_contract": contract}, now=NOW)

        self.assertEqual(validation["status"], "VALID")
        self.assertFalse(validation["stale"])
        self.assertIn("CONTRACT_ENTRY_WATCH_DEADLINE_EXPIRED", {issue["code"] for issue in validation["issues"]})
        self.assertNotIn("CONTRACT_STALE", {event.event_type for event in events})

    def test_open_exposure_with_valid_deadline_and_triggers_is_valid(self) -> None:
        contract = _contract(
            entry_overrides={
                **_open_required_trigger_fields(),
                "trade_id": "valid-open",
                "units": 1000,
                "avg_entry": 1.171,
            }
        )
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "units": 1000,
                    "owner": "trader",
                    "trade_id": "valid-open",
                }
            ]
        )

        validation = validate_guardian_trigger_contract(contract, now=NOW)
        events = detect_guardian_events(inputs={"snapshot": snapshot, "trigger_contract": contract}, now=NOW)

        self.assertEqual(validation["status"], "VALID")
        self.assertFalse(validation["stale"])
        self.assertNotIn("CONTRACT_STALE", {event.event_type for event in events})

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

    def test_missing_current_open_exposure_in_contract_is_blocked(self) -> None:
        contract = _contract()
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": 30000,
                    "owner": "unknown",
                    "trade_id": "472987",
                    "entry_price": 1.14048,
                }
            ]
        )

        validation = validate_guardian_trigger_contract(contract, now=NOW, snapshot=snapshot)
        events = detect_guardian_events(inputs={"snapshot": snapshot, "trigger_contract": contract}, now=NOW)

        self.assertEqual(validation["status"], "INVALID")
        self.assertIn("CONTRACT_OPEN_EXPOSURE_MISSING", {issue["code"] for issue in validation["issues"]})
        self.assertIn("CONTRACT_STALE", {event.event_type for event in events})

    def test_selected_candidate_empty_triggers_blocks_without_watch_only_reason(self) -> None:
        contract = _contract(entry_overrides={"selected": True, "status": "LIVE_READY"})

        validation = validate_guardian_trigger_contract(contract, now=NOW)

        self.assertEqual(validation["status"], "INVALID")
        blockers = {issue["code"] for issue in validation["issues"] if issue["severity"] == "BLOCK"}
        self.assertIn("TRIGGER_CONTRACT_EMPTY_FOR_ACTIVE_PAIR", blockers)

    def test_watch_only_candidate_may_omit_triggers_with_explicit_reason(self) -> None:
        contract = _contract(
            entry_overrides={
                "watch_only": True,
                "watch_only_reason": "no_current_thesis",
                "thesis_state": "UNKNOWN",
            }
        )

        validation = validate_guardian_trigger_contract(contract, now=NOW)

        self.assertEqual(validation["status"], "VALID")
        self.assertNotIn("WATCH_ONLY_NO_TRIGGER_CONTRACT", {issue["code"] for issue in validation["issues"]})
        self.assertNotIn("TRIGGER_CONTRACT_EMPTY_FOR_ACTIVE_PAIR", {issue["code"] for issue in validation["issues"]})

    def test_watch_only_candidate_without_reason_warns_when_triggers_omitted(self) -> None:
        contract = _contract(entry_overrides={"watch_only": True, "thesis_state": "UNKNOWN"})

        validation = validate_guardian_trigger_contract(contract, now=NOW)

        self.assertEqual(validation["status"], "VALID")
        self.assertIn("WATCH_ONLY_NO_TRIGGER_CONTRACT", {issue["code"] for issue in validation["issues"]})

    def test_candidate_builder_adds_default_triggers_for_tradable_thesis(self) -> None:
        contract = build_guardian_trigger_contract(
            snapshot=_snapshot(),
            order_intents={
                "selected_lane_id": "market-read:USD_JPY:SHORT",
                "results": [
                    {
                        "lane_id": "market-read:USD_JPY:SHORT",
                        "status": "LIVE_READY",
                        "intent": {
                            "pair": "USD_JPY",
                            "side": "SHORT",
                            "thesis": "USD_JPY failed acceptance short",
                            "entry": 162.0,
                            "take_profit": 161.72,
                            "stop_loss": 162.18,
                            "metadata": {"max_spread_pips": 1.2},
                        },
                    }
                ],
            },
            existing_contract={},
            now=NOW,
        )
        entry = contract["entries"][0]

        self.assertTrue(entry["selected"])
        for bucket in (
            "harvest_triggers",
            "no_add_triggers",
            "wounded_triggers",
            "invalidation_triggers",
            "emergency_triggers",
        ):
            self.assertTrue(entry[bucket], bucket)
        self.assertEqual(validate_guardian_trigger_contract(contract, now=NOW)["status"], "VALID")

    def test_candidate_builder_marks_missing_thesis_as_watch_only(self) -> None:
        contract = build_guardian_trigger_contract(
            snapshot=_snapshot(),
            order_intents={
                "results": [
                    {
                        "lane_id": "market-read:USD_JPY:SHORT",
                        "status": "MARKET_READ_FIRST",
                        "intent": {"pair": "USD_JPY", "side": "SHORT"},
                    }
                ],
            },
            existing_contract={},
            now=NOW,
        )
        entry = contract["entries"][0]

        self.assertTrue(entry["watch_only"])
        self.assertEqual(entry["watch_only_reason"], "no_current_thesis")
        self.assertEqual(validate_guardian_trigger_contract(contract, now=NOW)["status"], "VALID")

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

    def test_router_preserves_accepted_receipt_lifecycle_when_reviewing_output_path(self) -> None:
        event_input = {
            "event_type": "FAILED_ACCEPTANCE",
            "thesis": "major figure rejection",
            "price_zone": "EUR_USD 1.1700 rejection",
            "severity": "P1",
            "action_hint": "TRADE",
        }
        event = _events_from_chart(event_input)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            pair_charts = root / "pair_charts.json"
            order_intents = root / "order_intents.json"
            position_management = root / "position_management.json"
            thesis_evolution = root / "thesis_evolution.json"
            forecast_persistence = root / "forecast_persistence.json"
            market_context = root / "market_context.json"
            action_receipt = root / "guardian_action_receipt.json"
            snapshot.write_text(json.dumps(_snapshot()))
            pair_charts.write_text(
                json.dumps({"charts": [{"pair": "EUR_USD", "guardian_events": [event_input]}]})
            )
            for path in (order_intents, position_management, thesis_evolution, forecast_persistence, market_context):
                path.write_text("{}")
            action_receipt.write_text(
                json.dumps(
                    {
                        "receipt_status": "ACCEPTED",
                        "receipt_lifecycle": "ACTIVE",
                        "generated_at_utc": NOW.isoformat(),
                        "selected_event_id": event.event_id,
                        "selected_event_dedupe_key": event.dedupe_key,
                        "expires_at_utc": (NOW + timedelta(minutes=75)).isoformat(),
                        "consumed_by_trader": False,
                        "superseded_by_event_id": None,
                        "action": "HOLD",
                        "new_information": True,
                        "event_id": event.event_id,
                        "thesis_state": "ALIVE",
                        "reason": "hold until price accepts away from the failed acceptance zone",
                        "invalidation": "accepted break through the failed acceptance zone",
                        "harvest_trigger": "range rail reached",
                        "gateway_required": True,
                    }
                )
            )

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
                action_receipt_input_path=action_receipt,
                action_receipt_output_path=action_receipt,
                action_review_report_path=root / "guardian_action_review.md",
                now=NOW,
            )

            payload = json.loads(action_receipt.read_text())
            self.assertEqual(summary.action_review_status, "ACCEPTED")
            self.assertEqual(payload["receipt_status"], "ACCEPTED")
            self.assertEqual(payload["receipt_lifecycle"], "ACTIVE")
            self.assertEqual(payload["selected_event_id"], event.event_id)
            self.assertEqual(payload["router_review"]["status"], "ACCEPTED")
            self.assertEqual(payload["router_review"]["receipt"]["event_id"], event.event_id)


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


def _open_required_trigger_fields() -> dict:
    return {
        bucket: [{"trigger_id": f"{bucket}:fixture", "metric": "mid", "operator": ">=", "value": 1.17}]
        for bucket in (
            "harvest_triggers",
            "no_add_triggers",
            "wounded_triggers",
            "invalidation_triggers",
            "emergency_triggers",
        )
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
