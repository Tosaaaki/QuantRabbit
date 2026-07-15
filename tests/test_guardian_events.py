from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import quant_rabbit.guardian_events as guardian_events_module
from quant_rabbit.analysis.candles import _technical_candles_from_payload
from quant_rabbit.cli import main
from quant_rabbit.guardian_events import (
    CONTRACT_TRIGGER_BUCKETS,
    build_guardian_trigger_contract,
    detect_guardian_events,
    evaluate_guardian_escalation,
    review_guardian_action_receipt,
    run_guardian_event_router,
    validate_guardian_trigger_contract,
    write_guardian_trigger_contract_report,
)
from quant_rabbit.operator_manual import OPERATOR_MANUAL_POSITION_PACKET
from quant_rabbit.instruments import (
    NORMAL_SPREAD_PIPS,
    OANDA_SPREAD_CALIBRATION_V1,
    instrument_pip_factor,
)
from quant_rabbit.risk import RiskPolicy


NOW = datetime(2026, 6, 30, 3, 30, tzinfo=timezone.utc)
SUMMER_WEEKEND = datetime(2026, 7, 11, 3, 30, tzinfo=timezone.utc)


class GuardianEventRouterTest(unittest.TestCase):
    def test_atomic_guardian_write_failure_preserves_last_good_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guardian_escalation.json"
            path.write_text('{"wake_gpt":false}\n', encoding="utf-8")

            with (
                patch.object(guardian_events_module.os, "replace", side_effect=OSError("ENOSPC")),
                self.assertRaisesRegex(OSError, "ENOSPC"),
            ):
                guardian_events_module._write_json(path, {"wake_gpt": True})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"wake_gpt": False})
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_manual_open_pair_gets_fresh_bounded_technical_state_event(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": -30000,
                    "owner": "operator_manual",
                }
            ]
        )
        events = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": _technical_chart_payload(mid=1.17305, generated_at=NOW),
            },
            now=NOW,
        )

        technical = next(event for event in events if event.event_type == "TECHNICAL_STATE_CHANGE")
        self.assertEqual(technical.pair, "EUR_USD")
        self.assertEqual(technical.action_hint, "HOLD")
        self.assertTrue(technical.details["open_exposure"])
        self.assertFalse(technical.details["live_permission_allowed"])
        self.assertEqual(
            technical.details["material_fingerprint"]["family_consensus"]["trend"],
            "DOWN",
        )

    def test_market_closed_technical_event_stays_visible_without_waking(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": -30000,
                    "owner": "operator_manual",
                }
            ]
        )
        snapshot["fetched_at_utc"] = SUMMER_WEEKEND.isoformat()
        events = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17305,
                    generated_at=SUMMER_WEEKEND,
                ),
            },
            now=SUMMER_WEEKEND,
        )
        technical = next(
            event for event in events if event.event_type == "TECHNICAL_STATE_CHANGE"
        )

        escalation, state = evaluate_guardian_escalation(
            events=[technical],
            previous_state={},
            now=SUMMER_WEEKEND,
        )

        self.assertFalse(escalation["wake_gpt"])
        self.assertEqual(escalation["events_to_review"], [])
        self.assertFalse(escalation["market_status"]["is_fx_open"])
        self.assertEqual(
            escalation["suppressed_events"][0]["suppressed_reason"],
            "MARKET_CLOSED_TUNING_OBSERVATION",
        )
        self.assertIn(
            "NEW_EVENT",
            escalation["suppressed_events"][0]["suppressed_wake_reason_codes"],
        )
        self.assertIn(technical.dedupe_key, state["events"])
        self.assertFalse(state["market_status"]["is_fx_open"])

    def test_market_closed_failed_acceptance_entry_watches_do_not_wake(self) -> None:
        trade_watch = _events_from_chart(
            {
                "event_type": "FAILED_ACCEPTANCE",
                "thesis": "breakout failure entry watch",
                "price_zone": "GBP_JPY 198.00 rejection",
                "severity": "P1",
                "action_hint": "TRADE",
            },
            pair="GBP_JPY",
        )[0]
        hold_watch = _events_from_chart(
            {
                "event_type": "FAILED_ACCEPTANCE",
                "thesis": "major figure observation",
                "price_zone": "AUD_JPY 105.00 rejection",
                "severity": "P2",
                "action_hint": "HOLD",
            },
            pair="AUD_JPY",
        )[0]

        escalation, state = evaluate_guardian_escalation(
            events=[trade_watch, hold_watch],
            previous_state={},
            now=SUMMER_WEEKEND,
        )

        self.assertFalse(escalation["wake_gpt"])
        self.assertEqual(escalation["events_to_review"], [])
        self.assertEqual(
            {
                event["suppressed_reason"]
                for event in escalation["suppressed_events"]
            },
            {"MARKET_CLOSED_FAILED_ACCEPTANCE_WATCH"},
        )
        self.assertEqual(
            {
                event["action_hint"]
                for event in escalation["suppressed_events"]
            },
            {"TRADE", "HOLD"},
        )
        self.assertIn(trade_watch.dedupe_key, state["events"])
        self.assertIn(hold_watch.dedupe_key, state["events"])

    def test_market_closed_failed_acceptance_watch_wakes_once_after_reopen(self) -> None:
        trade_watch = _events_from_chart(
            {
                "event_type": "FAILED_ACCEPTANCE",
                "thesis": "breakout failure entry watch",
                "price_zone": "GBP_JPY 198.00 rejection",
                "severity": "P1",
                "action_hint": "TRADE",
            },
            pair="GBP_JPY",
        )[0]
        closed_escalation, closed_state = evaluate_guardian_escalation(
            events=[trade_watch],
            previous_state={},
            now=SUMMER_WEEKEND,
        )

        reopened_at = datetime(2026, 7, 13, 0, 0, tzinfo=timezone.utc)
        reopened_escalation, reopened_state = evaluate_guardian_escalation(
            events=[trade_watch],
            previous_state=closed_state,
            now=reopened_at,
        )
        repeated_escalation, _ = evaluate_guardian_escalation(
            events=[trade_watch],
            previous_state=reopened_state,
            now=reopened_at + timedelta(minutes=1),
        )

        self.assertFalse(closed_escalation["wake_gpt"])
        self.assertTrue(reopened_escalation["market_status"]["is_fx_open"])
        self.assertTrue(reopened_escalation["wake_gpt"])
        self.assertIn(
            "MARKET_REOPEN_FAILED_ACCEPTANCE_WATCH",
            reopened_escalation["wake_reason_codes"],
        )
        self.assertFalse(repeated_escalation["wake_gpt"])

    def test_market_closed_fresh_entry_observations_wake_once_after_reopen(self) -> None:
        cases = (
            ("THEME_CONFIRMATION", "TRADE"),
            ("RANGE_RAIL_TOUCH", "TRADE"),
            ("SESSION_EXPANSION", "TRADE"),
            ("SQUEEZE_RELEASE", "ADD"),
        )
        reopened_at = datetime(2026, 7, 13, 0, 0, tzinfo=timezone.utc)

        for event_type, action_hint in cases:
            with self.subTest(event_type=event_type, action_hint=action_hint):
                event = next(
                    event
                    for event in _events_from_chart(
                        {
                            "event_type": event_type,
                            "thesis": f"{event_type.lower()} fresh entry watch",
                            "price_zone": f"{event_type.lower()} trigger zone",
                            "severity": "P1",
                            "recommended_review_type": "ENTRY_REVIEW",
                            "action_hint": action_hint,
                        }
                    )
                    if event.event_type == event_type
                )
                closed_escalation, closed_state = evaluate_guardian_escalation(
                    events=[event],
                    previous_state={},
                    now=SUMMER_WEEKEND,
                )
                reopened_escalation, reopened_state = evaluate_guardian_escalation(
                    events=[event],
                    previous_state=closed_state,
                    now=reopened_at,
                )
                repeated_escalation, _ = evaluate_guardian_escalation(
                    events=[event],
                    previous_state=reopened_state,
                    now=reopened_at + timedelta(minutes=1),
                )

                self.assertFalse(closed_escalation["wake_gpt"])
                self.assertEqual(closed_escalation["events_to_review"], [])
                self.assertEqual(
                    closed_escalation["suppressed_events"][0]["suppressed_reason"],
                    "MARKET_CLOSED_ENTRY_OBSERVATION",
                )
                self.assertTrue(reopened_escalation["wake_gpt"])
                self.assertIn(
                    "MARKET_REOPEN_ENTRY_OBSERVATION",
                    reopened_escalation["wake_reason_codes"],
                )
                self.assertFalse(repeated_escalation["wake_gpt"])

    def test_market_closed_entry_review_with_open_trade_identity_is_not_suppressed(self) -> None:
        event = next(
            event
            for event in _events_from_chart(
                {
                    "event_type": "SQUEEZE_RELEASE",
                    "thesis": "open-position add review",
                    "price_zone": "M5 squeeze release",
                    "severity": "P1",
                    "recommended_review_type": "ENTRY_REVIEW",
                    "action_hint": "ADD",
                    "trade_id": "open-trade-1",
                }
            )
            if event.event_type == "SQUEEZE_RELEASE"
        )

        escalation, _ = evaluate_guardian_escalation(
            events=[event],
            previous_state={},
            now=SUMMER_WEEKEND,
        )

        self.assertTrue(escalation["wake_gpt"])
        self.assertEqual(
            escalation["events_to_review"][0]["details"]["trade_id"],
            "open-trade-1",
        )

    def test_market_closed_candidate_harvest_is_suppressed_but_open_position_harvest_is_not(self) -> None:
        candidate_events = detect_guardian_events(
            inputs={
                "order_intents": {
                    "results": [
                        {
                            "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                            "status": "LIVE_READY",
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "thesis": "failed acceptance harvest",
                                "entry": 1.17,
                                "market_context": {"method": "BREAKOUT_FAILURE"},
                                "metadata": {
                                    "opportunity_mode": "HARVEST",
                                    "failed_acceptance": True,
                                    "oanda_m5_failed_break_long": False,
                                    "oanda_m5_failed_break_short": True,
                                    "acceptance_zone": "1.1700",
                                },
                            },
                        }
                    ]
                }
            },
            now=SUMMER_WEEKEND,
        )
        candidate_escalation, _ = evaluate_guardian_escalation(
            events=candidate_events,
            previous_state={},
            now=SUMMER_WEEKEND,
        )

        self.assertEqual(
            {event.event_type for event in candidate_events},
            {"FAILED_ACCEPTANCE", "HARVEST_ZONE"},
        )
        self.assertFalse(candidate_escalation["wake_gpt"])
        self.assertEqual(candidate_escalation["events_to_review"], [])
        candidate_harvest = next(
            event
            for event in candidate_escalation["suppressed_events"]
            if event["event_type"] == "HARVEST_ZONE"
        )
        self.assertEqual(
            candidate_harvest["suppressed_reason"],
            "MARKET_CLOSED_CANDIDATE_HARVEST_WATCH",
        )

        open_position_events = detect_guardian_events(
            inputs={
                "position_management": {
                    "positions": [
                        {
                            "trade_id": "t1",
                            "pair": "EUR_USD",
                            "side": "SHORT",
                            "action": "TAKE_PROFIT_MARKET",
                            "thesis": "open winner",
                        }
                    ]
                }
            },
            now=SUMMER_WEEKEND,
        )
        open_escalation, _ = evaluate_guardian_escalation(
            events=open_position_events,
            previous_state={},
            now=SUMMER_WEEKEND,
        )

        self.assertTrue(open_escalation["wake_gpt"])
        self.assertEqual(
            open_escalation["events_to_review"][0]["event_type"],
            "HARVEST_ZONE",
        )

    def test_breakout_failure_method_alone_does_not_emit_failed_acceptance(self) -> None:
        events = detect_guardian_events(
            inputs={
                "order_intents": {
                    "results": [
                        {
                            "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                            "status": "LIVE_READY",
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "entry": 1.17,
                                "market_context": {"method": "BREAKOUT_FAILURE"},
                                "metadata": {},
                            },
                        }
                    ]
                }
            },
            now=NOW,
        )

        self.assertFalse(any(event.event_type == "FAILED_ACCEPTANCE" for event in events))

    def test_matching_side_failed_break_emits_failed_acceptance(self) -> None:
        events = detect_guardian_events(
            inputs={
                "order_intents": {
                    "results": [
                        {
                            "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                            "status": "LIVE_READY",
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "entry": 1.17,
                                "market_context": {"method": "BREAKOUT_FAILURE"},
                                "metadata": {
                                    "oanda_m5_failed_break_long": False,
                                    "oanda_m5_failed_break_short": True,
                                    "acceptance_zone": 1.17,
                                },
                            },
                        }
                    ]
                }
            },
            now=NOW,
        )

        failed = [event for event in events if event.event_type == "FAILED_ACCEPTANCE"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].direction, "SHORT")
        self.assertEqual(failed[0].price_zone, "1.17")

    def test_opposite_side_failed_break_does_not_emit_failed_acceptance(self) -> None:
        events = detect_guardian_events(
            inputs={
                "order_intents": {
                    "results": [
                        {
                            "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                            "status": "LIVE_READY",
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "entry": 1.17,
                                "market_context": {"method": "BREAKOUT_FAILURE"},
                                "metadata": {
                                    "oanda_m5_failed_break_long": True,
                                    "oanda_m5_failed_break_short": False,
                                    "acceptance_zone": 1.17,
                                },
                            },
                        }
                    ]
                }
            },
            now=NOW,
        )

        self.assertFalse(any(event.event_type == "FAILED_ACCEPTANCE" for event in events))

    def test_market_closed_suppression_does_not_stop_safety_or_manual_monitoring(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": -30000,
                    "owner": "unknown",
                    "trade_id": "manual-outside-gateway",
                }
            ]
        )
        snapshot["fetched_at_utc"] = SUMMER_WEEKEND.isoformat()
        events = detect_guardian_events(
            inputs={"snapshot": snapshot, "pair_charts": {}},
            now=SUMMER_WEEKEND,
        )
        failed_acceptance_watch = _events_from_chart(
            {
                "event_type": "FAILED_ACCEPTANCE",
                "thesis": "fresh entry watch must wait for an open market",
                "price_zone": "GBP_JPY 198.00 rejection",
                "severity": "P1",
                "action_hint": "TRADE",
            },
            pair="GBP_JPY",
        )[0]

        escalation, _ = evaluate_guardian_escalation(
            events=[*events, failed_acceptance_watch],
            previous_state={},
            now=SUMMER_WEEKEND,
        )

        event_types = {event.event_type for event in events}
        review_types = {
            event["event_type"] for event in escalation["events_to_review"]
        }
        self.assertIn("UNKNOWN_ORDER", event_types)
        self.assertIn("TECHNICAL_INPUT_STALE", event_types)
        self.assertIn("UNKNOWN_ORDER", review_types)
        self.assertNotIn("TECHNICAL_INPUT_STALE", review_types)
        self.assertNotIn("FAILED_ACCEPTANCE", review_types)
        failed_acceptance_suppressed = next(
            event
            for event in escalation["suppressed_events"]
            if event["event_type"] == "FAILED_ACCEPTANCE"
        )
        self.assertEqual(
            failed_acceptance_suppressed["suppressed_reason"],
            "MARKET_CLOSED_FAILED_ACCEPTANCE_WATCH",
        )
        self.assertTrue(escalation["wake_gpt"])

    def test_technical_input_still_stale_becomes_reviewable_after_reopen(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": -30000,
                    "owner": "operator_manual",
                }
            ]
        )
        snapshot["fetched_at_utc"] = SUMMER_WEEKEND.isoformat()
        stale_event = next(
            event
            for event in detect_guardian_events(
                inputs={"snapshot": snapshot, "pair_charts": {}},
                now=SUMMER_WEEKEND,
            )
            if event.event_type == "TECHNICAL_INPUT_STALE"
        )
        closed_escalation, closed_state = evaluate_guardian_escalation(
            events=[stale_event],
            previous_state={},
            now=SUMMER_WEEKEND,
        )

        reopened_at = datetime(2026, 7, 13, 0, 0, tzinfo=timezone.utc)
        reopened_escalation, _ = evaluate_guardian_escalation(
            events=[stale_event],
            previous_state=closed_state,
            now=reopened_at,
        )

        self.assertFalse(closed_escalation["wake_gpt"])
        self.assertTrue(reopened_escalation["market_status"]["is_fx_open"])
        self.assertTrue(reopened_escalation["wake_gpt"])
        self.assertIn(
            "MARKET_REOPEN_TECHNICAL_INPUT_STILL_STALE",
            reopened_escalation["wake_reason_codes"],
        )

    def test_reopen_stale_review_does_not_require_a_closed_guardian_pass(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": -30000,
                    "owner": "operator_manual",
                }
            ]
        )
        friday_before_close = datetime(2026, 7, 10, 20, 59, tzinfo=timezone.utc)
        snapshot["fetched_at_utc"] = friday_before_close.isoformat()
        stale_event = next(
            event
            for event in detect_guardian_events(
                inputs={"snapshot": snapshot, "pair_charts": {}},
                now=friday_before_close,
            )
            if event.event_type == "TECHNICAL_INPUT_STALE"
        )
        open_escalation, preweekend_state = evaluate_guardian_escalation(
            events=[stale_event],
            previous_state={},
            now=friday_before_close,
        )
        self.assertTrue(open_escalation["market_status"]["is_fx_open"])

        monday_after_open = datetime(2026, 7, 12, 21, 1, tzinfo=timezone.utc)
        reopened_escalation, _ = evaluate_guardian_escalation(
            events=[stale_event],
            previous_state=preweekend_state,
            now=monday_after_open,
        )

        self.assertTrue(reopened_escalation["wake_gpt"])
        self.assertIn(
            "MARKET_REOPEN_TECHNICAL_INPUT_STILL_STALE",
            reopened_escalation["wake_reason_codes"],
        )

    def test_negative_breakout_pressure_uses_independent_direction_not_score_sign(self) -> None:
        charts = _technical_chart_payload(
            mid=1.17305,
            generated_at=NOW,
            family_sign=1,
            structure_kind="BOS_UP",
        )
        for view in charts["charts"][0]["views"]:
            view["family_scores"]["breakout_score"] = -0.8
        events = detect_guardian_events(
            inputs={"snapshot": _snapshot(), "pair_charts": charts},
            now=NOW,
        )

        technical = next(event for event in events if event.event_type == "TECHNICAL_STATE_CHANGE")
        self.assertEqual(
            technical.details["material_fingerprint"]["family_consensus"]["breakout"],
            "UP",
        )

    def test_fresh_chart_missing_m1_and_m5_emits_partial_input_stale_event(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": -30000,
                    "owner": "operator_manual",
                }
            ]
        )
        charts = _technical_chart_payload(mid=1.17305, generated_at=NOW)
        charts["charts"][0]["views"] = [
            view for view in charts["charts"][0]["views"] if view["granularity"] == "M15"
        ]

        events = detect_guardian_events(
            inputs={"snapshot": snapshot, "pair_charts": charts},
            now=NOW,
        )

        technical = [event for event in events if event.event_type.startswith("TECHNICAL_")]
        self.assertEqual([event.event_type for event in technical], ["TECHNICAL_INPUT_STALE"])
        self.assertEqual(technical[0].severity, "P1")
        self.assertEqual(technical[0].details["status"], "REQUIRED_FAST_VIEWS_MISSING")

    def test_first_observation_without_complete_fast_candle_is_stale_not_new_state(self) -> None:
        charts = _technical_chart_payload(mid=1.17305, generated_at=NOW)
        for view in charts["charts"][0]["views"]:
            view["recent_candles"] = []

        events = detect_guardian_events(
            inputs={"snapshot": _snapshot(), "pair_charts": charts},
            now=NOW,
        )
        escalation, _ = evaluate_guardian_escalation(
            events=events,
            previous_state={},
            now=NOW,
        )

        technical = [event for event in events if event.event_type.startswith("TECHNICAL_")]
        self.assertEqual([event.event_type for event in technical], ["TECHNICAL_INPUT_STALE"])
        self.assertEqual(technical[0].details["status"], "COMPLETE_FAST_CANDLE_MISSING")
        self.assertFalse(
            any(
                item.get("event_type") == "TECHNICAL_STATE_CHANGE"
                for item in escalation["events_to_review"]
            )
        )

    def test_current_integrity_block_emits_stale_input_not_technical_state(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": -30000,
                    "owner": "operator_manual",
                }
            ]
        )
        charts = _technical_chart_payload(mid=1.17305, generated_at=NOW)
        m5 = next(
            view
            for view in charts["charts"][0]["views"]
            if view["granularity"] == "M5"
        )
        m5["recent_candles"] = []
        m5["candle_integrity"] = {
            "evaluation_status": "BLOCKED",
            "forecast_blocking": True,
            "blocking_codes": ["TECHNICAL_CANDLE_SPREAD_CONTAMINATED"],
            "latest_complete_timestamp_utc": (NOW - timedelta(minutes=5)).isoformat(),
            "recent_clean_tail_count": 0,
            "recent_tail_state": "SPREAD_CONTAMINATED",
        }
        charts["charts"][0]["technical_candle_integrity"] = {
            "evaluation_status": "BLOCKED",
            "forecast_blocking": True,
            "blocking_codes": ["TECHNICAL_CANDLE_SPREAD_CONTAMINATED"],
        }

        events = detect_guardian_events(
            inputs={"snapshot": snapshot, "pair_charts": charts},
            now=NOW,
        )

        technical = [event for event in events if event.event_type.startswith("TECHNICAL_")]
        self.assertEqual([event.event_type for event in technical], ["TECHNICAL_INPUT_STALE"])
        self.assertEqual(
            technical[0].details["status"],
            "TECHNICAL_CANDLE_INTEGRITY_BLOCKED",
        )
        self.assertEqual(technical[0].details["blocked_timeframes"], ["M5"])
        self.assertEqual(
            technical[0].details["blocking_codes"],
            ["TECHNICAL_CANDLE_SPREAD_CONTAMINATED"],
        )
        self.assertEqual(technical[0].action_hint, "HOLD")
        self.assertFalse(technical[0].details["live_permission_allowed"])

    def test_external_freshness_receipt_revalidates_each_timeframe_clock(self) -> None:
        charts = _technical_chart_payload(
            mid=1.17305,
            generated_at=NOW,
            timeframes=("M1", "M5", "M15"),
        )
        charts["charts"][0]["guardian_events"] = [
            {
                "event_type": "SQUEEZE_RELEASE",
                "direction": "LONG",
                "thesis": "must not bypass stale M1 evidence",
                "price_zone": "M1 release",
                "severity": "P1",
                "action_hint": "TRADE",
            }
        ]
        pair_charts_sha256 = "a" * 64
        freshness = {
            "checked_at_utc": NOW.isoformat(),
            "source_generated_at_utc": NOW.isoformat(),
            "source_pair_charts_sha256": pair_charts_sha256,
            "rows": [
                {
                    "pair": "EUR_USD",
                    "timeframe": timeframe,
                    "status": "FRESH",
                    "max_age_seconds": max_age,
                    "latest_complete_candle_closed_at_utc": closed_at.isoformat(),
                }
                for timeframe, max_age, closed_at in (
                    ("M1", 120.0, NOW - timedelta(minutes=20)),
                    ("M5", 600.0, NOW),
                    ("M15", 1800.0, NOW),
                )
            ],
        }

        events = detect_guardian_events(
            inputs={
                "snapshot": _snapshot(),
                "pair_charts": charts,
                "chart_freshness": freshness,
                "chart_freshness_required": True,
                "pair_charts_sha256": pair_charts_sha256,
            },
            now=NOW,
        )

        self.assertEqual(
            [event.event_type for event in events],
            ["TECHNICAL_INPUT_STALE"],
        )
        self.assertEqual(
            events[0].details["status"],
            "GUARDIAN_CHART_FRESHNESS_BLOCKED",
        )
        self.assertIn(
            "M1_CLOSED_AT_STALE",
            events[0].details["freshness_issue"]["reasons"],
        )

    def test_external_freshness_receipt_accepts_exact_current_three_timeframe_binding(self) -> None:
        charts = _technical_chart_payload(
            mid=1.17305,
            generated_at=NOW,
            timeframes=("M1", "M5", "M15"),
        )
        pair_charts_sha256 = "a" * 64

        events = detect_guardian_events(
            inputs={
                "snapshot": _snapshot(),
                "pair_charts": charts,
                "chart_freshness": _external_freshness_receipt(
                    pair_charts_sha256=pair_charts_sha256,
                ),
                "chart_freshness_required": True,
                "pair_charts_sha256": pair_charts_sha256,
            },
            now=NOW,
        )

        self.assertEqual(
            [event.event_type for event in events],
            ["TECHNICAL_STATE_CHANGE"],
        )

    def test_external_freshness_receipt_rejects_pair_chart_sha_mismatch(self) -> None:
        charts = _technical_chart_payload(
            mid=1.17305,
            generated_at=NOW,
            timeframes=("M1", "M5", "M15"),
        )

        events = detect_guardian_events(
            inputs={
                "snapshot": _snapshot(),
                "pair_charts": charts,
                "chart_freshness": _external_freshness_receipt(
                    pair_charts_sha256="a" * 64,
                ),
                "chart_freshness_required": True,
                "pair_charts_sha256": "b" * 64,
            },
            now=NOW,
        )

        self.assertEqual(
            [event.event_type for event in events],
            ["TECHNICAL_INPUT_STALE"],
        )
        self.assertIn(
            "PAIR_CHART_SHA256_BINDING_MISMATCH",
            events[0].details["freshness_issue"]["reasons"],
        )

    def test_router_integrity_mode_requires_canonical_receipts_for_all_fast_views(self) -> None:
        charts = _technical_chart_payload(
            mid=1.17305,
            generated_at=NOW,
            timeframes=("M1", "M5", "M15"),
        )
        charts["charts"][0]["guardian_events"] = [
            {
                "event_type": "FAILED_ACCEPTANCE",
                "direction": "LONG",
                "thesis": "receipt-less entry must stay blocked",
                "price_zone": "M5 rejection",
                "severity": "P1",
                "action_hint": "TRADE",
            }
        ]

        events = detect_guardian_events(
            inputs={
                "snapshot": _snapshot(),
                "pair_charts": charts,
                "technical_integrity_required": True,
            },
            now=NOW,
        )

        self.assertEqual(
            [event.event_type for event in events],
            ["TECHNICAL_INPUT_STALE"],
        )
        self.assertEqual(
            events[0].details["status"],
            "TECHNICAL_CANDLE_INTEGRITY_RECEIPT_INVALID",
        )
        self.assertEqual(
            events[0].details["integrity_issues"],
            ["M1_RECEIPT_INVALID", "M5_RECEIPT_INVALID", "M15_RECEIPT_INVALID"],
        )

    def test_router_integrity_mode_accepts_current_canonical_three_timeframe_receipts(self) -> None:
        events = detect_guardian_events(
            inputs={
                "snapshot": _snapshot(),
                "pair_charts": _canonical_integrity_chart_payload(),
                "technical_integrity_required": True,
            },
            now=NOW,
        )

        self.assertEqual(
            [event.event_type for event in events],
            ["TECHNICAL_STATE_CHANGE"],
        )

    def test_stale_canonical_receipts_preserve_block_and_suppress_entry_without_external_proof(self) -> None:
        charts = _canonical_integrity_chart_payload(stale_by=timedelta(minutes=20))
        charts["charts"][0]["guardian_events"] = [
            {
                "event_type": "FAILED_ACCEPTANCE",
                "direction": "LONG",
                "thesis": "stale receipt must not authorize entry",
                "price_zone": "M5 rejection",
                "severity": "P1",
                "action_hint": "TRADE",
            }
        ]

        events = detect_guardian_events(
            inputs={
                "snapshot": _snapshot(),
                "pair_charts": charts,
                "technical_integrity_required": True,
            },
            now=NOW,
        )
        _escalation, state = evaluate_guardian_escalation(
            events=events,
            previous_state={
                "generated_at_utc": (NOW - timedelta(minutes=1)).isoformat(),
                "events": {
                    "EUR_USD|prior-stale": {
                        "event_type": "TECHNICAL_INPUT_STALE",
                        "pair": "EUR_USD",
                        "last_seen_at_utc": (NOW - timedelta(minutes=1)).isoformat(),
                    }
                },
            },
            now=NOW,
        )

        self.assertEqual(
            [event.event_type for event in events],
            ["TECHNICAL_INPUT_STALE"],
        )
        self.assertIn("M1_LATEST_COMPLETE_STALE", events[0].details["integrity_issues"])
        self.assertIn("M5_LATEST_COMPLETE_STALE", events[0].details["integrity_issues"])
        self.assertTrue(
            any(
                item.get("event_type") == "TECHNICAL_INPUT_STALE"
                for item in state["events"].values()
            )
        )

    def test_integrity_block_suppresses_same_pair_entries_but_keeps_other_pair_and_safety(self) -> None:
        charts = _technical_chart_payload(mid=1.17305, generated_at=NOW)
        chart = charts["charts"][0]
        chart["chart_story"] = "failed acceptance below the major figure"
        chart["guardian_events"] = [
            {
                "event_type": "SQUEEZE_RELEASE",
                "direction": "LONG",
                "thesis": "same-pair squeeze",
                "price_zone": "M5 release",
                "severity": "P1",
                "action_hint": "TRADE",
            }
        ]
        m5 = next(view for view in chart["views"] if view["granularity"] == "M5")
        m5["recent_candles"] = []
        m5["candle_integrity"] = {
            "evaluation_status": "BLOCKED",
            "forecast_blocking": True,
            "blocking_codes": ["TECHNICAL_CANDLE_SPREAD_CONTAMINATED"],
            "latest_complete_timestamp_utc": (NOW - timedelta(minutes=5)).isoformat(),
            "recent_clean_tail_count": 0,
            "recent_tail_state": "SPREAD_CONTAMINATED",
        }
        order_intents = {
            "results": [
                {
                    "lane_id": "eur-entry",
                    "status": "LIVE_READY",
                    "intent": {
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "thesis": "same-pair intent",
                        "entry": 1.17,
                        "metadata": {"squeeze_release": True},
                    },
                },
                {
                    "lane_id": "gbp-entry",
                    "status": "LIVE_READY",
                    "intent": {
                        "pair": "GBP_USD",
                        "side": "LONG",
                        "thesis": "unrelated intent",
                        "entry": 1.31,
                        "metadata": {"squeeze_release": True},
                    },
                },
            ]
        }
        events = detect_guardian_events(
            inputs={
                "snapshot": _snapshot(),
                "pair_charts": charts,
                "order_intents": order_intents,
                "position_management": {
                    "positions": [
                        {
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "trade_id": "safe-reduce",
                            "action": "RECOMMEND_CLOSE",
                            "thesis": "risk exit",
                            "reasons": ["structural BROKEN"],
                        }
                    ]
                },
                "market_context_matrix": {
                    "pairs": {
                        "EUR_USD": {
                            "LONG": {
                                "theme_confirmation": True,
                                "support_count": 3,
                                "reject_count": 0,
                            }
                        }
                    }
                },
            },
            now=NOW,
        )

        self.assertFalse(
            any(
                event.pair == "EUR_USD" and event.action_hint in {"TRADE", "ADD"}
                for event in events
            )
        )
        self.assertTrue(
            any(event.pair == "GBP_USD" and event.action_hint == "TRADE" for event in events)
        )
        self.assertTrue(
            any(event.pair == "EUR_USD" and event.action_hint == "REDUCE" for event in events)
        )

    def test_retained_stale_baseline_suppresses_rotated_out_pair_entry(self) -> None:
        stale = guardian_events_module._event(
            event_type="TECHNICAL_INPUT_STALE",
            pair="EUR_USD",
            direction=None,
            thesis="technical input",
            price_zone="quarantined",
            severity="P2",
            recommended_review_type="TUNING_REVIEW",
            action_hint="NO_ACTION",
            now=NOW,
        )
        _, previous_state = evaluate_guardian_escalation(
            events=[stale],
            previous_state={},
            now=NOW,
        )
        entry = guardian_events_module._event(
            event_type="SQUEEZE_RELEASE",
            pair="EUR_USD",
            direction="LONG",
            thesis="rotated-out stale pair",
            price_zone="M5 release",
            severity="P1",
            recommended_review_type="ENTRY_REVIEW",
            action_hint="TRADE",
            now=NOW + timedelta(minutes=1),
        )

        escalation, next_state = evaluate_guardian_escalation(
            events=[entry],
            previous_state=previous_state,
            now=NOW + timedelta(minutes=1),
        )

        self.assertEqual(escalation["technical_input_blocked_pairs"], ["EUR_USD"])
        self.assertFalse(
            any(item.get("action_hint") == "TRADE" for item in escalation["events_to_review"])
        )
        self.assertTrue(
            any(
                item.get("event_type") == "TECHNICAL_INPUT_STALE"
                for item in next_state["events"].values()
            )
        )

    def test_technical_block_keeps_explicit_no_add_hold_signal(self) -> None:
        stale = guardian_events_module._event(
            event_type="TECHNICAL_INPUT_STALE",
            pair="EUR_USD",
            direction=None,
            thesis="technical input",
            price_zone="quarantined",
            severity="P2",
            recommended_review_type="TUNING_REVIEW",
            action_hint="NO_ACTION",
            now=NOW,
        )
        no_add = guardian_events_module._event(
            event_type="CONTRACT_NO_ADD_TRIGGER",
            pair="EUR_USD",
            direction="LONG",
            thesis="do not add",
            price_zone="no-add boundary",
            severity="P1",
            recommended_review_type="ADD_REVIEW",
            action_hint="HOLD",
            thesis_state="WOUNDED",
            now=NOW,
        )

        filtered = guardian_events_module._suppress_entry_events_for_technical_blocks(
            [stale, no_add],
            blocked_pairs={"EUR_USD"},
        )

        self.assertEqual({event.event_type for event in filtered}, {
            "TECHNICAL_INPUT_STALE",
            "CONTRACT_NO_ADD_TRIGGER",
        })

    def test_trade_receipt_rejects_same_pair_stale_sibling(self) -> None:
        entry = guardian_events_module._event(
            event_type="SQUEEZE_RELEASE",
            pair="EUR_USD",
            direction="LONG",
            thesis="entry",
            price_zone="M5 release",
            severity="P1",
            recommended_review_type="ENTRY_REVIEW",
            action_hint="TRADE",
            thesis_state="ALIVE",
            now=NOW,
        )
        stale = guardian_events_module._event(
            event_type="TECHNICAL_INPUT_STALE",
            pair="EUR_USD",
            direction=None,
            thesis="technical input",
            price_zone="quarantined",
            severity="P2",
            recommended_review_type="TUNING_REVIEW",
            action_hint="NO_ACTION",
            now=NOW,
        )
        review = review_guardian_action_receipt(
            {
                "action": "TRADE",
                "new_information": True,
                "event_id": entry.event_id,
                "pair": "EUR_USD",
                "side": "LONG",
                "thesis_state": "ALIVE",
                "reason": "fresh signal",
                "invalidation": "break failed",
                "harvest_trigger": "target reached",
                "gateway_required": True,
            },
            events=[entry, stale],
            selected_event=entry,
            now=NOW,
        )

        self.assertEqual(review["status"], "REJECTED")
        self.assertIn("GUARDIAN_ACTION_TECHNICAL_INPUT_STALE", _issue_codes(review))

    def test_trade_receipt_cannot_upgrade_hold_or_directionless_event(self) -> None:
        for action_hint, direction, expected_code in (
            ("HOLD", "LONG", "GUARDIAN_ACTION_EVENT_DOES_NOT_AUTHORIZE_ENTRY"),
            ("TRADE", None, "GUARDIAN_ACTION_EVENT_DIRECTION_REQUIRED"),
        ):
            with self.subTest(action_hint=action_hint, direction=direction):
                event = guardian_events_module._event(
                    event_type="FAILED_ACCEPTANCE",
                    pair="EUR_USD",
                    direction=direction,
                    thesis="review only",
                    price_zone="review zone",
                    severity="P1",
                    recommended_review_type="ENTRY_REVIEW",
                    action_hint=action_hint,
                    thesis_state="ALIVE",
                    now=NOW,
                )
                review = review_guardian_action_receipt(
                    {
                        "action": "TRADE",
                        "new_information": True,
                        "event_id": event.event_id,
                        "pair": "EUR_USD",
                        "side": direction or "LONG",
                        "thesis_state": "ALIVE",
                        "reason": "attempted upgrade",
                        "invalidation": "invalidated",
                        "harvest_trigger": "harvest",
                        "gateway_required": True,
                    },
                    events=[event],
                    selected_event=event,
                    now=NOW,
                )

                self.assertEqual(review["status"], "REJECTED")
                self.assertIn(expected_code, _issue_codes(review))

    def test_missing_complete_flag_cannot_be_used_as_closed_candle_watermark(self) -> None:
        charts = _technical_chart_payload(mid=1.17305, generated_at=NOW)
        for view in charts["charts"][0]["views"]:
            for candle in view["recent_candles"]:
                candle.pop("complete", None)

        events = detect_guardian_events(
            inputs={"snapshot": _snapshot(), "pair_charts": charts},
            now=NOW,
        )

        technical = [event for event in events if event.event_type.startswith("TECHNICAL_")]
        self.assertEqual([event.event_type for event in technical], ["TECHNICAL_INPUT_STALE"])
        self.assertEqual(technical[0].details["status"], "COMPLETE_FAST_CANDLE_MISSING")

    def test_tuning_only_technical_event_rejects_live_action_for_manual_position(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": -30000,
                    "owner": "operator_manual",
                }
            ]
        )
        event = next(
            item
            for item in detect_guardian_events(
                inputs={
                    "snapshot": snapshot,
                    "pair_charts": _technical_chart_payload(mid=1.17305, generated_at=NOW),
                },
                now=NOW,
            )
            if item.event_type == "TECHNICAL_STATE_CHANGE"
        )

        review = review_guardian_action_receipt(
            {
                "action": "REDUCE",
                "new_information": True,
                "event_id": event.event_id,
                "pair": "EUR_USD",
                "side": "SHORT",
                "thesis_state": "WOUNDED",
                "reason": "technical state changed",
                "invalidation": "fresh chart invalidation",
                "harvest_trigger": "fresh chart harvest",
                "gateway_required": True,
            },
            events=[event],
            selected_event=event,
            now=NOW,
        )

        self.assertEqual(review["status"], "REJECTED")
        self.assertIn("GUARDIAN_TUNING_EVENT_LIVE_ACTION_FORBIDDEN", _issue_codes(review))

    def test_slow_price_drift_accumulates_against_unacknowledged_material_baseline(self) -> None:
        position = {"pair": "EUR_USD", "side": "SHORT", "units": -30000, "owner": "operator_manual"}
        first_snapshot = _snapshot(positions=[position])
        first_events = detect_guardian_events(
            inputs={
                "snapshot": first_snapshot,
                "pair_charts": _technical_chart_payload(mid=1.17305, generated_at=NOW),
            },
            now=NOW,
        )
        _, state = evaluate_guardian_escalation(events=first_events, previous_state={}, now=NOW)

        small_snapshot = _snapshot(positions=[position])
        small_snapshot["quotes"]["EUR_USD"] = {"bid": 1.17305, "ask": 1.17315}
        small_events = detect_guardian_events(
            inputs={
                "snapshot": small_snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17310,
                    generated_at=NOW + timedelta(minutes=10),
                ),
            },
            now=NOW + timedelta(minutes=10),
        )
        small_escalation, state = evaluate_guardian_escalation(
            events=small_events,
            previous_state=state,
            now=NOW + timedelta(minutes=10),
        )
        self.assertFalse(
            any(
                event.get("event_type") == "TECHNICAL_STATE_CHANGE"
                for event in small_escalation["events_to_review"]
            )
        )

        moved_snapshot = _snapshot(positions=[position])
        moved_snapshot["quotes"]["EUR_USD"] = {"bid": 1.17335, "ask": 1.17345}
        moved_events = detect_guardian_events(
            inputs={
                "snapshot": moved_snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17340,
                    generated_at=NOW + timedelta(minutes=31),
                ),
            },
            now=NOW + timedelta(minutes=31),
        )
        escalation, _ = evaluate_guardian_escalation(
            events=moved_events,
            previous_state=state,
            now=NOW + timedelta(minutes=31),
        )

        technical = [
            event
            for event in escalation["events_to_review"]
            if event.get("event_type") == "TECHNICAL_STATE_CHANGE"
        ]
        self.assertEqual(len(technical), 1)
        self.assertIn("LARGE_PRICE_DISPLACEMENT_STATE_CHANGE", technical[0]["wake_reason_codes"])
        self.assertAlmostEqual(technical[0]["details"]["material_threshold_pips"], 2.0)

    def test_closed_candle_family_and_regime_flip_is_material_tuning_wake(self) -> None:
        position = {"pair": "EUR_USD", "side": "SHORT", "units": -30000, "owner": "operator_manual"}
        snapshot = _snapshot(positions=[position])
        first = detect_guardian_events(
            inputs={"snapshot": snapshot, "pair_charts": _technical_chart_payload(mid=1.17305, generated_at=NOW)},
            now=NOW,
        )
        _, state = evaluate_guardian_escalation(events=first, previous_state={}, now=NOW)
        flipped = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17305,
                    generated_at=NOW + timedelta(minutes=31),
                    regime="TREND_UP",
                    volatility="VOLATILE",
                    family_sign=1,
                    structure_kind="BOS_UP",
                    structure_time=NOW + timedelta(minutes=30),
                ),
            },
            now=NOW + timedelta(minutes=31),
        )
        escalation, _ = evaluate_guardian_escalation(
            events=flipped,
            previous_state=state,
            now=NOW + timedelta(minutes=31),
        )

        reasons = set(escalation["wake_reason_codes"])
        self.assertIn("TECHNICAL_STATE_CHANGE", reasons)
        self.assertIn("REGIME_STATE_CHANGE", reasons)
        self.assertIn("VOLATILITY_BUCKET_CHANGE", reasons)
        self.assertIn("TECHNICAL_FAMILY_STATE_CHANGE", reasons)
        self.assertIn("CLOSED_CANDLE_STRUCTURE_CHANGE", reasons)

    def test_same_closed_candle_recompute_drift_does_not_wake_tuning(self) -> None:
        snapshot = _snapshot()
        source_time = NOW - timedelta(minutes=1)
        first = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17305,
                    generated_at=NOW,
                    regime="TREND_DOWN",
                    family_sign=-1,
                    structure_kind="BOS_DOWN",
                    structure_time=source_time,
                ),
            },
            now=NOW,
        )
        _, state = evaluate_guardian_escalation(events=first, previous_state={}, now=NOW)

        # Recompute the same frozen closed-candle packet with contradictory
        # derived labels.  This is diagnostic drift, not new market evidence.
        recomputed = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17305,
                    generated_at=NOW + timedelta(minutes=4),
                    regime="RANGE",
                    family_sign=1,
                    structure_kind="BOS_UP",
                    structure_time=source_time,
                ),
            },
            now=NOW + timedelta(minutes=4),
        )
        escalation, _ = evaluate_guardian_escalation(
            events=recomputed,
            previous_state=state,
            now=NOW + timedelta(minutes=4),
        )

        self.assertFalse(escalation["wake_gpt"])
        self.assertNotIn("TECHNICAL_STATE_CHANGE", escalation["wake_reason_codes"])

    def test_rotating_pair_scope_retains_baseline_and_does_not_emit_false_new_event(self) -> None:
        snapshot = _snapshot()
        source_time = NOW - timedelta(minutes=1)
        first_events = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17305,
                    generated_at=NOW,
                    structure_time=source_time,
                    timeframes=("M1", "M5", "M15"),
                ),
            },
            now=NOW,
        )
        _, first_state = evaluate_guardian_escalation(
            events=first_events,
            previous_state={},
            now=NOW,
        )

        rotated_out, retained_state = evaluate_guardian_escalation(
            events=[],
            previous_state=first_state,
            now=NOW + timedelta(seconds=30),
        )
        self.assertFalse(rotated_out["wake_gpt"])
        retained = [
            item
            for item in retained_state["events"].values()
            if item.get("event_type") == "TECHNICAL_STATE_CHANGE"
        ]
        self.assertEqual(len(retained), 1)
        self.assertTrue(retained[0]["baseline_retained_out_of_current_event_set"])

        returned_events = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17305,
                    generated_at=NOW + timedelta(minutes=1),
                    structure_time=source_time,
                    timeframes=("M1", "M5", "M15"),
                ),
            },
            now=NOW + timedelta(minutes=1),
        )
        returned, _ = evaluate_guardian_escalation(
            events=returned_events,
            previous_state=retained_state,
            now=NOW + timedelta(minutes=1),
        )

        self.assertFalse(returned["wake_gpt"])
        self.assertNotIn("NEW_EVENT", returned["wake_reason_codes"])

    def test_current_technical_action_replaces_prior_pair_type_baseline(self) -> None:
        chart = _technical_chart_payload(
            mid=1.17305,
            generated_at=NOW,
            timeframes=("M1", "M5", "M15"),
        )
        flat_events = detect_guardian_events(
            inputs={"snapshot": _snapshot(), "pair_charts": chart},
            now=NOW,
        )
        _, flat_state = evaluate_guardian_escalation(
            events=flat_events,
            previous_state={},
            now=NOW,
        )
        flat_technical = [
            item
            for item in flat_state["events"].values()
            if item.get("event_type") == "TECHNICAL_STATE_CHANGE"
        ]
        self.assertEqual(len(flat_technical), 1)
        self.assertEqual(flat_technical[0]["action_hint"], "NO_ACTION")

        open_events = detect_guardian_events(
            inputs={
                "snapshot": _snapshot(
                    positions=[
                        {
                            "pair": "EUR_USD",
                            "side": "SHORT",
                            "units": -1000,
                            "owner": "trader",
                        }
                    ]
                ),
                "pair_charts": chart,
            },
            now=NOW + timedelta(seconds=30),
        )
        _, open_state = evaluate_guardian_escalation(
            events=open_events,
            previous_state=flat_state,
            now=NOW + timedelta(seconds=30),
        )
        current_technical = [
            item
            for item in open_state["events"].values()
            if item.get("event_type") == "TECHNICAL_STATE_CHANGE"
            and item.get("pair") == "EUR_USD"
        ]

        self.assertEqual(len(current_technical), 1)
        self.assertEqual(current_technical[0]["action_hint"], "HOLD")
        self.assertFalse(
            current_technical[0].get("baseline_retained_out_of_current_event_set", False)
        )

    def test_rotating_pair_scope_retains_stale_input_baseline_without_false_new_event(self) -> None:
        snapshot = _snapshot()
        stale_packet = _technical_chart_payload(
            mid=1.17305,
            generated_at=NOW,
            timeframes=("M15",),
        )
        first_events = detect_guardian_events(
            inputs={"snapshot": snapshot, "pair_charts": stale_packet},
            now=NOW,
        )
        self.assertEqual(
            [event.event_type for event in first_events if event.event_type.startswith("TECHNICAL_")],
            ["TECHNICAL_INPUT_STALE"],
        )
        _, first_state = evaluate_guardian_escalation(
            events=first_events,
            previous_state={},
            now=NOW,
        )
        _, retained_state = evaluate_guardian_escalation(
            events=[],
            previous_state=first_state,
            now=NOW + timedelta(seconds=30),
        )

        returned_events = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": {
                    **stale_packet,
                    "generated_at_utc": (NOW + timedelta(minutes=1)).isoformat(),
                },
            },
            now=NOW + timedelta(minutes=1),
        )
        returned, _ = evaluate_guardian_escalation(
            events=returned_events,
            previous_state=retained_state,
            now=NOW + timedelta(minutes=1),
        )

        self.assertFalse(returned["wake_gpt"])
        self.assertNotIn("NEW_EVENT", returned["wake_reason_codes"])

    def test_quick_full_surface_switch_without_fast_close_does_not_wake_tuning(self) -> None:
        snapshot = _snapshot()
        source_time = NOW - timedelta(minutes=1)
        full = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17305,
                    generated_at=NOW,
                    regime="TREND_DOWN",
                    family_sign=-1,
                    structure_time=source_time,
                ),
            },
            now=NOW,
        )
        _, state = evaluate_guardian_escalation(events=full, previous_state={}, now=NOW)
        quick = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17305,
                    generated_at=NOW + timedelta(minutes=4),
                    regime="RANGE",
                    family_sign=1,
                    structure_kind="BOS_UP",
                    structure_time=source_time,
                    timeframes=("M1", "M5", "M15"),
                ),
            },
            now=NOW + timedelta(minutes=4),
        )

        escalation, _ = evaluate_guardian_escalation(
            events=quick,
            previous_state=state,
            now=NOW + timedelta(minutes=4),
        )

        self.assertFalse(escalation["wake_gpt"])
        self.assertNotIn("REGIME_STATE_CHANGE", escalation["wake_reason_codes"])

    def test_each_fast_timeframe_close_can_prove_new_technical_source(self) -> None:
        snapshot = _snapshot()
        source_time = NOW - timedelta(minutes=1)
        for timeframe in ("M1", "M5", "M15"):
            with self.subTest(timeframe=timeframe):
                first = detect_guardian_events(
                    inputs={
                        "snapshot": snapshot,
                        "pair_charts": _technical_chart_payload(
                            mid=1.17305,
                            generated_at=NOW,
                            regime="TREND_DOWN",
                            family_sign=-1,
                            structure_time=source_time,
                            timeframes=("M1", "M5", "M15"),
                        ),
                    },
                    now=NOW,
                )
                _, state = evaluate_guardian_escalation(events=first, previous_state={}, now=NOW)
                advanced_times = {
                    "M1": source_time,
                    "M5": source_time,
                    "M15": source_time,
                }
                advanced_times[timeframe] = source_time + timedelta(minutes=1)
                changed = detect_guardian_events(
                    inputs={
                        "snapshot": snapshot,
                        "pair_charts": _technical_chart_payload(
                            mid=1.17305,
                            generated_at=NOW + timedelta(minutes=31),
                            regime="TREND_UP",
                            family_sign=1,
                            structure_kind="BOS_UP",
                            structure_time=source_time,
                            timeframes=("M1", "M5", "M15"),
                            candle_times=advanced_times,
                        ),
                    },
                    now=NOW + timedelta(minutes=31),
                )

                escalation, _ = evaluate_guardian_escalation(
                    events=changed,
                    previous_state=state,
                    now=NOW + timedelta(minutes=31),
                )

                self.assertTrue(escalation["wake_gpt"])
                self.assertIn("REGIME_STATE_CHANGE", escalation["wake_reason_codes"])

    def test_legacy_technical_state_without_watermarks_allows_one_compatible_review(self) -> None:
        snapshot = _snapshot()
        source_time = NOW - timedelta(minutes=1)
        first = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17305,
                    generated_at=NOW,
                    regime="TREND_DOWN",
                    family_sign=-1,
                    structure_time=source_time,
                    timeframes=("M1", "M5", "M15"),
                ),
            },
            now=NOW,
        )
        _, state = evaluate_guardian_escalation(events=first, previous_state={}, now=NOW)
        technical_state = next(
            item
            for item in state["events"].values()
            if item.get("event_type") == "TECHNICAL_STATE_CHANGE"
        )
        technical_state["details"].pop("closed_candle_watermarks", None)
        changed = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17305,
                    generated_at=NOW + timedelta(minutes=31),
                    regime="TREND_UP",
                    family_sign=1,
                    structure_time=source_time,
                    timeframes=("M1", "M5", "M15"),
                ),
            },
            now=NOW + timedelta(minutes=31),
        )

        escalation, _ = evaluate_guardian_escalation(
            events=changed,
            previous_state=state,
            now=NOW + timedelta(minutes=31),
        )

        self.assertTrue(escalation["wake_gpt"])
        self.assertIn("TECHNICAL_STATE_CHANGE", escalation["wake_reason_codes"])

    def test_present_but_empty_watermark_contract_cannot_authorize_derived_drift(self) -> None:
        snapshot = _snapshot()
        source_time = NOW - timedelta(minutes=1)
        first = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17305,
                    generated_at=NOW,
                    regime="TREND_DOWN",
                    family_sign=-1,
                    structure_time=source_time,
                    timeframes=("M1", "M5", "M15"),
                ),
            },
            now=NOW,
        )
        _, state = evaluate_guardian_escalation(events=first, previous_state={}, now=NOW)
        missing_candles = _technical_chart_payload(
            mid=1.17305,
            generated_at=NOW + timedelta(minutes=31),
            regime="RANGE",
            family_sign=1,
            structure_kind="BOS_UP",
            structure_time=source_time,
            timeframes=("M1", "M5", "M15"),
        )
        for view in missing_candles["charts"][0]["views"]:
            view["recent_candles"] = []
        snapshot["fetched_at_utc"] = (NOW + timedelta(minutes=31)).isoformat()
        recomputed = detect_guardian_events(
            inputs={"snapshot": snapshot, "pair_charts": missing_candles},
            now=NOW + timedelta(minutes=31),
        )

        escalation, _ = evaluate_guardian_escalation(
            events=recomputed,
            previous_state=state,
            now=NOW + timedelta(minutes=31),
        )
        legacy_state = json.loads(json.dumps(state))
        legacy_technical = next(
            item
            for item in legacy_state["events"].values()
            if item.get("event_type") == "TECHNICAL_STATE_CHANGE"
        )
        legacy_technical["details"].pop("closed_candle_watermarks", None)
        legacy_escalation, _ = evaluate_guardian_escalation(
            events=recomputed,
            previous_state=legacy_state,
            now=NOW + timedelta(minutes=31),
        )

        self.assertTrue(escalation["wake_gpt"])
        self.assertTrue(
            any(
                item.get("event_type") == "TECHNICAL_INPUT_STALE"
                for item in escalation["events_to_review"]
            )
        )
        self.assertNotIn("REGIME_STATE_CHANGE", escalation["wake_reason_codes"])
        self.assertFalse(
            any(
                item.get("event_type") == "TECHNICAL_STATE_CHANGE"
                for item in escalation["events_to_review"]
            )
        )
        self.assertFalse(
            any(
                item.get("event_type") == "TECHNICAL_STATE_CHANGE"
                for item in legacy_escalation["events_to_review"]
            )
        )

    def test_rollover_spread_does_not_create_major_figure_failed_acceptance(self) -> None:
        snapshot = _snapshot()
        snapshot["quotes"] = {
            "EUR_NZD": {"bid": 1.97986, "ask": 1.98139},
        }

        events = detect_guardian_events(
            inputs={"snapshot": snapshot, "pair_charts": {}},
            now=NOW,
        )

        self.assertFalse(any(event.event_type == "FAILED_ACCEPTANCE" for event in events))
        self.assertTrue(any(event.event_type == "SPREAD_ANOMALY" for event in events))

    def test_stale_bounded_chart_emits_one_stale_input_event_not_fake_technical_state(self) -> None:
        snapshot = _snapshot(
            positions=[{"pair": "EUR_USD", "side": "SHORT", "units": -30000, "owner": "operator_manual"}]
        )
        events = detect_guardian_events(
            inputs={
                "snapshot": snapshot,
                "pair_charts": _technical_chart_payload(
                    mid=1.17305,
                    generated_at=NOW - timedelta(minutes=20),
                ),
            },
            now=NOW,
        )

        self.assertEqual(
            [event.event_type for event in events if event.event_type.startswith("TECHNICAL_")],
            ["TECHNICAL_INPUT_STALE"],
        )

    def test_naive_or_future_chart_generated_at_is_fail_closed(self) -> None:
        for generated_at, expected_status in (
            ("2026-06-30T03:30:00", "INVALID_GENERATED_AT"),
            ((NOW + timedelta(minutes=10)).isoformat(), "FUTURE_GENERATED_AT"),
        ):
            with self.subTest(generated_at=generated_at):
                charts = _technical_chart_payload(mid=1.17305, generated_at=NOW)
                charts["generated_at_utc"] = generated_at
                events = detect_guardian_events(
                    inputs={"snapshot": _snapshot(), "pair_charts": charts},
                    now=NOW,
                )

                technical = [
                    event for event in events if event.event_type == "TECHNICAL_INPUT_STALE"
                ]
                self.assertEqual(len(technical), 1)
                self.assertEqual(technical[0].details["status"], expected_status)

    def test_accepted_harvest_ack_does_not_retrigger_price_entered_every_probe(self) -> None:
        events = _events_from_chart(
            {
                "event_type": "HARVEST_ZONE",
                "thesis": "open position harvest",
                "price_zone": "upper rail",
                "severity": "P1",
                "action_hint": "HARVEST",
            }
        )
        first, state = evaluate_guardian_escalation(events=events, previous_state={}, now=NOW)
        event = next(item for item in events if item.event_type == "HARVEST_ZONE")
        dispatcher_state = {
            "reviewed_events": {
                event.dedupe_key: {
                    "event_id": event.event_id,
                    "receipt_written": True,
                    "last_reviewed_at_utc": NOW.isoformat(),
                    "selected_event": event.to_payload(),
                }
            }
        }

        second, _ = evaluate_guardian_escalation(
            events=events,
            previous_state=state,
            dispatcher_state=dispatcher_state,
            now=NOW + timedelta(seconds=30),
        )

        self.assertIn("PRICE_ENTERED_HARVEST_ZONE", first["wake_reason_codes"])
        self.assertFalse(second["wake_gpt"])

    def test_stale_technical_input_reappears_after_disappearance_even_with_old_dispatch_ack(self) -> None:
        snapshot = _snapshot(
            positions=[{"pair": "EUR_USD", "side": "SHORT", "units": -30000, "owner": "operator_manual"}]
        )
        stale_events = detect_guardian_events(inputs={"snapshot": snapshot, "pair_charts": {}}, now=NOW)
        _, stale_state = evaluate_guardian_escalation(events=stale_events, previous_state={}, now=NOW)
        stale_event = next(event for event in stale_events if event.event_type == "TECHNICAL_INPUT_STALE")
        dispatcher_state = {
            "reviewed_events": {
                stale_event.dedupe_key: {
                    "event_id": stale_event.event_id,
                    "receipt_written": True,
                    "last_reviewed_at_utc": NOW.isoformat(),
                    "selected_event": stale_event.to_payload(),
                }
            }
        }
        fresh_events = detect_guardian_events(
            inputs={"snapshot": snapshot, "pair_charts": _technical_chart_payload(mid=1.17305, generated_at=NOW)},
            now=NOW,
        )
        _, fresh_state = evaluate_guardian_escalation(
            events=fresh_events,
            previous_state=stale_state,
            dispatcher_state=dispatcher_state,
            now=NOW,
        )
        reappeared = detect_guardian_events(
            inputs={"snapshot": snapshot, "pair_charts": {}},
            now=NOW + timedelta(minutes=1),
        )
        escalation, _ = evaluate_guardian_escalation(
            events=reappeared,
            previous_state=fresh_state,
            dispatcher_state=dispatcher_state,
            now=NOW + timedelta(minutes=1),
        )

        self.assertTrue(escalation["wake_gpt"])
        self.assertIn("NEW_EVENT", escalation["wake_reason_codes"])

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

    def test_margin_ratio_drift_is_not_fx_price_displacement(self) -> None:
        first_snapshot = _snapshot()
        first_snapshot["account"].update(
            {
                "nav_jpy": 295832.6972,
                "margin_used_jpy": 254416.119592,
                "margin_available_jpy": 41416.577608,
            }
        )
        first_events = detect_guardian_events(
            inputs={"snapshot": first_snapshot},
            now=NOW,
        )
        first, state = evaluate_guardian_escalation(
            events=first_events,
            previous_state={},
            now=NOW,
        )
        first_margin = [
            event
            for event in first["events_to_review"]
            if event.get("event_type") == "MARGIN_PRESSURE"
        ]
        self.assertEqual(len(first_margin), 1)
        self.assertIn(
            "MARGIN_RISK_THRESHOLD_CROSSED",
            first_margin[0]["wake_reason_codes"],
        )
        self.assertEqual(
            first_margin[0]["details"]["max_margin_utilization_pct"],
            95.0,
        )
        self.assertEqual(first_margin[0]["severity"], "P1")
        self.assertFalse(
            first_margin[0]["details"]["fresh_entry_risk_block_active"]
        )
        self.assertTrue(
            first_margin[0]["details"]["fresh_entry_risk_observation_only"]
        )
        self.assertEqual(
            first_margin[0]["details"]["fresh_entry_margin_contract"],
            "QR_GUARDIAN_P1_MARGIN_WARNING_V1",
        )

        later = NOW + timedelta(minutes=5)
        second_snapshot = _snapshot()
        second_snapshot["fetched_at_utc"] = later.isoformat()
        second_snapshot["account"].update(
            {
                "nav_jpy": 291704.3565,
                "margin_used_jpy": 254366.198868,
                "margin_available_jpy": 37338.157632,
            }
        )
        second_events = detect_guardian_events(
            inputs={"snapshot": second_snapshot},
            now=later,
        )
        second, _ = evaluate_guardian_escalation(
            events=second_events,
            previous_state=state,
            now=later,
        )

        self.assertFalse(second["wake_gpt"])
        suppressed_margin = [
            event
            for event in second["suppressed_events"]
            if event.get("event_type") == "MARGIN_PRESSURE"
        ]
        self.assertEqual(len(suppressed_margin), 1)
        self.assertEqual(
            suppressed_margin[0]["suppressed_reason"],
            "THROTTLED",
        )
        self.assertNotIn(
            "LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",
            suppressed_margin[0].get("wake_reason_codes", []),
        )

    def test_p0_margin_pressure_manual_only_reports_without_reduction_target(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "currentUnits": "-30000",
                    "owner": "trader",
                    "id": "manual-a",
                    "entry_price": 1.14,
                    "raw": {
                        "operator_manual_position": {
                            "packet_type": "OPERATOR_MANUAL_POSITION",
                        }
                    },
                },
                {
                    "pair": "USD_JPY",
                    "side": "LONG",
                    "current_units": 10_000,
                    "owner": "trader",
                    "tradeID": "manual-b",
                    "entry_price": 157.0,
                    "operator_manual_position": {
                        "packet_type": "OPERATOR_MANUAL_POSITION",
                    },
                },
                {
                    "pair": "GBP_USD",
                    "side": "SHORT",
                    "units": 5_000,
                    "owner": "operator_manual",
                    "trade_id": "manual-c",
                    "entry_price": 1.33,
                },
            ]
        )
        snapshot["account"].update(
            {
                "margin_used_jpy": 190_000.0,
                "margin_available_jpy": 10_000.0,
            }
        )

        events = detect_guardian_events(inputs={"snapshot": snapshot}, now=NOW)
        margin = next(event for event in events if event.event_type == "MARGIN_PRESSURE")

        self.assertEqual(margin.severity, "P0")
        self.assertEqual(margin.action_hint, "HOLD")
        self.assertEqual(margin.thesis_state, "EMERGENCY")
        self.assertTrue(margin.details["operator_manual_only_exposure"])
        self.assertEqual(
            margin.details["operator_manual_trade_ids"],
            ["manual-a", "manual-b", "manual-c"],
        )
        self.assertEqual(
            margin.details["system_reduction_candidate_trade_ids"],
            [],
        )
        self.assertEqual(
            margin.details["executable_reduction_target_trade_ids"],
            [],
        )
        self.assertTrue(margin.details["fresh_entry_risk_block_active"])
        self.assertFalse(margin.details["fresh_entry_risk_observation_only"])
        self.assertEqual(
            margin.details["fresh_entry_margin_contract"],
            "QR_GUARDIAN_P0_MARGIN_HARD_CAP_V1",
        )
        self.assertEqual(
            margin.details["fresh_entry_risk_block_reason"],
            "MARGIN_PRESSURE",
        )

        contract = build_guardian_trigger_contract(
            snapshot=snapshot,
            order_intents={},
            existing_contract={},
            now=NOW,
        )
        for entry in contract["entries"]:
            self.assertEqual(entry["owner"], "OPERATOR_MANUAL")
            for bucket in CONTRACT_TRIGGER_BUCKETS:
                for trigger in entry[bucket]:
                    self.assertEqual(trigger["action_hint"], "HOLD")

    def test_system_contract_actions_do_not_survive_operator_manual_transition(self) -> None:
        system_position = {
            "pair": "EUR_USD",
            "side": "SHORT",
            "units": 3_000,
            "owner": "trader",
            "trade_id": "manual-transition-a",
            "entry_price": 1.14,
            "thesis": "same durable thesis",
        }
        prior = build_guardian_trigger_contract(
            snapshot=_snapshot(positions=[system_position]),
            order_intents={},
            existing_contract={},
            now=NOW,
        )
        self.assertEqual(
            prior["entries"][0]["invalidation_triggers"][0]["action_hint"],
            "REDUCE",
        )
        manual_position = {
            **system_position,
            "owner": "trader",
            "raw": {
                "operator_manual_position": {
                    "packet_type": OPERATOR_MANUAL_POSITION_PACKET,
                    "operator_decision": "OPERATOR_CONFIRMED_MANUAL_OWNED",
                    "management_intent": "NO_TOUCH_OBSERVE_ONLY",
                    "no_live_side_effects": True,
                    "loss_side_auto_close_allowed": False,
                    "auto_sl_attach_allowed": False,
                    "auto_tp_modify_allowed": False,
                }
            },
        }

        current = build_guardian_trigger_contract(
            snapshot=_snapshot(positions=[manual_position]),
            order_intents={},
            existing_contract=prior,
            now=NOW,
        )
        entry = current["entries"][0]

        self.assertEqual(entry["owner"], "OPERATOR_MANUAL")
        self.assertEqual(entry["ownership_audit"]["status"], "OPERATOR_MANUAL")
        self.assertIn(
            "operator_manual_position packet present",
            entry["ownership_audit"]["evidence"],
        )
        for bucket in CONTRACT_TRIGGER_BUCKETS:
            for trigger in entry[bucket]:
                self.assertEqual(trigger["owner"], "OPERATOR_MANUAL")
                self.assertEqual(trigger["action_hint"], "HOLD")

    def test_p0_margin_pressure_mixed_exposure_targets_only_system_positions(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": 30_000,
                    "owner": "operator_manual",
                    "trade_id": "manual-a",
                    "entry_price": 1.14,
                },
                {
                    "pair": "USD_JPY",
                    "side": "LONG",
                    "units": 10_000,
                    "owner": "trader",
                    "trade_id": "system-a",
                    "entry_price": 157.0,
                },
            ]
        )
        snapshot["account"].update(
            {
                "margin_used_jpy": 190_000.0,
                "margin_available_jpy": 10_000.0,
            }
        )

        events = detect_guardian_events(inputs={"snapshot": snapshot}, now=NOW)
        margin = next(event for event in events if event.event_type == "MARGIN_PRESSURE")

        self.assertEqual(margin.action_hint, "REDUCE")
        self.assertFalse(margin.details["operator_manual_only_exposure"])
        self.assertEqual(
            margin.details["system_reduction_candidate_trade_ids"],
            ["system-a"],
        )
        self.assertEqual(
            margin.details["executable_reduction_target_trade_ids"],
            ["system-a"],
        )
        self.assertNotIn(
            "manual-a",
            margin.details["executable_reduction_target_trade_ids"],
        )

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

        self.assertEqual(event_types, {"CONTRACT_STALE", "TECHNICAL_INPUT_STALE"})

    def test_self_improvement_pending_cancel_review_wakes_before_age_stale(self) -> None:
        snapshot = _snapshot(
            orders=[
                {
                    "order_id": "473022",
                    "pair": "EUR_USD",
                    "owner": "trader",
                    "state": "PENDING",
                    "units": 11000,
                    "price": 1.14334,
                    "raw": {
                        "createTime": (NOW - timedelta(minutes=10)).isoformat(),
                        "clientExtensions": {
                            "comment": "qr-vnext lane=failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
                        },
                    },
                }
            ]
        )
        self_improvement = {
            "findings": [
                {
                    "code": "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                    "evidence": {
                        "cancel_review_order_ids": ["473022"],
                        "orders": [
                            {
                                "order_id": "473022",
                                "parent_lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                                "current_candidate_count": 1,
                                "current_live_ready_candidate_count": 0,
                                "review_reasons": [
                                    {"code": "PENDING_CURRENT_CANDIDATE_NOT_LIVE_READY"}
                                ],
                            }
                        ],
                    },
                }
            ]
        }

        events = detect_guardian_events(
            inputs={"snapshot": snapshot, "self_improvement_audit": self_improvement},
            now=NOW,
        )
        escalation, _ = evaluate_guardian_escalation(events=events, previous_state={}, now=NOW)
        pending_events = [event for event in events if event.event_type == "STALE_PENDING"]

        self.assertEqual(len(pending_events), 1)
        self.assertEqual(pending_events[0].action_hint, "CANCEL_PENDING")
        self.assertEqual(pending_events[0].recommended_review_type, "PENDING_CANCEL_REVIEW")
        self.assertEqual(pending_events[0].details["source"], "self_improvement_audit")
        self.assertEqual(pending_events[0].details["order_id"], "473022")
        self.assertTrue(escalation["wake_gpt"])

    def test_self_improvement_pending_cancel_review_requires_current_broker_order(self) -> None:
        self_improvement = {
            "findings": [
                {
                    "code": "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                    "evidence": {"cancel_review_order_ids": ["missing-order"]},
                }
            ]
        }

        events = detect_guardian_events(
            inputs={"snapshot": _snapshot(), "self_improvement_audit": self_improvement},
            now=NOW,
        )

        self.assertEqual(events, [])

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

    def test_operator_confirmed_eur_usd_472987_is_monitored_as_operator_manual_keep(self) -> None:
        snapshot = _snapshot(
            positions=[
                {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": 30000,
                    "owner": "operator_manual",
                    "thesis": "operator-confirmed manual EUR_USD short; keep open",
                    "thesis_state": "ALIVE",
                    "trade_id": "472987",
                    "entry_price": 1.14048,
                    "take_profit": 1.13800,
                    "stop_loss": None,
                    "raw": {
                        "id": "472987",
                        "instrument": "EUR_USD",
                        "currentUnits": "-30000",
                        "price": "1.14048",
                        "operator_manual_position": {
                            "packet_type": OPERATOR_MANUAL_POSITION_PACKET,
                            "classification": "OPERATOR_MANUAL",
                            "operator_decision": "OPERATOR_CONFIRMED_MANUAL_OWNED",
                            "management_intent": "KEEP",
                            "operator_confirmation_source": "chat_operator_confirmation",
                            "no_live_side_effects": True,
                            "system_pl_counted": False,
                            "same_theme_auto_add_allowed": False,
                            "loss_side_auto_close_allowed": False,
                            "auto_sl_attach_allowed": False,
                            "auto_tp_modify_allowed": False,
                        },
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
        self.assertEqual(entry["owner"], "OPERATOR_MANUAL")
        self.assertEqual(entry["thesis_state"], "ALIVE")
        self.assertEqual(audit["status"], "OPERATOR_MANUAL")
        self.assertEqual(audit["operator_decision"], "OPERATOR_CONFIRMED_MANUAL_OWNED")
        self.assertEqual(audit["management_intent"], "KEEP")
        self.assertEqual(audit["operator_confirmation_source"], "chat_operator_confirmation")
        self.assertFalse(audit["system_pl_counted"])
        self.assertFalse(audit["same_theme_auto_add_allowed"])
        self.assertFalse(audit["loss_side_auto_close_allowed"])
        self.assertFalse(audit["auto_sl_attach_allowed"])
        self.assertFalse(audit["auto_tp_modify_allowed"])
        for bucket in (
            "harvest_triggers",
            "no_add_triggers",
            "wounded_triggers",
            "invalidation_triggers",
            "emergency_triggers",
        ):
            self.assertTrue(entry[bucket], bucket)
            self.assertEqual(entry[bucket][0]["trade_id"], "472987")
            self.assertEqual(entry[bucket][0]["owner"], "OPERATOR_MANUAL")
        self.assertEqual(entry["harvest_triggers"][0]["action_hint"], "HOLD")
        self.assertEqual(entry["invalidation_triggers"][0]["action_hint"], "HOLD")
        self.assertEqual(entry["emergency_triggers"][0]["action_hint"], "HOLD")
        self.assertIn("do not automatically loss-close", entry["invalidation_triggers"][0]["evidence_required"])

    def test_guardian_trigger_contract_cli_applies_operator_review_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "snapshot.json"
            intents_path = root / "order_intents.json"
            manual_path = root / "operator_manual_positions.json"
            review_path = root / "guardian_receipt_operator_review.json"
            output_path = root / "guardian_trigger_contract.json"
            report_path = root / "guardian_trigger_contract.md"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "fetched_at_utc": "2026-07-02T13:10:16+00:00",
                        "positions": [
                            {
                                "trade_id": "472987",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "units": 30_000,
                                "entry_price": 1.14048,
                                "avg_entry": 1.14048,
                                "unrealized_pl_jpy": -20723.0081,
                                "take_profit": 1.1388,
                                "stop_loss": None,
                                "owner": "unknown",
                                "raw": {"currentUnits": "-30000"},
                            }
                        ],
                        "quotes": {
                            "EUR_USD": {
                                "bid": 1.14730,
                                "ask": 1.14738,
                                "timestamp_utc": "2026-07-02T13:10:16+00:00",
                            }
                        },
                    }
                )
            )
            intents_path.write_text("{}")
            review_path.write_text(
                json.dumps(
                    {
                        "operator_position_reviews": [
                            {
                                "trade_id": "472987",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "units": 30_000,
                                "avg_entry": 1.14048,
                                "owner": "OPERATOR_MANUAL",
                                "operator_decision": "OPERATOR_CONFIRMED_MANUAL_OWNED",
                                "management_intent": "KEEP",
                                "reason": "operator explicitly confirmed manual EUR_USD should remain open",
                                "operator_confirmation_source": "chat_operator_confirmation",
                                "no_live_side_effects": True,
                                "system_pl_counted": False,
                                "same_theme_auto_add_allowed": False,
                                "loss_side_auto_close_allowed": False,
                                "auto_sl_attach_allowed": False,
                                "auto_tp_modify_allowed": False,
                            }
                        ]
                    }
                )
            )

            exit_code = main(
                [
                    "guardian-trigger-contract",
                    "--snapshot",
                    str(snapshot_path),
                    "--order-intents",
                    str(intents_path),
                    "--existing",
                    str(root / "missing_existing.json"),
                    "--operator-manual-positions",
                    str(manual_path),
                    "--operator-review",
                    str(review_path),
                    "--output",
                    str(output_path),
                    "--report",
                    str(report_path),
                ]
            )

            payload = json.loads(output_path.read_text())

        self.assertEqual(exit_code, 0)
        entry = next(item for item in payload["entries"] if item.get("trade_id") == "472987")
        self.assertEqual(entry["owner"], "OPERATOR_MANUAL")
        self.assertEqual(entry["thesis_state"], "ALIVE")
        audit = entry["ownership_audit"]
        self.assertEqual(audit["operator_decision"], "OPERATOR_CONFIRMED_MANUAL_OWNED")
        self.assertEqual(audit["management_intent"], "KEEP")
        self.assertFalse(audit["system_pl_counted"])
        self.assertFalse(audit["same_theme_auto_add_allowed"])
        self.assertFalse(audit["loss_side_auto_close_allowed"])
        self.assertFalse(audit["auto_sl_attach_allowed"])
        self.assertFalse(audit["auto_tp_modify_allowed"])
        for bucket in (
            "harvest_triggers",
            "no_add_triggers",
            "wounded_triggers",
            "invalidation_triggers",
            "emergency_triggers",
        ):
            self.assertTrue(entry[bucket], bucket)

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
            self.assertEqual(trigger["action_hint"], "HOLD")

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

    def test_contract_stale_reason_uses_blocking_issues_not_watch_only_warnings(self) -> None:
        contract = _contract(
            entry_overrides={
                **_open_required_trigger_fields(),
                "trade_id": "1",
                "units": 1000,
                "avg_entry": 1.171,
                "next_review_deadline_utc": (NOW - timedelta(minutes=1)).isoformat(),
            }
        )
        watch_only = _contract(entry_overrides={"next_review_deadline_utc": (NOW - timedelta(minutes=1)).isoformat()})[
            "entries"
        ][0]
        contract["entries"].append(watch_only)
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

        contract_stale = next(event for event in events if event.event_type == "CONTRACT_STALE")
        self.assertIn("CONTRACT_ENTRY_DEADLINE_EXPIRED", contract_stale.price_zone)
        self.assertNotIn("CONTRACT_ENTRY_WATCH_DEADLINE_EXPIRED", contract_stale.price_zone)

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

    def test_range_rail_repair_adds_watch_only_add_trigger_without_permission(self) -> None:
        contract = build_guardian_trigger_contract(
            snapshot=_snapshot(),
            order_intents={},
            existing_contract={},
            range_rail_geometry_repair=_range_rail_repair_payload(),
            now=NOW,
        )

        entry = contract["entries"][0]
        trigger = entry["add_triggers"][0]

        self.assertEqual(entry["pair"], "USD_CAD")
        self.assertEqual(entry["side"], "LONG")
        self.assertTrue(entry["watch_only"])
        self.assertEqual(entry["watch_only_reason"], "WAIT_FOR_RANGE_RAIL_RECHECK")
        self.assertTrue(entry["current"])
        self.assertFalse(entry["range_rail_watch"]["live_permission_allowed"])
        self.assertFalse(trigger["live_permission_allowed"])
        self.assertTrue(trigger["contract_triggers_do_not_execute"])
        self.assertEqual(trigger["condition"]["metric"], "mid")
        self.assertEqual(trigger["condition"]["operator"], "<=")
        self.assertAlmostEqual(trigger["condition"]["value"], 1.4165785)
        self.assertIn("NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION", trigger["preserve_blockers"])
        self.assertEqual(validate_guardian_trigger_contract(contract, now=NOW)["status"], "VALID")

    def test_range_rail_watch_trigger_waits_until_quote_reaches_rail(self) -> None:
        contract = build_guardian_trigger_contract(
            snapshot=_snapshot(),
            order_intents={},
            existing_contract={},
            range_rail_geometry_repair=_range_rail_repair_payload(),
            now=NOW,
        )
        snapshot = _snapshot()
        snapshot["quotes"]["USD_CAD"] = {"bid": 1.4170, "ask": 1.4171}

        events = detect_guardian_events(inputs={"snapshot": snapshot, "trigger_contract": contract}, now=NOW)

        self.assertNotIn("CONTRACT_ADD_TRIGGER", {event.event_type for event in events})

    def test_range_rail_watch_trigger_wakes_when_quote_reaches_rail(self) -> None:
        contract = build_guardian_trigger_contract(
            snapshot=_snapshot(),
            order_intents={},
            existing_contract={},
            range_rail_geometry_repair=_range_rail_repair_payload(),
            now=NOW,
        )
        snapshot = _snapshot()
        snapshot["quotes"]["USD_CAD"] = {"bid": 1.41650, "ask": 1.41654}

        events = detect_guardian_events(inputs={"snapshot": snapshot, "trigger_contract": contract}, now=NOW)
        escalation, _ = evaluate_guardian_escalation(events=events, previous_state={}, now=NOW)

        add_event = next(event for event in events if event.event_type == "CONTRACT_ADD_TRIGGER")
        self.assertEqual(add_event.pair, "USD_CAD")
        self.assertEqual(add_event.direction, "LONG")
        self.assertEqual(add_event.action_hint, "ADD")
        self.assertIn("mid <= 1.4165785 fired", add_event.price_zone)
        self.assertFalse(add_event.details["contract_trigger"]["live_permission_allowed"])
        self.assertTrue(escalation["wake_gpt"])

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


def _range_rail_repair_payload() -> dict:
    return {
        "schema_version": "range_rail_geometry_repair_v1",
        "status": "RANGE_RAIL_RECHECK_BUILT",
        "generated_at_utc": NOW.isoformat(),
        "read_only": True,
        "live_side_effects": [],
        "live_permission_allowed": False,
        "top_lane": {
            "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
            "pair": "USD_CAD",
            "direction": "LONG",
            "strategy_family": "BREAKOUT_FAILURE",
            "vehicle": "LIMIT",
            "status": "EVIDENCE_ACQUISITION",
            "blockers": ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION", "SPREAD_TOO_WIDE"],
            "range_box": {
                "low": 1.41528,
                "high": 1.41899,
                "current": 1.4181,
                "box_position": 0.7601,
                "required_zone": "LONG_DISCOUNT_LOWER_RAIL",
                "threshold": {"box_position_lte": 0.35},
                "rail_status": "RANGE_RAIL_NOT_REACHED",
            },
            "range_rotation_counterpart": {
                "preferred_lane_id": "range_trader:USD_CAD:LONG:RANGE_ROTATION",
                "blocker_codes": ["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
            },
            "counterpart_geometry": {
                "status": "COUNTERPART_PRICE_GEOMETRY_INCOMPLETE",
                "geometry_ready": False,
                "reasons": ["ENTRY_NOT_INSIDE_RANGE_BOX"],
            },
            "rail_success_condition": {
                "pair": "USD_CAD",
                "direction": "LONG",
                "required_zone": "LONG_DISCOUNT_LOWER_RAIL",
                "current_box_position": 0.7601,
                "range_low_price": 1.41528,
                "range_high_price": 1.41899,
                "current_price": 1.4181,
                "required_box_position_lte": 0.35,
                "must_preserve_blockers": [
                    "SPREAD_TOO_WIDE",
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                ],
            },
            "repair_status": "RANGE_RAIL_RECHECK_BUILT",
        },
    }


def _events_from_chart(event: dict, *, pair: str = "EUR_USD"):
    return detect_guardian_events(
        inputs={
            "snapshot": _snapshot(),
            "pair_charts": {"charts": [{"pair": pair, "guardian_events": [event]}]},
        },
        now=NOW,
    )


def _technical_chart_payload(
    *,
    mid: float,
    generated_at: datetime,
    regime: str = "TREND_DOWN",
    volatility: str = "NORMAL",
    family_sign: int = -1,
    structure_kind: str = "BOS_DOWN",
    structure_time: datetime = NOW - timedelta(minutes=1),
    timeframes: tuple[str, ...] = ("M5", "M15", "M30", "H1"),
    candle_times: dict[str, datetime] | None = None,
) -> dict:
    views = []
    for timeframe in timeframes:
        candle_time = (candle_times or {}).get(timeframe, structure_time)
        structure_events = []
        if timeframe in {"M15", "M30", "H1"}:
            structure_events = [
                {
                    "kind": structure_kind,
                    "timestamp": structure_time.isoformat(),
                    "close_confirmed": True,
                }
            ]
        views.append(
            {
                "granularity": timeframe,
                "recent_candles": [
                    {
                        "t": candle_time.isoformat(),
                        "o": mid,
                        "h": mid,
                        "l": mid,
                        "c": mid,
                        "complete": True,
                    }
                ],
                "indicators": {
                    "close": mid,
                    "atr_pips": 2.0 if timeframe == "M5" else 4.0,
                    "regime_quantile": volatility,
                    "supertrend_dir": family_sign,
                    "linreg_slope_20": float(family_sign),
                },
                "family_scores": {
                    "trend_score": 0.8 * family_sign,
                    "mean_rev_score": 0.6 * family_sign,
                    "breakout_score": 0.7,
                },
                "structure": {"structure_events": structure_events},
            }
        )
    return {
        "generated_at_utc": generated_at.isoformat(),
        "guardian_monitor_pairs": ["EUR_USD"],
        "guardian_monitor_scope": {"EUR_USD": ["open_position", "operator_manual_read_only"]},
        "charts": [
            {
                "pair": "EUR_USD",
                "dominant_regime": regime,
                "views": views,
            }
        ],
    }


def _external_freshness_receipt(*, pair_charts_sha256: str) -> dict:
    return {
        "checked_at_utc": NOW.isoformat(),
        "source_generated_at_utc": NOW.isoformat(),
        "source_pair_charts_sha256": pair_charts_sha256,
        "rows": [
            {
                "pair": "EUR_USD",
                "timeframe": timeframe,
                "status": "FRESH",
                "max_age_seconds": max_age,
                "latest_complete_candle_closed_at_utc": NOW.isoformat(),
            }
            for timeframe, max_age in (
                ("M1", 120.0),
                ("M5", 600.0),
                ("M15", 1800.0),
            )
        ],
    }


def _canonical_integrity_chart_payload(
    *,
    stale_by: timedelta = timedelta(0),
) -> dict:
    pair = "EUR_USD"
    factor = instrument_pip_factor(pair)
    normal_spread_pips = NORMAL_SPREAD_PIPS[pair]
    half_spread = normal_spread_pips / factor / 2.0
    base = 1.17305
    views = []
    for timeframe, seconds in (("M1", 60), ("M5", 300), ("M15", 900)):
        reference = NOW - stale_by
        latest_epoch = ((int(reference.timestamp()) - 1) // seconds) * seconds
        latest = datetime.fromtimestamp(latest_epoch, tz=timezone.utc)
        candles = []
        for index in range(120):
            started = latest - timedelta(seconds=seconds * (119 - index))
            bid = f"{base - half_spread:.5f}"
            ask = f"{base + half_spread:.5f}"
            mid = f"{base:.5f}"
            candles.append(
                {
                    "time": started.isoformat().replace("+00:00", "Z"),
                    "complete": True,
                    "volume": 1,
                    "mid": {"o": mid, "h": mid, "l": mid, "c": mid},
                    "bid": {"o": bid, "h": bid, "l": bid, "c": bid},
                    "ask": {"o": ask, "h": ask, "l": ask, "c": ask},
                }
            )
        batch = _technical_candles_from_payload(
            {
                "instrument": pair,
                "granularity": timeframe,
                "candles": candles,
            },
            pair=pair,
            granularity=timeframe,
            requested_count=120,
            pip_factor=factor,
            normal_spread_pips=normal_spread_pips,
            max_spread_multiple=RiskPolicy().max_spread_multiple,
            spread_anomaly_cap_pips=OANDA_SPREAD_CALIBRATION_V1.pairs[pair].max_pips,
            spread_calibration_sha256=OANDA_SPREAD_CALIBRATION_V1.calibration_sha256,
        )
        integrity = dict(batch.integrity)
        clean_tail_count = integrity["recent_clean_tail_count"]
        published = list(batch.candles[-clean_tail_count:])[-30:]
        views.append(
            {
                "granularity": timeframe,
                "recent_candles": [
                    {
                        "t": candle.timestamp_utc.isoformat(),
                        "complete": candle.complete,
                        "o": candle.open,
                        "h": candle.high,
                        "l": candle.low,
                        "c": candle.close,
                        "v": candle.volume,
                    }
                    for candle in published
                ],
                "indicators": {
                    "candles_count": clean_tail_count,
                    "close": base,
                    "atr_pips": 2.0,
                    "regime_quantile": "NORMAL",
                    "supertrend_dir": -1,
                    "linreg_slope_20": -1.0,
                },
                "family_scores": {
                    "trend_score": -0.8,
                    "mean_rev_score": -0.6,
                    "breakout_score": 0.7,
                },
                "candle_integrity": integrity,
            }
        )
    return {
        "generated_at_utc": NOW.isoformat(),
        "guardian_monitor_pairs": [pair],
        "guardian_monitor_scope": {
            pair: ["configured_rotation", "canonical_integrity_fixture"]
        },
        "charts": [
            {
                "pair": pair,
                "mid": base,
                "dominant_regime": "TREND_DOWN",
                "views": views,
            }
        ],
    }


def _issue_codes(review: dict) -> set[str]:
    return {str(issue.get("code")) for issue in review.get("issues", [])}


if __name__ == "__main__":
    unittest.main()
