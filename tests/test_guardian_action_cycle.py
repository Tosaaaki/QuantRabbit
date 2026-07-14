from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quant_rabbit.guardian_action_cycle import (
    GuardianActionCyclePaths,
    _run_live_order_gateway,
    run_guardian_action_cycle,
)


NOW = datetime(2026, 6, 30, 4, 0, tzinfo=timezone.utc)
LANE_ID = "guardian:EUR_USD:LONG:FAILED_ACCEPTANCE"
ROOT = Path(__file__).resolve().parents[1]


class GuardianActionCycleTest(unittest.TestCase):
    def test_guardian_gateway_handoff_configures_pre_post_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "data" / "order_intents.json"
            target = root / "data" / "daily_target_state.json"
            target_report = root / "docs" / "daily_target_report.md"
            ledger = root / "data" / "execution_ledger.db"
            ledger_report = root / "docs" / "execution_ledger_report.md"
            verified = root / "data" / "gpt_trader_decision.json"
            verified.parent.mkdir(parents=True, exist_ok=True)
            verified.write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "TRADE",
                            "capital_allocation": {
                                "decision": "ALLOCATE",
                                "size_multiple": 0.75,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            gateway_summary = SimpleNamespace(
                status="BLOCKED",
                lane_id=LANE_ID,
                sent=False,
                risk_issues=1,
                strategy_issues=0,
                output_path=root / "request.json",
                report_path=root / "report.md",
            )

            with patch(
                "quant_rabbit.broker.oanda.OandaExecutionClient",
                return_value=object(),
            ), patch("quant_rabbit.broker.execution.LiveOrderGateway") as gateway_cls:
                gateway_cls.return_value.run.return_value = gateway_summary
                result = _run_live_order_gateway(
                    LANE_ID,
                    True,
                    intents_path=intents,
                    target_state_path=target,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger,
                    execution_ledger_report_path=ledger_report,
                    verified_decision_path=verified,
                    live_enabled=True,
                )

            kwargs = gateway_cls.call_args.kwargs
            self.assertEqual(kwargs["target_state_path"], target)
            self.assertEqual(kwargs["target_report_path"], target_report)
            self.assertEqual(kwargs["execution_ledger_db_path"], ledger)
            self.assertEqual(kwargs["execution_ledger_report_path"], ledger_report)
            self.assertEqual(kwargs["verified_decision_path"], verified)
            self.assertEqual(
                gateway_cls.return_value.run.call_args.kwargs["size_multiple"],
                0.75,
            )
            self.assertEqual(result["status"], "BLOCKED")

    def test_default_flags_off_verifies_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[tuple[str, bool]] = []

            result = run_guardian_action_cycle(paths=paths, now=NOW, env={}, gateway_runner=_gateway(calls))

            self.assertEqual(result["status"], "VERIFIED_NO_SEND")
            self.assertIn("LIVE_FLAGS_DISABLED", result["no_send_reason"])
            self.assertEqual(calls, [])
            self.assertTrue(paths.result.exists())
            self.assertTrue(paths.report.exists())
            self.assertTrue(paths.log.exists())
            self.assertTrue(result["no_direct_oanda"])
            self.assertTrue(result["guardian_escalation"]["wake_gpt"])

    def test_missing_guardian_action_execute_blocks_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[tuple[str, bool]] = []

            result = run_guardian_action_cycle(
                paths=paths,
                now=NOW,
                env={"QR_LIVE_ENABLED": "1", "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1"},
                gateway_runner=_gateway(calls),
            )

            self.assertEqual(result["status"], "VERIFIED_NO_SEND")
            self.assertIn("LIVE_FLAGS_DISABLED", result["no_send_reason"])
            self.assertEqual(calls, [])

    def test_stale_broker_snapshot_rejects_without_live_refresh_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), snapshot_time=NOW - timedelta(minutes=30))

            result = run_guardian_action_cycle(paths=paths, now=NOW, env={})

            codes = {issue["code"] for issue in result["strict_receipt_issues"]}
            self.assertEqual(result["status"], "REJECTED")
            self.assertIn("BROKER_SNAPSHOT_STALE", codes)

    def test_active_lock_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            paths.live_lock.mkdir(parents=True)
            (paths.live_lock / "pid").write_text(str(os.getpid()))
            (paths.live_lock / "command").write_text("run-autotrade-live")

            result = run_guardian_action_cycle(paths=paths, now=NOW, env={})

            codes = {issue["code"] for issue in result["strict_receipt_issues"]}
            self.assertEqual(result["status"], "REJECTED")
            self.assertIn("ACTIVE_TRADER_OR_GATEWAY_LOCK", codes)

    def test_invalid_receipt_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), receipt_overrides={"action": "PANIC"})

            result = run_guardian_action_cycle(paths=paths, now=NOW, env={})

            codes = {issue["code"] for issue in result["strict_receipt_issues"]}
            self.assertEqual(result["status"], "REJECTED")
            self.assertIn("GUARDIAN_ACTION_BAD_ACTION", codes)

    def test_no_new_information_rejects_trade_add(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), receipt_overrides={"new_information": False})

            result = run_guardian_action_cycle(paths=paths, now=NOW, env={})

            codes = {issue["code"] for issue in result["strict_receipt_issues"]}
            self.assertIn("GUARDIAN_ACTION_REQUIRES_NEW_INFORMATION", codes)

    def test_wounded_or_invalidated_blocks_trade_add(self) -> None:
        for thesis_state in ("WOUNDED", "INVALIDATED"):
            with self.subTest(thesis_state=thesis_state), tempfile.TemporaryDirectory() as tmp:
                paths = _fixture(Path(tmp), event_overrides={"thesis_state": thesis_state}, receipt_overrides={"thesis_state": thesis_state})

                result = run_guardian_action_cycle(paths=paths, now=NOW, env={})

                codes = {issue["code"] for issue in result["strict_receipt_issues"]}
                self.assertIn("GUARDIAN_ACTION_THESIS_STATE_BLOCKS_ENTRY", codes)

    def test_live_order_gateway_required_for_trade_add(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[tuple[str, bool]] = []

            with patch("quant_rabbit.guardian_action_cycle._risk_result", return_value={"status": "ALLOWED", "allowed": True}):
                result = run_guardian_action_cycle(
                    paths=paths,
                    now=NOW,
                    env={
                        "QR_LIVE_ENABLED": "1",
                        "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1",
                        "QR_GUARDIAN_ACTION_EXECUTE": "1",
                    },
                    command_runner=_command_ok,
                    gateway_runner=_gateway(calls, status="SENT"),
                )

            self.assertEqual(result["status"], "EXECUTED")
            self.assertEqual(result["required_gateway"], "LiveOrderGateway")
            self.assertEqual(calls, [(LANE_ID, True)])
            self.assertTrue(result["no_direct_oanda"])

    def test_retained_technical_input_stale_blocks_gateway(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            paths.event_state.write_text(
                json.dumps(
                    {
                        "generated_at_utc": NOW.isoformat(),
                        "events": {
                            "EUR_USD|stale": {
                                "event_id": "stale-1",
                                "event_type": "TECHNICAL_INPUT_STALE",
                                "pair": "EUR_USD",
                                "action_hint": "NO_ACTION",
                            }
                        },
                    }
                )
            )
            calls: list[tuple[str, bool]] = []

            with patch(
                "quant_rabbit.guardian_action_cycle._risk_result",
                return_value={"status": "ALLOWED", "allowed": True},
            ):
                result = run_guardian_action_cycle(
                    paths=paths,
                    now=NOW,
                    env={
                        "QR_LIVE_ENABLED": "1",
                        "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1",
                        "QR_GUARDIAN_ACTION_EXECUTE": "1",
                    },
                    command_runner=_command_ok,
                    gateway_runner=_gateway(calls, status="SENT"),
                )

            codes = {issue["code"] for issue in result["strict_receipt_issues"]}
            self.assertEqual(result["status"], "REJECTED")
            self.assertIn("GUARDIAN_ACTION_TECHNICAL_INPUT_STALE", codes)
            self.assertEqual(calls, [])

    def test_receipt_event_intent_pair_or_side_mismatch_blocks_gateway(self) -> None:
        for receipt_overrides, expected_code in (
            ({"pair": "GBP_USD"}, "GUARDIAN_ACTION_ENTRY_PAIR_MISMATCH"),
            ({"side": "SHORT"}, "GUARDIAN_ACTION_ENTRY_SIDE_MISMATCH"),
        ):
            with self.subTest(receipt_overrides=receipt_overrides), tempfile.TemporaryDirectory() as tmp:
                paths = _fixture(Path(tmp), receipt_overrides=receipt_overrides)
                calls: list[tuple[str, bool]] = []

                with patch(
                    "quant_rabbit.guardian_action_cycle._risk_result",
                    return_value={"status": "ALLOWED", "allowed": True},
                ):
                    result = run_guardian_action_cycle(
                        paths=paths,
                        now=NOW,
                        env={
                            "QR_LIVE_ENABLED": "1",
                            "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1",
                            "QR_GUARDIAN_ACTION_EXECUTE": "1",
                        },
                        command_runner=_command_ok,
                        gateway_runner=_gateway(calls, status="SENT"),
                    )

                codes = {issue["code"] for issue in result["strict_receipt_issues"]}
                self.assertEqual(result["status"], "REJECTED")
                self.assertIn(expected_code, codes)
                self.assertEqual(calls, [])

    def test_trade_receipt_cannot_upgrade_hold_or_directionless_event(self) -> None:
        for event_overrides, expected_code in (
            (
                {"action_hint": "HOLD"},
                "GUARDIAN_ACTION_EVENT_DOES_NOT_AUTHORIZE_ENTRY",
            ),
            (
                {"direction": None},
                "GUARDIAN_ACTION_EVENT_DIRECTION_REQUIRED",
            ),
        ):
            with self.subTest(event_overrides=event_overrides), tempfile.TemporaryDirectory() as tmp:
                paths = _fixture(Path(tmp), event_overrides=event_overrides)
                calls: list[tuple[str, bool]] = []

                with patch(
                    "quant_rabbit.guardian_action_cycle._risk_result",
                    return_value={"status": "ALLOWED", "allowed": True},
                ):
                    result = run_guardian_action_cycle(
                        paths=paths,
                        now=NOW,
                        env={
                            "QR_LIVE_ENABLED": "1",
                            "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1",
                            "QR_GUARDIAN_ACTION_EXECUTE": "1",
                        },
                        command_runner=_command_ok,
                        gateway_runner=_gateway(calls, status="SENT"),
                    )

                codes = {issue["code"] for issue in result["strict_receipt_issues"]}
                self.assertEqual(result["status"], "REJECTED")
                self.assertIn(expected_code, codes)
                self.assertEqual(calls, [])

    def test_current_raw_technical_event_cannot_replace_persisted_state_proof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            paths.event_state.write_text("{}")
            events_payload = json.loads(paths.events.read_text())
            events_payload["events"].append(
                {
                    "event_id": "raw-technical",
                    "event_type": "TECHNICAL_STATE_CHANGE",
                    "pair": "EUR_USD",
                    "action_hint": "NO_ACTION",
                    "last_seen_at_utc": NOW.isoformat(),
                }
            )
            paths.events.write_text(json.dumps(events_payload))
            calls: list[tuple[str, bool]] = []

            with patch(
                "quant_rabbit.guardian_action_cycle._risk_result",
                return_value={"status": "ALLOWED", "allowed": True},
            ):
                result = run_guardian_action_cycle(
                    paths=paths,
                    now=NOW,
                    env={
                        "QR_LIVE_ENABLED": "1",
                        "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1",
                        "QR_GUARDIAN_ACTION_EXECUTE": "1",
                    },
                    command_runner=_command_ok,
                    gateway_runner=_gateway(calls, status="SENT"),
                )

            codes = {issue["code"] for issue in result["strict_receipt_issues"]}
            self.assertEqual(result["status"], "REJECTED")
            self.assertIn("GUARDIAN_TECHNICAL_STATE_UNAVAILABLE", codes)
            self.assertEqual(calls, [])

    def test_missing_or_stale_technical_state_blocks_gateway(self) -> None:
        for state_payload, events_generated_at in (
            ({}, None),
            (
                {
                    "generated_at_utc": (NOW - timedelta(minutes=10)).isoformat(),
                    "events": {
                        "EUR_USD|technical": {
                            "event_type": "TECHNICAL_STATE_CHANGE",
                            "pair": "EUR_USD",
                        }
                    },
                },
                (NOW - timedelta(minutes=10)).isoformat(),
            ),
        ):
            with self.subTest(state_payload=state_payload), tempfile.TemporaryDirectory() as tmp:
                paths = _fixture(Path(tmp))
                paths.event_state.write_text(json.dumps(state_payload))
                events_payload = json.loads(paths.events.read_text())
                if events_generated_at is None:
                    events_payload.pop("generated_at_utc", None)
                else:
                    events_payload["generated_at_utc"] = events_generated_at
                paths.events.write_text(json.dumps(events_payload))
                calls: list[tuple[str, bool]] = []

                with patch(
                    "quant_rabbit.guardian_action_cycle._risk_result",
                    return_value={"status": "ALLOWED", "allowed": True},
                ):
                    result = run_guardian_action_cycle(
                        paths=paths,
                        now=NOW,
                        env={
                            "QR_LIVE_ENABLED": "1",
                            "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1",
                            "QR_GUARDIAN_ACTION_EXECUTE": "1",
                        },
                        command_runner=_command_ok,
                        gateway_runner=_gateway(calls, status="SENT"),
                    )

                codes = {issue["code"] for issue in result["strict_receipt_issues"]}
                self.assertEqual(result["status"], "REJECTED")
                self.assertIn("GUARDIAN_TECHNICAL_STATE_UNAVAILABLE", codes)
                self.assertEqual(calls, [])

    def test_missing_lane_or_risk_exception_never_calls_gateway(self) -> None:
        cases = (
            (
                {"lane_id": "missing-lane"},
                None,
            ),
            (
                {},
                {
                    "status": "REJECTED",
                    "issues": [{"code": "RISK_ENGINE_EXCEPTION"}],
                },
            ),
            ({}, {"status": "ALLOWED", "allowed": False}),
            ({}, {"status": "READY", "allowed": True}),
        )
        for receipt_overrides, mocked_risk in cases:
            with self.subTest(receipt_overrides=receipt_overrides), tempfile.TemporaryDirectory() as tmp:
                paths = _fixture(Path(tmp), receipt_overrides=receipt_overrides)
                calls: list[tuple[str, bool]] = []
                risk_patch = (
                    patch(
                        "quant_rabbit.guardian_action_cycle._risk_result",
                        return_value=mocked_risk,
                    )
                    if mocked_risk is not None
                    else patch(
                        "quant_rabbit.guardian_action_cycle.RiskEngine.validate",
                        side_effect=RuntimeError("risk failure"),
                    )
                )
                with risk_patch:
                    result = run_guardian_action_cycle(
                        paths=paths,
                        now=NOW,
                        env={
                            "QR_LIVE_ENABLED": "1",
                            "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1",
                            "QR_GUARDIAN_ACTION_EXECUTE": "1",
                        },
                        command_runner=_command_ok,
                        gateway_runner=_gateway(calls, status="SENT"),
                    )

                codes = {issue["code"] for issue in result["strict_receipt_issues"]}
                self.assertEqual(result["status"], "REJECTED")
                self.assertIn("RISK_ENGINE_NOT_ALLOWED", codes)
                self.assertEqual(calls, [])

    def test_action_cycle_does_not_call_direct_oanda_write_methods(self) -> None:
        source = (ROOT / "src" / "quant_rabbit" / "guardian_action_cycle.py").read_text()

        self.assertNotIn(".post_order_json(", source)
        self.assertNotIn(".close_trade(", source)
        self.assertNotIn(".cancel_order(", source)

    def test_manual_exposure_cannot_be_loss_closed_by_guardian_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(
                Path(tmp),
                event_overrides={"action_hint": "REDUCE", "thesis_state": "INVALIDATED"},
                receipt_overrides={
                    "action": "REDUCE",
                    "new_information": False,
                    "thesis_state": "INVALIDATED",
                    "trade_ids": ["manual-1"],
                },
                manual_position={"trade_id": "manual-1", "unrealized_pl_jpy": -120.0},
            )

            result = run_guardian_action_cycle(paths=paths, now=NOW, env={})

            codes = {issue["code"] for issue in result["strict_receipt_issues"]}
            self.assertIn("MANUAL_LOSS_CLOSE_FORBIDDEN", codes)
            self.assertFalse(result["manual_exposure_safety"]["manual_pl_counts_as_system_pl"])

    def test_duplicate_receipt_cannot_execute_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[tuple[str, bool]] = []
            env = {
                "QR_LIVE_ENABLED": "1",
                "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1",
                "QR_GUARDIAN_ACTION_EXECUTE": "1",
            }

            with patch("quant_rabbit.guardian_action_cycle._risk_result", return_value={"status": "ALLOWED", "allowed": True}):
                first = run_guardian_action_cycle(
                    paths=paths,
                    now=NOW,
                    env=env,
                    command_runner=_command_ok,
                    gateway_runner=_gateway(calls, status="SENT"),
                )
                second = run_guardian_action_cycle(
                    paths=paths,
                    now=NOW + timedelta(seconds=1),
                    env=env,
                    command_runner=_command_ok,
                    gateway_runner=_gateway(calls, status="SENT"),
                )

            self.assertEqual(first["status"], "EXECUTED")
            self.assertEqual(second["status"], "REJECTED")
            self.assertEqual(calls, [(LANE_ID, True)])
            codes = {issue["code"] for issue in second["strict_receipt_issues"]}
            self.assertIn("GUARDIAN_ACTION_DUPLICATE_RECEIPT", codes)

    def test_hold_and_no_action_never_execute_and_mark_consumed(self) -> None:
        for action in ("HOLD", "NO_ACTION"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmp:
                paths = _fixture(Path(tmp), receipt_overrides={"action": action, "new_information": False, "lane_id": ""})
                calls: list[tuple[str, bool]] = []

                result = run_guardian_action_cycle(
                    paths=paths,
                    now=NOW,
                    env={
                        "QR_LIVE_ENABLED": "1",
                        "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1",
                        "QR_GUARDIAN_ACTION_EXECUTE": "1",
                    },
                    command_runner=_command_ok,
                    gateway_runner=_gateway(calls, status="SENT"),
                )

                self.assertEqual(result["status"], "VERIFIED_NO_ACTION")
                self.assertIn("NO_EXECUTABLE_ACTION", result["no_send_reason"])
                self.assertEqual(calls, [])
                self.assertEqual(result["receipt"]["action"], action)
                updated = json.loads(paths.action_receipt.read_text())
                self.assertEqual(updated["receipt_lifecycle"], "CONSUMED")
                self.assertTrue(updated["consumed_by_trader"])
                self.assertEqual(result["receipt_lifecycle_update"]["receipt_lifecycle"], "CONSUMED")


def _fixture(
    root: Path,
    *,
    snapshot_time: datetime = NOW,
    event_overrides: dict | None = None,
    receipt_overrides: dict | None = None,
    manual_position: dict | None = None,
) -> GuardianActionCyclePaths:
    paths = GuardianActionCyclePaths.from_root(root, live_root=root / "live")
    for directory in (root / "data", root / "docs", root / "logs", paths.live_root):
        directory.mkdir(parents=True, exist_ok=True)
    event = {
        "event_id": "event-1",
        "event_type": "FAILED_ACCEPTANCE",
        "pair": "EUR_USD",
        "direction": "LONG",
        "thesis": "major figure rejection",
        "price_zone": "1.1700 failed acceptance",
        "severity": "P1",
        "recommended_review_type": "ENTRY_REVIEW",
        "dedupe_key": "EUR_USD|major_figure_rejection|FAILED_ACCEPTANCE|TRADE",
        "action_hint": "TRADE",
        "thesis_state": "ALIVE",
        "detected_at_utc": NOW.isoformat(),
        "details": {},
    }
    event.update(event_overrides or {})
    receipt = {
        "action": "TRADE",
        "event_id": event["event_id"],
        "dedupe_key": event["dedupe_key"],
        "new_information": True,
        "pair": "EUR_USD",
        "side": "LONG",
        "lane_id": LANE_ID,
        "thesis_state": "ALIVE",
        "reason": "fresh failed acceptance at the major figure",
        "invalidation": "accepted trade back below failed-acceptance support",
        "harvest_trigger": "upper range rail",
        "margin_state": "margin available",
        "ownership": "system",
        "gateway_required": True,
        "no_direct_oanda": True,
    }
    receipt.update(receipt_overrides or {})
    positions = []
    if manual_position:
        positions.append(
            {
                "trade_id": manual_position.get("trade_id", "manual-1"),
                "pair": "EUR_USD",
                "side": "LONG",
                "units": 1000,
                "entry_price": 1.17,
                "unrealized_pl_jpy": manual_position.get("unrealized_pl_jpy", -100.0),
                "take_profit": None,
                "stop_loss": None,
                "owner": "unknown",
                "raw": {},
            }
        )
    paths.events.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "events": [event]}))
    paths.event_state.write_text(
        json.dumps(
            {
                "generated_at_utc": NOW.isoformat(),
                "events": {
                    "EUR_USD|technical": {
                        "event_id": "technical-1",
                        "event_type": "TECHNICAL_STATE_CHANGE",
                        "pair": "EUR_USD",
                        "action_hint": "NO_ACTION",
                        "last_seen_at_utc": NOW.isoformat(),
                    }
                },
            }
        )
    )
    paths.escalation.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "wake_gpt": True, "events_to_review": [event]}))
    paths.action_receipt.write_text(
        json.dumps(
            {
                "status": "ACCEPTED",
                "receipt_status": "ACCEPTED",
                "receipt_lifecycle": "ACTIVE",
                "generated_at_utc": NOW.isoformat(),
                "selected_event_id": event["event_id"],
                "selected_event_dedupe_key": event["dedupe_key"],
                "expires_at_utc": (NOW + timedelta(minutes=75)).isoformat(),
                "consumed_by_trader": False,
                "superseded_by_event_id": None,
                "source": "guardian_wake_dispatcher",
                "model": "gpt-5.5",
                "no_direct_oanda": True,
                "selected_event": event,
                "receipt": receipt,
            }
        )
    )
    paths.broker_snapshot.write_text(
        json.dumps(
            {
                "fetched_at_utc": snapshot_time.isoformat(),
                "account": {
                    "nav_jpy": 200000,
                    "balance_jpy": 200000,
                    "margin_available_jpy": 180000,
                    "margin_used_jpy": 0,
                    "last_transaction_id": "1",
                    "fetched_at_utc": snapshot_time.isoformat(),
                    "hedging_enabled": True,
                },
                "positions": positions,
                "orders": [],
                "quotes": {"EUR_USD": {"bid": 1.17, "ask": 1.1701, "timestamp_utc": snapshot_time.isoformat()}},
            }
        )
    )
    paths.daily_target_state.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "mode": "PURSUE_TARGET"}))
    paths.order_intents.write_text(
        json.dumps(
            {
                "generated_at_utc": NOW.isoformat(),
                "results": [
                    {
                        "lane_id": LANE_ID,
                        "status": "LIVE_READY",
                        "risk_allowed": True,
                        "intent": {
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "order_type": "MARKET",
                            "units": 1000,
                            "entry": 1.1701,
                            "tp": 1.1715,
                            "sl": 1.1685,
                            "thesis": "major figure rejection",
                            "owner": "trader",
                            "market_context": {
                                "regime": "RANGE",
                                "narrative": "fixture",
                                "chart_story": "fixture",
                                "method": "RANGE_ROTATION",
                                "invalidation": "below support",
                            },
                            "metadata": {"guardian_event_id": event["event_id"], "guardian_event_dedupe_key": event["dedupe_key"]},
                        },
                    }
                ],
            }
        )
    )
    paths.action_review.write_text("# Guardian Action Review\n")
    return paths


def _gateway(calls: list[tuple[str, bool]], *, status: str = "STAGED"):
    def run(lane_id: str, send: bool) -> dict:
        calls.append((lane_id, send))
        return {"status": status, "sent": status == "SENT", "lane_id": lane_id}

    return run


def _command_ok(*args, **kwargs):
    return SimpleNamespace(returncode=0, stdout="", stderr="")


if __name__ == "__main__":
    unittest.main()
