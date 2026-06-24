from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.cli import main
from quant_rabbit.execution_timing_contracts import MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND
from quant_rabbit.trader_repair_orchestrator import (
    STATUS_APPROVAL_REQUIRED,
    STATUS_BLOCKED,
    STATUS_READY,
    TraderRepairOrchestrator,
)
from quant_rabbit.trader_support_bot import (
    DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST,
    DIRECTIONAL_INVERSION_REPLAY_WAIT_STATUS,
    FRONTIER_MARGIN_CAPACITY_WAIT_STATUS,
    FRONTIER_QUOTE_FRESHNESS_WAIT_STATUS,
    OANDA_AUDIT_ONLY_LOCAL_TP_EDGE_REQUEST,
    OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_UNPROVED_STATUS,
    REPAIR_AUTOMATION_ALLOWED_ACTIONS,
    REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS,
    REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS,
    TP_PROGRESS_GUARDIAN_WAIT_STATUS,
)


class TraderRepairOrchestratorTest(unittest.TestCase):
    def test_builds_codex_repair_queue_without_live_side_effects(self) -> None:
        now = datetime(2026, 6, 23, 10, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                        priority="P0",
                        suggested_files=[
                            "src/quant_rabbit/profit_capture_bot.py",
                            "tests/test_profit_capture_bot.py",
                        ],
                        verification_commands=[
                            MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND
                        ],
                        evidence_summary={
                            "top_entry_quality_residual_method_rollups": [
                                {
                                    "method": "RANGE_ROTATION",
                                    "pair_count": 7,
                                    "repair_replay_pl_jpy": -10269.1823,
                                }
                            ]
                        },
                    ),
                    _request(
                        "RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT",
                        priority="P0",
                        requires_explicit_operator_approval=True,
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="利確 bot を直して",
                now_utc=now,
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            self.assertTrue(payload["read_only"])
            self.assertEqual(payload["live_side_effects"], [])
            self.assertEqual(payload["selected_request"]["code"], "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY")
            self.assertEqual(
                payload["selected_request"]["automation_status"],
                "READY_FOR_CODEX_IMPLEMENTATION",
            )
            self.assertEqual(
                payload["selected_request"]["evidence_summary"][
                    "top_entry_quality_residual_method_rollups"
                ][0]["method"],
                "RANGE_ROTATION",
            )
            self.assertIn(
                "PYTHONPATH=src python3 -m unittest tests.test_profit_capture_bot -v",
                payload["selected_request"]["targeted_test_commands"],
            )
            contract = payload["execution_contract"]
            self.assertEqual(contract["codex_may_execute"], REPAIR_AUTOMATION_ALLOWED_ACTIONS)
            self.assertEqual(
                contract["requires_explicit_operator_approval_for"],
                REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS,
            )
            self.assertEqual(contract["forbidden_direct_actions"], REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS)
            self.assertFalse(contract["quant_rabbit_code_may_call_model_api"])
            self.assertIn("Order send", contract["orders_closes_launchd_policy"])
            self.assertEqual(
                payload["queue_summary"]["selected_request_code"],
                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
            )
            self.assertEqual(
                payload["selected_request_code"],
                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
            )
            self.assertEqual(payload["approval_boundary"]["live_side_effects_allowed"], [])
            self.assertTrue(payload["approval_boundary"]["read_only_until_gateway_or_operator_approval"])
            self.assertEqual(
                payload["approval_boundary"]["existing_gateway_paths"]["order_send"],
                "LiveOrderGateway",
            )
            work_order = payload["codex_work_order"]
            self.assertEqual(work_order["status"], "READY_FOR_CODEX_IMPLEMENTATION")
            self.assertEqual(work_order["selected_request_code"], "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY")
            self.assertEqual(work_order["dependency_rank"], 0)
            self.assertIn("TP-progress", work_order["selection_reason"])
            self.assertIn("regression_tests_for_the_named_failure", work_order["deliverables"])
            self.assertIn("git_commit_with_codex_attribution", work_order["deliverables"])
            self.assertTrue(work_order["commit_and_live_sync_required"])
            self.assertFalse(work_order["quant_rabbit_code_may_call_model_api"])
            self.assertEqual(
                work_order["evidence_summary"][
                    "top_entry_quality_residual_method_rollups"
                ][0]["pair_count"],
                7,
            )
            self.assertIn("Do not send orders", work_order["automation_prompt"])
            loop_prompt = payload["loop_engineering_prompt"]
            self.assertEqual(loop_prompt["version"], "loop_engineering_prompt_v1")
            self.assertEqual(
                loop_prompt["current_state"]["selected_request_code"],
                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
            )
            self.assertIn("implementation work", loop_prompt["current_hypothesis"])
            self.assertIn("Commit", " ".join(loop_prompt["next_loop"]))
            self.assertIn("market returns", loop_prompt["prompt_text"])
            self.assertFalse(loop_prompt["approval_boundary"]["quant_rabbit_code_may_call_model_api"])
            report_text = report.read_text()
            self.assertIn("REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY", report_text)
            self.assertIn("Codex Work Order", report_text)
            self.assertIn("Loop Engineering Prompt", report_text)
            self.assertIn("Evidence summary keys", report_text)
            self.assertIn("Dependency", report_text)

    def test_loop_prompt_marks_lower_priority_selected_work_as_auxiliary_to_waiting_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            requests = [
                _request(
                    "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                    priority="P0",
                    status=TP_PROGRESS_GUARDIAN_WAIT_STATUS,
                    evidence_summary={"loss_closes_repair_replay_triggered": 13},
                ),
                _request(
                    "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY",
                    priority="P0",
                    status="RESIDUAL_GROUPS_ALREADY_BLOCKED_WAITING_FOR_REPLAY",
                ),
                _request(
                    OANDA_AUDIT_ONLY_LOCAL_TP_EDGE_REQUEST,
                    priority="P1",
                    status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                ),
            ]
            support.write_text(
                json.dumps(
                    {
                        "status": "SUPPORT_BLOCKED",
                        "blockers": [
                            {"code": "LOSS_CLOSE_PROFIT_CAPTURE_MISSED", "severity": "P0"},
                            {"code": "NO_LIVE_READY_LANES", "severity": "P1"},
                        ],
                        "target": {"status": "PURSUE_TARGET"},
                        "guardian": {
                            "active": True,
                            "active_source": "launchd+heartbeat",
                            "heartbeat_status": "NO_POSITION",
                        },
                        "entry_readiness": {
                            "live_ready_lanes": 0,
                            "guardian_blocked_lanes": 98,
                        },
                        "profitability_acceptance": {
                            "status": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                            "target_firepower": {
                                "operational_minimum_5pct_reachable": False,
                                "minimum_5pct_estimated_reachable": True,
                            },
                        },
                        "repair_requests": requests,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            self.assertEqual(
                payload["selected_request"]["code"],
                OANDA_AUDIT_ONLY_LOCAL_TP_EDGE_REQUEST,
            )
            loop_prompt = payload["loop_engineering_prompt"]
            state = loop_prompt["current_state"]
            self.assertEqual(
                state["waiting_p0_request_codes"],
                [
                    "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                    "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY",
                ],
            )
            self.assertEqual(
                state["primary_waiting_p0_request_code"],
                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
            )
            self.assertTrue(state["selected_request_is_auxiliary_to_waiting_p0"])
            self.assertEqual(
                state["artifact_contradiction_codes"],
                ["GUARDIAN_ACTIVE_BUT_INTENTS_CARRY_GUARDIAN_BLOCKERS"],
            )
            self.assertIn("artifact-stale", " ".join(loop_prompt["anti_loop_rules"]))
            self.assertIn("Resolve artifact contradictions", loop_prompt["next_loop"][0])
            self.assertIn(
                "generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts",
                " ".join(loop_prompt["verification_commands"]),
            )
            self.assertIn("LOSS_CLOSE_PROFIT_CAPTURE_MISSED", state["support_blocker_codes"])
            self.assertIn(
                "causal P0 blocker remains REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                loop_prompt["current_hypothesis"],
            )
            self.assertIn("auxiliary work", loop_prompt["current_hypothesis"])
            self.assertNotIn("..", loop_prompt["current_hypothesis"])
            self.assertIn("waiting P0 blockers", " ".join(loop_prompt["next_loop"]))
            self.assertIn(
                "waiting_p0=REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                loop_prompt["prompt_text"],
            )
            self.assertIn(
                "Artifact contradictions: GUARDIAN_ACTIVE_BUT_INTENTS_CARRY_GUARDIAN_BLOCKERS",
                loop_prompt["prompt_text"],
            )

    def test_loop_prompt_marks_order_intents_older_than_broker_snapshot_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            requests = [
                _request(
                    "REPAIR_FRONTIER_LANE_BLOCKER",
                    priority="P1",
                    status="ORDER_INTENTS_ARTIFACT_REFRESH_REQUIRED",
                    source_findings=[
                        "ORDER_INTENTS_STALE_AGAINST_BROKER_SNAPSHOT",
                        "MARGIN_TOO_THIN_FOR_MIN_LOT",
                    ],
                    verification_commands=[
                        "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts",
                        "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                    ],
                )
            ]
            support.write_text(
                json.dumps(
                    {
                        "status": "SUPPORT_BLOCKED",
                        "blockers": [
                            {"code": "NO_LIVE_READY_LANES", "severity": "P1"},
                            {
                                "code": "ORDER_INTENTS_STALE_AGAINST_BROKER_SNAPSHOT",
                                "severity": "P1",
                            },
                        ],
                        "target": {"status": "PURSUE_TARGET"},
                        "guardian": {"active": True, "active_source": "launchd+heartbeat"},
                        "entry_readiness": {
                            "live_ready_lanes": 0,
                            "guardian_blocked_lanes": 0,
                            "artifact_freshness": {
                                "status": "ORDER_INTENTS_ARTIFACT_REFRESH_REQUIRED",
                                "order_intents_generated_at_utc": "2026-06-24T15:21:06+00:00",
                                "broker_snapshot_fetched_at_utc": "2026-06-24T15:24:56+00:00",
                                "order_intents_staleness_seconds": 230.0,
                                "order_intents_stale_against_broker_snapshot": True,
                            },
                        },
                        "profitability_acceptance": {
                            "status": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                            "target_firepower": {
                                "operational_minimum_5pct_reachable": False,
                                "minimum_5pct_estimated_reachable": True,
                            },
                        },
                        "repair_requests": requests,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            self.assertEqual(summary.status, STATUS_BLOCKED)
            payload = json.loads(output.read_text())
            loop_prompt = payload["loop_engineering_prompt"]
            state = loop_prompt["current_state"]
            self.assertEqual(
                state["artifact_contradiction_codes"],
                ["ORDER_INTENTS_STALE_AGAINST_BROKER_SNAPSHOT"],
            )
            self.assertIn("ORDER_INTENTS_STALE_AGAINST_BROKER_SNAPSHOT", state["support_blocker_codes"])
            self.assertIn("Resolve artifact contradictions", loop_prompt["next_loop"][0])
            self.assertIn(
                "generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts",
                " ".join(loop_prompt["verification_commands"]),
            )
            self.assertIn(
                "Artifact contradictions: ORDER_INTENTS_STALE_AGAINST_BROKER_SNAPSHOT",
                loop_prompt["prompt_text"],
            )
            self.assertEqual(payload["codex_work_order"]["status"], "NO_ACTIONABLE_CODEX_WORK")

    def test_loop_prompt_does_not_mark_guardian_blockers_stale_when_guardian_is_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            support.write_text(
                json.dumps(
                    {
                        "status": "SUPPORT_BLOCKED",
                        "guardian": {"active": False, "heartbeat_status": "STALE"},
                        "entry_readiness": {
                            "live_ready_lanes": 0,
                            "guardian_blocked_lanes": 4,
                        },
                        "repair_requests": [
                            _request(
                                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                                priority="P0",
                            )
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            loop_prompt = payload["loop_engineering_prompt"]
            state = loop_prompt["current_state"]
            self.assertEqual(state["guardian_blocked_lanes"], 4)
            self.assertEqual(state["artifact_contradictions"], [])
            self.assertEqual(state["artifact_contradiction_codes"], [])
            self.assertNotIn("Resolve artifact contradictions", loop_prompt["next_loop"][0])
            self.assertIn("Artifact contradictions: (none)", loop_prompt["prompt_text"])

    def test_only_approval_required_requests_return_diagnostic_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT",
                        priority="P0",
                        requires_explicit_operator_approval=True,
                    )
                ],
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "trader-repair-orchestrator",
                        "--trader-support-bot",
                        str(support),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                    ]
                )

            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], STATUS_APPROVAL_REQUIRED)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["actionable_requests"], [])
            self.assertEqual(payload["approval_required_requests"][0]["automation_status"], "WAITING_FOR_OPERATOR_APPROVAL")
            self.assertIn("launchd_load", payload["approval_required_requests"][0]["automation_contract"]["requires_explicit_operator_approval_for"])
            self.assertEqual(payload["codex_work_order"]["status"], "NO_ACTIONABLE_CODEX_WORK")
            self.assertEqual(payload["codex_work_order"]["approval_boundary"]["live_side_effects_allowed"], [])
            loop_prompt = payload["loop_engineering_prompt"]
            self.assertEqual(
                loop_prompt["current_state"]["approval_required_request_codes"],
                ["RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT"],
            )
            self.assertIn("approval-bound", loop_prompt["current_hypothesis"])
            self.assertIn("explicit operator approval", " ".join(loop_prompt["next_loop"]))

    def test_loop_prompt_keeps_unknown_owner_review_out_of_approval_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                        priority="P0",
                        status=TP_PROGRESS_GUARDIAN_WAIT_STATUS,
                        evidence_summary={"loss_closes_repair_replay_triggered": 13},
                    ),
                    _request(
                        "REVIEW_UNKNOWN_OWNER_EXPOSURE",
                        priority="P1",
                        status="OPERATOR_REVIEW_RECOMMENDED",
                        requires_explicit_operator_approval=False,
                        source_findings=[
                            "BROKER_TRUTH_UNKNOWN_OWNER_EXPOSURE",
                            "MARGIN_TOO_THIN_FOR_MIN_LOT",
                        ],
                        evidence_summary={
                            "unknown_owner_positions": 1,
                            "examples": [
                                {
                                    "trade_id": "472802",
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "units": 20000,
                                    "owner": "unknown",
                                    "take_profit": 1.13834,
                                    "stop_loss": None,
                                    "unrealized_pl_jpy": -11765.663,
                                }
                            ],
                            "margin_available_jpy": 16001.268,
                            "nav_jpy": 162590.792,
                        },
                    )
                ],
            )
            _add_execution_frontier_fixture(support)

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            self.assertEqual(summary.status, STATUS_BLOCKED)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["actionable_requests"], [])
            self.assertEqual(payload["approval_required_requests"], [])
            self.assertEqual(payload["codex_work_order"]["status"], "NO_ACTIONABLE_CODEX_WORK")
            self.assertEqual(payload["approval_boundary"]["live_side_effects_allowed"], [])
            loop_prompt = payload["loop_engineering_prompt"]
            self.assertEqual(loop_prompt["current_state"]["approval_required_details"], [])
            self.assertEqual(loop_prompt["current_state"]["approval_required_request_codes"], [])
            self.assertEqual(
                loop_prompt["current_state"]["primary_waiting_p0_request_code"],
                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
            )
            self.assertIn("REVIEW_UNKNOWN_OWNER_EXPOSURE", payload["queue_summary"]["waiting_request_codes"])
            review = next(item for item in payload["queue"] if item["code"] == "REVIEW_UNKNOWN_OWNER_EXPOSURE")
            self.assertEqual(review["automation_status"], "WAITING_FOR_EVIDENCE")
            self.assertFalse(review["requires_explicit_operator_approval"])
            self.assertIn("evidence-window work", loop_prompt["current_hypothesis"])
            self.assertNotIn("approval target", loop_prompt["prompt_text"])
            frontier = loop_prompt["current_state"]["execution_frontier"]
            self.assertEqual(
                frontier["repair_frontier_top_lanes"][0]["lane_id"],
                "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
            )
            self.assertEqual(
                frontier["repair_frontier_top_blockers"][0]["code"],
                "MARGIN_TOO_THIN_FOR_MIN_LOT",
            )
            self.assertEqual(
                frontier["unknown_owner_context"]["unknown_owner_positions"],
                1,
            )
            self.assertIn("Execution frontier:", loop_prompt["prompt_text"])
            self.assertIn("TP_PROVEN_HARVEST", loop_prompt["prompt_text"])
            self.assertIn("MARGIN_TOO_THIN_FOR_MIN_LOT", loop_prompt["prompt_text"])
            self.assertIn("unknown_owner_positions=1", loop_prompt["prompt_text"])

    def test_directional_inversion_without_repeated_replay_evidence_is_not_codex_ready(self) -> None:
        now = datetime(2026, 6, 24, 11, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST,
                        priority="P0",
                        status=DIRECTIONAL_INVERSION_REPLAY_WAIT_STATUS,
                        source_findings=[
                            "BROKER_TRUTH_OPPOSITE_SIDE_WOULD_CLEAR_MINIMUM_5PCT",
                            "DIRECTIONAL_INVERSION_REPLAY_EVIDENCE_MISSING",
                        ],
                        evidence_summary={
                            "counterfactuals": [
                                {
                                    "pair": "EUR_USD",
                                    "actual_side": "LONG",
                                    "opposite_side": "SHORT",
                                    "would_clear_minimum_5pct": True,
                                    "has_repeated_spread_included_inversion_evidence": False,
                                }
                            ]
                        },
                    )
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                now_utc=now,
            ).run()

            payload = json.loads(output.read_text())
            self.assertEqual(summary.status, STATUS_BLOCKED)
            self.assertEqual(payload["actionable_requests"], [])
            self.assertEqual(payload["codex_work_order"]["status"], "NO_ACTIONABLE_CODEX_WORK")
            waiting = payload["queue"][0]
            self.assertEqual(waiting["code"], DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST)
            self.assertEqual(waiting["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertEqual(
                payload["loop_engineering_prompt"]["current_state"]["waiting_request_codes"],
                [DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST],
            )

    def test_recovers_repair_queue_from_embedded_support_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            support.write_text(
                json.dumps(
                    {
                        "status": "SUPPORT_BLOCKED",
                        "guardian": {
                            "required": True,
                            "active": False,
                            "active_source": "launchd",
                            "launchd_loaded": False,
                            "heartbeat_fresh": False,
                        },
                        "profit_capture": {},
                        "entry_readiness": {},
                        "profitability_acceptance": {
                            "repair_plan": {
                                "items": [
                                    {
                                        "code": "TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED",
                                        "priority": "P0",
                                        "message": "guardian inactive",
                                        "clearance_condition": "prove guardian capture",
                                        "verification_command": "scripts/install-position-guardian.sh --status",
                                        "evidence_summary": {
                                            "loss_closes_repair_replay_triggered": 13
                                        },
                                    }
                                ],
                                "evidence_collection_items": [],
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="利確 bot を直して",
            ).run()

            self.assertEqual(summary.status, STATUS_APPROVAL_REQUIRED)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["metrics"]["repair_request_source"], "embedded_support_payload")
            self.assertTrue(payload["metrics"]["recovered_from_embedded_support"])
            self.assertEqual(payload["selected_request"], {})
            self.assertEqual(payload["actionable_requests"], [])
            self.assertIn(
                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                [item["code"] for item in payload["approval_required_requests"]],
            )
            self.assertIn(
                "RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT",
                [item["code"] for item in payload["approval_required_requests"]],
            )
            self.assertIn("embedded_support_payload", report.read_text())

    def test_direct_tp_capture_repair_beats_residual_when_request_match_ties(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY",
                        priority="P0",
                        suggested_files=[
                            "src/quant_rabbit/strategy/intent_generator.py",
                            "tests/test_intent_generator.py",
                        ],
                    ),
                    _request(
                        "REPAIR_CLOSE_GATE_EVIDENCE_PERSISTENCE",
                        priority="P0",
                        suggested_files=[
                            "src/quant_rabbit/gpt_trader.py",
                            "tests/test_gpt_trader.py",
                        ],
                    ),
                    _request(
                        "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                        priority="P0",
                        suggested_files=[
                            "src/quant_rabbit/strategy/position_manager.py",
                            "tests/test_trader_support_bot.py",
                        ],
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="利益をプラスで決済するBot/修正",
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            self.assertEqual(
                payload["selected_request"]["code"],
                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
            )
            self.assertEqual(payload["selected_request"]["dependency_rank"], 0)
            self.assertEqual(
                [item["code"] for item in payload["actionable_requests"][:3]],
                [
                    "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                    "REPAIR_CLOSE_GATE_EVIDENCE_PERSISTENCE",
                    "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY",
                ],
            )

    def test_guardian_blocked_tp_capture_repair_does_not_loop_as_codex_code_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                        priority="P0",
                        evidence_summary={
                            "guardian_profit_capture_inactive": True,
                            "loss_closes_repair_replay_triggered": 13,
                            "repair_replay_contract": "TP_PROGRESS_PRODUCTION_GATE_REPLAY_V1",
                        },
                    ),
                    _request(
                        "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY",
                        priority="P0",
                        suggested_files=[
                            "src/quant_rabbit/strategy/intent_generator.py",
                            "tests/test_intent_generator.py",
                        ],
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            tp_repair = next(
                item
                for item in payload["queue"]
                if item["code"] == "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY"
            )
            self.assertEqual(tp_repair["automation_status"], "WAITING_FOR_OPERATOR_APPROVAL")
            self.assertEqual(
                tp_repair["approval_dependency"]["code"],
                "RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT",
            )
            self.assertNotIn(
                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                [item["code"] for item in payload["actionable_requests"]],
            )
            self.assertIn(
                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                [item["code"] for item in payload["approval_required_requests"]],
            )
            self.assertEqual(
                payload["selected_request"]["code"],
                "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY",
            )
            self.assertEqual(payload["selected_request_code"], "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY")

    def test_stale_tp_guardian_inactive_evidence_waits_when_current_guardian_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                        priority="P0",
                        status=TP_PROGRESS_GUARDIAN_WAIT_STATUS,
                        source_findings=[
                            "TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED",
                            "TP_PROGRESS_REPLAY_REPAIR_UNPROVED",
                        ],
                        evidence_summary={
                            "guardian_profit_capture_inactive": True,
                            "current_guardian_active": True,
                            "current_guardian_heartbeat_fresh": True,
                            "guardian_inactive_evidence_status": "STALE_CURRENT_GUARDIAN_ACTIVE",
                            "loss_closes_repair_replay_triggered": 13,
                            "repair_replay_contract": "TP_PROGRESS_PRODUCTION_GATE_REPLAY_V1",
                        },
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            self.assertEqual(summary.status, STATUS_BLOCKED)
            payload = json.loads(output.read_text())
            tp_repair = payload["queue"][0]
            self.assertEqual(tp_repair["code"], "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY")
            self.assertEqual(tp_repair["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertIsNone(tp_repair["approval_dependency"])
            self.assertEqual(payload["approval_required_requests"], [])
            self.assertEqual(payload["selected_request"], {})
            self.assertIn(
                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                payload["queue_summary"]["waiting_request_codes"],
            )

    def test_runtime_lock_busy_tp_wait_does_not_emit_restore_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                        priority="P0",
                        status=TP_PROGRESS_GUARDIAN_WAIT_STATUS,
                        source_findings=[
                            "TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED",
                            "TP_PROGRESS_REPLAY_REPAIR_UNPROVED",
                        ],
                        evidence_summary={
                            "guardian_profit_capture_inactive": True,
                            "current_guardian_active": False,
                            "current_guardian_active_source": "live_runtime_lock_busy",
                            "current_guardian_heartbeat_fresh": False,
                            "current_guardian_live_runtime_lock_active": True,
                            "current_guardian_live_runtime_lock_command": "cycle-refresh",
                            "guardian_inactive_evidence_status": "CURRENT_GUARDIAN_LOCK_BUSY",
                            "loss_closes_repair_replay_triggered": 13,
                            "repair_replay_contract": "TP_PROGRESS_PRODUCTION_GATE_REPLAY_V1",
                        },
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            self.assertEqual(summary.status, STATUS_BLOCKED)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["approval_required_requests"], [])
            self.assertEqual(payload["actionable_requests"], [])
            self.assertNotIn(
                "RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT",
                [item["code"] for item in payload["queue"]],
            )
            tp_repair = next(
                item
                for item in payload["queue"]
                if item["code"] == "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY"
            )
            self.assertEqual(tp_repair["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertIsNone(tp_repair["approval_dependency"])
            self.assertIn(
                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                payload["queue_summary"]["waiting_request_codes"],
            )

    def test_specific_trader_request_can_select_residual_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request("REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY", priority="P0"),
                    _request("REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY", priority="P0"),
                ],
            )

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="residual method month",
            ).run()

            payload = json.loads(output.read_text())
            self.assertEqual(
                payload["selected_request"]["code"],
                "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY",
            )

    def test_residual_repair_waits_for_replay_when_current_intents_already_block_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY",
                        priority="P0",
                        status="RESIDUAL_GROUPS_ALREADY_BLOCKED_WAITING_FOR_REPLAY",
                        evidence_summary={
                            "current_residual_block_status": {
                                "current_residual_blocked_intents_count": 5,
                                "status": "CURRENT_INTENTS_BLOCK_RESIDUAL_GROUPS_WAIT_FOR_744H_REPLAY",
                            }
                        },
                    ),
                    _request("REPAIR_FRONTIER_LANE_BLOCKER", priority="P1"),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="residual method month",
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            residual = next(
                item
                for item in payload["queue"]
                if item["code"] == "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY"
            )
            self.assertEqual(residual["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertEqual(payload["selected_request"]["code"], "REPAIR_FRONTIER_LANE_BLOCKER")
            self.assertIn(
                "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY",
                payload["queue_summary"]["waiting_request_codes"],
            )
            self.assertNotIn(
                "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY",
                [item["code"] for item in payload["actionable_requests"]],
            )

    def test_directional_inversion_counterfactual_is_selected_for_forecast_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request("REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY", priority="P0"),
                    _request(
                        DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST,
                        priority="P0",
                        status="READY_FOR_CODE_OR_EVIDENCE_REPAIR",
                        evidence_summary={
                            "counterfactuals": [
                                {
                                    "trade_id": "472802",
                                    "pair": "EUR_USD",
                                    "actual_side": "LONG",
                                    "opposite_side": "SHORT",
                                    "opposite_gross_counterfactual_pl_jpy": 12090.085,
                                    "would_clear_minimum_5pct": True,
                                }
                            ]
                        },
                        suggested_files=[
                            "src/quant_rabbit/trader_support_bot.py",
                            "tests/test_trader_support_bot.py",
                        ],
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="予測 精度 逆 5%",
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["selected_request"]["code"], DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST)
            self.assertEqual(payload["codex_work_order"]["status"], "READY_FOR_CODEX_IMPLEMENTATION")
            self.assertEqual(payload["codex_work_order"]["selected_request_code"], DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST)
            self.assertIn("opposite-side counterfactual", payload["codex_work_order"]["selection_reason"])
            self.assertIn(
                "PYTHONPATH=src python3 -m unittest tests.test_trader_support_bot -v",
                payload["codex_work_order"]["targeted_test_commands"],
            )

    def test_forecast_frontier_waits_and_selects_bidask_evidence_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_FRONTIER_LANE_BLOCKER",
                        priority="P1",
                        status="FORECAST_FRONTIER_WAITING_FOR_LIVE_PRECISION_EVIDENCE",
                        evidence_summary={
                            "code": "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
                            "forecast_support_examples": [
                                {
                                    "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                                    "forecast_support": {
                                        "forecast_direction": "UNCLEAR",
                                        "forecast_market_support_ok": False,
                                        "top_unselected_signal": {"live_precision_ok": False},
                                    },
                                }
                            ],
                        },
                    ),
                    _request(
                        "COLLECT_BIDASK_REPLAY_EVIDENCE",
                        priority="P1",
                        status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="frontier forecast",
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            frontier = next(item for item in payload["queue"] if item["code"] == "REPAIR_FRONTIER_LANE_BLOCKER")
            self.assertEqual(frontier["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertEqual(payload["selected_request"]["code"], "COLLECT_BIDASK_REPLAY_EVIDENCE")
            self.assertIn(
                "REPAIR_FRONTIER_LANE_BLOCKER",
                payload["queue_summary"]["waiting_request_codes"],
            )
            self.assertNotIn(
                "REPAIR_FRONTIER_LANE_BLOCKER",
                [item["code"] for item in payload["actionable_requests"]],
            )

    def test_protective_frontier_guardrail_waits_and_selects_bidask_evidence_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_FRONTIER_LANE_BLOCKER",
                        priority="P1",
                        status="FRONTIER_PROTECTIVE_GUARDRAIL_ACTIVE",
                        evidence_summary={
                            "code": "REWARD_RISK_TOO_LOW",
                            "example_lane_ids": [
                                "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"
                            ],
                        },
                    ),
                    _request(
                        "COLLECT_BIDASK_REPLAY_EVIDENCE",
                        priority="P1",
                        status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="frontier reward risk",
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            frontier = next(item for item in payload["queue"] if item["code"] == "REPAIR_FRONTIER_LANE_BLOCKER")
            self.assertEqual(frontier["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertEqual(payload["selected_request"]["code"], "COLLECT_BIDASK_REPLAY_EVIDENCE")
            self.assertIn(
                "REPAIR_FRONTIER_LANE_BLOCKER",
                payload["queue_summary"]["waiting_request_codes"],
            )
            self.assertNotIn(
                "REPAIR_FRONTIER_LANE_BLOCKER",
                [item["code"] for item in payload["actionable_requests"]],
            )

    def test_quote_freshness_frontier_wait_is_not_codex_implementation_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_FRONTIER_LANE_BLOCKER",
                        priority="P1",
                        status=FRONTIER_QUOTE_FRESHNESS_WAIT_STATUS,
                        evidence_summary={
                            "code": "STALE_QUOTE",
                            "example_lane_ids": [
                                "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
                            ],
                        },
                    ),
                    _request(
                        "COLLECT_BIDASK_REPLAY_EVIDENCE",
                        priority="P1",
                        status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="frontier stale quote",
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            frontier = next(item for item in payload["queue"] if item["code"] == "REPAIR_FRONTIER_LANE_BLOCKER")
            self.assertEqual(frontier["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertEqual(payload["selected_request"]["code"], "COLLECT_BIDASK_REPLAY_EVIDENCE")
            self.assertIn(
                "REPAIR_FRONTIER_LANE_BLOCKER",
                payload["queue_summary"]["waiting_request_codes"],
            )
            self.assertNotIn(
                "REPAIR_FRONTIER_LANE_BLOCKER",
                [item["code"] for item in payload["actionable_requests"]],
            )

    def test_margin_capacity_frontier_wait_is_not_codex_implementation_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_FRONTIER_LANE_BLOCKER",
                        priority="P1",
                        status=FRONTIER_MARGIN_CAPACITY_WAIT_STATUS,
                        evidence_summary={
                            "code": "MARGIN_TOO_THIN_FOR_MIN_LOT",
                            "example_lane_ids": [
                                "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
                            ],
                        },
                    ),
                    _request(
                        "COLLECT_BIDASK_REPLAY_EVIDENCE",
                        priority="P1",
                        status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="frontier margin floor",
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            frontier = next(item for item in payload["queue"] if item["code"] == "REPAIR_FRONTIER_LANE_BLOCKER")
            self.assertEqual(frontier["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertEqual(payload["selected_request"]["code"], "COLLECT_BIDASK_REPLAY_EVIDENCE")
            self.assertIn(
                "REPAIR_FRONTIER_LANE_BLOCKER",
                payload["queue_summary"]["waiting_request_codes"],
            )
            self.assertNotIn(
                "REPAIR_FRONTIER_LANE_BLOCKER",
                [item["code"] for item in payload["actionable_requests"]],
            )

    def test_cli_accepts_trader_request_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request("REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY", priority="P0"),
                    _request("REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY", priority="P0"),
                ],
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "trader-repair-orchestrator",
                        "--trader-support-bot",
                        str(support),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                        "--trader-request",
                        "residual method month",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(output.read_text())["selected_request"]["code"],
                "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY",
            )
            self.assertEqual(json.loads(stdout.getvalue())["selected_request_code"], "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY")

    def test_historical_acceptance_window_is_not_codex_implementation_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "REVIEW_CLOSE_GATE_EVIDENCE_FAILURES",
                        priority="P0",
                        status="HISTORICAL_ACCEPTANCE_WINDOW_ACTIVE",
                    ),
                    _request("REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY", priority="P0"),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="close evidence",
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            review = next(
                item for item in payload["queue"] if item["code"] == "REVIEW_CLOSE_GATE_EVIDENCE_FAILURES"
            )
            self.assertEqual(review["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertEqual(
                payload["selected_request"]["code"],
                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
            )
            self.assertNotIn(
                "REVIEW_CLOSE_GATE_EVIDENCE_FAILURES",
                [item["code"] for item in payload["actionable_requests"]],
            )
            self.assertIn(
                "REVIEW_CLOSE_GATE_EVIDENCE_FAILURES",
                payload["queue_summary"]["waiting_request_codes"],
            )

    def test_oanda_audit_only_unproved_wait_is_not_codex_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        OANDA_AUDIT_ONLY_LOCAL_TP_EDGE_REQUEST,
                        priority="P1",
                        status=OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_UNPROVED_STATUS,
                        evidence_summary={
                            "history_complete": True,
                            "historical_replay_can_clear_local_tp_proof": False,
                            "read_only_replay_loop_exhausted": True,
                            "history_coverage": {
                                "status": "LOCAL_HISTORY_COMPLETE",
                                "fetch_commands": [],
                                "complete": True,
                            },
                        },
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="oanda proof",
            ).run()

            self.assertEqual(summary.status, STATUS_BLOCKED)
            payload = json.loads(output.read_text())
            oanda = payload["queue"][0]
            self.assertEqual(oanda["code"], OANDA_AUDIT_ONLY_LOCAL_TP_EDGE_REQUEST)
            self.assertEqual(oanda["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertEqual(payload["selected_request"], {})
            self.assertEqual(payload["actionable_requests"], [])
            self.assertIn(
                OANDA_AUDIT_ONLY_LOCAL_TP_EDGE_REQUEST,
                payload["queue_summary"]["waiting_request_codes"],
            )
            self.assertEqual(payload["codex_work_order"]["status"], "NO_ACTIONABLE_CODEX_WORK")
            loop_prompt = payload["loop_engineering_prompt"]
            self.assertIn("evidence-window work", loop_prompt["current_hypothesis"])
            self.assertIn("validate/mine/package", " ".join(loop_prompt["anti_loop_rules"]))
            self.assertIn("waiting for evidence", " ".join(loop_prompt["next_loop"]))

    def test_bidask_forecast_sample_wait_is_not_codex_implementation_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "COLLECT_BIDASK_REPLAY_EVIDENCE",
                        priority="P1",
                        status="BIDASK_REPLAY_WAITING_FOR_FORECAST_SAMPLE_COVERAGE",
                        evidence_summary={
                            "price_truth_fetch_required": False,
                            "price_truth_coverage": {
                                "status": "PRICE_TRUTH_OK",
                                "missing_price_truth_samples": 0,
                                "missing_price_window_group_count": 0,
                                "history_fetch_command_count": 0,
                            },
                        },
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="bidask replay",
            ).run()

            self.assertEqual(summary.status, STATUS_BLOCKED)
            payload = json.loads(output.read_text())
            bidask = payload["queue"][0]
            self.assertEqual(bidask["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertEqual(payload["selected_request"], {})
            self.assertEqual(payload["actionable_requests"], [])
            self.assertIn(
                "COLLECT_BIDASK_REPLAY_EVIDENCE",
                payload["queue_summary"]["waiting_request_codes"],
            )
            self.assertEqual(payload["codex_work_order"]["status"], "NO_ACTIONABLE_CODEX_WORK")
            loop_prompt = payload["loop_engineering_prompt"]
            self.assertEqual(
                loop_prompt["current_state"]["waiting_request_codes"],
                ["COLLECT_BIDASK_REPLAY_EVIDENCE"],
            )
            self.assertIn("evidence-window work", loop_prompt["current_hypothesis"])
            self.assertIn("waiting for evidence", " ".join(loop_prompt["next_loop"]))
            self.assertIn("trader-repair-orchestrator", " ".join(loop_prompt["verification_commands"]))


def _write_support(path: Path, requests: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "SUPPORT_BLOCKED",
                "repair_requests": requests,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _add_execution_frontier_fixture(path: Path) -> None:
    payload = json.loads(path.read_text())
    payload["entry_readiness"] = {
        "repair_frontier": [
            {
                "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                "pair": "EUR_USD",
                "side": "LONG",
                "method": "BREAKOUT_FAILURE",
                "order_type": "LIMIT",
                "status": "DRY_RUN_BLOCKED",
                "repair_mode": "TP_HARVEST_REPAIR",
                "remaining_blocker_codes_after_guardian_and_repair_exemption": [
                    "MARGIN_TOO_THIN_FOR_MIN_LOT"
                ],
                "tp_proof": {
                    "positive_rotation_mode": "TP_PROVEN_HARVEST",
                    "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                    "capture_take_profit_scope_key": (
                        "EUR_USD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER"
                    ),
                    "capture_take_profit_trades": 20,
                    "capture_take_profit_losses": 0,
                    "positive_rotation_pessimistic_expectancy_jpy": 335.3837,
                },
            }
        ],
        "repair_frontier_remaining_blockers": [
            {
                "code": "MARGIN_TOO_THIN_FOR_MIN_LOT",
                "count": 1,
                "example_lane_ids": ["failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"],
            }
        ],
        "global_unlock_frontier": [
            {
                "lane_id": "range_trader:AUD_CAD:LONG:RANGE_ROTATION",
                "pair": "AUD_CAD",
                "side": "LONG",
                "method": "RANGE_ROTATION",
                "order_type": "LIMIT",
                "remaining_blocker_codes_after_global_unlock": [],
                "global_blocker_codes": [
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"
                ],
                "tp_proof": {
                    "capture_take_profit_scope": "MISSING_METHOD_EXIT",
                    "capture_take_profit_scope_key": (
                        "AUD_CAD|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER"
                    ),
                },
            }
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _request(
    code: str,
    *,
    priority: str,
    status: str = "READY_FOR_CODE_REPAIR",
    suggested_files: list[str] | None = None,
    verification_commands: list[str] | None = None,
    source_findings: list[str] | None = None,
    requires_explicit_operator_approval: bool = False,
    evidence_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "code": code,
        "priority": priority,
        "status": status,
        "source_findings": source_findings or [code.replace("REPAIR_", "")],
        "problem": f"{code} problem",
        "why_now": f"{code} why now",
        "evidence_summary": evidence_summary or {},
        "clearance_conditions": [f"{code} clears"],
        "verification_commands": verification_commands or ["PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot"],
        "suggested_files": suggested_files or ["tests/test_trader_support_bot.py"],
        "required_tests": ["regression", "positive path"],
        "requires_explicit_operator_approval": requires_explicit_operator_approval,
        "automation_contract": {
            "codex_may_execute": REPAIR_AUTOMATION_ALLOWED_ACTIONS,
            "commit_and_live_sync_required": True,
            "quant_rabbit_code_may_call_model_api": False,
            "live_side_effects_allowed": [],
            "requires_explicit_operator_approval_for": REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS,
            "forbidden_direct_actions": REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS,
            "orders_closes_launchd_policy": "approval required",
        },
        "read_only": True,
        "live_side_effects": [],
    }
