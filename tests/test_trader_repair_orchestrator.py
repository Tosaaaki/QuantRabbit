from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.cli import main
from quant_rabbit.trader_repair_orchestrator import (
    STATUS_APPROVAL_REQUIRED,
    STATUS_BLOCKED,
    STATUS_READY,
    TraderRepairOrchestrator,
)
from quant_rabbit.trader_support_bot import (
    REPAIR_AUTOMATION_ALLOWED_ACTIONS,
    REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS,
    REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS,
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
                            "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --max-events 80"
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
            report_text = report.read_text()
            self.assertIn("REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY", report_text)
            self.assertIn("Codex Work Order", report_text)
            self.assertIn("Evidence summary keys", report_text)
            self.assertIn("Dependency", report_text)

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
                        "entry_readiness": {
                            "repair_frontier_remaining_blockers": [
                                {
                                    "code": "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
                                    "count": 2,
                                    "example_lane_ids": [
                                        "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"
                                    ],
                                }
                            ]
                        },
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

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["metrics"]["repair_request_source"], "embedded_support_payload")
            self.assertTrue(payload["metrics"]["recovered_from_embedded_support"])
            self.assertEqual(
                payload["selected_request"]["code"],
                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
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


def _request(
    code: str,
    *,
    priority: str,
    status: str = "READY_FOR_CODE_REPAIR",
    suggested_files: list[str] | None = None,
    verification_commands: list[str] | None = None,
    requires_explicit_operator_approval: bool = False,
    evidence_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "code": code,
        "priority": priority,
        "status": status,
        "source_findings": [code.replace("REPAIR_", "")],
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
