from __future__ import annotations

import copy
import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.cli import main
from quant_rabbit.execution_timing_contracts import MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND
from quant_rabbit.trader_repair_orchestrator import (
    EVIDENCE_ACTION_MATERIAL_SOURCE_NEXT_ACTION,
    EVIDENCE_ACTION_MATERIAL_SOURCE_PROGRESS_RECEIPT,
    READ_ONLY_EVIDENCE_WORK_STATUS,
    READ_ONLY_EVIDENCE_WAIT_STATUS,
    STATUS_APPROVAL_REQUIRED,
    STATUS_BLOCKED,
    STATUS_READY,
    TraderRepairOrchestrator,
    _canonical_sha256,
    _count_watermark,
    _evidence_action_has_pending_valid_watermark,
    _evaluate_success_condition,
    _next_evidence_actions_for_empty_proof_queue,
    validated_evidence_action_material,
)
from quant_rabbit.trader_support_bot import (
    DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST,
    DIRECTIONAL_INVERSION_REPLAY_WAIT_STATUS,
    FRONTIER_MARGIN_CAPACITY_WAIT_STATUS,
    FRONTIER_PROOF_EVIDENCE_WAIT_STATUS,
    FRONTIER_QUOTE_FRESHNESS_WAIT_STATUS,
    FRONTIER_STRATEGY_PROFILE_EVIDENCE_WAIT_STATUS,
    OANDA_AUDIT_ONLY_LOCAL_TP_EDGE_REQUEST,
    OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_UNPROVED_STATUS,
    PENDING_CANCEL_RECEIPT_WAIT_STATUS,
    PENDING_CANCEL_REVIEW_CODE,
    REPAIR_AUTOMATION_ALLOWED_ACTIONS,
    REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS,
    REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS,
    TP_PROGRESS_GUARDIAN_WAIT_STATUS,
    TP_PROGRESS_LIVE_EVIDENCE_WAIT_STATUS,
)


class TraderRepairOrchestratorTest(unittest.TestCase):
    def test_validated_evidence_action_material_accepts_recomputed_not_met_and_met(
        self,
    ) -> None:
        origin_state = {
            "proof_normal_routing_status": "BLOCKED",
            "proof_routing_allowed": False,
            "can_create_live_permission_count": 0,
        }
        action = _next_evidence_actions_for_empty_proof_queue(
            {"categories": [{"category": "lane_board"}]},
            current_state=origin_state,
        )[0]

        baseline_material = validated_evidence_action_material(
            action,
            current_state=origin_state,
            source=EVIDENCE_ACTION_MATERIAL_SOURCE_NEXT_ACTION,
        )
        self.assertIsNotNone(baseline_material)
        self.assertEqual(baseline_material["evaluation_status"], "NOT_MET")
        self.assertFalse(baseline_material["evaluation_passed"])
        self.assertTrue(
            _evidence_action_has_pending_valid_watermark(
                action,
                previous_current_state=origin_state,
            )
        )

        still_pending_state = {
            **origin_state,
            "proof_normal_routing_status": "OPEN",
        }
        carried_action = copy.deepcopy(action)
        carried_action["success_condition_evaluation"] = _evaluate_success_condition(
            carried_action["success_condition"],
            still_pending_state,
        )
        carried_material = validated_evidence_action_material(
            carried_action,
            current_state=still_pending_state,
            source=EVIDENCE_ACTION_MATERIAL_SOURCE_NEXT_ACTION,
        )
        self.assertIsNotNone(carried_material)
        self.assertEqual(carried_material["evaluation_status"], "NOT_MET")
        self.assertNotEqual(
            baseline_material["evaluation_sha256"],
            carried_material["evaluation_sha256"],
        )

        met_state = {
            **still_pending_state,
            "proof_routing_allowed": True,
            "can_create_live_permission_count": 1,
        }
        met_action = copy.deepcopy(action)
        met_action["success_condition_evaluation"] = _evaluate_success_condition(
            met_action["success_condition"],
            met_state,
        )
        met_material = validated_evidence_action_material(
            met_action,
            current_state=met_state,
            source=EVIDENCE_ACTION_MATERIAL_SOURCE_NEXT_ACTION,
        )
        self.assertIsNotNone(met_material)
        self.assertEqual(met_material["evaluation_status"], "MET")
        self.assertTrue(met_material["evaluation_passed"])
        self.assertFalse(
            _evidence_action_has_pending_valid_watermark(
                met_action,
                previous_current_state=met_state,
            )
        )

        progress_receipt = {
            key: copy.deepcopy(met_action[key])
            for key in (
                "action_id",
                "category",
                "progress_watermark_contract",
                "progress_watermark_origin",
                "success_condition",
                "success_condition_evaluation",
            )
        }
        progress_material = validated_evidence_action_material(
            progress_receipt,
            current_state=met_state,
            source=EVIDENCE_ACTION_MATERIAL_SOURCE_PROGRESS_RECEIPT,
        )
        self.assertIsNotNone(progress_material)
        self.assertEqual(progress_material["evaluation_status"], "MET")
        self.assertEqual(
            progress_material["evaluation_sha256"],
            met_material["evaluation_sha256"],
        )

    def test_validated_evidence_action_material_rejects_tampering_and_source_confusion(
        self,
    ) -> None:
        current_state = {
            "proof_normal_routing_status": "BLOCKED",
            "proof_routing_allowed": False,
            "can_create_live_permission_count": 0,
        }
        action = _next_evidence_actions_for_empty_proof_queue(
            {"categories": [{"category": "lane_board"}]},
            current_state=current_state,
        )[0]

        tampered_condition_sha = copy.deepcopy(action)
        tampered_condition_sha["progress_watermark_origin"]["condition_sha256"] = (
            "0" * 64
        )

        tampered_condition = copy.deepcopy(action)
        tampered_condition["success_condition"]["description"] = "forged condition"
        tampered_condition["progress_watermark_origin"]["condition_sha256"] = (
            _canonical_sha256(tampered_condition["success_condition"])
        )

        tampered_state_sha = copy.deepcopy(action)
        tampered_state_sha["progress_watermark_origin"]["condition_state_sha256"] = (
            "0" * 64
        )

        tampered_state_fields = copy.deepcopy(action)
        tampered_origin = tampered_state_fields["progress_watermark_origin"]
        tampered_origin["condition_state"]["unexpected"] = True
        tampered_origin["condition_state_sha256"] = _canonical_sha256(
            tampered_origin["condition_state"]
        )

        lowered_count_watermark = copy.deepcopy(action)
        lowered_origin = lowered_count_watermark["progress_watermark_origin"]
        lowered_origin["condition_state"]["can_create_live_permission_count"] = 1
        lowered_origin["condition_state_sha256"] = _canonical_sha256(
            lowered_origin["condition_state"]
        )

        tampered_evaluation = copy.deepcopy(action)
        tampered_evaluation["success_condition_evaluation"]["checks"][0]["actual"] = (
            "OPEN"
        )

        unknown_action = copy.deepcopy(action)
        unknown_action["action_id"] = "unknown_action"
        unknown_action["progress_watermark_origin"]["action_id"] = "unknown_action"

        for label, candidate in (
            ("condition_sha", tampered_condition_sha),
            ("condition", tampered_condition),
            ("state_sha", tampered_state_sha),
            ("state_fields", tampered_state_fields),
            ("count_watermark", lowered_count_watermark),
            ("evaluation", tampered_evaluation),
            ("unknown_action", unknown_action),
        ):
            with self.subTest(tamper=label):
                self.assertIsNone(
                    validated_evidence_action_material(
                        candidate,
                        current_state=current_state,
                        source=EVIDENCE_ACTION_MATERIAL_SOURCE_NEXT_ACTION,
                    )
                )

        not_met_receipt = {
            key: copy.deepcopy(action[key])
            for key in (
                "action_id",
                "category",
                "progress_watermark_contract",
                "progress_watermark_origin",
                "success_condition",
                "success_condition_evaluation",
            )
        }
        self.assertIsNone(
            validated_evidence_action_material(
                not_met_receipt,
                current_state=current_state,
                source=EVIDENCE_ACTION_MATERIAL_SOURCE_PROGRESS_RECEIPT,
            )
        )
        self.assertIsNone(
            validated_evidence_action_material(
                not_met_receipt,
                current_state=current_state,
                source=EVIDENCE_ACTION_MATERIAL_SOURCE_NEXT_ACTION,
            )
        )
        self.assertIsNone(
            validated_evidence_action_material(
                action,
                current_state=current_state,
                source=EVIDENCE_ACTION_MATERIAL_SOURCE_PROGRESS_RECEIPT,
            )
        )
        receipt_with_extra_field = copy.deepcopy(not_met_receipt)
        receipt_with_extra_field["unexpected"] = True
        self.assertIsNone(
            validated_evidence_action_material(
                receipt_with_extra_field,
                current_state=current_state,
                source=EVIDENCE_ACTION_MATERIAL_SOURCE_PROGRESS_RECEIPT,
            )
        )
        self.assertIsNone(
            validated_evidence_action_material(
                action,
                current_state=current_state,
                source="unknown_source",
            )
        )

    def test_count_watermark_accepts_only_exact_non_negative_integer_counts(self) -> None:
        self.assertEqual(_count_watermark(0), 0)
        self.assertEqual(_count_watermark(3), 3)
        for malformed in (True, False, -1, 1.0, 1.5, "1", None, [], {}):
            with self.subTest(malformed=malformed):
                self.assertIsNone(_count_watermark(malformed))

    def test_count_success_conditions_fail_closed_for_malformed_values(self) -> None:
        increasing = {
            "mode": "all",
            "checks": [{"field": "proof_queue_count", "operator": "gt", "value": 0}],
        }
        for malformed in (True, False, -1, 1.0, 0.5, "1", None, [], {}):
            with self.subTest(direction="increase", malformed=malformed):
                result = _evaluate_success_condition(
                    increasing,
                    {"proof_queue_count": malformed},
                )
                self.assertEqual(result["status"], "NOT_MET")
        self.assertEqual(
            _evaluate_success_condition(increasing, {"proof_queue_count": 1})["status"],
            "MET",
        )

        decreasing = {
            "mode": "all",
            "checks": [
                {
                    "field": "rejected_proof_candidate_count",
                    "operator": "lt",
                    "value": 1,
                }
            ],
        }
        for malformed in (True, False, -1, 0.0, 0.5, "0", None, [], {}):
            with self.subTest(direction="decrease", malformed=malformed):
                result = _evaluate_success_condition(
                    decreasing,
                    {"rejected_proof_candidate_count": malformed},
                )
                self.assertEqual(result["status"], "NOT_MET")
        self.assertEqual(
            _evaluate_success_condition(
                decreasing,
                {"rejected_proof_candidate_count": 0},
            )["status"],
            "MET",
        )

    def test_count_success_condition_requires_exact_integer_expected_watermark(self) -> None:
        for malformed in (True, False, -1, 1.0, 0.5, "1", None, [], {}):
            with self.subTest(malformed=malformed):
                result = _evaluate_success_condition(
                    {
                        "mode": "all",
                        "checks": [
                            {
                                "field": "proof_ready_count",
                                "operator": "gt",
                                "value": malformed,
                            }
                        ],
                    },
                    {"proof_ready_count": 2},
                )
                self.assertEqual(result["status"], "NOT_MET")

    def test_wait_stage_does_not_dispatch_follow_up_evidence_work_early(self) -> None:
        now = datetime(2026, 7, 10, 0, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            support.write_text(
                json.dumps(
                    {
                        "generated_at_utc": now.isoformat(),
                        "status": "SUPPORT_BLOCKED",
                        "repair_requests": [],
                        "entry_readiness": {
                            "live_ready_lanes": 0,
                            "shortest_live_ready_path": {
                                "lane_id": "range_trader:GBP_USD:LONG:RANGE_ROTATION",
                                "pair": "GBP_USD",
                                "side": "LONG",
                                "method": "RANGE_ROTATION",
                                "order_type": "LIMIT",
                                "status": "ACTIVE_PATH_BLOCKED_NEAR_READY_LANE",
                                "selection_basis": "active_trader_contract",
                                "blocker_codes": [
                                    "LIMIT_ENTRY_NOT_BELOW_MARKET",
                                    "FORECAST_WATCH_ONLY",
                                ],
                                "first_next_step": (
                                    "Consume range_rail_geometry_repair artifact: "
                                    "WAIT_FOR_RANGE_RAIL_RECHECK. Follow-up evidence actions: "
                                    "VERIFY_TRIGGER_PROJECTIONS, EXACT_TP_PROOF_COLLECTION. "
                                    "Do not send."
                                ),
                                "active_path": {
                                    "lane_id": "range_trader:GBP_USD:LONG:RANGE_ROTATION",
                                    "pair": "GBP_USD",
                                    "side": "LONG",
                                    "method": "RANGE_ROTATION",
                                    "order_type": "LIMIT",
                                    "status": "EVIDENCE_ACQUISITION",
                                    "live_permission": False,
                                    "blocker_codes": [
                                        "LIMIT_ENTRY_NOT_BELOW_MARKET",
                                        "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                                        "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                                    ],
                                    "next_action": (
                                        "Consume data/range_rail_geometry_repair.json for "
                                        "range_trader:GBP_USD:LONG:RANGE_ROTATION: next safe action "
                                        "is WAIT_FOR_RANGE_RAIL_RECHECK; Follow-up evidence actions: "
                                        "VERIFY_TRIGGER_PROJECTIONS, EXACT_TP_PROOF_COLLECTION. "
                                        "Do not send, cancel, close, or relax gates."
                                    ),
                                },
                            },
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
                now_utc=now,
            ).run()

            self.assertEqual(summary.status, "NO_REPAIR_REQUESTS")
            payload = json.loads(output.read_text())
            work_order = payload["codex_work_order"]
            self.assertEqual(work_order["status"], READ_ONLY_EVIDENCE_WAIT_STATUS)
            self.assertFalse(work_order["live_permission_allowed"])
            self.assertEqual(work_order["live_side_effects"], [])
            self.assertFalse(work_order["commit_and_live_sync_required"])
            self.assertFalse(work_order["dispatch_allowed"])
            self.assertFalse(work_order["repeat_suppressed"])
            self.assertEqual(work_order["action_code"], "WAIT_FOR_RANGE_RAIL_RECHECK")
            self.assertEqual(
                work_order["active_lane_evidence_work"]["lane_id"],
                "range_trader:GBP_USD:LONG:RANGE_ROTATION",
            )
            self.assertEqual(work_order["suggested_commands"], [])
            self.assertEqual(
                work_order["material_change_condition_evaluation"]["status"],
                "NOT_MET",
            )
            self.assertIn(
                "active_lane_evidence_work",
                work_order["proof_state"],
            )
            report_text = report.read_text()
            self.assertIn(READ_ONLY_EVIDENCE_WAIT_STATUS, report_text)
            self.assertNotIn("verify-projections", work_order["suggested_commands"])
            self.assertNotIn("range-rail-geometry-repair", work_order["suggested_commands"])
            self.assertNotIn("as-live-ready-evidence-loop", work_order["suggested_commands"])

            _write_support_with_active_lane(
                support,
                requests=[],
                next_action=(
                    "WAIT_FOR_RANGE_RAIL_RECHECK. Later, the next action is "
                    "VERIFY_TRIGGER_PROJECTIONS; do not run it before the rail wake."
                ),
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                now_utc=now + timedelta(minutes=1),
            ).run()
            later_follow_up = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(later_follow_up["action_code"], "WAIT_FOR_RANGE_RAIL_RECHECK")
            self.assertEqual(later_follow_up["status"], READ_ONLY_EVIDENCE_WAIT_STATUS)
            self.assertEqual(later_follow_up["suggested_commands"], [])

            _write_support_with_active_lane(
                support,
                requests=[],
                next_action=(
                    "Follow-up next safe action is EXACT_TP_PROOF_COLLECTION; "
                    "the current action is WAIT_FOR_RANGE_RAIL_RECHECK."
                ),
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                now_utc=now + timedelta(minutes=2),
            ).run()
            prefixed_follow_up = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(prefixed_follow_up["action_code"], "WAIT_FOR_RANGE_RAIL_RECHECK")
            self.assertEqual(prefixed_follow_up["suggested_commands"], [])

    def test_active_lane_evidence_dispatches_once_then_waits_until_material_change(self) -> None:
        now = datetime(2026, 7, 10, 0, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support_with_active_lane(
                support,
                requests=[],
                generated_at_utc=now.isoformat(),
                local_tp_trades=2,
            )

            first_orchestrator = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                now_utc=now,
            )
            first_orchestrator.run()
            first = json.loads(output.read_text())
            first_work = first["codex_work_order"]
            first_digest = first_work["material_digest"]
            first_condition = first_work["material_change_condition"]
            self.assertEqual(first_work["status"], READ_ONLY_EVIDENCE_WORK_STATUS)
            self.assertEqual(first_work["action_code"], "EXACT_TP_PROOF_COLLECTION")
            self.assertTrue(first_work["dispatch_allowed"])
            self.assertFalse(first_work["repeat_suppressed"])
            self.assertIn(
                "PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop",
                first_work["suggested_commands"],
            )
            self.assertNotIn(
                "PYTHONPATH=src python3 -m quant_rabbit.cli forecast-pattern-refresh",
                first_work["suggested_commands"],
            )
            commands = first_work["suggested_commands"]
            support_indexes = [
                index
                for index, command in enumerate(commands)
                if command == "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot"
            ]
            self.assertEqual(len(support_indexes), 2)
            self.assertLess(
                support_indexes[0],
                commands.index("PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop"),
            )
            self.assertLess(
                commands.index("PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop"),
                commands.index("PYTHONPATH=src python3 -m quant_rabbit.cli as-4x-proof-path"),
            )
            self.assertLess(
                commands.index("PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract"),
                support_indexes[-1],
            )
            self.assertLess(
                support_indexes[-1],
                next(
                    index
                    for index, command in enumerate(commands)
                    if "trader-repair-orchestrator --ack-active-lane-dispatch" in command
                ),
            )
            self.assertLess(
                next(
                    index
                    for index, command in enumerate(commands)
                    if "trader-repair-orchestrator --ack-active-lane-dispatch" in command
                ),
                commands.index("PYTHONPATH=src python3 -m quant_rabbit.cli trader-goal-loop-orchestrator"),
            )
            self.assertTrue(
                any(first_digest in command for command in commands if "--ack-active-lane-dispatch" in command)
            )
            support_steps = [
                step
                for step in first_work["suggested_command_steps"]
                if "trader-support-bot" in step["command"]
            ]
            self.assertEqual(len(support_steps), 2)
            self.assertTrue(all(step["ok_rcs"] == [0, 2] for step in support_steps))
            ack_step = next(
                step
                for step in first_work["suggested_command_steps"]
                if "--ack-active-lane-dispatch" in step["command"]
            )
            self.assertEqual(ack_step["ok_rcs"], [0])

            # A normal refresh preserves the stable pending dispatch; it does
            # not mistake issuance for successful execution.
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                now_utc=now + timedelta(minutes=1),
            ).run()
            pending_work = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(pending_work["status"], READ_ONLY_EVIDENCE_WORK_STATUS)
            self.assertTrue(pending_work["execution_pending"])
            self.assertFalse(pending_work["new_dispatch_issued"])
            self.assertEqual(pending_work["reason_code"], "DISPATCH_PENDING_ACK")
            self.assertEqual(pending_work["pending_dispatch_id"], first_digest)
            self.assertEqual(pending_work["suggested_commands"], commands)

            # Only the exact completion acknowledgement moves the digest into
            # history and closes this work order.
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = main(
                    [
                        "trader-repair-orchestrator",
                        "--trader-support-bot",
                        str(support),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                        "--ack-active-lane-dispatch",
                        first_digest,
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(
                json.loads(stdout.getvalue())["acknowledged_active_lane_dispatch"],
                first_digest,
            )
            second = json.loads(output.read_text())
            second_work = second["codex_work_order"]
            self.assertEqual(second_work["status"], READ_ONLY_EVIDENCE_WAIT_STATUS)
            self.assertFalse(second_work["dispatch_allowed"])
            self.assertTrue(second_work["repeat_suppressed"])
            self.assertEqual(second_work["reason_code"], "MATERIAL_EVIDENCE_UNCHANGED")
            self.assertEqual(second_work["material_digest"], first_digest)
            self.assertEqual(second_work["suggested_commands"], [])
            self.assertEqual(second_work["suggested_command_steps"], [])
            self.assertNotIn(
                "suggested_commands",
                second_work["active_lane_evidence_work"],
            )
            self.assertFalse(
                second_work["active_lane_evidence_work"]["command_plan_available"]
            )
            self.assertNotIn(
                "suggested_commands",
                second_work["proof_state"]["active_lane_evidence_work"],
            )
            self.assertNotIn(
                "suggested_commands",
                second_work["active_lane_evidence_work"]["material_state"],
            )

            # Retrying the same exact ACK is idempotent. This recovers a
            # derivative report/downstream failure after authoritative state
            # already committed without reopening or duplicating the work.
            replay_stdout = io.StringIO()
            with redirect_stdout(replay_stdout):
                replay_rc = main(
                    [
                        "trader-repair-orchestrator",
                        "--trader-support-bot",
                        str(support),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                        "--ack-active-lane-dispatch",
                        first_digest,
                    ]
                )
            self.assertEqual(replay_rc, 0)
            replay_work = json.loads(output.read_text())["codex_work_order"]
            self.assertTrue(replay_work["acknowledgement_replayed"])
            self.assertEqual(replay_work["status"], READ_ONLY_EVIDENCE_WAIT_STATUS)
            self.assertEqual(replay_work["suggested_commands"], [])
            acknowledged_bytes = output.read_bytes()
            mismatch_stdout = io.StringIO()
            with redirect_stdout(mismatch_stdout):
                mismatch_rc = main(
                    [
                        "trader-repair-orchestrator",
                        "--trader-support-bot",
                        str(support),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                        "--ack-active-lane-dispatch",
                        "0" * 64,
                    ]
                )
            self.assertEqual(mismatch_rc, 3)
            self.assertIn("requires an existing pending", mismatch_stdout.getvalue())
            self.assertEqual(output.read_bytes(), acknowledged_bytes)

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                now_utc=now + timedelta(minutes=3),
            ).run()
            third_work = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(third_work["status"], READ_ONLY_EVIDENCE_WAIT_STATUS)
            self.assertTrue(third_work["repeat_suppressed"])
            self.assertEqual(third_work["suggested_commands"], [])

            # A regenerated timestamp alone is not material and must not reopen work.
            _write_support_with_active_lane(
                support,
                requests=[],
                generated_at_utc=(now + timedelta(minutes=4)).isoformat(),
                local_tp_trades=2,
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                now_utc=now + timedelta(minutes=4),
            ).run()
            timestamp_only_work = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(timestamp_only_work["status"], READ_ONLY_EVIDENCE_WAIT_STATUS)
            self.assertEqual(timestamp_only_work["material_digest"], first_digest)

            # A new exact-vehicle TP receipt is material and permits one new pass.
            _write_support_with_active_lane(
                support,
                requests=[],
                generated_at_utc=(now + timedelta(minutes=5)).isoformat(),
                local_tp_trades=3,
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                now_utc=now + timedelta(minutes=5),
            ).run()
            changed = json.loads(output.read_text())
            changed_work = changed["codex_work_order"]
            self.assertEqual(changed_work["status"], READ_ONLY_EVIDENCE_WORK_STATUS)
            self.assertTrue(changed_work["dispatch_allowed"])
            self.assertNotEqual(changed_work["material_digest"], first_digest)
            self.assertEqual(
                _evaluate_success_condition(
                    first_condition,
                    changed["loop_engineering_prompt"]["current_state"],
                )["status"],
                "MET",
            )

    def test_whitespace_ack_is_rejected_before_state_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support_with_active_lane(support, requests=[])
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                rc = main(
                    [
                        "trader-repair-orchestrator",
                        "--trader-support-bot",
                        str(support),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                        "--ack-active-lane-dispatch",
                        "   ",
                    ]
                )

        self.assertEqual(rc, 3)
        self.assertIn("requires a non-empty exact digest", stdout.getvalue())
        self.assertFalse(output.exists())
        self.assertFalse(report.exists())

    def test_fired_guardian_and_explicit_action_aliases_advance_one_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"

            _write_support_with_active_lane(
                support,
                requests=[],
                next_action=(
                    "Consume data/guardian_events.json: CONTRACT_ADD_TRIGGER fired for the "
                    "watched range rail. Do not repeat WAIT_FOR_RANGE_RAIL_RECHECK; refresh "
                    "broker truth and active board, then reprice the RANGE_ROTATION counterpart "
                    "and continue exact TP-proof collection."
                ),
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            fired_work = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(
                fired_work["action_code"],
                "REPRICE_RANGE_ROTATION_COUNTERPART",
            )
            self.assertEqual(fired_work["status"], READ_ONLY_EVIDENCE_WORK_STATUS)
            self.assertTrue(
                any("broker-snapshot" in command for command in fired_work["suggested_commands"])
            )
            self.assertIn(
                "PYTHONPATH=src python3 -m quant_rabbit.cli range-rail-geometry-repair",
                fired_work["suggested_commands"],
            )
            self.assertLess(
                next(
                    index
                    for index, command in enumerate(fired_work["suggested_commands"])
                    if "generate-intents" in command
                ),
                fired_work["suggested_commands"].index(
                    "PYTHONPATH=src python3 -m quant_rabbit.cli range-rail-geometry-repair"
                ),
            )
            fired_commands = fired_work["suggested_commands"]
            frontier_index = fired_commands.index(
                "PYTHONPATH=src python3 -m quant_rabbit.cli non-eurusd-live-grade-frontier"
            )
            entry_recovery_index = fired_commands.index(
                "PYTHONPATH=src python3 -m quant_rabbit.cli entry-frequency-recovery"
            )
            forecast_refresh_index = fired_commands.index(
                "PYTHONPATH=src python3 -m quant_rabbit.cli forecast-pattern-refresh"
            )
            range_repair_index = fired_commands.index(
                "PYTHONPATH=src python3 -m quant_rabbit.cli range-rail-geometry-repair"
            )
            self.assertLess(frontier_index, entry_recovery_index)
            self.assertLess(entry_recovery_index, forecast_refresh_index)
            self.assertLess(forecast_refresh_index, range_repair_index)
            self.assertTrue(
                any(
                    command == "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract"
                    for command in fired_commands[forecast_refresh_index + 1 : range_repair_index]
                )
            )

            _write_support_with_active_lane(
                support,
                requests=[],
                next_action=(
                    "Consume the forecast artifact: next safe action is "
                    "REFRESH_FORECAST_RANGE_BOX; follow-up evidence action is "
                    "EXACT_TP_PROOF_COLLECTION."
                ),
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                ack_active_lane_dispatch=fired_work["material_digest"],
            ).run()
            forecast_work = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(forecast_work["action_code"], "FORECAST_PATTERN_REFRESH")
            self.assertIn(
                "PYTHONPATH=src python3 -m quant_rabbit.cli forecast-pattern-refresh",
                forecast_work["suggested_commands"],
            )
            self.assertNotIn(
                "PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop",
                forecast_work["suggested_commands"],
            )

            _write_support_with_active_lane(
                support,
                requests=[],
                next_action=(
                    "Consume entry recovery: next safe action is "
                    "TRIGGER_PROJECTION_TO_LIMIT_PROOF, METHOD_SCOPED_PROFILE_PROMOTION, "
                    "and EXACT_TP_PROOF_COLLECTION."
                ),
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                ack_active_lane_dispatch=forecast_work["material_digest"],
            ).run()
            projection_work = json.loads(output.read_text())["codex_work_order"]
            projection_commands = projection_work["suggested_commands"]
            self.assertEqual(
                projection_work["action_code"],
                "TRIGGER_PROJECTION_TO_LIMIT_PROOF",
            )
            self.assertLess(
                projection_commands.index(
                    "PYTHONPATH=src python3 -m quant_rabbit.cli verify-projections"
                ),
                projection_commands.index(
                    "PYTHONPATH=src python3 -m quant_rabbit.cli forecast-pattern-refresh"
                ),
            )
            self.assertLess(
                projection_commands.index(
                    "PYTHONPATH=src python3 -m quant_rabbit.cli forecast-pattern-refresh"
                ),
                projection_commands.index(
                    "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract"
                ),
            )
            self.assertNotIn(
                "PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop",
                projection_commands,
            )

            _write_support_with_active_lane(
                support,
                requests=[],
                next_action=(
                    "Consume entry_frequency_recovery: next safe tuning action is "
                    "CURRENT_INTENT_REGEN_REQUIRED; do not send."
                ),
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                ack_active_lane_dispatch=projection_work["material_digest"],
            ).run()
            regen_work = json.loads(output.read_text())["codex_work_order"]
            regen_commands = regen_work["suggested_commands"]
            self.assertEqual(regen_work["action_code"], "CURRENT_INTENT_REGEN_REQUIRED")
            self.assertLess(
                next(index for index, command in enumerate(regen_commands) if "broker-snapshot" in command),
                next(index for index, command in enumerate(regen_commands) if "daily-target-state" in command),
            )
            self.assertLess(
                next(index for index, command in enumerate(regen_commands) if "daily-target-state" in command),
                next(index for index, command in enumerate(regen_commands) if "generate-intents" in command),
            )

            _write_support_with_active_lane(
                support,
                requests=[],
                next_action=(
                    "Consume entry_frequency_recovery: next safe tuning action is "
                    "METHOD_SCOPED_PROFILE_PROMOTION; do not send."
                ),
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                ack_active_lane_dispatch=regen_work["material_digest"],
            ).run()
            unmapped = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(unmapped["status"], "ACTIVE_LANE_ACTION_MAPPING_REQUIRED")
            self.assertEqual(unmapped["reason_code"], "ACTION_STAGE_MAPPING_REQUIRED")
            self.assertFalse(unmapped["dispatch_allowed"])
            self.assertEqual(unmapped["suggested_commands"], [])
            self.assertTrue(unmapped["commit_and_live_sync_required"])
            self.assertIn(
                "src/quant_rabbit/trader_repair_orchestrator.py",
                unmapped["suggested_files"],
            )
            self.assertTrue(unmapped["targeted_test_commands"])

    def test_frontier_tp_proof_collection_alias_dispatches_exact_tp_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support_with_active_lane(
                support,
                requests=[],
                next_action=(
                    "TP_PROOF_COLLECTION: collect exact TAKE_PROFIT_ORDER proof for "
                    "range_trader:AUD_CAD:SHORT:RANGE_ROTATION; do not mix market-close losses."
                ),
            )

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            work = json.loads(output.read_text())["codex_work_order"]

        self.assertEqual(work["status"], "READ_ONLY_EVIDENCE_WORK")
        self.assertEqual(work["action_code"], "EXACT_TP_PROOF_COLLECTION")
        self.assertIn(
            "PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop",
            work["suggested_commands"],
        )
        self.assertFalse(work["commit_and_live_sync_required"])

    def test_primary_exact_tp_action_outranks_parallel_frontier_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support_with_active_lane(
                support,
                requests=[],
                next_action=(
                    "Use the latest active board. Collect exact local TAKE_PROFIT_ORDER proof "
                    "for EUR_USD|SHORT|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER. "
                    "Pair this with non_eurusd_live_grade_frontier evidence lane "
                    "range_trader:AUD_CAD:SHORT:RANGE_ROTATION. The next safe action is "
                    "WAIT_FOR_RANGE_RAIL_RECHECK; do not send."
                ),
            )

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            work = json.loads(output.read_text())["codex_work_order"]

        self.assertEqual(work["status"], "READ_ONLY_EVIDENCE_WORK")
        self.assertEqual(work["action_code"], "EXACT_TP_PROOF_COLLECTION")
        self.assertTrue(work["dispatch_allowed"])
        self.assertIn(
            "PYTHONPATH=src python3 -m quant_rabbit.cli as-4x-proof-path",
            work["suggested_commands"],
        )

    def test_primary_exact_tp_action_outranks_parallel_fired_guardian_reprice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support_with_active_lane(
                support,
                requests=[],
                next_action=(
                    "Collect exact local TAKE_PROFIT_ORDER proof for "
                    "EUR_USD|SHORT|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER. "
                    "Parallel non_eurusd_live_grade_frontier evidence lane "
                    "range_trader:AUD_CAD:SHORT:RANGE_ROTATION. CONTRACT_ADD_TRIGGER fired; "
                    "do not repeat WAIT_FOR_RANGE_RAIL_RECHECK; reprice the RANGE_ROTATION "
                    "counterpart."
                ),
            )

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            work = json.loads(output.read_text())["codex_work_order"]

        self.assertEqual(work["action_code"], "EXACT_TP_PROOF_COLLECTION")
        self.assertNotIn(
            "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot "
            "data/broker_snapshot.json --reuse-market-artifacts",
            work["suggested_commands"],
        )

    def test_state_lock_serializes_ack_against_ordinary_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support_with_active_lane(support, requests=[])
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            digest = json.loads(output.read_text())["codex_work_order"]["material_digest"]

            import quant_rabbit.trader_repair_orchestrator as repair_module

            original_read = repair_module._read_previous_orchestrator_output
            ack_read_entered = threading.Event()
            release_ack_read = threading.Event()
            refresh_read_entered = threading.Event()
            errors: list[BaseException] = []

            def controlled_read(path: Path):
                if threading.current_thread().name == "ack-run":
                    ack_read_entered.set()
                    if not release_ack_read.wait(timeout=3):
                        raise TimeoutError("test did not release ACK read")
                elif threading.current_thread().name == "refresh-run":
                    refresh_read_entered.set()
                return original_read(path)

            def execute(orchestrator: TraderRepairOrchestrator) -> None:
                try:
                    orchestrator.run()
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            ack_thread = threading.Thread(
                name="ack-run",
                target=execute,
                args=(
                    TraderRepairOrchestrator(
                        support_bot_path=support,
                        output_path=output,
                        report_path=report,
                        ack_active_lane_dispatch=digest,
                    ),
                ),
            )
            refresh_thread = threading.Thread(
                name="refresh-run",
                target=execute,
                args=(
                    TraderRepairOrchestrator(
                        support_bot_path=support,
                        output_path=output,
                        report_path=report,
                    ),
                ),
            )
            with patch(
                "quant_rabbit.trader_repair_orchestrator._read_previous_orchestrator_output",
                side_effect=controlled_read,
            ):
                ack_thread.start()
                self.assertTrue(ack_read_entered.wait(timeout=2))
                refresh_thread.start()
                refresh_entered_while_ack_held_lock = refresh_read_entered.wait(timeout=0.15)
                release_ack_read.set()
                ack_thread.join(timeout=3)
                refresh_thread.join(timeout=3)

            final_work = json.loads(output.read_text())["codex_work_order"]

        self.assertFalse(refresh_entered_while_ack_held_lock)
        self.assertFalse(ack_thread.is_alive())
        self.assertFalse(refresh_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(final_work["status"], READ_ONLY_EVIDENCE_WAIT_STATUS)
        self.assertIsNone(final_work["pending_material_digest"])
        self.assertIn(digest, final_work["dispatched_material_digests"])

    def test_report_publish_failure_keeps_pending_ack_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support_with_active_lane(support, requests=[])
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            pending = json.loads(output.read_text())["codex_work_order"]
            digest = pending["material_digest"]

            import quant_rabbit.trader_repair_orchestrator as repair_module

            original_replace = repair_module._replace_prepared_text

            def fail_report_replace(temp_path: Path, destination: Path) -> None:
                if destination == report:
                    raise OSError("simulated report ENOSPC")
                original_replace(temp_path, destination)

            with patch(
                "quant_rabbit.trader_repair_orchestrator._replace_prepared_text",
                side_effect=fail_report_replace,
            ):
                with self.assertRaisesRegex(OSError, "simulated report ENOSPC"):
                    TraderRepairOrchestrator(
                        support_bot_path=support,
                        output_path=output,
                        report_path=report,
                        ack_active_lane_dispatch=digest,
                    ).run()

            after_failure = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(after_failure["pending_material_digest"], digest)
            self.assertNotIn(digest, after_failure["dispatched_material_digests"])

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                ack_active_lane_dispatch=digest,
            ).run()
            recovered = json.loads(output.read_text())["codex_work_order"]
            recovered_report = report.read_text()

        self.assertEqual(recovered["status"], READ_ONLY_EVIDENCE_WAIT_STATUS)
        self.assertIn(digest, recovered["dispatched_material_digests"])
        self.assertTrue(recovered_report)

    def test_approval_or_evidence_wait_takes_precedence_over_active_lane_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support_with_active_lane(
                support,
                requests=[
                    _request(
                        "RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT",
                        priority="P0",
                        requires_explicit_operator_approval=True,
                    )
                ],
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            approval = json.loads(output.read_text())
            self.assertEqual(approval["status"], STATUS_APPROVAL_REQUIRED)
            self.assertEqual(
                approval["codex_work_order"]["status"],
                "NO_ACTIONABLE_CODEX_WORK",
            )

            _write_support_with_active_lane(
                support,
                requests=[
                    _request(
                        "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                        priority="P0",
                        status=TP_PROGRESS_LIVE_EVIDENCE_WAIT_STATUS,
                    )
                ],
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            waiting = json.loads(output.read_text())
            self.assertEqual(waiting["status"], STATUS_BLOCKED)
            self.assertEqual(
                waiting["codex_work_order"]["status"],
                "NO_ACTIONABLE_CODEX_WORK",
            )

    def test_last_evidence_dispatch_survives_an_intervening_repair_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support_with_active_lane(support, requests=[])
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            first_work = json.loads(output.read_text())["codex_work_order"]
            digest = first_work["material_digest"]
            self.assertEqual(first_work["status"], READ_ONLY_EVIDENCE_WORK_STATUS)

            _write_support_with_active_lane(
                support,
                requests=[
                    _request(
                        "RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT",
                        priority="P0",
                        requires_explicit_operator_approval=True,
                    )
                ],
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            queued_work = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(queued_work["status"], "NO_ACTIONABLE_CODEX_WORK")
            self.assertEqual(queued_work["pending_material_digest"], digest)
            self.assertIsNone(queued_work["last_dispatched_material_digest"])

            _write_support_with_active_lane(support, requests=[])
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            resumed_work = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(resumed_work["status"], READ_ONLY_EVIDENCE_WORK_STATUS)
            self.assertTrue(resumed_work["execution_pending"])
            self.assertFalse(resumed_work["new_dispatch_issued"])
            self.assertEqual(resumed_work["material_digest"], digest)
            self.assertTrue(resumed_work["suggested_commands"])

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                ack_active_lane_dispatch=digest,
            ).run()
            acknowledged = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(acknowledged["status"], READ_ONLY_EVIDENCE_WAIT_STATUS)
            self.assertTrue(acknowledged["repeat_suppressed"])
            self.assertEqual(acknowledged["suggested_commands"], [])

    def test_dispatch_history_suppresses_a_material_state_after_lane_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support_with_active_lane(support, requests=[])
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            first = json.loads(output.read_text())["codex_work_order"]
            first_digest = first["material_digest"]
            self.assertEqual(first["status"], READ_ONLY_EVIDENCE_WORK_STATUS)

            _write_support_with_active_lane(
                support,
                requests=[],
                next_action="The next safe action is FORECAST_PATTERN_REFRESH; do not send.",
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                ack_active_lane_dispatch=first_digest,
            ).run()
            second = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(second["status"], READ_ONLY_EVIDENCE_WORK_STATUS)
            self.assertNotEqual(second["material_digest"], first_digest)

            _write_support_with_active_lane(support, requests=[])
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                ack_active_lane_dispatch=second["material_digest"],
            ).run()
            cycled = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(cycled["material_digest"], first_digest)
            self.assertEqual(cycled["status"], READ_ONLY_EVIDENCE_WAIT_STATUS)
            self.assertTrue(cycled["repeat_suppressed"])
            self.assertEqual(cycled["suggested_commands"], [])
            self.assertIn(first_digest, cycled["dispatched_material_digests"])
            self.assertIn(second["material_digest"], cycled["dispatched_material_digests"])

    def test_terminal_material_change_supersedes_unexecuted_stale_pending_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support_with_active_lane(support, requests=[])
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            first = json.loads(output.read_text())["codex_work_order"]

            _write_support_with_active_lane(
                support,
                requests=[],
                next_action="The next safe action is FORECAST_PATTERN_REFRESH; do not send.",
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            terminal = json.loads(output.read_text())["codex_work_order"]

            self.assertEqual(terminal["status"], READ_ONLY_EVIDENCE_WORK_STATUS)
            self.assertTrue(terminal["new_dispatch_issued"])
            self.assertTrue(terminal["execution_pending"])
            self.assertEqual(
                terminal["superseded_pending_material_digest"],
                first["material_digest"],
            )
            self.assertNotEqual(terminal["pending_dispatch_id"], first["pending_dispatch_id"])
            self.assertNotIn(first["material_digest"], terminal["dispatched_material_digests"])

    def test_completed_ack_replay_preserves_newer_material_pending_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support_with_active_lane(support, requests=[])
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            first = json.loads(output.read_text())["codex_work_order"]

            _write_support_with_active_lane(
                support,
                requests=[],
                next_action="The next safe action is FORECAST_PATTERN_REFRESH; do not send.",
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                ack_active_lane_dispatch=first["material_digest"],
            ).run()
            newer = json.loads(output.read_text())["codex_work_order"]
            self.assertEqual(newer["status"], READ_ONLY_EVIDENCE_WORK_STATUS)
            self.assertIn(first["material_digest"], newer["dispatched_material_digests"])
            self.assertNotEqual(newer["pending_material_digest"], first["material_digest"])

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                ack_active_lane_dispatch=first["material_digest"],
            ).run()
            replayed = json.loads(output.read_text())["codex_work_order"]

        self.assertTrue(replayed["acknowledgement_replayed"])
        self.assertEqual(replayed["status"], READ_ONLY_EVIDENCE_WORK_STATUS)
        self.assertEqual(replayed["pending_material_digest"], newer["pending_material_digest"])
        self.assertIn(first["material_digest"], replayed["dispatched_material_digests"])

    def test_corrupt_previous_output_self_recovers_to_atomic_valid_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support_with_active_lane(support, requests=[])
            output.write_text("{", encoding="utf-8")

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            payload = json.loads(output.read_text())

            self.assertEqual(
                payload["metrics"]["previous_output_recovery"],
                "CORRUPT_PREVIOUS_OUTPUT_IGNORED",
            )
            self.assertEqual(
                payload["codex_work_order"]["status"],
                READ_ONLY_EVIDENCE_WORK_STATUS,
            )
            self.assertTrue(report.read_text())

    def test_pending_cancel_review_waits_for_trader_receipt_not_codex_implementation(self) -> None:
        now = datetime(2026, 6, 25, 2, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        PENDING_CANCEL_REVIEW_CODE,
                        priority="P0",
                        status=PENDING_CANCEL_RECEIPT_WAIT_STATUS,
                        source_findings=[PENDING_CANCEL_REVIEW_CODE],
                        evidence_summary={"cancel_review_order_ids": ["472818"]},
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
        self.assertEqual(summary.actionable_request_count, 0)
        self.assertEqual(summary.waiting_request_count, 1)
        self.assertEqual(payload["selected_request_code"], None)
        self.assertEqual(payload["queue"][0]["code"], PENDING_CANCEL_REVIEW_CODE)
        self.assertEqual(payload["queue"][0]["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
        self.assertEqual(
            payload["loop_engineering_prompt"]["current_state"]["waiting_p0_request_codes"],
            [PENDING_CANCEL_REVIEW_CODE],
        )
        self.assertIn(PENDING_CANCEL_REVIEW_CODE, payload["loop_engineering_prompt"]["prompt_text"])
        self.assertIn("472818", json.dumps(payload["queue"][0]["evidence_summary"]))

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
            self.assertEqual(summary.repair_request_count, 2)
            self.assertEqual(summary.actionable_request_count, 1)
            self.assertEqual(summary.approval_required_request_count, 1)
            self.assertEqual(summary.waiting_request_count, 0)
            payload = json.loads(output.read_text())
            self.assertTrue(payload["read_only"])
            self.assertEqual(payload["live_side_effects"], [])
            self.assertEqual(payload["repair_request_count"], 2)
            self.assertEqual(payload["actionable_request_count"], 1)
            self.assertEqual(payload["approval_required_request_count"], 1)
            self.assertEqual(payload["waiting_request_count"], 0)
            self.assertEqual(payload["metrics"]["repair_request_count"], payload["repair_request_count"])
            self.assertEqual(payload["metrics"]["actionable_request_count"], payload["actionable_request_count"])
            self.assertEqual(
                payload["metrics"]["approval_required_request_count"],
                payload["approval_required_request_count"],
            )
            self.assertEqual(payload["metrics"]["waiting_request_count"], payload["waiting_request_count"])
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

    def test_loop_prompt_embeds_profitability_rca_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            support.write_text(
                json.dumps(
                    {
                        "status": "SUPPORT_BLOCKED",
                        "target": {
                            "status": "PURSUE_TARGET",
                            "current_equity_raw": 270516.42,
                            "capital_flows_30d": 100000.0,
                            "funding_adjusted_equity": 170516.42,
                            "rolling_30d_multiplier_funding_adjusted": 0.994641,
                            "remaining_to_4x_funding_adjusted": 515223.801,
                            "required_calendar_daily_return_funding_adjusted": 5.633087,
                            "required_active_day_return_funding_adjusted": 7.759236,
                            "performance_basis": "funding_adjusted",
                            "sizing_basis": "raw_nav",
                            "pace_state": "BEHIND_4X_PACE",
                        },
                        "entry_readiness": {"live_ready_lanes": 0},
                        "profitability_acceptance": {
                            "status": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                            "blockers": [
                                (
                                    "NEGATIVE_EXPECTANCY_ACTIVE: capture economics "
                                    "is still NEGATIVE_EXPECTANCY"
                                ),
                                (
                                    "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE: "
                                    "TP-proven segment damaged by market-close leakage"
                                ),
                            ],
                            "capture_economics": {
                                "status": "NEGATIVE_EXPECTANCY",
                                "overall": {
                                    "expectancy_jpy_per_trade": -162.0,
                                    "net_jpy": -36278.3,
                                    "trades": 224,
                                },
                                "take_profit": {
                                    "expectancy_jpy_per_trade": 508.4,
                                    "net_jpy": 48804.8,
                                    "trades": 96,
                                },
                                "market_close": {
                                    "expectancy_jpy_per_trade": -815.9,
                                    "net_jpy": -75879.8,
                                    "trades": 93,
                                },
                                "tp_proven_market_close_leak_segments": 1,
                            },
                            "target_firepower": {
                                "operational_minimum_5pct_reachable": False,
                                "minimum_5pct_estimated_reachable": True,
                                "operational_blocker_codes": [
                                    "LOSS_CLOSE_PROFIT_CAPTURE_MISSED",
                                    "NO_LIVE_READY_LANES",
                                ],
                            },
                        },
                        "repair_requests": [
                            _request(
                                "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                                priority="P0",
                                status=TP_PROGRESS_LIVE_EVIDENCE_WAIT_STATUS,
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

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            payload = json.loads(output.read_text())
            loop_prompt = payload["loop_engineering_prompt"]
            state = loop_prompt["current_state"]
            self.assertEqual(state["funding_adjusted_equity"], 170516.42)
            self.assertEqual(state["current_equity_raw"], 270516.42)
            self.assertEqual(state["capital_flows_30d"], 100000.0)
            self.assertEqual(state["rolling_30d_multiplier_funding_adjusted"], 0.994641)
            self.assertEqual(state["remaining_to_4x_funding_adjusted"], 515223.801)
            self.assertEqual(state["required_calendar_daily_return_funding_adjusted"], 5.633087)
            self.assertEqual(state["required_active_day_return_funding_adjusted"], 7.759236)
            self.assertEqual(state["performance_basis"], "funding_adjusted")
            self.assertEqual(state["sizing_basis"], "raw_nav")
            self.assertEqual(state["pace_state"], "BEHIND_4X_PACE")
            rca = loop_prompt["current_state"]["profitability_rca_summary"]
            self.assertEqual(rca["capture_economics_status"], "NEGATIVE_EXPECTANCY")
            self.assertEqual(rca["overall_expectancy_jpy_per_trade"], -162.0)
            self.assertEqual(rca["take_profit_expectancy_jpy_per_trade"], 508.4)
            self.assertEqual(rca["market_close_expectancy_jpy_per_trade"], -815.9)
            self.assertEqual(
                rca["acceptance_blocker_codes"],
                [
                    "NEGATIVE_EXPECTANCY_ACTIVE",
                    "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
                ],
            )
            self.assertFalse(rca["operational_minimum_5pct_reachable"])
            self.assertTrue(rca["audit_minimum_5pct_estimated_reachable"])
            self.assertIn("Profitability RCA: capture=NEGATIVE_EXPECTANCY", loop_prompt["prompt_text"])
            self.assertIn("overall_exp_jpy=-162.0", loop_prompt["prompt_text"])
            self.assertIn("tp_exp_jpy=508.4", loop_prompt["prompt_text"])
            self.assertIn("market_close_exp_jpy=-815.9", loop_prompt["prompt_text"])
            self.assertIn("NEGATIVE_EXPECTANCY_ACTIVE", loop_prompt["prompt_text"])
            self.assertIn("rolling 30d/monthly 4x funding-adjusted equity", loop_prompt["prompt_text"])
            self.assertIn("funding_adjusted_equity=170516.42", loop_prompt["prompt_text"])
            self.assertIn("remaining_to_4x_funding_adjusted=515223.801", loop_prompt["prompt_text"])
            self.assertIn("sizing_basis=raw_nav", loop_prompt["prompt_text"])
            self.assertIn(
                "Do not derive lot size from remaining_to_4x_funding_adjusted",
                " ".join(loop_prompt["anti_loop_rules"]),
            )
            report_text = report.read_text()
            self.assertIn("4x funding-adjusted multiplier", report_text)
            self.assertIn("Remaining to 4x funding-adjusted: `515223.801`", report_text)

    def test_loop_prompt_embeds_as_proof_queue_and_gateway_state(self) -> None:
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
                        status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                    )
                ],
            )
            _write_as_proof_artifacts(root)

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            state = payload["loop_engineering_prompt"]["current_state"]
            self.assertEqual(state["proof_queue_count"], 0)
            self.assertEqual(state["proof_ready_count"], 0)
            self.assertEqual(state["can_create_live_permission_count"], 0)
            self.assertEqual(state["rejected_proof_candidate_count"], 4)
            self.assertFalse(state["as_live_ready_path_exists"])
            self.assertEqual(state["proof_primary_blocker"], "PROFITABILITY_ACCEPTANCE_BLOCKED")
            self.assertEqual(state["proof_normal_routing_status"], "BLOCKED")
            self.assertFalse(state["proof_routing_allowed"])
            self.assertEqual(state["portfolio_status"], "NO_LIVE_READY_PORTFOLIO")
            self.assertFalse(state["portfolio_can_reach_4x_now"])
            self.assertEqual(state["gateway_status"], "NO_LIVE_READY_INTENT")
            self.assertIn("NEGATIVE_EXPECTANCY_ACTIVE", state["proof_global_blockers"])
            reason = payload["proof_queue_empty_reason"]
            self.assertEqual(reason["status"], "EMPTY")
            self.assertEqual(reason["primary_category"], "lane_board")
            self.assertEqual(reason["primary_reason_code"], "LANE_BOARD_NORMAL_ROUTING_BLOCKED")
            self.assertEqual(reason["primary_causal_rank"], 0)
            self.assertEqual(reason["proof_queue_count"], 0)
            self.assertEqual(reason["can_create_live_permission_count"], 0)
            reason_categories = [item["category"] for item in reason["categories"]]
            self.assertEqual(
                reason_categories,
                [
                    "rejected_proof_candidates",
                    "lane_board",
                    "portfolio_planner",
                    "gateway_issue",
                ],
            )
            by_category = {item["category"]: item for item in reason["categories"]}
            for category in by_category.values():
                self.assertIn("freshness", category)
                self.assertIn("causal_rank", category)
                self.assertIn("blocking_depth", category)
                self.assertIn("causal_basis", category)
                self.assertIn("status", category["freshness"])
                self.assertIn("freshness_age_seconds", category["freshness"])
                self.assertIn("freshness_reference_timestamp", category["freshness"])
                self.assertIn("freshness_max_age_seconds", category["freshness"])
            self.assertLess(
                by_category["lane_board"]["causal_rank"],
                by_category["rejected_proof_candidates"]["causal_rank"],
            )
            self.assertGreater(
                by_category["lane_board"]["blocking_depth"],
                by_category["gateway_issue"]["blocking_depth"],
            )
            self.assertEqual(
                reason["categories"][0]["reason_code"],
                "ALL_CURRENT_PROOF_CANDIDATES_REJECTED_BEFORE_QUEUE",
            )
            self.assertIn(
                "spread_included_bidask_replay_negative_for_exact_lane",
                reason["categories"][0]["rejection_reasons"],
            )
            self.assertEqual(
                reason["categories"][1]["primary_blocker"],
                "PROFITABILITY_ACCEPTANCE_BLOCKED",
            )
            actions = payload["next_evidence_actions"]
            self.assertEqual(
                [item["category"] for item in actions],
                [
                    "rejected_proof_candidates",
                    "lane_board",
                    "portfolio_planner",
                    "gateway_issue",
                ],
            )
            self.assertTrue(all(item["read_only"] for item in actions))
            self.assertTrue(all(item["live_side_effects"] == [] for item in actions))
            for action in actions:
                self.assertIsInstance(action["success_condition"], dict)
                self.assertEqual(
                    action["success_condition"]["schema_version"],
                    "success_condition_v1",
                )
                self.assertIn("checks", action["success_condition"])
                self.assertEqual(action["success_condition_evaluation"]["status"], "NOT_MET")
            rejected_action = actions[0]
            rejected_eval = _evaluate_success_condition(
                rejected_action["success_condition"],
                {
                    **state,
                    "proof_queue_count": 1,
                    "proof_ready_count": 0,
                    "can_create_live_permission_count": 0,
                },
            )
            self.assertEqual(rejected_eval["status"], "MET")
            loop_prompt = payload["loop_engineering_prompt"]
            self.assertIn("proof_queue=0", loop_prompt["prompt_text"])
            self.assertIn("live_permission_candidates=0", loop_prompt["prompt_text"])
            self.assertIn("rejected_proof_candidates=4", loop_prompt["prompt_text"])
            self.assertIn("gateway=NO_LIVE_READY_INTENT", loop_prompt["prompt_text"])
            self.assertIn("A/S proof state:", loop_prompt["prompt_text"])
            self.assertIn("A/S proof empty reason:", loop_prompt["prompt_text"])
            self.assertIn("A/S proof queue as empty", loop_prompt["next_loop"][0])
            self.assertIn(
                "as-live-ready-evidence-loop",
                " ".join(loop_prompt["verification_commands"]),
            )
            self.assertIn(
                "as-4x-proof-path",
                " ".join(loop_prompt["verification_commands"]),
            )
            self.assertEqual(
                payload["codex_work_order"]["proof_state"]["gateway_status"],
                "NO_LIVE_READY_INTENT",
            )
            self.assertEqual(
                payload["codex_work_order"]["proof_state"]["proof_queue_count"],
                0,
            )
            self.assertEqual(
                payload["codex_work_order"]["proof_state"]["proof_queue_empty_reason"][
                    "primary_category"
                ],
                "lane_board",
            )
            self.assertEqual(
                payload["codex_work_order"]["proof_state"]["next_evidence_actions"][0][
                    "action_id"
                ],
                "collect_exact_tp_or_live_grade_harvest_evidence",
            )
            report_text = report.read_text()
            self.assertIn("Proof queue count: `0`", report_text)
            self.assertIn("Gateway status: `NO_LIVE_READY_INTENT`", report_text)
            self.assertIn("Proof queue empty reason", report_text)
            self.assertIn("Next evidence actions", report_text)

    def test_rejected_proof_action_requires_progress_beyond_current_queue_watermark(self) -> None:
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
                        status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                    )
                ],
            )
            _write_as_proof_artifacts(root)
            proof_path = root / "as_proof_pack_queue.json"
            proof_payload = json.loads(proof_path.read_text())
            proof_payload["summary"]["queue_count"] = 1
            proof_payload["summary"]["proof_ready_count"] = 1
            proof_payload["queue"] = [
                {
                    "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                    "proof_classification": "EVIDENCE_GAP",
                    "can_enter_proof_pack": True,
                    "can_create_live_permission": False,
                }
            ]
            proof_path.write_text(
                json.dumps(proof_payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            payload = json.loads(output.read_text())
            state = payload["loop_engineering_prompt"]["current_state"]
            action = next(
                item
                for item in payload["next_evidence_actions"]
                if item["action_id"]
                == "collect_exact_tp_or_live_grade_harvest_evidence"
            )
            evaluations = {
                row["field"]: row
                for row in action["success_condition_evaluation"]["checks"]
            }
            self.assertEqual(state["proof_queue_count"], 1)
            self.assertEqual(action["success_condition_evaluation"]["status"], "NOT_MET")
            self.assertEqual(evaluations["proof_queue_count"]["expected"], 1)
            self.assertFalse(evaluations["proof_queue_count"]["passed"])

            progressed = _evaluate_success_condition(
                action["success_condition"],
                {**state, "proof_queue_count": 2},
            )
            self.assertEqual(progressed["status"], "MET")

            portfolio_action = next(
                item
                for item in payload["next_evidence_actions"]
                if item["action_id"] == "refresh_portfolio_4x_path_planner"
            )
            portfolio_checks = {
                row["field"]: row
                for row in portfolio_action["success_condition_evaluation"]["checks"]
            }
            self.assertEqual(state["proof_ready_count"], 1)
            self.assertEqual(
                portfolio_action["success_condition_evaluation"]["status"],
                "NOT_MET",
            )
            self.assertEqual(portfolio_checks["proof_ready_count"]["expected"], 1)
            self.assertFalse(portfolio_checks["proof_ready_count"]["passed"])
            portfolio_progressed = _evaluate_success_condition(
                portfolio_action["success_condition"],
                {**state, "proof_ready_count": 2},
            )
            self.assertEqual(portfolio_progressed["status"], "MET")
            malformed_portfolio_clearance = _evaluate_success_condition(
                portfolio_action["success_condition"],
                {
                    **state,
                    "portfolio_status": "ERROR",
                    "portfolio_can_create_live_permission": 1,
                    "portfolio_can_reach_4x_now": 1.0,
                },
            )
            self.assertEqual(malformed_portfolio_clearance["status"], "NOT_MET")
            valid_portfolio_clearance = _evaluate_success_condition(
                portfolio_action["success_condition"],
                {**state, "portfolio_status": "LIVE_READY_PORTFOLIO"},
            )
            self.assertEqual(valid_portfolio_clearance["status"], "MET")

            valid_first_output = output.read_text()
            tampered_variants: list[tuple[str, dict[str, object], str | None]] = []

            missing_status_value = json.loads(valid_first_output)
            tampered_portfolio = next(
                item
                for item in missing_status_value["next_evidence_actions"]
                if item["action_id"] == "refresh_portfolio_4x_path_planner"
            )
            next(
                check
                for check in tampered_portfolio["success_condition"]["checks"]
                if check["field"] == "portfolio_status"
            ).pop("value")
            tampered_variants.append(
                ("missing_non_count_value", missing_status_value, "refresh_portfolio_4x_path_planner")
            )

            lowered_count_watermark = json.loads(valid_first_output)
            tampered_collection = next(
                item
                for item in lowered_count_watermark["next_evidence_actions"]
                if item["action_id"]
                == "collect_exact_tp_or_live_grade_harvest_evidence"
            )
            next(
                check
                for check in tampered_collection["success_condition"]["checks"]
                if check["field"] == "proof_queue_count"
            )["value"] = 0
            tampered_variants.append(
                (
                    "lowered_count_watermark",
                    lowered_count_watermark,
                    "collect_exact_tp_or_live_grade_harvest_evidence",
                )
            )

            unknown_action = json.loads(valid_first_output)
            unknown_action["next_evidence_actions"].append(
                {
                    "action_id": "unknown_forged_action",
                    "category": "gateway_issue",
                    "progress_watermark_contract": "evidence_action_progress_v1",
                    "success_condition": {
                        "description": "forged",
                        "schema_version": "success_condition_v1",
                        "verification_scope": "forged",
                        "mode": "all",
                        "checks": [
                            {"field": "proof_routing_allowed", "operator": "is_true"}
                        ],
                    },
                    "success_condition_evaluation": {
                        "status": "NOT_MET",
                        "passed": False,
                        "mode": "all",
                        "checks": [
                            {
                                "field": "proof_routing_allowed",
                                "operator": "is_true",
                                "expected": None,
                                "actual": False,
                                "passed": False,
                            }
                        ],
                    },
                }
            )
            tampered_variants.append(("unknown_action", unknown_action, None))

            for label, tampered, reset_action_id in tampered_variants:
                with self.subTest(tamper=label):
                    output.write_text(
                        json.dumps(tampered, ensure_ascii=False, indent=2, sort_keys=True)
                        + "\n",
                        encoding="utf-8",
                    )
                    TraderRepairOrchestrator(
                        support_bot_path=support,
                        output_path=output,
                        report_path=report,
                    ).run()
                    hardened = json.loads(output.read_text())
                    self.assertNotIn("evidence_action_progress", hardened)
                    self.assertNotIn(
                        "unknown_forged_action",
                        [
                            item["action_id"]
                            for item in hardened.get("next_evidence_actions", [])
                        ],
                    )
                    if reset_action_id:
                        hardened_action = next(
                            item
                            for item in hardened["next_evidence_actions"]
                            if item["action_id"] == reset_action_id
                        )
                        self.assertEqual(
                            hardened_action["progress_watermark_source"],
                            "CURRENT_REFRESH",
                        )
                    output.write_text(valid_first_output, encoding="utf-8")

            proof_payload["summary"]["queue_count"] = 2
            proof_payload["summary"]["proof_ready_count"] = 2
            proof_payload["queue"].append(
                {
                    "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                    "proof_classification": "EVIDENCE_GAP",
                    "can_enter_proof_pack": True,
                    "can_create_live_permission": False,
                }
            )
            proof_path.write_text(
                json.dumps(proof_payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            advanced_payload = json.loads(output.read_text())
            advanced_action = next(
                item
                for item in advanced_payload["next_evidence_actions"]
                if item["action_id"]
                == "collect_exact_tp_or_live_grade_harvest_evidence"
            )
            advanced_queue_check = next(
                row
                for row in advanced_action["success_condition_evaluation"]["checks"]
                if row["field"] == "proof_queue_count"
            )
            self.assertEqual(advanced_action["progress_watermark_source"], "PREVIOUS_REFRESH")
            self.assertEqual(advanced_queue_check["expected"], 1)
            self.assertEqual(advanced_queue_check["actual"], 2)
            self.assertTrue(advanced_queue_check["passed"])
            self.assertEqual(
                advanced_action["success_condition_evaluation"]["status"],
                "MET",
            )
            progress_receipt = next(
                item
                for item in advanced_payload["evidence_action_progress"]
                if item["action_id"]
                == "collect_exact_tp_or_live_grade_harvest_evidence"
            )
            self.assertEqual(
                progress_receipt["success_condition_evaluation"]["status"],
                "MET",
            )

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            reset_payload = json.loads(output.read_text())
            reset_action = next(
                item
                for item in reset_payload["next_evidence_actions"]
                if item["action_id"]
                == "collect_exact_tp_or_live_grade_harvest_evidence"
            )
            reset_queue_check = next(
                row
                for row in reset_action["success_condition_evaluation"]["checks"]
                if row["field"] == "proof_queue_count"
            )
            self.assertEqual(reset_action["progress_watermark_source"], "CURRENT_REFRESH")
            self.assertEqual(reset_queue_check["expected"], 2)
            self.assertEqual(reset_queue_check["actual"], 2)
            self.assertFalse(reset_queue_check["passed"])
            self.assertEqual(
                reset_action["success_condition_evaluation"]["status"],
                "NOT_MET",
            )

            proof_payload["summary"]["can_create_live_permission_count"] = 1
            proof_path.write_text(
                json.dumps(proof_payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            cleared_payload = json.loads(output.read_text())
            self.assertNotIn("next_evidence_actions", cleared_payload)
            cleared_progress = next(
                item
                for item in cleared_payload["evidence_action_progress"]
                if item["action_id"]
                == "collect_exact_tp_or_live_grade_harvest_evidence"
            )
            self.assertEqual(
                cleared_progress["success_condition_evaluation"]["status"],
                "MET",
            )

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()
            one_cycle_later_payload = json.loads(output.read_text())
            self.assertNotIn("next_evidence_actions", one_cycle_later_payload)
            self.assertNotIn("evidence_action_progress", one_cycle_later_payload)

    def test_evidence_watermark_does_not_rebaseline_after_count_regression(self) -> None:
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
                        status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                    )
                ],
            )
            _write_as_proof_artifacts(root)
            proof_path = root / "as_proof_pack_queue.json"
            proof_payload = json.loads(proof_path.read_text())

            def refresh_counts(
                queue_count: int,
                proof_ready_count: int,
                rejected_count: int,
            ) -> dict[str, object]:
                proof_payload["summary"]["queue_count"] = queue_count
                proof_payload["summary"]["proof_ready_count"] = proof_ready_count
                proof_payload["summary"]["rejected_candidate_count"] = rejected_count
                proof_payload["queue"] = [
                    {"lane_id": f"lane-{index}"} for index in range(queue_count)
                ]
                proof_path.write_text(
                    json.dumps(
                        proof_payload,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                TraderRepairOrchestrator(
                    support_bot_path=support,
                    output_path=output,
                    report_path=report,
                ).run()
                return json.loads(output.read_text())

            def collection_action(payload: dict[str, object]) -> dict[str, object]:
                return next(
                    item
                    for item in payload["next_evidence_actions"]
                    if item["action_id"]
                    == "collect_exact_tp_or_live_grade_harvest_evidence"
                )

            baseline = collection_action(refresh_counts(1, 1, 4))
            self.assertEqual(baseline["progress_watermark_source"], "CURRENT_REFRESH")

            regressed = collection_action(refresh_counts(0, 0, 5))
            regressed_checks = {
                row["field"]: row
                for row in regressed["success_condition_evaluation"]["checks"]
            }
            self.assertEqual(regressed["progress_watermark_source"], "PREVIOUS_REFRESH")
            self.assertEqual(regressed_checks["proof_queue_count"]["expected"], 1)
            self.assertEqual(
                regressed_checks["rejected_proof_candidate_count"]["expected"],
                4,
            )
            self.assertEqual(regressed["success_condition_evaluation"]["status"], "NOT_MET")

            still_regressed_payload = refresh_counts(0, 0, 5)
            still_regressed = collection_action(still_regressed_payload)
            self.assertEqual(
                still_regressed["progress_watermark_source"],
                "PREVIOUS_REFRESH",
            )
            self.assertEqual(
                still_regressed["progress_watermark_origin"]["condition_state"][
                    "proof_queue_count"
                ],
                1,
            )
            self.assertNotIn("evidence_action_progress", still_regressed_payload)

            merely_recovered_payload = refresh_counts(1, 1, 4)
            merely_recovered = collection_action(merely_recovered_payload)
            recovered_checks = {
                row["field"]: row
                for row in merely_recovered["success_condition_evaluation"]["checks"]
            }
            self.assertEqual(
                merely_recovered["progress_watermark_source"],
                "PREVIOUS_REFRESH",
            )
            self.assertFalse(recovered_checks["proof_queue_count"]["passed"])
            self.assertFalse(
                recovered_checks["rejected_proof_candidate_count"]["passed"]
            )
            self.assertEqual(
                merely_recovered["success_condition_evaluation"]["status"],
                "NOT_MET",
            )
            self.assertNotIn("evidence_action_progress", merely_recovered_payload)

            genuinely_advanced_payload = refresh_counts(2, 2, 3)
            genuinely_advanced = collection_action(genuinely_advanced_payload)
            self.assertEqual(
                genuinely_advanced["success_condition_evaluation"]["status"],
                "MET",
            )
            self.assertIn("evidence_action_progress", genuinely_advanced_payload)

    def test_proof_queue_empty_reason_marks_stale_artifact_and_lowers_primary_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            broker = root / "broker_snapshot.json"
            base = datetime(2026, 7, 7, 0, 0, tzinfo=timezone.utc)
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_FRONTIER_LANE_BLOCKER",
                        priority="P1",
                        status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                    )
                ],
                generated_at_utc=(base + timedelta(seconds=10)).isoformat(),
            )
            broker.write_text(
                json.dumps({"fetched_at_utc": base.isoformat()}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_as_proof_artifacts(
                root,
                proof_generated_at_utc=(base + timedelta(seconds=20)).isoformat(),
                board_generated_at_utc=(base + timedelta(seconds=20)).isoformat(),
                planner_generated_at_utc=(base + timedelta(seconds=25)).isoformat(),
                live_order_generated_at_utc=(base + timedelta(seconds=700)).isoformat(),
            )

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                broker_snapshot_path=broker,
            ).run()

            payload = json.loads(output.read_text())
            reason = payload["proof_queue_empty_reason"]
            by_category = {item["category"]: item for item in reason["categories"]}
            self.assertEqual(by_category["lane_board"]["freshness"]["status"], "STALE")
            self.assertEqual(by_category["gateway_issue"]["freshness"]["status"], "FRESH")
            self.assertEqual(reason["primary_category"], "gateway_issue")
            self.assertGreater(
                by_category["lane_board"]["freshness"]["freshness_age_seconds"],
                by_category["lane_board"]["freshness"]["freshness_max_age_seconds"],
            )
            self.assertEqual(
                by_category["gateway_issue"]["freshness"]["freshness_reference_timestamp"],
                (base + timedelta(seconds=700)).isoformat(),
            )

    def test_proof_queue_empty_reason_excludes_contradicted_artifact_from_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            broker = root / "broker_snapshot.json"
            base = datetime(2026, 7, 7, 1, 0, tzinfo=timezone.utc)
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_FRONTIER_LANE_BLOCKER",
                        priority="P1",
                        status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                    )
                ],
                generated_at_utc=(base + timedelta(seconds=60)).isoformat(),
            )
            broker.write_text(
                json.dumps({"fetched_at_utc": base.isoformat()}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_as_proof_artifacts(
                root,
                proof_generated_at_utc=(base + timedelta(seconds=70)).isoformat(),
                board_generated_at_utc=(base + timedelta(seconds=10)).isoformat(),
                planner_generated_at_utc=(base + timedelta(seconds=75)).isoformat(),
                live_order_generated_at_utc=(base + timedelta(seconds=80)).isoformat(),
            )

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                broker_snapshot_path=broker,
            ).run()

            payload = json.loads(output.read_text())
            reason = payload["proof_queue_empty_reason"]
            by_category = {item["category"]: item for item in reason["categories"]}
            self.assertEqual(by_category["lane_board"]["freshness"]["status"], "CONTRADICTED")
            self.assertNotEqual(reason["primary_category"], "lane_board")
            self.assertEqual(reason["primary_category"], "gateway_issue")
            self.assertIn(
                "predates required upstream evidence",
                by_category["lane_board"]["freshness"]["freshness_reason"],
            )

    def test_no_send_gateway_older_than_blocked_proof_packet_is_not_contradicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            broker = root / "broker_snapshot.json"
            base = datetime(2026, 7, 7, 2, 0, tzinfo=timezone.utc)
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_FRONTIER_LANE_BLOCKER",
                        priority="P1",
                        status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                    )
                ],
                generated_at_utc=(base + timedelta(seconds=300)).isoformat(),
            )
            broker.write_text(
                json.dumps({"fetched_at_utc": (base + timedelta(seconds=240)).isoformat()}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_as_proof_artifacts(
                root,
                proof_generated_at_utc=(base + timedelta(seconds=300)).isoformat(),
                board_generated_at_utc=(base + timedelta(seconds=305)).isoformat(),
                planner_generated_at_utc=(base + timedelta(seconds=310)).isoformat(),
                live_order_generated_at_utc=(base + timedelta(seconds=10)).isoformat(),
            )

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                broker_snapshot_path=broker,
            ).run()

            payload = json.loads(output.read_text())
            reason = payload["proof_queue_empty_reason"]
            by_category = {item["category"]: item for item in reason["categories"]}
            gateway_freshness = by_category["gateway_issue"]["freshness"]
            self.assertEqual(gateway_freshness["status"], "FRESH")
            self.assertTrue(gateway_freshness["dependency_lag_exempted"])
            self.assertIn("no-send gateway status is aligned", gateway_freshness["freshness_reason"])

    def test_no_send_gateway_still_contradicted_when_proof_ready_candidate_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            broker = root / "broker_snapshot.json"
            base = datetime(2026, 7, 7, 3, 0, tzinfo=timezone.utc)
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_FRONTIER_LANE_BLOCKER",
                        priority="P1",
                        status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                    )
                ],
                generated_at_utc=(base + timedelta(seconds=300)).isoformat(),
            )
            broker.write_text(
                json.dumps({"fetched_at_utc": (base + timedelta(seconds=240)).isoformat()}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_as_proof_artifacts(
                root,
                proof_generated_at_utc=(base + timedelta(seconds=300)).isoformat(),
                board_generated_at_utc=(base + timedelta(seconds=305)).isoformat(),
                planner_generated_at_utc=(base + timedelta(seconds=310)).isoformat(),
                live_order_generated_at_utc=(base + timedelta(seconds=10)).isoformat(),
            )
            proof_path = root / "as_proof_pack_queue.json"
            proof = json.loads(proof_path.read_text())
            proof["summary"]["queue_count"] = 1
            proof["summary"]["proof_ready_count"] = 1
            proof["queue"] = [{"lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION"}]
            proof_path.write_text(json.dumps(proof, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            planner_path = root / "portfolio_4x_path_planner.json"
            planner = json.loads(planner_path.read_text())
            planner["summary"]["proof_ready_candidates"] = 1
            planner_path.write_text(json.dumps(planner, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                broker_snapshot_path=broker,
            ).run()

            payload = json.loads(output.read_text())
            reason = payload["proof_queue_empty_reason"]
            by_category = {item["category"]: item for item in reason["categories"]}
            gateway_freshness = by_category["gateway_issue"]["freshness"]
            self.assertEqual(gateway_freshness["status"], "CONTRADICTED")
            self.assertNotIn("dependency_lag_exempted", gateway_freshness)

    def test_temp_orchestrator_without_as_artifacts_does_not_read_repo_runtime_artifacts(self) -> None:
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
                        status=TP_PROGRESS_LIVE_EVIDENCE_WAIT_STATUS,
                    )
                ],
            )

            TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            payload = json.loads(output.read_text())
            state = payload["loop_engineering_prompt"]["current_state"]
            self.assertNotIn("proof_queue_count", state)
            self.assertNotIn("rejected_proof_candidate_count", state)
            self.assertEqual(payload["codex_work_order"]["proof_state"], {})

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
            stdout_payload = json.loads(stdout.getvalue())
            self.assertEqual(stdout_payload["status"], STATUS_APPROVAL_REQUIRED)
            self.assertEqual(stdout_payload["repair_request_count"], 1)
            self.assertEqual(stdout_payload["actionable_request_count"], 0)
            self.assertEqual(stdout_payload["approval_required_request_count"], 1)
            self.assertEqual(stdout_payload["waiting_request_count"], 0)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["repair_request_count"], 1)
            self.assertEqual(payload["actionable_request_count"], 0)
            self.assertEqual(payload["approval_required_request_count"], 1)
            self.assertEqual(payload["waiting_request_count"], 0)
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
            self.assertIn("co_blocked_by=SPREAD_TOO_WIDE", loop_prompt["prompt_text"])
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

    def test_guardian_operator_review_frontier_is_approval_bound_not_codex_work(self) -> None:
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
                        status="READY_FOR_CODE_OR_EVIDENCE_REPAIR",
                        source_findings=["GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"],
                        evidence_summary={
                            "code": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                            "co_blocker_codes": [
                                "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                                "EXHAUSTION_RANGE_CHASE",
                            ],
                            "example_lane_ids": [
                                "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"
                            ],
                        },
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="frontier guardian",
            ).run()

            self.assertEqual(summary.status, STATUS_APPROVAL_REQUIRED)
            payload = json.loads(output.read_text())
            frontier = payload["queue"][0]
            self.assertEqual(frontier["automation_status"], "WAITING_FOR_OPERATOR_APPROVAL")
            self.assertTrue(frontier["requires_explicit_operator_approval"])
            self.assertEqual(frontier["approval_dependency"]["code"], "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED")
            self.assertIn(
                "guardian_operator_review",
                frontier["automation_contract"]["requires_explicit_operator_approval_for"],
            )
            self.assertEqual(payload["selected_request"], {})
            self.assertEqual(payload["actionable_requests"], [])
            self.assertEqual(
                payload["queue_summary"]["approval_required_request_codes"],
                ["REPAIR_FRONTIER_LANE_BLOCKER"],
            )
            self.assertEqual(payload["codex_work_order"]["status"], "NO_ACTIONABLE_CODEX_WORK")
            self.assertIn("approval-bound", payload["loop_engineering_prompt"]["current_hypothesis"])

    def test_guardian_co_blocker_does_not_hide_bidask_evidence_work(self) -> None:
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
                        status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                        source_findings=["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                        evidence_summary={
                            "code": "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                            "co_blocker_codes": [
                                "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                            ],
                            "example_lane_ids": [
                                "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"
                            ],
                        },
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="frontier bidask",
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            frontier = payload["queue"][0]
            self.assertEqual(frontier["automation_status"], "READY_FOR_CODEX_IMPLEMENTATION")
            self.assertFalse(frontier["requires_explicit_operator_approval"])
            self.assertEqual(payload["selected_request"]["code"], "REPAIR_FRONTIER_LANE_BLOCKER")
            self.assertEqual(payload["approval_required_requests"], [])

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

    def test_strategy_profile_frontier_wait_is_not_codex_implementation_work(self) -> None:
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
                        status=FRONTIER_STRATEGY_PROFILE_EVIDENCE_WAIT_STATUS,
                        evidence_summary={
                            "code": "MATRIX_REPAIR_REJECT_CONTEXT",
                            "matrix_repair_profile_status_counts": {
                                "BLOCK_UNTIL_NEW_EVIDENCE": 2,
                            },
                            "example_lane_ids": [
                                "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
                            ],
                        },
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="matrix profile",
            ).run()

            self.assertEqual(summary.status, STATUS_BLOCKED)
            payload = json.loads(output.read_text())
            frontier = payload["queue"][0]
            self.assertEqual(frontier["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertEqual(payload["selected_request"], {})
            self.assertEqual(payload["actionable_requests"], [])
            self.assertEqual(payload["codex_work_order"]["status"], "NO_ACTIONABLE_CODEX_WORK")
            self.assertIn(
                "REPAIR_FRONTIER_LANE_BLOCKER",
                payload["queue_summary"]["waiting_request_codes"],
            )

    def test_proof_sample_frontier_wait_is_not_codex_implementation_work(self) -> None:
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
                        status=FRONTIER_PROOF_EVIDENCE_WAIT_STATUS,
                        evidence_summary={
                            "code": "ACTIVE_DAY_FLOOR_NOT_MET",
                            "proof_evidence_wait": {
                                "requires_material_input_change": True,
                                "same_input_rerun_is_progress": False,
                                "code_or_gate_change_can_clear": False,
                            },
                            "example_lane_ids": [
                                "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"
                            ],
                        },
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="active-day proof floor",
            ).run()

            self.assertEqual(summary.status, STATUS_BLOCKED)
            payload = json.loads(output.read_text())
            frontier = payload["queue"][0]
            self.assertEqual(frontier["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertEqual(payload["selected_request"], {})
            self.assertEqual(payload["actionable_requests"], [])
            self.assertEqual(payload["codex_work_order"]["status"], "NO_ACTIONABLE_CODEX_WORK")
            self.assertIn(
                "REPAIR_FRONTIER_LANE_BLOCKER",
                payload["queue_summary"]["waiting_request_codes"],
            )

    def test_stale_actionable_status_cannot_promote_proof_sample_frontier(self) -> None:
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
                        status="READY_FOR_CODE_OR_EVIDENCE_REPAIR",
                        source_findings=["ACTIVE_DAY_FLOOR_NOT_MET"],
                        evidence_summary={
                            "code": "ACTIVE_DAY_FLOOR_NOT_MET",
                            "example_lane_ids": [
                                "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"
                            ],
                        },
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="active-day proof floor",
            ).run()

            self.assertEqual(summary.status, STATUS_BLOCKED)
            payload = json.loads(output.read_text())
            self.assertEqual(
                payload["queue"][0]["automation_status"],
                "WAITING_FOR_LIVE_EVIDENCE_WINDOW",
            )
            self.assertEqual(payload["selected_request"], {})
            self.assertEqual(payload["actionable_requests"], [])

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
            report_text = report.read_text()
            self.assertIn("Repair status", report_text)
            self.assertIn("BIDASK_REPLAY_WAITING_FOR_FORECAST_SAMPLE_COVERAGE", report_text)
            self.assertIn("COLLECT_BIDASK_REPLAY_EVIDENCE", report_text)

    def test_tp_progress_live_evidence_waiting_status_is_non_actionable(self) -> None:
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
                        status=TP_PROGRESS_LIVE_EVIDENCE_WAIT_STATUS,
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
            ).run()

            payload = json.loads(output.read_text())
            self.assertEqual(summary.status, STATUS_BLOCKED)
            self.assertEqual(payload["codex_work_order"]["status"], "NO_ACTIONABLE_CODEX_WORK")
            waiting = payload["queue"][0]
            self.assertEqual(waiting["code"], "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY")
            self.assertEqual(waiting["repair_status"], TP_PROGRESS_LIVE_EVIDENCE_WAIT_STATUS)
            self.assertEqual(waiting["automation_status"], "WAITING_FOR_LIVE_EVIDENCE_WINDOW")
            self.assertEqual(
                payload["queue_summary"]["waiting_request_codes"],
                ["REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY"],
            )


def _write_support(
    path: Path,
    requests: list[dict[str, object]],
    *,
    generated_at_utc: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "status": "SUPPORT_BLOCKED",
        "repair_requests": requests,
    }
    if generated_at_utc:
        payload["generated_at_utc"] = generated_at_utc
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_support_with_active_lane(
    path: Path,
    *,
    requests: list[dict[str, object]],
    generated_at_utc: str = "2026-07-10T00:20:00+00:00",
    local_tp_trades: int = 2,
    next_action: str | None = None,
) -> None:
    lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
    action = next_action or (
        "Use the latest active_opportunity_board rerank. Collect exact local "
        "TAKE_PROFIT_ORDER proof for EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT|TAKE_PROFIT_ORDER; "
        "require positive expectancy, zero TP losses, and positive Wilson-stressed "
        "expectancy before reranking. Do not send."
    )
    blockers = [
        "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
        "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
        "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
    ]
    payload = {
        "generated_at_utc": generated_at_utc,
        "status": "SUPPORT_BLOCKED",
        "repair_requests": requests,
        "entry_readiness": {
            "live_ready_lanes": 0,
            "shortest_live_ready_path": {
                "lane_id": lane_id,
                "pair": "EUR_USD",
                "side": "LONG",
                "method": "BREAKOUT_FAILURE",
                "order_type": "LIMIT",
                "status": "ACTIVE_PATH_BLOCKED_NEAR_READY_LANE",
                "current_status": "DRY_RUN_BLOCKED",
                "selection_basis": "active_trader_contract",
                "blocker_codes": blockers,
                "blocker_groups": ["forecast_telemetry", "proof_gap", "global"],
                "evidence_needed": ["collect exact-vehicle TP proof"],
                "first_next_step": action,
                "active_path": {
                    "lane_id": lane_id,
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "method": "BREAKOUT_FAILURE",
                    "order_type": "LIMIT",
                    "status": "EVIDENCE_ACQUISITION",
                    "contract_status": "ACTIVE_PATH_SELECTED_REPLAY_PASSED_STILL_BLOCKED",
                    "generated_at_utc": generated_at_utc,
                    "live_permission": False,
                    "blocker_codes": blockers,
                    "next_action": action,
                    "local_tp_proof": {
                        "capture_take_profit_scope_key": (
                            "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT|TAKE_PROFIT_ORDER"
                        ),
                        "capture_take_profit_trades": local_tp_trades,
                        "capture_take_profit_wins": local_tp_trades,
                        "capture_take_profit_losses": 0,
                        "capture_take_profit_expectancy_jpy": 1146.7898,
                        "capture_take_profit_proof_floor": 20,
                        "generated_at_utc": generated_at_utc,
                    },
                },
            },
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_as_proof_artifacts(
    root: Path,
    *,
    proof_generated_at_utc: str = "2026-07-06T18:30:26Z",
    board_generated_at_utc: str = "2026-07-06T18:30:26Z",
    planner_generated_at_utc: str = "2026-07-06T18:30:29Z",
    live_order_generated_at_utc: str | None = None,
) -> None:
    (root / "as_proof_pack_queue.json").write_text(
        json.dumps(
            {
                "generated_at_utc": proof_generated_at_utc,
                "summary": {
                    "as_live_ready_path_exists": False,
                    "can_create_live_permission_count": 0,
                    "proof_ready_count": 0,
                    "queue_count": 0,
                    "rejected_candidate_count": 4,
                    "remaining_p0_rows": 4,
                },
                "queue": [],
                "rejected_candidates": [
                    {
                        "lane_id": "range_trader:GBP_USD:LONG:RANGE_ROTATION",
                        "rejection_reasons": [
                            "spread_included_bidask_replay_negative_for_exact_lane"
                        ],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "as_lane_candidate_board.json").write_text(
        json.dumps(
            {
                "generated_at_utc": board_generated_at_utc,
                "as_live_ready_path_exists": False,
                "live_ready_lanes": 0,
                "normal_routing_status": "BLOCKED",
                "routing_allowed": False,
                "exact_blocker_preventing_live_ready": {
                    "as_live_ready_stays_zero": True,
                    "global_blockers": [
                        "NEGATIVE_EXPECTANCY_ACTIVE",
                        "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
                        "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                    ],
                    "normal_routing_must_remain_blocked": True,
                    "p0_rows": [
                        "NEGATIVE_EXPECTANCY_ACTIVE",
                        "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
                    ],
                    "primary": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                },
                "firepower_board_summary": {
                    "can_create_live_permission_rows": 0,
                    "can_enter_proof_pack_rows": 2,
                    "candidate_rows_after_hard_exclusions": 51,
                    "rejected_candidate_count": 4,
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "portfolio_4x_path_planner.json").write_text(
        json.dumps(
            {
                "generated_at_utc": planner_generated_at_utc,
                "can_reach_4x_now": False,
                "live_ready_lanes": 0,
                "normal_routing_status": "BLOCKED",
                "portfolio_status": "NO_LIVE_READY_PORTFOLIO",
                "summary": {
                    "can_create_live_permission": False,
                    "planner_rejected_candidates": 46,
                    "proof_ready_candidates": 0,
                    "standalone_live_ready_candidates": 0,
                    "standalone_math_candidates_meeting_required_return": 0,
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "live_order_request.json").write_text(
        json.dumps(
            {
                **({"generated_at_utc": live_order_generated_at_utc} if live_order_generated_at_utc else {}),
                "status": "NO_LIVE_READY_INTENT",
                "risk_issues": [{"code": "NO_LIVE_READY_LANES"}],
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
                "co_blocker_codes": ["SPREAD_TOO_WIDE", "TARGET_TOO_THIN_FOR_SPREAD"],
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
