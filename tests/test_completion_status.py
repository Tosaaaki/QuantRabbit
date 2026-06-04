from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.completion import CompletionAuditor


class CompletionAuditorTest(unittest.TestCase):
    def test_reports_root_blockers_and_next_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _blocked_fixture(root)

            summary = CompletionAuditor(
                broker_snapshot_path=files["broker"],
                order_intents_path=files["intents"],
                target_state_path=files["target"],
                coverage_path=files["coverage"],
                replay_backtest_path=files["replay"],
                execution_replay_path=files["execution"],
                dry_run_certification_path=files["certification"],
                live_order_path=files["live_order"],
                output_path=root / "completion.json",
                report_path=root / "completion.md",
            ).run()

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(summary.live_ready_lanes, 0)
            payload = json.loads((root / "completion.json").read_text())
            codes = {item["code"] for item in payload["blockers"]}
            self.assertIn("BROKER_EXPOSURE_OPEN", codes)
            self.assertIn("NO_LIVE_READY_INTENTS", codes)
            self.assertIn("LIVE_SEND_ARTIFACT_PRESENT", codes)
            self.assertIn("Completion Contract", (root / "completion.md").read_text())

    def test_complete_when_all_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _complete_fixture(root)

            summary = CompletionAuditor(
                broker_snapshot_path=files["broker"],
                order_intents_path=files["intents"],
                target_state_path=files["target"],
                coverage_path=files["coverage"],
                replay_backtest_path=files["replay"],
                execution_replay_path=files["execution"],
                dry_run_certification_path=files["certification"],
                live_order_path=files["live_order"],
                output_path=root / "completion.json",
                report_path=root / "completion.md",
            ).run()

            self.assertEqual(summary.status, "COMPLETE")
            self.assertEqual(summary.blockers, 0)

    def test_protected_trader_exposure_is_not_reported_as_fresh_entry_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _complete_fixture(root)
            files["broker"].write_text(
                json.dumps(
                    {
                        "positions": [
                            {
                                "trade_id": "1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "owner": "trader",
                                "take_profit": 1.2,
                                "stop_loss": 1.1,
                            }
                        ],
                        "orders": [],
                    }
                )
            )

            summary = CompletionAuditor(
                broker_snapshot_path=files["broker"],
                order_intents_path=files["intents"],
                target_state_path=files["target"],
                coverage_path=files["coverage"],
                replay_backtest_path=files["replay"],
                execution_replay_path=files["execution"],
                dry_run_certification_path=files["certification"],
                live_order_path=files["live_order"],
                output_path=root / "completion.json",
                report_path=root / "completion.md",
            ).run()

            self.assertEqual(summary.status, "COMPLETE")
            payload = json.loads((root / "completion.json").read_text())
            self.assertNotIn("BROKER_EXPOSURE_OPEN", {item["code"] for item in payload["blockers"]})

    def test_trader_pending_entry_does_not_blanket_block_open_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _complete_fixture(root)
            files["broker"].write_text(
                json.dumps(
                    {
                        "positions": [],
                        "orders": [
                            {
                                "order_id": "pending-1",
                                "pair": "GBP_USD",
                                "order_type": "LIMIT",
                                "trade_id": None,
                                "owner": "trader",
                                "state": "PENDING",
                            }
                        ],
                    }
                )
            )
            files["target"].write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 22000.0,
                        "remaining_risk_budget_jpy": 500.0,
                    }
                )
            )

            summary = CompletionAuditor(
                broker_snapshot_path=files["broker"],
                order_intents_path=files["intents"],
                target_state_path=files["target"],
                coverage_path=files["coverage"],
                replay_backtest_path=files["replay"],
                execution_replay_path=files["execution"],
                dry_run_certification_path=files["certification"],
                live_order_path=files["live_order"],
                output_path=root / "completion.json",
                report_path=root / "completion.md",
            ).run()

            self.assertEqual(summary.status, "COMPLETE")
            payload = json.loads((root / "completion.json").read_text())
            blocker_codes = {item["code"] for item in payload["blockers"]}
            action_codes = {item["code"] for item in payload["next_actions"]}
            self.assertNotIn("PENDING_ENTRY_OPEN", blocker_codes)
            self.assertNotIn("PENDING_ENTRY_BLOCKED", blocker_codes)
            self.assertIn("BASKET_VALIDATE_PENDING_ENTRIES", action_codes)
            self.assertNotIn("RESOLVE_PENDING_ENTRIES", action_codes)

    def test_trader_pending_entry_blocks_after_target_reached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _complete_fixture(root)
            files["broker"].write_text(
                json.dumps(
                    {
                        "positions": [],
                        "orders": [
                            {
                                "order_id": "pending-1",
                                "pair": "GBP_USD",
                                "order_type": "LIMIT",
                                "trade_id": None,
                                "owner": "trader",
                                "state": "PENDING",
                            }
                        ],
                    }
                )
            )

            summary = CompletionAuditor(
                broker_snapshot_path=files["broker"],
                order_intents_path=files["intents"],
                target_state_path=files["target"],
                coverage_path=files["coverage"],
                replay_backtest_path=files["replay"],
                execution_replay_path=files["execution"],
                dry_run_certification_path=files["certification"],
                live_order_path=files["live_order"],
                output_path=root / "completion.json",
                report_path=root / "completion.md",
            ).run()

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "completion.json").read_text())
            blocker_codes = {item["code"] for item in payload["blockers"]}
            action_codes = {item["code"] for item in payload["next_actions"]}
            self.assertIn("PENDING_ENTRY_BLOCKED", blocker_codes)
            self.assertIn("RESOLVE_PENDING_ENTRIES", action_codes)

    def test_stale_coverage_does_not_override_current_live_ready_intents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _complete_fixture(root)
            files["target"].write_text(
                json.dumps(
                    {
                        "status": "PURSUE_TARGET",
                        "remaining_target_jpy": 22000.0,
                        "remaining_risk_budget_jpy": 500.0,
                    }
                )
            )
            files["intents"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-05-01T06:00:00+00:00",
                        "results": [{"status": "LIVE_READY", "intent": {"pair": "EUR_USD"}}],
                    }
                )
            )
            files["coverage"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-05-01T05:59:00+00:00",
                        "status": "COVERAGE_GAP",
                        "blockers": ["no LIVE_READY lanes exist"],
                        "action_items": ["build receipts"],
                        "lanes": [
                            {
                                "lane_id": "old:EUR_USD:LONG:TREND_CONTINUATION",
                                "status": "DRY_RUN_BLOCKED",
                                "counts_live_ready": False,
                                "blockers": ["stale quote"],
                            }
                        ],
                    }
                )
            )

            summary = CompletionAuditor(
                broker_snapshot_path=files["broker"],
                order_intents_path=files["intents"],
                target_state_path=files["target"],
                coverage_path=files["coverage"],
                replay_backtest_path=files["replay"],
                execution_replay_path=files["execution"],
                dry_run_certification_path=files["certification"],
                live_order_path=files["live_order"],
                output_path=root / "completion.json",
                report_path=root / "completion.md",
            ).run()

            self.assertEqual(summary.live_ready_lanes, 1)
            payload = json.loads((root / "completion.json").read_text())
            blocker_codes = {item["code"] for item in payload["blockers"]}
            blocker_messages = {item["message"] for item in payload["blockers"]}
            action_codes = {item["code"] for item in payload["next_actions"]}
            self.assertIn("COVERAGE_STALE", blocker_codes)
            self.assertIn("RUN_COVERAGE_OPTIMIZATION", action_codes)
            self.assertNotIn("NO_LIVE_READY_INTENTS", blocker_codes)
            self.assertNotIn("no LIVE_READY lanes exist", blocker_messages)

    def test_stale_certification_blocks_completion_even_when_certified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _complete_fixture(root)
            files["intents"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-01T10:00:00+00:00",
                        "results": [{"status": "LIVE_READY", "intent": {"pair": "EUR_USD"}}],
                    }
                )
            )
            files["coverage"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-01T10:00:00+00:00",
                        "status": "TARGET_REACHED_PROTECT",
                        "blockers": [],
                        "action_items": [],
                    }
                )
            )
            files["execution"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-01T10:00:00+00:00",
                        "status": "TARGET_HIT",
                        "target_hit": True,
                        "blockers": [],
                    }
                )
            )
            files["live_order"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-01T10:00:00+00:00",
                        "sent": False,
                        "send_requested": False,
                    }
                )
            )
            files["certification"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-01T09:59:00+00:00",
                        "status": "CERTIFIED",
                        "blockers": [],
                    }
                )
            )

            summary = CompletionAuditor(
                broker_snapshot_path=files["broker"],
                order_intents_path=files["intents"],
                target_state_path=files["target"],
                coverage_path=files["coverage"],
                replay_backtest_path=files["replay"],
                execution_replay_path=files["execution"],
                dry_run_certification_path=files["certification"],
                live_order_path=files["live_order"],
                output_path=root / "completion.json",
                report_path=root / "completion.md",
            ).run()

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "completion.json").read_text())
            self.assertIn("CERTIFICATION_STALE", {item["code"] for item in payload["blockers"]})
            self.assertTrue(payload["certification"]["stale"])

    def test_fresh_close_recommendation_blocks_when_gate_b_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _complete_fixture(root)
            _write_protected_position_with_close_recommendation(root, files["broker"])
            prior_override = os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)
            try:
                summary = CompletionAuditor(
                    broker_snapshot_path=files["broker"],
                    order_intents_path=files["intents"],
                    target_state_path=files["target"],
                    coverage_path=files["coverage"],
                    replay_backtest_path=files["replay"],
                    execution_replay_path=files["execution"],
                    dry_run_certification_path=files["certification"],
                    live_order_path=files["live_order"],
                    output_path=root / "completion.json",
                    report_path=root / "completion.md",
                ).run()
            finally:
                if prior_override is not None:
                    os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = prior_override

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "completion.json").read_text())
            codes = {item["code"] for item in payload["blockers"]}
            self.assertIn("CLOSE_AUTHORIZATION_REQUIRED", codes)
            self.assertEqual(payload["close_recommendations"]["count"], 1)
            self.assertFalse(payload["close_recommendations"]["gate_b_authorized"])

    def test_fresh_close_recommendation_still_requires_close_receipt_when_gate_b_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _complete_fixture(root)
            _write_protected_position_with_close_recommendation(root, files["broker"])
            (root / ".operator_close_token").write_text("ok")

            summary = CompletionAuditor(
                broker_snapshot_path=files["broker"],
                order_intents_path=files["intents"],
                target_state_path=files["target"],
                coverage_path=files["coverage"],
                replay_backtest_path=files["replay"],
                execution_replay_path=files["execution"],
                dry_run_certification_path=files["certification"],
                live_order_path=files["live_order"],
                output_path=root / "completion.json",
                report_path=root / "completion.md",
            ).run()

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "completion.json").read_text())
            codes = {item["code"] for item in payload["blockers"]}
            actions = {item["code"] for item in payload["next_actions"]}
            self.assertIn("CLOSE_RECEIPT_REQUIRED", codes)
            self.assertIn("SUBMIT_VERIFIED_CLOSE_RECEIPT", actions)
            self.assertTrue(payload["close_recommendations"]["gate_b_authorized"])
            self.assertTrue(payload["close_recommendations"]["explicit_gate_b_authorized"])

    def test_hard_close_recommendation_requires_close_receipt_without_explicit_gate_b(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _complete_fixture(root)
            _write_protected_position_with_hard_close_recommendation(root, files["broker"])
            prior_override = os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)
            try:
                summary = CompletionAuditor(
                    broker_snapshot_path=files["broker"],
                    order_intents_path=files["intents"],
                    target_state_path=files["target"],
                    coverage_path=files["coverage"],
                    replay_backtest_path=files["replay"],
                    execution_replay_path=files["execution"],
                    dry_run_certification_path=files["certification"],
                    live_order_path=files["live_order"],
                    output_path=root / "completion.json",
                    report_path=root / "completion.md",
                ).run()
            finally:
                if prior_override is not None:
                    os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = prior_override

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "completion.json").read_text())
            codes = {item["code"] for item in payload["blockers"]}
            actions = {item["code"] for item in payload["next_actions"]}
            self.assertIn("CLOSE_RECEIPT_REQUIRED", codes)
            self.assertNotIn("CLOSE_AUTHORIZATION_REQUIRED", codes)
            self.assertIn("SUBMIT_VERIFIED_CLOSE_RECEIPT", actions)
            self.assertTrue(payload["close_recommendations"]["gate_b_authorized"])
            self.assertFalse(payload["close_recommendations"]["explicit_gate_b_authorized"])


def _blocked_fixture(root: Path) -> dict[str, Path]:
    files = _paths(root)
    files["broker"].write_text(
        json.dumps(
            {
                "positions": [{"trade_id": "1", "pair": "EUR_USD", "side": "LONG"}],
                "orders": [{"order_id": "sl", "order_type": "STOP_LOSS", "trade_id": "1"}],
            }
        )
    )
    files["intents"].write_text(json.dumps({"results": [{"status": "DRY_RUN_BLOCKED", "intent": {"pair": "EUR_USD"}}]}))
    files["target"].write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 22000.0, "remaining_risk_budget_jpy": 500.0}))
    files["coverage"].write_text(json.dumps({"status": "COVERAGE_GAP", "blockers": ["no LIVE_READY lanes exist"], "action_items": ["build receipts"]}))
    files["replay"].write_text(json.dumps({"summary": {"days": 10, "evidence_target_covered": 2, "historical_target_hits": 0}}))
    files["execution"].write_text(json.dumps({"status": "BLOCKED", "blockers": ["no LIVE_READY order receipts were available to replay"]}))
    files["certification"].write_text(json.dumps({"status": "BLOCKED", "blockers": ["coverage optimization still has blockers"]}))
    files["live_order"].write_text(json.dumps({"sent": True, "send_requested": True}))
    return files


def _complete_fixture(root: Path) -> dict[str, Path]:
    files = _paths(root)
    files["broker"].write_text(json.dumps({"positions": [], "orders": []}))
    files["intents"].write_text(json.dumps({"results": [{"status": "LIVE_READY", "intent": {"pair": "EUR_USD"}}]}))
    files["target"].write_text(json.dumps({"status": "TARGET_REACHED_PROTECT", "remaining_target_jpy": 0.0, "remaining_risk_budget_jpy": 500.0}))
    files["coverage"].write_text(json.dumps({"status": "TARGET_REACHED_PROTECT", "blockers": [], "action_items": []}))
    files["replay"].write_text(json.dumps({"summary": {"days": 1, "evidence_target_covered": 1, "historical_target_hits": 1}}))
    files["execution"].write_text(json.dumps({"status": "TARGET_HIT", "target_hit": True, "blockers": []}))
    files["certification"].write_text(json.dumps({"status": "CERTIFIED", "blockers": []}))
    files["live_order"].write_text(json.dumps({"sent": False, "send_requested": False}))
    return files


def _write_protected_position_with_close_recommendation(root: Path, broker_path: Path) -> None:
    broker_path.write_text(
        json.dumps(
            {
                "fetched_at_utc": "2026-06-01T10:00:00+00:00",
                "positions": [
                    {
                        "trade_id": "1",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "owner": "trader",
                        "take_profit": 1.15,
                        "stop_loss": 1.18,
                    }
                ],
                "orders": [],
            }
        )
    )
    (root / "position_thesis_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-06-01T10:01:00+00:00",
                "assessments": [
                    {
                        "trade_id": "1",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "verdict": "REVIEW_CLOSE",
                        "rationale_lines": ["prediction stack invalidated SHORT recovery"],
                        "context_notes": [],
                    }
                ],
            }
        )
    )


def _write_protected_position_with_hard_close_recommendation(root: Path, broker_path: Path) -> None:
    broker_path.write_text(
        json.dumps(
            {
                "fetched_at_utc": "2026-06-01T10:00:00+00:00",
                "positions": [
                    {
                        "trade_id": "1",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "owner": "trader",
                        "take_profit": 1.15,
                        "stop_loss": 1.18,
                    }
                ],
                "orders": [],
            }
        )
    )
    (root / "thesis_evolution_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-06-01T10:01:00+00:00",
                "evolutions": [
                    {
                        "trade_id": "1",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "status": "BROKEN",
                        "verdict": "RECOMMEND_CLOSE",
                        "rationale": "invalidation hit and technical invalidation confirmed",
                    }
                ],
            }
        )
    )


def _paths(root: Path) -> dict[str, Path]:
    return {
        "broker": root / "broker.json",
        "intents": root / "intents.json",
        "target": root / "target.json",
        "coverage": root / "coverage.json",
        "replay": root / "replay.json",
        "execution": root / "execution.json",
        "certification": root / "certification.json",
        "live_order": root / "live_order.json",
    }


if __name__ == "__main__":
    unittest.main()
