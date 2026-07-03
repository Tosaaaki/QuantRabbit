from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.cli import main
from quant_rabbit.guardian_receipt_consumption import (
    WATCHDOG_BLOCK_NEW_ENTRY_CODE,
    build_guardian_receipt_consumption,
    guardian_receipt_new_entry_blockers,
)
from quant_rabbit.guardian_receipt_operator_review import (
    OPERATOR_ACKNOWLEDGED_HISTORICAL,
    OPERATOR_CONFIRMED_MANUAL_OWNED,
    build_guardian_receipt_operator_review,
    operator_review_clearance_status,
)


class GuardianReceiptOperatorReviewTest(unittest.TestCase):
    def test_missing_operator_review_blocks_reduce_new_entries(self) -> None:
        consumption = {
            "normal_routing_allowed": False,
            "classifications": [_consumption_row(classification="NEEDS_OPERATOR_REVIEW")],
        }

        blockers = guardian_receipt_new_entry_blockers(
            {},
            consumption,
            {},
            {"positions": [], "orders": []},
        )

        self.assertEqual({item["code"] for item in blockers}, {"GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"})

    def test_watchdog_p0_block_is_not_reported_as_operator_review_required(self) -> None:
        watchdog = {
            "status": "STALE",
            "severity": "P0",
            "guardian_receipt_issues": [
                {
                    "code": "QR_TRADER_RUN_STALE",
                    "severity": "P0",
                }
            ],
        }
        acknowledged_row = _consumption_row(classification="HISTORICAL_ONLY", receipt_action="HOLD")
        acknowledged_row["normal_routing_allowed"] = True
        consumption = {
            "status": "GUARDIAN_RECEIPT_ISSUES_ACKNOWLEDGED_CURRENT_P0_BLOCKS_ROUTING",
            "normal_routing_allowed": False,
            "current_p0_p1_blocks_routing": True,
            "classifications": [acknowledged_row],
        }

        blockers = guardian_receipt_new_entry_blockers(
            watchdog,
            consumption,
            {"normal_routing_allowed": True},
            {"positions": [], "orders": []},
        )

        self.assertEqual({item["code"] for item in blockers}, {WATCHDOG_BLOCK_NEW_ENTRY_CODE})

    def test_stale_watchdog_keeps_consumption_closed_after_receipt_acknowledgement(self) -> None:
        now = datetime(2026, 7, 2, tzinfo=timezone.utc)
        receipt_issue = {
            "code": "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
            "severity": "WARN",
            "receipt_event_id": "receipt-reduce",
            "receipt_action": "HOLD",
            "receipt_lifecycle": "EXPIRED",
            "consumed_by_trader": False,
        }
        watchdog = {
            "status": "STALE",
            "issue_status": "P0",
            "severity": "P0",
            "issues": [{"code": "QR_TRADER_RUN_STALE", "severity": "P0"}],
            "guardian_receipt": {"issues": [receipt_issue]},
        }
        consumption = build_guardian_receipt_consumption(
            watchdog,
            existing={
                "classifications": [
                    _consumption_row(classification="EXPIRED_ACKNOWLEDGED", receipt_action="HOLD")
                ]
            },
            operator_review={},
            broker_snapshot={"positions": [], "orders": []},
            now_utc=now,
        )

        self.assertEqual(consumption["status"], "GUARDIAN_RECEIPT_ISSUES_ACKNOWLEDGED_CURRENT_P0_BLOCKS_ROUTING")
        self.assertFalse(consumption["normal_routing_allowed"])
        self.assertTrue(consumption["current_p0_p1_blocks_routing"])

    def test_acknowledged_historical_rows_survive_watchdog_issue_rotation(self) -> None:
        now = datetime(2026, 7, 2, tzinfo=timezone.utc)
        old_row = _consumption_row(
            classification="HISTORICAL_ONLY",
            receipt_event_id="old-historical",
            receipt_action="HOLD",
        )
        old_row["normal_routing_allowed"] = True
        current_issue = {
            "code": "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
            "severity": "WARN",
            "receipt_event_id": "new-expired",
            "receipt_action": "HOLD",
            "receipt_lifecycle": "EXPIRED",
            "receipt_sources": ["archive"],
            "consumed_by_trader": False,
        }
        watchdog = {
            "status": "STALE",
            "issue_status": "P0",
            "issues": [{"code": "QR_TRADER_RUN_STALE", "severity": "P0"}],
            "guardian_receipt": {"issues": [current_issue]},
        }

        consumption = build_guardian_receipt_consumption(
            watchdog,
            existing={"classifications": [old_row]},
            operator_review={},
            broker_snapshot={"positions": [], "orders": []},
            now_utc=now,
        )

        by_event = {row["receipt_event_id"]: row for row in consumption["classifications"]}
        self.assertEqual(set(by_event), {"old-historical", "new-expired"})
        self.assertEqual(by_event["old-historical"]["classification"], "HISTORICAL_ONLY")
        self.assertTrue(by_event["old-historical"]["normal_routing_allowed"])
        self.assertEqual(by_event["new-expired"]["classification"], "HISTORICAL_ONLY")
        self.assertTrue(by_event["new-expired"]["normal_routing_allowed"])
        blockers = guardian_receipt_new_entry_blockers(
            watchdog,
            consumption,
            {"normal_routing_allowed": True},
            {"positions": [], "orders": []},
        )
        self.assertEqual({item["code"] for item in blockers}, {WATCHDOG_BLOCK_NEW_ENTRY_CODE})

    def test_stale_operator_review_keeps_blocker_active(self) -> None:
        now = datetime(2026, 7, 2, tzinfo=timezone.utc)
        issue = _issue()
        review = {
            "classifications": [
                {
                    "receipt_event_id": "receipt-reduce",
                    "receipt_action": "REDUCE",
                    "receipt_lifecycle": "EXPIRED",
                    "original_issue_code": "GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW",
                    "operator_decision": OPERATOR_ACKNOWLEDGED_HISTORICAL,
                    "reason": "historical receipt reviewed",
                    "generated_at_utc": (now - timedelta(days=2)).isoformat(),
                    "expires_at_utc": (now - timedelta(days=1)).isoformat(),
                    "normal_routing_allowed": True,
                    "no_live_side_effects": True,
                }
            ]
        }

        status = operator_review_clearance_status(
            issue,
            review,
            watchdog_payload={"status": "OK", "issue_status": "OK", "guardian_receipt": {"issues": []}},
            broker_snapshot_payload={"positions": [], "orders": []},
            now_utc=now,
        )

        self.assertFalse(status["normal_routing_allowed"])
        self.assertEqual(status["status"], "OPERATOR_REVIEW_STALE")

    def test_valid_review_marks_expired_historical_receipt_acknowledged(self) -> None:
        now = datetime(2026, 7, 2, tzinfo=timezone.utc)
        review = build_guardian_receipt_operator_review(
            {"status": "OK", "issue_status": "OK", "guardian_receipt": {"issues": []}},
            {"classifications": [_consumption_row(classification="NEEDS_OPERATOR_REVIEW")]},
            broker_snapshot_payload={"positions": [], "orders": []},
            operator_decision_payload={
                "decisions": [
                    {
                        "receipt_event_id": "receipt-reduce",
                        "receipt_action": "REDUCE",
                        "receipt_lifecycle": "EXPIRED",
                        "operator_decision": OPERATOR_ACKNOWLEDGED_HISTORICAL,
                        "reason": "operator verified the receipt is historical and no active emergency remains",
                        "expires_at_utc": (now + timedelta(hours=2)).isoformat(),
                        "no_live_side_effects": True,
                    }
                ]
            },
            now_utc=now,
        )

        self.assertEqual(review["status"], "GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED")
        self.assertTrue(review["normal_routing_allowed"])
        self.assertEqual(review["classifications"][0]["operator_decision"], OPERATOR_ACKNOWLEDGED_HISTORICAL)
        self.assertTrue(review["classifications"][0]["normal_routing_allowed"])
        self.assertTrue(review["no_live_side_effects"])
        self.assertEqual(review["live_side_effects"], [])

    def test_historical_receipt_can_clear_while_current_p0_blocks_global_routing(self) -> None:
        now = datetime(2026, 7, 2, tzinfo=timezone.utc)
        watchdog = {
            "status": "BLOCKED",
            "issue_status": "P0",
            "guardian_receipt": {"issues": [_current_aud_usd_p0_issue()]},
        }
        review = build_guardian_receipt_operator_review(
            watchdog,
            {"classifications": [_consumption_row(classification="NEEDS_OPERATOR_REVIEW")]},
            broker_snapshot_payload={"positions": [], "orders": []},
            operator_decision_payload={
                "decisions": [
                    {
                        "receipt_event_id": "receipt-reduce",
                        "receipt_action": "REDUCE",
                        "receipt_lifecycle": "EXPIRED",
                        "operator_decision": OPERATOR_ACKNOWLEDGED_HISTORICAL,
                        "reason": "operator verified the receipt is historical and no active emergency remains",
                        "expires_at_utc": (now + timedelta(hours=2)).isoformat(),
                        "no_live_side_effects": True,
                    }
                ]
            },
            now_utc=now,
        )
        consumption = build_guardian_receipt_consumption(
            watchdog,
            existing={"classifications": [_consumption_row(classification="NEEDS_OPERATOR_REVIEW")]},
            operator_review=review,
            broker_snapshot={"positions": [], "orders": []},
            now_utc=now,
        )

        self.assertEqual(
            review["status"],
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED_CURRENT_P0_BLOCKS_ROUTING",
        )
        self.assertFalse(review["normal_routing_allowed"])
        self.assertEqual(review["unresolved_review_count"], 0)
        self.assertTrue(review["classifications"][0]["normal_routing_allowed"])
        historical_row = next(
            item
            for item in consumption["classifications"]
            if item["receipt_event_id"] == "receipt-reduce"
        )
        current_p0_row = next(
            item
            for item in consumption["classifications"]
            if item["receipt_event_id"] == "aud-current-p0"
        )
        self.assertEqual(historical_row["classification"], "HISTORICAL_ONLY")
        self.assertTrue(historical_row["normal_routing_allowed"])
        self.assertEqual(current_p0_row["classification"], "NEEDS_OPERATOR_REVIEW")
        self.assertFalse(current_p0_row["normal_routing_allowed"])
        self.assertFalse(consumption["normal_routing_allowed"])
        self.assertTrue(consumption["current_p0_p1_blocks_routing"])

    def test_manual_ownership_confirmation_does_not_unlock_global_routing_with_other_p0(self) -> None:
        now = datetime(2026, 7, 2, tzinfo=timezone.utc)
        watchdog = {
            "status": "BLOCKED",
            "issue_status": "P0",
            "guardian_receipt": {"issues": [_current_aud_usd_p0_issue()]},
        }
        review = build_guardian_receipt_operator_review(
            watchdog,
            {"classifications": [_consumption_row(classification="NEEDS_OPERATOR_REVIEW")]},
            broker_snapshot_payload={
                "positions": [
                    {
                        "trade_id": "472987",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "units": 30000,
                        "entry_price": 1.14048,
                        "owner": "operator_manual",
                    }
                ],
                "orders": [],
            },
            operator_decision_payload={
                "decisions": [
                    {
                        "receipt_event_id": "receipt-reduce",
                        "receipt_action": "REDUCE",
                        "receipt_lifecycle": "EXPIRED",
                        "trade_id": "472987",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "units": 30000,
                        "avg_entry": 1.14048,
                        "operator_decision": OPERATOR_CONFIRMED_MANUAL_OWNED,
                        "management_intent": "KEEP",
                        "owner": "OPERATOR_MANUAL",
                        "operator_confirmation_source": "chat_operator_confirmation",
                        "system_pl_counted": False,
                        "same_theme_auto_add_allowed": False,
                        "loss_side_auto_close_allowed": False,
                        "auto_sl_attach_allowed": False,
                        "auto_tp_modify_allowed": False,
                        "reason": "operator explicitly confirmed manual EUR_USD should remain open",
                        "expires_at_utc": (now + timedelta(hours=2)).isoformat(),
                        "no_live_side_effects": True,
                    }
                ]
            },
            now_utc=now,
        )

        self.assertEqual(
            review["status"],
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED_CURRENT_P0_BLOCKS_ROUTING",
        )
        self.assertFalse(review["normal_routing_allowed"])
        self.assertTrue(review["current_p0_p1_blocks_routing"])
        self.assertEqual(review["classifications"][0]["operator_decision"], OPERATOR_CONFIRMED_MANUAL_OWNED)
        self.assertEqual(review["classifications"][0]["trade_id"], "472987")
        self.assertEqual(review["classifications"][0]["owner"], "OPERATOR_MANUAL")
        self.assertEqual(review["classifications"][0]["management_intent"], "KEEP")
        self.assertEqual(
            review["classifications"][0]["operator_confirmation_source"],
            "chat_operator_confirmation",
        )
        self.assertFalse(review["classifications"][0]["system_pl_counted"])
        self.assertFalse(review["classifications"][0]["same_theme_auto_add_allowed"])
        self.assertFalse(review["classifications"][0]["loss_side_auto_close_allowed"])
        self.assertFalse(review["classifications"][0]["auto_sl_attach_allowed"])
        self.assertFalse(review["classifications"][0]["auto_tp_modify_allowed"])
        self.assertTrue(review["classifications"][0]["normal_routing_allowed"])

    def test_emergency_hold_receipt_clears_with_manual_ownership_review(self) -> None:
        now = datetime(2026, 7, 2, tzinfo=timezone.utc)
        watchdog = {
            "status": "BLOCKED",
            "issue_status": "P0",
            "guardian_receipt": {"issues": [_emergency_hold_receipt_issue()]},
        }
        consumption_row = _consumption_row(
            classification="NEEDS_OPERATOR_REVIEW",
            receipt_event_id="receipt-hold",
            receipt_action="HOLD",
        )
        broker_snapshot = {
            "positions": [
                {
                    "trade_id": "472987",
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": 30000,
                    "entry_price": 1.14048,
                    "owner": "operator_manual",
                }
            ],
            "orders": [],
        }
        review = build_guardian_receipt_operator_review(
            watchdog,
            {"classifications": [consumption_row]},
            broker_snapshot_payload=broker_snapshot,
            operator_decision_payload={
                "decisions": [
                    {
                        "receipt_event_id": "receipt-hold",
                        "receipt_action": "HOLD",
                        "receipt_lifecycle": "EXPIRED",
                        "trade_id": "472987",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "units": 30000,
                        "avg_entry": 1.14048,
                        "operator_decision": OPERATOR_CONFIRMED_MANUAL_OWNED,
                        "management_intent": "KEEP",
                        "owner": "OPERATOR_MANUAL",
                        "operator_confirmation_source": "chat_operator_confirmation",
                        "system_pl_counted": False,
                        "same_theme_auto_add_allowed": False,
                        "loss_side_auto_close_allowed": False,
                        "auto_sl_attach_allowed": False,
                        "auto_tp_modify_allowed": False,
                        "reason": "operator explicitly confirmed manual EUR_USD should remain open",
                        "expires_at_utc": (now + timedelta(hours=2)).isoformat(),
                        "no_live_side_effects": True,
                    }
                ]
            },
            now_utc=now,
        )
        consumption = build_guardian_receipt_consumption(
            watchdog,
            existing={"classifications": [consumption_row]},
            operator_review=review,
            broker_snapshot=broker_snapshot,
            now_utc=now,
        )

        self.assertEqual(review["status"], "GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED")
        self.assertTrue(review["normal_routing_allowed"])
        self.assertEqual(review["classifications"][0]["operator_decision"], OPERATOR_CONFIRMED_MANUAL_OWNED)
        self.assertTrue(review["classifications"][0]["normal_routing_allowed"])
        self.assertEqual(consumption["status"], "GUARDIAN_RECEIPT_ISSUES_ACKNOWLEDGED")
        self.assertTrue(consumption["normal_routing_allowed"])
        self.assertEqual(consumption["classifications"][0]["classification"], "HISTORICAL_ONLY")
        self.assertTrue(consumption["classifications"][0]["normal_routing_allowed"])

    def test_legacy_expired_acknowledged_reduce_row_is_reblocked_without_review(self) -> None:
        now = datetime(2026, 7, 2, tzinfo=timezone.utc)
        consumption = build_guardian_receipt_consumption(
            {"status": "OK", "issue_status": "OK", "guardian_receipt": {"issues": []}},
            existing={"classifications": [_consumption_row(classification="EXPIRED_ACKNOWLEDGED")]},
            operator_review={},
            broker_snapshot={"positions": [], "orders": []},
            now_utc=now,
        )

        self.assertFalse(consumption["normal_routing_allowed"])
        self.assertEqual(consumption["classifications"][0]["classification"], "NEEDS_OPERATOR_REVIEW")
        self.assertEqual(consumption["classifications"][0]["operator_review_status"], "OPERATOR_REVIEW_MISSING")

    def test_operator_review_cli_default_mode_is_report_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watchdog = root / "watchdog.json"
            consumption = root / "consumption.json"
            broker = root / "broker.json"
            output = root / "review.json"
            report = root / "review.md"
            _write_review_inputs(watchdog, consumption, broker)

            rc = main(
                [
                    "guardian-receipt-operator-review",
                    "--qr-trader-run-watchdog",
                    str(watchdog),
                    "--guardian-receipt-consumption",
                    str(consumption),
                    "--broker-snapshot",
                    str(broker),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                ]
            )

            self.assertEqual(rc, 0)
            self.assertFalse(output.exists())
            self.assertFalse(report.exists())

    def test_operator_review_cli_writes_only_with_explicit_decision_file(self) -> None:
        now = datetime(2026, 7, 2, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watchdog = root / "watchdog.json"
            consumption = root / "consumption.json"
            broker = root / "broker.json"
            decision = root / "decision.json"
            output = root / "review.json"
            report = root / "review.md"
            _write_review_inputs(watchdog, consumption, broker)
            decision.write_text(
                "{"
                '"decisions": [{"receipt_event_id": "receipt-reduce", "receipt_action": "REDUCE", '
                '"receipt_lifecycle": "EXPIRED", "operator_decision": "OPERATOR_ACKNOWLEDGED_HISTORICAL", '
                '"reason": "operator confirmed historical", '
                f'"expires_at_utc": "{(now + timedelta(hours=1)).isoformat()}", '
                '"no_live_side_effects": true}]'
                "}"
            )

            rc = main(
                [
                    "guardian-receipt-operator-review",
                    "--qr-trader-run-watchdog",
                    str(watchdog),
                    "--guardian-receipt-consumption",
                    str(consumption),
                    "--broker-snapshot",
                    str(broker),
                    "--operator-decision-file",
                    str(decision),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(output.exists())
            self.assertTrue(report.exists())
            self.assertIn("no_live_side_effects", output.read_text())
            self.assertIn("does not place orders", report.read_text())

    def test_operator_review_cli_preserves_operator_position_reviews_from_decision_file(self) -> None:
        now = datetime(2026, 7, 2, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watchdog = root / "watchdog.json"
            consumption = root / "consumption.json"
            broker = root / "broker.json"
            decision = root / "decision.json"
            output = root / "review.json"
            report = root / "review.md"
            _write_review_inputs(watchdog, consumption, broker)
            decision.write_text(
                json.dumps(
                    {
                        "decisions": [
                            {
                                "receipt_event_id": "receipt-reduce",
                                "receipt_action": "REDUCE",
                                "receipt_lifecycle": "EXPIRED",
                                "operator_decision": "OPERATOR_ACKNOWLEDGED_HISTORICAL",
                                "reason": "operator confirmed historical",
                                "expires_at_utc": (now + timedelta(hours=1)).isoformat(),
                                "no_live_side_effects": True,
                            }
                        ],
                        "operator_position_reviews": [
                            {
                                "trade_id": "472987",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "units": 30000,
                                "avg_entry": 1.14048,
                                "owner": "OPERATOR_MANUAL",
                                "operator_decision": "OPERATOR_CONFIRMED_MANUAL_OWNED",
                                "management_intent": "KEEP",
                                "operator_confirmation_source": "chat_operator_confirmation",
                                "system_pl_counted": False,
                                "same_theme_auto_add_allowed": False,
                                "loss_side_auto_close_allowed": False,
                                "auto_sl_attach_allowed": False,
                                "auto_tp_modify_allowed": False,
                            }
                        ],
                    }
                )
            )

            rc = main(
                [
                    "guardian-receipt-operator-review",
                    "--qr-trader-run-watchdog",
                    str(watchdog),
                    "--guardian-receipt-consumption",
                    str(consumption),
                    "--broker-snapshot",
                    str(broker),
                    "--operator-decision-file",
                    str(decision),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                ]
            )

            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["operator_position_reviews"][0]["trade_id"], "472987")
            report_text = report.read_text()
            self.assertIn("Operator Position Review", report_text)
            self.assertIn("472987", report_text)

    def test_guardian_receipt_consumption_cli_rebuilds_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watchdog = root / "watchdog.json"
            existing = root / "consumption.json"
            review = root / "review.json"
            broker = root / "broker.json"
            report = root / "consumption.md"
            _write_review_inputs(watchdog, existing, broker)
            review.write_text(
                json.dumps(
                    {
                        "classifications": [
                            {
                                "receipt_event_id": "receipt-reduce",
                                "receipt_action": "REDUCE",
                                "receipt_lifecycle": "EXPIRED",
                                "operator_decision": "OPERATOR_ACKNOWLEDGED_HISTORICAL",
                                "reason": "operator confirmed historical",
                                "generated_at_utc": "2026-07-02T00:00:00+00:00",
                                "expires_at_utc": "2099-07-02T00:00:00+00:00",
                                "normal_routing_allowed": True,
                                "no_live_side_effects": True,
                            }
                        ]
                    }
                )
            )

            rc = main(
                [
                    "guardian-receipt-consumption",
                    "--qr-trader-run-watchdog",
                    str(watchdog),
                    "--existing",
                    str(existing),
                    "--operator-review",
                    str(review),
                    "--broker-snapshot",
                    str(broker),
                    "--output",
                    str(existing),
                    "--report",
                    str(report),
                ]
            )

            self.assertEqual(rc, 0)
            payload = json.loads(existing.read_text())
            self.assertEqual(payload["classifications"][0]["classification"], "HISTORICAL_ONLY")
            self.assertTrue(payload["classifications"][0]["normal_routing_allowed"])
            self.assertTrue(report.exists())


def _issue() -> dict[str, object]:
    return {
        "code": "GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW",
        "severity": "WARN",
        "receipt_event_id": "receipt-reduce",
        "receipt_action": "REDUCE",
        "receipt_lifecycle": "EXPIRED",
        "consumed_by_trader": False,
    }


def _current_aud_usd_p0_issue() -> dict[str, object]:
    return {
        "code": "CURRENT_GUARDIAN_P0_UNKNOWN_EXPOSURE",
        "severity": "P0",
        "receipt_event_id": "aud-current-p0",
        "receipt_action": "REDUCE",
        "receipt_lifecycle": "CURRENT_GUARDIAN_EVENT",
    }


def _emergency_hold_receipt_issue() -> dict[str, object]:
    return {
        "code": "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
        "severity": "P0",
        "receipt_event_id": "receipt-hold",
        "receipt_action": "HOLD",
        "receipt_lifecycle": "EXPIRED",
        "trade_id": "472987",
        "emergency_or_margin_risk": True,
        "consumed_by_trader": False,
    }


def _write_review_inputs(watchdog: Path, consumption: Path, broker: Path) -> None:
    watchdog.write_text('{"status": "OK", "issue_status": "OK", "guardian_receipt": {"issues": []}}')
    consumption.write_text(
        '{"classifications": [{"issue_code": "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER", '
        '"receipt_event_id": "receipt-reduce", "receipt_action": "REDUCE", '
        '"receipt_lifecycle": "EXPIRED", "consumed_by_trader": false, '
        '"classification": "NEEDS_OPERATOR_REVIEW", "normal_routing_allowed": false}]}'
    )
    broker.write_text('{"positions": [], "orders": []}')


def _consumption_row(
    *,
    classification: str,
    receipt_event_id: str = "receipt-reduce",
    receipt_action: str = "REDUCE",
) -> dict[str, object]:
    return {
        "issue_code": "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
        "receipt_event_id": receipt_event_id,
        "receipt_action": receipt_action,
        "receipt_lifecycle": "EXPIRED",
        "consumed_by_trader": False,
        "classification": classification,
        "normal_routing_allowed": classification == "EXPIRED_ACKNOWLEDGED",
        "reason": "fixture",
    }


if __name__ == "__main__":
    unittest.main()
