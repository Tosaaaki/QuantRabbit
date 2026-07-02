from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.cli import main
from quant_rabbit.guardian_receipt_consumption import (
    build_guardian_receipt_consumption,
    guardian_receipt_new_entry_blockers,
)
from quant_rabbit.guardian_receipt_operator_review import (
    OPERATOR_ACKNOWLEDGED_HISTORICAL,
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


def _issue() -> dict[str, object]:
    return {
        "code": "GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW",
        "severity": "WARN",
        "receipt_event_id": "receipt-reduce",
        "receipt_action": "REDUCE",
        "receipt_lifecycle": "EXPIRED",
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


def _consumption_row(*, classification: str) -> dict[str, object]:
    return {
        "issue_code": "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
        "receipt_event_id": "receipt-reduce",
        "receipt_action": "REDUCE",
        "receipt_lifecycle": "EXPIRED",
        "consumed_by_trader": False,
        "classification": classification,
        "normal_routing_allowed": classification == "EXPIRED_ACKNOWLEDGED",
        "reason": "fixture",
    }


if __name__ == "__main__":
    unittest.main()
