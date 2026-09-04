from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.capital_flows import (
    DEFAULT_OPERATOR_DEPOSIT_NOTE,
    ensure_operator_deposit_artifact,
    sync_broker_capital_flows_from_execution_ledger,
)


class CapitalFlowsTest(unittest.TestCase):
    def test_syncs_broker_withdrawal_idempotently_and_preserves_operator_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flows_path = root / "data" / "capital_flows.json"
            report_path = root / "docs" / "capital_flow_report.md"
            ledger_path = root / "data" / "execution_ledger.db"
            ensure_operator_deposit_artifact(flows_path, report_path)
            with sqlite3.connect(ledger_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE oanda_transactions (
                        transaction_id TEXT PRIMARY KEY,
                        type TEXT NOT NULL,
                        time_utc TEXT,
                        raw_json TEXT NOT NULL
                    )
                    """
                )
                transaction = {
                    "id": "473459",
                    "time": "2026-09-04T01:07:23.377685213Z",
                    "type": "TRANSFER_FUNDS",
                    "amount": "-200000",
                    "fundingReason": "CLIENT_FUNDING",
                }
                conn.execute(
                    "INSERT INTO oanda_transactions VALUES (?, ?, ?, ?)",
                    (
                        transaction["id"],
                        transaction["type"],
                        transaction["time"],
                        json.dumps(transaction),
                    ),
                )

            first = sync_broker_capital_flows_from_execution_ledger(
                flows_path,
                report_path,
                execution_ledger_path=ledger_path,
            )
            second = sync_broker_capital_flows_from_execution_ledger(
                flows_path,
                report_path,
                execution_ledger_path=ledger_path,
            )
            payload = json.loads(flows_path.read_text())
            broker = [
                row
                for row in payload["capital_flows"]
                if row.get("broker_transaction_id") == "473459"
            ]

            self.assertEqual(first.flow_count, 2)
            self.assertEqual(second.flow_count, 2)
            self.assertEqual(len(broker), 1)
            self.assertEqual(broker[0]["type"], "WITHDRAWAL")
            self.assertEqual(broker[0]["amount_jpy"], 200_000.0)
            self.assertEqual(broker[0]["broker_signed_amount_jpy"], -200_000.0)

    def test_ensure_operator_deposit_artifact_creates_json_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flows_path = root / "data" / "capital_flows.json"
            report_path = root / "docs" / "capital_flow_report.md"
            target_state_path = root / "data" / "daily_target_state.json"
            target_state_path.parent.mkdir(parents=True)
            target_state_path.write_text(
                json.dumps(
                    {
                        "current_equity_raw": 300_000.0,
                        "capital_flows_30d": 100_000.0,
                        "funding_adjusted_equity": 200_000.0,
                        "rolling_30d_multiplier_raw": 1.5,
                        "rolling_30d_multiplier_funding_adjusted": 1.0,
                        "remaining_to_4x_raw": 500_000.0,
                        "remaining_to_4x_funding_adjusted": 600_000.0,
                        "required_calendar_daily_return_funding_adjusted": 4.75,
                        "required_active_day_return_funding_adjusted": 6.6,
                        "performance_basis": "funding_adjusted",
                        "sizing_basis": "raw_nav",
                    }
                )
            )

            result = ensure_operator_deposit_artifact(
                flows_path,
                report_path,
                target_state_path=target_state_path,
            )

            payload = json.loads(flows_path.read_text())
            report = report_path.read_text()
            self.assertTrue(result.created)
            self.assertEqual(payload["capital_flows"][0]["amount_jpy"], 100_000)
            self.assertEqual(payload["capital_flows"][0]["type"], "DEPOSIT")
            self.assertEqual(payload["capital_flows"][0]["source"], "operator")
            self.assertEqual(payload["capital_flows"][0]["timestamp_utc"], "2026-07-02T08:33:11Z")
            self.assertEqual(payload["capital_flows"][0]["note"], DEFAULT_OPERATOR_DEPOSIT_NOTE)
            self.assertIs(payload["capital_flows"][0]["included_in_raw_equity"], True)
            self.assertIs(payload["capital_flows"][0]["excluded_from_funding_adjusted_return"], True)
            self.assertIn("funding_adjusted_equity", report)
            self.assertIn("required_calendar_daily_return_funding_adjusted", report)
            self.assertIn("Risk, margin, and sizing use raw broker NAV", report)

    def test_ensure_operator_deposit_artifact_preserves_existing_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flows_path = root / "data" / "capital_flows.json"
            report_path = root / "docs" / "capital_flow_report.md"
            flows_path.parent.mkdir(parents=True)
            flows_path.write_text(
                json.dumps(
                    {
                        "capital_flows": [
                            {
                                "amount_jpy": 100_000,
                                "excluded_from_funding_adjusted_return": True,
                                "included_in_raw_equity": True,
                                "note": "old note",
                                "source": "operator",
                                "timestamp_utc": "2026-07-02T08:33:11Z",
                                "type": "DEPOSIT",
                            }
                        ],
                        "schema_version": 1,
                    }
                )
            )

            ensure_operator_deposit_artifact(flows_path, report_path)

            payload = json.loads(flows_path.read_text())
            self.assertEqual(payload["capital_flows"][0]["timestamp_utc"], "2026-07-02T08:33:11Z")
            self.assertEqual(payload["capital_flows"][0]["note"], DEFAULT_OPERATOR_DEPOSIT_NOTE)


if __name__ == "__main__":
    unittest.main()
