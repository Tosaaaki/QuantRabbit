from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.execution_ledger import ExecutionLedger
from quant_rabbit.models import AccountSummary


class ExecutionLedgerTest(unittest.TestCase):
    def test_syncs_oanda_transactions_raw_and_normalized_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")
            client = FakeTransactionClient()

            summary = ledger.sync_oanda_transactions(client, since_transaction_id="100")
            duplicate = ledger.sync_oanda_transactions(client, since_transaction_id="100")

            self.assertEqual(summary.status, "SYNCED")
            self.assertEqual(summary.transactions_seen, 4)
            self.assertEqual(summary.transactions_inserted, 4)
            self.assertEqual(duplicate.transactions_seen, 4)
            self.assertEqual(duplicate.transactions_inserted, 0)
            with sqlite3.connect(root / "ledger.db") as conn:
                tx_count = conn.execute("SELECT COUNT(*) FROM oanda_transactions").fetchone()[0]
                event_rows = conn.execute(
                    "SELECT event_type, trade_id, realized_pl_jpy, exit_reason FROM execution_events ORDER BY event_uid"
                ).fetchall()
                last_id = conn.execute(
                    "SELECT value FROM sync_state WHERE key='last_oanda_transaction_id'"
                ).fetchone()[0]

            self.assertEqual(tx_count, 4)
            event_types = {row[0] for row in event_rows}
            self.assertIn("ORDER_ACCEPTED", event_types)
            self.assertIn("ORDER_FILLED", event_types)
            self.assertIn("TRADE_CLOSED", event_types)
            self.assertIn("OANDA_TRANSACTION", event_types)
            close_rows = [row for row in event_rows if row[0] == "TRADE_CLOSED"]
            self.assertEqual(close_rows[0][1], "200")
            self.assertEqual(close_rows[0][2], 350.0)
            self.assertEqual(close_rows[0][3], "TAKE_PROFIT_ORDER")
            self.assertEqual(last_id, "104")

    def test_records_gateway_receipt_as_append_only_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "live_order.json"
            receipt.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-05-06T00:00:00+00:00",
                        "status": "SENT",
                        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                        "order_request": {
                            "instrument": "EUR_USD",
                            "type": "MARKET",
                            "units": "1000",
                            "takeProfitOnFill": {"price": "1.18000"},
                            "stopLossOnFill": {"price": "1.17000"},
                        },
                        "response": {
                            "orderCreateTransaction": {"id": "101"},
                            "orderFillTransaction": {
                                "orderID": "101",
                                "tradeOpened": {"tradeID": "200"},
                            },
                            "relatedTransactionIDs": ["101", "102"],
                        },
                        "send_requested": True,
                        "sent": True,
                    }
                )
            )
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            summary = ledger.record_gateway_receipt(kind="live_order", receipt_path=receipt)
            duplicate = ledger.record_gateway_receipt(kind="live_order", receipt_path=receipt)

            self.assertEqual(summary.gateway_receipts_inserted, 1)
            self.assertEqual(summary.events_inserted, 1)
            self.assertEqual(duplicate.gateway_receipts_inserted, 0)
            with sqlite3.connect(root / "ledger.db") as conn:
                row = conn.execute(
                    """
                    SELECT event_type, lane_id, order_id, trade_id, pair, side, units, tp, sl
                    FROM execution_events
                    """
                ).fetchone()
            self.assertEqual(row, (
                "GATEWAY_ORDER_SENT",
                "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                "101",
                "200",
                "EUR_USD",
                "LONG",
                1000,
                1.18,
                1.17,
            ))


class FakeTransactionClient:
    def account_summary(self, *, now_utc: datetime | None = None) -> AccountSummary:
        return AccountSummary(
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            last_transaction_id="100",
            fetched_at_utc=now_utc or datetime.now(timezone.utc),
        )

    def transactions_since_id(self, transaction_id: str) -> dict:
        return {
            "lastTransactionID": "104",
            "transactions": [
                {
                    "id": "101",
                    "time": "2026-05-06T00:00:01.000000000Z",
                    "type": "LIMIT_ORDER",
                    "instrument": "EUR_USD",
                    "units": "1000",
                    "price": "1.17500",
                    "clientExtensions": {"tag": "trader"},
                },
                {
                    "id": "102",
                    "time": "2026-05-06T00:00:02.000000000Z",
                    "type": "ORDER_FILL",
                    "orderID": "101",
                    "instrument": "EUR_USD",
                    "units": "1000",
                    "price": "1.17510",
                    "reason": "LIMIT_ORDER",
                    "tradeOpened": {"tradeID": "200", "units": "1000", "price": "1.17510"},
                },
                {
                    "id": "103",
                    "time": "2026-05-06T00:10:00.000000000Z",
                    "type": "ORDER_FILL",
                    "orderID": "150",
                    "instrument": "EUR_USD",
                    "units": "-1000",
                    "price": "1.17900",
                    "reason": "TAKE_PROFIT_ORDER",
                    "pl": "350.0",
                    "tradesClosed": [
                        {
                            "tradeID": "200",
                            "units": "1000",
                            "price": "1.17900",
                            "realizedPL": "350.0",
                        }
                    ],
                },
                {
                    "id": "104",
                    "time": "2026-05-06T23:00:00.000000000Z",
                    "type": "DAILY_FINANCING",
                    "financing": "-1.2",
                },
            ],
        }
