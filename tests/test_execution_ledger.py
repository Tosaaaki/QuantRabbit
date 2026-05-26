from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.execution_ledger import ExecutionLedger
from quant_rabbit.models import AccountSummary
from quant_rabbit.strategy.entry_thesis_ledger import (
    PendingEntryThesis,
    load_entry_thesis,
    record_pending_entry_thesis,
)


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
                    "SELECT event_type, trade_id, realized_pl_jpy, exit_reason, side FROM execution_events ORDER BY event_uid"
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
            self.assertEqual(close_rows[0][4], "LONG")
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

    def test_records_blocked_gateway_child_even_without_send_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "live_order.json"
            receipt.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-05-15T00:00:00+00:00",
                        "status": "STAGED",
                        "orders": [
                            {
                                "generated_at_utc": "2026-05-15T00:00:01+00:00",
                                "status": "BLOCKED",
                                "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                "order_request": {
                                    "instrument": "EUR_USD",
                                    "type": "MARKET",
                                    "units": "1000",
                                },
                                "risk_issues": [
                                    {
                                        "severity": "BLOCK",
                                        "code": "BASKET_MARGIN_UTILIZATION_CAP_EXCEEDED",
                                    }
                                ],
                                "send_requested": False,
                                "sent": False,
                            }
                        ],
                    }
                )
            )
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            summary = ledger.record_gateway_receipt(kind="live_order", receipt_path=receipt)

            self.assertEqual(summary.events_inserted, 1)
            with sqlite3.connect(root / "ledger.db") as conn:
                row = conn.execute(
                    "SELECT event_type, lane_id, pair, side, units FROM execution_events"
                ).fetchone()
            self.assertEqual(
                row,
                (
                    "GATEWAY_ORDER_BLOCKED",
                    "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                    "EUR_USD",
                    "LONG",
                    1000,
                ),
            )

    def test_sync_promotes_pending_entry_thesis_on_order_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record_pending_entry_thesis(
                PendingEntryThesis(
                    timestamp_utc="2026-05-06T00:00:01Z",
                    order_id="101",
                    pair="EUR_USD",
                    side="LONG",
                    entry_price=1.175,
                    forecast_direction="UP",
                    forecast_confidence=0.71,
                    regime="TREND_CONTINUATION",
                    invalidation_price=1.1725,
                    target_price=1.179,
                    key_drivers=["forecast=UP@conf=0.71", "desk=trend_trader"],
                    lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                ),
                root,
            )
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            summary = ledger.sync_oanda_transactions(FakeTransactionClient(), since_transaction_id="100")
            duplicate = ledger.sync_oanda_transactions(FakeTransactionClient(), since_transaction_id="100")

            self.assertEqual(summary.status, "SYNCED")
            self.assertEqual(duplicate.transactions_inserted, 0)
            thesis = load_entry_thesis("200", root)
            self.assertIsNotNone(thesis)
            assert thesis is not None
            self.assertEqual(thesis.pair, "EUR_USD")
            self.assertEqual(thesis.side, "LONG")
            self.assertEqual(thesis.forecast_direction, "UP")
            self.assertAlmostEqual(thesis.entry_price, 1.1751)

    def test_closed_short_trade_records_original_trade_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            summary = ledger.sync_oanda_transactions(FakeShortCloseClient(), since_transaction_id="100")

            self.assertEqual(summary.status, "SYNCED")
            with sqlite3.connect(root / "ledger.db") as conn:
                row = conn.execute(
                    """
                    SELECT event_type, trade_id, side, units, realized_pl_jpy
                    FROM execution_events
                    WHERE event_type='TRADE_CLOSED'
                    """
                ).fetchone()
            self.assertEqual(row, ("TRADE_CLOSED", "300", "SHORT", 1000, 125.0))

    def test_protection_order_events_record_tp_and_sl_prices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            summary = ledger.sync_oanda_transactions(FakeProtectionClient(), since_transaction_id="100")

            self.assertEqual(summary.status, "SYNCED")
            with sqlite3.connect(root / "ledger.db") as conn:
                rows = conn.execute(
                    """
                    SELECT event_type, trade_id, tp, sl, exit_reason
                    FROM execution_events
                    WHERE event_type='PROTECTION_CREATED'
                    ORDER BY oanda_transaction_id
                    """
                ).fetchall()
            self.assertEqual(
                rows,
                [
                    ("PROTECTION_CREATED", "300", 1.1592, None, "ON_FILL"),
                    ("PROTECTION_CREATED", "300", None, 1.161, "ON_FILL"),
                ],
            )


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


class FakeShortCloseClient:
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
                    "type": "STOP_ORDER",
                    "instrument": "EUR_USD",
                    "units": "-1000",
                    "price": "1.16000",
                    "clientExtensions": {"tag": "trader"},
                },
                {
                    "id": "102",
                    "time": "2026-05-06T00:00:02.000000000Z",
                    "type": "ORDER_FILL",
                    "orderID": "101",
                    "instrument": "EUR_USD",
                    "units": "-1000",
                    "price": "1.16000",
                    "reason": "STOP_ORDER",
                    "tradeOpened": {"tradeID": "300", "units": "-1000", "price": "1.16000"},
                },
                {
                    "id": "103",
                    "time": "2026-05-06T00:10:00.000000000Z",
                    "type": "ORDER_FILL",
                    "orderID": "150",
                    "instrument": "EUR_USD",
                    "units": "1000",
                    "price": "1.15920",
                    "reason": "TAKE_PROFIT_ORDER",
                    "pl": "125.0",
                    "tradesClosed": [
                        {
                            "tradeID": "300",
                            "units": "1000",
                            "price": "1.15920",
                            "realizedPL": "125.0",
                        }
                    ],
                },
            ],
        }


class FakeProtectionClient:
    def account_summary(self, *, now_utc: datetime | None = None) -> AccountSummary:
        return AccountSummary(
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            last_transaction_id="100",
            fetched_at_utc=now_utc or datetime.now(timezone.utc),
        )

    def transactions_since_id(self, transaction_id: str) -> dict:
        return {
            "lastTransactionID": "102",
            "transactions": [
                {
                    "id": "101",
                    "time": "2026-05-06T00:00:02.000000000Z",
                    "type": "TAKE_PROFIT_ORDER",
                    "tradeID": "300",
                    "price": "1.15920",
                    "reason": "ON_FILL",
                },
                {
                    "id": "102",
                    "time": "2026-05-06T00:00:03.000000000Z",
                    "type": "STOP_LOSS_ORDER",
                    "tradeID": "300",
                    "price": "1.16100",
                    "reason": "ON_FILL",
                },
            ],
        }
