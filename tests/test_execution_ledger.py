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
                accepted_row = conn.execute(
                    """
                    SELECT lane_id, client_order_id
                    FROM execution_events
                    WHERE event_type = 'ORDER_ACCEPTED'
                    """
                ).fetchone()
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
            self.assertEqual(accepted_row[0], "trend_trader:EUR_USD:LONG:TREND_CONTINUATION")
            self.assertEqual(accepted_row[1], "qrv1-EURUSD-L-test")
            self.assertEqual(last_id, "104")

    def test_sync_recovers_legacy_qrvnext_comment_lane_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            summary = ledger.sync_oanda_transactions(FakeLegacyCommentClient(), since_transaction_id="200")

            self.assertEqual(summary.status, "SYNCED")
            with sqlite3.connect(root / "ledger.db") as conn:
                rows = conn.execute(
                    """
                    SELECT event_type, lane_id, client_order_id, side
                    FROM execution_events
                    WHERE event_type IN ('ORDER_ACCEPTED', 'ORDER_FILLED')
                    ORDER BY event_type
                    """
                ).fetchall()
            self.assertEqual(
                rows,
                [
                    (
                        "ORDER_ACCEPTED",
                        "failure_trader:GBP_USD:SHORT:BREAKOUT_FAILURE",
                        "qrv1-GBPUSD-S-legacy",
                        "SHORT",
                    ),
                    (
                        "ORDER_FILLED",
                        "failure_trader:GBP_USD:SHORT:BREAKOUT_FAILURE",
                        "qrv1-GBPUSD-S-legacy",
                        "SHORT",
                    ),
                ],
            )

    def test_init_backfills_existing_legacy_qrvnext_comment_lane_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")
            ledger.sync_oanda_transactions(FakeLegacyCommentClient(), since_transaction_id="200")
            with sqlite3.connect(root / "ledger.db") as conn:
                conn.execute("UPDATE execution_events SET lane_id = NULL WHERE event_type='ORDER_ACCEPTED'")
                conn.commit()

            ledger.record_gateway_receipt(kind="live_order", receipt_path=root / "missing.json")

            with sqlite3.connect(root / "ledger.db") as conn:
                row = conn.execute(
                    "SELECT lane_id FROM execution_events WHERE event_type='ORDER_ACCEPTED'"
                ).fetchone()
            self.assertEqual(row[0], "failure_trader:GBP_USD:SHORT:BREAKOUT_FAILURE")

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

    def test_records_position_close_receipt_order_id_for_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "position_execution.json"
            receipt.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-08T00:00:00+00:00",
                        "send_requested": True,
                        "actions": [
                            {
                                "trade_id": "T-100",
                                "pair": "EUR_USD",
                                "management_action": "REVIEW_EXIT",
                                "sent": True,
                                "request": {"type": "CLOSE", "trade_id": "T-100", "units": "ALL"},
                                "response": {
                                    "orderCreateTransaction": {
                                        "id": "900",
                                        "reason": "TRADE_CLOSE",
                                        "tradeClose": {"tradeID": "T-100"},
                                    },
                                    "orderFillTransaction": {
                                        "id": "901",
                                        "orderID": "900",
                                        "tradesClosed": [{"tradeID": "T-100"}],
                                    },
                                    "relatedTransactionIDs": ["900", "901"],
                                },
                            }
                        ],
                    }
                )
            )
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            summary = ledger.record_gateway_receipt(kind="position_execution", receipt_path=receipt)

            self.assertEqual(summary.events_inserted, 1)
            with sqlite3.connect(root / "ledger.db") as conn:
                row = conn.execute(
                    """
                    SELECT event_type, order_id, trade_id, pair, exit_reason, related_transaction_ids_json
                    FROM execution_events
                    """
                ).fetchone()
            self.assertEqual(row[0:5], ("GATEWAY_TRADE_CLOSE_SENT", "900", "T-100", "EUR_USD", "REVIEW_EXIT"))
            self.assertEqual(json.loads(row[5]), ["900", "901"])

    def test_init_backfills_gateway_position_order_id_from_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "position_execution.json"
            receipt.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-08T00:00:00+00:00",
                        "send_requested": True,
                        "actions": [
                            {
                                "trade_id": "T-101",
                                "pair": "EUR_USD",
                                "management_action": "REVIEW_EXIT",
                                "sent": True,
                                "request": {"type": "CLOSE", "trade_id": "T-101", "units": "ALL"},
                                "response": {
                                    "orderCreateTransaction": {
                                        "id": "910",
                                        "reason": "TRADE_CLOSE",
                                        "tradeClose": {"tradeID": "T-101"},
                                    },
                                    "orderFillTransaction": {
                                        "id": "911",
                                        "orderID": "910",
                                        "tradesClosed": [{"tradeID": "T-101"}],
                                    },
                                },
                            }
                        ],
                    }
                )
            )
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")
            ledger.record_gateway_receipt(kind="position_execution", receipt_path=receipt)
            with sqlite3.connect(root / "ledger.db") as conn:
                conn.execute("UPDATE execution_events SET order_id = NULL WHERE event_type='GATEWAY_TRADE_CLOSE_SENT'")
                conn.commit()

            ledger.record_gateway_receipt(kind="position_execution", receipt_path=root / "missing.json")

            with sqlite3.connect(root / "ledger.db") as conn:
                row = conn.execute(
                    "SELECT order_id FROM execution_events WHERE event_type='GATEWAY_TRADE_CLOSE_SENT'"
                ).fetchone()
            self.assertEqual(row[0], "910")

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

    def test_records_no_action_gateway_receipt_without_staged_order_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "live_order.json"
            receipt.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-01T10:35:45+00:00",
                        "status": "NO_ACTION",
                        "reason": "cleared stale latest-state live order artifact before current cycle decision",
                        "order_request": None,
                        "send_requested": False,
                        "sent": False,
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
            self.assertEqual(row, ("GATEWAY_ORDER_NO_ACTION", None, None, None, None))

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

    def test_trade_close_order_acceptance_records_closed_trade_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            summary = ledger.sync_oanda_transactions(FakeTradeCloseAcceptClient(), since_transaction_id="500")

            self.assertEqual(summary.status, "SYNCED")
            with sqlite3.connect(root / "ledger.db") as conn:
                row = conn.execute(
                    """
                    SELECT event_type, order_id, trade_id, exit_reason, related_transaction_ids_json
                    FROM execution_events
                    WHERE event_type='ORDER_ACCEPTED'
                    """
                ).fetchone()
            self.assertEqual(row[0], "ORDER_ACCEPTED")
            self.assertEqual(row[1], "501")
            self.assertEqual(row[2], "T501")
            self.assertEqual(row[3], "TRADE_CLOSE")
            self.assertIn("T501", json.loads(row[4]))

    def test_init_backfills_existing_trade_close_order_trade_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")
            ledger.sync_oanda_transactions(FakeTradeCloseAcceptClient(), since_transaction_id="500")
            with sqlite3.connect(root / "ledger.db") as conn:
                conn.execute("UPDATE execution_events SET trade_id = NULL WHERE event_type='ORDER_ACCEPTED'")
                conn.commit()

            ledger.record_gateway_receipt(kind="live_order", receipt_path=root / "missing.json")

            with sqlite3.connect(root / "ledger.db") as conn:
                row = conn.execute(
                    "SELECT trade_id FROM execution_events WHERE event_type='ORDER_ACCEPTED'"
                ).fetchone()
            self.assertEqual(row[0], "T501")


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
                    "clientExtensions": {
                        "id": "qrv1-EURUSD-L-test",
                        "tag": "trader",
                        "comment": "qr-vnext lane=trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                    },
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


class FakeLegacyCommentClient:
    def account_summary(self, *, now_utc: datetime | None = None) -> AccountSummary:
        return AccountSummary(
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            last_transaction_id="200",
            fetched_at_utc=now_utc or datetime.now(timezone.utc),
        )

    def transactions_since_id(self, transaction_id: str) -> dict:
        return {
            "lastTransactionID": "202",
            "transactions": [
                {
                    "id": "201",
                    "time": "2026-06-05T00:00:01.000000000Z",
                    "type": "MARKET_ORDER",
                    "instrument": "GBP_USD",
                    "units": "-6000",
                    "clientExtensions": {
                        "id": "qrv1-GBPUSD-S-legacy",
                        "tag": "trader",
                        "comment": "qr-vnext failure_trader FORECAST_FIRST",
                    },
                    "tradeClientExtensions": {
                        "id": "qrv1-GBPUSD-S-legacy",
                        "tag": "trader",
                        "comment": "qr-vnext failure_trader FORECAST_FIRST",
                    },
                },
                {
                    "id": "202",
                    "time": "2026-06-05T00:00:02.000000000Z",
                    "type": "ORDER_FILL",
                    "orderID": "201",
                    "instrument": "GBP_USD",
                    "units": "-6000",
                    "price": "1.34300",
                    "reason": "MARKET_ORDER",
                    "tradeOpened": {
                        "tradeID": "900",
                        "units": "-6000",
                        "price": "1.34300",
                        "clientExtensions": {
                            "id": "qrv1-GBPUSD-S-legacy",
                            "tag": "trader",
                            "comment": "qr-vnext failure_trader FORECAST_FIRST",
                        },
                    },
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


class FakeTradeCloseAcceptClient:
    def account_summary(self, *, now_utc: datetime | None = None) -> AccountSummary:
        return AccountSummary(
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            last_transaction_id="500",
            fetched_at_utc=now_utc or datetime.now(timezone.utc),
        )

    def transactions_since_id(self, transaction_id: str) -> dict:
        return {
            "lastTransactionID": "501",
            "transactions": [
                {
                    "id": "501",
                    "time": "2026-06-05T16:06:46.688940451Z",
                    "type": "MARKET_ORDER",
                    "instrument": "EUR_JPY",
                    "units": "-2700",
                    "reason": "TRADE_CLOSE",
                    "positionFill": "REDUCE_ONLY",
                    "tradeClose": {"tradeID": "T501", "units": "ALL"},
                },
            ],
        }
