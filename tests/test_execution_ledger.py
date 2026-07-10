from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.execution_ledger import (
    LEGACY_EVENT_BACKFILL_MIGRATION_KEY,
    LEGACY_EVENT_BACKFILL_MIGRATION_VERSION,
    OANDA_TRANSACTION_COVERAGE_START_KEY,
    ExecutionLedger,
    _events_from_transaction,
)
import quant_rabbit.execution_ledger as execution_ledger_module
from quant_rabbit.models import AccountSummary
from quant_rabbit.strategy.entry_thesis_ledger import (
    PendingEntryThesis,
    load_entry_thesis,
    record_pending_entry_thesis,
)


class ExecutionLedgerTest(unittest.TestCase):
    def test_large_gateway_input_packet_is_content_addressed_and_not_duplicated_in_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")
            input_packet = {
                "lanes": [{"lane_id": "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"}],
                "allowed_evidence_refs": ["chart:EUR_USD:M5", "news:health"],
                "broker_snapshot": {"fetched_at_utc": "2026-07-10T10:00:00+00:00"},
                "artifact_timestamps": {"pair_charts": "2026-07-10T10:00:00+00:00"},
                "large_evidence": "market-evidence-" * 40000,
            }
            first = {
                "generated_at_utc": "2026-07-10T10:01:00+00:00",
                "status": "ACCEPTED",
                "decision": {"action": "WAIT"},
                "input_packet": input_packet,
            }
            second = {
                **first,
                "generated_at_utc": "2026-07-10T11:01:00+00:00",
                "input_packet": input_packet,
            }

            ledger.record_gateway_payload(kind="gpt_decision", receipt_path=root / "first.json", payload=first)
            ledger.record_gateway_payload(kind="gpt_decision", receipt_path=root / "second.json", payload=second)

            self.assertIs(first["input_packet"], input_packet)
            with sqlite3.connect(root / "ledger.db") as conn:
                rows = conn.execute(
                    "SELECT payload_json FROM gateway_receipts ORDER BY ts_utc"
                ).fetchall()
            self.assertEqual(len(rows), 2)
            stored = [json.loads(row[0]) for row in rows]
            self.assertTrue(all("input_packet" not in item for item in stored))
            self.assertEqual(
                stored[0]["input_packet_archive"]["sha256"],
                stored[1]["input_packet_archive"]["sha256"],
            )
            self.assertLess(len(rows[0][0]), 10000)
            archive_path = root / stored[0]["input_packet_archive"]["path"]
            self.assertTrue(archive_path.exists())
            self.assertEqual(len(list((root / "execution_evidence" / "gateway_input_packets").rglob("*.json.gz"))), 1)
            with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), input_packet)
            self.assertEqual(stored[0]["input_packet_summary"]["lane_count"], 1)
            self.assertEqual(stored[0]["input_packet_summary"]["allowed_evidence_ref_count"], 2)

    def test_corrupt_existing_gateway_packet_archive_refuses_new_ledger_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")
            payload = {
                "generated_at_utc": "2026-07-10T10:01:00+00:00",
                "status": "ACCEPTED",
                "input_packet": {"large_evidence": "x" * 10000},
            }
            ledger.record_gateway_payload(kind="gpt_decision", receipt_path=root / "first.json", payload=payload)
            archive = next((root / "execution_evidence" / "gateway_input_packets").rglob("*.json.gz"))
            archive.write_bytes(b"not-gzip")
            retry = {**payload, "generated_at_utc": "2026-07-10T11:01:00+00:00"}

            with self.assertRaises(ValueError):
                ledger.record_gateway_payload(kind="gpt_decision", receipt_path=root / "second.json", payload=retry)

            with sqlite3.connect(root / "ledger.db") as conn:
                count = conn.execute("SELECT COUNT(*) FROM gateway_receipts").fetchone()[0]
            self.assertEqual(count, 1)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink unavailable")
    def test_gateway_packet_archive_rejects_symlinked_storage_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(tmp)
            outside = Path(outside_tmp)
            (root / "execution_evidence").symlink_to(outside, target_is_directory=True)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")
            payload = {
                "generated_at_utc": "2026-07-10T10:01:00+00:00",
                "status": "ACCEPTED",
                "input_packet": {"large_evidence": "x" * 10000},
            }

            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                ledger.record_gateway_payload(
                    kind="gpt_decision",
                    receipt_path=root / "receipt.json",
                    payload=payload,
                )

            self.assertEqual(list(outside.rglob("*")), [])

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink unavailable")
    def test_gateway_packet_archive_rejects_existing_symlink_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(tmp)
            packet = {"large_evidence": "x" * 10000}
            canonical = execution_ledger_module._json(packet).encode("utf-8")
            digest = hashlib.sha256(canonical).hexdigest()
            archive_dir = root / "execution_evidence" / "gateway_input_packets" / digest[:2]
            archive_dir.mkdir(parents=True)
            outside = Path(outside_tmp) / "matching.json.gz"
            with gzip.open(outside, "wb") as handle:
                handle.write(canonical)
            (archive_dir / f"{digest}.json.gz").symlink_to(outside)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                ledger.record_gateway_payload(
                    kind="gpt_decision",
                    receipt_path=root / "receipt.json",
                    payload={
                        "generated_at_utc": "2026-07-10T10:01:00+00:00",
                        "status": "ACCEPTED",
                        "input_packet": packet,
                    },
                )

    @staticmethod
    def _scout_reservation_payload(
        *,
        signal_id: str,
        vehicle_id: str,
        expires_at_utc: str,
        candidate_risk_jpy: float = 100.0,
    ) -> dict[str, object]:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PREDICTIVE_SCOUT_POST_RESERVED",
            "predictive_scout": True,
            "predictive_scout_post_reserved": True,
            "predictive_scout_receipt": {
                "predictive_scout": True,
                "predictive_scout_signal_id": signal_id,
                "predictive_scout_vehicle_id": vehicle_id,
                "predictive_scout_expires_at_utc": expires_at_utc,
                "predictive_scout_fresh_actual_initial_risk_jpy": candidate_risk_jpy,
            },
            "sent": False,
        }

    def test_predictive_scout_atomic_reservation_enforces_daily_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(
                db_path=root / "ledger.db",
                report_path=root / "ledger.md",
            )
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=45)).isoformat()

            results = []
            for index in range(3):
                signal_id = f"signal-{index}"
                vehicle_id = f"vehicle-{index}"
                results.append(
                    ledger.reserve_predictive_scout_gateway_payload(
                        kind="live_order",
                        receipt_path=root / f"reservation-{index}.json",
                        payload=self._scout_reservation_payload(
                            signal_id=signal_id,
                            vehicle_id=vehicle_id,
                            expires_at_utc=expires_at,
                        ),
                        signal_id=signal_id,
                        experiment_id=f"experiment-{index}",
                        vehicle_id=vehicle_id,
                        expires_at_utc=expires_at,
                        max_daily=2,
                        max_concurrent=10,
                        broker_active_total=0,
                        candidate_risk_jpy=100.0,
                        broker_active_risk_jpy=0.0,
                        concurrent_risk_cap_jpy=1000.0,
                    )
                )

            with sqlite3.connect(root / "ledger.db") as conn:
                claim_count = conn.execute(
                    "SELECT COUNT(*) FROM predictive_scout_signal_claims"
                ).fetchone()[0]

        self.assertEqual([result.reserved for result in results], [True, True, False])
        self.assertEqual(results[-1].status, "DAILY_CAP_REACHED")
        self.assertEqual(results[-1].daily_reserved, 2)
        self.assertEqual(claim_count, 2)

    def test_predictive_scout_atomic_daily_cap_includes_legacy_sent_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(
                db_path=root / "ledger.db",
                report_path=root / "ledger.md",
            )
            ledger.record_gateway_payload(
                kind="live_order",
                receipt_path=root / "legacy-sent.json",
                payload={
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "SENT",
                    "predictive_scout": True,
                    "send_requested": True,
                    "sent": True,
                },
            )
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=45)).isoformat()

            result = ledger.reserve_predictive_scout_gateway_payload(
                kind="live_order",
                receipt_path=root / "new-reservation.json",
                payload=self._scout_reservation_payload(
                    signal_id="new-signal",
                    vehicle_id="new-vehicle",
                    expires_at_utc=expires_at,
                ),
                signal_id="new-signal",
                experiment_id="new-experiment",
                vehicle_id="new-vehicle",
                expires_at_utc=expires_at,
                max_daily=1,
                max_concurrent=2,
                broker_active_total=0,
                candidate_risk_jpy=100.0,
                broker_active_risk_jpy=0.0,
                concurrent_risk_cap_jpy=1000.0,
            )

        self.assertFalse(result.reserved)
        self.assertEqual(result.status, "DAILY_CAP_REACHED")
        self.assertEqual(result.daily_reserved, 1)

    def test_predictive_scout_atomic_reservation_serializes_distinct_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(
                db_path=root / "ledger.db",
                report_path=root / "ledger.md",
            )
            # Initialize/migrate once before the threads race on reservation.
            ledger.record_gateway_receipt(
                kind="live_order",
                receipt_path=root / "missing.json",
            )
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=45)).isoformat()

            def reserve(index: int):
                signal_id = f"parallel-signal-{index}"
                vehicle_id = f"parallel-vehicle-{index}"
                return ledger.reserve_predictive_scout_gateway_payload(
                    kind="live_order",
                    receipt_path=root / f"parallel-{index}.json",
                    payload=self._scout_reservation_payload(
                        signal_id=signal_id,
                        vehicle_id=vehicle_id,
                        expires_at_utc=expires_at,
                    ),
                    signal_id=signal_id,
                    experiment_id=f"parallel-experiment-{index}",
                    vehicle_id=vehicle_id,
                    expires_at_utc=expires_at,
                    max_daily=8,
                    max_concurrent=2,
                    broker_active_total=0,
                    candidate_risk_jpy=100.0,
                    broker_active_risk_jpy=0.0,
                    concurrent_risk_cap_jpy=1000.0,
                )

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(reserve, range(4)))
            with sqlite3.connect(root / "ledger.db") as conn:
                claim_count = conn.execute(
                    "SELECT COUNT(*) FROM predictive_scout_signal_claims"
                ).fetchone()[0]

        self.assertEqual(sum(result.reserved for result in results), 2)
        self.assertEqual(
            sum(result.status == "CONCURRENT_CAP_REACHED" for result in results),
            2,
        )
        self.assertEqual(claim_count, 2)

    def test_predictive_scout_atomic_reservation_enforces_aggregate_nav_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(
                db_path=root / "ledger.db",
                report_path=root / "ledger.md",
            )
            expires_at = (
                datetime.now(timezone.utc) + timedelta(minutes=45)
            ).isoformat()

            def reserve(signal_id: str):
                return ledger.reserve_predictive_scout_gateway_payload(
                    kind="live_order",
                    receipt_path=root / f"{signal_id}.json",
                    payload=self._scout_reservation_payload(
                        signal_id=signal_id,
                        vehicle_id=f"vehicle-{signal_id}",
                        expires_at_utc=expires_at,
                        candidate_risk_jpy=150.0,
                    ),
                    signal_id=signal_id,
                    experiment_id=f"experiment-{signal_id}",
                    vehicle_id=f"vehicle-{signal_id}",
                    expires_at_utc=expires_at,
                    max_daily=8,
                    max_concurrent=2,
                    broker_active_total=0,
                    candidate_risk_jpy=150.0,
                    broker_active_risk_jpy=0.0,
                    concurrent_risk_cap_jpy=200.0,
                )

            first = reserve("risk-1")
            second = reserve("risk-2")
            with sqlite3.connect(root / "ledger.db") as conn:
                claim_count = conn.execute(
                    "SELECT COUNT(*) FROM predictive_scout_signal_claims"
                ).fetchone()[0]

        self.assertTrue(first.reserved)
        self.assertFalse(second.reserved)
        self.assertEqual(second.status, "CONCURRENT_RISK_CAP_REACHED")
        self.assertEqual(second.aggregate_risk_jpy, 300.0)
        self.assertEqual(second.concurrent_risk_cap_jpy, 200.0)
        self.assertEqual(claim_count, 1)

    def test_predictive_scout_cold_start_parallel_initialization_fails_closed_without_lock_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=45)).isoformat()

            def reserve(index: int):
                signal_id = f"cold-signal-{index}"
                vehicle_id = f"cold-vehicle-{index}"
                return ExecutionLedger(
                    db_path=root / "cold-ledger.db",
                    report_path=root / f"cold-ledger-{index}.md",
                ).reserve_predictive_scout_gateway_payload(
                    kind="live_order",
                    receipt_path=root / f"cold-{index}.json",
                    payload=self._scout_reservation_payload(
                        signal_id=signal_id,
                        vehicle_id=vehicle_id,
                        expires_at_utc=expires_at,
                    ),
                    signal_id=signal_id,
                    experiment_id=f"cold-experiment-{index}",
                    vehicle_id=vehicle_id,
                    expires_at_utc=expires_at,
                    max_daily=8,
                    max_concurrent=2,
                    broker_active_total=0,
                    candidate_risk_jpy=100.0,
                    broker_active_risk_jpy=0.0,
                    concurrent_risk_cap_jpy=1000.0,
                )

            with ThreadPoolExecutor(max_workers=12) as executor:
                results = list(executor.map(reserve, range(12)))
            with sqlite3.connect(root / "cold-ledger.db") as conn:
                migration_rows = conn.execute(
                    "SELECT value FROM sync_state WHERE key = ?",
                    (LEGACY_EVENT_BACKFILL_MIGRATION_KEY,),
                ).fetchall()

        self.assertEqual(sum(result.reserved for result in results), 2)
        self.assertEqual(
            sum(result.status == "CONCURRENT_CAP_REACHED" for result in results),
            10,
        )
        self.assertEqual(migration_rows, [(LEGACY_EVENT_BACKFILL_MIGRATION_VERSION,)])

    def test_predictive_scout_broker_reflection_does_not_double_count_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(
                db_path=root / "ledger.db",
                report_path=root / "ledger.md",
            )
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=45)).isoformat()
            first = ledger.reserve_predictive_scout_gateway_payload(
                kind="live_order",
                receipt_path=root / "first.json",
                payload=self._scout_reservation_payload(
                    signal_id="signal-first",
                    vehicle_id="vehicle-first",
                    expires_at_utc=expires_at,
                ),
                signal_id="signal-first",
                experiment_id="experiment-first",
                vehicle_id="vehicle-first",
                expires_at_utc=expires_at,
                max_daily=8,
                max_concurrent=2,
                broker_active_total=0,
                candidate_risk_jpy=100.0,
                broker_active_risk_jpy=0.0,
                concurrent_risk_cap_jpy=1000.0,
            )
            second = ledger.reserve_predictive_scout_gateway_payload(
                kind="live_order",
                receipt_path=root / "second.json",
                payload=self._scout_reservation_payload(
                    signal_id="signal-second",
                    vehicle_id="vehicle-second",
                    expires_at_utc=expires_at,
                ),
                signal_id="signal-second",
                experiment_id="experiment-second",
                vehicle_id="vehicle-second",
                expires_at_utc=expires_at,
                max_daily=8,
                max_concurrent=2,
                broker_active_total=1,
                candidate_risk_jpy=100.0,
                broker_active_risk_jpy=100.0,
                concurrent_risk_cap_jpy=1000.0,
                broker_active_signal_ids={"signal-first"},
            )

        self.assertTrue(first.reserved)
        self.assertTrue(second.reserved)
        self.assertEqual(second.active_slots, 2)

    def test_predictive_scout_reflected_then_resolved_releases_slot_before_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(
                db_path=root / "ledger.db",
                report_path=root / "ledger.md",
            )
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=45)).isoformat()

            def reserve(
                signal_id: str,
                *,
                broker_total: int,
                broker_signals: set[str],
            ):
                return ledger.reserve_predictive_scout_gateway_payload(
                    kind="live_order",
                    receipt_path=root / f"{signal_id}.json",
                    payload=self._scout_reservation_payload(
                        signal_id=signal_id,
                        vehicle_id="shared-vehicle",
                        expires_at_utc=expires_at,
                    ),
                    signal_id=signal_id,
                    experiment_id=f"experiment-{signal_id}",
                    vehicle_id="shared-vehicle",
                    expires_at_utc=expires_at,
                    max_daily=8,
                    max_concurrent=2,
                    broker_active_total=broker_total,
                    candidate_risk_jpy=100.0,
                    broker_active_risk_jpy=100.0 * broker_total,
                    concurrent_risk_cap_jpy=1000.0,
                    broker_active_signal_ids=broker_signals,
                )

            first = reserve("signal-1", broker_total=0, broker_signals=set())
            second = reserve(
                "signal-2",
                broker_total=1,
                broker_signals={"signal-1"},
            )
            # signal-1 has resolved and disappeared. signal-2 is now broker
            # active. The durable reflection marker must stop signal-1's claim
            # from resurrecting as an in-flight shadow slot.
            third = reserve(
                "signal-3",
                broker_total=1,
                broker_signals={"signal-2"},
            )
            with sqlite3.connect(root / "ledger.db") as conn:
                reflected = conn.execute(
                    """
                    SELECT signal_id, broker_reflected_at_utc
                    FROM predictive_scout_signal_claims
                    WHERE signal_id IN ('signal-1', 'signal-2')
                    ORDER BY signal_id
                    """
                ).fetchall()

        self.assertTrue(first.reserved)
        self.assertTrue(second.reserved)
        self.assertTrue(third.reserved)
        self.assertEqual(third.active_slots, 2)
        self.assertTrue(all(row[1] for row in reflected))

    def test_predictive_scout_same_vehicle_different_signal_does_not_hide_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(
                db_path=root / "ledger.db",
                report_path=root / "ledger.md",
            )
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=45)).isoformat()
            first = ledger.reserve_predictive_scout_gateway_payload(
                kind="live_order",
                receipt_path=root / "first.json",
                payload=self._scout_reservation_payload(
                    signal_id="new-signal-first",
                    vehicle_id="shared-vehicle",
                    expires_at_utc=expires_at,
                ),
                signal_id="new-signal-first",
                experiment_id="experiment-first",
                vehicle_id="shared-vehicle",
                expires_at_utc=expires_at,
                max_daily=8,
                max_concurrent=2,
                broker_active_total=0,
                candidate_risk_jpy=100.0,
                broker_active_risk_jpy=0.0,
                concurrent_risk_cap_jpy=1000.0,
            )
            second = ledger.reserve_predictive_scout_gateway_payload(
                kind="live_order",
                receipt_path=root / "second.json",
                payload=self._scout_reservation_payload(
                    signal_id="new-signal-second",
                    vehicle_id="shared-vehicle",
                    expires_at_utc=expires_at,
                ),
                signal_id="new-signal-second",
                experiment_id="experiment-second",
                vehicle_id="shared-vehicle",
                expires_at_utc=expires_at,
                max_daily=8,
                max_concurrent=2,
                broker_active_total=1,
                candidate_risk_jpy=100.0,
                broker_active_risk_jpy=100.0,
                concurrent_risk_cap_jpy=1000.0,
                broker_active_signal_ids={"older-filled-signal"},
            )

        self.assertTrue(first.reserved)
        self.assertFalse(second.reserved)
        self.assertEqual(second.status, "CONCURRENT_CAP_REACHED")
        self.assertEqual(second.active_slots, 2)

    def test_multi_trade_close_preserves_nested_zero_pl_and_trade_financing(self) -> None:
        events = _events_from_transaction(
            {
                "id": "multi-close-1",
                "type": "ORDER_FILL",
                "time": "2026-07-10T00:00:00Z",
                "instrument": "EUR_USD",
                "units": "-2000",
                "pl": "100.0",
                "financing": "-9.0",
                "reason": "MARKET_ORDER_TRADE_CLOSE",
                "tradesClosed": [
                    {
                        "tradeID": "scout-trade",
                        "units": "1000",
                        "price": "1.17000",
                        "realizedPL": "0.0",
                        "financing": "1.0",
                    },
                    {
                        "tradeID": "normal-trade",
                        "units": "1000",
                        "price": "1.17000",
                        "realizedPL": "100.0",
                        "financing": "-10.0",
                    },
                ],
            },
            "2026-07-10T00:00:01Z",
        )

        self.assertEqual(
            [
                (
                    event["trade_id"],
                    event["realized_pl_jpy"],
                    event["financing_jpy"],
                )
                for event in events
            ],
            [
                ("scout-trade", 0.0, 1.0),
                ("normal-trade", 100.0, -10.0),
            ],
        )

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

    def test_cold_baseline_persists_nonretroactive_transaction_coverage_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")
            baseline_at = "2026-07-11T02:00:00+00:00"

            with patch.object(execution_ledger_module, "_now", return_value=baseline_at):
                summary = ledger.sync_oanda_transactions(FakeTransactionClient())

            with sqlite3.connect(root / "ledger.db") as conn:
                coverage = conn.execute(
                    "SELECT value, updated_at_utc FROM sync_state WHERE key = ?",
                    (OANDA_TRANSACTION_COVERAGE_START_KEY,),
                ).fetchone()

            self.assertEqual(summary.status, "BASELINED")
            self.assertEqual(summary.baseline_transaction_id, "100")
            self.assertEqual(coverage, (baseline_at, baseline_at))

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
                conn.execute(
                    "DELETE FROM sync_state WHERE key = ?",
                    (LEGACY_EVENT_BACKFILL_MIGRATION_KEY,),
                )
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

    def test_records_accepted_gpt_close_receipt_for_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "gpt_trader_decision.json"
            receipt.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-08T00:00:00+00:00",
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "CLOSE",
                            "selected_lane_id": None,
                            "close_trade_ids": ["T-100", "T-200"],
                        },
                        "close_gate_evidence": [
                            {
                                "trade_id": "T-100",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "unrealized_pl_jpy": -125.0,
                                "loss_side_close": True,
                                "gate_a_invalidated": True,
                                "gate_a_reason": "fresh position_thesis REVIEW_CLOSE",
                                "gate_b_standing_authorized": True,
                                "gate_b_explicit_operator_authorized": False,
                                "explicit_gate_b_required": False,
                                "profitability_p0_context_required": True,
                                "profitability_p0_context_cited": True,
                                "timing_audit_required": True,
                                "timing_evidence_cited": True,
                                "hard_timing_gate_required": False,
                                "same_direction_support_conflict": None,
                            },
                            {
                                "trade_id": "T-200",
                                "pair": "GBP_USD",
                                "side": "SHORT",
                                "unrealized_pl_jpy": -80.0,
                                "loss_side_close": True,
                                "gate_a_invalidated": False,
                                "gate_a_reason": "no reproducible Gate A invalidation evidence",
                                "gate_b_standing_authorized": False,
                                "gate_b_explicit_operator_authorized": False,
                                "explicit_gate_b_required": True,
                                "profitability_p0_context_required": False,
                                "profitability_p0_context_cited": False,
                                "timing_audit_required": False,
                                "timing_evidence_cited": False,
                                "hard_timing_gate_required": False,
                                "same_direction_support_conflict": None,
                            },
                        ],
                        "verification_issues": [],
                    }
                )
            )
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            summary = ledger.record_gateway_receipt(kind="gpt_decision", receipt_path=receipt)

            self.assertEqual(summary.events_inserted, 2)
            self.assertEqual(summary.verification_observations_inserted, 2)
            with sqlite3.connect(root / "ledger.db") as conn:
                rows = conn.execute(
                    """
                    SELECT event_type, trade_id, exit_reason, raw_json
                    FROM execution_events
                    ORDER BY trade_id
                    """
                ).fetchall()
                close_gate_rows = conn.execute(
                    """
                    SELECT subject_id, status, severity, evidence_json
                    FROM verification_observations
                    WHERE check_name='close_gate_evidence'
                    ORDER BY subject_id
                    """
                ).fetchall()
            self.assertEqual(
                [row[:3] for row in rows],
                [
                    ("GATEWAY_GPT_CLOSE_ACCEPTED", "T-100", "GPT_CLOSE_ACCEPTED"),
                    ("GATEWAY_GPT_CLOSE_ACCEPTED", "T-200", "GPT_CLOSE_ACCEPTED"),
                ],
            )
            raw_t100 = json.loads(rows[0][3])
            raw_t200 = json.loads(rows[1][3])
            self.assertEqual(raw_t100["close_gate_evidence"][0]["trade_id"], "T-100")
            self.assertEqual(raw_t200["close_gate_evidence"][0]["trade_id"], "T-200")
            self.assertEqual(close_gate_rows[0][0:3], ("T-100", "PASS", "INFO"))
            self.assertEqual(close_gate_rows[1][0:3], ("T-200", "BLOCK", "BLOCK"))
            self.assertEqual(json.loads(close_gate_rows[0][3])["gate_a_reason"], "fresh position_thesis REVIEW_CLOSE")

    def test_sync_reconciles_accepted_gpt_close_to_broker_trade_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "gpt_trader_decision.json"
            receipt.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-05T16:06:44+00:00",
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "CLOSE",
                            "selected_lane_id": None,
                            "close_trade_ids": ["T501"],
                        },
                        "verification_issues": [],
                    }
                )
            )
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            ledger.record_gateway_receipt(kind="gpt_decision", receipt_path=receipt)
            summary = ledger.sync_oanda_transactions(FakeGptCloseBrokerCloseClient(), since_transaction_id="500")
            duplicate = ledger.sync_oanda_transactions(FakeGptCloseBrokerCloseClient(), since_transaction_id="500")

            self.assertEqual(summary.reconciled_gateway_events_inserted, 1)
            self.assertEqual(duplicate.reconciled_gateway_events_inserted, 0)
            with sqlite3.connect(root / "ledger.db") as conn:
                row = conn.execute(
                    """
                    SELECT event_type, source, order_id, trade_id, pair, exit_reason,
                           related_transaction_ids_json, raw_json
                    FROM execution_events
                    WHERE event_type='GATEWAY_TRADE_CLOSE_RECONCILED'
                    """
                ).fetchone()
            self.assertEqual(row[0:6], (
                "GATEWAY_TRADE_CLOSE_RECONCILED",
                "ledger_reconcile",
                "501",
                "T501",
                "EUR_JPY",
                "GPT_CLOSE_RECONCILED",
            ))
            self.assertEqual(json.loads(row[6]), ["501", "502"])
            self.assertEqual(json.loads(row[7])["reconciled_from"][0], "GATEWAY_GPT_CLOSE_ACCEPTED")

    def test_sync_reconciles_trader_entry_broker_close_without_local_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            summary = ledger.sync_oanda_transactions(
                FakeTraderEntryBrokerCloseClient(),
                since_transaction_id="600",
            )
            duplicate = ledger.sync_oanda_transactions(
                FakeTraderEntryBrokerCloseClient(),
                since_transaction_id="600",
            )

            self.assertEqual(summary.reconciled_gateway_events_inserted, 1)
            self.assertEqual(duplicate.reconciled_gateway_events_inserted, 0)
            with sqlite3.connect(root / "ledger.db") as conn:
                row = conn.execute(
                    """
                    SELECT event_type, source, lane_id, order_id, trade_id, pair, side,
                           exit_reason, related_transaction_ids_json, raw_json
                    FROM execution_events
                    WHERE event_type='GATEWAY_TRADE_CLOSE_RECONCILED'
                    """
                ).fetchone()
            raw = json.loads(row[9])
            self.assertEqual(row[0:8], (
                "GATEWAY_TRADE_CLOSE_RECONCILED",
                "ledger_reconcile",
                "range_trader:NZD_CAD:SHORT:RANGE_ROTATION",
                "603",
                "T601",
                "NZD_CAD",
                "SHORT",
                "BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED",
            ))
            self.assertEqual(json.loads(row[8]), ["603", "604"])
            self.assertEqual(raw["reconciled_from"][0], "TRADER_ENTRY_LANE_ID")
            self.assertEqual(raw["reconcile_reason"], "NO_LOCAL_POSITION_EXECUTION_RECEIPT")

    def test_sync_does_not_reconcile_manual_broker_close_without_entry_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            summary = ledger.sync_oanda_transactions(
                FakeManualBrokerCloseClient(),
                since_transaction_id="700",
            )

            self.assertEqual(summary.reconciled_gateway_events_inserted, 0)
            with sqlite3.connect(root / "ledger.db") as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM execution_events WHERE event_type='GATEWAY_TRADE_CLOSE_RECONCILED'"
                ).fetchone()[0]
            self.assertEqual(count, 0)

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
                conn.execute(
                    "DELETE FROM sync_state WHERE key = ?",
                    (LEGACY_EVENT_BACKFILL_MIGRATION_KEY,),
                )
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

    def test_sync_backfills_entry_thesis_when_pending_sidecar_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "forecast_history.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp_utc": "2026-06-19T07:50:00Z",
                        "cycle_id": "cycle-before-fill",
                        "pair": "EUR_USD",
                        "direction": "DOWN",
                        "confidence": 0.61,
                        "horizon_min": 60,
                    }
                )
                + "\n"
            )
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")

            summary = ledger.sync_oanda_transactions(FakeBackfillFillClient(), since_transaction_id="472729")
            thesis = load_entry_thesis("472732", root)

            self.assertEqual(summary.status, "SYNCED")
            self.assertIsNotNone(thesis)
            assert thesis is not None
            self.assertEqual(thesis.pair, "EUR_USD")
            self.assertEqual(thesis.side, "SHORT")
            self.assertEqual(thesis.forecast_direction, "DOWN")
            self.assertAlmostEqual(thesis.target_price or 0.0, 1.14414)
            self.assertAlmostEqual(thesis.invalidation_price or 0.0, 1.15171)
            self.assertTrue(thesis.context_evidence["broker_backfill_from_execution_ledger"])

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
                conn.execute(
                    "DELETE FROM sync_state WHERE key = ?",
                    (LEGACY_EVENT_BACKFILL_MIGRATION_KEY,),
                )
                conn.commit()

            ledger.record_gateway_receipt(kind="live_order", receipt_path=root / "missing.json")

            with sqlite3.connect(root / "ledger.db") as conn:
                row = conn.execute(
                    "SELECT trade_id FROM execution_events WHERE event_type='ORDER_ACCEPTED'"
                ).fetchone()
            self.assertEqual(row[0], "T501")

    def test_legacy_backfill_migration_does_not_rescan_unattributed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")
            ledger.sync_oanda_transactions(
                FakeManualBrokerCloseClient(),
                since_transaction_id="700",
            )
            with sqlite3.connect(root / "ledger.db") as conn:
                conn.execute(
                    "DELETE FROM sync_state WHERE key = ?",
                    (LEGACY_EVENT_BACKFILL_MIGRATION_KEY,),
                )
                conn.commit()

            # Simulate the first initialization after upgrading an existing DB.
            # Manual/tagless OANDA rows legitimately remain without a lane.
            ledger.record_gateway_receipt(kind="live_order", receipt_path=root / "missing.json")

            with sqlite3.connect(root / "ledger.db") as conn:
                unattributed = conn.execute(
                    "SELECT COUNT(*) FROM execution_events WHERE source='oanda' AND lane_id IS NULL"
                ).fetchone()[0]
                migration_value = conn.execute(
                    "SELECT value FROM sync_state WHERE key = ?",
                    (LEGACY_EVENT_BACKFILL_MIGRATION_KEY,),
                ).fetchone()[0]
            self.assertGreater(unattributed, 0)
            self.assertEqual(migration_value, LEGACY_EVENT_BACKFILL_MIGRATION_VERSION)

            # A durable marker must keep later receipt/sync initialization from
            # reparsing the same legitimate no-lane rows forever.
            with (
                patch(
                    "quant_rabbit.execution_ledger._backfill_legacy_lane_ids",
                    side_effect=AssertionError("legacy lane backfill reran"),
                ),
                patch(
                    "quant_rabbit.execution_ledger._backfill_legacy_trade_close_ids",
                    side_effect=AssertionError("legacy trade-id backfill reran"),
                ),
                patch(
                    "quant_rabbit.execution_ledger._backfill_gateway_position_order_ids",
                    side_effect=AssertionError("legacy order-id backfill reran"),
                ),
            ):
                ledger.record_gateway_receipt(
                    kind="live_order",
                    receipt_path=root / "still-missing.json",
                )

    def test_legacy_backfill_migration_rolls_back_marker_and_updates_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")
            ledger.sync_oanda_transactions(FakeLegacyCommentClient(), since_transaction_id="200")
            with sqlite3.connect(root / "ledger.db") as conn:
                conn.execute("UPDATE execution_events SET lane_id = NULL WHERE event_type='ORDER_ACCEPTED'")
                conn.execute(
                    "DELETE FROM sync_state WHERE key = ?",
                    (LEGACY_EVENT_BACKFILL_MIGRATION_KEY,),
                )
                conn.commit()

            with patch(
                "quant_rabbit.execution_ledger._backfill_legacy_trade_close_ids",
                side_effect=RuntimeError("migration interrupted"),
            ):
                with self.assertRaisesRegex(RuntimeError, "migration interrupted"):
                    ledger.record_gateway_receipt(
                        kind="live_order",
                        receipt_path=root / "missing.json",
                    )

            with sqlite3.connect(root / "ledger.db") as conn:
                lane_after_failure = conn.execute(
                    "SELECT lane_id FROM execution_events WHERE event_type='ORDER_ACCEPTED'"
                ).fetchone()[0]
                marker_after_failure = conn.execute(
                    "SELECT value FROM sync_state WHERE key = ?",
                    (LEGACY_EVENT_BACKFILL_MIGRATION_KEY,),
                ).fetchone()
            self.assertIsNone(lane_after_failure)
            self.assertIsNone(marker_after_failure)

            ledger.record_gateway_receipt(kind="live_order", receipt_path=root / "retry-missing.json")
            with sqlite3.connect(root / "ledger.db") as conn:
                lane_after_retry = conn.execute(
                    "SELECT lane_id FROM execution_events WHERE event_type='ORDER_ACCEPTED'"
                ).fetchone()[0]
                marker_after_retry = conn.execute(
                    "SELECT value FROM sync_state WHERE key = ?",
                    (LEGACY_EVENT_BACKFILL_MIGRATION_KEY,),
                ).fetchone()[0]
            self.assertEqual(
                lane_after_retry,
                "failure_trader:GBP_USD:SHORT:BREAKOUT_FAILURE",
            )
            self.assertEqual(marker_after_retry, LEGACY_EVENT_BACKFILL_MIGRATION_VERSION)

    def test_schema_adds_predictive_scout_sent_day_expression_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = ExecutionLedger(db_path=root / "ledger.db", report_path=root / "ledger.md")
            ledger.record_gateway_receipt(kind="live_order", receipt_path=root / "missing.json")

            with sqlite3.connect(root / "ledger.db") as conn:
                plan = conn.execute(
                    """
                    EXPLAIN QUERY PLAN
                    SELECT rowid, payload_json
                    FROM gateway_receipts
                    WHERE sent = 1
                      AND substr(ts_utc, 1, 10) = ?
                    """,
                    ("2026-07-10",),
                ).fetchall()
            self.assertTrue(
                any("idx_gateway_receipts_sent_day" in str(row[3]) for row in plan),
                plan,
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


class FakeBackfillFillClient:
    def account_summary(self, *, now_utc: datetime | None = None) -> AccountSummary:
        return AccountSummary(
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            last_transaction_id="472729",
            fetched_at_utc=now_utc or datetime.now(timezone.utc),
        )

    def transactions_since_id(self, transaction_id: str) -> dict:
        return {
            "lastTransactionID": "472734",
            "transactions": [
                {
                    "id": "472732",
                    "time": "2026-06-19T08:01:32.903433014Z",
                    "type": "ORDER_FILL",
                    "orderID": "472730",
                    "instrument": "EUR_USD",
                    "units": "-6300",
                    "price": "1.14486",
                    "reason": "LIMIT_ORDER",
                    "clientOrderID": "qrv1-EURUSD-S-81b9490de070",
                    "tradeOpened": {
                        "tradeID": "472732",
                        "units": "-6300",
                        "price": "1.14486",
                        "clientExtensions": {
                            "id": "qrv1-EURUSD-S-d0dda4b89776",
                            "tag": "trader",
                            "comment": (
                                "qr-vnext lane=failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT "
                                "desk=failure_trader"
                            ),
                        },
                    },
                },
                {
                    "id": "472733",
                    "time": "2026-06-19T08:01:32.903433014Z",
                    "type": "TAKE_PROFIT_ORDER",
                    "batchID": "472732",
                    "tradeID": "472732",
                    "price": "1.14414",
                    "reason": "ON_FILL",
                },
                {
                    "id": "472734",
                    "time": "2026-06-19T08:01:32.903433014Z",
                    "type": "STOP_LOSS_ORDER",
                    "batchID": "472732",
                    "tradeID": "472732",
                    "price": "1.15171",
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


class FakeTraderEntryBrokerCloseClient:
    def account_summary(self, *, now_utc: datetime | None = None) -> AccountSummary:
        return AccountSummary(
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            last_transaction_id="600",
            fetched_at_utc=now_utc or datetime.now(timezone.utc),
        )

    def transactions_since_id(self, transaction_id: str) -> dict:
        return {
            "lastTransactionID": "604",
            "transactions": [
                {
                    "id": "601",
                    "time": "2026-06-12T17:53:49.672953223Z",
                    "type": "MARKET_ORDER",
                    "instrument": "NZD_CAD",
                    "units": "-2700",
                    "clientExtensions": {
                        "id": "qrv1-NZDCAD-S-test",
                        "tag": "trader",
                        "comment": "qr-vnext lane=range_trader:NZD_CAD:SHORT:RANGE_ROTATION",
                    },
                    "tradeClientExtensions": {
                        "id": "qrv1-NZDCAD-S-test",
                        "tag": "trader",
                        "comment": "qr-vnext lane=range_trader:NZD_CAD:SHORT:RANGE_ROTATION",
                    },
                },
                {
                    "id": "602",
                    "time": "2026-06-12T17:53:49.672953223Z",
                    "type": "ORDER_FILL",
                    "orderID": "601",
                    "instrument": "NZD_CAD",
                    "units": "-2700",
                    "price": "0.81540",
                    "reason": "MARKET_ORDER",
                    "tradeOpened": {
                        "tradeID": "T601",
                        "units": "-2700",
                        "price": "0.81540",
                        "clientExtensions": {
                            "id": "qrv1-NZDCAD-S-test",
                            "tag": "trader",
                            "comment": "qr-vnext lane=range_trader:NZD_CAD:SHORT:RANGE_ROTATION",
                        },
                    },
                },
                {
                    "id": "603",
                    "time": "2026-06-12T18:54:26.862705160Z",
                    "type": "MARKET_ORDER",
                    "instrument": "NZD_CAD",
                    "units": "2700",
                    "reason": "TRADE_CLOSE",
                    "positionFill": "REDUCE_ONLY",
                    "tradeClose": {"tradeID": "T601", "units": "ALL"},
                },
                {
                    "id": "604",
                    "time": "2026-06-12T18:54:26.862705160Z",
                    "type": "ORDER_FILL",
                    "orderID": "603",
                    "instrument": "NZD_CAD",
                    "units": "2700",
                    "price": "0.81702",
                    "reason": "MARKET_ORDER_TRADE_CLOSE",
                    "pl": "-548.9",
                    "tradesClosed": [
                        {
                            "tradeID": "T601",
                            "units": "2700",
                            "price": "0.81702",
                            "realizedPL": "-548.9",
                        }
                    ],
                },
            ],
        }


class FakeManualBrokerCloseClient:
    def account_summary(self, *, now_utc: datetime | None = None) -> AccountSummary:
        return AccountSummary(
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            last_transaction_id="700",
            fetched_at_utc=now_utc or datetime.now(timezone.utc),
        )

    def transactions_since_id(self, transaction_id: str) -> dict:
        return {
            "lastTransactionID": "704",
            "transactions": [
                {
                    "id": "701",
                    "time": "2026-06-12T17:53:49.672953223Z",
                    "type": "MARKET_ORDER",
                    "instrument": "NZD_CAD",
                    "units": "-2700",
                },
                {
                    "id": "702",
                    "time": "2026-06-12T17:53:49.672953223Z",
                    "type": "ORDER_FILL",
                    "orderID": "701",
                    "instrument": "NZD_CAD",
                    "units": "-2700",
                    "price": "0.81540",
                    "reason": "MARKET_ORDER",
                    "tradeOpened": {"tradeID": "T701", "units": "-2700", "price": "0.81540"},
                },
                {
                    "id": "703",
                    "time": "2026-06-12T18:54:26.862705160Z",
                    "type": "MARKET_ORDER",
                    "instrument": "NZD_CAD",
                    "units": "2700",
                    "reason": "TRADE_CLOSE",
                    "positionFill": "REDUCE_ONLY",
                    "tradeClose": {"tradeID": "T701", "units": "ALL"},
                },
                {
                    "id": "704",
                    "time": "2026-06-12T18:54:26.862705160Z",
                    "type": "ORDER_FILL",
                    "orderID": "703",
                    "instrument": "NZD_CAD",
                    "units": "2700",
                    "price": "0.81702",
                    "reason": "MARKET_ORDER_TRADE_CLOSE",
                    "pl": "-548.9",
                    "tradesClosed": [
                        {
                            "tradeID": "T701",
                            "units": "2700",
                            "price": "0.81702",
                            "realizedPL": "-548.9",
                        }
                    ],
                },
            ],
        }


class FakeGptCloseBrokerCloseClient:
    def account_summary(self, *, now_utc: datetime | None = None) -> AccountSummary:
        return AccountSummary(
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            last_transaction_id="500",
            fetched_at_utc=now_utc or datetime.now(timezone.utc),
        )

    def transactions_since_id(self, transaction_id: str) -> dict:
        return {
            "lastTransactionID": "502",
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
                {
                    "id": "502",
                    "time": "2026-06-05T16:06:46.821383000Z",
                    "type": "ORDER_FILL",
                    "orderID": "501",
                    "instrument": "EUR_JPY",
                    "units": "-2700",
                    "price": "165.000",
                    "reason": "MARKET_ORDER_TRADE_CLOSE",
                    "pl": "-1071.9",
                    "tradesClosed": [
                        {
                            "tradeID": "T501",
                            "units": "2700",
                            "price": "165.000",
                            "realizedPL": "-1071.9",
                        }
                    ],
                },
            ],
        }
