from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.execution_ledger import ExecutionLedger
from quant_rabbit.ledger_schema import (
    PROFITABILITY_ACCEPTANCE_INDEX_NAMES,
    ensure_profitability_acceptance_indexes,
)
from quant_rabbit.profitability_acceptance import _execution_ledger_close_findings
from quant_rabbit.verification_ledger import VerificationLedger


class LedgerProfitabilityIndexesTest(unittest.TestCase):
    def test_existing_ledger_parallel_initializers_install_exact_indexes_idempotently(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "execution_ledger.db"
            ExecutionLedger(
                db_path=db_path,
                report_path=root / "execution_ledger.md",
            )._init_db()
            with sqlite3.connect(db_path) as conn:
                for name in PROFITABILITY_ACCEPTANCE_INDEX_NAMES:
                    conn.execute(f"DROP INDEX {name}")

            start_barrier = threading.Barrier(3)

            def initialize(index: int) -> None:
                start_barrier.wait()
                if index % 2:
                    VerificationLedger(
                        db_path=db_path,
                        output_path=root / f"verification-{index}.json",
                        report_path=root / f"verification-{index}.md",
                    )._init_db()
                else:
                    ExecutionLedger(
                        db_path=db_path,
                        report_path=root / f"execution-{index}.md",
                    )._init_db()

            # Force both initializer types to converge after a real writer-lock
            # wait instead of relying on fast index DDL to happen to overlap.
            with sqlite3.connect(db_path) as lock_holder:
                lock_holder.execute("BEGIN IMMEDIATE")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(initialize, index) for index in range(2)]
                    start_barrier.wait()
                    time.sleep(0.05)
                    both_waited = all(not future.done() for future in futures)
                    lock_holder.rollback()
                    for future in futures:
                        future.result(timeout=5.0)
            self.assertTrue(both_waited)

            # Repeated initialization is the normal hot path and must be a no-op.
            ExecutionLedger(
                db_path=db_path,
                report_path=root / "execution-hot.md",
            )._init_db()
            VerificationLedger(
                db_path=db_path,
                output_path=root / "verification-hot.json",
                report_path=root / "verification-hot.md",
            )._init_db()
            with sqlite3.connect(db_path) as conn:
                index_rows = conn.execute(
                    """
                    SELECT name, sql
                    FROM sqlite_master
                    WHERE type = 'index' AND name IN (?, ?, ?)
                    ORDER BY name
                    """,
                    PROFITABILITY_ACCEPTANCE_INDEX_NAMES,
                ).fetchall()
                columns = {
                    name: tuple(
                        row[2]
                        for row in conn.execute(f"PRAGMA index_info({name})").fetchall()
                    )
                    for name in PROFITABILITY_ACCEPTANCE_INDEX_NAMES
                }

        self.assertEqual(
            [row[0] for row in index_rows],
            sorted(PROFITABILITY_ACCEPTANCE_INDEX_NAMES),
        )
        self.assertEqual(
            columns["idx_execution_events_type_trade_ts"],
            ("event_type", "trade_id", "ts_utc"),
        )
        self.assertEqual(
            columns["idx_execution_events_type_order_trade"],
            ("event_type", "order_id", "trade_id"),
        )
        self.assertEqual(
            columns["idx_verification_close_gate_subject_ts"],
            ("check_name", "subject_id", "ts_utc", "status"),
        )
        verification_sql = next(
            str(sql).lower()
            for name, sql in index_rows
            if name == "idx_verification_close_gate_subject_ts"
        )
        self.assertIn("where check_name = 'close_gate_evidence'", verification_sql)

    def test_concurrent_index_migrations_recheck_after_waiting_for_writer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            with sqlite3.connect(db_path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE execution_events (
                        event_type TEXT,
                        trade_id TEXT,
                        ts_utc TEXT,
                        order_id TEXT
                    );
                    CREATE TABLE verification_observations (
                        check_name TEXT,
                        subject_id TEXT,
                        ts_utc TEXT,
                        status TEXT
                    );
                    """
                )

            start_barrier = threading.Barrier(3)
            begin_seen = (threading.Event(), threading.Event())

            def migrate(index: int) -> tuple[tuple[str, ...], bool]:
                with sqlite3.connect(db_path, timeout=30.0) as conn:
                    conn.set_trace_callback(
                        lambda statement: begin_seen[index].set()
                        if statement.strip().upper() == "BEGIN IMMEDIATE"
                        else None
                    )
                    start_barrier.wait(timeout=2.0)
                    installed = ensure_profitability_acceptance_indexes(conn)
                    return installed, conn.in_transaction

            with sqlite3.connect(db_path) as lock_holder:
                lock_holder.execute("BEGIN IMMEDIATE")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(migrate, index) for index in range(2)]
                    try:
                        start_barrier.wait(timeout=2.0)
                        both_attempted_begin = all(
                            event.wait(timeout=2.0) for event in begin_seen
                        )
                        both_waited = all(not future.done() for future in futures)
                    finally:
                        lock_holder.rollback()
                    results = [future.result(timeout=5.0) for future in futures]

            with sqlite3.connect(db_path) as conn:
                installed_indexes = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index'"
                    ).fetchall()
                }

        self.assertTrue(both_attempted_begin)
        self.assertTrue(both_waited)
        self.assertEqual(
            sorted((result[0] for result in results), key=len),
            [(), PROFITABILITY_ACCEPTANCE_INDEX_NAMES],
        )
        self.assertTrue(all(result[1] is False for result in results))
        self.assertTrue(
            set(PROFITABILITY_ACCEPTANCE_INDEX_NAMES).issubset(installed_indexes)
        )

    def test_index_migration_rolls_back_every_new_index_and_retries_cleanly(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            with sqlite3.connect(db_path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE execution_events (
                        event_type TEXT,
                        trade_id TEXT,
                        ts_utc TEXT,
                        order_id TEXT
                    );
                    CREATE TABLE verification_observations (
                        check_name TEXT,
                        subject_id TEXT,
                        ts_utc TEXT,
                        status TEXT
                    );
                    """
                )

                def deny_second_index(
                    action: int,
                    argument_one: str | None,
                    _argument_two: str | None,
                    _database: str | None,
                    _trigger: str | None,
                ) -> int:
                    if (
                        action == sqlite3.SQLITE_CREATE_INDEX
                        and argument_one == "idx_execution_events_type_order_trade"
                    ):
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK

                conn.set_authorizer(deny_second_index)
                with self.assertRaises(sqlite3.DatabaseError):
                    ensure_profitability_acceptance_indexes(conn)
                conn.set_authorizer(None)

                after_failure = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index'"
                    ).fetchall()
                }
                tables_after_failure = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                self.assertFalse(
                    after_failure.intersection(PROFITABILITY_ACCEPTANCE_INDEX_NAMES)
                )
                self.assertEqual(
                    tables_after_failure,
                    {"execution_events", "verification_observations"},
                )
                self.assertFalse(conn.in_transaction)

                installed = ensure_profitability_acceptance_indexes(conn)
                retry_indexes = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index'"
                    ).fetchall()
                }

        self.assertEqual(installed, PROFITABILITY_ACCEPTANCE_INDEX_NAMES)
        self.assertTrue(set(PROFITABILITY_ACCEPTANCE_INDEX_NAMES).issubset(retry_indexes))

    def test_partial_legacy_schema_is_skipped_until_required_columns_exist(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "execution_ledger.db"
            with sqlite3.connect(db_path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE execution_events (
                        event_type TEXT,
                        ts_utc TEXT
                    );
                    CREATE TABLE verification_observations (
                        check_name TEXT,
                        ts_utc TEXT
                    );
                    """
                )
                first_attempt = ensure_profitability_acceptance_indexes(conn)
                self.assertEqual(first_attempt, ())

                conn.executescript(
                    """
                    ALTER TABLE execution_events ADD COLUMN trade_id TEXT;
                    ALTER TABLE execution_events ADD COLUMN order_id TEXT;
                    ALTER TABLE verification_observations ADD COLUMN subject_id TEXT;
                    ALTER TABLE verification_observations ADD COLUMN status TEXT;
                    """
                )
                second_attempt = ensure_profitability_acceptance_indexes(conn)

        self.assertEqual(second_attempt, PROFITABILITY_ACCEPTANCE_INDEX_NAMES)

    def test_close_audit_is_identical_and_correlated_evidence_probe_uses_search(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "execution_ledger.db"
            report_path = root / "execution_ledger.md"
            ledger = ExecutionLedger(db_path=db_path, report_path=report_path)
            ledger._init_db()
            with sqlite3.connect(db_path) as conn:
                for name in PROFITABILITY_ACCEPTANCE_INDEX_NAMES:
                    conn.execute(f"DROP INDEX {name}")
                _seed_realistic_close_audit(conn)

            audit_now = datetime(2026, 7, 10, 0, 0, tzinfo=timezone.utc)
            timing_path = root / "missing_execution_timing_audit.json"
            before_metrics, before_findings = _execution_ledger_close_findings(
                db_path,
                execution_timing_audit_path=timing_path,
                now_utc=audit_now,
            )

            # Every public ledger operation initializes/migrates the existing DB.
            ledger.record_gateway_receipt(
                kind="live_order",
                receipt_path=root / "missing_gateway_receipt.json",
            )
            after_metrics, after_findings = _execution_ledger_close_findings(
                db_path,
                execution_timing_audit_path=timing_path,
                now_utc=audit_now,
            )

            with sqlite3.connect(db_path) as conn:
                evidence_plan = conn.execute(
                    """
                    EXPLAIN QUERY PLAN
                    SELECT g.trade_id
                    FROM execution_events g
                    WHERE g.event_type = 'GATEWAY_TRADE_CLOSE_SENT'
                      AND EXISTS (
                          SELECT 1
                          FROM verification_observations v
                          WHERE v.check_name = 'close_gate_evidence'
                            AND v.subject_id = g.trade_id
                            AND (
                                v.evidence_json IS NULL
                                OR instr(
                                    replace(v.evidence_json, ' ', ''),
                                    '"reason":"close_gate_evidence_missing"'
                                ) = 0
                            )
                            AND (
                                v.ts_utc = g.ts_utc
                                OR EXISTS (
                                    SELECT 1
                                    FROM execution_events accepted
                                    WHERE accepted.event_type = 'GATEWAY_GPT_CLOSE_ACCEPTED'
                                      AND accepted.trade_id = g.trade_id
                                      AND accepted.ts_utc = v.ts_utc
                                      AND accepted.ts_utc <= g.ts_utc
                                )
                            )
                      )
                    """
                ).fetchall()
                order_plan = conn.execute(
                    """
                    EXPLAIN QUERY PLAN
                    SELECT 1
                    FROM execution_events
                    WHERE event_type = 'GATEWAY_TRADE_CLOSE_SENT'
                      AND order_id = 'O-pass'
                      AND trade_id = 'T-pass'
                    """
                ).fetchall()

        self.assertEqual(before_metrics, after_metrics)
        self.assertEqual(before_findings, after_findings)
        self.assertEqual(after_metrics["recent_loss_closes"], 3)
        self.assertEqual(after_metrics["recent_close_gate_unverified_loss_closes"], 2)
        self.assertEqual(after_metrics["recent_close_gate_missing_loss_closes"], 1)
        self.assertEqual(after_metrics["recent_close_gate_not_passing_loss_closes"], 1)

        evidence_details = [str(row[3]) for row in evidence_plan]
        self.assertTrue(
            any(
                "SEARCH v USING INDEX idx_verification_close_gate_subject_ts" in detail
                for detail in evidence_details
            ),
            evidence_details,
        )
        self.assertFalse(
            any("SCAN v" in detail for detail in evidence_details),
            evidence_details,
        )
        self.assertTrue(
            any(
                "idx_execution_events_type_trade_ts" in detail
                for detail in evidence_details
            ),
            evidence_details,
        )
        order_details = [str(row[3]) for row in order_plan]
        self.assertTrue(
            any(
                "idx_execution_events_type_order_trade" in detail
                for detail in order_details
            ),
            order_details,
        )


def _seed_realistic_close_audit(conn: sqlite3.Connection) -> None:
    unrelated_sql = """
        INSERT INTO verification_observations(
            observation_uid, ts_utc, source, source_path, subject_type,
            subject_id, check_name, status, severity, metric_value,
            metric_unit, evidence_json, inserted_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    conn.executemany(
        unrelated_sql,
        (
            (
                f"unrelated:{index}",
                "2026-07-09T00:00:00+00:00",
                "fixture",
                None,
                "artifact",
                f"artifact-{index}",
                "artifact_readable",
                "PASS",
                "INFO",
                None,
                None,
                "{}",
                "2026-07-09T00:00:00+00:00",
            )
            for index in range(20_000)
        ),
    )

    close_cases = (
        ("T-pass", "O-pass", "PASS", {}),
        ("T-block", "O-block", "BLOCK", {"gate_a_invalidated": False}),
        (
            "T-missing",
            "O-missing",
            "BLOCK",
            {"reason": "close_gate_evidence_missing"},
        ),
    )
    for offset, (trade_id, order_id, status, evidence) in enumerate(close_cases):
        accepted_ts = f"2026-07-09T10:00:{offset * 10:02d}+00:00"
        sent_ts = f"2026-07-09T10:00:{offset * 10 + 1:02d}+00:00"
        closed_ts = f"2026-07-09T10:00:{offset * 10 + 2:02d}+00:00"
        accepted_payload = {
            "status": "ACCEPTED",
            "decision": {"action": "CLOSE", "close_trade_ids": [trade_id]},
            "close_gate_evidence": [{"trade_id": trade_id, **evidence}],
        }
        _insert_event(
            conn,
            event_uid=f"accepted:{trade_id}",
            ts_utc=accepted_ts,
            source="gateway",
            event_type="GATEWAY_GPT_CLOSE_ACCEPTED",
            trade_id=trade_id,
            order_id=None,
            realized_pl_jpy=None,
            exit_reason="GPT_CLOSE_ACCEPTED",
            raw_json=json.dumps(accepted_payload, sort_keys=True),
        )
        _insert_event(
            conn,
            event_uid=f"sent:{trade_id}",
            ts_utc=sent_ts,
            source="gateway",
            event_type="GATEWAY_TRADE_CLOSE_SENT",
            trade_id=trade_id,
            order_id=order_id,
            realized_pl_jpy=None,
            exit_reason="GPT_CLOSE",
            raw_json="{}",
        )
        _insert_event(
            conn,
            event_uid=f"closed:{trade_id}",
            ts_utc=closed_ts,
            source="oanda",
            event_type="TRADE_CLOSED",
            trade_id=trade_id,
            order_id=order_id,
            realized_pl_jpy=-100.0 - offset,
            exit_reason="MARKET_ORDER_TRADE_CLOSE",
            raw_json="{}",
        )
        conn.execute(
            unrelated_sql,
            (
                f"close-gate:{trade_id}",
                accepted_ts,
                "gpt_decision",
                None,
                "close_gate",
                trade_id,
                "close_gate_evidence",
                status,
                "INFO" if status == "PASS" else "BLOCK",
                None,
                None,
                json.dumps(evidence, sort_keys=True),
                accepted_ts,
            ),
        )

    _insert_event(
        conn,
        event_uid="gateway:latest",
        ts_utc="2026-07-10T00:00:00+00:00",
        source="gateway",
        event_type="GATEWAY_POSITION_NO_ACTION",
        trade_id=None,
        order_id=None,
        realized_pl_jpy=None,
        exit_reason="HOLD_PROTECTED",
        raw_json="{}",
    )


def _insert_event(
    conn: sqlite3.Connection,
    *,
    event_uid: str,
    ts_utc: str,
    source: str,
    event_type: str,
    trade_id: str | None,
    order_id: str | None,
    realized_pl_jpy: float | None,
    exit_reason: str,
    raw_json: str,
) -> None:
    conn.execute(
        """
        INSERT INTO execution_events(
            event_uid, ts_utc, source, event_type, lane_id, order_id, trade_id,
            client_order_id, pair, side, units, price, tp, sl, realized_pl_jpy,
            financing_jpy, exit_reason, oanda_transaction_id,
            related_transaction_ids_json, raw_json, inserted_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_uid,
            ts_utc,
            source,
            event_type,
            "fixture:EUR_USD:LONG:RANGE_ROTATION",
            order_id,
            trade_id,
            None,
            "EUR_USD",
            "LONG",
            None,
            None,
            None,
            None,
            realized_pl_jpy,
            None,
            exit_reason,
            None,
            "[]",
            raw_json,
            ts_utc,
        ),
    )


if __name__ == "__main__":
    unittest.main()
