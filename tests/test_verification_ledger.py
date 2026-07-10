from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import quant_rabbit.verification_ledger as verification_ledger_module
from quant_rabbit.cli import main
from quant_rabbit.verification_ledger import VerificationLedger, VerificationLedgerSummary


class VerificationLedgerTest(unittest.TestCase):
    def test_records_verification_blockers_and_effect_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            _seed_execution_events(root / "execution_ledger.db")

            summary = VerificationLedger(
                db_path=root / "execution_ledger.db",
                output_path=root / "verification.json",
                report_path=root / "verification.md",
            ).run(
                snapshot_path=files["snapshot"],
                order_intents_path=files["intents"],
                gpt_decision_path=files["gpt"],
                live_order_path=files["live_order"],
                position_execution_path=files["position_execution"],
                thesis_evolution_path=files["thesis_evolution"],
                position_thesis_path=files["position_thesis"],
                forecast_persistence_path=files["forecast_persistence"],
                ai_backtest_path=files["ai_backtest"],
                outcome_mart_path=files["outcome_mart"],
                post_trade_learning_path=files["post_trade_learning"],
                ai_attack_advice_path=files["ai_attack_advice"],
                learning_audit_path=files["learning_audit"],
                window_hours=168,
                now=datetime(2026, 6, 3, 0, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertGreater(summary.blocking_observations, 0)
            self.assertEqual(summary.closed_trades, 2)
            self.assertAlmostEqual(summary.net_jpy, 50.0)
            self.assertAlmostEqual(summary.profit_factor or 0.0, 1.5)
            self.assertAlmostEqual(summary.win_rate or 0.0, 0.5)
            self.assertAlmostEqual(summary.expectancy_jpy or 0.0, 25.0)

            with sqlite3.connect(root / "execution_ledger.db") as conn:
                unverifiable = conn.execute(
                    "SELECT COUNT(*) FROM verification_observations WHERE status='UNVERIFIABLE'"
                ).fetchone()[0]
                net_row = conn.execute(
                    """
                    SELECT metric_value, sample_size
                    FROM effect_measurements
                    WHERE segment='all' AND metric_name='net_jpy'
                    """
                ).fetchone()
                learning_rows = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM verification_observations
                    WHERE source IN ('ai_backtest', 'outcome_mart', 'post_trade_learning', 'ai_attack_advice', 'learning_audit')
                      AND check_name != 'artifact_readable'
                    """
                ).fetchone()[0]
                advice_delta = conn.execute(
                    """
                    SELECT metric_value
                    FROM verification_observations
                    WHERE source='ai_attack_advice'
                      AND check_name='recommended_learning_influence'
                    """
                ).fetchone()[0]
                close_gate_row = conn.execute(
                    """
                    SELECT status, severity, evidence_json
                    FROM verification_observations
                    WHERE source='gpt_decision'
                      AND check_name='close_gate_evidence'
                      AND subject_id='555'
                    """
                ).fetchone()

            self.assertGreater(unverifiable, 0)
            self.assertEqual(net_row, (50.0, 2))
            self.assertGreaterEqual(learning_rows, 5)
            self.assertEqual(advice_delta, 8.0)
            self.assertIsNotNone(close_gate_row)
            self.assertEqual(close_gate_row[0], "BLOCK")
            self.assertEqual(close_gate_row[1], "BLOCK")
            close_gate_evidence = json.loads(close_gate_row[2])
            self.assertEqual(close_gate_evidence["gate_a_reason"], "M15 BOS_UP")
            self.assertTrue(close_gate_evidence["explicit_gate_b_required"])
            self.assertIn("INSUFFICIENT_SAMPLE_LT_30", (root / "verification.md").read_text())
            self.assertIn("Learning Evidence", (root / "verification.md").read_text())
            packet = json.loads((root / "verification.json").read_text())
            self.assertEqual(packet["status"], "BLOCKED")
            self.assertEqual(packet["effect_metrics"]["closed_trades"], 2)
            self.assertTrue(packet["blocking_evidence"][0]["evidence_ref"].startswith("verification:"))
            self.assertEqual(packet["contract"]["json_packet_is_trader_readable"], True)

    def test_cli_records_verification_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            _seed_execution_events(root / "execution_ledger.db")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "verification-ledger-audit",
                        "--db",
                        str(root / "execution_ledger.db"),
                        "--output",
                        str(root / "verification.json"),
                        "--report",
                        str(root / "verification.md"),
                        "--snapshot",
                        str(files["snapshot"]),
                        "--order-intents",
                        str(files["intents"]),
                        "--gpt-decision",
                        str(files["gpt"]),
                        "--live-order",
                        str(files["live_order"]),
                        "--position-execution",
                        str(files["position_execution"]),
                        "--thesis-evolution",
                        str(files["thesis_evolution"]),
                        "--position-thesis",
                        str(files["position_thesis"]),
                        "--forecast-persistence",
                        str(files["forecast_persistence"]),
                        "--ai-backtest",
                        str(files["ai_backtest"]),
                        "--outcome-mart",
                        str(files["outcome_mart"]),
                        "--post-trade-learning",
                        str(files["post_trade_learning"]),
                        "--ai-attack-advice",
                        str(files["ai_attack_advice"]),
                        "--learning-audit",
                        str(files["learning_audit"]),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "BLOCKED")
            self.assertEqual(payload["output_path"], str(root / "verification.json"))
            self.assertGreater(payload["observations_inserted"], 0)
            self.assertGreater(payload["measurements_inserted"], 0)
            self.assertTrue((root / "verification.json").exists())

    def test_backfills_close_gate_evidence_from_gateway_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            db_path = root / "execution_ledger.db"
            _seed_execution_events(db_path)
            decision_path = root / "gpt_trader_decision.accepted.json"
            receipt_payload = {
                "generated_at_utc": "2026-06-21T00:00:00+00:00",
                "status": "ACCEPTED",
                "decision": {
                    "action": "CLOSE",
                    "close_trade_ids": ["T-100"],
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
                    }
                ],
            }
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE gateway_receipts (
                        receipt_uid TEXT PRIMARY KEY,
                        ts_utc TEXT,
                        kind TEXT,
                        status TEXT,
                        sent INTEGER,
                        lane_id TEXT,
                        lane_ids_json TEXT,
                        path TEXT,
                        payload_json TEXT,
                        inserted_at_utc TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO gateway_receipts(
                        receipt_uid, ts_utc, kind, status, sent, lane_id,
                        lane_ids_json, path, payload_json, inserted_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "receipt:T-100",
                        "2026-06-21T00:00:00+00:00",
                        "gpt_decision",
                        "ACCEPTED",
                        0,
                        None,
                        "[]",
                        str(decision_path),
                        json.dumps(receipt_payload),
                        "2026-06-21T00:00:01+00:00",
                    ),
                )

            VerificationLedger(
                db_path=db_path,
                output_path=root / "verification.json",
                report_path=root / "verification.md",
            ).run(
                snapshot_path=files["snapshot"],
                order_intents_path=files["intents"],
                gpt_decision_path=files["gpt"],
                live_order_path=files["live_order"],
                position_execution_path=files["position_execution"],
                thesis_evolution_path=files["thesis_evolution"],
                position_thesis_path=files["position_thesis"],
                forecast_persistence_path=files["forecast_persistence"],
                ai_backtest_path=files["ai_backtest"],
                outcome_mart_path=files["outcome_mart"],
                post_trade_learning_path=files["post_trade_learning"],
                ai_attack_advice_path=files["ai_attack_advice"],
                learning_audit_path=files["learning_audit"],
                now=datetime(2026, 6, 21, 0, 5, tzinfo=timezone.utc),
            )

            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    """
                    SELECT ts_utc, source, source_path, status, severity, evidence_json
                    FROM verification_observations
                    WHERE check_name='close_gate_evidence'
                      AND subject_id='T-100'
                    """
                ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "2026-06-21T00:00:00+00:00")
        self.assertEqual(row[1], "gpt_decision")
        self.assertEqual(row[2], str(decision_path))
        self.assertEqual(row[3:5], ("PASS", "INFO"))
        self.assertEqual(json.loads(row[5])["gate_a_reason"], "fresh position_thesis REVIEW_CLOSE")

    def test_marks_accepted_gpt_close_missing_close_gate_evidence_from_ledger_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            db_path = root / "execution_ledger.db"
            with sqlite3.connect(db_path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE sync_state (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL
                    );
                    CREATE TABLE execution_events (
                        event_uid TEXT,
                        ts_utc TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        trade_id TEXT,
                        pair TEXT,
                        side TEXT,
                        exit_reason TEXT,
                        realized_pl_jpy REAL,
                        raw_json TEXT
                    );
                    """
                )
                conn.execute(
                    """
                    INSERT INTO sync_state(key, value, updated_at_utc)
                    VALUES ('last_oanda_transaction_id', '104', '2026-06-21T00:00:00+00:00')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO execution_events(
                        event_uid, ts_utc, event_type, trade_id, pair, side,
                        exit_reason, realized_pl_jpy, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "gateway:gpt_decision:missing:T-missing",
                        "2026-06-21T00:00:00+00:00",
                        "GATEWAY_GPT_CLOSE_ACCEPTED",
                        "T-missing",
                        "EUR_USD",
                        "LONG",
                        "GPT_CLOSE_ACCEPTED",
                        None,
                        json.dumps(
                            {
                                "generated_at_utc": "2026-06-21T00:00:00+00:00",
                                "status": "ACCEPTED",
                                "decision": {
                                    "action": "CLOSE",
                                    "close_trade_ids": ["T-missing"],
                                },
                                "verification_issues": [],
                            }
                        ),
                    ),
                )

            VerificationLedger(
                db_path=db_path,
                output_path=root / "verification.json",
                report_path=root / "verification.md",
            ).run(
                snapshot_path=files["snapshot"],
                order_intents_path=files["intents"],
                gpt_decision_path=files["gpt"],
                live_order_path=files["live_order"],
                position_execution_path=files["position_execution"],
                thesis_evolution_path=files["thesis_evolution"],
                position_thesis_path=files["position_thesis"],
                forecast_persistence_path=files["forecast_persistence"],
                ai_backtest_path=files["ai_backtest"],
                outcome_mart_path=files["outcome_mart"],
                post_trade_learning_path=files["post_trade_learning"],
                ai_attack_advice_path=files["ai_attack_advice"],
                learning_audit_path=files["learning_audit"],
                now=datetime(2026, 6, 21, 0, 5, tzinfo=timezone.utc),
            )

            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    """
                    SELECT ts_utc, source, status, severity, evidence_json
                    FROM verification_observations
                    WHERE check_name='close_gate_evidence'
                      AND subject_id='T-missing'
                    """
                ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "2026-06-21T00:00:00+00:00")
        self.assertEqual(row[1], "execution_ledger")
        self.assertEqual(row[2:4], ("BLOCK", "BLOCK"))
        evidence = json.loads(row[4])
        self.assertEqual(evidence["reason"], "close_gate_evidence_missing")
        self.assertEqual(evidence["event_uid"], "gateway:gpt_decision:missing:T-missing")

    def test_legacy_gateway_receipt_backfill_reader_runs_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            db_path = root / "execution_ledger.db"
            _seed_execution_events(db_path)
            _seed_legacy_gateway_receipt(
                db_path,
                trade_id="T-once",
                decision_path=root / "legacy-gpt-decision.json",
            )
            ledger = VerificationLedger(
                db_path=db_path,
                output_path=root / "verification.json",
                report_path=root / "verification.md",
            )

            with mock.patch(
                "quant_rabbit.verification_ledger._gateway_receipt_close_gate_observations",
                wraps=verification_ledger_module._gateway_receipt_close_gate_observations,
            ) as receipt_reader:
                _run_ledger(ledger, files, now=datetime(2026, 6, 21, 0, 5, tzinfo=timezone.utc))
                _run_ledger(ledger, files, now=datetime(2026, 6, 21, 0, 10, tzinfo=timezone.utc))

            self.assertEqual(receipt_reader.call_count, 1)
            with sqlite3.connect(db_path) as conn:
                marker = conn.execute(
                    "SELECT status, cursor FROM verification_backfill_state"
                ).fetchone()
                normalized = conn.execute(
                    """
                    SELECT status
                    FROM verification_observations
                    WHERE check_name='close_gate_evidence'
                      AND subject_id='T-once'
                    """
                ).fetchone()

        self.assertEqual(marker[0], "COMPLETE")
        self.assertGreaterEqual(int(marker[1]), 1)
        self.assertEqual(normalized[0], "PASS")

    def test_legacy_backfill_captures_receipt_committed_before_atomic_scan_and_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            db_path = root / "execution_ledger.db"
            _seed_execution_events(db_path)
            _seed_legacy_gateway_receipt(
                db_path,
                trade_id="T-before",
                decision_path=root / "before.json",
            )
            ledger = VerificationLedger(
                db_path=db_path,
                output_path=root / "verification.json",
                report_path=root / "verification.md",
            )
            original_effect_metrics = verification_ledger_module._effect_metrics

            def append_receipt_during_old_scan_insert_window(*args: object, **kwargs: object) -> dict[str, object]:
                _append_legacy_gateway_receipt(
                    db_path,
                    trade_id="T-raced",
                    decision_path=root / "raced.json",
                    generated_at_utc="2026-06-21T00:01:00+00:00",
                )
                return original_effect_metrics(*args, **kwargs)

            with mock.patch(
                "quant_rabbit.verification_ledger._effect_metrics",
                side_effect=append_receipt_during_old_scan_insert_window,
            ):
                first_summary = _run_ledger(
                    ledger,
                    files,
                    now=datetime(2026, 6, 21, 0, 5, tzinfo=timezone.utc),
                )

            with sqlite3.connect(db_path) as conn:
                first_observation_count = conn.execute(
                    "SELECT COUNT(*) FROM verification_observations"
                ).fetchone()[0]
                marker = conn.execute(
                    "SELECT status, cursor FROM verification_backfill_state"
                ).fetchone()
                raced = conn.execute(
                    """
                    SELECT status
                    FROM verification_observations
                    WHERE check_name='close_gate_evidence'
                      AND subject_id='T-raced'
                    """
                ).fetchone()

            self.assertEqual(first_summary.observations_inserted, first_observation_count)
            self.assertEqual(marker[0], "COMPLETE")
            self.assertGreaterEqual(int(marker[1]), 2)
            self.assertEqual(raced[0], "PASS")

            _run_ledger(
                ledger,
                files,
                now=datetime(2026, 6, 21, 0, 10, tzinfo=timezone.utc),
            )
            with sqlite3.connect(db_path) as conn:
                raced_count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM verification_observations
                    WHERE check_name='close_gate_evidence'
                      AND subject_id='T-raced'
                    """
                ).fetchone()[0]
            self.assertEqual(raced_count, 1)

    def test_legacy_gateway_receipt_backfill_failure_does_not_write_complete_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            db_path = root / "execution_ledger.db"
            _seed_execution_events(db_path)
            _seed_legacy_gateway_receipt(
                db_path,
                trade_id="T-retry",
                decision_path=root / "legacy-gpt-decision.json",
            )
            ledger = VerificationLedger(
                db_path=db_path,
                output_path=root / "verification.json",
                report_path=root / "verification.md",
            )

            with mock.patch(
                "quant_rabbit.verification_ledger._gateway_receipt_close_gate_observations",
                side_effect=sqlite3.OperationalError("forced legacy receipt read failure"),
            ):
                with self.assertRaises(sqlite3.OperationalError):
                    _run_ledger(ledger, files, now=datetime(2026, 6, 21, 0, 5, tzinfo=timezone.utc))

            with sqlite3.connect(db_path) as conn:
                marker_count = conn.execute(
                    "SELECT COUNT(*) FROM verification_backfill_state"
                ).fetchone()[0]
            self.assertEqual(marker_count, 0)

            _run_ledger(ledger, files, now=datetime(2026, 6, 21, 0, 10, tzinfo=timezone.utc))
            with sqlite3.connect(db_path) as conn:
                marker = conn.execute(
                    "SELECT status FROM verification_backfill_state"
                ).fetchone()
                normalized = conn.execute(
                    """
                    SELECT status
                    FROM verification_observations
                    WHERE check_name='close_gate_evidence'
                      AND subject_id='T-retry'
                    """
                ).fetchone()

        self.assertEqual(marker[0], "COMPLETE")
        self.assertEqual(normalized[0], "PASS")

    def test_event_only_multi_trade_close_normalizes_each_event_to_its_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            db_path = root / "execution_ledger.db"
            _seed_multi_trade_close_events(db_path)
            ledger = VerificationLedger(
                db_path=db_path,
                output_path=root / "verification.json",
                report_path=root / "verification.md",
            )

            _run_ledger(ledger, files, now=datetime(2026, 6, 21, 0, 5, tzinfo=timezone.utc))

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT subject_id, status, severity, evidence_json
                    FROM verification_observations
                    WHERE source='execution_ledger'
                      AND check_name='close_gate_evidence'
                    ORDER BY subject_id, status
                    """
                ).fetchall()

        self.assertEqual([(row[0], row[1], row[2]) for row in rows], [
            ("T-event-1", "PASS", "INFO"),
            ("T-event-2", "PASS", "INFO"),
        ])
        self.assertEqual(
            [json.loads(row[3])["trade_id"] for row in rows],
            ["T-event-1", "T-event-2"],
        )


def _fixtures(root: Path) -> dict[str, Path]:
    now = "2026-06-03T00:00:00+00:00"
    files = {
        "snapshot": root / "broker_snapshot.json",
        "intents": root / "order_intents.json",
        "gpt": root / "gpt_decision.json",
        "live_order": root / "live_order.json",
        "position_execution": root / "position_execution.json",
        "thesis_evolution": root / "thesis_evolution_report.json",
        "position_thesis": root / "position_thesis_report.json",
        "forecast_persistence": root / "forecast_persistence_report.json",
        "ai_backtest": root / "ai_test_bot_backtest.json",
        "outcome_mart": root / "outcome_mart.json",
        "post_trade_learning": root / "post_trade_learning.json",
        "ai_attack_advice": root / "ai_attack_advice.json",
        "learning_audit": root / "learning_audit.json",
    }
    files["snapshot"].write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "positions": [{"trade_id": "101", "pair": "EUR_USD", "side": "LONG", "owner": "trader"}],
                "orders": [],
                "account": {"last_transaction_id": "104"},
            }
        )
    )
    files["intents"].write_text(
        json.dumps(
            {
                "results": [
                    {
                        "lane_id": "lane:EUR_USD:LONG",
                        "status": "LIVE_READY",
                        "risk_issues": [],
                        "strategy_issues": [],
                        "live_blockers": [],
                    }
                ]
            }
        )
    )
    files["gpt"].write_text(
        json.dumps(
            {
                "status": "REJECTED",
                "decision": {"action": "TRADE"},
                "verification_issues": [
                    {
                        "severity": "BLOCK",
                        "code": "ENTRY_THESIS_REPAIR_REQUIRED",
                        "message": "unverifiable",
                    }
                ],
                "close_gate_evidence": [
                    {
                        "trade_id": "555",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "unrealized_pl_jpy": -120.0,
                        "loss_side_close": True,
                        "gate_a_invalidated": True,
                        "gate_a_reason": "M15 BOS_UP",
                        "gate_b_standing_authorized": False,
                        "gate_b_explicit_operator_authorized": False,
                        "explicit_gate_b_required": True,
                        "profitability_p0_context_required": False,
                        "profitability_p0_context_cited": False,
                        "timing_audit_required": False,
                        "timing_evidence_cited": False,
                        "hard_timing_gate_required": False,
                        "same_direction_support_conflict": None,
                    }
                ],
            }
        )
    )
    files["live_order"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "status": "SENT",
                "sent": True,
                "lane_id": "lane:EUR_USD:LONG",
                "entry_thesis_record": {"status": "FAILED", "issue": "write failed"},
                "risk_issues": [],
                "strategy_issues": [],
            }
        )
    )
    files["position_execution"].write_text(json.dumps({"generated_at_utc": now, "status": "NO_ACTION"}))
    files["thesis_evolution"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "entry_thesis_coverage": {
                    "missing": 1,
                    "missing_trade_ids": ["101"],
                    "blocking": True,
                    "blocking_trade_ids": ["101"],
                },
                "evolutions": [],
            }
        )
    )
    files["position_thesis"].write_text(json.dumps({"generated_at_utc": now, "assessments": []}))
    files["forecast_persistence"].write_text(json.dumps({"generated_at_utc": now, "verdicts": []}))
    files["ai_backtest"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "status": "TARGET_COVERAGE_CERTIFIED",
                "live_permission": False,
                "training_days": 12,
                "min_train_trades": 5,
                "max_active_buckets": 4,
                "blockers": [],
                "summary": {
                    "validation_days": 12,
                    "selected_trades": 40,
                    "total_managed_net_jpy": 1200.0,
                    "profit_factor": 1.4,
                },
            }
        )
    )
    files["outcome_mart"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "read_only": True,
                "live_permission": False,
                "source_coverage": {
                    "archive_outcomes": 20,
                    "execution_ledger_outcomes": 2,
                    "story_observations": 4,
                },
                "condition_validation": {
                    "status": "CONDITION_WALK_FORWARD_READY",
                    "min_prior_outcomes": 5,
                    "eligible_outcomes": 22,
                    "validated_outcomes": 10,
                    "directional_hit_rate_pct": 60.0,
                },
            }
        )
    )
    files["post_trade_learning"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "status": "READY_FOR_REVIEW",
                "candidates": [{"source_ref": "live_order", "recommendation": "NO_PROFILE_CHANGE"}],
                "profile_update_candidates": [],
                "blockers": [],
            }
        )
    )
    files["ai_attack_advice"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "status": "ATTACK_PARTIAL",
                "read_only": True,
                "live_permission": False,
                "recommended_now_lane_ids": ["lane:EUR_USD:LONG"],
                "lanes": [
                    {
                        "lane_id": "lane:EUR_USD:LONG",
                        "learning_influences": ["ai_backtest_research_positive_edge"],
                        "learning_score_delta": 8.0,
                        "learning_influence_details": [
                            {
                                "source": "ai_backtest",
                                "influence": "ai_backtest_research_positive_edge",
                                "score_delta": 8.0,
                                "source_status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                            }
                        ],
                    }
                ],
            }
        )
    )
    files["learning_audit"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "status": "LEARNING_AUDIT_WARN",
                "min_effect_sample": 30,
                "blockers": [],
                "warnings": ["recommended lanes are influenced by learning; audit review required"],
                "learning_influence": {
                    "influenced_lanes": 1,
                    "total_learning_score_delta": 8.0,
                    "lanes": [
                        {
                            "lane_id": "lane:EUR_USD:LONG",
                            "learning_influences": ["ai_backtest_research_positive_edge"],
                            "learning_score_delta": 8.0,
                        }
                    ],
                },
                "effect_metrics": {
                    "closed_trades": 2,
                    "net_jpy": 50.0,
                    "profit_factor": 1.5,
                    "expectancy_jpy": 25.0,
                },
            }
        )
    )
    return files


def _seed_execution_events(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            );
            CREATE TABLE execution_events (
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                pair TEXT,
                side TEXT,
                exit_reason TEXT,
                realized_pl_jpy REAL
            );
            """
        )
        conn.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES ('last_oanda_transaction_id', '104', '2026-06-03T00:00:00+00:00')"
        )
        conn.execute(
            """
            INSERT INTO execution_events(ts_utc, event_type, pair, side, exit_reason, realized_pl_jpy)
            VALUES ('2026-06-02T23:00:00.123456789Z', 'TRADE_CLOSED', 'EUR_USD', 'LONG', 'TAKE_PROFIT_ORDER', 150.0)
            """
        )
        conn.execute(
            """
            INSERT INTO execution_events(ts_utc, event_type, pair, side, exit_reason, realized_pl_jpy)
            VALUES ('2026-06-02T22:00:00.123456789Z', 'TRADE_CLOSED', 'EUR_USD', 'SHORT', 'MARKET_ORDER_TRADE_CLOSE', -100.0)
            """
        )


def _run_ledger(
    ledger: VerificationLedger,
    files: dict[str, Path],
    *,
    now: datetime,
) -> VerificationLedgerSummary:
    return ledger.run(
        snapshot_path=files["snapshot"],
        order_intents_path=files["intents"],
        gpt_decision_path=files["gpt"],
        live_order_path=files["live_order"],
        position_execution_path=files["position_execution"],
        thesis_evolution_path=files["thesis_evolution"],
        position_thesis_path=files["position_thesis"],
        forecast_persistence_path=files["forecast_persistence"],
        ai_backtest_path=files["ai_backtest"],
        outcome_mart_path=files["outcome_mart"],
        post_trade_learning_path=files["post_trade_learning"],
        ai_attack_advice_path=files["ai_attack_advice"],
        learning_audit_path=files["learning_audit"],
        now=now,
    )


def _close_gate_evidence(trade_id: str) -> dict[str, object]:
    return {
        "trade_id": trade_id,
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
    }


def _seed_legacy_gateway_receipt(path: Path, *, trade_id: str, decision_path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE gateway_receipts (
                receipt_uid TEXT PRIMARY KEY,
                ts_utc TEXT,
                kind TEXT,
                status TEXT,
                sent INTEGER,
                lane_id TEXT,
                lane_ids_json TEXT,
                path TEXT,
                payload_json TEXT,
                inserted_at_utc TEXT
            )
            """
        )
    _append_legacy_gateway_receipt(
        path,
        trade_id=trade_id,
        decision_path=decision_path,
        generated_at_utc="2026-06-21T00:00:00+00:00",
    )


def _append_legacy_gateway_receipt(
    path: Path,
    *,
    trade_id: str,
    decision_path: Path,
    generated_at_utc: str,
) -> None:
    receipt_payload = {
        "generated_at_utc": generated_at_utc,
        "status": "ACCEPTED",
        "decision": {"action": "CLOSE", "close_trade_ids": [trade_id]},
        "close_gate_evidence": [_close_gate_evidence(trade_id)],
    }
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO gateway_receipts(
                receipt_uid, ts_utc, kind, status, sent, lane_id,
                lane_ids_json, path, payload_json, inserted_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"receipt:{trade_id}",
                generated_at_utc,
                "gpt_decision",
                "ACCEPTED",
                0,
                None,
                "[]",
                str(decision_path),
                json.dumps(receipt_payload),
                generated_at_utc,
            ),
        )


def _seed_multi_trade_close_events(path: Path) -> None:
    close_trade_ids = ["T-event-1", "T-event-2"]
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            );
            CREATE TABLE execution_events (
                event_uid TEXT,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                exit_reason TEXT,
                realized_pl_jpy REAL,
                raw_json TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO sync_state(key, value, updated_at_utc)
            VALUES ('last_oanda_transaction_id', '104', '2026-06-21T00:00:00+00:00')
            """
        )
        for trade_id in close_trade_ids:
            raw_payload = {
                "generated_at_utc": "2026-06-21T00:00:00+00:00",
                "status": "ACCEPTED",
                "decision": {
                    "action": "CLOSE",
                    "close_trade_ids": close_trade_ids,
                },
                "close_gate_evidence": [_close_gate_evidence(trade_id)],
            }
            conn.execute(
                """
                INSERT INTO execution_events(
                    event_uid, ts_utc, event_type, trade_id, pair, side,
                    exit_reason, realized_pl_jpy, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"gateway:gpt_decision:close:{trade_id}",
                    "2026-06-21T00:00:00+00:00",
                    "GATEWAY_GPT_CLOSE_ACCEPTED",
                    trade_id,
                    "EUR_USD",
                    "LONG",
                    "GPT_CLOSE_ACCEPTED",
                    None,
                    json.dumps(raw_payload),
                ),
            )


if __name__ == "__main__":
    unittest.main()
