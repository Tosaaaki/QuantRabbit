from __future__ import annotations

import hashlib
import json
import os
import plistlib
import shutil
import subprocess
import tempfile
import unittest
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from quant_rabbit.qr_trader_run_watchdog import WatchdogPaths, run_watchdog


ROOT = Path(__file__).resolve().parents[1]
PLIST = ROOT / "scripts" / "guardian" / "com.quantrabbit.qr-trader-run-watchdog.plist"
WATCHDOG_SOURCE = ROOT / "src" / "quant_rabbit" / "qr_trader_run_watchdog.py"
HEARTBEAT_SOURCE = ROOT / "scripts" / "qr_trader_heartbeat_watch.sh"


class QRTraderRunWatchdogTest(unittest.TestCase):
    def test_active_recent_run_is_ok(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:52:00+00:00"))
            _write_journal(root, _dt("2026-07-01T02:51:00+00:00"))
            _write_memory(automation_dir, _dt("2026-07-01T02:53:00+00:00"))

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "OK")
            self.assertFalse(payload["missed_expected_window"])
            self.assertLess(payload["minutes_since_last_run"], 15)
            self.assertTrue(paths.output_json.exists())
            self.assertTrue(paths.output_report.exists())
            self.assertTrue(paths.output_log.exists())
            self.assertEqual(payload["last_trader_run_source"], "qr_trader_automation_memory.timestamp")
            self.assertEqual(payload["last_trader_run_path"], str(automation_dir / "memory.md"))
            self.assertEqual(
                payload["accepted_timestamp_candidate"]["source"],
                "qr_trader_automation_memory.timestamp",
            )

    def test_active_stale_run_is_stale(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-06-30T20:30:00+00:00"))
            _write_journal(root, _dt("2026-06-30T20:29:00+00:00"))
            _write_memory(automation_dir, _dt("2026-06-30T20:31:00+00:00"))

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "STALE")
            self.assertTrue(payload["missed_expected_window"])
            self.assertIn("QR_TRADER_RUN_STALE", _issue_codes(payload))
            self.assertEqual(payload["severity"], "P0")

    def test_valid_sealed_ai_supervision_is_preferred_run_evidence(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:59:00+00:00"))
            _write_ai_regime_supervision(
                root,
                generated_at=_dt("2026-07-01T02:30:00+00:00"),
                body_overrides={
                    "pairs": {
                        "EUR_USD": {
                            "mode": "CAUTION",
                            "reason": "bounded volatility transition review",
                            "expires_at_utc": "2026-07-01T08:30:00+00:00",
                        }
                    }
                },
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "OK")
            self.assertEqual(payload["expected_cadence_minutes"], 360)
            self.assertEqual(payload["grace_minutes"], 15)
            self.assertEqual(payload["threshold_minutes"], 375)
            self.assertEqual(payload["last_trader_run_at"], "2026-07-01T02:30:00+00:00")
            self.assertEqual(
                payload["last_trader_run_source"],
                "ai_regime_supervision.generated_at_utc",
            )
            self.assertEqual(
                payload["latest_run_evidence"]["evidence_priority"],
                "AI_REGIME_SUPERVISION",
            )
            self.assertTrue(
                payload["latest_run_evidence"]["ai_regime_supervision"]["valid_sealed_artifact"]
            )

    def test_existing_invalid_ai_supervision_fails_closed_without_legacy_fallback(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        for authority, broker_mutation in (("TRADE", False), ("NONE", True)):
            with self.subTest(authority=authority, broker_mutation=broker_mutation), tempfile.TemporaryDirectory() as tmp:
                root, automation_dir, paths = _fixture(tmp, now=now)
                _write_automation(automation_dir)
                _write_decision(root, _dt("2026-07-01T02:58:00+00:00"))
                _write_ai_regime_supervision(
                    root,
                    generated_at=_dt("2026-07-01T02:59:00+00:00"),
                    authority=authority,
                    broker_mutation_allowed=broker_mutation,
                )

                payload = run_watchdog(paths=paths, now_utc=now)

                self.assertEqual(payload["status"], "BROKEN")
                self.assertIsNone(payload["last_trader_run_source"])
                self.assertEqual(
                    payload["latest_run_evidence"]["evidence_priority"],
                    "INVALID_AI_REGIME_SUPERVISION_FAIL_CLOSED",
                )
                self.assertFalse(
                    payload["latest_run_evidence"]["ai_regime_supervision"]["valid_sealed_artifact"]
                )
                self.assertIn("AI_REGIME_SUPERVISION_INVALID", _issue_codes(payload))

    def test_ai_supervision_consumer_rejects_invalid_boundary_shape_and_future_clock(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        cases = (
            ({"schema_version": True}, "contract or schema_version"),
            ({"ai_role": "ORDER_DECISION"}, "ai_role"),
            ({"live_permission": True}, "live_permission"),
            ({"pairs": []}, "pairs"),
        )
        for mutation, expected_reason in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root, automation_dir, paths = _fixture(tmp, now=now)
                _write_automation(automation_dir)
                _write_decision(root, _dt("2026-07-01T02:58:00+00:00"))
                _write_ai_regime_supervision(
                    root,
                    generated_at=_dt("2026-07-01T02:59:00+00:00"),
                    body_overrides=mutation,
                )

                payload = run_watchdog(paths=paths, now_utc=now)

                supervision = payload["latest_run_evidence"]["ai_regime_supervision"]
                self.assertFalse(supervision["valid_sealed_artifact"])
                self.assertIn(expected_reason, supervision["invalid_reason"])
                self.assertEqual(
                    payload["latest_run_evidence"]["evidence_priority"],
                    "INVALID_AI_REGIME_SUPERVISION_FAIL_CLOSED",
                )
                self.assertEqual(payload["status"], "BROKEN")
                self.assertIsNone(payload["last_trader_run_source"])
                self.assertIn("AI_REGIME_SUPERVISION_INVALID", _issue_codes(payload))

        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:58:00+00:00"))
            _write_ai_regime_supervision(
                root,
                generated_at=_dt("2026-07-01T03:00:01+00:00"),
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            supervision = payload["latest_run_evidence"]["ai_regime_supervision"]
            self.assertFalse(supervision["valid_sealed_artifact"])
            self.assertEqual(supervision["invalid_reason"], "generated_at_utc is future-dated")
            self.assertEqual(payload["status"], "BROKEN")
            self.assertIsNone(payload["last_trader_run_source"])
            self.assertEqual(
                payload["latest_run_evidence"]["evidence_priority"],
                "INVALID_AI_REGIME_SUPERVISION_FAIL_CLOSED",
            )
            self.assertIn("AI_REGIME_SUPERVISION_INVALID", _issue_codes(payload))

    def test_sealed_malformed_ai_supervision_pair_row_is_broken(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        valid_row = {
            "mode": "GO",
            "reason": "bounded six-hour review",
            "expires_at_utc": "2026-07-01T08:59:00+00:00",
        }
        malformed_rows = (
            {**valid_row, "units": 1000},
            {key: value for key, value in valid_row.items() if key != "reason"},
            {**valid_row, "mode": "PAUSE"},
            {**valid_row, "mode": "go"},
            {**valid_row, "reason": " "},
            {**valid_row, "reason": " padded reason "},
            {**valid_row, "expires_at_utc": "2026-07-01T08:59:00"},
            {**valid_row, "expires_at_utc": "2026-07-01T09:00:00+00:00"},
        )
        for row in malformed_rows:
            with self.subTest(row=row), tempfile.TemporaryDirectory() as tmp:
                root, automation_dir, paths = _fixture(tmp, now=now)
                _write_automation(automation_dir)
                _write_decision(root, _dt("2026-07-01T02:58:00+00:00"))
                _write_ai_regime_supervision(
                    root,
                    generated_at=_dt("2026-07-01T02:59:00+00:00"),
                    body_overrides={"pairs": {"EUR_USD": row}},
                )

                payload = run_watchdog(paths=paths, now_utc=now)

                supervision = payload["latest_run_evidence"]["ai_regime_supervision"]
                self.assertEqual(payload["status"], "BROKEN")
                self.assertFalse(supervision["valid_sealed_artifact"])
                self.assertIsNone(payload["last_trader_run_source"])
                self.assertIn("AI_REGIME_SUPERVISION_INVALID", _issue_codes(payload))

    def test_weekend_paused_trader_automation_is_ok(self) -> None:
        now = _dt("2026-07-03T21:30:00+00:00")  # Saturday 06:30 JST.
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir, status="PAUSED")
            _write_weekend_state(paths.weekend_state)
            _write_decision(root, _dt("2026-07-03T20:00:00+00:00"))
            _write_journal(root, _dt("2026-07-03T19:59:00+00:00"))

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "OK")
            self.assertEqual(payload["runtime_status"], "OK")
            self.assertFalse(payload["missed_expected_window"])
            self.assertTrue(payload["weekend_pause"]["active"])
            self.assertNotIn("QR_TRADER_AUTOMATION_INACTIVE", _issue_codes(payload))
            self.assertNotIn("QR_TRADER_RUN_STALE", _issue_codes(payload))

    def test_midweek_paused_trader_automation_is_broken(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir, status="PAUSED")
            _write_weekend_state(paths.weekend_state)
            _write_decision(root, _dt("2026-07-01T02:58:00+00:00"))

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "BROKEN")
            self.assertFalse(payload["weekend_pause"]["active"])
            self.assertIn("QR_TRADER_AUTOMATION_INACTIVE", _issue_codes(payload))

    def test_wrong_model_cadence_and_cwd_is_broken(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(
                automation_dir,
                model="gpt-5.4-mini",
                rrule="RRULE:FREQ=MINUTELY;INTERVAL=20",
                cwds=["/Users/tossaki/App/QuantRabbit"],
            )
            _write_decision(root, _dt("2026-07-01T02:58:00+00:00"))

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "BROKEN")
            codes = _issue_codes(payload)
            self.assertIn("QR_TRADER_AUTOMATION_WRONG_MODEL", codes)
            self.assertIn("QR_TRADER_AUTOMATION_WRONG_CADENCE", codes)
            self.assertIn("QR_TRADER_AUTOMATION_WRONG_CWD", codes)

    def test_active_guardian_reduce_receipt_not_consumed_before_expiry_is_p1(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T01:00:00+00:00"))
            _write_guardian_receipt(
                root,
                action="REDUCE",
                generated_at=_dt("2026-07-01T01:10:00+00:00"),
                expires_at=_dt("2026-07-01T02:25:00+00:00"),
                consumed=False,
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "BLOCKED")
            self.assertEqual(payload["runtime_status"], "OK")
            self.assertEqual(payload["overall_status"], "BLOCKED")
            self.assertEqual(payload["issue_status"], "P1")
            guardian_issues = payload["guardian_receipt"]["issues"]
            self.assertEqual(guardian_issues[0]["code"], "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER")
            self.assertEqual(guardian_issues[0]["severity"], "P1")
            self.assertTrue(payload["guardian_receipt"]["high_urgency_action"])

    def test_active_guardian_receipt_before_next_run_is_dependency_not_issue(self) -> None:
        now = _dt("2026-07-01T02:30:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:00:00+00:00"))
            _write_guardian_receipt(
                root,
                action="HOLD",
                generated_at=_dt("2026-07-01T02:10:00+00:00"),
                expires_at=_dt("2026-07-01T03:05:00+00:00"),
                consumed=False,
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "OK")
            self.assertEqual(payload["guardian_receipt"]["issues"], [])
            self.assertTrue(payload["guardian_receipt"]["dependency_before_next_run"])
            self.assertTrue(payload["guardian_receipt"]["will_expire_before_next_run"])

    def test_expired_guardian_receipt_not_consumed_emits_issue(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:55:00+00:00"))
            _write_guardian_receipt(
                root,
                action="REDUCE",
                generated_at=_dt("2026-07-01T01:10:00+00:00"),
                expires_at=_dt("2026-07-01T02:25:00+00:00"),
                consumed=False,
                lifecycle="EXPIRED",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            guardian_issues = payload["guardian_receipt"]["issues"]
            self.assertEqual(payload["status"], "BLOCKED")
            self.assertEqual(payload["runtime_status"], "OK")
            self.assertEqual(payload["overall_status"], "BLOCKED")
            self.assertEqual(payload["issue_status"], "P1")
            self.assertEqual(len(guardian_issues), 1)
            self.assertEqual(guardian_issues[0]["code"], "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER")
            self.assertIn("receipt_lifecycle=EXPIRED", guardian_issues[0]["message"])

    def test_expired_guardian_receipt_consumed_does_not_emit_issue(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:55:00+00:00"))
            _write_guardian_receipt(
                root,
                action="REDUCE",
                generated_at=_dt("2026-07-01T01:10:00+00:00"),
                expires_at=_dt("2026-07-01T02:25:00+00:00"),
                consumed=True,
                lifecycle="EXPIRED",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["guardian_receipt"]["issues"], [])

    def test_expired_hold_and_no_action_unconsumed_are_warn(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        for action in ("HOLD", "NO_ACTION"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmp:
                root, automation_dir, paths = _fixture(tmp, now=now)
                _write_automation(automation_dir)
                _write_decision(root, _dt("2026-07-01T02:55:00+00:00"))
                _write_guardian_receipt(
                    root,
                    action=action,
                    generated_at=_dt("2026-07-01T01:10:00+00:00"),
                    expires_at=_dt("2026-07-01T02:25:00+00:00"),
                    consumed=False,
                    lifecycle="EXPIRED",
                )

                payload = run_watchdog(paths=paths, now_utc=now)

                self.assertEqual(payload["guardian_receipt"]["issues"][0]["severity"], "WARN")

    def test_expired_reduce_harvest_and_cancel_pending_unconsumed_are_p1(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        for action in ("REDUCE", "HARVEST", "CANCEL_PENDING"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmp:
                root, automation_dir, paths = _fixture(tmp, now=now)
                _write_automation(automation_dir)
                _write_decision(root, _dt("2026-07-01T02:55:00+00:00"))
                _write_guardian_receipt(
                    root,
                    action=action,
                    generated_at=_dt("2026-07-01T01:10:00+00:00"),
                    expires_at=_dt("2026-07-01T02:25:00+00:00"),
                    consumed=False,
                    lifecycle="EXPIRED",
                )

                payload = run_watchdog(paths=paths, now_utc=now)

                self.assertEqual(payload["guardian_receipt"]["issues"][0]["severity"], "P1")

    def test_expired_emergency_or_margin_receipt_is_p0(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:55:00+00:00"))
            _write_guardian_receipt(
                root,
                action="HOLD",
                generated_at=_dt("2026-07-01T01:10:00+00:00"),
                expires_at=_dt("2026-07-01T02:25:00+00:00"),
                consumed=False,
                lifecycle="EXPIRED",
                thesis_state="EMERGENCY",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["guardian_receipt"]["issues"][0]["severity"], "P0")
            self.assertTrue(payload["guardian_receipt"]["emergency_or_margin_risk"])

    def test_expired_p1_margin_warning_preserves_source_semantics(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        for source_severity, expected_issue_severity in (("P1", "P1"), (None, "P0")):
            with self.subTest(source_severity=source_severity), tempfile.TemporaryDirectory() as tmp:
                root, automation_dir, paths = _fixture(tmp, now=now)
                _write_automation(automation_dir)
                _write_decision(root, _dt("2026-07-01T02:55:00+00:00"))
                receipt_path = root / "data" / "guardian_action_receipt.json"
                _write_guardian_receipt(
                    root,
                    action="HOLD",
                    generated_at=_dt("2026-07-01T01:10:00+00:00"),
                    expires_at=_dt("2026-07-01T02:25:00+00:00"),
                    consumed=False,
                    lifecycle="EXPIRED",
                )
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt["selected_event"].update(
                    {
                        "event_type": "MARGIN_PRESSURE",
                        "action_hint": "HOLD",
                        "details": {
                            "nav_jpy": 100_000.0,
                            "margin_used_jpy": 92_000.0,
                            "margin_available_jpy": 8_000.0,
                            "max_margin_utilization_pct": 95.0,
                            "fresh_entry_risk_block_active": False,
                            "fresh_entry_risk_block_reason": "MARGIN_PRESSURE",
                            "fresh_entry_risk_observation_only": True,
                            "fresh_entry_margin_contract": "QR_GUARDIAN_P1_MARGIN_WARNING_V1",
                        },
                    }
                )
                if source_severity is None:
                    receipt["selected_event"].pop("severity", None)
                else:
                    receipt["selected_event"]["severity"] = source_severity
                receipt_path.write_text(
                    json.dumps(receipt) + "\n",
                    encoding="utf-8",
                )

                payload = run_watchdog(paths=paths, now_utc=now)

                issue = payload["guardian_receipt"]["issues"][0]
                self.assertEqual(issue["severity"], expected_issue_severity)
                self.assertEqual(
                    payload["issue_status"],
                    expected_issue_severity,
                )
                self.assertEqual(issue["event_type"], "MARGIN_PRESSURE")
                self.assertEqual(issue["event_severity"], source_severity)
                self.assertEqual(issue["event_action_hint"], "HOLD")
                self.assertEqual(
                    issue["event_details"]["fresh_entry_margin_contract"],
                    "QR_GUARDIAN_P1_MARGIN_WARNING_V1",
                )

    def test_missing_canonical_receipt_scans_archive_for_expired_unconsumed(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:55:00+00:00"))
            archive = root / "data" / "guardian_action_receipts" / "receipt_EXPIRED.json"
            _write_guardian_receipt(
                root,
                action="REDUCE",
                generated_at=_dt("2026-07-01T01:10:00+00:00"),
                expires_at=_dt("2026-07-01T02:25:00+00:00"),
                consumed=False,
                lifecycle="EXPIRED",
                path=archive,
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertFalse(payload["guardian_receipt"]["exists"])
            self.assertEqual(payload["guardian_receipt"]["archive_receipts_checked"], 1)
            self.assertEqual(payload["guardian_receipt"]["issues"][0]["code"], "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER")

    def test_duplicate_current_and_archive_receipt_does_not_double_count(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:55:00+00:00"))
            event_id = "receipt-duplicate"
            _write_guardian_receipt(
                root,
                action="REDUCE",
                generated_at=_dt("2026-07-01T01:10:00+00:00"),
                expires_at=_dt("2026-07-01T02:25:00+00:00"),
                consumed=False,
                lifecycle="EXPIRED",
                event_id=event_id,
            )
            _write_guardian_receipt(
                root,
                action="REDUCE",
                generated_at=_dt("2026-07-01T01:10:00+00:00"),
                expires_at=_dt("2026-07-01T02:25:00+00:00"),
                consumed=False,
                lifecycle="EXPIRED",
                event_id=event_id,
                path=root / "data" / "guardian_action_receipts" / "receipt_EXPIRED.json",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["guardian_receipt"]["receipts_checked"], 2)
            self.assertEqual(len(payload["guardian_receipt"]["receipt_summaries"]), 1)
            self.assertEqual(len(payload["guardian_receipt"]["issues"]), 1)

    def test_receipt_expiry_is_rejected_as_last_trader_run(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_guardian_receipt(
                root,
                action="HOLD",
                generated_at=_dt("2026-07-01T02:40:00+00:00"),
                expires_at=_dt("2026-07-01T02:59:00+00:00"),
                consumed=False,
                lifecycle="EXPIRED",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertIsNone(payload["last_trader_run_at"])
            self.assertEqual(payload["status"], "UNKNOWN")
            rejected_sources = {item["source"] for item in payload["rejected_timestamp_candidates"]}
            self.assertIn("guardian_action_receipt.expires_at_utc", rejected_sources)
            self.assertIn("guardian_action_receipt.generated_at_utc", rejected_sources)

    def test_receipt_timestamps_inside_memory_are_rejected_as_last_trader_run(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            _, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            (automation_dir / "memory.md").write_text(
                "\n".join(
                    [
                        "# memory",
                        "- Guardian receipt expires_at_utc=2026-07-01T02:59:00+00:00 remains open.",
                        "- Guardian receipt generated_at_utc=2026-07-01T02:58:00+00:00 was reviewed.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertIsNone(payload["last_trader_run_at"])
            self.assertIsNone(payload["accepted_timestamp_candidate"])
            memory_rejections = [
                item
                for item in payload["rejected_timestamp_candidates"]
                if item["source"] == "qr_trader_automation_memory.timestamp"
            ]
            reasons = {item["rejected_reason"] for item in memory_rejections}
            self.assertIn("receipt expiry timestamp is not trader-run evidence", reasons)
            self.assertIn("receipt or guardian generated_at timestamp is not trader-run evidence", reasons)

    def test_json_and_code_block_timestamps_inside_memory_are_rejected(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            _, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            (automation_dir / "memory.md").write_text(
                "\n".join(
                    [
                        "# memory",
                        '> {"generated_at_utc": "2026-07-01T02:59:00+00:00"}',
                        "```json",
                        '{"ts": "2026-07-01T02:58:00+00:00", "action": "WAIT"}',
                        "```",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertIsNone(payload["last_trader_run_at"])
            memory_rejections = [
                item
                for item in payload["rejected_timestamp_candidates"]
                if item["source"] == "qr_trader_automation_memory.timestamp"
            ]
            reasons = {item["rejected_reason"] for item in memory_rejections}
            self.assertIn("timestamp appears inside a JSON snippet or quoted JSON block", reasons)
            self.assertIn("timestamp appears inside a code block", reasons)

    def test_guardian_review_and_trigger_deadline_are_rejected_as_last_trader_run(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            (root / "docs" / "guardian_action_review.md").write_text(
                "# Guardian Action Review\n\n- Generated at UTC: `2026-07-01T02:59:00+00:00`\n",
                encoding="utf-8",
            )
            _write_json(
                root / "data" / "guardian_trigger_contract.json",
                {
                    "generated_at_utc": "2026-07-01T02:57:00+00:00",
                    "next_review_deadline_utc": "2026-07-01T02:59:00+00:00",
                },
            )
            _write_json(
                paths.output_json,
                {
                    "generated_at_utc": "2026-07-01T02:58:00+00:00",
                    "status": "OK",
                },
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertIsNone(payload["last_trader_run_at"])
            rejected_sources = {item["source"] for item in payload["rejected_timestamp_candidates"]}
            self.assertIn("guardian_action_review.timestamp", rejected_sources)
            self.assertIn("guardian_trigger_contract.next_review_deadline_utc", rejected_sources)
            self.assertIn("qr_trader_run_watchdog.generated_at_utc", rejected_sources)

    def test_trader_journal_ts_is_accepted_as_last_trader_run(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_journal(root, _dt("2026-07-01T02:59:00+00:00"))

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["last_trader_run_at"], "2026-07-01T02:59:00+00:00")
            self.assertEqual(payload["last_trader_run_source"], "trader_journal.ts")

    def test_automation_memory_timestamp_is_accepted_as_last_trader_run(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            _, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_memory(automation_dir, _dt("2026-07-01T02:58:00+00:00"))

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["last_trader_run_at"], "2026-07-01T02:58:00+00:00")
            self.assertEqual(payload["last_trader_run_source"], "qr_trader_automation_memory.timestamp")
            self.assertEqual(
                payload["accepted_timestamp_candidate"]["timestamp_utc"],
                "2026-07-01T02:58:00+00:00",
            )

    def test_automation_memory_hourly_trader_cycle_heading_is_accepted(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            _, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            (automation_dir / "memory.md").write_text(
                "2026-07-01T02:58Z hourly trader cycle:\n"
                "- Guardian receipt expiring 2026-07-01T03:20:00+00:00 was reviewed.\n",
                encoding="utf-8",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["last_trader_run_at"], "2026-07-01T02:58:00+00:00")
            memory_rejections = [
                item
                for item in payload["rejected_timestamp_candidates"]
                if item["source"] == "qr_trader_automation_memory.timestamp"
            ]
            self.assertEqual(len(memory_rejections), 1)
            self.assertEqual(
                memory_rejections[0]["rejected_reason"],
                "receipt expiry timestamp is not trader-run evidence",
            )

    def test_automation_memory_timestamp_heading_accepts_section_run_marker(self) -> None:
        now = _dt("2026-07-09T16:51:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            _, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            (automation_dir / "memory.md").write_text(
                "## 2026-07-09T16:16:30Z\n\n"
                "- Ran one deeper hourly QR vNext trader cycle from `/Users/tossaki/App/QuantRabbit-live`.\n"
                "- Latest guardian receipt expires `2026-07-09T17:02:13.015955+00:00`.\n",
                encoding="utf-8",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "OK")
            self.assertEqual(payload["last_trader_run_at"], "2026-07-09T16:16:30+00:00")
            self.assertEqual(payload["last_trader_run_source"], "qr_trader_automation_memory.timestamp")
            memory_rejections = [
                item
                for item in payload["rejected_timestamp_candidates"]
                if item["source"] == "qr_trader_automation_memory.timestamp"
            ]
            self.assertEqual(len(memory_rejections), 1)
            self.assertEqual(
                memory_rejections[0]["rejected_reason"],
                "receipt expiry timestamp is not trader-run evidence",
            )

    def test_automation_memory_attempted_cycle_heading_counts_as_wake_evidence(self) -> None:
        now = _dt("2026-07-09T20:30:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            _, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            (automation_dir / "memory.md").write_text(
                "## 2026-07-09T20:04:52Z\n\n"
                "- Attempted one deeper hourly QR vNext trader cycle from "
                "`/Users/tossaki/App/QuantRabbit-live`, but stopped before normal routing "
                "because the live runtime concurrency gate fired.\n",
                encoding="utf-8",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "OK")
            self.assertFalse(payload["missed_expected_window"])
            self.assertEqual(payload["last_trader_run_at"], "2026-07-09T20:04:52+00:00")
            self.assertEqual(payload["last_trader_run_source"], "qr_trader_automation_memory.timestamp")

    def test_automation_memory_timestamp_heading_without_run_marker_is_rejected(self) -> None:
        now = _dt("2026-07-09T16:51:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            _, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            (automation_dir / "memory.md").write_text(
                "## 2026-07-09T16:16:30Z\n\n"
                "- Guardian receipt was reviewed; no trader cycle marker is present.\n",
                encoding="utf-8",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertIsNone(payload["last_trader_run_at"])
            memory_rejections = [
                item
                for item in payload["rejected_timestamp_candidates"]
                if item["source"] == "qr_trader_automation_memory.timestamp"
            ]
            self.assertEqual(len(memory_rejections), 1)
            self.assertEqual(
                memory_rejections[0]["rejected_reason"],
                "automation memory timestamp is not attached to a qr-trader run marker",
            )

    def test_decision_artifact_generated_at_requires_trader_decision_shape(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_json(
                root / "data" / "codex_trader_decision_response.json",
                {
                    "generated_at_utc": "2026-07-01T02:58:00+00:00",
                    "action": "HOLD",
                    "receipt_status": "ACCEPTED",
                },
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertIsNone(payload["last_trader_run_at"])
            rejected_sources = {item["source"] for item in payload["rejected_timestamp_candidates"]}
            self.assertIn("decision_response.generated_at_utc", rejected_sources)

            _write_json(
                root / "data" / "codex_trader_decision_response.json",
                {
                    "generated_at_utc": "2026-07-01T02:58:00+00:00",
                    "action": "WAIT",
                    "market_read_first": {"naked_read": {"tape_state": "RANGE"}},
                    "twenty_minute_plan": {"horizon_minutes": 60},
                },
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["last_trader_run_at"], "2026-07-01T02:58:00+00:00")
            self.assertEqual(payload["last_trader_run_source"], "decision_response.generated_at_utc")

    def test_report_generated_at_is_accepted_as_trader_run_evidence(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            (root / "docs" / "autotrade_cycle_report.md").write_text(
                "# Autotrade Cycle Report\n\n- Generated at UTC: `2026-07-01T02:56:00+00:00`\n",
                encoding="utf-8",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["last_trader_run_at"], "2026-07-01T02:56:00+00:00")
            self.assertEqual(payload["last_trader_run_source"], "autotrade_report.generated_at_utc")

    def test_gpt_decision_report_generated_at_requires_trader_report_shape(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            (root / "docs" / "gpt_trader_decision_report.md").write_text(
                "# Guardian Action Review\n\n- Generated at UTC: `2026-07-01T02:56:00+00:00`\n",
                encoding="utf-8",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertIsNone(payload["last_trader_run_at"])
            rejected_sources = {item["source"] for item in payload["rejected_timestamp_candidates"]}
            self.assertIn("gpt_decision_report.generated_at_utc", rejected_sources)

            (root / "docs" / "gpt_trader_decision_report.md").write_text(
                "# GPT Trader Decision Report\n\n- Generated at UTC: `2026-07-01T02:56:00+00:00`\n",
                encoding="utf-8",
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["last_trader_run_at"], "2026-07-01T02:56:00+00:00")
            self.assertEqual(payload["last_trader_run_source"], "gpt_decision_report.generated_at_utc")

    def test_expired_acknowledged_receipt_stops_repeating_active_issue(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:55:00+00:00"))
            _write_guardian_receipt(
                root,
                action="HOLD",
                generated_at=_dt("2026-07-01T01:10:00+00:00"),
                expires_at=_dt("2026-07-01T02:25:00+00:00"),
                consumed=False,
                lifecycle="EXPIRED",
                event_id="receipt-expired-ack",
            )
            _write_consumption(
                root,
                issue_code="GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
                event_id="receipt-expired-ack",
                action="HOLD",
                lifecycle="EXPIRED",
                classification="EXPIRED_ACKNOWLEDGED",
                normal_routing_allowed=True,
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["guardian_receipt"]["issues"], [])
            self.assertEqual(
                payload["guardian_receipt"]["receipt_summaries"][0]["acknowledgement_classification"],
                "EXPIRED_ACKNOWLEDGED",
            )

    def test_needs_operator_review_remains_visible_without_consumed_flag(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:55:00+00:00"))
            _write_guardian_receipt(
                root,
                action="HOLD",
                generated_at=_dt("2026-07-01T01:10:00+00:00"),
                expires_at=_dt("2026-07-01T02:25:00+00:00"),
                consumed=False,
                lifecycle="EXPIRED",
                event_id="receipt-needs-review",
                thesis_state="EMERGENCY",
            )
            _write_consumption(
                root,
                issue_code="GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW",
                event_id="receipt-needs-review",
                action="HOLD",
                lifecycle="EXPIRED",
                classification="NEEDS_OPERATOR_REVIEW",
                normal_routing_allowed=False,
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            issue = payload["guardian_receipt"]["issues"][0]
            self.assertEqual(issue["code"], "GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW")
            self.assertFalse(issue["consumed_by_trader"])
            self.assertFalse(issue["normal_routing_allowed"])

    def test_missing_evidence_is_unknown(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            _, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "UNKNOWN")
            self.assertIn("QR_TRADER_RUN_EVIDENCE_MISSING", _issue_codes(payload))

    def test_watchdog_never_calls_oanda_or_codex_by_default(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        source = WATCHDOG_SOURCE.read_text()
        self.assertNotIn("Oanda", source)
        self.assertNotIn("requests.", source)
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:59:00+00:00"))

            with mock.patch.object(urllib.request, "urlopen") as urlopen, mock.patch.object(
                subprocess, "run"
            ) as subprocess_run:
                payload = run_watchdog(paths=paths, now_utc=now, env={"QR_TRADER_WATCHDOG_CAN_WAKE": "0"})

            urlopen.assert_not_called()
            subprocess_run.assert_not_called()
            self.assertFalse(payload["execution_boundary"]["calls_oanda"])
            self.assertFalse(payload["execution_boundary"]["runs_codex_by_default"])
            self.assertTrue(payload["no_live_side_effects"])
            self.assertFalse(payload["codex_exec_enabled"])
            self.assertFalse(payload["broker_writes_enabled"])
            self.assertFalse(payload["execution_boundary"]["broker_writes_enabled"])
            self.assertEqual(payload["environment"]["QR_TRADER_WATCHDOG_CAN_WAKE"], "0")
            report = paths.output_report.read_text(encoding="utf-8")
            self.assertIn("no_live_side_effects=true", report)

    def test_heartbeat_uses_six_hour_supervisor_ceiling_and_sealed_artifact(self) -> None:
        source = HEARTBEAT_SOURCE.read_text(encoding="utf-8")

        self.assertIn('QR_HEARTBEAT_MAX_SILENCE:-65100', source)
        self.assertIn('data/ai_regime_supervision.json', source)
        self.assertIn('QR_AI_REGIME_SUPERVISION_V1', source)
        self.assertIn('type(v) is int', source)
        self.assertIn('REGIME_REVIEW_AND_PERIODIC_TUNING_ONLY', source)
        self.assertIn('ai_order_authority', source)
        self.assertIn('live_permission', source)
        self.assertIn('broker_mutation_allowed', source)
        self.assertIn('isinstance(p.get("pairs"),dict)', source)
        self.assertIn('if [ "$supervision_present" -eq 0 ]', source)

    def test_launchd_plist_has_safe_defaults_and_lints_when_available(self) -> None:
        payload = plistlib.loads(PLIST.read_bytes())

        self.assertEqual(payload["Label"], "com.quantrabbit.qr-trader-run-watchdog")
        self.assertEqual(payload["WorkingDirectory"], "/Users/tossaki/App/QuantRabbit-live")
        self.assertEqual(payload["StartInterval"], 300)
        self.assertEqual(
            payload["StandardOutPath"],
            "/Users/tossaki/App/QuantRabbit-live/logs/qr_trader_run_watchdog.launchd.log",
        )
        self.assertEqual(
            payload["StandardErrorPath"],
            "/Users/tossaki/App/QuantRabbit-live/logs/qr_trader_run_watchdog.launchd.err",
        )
        command = " ".join(payload["ProgramArguments"])
        self.assertIn("tools/qr_trader_run_watchdog.py", command)
        self.assertIn("/Users/tossaki/App/QuantRabbit-live", command)
        env = payload["EnvironmentVariables"]
        self.assertEqual(env["QR_TRADER_WATCHDOG_CAN_WAKE"], "0")
        self.assertEqual(env["CODEX_DISABLE_UPDATE_CHECK"], "1")

        if shutil.which("plutil"):
            result = subprocess.run(["plutil", "-lint", str(PLIST)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


def _fixture(tmp: str, *, now: datetime) -> tuple[Path, Path, WatchdogPaths]:
    root = Path(tmp) / "live"
    automation_dir = Path(tmp) / "codex" / "automations" / "qr-trader"
    for rel in ("data", "docs", "logs"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    automation_dir.mkdir(parents=True, exist_ok=True)
    paths = WatchdogPaths.from_root(
        root,
        automation_dir=automation_dir,
        weekend_state=Path(tmp) / "codex" / "quant_rabbit_weekend_task_state.json",
        codex_logs=Path(tmp) / "codex" / "logs_2.sqlite",
    )
    return root, automation_dir, paths


def _write_automation(
    automation_dir: Path,
    *,
    status: str = "ACTIVE",
    model: str = "gpt-5.5",
    reasoning_effort: str = "high",
    rrule: str = "FREQ=MINUTELY;INTERVAL=360;BYDAY=SU,MO,TU,WE,TH,FR,SA",
    cwds: list[str] | None = None,
) -> None:
    cwds = cwds or ["/Users/tossaki/App/QuantRabbit-live"]
    automation_dir.joinpath("automation.toml").write_text(
        "\n".join(
            [
                'id = "qr-trader"',
                f'status = "{status}"',
                f'rrule = "{rrule}"',
                f'model = "{model}"',
                f'reasoning_effort = "{reasoning_effort}"',
                "cwds = [" + ", ".join(json.dumps(item) for item in cwds) + "]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_decision(root: Path, generated_at: datetime) -> None:
    _write_json(root / "data" / "codex_trader_decision_response.json", {"generated_at_utc": generated_at.isoformat(), "action": "WAIT"})
    (root / "docs" / "gpt_trader_decision_report.md").write_text(
        f"# GPT Trader Decision Report\n\n- Generated at UTC: `{generated_at.isoformat()}`\n- Status: `ACCEPTED`\n",
        encoding="utf-8",
    )
    (root / "docs" / "autotrade_cycle_report.md").write_text(
        f"# Autotrade Cycle Report\n\n- Generated at UTC: `{generated_at.isoformat()}`\n- Status: `NO_LIVE_READY_INTENT`\n",
        encoding="utf-8",
    )


def _write_ai_regime_supervision(
    root: Path,
    *,
    generated_at: datetime,
    authority: str = "NONE",
    broker_mutation_allowed: bool = False,
    body_overrides: dict | None = None,
) -> None:
    body = {
        "contract": "QR_AI_REGIME_SUPERVISION_V1",
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "last_tuned_at_utc": generated_at.isoformat(),
        "ai_role": "REGIME_REVIEW_AND_PERIODIC_TUNING_ONLY",
        "ai_order_authority": authority,
        "live_permission": False,
        "broker_mutation_allowed": broker_mutation_allowed,
        "pairs": {},
    }
    body.update(body_overrides or {})
    raw = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    _write_json(
        root / "data" / "ai_regime_supervision.json",
        {**body, "contract_sha256": hashlib.sha256(raw).hexdigest()},
    )


def _write_journal(root: Path, generated_at: datetime) -> None:
    journal = root / "logs" / "trader_journal.jsonl"
    journal.write_text(json.dumps({"ts": generated_at.isoformat(), "status": "NO_LIVE_READY_INTENT"}) + "\n")


def _write_memory(automation_dir: Path, generated_at: datetime) -> None:
    memory = automation_dir / "memory.md"
    memory.write_text(f"# memory\n\n- {generated_at.isoformat()} qr-trader run completed\n", encoding="utf-8")
    os.utime(memory, (generated_at.timestamp(), generated_at.timestamp()))


def _write_weekend_state(path: Path) -> None:
    _write_json(
        path,
        {
            "mode": "paused",
            "pause_applied_at_utc": "2026-07-03T21:01:00+00:00",
            "managed_task_keys": ["codex:qr-trader"],
            "last_changes": [
                {
                    "key": "codex:qr-trader",
                    "field": "status",
                    "before": "ACTIVE",
                    "after": "PAUSED",
                    "changed": True,
                }
            ],
            "tasks": {
                "codex:qr-trader": {
                    "kind": "codex",
                    "path": "/Users/tossaki/.codex/automations/qr-trader/automation.toml",
                }
            },
        },
    )


def _write_guardian_receipt(
    root: Path,
    *,
    action: str,
    generated_at: datetime,
    expires_at: datetime,
    consumed: bool,
    lifecycle: str = "ACTIVE",
    event_id: str = "guardian-event-1",
    thesis_state: str = "WOUNDED",
    path: Path | None = None,
) -> None:
    path = path or root / "data" / "guardian_action_receipt.json"
    _write_json(
        path,
        {
            "receipt_status": "ACCEPTED",
            "receipt_lifecycle": lifecycle,
            "generated_at_utc": generated_at.isoformat(),
            "expires_at_utc": expires_at.isoformat(),
            "consumed_by_trader": consumed,
            "selected_event": {
                "dedupe_key": f"PAIR|THESIS|EVENT|{action}",
                "event_id": event_id,
                "event_type": "CONTRACT_TEST_EVENT",
                "recommended_review_type": "RISK_REVIEW",
                "severity": "P1",
                "thesis_state": thesis_state,
            },
            "receipt": {
                "action": action,
                "dedupe_key": f"PAIR|THESIS|EVENT|{action}",
                "event_id": event_id,
                "margin_state": "margin_available_jpy=100000 margin_used_jpy=0 margin_pressure=false",
                "thesis_state": thesis_state,
            },
        },
    )
    (root / "docs" / "guardian_action_review.md").write_text(
        "# Guardian Action Review\n\n- Status: `RECEIPT_WRITTEN`\n", encoding="utf-8"
    )


def _write_consumption(
    root: Path,
    *,
    issue_code: str,
    event_id: str,
    action: str,
    lifecycle: str,
    classification: str,
    normal_routing_allowed: bool,
) -> None:
    _write_json(
        root / "data" / "guardian_receipt_consumption.json",
        {
            "generated_at_utc": "2026-07-01T02:56:00+00:00",
            "status": "GUARDIAN_RECEIPT_ISSUES_ACKNOWLEDGED"
            if normal_routing_allowed
            else "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
            "normal_routing_allowed": normal_routing_allowed,
            "classifications": [
                {
                    "issue_code": issue_code,
                    "receipt_event_id": event_id,
                    "receipt_action": action,
                    "receipt_lifecycle": lifecycle,
                    "consumed_by_trader": False,
                    "classification": classification,
                    "reason": "test classification",
                    "normal_routing_allowed": normal_routing_allowed,
                }
            ],
        },
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _issue_codes(payload: dict[str, object]) -> set[str]:
    return {str(item["code"]) for item in payload["issues"] if isinstance(item, dict)}


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


if __name__ == "__main__":
    unittest.main()
