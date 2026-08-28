from __future__ import annotations

import ast
import fcntl
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import oanda_launchd_runtime as runtime
from oanda_launchd_manage import plist_paths, preinstall
from shadow_runtime import HashLedger, IntegrityError, atomic_json, canonical_bytes, canonical_hash, utc_text


class LaunchdRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _raw(self):
        path = self.root / "feed" / "ledgers" / "raw_bbo.jsonl"
        path.parent.mkdir(parents=True)
        ledger = HashLedger(path)
        planned = []
        for index, (symbol, minute) in enumerate((
            ("EUR_USD", 0), ("USD_JPY", 0), ("EUR_USD", 5), ("USD_JPY", 5)
        )):
            stamp = datetime(2026, 8, 28, 1, minute, 1, tzinfo=timezone.utc)
            payload = {
                "event_id": f"event-{index}", "instrument": symbol,
                "event_time_utc": utc_text(stamp),
                "arrival_time_utc": utc_text(stamp + timedelta(milliseconds=10)),
                "bid": 1.1 + index / 1000, "ask": 1.2 + index / 1000,
            }
            ledger.plan(payload, f"event-{index}", planned)
        ledger.append_rows(planned)

    def _segmented_raw(self, entries, root=None):
        target_root = self.root if root is None else Path(root)
        path = target_root / "feed" / "ledgers" / "raw_bbo.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        ledger = HashLedger(path)
        planned = []
        connections = {}
        for index, item in enumerate(entries):
            symbol, stamp, segment_id, segment_start = item[:4]
            arrival = item[4] if len(item) >= 5 else stamp + timedelta(milliseconds=10)
            feed_attestation = item[5] if len(item) >= 6 else "f" * 64
            connections.setdefault(segment_id, (segment_start, feed_attestation))
            payload = {
                "event_id": f"segmented-event-{index}",
                "instrument": symbol,
                "event_time_utc": utc_text(stamp),
                "arrival_time_utc": utc_text(arrival),
                "bid": 1.1 + index / 10000,
                "ask": 1.2 + index / 10000,
                "segment_id": segment_id,
                "segment_started_at_utc": utc_text(segment_start),
                "feed_service_attestation_hash": feed_attestation,
                "feed_provenance_status": "ATTESTED",
            }
            ledger.plan(payload, f"segmented-event-{index}", planned)
        ledger.append_rows(planned)
        control = HashLedger(path.parent / "control.jsonl")
        control_planned = []
        for index, (segment_id, (segment_start, feed_attestation)) in enumerate(connections.items(), 1):
            control.plan({
                "event": "LIVE_PRICING_CONNECTED",
                "segment_id": segment_id,
                "segment_started_at_utc": utc_text(segment_start),
                "feed_service_attestation_hash": feed_attestation,
                "feed_provenance_status": "ATTESTED",
            }, f"fixture-connect::{index}", control_planned)
        control.append_rows(control_planned)
        return ledger

    def test_plists_lint_and_have_only_oanda_labels(self):
        result = preinstall()
        self.assertEqual(result["plists"], 4)
        self.assertEqual(result["lint_failures"], 0)
        self.assertEqual(result["candidate_runtime_hash"], runtime.SHARED_RUNTIME_HASH)
        self.assertEqual(result["service_attestation_hash"], runtime.SERVICE_ATTESTATION_HASH)
        self.assertEqual(runtime.SERVICE_ROOT.name, "oanda_live_launchd_v2")
        for path in plist_paths():
            plist_text = path.read_text(encoding="utf-8")
            self.assertIn("oanda_live_launchd_v2", plist_text)
            self.assertNotIn("oanda_live_launchd_v1", plist_text)

    def test_service_attestation_binds_every_executable_source(self):
        expected = {
            "oanda_launchd_runtime.py",
            "oanda_live_feed.py",
            "oanda_paper_execution.py",
            "shadow_runtime.py",
            "oanda_live_runtime_contract.json",
            "oanda_launchagents/com.quantrabbit.oanda-live.feed-recorder.plist",
            "oanda_launchagents/com.quantrabbit.oanda-live.bot-shadow.plist",
            "oanda_launchagents/com.quantrabbit.oanda-live.llm-inventory.plist",
            "oanda_launchagents/com.quantrabbit.oanda-live.watchdog.plist",
        }
        self.assertEqual(set(runtime.RUNTIME_SOURCE_HASHES), expected)
        self.assertEqual(runtime.runtime_source_hashes(), runtime.RUNTIME_SOURCE_HASHES)
        self.assertEqual(
            runtime.SERVICE_ATTESTATION_HASH,
            canonical_hash({
                "candidate_runtime_hash": runtime.SHARED_RUNTIME_HASH,
                "runtime_source_sha256": runtime.RUNTIME_SOURCE_HASHES,
            }),
        )

    def test_launchd_feed_wrapper_passes_verified_source_attestation(self):
        with (
            patch.object(runtime, "load_approved_live_credentials", return_value=("account", "token")),
            patch.object(runtime, "OandaLiveRecorder") as recorder_type,
        ):
            recorder_type.return_value.run_live.return_value = {"feed_blocked": False}
            exit_code = runtime.run_feed(1.25)
        self.assertEqual(exit_code, 0)
        recorder_type.return_value.run_live.assert_called_once_with(
            "account",
            "token",
            1.25,
            runtime_hash=runtime.SERVICE_ATTESTATION_HASH,
        )

    def test_bot_replay_is_idempotent_and_hash_bound(self):
        self._raw()
        with patch.object(runtime, "SERVICE_ROOT", self.root):
            first = runtime.bot_process_once()
            second = runtime.bot_process_once()
        self.assertEqual(first, second)
        self.assertEqual(first["completed_m5"], 2)
        self.assertEqual(first["completed_m5_total"], 2)
        self.assertEqual(first["completed_m5_eligible"], 0)
        self.assertEqual(first["legacy_invalidated_m5"], 2)
        self.assertEqual(first["skipped_m5"], 0)
        self.assertEqual(first["natural_r5_proposals"], 0)
        self.assertEqual(first["external_orders"], 0)
        control = HashLedger(self.root / "bot" / "ledgers" / "control.jsonl")
        invalidations = [
            row["payload"] for row in control.rows
            if row["payload"].get("event") == "BAR_EVIDENCE_INVALIDATED"
        ]
        self.assertEqual(len(invalidations), 2)
        self.assertTrue(all(row["evidence_eligible"] is False for row in invalidations))
        self.assertTrue(all(row["external_orders"] == 0 for row in invalidations))

    def test_incremental_bot_tails_new_rows_and_matches_full_replay(self):
        self._raw()
        raw_path = self.root / "feed" / "ledgers" / "raw_bbo.jsonl"
        with patch.object(runtime, "SERVICE_ROOT", self.root):
            processor = runtime.IncrementalBot()
            initial = processor.process_once()
            writer = HashLedger(raw_path)
            planned = []
            for index, symbol in enumerate(("EUR_USD", "USD_JPY"), 4):
                stamp = datetime(2026, 8, 28, 1, 10, 1, tzinfo=timezone.utc)
                payload = {
                    "event_id": f"event-{index}", "instrument": symbol,
                    "event_time_utc": utc_text(stamp),
                    "arrival_time_utc": utc_text(stamp + timedelta(milliseconds=10)),
                    "bid": 1.1 + index / 1000, "ask": 1.2 + index / 1000,
                }
                writer.plan(payload, f"event-{index}", planned)
            writer.append_rows(planned)
            updated = processor.process_once()
            repeated = processor.process_once()
            expected = runtime._completed_bars(writer)
            actual = [
                row["payload"] for row in processor.completed.rows
                if row["payload"].get("feed_continuity_eligible") is True
            ]
        self.assertEqual(initial["completed_m5"], 2)
        self.assertEqual(updated["completed_m5"], 4)
        self.assertEqual(updated["completed_m5_eligible"], 0)
        self.assertEqual(updated["legacy_invalidated_m5"], 4)
        self.assertEqual(repeated, updated)
        order = lambda bar: (bar["instrument"], bar["start_utc"])
        self.assertEqual(sorted(actual, key=order), sorted(expected, key=order))
        self.assertEqual(expected, [])
        self.assertEqual(processor.processed_rows, len(writer.rows))

    def test_natural_signal_fans_out_to_four_virtual_arms_and_realizes_pnl(self):
        base = datetime(2026, 8, 28, 3, 0, tzinfo=timezone.utc)
        segment_start = base - timedelta(minutes=5)
        raw_path = self.root / "feed" / "ledgers" / "raw_bbo.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        writer = HashLedger(raw_path)
        feed_control = HashLedger(raw_path.parent / "control.jsonl")
        control_planned = []
        feed_control.plan({
            "event": "LIVE_PRICING_CONNECTED",
            "segment_id": "segment-paper",
            "segment_started_at_utc": utc_text(segment_start),
            "feed_service_attestation_hash": "f" * 64,
            "feed_provenance_status": "ATTESTED",
        }, "fixture-connect::paper", control_planned)
        feed_control.append_rows(control_planned)

        def event(index, stamp, mid):
            spread = 0.00008
            return {
                "event_id": f"paper-event-{index}",
                "instrument": "EUR_USD",
                "event_time_utc": utc_text(stamp),
                "arrival_time_utc": utc_text(stamp + timedelta(milliseconds=20)),
                "bid": mid - spread / 2,
                "ask": mid + spread / 2,
                "segment_id": "segment-paper",
                "segment_started_at_utc": utc_text(segment_start),
                "feed_service_attestation_hash": "f" * 64,
                "feed_provenance_status": "ATTESTED",
            }

        initial_mids = [
            1.10000, 1.10008, 1.10018, 1.10032,
            1.10051, 1.10076, 1.10108, 1.10147,
        ]
        planned = []
        for index, mid in enumerate(initial_mids):
            payload = event(index, base + timedelta(minutes=5 * index, seconds=1), mid)
            writer.plan(payload, payload["event_id"], planned)
        writer.append_rows(planned)

        with patch.object(runtime, "SERVICE_ROOT", self.root):
            processor = runtime.IncrementalBot()
            warm = processor.process_once()
            self.assertEqual(warm["natural_paper_proposals"], 0)

            planned = []
            first_executable = event(8, base + timedelta(minutes=40, seconds=1), 1.10160)
            writer.plan(first_executable, first_executable["event_id"], planned)
            writer.append_rows(planned)
            opened = processor.process_once()
            self.assertEqual(opened["natural_paper_proposals"], 1)
            self.assertEqual(opened["expected_orders"], 4)
            self.assertEqual(opened["virtual_fills"], 3)
            self.assertEqual(opened["open_inventory_count"], 3)

            planned = []
            adverse_fill = event(9, base + timedelta(minutes=40, seconds=2), 1.10162)
            writer.plan(adverse_fill, adverse_fill["event_id"], planned)
            writer.append_rows(planned)
            all_open = processor.process_once()
            self.assertEqual(all_open["virtual_fills"], 4)
            self.assertEqual(all_open["open_inventory_count"], 4)
            processor.record_terminal_mtm()
            marked = processor.process_once()
            self.assertEqual(marked["terminal_mtm_records"], 4)

            planned = []
            tp_quote = event(10, base + timedelta(minutes=40, seconds=3), 1.10250)
            writer.plan(tp_quote, tp_quote["event_id"], planned)
            writer.append_rows(planned)
            closed = processor.process_once()
            self.assertEqual(closed["virtual_exits"], 4)
            self.assertEqual(closed["realized_pnl_records"], 4)
            self.assertEqual(closed["pnl_records"], 8)
            self.assertEqual(closed["open_inventory_count"], 0)
            self.assertEqual(closed["external_order_attempts"], 0)
            self.assertEqual(closed["external_orders"], 0)

            signal_ids = {
                row["payload"]["signal_id"]
                for ledger in (
                    processor.paper_ledgers["proposals"],
                    processor.paper_ledgers["expected_orders"],
                    processor.paper_ledgers["virtual_fills"],
                    processor.paper_ledgers["pnl"],
                )
                for row in ledger.rows
            }
            self.assertEqual(len(signal_ids), 1)
            self.assertTrue(
                all(
                    row["payload"].get("external_orders") == 0
                    for ledger in processor.paper_ledgers.values()
                    for row in ledger.rows
                )
            )
            base_pnl = next(
                row["payload"]
                for row in processor.paper_ledgers["pnl"].rows
                if row["payload"]["execution_arm"] == "EXECUTABLE_BASE"
                and row["payload"]["event"] == "REALIZED_PNL"
            )
            self.assertGreater(base_pnl["net_pips"], 0)
            self.assertGreater(base_pnl["execution_cost_pips"], 0)
            trigger = json.loads((self.root / "triggers" / "llm_inventory_request.json").read_text())
            self.assertEqual(trigger["event_kind"], "INVENTORY_CLOSED")
            self.assertEqual(trigger["open_inventory_count"], 0)
            self.assertFalse(trigger["individual_order_control_allowed"])

            proposal_head = processor.paper_ledgers["proposals"].last_hash
            restarted = runtime.IncrementalBot()
            repeated = restarted.process_once()
            self.assertEqual(repeated["natural_paper_proposals"], 1)
            self.assertEqual(restarted.paper_ledgers["proposals"].last_hash, proposal_head)
            self.assertEqual(repeated["external_orders"], 0)

    def test_reconnect_mid_bucket_skips_boundary_and_resumes_next_full_bucket(self):
        base = datetime(2026, 8, 28, 1, 0, tzinfo=timezone.utc)
        writer = self._segmented_raw((
            ("EUR_USD", base + timedelta(minutes=2), "segment-00000001", base + timedelta(minutes=2)),
            ("EUR_USD", base + timedelta(minutes=5, seconds=5), "segment-00000001", base + timedelta(minutes=2)),
            ("EUR_USD", base + timedelta(minutes=6), "segment-00000001", base + timedelta(minutes=2)),
            ("EUR_USD", base + timedelta(minutes=7), "segment-00000002", base + timedelta(minutes=7)),
            ("EUR_USD", base + timedelta(minutes=10, seconds=5), "segment-00000002", base + timedelta(minutes=7)),
            ("EUR_USD", base + timedelta(minutes=14, seconds=55), "segment-00000002", base + timedelta(minutes=7)),
            ("EUR_USD", base + timedelta(minutes=15, seconds=5), "segment-00000002", base + timedelta(minutes=7)),
        ))
        with patch.object(runtime, "SERVICE_ROOT", self.root):
            processor = runtime.IncrementalBot()
            first = processor.process_once()
            control_before = [row["record_hash"] for row in processor.control.rows]
            restarted = runtime.IncrementalBot()
            second = restarted.process_once()
            control_after = [row["record_hash"] for row in restarted.control.rows]
        self.assertEqual(first, second)
        self.assertEqual(first["completed_m5_total"], 1)
        self.assertEqual(first["completed_m5_eligible"], 1)
        self.assertEqual(first["skipped_m5"], 2)
        self.assertEqual(first["legacy_invalidated_m5"], 0)
        self.assertEqual(first["natural_r5_proposals"], 0)
        self.assertEqual(first["virtual_fills"], 0)
        self.assertEqual(first["external_orders"], 0)
        self.assertEqual(control_before, control_after)
        bar = restarted.completed.rows[0]["payload"]
        self.assertEqual(bar["start_utc"], utc_text(base + timedelta(minutes=10)))
        self.assertEqual(bar["segment_id"], "segment-00000002")
        self.assertEqual(bar["feed_service_attestation_hash"], "f" * 64)
        self.assertEqual(bar["bot_service_attestation_hash"], runtime.SERVICE_ATTESTATION_HASH)
        self.assertTrue(bar["feed_continuity_eligible"])
        skip_rows = [
            row["payload"] for row in restarted.control.rows
            if row["payload"].get("event") == "M5_CAUSAL_EVIDENCE_SKIPPED"
        ]
        self.assertEqual(
            {row["reason"] for row in skip_rows},
            {"M5_FIRST_PARTIAL_BUCKET_AFTER_CONNECT", "M5_SEGMENT_BOUNDARY_WITHIN_BUCKET"},
        )
        self.assertTrue(all(row["external_orders"] == 0 for row in skip_rows))
        self.assertEqual(runtime._completed_bars(writer), [bar])

    def test_clean_reconnect_still_skips_first_new_segment_bucket(self):
        base = datetime(2026, 8, 28, 2, 0, tzinfo=timezone.utc)
        writer = self._segmented_raw((
            ("USD_JPY", base + timedelta(seconds=1), "segment-00000001", base),
            ("USD_JPY", base + timedelta(minutes=5, seconds=1), "segment-00000001", base),
            ("USD_JPY", base + timedelta(minutes=10, seconds=1), "segment-00000002", base + timedelta(minutes=10)),
            ("USD_JPY", base + timedelta(minutes=15, seconds=1), "segment-00000002", base + timedelta(minutes=10)),
            ("USD_JPY", base + timedelta(minutes=20, seconds=1), "segment-00000002", base + timedelta(minutes=10)),
        ))
        with patch.object(runtime, "SERVICE_ROOT", self.root):
            counters = runtime.bot_process_once()
            completed = HashLedger(self.root / "bot" / "ledgers" / "completed_m5.jsonl")
            control = HashLedger(self.root / "bot" / "ledgers" / "control.jsonl")
        self.assertEqual(counters["completed_m5_total"], 2)
        self.assertEqual(counters["completed_m5_eligible"], 2)
        self.assertEqual(counters["skipped_m5"], 2)
        self.assertEqual(
            [row["payload"]["start_utc"] for row in completed.rows],
            [utc_text(base + timedelta(minutes=5)), utc_text(base + timedelta(minutes=15))],
        )
        skips = [row["payload"] for row in control.rows if row["payload"].get("event") == "M5_CAUSAL_EVIDENCE_SKIPPED"]
        self.assertEqual({row["start_utc"] for row in skips}, {
            utc_text(base), utc_text(base + timedelta(minutes=10)),
        })
        self.assertTrue(all(row["reason"] == "M5_FIRST_PARTIAL_BUCKET_AFTER_CONNECT" for row in skips))

    def test_snapshot_source_before_connect_bucket_does_not_admit_connect_bucket(self):
        base = datetime(2026, 8, 28, 4, 0, tzinfo=timezone.utc)
        segment_start = base + timedelta(seconds=1)
        self._segmented_raw((
            ("EUR_USD", base - timedelta(seconds=1), "segment-00000003", segment_start, segment_start + timedelta(milliseconds=10)),
            ("EUR_USD", base + timedelta(seconds=5), "segment-00000003", segment_start, base + timedelta(seconds=6)),
            ("EUR_USD", base + timedelta(minutes=5, seconds=1), "segment-00000003", segment_start, base + timedelta(minutes=5, seconds=2)),
            ("EUR_USD", base + timedelta(minutes=10, seconds=1), "segment-00000003", segment_start, base + timedelta(minutes=10, seconds=2)),
        ))
        with patch.object(runtime, "SERVICE_ROOT", self.root):
            counters = runtime.bot_process_once()
            completed = HashLedger(self.root / "bot" / "ledgers" / "completed_m5.jsonl")
            control = HashLedger(self.root / "bot" / "ledgers" / "control.jsonl")
        self.assertEqual(counters["completed_m5_eligible"], 1)
        self.assertEqual(counters["skipped_m5"], 2)
        self.assertEqual(completed.rows[0]["payload"]["start_utc"], utc_text(base + timedelta(minutes=5)))
        skips = [row["payload"] for row in control.rows if row["payload"].get("event") == "M5_CAUSAL_EVIDENCE_SKIPPED"]
        self.assertEqual(
            {row["reason"] for row in skips},
            {"M5_PRECONNECT_STALE_SNAPSHOT_BUCKET", "M5_FIRST_PARTIAL_BUCKET_AFTER_CONNECT"},
        )

    def test_reconnect_snapshot_regression_invalidates_previously_sealed_bucket(self):
        base = datetime(2026, 8, 28, 4, 0, tzinfo=timezone.utc)
        first_start = base - timedelta(minutes=10)
        second_start = base + timedelta(seconds=1)
        writer = self._segmented_raw((
            ("EUR_USD", base - timedelta(minutes=5) + timedelta(seconds=1), "segment-00000001", first_start),
            ("EUR_USD", base + timedelta(milliseconds=500), "segment-00000001", first_start),
            ("EUR_USD", base - timedelta(seconds=1), "segment-00000002", second_start, second_start + timedelta(milliseconds=10)),
            ("EUR_USD", base + timedelta(seconds=5), "segment-00000002", second_start, base + timedelta(seconds=6)),
            ("EUR_USD", base + timedelta(minutes=5, seconds=1), "segment-00000002", second_start),
            ("EUR_USD", base + timedelta(minutes=10, seconds=1), "segment-00000002", second_start),
        ))
        with patch.object(runtime, "SERVICE_ROOT", self.root):
            processor = runtime.IncrementalBot()
            first = processor.process_once()
            control_head = processor.control.last_hash
            full = runtime.bot_process_once()
            after_full = HashLedger(self.root / "bot" / "ledgers" / "control.jsonl")
            restarted = runtime.IncrementalBot()
            second = restarted.process_once()
        self.assertEqual(first, full)
        self.assertEqual(first, second)
        self.assertEqual(control_head, after_full.last_hash)
        self.assertEqual(control_head, restarted.control.last_hash)
        self.assertEqual(first["completed_m5_total"], 2)
        self.assertEqual(first["completed_m5_eligible"], 1)
        self.assertEqual(first["skipped_m5"], 1)
        self.assertEqual(first["late_stale_invalidated_m5"], 1)
        self.assertEqual(first["external_orders"], 0)
        invalidated_hashes = {
            control["payload"]["bar_record_hash"]
            for control in restarted.control.rows
            if control["payload"].get("event") == "BAR_EVIDENCE_INVALIDATED"
        }
        eligible = [
            row["payload"] for row in restarted.completed.rows
            if row["payload"].get("feed_continuity_eligible") is True
            and row["record_hash"] not in invalidated_hashes
            and (row["payload"]["instrument"], row["payload"]["start_utc"])
            not in {
                (control["payload"]["instrument"], control["payload"]["start_utc"])
                for control in restarted.control.rows
                if control["payload"].get("event") == "M5_CAUSAL_EVIDENCE_SKIPPED"
            }
        ]
        self.assertEqual([bar["start_utc"] for bar in eligible], [utc_text(base + timedelta(minutes=5))])
        reasons = {
            row["payload"]["reason"] for row in restarted.control.rows
            if row["payload"].get("event") == "M5_CAUSAL_EVIDENCE_SKIPPED"
        }
        self.assertEqual(reasons, {"M5_SEGMENT_BOUNDARY_WITHIN_BUCKET"})
        self.assertEqual(
            {
                row["payload"]["reason"] for row in restarted.control.rows
                if row["payload"].get("event") == "BAR_EVIDENCE_INVALIDATED"
            },
            {"LATE_SEGMENT_STALE_SNAPSHOT"},
        )

        fresh_root = self.root / "fresh-full"
        fresh_raw = HashLedger(fresh_root / "feed" / "ledgers" / "raw_bbo.jsonl")
        planned = []
        for row in writer.rows:
            fresh_raw.plan(row["payload"], row["record_id"], planned)
        fresh_raw.append_rows(planned)
        source_feed_control = HashLedger(self.root / "feed" / "ledgers" / "control.jsonl")
        fresh_feed_control = HashLedger(fresh_root / "feed" / "ledgers" / "control.jsonl")
        control_planned = []
        for row in source_feed_control.rows:
            fresh_feed_control.plan(row["payload"], row["record_id"], control_planned)
        fresh_feed_control.append_rows(control_planned)
        with patch.object(runtime, "SERVICE_ROOT", fresh_root):
            fresh_counts = runtime.bot_process_once()
            fresh_completed = HashLedger(fresh_root / "bot" / "ledgers" / "completed_m5.jsonl")
            fresh_control = HashLedger(fresh_root / "bot" / "ledgers" / "control.jsonl")
        fresh_invalidated_hashes = {
            row["payload"]["bar_record_hash"]
            for row in fresh_control.rows
            if row["payload"].get("event") == "BAR_EVIDENCE_INVALIDATED"
        }
        fresh_skipped_keys = {
            (row["payload"]["instrument"], row["payload"]["start_utc"])
            for row in fresh_control.rows
            if row["payload"].get("event") == "M5_CAUSAL_EVIDENCE_SKIPPED"
        }
        fresh_eligible = [
            row["payload"] for row in fresh_completed.rows
            if row["payload"].get("feed_continuity_eligible") is True
            and row["record_hash"] not in fresh_invalidated_hashes
            and (row["payload"]["instrument"], row["payload"]["start_utc"])
            not in fresh_skipped_keys
        ]
        self.assertEqual(fresh_counts, first)
        self.assertEqual(
            {canonical_hash(bar) for bar in eligible},
            {canonical_hash(bar) for bar in fresh_eligible},
        )
        evidence_semantics = lambda ledger: {
            (
                row["payload"].get("event"),
                row["payload"].get("reason"),
                row["payload"].get("instrument"),
                row["payload"].get("start_utc"),
            )
            for row in ledger.rows
            if row["payload"].get("event") in {
                "M5_CAUSAL_EVIDENCE_SKIPPED",
                "BAR_EVIDENCE_INVALIDATED",
            }
        }
        self.assertEqual(
            evidence_semantics(restarted.control),
            evidence_semantics(fresh_control),
        )

    def test_missing_mismatched_or_malformed_feed_attestation_fails_closed(self):
        base = datetime(2026, 8, 28, 5, 0, tzinfo=timezone.utc)
        missing_root = self.root / "missing-provenance"
        missing_raw = HashLedger(missing_root / "feed" / "ledgers" / "raw_bbo.jsonl")
        missing_planned = []
        missing_raw.plan({
            "event_id": "missing-provenance-event",
            "instrument": "EUR_USD",
            "event_time_utc": utc_text(base + timedelta(minutes=1)),
            "arrival_time_utc": utc_text(base + timedelta(minutes=1, milliseconds=10)),
            "bid": 1.1,
            "ask": 1.2,
            "segment_id": "segment-missing",
            "segment_started_at_utc": utc_text(base),
            "feed_service_attestation_hash": "f" * 64,
            "feed_provenance_status": "ATTESTED",
        }, "missing-provenance-event", missing_planned)
        missing_raw.append_rows(missing_planned)

        malformed_root = self.root / "malformed-provenance"
        self._segmented_raw((
            ("EUR_USD", base + timedelta(minutes=1), "segment-malformed", base, base + timedelta(minutes=1, seconds=1), "bad"),
            ("EUR_USD", base + timedelta(minutes=6), "segment-malformed", base, base + timedelta(minutes=6, seconds=1), "bad"),
        ), malformed_root)

        mismatch_root = self.root / "mismatched-provenance"
        self._segmented_raw((
            ("EUR_USD", base + timedelta(minutes=1), "segment-mismatch", base - timedelta(minutes=5), base + timedelta(minutes=1, seconds=1), "a" * 64),
            ("EUR_USD", base + timedelta(minutes=6), "segment-mismatch", base - timedelta(minutes=5), base + timedelta(minutes=6, seconds=1), "b" * 64),
        ), mismatch_root)

        for target_root in (missing_root, malformed_root, mismatch_root):
            with self.subTest(root=target_root.name), patch.object(runtime, "SERVICE_ROOT", target_root):
                with self.assertRaises(IntegrityError):
                    runtime.bot_process_once()
                self.assertFalse((target_root / "bot" / "ledgers" / "expected_orders.jsonl").exists())

    def test_rolling_hybrid_old_bar_is_invalidated_and_next_full_bucket_resumes(self):
        base = datetime(2026, 8, 28, 6, 0, tzinfo=timezone.utc)
        hybrid_root = self.root / "hybrid"
        writer = self._segmented_raw((
            ("USD_JPY", base + timedelta(seconds=1), "segment-new-feed", base - timedelta(minutes=5)),
            ("USD_JPY", base + timedelta(minutes=5, seconds=1), "segment-new-feed", base - timedelta(minutes=5)),
            ("USD_JPY", base + timedelta(minutes=10, seconds=1), "segment-new-feed", base - timedelta(minutes=5)),
        ), hybrid_root)
        with patch.object(runtime, "SERVICE_ROOT", hybrid_root):
            first_bar = runtime._closed_bucket_outcomes(writer)[0][0]
            self.assertIsNotNone(first_bar)
            old_payload = {
                key: value for key, value in first_bar.items()
                if key not in {
                    "segment_id",
                    "segment_started_at_utc",
                    "feed_service_attestation_hash",
                    "bot_service_attestation_hash",
                    "feed_continuity_eligible",
                }
            }
            self.assertTrue({
                "segment_id",
                "segment_started_at_utc",
                "feed_service_attestation_hash",
                "bot_service_attestation_hash",
                "feed_continuity_eligible",
            }.isdisjoint(old_payload))
            completed = HashLedger(hybrid_root / "bot" / "ledgers" / "completed_m5.jsonl")
            completed_planned = []
            old_row = completed.plan(
                old_payload,
                f"m5::{old_payload['instrument']}::{old_payload['start_utc']}",
                completed_planned,
            )
            completed.append_rows(completed_planned)

            processor = runtime.IncrementalBot()
            incremental = processor.process_once()
            control_head = processor.control.last_hash
            full = runtime.bot_process_once()
            restarted = runtime.IncrementalBot()
            after_restart = restarted.process_once()

        self.assertEqual(incremental, full)
        self.assertEqual(incremental, after_restart)
        self.assertEqual(control_head, restarted.control.last_hash)
        self.assertEqual(incremental["completed_m5_total"], 2)
        self.assertEqual(incremental["completed_m5_eligible"], 1)
        self.assertEqual(incremental["hybrid_invalidated_m5"], 1)
        self.assertEqual(incremental["skipped_m5"], 0)
        self.assertEqual(incremental["virtual_fills"], 0)
        self.assertEqual(incremental["external_orders"], 0)
        invalidations = [
            row["payload"] for row in restarted.control.rows
            if row["payload"].get("reason") == "ROLLING_HYBRID_ATTESTATION_MIGRATION"
        ]
        self.assertEqual(len(invalidations), 1)
        self.assertEqual(invalidations[0]["bar_record_hash"], old_row["record_hash"])
        eligible = [
            row["payload"] for row in restarted.completed.rows
            if row["payload"].get("feed_continuity_eligible") is True
            and row["record_hash"] != old_row["record_hash"]
        ]
        self.assertEqual([bar["start_utc"] for bar in eligible], [utc_text(base + timedelta(minutes=5))])
        self.assertEqual(eligible[0]["feed_service_attestation_hash"], "f" * 64)
        self.assertEqual(eligible[0]["bot_service_attestation_hash"], runtime.SERVICE_ATTESTATION_HASH)

    def test_hash_ledger_reader_waits_for_locked_append(self):
        path = self.root / "concurrent.jsonl"
        ledger = HashLedger(path)
        first = []
        ledger.plan({"value": 1}, "row-1", first)
        ledger.append_rows(first)
        second = []
        ledger.plan({"value": 2}, "row-2", second)
        encoded = canonical_bytes(second[0]) + b"\n"
        midpoint = len(encoded) // 2
        ready_read, ready_write = os.pipe()
        child = os.fork()
        if child == 0:
            try:
                os.close(ready_read)
                descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    os.write(descriptor, encoded[:midpoint])
                    os.write(ready_write, b"1")
                    time.sleep(0.1)
                    os.write(descriptor, encoded[midpoint:])
                    os.fsync(descriptor)
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
            finally:
                os.close(ready_write)
                os._exit(0)
        os.close(ready_write)
        try:
            self.assertEqual(os.read(ready_read, 1), b"1")
            concurrent = HashLedger(path)
        finally:
            os.close(ready_read)
            _, status = os.waitpid(child, 0)
        self.assertEqual(status, 0)
        self.assertEqual(len(concurrent.rows), 2)

    def test_llm_worker_is_triggered_once_and_fake_dependency_is_bounded(self):
        trigger_dir = self.root / "triggers"
        trigger_dir.mkdir(parents=True)
        created_at = datetime.now(timezone.utc)
        inventory_summary = {
            "positions": [{
                "position_id": "position-1",
                "instrument": "EUR_USD",
                "direction": 1,
                "opened_at_utc": utc_text(created_at - timedelta(minutes=1)),
                "max_age_at_utc": utc_text(created_at + timedelta(minutes=29)),
                "unrealized_pips": -0.8,
            }],
            "open_inventory_count": 1,
            "realized_pnl_jpy": 0.0,
            "hard_max_open_positions": 2,
            "current_policy": {
                "action": "ADD",
                "max_open_positions": 2,
                "source": "BOT_DEFAULT_BEFORE_FIRST_LLM_RECEIPT",
                "valid_until": None,
            },
        }
        trigger = {
            "schema_version": 2,
            "trigger_id": "fixture-1",
            "runtime_hash": runtime.SHARED_RUNTIME_HASH,
            "inventory_snapshot_hash": canonical_hash(inventory_summary),
            "open_inventory_count": 1,
            "created_at_utc": utc_text(created_at),
            "event_kind": "INVENTORY_OPENED",
            "inventory_summary": inventory_summary,
            "allowed_actions": list(runtime.ALLOWED_ACTIONS),
            "hard_guard_mutation_allowed": False,
            "individual_order_control_allowed": False,
            "research_status": "RESEARCH_NOT_ADMITTED",
            "profit_proven": False,
            "external_orders": 0,
        }
        atomic_json(trigger_dir / "llm_inventory_request.json", trigger)
        calls = []

        def fake(prompt):
            calls.append(prompt)
            return {
                "action": "FREEZE", "max_open_positions": 0, "mode": "SHADOW_ONLY",
                "valid_until": utc_text(datetime.now(timezone.utc) + timedelta(hours=1)),
                "confidence": 0.9, "reason": "Fixture dependency check.",
            }

        with patch.object(runtime, "SERVICE_ROOT", self.root):
            one = runtime.process_llm_trigger(fake)
            two = runtime.process_llm_trigger(fake)
        self.assertEqual(len(calls), 1)
        self.assertEqual(one["llm_calls"], 1)
        self.assertEqual(two["llm_calls"], 1)
        self.assertEqual(two["external_orders"], 0)
        with patch.object(runtime, "SERVICE_ROOT", self.root):
            processor = runtime.IncrementalBot()
        self.assertEqual(processor.llm_policy["action"], "FREEZE")
        self.assertEqual(processor.llm_policy["max_open_positions"], 0)

    def test_runtime_source_has_no_broker_or_write_http_surface(self):
        source = Path(runtime.__file__).read_text()
        tree = ast.parse(source)
        imports = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(any(name.startswith("quant_rabbit.broker") for name in imports))
        methods = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value in {"POST", "PUT", "PATCH", "DELETE"}}
        self.assertEqual(methods, set())
        lowered = source.lower()
        for endpoint in ("/orders", "/trades", "/positions"):
            self.assertNotIn(endpoint, lowered)


if __name__ == "__main__":
    unittest.main()
