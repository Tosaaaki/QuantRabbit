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
from oanda_launchd_manage import preinstall
from shadow_runtime import HashLedger, atomic_json, canonical_bytes, canonical_hash, utc_text


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

    def test_plists_lint_and_have_only_oanda_labels(self):
        result = preinstall()
        self.assertEqual(result["plists"], 4)
        self.assertEqual(result["lint_failures"], 0)
        self.assertEqual(result["candidate_runtime_hash"], runtime.SHARED_RUNTIME_HASH)
        self.assertEqual(result["service_attestation_hash"], runtime.SERVICE_ATTESTATION_HASH)

    def test_service_attestation_binds_every_executable_source(self):
        expected = {
            "oanda_launchd_runtime.py",
            "oanda_live_feed.py",
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

    def test_bot_replay_is_idempotent_and_hash_bound(self):
        self._raw()
        with patch.object(runtime, "SERVICE_ROOT", self.root):
            first = runtime.bot_process_once()
            second = runtime.bot_process_once()
        self.assertEqual(first, second)
        self.assertEqual(first["completed_m5"], 2)
        self.assertEqual(first["natural_r5_proposals"], 0)
        self.assertEqual(first["external_orders"], 0)

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
            actual = [row["payload"] for row in processor.completed.rows]
        self.assertEqual(initial["completed_m5"], 2)
        self.assertEqual(updated["completed_m5"], 4)
        self.assertEqual(repeated, updated)
        order = lambda bar: (bar["instrument"], bar["start_utc"])
        self.assertEqual(sorted(actual, key=order), sorted(expected, key=order))
        self.assertEqual(processor.processed_rows, len(writer.rows))

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
        trigger = {
            "trigger_id": "fixture-1", "runtime_hash": runtime.SHARED_RUNTIME_HASH,
            "inventory_snapshot_hash": "a" * 64, "open_inventory_count": 1,
            "created_at_utc": "2026-08-28T01:00:00Z",
            "evidence_eligible": False, "profit_evidence": False, "external_orders": 0,
        }
        atomic_json(trigger_dir / "llm_inventory_request.json", trigger)
        calls = []

        def fake(prompt):
            calls.append(prompt)
            return {
                "action": "FREEZE", "currency_cap": 0, "mode": "SHADOW_ONLY",
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
