from __future__ import annotations

import ast
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import oanda_launchd_runtime as runtime
from oanda_launchd_manage import preinstall
from shadow_runtime import HashLedger, atomic_json, canonical_hash, utc_text


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

    def test_bot_replay_is_idempotent_and_hash_bound(self):
        self._raw()
        with patch.object(runtime, "SERVICE_ROOT", self.root):
            first = runtime.bot_process_once()
            second = runtime.bot_process_once()
        self.assertEqual(first, second)
        self.assertEqual(first["completed_m5"], 2)
        self.assertEqual(first["natural_r5_proposals"], 0)
        self.assertEqual(first["external_orders"], 0)

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
