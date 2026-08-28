from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from oanda_live_llm_inventory import (
    SOURCE_COMMIT,
    run_once,
)
from shadow_runtime import HashLedger


class LlmInventoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "ledgers").mkdir()
        completed = HashLedger(self.root / "ledgers" / "completed_m5.jsonl")
        planned = []
        completed.plan({"instrument": "EUR_USD"}, "bar-1", planned)
        completed.plan({"instrument": "USD_JPY"}, "bar-2", planned)
        completed.append_rows(planned)
        canary = HashLedger(self.root / "ledgers" / "plumbing_canary.jsonl")
        planned = []
        for index in range(1, 9):
            payload = {"kind": f"fixture-{index}"}
            if index == 7:
                payload = {"kind": "inventory_snapshot", "open_inventory_count": 2}
            canary.plan(payload, f"canary-{index}", planned)
        canary.append_rows(planned)
        self.completed_head = completed.last_hash
        self.canary_head = canary.last_hash
        self.inventory_hash = canary.rows[6]["record_hash"]

    def tearDown(self):
        self.temp.cleanup()

    def test_one_call_receipt_is_bounded_and_idempotent(self):
        request = datetime(2026, 8, 28, 2, 0, tzinfo=timezone.utc)
        decision = {
            "action": "FREEZE",
            "currency_cap": 0,
            "mode": "SHADOW_ONLY",
            "valid_until": "2026-08-28T03:00:00Z",
            "confidence": 0.9,
            "reason": "Non-evidence canary; keep new virtual inventory frozen.",
        }

        with patch("oanda_live_llm_inventory.COMPLETED_M5_HEAD", self.completed_head), \
             patch("oanda_live_llm_inventory.CANARY_HEAD", self.canary_head), \
             patch("oanda_live_llm_inventory.INVENTORY_SNAPSHOT_HASH", self.inventory_hash):
            result = run_once(
                self.root, SOURCE_COMMIT, decision, request, request + timedelta(seconds=3)
            )
        self.assertEqual(result["llm_calls"], 1)
        self.assertEqual(result["external_orders"], 0)
        self.assertFalse(result["individual_order_control"])
        self.assertFalse(result["hard_guard_mutation"])
        self.assertNotIn('"entry_price":', result["prompt_full"])
        self.assertNotIn('"direction":', result["prompt_full"])
        with self.assertRaisesRegex(RuntimeError, "LLM_CALL_ALREADY_RECORDED"):
            run_once(self.root, SOURCE_COMMIT, decision, request, request)


if __name__ == "__main__":
    unittest.main()
