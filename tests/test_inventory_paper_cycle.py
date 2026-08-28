from __future__ import annotations

import unittest
from datetime import datetime, timezone

from quant_rabbit.models import Quote
from tools.run_inventory_paper_cycle import run_paper_cycle


class InventoryPaperCycleTest(unittest.TestCase):
    def test_event_to_llm_to_fast_bot_gateway_partial_and_flat_readback(self) -> None:
        now = datetime(2026, 8, 28, 10, 0, tzinfo=timezone.utc)
        result = run_paper_cycle(
            quotes={
                "EUR_USD": Quote("EUR_USD", 1.16410, 1.16418, now),
                "USD_JPY": Quote("USD_JPY", 159.662, 159.670, now),
            },
            now_utc=now,
        )
        self.assertEqual(result["status"], "PAPER_LOOP_FLAT")
        self.assertEqual(result["orders_sent"], 0)
        self.assertFalse(result["broker_write_performed"])
        self.assertEqual(result["fast_bot_signal_count"], 2)
        self.assertEqual(result["staged_gateway_order_count"], 2)
        self.assertEqual(result["partial_scale_out_lot_count"], 2)
        self.assertEqual(result["terminal_liquidation_lot_count"], 2)
        self.assertEqual(result["duplicate_effective_applications"], 0)
        self.assertEqual(result["stale_decision_applications"], 0)
        self.assertEqual(result["remaining_bot_owned_units"], 0)
        self.assertEqual(result["final_inventory_state"], "STOPPED")
        self.assertEqual(len(set(result["staged_gateway_client_ids"])), 2)


if __name__ == "__main__":
    unittest.main()
