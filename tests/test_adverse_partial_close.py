"""Unit tests for strategy/adverse_partial_close.py."""

from __future__ import annotations

import os
import unittest

from quant_rabbit.strategy.adverse_partial_close import (
    ADVERSE_PARTIAL_TRIGGER_ATR_MULT,
    PARTIAL_CLOSE_FRACTION,
    PartialCloseAction,
    apply_partial_closes,
    compute_partial_close,
)


class ComputePartialCloseTest(unittest.TestCase):
    def _kill_switch_off(self) -> None:
        if "QR_DISABLE_ADVERSE_PARTIAL_CLOSE" in os.environ:
            del os.environ["QR_DISABLE_ADVERSE_PARTIAL_CLOSE"]

    def test_no_action_if_in_profit(self) -> None:
        self._kill_switch_off()
        a = compute_partial_close(
            trade_id="t1", pair="EUR_USD", side="LONG",
            units=10000, entry_price=1.3000,
            current_price=1.3050,  # in profit
            atr_pips=20, is_reversal_firing=False,
        )
        self.assertIsNone(a)

    def test_no_action_if_only_mildly_adverse(self) -> None:
        self._kill_switch_off()
        # 10 pip underwater, threshold = 1.5 × 20 = 30 pip
        a = compute_partial_close(
            trade_id="t2", pair="EUR_USD", side="LONG",
            units=10000, entry_price=1.3000,
            current_price=1.2990,
            atr_pips=20, is_reversal_firing=False,
        )
        self.assertIsNone(a)

    def test_partial_close_fires_when_significantly_adverse(self) -> None:
        self._kill_switch_off()
        # 50 pip underwater, threshold = 1.5 × 20 = 30 pip → fires
        a = compute_partial_close(
            trade_id="t3", pair="EUR_USD", side="LONG",
            units=10000, entry_price=1.3000,
            current_price=1.2950,
            atr_pips=20, is_reversal_firing=False,
        )
        self.assertIsNotNone(a)
        self.assertEqual(a.original_units, 10000)
        # 50% of 10000 = 5000, rounded down to 5000
        self.assertEqual(a.close_units, 5000)
        self.assertEqual(a.remaining_units, 5000)
        self.assertGreaterEqual(a.adverse_pips, ADVERSE_PARTIAL_TRIGGER_ATR_MULT * 20)

    def test_short_underwater_fires(self) -> None:
        self._kill_switch_off()
        # SHORT, price ABOVE entry by 50 pip
        a = compute_partial_close(
            trade_id="t4", pair="USD_JPY", side="SHORT",
            units=-13000, entry_price=160.00,
            current_price=160.50,
            atr_pips=20, is_reversal_firing=False,
        )
        self.assertIsNotNone(a)
        self.assertEqual(a.original_units, 13000)
        # 50% of 13000 = 6500, rounded down to 6500
        self.assertEqual(a.close_units, 6500)

    def test_reversal_firing_blocks_partial_close(self) -> None:
        """Position underwater BUT reversal signal firing → don't reduce
        exposure now."""
        self._kill_switch_off()
        a = compute_partial_close(
            trade_id="t5", pair="GBP_USD", side="LONG",
            units=10000, entry_price=1.3500,
            current_price=1.3450,  # underwater
            atr_pips=20, is_reversal_firing=True,  # but reversal!
        )
        self.assertIsNone(a)

    def test_skips_manual_owned(self) -> None:
        self._kill_switch_off()
        a = compute_partial_close(
            trade_id="t6", pair="EUR_USD", side="LONG",
            units=10000, entry_price=1.30, current_price=1.295,
            atr_pips=20, is_reversal_firing=False,
            owner="manual",
        )
        self.assertIsNone(a)

    def test_skips_small_positions(self) -> None:
        """Don't partial a 1000u position — would close 500u, too small."""
        self._kill_switch_off()
        a = compute_partial_close(
            trade_id="t7", pair="EUR_USD", side="LONG",
            units=1000, entry_price=1.30, current_price=1.295,
            atr_pips=20, is_reversal_firing=False,
        )
        self.assertIsNone(a)

    def test_kill_switch_disables(self) -> None:
        os.environ["QR_DISABLE_ADVERSE_PARTIAL_CLOSE"] = "1"
        try:
            a = compute_partial_close(
                trade_id="t-k", pair="EUR_USD", side="LONG",
                units=10000, entry_price=1.30, current_price=1.295,
                atr_pips=20, is_reversal_firing=False,
            )
            self.assertIsNone(a)
        finally:
            del os.environ["QR_DISABLE_ADVERSE_PARTIAL_CLOSE"]

    def test_zero_atr_returns_none(self) -> None:
        self._kill_switch_off()
        a = compute_partial_close(
            trade_id="t-z", pair="EUR_USD", side="LONG",
            units=10000, entry_price=1.30, current_price=1.295,
            atr_pips=0, is_reversal_firing=False,
        )
        self.assertIsNone(a)

    def test_close_units_rounded_to_100(self) -> None:
        """OANDA accepts integer units. Round close_units DOWN to nearest 100."""
        self._kill_switch_off()
        a = compute_partial_close(
            trade_id="t-r", pair="EUR_USD", side="LONG",
            units=7777, entry_price=1.30, current_price=1.295,
            atr_pips=20, is_reversal_firing=False,
        )
        # 50% of 7777 = 3888 → rounded down to 3800
        self.assertIsNotNone(a)
        self.assertEqual(a.close_units, 3800)
        self.assertEqual(a.close_units % 100, 0)


class ApplyPartialClosesTest(unittest.TestCase):
    def test_dry_run_does_not_call_broker(self) -> None:
        action = PartialCloseAction(
            trade_id="t1", pair="EUR_USD", side="LONG",
            original_units=10000, close_units=5000, remaining_units=5000,
            adverse_pips=50, atr_pips=20, rationale="test",
        )

        class MockClient:
            calls = []
            def close_trade(self, *a, **kw):
                self.calls.append((a, kw))

        client = MockClient()
        results = apply_partial_closes([action], client, dry_run=True)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["sent"])
        self.assertEqual(client.calls, [])

    def test_apply_calls_broker_with_units(self) -> None:
        action = PartialCloseAction(
            trade_id="t2", pair="USD_JPY", side="SHORT",
            original_units=13000, close_units=6500, remaining_units=6500,
            adverse_pips=40, atr_pips=20, rationale="test",
        )

        class MockClient:
            calls = []
            def close_trade(self, trade_id, units):
                self.calls.append((trade_id, units))
                return {"ok": True}

        client = MockClient()
        results = apply_partial_closes([action], client, dry_run=False)
        self.assertTrue(results[0]["sent"])
        self.assertEqual(client.calls[0], ("t2", "6500"))
        self.assertEqual(results[0]["provenance"], "adverse_partial_close")

    def test_apply_uses_provenance_method_when_supported(self) -> None:
        action = PartialCloseAction(
            trade_id="t3", pair="USD_JPY", side="SHORT",
            original_units=13000, close_units=6500, remaining_units=6500,
            adverse_pips=40, atr_pips=20, rationale="test",
        )

        class MockClient:
            calls = []
            def close_trade(self, trade_id, units):
                raise AssertionError("adverse partial close must use provenance-aware close when available")

            def close_trade_with_provenance(self, trade_id, units="ALL", *, provenance):
                self.calls.append((trade_id, units, provenance))
                return {"ok": True}

        client = MockClient()
        results = apply_partial_closes([action], client, dry_run=False)
        self.assertTrue(results[0]["sent"])
        self.assertEqual(client.calls[0], ("t3", "6500", "adverse_partial_close"))


if __name__ == "__main__":
    unittest.main()
