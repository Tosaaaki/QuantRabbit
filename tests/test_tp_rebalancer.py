"""Unit tests for strategy/tp_rebalancer.py."""

from __future__ import annotations

import os
import unittest

from quant_rabbit.strategy.tp_rebalancer import (
    HYSTERESIS_PIPS,
    TPAdjustment,
    _chart_context_from_chart,
    _extract_atr_pips,
    apply_tp_adjustments,
    compute_tp_adjustment,
)


class ComputeTPAdjustmentTest(unittest.TestCase):
    def _kill_switch_off(self) -> None:
        if "QR_DISABLE_TP_REBALANCE" in os.environ:
            del os.environ["QR_DISABLE_TP_REBALANCE"]

    def test_long_expanded_tp(self) -> None:
        self._kill_switch_off()
        # Entry 1.3000, current price 1.3050 (50 pip in profit),
        # current TP 1.3050 (50 pip from entry), reward_risk=3.0, ATR=30
        # → desired distance = 90 pip → new TP 1.3090 (40 pip wider)
        adj = compute_tp_adjustment(
            trade_id="t1", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=1.3050,
            current_price=1.3050, atr_pips=30, reward_risk=3.0,
        )
        self.assertIsNotNone(adj)
        self.assertGreater(adj.new_tp, adj.current_tp)  # expanded
        self.assertGreater(adj.distance_pips_new, adj.distance_pips_old)

    def test_short_expanded_tp(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t2", pair="USD_JPY", side="SHORT",
            entry_price=160.00, current_tp=159.70,
            current_price=159.70, atr_pips=40, reward_risk=2.5,
        )
        self.assertIsNotNone(adj)
        self.assertLess(adj.new_tp, adj.current_tp)  # TP moves DOWN for SHORT = wider
        self.assertGreater(adj.distance_pips_new, adj.distance_pips_old)

    def test_contraction_is_refused(self) -> None:
        """TP must never contract toward entry (2026-05-13 fix after the
        471029 incident where an 88pip TP was clamped to 2.6pip and fired
        for +442JPY instead of riding to the original 88pip target).
        When market reward_risk × ATR is SHORTER than the existing TP
        distance, leave the existing TP alone."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t3", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=1.3100,  # 100 pip TP
            current_price=1.3010,  # tiny profit
            atr_pips=15, reward_risk=1.5,  # 22.5 pip desired (< 100)
        )
        self.assertIsNone(adj)

    def test_regression_471029_long_tp_must_not_collapse_to_safety_margin(self) -> None:
        """Exact 471029 incident: entry 1.35077, original TP 1.35956
        (88pip), current price 1.35100, market_rr small enough that
        candidate_tp falls below current+5pip. Old logic clamped to
        current+5pip = 1.35150, firing immediately. New logic must
        refuse the contraction and leave TP alone."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="471029", pair="GBP_USD", side="LONG",
            entry_price=1.35077, current_tp=1.35956,
            current_price=1.35100,
            atr_pips=2.0, reward_risk=1.5,  # 3 pip desired — well below 88pip
        )
        self.assertIsNone(adj)

    def test_hysteresis_blocks_small_change(self) -> None:
        self._kill_switch_off()
        # Existing 50pip TP, new candidate 51pip → 1pip change < HYSTERESIS
        adj = compute_tp_adjustment(
            trade_id="t4", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=1.3050,
            current_price=1.3010, atr_pips=20, reward_risk=2.55,  # 51 pip
        )
        self.assertIsNone(adj)

    def test_skips_manual_owned(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t5", pair="EUR_USD", side="LONG",
            entry_price=1.30, current_tp=1.31,
            current_price=1.305, atr_pips=30, reward_risk=3.0,
            owner="manual",
        )
        self.assertIsNone(adj)

    def test_skips_position_without_tp(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t6", pair="EUR_USD", side="LONG",
            entry_price=1.30, current_tp=None,
            current_price=1.305, atr_pips=30, reward_risk=3.0,
        )
        self.assertIsNone(adj)

    def test_long_tp_never_below_entry(self) -> None:
        """Even with extreme contraction, LONG TP must stay > entry."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t7", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=1.3050,
            current_price=1.2950,  # underwater
            atr_pips=5, reward_risk=1.5,  # 7.5 pip desired
        )
        if adj is not None:
            self.assertGreater(adj.new_tp, 1.3000)  # above entry

    def test_short_tp_never_above_entry(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t8", pair="USD_JPY", side="SHORT",
            entry_price=160.00, current_tp=159.50,
            current_price=160.50,  # underwater for SHORT
            atr_pips=10, reward_risk=1.5,
        )
        if adj is not None:
            self.assertLess(adj.new_tp, 160.00)

    def test_safety_margin_from_current_price_long(self) -> None:
        """LONG TP must stay above current price + safety margin so it
        doesn't fire on the same tick."""
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t9", pair="EUR_USD", side="LONG",
            entry_price=1.3000, current_tp=1.3100,
            current_price=1.3080,  # close to current TP
            atr_pips=10, reward_risk=1.5,  # 15 pip → candidate 1.3015 (below current!)
        )
        # candidate below current would be raised to current+safety, but
        # then it must stay > entry (already true here). Either we get
        # adj with new_tp >= current+safety, or None.
        if adj is not None:
            self.assertGreaterEqual(adj.new_tp, 1.3080)

    def test_kill_switch_disables(self) -> None:
        os.environ["QR_DISABLE_TP_REBALANCE"] = "1"
        try:
            adj = compute_tp_adjustment(
                trade_id="t-k", pair="EUR_USD", side="LONG",
                entry_price=1.30, current_tp=1.31,
                current_price=1.305, atr_pips=30, reward_risk=3.0,
            )
            self.assertIsNone(adj)
        finally:
            del os.environ["QR_DISABLE_TP_REBALANCE"]

    def test_zero_atr_returns_none(self) -> None:
        self._kill_switch_off()
        adj = compute_tp_adjustment(
            trade_id="t-z", pair="EUR_USD", side="LONG",
            entry_price=1.30, current_tp=1.31,
            current_price=1.305, atr_pips=0, reward_risk=3.0,
        )
        self.assertIsNone(adj)


class ChartContextExtractionTest(unittest.TestCase):
    def test_atr_from_confluence_h4(self) -> None:
        chart = {"pair": "EUR_USD", "confluence": {"h4_atr_pips": 25}}
        self.assertEqual(_extract_atr_pips(chart, "EUR_USD"), 25.0)

    def test_atr_fallback_to_views_indicators(self) -> None:
        chart = {
            "pair": "EUR_USD",
            "confluence": {},
            "views": [
                {"granularity": "H1", "indicators": {"atr_pips": 18}},
                {"granularity": "H4", "indicators": {"atr_pips": 30}},
            ],
        }
        # Preference order picks H4 first.
        self.assertEqual(_extract_atr_pips(chart, "EUR_USD"), 30.0)

    def test_chart_context_extracts_h1_adx(self) -> None:
        chart = {
            "pair": "EUR_USD",
            "confluence": {"atr_percentile_24h": 0.6},
            "views": [{"granularity": "H1", "indicators": {"adx_14": 28}}],
        }
        ctx = _chart_context_from_chart(chart)
        self.assertEqual(ctx["h1_adx"], 28)
        self.assertEqual(ctx["confluence"]["atr_percentile_24h"], 0.6)


class ApplyTPAdjustmentsTest(unittest.TestCase):
    def test_dry_run_does_not_call_broker(self) -> None:
        adj = TPAdjustment(
            trade_id="t1", pair="EUR_USD", side="LONG",
            entry_price=1.30, current_tp=1.31, new_tp=1.32,
            distance_pips_old=100, distance_pips_new=200, rationale="test",
        )

        class MockClient:
            calls = []
            def replace_trade_dependent_orders(self, *a, **kw):
                self.calls.append((a, kw))

        client = MockClient()
        results = apply_tp_adjustments([adj], client, dry_run=True)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["sent"])
        self.assertEqual(client.calls, [])

    def test_apply_calls_broker(self) -> None:
        adj = TPAdjustment(
            trade_id="t2", pair="USD_JPY", side="SHORT",
            entry_price=160.0, current_tp=159.7, new_tp=159.5,
            distance_pips_old=30, distance_pips_new=50, rationale="test",
        )

        class MockClient:
            calls = []
            def replace_trade_dependent_orders(self, trade_id, payload):
                self.calls.append((trade_id, payload))
                return {"ok": True}

        client = MockClient()
        results = apply_tp_adjustments([adj], client, dry_run=False)
        self.assertTrue(results[0]["sent"])
        self.assertIsNone(results[0]["error"])
        self.assertEqual(client.calls[0][0], "t2")
        self.assertIn("takeProfit", client.calls[0][1])

    def test_broker_exception_captured_per_adjustment(self) -> None:
        adj = TPAdjustment(
            trade_id="t3", pair="EUR_USD", side="LONG",
            entry_price=1.30, current_tp=1.31, new_tp=1.32,
            distance_pips_old=100, distance_pips_new=200, rationale="test",
        )

        class FlakyClient:
            def replace_trade_dependent_orders(self, *a, **kw):
                raise RuntimeError("broker timeout")

        client = FlakyClient()
        results = apply_tp_adjustments([adj], client, dry_run=False)
        self.assertFalse(results[0]["sent"])
        self.assertEqual(results[0]["error"], "broker timeout")


if __name__ == "__main__":
    unittest.main()
