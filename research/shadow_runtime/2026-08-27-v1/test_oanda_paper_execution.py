from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone

import oanda_launchd_runtime as runtime
from oanda_paper_execution import (
    PaperConfigError,
    completed_bar_input_window,
    evaluate_completed_bar_signal,
    executable_bbo_available,
    pip_size,
    pnl_pips,
    quote_pnl,
    validate_paper_config,
    virtual_price,
)
from shadow_runtime import utc_text


class PaperExecutionPrimitiveTest(unittest.TestCase):
    def _bars(self, prices: list[float]) -> list[dict]:
        base = datetime(2026, 8, 28, 0, 0, tzinfo=timezone.utc)
        rows = []
        for index, price in enumerate(prices):
            start = base + timedelta(minutes=5 * index)
            spread = 0.00008
            rows.append({
                "instrument": "EUR_USD",
                "segment_id": "segment-paper-fixture",
                "start_utc": utc_text(start),
                "end_utc": utc_text(start + timedelta(minutes=5)),
                "bid_o": price - spread / 2,
                "bid_h": price + 0.00005 - spread / 2,
                "bid_l": price - 0.00005 - spread / 2,
                "bid_c": price - spread / 2,
                "ask_o": price + spread / 2,
                "ask_h": price + 0.00005 + spread / 2,
                "ask_l": price - 0.00005 + spread / 2,
                "ask_c": price + spread / 2,
            })
        return rows

    def test_long_and_short_signals_are_cost_independent(self):
        rising = [1.10000, 1.10008, 1.10018, 1.10032, 1.10051, 1.10076, 1.10108, 1.10147]
        long_signal = evaluate_completed_bar_signal(self._bars(rising), runtime.PAPER_CONFIG)
        self.assertIsNotNone(long_signal)
        self.assertEqual(long_signal["direction"], 1)
        self.assertFalse(long_signal["entry_cost_gate_used"])
        self.assertGreaterEqual(
            long_signal["tp_distance_price"],
            long_signal["observed_spread_price"]
            * runtime.PAPER_CONFIG["tp_spread_multiple_floor"],
        )

        falling = list(reversed(rising))
        short_signal = evaluate_completed_bar_signal(self._bars(falling), runtime.PAPER_CONFIG)
        self.assertIsNotNone(short_signal)
        self.assertEqual(short_signal["direction"], -1)
        self.assertFalse(short_signal["entry_cost_gate_used"])

    def test_non_contiguous_or_cost_gated_configuration_fails_closed(self):
        rising = [1.10000, 1.10008, 1.10018, 1.10032, 1.10051, 1.10076, 1.10108, 1.10147]
        bars = self._bars(rising)
        bars[-1]["start_utc"] = utc_text(parse := datetime(2026, 8, 28, 2, 0, tzinfo=timezone.utc))
        bars[-1]["end_utc"] = utc_text(parse + timedelta(minutes=5))
        self.assertIsNone(evaluate_completed_bar_signal(bars, runtime.PAPER_CONFIG))
        bad = copy.deepcopy(runtime.PAPER_CONFIG)
        bad["entry_cost_gate_used"] = True
        with self.assertRaises(PaperConfigError):
            validate_paper_config(bad)

        safety_mutations = {
            "require_tradeable_bbo": False,
            "require_side_liquidity": False,
            "persist_latency_event_consumption": False,
            "llm_unwind_semantics": "CLOSE_ALL",
        }
        for field, value in safety_mutations.items():
            with self.subTest(field=field):
                bad = copy.deepcopy(runtime.PAPER_CONFIG)
                bad[field] = value
                with self.assertRaises(PaperConfigError):
                    validate_paper_config(bad)

    def test_historical_warmup_is_feature_only_and_must_end_in_later_live_bar(self):
        bars = self._bars([1.10000, 1.10008, 1.10018, 1.10032])
        for bar in bars[:3]:
            bar.pop("segment_id")
            bar.update(
                feature_source="OANDA_HISTORICAL_M5_WARMUP",
                warmup_only=True,
                excluded_from_forward_pnl=True,
            )
        bars[-1]["feature_source"] = "LIVE_ATTESTED_M5"
        self.assertEqual(completed_bar_input_window(bars, 4), bars)

        warmup_only = copy.deepcopy(bars)
        warmup_only[-1].pop("segment_id")
        warmup_only[-1].update(
            feature_source="OANDA_HISTORICAL_M5_WARMUP",
            warmup_only=True,
            excluded_from_forward_pnl=True,
        )
        self.assertIsNone(completed_bar_input_window(warmup_only, 4))

        warmup_after_live = copy.deepcopy(bars)
        warmup_after_live[1]["feature_source"] = "LIVE_ATTESTED_M5"
        warmup_after_live[1]["segment_id"] = "segment-paper-fixture"
        self.assertIsNone(completed_bar_input_window(warmup_after_live, 4))

        gap = copy.deepcopy(bars)
        gap[-1]["start_utc"] = utc_text(datetime(2026, 8, 28, 0, 20, tzinfo=timezone.utc))
        gap[-1]["end_utc"] = utc_text(datetime(2026, 8, 28, 0, 25, tzinfo=timezone.utc))
        self.assertIsNone(completed_bar_input_window(gap, 4))

    def test_bid_ask_and_stress_accounting_are_direction_correct(self):
        event = {"instrument": "EUR_USD", "bid": 1.10000, "ask": 1.10008}
        base = runtime.PAPER_CONFIG["arms"]["EXECUTABLE_BASE"]
        stress = runtime.PAPER_CONFIG["arms"]["ADVERSE_STRESS"]
        long_entry = virtual_price(event, 1, base, entry=True)
        long_exit = virtual_price(event, 1, base, entry=False)
        self.assertEqual(long_entry, event["ask"])
        self.assertEqual(long_exit, event["bid"])
        self.assertLess(virtual_price(event, -1, stress, entry=True), event["bid"])
        self.assertGreater(virtual_price(event, -1, stress, entry=False), event["ask"])
        self.assertAlmostEqual(quote_pnl(1.10000, 1.10020, 1, 1000), 0.2)
        self.assertAlmostEqual(pnl_pips(1.10000, 1.10020, 1, "EUR_USD"), 2.0)
        self.assertEqual(pip_size("USD_JPY"), 0.01)

    def test_tradeable_selected_side_liquidity_is_required_for_virtual_execution(self):
        event = {
            "instrument": "EUR_USD",
            "bid": 1.10000,
            "ask": 1.10008,
            "bid_liquidity": 1000,
            "ask_liquidity": 2000,
            "tradeable": True,
        }
        self.assertTrue(executable_bbo_available(event, 1, entry=True, required_units=2000))
        self.assertFalse(executable_bbo_available(event, 1, entry=False, required_units=2000))
        self.assertTrue(executable_bbo_available(event, -1, entry=False, required_units=2000))
        self.assertFalse(executable_bbo_available(event, -1, entry=True, required_units=2000))

        halted = {**event, "tradeable": False}
        self.assertFalse(executable_bbo_available(halted, 1, entry=True, required_units=1))
        with self.assertRaisesRegex(ValueError, "not executable"):
            virtual_price(
                halted,
                1,
                runtime.PAPER_CONFIG["arms"]["EXECUTABLE_BASE"],
                entry=True,
                required_units=1000,
            )


if __name__ == "__main__":
    unittest.main()
