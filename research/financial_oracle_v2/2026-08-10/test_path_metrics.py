#!/usr/bin/env python3
"""Bounded invariants for executable-path diagnostics."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("path_metrics", HERE / "build_path_metrics.py")
assert SPEC and SPEC.loader
path_metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(path_metrics)


def candle(bid_low: float, bid_high: float, ask_low: float, ask_high: float) -> dict:
    return {
        "complete": True,
        "bid": {"l": str(bid_low), "h": str(bid_high)},
        "ask": {"l": str(ask_low), "h": str(ask_high)},
    }


def state(side: str) -> dict:
    return {
        "trade": {"side": side, "entry_price": 100.0},
        "observed_starts": set(), "duplicate_bars": 0, "incomplete_bars": 0,
        "mfe": None, "mae": None, "mfe_time": None, "mae_time": None,
        "first_touch": None, "tp_pre_mae": None, "tp_pre_mae_time": None,
        "schedule_change_bars": 0, "dual_touch_bars": 0, "touch_count": 0,
    }


class PathMetricTests(unittest.TestCase):
    def test_nanosecond_parser_and_bar_rounding(self) -> None:
        value = path_metrics.parse_ns("2026-05-06T17:25:15.850070092Z")
        self.assertEqual(value % path_metrics.NS, 850070092)
        self.assertEqual(path_metrics.ns_to_utc(value), "2026-05-06T17:25:15.850070092Z")
        self.assertEqual(path_metrics.ceil_bar(value) % path_metrics.BAR_NS, 0)

    def test_long_uses_bid_wick(self) -> None:
        current = state("LONG")
        path_metrics.update_state(current, candle(98, 103, 99, 999), 0, [])
        self.assertEqual(current["mfe"], 3)
        self.assertEqual(current["mae"], 2)

    def test_short_uses_ask_wick(self) -> None:
        current = state("SHORT")
        path_metrics.update_state(current, candle(1, 999, 97, 104), 0, [])
        self.assertEqual(current["mfe"], 3)
        self.assertEqual(current["mae"], 4)

    def test_dual_touch_is_unresolved(self) -> None:
        current = state("LONG")
        events = [{"ts_ns": 0, "event_kind": "CREATE", "active_tp_price": 102.0, "active_sl_price": 98.0}]
        path_metrics.update_state(current, candle(97, 103, 98, 104), 0, events)
        self.assertEqual(current["first_touch"]["status"], "UNRESOLVED_DUAL_TOUCH_WITHIN_S5")
        self.assertEqual(current["dual_touch_bars"], 1)

    def test_protection_change_inside_touch_bar_is_unresolved(self) -> None:
        current = state("LONG")
        events = [
            {"ts_ns": 0, "event_kind": "CREATE", "active_tp_price": 102.0, "active_sl_price": None},
            {"ts_ns": 1, "event_kind": "CREATE", "active_tp_price": 103.0, "active_sl_price": None},
        ]
        path_metrics.update_state(current, candle(99, 102.5, 100, 103), 0, events)
        self.assertEqual(current["first_touch"]["status"], "UNRESOLVED_PROTECTION_CHANGE_WITHIN_S5")

    def test_margin_timeline_applies_same_transaction_atomically(self) -> None:
        execution = {
            "a": {"entry_units": 10, "entry_actual_initial_margin_jpy": 100},
            "b": {"entry_units": 10, "entry_actual_initial_margin_jpy": 200},
        }
        events = [
            {"ts_utc": "2026-01-01T00:00:00Z", "transaction_id": "1", "trade_id": "a", "kind": "ENTRY", "delta_units": 10},
            {"ts_utc": "2026-01-01T00:01:00Z", "transaction_id": "2", "trade_id": "a", "kind": "TERMINAL_CLOSE", "delta_units": -10},
            {"ts_utc": "2026-01-01T00:01:00Z", "transaction_id": "2", "trade_id": "b", "kind": "ENTRY", "delta_units": 10},
            {"ts_utc": "2026-01-01T00:02:00Z", "transaction_id": "3", "trade_id": "b", "kind": "TERMINAL_CLOSE", "delta_units": -10},
        ]
        timeline, report = path_metrics.build_margin_timeline(execution, events)
        self.assertEqual(timeline[1]["event_count"], 2)
        self.assertEqual(timeline[1]["cohort_required_margin_proxy_jpy"], 200)
        self.assertEqual(report["peak_gross_trade_required_margin_proxy_jpy"], 200)

    def test_saved_missingness_is_not_zero_imputed(self) -> None:
        rows = path_metrics.read_jsonl(HERE / "path_metrics_v1.jsonl")
        missing_pair = [row for row in rows if "PAIR_S5_BID_ASK_SOURCE_MISSING" in row["path_reason_codes"]]
        self.assertTrue(missing_pair)
        self.assertTrue(all(row["expected_full_s5_endpoints"] is None for row in missing_pair))
        self.assertTrue(all(row["mfe_observed_lower_bound_price"] is None for row in missing_pair))

    def test_saved_financial_gate_and_strict_counts(self) -> None:
        report = json.loads((HERE / "path_report_v1.json").read_text())
        self.assertEqual(report["financial_gate"]["corrected_64d_validation_net_jpy"], 11706.0523)
        self.assertEqual(report["episodes"], 251)
        self.assertEqual(report["strict_path_pass"] + report["strict_path_unresolved"], 251)
        self.assertEqual(report["margin"]["account_available_margin_coverage"], 0)


if __name__ == "__main__":
    unittest.main()
