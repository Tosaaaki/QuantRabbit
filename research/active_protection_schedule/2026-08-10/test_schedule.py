#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("build_schedule", ROOT / "build_schedule.py")
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def episode() -> dict:
    return {
        "episode_id": "episode-1",
        "trade_id": "101",
        "pair": "EUR_USD",
        "side": "LONG",
        "feature_at_utc": "2026-01-01T00:00:00Z",
        "fill_at_utc": "2026-01-01T00:00:00Z",
        "close_at_utc": "2026-01-01T01:00:00Z",
    }


def create(order_id: str, tx: str, kind: str, ts: str, *, reason: str = "ON_FILL", old: str | None = None, cancel_tx: str | None = None) -> dict:
    return {
        "event_kind": "CREATE",
        "event_uid": f"create-{tx}",
        "ts_utc": ts,
        "transaction_id": tx,
        "trade_id": "101",
        "protection_kind": kind,
        "protection_type": "TAKE_PROFIT_ORDER" if kind == "TP" else "STOP_LOSS_ORDER",
        "protection_order_id": order_id,
        "price": 1.2 if kind == "TP" else 1.1,
        "reason": reason,
        "replaces_order_id": old,
        "cancelling_transaction_id": cancel_tx,
        "normalized_order_id": old or order_id,
        "raw_sha256": "a" * 64,
    }


def cancel(order_id: str, tx: str, ts: str, *, new: str | None = None, reason: str = "CLIENT_REQUEST_REPLACED") -> dict:
    return {
        "event_kind": "CANCEL",
        "event_uid": f"cancel-{tx}",
        "ts_utc": ts,
        "transaction_id": tx,
        "cancelled_order_id": order_id,
        "reason": reason,
        "replaced_by_order_id": new,
        "closed_trade_id": None,
        "raw_sha256": "b" * 64,
        "trade_id": "101",
        "protection_kind": "TP",
    }


def terminal(order_id: str, tx: str = "20") -> dict:
    return {
        "event_kind": "TERMINAL",
        "event_uid": f"terminal-{tx}",
        "ts_utc": "2026-01-01T01:00:00Z",
        "transaction_id": tx,
        "trade_id": "101",
        "terminal_order_id": order_id,
        "terminal_reason": "TAKE_PROFIT_ORDER",
        "raw_sha256": "c" * 64,
    }


class ScheduleTests(unittest.TestCase):
    def test_initial_tp_and_sl_are_both_active(self) -> None:
        tp = create("10", "10", "TP", "2026-01-01T00:00:00Z")
        sl = create("11", "11", "SL", "2026-01-01T00:00:00Z")
        rows, summary = module.build_trade_schedule(episode(), [tp, sl], {"10": tp, "11": sl}, [], [terminal("10")])
        self.assertTrue(summary["strict_schedule_eligible"])
        self.assertEqual(rows[-1]["active_tp_order_id"], "10")
        self.assertEqual(rows[-1]["active_sl_order_id"], "11")
        self.assertTrue(summary["terminal_active_match"])

    def test_replacement_cancel_then_create_uses_numeric_transaction_order(self) -> None:
        first = create("10", "10", "TP", "2026-01-01T00:00:00Z")
        removed = cancel("10", "12", "2026-01-01T00:30:00Z", new="13")
        second = create("13", "13", "TP", "2026-01-01T00:30:00Z", reason="REPLACEMENT", old="10", cancel_tx="12")
        rows, summary = module.build_trade_schedule(episode(), [first, second], {"10": first, "13": second}, [removed], [terminal("13", "20")])
        self.assertTrue(summary["strict_schedule_eligible"])
        self.assertEqual([row["event_kind"] for row in rows[1:3]], ["CANCEL", "CREATE"])
        self.assertEqual(rows[-1]["active_tp_order_id"], "13")

    def test_missing_replacement_cancel_fails_closed(self) -> None:
        first = create("10", "10", "TP", "2026-01-01T00:00:00Z")
        second = create("13", "13", "TP", "2026-01-01T00:30:00Z", reason="REPLACEMENT", old="10", cancel_tx="12")
        _, summary = module.build_trade_schedule(episode(), [first, second], {"10": first, "13": second}, [], [terminal("13")])
        self.assertFalse(summary["strict_schedule_eligible"])
        self.assertIn("REPLACEMENT_CANCEL_MISSING", summary["issues"])

    def test_replacement_new_id_mismatch_fails_closed(self) -> None:
        first = create("10", "10", "TP", "2026-01-01T00:00:00Z")
        removed = cancel("10", "12", "2026-01-01T00:30:00Z", new="99")
        second = create("13", "13", "TP", "2026-01-01T00:30:00Z", reason="REPLACEMENT", old="10", cancel_tx="12")
        _, summary = module.build_trade_schedule(episode(), [first, second], {"10": first, "13": second}, [removed], [terminal("13")])
        self.assertFalse(summary["strict_schedule_eligible"])
        self.assertIn("REPLACEMENT_CANCEL_NEW_MISMATCH", summary["issues"])

    def test_terminal_must_match_active_protection(self) -> None:
        first = create("10", "10", "TP", "2026-01-01T00:00:00Z")
        other = create("99", "99", "TP", "2026-01-02T00:00:00Z")
        _, summary = module.build_trade_schedule(episode(), [first], {"10": first, "99": other}, [], [terminal("99")])
        self.assertFalse(summary["strict_schedule_eligible"])
        self.assertIn("TERMINAL_ACTIVE_ORDER_MISMATCH", summary["issues"])

    def test_create_before_fill_is_rejected(self) -> None:
        first = create("10", "10", "TP", "2025-12-31T23:59:59Z")
        _, summary = module.build_trade_schedule(episode(), [first], {"10": first}, [], [terminal("10")])
        self.assertFalse(summary["strict_schedule_eligible"])
        self.assertIn("CREATE_BEFORE_FILL", summary["issues"])

    def test_linked_close_cancel_does_not_erase_preterminal_state(self) -> None:
        first = create("10", "10", "TP", "2026-01-01T00:00:00Z")
        linked = cancel("10", "21", "2026-01-01T01:00:00Z", reason="LINKED_TRADE_CLOSED")
        rows, summary = module.build_trade_schedule(episode(), [first], {"10": first}, [linked], [terminal("10", "20")])
        self.assertTrue(summary["strict_schedule_eligible"])
        self.assertEqual(rows[-1]["effect"], "POST_TERMINAL_LINKED_CANCEL")
        self.assertEqual(rows[-1]["active_tp_order_id"], "10")

    def test_output_hash_is_deterministic(self) -> None:
        first = create("10", "10", "TP", "2026-01-01T00:00:00Z")
        args = (episode(), [first], {"10": first}, [], [terminal("10")])
        rows_a, summary_a = module.build_trade_schedule(*args)
        rows_b, summary_b = module.build_trade_schedule(*args)
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(summary_a, summary_b)


if __name__ == "__main__":
    unittest.main()
