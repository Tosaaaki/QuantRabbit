#!/usr/bin/env python3
"""Small fixed fixtures for the no-forced-loss-close accounting contract."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("run_no_forced_loss_replay.py")
SPEC = importlib.util.spec_from_file_location("no_forced_loss_replay", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def bar(minute: int, mid: float, spread: float = 0.02) -> tuple[object, ...]:
    ts = datetime(2025, 1, 2, 8, 0, tzinfo=timezone.utc) + timedelta(minutes=minute)
    bid, ask = mid - spread / 2.0, mid + spread / 2.0
    return (ts, bid, bid, bid, bid, ask, ask, ask, ask)


def decision() -> dict[str, object]:
    return {
        "decision_id": "FIXTURE-001",
        "pair": "USD_JPY",
        "status": "CONFIRMED",
        "side": "LONG",
        "atr": 1.0,
        "quote_to_jpy": 1.0,
        "entry_index": 0,
    }


class AccountingOracleTest(unittest.TestCase):
    def test_naked_inventory_is_mtm_not_force_closed_at_end(self) -> None:
        rows = [bar(0, 100.0), bar(1, 99.8), bar(2, 99.5)]
        result = MODULE.simulate(decision(), rows, "B_NO_SL_NAKED_RETURN_WAIT", "HEDGING")
        expected = MODULE.directional("LONG", 100.01, 99.49, 5_000, 1.0) - 10.0
        self.assertTrue(result["original_open"])
        self.assertFalse(result["margin_closeout"])
        self.assertAlmostEqual(result["terminal_contribution_jpy"], expected)
        self.assertAlmostEqual(result["original_mtm_jpy"], -2_600.0)

    def test_hard_sl_is_comparison_only_realized_loss(self) -> None:
        rows = [bar(0, 100.0), bar(1, 98.8), bar(2, 98.7)]
        result = MODULE.simulate(decision(), rows, "A_HARD_SL_BASELINE", "HEDGING")
        self.assertFalse(result["original_open"])
        self.assertLess(result["original_realized_jpy"], 0.0)
        self.assertEqual(result["original_mtm_jpy"], 0.0)

    def test_hedging_account_keeps_original_and_opposite_leg(self) -> None:
        rows = [bar(0, 100.0), bar(1, 99.9), bar(4, 99.4), bar(5, 99.3), bar(6, 99.2)]
        result = MODULE.simulate(decision(), rows, "H1_LOCK_AT_ADVERSE_LEVEL_AND_WAIT_050", "HEDGING")
        self.assertTrue(result["original_open"])
        self.assertTrue(result["hedge_open"])
        self.assertEqual(result["repeated_hedge_count"], 1)
        self.assertEqual(result["netting_reduction_events"], 0)
        self.assertEqual(result["max_gross_exposure_units"], 10_000)
        self.assertEqual(result["max_net_exposure_units"], 5_000)

    def test_netting_account_reduces_original_instead_of_claiming_hedge(self) -> None:
        rows = [bar(0, 100.0), bar(1, 99.9), bar(4, 99.4), bar(5, 99.3), bar(6, 99.2)]
        result = MODULE.simulate(decision(), rows, "H1_LOCK_AT_ADVERSE_LEVEL_AND_WAIT_050", "NETTING")
        self.assertFalse(result["original_open"])
        self.assertFalse(result["hedge_open"])
        self.assertEqual(result["repeated_hedge_count"], 0)
        self.assertEqual(result["netting_reduction_events"], 1)
        self.assertLess(result["original_realized_jpy"], 0.0)

    def test_recovered_original_is_closed_exactly_once_while_hedge_remains(self) -> None:
        rows = [bar(0, 100.0), bar(1, 99.9), bar(4, 99.4), bar(5, 99.3), bar(6, 100.1), bar(7, 100.2), bar(8, 100.3)]
        result = MODULE.simulate(decision(), rows, "H2_HEDGE_TP_KEEP_ORIGINAL_050", "HEDGING")
        self.assertFalse(result["original_open"])
        self.assertTrue(result["hedge_open"])
        self.assertGreater(result["original_realized_jpy"], 0.0)
        self.assertLess(result["original_realized_jpy"], 2_000.0)

    def test_unknown_financing_never_becomes_zero(self) -> None:
        rows = [bar(0, 100.0)]
        for day in range(1, 3):
            ts = rows[0][0] + timedelta(days=day)
            rows.append((ts, 99.0, 99.0, 99.0, 99.0, 99.02, 99.02, 99.02, 99.02))
        result = MODULE.simulate(decision(), rows, "B_NO_SL_NAKED_RETURN_WAIT", "HEDGING")
        self.assertEqual(result["status"], "NOT_EVALUABLE_FINANCING")
        self.assertIsNone(result["financing_jpy"])
        summary = MODULE.aggregate([result])
        self.assertEqual(summary["accounting_status"], "NOT_EVALUABLE_FINANCING")
        self.assertEqual(summary["decision"], "REJECT")

    def test_broker_forced_liquidation_is_explicit_failure(self) -> None:
        rows = [bar(0, 100.0), bar(1, 49.0), bar(2, 48.0)]
        result = MODULE.simulate(decision(), rows, "B_NO_SL_NAKED_RETURN_WAIT", "HEDGING")
        self.assertTrue(result["margin_closeout"])
        self.assertEqual(result["status"], "MARGIN_CLOSEOUT_FAILURE")
        self.assertFalse(result["profit_only_original_close"])
        self.assertEqual(MODULE.aggregate([result])["decision"], "REJECT")

    def test_admission_limits_preserve_decision_identity(self) -> None:
        rows = [
            {"decision_id": "D1", "pair": "EUR_USD", "arm": "B_NO_SL_NAKED_RETURN_WAIT", "account_mode": "HEDGING", "executed": True, "entry_utc": "2025-01-01T00:00:00Z", "terminal_utc": "2025-01-03T00:00:00Z", "terminal_contribution_jpy": 1.0},
            {"decision_id": "D2", "pair": "USD_JPY", "arm": "B_NO_SL_NAKED_RETURN_WAIT", "account_mode": "HEDGING", "executed": True, "entry_utc": "2025-01-02T00:00:00Z", "terminal_utc": "2025-01-02T01:00:00Z", "terminal_contribution_jpy": 1.0},
        ]
        MODULE.apply_admission_limits(rows)
        self.assertEqual(rows[1]["decision_id"], "D2")
        self.assertEqual(rows[1]["status"], "SKIP_INVENTORY_BUSY_NO_CHASE")


if __name__ == "__main__":
    unittest.main()
