#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("validate_oanda_extension", ROOT / "validate_oanda_extension.py")
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def row(time: str, bid: float = 1.0, ask: float = 1.1) -> dict:
    return {
        "time": time,
        "complete": True,
        "granularity": "S5",
        "price": "BA",
        "bid": {"o": bid, "h": bid, "l": bid, "c": bid},
        "ask": {"o": ask, "h": ask, "l": ask, "c": ask},
    }


class ExtensionValidationTests(unittest.TestCase):
    def test_valid_rows_and_unresolved_gap_are_distinct(self) -> None:
        result = module.validate_rows([
            row("2026-07-09T00:00:00.000000000Z"),
            row("2026-07-09T00:00:10.000000000Z"),
        ])
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["gap_count_over_5s"], 1)
        self.assertIn("UNRESOLVED", result["gap_classification"])

    def test_duplicate_time_fails(self) -> None:
        result = module.validate_rows([
            row("2026-07-09T00:00:00.000000000Z"),
            row("2026-07-09T00:00:00.000000000Z"),
        ])
        self.assertIn("DUPLICATE_TIME", result["issues"])

    def test_bid_above_ask_fails(self) -> None:
        result = module.validate_rows([row("2026-07-09T00:00:00.000000000Z", bid=1.2, ask=1.1)])
        self.assertIn("BID_ABOVE_ASK", result["issues"])

    def test_incomplete_fails(self) -> None:
        sample = row("2026-07-09T00:00:00.000000000Z")
        sample["complete"] = False
        self.assertIn("INCOMPLETE_CANDLE", module.validate_rows([sample])["issues"])


if __name__ == "__main__":
    unittest.main()
