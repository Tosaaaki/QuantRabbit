"""Unit tests for capture_economics.py."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.capture_economics import build_capture_economics


def _make_db(path: Path, closes: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE execution_events (
            event_uid TEXT PRIMARY KEY,
            ts_utc TEXT, event_type TEXT, lane_id TEXT, order_id TEXT,
            trade_id TEXT, pair TEXT, side TEXT, units INTEGER,
            realized_pl_jpy REAL, exit_reason TEXT
        )
        """
    )
    rows = []
    for i, c in enumerate(closes):
        trade_id = c.get("trade_id", f"t{i}")
        if c.get("attributed", True):
            rows.append(
                (f"g{i}", c["ts_utc"], "GATEWAY_ORDER_SENT", "lane:X", f"o{i}", trade_id,
                 c["pair"], "LONG", 1000, None, None)
            )
            rows.append(
                (f"f{i}", c["ts_utc"], "ORDER_FILLED", None, f"o{i}", trade_id,
                 c["pair"], "LONG", 1000, None, None)
            )
        rows.append(
            (f"c{i}", c["ts_utc"], "TRADE_CLOSED", None, f"x{i}", trade_id,
             c["pair"], "SHORT", 1000, c["pl"], c.get("exit_reason", "TAKE_PROFIT_ORDER"))
        )
    conn.executemany(
        "INSERT INTO execution_events VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()


class CaptureEconomicsTest(unittest.TestCase):
    def test_missing_ledger_reports_low_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_capture_economics(
                ledger_path=root / "missing.db",
                output_path=root / "out.json",
                report_path=root / "report.md",
            )
            self.assertEqual(summary.status, "LOW_SAMPLE")
            self.assertEqual(summary.trades, 0)

    def test_negative_expectancy_flagged_from_payoff_vs_breakeven(self) -> None:
        """Model of the 2026-05/06 ledger: high win rate, tiny wins, big losses."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closes = []
            for i in range(21):
                closes.append({"ts_utc": f"2026-06-0{(i % 5) + 1}T10:{i:02d}:00Z", "pair": "EUR_USD", "pl": 300.0})
            for i in range(9):
                closes.append(
                    {"ts_utc": f"2026-06-0{(i % 5) + 1}T12:{i:02d}:00Z", "pair": "EUR_USD",
                     "pl": -1500.0, "exit_reason": "MARKET_ORDER_TRADE_CLOSE"}
                )
            db = root / "ledger.db"
            _make_db(db, closes)
            summary = build_capture_economics(
                ledger_path=db, output_path=root / "out.json", report_path=root / "report.md"
            )
            self.assertEqual(summary.status, "NEGATIVE_EXPECTANCY")
            self.assertEqual(summary.trades, 30)
            self.assertAlmostEqual(summary.win_rate or 0, 0.7)
            self.assertAlmostEqual(summary.payoff_ratio or 0, 0.2)
            # breakeven at 70% win rate = 0.3/0.7 ≈ 0.4286 > payoff 0.2
            self.assertAlmostEqual(summary.breakeven_payoff or 0, 0.4286, places=3)
            payload = json.loads((root / "out.json").read_text())
            self.assertIn("MARKET_ORDER_TRADE_CLOSE", payload["by_exit_reason"])

    def test_positive_expectancy_and_manual_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closes = []
            for i in range(15):
                closes.append({"ts_utc": f"2026-06-02T10:{i:02d}:00Z", "pair": "EUR_USD", "pl": 1000.0})
            for i in range(10):
                closes.append({"ts_utc": f"2026-06-02T12:{i:02d}:00Z", "pair": "EUR_USD", "pl": -800.0})
            # Manual close must be excluded from the audit entirely.
            closes.append({"ts_utc": "2026-06-02T13:00:00Z", "pair": "USD_JPY", "pl": -99999.0, "attributed": False})
            db = root / "ledger.db"
            _make_db(db, closes)
            summary = build_capture_economics(
                ledger_path=db, output_path=root / "out.json", report_path=root / "report.md"
            )
            self.assertEqual(summary.trades, 25)
            self.assertEqual(summary.status, "POSITIVE_EXPECTANCY")
            payload = json.loads((root / "out.json").read_text())
            self.assertNotIn("USD_JPY", json.dumps(payload["by_exit_reason"]))


if __name__ == "__main__":
    unittest.main()
