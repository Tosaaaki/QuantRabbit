from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.analysis.market_status import compute_market_status, write_report, write_snapshot


class MarketStatusTest(unittest.TestCase):
    def test_monday_open_regression_not_sunday_closed(self) -> None:
        status = compute_market_status(datetime(2026, 5, 4, 15, 53, tzinfo=timezone.utc))

        self.assertTrue(status.is_fx_open)
        self.assertIsNone(status.closed_reason)
        self.assertIn("London", status.active_sessions)
        self.assertIn("New_York", status.active_sessions)
        self.assertEqual(status.minutes_to_next_open, None)
        self.assertGreater(status.minutes_to_next_close or 0, 0)

    def test_sunday_pre_open_is_closed(self) -> None:
        status = compute_market_status(datetime(2026, 5, 3, 15, 53, tzinfo=timezone.utc))

        self.assertFalse(status.is_fx_open)
        self.assertEqual(status.closed_reason, "SUNDAY_PRE_OPEN")
        self.assertGreater(status.minutes_to_next_open or 0, 0)
        self.assertEqual(status.active_sessions, ())

    def test_writes_snapshot_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = compute_market_status(datetime(2026, 5, 4, 15, 53, tzinfo=timezone.utc))
            output = root / "market_status.json"
            report = root / "market_status.md"

            write_snapshot(status, output)
            write_report(status, report)

            payload = json.loads(output.read_text())
            self.assertEqual(payload["evidence_ref"], "market:status")
            self.assertIn("Market Status Report", report.read_text())


if __name__ == "__main__":
    unittest.main()
