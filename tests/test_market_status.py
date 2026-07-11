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

    def test_summer_week_boundary_follows_new_york_daylight_saving_time(self) -> None:
        before_close = compute_market_status(
            datetime(2026, 7, 10, 20, 59, 59, tzinfo=timezone.utc)
        )
        at_close = compute_market_status(
            datetime(2026, 7, 10, 21, 0, tzinfo=timezone.utc)
        )
        before_open = compute_market_status(
            datetime(2026, 7, 12, 20, 59, 59, tzinfo=timezone.utc)
        )
        at_open = compute_market_status(
            datetime(2026, 7, 12, 21, 0, tzinfo=timezone.utc)
        )

        self.assertTrue(before_close.is_fx_open)
        self.assertFalse(at_close.is_fx_open)
        self.assertEqual(at_close.closed_reason, "FRIDAY_AFTER_CLOSE")
        self.assertFalse(before_open.is_fx_open)
        self.assertTrue(at_open.is_fx_open)
        self.assertEqual(
            at_open.most_recent_open_utc,
            "2026-07-12T21:00:00+00:00",
        )

    def test_winter_week_boundary_remains_at_2200_utc(self) -> None:
        before_close = compute_market_status(
            datetime(2026, 1, 9, 21, 59, 59, tzinfo=timezone.utc)
        )
        at_close = compute_market_status(
            datetime(2026, 1, 9, 22, 0, tzinfo=timezone.utc)
        )
        before_open = compute_market_status(
            datetime(2026, 1, 11, 21, 59, 59, tzinfo=timezone.utc)
        )
        at_open = compute_market_status(
            datetime(2026, 1, 11, 22, 0, tzinfo=timezone.utc)
        )

        self.assertTrue(before_close.is_fx_open)
        self.assertFalse(at_close.is_fx_open)
        self.assertFalse(before_open.is_fx_open)
        self.assertTrue(at_open.is_fx_open)
        self.assertEqual(
            at_open.most_recent_open_utc,
            "2026-01-11T22:00:00+00:00",
        )

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
            self.assertEqual(payload["contract"]["market_timezone"], "America/New_York")
            self.assertTrue(payload["contract"]["utc_boundary_is_dst_aware"])
            self.assertIn("Market Status Report", report.read_text())
            self.assertIn("America/New_York", report.read_text())


if __name__ == "__main__":
    unittest.main()
