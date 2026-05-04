"""Tests for the tier-aware economic-calendar gating."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.analysis.calendar_tier import (
    CalendarEvent,
    CalendarSnapshot,
    PairWindow,
    build_pair_windows,
    event_tier,
)


def _evt(name: str, currency: str, when: datetime) -> CalendarEvent:
    return CalendarEvent(name=name, currency=currency, scheduled_at_utc=when)


class EventTierTests(unittest.TestCase):
    def test_tier_s_examples(self) -> None:
        now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
        for name in ("FOMC Rate Decision", "US NFP report", "BOJ Rate Decision"):
            self.assertEqual(event_tier(_evt(name, "USD", now)), "S", msg=name)

    def test_tier_a_examples(self) -> None:
        now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
        for name in ("ECB Rate Decision", "BoE Bank Rate", "Core PCE"):
            self.assertEqual(event_tier(_evt(name, "EUR", now)), "A", msg=name)

    def test_tier_b_examples(self) -> None:
        now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
        for name in ("US Retail Sales", "ADP Employment", "JOLTS Job Openings"):
            self.assertEqual(event_tier(_evt(name, "USD", now)), "B", msg=name)

    def test_tier_c_examples(self) -> None:
        now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
        for name in ("US Trade Balance", "Housing Starts", "Industrial Production"):
            self.assertEqual(event_tier(_evt(name, "USD", now)), "C", msg=name)

    def test_unknown_event_is_none_tier(self) -> None:
        now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(event_tier(_evt("Random Speech", "USD", now)), "none")


class PairWindowTests(unittest.TestCase):
    def test_pair_picks_highest_tier_inside_window(self) -> None:
        now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
        events = [
            _evt("US Retail Sales", "USD", now + timedelta(minutes=10)),
            _evt("FOMC Rate Decision", "USD", now + timedelta(minutes=15)),
        ]
        windows = build_pair_windows(pairs=["USD_JPY"], events=events, now=now)
        self.assertEqual(len(windows), 1)
        w = windows[0]
        self.assertTrue(w.in_window)
        self.assertEqual(w.tier, "S")
        self.assertEqual(w.minutes_to_event, 15)

    def test_event_outside_window_is_skipped(self) -> None:
        now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
        events = [
            _evt("FOMC Rate Decision", "USD", now + timedelta(minutes=120)),
        ]
        windows = build_pair_windows(pairs=["USD_JPY"], events=events, now=now)
        self.assertFalse(windows[0].in_window)
        self.assertEqual(windows[0].tier, "none")

    def test_currency_filter(self) -> None:
        now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
        events = [_evt("ECB Rate Decision", "EUR", now + timedelta(minutes=5))]
        windows = build_pair_windows(pairs=["USD_JPY"], events=events, now=now)
        self.assertFalse(windows[0].in_window)


class CalendarSnapshotBackwardsCompatTests(unittest.TestCase):
    def test_to_dict_has_legacy_in_window_field(self) -> None:
        now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
        evt = _evt("FOMC Rate Decision", "USD", now + timedelta(minutes=10))
        windows = build_pair_windows(pairs=["USD_JPY"], events=[evt], now=now)
        snapshot = CalendarSnapshot(
            fetched_at_utc=now, events=(evt,), pair_windows=windows
        )
        d = snapshot.to_dict()
        self.assertIn("events", d)
        self.assertIn("pair_windows", d)
        # Legacy boolean preserved alongside new fields.
        self.assertIn("in_window", d["pair_windows"][0])
        self.assertIn("tier", d["pair_windows"][0])
        self.assertIn("minutes_to_event", d["pair_windows"][0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
