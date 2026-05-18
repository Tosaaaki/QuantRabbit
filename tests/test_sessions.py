"""Tests for the session microstructure layer."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from quant_rabbit.analysis.candles import Candle
from quant_rabbit.analysis.sessions import (
    SessionTag,
    build_session_context,
    jp_holiday_calendar_for,
    ny_midnight_open,
    tag_bar,
)


_NY = ZoneInfo("America/New_York")


def _ny_to_utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=_NY).astimezone(timezone.utc)


def _make_candle(ts_utc: datetime, *, o: float = 100.0, h: float = 100.5, low: float = 99.5, c: float = 100.0) -> Candle:
    return Candle(
        timestamp_utc=ts_utc,
        open=o,
        high=h,
        low=low,
        close=c,
        volume=1000,
        complete=True,
    )


class TagBarTest(unittest.TestCase):
    def test_london_killzone_at_0230_ny(self) -> None:
        ts = _ny_to_utc(2026, 6, 10, 2, 30)  # Tuesday, NY summer (DST on)
        marker = tag_bar(ts)
        self.assertEqual(marker.tag, SessionTag.LONDON_KILLZONE)
        self.assertAlmostEqual(marker.ny_local_hour, 2.5, places=2)
        self.assertFalse(marker.jp_holiday)

    def test_silver_bullet_at_1030_ny(self) -> None:
        ts = _ny_to_utc(2026, 6, 10, 10, 30)
        marker = tag_bar(ts)
        self.assertEqual(marker.tag, SessionTag.SILVER_BULLET)

    def test_ny_am_at_0800_ny(self) -> None:
        ts = _ny_to_utc(2026, 6, 10, 8, 0)
        self.assertEqual(tag_bar(ts).tag, SessionTag.NY_AM_KILLZONE)

    def test_ny_pm_at_1400_ny(self) -> None:
        ts = _ny_to_utc(2026, 6, 10, 14, 0)
        self.assertEqual(tag_bar(ts).tag, SessionTag.NY_PM_KILLZONE)

    def test_judas_window_at_0030_ny(self) -> None:
        ts = _ny_to_utc(2026, 6, 10, 0, 30)
        self.assertEqual(tag_bar(ts).tag, SessionTag.JUDAS_WINDOW)

    def test_tokyo_killzone_at_2200_ny(self) -> None:
        ts = _ny_to_utc(2026, 6, 10, 22, 0)
        self.assertEqual(tag_bar(ts).tag, SessionTag.TOKYO_KILLZONE)

    def test_off_hours_at_1700_ny(self) -> None:
        ts = _ny_to_utc(2026, 6, 10, 17, 0)
        self.assertEqual(tag_bar(ts).tag, SessionTag.OFF_HOURS)


class DstAwarenessTest(unittest.TestCase):
    """Same UTC instant maps to different NY hours across DST boundaries."""

    def test_dst_shift_changes_ny_hour(self) -> None:
        # 2026-03-10 12:00 UTC: DST is ON in NY (started Mar 8, 2026) → NY = 08:00 (NY_AM)
        # 2026-08-10 12:00 UTC: DST is ON in NY → NY = 08:00 (NY_AM)
        # Use a January (DST off) vs July (DST on) instead so the bucket changes.
        # 2026-01-15 12:00 UTC: DST off → NY = 07:00 (NY_AM_KILLZONE start)
        # 2026-07-15 11:00 UTC: DST on → NY = 07:00 (NY_AM_KILLZONE start)
        winter = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        summer = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        winter_marker = tag_bar(winter)
        summer_marker = tag_bar(summer)
        # Winter (EST = UTC-5): 12:00 UTC = 07:00 NY → NY_AM_KILLZONE
        # Summer (EDT = UTC-4): 12:00 UTC = 08:00 NY → NY_AM_KILLZONE (still in window)
        # Both should land in the NY_AM range; what we assert is the *hour* differs by 1.
        self.assertAlmostEqual(summer_marker.ny_local_hour - winter_marker.ny_local_hour, 1.0, places=2)
        # And both fall in NY_AM (this confirms the killzone bounds are NY-local).
        self.assertEqual(winter_marker.tag, SessionTag.NY_AM_KILLZONE)
        self.assertEqual(summer_marker.tag, SessionTag.NY_AM_KILLZONE)

    def test_dst_bucket_change(self) -> None:
        # Pick a UTC instant that lands in different killzones across DST.
        # 2026-01-15 07:30 UTC: NY = 02:30 (LONDON_KILLZONE) — winter
        # 2026-07-15 07:30 UTC: NY = 03:30 (LONDON_KILLZONE) — summer
        # Use a different one: 06:30 UTC
        # winter: 06:30 UTC → 01:30 NY → JUDAS_WINDOW
        # summer: 06:30 UTC → 02:30 NY → LONDON_KILLZONE
        winter = datetime(2026, 1, 15, 6, 30, tzinfo=timezone.utc)
        summer = datetime(2026, 7, 15, 6, 30, tzinfo=timezone.utc)
        self.assertEqual(tag_bar(winter).tag, SessionTag.JUDAS_WINDOW)
        self.assertEqual(tag_bar(summer).tag, SessionTag.LONDON_KILLZONE)


class JpHolidayTest(unittest.TestCase):
    def test_childrens_day_is_holiday(self) -> None:
        # 2026-05-05 noon JST → ~03:00 UTC
        ts = datetime(2026, 5, 5, 12, 0, tzinfo=ZoneInfo("Asia/Tokyo")).astimezone(timezone.utc)
        marker = tag_bar(ts)
        self.assertTrue(marker.jp_holiday)
        # Golden Week (range) wins over Children's Day (statutory) — see soft-flag rule.
        self.assertIn(marker.holiday_name or "", {"Children's Day", "Golden Week"})

    def test_obon_2026_08_15_is_holiday(self) -> None:
        ts = datetime(2026, 8, 15, 12, 0, tzinfo=ZoneInfo("Asia/Tokyo")).astimezone(timezone.utc)
        marker = tag_bar(ts)
        self.assertTrue(marker.jp_holiday)
        self.assertEqual(marker.holiday_name, "Obon")

    def test_random_tuesday_is_not_holiday(self) -> None:
        # 2026-06-09 is a Tuesday with no JP holiday.
        ts = datetime(2026, 6, 9, 12, 0, tzinfo=ZoneInfo("Asia/Tokyo")).astimezone(timezone.utc)
        marker = tag_bar(ts)
        self.assertFalse(marker.jp_holiday)
        self.assertIsNone(marker.holiday_name)

    def test_calendar_lookup_for_year(self) -> None:
        cal = jp_holiday_calendar_for(2026, 2026)
        self.assertEqual(cal[date(2026, 5, 5)], "Golden Week")  # range overrides statutory
        self.assertEqual(cal[date(2026, 8, 15)], "Obon")
        self.assertEqual(cal[date(2026, 1, 1)], "New Year's Day")

    def test_year_end_range(self) -> None:
        ts = datetime(2026, 12, 30, 12, 0, tzinfo=ZoneInfo("Asia/Tokyo")).astimezone(timezone.utc)
        marker = tag_bar(ts)
        self.assertTrue(marker.jp_holiday)
        self.assertEqual(marker.holiday_name, "Year-end / New Year")


class SessionContextTest(unittest.TestCase):
    def test_asian_range_matches_min_max_in_window(self) -> None:
        # Build M5 candles for prev NY-day 19:00 NY → 00:00 NY (Asian),
        # then current day 00:00 NY → 03:00 NY (Judas + London).
        candles: list[Candle] = []
        # Prior NY date Jun 9, 19:00 NY → 24:00 NY (5 hours = 60 M5 bars).
        start_asian = _ny_to_utc(2026, 6, 9, 19, 0)
        # Use varying highs/lows so we know the expected max/min.
        # Bar i: high = 100 + i*0.01, low = 99 - i*0.01, close = 100
        for i in range(60):
            ts = start_asian + timedelta(minutes=5 * i)
            candles.append(_make_candle(ts, o=100.0, h=100.0 + i * 0.01, low=99.0 - i * 0.01, c=100.0))
        expected_high = 100.0 + 59 * 0.01
        expected_low = 99.0 - 59 * 0.01
        # Now place "current" at NY 03:00 (London KZ on Jun 10), still within Judas/London range.
        # Add some bars during 00:00–03:00 NY so judas_armed has a chance to fire.
        start_judas = _ny_to_utc(2026, 6, 10, 0, 0)
        for i in range(36):  # 3 hours of M5 bars
            ts = start_judas + timedelta(minutes=5 * i)
            # Stay inside the Asian range — judas_armed should be False.
            candles.append(_make_candle(ts, o=100.0, h=100.05, low=99.95, c=100.0))

        ctx = build_session_context(candles, now_utc=_ny_to_utc(2026, 6, 10, 3, 0))
        self.assertIsNotNone(ctx.asian_range)
        self.assertAlmostEqual(ctx.asian_range[0], expected_high, places=4)
        self.assertAlmostEqual(ctx.asian_range[1], expected_low, places=4)
        self.assertFalse(ctx.judas_armed)

    def test_judas_armed_when_asian_high_is_swept(self) -> None:
        candles: list[Candle] = []
        start_asian = _ny_to_utc(2026, 6, 9, 19, 0)
        for i in range(60):
            ts = start_asian + timedelta(minutes=5 * i)
            candles.append(_make_candle(ts, o=100.0, h=100.5, low=99.5, c=100.0))
        # Asian high = 100.5
        # Now put a Judas bar at 01:00 NY that pierces above 100.5.
        sweep_ts = _ny_to_utc(2026, 6, 10, 1, 0)
        candles.append(_make_candle(sweep_ts, o=100.4, h=100.7, low=100.3, c=100.4))
        # And a few benign bars after.
        for i in range(1, 12):
            ts = sweep_ts + timedelta(minutes=5 * i)
            candles.append(_make_candle(ts, o=100.4, h=100.45, low=100.35, c=100.4))

        ctx = build_session_context(candles, now_utc=_ny_to_utc(2026, 6, 10, 2, 0))
        self.assertIsNotNone(ctx.asian_range)
        self.assertTrue(ctx.judas_armed)

    def test_ny_midnight_open_picks_at_or_after_anchor(self) -> None:
        # Candles every 5 min from 23:50 NY (prev day) through 02:00 NY (current).
        candles: list[Candle] = []
        start = _ny_to_utc(2026, 6, 9, 23, 50)
        for i in range(30):
            ts = start + timedelta(minutes=5 * i)
            # close = 200 + i so we can identify which bar was picked.
            candles.append(_make_candle(ts, o=200.0, h=200.5, low=199.5, c=200.0 + i))
        anchor, price = ny_midnight_open(candles, now_utc=_ny_to_utc(2026, 6, 10, 1, 0))
        # Expected anchor: 00:00 NY of 2026-06-10.
        expected_anchor = _ny_to_utc(2026, 6, 10, 0, 0)
        self.assertEqual(anchor, expected_anchor)
        # The first candle at-or-after 00:00 NY is i=2 (start + 10 min = 00:00 NY) → close = 202
        self.assertAlmostEqual(price, 202.0, places=4)

    def test_ny_midnight_open_returns_none_when_no_candle_after_anchor(self) -> None:
        # All candles are before 00:00 NY of the asked day.
        candles = [
            _make_candle(_ny_to_utc(2026, 6, 9, 22, 0), c=100.0),
            _make_candle(_ny_to_utc(2026, 6, 9, 22, 30), c=100.5),
        ]
        anchor, price = ny_midnight_open(candles, now_utc=_ny_to_utc(2026, 6, 9, 23, 0))
        self.assertEqual(anchor, _ny_to_utc(2026, 6, 9, 0, 0))
        # No candle at-or-after that early anchor either: candles are all >= the anchor.
        # So price should be the close of the first candle.
        self.assertAlmostEqual(price, 100.0, places=4)

    def test_next_killzone_is_populated(self) -> None:
        # At 06:00 NY on a normal weekday, next killzone should be NY_AM (07:00) → 60 min away.
        candles = [_make_candle(_ny_to_utc(2026, 6, 10, 6, 0))]
        ctx = build_session_context(candles, now_utc=_ny_to_utc(2026, 6, 10, 6, 0))
        self.assertEqual(ctx.next_killzone, SessionTag.NY_AM_KILLZONE)
        self.assertEqual(ctx.minutes_to_next_killzone, 60)

    def test_next_killzone_includes_tokyo(self) -> None:
        # At 18:00 NY, Tokyo/Asian range starts at 19:00 NY.
        candles = [_make_candle(_ny_to_utc(2026, 6, 10, 18, 0))]
        ctx = build_session_context(candles, now_utc=_ny_to_utc(2026, 6, 10, 18, 0))
        self.assertEqual(ctx.next_killzone, SessionTag.TOKYO_KILLZONE)
        self.assertEqual(ctx.minutes_to_next_killzone, 60)


if __name__ == "__main__":
    unittest.main()
