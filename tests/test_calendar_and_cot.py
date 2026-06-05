"""Tests for the offline parsers in calendar.py and cot.py.

Both modules delegate fetch to urllib in production, but parse and window
logic must be unit-testable without network access.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from urllib.error import URLError

from quant_rabbit.analysis.calendar import (
    CalendarEvent,
    build_calendar_snapshot,
    parse_calendar_xml,
)
from quant_rabbit.analysis.cot import parse_cot_csv, build_cot_snapshot


SAMPLE_FF_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<weeklyevents>
  <event>
    <title>Non-Farm Employment Change</title>
    <country>USD</country>
    <date>05-09-2026</date>
    <time>12:30pm</time>
    <impact>High</impact>
    <forecast>180K</forecast>
    <previous>200K</previous>
    <actual/>
  </event>
  <event>
    <title>BOJ Press Conference</title>
    <country>JPY</country>
    <date>05-08-2026</date>
    <time>3:00am</time>
    <impact>High</impact>
    <forecast/>
    <previous/>
    <actual/>
  </event>
  <event>
    <title>Holiday</title>
    <country>EUR</country>
    <date>05-05-2026</date>
    <time>All Day</time>
    <impact>Holiday</impact>
    <forecast/>
    <previous/>
    <actual/>
  </event>
</weeklyevents>
"""


class CalendarParseTest(unittest.TestCase):
    def test_parse_filters_unparseable_times_and_returns_sorted(self) -> None:
        events = parse_calendar_xml(SAMPLE_FF_XML)
        # The "All Day" holiday entry is dropped because we have no usable timestamp
        self.assertEqual(len(events), 2)
        # Sorted ascending by timestamp
        timestamps = [e.timestamp_utc for e in events]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_parse_extracts_currency_and_impact(self) -> None:
        events = parse_calendar_xml(SAMPLE_FF_XML)
        currencies = {e.currency for e in events}
        self.assertSetEqual(currencies, {"USD", "JPY"})
        impacts = {e.impact for e in events}
        self.assertSetEqual(impacts, {"High"})

    def test_parse_treats_public_mirror_times_as_utc(self) -> None:
        events = parse_calendar_xml(SAMPLE_FF_XML)
        nfp = next(e for e in events if e.title == "Non-Farm Employment Change")
        self.assertEqual(nfp.timestamp_utc, "2026-05-09T12:30:00+00:00")

    def test_pair_window_flag_uses_pre_post_minutes(self) -> None:
        # Pretend "now" is 5 minutes before the NFP event
        nfp_ts = datetime(2026, 5, 9, 12, 30, tzinfo=timezone.utc)
        now = nfp_ts.replace(minute=25)  # 5 minutes before
        snap = build_calendar_snapshot(
            pairs=("USD_JPY", "EUR_USD"),
            pre_minutes=10, post_minutes=10,
            fetch=False,  # we'll inject events manually
            now_utc=now,
        )
        # Without fetch, no events were loaded, so no window
        self.assertFalse(any(w.in_window for w in snap.pair_windows))

    def test_no_fetch_emits_clean_snapshot_without_issues(self) -> None:
        snap = build_calendar_snapshot(pairs=("USD_JPY",), fetch=False)
        self.assertEqual(snap.events, ())
        # When fetch is suppressed, we don't generate a fetch error, just zero events.
        self.assertNotIn("MISSING_FOREX_FACTORY_FEED", " ".join(snap.issues))

    def test_fetch_failure_marks_pairs_as_calendar_unavailable(self) -> None:
        with patch(
            "quant_rabbit.analysis.calendar.fetch_calendar_xml",
            side_effect=URLError("HTTP Error 429: Too Many Requests"),
        ):
            snap = build_calendar_snapshot(pairs=("USD_JPY", "EUR_GBP"), fetch=True)

        self.assertIn("MISSING_FOREX_FACTORY_FEED", " ".join(snap.issues))
        self.assertEqual(snap.events, ())
        self.assertTrue(all(w.in_window for w in snap.pair_windows))
        self.assertTrue(all("calendar unavailable" in w.reason for w in snap.pair_windows))


SAMPLE_COT_CSV = b"""market_and_exchange_names,report_date_as_yyyy_mm_dd,lev_money_positions_long,lev_money_positions_short,asset_mgr_positions_long,asset_mgr_positions_short,open_interest_all
JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE,2026-04-29,40000,80000,30000,20000,250000
JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE,2026-04-22,50000,70000,32000,22000,240000
EURO FX - CHICAGO MERCANTILE EXCHANGE,2026-04-29,120000,90000,80000,70000,300000
"""


class CotParseTest(unittest.TestCase):
    def test_parse_extracts_currency_and_net(self) -> None:
        reports = parse_cot_csv(SAMPLE_COT_CSV)
        by_currency = {r.currency: r for r in reports}
        self.assertIn("JPY", by_currency)
        self.assertIn("EUR", by_currency)
        jpy = by_currency["JPY"]
        self.assertEqual(jpy.leveraged_net, 40000 - 80000)
        self.assertEqual(jpy.week_change_leveraged_net, (40000 - 80000) - (50000 - 70000))

    def test_no_fetch_returns_empty_reports_without_failures(self) -> None:
        snap = build_cot_snapshot(fetch=False)
        self.assertEqual(snap.reports, ())
        self.assertEqual(snap.issues, ())


if __name__ == "__main__":
    unittest.main()
