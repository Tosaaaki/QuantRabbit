from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.analysis.news import (
    ENTRY_SEPARATOR,
    FXSTREET_NEWS_SOURCE,
    MARKETPULSE_SOURCE,
    build_news_snapshot,
    parse_marketpulse_rss,
    render_flow_entry,
    render_news_digest,
    write_news_artifacts,
)


SAMPLE_MARKETPULSE_RSS = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
  <channel>
    <title>MarketPulse</title>
    <item>
      <title>Chart alert: USD/JPY bearish breakdown after intervention</title>
      <link>https://www.marketpulse.com/markets/usdjpy-breakdown/</link>
      <description>USD/JPY extends downside after Japan intervention and a break below support.</description>
      <pubDate>Wed, 06 May 2026 05:04:00 +0000</pubDate>
      <category>FX_USDJPY</category>
      <category>TOP_CentralBankJapan</category>
    </item>
    <item>
      <title>Asia open: risk-on equities lift AUD/USD</title>
      <link>https://www.marketpulse.com/markets/asia-open-risk-on/</link>
      <description>Risk-on sentiment supports the Australian dollar while oil dips.</description>
      <pubDate>Wed, 06 May 2026 04:02:00 +0000</pubDate>
      <category>FX_AUDUSD</category>
      <category>TOP_RiskOn</category>
      <category>COM_Oil</category>
    </item>
  </channel>
</rss>
"""

SAMPLE_FXSTREET_RSS = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
  <channel>
    <title>FXStreet News</title>
    <item>
      <title>Forex Today: US Dollar weakens after jobs data</title>
      <link>https://www.fxstreet.com/news/forex-today-dollar-jobs/</link>
      <description>The US Dollar turns lower as traders react to employment data and Fed pricing.</description>
      <pubDate>Fri, 08 May 2026 04:45:00 +0000</pubDate>
      <category>Forex News</category>
    </item>
  </channel>
</rss>
"""


class NewsSnapshotTest(unittest.TestCase):
    def test_parse_marketpulse_rss_extracts_pairs_and_topics(self) -> None:
        items = parse_marketpulse_rss(SAMPLE_MARKETPULSE_RSS)

        self.assertEqual(len(items), 2)
        by_title = {item.title: item for item in items}
        usdjpy = by_title["Chart alert: USD/JPY bearish breakdown after intervention"]
        self.assertIn("USD_JPY", usdjpy.pairs)
        self.assertIn("JPY", usdjpy.currencies)
        self.assertIn("intervention", usdjpy.topics)
        audusd = by_title["Asia open: risk-on equities lift AUD/USD"]
        self.assertIn("AUD_USD", audusd.pairs)
        self.assertIn("risk_on", audusd.topics)
        self.assertIn("oil", audusd.topics)

    def test_build_snapshot_flags_stale_feed_without_dropping_evidence(self) -> None:
        snap = build_news_snapshot(
            now_utc=datetime(2026, 5, 8, 5, 0, tzinfo=timezone.utc),
            lookback_hours=1,
            marketpulse_payload=SAMPLE_MARKETPULSE_RSS,
            fetch=False,
        )

        self.assertEqual(len(snap.items), 2)
        self.assertTrue(any(issue.startswith("STALE_NEWS_FEED") for issue in snap.issues))

    def test_fresh_secondary_source_prevents_stale_feed_replay(self) -> None:
        snap = build_news_snapshot(
            now_utc=datetime(2026, 5, 8, 5, 0, tzinfo=timezone.utc),
            lookback_hours=1,
            source_payloads={
                MARKETPULSE_SOURCE: SAMPLE_MARKETPULSE_RSS,
                FXSTREET_NEWS_SOURCE: SAMPLE_FXSTREET_RSS,
            },
            fetch=False,
        )

        self.assertEqual(len(snap.items), 1)
        self.assertEqual(snap.items[0].source, FXSTREET_NEWS_SOURCE)
        self.assertIn("employment", snap.items[0].topics)
        self.assertTrue(any(issue.startswith("STALE_MARKETPULSE_FEED") for issue in snap.issues))
        self.assertFalse(any(issue.startswith("STALE_NEWS_FEED") for issue in snap.issues))

    def test_malformed_source_does_not_drop_other_fresh_sources(self) -> None:
        snap = build_news_snapshot(
            now_utc=datetime(2026, 5, 8, 5, 0, tzinfo=timezone.utc),
            lookback_hours=1,
            source_payloads={
                MARKETPULSE_SOURCE: b"<rss><channel><item>",
                FXSTREET_NEWS_SOURCE: SAMPLE_FXSTREET_RSS,
            },
            fetch=False,
        )

        self.assertEqual(len(snap.items), 1)
        self.assertEqual(snap.items[0].source, FXSTREET_NEWS_SOURCE)
        self.assertTrue(any(issue.startswith("MALFORMED_MARKETPULSE_FEED") for issue in snap.issues))

    def test_render_digest_and_flow_are_market_story_compatible(self) -> None:
        snap = build_news_snapshot(
            now_utc=datetime(2026, 5, 6, 6, 0, tzinfo=timezone.utc),
            marketpulse_payload=SAMPLE_MARKETPULSE_RSS,
            fetch=False,
        )

        digest = render_news_digest(snap)
        flow = render_flow_entry(snap)

        self.assertIn("High Impact", digest)
        self.assertIn("**USD_JPY**", digest)
        self.assertIn("Source: MarketPulse", digest)
        self.assertIn("- HOT:", flow)
        self.assertIn("- WATCH:", flow)

    def test_write_news_artifacts_writes_ignored_style_outputs(self) -> None:
        snap = build_news_snapshot(
            now_utc=datetime(2026, 5, 6, 6, 0, tzinfo=timezone.utc),
            marketpulse_payload=SAMPLE_MARKETPULSE_RSS,
            fetch=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "data" / "news_items.json"
            digest = root / "logs" / "news_digest.md"
            flow = root / "logs" / "news_flow_log.md"

            write_news_artifacts(
                snap,
                output_path=output,
                digest_path=digest,
                flow_log_path=flow,
                flow_entries=1,
            )

            payload = json.loads(output.read_text())
            self.assertEqual(payload["items"][0]["source"], "MarketPulse")
            self.assertIn("Pair-Specific Notes", digest.read_text())
            self.assertEqual(flow.read_text().count(ENTRY_SEPARATOR), 1)


if __name__ == "__main__":
    unittest.main()
