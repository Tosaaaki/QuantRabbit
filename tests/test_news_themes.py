"""Unit tests for strategy/news_themes.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.strategy.news_themes import (
    NEWS_EXPLICIT_PAIR_BIAS,
    NEWS_MAX_TOTAL_BIAS,
    parse_news_themes,
)


def _write_digest(tmp: Path, content: str) -> Path:
    path = tmp / "news_digest.md"
    path.write_text(content, encoding="utf-8")
    return path


class NewsThemesTest(unittest.TestCase):
    def test_missing_file_returns_empty(self) -> None:
        themes = parse_news_themes(Path("/nonexistent.md"))
        self.assertEqual(themes.biases, {})
        self.assertEqual(themes.detected_themes, ())

    def test_empty_digest_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_digest(Path(tmp), "")
            themes = parse_news_themes(path)
            self.assertEqual(themes.biases, {})

    def test_usd_strong_biases_usd_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            content = "## Today\n\nThe US Dollar rallies strong after CPI. USD surge continues.\n"
            path = _write_digest(Path(tmp), content)
            themes = parse_news_themes(path)
            # USD-quote pairs should be biased SHORT (sell EUR/USD when USD strong)
            self.assertLess(themes.biases[("EUR_USD", "LONG")], 0)
            self.assertGreater(themes.biases[("EUR_USD", "SHORT")], 0)
            self.assertLess(themes.biases[("GBP_USD", "LONG")], 0)
            # USD-base pairs should be biased LONG (buy USD/JPY when USD strong)
            self.assertGreater(themes.biases[("USD_JPY", "LONG")], 0)
            self.assertLess(themes.biases[("USD_JPY", "SHORT")], 0)
            self.assertIn("USD", "|".join(themes.detected_themes))

    def test_jpy_weakness_biases_jpy_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            content = "Yen tumbling against majors. JPY weakness persists.\n"
            path = _write_digest(Path(tmp), content)
            themes = parse_news_themes(path)
            # JPY weak → JPY-quote pairs LONG +bias (USD/JPY LONG favored)
            self.assertGreater(themes.biases[("USD_JPY", "LONG")], 0)
            self.assertGreater(themes.biases[("EUR_JPY", "LONG")], 0)

    def test_jpy_intervention_biases_against_jpy_shorts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            content = "JPY intervention and rate-check risk: avoid being short JPY through USD_JPY longs.\n"
            path = _write_digest(Path(tmp), content)
            themes = parse_news_themes(path)
            # JPY intervention risk is a yen-strength shock risk, not a
            # momentum reason to buy JPY crosses.
            self.assertLess(themes.biases[("USD_JPY", "LONG")], 0)
            self.assertGreater(themes.biases[("USD_JPY", "SHORT")], 0)
            self.assertLess(themes.biases[("EUR_JPY", "LONG")], 0)

    def test_explicit_pair_note_bearish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            content = "**EUR_USD**: Bearish below 1.18, breakdown confirmed\n"
            path = _write_digest(Path(tmp), content)
            themes = parse_news_themes(path)
            self.assertLessEqual(themes.biases[("EUR_USD", "LONG")], -NEWS_EXPLICIT_PAIR_BIAS / 2)
            self.assertGreaterEqual(themes.biases[("EUR_USD", "SHORT")], NEWS_EXPLICIT_PAIR_BIAS / 2)

    def test_explicit_pair_note_bullish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            content = "**USD_JPY**: Bullish breakout above 160, momentum favors longs\n"
            path = _write_digest(Path(tmp), content)
            themes = parse_news_themes(path)
            self.assertGreater(themes.biases[("USD_JPY", "LONG")], 0)
            self.assertLess(themes.biases[("USD_JPY", "SHORT")], 0)

    def test_multi_pair_note_splits_bias(self) -> None:
        """`**GBP_USD, USD_JPY** — bearish ...` applies bias to BOTH pairs."""
        with tempfile.TemporaryDirectory() as tmp:
            content = "**GBP_USD, USD_JPY** — Both bearish on intervention talk\n"
            path = _write_digest(Path(tmp), content)
            themes = parse_news_themes(path)
            self.assertLess(themes.biases[("GBP_USD", "LONG")], 0)
            self.assertLess(themes.biases[("USD_JPY", "LONG")], 0)

    def test_plain_pair_specific_bullets_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            content = (
                "# FX News Digest — 2026-06-06 05:18 JST\n"
                "## 💱 Pair-Specific Notes\n"
                "- GBP_USD: momentum remains lower while the USD impulse is intact.\n"
                "- AUD_USD / NZD_USD: vulnerable if risk appetite softens.\n"
                "- USD_JPY: intervention risk can cap upside even when USD is broadly firm.\n"
            )
            path = _write_digest(Path(tmp), content)
            themes = parse_news_themes(path)

        self.assertGreater(themes.biases[("GBP_USD", "SHORT")], 0)
        self.assertLess(themes.biases[("AUD_USD", "LONG")], 0)
        self.assertLess(themes.biases[("NZD_USD", "LONG")], 0)
        self.assertLess(themes.biases[("USD_JPY", "LONG")], 0)

    def test_risk_off_biases_safe_havens_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            content = "Risk-off mode dominating. Sell-off in equities.\n"
            path = _write_digest(Path(tmp), content)
            themes = parse_news_themes(path)
            # JPY should be strengthened → JPY pairs SHORT favored
            self.assertGreater(themes.biases[("USD_JPY", "SHORT")], 0)
            # AUD should weaken in risk-off → AUD/USD SHORT favored
            self.assertGreater(themes.biases[("AUD_USD", "SHORT")], 0)
            self.assertIn("risk-off", themes.detected_themes)

    def test_bias_clamped_to_max(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Stack many positive USD signals + bearish EUR note
            content = (
                "USD strong USD rally USD surging USD bid USD higher\n"
                "Dollar climbing Dollar gaining Dollar firmer.\n"
                "**EUR_USD**: Bearish breakdown below 1.17\n"
            )
            path = _write_digest(Path(tmp), content)
            themes = parse_news_themes(path)
            # EUR_USD SHORT should be capped at NEWS_MAX_TOTAL_BIAS even
            # with multiple compounding signals.
            self.assertLessEqual(themes.biases[("EUR_USD", "SHORT")], NEWS_MAX_TOTAL_BIAS)
            self.assertGreaterEqual(themes.biases[("EUR_USD", "LONG")], -NEWS_MAX_TOTAL_BIAS)

    def test_for_pair_lookup_returns_rationale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            content = "USD rally strong after CPI\n"
            path = _write_digest(Path(tmp), content)
            themes = parse_news_themes(path)
            delta, rationale = themes.for_pair("USD_JPY", "LONG")
            self.assertGreater(delta, 0)
            self.assertIsNotNone(rationale)
            self.assertIn("USD_JPY", rationale)

    def test_for_pair_unknown_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            content = "Nothing relevant here.\n"
            path = _write_digest(Path(tmp), content)
            themes = parse_news_themes(path)
            delta, rationale = themes.for_pair("EUR_USD", "LONG")
            self.assertEqual(delta, 0.0)
            self.assertIsNone(rationale)

    def test_pre_event_digest_is_ignored_after_named_event_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = _write_digest(
                root,
                "# FX News Digest — 2026-06-05 21:18 JST\n"
                "NFP is the dominant macro event in the next 1-4 hours. USD softer ahead of the release.\n",
            )
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-05T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "Non-Farm Employment Change",
                            }
                        ]
                    }
                )
            )

            themes = parse_news_themes(
                digest,
                calendar_path=calendar,
                now_utc=datetime(2026, 6, 5, 13, 0, tzinfo=timezone.utc),
            )

        self.assertEqual(themes.biases, {})
        self.assertIn("stale_pre_event_digest", themes.detected_themes)

    def test_post_event_digest_with_pre_event_language_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = _write_digest(
                root,
                "# FX News Digest — 2026-06-05 22:17 JST\n"
                "NFP is the dominant macro event in the next 1-4 hours. USD softer ahead of payrolls.\n",
            )
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-05T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "Non-Farm Employment Change",
                            }
                        ]
                    }
                )
            )

            themes = parse_news_themes(
                digest,
                calendar_path=calendar,
                now_utc=datetime(2026, 6, 5, 13, 30, tzinfo=timezone.utc),
            )

        self.assertEqual(themes.biases, {})
        self.assertIn("stale_pre_event_digest", themes.detected_themes)

    def test_pre_event_digest_still_applies_before_event_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = _write_digest(
                root,
                "# FX News Digest — 2026-06-05 21:18 JST\n"
                "NFP is the dominant macro event in the next 1-4 hours. USD softer ahead of the release.\n",
            )
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-05T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "Non-Farm Employment Change",
                            }
                        ]
                    }
                )
            )

            themes = parse_news_themes(
                digest,
                calendar_path=calendar,
                now_utc=datetime(2026, 6, 5, 12, 20, tzinfo=timezone.utc),
            )

        self.assertGreater(themes.biases[("EUR_USD", "LONG")], 0)

    def test_calendar_feed_missing_drops_aged_pre_event_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = _write_digest(
                root,
                "# FX News Digest — 2026-06-05 21:18 JST\n"
                "NFP is the dominant macro event in the next 1-4 hours. USD softer ahead of payrolls.\n",
            )
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [],
                        "issues": ["MISSING_FOREX_FACTORY_FEED: HTTP Error 429: Too Many Requests"],
                    }
                )
            )

            themes = parse_news_themes(
                digest,
                calendar_path=calendar,
                now_utc=datetime(2026, 6, 5, 13, 10, tzinfo=timezone.utc),
            )

        self.assertEqual(themes.biases, {})
        self.assertIn("stale_pre_event_digest", themes.detected_themes)

    def test_post_event_news_items_feed_currency_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = _write_digest(root, "")
            news_items = root / "news_items.json"
            news_items.write_text(
                json.dumps(
                    {
                        "lookback_hours": 24,
                        "items": [
                            {
                                "title": "Euro drops as strong US jobs data lifts Greenback",
                                "summary": "Blowout NFP sends the US Dollar higher.",
                                "published_at_utc": "2026-06-05T14:20:00+00:00",
                                "currencies": ["EUR", "USD"],
                                "pairs": ["EUR_USD"],
                                "topics": ["employment"],
                            }
                        ],
                    }
                )
            )

            themes = parse_news_themes(
                digest,
                news_items_path=news_items,
                now_utc=datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc),
            )

        self.assertGreater(themes.biases[("EUR_USD", "SHORT")], 0)
        self.assertLess(themes.biases[("EUR_USD", "LONG")], 0)
        self.assertIn("news_items:1", themes.detected_themes)

    def test_stale_pre_event_digest_does_not_suppress_post_event_news_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = _write_digest(
                root,
                "# FX News Digest — 2026-06-05 21:18 JST\n"
                "NFP is ahead of the release and USD is lower before payrolls.\n",
            )
            calendar = root / "economic_calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-05T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "Non-Farm Employment Change",
                            }
                        ]
                    }
                )
            )
            news_items = root / "news_items.json"
            news_items.write_text(
                json.dumps(
                    {
                        "lookback_hours": 24,
                        "items": [
                            {
                                "title": "NFP steamrolls US Dollar bears as Greenback rallies",
                                "summary": "",
                                "published_at_utc": "2026-06-05T16:50:00+00:00",
                                "currencies": ["USD"],
                                "pairs": [],
                                "topics": ["employment"],
                            }
                        ],
                    }
                )
            )

            themes = parse_news_themes(
                digest,
                calendar_path=calendar,
                news_items_path=news_items,
                now_utc=datetime(2026, 6, 5, 17, 0, tzinfo=timezone.utc),
            )

        self.assertIn("stale_pre_event_digest", themes.detected_themes)
        self.assertGreater(themes.biases[("EUR_USD", "SHORT")], 0)


if __name__ == "__main__":
    unittest.main()
