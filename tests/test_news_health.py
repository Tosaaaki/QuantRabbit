from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.analysis.news_health import build_news_health
from quant_rabbit.cli import main


FRESH_NOW = datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _touch(path: Path, ts: datetime) -> None:
    os.utime(path, (ts.timestamp(), ts.timestamp()))


def _digest(stamp: str = "2026-06-04 20:30 JST", *, websearch: bool = True) -> str:
    source = "WebSearch + news_fetcher.py" if websearch else "news_fetcher.py"
    return f"""# FX News Digest — {stamp}

## High Impact
- USD remains bid after strong labor evidence.

## Watch List
- EUR/USD watches US data and yields.

## Economic Calendar Today (JST)
- No high-impact release in the next test window.

## Pre-Event Nowcast
- NFP evidence stack: ADP, claims, JOLTS, ISM employment.

## Central Bank Tracker
- Fed: data-dependent.

## Pair-Specific Notes
- **EUR_USD**: Bearish below resistance.

## Risk Events
- Watch US yields.

---
Updated: {stamp} | Sources: {source}
"""


def _write_good_packet(root: Path, now: datetime = FRESH_NOW) -> dict[str, Path]:
    data = root / "data"
    logs = root / "logs"
    news_items = data / "news_items.json"
    digest = logs / "news_digest.md"
    flow = logs / "news_flow_log.md"
    profile = data / "market_story_profile.json"
    calendar = data / "economic_calendar.json"
    automation = root / "automation.toml"
    weekend = root / "weekend_state.json"

    items = []
    for idx in range(10):
        items.append(
            {
                "source": "FXStreet News" if idx % 2 else "ForexLive",
                "title": f"USD jobs and yields item {idx}",
                "summary": "Fresh macro evidence for the dollar.",
                "published_at_utc": (now - timedelta(minutes=20 + idx)).isoformat(),
                "topics": ["employment", "yields"],
            }
        )
    _write_json(
        news_items,
        {
            "generated_at_utc": (now - timedelta(minutes=30)).isoformat(),
            "lookback_hours": 24,
            "sources": [{"name": "FXStreet News"}, {"name": "ForexLive"}],
            "items": items,
            "issues": [],
        },
    )
    digest.parent.mkdir(parents=True, exist_ok=True)
    digest.write_text(_digest(), encoding="utf-8")
    flow.write_text("### 2026-06-04 20:30 JST\n- HOT: USD jobs\n---ENTRY---\n", encoding="utf-8")
    _write_json(profile, {"generated_at_utc": (now - timedelta(minutes=5)).isoformat(), "pair_profiles": []})
    _write_json(calendar, {"generated_at_utc": now.isoformat(), "events": [], "issues": []})
    automation.write_text(
        'version = 1\nid = "qr-news-digest"\nstatus = "ACTIVE"\n'
        'rrule = "RRULE:FREQ=HOURLY;INTERVAL=1;BYMINUTE=16"\n'
        'cwds = ["/Users/tossaki/App/QuantRabbit-live"]\n',
        encoding="utf-8",
    )
    _write_json(weekend, {"mode": "restored"})
    for path in (news_items, digest, flow):
        _touch(path, now - timedelta(minutes=30))
    _touch(profile, now - timedelta(minutes=5))
    return {
        "news_items": news_items,
        "digest": digest,
        "flow": flow,
        "profile": profile,
        "calendar": calendar,
        "automation": automation,
        "weekend": weekend,
    }


class NewsHealthTest(unittest.TestCase):
    def test_fresh_active_news_packet_is_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = _write_good_packet(Path(tmp))

            payload = build_news_health(
                news_items_path=files["news_items"],
                digest_path=files["digest"],
                flow_log_path=files["flow"],
                market_story_profile_path=files["profile"],
                calendar_path=files["calendar"],
                automation_path=files["automation"],
                weekend_state_path=files["weekend"],
                now_utc=FRESH_NOW,
            )

        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["market_window"], "ACTIVE")

    def test_active_market_warns_on_raw_rss_digest_when_structured_news_is_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = _write_good_packet(Path(tmp))
            files["digest"].write_text(_digest(websearch=False), encoding="utf-8")
            _touch(files["digest"], FRESH_NOW - timedelta(minutes=30))

            payload = build_news_health(
                news_items_path=files["news_items"],
                digest_path=files["digest"],
                flow_log_path=files["flow"],
                market_story_profile_path=files["profile"],
                calendar_path=files["calendar"],
                automation_path=files["automation"],
                weekend_state_path=files["weekend"],
                now_utc=FRESH_NOW,
            )

        self.assertEqual(payload["status"], "WARN")
        self.assertTrue(any("news_digest_websearch" in issue for issue in payload["issues"]))

    def test_active_market_warns_but_does_not_block_when_rss_is_thin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = _write_good_packet(Path(tmp))
            _write_json(
                files["news_items"],
                {
                    "generated_at_utc": (FRESH_NOW - timedelta(minutes=30)).isoformat(),
                    "lookback_hours": 24,
                    "items": [],
                    "issues": ["STALE_NEWS_FEED: newest=2026-06-02T00:00:00+00:00 lookback_hours=24"],
                },
            )
            _touch(files["news_items"], FRESH_NOW - timedelta(minutes=30))

            payload = build_news_health(
                news_items_path=files["news_items"],
                digest_path=files["digest"],
                flow_log_path=files["flow"],
                market_story_profile_path=files["profile"],
                calendar_path=files["calendar"],
                automation_path=files["automation"],
                weekend_state_path=files["weekend"],
                now_utc=FRESH_NOW,
            )

        self.assertEqual(payload["status"], "WARN")
        self.assertTrue(any("news_items_count" in issue for issue in payload["issues"]))

    def test_active_market_blocks_market_story_profile_older_than_news(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = _write_good_packet(Path(tmp))
            _touch(files["profile"], FRESH_NOW - timedelta(hours=2))

            payload = build_news_health(
                news_items_path=files["news_items"],
                digest_path=files["digest"],
                flow_log_path=files["flow"],
                market_story_profile_path=files["profile"],
                calendar_path=files["calendar"],
                automation_path=files["automation"],
                weekend_state_path=files["weekend"],
                now_utc=FRESH_NOW,
            )

        self.assertEqual(payload["status"], "BLOCK")
        self.assertTrue(any("market_story_news_sync" in issue for issue in payload["issues"]))

    def test_active_market_blocks_stale_calendar_before_nowcast_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = _write_good_packet(Path(tmp))
            _write_json(
                files["calendar"],
                {
                    "generated_at_utc": (FRESH_NOW - timedelta(hours=12)).isoformat(),
                    "events": [],
                    "issues": [],
                },
            )

            payload = build_news_health(
                news_items_path=files["news_items"],
                digest_path=files["digest"],
                flow_log_path=files["flow"],
                market_story_profile_path=files["profile"],
                calendar_path=files["calendar"],
                automation_path=files["automation"],
                weekend_state_path=files["weekend"],
                now_utc=FRESH_NOW,
            )

        self.assertEqual(payload["status"], "BLOCK")
        self.assertTrue(any("calendar_context" in issue for issue in payload["issues"]))

    def test_weekend_paused_news_automation_is_ok(self) -> None:
        weekend_now = datetime(2026, 6, 6, 13, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            files = _write_good_packet(Path(tmp), now=weekend_now)
            files["automation"].write_text(
                'version = 1\nid = "qr-news-digest"\nstatus = "PAUSED"\n'
                'rrule = "RRULE:FREQ=HOURLY;INTERVAL=1;BYMINUTE=16"\n'
                'cwds = ["/Users/tossaki/App/QuantRabbit-live"]\n',
                encoding="utf-8",
            )
            _write_json(files["weekend"], {"mode": "paused"})

            payload = build_news_health(
                news_items_path=files["news_items"],
                digest_path=files["digest"],
                flow_log_path=files["flow"],
                market_story_profile_path=files["profile"],
                calendar_path=files["calendar"],
                automation_path=files["automation"],
                weekend_state_path=files["weekend"],
                now_utc=weekend_now,
            )

        self.assertEqual(payload["market_window"], "WEEKEND_PAUSED")
        self.assertEqual(payload["status"], "OK")

    def test_cli_strict_returns_nonzero_on_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _write_good_packet(root)
            files["digest"].write_text(
                _digest().replace(
                    "## Pre-Event Nowcast\n- NFP evidence stack: ADP, claims, JOLTS, ISM employment.\n\n",
                    "",
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "news-health",
                        "--news-items",
                        str(files["news_items"]),
                        "--digest",
                        str(files["digest"]),
                        "--flow-log",
                        str(files["flow"]),
                        "--market-story-profile",
                        str(files["profile"]),
                        "--calendar",
                        str(files["calendar"]),
                        "--automation",
                        str(files["automation"]),
                        "--weekend-state",
                        str(files["weekend"]),
                        "--output",
                        str(root / "health.json"),
                        "--report",
                        str(root / "health.md"),
                        "--now-utc",
                        FRESH_NOW.isoformat(),
                        "--strict",
                    ]
                )

        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "BLOCK")


if __name__ == "__main__":
    unittest.main()
