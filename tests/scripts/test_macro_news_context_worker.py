from __future__ import annotations

from datetime import datetime, timezone
from urllib.request import Request

from scripts import macro_news_context_worker


def test_fetch_bytes_uses_browser_like_headers(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"<rss/>"

    def _fake_urlopen(request: Request, timeout: float):
        captured["url"] = request.full_url
        captured["user_agent"] = request.get_header("User-agent") or ""
        captured["accept"] = request.get_header("Accept") or ""
        captured["referer"] = request.get_header("Referer") or ""
        captured["timeout"] = str(timeout)
        return _Resp()

    monkeypatch.setattr(macro_news_context_worker.urllib.request, "urlopen", _fake_urlopen)

    body = macro_news_context_worker._fetch_bytes(
        "https://www.federalreserve.gov/feeds/press_all.xml",
        timeout_sec=3.0,
        retries=0,
    )

    assert body == b"<rss/>"
    assert captured["url"] == "https://www.federalreserve.gov/feeds/press_all.xml"
    assert "QuantRabbit" in captured["user_agent"]
    assert "application/rss+xml" in captured["accept"]
    assert captured["referer"] == "https://www.federalreserve.gov/newsevents/pressreleases.htm"


def test_build_report_reads_official_rss_and_scores_context(monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    pub_date = now.strftime("%a, %d %b %Y %H:%M:%S GMT")
    rss = f"""
    <rss version="2.0">
      <channel>
        <title>Feed</title>
        <item>
          <title>FOMC statement on inflation outlook</title>
          <link>https://example.test/fomc</link>
          <pubDate>{pub_date}</pubDate>
        </item>
      </channel>
    </rss>
    """.encode("utf-8")

    monkeypatch.setattr(
        macro_news_context_worker,
        "DEFAULT_FEEDS",
        (
            ("fed_press_all", "https://federalreserve.example/feed.xml", "fed"),
            ("boj_whatsnew", "https://boj.example/feed.xml", "boj"),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        macro_news_context_worker,
        "_fetch_bytes",
        lambda _url, timeout_sec: rss,
        raising=False,
    )

    payload = macro_news_context_worker.build_report(
        lookback_hours=24,
        timeout_sec=3.0,
        per_feed_limit=2,
    )

    assert payload["feed_count"] == 2
    assert payload["source_error_count"] == 0
    assert payload["event_severity"] == "high"
    assert payload["caution_window_active"] is True
    assert payload["usd_jpy_bias"] == "up"
    assert payload["headlines"][0]["title"] == "FOMC statement on inflation outlook"
