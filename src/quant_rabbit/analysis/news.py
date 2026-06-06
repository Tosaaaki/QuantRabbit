"""News feed ingestion for the trader's market-story layer.

This module fetches public RSS/API-style feeds and writes compact, sourced
artifacts for the discretionary trader. It intentionally stores headlines,
short summaries, timestamps, categories, and links only; full article bodies
belong in the source site, not in QuantRabbit artifacts.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, G8_CURRENCIES


MARKETPULSE_RSS_URL = "https://www.marketpulse.com/feed/"
MARKETPULSE_SOURCE = "MarketPulse"
FXSTREET_NEWS_RSS_URL = "https://www.fxstreet.com/rss/news"
FXSTREET_NEWS_SOURCE = "FXStreet News"
FOREXLIVE_RSS_URL = "https://www.forexlive.com/feed"
FOREXLIVE_SOURCE = "ForexLive"

# (a) Freshness window for macro/news headlines used by the intraday trader.
# (b) Constant because MarketPulse declares an hourly RSS cadence and the trader
#     needs same-session narrative pressure, not multi-day archive memory.
# (c) Replace with a source-specific TTL table if multiple licensed feeds are
#     registered with different update cadences.
DEFAULT_LOOKBACK_HOURS = 24

# (a) Upper bound on RSS items retained in the JSON snapshot.
# (b) Constant because RSS feeds can publish large archives; the trader consumes
#     the current session story, and retaining more items only increases stale
#     narrative risk.
# (c) Replace with a feed-specific retention policy if a vendor exposes reliable
#     pagination plus explicit article priority metadata.
DEFAULT_MAX_ITEMS = 80

# (a) Number of top items written into the human-readable digest sections.
# (b) Constant because the trader cycle must scan the digest quickly while still
#     seeing enough cross-asset breadth for G8 FX decisions.
# (c) Replace with an operator-visible CLI option per source if feed volume grows.
DEFAULT_DIGEST_ITEMS = 12

# (a) Number of entries retained in logs/news_flow_log.md.
# (b) Constant because one hourly news task needs about two days of flow context
#     without letting old headlines dominate the market-story miner.
# (c) Replace with a market-session-aware retention policy if news cadence changes.
DEFAULT_FLOW_ENTRIES = 48

# (a) Short summary character budget stored from RSS descriptions.
# (b) Constant because descriptions are copyrighted source text; the trader needs
#     only a compact factual cue plus source link.
# (c) Replace with a provider metadata field if a licensed feed supplies explicit
#     summary snippets.
SUMMARY_CHAR_LIMIT = 360

# (a) Network timeout for one public feed request.
# (b) Constant because the news task is auxiliary and must not block trader
#     evidence refresh for long network stalls.
# (c) Replace with per-source timeout config if a licensed feed SLA requires it.
FETCH_TIMEOUT_SECONDS = 15

ENTRY_SEPARATOR = "---ENTRY---"


@dataclass(frozen=True)
class NewsSource:
    name: str
    url: str


DEFAULT_NEWS_SOURCES: tuple[NewsSource, ...] = (
    NewsSource(name=MARKETPULSE_SOURCE, url=MARKETPULSE_RSS_URL),
    NewsSource(name=FXSTREET_NEWS_SOURCE, url=FXSTREET_NEWS_RSS_URL),
    NewsSource(name=FOREXLIVE_SOURCE, url=FOREXLIVE_RSS_URL),
)


@dataclass(frozen=True)
class NewsItem:
    source: str
    title: str
    link: str
    published_at_utc: str
    summary: str
    categories: tuple[str, ...] = ()
    currencies: tuple[str, ...] = ()
    pairs: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "title": self.title,
            "link": self.link,
            "published_at_utc": self.published_at_utc,
            "summary": self.summary,
            "categories": list(self.categories),
            "currencies": list(self.currencies),
            "pairs": list(self.pairs),
            "topics": list(self.topics),
        }


@dataclass(frozen=True)
class NewsSnapshot:
    generated_at_utc: str
    lookback_hours: int
    sources: tuple[NewsSource, ...]
    items: tuple[NewsItem, ...]
    issues: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "lookback_hours": self.lookback_hours,
            "sources": [{"name": source.name, "url": source.url} for source in self.sources],
            "items": [item.to_dict() for item in self.items],
            "issues": list(self.issues),
        }


def fetch_rss(url: str, *, timeout: int = FETCH_TIMEOUT_SECONDS) -> bytes:
    """Fetch raw RSS bytes with a stable user agent."""

    req = Request(url, headers={"User-Agent": "QuantRabbit/1.0 news-snapshot"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_marketpulse_rss(payload: bytes) -> tuple[NewsItem, ...]:
    """Parse MarketPulse RSS into compact news items."""

    return parse_rss_feed(payload, source_name=MARKETPULSE_SOURCE)


def parse_rss_feed(payload: bytes, *, source_name: str) -> tuple[NewsItem, ...]:
    """Parse a standard RSS feed into compact news items."""

    root = ET.fromstring(payload)
    items: list[NewsItem] = []
    for node in root.findall(".//item"):
        title = _clean_text(node.findtext("title"))
        link = _clean_text(node.findtext("link"))
        raw_description = node.findtext("description") or ""
        published_at = _parse_pub_date(node.findtext("pubDate") or "")
        if not title or not link or published_at is None:
            continue
        categories = tuple(
            sorted(
                {
                    _clean_text(category.text)
                    for category in node.findall("category")
                    if _clean_text(category.text)
                }
            )
        )
        summary = _clip(_strip_html(raw_description), SUMMARY_CHAR_LIMIT)
        currencies, pairs = _extract_markets(title, summary, categories)
        topics = _extract_topics(title, summary, categories)
        items.append(
            NewsItem(
                source=source_name,
                title=title,
                link=link,
                published_at_utc=published_at.isoformat(),
                summary=summary,
                categories=categories,
                currencies=currencies,
                pairs=pairs,
                topics=topics,
            )
        )
    return tuple(sorted(_dedupe_items(items), key=_sort_key, reverse=True))


def build_news_snapshot(
    *,
    now_utc: datetime | None = None,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    max_items: int = DEFAULT_MAX_ITEMS,
    fetch: bool = True,
    marketpulse_payload: bytes | None = None,
    source_payloads: Mapping[str, bytes] | None = None,
    sources: Sequence[NewsSource] = DEFAULT_NEWS_SOURCES,
) -> NewsSnapshot:
    """Fetch and normalize configured news sources."""

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    configured_sources = tuple(sources)
    issues: list[str] = []
    items: list[NewsItem] = []

    payloads: dict[str, bytes] = dict(source_payloads or {})
    if marketpulse_payload is not None:
        payloads[MARKETPULSE_SOURCE] = marketpulse_payload

    source_items: dict[str, tuple[NewsItem, ...]] = {}
    for source in configured_sources:
        issue_slug = _source_issue_slug(source.name)
        try:
            payload = payloads.get(source.name) or payloads.get(source.url)
            if payload is None:
                if not fetch:
                    continue
                payload = fetch_rss(source.url)
            parsed = parse_rss_feed(payload, source_name=source.name)
            source_items[source.name] = parsed
            items.extend(parsed)
        except ET.ParseError as exc:
            issues.append(f"MALFORMED_{issue_slug}_FEED: {exc}")
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            issues.append(f"MISSING_{issue_slug}_FEED: {exc}")

    items = list(sorted(_dedupe_items(items), key=_sort_key, reverse=True))
    cutoff = now - timedelta(hours=lookback_hours)
    for source_name, parsed_items in source_items.items():
        if parsed_items and not any(_published_at(item) >= cutoff for item in parsed_items):
            newest = max(_published_at(item) for item in parsed_items)
            issues.append(
                f"STALE_{_source_issue_slug(source_name)}_FEED: "
                f"newest={newest.isoformat()} lookback_hours={lookback_hours}"
            )
    fresh_items = [item for item in items if _published_at(item) >= cutoff]
    if items and not fresh_items:
        newest = max(_published_at(item) for item in items)
        issues.append(f"STALE_NEWS_FEED: newest={newest.isoformat()} lookback_hours={lookback_hours}")
    if not items and not issues:
        issues.append("MISSING_FRESH_NEWS_FEED: no configured source returned items")

    retained = tuple((fresh_items or items)[:max_items])
    return NewsSnapshot(
        generated_at_utc=now.isoformat(),
        lookback_hours=lookback_hours,
        sources=configured_sources,
        items=retained,
        issues=tuple(issues),
    )


def write_news_artifacts(
    snapshot: NewsSnapshot,
    *,
    output_path: Path,
    digest_path: Path,
    flow_log_path: Path,
    digest_items: int = DEFAULT_DIGEST_ITEMS,
    flow_entries: int = DEFAULT_FLOW_ENTRIES,
) -> None:
    """Write JSON, digest markdown, and append a compact flow entry."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(render_news_digest(snapshot, digest_items=digest_items), encoding="utf-8")

    flow_log_path.parent.mkdir(parents=True, exist_ok=True)
    entries = _read_flow_entries(flow_log_path)
    entries.append(render_flow_entry(snapshot))
    flow_log_path.write_text(_render_flow_log(entries[-flow_entries:]), encoding="utf-8")


def render_news_digest(snapshot: NewsSnapshot, *, digest_items: int = DEFAULT_DIGEST_ITEMS) -> str:
    ranked = _rank_items(snapshot.items)
    high_impact = _topic_items(
        ranked,
        {"central_bank", "inflation", "employment", "intervention", "geopolitics", "risk_off"},
    )[:4]
    watch = [item for item in ranked if item not in high_impact][:4]
    pair_items = [item for item in ranked if item.pairs][:6]
    risk_events = _topic_items(ranked, {"central_bank", "inflation", "employment", "geopolitics"})[:4]

    lines = [
        f"# FX News Digest — {_jst_stamp(snapshot.generated_at_utc)}",
        "",
        "## 🔴 High Impact (act on this)",
    ]
    if high_impact:
        lines.extend(f"- {_trader_bullet(item)}" for item in high_impact)
    else:
        lines.append("- Low fresh high-impact flow from configured feeds; avoid inventing a macro thesis.")

    lines.extend(["", "## 🟡 Watch List"])
    if watch:
        lines.extend(f"- {_trader_bullet(item)}" for item in watch)
    else:
        lines.append("- No secondary fresh headlines detected.")

    lines.extend(["", "## 📅 Economic Calendar Today (JST)"])
    calendar_like = [
        item for item in ranked
        if any(topic in item.topics for topic in ("inflation", "employment", "growth", "central_bank"))
    ][:4]
    if calendar_like:
        lines.extend(f"- {_calendar_bullet(item)}" for item in calendar_like)
    else:
        lines.append("- No calendar-specific item in RSS cache; use `data/economic_calendar.json` for exact event windows.")

    lines.extend(["", "## 🏦 Central Bank Tracker"])
    central_bank_items = _topic_items(ranked, {"central_bank", "intervention"})[:4]
    if central_bank_items:
        lines.extend(f"- {_trader_bullet(item)}" for item in central_bank_items)
    else:
        lines.append("- Fed/BOJ/ECB/RBA: no new central-bank headline in configured feeds.")

    lines.extend(["", "## 💱 Pair-Specific Notes"])
    if pair_items:
        lines.extend(f"- {_pair_bullet(item)}" for item in pair_items)
    else:
        lines.append("- USD_JPY / EUR_USD / GBP_USD / AUD_USD: no pair-specific fresh headline detected.")

    lines.extend(["", "## ⚠️ Risk Events (48h)"])
    if risk_events:
        lines.extend(f"- {_trader_bullet(item)}" for item in risk_events)
    else:
        lines.append("- No scheduled-risk headline detected in news feed; confirm with economic-calendar snapshot.")

    if snapshot.issues:
        lines.extend(["", "## Feed Issues"])
        lines.extend(f"- {issue}" for issue in snapshot.issues)

    source_names = " + ".join(source.name for source in snapshot.sources)
    lines.extend(
        [
            "",
            "---",
            f"Updated: {_jst_stamp(snapshot.generated_at_utc)} | Sources: {source_names} + news_fetcher.py",
        ]
    )
    return "\n".join(lines) + "\n"


def render_flow_entry(snapshot: NewsSnapshot) -> str:
    now = _jst_stamp(snapshot.generated_at_utc)
    ranked = _rank_items(snapshot.items)
    hot = _flow_text(ranked[0]) if ranked else "MISSING_FRESH_NEWS_FEED"
    theme = _theme_text(ranked[:2]) if ranked else "N/A"
    watch_item = next((item for item in ranked if item.pairs), ranked[0] if ranked else None)
    watch = _flow_text(watch_item) if watch_item else "N/A"
    return f"### {now}\n- HOT: {hot}\n- THEME: {theme}\n- WATCH: {watch}"


def _read_flow_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [part.strip() for part in path.read_text(encoding="utf-8").split(ENTRY_SEPARATOR) if part.strip()]


def _render_flow_log(entries: Sequence[str]) -> str:
    if not entries:
        return ""
    return f"\n{ENTRY_SEPARATOR}\n".join(entries) + f"\n{ENTRY_SEPARATOR}\n"


def _rank_items(items: Iterable[NewsItem]) -> list[NewsItem]:
    return sorted(items, key=_sort_key, reverse=True)


def _topic_items(items: Sequence[NewsItem], topics: set[str]) -> list[NewsItem]:
    return [item for item in items if any(topic in topics for topic in item.topics)]


def _sort_key(item: NewsItem) -> tuple[int, str]:
    market_refs = len(item.pairs) + len(item.currencies) + len(item.topics)
    return market_refs, item.published_at_utc


def _dedupe_items(items: Iterable[NewsItem]) -> tuple[NewsItem, ...]:
    seen: set[str] = set()
    out: list[NewsItem] = []
    for item in items:
        key = _dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return tuple(out)


def _dedupe_key(item: NewsItem) -> str:
    return (item.link or item.title).strip().rstrip("/").lower()


def _source_issue_slug(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def _published_at(item: NewsItem) -> datetime:
    return datetime.fromisoformat(item.published_at_utc)


def _parse_pub_date(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _extract_markets(title: str, summary: str, categories: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    text = " ".join([title, summary, *categories]).upper()
    pairs: set[str] = set()
    for category in categories:
        upper = category.upper()
        if upper.startswith("FX_"):
            pair = _compact_pair_to_symbol(upper.removeprefix("FX_"))
            if pair:
                pairs.add(pair)
    for match in re.finditer(r"\b([A-Z]{3})[\/_]([A-Z]{3})\b", text):
        pair = f"{match.group(1)}_{match.group(2)}"
        if pair in DEFAULT_TRADER_PAIRS:
            pairs.add(pair)
    for pair in DEFAULT_TRADER_PAIRS:
        if pair.replace("_", "") in text or pair.replace("_", "/") in text:
            pairs.add(pair)

    currencies = {cur for cur in G8_CURRENCIES if re.search(rf"\b{cur}\b", text)}
    alias_map = {
        "DOLLAR": "USD",
        "GREENBACK": "USD",
        "EURO": "EUR",
        "STERLING": "GBP",
        "POUND": "GBP",
        "YEN": "JPY",
        "AUSSIE": "AUD",
        "KIWI": "NZD",
        "LOONIE": "CAD",
        "FRANC": "CHF",
    }
    for alias, currency in alias_map.items():
        if alias in text:
            currencies.add(currency)
    for pair in pairs:
        base, _, quote = pair.partition("_")
        currencies.update((base, quote))
    return tuple(sorted(currencies)), tuple(sorted(pairs))


def _compact_pair_to_symbol(value: str) -> str | None:
    compact = re.sub(r"[^A-Z]", "", value)
    if len(compact) != 6:
        return None
    pair = f"{compact[:3]}_{compact[3:]}"
    return pair if pair in DEFAULT_TRADER_PAIRS else None


def _extract_topics(title: str, summary: str, categories: Sequence[str]) -> tuple[str, ...]:
    text = " ".join([title, summary, *categories]).upper()
    topic_needles: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("central_bank", ("BOJ", "FOMC", "FED", "ECB", "BOE", "RBA", "RATE DECISION")),
        ("inflation", ("CPI", "PCE", "INFLATION")),
        (
            "employment",
            (
                "NFP",
                "NON-FARM",
                "NONFARM",
                "ADP",
                "JOBS",
                "EMPLOYMENT",
                "JOBLESS",
                "CLAIMS",
                "JOLTS",
                "JOB OPENINGS",
                "HIRING",
                "LAYOFF",
                "LAYOFFS",
                "JOB CUTS",
                "WAGES",
            ),
        ),
        ("growth", ("GDP", "PMI", "ISM", "RETAIL SALES")),
        ("intervention", ("INTERVENTION", "RATE CHECK", "MOF")),
        ("geopolitics", ("WAR", "IRAN", "GEOPOLITICAL", "TARIFF")),
        ("risk_on", ("RISK-ON", "RISK ON", "EQUITIES RALLY", "RECORD HIGH")),
        ("risk_off", ("RISK-OFF", "RISK OFF", "SAFE-HAVEN", "SAFE HAVEN")),
        ("oil", ("OIL", "WTI", "BRENT")),
        ("yields", ("YIELD", "TREASURY", "BOND")),
    )
    topics = [topic for topic, needles in topic_needles if any(needle in text for needle in needles)]
    return tuple(topics)


def _item_label(item: NewsItem) -> str:
    if item.pairs:
        return ", ".join(item.pairs)
    if item.currencies:
        return ", ".join(item.currencies)
    if item.topics:
        return ", ".join(item.topics)
    return item.source


def _theme_text(items: Sequence[NewsItem]) -> str:
    labels: list[str] = []
    for item in items:
        label = _item_label(item)
        if label not in labels:
            labels.append(label)
    return " / ".join(labels) if labels else "N/A"


def _flow_text(item: NewsItem | None) -> str:
    if item is None:
        return "N/A"
    label = _item_label(item)
    text = f"{label}: {item.title}"
    return _clip(text, SUMMARY_CHAR_LIMIT)


def _trader_bullet(item: NewsItem) -> str:
    label = _item_label(item)
    topic_text = ", ".join(item.topics) if item.topics else "macro"
    return (
        f"**{label}** — {item.title}. Trade read: {topic_text}; "
        f"do not fade this without chart confirmation. Source: {item.source}; {item.link}"
    )


def _calendar_bullet(item: NewsItem) -> str:
    label = ", ".join(item.currencies) if item.currencies else _item_label(item)
    return f"**{label}** — {item.title}. Consensus/prior: check source and `data/economic_calendar.json`. {item.link}"


def _pair_bullet(item: NewsItem) -> str:
    pairs = ", ".join(item.pairs) if item.pairs else _item_label(item)
    return f"**{pairs}** — {item.title}. Thesis driver: {', '.join(item.topics) if item.topics else 'price-action headline'}. {item.link}"


def _jst_stamp(value: str) -> str:
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST")


def _strip_html(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return _clean_text(html.unescape(without_tags))


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clip(value: str, limit: int) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


__all__ = [
    "DEFAULT_DIGEST_ITEMS",
    "DEFAULT_FLOW_ENTRIES",
    "DEFAULT_LOOKBACK_HOURS",
    "DEFAULT_MAX_ITEMS",
    "MARKETPULSE_RSS_URL",
    "FXSTREET_NEWS_RSS_URL",
    "FOREXLIVE_RSS_URL",
    "DEFAULT_NEWS_SOURCES",
    "NewsItem",
    "NewsSnapshot",
    "NewsSource",
    "build_news_snapshot",
    "fetch_rss",
    "parse_marketpulse_rss",
    "parse_rss_feed",
    "render_flow_entry",
    "render_news_digest",
    "write_news_artifacts",
]
