"""
market_data.news_fetcher
~~~~~~~~~~~~~~~~~~~~~~~~
USD/JPY 関連のニュース/指標フィードを 60 秒おきに巡回し、高インパクト情報を
Cloud Storage `fx-news/raw/*.json` へアップロードする。

2025-10: OANDA Labs API が 403 で利用できないため、FairEconomy / DailyFX /
Google News (USDJPY, BoJ キーワード) などの公開 RSS を利用する。
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import os
import pathlib
import re
import uuid
from typing import Iterable

import feedparser
import toml
from google.cloud import storage

# --- 設定・定数 -------------------------------------------------------------

FETCH_LIMIT = int(os.environ.get("NEWS_FETCH_MAX", "20"))
FETCH_INTERVAL_SECONDS = int(os.environ.get("NEWS_FETCH_INTERVAL_SEC", "60"))
MIN_IMPACT = int(os.environ.get("NEWS_MIN_IMPACT", "2"))  # 2=medium, 3=high

# env.toml 優先で GCS バケットを決定
try:
    cfg = None
    cfg_path = pathlib.Path("config/env.local.toml")
    if cfg_path.exists():
        cfg = toml.load(cfg_path.open("r"))
    else:
        toml_path = pathlib.Path("config/env.toml")
        if toml_path.exists():
            cfg = toml.load(toml_path.open("r"))
except Exception:
    cfg = None

if cfg:
    BUCKET = cfg.get("gcp", {}).get("bucket_news")
else:
    BUCKET = None

if not BUCKET:
    BUCKET = os.environ.get("BUCKET") or os.environ.get("BUCKET_NEWS") or "fx-news"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)

DEFAULT_FEEDS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.xml",
    "https://www.dailyfx.com/feeds/most-recent-stories",
    "https://news.google.com/rss/search?q=USDJPY&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=BoJ+OR+%22Bank+of+Japan%22&hl=en-US&gl=US&ceid=US:en",
]

extra_feeds = os.environ.get("NEWS_EXTRA_FEEDS")
if extra_feeds:
    DEFAULT_FEEDS.extend([url.strip() for url in extra_feeds.split(",") if url.strip()])

HIGH_WORDS = re.compile("|".join(["BoJ", "FOMC", "Non-Farm", "CPI", "政策金利", "雇用統計"]), re.IGNORECASE)

# --- ユーティリティ ---------------------------------------------------------

def _impact(entry) -> int:
    impact = entry.get("impact") or entry.get("importance")
    if isinstance(impact, str):
        impact = impact.lower()
        if impact == "high":
            return 3
        if impact in ("medium", "med"):
            return 2
        return 1
    title = entry.get("title", "")
    summary = entry.get("summary", "")
    if HIGH_WORDS.search(title) or HIGH_WORDS.search(summary):
        return 3
    return 2

def _event_time(entry) -> str | None:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime.datetime(*entry.published_parsed[:6]).isoformat()
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        return datetime.datetime(*entry.updated_parsed[:6]).isoformat()
    if entry.get("published"):
        return entry["published"]
    if entry.get("updated"):
        return entry["updated"]
    return None

def _currency(entry) -> str:
    currency = entry.get("currency")
    if currency:
        return currency
    text = f"{entry.get('title','')} {entry.get('summary','')}".upper()
    if "USD" in text and "JPY" in text:
        return "USD_JPY"
    if "JPY" in text:
        return "JPY"
    if "USD" in text:
        return "USD"
    return ""

def _uid(entry) -> str:
    for key in ("id", "guid", "link"):
        val = entry.get(key)
        if val:
            return str(val)
    joined = f"{entry.get('title','')}{entry.get('summary','')}{entry.get('published','')}"
    return hashlib.sha256(joined.encode("utf-8", errors="ignore")).hexdigest()

def _push(blob_name: str, data: dict):
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data), content_type="application/json")

# --- メイン処理 -------------------------------------------------------------

async def fetch_loop():
    logging.info("[NEWS] fetch loop started interval=%ss limit=%s", FETCH_INTERVAL_SECONDS, FETCH_LIMIT)
    while True:
        await _rss_once()
        await asyncio.sleep(FETCH_INTERVAL_SECONDS)

async def _rss_once():
    fetched_count = 0
    for url in DEFAULT_FEEDS:
        if fetched_count >= FETCH_LIMIT:
            break
        try:
            parsed = feedparser.parse(url)
        except Exception as exc:
            logging.exception("[NEWS] failed to parse feed url=%s err=%s", url, exc)
            continue
        feed_title = getattr(parsed.feed, "title", url)
        for entry in parsed.entries:
            if fetched_count >= FETCH_LIMIT:
                break

            uid = _uid(entry) or str(uuid.uuid4())
            blob_name = f"raw/{uid}.json"
            blob = bucket.blob(blob_name)
            if blob.exists():
                continue

            impact_val = _impact(entry)
            if impact_val < MIN_IMPACT:
                continue

            event_time_val = _event_time(entry)
            data = {
                "uid": uid,
                "src": feed_title,
                "impact": impact_val,
                "event_time": event_time_val,
                "title": entry.get("title", ""),
                "currency": _currency(entry),
                "time_utc": entry.get("published", ""),
                "body": entry.get("summary", ""),
            }
            _push(blob_name, data)
            fetched_count += 1

    logging.info("[NEWS] cycle completed fetched=%s", fetched_count)

def publish_from_iter(items: Iterable[dict]) -> int:
    count = 0
    for entry in items:
        uid = entry.get("uid") or str(uuid.uuid4())
        blob_name = f"raw/{uid}.json"
        blob = bucket.blob(blob_name)
        if blob.exists():
            continue
        _push(blob_name, entry)
        count += 1
    return count

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("Start RSS fetcher…  Ctrl‑C to stop")
    try:
        asyncio.run(_rss_once())
    except KeyboardInterrupt:
        pass
