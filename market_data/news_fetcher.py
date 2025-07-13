"""
market_data.news_fetcher
~~~~~~~~~~~~~~~~~~~~~~~~
60 秒おきに RSS を取得し、高インパクトのイベントを
Cloud Storage `fx-news/raw/*.json` へアップロードする。
"""

from __future__ import annotations
import asyncio
import datetime
import json
import toml
import uuid
import feedparser
import re
from google.cloud import storage

CFG = toml.load(open("config/env.local.toml", "r"))
BUCKET = CFG["gcp"]["bucket_news"]
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
FF_NEXT = "https://nfs.faireconomy.media/ff_calendar_nextweek.xml"
DAILYFX = "https://www.dailyfx.com/feeds/most-recent-stories"

HIGH_WORDS = re.compile(
    "|".join(["BoJ", "FOMC", "Non-Farm", "CPI", "政策金利", "雇用統計"])
)


def _impact(item) -> int:
    impact_str = "low"
    if "impact" in item:
        impact_str = item["impact"].strip().lower()  # forex factory
    elif "title" in item:
        title = item["title"]
        if "high impact" in title.lower():
            impact_str = "high"
        elif "medium impact" in title.lower():
            impact_str = "medium"

    if impact_str == "high":
        return 3
    if impact_str == "medium":
        return 2
    return 1


def _event_time(item) -> str | None:
    # Forex Factory の RSS は 'published' に時刻情報があることが多い
    if hasattr(item, "published_parsed"):
        return datetime.datetime(*item.published_parsed[:6]).isoformat()
    # DailyFX など、他のフィードの 'updated' なども考慮
    if hasattr(item, "updated_parsed"):
        return datetime.datetime(*item.updated_parsed[:6]).isoformat()
    return None


def _push(blob_name: str, data: dict):
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data), content_type="application/json")


async def fetch_loop():
    while True:
        await _rss_once()
        await asyncio.sleep(60)


async def _rss_once():
    now = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    for url in (FF_URL, FF_NEXT, DAILYFX):
        parsed = feedparser.parse(url)
        for entry in parsed.entries:
            impact_val = _impact(entry)
            event_time_val = _event_time(entry)

            # 高インパクトのイベントのみを対象
            if impact_val < 2:  # medium (2) or high (3) のみ
                continue

            uid = entry.get("id") or str(uuid.uuid4())
            data = {
                "uid": uid,
                "src": parsed.feed.title,
                "impact": impact_val,
                "event_time": event_time_val,
                "title": entry.title,
                "currency": entry.get("currency", ""),
                "time_utc": entry.get("date", ""),  # これは元のまま残しておく
                "body": entry.get("summary", ""),
            }
            blob_name = f"raw/{now}_{uid}.json"
            _push(blob_name, data)


if __name__ == "__main__":
    print("Start RSS fetcher…  Ctrl‑C to stop")
    try:
        asyncio.run(fetch_loop())
    except KeyboardInterrupt:
        pass
