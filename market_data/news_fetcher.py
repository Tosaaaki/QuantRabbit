"""
market_data.news_fetcher
~~~~~~~~~~~~~~~~~~~~~~~~
60 秒おきに RSS を取得し、高インパクトのイベントを
Cloud Storage `fx-news/raw/*.json` へアップロードする。
"""

from __future__ import annotations
import asyncio, datetime, json, tomllib, uuid, feedparser, re
from google.cloud import storage

CFG = tomllib.loads(open("config/env.toml", "rb").read())
BUCKET = CFG["gcp"]["bucket_news"]
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)

FF_URL  = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
FF_NEXT = "https://nfs.faireconomy.media/ff_calendar_nextweek.xml"
DAILYFX = "https://www.dailyfx.com/feeds/most-recent-stories"

HIGH_WORDS = re.compile("|".join(["BoJ", "FOMC", "Non-Farm", "CPI", "政策金利", "雇用統計"]))


def _impact(item) -> str:
    if "impact" in item:
        return item["impact"].strip().lower()  # forex factory
    title = item.get("title", "")
    return "high" if HIGH_WORDS.search(title) else "low"


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
            impact = _impact(entry)
            if impact not in ("high", "medium"):
                continue
            uid = entry.get("id") or str(uuid.uuid4())
            data = {
                "uid": uid,
                "src": parsed.feed.title,
                "impact": impact,
                "title": entry.title,
                "currency": entry.get("currency", ""),
                "time_utc": entry.get("date", ""),
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