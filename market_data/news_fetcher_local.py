"""
market_data.news_fetcher_local
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fetches RSS feeds and saves high-impact events to local `news_articles/*.json` files.
"""

from __future__ import annotations
import asyncio
import datetime
import json
import uuid
import feedparser
import re
import os
import pathlib
import hashlib

# --- Settings and Constants ---
SAVE_DIR = pathlib.Path("news_articles")
SAVE_DIR.mkdir(exist_ok=True)

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
FF_NEXT = "https://nfs.faireconomy.media/ff_calendar_nextweek.xml"
DAILYFX = "https://www.dailyfx.com/feeds/most-recent-stories"

NEWS_FEEDS = [
    FF_URL,
    FF_NEXT,
    DAILYFX,
]

HIGH_WORDS = re.compile(
    "|".join(["BoJ", "FOMC", "Non-Farm", "CPI", "Policy Rate", "Employment Statistics"])
)

# --- Helper Functions ---

def _impact(item) -> int:
    impact_str = "low"
    if "impact" in item:
        impact_str = item["impact"].strip().lower()
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
    if hasattr(item, "published_parsed"):
        return datetime.datetime(*item.published_parsed[:6]).isoformat()
    if hasattr(item, "updated_parsed"):
        return datetime.datetime(*item.published_parsed[:6]).isoformat()
    return None

def _save_local(file_path: pathlib.Path, data: dict):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- Main Processing ---

async def fetch_news_local():
    """Fetches news from RSS feeds and saves them locally."""
    fetched_count = 0
    for url in NEWS_FEEDS:
        if fetched_count >= 10: # Limit to 10 articles for demonstration
            break

        parsed = feedparser.parse(url)
        
        for entry in parsed.entries:
            if fetched_count >= 10:
                break

            original_uid = entry.get("id") or str(uuid.uuid4())
            # Use a hash of the original UID for the filename
            uid_hash = hashlib.sha256(original_uid.encode('utf-8')).hexdigest()
            
            # Check if a file with the same UID already exists locally
            file_name = f"{uid_hash}.json"
            file_path = SAVE_DIR / file_name
            if file_path.exists(): # Re-enabled duplicate check
                continue

            impact_val = _impact(entry)
            if impact_val < 2:  # Filtering re-enabled
                continue

            event_time_val = _event_time(entry)
            data = {
                "uid": original_uid, # Store original UID in the JSON
                "src": parsed.feed.title,
                "impact": impact_val,
                "event_time": event_time_val,
                "title": entry.title,
                "currency": entry.get("currency", ""),
                "time_utc": entry.get("date", ""),
                "body": entry.get("summary", ""),
            }
            _save_local(file_path, data)
            fetched_count += 1
    print(f"Fetched {fetched_count} new articles.")

if __name__ == "__main__":
    print("Start local RSS fetcher…")
    asyncio.run(fetch_news_local())
    print("Finished local RSS fetcher.")