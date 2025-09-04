"""
news_pipeline_local.py
~~~~~~~~~~~~~~~~~~~~~~
A pipeline that automatically fetches, summarizes, and manages news locally.
"""

import asyncio
from datetime import datetime, timedelta
import json
import os
import pathlib
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from market_data.news_fetcher_local import fetch_news_local
from analysis.llm_client import summarize

# --- Directory Settings ---
NEWS_DIR = pathlib.Path("news_articles")
SUMMARIES_DIR = pathlib.Path("summaries")
NEWS_DIR.mkdir(exist_ok=True)
SUMMARIES_DIR.mkdir(exist_ok=True)

# --- Pipeline Functions ---

async def summarize_and_archive():
    """Summarizes articles in news_articles, saves them to summaries, and deletes originals."""
    print(f"[{datetime.now()}] Checking for new articles to summarize...")
    for article_file in NEWS_DIR.glob("*.json"):
        try:
            with open(article_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            title = data.get("title", "")
            body = data.get("body", "")
            text_to_summarize = f"Title: {title}\n\n{body}"

            print(f"  Summarizing: {article_file.name}")
            summary = summarize(text_to_summarize)

            summary_data = {
                "uid": data.get("uid"),
                "original_title": title,
                "summary": summary,
                "summarized_at": datetime.now().isoformat(),
            }

            summary_filename = f"summary_{article_file.name}"
            summary_filepath = SUMMARIES_DIR / summary_filename
            with open(summary_filepath, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=4)
            
            print(f"  Saved summary to: {summary_filepath}")
            
            # Delete original article
            os.remove(article_file)
            print(f"  Removed original article: {article_file.name}")

        except Exception as e:
            print(f"Error processing {article_file.name}: {e}")

async def cleanup_old_summaries():
    """Deletes summary files older than 2 hours."""
    print(f"[{datetime.now()}] Cleaning up old summaries...")
    two_hours_ago = datetime.now() - timedelta(hours=2)

    for summary_file in SUMMARIES_DIR.glob("*.json"):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            summarized_at_str = data.get("summarized_at")
            if summarized_at_str:
                summarized_at = datetime.fromisoformat(summarized_at_str)
                if summarized_at < two_hours_ago:
                    os.remove(summary_file)
                    print(f"  Removed old summary: {summary_file.name}")

        except Exception as e:
            print(f"Error cleaning up {summary_file.name}: {e}")


async def scheduled_news_task():
    """Task to fetch and summarize news in sequence."""
    print(f"--- Running hourly news task at {datetime.now()} ---")
    await fetch_news_local()
    await summarize_and_archive()
    print("--- Finished hourly news task ---")


# --- Main Execution ---

if __name__ == "__main__":
    scheduler = AsyncIOScheduler(timezone="UTC")

    # Run the news task every hour
    scheduler.add_job(scheduled_news_task, "interval", hours=1, id="news_task")
    
    # Run the cleanup task every hour
    scheduler.add_job(cleanup_old_summaries, "interval", hours=1, id="cleanup_task")

    print("Starting news pipeline scheduler...")
    print("Press Ctrl+C to exit.")

    scheduler.start()

    try:
        # Keep the event loop running forever
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")
        scheduler.shutdown()
