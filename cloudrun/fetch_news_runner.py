import logging

logging.basicConfig(level=logging.INFO)

import os
import json
import feedparser
from google.cloud import storage
from flask import Flask

app = Flask(__name__)

BUCKET = os.environ.get("BUCKET")
RSS_URL = "https://news.yahoo.co.jp/rss/categories/business.xml"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)

def fetch_and_store_news():
    feed = feedparser.parse(RSS_URL)
    logging.info(f"Found {len(feed.entries)} entries in the feed.")
    for entry in feed.entries:
        # Use a hash of the link as a unique ID
        uid = hash(entry.link)
        blob = bucket.blob(f"raw/{uid}.json")
        data = {
            "uid": uid,
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "summary": entry.summary,
        }
        blob.upload_from_string(
            json.dumps(data, indent=2, ensure_ascii=False),
            content_type="application/json",
        )

@app.route("/", methods=["GET"])
def run():
    fetch_and_store_news()
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
