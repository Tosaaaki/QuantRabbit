# QuantRabbit/cloudrun/fetch_news_runner.py
"""
Cloud Run – Fetch News
----------------------
• 1 回呼ばれると Google ニュース RSS から 3 本だけ取って
  GCS (fx-news/raw/) に JSON を置く
• 失敗しても例外を握りつぶさず 500 を返す（Pub/Sub/Scheduler が再試行できるように）
"""

import os, json, uuid, datetime, feedparser
from flask import Flask, request, abort
from google.cloud import storage

BUCKET = os.environ["BUCKET"]
RSS    = os.environ.get("RSS_FEED", "https://news.google.com/rss/search?q=usd+jpy&hl=ja&gl=JP&ceid=JP:ja")

app = Flask(__name__)
bucket = storage.Client().bucket(BUCKET)

@app.route("/", methods=["POST", "GET"])
def run():
    feed = feedparser.parse(RSS)
    if not feed.entries:
        return abort(500, "no entry")

    for e in feed.entries[:3]:           # 3 本だけ
        uid  = str(uuid.uuid4())
        blob = bucket.blob(f"raw/{uid}.json")
        raw  = {
            "uid"  : uid,
            "title": e.title,
            "body" : e.summary,
            "link" : e.link,
            "fetched_at": datetime.datetime.utcnow().isoformat(timespec="seconds")
        }
        blob.upload_from_string(json.dumps(raw, ensure_ascii=False),
                                content_type="application/json")
    return f"uploaded {min(3,len(feed.entries))}", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))