"""
Cloud Run – News Summarizer
---------------------------
expects:  POST  from Pub/Sub push (JSON envelope)
env:
  BUCKET            : GCS bucket name  (fx-news)
  OPENAI_API_KEY    : OpenAI key (Secret Manager → env var)
"""

import os
import json
import base64
import logging
import datetime
from flask import Flask, request, abort
from google.cloud import storage
import openai

openai.api_key = os.getenv("OPENAI_API_KEY", "")
BUCKET = os.environ["BUCKET"]  # Fail fast if unset

app = Flask(__name__)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)


def _normalize_result(res: object) -> dict:
    """Return dict with keys 'summary' (str) and 'sentiment' (int)."""
    if isinstance(res, str):
        return {"summary": res.strip(), "sentiment": 0}
    if not isinstance(res, dict):
        return {"summary": str(res).strip(), "sentiment": 0}

    summary = res.get("summary") or res.get("要約") or res.get("title") or ""
    sentiment = res.get("sentiment") or res.get("impact") or 0
    try:
        sentiment = int(sentiment)
    except Exception:
        sentiment = 0
    return {"summary": summary.strip(), "sentiment": sentiment}


def summarize(text: str) -> dict:
    """GPT‑4o‑mini summarizer → return dict {summary, sentiment}"""
    prompt = (
        "以下のニュース本文を最大120字で日本語要約し、"
        "USD/JPYへのインパクトを -2〜+2 の整数で付与し "
        "次の JSON 形式で返してください。\\n"
        '{"summary":"...", "sentiment":-2〜+2}\\n'
        "### 原文\\n" + text[:1500]
    )
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "あなたは金融ニュースの要約アシスタントです。",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=120,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        data = _normalize_result(data)
    except Exception:
        data = _normalize_result(resp.choices[0].message.content.strip())
    return data


@app.route("/", methods=["POST"])
def ingest():
    envelope = request.get_json(force=True, silent=True)
    if not envelope or "message" not in envelope:
        logging.error("Invalid Pub/Sub message")
        return abort(400)

    msg = envelope["message"]
    attrs = msg.get("attributes", {})
    object_name = attrs.get("objectId")
    # Pub/Sub push without attributes? — fall back to decoding the data field
    if not object_name and "data" in msg:
        try:
            decoded = json.loads(base64.b64decode(msg["data"]))
            object_name = decoded.get("name", "")
        except Exception:
            object_name = ""

    if not object_name.startswith("raw/"):
        return "skip", 200  # heartbeat or non‑raw

    # download raw news
    blob = bucket.blob(object_name)
    raw_text = blob.download_as_text()
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        raw = {"body": raw_text}

    result = summarize(raw.get("title", "") + "\n" + raw.get("body", ""))

    summary_blob = bucket.blob(object_name.replace("raw/", "summary/", 1))
    payload = {
        "uid": raw.get("uid", object_name),
        "ts_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "summary": result["summary"],
        "sentiment": int(result.get("sentiment", 0) or 0),
        "horizon": "short",
        "pair_bias": "USD/JPY",
    }
    summary_blob.upload_from_string(
        json.dumps(payload),
        content_type="application/json",
        if_generation_match=0,  # create-only: don’t overwrite existing object ⇒ no delete permission needed
    )
    logging.info("summarized %s", object_name)
    return "OK", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
