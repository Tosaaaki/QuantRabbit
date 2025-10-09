import os
import json
import base64
import logging
import datetime
import time
from flask import Flask, request, abort
from google.cloud import storage
from google.cloud.exceptions import NotFound as CloudNotFound
from google.api_core.exceptions import NotFound as ApiNotFound
from openai import OpenAI
import requests

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_SUMMARIZER_MODEL = os.environ.get("OPENAI_SUMMARIZER_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5-nano"
BUCKET = os.environ.get("BUCKET")

if not BUCKET:
    logging.error("BUCKET environment variable is not set. Falling back to default bucket name.")
    # Keep default aligned with config/env.local.toml `news_bucket_name`
    BUCKET = "quantrabbit-fx-news"

app = Flask(__name__)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
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
    except (ValueError, TypeError):
        sentiment = 0
    return {"summary": summary.strip(), "sentiment": sentiment}

def summarize(text: str) -> dict:
    """GPT‑4o‑mini summarizer → return dict {summary, sentiment}
    If OPENAI_API_KEY is not set, fall back to a simple truncation summary.
    """
    if not client:
        logging.error("OPENAI_API_KEY is not set. Falling back to title/lead truncation.")
        # naive summary: first 120 chars
        t = (text or "").strip().splitlines()[0:2]
        joined = " ".join([x.strip() for x in t if x.strip()])
        return {"summary": joined[:120], "sentiment": 0}

    prompt = (
        "以下のニュース本文を最大120字で日本語要約し、"
        "USD/JPYへのインパクトを -2〜+2 の整数で付与し "
        "次の JSON 形式で返してください。\n"
        '{"summary":"...", "sentiment":-2〜+2}\n' 
        "### 原文\n" + text[:800]
    )
    try:
        logging.info("[summarize_start] sending_openai_request")
        resp = client.chat.completions.create(
            model=OPENAI_SUMMARIZER_MODEL,
            response_format={"type": "json_object"},
            temperature=0.2,
            messages=[
                {"role": "system", "content": "あなたは金融ニュースの要約アシスタントです。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=80,
        )
        logging.info("[summarize_done] received_openai_response")
        content = resp.choices[0].message.content
        logging.info(f"OpenAI response content: {content}")
        data = json.loads(content)
        return _normalize_result(data)
    except Exception as e:
        logging.error(f"[openai_error] An error occurred with OpenAI API: {e}")
        return {"summary": f"Error during summarization: {e}", "sentiment": 0}

def _process_one(object_name: str) -> bool:
    try:
        t0 = time.time()
        blob = bucket.blob(object_name)
        try:
            raw_text = blob.download_as_text()
        except (CloudNotFound, ApiNotFound):
            logging.info(f"[summarized_skip] object={object_name} reason=not_found")
            return True
        raw = json.loads(raw_text)
        title = raw.get("title", "")
        body = raw.get("body", "")
        text_to_summarize = (title + "\n" + body).strip()
        # When OPENAI is unavailable, summarize() will fallback; ensure title preferred
        result = summarize(text_to_summarize if text_to_summarize else title)
        summary_blob_name = object_name.replace("raw/", "summary/", 1)
        payload = {
            "uid": raw.get("uid", object_name),
            "ts_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "event_time": raw.get("event_time"),
            "impact": raw.get("impact", 1),
            "summary": result["summary"],
            "sentiment": int(result.get("sentiment", 0) or 0),
            "horizon": "short",
            "pair_bias": "USD/JPY",
        }
        summary_blob = bucket.blob(summary_blob_name)
        summary_blob.upload_from_string(json.dumps(payload), content_type="application/json")
        # optionally move raw -> processed
        new_name = object_name.replace("raw/", "processed/", 1)
        bucket.rename_blob(blob, new_name)
        elapsed_ms = int((time.time() - t0) * 1000)
        logging.info(
            f"[summarized_ok] object={object_name} summary_obj={summary_blob_name} ms={elapsed_ms} impact={raw.get('impact',1)}"
        )
        return True
    except (CloudNotFound, ApiNotFound):
        logging.info(f"[summarized_skip] object={object_name} reason=not_found_after_download")
        return True
    except Exception as e:
        logging.error(f"[summarized_err] object={object_name} error={e}")
        return False


@app.route("/", methods=["POST"]) 
def ingest():
    """Accept both Pub/Sub push and CloudEvent (GCS finalize) payloads.
    - Pub/Sub push: {"message": {"attributes": {objectId}, "data": base64(json)}}
    - CloudEvent (binary mode): Ce-* headers present, body JSON with {name, bucket}
    """
    try:
        # 1) CloudEvent (binary mode) detection
        ce_type = request.headers.get("Ce-Type") or request.headers.get("ce-type")
        object_name = None
        if ce_type:
            try:
                payload = request.get_json(silent=True) or {}
                object_name = payload.get("name")
                logging.info(f"[ingest-ce] Ce-Type={ce_type} bucket={payload.get('bucket')} name={object_name}")
            except Exception:
                object_name = None

        # 2) Pub/Sub envelope (fallback)
        if not object_name:
            envelope = request.get_json(silent=True) or {}
            msg = envelope.get("message") or {}
            attrs = msg.get("attributes", {})
            object_name = attrs.get("objectId")
            if not object_name and "data" in msg:
                try:
                    decoded_data = base64.b64decode(msg["data"]).decode('utf-8')
                    data_json = json.loads(decoded_data)
                    object_name = data_json.get("name")
                    logging.info(f"[ingest-ps] decoded name={object_name}")
                except Exception as e:
                    logging.info(f"[ingest-ps] decode failed: {e}")

        if not object_name or not isinstance(object_name, str):
            logging.info("[ingest] missing object name; skip")
            return "Skipping", 200

        # Only handle raw/ objects; ignore processed/ notifications to avoid noise
        if not object_name.startswith("raw/"):
            logging.info(f"[ingest] ignore non-raw object: {object_name}")
            return "Skipping", 200

        ok = _process_one(object_name)
        return ("OK", 200) if ok else ("ERROR", 500)

    except Exception as e:
        logging.error(f"Unhandled error in ingest: {e}", exc_info=True)
        return "Internal Server Error", 500

@app.route("/run", methods=["GET"])  # polling mode for Scheduler
def run_poll():
    try:
        processed = 0
        # process up to N raw objects per call
        limit = int(os.environ.get("SUM_LIMIT", "5"))
        for blob in bucket.list_blobs(prefix="raw/"):
            if processed >= limit:
                break
            if blob.name.endswith("/"):
                continue
            if _process_one(blob.name):
                processed += 1
        return f"processed={processed}", 200
    except Exception as e:
        logging.error(f"run_poll error: {e}")
        return "ERROR", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
