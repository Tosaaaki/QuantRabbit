import os
import json
import logging
from datetime import datetime, timezone

try:
    from flask import Flask
except Exception:
    Flask = None  # type: ignore[assignment]
from google.cloud import firestore
from openai import OpenAI

from indicators.calc_core import IndicatorEngine
from market_data.candle_fetcher import fetch_historical_candles
import asyncio
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__) if Flask else None
fs = firestore.Client()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_PLANNER_MODEL = (
    os.environ.get("OPENAI_PLANNER_MODEL")
    or os.environ.get("OPENAI_MODEL")
    or "gpt-5-mini"
)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

INSTRUMENT = os.environ.get("INSTRUMENT", "USD_JPY")


def _segment_start(dt: datetime) -> datetime:
    h = (dt.hour // 4) * 4
    return dt.replace(hour=h, minute=0, second=0, microsecond=0)


def _get_latest_news(n=5):
    docs = list(
        fs.collection("news").order_by("ts_utc", direction=firestore.Query.DESCENDING).limit(n).stream()
    )
    items = []
    for d in docs:
        x = d.to_dict() or {}
        items.append({
            "summary": x.get("summary", ""),
            "sentiment": x.get("sentiment", 0),
            "impact": x.get("impact", 1),
        })
    return items


def _build_plan_payload(h4_fac: dict, m1_fac: dict) -> dict:
    return {
        "factors_h4": h4_fac,
        "factors_m1": m1_fac,
        "recent_news": _get_latest_news(5),
    }


def _call_gpt_plan(payload: dict) -> dict:
    if os.environ.get("GPT_DISABLE_ALL") or os.environ.get("DISABLE_GPT_PLANNER"):
        logging.info("[planner] GPT disabled via env; returning rule-based plan")
        client = None
    if not client:
        return {
            "bias": "hybrid",
            "ranked_strategies": ["TrendMA", "Donchian55", "BB_RSI"],
            "pocket_weights": {"micro": 0.4, "macro": 0.6},
            "risk_scale": 1.0,
            "tp_sl_profile": {"tp": 20, "sl": 20},
            "reversal_rules": {"enabled": True},
            "news_sensitivity": 0.5,
            "max_trades": 3,
        }
    system = (
        "You are an FX strategy planner. Generate a compact JSON plan for the next 4 hours "
        "on USD/JPY using H4/M1 technicals and recent news sentiment."
    )
    user = (
        "Provide a JSON with keys: bias('up'|'down'|'flat'|'hybrid'), ranked_strategies(array of names), "
        "pocket_weights({micro,macro} 0..1, sum~1), risk_scale(0.8..1.2), tp_sl_profile({tp,sl} pips), "
        "reversal_rules({enabled:bool}), news_sensitivity(0..1), max_trades(1..5). "
        f"Payload: {json.dumps(payload)}"
    )
    res = client.chat.completions.create(
        model=OPENAI_PLANNER_MODEL,
        response_format={"type": "json_object"},
        max_completion_tokens=180,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    content = res.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        return {
            "bias": "hybrid",
            "ranked_strategies": ["TrendMA", "Donchian55", "BB_RSI"],
            "pocket_weights": {"micro": 0.4, "macro": 0.6},
            "risk_scale": 1.0,
            "tp_sl_profile": {"tp": 20, "sl": 20},
            "reversal_rules": {"enabled": True},
            "news_sensitivity": 0.5,
            "max_trades": 3,
        }


def _plan_once_impl():
    try:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        seg = _segment_start(now)
        seg_id = seg.isoformat()

        doc = fs.collection("strategy_plans").document(seg_id).get()
        if doc.exists:
            return f"exists:{seg_id}", 200

        # 指標取得
        h4_c = asyncio.get_event_loop().run_until_complete(fetch_historical_candles(INSTRUMENT, "H4", 60))
        m1_c = asyncio.get_event_loop().run_until_complete(fetch_historical_candles(INSTRUMENT, "M1", 60))
        if len(h4_c) < 20 or len(m1_c) < 20:
            return "WAIT", 200
        h4 = IndicatorEngine.compute(pd.DataFrame(h4_c)[["open","high","low","close"]])
        m1 = IndicatorEngine.compute(pd.DataFrame(m1_c)[["open","high","low","close"]])

        payload = _build_plan_payload(h4, m1)
        plan = _call_gpt_plan(payload)
        plan_doc = {
            "segment_start": seg_id,
            "created_at": now.isoformat(timespec="seconds"),
            "instrument": INSTRUMENT,
            "payload": payload,
            "plan": plan,
        }
        fs.collection("strategy_plans").document(seg_id).set(plan_doc)
        return f"created:{seg_id}", 200
    except Exception as e:
        logging.error(f"plan_once error: {e}")
        return "ERROR", 500


# If running under Flask, expose routes; otherwise keep the implementation callable
if app:
    @app.route("/", methods=["GET"])  # セグメント計画を作成
    def plan_once():
        return _plan_once_impl()


def _get_or_create_plan_for_segment(seg_dt: datetime) -> dict:
    """Fetch the plan doc for the segment; create if missing and data available.

    Returns a dict with keys: segment_start, created_at, instrument, payload, plan
    Raises on unexpected errors.
    """
    seg_id = _segment_start(seg_dt).isoformat()
    ref = fs.collection("strategy_plans").document(seg_id)
    snap = ref.get()
    if snap.exists:
        return snap.to_dict() or {}

    # Compute factors and create a plan
    h4_c = asyncio.get_event_loop().run_until_complete(fetch_historical_candles(INSTRUMENT, "H4", 60))
    m1_c = asyncio.get_event_loop().run_until_complete(fetch_historical_candles(INSTRUMENT, "M1", 60))
    if len(h4_c) < 20 or len(m1_c) < 20:
        # Not enough data; return a minimal placeholder so caller can decide
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        return {
            "segment_start": seg_id,
            "created_at": now.isoformat(timespec="seconds"),
            "instrument": INSTRUMENT,
            "payload": {},
            "plan": {"status": "WAIT", "reason": "insufficient_candles"},
        }
    h4 = IndicatorEngine.compute(pd.DataFrame(h4_c)[["open", "high", "low", "close"]])
    m1 = IndicatorEngine.compute(pd.DataFrame(m1_c)[["open", "high", "low", "close"]])
    payload = _build_plan_payload(h4, m1)
    plan = _call_gpt_plan(payload)

    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    plan_doc = {
        "segment_start": seg_id,
        "created_at": now.isoformat(timespec="seconds"),
        "instrument": INSTRUMENT,
        "payload": payload,
        "plan": plan,
    }
    ref.set(plan_doc)
    return plan_doc


def _get_current_plan_impl():
    try:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        doc = _get_or_create_plan_for_segment(now)
        # When Flask is available, route will wrap this. Otherwise return tuple.
        if app:
            return app.response_class(
                response=json.dumps(doc, ensure_ascii=False),
                status=200,
                mimetype="application/json",
            )
        return json.dumps(doc, ensure_ascii=False), 200
    except Exception as e:
        logging.error(f"get_current_plan error: {e}")
        return "ERROR", 500

if app:
    @app.route("/current", methods=["GET"])  # 現在の4時間セグメントの戦略を返す
    def get_current_plan():
        return _get_current_plan_impl()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
