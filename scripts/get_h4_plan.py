import os
import json
from datetime import datetime, timezone


def _prefer_local_env_toml():
    try:
        from utils import secrets as sec
        import pathlib
        p = pathlib.Path("config/env.local.toml")
        if p.exists():
            sec._ENV_PATH = p  # type: ignore[attr-defined]
    except Exception:
        pass


def _compute_plan_fallback(instrument: str) -> dict:
    """No Firestore path: compute factors and return plan JSON without storing."""
    import asyncio
    missing: list[str] = []
    try:
        import pandas as pd  # type: ignore
    except Exception:
        missing.append("pandas")
        # pandas_ta と httpx も必要
    try:
        from indicators.calc_core import IndicatorEngine  # type: ignore
    except Exception:
        # pandas_ta 未インストール等
        if "pandas_ta" not in missing:
            missing.append("pandas_ta")
    try:
        from market_data.candle_fetcher import fetch_historical_candles  # type: ignore
    except Exception:
        missing.append("httpx")
    try:
        from analysis.summary_ingestor import get_latest_news  # type: ignore
    except Exception:
        # ニュースは任意
        get_latest_news = lambda: {"short": [], "long": []}  # type: ignore
    try:
        from cloudrun.h4_planner_service import _call_gpt_plan  # type: ignore
    except Exception:
        _call_gpt_plan = lambda payload: {"status": "FALLBACK"}  # type: ignore

    if missing:
        return {"plan": {"status": "WAIT", "reason": "missing_deps", "missing": missing}}

    h4_c = asyncio.get_event_loop().run_until_complete(
        fetch_historical_candles(instrument, "H4", 60)
    )
    m1_c = asyncio.get_event_loop().run_until_complete(
        fetch_historical_candles(instrument, "M1", 60)
    )
    if len(h4_c) < 20 or len(m1_c) < 20:
        return {"plan": {"status": "WAIT", "reason": "insufficient_candles"}}

    h4 = IndicatorEngine.compute(pd.DataFrame(h4_c)[["open", "high", "low", "close"]])
    m1 = IndicatorEngine.compute(pd.DataFrame(m1_c)[["open", "high", "low", "close"]])

    # Use local ingested news (SQLite) so it works without Firestore
    news = get_latest_news()
    recent_news = []
    for item in (news.get("short", []) + news.get("long", [])):
        recent_news.append({
            "summary": item.get("summary", ""),
            "sentiment": int(item.get("sentiment", 0) or 0),
            "impact": int(item.get("impact", 1) or 1),
        })

    payload = {
        "factors_h4": h4,
        "factors_m1": m1,
        "recent_news": recent_news[:5],
    }
    plan = _call_gpt_plan(payload)
    return {"plan": plan, "payload": payload}


def main():
    _prefer_local_env_toml()
    instrument = os.environ.get("INSTRUMENT", "USD_JPY")
    # Ensure repo root is on sys.path
    import sys, pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    now = datetime.utcnow().replace(tzinfo=timezone.utc)

    # Try Firestore path first
    try:
        from cloudrun.h4_planner_service import _get_or_create_plan_for_segment
        doc = _get_or_create_plan_for_segment(now)
    except Exception:
        doc = _compute_plan_fallback(instrument)

    print(json.dumps(doc, ensure_ascii=False))


if __name__ == "__main__":
    main()
