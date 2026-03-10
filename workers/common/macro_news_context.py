from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any


_CACHE: dict[str, dict[str, Any]] = {}


def _parse_iso8601_epoch(raw: Any) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).timestamp()


def _load_payload(path: Path, ttl_sec: float) -> dict[str, Any] | None:
    cache_key = str(path)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached is not None and now - float(cached.get("loaded_ts", 0.0)) < max(1.0, ttl_sec):
        payload = cached.get("payload")
        return payload if isinstance(payload, dict) else None
    if not path.exists():
        _CACHE.pop(cache_key, None)
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    if cached is not None and float(cached.get("mtime", 0.0)) == float(stat.st_mtime):
        cached["loaded_ts"] = now
        payload = cached.get("payload")
        return payload if isinstance(payload, dict) else None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        logging.warning("[MACRO_NEWS] failed to read %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    _CACHE[cache_key] = {"mtime": float(stat.st_mtime), "loaded_ts": now, "payload": payload}
    return payload


def load_current_context(
    *,
    path: str | Path = "logs/macro_news_context.json",
    ttl_sec: float = 60.0,
    max_age_sec: float = 7200.0,
) -> dict[str, Any]:
    payload = _load_payload(Path(path), ttl_sec=ttl_sec)
    base = {
        "found": False,
        "generated_at": "",
        "age_sec": None,
        "stale": True,
        "event_severity": "unknown",
        "caution_window_active": False,
        "usd_jpy_bias": "neutral",
        "headlines": [],
        "sources": [],
        "market_snapshot": {},
    }
    if not payload:
        return base
    generated_at = str(payload.get("generated_at") or "").strip()
    epoch = _parse_iso8601_epoch(generated_at)
    age_sec = max(0.0, time.time() - epoch) if epoch is not None else None
    stale = bool(age_sec is None or age_sec > max(0.0, float(max_age_sec)))
    return {
        "found": True,
        "generated_at": generated_at,
        "age_sec": age_sec,
        "stale": stale,
        "event_severity": str(payload.get("event_severity") or "unknown"),
        "caution_window_active": bool(payload.get("caution_window_active")),
        "usd_jpy_bias": str(payload.get("usd_jpy_bias") or "neutral"),
        "headlines": payload.get("headlines") if isinstance(payload.get("headlines"), list) else [],
        "sources": payload.get("sources") if isinstance(payload.get("sources"), list) else [],
        "market_snapshot": payload.get("market_snapshot") if isinstance(payload.get("market_snapshot"), dict) else {},
    }
