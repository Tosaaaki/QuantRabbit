from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

_PATH_RAW = os.getenv("MARKET_CONTEXT_PATH", "logs/market_context_latest.json")
_PATH = (
    Path(_PATH_RAW)
    if _PATH_RAW and _PATH_RAW.strip().lower() not in {"", "off", "none"}
    else None
)
_REFRESH_SEC = float(os.getenv("MARKET_CONTEXT_REFRESH_SEC", "30") or 30.0)
_STALE_MAX_AGE_SEC = max(
    60.0, float(os.getenv("MARKET_CONTEXT_MAX_AGE_SEC", "1800") or 1800.0)
)
_CACHE: dict[str, Any] = {"loaded": 0.0, "mtime": None, "payload": None}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _parse_iso_epoch(raw: Any) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        import datetime as dt

        parsed = dt.datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).timestamp()


def _load_payload(path: str | Path | None = None) -> Optional[dict[str, Any]]:
    effective_path = Path(path) if path is not None else _PATH
    if effective_path is None or not effective_path.exists():
        return None
    now = time.time()
    try:
        stat = effective_path.stat()
    except OSError:
        return None
    if (
        _CACHE.get("payload") is not None
        and (now - float(_CACHE.get("loaded", 0.0))) < max(1.0, _REFRESH_SEC)
        and _CACHE.get("mtime") == float(stat.st_mtime)
    ):
        payload = _CACHE.get("payload")
        return payload if isinstance(payload, dict) else None
    try:
        payload = json.loads(effective_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    _CACHE.update({"loaded": now, "mtime": float(stat.st_mtime), "payload": payload})
    return payload


def current_context(
    pair: str = "USD_JPY",
    *,
    path: str | Path | None = None,
    max_age_sec: float | None = None,
) -> Optional[dict[str, Any]]:
    payload = _load_payload(path=path)
    if not payload:
        return None
    pairs = payload.get("pairs") if isinstance(payload.get("pairs"), dict) else {}
    pair_key = str(pair or "").strip().lower()
    pair_row = pairs.get(pair_key) if isinstance(pairs, dict) else None
    pair_change = (
        _safe_float(pair_row.get("change_pct_24h"), 0.0)
        if isinstance(pair_row, dict)
        else 0.0
    )
    pair_price = (
        _safe_float(pair_row.get("price"), 0.0) if isinstance(pair_row, dict) else 0.0
    )

    dollar = payload.get("dollar") if isinstance(payload.get("dollar"), dict) else {}
    rates = payload.get("rates") if isinstance(payload.get("rates"), dict) else {}
    risk = payload.get("risk") if isinstance(payload.get("risk"), dict) else {}
    events = (
        payload.get("events_summary")
        if isinstance(payload.get("events_summary"), dict)
        else {}
    )

    dxy_change = _safe_float(dollar.get("dxy_change_pct_24h"), 0.0)
    spread_10y = _safe_float(rates.get("us_jp_10y_spread"), 0.0)
    risk_mode = str(risk.get("mode") or "neutral").strip().lower()
    high_impact_events = _safe_int(events.get("high_impact_events"), 0)
    total_events = _safe_int(events.get("total_events"), 0)
    next_event = (
        events.get("next_event") if isinstance(events.get("next_event"), dict) else {}
    )
    minutes_to_next_event = _safe_int(next_event.get("minutes_to_event"), 999999)

    risk_bias = 0.0
    if risk_mode == "risk_off":
        risk_bias = -0.25
    elif risk_mode == "risk_on":
        risk_bias = 0.10

    bias_score = (
        0.40 * math.tanh(pair_change / 0.25)
        + 0.30 * math.tanh(dxy_change / 0.20)
        + 0.30 * math.tanh(spread_10y / 1.50)
        + risk_bias
    )
    bias_score = max(-1.0, min(1.0, bias_score))

    event_severity = "none"
    if high_impact_events > 0 and -30 <= minutes_to_next_event <= 120:
        event_severity = "high"
    elif high_impact_events > 0 or (-120 <= minutes_to_next_event <= 240):
        event_severity = "medium"
    elif total_events > 0:
        event_severity = "low"

    generated_at = str(payload.get("generated_at") or "").strip()
    generated_epoch = _parse_iso_epoch(generated_at)
    age_sec = (
        max(0.0, time.time() - generated_epoch) if generated_epoch is not None else None
    )
    stale_max_age_sec = (
        _STALE_MAX_AGE_SEC if max_age_sec is None else max(60.0, float(max_age_sec))
    )
    stale = bool(age_sec is None or age_sec > stale_max_age_sec)

    if bias_score >= 0.15:
        bias_label = "usd_jpy_bullish"
    elif bias_score <= -0.15:
        bias_label = "usd_jpy_bearish"
    else:
        bias_label = "neutral"

    return {
        "generated_at": generated_at or None,
        "age_sec": age_sec,
        "stale": stale,
        "pair": str(pair or "").upper(),
        "pair_price": round(pair_price, 6) if pair_price > 0.0 else None,
        "pair_change_pct_24h": round(pair_change, 6),
        "dxy_change_pct_24h": round(dxy_change, 6),
        "us_jp_10y_spread": round(spread_10y, 6),
        "risk_mode": risk_mode,
        "high_impact_events": high_impact_events,
        "total_events": total_events,
        "minutes_to_next_event": (
            minutes_to_next_event if minutes_to_next_event != 999999 else None
        ),
        "next_event_name": str(next_event.get("name") or "").strip() or None,
        "event_severity": event_severity,
        "bias_score": round(bias_score, 6),
        "bias_label": bias_label,
    }
