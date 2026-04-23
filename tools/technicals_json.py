#!/usr/bin/env python3
"""Helpers for loading technical-cache JSON safely.

Some live cache files can contain trailing junk after an otherwise valid JSON
object. Treat the first valid JSON object as canonical so runtime readers do
not fail hard on recoverable cache corruption.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


def load_technicals_timeframes(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}

    text = file_path.read_text()
    if not text.strip():
        return {}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        stripped = text.lstrip()
        try:
            payload, _ = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            return {}

    if not isinstance(payload, dict):
        return {}
    return payload.get("timeframes", {}) or {}


def parse_cache_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    match = re.match(r"^(.*?\.\d{6})\d+([+-]\d\d:\d\d)$", text)
    if match:
        text = f"{match.group(1)}{match.group(2)}"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def timeframe_age_minutes(
    timeframes: dict,
    timeframe: str,
    now: datetime | None = None,
) -> float | None:
    tf_payload = timeframes.get(timeframe) or {}
    ts = parse_cache_timestamp(tf_payload.get("timestamp") or tf_payload.get("time"))
    if ts is None:
        return None
    now_utc = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)
    return max(0.0, (now_utc - ts).total_seconds() / 60.0)
