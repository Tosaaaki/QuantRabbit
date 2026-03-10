from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from utils.strategy_tags import resolve_strategy_tag


_PATH_RAW = os.getenv("AUTO_CANARY_PATH", "config/auto_canary_overrides.json")
_PATH = Path(_PATH_RAW) if _PATH_RAW and _PATH_RAW.strip().lower() not in {"", "off", "none"} else None
_REFRESH_SEC = float(os.getenv("AUTO_CANARY_REFRESH_SEC", "30") or 30.0)
_CACHE: dict[str, Any] = {"loaded": 0.0, "mtime": None, "payload": None}


def _normalize_strategy_key(value: Optional[str]) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _base_strategy_tag(value: Optional[str]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    for suffix in ("-long", "-short", "_long", "_short", "/long", "/short"):
        if lowered.endswith(suffix):
            base = text[: -len(suffix)].rstrip("-_/ ")
            if base:
                return base
    return ""


def _strategy_lookup_candidates(strategy_tag: Optional[str], *, known_keys: Optional[list[str]] = None) -> list[str]:
    candidates: list[str] = []

    def _append(raw: Optional[str]) -> None:
        key = _normalize_strategy_key(raw)
        if key and key not in candidates:
            candidates.append(key)

    base_tag = _base_strategy_tag(strategy_tag)
    _append(strategy_tag)
    _append(base_tag)
    for raw in (strategy_tag, base_tag):
        if not raw:
            continue
        try:
            _append(resolve_strategy_tag(raw, known_keys=known_keys))
        except Exception:
            continue
    return candidates


def _load_payload() -> Optional[dict[str, Any]]:
    if _PATH is None or not _PATH.exists():
        return None
    now = time.time()
    try:
        stat = _PATH.stat()
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
        payload = json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    _CACHE.update({"loaded": now, "mtime": float(stat.st_mtime), "payload": payload})
    return payload


def current_override(strategy_tag: Optional[str]) -> Optional[dict[str, Any]]:
    payload = _load_payload()
    if not payload:
        return None
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict):
        return None
    known_keys = [str(key) for key in strategies.keys() if isinstance(key, str)]
    candidates = _strategy_lookup_candidates(strategy_tag, known_keys=known_keys)
    for candidate in candidates:
        for key, advice in strategies.items():
            if not isinstance(key, str) or not isinstance(advice, dict):
                continue
            if _normalize_strategy_key(key) != candidate:
                continue
            merged = dict(advice)
            merged.setdefault("strategy_key", key)
            merged.setdefault("generated_at", payload.get("generated_at"))
            return merged
    return None
