from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


_CACHE: Dict[str, Dict[str, Any]] = {}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _candidate_keys(strategy: str) -> list[str]:
    raw = str(strategy or "").strip()
    if not raw:
        return []
    keys = [raw]
    if "-" in raw:
        keys.append(raw.split("-", 1)[0])
    if ":" in raw:
        keys.append(raw.split(":", 1)[0])
    dedup: list[str] = []
    for key in keys:
        if key and key not in dedup:
            dedup.append(key)
    return dedup


def _load_payload(path: Path, ttl_sec: float) -> Optional[Dict[str, Any]]:
    cache_key = str(path)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached is not None and now - float(cached.get("loaded_ts", 0.0)) < max(1.0, ttl_sec):
        return cached.get("payload")
    if not path.exists():
        _CACHE.pop(cache_key, None)
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    if cached is not None and float(cached.get("mtime", 0.0)) == float(stat.st_mtime):
        cached["loaded_ts"] = now
        return cached.get("payload")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        logging.warning("[DYN_ALLOC] failed to read %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    _CACHE[cache_key] = {"mtime": float(stat.st_mtime), "loaded_ts": now, "payload": payload}
    return payload


def load_strategy_profile(
    strategy: str,
    pocket: str,
    *,
    path: str | Path | None = None,
    ttl_sec: float = 20.0,
) -> Dict[str, Any]:
    file_path = Path(path or os.getenv("WORKER_DYNAMIC_ALLOC_PATH", "config/dynamic_alloc.json"))
    payload = _load_payload(file_path, ttl_sec=ttl_sec)
    base = {
        "found": False,
        "strategy_key": "",
        "score": 0.0,
        "lot_multiplier": 1.0,
        "trades": 0,
        "pf": 0.0,
        "win_rate": 0.0,
    }
    if not payload:
        return base
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict):
        return base
    pocket_l = str(pocket or "").strip().lower()
    for key in _candidate_keys(strategy):
        item = strategies.get(key)
        if not isinstance(item, dict):
            continue
        item_pocket = str(item.get("pocket") or "").strip().lower()
        if item_pocket and pocket_l and item_pocket != pocket_l:
            continue
        score = _safe_float(item.get("score"), 0.0)
        mult = _safe_float(item.get("lot_multiplier"), 1.0)
        if mult <= 0:
            mult = 1.0
        return {
            "found": True,
            "strategy_key": key,
            "score": score,
            "lot_multiplier": mult,
            "trades": _safe_int(item.get("trades"), 0),
            "pf": _safe_float(item.get("pf"), 0.0),
            "win_rate": _safe_float(item.get("win_rate"), 0.0),
        }
    return base

