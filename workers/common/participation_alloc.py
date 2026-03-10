from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from utils.strategy_tags import resolve_strategy_tag


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
    resolved = resolve_strategy_tag(raw)
    if resolved and resolved not in keys:
        keys.append(resolved)
    if "-" in raw:
        base = raw.split("-", 1)[0]
        if base and base not in keys:
            keys.append(base)
    if ":" in raw:
        base = raw.split(":", 1)[0]
        if base and base not in keys:
            keys.append(base)
    dedup: list[str] = []
    for key in keys:
        if key and key not in dedup:
            dedup.append(key)
    return dedup


def _parse_iso8601_epoch(raw: Any) -> Optional[float]:
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


def _payload_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_as_of = payload.get("as_of")
    as_of = str(raw_as_of).strip() if raw_as_of is not None else ""
    age_sec: Optional[float] = None
    epoch = _parse_iso8601_epoch(as_of)
    if epoch is not None:
        age_sec = max(0.0, time.time() - epoch)
    stale_max_age_sec = max(
        0.0,
        _safe_float(os.getenv("PARTICIPATION_ALLOC_MAX_AGE_SEC"), 1800.0),
    )
    is_stale = bool(
        as_of
        and age_sec is not None
        and stale_max_age_sec > 0.0
        and age_sec > stale_max_age_sec
    )
    return {
        "payload_as_of": as_of,
        "payload_age_sec": age_sec,
        "payload_stale": is_stale,
    }


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
        logging.warning("[PARTICIPATION_ALLOC] failed to read %s: %s", path, exc)
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
    file_path = Path(path or os.getenv("PARTICIPATION_ALLOC_PATH", "config/participation_alloc.json"))
    payload = _load_payload(file_path, ttl_sec=ttl_sec)
    base = {
        "found": False,
        "strategy_key": "",
        "units_multiplier": 1.0,
        "lot_multiplier": 1.0,
        "probability_multiplier": 1.0,
        "probability_offset": 0.0,
        "probability_boost": 0.0,
        "cadence_floor": 1.0,
        "action": "hold",
        "attempts": 0,
        "preflights": 0,
        "fills": 0,
        "filled": 0,
        "filled_rate": 0.0,
        "fill_rate": 0.0,
        "attempt_share": 0.0,
        "current_share": 0.0,
        "fill_share": 0.0,
        "target_share": 0.0,
        "share_gap": 0.0,
        "quality_score": 0.0,
        "hard_block_rate": 0.0,
        **_payload_meta(payload or {}),
    }
    if not payload:
        return base
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict):
        return base

    pocket_l = str(pocket or "").strip().lower()
    lower_key_map: Dict[str, str] = {}
    for raw_key in strategies.keys():
        if not isinstance(raw_key, str):
            continue
        norm_key = raw_key.strip().lower()
        if norm_key and norm_key not in lower_key_map:
            lower_key_map[norm_key] = raw_key

    for key in _candidate_keys(strategy):
        lookup_key = key
        item = strategies.get(lookup_key)
        if not isinstance(item, dict):
            mapped_key = lower_key_map.get(str(key).strip().lower())
            if mapped_key:
                lookup_key = mapped_key
                item = strategies.get(lookup_key)
        if not isinstance(item, dict):
            continue
        item_pocket = str(item.get("pocket") or "").strip().lower()
        if item_pocket and pocket_l and item_pocket != pocket_l:
            continue
        return {
            "found": True,
            "strategy_key": str(lookup_key),
            "units_multiplier": _safe_float(
                item.get("units_multiplier"),
                _safe_float(item.get("lot_multiplier"), 1.0),
            ),
            "lot_multiplier": _safe_float(
                item.get("lot_multiplier"),
                _safe_float(item.get("units_multiplier"), 1.0),
            ),
            "probability_multiplier": _safe_float(item.get("probability_multiplier"), 1.0),
            "probability_offset": _safe_float(item.get("probability_offset"), 0.0),
            "probability_boost": _safe_float(
                item.get("probability_boost"),
                _safe_float(item.get("probability_offset"), 0.0),
            ),
            "cadence_floor": _safe_float(item.get("cadence_floor"), 1.0),
            "action": str(item.get("action") or "hold"),
            "attempts": _safe_int(item.get("attempts"), _safe_int(item.get("preflights"), 0)),
            "preflights": _safe_int(item.get("preflights"), _safe_int(item.get("attempts"), 0)),
            "fills": _safe_int(item.get("fills"), _safe_int(item.get("filled"), 0)),
            "filled": _safe_int(item.get("filled"), _safe_int(item.get("fills"), 0)),
            "filled_rate": _safe_float(item.get("filled_rate"), _safe_float(item.get("fill_rate"), 0.0)),
            "fill_rate": _safe_float(item.get("fill_rate"), _safe_float(item.get("filled_rate"), 0.0)),
            "attempt_share": _safe_float(item.get("attempt_share"), _safe_float(item.get("current_share"), 0.0)),
            "current_share": _safe_float(item.get("current_share"), _safe_float(item.get("attempt_share"), 0.0)),
            "fill_share": _safe_float(item.get("fill_share"), _safe_float(item.get("target_share"), 0.0)),
            "target_share": _safe_float(item.get("target_share"), _safe_float(item.get("fill_share"), 0.0)),
            "share_gap": _safe_float(item.get("share_gap"), 0.0),
            "quality_score": _safe_float(item.get("quality_score"), 0.0),
            "hard_block_rate": _safe_float(item.get("hard_block_rate"), 0.0),
            **_payload_meta(payload),
        }
    return base
