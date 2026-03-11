from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from utils.strategy_tags import resolve_strategy_tag
from workers.common.setup_context import extract_setup_identity


_PATH_RAW = os.getenv("AUTO_CANARY_PATH", "config/auto_canary_overrides.json")
_PATH = Path(_PATH_RAW) if _PATH_RAW and _PATH_RAW.strip().lower() not in {"", "off", "none"} else None
_REFRESH_SEC = float(os.getenv("AUTO_CANARY_REFRESH_SEC", "30") or 30.0)
_MAX_AGE_SEC = max(0.0, float(os.getenv("AUTO_CANARY_MAX_AGE_SEC", "1800") or 1800.0))
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


def _parse_iso8601_epoch(raw: object) -> Optional[float]:
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


def _payload_is_stale(payload: Optional[dict[str, Any]]) -> bool:
    if _PATH is None or not isinstance(payload, dict) or _MAX_AGE_SEC <= 0.0:
        return False
    age_sec: Optional[float] = None
    for key in ("generated_at", "updated_at", "as_of", "timestamp"):
        epoch = _parse_iso8601_epoch(payload.get(key))
        if epoch is None:
            continue
        age_sec = max(0.0, time.time() - epoch)
        break
    if age_sec is None:
        try:
            age_sec = max(0.0, time.time() - float(_PATH.stat().st_mtime))
        except OSError:
            age_sec = None
    return bool(age_sec is not None and age_sec > _MAX_AGE_SEC)


def _setup_match_specificity(
    override: dict[str, Any],
    *,
    setup_context: dict[str, str],
) -> int:
    if not setup_context:
        return 0
    setup_fingerprint = str(override.get("setup_fingerprint") or "").strip()
    flow_regime = str(override.get("flow_regime") or "").strip()
    microstructure_bucket = str(override.get("microstructure_bucket") or "").strip()
    if setup_fingerprint:
        return 4 if setup_context.get("setup_fingerprint") == setup_fingerprint else 0
    if flow_regime and microstructure_bucket:
        return (
            3
            if setup_context.get("flow_regime") == flow_regime
            and setup_context.get("microstructure_bucket") == microstructure_bucket
            else 0
        )
    if flow_regime:
        return 2 if setup_context.get("flow_regime") == flow_regime else 0
    if microstructure_bucket:
        return 1 if setup_context.get("microstructure_bucket") == microstructure_bucket else 0
    return 0


def _select_setup_override(
    setup_overrides: object,
    *,
    entry_thesis: Optional[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    if not isinstance(setup_overrides, list):
        return None, None
    setup_context = extract_setup_identity(entry_thesis)
    if not setup_context:
        return None, None
    allowed_fields = {
        "enabled",
        "mode",
        "confidence",
        "units_multiplier",
        "probability_offset",
        "reason",
        "reasons",
        "source",
    }
    best_override: Optional[dict[str, Any]] = None
    best_meta: Optional[dict[str, Any]] = None
    best_rank = (0, 0)
    for item in setup_overrides:
        if not isinstance(item, dict):
            continue
        specificity = _setup_match_specificity(item, setup_context=setup_context)
        if specificity <= 0:
            continue
        samples = 0
        try:
            samples = int(float(item.get("samples") or item.get("trades") or 0))
        except (TypeError, ValueError):
            samples = 0
        rank = (specificity, samples)
        if rank < best_rank:
            continue
        payload = {key: item.get(key) for key in allowed_fields if key in item}
        if not payload:
            continue
        best_rank = rank
        best_override = payload
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        best_meta = {
            "match_dimension": str(item.get("match_dimension") or ""),
            "setup_fingerprint": str(item.get("setup_fingerprint") or "") or None,
            "flow_regime": str(item.get("flow_regime") or "") or None,
            "microstructure_bucket": str(item.get("microstructure_bucket") or "") or None,
            "samples": samples,
            "confidence": max(0.0, min(1.0, confidence)),
        }
    return best_override, best_meta


def current_override(
    strategy_tag: Optional[str],
    *,
    entry_thesis: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    payload = _load_payload()
    if not payload:
        return None
    if _payload_is_stale(payload):
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
            setup_override, setup_meta = _select_setup_override(
                advice.get("setup_overrides"),
                entry_thesis=entry_thesis,
            )
            if isinstance(setup_override, dict):
                merged.update(setup_override)
                if isinstance(setup_meta, dict):
                    merged["setup_override"] = setup_meta
            return merged
    return None
