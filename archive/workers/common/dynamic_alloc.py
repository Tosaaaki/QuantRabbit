from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from workers.common.setup_context import extract_setup_identity

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


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


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
        _safe_float(
            os.getenv("WORKER_DYNAMIC_ALLOC_UNKNOWN_FALLBACK_MAX_AGE_SEC"), 600.0
        ),
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
    if cached is not None and now - float(cached.get("loaded_ts", 0.0)) < max(
        1.0, ttl_sec
    ):
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
    _CACHE[cache_key] = {
        "mtime": float(stat.st_mtime),
        "loaded_ts": now,
        "payload": payload,
    }
    return payload


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
        return (
            1
            if setup_context.get("microstructure_bucket") == microstructure_bucket
            else 0
        )
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
        "score",
        "lot_multiplier",
        "strategy_lot_multiplier",
        "effective_min_lot_multiplier",
        "trades",
        "win_rate",
        "weighted_win_rate",
        "pf",
        "jpy_pf",
        "avg_pips",
        "sum_pips",
        "avg_realized_jpy",
        "sum_realized_jpy",
        "realized_jpy_per_1k_units",
        "sl_rate",
        "margin_closeout_rate",
        "market_close_rate",
        "market_close_loss_rate",
        "market_close_loss_share",
        "downside_share",
        "jpy_downside_share",
        "allow_loser_block",
        "allow_winner_only",
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
        trades = _safe_int(item.get("trades"), 0)
        rank = (specificity, trades)
        if rank < best_rank:
            continue
        payload = {key: item.get(key) for key in allowed_fields if key in item}
        if not payload:
            continue
        best_rank = rank
        best_override = payload
        best_meta = {
            "match_dimension": str(item.get("match_dimension") or ""),
            "setup_fingerprint": str(item.get("setup_fingerprint") or "") or None,
            "flow_regime": str(item.get("flow_regime") or "") or None,
            "microstructure_bucket": str(item.get("microstructure_bucket") or "")
            or None,
            "trades": trades,
            "score": _safe_float(item.get("score"), 0.0),
            "pf": _safe_float(item.get("pf"), 0.0),
            "win_rate": _safe_float(item.get("win_rate"), 0.0),
        }
    return best_override, best_meta


def load_strategy_profile(
    strategy: str,
    pocket: str,
    *,
    entry_thesis: Optional[dict[str, Any]] = None,
    path: str | Path | None = None,
    ttl_sec: float = 20.0,
) -> Dict[str, Any]:
    file_path = Path(
        path or os.getenv("WORKER_DYNAMIC_ALLOC_PATH", "config/dynamic_alloc.json")
    )
    payload = _load_payload(file_path, ttl_sec=ttl_sec)
    base = {
        "found": False,
        "strategy_key": "",
        "score": 0.0,
        "lot_multiplier": 1.0,
        "trades": 0,
        "pf": 0.0,
        "win_rate": 0.0,
        "allow_loser_block": True,
        "allow_winner_only": True,
        "soft_participation": False,
        "payload_as_of": "",
        "payload_age_sec": None,
        "payload_stale": False,
    }
    if not payload:
        return base
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict):
        return base
    payload_meta = _payload_meta(payload)
    policy = payload.get("allocation_policy")
    policy_dict = policy if isinstance(policy, dict) else {}
    policy_soft_participation = _safe_bool(policy_dict.get("soft_participation"), False)
    policy_allow_loser_block = _safe_bool(policy_dict.get("allow_loser_block"), True)
    policy_allow_winner_only = _safe_bool(policy_dict.get("allow_winner_only"), True)
    policy_min_mult = _safe_float(policy_dict.get("min_lot_multiplier"), 0.0)
    policy_max_mult = _safe_float(policy_dict.get("max_lot_multiplier"), 0.0)

    base["allow_loser_block"] = policy_allow_loser_block
    base["allow_winner_only"] = policy_allow_winner_only
    base["soft_participation"] = policy_soft_participation
    base.update(payload_meta)
    if policy_soft_participation and policy_min_mult > 0.0:
        base["lot_multiplier"] = max(0.0, float(policy_min_mult))

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
        score = _safe_float(item.get("score"), 0.0)
        mult = _safe_float(item.get("lot_multiplier"), 1.0)
        if mult <= 0:
            mult = 1.0
        item_min_mult_raw = item.get("effective_min_lot_multiplier")
        if item_min_mult_raw is None:
            item_min_mult_raw = item.get("min_lot_multiplier")
        item_min_mult = _safe_float(item_min_mult_raw, 0.0)
        item_max_mult = _safe_float(item.get("max_lot_multiplier"), policy_max_mult)
        if item_min_mult > 0.0:
            mult = max(mult, item_min_mult)
        if item_max_mult > 0.0:
            mult = min(
                mult, max(item_min_mult if item_min_mult > 0.0 else 0.0, item_max_mult)
            )
        allow_loser_block = _safe_bool(
            item.get("allow_loser_block"), policy_allow_loser_block
        )
        allow_winner_only = _safe_bool(
            item.get("allow_winner_only"), policy_allow_winner_only
        )
        profile = {
            "found": True,
            "strategy_key": str(lookup_key),
            "score": score,
            "lot_multiplier": mult,
            "trades": _safe_int(item.get("trades"), 0),
            "pf": _safe_float(item.get("pf"), 0.0),
            "win_rate": _safe_float(item.get("win_rate"), 0.0),
            "allow_loser_block": allow_loser_block,
            "allow_winner_only": allow_winner_only,
            "soft_participation": policy_soft_participation,
            **payload_meta,
        }
        setup_override, setup_meta = _select_setup_override(
            item.get("setup_overrides"),
            entry_thesis=entry_thesis,
        )
        if isinstance(setup_override, dict):
            profile.update(setup_override)
            if isinstance(setup_meta, dict):
                profile["setup_override"] = setup_meta
        else:
            setup_context = extract_setup_identity(entry_thesis)
            if setup_context and mult < 1.0:
                profile["setup_identity"] = dict(setup_context)
                profile["setup_trim_fallback"] = "strategy_level_trim"
        return profile
    if policy_soft_participation and policy_min_mult > 0.0:
        if payload_meta["payload_stale"]:
            return {
                **base,
                "lot_multiplier": 1.0,
                "soft_participation_skip_reason": "stale_unknown_strategy",
            }
        fallback_key = ""
        candidates = _candidate_keys(strategy)
        if candidates:
            fallback_key = candidates[0]
        return {
            "found": True,
            "strategy_key": fallback_key,
            "score": 0.0,
            "lot_multiplier": max(0.0, policy_min_mult),
            "trades": 0,
            "pf": 0.0,
            "win_rate": 0.0,
            "allow_loser_block": policy_allow_loser_block,
            "allow_winner_only": policy_allow_winner_only,
            "soft_participation": policy_soft_participation,
            **payload_meta,
        }
    return base
