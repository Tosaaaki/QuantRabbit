from __future__ import annotations

import logging
import os
import pathlib
import time
from typing import Any, Optional

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None

LOG = logging.getLogger(__name__)

_FALSEY = {"", "0", "false", "no", "off"}

_STRATEGY_PROTECTION_ENABLED = os.getenv("STRATEGY_PROTECTION_ENABLED", "1").strip().lower() not in _FALSEY
_STRATEGY_PROTECTION_PATH = pathlib.Path(
    os.getenv("STRATEGY_PROTECTION_PATH", "config/strategy_exit_protections.yaml")
)
_STRATEGY_PROTECTION_TTL_SEC = max(1.0, float(os.getenv("STRATEGY_PROTECTION_TTL_SEC", "12.0") or 12.0))
_STRATEGY_PROTECTION_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}

# Keep consistent with execution.order_manager / workers.scalp_precision.exit_worker.
_STRATEGY_ALIAS_BASE = {
    "bbrsi": "BB_RSI",
    "bb_rsi": "BB_RSI",
    "trendma": "TrendMA",
    "donchian": "Donchian55",
    "donchian55": "Donchian55",
    "h1momentum": "H1Momentum",
    "m1scalper": "M1Scalper",
    "microlevelreactor": "MicroLevelReactor",
    "microrangebreak": "MicroRangeBreak",
    "microvwapbound": "MicroVWAPBound",
    "momentumburst": "MomentumBurst",
    "techfusion": "TechFusion",
    "macrotechfusion": "MacroTechFusion",
    "micropullbackfib": "MicroPullbackFib",
    "scalpreversalnwave": "ScalpReversalNWave",
    "rangecompressionbreak": "RangeCompressionBreak",
}


def base_strategy_tag(tag: Optional[str]) -> str:
    if not tag:
        return ""
    text = str(tag).strip()
    if not text:
        return ""
    base = text.split("-", 1)[0].strip() or text
    alias = _STRATEGY_ALIAS_BASE.get(base.lower())
    return alias or base


def _load_strategy_protection_config() -> dict:
    if not _STRATEGY_PROTECTION_ENABLED:
        return {"defaults": {}, "strategies": {}}
    now = time.monotonic()
    cached_ts = float(_STRATEGY_PROTECTION_CACHE.get("ts") or 0.0)
    if (now - cached_ts) < _STRATEGY_PROTECTION_TTL_SEC and isinstance(
        _STRATEGY_PROTECTION_CACHE.get("data"), dict
    ):
        return _STRATEGY_PROTECTION_CACHE["data"]  # type: ignore[return-value]

    # Keep last-known-good config on transient read/parse errors.
    payload: dict[str, Any] = {"defaults": {}, "strategies": {}}
    if isinstance(_STRATEGY_PROTECTION_CACHE.get("data"), dict):
        payload = _STRATEGY_PROTECTION_CACHE.get("data")  # type: ignore[assignment]
    if yaml is not None and _STRATEGY_PROTECTION_PATH.exists():
        try:
            loaded = yaml.safe_load(_STRATEGY_PROTECTION_PATH.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                payload = loaded
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Strategy protection config load failed (using cached): %s", exc)
    _STRATEGY_PROTECTION_CACHE["ts"] = now
    _STRATEGY_PROTECTION_CACHE["data"] = payload
    return payload


def _strategy_override(config: dict, strategy_tag: Optional[str]) -> dict:
    if not isinstance(config, dict):
        return {}
    strategies = config.get("strategies")
    if not isinstance(strategies, dict) or not strategy_tag:
        return {}
    base = base_strategy_tag(strategy_tag)
    candidates = [
        strategy_tag,
        base,
        strategy_tag.lower(),
        base.lower(),
    ]
    for key in candidates:
        if not key:
            continue
        override = strategies.get(key)
        if isinstance(override, dict):
            return override
    return {}


def _merge_profile(base: Optional[dict], override: Optional[dict]) -> dict:
    merged = dict(base) if isinstance(base, dict) else {}
    if isinstance(override, dict):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                nested = dict(merged.get(key) or {})
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
    return merged


def exit_profile_for_tag(strategy_tag: Optional[str]) -> dict:
    cfg = _load_strategy_protection_config()
    defaults = cfg.get("defaults") if isinstance(cfg, dict) else {}
    defaults_profile = defaults.get("exit_profile") if isinstance(defaults, dict) else None
    override = _strategy_override(cfg, strategy_tag)
    override_profile = override.get("exit_profile") if isinstance(override, dict) else None
    return _merge_profile(defaults_profile, override_profile)

