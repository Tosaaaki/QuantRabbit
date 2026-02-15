"""Dynamic per-strategy analysis feedback bus.

Analysis services can emit JSON updates and this module exposes them as lightweight
runtime hints for strategy entries.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional


_FEEDBACK_ENABLED = os.getenv("STRATEGY_FEEDBACK_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}
_FEEDBACK_PATH = os.getenv("STRATEGY_FEEDBACK_PATH") or os.getenv(
    "STRATEGY_FEEDBACK_JSON", "logs/strategy_feedback.json"
)
if _FEEDBACK_PATH is not None and _FEEDBACK_PATH.strip().lower() in {"", "none", "off"}:
    _FEEDBACK_PATH = None

_FEEDBACK_REFRESH_SEC = float(os.getenv("STRATEGY_FEEDBACK_REFRESH_SEC", "30") or 30)

_PATH: Optional[Path] = Path(_FEEDBACK_PATH) if _FEEDBACK_PATH else None
_CACHE: dict[str, Any] = {
    "loaded": 0.0,
    "mtime": None,
    "payload": None,
}
_PARAM_CACHE: dict[str, Any] = {"loaded": 0.0, "paths": None, "payloads": []}


def _parse_csv_paths(raw: Optional[str]) -> list[Path]:
    if not raw:
        return []
    paths: list[Path] = []
    for token in re.split(r"[,;\n]+", str(raw)):
        token = token.strip()
        if token:
            paths.append(Path(token))
    return paths


def _normalize_strategy_key(value: Optional[str]) -> str:
    if not value:
        return ""
    return "".join(ch for ch in str(value).lower() if ch.isalnum() or ch == "_")


def _read_file_payload(path: Path) -> Optional[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None
    parsed: object
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            import yaml  # type: ignore
        except Exception:
            return None
        try:
            parsed = yaml.safe_load(raw)  # type: ignore
        except Exception:
            return None
    if not isinstance(parsed, dict):
        return None
    return parsed


_PARAM_SOURCE_PATHS = _parse_csv_paths(
    os.getenv("STRATEGY_PARAMETER_PATHS")
    or os.getenv("STRATEGY_PARAMS_PATHS", "config/tuning_overlay.yaml,config/tuning_overrides.yaml")
)


def _lookup_strategy_section(
    payload: Optional[dict[str, Any]],
    strategy_key: str,
) -> Optional[dict[str, Any]]:
    if not payload:
        return None
    if strategy_key in payload:
        strategy_cfg = payload.get(strategy_key)
        if isinstance(strategy_cfg, dict):
            return strategy_cfg
    if strategy_key.upper() in payload:
        strategy_cfg = payload.get(strategy_key.upper())
        if isinstance(strategy_cfg, dict):
            return strategy_cfg
    for key, strategy_cfg in payload.items():
        if not isinstance(key, str) or not isinstance(strategy_cfg, dict):
            continue
        if _normalize_strategy_key(key) == strategy_key:
            return strategy_cfg
    return None


def _resolve_strategy_metadata(
    payload: dict[str, Any],
    strategy_key: str,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for section_key in (
        "strategies",
        "strategy_profiles",
        "strategy_params",
        "strategy_parameters",
        "strategy_configs",
        "strategy_config",
    ):
        section = payload.get(section_key)
        if not isinstance(section, dict):
            continue
        matched = _lookup_strategy_section(section, strategy_key)
        if not matched:
            continue
        for mkey, mvalue in matched.items():
            if mkey == "enabled":
                resolved["enabled"] = mvalue
            elif mkey in {"status", "state"}:
                resolved[mkey] = mvalue
            else:
                resolved.setdefault("strategy_params", {})[mkey] = mvalue
    return resolved


def _load_param_payloads() -> list[dict[str, Any]]:
    now = time.time()
    if (
        _PARAM_CACHE["payloads"]
        and _PARAM_CACHE.get("loaded", 0.0)
        and (now - float(_PARAM_CACHE["loaded"])) < _FEEDBACK_REFRESH_SEC
        and _PARAM_CACHE.get("paths") == [_p.as_posix() for _p in _PARAM_SOURCE_PATHS]
    ):
        return _PARAM_CACHE["payloads"]  # type: ignore[return-value]

    payloads: list[dict[str, Any]] = []
    for path in _PARAM_SOURCE_PATHS:
        payload = _read_file_payload(path)
        if payload:
            payloads.append(payload)
    _PARAM_CACHE.update(
        {
            "loaded": now,
            "paths": [_p.as_posix() for _p in _PARAM_SOURCE_PATHS],
            "payloads": payloads,
        }
    )
    return payloads


def _to_float(value: object, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed or parsed == float("inf") or parsed == float("-inf"):
        return default
    return parsed


def _to_bool(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_multiplier(value: object, default: float, min_value: float = 0.0, max_value: float = 10.0) -> float:
    parsed = _to_float(value, default=default)
    if parsed is None:
        return default
    if parsed < min_value:
        return min_value
    if parsed > max_value:
        return max_value
    return parsed


def _coerce_abs_int(value: object) -> Optional[float]:
    parsed = _to_float(value, default=None)
    if parsed is None:
        return None
    if parsed < 0:
        parsed = 0.0
    return float(abs(parsed))


def _read_payload() -> dict[str, Any]:
    if _PATH is None:
        return {}
    try:
        stat = _PATH.stat()
    except FileNotFoundError:
        return {}
    except OSError:
        return {}
    if (
        _CACHE["mtime"] is not None
        and _CACHE["payload"] is not None
        and float(_CACHE["mtime"]) == float(stat.st_mtime)
    ):
        return _CACHE["payload"]
    payload = _read_file_payload(_PATH)
    if payload is None:
        return {}
    _CACHE["mtime"] = float(stat.st_mtime)
    _CACHE["loaded"] = time.time()
    _CACHE["payload"] = payload
    return payload


def _refresh_if_needed(force: bool = False) -> None:
    if not _FEEDBACK_ENABLED or _PATH is None:
        return
    if not force and (_CACHE["loaded"] or 0.0) and (
        time.time() - float(_CACHE["loaded"]) < _FEEDBACK_REFRESH_SEC
    ):
        return
    if _PATH is None:
        return
    _read_payload()


def _apply_pocket_override(
    base: dict[str, Any],
    strategy_cfg: dict[str, Any],
    pocket: Optional[str],
) -> None:
    if not strategy_cfg:
        return
    for key in ("pockets", "pocket_profiles"):
        bucket = strategy_cfg.get(key)
        if not isinstance(bucket, dict):
            continue
        if not pocket:
            break
        pkey = _normalize_strategy_key(pocket)
        override = bucket.get(pkey)
        if isinstance(override, dict):
            base.update(override)
            return
        override = bucket.get(str(pocket).strip().lower())
        if isinstance(override, dict):
            base.update(override)
            return


def current_advice(
    strategy_tag: Optional[str],
    *,
    pocket: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    if not _FEEDBACK_ENABLED or not strategy_tag:
        return None
    _refresh_if_needed()
    payload = _read_payload()
    if not isinstance(payload, dict):
        return None
    strategy_key = _normalize_strategy_key(strategy_tag)

    # Resolve optional strategy metadata from external sources (tuning/overlay/reentry
    # profiles). Strategy-specific parameters are not directly used as hard gates;
    # they are carried as analysis metadata and used by external analysis services.
    strategies = payload.get("strategies")
    strategy_cfg = None
    if isinstance(strategies, dict):
        strategy_cfg = _lookup_strategy_section(strategies, strategy_key)

    metadata = _resolve_strategy_metadata(payload, strategy_key)
    for path_payload in _load_param_payloads():
        merged_meta = _resolve_strategy_metadata(path_payload, strategy_key)
        if merged_meta:
            metadata = dict(metadata)
            if "enabled" not in metadata and merged_meta.get("enabled") is not None:
                metadata["enabled"] = merged_meta["enabled"]
            if not metadata.get("status") and merged_meta.get("status"):
                metadata["status"] = merged_meta["status"]
            strategy_params = metadata.get("strategy_params")
            if not isinstance(strategy_params, dict):
                strategy_params = {}
            if isinstance(merged_meta.get("strategy_params"), dict):
                strategy_params.update(merged_meta["strategy_params"])  # type: ignore[union-attr]
            if strategy_params:
                metadata["strategy_params"] = strategy_params

    if strategy_cfg is None and not metadata:
        return None
    merged: dict[str, Any] = {}
    defaults = payload.get("strategy_defaults")
    if isinstance(defaults, dict):
        merged.update(defaults)
    if isinstance(strategy_cfg, dict):
        merged.update(strategy_cfg)
    if isinstance(metadata.get("strategy_params"), dict):
        merged.update(metadata.get("strategy_params") or {})
    strategy_pocket_source = None
    if isinstance(strategy_cfg, dict):
        strategy_pocket_source = strategy_cfg
    elif isinstance(metadata.get("strategy_params"), dict):
        strategy_pocket_source = metadata.get("strategy_params")
    if isinstance(strategy_pocket_source, dict):
        _apply_pocket_override(merged, strategy_pocket_source, pocket)
    if not _to_bool(merged.get("enabled"), True):
        return None
    metadata_status = str(metadata.get("status") or "").strip().lower()
    if metadata_status in {"off", "inactive", "stopped", "pause", "paused", "disabled", "stop"}:
        return None
    advice: dict[str, Any] = {}
    advice["entry_probability_multiplier"] = _coerce_multiplier(
        merged.get("entry_probability_multiplier", merged.get("probability_multiplier", 1.0)),
        default=1.0,
    )
    advice["entry_probability_delta"] = max(
        -1.0, min(1.0, _to_float(merged.get("entry_probability_delta"), 0.0) or 0.0)
    )
    advice["entry_units_multiplier"] = _coerce_multiplier(
        merged.get("entry_units_multiplier", merged.get("units_multiplier", 1.0)),
        default=1.0,
    )
    entry_units_min = _coerce_abs_int(
        merged.get("entry_units_min")
        if merged.get("entry_units_min") is not None
        else merged.get("units_abs_min")
    )
    entry_units_max = _coerce_abs_int(
        merged.get("entry_units_max")
        if merged.get("entry_units_max") is not None
        else merged.get("units_abs_max")
    )
    if entry_units_min is not None:
        advice["entry_units_min"] = entry_units_min
    if entry_units_max is not None:
        advice["entry_units_max"] = entry_units_max
    advice["sl_distance_multiplier"] = _coerce_multiplier(
        merged.get("sl_distance_multiplier", merged.get("sl_multiplier", 1.0)),
        default=1.0,
        min_value=0.05,
        max_value=5.0,
    )
    advice["tp_distance_multiplier"] = _coerce_multiplier(
        merged.get("tp_distance_multiplier", merged.get("tp_multiplier", 1.0)),
        default=1.0,
        min_value=0.05,
        max_value=5.0,
    )
    notes = merged.get("notes")
    if notes is not None:
        advice["notes"] = str(notes)
    strategy_params = metadata.get("strategy_params")
    if isinstance(strategy_params, dict) and strategy_params:
        advice["strategy_params"] = dict(strategy_params)
    has_strategy_params = isinstance(advice.get("strategy_params"), dict) and bool(advice["strategy_params"])
    advice["_meta"] = {
        "strategy_tag": strategy_tag,
        "pocket": pocket,
        "source": str(_PATH) if _PATH else "",
        "version": payload.get("version"),
        "updated_at": payload.get("updated_at"),
    }

    # Return None when there are no actual tuning knobs (no-op).
    if (
        advice["entry_probability_multiplier"] == 1.0
        and advice["entry_probability_delta"] == 0.0
        and advice["entry_units_multiplier"] == 1.0
        and advice["sl_distance_multiplier"] == 1.0
        and advice["tp_distance_multiplier"] == 1.0
        and "entry_units_min" not in advice
        and "entry_units_max" not in advice
        and not has_strategy_params
    ):
        return None
    return advice
