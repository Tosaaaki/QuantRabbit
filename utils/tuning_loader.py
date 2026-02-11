from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import os
import threading

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None

_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_DIR = _ROOT / "logs" / "tuning"

_OVERLAY_ENV = os.getenv("TUNING_OVERLAY_PATH")
_OVERRIDES_ENV = os.getenv("TUNING_OVERRIDES_PATH")
_PRESETS_ENV = os.getenv("TUNING_PRESETS_PATH")

# Precedence: runtime overlay/overrides (git-untracked) -> legacy config overlay/overrides -> presets.
# If the operator explicitly sets TUNING_*_PATH, honor only that path (no fallback) to keep tests/ops deterministic.
_overlay_paths = (
    [Path(_OVERLAY_ENV)]
    if _OVERLAY_ENV
    else [_RUNTIME_DIR / "tuning_overlay.yaml", _ROOT / "config" / "tuning_overlay.yaml"]
)
_override_paths = (
    [Path(_OVERRIDES_ENV)]
    if _OVERRIDES_ENV
    else [_RUNTIME_DIR / "tuning_overrides.yaml", _ROOT / "config" / "tuning_overrides.yaml"]
)
_preset_paths = [Path(_PRESETS_ENV)] if _PRESETS_ENV else [_ROOT / "config" / "tuning_presets.yaml"]

_PATHS: list[Path] = []
_SEEN: set[str] = set()
for _p in [*_overlay_paths, *_override_paths, *_preset_paths]:
    key = str(_p)
    if key in _SEEN:
        continue
    _SEEN.add(key)
    _PATHS.append(_p)

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"data": None, "mtimes": {}}


def _load_yaml(path: Path) -> dict:
    if yaml is None:
        return {}
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _ensure_loaded() -> None:
    with _CACHE_LOCK:
        mtimes: dict[str, float | None] = {}
        changed = False
        for path in _PATHS:
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                mtime = None
            mtimes[str(path)] = mtime
            if _CACHE["mtimes"].get(str(path)) != mtime:
                changed = True
        if not changed and _CACHE.get("data") is not None:
            return
        data = {}
        for path in _PATHS:
            data[str(path)] = _load_yaml(path)
        _CACHE["data"] = data
        _CACHE["mtimes"] = mtimes


def _get_nested(data: dict, keys: Iterable[str]) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        if key not in cur:
            return None
        cur = cur.get(key)
    return cur


def get_tuning_value(keys: Iterable[str], default: Any = None) -> Any:
    """
    Resolve a tuning value from overlay -> overrides -> presets.
    keys: iterable path like ["exit","lowvol","upper_bound_max_sec"]
    """
    _ensure_loaded()
    data = _CACHE.get("data") or {}
    for path in _PATHS:
        payload = data.get(str(path)) or {}
        value = _get_nested(payload, keys)
        if value is not None:
            return value
    return default
