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
_PATHS = [
    Path(os.getenv("TUNING_OVERLAY_PATH", _ROOT / "config" / "tuning_overlay.yaml")),
    Path(os.getenv("TUNING_OVERRIDES_PATH", _ROOT / "config" / "tuning_overrides.yaml")),
    Path(os.getenv("TUNING_PRESETS_PATH", _ROOT / "config" / "tuning_presets.yaml")),
]

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

