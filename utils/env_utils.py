from __future__ import annotations

import os
from typing import Optional

_FALSEY = {"", "0", "false", "no", "off"}


def env_get(name: str, default: Optional[str] = None, *, prefix: Optional[str] = None) -> Optional[str]:
    """
    Environment lookup with optional per-worker prefix.

    Precedence (when prefix is provided):
    1) {PREFIX}_UNIT_{NAME}  (per-systemd-unit / single-strategy override)
    2) {PREFIX}_{NAME}       (per-worker override)
    3) {NAME}                (global fallback)
    """
    if prefix:
        p = str(prefix).strip()
        if p:
            unit_key = f"{p}_UNIT_{name}"
            if unit_key in os.environ:
                return os.environ[unit_key]
            pref_key = f"{p}_{name}"
            if pref_key in os.environ:
                return os.environ[pref_key]
    return os.getenv(name, default)


def env_bool(name: str, default: bool, *, prefix: Optional[str] = None) -> bool:
    raw = env_get(name, None, prefix=prefix)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in _FALSEY


def env_float(name: str, default: float, *, prefix: Optional[str] = None) -> float:
    raw = env_get(name, None, prefix=prefix)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def env_int(name: str, default: int, *, prefix: Optional[str] = None) -> int:
    raw = env_get(name, None, prefix=prefix)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)

