"""Pocket-level limit constants shared across executors."""

from __future__ import annotations

import os


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


POCKET_EXIT_COOLDOWNS = {
    "macro": 540,
    "micro": 240,
    "scalp": 180,
}

_SCALP_LOSS_COOLDOWN = max(60, _int_env("SCALP_LOSS_COOLDOWN_SEC", 150))

POCKET_LOSS_COOLDOWNS = {
    "macro": _int_env("MACRO_LOSS_COOLDOWN_SEC", 720),
    "micro": _int_env("MICRO_LOSS_COOLDOWN_SEC", 60),
    "scalp": _SCALP_LOSS_COOLDOWN,
}

POCKET_ENTRY_MIN_INTERVAL = {
    "macro": _int_env("MACRO_ENTRY_MIN_INTERVAL_SEC", 90),
    "micro": _int_env("MICRO_ENTRY_MIN_INTERVAL_SEC", 20),
    "scalp": _int_env("SCALP_ENTRY_MIN_INTERVAL_SEC", 45),
}

DEFAULT_COOLDOWN_SECONDS = 180
RANGE_COOLDOWN_SECONDS = 420


def cooldown_for_pocket(pocket: str, *, range_mode: bool) -> int:
    base = POCKET_EXIT_COOLDOWNS.get(pocket, DEFAULT_COOLDOWN_SECONDS)
    if range_mode:
        base = max(base, RANGE_COOLDOWN_SECONDS)
    return base
