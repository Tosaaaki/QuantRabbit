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
    "macro": 960,
    "micro": 600,
    "scalp": _SCALP_LOSS_COOLDOWN,
}

POCKET_ENTRY_MIN_INTERVAL = {
    "macro": 180,
    "micro": 120,
    "scalp": 60,
}

DEFAULT_COOLDOWN_SECONDS = 180
RANGE_COOLDOWN_SECONDS = 420


def cooldown_for_pocket(pocket: str, *, range_mode: bool) -> int:
    base = POCKET_EXIT_COOLDOWNS.get(pocket, DEFAULT_COOLDOWN_SECONDS)
    if range_mode:
        base = max(base, RANGE_COOLDOWN_SECONDS)
    return base
