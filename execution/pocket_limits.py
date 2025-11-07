"""Pocket-level limit constants shared across executors."""

from __future__ import annotations

POCKET_EXIT_COOLDOWNS = {
    "macro": 540,
    "micro": 240,
    "scalp": 180,
}

POCKET_LOSS_COOLDOWNS = {
    "macro": 960,
    "micro": 600,
    "scalp": 360,
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

