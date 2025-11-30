"""Centralized stop-loss policy flags.

Defaults:
- Entry stop-loss is disabled (matches prior ORDER_DISABLE_STOP_LOSS default).
- Trailing/BE updates are allowed unless explicitly turned off.
"""

from __future__ import annotations

import os

_FALSEY = {"", "0", "false", "no", "off"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSEY


# Keep legacy env names; default to disabled to preserve existing behavior.
STOP_LOSS_DISABLED: bool = _env_flag("ORDER_DISABLE_STOP_LOSS", True) or _env_flag(
    "DISABLE_STOP_LOSS", True
)
# Trailing/BE SL updates can be toggled separately; default allow.
TRAILING_SL_ALLOWED: bool = _env_flag("ALLOW_TRAILING_STOP_LOSS", True)


def stop_loss_disabled() -> bool:
    return STOP_LOSS_DISABLED


def entry_sl_enabled() -> bool:
    return not STOP_LOSS_DISABLED


def trailing_sl_allowed() -> bool:
    return TRAILING_SL_ALLOWED
