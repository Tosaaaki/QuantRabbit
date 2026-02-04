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


def _env_flag_optional(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
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


def entry_sl_enabled_for_pocket(pocket: str | None) -> bool:
    """Return whether stopLossOnFill should be attached for the given pocket."""

    base_enabled = not STOP_LOSS_DISABLED
    if not pocket:
        return base_enabled
    pocket_key = str(pocket).strip().upper()
    if not pocket_key:
        return base_enabled

    # Explicit per-pocket overrides (allows gradual rollout while global stays disabled)
    disable_override = _env_flag_optional(f"ORDER_DISABLE_STOP_LOSS_{pocket_key}")
    if disable_override is None:
        disable_override = _env_flag_optional(f"DISABLE_STOP_LOSS_{pocket_key}")
    if disable_override is not None:
        return not disable_override

    enable_override = _env_flag_optional(f"ORDER_ENABLE_STOP_LOSS_{pocket_key}")
    if enable_override is None:
        enable_override = _env_flag_optional(f"ENABLE_STOP_LOSS_{pocket_key}")
    if enable_override is not None:
        return enable_override

    return base_enabled


def stop_loss_disabled_for_pocket(pocket: str | None) -> bool:
    return not entry_sl_enabled_for_pocket(pocket)


def trailing_sl_allowed() -> bool:
    return TRAILING_SL_ALLOWED
