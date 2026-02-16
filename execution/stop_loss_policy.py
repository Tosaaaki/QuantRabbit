"""Centralized stop-loss policy flags.

In v1 mode, stop-loss attach behavior is controlled only by
`ORDER_FIXED_SL_MODE`:
- `1` : always attach stopLossOnFill
- `0` : never attach stopLossOnFill
- unset: treat as OFF for safety
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

# Trailing/BE SL updates can be toggled separately; default allow.
TRAILING_SL_ALLOWED: bool = _env_flag("ALLOW_TRAILING_STOP_LOSS", True)
_FIXED_SL_MODE: bool | None = _env_flag_optional("ORDER_FIXED_SL_MODE")


def _resolved_fixed_mode() -> bool:
    if _FIXED_SL_MODE is None:
        return False
    return _FIXED_SL_MODE


def stop_loss_disabled() -> bool:
    return not _resolved_fixed_mode()


def entry_sl_enabled() -> bool:
    return _resolved_fixed_mode()


def entry_sl_enabled_for_pocket(pocket: str | None) -> bool:
    """Return whether stopLossOnFill should be attached for the given pocket."""
    return _resolved_fixed_mode()


def stop_loss_disabled_for_pocket(pocket: str | None) -> bool:
    return not entry_sl_enabled_for_pocket(pocket)


def trailing_sl_allowed() -> bool:
    return TRAILING_SL_ALLOWED


def fixed_sl_mode() -> bool | None:
    """Return fixed-mode global override.

    - True: always attach broker SL.
    - False: never attach broker SL.
    - None: defaults to False for backward-compat behavior in legacy deploy scripts.
    """
    return _FIXED_SL_MODE
