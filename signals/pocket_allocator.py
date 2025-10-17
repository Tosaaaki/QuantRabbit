"""
signals.pocket_allocator
~~~~~~~~~~~~~~~~~~~~~~~~

Lot の総量と macro 重みから、pocket ごとの配分を返す。
"""

from __future__ import annotations

import os
import time
from typing import Dict

from analysis.scalp_config import ensure_overrides, load_overrides

_MIN_MICRO_SHARE = float(os.getenv("MIN_MICRO_SHARE", "0.15"))
_MIN_MICRO_ABS = float(os.getenv("MIN_MICRO_ABS", "0.007"))
_SCALP_MIN_ABS = max(0.0, float(os.getenv("SCALP_MIN_ABS", "0.0")))
_SCALP_MAX_SHARE = max(0.0, min(1.0, float(os.getenv("SCALP_MAX_SHARE", "0.03"))))

_OVERRIDES = ensure_overrides()
_SCALP_SHARE_REFRESH_SEC = int(os.getenv("SCALP_SHARE_REFRESH_SEC", "600"))
def _initial_scalp_share() -> float:
    raw = os.getenv("SCALP_SHARE")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    try:
        override_share = float(_OVERRIDES.get("share", 0.05))
    except (TypeError, ValueError):
        override_share = 0.05
    return override_share


_SCALP_SHARE_CACHE = {
    "value": max(0.0, min(_SCALP_MAX_SHARE, _initial_scalp_share())),
    "ts": 0.0,
}


def _current_scalp_share() -> float:
    now = time.time()
    if now - _SCALP_SHARE_CACHE["ts"] >= _SCALP_SHARE_REFRESH_SEC:
        overrides = load_overrides() or {}
        raw_share = overrides.get("share")
        try:
            if raw_share is not None:
                share = max(0.0, min(_SCALP_MAX_SHARE, float(raw_share)))
                _SCALP_SHARE_CACHE.update({"value": share, "ts": now})
            else:
                _SCALP_SHARE_CACHE["ts"] = now
        except (TypeError, ValueError):
            _SCALP_SHARE_CACHE["ts"] = now
    return _SCALP_SHARE_CACHE["value"]


def _alloc_macro_micro(remaining: float, weight_macro: float | None) -> Dict[str, float]:
    if remaining <= 0:
        return {"macro": 0.0, "micro": 0.0}

    if weight_macro is None:
        w = 0.5
    else:
        try:
            w = max(0.0, min(1.0, float(weight_macro)))
        except Exception:
            w = 0.5

    macro = remaining * w
    micro = max(remaining - macro, 0.0)

    if remaining > 0:
        share = micro / remaining if remaining else 0.0
        if share < _MIN_MICRO_SHARE:
            micro = remaining * _MIN_MICRO_SHARE
            macro = max(remaining - micro, 0.0)

    macro = round(max(macro, 0.0), 3)
    micro = round(max(micro, 0.0), 3)

    if remaining > 0 and micro == 0.0 and remaining >= _MIN_MICRO_ABS:
        micro = round(min(remaining, _MIN_MICRO_ABS), 3)
        macro = round(max(remaining - micro, 0.0), 3)

    return {"macro": macro, "micro": micro}


def alloc(
    total_lot: float,
    weight_macro: float | None,
    weight_scalp: float | None = None,
) -> Dict[str, float]:
    """Allocate lot across macro/micro/scalp pockets.

    weight_macro / weight_scalp は全体比率で指定する。
    """
    total_lot = max(total_lot, 0.0)
    if total_lot == 0.0:
        return {"macro": 0.0, "micro": 0.0, "scalp": 0.0}

    if weight_scalp is not None:
        try:
            scalp_share_value = max(0.0, min(1.0, float(weight_scalp)))
        except (TypeError, ValueError):
            scalp_share_value = 0.0
        scalp = round(min(total_lot, total_lot * scalp_share_value), 3)
    else:
        scalp = 0.0
        scalp_share_value = max(0.0, min(_SCALP_MAX_SHARE, _current_scalp_share()))
        if scalp_share_value > 0:
            scalp = total_lot * scalp_share_value
            if total_lot >= _SCALP_MIN_ABS:
                scalp = max(scalp, _SCALP_MIN_ABS)
            scalp = min(scalp, total_lot)
        scalp = round(max(scalp, 0.0), 3)

    remaining = max(total_lot - scalp, 0.0)
    macro_weight = weight_macro
    if weight_scalp is not None and remaining > 0 and macro_weight is not None:
        try:
            macro_weight = float(macro_weight)
        except (TypeError, ValueError):
            macro_weight = 0.5
        base = max(1e-6, 1.0 - scalp_share_value)
        macro_weight = max(0.0, min(1.0, macro_weight / base))
    shares = _alloc_macro_micro(remaining, macro_weight)

    macro = shares["macro"]
    micro = shares["micro"]

    # Ensure total matches after rounding adjustments
    used = macro + micro + scalp
    diff = round(total_lot - used, 3)
    if abs(diff) >= 0.001:
        macro = round(max(macro + diff, 0.0), 3)
        used = macro + micro + scalp
        if used > total_lot:
            macro = round(max(macro - (used - total_lot), 0.0), 3)

    return {"macro": macro, "micro": micro, "scalp": scalp}
