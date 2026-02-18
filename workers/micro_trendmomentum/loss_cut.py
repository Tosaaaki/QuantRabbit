from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "off"}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    return bool(default)


@dataclass(frozen=True, slots=True)
class LossCutParams:
    enabled: bool
    require_sl: bool
    soft_pips: float
    hard_pips: float
    max_hold_sec: float
    cooldown_sec: float
    reason_soft: str
    reason_hard: str
    reason_time: str


def resolve_loss_cut(exit_profile: dict, *, sl_pips: Optional[float] = None) -> LossCutParams:
    """Resolve per-trade loss-cut parameters from the strategy exit_profile.

    Supports optional derived thresholds:
    - loss_cut_soft_sl_mult / loss_cut_hard_sl_mult: multiply sl_pips/hard_stop_pips.
    - If *_pips are <= 0 and a *_sl_mult is set, derived thresholds apply.
    """
    if not isinstance(exit_profile, dict):
        exit_profile = {}

    enabled = _coerce_bool(exit_profile.get("loss_cut_enabled"), False)
    require_sl = _coerce_bool(exit_profile.get("loss_cut_require_sl"), True)

    soft_pips = max(0.0, float(_safe_float(exit_profile.get("loss_cut_soft_pips")) or 0.0))
    hard_pips = float(_safe_float(exit_profile.get("loss_cut_hard_pips")) or 0.0)
    max_hold_sec = max(0.0, float(_safe_float(exit_profile.get("loss_cut_max_hold_sec")) or 0.0))
    cooldown_sec = max(0.0, float(_safe_float(exit_profile.get("loss_cut_cooldown_sec")) or 0.0))

    reason_soft = str(exit_profile.get("loss_cut_reason_soft") or "hard_stop").strip() or "hard_stop"
    reason_hard = str(exit_profile.get("loss_cut_reason_hard") or "max_adverse").strip() or "max_adverse"
    reason_time = str(exit_profile.get("loss_cut_reason_time") or "time_stop").strip() or "time_stop"

    hint = float(sl_pips) if sl_pips is not None and sl_pips > 0 else None
    if hint is not None:
        soft_mult = float(_safe_float(exit_profile.get("loss_cut_soft_sl_mult")) or 0.0)
        hard_mult = float(_safe_float(exit_profile.get("loss_cut_hard_sl_mult")) or 0.0)
        if soft_pips <= 0.0 and soft_mult > 0.0:
            soft_pips = max(0.0, hint * soft_mult)
        if hard_pips <= 0.0 and hard_mult > 0.0:
            hard_pips = max(0.0, hint * hard_mult)

    hard_pips = max(0.0, hard_pips)
    if hard_pips > 0.0:
        hard_pips = max(hard_pips, soft_pips)

    # A zero cooldown is allowed, but default to a small backoff when enabled to avoid tight loops.
    if enabled and cooldown_sec <= 0.0:
        cooldown_sec = 6.0

    return LossCutParams(
        enabled=enabled,
        require_sl=require_sl,
        soft_pips=soft_pips,
        hard_pips=hard_pips,
        max_hold_sec=max_hold_sec,
        cooldown_sec=cooldown_sec,
        reason_soft=reason_soft,
        reason_hard=reason_hard,
        reason_time=reason_time,
    )


def pick_loss_cut_reason(
    *,
    pnl_pips: float,
    hold_sec: float,
    params: LossCutParams,
    has_stop_loss: bool,
) -> Optional[str]:
    """Return an exit reason when a loss-cut threshold is hit, else None."""
    if not params.enabled:
        return None
    if params.require_sl and not has_stop_loss:
        return None
    if pnl_pips > 0:
        return None
    adverse = abs(float(pnl_pips))
    if params.max_hold_sec > 0.0 and hold_sec >= params.max_hold_sec:
        return params.reason_time
    if params.hard_pips > 0.0 and adverse >= params.hard_pips:
        return params.reason_hard
    if params.soft_pips > 0.0 and adverse >= params.soft_pips:
        return params.reason_soft
    return None

