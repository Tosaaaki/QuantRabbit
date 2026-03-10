from __future__ import annotations

from typing import Mapping


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def wick_blend_entry_quality(
    *,
    side: str,
    rsi: float,
    adx: float,
    atr_pips: float,
    range_score: float,
    wick_ratio: float,
    tick_strength: float,
    follow_pips: float,
    retrace_from_extreme_pips: float,
    projection_score: float,
) -> dict[str, object]:
    direction_rsi = (rsi - 50.0) if side == "short" else (50.0 - rsi)
    atr_norm = max(0.8, atr_pips)

    stretch = _clamp((direction_rsi - 2.0) / max(6.0, 9.0 - min(adx, 30.0) * 0.10))
    wick = _clamp((wick_ratio - 0.35) / 0.30)
    tick = _clamp((tick_strength - 0.18) / 0.30)
    follow = _clamp(follow_pips / max(0.30, atr_norm * 0.18))
    retrace = _clamp(retrace_from_extreme_pips / max(0.35, atr_norm * 0.20))
    projection = _clamp((projection_score + 0.15) / 0.55)
    regime = _clamp(range_score / 0.45)

    quality = (
        stretch * 0.24
        + wick * 0.18
        + tick * 0.18
        + follow * 0.12
        + retrace * 0.14
        + projection * 0.10
        + regime * 0.04
    )
    neutral_rsi = direction_rsi < max(5.0, min(8.0, 3.2 + adx * 0.11))
    structure_strong = wick >= 0.65 and tick >= 0.45 and retrace >= 0.55
    threshold = 0.52 if neutral_rsi else 0.46
    allow = quality >= threshold and (projection >= 0.12 or structure_strong)

    return {
        "allow": allow,
        "quality": round(_clamp(quality), 3),
        "threshold": round(threshold, 3),
        "neutral_rsi": neutral_rsi,
        "structure_strong": structure_strong,
        "components": {
            "stretch": round(stretch, 3),
            "wick": round(wick, 3),
            "tick": round(tick, 3),
            "follow": round(follow, 3),
            "retrace": round(retrace, 3),
            "projection": round(projection, 3),
            "range": round(regime, 3),
        },
    }


def wick_blend_exit_adjustments(
    *,
    side: str,
    thesis: Mapping[str, object],
    atr_pips: float,
    profit_take: float,
    trail_start: float,
    trail_backoff: float,
    lock_buffer: float,
    loss_cut_hard_pips: float,
    loss_cut_max_hold_sec: float,
) -> dict[str, float]:
    entry_sl = max(0.0, _to_float(thesis.get("sl_pips"), 0.0))
    entry_tp = max(0.0, _to_float(thesis.get("tp_pips"), 0.0))
    quality = _clamp(_to_float(thesis.get("wick_blend_quality"), -1.0), 0.0, 1.0)
    if quality <= 0.0:
        wick = thesis.get("wick")
        projection = thesis.get("projection")
        if isinstance(wick, Mapping):
            projection_score = _to_float(
                projection.get("score") if isinstance(projection, Mapping) else None,
                0.0,
            )
            rebuilt = wick_blend_entry_quality(
                side=side,
                rsi=_to_float(thesis.get("rsi"), 50.0),
                adx=_to_float(thesis.get("adx"), 0.0),
                atr_pips=max(0.8, _to_float(thesis.get("atr_pips"), atr_pips)),
                range_score=_to_float(thesis.get("range_score"), 0.0),
                wick_ratio=_to_float(wick.get("ratio"), 0.0),
                tick_strength=_to_float(wick.get("tick_strength"), 0.0),
                follow_pips=_to_float(wick.get("follow_pips"), 0.0),
                retrace_from_extreme_pips=_to_float(wick.get("retrace_from_extreme_pips"), 0.0),
                projection_score=projection_score,
            )
            quality = _clamp(_to_float(rebuilt.get("quality"), 0.0))
    atr_norm = max(0.8, atr_pips)

    adjusted_profit_take = profit_take
    adjusted_trail_start = trail_start
    adjusted_trail_backoff = trail_backoff
    adjusted_lock_buffer = lock_buffer
    adjusted_loss_cut_hard = loss_cut_hard_pips
    adjusted_loss_cut_max_hold_sec = loss_cut_max_hold_sec

    if entry_tp > 0.0:
        profit_from_tp = entry_tp * (0.56 + quality * 0.20)
        adjusted_profit_take = max(profit_take, profit_from_tp)
        adjusted_trail_start = max(
            trail_start,
            min(adjusted_profit_take - 0.10, adjusted_profit_take * (0.72 + (1.0 - quality) * 0.08)),
        )
    if entry_sl > 0.0:
        adjusted_trail_backoff = min(
            trail_backoff,
            max(0.22, min(entry_sl * 0.30, max(0.24, adjusted_profit_take * 0.26))),
        )
        adjusted_lock_buffer = max(
            lock_buffer,
            min(max(0.18, adjusted_profit_take * 0.24), max(0.25, entry_sl * 0.38)),
        )
        dynamic_hard_cut = max(entry_sl * (1.05 + (1.0 - quality) * 0.20), atr_norm * 0.95, 2.0)
        adjusted_loss_cut_hard = (
            min(loss_cut_hard_pips, dynamic_hard_cut) if loss_cut_hard_pips > 0.0 else dynamic_hard_cut
        )
    dynamic_hold_sec = max(120.0, min(360.0, 150.0 + entry_tp * 55.0 + quality * 60.0))
    adjusted_loss_cut_max_hold_sec = (
        min(loss_cut_max_hold_sec, dynamic_hold_sec)
        if loss_cut_max_hold_sec > 0.0
        else dynamic_hold_sec
    )

    return {
        "profit_take": round(max(0.5, adjusted_profit_take), 3),
        "trail_start": round(max(0.5, adjusted_trail_start), 3),
        "trail_backoff": round(max(0.12, adjusted_trail_backoff), 3),
        "lock_buffer": round(max(0.10, adjusted_lock_buffer), 3),
        "loss_cut_hard_pips": round(max(0.10, adjusted_loss_cut_hard), 3),
        "loss_cut_max_hold_sec": round(max(30.0, adjusted_loss_cut_max_hold_sec), 3),
    }
