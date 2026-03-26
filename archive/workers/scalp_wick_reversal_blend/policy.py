from __future__ import annotations

from typing import Mapping


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _projection_headwind(projection_score: float) -> float:
    return _clamp((-projection_score - 0.02) / 0.28)


def _fallback_wick_blend_trade_quality(
    *,
    side: str,
    thesis: Mapping[str, object],
    atr_pips: float,
    projection_score: float,
) -> float:
    rsi = _to_float(thesis.get("rsi"), 50.0)
    adx = _to_float(thesis.get("adx"), 18.0)
    range_score = _to_float(thesis.get("range_score"), 0.0)
    atr_norm = max(0.8, _to_float(thesis.get("atr_pips"), atr_pips))
    direction_rsi = (rsi - 50.0) if side == "short" else (50.0 - rsi)
    stretch = _clamp((direction_rsi - 4.5) / max(6.0, 10.0 - min(adx, 28.0) * 0.08))
    regime = _clamp(range_score / 0.50)
    adx_reversion = _clamp((28.0 - adx) / 14.0)
    projection = _clamp((projection_score + 0.10) / 0.45)
    vwap_gap = abs(_to_float(thesis.get("vwap_gap"), 0.0))
    vwap_ext = _clamp(vwap_gap / max(4.0, atr_norm * 2.6))
    wick_missing_penalty = 0.18 if not isinstance(thesis.get("wick"), Mapping) else 0.0
    quality = (
        0.10
        + stretch * 0.28
        + regime * 0.18
        + adx_reversion * 0.18
        + vwap_ext * 0.14
        + projection * 0.10
        - wick_missing_penalty
    )
    return _clamp(quality)


def _flow_guard_regime(continuation_pressure: float) -> str:
    return "continuation_headwind" if continuation_pressure >= 0.60 else "range_fade"


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
    range_reason: str | None = None,
    macd_hist_pips: float | None = None,
    di_gap: float | None = None,
) -> dict[str, object]:
    direction_rsi = (rsi - 50.0) if side == "short" else (50.0 - rsi)
    atr_norm = max(0.8, atr_pips)
    projection_headwind = _projection_headwind(projection_score)
    range_reason_key = str(range_reason or "").strip().lower()

    stretch = _clamp((direction_rsi - 2.0) / max(6.0, 9.0 - min(adx, 30.0) * 0.10))
    wick = _clamp((wick_ratio - 0.35) / 0.30)
    tick = _clamp((tick_strength - 0.18) / 0.30)
    follow = _clamp(follow_pips / max(0.30, atr_norm * 0.18))
    retrace = _clamp(retrace_from_extreme_pips / max(0.35, atr_norm * 0.20))
    projection = _clamp((projection_score + 0.15) / 0.55)
    regime = _clamp(range_score / 0.45)
    signed_macd = (
        (-_to_float(macd_hist_pips, 0.0))
        if side == "long"
        else _to_float(macd_hist_pips, 0.0)
    )
    macd_headwind = _clamp((signed_macd - 0.05) / 0.35)
    signed_di_gap = (
        (-_to_float(di_gap, 0.0)) if side == "long" else _to_float(di_gap, 0.0)
    )
    di_headwind = _clamp((signed_di_gap - 4.0) / 6.0)
    trend_headwind = _clamp(macd_headwind * 0.70 + di_headwind * 0.30)

    quality = (
        stretch * 0.24
        + wick * 0.18
        + tick * 0.18
        + follow * 0.12
        + retrace * 0.14
        + projection * 0.10
        + regime * 0.04
        - projection_headwind * 0.14
        - trend_headwind * 0.08
    )
    neutral_rsi = direction_rsi < max(5.0, min(8.0, 3.2 + adx * 0.11))
    structure_strong = wick >= 0.65 and tick >= 0.45 and retrace >= 0.55
    threshold = (
        (0.52 if neutral_rsi else 0.46)
        + projection_headwind * (0.04 if neutral_rsi else 0.03)
        + trend_headwind * (0.05 if neutral_rsi else 0.03)
    )
    weak_countertrend_lane = (
        side == "long"
        and range_reason_key == "volatility_compression"
        and rsi < 50.0
        and projection_score <= 0.15
        and signed_macd > 0.05
    )
    allow = (
        quality >= threshold
        and (projection >= 0.12 or structure_strong)
        and not weak_countertrend_lane
    )

    return {
        "allow": allow,
        "quality": round(_clamp(quality), 3),
        "threshold": round(threshold, 3),
        "neutral_rsi": neutral_rsi,
        "structure_strong": structure_strong,
        "weak_countertrend_lane": weak_countertrend_lane,
        "components": {
            "stretch": round(stretch, 3),
            "wick": round(wick, 3),
            "tick": round(tick, 3),
            "follow": round(follow, 3),
            "retrace": round(retrace, 3),
            "projection": round(projection, 3),
            "projection_headwind": round(projection_headwind, 3),
            "trend_headwind": round(trend_headwind, 3),
            "macd_headwind": round(macd_headwind, 3),
            "di_headwind": round(di_headwind, 3),
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
    projection = thesis.get("projection")
    projection_score = _to_float(
        projection.get("score") if isinstance(projection, Mapping) else None,
        0.0,
    )
    projection_headwind = _projection_headwind(projection_score)
    quality = _clamp(_to_float(thesis.get("wick_blend_quality"), -1.0), 0.0, 1.0)
    quality_rebuilt = False
    if quality <= 0.0:
        wick = thesis.get("wick")
        if isinstance(wick, Mapping):
            rebuilt = wick_blend_entry_quality(
                side=side,
                rsi=_to_float(thesis.get("rsi"), 50.0),
                adx=_to_float(thesis.get("adx"), 0.0),
                atr_pips=max(0.8, _to_float(thesis.get("atr_pips"), atr_pips)),
                range_score=_to_float(thesis.get("range_score"), 0.0),
                wick_ratio=_to_float(wick.get("ratio"), 0.0),
                tick_strength=_to_float(wick.get("tick_strength"), 0.0),
                follow_pips=_to_float(wick.get("follow_pips"), 0.0),
                retrace_from_extreme_pips=_to_float(
                    wick.get("retrace_from_extreme_pips"), 0.0
                ),
                projection_score=projection_score,
                range_reason=(
                    str(thesis.get("range_reason"))
                    if thesis.get("range_reason") is not None
                    else None
                ),
                macd_hist_pips=_to_float(thesis.get("macd_hist_pips"), None),
                di_gap=(
                    _to_float(thesis.get("plus_di"), 0.0)
                    - _to_float(thesis.get("minus_di"), 0.0)
                    if thesis.get("plus_di") is not None
                    and thesis.get("minus_di") is not None
                    else None
                ),
            )
            quality = _clamp(_to_float(rebuilt.get("quality"), 0.0))
            quality_rebuilt = True
        else:
            quality = _fallback_wick_blend_trade_quality(
                side=side,
                thesis=thesis,
                atr_pips=atr_pips,
                projection_score=projection_score,
            )
            quality_rebuilt = True
    atr_norm = max(0.8, atr_pips)

    adjusted_profit_take = profit_take
    adjusted_trail_start = trail_start
    adjusted_trail_backoff = trail_backoff
    adjusted_lock_buffer = lock_buffer
    adjusted_loss_cut_hard = loss_cut_hard_pips
    adjusted_loss_cut_max_hold_sec = loss_cut_max_hold_sec
    continuation_pressure = _to_float(thesis.get("continuation_pressure"), 0.0)
    if continuation_pressure <= 0.0:
        continuation_pressure = projection_headwind
    headwind_regime = (
        _flow_guard_regime(continuation_pressure) == "continuation_headwind"
    )

    if entry_tp > 0.0:
        profit_factor = _clamp(
            0.60 + quality * 0.20 - projection_headwind * 0.18, 0.34, 0.78
        )
        adjusted_profit_take = max(0.5, entry_tp * profit_factor)
        trail_anchor = adjusted_profit_take * (
            0.78 + (1.0 - quality) * 0.04 - projection_headwind * 0.08
        )
        trail_floor_mult = (
            1.0
            if quality >= 0.58 and not headwind_regime and projection_headwind <= 0.10
            else 0.82
        )
        adjusted_trail_start = max(
            0.5,
            min(
                adjusted_profit_take - 0.10,
                max(
                    trail_start
                    * max(0.72, trail_floor_mult - projection_headwind * 0.12),
                    trail_anchor,
                ),
            ),
        )
    if entry_sl > 0.0:
        adjusted_trail_backoff = min(
            trail_backoff,
            max(
                0.18,
                min(
                    entry_sl * (0.26 - projection_headwind * 0.04 + quality * 0.03),
                    max(
                        0.22, adjusted_profit_take * (0.24 - projection_headwind * 0.05)
                    ),
                ),
            ),
        )
        adjusted_lock_buffer = min(
            max(lock_buffer * (0.88 - projection_headwind * 0.10), 0.10),
            min(
                max(0.16, adjusted_profit_take * (0.22 - projection_headwind * 0.05)),
                max(0.22, entry_sl * (0.36 - projection_headwind * 0.05)),
            ),
        )
        dynamic_hard_cut = max(
            entry_sl * (0.95 + (1.0 - quality) * 0.22 - projection_headwind * 0.16),
            atr_norm * (0.82 - projection_headwind * 0.08),
            1.4,
        )
        adjusted_loss_cut_hard = (
            min(loss_cut_hard_pips, dynamic_hard_cut)
            if loss_cut_hard_pips > 0.0
            else dynamic_hard_cut
        )
    rebuilt_penalty_sec = 60.0 if quality_rebuilt else 0.0
    dynamic_hold_sec = max(
        120.0,
        min(
            360.0,
            120.0
            + entry_tp * 35.0
            + quality * 60.0
            - projection_headwind * 80.0
            - rebuilt_penalty_sec,
        ),
    )
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
