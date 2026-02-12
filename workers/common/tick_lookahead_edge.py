"""Short-horizon tick lookahead gate for fast scalp entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from typing import Sequence


@dataclass(frozen=True)
class TickLookaheadDecision:
    allow_entry: bool
    units_mult: float
    reason: str
    pred_move_pips: float
    cost_pips: float
    edge_pips: float
    slippage_est_pips: float
    range_pips: float
    momentum_aligned_pips: float
    direction_bias_aligned: float


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _recent_range_pips(
    ticks: Sequence[dict],
    *,
    pip_value: float,
    latest_epoch: float,
    window_sec: float,
) -> float:
    if pip_value <= 0.0:
        return 0.0
    cutoff = latest_epoch - max(0.5, window_sec)
    mids: list[float] = []
    fallback_mid = 0.0
    for row in ticks:
        epoch = _safe_float(row.get("epoch"), 0.0)
        if epoch < cutoff:
            continue
        bid = _safe_float(row.get("bid"), 0.0)
        ask = _safe_float(row.get("ask"), 0.0)
        mid = _safe_float(row.get("mid"), 0.0)
        if mid <= 0.0 and bid > 0.0 and ask > 0.0:
            mid = (bid + ask) * 0.5
        if mid <= 0.0:
            mid = fallback_mid
        if mid > 0.0:
            mids.append(mid)
            fallback_mid = mid
    if len(mids) < 2:
        return 0.0
    return max(0.0, (max(mids) - min(mids)) / pip_value)


def decide_tick_lookahead_edge(
    *,
    ticks: Sequence[dict],
    side: str,
    spread_pips: float,
    momentum_pips: float,
    trigger_pips: float,
    imbalance: float,
    tick_rate: float,
    signal_span_sec: float,
    pip_value: float,
    horizon_sec: float,
    edge_min_pips: float,
    edge_ref_pips: float,
    units_min_mult: float,
    units_max_mult: float,
    slippage_base_pips: float,
    slippage_spread_mult: float,
    slippage_range_mult: float,
    latency_penalty_pips: float,
    safety_margin_pips: float,
    momentum_decay: float,
    momentum_weight: float,
    flow_weight: float,
    rate_weight: float,
    bias_weight: float,
    trigger_weight: float,
    counter_penalty: float,
    direction_bias_score: Optional[float] = None,
    allow_thin_edge: bool = True,
    fail_open: bool = True,
) -> TickLookaheadDecision:
    side_key = str(side or "").strip().lower()
    if side_key not in {"long", "short"}:
        return TickLookaheadDecision(
            allow_entry=bool(fail_open),
            units_mult=1.0,
            reason="invalid_side",
            pred_move_pips=0.0,
            cost_pips=0.0,
            edge_pips=0.0,
            slippage_est_pips=0.0,
            range_pips=0.0,
            momentum_aligned_pips=0.0,
            direction_bias_aligned=0.0,
        )

    latest_epoch = 0.0
    if ticks:
        latest_epoch = _safe_float(ticks[-1].get("epoch"), 0.0)
    if latest_epoch <= 0.0:
        return TickLookaheadDecision(
            allow_entry=bool(fail_open),
            units_mult=1.0,
            reason="missing_ticks",
            pred_move_pips=0.0,
            cost_pips=max(0.0, _safe_float(spread_pips)),
            edge_pips=0.0,
            slippage_est_pips=0.0,
            range_pips=0.0,
            momentum_aligned_pips=0.0,
            direction_bias_aligned=0.0,
        )

    side_sign = 1.0 if side_key == "long" else -1.0
    momentum_aligned = side_sign * _safe_float(momentum_pips, 0.0)
    trigger = max(0.05, abs(_safe_float(trigger_pips, 0.0)))
    signal_span = max(0.2, _safe_float(signal_span_sec, 0.2))
    horizon = max(0.5, _safe_float(horizon_sec, 1.0))
    horizon_scale = _clamp(horizon / signal_span, 0.25, 2.8)

    range_pips = _recent_range_pips(
        ticks,
        pip_value=max(1e-9, float(pip_value)),
        latest_epoch=latest_epoch,
        window_sec=max(1.0, horizon * 1.4),
    )
    range_cap = max(trigger * 3.2, max(0.35, range_pips * 1.6))

    mom_proj = momentum_aligned * max(0.0, momentum_decay) * horizon_scale
    mom_proj = _clamp(mom_proj, -range_cap, range_cap)

    flow_pressure = _clamp((abs(_safe_float(imbalance, 0.5)) - 0.5) * 2.0, 0.0, 1.0)
    flow_term = max(0.0, flow_weight) * flow_pressure * max(0.25, trigger * 0.9)

    rate_norm = _clamp(_safe_float(tick_rate, 0.0) / 5.0, 0.0, 2.0)
    rate_term = max(0.0, rate_weight) * (rate_norm - 0.65) * max(0.22, trigger * 0.6)

    bias_raw = _safe_float(direction_bias_score, 0.0)
    bias_aligned = side_sign * bias_raw
    bias_term = max(0.0, bias_weight) * bias_aligned * max(0.2, trigger * 0.7)

    trigger_excess = max(0.0, momentum_aligned - trigger)
    trigger_term = max(0.0, trigger_weight) * trigger_excess

    oppose_penalty = 0.0
    if momentum_aligned < 0.0:
        oppose_penalty = abs(momentum_aligned) * max(0.0, counter_penalty)

    pred_move_pips = (
        max(0.0, momentum_weight) * mom_proj
        + flow_term
        + rate_term
        + bias_term
        + trigger_term
        - oppose_penalty
    )

    pred_cap = max(0.25, range_cap * 1.3)
    pred_move_pips = _clamp(pred_move_pips, -pred_cap, pred_cap)

    spread = max(0.0, _safe_float(spread_pips, 0.0))
    slippage_est = (
        max(0.0, _safe_float(slippage_base_pips, 0.0))
        + max(0.0, _safe_float(slippage_spread_mult, 0.0)) * spread
        + max(0.0, _safe_float(slippage_range_mult, 0.0)) * max(0.0, range_pips)
    )
    cost_pips = (
        spread
        + slippage_est
        + max(0.0, _safe_float(latency_penalty_pips, 0.0))
        + max(0.0, _safe_float(safety_margin_pips, 0.0))
    )
    edge_pips = pred_move_pips - cost_pips

    min_mult = _clamp(_safe_float(units_min_mult, 0.5), 0.05, 1.0)
    max_mult = max(1.0, _safe_float(units_max_mult, 1.0))
    edge_min = max(0.0, _safe_float(edge_min_pips, 0.0))
    edge_ref = max(0.05, _safe_float(edge_ref_pips, 0.5))

    if edge_pips <= 0.0:
        return TickLookaheadDecision(
            allow_entry=False,
            units_mult=0.0,
            reason="edge_negative_block",
            pred_move_pips=round(pred_move_pips, 6),
            cost_pips=round(cost_pips, 6),
            edge_pips=round(edge_pips, 6),
            slippage_est_pips=round(slippage_est, 6),
            range_pips=round(range_pips, 6),
            momentum_aligned_pips=round(momentum_aligned, 6),
            direction_bias_aligned=round(bias_aligned, 6),
        )

    if edge_pips < edge_min:
        if not allow_thin_edge:
            return TickLookaheadDecision(
                allow_entry=False,
                units_mult=0.0,
                reason="edge_thin_block",
                pred_move_pips=round(pred_move_pips, 6),
                cost_pips=round(cost_pips, 6),
                edge_pips=round(edge_pips, 6),
                slippage_est_pips=round(slippage_est, 6),
                range_pips=round(range_pips, 6),
                momentum_aligned_pips=round(momentum_aligned, 6),
                direction_bias_aligned=round(bias_aligned, 6),
            )
        thin_ratio = _clamp(edge_pips / max(edge_min, 1e-6), 0.0, 1.0)
        units_mult = min_mult + thin_ratio * (1.0 - min_mult)
        return TickLookaheadDecision(
            allow_entry=True,
            units_mult=round(_clamp(units_mult, min_mult, 1.0), 6),
            reason="edge_thin_scale",
            pred_move_pips=round(pred_move_pips, 6),
            cost_pips=round(cost_pips, 6),
            edge_pips=round(edge_pips, 6),
            slippage_est_pips=round(slippage_est, 6),
            range_pips=round(range_pips, 6),
            momentum_aligned_pips=round(momentum_aligned, 6),
            direction_bias_aligned=round(bias_aligned, 6),
        )

    boost_ratio = _clamp((edge_pips - edge_min) / edge_ref, 0.0, 1.0)
    units_mult = 1.0 + boost_ratio * (max_mult - 1.0)
    return TickLookaheadDecision(
        allow_entry=True,
        units_mult=round(_clamp(units_mult, min_mult, max_mult), 6),
        reason="edge_ok",
        pred_move_pips=round(pred_move_pips, 6),
        cost_pips=round(cost_pips, 6),
        edge_pips=round(edge_pips, 6),
        slippage_est_pips=round(slippage_est, 6),
        range_pips=round(range_pips, 6),
        momentum_aligned_pips=round(momentum_aligned, 6),
        direction_bias_aligned=round(bias_aligned, 6),
    )


__all__ = ["TickLookaheadDecision", "decide_tick_lookahead_edge"]

