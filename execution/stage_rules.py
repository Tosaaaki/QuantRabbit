"""Shared stage gating and lot computation utilities."""

from __future__ import annotations

import logging
from typing import Dict

LOG = logging.getLogger(__name__)

STAGE_RATIOS = {
    "macro": (0.35,),
    "micro": (0.22, 0.18, 0.16, 0.14, 0.12, 0.08, 0.05, 0.05),
    "scalp": (0.5, 0.3, 0.15, 0.05),
}

MACRO_MAX_TOTAL_LOT = 0.003
MACRO_MIN_TOTAL_LOT = 0.001
MIN_SCALP_STAGE_LOT = 0.01  # 1000 units baseline so sizing never vanishes


def macro_direction_allowed(
    action: str,
    fac_m1: Dict[str, float],
    fac_h4: Dict[str, float],
) -> bool:
    ma10_h4 = fac_h4.get("ma10")
    ma20_h4 = fac_h4.get("ma20")
    atr_h4 = fac_h4.get("atr")
    adx_h4 = fac_h4.get("adx")
    if ma10_h4 is None or ma20_h4 is None:
        return True
    if adx_h4 is not None and adx_h4 < 18.0:
        return False
    if atr_h4:
        gap_ratio = abs(ma10_h4 - ma20_h4) / max(atr_h4, 1e-6)
        if gap_ratio < 0.25:
            return False
    close_m1 = fac_m1.get("close")
    ema20_m1 = fac_m1.get("ema20") or fac_m1.get("ma20")
    if close_m1 is None or ema20_m1 is None:
        return True

    if action == "OPEN_LONG":
        if ma10_h4 <= ma20_h4 - 0.0002:
            return False
        if close_m1 < ema20_m1 - 0.002:
            return False
    else:
        if ma10_h4 >= ma20_h4 + 0.0002:
            return False
        if close_m1 > ema20_m1 + 0.002:
            return False
    return True


def _stage_conditions_met(
    pocket: str,
    stage_idx: int,
    action: str,
    fac_m1: Dict[str, float],
    fac_h4: Dict[str, float],
    open_info: Dict[str, float],
) -> bool:
    if stage_idx == 0:
        return True

    price = fac_m1.get("close")
    avg_price = open_info.get("avg_price", price or 0.0)
    rsi = fac_m1.get("rsi", 50.0)
    adx_h4 = fac_h4.get("adx", 0.0)
    slope_h4 = abs(fac_h4.get("ma20", 0.0) - fac_h4.get("ma10", 0.0))
    atr_h4_raw = fac_h4.get("atr")
    atr_h4_pips = (atr_h4_raw or 0.0) * 100.0
    gap_h4_pips = abs((fac_h4.get("ma10", 0.0) or 0.0) - (fac_h4.get("ma20", 0.0) or 0.0)) * 100.0
    strength_ratio = gap_h4_pips / atr_h4_pips if atr_h4_pips > 1e-6 else 0.0

    if pocket == "macro":
        ma10_h4 = fac_h4.get("ma10")
        ma20_h4 = fac_h4.get("ma20")
        ma10_m1 = fac_m1.get("ma10")
        ma20_m1 = fac_m1.get("ma20")
        ema20_m1 = fac_m1.get("ema20") or ma20_m1
        close_m1 = fac_m1.get("close")

        if action == "OPEN_LONG":
            if ma10_h4 is not None and ma20_h4 is not None and ma10_h4 < ma20_h4:
                LOG.info(
                    "[STAGE] Macro buy gating: H4 trend down (ma10 %.3f < ma20 %.3f).",
                    ma10_h4,
                    ma20_h4,
                )
                return False
            if (
                close_m1 is not None
                and ema20_m1 is not None
                and close_m1 < ema20_m1 - 0.005
            ):
                LOG.info(
                    "[STAGE] Macro buy gating: M1 close %.3f below ema20 %.3f.",
                    close_m1,
                    ema20_m1,
                )
                return False
            if ma10_m1 is not None and ma20_m1 is not None and ma10_m1 < ma20_m1:
                LOG.info(
                    "[STAGE] Macro buy gating: M1 ma10 %.3f < ma20 %.3f.",
                    ma10_m1,
                    ma20_m1,
                )
                return False

        if action == "OPEN_SHORT":
            if ma10_h4 is not None and ma20_h4 is not None and ma10_h4 > ma20_h4:
                LOG.info(
                    "[STAGE] Macro sell gating: H4 trend up (ma10 %.3f > ma20 %.3f).",
                    ma10_h4,
                    ma20_h4,
                )
                return False
            if (
                close_m1 is not None
                and ema20_m1 is not None
                and close_m1 > ema20_m1 + 0.005
            ):
                LOG.info(
                    "[STAGE] Macro sell gating: M1 close %.3f above ema20 %.3f.",
                    close_m1,
                    ema20_m1,
                )
                return False
            if ma10_m1 is not None and ma20_m1 is not None and ma10_m1 > ma20_m1:
                LOG.info(
                    "[STAGE] Macro sell gating: M1 ma10 %.3f > ma20 %.3f.",
                    ma10_m1,
                    ma20_m1,
                )
                return False

        if stage_idx >= 2:
            if strength_ratio < 0.9 or adx_h4 < 27 or slope_h4 < 0.0006:
                LOG.info(
                    "[STAGE] Macro gating failed (strength %.2f, ADX %.2f) for stage %d.",
                    strength_ratio,
                    adx_h4,
                    stage_idx,
                )
                return False
        elif stage_idx == 1:
            if strength_ratio < 0.55 or adx_h4 < 22 or slope_h4 < 0.00045:
                LOG.info(
                    "[STAGE] Macro gating weak (strength %.2f, ADX %.2f) for stage %d.",
                    strength_ratio,
                    adx_h4,
                    stage_idx,
                )
                return False
        if price is not None and avg_price:
            if action == "OPEN_LONG" and price < avg_price - 0.02:
                LOG.info(
                    "[STAGE] Macro buy gating: price %.3f below avg %.3f.", price, avg_price
                )
                return False
            if action == "OPEN_SHORT" and price > avg_price + 0.02:
                LOG.info(
                    "[STAGE] Macro sell gating: price %.3f above avg %.3f.", price, avg_price
                )
                return False
        if action == "OPEN_LONG":
            threshold = 60 - stage_idx * 5
            if rsi > threshold:
                LOG.info(
                    "[STAGE] Macro buy gating: RSI %.1f > %.1f for stage %d.",
                    rsi,
                    threshold,
                    stage_idx,
                )
                return False
        else:
            threshold = 40 + stage_idx * 5
            if rsi < threshold:
                LOG.info(
                    "[STAGE] Macro sell gating: RSI %.1f < %.1f for stage %d.",
                    rsi,
                    threshold,
                    stage_idx,
                )
                return False
        return True

    if pocket == "micro":
        if action == "OPEN_LONG":
            threshold = 45 - min(stage_idx * 5, 15)
            if rsi > threshold:
                LOG.info(
                    "[STAGE] Micro buy gating: RSI %.1f > %.1f for stage %d.",
                    rsi,
                    threshold,
                    stage_idx,
                )
                return False
        else:
            threshold = 55 + min(stage_idx * 5, 15)
            if rsi < threshold:
                LOG.info(
                    "[STAGE] Micro sell gating: RSI %.1f < %.1f for stage %d.",
                    rsi,
                    threshold,
                    stage_idx,
                )
                return False
        return True

    if pocket == "scalp":
        atr = fac_m1.get("atr", 0.0) * 100
        if atr < 2.2:
            LOG.info("[STAGE] Scalp gating: ATR %.2f too low for stage %d.", atr, stage_idx)
            return False
        momentum = (fac_m1.get("close") or 0.0) - (fac_m1.get("ema20") or 0.0)
        if action == "OPEN_LONG" and momentum > 0:
            LOG.info(
                "[STAGE] Scalp buy gating: momentum %.4f positive (stage %d).",
                momentum,
                stage_idx,
            )
            return False
        if action == "OPEN_SHORT" and momentum < 0:
            LOG.info(
                "[STAGE] Scalp sell gating: momentum %.4f negative (stage %d).",
                momentum,
                stage_idx,
            )
            return False
        if action == "OPEN_LONG":
            if rsi > 55 - min(stage_idx * 4, 12):
                LOG.info(
                    "[STAGE] Scalp buy gating: RSI %.1f too high (stage %d).",
                    rsi,
                    stage_idx,
                )
                return False
        else:
            if rsi < 45 + min(stage_idx * 4, 12):
                LOG.info(
                    "[STAGE] Scalp sell gating: RSI %.1f too low (stage %d).",
                    rsi,
                    stage_idx,
                )
                return False
        return True

    return True


def compute_stage_lot(
    pocket: str,
    total_lot: float,
    open_units_same_dir: int,
    action: str,
    fac_m1: Dict[str, float],
    fac_h4: Dict[str, float],
    open_info: Dict[str, float],
) -> tuple[float, int]:
    plan = STAGE_RATIOS.get(pocket, (1.0,))
    current_lot = max(open_units_same_dir, 0) / 100000.0
    cumulative = 0.0
    if pocket == "macro" and not macro_direction_allowed(action, fac_m1, fac_h4):
        LOG.info("[STAGE] Macro entry gated by H4 alignment (action=%s).", action)
        return 0.0, 0
    if pocket == "macro":
        total_lot = min(total_lot, MACRO_MAX_TOTAL_LOT)
        total_lot = max(total_lot, MACRO_MIN_TOTAL_LOT)
    for stage_idx, fraction in enumerate(plan):
        cumulative += fraction
        stage_target = total_lot * cumulative
        if current_lot + 1e-4 < stage_target:
            if not _stage_conditions_met(
                pocket, stage_idx, action, fac_m1, fac_h4, open_info
            ):
                return 0.0, stage_idx
            next_lot = max(stage_target - current_lot, 0.0)
            remaining = max(total_lot - current_lot, 0.0)
            if pocket == "scalp" and remaining > 0:
                floor = min(MIN_SCALP_STAGE_LOT, remaining)
                next_lot = max(next_lot, floor)
            if remaining > 0:
                next_lot = min(next_lot, remaining)
            LOG.info(
                "[STAGE] %s pocket total=%.3f current=%.3f -> next=%.3f (stage %d)",
                pocket,
                round(stage_target, 4),
                round(current_lot, 4),
                round(next_lot, 4),
                stage_idx,
            )
            return next_lot, stage_idx
    return 0.0, len(plan)

