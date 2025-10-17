from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

_PIP = 0.01  # USD/JPY pip value


@dataclass
class ScalpSignal:
    action: str  # "buy" or "sell"
    sl_pips: float
    tp_pips: float


class BasicScalpStrategy:
    """Simple mean-reversion scalping strategy on top of M1 factors."""

    name = "ScalpMeanRevert"
    pocket = "scalp"

    @staticmethod
    def evaluate(
        fac_m1: Dict[str, Any],
        *,
        spread_pips: float,
        max_spread_pips: float,
        min_atr_pips: float,
        deviation_pips: float,
        atr_threshold_mult: float,
        min_sl_pips: float,
        tp_multiplier: float,
        momentum_flip_pips: float,
        momentum_confirm_pips: float,
        momentum_velocity_cap: float,
        momentum_range_min: float,
        enable_trend_follow: bool = False,
        trend_velocity_min: float = 0.0,
        trend_range_min: float = 0.0,
        trend_momentum_min: float = 0.0,
        trend_rsi_buy: float = 60.0,
        trend_rsi_sell: float = 40.0,
        trend_sl_pips: float = 7.0,
        trend_tp_multiplier: float = 1.5,
        candle_range_pips: float = 0.0,
        revert_range_block_pips: float = 0.0,
        revert_range_widen_pips: float = 0.0,
        revert_range_sl_boost: float = 1.0,
        revert_gap_pips: float = 0.0,
        revert_rsi_long_max: float = 40.0,
        revert_rsi_short_min: float = 60.0,
        force_trend: bool = False,
        forced_direction: str | None = None,
        allow_mean_revert: bool = True,
    ) -> Tuple[ScalpSignal | None, str]:
        price = fac_m1.get("close")
        ma10 = fac_m1.get("ma10")
        rsi = fac_m1.get("rsi")
        atr = fac_m1.get("atr")

        try:
            price = float(price)
            ma10 = float(ma10)
            rsi = float(rsi)
            atr = float(atr)
        except (TypeError, ValueError):
            return None, "missing_factors"

        atr_pips = atr / _PIP
        ultra_min = max(0.5, min_atr_pips * 0.35)
        if atr_pips < ultra_min:
            return None, "atr_low"
        atr_soft = atr_pips < (min_atr_pips * 0.85)
        atr_medium = atr_pips < (min_atr_pips * 1.1)
        momentum_5 = float(fac_m1.get("tick_momentum_5") or 0.0)
        momentum_10 = float(fac_m1.get("tick_momentum_10") or 0.0)
        velocity_30 = abs(float(fac_m1.get("tick_velocity_30s") or 0.0))
        range_30 = float(fac_m1.get("tick_range_30s") or 0.0)

        force_trend = bool(force_trend)
        allow_mean_revert = bool(allow_mean_revert)
        forced_direction = (forced_direction or "").lower() or None

        trend_long = False
        trend_short = False
        if enable_trend_follow:
            if (
                velocity_30 >= trend_velocity_min
                and range_30 >= trend_range_min
                and momentum_5 >= trend_momentum_min
                and momentum_10 >= trend_momentum_min * 0.8
                and rsi >= trend_rsi_buy
                and price >= ma10
            ):
                trend_long = True
            if (
                velocity_30 >= trend_velocity_min
                and range_30 >= trend_range_min
                and momentum_5 <= -trend_momentum_min
                and momentum_10 <= -trend_momentum_min * 0.8
                and rsi <= trend_rsi_sell
                and price <= ma10
            ):
                trend_short = True

        if force_trend:
            if forced_direction == "long":
                trend_short = False
                if not trend_long and price >= ma10 and rsi >= 48:
                    trend_long = True
            elif forced_direction == "short":
                trend_long = False
                if not trend_short and price <= ma10 and rsi <= 52:
                    trend_short = True
            else:
                if not trend_long and price >= ma10 and rsi >= 55:
                    trend_long = True
                if not trend_short and price <= ma10 and rsi <= 45:
                    trend_short = True

        candle_range_pips = float(candle_range_pips or 0.0)
        revert_range_block_pips = max(0.0, float(revert_range_block_pips or 0.0))
        revert_range_widen_pips = max(0.0, float(revert_range_widen_pips or 0.0))
        revert_range_sl_boost = max(1.0, float(revert_range_sl_boost or 1.0))
        revert_gap_pips = max(0.0, float(revert_gap_pips or 0.0))
        revert_rsi_long_max = max(5.0, float(revert_rsi_long_max or 40.0))
        revert_rsi_short_min = min(95.0, float(revert_rsi_short_min or 60.0))

        if spread_pips > max_spread_pips:
            return None, "spread_high"

        deviation = abs(price - ma10) / _PIP
        base_threshold = max(deviation_pips, atr_pips * atr_threshold_mult)
        threshold = base_threshold * (0.75 if atr_soft else 0.95 if atr_medium else 1.05)
        if deviation < threshold and not (trend_long or trend_short):
            return None, "no_deviation"

        sl_multiplier = 1.0
        if not (trend_long or trend_short):
            if revert_range_block_pips > 0.0 and candle_range_pips >= revert_range_block_pips:
                return None, "candle_range_block"
            if (
                revert_range_widen_pips > 0.0
                and candle_range_pips >= revert_range_widen_pips
            ):
                sl_multiplier = revert_range_sl_boost

        if not (trend_long or trend_short):
            if velocity_30 > momentum_velocity_cap:
                return None, "momentum_fast"
            if range_30 < momentum_range_min:
                return None, "range_low"

        min_sl_adapted = max(min_sl_pips, 6.0)
        sl_floor = min_sl_adapted * (0.9 if atr_soft else 1.05 if atr_medium else 1.2)
        atr_stop = atr_pips * (1.4 if atr_soft else 1.7 if atr_medium else 2.0)
        deviation_stop = deviation * 0.6
        sl_pips = max(sl_floor, atr_stop, deviation_stop, 6.0) * sl_multiplier

        tp_multiplier_adj = (
            max(tp_multiplier, 1.25)
            if not atr_soft and not atr_medium
            else max(1.05, tp_multiplier * (0.7 if atr_soft else 0.85))
        )
        tp_floor = sl_pips + max(2.5, atr_pips * 0.9)
        tp_pips = max(tp_floor, sl_pips * tp_multiplier_adj)

        widened = sl_multiplier > 1.0

        if force_trend:
            if trend_long and forced_direction != "short":
                sl_long = max(trend_sl_pips, sl_pips, atr_pips * 1.1)
                tp_long = max(sl_long * trend_tp_multiplier, sl_long + max(3.0, atr_pips))
                return ScalpSignal(action="buy", sl_pips=sl_long, tp_pips=tp_long), "trend_forced_long"
            if trend_short and forced_direction != "long":
                sl_short = max(trend_sl_pips, sl_pips, atr_pips * 1.1)
                tp_short = max(sl_short * trend_tp_multiplier, sl_short + max(3.0, atr_pips))
                return ScalpSignal(action="sell", sl_pips=sl_short, tp_pips=tp_short), "trend_forced_short"
            return None, "trend_override_no_signal"

        deviation_pips = abs(price - ma10) / _PIP

        if allow_mean_revert and price > ma10 and rsi > 56 and not trend_short:
            if not (
                momentum_5 <= -momentum_flip_pips
                and momentum_10 <= -momentum_confirm_pips
            ):
                return None, "momentum_short_missing"
            if deviation_pips < revert_gap_pips or rsi < revert_rsi_short_min:
                return None, "revert_gap_short"
            reason = "short_revert_wide" if widened else "short_revert"
            return ScalpSignal(action="sell", sl_pips=sl_pips, tp_pips=tp_pips), reason
        if allow_mean_revert and price < ma10 and rsi < 44 and not trend_long:
            if not (
                momentum_5 >= momentum_flip_pips
                and momentum_10 >= momentum_confirm_pips
            ):
                return None, "momentum_long_missing"
            if deviation_pips < revert_gap_pips or rsi > revert_rsi_long_max:
                return None, "revert_gap_long"
            reason = "long_revert_wide" if widened else "long_revert"
            return ScalpSignal(action="buy", sl_pips=sl_pips, tp_pips=tp_pips), reason
        if not allow_mean_revert and not force_trend and not (trend_long or trend_short):
            return None, "mean_revert_disabled"
        if trend_long:
            sl_long = max(trend_sl_pips, 6.0, atr_pips * 1.2)
            tp_long = max(sl_long * trend_tp_multiplier, sl_long + max(3.0, atr_pips))
            return ScalpSignal(action="buy", sl_pips=sl_long, tp_pips=tp_long), "trend_long"
        if trend_short:
            sl_short = max(trend_sl_pips, 6.0, atr_pips * 1.2)
            tp_short = max(sl_short * trend_tp_multiplier, sl_short + max(3.0, atr_pips))
            return ScalpSignal(action="sell", sl_pips=sl_short, tp_pips=tp_short), "trend_short"
        return None, "no_pattern"
