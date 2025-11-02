from __future__ import annotations

import os
from typing import Dict

from execution.micro_guard import (
    micro_loss_cooldown_active,
    micro_recent_loss_guard,
)

_PIP = 0.01  # USD/JPY pip size


def _float_env(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _bool_env(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


_MIN_ADX = _float_env("MICRO_TREND_MIN_ADX", 12.0)
_MIN_ATR_PIPS = _float_env("MICRO_TREND_MIN_ATR_PIPS", 0.9)
_MAX_PULLBACK_PIPS = _float_env("MICRO_TREND_MAX_PULLBACK_PIPS", 6.0)
_MAX_RSI_BUY = _float_env("MICRO_TREND_MAX_RSI", 63.0)
_MIN_RSI_SELL = _float_env("MICRO_TREND_MIN_RSI", 37.0)
_SL_ATR_MULT = _float_env("MICRO_TREND_SL_ATR_MULT", 0.7)
_TP_MULT = _float_env("MICRO_TREND_TP_MULT", 1.55)
_TTL_SEC = int(_float_env("MICRO_TREND_TTL_SEC", 1200))
_MIN_MA_SPREAD_PIPS = _float_env("MICRO_TREND_MIN_MA_SPREAD_PIPS", 2.5)
_REQUIRE_H4_CONFIRM = _bool_env("MICRO_TREND_REQUIRE_H4_CONFIRM", False)

_EXHAUST_ATR_MULT = _float_env("MICRO_TREND_EXHAUST_ATR_MULT", 2.0)
_EXHAUST_MIN_PIPS = _float_env("MICRO_TREND_EXHAUST_MIN_PIPS", 8.0)
_EXHAUST_RSI_BUY = _float_env("MICRO_TREND_EXHAUST_RSI_BUY", 38.0)
_EXHAUST_RSI_SELL = _float_env("MICRO_TREND_EXHAUST_RSI_SELL", 62.0)
_COUNTER_SL_MULT = _float_env("MICRO_TREND_COUNTER_SL_ATR_MULT", 1.1)
_COUNTER_TP_MULT = _float_env("MICRO_TREND_COUNTER_TP_MULT", 1.25)
_COUNTER_SL_MIN = _float_env("MICRO_TREND_COUNTER_SL_MIN_PIPS", 6.0)
_MOMENTUM_FLOOR = _float_env("MICRO_TREND_MOMENTUM_FLOOR", -0.2)
_MOMENTUM_CAP = _float_env("MICRO_TREND_MOMENTUM_CAP", 1.8)
_PULLBACK_MOM_ABS = _float_env("MICRO_TREND_PULLBACK_MOM_ABS", 1.4)
_PULLBACK_VEL_ABS = _float_env("MICRO_TREND_PULLBACK_VEL_ABS", 1.8)
_PULLBACK_VWAP_ABS = _float_env("MICRO_TREND_PULLBACK_VWAP_ABS", 3.0)
_COUNTER_TICK_CAP = _float_env("MICRO_TREND_COUNTER_TICK_CAP", 1.2)
_COUNTER_VEL_CAP = _float_env("MICRO_TREND_COUNTER_VEL_CAP", 1.1)
_COUNTER_VWAP_MIN = _float_env("MICRO_TREND_COUNTER_VWAP_MIN", 0.6)
_MIN_TICK_RANGE = _float_env("MICRO_TREND_MIN_TICK_RANGE", 0.3)
_STRONG_TREND_ADX = _float_env("MICRO_TREND_STRONG_ADX", 24.0)
_STRONG_TREND_SPREAD_PIPS = _float_env("MICRO_TREND_STRONG_SPREAD_PIPS", 4.5)
_STRONG_TREND_SL_BOOST = _float_env("MICRO_TREND_STRONG_SL_BOOST", 1.15)
_ALLOW_MIXED_TREND = _bool_env("MICRO_TREND_ALLOW_MIXED", True)
_MIXED_SLOPE_MAX = _float_env("MICRO_TREND_MIXED_SLOPE_MAX", 1.2)
_MIXED_VELOCITY_MIN = _float_env("MICRO_TREND_MIXED_VELOCITY_MIN", -0.6)
_MIXED_VELOCITY_MAX = _float_env("MICRO_TREND_MIXED_VELOCITY_MAX", 0.9)
_MIXED_MOMENTUM_MIN = _float_env("MICRO_TREND_MIXED_MOMENTUM_MIN", -0.35)


class MicroTrendPullback:
    name = "MicroTrendPullback"
    pocket = "micro"
    requires_h4 = True

    @staticmethod
    def check(fac_m1: Dict, fac_h4: Dict) -> Dict | None:
        ma10_h4 = fac_h4.get("ma10")
        ma20_h4 = fac_h4.get("ma20")
        h4_close = fac_h4.get("close")
        if not ma10_h4 or not ma20_h4 or h4_close is None:
            return None
        try:
            ma10_h4 = float(ma10_h4)
            ma20_h4 = float(ma20_h4)
            h4_close = float(h4_close)
        except (TypeError, ValueError):
            return None

        adx_h4 = float(fac_h4.get("adx", 0.0) or 0.0)
        if adx_h4 < _MIN_ADX:
            return None

        ma_spread_pips = abs(ma10_h4 - ma20_h4) / _PIP
        if ma_spread_pips < _MIN_MA_SPREAD_PIPS:
            return None

        price = fac_m1.get("close")
        ma20_m1 = fac_m1.get("ma20")
        ma10_m1 = fac_m1.get("ma10")
        atr = fac_m1.get("atr")
        rsi = fac_m1.get("rsi")
        try:
            price = float(price)
            ma20_m1 = float(ma20_m1)
            ma10_m1 = float(ma10_m1)
            atr = float(atr)
            rsi = float(rsi)
        except (TypeError, ValueError):
            return None

        atr_pips = atr / _PIP
        if atr_pips < _MIN_ATR_PIPS:
            return None

        pullback_pips = (price - ma20_m1) / _PIP
        max_pull = max(_MAX_PULLBACK_PIPS, atr_pips * 1.1)
        exhaust_limit = max(_EXHAUST_MIN_PIPS, atr_pips * _EXHAUST_ATR_MULT)
        exhaustion_up = (
            abs(pullback_pips) >= exhaust_limit
            and pullback_pips >= 0
            and rsi >= _EXHAUST_RSI_SELL
        )
        exhaustion_down = (
            abs(pullback_pips) >= exhaust_limit
            and pullback_pips <= 0
            and rsi <= _EXHAUST_RSI_BUY
        )

        up_trend = ma10_h4 > ma20_h4 and ma10_m1 >= ma20_m1
        down_trend = ma10_h4 < ma20_h4 and ma10_m1 <= ma20_m1

        candles = fac_m1.get("candles") or []
        prev_close = candles[-2]["close"] if len(candles) >= 2 else price
        momentum_pips = (price - prev_close) / _PIP
        tick_mom5 = float(fac_m1.get("tick_momentum_5", 0.0) or 0.0)
        tick_velocity = float(fac_m1.get("tick_velocity_30s", 0.0) or 0.0)
        tick_vwap_delta = float(fac_m1.get("tick_vwap_delta", 0.0) or 0.0)
        tick_range = float(fac_m1.get("tick_range_30s", 0.0) or 0.0)
        slope_m1_pips = (ma10_m1 - ma20_m1) / _PIP

        if not up_trend and _ALLOW_MIXED_TREND and ma10_h4 > ma20_h4:
            if slope_m1_pips >= -_MIXED_SLOPE_MAX and momentum_pips >= _MIXED_MOMENTUM_MIN and _MIXED_VELOCITY_MIN <= tick_velocity <= _MIXED_VELOCITY_MAX:
                up_trend = True
        if not down_trend and _ALLOW_MIXED_TREND and ma10_h4 < ma20_h4:
            if slope_m1_pips <= _MIXED_SLOPE_MAX and momentum_pips <= -_MIXED_MOMENTUM_MIN and _MIXED_VELOCITY_MIN <= tick_velocity <= _MIXED_VELOCITY_MAX:
                down_trend = True

        if micro_loss_cooldown_active() or micro_recent_loss_guard():
            return None

        if up_trend:
            if _REQUIRE_H4_CONFIRM and h4_close < ma20_h4:
                return None
            if momentum_pips < _MOMENTUM_FLOOR:
                return None
            if momentum_pips > _MOMENTUM_CAP and pullback_pips >= 0:
                return None
            if tick_mom5 <= -_PULLBACK_MOM_ABS or tick_velocity <= -_PULLBACK_VEL_ABS:
                return None
            if tick_vwap_delta <= -_PULLBACK_VWAP_ABS:
                return None
            if tick_range < _MIN_TICK_RANGE:
                return None
            if exhaustion_up:
                if momentum_pips > 0:
                    return None
                if tick_mom5 > _COUNTER_TICK_CAP or tick_velocity > _COUNTER_VEL_CAP:
                    return None
                if tick_vwap_delta < _COUNTER_VWAP_MIN:
                    return None
                sl_pips = max(_COUNTER_SL_MULT * atr_pips, _COUNTER_SL_MIN)
                tp_pips = max(sl_pips * _COUNTER_TP_MULT, sl_pips + 4.0)
                return {
                    "action": "sell",
                    "sl_pips": round(sl_pips, 1),
                    "tp_pips": round(tp_pips, 1),
                    "ttl_sec": int(_TTL_SEC * 1.2),
                    "be_rr": 0.6,
                    "trail_at_rr": 1.0,
                    "trail_atr_mult": 0.8,
                }

            if pullback_pips > max_pull or pullback_pips < -2.2 * max_pull:
                return None
            if rsi > _MAX_RSI_BUY:
                return None
            sl_pips = max(_SL_ATR_MULT * atr_pips, 8.0)
            if adx_h4 >= _STRONG_TREND_ADX and ma_spread_pips >= _STRONG_TREND_SPREAD_PIPS:
                sl_pips = max(sl_pips, atr_pips * _STRONG_TREND_SL_BOOST, 10.0)
                tp_pips = max(sl_pips * (_TP_MULT - 0.2), sl_pips + 6.0)
            else:
                tp_pips = max(sl_pips * _TP_MULT, 12.0)
            return {
                "action": "buy",
                "sl_pips": round(sl_pips, 1),
                "tp_pips": round(tp_pips, 1),
                "ttl_sec": _TTL_SEC,
                "be_rr": 0.75,
                "trail_at_rr": 1.15,
                "trail_atr_mult": 1.0,
            }

        if down_trend:
            if _REQUIRE_H4_CONFIRM and h4_close > ma20_h4:
                return None
            if momentum_pips > -_MOMENTUM_FLOOR:
                return None
            if momentum_pips < -_MOMENTUM_CAP and pullback_pips <= 0:
                return None
            if tick_mom5 >= _PULLBACK_MOM_ABS or tick_velocity >= _PULLBACK_VEL_ABS:
                return None
            if tick_vwap_delta >= _PULLBACK_VWAP_ABS:
                return None
            if tick_range < _MIN_TICK_RANGE:
                return None
            if exhaustion_down:
                if momentum_pips < 0:
                    return None
                if tick_mom5 < -_COUNTER_TICK_CAP or tick_velocity < -_COUNTER_VEL_CAP:
                    return None
                if tick_vwap_delta > -_COUNTER_VWAP_MIN:
                    return None
                sl_pips = max(_COUNTER_SL_MULT * atr_pips, _COUNTER_SL_MIN)
                tp_pips = max(sl_pips * _COUNTER_TP_MULT, sl_pips + 4.0)
                return {
                    "action": "buy",
                    "sl_pips": round(sl_pips, 1),
                    "tp_pips": round(tp_pips, 1),
                    "ttl_sec": int(_TTL_SEC * 1.2),
                    "be_rr": 0.6,
                    "trail_at_rr": 1.0,
                    "trail_atr_mult": 0.8,
                }

            if pullback_pips < -max_pull or pullback_pips > 2.2 * max_pull:
                return None
            if rsi < _MIN_RSI_SELL:
                return None
            sl_pips = max(_SL_ATR_MULT * atr_pips, 8.0)
            if adx_h4 >= _STRONG_TREND_ADX and ma_spread_pips >= _STRONG_TREND_SPREAD_PIPS:
                sl_pips = max(sl_pips, atr_pips * _STRONG_TREND_SL_BOOST, 10.0)
                tp_pips = max(sl_pips * (_TP_MULT - 0.2), sl_pips + 6.0)
            else:
                tp_pips = max(sl_pips * _TP_MULT, 12.0)
            return {
                "action": "sell",
                "sl_pips": round(sl_pips, 1),
                "tp_pips": round(tp_pips, 1),
                "ttl_sec": _TTL_SEC,
                "be_rr": 0.75,
                "trail_at_rr": 1.15,
                "trail_atr_mult": 1.0,
            }

        return None
