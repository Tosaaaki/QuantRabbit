from __future__ import annotations

import datetime
import os
from typing import Dict

from execution.micro_guard import (
    micro_loss_cooldown_active,
    micro_recent_loss_guard,
)

_PIP = 0.01

_MAX_ADX = float(os.getenv("BBRSI_MAX_ADX", "48"))
_MAX_SLOPE = float(os.getenv("BBRSI_MAX_SLOPE", "0.10"))
_MAX_BBW = float(os.getenv("BBRSI_MAX_BBW", "0.12"))
_MIN_RSI_BUY = float(os.getenv("BBRSI_MIN_RSI_BUY", "45"))
_MAX_RSI_SELL = float(os.getenv("BBRSI_MAX_RSI_SELL", "55"))

_MAX_TICK_MOM = float(os.getenv("BBRSI_MAX_TICK_MOM", "1.4"))
_MAX_TICK_VELOCITY = float(os.getenv("BBRSI_MAX_TICK_VELOCITY", "2.0"))
_MAX_TICK_VWAP = float(os.getenv("BBRSI_MAX_TICK_VWAP_DELTA", "3.0"))
_MIN_TICK_RANGE = float(os.getenv("BBRSI_MIN_TICK_RANGE", "0.6"))

_MIN_TREND_SPREAD_PIPS = float(os.getenv("BBRSI_MIN_TREND_SPREAD_PIPS", "5.0"))
_H4_ADX_STRONG = float(os.getenv("BBRSI_H4_ADX_STRONG", "24.0"))

_TTL_SEC = int(os.getenv("BBRSI_TTL_SEC", "1500"))

_SESSION_WINDOWS: list[tuple[int, int]] = []
for raw in os.getenv("BBRSI_BLOCK_WINDOWS", "7:00,12:30").split(","):
    raw = raw.strip()
    if not raw:
        continue
    try:
        hour_str, minute_str = raw.split(":", 1)
        _SESSION_WINDOWS.append((int(hour_str), int(minute_str)))
    except ValueError:
        continue
if not _SESSION_WINDOWS:
    _SESSION_WINDOWS = [(7, 0), (12, 30)]

_BLOCK_WINDOW_MIN = int(os.getenv("BBRSI_BLOCK_WINDOW_MINUTES", "45"))


def _parse_timestamp(raw) -> datetime.datetime | None:
    if not raw:
        return None
    try:
        return datetime.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None


def _last_timestamp(fac: Dict) -> datetime.datetime | None:
    candles = fac.get("candles") or []
    if not candles:
        return None
    return _parse_timestamp(candles[-1].get("timestamp"))


def _is_session_blocked(ts: datetime.datetime | None) -> bool:
    if ts is None:
        return False
    span_half_sec = max(_BLOCK_WINDOW_MIN, 0) * 30.0  # half the window in seconds
    if span_half_sec <= 0:
        return False
    for hour, minute in _SESSION_WINDOWS:
        target = ts.replace(hour=hour, minute=minute, second=0, microsecond=0)
        deltas = [
            abs((ts - target).total_seconds()),
            abs((ts - (target + datetime.timedelta(days=1))).total_seconds()),
            abs((ts - (target - datetime.timedelta(days=1))).total_seconds()),
        ]
        if min(deltas) <= span_half_sec:
            return True
    return False


def _trend_bias(fac_h4: Dict | None) -> int:
    if not fac_h4:
        return 0
    ma10 = fac_h4.get("ma10")
    ma20 = fac_h4.get("ma20")
    adx = fac_h4.get("adx")
    if ma10 is None or ma20 is None:
        return 0
    try:
        ma10_f = float(ma10)
        ma20_f = float(ma20)
        adx_f = float(adx or 0.0)
    except (TypeError, ValueError):
        return 0
    spread_pips = (ma10_f - ma20_f) / _PIP
    if abs(spread_pips) < _MIN_TREND_SPREAD_PIPS or adx_f < _H4_ADX_STRONG:
        return 0
    return 1 if spread_pips > 0 else -1


def _calc_sl_tp(atr: float, tighten: bool) -> tuple[int, int]:
    if atr <= 0:
        return (9, 12) if tighten else (10, 14)
    atr_pips = atr * 100
    if tighten:
        sl = max(7, int(round(atr_pips * 0.75)))
        tp = max(sl + 3, int(round(atr_pips * 1.05)))
    else:
        sl = max(8, int(round(atr_pips * 0.85)))
        tp = max(sl + 4, int(round(atr_pips * 1.10)))
    return sl, tp


class BBRsi:
    name = "BB_RSI"
    pocket = "micro"

    @staticmethod
    def check(fac: Dict, fac_h4: Dict | None = None) -> Dict | None:
        if micro_loss_cooldown_active() or micro_recent_loss_guard():
            return None

        rsi = fac.get("rsi")
        bbw = fac.get("bbw")
        ma = fac.get("ma20")
        adx = fac.get("adx")
        slope = fac.get("ma_slope")

        if not all([rsi, bbw, ma]):
            return None

        try:
            rsi_val = float(rsi)
            bbw_val = float(bbw)
            ma_val = float(ma)
        except (TypeError, ValueError):
            return None

        if adx and float(adx) > _MAX_ADX:
            return None
        if slope and abs(float(slope)) > _MAX_SLOPE:
            return None
        if bbw_val > _MAX_BBW or bbw_val <= 0:
            return None

        last_ts = _last_timestamp(fac)
        if _is_session_blocked(last_ts):
            return None

        tick_mom = float(fac.get("tick_momentum_5", 0.0) or 0.0)
        tick_velocity = float(fac.get("tick_velocity_30s", 0.0) or 0.0)
        tick_vwap = float(fac.get("tick_vwap_delta", 0.0) or 0.0)
        tick_range = float(fac.get("tick_range_30s", 0.0) or 0.0)

        if tick_range < _MIN_TICK_RANGE:
            return None

        price = float(fac.get("close", ma_val) or ma_val)
        upper = ma_val + (ma_val * bbw_val / 2)
        lower = ma_val - (ma_val * bbw_val / 2)

        atr = float(fac.get("atr", 0.0) or 0.0)
        bias = _trend_bias(fac_h4)

        if price < lower and rsi_val < _MIN_RSI_BUY:
            if bias < 0:
                return None
            if tick_mom < -_MAX_TICK_MOM or tick_velocity < -_MAX_TICK_VELOCITY:
                return None
            if tick_vwap < -_MAX_TICK_VWAP:
                return None
            sl_pips, tp_pips = _calc_sl_tp(atr, tighten=True)
            return {
                "action": "buy",
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "ttl_sec": _TTL_SEC,
                "be_rr": 0.7,
                "trail_at_rr": 1.05,
                "trail_atr_mult": 0.9,
            }

        if price > upper and rsi_val > _MAX_RSI_SELL:
            if bias > 0:
                return None
            if tick_mom > _MAX_TICK_MOM or tick_velocity > _MAX_TICK_VELOCITY:
                return None
            if tick_vwap > _MAX_TICK_VWAP:
                return None
            sl_pips, tp_pips = _calc_sl_tp(atr, tighten=False)
            return {
                "action": "sell",
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "ttl_sec": _TTL_SEC,
                "be_rr": 0.7,
                "trail_at_rr": 1.05,
                "trail_atr_mult": 0.9,
            }

        return None

