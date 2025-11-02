from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict

from utils.trade_repository import get_trade_repository

PIP = 0.01  # USD/JPY

_MAX_TICK_VELOCITY = float(os.getenv("RANGE_BOUNCE_MAX_TICK_VELOCITY", "3.2"))
_MIN_TICK_RANGE = float(os.getenv("RANGE_BOUNCE_MIN_TICK_RANGE", "0.9"))
_MAX_TICK_RANGE = float(os.getenv("RANGE_BOUNCE_MAX_TICK_RANGE", "6.0"))
_MIN_ATR_PIPS = float(os.getenv("RANGE_BOUNCE_MIN_ATR_PIPS", "1.1"))
_MAX_ATR_PIPS = float(os.getenv("RANGE_BOUNCE_MAX_ATR_PIPS", "7.5"))
_BASE_DEVIATION_PIPS = float(os.getenv("RANGE_BOUNCE_BASE_DEVIATION_PIPS", "0.6"))
_ATR_DEVIATION_MULT = float(os.getenv("RANGE_BOUNCE_ATR_DEVIATION_MULT", "0.32"))
_FAR_MULT = float(os.getenv("RANGE_BOUNCE_FAR_MULT", "1.8"))
_VWAP_MIN = float(os.getenv("RANGE_BOUNCE_VWAP_MIN", "0.35"))
_RSI_LONG_MAX = float(os.getenv("RANGE_BOUNCE_RSI_LONG_MAX", "42.0"))
_RSI_SHORT_MIN = float(os.getenv("RANGE_BOUNCE_RSI_SHORT_MIN", "58.0"))
_TICK_MOM_REV_MIN = float(os.getenv("RANGE_BOUNCE_TICK_MOM_REV_MIN", "0.05"))
_MA_SLOPE_MAX = float(os.getenv("RANGE_BOUNCE_MA_SLOPE_MAX", "1.4"))
_SL_ATR_MULT = float(os.getenv("RANGE_BOUNCE_SL_ATR_MULT", "0.75"))
_SL_MIN_PIPS = float(os.getenv("RANGE_BOUNCE_SL_MIN_PIPS", "2.2"))
_TP_MULT = float(os.getenv("RANGE_BOUNCE_TP_MULT", "1.8"))
_TP_FAR_MULT = float(os.getenv("RANGE_BOUNCE_TP_FAR_MULT", "2.2"))

_GUARD_LOSS_PIPS = max(0.0, float(os.getenv("RANGE_BOUNCE_RECENT_GUARD_PIPS", "6.0")))
_GUARD_COOLDOWN_MINUTES = max(0.0, float(os.getenv("RANGE_BOUNCE_RECENT_GUARD_MINUTES", "45.0")))
_GUARD_MAX_TRADES = max(1, int(os.getenv("RANGE_BOUNCE_RECENT_GUARD_MAX_TRADES", "5")))
_GUARD_SINGLE_LOSS_PIPS = max(0.0, float(os.getenv("RANGE_BOUNCE_COOLDOWN_LOSS_PIPS", "2.2")))
_GUARD_CACHE_SEC = max(0.0, float(os.getenv("RANGE_BOUNCE_GUARD_CACHE_SEC", "15.0")))
_TRADE_REPO = get_trade_repository()
_GUARD_CACHE = {"ts": 0.0, "block": False}


@dataclass
class RangeSignal:
    action: str
    sl_pips: float
    tp_pips: float


class RangeBounce:
    name = "RangeBounce"
    pocket = "micro"

    @staticmethod
    def check(
        fac_m1: Dict[str, float],
        fac_h4: Dict[str, float],
        micro_regime: str,
        macro_regime: str,
    ) -> Dict[str, float] | None:
        if _should_block_new_entry():
            return None

        if micro_regime not in ("Range", "Mixed"):
            return None

        close = fac_m1.get("close")
        ma20 = fac_m1.get("ma20")
        bbw = fac_m1.get("bbw")
        atr = fac_m1.get("atr")
        rsi = fac_m1.get("rsi")
        ma10 = fac_m1.get("ma10")
        tick_velocity = fac_m1.get("tick_velocity_30s")
        tick_range = fac_m1.get("tick_range_30s")
        tick_mom5 = fac_m1.get("tick_momentum_5")
        vwap_delta = fac_m1.get("tick_vwap_delta")
        if None in (close, ma20, bbw, atr, ma10, tick_velocity, tick_range, tick_mom5, vwap_delta):
            return None

        close = float(close)
        ma20 = float(ma20)
        ma10 = float(ma10)
        bbw = float(bbw)
        atr = float(atr)
        rsi = float(rsi or 50.0)
        tick_velocity = float(tick_velocity or 0.0)
        tick_range = float(tick_range or 0.0)
        tick_mom5 = float(tick_mom5 or 0.0)
        vwap_delta = float(vwap_delta or 0.0)
        atr_pips = max(atr / PIP, 0.1)
        ma_slope_pips = abs(ma10 - ma20) / PIP

        # ブレイク仕掛け中はスキップ
        bbw_h4 = float(fac_h4.get("bbw", 1.0) or 1.0)
        if macro_regime == "Breakout" or bbw_h4 > 0.6:
            return None

        if atr_pips < _MIN_ATR_PIPS or atr_pips > _MAX_ATR_PIPS:
            return None
        if abs(tick_velocity) > _MAX_TICK_VELOCITY:
            return None
        if tick_range < _MIN_TICK_RANGE or tick_range > _MAX_TICK_RANGE:
            return None
        if abs(vwap_delta) < _VWAP_MIN:
            return None
        if ma_slope_pips > _MA_SLOPE_MAX:
            return None

        deviation_pips = (close - ma20) / PIP
        regime_factor = 1.0 if micro_regime == "Range" else 1.2
        threshold = max(_BASE_DEVIATION_PIPS * regime_factor, atr_pips * _ATR_DEVIATION_MULT)
        far_threshold = threshold * _FAR_MULT

        def _signal(action: str, far: bool) -> Dict[str, float]:
            sl = max(_SL_MIN_PIPS, atr_pips * _SL_ATR_MULT)
            if far:
                sl *= 1.15
            tp_ratio = _TP_FAR_MULT if far else _TP_MULT
            tp = max(sl * tp_ratio, sl + 2.0)
            return {"action": action, "sl_pips": round(sl, 2), "tp_pips": round(tp, 2)}

        if deviation_pips <= -threshold and rsi <= _RSI_LONG_MAX:
            if tick_mom5 < _TICK_MOM_REV_MIN or tick_velocity < -_TICK_MOM_REV_MIN * 0.6:
                return None
            far = deviation_pips <= -far_threshold or vwap_delta <= -1.2 * _VWAP_MIN
            return _signal("buy", far)
        if deviation_pips >= threshold and rsi >= _RSI_SHORT_MIN:
            if tick_mom5 > -_TICK_MOM_REV_MIN or tick_velocity > _TICK_MOM_REV_MIN * 0.6:
                return None
            far = deviation_pips >= far_threshold or vwap_delta >= 1.2 * _VWAP_MIN
            return _signal("sell", far)
        return None


def _should_block_new_entry() -> bool:
    if _GUARD_LOSS_PIPS <= 0 and _GUARD_SINGLE_LOSS_PIPS <= 0:
        return False
    now = time.time()
    if _GUARD_CACHE_SEC > 0 and now - _GUARD_CACHE["ts"] < _GUARD_CACHE_SEC:
        return bool(_GUARD_CACHE["block"])

    block = False
    loss_sum = 0.0
    recent_trades = []
    try:
        since = datetime.now(timezone.utc) - timedelta(minutes=_GUARD_COOLDOWN_MINUTES)
        history = _TRADE_REPO.recent_trades(
            "micro",
            limit=max(_GUARD_MAX_TRADES * 3, 10),
            closed_only=True,
            since=since if _GUARD_COOLDOWN_MINUTES > 0 else None,
        )
        for trade in history:
            if trade.strategy != RangeBounce.name or trade.pl_pips is None or trade.pl_pips >= 0:
                continue
            recent_trades.append(trade)
            loss_sum += abs(trade.pl_pips)
            if len(recent_trades) >= _GUARD_MAX_TRADES:
                break
        if recent_trades:
            single_breach = any(abs(t.pl_pips or 0.0) >= _GUARD_SINGLE_LOSS_PIPS for t in recent_trades)
            cumulative_breach = loss_sum >= _GUARD_LOSS_PIPS > 0
            block = single_breach or cumulative_breach
            if block:
                last_time = recent_trades[0].close_time.isoformat() if recent_trades[0].close_time else "unknown"
                logging.info(
                    "[RANGE_BOUNCE_GUARD] cooldown active count=%s loss_sum=%.2f last_close=%s",
                    len(recent_trades),
                    loss_sum,
                    last_time,
                )
    except Exception as exc:  # noqa: BLE001
        logging.warning("[RANGE_BOUNCE_GUARD] failed to evaluate guard: %s", exc)
        block = False

    _GUARD_CACHE.update({"ts": now, "block": block})
    return block
