from __future__ import annotations

from typing import Dict

PIP = 0.01
MIN_GAP_TREND = 0.32
MIN_ADX = 24.0
MIN_ATR = 1.2
VOL_MIN = 0.7
RSI_LONG_MIN = 54
RSI_SHORT_MAX = 46


class MomentumBurstMicro:
    name = "MomentumBurst"
    pocket = "micro"

    @staticmethod
    def _attr(fac: Dict, key: str, default: float = 0.0) -> float:
        try:
            return float(fac.get(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _atr_pips(fac: Dict) -> float:
        atr = fac.get("atr_pips")
        if atr is not None:
            try:
                return float(atr)
            except (TypeError, ValueError):
                return 0.0
        raw = fac.get("atr")
        if raw is None:
            return 0.0
        try:
            return float(raw) * 100.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = fac.get("close")
        ma10 = fac.get("ma10")
        ma20 = fac.get("ma20")
        if close is None or ma10 is None or ma20 is None:
            return None
        atr_pips = MomentumBurstMicro._atr_pips(fac)
        if atr_pips < MIN_ATR:
            return None
        vol_5m = MomentumBurstMicro._attr(fac, "vol_5m", 1.0)
        if vol_5m < VOL_MIN:
            return None
        adx = MomentumBurstMicro._attr(fac, "adx", 0.0)
        gap_pips = (ma10 - ma20) / PIP
        ema20 = fac.get("ema20") or ma20
        rsi = MomentumBurstMicro._attr(fac, "rsi", 50.0)

        def _build_signal(action: str, bias_pips: float) -> Dict:
            strength = abs(gap_pips)
            sl = max(0.9, min(atr_pips * 1.05, 0.45 * strength + 0.75))
            tp = max(sl * 1.45, min(atr_pips * 2.2, sl + strength * 0.6))
            confidence = int(
                max(
                    55.0,
                    min(
                        97.0,
                        60.0
                        + (strength - MIN_GAP_TREND) * 6.0
                        + max(0.0, adx - MIN_ADX) * 1.2
                        + max(0.0, (atr_pips - MIN_ATR) * 2.5),
                    ),
                )
            )
            profile = "momentum_burst"
            min_hold = max(90.0, min(540.0, tp * 42.0))
            return {
                "action": action,
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "profile": profile,
                "loss_guard_pips": round(sl, 2),
                "target_tp_pips": round(tp, 2),
                "min_hold_sec": round(min_hold, 1),
                "tag": f"{MomentumBurstMicro.name}-{action.lower()}",
            }

        if gap_pips >= MIN_GAP_TREND and adx >= MIN_ADX and close > ema20 + 0.0015:
            if rsi >= RSI_LONG_MIN:
                return _build_signal("OPEN_LONG", gap_pips)

        if gap_pips <= -MIN_GAP_TREND and adx >= MIN_ADX and close < ema20 - 0.0015:
            if rsi <= RSI_SHORT_MAX:
                return _build_signal("OPEN_SHORT", gap_pips)

        return None
