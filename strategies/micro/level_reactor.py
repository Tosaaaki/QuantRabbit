from __future__ import annotations

from typing import Dict, Optional

PIP = 0.01


class MicroLevelReactor:
    """
    Simple level-based micro strategy that reacts around pre-set intraday levels.
    It covers both breakout/bounce and fade scenarios so we can enter even when
    other momentum filters stay quiet.

    Levels are dynamic by default: anchored at current price ± (ATR * multiplier).
    You can still pin absolute levels via env if必要:
      - MLR_LEVEL_UP / MLR_LEVEL_DOWN (absolute price)
      - MLR_BAND_PIPS (band around levels, in pips)
      - MLR_UP_ATR_MULT / MLR_DOWN_ATR_MULT (multiplier for dynamic levels)
    """

    name = "MicroLevelReactor"
    pocket = "micro"

    # Optional absolute overrides (price). If unset, dynamic ATR-based levels are used.
    ABS_LEVEL_UP = __import__("os").getenv("MLR_LEVEL_UP")
    ABS_LEVEL_DOWN = __import__("os").getenv("MLR_LEVEL_DOWN")
    BAND_PIPS = float(__import__("os").getenv("MLR_BAND_PIPS", "1.2"))  # allow +- band checks
    UP_ATR_MULT = float(__import__("os").getenv("MLR_UP_ATR_MULT", "2.5"))
    DOWN_ATR_MULT = float(__import__("os").getenv("MLR_DOWN_ATR_MULT", "2.5"))

    @staticmethod
    def _as_float(val: Optional[float], default: float = 0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _mk_decision(
        cls,
        action: str,
        sl_pips: float,
        tp_pips: float,
        confidence: int,
        tag: str,
    ) -> Dict:
        return {
            "action": action,
            "sl_pips": round(sl_pips, 2),
            "tp_pips": round(tp_pips, 2),
            "confidence": max(40, min(95, int(confidence))),
            "profile": "level_reactor",
            "loss_guard_pips": round(sl_pips, 2),
            "target_tp_pips": round(tp_pips, 2),
            "min_hold_sec": 60.0,
            "tag": tag,
        }

    @classmethod
    def check(cls, fac: Dict) -> Optional[Dict]:
        price = cls._as_float(fac.get("close"), None)
        if price is None:
            return None
        rsi = cls._as_float(fac.get("rsi"), 50.0)
        atr_pips = cls._as_float(fac.get("atr_pips"), 2.0)

        # Determine levels: prefer absolute env overrides, otherwise dynamic ATR-based.
        if cls.ABS_LEVEL_UP:
            up = cls._as_float(cls.ABS_LEVEL_UP, price + atr_pips * cls.UP_ATR_MULT * PIP)
        else:
            up = price + atr_pips * cls.UP_ATR_MULT * PIP
        if cls.ABS_LEVEL_DOWN:
            down = cls._as_float(cls.ABS_LEVEL_DOWN, price - atr_pips * cls.DOWN_ATR_MULT * PIP)
        else:
            down = price - atr_pips * cls.DOWN_ATR_MULT * PIP

        band = cls.BAND_PIPS * PIP

        # Breakout long: push above upper level with supportive RSI
        if price >= up and rsi >= 50.0:
            sl = max(6.0, atr_pips * 1.6)
            tp = max(10.0, sl * 1.4)
            conf = 70 + min(15, int((rsi - 50.0) * 0.6))
            return cls._mk_decision("OPEN_LONG", sl, tp, conf, "MLR-breakout-long")

        # Fade short near upper band if stretched and RSI hot
        if price >= up - band and rsi >= 63.0:
            sl = max(7.0, atr_pips * 1.7)
            tp = max(8.0, sl * 1.2)
            conf = 65 + min(20, int((rsi - 60.0) * 0.5))
            return cls._mk_decision("OPEN_SHORT", sl, tp, conf, "MLR-fade-upper")

        # Bounce long near lower band
        if price <= down + band and rsi <= 55.0:
            sl = max(7.0, atr_pips * 1.7)
            tp = max(10.0, sl * 1.3)
            conf = 68 + min(18, int((55.0 - rsi) * 0.4))
            return cls._mk_decision("OPEN_LONG", sl, tp, conf, "MLR-bounce-lower")

        # Breakdown short under lower level with soft RSI
        if price <= down and rsi <= 48.0:
            sl = max(6.0, atr_pips * 1.6)
            tp = max(10.0, sl * 1.4)
            conf = 70 + min(15, int((48.0 - rsi) * 0.6))
            return cls._mk_decision("OPEN_SHORT", sl, tp, conf, "MLR-breakdown-short")

        return None
