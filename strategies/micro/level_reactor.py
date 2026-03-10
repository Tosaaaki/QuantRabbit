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
    BAND_PIPS = float(__import__("os").getenv("MLR_BAND_PIPS", "1.0"))  # allow +- band checks
    UP_ATR_MULT = float(__import__("os").getenv("MLR_UP_ATR_MULT", "2.0"))
    DOWN_ATR_MULT = float(__import__("os").getenv("MLR_DOWN_ATR_MULT", "2.0"))
    LONG_RSI_MIN = float(__import__("os").getenv("MLR_LONG_RSI_MIN", "54.0"))
    LONG_BOUNCE_RSI_MAX = float(__import__("os").getenv("MLR_LONG_BOUNCE_RSI_MAX", "52.0"))
    BOUNCE_BODY_BEAR_MAX_PIPS = float(__import__("os").getenv("MLR_BOUNCE_BODY_BEAR_MAX_PIPS", "1.2"))
    BOUNCE_MIN_LOWER_WICK_PIPS = float(__import__("os").getenv("MLR_BOUNCE_MIN_LOWER_WICK_PIPS", "0.8"))
    BOUNCE_COUNTERTREND_MIN_GAP_PIPS = float(
        __import__("os").getenv("MLR_BOUNCE_COUNTERTREND_MIN_GAP_PIPS", "0.6")
    )
    BOUNCE_COUNTERTREND_MIN_BODY_PIPS = float(
        __import__("os").getenv("MLR_BOUNCE_COUNTERTREND_MIN_BODY_PIPS", "0.2")
    )
    BOUNCE_COUNTERTREND_MIN_LOWER_WICK_PIPS = float(
        __import__("os").getenv("MLR_BOUNCE_COUNTERTREND_MIN_LOWER_WICK_PIPS", "1.0")
    )
    BOUNCE_CONTINUATION_ATR_MAX = float(
        __import__("os").getenv("MLR_BOUNCE_CONTINUATION_ATR_MAX", "1.8")
    )
    BOUNCE_CONTINUATION_ADX_MIN = float(
        __import__("os").getenv("MLR_BOUNCE_CONTINUATION_ADX_MIN", "22.0")
    )
    BOUNCE_CONTINUATION_DI_GAP_MIN = float(
        __import__("os").getenv("MLR_BOUNCE_CONTINUATION_DI_GAP_MIN", "18.0")
    )
    BOUNCE_CONTINUATION_MIN_LOWER_WICK_PIPS = float(
        __import__("os").getenv("MLR_BOUNCE_CONTINUATION_MIN_LOWER_WICK_PIPS", "0.4")
    )

    @staticmethod
    def _as_float(val: Optional[float], default: float = 0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _candle_bias(cls, fac: Dict, price: float) -> tuple[float, float, float]:
        open_price = cls._as_float(fac.get("open"), price)
        high = cls._as_float(fac.get("high"), price)
        low = cls._as_float(fac.get("low"), price)
        body_pips = (price - open_price) / PIP
        upper_wick_pips = max(0.0, (high - max(price, open_price)) / PIP)
        lower_wick_pips = max(0.0, (min(price, open_price) - low) / PIP)
        return body_pips, upper_wick_pips, lower_wick_pips

    @classmethod
    def _ma_gap_pips(cls, fac: Dict) -> Optional[float]:
        ma_fast = fac.get("ma10") if fac.get("ma10") is not None else fac.get("ema10")
        ma_slow = fac.get("ma20") if fac.get("ma20") is not None else fac.get("ema20")
        try:
            if ma_fast is None or ma_slow is None:
                return None
            return (float(ma_fast) - float(ma_slow)) / PIP
        except (TypeError, ValueError):
            return None

    @classmethod
    def _bounce_continuation_probe(
        cls,
        fac: Dict,
        *,
        body_pips: float,
        lower_wick_pips: float,
        atr_pips: float,
    ) -> bool:
        if body_pips < cls.BOUNCE_COUNTERTREND_MIN_BODY_PIPS:
            return False
        if lower_wick_pips >= cls.BOUNCE_CONTINUATION_MIN_LOWER_WICK_PIPS:
            return False
        if atr_pips > cls.BOUNCE_CONTINUATION_ATR_MAX:
            return False
        adx = cls._as_float(fac.get("adx"), 0.0)
        if adx < cls.BOUNCE_CONTINUATION_ADX_MIN:
            return False
        plus_di = cls._as_float(fac.get("plus_di"), 0.0)
        minus_di = cls._as_float(fac.get("minus_di"), 0.0)
        return (minus_di - plus_di) >= cls.BOUNCE_CONTINUATION_DI_GAP_MIN

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
            "tag": f"{MicroLevelReactor.name}-{tag}",
        }

    @classmethod
    def check(cls, fac: Dict) -> Optional[Dict]:
        price = cls._as_float(fac.get("close"), None)
        if price is None:
            return None

        # Spread filter: skip when spread eats into expected profit
        spread_pips = cls._as_float(fac.get("spread_pips"), 0.0)
        atr_pips = cls._as_float(fac.get("atr_pips"), 2.0)
        if spread_pips > 0:
            spread_cap = max(1.0, atr_pips * 0.30)
            if spread_pips > spread_cap:
                return None

        # Volatility floor: skip if ATR is too low (noise-dominated market)
        if atr_pips < 1.0:
            return None

        rsi = cls._as_float(fac.get("rsi"), 50.0)
        body_pips, upper_wick_pips, lower_wick_pips = cls._candle_bias(fac, price)
        ma_gap_pips = cls._ma_gap_pips(fac)
        # Dynamic levels must be anchored to a slower-moving reference (not the current price),
        # otherwise they would never be crossed.
        anchor = cls._as_float(
            fac.get("ma20") or fac.get("ema20") or fac.get("vwap") or price,
            price,
        )

        # Determine levels: prefer absolute env overrides, otherwise dynamic ATR-based.
        if cls.ABS_LEVEL_UP:
            up = cls._as_float(cls.ABS_LEVEL_UP, anchor + atr_pips * cls.UP_ATR_MULT * PIP)
        else:
            up = anchor + atr_pips * cls.UP_ATR_MULT * PIP
        if cls.ABS_LEVEL_DOWN:
            down = cls._as_float(cls.ABS_LEVEL_DOWN, anchor - atr_pips * cls.DOWN_ATR_MULT * PIP)
        else:
            down = anchor - atr_pips * cls.DOWN_ATR_MULT * PIP

        band = cls.BAND_PIPS * PIP
        breakout_supportive = (
            body_pips >= 0.0
            or (
                lower_wick_pips >= cls.BOUNCE_MIN_LOWER_WICK_PIPS
                and lower_wick_pips + 0.2 >= upper_wick_pips
            )
        )

        # Breakout long: push above upper level with supportive RSI
        if price >= up and rsi >= cls.LONG_RSI_MIN and breakout_supportive:
            sl = max(6.0, atr_pips * 1.6)
            tp = max(10.0, sl * 1.4)
            conf = 70 + min(15, int((rsi - 50.0) * 0.6))
            return cls._mk_decision("OPEN_LONG", sl, tp, conf, "breakout-long")

        # Fade short near upper band if stretched and RSI hot
        if price >= up - band and rsi >= 63.0:
            sl = max(7.0, atr_pips * 1.7)
            tp = max(8.0, sl * 1.2)
            conf = 65 + min(20, int((rsi - 60.0) * 0.5))
            return cls._mk_decision("OPEN_SHORT", sl, tp, conf, "fade-upper")

        # Bounce long near lower band
        if price <= down + band and rsi <= cls.LONG_BOUNCE_RSI_MAX:
            bullish_reject = (
                body_pips >= -cls.BOUNCE_BODY_BEAR_MAX_PIPS
                and (body_pips >= 0.0 or lower_wick_pips >= cls.BOUNCE_MIN_LOWER_WICK_PIPS)
                and lower_wick_pips + 0.1 >= upper_wick_pips
            )
            countertrend_probe = (
                ma_gap_pips is not None and ma_gap_pips <= -cls.BOUNCE_COUNTERTREND_MIN_GAP_PIPS
            )
            continuation_probe = cls._bounce_continuation_probe(
                fac,
                body_pips=body_pips,
                lower_wick_pips=lower_wick_pips,
                atr_pips=atr_pips,
            )
            if countertrend_probe:
                bullish_reject = (
                    bullish_reject
                    and body_pips >= cls.BOUNCE_COUNTERTREND_MIN_BODY_PIPS
                    and lower_wick_pips >= cls.BOUNCE_COUNTERTREND_MIN_LOWER_WICK_PIPS
                    and lower_wick_pips > upper_wick_pips
                )
            if continuation_probe:
                bullish_reject = (
                    bullish_reject
                    and lower_wick_pips >= cls.BOUNCE_CONTINUATION_MIN_LOWER_WICK_PIPS
                    and lower_wick_pips > upper_wick_pips
                )
            if not bullish_reject:
                return None
            sl = max(7.0, atr_pips * 1.7)
            tp = max(10.0, sl * 1.3)
            conf = 68 + min(18, int((55.0 - rsi) * 0.4))
            return cls._mk_decision("OPEN_LONG", sl, tp, conf, "bounce-lower")

        # Breakdown short under lower level with soft RSI
        if price <= down and rsi <= 48.0:
            sl = max(6.0, atr_pips * 1.6)
            tp = max(10.0, sl * 1.4)
            conf = 70 + min(15, int((48.0 - rsi) * 0.6))
            return cls._mk_decision("OPEN_SHORT", sl, tp, conf, "breakdown-short")

        return None
