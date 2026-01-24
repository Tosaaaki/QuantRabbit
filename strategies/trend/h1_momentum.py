from __future__ import annotations

import os
from typing import Dict, Optional

from indicators.factor_cache import all_factors

PIP = 0.01


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _opt_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _trend_from_ma(
    fac: Optional[Dict[str, float]],
) -> tuple[Optional[str], float, bool]:
    if not fac:
        return None, 0.0, False
    ma10 = fac.get("ma10")
    ma20 = fac.get("ma20")
    if ma10 is None or ma20 is None:
        return None, 0.0, False
    try:
        ma10_val = float(ma10)
        ma20_val = float(ma20)
    except (TypeError, ValueError):
        return None, 0.0, False
    gap_pips = abs(ma10_val - ma20_val) / PIP
    direction = "long" if ma10_val > ma20_val else "short" if ma10_val < ma20_val else None
    return direction, gap_pips, True


class H1MomentumSwing:
    """
    Intermediate-term (H1) trend follower that bridges macro and micro cadence.

    Aggregates H1 trend strength with M1 overextension checks to deliver
    faster swing entries in the macro pocket.
    """

    name = "H1Momentum"
    pocket = "macro"

    _MIN_ADX = 12.0
    _MIN_GAP_PIPS = 2.8
    _MIN_ATR_PIPS = 1.0
    _MAX_M1_RSI_EXTREME = 80.0
    _MIN_M1_RSI_SUPPORT = 20.0

    @classmethod
    def _tf_factors(
        cls, factors: Dict[str, Dict[str, float]], tf: str
    ) -> Optional[Dict[str, float]]:
        fac = factors.get(tf)
        if not fac:
            return None
        close = fac.get("close")
        ma10 = fac.get("ma10")
        ma20 = fac.get("ma20")
        if close is None or ma10 is None or ma20 is None:
            return None
        return fac

    @classmethod
    def _h1_factors(cls, factors: Optional[Dict[str, Dict[str, float]]] = None) -> Optional[Dict[str, float]]:
        if factors is None:
            factors = all_factors()
        return cls._tf_factors(factors, "H1")

    @classmethod
    def _mtf_guard(cls, factors: Dict[str, Dict[str, float]], direction: str) -> bool:
        if not _env_bool("H1M_MTF_ENABLED", True):
            return True

        h4_required = _env_bool("H1M_MTF_H4_REQUIRED", True)
        h4_min_gap = max(0.0, _env_float("H1M_MTF_H4_MIN_GAP_PIPS", 1.6))
        fac_h4 = cls._tf_factors(factors, "H4")
        h4_dir, h4_gap, h4_ready = _trend_from_ma(fac_h4)
        if h4_required:
            if not h4_ready:
                return False
            if h4_dir != direction or h4_gap < h4_min_gap:
                return False

        if _env_bool("H1M_MTF_D1_VETO", True):
            fac_d1 = cls._tf_factors(factors, "D1")
            d1_dir, d1_gap, d1_ready = _trend_from_ma(fac_d1)
            if d1_ready and d1_dir and d1_dir != direction:
                d1_adx = _safe_float(fac_d1.get("adx")) if fac_d1 else 0.0
                d1_min_gap = max(0.0, _env_float("H1M_MTF_D1_MIN_GAP_PIPS", 3.5))
                d1_min_adx = max(0.0, _env_float("H1M_MTF_D1_MIN_ADX", 20.0))
                if d1_gap >= d1_min_gap or d1_adx >= d1_min_adx:
                    return False

        return True

    @classmethod
    def _candle_guard(cls, fac_h1: Dict[str, float], direction: str) -> bool:
        if not _env_bool("H1M_CANDLE_GUARD_ENABLED", True):
            return True
        candles = fac_h1.get("candles")
        if not isinstance(candles, list) or not candles:
            return True
        last = candles[-1]
        if not isinstance(last, dict):
            return True
        open_ = _opt_float(last.get("open"))
        high_ = _opt_float(last.get("high"))
        low_ = _opt_float(last.get("low"))
        close_ = _opt_float(last.get("close"))
        if open_ is None or high_ is None or low_ is None or close_ is None:
            return True

        range_pips = max(0.0, (high_ - low_) / PIP)
        if range_pips <= 0:
            return True
        body_pips = abs(close_ - open_) / PIP
        last_dir = 1 if close_ > open_ else -1 if close_ < open_ else 0
        desired_dir = 1 if direction == "long" else -1 if direction == "short" else 0

        max_range = _env_float("H1M_CANDLE_MAX_RANGE_PIPS", 90.0)
        if max_range > 0.0 and range_pips >= max_range:
            return False

        if _env_bool("H1M_CANDLE_DIR_REQUIRED", True):
            min_body = _env_float("H1M_CANDLE_DIR_MIN_BODY_PIPS", 2.0)
            if last_dir and desired_dir and last_dir != desired_dir and body_pips >= min_body:
                return False

        return True

    @classmethod
    def check(cls, fac_m1: Dict) -> Optional[Dict]:
        factors = all_factors()
        fac_h1 = cls._h1_factors(factors)
        if not fac_h1:
            return None

        ma10 = _safe_float(fac_h1.get("ma10"))
        ma20 = _safe_float(fac_h1.get("ma20"))
        ema12 = _safe_float(fac_h1.get("ema12"))
        ema24 = _safe_float(fac_h1.get("ema24"))
        adx = _safe_float(fac_h1.get("adx"))
        rsi_h1 = _safe_float(fac_h1.get("rsi"), 50.0)
        atr_pips = _safe_float(
            fac_h1.get("atr_pips"),
            default=_safe_float(fac_h1.get("atr")) * 100.0,
        )
        gap_pips = abs(ma10 - ma20) / PIP

        if adx < cls._MIN_ADX or gap_pips < cls._MIN_GAP_PIPS:
            return None
        if atr_pips < cls._MIN_ATR_PIPS:
            return None

        direction = "long" if ma10 > ma20 else "short" if ma10 < ma20 else None
        if direction is None:
            return None

        if not cls._candle_guard(fac_h1, direction):
            return None
        if not cls._mtf_guard(factors, direction):
            return None

        # EMA alignment confirms momentum slope; require fast over slow post gap.
        if direction == "long" and ema12 <= ema24:
            return None
        if direction == "short" and ema12 >= ema24:
            return None

        # Suppress entries when H1 RSI already at extremes to avoid exhaustion.
        if direction == "long" and rsi_h1 >= 76.5:
            return None
        if direction == "short" and rsi_h1 <= 23.5:
            return None

        rsi_m1 = _safe_float(fac_m1.get("rsi"), 50.0)
        if direction == "long" and rsi_m1 >= cls._MAX_M1_RSI_EXTREME:
            return None
        if direction == "short" and rsi_m1 <= cls._MIN_M1_RSI_SUPPORT:
            return None

        # M1短期ドリフトが逆行しているときは方向精度のためスキップ
        drift_keys = (
            "drift_pips_15m",
            "drift_15m",
            "return_15m_pips",
            "drift_pips_30m",
            "return_30m_pips",
        )
        drift_pips = 0.0
        for k in drift_keys:
            val = fac_m1.get(k)
            if val is None:
                continue
            try:
                drift_pips = float(val)
                break
            except (TypeError, ValueError):
                continue
        if direction == "long" and drift_pips < -1.5:
            return None
        if direction == "short" and drift_pips > 1.5:
            return None

        # Derive a wide "insurance" stop; real exit timing is delegated to ExitManager.
        atr_floor = max(cls._MIN_ATR_PIPS, atr_pips)
        sl_base = atr_floor * 2.1
        sl_cap = 36.0
        sl_pips = round(min(max(18.0, sl_base), sl_cap), 2)

        # Provide a soft TP so the order manager can register protections, but expect ExitManager to manage exits.
        if adx >= 32.0 or gap_pips >= 9.0:
            tp_ratio = 1.05
        elif adx >= 24.0:
            tp_ratio = 0.95
        else:
            tp_ratio = 0.85
        tp_pips = round(max(sl_pips * tp_ratio, atr_floor * 1.3), 2)

        confidence = cls._confidence_score(adx, gap_pips, atr_floor, rsi_h1, direction)

        action = "OPEN_LONG" if direction == "long" else "OPEN_SHORT"
        tag = f"{cls.name}-bull" if direction == "long" else f"{cls.name}-bear"

        return {
            "action": action,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "confidence": confidence,
            "tag": tag,
            "notes": {
                "adx_h1": round(adx, 2),
                "gap_pips": round(gap_pips, 2),
                "atr_pips": round(atr_floor, 2),
                "rsi_h1": round(rsi_h1, 1),
                "insurance_sl": sl_pips,
            },
            "hard_stop_pips": sl_pips,
        }

    @classmethod
    def _confidence_score(
        cls,
        adx: float,
        gap_pips: float,
        atr_pips: float,
        rsi_h1: float,
        direction: str,
    ) -> int:
        base = 52.0
        base += max(0.0, min(18.0, (adx - cls._MIN_ADX) * 0.9))
        base += max(0.0, min(14.0, (gap_pips - cls._MIN_GAP_PIPS) * 0.7))
        base += min(9.0, atr_pips * 0.35)

        if direction == "long":
            rsi_bias = max(0.0, min(8.0, (rsi_h1 - 47.0) * 0.4))
        else:
            rsi_bias = max(0.0, min(8.0, (53.0 - rsi_h1) * 0.4))
        base += rsi_bias

        if atr_pips > 8.5:
            base -= min(10.0, (atr_pips - 8.5) * 0.8)

        return int(max(30.0, min(90.0, base)))


__all__ = ["H1MomentumSwing"]
