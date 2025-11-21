from __future__ import annotations

from typing import Dict

PIP = 0.01
MIN_ATR = 1.35
MAX_SPREAD = 1.45
MIN_DISLOCATION = 1.4  # pips away from ema20
RSI_LONG_MAX = 38
RSI_SHORT_MIN = 62
VOL_MIN = 0.6


class ImpulseRetraceScalp:
    name = "ImpulseRetrace"
    pocket = "scalp"

    @staticmethod
    def _atr_pips(fac: Dict) -> float:
        atr = fac.get("atr_pips")
        if atr is not None:
            return float(atr)
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
        ema20 = fac.get("ema20") or fac.get("ma20")
        ema10 = fac.get("ema10")
        if close is None or ema20 is None:
            return None
        rsi = fac.get("rsi")
        if rsi is None:
            return None
        try:
            rsi_val = float(rsi)
        except (TypeError, ValueError):
            return None
        atr_pips = ImpulseRetraceScalp._atr_pips(fac)
        if atr_pips < MIN_ATR:
            return None
        spread = fac.get("spread_pips")
        if spread is not None:
            try:
                if float(spread) > MAX_SPREAD:
                    return None
            except (TypeError, ValueError):
                pass
        vol_5m = fac.get("vol_5m")
        try:
            if vol_5m is not None and float(vol_5m) < VOL_MIN:
                return None
        except (TypeError, ValueError):
            return None

        dislocation_pips = (close - ema20) / PIP
        ema_gap_pips = 0.0
        if ema10 is not None:
            ema_gap_pips = (ema10 - ema20) / PIP

        def _build_signal(
            *,
            action: str,
            dist: float,
            rsi_value: float,
        ) -> Dict | None:
            if dist < MIN_DISLOCATION:
                return None
            sl = max(0.75, min(atr_pips * 0.9, dist * 0.55))
            tp = max(sl * 1.35, min(atr_pips * 1.7, sl + dist * 0.65))
            confidence = int(
                max(
                    48,
                    min(
                        95,
                        52
                        + (dist - MIN_DISLOCATION) * 8.0
                        + max(0.0, abs(ema_gap_pips)) * 1.2
                        + max(0.0, (atr_pips - MIN_ATR)) * 2.5,
                    ),
                )
            )
            profile = "impulse_retrace"
            min_hold = max(60.0, min(420.0, tp * 36.0))
            return {
                "action": action,
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "profile": profile,
                "loss_guard_pips": round(sl, 2),
                "target_tp_pips": round(tp, 2),
                "min_hold_sec": round(min_hold, 1),
                "tag": f"{ImpulseRetraceScalp.name}-{action.lower()}",
            }

        if dislocation_pips <= -MIN_DISLOCATION and rsi_val <= RSI_LONG_MAX:
            # oversold spike, look for retrace long
            distance = abs(dislocation_pips)
            return _build_signal(action="OPEN_LONG", dist=distance, rsi_value=rsi_val)

        if dislocation_pips >= MIN_DISLOCATION and rsi_val >= RSI_SHORT_MIN:
            distance = abs(dislocation_pips)
            return _build_signal(action="OPEN_SHORT", dist=distance, rsi_value=rsi_val)

        return None
