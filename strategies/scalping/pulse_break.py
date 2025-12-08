from __future__ import annotations

from typing import Dict
import logging


def _attach_kill(signal: Dict) -> Dict:
    tags = []
    raw_tags = signal.get("exit_tags") or signal.get("tags")
    if raw_tags:
        if isinstance(raw_tags, str):
            tags = [raw_tags]
        elif isinstance(raw_tags, (list, tuple)):
            tags = list(raw_tags)
    tags = [t for t in tags if isinstance(t, str)]
    lower = [t.lower() for t in tags]
    if "kill" not in lower:
        tags.append("kill")
    if "fast_cut" not in lower:
        tags.append("fast_cut")
    signal["exit_tags"] = tags
    signal["kill_switch"] = True
    return signal


class PulseBreak:
    name = "PulseBreak"
    pocket = "scalp"

    @staticmethod
    def _log_skip(reason: str, **kwargs) -> None:
        extras = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        logging.info("[STRAT_SKIP_DETAIL] PulseBreak reason=%s %s", reason, extras)

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = fac.get("close")
        ema20 = fac.get("ema20")
        ema50 = fac.get("ema50") or fac.get("ma20")
        ema100 = fac.get("ema100") or fac.get("ma50")
        atr_pips = fac.get("atr_pips")
        vol_5m = fac.get("vol_5m", 1.0) or 0.0
        adx = fac.get("adx", 0.0) or 0.0
        if close is None or ema20 is None or ema50 is None:
            PulseBreak._log_skip("missing_inputs", close=close, ema20=ema20, ema50=ema50)
            return None

        if atr_pips is None:
            atr = fac.get("atr")
            atr_pips = (atr or 0.0) * 100

        # 低ボラ時は閾値を少し緩めるが、スプレッド/ATRのバランスは別途 main 側が管理
        if atr_pips < 2.2 or vol_5m < 1.1:
            PulseBreak._log_skip(
                "vol_or_atr_low",
                atr_pips=round(atr_pips, 3),
                vol_5m=round(vol_5m, 3),
            )
            return None

        momentum = close - ema20
        bias = ema20 - ema50
        if abs(momentum) < 0.0048:
            PulseBreak._log_skip("momentum_small", momentum=round(momentum, 5))
            return None

        if adx < 22.5:
            PulseBreak._log_skip("adx_low", adx=round(adx, 3))
            return None

        adx_slope = fac.get("adx_slope_per_bar", 0.0) or 0.0
        atr_slope = fac.get("atr_slope_pips", 0.0) or 0.0
        fast_cut = max(6.0, atr_pips * 0.9)
        fast_cut_time = max(60.0, atr_pips * 15.0)
        def _build_payload(action: str, slope_bonus: float, tag_suffix: str) -> Dict:
            base_conf = 52.0 + abs(momentum + bias) * 6200 + vol_5m * 5.0 + slope_bonus
            confidence = int(max(50.0, min(92.0, base_conf)))
            if atr_pips > 4.5 or abs(momentum) > 0.05:
                confidence = int(confidence * 0.8)
            tp = max(4.8, min(6.4, atr_pips * 1.9))
            sl = max(3.1, min(tp * 0.7, atr_pips * 1.4))
            return _attach_kill({
                "action": action,
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "fast_cut_pips": round(fast_cut, 2),
                "fast_cut_time_sec": int(fast_cut_time),
                "fast_cut_hard_mult": 1.6,
                "tag": f"{PulseBreak.name}-{tag_suffix}",
            })

        if momentum > 0 and bias > 0.05:
            if ema100 is not None and ema20 < ema100:
                PulseBreak._log_skip(
                    "ema100_block",
                    ema20=round(ema20, 5) if ema20 is not None else None,
                    ema100=round(ema100, 5) if ema100 is not None else None,
                )
                return None
            if adx_slope < 0.05:
                PulseBreak._log_skip("adx_slope_flat", adx_slope=round(adx_slope, 4))
                return None
            slope_bonus = max(0.0, min(7.0, adx_slope * 35.0 + max(0.0, atr_slope) * 2.0))
            return _build_payload("OPEN_LONG", slope_bonus, "momentum-up")

        if momentum < 0 and bias < -0.06:
            # 2024Q4 環境ではショートの期待値が低いため、厳格な条件が整うまで見送り。
            PulseBreak._log_skip("short_disabled", momentum=round(momentum, 5), bias=round(bias, 5))
            return None

        PulseBreak._log_skip(
            "no_alignment",
            momentum=round(momentum, 5),
            bias=round(bias, 5),
            adx=round(adx, 3),
            vol_5m=round(vol_5m, 3),
        )
        return None
