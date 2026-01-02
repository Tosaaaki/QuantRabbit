from __future__ import annotations

from typing import Dict
import logging
import os


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
        try:
            vol_5m = float(vol_5m)
        except Exception:
            vol_5m = 0.0
        adx = fac.get("adx", 0.0) or 0.0
        try:
            adx = float(adx)
        except Exception:
            adx = 0.0
        if close is None or ema20 is None or ema50 is None:
            PulseBreak._log_skip("missing_inputs", close=close, ema20=ema20, ema50=ema50)
            return None

        if atr_pips is None:
            atr = fac.get("atr")
            atr_pips = (atr or 0.0) * 100
        try:
            atr_pips = float(atr_pips)
        except Exception:
            atr_pips = 0.0
        spread = fac.get("spread_pips")
        try:
            spread = float(spread) if spread is not None else None
        except Exception:
            spread = None
        atr_floor = 0.9
        if spread is not None:
            if spread <= 0.95 and vol_5m >= 0.75:
                atr_floor = 0.72
            if spread <= 0.8 and vol_5m >= 0.8:
                atr_floor = 0.65

        # ATR が薄いときは vol_5m の閾値をスライドさせる（低ボラ帯でも相対ブレイクを拾う）
        # 超低ボラでも通す: ATR/vol ガードを解除
        vol_thresh = 0.0
        short_enabled_env = os.getenv("PULSE_BREAK_ENABLE_SHORT", "1").strip().lower() not in {"", "0", "false", "no"}
        try:
            short_adx_gate = float(os.getenv("PULSE_BREAK_SHORT_ADX", "28.0"))
        except Exception:
            short_adx_gate = 28.0
        try:
            short_vol_cushion = float(os.getenv("PULSE_BREAK_SHORT_VOL_CUSHION", "0.12"))
        except Exception:
            short_vol_cushion = 0.12
        short_vol_cushion = max(0.0, min(short_vol_cushion, 0.4))

        momentum = close - ema20
        bias = ema20 - ema50
        # モメンタム/ADX 下限も解除してスキップしない

        adx_slope = fac.get("adx_slope_per_bar", 0.0) or 0.0
        atr_slope = fac.get("atr_slope_pips", 0.0) or 0.0
        fast_cut = max(6.0, atr_pips * 0.9)
        fast_cut_time = max(60.0, atr_pips * 15.0)
        def _build_payload(action: str, slope_bonus: float, tag_suffix: str) -> Dict:
            base_conf = 52.0 + abs(momentum + bias) * 6200 + vol_5m * 5.0 + slope_bonus
            confidence = int(max(50.0, min(92.0, base_conf)))
            if atr_pips > 4.5 or abs(momentum) > 0.05:
                confidence = int(confidence * 0.8)
            if action == "OPEN_SHORT":
                confidence = int(confidence * 0.93)
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
            allow_short = short_enabled_env or (adx >= short_adx_gate)
            if not allow_short:
                PulseBreak._log_skip(
                    "short_disabled",
                    momentum=round(momentum, 5),
                    bias=round(bias, 5),
                    adx=round(adx, 3),
                )
                return None
            if adx < short_adx_gate:
                PulseBreak._log_skip(
                    "adx_low_short",
                    adx=round(adx, 3),
                    gate=round(short_adx_gate, 2),
                )
                return None
            if ema100 is not None and ema20 > ema100:
                PulseBreak._log_skip(
                    "ema100_block_short",
                    ema20=round(ema20, 5) if ema20 is not None else None,
                    ema100=round(ema100, 5) if ema100 is not None else None,
                )
                return None
            vol_gate = vol_thresh * (1.0 - short_vol_cushion)
            vol_gate = max(0.78, vol_gate)
            if vol_5m < vol_gate:
                PulseBreak._log_skip(
                    "vol_dyn_low_short",
                    atr_pips=round(atr_pips, 3),
                    vol_5m=round(vol_5m, 3),
                    vol_thresh=round(vol_gate, 3),
                )
                return None
            if adx_slope < 0.02:
                PulseBreak._log_skip("adx_slope_flat_short", adx_slope=round(adx_slope, 4))
                return None
            slope_bonus = max(0.0, min(7.0, adx_slope * 35.0 + max(0.0, atr_slope) * 2.0))
            return _build_payload("OPEN_SHORT", slope_bonus, "momentum-down")

        PulseBreak._log_skip(
            "no_alignment",
            momentum=round(momentum, 5),
            bias=round(bias, 5),
            adx=round(adx, 3),
            vol_5m=round(vol_5m, 3),
        )
        return None
