from __future__ import annotations

from typing import Dict
import os
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


class RangeFader:
    name = "RangeFader"
    pocket = "scalp"

    @staticmethod
    def _log_skip(reason: str, **kwargs) -> None:
        extras = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        logging.info("[STRAT_SKIP_DETAIL] RangeFader reason=%s %s", reason, extras)

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = fac.get("close")
        ema20 = fac.get("ema20")
        rsi = fac.get("rsi")
        atr_pips = fac.get("atr_pips")
        vol_5m = fac.get("vol_5m", 1.0)
        if close is None or ema20 is None or rsi is None:
            RangeFader._log_skip("missing_inputs", close=close, ema20=ema20, rsi=rsi)
            return None

        if atr_pips is None:
            atr = fac.get("atr")
            atr_pips = (atr or 0.0) * 100

        scalp_tactical = os.getenv("SCALP_TACTICAL", "0").strip().lower() not in {"", "0", "false", "no"}
        spread = fac.get("spread_pips")
        try:
            spread = float(spread) if spread is not None else None
        except (TypeError, ValueError):
            spread = None
        # In tactical mode broaden acceptance slightly on the low side and cap top-end volatility
        low_atr = 0.9 if scalp_tactical else 1.0
        high_atr = 4.2
        if spread is not None and atr_pips > 0:
            ratio = spread / max(atr_pips, 1e-6)
            # スプレッド負担が軽いときはATR下限を追加で緩める
            if ratio <= 0.35:
                low_atr = max(0.85, low_atr * 0.9)
        if atr_pips < low_atr or atr_pips > high_atr:
            RangeFader._log_skip(
                "atr_out_of_range",
                atr_pips=round(atr_pips, 3),
                low=low_atr,
                high=high_atr,
            )
            return None
        if vol_5m < 0.5 or vol_5m > 2.6:
            RangeFader._log_skip("vol_out_of_range", vol_5m=round(vol_5m, 3))
            return None

        bbw = fac.get("bbw", 0.0) or 0.0
        bbw_eta = fac.get("bbw_squeeze_eta_min")
        # レンジ確度：BBW が小さい／またはさらに縮小方向
        if bbw > 0.35:
            RangeFader._log_skip("bbw_too_wide", bbw=round(bbw, 4))
            return None
        if bbw_eta is not None and bbw_eta > 12.0:
            RangeFader._log_skip("bbw_eta_high", bbw_eta=round(bbw_eta, 3))
            return None

        momentum_pips = abs(close - ema20) / 0.01
        drift_cap = max(2.2, min(5.0, atr_pips * 1.25))
        if momentum_pips > drift_cap:
            RangeFader._log_skip(
                "too_far_from_mean",
                momentum_pips=round(momentum_pips, 3),
                drift_cap=round(drift_cap, 3),
            )
            return None

        # RSIゲートをボラ/セッションに合わせて緩める
        long_gate = 43 if scalp_tactical else 41
        short_gate = 57 if scalp_tactical else 59
        if atr_pips <= 1.6 or vol_5m < 1.0:
            long_gate += 2  # 低ボラ時は門を広げて約定機会を増やす
            short_gate -= 2

        fast_cut = max(6.0, atr_pips * 0.9)
        fast_cut_time = max(60.0, atr_pips * 15.0)

        confidence_scale = 1.0
        if atr_pips > 3.0 or momentum_pips > drift_cap * 0.8:
            confidence_scale = 0.75

        if rsi <= long_gate:
            if scalp_tactical:
                sl = max(2.0, min(3.2, atr_pips * 1.2))
                tp = max(2.0, min(3.4, atr_pips * 1.25))
            else:
                sl = max(3.0, min(4.8, atr_pips * 1.5))
                tp = max(2.8, min(4.2, atr_pips * 1.35))
            eta_bonus = 0.0
            rsi_eta_up = fac.get("rsi_eta_upper_min")
            if rsi_eta_up is not None:
                eta_bonus = max(0.0, min(8.0, (6.0 - min(6.0, rsi_eta_up)) * 1.0))
            confidence = int(min(90, max(45, (38 - rsi) * 2.6 + vol_5m * 5.5 + eta_bonus)))
            return _attach_kill({
                "action": "OPEN_LONG",
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": int(confidence * confidence_scale),
                "fast_cut_pips": round(fast_cut, 2),
                "fast_cut_time_sec": int(fast_cut_time),
                "fast_cut_hard_mult": 1.6,
                "tag": f"{RangeFader.name}-buy-fade",
            })

        if rsi >= short_gate:
            if scalp_tactical:
                sl = max(2.0, min(3.2, atr_pips * 1.2))
                tp = max(2.0, min(3.4, atr_pips * 1.25))
            else:
                sl = max(3.0, min(4.8, atr_pips * 1.5))
                tp = max(2.8, min(4.2, atr_pips * 1.35))
            eta_bonus = 0.0
            rsi_eta_dn = fac.get("rsi_eta_lower_min")
            if rsi_eta_dn is not None:
                eta_bonus = max(0.0, min(8.0, (6.0 - min(6.0, rsi_eta_dn)) * 1.0))
            confidence = int(min(90, max(45, (rsi - 62) * 2.6 + vol_5m * 5.5 + eta_bonus)))
            return _attach_kill({
                "action": "OPEN_SHORT",
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": int(confidence * confidence_scale),
                "fast_cut_pips": round(fast_cut, 2),
                "fast_cut_time_sec": int(fast_cut_time),
                "fast_cut_hard_mult": 1.6,
                "tag": f"{RangeFader.name}-sell-fade",
            })

        RangeFader._log_skip(
            "rsi_neutral",
            rsi=round(rsi, 2),
            long_gate=long_gate,
            short_gate=short_gate,
            atr_pips=round(atr_pips, 3),
            vol_5m=round(vol_5m, 3),
        )
        return None
