from __future__ import annotations

from typing import Dict, Optional, Sequence

from analysis.ma_projection import compute_ma_projection, MACrossProjection

# Helper thresholds for directional quality
PIP = 0.01
MIN_GAP_TREND = 0.20
MIN_ADX = 14.0
MIN_ATR = 0.8
VOL_MIN = 0.5
RSI_LONG_MIN = 54
RSI_SHORT_MAX = 46
DRIFT_PIPS_FLOOR = -0.5  # block longs if short-term drift is negative
DRIFT_PIPS_CEIL = 0.5    # block shorts if short-term drift is positive
SPREAD_PIPS_MAX = 1.2    # hard cap; additionally scaled by ATR below
REACCEL_EMA_DIST_PIPS = 2.0
REACCEL_DI_GAP = 6.0
REACCEL_ROC5_MIN = 0.02


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
    def _price_action_direction(
        candles: Sequence[Dict], direction: str, lookback: int = 4
    ) -> bool:
        """
        Lightweight check: allow one noisy bar while the recent highs/lows still
        mostly stair-step in the intended direction.
        """
        if not candles or len(candles) < lookback:
            return True  # no data; don't block
        recent = candles[-lookback:]
        highs = [c.get("high") or c.get("h") or c.get("H") for c in recent]
        lows = [c.get("low") or c.get("l") or c.get("L") for c in recent]
        try:
            highs = [float(h) for h in highs]
            lows = [float(l) for l in lows]
        except (TypeError, ValueError):
            return True
        transitions = list(zip(highs, highs[1:], lows, lows[1:]))
        required_votes = max(1, len(transitions) - 1)
        if direction == "long":
            votes = sum(
                1 for h1, h2, l1, l2 in transitions if h2 >= h1 and l2 >= l1
            )
            return votes >= required_votes
        if direction == "short":
            votes = sum(
                1 for h1, h2, l1, l2 in transitions if h2 <= h1 and l2 <= l1
            )
            return votes >= required_votes
        return True

    @staticmethod
    def _reaccel_break(
        candles: Sequence[Dict],
        fac: Dict,
        direction: str,
        close: float,
        ema20: float,
    ) -> bool:
        if not candles or len(candles) < 4:
            return False
        recent = candles[-4:]
        prev = recent[:-1]
        try:
            highs = [float(c.get("high") or c.get("h") or c.get("H")) for c in prev]
            lows = [float(c.get("low") or c.get("l") or c.get("L")) for c in prev]
        except (TypeError, ValueError):
            return False
        plus_di = MomentumBurstMicro._attr(fac, "plus_di", 0.0)
        minus_di = MomentumBurstMicro._attr(fac, "minus_di", 0.0)
        roc5 = MomentumBurstMicro._attr(fac, "roc5", 0.0)
        ema_slope_10 = MomentumBurstMicro._attr(fac, "ema_slope_10", 0.0)
        ema_dist_pips = (close - ema20) / PIP
        if direction == "long":
            return (
                close >= max(highs)
                and ema_dist_pips >= REACCEL_EMA_DIST_PIPS
                and plus_di >= minus_di + REACCEL_DI_GAP
                and roc5 >= REACCEL_ROC5_MIN
                and ema_slope_10 > 0.0
            )
        if direction == "short":
            return (
                close <= min(lows)
                and ema_dist_pips <= -REACCEL_EMA_DIST_PIPS
                and minus_di >= plus_di + REACCEL_DI_GAP
                and roc5 <= -REACCEL_ROC5_MIN
                and ema_slope_10 < 0.0
            )
        return False

    @staticmethod
    def _mtf_supports(direction: str, fac: Dict) -> bool:
        """
        Use optional MTF candles if provided to confirm direction.
        Requires at least two frames in agreement to enforce; otherwise allow.
        """
        mtf = fac.get("mtf")
        if not isinstance(mtf, dict):
            return True

        def _proj(candles: Optional[Sequence[Dict]], minutes: float) -> Optional[MACrossProjection]:
            if not candles:
                return None
            try:
                return compute_ma_projection({"candles": candles}, timeframe_minutes=minutes)
            except Exception:
                return None

        frames = [
            ("m5", _proj(mtf.get("candles_m5"), 5.0)),
            ("m15", _proj(mtf.get("candles_m15"), 15.0)),
            ("h1", _proj(mtf.get("candles_h1"), 60.0)),
        ]
        votes = []
        for _, proj in frames:
            if not proj or proj.fast_ma is None or proj.slow_ma is None:
                continue
            if proj.fast_ma > proj.slow_ma:
                votes.append("long")
            elif proj.fast_ma < proj.slow_ma:
                votes.append("short")
        if len(votes) < 2:
            return True  # not enough data to enforce
        agree = sum(1 for v in votes if v == direction)
        oppose = sum(1 for v in votes if v != direction)
        return agree >= 2 and oppose == 0

    @staticmethod
    def _drift_pips(fac: Dict) -> float:
        """
        Try to read a short-horizon drift (15–30m) if available.
        Falls back to 0.0 when not provided to keep backward-compatible behaviour.
        """
        for key in (
            "drift_pips_15m",
            "drift_15m",
            "return_15m_pips",
            "drift_pips_30m",
            "return_30m_pips",
        ):
            val = fac.get(key)
            if val is None:
                continue
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
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
        drift_pips = MomentumBurstMicro._drift_pips(fac)
        spread_pips = MomentumBurstMicro._attr(fac, "spread_pips", 0.0)
        candles = fac.get("candles") or []
        long_reaccel = MomentumBurstMicro._reaccel_break(
            candles,
            fac,
            "long",
            float(close),
            float(ema20),
        )
        short_reaccel = MomentumBurstMicro._reaccel_break(
            candles,
            fac,
            "short",
            float(close),
            float(ema20),
        )

        # Guard against wide spreads relative to current volatility
        spread_cap = max(SPREAD_PIPS_MAX, atr_pips * 0.35)
        if spread_pips and spread_pips > spread_cap:
            return None

        def _build_signal(action: str, bias_pips: float) -> Dict:
            strength = abs(gap_pips)
            # Previous: sl = max(0.9, min(atr * 1.05, 0.45*strength+0.75)) -- the min() cap
            # crushed SL to ~1p when strength was low, causing 83% stop-outs.
            # New: ATR-anchored SL with a meaningful floor; no more strength-based cap.
            sl = max(2.4, atr_pips * 1.25)
            tp = max(sl * 1.6, sl + atr_pips * 0.85)
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
            sig = _attach_kill({
                "action": action,
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "profile": profile,
                "loss_guard_pips": round(sl, 2),
                "target_tp_pips": round(tp, 2),
                "min_hold_sec": round(min_hold, 1),
                "tag": f"{MomentumBurstMicro.name}-{action.lower()}",
            })
            # ソフトガード用メタを添付（ハードSLは使わず fast_cut 相当の情報だけ持たせる）
            if "fast_cut_pips" not in sig:
                sig["fast_cut_pips"] = round(max(6.0, atr_pips * 1.3), 2)
            if "fast_cut_time_sec" not in sig:
                sig["fast_cut_time_sec"] = int(max(300.0, atr_pips * 90.0))
            if "fast_cut_hard_mult" not in sig:
                sig["fast_cut_hard_mult"] = 2.5
            return sig

        def _bull_run_len(candles, lookback: int = 5) -> int:
            closes = []
            for c in candles[-lookback:]:
                try:
                    closes.append(float(c.get("close") or c.get("c") or c.get("C")))
                except Exception:
                    return 0
            run = 0
            for prev, cur in zip(closes, closes[1:]):
                if cur > prev:
                    run += 1
                else:
                    run = 0
            return run

        if (
            (gap_pips >= MIN_GAP_TREND or long_reaccel)
            and adx >= MIN_ADX
            and close > ema20 + 0.0015
            and drift_pips > DRIFT_PIPS_FLOOR
            and MomentumBurstMicro._mtf_supports("long", fac)
            and (MomentumBurstMicro._price_action_direction(candles, "long") or long_reaccel)
        ):
            # 高値追い抑制: RSI過熱・連続陽線が続くときは見送り
            if rsi >= RSI_LONG_MIN and rsi < 70 and _bull_run_len(candles, 5) < 4:
                return _build_signal("OPEN_LONG", gap_pips)

        if (
            (gap_pips <= -MIN_GAP_TREND or short_reaccel)
            and adx >= MIN_ADX
            and close < ema20 - 0.0015
            and drift_pips < DRIFT_PIPS_CEIL
            and MomentumBurstMicro._mtf_supports("short", fac)
            and (MomentumBurstMicro._price_action_direction(candles, "short") or short_reaccel)
        ):
            if rsi <= RSI_SHORT_MAX:
                return _build_signal("OPEN_SHORT", gap_pips)

        return None
