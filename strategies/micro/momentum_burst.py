from __future__ import annotations

from typing import Dict, Optional, Sequence

from analysis.ma_projection import compute_ma_projection, MACrossProjection

# Helper thresholds for directional quality
PIP = 0.01
MIN_GAP_TREND = 0.32
MIN_ADX = 24.0
MIN_ATR = 1.2
VOL_MIN = 0.7
RSI_LONG_MIN = 54
RSI_SHORT_MAX = 46
DRIFT_PIPS_FLOOR = -0.5  # block longs if short-term drift is negative
DRIFT_PIPS_CEIL = 0.5    # block shorts if short-term drift is positive
SPREAD_PIPS_MAX = 1.2    # hard cap; additionally scaled by ATR below


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
        Lightweight check: require recent highs/lows to stair-step in the intended direction.
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
        if direction == "long":
            return all(h2 >= h1 and l2 >= l1 for h1, h2, l1, l2 in zip(highs, highs[1:], lows, lows[1:]))
        if direction == "short":
            return all(h2 <= h1 and l2 <= l1 for h1, h2, l1, l2 in zip(highs, highs[1:], lows, lows[1:]))
        return True

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

        # Guard against wide spreads relative to current volatility
        spread_cap = max(SPREAD_PIPS_MAX, atr_pips * 0.35)
        if spread_pips and spread_pips > spread_cap:
            return None

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
            return _attach_kill({
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

        if (
            gap_pips >= MIN_GAP_TREND
            and adx >= MIN_ADX
            and close > ema20 + 0.0015
            and drift_pips > DRIFT_PIPS_FLOOR
            and MomentumBurstMicro._mtf_supports("long", fac)
            and MomentumBurstMicro._price_action_direction(candles, "long")
        ):
            if rsi >= RSI_LONG_MIN:
                return _build_signal("OPEN_LONG", gap_pips)

        if (
            gap_pips <= -MIN_GAP_TREND
            and adx >= MIN_ADX
            and close < ema20 - 0.0015
            and drift_pips < DRIFT_PIPS_CEIL
            and MomentumBurstMicro._mtf_supports("short", fac)
            and MomentumBurstMicro._price_action_direction(candles, "short")
        ):
            if rsi <= RSI_SHORT_MAX:
                return _build_signal("OPEN_SHORT", gap_pips)

        return None
