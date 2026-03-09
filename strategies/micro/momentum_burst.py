from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

from analysis.ma_projection import compute_ma_projection, MACrossProjection

# Helper thresholds for directional quality
PIP = 0.01
MIN_GAP_TREND = 0.20
MIN_ADX = 14.0
MIN_ATR = 0.8
VOL_MIN = 0.5
RSI_LONG_MIN = 54
RSI_LONG_MAX = 70
RSI_SHORT_MIN = 30
RSI_SHORT_MAX = 46
DRIFT_PIPS_FLOOR = -0.5  # block longs if short-term drift is negative
DRIFT_PIPS_CEIL = 0.5    # block shorts if short-term drift is positive
SPREAD_PIPS_MAX = 1.2    # hard cap; additionally scaled by ATR below
REACCEL_EMA_DIST_PIPS = 2.0
REACCEL_DI_GAP = 6.0
REACCEL_ROC5_MIN = 0.02
TREND_SNAPSHOT_ADX_MIN = 22.0
TREND_SNAPSHOT_GAP_MIN = 8.0
ENTRY_SL_MIN_PIPS = 2.4
ENTRY_SL_ATR_MULT = 1.15
ENTRY_TP_SL_MULT = 1.6
ENTRY_TP_ATR_BUFFER_MULT = 0.85
RANGE_SCORE_SOFT_MAX = float(os.getenv("MOMENTUMBURST_RANGE_SCORE_SOFT_MAX", "0.34"))
CHOP_SCORE_SOFT_MAX = float(os.getenv("MOMENTUMBURST_CHOP_SCORE_SOFT_MAX", "0.58"))
CONTEXT_CONF_PENALTY_MAX = float(os.getenv("MOMENTUMBURST_CONTEXT_CONF_PENALTY_MAX", "18.0"))
CONTEXT_BLOCK_THRESHOLD = float(os.getenv("MOMENTUMBURST_CONTEXT_BLOCK_THRESHOLD", "0.92"))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


def _annotate_reaccel(signal: Dict, *, direction: str, reaccel: bool) -> Dict:
    entry_mode = "reaccel" if reaccel else "trend"
    momentum_meta = {
        "direction": direction,
        "entry_mode": entry_mode,
        "reaccel": bool(reaccel),
    }
    notes = dict(signal.get("notes") or {})
    notes["momentum_burst"] = momentum_meta
    signal["notes"] = notes
    metadata = dict(signal.get("metadata") or {})
    metadata["momentum_burst"] = momentum_meta
    signal["metadata"] = metadata
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
    def _indicator_quality_ok(
        direction: str,
        fac: Dict,
        close: float,
        ema20: float,
        gap_pips: float,
        atr_pips: float,
    ) -> bool:
        ema_dist_pips = abs(close - ema20) / PIP
        stretch_threshold = max(
            REACCEL_EMA_DIST_PIPS + 0.5,
            atr_pips * 0.55,
            abs(gap_pips) * 1.25,
        )
        if ema_dist_pips <= stretch_threshold:
            return True

        plus_di = MomentumBurstMicro._attr(fac, "plus_di", 0.0)
        minus_di = MomentumBurstMicro._attr(fac, "minus_di", 0.0)
        roc5 = MomentumBurstMicro._attr(fac, "roc5", 0.0)
        ema_slope_10 = MomentumBurstMicro._attr(fac, "ema_slope_10", 0.0)
        rsi = MomentumBurstMicro._attr(fac, "rsi", 50.0)

        if direction == "long":
            di_gap = plus_di - minus_di
            roc_push = roc5
            slope_push = ema_slope_10
            directional_rsi = rsi
        elif direction == "short":
            di_gap = minus_di - plus_di
            roc_push = -roc5
            slope_push = -ema_slope_10
            directional_rsi = 100.0 - rsi
        else:
            return True

        if (
            di_gap < REACCEL_DI_GAP - 1.0
            or roc_push < REACCEL_ROC5_MIN * 0.8
            or slope_push <= 0.0
        ):
            return False

        if directional_rsi <= 68.0:
            return True

        return (
            di_gap >= REACCEL_DI_GAP + 4.0
            and roc_push >= REACCEL_ROC5_MIN * 1.35
            and slope_push >= 0.0010
        )

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
    def _trend_snapshot_supports(direction: str, fac: Dict) -> bool:
        snapshot = fac.get("trend_snapshot")
        if not isinstance(snapshot, dict):
            return True
        snap_direction = str(snapshot.get("direction") or "").strip().lower()
        if snap_direction not in {"long", "short"}:
            return True
        if snap_direction == direction:
            return True
        tf = str(snapshot.get("tf") or "").strip().upper()
        if tf not in {"H1", "H4"}:
            return True
        try:
            adx = float(snapshot.get("adx") or 0.0)
        except (TypeError, ValueError):
            adx = 0.0
        try:
            gap_pips = abs(float(snapshot.get("gap_pips") or 0.0))
        except (TypeError, ValueError):
            gap_pips = 0.0
        return not (adx >= TREND_SNAPSHOT_ADX_MIN and gap_pips >= TREND_SNAPSHOT_GAP_MIN)

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
    def _apply_context_tilt(signal: Dict, fac: Dict, *, reaccel: bool) -> Dict | None:
        range_active = bool(fac.get("range_active"))
        range_score = _clamp01(MomentumBurstMicro._attr(fac, "range_score", 0.0))
        chop_active = bool(fac.get("micro_chop_active"))
        chop_score = _clamp01(MomentumBurstMicro._attr(fac, "micro_chop_score", 0.0))
        if reaccel:
            return signal

        context_penalty = 0.0
        if range_active:
            context_penalty = max(context_penalty, max(0.65, range_score))
        elif range_score >= RANGE_SCORE_SOFT_MAX:
            context_penalty = max(context_penalty, range_score)
        if chop_active:
            context_penalty = max(context_penalty, max(0.60, chop_score))
        elif chop_score >= CHOP_SCORE_SOFT_MAX:
            context_penalty = max(context_penalty, chop_score)

        if context_penalty <= 0.0:
            return signal
        if context_penalty >= CONTEXT_BLOCK_THRESHOLD:
            return None

        confidence_before = int(signal.get("confidence") or 0)
        penalty = int(round(CONTEXT_CONF_PENALTY_MAX * context_penalty))
        confidence_after = max(40, confidence_before - penalty)
        signal["confidence"] = confidence_after
        signal["entry_probability"] = round(max(0.05, confidence_after / 100.0), 3)
        notes = dict(signal.get("notes") or {})
        notes["context_tilt"] = {
            "range_active": range_active,
            "range_score": round(range_score, 3),
            "chop_active": chop_active,
            "chop_score": round(chop_score, 3),
            "penalty": penalty,
        }
        signal["notes"] = notes
        return signal

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
            # Live RCA showed loser clips still running deeper than realized winner size.
            # Keep the ATR-anchored shape, but tighten it modestly without touching shared risk logic.
            sl = max(ENTRY_SL_MIN_PIPS, atr_pips * ENTRY_SL_ATR_MULT)
            tp = max(sl * ENTRY_TP_SL_MULT, sl + atr_pips * ENTRY_TP_ATR_BUFFER_MULT)
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

        if (
            (gap_pips >= MIN_GAP_TREND or long_reaccel)
            and adx >= MIN_ADX
            and close > ema20 + 0.0015
            and drift_pips > DRIFT_PIPS_FLOOR
            and MomentumBurstMicro._mtf_supports("long", fac)
            and MomentumBurstMicro._trend_snapshot_supports("long", fac)
            and (MomentumBurstMicro._price_action_direction(candles, "long") or long_reaccel)
        ):
            if (
                RSI_LONG_MIN <= rsi < RSI_LONG_MAX
                and MomentumBurstMicro._indicator_quality_ok(
                    "long",
                    fac,
                    float(close),
                    float(ema20),
                    gap_pips,
                    atr_pips,
                )
            ):
                return MomentumBurstMicro._apply_context_tilt(
                    _annotate_reaccel(
                        _build_signal("OPEN_LONG", gap_pips),
                        direction="long",
                        reaccel=long_reaccel,
                    ),
                    fac,
                    reaccel=long_reaccel,
                )

        if (
            (gap_pips <= -MIN_GAP_TREND or short_reaccel)
            and adx >= MIN_ADX
            and close < ema20 - 0.0015
            and drift_pips < DRIFT_PIPS_CEIL
            and MomentumBurstMicro._mtf_supports("short", fac)
            and MomentumBurstMicro._trend_snapshot_supports("short", fac)
            and (MomentumBurstMicro._price_action_direction(candles, "short") or short_reaccel)
        ):
            if (
                RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX
                and MomentumBurstMicro._indicator_quality_ok(
                    "short",
                    fac,
                    float(close),
                    float(ema20),
                    gap_pips,
                    atr_pips,
                )
            ):
                return MomentumBurstMicro._apply_context_tilt(
                    _annotate_reaccel(
                        _build_signal("OPEN_SHORT", gap_pips),
                        direction="short",
                        reaccel=short_reaccel,
                    ),
                    fac,
                    reaccel=short_reaccel,
                )

        return None
