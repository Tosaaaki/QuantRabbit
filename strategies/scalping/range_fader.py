from __future__ import annotations

from typing import Dict, Iterable, Optional
import os
import logging

BUY_SUPPORT_DI_GAP_FLOOR = float(os.getenv("RANGE_FADER_BUY_SUPPORT_DI_GAP_FLOOR", "-2.0"))
BUY_SUPPORT_EMA_SLOPE_FLOOR = float(os.getenv("RANGE_FADER_BUY_SUPPORT_EMA_SLOPE_FLOOR", "-0.0012"))
BUY_SUPPORT_MAX_ADX = float(os.getenv("RANGE_FADER_BUY_SUPPORT_MAX_ADX", "32.0"))
BUY_SUPPORT_MOMENTUM_PIPS_CAP = float(os.getenv("RANGE_FADER_BUY_SUPPORT_MOMENTUM_PIPS_CAP", "2.4"))
BUY_SUPPORT_GATE_EXTRA = int(float(os.getenv("RANGE_FADER_BUY_SUPPORT_GATE_EXTRA", "2.0")))
BUY_SUPPORT_CONF_BONUS = int(float(os.getenv("RANGE_FADER_BUY_SUPPORT_CONF_BONUS", "6.0")))
FADE_HEADWIND_RANGE_SCORE_MAX = float(os.getenv("RANGE_FADER_FADE_HEADWIND_RANGE_SCORE_MAX", "0.28"))
FADE_HEADWIND_ADX_MIN = float(os.getenv("RANGE_FADER_FADE_HEADWIND_ADX_MIN", "26.0"))
FADE_HEADWIND_DI_GAP_MIN = float(os.getenv("RANGE_FADER_FADE_HEADWIND_DI_GAP_MIN", "9.0"))
FADE_HEADWIND_EMA_SLOPE_MIN = float(os.getenv("RANGE_FADER_FADE_HEADWIND_EMA_SLOPE_MIN", "0.0020"))
FADE_HEADWIND_MOMENTUM_ATR_MULT = float(os.getenv("RANGE_FADER_FADE_HEADWIND_MOMENTUM_ATR_MULT", "1.05"))
FADE_HEADWIND_MOMENTUM_PIPS_CAP = float(os.getenv("RANGE_FADER_FADE_HEADWIND_MOMENTUM_PIPS_CAP", "3.0"))
FADE_HEADWIND_GATE_EXTRA = int(float(os.getenv("RANGE_FADER_FADE_HEADWIND_GATE_EXTRA", "6.0")))
FLOW_HEADWIND_RANGE_SCORE_MAX = float(os.getenv("RANGE_FADER_FLOW_HEADWIND_RANGE_SCORE_MAX", "0.36"))
FLOW_HEADWIND_ADX_MIN = float(os.getenv("RANGE_FADER_FLOW_HEADWIND_ADX_MIN", "22.0"))
FLOW_HEADWIND_GAP_ATR_MIN = float(os.getenv("RANGE_FADER_FLOW_HEADWIND_GAP_ATR_MIN", "0.55"))
FLOW_HEADWIND_MOMENTUM_ATR_MIN = float(os.getenv("RANGE_FADER_FLOW_HEADWIND_MOMENTUM_ATR_MIN", "0.70"))
FLOW_HEADWIND_CLOSE_POS_MIN = float(os.getenv("RANGE_FADER_FLOW_HEADWIND_CLOSE_POS_MIN", "0.58"))
FLOW_HEADWIND_SHORT_EXTREME_RSI_MIN = float(os.getenv("RANGE_FADER_FLOW_HEADWIND_SHORT_EXTREME_RSI_MIN", "71.0"))
FLOW_HEADWIND_LONG_EXTREME_RSI_MAX = float(os.getenv("RANGE_FADER_FLOW_HEADWIND_LONG_EXTREME_RSI_MAX", "29.0"))
FLOW_HEADWIND_GATE_STEP = int(float(os.getenv("RANGE_FADER_FLOW_HEADWIND_GATE_STEP", "3.0")))
FLOW_HEADWIND_CONF_CUT = float(os.getenv("RANGE_FADER_FLOW_HEADWIND_CONF_CUT", "0.16"))
SETUP_QUALITY_BLOCK_MIN = float(os.getenv("RANGE_FADER_SETUP_QUALITY_BLOCK_MIN", "0.26"))
SHALLOW_PROBE_RANGE_SCORE_MIN = float(os.getenv("RANGE_FADER_SHALLOW_PROBE_RANGE_SCORE_MIN", "0.28"))
SHALLOW_PROBE_RANGE_SCORE_MAX = float(os.getenv("RANGE_FADER_SHALLOW_PROBE_RANGE_SCORE_MAX", "0.36"))
SHALLOW_PROBE_QUALITY_MAX = float(os.getenv("RANGE_FADER_SHALLOW_PROBE_QUALITY_MAX", "0.58"))
SHALLOW_PROBE_MOMENTUM_ATR_MAX = float(os.getenv("RANGE_FADER_SHALLOW_PROBE_MOMENTUM_ATR_MAX", "0.95"))
SHALLOW_PROBE_MOMENTUM_PIPS_CAP = float(os.getenv("RANGE_FADER_SHALLOW_PROBE_MOMENTUM_PIPS_CAP", "1.8"))
SHALLOW_PROBE_BUY_RSI_DIST_MAX = float(os.getenv("RANGE_FADER_SHALLOW_PROBE_BUY_RSI_DIST_MAX", "3.5"))
SHALLOW_PROBE_NEUTRAL_RSI_DIST_MAX = float(os.getenv("RANGE_FADER_SHALLOW_PROBE_NEUTRAL_RSI_DIST_MAX", "6.5"))
FRAGILE_SELL_TRANSITION_QUALITY_MAX = float(os.getenv("RANGE_FADER_FRAGILE_SELL_TRANSITION_QUALITY_MAX", "0.58"))
FRAGILE_SELL_RANGE_P1_QUALITY_MAX = float(os.getenv("RANGE_FADER_FRAGILE_SELL_RANGE_P1_QUALITY_MAX", "0.60"))
FRAGILE_NEUTRAL_LONG_QUALITY_MAX = float(os.getenv("RANGE_FADER_FRAGILE_NEUTRAL_LONG_QUALITY_MAX", "0.50"))


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
    def _float_attr(fac: Dict, key: str, default: float = 0.0) -> float:
        try:
            return float(fac.get(key, default))
        except Exception:
            return default

    @staticmethod
    def _clamp(value: float, floor: float = 0.0, ceil: float = 1.0) -> float:
        return max(floor, min(ceil, value))

    @staticmethod
    def _candles(fac: Dict, count: int) -> list[dict]:
        candles = fac.get("candles") or []
        if not isinstance(candles, Iterable):
            return []
        tail: list[dict] = []
        for candle in list(candles)[-count:]:
            if isinstance(candle, dict):
                tail.append(candle)
        return tail

    @classmethod
    def _ma_gap_pips(cls, fac: Dict) -> Optional[float]:
        ma_fast = fac.get("ma10") if fac.get("ma10") is not None else fac.get("ema20")
        ma_slow = fac.get("ma20") if fac.get("ma20") is not None else fac.get("ema24")
        try:
            if ma_fast is None or ma_slow is None:
                return None
            return (float(ma_fast) - float(ma_slow)) / 0.01
        except Exception:
            return None

    @classmethod
    def _recent_flow_headwind(
        cls,
        fac: Dict,
        side: str,
        *,
        atr_pips: float,
        ema20: float,
    ) -> int:
        candles = cls._candles(fac, 5)
        if len(candles) < 4:
            return 0
        recent = candles[-4:]
        closes: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        close_pos_sum = 0.0
        close_pos_count = 0
        for candle in recent:
            close = cls._float_attr(candle, "close", 0.0)
            open_price = cls._float_attr(candle, "open", close)
            high = cls._float_attr(candle, "high", max(open_price, close))
            low = cls._float_attr(candle, "low", min(open_price, close))
            if high <= 0.0 or low <= 0.0 or high < low:
                return 0
            closes.append(close)
            highs.append(high)
            lows.append(low)
            candle_range = max(high - low, 1e-9)
            close_pos_sum += (close - low) / candle_range
            close_pos_count += 1
        if len(closes) < 4:
            return 0
        avg_close_pos = close_pos_sum / max(1, close_pos_count)
        pressure = 0
        if side == "short":
            rising_closes = sum(1 for idx in range(1, len(closes)) if closes[idx] >= closes[idx - 1] - 1e-9)
            higher_highs = sum(1 for idx in range(1, len(highs)) if highs[idx] >= highs[idx - 1] - 1e-9)
            net_move_pips = max(0.0, (closes[-1] - closes[0]) / 0.01)
            if rising_closes >= 2 and net_move_pips >= max(0.8, atr_pips * FLOW_HEADWIND_MOMENTUM_ATR_MIN):
                pressure += 1
            if higher_highs >= 2 and avg_close_pos >= FLOW_HEADWIND_CLOSE_POS_MIN and closes[-1] > ema20:
                pressure += 1
        else:
            falling_closes = sum(1 for idx in range(1, len(closes)) if closes[idx] <= closes[idx - 1] + 1e-9)
            lower_lows = sum(1 for idx in range(1, len(lows)) if lows[idx] <= lows[idx - 1] + 1e-9)
            net_move_pips = max(0.0, (closes[0] - closes[-1]) / 0.01)
            if falling_closes >= 2 and net_move_pips >= max(0.8, atr_pips * FLOW_HEADWIND_MOMENTUM_ATR_MIN):
                pressure += 1
            if lower_lows >= 2 and avg_close_pos <= (1.0 - FLOW_HEADWIND_CLOSE_POS_MIN) and closes[-1] < ema20:
                pressure += 1
        return min(2, pressure)

    @classmethod
    def _flow_headwind_context(
        cls,
        fac: Dict,
        side: str,
        *,
        close: float,
        ema20: float,
        atr_pips: float,
        adx_val: float,
    ) -> tuple[int, Optional[float], Optional[float], str]:
        ma_gap_pips = cls._ma_gap_pips(fac)
        gap_ratio = abs(ma_gap_pips) / max(1.0, atr_pips) if ma_gap_pips is not None else None
        range_score = cls._float_attr(fac, "range_score", 0.0)
        pressure = cls._recent_flow_headwind(
            fac,
            side,
            atr_pips=atr_pips,
            ema20=ema20,
        )
        if (
            range_score <= FLOW_HEADWIND_RANGE_SCORE_MAX
            and adx_val >= FLOW_HEADWIND_ADX_MIN
            and ma_gap_pips is not None
            and gap_ratio is not None
        ):
            if side == "short" and close > ema20 and ma_gap_pips > 0.0 and gap_ratio >= FLOW_HEADWIND_GAP_ATR_MIN:
                pressure += 1
            elif side == "long" and close < ema20 and ma_gap_pips < 0.0 and gap_ratio >= FLOW_HEADWIND_GAP_ATR_MIN:
                pressure += 1
        pressure = min(2, pressure)
        if gap_ratio is not None and gap_ratio >= 0.85 and adx_val >= FLOW_HEADWIND_ADX_MIN:
            flow_regime = "trend_long" if (ma_gap_pips or 0.0) > 0 else "trend_short"
        elif range_score >= 0.28 and adx_val < 32.0:
            flow_regime = "range_fade"
        else:
            flow_regime = "transition"
        return pressure, ma_gap_pips, gap_ratio, flow_regime

    @staticmethod
    def _buy_supportive_context(
        fac: Dict,
        *,
        close: float,
        ema20: float,
        atr_pips: float,
        momentum_pips: float,
        spread: float | None,
        adx_val: float,
        vol_5m: float,
    ) -> bool:
        plus_di = RangeFader._float_attr(fac, "plus_di", 0.0)
        minus_di = RangeFader._float_attr(fac, "minus_di", 0.0)
        ema_slope_10 = RangeFader._float_attr(fac, "ema_slope_10", 0.0)
        if close > ema20:
            return False
        if adx_val > BUY_SUPPORT_MAX_ADX or vol_5m < 0.85:
            return False
        if spread is not None and spread > 1.0:
            return False
        if (plus_di - minus_di) < BUY_SUPPORT_DI_GAP_FLOOR:
            return False
        if ema_slope_10 < BUY_SUPPORT_EMA_SLOPE_FLOOR:
            return False
        momentum_cap = max(BUY_SUPPORT_MOMENTUM_PIPS_CAP, atr_pips * 1.1)
        return momentum_pips <= momentum_cap

    @staticmethod
    def _fade_headwind_context(
        fac: Dict,
        side: str,
        *,
        close: float,
        ema20: float,
        atr_pips: float,
        momentum_pips: float,
        spread: float | None,
        adx_val: float,
        vol_5m: float,
    ) -> bool:
        range_score_raw = fac.get("range_score")
        if range_score_raw is None:
            return False
        try:
            range_score = float(range_score_raw)
        except Exception:
            return False
        plus_di = RangeFader._float_attr(fac, "plus_di", 0.0)
        minus_di = RangeFader._float_attr(fac, "minus_di", 0.0)
        ema_slope_10 = RangeFader._float_attr(fac, "ema_slope_10", 0.0)
        if range_score > FADE_HEADWIND_RANGE_SCORE_MAX:
            return False
        if adx_val < FADE_HEADWIND_ADX_MIN or vol_5m < 0.85:
            return False
        if spread is not None and spread > 1.0:
            return False
        momentum_cap = max(FADE_HEADWIND_MOMENTUM_PIPS_CAP, atr_pips * FADE_HEADWIND_MOMENTUM_ATR_MULT)
        if momentum_pips > momentum_cap:
            return False
        if side == "short":
            if close <= ema20:
                return False
            if (plus_di - minus_di) < FADE_HEADWIND_DI_GAP_MIN:
                return False
            return ema_slope_10 >= FADE_HEADWIND_EMA_SLOPE_MIN
        if close >= ema20:
            return False
        if (minus_di - plus_di) < FADE_HEADWIND_DI_GAP_MIN:
            return False
        return ema_slope_10 <= -FADE_HEADWIND_EMA_SLOPE_MIN

    @classmethod
    def _fade_setup_quality(
        cls,
        fac: Dict,
        side: str,
        *,
        rsi: float,
        gate: float,
        atr_pips: float,
        momentum_pips: float,
        adx_val: float,
        continuation_pressure: int,
        flow_regime: str,
        gap_ratio: Optional[float],
        supportive_bias: float = 0.0,
    ) -> float:
        range_score = cls._float_attr(fac, "range_score", 0.28)
        atr_norm = max(0.8, atr_pips)
        if side == "short":
            rsi_distance = max(0.0, float(rsi) - float(gate))
        else:
            rsi_distance = max(0.0, float(gate) - float(rsi))
        rsi_component = cls._clamp(rsi_distance / 9.0)
        stretch_component = cls._clamp(momentum_pips / max(1.2, atr_norm * 1.05))
        range_component = cls._clamp((range_score - 0.20) / 0.18)
        gap_component = 0.5
        if gap_ratio is not None:
            gap_component = cls._clamp(1.0 - max(0.0, gap_ratio - 0.40) / 0.70)
        adx_component = cls._clamp(1.0 - max(0.0, adx_val - 20.0) / 18.0)
        regime_bias = 0.04 if flow_regime == "range_fade" else 0.0 if flow_regime == "transition" else -0.12
        pressure_penalty = continuation_pressure * 0.18
        quality = (
            0.05
            + rsi_component * 0.24
            + stretch_component * 0.22
            + range_component * 0.12
            + gap_component * 0.10
            + adx_component * 0.06
            + regime_bias
            + supportive_bias
            - pressure_penalty
        )
        return round(cls._clamp(quality), 3)

    @classmethod
    def _setup_size_mult(
        cls,
        *,
        setup_quality: float,
        continuation_pressure: int,
        flow_regime: str,
    ) -> float:
        regime_bonus = 0.03 if flow_regime == "range_fade" else 0.0 if flow_regime == "transition" else -0.06
        size_mult = 0.62 + setup_quality * 0.55 - continuation_pressure * 0.07 + regime_bonus
        return round(cls._clamp(size_mult, 0.55, 1.10), 3)

    @classmethod
    def _thin_fade_setup(
        cls,
        *,
        setup_quality: float,
        rsi: float,
        gate: float,
        atr_pips: float,
        momentum_pips: float,
        continuation_pressure: int,
    ) -> bool:
        rsi_distance = abs(float(rsi) - float(gate))
        extreme_rsi_distance = max(9.0, atr_pips * 2.4)
        extreme_stretch = momentum_pips >= max(3.4, atr_pips * 1.15)
        if continuation_pressure >= 2:
            return False
        if extreme_stretch or rsi_distance >= extreme_rsi_distance:
            return False
        return setup_quality < SETUP_QUALITY_BLOCK_MIN

    @classmethod
    def _shallow_probe_guard(
        cls,
        fac: Dict,
        *,
        side: str,
        tag_kind: str,
        flow_regime: str,
        continuation_pressure: int,
        setup_quality: float,
        rsi: float,
        gate: float,
        atr_pips: float,
        momentum_pips: float,
        buy_supportive: bool = False,
    ) -> bool:
        if side != "long":
            return False
        if tag_kind not in {"buy-fade", "neutral-fade"}:
            return False
        if buy_supportive or flow_regime != "range_fade" or continuation_pressure != 0:
            return False
        range_score = cls._float_attr(fac, "range_score", 0.0)
        if not (SHALLOW_PROBE_RANGE_SCORE_MIN <= range_score <= SHALLOW_PROBE_RANGE_SCORE_MAX):
            return False
        momentum_cap = max(
            SHALLOW_PROBE_MOMENTUM_PIPS_CAP,
            atr_pips * SHALLOW_PROBE_MOMENTUM_ATR_MAX,
        )
        if momentum_pips > momentum_cap:
            return False
        rsi_distance = abs(float(rsi) - float(gate))
        rsi_distance_cap = (
            max(SHALLOW_PROBE_BUY_RSI_DIST_MAX, atr_pips * 1.25)
            if tag_kind == "buy-fade"
            else max(SHALLOW_PROBE_NEUTRAL_RSI_DIST_MAX, atr_pips * 2.0)
        )
        if rsi_distance > rsi_distance_cap:
            return False
        return setup_quality <= SHALLOW_PROBE_QUALITY_MAX

    @classmethod
    def _fragile_neutral_short_range_guard(
        cls,
        *,
        side: str,
        flow_regime: str,
        continuation_pressure: int,
        setup_quality: float,
        gap_ratio: Optional[float],
        range_score: float,
        atr_pips: float,
        momentum_pips: float,
    ) -> bool:
        if side == "short":
            if flow_regime != "range_fade" or continuation_pressure != 0:
                return False
            if gap_ratio is None or gap_ratio < 0.35 or range_score < 0.45:
                return False
            if momentum_pips >= max(2.2, atr_pips * 1.05):
                return False
            quality_floor = 0.44 if gap_ratio < 0.40 else 0.46
            return setup_quality < quality_floor
        if side != "long":
            return False
        if flow_regime != "range_fade" or continuation_pressure != 0:
            return False
        if range_score < 0.44:
            return False
        if gap_ratio is not None and gap_ratio >= 0.32:
            return False
        if momentum_pips >= max(2.0, atr_pips * 0.95):
            return False
        return setup_quality < FRAGILE_NEUTRAL_LONG_QUALITY_MAX

    @classmethod
    def _fragile_sell_fade_short_guard(
        cls,
        *,
        flow_regime: str,
        continuation_pressure: int,
        setup_quality: float,
        gap_ratio: Optional[float],
        range_score: float,
        atr_pips: float,
        momentum_pips: float,
        rsi: float,
        gate: float,
    ) -> bool:
        rsi_distance = max(0.0, float(rsi) - float(gate))
        if rsi_distance >= max(12.0, atr_pips * 3.0):
            return False
        if flow_regime == "transition":
            if continuation_pressure > 1:
                return False
            if gap_ratio is not None and gap_ratio >= 0.70 and momentum_pips >= max(2.8, atr_pips * 1.15):
                return False
            return (
                setup_quality < FRAGILE_SELL_TRANSITION_QUALITY_MAX
                and momentum_pips < max(2.6, atr_pips * 1.15)
            )
        if flow_regime == "range_fade" and continuation_pressure == 1:
            if range_score < 0.28:
                return False
            if gap_ratio is not None and gap_ratio >= 0.52 and momentum_pips >= max(2.8, atr_pips * 1.10):
                return False
            return (
                setup_quality < FRAGILE_SELL_RANGE_P1_QUALITY_MAX
                and momentum_pips < max(2.6, atr_pips * 1.08)
            )
        return False

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = fac.get("close")
        ema20 = fac.get("ema20")
        rsi = fac.get("rsi")
        atr_pips = fac.get("atr_pips")
        vol_5m = fac.get("vol_5m", 1.0)
        try:
            vol_5m = float(vol_5m)
        except Exception:
            vol_5m = 1.0
        try:
            adx_val = float(fac.get("adx") or 0.0)
        except Exception:
            adx_val = 0.0
        if close is None or ema20 is None or rsi is None:
            RangeFader._log_skip("missing_inputs", close=close, ema20=ema20, rsi=rsi)
            return None

        if atr_pips is None:
            atr = fac.get("atr")
            atr_pips = (atr or 0.0) * 100
        try:
            atr_pips = float(atr_pips)
        except Exception:
            atr_pips = 0.0

        scalp_tactical = os.getenv("SCALP_TACTICAL", "0").strip().lower() not in {"", "0", "false", "no"}
        spread = fac.get("spread_pips")
        try:
            spread = float(spread) if spread is not None else None
        except (TypeError, ValueError):
            spread = None
        low_atr_base = float(os.getenv("RANGE_FADER_ATR_LOW", "0.8"))
        low_atr = min(low_atr_base, 0.9) if scalp_tactical else low_atr_base
        base_high_atr = float(os.getenv("RANGE_FADER_ATR_HIGH", "6.0"))
        hard_atr_cap = float(os.getenv("RANGE_FADER_ATR_HARD", "10.5"))
        high_atr = min(hard_atr_cap, max(4.2, base_high_atr))
        if scalp_tactical:
            high_atr = min(hard_atr_cap, max(high_atr, base_high_atr + 1.2))
        if vol_5m is not None:
            try:
                vol_val = float(vol_5m)
            except Exception:
                vol_val = None
            if vol_val is not None:
                high_atr = min(
                    hard_atr_cap,
                    max(high_atr, 4.2 + max(0.0, (vol_val - 1.0) * 2.4)),
                )
        if adx_val >= 45.0:
            high_atr = min(hard_atr_cap, max(high_atr, 7.2))
        low_atr_dyn = low_atr
        if spread is not None and atr_pips > 0:
            ratio = spread / max(atr_pips, 1e-6)
            # スプレッド負担が軽いときはATR下限を追加で緩める
            if ratio <= 0.35:
                low_atr_dyn = min(low_atr_dyn, low_atr * 0.85)
                if vol_5m is not None and vol_5m >= 0.8:
                    low_atr_dyn = min(low_atr_dyn, max(0.65, low_atr * 0.7))
            if spread <= 1.0 and (vol_5m is None or vol_5m >= 0.85):
                low_atr_dyn = min(low_atr_dyn, max(0.7, low_atr * 0.82))
        low_atr_dyn = max(0.6, low_atr_dyn)
        if atr_pips < low_atr_dyn:
            RangeFader._log_skip(
                "atr_out_of_range",
                atr_pips=round(atr_pips, 3),
                low=round(low_atr_dyn, 3),
                high=high_atr,
            )
            return None
        if atr_pips > hard_atr_cap:
            RangeFader._log_skip(
                "atr_out_of_range",
                atr_pips=round(atr_pips, 3),
                low=low_atr,
                high=hard_atr_cap,
            )
            return None
        if vol_5m < 0.4 or vol_5m > 3.0:
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
        drift_cap = max(3.0, min(6.5, atr_pips * 2.5))
        if momentum_pips > drift_cap:
            RangeFader._log_skip(
                "too_far_from_mean",
                momentum_pips=round(momentum_pips, 3),
                drift_cap=round(drift_cap, 3),
            )
            return None

        # RSIゲートを広げ、ニュートラル近辺でもフェードを許容する
        long_gate = 47 if scalp_tactical else 45
        short_gate = 53 if scalp_tactical else 55
        if atr_pips <= 1.6 or vol_5m < 1.0:
            long_gate += 2  # 低ボラ時は門を広げて約定機会を増やす
            short_gate -= 2

        fast_cut = max(6.0, atr_pips * 0.9)
        fast_cut_time = max(60.0, atr_pips * 15.0)
        buy_supportive = RangeFader._buy_supportive_context(
            fac,
            close=float(close),
            ema20=float(ema20),
            atr_pips=atr_pips,
            momentum_pips=momentum_pips,
            spread=spread,
            adx_val=adx_val,
            vol_5m=vol_5m,
        )
        long_headwind = RangeFader._fade_headwind_context(
            fac,
            "long",
            close=float(close),
            ema20=float(ema20),
            atr_pips=atr_pips,
            momentum_pips=momentum_pips,
            spread=spread,
            adx_val=adx_val,
            vol_5m=vol_5m,
        )
        short_headwind = RangeFader._fade_headwind_context(
            fac,
            "short",
            close=float(close),
            ema20=float(ema20),
            atr_pips=atr_pips,
            momentum_pips=momentum_pips,
            spread=spread,
            adx_val=adx_val,
            vol_5m=vol_5m,
        )
        long_flow_pressure, long_ma_gap_pips, long_gap_ratio, long_flow_regime = RangeFader._flow_headwind_context(
            fac,
            "long",
            close=float(close),
            ema20=float(ema20),
            atr_pips=atr_pips,
            adx_val=adx_val,
        )
        short_flow_pressure, short_ma_gap_pips, short_gap_ratio, short_flow_regime = RangeFader._flow_headwind_context(
            fac,
            "short",
            close=float(close),
            ema20=float(ema20),
            atr_pips=atr_pips,
            adx_val=adx_val,
        )
        buy_long_gate = long_gate + (BUY_SUPPORT_GATE_EXTRA if buy_supportive else 0)
        if long_headwind:
            buy_long_gate -= FADE_HEADWIND_GATE_EXTRA
        if short_headwind:
            short_gate += FADE_HEADWIND_GATE_EXTRA
        if long_flow_pressure > 0:
            buy_long_gate -= FLOW_HEADWIND_GATE_STEP * long_flow_pressure
        if short_flow_pressure > 0:
            short_gate += FLOW_HEADWIND_GATE_STEP * short_flow_pressure

        confidence_scale = 1.0
        long_conf_scale = 1.0
        short_conf_scale = 1.0
        high_atr_profile = atr_pips > high_atr
        if high_atr_profile:
            confidence_scale *= max(0.55, 1.0 - max(0.0, (atr_pips - high_atr)) * 0.06)
        if atr_pips > 3.0 or momentum_pips > drift_cap * 0.8:
            confidence_scale *= 0.75
        if long_flow_pressure > 0:
            long_conf_scale *= max(0.55, 1.0 - FLOW_HEADWIND_CONF_CUT * long_flow_pressure)
        if short_flow_pressure > 0:
            short_conf_scale *= max(0.55, 1.0 - FLOW_HEADWIND_CONF_CUT * short_flow_pressure)
        confidence_scale = max(0.45, min(confidence_scale, 1.0))

        if rsi <= buy_long_gate:
            if long_flow_pressure >= 2 and rsi > FLOW_HEADWIND_LONG_EXTREME_RSI_MAX:
                RangeFader._log_skip(
                    "flow_headwind_long",
                    continuation_pressure=long_flow_pressure,
                    flow_regime=long_flow_regime,
                    rsi=round(float(rsi), 3),
                    ma_gap_pips=round(long_ma_gap_pips, 3) if long_ma_gap_pips is not None else None,
                    gap_ratio=round(long_gap_ratio, 3) if long_gap_ratio is not None else None,
                )
                return None
            long_setup_quality = RangeFader._fade_setup_quality(
                fac,
                "long",
                rsi=float(rsi),
                gate=float(buy_long_gate),
                atr_pips=atr_pips,
                momentum_pips=momentum_pips,
                adx_val=adx_val,
                continuation_pressure=long_flow_pressure,
                flow_regime=long_flow_regime,
                gap_ratio=long_gap_ratio,
                supportive_bias=0.08 if buy_supportive else 0.0,
            )
            if RangeFader._thin_fade_setup(
                setup_quality=long_setup_quality,
                rsi=float(rsi),
                gate=float(buy_long_gate),
                atr_pips=atr_pips,
                momentum_pips=momentum_pips,
                continuation_pressure=long_flow_pressure,
            ):
                RangeFader._log_skip(
                    "thin_fade_setup_long",
                    setup_quality=round(long_setup_quality, 3),
                    continuation_pressure=long_flow_pressure,
                    flow_regime=long_flow_regime,
                    rsi=round(float(rsi), 3),
                    gate=round(float(buy_long_gate), 3),
                    momentum_pips=round(momentum_pips, 3),
                )
                return None
            long_setup_size_mult = RangeFader._setup_size_mult(
                setup_quality=long_setup_quality,
                continuation_pressure=long_flow_pressure,
                flow_regime=long_flow_regime,
            )
            if high_atr_profile:
                sl = max(3.0, min(6.5, atr_pips * 0.95))
                tp = max(sl * 1.25, min(8.5, atr_pips * 1.35))
            elif scalp_tactical:
                sl = max(2.0, min(3.2, atr_pips * 1.2))
                tp = max(2.0, min(3.4, atr_pips * 1.25))
            else:
                sl = max(3.0, min(4.8, atr_pips * 1.5))
                tp = max(2.8, min(4.2, atr_pips * 1.35))
            eta_bonus = 0.0
            rsi_eta_up = fac.get("rsi_eta_upper_min")
            if rsi_eta_up is not None:
                eta_bonus = max(0.0, min(8.0, (6.0 - min(6.0, rsi_eta_up)) * 1.0))
            confidence = int(
                min(
                    90,
                    max(
                        45,
                        (38 - rsi) * 2.6 + vol_5m * 5.5 + eta_bonus,
                    ),
                )
            )
            tag = f"{RangeFader.name}-buy-supportive" if buy_supportive and rsi > long_gate else f"{RangeFader.name}-buy-fade"
            tag_kind = tag.split("-", 1)[-1]
            if RangeFader._shallow_probe_guard(
                fac,
                side="long",
                tag_kind=tag_kind,
                flow_regime=long_flow_regime,
                continuation_pressure=long_flow_pressure,
                setup_quality=long_setup_quality,
                rsi=float(rsi),
                gate=float(buy_long_gate),
                atr_pips=atr_pips,
                momentum_pips=momentum_pips,
                buy_supportive=buy_supportive,
            ):
                RangeFader._log_skip(
                    "shallow_probe_guard_long",
                    tag=tag_kind,
                    setup_quality=round(long_setup_quality, 3),
                    flow_regime=long_flow_regime,
                    continuation_pressure=long_flow_pressure,
                    range_score=round(RangeFader._float_attr(fac, "range_score", 0.0), 3),
                    rsi=round(float(rsi), 3),
                    gate=round(float(buy_long_gate), 3),
                    momentum_pips=round(momentum_pips, 3),
                )
                return None
            final_confidence = int(confidence * confidence_scale * long_conf_scale)
            if tag.endswith("buy-supportive"):
                final_confidence = min(90, final_confidence + BUY_SUPPORT_CONF_BONUS)
            return _attach_kill({
                "action": "OPEN_LONG",
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": final_confidence,
                "fast_cut_pips": round(fast_cut, 2),
                "fast_cut_time_sec": int(fast_cut_time),
                "fast_cut_hard_mult": 1.6,
                "tag": tag,
                "continuation_pressure": long_flow_pressure,
                "flow_regime": long_flow_regime,
                "ma_gap_pips": round(long_ma_gap_pips, 3) if long_ma_gap_pips is not None else None,
                "gap_ratio": round(long_gap_ratio, 3) if long_gap_ratio is not None else None,
                "setup_quality": long_setup_quality,
                "setup_size_mult": long_setup_size_mult,
                "setup_fingerprint": f"{RangeFader.name}|long|{tag.split('-', 1)[-1]}|{long_flow_regime}|p{long_flow_pressure}",
            })

        if rsi >= short_gate:
            if short_flow_pressure >= 2 and rsi < FLOW_HEADWIND_SHORT_EXTREME_RSI_MIN:
                RangeFader._log_skip(
                    "flow_headwind_short",
                    continuation_pressure=short_flow_pressure,
                    flow_regime=short_flow_regime,
                    rsi=round(float(rsi), 3),
                    ma_gap_pips=round(short_ma_gap_pips, 3) if short_ma_gap_pips is not None else None,
                    gap_ratio=round(short_gap_ratio, 3) if short_gap_ratio is not None else None,
                )
                return None
            short_setup_quality = RangeFader._fade_setup_quality(
                fac,
                "short",
                rsi=float(rsi),
                gate=float(short_gate),
                atr_pips=atr_pips,
                momentum_pips=momentum_pips,
                adx_val=adx_val,
                continuation_pressure=short_flow_pressure,
                flow_regime=short_flow_regime,
                gap_ratio=short_gap_ratio,
            )
            if RangeFader._thin_fade_setup(
                setup_quality=short_setup_quality,
                rsi=float(rsi),
                gate=float(short_gate),
                atr_pips=atr_pips,
                momentum_pips=momentum_pips,
                continuation_pressure=short_flow_pressure,
            ):
                RangeFader._log_skip(
                    "thin_fade_setup_short",
                    setup_quality=round(short_setup_quality, 3),
                    continuation_pressure=short_flow_pressure,
                    flow_regime=short_flow_regime,
                    rsi=round(float(rsi), 3),
                    gate=round(float(short_gate), 3),
                    momentum_pips=round(momentum_pips, 3),
                )
                return None
            short_setup_size_mult = RangeFader._setup_size_mult(
                setup_quality=short_setup_quality,
                continuation_pressure=short_flow_pressure,
                flow_regime=short_flow_regime,
            )
            if RangeFader._fragile_sell_fade_short_guard(
                flow_regime=short_flow_regime,
                continuation_pressure=short_flow_pressure,
                setup_quality=short_setup_quality,
                gap_ratio=short_gap_ratio,
                range_score=RangeFader._float_attr(fac, "range_score", 0.0),
                atr_pips=atr_pips,
                momentum_pips=momentum_pips,
                rsi=float(rsi),
                gate=float(short_gate),
            ):
                RangeFader._log_skip(
                    "fragile_sell_fade_short",
                    setup_quality=round(short_setup_quality, 3),
                    continuation_pressure=short_flow_pressure,
                    flow_regime=short_flow_regime,
                    range_score=round(RangeFader._float_attr(fac, "range_score", 0.0), 3),
                    gap_ratio=round(short_gap_ratio, 3) if short_gap_ratio is not None else None,
                    momentum_pips=round(momentum_pips, 3),
                )
                return None
            if high_atr_profile:
                sl = max(3.0, min(6.5, atr_pips * 0.95))
                tp = max(sl * 1.25, min(8.5, atr_pips * 1.35))
            elif scalp_tactical:
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
                "confidence": int(confidence * confidence_scale * short_conf_scale),
                "fast_cut_pips": round(fast_cut, 2),
                "fast_cut_time_sec": int(fast_cut_time),
                "fast_cut_hard_mult": 1.6,
                "tag": f"{RangeFader.name}-sell-fade",
                "continuation_pressure": short_flow_pressure,
                "flow_regime": short_flow_regime,
                "ma_gap_pips": round(short_ma_gap_pips, 3) if short_ma_gap_pips is not None else None,
                "gap_ratio": round(short_gap_ratio, 3) if short_gap_ratio is not None else None,
                "setup_quality": short_setup_quality,
                "setup_size_mult": short_setup_size_mult,
                "setup_fingerprint": f"{RangeFader.name}|short|sell-fade|{short_flow_regime}|p{short_flow_pressure}",
            })

        # ニュートラルRSIでもEMAからの乖離方向にフェードを打つ
        neutral_side = "short" if close > ema20 else "long"
        if neutral_side == "short" and short_headwind:
            RangeFader._log_skip(
                "neutral_headwind_short",
                range_score=round(float(fac.get("range_score") or 0.0), 3),
                adx=round(adx_val, 3),
                di_gap=round(RangeFader._float_attr(fac, "plus_di", 0.0) - RangeFader._float_attr(fac, "minus_di", 0.0), 3),
            )
            return None
        if neutral_side == "short" and short_flow_pressure > 0:
            RangeFader._log_skip(
                "neutral_flow_headwind_short",
                continuation_pressure=short_flow_pressure,
                flow_regime=short_flow_regime,
                ma_gap_pips=round(short_ma_gap_pips, 3) if short_ma_gap_pips is not None else None,
                gap_ratio=round(short_gap_ratio, 3) if short_gap_ratio is not None else None,
            )
            return None
        if neutral_side == "long" and long_headwind:
            RangeFader._log_skip(
                "neutral_headwind_long",
                range_score=round(float(fac.get("range_score") or 0.0), 3),
                adx=round(adx_val, 3),
                di_gap=round(RangeFader._float_attr(fac, "minus_di", 0.0) - RangeFader._float_attr(fac, "plus_di", 0.0), 3),
            )
            return None
        if neutral_side == "long" and long_flow_pressure > 0:
            RangeFader._log_skip(
                "neutral_flow_headwind_long",
                continuation_pressure=long_flow_pressure,
                flow_regime=long_flow_regime,
                ma_gap_pips=round(long_ma_gap_pips, 3) if long_ma_gap_pips is not None else None,
                gap_ratio=round(long_gap_ratio, 3) if long_gap_ratio is not None else None,
            )
            return None
        sl = max(2.4, min(5.0, atr_pips * 1.2))
        tp = max(sl * 1.2, min(6.0, atr_pips * 1.6))
        conf_base = 48 + min(12.0, abs(momentum_pips) * 1.8)
        neutral_conf_scale = short_conf_scale if neutral_side == "short" else long_conf_scale
        confidence = int(max(40, min(85, conf_base * confidence_scale * neutral_conf_scale)))
        neutral_flow_regime = short_flow_regime if neutral_side == "short" else long_flow_regime
        neutral_flow_pressure = short_flow_pressure if neutral_side == "short" else long_flow_pressure
        neutral_ma_gap_pips = short_ma_gap_pips if neutral_side == "short" else long_ma_gap_pips
        neutral_gap_ratio = short_gap_ratio if neutral_side == "short" else long_gap_ratio
        neutral_gate = short_gate if neutral_side == "short" else buy_long_gate
        neutral_setup_quality = RangeFader._fade_setup_quality(
            fac,
            neutral_side,
            rsi=float(rsi),
            gate=float(neutral_gate),
            atr_pips=atr_pips,
            momentum_pips=momentum_pips,
            adx_val=adx_val,
            continuation_pressure=neutral_flow_pressure,
            flow_regime=neutral_flow_regime,
            gap_ratio=neutral_gap_ratio,
        )
        if RangeFader._fragile_neutral_short_range_guard(
            side=neutral_side,
            flow_regime=neutral_flow_regime,
            continuation_pressure=neutral_flow_pressure,
            setup_quality=neutral_setup_quality,
            gap_ratio=neutral_gap_ratio,
            range_score=RangeFader._float_attr(fac, "range_score", 0.0),
            atr_pips=atr_pips,
            momentum_pips=momentum_pips,
        ):
            RangeFader._log_skip(
                "fragile_neutral_range",
                setup_quality=round(neutral_setup_quality, 3),
                continuation_pressure=neutral_flow_pressure,
                flow_regime=neutral_flow_regime,
                range_score=round(RangeFader._float_attr(fac, "range_score", 0.0), 3),
                gap_ratio=round(neutral_gap_ratio, 3) if neutral_gap_ratio is not None else None,
                momentum_pips=round(momentum_pips, 3),
            )
            return None
        if RangeFader._thin_fade_setup(
            setup_quality=neutral_setup_quality,
            rsi=float(rsi),
            gate=float(neutral_gate),
            atr_pips=atr_pips,
            momentum_pips=momentum_pips,
            continuation_pressure=neutral_flow_pressure,
        ):
            RangeFader._log_skip(
                f"thin_fade_setup_{neutral_side}",
                setup_quality=round(neutral_setup_quality, 3),
                continuation_pressure=neutral_flow_pressure,
                flow_regime=neutral_flow_regime,
                rsi=round(float(rsi), 3),
                momentum_pips=round(momentum_pips, 3),
            )
            return None
        if RangeFader._shallow_probe_guard(
            fac,
            side=neutral_side,
            tag_kind="neutral-fade",
            flow_regime=neutral_flow_regime,
            continuation_pressure=neutral_flow_pressure,
            setup_quality=neutral_setup_quality,
            rsi=float(rsi),
            gate=float(neutral_gate),
            atr_pips=atr_pips,
            momentum_pips=momentum_pips,
        ):
            RangeFader._log_skip(
                f"shallow_probe_guard_{neutral_side}",
                tag="neutral-fade",
                setup_quality=round(neutral_setup_quality, 3),
                flow_regime=neutral_flow_regime,
                continuation_pressure=neutral_flow_pressure,
                range_score=round(RangeFader._float_attr(fac, "range_score", 0.0), 3),
                rsi=round(float(rsi), 3),
                gate=round(float(neutral_gate), 3),
                momentum_pips=round(momentum_pips, 3),
            )
            return None
        neutral_setup_size_mult = RangeFader._setup_size_mult(
            setup_quality=neutral_setup_quality,
            continuation_pressure=neutral_flow_pressure,
            flow_regime=neutral_flow_regime,
        )
        return _attach_kill({
            "action": "OPEN_SHORT" if neutral_side == "short" else "OPEN_LONG",
            "sl_pips": round(sl, 2),
            "tp_pips": round(tp, 2),
            "confidence": confidence,
            "fast_cut_pips": round(max(5.5, sl * 0.9), 2),
            "fast_cut_time_sec": int(max(60.0, atr_pips * 12.0)),
            "fast_cut_hard_mult": 1.6,
            "tag": f"{RangeFader.name}-neutral-fade",
            "continuation_pressure": neutral_flow_pressure,
            "flow_regime": neutral_flow_regime,
            "ma_gap_pips": round(neutral_ma_gap_pips, 3) if neutral_ma_gap_pips is not None else None,
            "gap_ratio": round(neutral_gap_ratio, 3) if neutral_gap_ratio is not None else None,
            "setup_quality": neutral_setup_quality,
            "setup_size_mult": neutral_setup_size_mult,
            "setup_fingerprint": f"{RangeFader.name}|{neutral_side}|neutral-fade|{neutral_flow_regime}|p{neutral_flow_pressure}",
        })
