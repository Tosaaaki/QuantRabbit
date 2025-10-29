from __future__ import annotations
from typing import Dict, Optional
import json
import os
from pathlib import Path

try:
    from analysis.patterns import NWaveStructure, detect_latest_n_wave
except ModuleNotFoundError:  # pragma: no cover - fallback when optional module not deployed
    NWaveStructure = None  # type: ignore

    def detect_latest_n_wave(*args, **kwargs):  # type: ignore
        return None

_PIP = 0.01
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "scalp_active_params.json"
_PARAM_CACHE: Dict[str, Dict] = {"mtime": None, "data": {}}


def _load_scalper_config() -> Dict:
    try:
        mtime = _CONFIG_PATH.stat().st_mtime
    except FileNotFoundError:
        return {}
    cached_mtime = _PARAM_CACHE.get("mtime")
    if cached_mtime != mtime:
        try:
            with _CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        _PARAM_CACHE["mtime"] = mtime
        _PARAM_CACHE["data"] = data.get("M1Scalper", data.get("m1scalper", {}))
    return _PARAM_CACHE.get("data", {})


def _to_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _candle_body_pips(candle: Dict[str, float]) -> Optional[float]:
    open_px = _to_float(candle.get("open"))
    close_px = _to_float(candle.get("close"))
    if open_px is None or close_px is None:
        return None
    return (close_px - open_px) / _PIP


class M1Scalper:
    name = "M1Scalper"
    pocket = "scalp"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        params = _load_scalper_config()
        fallback_cfg = params.get("fallback", {})
        nwave_cfg = params.get("nwave", {})

        close = _to_float(fac.get("close"))
        ema20 = _to_float(fac.get("ema20"))
        rsi = _to_float(fac.get("rsi"))
        atr = _to_float(fac.get("atr"), 0.02)
        if close is None or ema20 is None or rsi is None:
            return None

        momentum = close - ema20
        # Prefer explicit atr_pips if provided; otherwise convert ATR (price units) to pips
        atr_pips = _to_float(fac.get("atr_pips"))
        if atr_pips is None:
            atr_pips = (atr or 0.0) * 100

        # Tactical mode lowers the ATR floor slightly to allow more triggers
        scalp_tactical = os.getenv("SCALP_TACTICAL", "0").strip().lower() not in {"", "0", "false", "no"}
        atr_floor_default = float(fallback_cfg.get("atr_floor", 1.2))
        atr_floor_tactical = float(fallback_cfg.get("atr_floor_tactical", 0.85))
        atr_floor_tactical = max(0.75, min(atr_floor_tactical, 1.2))
        atr_floor = atr_floor_tactical if scalp_tactical else max(1.05, atr_floor_default)
        if atr_pips < atr_floor:
            return None

        candles = fac.get("candles") or []
        nwave: Optional[NWaveStructure] = None
        min_leg_pips = float(nwave_cfg.get("min_leg_pips", 2.6))
        max_points = int(nwave_cfg.get("max_points", 140))
        if len(candles) >= 40:
            nwave = detect_latest_n_wave(
                candles, window=3, min_leg_pips=min_leg_pips, max_points=max_points
            )

        def _alignment_ok(side: str) -> bool:
            if len(candles) < 2:
                return True
            last_body = _candle_body_pips(candles[-1])
            prev_body = _candle_body_pips(candles[-2])
            if last_body is None or prev_body is None:
                return True
            if side == "long":
                return prev_body <= -0.5 and last_body >= 0.3
            return prev_body >= 0.5 and last_body <= -0.3

        story_levels = fac.get("story_levels") or {}
        d1_levels = story_levels.get("d1", {})
        h4_levels = story_levels.get("h4", {})

        def _level_bias(price: float) -> float:
            candidates = []
            for level in (*d1_levels.values(), *h4_levels.values()):
                val = _to_float(level)
                if val is None:
                    continue
                candidates.append(abs(price - val) / _PIP)
            if not candidates:
                return 0.0
            closest = min(candidates)
            if closest <= 6.0:
                return max(0.0, 1.0 - closest / 6.0)
            return 0.0

        if nwave:
            direction = nwave.direction
            entry_price = round(nwave.entry_price, 3)
            pullback_mult = float(nwave_cfg.get("pullback_mult", 1.8))
            invalid_floor = float(nwave_cfg.get("hard_sl_floor", 6.0))
            target_scale = float(nwave_cfg.get("target_scale", 0.45))
            target_floor = float(nwave_cfg.get("target_floor", 0.9))
            target_cap = float(nwave_cfg.get("target_cap", 1.4))
            invalid_pips = max(invalid_floor, nwave.pullback_pips * pullback_mult)
            target_pips = max(target_floor, min(target_cap, nwave.amplitude_pips * target_scale))
            quality = nwave.quality
            proximity_bias = _level_bias(entry_price)
            base_conf = 55.0 + (min(quality, 2.0) * 20.0)
            base_conf -= proximity_bias * 15.0
            base_conf = max(40.0, min(96.0, base_conf))
            tolerance_default = float(nwave_cfg.get("tolerance_default", 0.26))
            tolerance_tactical = float(nwave_cfg.get("tolerance_tactical", tolerance_default + 0.08))
            tolerance_pips = tolerance_tactical if scalp_tactical else tolerance_default
            hard_sl_mult = float(nwave_cfg.get("hard_sl_atr_mult", 2.6))
            hard_sl = max(invalid_pips, atr_pips * hard_sl_mult, float(nwave_cfg.get("hard_sl_floor", 6.6)))

            if direction == "long":
                if close > entry_price + tolerance_pips * _PIP:
                    return None
                if not _alignment_ok("long"):
                    return None
                return {
                    "action": "OPEN_LONG",
                    "entry_type": "limit",
                    "entry_price": entry_price,
                    "entry_tolerance_pips": tolerance_pips,
                    "limit_expiry_seconds": 70 if scalp_tactical else 90,
                    "sl_pips": round(hard_sl, 2),
                    "tp_pips": round(target_pips, 2),
                    "confidence": int(base_conf),
                    "tag": f"{M1Scalper.name}-nwave-long",
                }

            if close < entry_price - tolerance_pips * _PIP:
                return None
            if not _alignment_ok("short"):
                return None
            return {
                "action": "OPEN_SHORT",
                "entry_type": "limit",
                "entry_price": entry_price,
                "entry_tolerance_pips": tolerance_pips,
                "limit_expiry_seconds": 70 if scalp_tactical else 90,
                "sl_pips": round(hard_sl, 2),
                "tp_pips": round(target_pips, 2),
                "confidence": int(base_conf),
                "tag": f"{M1Scalper.name}-nwave-short",
            }

        # Fallback momentum scalp (market entry)
        if scalp_tactical:
            tp_floor = float(fallback_cfg.get("tp_floor", 0.9))
            tp_cap = float(fallback_cfg.get("tp_cap", 1.4))
            tp_mult = float(fallback_cfg.get("tp_atr_mult", 0.6))
            sl_floor = float(fallback_cfg.get("sl_floor", 6.8))
            sl_mult = float(fallback_cfg.get("sl_atr_mult", 2.7))
            mom_thresh = float(fallback_cfg.get("momentum_thresh", 0.0010))
            wick_min = float(fallback_cfg.get("wick_min", 0.00075))
            tp_dyn = max(tp_floor, min(tp_cap, atr_pips * tp_mult))
            sl_dyn = max(sl_floor, atr_pips * sl_mult)
            prev_candle = candles[-2] if len(candles) >= 2 else (candles[-1] if candles else None)
            bullish_rejection = False
            bearish_rejection = False
            if prev_candle:
                po = _to_float(prev_candle.get("open"))
                pc = _to_float(prev_candle.get("close"))
                ph = _to_float(prev_candle.get("high"))
                pl = _to_float(prev_candle.get("low"))
                if None not in (po, pc, ph, pl):
                    body = pc - po
                    lower_wick = (pc if body >= 0 else po) - pl
                    upper_wick = ph - (pc if body >= 0 else po)
                    bullish_rejection = body >= 0 and lower_wick >= wick_min
                    bearish_rejection = body <= 0 and upper_wick >= wick_min
            if momentum < -mom_thresh and rsi <= 54:
                if not bullish_rejection:
                    return None
                if not _alignment_ok("long"):
                    return None
                conf = int(max(40.0, min(88.0, 52.0 + (abs(momentum) / max(0.0005, atr)) * 28.0)))
                return {
                    "action": "OPEN_LONG",
                    "sl_pips": round(sl_dyn, 2),
                    "tp_pips": round(tp_dyn, 2),
                    "confidence": conf,
                    "tag": f"{M1Scalper.name}-fallback-long",
                }
            if momentum > mom_thresh and rsi >= 46:
                if not bearish_rejection:
                    return None
                if not _alignment_ok("short"):
                    return None
                conf = int(max(40.0, min(88.0, 52.0 + (abs(momentum) / max(0.0005, atr)) * 28.0)))
                return {
                    "action": "OPEN_SHORT",
                    "sl_pips": round(sl_dyn, 2),
                    "tp_pips": round(tp_dyn, 2),
                    "confidence": conf,
                    "tag": f"{M1Scalper.name}-fallback-short",
                }

        return None
