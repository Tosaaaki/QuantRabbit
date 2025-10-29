from __future__ import annotations
from typing import Dict, Optional
import json
import os
import logging
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
_LOGGER = logging.getLogger(__name__)


def _log(reason: str, **kwargs: object) -> None:
    if not kwargs:
        _LOGGER.info("[M1SCALPER] %s", reason)
        return
    payload = " ".join(f"{key}={value}" for key, value in kwargs.items())
    _LOGGER.info("[M1SCALPER] %s %s", reason, payload)


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


def _to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no"}
    if value is None:
        return default
    try:
        return bool(value)
    except Exception:
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

        def _fallback_float(name: str, default: float) -> float:
            value = fallback_cfg.get(name, default)
            converted = _to_float(value, default)
            return default if converted is None else float(converted)

        def _nwave_float(name: str, default: float) -> float:
            value = nwave_cfg.get(name, default)
            converted = _to_float(value, default)
            return default if converted is None else float(converted)

        atr_floor_default = _fallback_float("atr_floor", 0.95)
        atr_floor_tactical = _fallback_float(
            "atr_floor_tactical",
            max(0.65, atr_floor_default - 0.25),
        )
        atr_floor_tactical = max(0.55, min(atr_floor_tactical, 1.05))
        atr_floor = atr_floor_tactical if scalp_tactical else max(0.85, atr_floor_default)
        if atr_pips < atr_floor:
            _log(
                "skip_atr_floor",
                atr=round(atr_pips, 3),
                floor=round(atr_floor, 3),
                tactical=scalp_tactical,
            )
            return None

        candles = fac.get("candles") or []
        nwave: Optional[NWaveStructure] = None
        min_leg_pips = _nwave_float("min_leg_pips", 1.8)
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
            pullback_mult = _nwave_float("pullback_mult", 1.6)
            hard_sl_floor = _nwave_float("hard_sl_floor", 4.0)
            target_scale = _nwave_float("target_scale", 0.55)
            target_floor = _nwave_float("target_floor", 1.05)
            target_cap = _nwave_float("target_cap", 1.8)
            invalid_pips = max(hard_sl_floor, nwave.pullback_pips * pullback_mult)
            target_pips = max(target_floor, min(target_cap, nwave.amplitude_pips * target_scale))
            quality = nwave.quality
            proximity_bias = _level_bias(entry_price)
            base_conf = 55.0 + (min(quality, 2.0) * 20.0)
            base_conf -= proximity_bias * 15.0
            base_conf = max(40.0, min(96.0, base_conf))
            tolerance_default = _nwave_float("tolerance_default", 0.24)
            tolerance_tactical = _nwave_float("tolerance_tactical", tolerance_default + 0.08)
            tolerance_pips = tolerance_tactical if scalp_tactical else tolerance_default
            hard_sl_mult = _nwave_float("hard_sl_atr_mult", 1.8)
            hard_sl = max(invalid_pips, atr_pips * hard_sl_mult, hard_sl_floor)

            if direction == "long":
                if close > entry_price + tolerance_pips * _PIP:
                    _log(
                        "skip_nwave_long_late",
                        price=round(close, 3),
                        entry=entry_price,
                        tolerance=tolerance_pips,
                    )
                    return None
                if not _alignment_ok("long"):
                    _log("skip_nwave_long_alignment", price=round(close, 3))
                    return None
                signal = {
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
                _log(
                    "signal_nwave_long",
                    entry=entry_price,
                    sl=signal["sl_pips"],
                    tp=signal["tp_pips"],
                    conf=signal["confidence"],
                    atr=round(atr_pips, 2),
                    rsi=round(rsi, 2),
                )
                return signal

            if close < entry_price - tolerance_pips * _PIP:
                _log(
                    "skip_nwave_short_late",
                    price=round(close, 3),
                    entry=entry_price,
                    tolerance=tolerance_pips,
                )
                return None
            if not _alignment_ok("short"):
                _log("skip_nwave_short_alignment", price=round(close, 3))
                return None
            signal = {
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
            _log(
                "signal_nwave_short",
                entry=entry_price,
                sl=signal["sl_pips"],
                tp=signal["tp_pips"],
                conf=signal["confidence"],
                atr=round(atr_pips, 2),
                rsi=round(rsi, 2),
            )
            return signal

        # Fallback microstructure scalp (limit entry every cycle)
        fallback_enabled = _to_bool(fallback_cfg.get("enabled", True), True)
        if fallback_enabled:
            ticks = fac.get("recent_ticks") or []
            summary = fac.get("recent_tick_summary") or {}
            if ticks:
                try:
                    mid_latest = float(ticks[-1]["mid"])
                except (TypeError, ValueError):
                    mid_latest = close
                if summary:
                    high_mid = float(summary.get("high_mid", mid_latest) or mid_latest)
                    low_mid = float(summary.get("low_mid", mid_latest) or mid_latest)
                else:
                    highs = [t.get("mid") for t in ticks if t.get("mid") is not None]
                    if highs:
                        high_mid = max(float(x) for x in highs)
                        low_mid = min(float(x) for x in highs)
                    else:
                        high_mid = mid_latest
                        low_mid = mid_latest
            else:
                mid_latest = close
                span = max(0.0008, min(0.003, abs(momentum) * 4 or 0.0012))
                high_mid = mid_latest + span
                low_mid = mid_latest - span

            span_mid = max(high_mid - low_mid, 0.0002)
            span_pips = max(0.2, min(4.0, span_mid / _PIP))
            dist_high_pips = max((high_mid - mid_latest) / _PIP, 0.05)
            dist_low_pips = max((mid_latest - low_mid) / _PIP, 0.05)

            base_tp_floor = _fallback_float("tp_floor", 1.2)
            tp_cap = _fallback_float("tp_cap", 1.9)
            if scalp_tactical:
                tp_cap = _fallback_float("tp_cap_tactical", tp_cap)
            tp_dyn = max(
                base_tp_floor,
                min(tp_cap, (span_pips * 0.8) + 0.6),
            )
            if scalp_tactical:
                tp_dyn = min(tp_cap, max(1.0, tp_dyn * 0.9))

            sl_floor = _fallback_float("sl_floor", 8.5)
            if scalp_tactical:
                sl_floor = _fallback_float("sl_floor_tactical", sl_floor)
            sl_cap = _fallback_float("sl_cap", 13.0)
            if scalp_tactical:
                sl_cap = _fallback_float("sl_cap_tactical", sl_cap)
            sl_mult = _fallback_float("sl_atr_mult", 3.2)
            if scalp_tactical:
                sl_mult = _fallback_float("sl_atr_mult_tactical", sl_mult)
            sl_dyn = max(sl_floor, atr_pips * sl_mult)
            sl_dyn = min(sl_cap, sl_dyn)

            entry_base = span_pips * 0.45
            entry_offset_pips = max(0.08, min(0.32, entry_base))
            tolerance_pips = max(0.14, min(0.52, entry_offset_pips * 1.6))

            direction = None
            if rsi >= 58 or momentum >= 0.00018:
                direction = "long"
            elif rsi <= 42 or momentum <= -0.00018:
                direction = "short"
            else:
                # どちらか近い側に吸い寄せる
                direction = "long" if dist_low_pips <= dist_high_pips else "short"

            if direction == "long":
                entry_price = round(mid_latest - entry_offset_pips * _PIP, 3)
                floor_price = round(low_mid + 0.0004, 3)
                if entry_price < floor_price:
                    entry_price = floor_price
                if not _alignment_ok("long"):
                    _log("skip_fallback_long_alignment", momentum=round(momentum, 5))
                    return None
            else:
                entry_price = round(mid_latest + entry_offset_pips * _PIP, 3)
                cap_price = round(high_mid - 0.0004, 3)
                if entry_price > cap_price:
                    entry_price = cap_price
                if not _alignment_ok("short"):
                    _log("skip_fallback_short_alignment", momentum=round(momentum, 5))
                    return None

            mom_norm = abs(momentum) / max(0.0001, atr or 0.0001)
            rsi_bias = abs(rsi - 50.0) / 25.0
            conf_base = 58.0 + (mom_norm * 18.0) + (rsi_bias * 14.0)
            if scalp_tactical:
                conf_base += 6.0
            conf_base += max(0.0, (span_pips - 0.6) * 2.5)
            confidence = int(max(48.0, min(96.0, conf_base)))

            action = "OPEN_LONG" if direction == "long" else "OPEN_SHORT"
            signal = {
                "action": action,
                "entry_type": "limit",
                "entry_price": entry_price,
                "entry_tolerance_pips": round(tolerance_pips, 2),
                "limit_expiry_seconds": 35 if scalp_tactical else 50,
                "sl_pips": round(sl_dyn, 2),
                "tp_pips": round(tp_dyn, 2),
                "confidence": confidence,
                "tag": f"{M1Scalper.name}-micro-{direction}",
                "notes": {
                    "span_pips": round(span_pips, 2),
                    "dist_high": round(dist_high_pips, 2),
                    "dist_low": round(dist_low_pips, 2),
                },
            }
            _log(
                "signal_micro_limit",
                direction=direction,
                entry=entry_price,
                tp=signal["tp_pips"],
                sl=signal["sl_pips"],
                span=round(span_pips, 2),
                momentum=round(momentum, 5),
                rsi=round(rsi, 2),
                ticks=len(ticks),
            )
            return signal

        _log(
            "skip_no_trigger",
            momentum=round(momentum, 5),
            rsi=round(rsi, 2),
            tactical=scalp_tactical,
        )
        return None
