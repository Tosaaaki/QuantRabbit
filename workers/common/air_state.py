"""
workers.common.air_state
~~~~~~~~~~~~~~~~~~~~~~~~
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from analysis.range_guard import detect_range_mode
from market_data import spread_monitor, tick_window
from utils.metrics_logger import log_metric


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no"}


AIR_ENABLED = _env_bool("AIR_ENABLED", True)
AIR_TICK_WINDOW_SEC = _env_float("AIR_TICK_WINDOW_SEC", 6.0)
AIR_TICK_MIN_SAMPLES = _env_int("AIR_TICK_MIN_SAMPLES", 12)
AIR_TICK_RATE_REF = _env_float("AIR_TICK_RATE_REF", 2.5)
AIR_VELOCITY_REF_PPS = _env_float("AIR_VELOCITY_REF_PPS", 0.35)
AIR_IMBALANCE_MIN = _env_float("AIR_IMBALANCE_MIN", 0.58)

AIR_SPREAD_TIGHT_PIPS = _env_float("AIR_SPREAD_TIGHT_PIPS", 0.9)
AIR_SPREAD_WIDE_PIPS = _env_float("AIR_SPREAD_WIDE_PIPS", 1.4)
AIR_SPREAD_BLOCK_WIDE = _env_bool("AIR_SPREAD_BLOCK_WIDE", True)

AIR_EXEC_STALE_MS = _env_float("AIR_EXEC_STALE_MS", 3000.0)
AIR_EXEC_HARD_BLOCK_MS = _env_float("AIR_EXEC_HARD_BLOCK_MS", 5000.0)
AIR_EXEC_MIN_QUALITY = _env_float("AIR_EXEC_MIN_QUALITY", 0.35)
AIR_EXEC_MIN_TICK_RATE = _env_float("AIR_EXEC_MIN_TICK_RATE", 0.8)

AIR_RANGE_PREF_HIGH = _env_float("AIR_RANGE_PREF_HIGH", 0.62)
AIR_RANGE_PREF_LOW = _env_float("AIR_RANGE_PREF_LOW", 0.42)

AIR_SIZE_MIN = _env_float("AIR_SIZE_MIN", 0.65)
AIR_SIZE_MAX = _env_float("AIR_SIZE_MAX", 1.25)

AIR_LOG_ENABLED = _env_bool("AIR_LOG_ENABLED", False)
AIR_LOG_INTERVAL_SEC = _env_float("AIR_LOG_INTERVAL_SEC", 15.0)

_PIP = 0.01
_LAST_LOG_TS = 0.0
_PREV_RANGE_SCORE: Optional[float] = None
_PREV_ADX: Optional[float] = None
_PREV_TS: float = 0.0


@dataclass(slots=True)
class AirSnapshot:
    ts: float
    enabled: bool
    air_score: float
    allow_entry: bool
    size_mult: float
    pressure_dir: str
    pressure_score: float
    tick_rate: float
    velocity_pps: float
    imbalance: float
    spread_state: str
    spread_p25: Optional[float]
    spread_p50: Optional[float]
    spread_p95: Optional[float]
    spread_blocked: bool
    data_age_ms: Optional[float]
    range_mode: bool
    range_score: float
    range_pref: str
    regime_shift: float
    exec_quality: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "ts": self.ts,
            "air_score": round(self.air_score, 3),
            "allow_entry": self.allow_entry,
            "size_mult": round(self.size_mult, 3),
            "pressure_dir": self.pressure_dir,
            "pressure_score": round(self.pressure_score, 3),
            "tick_rate": round(self.tick_rate, 3),
            "velocity_pps": round(self.velocity_pps, 3),
            "imbalance": round(self.imbalance, 3),
            "spread_state": self.spread_state,
            "spread_p25": None if self.spread_p25 is None else round(self.spread_p25, 3),
            "spread_p50": None if self.spread_p50 is None else round(self.spread_p50, 3),
            "spread_p95": None if self.spread_p95 is None else round(self.spread_p95, 3),
            "spread_blocked": self.spread_blocked,
            "data_age_ms": None if self.data_age_ms is None else round(self.data_age_ms, 1),
            "range_mode": self.range_mode,
            "range_score": round(self.range_score, 3),
            "range_pref": self.range_pref,
            "regime_shift": round(self.regime_shift, 3),
            "exec_quality": round(self.exec_quality, 3),
            "reasons": list(self.reasons),
        }


_RANGE_TAGS = {
    "spreadrangerevert",
    "rangefaderpro",
    "vwapreverts",
    "stochbollbounce",
    "divergencerevert",
    "levelreject",
    "wickreversal",
    "rangefader",
}
_TREND_TAGS = {
    "compressionretest",
    "htfpullbacks",
    "macdtrendride",
    "emaslopepull",
    "sessionedge",
}
_FLOW_TAGS = {
    "tickimbalance",
    "fast_scalp",
}


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _range_pref(range_mode: bool, score: float) -> str:
    if range_mode or score >= AIR_RANGE_PREF_HIGH:
        return "range"
    if score <= AIR_RANGE_PREF_LOW:
        return "trend"
    return "neutral"


def _strategy_style(tag: Optional[str]) -> str:
    if not tag:
        return "neutral"
    key = "".join(ch for ch in str(tag).lower().strip() if ch.isalnum())
    if key in _RANGE_TAGS:
        return "range"
    if key in _TREND_TAGS:
        return "trend"
    if key in _FLOW_TAGS:
        return "flow"
    return "neutral"


def _tick_pressure(window_sec: float) -> tuple[float, str, float, float, float]:
    rows = tick_window.recent_ticks(window_sec)
    if len(rows) < 2:
        return 0.0, "neutral", 0.0, 0.0, 0.0
    mids = [float(r.get("mid") or 0.0) for r in rows]
    epochs = [float(r.get("epoch") or 0.0) for r in rows]
    if not mids or not epochs:
        return 0.0, "neutral", 0.0, 0.0, 0.0
    span = max(0.001, epochs[-1] - epochs[0])
    deltas = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
    up = sum(1 for d in deltas if d > 0)
    down = sum(1 for d in deltas if d < 0)
    total = max(1, up + down)
    imbalance = max(up, down) / total
    net_pips = (mids[-1] - mids[0]) / _PIP
    velocity_pps = net_pips / span
    tick_rate = float(len(mids) - 1) / span
    tick_score = _clamp(tick_rate / max(0.1, AIR_TICK_RATE_REF), 0.0, 1.0)
    vel_score = _clamp(abs(velocity_pps) / max(0.05, AIR_VELOCITY_REF_PPS), 0.0, 1.0)
    pressure = _clamp((tick_score * 0.55) + (vel_score * 0.45), 0.0, 1.0)
    direction = "neutral"
    if imbalance >= AIR_IMBALANCE_MIN and abs(velocity_pps) >= AIR_VELOCITY_REF_PPS * 0.4:
        if velocity_pps > 0:
            direction = "long"
        elif velocity_pps < 0:
            direction = "short"
    return pressure, direction, tick_rate, velocity_pps, imbalance


def evaluate_air(
    fac_m1: Optional[Dict[str, object]] = None,
    fac_h4: Optional[Dict[str, object]] = None,
    *,
    range_ctx: Optional[object] = None,
    range_active: Optional[bool] = None,
    tag: Optional[str] = None,
) -> AirSnapshot:
    now = time.time()
    if not AIR_ENABLED:
        return AirSnapshot(
            ts=now,
            enabled=False,
            air_score=0.0,
            allow_entry=True,
            size_mult=1.0,
            pressure_dir="neutral",
            pressure_score=0.0,
            tick_rate=0.0,
            velocity_pps=0.0,
            imbalance=0.0,
            spread_state="unknown",
            spread_p25=None,
            spread_p50=None,
            spread_p95=None,
            spread_blocked=False,
            data_age_ms=None,
            range_mode=False,
            range_score=0.0,
            range_pref="neutral",
            regime_shift=0.0,
            exec_quality=1.0,
            reasons=[],
        )

    pressure_score, pressure_dir, tick_rate, velocity_pps, imbalance = _tick_pressure(
        AIR_TICK_WINDOW_SEC
    )

    blocked, _, spread_state, _ = spread_monitor.is_blocked()
    spread_info = spread_state or spread_monitor.get_state()
    spread_p25 = spread_info.get("p25_pips") if spread_info else None
    spread_p50 = spread_info.get("median_pips") if spread_info else None
    spread_p95 = spread_info.get("p95_pips") if spread_info else None
    data_age_ms = spread_info.get("age_ms") if spread_info else None

    spread_state_label = "unknown"
    spread_score = 0.6
    if spread_info:
        if spread_info.get("stale"):
            spread_state_label = "stale"
            spread_score = 0.0
        else:
            p50 = float(spread_p50 or 0.0)
            p95 = float(spread_p95 or 0.0)
            if p50 <= AIR_SPREAD_TIGHT_PIPS and p95 <= AIR_SPREAD_TIGHT_PIPS * 1.15:
                spread_state_label = "tight"
                spread_score = 1.0
            elif p50 >= AIR_SPREAD_WIDE_PIPS or p95 >= AIR_SPREAD_WIDE_PIPS * 1.1:
                spread_state_label = "wide"
                spread_score = 0.3
            else:
                spread_state_label = "normal"
                spread_score = 0.7

    if range_ctx is None and fac_m1 is not None and fac_h4 is not None:
        try:
            range_ctx = detect_range_mode(fac_m1 or {}, fac_h4 or {})
        except Exception:
            range_ctx = None

    range_mode = bool(getattr(range_ctx, "active", False)) if range_ctx else bool(range_active)
    range_score = float(getattr(range_ctx, "score", 0.0) or 0.0) if range_ctx else 0.0
    range_pref = _range_pref(range_mode, range_score)

    adx = 0.0
    if fac_m1:
        try:
            adx = float(fac_m1.get("adx") or 0.0)
        except Exception:
            adx = 0.0

    global _PREV_RANGE_SCORE, _PREV_ADX, _PREV_TS
    regime_shift = 0.0
    if _PREV_RANGE_SCORE is not None and _PREV_ADX is not None:
        score_delta = abs(range_score - _PREV_RANGE_SCORE)
        adx_delta = abs(adx - _PREV_ADX) / 25.0
        regime_shift = _clamp(score_delta * 1.4 + adx_delta, 0.0, 1.0)
    _PREV_RANGE_SCORE = range_score
    _PREV_ADX = adx
    _PREV_TS = now

    exec_quality = 1.0
    reasons = []
    if data_age_ms is not None:
        if data_age_ms >= AIR_EXEC_HARD_BLOCK_MS:
            exec_quality *= 0.05
            reasons.append("data_stale_hard")
        elif data_age_ms >= AIR_EXEC_STALE_MS:
            exec_quality *= 0.4
            reasons.append("data_stale")
    if blocked:
        exec_quality *= 0.5
        reasons.append("spread_blocked")
    if tick_rate and tick_rate < AIR_EXEC_MIN_TICK_RATE:
        exec_quality *= 0.7
        reasons.append("tick_sparse")

    regime_stability = 1.0 - regime_shift
    air_score = _clamp(
        (pressure_score * 0.35)
        + (spread_score * 0.25)
        + (exec_quality * 0.2)
        + (regime_stability * 0.2),
        0.0,
        1.0,
    )

    size_mult = _clamp(0.7 + air_score * 0.6, AIR_SIZE_MIN, AIR_SIZE_MAX)
    if pressure_score >= 0.8 and spread_state_label != "wide":
        size_mult = _clamp(size_mult + 0.05, AIR_SIZE_MIN, AIR_SIZE_MAX)
    if spread_state_label == "wide":
        size_mult = _clamp(size_mult * 0.85, AIR_SIZE_MIN, AIR_SIZE_MAX)

    allow_entry = True
    if spread_state_label == "stale":
        allow_entry = False
        reasons.append("spread_stale")
    if AIR_SPREAD_BLOCK_WIDE and spread_state_label == "wide":
        allow_entry = False
        reasons.append("spread_wide")
    if data_age_ms is not None and data_age_ms >= AIR_EXEC_HARD_BLOCK_MS:
        allow_entry = False
    if exec_quality < AIR_EXEC_MIN_QUALITY:
        allow_entry = False
        reasons.append("exec_quality_low")

    snap = AirSnapshot(
        ts=now,
        enabled=True,
        air_score=air_score,
        allow_entry=allow_entry,
        size_mult=size_mult,
        pressure_dir=pressure_dir,
        pressure_score=pressure_score,
        tick_rate=tick_rate,
        velocity_pps=velocity_pps,
        imbalance=imbalance,
        spread_state=spread_state_label,
        spread_p25=spread_p25,
        spread_p50=spread_p50,
        spread_p95=spread_p95,
        spread_blocked=blocked,
        data_age_ms=data_age_ms,
        range_mode=range_mode,
        range_score=range_score,
        range_pref=range_pref,
        regime_shift=regime_shift,
        exec_quality=exec_quality,
        reasons=reasons,
    )

    if AIR_LOG_ENABLED:
        global _LAST_LOG_TS
        if now - _LAST_LOG_TS >= max(1.0, AIR_LOG_INTERVAL_SEC):
            _LAST_LOG_TS = now
            tags = {
                "tag": tag or "",
                "range_pref": range_pref,
                "spread": spread_state_label,
            }
            log_metric("air_score", air_score, tags=tags)
            log_metric("air_pressure", pressure_score, tags=tags)
            log_metric("air_exec_quality", exec_quality, tags=tags)

    return snap


def adjust_signal(signal: Dict[str, object], air: AirSnapshot) -> Dict[str, object]:
    if not air.enabled:
        return signal
    conf = int(signal.get("confidence", 0) or 0)
    size_mult = float(signal.get("size_mult", 1.0) or 1.0)
    style = _strategy_style(signal.get("tag"))
    action = str(signal.get("action") or "")

    if style == "range":
        if air.range_pref == "range":
            conf += 4
        elif air.range_pref == "trend":
            conf -= 6
    elif style in {"trend", "flow"}:
        if air.range_pref == "range":
            conf -= 5
        if air.pressure_dir in {"long", "short"}:
            if (air.pressure_dir == "long" and "LONG" in action) or (
                air.pressure_dir == "short" and "SHORT" in action
            ):
                conf += 3
            else:
                conf -= 4

    conf = int(_clamp(conf, 0, 100))
    size_mult = _clamp(size_mult * air.size_mult, 0.5, 1.6)

    signal["confidence"] = conf
    signal["size_mult"] = size_mult
    signal["air_score"] = air.air_score
    signal["air_pressure"] = air.pressure_score
    signal["air_pressure_dir"] = air.pressure_dir
    signal["air_spread_state"] = air.spread_state
    signal["air_exec_quality"] = air.exec_quality
    signal["air_regime_shift"] = air.regime_shift
    signal["air_range_pref"] = air.range_pref
    return signal
