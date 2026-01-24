"""Shared helpers for TP比スケールと仮想SLフロア適用."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from indicators.factor_cache import all_factors
from utils.metrics_logger import log_metric


@dataclass
class TPScaleConfig:
    tp_floor_ratio: float = 1.0
    trail_from_tp_ratio: float = 0.82
    lock_from_tp_ratio: float = 0.45
    virtual_sl_ratio: float = 0.72
    min_profit_take: float = 1.0


def apply_tp_virtual_floor(
    profit_take: float,
    trail_start: float,
    lock_buffer: float,
    stop_loss: float,
    state: Any,
    cfg: TPScaleConfig | None = None,
) -> Tuple[float, float, float, float]:
    """
    Ensure TPを基準にしたトレール/ロック/仮想SLの下限を揃える。
    - tp_hint があれば profit_take を底上げし、trail/lock を比率で拡張
    - 仮想SL は TP 比率（virtual_sl_ratio）でフロア設定
    """
    if cfg is None:
        cfg = TPScaleConfig()

    pt = max(float(profit_take), cfg.min_profit_take)
    ts = float(trail_start)
    lb = float(lock_buffer)
    sl = float(stop_loss)

    tp_hint_val = None
    try:
        tp_hint_val = float(getattr(state, "tp_hint", None))
    except Exception:
        tp_hint_val = None

    if tp_hint_val is not None:
        pt = max(pt, max(cfg.min_profit_take, tp_hint_val * cfg.tp_floor_ratio))
        ts = max(ts, max(cfg.min_profit_take, pt * cfg.trail_from_tp_ratio))
        lb = max(lb, pt * cfg.lock_from_tp_ratio)

    sl = max(sl, pt * cfg.virtual_sl_ratio)
    ts = max(ts, pt * cfg.trail_from_tp_ratio)
    lb = max(lb, pt * cfg.lock_from_tp_ratio)

    return pt, ts, lb, sl


_MOMENTUM_ENABLED = os.getenv("EXIT_MOMENTUM_SCALE_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_MOMENTUM_LOG_ENABLED = os.getenv("EXIT_MOMENTUM_SCALE_LOG", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
_MOMENTUM_INVERT_REVERSAL = os.getenv("EXIT_MOMENTUM_INVERT_REVERSAL", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_MOMENTUM_DEFAULT_TF = {
    "macro": "H1",
    "micro": "M5",
    "scalp": "M1",
    "scalp_fast": "M1",
}
_MOMENTUM_VALID_TF = {"M1", "M5", "H1", "H4", "D1"}
_MOMENTUM_TREND_HINTS = {
    "trend",
    "momentum",
    "donchian",
    "break",
    "impulse",
    "squeeze",
    "runner",
    "session",
    "mtf",
}
_MOMENTUM_REVERSAL_HINTS = {
    "bbrsi",
    "bb_rsi",
    "range",
    "reversal",
    "revert",
    "fade",
    "vwap",
    "level",
    "mirror",
    "spike",
}
_LAST_MOMENTUM_LOG_TS = 0.0


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _env_float(name: str, pocket: Optional[str], default: float) -> float:
    if pocket:
        key = f"{name}_{str(pocket).upper()}"
        raw = os.getenv(key)
        if raw is not None:
            try:
                return float(raw)
            except ValueError:
                return default
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, pocket: Optional[str], default: str) -> str:
    if pocket:
        key = f"{name}_{str(pocket).upper()}"
        raw = os.getenv(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = str(raw).strip()
    return raw if raw else default


def _normalize_score(value: Optional[float], low: float, high: float) -> Optional[float]:
    if value is None:
        return None
    if high <= low:
        return 0.0
    score = (value - low) / (high - low)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _resolve_strategy_mode(strategy_tag: Optional[str], entry_thesis: Optional[dict]) -> str:
    if isinstance(entry_thesis, dict):
        raw_mode = entry_thesis.get("strategy_mode") or entry_thesis.get("mode")
        if raw_mode:
            mode = str(raw_mode).strip().lower()
            if mode in {"trend", "reversal", "mean_reversion"}:
                return "reversal" if mode == "mean_reversion" else mode
    if not strategy_tag:
        return "neutral"
    tag = str(strategy_tag).strip().lower()
    for hint in _MOMENTUM_REVERSAL_HINTS:
        if hint in tag:
            return "reversal"
    for hint in _MOMENTUM_TREND_HINTS:
        if hint in tag:
            return "trend"
    return "neutral"


def momentum_scale(
    *,
    pocket: str,
    strategy_tag: Optional[str] = None,
    entry_thesis: Optional[dict] = None,
    range_active: Optional[bool] = None,
) -> tuple[float, dict]:
    if not _MOMENTUM_ENABLED:
        return 1.0, {"enabled": False}

    pocket_key = str(pocket or "").lower()
    tf_default = _MOMENTUM_DEFAULT_TF.get(pocket_key, "M1")
    tf = _env_str("EXIT_MOMENTUM_TF", pocket_key, tf_default).upper()
    if tf not in _MOMENTUM_VALID_TF:
        tf = tf_default

    fac = {}
    try:
        fac = (all_factors().get(tf) or {}) if tf else {}
    except Exception:
        fac = {}

    adx = _safe_float(fac.get("adx"))
    atr_pips = _safe_float(fac.get("atr_pips"))
    if atr_pips is None:
        atr_val = _safe_float(fac.get("atr"))
        if atr_val is not None:
            atr_pips = atr_val * 100.0

    adx_low = _env_float("EXIT_MOMENTUM_ADX_LOW", pocket_key, 18.0)
    adx_high = _env_float("EXIT_MOMENTUM_ADX_HIGH", pocket_key, 35.0)
    atr_low = _env_float("EXIT_MOMENTUM_ATR_LOW", pocket_key, 6.0)
    atr_high = _env_float("EXIT_MOMENTUM_ATR_HIGH", pocket_key, 14.0)
    trend_score = _normalize_score(adx, adx_low, adx_high)
    vol_score = _normalize_score(atr_pips, atr_low, atr_high)

    weight_trend = _env_float("EXIT_MOMENTUM_WEIGHT_TREND", pocket_key, 0.6)
    weight_vol = _env_float("EXIT_MOMENTUM_WEIGHT_VOL", pocket_key, 0.4)
    weight_sum = 0.0
    score_sum = 0.0
    if trend_score is not None:
        weight_sum += weight_trend
        score_sum += weight_trend * trend_score
    if vol_score is not None:
        weight_sum += weight_vol
        score_sum += weight_vol * vol_score

    if weight_sum <= 0.0:
        return 1.0, {"enabled": True, "tf": tf, "missing": True}

    score = max(0.0, min(1.0, score_sum / weight_sum))
    mode = _resolve_strategy_mode(strategy_tag, entry_thesis)
    if mode == "reversal" and _MOMENTUM_INVERT_REVERSAL:
        score = 1.0 - score

    min_mult = _env_float("EXIT_MOMENTUM_MULT_MIN", pocket_key, 1.0)
    max_mult = _env_float("EXIT_MOMENTUM_MULT_MAX", pocket_key, 1.6)
    if max_mult < min_mult:
        max_mult = min_mult
    scale = min_mult + (max_mult - min_mult) * score

    if range_active:
        range_mult = _env_float("EXIT_MOMENTUM_RANGE_MULT", pocket_key, 1.0)
        scale = min(scale, range_mult)

    meta = {
        "enabled": True,
        "tf": tf,
        "mode": mode,
        "score": round(score, 3),
        "scale": round(scale, 3),
        "adx": round(adx, 2) if adx is not None else None,
        "atr_pips": round(atr_pips, 2) if atr_pips is not None else None,
    }

    if _MOMENTUM_LOG_ENABLED:
        global _LAST_MOMENTUM_LOG_TS
        now = time.monotonic()
        if now - _LAST_MOMENTUM_LOG_TS >= 30.0:
            log_metric(
                "exit_momentum_scale",
                float(scale),
                tags={
                    "pocket": pocket_key or "unknown",
                    "mode": mode,
                    "tf": tf,
                    "score": str(meta["score"]),
                },
            )
            _LAST_MOMENTUM_LOG_TS = now

    return scale, meta


def scale_value(
    value: float,
    *,
    scale: float,
    floor: Optional[float] = None,
    cap: Optional[float] = None,
) -> float:
    scaled = float(value) * float(scale)
    if floor is not None:
        scaled = max(scaled, float(floor))
    if cap is not None:
        scaled = min(scaled, float(cap))
    return scaled
