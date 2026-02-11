"""MTF heat score for entry confidence / size / TP adjustment.

This module blends short/mid/long RSI heat with BB expansion state and
pivot-location context. It is direction-aware and returns multipliers
that workers can apply without switching strategy style.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

from indicators.factor_cache import get_candles_snapshot

PIP = 0.01


@dataclass(slots=True)
class MtfHeatDecision:
    enabled: bool
    score: float
    confidence_delta: float
    lot_mult: float
    tp_mult: float
    debug: Dict[str, object]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_float(value: object) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if val != val:  # NaN
        return None
    return val


def _env_key(name: str, prefix: str) -> str:
    p = (prefix or "").strip()
    return f"{p}_{name}" if p else name


def _env_float(name: str, default: float, *, prefix: str = "") -> float:
    raw = os.getenv(_env_key(name, prefix))
    if raw is None:
        raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_bool(name: str, default: bool, *, prefix: str = "") -> bool:
    raw = os.getenv(_env_key(name, prefix))
    if raw is None:
        raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _resolve_price(
    factors: Dict[str, Dict[str, object]],
    tf_candidates: Iterable[str],
    fallback: Optional[float] = None,
) -> Optional[float]:
    for tf in tf_candidates:
        fac = factors.get(tf) or {}
        for key in ("close", "mid", "ma10", "ma20"):
            val = _safe_float(fac.get(key))
            if val is not None and val > 0:
                return val
    return fallback if fallback and fallback > 0 else None


def _rsi_heat(rsi: Optional[float], side_sign: int) -> Optional[float]:
    if rsi is None:
        return None
    if rsi <= 0.0 or rsi >= 100.0:
        return None
    raw = ((rsi - 50.0) / 50.0) * float(side_sign)
    return _clamp(raw, -1.0, 1.0)


def _pivot_levels_from_candles(candles: Sequence[dict]) -> Optional[Dict[str, float]]:
    if len(candles) < 2:
        return None
    prev = candles[-2]
    try:
        high = float(prev.get("high"))
        low = float(prev.get("low"))
        close = float(prev.get("close"))
    except (TypeError, ValueError):
        return None
    if high <= 0.0 or low <= 0.0 or close <= 0.0:
        return None
    pivot = (high + low + close) / 3.0
    r1 = 2.0 * pivot - low
    s1 = 2.0 * pivot - high
    return {"pivot": pivot, "r1": r1, "s1": s1}


def _pivot_levels(
    factors: Dict[str, Dict[str, object]],
    tf: str,
) -> Optional[Dict[str, float]]:
    fac = factors.get(tf) or {}
    raw = fac.get("candles")
    if isinstance(raw, list):
        levels = _pivot_levels_from_candles(raw)
        if levels:
            return levels
    candles = get_candles_snapshot(tf, limit=3)
    if candles:
        return _pivot_levels_from_candles(candles)
    return None


def _side_sign(side: str) -> int:
    s = str(side or "").lower()
    if s in {"sell", "short", "open_short"}:
        return -1
    return 1


def evaluate_mtf_heat(
    side: str,
    factors: Dict[str, Dict[str, object]],
    *,
    price: Optional[float] = None,
    env_prefix: str = "",
    short_tf: str = "M1",
    mid_tf: str = "M5",
    long_tf: str = "H1",
    macro_tf: str = "H4",
    pivot_tfs: Tuple[str, ...] = ("H1", "H4"),
) -> MtfHeatDecision:
    """Compute MTF heat decision for entry sizing and TP horizon."""

    enabled = _env_bool("MTF_HEAT_ENABLED", True, prefix=env_prefix)
    if not enabled:
        return MtfHeatDecision(
            enabled=False,
            score=0.0,
            confidence_delta=0.0,
            lot_mult=1.0,
            tp_mult=1.0,
            debug={"enabled": False, "reason": "disabled"},
        )

    sign = _side_sign(side)
    fac_short = factors.get(short_tf) or {}
    fac_mid = factors.get(mid_tf) or {}
    fac_long = factors.get(long_tf) or {}
    fac_macro = factors.get(macro_tf) or {}

    rsi_short = _safe_float(fac_short.get("rsi"))
    rsi_mid = _safe_float(fac_mid.get("rsi"))
    rsi_long = _safe_float(fac_long.get("rsi"))
    rsi_macro = _safe_float(fac_macro.get("rsi"))

    heat_short = _rsi_heat(rsi_short, sign)
    heat_mid = _rsi_heat(rsi_mid, sign)
    heat_long = _rsi_heat(rsi_long, sign)
    heat_macro = _rsi_heat(rsi_macro, sign)

    long_comp = None
    long_items = [h for h in (heat_long, heat_macro) if h is not None]
    if long_items:
        long_comp = sum(long_items) / max(1, len(long_items))

    short_w = max(0.0, _env_float("MTF_HEAT_RSI_SHORT_W", 0.30, prefix=env_prefix))
    mid_w = max(0.0, _env_float("MTF_HEAT_RSI_MID_W", 0.25, prefix=env_prefix))
    long_w = max(0.0, _env_float("MTF_HEAT_RSI_LONG_W", 0.45, prefix=env_prefix))

    weighted_terms: list[tuple[str, float, float]] = []
    if heat_short is not None:
        weighted_terms.append(("short", heat_short, short_w))
    if heat_mid is not None:
        weighted_terms.append(("mid", heat_mid, mid_w))
    if long_comp is not None:
        weighted_terms.append(("long", long_comp, long_w))

    weighted_score = 0.0
    if weighted_terms:
        w_sum = sum(w for _, _, w in weighted_terms)
        if w_sum > 0:
            weighted_score = sum(v * w for _, v, w in weighted_terms) / w_sum

    short_hot = _env_float("MTF_HEAT_SHORT_HOT", 0.55, prefix=env_prefix)
    long_hot = _env_float("MTF_HEAT_LONG_HOT", 0.55, prefix=env_prefix)
    long_mid = _env_float("MTF_HEAT_LONG_MID", 0.30, prefix=env_prefix)
    align_bonus = _env_float("MTF_HEAT_ALIGN_BONUS", 0.28, prefix=env_prefix)
    mismatch_penalty = _env_float("MTF_HEAT_MISMATCH_PENALTY", 0.34, prefix=env_prefix)
    opposite_penalty = _env_float("MTF_HEAT_OPPOSITE_PENALTY", 0.45, prefix=env_prefix)

    align_component = 0.0
    if heat_short is not None and long_comp is not None:
        if heat_short >= short_hot and long_comp >= long_hot:
            align_component += align_bonus * min(1.0, (heat_short + long_comp) / 2.0)
        if heat_short >= short_hot and long_comp < long_mid:
            gap = min(1.0, (heat_short - short_hot + 0.15) + max(0.0, long_mid - long_comp))
            align_component -= mismatch_penalty * gap
        if heat_short * long_comp < 0.0:
            diff = min(1.0, abs(heat_short - long_comp))
            align_component -= opposite_penalty * diff
    if heat_mid is not None and long_comp is not None and heat_mid * long_comp < 0.0:
        align_component -= mismatch_penalty * 0.35

    bb_component = 0.0
    bb_short = _safe_float(fac_short.get("bbw"))
    bb_long = _safe_float((fac_long or {}).get("bbw")) or _safe_float((fac_macro or {}).get("bbw"))
    bb_floor = max(0.0, _env_float("MTF_HEAT_BB_FLOOR", 0.0011, prefix=env_prefix))
    bb_expand_ratio = max(0.1, _env_float("MTF_HEAT_BB_EXPAND_RATIO", 1.15, prefix=env_prefix))
    bb_squeeze_ratio = max(0.1, _env_float("MTF_HEAT_BB_SQUEEZE_RATIO", 0.90, prefix=env_prefix))
    bb_expand_bonus = max(0.0, _env_float("MTF_HEAT_BB_EXPAND_BONUS", 0.12, prefix=env_prefix))
    bb_squeeze_penalty = max(0.0, _env_float("MTF_HEAT_BB_SQUEEZE_PENALTY", 0.15, prefix=env_prefix))
    bb_ratio = None
    if bb_short is not None and bb_long is not None and bb_short > 0 and bb_long > 0:
        bb_ratio = bb_short / max(bb_long, 1e-6)
        if heat_short is not None and heat_short > 0:
            if bb_ratio >= bb_expand_ratio:
                bb_component += bb_expand_bonus * min(1.0, (bb_ratio - bb_expand_ratio) / 0.6)
            if bb_ratio <= bb_squeeze_ratio and bb_short <= bb_floor:
                bb_component -= bb_squeeze_penalty * min(1.0, (bb_squeeze_ratio - bb_ratio) / 0.4 + (bb_floor - bb_short) / max(bb_floor, 1e-6))
        if heat_short is not None and heat_short < 0 and bb_ratio >= bb_expand_ratio:
            bb_component -= bb_expand_bonus * min(1.0, (bb_ratio - bb_expand_ratio) / 0.6)

    price_val = _resolve_price(
        factors,
        (short_tf, mid_tf, long_tf, macro_tf),
        fallback=price,
    )
    pivot_buffer_pips = max(0.0, _env_float("MTF_HEAT_PIVOT_BUFFER_PIPS", 0.8, prefix=env_prefix))
    pivot_weight = max(0.0, _env_float("MTF_HEAT_PIVOT_WEIGHT", 0.22, prefix=env_prefix))
    pivot_component = 0.0
    pivot_rows: list[dict] = []
    if price_val is not None and price_val > 0 and pivot_tfs:
        pivot_scores: list[float] = []
        buffer_px = pivot_buffer_pips * PIP
        for tf in pivot_tfs:
            levels = _pivot_levels(factors, tf)
            if not levels:
                continue
            pivot = levels["pivot"]
            r1 = levels["r1"]
            s1 = levels["s1"]
            if sign > 0:
                if price_val >= r1 + buffer_px:
                    p_score = 1.0
                elif price_val >= pivot + buffer_px:
                    p_score = 0.45
                elif price_val <= pivot - buffer_px:
                    p_score = -0.45
                else:
                    p_score = 0.0
            else:
                if price_val <= s1 - buffer_px:
                    p_score = 1.0
                elif price_val <= pivot - buffer_px:
                    p_score = 0.45
                elif price_val >= pivot + buffer_px:
                    p_score = -0.45
                else:
                    p_score = 0.0
            pivot_scores.append(p_score)
            pivot_rows.append(
                {
                    "tf": tf,
                    "pivot": round(pivot, 5),
                    "r1": round(r1, 5),
                    "s1": round(s1, 5),
                    "score": round(p_score, 3),
                }
            )
        if pivot_scores:
            pivot_component = (sum(pivot_scores) / len(pivot_scores)) * pivot_weight

    score = _clamp(weighted_score + align_component + bb_component + pivot_component, -1.0, 1.0)

    conf_max_delta = max(0.0, _env_float("MTF_HEAT_CONF_MAX_DELTA", 12.0, prefix=env_prefix))
    lot_scale = max(0.0, _env_float("MTF_HEAT_LOT_SCALE", 0.28, prefix=env_prefix))
    lot_min = max(0.05, _env_float("MTF_HEAT_LOT_MIN", 0.70, prefix=env_prefix))
    lot_max = max(lot_min, _env_float("MTF_HEAT_LOT_MAX", 1.35, prefix=env_prefix))
    tp_scale = max(0.0, _env_float("MTF_HEAT_TP_SCALE", 0.38, prefix=env_prefix))
    tp_min = max(0.1, _env_float("MTF_HEAT_TP_MIN", 0.72, prefix=env_prefix))
    tp_max = max(tp_min, _env_float("MTF_HEAT_TP_MAX", 1.50, prefix=env_prefix))

    confidence_delta = score * conf_max_delta
    lot_mult = _clamp(1.0 + score * lot_scale, lot_min, lot_max)
    tp_mult = _clamp(1.0 + score * tp_scale, tp_min, tp_max)

    debug: Dict[str, object] = {
        "enabled": True,
        "side": "long" if sign > 0 else "short",
        "score": round(score, 3),
        "weighted_score": round(weighted_score, 3),
        "align_component": round(align_component, 3),
        "bb_component": round(bb_component, 3),
        "pivot_component": round(pivot_component, 3),
        "confidence_delta": round(confidence_delta, 2),
        "lot_mult": round(lot_mult, 3),
        "tp_mult": round(tp_mult, 3),
        "rsi_heat": {
            "short_tf": short_tf,
            "short": None if heat_short is None else round(heat_short, 3),
            "mid_tf": mid_tf,
            "mid": None if heat_mid is None else round(heat_mid, 3),
            "long_tf": long_tf,
            "long": None if heat_long is None else round(heat_long, 3),
            "macro_tf": macro_tf,
            "macro": None if heat_macro is None else round(heat_macro, 3),
            "long_comp": None if long_comp is None else round(long_comp, 3),
        },
        "rsi_values": {
            "short": rsi_short,
            "mid": rsi_mid,
            "long": rsi_long,
            "macro": rsi_macro,
        },
        "bbw": {
            "short": bb_short,
            "long": bb_long,
            "ratio": None if bb_ratio is None else round(bb_ratio, 4),
        },
        "pivot": {
            "price": None if price_val is None else round(price_val, 5),
            "buffer_pips": round(pivot_buffer_pips, 3),
            "rows": pivot_rows,
        },
    }
    if weighted_terms:
        debug["weighted_terms"] = [
            {"name": n, "value": round(v, 3), "weight": round(w, 3)}
            for n, v, w in weighted_terms
        ]

    return MtfHeatDecision(
        enabled=True,
        score=score,
        confidence_delta=confidence_delta,
        lot_mult=lot_mult,
        tp_mult=tp_mult,
        debug=debug,
    )

