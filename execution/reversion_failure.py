"""Mean-reversion exit evaluator (structure break + reversion failure)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence

from analysis.range_guard import detect_range_mode_for_tf
from analysis.range_model import compute_range_snapshot, RangeSnapshot
from indicators.factor_cache import all_factors, get_candles_snapshot

PIP = 0.01


@dataclass(slots=True)
class ReversionFailureDecision:
    should_exit: bool
    reason: Optional[str]
    debug: Dict[str, float]
    trend_hits: int


@dataclass(slots=True)
class TpZoneDecision:
    should_exit: bool
    reason: Optional[str]
    debug: Dict[str, float]


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _tf_seconds(tf: str) -> int:
    return {
        "M1": 60,
        "M5": 5 * 60,
        "H1": 60 * 60,
        "H4": 4 * 60 * 60,
        "D1": 24 * 60 * 60,
    }.get(tf.upper(), 60)


def _extract_range_snapshot(
    thesis: Dict[str, object],
    candles: Sequence[Dict[str, object]],
    *,
    lookback: int,
    hi_pct: float,
    lo_pct: float,
    method: str,
) -> Optional[RangeSnapshot]:
    raw = thesis.get("range_snapshot") or thesis.get("range")
    if isinstance(raw, dict):
        high = _safe_float(raw.get("high"))
        low = _safe_float(raw.get("low"))
        mid = _safe_float(raw.get("mid"))
        if high is not None and low is not None:
            if mid is None:
                mid = 0.5 * (high + low)
            return RangeSnapshot(
                high=high,
                low=low,
                mid=mid,
                method=str(raw.get("method") or method or "percentile").lower(),
                lookback=int(raw.get("lookback") or lookback),
                hi_pct=float(raw.get("hi_pct") or hi_pct),
                lo_pct=float(raw.get("lo_pct") or lo_pct),
                end_time=str(raw.get("end_time") or "") or None,
            )
    if candles:
        return compute_range_snapshot(
            candles,
            lookback=lookback,
            method=method,
            hi_pct=hi_pct,
            lo_pct=lo_pct,
        )
    return None


def _soft_tp_mode(thesis: Dict[str, object]) -> bool:
    mode = thesis.get("tp_mode")
    if not mode and isinstance(thesis.get("execution"), dict):
        mode = thesis.get("execution", {}).get("tp_mode")
    return str(mode or "").lower() in {"soft_zone", "soft"}


def _atr_pips_from_factors(factors: Dict[str, Dict[str, float]], tf: str) -> Optional[float]:
    fac = factors.get(tf, {}) if isinstance(factors, dict) else {}
    atr_pips = _safe_float(fac.get("atr_pips"))
    if atr_pips is None:
        atr = _safe_float(fac.get("atr"))
        if atr is not None:
            atr_pips = atr * 100.0
    return atr_pips


def evaluate_reversion_failure(
    trade: Dict[str, object],
    *,
    current_price: float,
    now: datetime,
    side: str,
    env_tf: str,
    struct_tf: str,
    trend_hits: int = 0,
    factors: Optional[Dict[str, Dict[str, float]]] = None,
    env_candles: Optional[Sequence[Dict[str, object]]] = None,
    struct_candles: Optional[Sequence[Dict[str, object]]] = None,
) -> ReversionFailureDecision:
    thesis = trade.get("entry_thesis") or {}
    if not isinstance(thesis, dict):
        thesis = {}

    entry_price = _safe_float(trade.get("price") or trade.get("entry_price"))
    if entry_price is None or current_price <= 0.0:
        return ReversionFailureDecision(False, None, {}, trend_hits)

    rev_cfg = thesis.get("reversion_failure") or {}
    if not isinstance(rev_cfg, dict):
        rev_cfg = {}
    bars_cfg = rev_cfg.get("bars_budget") or {}
    if not isinstance(bars_cfg, dict):
        bars_cfg = {}
    trend_cfg = rev_cfg.get("trend_takeover") or {}
    if not isinstance(trend_cfg, dict):
        trend_cfg = {}
    struct_cfg = thesis.get("structure_break") or {}
    if not isinstance(struct_cfg, dict):
        struct_cfg = {}

    lookback = _safe_int(thesis.get("range_lookback")) or 20
    hi_pct = _safe_float(thesis.get("range_hi_pct")) or 95.0
    lo_pct = _safe_float(thesis.get("range_lo_pct")) or 5.0
    method = str(thesis.get("range_method") or "percentile").lower()

    if factors is None:
        factors = all_factors()
    if env_candles is None:
        env_candles = get_candles_snapshot(env_tf, limit=max(lookback, 6))
    if struct_candles is None:
        struct_candles = get_candles_snapshot(struct_tf, limit=max(lookback, 6))

    snapshot = _extract_range_snapshot(
        thesis,
        env_candles,
        lookback=lookback,
        hi_pct=hi_pct,
        lo_pct=lo_pct,
        method=method,
    )
    if snapshot is None:
        return ReversionFailureDecision(False, None, {}, trend_hits)

    entry_mean = _safe_float(thesis.get("entry_mean"))
    if entry_mean is None:
        entry_mean = snapshot.mid

    atr_entry = _safe_float(thesis.get("atr_entry")) or _atr_pips_from_factors(factors, struct_tf)
    if atr_entry is None or atr_entry <= 0.0:
        return ReversionFailureDecision(False, None, {}, trend_hits)

    price_diff_pips = (current_price - entry_mean) / PIP
    entry_diff_pips = (entry_price - entry_mean) / PIP
    z = price_diff_pips / atr_entry
    z0 = _safe_float(thesis.get("z0"))
    if z0 is None:
        z0 = entry_diff_pips / atr_entry
    if z0 is None or abs(z0) <= 1e-6:
        return ReversionFailureDecision(False, None, {}, trend_hits)

    buffer_atr = _safe_float(struct_cfg.get("buffer_atr")) or 0.10
    confirm_closes = _safe_int(struct_cfg.get("confirm_closes")) or 2
    buffer_price = max(0.0, buffer_atr) * atr_entry * PIP
    if confirm_closes > 0 and struct_candles:
        closes = [
            _safe_float(c.get("close"))
            for c in struct_candles[-confirm_closes:]
            if _safe_float(c.get("close")) is not None
        ]
        if len(closes) >= confirm_closes:
            if side == "long" and all(c <= snapshot.low - buffer_price for c in closes):
                return ReversionFailureDecision(
                    True,
                    "structure_break",
                    {"range_low": snapshot.low, "buffer": buffer_price},
                    trend_hits,
                )
            if side == "short" and all(c >= snapshot.high + buffer_price for c in closes):
                return ReversionFailureDecision(
                    True,
                    "structure_break",
                    {"range_high": snapshot.high, "buffer": buffer_price},
                    trend_hits,
                )

    z_ext = _safe_float(rev_cfg.get("z_ext")) or 0.50
    contraction_min = _safe_float(rev_cfg.get("contraction_min")) or 0.50
    k_per_z = _safe_float(bars_cfg.get("k_per_z")) or 3.0
    min_bars = _safe_int(bars_cfg.get("min")) or 2
    max_bars = _safe_int(bars_cfg.get("max")) or 10

    abs_z0 = abs(z0)
    abs_z = abs(z)
    if abs_z >= abs_z0 + max(0.0, z_ext):
        return ReversionFailureDecision(
            True,
            "reversion_extension",
            {"z": z, "z0": z0, "abs_z": abs_z, "abs_z0": abs_z0},
            trend_hits,
        )

    entry_time = _parse_time(trade.get("open_time") or trade.get("entry_time"))
    if entry_time is None:
        bars_since = 0
    else:
        elapsed = max(0.0, (now - entry_time).total_seconds())
        bars_since = int(elapsed // _tf_seconds(struct_tf))

    bars_budget = int(round(max(min_bars, min(max_bars, k_per_z * abs_z0))))
    progress = 1.0 - (abs_z / abs_z0 if abs_z0 > 0 else 1.0)

    range_ctx = detect_range_mode_for_tf(factors, env_tf)
    if range_ctx.mode == "TREND":
        trend_hits += 1
    else:
        trend_hits = 0
    trend_confirm = _safe_int(trend_cfg.get("require_env_trend_bars")) or 2
    trend_takeover = trend_hits >= max(1, trend_confirm)

    if bars_since >= bars_budget and progress < contraction_min and trend_takeover:
        return ReversionFailureDecision(
            True,
            "reversion_trend_takeover",
            {
                "z": z,
                "z0": z0,
                "progress": progress,
                "bars_since": bars_since,
                "bars_budget": bars_budget,
                "trend_hits": trend_hits,
            },
            trend_hits,
        )

    return ReversionFailureDecision(
        False,
        None,
        {
            "z": z,
            "z0": z0,
            "progress": progress,
            "bars_since": bars_since,
            "bars_budget": bars_budget,
            "trend_hits": trend_hits,
        },
        trend_hits,
    )


def evaluate_tp_zone(
    trade: Dict[str, object],
    *,
    current_price: float,
    side: str,
    env_tf: str,
    struct_tf: str,
    factors: Optional[Dict[str, Dict[str, float]]] = None,
    env_candles: Optional[Sequence[Dict[str, object]]] = None,
) -> TpZoneDecision:
    thesis = trade.get("entry_thesis") or {}
    if not isinstance(thesis, dict) or not _soft_tp_mode(thesis):
        return TpZoneDecision(False, None, {})
    if current_price <= 0.0:
        return TpZoneDecision(False, None, {})

    if factors is None:
        factors = all_factors()
    if env_candles is None:
        lookback = _safe_int(thesis.get("range_lookback")) or 20
        env_candles = get_candles_snapshot(env_tf, limit=max(lookback, 6))

    entry_mean = _safe_float(thesis.get("entry_mean"))
    if entry_mean is None:
        lookback = _safe_int(thesis.get("range_lookback")) or 20
        hi_pct = _safe_float(thesis.get("range_hi_pct")) or 95.0
        lo_pct = _safe_float(thesis.get("range_lo_pct")) or 5.0
        method = str(thesis.get("range_method") or "percentile").lower()
        snapshot = _extract_range_snapshot(
            thesis,
            env_candles or [],
            lookback=lookback,
            hi_pct=hi_pct,
            lo_pct=lo_pct,
            method=method,
        )
        if snapshot is None:
            return TpZoneDecision(False, None, {})
        entry_mean = snapshot.mid

    atr_entry = _safe_float(thesis.get("atr_entry")) or _atr_pips_from_factors(factors, struct_tf)
    if atr_entry is None or atr_entry <= 0.0:
        return TpZoneDecision(False, None, {})

    pad_pips = _safe_float(thesis.get("tp_pad_pips"))
    if pad_pips is None:
        pad_atr = _safe_float(thesis.get("tp_pad_atr")) or 0.05
        pad_pips = max(0.0, pad_atr) * atr_entry
    pad_price = pad_pips * PIP

    hit = False
    if side == "long":
        hit = current_price >= entry_mean - pad_price
    elif side == "short":
        hit = current_price <= entry_mean + pad_price

    if not hit:
        return TpZoneDecision(False, None, {"entry_mean": entry_mean, "pad_pips": pad_pips})
    return TpZoneDecision(
        True,
        "take_profit_zone",
        {
            "entry_mean": entry_mean,
            "pad_pips": pad_pips,
            "current_price": current_price,
        },
    )
