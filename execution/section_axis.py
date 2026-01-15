"""Section-axis logic for median/fib based exits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional
import json
import os

from analysis.range_model import RangeSnapshot, compute_range_snapshot
from analysis.technique_engine import evaluate_exit_techniques
from indicators.factor_cache import get_candles_snapshot

PIP = 0.01

_TF_ALIASES = {
    "1m": "M1",
    "m1": "M1",
    "5m": "M5",
    "m5": "M5",
    "1h": "H1",
    "h1": "H1",
    "4h": "H4",
    "h4": "H4",
    "1d": "D1",
    "d1": "D1",
}
_VALID_TFS = {"M1", "M5", "H1", "H4", "D1"}
_DEFAULT_TF_BY_POCKET = {
    "scalp": "M1",
    "scalp_fast": "M1",
    "micro": "H1",
    "macro": "H1",
    "manual": "H1",
}
_DEFAULT_LOOKBACK_BY_TF = {
    "M1": 20,
    "M5": 20,
    "H1": 20,
    "H4": 20,
    "D1": 20,
}
_DEFAULT_MIN_RANGE_PIPS = {
    "scalp": 4.0,
    "scalp_fast": 3.0,
    "micro": 6.0,
    "macro": 10.0,
    "manual": 10.0,
}
_DEFAULT_MIN_HOLD_SEC = {
    "scalp": 20.0,
    "scalp_fast": 15.0,
    "micro": 120.0,
    "macro": 600.0,
    "manual": 600.0,
}
_DEFAULT_MID_BUFFER_PIPS = {
    "scalp": 0.3,
    "scalp_fast": 0.25,
    "micro": 0.6,
    "macro": 1.2,
    "manual": 1.2,
}
_DEFAULT_MID_DISTANCE_PIPS = {
    "scalp": 0.8,
    "scalp_fast": 0.6,
    "micro": 1.5,
    "macro": 3.0,
    "manual": 3.0,
}
_DEFAULT_MAX_PROFIT_PIPS = {
    "scalp": 0.4,
    "scalp_fast": 0.3,
    "micro": 0.8,
    "macro": 1.5,
    "manual": 1.5,
}
_DEFAULT_SECTION_DELTA = {
    "scalp": 2,
    "scalp_fast": 2,
    "micro": 2,
    "macro": 2,
    "manual": 2,
}
_DEFAULT_LEFT_BEHIND_HOLD_SEC = {
    "scalp": 600.0,
    "scalp_fast": 480.0,
    "micro": 1800.0,
    "macro": 3600.0,
    "manual": 21600.0,
}
_DEFAULT_LEFT_BEHIND_MIN_PIPS = {
    "scalp": 1.2,
    "scalp_fast": 1.0,
    "micro": 2.5,
    "macro": 4.0,
    "manual": 8.0,
}
_DEFAULT_LEFT_BEHIND_RETURN_SCORE = {"default": -0.15}
_DEFAULT_LEFT_BEHIND_MIN_COVERAGE = {"default": 0.55}
_DEFAULT_LEFT_BEHIND_NEG_COUNT = {"default": 2}


@dataclass(slots=True)
class SectionAxis:
    high: float
    low: float
    mid: float
    method: str
    lookback: int
    hi_pct: float
    lo_pct: float
    tf: Optional[str] = None
    end_time: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "high": self.high,
            "low": self.low,
            "mid": self.mid,
            "method": self.method,
            "lookback": self.lookback,
            "hi_pct": self.hi_pct,
            "lo_pct": self.lo_pct,
            "end_time": self.end_time,
        }
        if self.tf:
            payload["tf"] = self.tf
        if self.source:
            payload["source"] = self.source
        return payload


@dataclass(slots=True)
class SectionExitDecision:
    should_exit: bool
    reason: Optional[str]
    allow_negative: bool
    debug: Dict[str, float | int | str]


def _env_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() not in {"", "0", "false", "no"}


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


def _normalize_tf(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    alias = _TF_ALIASES.get(text.lower())
    if alias:
        return alias
    upper = text.upper()
    return upper if upper in _VALID_TFS else None


def _resolve_tf(thesis: Dict[str, object], pocket: str) -> str:
    for key in ("section_tf", "range_tf", "tf", "timeframe", "entry_tf", "timeframe_entry"):
        tf = _normalize_tf(thesis.get(key))
        if tf:
            return tf
    pocket_upper = pocket.upper()
    tf_env = _normalize_tf(os.getenv(f"SECTION_AXIS_TF_{pocket_upper}") or os.getenv("SECTION_AXIS_TF"))
    if tf_env:
        return tf_env
    return _DEFAULT_TF_BY_POCKET.get(pocket, "M1")


def _resolve_lookback(thesis: Dict[str, object], tf: str) -> int:
    lb = _safe_int(thesis.get("section_lookback") or thesis.get("range_lookback"))
    if lb is None:
        lb = _env_int(f"SECTION_AXIS_LOOKBACK_{tf}") or _env_int("SECTION_AXIS_LOOKBACK")
    if lb is None or lb <= 0:
        lb = _DEFAULT_LOOKBACK_BY_TF.get(tf, 20)
    return max(5, lb)


def _resolve_method(thesis: Dict[str, object]) -> str:
    method = thesis.get("section_method") or thesis.get("range_method")
    if method:
        return str(method).lower()
    return (os.getenv("SECTION_AXIS_METHOD") or "percentile").strip().lower()


def _resolve_hi_lo_pct(thesis: Dict[str, object]) -> tuple[float, float]:
    hi = _safe_float(thesis.get("section_hi_pct") or thesis.get("range_hi_pct"))
    lo = _safe_float(thesis.get("section_lo_pct") or thesis.get("range_lo_pct"))
    if hi is None:
        hi = _env_float("SECTION_AXIS_HI_PCT") or 95.0
    if lo is None:
        lo = _env_float("SECTION_AXIS_LO_PCT") or 5.0
    return float(hi), float(lo)


def _axis_from_mapping(raw: Dict[str, object], source: str) -> Optional[SectionAxis]:
    high = _safe_float(raw.get("high"))
    low = _safe_float(raw.get("low"))
    if high is None or low is None or high <= low:
        return None
    mid = _safe_float(raw.get("mid")) or 0.5 * (high + low)
    method = str(raw.get("method") or "percentile").lower()
    lookback = _safe_int(raw.get("lookback")) or 0
    hi_pct = _safe_float(raw.get("hi_pct")) or 95.0
    lo_pct = _safe_float(raw.get("lo_pct")) or 5.0
    tf = _normalize_tf(raw.get("tf") or raw.get("timeframe"))
    end_time = raw.get("end_time")
    return SectionAxis(
        high=high,
        low=low,
        mid=mid,
        method=method,
        lookback=lookback,
        hi_pct=hi_pct,
        lo_pct=lo_pct,
        tf=tf,
        end_time=str(end_time) if end_time else None,
        source=source,
    )


def _extract_axis(thesis: Dict[str, object]) -> Optional[SectionAxis]:
    raw = thesis.get("section_axis") or thesis.get("range_snapshot") or thesis.get("range")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            raw = parsed
    if isinstance(raw, dict):
        return _axis_from_mapping(raw, "thesis")
    return None


def _axis_from_snapshot(snapshot: RangeSnapshot, tf: str, source: str) -> SectionAxis:
    return SectionAxis(
        high=snapshot.high,
        low=snapshot.low,
        mid=snapshot.mid,
        method=snapshot.method,
        lookback=snapshot.lookback,
        hi_pct=snapshot.hi_pct,
        lo_pct=snapshot.lo_pct,
        tf=tf,
        end_time=snapshot.end_time,
        source=source,
    )


def _compute_axis(thesis: Dict[str, object], pocket: str) -> Optional[SectionAxis]:
    tf = _resolve_tf(thesis, pocket)
    lookback = _resolve_lookback(thesis, tf)
    method = _resolve_method(thesis)
    hi_pct, lo_pct = _resolve_hi_lo_pct(thesis)
    candles = get_candles_snapshot(tf, limit=max(lookback, 6))
    snapshot = compute_range_snapshot(
        candles,
        lookback=lookback,
        method=method,
        hi_pct=hi_pct,
        lo_pct=lo_pct,
    )
    if snapshot is None:
        return None
    return _axis_from_snapshot(snapshot, tf, "cache")


def _strategy_from_client_id(client_order_id: Optional[str]) -> Optional[str]:
    if not client_order_id:
        return None
    parts = str(client_order_id).split("-")
    if len(parts) < 5:
        return None
    tag = parts[3].strip()
    return tag or None


def _extract_strategy_tag(trade: Dict[str, object]) -> Optional[str]:
    thesis = trade.get("entry_thesis") or {}
    if isinstance(thesis, str):
        try:
            parsed = json.loads(thesis)
            if isinstance(parsed, dict):
                thesis = parsed
        except Exception:
            thesis = {}
    if isinstance(thesis, dict):
        for key in ("strategy_tag", "strategy", "tag"):
            val = thesis.get(key)
            if val:
                return str(val)
    for key in ("strategy_tag", "strategy", "tag"):
        val = trade.get(key)
        if val:
            return str(val)
    client_id = trade.get("client_order_id")
    if not client_id and isinstance(trade.get("clientExtensions"), dict):
        client_id = trade["clientExtensions"].get("id")
    return _strategy_from_client_id(str(client_id)) if client_id else None


def _strategy_key_candidates(tag: Optional[str]) -> tuple[str, ...]:
    if not tag:
        return ()
    raw = "".join(ch for ch in str(tag) if ch.isalnum()).upper()
    if not raw:
        return ()
    keys = [raw]
    if len(raw) > 9:
        keys.append(raw[:9])
    return tuple(dict.fromkeys(keys))


def _resolve_by_pocket(
    name: str, pocket: str, defaults: Dict[str, float], strategy_keys: tuple[str, ...] | None = None
) -> float:
    if strategy_keys:
        for key in strategy_keys:
            specific = _env_float(f"{name}_{key}")
            if specific is not None:
                return specific
    pocket_upper = pocket.upper()
    specific = _env_float(f"{name}_{pocket_upper}")
    if specific is not None:
        return specific
    common = _env_float(name)
    if common is not None:
        return common
    return float(defaults.get(pocket, defaults.get("default", 0.0)))


def _resolve_int_by_pocket(
    name: str, pocket: str, defaults: Dict[str, int], strategy_keys: tuple[str, ...] | None = None
) -> int:
    if strategy_keys:
        for key in strategy_keys:
            specific = _env_int(f"{name}_{key}")
            if specific is not None:
                return specific
    pocket_upper = pocket.upper()
    specific = _env_int(f"{name}_{pocket_upper}")
    if specific is not None:
        return specific
    common = _env_int(name)
    if common is not None:
        return common
    return int(defaults.get(pocket, defaults.get("default", 0)))


def _section_exit_enabled(pocket: str, strategy_keys: tuple[str, ...] | None = None) -> bool:
    if strategy_keys:
        for key in strategy_keys:
            specific = _env_bool(f"SECTION_EXIT_ENABLED_{key}")
            if specific is not None:
                return specific
    pocket_upper = pocket.upper()
    specific = _env_bool(f"SECTION_EXIT_ENABLED_{pocket_upper}")
    if specific is not None:
        return specific
    common = _env_bool("SECTION_EXIT_ENABLED")
    if common is not None:
        return common
    return True


def _entry_attach_enabled() -> bool:
    val = _env_bool("SECTION_ENTRY_ATTACH")
    return True if val is None else val


def _section_index(rel: float) -> int:
    if rel <= 0.236:
        return 0
    if rel <= 0.382:
        return 1
    if rel <= 0.618:
        return 2
    if rel <= 0.786:
        return 3
    return 4


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def evaluate_section_exit(
    trade: Dict[str, object],
    *,
    current_price: float,
    now: datetime,
    side: str,
    pocket: str,
    hold_sec: Optional[float] = None,
    min_hold_sec: Optional[float] = None,
    entry_price: Optional[float] = None,
) -> SectionExitDecision:
    strategy_tag = _extract_strategy_tag(trade)
    strategy_keys = _strategy_key_candidates(strategy_tag)
    if not _section_exit_enabled(pocket, strategy_keys):
        return SectionExitDecision(False, None, False, {})
    if current_price <= 0.0:
        return SectionExitDecision(False, None, False, {})
    if side not in {"long", "short"}:
        return SectionExitDecision(False, None, False, {})

    entry = entry_price or _safe_float(trade.get("price") or trade.get("entry_price"))
    if entry is None or entry <= 0.0:
        return SectionExitDecision(False, None, False, {})

    if hold_sec is None:
        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0

    pnl_pips = (current_price - entry) / PIP if side == "long" else (entry - current_price) / PIP

    pocket_key = pocket or "default"
    min_hold = _resolve_by_pocket(
        "SECTION_EXIT_MIN_HOLD_SEC",
        pocket_key,
        _DEFAULT_MIN_HOLD_SEC,
        strategy_keys,
    )
    if min_hold_sec is not None:
        min_hold = max(min_hold, float(min_hold_sec))
    if hold_sec < min_hold:
        return SectionExitDecision(False, None, False, {})

    thesis = trade.get("entry_thesis") or {}
    if not isinstance(thesis, dict):
        thesis = {}

    tech_exit = evaluate_exit_techniques(
        trade=trade,
        current_price=current_price,
        side=side,
        pocket=pocket,
    )
    if tech_exit.should_exit:
        return SectionExitDecision(
            True,
            tech_exit.reason,
            tech_exit.allow_negative,
            tech_exit.debug,
        )

    if pocket_key not in {"manual", "unknown"}:
        left_hold_sec = _resolve_by_pocket(
            "SECTION_EXIT_LEFT_BEHIND_HOLD_SEC",
            pocket_key,
            _DEFAULT_LEFT_BEHIND_HOLD_SEC,
            strategy_keys,
        )
        left_min_pips = _resolve_by_pocket(
            "SECTION_EXIT_LEFT_BEHIND_MIN_PIPS",
            pocket_key,
            _DEFAULT_LEFT_BEHIND_MIN_PIPS,
            strategy_keys,
        )
        left_return_score = _resolve_by_pocket(
            "SECTION_EXIT_LEFT_BEHIND_RETURN_SCORE",
            pocket_key,
            _DEFAULT_LEFT_BEHIND_RETURN_SCORE,
            strategy_keys,
        )
        left_min_coverage = _resolve_by_pocket(
            "SECTION_EXIT_LEFT_BEHIND_MIN_COVERAGE",
            pocket_key,
            _DEFAULT_LEFT_BEHIND_MIN_COVERAGE,
            strategy_keys,
        )
        left_neg_count = _resolve_int_by_pocket(
            "SECTION_EXIT_LEFT_BEHIND_NEG_COUNT",
            pocket_key,
            _DEFAULT_LEFT_BEHIND_NEG_COUNT,
            strategy_keys,
        )
        if (
            left_hold_sec > 0
            and left_min_pips > 0
            and hold_sec >= left_hold_sec
            and pnl_pips <= -left_min_pips
        ):
            debug = dict(tech_exit.debug) if isinstance(tech_exit.debug, dict) else {}
            return_score = debug.get("return_score")
            coverage = float(debug.get("coverage") or 0.0)
            neg_count = int(debug.get("neg_count") or 0)
            pos_count = int(debug.get("pos_count") or 0)
            debug.update(
                {
                    "left_hold_sec": round(left_hold_sec, 3),
                    "left_min_pips": round(left_min_pips, 3),
                    "left_return_score": round(left_return_score, 3),
                    "left_min_coverage": round(left_min_coverage, 3),
                    "left_neg_count": left_neg_count,
                    "pnl_pips": round(pnl_pips, 3),
                    "hold_sec": round(hold_sec, 3),
                }
            )
            if (
                return_score is not None
                and coverage >= left_min_coverage
                and float(return_score) <= left_return_score
            ):
                return SectionExitDecision(True, "left_behind_return", True, debug)
            if coverage >= left_min_coverage and neg_count >= left_neg_count and pos_count == 0:
                return SectionExitDecision(True, "left_behind_signal", True, debug)

    axis = _extract_axis(thesis) or _compute_axis(thesis, pocket_key)
    if axis is None:
        return SectionExitDecision(False, None, False, {})

    range_span = axis.high - axis.low
    if range_span <= 0.0:
        return SectionExitDecision(False, None, False, {})
    range_pips = range_span / PIP

    min_range_pips = _resolve_by_pocket(
        "SECTION_EXIT_MIN_RANGE_PIPS",
        pocket_key,
        _DEFAULT_MIN_RANGE_PIPS,
        strategy_keys,
    )
    if range_pips < min_range_pips:
        return SectionExitDecision(False, None, False, {})

    max_profit = _resolve_by_pocket(
        "SECTION_EXIT_MAX_PROFIT_PIPS",
        pocket_key,
        _DEFAULT_MAX_PROFIT_PIPS,
        strategy_keys,
    )
    if pnl_pips > max_profit:
        return SectionExitDecision(False, None, False, {})

    entry_rel = _clamp((entry - axis.low) / range_span, 0.0, 1.0)
    current_rel = _clamp((current_price - axis.low) / range_span, 0.0, 1.0)
    entry_section = _section_index(entry_rel)
    current_section = _section_index(current_rel)
    section_delta = entry_section - current_section if side == "long" else current_section - entry_section
    section_delta_threshold = _resolve_int_by_pocket(
        "SECTION_EXIT_SECTION_DELTA",
        pocket_key,
        _DEFAULT_SECTION_DELTA,
        strategy_keys,
    )

    fib_trigger = _resolve_by_pocket(
        "SECTION_EXIT_FIB_TRIGGER",
        pocket_key,
        {"default": 0.382},
        strategy_keys,
    )
    fib_deep = _resolve_by_pocket(
        "SECTION_EXIT_FIB_DEEP",
        pocket_key,
        {"default": 0.236},
        strategy_keys,
    )
    fib_trigger = _clamp(fib_trigger, 0.05, 0.49)
    fib_deep = _clamp(fib_deep, 0.05, fib_trigger)
    fib_low = axis.low + range_span * fib_trigger
    fib_high = axis.high - range_span * fib_trigger
    fib_low_deep = axis.low + range_span * fib_deep
    fib_high_deep = axis.high - range_span * fib_deep

    mid_buffer_pips = _resolve_by_pocket(
        "SECTION_EXIT_MID_BUFFER_PIPS",
        pocket_key,
        _DEFAULT_MID_BUFFER_PIPS,
        strategy_keys,
    )
    mid_distance_pips = _resolve_by_pocket(
        "SECTION_EXIT_MID_DISTANCE_PIPS",
        pocket_key,
        _DEFAULT_MID_DISTANCE_PIPS,
        strategy_keys,
    )
    mid_distance_frac = _resolve_by_pocket(
        "SECTION_EXIT_MID_DISTANCE_FRAC",
        pocket_key,
        {"default": 0.15},
        strategy_keys,
    )
    mid_distance_pips = max(mid_distance_pips, range_pips * mid_distance_frac)
    mid_buffer_price = max(0.0, mid_buffer_pips) * PIP

    if side == "long":
        wrong_side = current_price <= axis.mid - mid_buffer_price
        median_distance = max(0.0, (axis.mid - current_price) / PIP)
        fib_break = current_price <= fib_low - mid_buffer_price
        fib_deep_break = current_price <= fib_low_deep - mid_buffer_price
        entry_bias = entry_section >= 2
    else:
        wrong_side = current_price >= axis.mid + mid_buffer_price
        median_distance = max(0.0, (current_price - axis.mid) / PIP)
        fib_break = current_price >= fib_high + mid_buffer_price
        fib_deep_break = current_price >= fib_high_deep + mid_buffer_price
        entry_bias = entry_section <= 2

    if not wrong_side:
        return SectionExitDecision(False, None, False, {})

    allow_shift = section_delta >= section_delta_threshold
    allow_fib = fib_break and entry_bias
    allow_deep = fib_deep_break
    allow_mid = entry_bias and median_distance >= mid_distance_pips

    reason = None
    if allow_deep:
        reason = "section_deep"
    elif allow_shift:
        reason = "section_shift"
    elif allow_fib:
        reason = "section_fib"
    elif allow_mid:
        reason = "section_mid"

    if not reason:
        return SectionExitDecision(False, None, False, {})

    debug = {
        "pnl_pips": round(pnl_pips, 3),
        "range_pips": round(range_pips, 3),
        "entry_rel": round(entry_rel, 3),
        "current_rel": round(current_rel, 3),
        "entry_section": entry_section,
        "current_section": current_section,
        "section_delta": section_delta,
        "mid": round(axis.mid, 5),
        "fib_low": round(fib_low, 5),
        "fib_high": round(fib_high, 5),
        "axis_tf": axis.tf or "",
        "axis_source": axis.source or "",
        "strategy_tag": strategy_tag or "",
    }
    return SectionExitDecision(
        True,
        reason,
        pnl_pips <= 0.0,
        debug,
    )


def attach_section_axis(
    entry_thesis: dict,
    *,
    pocket: str,
) -> dict:
    if not isinstance(entry_thesis, dict):
        return entry_thesis
    if not _entry_attach_enabled():
        return entry_thesis
    if entry_thesis.get("section_axis") or entry_thesis.get("range_snapshot") or entry_thesis.get("range"):
        return entry_thesis
    axis = _compute_axis(entry_thesis, pocket)
    if axis is None:
        return entry_thesis
    payload = axis.to_dict()
    payload.setdefault("source", "auto")
    updated = dict(entry_thesis)
    updated["section_axis"] = payload
    return updated
