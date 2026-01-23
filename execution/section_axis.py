"""Section-axis logic for median/fib based exits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional
import json
import os

from analysis.range_model import RangeSnapshot, compute_range_snapshot
from analysis.technique_engine import evaluate_exit_techniques
from indicators.factor_cache import all_factors, get_candles_snapshot

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
_TF_ORDER = {"M1": 1, "M5": 2, "H1": 3, "H4": 4, "D1": 5}
_DEFAULT_TF_BY_POCKET = {
    "scalp": "M1",
    "scalp_fast": "M1",
    "micro": "H1",
    "macro": "H1",
    "manual": "H1",
}
_DEFAULT_EXIT_MTF_TFS = {
    "scalp": ("M1", "M5"),
    "scalp_fast": ("M1", "M5"),
    "micro": ("M5", "H1"),
    "macro": ("H1", "H4"),
    "manual": ("H1", "H4"),
}
_DEFAULT_MTF_NEG_PIPS = {
    "scalp": 1.2,
    "scalp_fast": 1.0,
    "micro": 2.4,
    "macro": 4.0,
    "manual": 6.0,
}
_DEFAULT_MTF_TREND_ADX_MIN = {
    "scalp": 20.0,
    "scalp_fast": 20.0,
    "micro": 18.0,
    "macro": 22.0,
    "manual": 22.0,
}
_DEFAULT_MTF_TREND_MA_DIFF_PIPS = {
    "scalp": 0.8,
    "scalp_fast": 0.6,
    "micro": 1.2,
    "macro": 2.0,
    "manual": 2.0,
}
_DEFAULT_MTF_TREND_MIN_SIGNALS = {
    "scalp": 3,
    "scalp_fast": 3,
    "micro": 4,
    "macro": 4,
    "manual": 4,
}
_DEFAULT_MTF_MOMENTUM_MIN_SIGNALS = {
    "scalp": 4,
    "scalp_fast": 4,
    "micro": 5,
    "macro": 5,
    "manual": 5,
}
_DEFAULT_MTF_MACD_MIN = {"default": 0.0}
_DEFAULT_MTF_ROC_MIN = {"default": 0.0}
_DEFAULT_MTF_SLOPE_MIN = {"default": 0.0}
_DEFAULT_MTF_CLOUD_MIN = {"default": 0.0}
_DEFAULT_MTF_VWAP_GAP_MIN = {"default": 0.4}
_DEFAULT_MTF_DI_GAP_MIN = {"default": 6.0}
_DEFAULT_MTF_STOCH_LOW = {"default": 0.2}
_DEFAULT_MTF_STOCH_HIGH = {"default": 0.8}
_DEFAULT_MTF_CCI_LOW = {"default": -100.0}
_DEFAULT_MTF_CCI_HIGH = {"default": 100.0}
_DEFAULT_MTF_RSI_LOW = {"default": 45.0}
_DEFAULT_MTF_RSI_HIGH = {"default": 55.0}
_DEFAULT_MTF_HOLD_SEC = {"default": 0.0}
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
_DEFAULT_AXIS_MIN_LOOKBACK_RATIO = {"default": 0.7}
_DEFAULT_AXIS_MAX_AGE_MULT = {"default": 2.5}
_DEFAULT_AXIS_MAX_AGE_SEC = {"default": 0.0}


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


def _tf_seconds(tf: str) -> int:
    return {
        "M1": 60,
        "M5": 5 * 60,
        "H1": 60 * 60,
        "H4": 4 * 60 * 60,
        "D1": 24 * 60 * 60,
    }.get(tf.upper(), 60)


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


def _parse_tf_list(value: object) -> list[str]:
    if value is None:
        return []
    items: list[object]
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    tfs: list[str] = []
    for item in items:
        text = str(item)
        for token in text.replace("|", ",").replace("/", ",").split(","):
            token = token.strip()
            if not token:
                continue
            tf = _normalize_tf(token)
            if tf:
                tfs.append(tf)
    if not tfs:
        return []
    return list(dict.fromkeys(tfs))


def _pick_tf_extremes(tfs: list[str]) -> tuple[Optional[str], Optional[str]]:
    if not tfs:
        return None, None
    valid = [tf for tf in tfs if tf in _TF_ORDER]
    if not valid:
        return None, None
    lower_tf = min(valid, key=lambda tf: _TF_ORDER[tf])
    higher_tf = max(valid, key=lambda tf: _TF_ORDER[tf])
    return lower_tf, higher_tf


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


def _resolve_bool_by_pocket(
    name: str,
    pocket: str,
    default: Optional[bool] = None,
    strategy_keys: tuple[str, ...] | None = None,
) -> Optional[bool]:
    if strategy_keys:
        for key in strategy_keys:
            specific = _env_bool(f"{name}_{key}")
            if specific is not None:
                return specific
    pocket_upper = pocket.upper()
    specific = _env_bool(f"{name}_{pocket_upper}")
    if specific is not None:
        return specific
    common = _env_bool(name)
    if common is not None:
        return common
    return default


def _fast_cut_config(thesis: Dict[str, object]) -> Optional[tuple[float, float, float]]:
    if not isinstance(thesis, dict):
        return None
    if thesis.get("kill_switch") is False:
        return None
    pips = _safe_float(thesis.get("fast_cut_pips"))
    time_sec = _safe_float(thesis.get("fast_cut_time_sec"))
    hard_mult = _safe_float(thesis.get("fast_cut_hard_mult")) or 1.6
    if pips is None or time_sec is None:
        return None
    if pips <= 0 or time_sec <= 0:
        return None
    return float(pips), float(time_sec), float(hard_mult)


def _resolve_exit_mtf_tfs(
    thesis: Dict[str, object],
    pocket: str,
    strategy_keys: tuple[str, ...] | None,
) -> list[str]:
    if isinstance(thesis, dict):
        for key in ("exit_mtf_tfs", "mtf_tfs", "mtf_timeframes", "timeframes"):
            tfs = _parse_tf_list(thesis.get(key))
            if tfs:
                return tfs
    if strategy_keys:
        for key in strategy_keys:
            raw = os.getenv(f"SECTION_EXIT_MTF_TFS_{key}")
            if raw:
                tfs = _parse_tf_list(raw)
                if tfs:
                    return tfs
    pocket_upper = pocket.upper()
    raw = os.getenv(f"SECTION_EXIT_MTF_TFS_{pocket_upper}") or os.getenv("SECTION_EXIT_MTF_TFS")
    if raw:
        tfs = _parse_tf_list(raw)
        if tfs:
            return tfs
    defaults = _DEFAULT_EXIT_MTF_TFS.get(pocket, ())
    return list(defaults)


def _trend_metrics(
    fac: Dict[str, object],
) -> tuple[Optional[int], Optional[float], Optional[float], Optional[float], Optional[float]]:
    ma10 = _safe_float(fac.get("ma10"))
    ma20 = _safe_float(fac.get("ma20"))
    trend_dir = None
    ma_diff_pips = None
    if ma10 is not None and ma20 is not None and ma10 > 0.0 and ma20 > 0.0:
        if ma10 > ma20:
            trend_dir = 1
        elif ma10 < ma20:
            trend_dir = -1
        else:
            trend_dir = 0
        ma_diff_pips = abs(ma10 - ma20) / PIP
    adx = _safe_float(fac.get("adx"))
    return trend_dir, adx, ma_diff_pips, ma10, ma20


def _collect_trend_signals(
    fac: Dict[str, object],
    side_dir: int,
    *,
    adx_min: float,
    ma_min: float,
    di_gap_min: float,
    macd_min: float,
    roc_min: float,
    slope_min: float,
    cloud_min: float,
    vwap_min: float,
) -> tuple[int, int, Dict[str, object], Dict[str, object]]:
    signals: Dict[str, object] = {}
    against_count = 0
    signal_count = 0

    def add_signal(name: str, against: bool, **details: object) -> None:
        nonlocal against_count, signal_count
        signal_count += 1
        if against:
            against_count += 1
        payload = dict(details)
        payload["against"] = against
        signals[name] = payload

    trend_dir, adx, ma_diff_pips, ma10, ma20 = _trend_metrics(fac)
    dir_hint: Optional[int] = None
    dir_source: Optional[str] = None
    if trend_dir is not None and trend_dir != 0 and ma10 is not None and ma20 is not None:
        add_signal(
            "ma_cross",
            trend_dir == -side_dir,
            ma10=round(ma10, 5),
            ma20=round(ma20, 5),
            dir="up" if trend_dir > 0 else "down",
            ma_diff_pips=round(ma_diff_pips or 0.0, 2),
        )
        dir_hint = trend_dir
        dir_source = "ma_cross"

    plus_di = _safe_float(fac.get("plus_di"))
    minus_di = _safe_float(fac.get("minus_di"))
    di_gap = None
    if plus_di is not None and minus_di is not None:
        di_dir = 1 if plus_di > minus_di else -1 if plus_di < minus_di else 0
        di_gap = abs(plus_di - minus_di)
        if di_dir != 0:
            add_signal(
                "di_cross",
                di_dir == -side_dir,
                plus_di=round(plus_di, 2),
                minus_di=round(minus_di, 2),
                gap=round(di_gap, 2),
                dir="up" if di_dir > 0 else "down",
            )
            if dir_hint is None:
                dir_hint = di_dir
                dir_source = "di_cross"

    macd_hist = _safe_float(fac.get("macd_hist"))
    if macd_hist is not None and abs(macd_hist) >= macd_min:
        macd_dir = 1 if macd_hist > 0 else -1 if macd_hist < 0 else 0
        if macd_dir != 0:
            add_signal(
                "macd_hist",
                macd_dir == -side_dir,
                macd_hist=round(macd_hist, 5),
                macd_min=round(macd_min, 5),
                dir="up" if macd_dir > 0 else "down",
            )
            if dir_hint is None:
                dir_hint = macd_dir
                dir_source = "macd_hist"

    roc10 = _safe_float(fac.get("roc10"))
    if roc10 is not None and abs(roc10) >= roc_min:
        roc_dir = 1 if roc10 > 0 else -1 if roc10 < 0 else 0
        if roc_dir != 0:
            add_signal(
                "roc10",
                roc_dir == -side_dir,
                roc10=round(roc10, 4),
                roc_min=round(roc_min, 4),
                dir="up" if roc_dir > 0 else "down",
            )
            if dir_hint is None:
                dir_hint = roc_dir
                dir_source = "roc10"

    slope20 = _safe_float(fac.get("ema_slope_20"))
    if slope20 is not None and abs(slope20) >= slope_min:
        slope_dir = 1 if slope20 > 0 else -1 if slope20 < 0 else 0
        if slope_dir != 0:
            add_signal(
                "ema_slope_20",
                slope_dir == -side_dir,
                ema_slope_20=round(slope20, 6),
                slope_min=round(slope_min, 6),
                dir="up" if slope_dir > 0 else "down",
            )
            if dir_hint is None:
                dir_hint = slope_dir
                dir_source = "ema_slope_20"

    cloud_pos = _safe_float(fac.get("ichimoku_cloud_pos"))
    if cloud_pos is not None and abs(cloud_pos) >= cloud_min:
        cloud_dir = 1 if cloud_pos > 0 else -1 if cloud_pos < 0 else 0
        if cloud_dir != 0:
            add_signal(
                "ichimoku_cloud",
                cloud_dir == -side_dir,
                cloud_pos=round(cloud_pos, 4),
                cloud_min=round(cloud_min, 4),
                dir="up" if cloud_dir > 0 else "down",
            )
            if dir_hint is None:
                dir_hint = cloud_dir
                dir_source = "ichimoku_cloud"

    close = _safe_float(fac.get("close"))
    ema20 = _safe_float(fac.get("ema20") or fac.get("ma20"))
    if close is not None and ema20 is not None and close > 0.0 and ema20 > 0.0:
        price_dir = 1 if close > ema20 else -1 if close < ema20 else 0
        if price_dir != 0:
            add_signal(
                "price_vs_ema20",
                price_dir == -side_dir,
                close=round(close, 5),
                ema20=round(ema20, 5),
                gap_pips=round((close - ema20) / PIP, 2),
                dir="up" if price_dir > 0 else "down",
            )

    vwap_gap = _safe_float(fac.get("vwap_gap"))
    if vwap_gap is not None and abs(vwap_gap) >= vwap_min:
        vwap_dir = 1 if vwap_gap > 0 else -1 if vwap_gap < 0 else 0
        if vwap_dir != 0:
            add_signal(
                "vwap_gap",
                vwap_dir == -side_dir,
                vwap_gap=round(vwap_gap, 3),
                vwap_min=round(vwap_min, 3),
                dir="up" if vwap_dir > 0 else "down",
            )
            if dir_hint is None:
                dir_hint = vwap_dir
                dir_source = "vwap_gap"

    adx_val = adx or 0.0
    ma_diff_val = ma_diff_pips or 0.0
    di_gap_val = di_gap or 0.0
    strength_ok = (
        dir_hint is not None
        and dir_hint == -side_dir
        and (adx_val >= adx_min or ma_diff_val >= ma_min or di_gap_val >= di_gap_min)
    )
    strength_debug = {
        "dir": "up" if (dir_hint or 0) > 0 else "down" if (dir_hint or 0) < 0 else "flat",
        "dir_source": dir_source,
        "adx": round(adx_val, 2),
        "ma_diff_pips": round(ma_diff_val, 2),
        "di_gap": round(di_gap_val, 2),
        "adx_min": round(adx_min, 2),
        "ma_min": round(ma_min, 2),
        "di_gap_min": round(di_gap_min, 2),
        "strength_ok": strength_ok,
    }
    return against_count, signal_count, signals, strength_debug


def _collect_momentum_signals(
    fac: Dict[str, object],
    side_dir: int,
    *,
    rsi_low: float,
    rsi_high: float,
    stoch_low: float,
    stoch_high: float,
    cci_low: float,
    cci_high: float,
    macd_min: float,
    roc_min: float,
    slope_min: float,
    vwap_min: float,
) -> tuple[int, int, Dict[str, object]]:
    signals: Dict[str, object] = {}
    against_count = 0
    signal_count = 0

    def add_signal(name: str, against: bool, **details: object) -> None:
        nonlocal against_count, signal_count
        signal_count += 1
        if against:
            against_count += 1
        payload = dict(details)
        payload["against"] = against
        signals[name] = payload

    rsi = _safe_float(fac.get("rsi"))
    if rsi is not None and 0.0 < rsi < 100.0:
        rsi_against = rsi <= rsi_low if side_dir > 0 else rsi >= rsi_high
        add_signal(
            "rsi",
            rsi_against,
            rsi=round(rsi, 2),
            rsi_low=round(rsi_low, 2),
            rsi_high=round(rsi_high, 2),
        )

    stoch_rsi = _safe_float(fac.get("stoch_rsi"))
    if stoch_rsi is not None and 0.0 <= stoch_rsi <= 1.0:
        stoch_against = stoch_rsi <= stoch_low if side_dir > 0 else stoch_rsi >= stoch_high
        add_signal(
            "stoch_rsi",
            stoch_against,
            stoch_rsi=round(stoch_rsi, 3),
            stoch_low=round(stoch_low, 2),
            stoch_high=round(stoch_high, 2),
        )

    cci = _safe_float(fac.get("cci"))
    if cci is not None:
        cci_against = cci <= cci_low if side_dir > 0 else cci >= cci_high
        add_signal(
            "cci",
            cci_against,
            cci=round(cci, 2),
            cci_low=round(cci_low, 1),
            cci_high=round(cci_high, 1),
        )

    macd_hist = _safe_float(fac.get("macd_hist"))
    if macd_hist is not None and abs(macd_hist) >= macd_min:
        macd_dir = 1 if macd_hist > 0 else -1 if macd_hist < 0 else 0
        if macd_dir != 0:
            add_signal(
                "macd_hist",
                macd_dir == -side_dir,
                macd_hist=round(macd_hist, 5),
                macd_min=round(macd_min, 5),
                dir="up" if macd_dir > 0 else "down",
            )

    roc5 = _safe_float(fac.get("roc5"))
    if roc5 is not None and abs(roc5) >= roc_min:
        roc_dir = 1 if roc5 > 0 else -1 if roc5 < 0 else 0
        if roc_dir != 0:
            add_signal(
                "roc5",
                roc_dir == -side_dir,
                roc5=round(roc5, 4),
                roc_min=round(roc_min, 4),
                dir="up" if roc_dir > 0 else "down",
            )

    roc10 = _safe_float(fac.get("roc10"))
    if roc10 is not None and abs(roc10) >= roc_min:
        roc_dir = 1 if roc10 > 0 else -1 if roc10 < 0 else 0
        if roc_dir != 0:
            add_signal(
                "roc10",
                roc_dir == -side_dir,
                roc10=round(roc10, 4),
                roc_min=round(roc_min, 4),
                dir="up" if roc_dir > 0 else "down",
            )

    slope5 = _safe_float(fac.get("ema_slope_5"))
    if slope5 is not None and abs(slope5) >= slope_min:
        slope_dir = 1 if slope5 > 0 else -1 if slope5 < 0 else 0
        if slope_dir != 0:
            add_signal(
                "ema_slope_5",
                slope_dir == -side_dir,
                ema_slope_5=round(slope5, 6),
                slope_min=round(slope_min, 6),
                dir="up" if slope_dir > 0 else "down",
            )

    slope10 = _safe_float(fac.get("ema_slope_10"))
    if slope10 is not None and abs(slope10) >= slope_min:
        slope_dir = 1 if slope10 > 0 else -1 if slope10 < 0 else 0
        if slope_dir != 0:
            add_signal(
                "ema_slope_10",
                slope_dir == -side_dir,
                ema_slope_10=round(slope10, 6),
                slope_min=round(slope_min, 6),
                dir="up" if slope_dir > 0 else "down",
            )

    plus_di = _safe_float(fac.get("plus_di"))
    minus_di = _safe_float(fac.get("minus_di"))
    if plus_di is not None and minus_di is not None:
        di_dir = 1 if plus_di > minus_di else -1 if plus_di < minus_di else 0
        if di_dir != 0:
            add_signal(
                "di_cross",
                di_dir == -side_dir,
                plus_di=round(plus_di, 2),
                minus_di=round(minus_di, 2),
                dir="up" if di_dir > 0 else "down",
            )

    close = _safe_float(fac.get("close"))
    ema20 = _safe_float(fac.get("ema20") or fac.get("ma20"))
    if close is not None and ema20 is not None and close > 0.0 and ema20 > 0.0:
        price_dir = 1 if close > ema20 else -1 if close < ema20 else 0
        if price_dir != 0:
            add_signal(
                "price_vs_ema20",
                price_dir == -side_dir,
                close=round(close, 5),
                ema20=round(ema20, 5),
                gap_pips=round((close - ema20) / PIP, 2),
                dir="up" if price_dir > 0 else "down",
            )

    vwap_gap = _safe_float(fac.get("vwap_gap"))
    if vwap_gap is not None and abs(vwap_gap) >= vwap_min:
        vwap_dir = 1 if vwap_gap > 0 else -1 if vwap_gap < 0 else 0
        if vwap_dir != 0:
            add_signal(
                "vwap_gap",
                vwap_dir == -side_dir,
                vwap_gap=round(vwap_gap, 3),
                vwap_min=round(vwap_min, 3),
                dir="up" if vwap_dir > 0 else "down",
            )

    return against_count, signal_count, signals


def _evaluate_mtf_reversal(
    *,
    pocket: str,
    side: str,
    hold_sec: float,
    min_hold: float,
    pnl_pips: float,
    thesis: Dict[str, object],
    strategy_keys: tuple[str, ...],
) -> Optional[SectionExitDecision]:
    if pocket in {"manual", "unknown"}:
        return None
    enabled = _resolve_bool_by_pocket(
        "SECTION_EXIT_MTF_ENABLED",
        pocket,
        True,
        strategy_keys,
    )
    if enabled is False:
        return None
    if pnl_pips >= 0:
        return None

    tfs = _resolve_exit_mtf_tfs(thesis, pocket, strategy_keys)
    if len(tfs) < 2:
        return None
    lower_tf, higher_tf = _pick_tf_extremes(tfs)
    if not lower_tf or not higher_tf or lower_tf == higher_tf:
        return None

    hold_req = _resolve_by_pocket(
        "SECTION_EXIT_MTF_HOLD_SEC",
        pocket,
        _DEFAULT_MTF_HOLD_SEC,
        strategy_keys,
    )
    if hold_sec < max(min_hold, hold_req):
        return None

    neg_pips = _resolve_by_pocket(
        "SECTION_EXIT_MTF_NEG_PIPS",
        pocket,
        _DEFAULT_MTF_NEG_PIPS,
        strategy_keys,
    )
    if pnl_pips > -neg_pips:
        return None

    factors = all_factors()
    fac_high = factors.get(higher_tf) or {}
    fac_low = factors.get(lower_tf) or {}
    side_dir = 1 if side == "long" else -1

    adx_min = _resolve_by_pocket(
        "SECTION_EXIT_MTF_TREND_ADX_MIN",
        pocket,
        _DEFAULT_MTF_TREND_ADX_MIN,
        strategy_keys,
    )
    ma_min = _resolve_by_pocket(
        "SECTION_EXIT_MTF_TREND_MA_DIFF_PIPS",
        pocket,
        _DEFAULT_MTF_TREND_MA_DIFF_PIPS,
        strategy_keys,
    )
    di_gap_min = _resolve_by_pocket(
        "SECTION_EXIT_MTF_DI_GAP_MIN",
        pocket,
        _DEFAULT_MTF_DI_GAP_MIN,
        strategy_keys,
    )
    macd_min = _resolve_by_pocket(
        "SECTION_EXIT_MTF_MACD_MIN",
        pocket,
        _DEFAULT_MTF_MACD_MIN,
        strategy_keys,
    )
    roc_min = _resolve_by_pocket(
        "SECTION_EXIT_MTF_ROC_MIN",
        pocket,
        _DEFAULT_MTF_ROC_MIN,
        strategy_keys,
    )
    slope_min = _resolve_by_pocket(
        "SECTION_EXIT_MTF_SLOPE_MIN",
        pocket,
        _DEFAULT_MTF_SLOPE_MIN,
        strategy_keys,
    )
    cloud_min = _resolve_by_pocket(
        "SECTION_EXIT_MTF_CLOUD_MIN",
        pocket,
        _DEFAULT_MTF_CLOUD_MIN,
        strategy_keys,
    )
    vwap_min = _resolve_by_pocket(
        "SECTION_EXIT_MTF_VWAP_GAP_MIN",
        pocket,
        _DEFAULT_MTF_VWAP_GAP_MIN,
        strategy_keys,
    )
    trend_min_signals = _resolve_int_by_pocket(
        "SECTION_EXIT_MTF_TREND_MIN_SIGNALS",
        pocket,
        _DEFAULT_MTF_TREND_MIN_SIGNALS,
        strategy_keys,
    )
    momentum_min_signals = _resolve_int_by_pocket(
        "SECTION_EXIT_MTF_MOMENTUM_MIN_SIGNALS",
        pocket,
        _DEFAULT_MTF_MOMENTUM_MIN_SIGNALS,
        strategy_keys,
    )

    high_against, high_total, high_signals, trend_debug = _collect_trend_signals(
        fac_high,
        side_dir,
        adx_min=adx_min,
        ma_min=ma_min,
        di_gap_min=di_gap_min,
        macd_min=macd_min,
        roc_min=roc_min,
        slope_min=slope_min,
        cloud_min=cloud_min,
        vwap_min=vwap_min,
    )

    rsi_low = _resolve_by_pocket(
        "SECTION_EXIT_MTF_RSI_LOW",
        pocket,
        _DEFAULT_MTF_RSI_LOW,
        strategy_keys,
    )
    rsi_high = _resolve_by_pocket(
        "SECTION_EXIT_MTF_RSI_HIGH",
        pocket,
        _DEFAULT_MTF_RSI_HIGH,
        strategy_keys,
    )
    stoch_low = _resolve_by_pocket(
        "SECTION_EXIT_MTF_STOCH_LOW",
        pocket,
        _DEFAULT_MTF_STOCH_LOW,
        strategy_keys,
    )
    stoch_high = _resolve_by_pocket(
        "SECTION_EXIT_MTF_STOCH_HIGH",
        pocket,
        _DEFAULT_MTF_STOCH_HIGH,
        strategy_keys,
    )
    cci_low = _resolve_by_pocket(
        "SECTION_EXIT_MTF_CCI_LOW",
        pocket,
        _DEFAULT_MTF_CCI_LOW,
        strategy_keys,
    )
    cci_high = _resolve_by_pocket(
        "SECTION_EXIT_MTF_CCI_HIGH",
        pocket,
        _DEFAULT_MTF_CCI_HIGH,
        strategy_keys,
    )

    low_against, low_total, low_signals = _collect_momentum_signals(
        fac_low,
        side_dir,
        rsi_low=rsi_low,
        rsi_high=rsi_high,
        stoch_low=stoch_low,
        stoch_high=stoch_high,
        cci_low=cci_low,
        cci_high=cci_high,
        macd_min=macd_min,
        roc_min=roc_min,
        slope_min=slope_min,
        vwap_min=vwap_min,
    )

    if not trend_debug.get("strength_ok"):
        return None
    if high_total < trend_min_signals or low_total < momentum_min_signals:
        return None
    if high_against < trend_min_signals or low_against < momentum_min_signals:
        return None

    debug: Dict[str, object] = {
        "tfs": list(tfs),
        "lower_tf": lower_tf,
        "higher_tf": higher_tf,
        "pnl_pips": round(pnl_pips, 3),
        "neg_pips": round(neg_pips, 3),
        "hold_sec": round(hold_sec, 3),
        "min_hold": round(min_hold, 3),
        "trend": {
            "tf": higher_tf,
            "strength": trend_debug,
            "signals": high_signals,
            "against_count": int(high_against),
            "signal_count": int(high_total),
            "min_signals": int(trend_min_signals),
        },
        "momentum": {
            "tf": lower_tf,
            "signals": low_signals,
            "against_count": int(low_against),
            "signal_count": int(low_total),
            "min_signals": int(momentum_min_signals),
        },
    }
    return SectionExitDecision(True, "mtf_reversal", True, debug)


def _axis_is_usable(
    axis: SectionAxis,
    *,
    now: datetime,
    pocket: str,
    strategy_keys: tuple[str, ...] | None,
    expected_lookback: int,
) -> bool:
    ratio = _resolve_by_pocket(
        "SECTION_AXIS_MIN_LOOKBACK_RATIO",
        pocket,
        _DEFAULT_AXIS_MIN_LOOKBACK_RATIO,
        strategy_keys,
    )
    if expected_lookback > 0 and axis.lookback > 0 and ratio > 0:
        if axis.lookback < expected_lookback * ratio:
            return False

    max_age_sec = _resolve_by_pocket(
        "SECTION_AXIS_MAX_AGE_SEC",
        pocket,
        _DEFAULT_AXIS_MAX_AGE_SEC,
        strategy_keys,
    )
    if max_age_sec <= 0:
        mult = _resolve_by_pocket(
            "SECTION_AXIS_MAX_AGE_MULT",
            pocket,
            _DEFAULT_AXIS_MAX_AGE_MULT,
            strategy_keys,
        )
        tf_sec = _tf_seconds(axis.tf or "")
        if tf_sec > 0 and mult > 0:
            max_age_sec = max(30.0, tf_sec * mult)

    if max_age_sec > 0 and axis.end_time:
        axis_time = _parse_time(axis.end_time)
        if axis_time is not None:
            age_sec = (now - axis_time).total_seconds()
            if age_sec > max_age_sec:
                return False
    return True


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

    fast_cut_enabled = _resolve_bool_by_pocket(
        "SECTION_EXIT_FAST_CUT_ENABLED",
        pocket_key,
        True,
        strategy_keys,
    )
    fast_cut = _fast_cut_config(thesis)
    if fast_cut_enabled and fast_cut and pnl_pips is not None:
        fast_pips, fast_time_sec, fast_hard_mult = fast_cut
        hard_stop_pips = max(fast_pips, fast_pips * max(1.0, fast_hard_mult))
        if hold_sec >= max(min_hold, fast_time_sec) and pnl_pips <= -fast_pips:
            debug = {
                "pnl_pips": round(pnl_pips, 3),
                "hold_sec": round(hold_sec, 3),
                "fast_cut_pips": round(fast_pips, 3),
                "fast_cut_time_sec": round(fast_time_sec, 3),
                "fast_cut_hard_mult": round(fast_hard_mult, 3),
            }
            return SectionExitDecision(True, "fast_cut_time", True, debug)
        if hold_sec >= min_hold and pnl_pips <= -hard_stop_pips:
            debug = {
                "pnl_pips": round(pnl_pips, 3),
                "hold_sec": round(hold_sec, 3),
                "fast_cut_pips": round(fast_pips, 3),
                "fast_cut_time_sec": round(fast_time_sec, 3),
                "fast_cut_hard_mult": round(fast_hard_mult, 3),
                "fast_cut_hard_pips": round(hard_stop_pips, 3),
            }
            return SectionExitDecision(True, "fast_cut_hard", True, debug)

    if pnl_pips is not None:
        mtf_decision = _evaluate_mtf_reversal(
            pocket=pocket_key,
            side=side,
            hold_sec=hold_sec,
            min_hold=min_hold,
            pnl_pips=pnl_pips,
            thesis=thesis,
            strategy_keys=strategy_keys,
        )
        if mtf_decision:
            return mtf_decision

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

    expected_lookback = _resolve_lookback(
        thesis,
        axis.tf or _DEFAULT_TF_BY_POCKET.get(pocket_key, "M1"),
    )
    if not _axis_is_usable(
        axis,
        now=now,
        pocket=pocket_key,
        strategy_keys=strategy_keys,
        expected_lookback=expected_lookback,
    ):
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
