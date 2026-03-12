"""Scalp extrema reversal dedicated ENTRY worker.

Purpose:
- Catch short entries near local highs after reversal confirmation.
- Catch long entries near local lows after reversal confirmation.
- Keep strategy intent explicit (`entry_probability`, `entry_units_intent`) and
  delegate final allow/scale/reject to order_manager preflight.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import os
import sqlite3
import time
from typing import Dict, Optional, Sequence, Set

from analysis.range_guard import detect_range_mode
from analysis.range_model import compute_range_snapshot
from execution.position_manager import PositionManager
from execution.risk_guard import can_trade, clamp_sl_tp
from execution.strategy_entry import market_order
from indicators.factor_cache import all_factors, get_candles_snapshot
from market_data import spread_monitor, tick_window
from utils.market_hours import is_market_open
from workers.common.air_state import adjust_signal, evaluate_air
from workers.common.dyn_cap import compute_cap
from workers.common.dyn_size import compute_units
from workers.common.quality_gate import current_regime

LOG = logging.getLogger(__name__)

ENV_PREFIX = "SCALP_EXTREMA_REVERSAL"
STRATEGY_TAG = "scalp_extrema_reversal_live"
PIP = 0.01


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(f"{ENV_PREFIX}_{key}")
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(f"{ENV_PREFIX}_{key}")
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(f"{ENV_PREFIX}_{key}")
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _env_str(key: str, default: str) -> str:
    raw = os.getenv(f"{ENV_PREFIX}_{key}")
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip()


def _env_set(key: str, default: str = "") -> Set[str]:
    raw = os.getenv(f"{ENV_PREFIX}_{key}", default)
    return {s.strip().lower() for s in str(raw).split(",") if s.strip()}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_probability(value: object) -> float:
    raw = _to_float(value, 0.0)
    if raw > 1.0:
        raw = raw / 100.0
    return max(0.0, min(1.0, raw))


def _confidence_scale(conf: int, *, lo: int, hi: int) -> float:
    if conf <= lo:
        return 0.5
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.5 + span * 0.5


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    pct = max(0.0, min(pct, 100.0))
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=8.0, limit=1)
    if ticks:
        tick = ticks[-1]
        mid_value = tick.get("mid")
        if mid_value is not None:
            try:
                return float(mid_value)
            except (TypeError, ValueError):
                return fallback
        bid = tick.get("bid")
        ask = tick.get("ask")
        try:
            if bid is not None and ask is not None:
                return (float(bid) + float(ask)) / 2.0
        except (TypeError, ValueError):
            return fallback
    return fallback


def _latest_price(fac_m1: Dict[str, object]) -> float:
    try:
        price = float(fac_m1.get("close") or 0.0)
    except Exception:
        price = 0.0
    return _latest_mid(price)


def _atr_pips(fac_m1: Dict[str, object]) -> float:
    return _to_float(fac_m1.get("atr_pips"), 0.0)


def _rsi(fac_m1: Dict[str, object]) -> float:
    return _to_float(fac_m1.get("rsi"), 50.0)


def _adx(fac_m1: Dict[str, object]) -> float:
    return _to_float(fac_m1.get("adx"), 0.0)


def spread_ok(
    *,
    max_pips: Optional[float] = None,
    p25_max: Optional[float] = None,
) -> tuple[bool, Optional[dict[str, float]]]:
    state = spread_monitor.get_state()
    if state is not None:
        try:
            spread_pips = float(state.get("spread_pips") or 0.0)
        except (TypeError, ValueError):
            spread_pips = None
        if spread_pips is not None:
            if max_pips is not None and spread_pips > max_pips:
                return False, {"spread_pips": spread_pips}
            if p25_max is not None and spread_pips > p25_max:
                return False, {"spread_pips": spread_pips, "p25_max": p25_max}
            return True, {"spread_pips": spread_pips}

    ticks = tick_window.recent_ticks(seconds=8.0, limit=120)
    if not ticks:
        return False, None
    spreads: list[float] = []
    for tick in ticks:
        bid = tick.get("bid")
        ask = tick.get("ask")
        try:
            if bid is None or ask is None:
                continue
            spreads.append(max(0.0, float(ask) - float(bid)) / PIP)
        except (TypeError, ValueError):
            continue
    if not spreads:
        return False, None
    spread_pips = sum(spreads) / len(spreads)
    p25 = _percentile(spreads, 25.0)
    if max_pips is not None and spread_pips > max_pips:
        return False, {"spread_pips": spread_pips, "spread_p25": p25}
    if p25_max is not None and p25 > p25_max:
        return False, {"spread_pips": spread_pips, "spread_p25": p25}
    return True, {"spread_pips": spread_pips, "spread_p25": p25}


def tick_snapshot(seconds: float, *, limit: int = 120) -> tuple[list[float], Optional[dict]]:
    raw_ticks = tick_window.recent_ticks(seconds=seconds, limit=limit)
    mids: list[float] = []
    for item in raw_ticks:
        try:
            if item.get("mid") is None:
                bid = item.get("bid")
                ask = item.get("ask")
                if bid is not None and ask is not None:
                    mids.append((float(bid) + float(ask)) / 2.0)
                continue
            mids.append(float(item.get("mid")))
        except (TypeError, ValueError):
            continue
    return mids, None


def tick_reversal(mids: Sequence[float], *, min_ticks: int = 6) -> tuple[bool, Optional[str], float]:
    if len(mids) < min_ticks:
        return False, None, 0.0
    deltas = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
    if len(deltas) < 5:
        return False, None, 0.0
    prev = deltas[-5:-2]
    recent = deltas[-2:]
    prev_sum = sum(prev)
    recent_sum = sum(recent)
    if prev_sum == 0 or recent_sum == 0:
        return False, None, 0.0
    if prev_sum < 0 < recent_sum:
        return True, "long", abs(recent_sum) / max(PIP, abs(prev_sum))
    if prev_sum > 0 > recent_sum:
        return True, "short", abs(recent_sum) / max(PIP, abs(prev_sum))
    return False, None, 0.0


def _client_order_id(tag: str) -> str:
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in str(tag) if ch.isalnum())[:10] or "exr"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-scalp-exr-{sanitized}{digest}"


ENABLED = _env_bool("ENABLED", True)
POCKET = _env_str("POCKET", "scalp_fast")
TAG = _env_str("STRATEGY_TAG", STRATEGY_TAG)
LOOP_INTERVAL_SEC = _env_float("LOOP_INTERVAL_SEC", 2.0)
LOG_PREFIX = _env_str("LOG_PREFIX", "[Scalp:ExtremaReversal]")
CONFIDENCE_FLOOR = _env_int("CONFIDENCE_FLOOR", 30)
CONFIDENCE_CEIL = _env_int("CONFIDENCE_CEIL", 95)
MIN_ENTRY_CONF = _env_int("MIN_ENTRY_CONF", 0)
MAX_SPREAD_PIPS = _env_float("MAX_SPREAD_PIPS", 1.2)
COOLDOWN_SEC = _env_float("COOLDOWN_SEC", 90.0)
MAX_OPEN_TRADES = _env_int("MAX_OPEN_TRADES", 1)
MAX_OPEN_TRADES_GLOBAL = _env_int("MAX_OPEN_TRADES_GLOBAL", 0)
OPEN_TRADES_SCOPE = _env_str("OPEN_TRADES_SCOPE", "tag").strip().lower()
MIN_UNITS = _env_int("MIN_UNITS", 30)
BASE_ENTRY_UNITS = _env_int("BASE_UNITS", 3000)
MAX_MARGIN_USAGE = _env_float("MAX_MARGIN_USAGE", 0.92)
CAP_MIN = _env_float("CAP_MIN", 0.12)
CAP_MAX = _env_float("CAP_MAX", 0.95)

EXTREMA_LOOKBACK = _env_int("LOOKBACK", 32)
EXTREMA_HIGH_BAND_PIPS = _env_float("HIGH_BAND_PIPS", 1.2)
EXTREMA_LOW_BAND_PIPS = _env_float("LOW_BAND_PIPS", 1.2)
EXTREMA_SHORT_ENABLED = _env_bool("SHORT_ENABLED", True)
EXTREMA_LONG_ENABLED = _env_bool("LONG_ENABLED", True)
EXTREMA_RSI_LONG_MAX = _env_float("RSI_LONG_MAX", 44.0)
EXTREMA_RSI_SHORT_MIN = _env_float("RSI_SHORT_MIN", 56.0)
EXTREMA_LONG_SUPPORT_ENABLED = _env_bool("LONG_SUPPORT_ENABLED", True)
EXTREMA_LONG_SUPPORT_M5_RSI_MIN = _env_float("LONG_SUPPORT_M5_RSI_MIN", 56.0)
EXTREMA_LONG_SUPPORT_M5_DI_GAP_MIN = _env_float("LONG_SUPPORT_M5_DI_GAP_MIN", 0.0)
EXTREMA_LONG_SUPPORT_M5_EMA_SLOPE_MIN = _env_float("LONG_SUPPORT_M5_EMA_SLOPE_MIN", 0.0)
EXTREMA_LONG_SUPPORT_M1_ADX_MAX = _env_float("LONG_SUPPORT_M1_ADX_MAX", 24.0)
EXTREMA_LONG_SUPPORT_M1_EMA_GAP_MAX_PIPS = _env_float("LONG_SUPPORT_M1_EMA_GAP_MAX_PIPS", 1.4)
EXTREMA_LONG_SUPPORT_RSI_CAP = _env_float("LONG_SUPPORT_RSI_CAP", 50.0)
EXTREMA_LONG_SUPPORT_LOW_BAND_PIPS = _env_float("LONG_SUPPORT_LOW_BAND_PIPS", 1.2)
EXTREMA_LONG_SUPPORT_CONF_BONUS = _env_int("LONG_SUPPORT_CONF_BONUS", 4)
EXTREMA_SHORT_SUPPORT_ENABLED = _env_bool("SHORT_SUPPORT_ENABLED", True)
EXTREMA_SHORT_SUPPORT_M5_RSI_MAX = _env_float("SHORT_SUPPORT_M5_RSI_MAX", 44.0)
EXTREMA_SHORT_SUPPORT_M5_DI_GAP_MAX = _env_float("SHORT_SUPPORT_M5_DI_GAP_MAX", 0.0)
EXTREMA_SHORT_SUPPORT_M5_EMA_SLOPE_MAX = _env_float("SHORT_SUPPORT_M5_EMA_SLOPE_MAX", 0.0)
EXTREMA_SHORT_SUPPORT_M1_ADX_MAX = _env_float("SHORT_SUPPORT_M1_ADX_MAX", 24.0)
EXTREMA_SHORT_SUPPORT_M1_EMA_GAP_MAX_PIPS = _env_float("SHORT_SUPPORT_M1_EMA_GAP_MAX_PIPS", 1.4)
EXTREMA_LONG_COUNTERTREND_GAP_BLOCK_PIPS = _env_float("LONG_COUNTERTREND_GAP_BLOCK_PIPS", 0.5)
EXTREMA_SHORT_COUNTERTREND_GAP_BLOCK_PIPS = _env_float("SHORT_COUNTERTREND_GAP_BLOCK_PIPS", 0.45)
EXTREMA_LONG_SHALLOW_PROBE_DIST_LOW_MAX_PIPS = _env_float(
    "LONG_SHALLOW_PROBE_DIST_LOW_MAX_PIPS", 0.30
)
EXTREMA_LONG_SHALLOW_PROBE_BOUNCE_MAX_PIPS = _env_float(
    "LONG_SHALLOW_PROBE_BOUNCE_MAX_PIPS", 0.30
)
EXTREMA_LONG_SHALLOW_PROBE_TICK_STRENGTH_MAX = _env_float(
    "LONG_SHALLOW_PROBE_TICK_STRENGTH_MAX", 0.20
)
EXTREMA_LONG_SHALLOW_PROBE_ADX_MAX = _env_float(
    "LONG_SHALLOW_PROBE_ADX_MAX", 13.0
)
EXTREMA_LONG_SHALLOW_PROBE_RANGE_SCORE_MAX = _env_float(
    "LONG_SHALLOW_PROBE_RANGE_SCORE_MAX", 0.32
)
EXTREMA_LONG_MID_RSI_PROBE_RSI_MIN = _env_float("LONG_MID_RSI_PROBE_RSI_MIN", 40.0)
EXTREMA_LONG_MID_RSI_PROBE_DIST_LOW_MAX_PIPS = _env_float(
    "LONG_MID_RSI_PROBE_DIST_LOW_MAX_PIPS", 0.85
)
EXTREMA_LONG_MID_RSI_PROBE_BOUNCE_MAX_PIPS = _env_float(
    "LONG_MID_RSI_PROBE_BOUNCE_MAX_PIPS", 0.25
)
EXTREMA_LONG_MID_RSI_PROBE_TICK_STRENGTH_MAX = _env_float(
    "LONG_MID_RSI_PROBE_TICK_STRENGTH_MAX", 0.25
)
EXTREMA_LONG_MID_RSI_PROBE_ADX_MIN = _env_float("LONG_MID_RSI_PROBE_ADX_MIN", 15.0)
EXTREMA_LONG_MID_RSI_PROBE_RANGE_SCORE_MIN = _env_float(
    "LONG_MID_RSI_PROBE_RANGE_SCORE_MIN", 0.55
)
EXTREMA_LONG_DRIFT_PROBE_DIST_LOW_MAX_PIPS = _env_float(
    "LONG_DRIFT_PROBE_DIST_LOW_MAX_PIPS", 0.35
)
EXTREMA_LONG_DRIFT_PROBE_BOUNCE_MAX_PIPS = _env_float(
    "LONG_DRIFT_PROBE_BOUNCE_MAX_PIPS", 0.35
)
EXTREMA_LONG_DRIFT_PROBE_TICK_STRENGTH_MAX = _env_float(
    "LONG_DRIFT_PROBE_TICK_STRENGTH_MAX", 0.25
)
EXTREMA_LONG_DRIFT_PROBE_ADX_MIN = _env_float("LONG_DRIFT_PROBE_ADX_MIN", 24.0)
EXTREMA_LONG_DRIFT_PROBE_RANGE_SCORE_MIN = _env_float(
    "LONG_DRIFT_PROBE_RANGE_SCORE_MIN", 0.38
)
EXTREMA_LONG_DRIFT_PROBE_MA_GAP_MIN_PIPS = _env_float(
    "LONG_DRIFT_PROBE_MA_GAP_MIN_PIPS", 0.15
)
EXTREMA_LONG_DRIFT_PROBE_RSI_MIN = _env_float("LONG_DRIFT_PROBE_RSI_MIN", 36.0)
EXTREMA_SHORT_SHALLOW_PROBE_DIST_HIGH_MAX_PIPS = _env_float(
    "SHORT_SHALLOW_PROBE_DIST_HIGH_MAX_PIPS", 0.45
)
EXTREMA_SHORT_SHALLOW_PROBE_BOUNCE_MAX_PIPS = _env_float(
    "SHORT_SHALLOW_PROBE_BOUNCE_MAX_PIPS", 0.45
)
EXTREMA_SHORT_SHALLOW_PROBE_TICK_STRENGTH_MAX = _env_float(
    "SHORT_SHALLOW_PROBE_TICK_STRENGTH_MAX", 0.45
)
EXTREMA_SHORT_SHALLOW_PROBE_ADX_MAX = _env_float(
    "SHORT_SHALLOW_PROBE_ADX_MAX", 26.0
)
EXTREMA_SHORT_SHALLOW_PROBE_RANGE_SCORE_MIN = _env_float(
    "SHORT_SHALLOW_PROBE_RANGE_SCORE_MIN", 0.40
)
EXTREMA_SHORT_SHALLOW_PROBE_MA_GAP_MIN_PIPS = _env_float(
    "SHORT_SHALLOW_PROBE_MA_GAP_MIN_PIPS", 0.10
)
EXTREMA_SHORT_MID_RSI_PROBE_RSI_MAX = _env_float("SHORT_MID_RSI_PROBE_RSI_MAX", 60.0)
EXTREMA_SHORT_MID_RSI_PROBE_DIST_HIGH_MAX_PIPS = _env_float(
    "SHORT_MID_RSI_PROBE_DIST_HIGH_MAX_PIPS", 0.90
)
EXTREMA_SHORT_MID_RSI_PROBE_BOUNCE_MAX_PIPS = _env_float(
    "SHORT_MID_RSI_PROBE_BOUNCE_MAX_PIPS", 0.85
)
EXTREMA_SHORT_MID_RSI_PROBE_TICK_STRENGTH_MAX = _env_float(
    "SHORT_MID_RSI_PROBE_TICK_STRENGTH_MAX", 0.25
)
EXTREMA_SHORT_MID_RSI_PROBE_RANGE_SCORE_MIN = _env_float(
    "SHORT_MID_RSI_PROBE_RANGE_SCORE_MIN", 0.50
)
EXTREMA_SHORT_DRIFT_PROBE_DIST_HIGH_MAX_PIPS = _env_float(
    "SHORT_DRIFT_PROBE_DIST_HIGH_MAX_PIPS", 0.90
)
EXTREMA_SHORT_DRIFT_PROBE_BOUNCE_MAX_PIPS = _env_float(
    "SHORT_DRIFT_PROBE_BOUNCE_MAX_PIPS", 0.15
)
EXTREMA_SHORT_DRIFT_PROBE_TICK_STRENGTH_MAX = _env_float(
    "SHORT_DRIFT_PROBE_TICK_STRENGTH_MAX", 0.15
)
EXTREMA_SHORT_DRIFT_PROBE_RANGE_SCORE_MAX = _env_float(
    "SHORT_DRIFT_PROBE_RANGE_SCORE_MAX", 0.48
)
EXTREMA_SHORT_DRIFT_PROBE_MA_GAP_MIN_PIPS = _env_float(
    "SHORT_DRIFT_PROBE_MA_GAP_MIN_PIPS", 0.0
)
EXTREMA_SHORT_DRIFT_PROBE_MA_GAP_MAX_PIPS = _env_float(
    "SHORT_DRIFT_PROBE_MA_GAP_MAX_PIPS", 0.35
)
EXTREMA_SHORT_DRIFT_PROBE_RSI_MAX = _env_float("SHORT_DRIFT_PROBE_RSI_MAX", 60.0)
EXTREMA_SETUP_PRESSURE_ENABLED = _env_bool("SETUP_PRESSURE_ENABLED", True)
EXTREMA_SETUP_PRESSURE_LOOKBACK_HOURS = _env_float("SETUP_PRESSURE_LOOKBACK_HOURS", 6.0)
EXTREMA_SETUP_PRESSURE_LOOKBACK_TRADES = _env_int("SETUP_PRESSURE_LOOKBACK_TRADES", 6)
EXTREMA_SETUP_PRESSURE_MIN_TRADES = _env_int("SETUP_PRESSURE_MIN_TRADES", 4)
EXTREMA_SETUP_PRESSURE_SL_RATE_MIN = _env_float("SETUP_PRESSURE_SL_RATE_MIN", 0.70)
EXTREMA_SETUP_PRESSURE_FAST_SL_RATE_MIN = _env_float(
    "SETUP_PRESSURE_FAST_SL_RATE_MIN", 0.50
)
EXTREMA_SETUP_PRESSURE_FAST_SL_MAX_HOLD_SEC = _env_float(
    "SETUP_PRESSURE_FAST_SL_MAX_HOLD_SEC", 90.0
)
EXTREMA_SETUP_PRESSURE_CACHE_TTL_SEC = _env_float("SETUP_PRESSURE_CACHE_TTL_SEC", 8.0)
EXTREMA_SHORT_SETUP_PRESSURE_RANGE_REASONS = _env_set(
    "SHORT_SETUP_PRESSURE_RANGE_REASONS", "volatility_compression"
)
EXTREMA_SHORT_SETUP_PRESSURE_DIST_HIGH_MAX_PIPS = _env_float(
    "SHORT_SETUP_PRESSURE_DIST_HIGH_MAX_PIPS", 0.90
)
EXTREMA_SHORT_SETUP_PRESSURE_BOUNCE_MAX_PIPS = _env_float(
    "SHORT_SETUP_PRESSURE_BOUNCE_MAX_PIPS", 0.50
)
EXTREMA_SHORT_SETUP_PRESSURE_TICK_STRENGTH_MAX = _env_float(
    "SHORT_SETUP_PRESSURE_TICK_STRENGTH_MAX", 0.50
)
EXTREMA_LONG_SETUP_PRESSURE_RANGE_REASONS = _env_set(
    "LONG_SETUP_PRESSURE_RANGE_REASONS", "volatility_compression"
)
EXTREMA_LONG_SETUP_PRESSURE_MIN_TRADES = _env_int("LONG_SETUP_PRESSURE_MIN_TRADES", 6)
EXTREMA_LONG_SETUP_PRESSURE_SL_RATE_MIN = _env_float("LONG_SETUP_PRESSURE_SL_RATE_MIN", 0.45)
EXTREMA_LONG_SETUP_PRESSURE_FAST_SL_RATE_MIN = _env_float(
    "LONG_SETUP_PRESSURE_FAST_SL_RATE_MIN", 0.40
)
EXTREMA_LONG_SETUP_PRESSURE_DIST_LOW_MAX_PIPS = _env_float(
    "LONG_SETUP_PRESSURE_DIST_LOW_MAX_PIPS", 0.90
)
EXTREMA_LONG_SETUP_PRESSURE_BOUNCE_MAX_PIPS = _env_float(
    "LONG_SETUP_PRESSURE_BOUNCE_MAX_PIPS", 0.35
)
EXTREMA_LONG_SETUP_PRESSURE_TICK_STRENGTH_MAX = _env_float(
    "LONG_SETUP_PRESSURE_TICK_STRENGTH_MAX", 0.30
)
EXTREMA_LONG_SETUP_PRESSURE_MA_GAP_MAX_PIPS = _env_float(
    "LONG_SETUP_PRESSURE_MA_GAP_MAX_PIPS", 0.0
)
EXTREMA_LONG_SETUP_PRESSURE_RANGE_SCORE_MIN = _env_float(
    "LONG_SETUP_PRESSURE_RANGE_SCORE_MIN", 0.45
)
EXTREMA_LONG_SETUP_PRESSURE_RANGE_SCORE_MAX = _env_float(
    "LONG_SETUP_PRESSURE_RANGE_SCORE_MAX", 0.55
)
EXTREMA_LONG_SETUP_PRESSURE_ADX_MAX = _env_float("LONG_SETUP_PRESSURE_ADX_MAX", 23.0)
EXTREMA_ADX_MAX = _env_float("ADX_MAX", 35.0)
EXTREMA_ATR_MAX = _env_float("ATR_MAX", 0.0)
EXTREMA_SPREAD_P25_MAX = _env_float("SPREAD_P25_MAX", 0.0)
EXTREMA_SIZE_MULT = _env_float("SIZE_MULT", 1.0)
EXTREMA_TICK_STRENGTH_MIN = _env_float("TICK_STRENGTH_MIN", 0.0)
EXTREMA_REV_WINDOW_SEC = _env_float("REV_WINDOW_SEC", 6.0)
EXTREMA_SWEEP_MIN_PIPS = _env_float("SWEEP_MIN_PIPS", 0.06)
EXTREMA_SL_ATR_MULT = _env_float("SL_ATR_MULT", 0.85)
EXTREMA_TP_ATR_MULT = _env_float("TP_ATR_MULT", 1.15)
EXTREMA_SL_MIN_PIPS = _env_float("SL_MIN_PIPS", 1.2)
EXTREMA_SL_MAX_PIPS = _env_float("SL_MAX_PIPS", 2.6)
EXTREMA_TP_MIN_PIPS = _env_float("TP_MIN_PIPS", 1.4)
EXTREMA_TP_MAX_PIPS = _env_float("TP_MAX_PIPS", 3.2)
EXTREMA_ALLOWED_REGIMES = _env_set("ALLOWED_REGIMES", "")
EXTREMA_TREND_GATE_ENABLED = _env_bool("TREND_GATE_ENABLED", True)
EXTREMA_TREND_GATE_MIN_RANGE_SCORE = _env_float("TREND_GATE_MIN_RANGE_SCORE", 0.6)
EXTREMA_TREND_GATE_ADX_MIN = _env_float("TREND_GATE_ADX_MIN", 24.0)
EXTREMA_TREND_GATE_MA_GAP_PIPS = _env_float("TREND_GATE_MA_GAP_PIPS", 1.6)
EXTREMA_TREND_GATE_RANGE_SCORE_MIN = _env_float("TREND_GATE_RANGE_SCORE_MIN", 0.40)
EXTREMA_TREND_GATE_RANGE_MAX_AGAINST_GAP_PIPS = _env_float(
    "TREND_GATE_RANGE_MAX_AGAINST_GAP_PIPS",
    1.0,
)
_TRADES_DB_URI = "file:logs/trades.db?mode=ro"
_SETUP_PRESSURE_CACHE: Dict[tuple[str, str], tuple[float, Dict[str, float]]] = {}


def _ma_gap_pips(fac_m1: Dict[str, object]) -> float:
    ma_fast = fac_m1.get("ma10")
    if ma_fast is None:
        ma_fast = fac_m1.get("ema10")
    ma_slow = fac_m1.get("ma20")
    if ma_slow is None:
        ma_slow = fac_m1.get("ema20")
    try:
        if ma_fast is None or ma_slow is None:
            return 0.0
        return (float(ma_fast) - float(ma_slow)) / PIP
    except (TypeError, ValueError):
        return 0.0


def _extrema_long_support_context(
    fac_m1: Dict[str, object],
    fac_m5: Optional[Dict[str, object]],
    *,
    price: float,
) -> tuple[bool, Dict[str, float]]:
    diag: Dict[str, float] = {}
    if not EXTREMA_LONG_SUPPORT_ENABLED or not fac_m5:
        return False, diag

    m5_close = _to_float(fac_m5.get("close"), 0.0)
    m5_ema20 = _to_float(fac_m5.get("ema20") or fac_m5.get("ma20"), 0.0)
    m5_rsi = _to_float(fac_m5.get("rsi"), 50.0)
    m5_plus_di = _to_float(fac_m5.get("plus_di"), 0.0)
    m5_minus_di = _to_float(fac_m5.get("minus_di"), 0.0)
    m5_ema_slope_10 = _to_float(fac_m5.get("ema_slope_10"), 0.0)
    m1_adx = _adx(fac_m1)
    m1_ema20 = _to_float(fac_m1.get("ema20") or fac_m1.get("ma20"), price)
    m1_ema_gap_pips = abs(price - m1_ema20) / PIP if m1_ema20 > 0.0 else 0.0

    diag = {
        "m5_close": round(m5_close, 3),
        "m5_ema20": round(m5_ema20, 3),
        "m5_rsi": round(m5_rsi, 3),
        "m5_di_gap": round(m5_plus_di - m5_minus_di, 3),
        "m5_ema_slope_10": round(m5_ema_slope_10, 6),
        "m1_adx": round(m1_adx, 3),
        "m1_ema_gap_pips": round(m1_ema_gap_pips, 3),
    }
    if m5_close < m5_ema20:
        return False, diag
    if m5_rsi < EXTREMA_LONG_SUPPORT_M5_RSI_MIN:
        return False, diag
    if (m5_plus_di - m5_minus_di) < EXTREMA_LONG_SUPPORT_M5_DI_GAP_MIN:
        return False, diag
    if m5_ema_slope_10 < EXTREMA_LONG_SUPPORT_M5_EMA_SLOPE_MIN:
        return False, diag
    if m1_adx > EXTREMA_LONG_SUPPORT_M1_ADX_MAX:
        return False, diag
    if m1_ema_gap_pips > EXTREMA_LONG_SUPPORT_M1_EMA_GAP_MAX_PIPS:
        return False, diag
    return True, diag


def _extrema_short_support_context(
    fac_m1: Dict[str, object],
    fac_m5: Optional[Dict[str, object]],
    *,
    price: float,
) -> tuple[bool, Dict[str, float]]:
    diag: Dict[str, float] = {}
    if not EXTREMA_SHORT_SUPPORT_ENABLED or not fac_m5:
        return False, diag

    m5_close = _to_float(fac_m5.get("close"), 0.0)
    m5_ema20 = _to_float(fac_m5.get("ema20") or fac_m5.get("ma20"), 0.0)
    m5_rsi = _to_float(fac_m5.get("rsi"), 50.0)
    m5_plus_di = _to_float(fac_m5.get("plus_di"), 0.0)
    m5_minus_di = _to_float(fac_m5.get("minus_di"), 0.0)
    m5_ema_slope_10 = _to_float(fac_m5.get("ema_slope_10"), 0.0)
    m1_adx = _adx(fac_m1)
    m1_ema20 = _to_float(fac_m1.get("ema20") or fac_m1.get("ma20"), price)
    m1_ema_gap_pips = abs(price - m1_ema20) / PIP if m1_ema20 > 0.0 else 0.0

    diag = {
        "m5_close": round(m5_close, 3),
        "m5_ema20": round(m5_ema20, 3),
        "m5_rsi": round(m5_rsi, 3),
        "m5_di_gap": round(m5_plus_di - m5_minus_di, 3),
        "m5_ema_slope_10": round(m5_ema_slope_10, 6),
        "m1_adx": round(m1_adx, 3),
        "m1_ema_gap_pips": round(m1_ema_gap_pips, 3),
    }
    if m5_close > m5_ema20:
        return False, diag
    if m5_rsi > EXTREMA_SHORT_SUPPORT_M5_RSI_MAX:
        return False, diag
    if (m5_plus_di - m5_minus_di) > EXTREMA_SHORT_SUPPORT_M5_DI_GAP_MAX:
        return False, diag
    if m5_ema_slope_10 > EXTREMA_SHORT_SUPPORT_M5_EMA_SLOPE_MAX:
        return False, diag
    if m1_adx > EXTREMA_SHORT_SUPPORT_M1_ADX_MAX:
        return False, diag
    if m1_ema_gap_pips > EXTREMA_SHORT_SUPPORT_M1_EMA_GAP_MAX_PIPS:
        return False, diag
    return True, diag


def _extrema_trend_gate_ok(
    side: str,
    fac_m1: Dict[str, object],
    *,
    range_ctx=None,
) -> tuple[bool, Dict[str, float]]:
    if not EXTREMA_TREND_GATE_ENABLED:
        return True, {}

    range_active = bool(getattr(range_ctx, "active", False))
    range_score = float(getattr(range_ctx, "score", 0.0) or 0.0) if range_ctx is not None else 0.0
    range_mode = str(getattr(range_ctx, "mode", "") or "").strip().upper() if range_ctx is not None else ""
    adx_val = _adx(fac_m1)
    ma_gap_pips = _ma_gap_pips(fac_m1)
    against_gap_pips = ma_gap_pips if side == "short" else -ma_gap_pips

    if range_mode == "RANGE":
        range_ready = range_score >= max(
            EXTREMA_TREND_GATE_MIN_RANGE_SCORE,
            EXTREMA_TREND_GATE_RANGE_SCORE_MIN,
        )
        range_gap_ok = (
            EXTREMA_TREND_GATE_RANGE_MAX_AGAINST_GAP_PIPS <= 0.0
            or against_gap_pips <= EXTREMA_TREND_GATE_RANGE_MAX_AGAINST_GAP_PIPS
        )
        allow = bool(range_ready and range_gap_ok)
        return allow, {
            "range_active": 1.0 if range_active else 0.0,
            "range_score": float(range_score),
            "range_mode": 0.0,
            "adx": float(adx_val),
            "ma_gap_pips": float(ma_gap_pips),
            "against_gap_pips": float(against_gap_pips),
            "range_score_floor": float(
                max(
                    EXTREMA_TREND_GATE_MIN_RANGE_SCORE,
                    EXTREMA_TREND_GATE_RANGE_SCORE_MIN,
                )
            ),
            "range_max_against_gap_pips": float(EXTREMA_TREND_GATE_RANGE_MAX_AGAINST_GAP_PIPS),
        }

    range_ready = range_active or range_score >= EXTREMA_TREND_GATE_MIN_RANGE_SCORE
    if range_mode == "TREND" and not range_ready:
        return False, {
            "range_active": 1.0 if range_active else 0.0,
            "range_score": float(range_score),
            "range_mode": 1.0,
            "adx": float(adx_val),
            "ma_gap_pips": float(ma_gap_pips),
            "against_gap_pips": float(against_gap_pips),
        }
    adx_strong = EXTREMA_TREND_GATE_ADX_MIN <= 0.0 or adx_val >= EXTREMA_TREND_GATE_ADX_MIN
    ma_gap_strong = (
        EXTREMA_TREND_GATE_MA_GAP_PIPS <= 0.0
        or against_gap_pips >= EXTREMA_TREND_GATE_MA_GAP_PIPS
    )
    continuation_strong = against_gap_pips > 0.0 and adx_strong and ma_gap_strong
    allow = range_ready or not continuation_strong

    diag = {
        "range_active": 1.0 if range_active else 0.0,
        "range_score": float(range_score),
        "range_mode": 1.0 if range_mode == "TREND" else 0.0,
        "adx": float(adx_val),
        "ma_gap_pips": float(ma_gap_pips),
        "against_gap_pips": float(against_gap_pips),
    }
    return allow, diag


def _recent_setup_pressure(side: str, range_reason: str) -> Dict[str, float]:
    reason_key = str(range_reason or "").strip().lower()
    side_key = str(side or "").strip().lower()
    if (
        not EXTREMA_SETUP_PRESSURE_ENABLED
        or not reason_key
        or not side_key
        or EXTREMA_SETUP_PRESSURE_LOOKBACK_TRADES <= 0
        or EXTREMA_SETUP_PRESSURE_LOOKBACK_HOURS <= 0.0
    ):
        return {}

    cache_key = (side_key, reason_key)
    now_mono = time.monotonic()
    cached = _SETUP_PRESSURE_CACHE.get(cache_key)
    if (
        cached is not None
        and EXTREMA_SETUP_PRESSURE_CACHE_TTL_SEC > 0.0
        and (now_mono - cached[0]) <= EXTREMA_SETUP_PRESSURE_CACHE_TTL_SEC
    ):
        return dict(cached[1])

    since_dt = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(hours=max(0.25, EXTREMA_SETUP_PRESSURE_LOOKBACK_HOURS))
    ).strftime("%Y-%m-%d %H:%M:%S")
    query = """
        SELECT
          close_reason,
          COALESCE(realized_pl, 0.0) AS realized_pl,
          COALESCE((julianday(close_time) - julianday(open_time)) * 86400.0, 0.0) AS hold_sec
        FROM trades
        WHERE strategy_tag = ?
          AND open_time >= ?
          AND json_extract(entry_thesis, '$.extrema.side') = ?
          AND lower(COALESCE(json_extract(entry_thesis, '$.range_reason'), '')) = ?
        ORDER BY open_time DESC
        LIMIT ?
    """
    summary: Dict[str, float] = {
        "trades": 0.0,
        "sl_rate": 0.0,
        "fast_sl_rate": 0.0,
        "net_jpy": 0.0,
        "active": 0.0,
    }
    con: Optional[sqlite3.Connection] = None
    try:
        con = sqlite3.connect(_TRADES_DB_URI, uri=True, timeout=1.0)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            query,
            (
                TAG,
                since_dt,
                side_key,
                reason_key,
                int(max(1, EXTREMA_SETUP_PRESSURE_LOOKBACK_TRADES)),
            ),
        ).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        if con is not None:
            con.close()

    if rows:
        trades = len(rows)
        sl_count = 0
        fast_sl_count = 0
        net_jpy = 0.0
        for row in rows:
            close_reason = str(row["close_reason"] or "").upper()
            realized_pl = _to_float(row["realized_pl"], 0.0)
            hold_sec = max(0.0, _to_float(row["hold_sec"], 0.0))
            net_jpy += realized_pl
            if close_reason == "STOP_LOSS_ORDER":
                sl_count += 1
                if hold_sec <= EXTREMA_SETUP_PRESSURE_FAST_SL_MAX_HOLD_SEC:
                    fast_sl_count += 1

        sl_rate = sl_count / max(1, trades)
        fast_sl_rate = fast_sl_count / max(1, trades)
        active = (
            trades >= max(1, EXTREMA_SETUP_PRESSURE_MIN_TRADES)
            and net_jpy < 0.0
            and sl_rate >= max(0.0, EXTREMA_SETUP_PRESSURE_SL_RATE_MIN)
            and fast_sl_rate >= max(0.0, EXTREMA_SETUP_PRESSURE_FAST_SL_RATE_MIN)
        )
        summary = {
            "trades": float(trades),
            "sl_rate": round(sl_rate, 3),
            "fast_sl_rate": round(fast_sl_rate, 3),
            "net_jpy": round(net_jpy, 3),
            "active": 1.0 if active else 0.0,
        }

    _SETUP_PRESSURE_CACHE[cache_key] = (now_mono, dict(summary))
    return summary


def _signal_extrema_reversal(
    fac_m1: Dict[str, object],
    *,
    fac_m5: Optional[Dict[str, object]] = None,
    range_ctx=None,
    tag: str,
) -> Optional[Dict[str, object]]:
    if EXTREMA_ALLOWED_REGIMES:
        regime = str(fac_m1.get("regime") or "").strip().lower()
        if not regime:
            try:
                regime = str(current_regime("M1", event_mode=False) or "").strip().lower()
            except Exception:
                regime = ""
        if regime and regime not in EXTREMA_ALLOWED_REGIMES:
            return None

    if EXTREMA_SPREAD_P25_MAX > 0.0:
        ok_spread, _ = spread_ok(max_pips=MAX_SPREAD_PIPS, p25_max=EXTREMA_SPREAD_P25_MAX)
        if not ok_spread:
            return None

    if EXTREMA_ADX_MAX > 0 and _adx(fac_m1) > EXTREMA_ADX_MAX:
        return None

    atr = _atr_pips(fac_m1)
    if EXTREMA_ATR_MAX > 0.0 and atr > EXTREMA_ATR_MAX:
        return None

    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    long_supportive, long_support_diag = _extrema_long_support_context(
        fac_m1,
        fac_m5,
        price=price,
    )
    short_supportive, short_support_diag = _extrema_short_support_context(
        fac_m1,
        fac_m5,
        price=price,
    )

    candles = get_candles_snapshot("M1", limit=max(80, EXTREMA_LOOKBACK + 8))
    snap = compute_range_snapshot(candles or [], lookback=EXTREMA_LOOKBACK, hi_pct=97.0, lo_pct=3.0)
    if not snap:
        return None

    mids, _ = tick_snapshot(EXTREMA_REV_WINDOW_SEC, limit=180)
    rev_ok, rev_dir, rev_strength = tick_reversal(mids, min_ticks=6) if mids else (False, None, 0.0)
    if not rev_ok:
        return None

    tick_strength = max(0.0, float(rev_strength or 0.0))
    if EXTREMA_TICK_STRENGTH_MIN > 0.0 and tick_strength < EXTREMA_TICK_STRENGTH_MIN:
        return None

    rsi = _rsi(fac_m1)
    recent_max = max(mids[-24:]) if mids else price
    recent_min = min(mids[-24:]) if mids else price

    high_ref = max(float(snap.high), recent_max)
    low_ref = min(float(snap.low), recent_min)
    dist_high = abs(price - high_ref) / PIP
    dist_low = abs(price - low_ref) / PIP

    short_bounce_pips = max(0.0, (recent_max - price) / PIP)
    long_bounce_pips = max(0.0, (price - recent_min) / PIP)
    long_rsi_cap = EXTREMA_RSI_LONG_MAX
    long_low_band_pips = EXTREMA_LOW_BAND_PIPS
    if long_supportive:
        long_rsi_cap = max(long_rsi_cap, EXTREMA_LONG_SUPPORT_RSI_CAP)
        long_low_band_pips = max(long_low_band_pips, EXTREMA_LONG_SUPPORT_LOW_BAND_PIPS)
    ma_gap_pips = _ma_gap_pips(fac_m1)
    range_score = (
        float(getattr(range_ctx, "score", 0.0) or 0.0) if range_ctx is not None else 0.0
    )
    range_mode = (
        str(getattr(range_ctx, "mode", "") or "").strip().upper()
        if range_ctx is not None
        else ""
    )
    adx_value = _adx(fac_m1)
    long_countertrend_block = (
        EXTREMA_LONG_COUNTERTREND_GAP_BLOCK_PIPS > 0.0
        and not long_supportive
        and ma_gap_pips <= -EXTREMA_LONG_COUNTERTREND_GAP_BLOCK_PIPS
    )
    short_countertrend_block = (
        EXTREMA_SHORT_COUNTERTREND_GAP_BLOCK_PIPS > 0.0
        and not short_supportive
        and range_mode == "RANGE"
        and ma_gap_pips >= EXTREMA_SHORT_COUNTERTREND_GAP_BLOCK_PIPS
    )
    long_shallow_probe_block = (
        not long_supportive
        and EXTREMA_LONG_SHALLOW_PROBE_DIST_LOW_MAX_PIPS > 0.0
        and dist_low <= EXTREMA_LONG_SHALLOW_PROBE_DIST_LOW_MAX_PIPS
        and long_bounce_pips <= EXTREMA_LONG_SHALLOW_PROBE_BOUNCE_MAX_PIPS
        and tick_strength <= EXTREMA_LONG_SHALLOW_PROBE_TICK_STRENGTH_MAX
        and adx_value <= EXTREMA_LONG_SHALLOW_PROBE_ADX_MAX
        and range_score <= EXTREMA_LONG_SHALLOW_PROBE_RANGE_SCORE_MAX
    )
    long_mid_rsi_probe_block = (
        not long_supportive
        and range_mode == "RANGE"
        and EXTREMA_LONG_MID_RSI_PROBE_DIST_LOW_MAX_PIPS > 0.0
        and dist_low <= EXTREMA_LONG_MID_RSI_PROBE_DIST_LOW_MAX_PIPS
        and long_bounce_pips <= EXTREMA_LONG_MID_RSI_PROBE_BOUNCE_MAX_PIPS
        and tick_strength <= EXTREMA_LONG_MID_RSI_PROBE_TICK_STRENGTH_MAX
        and adx_value >= EXTREMA_LONG_MID_RSI_PROBE_ADX_MIN
        and range_score >= EXTREMA_LONG_MID_RSI_PROBE_RANGE_SCORE_MIN
        and rsi >= EXTREMA_LONG_MID_RSI_PROBE_RSI_MIN
    )
    long_drift_probe_block = (
        not long_supportive
        and range_mode == "RANGE"
        and str(getattr(range_ctx, "reason", "") or "").strip().lower() == "volatility_compression"
        and EXTREMA_LONG_DRIFT_PROBE_DIST_LOW_MAX_PIPS > 0.0
        and dist_low <= EXTREMA_LONG_DRIFT_PROBE_DIST_LOW_MAX_PIPS
        and long_bounce_pips <= EXTREMA_LONG_DRIFT_PROBE_BOUNCE_MAX_PIPS
        and tick_strength <= EXTREMA_LONG_DRIFT_PROBE_TICK_STRENGTH_MAX
        and adx_value >= EXTREMA_LONG_DRIFT_PROBE_ADX_MIN
        and range_score >= EXTREMA_LONG_DRIFT_PROBE_RANGE_SCORE_MIN
        and ma_gap_pips >= EXTREMA_LONG_DRIFT_PROBE_MA_GAP_MIN_PIPS
        and rsi >= EXTREMA_LONG_DRIFT_PROBE_RSI_MIN
    )
    short_shallow_probe_block = (
        not short_supportive
        and range_mode == "RANGE"
        and EXTREMA_SHORT_SHALLOW_PROBE_DIST_HIGH_MAX_PIPS > 0.0
        and dist_high <= EXTREMA_SHORT_SHALLOW_PROBE_DIST_HIGH_MAX_PIPS
        and short_bounce_pips <= EXTREMA_SHORT_SHALLOW_PROBE_BOUNCE_MAX_PIPS
        and tick_strength <= EXTREMA_SHORT_SHALLOW_PROBE_TICK_STRENGTH_MAX
        and adx_value <= EXTREMA_SHORT_SHALLOW_PROBE_ADX_MAX
        and range_score >= EXTREMA_SHORT_SHALLOW_PROBE_RANGE_SCORE_MIN
        and ma_gap_pips >= EXTREMA_SHORT_SHALLOW_PROBE_MA_GAP_MIN_PIPS
    )
    short_mid_rsi_probe_block = (
        not short_supportive
        and range_mode == "RANGE"
        and EXTREMA_SHORT_MID_RSI_PROBE_DIST_HIGH_MAX_PIPS > 0.0
        and dist_high <= EXTREMA_SHORT_MID_RSI_PROBE_DIST_HIGH_MAX_PIPS
        and short_bounce_pips <= EXTREMA_SHORT_MID_RSI_PROBE_BOUNCE_MAX_PIPS
        and tick_strength <= EXTREMA_SHORT_MID_RSI_PROBE_TICK_STRENGTH_MAX
        and range_score >= EXTREMA_SHORT_MID_RSI_PROBE_RANGE_SCORE_MIN
        and rsi <= EXTREMA_SHORT_MID_RSI_PROBE_RSI_MAX
    )
    short_range_reason = str(getattr(range_ctx, "reason", "") or "").strip().lower()
    short_drift_probe_block = (
        not short_supportive
        and range_mode == "RANGE"
        and short_range_reason == "volatility_compression"
        and EXTREMA_SHORT_DRIFT_PROBE_DIST_HIGH_MAX_PIPS > 0.0
        and dist_high <= EXTREMA_SHORT_DRIFT_PROBE_DIST_HIGH_MAX_PIPS
        and short_bounce_pips <= EXTREMA_SHORT_DRIFT_PROBE_BOUNCE_MAX_PIPS
        and tick_strength <= EXTREMA_SHORT_DRIFT_PROBE_TICK_STRENGTH_MAX
        and range_score <= EXTREMA_SHORT_DRIFT_PROBE_RANGE_SCORE_MAX
        and ma_gap_pips >= EXTREMA_SHORT_DRIFT_PROBE_MA_GAP_MIN_PIPS
        and ma_gap_pips <= EXTREMA_SHORT_DRIFT_PROBE_MA_GAP_MAX_PIPS
        and rsi <= EXTREMA_SHORT_DRIFT_PROBE_RSI_MAX
    )
    long_range_reason = str(getattr(range_ctx, "reason", "") or "").strip().lower()
    long_setup_pressure_diag = _recent_setup_pressure("long", long_range_reason)
    long_setup_pressure_active = (
        long_setup_pressure_diag.get("trades", 0.0)
        >= max(1, EXTREMA_LONG_SETUP_PRESSURE_MIN_TRADES)
        and long_setup_pressure_diag.get("net_jpy", 0.0) < 0.0
        and long_setup_pressure_diag.get("sl_rate", 0.0) >= max(0.0, EXTREMA_LONG_SETUP_PRESSURE_SL_RATE_MIN)
        and long_setup_pressure_diag.get("fast_sl_rate", 0.0)
        >= max(0.0, EXTREMA_LONG_SETUP_PRESSURE_FAST_SL_RATE_MIN)
    )
    long_setup_pressure_block = (
        long_setup_pressure_active
        and not long_supportive
        and range_mode == "RANGE"
        and long_range_reason in EXTREMA_LONG_SETUP_PRESSURE_RANGE_REASONS
        and dist_low <= EXTREMA_LONG_SETUP_PRESSURE_DIST_LOW_MAX_PIPS
        and long_bounce_pips <= EXTREMA_LONG_SETUP_PRESSURE_BOUNCE_MAX_PIPS
        and tick_strength <= EXTREMA_LONG_SETUP_PRESSURE_TICK_STRENGTH_MAX
        and ma_gap_pips <= EXTREMA_LONG_SETUP_PRESSURE_MA_GAP_MAX_PIPS
        and range_score >= EXTREMA_LONG_SETUP_PRESSURE_RANGE_SCORE_MIN
        and range_score <= EXTREMA_LONG_SETUP_PRESSURE_RANGE_SCORE_MAX
        and adx_value <= EXTREMA_LONG_SETUP_PRESSURE_ADX_MAX
    )
    short_setup_pressure_diag = _recent_setup_pressure("short", short_range_reason)
    short_setup_pressure_active = bool(short_setup_pressure_diag.get("active", 0.0) >= 1.0)
    short_setup_pressure_block = (
        short_setup_pressure_active
        and range_mode == "RANGE"
        and short_range_reason in EXTREMA_SHORT_SETUP_PRESSURE_RANGE_REASONS
        and dist_high <= EXTREMA_SHORT_SETUP_PRESSURE_DIST_HIGH_MAX_PIPS
        and short_bounce_pips <= EXTREMA_SHORT_SETUP_PRESSURE_BOUNCE_MAX_PIPS
        and tick_strength <= EXTREMA_SHORT_SETUP_PRESSURE_TICK_STRENGTH_MAX
    )

    can_short = (
        EXTREMA_SHORT_ENABLED
        and rev_dir == "short"
        and dist_high <= EXTREMA_HIGH_BAND_PIPS
        and rsi >= EXTREMA_RSI_SHORT_MIN
        and short_bounce_pips >= EXTREMA_SWEEP_MIN_PIPS
        and not short_countertrend_block
        and not short_shallow_probe_block
        and not short_mid_rsi_probe_block
        and not short_drift_probe_block
        and not short_setup_pressure_block
    )
    can_long = (
        EXTREMA_LONG_ENABLED
        and rev_dir == "long"
        and dist_low <= long_low_band_pips
        and rsi <= long_rsi_cap
        and long_bounce_pips >= EXTREMA_SWEEP_MIN_PIPS
        and not long_countertrend_block
        and not long_shallow_probe_block
        and not long_mid_rsi_probe_block
        and not long_drift_probe_block
        and not long_setup_pressure_block
    )

    if not can_short and not can_long:
        return None

    if can_short and can_long:
        side = "short" if (dist_high / max(EXTREMA_HIGH_BAND_PIPS, 0.1)) <= (dist_low / max(EXTREMA_LOW_BAND_PIPS, 0.1)) else "long"
    elif can_short:
        side = "short"
    else:
        side = "long"

    trend_gate_ok, trend_diag = _extrema_trend_gate_ok(side, fac_m1, range_ctx=range_ctx)
    if not trend_gate_ok:
        return None

    sl = max(EXTREMA_SL_MIN_PIPS, min(EXTREMA_SL_MAX_PIPS, atr * EXTREMA_SL_ATR_MULT))
    tp = max(EXTREMA_TP_MIN_PIPS, min(EXTREMA_TP_MAX_PIPS, atr * EXTREMA_TP_ATR_MULT))

    confidence = 56
    confidence += int(min(12.0, abs(rsi - 50.0) * 0.7))
    confidence += int(min(10.0, tick_strength * 5.0))
    if side == "short":
        confidence += int(min(6.0, short_bounce_pips * 8.0))
    else:
        confidence += int(min(6.0, long_bounce_pips * 8.0))
        if long_supportive:
            confidence += EXTREMA_LONG_SUPPORT_CONF_BONUS
    confidence = int(max(45, min(94, confidence)))

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": confidence,
        "tag": tag,
        "reason": "extrema_reversal",
        "size_mult": round(EXTREMA_SIZE_MULT, 3),
        "extrema": {
            "side": side,
            "dist_high_pips": round(dist_high, 3),
            "dist_low_pips": round(dist_low, 3),
            "short_bounce_pips": round(short_bounce_pips, 3),
            "long_bounce_pips": round(long_bounce_pips, 3),
            "long_rsi_cap": round(long_rsi_cap, 3),
            "long_low_band_pips": round(long_low_band_pips, 3),
            "rsi": round(rsi, 2),
            "tick_strength": round(tick_strength, 4),
            "high_ref": round(high_ref, 3),
            "low_ref": round(low_ref, 3),
            "supportive_short": bool(short_supportive and side == "short"),
            "supportive_short_context": short_support_diag if short_supportive else {},
            "supportive_long": bool(long_supportive and side == "long"),
            "supportive_long_context": long_support_diag if long_supportive else {},
            "ma_gap_pips": round(ma_gap_pips, 3),
            "short_countertrend_block": bool(short_countertrend_block and side == "short"),
            "short_shallow_probe_block": bool(short_shallow_probe_block and side == "short"),
            "short_mid_rsi_probe_block": bool(short_mid_rsi_probe_block and side == "short"),
            "short_drift_probe_block": bool(short_drift_probe_block and side == "short"),
            "short_setup_pressure_block": bool(short_setup_pressure_block and side == "short"),
            "short_setup_pressure": short_setup_pressure_diag if side == "short" else {},
            "long_countertrend_block": bool(long_countertrend_block and side == "long"),
            "long_shallow_probe_block": bool(long_shallow_probe_block and side == "long"),
            "long_mid_rsi_probe_block": bool(long_mid_rsi_probe_block and side == "long"),
            "long_drift_probe_block": bool(long_drift_probe_block and side == "long"),
            "long_setup_pressure_block": bool(long_setup_pressure_block and side == "long"),
            "long_setup_pressure": long_setup_pressure_diag if side == "long" else {},
            "trend_gate": trend_diag,
        },
        "range_score": range_score,
    }


def _build_entry_thesis(signal: Dict[str, object], fac_m1: Dict[str, object], range_ctx) -> Dict[str, object]:
    conf = int(signal.get("confidence", 0) or 0)
    p_raw = _to_probability(conf)
    return {
        "strategy_tag": signal.get("tag") or TAG,
        "env_prefix": ENV_PREFIX,
        "confidence": conf,
        "entry_probability": p_raw,
        "entry_probability_raw": p_raw,
        "reason": signal.get("reason"),
        "sl_pips": signal.get("sl_pips"),
        "tp_pips": signal.get("tp_pips"),
        "range_active": bool(getattr(range_ctx, "active", False)),
        "range_score": float(getattr(range_ctx, "score", 0.0) or 0.0),
        "range_reason": getattr(range_ctx, "reason", None),
        "range_mode": getattr(range_ctx, "mode", None),
        "rsi": _rsi(fac_m1),
        "adx": _adx(fac_m1),
        "atr_pips": _atr_pips(fac_m1),
        "ma_gap_pips": _ma_gap_pips(fac_m1),
        "bbw": _to_float(fac_m1.get("bbw"), 0.0),
        "extrema": signal.get("extrema") if isinstance(signal.get("extrema"), dict) else {},
    }


async def _place_order(
    signal: Dict[str, object],
    *,
    fac_m1: Dict[str, object],
    fac_h4: Dict[str, object],
    range_ctx,
) -> Optional[str]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None

    side = "long" if signal.get("action") == "OPEN_LONG" else "short"
    sl_pips = float(signal.get("sl_pips") or 0.0)
    tp_pips = float(signal.get("tp_pips") or 0.0)
    if sl_pips <= 0:
        return None

    conf = int(signal.get("confidence", 0) or 0)
    if MIN_ENTRY_CONF > 0 and conf < MIN_ENTRY_CONF:
        return None
    conf_scale = _confidence_scale(conf, lo=CONFIDENCE_FLOOR, hi=CONFIDENCE_CEIL)
    size_mult = float(signal.get("size_mult", 1.0) or 1.0)
    size_mult = max(0.5, min(1.8, size_mult))

    try:
        from utils.oanda_account import get_account_snapshot

        free_ratio = float(get_account_snapshot().free_margin_ratio or 0.0)
    except Exception:
        free_ratio = 0.0

    atr = _atr_pips(fac_m1)
    cap_res = compute_cap(
        atr_pips=atr,
        free_ratio=free_ratio,
        range_active=bool(getattr(range_ctx, "active", False)),
        perf_pf=None,
        pos_bias=0.0,
        cap_min=CAP_MIN,
        cap_max=CAP_MAX,
        env_prefix=ENV_PREFIX,
    )
    cap = cap_res.cap
    if cap <= 0.0:
        cap = CAP_MIN

    spread_pips = None
    try:
        state = spread_monitor.get_state()
        if state is not None:
            spread_pips = float(state.get("spread_pips") or 0.0)
    except Exception:
        spread_pips = None

    sizing = compute_units(
        entry_price=price,
        sl_pips=sl_pips,
        base_entry_units=BASE_ENTRY_UNITS,
        min_units=MIN_UNITS,
        max_margin_usage=MAX_MARGIN_USAGE,
        spread_pips=float(spread_pips or 0.0),
        spread_soft_cap=MAX_SPREAD_PIPS,
        adx=_adx(fac_m1),
        signal_score=float(conf) / 100.0,
        pocket=POCKET,
        strategy_tag=str(signal.get("tag") or TAG),
        env_prefix=ENV_PREFIX,
    )

    units = int(round(sizing.units * cap * size_mult * conf_scale))
    if abs(units) < MIN_UNITS:
        return None

    if side == "short":
        units = -abs(units)
    else:
        units = abs(units)

    if side == "long":
        sl_price = round(price - sl_pips * PIP, 3)
        tp_price = round(price + tp_pips * PIP, 3) if tp_pips > 0 else None
    else:
        sl_price = round(price + sl_pips * PIP, 3)
        tp_price = round(price - tp_pips * PIP, 3) if tp_pips > 0 else None

    sl_price, tp_price = clamp_sl_tp(price=price, sl=sl_price, tp=tp_price, is_buy=(side == "long"))

    client_id = _client_order_id(str(signal.get("tag") or TAG))
    entry_thesis = _build_entry_thesis(signal, fac_m1, range_ctx)
    entry_thesis["entry_probability"] = _to_probability(conf)
    entry_thesis["entry_units_intent"] = abs(units)

    meta = {
        "cap": round(cap, 3),
        "conf_scale": round(conf_scale, 3),
        "sizing": sizing.factors if hasattr(sizing, "factors") else {},
    }

    return await market_order(
        instrument="USD_JPY",
        units=units,
        sl_price=sl_price,
        tp_price=tp_price,
        pocket=POCKET,
        client_order_id=client_id,
        strategy_tag=str(signal.get("tag") or TAG),
        entry_thesis=entry_thesis,
        meta=meta,
        confidence=conf,
    )


async def _run_worker() -> None:
    if not ENABLED:
        LOG.info("[disabled] %s", LOG_PREFIX)
        while True:
            await asyncio.sleep(3600.0)

    LOG.info("%s worker start tag=%s", LOG_PREFIX, TAG)
    pos_manager = PositionManager()
    last_entry_ts = 0.0

    while True:
        await asyncio.sleep(LOOP_INTERVAL_SEC)
        if not is_market_open(datetime.datetime.utcnow()):
            continue
        if not can_trade(POCKET):
            continue

        now_mono = time.monotonic()
        if COOLDOWN_SEC > 0.0 and now_mono - last_entry_ts < COOLDOWN_SEC:
            continue

        if MAX_OPEN_TRADES > 0 or MAX_OPEN_TRADES_GLOBAL > 0:
            try:
                positions = pos_manager.get_open_positions()
                pocket_info = positions.get(POCKET, {})
                open_trades_all = pocket_info.get("open_trades", []) or []
                if MAX_OPEN_TRADES_GLOBAL > 0 and len(open_trades_all) >= MAX_OPEN_TRADES_GLOBAL:
                    continue
                if OPEN_TRADES_SCOPE == "tag":
                    open_trades = [
                        tr
                        for tr in open_trades_all
                        if str(
                            tr.get("strategy_tag")
                            or tr.get("entry_thesis", {}).get("strategy_tag")
                            or tr.get("tag")
                            or tr.get("strategy")
                            or ""
                        ).lower()
                        == TAG.lower()
                    ]
                else:
                    open_trades = open_trades_all
                if MAX_OPEN_TRADES > 0 and len(open_trades) >= MAX_OPEN_TRADES:
                    continue
            except Exception:
                pass

        factors = all_factors()
        fac_m1 = factors.get("M1", {}) or {}
        fac_m5 = factors.get("M5", {}) or {}
        fac_h4 = factors.get("H4", {}) or {}
        if not fac_m1:
            continue

        range_ctx = detect_range_mode(fac_m1, fac_h4)
        air = evaluate_air(fac_m1, fac_h4, range_ctx=range_ctx, tag="extrema_reversal")
        if getattr(air, "enabled", False) and not bool(getattr(air, "allow_entry", True)):
            continue

        signal = _signal_extrema_reversal(
            fac_m1,
            fac_m5=fac_m5,
            range_ctx=range_ctx,
            tag=TAG,
        )
        if not signal:
            continue

        signal = adjust_signal(signal, air)
        if not signal:
            continue

        order_id = await _place_order(
            signal,
            fac_m1=fac_m1,
            fac_h4=fac_h4,
            range_ctx=range_ctx,
        )
        if order_id:
            last_entry_ts = now_mono
            LOG.info(
                "%s order placed id=%s side=%s conf=%s",
                LOG_PREFIX,
                order_id,
                signal.get("action"),
                signal.get("confidence"),
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    try:
        asyncio.run(_run_worker())
    except KeyboardInterrupt:
        raise
