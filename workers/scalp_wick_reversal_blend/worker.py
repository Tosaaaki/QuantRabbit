from __future__ import annotations

import sys
from pathlib import Path

# Allow running this worker file directly (for local replay/debug) without requiring `python -m`.
# When executed as a script, Python does not add the repo root to sys.path and relative imports fail.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    __package__ = f"workers.{Path(__file__).resolve().parent.name}"

import asyncio
import datetime
import hashlib
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from analysis.range_guard import detect_range_mode
from analysis.technique_engine import evaluate_entry_techniques
from analysis.range_model import compute_range_snapshot
from execution.strategy_entry import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from execution.stage_tracker import StageTracker
from indicators.factor_cache import all_factors, get_candles_snapshot
from market_data import tick_window, spread_monitor
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot, get_position_summary
from workers.common.dyn_cap import compute_cap
from workers.common.air_state import evaluate_air, adjust_signal
from workers.common.dyn_size import compute_units
from workers.common import perf_guard
from workers.common.quality_gate import current_regime

from . import config
from .common import (
    PIP,
    bb_entry_allowed,
    bb_levels,
    latest_mid,
    parse_hours,
    projection_decision,
    session_allowed,
    spread_ok,
    tick_imbalance,
    tick_reversal,
    tick_snapshot,
)
from .policy import wick_blend_entry_quality

LOG = logging.getLogger(__name__)


def _to_probability(value: object) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


_MODE_TAG_MAP = {
    'spread_revert': 'SpreadRangeRevert',
    'rangefaderpro': 'RangeFaderPro',
    'vwap_revert': 'VwapRevertS',
    'stoch_bounce': 'StochBollBounce',
    'divergence_revert': 'DivergenceRevert',
    'compression_retest': 'CompressionRetest',
    'htf_pullback': 'HTFPullbackS',
    'macd_trend': 'MacdTrendRide',
    'ema_slope_pull': 'EmaSlopePull',
    'tick_imbalance': 'TickImbalance',
    'tick_imbalance_rrplus': 'TickImbalanceRRPlus',
    'level_reject': 'LevelReject',
    'level_reject_plus': 'LevelRejectPlus',
    'wick_reversal': 'WickReversal',
    'wick_reversal_blend': 'WickReversalBlend',
    'wick_reversal_hf': 'WickReversalHF',
    'wick_reversal_pro': 'WickReversalPro',
    'tick_wick_reversal': 'TickWickReversal',
    'session_edge': 'SessionEdge',
    'drought_revert': 'DroughtRevert',
    'precision_lowvol': 'PrecisionLowVol',
    'liquidity_sweep': 'LiquiditySweep',
    'squeeze_pulse_break': 'SqueezePulseBreak',
    'false_break_fade': 'FalseBreakFade',
}

_RANGE_CTX_SIGNAL_NAMES = {
    "SpreadRangeRevert",
    "DroughtRevert",
    "PrecisionLowVol",
    "RangeFaderPro",
    "VwapRevertS",
    "StochBollBounce",
    "DivergenceRevert",
    "TickImbalance",
    "TickImbalanceRRPlus",
    "LevelRejectPlus",
    "WickReversalBlend",
    "WickReversalHF",
    "WickReversalPro",
    "TickWickReversal",
    "SqueezePulseBreak",
    "FalseBreakFade",
}
_M5_SIGNAL_NAMES = {"MacdTrendRide", "EmaSlopePull"}


def _mode_to_tag(mode: str) -> Optional[str]:
    return _MODE_TAG_MAP.get((mode or '').strip().lower())


def _dispatch_strategy_signal(
    *,
    name: str,
    fn: Callable[..., Optional[Dict[str, object]]],
    fac_m1: Dict[str, object],
    fac_h1: Dict[str, object],
    fac_h4: Dict[str, object],
    fac_m5: Dict[str, object],
    range_ctx,
    now_utc: datetime.datetime,
    kwargs: Dict[str, object],
) -> Optional[Dict[str, object]]:
    if name == "HTFPullbackS":
        return fn(fac_m1, fac_h1, fac_m5, **kwargs)
    if name in _M5_SIGNAL_NAMES:
        return fn(fac_m1, fac_m5, **kwargs)
    if name in {"DroughtRevert", "PrecisionLowVol"}:
        return fn(fac_m1, range_ctx, fac_m5=fac_m5, fac_h1=fac_h1, fac_h4=fac_h4, **kwargs)
    if name == "SessionEdge":
        return fn(fac_m1, range_ctx, now_utc=now_utc, **kwargs)
    if name in _RANGE_CTX_SIGNAL_NAMES:
        return fn(fac_m1, range_ctx, **kwargs)
    return fn(fac_m1, **kwargs)

# --- env helpers ---

def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except Exception:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no"}


def _env_csv(key: str, default: str = "") -> List[str]:
    raw = os.getenv(key, default)
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return items


def _perf_guard_bypass_enabled() -> bool:
    mode = str(getattr(config, "MODE", "") or "").strip().lower()
    if mode == "precision_lowvol":
        return not _env_bool("SCALP_PRECISION_LOWVOL_PERF_GUARD_ENABLED", True)
    if mode == "drought_revert":
        return not _env_bool("SCALP_PRECISION_DROUGHT_REVERT_PERF_GUARD_ENABLED", True)
    return not _env_bool("SCALP_PRECISION_PERF_GUARD_ENABLED", True)


_DROUGHT_CACHE_TS = 0.0
_DROUGHT_LAST_ENTRY: Optional[datetime.datetime] = None
_DROUGHT_LAST_OK: Optional[bool] = None


def _parse_iso_ts(raw: Optional[str]) -> Optional[datetime.datetime]:
    if not raw:
        return None
    ts = raw.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(ts)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        return parsed
    except ValueError:
        if "." in ts:
            head, tail = ts.split(".", 1)
            for sep in ("+", "-"):
                if sep in tail:
                    frac, tz = tail.split(sep, 1)
                    frac = frac[:6]
                    ts = f"{head}.{frac}{sep}{tz}"
                    break
            else:
                ts = head
            try:
                parsed = datetime.datetime.fromisoformat(ts)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=datetime.timezone.utc)
                return parsed
            except ValueError:
                return None
        return None


def _fetch_last_entry_time() -> Optional[datetime.datetime]:
    source = (config.DROUGHT_SOURCE or "trades").strip().lower()
    if source == "orders":
        db_path = "logs/orders.db"
        query = "select max(ts) from orders"
    else:
        db_path = "logs/trades.db"
        query = "select max(entry_time) from trades"
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
        cur = con.execute(query)
        row = cur.fetchone()
    except sqlite3.Error:
        return None
    finally:
        try:
            con.close()
        except Exception:
            pass
    if not row or not row[0]:
        return None
    return _parse_iso_ts(str(row[0]))


def _fetch_latest_closed_trade(tag: str) -> Optional[Tuple[datetime.datetime, float, float, int]]:
    if not tag:
        return None
    query = (
        "select close_time, entry_price, pl_pips, units "
        "from trades where strategy_tag = ? and close_time is not null "
        "order by close_time desc limit 1"
    )
    try:
        con = sqlite3.connect("file:logs/trades.db?mode=ro", uri=True, timeout=1.0)
        cur = con.execute(query, (tag,))
        row = cur.fetchone()
    except sqlite3.Error:
        return None
    finally:
        try:
            con.close()
        except Exception:
            pass
    if not row:
        return None
    close_ts = _parse_iso_ts(str(row[0] or ""))
    if close_ts is None:
        return None
    try:
        entry_price = float(row[1] or 0.0)
    except Exception:
        entry_price = 0.0
    try:
        pl_pips = float(row[2] or 0.0)
    except Exception:
        pl_pips = 0.0
    try:
        units = int(row[3] or 0)
    except Exception:
        units = 0
    if entry_price <= 0.0:
        return None
    return close_ts, entry_price, pl_pips, units


def _entry_drought_ok(now_utc: datetime.datetime) -> bool:
    if not config.DROUGHT_ENABLED:
        return True
    global _DROUGHT_CACHE_TS, _DROUGHT_LAST_ENTRY, _DROUGHT_LAST_OK
    now_mono = time.monotonic()
    if now_mono - _DROUGHT_CACHE_TS >= max(1.0, config.DROUGHT_REFRESH_SEC):
        _DROUGHT_LAST_ENTRY = _fetch_last_entry_time()
        _DROUGHT_CACHE_TS = now_mono
        _DROUGHT_LAST_OK = None
    if _DROUGHT_LAST_ENTRY is None:
        return bool(config.DROUGHT_FAIL_OPEN)
    now_ts = now_utc
    if now_ts.tzinfo is None:
        now_ts = now_ts.replace(tzinfo=datetime.timezone.utc)
    if _DROUGHT_LAST_OK is None:
        delta_sec = (now_ts - _DROUGHT_LAST_ENTRY).total_seconds()
        _DROUGHT_LAST_OK = delta_sec >= max(0.0, config.DROUGHT_MINUTES) * 60.0
    return bool(_DROUGHT_LAST_OK)


def _entry_guard_ok(snapshot) -> bool:
    if not config.ENTRY_GUARD_ENABLED or snapshot is None:
        return True
    try:
        free_ratio = snapshot.free_margin_ratio
    except Exception:
        free_ratio = None
    if free_ratio is not None and free_ratio < config.ENTRY_GUARD_MIN_FREE_MARGIN_RATIO:
        return False
    try:
        nav = float(snapshot.nav or 0.0)
        margin_used = float(snapshot.margin_used or 0.0)
    except Exception:
        nav = 0.0
        margin_used = 0.0
    if nav > 0.0 and margin_used > 0.0:
        usage = margin_used / nav
        if usage >= config.ENTRY_GUARD_MAX_MARGIN_USAGE:
            return False
    return True


def _signal_direction(signal: Dict[str, object]) -> Optional[str]:
    action = str(signal.get("action") or "").upper()
    if action == "OPEN_LONG":
        return "long"
    if action == "OPEN_SHORT":
        return "short"
    return None


# --- strategy params (defaults follow requested spec) ---
SPREAD_REV_SPREAD_P25 = _env_float("SPREAD_REV_SPREAD_P25", 0.9)
SPREAD_REV_ADX_MAX = _env_float("SPREAD_REV_ADX_MAX", 24.0)
# BBW is (upper-lower)/mid ratio (typical USD/JPY M1 ~= 0.0002..0.0020).
SPREAD_REV_BBW_MAX = _env_float("SPREAD_REV_BBW_MAX", 0.0012)
SPREAD_REV_ATR_MIN = _env_float("SPREAD_REV_ATR_MIN", 0.7)
SPREAD_REV_ATR_MAX = _env_float("SPREAD_REV_ATR_MAX", 3.2)
SPREAD_REV_RSI_LONG_MAX = _env_float("SPREAD_REV_RSI_LONG_MAX", 47.0)
SPREAD_REV_RSI_SHORT_MIN = _env_float("SPREAD_REV_RSI_SHORT_MIN", 53.0)
SPREAD_REV_BB_TOUCH_PIPS = _env_float("SPREAD_REV_BB_TOUCH_PIPS", 0.8)
SPREAD_REV_TICK_MIN = _env_int("SPREAD_REV_TICK_MIN", 6)
SPREAD_REV_RANGE_ONLY_SCORE = _env_float("SPREAD_REV_RANGE_ONLY_SCORE", 0.45)

COMPRESS_BBW_MAX = _env_float("COMPRESS_BBW_MAX", 0.00055)
COMPRESS_ATR_MAX = _env_float("COMPRESS_ATR_MAX", 2.2)
COMPRESS_BREAKOUT_PIPS = _env_float("COMPRESS_BREAKOUT_PIPS", 1.2)
COMPRESS_RETEST_BAND_PIPS = _env_float("COMPRESS_RETEST_BAND_PIPS", 0.6)
COMPRESS_RETEST_TIMEOUT_SEC = _env_float("COMPRESS_RETEST_TIMEOUT_SEC", 75.0)
COMPRESS_MIN_IMPULSE_PIPS = _env_float("COMPRESS_MIN_IMPULSE_PIPS", 0.8)
COMPRESS_MIN_TICKS = _env_int("COMPRESS_MIN_TICKS", 8)

HTF_ADX_MIN = _env_float("HTF_PULLBACK_ADX_MIN", 20.0)
HTF_GAP_PIPS = _env_float("HTF_PULLBACK_GAP_PIPS", 3.0)
HTF_PULLBACK_BAND_PIPS = _env_float("HTF_PULLBACK_BAND_PIPS", 0.9)
HTF_RSI_LONG_MAX = _env_float("HTF_PULLBACK_RSI_LONG_MAX", 48.0)
HTF_RSI_SHORT_MIN = _env_float("HTF_PULLBACK_RSI_SHORT_MIN", 52.0)
HTF_M5_BBW_MAX = _env_float("HTF_PULLBACK_BBW_MAX", 0.0032)

TICK_IMB_WINDOW_SEC = _env_float("TICK_IMB_WINDOW_SEC", 4.5)
TICK_IMB_RATIO_MIN = _env_float("TICK_IMB_RATIO_MIN", 0.68)
TICK_IMB_MOM_MIN_PIPS = _env_float("TICK_IMB_MOM_MIN_PIPS", 0.45)
TICK_IMB_RANGE_MIN_PIPS = _env_float("TICK_IMB_RANGE_MIN_PIPS", 0.25)
TICK_IMB_ATR_MIN = _env_float("TICK_IMB_ATR_MIN", 0.7)
TICK_IMB_ADX_MIN = _env_float("TICK_IMB_ADX_MIN", 18.0)
# BBW is (upper-lower)/mid ratio (typical USD/JPY M1 ~= 0.0003..0.0020).
TICK_IMB_BBW_MIN = _env_float("TICK_IMB_BBW_MIN", 0.00075)
TICK_IMB_RANGE_SCORE_MAX = _env_float("TICK_IMB_RANGE_SCORE_MAX", 0.60)
TICK_IMB_REQUIRE_MA_ALIGN = _env_int("TICK_IMB_REQUIRE_MA_ALIGN", 1)
TICK_IMB_MA_GAP_MIN_PIPS = _env_float("TICK_IMB_MA_GAP_MIN_PIPS", 0.0)
TICK_IMB_MA_ALIGN_STRICT = _env_int("TICK_IMB_MA_ALIGN_STRICT", 0)
TICK_IMB_SIZE_MULT = _env_float("TICK_IMB_SIZE_MULT", 1.25)
TICK_IMB_ALLOWED_REGIMES = {
    s.strip().lower()
    for s in _env_csv("TICK_IMB_ALLOWED_REGIMES", "")
    if s.strip()
}
TICK_IMB_BLOCK_RANGE_MODE = _env_bool("TICK_IMB_BLOCK_RANGE_MODE", True)
# Entry-quality gates (precision-first, no side/hour hard block).
TICK_IMB_ENTRY_QUALITY_ENABLED = _env_bool("TICK_IMB_ENTRY_QUALITY_ENABLED", True)
TICK_IMB_QUALITY_RATIO_MIN = _env_float("TICK_IMB_QUALITY_RATIO_MIN", 0.76)
TICK_IMB_QUALITY_MOM_MIN_PIPS = _env_float("TICK_IMB_QUALITY_MOM_MIN_PIPS", 0.65)
TICK_IMB_QUALITY_RANGE_MIN_PIPS = _env_float("TICK_IMB_QUALITY_RANGE_MIN_PIPS", 0.70)
TICK_IMB_CONFIRM_WINDOW_SEC = _env_float("TICK_IMB_CONFIRM_WINDOW_SEC", 8.0)
TICK_IMB_CONFIRM_RATIO_MIN = _env_float("TICK_IMB_CONFIRM_RATIO_MIN", 0.68)
TICK_IMB_CONFIRM_SIGNED_MOM_MIN_PIPS = _env_float("TICK_IMB_CONFIRM_SIGNED_MOM_MIN_PIPS", 0.20)
TICK_IMB_REQUIRE_CONFIRM_SIGN = _env_bool("TICK_IMB_REQUIRE_CONFIRM_SIGN", True)
TICK_IMB_REENTRY_LOOKBACK_SEC = _env_float("TICK_IMB_REENTRY_LOOKBACK_SEC", 0.0)
TICK_IMB_REENTRY_MIN_PRICE_GAP_PIPS = _env_float("TICK_IMB_REENTRY_MIN_PRICE_GAP_PIPS", 0.0)
TICK_IMB_REENTRY_REQUIRE_LAST_PROFIT = _env_bool("TICK_IMB_REENTRY_REQUIRE_LAST_PROFIT", True)
# Pattern gate (pattern_book-driven). TickImbalance currently has strong direction-only edge/avoid signals.
TICK_IMB_PATTERN_GATE_OPT_IN = _env_bool("TICK_IMB_PATTERN_GATE_OPT_IN", True)
TICK_IMB_PATTERN_GATE_ALLOW_GENERIC = _env_bool("TICK_IMB_PATTERN_GATE_ALLOW_GENERIC", True)

SPB_BBW_MAX = _env_float("SPB_BBW_MAX", 0.0016)
SPB_ATR_MIN = _env_float("SPB_ATR_MIN", 0.7)
SPB_ATR_MAX = _env_float("SPB_ATR_MAX", 4.2)
SPB_LOOKBACK = _env_int("SPB_LOOKBACK", 24)
SPB_HI_PCT = _env_float("SPB_HI_PCT", 95.0)
SPB_LO_PCT = _env_float("SPB_LO_PCT", 5.0)
SPB_BREAKOUT_PIPS = _env_float("SPB_BREAKOUT_PIPS", 0.9)
SPB_RANGE_SCORE_MIN = _env_float("SPB_RANGE_SCORE_MIN", 0.34)
SPB_TICK_WINDOW_SEC = _env_float("SPB_TICK_WINDOW_SEC", 6.0)
SPB_MIN_TICKS = _env_int("SPB_MIN_TICKS", 10)
SPB_TICK_RANGE_MAX_PIPS = _env_float("SPB_TICK_RANGE_MAX_PIPS", 1.6)
SPB_IMB_RATIO_MIN = _env_float("SPB_IMB_RATIO_MIN", 0.60)
SPB_MOM_MIN_PIPS = _env_float("SPB_MOM_MIN_PIPS", 0.55)
SPB_SPREAD_P25 = _env_float("SPB_SPREAD_P25", 1.0)
SPB_SIZE_MULT = _env_float("SPB_SIZE_MULT", 1.15)
SPB_DIAG = _env_bool("SPB_DIAG", False)
SPB_DIAG_INTERVAL_SEC = _env_float("SPB_DIAG_INTERVAL_SEC", 15.0)
SPB_ALLOWED_REGIMES = {
    s.strip().lower() for s in _env_csv("SPB_ALLOWED_REGIMES", "") if s.strip()
}
SPB_REQUIRE_AIR_MATCH = _env_bool("SPB_REQUIRE_AIR_MATCH", False)
SPB_MIN_AIR_SCORE = _env_float("SPB_MIN_AIR_SCORE", 0.0)
SPB_ALLOW_LONG = _env_bool("SPB_ALLOW_LONG", True)
SPB_ALLOW_SHORT = _env_bool("SPB_ALLOW_SHORT", True)
SPB_MAX_ADX = _env_float("SPB_MAX_ADX", 0.0)
SPB_LONG_RSI_MIN = _env_float("SPB_LONG_RSI_MIN", 0.0)
SPB_LONG_RSI_MAX = _env_float("SPB_LONG_RSI_MAX", 100.0)
SPB_SHORT_RSI_MIN = _env_float("SPB_SHORT_RSI_MIN", 0.0)
SPB_SHORT_RSI_MAX = _env_float("SPB_SHORT_RSI_MAX", 100.0)
SPB_BLOCK_JST_HOURS = parse_hours(os.getenv("SPB_BLOCK_JST_HOURS", ""))

FBF_BBW_MAX = _env_float("FBF_BBW_MAX", 0.0026)
FBF_ATR_MIN = _env_float("FBF_ATR_MIN", 0.7)
FBF_ATR_MAX = _env_float("FBF_ATR_MAX", 5.5)
FBF_ADX_MAX = _env_float("FBF_ADX_MAX", 32.0)
FBF_LOOKBACK = _env_int("FBF_LOOKBACK", 24)
FBF_HI_PCT = _env_float("FBF_HI_PCT", 95.0)
FBF_LO_PCT = _env_float("FBF_LO_PCT", 5.0)
FBF_BREAKOUT_PIPS = _env_float("FBF_BREAKOUT_PIPS", 0.8)
FBF_RECLAIM_PIPS = _env_float("FBF_RECLAIM_PIPS", 0.4)
FBF_TIMEOUT_SEC = _env_float("FBF_TIMEOUT_SEC", 55.0)
FBF_MIN_SWEEP_PIPS = _env_float("FBF_MIN_SWEEP_PIPS", 0.9)
FBF_RANGE_SCORE_MIN = _env_float("FBF_RANGE_SCORE_MIN", 0.30)
FBF_TICK_WINDOW_SEC = _env_float("FBF_TICK_WINDOW_SEC", 6.0)
FBF_REQUIRE_TICK_REVERSAL = _env_bool("FBF_REQUIRE_TICK_REVERSAL", True)
FBF_SPREAD_P25 = _env_float("FBF_SPREAD_P25", 1.1)
FBF_SIZE_MULT = _env_float("FBF_SIZE_MULT", 1.10)
# Precision filters: avoid fading against strong one-way pressure.
FBF_MAX_COUNTER_SLOPE_PIPS = _env_float("FBF_MAX_COUNTER_SLOPE_PIPS", 0.0)
FBF_MAX_COUNTER_VWAP_GAP_PIPS = _env_float("FBF_MAX_COUNTER_VWAP_GAP_PIPS", 0.0)
FBF_TICK_MIN_STRENGTH = _env_float("FBF_TICK_MIN_STRENGTH", 0.0)

TIRP_WINDOW_SEC = _env_float("TIRP_WINDOW_SEC", 5.5)
TIRP_RATIO_MIN = _env_float("TIRP_RATIO_MIN", 0.72)
TIRP_MOM_MIN_PIPS = _env_float("TIRP_MOM_MIN_PIPS", 0.55)
TIRP_RANGE_MIN_PIPS = _env_float("TIRP_RANGE_MIN_PIPS", 0.30)
TIRP_ATR_MIN = _env_float("TIRP_ATR_MIN", 0.8)
TIRP_ATR_MAX = _env_float("TIRP_ATR_MAX", 10.0)
TIRP_ADX_MIN = _env_float("TIRP_ADX_MIN", 18.0)
TIRP_BBW_MIN = _env_float("TIRP_BBW_MIN", 0.00085)
TIRP_RANGE_SCORE_MAX = _env_float("TIRP_RANGE_SCORE_MAX", 0.58)
TIRP_REQUIRE_MA_ALIGN = _env_int("TIRP_REQUIRE_MA_ALIGN", 1)
TIRP_MA_GAP_MIN_PIPS = _env_float("TIRP_MA_GAP_MIN_PIPS", 0.10)
TIRP_BLOCK_RANGE_MODE = _env_bool("TIRP_BLOCK_RANGE_MODE", False)
TIRP_SIZE_MULT = _env_float("TIRP_SIZE_MULT", 1.20)

LEVEL_LOOKBACK = _env_int("LEVEL_REJECT_LOOKBACK", 20)
LEVEL_BAND_PIPS = _env_float("LEVEL_REJECT_BAND_PIPS", 0.8)
LEVEL_RSI_LONG_MAX = _env_float("LEVEL_REJECT_RSI_LONG_MAX", 48.0)
LEVEL_RSI_SHORT_MIN = _env_float("LEVEL_REJECT_RSI_SHORT_MIN", 52.0)
LEVEL_REJECT_ADX_MAX = _env_float("LEVEL_REJECT_ADX_MAX", 30.0)
LEVEL_REJECT_ATR_MAX = _env_float("LEVEL_REJECT_ATR_MAX", 0.0)
LEVEL_REJECT_SPREAD_P25 = _env_float("LEVEL_REJECT_SPREAD_P25", 0.0)
LEVEL_REJECT_SIZE_MULT = _env_float("LEVEL_REJECT_SIZE_MULT", 1.15)
LEVEL_REJECT_ALLOWED_REGIMES = {
    s.strip().lower() for s in _env_csv("LEVEL_REJECT_ALLOWED_REGIMES", "") if s.strip()
}

# LevelRejectPlus (stricter + tick-confirmed variant; intended for higher precision)
LRP_RANGE_SCORE_MIN = _env_float("LRP_RANGE_SCORE_MIN", 0.52)
LRP_ADX_MAX = _env_float("LRP_ADX_MAX", 28.0)
LRP_BBW_MAX = _env_float("LRP_BBW_MAX", 0.0016)
LRP_SPREAD_P25 = _env_float("LRP_SPREAD_P25", 1.0)
LRP_BAND_PIPS = _env_float("LRP_BAND_PIPS", 0.9)
LRP_TICK_WINDOW_SEC = _env_float("LRP_TICK_WINDOW_SEC", 6.0)
LRP_TICK_MIN_TICKS = _env_int("LRP_TICK_MIN_TICKS", 6)
LRP_TICK_MIN_STRENGTH = _env_float("LRP_TICK_MIN_STRENGTH", 0.18)
LRP_WICK_RATIO_MIN = _env_float("LRP_WICK_RATIO_MIN", 0.30)
LRP_BODY_MAX_PIPS = _env_float("LRP_BODY_MAX_PIPS", 1.4)
LRP_BODY_RATIO_MAX = _env_float("LRP_BODY_RATIO_MAX", 0.70)
LRP_SIZE_MULT = _env_float("LRP_SIZE_MULT", 1.05)
LRP_ALLOWED_REGIMES = {
    s.strip().lower() for s in _env_csv("LRP_ALLOWED_REGIMES", "") if s.strip()
}

WICK_RANGE_MIN_PIPS = _env_float("WICK_REV_RANGE_MIN_PIPS", 2.0)
WICK_BODY_MAX_PIPS = _env_float("WICK_REV_BODY_MAX_PIPS", 0.9)
WICK_RATIO_MIN = _env_float("WICK_REV_RATIO_MIN", 0.55)
WICK_ADX_MAX = _env_float("WICK_REV_ADX_MAX", 24.0)
WICK_BBW_MAX = _env_float("WICK_REV_BBW_MAX", 0.0016)

# WickReversal Pro (strict gates; expects fewer trades but higher precision).
WICK_PRO_RANGE_SCORE_MIN = _env_float("WICK_PRO_RANGE_SCORE_MIN", 0.48)
WICK_PRO_RANGE_MIN_PIPS = _env_float("WICK_PRO_RANGE_MIN_PIPS", WICK_RANGE_MIN_PIPS)
WICK_PRO_BODY_MAX_PIPS = _env_float("WICK_PRO_BODY_MAX_PIPS", WICK_BODY_MAX_PIPS)
WICK_PRO_RATIO_MIN = _env_float("WICK_PRO_RATIO_MIN", 0.60)
WICK_PRO_ADX_MAX = _env_float("WICK_PRO_ADX_MAX", 22.0)
WICK_PRO_BBW_MAX = _env_float("WICK_PRO_BBW_MAX", 0.0011)
WICK_PRO_ATR_MIN = _env_float("WICK_PRO_ATR_MIN", 0.7)
WICK_PRO_ATR_MAX = _env_float("WICK_PRO_ATR_MAX", 6.0)
WICK_PRO_SPREAD_P25 = _env_float("WICK_PRO_SPREAD_P25", 1.0)
WICK_PRO_REQUIRE_TICK_REV = _env_bool("WICK_PRO_REQUIRE_TICK_REV", True)
WICK_PRO_TICK_WINDOW_SEC = _env_float("WICK_PRO_TICK_WINDOW_SEC", 6.0)
WICK_PRO_TICK_MIN_TICKS = _env_int("WICK_PRO_TICK_MIN_TICKS", 6)
WICK_PRO_TICK_MIN_STRENGTH = _env_float("WICK_PRO_TICK_MIN_STRENGTH", 0.25)
WICK_PRO_REQUIRE_BB_TOUCH = _env_bool("WICK_PRO_REQUIRE_BB_TOUCH", True)
WICK_PRO_BB_TOUCH_PIPS = _env_float("WICK_PRO_BB_TOUCH_PIPS", 0.9)
WICK_PRO_MIN_AIR_SCORE = _env_float("WICK_PRO_MIN_AIR_SCORE", 0.74)
WICK_PRO_MIN_EXEC_QUALITY = _env_float("WICK_PRO_MIN_EXEC_QUALITY", 0.98)
WICK_PRO_MAX_REGIME_SHIFT = _env_float("WICK_PRO_MAX_REGIME_SHIFT", 0.04)

# WickReversal HF (tick-confirmed, looser than Pro; aims for higher frequency without giving up precision).
WICK_HF_RANGE_SCORE_MIN = _env_float("WICK_HF_RANGE_SCORE_MIN", 0.45)
WICK_HF_RANGE_MIN_PIPS = _env_float("WICK_HF_RANGE_MIN_PIPS", 1.6)
WICK_HF_BODY_MAX_PIPS = _env_float("WICK_HF_BODY_MAX_PIPS", 1.1)
WICK_HF_BODY_RATIO_MAX = _env_float("WICK_HF_BODY_RATIO_MAX", 0.55)
WICK_HF_RATIO_MIN = _env_float("WICK_HF_RATIO_MIN", 0.50)
WICK_HF_ADX_MAX = _env_float("WICK_HF_ADX_MAX", 28.0)
# In higher-ADX regimes, only allow entries when wick rejection + tick reversal are exceptionally strong.
WICK_HF_ADX_OVERRIDE_ENABLED = _env_bool("WICK_HF_ADX_OVERRIDE_ENABLED", False)
WICK_HF_ADX_OVERRIDE_MAX_ADX = _env_float("WICK_HF_ADX_OVERRIDE_MAX_ADX", 55.0)
WICK_HF_ADX_OVERRIDE_MIN_RATIO = _env_float("WICK_HF_ADX_OVERRIDE_MIN_RATIO", 0.65)
WICK_HF_ADX_OVERRIDE_MIN_TICK_STRENGTH = _env_float("WICK_HF_ADX_OVERRIDE_MIN_TICK_STRENGTH", 0.20)
# If tick reversal detection is missing but wick rejection is extreme, allow entry even in high ADX.
WICK_HF_ADX_OVERRIDE_ALLOW_NO_TICK_REV = _env_bool("WICK_HF_ADX_OVERRIDE_ALLOW_NO_TICK_REV", True)
WICK_HF_ADX_OVERRIDE_NO_TICK_MIN_RATIO = _env_float("WICK_HF_ADX_OVERRIDE_NO_TICK_MIN_RATIO", 0.80)

WICK_HF_BBW_MAX = _env_float("WICK_HF_BBW_MAX", 0.0016)
WICK_HF_ATR_MIN = _env_float("WICK_HF_ATR_MIN", 0.7)
WICK_HF_ATR_MAX = _env_float("WICK_HF_ATR_MAX", 7.0)
WICK_HF_SPREAD_P25 = _env_float("WICK_HF_SPREAD_P25", 1.0)
WICK_HF_REQUIRE_TICK_REV = _env_bool("WICK_HF_REQUIRE_TICK_REV", True)
WICK_HF_TICK_WINDOW_SEC = _env_float("WICK_HF_TICK_WINDOW_SEC", 8.0)
WICK_HF_TICK_MIN_TICKS = _env_int("WICK_HF_TICK_MIN_TICKS", 8)
WICK_HF_TICK_MIN_STRENGTH = _env_float("WICK_HF_TICK_MIN_STRENGTH", 0.25)
# Entry precision first: avoid fading strong momentum.
WICK_HF_MOMENTUM_FILTER_ENABLED = _env_bool("WICK_HF_MOMENTUM_FILTER_ENABLED", False)
WICK_HF_MACD_HIST_LONG_MIN = _env_float("WICK_HF_MACD_HIST_LONG_MIN", -0.15)
WICK_HF_MACD_HIST_SHORT_MAX = _env_float("WICK_HF_MACD_HIST_SHORT_MAX", 0.35)
WICK_HF_EMA_SLOPE_10_SHORT_MAX = _env_float("WICK_HF_EMA_SLOPE_10_SHORT_MAX", 0.15)

# Allow hard-wick fallback when tick reversal is missing/weak (direction mismatch stays blocked).
WICK_HF_TICK_FALLBACK_HARD_WICK = _env_bool("WICK_HF_TICK_FALLBACK_HARD_WICK", False)
WICK_HF_HARD_WICK_EXTRA_RATIO = _env_float("WICK_HF_HARD_WICK_EXTRA_RATIO", 0.18)
WICK_HF_HARD_WICK_RANGE_MULT = _env_float("WICK_HF_HARD_WICK_RANGE_MULT", 1.40)

WICK_HF_REQUIRE_BB_TOUCH = _env_bool("WICK_HF_REQUIRE_BB_TOUCH", True)
WICK_HF_BB_TOUCH_PIPS = _env_float("WICK_HF_BB_TOUCH_PIPS", 0.9)
WICK_HF_DIAG = _env_bool("WICK_HF_DIAG", False)
WICK_HF_DIAG_INTERVAL_SEC = _env_float("WICK_HF_DIAG_INTERVAL_SEC", 20.0)

# WickReversal Blend (band-touch + tick reversal; strict trend filters)
WICK_BLEND_RANGE_SCORE_MIN = _env_float("WICK_BLEND_RANGE_SCORE_MIN", 0.45)
WICK_BLEND_SPREAD_P25 = _env_float("WICK_BLEND_SPREAD_P25", 1.0)
WICK_BLEND_ADX_MIN = _env_float("WICK_BLEND_ADX_MIN", 0.0)
WICK_BLEND_ADX_MAX = _env_float("WICK_BLEND_ADX_MAX", 24.0)
WICK_BLEND_BBW_MAX = _env_float("WICK_BLEND_BBW_MAX", 0.0014)
WICK_BLEND_ATR_MIN = _env_float("WICK_BLEND_ATR_MIN", 0.8)
WICK_BLEND_ATR_MAX = _env_float("WICK_BLEND_ATR_MAX", 4.0)
WICK_BLEND_RANGE_MIN_PIPS = _env_float("WICK_BLEND_RANGE_MIN_PIPS", 1.0)
WICK_BLEND_BODY_MAX_PIPS = _env_float("WICK_BLEND_BODY_MAX_PIPS", 2.2)
WICK_BLEND_BODY_RATIO_MAX = _env_float("WICK_BLEND_BODY_RATIO_MAX", 0.75)
WICK_BLEND_WICK_RATIO_MIN = _env_float("WICK_BLEND_WICK_RATIO_MIN", 0.35)
WICK_BLEND_BB_TOUCH_PIPS = _env_float("WICK_BLEND_BB_TOUCH_PIPS", 1.1)
WICK_BLEND_BB_TOUCH_RATIO = _env_float("WICK_BLEND_BB_TOUCH_RATIO", 0.22)
WICK_BLEND_REQUIRE_TICK_REV = _env_bool("WICK_BLEND_REQUIRE_TICK_REV", True)
WICK_BLEND_TICK_WINDOW_SEC = _env_float("WICK_BLEND_TICK_WINDOW_SEC", 10.0)
WICK_BLEND_TICK_MIN_TICKS = _env_int("WICK_BLEND_TICK_MIN_TICKS", 6)
WICK_BLEND_TICK_MIN_STRENGTH = _env_float("WICK_BLEND_TICK_MIN_STRENGTH", 0.28)
# Require a small follow-through away from the band after the rejection candle to avoid "too early" fades.
WICK_BLEND_FOLLOW_PIPS = _env_float("WICK_BLEND_FOLLOW_PIPS", 0.0)
WICK_BLEND_EXTREME_RETRACE_MIN_PIPS = _env_float("WICK_BLEND_EXTREME_RETRACE_MIN_PIPS", 0.0)
WICK_BLEND_DIAG = _env_bool("WICK_BLEND_DIAG", False)
WICK_BLEND_DIAG_INTERVAL_SEC = _env_float("WICK_BLEND_DIAG_INTERVAL_SEC", 20.0)
WICK_BLEND_LONG_SETUP_PRESSURE_ENABLED = _env_bool("WICK_BLEND_LONG_SETUP_PRESSURE_ENABLED", True)
WICK_BLEND_LONG_SETUP_PRESSURE_LOOKBACK_MINUTES = _env_float(
    "WICK_BLEND_LONG_SETUP_PRESSURE_LOOKBACK_MINUTES", 1440.0
)
WICK_BLEND_LONG_SETUP_PRESSURE_LOOKBACK_TRADES = _env_int(
    "WICK_BLEND_LONG_SETUP_PRESSURE_LOOKBACK_TRADES", 6
)
WICK_BLEND_LONG_SETUP_PRESSURE_MIN_TRADES = _env_int(
    "WICK_BLEND_LONG_SETUP_PRESSURE_MIN_TRADES", 2
)
WICK_BLEND_LONG_SETUP_PRESSURE_ACTIVE_MAX_AGE_SEC = _env_float(
    "WICK_BLEND_LONG_SETUP_PRESSURE_ACTIVE_MAX_AGE_SEC", 43200.0
)
WICK_BLEND_LONG_SETUP_PRESSURE_FAST_SL_MAX_HOLD_SEC = _env_float(
    "WICK_BLEND_LONG_SETUP_PRESSURE_FAST_SL_MAX_HOLD_SEC", 35.0
)
WICK_BLEND_LONG_SETUP_PRESSURE_STOP_LOSS_STREAK_MIN = _env_int(
    "WICK_BLEND_LONG_SETUP_PRESSURE_STOP_LOSS_STREAK_MIN", 2
)
WICK_BLEND_LONG_SETUP_PRESSURE_FAST_STOP_LOSS_STREAK_MIN = _env_int(
    "WICK_BLEND_LONG_SETUP_PRESSURE_FAST_STOP_LOSS_STREAK_MIN", 2
)
WICK_BLEND_LONG_SETUP_PRESSURE_NET_LOSS_MIN_JPY = _env_float(
    "WICK_BLEND_LONG_SETUP_PRESSURE_NET_LOSS_MIN_JPY", 10.0
)
WICK_BLEND_LONG_SETUP_PRESSURE_CACHE_TTL_SEC = _env_float(
    "WICK_BLEND_LONG_SETUP_PRESSURE_CACHE_TTL_SEC", 15.0
)
WICK_BLEND_LONG_SETUP_PRESSURE_BBW_MAX = _env_float(
    "WICK_BLEND_LONG_SETUP_PRESSURE_BBW_MAX", 0.00055
)
WICK_BLEND_LONG_SETUP_PRESSURE_RANGE_SCORE_MIN = _env_float(
    "WICK_BLEND_LONG_SETUP_PRESSURE_RANGE_SCORE_MIN", 0.40
)
WICK_BLEND_LONG_SETUP_PRESSURE_RSI_MAX = _env_float(
    "WICK_BLEND_LONG_SETUP_PRESSURE_RSI_MAX", 50.0
)
WICK_BLEND_LONG_SETUP_PRESSURE_QUALITY_MAX = _env_float(
    "WICK_BLEND_LONG_SETUP_PRESSURE_QUALITY_MAX", 0.83
)
WICK_BLEND_LONG_SETUP_PRESSURE_PROJECTION_SCORE_MAX = _env_float(
    "WICK_BLEND_LONG_SETUP_PRESSURE_PROJECTION_SCORE_MAX", 0.15
)
WICK_BLEND_SHORT_COUNTERTREND_GUARD_ENABLED = _env_bool(
    "WICK_BLEND_SHORT_COUNTERTREND_GUARD_ENABLED", True
)
WICK_BLEND_SHORT_COUNTERTREND_PROJECTION_SCORE_MIN = _env_float(
    "WICK_BLEND_SHORT_COUNTERTREND_PROJECTION_SCORE_MIN", 0.10
)
WICK_BLEND_SHORT_COUNTERTREND_QUALITY_MAX = _env_float(
    "WICK_BLEND_SHORT_COUNTERTREND_QUALITY_MAX", 0.78
)
WICK_BLEND_SHORT_COUNTERTREND_RSI_MAX = _env_float(
    "WICK_BLEND_SHORT_COUNTERTREND_RSI_MAX", 58.0
)
WICK_BLEND_SHORT_COUNTERTREND_ADX_MAX = _env_float(
    "WICK_BLEND_SHORT_COUNTERTREND_ADX_MAX", 20.0
)
WICK_BLEND_SHORT_COUNTERTREND_MACD_HIST_PIPS_MIN = _env_float(
    "WICK_BLEND_SHORT_COUNTERTREND_MACD_HIST_PIPS_MIN", 0.12
)

# Tick-window wick reversal (higher-frequency range-reversion).
TICK_WICK_WINDOW_SEC = _env_float("TICK_WICK_WINDOW_SEC", 9.0)
TICK_WICK_MIN_TICKS = _env_int("TICK_WICK_MIN_TICKS", 18)
TICK_WICK_RANGE_MIN_PIPS = _env_float("TICK_WICK_RANGE_MIN_PIPS", 0.75)
TICK_WICK_BODY_MAX_PIPS = _env_float("TICK_WICK_BODY_MAX_PIPS", 0.25)
TICK_WICK_RATIO_MIN = _env_float("TICK_WICK_RATIO_MIN", 0.62)
TICK_WICK_RANGE_SCORE_MIN = _env_float("TICK_WICK_RANGE_SCORE_MIN", 0.48)
TICK_WICK_ADX_MAX = _env_float("TICK_WICK_ADX_MAX", 23.0)
TICK_WICK_BBW_MAX = _env_float("TICK_WICK_BBW_MAX", 0.0016)
TICK_WICK_SPREAD_P25 = _env_float("TICK_WICK_SPREAD_P25", 1.0)
TICK_WICK_REQUIRE_BB_TOUCH = _env_bool("TICK_WICK_REQUIRE_BB_TOUCH", True)
TICK_WICK_BB_TOUCH_PIPS = _env_float("TICK_WICK_BB_TOUCH_PIPS", 0.9)
TICK_WICK_REQUIRE_TICK_REV = _env_bool("TICK_WICK_REQUIRE_TICK_REV", True)
TICK_WICK_MIN_REV_STRENGTH = _env_float("TICK_WICK_MIN_REV_STRENGTH", 0.25)

# Optional extra gates based on AIR (applied after adjust_signal)
TICK_WICK_MIN_AIR_SCORE = _env_float("TICK_WICK_MIN_AIR_SCORE", 0.72)
TICK_WICK_MIN_EXEC_QUALITY = _env_float("TICK_WICK_MIN_EXEC_QUALITY", 0.97)
TICK_WICK_MAX_REGIME_SHIFT = _env_float("TICK_WICK_MAX_REGIME_SHIFT", 0.06)

TICK_WICK_DIAG = _env_bool("TICK_WICK_DIAG", False)
TICK_WICK_DIAG_INTERVAL_SEC = _env_float("TICK_WICK_DIAG_INTERVAL_SEC", 15.0)
TICK_WICK_CLAMP_MIN_UNITS = _env_bool("TICK_WICK_CLAMP_MIN_UNITS", True)
TICK_WICK_MIN_UNITS_CLAMP_RATIO = _env_float("TICK_WICK_MIN_UNITS_CLAMP_RATIO", 0.85)

_LAST_TICK_WICK_PLACE_DIAG_TS = 0.0
_LAST_WICK_BLEND_DIAG_TS = 0.0
_DROUGHT_SETUP_PRESSURE_CACHE: Dict[Tuple[str, str], Tuple[float, Dict[str, float]]] = {}
_PREC_LOWVOL_SETUP_PRESSURE_CACHE: Dict[Tuple[str, str], Tuple[float, Dict[str, float]]] = {}
_WICK_BLEND_LONG_SETUP_PRESSURE_CACHE: Dict[str, Tuple[float, Dict[str, float]]] = {}


class _NoopStageTracker:
    def update_loss_streaks(self, *args, **kwargs) -> None:
        return None

    def is_strategy_blocked(self, *args, **kwargs):
        return False, 0.0, ""

    def is_blocked(self, *args, **kwargs):
        return False, 0.0, ""

    def close(self) -> None:
        return None


LSR_LOOKBACK = _env_int("LSR_LOOKBACK", 20)
LSR_SWEEP_PIPS = _env_float("LSR_SWEEP_PIPS", 0.45)
LSR_RECLAIM_PIPS = _env_float("LSR_RECLAIM_PIPS", 0.1)
LSR_BODY_MAX_PIPS = _env_float("LSR_BODY_MAX_PIPS", 1.4)
LSR_RANGE_MIN_PIPS = _env_float("LSR_RANGE_MIN_PIPS", 1.2)
LSR_WICK_RATIO_MIN = _env_float("LSR_WICK_RATIO_MIN", 0.5)
LSR_ADX_MAX = _env_float("LSR_ADX_MAX", 30.0)
LSR_BBW_MAX = _env_float("LSR_BBW_MAX", 0.0018)
LSR_RANGE_SCORE_MIN = _env_float("LSR_RANGE_SCORE_MIN", 0.35)
LSR_REQUIRE_TICK_REVERSAL = _env_bool("LSR_REQUIRE_TICK_REVERSAL", True)
LSR_SIZE_MULT = _env_float("LSR_SIZE_MULT", 1.0)
LSR_ALLOWED_REGIMES = {
    s.strip().lower() for s in _env_csv("LSR_ALLOWED_REGIMES", "") if s.strip()
}

SESSION_ALLOW_HOURS = parse_hours(os.getenv("SESSION_EDGE_ALLOW_HOURS_JST", "12,13,18-22"))
SESSION_BLOCK_HOURS = parse_hours(os.getenv("SESSION_EDGE_BLOCK_HOURS_JST", "00-02,16"))

VWAP_REV_GAP_MIN = _env_float("VWAP_REV_GAP_MIN", 1.2)
VWAP_REV_BB_TOUCH_PIPS = _env_float("VWAP_REV_BB_TOUCH_PIPS", 0.9)
VWAP_REV_RSI_LONG_MAX = _env_float("VWAP_REV_RSI_LONG_MAX", 46.0)
VWAP_REV_RSI_SHORT_MIN = _env_float("VWAP_REV_RSI_SHORT_MIN", 54.0)
VWAP_REV_STOCH_LONG_MAX = _env_float("VWAP_REV_STOCH_LONG_MAX", 0.2)
VWAP_REV_STOCH_SHORT_MIN = _env_float("VWAP_REV_STOCH_SHORT_MIN", 0.8)
VWAP_REV_ADX_MAX = _env_float("VWAP_REV_ADX_MAX", 22.0)
VWAP_REV_BBW_MAX = _env_float("VWAP_REV_BBW_MAX", 0.0014)
VWAP_REV_ATR_MIN = _env_float("VWAP_REV_ATR_MIN", 0.7)
VWAP_REV_ATR_MAX = _env_float("VWAP_REV_ATR_MAX", 3.2)
VWAP_REV_SPREAD_P25 = _env_float("VWAP_REV_SPREAD_P25", 0.9)
VWAP_REV_RANGE_SCORE = _env_float("VWAP_REV_RANGE_SCORE", 0.4)

STOCH_BOUNCE_STOCH_LONG_MAX = _env_float("STOCH_BOUNCE_STOCH_LONG_MAX", 0.18)
STOCH_BOUNCE_STOCH_SHORT_MIN = _env_float("STOCH_BOUNCE_STOCH_SHORT_MIN", 0.82)
STOCH_BOUNCE_RSI_LONG_MAX = _env_float("STOCH_BOUNCE_RSI_LONG_MAX", 46.0)
STOCH_BOUNCE_RSI_SHORT_MIN = _env_float("STOCH_BOUNCE_RSI_SHORT_MIN", 54.0)
STOCH_BOUNCE_ADX_MAX = _env_float("STOCH_BOUNCE_ADX_MAX", 22.0)
STOCH_BOUNCE_BBW_MAX = _env_float("STOCH_BOUNCE_BBW_MAX", 0.0014)
STOCH_BOUNCE_ATR_MIN = _env_float("STOCH_BOUNCE_ATR_MIN", 0.7)
STOCH_BOUNCE_ATR_MAX = _env_float("STOCH_BOUNCE_ATR_MAX", 3.0)
STOCH_BOUNCE_MACD_ABS_MAX = _env_float("STOCH_BOUNCE_MACD_ABS_MAX", 0.6)
STOCH_BOUNCE_BB_TOUCH_PIPS = _env_float("STOCH_BOUNCE_BB_TOUCH_PIPS", 0.9)
STOCH_BOUNCE_RANGE_SCORE = _env_float("STOCH_BOUNCE_RANGE_SCORE", 0.35)

MACD_TREND_ADX_MIN = _env_float("MACD_TREND_ADX_MIN", 20.0)
MACD_TREND_BBW_MIN = _env_float("MACD_TREND_BBW_MIN", 0.00075)
MACD_TREND_ATR_MIN = _env_float("MACD_TREND_ATR_MIN", 0.9)
MACD_TREND_HIST_MIN = _env_float("MACD_TREND_HIST_MIN", 0.25)
MACD_TREND_SLOPE_MIN = _env_float("MACD_TREND_SLOPE_MIN", 0.05)
MACD_TREND_RSI_LONG_MIN = _env_float("MACD_TREND_RSI_LONG_MIN", 52.0)
MACD_TREND_RSI_SHORT_MAX = _env_float("MACD_TREND_RSI_SHORT_MAX", 48.0)
MACD_TREND_VWAP_GAP_MAX = _env_float("MACD_TREND_VWAP_GAP_MAX", 2.2)
MACD_TREND_M5_SLOPE_MIN = _env_float("MACD_TREND_M5_SLOPE_MIN", 0.03)

DIV_REVERT_SCORE_MIN = _env_float("DIV_REVERT_SCORE_MIN", 0.32)
DIV_REVERT_MAX_AGE = _env_int("DIV_REVERT_MAX_AGE", 6)
DIV_REVERT_MIN_STRENGTH = _env_float("DIV_REVERT_MIN_STRENGTH", 0.35)
DIV_REVERT_RSI_LONG_MAX = _env_float("DIV_REVERT_RSI_LONG_MAX", 47.0)
DIV_REVERT_RSI_SHORT_MIN = _env_float("DIV_REVERT_RSI_SHORT_MIN", 53.0)
DIV_REVERT_BB_TOUCH_PIPS = _env_float("DIV_REVERT_BB_TOUCH_PIPS", 0.9)
DIV_REVERT_ADX_MAX = _env_float("DIV_REVERT_ADX_MAX", 24.0)

EMA_PULL_SLOPE_MIN = _env_float("EMA_PULL_SLOPE_MIN", 0.04)
EMA_PULL_BAND_PIPS = _env_float("EMA_PULL_BAND_PIPS", 0.8)
EMA_PULL_RSI_LONG_MIN = _env_float("EMA_PULL_RSI_LONG_MIN", 46.0)
EMA_PULL_RSI_LONG_MAX = _env_float("EMA_PULL_RSI_LONG_MAX", 60.0)
EMA_PULL_RSI_SHORT_MIN = _env_float("EMA_PULL_RSI_SHORT_MIN", 40.0)
EMA_PULL_RSI_SHORT_MAX = _env_float("EMA_PULL_RSI_SHORT_MAX", 54.0)
EMA_PULL_VWAP_GAP_MAX = _env_float("EMA_PULL_VWAP_GAP_MAX", 1.0)
EMA_PULL_MACD_ABS_MAX = _env_float("EMA_PULL_MACD_ABS_MAX", 0.8)
EMA_PULL_ADX_MIN = _env_float("EMA_PULL_ADX_MIN", 16.0)
EMA_PULL_STOCH_MIN = _env_float("EMA_PULL_STOCH_MIN", 0.25)
EMA_PULL_STOCH_MAX = _env_float("EMA_PULL_STOCH_MAX", 0.75)


@dataclass
class RetestState:
    direction: str
    level: float
    started_ts: float
    expires_ts: float
    breakout_pips: float


_RETEST_STATE: Dict[str, RetestState] = {}

@dataclass
class FalseBreakState:
    direction: str  # fade direction
    side: str  # "high" or "low"
    level: float
    started_ts: float
    expires_ts: float
    extreme: float


_FALSE_BREAK_STATE: Dict[str, FalseBreakState] = {}


def _confidence_scale(conf: int, *, lo: int, hi: int) -> float:
    if conf <= lo:
        return 0.5
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.5 + span * 0.5


def _client_order_id(tag: str) -> str:
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "scalp"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-scalp-{sanitized}{digest}"


def _latest_price(fac_m1: Dict[str, object]) -> float:
    try:
        price = float(fac_m1.get("close") or 0.0)
    except Exception:
        price = 0.0
    return latest_mid(price)


def _atr_pips(fac_m1: Dict[str, object]) -> float:
    try:
        return float(fac_m1.get("atr_pips") or 0.0)
    except Exception:
        return 0.0


def _rsi(fac_m1: Dict[str, object]) -> float:
    try:
        return float(fac_m1.get("rsi") or 50.0)
    except Exception:
        return 50.0


def _adx(fac_m1: Dict[str, object]) -> float:
    try:
        return float(fac_m1.get("adx") or 0.0)
    except Exception:
        return 0.0


def _bbw(fac_m1: Dict[str, object]) -> float:
    try:
        return float(fac_m1.get("bbw") or 0.0)
    except Exception:
        return 0.0

def _stoch_rsi(fac_m1: Dict[str, object]) -> float:
    try:
        return float(fac_m1.get("stoch_rsi") or 0.0)
    except Exception:
        return 0.0


def _macd_hist_pips(fac_m1: Dict[str, object]) -> float:
    try:
        return float(fac_m1.get("macd_hist") or 0.0) / PIP
    except Exception:
        return 0.0


def _macd_pips(fac_m1: Dict[str, object]) -> float:
    try:
        return float(fac_m1.get("macd") or 0.0) / PIP
    except Exception:
        return 0.0


def _macd_signal_pips(fac_m1: Dict[str, object]) -> float:
    try:
        return float(fac_m1.get("macd_signal") or 0.0) / PIP
    except Exception:
        return 0.0


def _ema_slope_pips(fac_m1: Dict[str, object], key: str) -> float:
    try:
        return float(fac_m1.get(key) or 0.0) / PIP
    except Exception:
        return 0.0


def _vwap_gap_pips(fac_m1: Dict[str, object]) -> float:
    try:
        return float(fac_m1.get("vwap_gap") or 0.0)
    except Exception:
        return 0.0


def _plus_di(fac_m1: Dict[str, object]) -> float:
    try:
        return float(fac_m1.get("plus_di") or 0.0)
    except Exception:
        return 0.0


def _minus_di(fac_m1: Dict[str, object]) -> float:
    try:
        return float(fac_m1.get("minus_di") or 0.0)
    except Exception:
        return 0.0


def _div_score(fac_m1: Dict[str, object]) -> float:
    try:
        rsi_score = float(fac_m1.get("div_rsi_score") or 0.0)
    except Exception:
        rsi_score = 0.0
    try:
        macd_score = float(fac_m1.get("div_macd_score") or 0.0)
    except Exception:
        macd_score = 0.0
    return rsi_score * 0.6 + macd_score * 0.4


def _unit_bound(value: object) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, numeric))


def _positive_norm(value: object, scale: float, *, offset: float = 0.0) -> float:
    try:
        numeric = float(value) - float(offset)
    except Exception:
        return 0.0
    if scale <= 0.0:
        return 0.0
    return _unit_bound(max(0.0, numeric) / float(scale))


def _mtf_frame_flow_snapshot(
    fac: Optional[Dict[str, object]],
) -> Dict[str, float | str]:
    if not isinstance(fac, dict) or not fac:
        return {}
    adx = _adx(fac)
    plus_di = _plus_di(fac)
    minus_di = _minus_di(fac)
    di_gap = plus_di - minus_di
    ma_fast = fac.get("ma10")
    if ma_fast is None:
        ma_fast = fac.get("ema12")
    if ma_fast is None:
        ma_fast = fac.get("ema20")
    ma_slow = fac.get("ma20")
    if ma_slow is None:
        ma_slow = fac.get("ema24")
    if ma_slow is None:
        ma_slow = fac.get("vwap")
    try:
        ma_gap_pips = ((float(ma_fast) - float(ma_slow)) / PIP) if ma_fast is not None and ma_slow is not None else 0.0
    except Exception:
        ma_gap_pips = 0.0
    slope_10 = _ema_slope_pips(fac, "ema_slope_10")
    slope_20 = _ema_slope_pips(fac, "ema_slope_20")
    long_score = _unit_bound(
        0.34 * _positive_norm(ma_gap_pips, 4.0)
        + 0.24 * _positive_norm(di_gap, 10.0)
        + 0.18 * _positive_norm(adx - 14.0, 14.0)
        + 0.14 * _positive_norm(slope_10, 0.18)
        + 0.10 * _positive_norm(slope_20, 0.12)
    )
    short_score = _unit_bound(
        0.34 * _positive_norm(-ma_gap_pips, 4.0)
        + 0.24 * _positive_norm(-di_gap, 10.0)
        + 0.18 * _positive_norm(adx - 14.0, 14.0)
        + 0.14 * _positive_norm(-slope_10, 0.18)
        + 0.10 * _positive_norm(-slope_20, 0.12)
    )
    direction = "neutral"
    flow_regime = "transition"
    strength = max(long_score, short_score)
    if strength >= 0.30:
        if long_score > short_score:
            direction = "long"
            flow_regime = "trend_long"
        elif short_score > long_score:
            direction = "short"
            flow_regime = "trend_short"
    return {
        "flow_regime": flow_regime,
        "direction": direction,
        "strength": round(strength, 3),
        "ma_gap_pips": round(ma_gap_pips, 3),
        "adx": round(adx, 3),
    }


def _reversion_mtf_context(
    *,
    side: str,
    fac_m5: Optional[Dict[str, object]] = None,
    fac_h1: Optional[Dict[str, object]] = None,
    fac_h4: Optional[Dict[str, object]] = None,
) -> Dict[str, float | str]:
    weights = (("m5", fac_m5, 0.9), ("h1", fac_h1, 1.15), ("h4", fac_h4, 1.35))
    long_bias = 0.0
    short_bias = 0.0
    total_bias = 0.0
    context: Dict[str, float | str] = {}
    for label, fac, weight in weights:
        snapshot = _mtf_frame_flow_snapshot(fac)
        if not snapshot:
            continue
        flow_regime = str(snapshot.get("flow_regime") or "transition")
        direction = str(snapshot.get("direction") or "neutral")
        strength = float(snapshot.get("strength") or 0.0)
        context[f"{label}_flow_regime"] = flow_regime
        context[f"{label}_trend_strength"] = round(strength, 3)
        if flow_regime in {"trend_long", "trend_short"}:
            weighted = strength * weight
            total_bias += weighted
            if direction == "long":
                long_bias += weighted
            elif direction == "short":
                short_bias += weighted
    if total_bias <= 0.0:
        context["macro_flow_regime"] = "transition"
        context["mtf_alignment"] = "neutral"
        context["mtf_countertrend_pressure"] = 0.0
        context["mtf_aligned_support"] = 0.0
        context["macro_trend_strength"] = 0.0
        return context

    bias_gap = abs(long_bias - short_bias) / max(total_bias, 1e-6)
    if bias_gap <= 0.18:
        macro_flow_regime = "transition"
        mtf_alignment = "mixed"
    elif long_bias > short_bias:
        macro_flow_regime = "trend_long"
        mtf_alignment = "aligned" if side == "long" else "countertrend"
    else:
        macro_flow_regime = "trend_short"
        mtf_alignment = "aligned" if side == "short" else "countertrend"
    aligned_bias = long_bias if side == "long" else short_bias if side == "short" else 0.0
    countertrend_bias = short_bias if side == "long" else long_bias if side == "short" else 0.0
    context["macro_flow_regime"] = macro_flow_regime
    context["mtf_alignment"] = mtf_alignment
    context["mtf_countertrend_pressure"] = round(_unit_bound(countertrend_bias / max(total_bias, 1e-6)), 3)
    context["mtf_aligned_support"] = round(_unit_bound(aligned_bias / max(total_bias, 1e-6)), 3)
    context["macro_trend_strength"] = round(_unit_bound(max(long_bias, short_bias) / 2.2), 3)
    return context


def _reversion_short_flow_guard(
    *,
    fac_m1: Dict[str, object],
    price: float,
    dist_upper_pips: float,
    band_pips: float,
    range_score: float,
    rev_strength: float,
    profile: str,
    mtf_context: Optional[Dict[str, object]] = None,
) -> Tuple[bool, Dict[str, float]]:
    try:
        adx = float(fac_m1.get("adx") or 0.0)
    except Exception:
        adx = 0.0
    anchor = 0.0
    for key in ("ema20", "ma20", "vwap"):
        try:
            candidate = float(fac_m1.get(key) or 0.0)
        except Exception:
            candidate = 0.0
        if candidate > 0.0:
            anchor = candidate
            break
    price_gap_pips = ((price - anchor) / PIP) if anchor > 0.0 else 0.0
    slope_10 = _ema_slope_pips(fac_m1, "ema_slope_10")
    slope_20 = _ema_slope_pips(fac_m1, "ema_slope_20")
    macd_hist = _macd_hist_pips(fac_m1)
    vgap = _vwap_gap_pips(fac_m1)
    di_gap = max(0.0, _plus_di(fac_m1) - _minus_di(fac_m1))

    touch_ratio = _unit_bound((max(0.0, band_pips) - max(0.0, dist_upper_pips)) / max(0.15, band_pips))
    range_support = _unit_bound((range_score - 0.25) / 0.45)
    reversal_support = _unit_bound((rev_strength - 0.18) / 0.72)
    vgap_support = _positive_norm(vgap, 1.4) * (0.55 + touch_ratio * 0.45)
    price_extension = _positive_norm(
        price_gap_pips,
        max(1.2, band_pips * 1.4),
        offset=max(0.2, band_pips * 0.35),
    )
    adx_reversion = _unit_bound((28.0 - adx) / 12.0)
    trend_stack = _unit_bound(
        0.34 * _positive_norm(slope_10, 0.14)
        + 0.22 * _positive_norm(slope_20, 0.10)
        + 0.20 * _positive_norm(macd_hist, 0.45)
        + 0.24 * _positive_norm(di_gap, 9.0)
    )
    stretch_pressure = _positive_norm(
        vgap,
        max(1.6, band_pips * 1.6),
        offset=max(0.5, band_pips * 0.55),
    ) * (0.30 + trend_stack * 0.70)

    continuation_pressure = (
        0.20 * _positive_norm(slope_10, 0.14)
        + 0.12 * _positive_norm(slope_20, 0.10)
        + 0.12 * _positive_norm(macd_hist, 0.45)
        + 0.12 * _positive_norm(adx - 18.0, 10.0)
        + 0.16 * _positive_norm(di_gap, 9.0)
        + 0.28 * stretch_pressure
    )
    reversion_support_score = (
        0.28 * touch_ratio
        + 0.20 * range_support
        + 0.22 * reversal_support
        + 0.18 * vgap_support
        + 0.12 * price_extension
    )
    mtf_countertrend_pressure = _unit_bound(float((mtf_context or {}).get("mtf_countertrend_pressure") or 0.0))
    mtf_aligned_support = _unit_bound(float((mtf_context or {}).get("mtf_aligned_support") or 0.0))
    macro_trend_strength = _unit_bound(float((mtf_context or {}).get("macro_trend_strength") or 0.0))
    continuation_pressure = _unit_bound(
        continuation_pressure
        + 0.24 * mtf_countertrend_pressure * (0.55 + macro_trend_strength * 0.45)
        - 0.08 * mtf_aligned_support
    )
    reversion_support_score = _unit_bound(
        reversion_support_score
        + 0.06 * mtf_aligned_support
        - 0.10 * mtf_countertrend_pressure
    )
    reversion_support_score = _unit_bound(reversion_support_score * 0.9 + adx_reversion * 0.1)
    setup_quality = _unit_bound(0.18 + reversion_support_score * 0.68 - continuation_pressure * 0.72)
    max_pressure = _unit_bound(
        0.32
        + reversion_support_score * 0.28
        + max(0.0, setup_quality - 0.55) * 0.08
        + 0.04 * mtf_aligned_support
        - 0.08 * mtf_countertrend_pressure
    )
    allow = continuation_pressure <= max_pressure and not (
        continuation_pressure >= 0.50 and setup_quality <= 0.40 and trend_stack >= 0.55
    )
    detail = {
        "profile": profile,
        "continuation_pressure": round(continuation_pressure, 3),
        "reversion_support": round(reversion_support_score, 3),
        "setup_quality": round(setup_quality, 3),
        "max_pressure": round(max_pressure, 3),
        "touch_ratio": round(touch_ratio, 3),
        "price_gap_pips": round(price_gap_pips, 3),
        "vgap_pips": round(vgap, 3),
        "di_gap": round(di_gap, 3),
        "trend_stack": round(trend_stack, 3),
        "stretch_pressure": round(stretch_pressure, 3),
        "adx": round(adx, 3),
        "macro_flow_regime": str((mtf_context or {}).get("macro_flow_regime") or "transition"),
        "mtf_alignment": str((mtf_context or {}).get("mtf_alignment") or "neutral"),
        "mtf_countertrend_pressure": round(mtf_countertrend_pressure, 3),
        "mtf_aligned_support": round(mtf_aligned_support, 3),
        "macro_trend_strength": round(macro_trend_strength, 3),
        "m5_flow_regime": str((mtf_context or {}).get("m5_flow_regime") or ""),
        "h1_flow_regime": str((mtf_context or {}).get("h1_flow_regime") or ""),
        "h4_flow_regime": str((mtf_context or {}).get("h4_flow_regime") or ""),
    }
    return allow, detail


def _reversion_long_flow_guard(
    *,
    fac_m1: Dict[str, object],
    price: float,
    atr_pips: float,
    dist_lower_pips: float,
    band_pips: float,
    range_score: float,
    rev_strength: float,
    rsi: float,
    rsi_long_max: float,
    profile: str,
    mtf_context: Optional[Dict[str, object]] = None,
) -> Tuple[bool, Dict[str, float]]:
    try:
        adx = float(fac_m1.get("adx") or 0.0)
    except Exception:
        adx = 0.0
    ma_fast = fac_m1.get("ma10")
    if ma_fast is None:
        ma_fast = fac_m1.get("ema20")
    ma_slow = fac_m1.get("ma20")
    if ma_slow is None:
        ma_slow = fac_m1.get("ema24")
    if ma_slow is None:
        ma_slow = fac_m1.get("vwap")
    try:
        ma_gap_pips = ((float(ma_fast) - float(ma_slow)) / PIP) if ma_fast is not None and ma_slow is not None else 0.0
    except Exception:
        ma_gap_pips = 0.0
    ema20 = fac_m1.get("ema20")
    try:
        price_vs_ema_pips = ((float(ema20) - price) / PIP) if ema20 is not None else 0.0
    except Exception:
        price_vs_ema_pips = 0.0

    touch_ratio = _unit_bound((max(0.0, band_pips) - max(0.0, dist_lower_pips)) / max(0.15, band_pips))
    continuation_pressure = _unit_bound(
        0.34 * _positive_norm(-ma_gap_pips, max(0.8, atr_pips * 0.55))
        + 0.18 * _positive_norm(_minus_di(fac_m1) - _plus_di(fac_m1), 8.0)
        + 0.14 * _positive_norm(adx - 14.0, 10.0)
        + 0.12 * _positive_norm(price_vs_ema_pips, max(0.6, band_pips * 0.55))
        + 0.12 * _positive_norm(-_ema_slope_pips(fac_m1, "ema_slope_10"), 0.14)
        + 0.10 * _positive_norm(-_ema_slope_pips(fac_m1, "ema_slope_20"), 0.10)
    )
    reversion_support_score = _unit_bound(
        0.34 * touch_ratio
        + 0.32 * _unit_bound((float(rsi_long_max) - rsi + 8.0) / 18.0)
        + 0.24 * _unit_bound((rev_strength - 0.20) / 0.70)
        + 0.10 * _unit_bound((range_score - 0.18) / 0.32)
    )
    mtf_countertrend_pressure = _unit_bound(float((mtf_context or {}).get("mtf_countertrend_pressure") or 0.0))
    mtf_aligned_support = _unit_bound(float((mtf_context or {}).get("mtf_aligned_support") or 0.0))
    macro_trend_strength = _unit_bound(float((mtf_context or {}).get("macro_trend_strength") or 0.0))
    continuation_pressure = _unit_bound(
        continuation_pressure
        + 0.24 * mtf_countertrend_pressure * (0.55 + macro_trend_strength * 0.45)
        - 0.08 * mtf_aligned_support
    )
    reversion_support_score = _unit_bound(
        reversion_support_score
        + 0.06 * mtf_aligned_support
        - 0.10 * mtf_countertrend_pressure
    )
    setup_quality = _unit_bound(0.24 + reversion_support_score * 0.62 - continuation_pressure * 0.72)
    max_pressure = _unit_bound(
        0.34
        + reversion_support_score * 0.30
        + max(0.0, setup_quality - 0.58) * 0.10
        + 0.04 * mtf_aligned_support
        - 0.08 * mtf_countertrend_pressure
    )
    strong_reclaim_probe = (
        rev_strength >= 0.78
        and touch_ratio >= 0.42
        and setup_quality >= 0.44
    )
    detail = {
        "profile": profile,
        "continuation_pressure": round(continuation_pressure, 3),
        "reversion_support": round(reversion_support_score, 3),
        "setup_quality": round(setup_quality, 3),
        "max_pressure": round(max_pressure, 3),
        "touch_ratio": round(touch_ratio, 3),
        "ma_gap_pips": round(ma_gap_pips, 3),
        "price_gap_pips": round(price_vs_ema_pips, 3),
        "di_gap": round(_minus_di(fac_m1) - _plus_di(fac_m1), 3),
        "adx": round(adx, 3),
        "strong_reclaim_probe": 1.0 if strong_reclaim_probe else 0.0,
        "macro_flow_regime": str((mtf_context or {}).get("macro_flow_regime") or "transition"),
        "mtf_alignment": str((mtf_context or {}).get("mtf_alignment") or "neutral"),
        "mtf_countertrend_pressure": round(mtf_countertrend_pressure, 3),
        "mtf_aligned_support": round(mtf_aligned_support, 3),
        "macro_trend_strength": round(macro_trend_strength, 3),
        "m5_flow_regime": str((mtf_context or {}).get("m5_flow_regime") or ""),
        "h1_flow_regime": str((mtf_context or {}).get("h1_flow_regime") or ""),
        "h4_flow_regime": str((mtf_context or {}).get("h4_flow_regime") or ""),
    }
    allow = continuation_pressure <= max_pressure or strong_reclaim_probe
    return allow, detail


def _precision_lowvol_setup_pressure(side: str, range_reason: str) -> Dict[str, float]:
    reason_key = str(range_reason or "").strip().lower()
    side_key = str(side or "").strip().lower()
    if (
        not getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_ENABLED", True)
        or reason_key != "volatility_compression"
        or side_key != "short"
        or getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_LOOKBACK_TRADES", 0) <= 0
        or getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_LOOKBACK_MINUTES", 0.0) <= 0.0
    ):
        return {}

    cache_key = (side_key, reason_key)
    now_mono = time.monotonic()
    cached = _PREC_LOWVOL_SETUP_PRESSURE_CACHE.get(cache_key)
    cache_ttl = max(0.0, float(getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_CACHE_TTL_SEC", 8.0)))
    if cached is not None and cache_ttl > 0.0 and (now_mono - cached[0]) <= cache_ttl:
        return dict(cached[1])

    lookback_minutes = max(
        5.0,
        float(getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_LOOKBACK_MINUTES", 20.0)),
    )
    query = """
        SELECT
          close_reason,
          COALESCE(realized_pl, 0.0) AS realized_pl,
          COALESCE((julianday(close_time) - julianday(open_time)) * 86400.0, 0.0) AS hold_sec,
          COALESCE((julianday('now') - julianday(close_time)) * 86400.0, 999999.0) AS age_sec
        FROM trades
        WHERE strategy_tag = 'PrecisionLowVol'
          AND close_time IS NOT NULL
          AND julianday(close_time) >= julianday('now', ?)
          AND lower(
                COALESCE(
                    json_extract(entry_thesis, '$.projection.side'),
                    CASE
                      WHEN units > 0 THEN 'long'
                      WHEN units < 0 THEN 'short'
                      ELSE ''
                    END
                )
              ) = ?
          AND lower(COALESCE(json_extract(entry_thesis, '$.range_reason'), '')) = ?
        ORDER BY julianday(close_time) DESC
        LIMIT ?
    """
    summary: Dict[str, float] = {
        "trades": 0.0,
        "sl_rate": 0.0,
        "fast_sl_rate": 0.0,
        "net_jpy": 0.0,
        "stop_loss_streak": 0.0,
        "fast_stop_loss_streak": 0.0,
        "last_close_age_sec": 999999.0,
        "active": 0.0,
    }
    con: Optional[sqlite3.Connection] = None
    try:
        con = sqlite3.connect("file:logs/trades.db?mode=ro", uri=True, timeout=1.0)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            query,
            (
                f"-{lookback_minutes:.1f} minutes",
                side_key,
                reason_key,
                int(max(1, getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_LOOKBACK_TRADES", 6))),
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
        stop_loss_streak = 0
        fast_stop_loss_streak = 0
        net_jpy = 0.0
        fast_sl_max_hold_sec = max(
            0.0,
            float(getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_FAST_SL_MAX_HOLD_SEC", 35.0)),
        )
        for idx, row in enumerate(rows):
            close_reason = str(row["close_reason"] or "").upper()
            realized_pl = float(row["realized_pl"] or 0.0)
            hold_sec = max(0.0, float(row["hold_sec"] or 0.0))
            net_jpy += realized_pl
            is_stop_loss = close_reason == "STOP_LOSS_ORDER"
            is_fast_stop_loss = is_stop_loss and hold_sec <= fast_sl_max_hold_sec
            if is_stop_loss:
                sl_count += 1
                if is_fast_stop_loss:
                    fast_sl_count += 1
            if idx == 0:
                last_close_age_sec = max(0.0, float(row["age_sec"] or 0.0))
            if stop_loss_streak == idx and is_stop_loss:
                stop_loss_streak += 1
            if fast_stop_loss_streak == idx and is_fast_stop_loss:
                fast_stop_loss_streak += 1

        sl_rate = sl_count / max(1, trades)
        fast_sl_rate = fast_sl_count / max(1, trades)
        active = (
            trades >= max(1, int(getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_MIN_TRADES", 3)))
            and net_jpy <= -max(0.0, float(getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_NET_LOSS_MIN_JPY", 10.0)))
            and stop_loss_streak >= max(
                1, int(getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_STOP_LOSS_STREAK_MIN", 2))
            )
            and fast_stop_loss_streak >= max(
                0, int(getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_FAST_STOP_LOSS_STREAK_MIN", 1))
            )
            and last_close_age_sec <= max(
                0.0, float(getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_ACTIVE_MAX_AGE_SEC", 180.0))
            )
        )
        summary = {
            "trades": float(trades),
            "sl_rate": round(sl_rate, 3),
            "fast_sl_rate": round(fast_sl_rate, 3),
            "net_jpy": round(net_jpy, 3),
            "stop_loss_streak": float(stop_loss_streak),
            "fast_stop_loss_streak": float(fast_stop_loss_streak),
            "last_close_age_sec": round(last_close_age_sec, 1),
            "active": 1.0 if active else 0.0,
        }

    _PREC_LOWVOL_SETUP_PRESSURE_CACHE[cache_key] = (now_mono, dict(summary))
    return summary


def _drought_revert_setup_pressure(side: str, range_reason: str) -> Dict[str, float]:
    reason_key = str(range_reason or "").strip().lower()
    side_key = str(side or "").strip().lower()
    if (
        not getattr(config, "DROUGHT_SETUP_PRESSURE_ENABLED", True)
        or reason_key != "volatility_compression"
        or side_key != "long"
        or getattr(config, "DROUGHT_SETUP_PRESSURE_LOOKBACK_TRADES", 0) <= 0
        or getattr(config, "DROUGHT_SETUP_PRESSURE_LOOKBACK_MINUTES", 0.0) <= 0.0
    ):
        return {}

    cache_key = (side_key, reason_key)
    now_mono = time.monotonic()
    cached = _DROUGHT_SETUP_PRESSURE_CACHE.get(cache_key)
    cache_ttl = max(0.0, float(getattr(config, "DROUGHT_SETUP_PRESSURE_CACHE_TTL_SEC", 15.0)))
    if cached is not None and cache_ttl > 0.0 and (now_mono - cached[0]) <= cache_ttl:
        return dict(cached[1])

    lookback_minutes = max(
        30.0,
        float(getattr(config, "DROUGHT_SETUP_PRESSURE_LOOKBACK_MINUTES", 1440.0)),
    )
    query = """
        SELECT
          close_reason,
          COALESCE(realized_pl, 0.0) AS realized_pl,
          COALESCE((julianday(close_time) - julianday(open_time)) * 86400.0, 0.0) AS hold_sec,
          COALESCE((julianday('now') - julianday(close_time)) * 86400.0, 999999.0) AS age_sec
        FROM trades
        WHERE strategy_tag = 'DroughtRevert'
          AND close_time IS NOT NULL
          AND julianday(close_time) >= julianday('now', ?)
          AND lower(
                COALESCE(
                    json_extract(entry_thesis, '$.projection.side'),
                    CASE
                      WHEN units > 0 THEN 'long'
                      WHEN units < 0 THEN 'short'
                      ELSE ''
                    END
                )
              ) = ?
          AND lower(COALESCE(json_extract(entry_thesis, '$.range_reason'), '')) = ?
        ORDER BY julianday(close_time) DESC
        LIMIT ?
    """
    summary: Dict[str, float] = {
        "trades": 0.0,
        "sl_rate": 0.0,
        "fast_sl_rate": 0.0,
        "net_jpy": 0.0,
        "last_close_age_sec": 999999.0,
        "active": 0.0,
    }
    con: Optional[sqlite3.Connection] = None
    try:
        con = sqlite3.connect("file:logs/trades.db?mode=ro", uri=True, timeout=1.0)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            query,
            (
                f"-{lookback_minutes:.1f} minutes",
                side_key,
                reason_key,
                int(max(1, getattr(config, "DROUGHT_SETUP_PRESSURE_LOOKBACK_TRADES", 8))),
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
        last_close_age_sec = 999999.0
        fast_sl_max_hold_sec = max(
            0.0,
            float(getattr(config, "DROUGHT_SETUP_PRESSURE_FAST_SL_MAX_HOLD_SEC", 35.0)),
        )
        for idx, row in enumerate(rows):
            close_reason = str(row["close_reason"] or "").upper()
            realized_pl = float(row["realized_pl"] or 0.0)
            hold_sec = max(0.0, float(row["hold_sec"] or 0.0))
            net_jpy += realized_pl
            is_stop_loss = close_reason == "STOP_LOSS_ORDER"
            is_fast_stop_loss = is_stop_loss and hold_sec <= fast_sl_max_hold_sec
            if is_stop_loss:
                sl_count += 1
            if is_fast_stop_loss:
                fast_sl_count += 1
            if idx == 0:
                last_close_age_sec = max(0.0, float(row["age_sec"] or 0.0))

        sl_rate = sl_count / max(1, trades)
        fast_sl_rate = fast_sl_count / max(1, trades)
        active = (
            trades >= max(1, int(getattr(config, "DROUGHT_SETUP_PRESSURE_MIN_TRADES", 6)))
            and net_jpy <= -max(0.0, float(getattr(config, "DROUGHT_SETUP_PRESSURE_NET_LOSS_MIN_JPY", 10.0)))
            and sl_rate >= max(0.0, float(getattr(config, "DROUGHT_SETUP_PRESSURE_STOP_LOSS_RATE_MIN", 0.50)))
            and fast_sl_rate
            >= max(0.0, float(getattr(config, "DROUGHT_SETUP_PRESSURE_FAST_STOP_LOSS_RATE_MIN", 0.25)))
            and last_close_age_sec
            <= max(0.0, float(getattr(config, "DROUGHT_SETUP_PRESSURE_ACTIVE_MAX_AGE_SEC", 43200.0)))
        )
        summary = {
            "trades": float(trades),
            "sl_rate": round(sl_rate, 3),
            "fast_sl_rate": round(fast_sl_rate, 3),
            "net_jpy": round(net_jpy, 3),
            "last_close_age_sec": round(last_close_age_sec, 1),
            "active": 1.0 if active else 0.0,
        }

    _DROUGHT_SETUP_PRESSURE_CACHE[cache_key] = (now_mono, dict(summary))
    return summary


def _wick_blend_long_setup_pressure(range_reason: str) -> Dict[str, float]:
    reason_key = str(range_reason or "").strip().lower()
    if (
        not WICK_BLEND_LONG_SETUP_PRESSURE_ENABLED
        or reason_key != "volatility_compression"
        or WICK_BLEND_LONG_SETUP_PRESSURE_LOOKBACK_TRADES <= 0
        or WICK_BLEND_LONG_SETUP_PRESSURE_LOOKBACK_MINUTES <= 0.0
    ):
        return {}

    now_mono = time.monotonic()
    cached = _WICK_BLEND_LONG_SETUP_PRESSURE_CACHE.get(reason_key)
    cache_ttl = max(0.0, WICK_BLEND_LONG_SETUP_PRESSURE_CACHE_TTL_SEC)
    if cached is not None and cache_ttl > 0.0 and (now_mono - cached[0]) <= cache_ttl:
        return dict(cached[1])

    query = """
        SELECT
          close_reason,
          COALESCE(realized_pl, 0.0) AS realized_pl,
          COALESCE((julianday(close_time) - julianday(open_time)) * 86400.0, 0.0) AS hold_sec,
          COALESCE((julianday('now') - julianday(close_time)) * 86400.0, 999999.0) AS age_sec
        FROM trades
        WHERE strategy_tag = 'WickReversalBlend'
          AND close_time IS NOT NULL
          AND julianday(close_time) >= julianday('now', ?)
          AND units > 0
          AND lower(COALESCE(json_extract(entry_thesis, '$.range_reason'), '')) = ?
        ORDER BY julianday(close_time) DESC
        LIMIT ?
    """
    summary: Dict[str, float] = {
        "trades": 0.0,
        "sl_rate": 0.0,
        "fast_sl_rate": 0.0,
        "net_jpy": 0.0,
        "stop_loss_streak": 0.0,
        "fast_stop_loss_streak": 0.0,
        "last_close_age_sec": 999999.0,
        "active": 0.0,
    }
    con: Optional[sqlite3.Connection] = None
    try:
        con = sqlite3.connect("file:logs/trades.db?mode=ro", uri=True, timeout=1.0)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            query,
            (
                f"-{max(30.0, WICK_BLEND_LONG_SETUP_PRESSURE_LOOKBACK_MINUTES):.1f} minutes",
                reason_key,
                int(max(1, WICK_BLEND_LONG_SETUP_PRESSURE_LOOKBACK_TRADES)),
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
        stop_loss_streak = 0
        fast_stop_loss_streak = 0
        net_jpy = 0.0
        last_close_age_sec = 999999.0
        fast_sl_max_hold_sec = max(0.0, WICK_BLEND_LONG_SETUP_PRESSURE_FAST_SL_MAX_HOLD_SEC)
        for idx, row in enumerate(rows):
            close_reason = str(row["close_reason"] or "").upper()
            realized_pl = float(row["realized_pl"] or 0.0)
            hold_sec = max(0.0, float(row["hold_sec"] or 0.0))
            net_jpy += realized_pl
            is_stop_loss = close_reason == "STOP_LOSS_ORDER"
            is_fast_stop_loss = is_stop_loss and hold_sec <= fast_sl_max_hold_sec
            if is_stop_loss:
                sl_count += 1
            if is_fast_stop_loss:
                fast_sl_count += 1
            if idx == 0:
                last_close_age_sec = max(0.0, float(row["age_sec"] or 0.0))
            if stop_loss_streak == idx and is_stop_loss:
                stop_loss_streak += 1
            if fast_stop_loss_streak == idx and is_fast_stop_loss:
                fast_stop_loss_streak += 1

        sl_rate = sl_count / max(1, trades)
        fast_sl_rate = fast_sl_count / max(1, trades)
        active = (
            trades >= max(1, WICK_BLEND_LONG_SETUP_PRESSURE_MIN_TRADES)
            and net_jpy <= -max(0.0, WICK_BLEND_LONG_SETUP_PRESSURE_NET_LOSS_MIN_JPY)
            and stop_loss_streak >= max(1, WICK_BLEND_LONG_SETUP_PRESSURE_STOP_LOSS_STREAK_MIN)
            and fast_stop_loss_streak >= max(0, WICK_BLEND_LONG_SETUP_PRESSURE_FAST_STOP_LOSS_STREAK_MIN)
            and last_close_age_sec <= max(0.0, WICK_BLEND_LONG_SETUP_PRESSURE_ACTIVE_MAX_AGE_SEC)
        )
        summary = {
            "trades": float(trades),
            "sl_rate": round(sl_rate, 3),
            "fast_sl_rate": round(fast_sl_rate, 3),
            "net_jpy": round(net_jpy, 3),
            "stop_loss_streak": float(stop_loss_streak),
            "fast_stop_loss_streak": float(fast_stop_loss_streak),
            "last_close_age_sec": round(last_close_age_sec, 1),
            "active": 1.0 if active else 0.0,
        }

    _WICK_BLEND_LONG_SETUP_PRESSURE_CACHE[reason_key] = (now_mono, dict(summary))
    return summary


def _wick_blend_long_pressure_blocked(
    *,
    range_reason: str,
    side: str,
    setup_pressure: Dict[str, float],
    bbw: float,
    range_score: float,
    rsi: float,
    wick_quality: float,
    projection_score: float,
) -> bool:
    if (
        not WICK_BLEND_LONG_SETUP_PRESSURE_ENABLED
        or str(side or "").strip().lower() != "long"
        or str(range_reason or "").strip().lower() != "volatility_compression"
        or float(setup_pressure.get("active") or 0.0) <= 0.0
    ):
        return False
    return (
        bbw <= WICK_BLEND_LONG_SETUP_PRESSURE_BBW_MAX
        and range_score >= WICK_BLEND_LONG_SETUP_PRESSURE_RANGE_SCORE_MIN
        and rsi <= WICK_BLEND_LONG_SETUP_PRESSURE_RSI_MAX
        and wick_quality <= WICK_BLEND_LONG_SETUP_PRESSURE_QUALITY_MAX
        and projection_score <= WICK_BLEND_LONG_SETUP_PRESSURE_PROJECTION_SCORE_MAX
    )


def _wick_blend_short_countertrend_blocked(
    *,
    range_reason: str,
    side: str,
    projection_score: float,
    wick_quality: float,
    rsi: float,
    adx: float,
    macd_hist_pips: float,
) -> bool:
    if (
        not WICK_BLEND_SHORT_COUNTERTREND_GUARD_ENABLED
        or str(side or "").strip().lower() != "short"
        or str(range_reason or "").strip().lower() != "volatility_compression"
    ):
        return False
    return (
        projection_score >= WICK_BLEND_SHORT_COUNTERTREND_PROJECTION_SCORE_MIN
        and wick_quality <= WICK_BLEND_SHORT_COUNTERTREND_QUALITY_MAX
        and rsi <= WICK_BLEND_SHORT_COUNTERTREND_RSI_MAX
        and adx <= WICK_BLEND_SHORT_COUNTERTREND_ADX_MAX
        and macd_hist_pips >= WICK_BLEND_SHORT_COUNTERTREND_MACD_HIST_PIPS_MIN
    )


def _attach_flow_guard_context(signal: Dict[str, object], flow_guard: Optional[Dict[str, float]]) -> None:
    if not isinstance(flow_guard, dict):
        return
    signal["flow_guard"] = flow_guard
    signal["continuation_pressure"] = flow_guard.get("continuation_pressure")
    signal["reversion_support"] = flow_guard.get("reversion_support")
    signal["setup_quality"] = flow_guard.get("setup_quality")
    for key in (
        "macro_flow_regime",
        "mtf_alignment",
        "mtf_countertrend_pressure",
        "mtf_aligned_support",
        "macro_trend_strength",
        "m5_flow_regime",
        "h1_flow_regime",
        "h4_flow_regime",
    ):
        value = flow_guard.get(key)
        if value not in {None, ""}:
            signal[key] = value
    continuation_pressure = float(flow_guard.get("continuation_pressure") or 0.0)
    # Keep the worker-local headwind label separate so strategy_entry can inject the
    # richer live setup context (`range_compression` / `transition` / `trend_*`) without
    # being flattened into a binary range/headwind bucket.
    signal["flow_headwind_regime"] = (
        "continuation_headwind" if continuation_pressure >= 0.6 else "range_fade"
    )


def _signal_spread_revert(
    fac_m1: Dict[str, object],
    range_ctx,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    levels = bb_levels(fac_m1)
    if not levels:
        return None
    upper, mid, lower, _, span_pips = levels

    range_score = float(range_ctx.score or 0.0) if range_ctx else 0.0
    range_ok = bool(range_ctx and (range_ctx.active or range_score >= SPREAD_REV_RANGE_ONLY_SCORE))
    if not range_ok:
        return None

    ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=SPREAD_REV_SPREAD_P25)
    if not ok_spread:
        return None

    adx = _adx(fac_m1)
    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    rsi = _rsi(fac_m1)
    if adx > SPREAD_REV_ADX_MAX or bbw > SPREAD_REV_BBW_MAX:
        return None
    if atr < SPREAD_REV_ATR_MIN or atr > SPREAD_REV_ATR_MAX:
        return None

    mids, _ = tick_snapshot(6.0, limit=60)
    rev_ok, rev_dir, rev_strength = tick_reversal(mids, min_ticks=SPREAD_REV_TICK_MIN)
    if not rev_ok:
        return None

    side = None
    dist_lower = (price - lower) / PIP
    dist_upper = (upper - price) / PIP
    if dist_lower <= max(SPREAD_REV_BB_TOUCH_PIPS, span_pips * 0.18) and rsi <= SPREAD_REV_RSI_LONG_MAX:
        side = "long"
    elif dist_upper <= max(SPREAD_REV_BB_TOUCH_PIPS, span_pips * 0.18) and rsi >= SPREAD_REV_RSI_SHORT_MIN:
        side = "short"
    if not side or rev_dir != side:
        return None

    sl = max(1.2, min(1.8, atr * 0.7))
    tp = max(1.2, min(2.0, atr * 0.9))
    conf = 58
    conf += int(min(12, abs(rsi - 50.0) * 0.6))
    conf += int(min(8, rev_strength * 4.0))
    conf -= int(min(8, max(0.0, adx - 18.0) * 0.4))

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(40, min(92, conf))),
        "tag": tag,
        "reason": "spread_range_revert",
        "range_score": round(range_score, 3),
    }


def _signal_drought_revert(
    fac_m1: Dict[str, object],
    range_ctx,
    *,
    tag: str,
    fac_m5: Optional[Dict[str, object]] = None,
    fac_h1: Optional[Dict[str, object]] = None,
    fac_h4: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    levels = bb_levels(fac_m1)
    if not levels:
        return None
    upper, _, lower, _, span_pips = levels

    range_score = float(range_ctx.score or 0.0) if range_ctx else 0.0
    range_ok = bool(range_ctx and (range_ctx.active or range_score >= config.DROUGHT_RANGE_SCORE))
    if not range_ok:
        return None

    ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=config.DROUGHT_SPREAD_P25)
    if not ok_spread:
        return None

    adx = _adx(fac_m1)
    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    rsi = _rsi(fac_m1)
    if adx > config.DROUGHT_ADX_MAX or bbw > config.DROUGHT_BBW_MAX:
        return None
    if atr < config.DROUGHT_ATR_MIN or atr > config.DROUGHT_ATR_MAX:
        return None

    mids, _ = tick_snapshot(6.0, limit=60)
    rev_ok, rev_dir, rev_strength = tick_reversal(mids, min_ticks=max(4, SPREAD_REV_TICK_MIN - 2))
    if not rev_ok:
        return None

    side = None
    band = max(config.DROUGHT_BB_TOUCH_PIPS, span_pips * 0.2)
    dist_lower = (price - lower) / PIP
    dist_upper = (upper - price) / PIP
    if dist_lower <= band and rsi <= config.DROUGHT_RSI_LONG_MAX:
        side = "long"
    elif dist_upper <= band and rsi >= config.DROUGHT_RSI_SHORT_MIN:
        side = "short"
    if not side or rev_dir != side:
        return None

    dist = dist_lower if side == "long" else dist_upper
    touch_ratio = max(0.0, (band - dist) / max(0.2, band))
    mtf_context = _reversion_mtf_context(
        side=side,
        fac_m5=fac_m5,
        fac_h1=fac_h1,
        fac_h4=fac_h4,
    )
    flow_guard = None
    if side == "short":
        short_ok, flow_guard = _reversion_short_flow_guard(
            fac_m1=fac_m1,
            price=price,
            dist_upper_pips=dist_upper,
            band_pips=band,
            range_score=range_score,
            rev_strength=rev_strength,
            profile=tag,
            mtf_context=mtf_context,
        )
        if not short_ok:
            return None
    else:
        long_ok, flow_guard = _reversion_long_flow_guard(
            fac_m1=fac_m1,
            price=price,
            atr_pips=atr,
            dist_lower_pips=dist_lower,
            band_pips=band,
            range_score=range_score,
            rev_strength=rev_strength,
            rsi=rsi,
            rsi_long_max=config.DROUGHT_RSI_LONG_MAX,
            profile=tag,
            mtf_context=mtf_context,
        )
        if not long_ok:
            return None
        trend_pressure = float(flow_guard.get("continuation_pressure") or 0.0)
        reversion_support_score = float(flow_guard.get("reversion_support") or 0.0)
        setup_quality = float(flow_guard.get("setup_quality") or 0.0)
        max_pressure = float(flow_guard.get("max_pressure") or 0.0)
        ma_gap_pips = float(flow_guard.get("ma_gap_pips") or 0.0)
        price_vs_ema_pips = float(flow_guard.get("price_gap_pips") or 0.0)
        strong_reclaim_probe = bool(flow_guard.get("strong_reclaim_probe"))
        # Current live loser cluster was flat-gap long reclaim with a deep mean stretch.
        # Keep strong directional reclaim lanes, but suppress oversold flat-gap longs that
        # are still far below the fast mean while the broader VWAP gap remains stretched.
        flat_gap_lane = abs(ma_gap_pips) <= max(0.65, atr * 0.24)
        mean_stretch_lane = price_vs_ema_pips >= max(2.8, atr * 0.95)
        vwap_stretch_lane = _vwap_gap_pips(fac_m1) >= max(18.0, atr * 7.0)
        oversold_flat_gap_long = (
            flat_gap_lane
            and mean_stretch_lane
            and vwap_stretch_lane
            and rsi <= min(float(config.DROUGHT_RSI_LONG_MAX) - 1.0, 45.0)
        )
        exceptional_reclaim_probe = (
            rev_strength >= 0.92
            and touch_ratio >= 1.60
            and setup_quality >= 0.62
        )
        if oversold_flat_gap_long and not exceptional_reclaim_probe:
            return None
        if trend_pressure > max_pressure and not strong_reclaim_probe:
            return None

    proj_allow, proj_size_mult, proj_detail = projection_decision(side, mode="range")
    if not proj_allow:
        return None
    projection_score = None
    if isinstance(proj_detail, dict):
        try:
            raw_projection_score = proj_detail.get("score")
            if raw_projection_score is not None:
                projection_score = float(raw_projection_score)
        except Exception:
            projection_score = None
    if side == "long" and flow_guard is not None:
        macro_flow_regime = str(flow_guard.get("macro_flow_regime") or "").strip().lower()
        di_gap = abs(float(flow_guard.get("di_gap") or 0.0))
        weak_trend_long_probe = (
            bool(getattr(config, "DROUGHT_WEAK_TREND_LONG_PROBE_GUARD_ENABLED", True))
            and str(getattr(range_ctx, "reason", "") or "").strip().lower() == "volatility_compression"
            and macro_flow_regime == "trend_long"
            and projection_score is not None
            and projection_score
            <= float(getattr(config, "DROUGHT_WEAK_TREND_LONG_PROBE_PROJECTION_SCORE_MAX", -0.18))
            and setup_quality
            <= float(getattr(config, "DROUGHT_WEAK_TREND_LONG_PROBE_SETUP_QUALITY_MAX", 0.40))
            and trend_pressure
            >= float(
                getattr(config, "DROUGHT_WEAK_TREND_LONG_PROBE_CONTINUATION_PRESSURE_MIN", 0.40)
            )
            and price_vs_ema_pips
            >= float(getattr(config, "DROUGHT_WEAK_TREND_LONG_PROBE_PRICE_GAP_MIN_PIPS", 5.0))
            and abs(ma_gap_pips)
            >= float(getattr(config, "DROUGHT_WEAK_TREND_LONG_PROBE_MA_GAP_ABS_MIN_PIPS", 0.65))
            and di_gap
            <= float(getattr(config, "DROUGHT_WEAK_TREND_LONG_PROBE_DI_GAP_MAX", 10.0))
            and not strong_reclaim_probe
        )
        if weak_trend_long_probe:
            return None
        flat_gap_soft_trend_long_probe = (
            bool(getattr(config, "DROUGHT_FLAT_GAP_SOFT_TREND_LONG_GUARD_ENABLED", True))
            and str(getattr(range_ctx, "reason", "") or "").strip().lower() == "volatility_compression"
            and macro_flow_regime == "trend_long"
            and projection_score is not None
            and projection_score
            <= float(getattr(config, "DROUGHT_FLAT_GAP_SOFT_TREND_LONG_PROJECTION_SCORE_MAX", 0.10))
            and float(rsi)
            >= float(getattr(config, "DROUGHT_FLAT_GAP_SOFT_TREND_LONG_RSI_MIN", 42.0))
            and float(rsi)
            <= float(getattr(config, "DROUGHT_FLAT_GAP_SOFT_TREND_LONG_RSI_MAX", 46.0))
            and float(adx)
            <= float(getattr(config, "DROUGHT_FLAT_GAP_SOFT_TREND_LONG_ADX_MAX", 12.5))
            and abs(ma_gap_pips)
            <= float(getattr(config, "DROUGHT_FLAT_GAP_SOFT_TREND_LONG_MA_GAP_ABS_MAX_PIPS", 0.40))
            and setup_quality
            < float(getattr(config, "DROUGHT_FLAT_GAP_SOFT_TREND_LONG_SETUP_QUALITY_MAX", 0.52))
            and reversion_support_score
            < float(getattr(config, "DROUGHT_FLAT_GAP_SOFT_TREND_LONG_REVERSION_SUPPORT_MAX", 0.60))
            and trend_pressure
            <= float(
                getattr(config, "DROUGHT_FLAT_GAP_SOFT_TREND_LONG_CONTINUATION_PRESSURE_MAX", 0.18)
            )
            and not strong_reclaim_probe
        )
        if flat_gap_soft_trend_long_probe:
            return None

    setup_pressure = {}
    if side == "long":
        setup_pressure = _drought_revert_setup_pressure(
            side=side,
            range_reason=getattr(range_ctx, "reason", None),
        )
        if float(setup_pressure.get("active") or 0.0) > 0.0:
            continuation_pressure = float(flow_guard.get("continuation_pressure") or 0.0) if flow_guard else 0.0
            reversion_support = float(flow_guard.get("reversion_support") or 0.0) if flow_guard else 0.0
            setup_quality = float(flow_guard.get("setup_quality") or 0.0) if flow_guard else 0.0
            strong_reentry_probe = (
                touch_ratio
                >= max(0.0, float(getattr(config, "DROUGHT_SETUP_PRESSURE_ALLOW_TOUCH_RATIO_MIN", 0.50)))
                and rev_strength
                >= max(
                    float(getattr(config, "DROUGHT_SETUP_PRESSURE_ALLOW_REV_STRENGTH_MIN", 0.82)),
                    0.0,
                )
                and (
                    setup_quality
                    >= max(
                        0.0,
                        float(getattr(config, "DROUGHT_SETUP_PRESSURE_ALLOW_SETUP_QUALITY_MIN", 0.44)),
                    )
                    or reversion_support
                    >= max(
                        0.0,
                        float(getattr(config, "DROUGHT_SETUP_PRESSURE_ALLOW_REVERSION_SUPPORT_MIN", 0.70)),
                    )
                    or (
                        projection_score is not None
                        and projection_score
                        >= float(getattr(config, "DROUGHT_SETUP_PRESSURE_ALLOW_PROJECTION_SCORE_MIN", 0.10))
                    )
                )
            )
            weak_reentry_lane = (
                projection_score is not None
                and projection_score
                <= float(getattr(config, "DROUGHT_SETUP_PRESSURE_BLOCK_PROJECTION_SCORE_MAX", 0.08))
                and (
                    setup_quality
                    < float(getattr(config, "DROUGHT_SETUP_PRESSURE_BLOCK_SETUP_QUALITY_MAX", 0.40))
                    or reversion_support
                    < float(getattr(config, "DROUGHT_SETUP_PRESSURE_BLOCK_REVERSION_SUPPORT_MAX", 0.60))
                    or continuation_pressure
                    >= float(
                        getattr(config, "DROUGHT_SETUP_PRESSURE_BLOCK_CONTINUATION_PRESSURE_MIN", 0.33)
                    )
                )
            )
            if weak_reentry_lane and not strong_reentry_probe:
                return None

    # local-v2 replay/live showed several drought-revert longs stopping inside ~10s
    # and later mean-reverting into the prior TP zone; widen the protective band a bit
    # and keep TP reachable in the current 1.8-2.2 pip ATR regime.
    sl = max(1.8, min(2.5, atr * 1.10))
    tp = max(
        max(2.0, sl * 1.08),
        min(2.5, atr * (1.00 + min(0.16, rev_strength * 0.10) + (0.06 if side == "long" else 0.02))),
    )
    conf = 54
    conf += int(min(10, abs(rsi - 50.0) * 0.5))
    conf += int(min(8, rev_strength * 4.0))
    conf += int(min(5, touch_ratio * 6.0))
    conf -= int(min(6, max(0.0, adx - 20.0) * 0.4))
    if flow_guard is not None:
        conf -= int(min(7.0, max(0.0, 0.64 - float(flow_guard["setup_quality"])) * 18.0))
    size_mult = 0.96 + min(0.08, touch_ratio * 0.12) + min(0.06, max(0.0, rev_strength - 0.55) * 0.16)
    size_mult = max(0.82, min(1.18, size_mult * float(proj_size_mult)))
    if flow_guard is not None:
        size_mult = max(
            0.82,
            min(1.10, min(size_mult, float(proj_size_mult) * (0.88 + float(flow_guard["setup_quality"]) * 0.18))),
        )
        if side == "long" and rev_strength >= 0.82 and touch_ratio >= 0.42:
            size_mult = max(size_mult, 0.96)

    signal = {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(40, min(90, conf))),
        "tag": tag,
        "reason": "drought_revert",
        "range_score": round(range_score, 3),
        "size_mult": round(size_mult, 3),
        "projection": proj_detail,
    }
    if setup_pressure:
        signal["setup_pressure"] = setup_pressure
    _attach_flow_guard_context(signal, flow_guard)
    return signal


def _signal_precision_lowvol(
    fac_m1: Dict[str, object],
    range_ctx,
    *,
    tag: str,
    fac_m5: Optional[Dict[str, object]] = None,
    fac_h1: Optional[Dict[str, object]] = None,
    fac_h4: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    levels = bb_levels(fac_m1)
    if not levels:
        return None
    upper, _, lower, _, span_pips = levels

    adx = _adx(fac_m1)
    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    rsi = _rsi(fac_m1)
    stoch = _stoch_rsi(fac_m1)
    vgap = _vwap_gap_pips(fac_m1)

    range_score = float(range_ctx.score or 0.0) if range_ctx else 0.0
    range_reason = str(getattr(range_ctx, "reason", "") or "").strip().lower() if range_ctx else ""
    range_ok = bool(range_ctx and (range_ctx.active or range_score >= config.PREC_LOWVOL_RANGE_SCORE))
    if not range_ok:
        range_ok = adx <= config.PREC_LOWVOL_ADX_MAX and bbw <= config.PREC_LOWVOL_BBW_MAX
    if not range_ok:
        return None

    ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=config.PREC_LOWVOL_SPREAD_P25)
    if not ok_spread:
        return None

    if adx > config.PREC_LOWVOL_ADX_MAX or bbw > config.PREC_LOWVOL_BBW_MAX:
        return None
    if atr < config.PREC_LOWVOL_ATR_MIN or atr > config.PREC_LOWVOL_ATR_MAX:
        return None

    mids, _ = tick_snapshot(6.0, limit=70)
    rev_ok, rev_dir, rev_strength = tick_reversal(mids, min_ticks=4)
    if not rev_ok or rev_strength < config.PREC_LOWVOL_REV_MIN_STRENGTH:
        return None

    band = max(config.PREC_LOWVOL_BB_TOUCH_PIPS, span_pips * 0.28)
    dist_lower = (price - lower) / PIP
    dist_upper = (upper - price) / PIP

    side = None
    long_osc_ok = rsi <= config.PREC_LOWVOL_RSI_LONG_MAX or stoch <= config.PREC_LOWVOL_STOCH_LONG_MAX
    short_osc_ok = rsi >= config.PREC_LOWVOL_RSI_SHORT_MIN or stoch >= config.PREC_LOWVOL_STOCH_SHORT_MIN
    if dist_lower <= band and long_osc_ok and rev_dir == "long":
        side = "long"
    elif dist_upper <= band and short_osc_ok and rev_dir == "short":
        side = "short"
    if not side:
        return None

    dist = dist_lower if side == "long" else dist_upper
    touch_ratio = max(0.0, (band - dist) / max(0.2, band))
    mtf_context = _reversion_mtf_context(
        side=side,
        fac_m5=fac_m5,
        fac_h1=fac_h1,
        fac_h4=fac_h4,
    )
    flow_guard = None
    if side == "short":
        short_ok, flow_guard = _reversion_short_flow_guard(
            fac_m1=fac_m1,
            price=price,
            dist_upper_pips=dist_upper,
            band_pips=band,
            range_score=range_score,
            rev_strength=rev_strength,
            profile=tag,
            mtf_context=mtf_context,
        )
        if not short_ok:
            return None
    else:
        long_ok, flow_guard = _reversion_long_flow_guard(
            fac_m1=fac_m1,
            price=price,
            atr_pips=atr,
            dist_lower_pips=dist_lower,
            band_pips=band,
            range_score=range_score,
            rev_strength=rev_strength,
            rsi=rsi,
            rsi_long_max=config.PREC_LOWVOL_RSI_LONG_MAX,
            profile=tag,
            mtf_context=mtf_context,
        )
        if not long_ok:
            return None

    vgap_block = config.PREC_LOWVOL_VWAP_GAP_BLOCK
    if vgap_block > 0.0:
        if side == "long" and vgap > vgap_block:
            return None
        if side == "short" and vgap < -vgap_block:
            return None

    vgap_bias_min = config.PREC_LOWVOL_VWAP_GAP_MIN
    vgap_bias_ok = (vgap <= -vgap_bias_min and side == "long") or (vgap >= vgap_bias_min and side == "short")
    if vgap_bias_ok and flow_guard is not None:
        continuation_pressure = float(flow_guard.get("continuation_pressure") or 0.0)
        max_pressure = float(flow_guard.get("max_pressure") or 0.0)
        setup_quality = float(flow_guard.get("setup_quality") or 0.0)
        if continuation_pressure + 0.05 > max_pressure or setup_quality < 0.66:
            vgap_bias_ok = False

    proj_allow, size_mult, proj_detail = projection_decision(side, mode="range")
    if not proj_allow:
        return None
    proj_size_mult = float(size_mult)
    projection_score = None
    hostile_projection_lane = False
    ma_fast = fac_m1.get("ma10")
    if ma_fast is None:
        ma_fast = fac_m1.get("ema20")
    ma_slow = fac_m1.get("ma20")
    if ma_slow is None:
        ma_slow = fac_m1.get("ema24")
    if ma_slow is None:
        ma_slow = fac_m1.get("ema20")
    try:
        ma_gap_pips = ((float(ma_fast) - float(ma_slow)) / PIP) if ma_fast is not None and ma_slow is not None else None
    except Exception:
        ma_gap_pips = None
    ma_gap_atr_ratio = abs(ma_gap_pips) / max(1.0, atr) if ma_gap_pips is not None else None
    short_up_lean = bool(
        side == "short"
        and ma_gap_pips is not None
        and ma_gap_pips > 0.0
        and ma_gap_atr_ratio is not None
        and 0.30 <= ma_gap_atr_ratio < 0.90
    )
    short_up_flat = bool(
        side == "short"
        and ma_gap_pips is not None
        and ma_gap_pips > 0.0
        and ma_gap_atr_ratio is not None
        and ma_gap_atr_ratio < float(
            getattr(config, "PREC_LOWVOL_UP_FLAT_SHALLOW_SHORT_GAP_ATR_RATIO_MAX", 0.30)
        )
    )
    long_up_flat = bool(
        side == "long"
        and ma_gap_pips is not None
        and ma_gap_pips > 0.0
        and ma_gap_atr_ratio is not None
        and ma_gap_atr_ratio
        < float(getattr(config, "PREC_LOWVOL_UP_FLAT_LONG_GAP_ATR_RATIO_MAX", 0.30))
    )
    short_down_flat = bool(
        side == "short"
        and ma_gap_pips is not None
        and ma_gap_pips < 0.0
        and ma_gap_atr_ratio is not None
        and ma_gap_atr_ratio < 0.35
    )
    if isinstance(proj_detail, dict):
        try:
            raw_projection_score = proj_detail.get("score")
            if raw_projection_score is not None:
                projection_score = float(raw_projection_score)
        except Exception:
            projection_score = None
    if side == "short" and flow_guard is not None and projection_score is not None:
        gap_atr_ratio = abs(vgap) / max(1.0, atr)
        setup_quality = float(flow_guard.get("setup_quality") or 0.0)
        hostile_projection_lane = (
            projection_score <= -0.10
            and gap_atr_ratio >= 2.5
            and setup_quality < 0.40
            and rsi < max(config.PREC_LOWVOL_RSI_SHORT_MIN + 10.0, 60.0)
        )

    weak_down_flat_lane = False
    if side == "short" and short_down_flat:
        setup_quality = float(flow_guard.get("setup_quality") or 0.0) if flow_guard is not None else 0.0
        weak_down_flat_lane = (
            rev_strength < 0.78
            and touch_ratio < 0.62
            and setup_quality < 0.58
            and (projection_score is None or projection_score < 0.18)
        )
    if weak_down_flat_lane:
        return None
    down_flat_low_score_short_lane = False
    if (
        side == "short"
        and short_down_flat
        and range_reason == "volatility_compression"
        and flow_guard is not None
        and projection_score is not None
        and bool(getattr(config, "PREC_LOWVOL_DOWN_FLAT_LOW_SCORE_SHORT_GUARD_ENABLED", True))
    ):
        setup_quality = float(flow_guard.get("setup_quality") or 0.0)
        continuation_pressure = float(flow_guard.get("continuation_pressure") or 0.0)
        down_flat_low_score_short_lane = (
            range_score
            <= float(getattr(config, "PREC_LOWVOL_DOWN_FLAT_LOW_SCORE_SHORT_RANGE_SCORE_MAX", 0.44))
            and continuation_pressure
            >= float(
                getattr(
                    config,
                    "PREC_LOWVOL_DOWN_FLAT_LOW_SCORE_SHORT_CONTINUATION_PRESSURE_MIN",
                    0.24,
                )
            )
            and rsi >= float(getattr(config, "PREC_LOWVOL_DOWN_FLAT_LOW_SCORE_SHORT_RSI_MIN", 54.0))
            and projection_score
            <= float(
                getattr(
                    config,
                    "PREC_LOWVOL_DOWN_FLAT_LOW_SCORE_SHORT_PROJECTION_SCORE_MAX",
                    0.30,
                )
            )
            and setup_quality
            < float(
                getattr(
                    config,
                    "PREC_LOWVOL_DOWN_FLAT_LOW_SCORE_SHORT_SETUP_QUALITY_MAX",
                    0.40,
                )
            )
        )
    if down_flat_low_score_short_lane:
        return None
    weak_overbought_short_lane = False
    if (
        side == "short"
        and range_reason == "volatility_compression"
        and flow_guard is not None
        and projection_score is not None
        and bool(getattr(config, "PREC_LOWVOL_WEAK_SHORT_GUARD_ENABLED", True))
    ):
        setup_quality = float(flow_guard.get("setup_quality") or 0.0)
        weak_overbought_short_lane = (
            rsi >= float(getattr(config, "PREC_LOWVOL_WEAK_SHORT_RSI_MIN", 60.0))
            and projection_score
            <= float(getattr(config, "PREC_LOWVOL_WEAK_SHORT_PROJECTION_SCORE_MAX", 0.0))
            and setup_quality
            < float(getattr(config, "PREC_LOWVOL_WEAK_SHORT_SETUP_QUALITY_MAX", 0.46))
        )
    marginal_short_lane = False
    if (
        side == "short"
        and range_reason == "volatility_compression"
        and flow_guard is not None
        and projection_score is not None
        and bool(getattr(config, "PREC_LOWVOL_MARGINAL_SHORT_GUARD_ENABLED", True))
    ):
        setup_quality = float(flow_guard.get("setup_quality") or 0.0)
        continuation_pressure = float(flow_guard.get("continuation_pressure") or 0.0)
        marginal_short_lane = (
            continuation_pressure
            >= float(getattr(config, "PREC_LOWVOL_MARGINAL_SHORT_CONTINUATION_PRESSURE_MIN", 0.33))
            and rsi >= float(getattr(config, "PREC_LOWVOL_MARGINAL_SHORT_RSI_MIN", 59.0))
            and projection_score
            <= float(getattr(config, "PREC_LOWVOL_MARGINAL_SHORT_PROJECTION_SCORE_MAX", 0.08))
            and setup_quality
            < float(getattr(config, "PREC_LOWVOL_MARGINAL_SHORT_SETUP_QUALITY_MAX", 0.44))
        )
    headwind_short_lane = False
    if (
        side == "short"
        and range_reason == "volatility_compression"
        and flow_guard is not None
        and projection_score is not None
        and bool(getattr(config, "PREC_LOWVOL_HEADWIND_SHORT_GUARD_ENABLED", True))
    ):
        setup_quality = float(flow_guard.get("setup_quality") or 0.0)
        continuation_pressure = float(flow_guard.get("continuation_pressure") or 0.0)
        headwind_short_lane = (
            continuation_pressure
            >= float(getattr(config, "PREC_LOWVOL_HEADWIND_SHORT_CONTINUATION_PRESSURE_MIN", 0.33))
            and rsi >= float(getattr(config, "PREC_LOWVOL_HEADWIND_SHORT_RSI_MIN", 58.0))
            and projection_score
            <= float(getattr(config, "PREC_LOWVOL_HEADWIND_SHORT_PROJECTION_SCORE_MAX", 0.05))
            and setup_quality
            < float(getattr(config, "PREC_LOWVOL_HEADWIND_SHORT_SETUP_QUALITY_MAX", 0.48))
        )
    up_lean_countertrend_short_lane = False
    if (
        side == "short"
        and short_up_lean
        and flow_guard is not None
        and projection_score is not None
        and bool(getattr(config, "PREC_LOWVOL_UP_LEAN_COUNTERTREND_SHORT_GUARD_ENABLED", True))
    ):
        setup_quality = float(flow_guard.get("setup_quality") or 0.0)
        reversion_support = float(flow_guard.get("reversion_support") or 0.0)
        continuation_pressure = float(flow_guard.get("continuation_pressure") or 0.0)
        macro_flow_regime = str(flow_guard.get("macro_flow_regime") or "").strip().lower()
        mtf_alignment = str(flow_guard.get("mtf_alignment") or "").strip().lower()
        up_lean_countertrend_short_lane = (
            macro_flow_regime == "trend_long"
            and mtf_alignment == "countertrend"
            and continuation_pressure
            >= float(
                getattr(
                    config,
                    "PREC_LOWVOL_UP_LEAN_COUNTERTREND_SHORT_CONTINUATION_PRESSURE_MIN",
                    0.22,
                )
            )
            and rsi
            <= float(getattr(config, "PREC_LOWVOL_UP_LEAN_COUNTERTREND_SHORT_RSI_MAX", 55.0))
            and projection_score
            <= float(
                getattr(
                    config,
                    "PREC_LOWVOL_UP_LEAN_COUNTERTREND_SHORT_PROJECTION_SCORE_MAX",
                    0.30,
                )
            )
            and setup_quality
            < float(
                getattr(
                    config,
                    "PREC_LOWVOL_UP_LEAN_COUNTERTREND_SHORT_SETUP_QUALITY_MAX",
                    0.30,
                )
            )
            and reversion_support
            < float(
                getattr(
                    config,
                    "PREC_LOWVOL_UP_LEAN_COUNTERTREND_SHORT_REVERSION_SUPPORT_MAX",
                    0.58,
                )
            )
        )
    if weak_overbought_short_lane:
        return None
    if marginal_short_lane:
        return None
    if headwind_short_lane:
        return None
    if up_lean_countertrend_short_lane:
        return None
    weak_oversold_long_lane = False
    if (
        side == "long"
        and range_reason == "volatility_compression"
        and flow_guard is not None
        and projection_score is not None
        and bool(getattr(config, "PREC_LOWVOL_WEAK_LONG_GUARD_ENABLED", True))
    ):
        setup_quality = float(flow_guard.get("setup_quality") or 0.0)
        continuation_pressure = float(flow_guard.get("continuation_pressure") or 0.0)
        strong_reclaim_probe = (
            rev_strength
            >= float(getattr(config, "PREC_LOWVOL_WEAK_LONG_STRONG_RECLAIM_REV_STRENGTH_MIN", 0.82))
            and touch_ratio
            >= float(getattr(config, "PREC_LOWVOL_WEAK_LONG_STRONG_RECLAIM_TOUCH_RATIO_MIN", 0.46))
            and setup_quality
            >= float(getattr(config, "PREC_LOWVOL_WEAK_LONG_STRONG_RECLAIM_SETUP_QUALITY_MIN", 0.52))
        )
        weak_oversold_long_lane = (
            rsi <= float(getattr(config, "PREC_LOWVOL_WEAK_LONG_RSI_MAX", 35.0))
            and projection_score
            <= float(getattr(config, "PREC_LOWVOL_WEAK_LONG_PROJECTION_SCORE_MAX", -0.05))
            and setup_quality
            < float(getattr(config, "PREC_LOWVOL_WEAK_LONG_SETUP_QUALITY_MAX", 0.46))
            and continuation_pressure
            >= float(getattr(config, "PREC_LOWVOL_WEAK_LONG_CONTINUATION_PRESSURE_MIN", 0.28))
            and not strong_reclaim_probe
        )
    if weak_oversold_long_lane:
        return None
    up_flat_shallow_long_lane = False
    if (
        side == "long"
        and long_up_flat
        and range_reason == "volatility_compression"
        and flow_guard is not None
        and projection_score is not None
        and bool(getattr(config, "PREC_LOWVOL_UP_FLAT_SHALLOW_LONG_GUARD_ENABLED", True))
    ):
        setup_quality = float(flow_guard.get("setup_quality") or 0.0)
        reversion_support = float(flow_guard.get("reversion_support") or 0.0)
        continuation_pressure = float(flow_guard.get("continuation_pressure") or 0.0)
        macro_flow_regime = str(flow_guard.get("macro_flow_regime") or "").strip().lower()
        strong_reclaim_probe = (
            rev_strength
            >= float(getattr(config, "PREC_LOWVOL_WEAK_LONG_STRONG_RECLAIM_REV_STRENGTH_MIN", 0.82))
            and touch_ratio
            >= float(getattr(config, "PREC_LOWVOL_WEAK_LONG_STRONG_RECLAIM_TOUCH_RATIO_MIN", 0.46))
            and setup_quality
            >= float(getattr(config, "PREC_LOWVOL_WEAK_LONG_STRONG_RECLAIM_SETUP_QUALITY_MIN", 0.52))
        )
        up_flat_shallow_long_lane = (
            macro_flow_regime == "trend_long"
            and not strong_reclaim_probe
            and continuation_pressure
            <= float(
                getattr(
                    config,
                    "PREC_LOWVOL_UP_FLAT_SHALLOW_LONG_CONTINUATION_PRESSURE_MAX",
                    0.28,
                )
            )
            and rsi
            >= float(getattr(config, "PREC_LOWVOL_UP_FLAT_SHALLOW_LONG_RSI_MIN", 44.0))
            and projection_score
            <= float(
                getattr(
                    config,
                    "PREC_LOWVOL_UP_FLAT_SHALLOW_LONG_PROJECTION_SCORE_MAX",
                    0.30,
                )
            )
            and setup_quality
            < float(
                getattr(
                    config,
                    "PREC_LOWVOL_UP_FLAT_SHALLOW_LONG_SETUP_QUALITY_MAX",
                    0.52,
                )
            )
            and reversion_support
            < float(
                getattr(
                    config,
                    "PREC_LOWVOL_UP_FLAT_SHALLOW_LONG_REVERSION_SUPPORT_MAX",
                    0.72,
                )
            )
        )
    if up_flat_shallow_long_lane:
        return None
    up_flat_shallow_short_lane = False
    if (
        side == "short"
        and short_up_flat
        and range_reason == "volatility_compression"
        and flow_guard is not None
        and projection_score is not None
        and bool(getattr(config, "PREC_LOWVOL_UP_FLAT_SHALLOW_SHORT_GUARD_ENABLED", True))
    ):
        setup_quality = float(flow_guard.get("setup_quality") or 0.0)
        up_flat_shallow_short_lane = (
            projection_score
            <= float(getattr(config, "PREC_LOWVOL_UP_FLAT_SHALLOW_SHORT_PROJECTION_SCORE_MAX", 0.28))
            and setup_quality
            < float(getattr(config, "PREC_LOWVOL_UP_FLAT_SHALLOW_SHORT_SETUP_QUALITY_MAX", 0.50))
        )
    if up_flat_shallow_short_lane:
        return None
    if hostile_projection_lane:
        strong_reversal_probe = (
            rev_strength >= max(config.PREC_LOWVOL_REV_MIN_STRENGTH + 0.46, 0.82)
            and touch_ratio >= 0.55
        )
        if not strong_reversal_probe:
            return None

    setup_pressure = {}
    if side == "short":
        setup_pressure = _precision_lowvol_setup_pressure(
            side=side,
            range_reason=getattr(range_ctx, "reason", None),
        )
        if float(setup_pressure.get("active") or 0.0) > 0.0:
            reversion_support = float(flow_guard.get("reversion_support") or 0.0) if flow_guard is not None else 0.0
            setup_quality = float(flow_guard.get("setup_quality") or 0.0) if flow_guard is not None else 0.0
            strong_reentry_probe = (
                touch_ratio >= max(
                    0.0,
                    float(getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_ALLOW_TOUCH_RATIO_MIN", 0.58)),
                )
                and rev_strength >= max(
                    config.PREC_LOWVOL_REV_MIN_STRENGTH,
                    float(getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_ALLOW_REV_STRENGTH_MIN", 0.76)),
                )
                and setup_quality >= max(
                    0.0,
                    float(getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_ALLOW_SETUP_QUALITY_MIN", 0.30)),
                )
                and (
                    reversion_support >= max(
                        0.0,
                        float(getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_ALLOW_REVERSION_SUPPORT_MIN", 0.70)),
                    )
                    or (
                        projection_score is not None
                        and projection_score >= float(
                            getattr(config, "PREC_LOWVOL_SETUP_PRESSURE_ALLOW_PROJECTION_SCORE_MIN", 0.08)
                        )
                    )
                )
            )
            if not strong_reentry_probe:
                return None

    # current precision-lowvol short losers were repeatedly stopping inside seconds while
    # a subset still reached the old 2.0 pip TP; widen SL modestly and avoid over-stretching TP.
    sl = max(1.8, min(2.1, atr * 1.0))
    tp = max(
        max(2.0, sl * 1.10),
        min(2.4, atr * (1.02 + min(0.14, rev_strength * 0.12))),
    )
    conf = 58
    conf += int(min(10, abs(rsi - 50.0) * 0.6))
    conf += int(min(10, rev_strength * 4.0))
    conf += int(min(6, touch_ratio * 6.0))
    conf += int(min(4, range_score * 4.0))
    if vgap_bias_ok:
        conf += 3
    if short_up_lean:
        conf += 4
    if range_ctx and getattr(range_ctx, "active", False):
        conf += 2
    if flow_guard is not None:
        conf -= int(min(5.0, max(0.0, 0.62 - float(flow_guard["setup_quality"])) * 18.0))
    if hostile_projection_lane:
        conf -= 4
    if short_down_flat:
        conf -= 5

    size_boost = 0.0
    if vgap_bias_ok:
        size_boost += 0.05
    if touch_ratio >= 0.5:
        size_boost += 0.06
    if rev_strength >= 0.75:
        size_boost += 0.06
    if short_up_lean:
        size_boost += 0.10
        tp = max(tp, min(2.2, atr * 1.05))
    if short_down_flat:
        size_boost -= 0.08
    size_cap = 1.35 if rev_strength >= 0.75 else 1.25
    if hostile_projection_lane:
        size_cap = min(size_cap, 1.02)
    size_mult = max(0.85, min(size_cap, size_mult + size_boost))
    if flow_guard is not None:
        size_mult = max(
            0.78,
            min(size_cap, min(size_mult, proj_size_mult * (0.86 + float(flow_guard["setup_quality"]) * 0.14))),
        )
    if hostile_projection_lane:
        size_mult = max(
            0.78,
            min(size_cap, min(size_mult, proj_size_mult * (0.78 + float(flow_guard["setup_quality"]) * 0.12))),
        )
    if short_up_lean:
        size_mult = max(size_mult, min(size_cap, proj_size_mult * 1.08))
    if short_down_flat:
        size_mult = max(0.78, min(size_mult, proj_size_mult * 0.92))

    signal = {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(45, min(92, conf))),
        "tag": tag,
        "reason": "precision_lowvol",
        "range_score": round(range_score, 3),
        "size_mult": round(size_mult, 3),
        "projection": proj_detail,
    }
    if setup_pressure:
        signal["setup_pressure"] = setup_pressure
    _attach_flow_guard_context(signal, flow_guard)
    return signal


def _signal_compression_retest(
    fac_m1: Dict[str, object],
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    if bbw > COMPRESS_BBW_MAX or atr > COMPRESS_ATR_MAX:
        return None

    candles = get_candles_snapshot("M1", limit=60)
    snap = compute_range_snapshot(candles or [], lookback=20, hi_pct=95.0, lo_pct=5.0)
    if not snap:
        return None

    mids, span = tick_snapshot(8.0, limit=120)
    imb = tick_imbalance(mids, span)
    if not imb or len(mids) < COMPRESS_MIN_TICKS:
        return None

    now = time.monotonic()
    state = _RETEST_STATE.get(tag)
    if state and now > state.expires_ts:
        _RETEST_STATE.pop(tag, None)
        state = None
    if state and now <= state.expires_ts:
        if abs(price - state.level) / PIP <= COMPRESS_RETEST_BAND_PIPS:
            rev_ok, rev_dir, _ = tick_reversal(mids, min_ticks=6)
            if rev_ok and rev_dir == state.direction:
                sl = max(1.5, min(2.0, atr * 0.9))
                tp = max(2.0, min(3.0, atr * 1.2))
                conf = 60 + int(min(15, abs(imb.momentum_pips)))
                return {
                    "action": "OPEN_LONG" if state.direction == "long" else "OPEN_SHORT",
                    "sl_pips": round(sl, 2),
                    "tp_pips": round(tp, 2),
                    "confidence": int(max(45, min(92, conf))),
                    "tag": tag,
                    "reason": "compression_retest",
                    "breakout_pips": round(state.breakout_pips, 2),
                }
        return None

    # detect breakout
    breakout_long = price >= snap.high + COMPRESS_BREAKOUT_PIPS * PIP
    breakout_short = price <= snap.low - COMPRESS_BREAKOUT_PIPS * PIP
    if not breakout_long and not breakout_short:
        return None
    if abs(imb.momentum_pips) < COMPRESS_MIN_IMPULSE_PIPS:
        return None
    direction = "long" if breakout_long else "short"
    level = snap.high if breakout_long else snap.low
    _RETEST_STATE[tag] = RetestState(
        direction=direction,
        level=level,
        started_ts=now,
        expires_ts=now + COMPRESS_RETEST_TIMEOUT_SEC,
        breakout_pips=abs(imb.momentum_pips),
    )
    return None


def _signal_squeeze_pulse_break(
    fac_m1: Dict[str, object],
    range_ctx,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=SPB_SPREAD_P25)
    if not ok_spread:
        return None
    if SPB_ALLOWED_REGIMES:
        regime = str(fac_m1.get("regime") or "").strip()
        if not regime:
            try:
                regime = str(current_regime("M1", event_mode=False) or "").strip()
            except Exception:
                regime = ""
        if regime and regime.lower() not in SPB_ALLOWED_REGIMES:
            return None

    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    if bbw <= 0.0 or atr <= 0.0:
        return None
    if bbw > SPB_BBW_MAX:
        return None
    if atr < SPB_ATR_MIN or atr > SPB_ATR_MAX:
        return None

    if range_ctx is not None:
        try:
            score = float(getattr(range_ctx, "score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        if SPB_RANGE_SCORE_MIN > 0.0 and score < SPB_RANGE_SCORE_MIN:
            return None

    candles = get_candles_snapshot("M1", limit=max(40, SPB_LOOKBACK + 6))
    snap = compute_range_snapshot(
        candles or [],
        lookback=max(10, SPB_LOOKBACK),
        hi_pct=float(SPB_HI_PCT),
        lo_pct=float(SPB_LO_PCT),
    )
    if not snap:
        return None

    mids, span = tick_snapshot(SPB_TICK_WINDOW_SEC, limit=200)
    if not mids or len(mids) < SPB_MIN_TICKS:
        return None
    tick_range_pips = (max(mids) - min(mids)) / PIP
    if tick_range_pips > SPB_TICK_RANGE_MAX_PIPS:
        return None

    imb = tick_imbalance(mids, span)
    if not imb:
        return None
    if imb.ratio < SPB_IMB_RATIO_MIN:
        return None
    if abs(imb.momentum_pips) < SPB_MOM_MIN_PIPS:
        return None

    breakout_long = price >= snap.high + SPB_BREAKOUT_PIPS * PIP
    breakout_short = price <= snap.low - SPB_BREAKOUT_PIPS * PIP
    if not breakout_long and not breakout_short:
        return None

    side = "long" if breakout_long else "short"
    # Require tick momentum in the breakout direction.
    if (side == "long" and imb.momentum_pips <= 0) or (side == "short" and imb.momentum_pips >= 0):
        return None

    sl = max(1.3, min(2.4, atr * 0.85))
    tp = max(sl * 1.45, min(4.2, atr * 1.35))
    conf = 62
    conf += int(min(12.0, abs(imb.momentum_pips) * 4.0))
    conf += int(min(10.0, (SPB_TICK_RANGE_MAX_PIPS - tick_range_pips) * 3.0))
    conf = int(max(45, min(92, conf)))

    size_mult = 1.0 + min(0.35, abs(imb.momentum_pips) * 0.08)
    size_mult = max(0.6, min(1.4, size_mult * SPB_SIZE_MULT))

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": conf,
        "tag": tag,
        "reason": "squeeze_pulse_break",
        "size_mult": round(size_mult, 3),
        "compression": {
            "bbw": round(bbw, 6),
            "tick_range_pips": round(tick_range_pips, 3),
            "momentum_pips": round(imb.momentum_pips, 3),
        },
    }


def _spb_post_entry_guard(
    signal: Dict[str, object],
    fac_m1: Dict[str, object],
    now_utc: datetime.datetime,
) -> Tuple[bool, str]:
    if str(signal.get("tag") or "").strip() != "SqueezePulseBreak":
        return True, "not_spb"

    action = str(signal.get("action") or "").strip().upper()
    side = "long" if action == "OPEN_LONG" else "short" if action == "OPEN_SHORT" else ""
    if not side:
        return False, "unknown_side"

    if side == "long" and not SPB_ALLOW_LONG:
        return False, "long_disabled"
    if side == "short" and not SPB_ALLOW_SHORT:
        return False, "short_disabled"

    if SPB_BLOCK_JST_HOURS:
        jst_hour = now_utc.astimezone(
            datetime.timezone(datetime.timedelta(hours=9))
        ).hour
        if jst_hour in SPB_BLOCK_JST_HOURS:
            return False, f"jst_block_{jst_hour}"

    adx = _adx(fac_m1)
    if SPB_MAX_ADX > 0.0 and adx > SPB_MAX_ADX:
        return False, "adx_cap"

    rsi = _rsi(fac_m1)
    if side == "long" and not (SPB_LONG_RSI_MIN <= rsi <= SPB_LONG_RSI_MAX):
        return False, "rsi_long_band"
    if side == "short" and not (SPB_SHORT_RSI_MIN <= rsi <= SPB_SHORT_RSI_MAX):
        return False, "rsi_short_band"

    try:
        air_score = float(signal.get("air_score") or 0.0)
    except Exception:
        air_score = 0.0
    if SPB_MIN_AIR_SCORE > 0.0 and air_score < SPB_MIN_AIR_SCORE:
        return False, "air_score_low"

    if SPB_REQUIRE_AIR_MATCH:
        air_dir = str(signal.get("air_pressure_dir") or "").strip().lower()
        if air_dir not in {"long", "short"}:
            return False, "air_dir_missing"
        if air_dir != side:
            return False, "air_dir_mismatch"

    return True, "ok"


def _spb_diag_metrics(fac_m1: Dict[str, object], range_ctx) -> Dict[str, object]:
    """Lightweight diag snapshot for SqueezePulseBreak tuning."""
    price = _latest_price(fac_m1)
    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    ok_spread, spread_state = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=SPB_SPREAD_P25)
    spread_pips = 0.0
    p25_pips = 0.0
    if isinstance(spread_state, dict):
        try:
            spread_pips = float(spread_state.get("spread_pips") or 0.0)
        except Exception:
            spread_pips = 0.0
        try:
            p25_pips = float(spread_state.get("p25_pips") or 0.0)
        except Exception:
            p25_pips = 0.0

    range_active = False
    range_score = 0.0
    if range_ctx is not None:
        try:
            range_active = bool(getattr(range_ctx, "active", False))
            range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
        except Exception:
            range_active = False
            range_score = 0.0

    snap = None
    try:
        candles = get_candles_snapshot("M1", limit=max(40, SPB_LOOKBACK + 6))
        snap = compute_range_snapshot(
            candles or [],
            lookback=max(10, SPB_LOOKBACK),
            hi_pct=float(SPB_HI_PCT),
            lo_pct=float(SPB_LO_PCT),
        )
    except Exception:
        snap = None

    breakout_long = False
    breakout_short = False
    if snap and price > 0:
        breakout_long = price >= float(snap.high) + SPB_BREAKOUT_PIPS * PIP
        breakout_short = price <= float(snap.low) - SPB_BREAKOUT_PIPS * PIP

    mids, span = tick_snapshot(SPB_TICK_WINDOW_SEC, limit=200)
    tick_n = len(mids or [])
    tick_range_pips = 0.0
    if mids:
        tick_range_pips = (max(mids) - min(mids)) / PIP

    imb = tick_imbalance(mids, span) if mids else None
    imb_ratio = float(getattr(imb, "ratio", 0.0) or 0.0) if imb else 0.0
    imb_mom = float(getattr(imb, "momentum_pips", 0.0) or 0.0) if imb else 0.0

    return {
        "price": round(price, 3),
        "bbw": round(bbw, 6),
        "atr": round(atr, 3),
        "spread_ok": bool(ok_spread),
        "spread_pips": round(spread_pips, 3),
        "p25_pips": round(p25_pips, 3),
        "range_active": bool(range_active),
        "range_score": round(range_score, 3),
        "breakout_long": bool(breakout_long),
        "breakout_short": bool(breakout_short),
        "tick_n": int(tick_n),
        "tick_range_pips": round(tick_range_pips, 3),
        "imb_ratio": round(imb_ratio, 3),
        "imb_mom_pips": round(imb_mom, 3),
    }


def _wick_hf_diag_metrics(fac_m1: Dict[str, object], range_ctx) -> Dict[str, object]:
    """Lightweight diag snapshot for WickReversalHF tuning."""
    adx = _adx(fac_m1)
    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)

    ok_spread, spread_state = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=WICK_HF_SPREAD_P25)
    spread_pips = 0.0
    p25_pips = 0.0
    if isinstance(spread_state, dict):
        try:
            spread_pips = float(spread_state.get("spread_pips") or 0.0)
        except Exception:
            spread_pips = 0.0
        try:
            p25_pips = float(spread_state.get("p25_pips") or 0.0)
        except Exception:
            p25_pips = 0.0

    range_active = False
    range_score = 0.0
    if range_ctx is not None:
        try:
            range_active = bool(getattr(range_ctx, "active", False))
            range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
        except Exception:
            range_active = False
            range_score = 0.0
    range_ok = True
    if not range_active and WICK_HF_RANGE_SCORE_MIN > 0.0 and range_score < WICK_HF_RANGE_SCORE_MIN:
        range_ok = False

    rng_pips = 0.0
    body_pips = 0.0
    upper_wick_pips = 0.0
    lower_wick_pips = 0.0
    wick_ratio = 0.0
    side = ""
    candle_high = 0.0
    candle_low = 0.0
    try:
        candles = get_candles_snapshot("M1", limit=2) or []
        last = candles[-1] if candles else None
        if isinstance(last, dict):
            o = float(last.get("open") or last.get("o") or 0.0)
            h = float(last.get("high") or last.get("h") or 0.0)
            l = float(last.get("low") or last.get("l") or 0.0)
            c = float(last.get("close") or last.get("c") or 0.0)
            if h > 0 and l > 0 and h >= l:
                candle_high = h
                candle_low = l
                rng_pips = (h - l) / PIP
                body_pips = abs(c - o) / PIP
                upper_wick_pips = (h - max(o, c)) / PIP
                lower_wick_pips = (min(o, c) - l) / PIP
                wick_ratio = max(upper_wick_pips, lower_wick_pips) / max(rng_pips, 0.01)
                side = "short" if upper_wick_pips > lower_wick_pips else "long"
    except Exception:
        pass

    rng_ok = rng_pips >= WICK_HF_RANGE_MIN_PIPS
    body_ok = body_pips <= WICK_HF_BODY_MAX_PIPS
    ratio_ok = wick_ratio >= WICK_HF_RATIO_MIN
    adx_ok = not (WICK_HF_ADX_MAX > 0.0 and adx > WICK_HF_ADX_MAX)
    bbw_ok = not (WICK_HF_BBW_MAX > 0.0 and bbw > WICK_HF_BBW_MAX)
    atr_ok = not (
        (WICK_HF_ATR_MIN > 0.0 and atr < WICK_HF_ATR_MIN) or (WICK_HF_ATR_MAX > 0.0 and atr > WICK_HF_ATR_MAX)
    )

    bb_ok = True
    bb_touch_pips = 0.0
    if WICK_HF_REQUIRE_BB_TOUCH:
        bb_ok = False
        levels = bb_levels(fac_m1)
        if levels:
            try:
                upper, _, lower, _, span_pips = levels
                bb_touch_pips = max(WICK_HF_BB_TOUCH_PIPS, float(span_pips or 0.0) * 0.18)
                if side == "short":
                    bb_ok = upper_wick_pips > 0 and (candle_high >= float(upper) - bb_touch_pips * PIP)
                elif side == "long":
                    bb_ok = lower_wick_pips > 0 and (candle_low <= float(lower) + bb_touch_pips * PIP)
            except Exception:
                bb_ok = False

    tick_n = 0
    tick_ok = True
    tick_dir = None
    tick_strength = 0.0
    tick_gate_ok = True
    if WICK_HF_REQUIRE_TICK_REV:
        tick_ok = False
        tick_gate_ok = False
        mids, _ = tick_snapshot(WICK_HF_TICK_WINDOW_SEC, limit=160)
        tick_n = len(mids or [])
        rev_ok, rev_dir, rev_strength = (
            tick_reversal(mids, min_ticks=WICK_HF_TICK_MIN_TICKS) if mids else (False, None, 0.0)
        )
        tick_ok = bool(rev_ok)
        tick_dir = rev_dir
        try:
            tick_strength = float(rev_strength or 0.0)
        except Exception:
            tick_strength = 0.0
        tick_gate_ok = tick_ok and (tick_dir == side) and (WICK_HF_TICK_MIN_STRENGTH <= 0.0 or tick_strength >= WICK_HF_TICK_MIN_STRENGTH)

    proj_allow = None
    try:
        allow, _, _ = projection_decision(side, mode="range")
        proj_allow = bool(allow)
    except Exception:
        proj_allow = None

    block = "ok"
    if not range_ok:
        block = "range"
    elif not ok_spread:
        block = "spread"
    elif not rng_ok:
        block = "rng"
    elif not body_ok:
        block = "body"
    elif not ratio_ok:
        block = "ratio"
    elif not adx_ok:
        block = "adx"
    elif not bbw_ok:
        block = "bbw"
    elif not atr_ok:
        block = "atr"
    elif not bb_ok:
        block = "bb_touch"
    elif not tick_gate_ok:
        block = "tick_rev"
    elif proj_allow is False:
        block = "projection"

    return {
        "block": block,
        "range_ok": bool(range_ok),
        "range_active": bool(range_active),
        "range_score": round(range_score, 3),
        "spread_ok": bool(ok_spread),
        "spread_pips": round(spread_pips, 3),
        "p25_pips": round(p25_pips, 3),
        "rng_pips": round(rng_pips, 2),
        "body_pips": round(body_pips, 2),
        "wick_ratio": round(wick_ratio, 3),
        "upper_wick_pips": round(upper_wick_pips, 2),
        "lower_wick_pips": round(lower_wick_pips, 2),
        "side": side,
        "rng_ok": bool(rng_ok),
        "body_ok": bool(body_ok),
        "ratio_ok": bool(ratio_ok),
        "adx": round(adx, 2),
        "bbw": round(bbw, 6),
        "atr": round(atr, 3),
        "adx_ok": bool(adx_ok),
        "bbw_ok": bool(bbw_ok),
        "atr_ok": bool(atr_ok),
        "bb_ok": bool(bb_ok),
        "bb_touch_pips": round(bb_touch_pips, 2),
        "tick_n": int(tick_n),
        "tick_ok": bool(tick_ok),
        "tick_dir": tick_dir,
        "tick_strength": round(tick_strength, 3),
        "proj_allow": proj_allow,
    }


def _signal_false_break_fade(
    fac_m1: Dict[str, object],
    range_ctx,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=FBF_SPREAD_P25)
    if not ok_spread:
        return None

    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    adx = _adx(fac_m1)
    if bbw <= 0.0 or atr <= 0.0:
        return None
    if bbw > FBF_BBW_MAX:
        return None
    if atr < FBF_ATR_MIN or atr > FBF_ATR_MAX:
        return None
    if adx > FBF_ADX_MAX:
        return None

    range_score = 0.0
    range_active = False
    if range_ctx is not None:
        try:
            range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
            range_active = bool(getattr(range_ctx, "active", False))
        except Exception:
            range_score = 0.0
            range_active = False
    if not range_active and FBF_RANGE_SCORE_MIN > 0.0 and range_score < FBF_RANGE_SCORE_MIN:
        return None

    candles = get_candles_snapshot("M1", limit=max(40, FBF_LOOKBACK + 6))
    snap = compute_range_snapshot(
        candles or [],
        lookback=max(10, FBF_LOOKBACK),
        hi_pct=float(FBF_HI_PCT),
        lo_pct=float(FBF_LO_PCT),
    )
    if not snap:
        return None

    now = time.monotonic()
    state = _FALSE_BREAK_STATE.get(tag)
    if state and now > state.expires_ts:
        _FALSE_BREAK_STATE.pop(tag, None)
        state = None

    # If a state exists, wait for reclaim + (optional) tick reversal then fade.
    if state and now <= state.expires_ts:
        if state.side == "high":
            state.extreme = max(state.extreme, price)
            sweep_pips = (state.extreme - state.level) / PIP
            reclaimed = price <= state.level - FBF_RECLAIM_PIPS * PIP
            want_dir = "short"
        else:
            state.extreme = min(state.extreme, price)
            sweep_pips = (state.level - state.extreme) / PIP
            reclaimed = price >= state.level + FBF_RECLAIM_PIPS * PIP
            want_dir = "long"
        if not reclaimed:
            return None
        if sweep_pips < FBF_MIN_SWEEP_PIPS:
            return None

        slope_10_pips = _ema_slope_pips(fac_m1, "ema_slope_10")
        vwap_gap_pips = _vwap_gap_pips(fac_m1)
        if want_dir == "long":
            if FBF_MAX_COUNTER_SLOPE_PIPS > 0.0 and slope_10_pips < -FBF_MAX_COUNTER_SLOPE_PIPS:
                return None
            if FBF_MAX_COUNTER_VWAP_GAP_PIPS > 0.0 and vwap_gap_pips < -FBF_MAX_COUNTER_VWAP_GAP_PIPS:
                return None
        else:
            if FBF_MAX_COUNTER_SLOPE_PIPS > 0.0 and slope_10_pips > FBF_MAX_COUNTER_SLOPE_PIPS:
                return None
            if FBF_MAX_COUNTER_VWAP_GAP_PIPS > 0.0 and vwap_gap_pips > FBF_MAX_COUNTER_VWAP_GAP_PIPS:
                return None

        mids, _ = tick_snapshot(FBF_TICK_WINDOW_SEC, limit=140)
        rev_ok, rev_dir, rev_strength = tick_reversal(mids, min_ticks=6) if mids else (False, None, 0.0)
        if FBF_REQUIRE_TICK_REVERSAL and (not rev_ok or rev_dir != want_dir):
            return None
        if FBF_TICK_MIN_STRENGTH > 0.0:
            try:
                rev_strength_f = float(rev_strength or 0.0)
            except Exception:
                rev_strength_f = 0.0
            if rev_strength_f < FBF_TICK_MIN_STRENGTH:
                return None

        sl = max(1.4, min(2.8, atr * 0.9))
        tp = max(sl * 1.35, min(4.0, atr * 1.25))
        conf = 60 + int(min(12.0, sweep_pips * 3.0)) + int(min(10.0, rev_strength * 4.0))
        conf = int(max(45, min(92, conf)))

        size_mult = max(0.6, min(1.4, FBF_SIZE_MULT * (1.0 + min(0.25, sweep_pips * 0.05))))
        _FALSE_BREAK_STATE.pop(tag, None)
        return {
            "action": "OPEN_LONG" if want_dir == "long" else "OPEN_SHORT",
            "sl_pips": round(sl, 2),
            "tp_pips": round(tp, 2),
            "confidence": conf,
            "tag": tag,
            "reason": "false_break_fade",
            "size_mult": round(size_mult, 3),
            "fade": {
                "side": state.side,
                "sweep_pips": round(sweep_pips, 2),
                "range_score": round(range_score, 3),
            },
        }

    # Detect a fresh breakout beyond the range and arm a fade state.
    breakout_high = price >= snap.high + FBF_BREAKOUT_PIPS * PIP
    breakout_low = price <= snap.low - FBF_BREAKOUT_PIPS * PIP
    if not breakout_high and not breakout_low:
        return None

    side = "high" if breakout_high else "low"
    level = snap.high if breakout_high else snap.low
    extreme = price
    direction = "short" if breakout_high else "long"
    _FALSE_BREAK_STATE[tag] = FalseBreakState(
        direction=direction,
        side=side,
        level=level,
        started_ts=now,
        expires_ts=now + FBF_TIMEOUT_SEC,
        extreme=extreme,
    )
    return None


def _signal_htf_pullback(
    fac_m1: Dict[str, object],
    fac_h1: Dict[str, object],
    fac_m5: Dict[str, object],
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    try:
        h1_close = float(fac_h1.get("close") or 0.0)
    except Exception:
        h1_close = 0.0
    try:
        h1_ma20 = float(fac_h1.get("ma20") or fac_h1.get("ema20") or 0.0)
    except Exception:
        h1_ma20 = 0.0
    try:
        h1_adx = float(fac_h1.get("adx") or 0.0)
    except Exception:
        h1_adx = 0.0
    if h1_close <= 0 or h1_ma20 <= 0 or h1_adx < HTF_ADX_MIN:
        return None

    gap_pips = (h1_close - h1_ma20) / PIP
    if abs(gap_pips) < HTF_GAP_PIPS:
        return None

    direction = "long" if gap_pips > 0 else "short"

    try:
        m5_bbw = float(fac_m5.get("bbw") or 0.0)
    except Exception:
        m5_bbw = 0.0
    if m5_bbw > HTF_M5_BBW_MAX:
        return None

    try:
        m1_ema20 = float(fac_m1.get("ema20") or fac_m1.get("ma20") or 0.0)
    except Exception:
        m1_ema20 = 0.0
    if m1_ema20 <= 0:
        return None
    dist = abs(price - m1_ema20) / PIP
    if dist > HTF_PULLBACK_BAND_PIPS:
        return None

    rsi = _rsi(fac_m1)
    if direction == "long" and rsi > HTF_RSI_LONG_MAX:
        return None
    if direction == "short" and rsi < HTF_RSI_SHORT_MIN:
        return None

    proj_allow, proj_mult, proj_detail = projection_decision(direction, mode="pullback")
    if not proj_allow:
        return None

    atr = _atr_pips(fac_m1)
    sl = max(1.6, min(2.2, atr * 0.9))
    tp = max(1.9, min(2.4, atr * 1.05))
    conf = 62 + int(min(12, abs(gap_pips) * 0.6))
    if proj_mult < 1.0:
        conf -= 6

    return {
        "action": "OPEN_LONG" if direction == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(45, min(92, conf))),
        "tag": tag,
        "reason": "htf_pullback",
        "projection": proj_detail,
    }


def _tick_imb_reentry_gap_ok(
    fac_m1: Dict[str, object],
    *,
    direction: str,
    tag: str,
) -> bool:
    if TICK_IMB_REENTRY_LOOKBACK_SEC <= 0.0 or TICK_IMB_REENTRY_MIN_PRICE_GAP_PIPS <= 0.0:
        return True
    latest = _fetch_latest_closed_trade(tag)
    if latest is None:
        return True
    close_ts, last_entry_price, last_pl_pips, last_units = latest
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    age_sec = (now_utc - close_ts).total_seconds()
    if age_sec < 0.0 or age_sec > TICK_IMB_REENTRY_LOOKBACK_SEC:
        return True
    if TICK_IMB_REENTRY_REQUIRE_LAST_PROFIT and last_pl_pips <= 0.0:
        return True
    last_dir = "long" if last_units > 0 else "short"
    if last_units == 0 or last_dir != direction:
        return True
    price = _latest_price(fac_m1)
    if price <= 0.0:
        return True
    gap_pips = abs(price - last_entry_price) / PIP
    return gap_pips >= TICK_IMB_REENTRY_MIN_PRICE_GAP_PIPS


def _signal_tick_imbalance(
    fac_m1: Dict[str, object],
    range_ctx=None,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    if range_ctx is not None:
        try:
            range_active = bool(getattr(range_ctx, "active", False))
            range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
        except Exception:
            range_active = False
            range_score = 0.0
        if TICK_IMB_BLOCK_RANGE_MODE and range_active:
            return None
        if TICK_IMB_RANGE_SCORE_MAX > 0.0 and range_score >= TICK_IMB_RANGE_SCORE_MAX:
            return None
    if TICK_IMB_ALLOWED_REGIMES:
        regime = str(fac_m1.get("regime") or "").strip()
        if not regime:
            try:
                regime = str(current_regime("M1", event_mode=False) or "").strip()
            except Exception:
                regime = ""
        if regime and regime.lower() not in TICK_IMB_ALLOWED_REGIMES:
            return None
    if _adx(fac_m1) < TICK_IMB_ADX_MIN:
        return None
    if _bbw(fac_m1) < TICK_IMB_BBW_MIN:
        return None
    mids, span = tick_snapshot(TICK_IMB_WINDOW_SEC, limit=160)
    imb = tick_imbalance(mids, span)
    if not imb:
        return None
    atr = _atr_pips(fac_m1)
    if atr < TICK_IMB_ATR_MIN:
        return None
    if imb.ratio < TICK_IMB_RATIO_MIN:
        return None
    if abs(imb.momentum_pips) < TICK_IMB_MOM_MIN_PIPS:
        return None
    if imb.range_pips < TICK_IMB_RANGE_MIN_PIPS:
        return None

    direction = "long" if imb.momentum_pips > 0 else "short"
    if not _tick_imb_reentry_gap_ok(fac_m1, direction=direction, tag=tag):
        return None
    if TICK_IMB_ENTRY_QUALITY_ENABLED:
        if imb.ratio < TICK_IMB_QUALITY_RATIO_MIN:
            return None
        if abs(imb.momentum_pips) < TICK_IMB_QUALITY_MOM_MIN_PIPS:
            return None
        if imb.range_pips < TICK_IMB_QUALITY_RANGE_MIN_PIPS:
            return None
        mids_confirm, span_confirm = tick_snapshot(TICK_IMB_CONFIRM_WINDOW_SEC, limit=220)
        imb_confirm = tick_imbalance(mids_confirm, span_confirm)
        if not imb_confirm:
            return None
        if imb_confirm.ratio < TICK_IMB_CONFIRM_RATIO_MIN:
            return None
        confirm_signed_mom = imb_confirm.momentum_pips if direction == "long" else -imb_confirm.momentum_pips
        if confirm_signed_mom <= TICK_IMB_CONFIRM_SIGNED_MOM_MIN_PIPS:
            return None
        if TICK_IMB_REQUIRE_CONFIRM_SIGN and (imb_confirm.momentum_pips > 0.0) != (imb.momentum_pips > 0.0):
            return None
    if TICK_IMB_REQUIRE_MA_ALIGN:
        try:
            ma10 = float(fac_m1.get("ma10") or 0.0)
            ma20 = float(fac_m1.get("ma20") or 0.0)
        except Exception:
            ma10 = 0.0
            ma20 = 0.0
        if ma10 <= 0.0 or ma20 <= 0.0:
            if TICK_IMB_MA_ALIGN_STRICT:
                return None
        else:
            gap_pips = (ma10 - ma20) / PIP
            if direction == "long" and gap_pips < TICK_IMB_MA_GAP_MIN_PIPS:
                return None
            if direction == "short" and gap_pips > -TICK_IMB_MA_GAP_MIN_PIPS:
                return None
    sl = max(1.2, min(2.0, atr * 0.8))
    tp = max(1.5, min(2.5, atr * 1.05))
    conf = 60 + int(min(15, imb.ratio * 20.0))
    return {
        "action": "OPEN_LONG" if direction == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(45, min(92, conf))),
        "tag": tag,
        "reason": "tick_imbalance",
        "size_mult": round(TICK_IMB_SIZE_MULT, 3),
        "imbalance": {"ratio": round(imb.ratio, 3), "momentum_pips": round(imb.momentum_pips, 3)},
    }


def _signal_tick_imbalance_rrplus(
    fac_m1: Dict[str, object],
    range_ctx=None,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None

    ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=1.2)
    if not ok_spread:
        return None

    if range_ctx is not None:
        try:
            range_active = bool(getattr(range_ctx, "active", False))
            range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
        except Exception:
            range_active = False
            range_score = 0.0
        if TIRP_BLOCK_RANGE_MODE and range_active:
            return None
        if TIRP_RANGE_SCORE_MAX > 0.0 and range_score >= TIRP_RANGE_SCORE_MAX:
            return None

    atr = _atr_pips(fac_m1)
    if atr < TIRP_ATR_MIN or atr > TIRP_ATR_MAX:
        return None
    if _adx(fac_m1) < TIRP_ADX_MIN:
        return None
    if _bbw(fac_m1) < TIRP_BBW_MIN:
        return None

    mids, span = tick_snapshot(TIRP_WINDOW_SEC, limit=200)
    imb = tick_imbalance(mids, span)
    if not imb:
        return None
    if imb.ratio < TIRP_RATIO_MIN:
        return None
    if abs(imb.momentum_pips) < TIRP_MOM_MIN_PIPS:
        return None
    if imb.range_pips < TIRP_RANGE_MIN_PIPS:
        return None

    direction = "long" if imb.momentum_pips > 0 else "short"
    if TIRP_REQUIRE_MA_ALIGN:
        try:
            ma10 = float(fac_m1.get("ma10") or 0.0)
            ma20 = float(fac_m1.get("ma20") or 0.0)
        except Exception:
            ma10 = 0.0
            ma20 = 0.0
        if ma10 > 0.0 and ma20 > 0.0:
            gap_pips = (ma10 - ma20) / PIP
            if direction == "long" and gap_pips < TIRP_MA_GAP_MIN_PIPS:
                return None
            if direction == "short" and gap_pips > -TIRP_MA_GAP_MIN_PIPS:
                return None

    sl = max(1.3, min(2.4, atr * 0.75 + 0.3))
    tp = max(sl * 1.55, min(4.4, atr * 1.45))
    conf = 62 + int(min(16.0, (imb.ratio - 0.6) * 30.0)) + int(min(10.0, abs(imb.momentum_pips) * 3.0))
    conf = int(max(45, min(92, conf)))

    return {
        "action": "OPEN_LONG" if direction == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": conf,
        "tag": tag,
        "reason": "tick_imbalance_rrplus",
        "size_mult": round(TIRP_SIZE_MULT, 3),
        "imbalance": {
            "ratio": round(imb.ratio, 3),
            "momentum_pips": round(imb.momentum_pips, 3),
            "range_pips": round(imb.range_pips, 3),
        },
    }


def _signal_level_reject(
    fac_m1: Dict[str, object],
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    if LEVEL_REJECT_ALLOWED_REGIMES:
        regime = str(fac_m1.get("regime") or "").strip()
        if not regime:
            try:
                regime = str(current_regime("M1", event_mode=False) or "").strip()
            except Exception:
                regime = ""
        if regime and regime.lower() not in LEVEL_REJECT_ALLOWED_REGIMES:
            return None
    if LEVEL_REJECT_SPREAD_P25 > 0.0:
        ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=LEVEL_REJECT_SPREAD_P25)
        if not ok_spread:
            return None
    if LEVEL_REJECT_ADX_MAX > 0 and _adx(fac_m1) > LEVEL_REJECT_ADX_MAX:
        return None
    atr = _atr_pips(fac_m1)
    if LEVEL_REJECT_ATR_MAX > 0.0 and atr > LEVEL_REJECT_ATR_MAX:
        return None
    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    candles = get_candles_snapshot("M1", limit=60)
    snap = compute_range_snapshot(candles or [], lookback=LEVEL_LOOKBACK, hi_pct=95.0, lo_pct=5.0)
    if not snap:
        return None

    mids, _ = tick_snapshot(6.0, limit=80)
    rev_ok, rev_dir, _ = tick_reversal(mids, min_ticks=6)
    if not rev_ok:
        return None

    rsi = _rsi(fac_m1)
    dist_high = abs(price - snap.high) / PIP
    dist_low = abs(price - snap.low) / PIP

    if dist_high <= LEVEL_BAND_PIPS and rsi >= LEVEL_RSI_SHORT_MIN and rev_dir == "short":
        side = "short"
    elif dist_low <= LEVEL_BAND_PIPS and rsi <= LEVEL_RSI_LONG_MAX and rev_dir == "long":
        side = "long"
    else:
        return None

    sl = max(1.2, min(1.8, atr * 0.8))
    tp = max(1.5, min(2.2, atr * 1.1))
    conf = 58 + int(min(12, abs(rsi - 50.0) * 0.6))

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(45, min(92, conf))),
        "tag": tag,
        "reason": "level_reject",
        "size_mult": round(LEVEL_REJECT_SIZE_MULT, 3),
    }

def _signal_level_reject_plus(
    fac_m1: Dict[str, object],
    range_ctx=None,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    """A stricter LevelReject with wick + tick confirmation (higher precision, fewer trades)."""
    if LRP_ALLOWED_REGIMES:
        regime = str(fac_m1.get("regime") or "").strip()
        if not regime:
            try:
                regime = str(current_regime("M1", event_mode=False) or "").strip()
            except Exception:
                regime = ""
        if regime and regime.lower() not in LRP_ALLOWED_REGIMES:
            return None

    price = _latest_price(fac_m1)
    if price <= 0:
        return None

    ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=LRP_SPREAD_P25)
    if not ok_spread:
        return None

    range_active = False
    range_score = 0.0
    if range_ctx is not None:
        try:
            range_active = bool(getattr(range_ctx, "active", False))
            range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
        except Exception:
            range_active = False
            range_score = 0.0
    if not range_active and LRP_RANGE_SCORE_MIN > 0.0 and range_score < LRP_RANGE_SCORE_MIN:
        return None

    adx = _adx(fac_m1)
    bbw = _bbw(fac_m1)
    if LRP_ADX_MAX > 0.0 and adx > LRP_ADX_MAX:
        return None
    if LRP_BBW_MAX > 0.0 and bbw > LRP_BBW_MAX:
        return None

    candles = get_candles_snapshot("M1", limit=max(60, LEVEL_LOOKBACK + 10))
    snap = compute_range_snapshot(candles or [], lookback=LEVEL_LOOKBACK, hi_pct=95.0, lo_pct=5.0)
    if not snap:
        return None

    # Require a rejection candle shape (wick-based evidence).
    try:
        last = (candles or [])[-1]
        o = float(last.get("open") or last.get("o") or 0.0)
        h = float(last.get("high") or last.get("h") or 0.0)
        l = float(last.get("low") or last.get("l") or 0.0)
        c = float(last.get("close") or last.get("c") or 0.0)
    except Exception:
        return None
    if h <= 0 or l <= 0 or h <= l:
        return None
    rng_pips = (h - l) / PIP
    body_pips = abs(c - o) / PIP
    body_ratio = body_pips / max(0.01, rng_pips)
    upper_wick = (h - max(o, c)) / PIP
    lower_wick = (min(o, c) - l) / PIP
    wick_ratio = max(upper_wick, lower_wick) / max(0.01, rng_pips)
    if wick_ratio < LRP_WICK_RATIO_MIN:
        return None
    if body_pips > LRP_BODY_MAX_PIPS or body_ratio > LRP_BODY_RATIO_MAX:
        return None

    mids, _ = tick_snapshot(LRP_TICK_WINDOW_SEC, limit=200)
    rev_ok, rev_dir, rev_strength = (
        tick_reversal(mids, min_ticks=LRP_TICK_MIN_TICKS) if mids else (False, None, 0.0)
    )
    if not rev_ok:
        return None
    try:
        tick_strength = float(rev_strength or 0.0)
    except Exception:
        tick_strength = 0.0
    if LRP_TICK_MIN_STRENGTH > 0.0 and tick_strength < LRP_TICK_MIN_STRENGTH:
        return None

    rsi = _rsi(fac_m1)
    dist_high = abs(price - snap.high) / PIP
    dist_low = abs(price - snap.low) / PIP

    side = None
    if (
        dist_high <= LRP_BAND_PIPS
        and rsi >= LEVEL_RSI_SHORT_MIN
        and str(rev_dir or "") == "short"
        and upper_wick >= lower_wick
    ):
        side = "short"
    elif (
        dist_low <= LRP_BAND_PIPS
        and rsi <= LEVEL_RSI_LONG_MAX
        and str(rev_dir or "") == "long"
        and lower_wick >= upper_wick
    ):
        side = "long"
    else:
        return None

    proj_allow, size_mult, proj_detail = projection_decision(side, mode="range")
    if not proj_allow:
        return None

    atr = _atr_pips(fac_m1)
    sl = max(1.2, min(2.2, atr * 0.85))
    tp = max(sl * 1.35, min(3.4, atr * 1.15))
    conf = 62
    conf += int(min(10.0, abs(rsi - 50.0) * 0.6))
    conf += int(min(10.0, wick_ratio * 18.0))
    conf += int(min(8.0, max(0.0, tick_strength) * 7.0))
    conf = int(max(45, min(92, conf)))

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": conf,
        "tag": tag,
        "reason": "level_reject_plus",
        "size_mult": round(LRP_SIZE_MULT * float(size_mult or 1.0), 3),
        "projection": proj_detail,
        "lrp": {
            "dist_high_pips": round(dist_high, 2),
            "dist_low_pips": round(dist_low, 2),
            "wick_ratio": round(wick_ratio, 3),
            "tick_strength": round(tick_strength, 3),
        },
    }


def _signal_liquidity_sweep(
    fac_m1: Dict[str, object],
    range_ctx=None,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    if LSR_ALLOWED_REGIMES:
        regime = str(fac_m1.get("regime") or "").strip()
        if not regime:
            try:
                regime = str(current_regime("M1", event_mode=False) or "").strip()
            except Exception:
                regime = ""
        if regime and regime.lower() not in LSR_ALLOWED_REGIMES:
            return None
    if _adx(fac_m1) > LSR_ADX_MAX or _bbw(fac_m1) > LSR_BBW_MAX:
        return None
    if range_ctx is not None:
        try:
            range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
            if LSR_RANGE_SCORE_MIN > 0.0 and range_score < LSR_RANGE_SCORE_MIN:
                return None
        except Exception:
            pass

    candles = get_candles_snapshot("M1", limit=LSR_LOOKBACK + 2)
    if not candles or len(candles) < LSR_LOOKBACK + 2:
        return None
    last = candles[-1]
    try:
        o = float(last.get("open") or last.get("o") or 0.0)
        h = float(last.get("high") or last.get("h") or 0.0)
        l = float(last.get("low") or last.get("l") or 0.0)
        c = float(last.get("close") or last.get("c") or 0.0)
    except Exception:
        return None
    if h <= 0 or l <= 0:
        return None
    rng_pips = (h - l) / PIP
    if rng_pips < LSR_RANGE_MIN_PIPS:
        return None

    body_pips = abs(c - o) / PIP
    if body_pips > LSR_BODY_MAX_PIPS:
        return None
    upper_wick = (h - max(o, c)) / PIP
    lower_wick = (min(o, c) - l) / PIP
    wick_ratio = max(upper_wick, lower_wick) / max(rng_pips, 0.01)
    if wick_ratio < LSR_WICK_RATIO_MIN:
        return None

    history = candles[-(LSR_LOOKBACK + 1):-1]
    highs = []
    lows = []
    for candle in history:
        try:
            highs.append(float(candle.get("high") or candle.get("h") or 0.0))
            lows.append(float(candle.get("low") or candle.get("l") or 0.0))
        except Exception:
            continue
    if not highs or not lows:
        return None
    level_high = max(highs)
    level_low = min(lows)

    side = None
    sweep_dist = 0.0
    if h >= level_high + LSR_SWEEP_PIPS * PIP and c < level_high - LSR_RECLAIM_PIPS * PIP:
        side = "short"
        sweep_dist = (h - level_high) / PIP
        if upper_wick < max(LSR_SWEEP_PIPS, 0.4):
            return None
    elif l <= level_low - LSR_SWEEP_PIPS * PIP and c > level_low + LSR_RECLAIM_PIPS * PIP:
        side = "long"
        sweep_dist = (level_low - l) / PIP
        if lower_wick < max(LSR_SWEEP_PIPS, 0.4):
            return None
    else:
        return None

    mids, _ = tick_snapshot(6.0, limit=90)
    rev_ok, rev_dir, rev_strength = tick_reversal(mids, min_ticks=6) if mids else (False, None, 0.0)
    if LSR_REQUIRE_TICK_REVERSAL and (not rev_ok or rev_dir != side):
        hard_sweep_ok = sweep_dist >= (LSR_SWEEP_PIPS * 2.0) and wick_ratio >= (LSR_WICK_RATIO_MIN + 0.15)
        if not hard_sweep_ok:
            return None

    atr = _atr_pips(fac_m1)
    sl = max(1.4, min(2.6, atr * 0.8))
    tp = max(1.8, min(3.2, atr * 1.15))
    conf = 58 + int(min(14.0, sweep_dist * 3.0 + wick_ratio * 8.0))
    if rev_ok and rev_strength is not None:
        conf += int(min(6.0, max(0.0, rev_strength) * 6.0))
    conf = max(45, min(92, conf))

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": conf,
        "tag": tag,
        "reason": "liquidity_sweep",
        "size_mult": round(LSR_SIZE_MULT, 3),
    }


def _signal_wick_reversal(
    fac_m1: Dict[str, object],
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    candles = get_candles_snapshot("M1", limit=2)
    if not candles:
        return None
    last = candles[-1]
    try:
        o = float(last.get("open") or last.get("o") or 0.0)
        h = float(last.get("high") or last.get("h") or 0.0)
        l = float(last.get("low") or last.get("l") or 0.0)
        c = float(last.get("close") or last.get("c") or 0.0)
    except Exception:
        return None
    if h <= 0 or l <= 0:
        return None
    rng = (h - l) / PIP
    if rng < WICK_RANGE_MIN_PIPS:
        return None
    body = abs(c - o) / PIP
    if body > WICK_BODY_MAX_PIPS:
        return None
    upper_wick = (h - max(o, c)) / PIP
    lower_wick = (min(o, c) - l) / PIP
    wick_ratio = max(upper_wick, lower_wick) / max(rng, 0.01)
    if wick_ratio < WICK_RATIO_MIN:
        return None

    if _adx(fac_m1) > WICK_ADX_MAX or _bbw(fac_m1) > WICK_BBW_MAX:
        return None

    side = "short" if upper_wick > lower_wick else "long"
    atr = _atr_pips(fac_m1)
    sl = max(1.2, min(2.0, atr * 0.85))
    tp = max(1.5, min(2.5, atr * 1.1))
    conf = 58 + int(min(10, wick_ratio * 20.0))

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(45, min(92, conf))),
        "tag": tag,
        "reason": "wick_reversal",
    }

def _signal_wick_reversal_blend(
    fac_m1: Dict[str, object],
    range_ctx=None,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    """Band-touch + tick-confirmed wick fade with strict trend filters."""
    range_reason = str(getattr(range_ctx, "reason", "") or "").strip().lower() if range_ctx else ""
    if range_ctx is not None:
        try:
            range_active = bool(getattr(range_ctx, "active", False))
            range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
        except Exception:
            range_active = False
            range_score = 0.0
        if not range_active and WICK_BLEND_RANGE_SCORE_MIN > 0.0 and range_score < WICK_BLEND_RANGE_SCORE_MIN:
            return None
    elif WICK_BLEND_RANGE_SCORE_MIN > 0.0:
        return None

    ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=WICK_BLEND_SPREAD_P25)
    if not ok_spread:
        return None

    candles = get_candles_snapshot("M1", limit=2)
    if not candles:
        return None
    last = candles[-1]
    try:
        o = float(last.get("open") or last.get("o") or 0.0)
        h = float(last.get("high") or last.get("h") or 0.0)
        l = float(last.get("low") or last.get("l") or 0.0)
        c = float(last.get("close") or last.get("c") or 0.0)
    except Exception:
        return None
    if h <= 0 or l <= 0:
        return None

    rng = (h - l) / PIP
    if rng < WICK_BLEND_RANGE_MIN_PIPS:
        return None
    body = abs(c - o) / PIP
    if body > WICK_BLEND_BODY_MAX_PIPS:
        return None
    body_ratio = body / max(rng, 0.01)
    if body_ratio > WICK_BLEND_BODY_RATIO_MAX:
        return None

    upper_wick = (h - max(o, c)) / PIP
    lower_wick = (min(o, c) - l) / PIP
    wick_ratio = max(upper_wick, lower_wick) / max(rng, 0.01)
    if wick_ratio < WICK_BLEND_WICK_RATIO_MIN:
        return None

    side = "short" if upper_wick > lower_wick else "long"

    adx = _adx(fac_m1)
    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    if WICK_BLEND_ADX_MIN > 0.0 and adx < WICK_BLEND_ADX_MIN:
        return None
    if WICK_BLEND_ADX_MAX > 0.0 and adx > WICK_BLEND_ADX_MAX:
        return None
    if WICK_BLEND_BBW_MAX > 0.0 and bbw > WICK_BLEND_BBW_MAX:
        return None
    if (WICK_BLEND_ATR_MIN > 0.0 and atr < WICK_BLEND_ATR_MIN) or (
        WICK_BLEND_ATR_MAX > 0.0 and atr > WICK_BLEND_ATR_MAX
    ):
        return None

    levels = bb_levels(fac_m1)
    if not levels:
        return None
    upper, _, lower, _, span_pips = levels
    touch_pips = max(WICK_BLEND_BB_TOUCH_PIPS, span_pips * max(0.0, WICK_BLEND_BB_TOUCH_RATIO))
    if side == "short":
        if h < upper - touch_pips * PIP:
            return None
    else:
        if l > lower + touch_pips * PIP:
            return None

    strength = 0.0
    if WICK_BLEND_REQUIRE_TICK_REV:
        mids, _ = tick_snapshot(WICK_BLEND_TICK_WINDOW_SEC, limit=160)
        rev_ok, rev_dir, rev_strength = (
            tick_reversal(mids, min_ticks=WICK_BLEND_TICK_MIN_TICKS) if mids else (False, None, 0.0)
        )
        if not rev_ok or rev_dir != side:
            return None
        try:
            strength = float(rev_strength or 0.0)
        except Exception:
            strength = 0.0
        if WICK_BLEND_TICK_MIN_STRENGTH > 0.0 and strength < WICK_BLEND_TICK_MIN_STRENGTH:
            return None

    price = _latest_price(fac_m1)
    if price <= 0.0:
        return None
    follow = 0.0
    if WICK_BLEND_FOLLOW_PIPS > 0.0:
        if side == "short":
            follow = (upper - price) / PIP
            if follow < WICK_BLEND_FOLLOW_PIPS:
                return None
        else:
            follow = (price - lower) / PIP
            if follow < WICK_BLEND_FOLLOW_PIPS:
                return None
    retrace_from_extreme = 0.0
    if WICK_BLEND_EXTREME_RETRACE_MIN_PIPS > 0.0:
        if side == "short":
            retrace_from_extreme = (h - price) / PIP
        else:
            retrace_from_extreme = (price - l) / PIP
        if retrace_from_extreme < WICK_BLEND_EXTREME_RETRACE_MIN_PIPS:
            return None

    proj_allow, size_mult, proj_detail = projection_decision(side, mode="range")
    if not proj_allow:
        return None
    projection_score = 0.0
    if isinstance(proj_detail, dict):
        try:
            projection_score = float(proj_detail.get("score") or 0.0)
        except Exception:
            projection_score = 0.0
    rsi = _rsi(fac_m1)
    macd_hist_pips = _macd_hist_pips(fac_m1)
    di_gap = _plus_di(fac_m1) - _minus_di(fac_m1)
    ma_fast = fac_m1.get("ma10")
    if ma_fast is None:
        ma_fast = fac_m1.get("ema20")
    ma_slow = fac_m1.get("ma20")
    if ma_slow is None:
        ma_slow = fac_m1.get("ema24")
    if ma_slow is None:
        ma_slow = fac_m1.get("ema20")
    try:
        ma_gap_pips = ((float(ma_fast) - float(ma_slow)) / PIP) if ma_fast is not None and ma_slow is not None else None
    except Exception:
        ma_gap_pips = None
    gap_ratio = abs(ma_gap_pips) / max(1.0, atr) if ma_gap_pips is not None else None
    quality = wick_blend_entry_quality(
        side=side,
        rsi=rsi,
        adx=adx,
        atr_pips=atr,
        range_score=range_score if range_ctx is not None else 0.0,
        wick_ratio=wick_ratio,
        tick_strength=strength,
        follow_pips=follow if WICK_BLEND_FOLLOW_PIPS > 0.0 else 0.0,
        retrace_from_extreme_pips=retrace_from_extreme if WICK_BLEND_EXTREME_RETRACE_MIN_PIPS > 0.0 else 0.0,
        projection_score=projection_score,
        range_reason=getattr(range_ctx, "reason", None) if range_ctx is not None else None,
        macd_hist_pips=macd_hist_pips,
        di_gap=di_gap,
    )
    if not bool(quality.get("allow")):
        return None
    wick_quality = float(quality.get("quality") or 0.0)
    lean_gap_long_lane = (
        bool(getattr(config, "WICK_BLEND_LEAN_GAP_LONG_GUARD_ENABLED", True))
        and side == "long"
        and range_reason in {"volatility_compression", "adx_squeeze"}
        and gap_ratio is not None
        and gap_ratio >= float(getattr(config, "WICK_BLEND_LEAN_GAP_LONG_GAP_RATIO_MIN", 0.35))
        and gap_ratio < float(getattr(config, "WICK_BLEND_LEAN_GAP_LONG_GAP_RATIO_MAX", 1.20))
        and projection_score <= float(getattr(config, "WICK_BLEND_LEAN_GAP_LONG_PROJECTION_SCORE_MAX", 0.28))
        and wick_quality < float(getattr(config, "WICK_BLEND_LEAN_GAP_LONG_QUALITY_MAX", 0.70))
        and rsi <= float(getattr(config, "WICK_BLEND_LEAN_GAP_LONG_RSI_MAX", 54.0))
    )
    if lean_gap_long_lane:
        return None
    if _wick_blend_short_countertrend_blocked(
        range_reason=range_reason,
        side=side,
        projection_score=projection_score,
        wick_quality=wick_quality,
        rsi=rsi,
        adx=adx,
        macd_hist_pips=macd_hist_pips,
    ):
        return None
    setup_pressure = _wick_blend_long_setup_pressure(range_reason)
    if _wick_blend_long_pressure_blocked(
        range_reason=range_reason,
        side=side,
        setup_pressure=setup_pressure,
        bbw=bbw,
        range_score=range_score if range_ctx is not None else 0.0,
        rsi=rsi,
        wick_quality=wick_quality,
        projection_score=projection_score,
    ):
        return None

    sl = max(1.4, min(2.4, atr * 0.95))
    tp = max(sl * 1.30, min(3.8, atr * 1.20))
    conf = 64
    conf += int(min(12.0, wick_ratio * 22.0))
    conf += int(min(6.0, max(0.0, rng - 1.0) * 3.0))
    if strength > 0.0:
        conf += int(min(10.0, strength * 8.0))
    conf = int(max(45, min(92, conf)))

    global _LAST_WICK_BLEND_DIAG_TS
    if WICK_BLEND_DIAG and time.monotonic() - _LAST_WICK_BLEND_DIAG_TS >= WICK_BLEND_DIAG_INTERVAL_SEC:
        _LAST_WICK_BLEND_DIAG_TS = time.monotonic()
        LOG.info(
            "%s wick_blend signal side=%s rng=%s body=%s ratio=%s adx=%s bbw=%s atr=%s strength=%s touch_pips=%s",
            config.LOG_PREFIX,
            side,
            round(rng, 2),
            round(body, 2),
            round(wick_ratio, 3),
            round(adx, 2),
            round(bbw, 6),
            round(atr, 2),
            round(strength, 3),
            round(touch_pips, 2),
        )

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": conf,
        "tag": tag,
        "reason": "wick_reversal_blend",
        "size_mult": round(size_mult, 3),
        "projection": proj_detail,
        "wick_blend_quality": quality.get("quality"),
        "wick_blend_components": quality.get("components"),
        "setup_pressure": setup_pressure or None,
        "wick": {
            "rng_pips": round(rng, 2),
            "body_pips": round(body, 2),
            "upper_wick_pips": round(upper_wick, 2),
            "lower_wick_pips": round(lower_wick, 2),
            "ratio": round(wick_ratio, 3),
            "tick_strength": round(strength, 3),
            "follow_pips": round(follow if WICK_BLEND_FOLLOW_PIPS > 0.0 else 0.0, 3),
            "retrace_from_extreme_pips": round(
                retrace_from_extreme if WICK_BLEND_EXTREME_RETRACE_MIN_PIPS > 0.0 else 0.0,
                3,
            ),
        },
    }


def _signal_wick_reversal_hf(
    fac_m1: Dict[str, object],
    range_ctx=None,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    """Higher-frequency WickReversal with tick confirmation (designed for better precision than TickWickReversal)."""
    if range_ctx is not None:
        try:
            range_active = bool(getattr(range_ctx, "active", False))
            range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
        except Exception:
            range_active = False
            range_score = 0.0
        if not range_active and WICK_HF_RANGE_SCORE_MIN > 0.0 and range_score < WICK_HF_RANGE_SCORE_MIN:
            return None

    ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=WICK_HF_SPREAD_P25)
    if not ok_spread:
        return None

    candles = get_candles_snapshot("M1", limit=2)
    if not candles:
        return None
    last = candles[-1]
    try:
        o = float(last.get("open") or last.get("o") or 0.0)
        h = float(last.get("high") or last.get("h") or 0.0)
        l = float(last.get("low") or last.get("l") or 0.0)
        c = float(last.get("close") or last.get("c") or 0.0)
    except Exception:
        return None
    if h <= 0 or l <= 0:
        return None

    rng = (h - l) / PIP
    if rng < WICK_HF_RANGE_MIN_PIPS:
        return None
    body = abs(c - o) / PIP
    if body > WICK_HF_BODY_MAX_PIPS:
        return None
    body_ratio = body / max(rng, 0.01)
    if WICK_HF_BODY_RATIO_MAX > 0.0 and body_ratio > WICK_HF_BODY_RATIO_MAX:
        return None

    upper_wick = (h - max(o, c)) / PIP
    lower_wick = (min(o, c) - l) / PIP
    wick_ratio = max(upper_wick, lower_wick) / max(rng, 0.01)
    if wick_ratio < WICK_HF_RATIO_MIN:
        return None

    adx = _adx(fac_m1)
    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    if WICK_HF_BBW_MAX > 0.0 and bbw > WICK_HF_BBW_MAX:
        return None
    if (WICK_HF_ATR_MIN > 0.0 and atr < WICK_HF_ATR_MIN) or (WICK_HF_ATR_MAX > 0.0 and atr > WICK_HF_ATR_MAX):
        return None
    if WICK_HF_ADX_MAX > 0.0 and adx > WICK_HF_ADX_MAX and not WICK_HF_ADX_OVERRIDE_ENABLED:
        return None

    side = "short" if upper_wick > lower_wick else "long"

    if WICK_HF_MOMENTUM_FILTER_ENABLED:
        macd_hist = _macd_hist_pips(fac_m1)
        if side == "long" and macd_hist < WICK_HF_MACD_HIST_LONG_MIN:
            return None
        if side == "short":
            if macd_hist > WICK_HF_MACD_HIST_SHORT_MAX:
                return None
            slope_10 = _ema_slope_pips(fac_m1, "ema_slope_10")
            if slope_10 > WICK_HF_EMA_SLOPE_10_SHORT_MAX:
                return None

    if WICK_HF_REQUIRE_BB_TOUCH:
        levels = bb_levels(fac_m1)
        if not levels:
            return None
        upper, _, lower, _, span_pips = levels
        touch_pips = max(WICK_HF_BB_TOUCH_PIPS, span_pips * 0.18)
        if side == "short":
            if h < upper - touch_pips * PIP:
                return None
        else:
            if l > lower + touch_pips * PIP:
                return None

    strength = 0.0
    tick_ok = not WICK_HF_REQUIRE_TICK_REV
    if WICK_HF_REQUIRE_TICK_REV or WICK_HF_ADX_OVERRIDE_ENABLED:
        mids, _ = tick_snapshot(WICK_HF_TICK_WINDOW_SEC, limit=160)
        rev_ok, rev_dir, rev_strength = (
            tick_reversal(mids, min_ticks=WICK_HF_TICK_MIN_TICKS) if mids else (False, None, 0.0)
        )
        # Direction mismatch stays blocked even if hard-wick fallback is enabled.
        if rev_ok and rev_dir != side:
            return None
        try:
            strength = float(rev_strength or 0.0)
        except Exception:
            strength = 0.0
        tick_ok = bool(rev_ok) and (rev_dir == side) and (
            WICK_HF_TICK_MIN_STRENGTH <= 0.0 or strength >= WICK_HF_TICK_MIN_STRENGTH
        )
        if WICK_HF_REQUIRE_TICK_REV and not tick_ok:
            if not WICK_HF_TICK_FALLBACK_HARD_WICK:
                return None
            hard_wick_ok = (
                wick_ratio >= (WICK_HF_RATIO_MIN + max(0.0, WICK_HF_HARD_WICK_EXTRA_RATIO))
                and rng >= (WICK_HF_RANGE_MIN_PIPS * max(1.0, WICK_HF_HARD_WICK_RANGE_MULT))
            )
            if not hard_wick_ok:
                return None

    if WICK_HF_ADX_MAX > 0.0 and adx > WICK_HF_ADX_MAX:
        # In high-ADX regimes, require exceptionally strong rejections.
        if adx > WICK_HF_ADX_OVERRIDE_MAX_ADX:
            return None
        if wick_ratio < WICK_HF_ADX_OVERRIDE_MIN_RATIO:
            return None
        if tick_ok:
            if strength < WICK_HF_ADX_OVERRIDE_MIN_TICK_STRENGTH:
                return None
        else:
            if not WICK_HF_ADX_OVERRIDE_ALLOW_NO_TICK_REV:
                return None
            if wick_ratio < WICK_HF_ADX_OVERRIDE_NO_TICK_MIN_RATIO:
                return None

    proj_allow, size_mult, proj_detail = projection_decision(side, mode="range")
    if not proj_allow:
        return None

    sl = max(1.2, min(2.2, atr * 0.85))
    tp = max(sl * 1.35, min(3.6, atr * 1.15))
    conf = 62
    conf += int(min(12.0, wick_ratio * 22.0))
    conf += int(min(6.0, max(0.0, rng - 1.0) * 3.0))
    if strength > 0.0:
        conf += int(min(8.0, strength * 8.0))
    conf = int(max(45, min(92, conf)))

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": conf,
        "tag": tag,
        "reason": "wick_reversal_hf",
        "size_mult": round(size_mult, 3),
        "projection": proj_detail,
        "wick": {
            "rng_pips": round(rng, 2),
            "body_pips": round(body, 2),
            "upper_wick_pips": round(upper_wick, 2),
            "lower_wick_pips": round(lower_wick, 2),
            "ratio": round(wick_ratio, 3),
            "tick_strength": round(strength, 3),
        },
    }

def _signal_wick_reversal_pro(
    fac_m1: Dict[str, object],
    range_ctx=None,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    # Intentionally strict gates to improve precision and reduce "timeout/market-close" losers.
    if range_ctx is not None:
        try:
            range_active = bool(getattr(range_ctx, "active", False))
            range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
        except Exception:
            range_active = False
            range_score = 0.0
        if not range_active:
            return None
        if WICK_PRO_RANGE_SCORE_MIN > 0.0 and range_score < WICK_PRO_RANGE_SCORE_MIN:
            return None

    ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=WICK_PRO_SPREAD_P25)
    if not ok_spread:
        return None

    candles = get_candles_snapshot("M1", limit=2)
    if not candles:
        return None
    last = candles[-1]
    try:
        o = float(last.get("open") or last.get("o") or 0.0)
        h = float(last.get("high") or last.get("h") or 0.0)
        l = float(last.get("low") or last.get("l") or 0.0)
        c = float(last.get("close") or last.get("c") or 0.0)
    except Exception:
        return None
    if h <= 0 or l <= 0:
        return None

    rng = (h - l) / PIP
    if rng < WICK_PRO_RANGE_MIN_PIPS:
        return None
    body = abs(c - o) / PIP
    if body > WICK_PRO_BODY_MAX_PIPS:
        return None
    upper_wick = (h - max(o, c)) / PIP
    lower_wick = (min(o, c) - l) / PIP
    wick_ratio = max(upper_wick, lower_wick) / max(rng, 0.01)
    if wick_ratio < WICK_PRO_RATIO_MIN:
        return None

    adx = _adx(fac_m1)
    bbw = _bbw(fac_m1)
    if WICK_PRO_ADX_MAX > 0.0 and adx > WICK_PRO_ADX_MAX:
        return None
    if WICK_PRO_BBW_MAX > 0.0 and bbw > WICK_PRO_BBW_MAX:
        return None

    atr = _atr_pips(fac_m1)
    if (WICK_PRO_ATR_MIN > 0.0 and atr < WICK_PRO_ATR_MIN) or (
        WICK_PRO_ATR_MAX > 0.0 and atr > WICK_PRO_ATR_MAX
    ):
        return None

    side = "short" if upper_wick > lower_wick else "long"

    if WICK_PRO_REQUIRE_BB_TOUCH:
        levels = bb_levels(fac_m1)
        if not levels:
            return None
        upper, _, lower, _, span_pips = levels
        touch_pips = max(WICK_PRO_BB_TOUCH_PIPS, span_pips * 0.18)
        if side == "short":
            # Require the rejection candle to probe the upper band.
            if h < upper - touch_pips * PIP:
                return None
        else:
            if l > lower + touch_pips * PIP:
                return None

    if WICK_PRO_REQUIRE_TICK_REV:
        mids, _ = tick_snapshot(WICK_PRO_TICK_WINDOW_SEC, limit=90)
        rev_ok, rev_dir, rev_strength = (
            tick_reversal(mids, min_ticks=WICK_PRO_TICK_MIN_TICKS) if mids else (False, None, 0.0)
        )
        if not rev_ok or rev_dir != side:
            return None
        try:
            strength = float(rev_strength or 0.0)
        except Exception:
            strength = 0.0
        if WICK_PRO_TICK_MIN_STRENGTH > 0.0 and strength < WICK_PRO_TICK_MIN_STRENGTH:
            return None
    else:
        strength = 0.0

    sl = max(1.1, min(1.9, atr * 0.8))
    tp = max(1.4, min(2.3, atr * 1.0))
    conf = 62 + int(min(12.0, wick_ratio * 22.0))
    if strength > 0.0:
        conf += int(min(8.0, strength * 8.0))

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(45, min(92, conf))),
        "tag": tag,
        "reason": "wick_reversal_pro",
        "wick": {
            "rng_pips": round(rng, 2),
            "body_pips": round(body, 2),
            "upper_wick_pips": round(upper_wick, 2),
            "lower_wick_pips": round(lower_wick, 2),
            "ratio": round(wick_ratio, 3),
            "tick_strength": round(strength, 3),
        },
    }


def _signal_tick_wick_reversal(
    fac_m1: Dict[str, object],
    range_ctx=None,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    # Higher-frequency wick reversal built from a short tick window (no M1 close dependency).
    range_score = 0.0
    range_ok = False
    if range_ctx is not None:
        try:
            range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
            range_ok = bool(getattr(range_ctx, "active", False)) or range_score >= TICK_WICK_RANGE_SCORE_MIN
        except Exception:
            range_score = 0.0
            range_ok = False
    if not range_ok:
        return None

    ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=TICK_WICK_SPREAD_P25)
    if not ok_spread:
        return None

    if _adx(fac_m1) > TICK_WICK_ADX_MAX or _bbw(fac_m1) > TICK_WICK_BBW_MAX:
        return None

    mids, span = tick_snapshot(TICK_WICK_WINDOW_SEC, limit=max(60, int(TICK_WICK_MIN_TICKS * 5)))
    if not mids or len(mids) < TICK_WICK_MIN_TICKS or span <= 0.0:
        return None

    o = float(mids[0])
    c = float(mids[-1])
    h = float(max(mids))
    l = float(min(mids))
    if h <= 0 or l <= 0 or h <= l:
        return None

    rng_pips = (h - l) / PIP
    if rng_pips < TICK_WICK_RANGE_MIN_PIPS:
        return None
    body_pips = abs(c - o) / PIP
    if body_pips > TICK_WICK_BODY_MAX_PIPS:
        return None

    upper_wick = (h - max(o, c)) / PIP
    lower_wick = (min(o, c) - l) / PIP
    wick_ratio = max(upper_wick, lower_wick) / max(rng_pips, 0.01)
    if wick_ratio < TICK_WICK_RATIO_MIN:
        return None

    side = "short" if upper_wick > lower_wick else "long"

    if TICK_WICK_REQUIRE_BB_TOUCH:
        levels = bb_levels(fac_m1)
        if not levels:
            return None
        upper, _, lower, _, span_pips = levels
        band = max(TICK_WICK_BB_TOUCH_PIPS, span_pips * 0.18)
        # Use tick-window extremes (h/l) rather than the latest price. Wick patterns
        # are defined by the probe + rejection, so the extreme should be near the band.
        if side == "long" and l > lower + band * PIP:
            return None
        if side == "short" and h < upper - band * PIP:
            return None

    rev_ok, rev_dir, rev_strength = tick_reversal(mids, min_ticks=max(6, int(TICK_WICK_MIN_TICKS * 0.35)))
    if TICK_WICK_REQUIRE_TICK_REV:
        if not rev_ok or rev_dir != side or (rev_strength is not None and rev_strength < TICK_WICK_MIN_REV_STRENGTH):
            hard_wick_ok = wick_ratio >= (TICK_WICK_RATIO_MIN + 0.12) and rng_pips >= (TICK_WICK_RANGE_MIN_PIPS * 1.35)
            if not hard_wick_ok:
                return None

    proj_allow, size_mult, proj_detail = projection_decision(side, mode="range")
    if not proj_allow:
        return None

    atr = _atr_pips(fac_m1)
    sl = max(1.1, min(1.9, atr * 0.75))
    tp = max(1.2, min(2.2, atr * 0.95))
    conf = 60
    conf += int(min(12.0, wick_ratio * 18.0))
    conf += int(min(8.0, max(0.0, rng_pips - 0.6) * 5.0))
    if rev_ok and rev_strength is not None:
        conf += int(min(6.0, max(0.0, rev_strength) * 4.0))
    conf += int(min(8.0, range_score * 12.0))
    conf = int(max(45, min(92, conf)))

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": conf,
        "tag": tag,
        "reason": "tick_wick_reversal",
        "size_mult": round(size_mult, 3),
        "projection": proj_detail,
        "tick_span_sec": round(span, 3),
        "tick_wick": {
            "range_pips": round(rng_pips, 3),
            "body_pips": round(body_pips, 3),
            "wick_ratio": round(wick_ratio, 3),
            "upper_wick": round(upper_wick, 3),
            "lower_wick": round(lower_wick, 3),
        },
        "range_score": round(range_score, 3),
    }


def _signal_vwap_revert(
    fac_m1: Dict[str, object],
    range_ctx,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    levels = bb_levels(fac_m1)
    if not levels:
        return None
    upper, _, lower, _, span_pips = levels

    range_score = float(range_ctx.score or 0.0) if range_ctx else 0.0
    range_ok = bool(range_ctx and (range_ctx.active or range_score >= VWAP_REV_RANGE_SCORE))
    if not range_ok:
        return None

    ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=VWAP_REV_SPREAD_P25)
    if not ok_spread:
        return None

    adx = _adx(fac_m1)
    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    if adx > VWAP_REV_ADX_MAX or bbw > VWAP_REV_BBW_MAX:
        return None
    if atr < VWAP_REV_ATR_MIN or atr > VWAP_REV_ATR_MAX:
        return None

    vgap = _vwap_gap_pips(fac_m1)
    if abs(vgap) < VWAP_REV_GAP_MIN:
        return None

    rsi = _rsi(fac_m1)
    stoch = _stoch_rsi(fac_m1)
    dist_lower = (price - lower) / PIP
    dist_upper = (upper - price) / PIP

    mids, _ = tick_snapshot(6.0, limit=80)
    rev_ok, rev_dir, rev_strength = tick_reversal(mids, min_ticks=6)
    if not rev_ok:
        return None

    side = None
    if vgap <= -VWAP_REV_GAP_MIN and dist_lower <= max(VWAP_REV_BB_TOUCH_PIPS, span_pips * 0.18):
        if rsi <= VWAP_REV_RSI_LONG_MAX and stoch <= VWAP_REV_STOCH_LONG_MAX and rev_dir == "long":
            side = "long"
    elif vgap >= VWAP_REV_GAP_MIN and dist_upper <= max(VWAP_REV_BB_TOUCH_PIPS, span_pips * 0.18):
        if rsi >= VWAP_REV_RSI_SHORT_MIN and stoch >= VWAP_REV_STOCH_SHORT_MIN and rev_dir == "short":
            side = "short"
    if not side:
        return None

    flow_guard = None
    if side == "short":
        short_ok, flow_guard = _reversion_short_flow_guard(
            fac_m1=fac_m1,
            price=price,
            dist_upper_pips=dist_upper,
            band_pips=max(VWAP_REV_BB_TOUCH_PIPS, span_pips * 0.18),
            range_score=range_score,
            rev_strength=rev_strength,
            profile=tag,
        )
        if not short_ok:
            return None

    proj_allow, size_mult, proj_detail = projection_decision(side, mode="range")
    if not proj_allow:
        return None
    proj_size_mult = float(size_mult)
    projection_score = 0.0
    ma_fast = fac_m1.get("ma10")
    if ma_fast is None:
        ma_fast = fac_m1.get("ema20")
    ma_slow = fac_m1.get("ma20")
    if ma_slow is None:
        ma_slow = fac_m1.get("ema24")
    if ma_slow is None:
        ma_slow = fac_m1.get("ema20")
    try:
        ma_gap_pips = ((float(ma_fast) - float(ma_slow)) / PIP) if ma_fast is not None and ma_slow is not None else None
    except Exception:
        ma_gap_pips = None
    ma_gap_atr_ratio = abs(ma_gap_pips) / max(1.0, atr) if ma_gap_pips is not None else None
    short_up_lean = bool(
        side == "short"
        and ma_gap_pips is not None
        and ma_gap_pips > 0.0
        and ma_gap_atr_ratio is not None
        and 0.30 <= ma_gap_atr_ratio < 0.90
    )
    short_down_flat = bool(
        side == "short"
        and ma_gap_pips is not None
        and ma_gap_pips < 0.0
        and ma_gap_atr_ratio is not None
        and ma_gap_atr_ratio < 0.35
    )
    if isinstance(proj_detail, dict):
        try:
            projection_score = float(proj_detail.get("score") or 0.0)
        except Exception:
            projection_score = 0.0
    if side == "short" and flow_guard is not None:
        gap_atr_ratio = abs(vgap) / max(1.0, atr)
        setup_quality = float(flow_guard.get("setup_quality") or 0.0)
        if (
            projection_score <= -0.10
            and gap_atr_ratio >= 6.5
            and range_score >= 0.30
            and rsi < max(VWAP_REV_RSI_SHORT_MIN + 6.0, 60.0)
            and rev_strength < 0.90
            and setup_quality < 0.58
        ):
            return None
        if (
            short_down_flat
            and projection_score <= 0.0
            and rev_strength < 0.88
            and setup_quality < 0.55
        ):
            return None

    sl = max(1.2, min(1.8, atr * 0.7))
    tp = max(1.4, min(2.4, atr * (0.95 + (0.08 if short_up_lean else 0.0))))
    conf = 60 + int(min(10, abs(vgap) * 1.6)) + int(min(8, rev_strength * 3.5))
    if short_up_lean:
        conf += 4
    if short_down_flat:
        conf -= 4
    if flow_guard is not None:
        conf -= int(min(5.0, max(0.0, 0.62 - float(flow_guard["setup_quality"])) * 18.0))
        size_mult = max(
            0.78,
            min(1.1, min(size_mult, proj_size_mult * (0.86 + float(flow_guard["setup_quality"]) * 0.14))),
        )
    if short_up_lean:
        size_mult = max(size_mult, min(1.18, proj_size_mult * 1.06))
    if short_down_flat:
        size_mult = max(0.76, min(size_mult, 0.92))
    signal = {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(45, min(92, conf))),
        "tag": tag,
        "reason": "vwap_revert",
        "size_mult": round(size_mult, 3),
        "projection": proj_detail,
    }
    _attach_flow_guard_context(signal, flow_guard)
    return signal


def _signal_stoch_bounce(
    fac_m1: Dict[str, object],
    range_ctx,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    levels = bb_levels(fac_m1)
    if not levels:
        return None
    upper, _, lower, _, span_pips = levels

    range_score = float(range_ctx.score or 0.0) if range_ctx else 0.0
    range_ok = bool(range_ctx and (range_ctx.active or range_score >= STOCH_BOUNCE_RANGE_SCORE))
    if not range_ok:
        return None

    adx = _adx(fac_m1)
    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    if adx > STOCH_BOUNCE_ADX_MAX or bbw > STOCH_BOUNCE_BBW_MAX:
        return None
    if atr < STOCH_BOUNCE_ATR_MIN or atr > STOCH_BOUNCE_ATR_MAX:
        return None

    ok_spread, _ = spread_ok(max_pips=config.MAX_SPREAD_PIPS, p25_max=VWAP_REV_SPREAD_P25)
    if not ok_spread:
        return None

    stoch = _stoch_rsi(fac_m1)
    rsi = _rsi(fac_m1)
    macd_abs = abs(_macd_hist_pips(fac_m1))
    if macd_abs > STOCH_BOUNCE_MACD_ABS_MAX:
        return None

    mids, _ = tick_snapshot(6.0, limit=80)
    rev_ok, rev_dir, rev_strength = tick_reversal(mids, min_ticks=6)
    if not rev_ok:
        return None

    dist_lower = (price - lower) / PIP
    dist_upper = (upper - price) / PIP
    side = None
    if stoch <= STOCH_BOUNCE_STOCH_LONG_MAX and dist_lower <= max(STOCH_BOUNCE_BB_TOUCH_PIPS, span_pips * 0.18):
        if rsi <= STOCH_BOUNCE_RSI_LONG_MAX and rev_dir == "long":
            side = "long"
    elif stoch >= STOCH_BOUNCE_STOCH_SHORT_MIN and dist_upper <= max(STOCH_BOUNCE_BB_TOUCH_PIPS, span_pips * 0.18):
        if rsi >= STOCH_BOUNCE_RSI_SHORT_MIN and rev_dir == "short":
            side = "short"
    if not side:
        return None

    proj_allow, size_mult, proj_detail = projection_decision(side, mode="range")
    if not proj_allow:
        return None

    sl = max(1.2, min(1.9, atr * 0.8))
    tp = max(1.4, min(2.2, atr * 1.0))
    conf = 58 + int(min(12, abs(stoch - 0.5) * 28.0)) + int(min(6, rev_strength * 3.0))
    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(45, min(92, conf))),
        "tag": tag,
        "reason": "stoch_bounce",
        "size_mult": round(size_mult, 3),
        "projection": proj_detail,
    }


def _signal_macd_trend(
    fac_m1: Dict[str, object],
    fac_m5: Dict[str, object],
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None

    adx = _adx(fac_m1)
    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    if adx < MACD_TREND_ADX_MIN or bbw < MACD_TREND_BBW_MIN or atr < MACD_TREND_ATR_MIN:
        return None

    macd_hist = _macd_hist_pips(fac_m1)
    if abs(macd_hist) < MACD_TREND_HIST_MIN:
        return None

    slope_10 = _ema_slope_pips(fac_m1, "ema_slope_10")
    slope_20 = _ema_slope_pips(fac_m1, "ema_slope_20")
    if abs(slope_10) < MACD_TREND_SLOPE_MIN or abs(slope_20) < MACD_TREND_SLOPE_MIN:
        return None

    direction = None
    if macd_hist > 0 and slope_10 > 0 and slope_20 > 0:
        direction = "long"
    elif macd_hist < 0 and slope_10 < 0 and slope_20 < 0:
        direction = "short"
    if not direction:
        return None

    m5_slope = _ema_slope_pips(fac_m5, "ema_slope_20")
    if direction == "long" and m5_slope < MACD_TREND_M5_SLOPE_MIN:
        return None
    if direction == "short" and m5_slope > -MACD_TREND_M5_SLOPE_MIN:
        return None

    vgap = _vwap_gap_pips(fac_m1)
    if abs(vgap) > MACD_TREND_VWAP_GAP_MAX:
        return None

    rsi = _rsi(fac_m1)
    if direction == "long" and rsi < MACD_TREND_RSI_LONG_MIN:
        return None
    if direction == "short" and rsi > MACD_TREND_RSI_SHORT_MAX:
        return None

    proj_allow, size_mult, proj_detail = projection_decision(direction, mode="trend")
    if not proj_allow:
        return None

    sl = max(1.6, min(2.3, atr * 0.95))
    tp = max(2.2, min(3.2, atr * 1.35))
    conf = 64 + int(min(12, abs(macd_hist) * 2.2)) + int(min(8, abs(slope_10) * 10.0))

    return {
        "action": "OPEN_LONG" if direction == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(48, min(92, conf))),
        "tag": tag,
        "reason": "macd_trend",
        "size_mult": round(size_mult, 3),
        "projection": proj_detail,
    }


def _signal_divergence_revert(
    fac_m1: Dict[str, object],
    range_ctx,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    levels = bb_levels(fac_m1)
    if not levels:
        return None
    upper, _, lower, _, span_pips = levels

    adx = _adx(fac_m1)
    if adx > DIV_REVERT_ADX_MAX:
        return None

    range_ok = bool(range_ctx and range_ctx.active)
    if not range_ok:
        return None

    div_score = _div_score(fac_m1)
    try:
        rsi_strength = float(fac_m1.get("div_rsi_strength") or 0.0)
    except Exception:
        rsi_strength = 0.0
    try:
        macd_strength = float(fac_m1.get("div_macd_strength") or 0.0)
    except Exception:
        macd_strength = 0.0
    strength = max(rsi_strength, macd_strength)

    if abs(div_score) < DIV_REVERT_SCORE_MIN or strength < DIV_REVERT_MIN_STRENGTH:
        return None

    try:
        rsi_age = int(fac_m1.get("div_rsi_age") or 0)
    except Exception:
        rsi_age = 0
    try:
        macd_age = int(fac_m1.get("div_macd_age") or 0)
    except Exception:
        macd_age = 0
    age = min([a for a in (rsi_age, macd_age) if a > 0] or [0])
    if age > DIV_REVERT_MAX_AGE:
        return None

    rsi = _rsi(fac_m1)
    dist_lower = (price - lower) / PIP
    dist_upper = (upper - price) / PIP

    mids, _ = tick_snapshot(6.0, limit=80)
    rev_ok, rev_dir, rev_strength = tick_reversal(mids, min_ticks=6)
    if not rev_ok:
        return None

    side = None
    if div_score > 0 and dist_lower <= max(DIV_REVERT_BB_TOUCH_PIPS, span_pips * 0.18):
        if rsi <= DIV_REVERT_RSI_LONG_MAX and rev_dir == "long":
            side = "long"
    elif div_score < 0 and dist_upper <= max(DIV_REVERT_BB_TOUCH_PIPS, span_pips * 0.18):
        if rsi >= DIV_REVERT_RSI_SHORT_MIN and rev_dir == "short":
            side = "short"
    if not side:
        return None

    proj_allow, size_mult, proj_detail = projection_decision(side, mode="range")
    if not proj_allow:
        return None

    atr = _atr_pips(fac_m1)
    sl = max(1.2, min(1.9, atr * 0.8))
    tp = max(1.4, min(2.2, atr * 1.0))
    conf = 60 + int(min(14, abs(div_score) * 40.0)) + int(min(6, rev_strength * 3.0))

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(45, min(92, conf))),
        "tag": tag,
        "reason": "divergence_revert",
        "size_mult": round(size_mult, 3),
        "projection": proj_detail,
    }


def _signal_ema_slope_pull(
    fac_m1: Dict[str, object],
    fac_m5: Dict[str, object],
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    ema12 = float(fac_m1.get("ema12") or 0.0)
    ema20 = float(fac_m1.get("ema20") or fac_m1.get("ma20") or 0.0)
    if ema12 <= 0 or ema20 <= 0:
        return None

    slope_10 = _ema_slope_pips(fac_m1, "ema_slope_10")
    slope_20 = _ema_slope_pips(fac_m1, "ema_slope_20")
    if abs(slope_10) < EMA_PULL_SLOPE_MIN or abs(slope_20) < EMA_PULL_SLOPE_MIN:
        return None

    direction = None
    if ema12 > ema20 and slope_10 > 0 and slope_20 > 0:
        direction = "long"
    elif ema12 < ema20 and slope_10 < 0 and slope_20 < 0:
        direction = "short"
    if not direction:
        return None

    adx = _adx(fac_m1)
    if adx < EMA_PULL_ADX_MIN:
        return None

    vgap = _vwap_gap_pips(fac_m1)
    if abs(vgap) > EMA_PULL_VWAP_GAP_MAX:
        return None

    macd_abs = abs(_macd_hist_pips(fac_m1))
    if macd_abs > EMA_PULL_MACD_ABS_MAX:
        return None

    stoch = _stoch_rsi(fac_m1)
    if stoch < EMA_PULL_STOCH_MIN or stoch > EMA_PULL_STOCH_MAX:
        return None

    dist = abs(price - ema20) / PIP
    if dist > EMA_PULL_BAND_PIPS:
        return None

    rsi = _rsi(fac_m1)
    if direction == "long" and not (EMA_PULL_RSI_LONG_MIN <= rsi <= EMA_PULL_RSI_LONG_MAX):
        return None
    if direction == "short" and not (EMA_PULL_RSI_SHORT_MIN <= rsi <= EMA_PULL_RSI_SHORT_MAX):
        return None

    m5_slope = _ema_slope_pips(fac_m5, "ema_slope_20")
    if direction == "long" and m5_slope < 0:
        return None
    if direction == "short" and m5_slope > 0:
        return None

    proj_allow, size_mult, proj_detail = projection_decision(direction, mode="pullback")
    if not proj_allow:
        return None

    atr = _atr_pips(fac_m1)
    sl = max(1.4, min(2.1, atr * 0.9))
    tp = max(1.8, min(2.6, atr * 1.15))
    conf = 62 + int(min(10, abs(slope_10) * 10.0))

    return {
        "action": "OPEN_LONG" if direction == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(45, min(92, conf))),
        "tag": tag,
        "reason": "ema_slope_pull",
        "size_mult": round(size_mult, 3),
        "projection": proj_detail,
    }


def _signal_session_edge(
    fac_m1: Dict[str, object],
    range_ctx,
    *,
    tag: str,
    now_utc: datetime.datetime,
) -> Optional[Dict[str, object]]:
    if not session_allowed(now_utc.hour, allow_hours=SESSION_ALLOW_HOURS, block_hours=SESSION_BLOCK_HOURS):
        return None
    return _signal_spread_revert(fac_m1, range_ctx, tag=tag)


def _build_entry_thesis(signal: Dict[str, object], fac_m1: Dict[str, object], range_ctx) -> Dict[str, object]:
    signal_confidence = int(signal.get("confidence", 0) or 0)
    thesis = {
        "strategy_tag": signal.get("tag"),
        "env_prefix": config.ENV_PREFIX,
        "confidence": signal_confidence,
        "entry_probability": round(_to_probability(signal_confidence), 3),
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
        "bbw": _bbw(fac_m1),
        "stoch_rsi": _stoch_rsi(fac_m1),
        "macd_hist_pips": _macd_hist_pips(fac_m1),
        "vwap_gap": _vwap_gap_pips(fac_m1),
        "ema_slope_10_pips": _ema_slope_pips(fac_m1, "ema_slope_10"),
        "plus_di": _plus_di(fac_m1),
        "minus_di": _minus_di(fac_m1),
        "div_score": _div_score(fac_m1),
        "air_score": signal.get("air_score"),
        "air_pressure": signal.get("air_pressure"),
        "air_pressure_dir": signal.get("air_pressure_dir"),
        "air_spread_state": signal.get("air_spread_state"),
        "air_exec_quality": signal.get("air_exec_quality"),
        "air_regime_shift": signal.get("air_regime_shift"),
        "air_range_pref": signal.get("air_range_pref"),
        "projection": signal.get("projection"),
        "wick": signal.get("wick"),
        "wick_blend_quality": signal.get("wick_blend_quality"),
        "wick_blend_components": signal.get("wick_blend_components"),
        "flow_guard": signal.get("flow_guard"),
        "setup_pressure": signal.get("setup_pressure"),
    }
    flow_guard = signal.get("flow_guard")
    if isinstance(flow_guard, dict):
        thesis["continuation_pressure"] = flow_guard.get("continuation_pressure")
        thesis["reversion_support"] = flow_guard.get("reversion_support")
        thesis["setup_quality"] = flow_guard.get("setup_quality")
        for key in (
            "macro_flow_regime",
            "mtf_alignment",
            "mtf_countertrend_pressure",
            "mtf_aligned_support",
            "macro_trend_strength",
            "m5_flow_regime",
            "h1_flow_regime",
            "h4_flow_regime",
        ):
            value = flow_guard.get(key)
            if value not in {None, ""}:
                thesis[key] = value
        continuation_pressure = float(flow_guard.get("continuation_pressure") or 0.0)
        thesis["flow_headwind_regime"] = (
            "continuation_headwind" if continuation_pressure >= 0.6 else "range_fade"
        )
    for key in (
        "continuation_pressure",
        "reversion_support",
        "setup_quality",
        "flow_regime",
        "flow_headwind_regime",
        "macro_flow_regime",
        "mtf_alignment",
        "mtf_countertrend_pressure",
        "mtf_aligned_support",
        "macro_trend_strength",
        "m5_flow_regime",
        "h1_flow_regime",
        "h4_flow_regime",
    ):
        value = signal.get(key)
        if value is not None:
            thesis[key] = value
    if "flow_headwind_regime" not in thesis and thesis.get("continuation_pressure") is not None:
        continuation_pressure = float(thesis.get("continuation_pressure") or 0.0)
        thesis["flow_headwind_regime"] = (
            "continuation_headwind" if continuation_pressure >= 0.6 else "range_fade"
        )
    tag = str(signal.get("tag") or "").strip()
    if tag in {"TickImbalance", "TickImbalanceRRPlus"}:
        thesis["pattern_gate_opt_in"] = bool(TICK_IMB_PATTERN_GATE_OPT_IN)
        if TICK_IMB_PATTERN_GATE_ALLOW_GENERIC:
            thesis["pattern_gate_allow_generic"] = True
    return thesis


async def _place_order(
    signal: Dict[str, object],
    *,
    fac_m1: Dict[str, object],
    fac_h4: Dict[str, object],
    range_ctx,
    now: datetime.datetime,
) -> Optional[str]:
    global _LAST_TICK_WICK_PLACE_DIAG_TS
    diag_tick_wick = (
        TICK_WICK_DIAG and str(signal.get("tag") or "").strip() == "TickWickReversal"
    )
    price = _latest_price(fac_m1)
    if price <= 0:
        if diag_tick_wick and time.monotonic() - _LAST_TICK_WICK_PLACE_DIAG_TS >= TICK_WICK_DIAG_INTERVAL_SEC:
            _LAST_TICK_WICK_PLACE_DIAG_TS = time.monotonic()
            LOG.info("%s tick_wick place skip reason=bad_price price=%s", config.LOG_PREFIX, price)
        return None
    side = "long" if signal.get("action") == "OPEN_LONG" else "short"
    sl_pips = float(signal.get("sl_pips") or 0.0)
    tp_pips = float(signal.get("tp_pips") or 0.0)
    if sl_pips <= 0:
        if diag_tick_wick and time.monotonic() - _LAST_TICK_WICK_PLACE_DIAG_TS >= TICK_WICK_DIAG_INTERVAL_SEC:
            _LAST_TICK_WICK_PLACE_DIAG_TS = time.monotonic()
            LOG.info("%s tick_wick place skip reason=bad_sl sl_pips=%s", config.LOG_PREFIX, sl_pips)
        return None

    snap = get_account_snapshot()
    free_ratio = float(snap.free_margin_ratio or 0.0) if snap.free_margin_ratio is not None else 0.0
    atr = _atr_pips(fac_m1)

    cap_res = compute_cap(
        atr_pips=atr,
        free_ratio=free_ratio,
        range_active=bool(getattr(range_ctx, "active", False)),
        perf_pf=None,
        pos_bias=0.0,
        cap_min=config.CAP_MIN,
        cap_max=config.CAP_MAX,
        env_prefix=config.ENV_PREFIX,
    )
    cap = cap_res.cap
    if cap <= 0.0:
        if diag_tick_wick and time.monotonic() - _LAST_TICK_WICK_PLACE_DIAG_TS >= TICK_WICK_DIAG_INTERVAL_SEC:
            _LAST_TICK_WICK_PLACE_DIAG_TS = time.monotonic()
            LOG.info(
                "%s tick_wick place skip reason=cap_zero cap=%s free_ratio=%s atr=%s range_active=%s",
                config.LOG_PREFIX,
                cap,
                free_ratio,
                atr,
                bool(getattr(range_ctx, "active", False)),
            )
        return None

    conf = int(signal.get("confidence", 0) or 0)
    if config.MIN_ENTRY_CONF > 0 and conf < config.MIN_ENTRY_CONF:
        if diag_tick_wick and time.monotonic() - _LAST_TICK_WICK_PLACE_DIAG_TS >= TICK_WICK_DIAG_INTERVAL_SEC:
            _LAST_TICK_WICK_PLACE_DIAG_TS = time.monotonic()
            LOG.info(
                "%s tick_wick place skip reason=conf_low conf=%s min=%s",
                config.LOG_PREFIX,
                conf,
                config.MIN_ENTRY_CONF,
            )
        return None
    conf_scale = _confidence_scale(conf, lo=config.CONFIDENCE_FLOOR, hi=config.CONFIDENCE_CEIL)
    size_mult = float(signal.get("size_mult", 1.0) or 1.0)
    size_mult = max(0.6, min(1.4, size_mult))

    long_units = 0.0
    short_units = 0.0
    try:
        long_units, short_units = get_position_summary("USD_JPY", timeout=3.0)
    except Exception:
        long_units, short_units = 0.0, 0.0

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
        base_entry_units=config.BASE_ENTRY_UNITS,
        min_units=config.MIN_UNITS,
        max_margin_usage=config.MAX_MARGIN_USAGE,
        spread_pips=float(spread_pips or 0.0),
        spread_soft_cap=config.MAX_SPREAD_PIPS,
        adx=_adx(fac_m1),
        signal_score=float(conf) / 100.0,
        pocket=config.POCKET,
        strategy_tag=str(signal.get("tag") or "scalp_wick_reversal_blend"),
        env_prefix=config.ENV_PREFIX,
    )
    units = int(round(sizing.units * cap * size_mult))
    if abs(units) < config.MIN_UNITS:
        if (
            str(signal.get("tag") or "").strip() == "TickWickReversal"
            and TICK_WICK_CLAMP_MIN_UNITS
            and config.MIN_UNITS > 0
            and abs(units) >= int(config.MIN_UNITS * TICK_WICK_MIN_UNITS_CLAMP_RATIO)
        ):
            # TickWickReversal is intentionally high-frequency; when the calculated size is very close to the
            # minimum tradable units, clamp up instead of skipping entirely to avoid "never enters".
            units = config.MIN_UNITS if units >= 0 else -config.MIN_UNITS
        else:
            if (
                diag_tick_wick
                and time.monotonic() - _LAST_TICK_WICK_PLACE_DIAG_TS >= TICK_WICK_DIAG_INTERVAL_SEC
            ):
                _LAST_TICK_WICK_PLACE_DIAG_TS = time.monotonic()
                LOG.info(
                    "%s tick_wick place skip reason=units_too_small units=%s min=%s cap=%s size_mult=%s base_units=%s free_ratio=%s",
                    config.LOG_PREFIX,
                    units,
                    config.MIN_UNITS,
                    cap,
                    size_mult,
                    config.BASE_ENTRY_UNITS,
                    free_ratio,
                )
            return None
    if side == "short":
        units = -abs(units)

    if side == "long":
        sl_price = round(price - sl_pips * PIP, 3)
        tp_price = round(price + tp_pips * PIP, 3) if tp_pips > 0 else None
    else:
        sl_price = round(price + sl_pips * PIP, 3)
        tp_price = round(price - tp_pips * PIP, 3) if tp_pips > 0 else None

    sl_price, tp_price = clamp_sl_tp(price=price, sl=sl_price, tp=tp_price, is_buy=side == "long")
    client_id = _client_order_id(str(signal.get("tag") or "scalp_wick_reversal_blend"))
    entry_thesis = _build_entry_thesis(signal, fac_m1, range_ctx)
    if isinstance(entry_thesis, dict):
        entry_thesis.setdefault("env_prefix", config.ENV_PREFIX)
        entry_thesis["entry_units_intent"] = abs(units)
    meta = {
        "cap": round(cap, 3),
        "conf_scale": round(conf_scale, 3),
        "sizing": sizing.factors,
    }

    entry_thesis_ctx = None
    for _name in ("entry_thesis", "thesis"):
        _candidate = locals().get(_name)
        if isinstance(_candidate, dict):
            entry_thesis_ctx = _candidate
            break
    if entry_thesis_ctx is None:
        entry_thesis_ctx = {}

    _tech_pocket = str(locals().get("pocket", config.POCKET))
    _tech_side_raw = str(locals().get("side", locals().get("direction", "long"))).lower()
    if _tech_side_raw in {"long", "short"}:
        _tech_side = _tech_side_raw
    else:
        _tech_side = "long"
    _tech_entry_price = locals().get("price")
    if not isinstance(_tech_entry_price, (int, float)):
        _tech_entry_price = locals().get("entry_price")
    if not isinstance(_tech_entry_price, (int, float)):
        _tech_entry_price = 0.0
    try:
        _tech_entry_price = float(_tech_entry_price)
    except (TypeError, ValueError):
        _tech_entry_price = 0.0

    _tech_signal_tag = str(
        locals().get("signal_tag")
        or locals().get("strategy_tag")
        or locals().get("STRATEGY_TAG")
        or getattr(config, "STRATEGY_TAG", "")
    )

    entry_thesis_ctx.setdefault(
        "tech_tfs",
        {"fib": ["H1", "M5"], "median": ["H1", "M5"], "nwave": ["M1", "M5"], "candle": ["M1", "M5"]},
    )
    entry_thesis_ctx.setdefault("technical_context_tfs", ["M1", "M5", "H1", "H4"])
    entry_thesis_ctx.setdefault(
        "technical_context_fields",
        [
            "ma10",
            "ma20",
            "rsi",
            "atr",
            "atr_pips",
            "adx",
            "macd",
            "macd_hist",
            "plus_di",
            "minus_di",
            "bbw",
            "kc_width",
            "vwap",
            "ema20",
            "ema24",
        ],
    )
    entry_thesis_ctx.setdefault(
        "technical_context_ticks",
        ["latest_bid", "latest_ask", "latest_mid", "spread_pips", "tick_rate"],
    )
    entry_thesis_ctx.setdefault("technical_context_candle_counts", {"M1": 120, "M5": 80, "H1": 70, "H4": 60})
    entry_thesis_ctx.setdefault("tech_allow_candle", True)
    entry_thesis_ctx.setdefault(
        "tech_policy",
        {
            "mode": "balanced",
            "min_score": 0.0,
            "min_coverage": 0.0,
            "weight_fib": 0.25,
            "weight_median": 0.25,
            "weight_nwave": 0.25,
            "weight_candle": 0.25,
            "require_fib": False,
            "require_median": False,
            "require_nwave": False,
            "require_candle": False,
            "size_scale": 0.15,
            "size_min": 0.6,
            "size_max": 1.25,
        },
    )
    entry_thesis_ctx.setdefault("tech_policy_locked", False)
    entry_thesis_ctx.setdefault("env_tf", "M1")
    entry_thesis_ctx.setdefault("struct_tf", "M1")
    entry_thesis_ctx.setdefault("entry_tf", "M1")

    tech_decision = evaluate_entry_techniques(
        entry_price=_tech_entry_price,
        side=_tech_side,
        pocket=_tech_pocket,
        strategy_tag=_tech_signal_tag,
        entry_thesis=entry_thesis_ctx,
        allow_candle=bool(entry_thesis_ctx.get("tech_allow_candle", False)),
    )
    if not tech_decision.allowed and not getattr(config, "TECH_FAILOPEN", True):
        return None

    entry_thesis_ctx["tech_score"] = round(tech_decision.score, 3) if tech_decision.score is not None else None
    entry_thesis_ctx["tech_coverage"] = (
        round(tech_decision.coverage, 3) if tech_decision.coverage is not None else None
    )
    entry_thesis_ctx["tech_entry"] = tech_decision.debug
    entry_thesis_ctx["tech_reason"] = tech_decision.reason
    entry_thesis_ctx["tech_decision_allowed"] = bool(tech_decision.allowed)

    _tech_units_raw = locals().get("units")
    if isinstance(_tech_units_raw, (int, float)):
        _tech_units = int(round(abs(float(_tech_units_raw)) * tech_decision.size_mult))
        if _tech_units <= 0:
            return None
        units = _tech_units if _tech_side == "long" else -_tech_units
        entry_thesis_ctx["entry_units_intent"] = abs(int(units))

    _tech_conf = locals().get("conf")
    if isinstance(_tech_conf, (int, float)):
        _tech_conf = float(_tech_conf)
        if tech_decision.score is not None:
            if tech_decision.score >= 0:
                _tech_conf += tech_decision.score * getattr(config, "TECH_CONF_BOOST", 0.0)
            else:
                _tech_conf += tech_decision.score * getattr(config, "TECH_CONF_PENALTY", 0.0)
        conf = _tech_conf


    order_id = await market_order(
        instrument="USD_JPY",
        units=units,
        sl_price=sl_price,
        tp_price=tp_price,
        pocket=config.POCKET,
        client_order_id=client_id,
        strategy_tag=str(signal.get("tag") or "scalp_wick_reversal_blend"),
        entry_thesis=entry_thesis,
        meta=meta,
        confidence=conf,
    )
    if diag_tick_wick and time.monotonic() - _LAST_TICK_WICK_PLACE_DIAG_TS >= TICK_WICK_DIAG_INTERVAL_SEC:
        _LAST_TICK_WICK_PLACE_DIAG_TS = time.monotonic()
        LOG.info(
            "%s tick_wick place result order_id=%s units=%s side=%s sl=%s tp=%s conf=%s cap=%s",
            config.LOG_PREFIX,
            order_id,
            units,
            side,
            sl_price,
            tp_price,
            conf,
            cap,
        )
    return order_id


async def scalp_wick_reversal_blend_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled (idle)", config.LOG_PREFIX)
        try:
            while True:
                await asyncio.sleep(3600.0)
        except asyncio.CancelledError:
            return

    LOG.info("%s worker start (interval=%.1fs mode=%s)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC, config.MODE)
    LOG.info("Application started!")
    pos_manager = PositionManager()
    try:
        stage_tracker = StageTracker()
    except Exception as exc:
        LOG.warning(
            "%s stage_tracker init failed; fallback=noop err=%s",
            config.LOG_PREFIX,
            exc,
        )
        stage_tracker = _NoopStageTracker()
    last_entry_ts = 0.0
    last_perf_sync = 0.0
    last_stage_sync = 0.0
    last_guard_log = 0.0
    last_diag_log = 0.0
    bypass_common_guard = config.MODE in config.GUARD_BYPASS_MODES
    bypass_perf_guard = _perf_guard_bypass_enabled()
    # Guard bypass is risky for modes that can generate large tail losses when data/exits degrade.
    # In particular, tick_imbalance suffered margin closeouts when common guards (spread/air/perf/stage)
    # were skipped. Keep these modes under the common guardrails regardless of env settings.
    if config.MODE in {"level_reject", "level_reject_plus", "tick_imbalance", "tick_imbalance_rrplus"}:
        if bypass_common_guard:
            LOG.warning(
                "%s guard bypass requested but disabled for mode=%s",
                config.LOG_PREFIX,
                config.MODE,
            )
        bypass_common_guard = False
    if bypass_perf_guard:
        LOG.info("%s perf guard bypass enabled mode=%s", config.LOG_PREFIX, config.MODE)

    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now = datetime.datetime.utcnow()
            now_mono = time.monotonic()
            if not is_market_open(now):
                continue
            if not can_trade(config.POCKET):
                continue
            if (
                not bypass_common_guard
                and config.COOLDOWN_SEC > 0.0
                and time.monotonic() - last_entry_ts < config.COOLDOWN_SEC
            ):
                continue
            if config.MODE == "drought_revert" and not _entry_drought_ok(now):
                continue
            snapshot = None
            if not bypass_common_guard and config.ENTRY_GUARD_ENABLED:
                try:
                    snapshot = get_account_snapshot()
                except Exception:
                    snapshot = None
                if not _entry_guard_ok(snapshot):
                    continue

            factors = all_factors()
            fac_m1 = factors.get("M1") or {}
            fac_m5 = factors.get("M5") or {}
            fac_h1 = factors.get("H1") or {}
            fac_h4 = factors.get("H4") or {}

            range_ctx = detect_range_mode(fac_m1, fac_h4)

            if (
                not bypass_common_guard
                and config.PERF_REFRESH_SEC > 0.0
                and now_mono - last_perf_sync >= config.PERF_REFRESH_SEC
            ):
                try:
                    pos_manager.sync_trades()
                except Exception as exc:
                    LOG.debug("%s sync_trades error: %s", config.LOG_PREFIX, exc)
                last_perf_sync = now_mono

            air = evaluate_air(fac_m1, fac_h4, range_ctx=range_ctx, tag=config.MODE)
            if air.enabled and not air.allow_entry and not bypass_common_guard:
                continue

            positions = None
            open_trades_all = None

            # max open trades guard
            if (config.MAX_OPEN_TRADES > 0 or config.MAX_OPEN_TRADES_GLOBAL > 0) and not bypass_common_guard:
                try:
                    positions = pos_manager.get_open_positions()
                    scalp_info = positions.get(config.POCKET) or {}
                    open_trades_all = scalp_info.get("open_trades") or []
                    if config.MAX_OPEN_TRADES_GLOBAL > 0 and len(open_trades_all) >= config.MAX_OPEN_TRADES_GLOBAL:
                        continue
                    open_trades = open_trades_all
                    if config.OPEN_TRADES_SCOPE == "tag":
                        mode_tag = _mode_to_tag(config.MODE)
                        if mode_tag:
                            tag_lower = mode_tag.lower()
                            open_trades = [
                                tr
                                for tr in open_trades_all
                                if str(tr.get("strategy_tag") or "").lower() == tag_lower
                            ]
                    if config.MAX_OPEN_TRADES > 0 and len(open_trades) >= config.MAX_OPEN_TRADES:
                        continue
                except Exception:
                    pass

            if (
                not bypass_common_guard
                and config.STAGE_REFRESH_SEC > 0.0
                and now_mono - last_stage_sync >= config.STAGE_REFRESH_SEC
            ):
                try:
                    if positions is None:
                        positions = pos_manager.get_open_positions()
                        scalp_info = positions.get(config.POCKET) or {}
                        open_trades_all = scalp_info.get("open_trades") or []
                    snap = snapshot or get_account_snapshot()
                    stage_tracker.update_loss_streaks(
                        now=now,
                        range_active=bool(range_ctx.active),
                        atr_pips=_atr_pips(fac_m1),
                        vol_5m=float(fac_m1.get("vol_5m") or 0.0),
                        adx_m1=float(fac_m1.get("adx") or 0.0),
                        nav=float(snap.nav or snap.balance or 0.0),
                        open_scalp_positions=len(open_trades_all or []),
                        atr_m5_pips=float(fac_m5.get("atr_pips") or 0.0),
                    )
                except Exception as exc:
                    LOG.debug("%s stage_tracker update error: %s", config.LOG_PREFIX, exc)
                last_stage_sync = now_mono

            # spread guard
            if not bypass_common_guard:
                blocked, _, _, _ = spread_monitor.is_blocked()
                if blocked:
                    continue

            strategies = []
            # /home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env may define a global allowlist
            # shared across multiple scalp_precision units. Prefer per-unit override when present so
            # a single unit can run an extra mode without editing the global env file.
            allowlist = set([s.lower() for s in _env_csv("SCALP_PRECISION_UNIT_ALLOWLIST", config.ALLOWLIST_RAW)])
            mode = (config.MODE or "").strip().lower()
            if mode:
                if config.MODE_FILTER_ALLOWLIST:
                    if allowlist:
                        if mode in allowlist:
                            allowlist = {mode}
                        else:
                            allowlist = {"__none__"}
                    else:
                        allowlist.add(mode)
                else:
                    if not allowlist:
                        allowlist.add(mode)

            def enabled(name: str) -> bool:
                if not allowlist:
                    return True
                return name.lower() in allowlist

            if enabled("spread_revert"):
                strategies.append(("SpreadRangeRevert", _signal_spread_revert, {"tag": "SpreadRangeRevert"}))
            if enabled("drought_revert"):
                strategies.append(("DroughtRevert", _signal_drought_revert, {"tag": "DroughtRevert"}))
            if enabled("precision_lowvol"):
                strategies.append(("PrecisionLowVol", _signal_precision_lowvol, {"tag": "PrecisionLowVol"}))
            if enabled("rangefaderpro"):
                strategies.append(("RangeFaderPro", _signal_spread_revert, {"tag": "RangeFaderPro"}))
            if enabled("vwap_revert"):
                strategies.append(("VwapRevertS", _signal_vwap_revert, {"tag": "VwapRevertS"}))
            if enabled("stoch_bounce"):
                strategies.append(("StochBollBounce", _signal_stoch_bounce, {"tag": "StochBollBounce"}))
            if enabled("divergence_revert"):
                strategies.append(("DivergenceRevert", _signal_divergence_revert, {"tag": "DivergenceRevert"}))
            if enabled("compression_retest"):
                strategies.append(("CompressionRetest", _signal_compression_retest, {"tag": "CompressionRetest"}))
            if enabled("htf_pullback"):
                strategies.append(("HTFPullbackS", _signal_htf_pullback, {"tag": "HTFPullbackS"}))
            if enabled("macd_trend"):
                strategies.append(("MacdTrendRide", _signal_macd_trend, {"tag": "MacdTrendRide"}))
            if enabled("ema_slope_pull"):
                strategies.append(("EmaSlopePull", _signal_ema_slope_pull, {"tag": "EmaSlopePull"}))
            if enabled("tick_imbalance"):
                strategies.append(("TickImbalance", _signal_tick_imbalance, {"tag": "TickImbalance"}))
            if enabled("tick_imbalance_rrplus"):
                strategies.append(("TickImbalanceRRPlus", _signal_tick_imbalance_rrplus, {"tag": "TickImbalanceRRPlus"}))
            if enabled("level_reject"):
                strategies.append(("LevelReject", _signal_level_reject, {"tag": "LevelReject"}))
            if enabled("level_reject_plus"):
                strategies.append(("LevelRejectPlus", _signal_level_reject_plus, {"tag": "LevelRejectPlus"}))
            if enabled("liquidity_sweep"):
                strategies.append(("LiquiditySweep", _signal_liquidity_sweep, {"tag": "LiquiditySweep"}))
            if enabled("wick_reversal"):
                strategies.append(("WickReversal", _signal_wick_reversal, {"tag": "WickReversal"}))
            if enabled("wick_reversal_blend"):
                strategies.append(("WickReversalBlend", _signal_wick_reversal_blend, {"tag": "WickReversalBlend"}))
            if enabled("wick_reversal_hf"):
                strategies.append(("WickReversalHF", _signal_wick_reversal_hf, {"tag": "WickReversalHF"}))
            if enabled("wick_reversal_pro"):
                strategies.append(("WickReversalPro", _signal_wick_reversal_pro, {"tag": "WickReversalPro"}))
            if enabled("tick_wick_reversal"):
                strategies.append(("TickWickReversal", _signal_tick_wick_reversal, {"tag": "TickWickReversal"}))
            if enabled("session_edge"):
                strategies.append(("SessionEdge", _signal_session_edge, {"tag": "SessionEdge"}))
            if enabled("squeeze_pulse_break"):
                strategies.append(("SqueezePulseBreak", _signal_squeeze_pulse_break, {"tag": "SqueezePulseBreak"}))
            if enabled("false_break_fade"):
                strategies.append(("FalseBreakFade", _signal_false_break_fade, {"tag": "FalseBreakFade"}))

            signals: List[Dict[str, object]] = []
            for name, fn, kwargs in strategies:
                signal = _dispatch_strategy_signal(
                    name=name,
                    fn=fn,
                    fac_m1=fac_m1,
                    fac_h1=fac_h1,
                    fac_h4=fac_h4,
                    fac_m5=fac_m5,
                    range_ctx=range_ctx,
                    now_utc=now,
                    kwargs=kwargs,
                )
                if signal:
                    signal = adjust_signal(signal, air)
                    if signal:
                        if str(signal.get("tag") or "").strip() == "SqueezePulseBreak":
                            ok_spb, spb_reason = _spb_post_entry_guard(signal, fac_m1, now)
                            if not ok_spb:
                                if SPB_DIAG and config.MODE == "squeeze_pulse_break":
                                    LOG.info(
                                        "%s spb guard blocked reason=%s action=%s rsi=%.1f adx=%.1f air_dir=%s air_score=%.3f",
                                        config.LOG_PREFIX,
                                        spb_reason,
                                        signal.get("action"),
                                        _rsi(fac_m1),
                                        _adx(fac_m1),
                                        signal.get("air_pressure_dir"),
                                        float(signal.get("air_score") or 0.0),
                                    )
                                continue
                        if (
                            str(signal.get("tag") or "").strip() == "WickReversalPro"
                            and air.enabled
                            and not bypass_common_guard
                        ):
                            if air.air_score < WICK_PRO_MIN_AIR_SCORE:
                                continue
                            if air.exec_quality < WICK_PRO_MIN_EXEC_QUALITY:
                                continue
                            if air.regime_shift > WICK_PRO_MAX_REGIME_SHIFT:
                                continue
                        if (
                            str(signal.get("tag") or "").strip() == "TickWickReversal"
                            and air.enabled
                            and not bypass_common_guard
                        ):
                            if air.air_score < TICK_WICK_MIN_AIR_SCORE:
                                continue
                            if air.exec_quality < TICK_WICK_MIN_EXEC_QUALITY:
                                continue
                            if air.regime_shift > TICK_WICK_MAX_REGIME_SHIFT:
                                continue
                        conf = int(signal.get("confidence", 0) or 0)
                        if config.MIN_ENTRY_CONF > 0 and conf < config.MIN_ENTRY_CONF:
                            continue
                        if not bypass_common_guard:
                            tag = str(signal.get("tag") or "").strip()
                            if tag:
                                pocket_decision = perf_guard.is_pocket_allowed(
                                    config.POCKET,
                                    env_prefix=config.ENV_PREFIX,
                                )
                                if not pocket_decision.allowed:
                                    if now_mono - last_guard_log > 30.0:
                                        LOG.info(
                                            "%s pocket guard blocked pocket=%s reason=%s",
                                            config.LOG_PREFIX,
                                            config.POCKET,
                                            pocket_decision.reason,
                                        )
                                        last_guard_log = now_mono
                                    continue
                                if not bypass_perf_guard:
                                    perf_decision = perf_guard.is_allowed(
                                        tag,
                                        config.POCKET,
                                        hour=now.hour,
                                        env_prefix=config.ENV_PREFIX,
                                    )
                                    if not perf_decision.allowed:
                                        if now_mono - last_guard_log > 30.0:
                                            LOG.info(
                                                "%s perf guard blocked tag=%s reason=%s",
                                                config.LOG_PREFIX,
                                                tag,
                                                perf_decision.reason,
                                            )
                                            last_guard_log = now_mono
                                        continue
                                blocked, remain, reason = stage_tracker.is_strategy_blocked(tag, now=now)
                                if blocked:
                                    if now_mono - last_guard_log > 30.0:
                                        LOG.info(
                                            "%s strategy cooldown tag=%s remain=%ss reason=%s",
                                            config.LOG_PREFIX,
                                            tag,
                                            remain,
                                            reason,
                                        )
                                        last_guard_log = now_mono
                                    continue
                                direction = _signal_direction(signal)
                                if direction:
                                    blocked, remain, reason = stage_tracker.is_blocked(
                                        config.POCKET, direction, now=now
                                    )
                                    if blocked:
                                        if now_mono - last_guard_log > 30.0:
                                            LOG.info(
                                                "%s pocket cooldown dir=%s remain=%ss reason=%s",
                                                config.LOG_PREFIX,
                                                direction,
                                                remain,
                                                reason,
                                            )
                                            last_guard_log = now_mono
                                        continue
                        signals.append(signal)

            if not signals:
                if TICK_WICK_DIAG and config.MODE == "tick_wick_reversal":
                    if now_mono - last_diag_log >= max(1.0, TICK_WICK_DIAG_INTERVAL_SEC):
                        last_diag_log = now_mono
                        LOG.info(
                            "%s tick_wick diag signals=0 air_score=%.3f allow_entry=%s range_active=%s range_score=%.3f adx=%.1f bbw=%.3f",
                            config.LOG_PREFIX,
                            float(getattr(air, "air_score", 0.0) or 0.0),
                            bool(getattr(air, "allow_entry", True)),
                            bool(getattr(range_ctx, "active", False)),
                            float(getattr(range_ctx, "score", 0.0) or 0.0),
                            _adx(fac_m1),
                            _bbw(fac_m1),
                        )
                if WICK_HF_DIAG and config.MODE == "wick_reversal_hf":
                    if now_mono - last_diag_log >= max(1.0, WICK_HF_DIAG_INTERVAL_SEC):
                        last_diag_log = now_mono
                        m = _wick_hf_diag_metrics(fac_m1, range_ctx)
                        LOG.info(
                            "%s wick_hf diag signals=0 block=%s spread_ok=%s spread=%.2fp p25=%.2fp range_ok=%s range_score=%.3f rng=%.2fp body=%.2fp ratio=%.3f side=%s bb_ok=%s bb_touch=%.2fp tick_n=%d tick_ok=%s tick_dir=%s tick_strength=%.2f proj=%s adx=%.1f bbw=%.6f atr=%.2f",
                            config.LOG_PREFIX,
                            str(m.get("block") or ""),
                            bool(m.get("spread_ok")),
                            float(m.get("spread_pips") or 0.0),
                            float(m.get("p25_pips") or 0.0),
                            bool(m.get("range_ok")),
                            float(m.get("range_score") or 0.0),
                            float(m.get("rng_pips") or 0.0),
                            float(m.get("body_pips") or 0.0),
                            float(m.get("wick_ratio") or 0.0),
                            str(m.get("side") or ""),
                            bool(m.get("bb_ok")),
                            float(m.get("bb_touch_pips") or 0.0),
                            int(m.get("tick_n") or 0),
                            bool(m.get("tick_ok")),
                            str(m.get("tick_dir") or ""),
                            float(m.get("tick_strength") or 0.0),
                            bool(m.get("proj_allow")) if m.get("proj_allow") is not None else None,
                            float(m.get("adx") or 0.0),
                            float(m.get("bbw") or 0.0),
                            float(m.get("atr") or 0.0),
                        )
                if SPB_DIAG and config.MODE == "squeeze_pulse_break":
                    if now_mono - last_diag_log >= max(1.0, SPB_DIAG_INTERVAL_SEC):
                        last_diag_log = now_mono
                        m = _spb_diag_metrics(fac_m1, range_ctx)
                        LOG.info(
                            "%s spb diag signals=0 spread_ok=%s spread=%.2fp bbw=%.6f atr=%.2f range_active=%s range_score=%.3f tick_n=%d tick_range=%.2fp imb_ratio=%.2f mom=%.2fp brk_long=%s brk_short=%s",
                            config.LOG_PREFIX,
                            bool(m.get("spread_ok")),
                            float(m.get("spread_pips") or 0.0),
                            float(m.get("bbw") or 0.0),
                            float(m.get("atr") or 0.0),
                            bool(m.get("range_active")),
                            float(m.get("range_score") or 0.0),
                            int(m.get("tick_n") or 0),
                            float(m.get("tick_range_pips") or 0.0),
                            float(m.get("imb_ratio") or 0.0),
                            float(m.get("imb_mom_pips") or 0.0),
                            bool(m.get("breakout_long")),
                            bool(m.get("breakout_short")),
                        )
                continue

            if TICK_WICK_DIAG and config.MODE == "tick_wick_reversal":
                if now_mono - last_diag_log >= max(1.0, TICK_WICK_DIAG_INTERVAL_SEC):
                    last_diag_log = now_mono
                    top = signals[0]
                    LOG.info(
                        "%s tick_wick diag signals=%d top_conf=%s top_action=%s air_score=%.3f allow_entry=%s range_active=%s range_score=%.3f",
                        config.LOG_PREFIX,
                        len(signals),
                        top.get("confidence"),
                        top.get("action"),
                        float(getattr(air, "air_score", 0.0) or 0.0),
                        bool(getattr(air, "allow_entry", True)),
                        bool(getattr(range_ctx, "active", False)),
                        float(getattr(range_ctx, "score", 0.0) or 0.0),
                    )
            if WICK_HF_DIAG and config.MODE == "wick_reversal_hf":
                if now_mono - last_diag_log >= max(1.0, WICK_HF_DIAG_INTERVAL_SEC):
                    last_diag_log = now_mono
                    top = signals[0]
                    wick = top.get("wick") if isinstance(top, dict) else None
                    try:
                        rng_pips = float((wick or {}).get("rng_pips") or 0.0)
                    except Exception:
                        rng_pips = 0.0
                    try:
                        ratio = float((wick or {}).get("ratio") or 0.0)
                    except Exception:
                        ratio = 0.0
                    try:
                        tick_strength = float((wick or {}).get("tick_strength") or 0.0)
                    except Exception:
                        tick_strength = 0.0
                    LOG.info(
                        "%s wick_hf diag signals=%d top_conf=%s top_action=%s rng=%.2fp ratio=%.3f tick_strength=%.2f air_score=%.3f allow_entry=%s range_active=%s range_score=%.3f",
                        config.LOG_PREFIX,
                        len(signals),
                        top.get("confidence"),
                        top.get("action"),
                        rng_pips,
                        ratio,
                        tick_strength,
                        float(getattr(air, "air_score", 0.0) or 0.0),
                        bool(getattr(air, "allow_entry", True)),
                        bool(getattr(range_ctx, "active", False)),
                        float(getattr(range_ctx, "score", 0.0) or 0.0),
                    )
            if SPB_DIAG and config.MODE == "squeeze_pulse_break":
                if now_mono - last_diag_log >= max(1.0, SPB_DIAG_INTERVAL_SEC):
                    last_diag_log = now_mono
                    top = signals[0]
                    m = _spb_diag_metrics(fac_m1, range_ctx)
                    LOG.info(
                        "%s spb diag signals=%d top_conf=%s top_action=%s spread_ok=%s bbw=%.6f atr=%.2f range_active=%s range_score=%.3f tick_n=%d tick_range=%.2fp imb_ratio=%.2f mom=%.2fp brk_long=%s brk_short=%s",
                        config.LOG_PREFIX,
                        len(signals),
                        top.get("confidence"),
                        top.get("action"),
                        bool(m.get("spread_ok")),
                        float(m.get("bbw") or 0.0),
                        float(m.get("atr") or 0.0),
                        bool(m.get("range_active")),
                        float(m.get("range_score") or 0.0),
                        int(m.get("tick_n") or 0),
                        float(m.get("tick_range_pips") or 0.0),
                        float(m.get("imb_ratio") or 0.0),
                        float(m.get("imb_mom_pips") or 0.0),
                        bool(m.get("breakout_long")),
                        bool(m.get("breakout_short")),
                    )

            # rank by confidence
            signals.sort(key=lambda s: int(s.get("confidence", 0)), reverse=True)
            selected = signals[: max(1, config.MAX_SIGNALS_PER_CYCLE)]
            for sig in selected:
                order_id = await _place_order(sig, fac_m1=fac_m1, fac_h4=fac_h4, range_ctx=range_ctx, now=now)
                if order_id:
                    last_entry_ts = time.monotonic()
    except asyncio.CancelledError:
        return
    finally:
        try:
            stage_tracker.close()
        except Exception:
            LOG.debug("%s stage_tracker close error", config.LOG_PREFIX)
        try:
            pos_manager.close()
        except Exception:
            LOG.debug("%s pos_manager close error", config.LOG_PREFIX)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    asyncio.run(scalp_wick_reversal_blend_worker())


if __name__ == "__main__":
    main()
