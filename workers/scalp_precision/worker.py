from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from analysis.range_guard import detect_range_mode
from analysis.range_model import compute_range_snapshot
from execution.order_manager import market_order
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

LOG = logging.getLogger(__name__)

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
    'level_reject': 'LevelReject',
    'wick_reversal': 'WickReversal',
    'session_edge': 'SessionEdge',
    'drought_revert': 'DroughtRevert',
    'precision_lowvol': 'PrecisionLowVol',
}

def _mode_to_tag(mode: str) -> Optional[str]:
    return _MODE_TAG_MAP.get((mode or '').strip().lower())

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
SPREAD_REV_BBW_MAX = _env_float("SPREAD_REV_BBW_MAX", 0.24)
SPREAD_REV_ATR_MIN = _env_float("SPREAD_REV_ATR_MIN", 0.7)
SPREAD_REV_ATR_MAX = _env_float("SPREAD_REV_ATR_MAX", 3.2)
SPREAD_REV_RSI_LONG_MAX = _env_float("SPREAD_REV_RSI_LONG_MAX", 47.0)
SPREAD_REV_RSI_SHORT_MIN = _env_float("SPREAD_REV_RSI_SHORT_MIN", 53.0)
SPREAD_REV_BB_TOUCH_PIPS = _env_float("SPREAD_REV_BB_TOUCH_PIPS", 0.8)
SPREAD_REV_TICK_MIN = _env_int("SPREAD_REV_TICK_MIN", 6)
SPREAD_REV_RANGE_ONLY_SCORE = _env_float("SPREAD_REV_RANGE_ONLY_SCORE", 0.45)

COMPRESS_BBW_MAX = _env_float("COMPRESS_BBW_MAX", 0.20)
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
HTF_M5_BBW_MAX = _env_float("HTF_PULLBACK_BBW_MAX", 0.32)

TICK_IMB_WINDOW_SEC = _env_float("TICK_IMB_WINDOW_SEC", 4.5)
TICK_IMB_RATIO_MIN = _env_float("TICK_IMB_RATIO_MIN", 0.68)
TICK_IMB_MOM_MIN_PIPS = _env_float("TICK_IMB_MOM_MIN_PIPS", 0.45)
TICK_IMB_RANGE_MIN_PIPS = _env_float("TICK_IMB_RANGE_MIN_PIPS", 0.25)
TICK_IMB_ATR_MIN = _env_float("TICK_IMB_ATR_MIN", 0.7)
TICK_IMB_SIZE_MULT = _env_float("TICK_IMB_SIZE_MULT", 1.25)
TICK_IMB_ALLOWED_REGIMES = {
    s.strip().lower()
    for s in _env_csv("TICK_IMB_ALLOWED_REGIMES", "")
    if s.strip()
}
TICK_IMB_BLOCK_RANGE_MODE = _env_bool("TICK_IMB_BLOCK_RANGE_MODE", True)

LEVEL_LOOKBACK = _env_int("LEVEL_REJECT_LOOKBACK", 20)
LEVEL_BAND_PIPS = _env_float("LEVEL_REJECT_BAND_PIPS", 0.8)
LEVEL_RSI_LONG_MAX = _env_float("LEVEL_REJECT_RSI_LONG_MAX", 48.0)
LEVEL_RSI_SHORT_MIN = _env_float("LEVEL_REJECT_RSI_SHORT_MIN", 52.0)
LEVEL_REJECT_SIZE_MULT = _env_float("LEVEL_REJECT_SIZE_MULT", 1.15)

WICK_RANGE_MIN_PIPS = _env_float("WICK_REV_RANGE_MIN_PIPS", 2.0)
WICK_BODY_MAX_PIPS = _env_float("WICK_REV_BODY_MAX_PIPS", 0.9)
WICK_RATIO_MIN = _env_float("WICK_REV_RATIO_MIN", 0.55)
WICK_ADX_MAX = _env_float("WICK_REV_ADX_MAX", 24.0)
WICK_BBW_MAX = _env_float("WICK_REV_BBW_MAX", 0.28)

SESSION_ALLOW_HOURS = parse_hours(os.getenv("SESSION_EDGE_ALLOW_HOURS_JST", "12,13,18-22"))
SESSION_BLOCK_HOURS = parse_hours(os.getenv("SESSION_EDGE_BLOCK_HOURS_JST", "00-02,16"))

VWAP_REV_GAP_MIN = _env_float("VWAP_REV_GAP_MIN", 1.2)
VWAP_REV_BB_TOUCH_PIPS = _env_float("VWAP_REV_BB_TOUCH_PIPS", 0.9)
VWAP_REV_RSI_LONG_MAX = _env_float("VWAP_REV_RSI_LONG_MAX", 46.0)
VWAP_REV_RSI_SHORT_MIN = _env_float("VWAP_REV_RSI_SHORT_MIN", 54.0)
VWAP_REV_STOCH_LONG_MAX = _env_float("VWAP_REV_STOCH_LONG_MAX", 0.2)
VWAP_REV_STOCH_SHORT_MIN = _env_float("VWAP_REV_STOCH_SHORT_MIN", 0.8)
VWAP_REV_ADX_MAX = _env_float("VWAP_REV_ADX_MAX", 22.0)
VWAP_REV_BBW_MAX = _env_float("VWAP_REV_BBW_MAX", 0.26)
VWAP_REV_ATR_MIN = _env_float("VWAP_REV_ATR_MIN", 0.7)
VWAP_REV_ATR_MAX = _env_float("VWAP_REV_ATR_MAX", 3.2)
VWAP_REV_SPREAD_P25 = _env_float("VWAP_REV_SPREAD_P25", 0.9)
VWAP_REV_RANGE_SCORE = _env_float("VWAP_REV_RANGE_SCORE", 0.4)

STOCH_BOUNCE_STOCH_LONG_MAX = _env_float("STOCH_BOUNCE_STOCH_LONG_MAX", 0.18)
STOCH_BOUNCE_STOCH_SHORT_MIN = _env_float("STOCH_BOUNCE_STOCH_SHORT_MIN", 0.82)
STOCH_BOUNCE_RSI_LONG_MAX = _env_float("STOCH_BOUNCE_RSI_LONG_MAX", 46.0)
STOCH_BOUNCE_RSI_SHORT_MIN = _env_float("STOCH_BOUNCE_RSI_SHORT_MIN", 54.0)
STOCH_BOUNCE_ADX_MAX = _env_float("STOCH_BOUNCE_ADX_MAX", 22.0)
STOCH_BOUNCE_BBW_MAX = _env_float("STOCH_BOUNCE_BBW_MAX", 0.26)
STOCH_BOUNCE_ATR_MIN = _env_float("STOCH_BOUNCE_ATR_MIN", 0.7)
STOCH_BOUNCE_ATR_MAX = _env_float("STOCH_BOUNCE_ATR_MAX", 3.0)
STOCH_BOUNCE_MACD_ABS_MAX = _env_float("STOCH_BOUNCE_MACD_ABS_MAX", 0.6)
STOCH_BOUNCE_BB_TOUCH_PIPS = _env_float("STOCH_BOUNCE_BB_TOUCH_PIPS", 0.9)
STOCH_BOUNCE_RANGE_SCORE = _env_float("STOCH_BOUNCE_RANGE_SCORE", 0.35)

MACD_TREND_ADX_MIN = _env_float("MACD_TREND_ADX_MIN", 20.0)
MACD_TREND_BBW_MIN = _env_float("MACD_TREND_BBW_MIN", 0.16)
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

    sl = max(1.0, min(1.7, atr * 0.75))
    tp = max(0.9, min(1.7, atr * 0.9))
    conf = 52
    conf += int(min(10, abs(rsi - 50.0) * 0.5))
    conf += int(min(8, rev_strength * 4.0))
    conf -= int(min(6, max(0.0, adx - 20.0) * 0.4))

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(40, min(90, conf))),
        "tag": tag,
        "reason": "drought_revert",
        "range_score": round(range_score, 3),
        "size_mult": 0.9,
    }


def _signal_precision_lowvol(
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
    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    rsi = _rsi(fac_m1)
    stoch = _stoch_rsi(fac_m1)
    vgap = _vwap_gap_pips(fac_m1)

    range_score = float(range_ctx.score or 0.0) if range_ctx else 0.0
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

    vgap_block = config.PREC_LOWVOL_VWAP_GAP_BLOCK
    if vgap_block > 0.0:
        if side == "long" and vgap > vgap_block:
            return None
        if side == "short" and vgap < -vgap_block:
            return None

    vgap_bias_min = config.PREC_LOWVOL_VWAP_GAP_MIN
    vgap_bias_ok = (vgap <= -vgap_bias_min and side == "long") or (vgap >= vgap_bias_min and side == "short")

    proj_allow, size_mult, proj_detail = projection_decision(side, mode="range")
    if not proj_allow:
        return None

    dist = dist_lower if side == "long" else dist_upper
    touch_ratio = max(0.0, (band - dist) / max(0.2, band))

    sl = max(1.0, min(1.6, atr * 0.75))
    tp = max(1.1, min(2.0, atr * (0.9 + min(0.2, rev_strength * 0.2))))
    conf = 58
    conf += int(min(10, abs(rsi - 50.0) * 0.6))
    conf += int(min(10, rev_strength * 4.0))
    conf += int(min(6, touch_ratio * 6.0))
    conf += int(min(4, range_score * 4.0))
    if vgap_bias_ok:
        conf += 3
    if range_ctx and getattr(range_ctx, "active", False):
        conf += 2

    size_boost = 0.0
    if vgap_bias_ok:
        size_boost += 0.05
    if touch_ratio >= 0.5:
        size_boost += 0.06
    if rev_strength >= 0.75:
        size_boost += 0.06
    size_cap = 1.35 if rev_strength >= 0.75 else 1.25
    size_mult = max(0.85, min(size_cap, size_mult + size_boost))

    return {
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


def _signal_tick_imbalance(
    fac_m1: Dict[str, object],
    range_ctx=None,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    if TICK_IMB_BLOCK_RANGE_MODE and range_ctx and getattr(range_ctx, "active", False):
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


def _signal_level_reject(
    fac_m1: Dict[str, object],
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
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

    atr = _atr_pips(fac_m1)
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

    proj_allow, size_mult, proj_detail = projection_decision(side, mode="range")
    if not proj_allow:
        return None

    sl = max(1.2, min(1.8, atr * 0.7))
    tp = max(1.4, min(2.2, atr * 0.95))
    conf = 60 + int(min(10, abs(vgap) * 1.6)) + int(min(8, rev_strength * 3.5))
    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": int(max(45, min(92, conf))),
        "tag": tag,
        "reason": "vwap_revert",
        "size_mult": round(size_mult, 3),
        "projection": proj_detail,
    }


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
    return {
        "strategy_tag": signal.get("tag"),
        "confidence": signal.get("confidence", 0),
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
        "div_score": _div_score(fac_m1),
        "air_score": signal.get("air_score"),
        "air_pressure": signal.get("air_pressure"),
        "air_pressure_dir": signal.get("air_pressure_dir"),
        "air_spread_state": signal.get("air_spread_state"),
        "air_exec_quality": signal.get("air_exec_quality"),
        "air_regime_shift": signal.get("air_regime_shift"),
        "air_range_pref": signal.get("air_range_pref"),
    }


async def _place_order(
    signal: Dict[str, object],
    *,
    fac_m1: Dict[str, object],
    fac_h4: Dict[str, object],
    range_ctx,
    now: datetime.datetime,
) -> Optional[str]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None
    side = "long" if signal.get("action") == "OPEN_LONG" else "short"
    sl_pips = float(signal.get("sl_pips") or 0.0)
    tp_pips = float(signal.get("tp_pips") or 0.0)
    if sl_pips <= 0:
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
    )
    cap = cap_res.cap
    if cap <= 0.0:
        return None

    conf = int(signal.get("confidence", 0) or 0)
    if config.MIN_ENTRY_CONF > 0 and conf < config.MIN_ENTRY_CONF:
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
        strategy_tag=str(signal.get("tag") or "scalp_precision"),
    )
    units = int(round(sizing.units * cap * size_mult))
    if abs(units) < config.MIN_UNITS:
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
    client_id = _client_order_id(str(signal.get("tag") or "scalp_precision"))
    entry_thesis = _build_entry_thesis(signal, fac_m1, range_ctx)
    meta = {
        "cap": round(cap, 3),
        "conf_scale": round(conf_scale, 3),
        "sizing": sizing.factors,
    }

    return await market_order(
        instrument="USD_JPY",
        units=units,
        sl_price=sl_price,
        tp_price=tp_price,
        pocket=config.POCKET,
        client_order_id=client_id,
        strategy_tag=str(signal.get("tag") or "scalp_precision"),
        entry_thesis=entry_thesis,
        meta=meta,
        confidence=conf,
    )


async def scalp_precision_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled (idle)", config.LOG_PREFIX)
        try:
            while True:
                await asyncio.sleep(3600.0)
        except asyncio.CancelledError:
            return

    LOG.info("%s worker start (interval=%.1fs mode=%s)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC, config.MODE)
    pos_manager = PositionManager()
    stage_tracker = StageTracker()
    last_entry_ts = 0.0
    last_perf_sync = 0.0
    last_stage_sync = 0.0
    last_guard_log = 0.0
    bypass_common_guard = config.MODE in config.GUARD_BYPASS_MODES

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
            allowlist = set([s.lower() for s in _env_csv("SCALP_PRECISION_ALLOWLIST", config.ALLOWLIST_RAW)])
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
            if enabled("level_reject"):
                strategies.append(("LevelReject", _signal_level_reject, {"tag": "LevelReject"}))
            if enabled("wick_reversal"):
                strategies.append(("WickReversal", _signal_wick_reversal, {"tag": "WickReversal"}))
            if enabled("session_edge"):
                strategies.append(("SessionEdge", _signal_session_edge, {"tag": "SessionEdge"}))

            signals: List[Dict[str, object]] = []
            for name, fn, kwargs in strategies:
                if fn is _signal_htf_pullback:
                    signal = fn(fac_m1, fac_h1, fac_m5, **kwargs)
                elif fn in (_signal_macd_trend, _signal_ema_slope_pull):
                    signal = fn(fac_m1, fac_m5, **kwargs)
                elif fn is _signal_session_edge:
                    signal = fn(fac_m1, range_ctx, now_utc=now, **kwargs)
                elif fn in (
                    _signal_spread_revert,
                    _signal_vwap_revert,
                    _signal_stoch_bounce,
                    _signal_divergence_revert,
                    _signal_tick_imbalance,
                ):
                    signal = fn(fac_m1, range_ctx, **kwargs)
                else:
                    signal = fn(fac_m1, **kwargs)
                if signal:
                    signal = adjust_signal(signal, air)
                    if signal:
                        conf = int(signal.get("confidence", 0) or 0)
                        if config.MIN_ENTRY_CONF > 0 and conf < config.MIN_ENTRY_CONF:
                            continue
                        if not bypass_common_guard:
                            tag = str(signal.get("tag") or "").strip()
                            if tag:
                                pocket_decision = perf_guard.is_pocket_allowed(config.POCKET)
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
                                perf_decision = perf_guard.is_allowed(tag, config.POCKET, hour=now.hour)
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
                continue

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
    asyncio.run(scalp_precision_worker())


if __name__ == "__main__":
    main()
