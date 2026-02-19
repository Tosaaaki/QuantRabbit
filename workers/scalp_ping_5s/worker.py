"""Entry-only 5s ping scalp worker.

This worker focuses on short-horizon entries and leaves exits to broker TP/SL
(or separate exit workers) by design.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import math
import pathlib
import sqlite3
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Sequence

import httpx

from analysis.technique_engine import evaluate_entry_techniques
from execution.strategy_entry import (
    get_last_order_status_by_client_id,
    market_order,
)
from execution.position_manager import PositionManager
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from indicators.factor_cache import all_factors, get_candles_snapshot
from market_data import spread_monitor, tick_window
from market_data.tick_fetcher import _parse_time
from utils.market_hours import is_market_open, seconds_until_open
from utils.oanda_account import get_account_snapshot
from utils.secrets import get_secret
from workers.common.exit_utils import close_trade
from workers.common.rate_limiter import SlidingWindowRateLimiter
from workers.common.size_utils import scale_base_units
from workers.common.tick_lookahead_edge import decide_tick_lookahead_edge

from . import config


LOG = logging.getLogger(__name__)


try:
    _OANDA_TOKEN = get_secret("oanda_token")
    _OANDA_ACCOUNT = get_secret("oanda_account_id")
    try:
        _OANDA_PRACTICE = get_secret("oanda_practice").lower() == "true"
    except Exception:
        _OANDA_PRACTICE = False
except Exception:
    _OANDA_TOKEN = ""
    _OANDA_ACCOUNT = ""
    _OANDA_PRACTICE = False

_PRICING_HOST = (
    "https://api-fxpractice.oanda.com"
    if _OANDA_PRACTICE
    else "https://api-fxtrade.oanda.com"
)
_PRICING_URL = f"{_PRICING_HOST}/v3/accounts/{_OANDA_ACCOUNT}/pricing"
_PRICING_HEADERS = {"Authorization": f"Bearer {_OANDA_TOKEN}"} if _OANDA_TOKEN else {}
_TRADES_DB = pathlib.Path("logs/trades.db")
_SNAPSHOT_FETCH_BACKOFF_UNTIL_MONO: float = 0.0
_SNAPSHOT_FETCH_FAILURES: int = 0
_SNAPSHOT_FETCH_BACKOFF_LOG_MONO: float = 0.0
_SNAPSHOT_AUTH_VALIDATED: Optional[bool] = None
_ENTRY_SKIP_SUMMARY_INTERVAL_SEC: float = 30.0
_POSITION_MANAGER_OPEN_POSITIONS_TIMEOUT_SEC: float = 6.0
_LOOP_HEARTBEAT_INTERVAL_SEC: float = 120.0
_LOOP_EXCEPTION_RECOVERY_SLEEP_SEC: float = 0.5
_JST_TIMEZONE = datetime.timezone(datetime.timedelta(hours=9))


@dataclass(slots=True)
class TpTimingProfile:
    multiplier: float = 1.0
    avg_tp_sec: float = 0.0
    sample: int = 0


_TP_TIMING_CACHE: dict[tuple[str, str], tuple[float, TpTimingProfile]] = {}
_TRADE_MFE_PIPS: dict[str, float] = {}
_TRADE_FORCE_EXIT_DEFER_LOG_MONO: dict[str, float] = {}
_PROFIT_BANK_STATS_CACHE: dict[tuple[str, str, str, str], tuple[float, tuple[float, float, float]]] = {}
_PROFIT_BANK_START_CACHE: tuple[str, Optional[datetime.datetime]] = ("", None)
_PROFIT_BANK_LAST_CLOSE_MONO: float = 0.0
_SIGNAL_WINDOW_STATS_CACHE: dict[
    tuple[str, str],
    tuple[float, list[dict[str, object]]],
] = {}
_SIGNAL_WINDOW_SHADOW_LOG_MONO: float = 0.0
_LAST_FAST_FLIP_MONO: float = 0.0
_LAST_SIDE_METRICS_FLIP_MONO: float = 0.0
_SL_STREAK_CACHE: dict[tuple[str, str], tuple[float, Optional["StopLossStreak"]]] = {}
_SL_METRICS_CACHE: dict[tuple[str, str], tuple[float, Optional["SideCloseMetrics"]]] = {}
_SIDE_CLOSE_METRICS_ALLOC_CACHE: dict[
    tuple[str, str],
    tuple[float, Optional["SideCloseMetrics"]],
] = {}
_SIDE_CLOSE_METRICS_FLIP_CACHE: dict[
    tuple[str, str],
    tuple[float, Optional["SideCloseMetrics"]],
] = {}
_ENTRY_PROB_BAND_METRICS_CACHE: dict[
    tuple[str, str, str],
    tuple[float, Optional["EntryProbabilityBandMetrics"]],
] = {}


def _mask_value(value: str, *, head: int = 4, tail: int = 2) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if len(text) <= head + tail:
        return "*" * len(text)
    return f"{text[:head]}***{text[-tail:]}"


def _entry_blocked_hour_jst(now_utc: datetime.datetime) -> Optional[int]:
    block_hours_raw = getattr(config, "BLOCK_HOURS_JST", ())
    block_hours: tuple[int, ...] = ()

    try:
        if isinstance(block_hours_raw, int):
            block_hours = (block_hours_raw,)
        elif isinstance(block_hours_raw, str):
            block_hours = tuple(
                int(item.strip())
                for item in block_hours_raw.replace("\n", ",").split(",")
                if item.strip()
            )
        else:
            parsed: list[int] = []
            for value in block_hours_raw:
                try:
                    parsed.append(int(str(value).strip()))
                except (TypeError, ValueError):
                    continue
            block_hours = tuple(parsed)
    except Exception:
        block_hours = ()

    if not block_hours:
        return None
    jst_hour = now_utc.astimezone(_JST_TIMEZONE).hour
    if jst_hour in block_hours:
        return jst_hour
    return None


@dataclass(slots=True)
class TickSignal:
    side: str
    mode: str
    mode_score: float
    momentum_score: float
    revert_score: float
    confidence: int
    momentum_pips: float
    trigger_pips: float
    imbalance: float
    tick_rate: float
    span_sec: float
    tick_age_ms: float
    spread_pips: float
    bid: float
    ask: float
    mid: float
    range_pips: float = 0.0
    instant_range_pips: float = 0.0
    signal_window_sec: float = 0.0


@dataclass(slots=True)
class DirectionBias:
    side: str
    score: float
    momentum_pips: float
    flow: float
    range_pips: float
    vol_norm: float
    tick_rate: float
    span_sec: float


@dataclass(slots=True)
class StopLossStreak:
    side: str
    streak: int
    age_sec: float
    latest_close_time: Optional[datetime.datetime]
    sample: int


@dataclass(slots=True)
class SideCloseMetrics:
    long_sl_hits: int
    short_sl_hits: int
    long_market_plus: int
    short_market_plus: int
    long_trades: int
    short_trades: int
    sample: int

    def sl_hits(self, side: str) -> int:
        key = str(side or "").strip().lower()
        if key == "long":
            return int(self.long_sl_hits)
        if key == "short":
            return int(self.short_sl_hits)
        return 0

    def market_plus(self, side: str) -> int:
        key = str(side or "").strip().lower()
        if key == "long":
            return int(self.long_market_plus)
        if key == "short":
            return int(self.short_market_plus)
        return 0


@dataclass(slots=True)
class EntryProbabilityBandMetrics:
    side: str
    sample: int
    high_sample: int
    high_mean_pips: float
    high_win_rate: float
    high_sl_rate: float
    low_sample: int
    low_mean_pips: float
    low_win_rate: float
    low_sl_rate: float


@dataclass(slots=True)
class SlFlipEval:
    streak: Optional[StopLossStreak]
    side_sl_hits_recent: int
    target_market_plus_recent: int
    direction_confirmed: bool
    horizon_confirmed: bool


@dataclass(slots=True)
class SideMetricsFlipEval:
    current_side: str
    target_side: str
    current_trades: int
    target_trades: int
    current_sl_rate: float
    target_sl_rate: float
    current_market_plus_rate: float
    target_market_plus_rate: float


@dataclass(slots=True)
class MtfRegime:
    side: str
    mode: str
    trend_score: float
    heat_score: float
    adx_m1: float
    adx_m5: float
    atr_m1: float
    atr_m5: float


@dataclass(slots=True)
class FibPullback:
    timeframe: str
    swing_high: float
    swing_low: float
    zone_low: float
    zone_high: float
    target_price: float
    recover_pips: float
    range_pips: float


@dataclass(slots=True)
class HorizonBias:
    long_side: str
    long_score: float
    mid_side: str
    mid_score: float
    short_side: str
    short_score: float
    micro_side: str
    micro_score: float
    composite_side: str
    composite_score: float
    agreement: int


@dataclass(slots=True)
class ExtremaGateDecision:
    allow_entry: bool
    reason: str
    units_mult: float
    m1_pos: Optional[float]
    m5_pos: Optional[float]
    h4_pos: Optional[float]


@dataclass(slots=True)
class TechnicalTradeProfile:
    tp_mult: float
    sl_mult: float
    hold_mult: float
    hard_loss_mult: float
    counter_pressure: bool
    route_reasons: tuple[str, ...]


@dataclass(slots=True)
class TrapState:
    active: bool
    long_units: float
    short_units: float
    net_ratio: float
    long_dd_pips: float
    short_dd_pips: float
    combined_dd_pips: float
    unrealized_pl: float


@dataclass(slots=True)
class ProfitBankCandidate:
    trade_id: str
    client_id: str
    units: int
    opened_at: datetime.datetime
    hold_sec: float
    unrealized_pl: float
    loss_jpy: float


def _load_tp_timing_profile(strategy_tag: str, pocket: str) -> TpTimingProfile:
    if not config.TP_TIME_ADAPT_ENABLED:
        return TpTimingProfile()

    key = ((strategy_tag or "").strip().lower(), (pocket or "").strip().lower())
    now_mono = time.monotonic()
    cached = _TP_TIMING_CACHE.get(key)
    if cached and now_mono - cached[0] <= config.TP_HOLD_STATS_TTL_SEC:
        return cached[1]

    profile = TpTimingProfile()
    if not _TRADES_DB.exists():
        _TP_TIMING_CACHE[key] = (now_mono, profile)
        return profile

    lookback_minutes = max(1, int(round(config.TP_HOLD_LOOKBACK_HOURS * 60.0)))
    lookback_expr = f"-{lookback_minutes} minutes"
    tag_key = key[0]
    pocket_key = key[1]

    try:
        con = sqlite3.connect(_TRADES_DB)
        row = con.execute(
            """
            SELECT
              COUNT(*) AS n,
              AVG((julianday(close_time) - julianday(open_time)) * 86400.0) AS avg_hold_sec
            FROM trades
            WHERE close_time IS NOT NULL
              AND close_time >= datetime('now', ?)
              AND lower(coalesce(strategy_tag, '')) = ?
              AND lower(coalesce(pocket, '')) = ?
              AND close_reason = 'TAKE_PROFIT_ORDER'
            """,
            (lookback_expr, tag_key, pocket_key),
        ).fetchone()
    except Exception:
        row = None
    finally:
        try:
            con.close()
        except Exception:
            pass

    try:
        sample = int(row[0] or 0) if row else 0
    except Exception:
        sample = 0
    try:
        avg_tp_sec = float(row[1] or 0.0) if row else 0.0
    except Exception:
        avg_tp_sec = 0.0

    multiplier = 1.0
    if sample >= config.TP_HOLD_MIN_TRADES and avg_tp_sec > 0.0:
        ratio = config.TP_TARGET_HOLD_SEC / max(1.0, avg_tp_sec)
        multiplier = ratio ** 0.5
    multiplier = max(config.TP_TIME_MULT_MIN, min(config.TP_TIME_MULT_MAX, multiplier))

    profile = TpTimingProfile(
        multiplier=float(multiplier),
        avg_tp_sec=max(0.0, float(avg_tp_sec)),
        sample=max(0, sample),
    )
    _TP_TIMING_CACHE[key] = (now_mono, profile)
    return profile


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _lerp(a: float, b: float, weight: float) -> float:
    return a + (b - a) * _clamp(weight, 0.0, 1.0)


def _norm01(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low), 0.0, 1.0)


def _signal_window_bucket(window_sec: float) -> float:
    step = max(0.01, _safe_float(getattr(config, "SIGNAL_WINDOW_ADAPTIVE_BUCKET_SEC", 0.05), 0.05))
    value = max(0.0, _safe_float(window_sec, 0.0))
    return round(round(value / step) * step, 4)


def _load_signal_window_stats(*, strategy_tag: str, pocket: str) -> list[dict[str, object]]:
    cache_key = (
        str(strategy_tag or "").strip().lower(),
        str(pocket or "").strip().lower(),
    )
    now_mono = time.monotonic()
    cached = _SIGNAL_WINDOW_STATS_CACHE.get(cache_key)
    if cached and now_mono - cached[0] <= config.SIGNAL_WINDOW_ADAPTIVE_STATS_TTL_SEC:
        return cached[1]

    rows: list[dict[str, object]] = []
    if not _TRADES_DB.exists():
        _SIGNAL_WINDOW_STATS_CACHE[cache_key] = (now_mono, rows)
        return rows

    con: Optional[sqlite3.Connection] = None
    lookback_minutes = max(
        1,
        int(round(max(1.0, config.SIGNAL_WINDOW_ADAPTIVE_LOOKBACK_HOURS) * 60.0)),
    )
    lookback_expr = f"-{lookback_minutes} minutes"
    try:
        con = sqlite3.connect(_TRADES_DB)
        fetched = con.execute(
            """
            SELECT
              CAST(json_extract(entry_thesis, '$.signal_window_sec') AS REAL) AS signal_window_sec,
              lower(COALESCE(json_extract(entry_thesis, '$.signal_mode'), 'unknown')) AS signal_mode,
              CASE WHEN units >= 0 THEN 'long' ELSE 'short' END AS side,
              COUNT(*) AS sample,
              AVG(pl_pips) AS mean_pips,
              AVG(CASE WHEN pl_pips > 0 THEN 1.0 ELSE 0.0 END) AS win_rate
            FROM trades
            WHERE close_time IS NOT NULL
              AND close_time >= datetime('now', ?)
              AND lower(COALESCE(strategy_tag, '')) = ?
              AND lower(COALESCE(pocket, '')) = ?
              AND pl_pips IS NOT NULL
              AND json_type(entry_thesis, '$.signal_window_sec') IN ('real', 'integer')
            GROUP BY 1, 2, 3
            """,
            (lookback_expr, cache_key[0], cache_key[1]),
        ).fetchall()
    except Exception:
        fetched = []
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass

    for raw_window, raw_mode, raw_side, raw_sample, raw_mean, raw_win in fetched:
        try:
            window_sec = float(raw_window)
            sample = int(raw_sample)
            mean_pips = float(raw_mean)
            win_rate = float(raw_win)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(window_sec) or window_sec <= 0.0:
            continue
        if sample <= 0:
            continue
        mode = str(raw_mode or "unknown").strip().lower() or "unknown"
        side = str(raw_side or "").strip().lower()
        if side not in {"long", "short"}:
            continue
        rows.append(
            {
                "window_sec": float(window_sec),
                "window_bucket": _signal_window_bucket(window_sec),
                "mode": mode,
                "side": side,
                "sample": sample,
                "mean_pips": float(mean_pips),
                "win_rate": _clamp(float(win_rate), 0.0, 1.0),
            }
        )

    _SIGNAL_WINDOW_STATS_CACHE[cache_key] = (now_mono, rows)
    return rows


def _candidate_signal_windows(
    *,
    live_window_sec: float,
    speed_scale: float,
) -> list[float]:
    windows = [max(0.3, _safe_float(live_window_sec, 0.3))]
    for raw in config.SIGNAL_WINDOW_ADAPTIVE_CANDIDATES_SEC:
        candidate = _safe_float(raw, 0.0)
        if candidate <= 0.0:
            continue
        if config.SIGNAL_WINDOW_ADAPTIVE_SCALE_WITH_SPEED:
            candidate *= max(0.1, speed_scale)
        candidate = _clamp(
            candidate,
            config.SIGNAL_WINDOW_ADAPTIVE_MIN_SEC,
            config.SIGNAL_WINDOW_ADAPTIVE_MAX_SEC,
        )
        windows.append(max(0.3, candidate))

    deduped: list[float] = []
    for value in windows:
        rounded = round(float(value), 4)
        if rounded not in deduped:
            deduped.append(rounded)

    base = deduped[0] if deduped else max(0.3, _safe_float(live_window_sec, 0.3))
    deduped.sort(key=lambda item: (abs(item - base), item))
    return deduped


def _score_signal_window_candidate(
    *,
    candidate: TickSignal,
    stats_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    target_bucket = _signal_window_bucket(candidate.signal_window_sec)
    tol = max(0.0, _safe_float(config.SIGNAL_WINDOW_ADAPTIVE_MATCH_TOL_SEC, 0.35))
    min_trades = max(1, int(_safe_float(config.SIGNAL_WINDOW_ADAPTIVE_MIN_TRADES, 30)))
    bonus = max(0.0, _safe_float(config.SIGNAL_WINDOW_ADAPTIVE_UCB_BONUS_PIPS, 0.0))
    cold_penalty = max(
        0.0,
        _safe_float(config.SIGNAL_WINDOW_ADAPTIVE_COLDSTART_PENALTY_PIPS, 0.0),
    )

    best_row: Optional[dict[str, object]] = None
    best_key: tuple[int, float, int] | None = None
    for row in stats_rows:
        row_bucket = _safe_float(row.get("window_bucket"), -1.0)
        if row_bucket <= 0.0:
            continue
        distance = abs(row_bucket - target_bucket)
        if distance > tol:
            continue
        side = str(row.get("side") or "").strip().lower()
        mode = str(row.get("mode") or "unknown").strip().lower() or "unknown"
        side_match = side == candidate.side
        mode_match = mode == candidate.mode
        priority = 0
        if side_match and mode_match:
            priority = 0
        elif side_match:
            priority = 1
        elif mode_match:
            priority = 2
        else:
            priority = 3
        sample = int(_safe_float(row.get("sample"), 0.0))
        key = (priority, distance, -sample)
        if best_key is None or key < best_key:
            best_key = key
            best_row = row

    if best_row is None:
        return {
            "score_pips": 0.0,
            "mean_pips": 0.0,
            "sample": 0,
            "win_rate": 0.5,
            "source": "no_stats",
            "bucket": target_bucket,
        }

    sample = max(0, int(_safe_float(best_row.get("sample"), 0.0)))
    mean_pips = _safe_float(best_row.get("mean_pips"), 0.0)
    win_rate = _clamp(_safe_float(best_row.get("win_rate"), 0.5), 0.0, 1.0)
    if sample > 0:
        score = mean_pips + (bonus / math.sqrt(sample))
    else:
        score = mean_pips
    if sample < min_trades:
        score -= cold_penalty
    return {
        "score_pips": float(score),
        "mean_pips": float(mean_pips),
        "sample": int(sample),
        "win_rate": float(win_rate),
        "source": str(best_row.get("mode") or "unknown"),
        "bucket": target_bucket,
    }


def _maybe_adapt_signal_window(
    *,
    ticks: Sequence[dict],
    spread_pips: float,
    base_signal: TickSignal,
) -> tuple[TickSignal, dict[str, object]]:
    adaptive_enabled = bool(config.SIGNAL_WINDOW_ADAPTIVE_ENABLED)
    shadow_enabled = bool(config.SIGNAL_WINDOW_ADAPTIVE_SHADOW_ENABLED)
    if not adaptive_enabled and not shadow_enabled:
        return base_signal, {}

    stats_rows = _load_signal_window_stats(
        strategy_tag=config.STRATEGY_TAG,
        pocket=config.POCKET,
    )
    speed_scale = _instant_speed_scale(base_signal.instant_range_pips)
    windows = _candidate_signal_windows(
        live_window_sec=base_signal.signal_window_sec,
        speed_scale=speed_scale,
    )

    candidates: list[dict[str, object]] = [
        {
            "window_sec": float(base_signal.signal_window_sec),
            "signal": base_signal,
            "reason": "live",
        }
    ]
    skipped: list[str] = []
    for window_sec in windows:
        if abs(window_sec - base_signal.signal_window_sec) < 1e-6:
            continue
        alt_signal, alt_reason = _build_tick_signal(
            ticks,
            spread_pips,
            signal_window_override_sec=float(window_sec),
            allow_window_fallback=False,
        )
        if alt_signal is None:
            skipped.append(f"{window_sec:.2f}:{alt_reason}")
            continue
        candidates.append(
            {
                "window_sec": float(alt_signal.signal_window_sec),
                "signal": alt_signal,
                "reason": str(alt_reason),
            }
        )

    scored: list[dict[str, object]] = []
    for candidate in candidates:
        signal = candidate["signal"]
        if not isinstance(signal, TickSignal):
            continue
        metrics = _score_signal_window_candidate(candidate=signal, stats_rows=stats_rows)
        scored.append({**candidate, **metrics})
    if not scored:
        return base_signal, {}

    live_candidate = next(
        (item for item in scored if item.get("reason") == "live"),
        scored[0],
    )
    best_candidate = max(
        scored,
        key=lambda item: (
            _safe_float(item.get("score_pips"), 0.0),
            _safe_float(item.get("mean_pips"), 0.0),
            _safe_float(item.get("sample"), 0.0),
        ),
    )

    margin = max(0.0, _safe_float(config.SIGNAL_WINDOW_ADAPTIVE_SELECTION_MARGIN_PIPS, 0.0))
    min_trades = max(1, int(_safe_float(config.SIGNAL_WINDOW_ADAPTIVE_MIN_TRADES, 30)))
    improvement = _safe_float(best_candidate.get("score_pips"), 0.0) - _safe_float(
        live_candidate.get("score_pips"),
        0.0,
    )
    should_apply = (
        adaptive_enabled
        and isinstance(best_candidate.get("signal"), TickSignal)
        and best_candidate is not live_candidate
        and int(_safe_float(best_candidate.get("sample"), 0.0)) >= min_trades
        and improvement >= margin
    )

    selected_candidate = best_candidate if should_apply else live_candidate
    selected_signal = selected_candidate.get("signal")
    if not isinstance(selected_signal, TickSignal):
        selected_signal = base_signal

    global _SIGNAL_WINDOW_SHADOW_LOG_MONO
    now_mono = time.monotonic()
    if shadow_enabled and (now_mono - _SIGNAL_WINDOW_SHADOW_LOG_MONO) >= config.SIGNAL_WINDOW_ADAPTIVE_SHADOW_LOG_INTERVAL_SEC:
        top = sorted(
            scored,
            key=lambda item: _safe_float(item.get("score_pips"), 0.0),
            reverse=True,
        )[:3]
        top_summary = ",".join(
            (
                f"{_safe_float(item.get('window_sec'), 0.0):.2f}s:"
                f"s={_safe_float(item.get('score_pips'), 0.0):+.3f}/"
                f"m={_safe_float(item.get('mean_pips'), 0.0):+.3f}/"
                f"n={int(_safe_float(item.get('sample'), 0.0))}"
            )
            for item in top
        )
        LOG.info(
            "%s signal_window shadow live=%.2fs best=%.2fs selected=%.2fs apply=%s improve=%.3f top=[%s] skipped=%s",
            config.LOG_PREFIX,
            _safe_float(base_signal.signal_window_sec, 0.0),
            _safe_float(best_candidate.get("window_sec"), 0.0),
            _safe_float(selected_signal.signal_window_sec, 0.0),
            should_apply,
            improvement,
            top_summary or "-",
            ";".join(skipped[:6]) if skipped else "-",
        )
        _SIGNAL_WINDOW_SHADOW_LOG_MONO = now_mono

    meta = {
        "enabled": adaptive_enabled,
        "shadow_enabled": shadow_enabled,
        "applied": should_apply,
        "live_window_sec": round(_safe_float(base_signal.signal_window_sec, 0.0), 3),
        "selected_window_sec": round(_safe_float(selected_signal.signal_window_sec, 0.0), 3),
        "best_window_sec": round(_safe_float(best_candidate.get("window_sec"), 0.0), 3),
        "live_score_pips": round(_safe_float(live_candidate.get("score_pips"), 0.0), 5),
        "selected_score_pips": round(_safe_float(selected_candidate.get("score_pips"), 0.0), 5),
        "best_score_pips": round(_safe_float(best_candidate.get("score_pips"), 0.0), 5),
        "best_sample": int(_safe_float(best_candidate.get("sample"), 0.0)),
        "candidate_count": len(scored),
    }
    return selected_signal, meta


def _instant_speed_scale(instant_range_pips: float) -> float:
    if not config.INSTANT_SPEED_ENABLED:
        return 1.0
    speed_bucket = _norm01(
        float(instant_range_pips),
        config.INSTANT_SPEED_VOL_LOW_PIPS,
        config.INSTANT_SPEED_VOL_HIGH_PIPS,
    )
    return _lerp(1.0, config.INSTANT_SPEED_SCALE_MIN, speed_bucket)


def _regime_vol_bucket(
    regime: Optional[MtfRegime],
    *,
    low_pips: float,
    high_pips: float,
) -> float:
    if regime is None:
        return 0.5
    atr_m1 = _safe_float(regime.atr_m1, 0.0)
    atr_m5 = _safe_float(regime.atr_m5, 0.0)
    atr = max(0.0, max(atr_m1, atr_m5))
    if atr <= 0.0:
        return 0.5
    return _norm01(atr, low_pips, high_pips)


def _trend_vol_bucket(
    atr_pips: float,
    *,
    low_pips: float,
    high_pips: float,
) -> float:
    return _norm01(_safe_float(atr_pips, 0.0), low_pips, high_pips)


def _tf_trend_score(fac: dict) -> float:
    close = _safe_float(fac.get("close"), 0.0)
    atr_pips = max(0.1, _safe_float(fac.get("atr_pips"), 0.0))
    ema_slow = _safe_float(fac.get("ema20"), _safe_float(fac.get("ma20"), 0.0))
    ema_fast = _safe_float(fac.get("ema12"), ema_slow)
    if ema_fast <= 0.0:
        ema_fast = ema_slow
    if not config.MTF_TREND_EMA_VOL_INTERP_ENABLED:
        trend_ema = ema_slow
        gap_scale = config.MTF_TREND_GAP_SCALE_LOW_VOL
    else:
        vol_bucket = _trend_vol_bucket(
            atr_pips,
            low_pips=config.MTF_TREND_EMA_VOL_LOW_PIPS,
            high_pips=config.MTF_TREND_EMA_VOL_HIGH_PIPS,
        )
        fast_weight = _lerp(
            config.MTF_TREND_EMA_FAST_WEIGHT_LOW_VOL,
            config.MTF_TREND_EMA_FAST_WEIGHT_HIGH_VOL,
            vol_bucket,
        )
        trend_ema = _lerp(ema_slow, ema_fast, fast_weight)
        gap_scale = _lerp(
            config.MTF_TREND_GAP_SCALE_LOW_VOL,
            config.MTF_TREND_GAP_SCALE_HIGH_VOL,
            vol_bucket,
        )

    rsi = _safe_float(fac.get("rsi"), 50.0)
    macd_hist = _safe_float(fac.get("macd_hist"), 0.0)

    gap_norm = 0.0
    if close > 0.0 and trend_ema > 0.0:
        gap_pips = (close - trend_ema) / config.PIP_VALUE
        gap_norm = _clamp(
            gap_pips / max(config.MTF_TREND_GAP_NORM_MIN_PIPS, atr_pips * gap_scale),
            -1.0,
            1.0,
        )

    rsi_norm = _clamp((rsi - 50.0) / 20.0, -1.0, 1.0)
    macd_pips = macd_hist / config.PIP_VALUE
    macd_norm = _clamp(macd_pips / max(0.6, atr_pips * 0.15), -1.0, 1.0)
    return (0.58 * gap_norm) + (0.27 * rsi_norm) + (0.15 * macd_norm)


def _build_mtf_regime(factors: Optional[dict] = None) -> Optional[MtfRegime]:
    if not config.MTF_REGIME_ENABLED:
        return None
    if factors is None:
        try:
            factors = all_factors()
        except Exception:
            return None
    if not isinstance(factors, dict):
        return None

    fac_m1 = factors.get("M1") or {}
    fac_m5 = factors.get("M5") or {}
    fac_h1 = factors.get("H1") or {}
    if not any(isinstance(f, dict) and f for f in (fac_m1, fac_m5, fac_h1)):
        return None

    score_h1 = _tf_trend_score(fac_h1 if isinstance(fac_h1, dict) else {})
    score_m5 = _tf_trend_score(fac_m5 if isinstance(fac_m5, dict) else {})
    score_m1 = _tf_trend_score(fac_m1 if isinstance(fac_m1, dict) else {})
    trend_score = _clamp(
        (0.55 * score_h1) + (0.30 * score_m5) + (0.15 * score_m1),
        -1.0,
        1.0,
    )

    adx_m1 = _safe_float((fac_m1 or {}).get("adx"), 0.0)
    adx_m5 = _safe_float((fac_m5 or {}).get("adx"), 0.0)
    atr_m1 = _safe_float((fac_m1 or {}).get("atr_pips"), 0.0)
    atr_m5 = _safe_float((fac_m5 or {}).get("atr_pips"), 0.0)

    adx_mix = (0.6 * adx_m1) + (0.4 * adx_m5)
    adx_norm = _norm01(adx_mix, config.MTF_ADX_LOW, config.MTF_ADX_HIGH)
    atr_norm_m1 = _norm01(atr_m1, config.MTF_ATR_M1_LOW_PIPS, config.MTF_ATR_M1_HIGH_PIPS)
    atr_norm_m5 = _norm01(atr_m5, config.MTF_ATR_M5_LOW_PIPS, config.MTF_ATR_M5_HIGH_PIPS)
    atr_norm = (0.55 * atr_norm_m1) + (0.45 * atr_norm_m5)
    heat_score = _clamp((0.60 * adx_norm) + (0.40 * atr_norm), 0.0, 1.0)

    trend_abs = abs(trend_score)
    if trend_score >= config.MTF_TREND_NEUTRAL_SCORE:
        side = "long"
    elif trend_score <= -config.MTF_TREND_NEUTRAL_SCORE:
        side = "short"
    else:
        side = "neutral"

    if (
        trend_abs >= config.MTF_TREND_STRONG_SCORE
        and heat_score >= config.MTF_HEAT_CONTINUATION_MIN
    ):
        mode = "continuation"
    elif heat_score <= config.MTF_HEAT_REVERSION_MAX:
        mode = "reversion"
    else:
        mode = "balanced"

    return MtfRegime(
        side=side,
        mode=mode,
        trend_score=float(trend_score),
        heat_score=float(heat_score),
        adx_m1=float(adx_m1),
        adx_m5=float(adx_m5),
        atr_m1=float(atr_m1),
        atr_m5=float(atr_m5),
    )


def _trade_side_from_units(units: int) -> Optional[str]:
    if units > 0:
        return "long"
    if units < 0:
        return "short"
    return None


def _trade_entry_price(trade: dict) -> Optional[float]:
    entry = _safe_float(
        trade.get("price")
        or trade.get("entry_price")
        or trade.get("open_price"),
        0.0,
    )
    return entry if entry > 0.0 else None


def _trade_mark_price_from_unrealized(
    *,
    entry_price: float,
    units: int,
    unrealized_pips: float,
) -> float:
    if units > 0:
        return entry_price + unrealized_pips * config.PIP_VALUE
    return entry_price - unrealized_pips * config.PIP_VALUE


def _extract_tf_candles(
    factors: dict,
    timeframe: str,
    *,
    limit: int,
) -> list[dict]:
    tf = factors.get(timeframe) if isinstance(factors, dict) else None
    if not isinstance(tf, dict):
        return []
    candles = tf.get("candles")
    if not isinstance(candles, list) or len(candles) < 5:
        return []
    cleaned: list[dict] = []
    for row in candles[-max(5, int(limit)):]:
        if not isinstance(row, dict):
            continue
        high = _safe_float(row.get("high"), 0.0)
        low = _safe_float(row.get("low"), 0.0)
        if high <= 0.0 or low <= 0.0 or high < low:
            continue
        cleaned.append(row)
    return cleaned


def _build_fib_pullback(
    *,
    side: str,
    current_price: float,
    factors: dict,
) -> Optional[FibPullback]:
    if side not in {"long", "short"} or current_price <= 0.0:
        return None

    candidates: list[FibPullback] = []
    tf_cfg = (
        ("M5", config.FORCE_EXIT_MTF_FIB_LOOKBACK_M5),
        ("H1", config.FORCE_EXIT_MTF_FIB_LOOKBACK_H1),
    )
    for tf, lookback in tf_cfg:
        rows = _extract_tf_candles(factors, tf, limit=lookback)
        if len(rows) < 5:
            continue
        highs = [_safe_float(r.get("high"), 0.0) for r in rows]
        lows = [_safe_float(r.get("low"), 0.0) for r in rows]
        swing_high = max(highs) if highs else 0.0
        swing_low = min(lows) if lows else 0.0
        if swing_high <= 0.0 or swing_low <= 0.0 or swing_high <= swing_low:
            continue
        range_pips = (swing_high - swing_low) / config.PIP_VALUE
        if range_pips < config.FORCE_EXIT_MTF_FIB_MIN_RANGE_PIPS:
            continue

        width = swing_high - swing_low
        lower = _clamp(config.FORCE_EXIT_MTF_FIB_LOWER, 0.0, 1.0)
        upper = _clamp(config.FORCE_EXIT_MTF_FIB_UPPER, lower, 1.0)
        if side == "long":
            zone_low = swing_low + width * lower
            zone_high = swing_low + width * upper
            target_price = zone_low if current_price <= zone_low else zone_high
            recover_pips = (target_price - current_price) / config.PIP_VALUE
        else:
            zone_high = swing_high - width * lower
            zone_low = swing_high - width * upper
            target_price = zone_high if current_price >= zone_high else zone_low
            recover_pips = (current_price - target_price) / config.PIP_VALUE
        if recover_pips <= 0.0:
            continue
        candidates.append(
            FibPullback(
                timeframe=tf,
                swing_high=float(swing_high),
                swing_low=float(swing_low),
                zone_low=float(min(zone_low, zone_high)),
                zone_high=float(max(zone_low, zone_high)),
                target_price=float(target_price),
                recover_pips=float(recover_pips),
                range_pips=float(range_pips),
            )
        )

    if not candidates:
        return None
    return min(candidates, key=lambda x: x.recover_pips)


def _is_adverse_momentum(
    *,
    side: str,
    regime: Optional[MtfRegime],
    factors: dict,
) -> bool:
    if side not in {"long", "short"}:
        return False
    if regime is not None and regime.side in {"long", "short"} and regime.side != side:
        if regime.heat_score >= config.FORCE_EXIT_MTF_FIB_OPPOSITE_HEAT_BLOCK:
            return True

    fac_m1 = factors.get("M1") if isinstance(factors, dict) else {}
    if not isinstance(fac_m1, dict):
        return False
    close = _safe_float(fac_m1.get("close"), 0.0)
    ema20 = _safe_float(fac_m1.get("ema20"), 0.0)
    rsi = _safe_float(fac_m1.get("rsi"), 50.0)
    if close <= 0.0 or ema20 <= 0.0:
        return False

    ema_gap_pips = abs(close - ema20) / config.PIP_VALUE
    if ema_gap_pips < config.FORCE_EXIT_MTF_FIB_OPPOSITE_EMA_GAP_PIPS:
        return False

    if side == "long":
        return close < ema20 and rsi <= config.FORCE_EXIT_MTF_FIB_OPPOSITE_RSI
    return close > ema20 and rsi >= (100.0 - config.FORCE_EXIT_MTF_FIB_OPPOSITE_RSI)


def _should_defer_force_exit(
    *,
    trade: dict,
    units: int,
    hold_sec: float,
    unrealized_pips: float,
    exit_reason: str,
) -> tuple[bool, str, Optional[FibPullback], Optional[MtfRegime]]:
    if not config.FORCE_EXIT_MTF_FIB_HOLD_ENABLED:
        return False, "disabled", None, None
    if exit_reason not in {config.FORCE_EXIT_REASON, config.FORCE_EXIT_RECOVERY_REASON}:
        return False, "reason_not_eligible", None, None
    if unrealized_pips >= 0.0:
        return False, "non_loss", None, None
    if hold_sec > config.FORCE_EXIT_MTF_FIB_MAX_WAIT_SEC:
        return False, "max_wait_elapsed", None, None
    if abs(unrealized_pips) > config.FORCE_EXIT_MTF_FIB_MAX_HOLD_LOSS_PIPS:
        return False, "loss_too_deep", None, None

    side = _trade_side_from_units(units)
    if side is None:
        return False, "units_zero", None, None
    entry_price = _trade_entry_price(trade)
    if entry_price is None:
        return False, "entry_missing", None, None
    current_price = _trade_mark_price_from_unrealized(
        entry_price=entry_price,
        units=units,
        unrealized_pips=unrealized_pips,
    )
    if current_price <= 0.0:
        return False, "mark_missing", None, None

    try:
        factors = all_factors()
    except Exception:
        return False, "factors_unavailable", None, None
    if not isinstance(factors, dict):
        return False, "factors_invalid", None, None
    regime = _build_mtf_regime(factors=factors)
    if _is_adverse_momentum(side=side, regime=regime, factors=factors):
        return False, "adverse_momentum", None, regime

    fib = _build_fib_pullback(side=side, current_price=current_price, factors=factors)
    if fib is None:
        return False, "fib_unavailable", None, regime
    if fib.recover_pips < config.FORCE_EXIT_MTF_FIB_MIN_RECOVER_PIPS:
        return False, "recover_small", fib, regime
    if fib.recover_pips > config.FORCE_EXIT_MTF_FIB_MAX_TARGET_PIPS:
        return False, "target_too_far", fib, regime

    projected_loss = abs(min(0.0, unrealized_pips + fib.recover_pips))
    if projected_loss > config.FORCE_EXIT_MTF_FIB_PROJECTED_MAX_LOSS_PIPS:
        return False, "projected_loss_large", fib, regime
    return True, "fib_wait", fib, regime


def _detect_momentum_stall_for_open_trade(
    *,
    trade_side: str,
    hold_sec: float,
    rows: Sequence[dict],
) -> tuple[bool, float, float]:
    if not config.FORCE_EXIT_MOMENTUM_STALL_ENABLED:
        return False, 0.0, 0.0
    if hold_sec < float(config.FORCE_EXIT_MOMENTUM_STALL_MIN_HOLD_SEC):
        return False, 0.0, 0.0
    if trade_side not in {"long", "short"}:
        return False, 0.0, 0.0

    if _latest_tick_age_ms(rows) > config.FORCE_EXIT_MOMENTUM_STALL_MAX_TICK_AGE_MS:
        return False, 0.0, 0.0

    mids: list[float] = []
    for row in rows:
        _, _, mid = _quotes_from_row(row)
        if mid > 0.0:
            mids.append(mid)
    if len(mids) < config.FORCE_EXIT_MOMENTUM_STALL_MIN_TICKS:
        return False, 0.0, 0.0

    split = max(2, len(mids) // 2)
    first_half = mids[:-split]
    last_half = mids[-split:]
    if len(first_half) < 2 or len(last_half) < 2:
        return False, 0.0, 0.0

    first_delta = (first_half[-1] - first_half[0]) / config.PIP_VALUE
    last_delta = (last_half[-1] - last_half[0]) / config.PIP_VALUE
    first_abs = abs(first_delta)
    flat_cap = max(
        config.FORCE_EXIT_MOMENTUM_STALL_MIN_LATE_PIPS,
        first_abs * config.FORCE_EXIT_MOMENTUM_STALL_FLAT_REMAIN_RATIO,
    )
    if first_abs < config.FORCE_EXIT_MOMENTUM_STALL_MIN_EARLY_PIPS:
        return False, first_delta, last_delta

    if trade_side == "long":
        if (
            first_delta >= config.FORCE_EXIT_MOMENTUM_STALL_MIN_EARLY_PIPS
            and (
                last_delta <= -config.FORCE_EXIT_MOMENTUM_STALL_MIN_LATE_PIPS
                or (0.0 <= last_delta <= flat_cap)
            )
        ):
            return True, first_delta, last_delta
        return False, first_delta, last_delta

    if (
        first_delta <= -config.FORCE_EXIT_MOMENTUM_STALL_MIN_EARLY_PIPS
        and (
            last_delta >= config.FORCE_EXIT_MOMENTUM_STALL_MIN_LATE_PIPS
            or ( -flat_cap <= last_delta <= 0.0 )
        )
    ):
        return True, first_delta, last_delta
    return False, first_delta, last_delta


def _parse_iso_utc(raw: object) -> Optional[datetime.datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        opened_at = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=datetime.timezone.utc)
    return opened_at.astimezone(datetime.timezone.utc)


def _parse_trade_open_time(raw: object) -> Optional[datetime.datetime]:
    return _parse_iso_utc(raw)


def _trade_strategy_tag(trade: dict) -> str:
    raw = trade.get("strategy_tag")
    if not raw:
        thesis = trade.get("entry_thesis")
        if isinstance(thesis, dict):
            raw = thesis.get("strategy_tag") or thesis.get("strategy")
    return str(raw or "").strip()


def _trade_policy_generation(trade: dict) -> str:
    thesis = trade.get("entry_thesis")
    if not isinstance(thesis, dict):
        return ""
    return str(thesis.get("policy_generation") or "").strip()


def _trade_unrealized_pips(trade: dict) -> float:
    direct = _safe_float(trade.get("unrealized_pl_pips"), float("nan"))
    if not math.isnan(direct):
        return direct
    unrealized_pl = _safe_float(trade.get("unrealized_pl"), float("nan"))
    if math.isnan(unrealized_pl):
        return 0.0
    units = abs(int(_safe_float(trade.get("units"), 0.0)))
    pip_value = units * config.PIP_VALUE
    if pip_value <= 0.0:
        return 0.0
    return unrealized_pl / pip_value


def _trade_entry_thesis(trade: dict) -> dict:
    thesis = trade.get("entry_thesis")
    if isinstance(thesis, dict):
        return thesis
    return {}


def _trade_entry_spread_pips(trade: dict) -> float:
    thesis = _trade_entry_thesis(trade)
    return _safe_float(thesis.get("spread_pips"), 0.0)


def _trade_force_exit_max_hold_sec(trade: dict, default_sec: float) -> float:
    if default_sec <= 0.0:
        return 0.0
    thesis = _trade_entry_thesis(trade)
    override_sec = _safe_float(thesis.get("force_exit_max_hold_sec"), 0.0)
    if override_sec <= 0.0:
        return default_sec
    return override_sec


def _trade_force_exit_hard_loss_pips(trade: dict, default_pips: float) -> float:
    if default_pips <= 0.0:
        return 0.0
    thesis = _trade_entry_thesis(trade)
    override_pips = _safe_float(thesis.get("force_exit_max_floating_loss_pips"), 0.0)
    if override_pips <= 0.0:
        return default_pips
    return override_pips


def _build_technical_trade_profile(
    *,
    route_reasons: Sequence[str],
    lookahead_decision: Optional[object],
    vol_bucket: float,
) -> TechnicalTradeProfile:
    if not config.TECH_ROUTER_ENABLED:
        return TechnicalTradeProfile(
            tp_mult=1.0,
            sl_mult=1.0,
            hold_mult=1.0,
            hard_loss_mult=1.0,
            counter_pressure=False,
            route_reasons=(),
        )

    reasons: tuple[str, ...] = tuple(
        str(token).strip() for token in route_reasons if str(token).strip()
    )
    counter_pressure = bool(reasons)

    tp_mult = 1.0
    sl_mult = 1.0
    hold_mult = 1.0
    hard_loss_mult = 1.0

    if counter_pressure:
        tp_mult *= config.TECH_ROUTER_COUNTER_TP_MULT
        sl_mult *= config.TECH_ROUTER_COUNTER_SL_MULT
        hold_mult *= config.TECH_ROUTER_COUNTER_HOLD_MULT
        hard_loss_mult *= config.TECH_ROUTER_COUNTER_HARD_LOSS_MULT

    edge_ratio = 0.0
    if lookahead_decision is not None and bool(getattr(lookahead_decision, "allow_entry", False)):
        edge_pips = _safe_float(getattr(lookahead_decision, "edge_pips", 0.0), 0.0)
        edge_ref = max(0.05, float(config.LOOKAHEAD_EDGE_REF_PIPS))
        if edge_pips > edge_ref:
            edge_ratio = _clamp((edge_pips - edge_ref) / edge_ref, 0.0, 1.0)

    if edge_ratio > 0.0:
        tp_mult *= 1.0 + (config.TECH_ROUTER_EDGE_TP_BOOST_MAX * edge_ratio)
        hold_mult *= 1.0 + (config.TECH_ROUTER_EDGE_HOLD_BOOST_MAX * edge_ratio)
        hard_loss_mult *= 1.0 + (config.TECH_ROUTER_EDGE_HARD_LOSS_BOOST_MAX * edge_ratio)

    hold_mult = max(0.2, float(hold_mult))
    hard_loss_mult = max(0.2, float(hard_loss_mult))
    vol_bucket = _clamp(vol_bucket, 0.0, 1.0)
    hold_mult *= _lerp(
        config.FORCE_EXIT_VOL_HOLD_MAX_MULT,
        config.FORCE_EXIT_VOL_HOLD_MIN_MULT,
        vol_bucket,
    )
    hard_loss_mult *= _lerp(
        config.FORCE_EXIT_VOL_LOSS_MAX_MULT,
        config.FORCE_EXIT_VOL_LOSS_MIN_MULT,
        vol_bucket,
    )

    return TechnicalTradeProfile(
        tp_mult=float(_clamp(tp_mult, 0.2, 3.0)),
        sl_mult=float(_clamp(sl_mult, 0.2, 3.0)),
        hold_mult=float(_clamp(hold_mult, 0.2, 3.0)),
        hard_loss_mult=float(_clamp(hard_loss_mult, 0.2, 3.0)),
        counter_pressure=counter_pressure,
        route_reasons=reasons,
    )


def _scaled_force_exit_thresholds(
    *,
    base_max_hold_sec: float,
    base_hard_loss_pips: float,
    profile: TechnicalTradeProfile,
    vol_bucket: float,
) -> tuple[float, float]:
    hold_mult = max(0.2, float(profile.hold_mult))
    loss_mult = max(0.2, float(profile.hard_loss_mult))
    if config.FORCE_EXIT_VOL_ADAPT_ENABLED:
        vol_bucket = _clamp(vol_bucket, 0.0, 1.0)
        hold_mult *= 1.0 - (1.0 - config.FORCE_EXIT_VOL_HOLD_MIN_MULT) * vol_bucket
        loss_mult *= 1.0 - (1.0 - config.FORCE_EXIT_VOL_LOSS_MIN_MULT) * vol_bucket

    hold_sec = 0.0
    hard_loss_pips = 0.0
    if base_max_hold_sec > 0.0:
        hold_cap = max(
            config.TECH_ROUTER_HOLD_MIN_SEC,
            base_max_hold_sec * max(0.5, float(config.TECH_ROUTER_HOLD_MAX_MULT)),
        )
        hold_sec = _clamp(
            base_max_hold_sec * hold_mult,
            config.TECH_ROUTER_HOLD_MIN_SEC,
            hold_cap,
        )
    if base_hard_loss_pips > 0.0:
        hard_loss_pips = max(0.1, base_hard_loss_pips * loss_mult)
    return float(hold_sec), float(hard_loss_pips)


def _force_exit_thresholds_for_side(side: str) -> tuple[float, float]:
    if side != "short":
        return float(config.FORCE_EXIT_MAX_HOLD_SEC), float(
            config.FORCE_EXIT_MAX_FLOATING_LOSS_PIPS
        )

    max_hold_sec = float(config.SHORT_FORCE_EXIT_MAX_HOLD_SEC)
    max_floating_loss_pips = float(config.SHORT_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS)
    if max_hold_sec <= 0.0:
        max_hold_sec = float(config.FORCE_EXIT_MAX_HOLD_SEC)
    if max_floating_loss_pips <= 0.0:
        max_floating_loss_pips = float(config.FORCE_EXIT_MAX_FLOATING_LOSS_PIPS)
    return max_hold_sec, max_floating_loss_pips


def _force_exit_hard_loss_trigger_pips(trade: dict, base_hard_loss_pips: float) -> float:
    """Return effective hard-loss trigger pips for force-exit decisions.

    `unrealized_pl_pips` includes spread costs; cutting exactly at `-base_hard_loss_pips`
    often creates churn. Add entry spread (if available) plus a small buffer to avoid
    "meaningless" market-close exits around the threshold.
    """

    if base_hard_loss_pips <= 0.0:
        return 0.0
    base_hard_loss_pips = max(
        base_hard_loss_pips,
        float(getattr(config, "FORCE_EXIT_FLOATING_LOSS_MIN_PIPS", 0.0)),
    )
    entry_spread_pips = max(0.0, _trade_entry_spread_pips(trade))
    buffer_pips = max(0.0, float(getattr(config, "SL_SPREAD_BUFFER_PIPS", 0.0)))
    force_buffer_pips = max(
        buffer_pips,
        max(0.0, float(getattr(config, "FORCE_EXIT_BID_ASK_BUFFER_PIPS", 0.0))),
    )
    return float(base_hard_loss_pips + entry_spread_pips + force_buffer_pips)


def _trade_client_id(trade: dict) -> str:
    return str(trade.get("client_id") or trade.get("client_order_id") or "").strip()


def _profit_bank_start_time_utc() -> Optional[datetime.datetime]:
    global _PROFIT_BANK_START_CACHE
    raw = str(config.PROFIT_BANK_START_TIME_UTC or "").strip()
    cached_raw, cached_ts = _PROFIT_BANK_START_CACHE
    if cached_raw == raw:
        return cached_ts
    parsed = _parse_iso_utc(raw) if raw else None
    _PROFIT_BANK_START_CACHE = (raw, parsed)
    return parsed


def _load_profit_bank_stats(
    *,
    strategy_tag: str,
    pocket: str,
    reason: str,
    start_time_utc: Optional[datetime.datetime],
) -> tuple[float, float, float]:
    now_mono = time.monotonic()
    start_key = start_time_utc.isoformat() if start_time_utc else "-"
    cache_key = (
        str(strategy_tag or "").strip().lower(),
        str(pocket or "").strip().lower(),
        str(reason or "").strip().lower(),
        start_key,
    )
    cached = _PROFIT_BANK_STATS_CACHE.get(cache_key)
    if cached and now_mono - cached[0] <= config.PROFIT_BANK_STATS_TTL_SEC:
        return cached[1]

    gross_profit = 0.0
    spent_loss = 0.0
    net_realized = 0.0
    if _TRADES_DB.exists():
        con: Optional[sqlite3.Connection] = None
        try:
            con = sqlite3.connect(_TRADES_DB)
            row = con.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN realized_pl > 0 THEN realized_pl ELSE 0 END), 0),
                  COALESCE(SUM(CASE WHEN lower(coalesce(close_reason, '')) = ? AND realized_pl < 0 THEN -realized_pl ELSE 0 END), 0),
                  COALESCE(SUM(realized_pl), 0)
                FROM trades
                WHERE close_time IS NOT NULL
                  AND lower(coalesce(strategy_tag, '')) = ?
                  AND lower(coalesce(pocket, '')) = ?
                  AND (? = '' OR datetime(close_time) >= datetime(?))
                """,
                (
                    cache_key[2],
                    cache_key[0],
                    cache_key[1],
                    start_key if start_key != "-" else "",
                    start_key if start_key != "-" else "",
                ),
            ).fetchone()
            if row:
                gross_profit = _safe_float(row[0], 0.0)
                spent_loss = _safe_float(row[1], 0.0)
                net_realized = _safe_float(row[2], 0.0)
        except Exception:
            pass
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

    stats = (gross_profit, spent_loss, net_realized)
    _PROFIT_BANK_STATS_CACHE[cache_key] = (now_mono, stats)
    return stats


def _profit_bank_available_budget_jpy(
    *,
    gross_profit_jpy: float,
    spent_loss_jpy: float,
    net_realized_jpy: float,
) -> float:
    if gross_profit_jpy < config.PROFIT_BANK_MIN_GROSS_PROFIT_JPY:
        return 0.0
    budget_cap = gross_profit_jpy * config.PROFIT_BANK_SPEND_RATIO
    budget_left = budget_cap - spent_loss_jpy - config.PROFIT_BANK_MIN_BUFFER_JPY
    net_left = net_realized_jpy - config.PROFIT_BANK_MIN_NET_KEEP_JPY
    return max(0.0, min(budget_left, net_left))


def _is_profit_bank_excluded_trade(
    *,
    trade_id: str,
    client_id: str,
    excluded_trade_ids: set[str],
    excluded_client_ids: set[str],
) -> bool:
    if trade_id and trade_id in excluded_trade_ids:
        return True
    if client_id and client_id in excluded_client_ids:
        return True
    return False


async def _apply_profit_bank_release(
    *,
    pocket_info: dict,
    now_utc: datetime.datetime,
    logger: logging.Logger,
    protected_trade_ids: Optional[set[str]] = None,
) -> int:
    global _PROFIT_BANK_LAST_CLOSE_MONO

    if not config.PROFIT_BANK_ENABLED:
        return 0
    if not isinstance(pocket_info, dict):
        return 0
    open_trades = pocket_info.get("open_trades")
    if not isinstance(open_trades, list):
        return 0
    if config.PROFIT_BANK_MAX_ACTIONS <= 0:
        return 0

    now_mono = time.monotonic()
    if (
        config.PROFIT_BANK_COOLDOWN_SEC > 0.0
        and now_mono - _PROFIT_BANK_LAST_CLOSE_MONO < config.PROFIT_BANK_COOLDOWN_SEC
    ):
        return 0

    start_utc = _profit_bank_start_time_utc()
    gross_profit, spent_loss, net_realized = _load_profit_bank_stats(
        strategy_tag=config.STRATEGY_TAG,
        pocket=config.POCKET,
        reason=config.PROFIT_BANK_REASON,
        start_time_utc=start_utc,
    )
    available_budget = _profit_bank_available_budget_jpy(
        gross_profit_jpy=gross_profit,
        spent_loss_jpy=spent_loss,
        net_realized_jpy=net_realized,
    )
    if available_budget <= 0.0:
        return 0

    target_tag = str(config.STRATEGY_TAG or "").strip().lower()
    excluded_trade_ids = {str(t).strip() for t in config.PROFIT_BANK_EXCLUDE_TRADE_IDS if str(t).strip()}
    excluded_client_ids = {str(t).strip() for t in config.PROFIT_BANK_EXCLUDE_CLIENT_IDS if str(t).strip()}
    candidates: list[ProfitBankCandidate] = []
    for trade in open_trades:
        if not isinstance(trade, dict):
            continue
        if target_tag:
            trade_tag = _trade_strategy_tag(trade).lower()
            if trade_tag != target_tag:
                continue
        trade_id = str(trade.get("trade_id") or "").strip()
        if not trade_id:
            continue
        if protected_trade_ids and trade_id in protected_trade_ids:
            continue
        client_id = _trade_client_id(trade)
        if _is_profit_bank_excluded_trade(
            trade_id=trade_id,
            client_id=client_id,
            excluded_trade_ids=excluded_trade_ids,
            excluded_client_ids=excluded_client_ids,
        ):
            continue
        units = int(_safe_float(trade.get("units"), 0.0))
        if units == 0:
            continue
        opened_at = _parse_trade_open_time(trade.get("open_time") or trade.get("entry_time"))
        if opened_at is None:
            continue
        if (
            config.PROFIT_BANK_TARGET_REQUIRE_OPEN_BEFORE_START
            and start_utc is not None
            and opened_at >= start_utc
        ):
            continue
        hold_sec = (now_utc - opened_at).total_seconds()
        if hold_sec < config.PROFIT_BANK_TARGET_MIN_HOLD_SEC:
            continue
        unrealized_pl = _safe_float(trade.get("unrealized_pl"), 0.0)
        loss_jpy = abs(min(0.0, unrealized_pl))
        if loss_jpy < config.PROFIT_BANK_MIN_TARGET_LOSS_JPY:
            continue
        if loss_jpy > config.PROFIT_BANK_MAX_TARGET_LOSS_JPY:
            continue
        candidates.append(
            ProfitBankCandidate(
                trade_id=trade_id,
                client_id=client_id,
                units=units,
                opened_at=opened_at,
                hold_sec=hold_sec,
                unrealized_pl=unrealized_pl,
                loss_jpy=loss_jpy,
            )
        )
    if not candidates:
        return 0

    if config.PROFIT_BANK_TARGET_ORDER == "oldest":
        candidates.sort(key=lambda row: (row.opened_at, -row.loss_jpy))
    else:
        candidates.sort(key=lambda row: (-row.loss_jpy, row.opened_at))

    closed_count = 0
    for candidate in candidates:
        if closed_count >= config.PROFIT_BANK_MAX_ACTIONS:
            break
        if candidate.loss_jpy > available_budget:
            continue
        closed = await close_trade(
            candidate.trade_id,
            units=-candidate.units,
            client_order_id=candidate.client_id or None,
            allow_negative=True,
            exit_reason=config.PROFIT_BANK_REASON,
            env_prefix=config.ENV_PREFIX,
        )
        if not closed:
            continue
        closed_count += 1
        available_budget = max(0.0, available_budget - candidate.loss_jpy)
        _PROFIT_BANK_LAST_CLOSE_MONO = now_mono
        logger.info(
            "%s profit_bank close trade=%s loss_jpy=%.1f hold=%.0fs budget_left=%.1f gross=%.1f spent=%.1f net=%.1f reason=%s",
            config.LOG_PREFIX,
            candidate.trade_id,
            candidate.loss_jpy,
            candidate.hold_sec,
            available_budget,
            gross_profit,
            spent_loss,
            net_realized,
            config.PROFIT_BANK_REASON,
        )
        # Force DB re-read on the next loop after a successful close.
        _PROFIT_BANK_STATS_CACHE.clear()
    return closed_count


async def _enforce_new_entry_time_stop(
    *,
    pocket_info: dict,
    now_utc: datetime.datetime,
    logger: logging.Logger,
    protected_trade_ids: Optional[set[str]] = None,
) -> int:
    global _TRADE_MFE_PIPS, _TRADE_FORCE_EXIT_DEFER_LOG_MONO

    if not config.FORCE_EXIT_ENABLED:
        return 0
    if not config.FORCE_EXIT_ACTIVE:
        return 0
    if config.FORCE_EXIT_MAX_ACTIONS <= 0:
        return 0
    max_hold_sec = float(config.FORCE_EXIT_MAX_HOLD_SEC)
    hard_loss_pips = float(config.FORCE_EXIT_MAX_FLOATING_LOSS_PIPS)
    recovery_window_sec = float(config.FORCE_EXIT_RECOVERY_WINDOW_SEC)
    recoverable_loss_pips = float(config.FORCE_EXIT_RECOVERABLE_LOSS_PIPS)
    giveback_enabled = bool(config.FORCE_EXIT_GIVEBACK_ENABLED)
    giveback_arm_pips = float(config.FORCE_EXIT_GIVEBACK_ARM_PIPS)
    giveback_backoff_pips = float(config.FORCE_EXIT_GIVEBACK_BACKOFF_PIPS)
    giveback_min_hold_sec = float(config.FORCE_EXIT_GIVEBACK_MIN_HOLD_SEC)
    giveback_protect_pips = float(config.FORCE_EXIT_GIVEBACK_PROTECT_PIPS)
    if (
        max_hold_sec <= 0.0
        and hard_loss_pips <= 0.0
        and (recovery_window_sec <= 0.0 or recoverable_loss_pips <= 0.0)
        and (
            (not giveback_enabled)
            or giveback_arm_pips <= 0.0
            or giveback_backoff_pips <= 0.0
        )
    ):
        return 0
    if not isinstance(pocket_info, dict):
        return 0

    open_trades = pocket_info.get("open_trades")
    if not isinstance(open_trades, list):
        return 0

    target_tag = str(config.STRATEGY_TAG or "").strip().lower()
    target_generation = str(config.FORCE_EXIT_POLICY_GENERATION or "").strip()
    require_generation = bool(getattr(config, "FORCE_EXIT_REQUIRE_POLICY_GENERATION", False))
    closed_count = 0
    seen_trade_ids: set[str] = set()
    now_mono = time.monotonic()
    momentum_rows: Optional[list[dict]] = None
    if config.FORCE_EXIT_MOMENTUM_STALL_ENABLED:
        momentum_rows = list(
            tick_window.recent_ticks(
                seconds=max(0.4, float(config.FORCE_EXIT_MOMENTUM_STALL_WINDOW_SEC)),
                limit=max(config.FORCE_EXIT_MOMENTUM_STALL_MIN_TICKS, 40),
            )
        )

    for trade in open_trades:
        if closed_count >= config.FORCE_EXIT_MAX_ACTIONS:
            break
        if not isinstance(trade, dict):
            continue
        if target_tag:
            trade_tag = _trade_strategy_tag(trade).lower()
            if trade_tag != target_tag:
                continue
        generation = _trade_policy_generation(trade)
        if require_generation:
            if not target_generation:
                continue
            if generation != target_generation:
                continue

        opened_at = _parse_trade_open_time(trade.get("open_time") or trade.get("entry_time"))
        if opened_at is None:
            continue
        trade_id = str(trade.get("trade_id") or "").strip()
        if not trade_id:
            continue
        if protected_trade_ids and trade_id in protected_trade_ids:
            continue
        units = int(_safe_float(trade.get("units"), 0.0))
        if units == 0:
            continue
        hold_sec = (now_utc - opened_at).total_seconds()
        unrealized_pips = _trade_unrealized_pips(trade)
        trade_side = _trade_side_from_units(units)
        if trade_side is None:
            continue
        max_hold_sec, hard_loss_pips = _force_exit_thresholds_for_side(trade_side)
        trade_max_hold_sec = _trade_force_exit_max_hold_sec(trade, max_hold_sec)
        trade_hard_loss_pips = _trade_force_exit_hard_loss_pips(trade, hard_loss_pips)
        seen_trade_ids.add(trade_id)

        prev_mfe = _safe_float(_TRADE_MFE_PIPS.get(trade_id), unrealized_pips)
        mfe_pips = max(float(prev_mfe), float(unrealized_pips))
        _TRADE_MFE_PIPS[trade_id] = float(mfe_pips)

        exit_reason: Optional[str] = None
        trigger_label: Optional[str] = None
        hard_loss_trigger_pips = _force_exit_hard_loss_trigger_pips(
            trade,
            trade_hard_loss_pips,
        )
        if (
            hold_sec >= float(getattr(config, "FORCE_EXIT_FLOATING_LOSS_MIN_HOLD_SEC", 0.0))
            and hard_loss_trigger_pips > 0.0
            and unrealized_pips <= -hard_loss_trigger_pips
        ):
            exit_reason = config.FORCE_EXIT_MAX_FLOATING_LOSS_REASON
            trigger_label = "max_floating_loss"
        elif (
            recovery_window_sec > 0.0
            and recoverable_loss_pips > 0.0
            and hold_sec >= recovery_window_sec
            and unrealized_pips <= -recoverable_loss_pips
        ):
            exit_reason = config.FORCE_EXIT_RECOVERY_REASON
            trigger_label = "recovery_timeout"
        elif (
            giveback_enabled
            and giveback_arm_pips > 0.0
            and giveback_backoff_pips > 0.0
            and hold_sec >= giveback_min_hold_sec
            and mfe_pips >= giveback_arm_pips
            and (mfe_pips - unrealized_pips) >= giveback_backoff_pips
            and unrealized_pips <= giveback_protect_pips
        ):
            exit_reason = config.FORCE_EXIT_GIVEBACK_REASON
            trigger_label = "giveback_lock"
        elif trade_max_hold_sec > 0.0 and hold_sec >= trade_max_hold_sec:
            exit_reason = config.FORCE_EXIT_REASON
            trigger_label = "time_stop"
        elif momentum_rows is not None:
            stalled, first_delta, last_delta = _detect_momentum_stall_for_open_trade(
                trade_side=trade_side,
                hold_sec=hold_sec,
                rows=momentum_rows,
            )
            if stalled:
                if unrealized_pips < 0.0:
                    exit_reason = config.FORCE_EXIT_MOMENTUM_STALL_LOSS_REASON
                    trigger_label = "momentum_stall"
                    logger.info(
                        "%s momentum_stall candidate trade=%s side=%s hold=%.0fs first_delta=%.3f last_delta=%.3f pnl=%.2fp",
                        config.LOG_PREFIX,
                        trade_id,
                        trade_side,
                        hold_sec,
                        first_delta,
                        last_delta,
                        unrealized_pips,
                    )
                else:
                    logger.debug(
                        "%s momentum_stall skip close on profit trade=%s side=%s hold=%.0fs pnl=%.2fp first_delta=%.3f last_delta=%.3f",
                        config.LOG_PREFIX,
                        trade_id,
                        trade_side,
                        hold_sec,
                        unrealized_pips,
                        first_delta,
                        last_delta,
                    )
        if not exit_reason:
            continue
        defer, defer_reason, fib, regime = _should_defer_force_exit(
            trade=trade,
            units=units,
            hold_sec=hold_sec,
            unrealized_pips=unrealized_pips,
            exit_reason=exit_reason,
        )
        if defer:
            last_log = _safe_float(_TRADE_FORCE_EXIT_DEFER_LOG_MONO.get(trade_id), 0.0)
            if now_mono - last_log >= config.FORCE_EXIT_MTF_FIB_LOG_INTERVAL_SEC:
                logger.info(
                    "%s force_exit defer trade=%s trigger=%s hold=%.0fs pnl=%.2fp fib=%.2fp tf=%s zone=%.3f-%.3f target=%.3f regime=%s/%s/%.2f/%.2f",
                    config.LOG_PREFIX,
                    trade_id,
                    trigger_label or "-",
                    hold_sec,
                    unrealized_pips,
                    (fib.recover_pips if fib is not None else 0.0),
                    (fib.timeframe if fib is not None else "-"),
                    (fib.zone_low if fib is not None else 0.0),
                    (fib.zone_high if fib is not None else 0.0),
                    (fib.target_price if fib is not None else 0.0),
                    (regime.mode if regime is not None else "none"),
                    (regime.side if regime is not None else "none"),
                    (_safe_float(regime.trend_score, 0.0) if regime is not None else 0.0),
                    (_safe_float(regime.heat_score, 0.0) if regime is not None else 0.0),
                )
                _TRADE_FORCE_EXIT_DEFER_LOG_MONO[trade_id] = now_mono
            continue
        if defer_reason == "adverse_momentum":
            logger.info(
                "%s force_exit execute reason=%s trigger=%s trade=%s hold=%.0fs pnl=%.2fp detail=%s",
                config.LOG_PREFIX,
                exit_reason,
                trigger_label or "-",
                trade_id,
                hold_sec,
                unrealized_pips,
                defer_reason,
            )

        client_id = str(trade.get("client_id") or trade.get("client_order_id") or "").strip() or None
        closed = await close_trade(
            trade_id,
            units=-units,
            client_order_id=client_id,
            allow_negative=True,
            exit_reason=exit_reason,
            env_prefix=config.ENV_PREFIX,
        )
        if not closed:
            continue
        closed_count += 1
        if trigger_label == "max_floating_loss":
            logger.info(
                "%s force_exit threshold=%s base=%.2fp entry_sp=%.2fp buffer=%.2fp effective=%.2fp",
                config.LOG_PREFIX,
                trigger_label,
                trade_hard_loss_pips,
                _trade_entry_spread_pips(trade),
                float(getattr(config, "SL_SPREAD_BUFFER_PIPS", 0.0)),
                hard_loss_trigger_pips,
            )
        logger.info(
            "%s force_exit reason=%s trigger=%s trade=%s hold=%.0fs max_hold=%.0fs pnl=%.2fp mfe=%.2fp units=%s generation=%s",
            config.LOG_PREFIX,
            exit_reason,
            trigger_label or "-",
            trade_id,
            hold_sec,
            trade_max_hold_sec,
            unrealized_pips,
            mfe_pips,
            units,
            generation or "-",
        )
        _TRADE_MFE_PIPS.pop(trade_id, None)
        _TRADE_FORCE_EXIT_DEFER_LOG_MONO.pop(trade_id, None)

    stale_ids = [tid for tid in _TRADE_MFE_PIPS.keys() if tid not in seen_trade_ids]
    for tid in stale_ids:
        _TRADE_MFE_PIPS.pop(tid, None)
    stale_defer_ids = [tid for tid in _TRADE_FORCE_EXIT_DEFER_LOG_MONO.keys() if tid not in seen_trade_ids]
    for tid in stale_defer_ids:
        _TRADE_FORCE_EXIT_DEFER_LOG_MONO.pop(tid, None)

    return closed_count


def _quotes_from_row(row: dict, fallback_mid: float = 0.0) -> tuple[float, float, float]:
    bid = _safe_float(row.get("bid"), 0.0)
    ask = _safe_float(row.get("ask"), 0.0)
    mid = _safe_float(row.get("mid"), 0.0)
    if mid <= 0.0 and bid > 0.0 and ask > 0.0:
        mid = (bid + ask) * 0.5
    if mid <= 0.0:
        mid = fallback_mid
    return bid, ask, mid


def _latest_spread_from_ticks(rows: Sequence[dict]) -> float:
    if not rows:
        return 0.0
    bid, ask, _ = _quotes_from_row(rows[-1])
    if bid <= 0.0 or ask <= 0.0 or ask < bid:
        return 0.0
    return (ask - bid) / config.PIP_VALUE


def _latest_tick_age_ms(rows: Sequence[dict]) -> float:
    if not rows:
        return float("inf")
    latest_epoch = _safe_float(rows[-1].get("epoch"), 0.0)
    if latest_epoch <= 0.0:
        return float("inf")
    return max(0.0, (time.time() - latest_epoch) * 1000.0)


def _tick_density(rows: Sequence[dict], window_sec: float) -> float:
    if window_sec <= 0.0 or not rows:
        return 0.0
    latest_epoch = _safe_float(rows[-1].get("epoch"), 0.0)
    if latest_epoch <= 0.0:
        return 0.0
    cutoff = latest_epoch - window_sec
    count = 0
    for row in rows:
        if _safe_float(row.get("epoch"), 0.0) >= cutoff:
            count += 1
    return max(0.0, count / window_sec)


def _tick_span_ratio(rows: Sequence[dict], window_sec: float) -> float:
    if window_sec <= 0.0 or not rows:
        return 0.0
    latest_epoch = _safe_float(rows[-1].get("epoch"), 0.0)
    if latest_epoch <= 0.0:
        return 0.0
    first_epoch = _safe_float(rows[0].get("epoch"), latest_epoch)
    span_sec = max(0.0, latest_epoch - first_epoch)
    return max(0.0, min(1.0, span_sec / window_sec))


def _tick_density_over_window(window_sec: float) -> float:
    if window_sec <= 0.0:
        return 0.0
    rows = tick_window.recent_ticks(
        seconds=window_sec,
        limit=int(window_sec * 25) + 50,
    )
    return _tick_density(rows, window_sec)


async def _maybe_keepalive_snapshot(
    *,
    now_mono: float,
    last_snapshot_fetch: float,
    rows: Sequence[dict],
    latest_tick_age_ms: float,
    logger: logging.Logger,
) -> tuple[float, Optional[dict[str, float | str]]]:
    if not config.SNAPSHOT_FALLBACK_ENABLED or not config.SNAPSHOT_KEEPALIVE_ENABLED:
        return last_snapshot_fetch, None
    if now_mono - last_snapshot_fetch < config.SNAPSHOT_KEEPALIVE_MIN_INTERVAL_SEC:
        return last_snapshot_fetch, None

    window_sec = float(config.ENTRY_QUALITY_WINDOW_SEC)
    density_before = _tick_density(rows, window_sec)
    span_ratio_before = _tick_span_ratio(rows, window_sec)
    reasons: list[str] = []
    if (
        config.SNAPSHOT_KEEPALIVE_MAX_AGE_MS > 0.0
        and latest_tick_age_ms > config.SNAPSHOT_KEEPALIVE_MAX_AGE_MS
    ):
        reasons.append("stale")
    if (
        config.SNAPSHOT_KEEPALIVE_MIN_DENSITY > 0.0
        and density_before < config.SNAPSHOT_KEEPALIVE_MIN_DENSITY
    ):
        reasons.append("density")
    if (
        config.SNAPSHOT_KEEPALIVE_MIN_SPAN_RATIO > 0.0
        and span_ratio_before < config.SNAPSHOT_KEEPALIVE_MIN_SPAN_RATIO
    ):
        reasons.append("span")
    if not reasons:
        return last_snapshot_fetch, None
    if not await _fetch_price_snapshot(logger):
        return last_snapshot_fetch, None
    return now_mono, {
        "reason": ",".join(reasons),
        "age_ms": float(latest_tick_age_ms),
        "density": float(density_before),
        "span_ratio": float(span_ratio_before),
    }


async def _maybe_topup_micro_density(
    *,
    now_mono: float,
    last_snapshot_fetch: float,
    logger: logging.Logger,
) -> tuple[float, Optional[dict[str, float]]]:
    if not config.SNAPSHOT_FALLBACK_ENABLED or not config.SNAPSHOT_TOPUP_ENABLED:
        return last_snapshot_fetch, None
    target_density = float(config.SNAPSHOT_TOPUP_TARGET_DENSITY)
    if target_density <= 0.0:
        return last_snapshot_fetch, None
    if now_mono - last_snapshot_fetch < config.SNAPSHOT_TOPUP_MIN_INTERVAL_SEC:
        return last_snapshot_fetch, None

    window_sec = float(config.ENTRY_QUALITY_WINDOW_SEC)
    density_before = _tick_density_over_window(window_sec)
    if density_before >= target_density:
        return last_snapshot_fetch, None
    if not await _fetch_price_snapshot(logger):
        return last_snapshot_fetch, None

    density_after = _tick_density_over_window(window_sec)
    return now_mono, {
        "before": float(density_before),
        "after": float(density_after),
        "target": float(target_density),
        "window_sec": float(window_sec),
    }


def _tail_flow_ratio(mids: Sequence[float], *, want_up: bool, lookback: int) -> float:
    if len(mids) < 2:
        return 0.0
    start = max(1, len(mids) - max(2, int(lookback)))
    up = 0
    down = 0
    for i in range(start, len(mids)):
        diff = mids[i] - mids[i - 1]
        if diff > 0.0:
            up += 1
        elif diff < 0.0:
            down += 1
    directional = max(1, up + down)
    return (up / directional) if want_up else (down / directional)


def _momentum_tail_continuation_ok(
    mids: Sequence[float],
    *,
    side: str,
    momentum_pips: float,
) -> bool:
    """Return False when momentum appears late-expired (likely chasing the tail).

    A momentum setup is likely chasing if most of the move happened earlier in the
    window and the latest tail segment contributes little directional progress.
    """
    if side not in {"long", "short"}:
        return False
    if len(mids) < 5:
        return True

    abs_momentum = abs(_safe_float(momentum_pips, 0.0))
    if abs_momentum <= 0.0:
        return False

    tail_len = max(2, int(math.ceil(len(mids) * 0.38)))
    tail_start = max(0, len(mids) - 1 - tail_len)
    tail_delta_pips = (mids[-1] - mids[tail_start]) / config.PIP_VALUE
    required_tail = max(
        config.MOMENTUM_TRIGGER_PIPS * 0.35,
        0.28 * abs_momentum,
    )

    if side == "long":
        return tail_delta_pips >= required_tail
    return tail_delta_pips <= -required_tail


def _build_tick_signal(
    rows: Sequence[dict],
    spread_pips: float,
    *,
    signal_window_override_sec: Optional[float] = None,
    allow_window_fallback: bool = True,
) -> tuple[Optional[TickSignal], str]:
    if len(rows) < config.MIN_TICKS:
        return None, f"insufficient_rows:{len(rows)}/{config.MIN_TICKS}"

    latest_epoch = _safe_float(rows[-1].get("epoch"), 0.0)
    if latest_epoch <= 0.0:
        return None, "invalid_latest_epoch"

    tick_age_ms = max(0.0, (time.time() - latest_epoch) * 1000.0)
    if tick_age_ms > config.MAX_TICK_AGE_MS:
        return (
            None,
            f"stale_tick:{int(tick_age_ms)}ms>{int(config.MAX_TICK_AGE_MS)}ms",
        )

    instant_cutoff_for_scale = latest_epoch - config.INSTANT_VOL_WINDOW_SEC
    instant_mids_for_scale: list[float] = []
    for row in rows:
        _, _, row_mid = _quotes_from_row(row, fallback_mid=0.0)
        if row_mid <= 0.0:
            continue
        row_epoch = _safe_float(row.get("epoch"), 0.0)
        if row_epoch <= 0.0 or row_epoch < instant_cutoff_for_scale:
            continue
        instant_mids_for_scale.append(row_mid)

    instant_range_pips = 0.0
    if len(instant_mids_for_scale) >= 2:
        instant_range_pips = (
            max(instant_mids_for_scale) - min(instant_mids_for_scale)
        ) / config.PIP_VALUE
    speed_scale = _instant_speed_scale(instant_range_pips)

    short_min_signal_ticks = max(config.MIN_SIGNAL_TICKS, int(round(config.SHORT_MIN_SIGNAL_TICKS * speed_scale)))
    long_min_signal_ticks = max(config.MIN_SIGNAL_TICKS, int(round(config.LONG_MIN_SIGNAL_TICKS * speed_scale)))
    min_signal_ticks = max(config.MIN_SIGNAL_TICKS, min(short_min_signal_ticks, long_min_signal_ticks))
    fallback_rate_windows_sec = None
    if allow_window_fallback and config.MIN_TICK_RATE > 0 and min_signal_ticks > 1:
        fallback_rate_windows_sec = max(
            0.0,
            min(
                config.WINDOW_SEC,
                (min_signal_ticks - 1) / config.MIN_TICK_RATE,
            ),
        )
    if allow_window_fallback:
        fallback_sec = max(
            0.0, _safe_float(getattr(config, "SIGNAL_WINDOW_FALLBACK_SEC", 0.0), 0.0)
        )
    else:
        fallback_sec = 0.0

    signal_window_seed = max(0.3, config.SIGNAL_WINDOW_SEC * speed_scale)
    if signal_window_override_sec is not None:
        signal_window_seed = max(0.3, _safe_float(signal_window_override_sec, signal_window_seed))

    if len(rows) < min_signal_ticks:
        row_min_epoch = latest_epoch
        for row in rows:
            row_epoch = _safe_float(row.get("epoch"), latest_epoch)
            if row_epoch > 0 and row_epoch < row_min_epoch:
                row_min_epoch = row_epoch
        rows_span_sec = max(0.0, latest_epoch - row_min_epoch)
        detail_parts = [
            f"insufficient_signal_rows:{len(rows)}/{min_signal_ticks}",
            f"window={signal_window_seed:.2f}",
            f"fallback_window={signal_window_seed:.2f}",
            f"fallback_count={len(rows)}/{min_signal_ticks}",
            "fallback_attempts=none",
            "fallback_used=no_data",
            f"rows_span={rows_span_sec:.2f}",
        ]
        if fallback_rate_windows_sec is not None:
            detail_parts.append(
                f"fallback_min_rate_window={fallback_rate_windows_sec:.2f}"
            )
        if fallback_sec > 0.0:
            detail_parts.append(f"fallback_sec={fallback_sec:.2f}")
        return (
            None,
            " ".join(detail_parts),
        )

    signal_window_sec = signal_window_seed
    base_signal_window_sec = signal_window_sec
    signal_rows = [r for r in rows if _safe_float(r.get("epoch"), 0.0) >= latest_epoch - signal_window_sec]
    fallback_signal_rows: list[dict] = signal_rows
    fallback_window_sec: float = signal_window_sec

    fallback_windows = [signal_window_sec]
    if (
        allow_window_fallback
        and fallback_rate_windows_sec is not None
        and fallback_rate_windows_sec > signal_window_sec
    ):
        fallback_windows.append(fallback_rate_windows_sec)
    fallback_attempted = False
    if allow_window_fallback and fallback_sec > 0.0:
        fallback_windows.append(min(config.WINDOW_SEC, max(signal_window_sec, fallback_sec)))
    if (
        allow_window_fallback
        and signal_window_override_sec is None
        and fallback_sec > 0.0
        and config.WINDOW_SEC > signal_window_sec
        and config.SIGNAL_WINDOW_FALLBACK_ALLOW_FULL_WINDOW
    ):
        # When explicit fallback is configured but still insufficient, do one final
        # recovery sweep over the full strategy window before giving up.
        fallback_windows.append(config.WINDOW_SEC)

    fallback_attempts: list[float] = []
    for fallback_window_sec in sorted(set(fallback_windows)):
        if fallback_window_sec <= signal_window_sec:
            continue
        fallback_attempted = True
        fallback_attempts.append(_safe_float(fallback_window_sec, signal_window_sec))
        fallback_cutoff = latest_epoch - fallback_window_sec
        fallback_signal_rows = [
            r for r in rows
            if _safe_float(r.get("epoch"), 0.0) >= fallback_cutoff
        ]
        if len(fallback_signal_rows) >= min_signal_ticks:
            signal_window_sec = fallback_window_sec
            signal_rows = fallback_signal_rows
            break
    if len(signal_rows) < min_signal_ticks:
        if signal_window_sec > base_signal_window_sec:
            fallback_used = "yes"
        elif fallback_attempted:
            fallback_used = "attempted"
        else:
            fallback_used = "no"
        detail_parts = [
            f"insufficient_signal_rows:{len(signal_rows)}/{min_signal_ticks}",
            f"window={signal_window_sec:.2f}",
            f"fallback_window={fallback_window_sec:.2f}",
            f"fallback_count={len(fallback_signal_rows)}/{min_signal_ticks}",
            (
                f"fallback_attempts={','.join(f'{w:.2f}' for w in fallback_attempts)}"
                if fallback_attempts
                else "fallback_attempts=none"
            ),
            f"fallback_used={fallback_used}",
        ]
        if fallback_rate_windows_sec is not None:
            detail_parts.append(
                f"fallback_min_rate_window={fallback_rate_windows_sec:.2f}"
            )
        return (
            None,
            " ".join(detail_parts),
        )

    mids: list[float] = []
    epochs: list[float] = []
    fallback_mid = 0.0
    for row in signal_rows:
        bid, ask, mid = _quotes_from_row(row, fallback_mid=fallback_mid)
        if mid <= 0.0:
            continue
        epoch = _safe_float(row.get("epoch"), 0.0)
        if epoch <= 0.0:
            continue
        mids.append(mid)
        epochs.append(epoch)
        fallback_mid = mid

    if len(mids) < min_signal_ticks:
        return (
            None,
            f"insufficient_mid_rows:{len(mids)}/{min_signal_ticks}",
        )

    signal_range_pips = (max(mids) - min(mids)) / config.PIP_VALUE
    if len(mids) >= 2:
        instant_cutoff = latest_epoch - config.INSTANT_VOL_WINDOW_SEC
        instant_mids = [
            mids[i]
            for i, ep in enumerate(epochs)
            if ep >= instant_cutoff and i < len(mids)
        ]
        if len(instant_mids) >= 2:
            instant_range_pips = (max(instant_mids) - min(instant_mids)) / config.PIP_VALUE

    span_sec = max(0.001, epochs[-1] - epochs[0])
    momentum_pips = (mids[-1] - mids[0]) / config.PIP_VALUE

    up = 0
    down = 0
    for i in range(1, len(mids)):
        diff = mids[i] - mids[i - 1]
        if diff > 0:
            up += 1
        elif diff < 0:
            down += 1
    directional = max(1, up + down)
    imbalance = max(up, down) / directional
    tick_rate = max(0.0, (len(mids) - 1) / span_sec)

    bid, ask, mid = _quotes_from_row(signal_rows[-1], fallback_mid=mids[-1])
    if spread_pips <= 0.0:
        if bid > 0.0 and ask > 0.0 and ask >= bid:
            spread_pips = (ask - bid) / config.PIP_VALUE
        else:
            spread_pips = 0.0

    long_edge_guard_pips = max(
        config.LONG_MOMENTUM_TRIGGER_PIPS,
        spread_pips * config.MOMENTUM_SPREAD_MULT + config.ENTRY_BID_ASK_EDGE_PIPS,
    )
    short_edge_guard_pips = max(
        config.SHORT_MOMENTUM_TRIGGER_PIPS,
        spread_pips * config.MOMENTUM_SPREAD_MULT + config.ENTRY_BID_ASK_EDGE_PIPS,
    )
    trigger_pips = max(long_edge_guard_pips, short_edge_guard_pips)

    side_momentum: Optional[str] = None
    momentum_score = 0.0
    momentum_tail_rejected = False
    if (
        len(mids) >= long_min_signal_ticks
        and momentum_pips >= long_edge_guard_pips
        and imbalance >= config.IMBALANCE_MIN
        and tick_rate >= config.LONG_MIN_TICK_RATE
    ):
        side_momentum = "long"
    elif (
        len(mids) >= short_min_signal_ticks
        and momentum_pips <= -short_edge_guard_pips
        and imbalance >= config.IMBALANCE_MIN
        and tick_rate >= config.SHORT_MIN_TICK_RATE
    ):
        side_momentum = "short"

    if side_momentum is not None and not _momentum_tail_continuation_ok(
        mids,
        side=side_momentum,
        momentum_pips=momentum_pips,
    ):
        side_momentum = None
        momentum_tail_rejected = True

    strength = abs(momentum_pips) / max(0.01, trigger_pips)
    if side_momentum is not None:
        momentum_score = (
            strength
            + max(0.0, (imbalance - 0.5) * 3.2)
            + min(2.0, tick_rate / max(0.1, config.MIN_TICK_RATE)) * 0.35
        )

    side_revert: Optional[str] = None
    revert_score = 0.0
    revert_momentum_pips = 0.0
    revert_trigger_pips = max(0.05, config.REVERT_BOUNCE_MIN_PIPS)
    revert_reason = "revert_not_found"
    if config.REVERT_ENABLED:
        revert_window_sec = max(
            0.35,
            config.REVERT_WINDOW_SEC * speed_scale,
        )
        revert_short_window_sec = max(
            0.15,
            config.REVERT_SHORT_WINDOW_SEC * speed_scale,
        )
        revert_min_ticks = max(3, int(round(config.REVERT_MIN_TICKS * speed_scale)))
        revert_confirm_ticks = max(2, int(round(config.REVERT_CONFIRM_TICKS * speed_scale)))
        revert_cutoff = latest_epoch - revert_window_sec
        revert_rows = [r for r in rows if _safe_float(r.get("epoch"), 0.0) >= revert_cutoff]
        if len(revert_rows) >= revert_min_ticks:
            mids_revert: list[float] = []
            epochs_revert: list[float] = []
            fallback_revert_mid = 0.0
            for row in revert_rows:
                _, _, mid = _quotes_from_row(row, fallback_mid=fallback_revert_mid)
                if mid <= 0.0:
                    continue
                epoch = _safe_float(row.get("epoch"), 0.0)
                if epoch <= 0.0:
                    continue
                mids_revert.append(mid)
                epochs_revert.append(epoch)
                fallback_revert_mid = mid

            if len(mids_revert) >= revert_min_ticks:
                span_revert_sec = max(0.001, epochs_revert[-1] - epochs_revert[0])
                tick_rate_revert = max(0.0, (len(mids_revert) - 1) / span_revert_sec)
                range_revert_pips = (max(mids_revert) - min(mids_revert)) / config.PIP_VALUE
                if (
                    range_revert_pips >= config.REVERT_RANGE_MIN_PIPS
                    and tick_rate_revert >= config.REVERT_MIN_TICK_RATE
                ):
                    short_cutoff = latest_epoch - revert_short_window_sec
                    mids_short = [
                        mids_revert[idx]
                        for idx, ep in enumerate(epochs_revert)
                        if ep >= short_cutoff
                    ]
                    if len(mids_short) >= 2:
                        long_leg_pips = (mids_revert[-1] - mids_revert[0]) / config.PIP_VALUE
                        short_leg_pips = (mids_short[-1] - mids_short[0]) / config.PIP_VALUE
                        down_flow_ratio = _tail_flow_ratio(
                            mids_revert,
                            want_up=False,
                            lookback=revert_confirm_ticks,
                        )
                        up_flow_ratio = _tail_flow_ratio(
                            mids_revert,
                            want_up=True,
                            lookback=revert_confirm_ticks,
                        )

                        short_ok = (
                            long_leg_pips >= config.REVERT_SWEEP_MIN_PIPS
                            and short_leg_pips <= -config.REVERT_BOUNCE_MIN_PIPS
                            and down_flow_ratio >= config.REVERT_CONFIRM_RATIO_MIN
                        )
                        long_ok = (
                            long_leg_pips <= -config.REVERT_SWEEP_MIN_PIPS
                            and short_leg_pips >= config.REVERT_BOUNCE_MIN_PIPS
                            and up_flow_ratio >= config.REVERT_CONFIRM_RATIO_MIN
                        )

                        short_score = 0.0
                        long_score = 0.0
                        if short_ok:
                            short_score = (
                                abs(long_leg_pips) / max(0.1, config.REVERT_SWEEP_MIN_PIPS)
                                + abs(short_leg_pips) / max(0.05, config.REVERT_BOUNCE_MIN_PIPS)
                                + (range_revert_pips / max(0.1, config.REVERT_RANGE_MIN_PIPS)) * 0.45
                            )
                        if long_ok:
                            long_score = (
                                abs(long_leg_pips) / max(0.1, config.REVERT_SWEEP_MIN_PIPS)
                                + abs(short_leg_pips) / max(0.05, config.REVERT_BOUNCE_MIN_PIPS)
                                + (range_revert_pips / max(0.1, config.REVERT_RANGE_MIN_PIPS)) * 0.45
                            )

                        if short_score > 0.0 or long_score > 0.0:
                            if long_score >= short_score:
                                side_revert = "long"
                                revert_score = float(long_score)
                                revert_reason = "revert_long"
                            else:
                                side_revert = "short"
                                revert_score = float(short_score)
                            revert_momentum_pips = short_leg_pips
                            if side_revert == "short":
                                revert_reason = "revert_short"
                            revert_trigger_pips = max(
                                0.05,
                                config.REVERT_BOUNCE_MIN_PIPS,
                                spread_pips * 0.6,
                                spread_pips * config.MOMENTUM_SPREAD_MULT + config.ENTRY_BID_ASK_EDGE_PIPS,
                            )

    side: Optional[str] = None
    mode = "momentum"
    mode_score = momentum_score
    selected_momentum_pips = momentum_pips
    selected_trigger_pips = trigger_pips
    if side_momentum and side_revert:
        if revert_score >= momentum_score * config.MODE_SWITCH_REVERT_DOMINANCE:
            side = side_revert
            mode = "revert"
            mode_score = revert_score
            selected_momentum_pips = revert_momentum_pips
            selected_trigger_pips = revert_trigger_pips
        else:
            side = side_momentum
    elif side_momentum:
        side = side_momentum
    elif side_revert:
        side = side_revert
        mode = "revert"
        mode_score = revert_score
        selected_momentum_pips = revert_momentum_pips
        selected_trigger_pips = revert_trigger_pips

    if side is None:
        if momentum_tail_rejected and not config.REVERT_ENABLED:
            return None, "momentum_tail_failed_no_revert"
        if momentum_tail_rejected and side_revert is None:
            return None, "momentum_tail_failed"
        if not config.REVERT_ENABLED:
            return None, "revert_disabled"
        if side_revert is None:
            return None, "revert_not_found"
        return None, revert_reason

    if config.DROP_FLOW_ONLY:
        drop_cutoff = latest_epoch - config.DROP_FLOW_WINDOW_SEC
        drop_rows: list[dict] = []
        for row in rows:
            row_epoch = _safe_float(row.get("epoch"), 0.0)
            if row_epoch <= 0.0 or row_epoch < drop_cutoff:
                continue
            drop_rows.append(row)
        if len(drop_rows) < config.DROP_FLOW_MIN_TICKS:
            return (
                None,
                "drop_flow_not_enough_rows:"
                f"{len(drop_rows)}/{config.DROP_FLOW_MIN_TICKS}",
            )

        drop_mids: list[float] = []
        for row in drop_rows:
            _, _, row_mid = _quotes_from_row(row, fallback_mid=0.0)
            if row_mid <= 0.0:
                continue
            drop_mids.append(row_mid)
        if len(drop_mids) < 2:
            return None, "drop_flow_invalid_rows"

        drop_pips = (drop_mids[-1] - drop_mids[0]) / config.PIP_VALUE
        drop_pips_abs = abs(drop_pips)
        if drop_pips_abs < config.DROP_FLOW_MIN_PIPS:
            return (
                None,
                f"drop_flow_strength_not_met:{drop_pips:.2f}>{config.DROP_FLOW_MIN_PIPS:.2f}",
            )

        if side == "short" and drop_pips >= 0.0:
            return None, f"drop_flow_direction_mismatch:{drop_pips:.2f}"
        if side == "long" and drop_pips <= 0.0:
            return None, f"drop_flow_direction_mismatch:{drop_pips:.2f}"

        if config.DROP_FLOW_MAX_BOUNCE_PIPS > 0.0:
            if side == "short":
                max_bounce = (drop_mids[-1] - min(drop_mids)) / config.PIP_VALUE
            else:
                max_bounce = (max(drop_mids) - drop_mids[-1]) / config.PIP_VALUE
            if max_bounce > config.DROP_FLOW_MAX_BOUNCE_PIPS:
                return (
                    None,
                    f"drop_flow_tail_bounce:{max_bounce:.2f}>{config.DROP_FLOW_MAX_BOUNCE_PIPS:.2f}",
                )

    if config.SIDE_FILTER and side != config.SIDE_FILTER:
        return None, f"side_filter_block:{side}"

    selected_strength = abs(selected_momentum_pips) / max(0.01, selected_trigger_pips)
    confidence = config.CONFIDENCE_FLOOR + 2 if mode == "revert" else config.CONFIDENCE_FLOOR
    confidence += int(min(22.0, selected_strength * 8.0))
    confidence += int(min(10.0, max(0.0, (imbalance - 0.5) * 25.0)))
    confidence += int(min(8.0, tick_rate))
    confidence += int(min(6.0, mode_score * 1.2))
    confidence = max(config.CONFIDENCE_FLOOR, min(config.CONFIDENCE_CEIL, confidence))

    return (
        TickSignal(
        side=side,
        mode=mode,
        mode_score=float(mode_score),
        momentum_score=float(momentum_score),
        revert_score=float(revert_score),
        confidence=int(confidence),
        momentum_pips=float(selected_momentum_pips),
        trigger_pips=float(selected_trigger_pips),
        imbalance=imbalance,
        tick_rate=tick_rate,
        span_sec=span_sec,
        range_pips=float(signal_range_pips),
        instant_range_pips=float(instant_range_pips),
        signal_window_sec=float(signal_window_sec),
        tick_age_ms=tick_age_ms,
        spread_pips=max(0.0, spread_pips),
        bid=bid,
        ask=ask,
        mid=mid,
        ),
        "ok",
    )


def _retarget_signal(signal: TickSignal, *, side: str, mode: Optional[str] = None, confidence_add: int = 0) -> TickSignal:
    conf = int(
        _clamp(
            float(signal.confidence + confidence_add),
            float(config.CONFIDENCE_FLOOR),
            float(config.CONFIDENCE_CEIL),
        )
    )
    return TickSignal(
        side=side,
        mode=mode or signal.mode,
        mode_score=float(signal.mode_score),
        momentum_score=float(signal.momentum_score),
        revert_score=float(signal.revert_score),
        confidence=conf,
        momentum_pips=float(signal.momentum_pips),
        trigger_pips=float(signal.trigger_pips),
        imbalance=float(signal.imbalance),
        tick_rate=float(signal.tick_rate),
        span_sec=float(signal.span_sec),
        range_pips=float(signal.range_pips),
        instant_range_pips=float(signal.instant_range_pips),
        tick_age_ms=float(signal.tick_age_ms),
        spread_pips=float(signal.spread_pips),
        bid=float(signal.bid),
        ask=float(signal.ask),
        mid=float(signal.mid),
        signal_window_sec=float(signal.signal_window_sec),
        )


def _apply_mtf_regime(signal: TickSignal, regime: Optional[MtfRegime]) -> tuple[Optional[TickSignal], float, str]:
    if regime is None:
        return signal, 1.0, "regime_unavailable"

    units_mult = 1.0
    gate = "mtf_balanced"
    adjusted = signal

    if regime.mode == "continuation":
        if regime.side != "neutral" and signal.side != regime.side:
            if regime.heat_score >= config.MTF_CONTINUATION_BLOCK_HEAT:
                units_mult *= config.MTF_CONTINUATION_OPPOSITE_BLOCK_MIN_MULT
                return adjusted, float(max(0.0, units_mult)), "mtf_continuation_block_counter"
            units_mult *= config.MTF_CONTINUATION_OPPOSITE_UNITS_MULT
            gate = "mtf_continuation_counter_scaled"
        else:
            heat_ratio = _clamp(
                (regime.heat_score - config.MTF_HEAT_CONTINUATION_MIN)
                / max(0.05, 1.0 - config.MTF_HEAT_CONTINUATION_MIN),
                0.0,
                1.0,
            )
            trend_ratio = _clamp(
                (abs(regime.trend_score) - config.MTF_TREND_STRONG_SCORE)
                / max(0.05, 1.0 - config.MTF_TREND_STRONG_SCORE),
                0.0,
                1.0,
            )
            boost_ratio = max(heat_ratio, trend_ratio * 0.9)
            units_mult *= 1.0 + config.MTF_CONTINUATION_ALIGN_BOOST_MAX * boost_ratio
            adjusted = _retarget_signal(
                adjusted,
                side=adjusted.side,
                mode=f"{adjusted.mode}_cont",
                confidence_add=int(round(6.0 * boost_ratio)),
            )
            gate = "mtf_continuation_follow"
    elif regime.mode == "reversion":
        overshoot = abs(signal.momentum_pips) / max(0.01, signal.trigger_pips)
        if (
            overshoot >= config.MTF_REVERSION_TRIGGER_MULT
            and signal.imbalance >= config.MTF_REVERSION_IMBALANCE_MIN
        ):
            if regime.side != "neutral" and signal.side != regime.side:
                adjusted = _retarget_signal(
                    adjusted,
                    side=regime.side,
                    mode=f"{adjusted.mode}_mtf_revert",
                    confidence_add=3,
                )
                gate = "mtf_reversion_to_trend"
            elif regime.side == "neutral":
                flipped = "short" if signal.side == "long" else "long"
                adjusted = _retarget_signal(
                    adjusted,
                    side=flipped,
                    mode=f"{adjusted.mode}_mtf_fade",
                    confidence_add=2,
                )
                gate = "mtf_reversion_fade"
            else:
                gate = "mtf_reversion_aligned"

            cool_ratio = _clamp(
                (config.MTF_HEAT_REVERSION_MAX - regime.heat_score)
                / max(0.05, config.MTF_HEAT_REVERSION_MAX),
                0.0,
                1.0,
            )
            units_mult *= 1.0 + config.MTF_REVERSION_BOOST_MAX * cool_ratio
        else:
            gate = "mtf_reversion_no_setup"
    else:
        if regime.side != "neutral" and signal.side == regime.side:
            units_mult *= 1.0 + min(0.25, abs(regime.trend_score) * 0.22)
            gate = "mtf_balanced_align"

    return adjusted, float(max(0.0, units_mult)), gate


def _score_to_side(score: float, neutral: float) -> str:
    if score >= neutral:
        return "long"
    if score <= -neutral:
        return "short"
    return "neutral"


def _tf_score_or_none(factors: dict, timeframe: str) -> Optional[float]:
    tf = factors.get(timeframe) if isinstance(factors, dict) else None
    if not isinstance(tf, dict) or not tf:
        return None
    return float(_tf_trend_score(tf))


def _micro_horizon_score(signal: TickSignal) -> float:
    side_sign = 1.0 if signal.side == "long" else -1.0
    strength = _clamp(abs(signal.momentum_pips) / max(0.01, signal.trigger_pips), 0.0, 2.0)
    imbalance_push = _clamp((signal.imbalance - 0.5) * 1.8, 0.0, 0.9)
    mode_factor = 0.75 if signal.mode == "revert" else 1.0
    score = side_sign * mode_factor * _clamp(0.20 + (strength * 0.34) + imbalance_push, 0.0, 1.0)
    return float(_clamp(score, -1.0, 1.0))


def _build_horizon_bias(signal: TickSignal, factors: dict) -> Optional[HorizonBias]:
    if not config.HORIZON_BIAS_ENABLED:
        return None
    if not isinstance(factors, dict):
        return None

    short_score = _tf_score_or_none(factors, "M1")
    mid_score = _tf_score_or_none(factors, "M5")
    h1_score = _tf_score_or_none(factors, "H1")
    h4_score = _tf_score_or_none(factors, "H4")
    d1_score = _tf_score_or_none(factors, "D1")
    micro_score = _micro_horizon_score(signal)

    long_parts: list[tuple[float, float]] = []
    if h1_score is not None:
        long_parts.append((h1_score, 0.15))
    if h4_score is not None:
        long_parts.append((h4_score, 0.55))
    if d1_score is not None:
        long_parts.append((d1_score, 0.30))
    long_score: Optional[float] = None
    if long_parts:
        wsum = sum(w for _, w in long_parts)
        if wsum > 0.0:
            long_score = sum(v * w for v, w in long_parts) / wsum

    weighted: list[tuple[float, float]] = []
    if long_score is not None and config.HORIZON_LONG_WEIGHT > 0.0:
        weighted.append((long_score, config.HORIZON_LONG_WEIGHT))
    if mid_score is not None and config.HORIZON_MID_WEIGHT > 0.0:
        weighted.append((mid_score, config.HORIZON_MID_WEIGHT))
    if short_score is not None and config.HORIZON_SHORT_WEIGHT > 0.0:
        weighted.append((short_score, config.HORIZON_SHORT_WEIGHT))
    if config.HORIZON_MICRO_WEIGHT > 0.0:
        weighted.append((micro_score, config.HORIZON_MICRO_WEIGHT))
    if not weighted:
        return None

    wsum = sum(w for _, w in weighted)
    if wsum <= 0.0:
        return None
    composite_score = sum(v * w for v, w in weighted) / wsum
    neutral = config.HORIZON_NEUTRAL_SCORE
    composite_side = _score_to_side(composite_score, neutral)

    long_side = _score_to_side(long_score if long_score is not None else 0.0, neutral)
    mid_side = _score_to_side(mid_score if mid_score is not None else 0.0, neutral)
    short_side = _score_to_side(short_score if short_score is not None else 0.0, neutral)
    micro_side = _score_to_side(micro_score, neutral)

    agreement = 0
    if composite_side in {"long", "short"}:
        for side in (long_side, mid_side, short_side, micro_side):
            if side == composite_side:
                agreement += 1

    return HorizonBias(
        long_side=long_side,
        long_score=float(long_score or 0.0),
        mid_side=mid_side,
        mid_score=float(mid_score or 0.0),
        short_side=short_side,
        short_score=float(short_score or 0.0),
        micro_side=micro_side,
        micro_score=float(micro_score),
        composite_side=composite_side,
        composite_score=float(composite_score),
        agreement=int(agreement),
    )


def _apply_horizon_bias(
    signal: TickSignal,
    horizon: Optional[HorizonBias],
) -> tuple[Optional[TickSignal], float, str]:
    if horizon is None or not config.HORIZON_BIAS_ENABLED:
        return signal, 1.0, "horizon_unavailable"
    if horizon.composite_side == "neutral":
        return signal, 1.0, "horizon_neutral"

    score_abs = abs(horizon.composite_score)
    if signal.side != horizon.composite_side:
        if score_abs >= config.HORIZON_BLOCK_SCORE:
            return signal, max(0.0, config.HORIZON_OPPOSITE_BLOCK_MIN_MULT), "horizon_block_counter"
        scale = max(0.0, min(1.0, config.HORIZON_OPPOSITE_UNITS_MULT))
        if horizon.agreement >= 3:
            scale *= 0.9
        return signal, float(scale), "horizon_counter_scaled"

    if score_abs < config.HORIZON_ALIGN_SCORE_MIN:
        return signal, 1.0, "horizon_align_weak"

    ratio = _clamp(
        (score_abs - config.HORIZON_ALIGN_SCORE_MIN)
        / max(0.05, 1.0 - config.HORIZON_ALIGN_SCORE_MIN),
        0.0,
        1.0,
    )
    agree_boost = min(0.22, max(0, horizon.agreement - 1) * 0.04) * ratio
    units_mult = 1.0 + (config.HORIZON_ALIGN_BOOST_MAX * ratio) + agree_boost
    adjusted = _retarget_signal(
        signal,
        side=signal.side,
        mode=f"{signal.mode}_hz",
        confidence_add=int(round(5.0 * ratio + max(0, horizon.agreement - 1))),
    )
    return adjusted, float(max(0.0, units_mult)), "horizon_align"


def _m1_trend_units_multiplier(
    signal_side: str,
    m1_score: Optional[float],
    *,
    regime: Optional[MtfRegime] = None,
) -> tuple[float, str]:
    if not config.M1_TREND_SCALE_ENABLED:
        return 1.0, "disabled"
    if m1_score is None:
        return 1.0, "m1_unavailable"
    if signal_side not in {"long", "short"}:
        return 1.0, "invalid_side"

    score_abs = abs(_safe_float(m1_score, 0.0))
    vol_bucket = _regime_vol_bucket(
        regime,
        low_pips=config.M1_TREND_VOL_LOW_PIPS,
        high_pips=config.M1_TREND_VOL_HIGH_PIPS,
    ) if config.M1_TREND_VOL_INTERP_ENABLED else 0.5
    align_score_min = (
        _lerp(
            config.M1_TREND_ALIGN_SCORE_MIN_LOW_VOL,
            config.M1_TREND_ALIGN_SCORE_MIN_HIGH_VOL,
            vol_bucket,
        )
        if config.M1_TREND_VOL_INTERP_ENABLED
        else config.M1_TREND_ALIGN_SCORE_MIN
    )
    opposite_score = (
        _lerp(
            config.M1_TREND_OPPOSITE_SCORE_LOW_VOL,
            config.M1_TREND_OPPOSITE_SCORE_HIGH_VOL,
            vol_bucket,
        )
        if config.M1_TREND_VOL_INTERP_ENABLED
        else config.M1_TREND_OPPOSITE_SCORE
    )
    align_boost_max = (
        _lerp(
            config.M1_TREND_ALIGN_BOOST_MAX_LOW_VOL,
            config.M1_TREND_ALIGN_BOOST_MAX_HIGH_VOL,
            vol_bucket,
        )
        if config.M1_TREND_VOL_INTERP_ENABLED
        else config.M1_TREND_ALIGN_BOOST_MAX
    )
    align_score_min = _clamp(align_score_min, 0.05, 0.95)
    opposite_score = _clamp(opposite_score, align_score_min, 1.0)
    align_boost_max = _clamp(align_boost_max, 0.0, 1.0)

    if score_abs <= 0.0:
        return 1.0, "m1_zero"

    side_sign = 1.0 if signal_side == "long" else -1.0
    aligned = score_abs > 0.0 and (m1_score * side_sign) > 0.0
    if not aligned:
        denom = max(0.05, float(opposite_score))
        penalty_ratio = _clamp(score_abs / denom, 0.0, 1.0)
        scale = 1.0 - (1.0 - config.M1_TREND_OPPOSITE_UNITS_MULT) * penalty_ratio
        return float(max(0.1, scale)), "m1_opposite"

    if score_abs < align_score_min:
        return 1.0, "m1_align_weak"

    ratio = _clamp(
        (score_abs - align_score_min)
        / max(0.05, 1.0 - align_score_min),
        0.0,
        1.0,
    )
    return 1.0 + (align_boost_max * ratio), "m1_align_boost"


def _directional_bias_scale(rows: Sequence[dict], side: str) -> tuple[float, dict[str, float]]:
    if not config.SIDE_BIAS_ENABLED:
        return 1.0, {"enabled": 0.0}
    if side not in {"long", "short"}:
        return 1.0, {"enabled": 1.0}
    if not rows:
        return 1.0, {"enabled": 1.0}

    latest_epoch = _safe_float(rows[-1].get("epoch"), 0.0)
    if latest_epoch <= 0.0:
        return 1.0, {"enabled": 1.0}
    cutoff = latest_epoch - max(1.0, config.SIDE_BIAS_WINDOW_SEC)
    sample = [r for r in rows if _safe_float(r.get("epoch"), 0.0) >= cutoff]
    min_ticks = max(2, int(config.SIDE_BIAS_MIN_TICKS))
    if len(sample) < min_ticks:
        return 1.0, {"enabled": 1.0, "ticks": float(len(sample))}

    mids: list[float] = []
    fallback_mid = 0.0
    for row in sample:
        _, _, mid = _quotes_from_row(row, fallback_mid=fallback_mid)
        if mid <= 0.0:
            continue
        mids.append(mid)
        fallback_mid = mid
    if len(mids) < 2:
        return 1.0, {"enabled": 1.0, "ticks": float(len(mids))}

    drift_pips = (mids[-1] - mids[0]) / config.PIP_VALUE
    side_sign = 1.0 if side == "long" else -1.0
    aligned_pips = drift_pips * side_sign
    min_drift = max(0.0, config.SIDE_BIAS_MIN_DRIFT_PIPS)
    contra_pips = max(0.0, -aligned_pips - min_drift)
    if contra_pips <= 0.0:
        return 1.0, {
            "enabled": 1.0,
            "drift_pips": float(drift_pips),
            "aligned_pips": float(aligned_pips),
            "contra_pips": 0.0,
        }

    gain = max(0.0, config.SIDE_BIAS_SCALE_GAIN)
    floor = max(0.1, min(1.0, config.SIDE_BIAS_SCALE_FLOOR))
    scale = max(floor, 1.0 - contra_pips * gain)
    block_threshold = max(0.0, min(1.0, config.SIDE_BIAS_BLOCK_THRESHOLD))
    if block_threshold > 0.0 and scale <= block_threshold:
        scale = 0.0

    return float(scale), {
        "enabled": 1.0,
        "drift_pips": float(drift_pips),
        "aligned_pips": float(aligned_pips),
        "contra_pips": float(contra_pips),
        "scale": float(scale),
    }


def _build_direction_bias(rows: Sequence[dict], spread_pips: float) -> Optional[DirectionBias]:
    if not config.DIRECTION_BIAS_ENABLED:
        return None
    if len(rows) < config.DIRECTION_BIAS_MIN_TICKS:
        return None
    latest_epoch = _safe_float(rows[-1].get("epoch"), 0.0)
    if latest_epoch <= 0.0:
        return None
    window_cutoff = latest_epoch - config.DIRECTION_BIAS_WINDOW_SEC
    bias_rows = [r for r in rows if _safe_float(r.get("epoch"), 0.0) >= window_cutoff]
    if len(bias_rows) < config.DIRECTION_BIAS_MIN_TICKS:
        return None

    mids: list[float] = []
    epochs: list[float] = []
    fallback_mid = 0.0
    for row in bias_rows:
        _, _, mid = _quotes_from_row(row, fallback_mid=fallback_mid)
        if mid <= 0.0:
            continue
        epoch = _safe_float(row.get("epoch"), 0.0)
        if epoch <= 0.0:
            continue
        mids.append(mid)
        epochs.append(epoch)
        fallback_mid = mid
    if len(mids) < config.DIRECTION_BIAS_MIN_TICKS:
        return None

    span_sec = max(0.001, epochs[-1] - epochs[0])
    momentum_pips = (mids[-1] - mids[0]) / config.PIP_VALUE
    range_pips = max(0.0, (max(mids) - min(mids)) / config.PIP_VALUE)
    tick_rate = max(0.0, (len(mids) - 1) / span_sec)

    up = 0
    down = 0
    for i in range(1, len(mids)):
        diff = mids[i] - mids[i - 1]
        if diff > 0.0:
            up += 1
        elif diff < 0.0:
            down += 1
    directional = max(1, up + down)
    flow = (up - down) / directional

    trigger_pips = max(
        config.MOMENTUM_TRIGGER_PIPS,
        max(spread_pips, 0.0) * config.MOMENTUM_SPREAD_MULT,
    )
    mom_norm = _clamp(momentum_pips / max(trigger_pips * 2.0, 0.2), -1.0, 1.0)
    vol_span = max(
        0.05,
        config.DIRECTION_BIAS_VOL_HIGH_PIPS - config.DIRECTION_BIAS_VOL_LOW_PIPS,
    )
    vol_norm = _clamp(
        (range_pips - config.DIRECTION_BIAS_VOL_LOW_PIPS) / vol_span,
        0.0,
        1.0,
    )
    raw_score = (0.68 * mom_norm) + (0.32 * flow)
    score = _clamp(raw_score * (0.6 + (0.4 * vol_norm)), -1.0, 1.0)

    if score >= config.DIRECTION_BIAS_NEUTRAL_SCORE:
        side = "long"
    elif score <= -config.DIRECTION_BIAS_NEUTRAL_SCORE:
        side = "short"
    else:
        side = "neutral"

    return DirectionBias(
        side=side,
        score=score,
        momentum_pips=momentum_pips,
        flow=flow,
        range_pips=range_pips,
        vol_norm=vol_norm,
        tick_rate=tick_rate,
        span_sec=span_sec,
    )


def _direction_units_multiplier(
    signal_side: str,
    bias: Optional[DirectionBias],
    *,
    signal_mode: str = "momentum",
) -> tuple[float, str]:
    if not config.DIRECTION_BIAS_ENABLED or bias is None:
        return 1.0, "disabled"
    if bias.side == "neutral":
        return 1.0, "neutral"

    score_abs = abs(_safe_float(bias.score, 0.0))
    if signal_side == "short":
        direction_opposite_mult = config.DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT
    elif signal_side == "long":
        direction_opposite_mult = config.DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT
    else:
        direction_opposite_mult = config.DIRECTION_BIAS_OPPOSITE_UNITS_MULT
    if signal_side != bias.side:
        if signal_mode == "revert":
            if score_abs >= config.REVERT_DIRECTION_HARD_BLOCK_SCORE:
                return config.REVERT_DIRECTION_OPPOSITE_BLOCK_MIN_MULT, "revert_opposite_block"
            return direction_opposite_mult, "revert_opposite_scale"
        if score_abs >= config.DIRECTION_BIAS_BLOCK_SCORE:
            return direction_opposite_mult, "opposite_block"
        return direction_opposite_mult, "opposite_scale"

    if score_abs < config.DIRECTION_BIAS_ALIGN_SCORE_MIN:
        return 1.0, "align_weak"

    span = max(0.01, 1.0 - config.DIRECTION_BIAS_ALIGN_SCORE_MIN)
    ratio = _clamp((score_abs - config.DIRECTION_BIAS_ALIGN_SCORE_MIN) / span, 0.0, 1.0)
    boost = config.DIRECTION_BIAS_ALIGN_UNITS_BOOST_MAX * ratio
    if signal_mode == "revert":
        boost *= 0.35
    return 1.0 + boost, "align_boost"


def _maybe_fast_direction_flip(
    signal: TickSignal,
    *,
    direction_bias: Optional[DirectionBias],
    horizon: Optional[HorizonBias],
    regime: Optional[MtfRegime],
    now_mono: float,
) -> tuple[Optional[TickSignal], str]:
    global _LAST_FAST_FLIP_MONO

    if not config.FAST_DIRECTION_FLIP_ENABLED:
        return None, "disabled"
    if direction_bias is None:
        return None, "no_direction_bias"
    if horizon is None:
        return None, "no_horizon"

    current_side = str(signal.side or "").strip().lower()
    target_side = str(direction_bias.side or "").strip().lower()
    if current_side not in {"long", "short"}:
        return None, "invalid_signal_side"
    if target_side not in {"long", "short"}:
        return None, "bias_neutral"
    if current_side == target_side:
        return None, "already_aligned"

    cooldown_sec = max(0.0, float(config.FAST_DIRECTION_FLIP_COOLDOWN_SEC))
    if now_mono - _LAST_FAST_FLIP_MONO < cooldown_sec:
        return None, "cooldown"

    bias_score = abs(_safe_float(getattr(direction_bias, "score", 0.0), 0.0))
    if bias_score < float(config.FAST_DIRECTION_FLIP_DIRECTION_SCORE_MIN):
        return None, "bias_weak"

    horizon_side = str(getattr(horizon, "composite_side", "")).strip().lower()
    horizon_score = abs(_safe_float(getattr(horizon, "composite_score", 0.0), 0.0))
    horizon_agree = int(_safe_float(getattr(horizon, "agreement", 0.0), 0.0))
    if horizon_side == target_side:
        if horizon_score < float(config.FAST_DIRECTION_FLIP_HORIZON_SCORE_MIN):
            return None, "horizon_weak"
        if horizon_agree < int(config.FAST_DIRECTION_FLIP_HORIZON_AGREE_MIN):
            return None, "horizon_disagree"
    elif horizon_side == "neutral":
        if bias_score < float(config.FAST_DIRECTION_FLIP_NEUTRAL_HORIZON_BIAS_SCORE_MIN):
            return None, "horizon_neutral_bias_weak"
    else:
        return None, "horizon_mismatch"

    bias_momentum = _safe_float(getattr(direction_bias, "momentum_pips", 0.0), 0.0)
    min_momentum = max(0.0, float(config.FAST_DIRECTION_FLIP_MOMENTUM_MIN_PIPS))
    if target_side == "long" and bias_momentum < min_momentum:
        return None, "bias_momentum_weak"
    if target_side == "short" and bias_momentum > -min_momentum:
        return None, "bias_momentum_weak"

    if regime is not None:
        regime_side = str(getattr(regime, "side", "")).strip().lower()
        regime_score = abs(_safe_float(getattr(regime, "trend_score", 0.0), 0.0))
        if (
            regime_side in {"long", "short"}
            and regime_side != target_side
            and regime_score >= float(config.FAST_DIRECTION_FLIP_REGIME_BLOCK_SCORE)
        ):
            return None, "regime_counter"

    confidence_add = int(config.FAST_DIRECTION_FLIP_CONFIDENCE_ADD)
    if horizon_agree >= 3:
        confidence_add += 1
    flipped = _retarget_signal(
        signal,
        side=target_side,
        mode=f"{signal.mode}_fflip",
        confidence_add=confidence_add,
    )
    _LAST_FAST_FLIP_MONO = now_mono
    reason = (
        f"{current_side}->{target_side}:"
        f"bias={bias_score:.2f},horizon={horizon_score:.2f},agree={horizon_agree}"
    )
    return flipped, reason


def _load_stop_loss_streak(
    *,
    strategy_tag: str,
    pocket: str,
    now_utc: datetime.datetime,
    now_mono: Optional[float] = None,
) -> Optional[StopLossStreak]:
    cache_key = (
        str(strategy_tag or "").strip().lower(),
        str(pocket or "").strip().lower(),
    )
    if not cache_key[0] or not cache_key[1]:
        return None

    if now_mono is None:
        now_mono = time.monotonic()
    ttl_sec = max(0.1, float(config.SL_STREAK_DIRECTION_FLIP_CACHE_TTL_SEC))
    cached = _SL_STREAK_CACHE.get(cache_key)
    if cached and now_mono - cached[0] <= ttl_sec:
        return cached[1]

    streak_info: Optional[StopLossStreak] = None
    if _TRADES_DB.exists():
        con: Optional[sqlite3.Connection] = None
        rows: list[tuple[object, object, object]] = []
        lookback = max(1, int(config.SL_STREAK_DIRECTION_FLIP_LOOKBACK_TRADES))
        try:
            con = sqlite3.connect(_TRADES_DB)
            rows = con.execute(
                """
                SELECT close_time, units, close_reason
                FROM trades
                WHERE close_time IS NOT NULL
                  AND lower(coalesce(strategy_tag, '')) = ?
                  AND lower(coalesce(pocket, '')) = ?
                ORDER BY datetime(close_time) DESC
                LIMIT ?
                """,
                (cache_key[0], cache_key[1], lookback),
            ).fetchall()
        except Exception:
            rows = []
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

        streak_side = ""
        streak_count = 0
        sample = 0
        latest_close_time: Optional[datetime.datetime] = None
        for raw_close_time, raw_units, raw_reason in rows:
            sample += 1
            close_reason = str(raw_reason or "").strip().upper()
            if close_reason != "STOP_LOSS_ORDER":
                break
            side = _trade_side_from_units(int(_safe_float(raw_units, 0.0)))
            if side not in {"long", "short"}:
                break
            if not streak_side:
                streak_side = side
                latest_close_time = _parse_iso_utc(raw_close_time)
            if side != streak_side:
                break
            streak_count += 1

        if streak_count > 0 and streak_side in {"long", "short"}:
            age_sec = float("inf")
            if latest_close_time is not None:
                age_sec = max(0.0, (now_utc - latest_close_time).total_seconds())
            streak_info = StopLossStreak(
                side=streak_side,
                streak=streak_count,
                age_sec=float(age_sec),
                latest_close_time=latest_close_time,
                sample=sample,
            )

    _SL_STREAK_CACHE[cache_key] = (now_mono, streak_info)
    return streak_info


def _load_recent_side_close_metrics(
    *,
    strategy_tag: str,
    pocket: str,
    now_mono: Optional[float] = None,
) -> Optional[SideCloseMetrics]:
    return _load_side_close_metrics(
        strategy_tag=strategy_tag,
        pocket=pocket,
        lookback=max(1, int(config.SL_STREAK_DIRECTION_FLIP_METRICS_LOOKBACK_TRADES)),
        cache_ttl_sec=max(0.1, float(config.SL_STREAK_DIRECTION_FLIP_METRICS_CACHE_TTL_SEC)),
        now_mono=now_mono,
        cache_store=_SL_METRICS_CACHE,
    )


def _load_side_close_metrics(
    *,
    strategy_tag: str,
    pocket: str,
    lookback: int,
    cache_ttl_sec: float,
    now_mono: Optional[float],
    cache_store: dict[tuple[str, str], tuple[float, Optional[SideCloseMetrics]]],
) -> Optional[SideCloseMetrics]:
    cache_key = (
        str(strategy_tag or "").strip().lower(),
        str(pocket or "").strip().lower(),
    )
    if not cache_key[0] or not cache_key[1]:
        return None

    if now_mono is None:
        now_mono = time.monotonic()
    ttl_sec = max(0.1, float(cache_ttl_sec))
    cached = cache_store.get(cache_key)
    if cached and now_mono - cached[0] <= ttl_sec:
        return cached[1]

    metrics: Optional[SideCloseMetrics] = None
    if _TRADES_DB.exists():
        con: Optional[sqlite3.Connection] = None
        bounded_lookback = max(1, int(lookback))
        row = None
        try:
            con = sqlite3.connect(_TRADES_DB)
            row = con.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN units > 0 AND upper(coalesce(close_reason, '')) = 'STOP_LOSS_ORDER' THEN 1 ELSE 0 END), 0),
                  COALESCE(SUM(CASE WHEN units < 0 AND upper(coalesce(close_reason, '')) = 'STOP_LOSS_ORDER' THEN 1 ELSE 0 END), 0),
                  COALESCE(SUM(CASE WHEN units > 0 AND upper(coalesce(close_reason, '')) = 'MARKET_ORDER_TRADE_CLOSE' AND COALESCE(realized_pl, 0) > 0 THEN 1 ELSE 0 END), 0),
                  COALESCE(SUM(CASE WHEN units < 0 AND upper(coalesce(close_reason, '')) = 'MARKET_ORDER_TRADE_CLOSE' AND COALESCE(realized_pl, 0) > 0 THEN 1 ELSE 0 END), 0),
                  COALESCE(SUM(CASE WHEN units > 0 THEN 1 ELSE 0 END), 0),
                  COALESCE(SUM(CASE WHEN units < 0 THEN 1 ELSE 0 END), 0),
                  COUNT(*)
                FROM (
                  SELECT units, close_reason, realized_pl
                  FROM trades
                  WHERE close_time IS NOT NULL
                    AND lower(coalesce(strategy_tag, '')) = ?
                    AND lower(coalesce(pocket, '')) = ?
                  ORDER BY datetime(close_time) DESC
                  LIMIT ?
                )
                """,
                (cache_key[0], cache_key[1], bounded_lookback),
            ).fetchone()
        except Exception:
            row = None
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

        if row is not None:
            metrics = SideCloseMetrics(
                long_sl_hits=int(_safe_float(row[0], 0.0)),
                short_sl_hits=int(_safe_float(row[1], 0.0)),
                long_market_plus=int(_safe_float(row[2], 0.0)),
                short_market_plus=int(_safe_float(row[3], 0.0)),
                long_trades=int(_safe_float(row[4], 0.0)),
                short_trades=int(_safe_float(row[5], 0.0)),
                sample=int(_safe_float(row[6], 0.0)),
            )

    cache_store[cache_key] = (now_mono, metrics)
    return metrics


def _load_recent_side_close_metrics_for_allocation(
    *,
    strategy_tag: str,
    pocket: str,
    now_mono: Optional[float] = None,
) -> Optional[SideCloseMetrics]:
    return _load_side_close_metrics(
        strategy_tag=strategy_tag,
        pocket=pocket,
        lookback=max(1, int(config.ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_LOOKBACK_TRADES)),
        cache_ttl_sec=max(
            0.1,
            float(config.ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_CACHE_TTL_SEC),
        ),
        now_mono=now_mono,
        cache_store=_SIDE_CLOSE_METRICS_ALLOC_CACHE,
    )


def _coerce_entry_probability(
    *,
    entry_probability: object,
    entry_probability_raw: object,
    confidence: object,
) -> Optional[float]:
    for raw in (entry_probability, entry_probability_raw, confidence):
        value = _safe_float(raw, float("nan"))
        if math.isnan(value):
            continue
        if value > 1.0:
            value /= 100.0
        if 0.0 <= value <= 1.0:
            return float(value)
    return None


def _load_entry_probability_band_metrics(
    *,
    strategy_tag: str,
    pocket: str,
    side: str,
    now_mono: Optional[float] = None,
) -> Optional[EntryProbabilityBandMetrics]:
    side_key = str(side or "").strip().lower()
    if side_key not in {"long", "short"}:
        return None
    cache_key = (
        str(strategy_tag or "").strip().lower(),
        str(pocket or "").strip().lower(),
        side_key,
    )
    if not cache_key[0] or not cache_key[1]:
        return None

    if now_mono is None:
        now_mono = time.monotonic()
    ttl_sec = max(0.2, float(config.ENTRY_PROBABILITY_BAND_ALLOC_CACHE_TTL_SEC))
    cached = _ENTRY_PROB_BAND_METRICS_CACHE.get(cache_key)
    if cached and now_mono - cached[0] <= ttl_sec:
        return cached[1]

    metrics: Optional[EntryProbabilityBandMetrics] = None
    if _TRADES_DB.exists():
        lookback = max(1, int(config.ENTRY_PROBABILITY_BAND_ALLOC_LOOKBACK_TRADES))
        low_cutoff = _clamp(float(config.ENTRY_PROBABILITY_BAND_ALLOC_LOW_THRESHOLD), 0.0, 1.0)
        high_cutoff = _clamp(float(config.ENTRY_PROBABILITY_BAND_ALLOC_HIGH_THRESHOLD), 0.0, 1.0)
        if low_cutoff < high_cutoff:
            side_cond = "units > 0" if side_key == "long" else "units < 0"
            con: Optional[sqlite3.Connection] = None
            rows: Sequence[tuple[object, object, object, object, object]] = []
            try:
                con = sqlite3.connect(_TRADES_DB)
                rows = con.execute(
                    f"""
                    SELECT
                      CAST(json_extract(entry_thesis, '$.entry_probability') AS REAL) AS entry_probability,
                      CAST(json_extract(entry_thesis, '$.entry_probability_raw') AS REAL) AS entry_probability_raw,
                      CAST(json_extract(entry_thesis, '$.confidence') AS REAL) AS confidence,
                      pl_pips,
                      upper(coalesce(close_reason, '')) AS close_reason
                    FROM (
                      SELECT entry_thesis, pl_pips, close_reason
                      FROM trades
                      WHERE close_time IS NOT NULL
                        AND pl_pips IS NOT NULL
                        AND lower(coalesce(strategy_tag, '')) = ?
                        AND lower(coalesce(pocket, '')) = ?
                        AND {side_cond}
                      ORDER BY datetime(close_time) DESC
                      LIMIT ?
                    )
                    """,
                    (cache_key[0], cache_key[1], lookback),
                ).fetchall()
            except Exception:
                rows = []
            finally:
                if con is not None:
                    try:
                        con.close()
                    except Exception:
                        pass

            sample = 0
            high_sample = 0
            low_sample = 0
            high_pl_sum = 0.0
            low_pl_sum = 0.0
            high_wins = 0
            low_wins = 0
            high_sl_hits = 0
            low_sl_hits = 0
            for raw_prob, raw_prob_raw, raw_conf, raw_pl_pips, raw_close_reason in rows:
                probability = _coerce_entry_probability(
                    entry_probability=raw_prob,
                    entry_probability_raw=raw_prob_raw,
                    confidence=raw_conf,
                )
                if probability is None:
                    continue
                pl_pips = _safe_float(raw_pl_pips, 0.0)
                close_reason = str(raw_close_reason or "").strip().upper()
                sample += 1
                if probability >= high_cutoff:
                    high_sample += 1
                    high_pl_sum += pl_pips
                    if pl_pips > 0.0:
                        high_wins += 1
                    if close_reason == "STOP_LOSS_ORDER":
                        high_sl_hits += 1
                elif probability < low_cutoff:
                    low_sample += 1
                    low_pl_sum += pl_pips
                    if pl_pips > 0.0:
                        low_wins += 1
                    if close_reason == "STOP_LOSS_ORDER":
                        low_sl_hits += 1

            metrics = EntryProbabilityBandMetrics(
                side=side_key,
                sample=sample,
                high_sample=high_sample,
                high_mean_pips=(high_pl_sum / high_sample) if high_sample > 0 else 0.0,
                high_win_rate=(high_wins / high_sample) if high_sample > 0 else 0.0,
                high_sl_rate=(high_sl_hits / high_sample) if high_sample > 0 else 0.0,
                low_sample=low_sample,
                low_mean_pips=(low_pl_sum / low_sample) if low_sample > 0 else 0.0,
                low_win_rate=(low_wins / low_sample) if low_sample > 0 else 0.0,
                low_sl_rate=(low_sl_hits / low_sample) if low_sample > 0 else 0.0,
            )

    _ENTRY_PROB_BAND_METRICS_CACHE[cache_key] = (now_mono, metrics)
    return metrics


def _maybe_sl_streak_direction_flip(
    signal: TickSignal,
    *,
    strategy_tag: str,
    pocket: str,
    now_utc: datetime.datetime,
    now_mono: float,
    direction_bias: Optional[DirectionBias],
    horizon: Optional[HorizonBias],
    fast_flip_applied: bool,
) -> tuple[Optional[TickSignal], str, SlFlipEval]:
    if not config.SL_STREAK_DIRECTION_FLIP_ENABLED:
        return None, "disabled", SlFlipEval(None, 0, 0, False, False)

    current_side = str(signal.side or "").strip().lower()
    if current_side not in {"long", "short"}:
        return None, "invalid_signal_side", SlFlipEval(None, 0, 0, False, False)

    target_side = "short" if current_side == "long" else "long"
    min_streak = max(1, int(config.SL_STREAK_DIRECTION_FLIP_MIN_STREAK))
    max_age_sec = max(0.0, float(config.SL_STREAK_DIRECTION_FLIP_MAX_AGE_SEC))
    min_side_sl_hits = max(1, int(config.SL_STREAK_DIRECTION_FLIP_MIN_SIDE_SL_HITS))
    min_target_market_plus = max(0, int(config.SL_STREAK_DIRECTION_FLIP_MIN_TARGET_MARKET_PLUS))

    metrics = _load_recent_side_close_metrics(
        strategy_tag=strategy_tag,
        pocket=pocket,
        now_mono=now_mono,
    )
    if metrics is None:
        return None, "metrics_unavailable", SlFlipEval(None, 0, 0, False, False)
    side_sl_hits_recent = int(metrics.sl_hits(current_side))
    target_market_plus_recent = int(metrics.market_plus(target_side))
    side_trades_recent = (
        int(metrics.long_trades)
        if current_side == "long"
        else int(metrics.short_trades)
    )
    side_sl_rate_recent = (
        float(side_sl_hits_recent) / float(max(1, side_trades_recent))
    )

    streak = _load_stop_loss_streak(
        strategy_tag=strategy_tag,
        pocket=pocket,
        now_utc=now_utc,
        now_mono=now_mono,
    )
    direction_confirmed = False
    horizon_confirmed = False
    if streak is not None and streak.side != current_side:
        return (
            None,
            "already_opposite",
            SlFlipEval(streak, side_sl_hits_recent, target_market_plus_recent, False, False),
        )

    streak_count = int(streak.streak) if streak is not None else 0
    streak_is_stale = bool(
        streak is not None and max_age_sec > 0.0 and streak.age_sec > max_age_sec
    )
    metrics_override = bool(config.SL_STREAK_DIRECTION_FLIP_METRICS_OVERRIDE_ENABLED)
    metrics_override = bool(
        metrics_override
        and side_sl_hits_recent >= min_side_sl_hits
        and target_market_plus_recent >= min_target_market_plus
        and side_trades_recent >= int(config.SL_STREAK_DIRECTION_FLIP_METRICS_SIDE_TRADES_MIN)
        and side_sl_rate_recent
        >= float(config.SL_STREAK_DIRECTION_FLIP_METRICS_SIDE_SL_RATE_MIN)
    )
    if streak is None and not metrics_override:
        return None, "no_streak", SlFlipEval(None, side_sl_hits_recent, target_market_plus_recent, False, False)
    if streak_count < min_streak and not metrics_override:
        return (
            None,
            "below_min_streak",
            SlFlipEval(streak, side_sl_hits_recent, target_market_plus_recent, False, False),
        )
    if streak_is_stale and not metrics_override:
        return (
            None,
            "streak_stale",
            SlFlipEval(streak, side_sl_hits_recent, target_market_plus_recent, False, False),
        )

    if side_sl_hits_recent < min_side_sl_hits:
        return (
            None,
            "side_sl_hits_weak",
            SlFlipEval(streak, side_sl_hits_recent, target_market_plus_recent, False, False),
        )

    force_streak = max(min_streak, int(config.SL_STREAK_DIRECTION_FLIP_FORCE_STREAK))
    force_flip_by_streak = streak_count >= force_streak
    if target_market_plus_recent < min_target_market_plus and not force_flip_by_streak:
        return (
            None,
            "target_market_plus_weak",
            SlFlipEval(streak, side_sl_hits_recent, target_market_plus_recent, False, False),
        )

    if not config.SL_STREAK_DIRECTION_FLIP_ALLOW_WITH_FAST_FLIP and fast_flip_applied:
        return (
            None,
            "fast_flip_priority",
            SlFlipEval(streak, side_sl_hits_recent, target_market_plus_recent, False, False),
        )

    # Tech alignment gate: keep SL-streak flip as secondary override.
    if direction_bias is not None:
        bias_side = str(getattr(direction_bias, "side", "")).strip().lower()
        bias_score = abs(_safe_float(getattr(direction_bias, "score", 0.0), 0.0))
        direction_confirmed = (
            bias_side == target_side
            and bias_score >= float(config.SL_STREAK_DIRECTION_FLIP_DIRECTION_SCORE_MIN)
        )
    if horizon is not None:
        horizon_side = str(getattr(horizon, "composite_side", "")).strip().lower()
        horizon_score = abs(_safe_float(getattr(horizon, "composite_score", 0.0), 0.0))
        horizon_confirmed = (
            horizon_side == target_side
            and horizon_score >= float(config.SL_STREAK_DIRECTION_FLIP_HORIZON_SCORE_MIN)
        )
    force_tech_bypass = (
        force_flip_by_streak
        and bool(config.SL_STREAK_DIRECTION_FLIP_FORCE_WITHOUT_TECH_CONFIRM)
    )
    if config.SL_STREAK_DIRECTION_FLIP_REQUIRE_TECH_CONFIRM and not (
        direction_confirmed or horizon_confirmed
    ) and not force_tech_bypass:
        return (
            None,
            "tech_not_confirmed",
            SlFlipEval(
                streak,
                side_sl_hits_recent,
                target_market_plus_recent,
                direction_confirmed,
                horizon_confirmed,
            ),
        )

    confidence_add = int(config.SL_STREAK_DIRECTION_FLIP_CONFIDENCE_ADD) + max(
        0,
        max(streak_count, side_sl_hits_recent) - min_streak,
    )
    flipped = _retarget_signal(
        signal,
        side=target_side,
        mode=f"{signal.mode}_slflip",
        confidence_add=confidence_add,
    )
    reason = (
        f"{current_side}->{target_side}:"
        f"slx{streak_count},age={(streak.age_sec if streak is not None else float('inf')):.0f}s,"
        f"slhits={side_sl_hits_recent},mktplus={target_market_plus_recent},"
        f"slrate={side_sl_rate_recent:.2f},m_ovr={int(metrics_override)},"
        f"tech={int(direction_confirmed or horizon_confirmed)},"
        f"force={int(force_flip_by_streak)},"
        f"force_tech={int(force_tech_bypass)}"
    )
    return flipped, reason, SlFlipEval(
        streak,
        side_sl_hits_recent,
        target_market_plus_recent,
        direction_confirmed,
        horizon_confirmed,
    )


def _maybe_side_metrics_direction_flip(
    signal: TickSignal,
    *,
    strategy_tag: str,
    pocket: str,
    now_mono: float,
) -> tuple[Optional[TickSignal], str, SideMetricsFlipEval]:
    global _LAST_SIDE_METRICS_FLIP_MONO

    current_side = str(signal.side or "").strip().lower()
    target_side = "short" if current_side == "long" else "long"
    empty_eval = SideMetricsFlipEval(
        current_side=current_side,
        target_side=target_side,
        current_trades=0,
        target_trades=0,
        current_sl_rate=0.0,
        target_sl_rate=0.0,
        current_market_plus_rate=0.0,
        target_market_plus_rate=0.0,
    )
    if not config.SIDE_METRICS_DIRECTION_FLIP_ENABLED:
        return None, "disabled", empty_eval
    if current_side not in {"long", "short"}:
        return None, "invalid_signal_side", empty_eval

    cooldown_sec = max(0.0, float(config.SIDE_METRICS_DIRECTION_FLIP_COOLDOWN_SEC))
    if now_mono - _LAST_SIDE_METRICS_FLIP_MONO < cooldown_sec:
        return None, "cooldown", empty_eval

    metrics = _load_side_close_metrics(
        strategy_tag=strategy_tag,
        pocket=pocket,
        lookback=max(4, int(config.SIDE_METRICS_DIRECTION_FLIP_LOOKBACK_TRADES)),
        cache_ttl_sec=max(0.1, float(config.SIDE_METRICS_DIRECTION_FLIP_CACHE_TTL_SEC)),
        now_mono=now_mono,
        cache_store=_SIDE_CLOSE_METRICS_FLIP_CACHE,
    )
    if metrics is None:
        return None, "metrics_unavailable", empty_eval

    current_trades = (
        int(metrics.long_trades) if current_side == "long" else int(metrics.short_trades)
    )
    target_trades = (
        int(metrics.short_trades) if current_side == "long" else int(metrics.long_trades)
    )
    current_sl_rate = _clamp(
        float(metrics.sl_hits(current_side)) / float(max(1, current_trades)),
        0.0,
        1.0,
    )
    target_sl_rate = _clamp(
        float(metrics.sl_hits(target_side)) / float(max(1, target_trades)),
        0.0,
        1.0,
    )
    current_market_plus_rate = _clamp(
        float(metrics.market_plus(current_side)) / float(max(1, current_trades)),
        0.0,
        1.0,
    )
    target_market_plus_rate = _clamp(
        float(metrics.market_plus(target_side)) / float(max(1, target_trades)),
        0.0,
        1.0,
    )
    eval_info = SideMetricsFlipEval(
        current_side=current_side,
        target_side=target_side,
        current_trades=current_trades,
        target_trades=target_trades,
        current_sl_rate=current_sl_rate,
        target_sl_rate=target_sl_rate,
        current_market_plus_rate=current_market_plus_rate,
        target_market_plus_rate=target_market_plus_rate,
    )

    min_current_trades = max(1, int(config.SIDE_METRICS_DIRECTION_FLIP_MIN_CURRENT_TRADES))
    min_target_trades = max(1, int(config.SIDE_METRICS_DIRECTION_FLIP_MIN_TARGET_TRADES))
    if current_trades < min_current_trades:
        return None, "current_sample_weak", eval_info
    if target_trades < min_target_trades:
        return None, "target_sample_weak", eval_info

    min_current_sl_rate = _clamp(
        float(config.SIDE_METRICS_DIRECTION_FLIP_MIN_CURRENT_SL_RATE),
        0.0,
        1.0,
    )
    if current_sl_rate < min_current_sl_rate:
        return None, "current_sl_rate_weak", eval_info

    min_sl_gap = max(0.0, float(config.SIDE_METRICS_DIRECTION_FLIP_MIN_SL_GAP))
    if (current_sl_rate - target_sl_rate) < min_sl_gap:
        return None, "sl_gap_weak", eval_info

    min_market_plus_gap = max(
        0.0,
        float(config.SIDE_METRICS_DIRECTION_FLIP_MIN_MARKET_PLUS_GAP),
    )
    if (target_market_plus_rate - current_market_plus_rate) < min_market_plus_gap:
        return None, "market_plus_gap_weak", eval_info

    flipped = _retarget_signal(
        signal,
        side=target_side,
        mode=f"{signal.mode}_smflip",
        confidence_add=int(config.SIDE_METRICS_DIRECTION_FLIP_CONFIDENCE_ADD),
    )
    _LAST_SIDE_METRICS_FLIP_MONO = now_mono
    reason = (
        f"{current_side}->{target_side}:"
        f"sl={current_sl_rate:.2f}/{target_sl_rate:.2f},"
        f"mplus={current_market_plus_rate:.2f}/{target_market_plus_rate:.2f},"
        f"n={current_trades}/{target_trades}"
    )
    return flipped, reason, eval_info


def _range_pos_from_candles(
    *,
    tf: str,
    lookback: int,
    min_span_pips: float,
) -> Optional[float]:
    candles = get_candles_snapshot(tf, limit=max(lookback, 2), include_live=True)
    if len(candles) < max(lookback, 2):
        return None
    window = candles[-lookback:]
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    for row in window:
        if not isinstance(row, dict):
            continue
        high = _safe_float(row.get("high"), 0.0)
        low = _safe_float(row.get("low"), 0.0)
        close = _safe_float(row.get("close"), 0.0)
        if high <= 0.0 or low <= 0.0 or close <= 0.0 or high < low:
            continue
        highs.append(high)
        lows.append(low)
        closes.append(close)
    if len(highs) < max(6, lookback // 2):
        return None

    highest = max(highs)
    lowest = min(lows)
    close_px = closes[-1]
    span = highest - lowest
    if span <= 0.0:
        return None
    span_pips = span / config.PIP_VALUE
    if span_pips < min_span_pips:
        return None

    pos = (close_px - lowest) / span
    return _clamp(pos, 0.0, 1.0)


def _extrema_tech_filter(
    *,
    side: str,
    reason: str,
    factors: Optional[dict],
    regime: Optional[MtfRegime],
) -> tuple[float, str]:
    if not config.EXTREMA_TECH_FILTER_ENABLED:
        return 1.0, str(reason)
    if not isinstance(regime, MtfRegime):
        return 1.0, str(reason)
    if not config.EXTREMA_TECH_FILTER_ALLOW_BLOCK_TO_SOFT:
        return 1.0, str(reason)
    if side not in {"long", "short"}:
        return 1.0, str(reason)

    reason_norm = str(reason).strip().lower()
    if reason_norm not in {"long_top_m1m5", "short_bottom_m1m5"}:
        return 1.0, str(reason)

    adx_mix = 0.5 * (
        _safe_float(getattr(regime, "adx_m1", 0.0))
        + _safe_float(getattr(regime, "adx_m5", 0.0))
    )
    if adx_mix >= config.EXTREMA_TECH_FILTER_ADX_WEAK_MAX:
        return 1.0, str(reason)

    tf_m1 = {}
    if isinstance(factors, dict):
        tf_m1_candidate = factors.get("M1")
        if isinstance(tf_m1_candidate, dict):
            tf_m1 = tf_m1_candidate

    if not tf_m1:
        return 1.0, str(reason)

    close = _safe_float(tf_m1.get("close"), 0.0)
    rsi = _safe_float(tf_m1.get("rsi"), 50.0)
    ema20 = _safe_float(tf_m1.get("ema20"), _safe_float(tf_m1.get("ma20"), 0.0))
    if close <= 0.0 or ema20 <= 0.0:
        return 1.0, str(reason)

    ema_gap_pips = abs(close - ema20) / config.PIP_VALUE
    if ema_gap_pips < config.EXTREMA_TECH_FILTER_REQ_EMA_GAP_PIPS:
        return 1.0, str(reason)

    if side == "long":
        if rsi <= config.EXTREMA_TECH_FILTER_LONG_TOP_RSI_MAX and close < ema20:
            return (
                config.EXTREMA_TECH_FILTER_BLOCK_SOFT_MULT,
                "long_top_m1m5_tech_soft",
            )
    else:
        if rsi >= config.EXTREMA_TECH_FILTER_SHORT_BOTTOM_RSI_MIN and close > ema20:
            return (
                config.EXTREMA_TECH_FILTER_BLOCK_SOFT_MULT,
                "short_bottom_m1m5_tech_soft",
            )

    return 1.0, str(reason)


def _extrema_gate_decision(
    side: str,
    factors: Optional[dict] = None,
    regime: Optional[MtfRegime] = None,
) -> ExtremaGateDecision:
    if not config.EXTREMA_GATE_ENABLED:
        return ExtremaGateDecision(True, "disabled", 1.0, None, None, None)

    m1_pos = _range_pos_from_candles(
        tf="M1",
        lookback=config.EXTREMA_M1_LOOKBACK,
        min_span_pips=config.EXTREMA_M1_MIN_SPAN_PIPS,
    )
    m5_pos = _range_pos_from_candles(
        tf="M5",
        lookback=config.EXTREMA_M5_LOOKBACK,
        min_span_pips=config.EXTREMA_M5_MIN_SPAN_PIPS,
    )
    h4_pos = _range_pos_from_candles(
        tf="H4",
        lookback=config.EXTREMA_H4_LOOKBACK,
        min_span_pips=config.EXTREMA_H4_MIN_SPAN_PIPS,
    )

    missing: list[str] = []
    if m1_pos is None:
        missing.append("m1")
    if m5_pos is None:
        missing.append("m5")
    if h4_pos is None:
        missing.append("h4")

    side_key = str(side or "").lower()
    is_long = side_key in {"buy", "long", "open_long"}
    is_short = side_key in {"sell", "short", "open_short"}
    if not (is_long or is_short):
        return ExtremaGateDecision(True, "unknown_side", 1.0, m1_pos, m5_pos, h4_pos)

    if is_long:
        m1_top = m1_pos is not None and m1_pos >= config.EXTREMA_LONG_TOP_BLOCK_POS
        m5_top = m5_pos is not None and m5_pos >= config.EXTREMA_LONG_TOP_BLOCK_POS
        if config.EXTREMA_REQUIRE_M1_M5_AGREE_LONG:
            top_block = m1_top and m5_top
        else:
            top_block = m1_top or m5_top
        if top_block:
            filter_mult, filter_reason = _extrema_tech_filter(
                side="long",
                reason="long_top_m1m5",
                factors=factors,
                regime=regime,
            )
            if filter_mult < 1.0:
                return ExtremaGateDecision(
                    True,
                    filter_reason,
                    _clamp(
                        float(config.EXTREMA_SOFT_UNITS_MULT) * filter_mult,
                        0.05,
                        1.0,
                    ),
                    m1_pos,
                    m5_pos,
                    h4_pos,
                )
            return ExtremaGateDecision(
                False,
                "long_top_m1m5",
                0.0,
                m1_pos,
                m5_pos,
                h4_pos,
            )
        soft_hit = (
            (m1_pos is not None and m1_pos >= config.EXTREMA_LONG_TOP_SOFT_POS)
            or (m5_pos is not None and m5_pos >= config.EXTREMA_LONG_TOP_SOFT_POS)
        )
        if soft_hit and config.EXTREMA_SOFT_UNITS_MULT < 0.999:
            return ExtremaGateDecision(
                True,
                "long_top_soft",
                config.EXTREMA_SOFT_UNITS_MULT,
                m1_pos,
                m5_pos,
                h4_pos,
            )
    else:
        m1_bottom = m1_pos is not None and m1_pos <= config.EXTREMA_SHORT_BOTTOM_BLOCK_POS
        m5_bottom = m5_pos is not None and m5_pos <= config.EXTREMA_SHORT_BOTTOM_BLOCK_POS
        if config.EXTREMA_REQUIRE_M1_M5_AGREE_SHORT:
            bottom_block = m1_bottom and m5_bottom
        else:
            bottom_block = m1_bottom or m5_bottom
        if bottom_block:
            filter_mult, filter_reason = _extrema_tech_filter(
                side="short",
                reason="short_bottom_m1m5",
                factors=factors,
                regime=regime,
            )
            if filter_mult < 1.0:
                return ExtremaGateDecision(
                    True,
                    filter_reason,
                    _clamp(
                        float(config.EXTREMA_SOFT_UNITS_MULT) * filter_mult,
                        0.05,
                        1.0,
                    ),
                    m1_pos,
                    m5_pos,
                    h4_pos,
                )
            return ExtremaGateDecision(
                False,
                "short_bottom_m1m5",
                0.0,
                m1_pos,
                m5_pos,
                h4_pos,
            )
        if h4_pos is not None and h4_pos <= config.EXTREMA_SHORT_H4_LOW_BLOCK_POS:
            return ExtremaGateDecision(
                False,
                "short_h4_low",
                0.0,
                m1_pos,
                m5_pos,
                h4_pos,
            )
        soft_hit = (
            (m1_pos is not None and m1_pos <= config.EXTREMA_SHORT_BOTTOM_SOFT_POS)
            or (m5_pos is not None and m5_pos <= config.EXTREMA_SHORT_BOTTOM_SOFT_POS)
            or (h4_pos is not None and h4_pos <= config.EXTREMA_SHORT_H4_LOW_SOFT_POS)
        )
        if soft_hit:
            soft_units_mult = float(config.EXTREMA_SHORT_BOTTOM_SOFT_UNITS_MULT)
            soft_reason = "short_bottom_soft"
            if regime is not None:
                regime_mode = str(getattr(regime, "mode", "")).strip().lower()
                regime_side = str(getattr(regime, "side", "")).strip().lower()
                if regime_mode == "balanced" and regime_side != "short":
                    soft_units_mult = min(
                        soft_units_mult,
                        float(config.EXTREMA_SHORT_BOTTOM_SOFT_BALANCED_UNITS_MULT),
                    )
                    soft_reason = "short_bottom_soft_balanced"
            if soft_units_mult < 0.999:
                return ExtremaGateDecision(
                    True,
                    soft_reason,
                    soft_units_mult,
                    m1_pos,
                    m5_pos,
                    h4_pos,
                )
        if soft_hit and config.EXTREMA_SOFT_UNITS_MULT < 0.999:
            return ExtremaGateDecision(
                True,
                "short_bottom_soft",
                config.EXTREMA_SOFT_UNITS_MULT,
                m1_pos,
                m5_pos,
                h4_pos,
            )

    if missing:
        reason = "missing_" + "_".join(missing)
        if config.EXTREMA_FAIL_OPEN:
            return ExtremaGateDecision(True, reason, 1.0, m1_pos, m5_pos, h4_pos)
        return ExtremaGateDecision(False, reason, 0.0, m1_pos, m5_pos, h4_pos)

    return ExtremaGateDecision(True, "ok", 1.0, m1_pos, m5_pos, h4_pos)


def _extrema_reversal_route(
    signal: TickSignal,
    extrema_decision: ExtremaGateDecision,
    *,
    regime: Optional[MtfRegime],
    horizon: Optional[HorizonBias],
    factors: Optional[dict],
) -> tuple[Optional[TickSignal], float, str, float]:
    if not config.EXTREMA_REVERSAL_ENABLED:
        return None, 1.0, "", 0.0

    side_key = str(signal.side or "").strip().lower()
    if side_key not in {"long", "short"}:
        return None, 1.0, "", 0.0
    if side_key == "long" and not config.EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT:
        return None, 1.0, "", 0.0
    reverse_side = "short" if side_key == "long" else "long"

    reason = str(extrema_decision.reason or "").strip().lower()
    if not any(token in reason for token in ("top", "bottom", "h4_low")):
        return None, 1.0, "", 0.0

    score = 0.0
    m1_pos = extrema_decision.m1_pos
    m5_pos = extrema_decision.m5_pos
    h4_pos = extrema_decision.h4_pos
    if side_key == "short":
        if m1_pos is not None and m1_pos <= config.EXTREMA_SHORT_BOTTOM_SOFT_POS:
            score += 0.70
        if m1_pos is not None and m1_pos <= config.EXTREMA_SHORT_BOTTOM_BLOCK_POS:
            score += 0.55
        if m5_pos is not None and m5_pos <= config.EXTREMA_SHORT_BOTTOM_SOFT_POS:
            score += 0.70
        if m5_pos is not None and m5_pos <= config.EXTREMA_SHORT_BOTTOM_BLOCK_POS:
            score += 0.55
        if h4_pos is not None and h4_pos <= config.EXTREMA_SHORT_H4_LOW_SOFT_POS:
            score += 0.45
        if h4_pos is not None and h4_pos <= config.EXTREMA_SHORT_H4_LOW_BLOCK_POS:
            score += 0.35
    else:
        if m1_pos is not None and m1_pos >= config.EXTREMA_LONG_TOP_SOFT_POS:
            score += 0.70
        if m1_pos is not None and m1_pos >= config.EXTREMA_LONG_TOP_BLOCK_POS:
            score += 0.55
        if m5_pos is not None and m5_pos >= config.EXTREMA_LONG_TOP_SOFT_POS:
            score += 0.70
        if m5_pos is not None and m5_pos >= config.EXTREMA_LONG_TOP_BLOCK_POS:
            score += 0.55

    if regime is not None:
        regime_side = str(getattr(regime, "side", "")).strip().lower()
        regime_mode = str(getattr(regime, "mode", "")).strip().lower()
        regime_heat = _safe_float(getattr(regime, "heat_score", 0.0), 0.0)
        if regime_side == reverse_side:
            score += 0.80 if regime_mode == "continuation" else 0.60
        elif (
            regime_side == side_key
            and regime_mode == "continuation"
            and regime_heat >= config.EXTREMA_REVERSAL_CONTINUATION_HEAT_MAX
        ):
            score -= 0.85
        elif regime_mode != "continuation" and regime_heat <= config.EXTREMA_REVERSAL_CONTINUATION_HEAT_MAX:
            score += 0.25

    if horizon is not None:
        horizon_side = str(getattr(horizon, "composite_side", "")).strip().lower()
        horizon_score = abs(_safe_float(getattr(horizon, "composite_score", 0.0), 0.0))
        horizon_agree = int(_safe_float(getattr(horizon, "agreement", 0.0), 0.0))
        if (
            horizon_side == reverse_side
            and horizon_score >= config.EXTREMA_REVERSAL_HORIZON_SCORE_MIN
        ):
            score += 0.80
            if horizon_agree >= config.EXTREMA_REVERSAL_HORIZON_AGREE_MIN:
                score += 0.35
        elif horizon_side == side_key and horizon_score >= config.HORIZON_BLOCK_SCORE:
            score -= 0.60

    tf_m1 = factors.get("M1") if isinstance(factors, dict) else {}
    if isinstance(tf_m1, dict):
        close = _safe_float(tf_m1.get("close"), 0.0)
        ema20 = _safe_float(tf_m1.get("ema20"), _safe_float(tf_m1.get("ma20"), 0.0))
        rsi = _safe_float(tf_m1.get("rsi"), 50.0)
        if close > 0.0 and ema20 > 0.0:
            ema_gap_pips = abs(close - ema20) / config.PIP_VALUE
            if ema_gap_pips >= config.EXTREMA_REVERSAL_EMA_GAP_MIN_PIPS:
                if reverse_side == "long" and close > ema20 and rsi >= config.EXTREMA_REVERSAL_RSI_CONFIRM:
                    score += 1.10
                elif (
                    reverse_side == "short"
                    and close < ema20
                    and rsi <= (100.0 - config.EXTREMA_REVERSAL_RSI_CONFIRM)
                ):
                    score += 1.10

    min_score = float(config.EXTREMA_REVERSAL_MIN_SCORE)
    if side_key == "long":
        min_score = max(min_score, float(config.EXTREMA_REVERSAL_LONG_TO_SHORT_MIN_SCORE))
    if score < min_score:
        return None, 1.0, "", float(score)

    reversed_signal = _retarget_signal(
        signal,
        side=reverse_side,
        mode=f"{signal.mode}_extrev",
        confidence_add=int(config.EXTREMA_REVERSAL_CONFIDENCE_ADD),
    )
    return (
        reversed_signal,
        max(0.1, float(config.EXTREMA_REVERSAL_UNITS_MULT)),
        f"{extrema_decision.reason}_reverse",
        float(score),
    )


def _confidence_scale(conf: int) -> float:
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    min_mult = max(0.1, float(config.CONFIDENCE_SCALE_MIN_MULT))
    max_mult = max(min_mult, float(config.CONFIDENCE_SCALE_MAX_MULT))
    if conf <= lo:
        return min_mult
    if conf >= hi:
        return max_mult
    ratio = (conf - lo) / max(1.0, hi - lo)
    return min_mult + ratio * (max_mult - min_mult)


def _probability_side_edge(side: str, raw_score: float) -> float:
    side_key = str(side or "").strip().lower()
    score = _safe_float(raw_score, 0.0)
    if side_key == "short":
        score *= -1.0
    return _clamp(score, -1.0, 1.0)


def _raw_entry_probability(signal: TickSignal) -> float:
    raw = _safe_float(getattr(signal, "confidence", 0.0), 0.0)
    if raw > 1.0:
        raw /= 100.0
    return float(_clamp(raw, 0.0, 1.0))


def _adjust_entry_probability_alignment(
    *,
    signal: TickSignal,
    raw_probability: float,
    direction_bias: Optional[DirectionBias],
    horizon: Optional[HorizonBias],
    m1_score: Optional[float],
) -> tuple[float, float, dict[str, object]]:
    side = str(signal.side or "").strip().lower()
    mode = str(signal.mode or "").strip().lower()
    raw = float(_clamp(raw_probability, 0.0, 1.0))
    meta: dict[str, object] = {
        "enabled": bool(config.ENTRY_PROBABILITY_ALIGN_ENABLED),
        "raw": raw,
        "adjusted": raw,
        "units_mult": 1.0,
        "support": 0.0,
        "counter": 0.0,
        "boost": 0.0,
        "penalty": 0.0,
        "counter_extra": 0.0,
        "direction_edge": None,
        "horizon_edge": None,
        "m1_edge": None,
        "floor_applied": False,
        "floor_block_reason": "",
    }
    if not config.ENTRY_PROBABILITY_ALIGN_ENABLED:
        return raw, 1.0, meta
    if side not in {"long", "short"}:
        meta["reason"] = "invalid_side"
        return raw, 1.0, meta

    weighted_edges: list[tuple[str, float, float]] = []
    direction_weight = max(0.0, float(config.ENTRY_PROBABILITY_ALIGN_DIRECTION_WEIGHT))
    horizon_weight = max(0.0, float(config.ENTRY_PROBABILITY_ALIGN_HORIZON_WEIGHT))
    m1_weight = max(0.0, float(config.ENTRY_PROBABILITY_ALIGN_M1_WEIGHT))

    if direction_bias is not None and direction_weight > 0.0:
        direction_edge = _probability_side_edge(side, _safe_float(direction_bias.score, 0.0))
        meta["direction_edge"] = round(direction_edge, 6)
        weighted_edges.append(("direction", direction_edge, direction_weight))

    horizon_edge = 0.0
    if horizon is not None and horizon_weight > 0.0:
        horizon_edge_raw = (
            0.7 * (_safe_float(horizon.long_score, 0.0) - _safe_float(horizon.short_score, 0.0))
            + 0.3 * _safe_float(horizon.composite_score, 0.0)
        )
        horizon_edge = _probability_side_edge(side, horizon_edge_raw)
        meta["horizon_edge"] = round(horizon_edge, 6)
        weighted_edges.append(("horizon", horizon_edge, horizon_weight))

    if m1_score is not None and m1_weight > 0.0:
        m1_edge = _probability_side_edge(side, _safe_float(m1_score, 0.0))
        meta["m1_edge"] = round(m1_edge, 6)
        weighted_edges.append(("m1", m1_edge, m1_weight))

    total_weight = sum(w for _, _, w in weighted_edges if w > 0.0)
    if total_weight <= 0.0:
        meta["reason"] = "no_alignment_inputs"
        return raw, 1.0, meta

    support = sum(w * max(0.0, edge) for _, edge, w in weighted_edges) / total_weight
    counter = sum(w * max(0.0, -edge) for _, edge, w in weighted_edges) / total_weight
    boost = support * max(0.0, float(config.ENTRY_PROBABILITY_ALIGN_BOOST_MAX))
    penalty = counter * max(0.0, float(config.ENTRY_PROBABILITY_ALIGN_PENALTY_MAX))
    if mode.startswith("revert"):
        penalty *= _clamp(float(config.ENTRY_PROBABILITY_ALIGN_REVERT_PENALTY_MULT), 0.0, 1.0)
    counter_extra = max(0.0, -horizon_edge) * max(
        0.0, float(config.ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX)
    )
    penalty = min(0.98, penalty + counter_extra)

    adjusted = raw * (1.0 - penalty)
    if boost > 0.0:
        adjusted += (1.0 - adjusted) * min(0.95, boost)

    floor_raw_min = _clamp(float(config.ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN), 0.0, 1.0)
    floor_prob = _clamp(float(config.ENTRY_PROBABILITY_ALIGN_FLOOR), 0.0, 1.0)
    floor_requires_support = bool(config.ENTRY_PROBABILITY_ALIGN_FLOOR_REQUIRE_SUPPORT)
    floor_counter_max = _clamp(float(config.ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER), 0.0, 1.0)
    floor_applied = False
    floor_block_reason = ""
    floor_allowed = True
    if floor_requires_support and support < counter:
        floor_allowed = False
        floor_block_reason = "support_lt_counter"
    if floor_allowed and counter > floor_counter_max:
        floor_allowed = False
        floor_block_reason = "counter_too_high"
    if raw >= floor_raw_min and adjusted < floor_prob and floor_allowed:
        adjusted = floor_prob
        floor_applied = True

    adjusted = _clamp(
        adjusted,
        float(config.ENTRY_PROBABILITY_ALIGN_MIN),
        float(config.ENTRY_PROBABILITY_ALIGN_MAX),
    )

    units_mult = 1.0
    if config.ENTRY_PROBABILITY_ALIGN_UNITS_FOLLOW_ENABLED and raw > 0.0:
        ratio = adjusted / raw
        units_mult = _clamp(
            ratio,
            float(config.ENTRY_PROBABILITY_ALIGN_UNITS_MIN_MULT),
            float(config.ENTRY_PROBABILITY_ALIGN_UNITS_MAX_MULT),
        )

    meta.update(
        {
            "adjusted": round(adjusted, 6),
            "units_mult": round(units_mult, 6),
            "support": round(support, 6),
            "counter": round(counter, 6),
            "boost": round(boost, 6),
            "penalty": round(penalty, 6),
            "counter_extra": round(counter_extra, 6),
            "floor_applied": bool(floor_applied),
            "floor_block_reason": floor_block_reason,
            "weights": {
                "direction": round(direction_weight, 6),
                "horizon": round(horizon_weight, 6),
                "m1": round(m1_weight, 6),
            },
        }
    )
    return float(adjusted), float(units_mult), meta


def _entry_probability_band_units_multiplier(
    *,
    strategy_tag: str,
    pocket: str,
    side: str,
    entry_probability: float,
    now_mono: Optional[float] = None,
) -> tuple[float, dict[str, object]]:
    side_key = str(side or "").strip().lower()
    probability = _clamp(float(entry_probability), 0.0, 1.0)
    min_mult = max(0.05, float(config.ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MIN_MULT))
    max_mult = max(min_mult, float(config.ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MAX_MULT))
    meta: dict[str, object] = {
        "enabled": bool(config.ENTRY_PROBABILITY_BAND_ALLOC_ENABLED),
        "entry_probability": round(probability, 6),
        "bucket": "mid",
        "band_mult": 1.0,
        "side_mult": 1.0,
        "units_mult": 1.0,
        "shift_strength": 0.0,
        "sample_strength": 0.0,
        "gap_pips": 0.0,
        "gap_win_rate": 0.0,
        "gap_sl_rate": 0.0,
        "high_sample": 0,
        "low_sample": 0,
        "high_mean_pips": 0.0,
        "low_mean_pips": 0.0,
        "high_win_rate": 0.0,
        "low_win_rate": 0.0,
        "high_sl_rate": 0.0,
        "low_sl_rate": 0.0,
        "side_trades": 0,
        "side_sl_hits": 0,
        "side_market_plus": 0,
        "side_sl_rate": 0.0,
        "side_market_plus_rate": 0.0,
        "reason": "disabled",
    }
    if not config.ENTRY_PROBABILITY_BAND_ALLOC_ENABLED:
        return 1.0, meta
    if side_key not in {"long", "short"}:
        meta["reason"] = "invalid_side"
        return 1.0, meta

    low_cutoff = _clamp(float(config.ENTRY_PROBABILITY_BAND_ALLOC_LOW_THRESHOLD), 0.0, 1.0)
    high_cutoff = _clamp(float(config.ENTRY_PROBABILITY_BAND_ALLOC_HIGH_THRESHOLD), 0.0, 1.0)
    if low_cutoff >= high_cutoff:
        meta["reason"] = "invalid_threshold"
        return 1.0, meta

    if probability < low_cutoff:
        bucket = "low"
    elif probability >= high_cutoff:
        bucket = "high"
    else:
        bucket = "mid"
    meta["bucket"] = bucket

    metrics = _load_entry_probability_band_metrics(
        strategy_tag=strategy_tag,
        pocket=pocket,
        side=side_key,
        now_mono=now_mono,
    )
    if metrics is None:
        meta["reason"] = "no_metrics"
        return 1.0, meta

    meta.update(
        {
            "high_sample": int(metrics.high_sample),
            "low_sample": int(metrics.low_sample),
            "high_mean_pips": round(float(metrics.high_mean_pips), 6),
            "low_mean_pips": round(float(metrics.low_mean_pips), 6),
            "high_win_rate": round(float(metrics.high_win_rate), 6),
            "low_win_rate": round(float(metrics.low_win_rate), 6),
            "high_sl_rate": round(float(metrics.high_sl_rate), 6),
            "low_sl_rate": round(float(metrics.low_sl_rate), 6),
        }
    )

    min_band_sample = max(1, int(config.ENTRY_PROBABILITY_BAND_ALLOC_MIN_TRADES_PER_BAND))
    band_mult = 1.0
    band_reason = "insufficient_band_sample"
    if metrics.high_sample >= min_band_sample and metrics.low_sample >= min_band_sample:
        pips_ref = max(0.05, float(config.ENTRY_PROBABILITY_BAND_ALLOC_GAP_PIPS_REF))
        win_ref = max(0.01, float(config.ENTRY_PROBABILITY_BAND_ALLOC_GAP_WIN_RATE_REF))
        sl_ref = max(0.01, float(config.ENTRY_PROBABILITY_BAND_ALLOC_GAP_SL_RATE_REF))
        gap_pips = float(metrics.low_mean_pips - metrics.high_mean_pips)
        gap_win_rate = float(metrics.low_win_rate - metrics.high_win_rate)
        gap_sl_rate = float(metrics.high_sl_rate - metrics.low_sl_rate)
        edge_strength = (
            _clamp(gap_pips / pips_ref, 0.0, 1.0)
            + _clamp(gap_win_rate / win_ref, 0.0, 1.0)
            + _clamp(gap_sl_rate / sl_ref, 0.0, 1.0)
        ) / 3.0
        strong_sample = max(1, int(config.ENTRY_PROBABILITY_BAND_ALLOC_SAMPLE_STRONG_TRADES))
        sample_strength = _clamp(
            min(metrics.high_sample, metrics.low_sample) / float(strong_sample),
            0.0,
            1.0,
        )
        shift_strength = _clamp(edge_strength * sample_strength, 0.0, 1.0)
        high_target = 1.0 - (
            max(0.0, float(config.ENTRY_PROBABILITY_BAND_ALLOC_HIGH_REDUCE_MAX))
            * shift_strength
        )
        low_target = 1.0 + (
            max(0.0, float(config.ENTRY_PROBABILITY_BAND_ALLOC_LOW_BOOST_MAX))
            * shift_strength
        )
        high_target = _clamp(high_target, min_mult, max_mult)
        low_target = _clamp(low_target, min_mult, max_mult)
        if bucket == "high":
            band_mult = high_target
        elif bucket == "low":
            band_mult = low_target
        else:
            band_mult = _lerp(
                low_target,
                high_target,
                (probability - low_cutoff) / max(1e-6, high_cutoff - low_cutoff),
            )
        band_reason = "ok"
        meta.update(
            {
                "shift_strength": round(float(shift_strength), 6),
                "sample_strength": round(float(sample_strength), 6),
                "gap_pips": round(gap_pips, 6),
                "gap_win_rate": round(gap_win_rate, 6),
                "gap_sl_rate": round(gap_sl_rate, 6),
            }
        )

    side_mult = 1.0
    if config.ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_ENABLED:
        side_metrics = _load_recent_side_close_metrics_for_allocation(
            strategy_tag=strategy_tag,
            pocket=pocket,
            now_mono=now_mono,
        )
        if side_metrics is not None:
            side_trades = (
                int(side_metrics.long_trades)
                if side_key == "long"
                else int(side_metrics.short_trades)
            )
            if side_trades > 0:
                side_sl_hits = int(side_metrics.sl_hits(side_key))
                side_market_plus = int(side_metrics.market_plus(side_key))
                side_sl_rate = _clamp(side_sl_hits / float(side_trades), 0.0, 1.0)
                side_market_plus_rate = _clamp(
                    side_market_plus / float(side_trades),
                    0.0,
                    1.0,
                )
                side_balance = side_market_plus_rate - side_sl_rate
                side_mult = _clamp(
                    1.0 + (
                        side_balance
                        * max(
                            0.0,
                            float(config.ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_GAIN),
                        )
                    ),
                    float(config.ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT),
                    float(config.ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MAX_MULT),
                )
                meta.update(
                    {
                        "side_trades": side_trades,
                        "side_sl_hits": side_sl_hits,
                        "side_market_plus": side_market_plus,
                        "side_sl_rate": round(side_sl_rate, 6),
                        "side_market_plus_rate": round(side_market_plus_rate, 6),
                    }
                )

    units_mult = _clamp(band_mult * side_mult, min_mult, max_mult)
    meta.update(
        {
            "band_mult": round(float(band_mult), 6),
            "side_mult": round(float(side_mult), 6),
            "units_mult": round(float(units_mult), 6),
            "reason": band_reason,
        }
    )
    return float(units_mult), meta


def _compute_targets(
    *,
    spread_pips: float,
    momentum_pips: float,
    side: str,
    tp_profile: TpTimingProfile,
    regime: Optional[MtfRegime],
    signal_range_pips: float,
    signal_instant_range_pips: float = 0.0,
) -> tuple[float, float]:
    is_short = str(side).strip().lower() == "short"
    tp_base_cfg = config.SHORT_TP_BASE_PIPS if is_short else config.TP_BASE_PIPS
    tp_net_min_cfg = config.SHORT_TP_NET_MIN_PIPS if is_short else config.TP_NET_MIN_PIPS
    tp_max_cfg = config.SHORT_TP_MAX_PIPS if is_short else config.TP_MAX_PIPS
    tp_bonus_cap_cfg = (
        config.SHORT_TP_MOMENTUM_BONUS_MAX
        if is_short
        else config.TP_MOMENTUM_BONUS_MAX
    )
    sl_min_cfg = config.SHORT_SL_MIN_PIPS if is_short else config.SL_MIN_PIPS
    sl_base_cfg = config.SHORT_SL_BASE_PIPS if is_short else config.SL_BASE_PIPS
    sl_max_cfg = config.SHORT_SL_MAX_PIPS if is_short else config.SL_MAX_PIPS

    # Time-aware TP scaling for ping scalp:
    # if average TP holding time drifts too long, reduce TP net edge.
    tp_mult = max(config.TP_TIME_MULT_MIN, min(config.TP_TIME_MULT_MAX, tp_profile.multiplier))
    if config.TP_VOL_ADAPT_ENABLED:
        vol_bucket = _norm01(
            float(signal_range_pips),
            config.TP_VOL_LOW_PIPS,
            config.TP_VOL_HIGH_PIPS,
        )
        if vol_bucket <= 0.0:
            vol_bucket = _regime_vol_bucket(
                regime,
                low_pips=config.TP_VOL_LOW_PIPS,
                high_pips=config.TP_VOL_HIGH_PIPS,
            )
        if config.INSTANT_VOL_ADAPT_ENABLED and signal_instant_range_pips > 0.0:
            instant_vol_bucket = _norm01(
                float(signal_instant_range_pips),
                config.TP_VOL_LOW_PIPS,
                config.TP_VOL_HIGH_PIPS,
            )
            vol_bucket = _lerp(
                vol_bucket,
                instant_vol_bucket,
                config.INSTANT_VOL_BUCKET_WEIGHT,
            )
        low_vol_target = (
            config.TP_VOL_MULT_LOW_VOL_MIN + config.TP_VOL_MULT_LOW_VOL_MAX
        ) * 0.5
        high_vol_target = (
            config.TP_VOL_MULT_HIGH_VOL_MIN + config.TP_VOL_MULT_HIGH_VOL_MAX
        ) * 0.5
        vol_mult = _lerp(low_vol_target, high_vol_target, vol_bucket)
        tp_mult = _clamp(
            tp_mult * vol_mult,
            config.TP_TIME_MULT_MIN,
            config.TP_TIME_MULT_MAX,
        )
        if config.TP_VOL_EXTEND_MAX_MULT > 1.0:
            tp_mult *= _lerp(1.0, config.TP_VOL_EXTEND_MAX_MULT, vol_bucket)
    if config.TP_ENABLED:
        tp_base = max(0.1, tp_base_cfg * tp_mult)
        tp_net_edge = max(config.TP_NET_MIN_FLOOR_PIPS, tp_net_min_cfg * tp_mult)
        tp_floor = spread_pips + tp_net_edge
        tp_pips = max(tp_base, tp_floor)
        tp_bonus_raw = max(0.0, abs(momentum_pips) - tp_pips)
        tp_bonus_cap = tp_bonus_cap_cfg * max(0.6, tp_mult)
        tp_pips += min(tp_bonus_cap, tp_bonus_raw * 0.25)
        tp_pips = min(tp_max_cfg, tp_pips)
    else:
        tp_pips = 0.0

    sl_floor = spread_pips * config.SL_SPREAD_MULT + config.SL_SPREAD_BUFFER_PIPS
    sl_pips = max(sl_min_cfg, sl_base_cfg, sl_floor)
    if config.SL_VOL_SHRINK_ENABLED:
        sl_vol_bucket = _norm01(
            float(signal_range_pips),
            config.TP_VOL_LOW_PIPS,
            config.TP_VOL_HIGH_PIPS,
        )
        if config.INSTANT_VOL_ADAPT_ENABLED and signal_instant_range_pips > 0.0:
            instant_sl_vol_bucket = _norm01(
                float(signal_instant_range_pips),
                config.TP_VOL_LOW_PIPS,
                config.TP_VOL_HIGH_PIPS,
            )
            sl_vol_bucket = _lerp(
                sl_vol_bucket,
                instant_sl_vol_bucket,
                config.INSTANT_VOL_BUCKET_WEIGHT,
            )
        sl_pips *= _lerp(
            1.0,
            config.SL_VOL_SHRINK_MIN_MULT,
            sl_vol_bucket,
        )
    if config.TP_ENABLED and abs(momentum_pips) > tp_pips * 1.4:
        sl_pips *= 1.08
    sl_pips = max(sl_min_cfg, min(sl_max_cfg, sl_pips))

    return tp_pips, sl_pips


def _strategy_trade_counts(pocket_info: dict, strategy_tag: str) -> tuple[int, int, int]:
    open_trades = pocket_info.get("open_trades") if isinstance(pocket_info, dict) else None
    if not isinstance(open_trades, list):
        return 0, 0, 0

    tag_key = (strategy_tag or "").strip().lower()
    total = 0
    long_count = 0
    short_count = 0

    for tr in open_trades:
        if not isinstance(tr, dict):
            continue
        raw_tag = tr.get("strategy_tag")
        if not raw_tag:
            thesis = tr.get("entry_thesis")
            if isinstance(thesis, dict):
                raw_tag = thesis.get("strategy_tag") or thesis.get("strategy")
        if tag_key and str(raw_tag or "").strip().lower() != tag_key:
            continue
        units = int(_safe_float(tr.get("units"), 0.0))
        if units == 0:
            continue
        total += 1
        if units > 0:
            long_count += 1
        else:
            short_count += 1

    return total, long_count, short_count


def _allow_signal_when_max_active(
    *,
    side: str,
    active_total: int,
    active_long: int,
    active_short: int,
    max_active_trades: int,
) -> bool:
    if active_total < max_active_trades:
        return True
    if not config.ALLOW_OPPOSITE_WHEN_MAX_ACTIVE:
        return False
    if side == "long":
        return active_short > active_long
    if side == "short":
        return active_long > active_short
    return False


def _resolve_active_caps(
    *,
    free_ratio: float,
    margin_available: float,
) -> tuple[int, int, bool]:
    total_cap = config.MAX_ACTIVE_TRADES
    side_cap = config.MAX_PER_DIRECTION
    if not config.ACTIVE_CAP_MARGIN_BYPASS_ENABLED:
        return total_cap, side_cap, False
    if free_ratio < config.ACTIVE_CAP_BYPASS_MIN_FREE_RATIO:
        return total_cap, side_cap, False
    if margin_available < config.ACTIVE_CAP_BYPASS_MIN_MARGIN_AVAILABLE_JPY:
        return total_cap, side_cap, False
    total_cap = max(total_cap, config.MAX_ACTIVE_TRADES + config.ACTIVE_CAP_BYPASS_EXTRA_TOTAL)
    side_cap = max(side_cap, config.MAX_PER_DIRECTION + config.ACTIVE_CAP_BYPASS_EXTRA_PER_DIRECTION)
    return total_cap, side_cap, True


def _compute_trap_state(positions: dict, *, mid_price: float) -> TrapState:
    if not isinstance(positions, dict):
        return TrapState(False, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    long_units = 0.0
    short_units = 0.0
    long_weighted = 0.0
    short_weighted = 0.0
    unrealized_pl = 0.0

    for pocket, info in positions.items():
        if str(pocket).startswith("__") or not isinstance(info, dict):
            continue
        pocket_long = max(0.0, _safe_float(info.get("long_units"), 0.0))
        pocket_short = max(0.0, _safe_float(info.get("short_units"), 0.0))
        long_avg = _safe_float(info.get("long_avg_price"), 0.0)
        short_avg = _safe_float(info.get("short_avg_price"), 0.0)
        long_units += pocket_long
        short_units += pocket_short
        if pocket_long > 0.0 and long_avg > 0.0:
            long_weighted += pocket_long * long_avg
        if pocket_short > 0.0 and short_avg > 0.0:
            short_weighted += pocket_short * short_avg
        unrealized_pl += _safe_float(info.get("unrealized_pl"), 0.0)

    total_units = long_units + short_units
    if total_units <= 0.0:
        return TrapState(False, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    net_ratio = abs(long_units - short_units) / max(total_units, 1.0)
    mid = _safe_float(mid_price, 0.0)
    long_avg_all = long_weighted / long_units if long_units > 0.0 else 0.0
    short_avg_all = short_weighted / short_units if short_units > 0.0 else 0.0
    long_dd_pips = (
        max(0.0, (long_avg_all - mid) / config.PIP_VALUE)
        if (long_avg_all > 0.0 and mid > 0.0)
        else 0.0
    )
    short_dd_pips = (
        max(0.0, (mid - short_avg_all) / config.PIP_VALUE)
        if (short_avg_all > 0.0 and mid > 0.0)
        else 0.0
    )
    combined_dd_pips = long_dd_pips + short_dd_pips

    active = (
        long_units >= float(config.TRAP_MIN_LONG_UNITS)
        and short_units >= float(config.TRAP_MIN_SHORT_UNITS)
        and net_ratio <= config.TRAP_MAX_NET_RATIO
        and combined_dd_pips >= config.TRAP_MIN_COMBINED_DD_PIPS
    )
    if active and config.TRAP_REQUIRE_NET_LOSS and unrealized_pl > 0.0:
        active = False

    return TrapState(
        active=active,
        long_units=long_units,
        short_units=short_units,
        net_ratio=net_ratio,
        long_dd_pips=long_dd_pips,
        short_dd_pips=short_dd_pips,
        combined_dd_pips=combined_dd_pips,
        unrealized_pl=unrealized_pl,
    )


def _client_order_id(side: str) -> str:
    ts_ms = int(time.time() * 1000)
    tag = str(config.STRATEGY_TAG or "scalp_ping_5s").strip() or "scalp_ping_5s"
    sanitized_tag = "".join(
        ch.lower() for ch in tag if (ch.isalnum() or ch in {"-", "_"})
    )
    sanitized_tag = sanitized_tag[:24] or "scalp_ping_5s"
    digest = hashlib.sha1(
        f"{sanitized_tag}-{side}-{ts_ms}".encode("utf-8")
    ).hexdigest()[:8]
    return f"qr-{ts_ms}-{sanitized_tag}-{side[0]}{digest}"


def _snapshot_retry_delay_seconds(failure_count: int) -> float:
    if failure_count <= 0:
        return 0.0
    multiplier = max(1.1, float(config.SNAPSHOT_FETCH_RETRY_BACKOFF_MULTIPLIER))
    delay = float(config.SNAPSHOT_FETCH_RETRY_BASE_SEC) * (multiplier ** (failure_count - 1))
    return min(float(config.SNAPSHOT_FETCH_RETRY_MAX_SEC), delay)


def _snapshot_is_transient_status(status_code: int) -> bool:
    return status_code in {429, 500, 502, 503, 504}


def _is_oanda_snapshot_auth_valid(logger: logging.Logger) -> bool:
    global _SNAPSHOT_AUTH_VALIDATED
    if _SNAPSHOT_AUTH_VALIDATED is not None:
        return _SNAPSHOT_AUTH_VALIDATED

    if not _OANDA_TOKEN or not _OANDA_ACCOUNT:
        logger.error("%s oanda credentials missing (account=%s token=%s)", config.LOG_PREFIX, _mask_value(_OANDA_ACCOUNT), bool(_OANDA_TOKEN))
        _SNAPSHOT_AUTH_VALIDATED = False
        return False

    validate_url = f"{_PRICING_HOST}/v3/accounts/{_OANDA_ACCOUNT}"
    try:
        with httpx.Client(timeout=config.SNAPSHOT_FETCH_TIMEOUT_SEC) as client:
            resp = client.get(validate_url, headers=_PRICING_HEADERS)
            status_code = int(resp.status_code)
            if status_code == 200:
                _SNAPSHOT_AUTH_VALIDATED = True
                return True

            body = ""
            try:
                body_obj = resp.json()
                body = str(body_obj.get("errorMessage", ""))[:140]
            except Exception:
                body = resp.text[:140]
            if status_code in {400, 401, 403}:
                logger.error(
                    "%s oanda auth precheck failed (account=%s, status=%s): %s",
                    config.LOG_PREFIX,
                    _mask_value(_OANDA_ACCOUNT),
                    status_code,
                    body,
                )
                _SNAPSHOT_AUTH_VALIDATED = False
                return False

            logger.warning(
                "%s oanda auth precheck failed transiently (account=%s, status=%s): %s",
                config.LOG_PREFIX,
                _mask_value(_OANDA_ACCOUNT),
                status_code,
                body,
            )
            _SNAPSHOT_AUTH_VALIDATED = None
            return False
    except httpx.HTTPError as exc:
        logger.warning("%s oanda auth precheck request failed: %s", config.LOG_PREFIX, exc)
        _SNAPSHOT_AUTH_VALIDATED = None
        return False


def _snapshot_is_degraded_mode() -> bool:
    return (
        config.SNAPSHOT_FETCH_FAILURE_STALE_MODE_ENABLED
        and _SNAPSHOT_FETCH_FAILURES >= config.SNAPSHOT_FETCH_FAILURE_STALE_THRESHOLD
    )


def _snapshot_stale_age_limit_ms() -> float:
    base_age_ms = float(config.MAX_TICK_AGE_MS)
    if not _snapshot_is_degraded_mode():
        return base_age_ms
    return max(base_age_ms, float(config.SNAPSHOT_FETCH_FAILURE_STALE_MAX_AGE_MS))


def _snapshot_stale_units_scale() -> float:
    if not _snapshot_is_degraded_mode():
        return 1.0
    return float(config.SNAPSHOT_FETCH_FAILURE_STALE_UNITS_SCALE)


async def _safe_get_open_positions(
    pos_manager: PositionManager,
    *,
    logger: logging.Logger,
    timeout_sec: float,
) -> tuple[dict[str, dict], Optional[str]]:
    start = time.monotonic()
    try:
        payload = await asyncio.wait_for(
            asyncio.to_thread(pos_manager.get_open_positions),
            timeout=timeout_sec,
        )
        if not isinstance(payload, dict):
            logger.warning(
                "%s position_manager open_positions returned non-dict payload=%s",
                config.LOG_PREFIX,
                type(payload).__name__,
            )
            return {}, "position_manager_invalid_payload"
        return payload, None
    except asyncio.TimeoutError:
        logger.warning(
            "%s position_manager open_positions timeout after %.2fs",
            config.LOG_PREFIX,
            time.monotonic() - start,
        )
        return {}, "position_manager_timeout"
    except Exception:
        logger.exception(
            "%s position_manager open_positions failed after %.2fs",
            config.LOG_PREFIX,
            time.monotonic() - start,
        )
        return {}, "position_manager_error"


async def _fetch_price_snapshot(logger: logging.Logger) -> bool:
    global _SNAPSHOT_FETCH_FAILURES, _SNAPSHOT_FETCH_BACKOFF_UNTIL_MONO, _SNAPSHOT_FETCH_BACKOFF_LOG_MONO
    if not _OANDA_TOKEN or not _OANDA_ACCOUNT:
        return False
    if not _is_oanda_snapshot_auth_valid(logger):
        return False

    now_mono = time.monotonic()
    if _SNAPSHOT_FETCH_BACKOFF_UNTIL_MONO > now_mono:
        if now_mono - _SNAPSHOT_FETCH_BACKOFF_LOG_MONO >= 10.0:
            _SNAPSHOT_FETCH_BACKOFF_LOG_MONO = now_mono
            logger.warning(
                "%s snapshot fetch waiting retry backoff: %.1fs remaining",
                config.LOG_PREFIX,
                max(0.0, _SNAPSHOT_FETCH_BACKOFF_UNTIL_MONO - now_mono),
            )
        return False

    params = {"instruments": "USD_JPY"}
    try:
        async with httpx.AsyncClient(timeout=config.SNAPSHOT_FETCH_TIMEOUT_SEC) as client:
            resp = await client.get(_PRICING_URL, headers=_PRICING_HEADERS, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPStatusError as exc:  # noqa: BLE001
        status_code = getattr(exc.response, "status_code", 0) if exc.response is not None else 0
        if _snapshot_is_transient_status(int(status_code or 0)):
            _SNAPSHOT_FETCH_FAILURES += 1
            delay = _snapshot_retry_delay_seconds(_SNAPSHOT_FETCH_FAILURES)
            _SNAPSHOT_FETCH_BACKOFF_UNTIL_MONO = now_mono + delay
            logger.warning(
                "%s snapshot fetch failed (http %s). retry in %.1fs (failures=%d): %s",
                config.LOG_PREFIX,
                status_code,
                delay,
                _SNAPSHOT_FETCH_FAILURES,
                exc,
            )
            return False
        _SNAPSHOT_FETCH_FAILURES += 1
        delay = _snapshot_retry_delay_seconds(_SNAPSHOT_FETCH_FAILURES)
        _SNAPSHOT_FETCH_BACKOFF_UNTIL_MONO = now_mono + delay
        logger.warning(
            "%s snapshot fetch failed (http %s). retry in %.1fs (failures=%d): %s",
            config.LOG_PREFIX,
            status_code,
            delay,
            _SNAPSHOT_FETCH_FAILURES,
            exc,
        )
        return False
    except (httpx.RequestError, ValueError, TypeError, OSError) as exc:  # noqa: BLE001
        _SNAPSHOT_FETCH_FAILURES += 1
        delay = _snapshot_retry_delay_seconds(_SNAPSHOT_FETCH_FAILURES)
        _SNAPSHOT_FETCH_BACKOFF_UNTIL_MONO = now_mono + delay
        logger.warning(
            "%s snapshot fetch request failed. retry in %.1fs (failures=%d): %s",
            config.LOG_PREFIX,
            delay,
            _SNAPSHOT_FETCH_FAILURES,
            exc,
        )
        return False
    except Exception as exc:
        logger.warning("%s snapshot fetch failed: %s", config.LOG_PREFIX, exc)
        _SNAPSHOT_FETCH_FAILURES += 1
        delay = _snapshot_retry_delay_seconds(_SNAPSHOT_FETCH_FAILURES)
        _SNAPSHOT_FETCH_BACKOFF_UNTIL_MONO = now_mono + delay
        return False

    prices = payload.get("prices") or []
    if not prices:
        _SNAPSHOT_FETCH_FAILURES += 1
        delay = _snapshot_retry_delay_seconds(_SNAPSHOT_FETCH_FAILURES)
        _SNAPSHOT_FETCH_BACKOFF_UNTIL_MONO = now_mono + delay
        logger.warning(
            "%s snapshot parse failed: no prices in payload. retry in %.1fs (failures=%d)",
            config.LOG_PREFIX,
            delay,
            _SNAPSHOT_FETCH_FAILURES,
        )
        return False

    price = prices[0]
    bids = price.get("bids") or []
    asks = price.get("asks") or []
    if not bids or not asks:
        _SNAPSHOT_FETCH_FAILURES += 1
        delay = _snapshot_retry_delay_seconds(_SNAPSHOT_FETCH_FAILURES)
        _SNAPSHOT_FETCH_BACKOFF_UNTIL_MONO = now_mono + delay
        logger.warning(
            "%s snapshot parse failed: missing bids/asks. retry in %.1fs (failures=%d)",
            config.LOG_PREFIX,
            delay,
            _SNAPSHOT_FETCH_FAILURES,
        )
        return False

    try:
        bid = float(bids[0]["price"])
        ask = float(asks[0]["price"])
        quote_ts = _parse_time(price.get("time", datetime.datetime.utcnow().isoformat() + "Z"))
    except Exception as exc:
        logger.warning("%s snapshot parse failed: %s", config.LOG_PREFIX, exc)
        _SNAPSHOT_FETCH_FAILURES += 1
        delay = _snapshot_retry_delay_seconds(_SNAPSHOT_FETCH_FAILURES)
        _SNAPSHOT_FETCH_BACKOFF_UNTIL_MONO = now_mono + delay
        logger.warning(
            "%s snapshot parse failed. retry in %.1fs (failures=%d)",
            config.LOG_PREFIX,
            delay,
            _SNAPSHOT_FETCH_FAILURES,
        )
        return False

    fetched_at = datetime.datetime.now(datetime.timezone.utc)
    if quote_ts.tzinfo is None:
        quote_ts = quote_ts.replace(tzinfo=datetime.timezone.utc)
    quote_ts_utc = quote_ts.astimezone(datetime.timezone.utc)
    quote_age_ms = max(0.0, (fetched_at - quote_ts_utc).total_seconds() * 1000.0)
    ts = quote_ts_utc
    if quote_age_ms > max(0.0, float(config.MAX_TICK_AGE_MS)):
        # Pricing API can return an unchanged quote timestamp during quiet periods.
        # Treat fallback snapshots as fresh at fetch time to avoid stale-loop lockups.
        ts = fetched_at

    tick = SimpleNamespace(bid=bid, ask=ask, time=ts)
    try:
        spread_monitor.update_from_tick(tick)
        tick_window.record(tick)
    except Exception as exc:
        logger.warning("%s snapshot cache update failed: %s", config.LOG_PREFIX, exc)
        _SNAPSHOT_FETCH_FAILURES += 1
        delay = _snapshot_retry_delay_seconds(_SNAPSHOT_FETCH_FAILURES)
        _SNAPSHOT_FETCH_BACKOFF_UNTIL_MONO = now_mono + delay
        logger.warning(
            "%s snapshot cache update failed. retry in %.1fs (failures=%d)",
            config.LOG_PREFIX,
            delay,
            _SNAPSHOT_FETCH_FAILURES,
        )
        return False

    _SNAPSHOT_FETCH_FAILURES = 0
    _SNAPSHOT_FETCH_BACKOFF_UNTIL_MONO = 0.0
    _SNAPSHOT_FETCH_BACKOFF_LOG_MONO = 0.0
    return True


async def scalp_ping_5s_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled (idle)", config.LOG_PREFIX)
        while True:
            await asyncio.sleep(30.0)

    env_name = "practice" if _OANDA_PRACTICE else "live"
    LOG.info(
        "%s env=%s require_practice=%s account=%s",
        config.LOG_PREFIX,
        env_name,
        int(config.REQUIRE_PRACTICE),
        _mask_value(_OANDA_ACCOUNT),
    )
    if config.REQUIRE_PRACTICE and not _OANDA_PRACTICE:
        LOG.error(
            "%s blocked: practice account required (set SCALP_PING_5S_REQUIRE_PRACTICE=0 to bypass)",
            config.LOG_PREFIX,
        )
        while True:
            await asyncio.sleep(30.0)

    LOG.info(
        "%s start loop=%.2fs pocket=%s tag=%s",
        config.LOG_PREFIX,
        config.LOOP_INTERVAL_SEC,
        config.POCKET,
        config.STRATEGY_TAG,
    )
    LOG.info("Application started!")

    pos_manager = PositionManager()
    rate_limiter = SlidingWindowRateLimiter(
        config.MAX_ORDERS_PER_MINUTE,
        config.MIN_ORDER_SPACING_SEC,
    )
    last_entry_mono = 0.0
    last_snapshot_fetch = 0.0
    last_stale_log_mono = 0.0
    last_snapshot_degrade_log_mono = 0.0
    last_trap_log_mono = 0.0
    last_density_topup_log_mono = 0.0
    last_keepalive_log_mono = 0.0
    last_max_active_bypass_log_mono = 0.0
    last_bias_log_mono = 0.0
    last_lookahead_log_mono = 0.0
    last_regime_log_mono = 0.0
    last_horizon_log_mono = 0.0
    last_profit_bank_log_mono = 0.0
    last_extrema_log_mono = 0.0
    last_sl_streak_log_mono = 0.0
    last_side_metrics_flip_log_mono = 0.0
    last_entry_skip_summary_mono = 0.0
    last_loop_heartbeat_mono = 0.0
    entry_skip_reasons: dict[str, int] = {}
    entry_skip_reasons_by_side: dict[str, dict[str, int]] = {}
    _signal_side_hint: Optional[str] = None

    def _normalize_no_signal_reason(
        reason: str,
        detail: Optional[str] = None,
    ) -> Optional[str]:
        if reason != "no_signal":
            return None
        if not isinstance(detail, str):
            return None
        token = detail.strip()
        if not token:
            return None
        base = token.split(":", 1)[0]
        if base.startswith("insufficient_signal_rows"):
            if "fallback_used=yes" in token:
                return "insufficient_signal_rows_fallback"
            if "fallback_used=attempted" in token:
                return "insufficient_signal_rows_fallback_exhausted"
            if "fallback_used=no_data" in token:
                return "insufficient_signal_rows_no_data"
            return "insufficient_signal_rows"
        if base.startswith("insufficient_mid_rows"):
            return "insufficient_mid_rows"
        if base.startswith("insufficient_rows"):
            return "insufficient_rows"
        if base.startswith("invalid_latest_epoch"):
            return "invalid_latest_epoch"
        if base.startswith("stale_tick"):
            return "stale_tick"
        if base.startswith("revert_not_enabled"):
            return "revert_disabled"
        if base.startswith("revert_disabled"):
            return "revert_disabled"
        if base.startswith("revert_not_found"):
            return "revert_not_found"
        if base.startswith("momentum_tail_failed_no_revert"):
            return "momentum_tail_failed_no_revert"
        if base.startswith("momentum_tail_failed"):
            return "momentum_tail_failed"
        if base.startswith("drop_flow_not_enough_rows"):
            return "drop_flow_not_enough_rows"
        if base.startswith("drop_flow_invalid_rows"):
            return "drop_flow_invalid_rows"
        if base.startswith("drop_flow_strength_not_met"):
            return "drop_flow_strength_not_met"
        if base.startswith("drop_flow_tail_bounce"):
            return "drop_flow_tail_bounce"
        if base.startswith("drop_flow_direction_mismatch"):
            return "drop_flow_direction_mismatch"
        if base.startswith("drop_flow_short_only"):
            return "drop_flow_short_only"
        return base

    def _infer_signal_side_from_reason(
        reason: str,
        detail: Optional[str] = None,
    ) -> Optional[str]:
        source = f"{reason} {detail or ''}".lower()
        if "revert_long" in source or "side=long" in source:
            return "long"
        if "revert_short" in source or "side=short" in source:
            return "short"
        if "side_filter_block" in source and "long" in source:
            return "long"
        if "side_filter_block" in source and "short" in source:
            return "short"
        if "drop_flow_strength_not_met" in source and "side=short" in source:
            return "short"
        if "drop_flow_direction_mismatch" in source and "side=short" in source:
            return "short"
        if "drop_flow_direction_mismatch" in source and "side=long" in source:
            return "long"
        if "drop_flow_strength_not_met" in source and "side=long" in source:
            return "long"
        if "drop_flow_tail_bounce" in source and "side=long" in source:
            return "long"
        if "drop_flow_tail_bounce" in source and "side=short" in source:
            return "short"
        if "drop_flow_short_only" in source and "side=short" in source:
            return "short"
        return None

    def _note_entry_skip(
        reason: str,
        detail: Optional[str] = None,
        side: Optional[str] = None,
    ) -> None:
        nonlocal last_entry_skip_summary_mono
        nonlocal entry_skip_reasons
        nonlocal entry_skip_reasons_by_side
        nonlocal _signal_side_hint

        entry_skip_reasons[reason] = entry_skip_reasons.get(reason, 0) + 1
        detailed_reason = reason
        no_signal_detail = _normalize_no_signal_reason(reason, detail)
        if no_signal_detail:
            detailed_reason = f"{reason}:{no_signal_detail}"
            entry_skip_reasons[detailed_reason] = (
                entry_skip_reasons.get(detailed_reason, 0) + 1
            )
        side_key = str(side).strip().lower() if side is not None else _signal_side_hint
        if side_key is None and reason == "no_signal":
            side_key = _infer_signal_side_from_reason(reason, detail=detail)
        if side_key:
            side_bucket = entry_skip_reasons_by_side.setdefault(side_key, {})
            side_bucket[detailed_reason] = side_bucket.get(detailed_reason, 0) + 1

        if side_key:
            if detail:
                detail = f"side={side_key} {detail}"
            else:
                detail = f"side={side_key}"

        if detail:
            LOG.debug("%s entry-skip %s %s", config.LOG_PREFIX, reason, detail)
        now = time.monotonic()
        if now - last_entry_skip_summary_mono >= _ENTRY_SKIP_SUMMARY_INTERVAL_SEC:
            if entry_skip_reasons:
                total = sum(entry_skip_reasons.values())
                summary = ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(
                        entry_skip_reasons.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
                LOG.info(
                    "%s entry-skip summary total=%d %s",
                    config.LOG_PREFIX,
                    total,
                    summary,
                )
            last_entry_skip_summary_mono = now
            entry_skip_reasons = {}
            for side, side_reasons in sorted(entry_skip_reasons_by_side.items()):
                side_total = sum(side_reasons.values())
                if side_total <= 0:
                    continue
                side_summary = ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(
                        side_reasons.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
                LOG.info(
                    "%s entry-skip summary side=%s total=%d %s",
                    config.LOG_PREFIX,
                    side,
                    side_total,
                    side_summary,
                )
            entry_skip_reasons_by_side = {}

    protected_trade_ids: set[str] = set()
    protected_seeded = False
    strategy_tag_key = str(config.STRATEGY_TAG or "").strip().lower()

    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            _signal_side_hint = None
            loop_start_mono = time.monotonic()
            if loop_start_mono - last_loop_heartbeat_mono >= _LOOP_HEARTBEAT_INTERVAL_SEC:
                last_loop_heartbeat_mono = loop_start_mono
                LOG.info(
                    "%s loop_heartbeat interval=%ds pocket=%s tag=%s",
                    config.LOG_PREFIX,
                    int(_LOOP_HEARTBEAT_INTERVAL_SEC),
                    config.POCKET,
                    config.STRATEGY_TAG,
                )

            try:
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                if not is_market_open(now_utc):
                    reopen_in_sec = seconds_until_open(now_utc)
                    reopen_in_sec = max(0.0, float(reopen_in_sec))
                    _note_entry_skip(
                        "market_closed",
                        detail=f"reopen_in={int(reopen_in_sec)}s",
                    )
                    continue
                if not can_trade(config.POCKET):
                    _note_entry_skip("pocket_disabled")
                    continue

                positions, pos_err = await _safe_get_open_positions(
                    pos_manager,
                    logger=LOG,
                    timeout_sec=_POSITION_MANAGER_OPEN_POSITIONS_TIMEOUT_SEC,
                )
                if pos_err is not None:
                    _note_entry_skip(pos_err)
                    continue

                pocket_info = positions.get(config.POCKET) or {}

                if not protected_seeded:
                    protected_seeded = True
                    if config.FORCE_EXIT_SKIP_EXISTING_ON_START:
                        open_trades = pocket_info.get("open_trades") if isinstance(pocket_info, dict) else None
                        protect_total = 0
                        skipped_eligible = 0
                        if isinstance(open_trades, list):
                            max_hold_sec = float(config.FORCE_EXIT_MAX_HOLD_SEC)
                            hard_loss_pips = float(config.FORCE_EXIT_MAX_FLOATING_LOSS_PIPS)
                            recovery_window_sec = float(config.FORCE_EXIT_RECOVERY_WINDOW_SEC)
                            recoverable_loss_pips = float(config.FORCE_EXIT_RECOVERABLE_LOSS_PIPS)

                            for trade in open_trades:
                                if not isinstance(trade, dict):
                                    continue
                                if strategy_tag_key:
                                    trade_tag = _trade_strategy_tag(trade).lower()
                                    if trade_tag != strategy_tag_key:
                                        continue
                                trade_id = str(trade.get("trade_id") or "").strip()
                                if not trade_id:
                                    continue

                                # Don't protect trades that are already eligible for a force-exit;
                                # otherwise they can linger forever across restarts and block margin.
                                opened_at = _parse_trade_open_time(
                                    trade.get("open_time") or trade.get("entry_time")
                                )
                                hold_sec = (now_utc - opened_at).total_seconds() if opened_at is not None else 0.0
                                unrealized_pips = _trade_unrealized_pips(trade)
                                units = int(_safe_float(trade.get("units"), 0.0))
                                trade_side = _trade_side_from_units(units)
                                if trade_side is None:
                                    continue
                                max_hold_sec, hard_loss_pips = _force_exit_thresholds_for_side(
                                    trade_side
                                )
                                trade_max_hold_sec = _trade_force_exit_max_hold_sec(trade, max_hold_sec)
                                trade_hard_loss_pips = _trade_force_exit_hard_loss_pips(trade, hard_loss_pips)
                                hard_loss_trigger_pips = _force_exit_hard_loss_trigger_pips(
                                    trade,
                                    trade_hard_loss_pips,
                                )
                                eligible = False
                                if trade_max_hold_sec > 0.0 and hold_sec >= trade_max_hold_sec:
                                    eligible = True
                                elif (
                                    hold_sec
                                    >= float(
                                        getattr(config, "FORCE_EXIT_FLOATING_LOSS_MIN_HOLD_SEC", 0.0)
                                    )
                                    and hard_loss_trigger_pips > 0.0
                                    and unrealized_pips <= -hard_loss_trigger_pips
                                ):
                                    eligible = True
                                elif (
                                    recovery_window_sec > 0.0
                                    and recoverable_loss_pips > 0.0
                                    and hold_sec >= recovery_window_sec
                                    and unrealized_pips <= -recoverable_loss_pips
                                ):
                                    eligible = True
                                if eligible:
                                    skipped_eligible += 1
                                    continue

                                protected_trade_ids.add(trade_id)
                                protect_total += 1
                        if protect_total > 0:
                            LOG.info(
                                "%s force_exit protect_existing=%d trade(s)",
                                config.LOG_PREFIX,
                                protect_total,
                            )
                        if skipped_eligible > 0:
                            LOG.info(
                                "%s force_exit protect_existing skipped=%d eligible_trade(s)",
                                config.LOG_PREFIX,
                                skipped_eligible,
                            )
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("%s loop exception; recovering", config.LOG_PREFIX)
                _note_entry_skip("loop_exception")
                await asyncio.sleep(_LOOP_EXCEPTION_RECOVERY_SLEEP_SEC)
                continue

            # Expire protection once a trade becomes force-exit eligible.
            if protected_trade_ids:
                open_trades = pocket_info.get("open_trades") if isinstance(pocket_info, dict) else None
                if isinstance(open_trades, list):
                    max_hold_sec = float(config.FORCE_EXIT_MAX_HOLD_SEC)
                    hard_loss_pips = float(config.FORCE_EXIT_MAX_FLOATING_LOSS_PIPS)
                    recovery_window_sec = float(config.FORCE_EXIT_RECOVERY_WINDOW_SEC)
                    recoverable_loss_pips = float(config.FORCE_EXIT_RECOVERABLE_LOSS_PIPS)
                    protected_seen: set[str] = set()
                    for trade in open_trades:
                        if not isinstance(trade, dict):
                            continue
                        if strategy_tag_key:
                            trade_tag = _trade_strategy_tag(trade).lower()
                            if trade_tag != strategy_tag_key:
                                continue
                        trade_id = str(trade.get("trade_id") or "").strip()
                        if not trade_id or trade_id not in protected_trade_ids:
                            continue
                        protected_seen.add(trade_id)
                        opened_at = _parse_trade_open_time(
                            trade.get("open_time") or trade.get("entry_time")
                        )
                        hold_sec = (now_utc - opened_at).total_seconds() if opened_at is not None else 0.0
                        unrealized_pips = _trade_unrealized_pips(trade)
                        units = int(_safe_float(trade.get("units"), 0.0))
                        trade_side = _trade_side_from_units(units)
                        if trade_side is None:
                            continue
                        max_hold_sec, hard_loss_pips = _force_exit_thresholds_for_side(
                            trade_side
                        )
                        trade_max_hold_sec = _trade_force_exit_max_hold_sec(
                            trade, max_hold_sec
                        )
                        trade_hard_loss_pips = _trade_force_exit_hard_loss_pips(
                            trade, hard_loss_pips
                        )
                        hard_loss_trigger_pips = _force_exit_hard_loss_trigger_pips(
                            trade,
                            trade_hard_loss_pips,
                        )
                        eligible = False
                        if trade_max_hold_sec > 0.0 and hold_sec >= trade_max_hold_sec:
                            eligible = True
                        elif (
                            hold_sec
                            >= float(
                                getattr(config, "FORCE_EXIT_FLOATING_LOSS_MIN_HOLD_SEC", 0.0)
                            )
                            and hard_loss_trigger_pips > 0.0
                            and unrealized_pips <= -hard_loss_trigger_pips
                        ):
                            eligible = True
                        elif (
                            recovery_window_sec > 0.0
                            and recoverable_loss_pips > 0.0
                            and hold_sec >= recovery_window_sec
                            and unrealized_pips <= -recoverable_loss_pips
                        ):
                            eligible = True
                        if eligible:
                            protected_trade_ids.discard(trade_id)

                    # Drop stale IDs that are no longer open.
                    protected_trade_ids.intersection_update(protected_seen)
            forced_closed = await _enforce_new_entry_time_stop(
                pocket_info=pocket_info,
                now_utc=now_utc,
                logger=LOG,
                protected_trade_ids=protected_trade_ids,
            )
            if forced_closed > 0:
                positions, pos_err = await _safe_get_open_positions(
                    pos_manager,
                    logger=LOG,
                    timeout_sec=_POSITION_MANAGER_OPEN_POSITIONS_TIMEOUT_SEC,
                )
                if pos_err is not None:
                    _note_entry_skip(pos_err)
                    continue
                pocket_info = positions.get(config.POCKET) or {}
            profit_bank_closed = await _apply_profit_bank_release(
                pocket_info=pocket_info,
                now_utc=now_utc,
                logger=LOG,
                protected_trade_ids=protected_trade_ids,
            )
            if profit_bank_closed > 0:
                positions, pos_err = await _safe_get_open_positions(
                    pos_manager,
                    logger=LOG,
                    timeout_sec=_POSITION_MANAGER_OPEN_POSITIONS_TIMEOUT_SEC,
                )
                if pos_err is not None:
                    _note_entry_skip(pos_err)
                    continue
                pocket_info = positions.get(config.POCKET) or {}
            elif (
                config.PROFIT_BANK_ENABLED
                and time.monotonic() - last_profit_bank_log_mono >= config.PROFIT_BANK_LOG_INTERVAL_SEC
            ):
                start_utc = _profit_bank_start_time_utc()
                gross_profit, spent_loss, net_realized = _load_profit_bank_stats(
                    strategy_tag=config.STRATEGY_TAG,
                    pocket=config.POCKET,
                    reason=config.PROFIT_BANK_REASON,
                    start_time_utc=start_utc,
                )
                budget = _profit_bank_available_budget_jpy(
                    gross_profit_jpy=gross_profit,
                    spent_loss_jpy=spent_loss,
                    net_realized_jpy=net_realized,
                )
                LOG.info(
                    "%s profit_bank status budget=%.1f gross=%.1f spent=%.1f net=%.1f start=%s",
                    config.LOG_PREFIX,
                    budget,
                    gross_profit,
                    spent_loss,
                    net_realized,
                    start_utc.isoformat() if start_utc is not None else "-",
                )
                last_profit_bank_log_mono = time.monotonic()

            blocked, remain, spread_state, spread_reason = spread_monitor.is_blocked()
            spread_pips = _safe_float((spread_state or {}).get("spread_pips"), 0.0)
            if blocked or spread_pips > config.MAX_SPREAD_PIPS:
                if blocked and remain > 0 and remain % 5 == 0:
                    LOG.info(
                        "%s spread block remain=%ss spread=%.2fp reason=%s",
                        config.LOG_PREFIX,
                        remain,
                        spread_pips,
                        spread_reason or "guard",
                    )
                _note_entry_skip(
                    "spread_blocked",
                    f"spread={spread_pips:.2f} max={config.MAX_SPREAD_PIPS:.2f} remain={remain}",
                )
                continue

            blocked_jst_hour = _entry_blocked_hour_jst(now_utc)
            if blocked_jst_hour is not None:
                _note_entry_skip(
                    "blocked_hour_jst",
                    f"hour={blocked_jst_hour}",
                )
                continue

            ticks = tick_window.recent_ticks(
                seconds=config.WINDOW_SEC,
                limit=int(config.WINDOW_SEC * 25) + 50,
            )
            latest_tick_age_ms = _latest_tick_age_ms(ticks)

            now_mono = time.monotonic()
            last_snapshot_fetch, keepalive = await _maybe_keepalive_snapshot(
                now_mono=now_mono,
                last_snapshot_fetch=last_snapshot_fetch,
                rows=ticks,
                latest_tick_age_ms=latest_tick_age_ms,
                logger=LOG,
            )
            if keepalive:
                ticks = tick_window.recent_ticks(
                    seconds=config.WINDOW_SEC,
                    limit=int(config.WINDOW_SEC * 25) + 50,
                )
                latest_tick_age_ms = _latest_tick_age_ms(ticks)
                if now_mono - last_keepalive_log_mono >= 5.0:
                    LOG.info(
                        "%s snapshot_keepalive reason=%s age=%.0fms density=%.3f span=%.3f",
                        config.LOG_PREFIX,
                        keepalive["reason"],
                        keepalive["age_ms"],
                        keepalive["density"],
                        keepalive["span_ratio"],
                    )
                    last_keepalive_log_mono = now_mono

            snapshot_degraded_mode = _snapshot_is_degraded_mode()
            snapshot_age_limit_ms = _snapshot_stale_age_limit_ms()
            needs_snapshot = (len(ticks) < config.MIN_TICKS) or (
                latest_tick_age_ms > snapshot_age_limit_ms
            )

            if spread_pips <= 0.0:
                spread_pips = _latest_spread_from_ticks(ticks)

            if needs_snapshot and config.SNAPSHOT_FALLBACK_ENABLED:
                now_mono = time.monotonic()
                if now_mono - last_snapshot_fetch >= config.SNAPSHOT_MIN_INTERVAL_SEC:
                    if await _fetch_price_snapshot(LOG):
                        last_snapshot_fetch = now_mono
                        ticks = tick_window.recent_ticks(
                            seconds=config.WINDOW_SEC,
                            limit=int(config.WINDOW_SEC * 25) + 50,
                        )
                        latest_tick_age_ms = _latest_tick_age_ms(ticks)
                        if spread_pips <= 0.0:
                            spread_pips = _latest_spread_from_ticks(ticks)

            if len(ticks) < config.MIN_TICKS:
                _note_entry_skip(
                    "low_tick_count",
                    f"len={len(ticks)} min={config.MIN_TICKS}",
                )
                if snapshot_degraded_mode and now_mono - last_snapshot_degrade_log_mono >= 10.0:
                    last_snapshot_degrade_log_mono = now_mono
                    LOG.warning(
                        "%s snapshot degraded: insufficient ticks len=%d < min_ticks=%d (failures=%d)",
                        config.LOG_PREFIX,
                        len(ticks),
                        config.MIN_TICKS,
                        _SNAPSHOT_FETCH_FAILURES,
                    )
                continue
            if latest_tick_age_ms > snapshot_age_limit_ms:
                now_mono = time.monotonic()
                if now_mono - last_stale_log_mono >= 10.0:
                    LOG.warning(
                        "%s stale ticks age=%.0fms (> %.0fms) waiting for fresh data (failures=%d degraded=%s)",
                        config.LOG_PREFIX,
                        latest_tick_age_ms,
                        snapshot_age_limit_ms,
                        _SNAPSHOT_FETCH_FAILURES,
                        snapshot_degraded_mode,
                    )
                    last_stale_log_mono = now_mono
                _note_entry_skip(
                    "stale_ticks",
                    f"age_ms={latest_tick_age_ms:.0f} limit={snapshot_age_limit_ms:.0f}",
                )
                continue

            signal_window_meta: dict[str, object] = {}
            signal, signal_reason = _build_tick_signal(ticks, spread_pips)
            if signal is None:
                _note_entry_skip(
                    "no_signal",
                    detail=signal_reason,
                    side=_infer_signal_side_from_reason("no_signal", detail=signal_reason),
                )
                continue
            _signal_side_hint = signal.side
            if signal.spread_pips > config.MAX_SPREAD_PIPS:
                _note_entry_skip(
                    "signal_spread_too_wide",
                    f"spread={signal.spread_pips:.2f} max={config.MAX_SPREAD_PIPS:.2f}",
                )
                continue
            signal, signal_window_meta = _maybe_adapt_signal_window(
                ticks=ticks,
                spread_pips=signal.spread_pips,
                base_signal=signal,
            )

            now_mono = time.monotonic()
            try:
                factors = all_factors()
            except Exception:
                factors = {}
            tech_route_reasons: list[str] = []
            regime = _build_mtf_regime(factors=factors if isinstance(factors, dict) else None)
            regime_input_signal = signal
            regime_signal, regime_units_mult, regime_gate = _apply_mtf_regime(signal, regime)
            if regime_signal is None or regime_units_mult <= 0.0:
                if now_mono - last_regime_log_mono >= config.MTF_REGIME_LOG_INTERVAL_SEC:
                    if regime is not None:
                        LOG.info(
                            "%s mtf %s gate=%s mode=%s side=%s trend=%.2f heat=%.2f adx(m1/m5)=%.1f/%.1f atr(m1/m5)=%.2f/%.2f",
                            config.LOG_PREFIX,
                            (
                                "route"
                                if config.TECH_ROUTER_ENABLED
                                else "block"
                            ),
                            regime_gate,
                            regime.mode,
                            regime.side,
                            regime.trend_score,
                            regime.heat_score,
                            regime.adx_m1,
                            regime.adx_m5,
                            regime.atr_m1,
                            regime.atr_m5,
                        )
                    else:
                        LOG.info(
                            "%s mtf %s gate=%s regime=none",
                            config.LOG_PREFIX,
                            ("route" if config.TECH_ROUTER_ENABLED else "block"),
                            regime_gate,
                        )
                    last_regime_log_mono = now_mono
                if config.TECH_ROUTER_ENABLED:
                    signal = regime_input_signal
                    regime_units_mult = max(0.1, float(config.TECH_ROUTER_MTF_BLOCK_UNITS_MULT))
                    regime_gate = f"{regime_gate}_route"
                    tech_route_reasons.append("mtf")
                else:
                    _note_entry_skip("mtf_block")
                    continue
            else:
                signal = regime_signal
            horizon = _build_horizon_bias(signal, factors if isinstance(factors, dict) else {})
            horizon_input_signal = signal
            horizon_signal, horizon_units_mult, horizon_gate = _apply_horizon_bias(signal, horizon)
            if horizon_signal is None or horizon_units_mult <= 0.0:
                if now_mono - last_horizon_log_mono >= config.HORIZON_LOG_INTERVAL_SEC:
                    if horizon is not None:
                        LOG.info(
                            "%s horizon %s gate=%s composite=%s(%.2f) agree=%d L/M/S/U=%s(%.2f)/%s(%.2f)/%s(%.2f)/%s(%.2f)",
                            config.LOG_PREFIX,
                            ("route" if config.TECH_ROUTER_ENABLED else "block"),
                            horizon_gate,
                            horizon.composite_side,
                            horizon.composite_score,
                            horizon.agreement,
                            horizon.long_side,
                            horizon.long_score,
                            horizon.mid_side,
                            horizon.mid_score,
                            horizon.short_side,
                            horizon.short_score,
                            horizon.micro_side,
                            horizon.micro_score,
                        )
                    else:
                        LOG.info(
                            "%s horizon %s gate=%s horizon=none",
                            config.LOG_PREFIX,
                            ("route" if config.TECH_ROUTER_ENABLED else "block"),
                            horizon_gate,
                        )
                    last_horizon_log_mono = now_mono
                if config.TECH_ROUTER_ENABLED:
                    signal = horizon_input_signal
                    horizon_units_mult = max(0.1, float(config.TECH_ROUTER_HORIZON_BLOCK_UNITS_MULT))
                    horizon_gate = f"{horizon_gate}_route"
                    tech_route_reasons.append("horizon")
                else:
                    _note_entry_skip("horizon_block")
                    continue
            else:
                signal = horizon_signal

            m1_score = _tf_score_or_none(
                factors if isinstance(factors, dict) else {},
                "M1",
            )
            m1_trend_units_mult, m1_trend_gate = _m1_trend_units_multiplier(
                signal.side,
                m1_score,
                regime=regime,
            )

            last_snapshot_fetch, density_topup = await _maybe_topup_micro_density(
                now_mono=now_mono,
                last_snapshot_fetch=last_snapshot_fetch,
                logger=LOG,
            )
            if (
                density_topup
                and now_mono - last_density_topup_log_mono >= 5.0
            ):
                LOG.info(
                    "%s snapshot_topup density=%.3f->%.3f target=%.3f window=%.0fs",
                    config.LOG_PREFIX,
                    density_topup["before"],
                    density_topup["after"],
                    density_topup["target"],
                    density_topup["window_sec"],
                )
                last_density_topup_log_mono = now_mono

            direction_bias = _build_direction_bias(ticks, spread_pips=signal.spread_pips)
            fast_flip_applied = False
            fast_flip_reason = ""
            sl_streak_flip_applied = False
            sl_streak_flip_reason = ""
            sl_streak_side = ""
            sl_streak_count = 0
            sl_streak_age_sec = -1.0
            sl_streak_side_sl_hits_recent = 0
            sl_streak_target_market_plus_recent = 0
            sl_streak_direction_confirmed = False
            sl_streak_horizon_confirmed = False
            side_metrics_flip_applied = False
            side_metrics_flip_reason = ""
            side_metrics_flip_current_sl_rate = 0.0
            side_metrics_flip_target_sl_rate = 0.0
            side_metrics_flip_current_market_plus_rate = 0.0
            side_metrics_flip_target_market_plus_rate = 0.0
            side_metrics_flip_current_trades = 0
            side_metrics_flip_target_trades = 0
            bias_units_mult, bias_gate = _direction_units_multiplier(
                signal.side,
                direction_bias,
                signal_mode=signal.mode,
            )
            if bias_units_mult <= 0.0:
                if now_mono - last_bias_log_mono >= config.DIRECTION_BIAS_LOG_INTERVAL_SEC:
                    score = _safe_float(getattr(direction_bias, "score", 0.0), 0.0)
                    LOG.info(
                        "%s bias %s signal=%s bias=%s score=%.2f mom=%.2fp rng=%.2fp flow=%.2f",
                        config.LOG_PREFIX,
                        ("route" if config.TECH_ROUTER_ENABLED else "block"),
                        signal.side,
                        getattr(direction_bias, "side", "none"),
                        score,
                        _safe_float(getattr(direction_bias, "momentum_pips", 0.0), 0.0),
                        _safe_float(getattr(direction_bias, "range_pips", 0.0), 0.0),
                        _safe_float(getattr(direction_bias, "flow", 0.0), 0.0),
                    )
                    last_bias_log_mono = now_mono
                if config.TECH_ROUTER_ENABLED:
                    bias_units_mult = max(0.1, float(config.TECH_ROUTER_DIRECTION_BLOCK_UNITS_MULT))
                    bias_gate = f"{bias_gate}_route"
                    tech_route_reasons.append("direction")
                else:
                    _note_entry_skip(
                        "direction_bias_block",
                        f"side={signal.side}",
                    )
                    continue

            lookahead_decision = None
            lookahead_units_mult = 1.0
            decision_speed_scale = _instant_speed_scale(signal.instant_range_pips)
            lookahead_horizon_sec = max(
                config.LOOKAHEAD_HORIZON_SEC * config.INSTANT_LOOKAHEAD_SCALE_MIN,
                config.LOOKAHEAD_HORIZON_SEC * decision_speed_scale,
            )
            if config.LOOKAHEAD_GATE_ENABLED:
                direction_score = None
                if direction_bias is not None:
                    direction_score = _safe_float(getattr(direction_bias, "score", 0.0), 0.0)
                lookahead_decision = decide_tick_lookahead_edge(
                    ticks=ticks,
                    side=signal.side,
                    spread_pips=signal.spread_pips,
                    momentum_pips=signal.momentum_pips,
                    trigger_pips=signal.trigger_pips,
                    imbalance=signal.imbalance,
                    tick_rate=signal.tick_rate,
                    signal_span_sec=signal.span_sec,
                    pip_value=config.PIP_VALUE,
                    horizon_sec=lookahead_horizon_sec,
                    edge_min_pips=config.LOOKAHEAD_EDGE_MIN_PIPS,
                    edge_ref_pips=config.LOOKAHEAD_EDGE_REF_PIPS,
                    units_min_mult=config.LOOKAHEAD_UNITS_MIN_MULT,
                    units_max_mult=config.LOOKAHEAD_UNITS_MAX_MULT,
                    slippage_base_pips=config.LOOKAHEAD_SLIP_BASE_PIPS,
                    slippage_spread_mult=config.LOOKAHEAD_SLIP_SPREAD_MULT,
                    slippage_range_mult=config.LOOKAHEAD_SLIP_RANGE_MULT,
                    latency_penalty_pips=config.LOOKAHEAD_LATENCY_PENALTY_PIPS,
                    safety_margin_pips=config.LOOKAHEAD_SAFETY_MARGIN_PIPS,
                    momentum_decay=config.LOOKAHEAD_MOMENTUM_DECAY,
                    momentum_weight=config.LOOKAHEAD_MOMENTUM_WEIGHT,
                    flow_weight=config.LOOKAHEAD_FLOW_WEIGHT,
                    rate_weight=config.LOOKAHEAD_RATE_WEIGHT,
                    bias_weight=config.LOOKAHEAD_BIAS_WEIGHT,
                    trigger_weight=config.LOOKAHEAD_TRIGGER_WEIGHT,
                    counter_penalty=config.LOOKAHEAD_COUNTER_PENALTY,
                    direction_bias_score=direction_score,
                    allow_thin_edge=config.LOOKAHEAD_ALLOW_THIN_EDGE,
                    fail_open=True,
                )
                if not lookahead_decision.allow_entry:
                    if now_mono - last_lookahead_log_mono >= config.LOOKAHEAD_LOG_INTERVAL_SEC:
                        LOG.info(
                            "%s lookahead %s side=%s reason=%s pred=%.3fp cost=%.3fp edge=%.3fp mom=%.3fp range=%.3fp",
                            config.LOG_PREFIX,
                            ("route" if config.TECH_ROUTER_ENABLED else "block"),
                            signal.side,
                            lookahead_decision.reason,
                            lookahead_decision.pred_move_pips,
                            lookahead_decision.cost_pips,
                            lookahead_decision.edge_pips,
                            lookahead_decision.momentum_aligned_pips,
                            lookahead_decision.range_pips,
                        )
                        last_lookahead_log_mono = now_mono
                    if config.TECH_ROUTER_ENABLED:
                        lookahead_units_mult = max(
                            0.1, float(config.TECH_ROUTER_LOOKAHEAD_BLOCK_UNITS_MULT)
                        )
                        tech_route_reasons.append("lookahead")
                    else:
                        _note_entry_skip(
                            "lookahead_block",
                            f"reason={lookahead_decision.reason}",
                        )
                        continue
                else:
                    lookahead_units_mult = max(0.1, float(lookahead_decision.units_mult))
            extrema_decision = _extrema_gate_decision(
                signal.side,
                factors=factors if isinstance(factors, dict) else None,
                regime=regime,
            )
            extrema_units_mult = max(0.1, float(extrema_decision.units_mult))
            m1_pos_log = (
                -1.0 if extrema_decision.m1_pos is None else float(extrema_decision.m1_pos)
            )
            m5_pos_log = (
                -1.0 if extrema_decision.m5_pos is None else float(extrema_decision.m5_pos)
            )
            h4_pos_log = (
                -1.0 if extrema_decision.h4_pos is None else float(extrema_decision.h4_pos)
            )
            extrema_reversal_applied = False
            extrema_reversal_score = 0.0
            reversed_signal, reversal_units_mult, reversal_reason, reversal_score = _extrema_reversal_route(
                signal,
                extrema_decision,
                regime=regime,
                horizon=horizon,
                factors=factors if isinstance(factors, dict) else None,
            )
            if reversed_signal is not None:
                signal = reversed_signal
                extrema_reversal_applied = True
                extrema_reversal_score = float(reversal_score)
                extrema_units_mult = max(extrema_units_mult, float(reversal_units_mult))
                extrema_decision = ExtremaGateDecision(
                    allow_entry=True,
                    reason=str(reversal_reason),
                    units_mult=float(extrema_units_mult),
                    m1_pos=extrema_decision.m1_pos,
                    m5_pos=extrema_decision.m5_pos,
                    h4_pos=extrema_decision.h4_pos,
                )
                tech_route_reasons.append("extrema_reverse")
                if now_mono - last_extrema_log_mono >= config.EXTREMA_LOG_INTERVAL_SEC:
                    LOG.info(
                        "%s extrema reverse side=%s reason=%s score=%.2f emult=%.2f m1=%.2f m5=%.2f h4=%.2f",
                        config.LOG_PREFIX,
                        signal.side,
                        extrema_decision.reason,
                        extrema_reversal_score,
                        extrema_units_mult,
                        m1_pos_log,
                        m5_pos_log,
                        h4_pos_log,
                    )
                    last_extrema_log_mono = now_mono
            if not extrema_decision.allow_entry:
                if now_mono - last_extrema_log_mono >= config.EXTREMA_LOG_INTERVAL_SEC:
                    LOG.info(
                        "%s extrema %s side=%s reason=%s m1=%.2f m5=%.2f h4=%.2f",
                        config.LOG_PREFIX,
                        ("route" if config.TECH_ROUTER_ENABLED else "block"),
                        signal.side,
                        extrema_decision.reason,
                        m1_pos_log,
                        m5_pos_log,
                        h4_pos_log,
                    )
                    last_extrema_log_mono = now_mono
                if config.TECH_ROUTER_ENABLED:
                    extrema_units_mult = max(0.1, float(config.TECH_ROUTER_EXTREMA_BLOCK_UNITS_MULT))
                    extrema_decision = ExtremaGateDecision(
                        allow_entry=True,
                        reason=f"{extrema_decision.reason}_route",
                        units_mult=extrema_units_mult,
                        m1_pos=extrema_decision.m1_pos,
                        m5_pos=extrema_decision.m5_pos,
                        h4_pos=extrema_decision.h4_pos,
                    )
                    tech_route_reasons.append("extrema")
                else:
                    _note_entry_skip(
                        "extrema_block",
                        f"reason={extrema_decision.reason}",
                    )
                    continue
            if (
                extrema_units_mult < 0.999
                or str(extrema_decision.reason).startswith("missing_")
            ) and now_mono - last_extrema_log_mono >= config.EXTREMA_LOG_INTERVAL_SEC:
                LOG.info(
                    "%s extrema pass side=%s reason=%s emult=%.2f m1=%.2f m5=%.2f h4=%.2f",
                    config.LOG_PREFIX,
                    signal.side,
                    extrema_decision.reason,
                    extrema_units_mult,
                    m1_pos_log,
                    m5_pos_log,
                    h4_pos_log,
                )
                last_extrema_log_mono = now_mono

            # Apply fast flip after extrema routing so late short/long overrides
            # can be corrected without suppressing entry frequency.
            post_flip_signal, post_flip_reason = _maybe_fast_direction_flip(
                signal,
                direction_bias=direction_bias,
                horizon=horizon,
                regime=regime,
                now_mono=now_mono,
            )
            if post_flip_signal is not None:
                signal = post_flip_signal
                fast_flip_applied = True
                fast_flip_reason = str(post_flip_reason)
                tech_route_reasons.append("fast_flip")
                refreshed_signal, refreshed_horizon_mult, refreshed_horizon_gate = _apply_horizon_bias(
                    signal,
                    horizon,
                )
                if refreshed_signal is not None and refreshed_horizon_mult > 0.0:
                    signal = refreshed_signal
                    horizon_units_mult = float(refreshed_horizon_mult)
                    horizon_gate = f"{refreshed_horizon_gate}_fflip"
                m1_trend_units_mult, m1_trend_gate = _m1_trend_units_multiplier(
                    signal.side,
                    m1_score,
                    regime=regime,
                )
                bias_units_mult, bias_gate = _direction_units_multiplier(
                    signal.side,
                    direction_bias,
                    signal_mode=signal.mode,
                )
                if bias_units_mult <= 0.0:
                    bias_units_mult = max(0.1, float(config.DIRECTION_BIAS_OPPOSITE_UNITS_MULT))
                    bias_gate = f"{bias_gate}_fflip_clamped"
                if now_mono - last_bias_log_mono >= config.DIRECTION_BIAS_LOG_INTERVAL_SEC:
                    LOG.info(
                        "%s fast_flip side=%s reason=%s bias=%s(%.2f,mom=%.2fp) horizon=%s(%.2f,agree=%d)",
                        config.LOG_PREFIX,
                        signal.side,
                        fast_flip_reason,
                        direction_bias.side if direction_bias is not None else "none",
                        _safe_float(getattr(direction_bias, "score", 0.0), 0.0),
                        _safe_float(getattr(direction_bias, "momentum_pips", 0.0), 0.0),
                        horizon.composite_side if horizon is not None else "none",
                        _safe_float(getattr(horizon, "composite_score", 0.0), 0.0),
                        int(_safe_float(getattr(horizon, "agreement", 0.0), 0.0)),
                    )
                    last_bias_log_mono = now_mono

            sl_flip_signal, sl_flip_reason, sl_flip_eval = _maybe_sl_streak_direction_flip(
                signal,
                strategy_tag=config.STRATEGY_TAG,
                pocket=config.POCKET,
                now_utc=now_utc,
                now_mono=now_mono,
                direction_bias=direction_bias,
                horizon=horizon,
                fast_flip_applied=fast_flip_applied,
            )
            sl_streak = sl_flip_eval.streak
            sl_streak_side_sl_hits_recent = int(sl_flip_eval.side_sl_hits_recent)
            sl_streak_target_market_plus_recent = int(sl_flip_eval.target_market_plus_recent)
            sl_streak_direction_confirmed = bool(sl_flip_eval.direction_confirmed)
            sl_streak_horizon_confirmed = bool(sl_flip_eval.horizon_confirmed)
            if sl_streak is not None:
                sl_streak_side = str(sl_streak.side)
                sl_streak_count = int(sl_streak.streak)
                sl_streak_age_sec = float(sl_streak.age_sec)
                sl_streak_flip_reason = str(sl_flip_reason)
            if sl_flip_signal is not None:
                signal = sl_flip_signal
                sl_streak_flip_applied = True
                sl_streak_flip_reason = str(sl_flip_reason)
                tech_route_reasons.append("sl_streak_flip")
                refreshed_signal, refreshed_horizon_mult, refreshed_horizon_gate = _apply_horizon_bias(
                    signal,
                    horizon,
                )
                if refreshed_signal is not None and refreshed_horizon_mult > 0.0:
                    signal = refreshed_signal
                    horizon_units_mult = float(refreshed_horizon_mult)
                    horizon_gate = f"{refreshed_horizon_gate}_slflip"
                m1_trend_units_mult, m1_trend_gate = _m1_trend_units_multiplier(
                    signal.side,
                    m1_score,
                    regime=regime,
                )
                bias_units_mult, bias_gate = _direction_units_multiplier(
                    signal.side,
                    direction_bias,
                    signal_mode=signal.mode,
                )
                if bias_units_mult <= 0.0:
                    bias_units_mult = max(0.1, float(config.DIRECTION_BIAS_OPPOSITE_UNITS_MULT))
                    bias_gate = f"{bias_gate}_slflip_clamped"
                if (
                    now_mono - last_sl_streak_log_mono
                    >= config.SL_STREAK_DIRECTION_FLIP_LOG_INTERVAL_SEC
                ):
                    LOG.info(
                        "%s sl_streak_flip side=%s reason=%s streak=%sx%d age=%.0fs sl_hits=%d mkt_plus=%d tech(dir=%d,hz=%d)",
                        config.LOG_PREFIX,
                        signal.side,
                        sl_streak_flip_reason,
                        sl_streak_side or "-",
                        sl_streak_count,
                        sl_streak_age_sec,
                        sl_streak_side_sl_hits_recent,
                        sl_streak_target_market_plus_recent,
                        int(sl_streak_direction_confirmed),
                        int(sl_streak_horizon_confirmed),
                    )
                    last_sl_streak_log_mono = now_mono

            side_metrics_flip_signal, side_metrics_flip_candidate_reason, side_metrics_eval = (
                _maybe_side_metrics_direction_flip(
                    signal,
                    strategy_tag=config.STRATEGY_TAG,
                    pocket=config.POCKET,
                    now_mono=now_mono,
                )
            )
            side_metrics_flip_reason = str(side_metrics_flip_candidate_reason)
            side_metrics_flip_current_sl_rate = float(side_metrics_eval.current_sl_rate)
            side_metrics_flip_target_sl_rate = float(side_metrics_eval.target_sl_rate)
            side_metrics_flip_current_market_plus_rate = float(
                side_metrics_eval.current_market_plus_rate
            )
            side_metrics_flip_target_market_plus_rate = float(
                side_metrics_eval.target_market_plus_rate
            )
            side_metrics_flip_current_trades = int(side_metrics_eval.current_trades)
            side_metrics_flip_target_trades = int(side_metrics_eval.target_trades)
            if side_metrics_flip_signal is not None:
                signal = side_metrics_flip_signal
                side_metrics_flip_applied = True
                side_metrics_flip_reason = str(side_metrics_flip_candidate_reason)
                tech_route_reasons.append("side_metrics_flip")
                refreshed_signal, refreshed_horizon_mult, refreshed_horizon_gate = _apply_horizon_bias(
                    signal,
                    horizon,
                )
                if refreshed_signal is not None and refreshed_horizon_mult > 0.0:
                    signal = refreshed_signal
                    horizon_units_mult = float(refreshed_horizon_mult)
                    horizon_gate = f"{refreshed_horizon_gate}_smflip"
                m1_trend_units_mult, m1_trend_gate = _m1_trend_units_multiplier(
                    signal.side,
                    m1_score,
                    regime=regime,
                )
                bias_units_mult, bias_gate = _direction_units_multiplier(
                    signal.side,
                    direction_bias,
                    signal_mode=signal.mode,
                )
                if bias_units_mult <= 0.0:
                    bias_units_mult = max(0.1, float(config.DIRECTION_BIAS_OPPOSITE_UNITS_MULT))
                    bias_gate = f"{bias_gate}_smflip_clamped"
                if (
                    now_mono - last_side_metrics_flip_log_mono
                    >= config.SIDE_METRICS_DIRECTION_FLIP_LOG_INTERVAL_SEC
                ):
                    LOG.info(
                        "%s side_metrics_flip side=%s reason=%s sl=%.2f/%.2f mplus=%.2f/%.2f n=%d/%d",
                        config.LOG_PREFIX,
                        signal.side,
                        side_metrics_flip_reason,
                        side_metrics_flip_current_sl_rate,
                        side_metrics_flip_target_sl_rate,
                        side_metrics_flip_current_market_plus_rate,
                        side_metrics_flip_target_market_plus_rate,
                        side_metrics_flip_current_trades,
                        side_metrics_flip_target_trades,
                    )
                    last_side_metrics_flip_log_mono = now_mono
            decision_cooldown_sec = max(
                config.ENTRY_COOLDOWN_SEC * config.INSTANT_COOLDOWN_SCALE_MIN,
                config.ENTRY_COOLDOWN_SEC * decision_speed_scale,
            )
            if now_mono - last_entry_mono < decision_cooldown_sec:
                _note_entry_skip(
                    "cooldown",
                    f"elapsed={now_mono-last_entry_mono:.2f}s limit={decision_cooldown_sec:.2f}s",
                )
                continue
            if not rate_limiter.allow(now_mono):
                _note_entry_skip("rate_limited")
                continue

            trap_state = _compute_trap_state(positions, mid_price=signal.mid)
            if trap_state.active and now_mono - last_trap_log_mono >= config.TRAP_LOG_INTERVAL_SEC:
                LOG.info(
                    "%s trap_active long=%.0f short=%.0f net=%.2f dd=(L%.2f/S%.2f/C%.2f)p unreal=%.0f",
                    config.LOG_PREFIX,
                    trap_state.long_units,
                    trap_state.short_units,
                    trap_state.net_ratio,
                    trap_state.long_dd_pips,
                    trap_state.short_dd_pips,
                    trap_state.combined_dd_pips,
                    trap_state.unrealized_pl,
                )
                last_trap_log_mono = now_mono

            # Use account-level long/short units for risk sizing so manual trades
            # (and other pockets) are correctly reflected in hedge sizing.
            account_long_units = _safe_float(getattr(trap_state, "long_units", 0.0), 0.0)
            account_short_units = _safe_float(getattr(trap_state, "short_units", 0.0), 0.0)

            long_units = _safe_float(pocket_info.get("long_units"), 0.0)
            short_units = _safe_float(pocket_info.get("short_units"), 0.0)
            trap_hedge_bypass = trap_state.active and config.TRAP_BYPASS_NO_HEDGE
            if config.NO_HEDGE_ENTRY and not trap_hedge_bypass:
                if signal.side == "long" and short_units > 0.0:
                    _note_entry_skip(
                        "hedge_block",
                        f"side=long short_units={short_units:.0f}",
                    )
                    continue
                if signal.side == "short" and long_units > 0.0:
                    _note_entry_skip(
                        "hedge_block",
                        f"side=short long_units={long_units:.0f}",
                    )
                    continue

            snap = get_account_snapshot(cache_ttl_sec=0.5)
            nav = max(_safe_float(snap.nav, 0.0), 1.0)
            balance = max(_safe_float(snap.balance, 0.0), nav)
            margin_available = max(_safe_float(snap.margin_available, 0.0), 0.0)
            margin_rate = _safe_float(snap.margin_rate, 0.0)
            free_ratio = _safe_float(snap.free_margin_ratio, 0.0)

            # 低マージン判定は order_manager の共通ガードへ委譲し、戦略側ではローカル拒否を行わない。

            active_total, active_long, active_short = _strategy_trade_counts(
                pocket_info,
                config.STRATEGY_TAG,
            )
            max_active_trades, max_per_direction, cap_expanded = _resolve_active_caps(
                free_ratio=free_ratio,
                margin_available=margin_available,
            )
            if not _allow_signal_when_max_active(
                side=signal.side,
                active_total=active_total,
                active_long=active_long,
                active_short=active_short,
                max_active_trades=max_active_trades,
            ):
                _note_entry_skip(
                    "max_active_cap",
                    f"total={active_total} long={active_long} short={active_short} cap={max_active_trades}",
                )
                continue
            if (
                active_total >= config.MAX_ACTIVE_TRADES
                and now_mono - last_max_active_bypass_log_mono >= config.ACTIVE_CAP_BYPASS_LOG_INTERVAL_SEC
            ):
                LOG.info(
                    "%s max_active bypass side=%s total=%d long=%d short=%d base_cap=%d eff_cap=%d expanded=%s free_ratio=%.3f margin_avail=%.0f",
                    config.LOG_PREFIX,
                    signal.side,
                    active_total,
                    active_long,
                    active_short,
                    config.MAX_ACTIVE_TRADES,
                    max_active_trades,
                    cap_expanded,
                    free_ratio,
                    margin_available,
                )
                last_max_active_bypass_log_mono = now_mono
            if signal.side == "long" and active_long >= max_per_direction:
                _note_entry_skip(
                    "direction_cap",
                    f"side=long active={active_long} cap={max_per_direction}",
                )
                continue
            if signal.side == "short" and active_short >= max_per_direction:
                _note_entry_skip(
                    "direction_cap",
                    f"side=short active={active_short} cap={max_per_direction}",
                )
                continue

            tp_profile = _load_tp_timing_profile(config.STRATEGY_TAG, config.POCKET)
            tp_pips, sl_pips = _compute_targets(
                spread_pips=signal.spread_pips,
                momentum_pips=signal.momentum_pips,
                side=signal.side,
                tp_profile=tp_profile,
                regime=regime,
                signal_range_pips=signal.range_pips,
                signal_instant_range_pips=signal.instant_range_pips,
            )
            signal_vol_bucket = _regime_vol_bucket(
                regime,
                low_pips=config.TP_VOL_LOW_PIPS,
                high_pips=config.TP_VOL_HIGH_PIPS,
            )
            if signal.range_pips > 0.0:
                signal_vol_bucket = _norm01(
                    signal.range_pips,
                    config.TP_VOL_LOW_PIPS,
                    config.TP_VOL_HIGH_PIPS,
                )
            if config.INSTANT_VOL_ADAPT_ENABLED and signal.instant_range_pips > 0.0:
                instant_vol_bucket = _norm01(
                    signal.instant_range_pips,
                    config.TP_VOL_LOW_PIPS,
                    config.TP_VOL_HIGH_PIPS,
                )
                signal_vol_bucket = _lerp(
                    signal_vol_bucket,
                    instant_vol_bucket,
                    config.INSTANT_VOL_BUCKET_WEIGHT,
                )
            tech_profile = _build_technical_trade_profile(
                route_reasons=tech_route_reasons,
                lookahead_decision=lookahead_decision,
                vol_bucket=signal_vol_bucket,
            )
            tp_pips *= max(0.2, tech_profile.tp_mult)
            tp_base_cfg = (
                config.SHORT_TP_BASE_PIPS
                if signal.side == "short"
                else config.TP_BASE_PIPS
            )
            tp_max_cfg = (
                config.SHORT_TP_MAX_PIPS if signal.side == "short" else config.TP_MAX_PIPS
            )
            tp_floor = max(tp_base_cfg, signal.spread_pips + config.TP_NET_MIN_FLOOR_PIPS)
            tp_pips = _clamp(tp_pips, tp_floor, tp_max_cfg)
            sl_floor = max(
                config.SHORT_SL_MIN_PIPS
                if signal.side == "short"
                else config.SL_MIN_PIPS,
                signal.spread_pips * config.SL_SPREAD_MULT + config.SL_SPREAD_BUFFER_PIPS,
            )
            sl_max_cfg = (
                config.SHORT_SL_MAX_PIPS if signal.side == "short" else config.SL_MAX_PIPS
            )
            sl_pips = _clamp(
                sl_pips * max(0.2, tech_profile.sl_mult),
                sl_floor,
                sl_max_cfg,
            )
            dynamic_max_hold_sec, dynamic_hard_loss_pips = _scaled_force_exit_thresholds(
                base_max_hold_sec=_force_exit_thresholds_for_side(signal.side)[0],
                base_hard_loss_pips=_force_exit_thresholds_for_side(signal.side)[1],
                profile=tech_profile,
                vol_bucket=signal_vol_bucket,
            )
            risk_sl_pips = float(sl_pips if config.USE_SL else dynamic_hard_loss_pips)
            if risk_sl_pips <= 0.0:
                risk_sl_pips = float(
                    config.SHORT_SL_MIN_PIPS if signal.side == "short" else config.SL_MIN_PIPS
                )
            raw_entry_probability = _raw_entry_probability(signal)
            entry_probability, probability_units_mult, probability_meta = _adjust_entry_probability_alignment(
                signal=signal,
                raw_probability=raw_entry_probability,
                direction_bias=direction_bias,
                horizon=horizon,
                m1_score=m1_score,
            )
            probability_band_units_mult, probability_band_meta = _entry_probability_band_units_multiplier(
                strategy_tag=config.STRATEGY_TAG,
                pocket=config.POCKET,
                side=signal.side,
                entry_probability=entry_probability,
                now_mono=now_mono,
            )

            conf_mult = _confidence_scale(signal.confidence)
            strength_mult = max(
                0.75,
                min(1.25, abs(signal.momentum_pips) / max(0.1, signal.trigger_pips)),
            )
            base_units = int(
                round(
                    scale_base_units(
                        config.BASE_ENTRY_UNITS,
                        equity=balance,
                        ref_equity=balance,
                        min_units=config.MIN_UNITS,
                        max_units=config.MAX_UNITS,
                        env_prefix=config.ENV_PREFIX,
                    )
                )
            )
            units = int(round(base_units * conf_mult * strength_mult * bias_units_mult * lookahead_units_mult))
            units = int(round(units * regime_units_mult))
            units = int(round(units * horizon_units_mult))
            units = int(round(units * m1_trend_units_mult))
            units = int(round(units * extrema_units_mult))
            units = int(round(units * probability_units_mult))
            units = int(round(units * probability_band_units_mult))
            snapshot_units_scale = _snapshot_stale_units_scale()
            if snapshot_units_scale < 1.0:
                units = int(round(units * snapshot_units_scale))

            lot = allowed_lot(
                nav,
                risk_sl_pips,
                margin_available=margin_available,
                margin_rate=margin_rate,
                price=signal.mid,
                pocket=config.POCKET,
                side=signal.side,
                open_long_units=account_long_units,
                open_short_units=account_short_units,
                strategy_tag=config.STRATEGY_TAG,
            )
            units_risk = int(round(max(0.0, lot) * 100000))
            units = min(units, units_risk, config.MAX_UNITS)
            bias_scale, bias_meta = _directional_bias_scale(ticks, signal.side)
            if bias_scale <= 0.0:
                _note_entry_skip("directional_bias_zero")
                continue
            if signal.mode == "revert":
                raw_side_bias = float(bias_scale)
                weight = _clamp(config.REVERT_SIDE_BIAS_PENALTY_WEIGHT, 0.0, 1.0)
                bias_scale = max(
                    config.REVERT_SIDE_BIAS_FLOOR,
                    1.0 - (1.0 - raw_side_bias) * weight,
                )
                bias_meta["raw_scale"] = raw_side_bias
                bias_meta["mode_adjusted_scale"] = float(bias_scale)
            units = int(round(units * bias_scale))
            if units < config.MIN_UNITS:
                _note_entry_skip(
                    "units_below_min",
                    f"units={units} min={config.MIN_UNITS}",
                )
                continue
            units = abs(units)
            if signal.side == "short":
                units = -units
            entry_price = signal.ask if signal.side == "long" else signal.bid
            if entry_price <= 0.0:
                entry_price = signal.mid
            if entry_price <= 0.0:
                _note_entry_skip("invalid_entry_price")
                continue

            effective_tp_pips = 0.0 if not config.TP_ENABLED else tp_pips
            if signal.side == "long":
                sl_price = (
                    entry_price - sl_pips * config.PIP_VALUE if config.USE_SL else None
                )
                tp_price = (
                    entry_price + effective_tp_pips * config.PIP_VALUE
                    if effective_tp_pips > 0.0
                    else None
                )
            else:
                sl_price = (
                    entry_price + sl_pips * config.PIP_VALUE if config.USE_SL else None
                )
                tp_price = (
                    entry_price - effective_tp_pips * config.PIP_VALUE
                    if effective_tp_pips > 0.0
                    else None
                )

            sl_price, tp_price = clamp_sl_tp(
                entry_price,
                sl_price,
                tp_price,
                is_buy=signal.side == "long",
                strategy_tag=config.STRATEGY_TAG,
                pocket=config.POCKET,
            )
            order_policy = "market_tech_router" if tech_route_reasons else "market_guarded"

            entry_thesis = {
                "strategy_tag": config.STRATEGY_TAG,
                "pattern_gate_opt_in": bool(config.PATTERN_GATE_OPT_IN),
                "env_prefix": config.ENV_PREFIX,
                "signal_window_sec": round(signal.signal_window_sec, 3),
                "signal_window_adaptive_enabled": bool(signal_window_meta.get("enabled", False)),
                "signal_window_adaptive_shadow_enabled": bool(
                    signal_window_meta.get("shadow_enabled", False)
                ),
                "signal_window_adaptive_applied": bool(signal_window_meta.get("applied", False)),
                "signal_window_adaptive_live_sec": round(
                    _safe_float(signal_window_meta.get("live_window_sec"), signal.signal_window_sec),
                    3,
                ),
                "signal_window_adaptive_selected_sec": round(
                    _safe_float(signal_window_meta.get("selected_window_sec"), signal.signal_window_sec),
                    3,
                ),
                "signal_window_adaptive_best_sec": round(
                    _safe_float(signal_window_meta.get("best_window_sec"), signal.signal_window_sec),
                    3,
                ),
                "signal_window_adaptive_live_score_pips": round(
                    _safe_float(signal_window_meta.get("live_score_pips"), 0.0),
                    5,
                ),
                "signal_window_adaptive_selected_score_pips": round(
                    _safe_float(signal_window_meta.get("selected_score_pips"), 0.0),
                    5,
                ),
                "signal_window_adaptive_best_score_pips": round(
                    _safe_float(signal_window_meta.get("best_score_pips"), 0.0),
                    5,
                ),
                "signal_window_adaptive_best_sample": int(
                    _safe_float(signal_window_meta.get("best_sample"), 0.0)
                ),
                "signal_window_adaptive_candidate_count": int(
                    _safe_float(signal_window_meta.get("candidate_count"), 0.0)
                ),
                "window_sec": config.WINDOW_SEC,
                "momentum_pips": round(signal.momentum_pips, 3),
                "trigger_pips": round(signal.trigger_pips, 3),
                "imbalance": round(signal.imbalance, 3),
                "tick_rate": round(signal.tick_rate, 3),
                "tick_age_ms": round(signal.tick_age_ms, 1),
                "spread_pips": round(signal.spread_pips, 3),
                "confidence": int(signal.confidence),
                "entry_probability_raw": round(raw_entry_probability, 6),
                "tp_pips": round(effective_tp_pips, 3),
                "sl_pips": 0.0 if not config.USE_SL else round(sl_pips, 3),
                "sl_risk_pips": round(risk_sl_pips, 3),
                "disable_entry_hard_stop": bool(config.DISABLE_ENTRY_HARD_STOP),
                "signal_mode": signal.mode,
                "signal_mode_score": round(signal.mode_score, 3),
                "signal_momentum_score": round(signal.momentum_score, 3),
                "signal_revert_score": round(signal.revert_score, 3),
                "tp_time_mult": round(tp_profile.multiplier, 3),
                "signal_range_pips": round(float(signal.range_pips), 3),
                "signal_instant_range_pips": round(float(signal.instant_range_pips), 3),
                "tp_vol_bucket": round(
                    _clamp(
                        signal_vol_bucket,
                        0.0,
                        1.0,
                    ),
                    3,
                ),
                "snapshot_degraded_mode": bool(snapshot_degraded_mode),
                "snapshot_fetch_failures": int(_SNAPSHOT_FETCH_FAILURES),
                "snapshot_age_limit_ms": round(snapshot_age_limit_ms, 1),
                "snapshot_stale_units_scale": round(float(snapshot_units_scale), 3),
                "vol_bucket_source": "signal_range_then_regime",
                "tp_time_avg_sec": (
                    round(tp_profile.avg_tp_sec, 1) if tp_profile.avg_tp_sec > 0.0 else None
                ),
                "entry_probability": round(entry_probability, 3),
                "entry_probability_units_mult": round(probability_units_mult, 3),
                "entry_probability_band_units_mult": round(probability_band_units_mult, 3),
                "tp_time_target_sec": round(config.TP_TARGET_HOLD_SEC, 1),
                "tp_time_sample": int(tp_profile.sample),
                "side_bias_scale": round(float(bias_scale), 3),
                "side_bias_drift_pips": round(float(_safe_float(bias_meta.get("drift_pips"), 0.0)), 3),
                "side_bias_aligned_pips": round(float(_safe_float(bias_meta.get("aligned_pips"), 0.0)), 3),
                "side_bias_contra_pips": round(float(_safe_float(bias_meta.get("contra_pips"), 0.0)), 3),
                "side_bias_mode_adjusted_scale": round(float(_safe_float(bias_meta.get("mode_adjusted_scale"), bias_scale)), 3),
                "entry_mode": "market_ping_5s",
                "mtf_regime_gate": regime_gate,
                "mtf_regime_units_mult": round(float(regime_units_mult), 3),
                "horizon_gate": horizon_gate,
                "horizon_units_mult": round(float(horizon_units_mult), 3),
                "m1_trend_score": round(_safe_float(m1_score, 0.0), 3),
                "m1_trend_vol_bucket": round(
                    _regime_vol_bucket(
                        regime,
                        low_pips=config.M1_TREND_VOL_LOW_PIPS,
                        high_pips=config.M1_TREND_VOL_HIGH_PIPS,
                    ),
                    3,
                ),
                "m1_trend_gate": m1_trend_gate,
                "m1_trend_units_mult": round(float(m1_trend_units_mult), 3),
                "trap_active": bool(trap_state.active),
                "entry_ref": round(entry_price, 3),
                "execution": {
                    "order_policy": order_policy,
                    "ideal_entry": round(entry_price, 3),
                    "chase_max": round(config.ENTRY_CHASE_MAX_PIPS * config.PIP_VALUE, 4),
                    "chase_max_pips": round(config.ENTRY_CHASE_MAX_PIPS, 3),
                },
                "direction_bias_gate": bias_gate,
                "direction_bias_units_mult": round(bias_units_mult, 3),
                "fast_direction_flip_enabled": bool(config.FAST_DIRECTION_FLIP_ENABLED),
                "fast_direction_flip_applied": bool(fast_flip_applied),
                "fast_direction_flip_reason": fast_flip_reason,
                "sl_streak_direction_flip_enabled": bool(config.SL_STREAK_DIRECTION_FLIP_ENABLED),
                "sl_streak_direction_flip_applied": bool(sl_streak_flip_applied),
                "sl_streak_direction_flip_reason": sl_streak_flip_reason,
                "sl_streak_side": sl_streak_side,
                "sl_streak_count": int(sl_streak_count),
                "sl_streak_age_sec": (
                    round(float(sl_streak_age_sec), 1) if sl_streak_age_sec >= 0.0 else None
                ),
                "sl_streak_side_sl_hits_recent": int(sl_streak_side_sl_hits_recent),
                "sl_streak_target_market_plus_recent": int(sl_streak_target_market_plus_recent),
                "sl_streak_direction_confirmed": bool(sl_streak_direction_confirmed),
                "sl_streak_horizon_confirmed": bool(sl_streak_horizon_confirmed),
                "side_metrics_direction_flip_enabled": bool(
                    config.SIDE_METRICS_DIRECTION_FLIP_ENABLED
                ),
                "side_metrics_direction_flip_applied": bool(side_metrics_flip_applied),
                "side_metrics_direction_flip_reason": side_metrics_flip_reason,
                "side_metrics_direction_flip_current_sl_rate": round(
                    side_metrics_flip_current_sl_rate,
                    4,
                ),
                "side_metrics_direction_flip_target_sl_rate": round(
                    side_metrics_flip_target_sl_rate,
                    4,
                ),
                "side_metrics_direction_flip_current_market_plus_rate": round(
                    side_metrics_flip_current_market_plus_rate,
                    4,
                ),
                "side_metrics_direction_flip_target_market_plus_rate": round(
                    side_metrics_flip_target_market_plus_rate,
                    4,
                ),
                "side_metrics_direction_flip_current_trades": int(
                    side_metrics_flip_current_trades
                ),
                "side_metrics_direction_flip_target_trades": int(
                    side_metrics_flip_target_trades
                ),
                "lookahead_gate_enabled": bool(config.LOOKAHEAD_GATE_ENABLED),
                "lookahead_units_mult": round(lookahead_units_mult, 3),
                "extrema_gate_enabled": bool(config.EXTREMA_GATE_ENABLED),
                "extrema_gate_reason": extrema_decision.reason,
                "extrema_units_mult": round(extrema_units_mult, 3),
                "extrema_reversal_enabled": bool(config.EXTREMA_REVERSAL_ENABLED),
                "extrema_reversal_applied": bool(extrema_reversal_applied),
                "tech_router_enabled": bool(config.TECH_ROUTER_ENABLED),
                "tech_route_reasons": list(tech_route_reasons),
                "tech_counter_pressure": bool(tech_profile.counter_pressure),
                "tech_tp_mult": round(float(tech_profile.tp_mult), 3),
                "tech_sl_mult": round(float(tech_profile.sl_mult), 3),
                "tech_hold_mult": round(float(tech_profile.hold_mult), 3),
                "tech_hard_loss_mult": round(float(tech_profile.hard_loss_mult), 3),
            }
            entry_thesis["entry_probability_alignment"] = {
                "enabled": bool(probability_meta.get("enabled", False)),
                "support": round(_safe_float(probability_meta.get("support"), 0.0), 6),
                "counter": round(_safe_float(probability_meta.get("counter"), 0.0), 6),
                "boost": round(_safe_float(probability_meta.get("boost"), 0.0), 6),
                "penalty": round(_safe_float(probability_meta.get("penalty"), 0.0), 6),
                "counter_extra": round(_safe_float(probability_meta.get("counter_extra"), 0.0), 6),
                "direction_edge": _safe_float(probability_meta.get("direction_edge"), None),
                "horizon_edge": _safe_float(probability_meta.get("horizon_edge"), None),
                "m1_edge": _safe_float(probability_meta.get("m1_edge"), None),
                "floor_applied": bool(probability_meta.get("floor_applied", False)),
                "weights": probability_meta.get("weights"),
            }
            entry_thesis["entry_probability_band_allocation"] = {
                "enabled": bool(probability_band_meta.get("enabled", False)),
                "reason": str(probability_band_meta.get("reason") or ""),
                "bucket": str(probability_band_meta.get("bucket") or "mid"),
                "band_mult": round(_safe_float(probability_band_meta.get("band_mult"), 1.0), 6),
                "side_mult": round(_safe_float(probability_band_meta.get("side_mult"), 1.0), 6),
                "units_mult": round(_safe_float(probability_band_meta.get("units_mult"), 1.0), 6),
                "shift_strength": round(
                    _safe_float(probability_band_meta.get("shift_strength"), 0.0),
                    6,
                ),
                "sample_strength": round(
                    _safe_float(probability_band_meta.get("sample_strength"), 0.0),
                    6,
                ),
                "high_sample": int(_safe_float(probability_band_meta.get("high_sample"), 0.0)),
                "low_sample": int(_safe_float(probability_band_meta.get("low_sample"), 0.0)),
                "high_mean_pips": round(
                    _safe_float(probability_band_meta.get("high_mean_pips"), 0.0),
                    6,
                ),
                "low_mean_pips": round(
                    _safe_float(probability_band_meta.get("low_mean_pips"), 0.0),
                    6,
                ),
                "high_win_rate": round(
                    _safe_float(probability_band_meta.get("high_win_rate"), 0.0),
                    6,
                ),
                "low_win_rate": round(
                    _safe_float(probability_band_meta.get("low_win_rate"), 0.0),
                    6,
                ),
                "high_sl_rate": round(
                    _safe_float(probability_band_meta.get("high_sl_rate"), 0.0),
                    6,
                ),
                "low_sl_rate": round(
                    _safe_float(probability_band_meta.get("low_sl_rate"), 0.0),
                    6,
                ),
                "gap_pips": round(_safe_float(probability_band_meta.get("gap_pips"), 0.0), 6),
                "gap_win_rate": round(
                    _safe_float(probability_band_meta.get("gap_win_rate"), 0.0),
                    6,
                ),
                "gap_sl_rate": round(
                    _safe_float(probability_band_meta.get("gap_sl_rate"), 0.0),
                    6,
                ),
                "side_trades": int(_safe_float(probability_band_meta.get("side_trades"), 0.0)),
                "side_sl_hits": int(_safe_float(probability_band_meta.get("side_sl_hits"), 0.0)),
                "side_market_plus": int(
                    _safe_float(probability_band_meta.get("side_market_plus"), 0.0)
                ),
                "side_sl_rate": round(
                    _safe_float(probability_band_meta.get("side_sl_rate"), 0.0),
                    6,
                ),
                "side_market_plus_rate": round(
                    _safe_float(probability_band_meta.get("side_market_plus_rate"), 0.0),
                    6,
                ),
            }
            if dynamic_max_hold_sec > 0.0:
                entry_thesis["force_exit_max_hold_sec"] = round(dynamic_max_hold_sec, 1)
            if dynamic_hard_loss_pips > 0.0:
                entry_thesis["force_exit_max_floating_loss_pips"] = round(dynamic_hard_loss_pips, 3)
            if extrema_decision.m1_pos is not None:
                entry_thesis["extrema_m1_pos"] = round(extrema_decision.m1_pos, 4)
            if extrema_decision.m5_pos is not None:
                entry_thesis["extrema_m5_pos"] = round(extrema_decision.m5_pos, 4)
            if extrema_decision.h4_pos is not None:
                entry_thesis["extrema_h4_pos"] = round(extrema_decision.h4_pos, 4)
            if extrema_reversal_applied:
                entry_thesis["extrema_reversal_score"] = round(extrema_reversal_score, 3)
            if direction_bias is not None:
                entry_thesis.update(
                    {
                        "direction_bias_side": direction_bias.side,
                        "direction_bias_score": round(direction_bias.score, 4),
                        "direction_bias_momentum_pips": round(direction_bias.momentum_pips, 3),
                        "direction_bias_flow": round(direction_bias.flow, 3),
                        "direction_bias_range_pips": round(direction_bias.range_pips, 3),
                        "direction_bias_vol_norm": round(direction_bias.vol_norm, 3),
                        "direction_bias_tick_rate": round(direction_bias.tick_rate, 3),
                        "direction_bias_span_sec": round(direction_bias.span_sec, 3),
                    }
                )
            if lookahead_decision is not None:
                entry_thesis.update(
                    {
                        "lookahead_reason": lookahead_decision.reason,
                        "lookahead_pred_move_pips": round(lookahead_decision.pred_move_pips, 4),
                        "lookahead_cost_pips": round(lookahead_decision.cost_pips, 4),
                        "lookahead_edge_pips": round(lookahead_decision.edge_pips, 4),
                        "lookahead_slippage_est_pips": round(
                            lookahead_decision.slippage_est_pips,
                            4,
                        ),
                        "lookahead_range_pips": round(lookahead_decision.range_pips, 4),
                        "lookahead_momentum_aligned_pips": round(
                            lookahead_decision.momentum_aligned_pips,
                            4,
                        ),
                        "lookahead_direction_bias_aligned": round(
                            lookahead_decision.direction_bias_aligned,
                            4,
                        ),
                    }
                )
            if regime is not None:
                entry_thesis.update(
                    {
                        "mtf_regime_mode": regime.mode,
                        "mtf_regime_side": regime.side,
                        "mtf_regime_trend_score": round(regime.trend_score, 4),
                        "mtf_regime_heat_score": round(regime.heat_score, 4),
                        "mtf_regime_adx_m1": round(regime.adx_m1, 3),
                        "mtf_regime_adx_m5": round(regime.adx_m5, 3),
                        "mtf_regime_atr_m1": round(regime.atr_m1, 3),
                        "mtf_regime_atr_m5": round(regime.atr_m5, 3),
                    }
                )
            if horizon is not None:
                entry_thesis.update(
                    {
                        "horizon_composite_side": horizon.composite_side,
                        "horizon_composite_score": round(horizon.composite_score, 4),
                        "horizon_agreement": int(horizon.agreement),
                        "horizon_long_side": horizon.long_side,
                        "horizon_long_score": round(horizon.long_score, 4),
                        "horizon_mid_side": horizon.mid_side,
                        "horizon_mid_score": round(horizon.mid_score, 4),
                        "horizon_short_side": horizon.short_side,
                        "horizon_short_score": round(horizon.short_score, 4),
                        "horizon_micro_side": horizon.micro_side,
                        "horizon_micro_score": round(horizon.micro_score, 4),
                    }
                )
            if config.FORCE_EXIT_POLICY_GENERATION:
                entry_thesis["policy_generation"] = config.FORCE_EXIT_POLICY_GENERATION
            if trap_state.active:
                entry_thesis.update(
                    {
                        "trap_long_units": int(round(trap_state.long_units)),
                        "trap_short_units": int(round(trap_state.short_units)),
                        "trap_net_ratio": round(trap_state.net_ratio, 3),
                        "trap_combined_dd_pips": round(trap_state.combined_dd_pips, 3),
                        "trap_unrealized_pl": round(trap_state.unrealized_pl, 1),
                    }
                )

            client_order_id = _client_order_id(signal.side)
            entry_thesis_ctx = None
            for _name in ("entry_thesis", "thesis"):
                _candidate = locals().get(_name)
                if isinstance(_candidate, dict):
                    entry_thesis_ctx = _candidate
                    break
            if entry_thesis_ctx is None:
                entry_thesis_ctx = {}

            _tech_pocket = str(locals().get("pocket", config.POCKET))
            _tech_side_raw = str(
                locals().get("side", locals().get("direction", getattr(signal, "side", "long")))
            ).lower()
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
            entry_thesis_ctx.setdefault("technical_context_ticks", ["latest_bid", "latest_ask", "latest_mid", "spread_pips"])
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
            entry_thesis_ctx.setdefault("forecast_profile", {"timeframe": "M1", "step_bars": 1})
            entry_thesis_ctx.setdefault("forecast_timeframe", "M1")
            entry_thesis_ctx.setdefault("forecast_step_bars", 1)
            entry_thesis_ctx.setdefault("forecast_horizon", "1m")
            entry_thesis_ctx.setdefault("forecast_technical_only", True)

            tech_decision = evaluate_entry_techniques(
                entry_price=_tech_entry_price,
                side=_tech_side,
                pocket=_tech_pocket,
                strategy_tag=_tech_signal_tag,
                entry_thesis=entry_thesis_ctx,
                allow_candle=bool(entry_thesis_ctx.get("tech_allow_candle", False)),
            )
            if not tech_decision.allowed and not getattr(config, "TECH_FAILOPEN", True):
                continue

            entry_thesis_ctx["tech_score"] = round(tech_decision.score, 3) if tech_decision.score is not None else None
            entry_thesis_ctx["tech_coverage"] = (
                round(tech_decision.coverage, 3) if tech_decision.coverage is not None else None
            )
            entry_thesis_ctx["tech_entry"] = tech_decision.debug
            entry_thesis_ctx["tech_reason"] = tech_decision.reason
            entry_thesis_ctx["tech_decision_allowed"] = bool(tech_decision.allowed)
            _tech_tp_mult = max(
                0.2,
                min(2.0, float(getattr(tech_decision, "tp_mult", 1.0) or 1.0)),
            )
            entry_thesis_ctx["tech_tp_mult"] = round(_tech_tp_mult, 3)
            if isinstance(tp_price, (int, float)) and tp_price > 0 and _tech_entry_price > 0:
                _tp_gap = abs(float(tp_price) - float(_tech_entry_price))
                if _tp_gap > 0:
                    _tp_target = (
                        float(_tech_entry_price) + (_tp_gap * _tech_tp_mult)
                        if _tech_side == "long"
                        else float(_tech_entry_price) - (_tp_gap * _tech_tp_mult)
                    )
                    sl_price, tp_price = clamp_sl_tp(
                        price=float(_tech_entry_price),
                        sl=sl_price,
                        tp=round(_tp_target, 3),
                        is_buy=(_tech_side == "long"),
                    )
                    if isinstance(tp_price, (int, float)):
                        entry_thesis_ctx["tp_pips"] = round(
                            abs(float(tp_price) - float(_tech_entry_price)) / 0.01,
                            3,
                        )

            _tech_units_raw = locals().get("units")
            if isinstance(_tech_units_raw, (int, float)):
                _tech_units = int(round(abs(float(_tech_units_raw)) * tech_decision.size_mult))
                if _tech_units <= 0:
                    continue
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

            _meta = {"env_prefix": config.ENV_PREFIX, "ENV_PREFIX": config.ENV_PREFIX}


            result = await market_order(
                instrument="USD_JPY",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                pocket=config.POCKET,
                client_order_id=client_order_id,
                strategy_tag=config.STRATEGY_TAG,
                confidence=int(signal.confidence),
                entry_thesis=dict(
                    entry_thesis,
                    env_prefix=config.ENV_PREFIX,
                    ENV_PREFIX=config.ENV_PREFIX,
                    entry_units_intent=abs(units),
                ),
                meta=_meta,
            )
            if result:
                pos_manager.register_open_trade(str(result), config.POCKET, client_order_id)
            else:
                order_status = get_last_order_status_by_client_id(client_order_id)
                status = (
                    order_status.get("status")
                    if isinstance(order_status, dict)
                    else None
                )
                detail = "-"
                if isinstance(order_status, dict):
                    detail_parts: list[str] = []
                    attempt = order_status.get("attempt")
                    err_code = order_status.get("error_code")
                    err_msg = order_status.get("error_message")
                    ts = order_status.get("ts")
                    if attempt is not None:
                        detail_parts.append(f"attempt={attempt}")
                    if err_code:
                        detail_parts.append(f"err_code={err_code}")
                    if err_msg:
                        detail_parts.append(f"err={err_msg}")
                    if ts:
                        detail_parts.append(f"ts={ts}")
                    if detail_parts:
                        detail = ", ".join(detail_parts)
                if status:
                    reason = str(status)
                else:
                    reason = "order_manager_none"
                LOG.warning(
                    "%s market_order rejected side=%s reason=%s detail=%s cid=%s",
                    config.LOG_PREFIX,
                    signal.side,
                    reason,
                    detail,
                    client_order_id,
                )
                _note_entry_skip(f"order_reject:{reason}", detail=detail, side=signal.side)

            last_entry_mono = now_mono
            rate_limiter.record(now_mono)

            LOG.info(
                "%s open mode=%s side=%s units=%s conf=%d prob=%.3f->%.3f p_mult=%.2f p_band=%.2f mom=%.2fp trig=%.2fp sp=%.2fp tp=%.2fp sl=%.2fp mode_score=%.2f mom_score=%.2f rev_score=%.2f side_bias=%.2f drift=%.2fp mtf=%s/%.2f/%.2f mtf_mult=%.2f hz=%s/%.2f/%.2f dir_bias=%s dir_score=%.2f dir_mult=%.2f look_mult=%.2f look_edge=%.3f tp_mult=%.2f tp_avg=%.0fs ext_mult=%.2f ext_reason=%s tech(tp/sl/hold/loss)=%.2f/%.2f/%.2f/%.2f route=%s hold_cap=%.0fs hard_loss=%.2fp res=%s",
                config.LOG_PREFIX,
                signal.mode,
                signal.side,
                units,
                signal.confidence,
                raw_entry_probability,
                entry_probability,
                probability_units_mult,
                probability_band_units_mult,
                signal.momentum_pips,
                signal.trigger_pips,
                signal.spread_pips,
                tp_pips,
                sl_pips,
                signal.mode_score,
                signal.momentum_score,
                signal.revert_score,
                bias_scale,
                _safe_float(bias_meta.get("drift_pips"), 0.0),
                (regime.mode if regime is not None else "none"),
                (_safe_float(regime.trend_score, 0.0) if regime is not None else 0.0),
                (_safe_float(regime.heat_score, 0.0) if regime is not None else 0.0),
                regime_units_mult,
                (horizon.composite_side if horizon is not None else "none"),
                (_safe_float(horizon.composite_score, 0.0) if horizon is not None else 0.0),
                horizon_units_mult,
                direction_bias.side if direction_bias is not None else "none",
                _safe_float(getattr(direction_bias, "score", 0.0), 0.0),
                bias_units_mult,
                lookahead_units_mult,
                (
                    _safe_float(getattr(lookahead_decision, "edge_pips", 0.0), 0.0)
                    if lookahead_decision is not None
                    else 0.0
                ),
                tp_profile.multiplier,
                tp_profile.avg_tp_sec,
                extrema_units_mult,
                extrema_decision.reason,
                tech_profile.tp_mult,
                tech_profile.sl_mult,
                tech_profile.hold_mult,
                tech_profile.hard_loss_mult,
                ",".join(tech_route_reasons) if tech_route_reasons else "-",
                dynamic_max_hold_sec,
                dynamic_hard_loss_pips,
                result or "none",
            )

    except asyncio.CancelledError:
        LOG.info("%s cancelled", config.LOG_PREFIX)
        raise
    finally:
        pos_manager.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    asyncio.run(scalp_ping_5s_worker())
