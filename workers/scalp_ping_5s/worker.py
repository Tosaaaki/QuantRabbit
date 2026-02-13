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

from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from indicators.factor_cache import all_factors, get_candles_snapshot
from market_data import spread_monitor, tick_window
from market_data.tick_fetcher import _parse_time
from utils.market_hours import is_market_open
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


def _mask_value(value: str, *, head: int = 4, tail: int = 2) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if len(text) <= head + tail:
        return "*" * len(text)
    return f"{text[:head]}***{text[-tail:]}"


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


def _norm01(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low), 0.0, 1.0)


def _tf_trend_score(fac: dict) -> float:
    close = _safe_float(fac.get("close"), 0.0)
    ema20 = _safe_float(fac.get("ema20"), 0.0)
    atr_pips = max(0.1, _safe_float(fac.get("atr_pips"), 0.0))
    rsi = _safe_float(fac.get("rsi"), 50.0)
    macd_hist = _safe_float(fac.get("macd_hist"), 0.0)

    gap_norm = 0.0
    if close > 0.0 and ema20 > 0.0:
        gap_pips = (close - ema20) / config.PIP_VALUE
        gap_norm = _clamp(gap_pips / max(0.8, atr_pips * 0.65), -1.0, 1.0)

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
) -> tuple[float, float]:
    hold_sec = 0.0
    hard_loss_pips = 0.0
    if base_max_hold_sec > 0.0:
        hold_cap = max(
            config.TECH_ROUTER_HOLD_MIN_SEC,
            base_max_hold_sec * max(0.5, float(config.TECH_ROUTER_HOLD_MAX_MULT)),
        )
        hold_sec = _clamp(
            base_max_hold_sec * max(0.2, profile.hold_mult),
            config.TECH_ROUTER_HOLD_MIN_SEC,
            hold_cap,
        )
    if base_hard_loss_pips > 0.0:
        hard_loss_pips = max(0.1, base_hard_loss_pips * max(0.2, profile.hard_loss_mult))
    return float(hold_sec), float(hard_loss_pips)


def _force_exit_hard_loss_trigger_pips(trade: dict, base_hard_loss_pips: float) -> float:
    """Return effective hard-loss trigger pips for force-exit decisions.

    `unrealized_pl_pips` includes spread costs; cutting exactly at `-base_hard_loss_pips`
    often creates churn. Add entry spread (if available) plus a small buffer to avoid
    "meaningless" market-close exits around the threshold.
    """

    if base_hard_loss_pips <= 0.0:
        return 0.0
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
        if hard_loss_trigger_pips > 0.0 and unrealized_pips <= -hard_loss_trigger_pips:
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


def _build_tick_signal(rows: Sequence[dict], spread_pips: float) -> Optional[TickSignal]:
    if len(rows) < config.MIN_TICKS:
        return None

    latest_epoch = _safe_float(rows[-1].get("epoch"), 0.0)
    if latest_epoch <= 0.0:
        return None

    tick_age_ms = max(0.0, (time.time() - latest_epoch) * 1000.0)
    if tick_age_ms > config.MAX_TICK_AGE_MS:
        return None

    window_cutoff = latest_epoch - config.SIGNAL_WINDOW_SEC
    signal_rows = [r for r in rows if _safe_float(r.get("epoch"), 0.0) >= window_cutoff]
    if len(signal_rows) < config.MIN_SIGNAL_TICKS:
        return None

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

    if len(mids) < config.MIN_SIGNAL_TICKS:
        return None

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

    edge_guard_pips = max(
        config.MOMENTUM_TRIGGER_PIPS,
        spread_pips * config.MOMENTUM_SPREAD_MULT + config.ENTRY_BID_ASK_EDGE_PIPS,
    )
    trigger_pips = edge_guard_pips

    side_momentum: Optional[str] = None
    momentum_score = 0.0
    if (
        momentum_pips >= trigger_pips
        and imbalance >= config.IMBALANCE_MIN
        and tick_rate >= config.MIN_TICK_RATE
    ):
        side_momentum = "long"
    elif (
        momentum_pips <= -trigger_pips
        and imbalance >= config.IMBALANCE_MIN
        and tick_rate >= config.MIN_TICK_RATE
    ):
        side_momentum = "short"

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
    if config.REVERT_ENABLED:
        revert_cutoff = latest_epoch - config.REVERT_WINDOW_SEC
        revert_rows = [r for r in rows if _safe_float(r.get("epoch"), 0.0) >= revert_cutoff]
        if len(revert_rows) >= config.REVERT_MIN_TICKS:
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

            if len(mids_revert) >= config.REVERT_MIN_TICKS:
                span_revert_sec = max(0.001, epochs_revert[-1] - epochs_revert[0])
                tick_rate_revert = max(0.0, (len(mids_revert) - 1) / span_revert_sec)
                range_revert_pips = (max(mids_revert) - min(mids_revert)) / config.PIP_VALUE
                if (
                    range_revert_pips >= config.REVERT_RANGE_MIN_PIPS
                    and tick_rate_revert >= config.REVERT_MIN_TICK_RATE
                ):
                    short_cutoff = latest_epoch - config.REVERT_SHORT_WINDOW_SEC
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
                            lookback=config.REVERT_CONFIRM_TICKS,
                        )
                        up_flow_ratio = _tail_flow_ratio(
                            mids_revert,
                            want_up=True,
                            lookback=config.REVERT_CONFIRM_TICKS,
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
                            else:
                                side_revert = "short"
                                revert_score = float(short_score)
                            revert_momentum_pips = short_leg_pips
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
        return None

    selected_strength = abs(selected_momentum_pips) / max(0.01, selected_trigger_pips)
    confidence = config.CONFIDENCE_FLOOR + 2 if mode == "revert" else config.CONFIDENCE_FLOOR
    confidence += int(min(22.0, selected_strength * 8.0))
    confidence += int(min(10.0, max(0.0, (imbalance - 0.5) * 25.0)))
    confidence += int(min(8.0, tick_rate))
    confidence += int(min(6.0, mode_score * 1.2))
    confidence = max(config.CONFIDENCE_FLOOR, min(config.CONFIDENCE_CEIL, confidence))

    return TickSignal(
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
        tick_age_ms=tick_age_ms,
        spread_pips=max(0.0, spread_pips),
        bid=bid,
        ask=ask,
        mid=mid,
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
        tick_age_ms=float(signal.tick_age_ms),
        spread_pips=float(signal.spread_pips),
        bid=float(signal.bid),
        ask=float(signal.ask),
        mid=float(signal.mid),
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
                return None, 0.0, "mtf_continuation_block_counter"
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
            return None, 0.0, "horizon_block_counter"
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
) -> tuple[float, str]:
    if not config.M1_TREND_SCALE_ENABLED:
        return 1.0, "disabled"
    if m1_score is None:
        return 1.0, "m1_unavailable"
    if signal_side not in {"long", "short"}:
        return 1.0, "invalid_side"

    score_abs = abs(_safe_float(m1_score, 0.0))
    if score_abs <= 0.0:
        return 1.0, "m1_zero"

    side_sign = 1.0 if signal_side == "long" else -1.0
    aligned = score_abs > 0.0 and (m1_score * side_sign) > 0.0
    if not aligned:
        denom = max(0.05, float(config.M1_TREND_OPPOSITE_SCORE))
        penalty_ratio = _clamp(score_abs / denom, 0.0, 1.0)
        scale = 1.0 - (1.0 - config.M1_TREND_OPPOSITE_UNITS_MULT) * penalty_ratio
        return float(max(0.1, scale)), "m1_opposite"

    if score_abs < config.M1_TREND_ALIGN_SCORE_MIN:
        return 1.0, "m1_align_weak"

    ratio = _clamp(
        (score_abs - config.M1_TREND_ALIGN_SCORE_MIN)
        / max(0.05, 1.0 - config.M1_TREND_ALIGN_SCORE_MIN),
        0.0,
        1.0,
    )
    return 1.0 + (config.M1_TREND_ALIGN_BOOST_MAX * ratio), "m1_align_boost"


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
    if signal_side != bias.side:
        if signal_mode == "revert":
            if score_abs >= config.REVERT_DIRECTION_HARD_BLOCK_SCORE:
                return 0.0, "revert_opposite_block"
            return config.REVERT_DIRECTION_OPPOSITE_UNITS_MULT, "revert_opposite_scale"
        if score_abs >= config.DIRECTION_BIAS_BLOCK_SCORE:
            return 0.0, "opposite_block"
        return config.DIRECTION_BIAS_OPPOSITE_UNITS_MULT, "opposite_scale"

    if score_abs < config.DIRECTION_BIAS_ALIGN_SCORE_MIN:
        return 1.0, "align_weak"

    span = max(0.01, 1.0 - config.DIRECTION_BIAS_ALIGN_SCORE_MIN)
    ratio = _clamp((score_abs - config.DIRECTION_BIAS_ALIGN_SCORE_MIN) / span, 0.0, 1.0)
    boost = config.DIRECTION_BIAS_ALIGN_UNITS_BOOST_MAX * ratio
    if signal_mode == "revert":
        boost *= 0.35
    return 1.0 + boost, "align_boost"


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


def _extrema_gate_decision(side: str) -> ExtremaGateDecision:
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
        if config.EXTREMA_REQUIRE_M1_M5_AGREE:
            top_block = m1_top and m5_top
        else:
            top_block = m1_top or m5_top
        if top_block:
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
        if config.EXTREMA_REQUIRE_M1_M5_AGREE:
            bottom_block = m1_bottom and m5_bottom
        else:
            bottom_block = m1_bottom or m5_bottom
        if bottom_block:
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


def _confidence_scale(conf: int) -> float:
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    if conf <= lo:
        return 0.65
    if conf >= hi:
        return 1.15
    ratio = (conf - lo) / max(1.0, hi - lo)
    return 0.65 + ratio * 0.5


def _compute_targets(
    *, spread_pips: float, momentum_pips: float, tp_profile: TpTimingProfile
) -> tuple[float, float]:
    # Time-aware TP scaling for ping scalp:
    # if average TP holding time drifts too long, reduce TP net edge.
    tp_mult = max(config.TP_TIME_MULT_MIN, min(config.TP_TIME_MULT_MAX, tp_profile.multiplier))
    tp_base = max(0.1, config.TP_BASE_PIPS * tp_mult)
    tp_net_edge = max(config.TP_NET_MIN_FLOOR_PIPS, config.TP_NET_MIN_PIPS * tp_mult)
    tp_floor = spread_pips + tp_net_edge
    tp_pips = max(tp_base, tp_floor)
    tp_bonus_raw = max(0.0, abs(momentum_pips) - tp_pips)
    tp_bonus_cap = config.TP_MOMENTUM_BONUS_MAX * max(0.6, tp_mult)
    tp_pips += min(tp_bonus_cap, tp_bonus_raw * 0.25)
    tp_pips = min(config.TP_MAX_PIPS, tp_pips)

    sl_floor = spread_pips * config.SL_SPREAD_MULT + config.SL_SPREAD_BUFFER_PIPS
    sl_pips = max(config.SL_MIN_PIPS, config.SL_BASE_PIPS, sl_floor)
    if abs(momentum_pips) > tp_pips * 1.4:
        sl_pips *= 1.08
    sl_pips = max(config.SL_MIN_PIPS, min(config.SL_MAX_PIPS, sl_pips))

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


def _allow_low_margin_hedge_relief(
    *,
    side: str,
    long_units: float,
    short_units: float,
    free_ratio: float,
    margin_available: float,
) -> bool:
    if not config.LOW_MARGIN_HEDGE_RELIEF_ENABLED:
        return False
    if free_ratio < config.LOW_MARGIN_HEDGE_RELIEF_MIN_FREE_RATIO:
        return False
    if margin_available < config.LOW_MARGIN_HEDGE_RELIEF_MIN_MARGIN_AVAILABLE_JPY:
        return False
    if side == "short":
        return long_units > short_units
    if side == "long":
        return short_units > long_units
    return False


def _cap_low_margin_hedge_units(
    *,
    side: str,
    units: int,
    long_units: float,
    short_units: float,
) -> int:
    if side == "short":
        imbalance_units = max(0.0, long_units - short_units)
    elif side == "long":
        imbalance_units = max(0.0, short_units - long_units)
    else:
        imbalance_units = 0.0

    if imbalance_units <= 0.0:
        return 0
    max_relief_units = int(
        max(
            float(config.MIN_UNITS),
            imbalance_units * config.LOW_MARGIN_HEDGE_RELIEF_MAX_IMBALANCE_FRACTION,
        )
    )
    abs_units = min(abs(int(units)), max_relief_units)
    if abs_units < config.MIN_UNITS:
        return 0
    return -abs_units if side == "short" else abs_units


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
    digest = hashlib.sha1(f"{config.STRATEGY_TAG}-{side}-{ts_ms}".encode("utf-8")).hexdigest()[:8]
    return f"qr-{ts_ms}-{config.STRATEGY_TAG[:12]}-{side[0]}{digest}"


async def _fetch_price_snapshot(logger: logging.Logger) -> bool:
    if not _OANDA_TOKEN or not _OANDA_ACCOUNT:
        return False

    params = {"instruments": "USD_JPY"}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(_PRICING_URL, headers=_PRICING_HEADERS, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.warning("%s snapshot fetch failed: %s", config.LOG_PREFIX, exc)
        return False

    prices = payload.get("prices") or []
    if not prices:
        return False

    price = prices[0]
    bids = price.get("bids") or []
    asks = price.get("asks") or []
    if not bids or not asks:
        return False

    try:
        bid = float(bids[0]["price"])
        ask = float(asks[0]["price"])
        quote_ts = _parse_time(price.get("time", datetime.datetime.utcnow().isoformat() + "Z"))
    except Exception as exc:
        logger.warning("%s snapshot parse failed: %s", config.LOG_PREFIX, exc)
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
        return False

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

    pos_manager = PositionManager()
    rate_limiter = SlidingWindowRateLimiter(
        config.MAX_ORDERS_PER_MINUTE,
        config.MIN_ORDER_SPACING_SEC,
    )
    last_entry_mono = 0.0
    last_snapshot_fetch = 0.0
    last_stale_log_mono = 0.0
    last_trap_log_mono = 0.0
    last_density_topup_log_mono = 0.0
    last_keepalive_log_mono = 0.0
    last_max_active_bypass_log_mono = 0.0
    last_margin_guard_bypass_log_mono = 0.0
    last_bias_log_mono = 0.0
    last_lookahead_log_mono = 0.0
    last_regime_log_mono = 0.0
    last_horizon_log_mono = 0.0
    last_profit_bank_log_mono = 0.0
    last_extrema_log_mono = 0.0
    protected_trade_ids: set[str] = set()
    protected_seeded = False
    strategy_tag_key = str(config.STRATEGY_TAG or "").strip().lower()

    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            if not is_market_open(now_utc):
                continue
            if not can_trade(config.POCKET):
                continue

            positions = pos_manager.get_open_positions()
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
                            trade_max_hold_sec = _trade_force_exit_max_hold_sec(trade, max_hold_sec)
                            trade_hard_loss_pips = _trade_force_exit_hard_loss_pips(trade, hard_loss_pips)
                            hard_loss_trigger_pips = _force_exit_hard_loss_trigger_pips(
                                trade,
                                trade_hard_loss_pips,
                            )
                            eligible = False
                            if trade_max_hold_sec > 0.0 and hold_sec >= trade_max_hold_sec:
                                eligible = True
                            elif hard_loss_trigger_pips > 0.0 and unrealized_pips <= -hard_loss_trigger_pips:
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
                        trade_max_hold_sec = _trade_force_exit_max_hold_sec(trade, max_hold_sec)
                        trade_hard_loss_pips = _trade_force_exit_hard_loss_pips(trade, hard_loss_pips)
                        hard_loss_trigger_pips = _force_exit_hard_loss_trigger_pips(
                            trade,
                            trade_hard_loss_pips,
                        )
                        eligible = False
                        if trade_max_hold_sec > 0.0 and hold_sec >= trade_max_hold_sec:
                            eligible = True
                        elif hard_loss_trigger_pips > 0.0 and unrealized_pips <= -hard_loss_trigger_pips:
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
                positions = pos_manager.get_open_positions()
                pocket_info = positions.get(config.POCKET) or {}
            profit_bank_closed = await _apply_profit_bank_release(
                pocket_info=pocket_info,
                now_utc=now_utc,
                logger=LOG,
                protected_trade_ids=protected_trade_ids,
            )
            if profit_bank_closed > 0:
                positions = pos_manager.get_open_positions()
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

            needs_snapshot = (len(ticks) < config.MIN_TICKS) or (
                latest_tick_age_ms > config.MAX_TICK_AGE_MS
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
                continue
            if latest_tick_age_ms > config.MAX_TICK_AGE_MS:
                now_mono = time.monotonic()
                if now_mono - last_stale_log_mono >= 10.0:
                    LOG.warning(
                        "%s stale ticks age=%.0fms (> %.0fms) waiting for fresh data",
                        config.LOG_PREFIX,
                        latest_tick_age_ms,
                        config.MAX_TICK_AGE_MS,
                    )
                    last_stale_log_mono = now_mono
                continue

            signal = _build_tick_signal(ticks, spread_pips)
            if signal is None:
                continue
            if signal.spread_pips > config.MAX_SPREAD_PIPS:
                continue

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
                    continue

            lookahead_decision = None
            lookahead_units_mult = 1.0
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
                    horizon_sec=config.LOOKAHEAD_HORIZON_SEC,
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
                        continue
                else:
                    lookahead_units_mult = max(0.1, float(lookahead_decision.units_mult))
            extrema_decision = _extrema_gate_decision(signal.side)
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
            if now_mono - last_entry_mono < config.ENTRY_COOLDOWN_SEC:
                continue
            if not rate_limiter.allow(now_mono):
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
                    continue
                if signal.side == "short" and long_units > 0.0:
                    continue

            snap = get_account_snapshot(cache_ttl_sec=0.5)
            nav = max(_safe_float(snap.nav, 0.0), 1.0)
            balance = max(_safe_float(snap.balance, 0.0), nav)
            margin_available = max(_safe_float(snap.margin_available, 0.0), 0.0)
            margin_rate = _safe_float(snap.margin_rate, 0.0)
            free_ratio = _safe_float(snap.free_margin_ratio, 0.0)

            low_margin_hedge_relief = False
            if free_ratio > 0.0 and free_ratio < config.MIN_FREE_MARGIN_RATIO:
                low_margin_hedge_relief = _allow_low_margin_hedge_relief(
                    side=signal.side,
                    long_units=account_long_units,
                    short_units=account_short_units,
                    free_ratio=free_ratio,
                    margin_available=margin_available,
                )
                if not low_margin_hedge_relief:
                    continue
                if (
                    now_mono - last_margin_guard_bypass_log_mono
                    >= config.LOW_MARGIN_HEDGE_RELIEF_LOG_INTERVAL_SEC
                ):
                    LOG.warning(
                        "%s margin_guard_hedge_relief side=%s free_ratio=%.3f(min=%.3f) margin_avail=%.0f long=%.0f short=%.0f",
                        config.LOG_PREFIX,
                        signal.side,
                        free_ratio,
                        config.MIN_FREE_MARGIN_RATIO,
                        margin_available,
                        long_units,
                        short_units,
                    )
                    last_margin_guard_bypass_log_mono = now_mono

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
                continue
            if signal.side == "short" and active_short >= max_per_direction:
                continue

            tp_profile = _load_tp_timing_profile(config.STRATEGY_TAG, config.POCKET)
            tp_pips, sl_pips = _compute_targets(
                spread_pips=signal.spread_pips,
                momentum_pips=signal.momentum_pips,
                tp_profile=tp_profile,
            )
            tech_profile = _build_technical_trade_profile(
                route_reasons=tech_route_reasons,
                lookahead_decision=lookahead_decision,
            )
            tp_pips *= max(0.2, tech_profile.tp_mult)
            tp_floor = max(config.TP_BASE_PIPS, signal.spread_pips + config.TP_NET_MIN_FLOOR_PIPS)
            tp_pips = _clamp(tp_pips, tp_floor, config.TP_MAX_PIPS)
            sl_floor = max(
                config.SL_MIN_PIPS,
                signal.spread_pips * config.SL_SPREAD_MULT + config.SL_SPREAD_BUFFER_PIPS,
            )
            sl_pips = _clamp(sl_pips * max(0.2, tech_profile.sl_mult), sl_floor, config.SL_MAX_PIPS)
            dynamic_max_hold_sec, dynamic_hard_loss_pips = _scaled_force_exit_thresholds(
                base_max_hold_sec=float(config.FORCE_EXIT_MAX_HOLD_SEC),
                base_hard_loss_pips=float(config.FORCE_EXIT_MAX_FLOATING_LOSS_PIPS),
                profile=tech_profile,
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

            lot = allowed_lot(
                nav,
                sl_pips,
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
                continue
            units = abs(units)
            if signal.side == "short":
                units = -units
            if low_margin_hedge_relief:
                units = _cap_low_margin_hedge_units(
                    side=signal.side,
                    units=units,
                    long_units=long_units,
                    short_units=short_units,
                )
                if units == 0:
                    continue

            entry_price = signal.ask if signal.side == "long" else signal.bid
            if entry_price <= 0.0:
                entry_price = signal.mid
            if entry_price <= 0.0:
                continue

            if signal.side == "long":
                sl_price = (
                    entry_price - sl_pips * config.PIP_VALUE if config.USE_SL else None
                )
                tp_price = entry_price + tp_pips * config.PIP_VALUE
            else:
                sl_price = (
                    entry_price + sl_pips * config.PIP_VALUE if config.USE_SL else None
                )
                tp_price = entry_price - tp_pips * config.PIP_VALUE

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
                "signal_window_sec": config.SIGNAL_WINDOW_SEC,
                "window_sec": config.WINDOW_SEC,
                "momentum_pips": round(signal.momentum_pips, 3),
                "trigger_pips": round(signal.trigger_pips, 3),
                "imbalance": round(signal.imbalance, 3),
                "tick_rate": round(signal.tick_rate, 3),
                "tick_age_ms": round(signal.tick_age_ms, 1),
                "spread_pips": round(signal.spread_pips, 3),
                "confidence": int(signal.confidence),
                "tp_pips": round(tp_pips, 3),
                "sl_pips": round(sl_pips, 3),
                "disable_entry_hard_stop": bool(config.DISABLE_ENTRY_HARD_STOP),
                "signal_mode": signal.mode,
                "signal_mode_score": round(signal.mode_score, 3),
                "signal_momentum_score": round(signal.momentum_score, 3),
                "signal_revert_score": round(signal.revert_score, 3),
                "tp_time_mult": round(tp_profile.multiplier, 3),
                "tp_time_avg_sec": (
                    round(tp_profile.avg_tp_sec, 1) if tp_profile.avg_tp_sec > 0.0 else None
                ),
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
                "lookahead_gate_enabled": bool(config.LOOKAHEAD_GATE_ENABLED),
                "lookahead_units_mult": round(lookahead_units_mult, 3),
                "extrema_gate_enabled": bool(config.EXTREMA_GATE_ENABLED),
                "extrema_gate_reason": extrema_decision.reason,
                "extrema_units_mult": round(extrema_units_mult, 3),
                "tech_router_enabled": bool(config.TECH_ROUTER_ENABLED),
                "tech_route_reasons": list(tech_route_reasons),
                "tech_counter_pressure": bool(tech_profile.counter_pressure),
                "tech_tp_mult": round(float(tech_profile.tp_mult), 3),
                "tech_sl_mult": round(float(tech_profile.sl_mult), 3),
                "tech_hold_mult": round(float(tech_profile.hold_mult), 3),
                "tech_hard_loss_mult": round(float(tech_profile.hard_loss_mult), 3),
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
            result = await market_order(
                instrument="USD_JPY",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                pocket=config.POCKET,
                client_order_id=client_order_id,
                strategy_tag=config.STRATEGY_TAG,
                confidence=int(signal.confidence),
                entry_thesis=entry_thesis,
            )
            if result:
                pos_manager.register_open_trade(str(result), config.POCKET, client_order_id)
            last_entry_mono = now_mono
            rate_limiter.record(now_mono)

            LOG.info(
                "%s open mode=%s side=%s units=%s conf=%d mom=%.2fp trig=%.2fp sp=%.2fp tp=%.2fp sl=%.2fp mode_score=%.2f mom_score=%.2f rev_score=%.2f side_bias=%.2f drift=%.2fp mtf=%s/%.2f/%.2f mtf_mult=%.2f hz=%s/%.2f/%.2f dir_bias=%s dir_score=%.2f dir_mult=%.2f look_mult=%.2f look_edge=%.3f tp_mult=%.2f tp_avg=%.0fs ext_mult=%.2f ext_reason=%s tech(tp/sl/hold/loss)=%.2f/%.2f/%.2f/%.2f route=%s hold_cap=%.0fs hard_loss=%.2fp res=%s",
                config.LOG_PREFIX,
                signal.mode,
                signal.side,
                units,
                signal.confidence,
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(scalp_ping_5s_worker())
