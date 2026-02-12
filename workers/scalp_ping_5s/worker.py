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
from market_data import spread_monitor, tick_window
from market_data.tick_fetcher import _parse_time
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from utils.secrets import get_secret
from workers.common.exit_utils import close_trade
from workers.common.rate_limiter import SlidingWindowRateLimiter
from workers.common.size_utils import scale_base_units

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
class TrapState:
    active: bool
    long_units: float
    short_units: float
    net_ratio: float
    long_dd_pips: float
    short_dd_pips: float
    combined_dd_pips: float
    unrealized_pl: float


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


def _parse_trade_open_time(raw: object) -> Optional[datetime.datetime]:
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


async def _enforce_new_entry_time_stop(
    *,
    pocket_info: dict,
    now_utc: datetime.datetime,
    logger: logging.Logger,
    protected_trade_ids: Optional[set[str]] = None,
) -> int:
    global _TRADE_MFE_PIPS

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
    closed_count = 0
    seen_trade_ids: set[str] = set()

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
        if config.FORCE_EXIT_REQUIRE_POLICY_GENERATION:
            if not generation:
                continue
            if target_generation and generation != target_generation:
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
        seen_trade_ids.add(trade_id)

        prev_mfe = _safe_float(_TRADE_MFE_PIPS.get(trade_id), unrealized_pips)
        mfe_pips = max(float(prev_mfe), float(unrealized_pips))
        _TRADE_MFE_PIPS[trade_id] = float(mfe_pips)

        exit_reason: Optional[str] = None
        trigger_label: Optional[str] = None
        if hard_loss_pips > 0.0 and unrealized_pips <= -hard_loss_pips:
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
        elif max_hold_sec > 0.0 and hold_sec >= max_hold_sec:
            exit_reason = config.FORCE_EXIT_REASON
            trigger_label = "time_stop"
        if not exit_reason:
            continue

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
        logger.info(
            "%s force_exit reason=%s trigger=%s trade=%s hold=%.0fs pnl=%.2fp mfe=%.2fp units=%s generation=%s",
            config.LOG_PREFIX,
            exit_reason,
            trigger_label or "-",
            trade_id,
            hold_sec,
            unrealized_pips,
            mfe_pips,
            units,
            generation or "-",
        )
        _TRADE_MFE_PIPS.pop(trade_id, None)

    stale_ids = [tid for tid in _TRADE_MFE_PIPS.keys() if tid not in seen_trade_ids]
    for tid in stale_ids:
        _TRADE_MFE_PIPS.pop(tid, None)

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

    trigger_pips = max(
        config.MOMENTUM_TRIGGER_PIPS,
        spread_pips * config.MOMENTUM_SPREAD_MULT,
    )

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
) -> bool:
    if active_total < config.MAX_ACTIVE_TRADES:
        return True
    if not config.ALLOW_OPPOSITE_WHEN_MAX_ACTIVE:
        return False
    if side == "long":
        return active_short > active_long
    if side == "short":
        return active_long > active_short
    return False


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
    last_bias_log_mono = 0.0
    protected_trade_ids: set[str] = set()
    protected_seeded = False

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
                    target_tag = str(config.STRATEGY_TAG or "").strip().lower()
                    open_trades = pocket_info.get("open_trades") if isinstance(pocket_info, dict) else None
                    if isinstance(open_trades, list):
                        for trade in open_trades:
                            if not isinstance(trade, dict):
                                continue
                            if target_tag:
                                trade_tag = _trade_strategy_tag(trade).lower()
                                if trade_tag != target_tag:
                                    continue
                            trade_id = str(trade.get("trade_id") or "").strip()
                            if trade_id:
                                protected_trade_ids.add(trade_id)
                    if protected_trade_ids:
                        LOG.info(
                            "%s force_exit protect_existing=%d trade(s)",
                            config.LOG_PREFIX,
                            len(protected_trade_ids),
                        )
            forced_closed = await _enforce_new_entry_time_stop(
                pocket_info=pocket_info,
                now_utc=now_utc,
                logger=LOG,
                protected_trade_ids=protected_trade_ids,
            )
            if forced_closed > 0:
                positions = pos_manager.get_open_positions()
                pocket_info = positions.get(config.POCKET) or {}

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
                        "%s bias block signal=%s bias=%s score=%.2f mom=%.2fp rng=%.2fp flow=%.2f",
                        config.LOG_PREFIX,
                        signal.side,
                        getattr(direction_bias, "side", "none"),
                        score,
                        _safe_float(getattr(direction_bias, "momentum_pips", 0.0), 0.0),
                        _safe_float(getattr(direction_bias, "range_pips", 0.0), 0.0),
                        _safe_float(getattr(direction_bias, "flow", 0.0), 0.0),
                    )
                    last_bias_log_mono = now_mono
                continue
            if now_mono - last_entry_mono < config.ENTRY_COOLDOWN_SEC:
                continue
            if not rate_limiter.allow(now_mono):
                continue

            active_total, active_long, active_short = _strategy_trade_counts(
                pocket_info,
                config.STRATEGY_TAG,
            )
            if not _allow_signal_when_max_active(
                side=signal.side,
                active_total=active_total,
                active_long=active_long,
                active_short=active_short,
            ):
                continue
            if (
                active_total >= config.MAX_ACTIVE_TRADES
                and now_mono - last_max_active_bypass_log_mono >= 5.0
            ):
                LOG.info(
                    "%s max_active bypass side=%s total=%d long=%d short=%d cap=%d",
                    config.LOG_PREFIX,
                    signal.side,
                    active_total,
                    active_long,
                    active_short,
                    config.MAX_ACTIVE_TRADES,
                )
                last_max_active_bypass_log_mono = now_mono
            if signal.side == "long" and active_long >= config.MAX_PER_DIRECTION:
                continue
            if signal.side == "short" and active_short >= config.MAX_PER_DIRECTION:
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

            if free_ratio > 0.0 and free_ratio < config.MIN_FREE_MARGIN_RATIO:
                continue

            tp_profile = _load_tp_timing_profile(config.STRATEGY_TAG, config.POCKET)
            tp_pips, sl_pips = _compute_targets(
                spread_pips=signal.spread_pips,
                momentum_pips=signal.momentum_pips,
                tp_profile=tp_profile,
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
            units = int(round(base_units * conf_mult * strength_mult * bias_units_mult))

            lot = allowed_lot(
                nav,
                sl_pips,
                margin_available=margin_available,
                margin_rate=margin_rate,
                price=signal.mid,
                pocket=config.POCKET,
                side=signal.side,
                open_long_units=long_units,
                open_short_units=short_units,
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

            entry_thesis = {
                "strategy_tag": config.STRATEGY_TAG,
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
                "trap_active": bool(trap_state.active),
                "entry_ref": round(entry_price, 3),
                "execution": {
                    "order_policy": "market_guarded",
                    "ideal_entry": round(entry_price, 3),
                    "chase_max": round(config.ENTRY_CHASE_MAX_PIPS * config.PIP_VALUE, 4),
                    "chase_max_pips": round(config.ENTRY_CHASE_MAX_PIPS, 3),
                },
                "direction_bias_gate": bias_gate,
                "direction_bias_units_mult": round(bias_units_mult, 3),
            }
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
                "%s open mode=%s side=%s units=%s conf=%d mom=%.2fp trig=%.2fp sp=%.2fp tp=%.2fp sl=%.2fp mode_score=%.2f mom_score=%.2f rev_score=%.2f side_bias=%.2f drift=%.2fp dir_bias=%s dir_score=%.2f dir_mult=%.2f tp_mult=%.2f tp_avg=%.0fs res=%s",
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
                direction_bias.side if direction_bias is not None else "none",
                _safe_float(getattr(direction_bias, "score", 0.0), 0.0),
                bias_units_mult,
                tp_profile.multiplier,
                tp_profile.avg_tp_sec,
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
