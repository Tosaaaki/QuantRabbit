"""Entry-only 5s ping scalp worker.

This worker focuses on short-horizon entries and leaves exits to broker TP/SL
(or separate exit workers) by design.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
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
from workers.common.exit_utils import close_trade as close_open_trade
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


@dataclass(slots=True)
class ForceExitState:
    peak_pips: float = float("-inf")


@dataclass(slots=True)
class ForceExitDecision:
    trade_id: str
    close_units: int
    client_order_id: str
    reason: str
    allow_negative: bool
    pnl_pips: float
    hold_sec: float
    priority: int


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


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

    side: Optional[str] = None
    if (
        momentum_pips >= trigger_pips
        and imbalance >= config.IMBALANCE_MIN
        and tick_rate >= config.MIN_TICK_RATE
    ):
        side = "long"
    elif (
        momentum_pips <= -trigger_pips
        and imbalance >= config.IMBALANCE_MIN
        and tick_rate >= config.MIN_TICK_RATE
    ):
        side = "short"

    if side is None:
        return None

    strength = abs(momentum_pips) / max(0.01, trigger_pips)
    confidence = config.CONFIDENCE_FLOOR
    confidence += int(min(22.0, strength * 8.0))
    confidence += int(min(10.0, max(0.0, (imbalance - 0.5) * 25.0)))
    confidence += int(min(8.0, tick_rate))
    confidence = max(config.CONFIDENCE_FLOOR, min(config.CONFIDENCE_CEIL, confidence))

    return TickSignal(
        side=side,
        confidence=int(confidence),
        momentum_pips=momentum_pips,
        trigger_pips=trigger_pips,
        imbalance=imbalance,
        tick_rate=tick_rate,
        span_sec=span_sec,
        tick_age_ms=tick_age_ms,
        spread_pips=max(0.0, spread_pips),
        bid=bid,
        ask=ask,
        mid=mid,
    )


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


def _direction_units_multiplier(signal_side: str, bias: Optional[DirectionBias]) -> tuple[float, str]:
    if not config.DIRECTION_BIAS_ENABLED or bias is None:
        return 1.0, "disabled"
    if bias.side == "neutral":
        return 1.0, "neutral"

    score_abs = abs(_safe_float(bias.score, 0.0))
    if signal_side != bias.side:
        if score_abs >= config.DIRECTION_BIAS_BLOCK_SCORE:
            return 0.0, "opposite_block"
        return config.DIRECTION_BIAS_OPPOSITE_UNITS_MULT, "opposite_scale"

    if score_abs < config.DIRECTION_BIAS_ALIGN_SCORE_MIN:
        return 1.0, "align_weak"

    span = max(0.01, 1.0 - config.DIRECTION_BIAS_ALIGN_SCORE_MIN)
    ratio = _clamp((score_abs - config.DIRECTION_BIAS_ALIGN_SCORE_MIN) / span, 0.0, 1.0)
    boost = config.DIRECTION_BIAS_ALIGN_UNITS_BOOST_MAX * ratio
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


def _compute_targets(*, spread_pips: float, momentum_pips: float) -> tuple[float, float]:
    # Fast scalp default: set TP from current spread + tiny net edge.
    tp_floor = spread_pips + config.TP_NET_MIN_PIPS
    tp_pips = max(config.TP_BASE_PIPS, tp_floor)
    tp_bonus_raw = max(0.0, abs(momentum_pips) - tp_pips)
    tp_pips += min(config.TP_MOMENTUM_BONUS_MAX, tp_bonus_raw * 0.25)
    tp_pips = min(config.TP_MAX_PIPS, tp_pips)

    sl_floor = spread_pips * config.SL_SPREAD_MULT + config.SL_SPREAD_BUFFER_PIPS
    sl_pips = max(config.SL_MIN_PIPS, config.SL_BASE_PIPS, sl_floor)
    if abs(momentum_pips) > tp_pips * 1.4:
        sl_pips *= 1.08
    sl_pips = max(config.SL_MIN_PIPS, min(config.SL_MAX_PIPS, sl_pips))

    return tp_pips, sl_pips


def _strategy_open_trades(pocket_info: dict, strategy_tag: str) -> list[dict]:
    open_trades = pocket_info.get("open_trades") if isinstance(pocket_info, dict) else None
    if not isinstance(open_trades, list):
        return []

    tag_key = (strategy_tag or "").strip().lower()
    picked: list[dict] = []

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
        picked.append(tr)

    return picked


def _strategy_trade_counts(pocket_info: dict, strategy_tag: str) -> tuple[int, int, int]:
    total = 0
    long_count = 0
    short_count = 0
    for tr in _strategy_open_trades(pocket_info, strategy_tag):
        units = int(_safe_float(tr.get("units"), 0.0))
        if units == 0:
            continue
        total += 1
        if units > 0:
            long_count += 1
        else:
            short_count += 1
    return total, long_count, short_count


def _trade_id(trade: dict) -> str:
    for key in ("trade_id", "id", "ticket_id"):
        value = trade.get(key)
        if value is None:
            continue
        tid = str(value).strip()
        if tid:
            return tid
    return ""


def _trade_client_order_id(trade: dict) -> str:
    client_id = str(trade.get("client_order_id") or "").strip()
    if client_id:
        return client_id
    ext = trade.get("clientExtensions")
    if isinstance(ext, dict):
        client_id = str(ext.get("id") or "").strip()
    return client_id


def _trade_policy_generation(trade: dict) -> str:
    thesis = trade.get("entry_thesis")
    if not isinstance(thesis, dict):
        return ""
    value = thesis.get("policy_generation") or thesis.get("force_exit_policy_generation")
    return str(value or "").strip()


def _trade_open_time(trade: dict) -> Optional[datetime.datetime]:
    raw = trade.get("open_time") or trade.get("entry_time")
    if isinstance(raw, datetime.datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=datetime.timezone.utc)
        return raw.astimezone(datetime.timezone.utc)
    if not raw:
        return None
    try:
        parsed = _parse_time(str(raw))
        if parsed is not None and parsed.tzinfo is None:
            return parsed.replace(tzinfo=datetime.timezone.utc)
        return parsed
    except Exception:
        return None


def _estimate_trade_pnl_pips(
    *,
    entry_price: float,
    units: int,
    bid: float,
    ask: float,
    mid: float,
) -> Optional[float]:
    if entry_price <= 0.0 or units == 0:
        return None
    if units > 0:
        px = bid if bid > 0.0 else (mid if mid > 0.0 else 0.0)
        if px <= 0.0:
            return None
        return (px - entry_price) / config.PIP_VALUE
    px = ask if ask > 0.0 else (mid if mid > 0.0 else 0.0)
    if px <= 0.0:
        return None
    return (entry_price - px) / config.PIP_VALUE


def _pick_force_exit_reason(
    *,
    hold_sec: float,
    pnl_pips: float,
    peak_pips: float,
) -> tuple[Optional[str], bool, int]:
    if config.FORCE_EXIT_MAX_FLOATING_LOSS_PIPS > 0.0 and (
        pnl_pips <= -config.FORCE_EXIT_MAX_FLOATING_LOSS_PIPS
    ):
        return config.FORCE_EXIT_REASON_MAX_FLOATING_LOSS, True, 0

    if (
        config.FORCE_EXIT_GIVEBACK_ENABLED
        and hold_sec >= config.FORCE_EXIT_GIVEBACK_MIN_HOLD_SEC
        and peak_pips >= config.FORCE_EXIT_GIVEBACK_ARM_PIPS
        and (peak_pips - pnl_pips) >= config.FORCE_EXIT_GIVEBACK_BACKOFF_PIPS
        and pnl_pips <= config.FORCE_EXIT_GIVEBACK_PROTECT_PIPS
    ):
        return config.FORCE_EXIT_REASON_GIVEBACK, pnl_pips <= 0.0, 1

    if (
        config.FORCE_EXIT_RECOVERY_WINDOW_SEC > 0.0
        and hold_sec >= config.FORCE_EXIT_RECOVERY_WINDOW_SEC
        and peak_pips < config.FORCE_EXIT_RECOVERABLE_LOSS_PIPS
        and pnl_pips <= 0.0
    ):
        return config.FORCE_EXIT_REASON_RECOVERY, True, 2

    if config.FORCE_EXIT_MAX_HOLD_SEC > 0.0 and hold_sec >= config.FORCE_EXIT_MAX_HOLD_SEC:
        return config.FORCE_EXIT_REASON_TIME, pnl_pips <= 0.0, 3

    return None, False, 99


def _collect_force_exit_decisions(
    *,
    trades: Sequence[dict],
    now_utc: datetime.datetime,
    bid: float,
    ask: float,
    mid: float,
    states: dict[str, ForceExitState],
    protected_trade_ids: set[str],
) -> list[ForceExitDecision]:
    decisions: list[ForceExitDecision] = []
    if not config.FORCE_EXIT_ACTIVE:
        return decisions

    expected_generation = config.FORCE_EXIT_POLICY_GENERATION
    for trade in trades:
        trade_id = _trade_id(trade)
        if not trade_id or trade_id in protected_trade_ids:
            continue

        if config.FORCE_EXIT_REQUIRE_POLICY_GENERATION:
            generation = _trade_policy_generation(trade)
            if not expected_generation or generation != expected_generation:
                continue

        client_order_id = _trade_client_order_id(trade)
        if not client_order_id:
            continue

        units = int(_safe_float(trade.get("units"), 0.0))
        if units == 0:
            continue
        entry_price = _safe_float(trade.get("price"), 0.0)
        if entry_price <= 0.0:
            entry_price = _safe_float(trade.get("entry_price"), 0.0)
        pnl_pips = _estimate_trade_pnl_pips(
            entry_price=entry_price,
            units=units,
            bid=bid,
            ask=ask,
            mid=mid,
        )
        if pnl_pips is None:
            continue

        state = states.get(trade_id)
        if state is None:
            state = ForceExitState(peak_pips=pnl_pips)
            states[trade_id] = state
        elif pnl_pips > state.peak_pips:
            state.peak_pips = pnl_pips

        open_time = _trade_open_time(trade)
        hold_sec = (
            max(0.0, (now_utc - open_time).total_seconds()) if open_time is not None else 0.0
        )
        reason, allow_negative, priority = _pick_force_exit_reason(
            hold_sec=hold_sec,
            pnl_pips=pnl_pips,
            peak_pips=state.peak_pips,
        )
        if not reason:
            continue
        decisions.append(
            ForceExitDecision(
                trade_id=trade_id,
                close_units=-units,
                client_order_id=client_order_id,
                reason=reason,
                allow_negative=allow_negative,
                pnl_pips=pnl_pips,
                hold_sec=hold_sec,
                priority=priority,
            )
        )

    decisions.sort(key=lambda x: (x.priority, x.pnl_pips))
    if config.FORCE_EXIT_MAX_ACTIONS > 0:
        decisions = decisions[: config.FORCE_EXIT_MAX_ACTIONS]
    return decisions


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
        ts = _parse_time(price.get("time", datetime.datetime.utcnow().isoformat() + "Z"))
    except Exception as exc:
        logger.warning("%s snapshot parse failed: %s", config.LOG_PREFIX, exc)
        return False

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
    last_bias_log_mono = 0.0
    force_exit_states: dict[str, ForceExitState] = {}
    protected_trade_ids: set[str] = set()
    protected_seeded = False

    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_utc = datetime.datetime.utcnow()
            if not is_market_open(now_utc):
                continue
            if not can_trade(config.POCKET):
                continue

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

            positions = pos_manager.get_open_positions()
            pocket_info = positions.get(config.POCKET) or {}
            strategy_trades = _strategy_open_trades(pocket_info, config.STRATEGY_TAG)

            active_trade_ids = {_trade_id(tr) for tr in strategy_trades if _trade_id(tr)}
            for tid in list(force_exit_states.keys()):
                if tid not in active_trade_ids:
                    force_exit_states.pop(tid, None)

            if not protected_seeded:
                protected_seeded = True
                if config.FORCE_EXIT_SKIP_EXISTING_ON_START:
                    protected_trade_ids = {
                        _trade_id(tr)
                        for tr in strategy_trades
                        if _trade_id(tr)
                    }
                    if protected_trade_ids:
                        LOG.info(
                            "%s force_exit protect_existing=%d trade(s)",
                            config.LOG_PREFIX,
                            len(protected_trade_ids),
                        )

            if strategy_trades and config.FORCE_EXIT_ACTIVE:
                bid, ask, mid = _quotes_from_row(ticks[-1], fallback_mid=0.0)
                decisions = _collect_force_exit_decisions(
                    trades=strategy_trades,
                    now_utc=datetime.datetime.now(datetime.timezone.utc),
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    states=force_exit_states,
                    protected_trade_ids=protected_trade_ids,
                )
                for d in decisions:
                    ok = await close_open_trade(
                        d.trade_id,
                        d.close_units,
                        client_order_id=d.client_order_id,
                        allow_negative=d.allow_negative,
                        exit_reason=d.reason,
                        env_prefix=config.ENV_PREFIX,
                    )
                    if ok:
                        LOG.info(
                            "%s force_exit close_ok trade=%s reason=%s pnl=%.2fp hold=%.0fs",
                            config.LOG_PREFIX,
                            d.trade_id,
                            d.reason,
                            d.pnl_pips,
                            d.hold_sec,
                        )
                        force_exit_states.pop(d.trade_id, None)
                    else:
                        LOG.warning(
                            "%s force_exit close_failed trade=%s reason=%s pnl=%.2fp hold=%.0fs",
                            config.LOG_PREFIX,
                            d.trade_id,
                            d.reason,
                            d.pnl_pips,
                            d.hold_sec,
                        )

            signal = _build_tick_signal(ticks, spread_pips)
            if signal is None:
                continue
            if signal.spread_pips > config.MAX_SPREAD_PIPS:
                continue

            now_mono = time.monotonic()
            direction_bias = _build_direction_bias(ticks, spread_pips=signal.spread_pips)
            bias_units_mult, bias_gate = _direction_units_multiplier(signal.side, direction_bias)
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

            active_total = 0
            active_long = 0
            active_short = 0
            for tr in strategy_trades:
                units = int(_safe_float(tr.get("units"), 0.0))
                if units == 0:
                    continue
                active_total += 1
                if units > 0:
                    active_long += 1
                else:
                    active_short += 1
            if active_total >= config.MAX_ACTIVE_TRADES:
                continue
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

            tp_pips, sl_pips = _compute_targets(
                spread_pips=signal.spread_pips,
                momentum_pips=signal.momentum_pips,
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
                "entry_mode": "market_ping_5s",
                "trap_active": bool(trap_state.active),
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
                "%s open side=%s units=%s conf=%d mom=%.2fp trig=%.2fp sp=%.2fp tp=%.2fp sl=%.2fp bias=%s score=%.2f mult=%.2f res=%s",
                config.LOG_PREFIX,
                signal.side,
                units,
                signal.confidence,
                signal.momentum_pips,
                signal.trigger_pips,
                signal.spread_pips,
                tp_pips,
                sl_pips,
                direction_bias.side if direction_bias is not None else "none",
                _safe_float(getattr(direction_bias, "score", 0.0), 0.0),
                bias_units_mult,
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
