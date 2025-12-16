"""
Async worker that drives ultra-short-term scalping based on tick data.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import httpx

from execution.order_manager import close_trade, market_order, set_trade_protections
from execution.risk_guard import (
    allowed_lot,
    can_trade,
    clamp_sl_tp,
    loss_cooldown_status,
)
from execution.stage_tracker import StageTracker
from execution.position_manager import PositionManager
from market_data import spread_monitor, tick_window
from utils.metrics_logger import log_metric
from utils.secrets import get_secret
from workers.common.quality_gate import current_regime, news_block_active

from . import config
from .rate_limiter import SlidingWindowRateLimiter
from .patterns import pattern_score
from .signal import (
    SignalFeatures,
    evaluate_signal,
    extract_features,
    span_requirement_ok,
)
from .state import FastScalpState
from .timeout_controller import TimeoutController
from .profiles import (
    DEFAULT_PROFILE,
    StrategyProfile,
    current_session,
    get_profile,
    resolve_timeout,
    select_profile,
)


@dataclass
class ActiveTrade:
    trade_id: str
    side: str
    units: int
    entry_price: float
    opened_at: datetime
    opened_monotonic: float
    client_order_id: str
    sl_price: float
    tp_price: float
    profile: StrategyProfile = field(default=DEFAULT_PROFILE)
    timeout_limit: Optional[float] = field(default=DEFAULT_PROFILE.timeout_sec)
    timeout_min_gain: float = field(default=DEFAULT_PROFILE.timeout_min_gain_pips)
    max_drawdown_close_pips: float = field(default=DEFAULT_PROFILE.drawdown_close_pips)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime:
    iso = value.replace("Z", "+00:00")
    if "." not in iso:
        return datetime.fromisoformat(iso)
    head, frac_and_tz = iso.split(".", 1)
    tz = "+00:00"
    if "+" in frac_and_tz:
        frac, tz_tail = frac_and_tz.split("+", 1)
        tz = "+" + tz_tail
    elif "-" in frac_and_tz[6:]:
        frac, tz_tail = frac_and_tz.split("-", 1)
        tz = "-" + tz_tail
    else:
        frac = frac_and_tz
    frac = (''.join(ch for ch in frac if ch.isdigit())[:6]).ljust(6, "0")
    return datetime.fromisoformat(f"{head}.{frac}{tz}")


class _SnapshotTick:
    __slots__ = ("bid", "ask", "time")

    def __init__(self, bid: float, ask: float, time_val: datetime) -> None:
        self.bid = bid
        self.ask = ask
        self.time = time_val


try:
    _OANDA_TOKEN = get_secret("oanda_token")
    _OANDA_ACCOUNT = get_secret("oanda_account_id")
    try:
        _OANDA_PRACTICE = get_secret("oanda_practice").lower() == "true"
    except KeyError:
        _OANDA_PRACTICE = False
except Exception as exc:  # pragma: no cover - secrets must be present
    logging.error("%s failed to load OANDA secrets: %s", config.LOG_PREFIX_TICK, exc)
    _OANDA_TOKEN = ""
    _OANDA_ACCOUNT = ""
    _OANDA_PRACTICE = False

_PRICING_HOST = (
    "https://api-fxpractice.oanda.com" if _OANDA_PRACTICE else "https://api-fxtrade.oanda.com"
)
_PRICING_URL = f"{_PRICING_HOST}/v3/accounts/{_OANDA_ACCOUNT}/pricing"
_PRICING_HEADERS = {"Authorization": f"Bearer {_OANDA_TOKEN}"} if _OANDA_TOKEN else {}


def _is_off_hours(now_utc: datetime) -> bool:
    jst = now_utc + timedelta(hours=9)
    start = config.JST_OFF_HOURS_START
    end = config.JST_OFF_HOURS_END
    if start <= end:
        return start <= jst.hour < end
    return jst.hour >= start or jst.hour < end


async def _fetch_price_snapshot(logger: logging.Logger) -> Optional[_SnapshotTick]:
    if not _OANDA_TOKEN or not _OANDA_ACCOUNT:
        return None
    params = {"instruments": "USD_JPY"}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(_PRICING_URL, headers=_PRICING_HEADERS, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s pricing snapshot failed: %s", config.LOG_PREFIX_TICK, exc)
        return None
    prices = payload.get("prices") or []
    if not prices:
        return None
    price = prices[0]
    bids = price.get("bids") or []
    asks = price.get("asks") or []
    if not bids or not asks:
        return None
    try:
        bid = float(bids[0]["price"])
        ask = float(asks[0]["price"])
        ts = _parse_time(price.get("time", datetime.utcnow().isoformat() + "Z"))
    except Exception as exc:
        logger.warning("%s pricing snapshot parse error: %s", config.LOG_PREFIX_TICK, exc)
        return None
    tick = _SnapshotTick(bid=bid, ask=ask, time_val=ts)
    logger.warning(
        "%s snapshot bid=%.3f ask=%.3f", config.LOG_PREFIX_TICK, bid, ask
    )
    try:
        spread_monitor.update_from_tick(tick)
        tick_window.record(tick)
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s failed to record snapshot tick: %s", config.LOG_PREFIX_TICK, exc)
    return tick


def _build_client_order_id(side: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(f"{ts_ms}-{side}".encode("utf-8")).hexdigest()[:6]
    return f"qr-fast-{ts_ms}-{side[0]}{digest}"


def _pips(delta_price: float) -> float:
    return delta_price / config.PIP_VALUE


async def _collect_features(
    logger: logging.Logger,
    spread_pips: float,
    last_snapshot_fetch: float,
) -> tuple[Optional[SignalFeatures], float, float]:
    """
    Try to assemble a complete SignalFeatures payload by sampling additional
    pricing snapshots when tick depth is insufficient.
    """
    attempts = 0
    updated_spread = spread_pips
    last_fetch = last_snapshot_fetch

    while attempts <= config.SNAPSHOT_BURST_MAX_ATTEMPTS:
        ticks = tick_window.recent_ticks(seconds=config.LONG_WINDOW_SEC, limit=180)
        if len(ticks) >= config.MIN_TICK_COUNT:
            features = extract_features(updated_spread, ticks=ticks)
            if features and features.rsi is not None and features.atr_pips is not None:
                return features, updated_spread, last_fetch
        if attempts == config.SNAPSHOT_BURST_MAX_ATTEMPTS:
            break
        wait = config.SNAPSHOT_MIN_INTERVAL_SEC - (time.monotonic() - last_fetch)
        if wait > 0:
            await asyncio.sleep(wait)
        fetched_tick = await _fetch_price_snapshot(logger)
        last_fetch = time.monotonic()
        attempts += 1
        if fetched_tick is None:
            await asyncio.sleep(config.SNAPSHOT_BURST_INTERVAL_SEC)
        state = spread_monitor.get_state()
        if state:
            try:
                updated_spread = float(state.get("spread_pips", updated_spread))
            except (TypeError, ValueError):
                pass

    return None, updated_spread, last_fetch


async def fast_scalp_worker(shared_state: Optional[FastScalpState] = None) -> None:
    logger = logging.getLogger(__name__)
    shared_state = shared_state or FastScalpState()
    if not config.FAST_SCALP_ENABLED:
        logger.info("%s disabled via env, worker idling.", config.LOG_PREFIX_TICK)
        while True:
            await asyncio.sleep(30.0)

    rate_limiter = SlidingWindowRateLimiter(
        config.MAX_ORDERS_PER_MINUTE, config.MIN_ORDER_SPACING_SEC
    )
    stage_tracker = StageTracker()
    pos_manager = PositionManager()
    timeout_controller = TimeoutController()
    active_trades: dict[str, ActiveTrade] = {}
    last_sync = time.monotonic()
    spread_block_logged = False
    dd_block_logged = False
    off_hours_logged = False
    news_block_logged = False
    regime_block_logged: Optional[str] = None
    loss_block_logged = False
    order_backoff: float = 0.0
    next_order_after: float = 0.0
    last_snapshot_fetch: float = 0.0
    low_vol_counter = 0
    next_exit_review = time.monotonic() + config.REVIEW_INTERVAL_SEC

    loop_counter = 0
    empty_tick_streak = 0
    try:
        while True:
            loop_start = time.monotonic()
            now = _now_utc()
            loop_counter += 1
            if loop_counter % 200 == 0:
                logger.warning(
                    "%s loop=%d active=%d", config.LOG_PREFIX_TICK, loop_counter, len(active_trades)
                )

            # Watchdog: detect stale/empty tick window and abort for supervisor restart
            ticks_now = tick_window.recent_ticks(seconds=config.LONG_WINDOW_SEC, limit=2)
            if ticks_now:
                if empty_tick_streak >= config.EMPTY_TICK_WARN_LOOPS:
                    logger.info(
                        "%s tick stream recovered after %d empty loops", config.LOG_PREFIX_TICK, empty_tick_streak
                    )
                empty_tick_streak = 0
            else:
                empty_tick_streak += 1
                if empty_tick_streak in {config.EMPTY_TICK_WARN_LOOPS, config.EMPTY_TICK_FATAL_LOOPS} or (
                    empty_tick_streak > config.EMPTY_TICK_WARN_LOOPS and empty_tick_streak % config.EMPTY_TICK_WARN_LOOPS == 0
                ):
                    logger.warning(
                        "%s no ticks for %d loops (%.1fs) window=%.1fs",
                        config.LOG_PREFIX_TICK,
                        empty_tick_streak,
                        empty_tick_streak * config.LOOP_INTERVAL_SEC,
                        config.LONG_WINDOW_SEC,
                    )
                if empty_tick_streak >= config.EMPTY_TICK_FATAL_LOOPS:
                    raise RuntimeError(f"{config.LOG_PREFIX_TICK} aborting due to empty tick window")

            if _is_off_hours(now):
                if not off_hours_logged:
                    logger.info("%s pause during JST off-hours window.", config.LOG_PREFIX_TICK)
                    off_hours_logged = True
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            off_hours_logged = False

            # Drawdown guard is monitored externally; keep warnings but do not block entries.
            if not can_trade("scalp_fast"):
                if not dd_block_logged:
                    logger.warning(
                        "%s drawdown guard triggered, proceeding without block (external override).",
                        config.LOG_PREFIX_TICK,
                    )
                    dd_block_logged = True
            else:
                dd_block_logged = False

            blocked, remain, spread_state, spread_reason = spread_monitor.is_blocked()
            spread_pips = spread_state["spread_pips"] if spread_state else 0.0
            if blocked or spread_pips > config.MAX_SPREAD_PIPS:
                if not spread_block_logged:
                    logger.info(
                        "%s skip due to spread %.2fp (remain=%ss reason=%s)",
                        config.LOG_PREFIX_TICK,
                        spread_pips,
                        remain,
                        spread_reason or "guard_active",
                    )
                    log_metric(
                        "fast_scalp_skip",
                        float(spread_pips),
                        tags={"reason": "spread", "guard": spread_reason or ""},
                        ts=now,
                    )
                    spread_block_logged = True
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            spread_block_logged = False

            if config.NEWS_BLOCK_MINUTES > 0 and news_block_active(
                config.NEWS_BLOCK_MINUTES, min_impact=config.NEWS_BLOCK_MIN_IMPACT
            ):
                if not news_block_logged:
                    logger.info(
                        "%s pause due to upcoming news (impact≥%s within %.0f min)",
                        config.LOG_PREFIX_TICK,
                        config.NEWS_BLOCK_MIN_IMPACT,
                        config.NEWS_BLOCK_MINUTES,
                    )
                    log_metric(
                        "fast_scalp_skip",
                        config.NEWS_BLOCK_MINUTES,
                        tags={"reason": "news"},
                        ts=now,
                    )
                    news_block_logged = True
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            news_block_logged = False

            regime_label = current_regime("M1", event_mode=False)
            if regime_label and regime_label in config.BLOCK_REGIMES:
                if regime_block_logged != regime_label:
                    logger.info(
                        "%s regime=%s blocked via config",
                        config.LOG_PREFIX_TICK,
                        regime_label,
                    )
                    log_metric(
                        "fast_scalp_skip",
                        1.0,
                        tags={"reason": "regime", "regime": regime_label},
                        ts=now,
                    )
                    regime_block_logged = regime_label
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            regime_block_logged = None

            if config.LOSS_STREAK_MAX > 0 and config.LOSS_STREAK_COOLDOWN_MIN > 0:
                loss_block, remain_sec = loss_cooldown_status(
                    "scalp",
                    max_losses=config.LOSS_STREAK_MAX,
                    cooldown_minutes=config.LOSS_STREAK_COOLDOWN_MIN,
                )
                if loss_block:
                    if not loss_block_logged:
                        logger.warning(
                            "%s cooldown after %d consecutive losses (%.0fs remain)",
                            config.LOG_PREFIX_TICK,
                            config.LOSS_STREAK_MAX,
                            remain_sec,
                        )
                        log_metric(
                            "fast_scalp_skip",
                            float(remain_sec),
                            tags={"reason": "loss_cooldown"},
                            ts=now,
                        )
                        loss_block_logged = True
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue
            loss_block_logged = False

            snapshot = shared_state.snapshot()
            range_active_flag = bool(getattr(snapshot, "range_active", False))
            m1_rsi_snapshot = getattr(snapshot, "m1_rsi", None)
            if m1_rsi_snapshot is not None:
                age_limit = getattr(snapshot, "m1_rsi_age_sec", None)
                if age_limit is not None and age_limit > config.M1_RSI_CONFIRM_SPAN_SEC:
                    m1_rsi_snapshot = None

            snapshot_needed = False
            age_ms = None
            if spread_state:
                age_ms = spread_state.get("age_ms")
                if spread_state.get("stale"):
                    snapshot_needed = True
                elif age_ms is not None and age_ms > config.STALE_TICK_MAX_SEC * 1000:
                    snapshot_needed = True
            else:
                snapshot_needed = True

            monotonic_now = time.monotonic()
            if snapshot_needed and monotonic_now - last_snapshot_fetch >= config.SNAPSHOT_MIN_INTERVAL_SEC:
                fetched_tick = await _fetch_price_snapshot(logger)
                last_snapshot_fetch = monotonic_now
                if fetched_tick:
                    spread_state = spread_monitor.get_state()
                    spread_pips = (
                        spread_state.get("spread_pips") if spread_state else spread_pips
                    )

            features, spread_pips, last_snapshot_fetch = await _collect_features(
                logger, spread_pips, last_snapshot_fetch
            )
            if not features:
                if loop_counter % 40 == 0:
                    logger.debug(
                        "%s unable to resolve indicators spread=%.3f attempts=%d",
                        config.LOG_PREFIX_TICK,
                        spread_pips,
                        config.SNAPSHOT_BURST_MAX_ATTEMPTS,
                    )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            # Quality gating: ensure volatility and sampling density are sufficient
            low_quality = False
            span_ok = span_requirement_ok(features.span_seconds, features.tick_count)
            if features.atr_pips is None or features.atr_pips < config.MIN_ENTRY_ATR_PIPS:
                low_quality = True
            elif features.tick_count < config.MIN_ENTRY_TICK_COUNT:
                low_quality = True
            elif not span_ok:
                low_quality = True

            if low_quality:
                logger.info(
                    "%s low-vol check atr=%.3f ticks=%d span=%.2fs thr_atr=%.2f thr_ticks=%d span_ok=%s",
                    config.LOG_PREFIX_TICK,
                    (features.atr_pips if features.atr_pips is not None else -1.0),
                    features.tick_count,
                    features.span_seconds,
                    config.MIN_ENTRY_ATR_PIPS,
                    config.MIN_ENTRY_TICK_COUNT,
                    span_ok,
                )
                low_vol_counter += 1
                if (
                    config.LOW_VOL_COOLDOWN_SEC > 0
                    and low_vol_counter >= config.LOW_VOL_MAX_CONSECUTIVE
                ):
                    logger.info(
                        "%s low-volatility cooldown triggered atr=%.3f ticks=%d span=%.2fs",
                        config.LOG_PREFIX_TICK,
                        features.atr_pips if features.atr_pips is not None else -1.0,
                        features.tick_count,
                        features.span_seconds,
                    )
                    for direction in ("long", "short"):
                        stage_tracker.set_cooldown(
                            "scalp_fast",
                            direction,
                            reason="low_volatility",
                            seconds=int(config.LOW_VOL_COOLDOWN_SEC),
                            now=now,
                        )
                    low_vol_counter = 0
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            low_vol_counter = 0

            tick_rate = (
                features.tick_count / max(features.span_seconds, 0.5)
                if features.span_seconds > 0.0
                else float(features.tick_count)
            )

            skip_new_entry = False
            pattern_prob: Optional[float] = None
            atr_value = features.atr_pips if features.atr_pips is not None else 0.0

            forced_exit_reasons = {"drawdown", "health_exit"}
            exit_review_due = False
            review_now = time.monotonic()
            if review_now >= next_exit_review:
                exit_review_due = True
                next_exit_review = review_now + config.REVIEW_INTERVAL_SEC

            if exit_review_due:
                for trade_id, active in list(active_trades.items()):
                    pips_gain = _pips(features.latest_mid - active.entry_price)
                    if active.side == "short":
                        pips_gain = -pips_gain
                    elapsed = time.monotonic() - active.opened_monotonic
                    latency_ms_val = float(age_ms) if isinstance(age_ms, (int, float)) else None

                    decision = timeout_controller.update(
                        trade_id,
                        elapsed_sec=elapsed,
                        pips_gain=pips_gain,
                        features=features,
                        tick_rate=tick_rate,
                        latency_ms=latency_ms_val,
                    )

                    close_reason: Optional[str] = None
                    max_drawdown_close = active.max_drawdown_close_pips
                    if features.rsi is not None and pips_gain < 0:
                        if active.side == "long" and features.rsi < config.RSI_EXIT_LONG:
                            close_reason = "rsi_fade"
                        elif active.side == "short" and features.rsi > config.RSI_EXIT_SHORT:
                            close_reason = "rsi_fade"
                    if close_reason is None and features.atr_pips is not None and pips_gain < 0:
                        if features.atr_pips >= config.ATR_HIGH_VOL_PIPS:
                            close_reason = "atr_spike"
                    if close_reason is None:
                        drawdown_hit = pips_gain <= -max_drawdown_close
                        if drawdown_hit:
                            close_reason = "drawdown"
                    if close_reason is None and decision.action == "close":
                        close_reason = decision.reason or "timeout_controller"

                    if (
                        elapsed < config.MIN_HOLD_SEC
                        and close_reason
                        and close_reason not in forced_exit_reasons
                        and pips_gain > -max_drawdown_close
                    ):
                        close_reason = None

                    if close_reason is not None:
                        # "損切り禁止" が有効な場合、含み損では決済せず見送り
                        # ただし強制理由（ドローダウン/ヘルス）は例外として許容
                        if (
                            config.NO_LOSS_CLOSE
                            and pips_gain < 0
                            and close_reason not in forced_exit_reasons
                        ):
                            log_metric(
                                "fast_scalp_skip",
                                float(pips_gain),
                                tags={"reason": "no_loss_close", "side": active.side},
                                ts=now,
                            )
                            # 短いクールダウンで再エントリーを抑制
                            stage_tracker.set_cooldown(
                                "scalp_fast",
                                active.side,
                                reason="no_loss_close",
                                seconds=int(config.ENTRY_COOLDOWN_SEC),
                                now=now,
                            )
                            skip_new_entry = True
                            continue
                        if rate_limiter.allow():
                            rate_limiter.record()
                            summary_payload: dict[str, float | str | bool] = {}
                            try:
                                await close_trade(trade_id)
                            finally:
                                summary_payload = timeout_controller.finalize(
                                    trade_id,
                                    reason=close_reason,
                                    pips_gain=pips_gain,
                                    tick_rate=tick_rate,
                                    spread_pips=features.spread_pips,
                                )
                                stage_tracker.set_cooldown(
                                    "scalp_fast",
                                    active.side,
                                    reason="manual_exit",
                                    seconds=int(config.ENTRY_COOLDOWN_SEC),
                                    now=now,
                                )
                                active_trades.pop(trade_id, None)
                            logger.info(
                                "%s close trade=%s side=%s reason=%s profile=%s pnl=%.2fp elapsed=%.1fs meta=%s",
                                config.LOG_PREFIX_TICK,
                                trade_id,
                                active.side,
                                close_reason,
                                active.profile.name,
                                pips_gain,
                                elapsed,
                                summary_payload if summary_payload else "{}",
                            )
                            log_metric(
                                "fast_scalp_exit",
                                pips_gain,
                                tags={
                                    "reason": close_reason,
                                    "profile": active.profile.name,
                                    "timeout_type": str(summary_payload.get("timeout_type", "")),
                                },
                                ts=now,
                            )
                            skip_new_entry = True
                        else:
                            skip_new_entry = True
            if skip_new_entry:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            # Periodic reconciliation with live positions
            now_monotonic = time.monotonic()
            if now_monotonic - last_sync >= config.SYNC_INTERVAL_SEC:
                last_sync = now_monotonic
                try:
                    positions = pos_manager.get_open_positions()
                except Exception as exc:
                    logger.warning("%s sync positions failed: %s", config.LOG_PREFIX_TICK, exc)
                else:
                    pocket = positions.get("scalp_fast")
                    open_trades = (pocket or {}).get("open_trades") or []
                    updated: dict[str, ActiveTrade] = {}
                    for tr in open_trades:
                        trade_id = str(tr.get("trade_id"))
                        if not trade_id:
                            continue
                        direction = tr.get("side") or ("long" if tr.get("units", 0) > 0 else "short")
                        units = int(tr.get("units", 0) or 0)
                        entry_price = float(tr.get("price", features.latest_mid))
                        client_id = str(tr.get("client_id") or "")
                        existing = active_trades.get(trade_id)
                        entry_meta = tr.get("entry_thesis") or {}
                        profile_name = str(entry_meta.get("profile") or (existing.profile.name if existing else ""))
                        profile = get_profile(profile_name)
                        timeout_raw = entry_meta.get("profile_timeout_sec")
                        timeout_min_gain_raw = entry_meta.get("profile_timeout_min_gain_pips")
                        drawdown_close_raw = entry_meta.get("profile_drawdown_close_pips")
                        timeout_limit = (
                            float(timeout_raw)
                            if timeout_raw not in (None, "")
                            else (existing.timeout_limit if existing else profile.timeout_sec)
                        )
                        timeout_min_gain = (
                            float(timeout_min_gain_raw)
                            if timeout_min_gain_raw not in (None, "")
                            else (existing.timeout_min_gain if existing else profile.timeout_min_gain_pips)
                        )
                        drawdown_close = (
                            float(drawdown_close_raw)
                            if drawdown_close_raw not in (None, "")
                            else (existing.max_drawdown_close_pips if existing else profile.drawdown_close_pips)
                        )
                        opened_monotonic = existing.opened_monotonic if existing else time.monotonic()
                        opened_at = existing.opened_at if existing else _now_utc()
                        updated[trade_id] = ActiveTrade(
                            trade_id=trade_id,
                            side=direction,
                            units=units,
                            entry_price=entry_price,
                            opened_at=opened_at,
                            opened_monotonic=opened_monotonic,
                            client_order_id=client_id,
                            sl_price=existing.sl_price if existing else entry_price,
                            tp_price=existing.tp_price if existing else entry_price,
                            profile=profile,
                            timeout_limit=timeout_limit,
                            timeout_min_gain=timeout_min_gain,
                            max_drawdown_close_pips=drawdown_close,
                        )
                        if not timeout_controller.has_trade(trade_id):
                            timeout_controller.register_trade(
                                trade_id,
                                side=direction,
                                entry_price=entry_price,
                                entry_monotonic=opened_monotonic,
                                features=features,
                                spread_pips=spread_pips,
                                tick_rate=tick_rate,
                                latency_ms=float(age_ms)
                                if isinstance(age_ms, (int, float))
                                else None,
                            )
                    removed_ids = set(active_trades.keys()) - set(updated.keys())
                    for removed_id in removed_ids:
                        removed_trade = active_trades.get(removed_id)
                        if removed_trade is not None:
                            removed_pips = _pips(features.latest_mid - removed_trade.entry_price)
                            if removed_trade.side == "short":
                                removed_pips = -removed_pips
                        else:
                            removed_pips = 0.0
                        timeout_controller.finalize(
                            removed_id,
                            reason="sync_prune",
                            pips_gain=removed_pips,
                            tick_rate=tick_rate,
                            spread_pips=features.spread_pips,
                        )
                    active_trades = updated

            # Max concurrent trades guard
            if len(active_trades) >= config.MAX_ACTIVE_TRADES:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            action = evaluate_signal(
                features,
                m1_rsi=m1_rsi_snapshot,
                range_active=range_active_flag,
            )
            if not action:
                # 強制エントリー（検証用）: スプレッドが許容内であれば短期モメンタム方向に入る
                if config.FORCE_ENTRIES and spread_pips <= config.MAX_SPREAD_PIPS:
                    action = "OPEN_LONG" if features.short_momentum_pips >= 0 else "OPEN_SHORT"
                else:
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

            reversal = action.startswith("REVERSAL")
            direction = "long" if action.endswith("LONG") else "short"

            if features.pattern_features is not None:
                pattern_prob = pattern_score(features.pattern_features, direction)
                if (
                    pattern_prob is not None
                    and pattern_prob < config.PATTERN_MIN_PROB
                ):
                    logger.debug(
                        "%s pattern model veto side=%s score=%.3f",
                        config.LOG_PREFIX_TICK,
                        direction,
                        pattern_prob,
                    )
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

            directional_count = sum(1 for tr in active_trades.values() if tr.side == direction)
            if directional_count >= config.MAX_PER_DIRECTION:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            cooldown = stage_tracker.get_cooldown("scalp_fast", direction, now=now)
            if cooldown and not reversal:
                logger.debug(
                    "%s cooldown active pocket=scalp_fast dir=%s until=%s reason=%s",
                    config.LOG_PREFIX_TICK,
                    direction,
                    cooldown.cooldown_until,
                    cooldown.reason,
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            if cooldown and reversal:
                stage_tracker.clear_cooldown("scalp_fast", direction, reason=cooldown.reason)

            monotonic_now = time.monotonic()
            if monotonic_now < next_order_after:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            profile = select_profile(action, features, range_active=range_active_flag)
            session_tag = current_session()

            if not rate_limiter.allow():
                logger.info(
                    "%s skip signal due to rate limit side=%s mom=%.2fp",
                    config.LOG_PREFIX_TICK,
                    direction,
                    features.momentum_pips,
                )
                log_metric(
                    "fast_scalp_skip",
                    features.momentum_pips,
                    tags={"reason": "rate_limit", "side": direction},
                    ts=now,
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            profile_sl_pips = profile.sl_pips if profile.sl_pips is not None else config.SL_PIPS
            adjusted_lot = False
            if config.FIXED_UNITS and abs(config.FIXED_UNITS) >= config.MIN_UNITS:
                units = abs(int(config.FIXED_UNITS))
                units = units if direction == "long" else -units
                allowed_lot_raw = units / 100000.0
                lot = allowed_lot_raw
            else:
                allowed_lot_raw = allowed_lot(
                    snapshot.account_equity,
                    sl_pips=profile_sl_pips,
                    margin_available=snapshot.margin_available,
                    price=features.latest_mid,
                    margin_rate=snapshot.margin_rate,
                    risk_pct_override=snapshot.risk_pct_override,
                    pocket="scalp",
                )
                lot = min(allowed_lot_raw, config.MAX_LOT)
                if lot <= 0.0:
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue
                min_lot = config.MIN_UNITS / 100000.0
                if 0.0 < lot < min_lot:
                    lot = min(config.MAX_LOT, min_lot)
                    adjusted_lot = True
                units = int(round(lot * 100000.0))
            if (
                snapshot.margin_rate > 0.0
                and snapshot.margin_available > 0.0
                and 0.0 < config.MAX_MARGIN_USAGE < 1.0
            ):
                margin_budget = snapshot.margin_available * config.MAX_MARGIN_USAGE
                margin_per_unit = features.latest_mid * snapshot.margin_rate
                if margin_per_unit > 0:
                    max_units_margin = int(margin_budget / margin_per_unit)
                    if max_units_margin < config.MIN_UNITS:
                        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                        continue
                    if abs(units) > max_units_margin:
                        units = max_units_margin if units > 0 else -max_units_margin
                        adjusted_lot = True
            if units < config.MIN_UNITS:
                units = config.MIN_UNITS
                adjusted_lot = True
            if not config.FIXED_UNITS:
                if direction == "short":
                    units = -units
            if abs(units) < config.MIN_UNITS:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            logger.info(
                "%s sizing action=%s side=%s profile=%s allowed_lot=%.3f final_lot=%.3f units=%d min_units=%d adj=%s",
                config.LOG_PREFIX_TICK,
                action,
                direction,
                profile.name,
                allowed_lot_raw,
                lot,
                units,
                config.MIN_UNITS,
                adjusted_lot,
            )
            if adjusted_lot:
                logger.debug(
                    "%s adjusted lot to %.3f units=%d (min=%d)",
                    config.LOG_PREFIX_TICK,
                    lot,
                    units,
                    config.MIN_UNITS,
                )

            client_id = _build_client_order_id(direction)

            state = spread_monitor.get_state()
            bid_quote = float(state.get("bid") or 0.0) if state else 0.0
            ask_quote = float(state.get("ask") or 0.0) if state else 0.0
            last_ticks = tick_window.recent_ticks(3.0, limit=1)
            if last_ticks:
                last_tick = last_ticks[-1]
                bid_quote = float(last_tick.get("bid") or bid_quote or features.latest_mid)
                ask_quote = float(last_tick.get("ask") or ask_quote or features.latest_mid)
            expected_entry_price = (
                ask_quote if direction == "long" else bid_quote
            ) or features.latest_mid
            spread_padding = max(features.spread_pips, config.TP_SPREAD_BUFFER_PIPS)
            tp_margin = max(config.TP_SAFE_MARGIN_PIPS, features.spread_pips * 0.5)
            base_tp = config.TP_BASE_PIPS + spread_padding + tp_margin
            tp_pips = max(0.2, base_tp * profile.tp_margin_multiplier + profile.tp_adjust)
            # Ensure TP nets out spread cost by at least TP_NET_MIN_PIPS
            tp_floor = max(features.spread_pips, config.TP_SPREAD_BUFFER_PIPS) + config.TP_NET_MIN_PIPS
            if tp_pips < tp_floor:
                logger.debug(
                    "%s tp_floor applied tp=%.2f -> %.2f (spread=%.2f floor=%.2f)",
                    config.LOG_PREFIX_TICK,
                    tp_pips,
                    tp_floor,
                    features.spread_pips,
                    tp_floor,
                )
                tp_pips = tp_floor
            sl_pips = profile_sl_pips
            timeout_limit_initial = resolve_timeout(profile, features.atr_pips)
            entry_price = expected_entry_price
            if direction == "long":
                sl_price = None if not config.USE_SL else entry_price - sl_pips * config.PIP_VALUE
                tp_price = entry_price + tp_pips * config.PIP_VALUE
            else:
                sl_price = None if not config.USE_SL else entry_price + sl_pips * config.PIP_VALUE
                tp_price = entry_price - tp_pips * config.PIP_VALUE
            sl_price, tp_price = clamp_sl_tp(entry_price, sl_price, tp_price, direction == "long")

            thesis = {
                "strategy_tag": "fast_scalp",
                "momentum_pips": round(features.momentum_pips, 3),
                "short_momentum_pips": round(features.short_momentum_pips, 3),
                "range_pips": round(features.range_pips, 3),
                "impulse_pips": round(features.impulse_pips, 3),
                "impulse_span_sec": round(features.impulse_span_sec, 3),
                "impulse_direction": features.impulse_direction,
                "consolidation_range_pips": round(features.consolidation_range_pips, 3),
                "consolidation_span_sec": round(features.consolidation_span_sec, 3),
                "consolidation_ok": features.consolidation_ok,
                "spread_pips": round(features.spread_pips, 3),
                "tick_count": features.tick_count,
                "tick_span_sec": round(features.span_seconds, 3),
                "weight_scalp": snapshot.weight_scalp,
                "tp_pips": round(tp_pips, 3),
                "entry_price_expect": round(entry_price, 5),
                "sl_price_initial": round(sl_price, 5),
                "tp_price_initial": round(tp_price, 5),
                "signal": action,
                "profile": profile.name,
                "session": session_tag,
                "profile_tp_pips": round(tp_pips, 3),
                "profile_sl_pips": round(sl_pips, 3),
                "profile_timeout_sec": None if timeout_limit_initial is None else float(timeout_limit_initial),
                "profile_timeout_min_gain_pips": profile.timeout_min_gain_pips,
                "profile_drawdown_close_pips": profile.drawdown_close_pips,
                "range_active_entry": range_active_flag,
                "m1_rsi": None if snapshot.m1_rsi is None else round(snapshot.m1_rsi, 2),
                "tick_rsi": None if features.rsi is None else round(features.rsi, 2),
                "tick_rsi_source": features.rsi_source,
                "tick_atr": None if features.atr_pips is None else round(features.atr_pips, 3),
                "tick_atr_source": features.atr_source,
                "pattern_tag": features.pattern_tag,
                "pattern_features": list(features.pattern_features)
                if features.pattern_features is not None
                else None,
                "pattern_score": None if pattern_prob is None else round(pattern_prob, 3),
            }

            try:
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl_price,
                    tp_price,
                    "scalp_fast",
                    client_order_id=client_id,
                    entry_thesis={**thesis, "strategy_tag": "fast_scalp"},
                )
            except Exception as exc:
                logger.error(
                    "%s order error side=%s exc=%s",
                    config.LOG_PREFIX_TICK,
                    direction,
                    exc,
                )
                order_backoff = max(order_backoff * 1.8, 0.3) if order_backoff else 0.3
                next_order_after = monotonic_now + order_backoff
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            if not trade_id:
                logger.warning(
                    "%s order did not return trade id side=%s units=%d client_id=%s",
                    config.LOG_PREFIX_TICK,
                    direction,
                    units,
                    client_id,
                )
                order_backoff = max(order_backoff * 1.8, 0.3) if order_backoff else 0.3
                next_order_after = monotonic_now + order_backoff
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            order_backoff = 0.0
            rate_limiter.record()
            stage_tracker.set_cooldown(
                "scalp_fast",
                direction,
                reason="reversal_entry" if reversal else "entry",
                seconds=int(config.ENTRY_COOLDOWN_SEC),
                now=now,
            )
            next_order_after = max(
                next_order_after,
                time.monotonic() + config.REENTRY_MIN_GAP_SEC,
            )
            actual_entry_price = entry_price
            sl_adjust_pips = sl_pips + config.SL_POST_ADJUST_BUFFER_PIPS
            if direction == "long":
                actual_sl_price = None if not config.USE_SL else round(actual_entry_price - sl_adjust_pips * config.PIP_VALUE, 3)
                actual_tp_price = round(actual_entry_price + tp_pips * config.PIP_VALUE, 3)
            else:
                actual_sl_price = None if not config.USE_SL else round(actual_entry_price + sl_adjust_pips * config.PIP_VALUE, 3)
                actual_tp_price = round(actual_entry_price - tp_pips * config.PIP_VALUE, 3)
            set_ok = await set_trade_protections(
                trade_id,
                sl_price=actual_sl_price,
                tp_price=actual_tp_price,
            )
            if not set_ok:
                logger.warning(
                    "%s protection update failed trade=%s", config.LOG_PREFIX_TICK, trade_id
                )
            active_trades[trade_id] = ActiveTrade(
                trade_id=trade_id,
                side=direction,
                units=units,
                entry_price=actual_entry_price,
                opened_at=now,
                opened_monotonic=monotonic_now,
                client_order_id=client_id,
                sl_price=actual_sl_price,
                tp_price=actual_tp_price,
                profile=profile,
                timeout_limit=timeout_limit_initial if timeout_limit_initial is not None else profile.timeout_sec,
                timeout_min_gain=profile.timeout_min_gain_pips,
                max_drawdown_close_pips=profile.drawdown_close_pips,
            )
            timeout_controller.register_trade(
                trade_id,
                side=direction,
                entry_price=actual_entry_price,
                entry_monotonic=monotonic_now,
                features=features,
                spread_pips=features.spread_pips,
                tick_rate=tick_rate,
                latency_ms=float(age_ms) if isinstance(age_ms, (int, float)) else None,
            )
            log_metric(
                "fast_scalp_signal",
                features.momentum_pips,
                tags={
                    "side": direction,
                    "action": action,
                    "range_pips": f"{features.range_pips:.2f}",
                    "profile": profile.name,
                },
                ts=now,
            )
            logger.info(
                "%s open trade=%s signal=%s side=%s profile=%s units=%d tp=%.3f sl=%.3f mom=%.2fp range=%.2fp spread=%.2fp",
                config.LOG_PREFIX_TICK,
                trade_id,
                action,
                direction,
                profile.name,
                units,
                actual_tp_price or 0.0,
                actual_sl_price or 0.0,
                features.momentum_pips,
                features.range_pips,
                features.spread_pips,
            )

            elapsed = time.monotonic() - loop_start
            await asyncio.sleep(max(0.05, config.LOOP_INTERVAL_SEC - elapsed))
    finally:
        stage_tracker.close()
        pos_manager.close()
        logger.info("%s worker shutdown", config.LOG_PREFIX_TICK)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def _main() -> None:  # pragma: no cover
    _configure_logging()
    logger = logging.getLogger(__name__)
    try:
        asyncio.run(fast_scalp_worker(shared_state=None))
    except TypeError as exc:
        # 再発防止のため、引数不足で失敗した場合はデフォルト state を明示して再実行する
        logger.exception("fast_scalp_worker entry failed: %s; retrying with default state", exc)
        asyncio.run(fast_scalp_worker(shared_state=FastScalpState()))


if __name__ == "__main__":  # pragma: no cover
    _main()
