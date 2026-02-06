"""
Async worker that drives ultra-short-term scalping based on tick data.
"""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from indicators.factor_cache import all_factors, get_candles_snapshot

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import httpx

from workers.common.exit_utils import close_trade
from execution.order_manager import market_order, set_trade_protections
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
from utils.oanda_account import get_account_snapshot, get_position_summary
from workers.common.quality_gate import current_regime
from workers.common.air_state import evaluate_air

from . import config

POCKET = config.POCKET
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



_PROJ_TF_MINUTES = {"M1": 1.0, "M5": 5.0, "H1": 60.0, "H4": 240.0, "D1": 1440.0}


def _projection_mode(pocket, mode_override=None):
    if mode_override:
        return mode_override
    if globals().get("IS_RANGE"):
        return "range"
    if globals().get("IS_PULLBACK"):
        return "pullback"
    if pocket in {"scalp", "scalp_fast"}:
        return "scalp"
    return "trend"


def _projection_tfs(pocket, mode):
    if pocket == "macro":
        return ("H4", "H1")
    if pocket == "micro":
        return ("M5", "M1")
    if pocket in {"scalp", "scalp_fast"}:
        return ("M1",)
    return ("M5", "M1")


def _projection_candles(tfs):
    for tf in tfs:
        candles = get_candles_snapshot(tf, limit=120)
        if candles and len(candles) >= 30:
            return tf, list(candles)
    return None, None


def _score_ma(ma, side, opp_block_bars):
    if ma is None:
        return None
    align = ma.gap_pips >= 0 if side == "long" else ma.gap_pips <= 0
    cross_soon = ma.projected_cross_bars is not None and ma.projected_cross_bars <= opp_block_bars
    if align and not cross_soon:
        return 0.7
    if align and cross_soon:
        return -0.4
    if cross_soon:
        return -0.8
    return -0.5


def _score_rsi(rsi, side, long_target, short_target, overheat_bars):
    if rsi is None:
        return None
    score = 0.0
    if side == "long":
        if rsi.rsi >= long_target and rsi.slope_per_bar > 0:
            score = 0.4
        elif rsi.rsi <= (long_target - 8) and rsi.slope_per_bar < 0:
            score = -0.4
        if rsi.eta_upper_bars is not None and rsi.eta_upper_bars <= overheat_bars:
            score -= 0.2
    else:
        if rsi.rsi <= short_target and rsi.slope_per_bar < 0:
            score = 0.4
        elif rsi.rsi >= (short_target + 8) and rsi.slope_per_bar > 0:
            score = -0.4
        if rsi.eta_lower_bars is not None and rsi.eta_lower_bars <= overheat_bars:
            score -= 0.2
    return score


def _score_adx(adx, trend_mode, threshold):
    if adx is None:
        return None
    if trend_mode:
        if adx.adx >= threshold and adx.slope_per_bar >= 0:
            return 0.4
        if adx.adx <= threshold and adx.slope_per_bar < 0:
            return -0.4
        return 0.0
    if adx.adx >= threshold and adx.slope_per_bar > 0:
        return -0.5
    if adx.adx <= threshold and adx.slope_per_bar < 0:
        return 0.3
    return 0.0


def _score_bbw(bbw, threshold):
    if bbw is None:
        return None
    if bbw.bbw <= threshold and bbw.slope_per_bar <= 0:
        return 0.5
    if bbw.bbw > threshold and bbw.slope_per_bar > 0:
        return -0.5
    return 0.0


def _projection_decision(side, pocket, mode_override=None):
    mode = _projection_mode(pocket, mode_override=mode_override)
    tfs = _projection_tfs(pocket, mode)
    tf, candles = _projection_candles(tfs)
    if not candles:
        return True, 1.0, {}
    minutes = _PROJ_TF_MINUTES.get(tf, 1.0)

    if mode == "trend":
        params = {
            "adx_threshold": 20.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 5.0,
            "long_target": 52.0,
            "short_target": 48.0,
            "overheat_bars": 3.0,
            "weights": {"ma": 0.45, "rsi": 0.25, "adx": 0.30},
            "block_score": -0.6,
            "size_scale": 0.18,
        }
    elif mode == "pullback":
        params = {
            "adx_threshold": 18.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 4.0,
            "long_target": 50.0,
            "short_target": 50.0,
            "overheat_bars": 3.0,
            "weights": {"ma": 0.40, "rsi": 0.40, "adx": 0.20},
            "block_score": -0.55,
            "size_scale": 0.15,
        }
    elif mode == "scalp":
        params = {
            "adx_threshold": 18.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 3.0,
            "long_target": 52.0,
            "short_target": 48.0,
            "overheat_bars": 2.0,
            "weights": {"ma": 0.50, "rsi": 0.30, "adx": 0.20},
            "block_score": -0.6,
            "size_scale": 0.12,
        }
    else:
        params = {
            "adx_threshold": 16.0,
            "bbw_threshold": 0.14,
            "opp_block_bars": 4.0,
            "long_target": 45.0,
            "short_target": 55.0,
            "overheat_bars": 3.0,
            "weights": {"bbw": 0.40, "rsi": 0.35, "adx": 0.25},
            "block_score": -0.5,
            "size_scale": 0.15,
        }

    ma = compute_ma_projection({"candles": candles}, timeframe_minutes=minutes)
    rsi = compute_rsi_projection(candles, timeframe_minutes=minutes)
    adx = compute_adx_projection(candles, timeframe_minutes=minutes, trend_threshold=params["adx_threshold"])
    bbw = None
    if mode == "range":
        bbw = compute_bbw_projection(candles, timeframe_minutes=minutes, squeeze_threshold=params["bbw_threshold"])

    scores = {}
    ma_score = _score_ma(ma, side, params["opp_block_bars"])
    if ma_score is not None and "ma" in params["weights"]:
        scores["ma"] = ma_score
    rsi_score = _score_rsi(rsi, side, params["long_target"], params["short_target"], params["overheat_bars"])
    if rsi_score is not None and "rsi" in params["weights"]:
        scores["rsi"] = rsi_score
    adx_score = _score_adx(adx, mode != "range", params["adx_threshold"])
    if adx_score is not None and "adx" in params["weights"]:
        scores["adx"] = adx_score
    bbw_score = _score_bbw(bbw, params["bbw_threshold"])
    if bbw_score is not None and "bbw" in params["weights"]:
        scores["bbw"] = bbw_score

    weight_sum = 0.0
    score_sum = 0.0
    for key, score in scores.items():
        weight = params["weights"].get(key, 0.0)
        weight_sum += weight
        score_sum += weight * score
    score = score_sum / weight_sum if weight_sum > 0 else 0.0

    allow = score > params["block_score"]
    size_mult = 1.0 + max(0.0, score) * params["size_scale"]
    size_mult = max(0.8, min(1.35, size_mult))

    detail = {
        "mode": mode,
        "tf": tf,
        "score": round(score, 3),
        "size_mult": round(size_mult, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
    }
    return allow, size_mult, detail
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
    if not config.OFF_HOURS_ENABLED:
        return False
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
    tag = (config.STRATEGY_TAG or "fast_scalp").lower()
    tag = "".join(ch for ch in tag if ch.isalnum())[:6] or "fast"
    digest = hashlib.sha1(f"{ts_ms}-{side}-{tag}".encode("utf-8")).hexdigest()[:6]
    return f"qr-fast-{ts_ms}-{tag}{side[0]}{digest}"


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
            if not can_trade(POCKET):
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
            equity = snapshot.account_equity
            margin_available = snapshot.margin_available
            margin_rate = snapshot.margin_rate
            if equity <= 0 or margin_available <= 0 or margin_rate <= 0:
                try:
                    live = get_account_snapshot(cache_ttl_sec=0.5)
                    equity = live.nav
                    margin_available = live.margin_available
                    margin_rate = live.margin_rate
                    logger.info(
                        "%s refreshed account snapshot equity=%.2f margin_available=%.2f margin_rate=%.4f",
                        config.LOG_PREFIX_TICK,
                        equity,
                        margin_available,
                        margin_rate,
                    )
                except Exception as exc:
                    logger.warning("%s account snapshot refresh failed: %s", config.LOG_PREFIX_TICK, exc)
            equity = max(equity, 1.0)
            margin_available = max(margin_available, 0.0)
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
                            POCKET,
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

            recent_tick = None
            latest_tick_mid: Optional[float] = None
            last_tick_age_ms: Optional[float] = None
            recent_ticks = tick_window.recent_ticks(seconds=3.0, limit=1)
            if recent_ticks:
                recent_tick = recent_ticks[-1]
                try:
                    latest_tick_mid = float(recent_tick.get("mid") or 0.0)
                except Exception:
                    latest_tick_mid = None
                try:
                    epoch_val = float(recent_tick.get("epoch") or 0.0)
                    if epoch_val > 0:
                        last_tick_age_ms = max(0.0, (time.time() - epoch_val) * 1000.0)
                except Exception:
                    last_tick_age_ms = None
            if last_tick_age_ms is None and isinstance(age_ms, (int, float)):
                last_tick_age_ms = float(age_ms)

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
                    if close_reason is not None and close_reason not in forced_exit_reasons:
                        reason_key = close_reason.lower()
                        if reason_key in config.EXIT_IGNORE_REASONS:
                            log_metric(
                                "fast_scalp_skip",
                                float(pips_gain),
                                tags={"reason": "exit_ignore", "exit": reason_key},
                                ts=now,
                            )
                            close_reason = None
                        elif config.EXIT_MIN_LOSS_PIPS > 0 and pips_gain > -config.EXIT_MIN_LOSS_PIPS:
                            log_metric(
                                "fast_scalp_skip",
                                float(pips_gain),
                                tags={"reason": "exit_min_loss", "exit": reason_key},
                                ts=now,
                            )
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
                                POCKET,
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
                                await close_trade(
                                    trade_id,
                                    client_order_id=active.client_order_id,
                                    exit_reason=close_reason,
                                )
                            finally:
                                summary_payload = timeout_controller.finalize(
                                    trade_id,
                                    reason=close_reason,
                                    pips_gain=pips_gain,
                                    tick_rate=tick_rate,
                                    spread_pips=features.spread_pips,
                                )
                                stage_tracker.set_cooldown(
                                    POCKET,
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
                    pocket = positions.get(POCKET)
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

            if (
                last_tick_age_ms is not None
                and last_tick_age_ms > config.MAX_SIGNAL_AGE_MS
            ):
                log_metric(
                    "fast_scalp_skip",
                    float(last_tick_age_ms),
                    tags={"reason": "signal_stale"},
                    ts=now,
                )
                logger.info(
                    "%s skip due to stale tick age_ms=%.0f limit=%.0f",
                    config.LOG_PREFIX_TICK,
                    last_tick_age_ms,
                    config.MAX_SIGNAL_AGE_MS,
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            if config.FAST_SCALP_RANGE_ONLY and not range_active_flag:
                log_metric(
                    "fast_scalp_skip",
                    1.0,
                    tags={"reason": "range_only"},
                    ts=now,
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            if config.FAST_SCALP_PATTERN_ALLOWLIST:
                if features.pattern_tag.lower() not in config.FAST_SCALP_PATTERN_ALLOWLIST:
                    log_metric(
                        "fast_scalp_skip",
                        1.0,
                        tags={"reason": "pattern_allowlist", "pattern": features.pattern_tag},
                        ts=now,
                    )
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue
            if config.FAST_SCALP_PATTERN_BLOCKLIST:
                if features.pattern_tag.lower() in config.FAST_SCALP_PATTERN_BLOCKLIST:
                    log_metric(
                        "fast_scalp_skip",
                        1.0,
                        tags={"reason": "pattern_blocklist", "pattern": features.pattern_tag},
                        ts=now,
                    )
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

            action = evaluate_signal(
                features,
                m1_rsi=m1_rsi_snapshot,
                range_active=range_active_flag,
            )
            if not action:
                span = max(features.span_seconds, 1e-6)
                velocity = abs(features.short_momentum_pips) / span
                density = features.tick_count / span
                logger.info(
                    "%s no_action range=%.3f mom=%.3f short_mom=%.3f atr=%.3f spread=%.3f ticks=%d span=%.2f vel=%.3f dens=%.3f range_active=%s",
                    config.LOG_PREFIX_TICK,
                    features.range_pips,
                    features.momentum_pips,
                    features.short_momentum_pips,
                    features.atr_pips if features.atr_pips is not None else -1.0,
                    features.spread_pips,
                    features.tick_count,
                    features.span_seconds,
                    velocity,
                    density,
                    range_active_flag,
                )
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

            cooldown = stage_tracker.get_cooldown(POCKET, direction, now=now)
            if cooldown and not reversal:
                logger.debug(
                    "%s cooldown active pocket=%s dir=%s until=%s reason=%s",
                    config.LOG_PREFIX_TICK,
                    POCKET,
                    direction,
                    cooldown.cooldown_until,
                    cooldown.reason,
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            if cooldown and reversal:
                stage_tracker.clear_cooldown(POCKET, direction, reason=cooldown.reason)

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
            fac_m1 = {}
            fac_h4 = {}
            try:
                factors = all_factors()
                fac_m1 = factors.get("M1") or {}
                fac_h4 = factors.get("H4") or {}
            except Exception:
                fac_m1 = {}
                fac_h4 = {}
            air = evaluate_air(fac_m1, fac_h4, range_active=range_active_flag, tag=config.STRATEGY_TAG)
            if air.enabled and not air.allow_entry:
                log_metric("fast_scalp_skip", 1.0, tags={"reason": "air_block", "pref": air.range_pref}, ts=now)
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            if config.FIXED_UNITS and abs(config.FIXED_UNITS) >= config.MIN_UNITS:
                units = abs(int(config.FIXED_UNITS))
                units = units if direction == "long" else -units
                allowed_lot_raw = units / 100000.0
                lot = allowed_lot_raw
            else:
                long_units = 0.0
                short_units = 0.0
                try:
                    long_units, short_units = get_position_summary("USD_JPY", timeout=3.0)
                except Exception:
                    long_units, short_units = 0.0, 0.0
                strategy_tag = config.STRATEGY_TAG
                allowed_lot_raw = allowed_lot(
                    equity,
                    sl_pips=profile_sl_pips,
                    margin_available=margin_available,
                    price=features.latest_mid,
                    margin_rate=margin_rate,
                    risk_pct_override=snapshot.risk_pct_override,
                    pocket="scalp",
                    side=direction,
                    open_long_units=long_units,
                    open_short_units=short_units,
                    strategy_tag=strategy_tag,
                    fac_m1=fac_m1,
                    fac_h4=fac_h4,
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
                margin_rate > 0.0
                and margin_available > 0.0
                and 0.0 < config.MAX_MARGIN_USAGE < 1.0
            ):
                margin_budget = margin_available * config.MAX_MARGIN_USAGE
                margin_per_unit = features.latest_mid * margin_rate
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
            if air.enabled:
                units = int(round(units * air.size_mult))
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
            last_tick = recent_tick
            if last_tick is None:
                last_ticks = tick_window.recent_ticks(3.0, limit=1)
                if last_ticks:
                    last_tick = last_ticks[-1]
            if last_tick:
                bid_quote = float(last_tick.get("bid") or bid_quote or features.latest_mid)
                ask_quote = float(last_tick.get("ask") or ask_quote or features.latest_mid)
                if latest_tick_mid is None:
                    try:
                        latest_tick_mid = float(last_tick.get("mid") or 0.0)
                    except Exception:
                        latest_tick_mid = None
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

            current_mid = latest_tick_mid
            if current_mid is None and bid_quote and ask_quote:
                current_mid = (bid_quote + ask_quote) / 2.0
            drift_pips = 0.0
            if current_mid is not None:
                drift_pips = abs(current_mid - features.latest_mid) / config.PIP_VALUE
            drift_limit = max(config.MAX_DRIFT_PIPS, tp_pips * config.DRIFT_TP_RATIO)
            if drift_limit > 0.0 and drift_pips > drift_limit:
                log_metric(
                    "fast_scalp_skip",
                    float(drift_pips),
                    tags={"reason": "price_drift", "side": direction},
                    ts=now,
                )
                logger.info(
                    "%s skip due to price drift drift=%.3fp limit=%.3fp mid_now=%.5f signal_mid=%.5f",
                    config.LOG_PREFIX_TICK,
                    drift_pips,
                    drift_limit,
                    current_mid if current_mid is not None else -1.0,
                    features.latest_mid,
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            sl_price_initial = None if sl_price is None else round(sl_price, 5)
            tp_price_initial = None if tp_price is None else round(tp_price, 5)

            thesis = {
                "strategy_tag": config.STRATEGY_TAG,
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
                "sl_pips": None if sl_pips is None else round(sl_pips, 3),
                "hard_stop_pips": None if sl_pips is None else round(sl_pips, 3),
                "entry_price_expect": round(entry_price, 5),
                "sl_price_initial": sl_price_initial,
                "tp_price_initial": tp_price_initial,
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
                "signal_age_ms": None if last_tick_age_ms is None else float(last_tick_age_ms),
                "price_drift_pips": round(drift_pips, 4),
                "air_score": round(air.air_score, 3) if air.enabled else None,
                "air_pressure": round(air.pressure_score, 3) if air.enabled else None,
                "air_pressure_dir": air.pressure_dir if air.enabled else None,
                "air_spread_state": air.spread_state if air.enabled else None,
                "air_exec_quality": round(air.exec_quality, 3) if air.enabled else None,
                "air_regime_shift": round(air.regime_shift, 3) if air.enabled else None,
                "air_range_pref": air.range_pref if air.enabled else None,
            }

            fac_m1 = all_factors().get("M1") or {}
            if not _bb_entry_allowed(BB_STYLE, direction, entry_price, fac_m1, range_active=range_active_flag):
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            proj_allow, proj_mult, proj_detail = _projection_decision(direction, POCKET)
            if not proj_allow:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            if proj_detail:
                thesis["projection"] = proj_detail
            if proj_mult > 1.0:
                sign = 1 if units > 0 else -1
                units = int(round(abs(units) * proj_mult)) * sign

            try:
                candle_allow, candle_mult = _entry_candle_guard("long" if units > 0 else "short")
                if not candle_allow:
                    continue
                if candle_mult != 1.0:
                    sign = 1 if units > 0 else -1
                    units = int(round(abs(units) * candle_mult)) * sign
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl_price,
                    tp_price,
                    POCKET,
                    strategy_tag=config.STRATEGY_TAG,
                    client_order_id=client_id,
                    entry_thesis={**thesis, "strategy_tag": config.STRATEGY_TAG},
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
                POCKET,
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



_BB_ENTRY_ENABLED = os.getenv("BB_ENTRY_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
_BB_ENTRY_REVERT_PIPS = float(os.getenv("BB_ENTRY_REVERT_PIPS", "2.4"))
_BB_ENTRY_REVERT_RATIO = float(os.getenv("BB_ENTRY_REVERT_RATIO", "0.22"))
_BB_ENTRY_TREND_EXT_PIPS = float(os.getenv("BB_ENTRY_TREND_EXT_PIPS", "3.5"))
_BB_ENTRY_TREND_EXT_RATIO = float(os.getenv("BB_ENTRY_TREND_EXT_RATIO", "0.40"))
_BB_ENTRY_SCALP_REVERT_PIPS = float(os.getenv("BB_ENTRY_SCALP_REVERT_PIPS", "2.0"))
_BB_ENTRY_SCALP_REVERT_RATIO = float(os.getenv("BB_ENTRY_SCALP_REVERT_RATIO", "0.20"))
_BB_ENTRY_SCALP_EXT_PIPS = float(os.getenv("BB_ENTRY_SCALP_EXT_PIPS", "2.4"))
_BB_ENTRY_SCALP_EXT_RATIO = float(os.getenv("BB_ENTRY_SCALP_EXT_RATIO", "0.30"))
_BB_PIP = 0.01


def _bb_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bb_levels(fac):
    if not fac:
        return None
    upper = _bb_float(fac.get("bb_upper"))
    lower = _bb_float(fac.get("bb_lower"))
    mid = _bb_float(fac.get("bb_mid")) or _bb_float(fac.get("ma20"))
    bbw = _bb_float(fac.get("bbw")) or 0.0
    if upper is None or lower is None:
        if mid is None or bbw <= 0:
            return None
        half = abs(mid) * bbw / 2.0
        upper = mid + half
        lower = mid - half
    span = upper - lower
    if span <= 0:
        return None
    return upper, mid if mid is not None else (upper + lower) / 2.0, lower, span, span / _BB_PIP


def _bb_entry_allowed(style, side, price, fac_m1, *, range_active=None):
    if not _BB_ENTRY_ENABLED:
        return True
    if price is None or price <= 0:
        return True
    levels = _bb_levels(fac_m1)
    if not levels:
        return True
    upper, mid, lower, span, span_pips = levels
    side_key = str(side or "").lower()
    if side_key in {"buy", "long", "open_long"}:
        direction = "long"
    else:
        direction = "short"
    orig_style = style
    if style == "scalp" and range_active:
        style = "reversion"
    if style == "reversion":
        base_pips = _BB_ENTRY_SCALP_REVERT_PIPS if orig_style == "scalp" else _BB_ENTRY_REVERT_PIPS
        base_ratio = _BB_ENTRY_SCALP_REVERT_RATIO if orig_style == "scalp" else _BB_ENTRY_REVERT_RATIO
        threshold = max(base_pips, span_pips * base_ratio)
        if direction == "long":
            dist = (price - lower) / _BB_PIP
        else:
            dist = (upper - price) / _BB_PIP
        return dist <= threshold
    if direction == "long":
        if price < mid:
            return False
        ext = max(0.0, price - upper) / _BB_PIP
    else:
        if price > mid:
            return False
        ext = max(0.0, lower - price) / _BB_PIP
    max_ext = max(_BB_ENTRY_TREND_EXT_PIPS, span_pips * _BB_ENTRY_TREND_EXT_RATIO)
    if orig_style == "scalp":
        max_ext = max(_BB_ENTRY_SCALP_EXT_PIPS, span_pips * _BB_ENTRY_SCALP_EXT_RATIO)
    return ext <= max_ext

BB_STYLE = "scalp"


_CANDLE_PIP = 0.01
_CANDLE_MIN_CONF = 0.35
_CANDLE_ENTRY_BLOCK = -0.7
_CANDLE_ENTRY_SCALE = 0.2
_CANDLE_ENTRY_MIN = 0.8
_CANDLE_ENTRY_MAX = 1.2
_CANDLE_WORKER_NAME = (__file__.replace("\\", "/").split("/")[-2] if "/" in __file__ else "").lower()


def _candle_tf_for_worker() -> str:
    name = _CANDLE_WORKER_NAME
    if "macro" in name or "trend_h1" in name or "manual" in name:
        return "H1"
    if "scalp" in name or "s5" in name or "fast" in name:
        return "M1"
    return "M5"


def _extract_candles(raw):
    candles = []
    for candle in raw or []:
        try:
            o = float(candle.get("open", candle.get("o")))
            h = float(candle.get("high", candle.get("h")))
            l = float(candle.get("low", candle.get("l")))
            c = float(candle.get("close", candle.get("c")))
        except Exception:
            continue
        if h <= 0 or l <= 0:
            continue
        candles.append((o, h, l, c))
    return candles


def _detect_candlestick_pattern(candles):
    if len(candles) < 2:
        return None
    o0, h0, l0, c0 = candles[-2]
    o1, h1, l1, c1 = candles[-1]
    body0 = abs(c0 - o0)
    body1 = abs(c1 - o1)
    range1 = max(h1 - l1, _CANDLE_PIP * 0.1)
    upper_wick = h1 - max(o1, c1)
    lower_wick = min(o1, c1) - l1

    if body1 <= range1 * 0.1:
        return {
            "type": "doji",
            "confidence": round(min(1.0, (range1 - body1) / range1), 3),
            "bias": None,
        }

    if (
        c1 > o1
        and c0 < o0
        and c1 >= max(o0, c0)
        and o1 <= min(o0, c0)
        and body1 > body0
    ):
        return {
            "type": "bullish_engulfing",
            "confidence": round(min(1.0, body1 / range1 + 0.3), 3),
            "bias": "up",
        }
    if (
        c1 < o1
        and c0 > o0
        and o1 >= min(o0, c0)
        and c1 <= max(o0, c0)
        and body1 > body0
    ):
        return {
            "type": "bearish_engulfing",
            "confidence": round(min(1.0, body1 / range1 + 0.3), 3),
            "bias": "down",
        }
    if lower_wick > body1 * 2.5 and upper_wick <= body1 * 0.6:
        return {
            "type": "hammer" if c1 >= o1 else "inverted_hammer",
            "confidence": round(min(1.0, lower_wick / range1 + 0.25), 3),
            "bias": "up",
        }
    if upper_wick > body1 * 2.5 and lower_wick <= body1 * 0.6:
        return {
            "type": "shooting_star" if c1 <= o1 else "hanging_man",
            "confidence": round(min(1.0, upper_wick / range1 + 0.25), 3),
            "bias": "down",
        }
    return None


def _score_candle(*, candles, side, min_conf):
    pattern = _detect_candlestick_pattern(_extract_candles(candles))
    if not pattern:
        return None, {}
    bias = pattern.get("bias")
    conf = float(pattern.get("confidence") or 0.0)
    if conf < min_conf:
        return None, {"type": pattern.get("type"), "confidence": round(conf, 3)}
    if bias is None:
        return 0.0, {"type": pattern.get("type"), "confidence": round(conf, 3), "bias": None}
    match = (side == "long" and bias == "up") or (side == "short" and bias == "down")
    score = conf if match else -conf * 0.7
    score = max(-1.0, min(1.0, score))
    return score, {"type": pattern.get("type"), "confidence": round(conf, 3), "bias": bias}


def _entry_candle_guard(side):
    tf = _candle_tf_for_worker()
    candles = get_candles_snapshot(tf, limit=4)
    if not candles:
        return True, 1.0
    score, _detail = _score_candle(candles=candles, side=side, min_conf=_CANDLE_MIN_CONF)
    if score is None:
        return True, 1.0
    if score <= _CANDLE_ENTRY_BLOCK:
        return False, 0.0
    mult = 1.0 + score * _CANDLE_ENTRY_SCALE
    mult = max(_CANDLE_ENTRY_MIN, min(_CANDLE_ENTRY_MAX, mult))
    return True, mult

if __name__ == "__main__":  # pragma: no cover
    _main()
