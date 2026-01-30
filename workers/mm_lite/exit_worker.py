from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence, Set

from analysis.range_guard import detect_range_mode
from execution.position_manager import PositionManager
from indicators.factor_cache import all_factors
from market_data import tick_window
from workers.common.exit_utils import close_trade, mark_pnl_pips
from workers.common.reentry_decider import decide_reentry


_BB_EXIT_ENABLED = os.getenv("BB_EXIT_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
_BB_EXIT_REVERT_PIPS = float(os.getenv("BB_EXIT_REVERT_PIPS", "2.0"))
_BB_EXIT_REVERT_RATIO = float(os.getenv("BB_EXIT_REVERT_RATIO", "0.20"))
_BB_EXIT_TREND_EXT_PIPS = float(os.getenv("BB_EXIT_TREND_EXT_PIPS", "3.0"))
_BB_EXIT_TREND_EXT_RATIO = float(os.getenv("BB_EXIT_TREND_EXT_RATIO", "0.35"))
_BB_EXIT_SCALP_REVERT_PIPS = float(os.getenv("BB_EXIT_SCALP_REVERT_PIPS", "1.6"))
_BB_EXIT_SCALP_REVERT_RATIO = float(os.getenv("BB_EXIT_SCALP_REVERT_RATIO", "0.18"))
_BB_EXIT_SCALP_EXT_PIPS = float(os.getenv("BB_EXIT_SCALP_EXT_PIPS", "2.0"))
_BB_EXIT_SCALP_EXT_RATIO = float(os.getenv("BB_EXIT_SCALP_EXT_RATIO", "0.28"))
_BB_EXIT_MID_BUFFER_PIPS = float(os.getenv("BB_EXIT_MID_BUFFER_PIPS", "0.4"))
_BB_EXIT_BYPASS_TOKENS = {
    "hard_stop",
    "structure",
    "time_stop",
    "timeout",
    "max_hold",
    "max_adverse",
    "force",
    "margin",
    "risk",
    "health",
    "event",
    "session",
    "halt",
    "liquid",
}
_BB_EXIT_TF = "M1"
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


def _bb_exit_price(fac):
    price = None
    latest_mid = globals().get("_latest_mid")
    if callable(latest_mid):
        try:
            price = latest_mid()
        except Exception:
            price = None
    if price is None:
        latest_quote = globals().get("_latest_quote")
        if callable(latest_quote):
            try:
                _, _, mid = latest_quote()
                price = mid
            except Exception:
                price = None
    if price is None or price <= 0:
        try:
            price = float(fac.get("close") or 0.0)
        except Exception:
            price = None
    return price


def _bb_exit_should_bypass(reason, pnl, allow_negative):
    if allow_negative:
        return True
    if pnl is not None:
        try:
            if float(pnl) <= 0:
                return True
        except Exception:
            pass
    if not reason:
        return False
    reason_key = str(reason).lower()
    for token in _BB_EXIT_BYPASS_TOKENS:
        if token in reason_key:
            return True
    return False


def _bb_exit_allowed(style, side, price, fac, *, range_active=None):
    if not _BB_EXIT_ENABLED:
        return True
    if price is None or price <= 0:
        return True
    levels = _bb_levels(fac)
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
    mid_buffer = max(_BB_EXIT_MID_BUFFER_PIPS, span_pips * 0.05)
    if style == "reversion":
        base_pips = _BB_EXIT_SCALP_REVERT_PIPS if orig_style == "scalp" else _BB_EXIT_REVERT_PIPS
        base_ratio = _BB_EXIT_SCALP_REVERT_RATIO if orig_style == "scalp" else _BB_EXIT_REVERT_RATIO
        threshold = max(base_pips, span_pips * base_ratio)
        if direction == "long":
            dist = (price - lower) / _BB_PIP
        else:
            dist = (upper - price) / _BB_PIP
        return dist >= threshold
    band_buffer = max(_BB_EXIT_TREND_EXT_PIPS, span_pips * _BB_EXIT_TREND_EXT_RATIO)
    if orig_style == "scalp":
        band_buffer = max(_BB_EXIT_SCALP_EXT_PIPS, span_pips * _BB_EXIT_SCALP_EXT_RATIO)
    if direction == "long":
        if price <= mid - mid_buffer * _BB_PIP:
            return True
        return price >= (upper - band_buffer * _BB_PIP)
    if price >= mid + mid_buffer * _BB_PIP:
        return True
    return price <= (lower + band_buffer * _BB_PIP)

BB_STYLE = "reversion"
LOG = logging.getLogger(__name__)

ALLOWED_TAGS: Set[str] = {'mm_lite'}


def _tags_env(key: str, default: Set[str]) -> Set[str]:
    raw = os.getenv(key)
    if raw is None:
        return default
    tags = {t.strip() for t in raw.replace(";", ",").split(",") if t.strip()}
    return tags or default


def _pocket_env(key: str, default: str) -> str:
    raw = os.getenv(key)
    return raw.strip().lower() if raw else default


POCKET = _pocket_env("MM_LITE_POCKET", "scalp")
ALLOWED_TAGS = _tags_env("MM_LITE_EXIT_TAGS", ALLOWED_TAGS)


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _latest_mid() -> Optional[float]:
    tick = tick_window.recent_ticks(seconds=2.0, limit=1)
    if tick:
        try:
            return float(tick[-1]["mid"])
        except Exception:
            pass
    try:
        return float(all_factors().get("M1", {}).get("close"))
    except Exception:
        return None


def _filter_trades(trades: Sequence[dict], tags: Set[str]) -> list[dict]:
    if not tags:
        return []
    filtered: list[dict] = []
    for tr in trades:
        thesis = tr.get("entry_thesis") or {}
        tag = (
            thesis.get("strategy_tag")
            or thesis.get("strategy_tag_raw")
            or thesis.get("strategy")
            or thesis.get("tag")
            or tr.get("strategy_tag")
            or tr.get("strategy")
        )
        if not tag:
            continue
        tag_str = str(tag)
        base_tag = tag_str.split("-", 1)[0]
        if tag_str in tags or base_tag in tags:
            filtered.append(tr)
    return filtered


@dataclass
class _TradeState:
    peak: float
    lock_floor: Optional[float] = None

    def update(self, pnl: float, lock_buffer: float) -> None:
        if pnl > self.peak:
            self.peak = pnl
        if pnl > 0:
            floor = max(0.0, pnl - lock_buffer)
            self.lock_floor = floor if self.lock_floor is None else max(self.lock_floor, floor)


class MMLiteExitWorker:
    def __init__(self) -> None:
        self.loop_interval = max(
            0.5,
            _float_env("MM_LITE_EXIT_LOOP_INTERVAL_SEC", 0.8),
        )
        self.min_hold_sec = max(
            5.0,
            _float_env("MM_LITE_EXIT_MIN_HOLD_SEC", 10.0),
        )
        self.profit_take = max(
            0.5,
            _float_env("MM_LITE_EXIT_PROFIT_PIPS", 1.2),
        )
        self.trail_start = max(
            0.5,
            _float_env("MM_LITE_EXIT_TRAIL_START_PIPS", 1.6),
        )
        self.trail_backoff = max(
            0.1,
            _float_env("MM_LITE_EXIT_TRAIL_BACKOFF_PIPS", 0.5),
        )
        self.lock_buffer = max(
            0.05,
            _float_env("MM_LITE_EXIT_LOCK_BUFFER_PIPS", 0.3),
        )
        self.range_profit_take = max(
            0.4,
            _float_env("MM_LITE_EXIT_RANGE_PROFIT_PIPS", 1.0),
        )
        self.range_trail_start = max(
            0.4,
            _float_env("MM_LITE_EXIT_RANGE_TRAIL_START_PIPS", 1.3),
        )
        self.range_trail_backoff = max(
            0.1,
            _float_env("MM_LITE_EXIT_RANGE_TRAIL_BACKOFF_PIPS", 0.4),
        )
        self.range_lock_buffer = max(
            0.05,
            _float_env("MM_LITE_EXIT_RANGE_LOCK_BUFFER_PIPS", 0.25),
        )
        self.range_max_hold_sec = max(
            60.0,
            _float_env("MM_LITE_EXIT_RANGE_MAX_HOLD_SEC", 1200.0),
        )


        self._pos_manager = PositionManager()
        self._states: dict[str, _TradeState] = {}

    def _context(self) -> tuple[Optional[float], bool]:
        fac_m1 = all_factors().get("M1") or {}
        fac_h4 = all_factors().get("H4") or {}
        range_active = False
        try:
            range_active = bool(detect_range_mode(fac_m1, fac_h4).active)
        except Exception:
            range_active = False
        return _latest_mid(), range_active

    async def _close(self, trade_id: str, units: int, reason: str, pnl: float, client_order_id: Optional[str]) -> None:
        if _BB_EXIT_ENABLED:
            allow_neg = bool(locals().get("allow_negative"))
            pnl_val = locals().get("pnl")
            if not _bb_exit_should_bypass(reason, pnl_val, allow_neg):
                fac = all_factors().get(_BB_EXIT_TF) or {}
                price = _bb_exit_price(fac)
                side = "long" if units > 0 else "short"
                if not _bb_exit_allowed(BB_STYLE, side, price, fac):
                    LOG.info("[exit-bb] trade=%s reason=%s price=%.3f", trade_id, reason, price or 0.0)
                    return
        ok = await close_trade(
            trade_id,
            units,
            client_order_id=client_order_id,
            allow_negative=False,
            exit_reason=reason,
        )
        if ok:
            LOG.info("[exit-mm-lite] trade=%s units=%s reason=%s pnl=%.2fp", trade_id, units, reason, pnl)
        else:
            LOG.error("[exit-mm-lite] close failed trade=%s units=%s reason=%s", trade_id, units, reason)

    async def _review_trade(self, trade: dict, now: datetime, mid: float, range_active: bool) -> None:
        trade_id = str(trade.get("trade_id"))
        if not trade_id:
            return
        units = int(trade.get("units", 0) or 0)
        if units == 0:
            return
        entry = float(trade.get("price") or 0.0)
        if entry <= 0.0:
            return

        side = "long" if units > 0 else "short"
        pnl = mark_pnl_pips(entry, units, mid=mid)
        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0

        client_ext = trade.get("clientExtensions")
        client_id = trade.get("client_order_id")
        if not client_id and isinstance(client_ext, dict):
            client_id = client_ext.get("id")
        if not client_id:
            LOG.warning("[exit-mm-lite] missing client_id trade=%s skip close", trade_id)
            return

        if hold_sec < self.min_hold_sec:
            return
        candle_reason = _exit_candle_reversal("long" if units > 0 else "short")
        if candle_reason and pnl >= 0:
            candle_client_id = trade.get("client_order_id")
            if not candle_client_id:
                client_ext = trade.get("clientExtensions")
                if isinstance(client_ext, dict):
                    candle_client_id = client_ext.get("id")
            if candle_client_id:
                await self._close(trade_id, -units, candle_reason, pnl, candle_client_id)
                if hasattr(self, "_states"):
                    self._states.pop(trade_id, None)
                return
        if pnl <= 0:
            fac_m1 = all_factors().get("M1") or {}
            rsi = _bb_float(fac_m1.get("rsi"))
            adx = _bb_float(fac_m1.get("adx"))
            atr_pips = _bb_float(fac_m1.get("atr_pips"))
            bbw = _bb_float(fac_m1.get("bbw"))
            vwap_gap = _bb_float(fac_m1.get("vwap_gap"))
            ma10 = _bb_float(fac_m1.get("ma10"))
            ma20 = _bb_float(fac_m1.get("ma20"))
            ma_pair = (ma10, ma20) if ma10 is not None and ma20 is not None else None
            reentry = decide_reentry(
                prefix="MM_LITE",
                side=side,
                pnl_pips=pnl,
                rsi=rsi,
                adx=adx,
                atr_pips=atr_pips,
                bbw=bbw,
                vwap_gap=vwap_gap,
                ma_pair=ma_pair,
                range_active=range_active,
                log_tags={"trade": trade_id},
            )
            if reentry.action == "hold":
                return
            if reentry.action == "exit_reentry" and not reentry.shadow:
                await self._close(trade_id, -units, "reentry_reset", pnl, client_id)
                self._states.pop(trade_id, None)
                return
            return

        lock_buffer = self.range_lock_buffer if range_active else self.lock_buffer
        profit_take = self.range_profit_take if range_active else self.profit_take
        trail_start = self.range_trail_start if range_active else self.trail_start
        trail_backoff = self.range_trail_backoff if range_active else self.trail_backoff

        state = self._states.get(trade_id)
        if state is None:
            state = _TradeState(peak=pnl)
            self._states[trade_id] = state
        state.update(pnl, lock_buffer)

        if pnl >= trail_start:
            candidate = max(0.0, pnl - trail_backoff)
            state.lock_floor = candidate if state.lock_floor is None else max(state.lock_floor, candidate)

        if pnl >= profit_take:
            await self._close(trade_id, -units, "take_profit", pnl, client_id)
            self._states.pop(trade_id, None)
            return

        if state.lock_floor is not None and pnl <= state.lock_floor:
            await self._close(trade_id, -units, "lock_floor", pnl, client_id)
            self._states.pop(trade_id, None)
            return

        if range_active and hold_sec >= self.range_max_hold_sec:
            await self._close(trade_id, -units, "range_timeout", pnl, client_id)
            self._states.pop(trade_id, None)

    async def run(self) -> None:
        LOG.info(
            "[exit-mm-lite] exit worker start interval=%.2fs tags=%s pocket=%s",
            self.loop_interval,
            ",".join(sorted(ALLOWED_TAGS)) if ALLOWED_TAGS else "none",
            POCKET,
        )
        if not ALLOWED_TAGS:
            LOG.info("[exit-mm-lite] no allowed tags configured; idle")
            try:
                while True:
                    await asyncio.sleep(3600.0)
            except asyncio.CancelledError:
                return
        try:
            while True:
                await asyncio.sleep(self.loop_interval)
                positions = self._pos_manager.get_open_positions()
                pocket_info = positions.get(POCKET) or {}
                trades = _filter_trades(pocket_info.get("open_trades") or [], ALLOWED_TAGS)
                active_ids = {str(tr.get("trade_id")) for tr in trades if tr.get("trade_id")}
                for tid in list(self._states.keys()):
                    if tid not in active_ids:
                        self._states.pop(tid, None)
                if not trades:
                    continue

                mid, range_active = self._context()
                if mid is None:
                    continue
                now = datetime.now(timezone.utc)
                for tr in trades:
                    try:
                        await self._review_trade(tr, now, mid, range_active)
                    except Exception:
                        LOG.exception("[exit-mm-lite] review failed trade=%s", tr.get("trade_id"))
        except asyncio.CancelledError:
            LOG.info("[exit-mm-lite] worker cancelled")
            raise
        finally:
            try:
                self._pos_manager.close()
            except Exception:
                LOG.exception("[exit-mm-lite] failed to close PositionManager")


async def mm_lite_exit_worker() -> None:
    worker = MMLiteExitWorker()
    await worker.run()


_CANDLE_PIP = 0.01
_CANDLE_EXIT_MIN_CONF = 0.35
_CANDLE_EXIT_SCORE = -0.5
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


def _exit_candle_reversal(side):
    tf = _candle_tf_for_worker()
    candles = (all_factors().get(tf) or {}).get("candles") or []
    if not candles:
        return None
    score, detail = _score_candle(candles=candles, side=side, min_conf=_CANDLE_EXIT_MIN_CONF)
    if score is None:
        return None
    if score <= _CANDLE_EXIT_SCORE:
        detail_type = detail.get("type") if isinstance(detail, dict) else None
        return f"candle_{detail_type}" if detail_type else "candle_reversal"
    return None
if __name__ == "__main__":
    asyncio.run(mm_lite_exit_worker())


