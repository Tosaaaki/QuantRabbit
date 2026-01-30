"""Exit loop for mirror_spike worker (scalp pocket)."""

from __future__ import annotations
import os

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Set

from workers.common.exit_utils import close_trade, mark_pnl_pips
from workers.common.reentry_decider import decide_reentry
from execution.position_manager import PositionManager
from execution.reversion_failure import evaluate_reversion_failure, evaluate_tp_zone
from indicators.factor_cache import all_factors
from market_data import tick_window
from utils.metrics_logger import log_metric


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

ALLOWED_TAGS = {"mirror_spike"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
        return list(trades)
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


def _client_id(trade: dict) -> Optional[str]:
    client_ext = trade.get("clientExtensions")
    client_id = trade.get("client_order_id")
    if not client_id and isinstance(client_ext, dict):
        client_id = client_ext.get("id")
    return client_id


@dataclass
class _TradeState:
    peak: float
    lock_floor: Optional[float] = None
    hard_stop: Optional[float] = None
    tp_hint: Optional[float] = None
    trend_hits: int = 0

    def update(self, pnl: float, lock_buffer: float) -> None:
        if pnl > self.peak:
            self.peak = pnl
        if self.lock_floor is None and pnl > 0:
            self.lock_floor = max(0.0, pnl - lock_buffer)


@dataclass
class _ExitParams:
    profit_take: float
    trail_start: float
    trail_backoff: float
    stop_loss: float
    max_hold_sec: float
    lock_buffer: float
    use_entry_meta: bool = True
    structure_break: bool = False
    structure_timeframe: str = "M1"
    structure_adx: float = 22.0
    structure_gap_pips: float = 2.0
    structure_adx_cold: float = 12.0
    virtual_sl_ratio: float = 0.72
    trail_from_tp_ratio: float = 0.82
    lock_from_tp_ratio: float = 0.45
    tp_floor_ratio: float = 1.0


async def _run_exit_loop(
    *,
    pocket: str,
    tags: Set[str],
    params: _ExitParams,
    loop_interval: float,
) -> None:
    pos_manager = PositionManager()
    states: Dict[str, _TradeState] = {}

    async def _close(
        trade_id: str,
        units: int,
        reason: str,
        client_id: str,
        allow_negative: bool = False,
    ) -> bool:
        if _BB_EXIT_ENABLED:
            allow_neg = bool(locals().get("allow_negative"))
            pnl_val = locals().get("pnl")
            if not _bb_exit_should_bypass(reason, pnl_val, allow_neg):
                fac = all_factors().get(_BB_EXIT_TF) or {}
                price = _bb_exit_price(fac)
                side = "long" if units > 0 else "short"
                if not _bb_exit_allowed(BB_STYLE, side, price, fac):
                    LOG.info("[exit-bb] trade=%s reason=%s price=%.3f", trade_id, reason, price or 0.0)
                    return False
        ok = await close_trade(
            trade_id,
            units,
            client_order_id=client_id,
            allow_negative=allow_negative,
            exit_reason=reason,
        )
        if ok:
            LOG.info("[EXIT-%s] trade=%s units=%s reason=%s", pocket, trade_id, units, reason)
        else:
            LOG.error("[EXIT-%s] close failed trade=%s units=%s reason=%s", pocket, trade_id, units, reason)
        return ok

    def _structure_break(units: int) -> bool:
        if not params.structure_break:
            return False
        try:
            factors = all_factors().get(params.structure_timeframe) or {}
            adx = float(factors.get("adx"))
            ma10 = float(factors.get("ma10"))
            ma20 = float(factors.get("ma20"))
            gap = abs(ma10 - ma20) / 0.01

            if adx < params.structure_adx:
                dir_long = units > 0
                if (dir_long and ma10 <= ma20) or ((not dir_long) and ma10 >= ma20) or (
                    adx < params.structure_adx_cold and gap < params.structure_gap_pips
                ):
                    return True
        except Exception:
            return False
        return False

    async def _review_trade(trade: dict, now: datetime) -> None:
        trade_id = str(trade.get("trade_id"))
        if not trade_id:
            return
        units = int(trade.get("units", 0) or 0)
        if units == 0:
            return

        price_entry = float(trade.get("price") or 0.0)
        if price_entry <= 0.0:
            return

        side = "long" if units > 0 else "short"
        current = _latest_mid()
        if current is None:
            return
        pnl = mark_pnl_pips(price_entry, units, mid=current)
        allow_negative = pnl <= 0

        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0

        state = states.get(trade_id)
        if state is None:
            thesis = trade.get("entry_thesis") or {}
            hard_stop = thesis.get("hard_stop_pips")
            tp_hint = thesis.get("tp_pips")
            try:
                hard_stop_val = float(hard_stop) if hard_stop is not None else None
            except Exception:
                hard_stop_val = None
            try:
                tp_hint_val = float(tp_hint) if tp_hint is not None else None
            except Exception:
                tp_hint_val = None
            state = _TradeState(peak=pnl, hard_stop=hard_stop_val, tp_hint=tp_hint_val)
            states[trade_id] = state

        profit_take = params.profit_take
        trail_start = params.trail_start
        stop_loss = params.stop_loss
        max_hold = params.max_hold_sec
        lock_buffer = params.lock_buffer

        if params.use_entry_meta:
            if state.tp_hint:
                profit_take = max(profit_take, max(1.0, state.tp_hint * params.tp_floor_ratio))
                trail_start = max(trail_start, max(1.0, profit_take * params.trail_from_tp_ratio))
                lock_buffer = max(lock_buffer, profit_take * params.lock_from_tp_ratio)
            if state.hard_stop:
                stop_loss = max(stop_loss, max(0.8, state.hard_stop * 0.5))
                lock_buffer = max(lock_buffer, stop_loss * 0.35)
                trail_start = max(trail_start, max(1.0, state.hard_stop * 0.6))
                max_hold = max(max_hold, params.max_hold_sec * 1.05)

        stop_loss = max(stop_loss, profit_take * params.virtual_sl_ratio)
        trail_start = max(trail_start, profit_take * params.trail_from_tp_ratio)
        lock_buffer = max(lock_buffer, profit_take * params.lock_from_tp_ratio)

        state.update(pnl, lock_buffer)

        client_id = _client_id(trade)
        if not client_id:
            LOG.warning("[EXIT-%s] missing client_id trade=%s skip close", pocket, trade_id)
            return

        if pnl < 0:
            fac_m1 = all_factors().get("M1") or {}
            def _opt(val):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
            rsi = _opt(fac_m1.get("rsi"))
            adx = _opt(fac_m1.get("adx"))
            atr_pips = _opt(fac_m1.get("atr_pips"))
            bbw = _opt(fac_m1.get("bbw"))
            vwap_gap = _opt(fac_m1.get("vwap_gap"))
            ma10 = _opt(fac_m1.get("ma10"))
            ma20 = _opt(fac_m1.get("ma20"))
            ma_pair = (ma10, ma20) if ma10 is not None and ma20 is not None else None
            reentry = decide_reentry(
                prefix="MIRROR_SPIKE",
                side=side,
                pnl_pips=pnl,
                rsi=rsi,
                adx=adx,
                atr_pips=atr_pips,
                bbw=bbw,
                vwap_gap=vwap_gap,
                ma_pair=ma_pair,
                range_active=False,
                log_tags={"trade": trade_id},
            )
            if reentry.action == "hold":
                return
            if reentry.action == "exit_reentry" and not reentry.shadow:
                await _close(trade_id, -units, "reentry_reset", client_id, allow_negative=True)
                states.pop(trade_id, None)
                return

        if _structure_break(units):
            await _close(trade_id, -units, "structure_break", client_id, allow_negative=True)
            states.pop(trade_id, None)
            return

        if pnl <= -stop_loss:
            await _close(trade_id, -units, "hard_stop", client_id, allow_negative=True)
            states.pop(trade_id, None)
            return

        if pnl <= 0:
            decision = evaluate_reversion_failure(
                trade,
                current_price=current,
                now=now,
                side=side,
                env_tf="M5",
                struct_tf="M1",
                trend_hits=state.trend_hits,
            )
            state.trend_hits = decision.trend_hits
            if decision.should_exit and decision.reason:
                log_metric(
                    "mirror_spike_reversion_exit",
                    pnl,
                    tags={"reason": decision.reason, "side": side},
                    ts=now,
                )
                await _close(
                    trade_id,
                    -units,
                    decision.reason,
                    client_id,
                    allow_negative=True,
                )
                states.pop(trade_id, None)
                return

        if pnl > 0:
            tp_decision = evaluate_tp_zone(
                trade,
                current_price=current,
                side=side,
                env_tf="M5",
                struct_tf="M1",
            )
            if tp_decision.should_exit:
                log_metric(
                    "mirror_spike_tp_zone",
                    pnl,
                    tags={"side": side},
                    ts=now,
                )
                await _close(trade_id, -units, "take_profit_zone", client_id, allow_negative=allow_negative)
                states.pop(trade_id, None)
                return

        if hold_sec >= max_hold and pnl <= profit_take * 0.5:
            await _close(trade_id, -units, "time_stop", client_id, allow_negative=allow_negative)
            states.pop(trade_id, None)
            return

        candle_reason = _exit_candle_reversal("long" if units > 0 else "short")
        if candle_reason and pnl >= 0:
            candle_client_id = trade.get("client_order_id")
            if not candle_client_id:
                client_ext = trade.get("clientExtensions")
                if isinstance(client_ext, dict):
                    candle_client_id = client_ext.get("id")
            if candle_client_id:
                await _close(
                    trade_id,
                    -units,
                    candle_reason,
                    candle_client_id,
                    allow_negative=allow_negative,
                )
                states.pop(trade_id, None)
                return
        if state.lock_floor is not None and pnl <= state.lock_floor:
            await _close(trade_id, -units, "lock_release", client_id, allow_negative=allow_negative)
            states.pop(trade_id, None)
            return

        if state.peak >= trail_start and pnl <= state.peak - params.trail_backoff:
            await _close(trade_id, -units, "trail_take", client_id, allow_negative=allow_negative)
            states.pop(trade_id, None)
            return

        if pnl >= profit_take:
            await _close(trade_id, -units, "take_profit", client_id, allow_negative=allow_negative)
            states.pop(trade_id, None)
            return

    try:
        while True:
            await asyncio.sleep(loop_interval)
            positions = pos_manager.get_open_positions()
            pocket_info = positions.get(pocket) or {}
            trades = pocket_info.get("open_trades") or []
            trades = _filter_trades(trades, tags)
            active_ids = {str(tr.get("trade_id")) for tr in trades if tr.get("trade_id")}
            for tid in list(states.keys()):
                if tid not in active_ids:
                    states.pop(tid, None)
            if not trades:
                continue
            now = _utc_now()
            for tr in trades:
                try:
                    await _review_trade(tr, now)
                except Exception:
                    LOG.exception("[EXIT-%s] review failed trade=%s", pocket, tr.get("trade_id"))
                    continue
    except asyncio.CancelledError:  # pragma: no cover - loop cancellation
        pass


async def mirror_spike_exit_worker() -> None:
    await _run_exit_loop(
        pocket="scalp",
        tags=ALLOWED_TAGS,
        params=_ExitParams(
            profit_take=1.8,
            trail_start=2.4,
            trail_backoff=0.8,
            stop_loss=1.2,
            max_hold_sec=18 * 60,
            lock_buffer=0.45,
        ),
        loop_interval=0.8,
    )


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

if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(mirror_spike_exit_worker())


