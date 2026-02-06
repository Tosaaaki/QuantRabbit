"""Exit worker for Micro Pullback-Fib."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from execution.position_manager import PositionManager
from indicators.factor_cache import all_factors
from market_data import tick_window
from workers.common.exit_utils import close_trade, mark_pnl_pips
from workers.common.reentry_decider import decide_reentry

from . import config
from workers.common.pro_stop import maybe_close_pro_stop

LOG = logging.getLogger(__name__)


@dataclass
class _TradeState:
    peak: float = 0.0
    lock_floor: Optional[float] = None

    def update(self, pnl: float, lock_buffer: float) -> None:
        if pnl > self.peak:
            self.peak = pnl
        if pnl > 0:
            floor = max(0.0, pnl - lock_buffer)
            self.lock_floor = floor if self.lock_floor is None else max(self.lock_floor, floor)


def _float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_time(value: object) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _latest_mid() -> float:
    ticks = tick_window.recent_ticks(seconds=3.0, limit=1)
    if ticks:
        tick = ticks[-1]
        mid = _float(tick.get("mid"))
        if mid is not None and mid > 0:
            return mid
        bid = _float(tick.get("bid"))
        ask = _float(tick.get("ask"))
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    fac = all_factors().get("M1") or {}
    close = _float(fac.get("close"))
    return close if close and close > 0 else 0.0


def _recent_move_pips(window_sec: float) -> float:
    ticks = tick_window.recent_ticks(seconds=window_sec, limit=int(window_sec * 15) + 5)
    if len(ticks) < 2:
        return 0.0
    first = ticks[0]
    last = ticks[-1]
    first_mid = _float(first.get("mid")) or 0.0
    last_mid = _float(last.get("mid")) or first_mid
    return (last_mid - first_mid) / config.PIP_VALUE


def _atr_pips() -> Optional[float]:
    fac = all_factors().get("M1") or {}
    atr = _float(fac.get("atr_pips"))
    if atr is None:
        atr_val = _float(fac.get("atr"))
        atr = atr_val * 100.0 if atr_val is not None else None
    return atr


def _strategy_tag(trade: dict) -> Optional[str]:
    thesis = trade.get("entry_thesis") or {}
    tag = thesis.get("strategy_tag") or trade.get("strategy_tag")
    if not tag:
        return None
    base = str(tag).split("-", 1)[0].strip()
    return base or None


class MicroPullbackFibExitWorker:
    def __init__(self) -> None:
        self.loop_interval = config.EXIT_LOOP_INTERVAL_SEC
        self.min_hold_sec = config.EXIT_MIN_HOLD_SEC
        self.max_hold_sec = config.EXIT_MAX_HOLD_SEC
        self.profit_take = config.EXIT_PROFIT_PIPS
        self.trail_start = config.EXIT_TRAIL_START_PIPS
        self.trail_backoff = config.EXIT_TRAIL_BACKOFF_PIPS
        self.lock_buffer = config.EXIT_LOCK_BUFFER_PIPS
        self.lock_trigger = config.EXIT_LOCK_TRIGGER_PIPS
        self.rev_window_sec = config.EXIT_REVERSAL_WINDOW_SEC
        self.rev_pips = config.EXIT_REVERSAL_PIPS
        self.atr_spike_pips = config.EXIT_ATR_SPIKE_PIPS
        self.hard_stop_pips = config.EXIT_HARD_STOP_PIPS
        self.tp_hint_ratio = config.EXIT_TP_HINT_RATIO

        self._states: Dict[str, _TradeState] = {}
        self._pos_manager = PositionManager()

    async def _close(
        self,
        trade_id: str,
        units: int,
        reason: str,
        pnl: float,
        client_order_id: Optional[str],
    ) -> None:
        allow_negative = pnl <= 0 or reason in {"atr_spike", "hazard_exit"}
        ok = await close_trade(
            trade_id,
            units,
            client_order_id=client_order_id,
            allow_negative=allow_negative,
            exit_reason=reason,
        )
        if ok:
            LOG.info("[EXIT-micro_pullback_fib] trade=%s units=%s reason=%s pnl=%.2fp", trade_id, units, reason, pnl)
        else:
            LOG.error("[EXIT-micro_pullback_fib] close failed trade=%s units=%s reason=%s", trade_id, units, reason)

    async def _review_trade(self, trade: dict, now: datetime, mid: float) -> None:
        trade_id = str(trade.get("trade_id") or "")
        if not trade_id:
            return
        units = int(trade.get("units", 0) or 0)
        if units == 0:
            return

        entry = _float(trade.get("price")) or 0.0
        if entry <= 0:
            return
        side = "long" if units > 0 else "short"
        pnl = mark_pnl_pips(entry, units, mid=mid)

        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0

        state = self._states.get(trade_id)
        if state is None:
            state = _TradeState()
            self._states[trade_id] = state

        client_ext = trade.get("clientExtensions")
        client_id = trade.get("client_order_id")
        if not client_id and isinstance(client_ext, dict):
            client_id = client_ext.get("id")
        if not client_id:
            return
        if await maybe_close_pro_stop(trade, now=now):
            return

        if hold_sec < self.min_hold_sec:
            return

        tp_hint = None
        thesis = trade.get("entry_thesis") or {}
        tp_hint_raw = thesis.get("tp_pips")
        if tp_hint_raw is not None:
            try:
                tp_hint = float(tp_hint_raw)
            except (TypeError, ValueError):
                tp_hint = None

        profit_take = self.profit_take
        trail_start = self.trail_start
        lock_buffer = self.lock_buffer
        lock_trigger = self.lock_trigger
        if tp_hint:
            profit_take = max(profit_take, tp_hint * self.tp_hint_ratio)
            trail_start = max(trail_start, profit_take * 0.9)
            lock_buffer = max(lock_buffer, profit_take * 0.35)
            lock_trigger = max(lock_trigger, profit_take * 0.35)

        state.update(pnl, lock_buffer)

        if pnl >= profit_take:
            await self._close(trade_id, -units, "take_profit", pnl, client_id)
            self._states.pop(trade_id, None)
            return

        if state.lock_floor is not None and state.peak >= lock_trigger and pnl <= state.lock_floor:
            await self._close(trade_id, -units, "lock_floor", pnl, client_id)
            self._states.pop(trade_id, None)
            return

        if hold_sec >= self.max_hold_sec and pnl > 0:
            await self._close(trade_id, -units, "time_stop", pnl, client_id)
            self._states.pop(trade_id, None)
            return

        atr_pips = _atr_pips()
        if pnl <= -self.hard_stop_pips:
            await self._close(trade_id, -units, "hazard_exit", pnl, client_id)
            self._states.pop(trade_id, None)
            return

        if pnl < 0:
            fac_m1 = all_factors().get("M1") or {}
            rsi = _float(fac_m1.get("rsi"))
            adx = _float(fac_m1.get("adx"))
            atr_ref = _float(fac_m1.get("atr_pips"))
            bbw = _float(fac_m1.get("bbw"))
            vwap_gap = _float(fac_m1.get("vwap_gap"))
            ma10 = _float(fac_m1.get("ma10"))
            ma20 = _float(fac_m1.get("ma20"))
            ma_pair = (ma10, ma20) if ma10 is not None and ma20 is not None else None
            reentry = decide_reentry(
                prefix="MICRO_PULLBACK_FIB",
                side=side,
                pnl_pips=pnl,
                rsi=rsi,
                adx=adx,
                atr_pips=atr_ref,
                bbw=bbw,
                vwap_gap=vwap_gap,
                ma_pair=ma_pair,
                range_active=False,
                log_tags={"trade": trade_id},
            )
            if reentry.action == "hold":
                return
            if reentry.action == "exit_reentry" and not reentry.shadow:
                await self._close(trade_id, -units, "reentry_reset", pnl, client_id)
                self._states.pop(trade_id, None)
                return

        if atr_pips is not None and atr_pips >= self.atr_spike_pips and pnl < 0:
            await self._close(trade_id, -units, "atr_spike", pnl, client_id)
            self._states.pop(trade_id, None)
            return

        recent_move = _recent_move_pips(self.rev_window_sec)
        if side == "long" and recent_move <= -self.rev_pips:
            await self._close(trade_id, -units, "hazard_exit", pnl, client_id)
            self._states.pop(trade_id, None)
            return
        if side == "short" and recent_move >= self.rev_pips:
            await self._close(trade_id, -units, "hazard_exit", pnl, client_id)
            self._states.pop(trade_id, None)
            return

        if state.peak >= trail_start and pnl <= state.peak - self.trail_backoff:
            await self._close(trade_id, -units, "trail_backoff", pnl, client_id)
            self._states.pop(trade_id, None)
            return

    async def run(self) -> None:
        LOG.info("[EXIT-micro_pullback_fib] worker starting (interval=%.2fs)", self.loop_interval)
        while True:
            await asyncio.sleep(self.loop_interval)
            now = datetime.now(timezone.utc)
            mid = _latest_mid()
            if mid <= 0:
                continue
            positions = self._pos_manager.get_open_positions()
            pocket = positions.get(config.POCKET) or {}
            trades = pocket.get("open_trades") or []
            if not trades:
                continue
            for tr in trades:
                if _strategy_tag(tr) != config.STRATEGY_TAG:
                    continue
                try:
                    await self._review_trade(tr, now, mid)
                except Exception:
                    LOG.exception("[EXIT-micro_pullback_fib] review failed")


async def micro_pullback_fib_exit_worker() -> None:
    worker = MicroPullbackFibExitWorker()
    try:
        await worker.run()
    except asyncio.CancelledError:
        LOG.info("[EXIT-micro_pullback_fib] worker cancelled")
        raise
    finally:
        try:
            worker._pos_manager.close()
        except Exception:
            LOG.exception("[EXIT-micro_pullback_fib] failed to close PositionManager")


if __name__ == "__main__":
    asyncio.run(micro_pullback_fib_exit_worker())
