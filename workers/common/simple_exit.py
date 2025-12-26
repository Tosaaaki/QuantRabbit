"""Lightweight per-worker exit loop with simple trail/stop rules."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Sequence

from execution.order_manager import close_trade
from execution.position_manager import PositionManager
from indicators.factor_cache import all_factors
from market_data import tick_window

LOG = logging.getLogger(__name__)


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


@dataclass
class TradeState:
    peak: float
    lock_floor: Optional[float] = None
    hard_stop: Optional[float] = None
    tp_hint: Optional[float] = None

    def update(self, pnl: float, lock_buffer: float) -> None:
        if pnl > self.peak:
            self.peak = pnl
        if self.lock_floor is None and pnl > 0:
            self.lock_floor = max(0.0, pnl - lock_buffer)


class SimpleExitWorker:
    """Attach per-worker exit logic scoped to strategy tags and pocket."""

    def __init__(
        self,
        *,
        pocket: str,
        strategy_tags: Iterable[str],
        profit_take: float,
        trail_start: float,
        trail_backoff: float,
        stop_loss: float,
        max_hold_sec: float,
        loop_interval: float = 1.0,
        lock_buffer: float = 0.6,
        use_entry_meta: bool = True,
        structure_break: bool = False,
        structure_timeframe: str = "M1",
        structure_adx: float = 22.0,
        structure_gap_pips: float = 2.0,
        structure_adx_cold: float = 12.0,
        virtual_sl_ratio: float = 0.72,
        trail_from_tp_ratio: float = 0.82,
        lock_from_tp_ratio: float = 0.45,
        tp_floor_ratio: float = 1.0,
    ) -> None:
        self.pocket = pocket
        self.strategy_tags = {tag.strip() for tag in strategy_tags if tag.strip()}
        self.profit_take = profit_take
        self.trail_start = trail_start
        self.trail_backoff = trail_backoff
        self.stop_loss = stop_loss
        self.max_hold_sec = max_hold_sec
        self.loop_interval = loop_interval
        self.lock_buffer = lock_buffer
        self.use_entry_meta = use_entry_meta
        self.structure_break = structure_break
        self.structure_timeframe = structure_timeframe
        self.structure_adx = structure_adx
        self.structure_gap_pips = structure_gap_pips
        self.structure_adx_cold = structure_adx_cold
        self.virtual_sl_ratio = max(0.0, virtual_sl_ratio)
        self.trail_from_tp_ratio = max(0.0, trail_from_tp_ratio)
        self.lock_from_tp_ratio = max(0.0, lock_from_tp_ratio)
        self.tp_floor_ratio = max(0.0, tp_floor_ratio)
        self._states: Dict[str, TradeState] = {}
        self._pos_manager = PositionManager()

    def _filter_trades(self, trades: Sequence[dict]) -> list[dict]:
        if not self.strategy_tags:
            return list(trades)
        filtered: list[dict] = []
        for tr in trades:
            thesis = tr.get("entry_thesis") or {}
            tag = thesis.get("strategy_tag") or thesis.get("strategy") or tr.get("strategy")
            if tag and str(tag) in self.strategy_tags:
                filtered.append(tr)
        return filtered

    async def _close(self, trade_id: str, units: int, reason: str) -> bool:
        ok = await close_trade(trade_id, units)
        if ok:
            LOG.info("[EXIT-%s] trade=%s units=%s reason=%s", self.pocket, trade_id, units, reason)
        else:
            LOG.error("[EXIT-%s] close failed trade=%s units=%s reason=%s", self.pocket, trade_id, units, reason)
        return ok

    async def _review_trade(self, trade: dict, now: datetime) -> None:
        trade_id = str(trade.get("trade_id"))
        if not trade_id:
            return
        units = int(trade.get("units", 0) or 0)
        if units == 0:
            return

        price_entry = float(trade.get("price") or 0.0)
        side = "long" if units > 0 else "short"
        current = _latest_mid()
        if current is None or price_entry <= 0:
            return
        pnl = (current - price_entry) * 100.0 if side == "long" else (price_entry - current) * 100.0

        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0

        state = self._states.get(trade_id)
        if state is None:
            thesis = trade.get("entry_thesis") or {}
            hard_stop = thesis.get("hard_stop_pips") or thesis.get("hard_stop") or thesis.get("stop_loss")
            tp_hint = thesis.get("tp_pips") or thesis.get("tp") or thesis.get("take_profit")
            try:
                hard_stop_val = float(hard_stop) if hard_stop is not None else None
            except Exception:
                hard_stop_val = None
            try:
                tp_hint_val = float(tp_hint) if tp_hint is not None else None
            except Exception:
                tp_hint_val = None
            state = TradeState(peak=pnl, hard_stop=hard_stop_val, tp_hint=tp_hint_val)
            self._states[trade_id] = state

        # lightly scale thresholds with entry meta when requested
        profit_take = self.profit_take
        trail_start = self.trail_start
        stop_loss = self.stop_loss
        max_hold = self.max_hold_sec
        lock_buffer = self.lock_buffer
        if self.use_entry_meta:
            if state.tp_hint:
                profit_take = max(profit_take, max(1.0, state.tp_hint * self.tp_floor_ratio))
                trail_start = max(trail_start, max(1.0, profit_take * self.trail_from_tp_ratio))
                lock_buffer = max(lock_buffer, profit_take * self.lock_from_tp_ratio)
            if state.hard_stop:
                stop_loss = max(stop_loss, max(0.8, state.hard_stop * 0.5))
                lock_buffer = max(lock_buffer, stop_loss * 0.35)
                trail_start = max(trail_start, max(1.0, state.hard_stop * 0.6))
                max_hold = max(max_hold, self.max_hold_sec * 1.05)

        # TP を優先しつつ、RR を維持する仮想SL/ロック位置を決定
        stop_loss = max(stop_loss, profit_take * self.virtual_sl_ratio)
        trail_start = max(trail_start, profit_take * self.trail_from_tp_ratio)
        lock_buffer = max(lock_buffer, profit_take * self.lock_from_tp_ratio)

        state.update(pnl, lock_buffer)

        # Structure break: MA10/MA20逆転＋ADX低下（エントリー優位性崩れ）
        if self.structure_break:
            try:
                factors = all_factors().get(self.structure_timeframe) or {}
                adx = float(factors.get("adx"))
                ma10 = float(factors.get("ma10"))
                ma20 = float(factors.get("ma20"))
                gap = abs(ma10 - ma20) / 0.01
                if adx < self.structure_adx:
                    dir_long = units > 0
                    if (dir_long and ma10 <= ma20) or ((not dir_long) and ma10 >= ma20) or (adx < self.structure_adx_cold and gap < self.structure_gap_pips):
                        await self._close(trade_id, -units, "structure_break")
                        self._states.pop(trade_id, None)
                        return
            except Exception:
                pass

        # Hard stop
        if pnl <= -stop_loss:
            await self._close(trade_id, -units, "hard_stop")
            self._states.pop(trade_id, None)
            return

        # Max hold timeout
        if hold_sec >= max_hold and pnl <= profit_take * 0.5:
            await self._close(trade_id, -units, "time_stop")
            self._states.pop(trade_id, None)
            return

        # Lock release
        if state.lock_floor is not None and pnl <= state.lock_floor:
            await self._close(trade_id, -units, "lock_release")
            self._states.pop(trade_id, None)
            return

        # Trail
        if state.peak >= trail_start and pnl <= state.peak - self.trail_backoff:
            await self._close(trade_id, -units, "trail_take")
            self._states.pop(trade_id, None)
            return

        # Take profit
        if pnl >= profit_take:
            await self._close(trade_id, -units, "take_profit")
            self._states.pop(trade_id, None)
            return

    async def run(self) -> None:
        LOG.info("[EXIT-%s] worker starting (interval=%.2fs tags=%s)", self.pocket, self.loop_interval, ",".join(self.strategy_tags) or "all")
        try:
            while True:
                await asyncio.sleep(self.loop_interval)
                positions = self._pos_manager.get_open_positions()
                pocket_info = positions.get(self.pocket) or {}
                trades = pocket_info.get("open_trades") or []
                trades = self._filter_trades(trades)
                active_ids = {str(tr.get("trade_id")) for tr in trades if tr.get("trade_id")}
                # cleanup stale states
                for tid in list(self._states.keys()):
                    if tid not in active_ids:
                        self._states.pop(tid, None)
                if not trades:
                    continue
                now = _utc_now()
                for tr in trades:
                    try:
                        await self._review_trade(tr, now)
                    except Exception:
                        LOG.exception("[EXIT-%s] review failed trade=%s", self.pocket, tr.get("trade_id"))
                        continue
        except asyncio.CancelledError:
            LOG.info("[EXIT-%s] worker cancelled", self.pocket)
            raise
        finally:
            try:
                self._pos_manager.close()
            except Exception:
                LOG.exception("[EXIT-%s] failed to close PositionManager", self.pocket)
