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
            state = TradeState(peak=pnl)
            self._states[trade_id] = state
        else:
            state.update(pnl, self.lock_buffer)

        # Hard stop
        if pnl <= -self.stop_loss:
            await self._close(trade_id, -units, "hard_stop")
            self._states.pop(trade_id, None)
            return

        # Max hold timeout
        if hold_sec >= self.max_hold_sec and pnl <= self.profit_take * 0.5:
            await self._close(trade_id, -units, "time_stop")
            self._states.pop(trade_id, None)
            return

        # Lock release
        if state.lock_floor is not None and pnl <= state.lock_floor:
            await self._close(trade_id, -units, "lock_release")
            self._states.pop(trade_id, None)
            return

        # Trail
        if state.peak >= self.trail_start and pnl <= state.peak - self.trail_backoff:
            await self._close(trade_id, -units, "trail_take")
            self._states.pop(trade_id, None)
            return

        # Take profit
        if pnl >= self.profit_take:
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
