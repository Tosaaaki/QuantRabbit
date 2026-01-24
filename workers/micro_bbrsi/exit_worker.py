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
from workers.common.exit_utils import close_trade

LOG = logging.getLogger(__name__)

ALLOWED_TAGS: Set[str] = {'BB_RSI_Fast', 'BB_RSI'}


def _tags_env(key: str, default: Set[str]) -> Set[str]:
    raw = os.getenv(key)
    if raw is None:
        return default
    tags = {t.strip() for t in raw.replace(";", ",").split(",") if t.strip()}
    return tags or default


def _pocket_env(key: str, default: str) -> str:
    raw = os.getenv(key)
    return raw.strip().lower() if raw else default


POCKET = _pocket_env("BBRSI_POCKET", "micro")
ALLOWED_TAGS = _tags_env("BBRSI_EXIT_TAGS", ALLOWED_TAGS)


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


class MicroBBRsiExitWorker:
    def __init__(self) -> None:
        self.loop_interval = max(
            0.5,
            _float_env("BBRSI_EXIT_LOOP_INTERVAL_SEC", 1.0),
        )
        self.min_hold_sec = max(
            5.0,
            _float_env("BBRSI_EXIT_MIN_HOLD_SEC", 20.0),
        )
        self.profit_take = max(
            0.5,
            _float_env("BBRSI_EXIT_PROFIT_PIPS", 2.0),
        )
        self.trail_start = max(
            0.5,
            _float_env("BBRSI_EXIT_TRAIL_START_PIPS", 2.6),
        )
        self.trail_backoff = max(
            0.1,
            _float_env("BBRSI_EXIT_TRAIL_BACKOFF_PIPS", 0.8),
        )
        self.lock_buffer = max(
            0.05,
            _float_env("BBRSI_EXIT_LOCK_BUFFER_PIPS", 0.5),
        )
        self.range_profit_take = max(
            0.4,
            _float_env("BBRSI_EXIT_RANGE_PROFIT_PIPS", 1.6),
        )
        self.range_trail_start = max(
            0.4,
            _float_env("BBRSI_EXIT_RANGE_TRAIL_START_PIPS", 2.1),
        )
        self.range_trail_backoff = max(
            0.1,
            _float_env("BBRSI_EXIT_RANGE_TRAIL_BACKOFF_PIPS", 0.6),
        )
        self.range_lock_buffer = max(
            0.05,
            _float_env("BBRSI_EXIT_RANGE_LOCK_BUFFER_PIPS", 0.3),
        )
        self.range_max_hold_sec = max(
            60.0,
            _float_env("BBRSI_EXIT_RANGE_MAX_HOLD_SEC", 1800.0),
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
        ok = await close_trade(
            trade_id,
            units,
            client_order_id=client_order_id,
            allow_negative=False,
            exit_reason=reason,
        )
        if ok:
            LOG.info("[exit-bbrsi] trade=%s units=%s reason=%s pnl=%.2fp", trade_id, units, reason, pnl)
        else:
            LOG.error("[exit-bbrsi] close failed trade=%s units=%s reason=%s", trade_id, units, reason)

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
        pnl = (mid - entry) * 100.0 if side == "long" else (entry - mid) * 100.0
        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0

        client_ext = trade.get("clientExtensions")
        client_id = trade.get("client_order_id")
        if not client_id and isinstance(client_ext, dict):
            client_id = client_ext.get("id")
        if not client_id:
            LOG.warning("[exit-bbrsi] missing client_id trade=%s skip close", trade_id)
            return

        if hold_sec < self.min_hold_sec:
            return
        if pnl <= 0:
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
            "[exit-bbrsi] exit worker start interval=%.2fs tags=%s pocket=%s",
            self.loop_interval,
            ",".join(sorted(ALLOWED_TAGS)) if ALLOWED_TAGS else "none",
            POCKET,
        )
        if not ALLOWED_TAGS:
            LOG.info("[exit-bbrsi] no allowed tags configured; idle")
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
                        LOG.exception("[exit-bbrsi] review failed trade=%s", tr.get("trade_id"))
        except asyncio.CancelledError:
            LOG.info("[exit-bbrsi] worker cancelled")
            raise
        finally:
            try:
                self._pos_manager.close()
            except Exception:
                LOG.exception("[exit-bbrsi] failed to close PositionManager")


async def micro_bbrsi_exit_worker() -> None:
    worker = MicroBBRsiExitWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(micro_bbrsi_exit_worker())
