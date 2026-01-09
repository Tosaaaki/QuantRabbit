"""Exit loop for Donchian55 macro worker – プラス決済専用。"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Set

from analysis.range_guard import detect_range_mode
from execution.order_manager import close_trade
from execution.position_manager import PositionManager
from indicators.factor_cache import all_factors
from market_data import tick_window
from utils.metrics_logger import log_metric

LOG = logging.getLogger(__name__)

ALLOWED_TAGS: Set[str] = {"Donchian55"}
POCKET = "macro"
PIP = 0.01


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(val: object) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _latest_mid() -> Optional[float]:
    tick = tick_window.recent_ticks(seconds=3.0, limit=1)
    if tick:
        try:
            return float(tick[-1]["mid"])
        except Exception:
            pass
    try:
        return float(all_factors().get("H1", {}).get("close"))
    except Exception:
        return None


def _filter_trades(trades: Sequence[dict], tags: Set[str]) -> list[dict]:
    if not tags:
        return list(trades)
    filtered: list[dict] = []
    for tr in trades:
        thesis = tr.get("entry_thesis") or {}
        tag = thesis.get("strategy_tag") or thesis.get("strategy") or tr.get("strategy")
        if tag and str(tag) in tags:
            filtered.append(tr)
    return filtered


def _donchian_failure(side: str, fac_h1: dict, buffer_pips: float) -> bool:
    candles = fac_h1.get("candles") or []
    if not isinstance(candles, list) or len(candles) < 56:
        return False
    window = candles[-56:]
    high_vals = []
    low_vals = []
    for cndl in window[:-1]:
        hval = _safe_float(cndl.get("high"))
        lval = _safe_float(cndl.get("low"))
        if hval is not None:
            high_vals.append(hval)
        if lval is not None:
            low_vals.append(lval)
    close = _safe_float(window[-1].get("close"))
    if not high_vals or not low_vals or close is None:
        return False
    high55 = max(high_vals)
    low55 = min(low_vals)
    buffer_price = max(0.0, buffer_pips) * PIP
    if side == "long":
        return close <= high55 - buffer_price
    return close >= low55 + buffer_price


@dataclass
class _TradeState:
    peak: float
    lock_floor: Optional[float] = None
    tp_hint: Optional[float] = None

    def update(self, pnl: float, lock_buffer: float) -> None:
        if pnl > self.peak:
            self.peak = pnl
        if self.lock_floor is None and pnl > 0:
            self.lock_floor = max(0.0, pnl - lock_buffer)


class DonchianExitWorker:
    """
    Donchian55 専用 EXIT（マクロ）。
    - 最低保有20分以上
    - PnL>0 のみクローズ（TP/トレール/RSI利確/レンジ長時間微益）
    """

    def __init__(self) -> None:
        self.loop_interval = max(1.0, _float_env("DONCHIAN55_EXIT_LOOP_INTERVAL_SEC", 2.0))
        self._pos_manager = PositionManager()
        self._states: Dict[str, _TradeState] = {}

        self.profit_take = max(5.0, _float_env("DONCHIAN55_EXIT_PROFIT_PIPS", 7.0))
        self.trail_start = max(6.0, _float_env("DONCHIAN55_EXIT_TRAIL_START_PIPS", 9.0))
        self.trail_backoff = max(0.8, _float_env("DONCHIAN55_EXIT_TRAIL_BACKOFF_PIPS", 3.0))
        self.lock_buffer = max(0.7, _float_env("DONCHIAN55_EXIT_LOCK_BUFFER_PIPS", 1.2))
        self.min_hold_sec = max(1200.0, _float_env("DONCHIAN55_EXIT_MIN_HOLD_SEC", 20 * 60))

        self.range_profit_take = max(3.5, _float_env("DONCHIAN55_EXIT_RANGE_PROFIT_PIPS", 5.4))
        self.range_trail_start = max(4.5, _float_env("DONCHIAN55_EXIT_RANGE_TRAIL_START_PIPS", 7.4))
        self.range_trail_backoff = max(0.6, _float_env("DONCHIAN55_EXIT_RANGE_TRAIL_BACKOFF_PIPS", 2.2))
        self.range_lock_buffer = max(0.5, _float_env("DONCHIAN55_EXIT_RANGE_LOCK_BUFFER_PIPS", 1.0))
        self.range_max_hold_sec = max(1500.0, _float_env("DONCHIAN55_EXIT_RANGE_MAX_HOLD_SEC", 5 * 3600))

        self.range_adx = max(5.0, _float_env("DONCHIAN55_EXIT_RANGE_ADX", 22.0))
        self.range_bbw = max(0.02, _float_env("DONCHIAN55_EXIT_RANGE_BBW", 0.24))
        self.range_atr = max(0.4, _float_env("DONCHIAN55_EXIT_RANGE_ATR", 7.0))

        self.rsi_take_long = _float_env("DONCHIAN55_EXIT_RSI_TAKE_LONG", 72.0)
        self.rsi_take_short = _float_env("DONCHIAN55_EXIT_RSI_TAKE_SHORT", 28.0)
        self.trend_fail_pips = max(10.0, _float_env("DONCHIAN55_EXIT_TREND_FAIL_PIPS", 32.0))
        self.trend_fail_buffer_pips = max(0.5, _float_env("DONCHIAN55_EXIT_TREND_FAIL_BUFFER_PIPS", 3.0))

    def _context(self) -> tuple[Optional[float], Optional[float], bool, dict]:
        factors = all_factors()
        fac_h1 = factors.get("H1") or {}
        fac_h4 = factors.get("H4") or {}

        adx = _safe_float(fac_h1.get("adx"))
        bbw = _safe_float(fac_h1.get("bbw"))
        atr = _safe_float(fac_h1.get("atr_pips")) or (_safe_float(fac_h1.get("atr")) or 0.0) * 100.0
        range_ctx = detect_range_mode(
            fac_h1,
            fac_h4,
            adx_threshold=self.range_adx,
            bbw_threshold=self.range_bbw,
            atr_threshold=self.range_atr,
        )
        rsi = _safe_float(fac_h1.get("rsi"))
        return _latest_mid(), rsi, bool(range_ctx.active), fac_h1

    async def _close(self, trade_id: str, units: int, reason: str, pnl: float, client_order_id: Optional[str]) -> None:
        ok = await close_trade(trade_id, units, client_order_id=client_order_id)
        if ok:
            LOG.info("[EXIT-donchian55] trade=%s units=%s reason=%s pnl=%.2fp", trade_id, units, reason, pnl)
        else:
            LOG.error("[EXIT-donchian55] close failed trade=%s units=%s reason=%s", trade_id, units, reason)

    async def _review_trade(
        self,
        trade: dict,
        now: datetime,
        mid: float,
        rsi: Optional[float],
        range_active: bool,
        fac_h1: dict,
    ) -> None:
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

        state = self._states.get(trade_id)
        if state is None:
            thesis = trade.get("entry_thesis") or {}
            tp_hint = thesis.get("tp_pips") or thesis.get("tp") or thesis.get("take_profit")
            try:
                tp_hint_val = float(tp_hint) if tp_hint is not None else None
            except Exception:
                tp_hint_val = None
            state = _TradeState(peak=pnl, tp_hint=tp_hint_val)
            self._states[trade_id] = state
        state.update(pnl, self.lock_buffer if not range_active else self.range_lock_buffer)

        client_ext = trade.get("clientExtensions")
        client_id = trade.get("client_order_id")
        if not client_id and isinstance(client_ext, dict):
            client_id = client_ext.get("id")
        if not client_id:
            LOG.warning("[EXIT-donchian] missing client_id trade=%s skip close", trade_id)
            return

        if hold_sec < self.min_hold_sec:
            return

        if (
            self.trend_fail_pips > 0
            and pnl <= -self.trend_fail_pips
            and _donchian_failure(side, fac_h1, self.trend_fail_buffer_pips)
        ):
            log_metric("donchian55_trend_failure", pnl, tags={"side": side})
            await self._close(trade_id, -units, "trend_failure", pnl, client_id)
            self._states.pop(trade_id, None)
            return

        profit_take = self.range_profit_take if range_active else self.profit_take
        trail_start = self.range_trail_start if range_active else self.trail_start
        trail_backoff = self.range_trail_backoff if range_active else self.trail_backoff
        lock_buffer = self.range_lock_buffer if range_active else self.lock_buffer

        if state.tp_hint:
            profit_take = max(profit_take, max(2.5, state.tp_hint * 0.9))
            trail_start = max(trail_start, profit_take * 0.9)
            lock_buffer = max(lock_buffer, profit_take * 0.35)

        if state.peak > 0 and state.peak >= trail_start and pnl > 0 and pnl <= state.peak - trail_backoff:
            await self._close(trade_id, -units, "trail_take", pnl, client_id)
            self._states.pop(trade_id, None)
            return

        if pnl >= profit_take:
            await self._close(trade_id, -units, "take_profit", pnl, client_id)
            self._states.pop(trade_id, None)
            return

        if pnl > 0 and rsi is not None:
            if side == "long" and rsi >= self.rsi_take_long:
                await self._close(trade_id, -units, "rsi_take", pnl, client_id)
                self._states.pop(trade_id, None)
                return
            if side == "short" and rsi <= self.rsi_take_short:
                await self._close(trade_id, -units, "rsi_take", pnl, client_id)
                self._states.pop(trade_id, None)
                return

        if range_active and hold_sec >= self.range_max_hold_sec and pnl > 0:
            await self._close(trade_id, -units, "range_timeout", pnl, client_id)
            self._states.pop(trade_id, None)
            return

    async def run(self) -> None:
        LOG.info(
            "[EXIT-donchian55] worker starting (interval=%.2fs tags=%s)",
            self.loop_interval,
            ",".join(sorted(ALLOWED_TAGS)),
        )
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

                mid, rsi, range_active, fac_h1 = self._context()
                if mid is None:
                    continue

                now = _utc_now()
                for tr in trades:
                    try:
                        await self._review_trade(tr, now, mid, rsi, range_active, fac_h1)
                    except Exception:
                        LOG.exception("[EXIT-donchian55] review failed trade=%s", tr.get("trade_id"))
                        continue
        except asyncio.CancelledError:
            LOG.info("[EXIT-donchian55] worker cancelled")
            raise
        finally:
            try:
                self._pos_manager.close()
            except Exception:
                LOG.exception("[EXIT-donchian55] failed to close PositionManager")


async def macro_donchian55_exit_worker() -> None:
    worker = DonchianExitWorker()
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(macro_donchian55_exit_worker())
