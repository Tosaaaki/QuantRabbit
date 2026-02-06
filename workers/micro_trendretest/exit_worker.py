"""Exit loop for MicroTrendRetest (micro pocket) – strategy-dedicated EXIT."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Set, Tuple

from execution.position_manager import PositionManager
from indicators.factor_cache import all_factors
from market_data import tick_window
from utils.metrics_logger import log_metric
from workers.common.exit_utils import close_trade, mark_pnl_pips
from workers.common.pro_stop import maybe_close_pro_stop

PIP = 0.01
LOG = logging.getLogger(__name__)

POCKET = "micro"
BASE_TAG = "MicroTrendRetest"
ALLOWED_TAGS: Set[str] = {BASE_TAG}


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no"}


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    try:
        tick = tick_window.recent_ticks(seconds=2.0, limit=1)
    except Exception:
        tick = None
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


def _ma_pair(tf: str) -> Tuple[Optional[float], Optional[float]]:
    fac = all_factors().get(tf) or {}
    return _safe_float(fac.get("ma10")), _safe_float(fac.get("ma20"))


def _rsi() -> Optional[float]:
    return _safe_float((all_factors().get("M1") or {}).get("rsi"))


@dataclass
class _TradeState:
    peak: float
    lock_floor: Optional[float] = None
    tp_hint: Optional[float] = None
    sl_hint: Optional[float] = None

    def update(self, pnl: float, lock_buffer: float) -> None:
        if pnl > self.peak:
            self.peak = pnl
        if pnl > 0:
            floor = max(0.0, pnl - lock_buffer)
            self.lock_floor = floor if self.lock_floor is None else max(self.lock_floor, floor)


class MicroTrendRetestExitWorker:
    """
    MicroTrendRetest dedicated EXIT.
    - hard_stop: entry_thesis.hard_stop_pips (fallback: sl_pips) triggers loss cut
    - trend_failure: MA10/MA20 flip or MA20 cross (buffer) triggers early exit
    - time_stop: max-hold exceeded and not profitable => exit
    - win side: TP / trailing / lock-floor / RSI take-profit
    """

    def __init__(self) -> None:
        self.enabled = _bool_env("MICRO_TRT_EXIT_ENABLED", False)
        self.loop_interval = max(0.5, _float_env("MICRO_TRT_EXIT_LOOP_INTERVAL_SEC", 1.0))

        self.exit_tf = (os.getenv("MICRO_TRT_EXIT_TF", "M5") or "M5").strip().upper()
        if self.exit_tf not in {"M1", "M5"}:
            self.exit_tf = "M5"

        self.min_hold_sec = max(5.0, _float_env("MICRO_TRT_EXIT_MIN_HOLD_SEC", 20.0))
        self.max_hold_sec = max(self.min_hold_sec, _float_env("MICRO_TRT_EXIT_MAX_HOLD_SEC", 18 * 60))

        self.profit_take = max(1.6, _float_env("MICRO_TRT_EXIT_PROFIT_PIPS", 3.2))
        self.trail_start = max(1.8, _float_env("MICRO_TRT_EXIT_TRAIL_START_PIPS", 4.2))
        self.trail_backoff = max(0.25, _float_env("MICRO_TRT_EXIT_TRAIL_BACKOFF_PIPS", 1.1))
        self.lock_buffer = max(0.2, _float_env("MICRO_TRT_EXIT_LOCK_BUFFER_PIPS", 0.6))
        self.lock_trigger = max(0.6, _float_env("MICRO_TRT_EXIT_LOCK_TRIGGER_PIPS", self.profit_take * 0.35))

        self.hard_stop_fallback = max(1.2, _float_env("MICRO_TRT_EXIT_HARD_STOP_PIPS", 9.5))
        self.soft_stop_ratio = max(0.2, min(0.95, _float_env("MICRO_TRT_EXIT_SOFT_STOP_RATIO", 0.75)))
        self.soft_stop_abs = max(0.0, _float_env("MICRO_TRT_EXIT_SOFT_STOP_PIPS", 0.0))

        self.trend_fail_hold_sec = max(self.min_hold_sec, _float_env("MICRO_TRT_EXIT_TREND_FAIL_HOLD_SEC", 45.0))
        self.trend_fail_ma_buffer_pips = max(0.0, _float_env("MICRO_TRT_EXIT_TREND_FAIL_MA_BUFFER_PIPS", 0.25))
        self.trend_fail_gap_pips = max(0.0, _float_env("MICRO_TRT_EXIT_TREND_FAIL_GAP_PIPS", 0.3))

        self.rsi_take_long = _float_env("MICRO_TRT_EXIT_RSI_TAKE_LONG", 70.0)
        self.rsi_take_short = _float_env("MICRO_TRT_EXIT_RSI_TAKE_SHORT", 30.0)

        self._pos_manager = PositionManager()
        self._states: Dict[str, _TradeState] = {}

    async def _close(
        self,
        trade_id: str,
        units: int,
        reason: str,
        pnl: float,
        client_order_id: Optional[str],
        allow_negative: bool,
    ) -> None:
        ok = await close_trade(
            trade_id,
            units,
            client_order_id=client_order_id,
            allow_negative=allow_negative,
            exit_reason=reason,
        )
        if ok:
            LOG.info("[EXIT-micro_trt] trade=%s units=%s reason=%s pnl=%.2fp", trade_id, units, reason, pnl)
            log_metric("micro_trt_exit", 1.0, tags={"reason": reason})
        else:
            LOG.error("[EXIT-micro_trt] close failed trade=%s units=%s reason=%s", trade_id, units, reason)
            log_metric("micro_trt_exit_fail", 1.0, tags={"reason": reason})

    async def _review_trade(self, trade: dict, now: datetime, mid: float) -> None:
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
        if pnl is None:
            return

        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0

        thesis = trade.get("entry_thesis") or {}
        if not isinstance(thesis, dict):
            thesis = {}

        state = self._states.get(trade_id)
        if state is None:
            tp_hint = _safe_float(thesis.get("tp_pips"))
            sl_hint = _safe_float(thesis.get("hard_stop_pips")) or _safe_float(thesis.get("sl_pips"))
            state = _TradeState(peak=pnl, tp_hint=tp_hint, sl_hint=sl_hint)
            self._states[trade_id] = state

        client_ext = trade.get("clientExtensions")
        client_id = trade.get("client_order_id")
        if not client_id and isinstance(client_ext, dict):
            client_id = client_ext.get("id")
        if not client_id:
            LOG.warning("[EXIT-micro_trt] missing client_id trade=%s skip close", trade_id)
            return

        if await maybe_close_pro_stop(trade, now=now):
            self._states.pop(trade_id, None)
            return

        if hold_sec < self.min_hold_sec:
            return

        hard_stop_pips = state.sl_hint if state.sl_hint and state.sl_hint > 0 else self.hard_stop_fallback
        soft_stop_pips = self.soft_stop_abs if self.soft_stop_abs > 0 else max(0.8, hard_stop_pips * self.soft_stop_ratio)

        if pnl <= -hard_stop_pips:
            await self._close(
                trade_id,
                -units,
                "hard_stop",
                pnl,
                client_id,
                allow_negative=True,
            )
            self._states.pop(trade_id, None)
            return

        if pnl < 0 and hold_sec >= self.trend_fail_hold_sec:
            ma10, ma20 = _ma_pair(self.exit_tf)
            if ma10 is not None and ma20 is not None:
                gap_pips = abs(ma10 - ma20) / PIP
                flip = (side == "long" and ma10 <= ma20) or (side == "short" and ma10 >= ma20)
                buffer = self.trend_fail_ma_buffer_pips * PIP
                cross = (side == "long" and mid < ma20 - buffer) or (side == "short" and mid > ma20 + buffer)
                weak = gap_pips <= self.trend_fail_gap_pips
                if pnl <= -soft_stop_pips and (flip or cross or weak):
                    await self._close(
                        trade_id,
                        -units,
                        "trend_failure",
                        pnl,
                        client_id,
                        allow_negative=True,
                    )
                    self._states.pop(trade_id, None)
                    return

        if hold_sec >= self.max_hold_sec and pnl <= 0.0:
            await self._close(trade_id, -units, "trend_exhaust", pnl, client_id, allow_negative=True)
            self._states.pop(trade_id, None)
            return

        state.update(pnl, self.lock_buffer)
        rsi = _rsi()

        profit_take = self.profit_take
        trail_start = self.trail_start
        trail_backoff = self.trail_backoff
        lock_buffer = self.lock_buffer
        lock_trigger = self.lock_trigger

        if state.tp_hint:
            profit_take = max(profit_take, max(1.0, state.tp_hint * 0.92))
            trail_start = max(trail_start, profit_take * 0.88)
            lock_buffer = max(lock_buffer, profit_take * 0.35)
            lock_trigger = max(lock_trigger, profit_take * 0.30)

        if state.lock_floor is not None and state.peak >= lock_trigger and pnl > 0 and pnl <= state.lock_floor:
            await self._close(trade_id, -units, "lock_floor", pnl, client_id, allow_negative=False)
            self._states.pop(trade_id, None)
            return

        if state.peak > 0 and state.peak >= trail_start and pnl > 0 and pnl <= state.peak - trail_backoff:
            await self._close(trade_id, -units, "trail_take", pnl, client_id, allow_negative=False)
            self._states.pop(trade_id, None)
            return

        if pnl >= profit_take:
            await self._close(trade_id, -units, "take_profit", pnl, client_id, allow_negative=False)
            self._states.pop(trade_id, None)
            return

        if pnl > 0 and rsi is not None:
            if side == "long" and rsi >= self.rsi_take_long:
                await self._close(trade_id, -units, "rsi_take", pnl, client_id, allow_negative=False)
                self._states.pop(trade_id, None)
                return
            if side == "short" and rsi <= self.rsi_take_short:
                await self._close(trade_id, -units, "rsi_take", pnl, client_id, allow_negative=False)
                self._states.pop(trade_id, None)
                return

    async def run(self) -> None:
        if not self.enabled:
            LOG.info("[EXIT-micro_trt] disabled (idle)")
            try:
                while True:
                    await asyncio.sleep(3600.0)
            except asyncio.CancelledError:
                return

        LOG.info(
            "[EXIT-micro_trt] worker starting (interval=%.2fs tf=%s tags=%s)",
            self.loop_interval,
            self.exit_tf,
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
                mid = _latest_mid()
                if mid is None:
                    continue
                now = _utc_now()
                for tr in trades:
                    try:
                        await self._review_trade(tr, now, mid)
                    except Exception:
                        LOG.exception("[EXIT-micro_trt] review failed trade=%s", tr.get("trade_id"))
                        continue
        except asyncio.CancelledError:
            LOG.info("[EXIT-micro_trt] worker cancelled")
            raise
        finally:
            try:
                self._pos_manager.close()
            except Exception:
                LOG.exception("[EXIT-micro_trt] failed to close PositionManager")


async def micro_trendretest_exit_worker() -> None:
    worker = MicroTrendRetestExitWorker()
    await worker.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(micro_trendretest_exit_worker())

