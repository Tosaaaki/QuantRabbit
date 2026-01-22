"""Exit loop for mirror_spike worker (scalp pocket)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Set

from analysis.technique_engine import evaluate_exit_techniques
from workers.common.exit_utils import close_trade
from execution.position_manager import PositionManager
from execution.reversion_failure import evaluate_reversion_failure, evaluate_tp_zone
from execution.section_axis import evaluate_section_exit
from indicators.factor_cache import all_factors
from market_data import tick_window
from utils.metrics_logger import log_metric

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
        pnl = (current - price_entry) * 100.0 if side == "long" else (price_entry - current) * 100.0
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

        section_decision = evaluate_section_exit(
            trade,
            current_price=current,
            now=now,
            side=side,
            pocket=pocket,
            hold_sec=hold_sec,
            entry_price=price_entry,
        )
        if section_decision.should_exit and section_decision.reason:
            LOG.info(
                "[EXIT-%s] section_exit trade=%s reason=%s detail=%s",
                pocket,
                trade_id,
                section_decision.reason,
                section_decision.debug,
            )
            await _close(
                trade_id,
                -units,
                section_decision.reason,
                client_id,
                allow_negative=section_decision.allow_negative,
            )
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
                tech_exit = evaluate_exit_techniques(
                    trade=trade,
                    current_price=current,
                    side=side,
                    pocket=pocket,
                )
                if tech_exit.should_exit and tech_exit.allow_negative:
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
                        allow_negative=tech_exit.allow_negative,
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


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(mirror_spike_exit_worker())
