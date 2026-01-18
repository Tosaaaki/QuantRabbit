"""Per-trade EXIT loop for M1Scalper (scalp pocket) – 利確優先 + 安全弁。"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Set

from workers.common.exit_utils import close_trade
from execution.position_manager import PositionManager
from execution.section_axis import evaluate_section_exit
from indicators.factor_cache import all_factors
from market_data import tick_window
from utils.metrics_logger import log_metric

LOG = logging.getLogger(__name__)

ALLOWED_TAGS = {"M1Scalper", "m1scalper", "m1_scalper"}


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


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


@dataclass
class _TradeState:
    peak: float
    lock_floor: Optional[float] = None
    hard_stop: Optional[float] = None
    tp_hint: Optional[float] = None

    def update(self, pnl: float, lock_buffer: float) -> None:
        if pnl > self.peak:
            self.peak = pnl
        if self.lock_floor is None and pnl > 0:
            self.lock_floor = max(0.0, pnl - lock_buffer)


async def _run_exit_loop(
    *,
    pocket: str,
    tags: Set[str],
    profit_take: float,
    trail_start: float,
    trail_backoff: float,
    lock_buffer: float,
    min_hold_sec: float,
    max_hold_sec: float,
    max_adverse_pips: float,
    trail_from_tp_ratio: float,
    lock_from_tp_ratio: float,
    loop_interval: float,
) -> None:
    pos_manager = PositionManager()
    states: Dict[str, _TradeState] = {}
    rsi_fade_long = _float_env("M1SCALP_EXIT_RSI_FADE_LONG", 44.0)
    rsi_fade_short = _float_env("M1SCALP_EXIT_RSI_FADE_SHORT", 56.0)
    vwap_gap_pips = _float_env("M1SCALP_EXIT_VWAP_GAP_PIPS", 0.8)
    structure_adx = _float_env("M1SCALP_EXIT_STRUCTURE_ADX", 20.0)
    structure_gap_pips = _float_env("M1SCALP_EXIT_STRUCTURE_GAP_PIPS", 1.8)
    atr_spike_pips = _float_env("M1SCALP_EXIT_ATR_SPIKE_PIPS", 5.0)
    allow_negative_exit = os.getenv("M1SCALP_EXIT_ALLOW_NEGATIVE", "1").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
    }

    def _context() -> tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[tuple[float, float]],
    ]:
        fac_m1 = all_factors().get("M1") or {}
        rsi = _safe_float(fac_m1.get("rsi"))
        adx = _safe_float(fac_m1.get("adx"))
        atr_pips = _safe_float(fac_m1.get("atr_pips"))
        if atr_pips is None:
            atr_val = _safe_float(fac_m1.get("atr"))
            if atr_val is not None:
                atr_pips = atr_val * 100.0
        vwap_gap = _safe_float(fac_m1.get("vwap_gap"))
        ma10 = _safe_float(fac_m1.get("ma10"))
        ma20 = _safe_float(fac_m1.get("ma20"))
        return rsi, adx, atr_pips, vwap_gap, (ma10, ma20) if ma10 is not None and ma20 is not None else None

    async def _close(
        trade_id: str,
        units: int,
        reason: str,
        client_order_id: Optional[str],
        allow_negative: bool = False,
    ) -> bool:
        ok = await close_trade(
            trade_id,
            units,
            client_order_id=client_order_id,
            allow_negative=allow_negative,
        )
        if ok:
            LOG.info("[EXIT-%s] trade=%s units=%s reason=%s", pocket, trade_id, units, reason)
        else:
            LOG.error("[EXIT-%s] close failed trade=%s units=%s reason=%s", pocket, trade_id, units, reason)
        return ok

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
        allow_negative = allow_negative_exit or pnl <= 0

        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0

        state = states.get(trade_id)
        if state is None:
            thesis = trade.get("entry_thesis") or {}
            hard_stop = thesis.get("hard_stop_pips") or thesis.get("hard_stop") or thesis.get("stop_loss")
            tp_hint = thesis.get("tp_pips") or thesis.get("tp") or thesis.get("take_profit") or thesis.get("target_tp_pips")
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

        tp = profit_take
        ts = trail_start
        lb = lock_buffer

        if state.tp_hint:
            tp = max(tp, max(1.0, state.tp_hint * 0.9))
            ts = max(ts, max(1.0, tp * trail_from_tp_ratio))
            lb = max(lb, tp * lock_from_tp_ratio)

        state.update(pnl, lb)

        client_ext = trade.get("clientExtensions")
        client_id = trade.get("client_order_id")
        if not client_id and isinstance(client_ext, dict):
            client_id = client_ext.get("id")

        # 最低保有時間まではクローズ禁止（スプレッド負け防止）
        if hold_sec < min_hold_sec:
            return

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
            min_hold_sec=min_hold_sec,
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

        if pnl < 0:
            rsi, adx, atr_pips, vwap_gap, ma_pair = _context()
            if rsi is not None:
                if side == "long" and rsi <= rsi_fade_long:
                    await _close(trade_id, -units, "rsi_fade", client_id, allow_negative=allow_negative)
                    states.pop(trade_id, None)
                    return
                if side == "short" and rsi >= rsi_fade_short:
                    await _close(trade_id, -units, "rsi_fade", client_id, allow_negative=allow_negative)
                    states.pop(trade_id, None)
                    return
            if vwap_gap is not None and abs(vwap_gap) <= vwap_gap_pips:
                await _close(trade_id, -units, "vwap_cut", client_id, allow_negative=allow_negative)
                states.pop(trade_id, None)
                return
            if adx is not None and ma_pair is not None:
                ma10, ma20 = ma_pair
                gap = abs(ma10 - ma20) / 0.01
                cross_bad = (side == "long" and ma10 <= ma20) or (side == "short" and ma10 >= ma20)
                if adx <= structure_adx and (cross_bad or gap <= structure_gap_pips):
                    await _close(trade_id, -units, "structure_break", client_id, allow_negative=allow_negative)
                    states.pop(trade_id, None)
                    return
            if atr_pips is not None and atr_pips >= atr_spike_pips:
                await _close(trade_id, -units, "atr_spike", client_id, allow_negative=allow_negative)
                states.pop(trade_id, None)
                return

        if max_adverse_pips > 0 and pnl <= -max_adverse_pips:
            log_metric("m1scalp_max_adverse", pnl, tags={"side": side})
            await _close(trade_id, -units, "max_adverse", client_id, allow_negative=allow_negative)
            states.pop(trade_id, None)
            return

        if max_hold_sec > 0 and hold_sec >= max_hold_sec and pnl <= 0:
            log_metric("m1scalp_max_hold", pnl, tags={"side": side})
            await _close(trade_id, -units, "max_hold", client_id, allow_negative=allow_negative)
            states.pop(trade_id, None)
            return

        trail_trigger = max(ts, 0.0)

        # プラス圏のみでクローズする
        if (
            state.peak > 0
            and state.peak >= trail_trigger
            and pnl > 0
            and pnl <= state.peak - trail_backoff
        ):
            await _close(trade_id, -units, "trail_take", client_id, allow_negative=allow_negative)
            states.pop(trade_id, None)
            return

        if pnl >= tp:
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


async def m1_scalper_exit_worker() -> None:
    min_hold_sec = 10.0
    max_hold_sec = max(min_hold_sec + 1.0, _float_env("M1SCALP_EXIT_MAX_HOLD_SEC", 20 * 60))
    max_adverse_pips = max(0.0, _float_env("M1SCALP_EXIT_MAX_ADVERSE_PIPS", 8.0))
    await _run_exit_loop(
        pocket="scalp",
        tags=ALLOWED_TAGS,
        profit_take=2.0,
        trail_start=2.6,
        trail_backoff=0.9,
        lock_buffer=0.5,
        min_hold_sec=min_hold_sec,
        max_hold_sec=max_hold_sec,
        max_adverse_pips=max_adverse_pips,
        trail_from_tp_ratio=0.82,
        lock_from_tp_ratio=0.45,
        loop_interval=0.8,
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(m1_scalper_exit_worker())
