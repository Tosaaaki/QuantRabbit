"""Exit loop for MomentumBurst (micro pocket) – プラス決済専用。"""

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

LOG = logging.getLogger(__name__)

ALLOWED_TAGS: Set[str] = {"MomentumBurst"}
POCKET = "micro"


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
    tp_hint: Optional[float] = None

    def update(self, pnl: float, lock_buffer: float) -> None:
        if pnl > self.peak:
            self.peak = pnl
        if self.lock_floor is None and pnl > 0:
            self.lock_floor = max(0.0, pnl - lock_buffer)


class MomentumBurstExitWorker:
    """
    MomentumBurst 専用 EXIT。
    - 最低保有 20s までクローズ禁止
    - PnL>0 のみクローズ（TP/トレール/RSI利確/レンジ長時間微益）
    """

    def __init__(self) -> None:
        self.loop_interval = max(0.5, _float_env("MOMB_EXIT_LOOP_INTERVAL_SEC", 1.0))
        self._pos_manager = PositionManager()
        self._states: Dict[str, _TradeState] = {}

        self.profit_take = max(1.5, _float_env("MOMB_EXIT_PROFIT_PIPS", 2.4))
        self.trail_start = max(1.7, _float_env("MOMB_EXIT_TRAIL_START_PIPS", 2.8))
        self.trail_backoff = max(0.25, _float_env("MOMB_EXIT_TRAIL_BACKOFF_PIPS", 0.9))
        self.lock_buffer = max(0.22, _float_env("MOMB_EXIT_LOCK_BUFFER_PIPS", 0.55))
        self.min_hold_sec = max(5.0, _float_env("MOMB_EXIT_MIN_HOLD_SEC", 20.0))

        self.range_profit_take = max(1.1, _float_env("MOMB_EXIT_RANGE_PROFIT_PIPS", 1.7))
        self.range_trail_start = max(1.3, _float_env("MOMB_EXIT_RANGE_TRAIL_START_PIPS", 2.3))
        self.range_trail_backoff = max(0.2, _float_env("MOMB_EXIT_RANGE_TRAIL_BACKOFF_PIPS", 0.7))
        self.range_lock_buffer = max(0.16, _float_env("MOMB_EXIT_RANGE_LOCK_BUFFER_PIPS", 0.35))
        self.range_max_hold_sec = max(90.0, _float_env("MOMB_EXIT_RANGE_MAX_HOLD_SEC", 30 * 60))

        self.rsi_take_long = _float_env("MOMB_EXIT_RSI_TAKE_LONG", 70.0)
        self.rsi_take_short = _float_env("MOMB_EXIT_RSI_TAKE_SHORT", 30.0)
        self.tech_exit_enabled = _bool_env("MOMB_EXIT_TECH_ENABLED", True)
        self.rsi_fade_long = _float_env("MOMB_EXIT_RSI_FADE_LONG", 44.0)
        self.rsi_fade_short = _float_env("MOMB_EXIT_RSI_FADE_SHORT", 56.0)
        self.vwap_gap_pips = _float_env("MOMB_EXIT_VWAP_GAP_PIPS", 0.8)
        self.structure_adx = _float_env("MOMB_EXIT_STRUCTURE_ADX", 20.0)
        self.structure_gap_pips = _float_env("MOMB_EXIT_STRUCTURE_GAP_PIPS", 1.8)
        self.atr_spike_pips = _float_env("MOMB_EXIT_ATR_SPIKE_PIPS", 5.0)
        self.tech_neg_min_pips = max(0.2, _float_env("MOMB_EXIT_TECH_NEG_MIN_PIPS", self.profit_take * 0.6))
        hard_default = max(self.tech_neg_min_pips + 0.8, self.profit_take * 1.6)
        self.tech_neg_hard_pips = max(
            self.tech_neg_min_pips,
            _float_env("MOMB_EXIT_TECH_NEG_HARD_PIPS", hard_default),
        )
        self.tech_neg_hold_sec = max(self.min_hold_sec, _float_env("MOMB_EXIT_TECH_NEG_HOLD_SEC", 45.0))
        self.tech_score_min = max(1, int(_float_env("MOMB_EXIT_TECH_SCORE_MIN", 2.0)))

    def _context(self) -> tuple[Optional[float], Optional[float], bool]:
        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}

        adx = _safe_float(fac_m1.get("adx"))
        bbw = _safe_float(fac_m1.get("bbw"))
        atr = _safe_float(fac_m1.get("atr_pips")) or (_safe_float(fac_m1.get("atr")) or 0.0) * 100.0
        range_active = False
        try:
            from analysis.range_guard import detect_range_mode
            range_active = bool(detect_range_mode(fac_m1, fac_h4).active)
        except Exception:
            range_active = False
        rsi = _safe_float(fac_m1.get("rsi"))
        return _latest_mid(), rsi, range_active

    def _tech_context(
        self,
    ) -> tuple[
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
        ma_pair = (ma10, ma20) if ma10 is not None and ma20 is not None else None
        return rsi, adx, atr_pips, vwap_gap, ma_pair

    async def _close(
        self,
        trade_id: str,
        units: int,
        reason: str,
        pnl: float,
        client_order_id: Optional[str],
        allow_negative: bool = False,
    ) -> None:
        if pnl <= 0:
            allow_negative = True
        ok = await close_trade(
            trade_id,
            units,
            client_order_id=client_order_id,
            allow_negative=allow_negative,
            exit_reason=reason,
        )
        if ok:
            LOG.info("[EXIT-momb] trade=%s units=%s reason=%s pnl=%.2fp", trade_id, units, reason, pnl)
        else:
            LOG.error("[EXIT-momb] close failed trade=%s units=%s reason=%s", trade_id, units, reason)

    async def _review_trade(self, trade: dict, now: datetime, mid: float, rsi: Optional[float], range_active: bool) -> None:
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
            tp_hint = thesis.get("tp_pips")
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
            LOG.warning("[EXIT-momb] missing client_id trade=%s skip close", trade_id)
            return

        if hold_sec < self.min_hold_sec:
            return

        section_decision = evaluate_section_exit(
            trade,
            current_price=mid,
            now=now,
            side=side,
            pocket=POCKET,
            hold_sec=hold_sec,
            min_hold_sec=self.min_hold_sec,
            entry_price=entry,
        )
        if section_decision.should_exit and section_decision.reason:
            LOG.info(
                "[EXIT-momb] section_exit trade=%s reason=%s detail=%s",
                trade_id,
                section_decision.reason,
                section_decision.debug,
            )
            await self._close(
                trade_id,
                -units,
                section_decision.reason,
                pnl,
                client_id,
                allow_negative=section_decision.allow_negative,
            )
            self._states.pop(trade_id, None)
            return

        if pnl < 0 and self.tech_exit_enabled:
            hard_stop_ready = pnl <= -self.tech_neg_hard_pips and hold_sec >= self.min_hold_sec
            soft_ready = pnl <= -self.tech_neg_min_pips and hold_sec >= self.tech_neg_hold_sec
            if hard_stop_ready or soft_ready:
                rsi_val, adx, atr_pips, vwap_gap, ma_pair = self._tech_context()
                score = 0
                reason = None
                if atr_pips is not None and atr_pips >= self.atr_spike_pips:
                    score += 2
                    if reason is None:
                        reason = "atr_spike"
                if adx is not None and ma_pair is not None:
                    ma10, ma20 = ma_pair
                    gap = abs(ma10 - ma20) / 0.01
                    cross_bad = (side == "long" and ma10 <= ma20) or (side == "short" and ma10 >= ma20)
                    if adx <= self.structure_adx and (cross_bad or gap <= self.structure_gap_pips):
                        score += 2
                        if reason is None:
                            reason = "structure_break"
                if rsi_val is not None:
                    if side == "long" and rsi_val <= self.rsi_fade_long:
                        score += 1
                        if reason is None:
                            reason = "rsi_fade"
                    if side == "short" and rsi_val >= self.rsi_fade_short:
                        score += 1
                        if reason is None:
                            reason = "rsi_fade"
                if vwap_gap is not None and abs(vwap_gap) <= self.vwap_gap_pips:
                    score += 1
                    if reason is None:
                        reason = "vwap_cut"
                if hard_stop_ready and score == 0:
                    await self._close(trade_id, -units, "tech_hard_stop", pnl, client_id)
                    self._states.pop(trade_id, None)
                    return
                if (hard_stop_ready and score > 0) or (soft_ready and score >= self.tech_score_min):
                    await self._close(trade_id, -units, reason or "tech_exit", pnl, client_id)
                    self._states.pop(trade_id, None)
                    return

        profit_take = self.range_profit_take if range_active else self.profit_take
        trail_start = self.range_trail_start if range_active else self.trail_start
        trail_backoff = self.range_trail_backoff if range_active else self.trail_backoff
        lock_buffer = self.range_lock_buffer if range_active else self.lock_buffer

        if state.tp_hint:
            profit_take = max(profit_take, max(1.0, state.tp_hint * 0.9))
            trail_start = max(trail_start, profit_take * 0.9)
            lock_buffer = max(lock_buffer, profit_take * 0.4)

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
            "[EXIT-momb] worker starting (interval=%.2fs tags=%s)",
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

                mid = _latest_mid()
                if mid is None:
                    continue
                rsi = None
                range_active = False
                try:
                    ctx_mid, ctx_rsi, ctx_range = self._context()
                    if ctx_mid is not None:
                        mid = ctx_mid
                    rsi = ctx_rsi
                    range_active = ctx_range
                except Exception:
                    pass

                now = _utc_now()
                for tr in trades:
                    try:
                        await self._review_trade(tr, now, mid, rsi, range_active)
                    except Exception:
                        LOG.exception("[EXIT-momb] review failed trade=%s", tr.get("trade_id"))
                        continue
        except asyncio.CancelledError:
            LOG.info("[EXIT-momb] worker cancelled")
            raise
        finally:
            try:
                self._pos_manager.close()
            except Exception:
                LOG.exception("[EXIT-momb] failed to close PositionManager")


async def micro_momentumburst_exit_worker() -> None:
    worker = MomentumBurstExitWorker()
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(micro_momentumburst_exit_worker())
