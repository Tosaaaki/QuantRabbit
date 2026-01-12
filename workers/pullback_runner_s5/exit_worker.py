"""Exit loop for pullback_runner_s5 worker (scalp pocket) with technical filters."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from analysis.range_guard import detect_range_mode
from workers.common.exit_utils import close_trade
from execution.position_manager import PositionManager
from execution.section_axis import evaluate_section_exit
from indicators.factor_cache import all_factors
from market_data import tick_window
from utils.metrics_logger import log_metric
from workers.common.exit_scaling import TPScaleConfig, apply_tp_virtual_floor

LOG = logging.getLogger(__name__)

ALLOWED_TAGS = {"pullback_runner_s5"}
POCKET = "scalp"


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
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


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


@dataclass
class _Context:
    mid: Optional[float]
    rsi: Optional[float]
    adx: Optional[float]
    bbw: Optional[float]
    atr_pips: Optional[float]
    vwap_gap_pips: Optional[float]
    range_active: bool


class PullbackRunnerExitWorker:
    """PnL + RSI/ATR/VWAP/レンジ判定を組み合わせた pullback_runner_s5 EXIT."""

    def __init__(self) -> None:
        self.tp_scale = TPScaleConfig()
        self.loop_interval = max(0.3, _float_env("PULLBACK_RUNNER_S5_EXIT_LOOP_INTERVAL_SEC", 1.0))
        self._pos_manager = PositionManager()
        self._states: Dict[str, _TradeState] = {}

        self.profit_take = max(1.0, _float_env("PULLBACK_RUNNER_S5_EXIT_PROFIT_PIPS", 2.4))
        self.trail_start = max(1.2, _float_env("PULLBACK_RUNNER_S5_EXIT_TRAIL_START_PIPS", 3.2))
        self.trail_backoff = max(0.3, _float_env("PULLBACK_RUNNER_S5_EXIT_TRAIL_BACKOFF_PIPS", 1.1))
        self.stop_loss = max(0.8, _float_env("PULLBACK_RUNNER_S5_EXIT_STOP_LOSS_PIPS", 1.7))
        self.max_hold_sec = max(60.0, _float_env("PULLBACK_RUNNER_S5_EXIT_MAX_HOLD_SEC", 25 * 60))
        self.lock_trigger = max(0.6, _float_env("PULLBACK_RUNNER_S5_EXIT_LOCK_TRIGGER_PIPS", 1.1))
        self.lock_buffer = max(0.1, _float_env("PULLBACK_RUNNER_S5_EXIT_LOCK_BUFFER_PIPS", 0.6))

        self.range_profit_take = max(0.8, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_PROFIT_PIPS", 1.8))
        self.range_trail_start = max(1.0, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_TRAIL_START_PIPS", 2.3))
        self.range_trail_backoff = max(0.2, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_TRAIL_BACKOFF_PIPS", 0.7))
        self.range_stop_loss = max(0.6, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_STOP_LOSS_PIPS", 1.1))
        self.range_max_hold_sec = max(45.0, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_MAX_HOLD_SEC", 17 * 60))
        self.range_lock_trigger = max(0.35, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_LOCK_TRIGGER_PIPS", 0.8))
        self.range_lock_buffer = max(0.1, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_LOCK_BUFFER_PIPS", 0.35))

        self.range_adx = max(5.0, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_ADX", 22.0))
        self.range_bbw = max(0.02, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_BBW", 0.20))
        self.range_atr = max(0.4, _float_env("PULLBACK_RUNNER_S5_EXIT_RANGE_ATR", 6.0))

        self.rsi_fade_long = _float_env("PULLBACK_RUNNER_S5_EXIT_RSI_FADE_LONG", 42.0)
        self.rsi_fade_short = _float_env("PULLBACK_RUNNER_S5_EXIT_RSI_FADE_SHORT", 58.0)
        self.rsi_take_long = _float_env("PULLBACK_RUNNER_S5_EXIT_RSI_TAKE_LONG", 70.0)
        self.rsi_take_short = _float_env("PULLBACK_RUNNER_S5_EXIT_RSI_TAKE_SHORT", 30.0)
        self.negative_hold_sec = max(20.0, _float_env("PULLBACK_RUNNER_S5_EXIT_NEG_HOLD_SEC", 150.0))
        self.allow_negative_exit = _bool_env("PULLBACK_RUNNER_S5_EXIT_ALLOW_NEGATIVE", True)

        self.vwap_grab_gap = max(0.15, _float_env("PULLBACK_RUNNER_S5_EXIT_VWAP_GAP_PIPS", 0.9))

        self.atr_hot = max(0.5, _float_env("PULLBACK_RUNNER_S5_EXIT_ATR_HOT_PIPS", 5.0))
        self.atr_cold = max(0.3, _float_env("PULLBACK_RUNNER_S5_EXIT_ATR_COLD_PIPS", 1.2))

    def _filter_trades(self, trades: list[dict]) -> list[dict]:
        if not ALLOWED_TAGS:
            return trades
        filtered: list[dict] = []
        for tr in trades:
            thesis = tr.get("entry_thesis") or {}
            tag = thesis.get("strategy_tag") or thesis.get("strategy") or tr.get("strategy")
            if tag and str(tag) in ALLOWED_TAGS:
                filtered.append(tr)
        return filtered

    def _context(self) -> _Context:
        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}

        def _safe_float(val: object, default: Optional[float] = None) -> Optional[float]:
            if val is None:
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        atr_pips = _safe_float(fac_m1.get("atr_pips"))
        if atr_pips is None:
            atr_pips = _safe_float(fac_m1.get("atr"), 0.0)
            if atr_pips is not None:
                atr_pips *= 100.0

        range_ctx = detect_range_mode(
            fac_m1,
            fac_h4,
            adx_threshold=self.range_adx,
            bbw_threshold=self.range_bbw,
            atr_threshold=self.range_atr,
        )

        return _Context(
            mid=_latest_mid(),
            rsi=_safe_float(fac_m1.get("rsi"), None),
            adx=_safe_float(fac_m1.get("adx"), None),
            bbw=_safe_float(fac_m1.get("bbw"), None),
            atr_pips=atr_pips,
            vwap_gap_pips=_safe_float(fac_m1.get("vwap_gap"), None),
            range_active=bool(range_ctx.active),
        )

    async def _close(
        self,
        trade_id: str,
        units: int,
        reason: str,
        pnl: float,
        side: str,
        range_mode: bool,
        client_id: str,
        allow_negative: bool = False,
    ) -> bool:
        ok = await close_trade(
            trade_id,
            units,
            client_order_id=client_id,
            allow_negative=allow_negative,
        )
        if ok:
            LOG.info(
                "[EXIT-pullback_runner_s5] trade=%s units=%s reason=%s pnl=%.2fp range=%s",
                trade_id,
                units,
                reason,
                pnl,
                range_mode,
            )
            log_metric(
                "pullback_runner_s5_exit",
                pnl,
                tags={"reason": reason, "range": str(range_mode), "side": side},
                ts=_utc_now(),
            )
        else:
            LOG.error("[EXIT-pullback_runner_s5] close failed trade=%s units=%s reason=%s", trade_id, units, reason)
        return ok

    def _evaluate(self, trade: dict, ctx: _Context, now: datetime) -> Optional[str]:
        trade_id = str(trade.get("trade_id"))
        if not trade_id or ctx.mid is None:
            return None

        units = int(trade.get("units", 0) or 0)
        if units == 0:
            return None

        entry_price = float(trade.get("price") or 0.0)
        if entry_price <= 0.0:
            return None

        side = "long" if units > 0 else "short"
        pnl = (ctx.mid - entry_price) * 100.0 if side == "long" else (entry_price - ctx.mid) * 100.0
        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0

        section_decision = evaluate_section_exit(
            trade,
            current_price=ctx.mid,
            now=now,
            side=side,
            pocket=POCKET,
            hold_sec=hold_sec,
            entry_price=entry_price,
        )
        if section_decision.should_exit and section_decision.reason:
            LOG.info(
                "[EXIT-pullback_runner_s5] section_exit trade=%s reason=%s detail=%s",
                trade_id,
                section_decision.reason,
                section_decision.debug,
            )
            return section_decision.reason

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
            state = _TradeState(peak=pnl, hard_stop=hard_stop_val, tp_hint=tp_hint_val)
            self._states[trade_id] = state
        else:
            state.peak = max(state.peak, pnl)

        range_mode = ctx.range_active or (
            ctx.adx is not None and ctx.adx <= self.range_adx and ctx.bbw is not None and ctx.bbw <= self.range_bbw
        )

        profit_take = self.range_profit_take if range_mode else self.profit_take
        trail_start = self.range_trail_start if range_mode else self.trail_start
        trail_backoff = self.range_trail_backoff if range_mode else self.trail_backoff
        stop_loss = self.range_stop_loss if range_mode else self.stop_loss
        lock_trigger = self.range_lock_trigger if range_mode else self.lock_trigger
        lock_buffer = self.range_lock_buffer if range_mode else self.lock_buffer
        max_hold = self.range_max_hold_sec if range_mode else self.max_hold_sec

        # エントリーメタに合わせてEXIT閾値をスケール
        if state.hard_stop:
            stop_loss = max(stop_loss, max(0.8, state.hard_stop * 0.5))
            lock_trigger = max(lock_trigger, max(0.25, state.hard_stop * 0.25))
            trail_start = max(trail_start, max(1.0, state.hard_stop * 0.6))
            max_hold = max(max_hold, self.max_hold_sec * 1.05)

        profit_take, trail_start, lock_buffer, stop_loss = apply_tp_virtual_floor(
            profit_take,
            trail_start,
            lock_buffer,
            stop_loss,
            state,
            self.tp_scale,
        )

        atr = ctx.atr_pips or 0.0
        if atr >= self.atr_hot:
            profit_take += 0.3
            trail_start += 0.3
        elif 0.0 < atr <= self.atr_cold:
            profit_take = max(1.0, profit_take * 0.9)
            stop_loss = max(0.7, stop_loss * 0.9)
        lock_buffer = max(lock_buffer, stop_loss * 0.35)

        # 構造崩れ（M1 MA逆転/ADX低下＋ギャップ縮小）で撤退
        if ctx.adx is not None and ctx.adx < self.range_adx:
            try:
                factors = all_factors().get("M1") or {}
                ma10 = float(factors.get("ma10"))
                ma20 = float(factors.get("ma20"))
                gap = abs(ma10 - ma20) / 0.01
                if (side == "long" and ma10 <= ma20) or (side == "short" and ma10 >= ma20) or (ctx.adx < 14.0 and gap < 2.0):
                    return "structure_break"
            except Exception:
                pass

        if pnl <= -stop_loss:
            return "hard_stop"

        if pnl < 0 and hold_sec >= self.negative_hold_sec:
            if self.allow_negative_exit:
                return "time_cut"

        if pnl < 0:
            if side == "long" and ctx.rsi is not None and ctx.rsi <= self.rsi_fade_long:
                if self.allow_negative_exit or atr >= self.atr_hot:
                    return "rsi_fade"
            if side == "short" and ctx.rsi is not None and ctx.rsi >= self.rsi_fade_short:
                if self.allow_negative_exit or atr >= self.atr_hot:
                    return "rsi_fade"

        if hold_sec >= max_hold and pnl <= profit_take * 0.7:
            return "time_stop"

        if state.lock_floor is None and pnl >= lock_trigger:
            state.lock_floor = max(0.0, pnl - lock_buffer)
        if state.lock_floor is not None and pnl <= state.lock_floor:
            return "lock_release"

        if state.peak >= trail_start and pnl <= state.peak - trail_backoff:
            return "trail_take"

        if ctx.vwap_gap_pips is not None and abs(ctx.vwap_gap_pips) <= self.vwap_grab_gap:
            if pnl > 0.25:
                return "vwap_gravity"
            if pnl < 0 and self.allow_negative_exit:
                return "vwap_cut"

        if pnl >= profit_take * 0.65 and ctx.rsi is not None:
            if side == "long" and ctx.rsi >= self.rsi_take_long:
                return "rsi_take"
            if side == "short" and ctx.rsi <= self.rsi_take_short:
                return "rsi_take"

        if range_mode and pnl >= profit_take:
            return "range_take"

        if pnl >= profit_take:
            return "take_profit"

        return None

    async def run(self) -> None:
        LOG.info(
            "[EXIT-pullback_runner_s5] worker starting (interval=%.2fs tags=%s)",
            self.loop_interval,
            ",".join(sorted(ALLOWED_TAGS)),
        )
        try:
            while True:
                await asyncio.sleep(self.loop_interval)
                positions = self._pos_manager.get_open_positions()
                pocket_info = positions.get(POCKET) or {}
                trades = self._filter_trades(pocket_info.get("open_trades") or [])
                active_ids = {str(tr.get("trade_id")) for tr in trades if tr.get("trade_id")}
                for tid in list(self._states.keys()):
                    if tid not in active_ids:
                        self._states.pop(tid, None)
                if not trades:
                    continue

                ctx = self._context()
                if ctx.mid is None:
                    continue

                now = _utc_now()
                for tr in trades:
                    try:
                        reason = self._evaluate(tr, ctx, now)
                    except Exception:
                        LOG.exception("[EXIT-pullback_runner_s5] evaluate failed trade=%s", tr.get("trade_id"))
                        continue
                    if not reason:
                        continue
                    trade_id = str(tr.get("trade_id"))
                    units = int(tr.get("units", 0) or 0)
                    client_id = _client_id(tr)
                    if not client_id:
                        LOG.warning("[EXIT-pullback_runner_s5] missing client_id trade=%s skip close", trade_id)
                        continue
                    side = "long" if units > 0 else "short"
                    pnl = (ctx.mid - float(tr.get("price") or 0.0)) * 100.0 if side == "long" else (float(tr.get("price") or 0.0) - ctx.mid) * 100.0
                    allow_negative = reason.startswith("section_")
                    await self._close(
                        trade_id,
                        -units,
                        reason,
                        pnl,
                        side,
                        ctx.range_active,
                        client_id,
                        allow_negative=allow_negative,
                    )
                    self._states.pop(trade_id, None)
        except asyncio.CancelledError:
            LOG.info("[EXIT-pullback_runner_s5] worker cancelled")
            raise
        finally:
            try:
                self._pos_manager.close()
            except Exception:
                LOG.exception("[EXIT-pullback_runner_s5] failed to close PositionManager")


async def pullback_runner_s5_exit_worker() -> None:
    worker = PullbackRunnerExitWorker()
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(pullback_runner_s5_exit_worker())
