"""Dedicated exit manager for scalp strategies."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Dict, Optional

from execution.order_manager import close_trade
from execution.position_manager import PositionManager
from indicators.factor_cache import all_factors
from market_data import tick_window
from analysis import policy_bus

from . import config

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
        except (KeyError, TypeError, ValueError):
            pass
    try:
        return float(all_factors().get("M1", {}).get("close"))
    except (TypeError, ValueError):
        return None


def _z_score(candles: list[dict], price: float) -> Optional[float]:
    closes = [float(c.get("close")) for c in candles[-20:] if c.get("close") is not None]
    if len(closes) < 20:
        return None
    sample = closes[-20:]
    mu = mean(sample)
    sigma = pstdev(sample)
    if sigma == 0:
        return 0.0
    return (price - mu) / sigma


class _TradeState:
    __slots__ = ("peak", "lock_floor")

    def __init__(self, initial_pips: float) -> None:
        self.peak = initial_pips
        self.lock_floor: Optional[float] = None

    def update(self, pnl: float) -> None:
        self.peak = max(self.peak, pnl)


class ScalpExitManager:
    def __init__(self) -> None:
        self._states: Dict[str, _TradeState] = {}
        self._policy: Dict[str, dict] = {}

    def cleanup(self, active_ids: set[str]) -> None:
        for tid in list(self._states.keys()):
            if tid not in active_ids:
                self._states.pop(tid, None)

    def update_policy(self, policy: Optional[dict]) -> None:
        if policy is None:
            self._policy = {}
        else:
            # shallow copy to avoid caller mutations
            self._policy = dict(policy)

    def evaluate(self, trade: dict, m1_factors: dict, now: datetime) -> Optional[str]:
        trade_id = str(trade.get("trade_id"))
        if not trade_id:
            return None
        units = float(trade.get("units", 0) or 0)
        if units == 0:
            return None
        thesis = trade.get("entry_thesis") or {}
        strategy_tag = thesis.get("strategy_tag")
        if config.STRATEGY_WHITELIST and (not strategy_tag or strategy_tag not in config.STRATEGY_WHITELIST):
            return None

        side = "long" if units > 0 else "short"
        entry_price = float(trade.get("price") or 0.0)
        current_price = _latest_mid()
        if current_price is None or entry_price <= 0.0:
            return None
        pnl_pips = (current_price - entry_price) * 100.0 if side == "long" else (entry_price - current_price) * 100.0

        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0

        state = self._states.get(trade_id)
        if state is None:
            state = _TradeState(initial_pips=pnl_pips)
            self._states[trade_id] = state
        else:
            state.update(pnl_pips)

        candles = m1_factors.get("candles") or []
        z_m1 = _z_score(candles, current_price) if candles else None
        try:
            rsi = float(m1_factors.get("rsi", 50.0))
        except (TypeError, ValueError):
            rsi = 50.0

        policy = self._policy or {}
        exit_policy = policy.get("exit_profile", {}) if isinstance(policy, dict) else {}
        allow_negative_exit = bool(exit_policy.get("allow_negative_exit", True))
        be_profile = policy.get("be_profile", {}) if isinstance(policy, dict) else {}
        partial_profile = policy.get("partial_profile", {}) if isinstance(policy, dict) else {}
        thresholds = partial_profile.get("thresholds_pips") if isinstance(partial_profile, dict) else None
        base_profit_pips = config.BASE_PROFIT_PIPS
        if thresholds:
            try:
                base_profit_pips = max(config.BASE_PROFIT_PIPS, float(thresholds[0]))
            except (TypeError, ValueError):
                base_profit_pips = config.BASE_PROFIT_PIPS
        hard_profit_pips = max(base_profit_pips + 0.5, config.HARD_PROFIT_PIPS)
        lock_trigger_pips = be_profile.get("trigger_pips") if isinstance(be_profile, dict) else None
        try:
            lock_trigger_pips = float(lock_trigger_pips) if lock_trigger_pips is not None else None
        except (TypeError, ValueError):
            lock_trigger_pips = None
        if lock_trigger_pips is None:
            lock_trigger_pips = max(base_profit_pips, config.LOCK_AT_PROFIT_PIPS)
        lock_buffer_pips = be_profile.get("min_lock_pips") if isinstance(be_profile, dict) else None
        try:
            lock_buffer_pips = float(lock_buffer_pips) if lock_buffer_pips is not None else None
        except (TypeError, ValueError):
            lock_buffer_pips = None
        if lock_buffer_pips is None:
            lock_buffer_pips = config.LOCK_BUFFER_PIPS
        trail_start_pips = config.TRAIL_START_PIPS
        if thresholds and len(thresholds) > 1:
            try:
                trail_start_pips = max(config.TRAIL_START_PIPS, float(thresholds[1]))
            except (TypeError, ValueError):
                trail_start_pips = config.TRAIL_START_PIPS

        hard_stop_pips = config.HARD_STOP_PIPS

        if pnl_pips <= -hard_stop_pips:
            if allow_negative_exit:
                return "scalp_hard_stop"
            return None
        if hold_sec >= config.NEGATIVE_HOLD_TIMEOUT_SEC and pnl_pips < 0.0:
            if allow_negative_exit:
                return "scalp_time_stop"
            return None
        if hold_sec >= config.MAX_HOLD_SEC:
            return "scalp_max_hold"

        if pnl_pips >= lock_trigger_pips and state.lock_floor is None:
            state.lock_floor = max(0.2, pnl_pips - lock_buffer_pips)
        if state.lock_floor is not None and pnl_pips <= state.lock_floor:
            return "scalp_lock_release"

        if state.peak >= trail_start_pips and pnl_pips <= state.peak - config.TRAIL_BACKOFF_PIPS:
            return "scalp_trail_take"

        profit_ready = pnl_pips >= base_profit_pips
        z_signal = False
        if z_m1 is not None:
            z_signal = z_m1 <= -config.PROFIT_Z_THRESHOLD if side == "short" else z_m1 >= config.PROFIT_Z_THRESHOLD
        rsi_signal = rsi <= config.RSI_EXIT_SHORT if side == "short" else rsi >= config.RSI_EXIT_LONG
        if (profit_ready and (z_signal or rsi_signal)) or pnl_pips >= hard_profit_pips:
            return "scalp_take_profit"

        return None


async def scalp_exit_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    LOG.info("%s worker starting", config.LOG_PREFIX)
    manager = ScalpExitManager()
    pos_manager = PositionManager()
    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            try:
                positions = pos_manager.get_open_positions()
            except Exception as exc:
                LOG.warning("%s position fetch failed: %s", config.LOG_PREFIX, exc)
                continue

            plan_snapshot = policy_bus.latest()
            scalp_policy = None
            if plan_snapshot and getattr(plan_snapshot, "pockets", None):
                scalp_policy = plan_snapshot.pockets.get("scalp")
            manager.update_policy(scalp_policy)

            scalp_info = positions.get("scalp") or {}
            trades = scalp_info.get("open_trades") or []
            active_ids = {str(tr.get("trade_id")) for tr in trades if tr.get("trade_id")}
            manager.cleanup(active_ids)
            if not trades:
                continue

            factors = all_factors().get("M1", {})
            now = _utc_now()
            for tr in trades:
                reason = manager.evaluate(tr, m1_factors=factors, now=now)
                if not reason:
                    continue
                trade_id = str(tr.get("trade_id"))
                LOG.info(
                    "%s closing trade=%s units=%s reason=%s",
                    config.LOG_PREFIX,
                    trade_id,
                    tr.get("units"),
                    reason,
                )
                try:
                    ok = await close_trade(trade_id)
                except Exception as exc:  # noqa: BLE001
                    LOG.error("%s close failed trade=%s err=%s", config.LOG_PREFIX, trade_id, exc)
                    continue
                if ok:
                    manager.cleanup(active_ids - {trade_id})
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            pos_manager.close()
        except Exception:
            LOG.exception("%s failed to close PositionManager", config.LOG_PREFIX)
