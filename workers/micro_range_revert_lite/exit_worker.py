from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence, Set, Tuple

from analysis.range_guard import detect_range_mode
from indicators.factor_cache import all_factors
from market_data import tick_window
from workers.common.pro_stop import maybe_close_pro_stop

from . import config

LOG = logging.getLogger(__name__)

# Optional: live close_trade import. In replay, this gets patched by the harness.
try:
    from workers.common.exit_utils import close_trade as _close_trade
except Exception:
    _close_trade = None

close_trade = _close_trade  # type: ignore[assignment]

ALLOWED_TAGS: Set[str] = {"RangeRevertLite"}
POCKET = os.getenv("RRL_EXIT_POCKET", "micro")


@dataclass
class _TradeState:
    max_pnl: float = 0.0
    partial_done: bool = False


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
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


def _mark_pnl_pips(entry_price: float, units: int, *, mid: Optional[float] = None) -> Optional[float]:
    if entry_price <= 0 or units == 0:
        return None
    bid = None
    ask = None
    if mid is None:
        tick = tick_window.recent_ticks(seconds=2.0, limit=1)
        if tick:
            try:
                bid = float(tick[-1].get("bid")) if tick[-1].get("bid") is not None else None
                ask = float(tick[-1].get("ask")) if tick[-1].get("ask") is not None else None
                mid = float(tick[-1].get("mid")) if tick[-1].get("mid") is not None else None
            except Exception:
                bid = None
                ask = None
                mid = None
    if units > 0:
        price = bid if bid is not None else mid
        if price is None:
            return None
        return (price - entry_price) / 0.01
    price = ask if ask is not None else mid
    if price is None:
        return None
    return (entry_price - price) / 0.01


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


def _exit_thresholds(range_active: bool) -> dict[str, float]:
    if range_active:
        return {
            "partial_pips": _float_env("RRL_EXIT_RANGE_PARTIAL_PIPS", 0.6),
            "full_pips": _float_env("RRL_EXIT_RANGE_FULL_PIPS", 1.2),
            "trail_start": _float_env("RRL_EXIT_RANGE_TRAIL_START_PIPS", 1.1),
            "trail_backoff": _float_env("RRL_EXIT_RANGE_TRAIL_BACKOFF_PIPS", 0.4),
            "mean_revert_pips": _float_env("RRL_EXIT_RANGE_MEAN_REVERT_PIPS", 0.4),
            "max_hold_sec": _float_env("RRL_EXIT_RANGE_MAX_HOLD_SEC", 1800.0),
        }
    return {
        "partial_pips": _float_env("RRL_EXIT_PARTIAL_PIPS", 0.9),
        "full_pips": _float_env("RRL_EXIT_FULL_PIPS", 1.7),
        "trail_start": _float_env("RRL_EXIT_TRAIL_START_PIPS", 1.5),
        "trail_backoff": _float_env("RRL_EXIT_TRAIL_BACKOFF_PIPS", 0.55),
        "mean_revert_pips": _float_env("RRL_EXIT_MEAN_REVERT_PIPS", 0.6),
        "max_hold_sec": _float_env("RRL_EXIT_MAX_HOLD_SEC", 3600.0),
    }


def evaluate_exit(
    trade: dict,
    *,
    now: datetime,
    mid: float,
    range_active: bool,
    fac: dict,
    state: _TradeState,
) -> Optional[Tuple[int, str]]:
    units = int(trade.get("units", 0) or 0)
    if units == 0:
        return None
    entry = float(trade.get("price") or 0.0)
    if entry <= 0:
        return None

    opened_at = _parse_time(trade.get("open_time"))
    hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0
    min_hold = max(5.0, _float_env("RRL_EXIT_MIN_HOLD_SEC", 20.0))
    if hold_sec < min_hold:
        return None

    pnl = _mark_pnl_pips(entry, units, mid=mid)
    if pnl is None:
        return None

    thresholds = _exit_thresholds(range_active)
    partial_pips = thresholds["partial_pips"]
    full_pips = thresholds["full_pips"]
    trail_start = thresholds["trail_start"]
    trail_backoff = thresholds["trail_backoff"]
    mean_revert_pips = thresholds["mean_revert_pips"]
    max_hold_sec = thresholds["max_hold_sec"]

    state.max_pnl = max(state.max_pnl, pnl)

    partial_ratio = max(0.2, min(0.7, _float_env("RRL_EXIT_PARTIAL_RATIO", 0.4)))
    min_partial_units = _int_env("RRL_EXIT_PARTIAL_MIN_UNITS", 1000)

    if not state.partial_done and pnl >= partial_pips:
        close_units = int(abs(units) * partial_ratio)
        if close_units >= min_partial_units:
            state.partial_done = True
            sign = 1 if units > 0 else -1
            return sign * close_units, "partial_take"

    if pnl >= full_pips:
        return units, "profit_take"

    if state.max_pnl >= trail_start and (state.max_pnl - pnl) >= trail_backoff and pnl > 0:
        return units, "trail_backoff"

    vwap_gap = None
    try:
        vwap_gap = float(fac.get("vwap_gap"))
    except Exception:
        vwap_gap = None
    if vwap_gap is not None and abs(vwap_gap) <= _float_env("RRL_EXIT_VWAP_NEAR", 0.35):
        if pnl >= mean_revert_pips:
            return units, "vwap_revert"

    bb_mid = None
    try:
        bb_mid = float(fac.get("bb_mid") or fac.get("ma20") or 0.0)
    except Exception:
        bb_mid = None
    if bb_mid and abs((mid - bb_mid) / 0.01) <= _float_env("RRL_EXIT_BB_MID_NEAR", 0.5):
        if pnl >= mean_revert_pips:
            return units, "bb_mid_revert"

    if hold_sec >= max_hold_sec and pnl > 0:
        return units, "time_stop"

    max_adverse = _float_env("RRL_EXIT_MAX_ADVERSE_PIPS", 6.0)
    neg_hold_sec = _float_env("RRL_EXIT_NEG_HOLD_SEC", max_hold_sec * 1.2)
    range_break_sec = _float_env("RRL_EXIT_RANGE_BREAK_SEC", max_hold_sec * 0.6)
    if pnl <= -max_adverse:
        return units, "max_adverse"
    if pnl < 0 and hold_sec >= neg_hold_sec and not range_active:
        return units, "time_stop_loss"
    if pnl < 0 and hold_sec >= range_break_sec and not range_active:
        return units, "range_break_stop"

    return None


class MicroRangeRevertLiteExitWorker:
    def __init__(self) -> None:
        self.loop_interval = max(0.5, _float_env("RRL_EXIT_LOOP_INTERVAL_SEC", 1.0))
        # Lazy import to avoid secrets/env.toml dependency in offline replays.
        from execution.position_manager import PositionManager

        self._pos_manager = PositionManager()
        self._states: dict[str, _TradeState] = {}

    def _context(self) -> Tuple[Optional[float], bool, dict]:
        fac_m1 = all_factors().get("M1") or {}
        fac_h4 = all_factors().get("H4") or {}
        range_active = False
        try:
            range_active = bool(detect_range_mode(fac_m1, fac_h4).active)
        except Exception:
            range_active = False
        mid = _latest_mid()
        return mid, range_active, fac_m1

    async def _close(
        self,
        trade_id: str,
        units: int,
        reason: str,
        client_id: Optional[str],
        *,
        allow_negative: bool,
    ) -> None:
        if close_trade is None:
            return
        ok = await close_trade(
            trade_id,
            units,
            client_order_id=client_id,
            allow_negative=allow_negative,
            exit_reason=reason,
            env_prefix=config.ENV_PREFIX,
        )
        if ok:
            LOG.info("[exit-rrl] trade=%s units=%s reason=%s", trade_id, units, reason)
        else:
            LOG.error("[exit-rrl] close failed trade=%s units=%s reason=%s", trade_id, units, reason)

    async def _review_trade(self, trade: dict, now: datetime, mid: float, range_active: bool, fac: dict) -> None:
        trade_id = str(trade.get("trade_id"))
        if not trade_id:
            return
        client_id = trade.get("client_order_id")
        client_ext = trade.get("clientExtensions")
        if not client_id and isinstance(client_ext, dict):
            client_id = client_ext.get("id")
        if not client_id:
            return
        if await maybe_close_pro_stop(trade, now=now):
            return

        state = self._states.get(trade_id)
        if state is None:
            state = _TradeState()
            self._states[trade_id] = state

        decision = evaluate_exit(
            trade,
            now=now,
            mid=mid,
            range_active=range_active,
            fac=fac,
            state=state,
        )
        if decision is None:
            return
        close_units, reason = decision
        if close_units == 0:
            return
        allow_negative = reason in {"max_adverse", "time_stop_loss", "range_break_stop"}
        await self._close(trade_id, close_units, reason, client_id, allow_negative=allow_negative)

    async def run_forever(self) -> None:
        while True:
            await asyncio.sleep(self.loop_interval)
            now = datetime.now(timezone.utc)
            mid, range_active, fac = self._context()
            if mid is None:
                continue
            positions = self._pos_manager.get_open_positions()
            pocket_info = positions.get(POCKET) or {}
            trades = _filter_trades(pocket_info.get("open_trades") or [], ALLOWED_TAGS)
            for trade in trades:
                try:
                    await self._review_trade(trade, now, mid, range_active, fac)
                except Exception:
                    LOG.exception("[exit-rrl] trade review failed")


async def micro_range_revert_lite_exit_worker() -> None:
    worker = MicroRangeRevertLiteExitWorker()
    await worker.run_forever()
