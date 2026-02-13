from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Set, Tuple

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

try:
    from workers.common.exit_utils import mark_pnl_pips as _mark_pnl_pips
except Exception:
    _mark_pnl_pips = None

mark_pnl_pips = _mark_pnl_pips


def _env_name(suffix: str) -> str:
    return f"HAR_EXIT_{suffix}"


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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no"}


def _parse_tags() -> Set[str]:
    raw = os.getenv(_env_name("TAGS"), "MicroAdaptiveRevert")
    tags = {v.strip() for v in str(raw).split(",") if str(v).strip()}
    return {v for v in tags if v}


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


def _mark_pnl(entry_price: float, units: int, *, mid: Optional[float] = None) -> Optional[float]:
    if mark_pnl_pips is not None:
        try:
            return mark_pnl_pips(entry_price, units, mid=mid)
        except Exception:
            pass
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


def _exit_thresholds() -> Dict[str, float]:
    return {
        "partial_pips": _float_env(_env_name("PARTIAL_PIPS"), 0.9),
        "full_pips": _float_env(_env_name("FULL_PIPS"), 1.5),
        "trail_start": _float_env(_env_name("TRAIL_START_PIPS"), 1.1),
        "trail_backoff": _float_env(_env_name("TRAIL_BACKOFF_PIPS"), 0.45),
        "max_adverse": _float_env(_env_name("MAX_ADVERSE_PIPS"), 6.5),
        "max_hold_sec": _float_env(_env_name("MAX_HOLD_SEC"), 3600.0),
        "neg_hold_sec": _float_env(_env_name("NEG_HOLD_SEC"), 3000.0),
        "range_partial": _float_env(_env_name("RANGE_PARTIAL_PIPS"), 0.75),
        "range_full": _float_env(_env_name("RANGE_FULL_PIPS"), 1.0),
        "range_trail_start": _float_env(_env_name("RANGE_TRAIL_START_PIPS"), 0.85),
        "range_trail_backoff": _float_env(_env_name("RANGE_TRAIL_BACKOFF_PIPS"), 0.3),
        "range_max_hold": _float_env(_env_name("RANGE_MAX_HOLD_SEC"), 1800.0),
        "vw_gap_revert": _float_env(_env_name("VWAP_GAP_REVERT_PIPS"), 0.35),
        "bb_mid_near": _float_env(_env_name("BB_MID_NEAR_PIPS"), 0.48),
        "min_hold": _float_env(_env_name("MIN_HOLD_SEC"), 18.0),
        "partial_ratio": _float_env(_env_name("PARTIAL_RATIO"), 0.4),
        "partial_min_units": _int_env(_env_name("PARTIAL_MIN_UNITS"), 900),
    }


@dataclass
class _TradeState:
    max_pnl: float = 0.0
    partial_done: bool = False


def _scale_from_notes(note_key: str, raw_default: float, notes: Dict[str, object]) -> float:
    if not isinstance(notes, dict):
        return raw_default
    value = notes.get(note_key)
    if value is None:
        return raw_default
    try:
        return float(value)
    except (TypeError, ValueError):
        return raw_default


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

    pnl = _mark_pnl(entry, units, mid=mid)
    if pnl is None:
        return None

    base_thresholds = _exit_thresholds()
    thesis = trade.get("entry_thesis") or {}
    history_mult = _scale_from_notes("history_lot_mult", 1.0, thesis.get("notes") if isinstance(thesis, dict) else {})

    partial_ratio = max(0.2, min(0.7, base_thresholds["partial_ratio"]))
    partial_units = _int_env(_env_name("PARTIAL_MIN_UNITS"), base_thresholds["partial_min_units"])
    thresholds = {
        "partial_pips": base_thresholds["partial_pips"] * history_mult,
        "full_pips": base_thresholds["full_pips"] * history_mult,
        "trail_start": base_thresholds["trail_start"] * history_mult,
        "trail_backoff": base_thresholds["trail_backoff"],
        "max_adverse": base_thresholds["max_adverse"] * history_mult,
        "max_hold": base_thresholds["range_max_hold"] if range_active else base_thresholds["max_hold_sec"],
        "neg_hold": base_thresholds["neg_hold_sec"],
        "range_partial": base_thresholds["range_partial"],
        "range_full": base_thresholds["range_full"],
        "range_trail_start": base_thresholds["range_trail_start"],
        "range_trail_backoff": base_thresholds["range_trail_backoff"],
    }
    if range_active:
        thresholds.update(
            {
                "partial_pips": base_thresholds["range_partial"],
                "full_pips": base_thresholds["range_full"],
                "trail_start": base_thresholds["range_trail_start"],
                "trail_backoff": base_thresholds["range_trail_backoff"],
                "max_hold": base_thresholds["range_max_hold"],
            }
        )

    if hold_sec < base_thresholds["min_hold"]:
        return None

    state.max_pnl = max(state.max_pnl, pnl)

    partial_pips = thresholds["partial_pips"]
    if not state.partial_done and pnl >= partial_pips:
        close_units = int(abs(units) * partial_ratio)
        if close_units >= partial_units:
            state.partial_done = True
            sign = 1 if units > 0 else -1
            return sign * close_units, "partial_take"

    if pnl >= thresholds["full_pips"]:
        return units, "profit_take"

    if state.max_pnl >= thresholds["trail_start"] and (state.max_pnl - pnl) >= thresholds["trail_backoff"] and pnl > 0:
        return units, "trail_backoff"

    vwap_gap = None
    try:
        vwap_gap = float(fac.get("vwap_gap"))
    except Exception:
        vwap_gap = None
    if vwap_gap is not None and abs(vwap_gap) <= base_thresholds["vw_gap_revert"] and pnl > 0:
        return units, "vwap_revert"

    bb_mid = None
    try:
        bb_mid = float(fac.get("bb_mid") or fac.get("ma20") or 0.0)
    except Exception:
        bb_mid = None
    if bb_mid and abs((mid - bb_mid) / 0.01) <= base_thresholds["bb_mid_near"] and pnl > 0:
        return units, "bb_mid_revert"

    if hold_sec >= thresholds["max_hold"] and pnl > 0:
        return units, "time_profit_take"

    if pnl <= -thresholds["max_adverse"]:
        return units, "max_adverse"
    if pnl < 0 and hold_sec >= thresholds["neg_hold"] and not range_active:
        return units, "time_stop_loss"

    return None


class MicroAdaptiveRevertExitWorker:
    def __init__(self) -> None:
        self.loop_interval = max(0.5, _float_env(_env_name("LOOP_INTERVAL_SEC"), 1.0))
        self.allowed_tags = _parse_tags()
        self.pocket = os.getenv(_env_name("POCKET"), config.POCKET)

        # Lazy import to avoid offline dependencies for factor providers.
        from execution.position_manager import PositionManager

        self._pos_manager = PositionManager()
        self._states: Dict[str, _TradeState] = {}

    def _context(self) -> Tuple[Optional[float], bool, Dict[str, object]]:
        fac_m1 = all_factors().get("M1") or {}
        fac_h4 = all_factors().get("H4") or {}
        range_active = False
        try:
            range_active = bool(detect_range_mode(fac_m1, fac_h4).active)
        except Exception:
            range_active = False
        mid = _latest_mid()
        return mid, range_active, fac_m1

    async def _close(self, trade_id: str, units: int, reason: str, client_id: Optional[str], *, allow_negative: bool) -> None:
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
            LOG.info("[exit-mar] trade=%s units=%s reason=%s", trade_id, units, reason)
        else:
            LOG.error("[exit-mar] close failed trade=%s units=%s reason=%s", trade_id, units, reason)

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

        decision = evaluate_exit(trade, now=now, mid=mid, range_active=range_active, fac=fac, state=state)
        if decision is None:
            return
        close_units, reason = decision
        if close_units == 0:
            return
        allow_negative = reason in {"max_adverse", "time_stop_loss"}
        await self._close(trade_id, close_units, reason, client_id, allow_negative=allow_negative)
        if reason in {"profit_take", "time_profit_take", "vwap_revert", "bb_mid_revert", "trail_backoff", "time_stop_loss", "max_adverse"}:
            self._states.pop(trade_id, None)

    async def run_forever(self) -> None:
        while True:
            await asyncio.sleep(self.loop_interval)
            now = datetime.now(timezone.utc)
            mid, range_active, fac = self._context()
            if mid is None:
                continue
            positions = self._pos_manager.get_open_positions()
            pocket_info = positions.get(self.pocket) or {}
            trades = _filter_trades(pocket_info.get("open_trades") or [], self.allowed_tags)
            for trade in trades:
                try:
                    await self._review_trade(trade, now, mid, range_active, fac)
                except Exception:
                    LOG.exception("[exit-mar] trade review failed")


async def micro_adaptive_revert_exit_worker() -> None:
    worker = MicroAdaptiveRevertExitWorker()
    await worker.run_forever()
