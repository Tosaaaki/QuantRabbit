from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from analysis.range_guard import detect_range_mode
from analysis.technique_engine import evaluate_exit_techniques
from execution.position_manager import PositionManager
from indicators.factor_cache import all_factors
from utils.market_hours import is_market_open
from utils.metrics_logger import log_metric
from utils.oanda_account import get_account_snapshot, get_position_summary
from workers.common.exit_utils import close_trade, mark_pnl_pips
from workers.common.pro_stop import maybe_close_pro_stop

from . import config

try:  # optional in offline/backtest
    from market_data import tick_window
except Exception:  # pragma: no cover
    tick_window = None

LOG = logging.getLogger(__name__)


@dataclass
class ReliefState:
    partial_done: dict[str, float]


@dataclass
class Candidate:
    trade_id: str
    units: int
    pnl_pips: float
    hold_sec: float
    client_id: str
    pocket: str
    time_stop: bool
    tech_exit: bool
    tech_reason: Optional[str]
    opened_at: datetime


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _latest_mid() -> Optional[float]:
    if tick_window is not None:
        try:
            ticks = tick_window.recent_ticks(seconds=2.0, limit=1)
        except Exception:
            ticks = None
        if ticks:
            try:
                return float(ticks[-1]["mid"])
            except Exception:
                pass
    try:
        return float(all_factors().get("M1", {}).get("close"))
    except Exception:
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _eval_tech_exit(trade: dict, side: str, pocket: str, current_price: Optional[float]) -> tuple[bool, Optional[str]]:
    if not config.TECH_EXIT_ENABLED:
        return False, None
    if current_price is None or current_price <= 0:
        return False, None
    try:
        tech = evaluate_exit_techniques(
            trade=trade,
            current_price=current_price,
            side=side,
            pocket=pocket,
        )
    except Exception:
        LOG.exception("%s tech exit evaluation failed trade=%s", config.LOG_PREFIX, trade.get("trade_id"))
        return False, None
    if tech and getattr(tech, "should_exit", False):
        return True, getattr(tech, "reason", None)
    return False, None


def _parse_ts(raw: str) -> Optional[float]:
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass


def _load_cutoff_ts() -> float:
    raw = os.getenv("MARGIN_RELIEF_START_TS")
    ts = _parse_ts(raw) if raw else None
    if ts:
        return ts
    if config.START_TS_PATH:
        try:
            with open(config.START_TS_PATH, "r", encoding="utf-8") as fh:
                ts = _parse_ts(fh.read().strip())
                if ts:
                    return ts
        except FileNotFoundError:
            pass
        except Exception:
            pass
    ts = time.time()
    if config.START_TS_PATH:
        _ensure_dir(config.START_TS_PATH)
        try:
            with open(config.START_TS_PATH, "w", encoding="utf-8") as fh:
                fh.write(f"{ts:.3f}")
        except Exception:
            pass
    return ts


def _load_state() -> ReliefState:
    if not config.STATE_PATH:
        return ReliefState(partial_done={})
    try:
        with open(config.STATE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return ReliefState(partial_done={})
    except Exception:
        return ReliefState(partial_done={})
    done = {}
    try:
        raw_done = payload.get("partial_done") if isinstance(payload, dict) else None
        if isinstance(raw_done, dict):
            for k, v in raw_done.items():
                try:
                    done[str(k)] = float(v)
                except Exception:
                    continue
    except Exception:
        done = {}
    return ReliefState(partial_done=done)


def _save_state(state: ReliefState) -> None:
    if not config.STATE_PATH:
        return
    _ensure_dir(config.STATE_PATH)
    payload = {"partial_done": state.partial_done}
    tmp_path = f"{config.STATE_PATH}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_path, config.STATE_PATH)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _prune_state(state: ReliefState) -> None:
    if config.STATE_TTL_SEC <= 0:
        return
    now = time.time()
    stale = [k for k, ts in state.partial_done.items() if (now - ts) > config.STATE_TTL_SEC]
    for k in stale:
        state.partial_done.pop(k, None)


def _range_active() -> bool:
    fac_m1 = all_factors().get("M1") or {}
    fac_h4 = all_factors().get("H4") or {}
    try:
        ctx = detect_range_mode(fac_m1, fac_h4)
    except Exception:
        ctx = None
    return bool(getattr(ctx, "active", False))


def _loss_trigger(value: float) -> float:
    if value <= 0:
        return value
    return -abs(value)


def _eligible_trade(
    *,
    trade: dict,
    pocket: str,
    side_bias: str,
    min_hold_sec: float,
    max_hold_sec: float,
    loss_trigger: float,
    cutoff_ts: float,
    mid: Optional[float],
    current_price: Optional[float],
) -> Optional[Candidate]:
    trade_id = str(trade.get("trade_id") or "")
    if not trade_id:
        return None
    units = int(trade.get("units") or 0)
    if units == 0:
        return None
    side = "long" if units > 0 else "short"
    if side != side_bias:
        return None
    client_id = trade.get("client_order_id") or trade.get("client_id")
    if not client_id:
        return None
    opened_at = _parse_time(trade.get("open_time"))
    if not opened_at:
        return None
    if opened_at.timestamp() < cutoff_ts:
        return None
    now = datetime.now(timezone.utc)
    hold_sec = max(0.0, (now - opened_at).total_seconds())
    entry_price = float(trade.get("price") or 0.0)
    if entry_price <= 0:
        return None
    pnl_pips = mark_pnl_pips(entry_price, units, mid=mid)
    if pnl_pips is None:
        try:
            pnl_pips = float(trade.get("unrealized_pl_pips"))
        except Exception:
            pnl_pips = None
    if pnl_pips is None:
        return None
    if pnl_pips > loss_trigger:
        return None
    time_stop = hold_sec >= max_hold_sec
    tech_exit = False
    tech_reason = None
    if config.TECH_EXIT_ENABLED and hold_sec >= config.TECH_EXIT_MIN_HOLD_SEC:
        tech_exit, tech_reason = _eval_tech_exit(trade, side, pocket, current_price)
    if hold_sec < min_hold_sec and not tech_exit:
        return None
    if config.TECH_EXIT_ENABLED and config.TECH_EXIT_REQUIRED and not (tech_exit or time_stop):
        return None
    return Candidate(
        trade_id=trade_id,
        units=units,
        pnl_pips=float(pnl_pips),
        hold_sec=hold_sec,
        client_id=str(client_id),
        pocket=pocket,
        time_stop=time_stop,
        tech_exit=tech_exit,
        tech_reason=tech_reason,
        opened_at=opened_at,
    )


async def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if not config.ENABLED:
        LOG.info("%s disabled via config", config.LOG_PREFIX)
        return
    cutoff_ts = _load_cutoff_ts()
    state = _load_state()
    _prune_state(state)
    _save_state(state)
    LOG.info(
        "%s start interval=%.1fs trigger_usage=%.2f trigger_free=%.3f cutoff=%s",
        config.LOG_PREFIX,
        config.LOOP_INTERVAL_SEC,
        config.TRIGGER_MARGIN_USAGE,
        config.TRIGGER_FREE_MARGIN_RATIO,
        datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).isoformat(),
    )

    pos_manager = PositionManager()
    last_action_ts = 0.0
    last_trigger_log = 0.0
    relief_active = False
    relief_reason: Optional[str] = None

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
        if not is_market_open(datetime.utcnow()):
            continue

        try:
            snapshot = get_account_snapshot(cache_ttl_sec=2.0)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s snapshot fetch failed: %s", config.LOG_PREFIX, exc)
            continue

        nav = snapshot.nav or 0.0
        usage_ratio = (snapshot.margin_used / nav) if nav > 0 else None
        free_ratio = snapshot.free_margin_ratio

        trigger_usage = usage_ratio is not None and usage_ratio >= config.TRIGGER_MARGIN_USAGE
        trigger_free = free_ratio is not None and free_ratio <= config.TRIGGER_FREE_MARGIN_RATIO
        if relief_active:
            release_usage = usage_ratio is not None and usage_ratio <= config.RELEASE_MARGIN_USAGE
            release_free = free_ratio is not None and free_ratio >= config.RELEASE_FREE_MARGIN_RATIO
            if release_usage and release_free:
                relief_active = False
                relief_reason = None
        if not relief_active:
            if not (trigger_usage or trigger_free):
                continue
            relief_active = True
        if trigger_free:
            relief_reason = "free_margin_low"
        elif trigger_usage:
            relief_reason = "margin_usage_high"
        reason = relief_reason or ("free_margin_low" if trigger_free else "margin_usage_high")

        try:
            long_units, short_units = get_position_summary(timeout=3.0)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s position fetch failed: %s", config.LOG_PREFIX, exc)
            continue

        gross_units = float(long_units + short_units)
        net_units = float(long_units - short_units)
        if gross_units <= 0:
            continue
        if gross_units < config.MIN_GROSS_UNITS or abs(net_units) < config.MIN_NET_UNITS:
            continue
        bias = abs(net_units) / gross_units if gross_units > 0 else 0.0
        if bias < config.BIAS_THRESHOLD:
            continue
        side_bias = "long" if net_units > 0 else "short"

        now = time.monotonic()
        if now - last_action_ts < config.COOLDOWN_SEC:
            continue

        range_active = _range_active()
        min_hold_sec = config.RANGE_MIN_HOLD_SEC if range_active else config.MIN_HOLD_SEC
        max_hold_sec = config.RANGE_MAX_HOLD_SEC if range_active else config.MAX_HOLD_SEC
        loss_trigger = _loss_trigger(
            config.RANGE_LOSS_TRIGGER_PIPS if range_active else config.LOSS_TRIGGER_PIPS
        )
        partial_fraction = (
            config.RANGE_PARTIAL_FRACTION if range_active else config.PARTIAL_FRACTION
        )

        positions = pos_manager.get_open_positions()
        meta = positions.get("__meta__") if isinstance(positions, dict) else None
        if isinstance(meta, dict) and meta.get("stale"):
            continue

        fac_m1 = all_factors().get("M1") or {}
        mid = _latest_mid()
        current_price = mid if mid is not None else _safe_float(fac_m1.get("close"))
        now = datetime.now(timezone.utc)
        candidates: list[Candidate] = []
        for pocket, info in (positions or {}).items():
            if pocket in {"__net__", "__meta__", "manual", "unknown"}:
                continue
            if config.POCKETS and pocket not in config.POCKETS:
                continue
            if not isinstance(info, dict):
                continue
            trades = info.get("open_trades") or []
            for trade in trades:
                if await maybe_close_pro_stop(trade, now=now, pocket=pocket):
                    continue
                cand = _eligible_trade(
                    trade=trade,
                    pocket=pocket,
                    side_bias=side_bias,
                    min_hold_sec=min_hold_sec,
                    max_hold_sec=max_hold_sec,
                    loss_trigger=loss_trigger,
                    cutoff_ts=cutoff_ts,
                    mid=mid,
                    current_price=current_price,
                )
                if not cand:
                    continue
                if cand.trade_id in state.partial_done and not cand.time_stop:
                    continue
                candidates.append(cand)

        if not candidates:
            now_mono = time.monotonic()
            if now_mono - last_trigger_log >= 30.0:
                log_metric(
                    "margin_relief_trigger",
                    float(bias),
                    tags={
                        "reason": reason,
                        "side": side_bias,
                        "range": str(range_active).lower(),
                    },
                )
                last_trigger_log = now_mono
            continue

        # Prefer tech exit, then time-stop, then longest hold, then worst pnl
        candidates.sort(
            key=lambda c: (0 if c.tech_exit else 1, 0 if c.time_stop else 1, -c.hold_sec, c.pnl_pips)
        )
        acted = 0
        for cand in candidates:
            if acted >= config.MAX_ACTIONS_PER_RUN:
                break
            abs_units = abs(cand.units)
            if abs_units <= 0:
                continue
            if cand.time_stop:
                close_units = abs_units
                action = "time_stop"
            else:
                close_units = int(abs_units * max(0.05, min(0.8, partial_fraction)))
                remaining = abs_units - close_units
                if close_units < config.PARTIAL_MIN_UNITS:
                    continue
                if remaining < config.PARTIAL_MIN_REMAIN:
                    continue
                action = "tech_exit" if cand.tech_exit else "partial"
            ok = await close_trade(
                cand.trade_id,
                close_units,
                client_order_id=cand.client_id,
                allow_negative=True,
                exit_reason=reason,
            )
            if ok:
                acted += 1
                last_action_ts = time.monotonic()
                if action in {"partial", "tech_exit"}:
                    state.partial_done[cand.trade_id] = time.time()
                    _prune_state(state)
                    _save_state(state)
                log_metric(
                    "margin_relief_close",
                    float(cand.pnl_pips),
                    tags={
                        "action": action,
                        "reason": reason,
                        "side": side_bias,
                        "pocket": cand.pocket,
                        "range": str(range_active).lower(),
                        "tech": str(cand.tech_exit).lower(),
                    },
                )
                LOG.info(
                    "%s %s trade=%s units=%d pnl=%.2f hold=%.0fs reason=%s range=%s tech=%s",
                    config.LOG_PREFIX,
                    action,
                    cand.trade_id,
                    close_units,
                    cand.pnl_pips,
                    cand.hold_sec,
                    reason,
                    range_active,
                    cand.tech_reason or "",
                )
            else:
                LOG.warning(
                    "%s close failed trade=%s action=%s", config.LOG_PREFIX, cand.trade_id, action
                )


if __name__ == "__main__":
    asyncio.run(run())
