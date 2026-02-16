#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tick replay with exit_worker logic for core workers.

Targets:
  - fast_scalp (exit_worker)
  - scalp_false_break_fade (exit_worker)
  - scalp_level_reject (exit_worker)
  - micro BB_RSI (exit_worker)
  - macro TrendMA (exit_worker)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indicators import factor_cache
from market_data import tick_window, spread_monitor
from analysis.range_guard import detect_range_mode
from workers.common.air_state import evaluate_air, adjust_signal
from workers.scalp_false_break_fade import exit_worker as sp_false_break_exit
from workers.scalp_false_break_fade import worker as sp_false_break_worker
from workers.scalp_level_reject import exit_worker as sp_level_reject_exit
from workers.scalp_level_reject import worker as sp_level_reject_worker
from utils.market_hours import is_market_open
import utils.oanda_account as oanda_account
import execution.order_manager as order_manager
from execution.risk_guard import clamp_sl_tp, allowed_lot, can_trade
from signals.pocket_allocator import alloc, DEFAULT_SCALP_SHARE
from analysis.focus_decider import decide_focus
from analysis.local_decider import heuristic_decision
from analysis.regime_classifier import classify
import main as main_mod

import workers.micro_bbrsi.exit_worker as bbrsi_exit


PIP = 0.01
PIP_VALUE = PIP


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    text = os.getenv(name)
    if text is None:
        return default
    return text.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_set(name: str, default: str = "") -> set[str]:
    return {
        item.strip().lower()
        for item in (os.getenv(name, default) or "").split(",")
        if item.strip()
    }


@dataclass(frozen=True)
class ReplayScalpConfig:
    MODE: str
    GUARD_BYPASS_MODES: set[str]
    ALLOWLIST_RAW: str
    LOOP_INTERVAL_SEC: float
    POCKET: str
    MAX_OPEN_TRADES: int
    MAX_OPEN_TRADES_GLOBAL: int
    OPEN_TRADES_SCOPE: str
    COOLDOWN_SEC: float
    MIN_ENTRY_CONF: int
    MIN_UNITS: int


def _build_replay_scalp_config() -> ReplayScalpConfig:
    env = os.environ
    return ReplayScalpConfig(
        MODE=env.get("SCALP_REPLAY_MODE", "spread_revert").strip().lower(),
        GUARD_BYPASS_MODES=_env_set(
            "SCALP_REPLAY_GUARD_BYPASS_MODES",
            "",
        ),
        ALLOWLIST_RAW=os.getenv(
            "SCALP_REPLAY_UNIT_ALLOWLIST",
            env.get("SCALP_REPLAY_ALLOWLIST", ""),
        ),
        LOOP_INTERVAL_SEC=_env_float("SCALP_REPLAY_LOOP_INTERVAL_SEC", "4.0"),
        POCKET=env.get("SCALP_REPLAY_POCKET", "scalp").strip() or "scalp",
        MAX_OPEN_TRADES=_env_int("SCALP_REPLAY_MAX_OPEN_TRADES", "2"),
        MAX_OPEN_TRADES_GLOBAL=_env_int(
            "SCALP_REPLAY_MAX_OPEN_TRADES_GLOBAL",
            "0",
        ),
        OPEN_TRADES_SCOPE=(
            env.get("SCALP_REPLAY_OPEN_TRADES_SCOPE", "tag").strip().lower() or "tag"
        ),
        COOLDOWN_SEC=_env_float("SCALP_REPLAY_COOLDOWN_SEC", "45.0"),
        MIN_ENTRY_CONF=_env_int("SCALP_REPLAY_MIN_ENTRY_CONF", "32"),
        MIN_UNITS=_env_int("SCALP_REPLAY_MIN_UNITS", "1000"),
    )


_SP_CFG = _build_replay_scalp_config()


def _unsupported_signal(*_args: object, **_kwargs: object) -> None:
    return None


def _signal_map():
    return {
        "SpreadRangeRevert": _unsupported_signal,
        "RangeFaderPro": _unsupported_signal,
        "DroughtRevert": _unsupported_signal,
        "PrecisionLowVol": _unsupported_signal,
        "VwapRevertS": _unsupported_signal,
        "StochBollBounce": _unsupported_signal,
        "DivergenceRevert": _unsupported_signal,
        "CompressionRetest": _unsupported_signal,
        "HTFPullbackS": _unsupported_signal,
        "MacdTrendRide": _unsupported_signal,
        "EmaSlopePull": _unsupported_signal,
        "TickImbalance": _unsupported_signal,
        "TickImbalanceRRPlus": _unsupported_signal,
        "LevelReject": lambda fac_m1, fac_m5, fac_h1, range_ctx, now: sp_level_reject_worker._signal_level_reject(
            fac_m1,
            range_ctx=range_ctx,
            tag="LevelReject",
        ),
        "TickWickReversal": _unsupported_signal,
        "SessionEdge": _unsupported_signal,
        "SqueezePulseBreak": _unsupported_signal,
        "FalseBreakFade": lambda fac_m1, fac_m5, fac_h1, range_ctx, now: sp_false_break_worker._signal_false_break_fade(
            fac_m1,
            range_ctx,
            tag="FalseBreakFade",
        ),
        "WickReversal": _unsupported_signal,
    }


def parse_iso8601(value: str) -> datetime:
    text = (value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


_FAST_SCALP_AVAILABLE = False
FastScalpReplayer: Any = None
ReplayTick: Any = None
fast_config: Any = None
fast_exit: Any = None
try:
    from scripts.replay_fast_scalp_ticks import (
        FastScalpReplayer as _FastScalpReplayer,
        ReplayTick as _ReplayTick,
    )
    import workers.fast_scalp.config as fast_config
    import workers.fast_scalp.exit_worker as fast_exit

    FastScalpReplayer = _FastScalpReplayer
    ReplayTick = _ReplayTick
    _FAST_SCALP_AVAILABLE = True
except Exception:
    _FAST_SCALP_AVAILABLE = False

trendma_exit: Any = None
try:
    import workers.macro_trendma.exit_worker as trendma_exit
except Exception:
    trendma_exit = None


@dataclass
class TickRow:
    ts: datetime
    epoch: float
    bid: float
    ask: float
    mid: float


class SimClock:
    def __init__(self) -> None:
        self.now = 0.0


class CandleBuilder:
    def __init__(self, minutes: int) -> None:
        self.minutes = minutes
        self.bucket_start: Optional[datetime] = None
        self.open: float = 0.0
        self.high: float = 0.0
        self.low: float = 0.0
        self.close: float = 0.0
        self.volume: int = 0

    def update(self, ts: datetime, price: float, volume: int = 1) -> Optional[dict]:
        bucket = ts.replace(second=0, microsecond=0)
        if self.minutes > 1:
            minute = (bucket.minute // self.minutes) * self.minutes
            bucket = bucket.replace(minute=minute)
        if self.bucket_start is None:
            self.bucket_start = bucket
            self.open = price
            self.high = price
            self.low = price
            self.close = price
            self.volume = volume
            return None
        if bucket != self.bucket_start:
            candle = {
                "time": self.bucket_start,
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
            }
            self.bucket_start = bucket
            self.open = price
            self.high = price
            self.low = price
            self.close = price
            self.volume = volume
            return candle
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume
        return None

    def flush(self) -> Optional[dict]:
        if self.bucket_start is None:
            return None
        candle = {
            "time": self.bucket_start,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
        }
        self.bucket_start = None
        return candle


def _iter_ticks(path: Path, instrument: str, start: Optional[datetime], end: Optional[datetime]) -> Iterable[TickRow]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("instrument") and payload["instrument"] != instrument:
                continue
            bid = payload.get("bid")
            ask = payload.get("ask")
            if bid is None or ask is None:
                continue
            try:
                bid_f = float(bid)
                ask_f = float(ask)
            except (TypeError, ValueError):
                continue
            ts_raw = payload.get("ts") or payload.get("time") or payload.get("timestamp")
            if not ts_raw:
                continue
            ts = parse_iso8601(str(ts_raw))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            mid = (bid_f + ask_f) / 2.0
            yield TickRow(ts=ts, epoch=ts.timestamp(), bid=bid_f, ask=ask_f, mid=mid)


def _reset_replay_state(sim_clock: SimClock) -> None:
    try:
        tick_window._TICKS.clear()  # type: ignore[attr-defined]
        tick_window._reload_cache_if_updated = lambda: None  # type: ignore[assignment]
        tick_window._persist_cache = lambda: None  # type: ignore[assignment]
    except Exception:
        pass

    try:
        spread_monitor._history.clear()  # type: ignore[attr-defined]
        spread_monitor._baseline_history.clear()  # type: ignore[attr-defined]
        spread_monitor._snapshot = None  # type: ignore[attr-defined]
        spread_monitor._blocked_until = 0.0  # type: ignore[attr-defined]
        spread_monitor._blocked_reason = ""  # type: ignore[attr-defined]
        spread_monitor._last_logged_blocked = False  # type: ignore[attr-defined]
        spread_monitor._stale_since = None  # type: ignore[attr-defined]
        spread_monitor.time.monotonic = lambda: sim_clock.now  # type: ignore[assignment]
    except Exception:
        pass

    try:
        for tf in factor_cache._CANDLES:  # type: ignore[attr-defined]
            factor_cache._CANDLES[tf].clear()  # type: ignore[index]
        factor_cache._FACTORS.clear()  # type: ignore[attr-defined]
        factor_cache._LAST_REGIME.clear()  # type: ignore[attr-defined]
        # Use a per-process cache path so parallel replays don't trample each other.
        factor_cache._CACHE_PATH = Path(f"tmp/factor_cache_exit_replay_{os.getpid()}.json")  # type: ignore[attr-defined]
        factor_cache._LAST_RESTORE_MTIME = float("inf")  # type: ignore[attr-defined]
    except Exception:
        pass


_TAG_TO_MODE = {
    "SpreadRangeRevert": "spread_revert",
    "RangeFaderPro": "rangefaderpro",
    "DroughtRevert": "drought_revert",
    "PrecisionLowVol": "precision_lowvol",
    "VwapRevertS": "vwap_revert",
    "StochBollBounce": "stoch_bounce",
    "DivergenceRevert": "divergence_revert",
    "CompressionRetest": "compression_retest",
    "HTFPullbackS": "htf_pullback",
    "MacdTrendRide": "macd_trend",
    "EmaSlopePull": "ema_slope_pull",
    "TickImbalance": "tick_imbalance",
    "TickImbalanceRRPlus": "tick_imbalance_rrplus",
    "LevelReject": "level_reject",
    "WickReversal": "wick_reversal",
    "TickWickReversal": "tick_wick_reversal",
    "SessionEdge": "session_edge",
    "SqueezePulseBreak": "squeeze_pulse_break",
    "FalseBreakFade": "false_break_fade",
}


class SimBroker:
    def __init__(
        self,
        *,
        slip_base_pips: float = 0.0,
        slip_spread_coef: float = 0.0,
        slip_atr_coef: float = 0.0,
        slip_latency_coef: float = 0.0,
        latency_ms: float = 0.0,
        fill_mode: str = "lko",
    ) -> None:
        self._seq = 0
        self.open_trades: Dict[str, dict] = {}
        self.closed_trades: List[dict] = []
        self.last_tick: Optional[TickRow] = None
        self.prev_tick: Optional[TickRow] = None
        self.pending_opens: List[dict] = []
        self.pending_closes: List[dict] = []
        self.on_close = None
        self.hard_sl = True
        self.hard_tp = True
        # Cache strategy protection lookups so replay doesn't waste time re-loading YAML/config
        # on repeated close attempts.
        self._close_guard_cache: Dict[tuple[str, str], dict] = {}
        self.slip_base_pips = float(slip_base_pips)
        self.slip_spread_coef = float(slip_spread_coef)
        self.slip_atr_coef = float(slip_atr_coef)
        self.slip_latency_coef = float(slip_latency_coef)
        self.latency_ms = float(latency_ms)
        self.fill_mode = (fill_mode or "lko").strip().lower()

    def _close_guard_params(self, pocket: Optional[str], strategy_tag: Optional[str]) -> dict:
        key = (str(pocket or "").strip().lower(), str(strategy_tag or "").strip())
        cached = self._close_guard_cache.get(key)
        if isinstance(cached, dict):
            return cached
        try:
            min_profit_pips = order_manager._min_profit_pips(pocket, strategy_tag)  # type: ignore[attr-defined]
        except Exception:
            min_profit_pips = None
        try:
            ratio = order_manager._min_profit_ratio(pocket, strategy_tag)  # type: ignore[attr-defined]
        except Exception:
            ratio = None
        ratio_reasons: set[str] = set()
        tp_min = 0.0
        if ratio is not None and float(ratio) > 0:
            try:
                ratio_reasons = set(order_manager._min_profit_ratio_reasons(strategy_tag))  # type: ignore[attr-defined]
            except Exception:
                ratio_reasons = set()
            try:
                tp_min = float(order_manager._min_profit_ratio_min_tp_pips(strategy_tag))  # type: ignore[attr-defined]
            except Exception:
                tp_min = 0.0
        payload = {
            "min_profit_pips": min_profit_pips,
            "ratio": ratio,
            "ratio_reasons": ratio_reasons,
            "tp_min": tp_min,
        }
        self._close_guard_cache[key] = payload
        return payload

    def set_last_tick(self, tick: TickRow) -> None:
        self.prev_tick = self.last_tick
        self.last_tick = tick
        self._flush_pending(tick)

    def _next_id(self) -> str:
        self._seq += 1
        return f"sim-{self._seq}"

    def _calc_slip_pips(self, tick: Optional[TickRow] = None) -> float:
        tick = tick or self.last_tick
        if tick is None:
            return 0.0
        spread_pips = max(0.0, tick.ask - tick.bid) / PIP
        atr_pips = None
        fac_m1 = factor_cache.all_factors().get("M1") or {}
        if fac_m1:
            try:
                atr_pips = float(fac_m1.get("atr_pips"))
            except Exception:
                atr_pips = None
            if atr_pips is None:
                try:
                    atr_val = float(fac_m1.get("atr"))
                except Exception:
                    atr_val = None
                if atr_val is not None:
                    atr_pips = atr_val * 100.0
        slip = (
            self.slip_base_pips
            + self.slip_spread_coef * spread_pips
            + self.slip_atr_coef * (atr_pips or 0.0)
            + self.slip_latency_coef * self.latency_ms
        )
        return max(0.0, slip)

    @staticmethod
    def _apply_slip(price: float, direction: str, *, is_entry: bool, slip_pips: float) -> float:
        slip = slip_pips * PIP
        if direction == "long":
            return price + slip if is_entry else price - slip
        return price - slip if is_entry else price + slip

    def _resolve_fill_tick(
        self,
        ready_epoch: float,
        current_tick: TickRow,
        prev_tick: Optional[TickRow],
    ) -> TickRow:
        if self.fill_mode == "next_tick":
            return current_tick
        if prev_tick is not None and prev_tick.epoch <= ready_epoch:
            return prev_tick
        return current_tick

    def _create_trade(
        self,
        *,
        trade_id: Optional[str],
        pocket: str,
        strategy_tag: str,
        direction: str,
        entry_price: float,
        entry_time: datetime,
        tp_pips: Optional[float],
        sl_pips: Optional[float],
        timeout_sec: Optional[float],
        units: int,
        source: str,
        entry_thesis: Optional[dict] = None,
        meta: Optional[dict] = None,
        signal_time: Optional[datetime] = None,
        signal_price: Optional[float] = None,
        latency_ms: Optional[float] = None,
        slip_tick: Optional[TickRow] = None,
    ) -> dict:
        trade_id = trade_id or self._next_id()
        signed_units = units if direction == "long" else -units
        client_id = f"sim-{trade_id}"
        slip_pips = 0.0
        if source != "fast_scalp":
            slip_pips = self._calc_slip_pips(slip_tick)
            entry_price = self._apply_slip(entry_price, direction, is_entry=True, slip_pips=slip_pips)
        thesis: dict = {
            "strategy_tag": strategy_tag,
            "strategy": strategy_tag,
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "timeout_sec": timeout_sec,
        }
        if isinstance(entry_thesis, dict) and entry_thesis:
            merged = dict(entry_thesis)
            merged.setdefault("strategy_tag", strategy_tag)
            merged.setdefault("strategy", strategy_tag)
            if tp_pips is not None:
                merged["tp_pips"] = tp_pips
            if sl_pips is not None:
                merged["sl_pips"] = sl_pips
            if timeout_sec is not None:
                merged.setdefault("timeout_sec", timeout_sec)
            thesis = merged
        sl_price = None
        tp_price = None
        if sl_pips and sl_pips > 0:
            if direction == "long":
                sl_price = entry_price - sl_pips * PIP
            else:
                sl_price = entry_price + sl_pips * PIP
        if tp_pips and tp_pips > 0:
            if direction == "long":
                tp_price = entry_price + tp_pips * PIP
            else:
                tp_price = entry_price - tp_pips * PIP
        if sl_price is not None or tp_price is not None:
            sl_price, tp_price = clamp_sl_tp(entry_price, sl_price or entry_price, tp_price or entry_price, direction == "long")
        trade = {
            "trade_id": trade_id,
            "units": signed_units,
            "price": entry_price,
            "open_time": entry_time.isoformat(),
            "client_order_id": client_id,
            "clientExtensions": {"id": client_id},
            "entry_thesis": thesis,
            "strategy_tag": strategy_tag,
            "strategy": strategy_tag,
            "pocket": pocket,
            "source": source,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "entry_slip_pips": slip_pips,
        }
        if isinstance(meta, dict) and meta:
            trade["entry_meta"] = meta
        if signal_time is not None:
            trade["signal_time"] = signal_time.isoformat()
        if signal_price is not None:
            trade["signal_price"] = signal_price
        if latency_ms is not None:
            trade["entry_latency_ms"] = latency_ms
        self.open_trades[trade_id] = trade
        return trade

    def _close_trade_record(
        self,
        trade: dict,
        tick: TickRow,
        reason: str,
        *,
        exit_price_override: Optional[float] = None,
        slip_tick: Optional[TickRow] = None,
        latency_ms: Optional[float] = None,
    ) -> dict:
        units = int(trade.get("units", 0) or 0)
        entry_price = float(trade.get("price") or 0.0)
        if exit_price_override is not None:
            exit_price = exit_price_override
        else:
            exit_price = tick.bid if units > 0 else tick.ask
        slip_pips = 0.0
        if trade.get("source") != "fast_scalp":
            direction = "long" if units > 0 else "short"
            slip_pips = self._calc_slip_pips(slip_tick)
            exit_price = self._apply_slip(exit_price, direction, is_entry=False, slip_pips=slip_pips)
        if units > 0:
            pnl_pips = (exit_price - entry_price) / PIP
        else:
            pnl_pips = (entry_price - exit_price) / PIP
        pnl_jpy = pnl_pips * (abs(units) * PIP)
        record = {
            "trade_id": trade.get("trade_id"),
            "pocket": trade.get("pocket"),
            "strategy_tag": trade.get("strategy_tag"),
            "entry_time": trade.get("open_time"),
            "exit_time": tick.ts.isoformat(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pips": round(pnl_pips, 3),
            "pnl_jpy": round(pnl_jpy, 2),
            "reason": reason,
            "source": trade.get("source"),
            "units": units,
            "exit_slip_pips": slip_pips,
        }
        if latency_ms is not None:
            record["exit_latency_ms"] = latency_ms
        self.closed_trades.append(record)
        if callable(self.on_close):
            self.on_close(trade)
        return record

    def _flush_pending(self, tick: TickRow) -> None:
        if not self.pending_opens and not self.pending_closes:
            return
        now_epoch = tick.epoch
        prev_tick = self.prev_tick

        if self.pending_opens:
            remaining_opens: List[dict] = []
            for pending in self.pending_opens:
                if now_epoch < pending["ready_epoch"]:
                    remaining_opens.append(pending)
                    continue
                fill_tick = self._resolve_fill_tick(pending["ready_epoch"], tick, prev_tick)
                direction = pending["direction"]
                expected_entry = fill_tick.ask if direction == "long" else fill_tick.bid
                if self.fill_mode == "next_tick":
                    fill_epoch = tick.epoch
                else:
                    fill_epoch = min(pending["ready_epoch"], tick.epoch)
                fill_time = datetime.fromtimestamp(fill_epoch, tz=timezone.utc)
                self._create_trade(
                    trade_id=pending["trade_id"],
                    pocket=pending["pocket"],
                    strategy_tag=pending["strategy_tag"],
                    direction=direction,
                    entry_price=expected_entry,
                    entry_time=fill_time,
                    tp_pips=pending.get("tp_pips"),
                    sl_pips=pending.get("sl_pips"),
                    timeout_sec=pending.get("timeout_sec"),
                    units=pending.get("units", 0),
                    source=pending.get("source", ""),
                    entry_thesis=pending.get("entry_thesis"),
                    meta=pending.get("meta"),
                    signal_time=pending.get("signal_time"),
                    signal_price=pending.get("signal_price"),
                    latency_ms=pending.get("latency_ms"),
                    slip_tick=fill_tick,
                )
            self.pending_opens = remaining_opens

        if self.pending_closes:
            remaining_closes: List[dict] = []
            for pending in self.pending_closes:
                if now_epoch < pending["ready_epoch"]:
                    remaining_closes.append(pending)
                    continue
                trade = self.open_trades.pop(pending["trade_id"], None)
                if trade is None:
                    continue
                fill_tick = self._resolve_fill_tick(pending["ready_epoch"], tick, prev_tick)
                self._close_trade_record(
                    trade,
                    fill_tick,
                    pending["reason"],
                    slip_tick=fill_tick,
                    latency_ms=pending.get("latency_ms"),
                )
            self.pending_closes = remaining_closes

    def open_trade(
        self,
        *,
        pocket: str,
        strategy_tag: str,
        direction: str,
        entry_price: float,
        entry_time: datetime,
        tp_pips: Optional[float] = None,
        sl_pips: Optional[float] = None,
        timeout_sec: Optional[float] = None,
        units: int = 10000,
        source: str = "",
        entry_thesis: Optional[dict] = None,
        meta: Optional[dict] = None,
    ) -> dict:
        if self.latency_ms > 0.0 and source != "fast_scalp":
            trade_id = self._next_id()
            ready_epoch = entry_time.timestamp() + (self.latency_ms / 1000.0)
            pending = {
                "trade_id": trade_id,
                "pocket": pocket,
                "strategy_tag": strategy_tag,
                "direction": direction,
                "entry_price": entry_price,
                "signal_price": entry_price,
                "entry_time": entry_time,
                "signal_time": entry_time,
                "tp_pips": tp_pips,
                "sl_pips": sl_pips,
                "timeout_sec": timeout_sec,
                "units": units,
                "source": source,
                "entry_thesis": entry_thesis,
                "meta": meta,
                "ready_epoch": ready_epoch,
                "latency_ms": self.latency_ms,
            }
            self.pending_opens.append(pending)
            return {"trade_id": trade_id, "pending": True, "ready_epoch": ready_epoch}
        return self._create_trade(
            trade_id=None,
            pocket=pocket,
            strategy_tag=strategy_tag,
            direction=direction,
            entry_price=entry_price,
            entry_time=entry_time,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            timeout_sec=timeout_sec,
            units=units,
            source=source,
            entry_thesis=entry_thesis,
            meta=meta,
            latency_ms=0.0,
        )

    def close_trade(self, trade_id: str, reason: str, *, exit_price_override: Optional[float] = None) -> bool:
        trade = self.open_trades.get(trade_id)
        if trade is None:
            return False
        tick = self.last_tick
        if tick is None:
            return False

        force_close = reason == "end_of_replay" or exit_price_override is not None
        if not force_close:
            # Match execution/order_manager.py profit-buffer behavior so replay results reflect
            # production close rejections (min_profit_pips / min_profit_ratio).
            units = int(trade.get("units", 0) or 0)
            entry_price = float(trade.get("price") or 0.0)
            if units != 0 and entry_price > 0:
                if units > 0:
                    est_pips = (tick.bid - entry_price) / PIP
                else:
                    est_pips = (entry_price - tick.ask) / PIP
                pocket = str(trade.get("pocket") or "").strip() or None
                strategy_tag = str(trade.get("strategy_tag") or "").strip() or None
                reason_key = str(reason or "").strip()
                try:
                    force_allow = bool(order_manager._reason_force_allow(reason_key))  # type: ignore[attr-defined]
                except Exception:
                    force_allow = False

                guard = self._close_guard_params(pocket, strategy_tag)
                min_profit_pips = guard.get("min_profit_pips")
                if (
                    min_profit_pips is not None
                    and est_pips >= 0
                    and est_pips < float(min_profit_pips)
                    and not force_allow
                ):
                    return False

                ratio = guard.get("ratio")
                ratio_reasons = guard.get("ratio_reasons") if isinstance(guard.get("ratio_reasons"), set) else set()
                tp_min = float(guard.get("tp_min") or 0.0)
                if ratio is not None and float(ratio) > 0 and est_pips >= 0 and not force_allow:
                    try:
                        matches = bool(
                            reason_key
                            and ratio_reasons
                            and order_manager._reason_matches_tokens(reason_key, list(ratio_reasons))  # type: ignore[attr-defined]
                        )
                    except Exception:
                        matches = False
                    if matches:
                        tp_price = trade.get("tp_price")
                        try:
                            tp_price_f = float(tp_price) if tp_price is not None else None
                        except Exception:
                            tp_price_f = None
                        if tp_price_f is not None:
                            tp_pips = abs(tp_price_f - entry_price) / PIP
                            if tp_pips >= tp_min:
                                min_ratio_pips = tp_pips * float(ratio)
                                if est_pips < min_ratio_pips:
                                    return False

        if not force_close and self.latency_ms > 0.0:
            if trade.get("closing"):
                return True
            ready_epoch = tick.epoch + (self.latency_ms / 1000.0)
            self.pending_closes.append(
                {
                    "trade_id": trade_id,
                    "reason": reason,
                    "ready_epoch": ready_epoch,
                    "latency_ms": self.latency_ms,
                }
            )
            trade["closing"] = True
            trade["close_requested_at"] = tick.ts.isoformat()
            return True
        trade = self.open_trades.pop(trade_id, None)
        if trade is None:
            return False
        self._close_trade_record(
            trade,
            tick,
            reason,
            exit_price_override=exit_price_override,
            slip_tick=tick,
            latency_ms=0.0 if force_close else self.latency_ms,
        )
        return True

    def check_sl_tp(self, tick: TickRow) -> None:
        if not self.hard_sl and not self.hard_tp:
            return
        closing: List[Tuple[str, str, float]] = []
        for trade_id, tr in list(self.open_trades.items()):
            units = int(tr.get("units", 0) or 0)
            if units == 0:
                continue
            direction = "long" if units > 0 else "short"
            sl_price = tr.get("sl_price")
            tp_price = tr.get("tp_price")
            if direction == "long":
                if self.hard_tp and tp_price is not None and tick.bid >= tp_price:
                    closing.append((trade_id, "tp_hit", float(tp_price)))
                elif self.hard_sl and sl_price is not None and tick.bid <= sl_price:
                    closing.append((trade_id, "sl_hit", float(sl_price)))
            else:
                if self.hard_tp and tp_price is not None and tick.ask <= tp_price:
                    closing.append((trade_id, "tp_hit", float(tp_price)))
                elif self.hard_sl and sl_price is not None and tick.ask >= sl_price:
                    closing.append((trade_id, "sl_hit", float(sl_price)))
        for trade_id, reason, price in closing:
            self.close_trade(trade_id, reason, exit_price_override=price)

    def get_open_positions(self) -> Dict[str, dict]:
        pockets: Dict[str, dict] = {}
        for tr in self.open_trades.values():
            pocket = tr.get("pocket") or "micro"
            payload = pockets.setdefault(pocket, {"open_trades": []})
            payload["open_trades"].append(tr)
        return pockets


class SimAccount:
    def __init__(self, *, balance: float, margin_rate: float) -> None:
        self.balance = float(balance)
        self.margin_rate = float(margin_rate)
        self.unrealized_pl = 0.0

    def _calc_unrealized(self, broker: SimBroker) -> float:
        tick = broker.last_tick
        if tick is None:
            return 0.0
        total = 0.0
        for trade in broker.open_trades.values():
            units = int(trade.get("units", 0) or 0)
            entry_price = float(trade.get("price") or 0.0)
            if units == 0 or entry_price <= 0:
                continue
            if units > 0:
                pnl_pips = (tick.bid - entry_price) / PIP
            else:
                pnl_pips = (entry_price - tick.ask) / PIP
            total += pnl_pips * (abs(units) * PIP)
        return total

    def snapshot(self, broker: SimBroker) -> oanda_account.AccountSnapshot:
        unrealized = self._calc_unrealized(broker)
        self.unrealized_pl = unrealized
        nav = self.balance + unrealized
        tick = broker.last_tick
        margin_used = 0.0
        if tick is not None and self.margin_rate > 0.0:
            price = tick.mid
            for trade in broker.open_trades.values():
                units = abs(int(trade.get("units", 0) or 0))
                if units <= 0:
                    continue
                margin_used += units * price * self.margin_rate
        margin_available = max(0.0, nav - margin_used)
        free_ratio = margin_available / nav if nav > 0 else 0.0
        return oanda_account.AccountSnapshot(
            nav=nav,
            balance=self.balance,
            margin_available=margin_available,
            margin_used=margin_used,
            margin_rate=self.margin_rate,
            unrealized_pl=unrealized,
            free_margin_ratio=free_ratio,
            health_buffer=free_ratio,
        )

    def apply_realized(self, pnl_jpy: float) -> None:
        self.balance += float(pnl_jpy)


class SimPositionManager:
    def __init__(self, broker: SimBroker) -> None:
        self._broker = broker

    def get_open_positions(self) -> Dict[str, dict]:
        return self._broker.get_open_positions()

    def close(self) -> None:
        return None


def _patch_live_deps(broker: SimBroker, account: SimAccount) -> None:
    def _snapshot_stub(*args, **kwargs):
        return account.snapshot(broker)

    def _pos_summary_stub(instrument: str = "USD_JPY", timeout: float = 7.0) -> tuple[float, float]:
        long_units = 0.0
        short_units = 0.0
        for trade in broker.open_trades.values():
            units = float(trade.get("units") or 0.0)
            if units > 0:
                long_units += units
            elif units < 0:
                short_units += abs(units)
        return long_units, short_units

    async def _market_order_stub(
        instrument: str,
        units: int,
        sl_price: Optional[float],
        tp_price: Optional[float],
        pocket: str,
        *,
        client_order_id: Optional[str] = None,
        strategy_tag: Optional[str] = None,
        reduce_only: bool = False,
        entry_thesis: Optional[dict] = None,
        meta: Optional[dict] = None,
        confidence: Optional[int] = None,
        stage_index: Optional[int] = None,
        arbiter_final: bool = False,
    ) -> Optional[str]:
        tick = broker.last_tick
        if tick is None or units == 0:
            return None
        thesis = entry_thesis if isinstance(entry_thesis, dict) else {}
        direction = "long" if units > 0 else "short"
        entry_price = tick.ask if units > 0 else tick.bid
        sl_pips = abs(entry_price - sl_price) / PIP if sl_price else None
        tp_pips = abs(entry_price - tp_price) / PIP if tp_price else None
        if (sl_pips is None or sl_pips <= 0.0) and thesis:
            for key in ("sl_pips", "hard_stop_pips", "loss_guard_pips", "fast_cut_pips"):
                raw = thesis.get(key)
                try:
                    hint = float(raw) if raw is not None else None
                except Exception:
                    hint = None
                if hint is not None and hint > 0.0:
                    sl_pips = hint
                    break
        if (tp_pips is None or tp_pips <= 0.0) and thesis:
            raw = thesis.get("tp_pips")
            try:
                hint = float(raw) if raw is not None else None
            except Exception:
                hint = None
            if hint is not None and hint > 0.0:
                tp_pips = hint
        tag = (
            strategy_tag
            or thesis.get("strategy_tag")
            or thesis.get("strategy")
            or "scalp_replay"
        )
        trade = broker.open_trade(
            pocket=pocket,
            strategy_tag=str(tag),
            direction=direction,
            entry_price=entry_price,
            entry_time=tick.ts,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            timeout_sec=None,
            units=abs(units),
            source="scalp_replay",
            entry_thesis=thesis if thesis else None,
            meta=meta if isinstance(meta, dict) else None,
        )
        return str(trade.get("trade_id"))

    oanda_account.get_account_snapshot = _snapshot_stub
    oanda_account.get_position_summary = _pos_summary_stub
    order_manager.market_order = _market_order_stub
    try:
        import workers.common.dyn_size as dyn_size

        dyn_size.get_account_snapshot = _snapshot_stub
    except Exception:
        pass
    try:
        import execution.risk_guard as risk_guard

        risk_guard.get_position_summary = _pos_summary_stub
    except Exception:
        pass


def _patch_exit_module(module, broker: SimBroker) -> None:
    module.PositionManager = lambda: SimPositionManager(broker)

    async def _close_trade_stub(
        trade_id: str,
        units: int | None = None,
        client_order_id: str | None = None,
        allow_negative: bool = False,
        exit_reason: str | None = None,
        **_kwargs,
    ) -> bool:
        return broker.close_trade(str(trade_id), exit_reason or "exit_worker")

    module.close_trade = _close_trade_stub


def _summarize(trades: List[dict]) -> dict:
    total_pips = sum(t["pnl_pips"] for t in trades)
    total_jpy = sum(t.get("pnl_jpy", 0.0) for t in trades)
    wins = [t for t in trades if t["pnl_pips"] > 0]
    losses = [t for t in trades if t["pnl_pips"] < 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    gross_win = sum(t["pnl_pips"] for t in wins)
    gross_loss = abs(sum(t["pnl_pips"] for t in losses))
    gross_win_jpy = sum(t.get("pnl_jpy", 0.0) for t in wins)
    gross_loss_jpy = abs(sum(t.get("pnl_jpy", 0.0) for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    pf_jpy = gross_win_jpy / gross_loss_jpy if gross_loss_jpy > 0 else float("inf")
    return {
        "trades": len(trades),
        "total_pnl_pips": round(total_pips, 3),
        "total_pnl_jpy": round(total_jpy, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 4) if pf != float("inf") else float("inf"),
        "profit_factor_jpy": round(pf_jpy, 4) if pf_jpy != float("inf") else float("inf"),
    }


class FastScalpEntryEngine:
    def __init__(self, broker: SimBroker) -> None:
        if not _FAST_SCALP_AVAILABLE or FastScalpReplayer is None or fast_config is None:
            raise RuntimeError("fast_scalp replay modules are unavailable")
        self._broker = broker
        self._active_trade_id: Optional[str] = None
        self._replayer = FastScalpReplayer(
            units=2000,
            range_active=False,
            m1_rsi=None,
            buffer_seconds=60.0,
            latency_ms=150.0,
            slip_base_pips=0.0,
            slip_spread_coef=0.0,
            slip_atr_coef=0.0,
            slip_latency_coef=0.0,
            fill_mode="lko",
            price_mode="bidask",
            spread_mode="actual",
            no_sl=False,
            positive_exit_only=False,
            min_hold_sec=fast_config.MIN_HOLD_SEC,
            no_loss_close=fast_config.NO_LOSS_CLOSE,
            exit_min_loss_pips=fast_config.EXIT_MIN_LOSS_PIPS,
            exit_ignore_reasons=fast_config.EXIT_IGNORE_REASONS,
        )
        self._replayer._check_exit = lambda *args, **kwargs: None  # type: ignore[assignment]

    def on_closed(self, trade: dict) -> None:
        if trade.get("source") != "fast_scalp":
            return
        self._active_trade_id = None
        self._replayer.active_trade = None
        self._replayer.pending_entry = None

    def on_tick(self, tick: TickRow) -> None:
        if ReplayTick is None:
            return
        rtick = ReplayTick(
            ts=tick.ts,
            epoch=tick.epoch,
            bid=tick.bid,
            ask=tick.ask,
            mid=tick.mid,
            spread_pips=max(0.0, tick.ask - tick.bid) / PIP_VALUE,
        )
        self._replayer.on_tick(rtick)
        if self._active_trade_id is not None:
            return
        if self._replayer.active_trade is None:
            return
        tr = self._replayer.active_trade
        trade = self._broker.open_trade(
            pocket="scalp_fast",
            strategy_tag="fast_scalp",
            direction=tr.direction,
            entry_price=tr.entry_price,
            entry_time=tr.entry_time,
            tp_pips=tr.tp_pips,
            sl_pips=tr.sl_pips,
            timeout_sec=None,
            units=abs(tr.units),
            source="fast_scalp",
        )
        self._active_trade_id = trade["trade_id"]


class ScalpReplayEntryEngine:
    def __init__(
        self,
        broker: SimBroker,
        *,
        live_entry: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._broker = broker
        self._live_entry = bool(live_entry)
        self._loop = loop
        self._config = _SP_CFG
        self._bypass_common_guard = (self._config.MODE or "").strip().lower() in self._config.GUARD_BYPASS_MODES
        raw_allow = os.getenv("SCALP_REPLAY_ALLOWLIST", self._config.ALLOWLIST_RAW)
        allowlist = {s.strip().lower() for s in (raw_allow or "").split(",") if s.strip()}
        mode = (self._config.MODE or "").strip().lower()
        if mode and not allowlist:
            allowlist.add(mode)
        modes = _signal_map()
        if allowlist:
            filtered = {}
            for tag, fn in modes.items():
                tag_key = tag.lower()
                mode_key = _TAG_TO_MODE.get(tag, "").lower()
                if tag_key in allowlist or (mode_key and mode_key in allowlist):
                    filtered[tag] = fn
            modes = filtered
        self._modes = modes
        self._state: Dict[str, dict] = {}
        for mode in self._modes:
            self._state[mode] = {"last_entry": 0.0}
        self._last_eval_epoch: Optional[float] = None

    def on_tick(self, tick: TickRow) -> None:
        if self._last_eval_epoch is not None:
            if tick.epoch - self._last_eval_epoch < self._config.LOOP_INTERVAL_SEC:
                return
        self._last_eval_epoch = tick.epoch
        if self._live_entry:
            if not is_market_open(tick.ts):
                return
            if not can_trade(self._config.POCKET):
                return
        factors = factor_cache.all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_m5 = factors.get("M5") or {}
        fac_h1 = factors.get("H1") or {}
        fac_h4 = factors.get("H4") or {}
        range_ctx = detect_range_mode(fac_m1, fac_h4)
        air = evaluate_air(fac_m1, fac_h4, range_ctx=range_ctx, tag="scalp_replay")
        if self._live_entry and air.enabled and not air.allow_entry and not self._bypass_common_guard:
            return
        if self._live_entry and not self._bypass_common_guard:
            blocked, _, _, _ = spread_monitor.is_blocked()
            if blocked:
                return
            if self._config.MAX_OPEN_TRADES > 0 or self._config.MAX_OPEN_TRADES_GLOBAL > 0:
                positions = self._broker.get_open_positions().get(self._config.POCKET) or {}
                open_trades_all = positions.get("open_trades") or []
                if self._config.MAX_OPEN_TRADES_GLOBAL > 0 and len(open_trades_all) >= self._config.MAX_OPEN_TRADES_GLOBAL:
                    return
                open_trades = open_trades_all
                if self._config.OPEN_TRADES_SCOPE == "tag":
                    mode_name = (self._config.MODE or "").strip().lower()
                    tag_filter = None
                    for tag, mode in _TAG_TO_MODE.items():
                        if mode == mode_name:
                            tag_filter = tag.lower()
                            break
                    if tag_filter:
                        open_trades = [
                            tr for tr in open_trades_all if str(tr.get("strategy_tag") or "").lower() == tag_filter
                        ]
                if self._config.MAX_OPEN_TRADES > 0 and len(open_trades) >= self._config.MAX_OPEN_TRADES:
                    return

        for mode, fn in self._modes.items():
            if self._config.COOLDOWN_SEC > 0.0:
                last_entry = float(self._state[mode]["last_entry"])
                if tick.epoch - last_entry < self._config.COOLDOWN_SEC:
                    continue
            signal = fn(fac_m1, fac_m5, fac_h1, range_ctx, tick.ts)
            if not signal:
                continue
            signal = adjust_signal(signal, air)
            if not signal:
                continue
            conf = int(signal.get("confidence", 0) or 0)
            if self._config.MIN_ENTRY_CONF > 0 and conf < self._config.MIN_ENTRY_CONF:
                continue
            action = signal.get("action")
            if action not in {"OPEN_LONG", "OPEN_SHORT"}:
                continue
            direction = "long" if action == "OPEN_LONG" else "short"
            sl_pips = float(signal.get("sl_pips") or 0.0)
            tp_pips = float(signal.get("tp_pips") or 0.0)
            if sl_pips <= 0:
                continue
            if self._live_entry:
                if self._loop is None:
                    continue
                conf = int(signal.get("confidence") or 0)
                units = 10000
                if direction == "short":
                    units = -10000
                sl_price = entry_price - sl_pips * PIP if direction == "long" else entry_price + sl_pips * PIP
                tp_price = (
                    entry_price + tp_pips * PIP
                    if direction == "long" and tp_pips > 0
                    else entry_price - tp_pips * PIP
                    if direction == "short" and tp_pips > 0
                    else None
                )
                order_id = self._loop.run_until_complete(
                    order_manager.market_order(
                        instrument="USD_JPY",
                        units=units,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        pocket=self._config.POCKET,
                        strategy_tag=mode,
                        entry_thesis={
                            "entry_probability": min(1.0, max(0.0, conf / 100.0)),
                            "entry_units_intent": abs(units),
                            **signal,
                        },
                        confidence=conf,
                        meta={"source": "replay_exit_workers"},
                    )
                )
                if order_id:
                    self._state[mode]["last_entry"] = tick.epoch
                continue
            entry_price = tick.ask if direction == "long" else tick.bid
            if direction == "long":
                sl_price = entry_price - sl_pips * PIP
                tp_price = entry_price + tp_pips * PIP if tp_pips > 0 else entry_price + 2.0 * PIP
            else:
                sl_price = entry_price + sl_pips * PIP
                tp_price = entry_price - tp_pips * PIP if tp_pips > 0 else entry_price - 2.0 * PIP
            sl_price, tp_price = clamp_sl_tp(entry_price, sl_price, tp_price, direction == "long")
            self._broker.open_trade(
                pocket=self._config.POCKET,
                strategy_tag=mode,
                direction=direction,
                entry_price=entry_price,
                entry_time=tick.ts,
                tp_pips=tp_pips,
                sl_pips=sl_pips,
                timeout_sec=float(signal.get("timeout_sec") or 0.0) or None,
                units=10000,
                source="scalp_replay",
            )
            self._state[mode]["last_entry"] = tick.epoch


class StrategyEntryEngine:
    def __init__(
        self,
        broker: SimBroker,
        *,
        equity: float,
        margin_available: float,
        margin_rate: float,
        macro_enabled: bool = True,
    ) -> None:
        self._broker = broker
        self._equity = equity
        self._margin_available = margin_available
        self._margin_rate = margin_rate
        self._macro_enabled = macro_enabled

    def on_candle_close(self, tick: TickRow) -> None:
        factors = factor_cache.all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}
        if not fac_m1 or not fac_h4:
            return
        macro_regime = classify(fac_h4, "H4")
        micro_regime = classify(fac_m1, "M1")
        focus_tag, weight_macro = decide_focus(macro_regime, micro_regime, event_soon=False)
        payload = {
            "ts": tick.ts.isoformat(timespec="seconds"),
            "reg_macro": macro_regime,
            "reg_micro": micro_regime,
            "factors_m1": {k: v for k, v in fac_m1.items() if k != "candles"},
            "factors_h4": {k: v for k, v in fac_h4.items() if k != "candles"},
            "perf": {},
            "event_soon": False,
        }
        gpt_like = heuristic_decision(payload)
        ranked = list(gpt_like.get("ranked_strategies", []))
        if not ranked:
            return
        weight_macro = gpt_like.get("weight_macro", weight_macro)
        weight_scalp_raw = gpt_like.get("weight_scalp")
        try:
            weight_scalp = max(0.0, min(1.0, float(weight_scalp_raw)))
        except (TypeError, ValueError):
            weight_scalp = None
        if not self._macro_enabled:
            weight_macro = 0.0
        range_ctx = detect_range_mode(fac_m1, fac_h4)
        if range_ctx.active:
            weight_macro = min(weight_macro, 0.15)
            if weight_scalp is not None:
                weight_scalp = min(weight_scalp, 0.2)

        signals: List[dict] = []
        for name in ranked:
            cls = main_mod.STRATEGIES.get(name)
            if not cls:
                continue
            if range_ctx.active and cls.name not in main_mod.ALLOWED_RANGE_STRATEGIES:
                continue
            raw = cls.check(fac_m1)
            if not raw:
                continue
            if name == "BB_RSI":
                if not range_ctx.active:
                    try:
                        min_score = float(os.getenv("BBRSI_RANGE_ONLY_SCORE", "0.35"))
                    except Exception:
                        min_score = 0.0
                    score_val = float(getattr(range_ctx, "score", 0.0) or 0.0)
                    if score_val < min_score:
                        continue
                try:
                    min_conf = int(float(os.getenv("BBRSI_MIN_ENTRY_CONF", "0")))
                except Exception:
                    min_conf = 0
                conf_val = int(raw.get("confidence", 0) or 0)
                if min_conf > 0 and conf_val < min_conf:
                    continue
            if raw.get("action") not in {"OPEN_LONG", "OPEN_SHORT"}:
                continue
            payload_sig = dict(raw)
            payload_sig["strategy"] = name
            payload_sig["pocket"] = getattr(cls, "pocket", "macro")
            signals.append(payload_sig)

        if not signals:
            return

        avg_sl = sum(float(sig.get("sl_pips") or 0.0) for sig in signals) / len(signals)
        price = float(fac_m1.get("close") or 0.0)
        lot_total = allowed_lot(
            self._equity,
            sl_pips=max(1.0, avg_sl),
            margin_available=self._margin_available,
            price=price if price > 0 else None,
            margin_rate=self._margin_rate,
        )
        scalp_share = DEFAULT_SCALP_SHARE if weight_scalp is None else 0.0
        pocket_lots = alloc(lot_total, weight_macro, weight_scalp=weight_scalp, scalp_share=scalp_share)

        for sig in signals:
            name = sig.get("strategy")
            if name not in {"TrendMA", "BB_RSI"}:
                continue
            pocket = sig.get("pocket") or "macro"
            if pocket == "macro" and not self._macro_enabled:
                continue
            if pocket_lots.get(pocket, 0.0) <= 0:
                continue
            units = int(round(max(0.01, pocket_lots[pocket]) * 100000))
            action = sig.get("action")
            direction = "long" if action == "OPEN_LONG" else "short"
            entry_price = tick.ask if direction == "long" else tick.bid
            tp_pips = float(sig.get("tp_pips") or 0.0)
            sl_pips = float(sig.get("sl_pips") or 0.0)
            timeout_sec = float(sig.get("timeout_sec") or 0.0) or None
            self._broker.open_trade(
                pocket=pocket,
                strategy_tag=str(name),
                direction=direction,
                entry_price=entry_price,
                entry_time=tick.ts,
                tp_pips=tp_pips,
                sl_pips=sl_pips,
                timeout_sec=timeout_sec,
                units=abs(units),
                source="strategy",
            )


class ExitRunner:
    def __init__(self, name: str, module, worker, broker: SimBroker) -> None:
        self.name = name
        self.module = module
        self.worker = worker
        self.broker = broker
        self.last_eval = 0.0
        self.interval = getattr(worker, "loop_interval", 1.0)

    async def step(self, now: datetime) -> None:
        if now.timestamp() - self.last_eval < self.interval:
            return
        self.last_eval = now.timestamp()
        positions = self.broker.get_open_positions()
        pocket = getattr(self.module, "POCKET", None)
        pocket = pocket or "micro"
        pocket_info = positions.get(pocket) or {}
        trades = pocket_info.get("open_trades") or []
        tags = getattr(self.module, "ALLOWED_TAGS", set())
        if hasattr(self.module, "_filter_trades"):
            trades = self.module._filter_trades(trades, tags)  # type: ignore[attr-defined]
        if not trades:
            return
        if self.name == "fast_scalp":
            mid, rsi, range_active = self.worker._context()
            if mid is None:
                return
            for tr in trades:
                await self.worker._review_trade(tr, now, mid, rsi, range_active)
            return
        if self.name in {"scalp_level_reject", "scalp_false_break_fade"}:
            mid, range_active = self.worker._context()
            if mid is None:
                return
            for tr in trades:
                await self.worker._review_trade(tr, now, mid, range_active)
            return
        if self.name == "micro_bbrsi":
            mid, range_active = self.worker._context()
            if mid is None:
                return
            for tr in trades:
                await self.worker._review_trade(tr, now, mid, range_active)
            return
        if self.name == "macro_trendma":
            bid, ask, mid, rsi, range_active, fac_h1 = self.worker._context()
            if mid is None and bid is None and ask is None:
                return
            for tr in trades:
                await self.worker._review_trade(tr, now, bid, ask, mid, rsi, range_active, fac_h1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay with exit_worker logic.")
    parser.add_argument("--ticks", type=Path, required=True, help="Tick JSONL (bid/ask).")
    parser.add_argument("--instrument", default="USD_JPY")
    parser.add_argument("--start", default="", help="ISO start (UTC).")
    parser.add_argument("--end", default="", help="ISO end (UTC).")
    parser.add_argument("--out", type=Path, default=Path("tmp/replay_exit_workers.json"))
    parser.add_argument("--quiet", action="store_true", help="Disable logging output (CRITICAL only).")
    parser.add_argument("--no-stdout", action="store_true", help="Do not print replay JSON to stdout (still writes --out).")
    parser.add_argument("--equity", type=float, default=12000.0)
    parser.add_argument("--margin-available", type=float, default=9000.0)
    parser.add_argument("--margin-rate", type=float, default=0.04)
    parser.add_argument("--no-exit-worker", action="store_true", help="Disable exit_worker processing.")
    parser.add_argument("--no-hard-sl-tp", action="store_true", help="Disable hard SL/TP triggers.")
    parser.add_argument("--no-hard-sl", action="store_true", help="Disable hard SL (keep TP if enabled).")
    parser.add_argument("--no-hard-tp", action="store_true", help="Disable hard TP (keep SL if enabled).")
    parser.add_argument(
        "--exclude-end-of-replay",
        action="store_true",
        help="Exclude end_of_replay forced closes from summary.",
    )
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="Replay fast_scalp only (skip other entry/exit workers).",
    )
    parser.add_argument(
        "--sp-only",
        action="store_true",
        help="Replay scalp replay workers only (skip fast_scalp entry/exit).",
    )
    parser.add_argument(
        "--sp-live-entry",
        action="store_true",
        help="Use live scalp replay entry path (market_order + risk/size guards).",
    )
    parser.add_argument("--disable-macro", action="store_true", help="Disable macro entries/exit.")
    parser.add_argument("--no-main-strategies", action="store_true", help="Disable main strategy entries.")
    parser.add_argument("--slip-base-pips", type=float, default=0.0)
    parser.add_argument("--slip-spread-coef", type=float, default=0.0)
    parser.add_argument("--slip-atr-coef", type=float, default=0.0)
    parser.add_argument("--slip-latency-coef", type=float, default=0.0)
    parser.add_argument("--latency-ms", type=float, default=150.0)
    parser.add_argument(
        "--fill-mode",
        choices=("lko", "next_tick"),
        default="lko",
        help="Fill policy: lko=last known quote at/before latency target, next_tick=first tick after latency.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast_only and not _FAST_SCALP_AVAILABLE:
        raise SystemExit("--fast-only is unavailable because fast_scalp modules were removed")
    if args.quiet:
        logging.disable(logging.CRITICAL)
    elif args.fast_only:
        args.no_main_strategies = True
        args.disable_macro = True
        logging.disable(logging.CRITICAL)
    else:
        logging.disable(logging.INFO)
    start = parse_iso8601(args.start) if args.start else None
    end = parse_iso8601(args.end) if args.end else None

    sim_clock = SimClock()
    _reset_replay_state(sim_clock)

    broker = SimBroker(
        slip_base_pips=args.slip_base_pips,
        slip_spread_coef=args.slip_spread_coef,
        slip_atr_coef=args.slip_atr_coef,
        slip_latency_coef=args.slip_latency_coef,
        latency_ms=args.latency_ms,
        fill_mode=args.fill_mode,
    )
    hard_sl = True
    hard_tp = True
    if args.no_hard_sl_tp:
        hard_sl = False
        hard_tp = False
    if args.no_hard_sl:
        hard_sl = False
    if args.no_hard_tp:
        hard_tp = False
    broker.hard_sl = hard_sl
    broker.hard_tp = hard_tp
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    fast_entry: Optional[FastScalpEntryEngine] = None
    if not args.sp_only and _FAST_SCALP_AVAILABLE:
        fast_entry = FastScalpEntryEngine(broker)
    sim_account: Optional[SimAccount] = None
    if args.sp_live_entry and not args.fast_only:
        sim_account = SimAccount(balance=args.equity, margin_rate=args.margin_rate)
        _patch_live_deps(broker, sim_account)
        def _on_close(trade: dict) -> None:
            if fast_entry is not None:
                fast_entry.on_closed(trade)
            pnl_jpy = trade.get("pnl_jpy")
            if pnl_jpy is not None and sim_account is not None:
                sim_account.apply_realized(float(pnl_jpy))
        broker.on_close = _on_close
    else:
        if fast_entry is not None:
            broker.on_close = fast_entry.on_closed
    sp_entry: Optional[ScalpReplayEntryEngine] = None
    if not args.fast_only:
        sp_entry = ScalpReplayEntryEngine(
            broker,
            live_entry=args.sp_live_entry,
            loop=loop if args.sp_live_entry else None,
        )
    strat_entry = StrategyEntryEngine(
        broker,
        equity=args.equity,
        margin_available=args.margin_available,
        margin_rate=args.margin_rate,
        macro_enabled=not args.disable_macro,
    )

    if fast_exit is not None:
        _patch_exit_module(fast_exit, broker)
    if not args.fast_only:
        _patch_exit_module(sp_level_reject_exit, broker)
        _patch_exit_module(sp_false_break_exit, broker)
        _patch_exit_module(bbrsi_exit, broker)
        if trendma_exit is not None:
            _patch_exit_module(trendma_exit, broker)

    fast_runner = None
    if not args.sp_only and fast_exit is not None:
        fast_runner = ExitRunner("fast_scalp", fast_exit, fast_exit.FastScalpExitWorker(), broker)
    sp_level_reject_runner = None
    sp_false_break_runner = None
    bbrsi_runner = None
    trend_runner = None
    if not args.fast_only:
        sp_level_reject_runner = ExitRunner(
            "scalp_level_reject",
            sp_level_reject_exit,
            sp_level_reject_exit.RangeFaderExitWorker(),
            broker,
        )
        sp_false_break_runner = ExitRunner(
            "scalp_false_break_fade",
            sp_false_break_exit,
            sp_false_break_exit.RangeFaderExitWorker(),
            broker,
        )
        bbrsi_runner = ExitRunner("micro_bbrsi", bbrsi_exit, bbrsi_exit.MicroBBRsiExitWorker(), broker)
        if not args.disable_macro and trendma_exit is not None:
            trend_runner = ExitRunner("macro_trendma", trendma_exit, trendma_exit.TrendMAExitWorker(), broker)

    # Pre-feed H4 candles (system_backtest-style) to align macro context.
    prefeed_h4: List[dict] = []
    h4_prefeed_builder = CandleBuilder(240)
    for tick in _iter_ticks(args.ticks, args.instrument, start, end):
        closed = h4_prefeed_builder.update(tick.ts, tick.mid, 1)
        if closed:
            prefeed_h4.append(closed)
    for candle in prefeed_h4:
        loop.run_until_complete(factor_cache.on_candle("H4", candle))

    m1_builder = CandleBuilder(1)
    m5_builder = CandleBuilder(5)
    h1_builder = CandleBuilder(60)
    h4_builder = CandleBuilder(240)

    for tick in _iter_ticks(args.ticks, args.instrument, start, end):
        broker.set_last_tick(tick)
        sim_clock.now = tick.epoch

        class _Tick:
            def __init__(self, bid: float, ask: float, ts: datetime) -> None:
                self.bid = bid
                self.ask = ask
                self.time = ts

        t = _Tick(tick.bid, tick.ask, tick.ts)
        tick_window.record(t)
        spread_monitor.update_from_tick(t)

        closed_m1 = m1_builder.update(tick.ts, tick.mid, 1)
        if closed_m1:
            loop.run_until_complete(factor_cache.on_candle("M1", closed_m1))
            closed_m5 = m5_builder.update(closed_m1["time"], closed_m1["close"], 1)
            if closed_m5:
                loop.run_until_complete(factor_cache.on_candle("M5", closed_m5))
            closed_h1 = h1_builder.update(closed_m1["time"], closed_m1["close"], 1)
            if closed_h1:
                loop.run_until_complete(factor_cache.on_candle("H1", closed_h1))
            # H4 is prefed; skip live updates to keep macro context stable.
            if not args.no_main_strategies:
                strat_entry.on_candle_close(tick)

        broker.check_sl_tp(tick)
        if fast_entry is not None:
            fast_entry.on_tick(tick)
        if sp_entry is not None:
            sp_entry.on_tick(tick)
        broker.check_sl_tp(tick)

        now_dt = tick.ts
        if not args.no_exit_worker:
            if fast_runner is not None:
                loop.run_until_complete(fast_runner.step(now_dt))
            if sp_level_reject_runner is not None:
                loop.run_until_complete(sp_level_reject_runner.step(now_dt))
            if sp_false_break_runner is not None:
                loop.run_until_complete(sp_false_break_runner.step(now_dt))
            if bbrsi_runner is not None:
                loop.run_until_complete(bbrsi_runner.step(now_dt))
            if trend_runner is not None:
                loop.run_until_complete(trend_runner.step(now_dt))

    # close remaining at end
    for trade_id in list(broker.open_trades.keys()):
        broker.close_trade(trade_id, "end_of_replay")

    filtered_trades = (
        [tr for tr in broker.closed_trades if tr.get("reason") != "end_of_replay"]
        if args.exclude_end_of_replay
        else list(broker.closed_trades)
    )

    summary_by_worker: Dict[str, dict] = {}
    trades_by_worker: Dict[str, List[dict]] = {}
    for tr in filtered_trades:
        key = tr.get("strategy_tag") or "unknown"
        trades_by_worker.setdefault(key, []).append(tr)
    for key, items in trades_by_worker.items():
        summary_by_worker[key] = _summarize(items)
    summary_by_pocket: Dict[str, dict] = {}
    trades_by_pocket: Dict[str, List[dict]] = {}
    for tr in filtered_trades:
        pocket = tr.get("pocket") or "unknown"
        trades_by_pocket.setdefault(pocket, []).append(tr)
    for pocket, items in trades_by_pocket.items():
        summary_by_pocket[pocket] = _summarize(items)
    summary_overall = _summarize(filtered_trades)

    out_payload = {
        "meta": {
            "ticks": str(args.ticks),
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
            "exit_worker_enabled": not args.no_exit_worker,
            "hard_sl_tp": bool(hard_sl and hard_tp),
            "hard_sl": bool(hard_sl),
            "hard_tp": bool(hard_tp),
            "macro_enabled": not args.disable_macro,
            "main_strategies_enabled": not args.no_main_strategies,
            "slip_base_pips": args.slip_base_pips,
            "slip_spread_coef": args.slip_spread_coef,
            "slip_atr_coef": args.slip_atr_coef,
            "slip_latency_coef": args.slip_latency_coef,
            "latency_ms": args.latency_ms,
            "scalp_entry_min_entry_conf": _SP_CFG.MIN_ENTRY_CONF,
            "scalp_entry_allowlist": os.getenv("SCALP_REPLAY_ALLOWLIST", _SP_CFG.ALLOWLIST_RAW),
            "scalp_entry_live_entry": bool(args.sp_live_entry),
            "bbrsi_min_entry_conf": os.getenv("BBRSI_MIN_ENTRY_CONF"),
            "exclude_end_of_replay": args.exclude_end_of_replay,
        },
        "summary": summary_by_worker,
        "summary_by_pocket": summary_by_pocket,
        "summary_overall": summary_overall,
        "trades": broker.closed_trades,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_stdout:
        print(json.dumps(out_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
