#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tick-level exit replay for QuantRabbit.

Replays recorded tick JSONL files and evaluates exit-worker logic against
historical trades to estimate counterfactual exits with live-like signals.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import os
import re
import sqlite3
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.technique_engine import evaluate_exit_techniques  # noqa: E402
from execution.section_axis import evaluate_section_exit  # noqa: E402
from indicators import factor_cache as factor_cache_module  # noqa: E402
from market_data.candle_fetcher import CandleAggregator  # noqa: E402
from market_data import tick_window as live_tick_window  # noqa: E402
from workers.common.exit_scaling import momentum_scale, scale_value  # noqa: E402

try:
    from execution.position_manager import _normalize_strategy_tag  # type: ignore
except Exception:  # pragma: no cover
    _normalize_strategy_tag = None  # type: ignore[assignment]


PIP = 0.01
INSTRUMENT = "USD_JPY"


def _parse_iso(ts: str) -> datetime:
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    if "." in ts and ("+" in ts[ts.index(".") :] or "-" in ts[ts.index(".") :]):
        head, frac_and_tz = ts.split(".", 1)
        if "+" in frac_and_tz:
            frac, tz_tail = frac_and_tz.split("+", 1)
            ts = f"{head}.{frac[:6].ljust(6, '0')}+{tz_tail}"
        elif "-" in frac_and_tz:
            frac, tz_tail = frac_and_tz.split("-", 1)
            ts = f"{head}.{frac[:6].ljust(6, '0')}-{tz_tail}"
    elif "." in ts:
        head, frac = ts.split(".", 1)
        ts = f"{head}.{frac[:6].ljust(6, '0')}+00:00"
    elif "+" not in ts and "-" not in ts[-6:]:
        ts = ts + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _parse_dt(value: Optional[str], *, end: bool = False) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        base = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
        if end:
            return base + timedelta(days=1) - timedelta(microseconds=1)
        return base
    try:
        return _parse_iso(text)
    except Exception:
        return None


def _normalize_tag(tag: Optional[str]) -> str:
    if not tag:
        return ""
    raw = str(tag).strip()
    if raw.startswith("main."):
        raw = raw.split(".", 1)[1]
    if _normalize_strategy_tag is None:
        return raw
    try:
        norm = _normalize_strategy_tag(raw)
        return norm or raw
    except Exception:
        return raw


def _client_id(trade: Dict[str, Any]) -> Optional[str]:
    client_id = trade.get("client_order_id") or trade.get("client_id")
    if not client_id and isinstance(trade.get("clientExtensions"), dict):
        client_id = trade["clientExtensions"].get("id")
    return str(client_id) if client_id else None


@dataclass
class TradeSim:
    trade_id: str
    pocket: str
    strategy_tag: str
    entry_time: datetime
    entry_price: float
    units: int
    entry_thesis: Dict[str, Any]
    client_order_id: Optional[str]
    actual_close_time: Optional[datetime]
    actual_exit_price: Optional[float]
    actual_pnl_pips: Optional[float]
    sim_exit_time: Optional[datetime] = None
    sim_exit_price: Optional[float] = None
    sim_pnl_pips: Optional[float] = None
    sim_exit_reason: Optional[str] = None
    adapter_name: Optional[str] = None
    skip_reason: Optional[str] = None

    def is_closed(self) -> bool:
        return self.sim_exit_time is not None


@dataclass
class ReplayState:
    now: Optional[datetime] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    trades: Dict[str, TradeSim] = field(default_factory=dict)


REPLAY_STATE = ReplayState()


async def _stub_close_trade(
    trade_id: str,
    units: Optional[int] = None,
    *,
    client_order_id: Optional[str] = None,
    allow_negative: bool = False,
    exit_reason: Optional[str] = None,
) -> bool:
    sim = REPLAY_STATE.trades.get(str(trade_id))
    if sim is None or sim.is_closed():
        return True
    if REPLAY_STATE.now is None or REPLAY_STATE.bid is None or REPLAY_STATE.ask is None:
        return False
    if sim.units > 0:
        exit_price = REPLAY_STATE.bid
    else:
        exit_price = REPLAY_STATE.ask
    sim.sim_exit_time = REPLAY_STATE.now
    sim.sim_exit_price = exit_price
    sim.sim_exit_reason = exit_reason
    if sim.entry_price > 0:
        if sim.units > 0:
            sim.sim_pnl_pips = (exit_price - sim.entry_price) / PIP
        else:
            sim.sim_pnl_pips = (sim.entry_price - exit_price) / PIP
    return True


class ReplayTickWindow:
    def __init__(self, max_seconds: float = 600.0, max_ticks: int = 6000) -> None:
        self._ticks: deque[Dict[str, float]] = deque(maxlen=max_ticks)
        self._max_seconds = max(1.0, max_seconds)

    def record(self, tick: Any) -> None:
        try:
            bid = float(tick.bid)
            ask = float(tick.ask)
            ts = float(tick.time.timestamp())
        except Exception:
            return
        mid = round((bid + ask) / 2.0, 5)
        self._ticks.append({"epoch": ts, "bid": bid, "ask": ask, "mid": mid})

    def recent_ticks(self, seconds: float = 60.0, *, limit: Optional[int] = None) -> List[Dict[str, float]]:
        if not self._ticks:
            return []
        cutoff = self._ticks[-1]["epoch"] - max(0.0, seconds)
        rows: List[Dict[str, float]] = []
        for row in reversed(self._ticks):
            if row["epoch"] < cutoff:
                break
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
        return rows


def _patch_tick_window() -> ReplayTickWindow:
    replay_window = ReplayTickWindow()
    live_tick_window.record = replay_window.record  # type: ignore[assignment]
    live_tick_window.recent_ticks = replay_window.recent_ticks  # type: ignore[assignment]
    return replay_window


def _patch_factor_cache() -> None:
    factor_cache_module.refresh_cache_from_disk = lambda: False  # type: ignore[assignment]
    factor_cache_module._persist_cache = lambda: None  # type: ignore[attr-defined]


def _patch_close_trade(modules: Iterable[Any]) -> None:
    for mod in modules:
        if mod is None:
            continue
        if hasattr(mod, "close_trade"):
            setattr(mod, "close_trade", _stub_close_trade)


@dataclass
class ReplayTick:
    time: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


def _parse_tick_line(line: str) -> Optional[ReplayTick]:
    try:
        payload = json.loads(line)
    except Exception:
        return None
    ts_raw = payload.get("ts") or payload.get("time") or payload.get("timestamp")
    if not ts_raw:
        return None
    try:
        ts = _parse_iso(str(ts_raw))
    except Exception:
        return None
    try:
        bid = float(payload.get("bid", 0.0))
        ask = float(payload.get("ask", 0.0))
    except Exception:
        return None
    if bid <= 0 or ask <= 0:
        return None
    return ReplayTick(time=ts, bid=bid, ask=ask)


def _list_tick_files(ticks_dir: Path, instrument: str) -> List[Tuple[datetime, Path]]:
    pattern = re.compile(rf"{re.escape(instrument)}_ticks_(\d{{8}})\.jsonl$")
    files: List[Tuple[datetime, Path]] = []
    for path in ticks_dir.glob(f"{instrument}_ticks_*.jsonl"):
        match = pattern.search(path.name)
        if not match:
            continue
        day = match.group(1)
        try:
            dt = datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        files.append((dt, path))
    files.sort(key=lambda x: x[0])
    return files


def _infer_range_from_ticks(
    files: List[Tuple[datetime, Path]],
    days: int,
) -> Tuple[datetime, datetime]:
    if not files:
        now = datetime.now(timezone.utc)
        return now - timedelta(days=days), now
    days = max(1, days)
    picked = files[-days:]
    start = picked[0][0]
    end = picked[-1][0] + timedelta(days=1) - timedelta(microseconds=1)
    return start, end


def _load_trades(
    db_path: Path,
    start: datetime,
    end: datetime,
    strategies: Optional[set[str]],
    pockets: Optional[set[str]],
) -> List[TradeSim]:
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT
          ticket_id,
          pocket,
          units,
          entry_price,
          close_price,
          fill_price,
          pl_pips,
          entry_time,
          open_time,
          close_time,
          strategy_tag,
          client_order_id,
          entry_thesis
        FROM trades
        WHERE close_time IS NOT NULL
        """
    ).fetchall()
    con.close()
    trades: List[TradeSim] = []
    for row in rows:
        ticket_id = row["ticket_id"]
        if not ticket_id:
            continue
        entry_time = _parse_dt(row["entry_time"]) or _parse_dt(row["open_time"])
        if entry_time is None:
            continue
        if entry_time < start or entry_time > end:
            continue
        pocket = str(row["pocket"] or "").strip().lower() or "unknown"
        if pockets and pocket not in pockets:
            continue
        units = int(row["units"] or 0)
        if units == 0:
            continue
        entry_price = float(row["entry_price"] or 0.0)
        if entry_price <= 0:
            continue
        entry_thesis: Dict[str, Any] = {}
        raw_thesis = row["entry_thesis"]
        if isinstance(raw_thesis, str) and raw_thesis.strip():
            try:
                parsed = json.loads(raw_thesis)
                if isinstance(parsed, dict):
                    entry_thesis = parsed
            except Exception:
                entry_thesis = {}
        strategy_tag = row["strategy_tag"]
        if not strategy_tag:
            strategy_tag = (
                entry_thesis.get("strategy_tag")
                or entry_thesis.get("strategy_tag_raw")
                or entry_thesis.get("strategy")
                or entry_thesis.get("tag")
            )
        strategy_tag = _normalize_tag(strategy_tag)
        if not strategy_tag:
            continue
        if strategies and strategy_tag not in strategies:
            continue
        close_time = _parse_dt(row["close_time"])
        close_price = row["close_price"]
        fill_price = row["fill_price"]
        actual_exit_price = None
        if close_price:
            try:
                actual_exit_price = float(close_price)
            except Exception:
                actual_exit_price = None
        if actual_exit_price is None and fill_price:
            try:
                actual_exit_price = float(fill_price)
            except Exception:
                actual_exit_price = None
        pl_pips_raw = row["pl_pips"]
        actual_pips = None
        if pl_pips_raw is not None:
            try:
                actual_pips = float(pl_pips_raw)
            except Exception:
                actual_pips = None
        if actual_pips is None and actual_exit_price:
            if units > 0:
                actual_pips = (actual_exit_price - entry_price) / PIP
            else:
                actual_pips = (entry_price - actual_exit_price) / PIP
        trades.append(
            TradeSim(
                trade_id=str(ticket_id),
                pocket=pocket,
                strategy_tag=strategy_tag,
                entry_time=entry_time,
                entry_price=entry_price,
                units=units,
                entry_thesis=entry_thesis,
                client_order_id=row["client_order_id"],
                actual_close_time=close_time,
                actual_exit_price=actual_exit_price,
                actual_pnl_pips=actual_pips,
            )
        )
    trades.sort(key=lambda t: t.entry_time)
    return trades


class BaseAdapter:
    name: str
    tags: set[str]
    loop_interval: float
    last_eval_ts: Optional[float]

    def matches(self, tag: str) -> bool:
        return tag in self.tags

    async def step(self, now: datetime, trades: List[TradeSim]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class WorkerAdapter(BaseAdapter):
    def __init__(self, name: str, worker: Any, tags: Iterable[str]) -> None:
        self.name = name
        self.worker = worker
        self.tags = {_normalize_tag(t) for t in tags if t}
        self.loop_interval = float(getattr(worker, "loop_interval", 1.0) or 1.0)
        self.last_eval_ts = None
        self._mode = "review" if hasattr(worker, "_review_trade") else "evaluate" if hasattr(worker, "_evaluate") else ""
        self._close_sig = inspect.signature(getattr(worker, "_close", lambda: None))

    def _due(self, now: datetime) -> bool:
        if self.loop_interval <= 0:
            return True
        now_ts = now.timestamp()
        if self.last_eval_ts is None:
            self.last_eval_ts = now_ts
            return True
        if now_ts - self.last_eval_ts >= self.loop_interval:
            self.last_eval_ts = now_ts
            return True
        return False

    async def step(self, now: datetime, trades: List[TradeSim]) -> None:
        if not trades or not self._due(now):
            return
        ctx = None
        if hasattr(self.worker, "_context"):
            try:
                ctx = self.worker._context()
            except Exception:
                return
        if self._mode == "review":
            if ctx is None or not isinstance(ctx, tuple):
                return
            for sim in trades:
                if sim.is_closed():
                    continue
                trade = _trade_to_open_entry(sim)
                try:
                    await self.worker._review_trade(trade, now, *ctx)
                except Exception:
                    continue
        elif self._mode == "evaluate":
            if ctx is None:
                return
            mid = getattr(ctx, "mid", None)
            if mid is None:
                return
            for sim in trades:
                if sim.is_closed():
                    continue
                trade = _trade_to_open_entry(sim)
                try:
                    reason = self.worker._evaluate(trade, ctx, now)
                except Exception:
                    continue
                if not reason:
                    continue
                await self._close_for_evaluate(sim, trade, ctx, now, reason)

    async def _close_for_evaluate(self, sim: TradeSim, trade: Dict[str, Any], ctx: Any, now: datetime, reason: str) -> None:
        mid = getattr(ctx, "mid", None)
        if mid is None:
            return
        side = "long" if sim.units > 0 else "short"
        pnl = (mid - sim.entry_price) * 100.0 if side == "long" else (sim.entry_price - mid) * 100.0
        client_id = _client_id(trade)
        range_mode = bool(getattr(ctx, "range_active", False))
        kwargs: Dict[str, Any] = {}
        for name in self._close_sig.parameters:
            if name == "trade_id":
                kwargs[name] = sim.trade_id
            elif name == "units":
                kwargs[name] = -sim.units
            elif name == "reason":
                kwargs[name] = reason
            elif name == "pnl":
                kwargs[name] = pnl
            elif name == "side":
                kwargs[name] = side
            elif name == "range_mode":
                kwargs[name] = range_mode
            elif name == "client_id":
                kwargs[name] = client_id
            elif name == "client_order_id":
                kwargs[name] = client_id
            elif name == "allow_negative":
                kwargs[name] = pnl <= 0
            elif name == "touch_count":
                kwargs[name] = None
        try:
            await self.worker._close(**kwargs)
        except Exception:
            return


def _trade_to_open_entry(sim: TradeSim) -> Dict[str, Any]:
    trade = {
        "trade_id": sim.trade_id,
        "units": sim.units,
        "price": sim.entry_price,
        "client_order_id": sim.client_order_id,
        "client_id": sim.client_order_id,
        "open_time": sim.entry_time.isoformat(),
        "strategy_tag": sim.strategy_tag,
        "entry_thesis": sim.entry_thesis,
        "side": "long" if sim.units > 0 else "short",
    }
    trade["clientExtensions"] = {"id": sim.client_order_id} if sim.client_order_id else None
    return trade


@dataclass
class _M1State:
    peak: float
    lock_floor: Optional[float] = None
    hard_stop: Optional[float] = None
    tp_hint: Optional[float] = None

    def update(self, pnl: float, lock_buffer: float) -> None:
        if pnl > self.peak:
            self.peak = pnl
        if pnl > 0:
            floor = max(0.0, pnl - lock_buffer)
            self.lock_floor = floor if self.lock_floor is None else max(self.lock_floor, floor)


class M1ScalperExitAdapter(BaseAdapter):
    def __init__(self) -> None:
        from workers.scalp_m1scalper import exit_worker as mod

        self.name = "M1ScalperExit"
        self.tags = {_normalize_tag(t) for t in getattr(mod, "ALLOWED_TAGS", {"M1Scalper"})}
        self.loop_interval = 0.8
        self.last_eval_ts = None
        self._states: Dict[str, _M1State] = {}

        self.profit_take = 2.2
        self.trail_start = 2.8
        self.trail_backoff = 0.9
        self.lock_buffer = 0.5
        self.min_hold_sec = 10.0
        self.max_hold_sec = max(
            self.min_hold_sec + 1.0, float(os.getenv("M1SCALP_EXIT_MAX_HOLD_SEC", "1200"))
        )
        self.max_adverse_pips = max(0.0, float(os.getenv("M1SCALP_EXIT_MAX_ADVERSE_PIPS", "8.0")))
        self.trail_from_tp_ratio = 0.82
        self.lock_from_tp_ratio = 0.45
        self.allow_negative_exit = os.getenv("M1SCALP_EXIT_ALLOW_NEGATIVE", "1").strip().lower() not in {
            "",
            "0",
            "false",
            "no",
        }
        self.rsi_fade_long = float(os.getenv("M1SCALP_EXIT_RSI_FADE_LONG", "44.0"))
        self.rsi_fade_short = float(os.getenv("M1SCALP_EXIT_RSI_FADE_SHORT", "56.0"))
        self.vwap_gap_pips = float(os.getenv("M1SCALP_EXIT_VWAP_GAP_PIPS", "0.8"))
        self.structure_adx = float(os.getenv("M1SCALP_EXIT_STRUCTURE_ADX", "20.0"))
        self.structure_gap_pips = float(os.getenv("M1SCALP_EXIT_STRUCTURE_GAP_PIPS", "1.8"))
        self.atr_spike_pips = float(os.getenv("M1SCALP_EXIT_ATR_SPIKE_PIPS", "5.0"))

    def _due(self, now: datetime) -> bool:
        if self.loop_interval <= 0:
            return True
        now_ts = now.timestamp()
        if self.last_eval_ts is None:
            self.last_eval_ts = now_ts
            return True
        if now_ts - self.last_eval_ts >= self.loop_interval:
            self.last_eval_ts = now_ts
            return True
        return False

    def _latest_mid(self) -> Optional[float]:
        ticks = live_tick_window.recent_ticks(seconds=2.0, limit=1)
        if ticks:
            try:
                return float(ticks[-1]["mid"])
            except Exception:
                pass
        try:
            return float(factor_cache_module.all_factors().get("M1", {}).get("close"))
        except Exception:
            return None

    def _context(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[Tuple[float, float]]]:
        fac_m1 = factor_cache_module.all_factors().get("M1") or {}
        def _safe_float(value: object) -> Optional[float]:
            try:
                return float(value)
            except Exception:
                return None
        rsi = _safe_float(fac_m1.get("rsi"))
        adx = _safe_float(fac_m1.get("adx"))
        atr_pips = _safe_float(fac_m1.get("atr_pips"))
        if atr_pips is None:
            atr = _safe_float(fac_m1.get("atr"))
            if atr is not None:
                atr_pips = atr * 100.0
        vwap_gap = _safe_float(fac_m1.get("vwap_gap"))
        ma10 = _safe_float(fac_m1.get("ma10"))
        ma20 = _safe_float(fac_m1.get("ma20"))
        ma_pair = (ma10, ma20) if ma10 is not None and ma20 is not None else None
        return rsi, adx, atr_pips, vwap_gap, ma_pair

    async def step(self, now: datetime, trades: List[TradeSim]) -> None:
        if not trades or not self._due(now):
            return
        for sim in trades:
            if sim.is_closed():
                continue
            trade = _trade_to_open_entry(sim)
            await self._review_trade(trade, now, sim)

    async def _review_trade(self, trade: Dict[str, Any], now: datetime, sim: TradeSim) -> None:
        trade_id = str(trade.get("trade_id"))
        units = int(trade.get("units", 0) or 0)
        if not trade_id or units == 0:
            return
        entry = float(trade.get("price") or 0.0)
        if entry <= 0.0:
            return
        side = "long" if units > 0 else "short"
        current = self._latest_mid()
        if current is None:
            return
        pnl = (current - entry) * 100.0 if side == "long" else (entry - current) * 100.0
        allow_negative = self.allow_negative_exit or pnl <= 0

        opened_at = trade.get("open_time")
        opened_dt = _parse_dt(opened_at) if isinstance(opened_at, str) else None
        hold_sec = (now - opened_dt).total_seconds() if opened_dt else 0.0

        thesis = trade.get("entry_thesis") or {}
        if not isinstance(thesis, dict):
            thesis = {}

        state = self._states.get(trade_id)
        if state is None:
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
            state = _M1State(peak=pnl, hard_stop=hard_stop_val, tp_hint=tp_hint_val)
            self._states[trade_id] = state

        strategy_tag = (
            thesis.get("strategy_tag")
            or thesis.get("strategy_tag_raw")
            or thesis.get("strategy")
            or thesis.get("tag")
            or trade.get("strategy_tag")
            or trade.get("strategy")
            or "m1scalper"
        )
        scale, _ = momentum_scale(
            pocket="scalp",
            strategy_tag=strategy_tag,
            entry_thesis=thesis,
        )

        min_hold = scale_value(self.min_hold_sec, scale=scale, floor=self.min_hold_sec)
        max_hold = scale_value(self.max_hold_sec, scale=scale, floor=self.max_hold_sec)
        max_adverse = scale_value(self.max_adverse_pips, scale=scale, floor=self.max_adverse_pips)

        tp = scale_value(self.profit_take, scale=scale, floor=self.profit_take)
        ts = scale_value(self.trail_start, scale=scale, floor=self.trail_start)
        tb = scale_value(self.trail_backoff, scale=scale, floor=self.trail_backoff)
        lb = scale_value(self.lock_buffer, scale=scale, floor=self.lock_buffer)

        if state.tp_hint:
            tp = max(tp, max(1.0, state.tp_hint * 0.9))
            ts = max(ts, max(1.0, tp * self.trail_from_tp_ratio))
            lb = max(lb, tp * self.lock_from_tp_ratio)

        state.update(pnl, lb)

        client_id = _client_id(trade)
        if hold_sec < min_hold or not client_id:
            return

        section_decision = evaluate_section_exit(
            trade,
            current_price=current,
            now=now,
            side=side,
            pocket="scalp",
            hold_sec=hold_sec,
            min_hold_sec=min_hold,
            entry_price=entry,
        )
        if section_decision.should_exit and section_decision.reason:
            await _stub_close_trade(
                trade_id,
                -units,
                client_order_id=client_id,
                allow_negative=section_decision.allow_negative,
                exit_reason=section_decision.reason,
            )
            self._states.pop(trade_id, None)
            return

        lock_trigger = max(0.6, tp * 0.35)
        if state.lock_floor is not None and state.peak >= lock_trigger and pnl > 0 and pnl <= state.lock_floor:
            await _stub_close_trade(
                trade_id,
                -units,
                client_order_id=client_id,
                allow_negative=allow_negative,
                exit_reason="lock_floor",
            )
            self._states.pop(trade_id, None)
            return

        if pnl < 0:
            rsi, adx, atr_pips, vwap_gap, ma_pair = self._context()
            if rsi is not None:
                if side == "long" and rsi <= self.rsi_fade_long:
                    await _stub_close_trade(
                        trade_id,
                        -units,
                        client_order_id=client_id,
                        allow_negative=allow_negative,
                        exit_reason="rsi_fade",
                    )
                    self._states.pop(trade_id, None)
                    return
                if side == "short" and rsi >= self.rsi_fade_short:
                    await _stub_close_trade(
                        trade_id,
                        -units,
                        client_order_id=client_id,
                        allow_negative=allow_negative,
                        exit_reason="rsi_fade",
                    )
                    self._states.pop(trade_id, None)
                    return
            if vwap_gap is not None and abs(vwap_gap) <= self.vwap_gap_pips:
                await _stub_close_trade(
                    trade_id,
                    -units,
                    client_order_id=client_id,
                    allow_negative=allow_negative,
                    exit_reason="vwap_cut",
                )
                self._states.pop(trade_id, None)
                return
            if adx is not None and ma_pair is not None:
                ma10, ma20 = ma_pair
                gap = abs(ma10 - ma20) / 0.01
                cross_bad = (side == "long" and ma10 <= ma20) or (side == "short" and ma10 >= ma20)
                if adx <= self.structure_adx and (cross_bad or gap <= self.structure_gap_pips):
                    await _stub_close_trade(
                        trade_id,
                        -units,
                        client_order_id=client_id,
                        allow_negative=allow_negative,
                        exit_reason="structure_break",
                    )
                    self._states.pop(trade_id, None)
                    return
            if atr_pips is not None and atr_pips >= self.atr_spike_pips:
                await _stub_close_trade(
                    trade_id,
                    -units,
                    client_order_id=client_id,
                    allow_negative=allow_negative,
                    exit_reason="atr_spike",
                )
                self._states.pop(trade_id, None)
                return

            if max_adverse > 0 and pnl <= -max_adverse:
                await _stub_close_trade(
                    trade_id,
                    -units,
                    client_order_id=client_id,
                    allow_negative=allow_negative,
                    exit_reason="max_adverse",
                )
                self._states.pop(trade_id, None)
                return

        if hold_sec >= max_hold and pnl > 0 and pnl <= tp * 0.7:
            await _stub_close_trade(
                trade_id,
                -units,
                client_order_id=client_id,
                allow_negative=allow_negative,
                exit_reason="time_stop",
            )
            self._states.pop(trade_id, None)
            return

        trail_trigger = max(ts, 0.0)
        if state.peak > 0 and state.peak >= trail_trigger and pnl > 0 and pnl <= state.peak - tb:
            await _stub_close_trade(
                trade_id,
                -units,
                client_order_id=client_id,
                allow_negative=allow_negative,
                exit_reason="trail_take",
            )
            self._states.pop(trade_id, None)
            return

        if pnl >= tp:
            await _stub_close_trade(
                trade_id,
                -units,
                client_order_id=client_id,
                allow_negative=allow_negative,
                exit_reason="take_profit",
            )
            self._states.pop(trade_id, None)


def _build_class_adapters() -> List[WorkerAdapter]:
    modules = [
        ("workers.macro_trendma.exit_worker", "TrendMAExitWorker"),
        ("workers.macro_donchian55.exit_worker", "DonchianExitWorker"),
        ("workers.macro_h1momentum.exit_worker", "H1MomentumExitWorker"),
        ("workers.micro_momentumburst.exit_worker", "MomentumBurstExitWorker"),
        ("workers.micro_trendmomentum.exit_worker", "TrendMomentumMicroExitWorker"),
        ("workers.micro_multistrat.exit_worker", "MicroMultiExitWorker"),
        ("workers.micro_rangebreak.exit_worker", "MicroRangeBreakExitWorker"),
        ("workers.micro_pullbackema.exit_worker", "MicroPullbackEMAExitWorker"),
        ("workers.micro_vwapbound.exit_worker", "MicroVWAPBoundExitWorker"),
        ("workers.micro_levelreactor.exit_worker", "MicroLevelReactorExitWorker"),
        ("workers.fast_scalp.exit_worker", "FastScalpExitWorker"),
        ("workers.scalp_multistrat.exit_worker", "ScalpMultiExitWorker"),
        ("workers.pullback_s5.exit_worker", "PullbackExitWorker"),
        ("workers.pullback_runner_s5.exit_worker", "PullbackRunnerExitWorker"),
        ("workers.pullback_scalp.exit_worker", "PullbackScalpExitWorker"),
        ("workers.impulse_momentum_s5.exit_worker", "ImpulseMomentumExitWorker"),
        ("workers.impulse_retest_s5.exit_worker", "ImpulseRetestExitWorker"),
        ("workers.squeeze_break_s5.exit_worker", "SqueezeBreakExitWorker"),
        ("workers.vwap_magnet_s5.exit_worker", "VWAPMagnetExitWorker"),
        ("workers.mirror_spike_s5.exit_worker", "MirrorSpikeExitWorker"),
        ("workers.mirror_spike_tight.exit_worker", "MirrorSpikeTightExitWorker"),
        ("workers.london_momentum.exit_worker", "LondonMomentumExitWorker"),
        ("workers.manual_swing.exit_worker", "ManualSwingExitWorker"),
    ]
    adapters: List[WorkerAdapter] = []
    patched_modules: List[Any] = []
    for module_path, class_name in modules:
        try:
            mod = importlib.import_module(module_path)
        except Exception:
            continue
        patched_modules.append(mod)
        worker_cls = getattr(mod, class_name, None)
        if worker_cls is None:
            continue
        try:
            worker = worker_cls()
        except Exception:
            continue
        tags = getattr(mod, "ALLOWED_TAGS", set())
        adapters.append(WorkerAdapter(class_name, worker, tags))
    _patch_close_trade(patched_modules)
    return adapters


def _build_adapters() -> List[BaseAdapter]:
    adapters: List[BaseAdapter] = []
    adapters.append(M1ScalperExitAdapter())
    adapters.extend(_build_class_adapters())
    return adapters


def _assign_adapters(
    trades: List[TradeSim],
    adapters: List[BaseAdapter],
) -> Tuple[Dict[str, BaseAdapter], Counter]:
    adapter_map: Dict[str, BaseAdapter] = {}
    for adapter in adapters:
        for tag in adapter.tags:
            if tag in adapter_map:
                continue
            adapter_map[tag] = adapter
    skipped = Counter()
    for sim in trades:
        adapter = adapter_map.get(sim.strategy_tag)
        if adapter is None:
            skipped["no_adapter"] += 1
            sim.skip_reason = "no_adapter"
            continue
        sim.adapter_name = adapter.name
    return adapter_map, skipped


async def _replay(
    *,
    ticks_dir: Path,
    instrument: str,
    start: datetime,
    end: datetime,
    warmup_min: int,
    trades: List[TradeSim],
    adapters: List[BaseAdapter],
) -> Dict[str, Any]:
    tick_files = _list_tick_files(ticks_dir, instrument)
    if not tick_files:
        raise FileNotFoundError(f"no tick files in {ticks_dir}")

    warmup_start = start - timedelta(minutes=warmup_min)
    tick_files = [item for item in tick_files if item[0] <= end]

    adapter_map, skipped = _assign_adapters(trades, adapters)

    pending = deque([t for t in trades if not t.skip_reason])
    active_by_adapter: Dict[BaseAdapter, List[TradeSim]] = {a: [] for a in adapters}

    def _activate_ready(now: datetime) -> None:
        while pending and pending[0].entry_time <= now:
            sim = pending.popleft()
            adapter = adapter_map.get(sim.strategy_tag)
            if adapter is None:
                sim.skip_reason = "no_adapter"
                skipped["no_adapter"] += 1
                continue
            active_by_adapter[adapter].append(sim)

    tick_count = 0
    last_mid = None

    async def _on_candle(tf: str, candle: Dict[str, Any]) -> None:
        await factor_cache_module.on_candle(tf, candle)

    agg = CandleAggregator(["M1", "M5", "H1", "H4", "D1"], instrument)
    for tf in ["M1", "M5", "H1", "H4", "D1"]:
        agg.subscribe(tf, lambda candle, tf=tf: _on_candle(tf, candle))

    for day, path in tick_files:
        if day > end:
            break
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                tick = _parse_tick_line(line)
                if tick is None:
                    continue
                if tick.time < warmup_start:
                    continue
                if tick.time > end and not any(active_by_adapter.values()):
                    break
                tick_count += 1
                REPLAY_STATE.now = tick.time
                REPLAY_STATE.bid = tick.bid
                REPLAY_STATE.ask = tick.ask
                REPLAY_STATE.mid = tick.mid
                last_mid = tick.mid
                live_tick_window.record(tick)
                await agg.on_tick(tick)
                if tick.time < start:
                    continue
                _activate_ready(tick.time)
                for adapter, bucket in active_by_adapter.items():
                    if not bucket:
                        continue
                    await adapter.step(tick.time, bucket)
                    bucket[:] = [t for t in bucket if not t.is_closed()]
                if not pending and not any(active_by_adapter.values()) and tick.time > end:
                    break

    if last_mid is not None:
        for bucket in active_by_adapter.values():
            for sim in bucket:
                if sim.is_closed():
                    continue
                sim.sim_exit_time = end
                sim.sim_exit_reason = "end_of_replay"
                sim.sim_exit_price = last_mid
                if sim.units > 0:
                    sim.sim_pnl_pips = (last_mid - sim.entry_price) / PIP
                else:
                    sim.sim_pnl_pips = (sim.entry_price - last_mid) / PIP

    summary = {
        "ticks": tick_count,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "warmup_start": warmup_start.isoformat(),
    }
    return {"summary": summary, "skipped": skipped}


def _summarize(trades: List[TradeSim]) -> Dict[str, Any]:
    totals = {
        "trades": len(trades),
        "actual_pips": 0.0,
        "sim_pips": 0.0,
        "delta_pips": 0.0,
        "missing_sim": 0,
        "missing_actual": 0,
        "improved": 0,
        "worsened": 0,
    }
    by_strategy: Dict[str, Dict[str, float]] = {}
    by_pocket: Dict[str, Dict[str, float]] = {}
    deltas: List[Tuple[float, TradeSim]] = []
    for sim in trades:
        if sim.actual_pnl_pips is None:
            totals["missing_actual"] += 1
            continue
        if sim.sim_pnl_pips is None:
            totals["missing_sim"] += 1
            continue
        delta = sim.sim_pnl_pips - sim.actual_pnl_pips
        totals["actual_pips"] += sim.actual_pnl_pips
        totals["sim_pips"] += sim.sim_pnl_pips
        totals["delta_pips"] += delta
        if delta > 0:
            totals["improved"] += 1
        elif delta < 0:
            totals["worsened"] += 1
        deltas.append((delta, sim))

        strat = sim.strategy_tag
        st = by_strategy.setdefault(
            strat, {"trades": 0, "actual_pips": 0.0, "sim_pips": 0.0, "delta_pips": 0.0}
        )
        st["trades"] += 1
        st["actual_pips"] += sim.actual_pnl_pips
        st["sim_pips"] += sim.sim_pnl_pips
        st["delta_pips"] += delta

        pk = sim.pocket
        pk_sum = by_pocket.setdefault(
            pk, {"trades": 0, "actual_pips": 0.0, "sim_pips": 0.0, "delta_pips": 0.0}
        )
        pk_sum["trades"] += 1
        pk_sum["actual_pips"] += sim.actual_pnl_pips
        pk_sum["sim_pips"] += sim.sim_pnl_pips
        pk_sum["delta_pips"] += delta

    deltas.sort(key=lambda x: x[0])
    worst = deltas[:10]
    best = deltas[-10:][::-1]
    return {
        "totals": totals,
        "by_strategy": by_strategy,
        "by_pocket": by_pocket,
        "worst": [
            {
                "trade_id": s.trade_id,
                "strategy_tag": s.strategy_tag,
                "pocket": s.pocket,
                "delta_pips": round(d, 3),
                "actual_pips": round(s.actual_pnl_pips or 0.0, 3),
                "sim_pips": round(s.sim_pnl_pips or 0.0, 3),
                "exit_reason": s.sim_exit_reason,
            }
            for d, s in worst
        ],
        "best": [
            {
                "trade_id": s.trade_id,
                "strategy_tag": s.strategy_tag,
                "pocket": s.pocket,
                "delta_pips": round(d, 3),
                "actual_pips": round(s.actual_pnl_pips or 0.0, 3),
                "sim_pips": round(s.sim_pnl_pips or 0.0, 3),
                "exit_reason": s.sim_exit_reason,
            }
            for d, s in best
        ],
    }


def _trade_payload(sim: TradeSim) -> Dict[str, Any]:
    return {
        "trade_id": sim.trade_id,
        "pocket": sim.pocket,
        "strategy_tag": sim.strategy_tag,
        "entry_time": sim.entry_time.isoformat(),
        "entry_price": sim.entry_price,
        "units": sim.units,
        "actual_close_time": sim.actual_close_time.isoformat() if sim.actual_close_time else None,
        "actual_exit_price": sim.actual_exit_price,
        "actual_pnl_pips": sim.actual_pnl_pips,
        "sim_exit_time": sim.sim_exit_time.isoformat() if sim.sim_exit_time else None,
        "sim_exit_price": sim.sim_exit_price,
        "sim_pnl_pips": sim.sim_pnl_pips,
        "sim_exit_reason": sim.sim_exit_reason,
        "adapter": sim.adapter_name,
        "skip_reason": sim.skip_reason,
    }


async def run(args: argparse.Namespace) -> int:
    _patch_tick_window()
    importlib.reload(factor_cache_module)
    _patch_factor_cache()

    ticks_dir = Path(args.ticks_dir)
    tick_files = _list_tick_files(ticks_dir, args.instrument)
    if not tick_files:
        raise SystemExit(f"no tick files under {ticks_dir}")

    if args.start and args.end:
        start = _parse_dt(args.start) or _infer_range_from_ticks(tick_files, args.days)[0]
        end = _parse_dt(args.end, end=True) or _infer_range_from_ticks(tick_files, args.days)[1]
    else:
        start, end = _infer_range_from_ticks(tick_files, args.days)

    strategies = {s.strip() for s in (args.strategies or "").split(",") if s.strip()}
    strategies = {_normalize_tag(s) for s in strategies} if strategies else None
    pockets = {p.strip().lower() for p in (args.pockets or "").split(",") if p.strip()}
    pockets = pockets if pockets else None

    trades = _load_trades(Path(args.db), start, end, strategies, pockets)
    if not trades:
        print("[replay] no trades in range")
        return 0
    REPLAY_STATE.trades = {t.trade_id: t for t in trades}

    adapters = _build_adapters()

    report_meta = await _replay(
        ticks_dir=ticks_dir,
        instrument=args.instrument,
        start=start,
        end=end,
        warmup_min=args.warmup_min,
        trades=trades,
        adapters=adapters,
    )

    summary = _summarize(trades)
    payload: Dict[str, Any] = {
        "summary": summary,
        "replay": report_meta,
    }
    if not args.no_trades:
        payload["trades"] = [_trade_payload(t) for t in trades]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    totals = summary["totals"]
    print(
        "[replay] trades=%d sim=%.2f actual=%.2f delta=%.2f improved=%d worsened=%d"
        % (
            totals["trades"],
            totals["sim_pips"],
            totals["actual_pips"],
            totals["delta_pips"],
            totals["improved"],
            totals["worsened"],
        )
    )
    print(f"[replay] report -> {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tick-level exit replay against trades.db.")
    parser.add_argument("--db", default="logs/trades.db", help="Path to trades.db.")
    parser.add_argument("--ticks-dir", default=f"logs/replay/{INSTRUMENT}", help="Tick log dir.")
    parser.add_argument("--instrument", default=INSTRUMENT, help="Instrument (USD_JPY).")
    parser.add_argument("--start", help="Start datetime (YYYY-MM-DD or ISO).")
    parser.add_argument("--end", help="End datetime (YYYY-MM-DD or ISO).")
    parser.add_argument("--days", type=int, default=2, help="Auto-range days if start/end omitted.")
    parser.add_argument("--warmup-min", type=int, default=360, help="Warmup minutes before start.")
    parser.add_argument("--strategies", help="Comma-separated strategy tags to include.")
    parser.add_argument("--pockets", help="Comma-separated pockets to include.")
    parser.add_argument("--out", default="tmp/replay_exit_report.json", help="Output JSON path.")
    parser.add_argument("--no-trades", action="store_true", help="Skip writing per-trade details.")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
