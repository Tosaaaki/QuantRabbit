#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay with exit_worker using entry schedules derived from replay_workers.

This matches the exit_worker-based replay pattern used for scalp_precision,
while reusing each worker's replay entry logic to generate entry events.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import importlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.replay_exit_workers as rew
import scripts.replay_workers as rw

# exit workers
import workers.impulse_break_s5.exit_worker as impulse_break_exit
import workers.impulse_retest_s5.exit_worker as impulse_retest_exit
import workers.impulse_momentum_s5.exit_worker as impulse_momentum_exit
import workers.pullback_s5.exit_worker as pullback_s5_exit

PIP = 0.01

WORKER_TAGS = {
    "impulse_break_s5": "impulse_break_s5",
    "impulse_retest_s5": "impulse_retest_s5",
    "impulse_momentum_s5": "impulse_momentum_s5",
    "pullback_s5": "pullback_s5",
}

EXIT_MODULES = {
    "impulse_break_s5": impulse_break_exit,
    "impulse_retest_s5": impulse_retest_exit,
    "impulse_momentum_s5": impulse_momentum_exit,
    "pullback_s5": pullback_s5_exit,
}

EXIT_WORKER_CLASSES = {
    "impulse_break_s5": "ImpulseBreakExitWorker",
    "impulse_retest_s5": "ImpulseRetestExitWorker",
    "impulse_momentum_s5": "ImpulseMomentumExitWorker",
    "pullback_s5": "PullbackExitWorker",
}


@dataclass
class EntryEvent:
    ts: datetime
    direction: str
    entry_price: float
    tp_pips: Optional[float]
    sl_pips: Optional[float]
    units: int


def _parse_iso(ts: str) -> datetime:
    return rew.parse_iso8601(ts)


def _load_entries_from_replay(path: Path) -> List[EntryEvent]:
    payload = json.loads(path.read_text())
    trades = payload.get("trades") or []
    entries: List[EntryEvent] = []
    for tr in trades:
        entry_time = tr.get("entry_time")
        direction = tr.get("direction")
        entry_price = tr.get("entry_price")
        if not entry_time or not direction or entry_price is None:
            continue
        try:
            entry_price = float(entry_price)
        except Exception:
            continue
        tp_price = tr.get("tp_price")
        sl_price = tr.get("sl_price")
        tp_pips = None
        sl_pips = None
        try:
            if tp_price is not None:
                tp_pips = abs(float(tp_price) - entry_price) / PIP
        except Exception:
            tp_pips = None
        try:
            if sl_price is not None:
                sl_pips = abs(float(sl_price) - entry_price) / PIP
        except Exception:
            sl_pips = None
        units = int(tr.get("units") or 10000)
        entries.append(
            EntryEvent(
                ts=_parse_iso(str(entry_time)),
                direction=str(direction),
                entry_price=entry_price,
                tp_pips=tp_pips,
                sl_pips=sl_pips,
                units=abs(units),
            )
        )
    entries.sort(key=lambda e: e.ts)
    return entries


def _run_replay_workers(
    *,
    worker: str,
    ticks_path: Path,
    out_path: Path,
    env: Optional[Dict[str, str]] = None,
    ticks_cache: Optional[List[rw.Tick]] = None,
) -> None:
    original_env = os.environ.copy()
    if env:
        os.environ.update(env)
    try:
        module_names = {
            "impulse_break_s5": ["workers.impulse_break_s5.config"],
            "impulse_retest_s5": ["workers.impulse_retest_s5.config"],
            "impulse_momentum_s5": ["workers.impulse_momentum_s5.config"],
            "pullback_s5": ["workers.pullback_s5.config"],
        }
        for mod_name in module_names.get(worker, []):
            mod = sys.modules.get(mod_name)
            if mod is not None:
                importlib.reload(mod)

        ticks = ticks_cache if ticks_cache is not None else rw.load_ticks(ticks_path)
        func_name = f"replay_{worker}"
        if not hasattr(rw, func_name):
            raise RuntimeError(f"replay function not found: {func_name}")
        result = getattr(rw, func_name)(ticks)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        for key in list(os.environ.keys()):
            if key not in original_env:
                os.environ.pop(key, None)
        for key, value in original_env.items():
            os.environ[key] = value


def _tuning_env(worker: str) -> Dict[str, str]:
    base = {"REPLAY_TUNED": "1"}
    # conservative tighten rules
    mapping = {
        "impulse_break_s5": {
            "IMPULSE_BREAK_S5_MAX_SPREAD_PIPS": "0.9",
            "IMPULSE_BREAK_S5_MIN_ATR_PIPS": "1.1",
        },
        "impulse_retest_s5": {
            "IMPULSE_RETEST_S5_MAX_SPREAD_PIPS": "0.9",
            "IMPULSE_RETEST_S5_MIN_ATR_PIPS": "1.0",
        },
        "impulse_momentum_s5": {
            "IMPULSE_MOMENTUM_S5_MAX_SPREAD_PIPS": "1.0",
            "IMPULSE_MOMENTUM_S5_MIN_ATR_PIPS": "0.8",
        },
        "pullback_s5": {
            "PULLBACK_S5_MAX_SPREAD_PIPS": "0.9",
            "PULLBACK_S5_MIN_ATR_PIPS": "1.2",
        },
    }
    tuned = mapping.get(worker, {})
    if not tuned:
        return base
    base.update(tuned)
    return base


class GenericExitRunner:
    def __init__(self, name: str, module, worker, broker: rew.SimBroker) -> None:
        self.name = name
        self.module = module
        self.worker = worker
        self.broker = broker
        self.last_eval = 0.0
        self.interval = getattr(worker, "loop_interval", 1.0)

    def _filter_trades(self, trades: list[dict]) -> list[dict]:
        if hasattr(self.worker, "_filter_trades"):
            try:
                return self.worker._filter_trades(trades)  # type: ignore[attr-defined]
            except Exception:
                return trades
        tags = getattr(self.module, "ALLOWED_TAGS", set())
        if hasattr(self.module, "_filter_trades"):
            try:
                return self.module._filter_trades(trades, tags)  # type: ignore[attr-defined]
            except Exception:
                return trades
        return trades

    def _client_id(self, trade: dict) -> Optional[str]:
        if hasattr(self.module, "_client_id"):
            try:
                cid = self.module._client_id(trade)  # type: ignore[attr-defined]
                if cid:
                    return cid
            except Exception:
                pass
        client_id = trade.get("client_order_id")
        if not client_id:
            ext = trade.get("clientExtensions")
            if isinstance(ext, dict):
                client_id = ext.get("id")
        return client_id

    def _extract_ctx(self, ctx) -> tuple[Optional[float], Optional[float], bool]:
        if ctx is None:
            return None, None, False
        if isinstance(ctx, tuple):
            mid = ctx[0] if len(ctx) > 0 else None
            rsi = ctx[1] if len(ctx) > 1 else None
            range_active = bool(ctx[2]) if len(ctx) > 2 else False
            return mid, rsi, range_active
        mid = getattr(ctx, "mid", None)
        rsi = getattr(ctx, "rsi", None)
        range_active = bool(getattr(ctx, "range_active", False))
        return mid, rsi, range_active

    async def step(self, now: datetime) -> None:
        if now.timestamp() - self.last_eval < self.interval:
            return
        self.last_eval = now.timestamp()
        positions = self.broker.get_open_positions()
        pocket = getattr(self.module, "POCKET", None) or "scalp"
        pocket_info = positions.get(pocket) or {}
        trades = list(pocket_info.get("open_trades") or [])
        trades = self._filter_trades(trades)
        if hasattr(self.worker, "_states"):
            try:
                active_ids = {str(tr.get("trade_id")) for tr in trades if tr.get("trade_id")}
                for tid in list(self.worker._states.keys()):  # type: ignore[attr-defined]
                    if tid not in active_ids:
                        self.worker._states.pop(tid, None)  # type: ignore[attr-defined]
            except Exception:
                pass
        if not trades:
            return

        if hasattr(self.worker, "_review_trade"):
            ctx = self.worker._context() if hasattr(self.worker, "_context") else None
            mid, rsi, range_active = self._extract_ctx(ctx)
            sig = inspect.signature(self.worker._review_trade)  # type: ignore[attr-defined]
            params = list(sig.parameters)
            argc = len(params)
            if params and params[0] == "self":
                argc = max(0, argc - 1)
            for tr in trades:
                if argc <= 2:
                    await self.worker._review_trade(tr, now)  # type: ignore[attr-defined]
                else:
                    if mid is None:
                        continue
                    if argc == 3:
                        await self.worker._review_trade(tr, now, mid)  # type: ignore[attr-defined]
                    elif argc == 4:
                        await self.worker._review_trade(tr, now, mid, rsi)  # type: ignore[attr-defined]
                    else:
                        await self.worker._review_trade(tr, now, mid, rsi, range_active)  # type: ignore[attr-defined]
            return

        if not hasattr(self.worker, "_evaluate"):
            return

        ctx = self.worker._context() if hasattr(self.worker, "_context") else None
        mid, rsi, range_active = self._extract_ctx(ctx)
        if mid is None:
            return

        close_fn = getattr(self.worker, "_close", None)
        close_sig = inspect.signature(close_fn) if close_fn else None

        for tr in trades:
            try:
                reason = self.worker._evaluate(tr, ctx, now)  # type: ignore[attr-defined]
            except Exception:
                continue
            if not reason:
                continue
            trade_id = str(tr.get("trade_id"))
            units = int(tr.get("units", 0) or 0)
            if units == 0:
                continue
            client_id = self._client_id(tr)
            if not client_id:
                continue
            entry_price = float(tr.get("price") or 0.0)
            if entry_price <= 0.0:
                continue
            mark_pnl = getattr(self.module, "mark_pnl_pips", None)
            pnl = mark_pnl(entry_price, units, mid=mid) if callable(mark_pnl) else 0.0
            allow_negative = pnl <= 0
            side = "long" if units > 0 else "short"
            touch_count = None
            if hasattr(self.worker, "_states"):
                try:
                    state = self.worker._states.get(trade_id)  # type: ignore[attr-defined]
                    touch_count = getattr(state, "last_touch_count", None)
                except Exception:
                    touch_count = None
            if close_fn and close_sig:
                candidate = {
                    "trade_id": trade_id,
                    "units": -units,
                    "reason": reason,
                    "pnl": pnl,
                    "side": side,
                    "range_mode": range_active,
                    "client_id": client_id,
                    "client_order_id": client_id,
                    "allow_negative": allow_negative,
                    "touch_count": touch_count,
                }
                kwargs = {k: v for k, v in candidate.items() if k in close_sig.parameters}
                await close_fn(**kwargs)
            else:
                self.broker.close_trade(trade_id, str(reason))
            if hasattr(self.worker, "_states"):
                try:
                    self.worker._states.pop(trade_id, None)  # type: ignore[attr-defined]
                except Exception:
                    pass


def _simulate(
    *,
    ticks_path: Path,
    ticks_cache: Optional[List[object]] = None,
    entries: List[EntryEvent],
    worker: str,
    out_path: Path,
    no_hard_sl: bool,
    no_hard_tp: bool,
    exclude_end_of_replay: bool,
    prefeed_h4: Optional[List[dict]] = None,
    latency_ms: float = 0.0,
    slip_base_pips: float = 0.0,
    slip_spread_coef: float = 0.0,
    slip_atr_coef: float = 0.0,
    slip_latency_coef: float = 0.0,
    fill_mode: str = "lko",
) -> dict:
    start = None
    end = None

    sim_clock = rew.SimClock()
    rew._reset_replay_state(sim_clock)

    broker = rew.SimBroker(
        slip_base_pips=slip_base_pips,
        slip_spread_coef=slip_spread_coef,
        slip_atr_coef=slip_atr_coef,
        slip_latency_coef=slip_latency_coef,
        latency_ms=latency_ms,
        fill_mode=fill_mode,
    )
    broker.hard_sl = not no_hard_sl
    broker.hard_tp = not no_hard_tp

    exit_mod = EXIT_MODULES[worker]
    rew._patch_exit_module(exit_mod, broker)

    worker_obj = None
    class_name = EXIT_WORKER_CLASSES.get(worker)
    if class_name and hasattr(exit_mod, class_name):
        worker_obj = getattr(exit_mod, class_name)()
    else:
        for name in dir(exit_mod):
            if name.endswith("ExitWorker"):
                worker_obj = getattr(exit_mod, name)()
                break
    if worker_obj is None:
        raise RuntimeError(f"exit worker not found for {worker}")

    runner = GenericExitRunner(worker, exit_mod, worker_obj, broker)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # pre-feed H4 candles (system_backtest style)
    if prefeed_h4 is None:
        prefeed_h4 = []
        h4_prefeed_builder = rew.CandleBuilder(240)
        for tick in rew._iter_ticks(ticks_path, "USD_JPY", start, end):
            closed = h4_prefeed_builder.update(tick.ts, tick.mid, 1)
            if closed:
                prefeed_h4.append(closed)
    for candle in prefeed_h4:
        loop.run_until_complete(rew.factor_cache.on_candle("H4", candle))

    builder_m1 = rew.CandleBuilder(1)
    builder_m5 = rew.CandleBuilder(5)
    builder_h1 = rew.CandleBuilder(60)

    entry_idx = 0
    total_entries = len(entries)

    tick_iter = ticks_cache if ticks_cache is not None else rew._iter_ticks(ticks_path, "USD_JPY", start, end)
    for tick in tick_iter:
        tick_ts = getattr(tick, "ts", None) or getattr(tick, "dt", None)
        if tick_ts is None:
            continue
        if not hasattr(tick, "ts"):
            try:
                setattr(tick, "ts", tick_ts)
            except Exception:
                pass
        sim_clock.now = tick.epoch
        broker.set_last_tick(tick)

        class _Tick:
            def __init__(self, bid: float, ask: float, ts: datetime) -> None:
                self.bid = bid
                self.ask = ask
                self.time = ts

        t = _Tick(tick.bid, tick.ask, tick_ts)
        rew.tick_window.record(t)
        try:
            rew.spread_monitor.update_from_tick(t)
        except Exception:
            pass

        closed_m1 = builder_m1.update(tick_ts, tick.mid, 1)
        if closed_m1:
            loop.run_until_complete(rew.factor_cache.on_candle("M1", closed_m1))
            closed_m5 = builder_m5.update(closed_m1["time"], closed_m1["close"], 1)
            if closed_m5:
                loop.run_until_complete(rew.factor_cache.on_candle("M5", closed_m5))
            closed_h1 = builder_h1.update(closed_m1["time"], closed_m1["close"], 1)
            if closed_h1:
                loop.run_until_complete(rew.factor_cache.on_candle("H1", closed_h1))

        # open entries scheduled up to this tick
        while entry_idx < total_entries and entries[entry_idx].ts <= tick_ts:
            ent = entries[entry_idx]
            direction = ent.direction.lower()
            if direction not in {"long", "short"}:
                entry_idx += 1
                continue
            broker.open_trade(
                pocket="scalp",
                strategy_tag=WORKER_TAGS[worker],
                direction=direction,
                entry_price=ent.entry_price,
                entry_time=ent.ts,
                tp_pips=ent.tp_pips,
                sl_pips=ent.sl_pips,
                units=ent.units,
                source=worker,
            )
            entry_idx += 1

        broker.check_sl_tp(tick)
        loop.run_until_complete(runner.step(tick_ts))

    # end of replay close
    last_tick = broker.last_tick
    if last_tick is not None:
        for trade_id in list(broker.open_trades.keys()):
            broker.close_trade(trade_id, "end_of_replay")

    trades = broker.closed_trades
    if exclude_end_of_replay:
        trades = [t for t in trades if t.get("reason") != "end_of_replay"]

    summary = rew._summarize(trades)
    payload = {
        "summary": summary,
        "trades": broker.closed_trades,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Exit-worker replay for S5/pullback groups.")
    ap.add_argument("--ticks", required=True, type=Path)
    ap.add_argument("--workers", required=True, help="Comma separated worker names")
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/replay_exit_workers_groups"))
    ap.add_argument("--no-hard-sl", action="store_true")
    ap.add_argument("--no-hard-tp", action="store_true")
    ap.add_argument("--exclude-end-of-replay", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--resample-sec", type=float, default=0.0, help="Optional downsample interval in seconds.")
    ap.add_argument("--realistic", action="store_true", help="Apply realistic latency/slippage defaults.")
    ap.add_argument("--slip-base-pips", type=float, default=0.0)
    ap.add_argument("--slip-spread-coef", type=float, default=0.0)
    ap.add_argument("--slip-atr-coef", type=float, default=0.0)
    ap.add_argument("--slip-latency-coef", type=float, default=0.0)
    ap.add_argument("--latency-ms", type=float, default=150.0)
    ap.add_argument(
        "--fill-mode",
        choices=("lko", "next_tick"),
        default="lko",
        help="Fill policy: lko=last known quote at/before latency target, next_tick=first tick after latency.",
    )
    return ap.parse_args()


def _resample_ticks(ticks: List[rw.Tick], sec: float) -> List[rw.Tick]:
    if sec <= 0:
        return ticks
    out: List[rw.Tick] = []
    bucket: Optional[int] = None
    last: Optional[rw.Tick] = None
    for tick in ticks:
        current_bucket = int(tick.epoch // sec)
        if bucket is None:
            bucket = current_bucket
            last = tick
            continue
        if current_bucket == bucket:
            last = tick
            continue
        if last is not None:
            out.append(last)
        bucket = current_bucket
        last = tick
    if last is not None:
        out.append(last)
    return out


def main() -> None:
    args = parse_args()
    logging.disable(logging.CRITICAL)
    if args.realistic:
        args.latency_ms = 180.0
        args.slip_base_pips = 0.02
        args.slip_spread_coef = 0.15
        args.slip_atr_coef = 0.02
        args.slip_latency_coef = 0.0006
        args.fill_mode = "next_tick"
    workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, dict] = {}

    # precompute H4 candles once to reuse across workers
    prefeed_h4: List[dict] = []
    h4_prefeed_builder = rew.CandleBuilder(240)
    replay_ticks = rw.load_ticks(args.ticks)
    if args.resample_sec and args.resample_sec > 0.0:
        replay_ticks = _resample_ticks(replay_ticks, args.resample_sec)
    for tick in replay_ticks:
        if not hasattr(tick, "ts"):
            try:
                setattr(tick, "ts", tick.dt)
            except Exception:
                pass
    for tick in replay_ticks:
        closed = h4_prefeed_builder.update(tick.dt, tick.mid, 1)
        if closed:
            prefeed_h4.append(closed)

    for worker in workers:
        if worker not in WORKER_TAGS:
            continue
        base_out = args.out_dir / f"replay_exit_{worker}_base.json"
        replay_out = args.out_dir / f"replay_workers_{worker}_base.json"
        _run_replay_workers(
            worker=worker,
            ticks_path=args.ticks,
            out_path=replay_out,
            ticks_cache=replay_ticks,
        )
        entries = _load_entries_from_replay(replay_out)
        base_summary = _simulate(
            ticks_path=args.ticks,
            ticks_cache=replay_ticks,
            entries=entries,
            worker=worker,
            out_path=base_out,
            no_hard_sl=args.no_hard_sl,
            no_hard_tp=args.no_hard_tp,
            exclude_end_of_replay=args.exclude_end_of_replay,
            prefeed_h4=prefeed_h4,
            latency_ms=args.latency_ms,
            slip_base_pips=args.slip_base_pips,
            slip_spread_coef=args.slip_spread_coef,
            slip_atr_coef=args.slip_atr_coef,
            slip_latency_coef=args.slip_latency_coef,
            fill_mode=args.fill_mode,
        )
        results[worker] = {"base": base_summary}

        if args.tune and base_summary.get("total_pnl_pips", 0.0) <= 0:
            tuned_env = os.environ.copy()
            tuned_env.update(_tuning_env(worker))
            replay_tuned_out = args.out_dir / f"replay_workers_{worker}_tuned.json"
            tuned_out = args.out_dir / f"replay_exit_{worker}_tuned.json"
            _run_replay_workers(
                worker=worker,
                ticks_path=args.ticks,
                out_path=replay_tuned_out,
                env=tuned_env,
                ticks_cache=replay_ticks,
            )
            entries = _load_entries_from_replay(replay_tuned_out)
            tuned_summary = _simulate(
                ticks_path=args.ticks,
                ticks_cache=replay_ticks,
                entries=entries,
                worker=worker,
                out_path=tuned_out,
                no_hard_sl=args.no_hard_sl,
                no_hard_tp=args.no_hard_tp,
                exclude_end_of_replay=args.exclude_end_of_replay,
                prefeed_h4=prefeed_h4,
                latency_ms=args.latency_ms,
                slip_base_pips=args.slip_base_pips,
                slip_spread_coef=args.slip_spread_coef,
                slip_atr_coef=args.slip_atr_coef,
                slip_latency_coef=args.slip_latency_coef,
                fill_mode=args.fill_mode,
            )
            results[worker]["tuned"] = tuned_summary

    summary_path = args.out_dir / "summary_all.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(summary_path))


if __name__ == "__main__":
    main()
