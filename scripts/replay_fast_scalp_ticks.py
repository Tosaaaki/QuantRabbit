#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tick-level replay harness for the FastScalp module.

This script replays recorded tick JSONL files (logs/replay/...) and runs the
FastScalp signal logic on top of the real tick sequence. Compared to the
synthetic candle interpolation harness, this preserves tick spacing, ATR/RSI,
and pattern signatures so the new strategy profile switching can be evaluated.

Example:
    python scripts/replay_fast_scalp_ticks.py \
        --ticks logs/replay/USD_JPY/USD_JPY_ticks_20251021.jsonl \
        --units 10000 --range-active
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.risk_guard import clamp_sl_tp  # noqa: E402
from workers.fast_scalp import config  # noqa: E402
from workers.fast_scalp.patterns import pattern_score  # noqa: E402
from workers.fast_scalp.profiles import StrategyProfile, select_profile  # noqa: E402
from workers.fast_scalp.signal import SignalFeatures, extract_features, evaluate_signal  # noqa: E402
from workers.fast_scalp.timeout_controller import TimeoutController  # noqa: E402


PIP_VALUE = config.PIP_VALUE


def parse_iso8601(ts: str) -> datetime:
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    if "." in ts and "+" in ts[ts.index(".") :]:
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


@dataclass(frozen=True)
class ReplayTick:
    ts: datetime
    epoch: float
    bid: float
    ask: float
    mid: float
    spread_pips: float


@dataclass
class SimTrade:
    trade_id: str
    direction: str  # "long" or "short"
    units: int
    entry_price: float
    entry_expected: float
    entry_slip_pips: float
    entry_latency_ms: float
    entry_epoch: float
    entry_time: datetime
    profile: StrategyProfile
    tp_price: float
    sl_price: float
    tp_pips: float
    sl_pips: float
    pattern_score: Optional[float]
    pattern_tag: str
    max_drawdown_close: float
    signal: str


@dataclass
class TradeResult:
    direction: str
    units: int
    entry_time: datetime
    exit_time: datetime
    hold_seconds: float
    entry_price: float
    entry_expected: float
    entry_slip_pips: float
    exit_price: float
    exit_expected: float
    exit_slip_pips: float
    tp_price: float
    sl_price: float
    pnl_pips: float
    pnl_jpy: float
    reason: str
    profile: str
    signal: str
    pattern_score: Optional[float]
    pattern_tag: str
    latency_ms: float = 0.0
    timeout_meta: Optional[Dict[str, float | str | bool]] = None


@dataclass
class PendingEntry:
    direction: str
    ready_epoch: float
    signal_time: datetime
    signal_epoch: float
    signal_price: float
    tp_price: float
    sl_price: float
    tp_pips: float
    sl_pips: float
    profile: StrategyProfile
    pattern_score: Optional[float]
    pattern_tag: str
    signal: str
    features: SignalFeatures
    tick_rate: float
    latency_ms: float


@dataclass
class PendingExit:
    reason: str
    ready_epoch: float
    trigger_time: datetime
    trigger_epoch: float
    trigger_price: float
    tick_rate: float
    latency_ms: float


def load_ticks(paths: Sequence[Path], instrument: str) -> List[ReplayTick]:
    ticks: List[ReplayTick] = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("instrument") and data["instrument"] != instrument:
                    continue
                bid = data.get("bid")
                ask = data.get("ask")
                if bid is None or ask is None:
                    continue
                try:
                    bid_f = float(bid)
                    ask_f = float(ask)
                except (TypeError, ValueError):
                    continue
                ts_raw = data.get("ts") or data.get("time")
                if not ts_raw:
                    continue
                ts = parse_iso8601(str(ts_raw))
                mid = (bid_f + ask_f) / 2.0
                spread_pips = max(0.0, ask_f - bid_f) / PIP_VALUE
                ticks.append(
                    ReplayTick(
                        ts=ts,
                        epoch=ts.timestamp(),
                        bid=bid_f,
                        ask=ask_f,
                        mid=mid,
                        spread_pips=spread_pips,
                    )
                )
    ticks.sort(key=lambda t: t.epoch)
    return ticks


def _parse_candle_time(ts: str) -> datetime:
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    if "." in ts and "+" in ts[ts.index(".") :]:
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


def load_candles(paths: Sequence[Path]) -> List[Tuple[datetime, float, float, float, float]]:
    candles: List[Tuple[datetime, float, float, float, float]] = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        items = payload.get("candles") or payload
        for cndl in items:
            mid = cndl.get("mid") or cndl
            try:
                ts = _parse_candle_time(str(cndl["time"]))
                o = float(mid["o"])
                h = float(mid["h"])
                l = float(mid["l"])
                c = float(mid["c"])
            except Exception:
                continue
            candles.append((ts, o, h, l, c))
    candles.sort(key=lambda c: c[0])
    return candles


def load_candles_payload(items: Sequence[Dict[str, object]]) -> List[Tuple[datetime, float, float, float, float]]:
    candles: List[Tuple[datetime, float, float, float, float]] = []
    for cndl in items:
        mid = cndl.get("mid") or cndl
        try:
            ts = _parse_candle_time(str(cndl["time"]))
            o = float(mid.get("o", mid.get("open", 0.0)))
            h = float(mid.get("h", mid.get("high", 0.0)))
            l = float(mid.get("l", mid.get("low", 0.0)))
            c = float(mid.get("c", mid.get("close", 0.0)))
        except Exception:
            continue
        candles.append((ts, o, h, l, c))
    candles.sort(key=lambda c: c[0])
    return candles


def synth_ticks_from_candles(
    candles: Sequence[Tuple[datetime, float, float, float, float]],
    *,
    spread_pips: float,
    steps_per_segment: int = 6,
) -> List[ReplayTick]:
    ticks: List[ReplayTick] = []
    spread_price = spread_pips * PIP_VALUE
    half_spread = spread_price / 2.0
    total = len(candles)
    for idx, (ts, o, h, l, c) in enumerate(candles):
        if total > 1:
            if idx + 1 < total:
                frame_seconds = (candles[idx + 1][0] - ts).total_seconds()
            else:
                frame_seconds = (ts - candles[idx - 1][0]).total_seconds()
        else:
            frame_seconds = 60.0
        if frame_seconds is None or frame_seconds <= 0:
            frame_seconds = 60.0
        frame_seconds = max(0.5, min(frame_seconds, 3600.0))
        if h < l:
            h, l = max(h, l), min(h, l)
        if c >= o:
            path = [o, h, l, c]
        else:
            path = [o, l, h, c]
        unique_vals = [path[0]]
        for val in path[1:]:
            if abs(val - unique_vals[-1]) < 1e-6:
                continue
            unique_vals.append(val)
        segments = len(unique_vals) - 1
        if segments <= 0:
            segments = 1
            unique_vals = [o, c]
        segment_duration = frame_seconds / float(segments)
        for seg_idx in range(segments):
            start_price = unique_vals[seg_idx]
            end_price = unique_vals[seg_idx + 1]
            for step in range(steps_per_segment):
                frac = step / float(steps_per_segment)
                price = start_price + (end_price - start_price) * frac
                dt_offset = seg_idx * segment_duration + frac * segment_duration
                tick_dt = ts + timedelta(seconds=dt_offset)
                bid = price - half_spread
                ask = price + half_spread
                ticks.append(
                    ReplayTick(
                        ts=tick_dt,
                        epoch=tick_dt.timestamp(),
                        bid=bid,
                        ask=ask,
                        mid=price,
                        spread_pips=spread_pips,
                    )
                )
        # Ensure final close tick is included
        close_dt = ts + timedelta(seconds=frame_seconds)
        bid = c - half_spread
        ask = c + half_spread
        ticks.append(
            ReplayTick(
                ts=close_dt,
                epoch=close_dt.timestamp(),
                bid=bid,
                ask=ask,
                mid=c,
                spread_pips=spread_pips,
            )
        )
    ticks.sort(key=lambda t: t.epoch)
    return ticks


class FastScalpReplayer:
    def __init__(
        self,
        *,
        units: int,
        range_active: bool,
        m1_rsi: Optional[float],
        buffer_seconds: float,
        latency_ms: float,
        slip_base_pips: float,
        slip_spread_coef: float,
        slip_atr_coef: float,
        slip_latency_coef: float,
    ) -> None:
        self.units = units
        self.range_active = range_active
        self.m1_rsi = m1_rsi
        self.buffer: Deque[dict] = deque()
        self.buffer_seconds = buffer_seconds
        self.active_trade: Optional[SimTrade] = None
        self.pending_entry: Optional[PendingEntry] = None
        self.pending_exit: Optional[PendingExit] = None
        self.results: List[TradeResult] = []
        self.profile_counter: Counter[str] = Counter()
        self.timeout_controller = TimeoutController()
        self.trade_seq = 0
        self.latency_ms = max(0.0, float(latency_ms))
        self.slip_base_pips = max(0.0, float(slip_base_pips))
        self.slip_spread_coef = max(0.0, float(slip_spread_coef))
        self.slip_atr_coef = max(0.0, float(slip_atr_coef))
        self.slip_latency_coef = max(0.0, float(slip_latency_coef))

    def _update_buffer(self, tick: ReplayTick) -> None:
        self.buffer.append({"epoch": tick.epoch, "mid": tick.mid})
        cutoff = tick.epoch - self.buffer_seconds
        while self.buffer and self.buffer[0]["epoch"] < cutoff:
            self.buffer.popleft()

    def _extract_features(self, tick: ReplayTick) -> Optional[SignalFeatures]:
        if len(self.buffer) < config.MIN_TICK_COUNT:
            return None
        ticks_list = list(self.buffer)
        return extract_features(tick.spread_pips, ticks=ticks_list)

    def _calc_slip_pips(
        self,
        *,
        spread_pips: float,
        atr_pips: Optional[float],
        latency_ms: float,
    ) -> float:
        atr_val = atr_pips or 0.0
        slip = (
            self.slip_base_pips
            + self.slip_spread_coef * max(0.0, spread_pips)
            + self.slip_atr_coef * max(0.0, atr_val)
            + self.slip_latency_coef * max(0.0, latency_ms)
        )
        return max(0.0, slip)

    def _expected_entry_price(self, tick: ReplayTick, direction: str) -> float:
        return tick.ask if direction == "long" else tick.bid

    def _expected_exit_price(self, tick: ReplayTick, direction: str) -> float:
        return tick.bid if direction == "long" else tick.ask

    def _apply_slip(self, price: float, direction: str, *, is_entry: bool, slip_pips: float) -> float:
        slip = slip_pips * PIP_VALUE
        if direction == "long":
            return price + slip if is_entry else price - slip
        return price - slip if is_entry else price + slip

    def _schedule_exit(
        self,
        *,
        reason: str,
        tick: ReplayTick,
        expected_exit_price: float,
        tick_rate: float,
        latency_ms: float,
    ) -> None:
        if self.pending_exit is not None:
            return
        ready_epoch = tick.epoch + max(0.0, latency_ms) / 1000.0
        self.pending_exit = PendingExit(
            reason=reason,
            ready_epoch=ready_epoch,
            trigger_time=tick.ts,
            trigger_epoch=tick.epoch,
            trigger_price=expected_exit_price,
            tick_rate=tick_rate,
            latency_ms=latency_ms,
        )

    def _execute_entry(self, tick: ReplayTick) -> None:
        if self.pending_entry is None or self.active_trade is not None:
            return
        pending = self.pending_entry
        expected = self._expected_entry_price(tick, pending.direction)
        slip_pips = self._calc_slip_pips(
            spread_pips=tick.spread_pips,
            atr_pips=pending.features.atr_pips,
            latency_ms=pending.latency_ms,
        )
        entry_price = self._apply_slip(expected, pending.direction, is_entry=True, slip_pips=slip_pips)

        sl_price = pending.sl_price
        tp_price = pending.tp_price
        sl_price, tp_price = clamp_sl_tp(entry_price, sl_price, tp_price, pending.direction == "long")

        trade_units = max(1, abs(self.units))
        self.trade_seq += 1
        trade_id = f"replay-{self.trade_seq}"
        self.active_trade = SimTrade(
            trade_id=trade_id,
            direction=pending.direction,
            units=trade_units,
            entry_price=entry_price,
            entry_expected=expected,
            entry_slip_pips=slip_pips,
            entry_latency_ms=pending.latency_ms,
            entry_epoch=tick.epoch,
            entry_time=tick.ts,
            profile=pending.profile,
            tp_price=tp_price,
            sl_price=sl_price,
            tp_pips=pending.tp_pips,
            sl_pips=pending.sl_pips,
            pattern_score=pending.pattern_score,
            pattern_tag=pending.pattern_tag,
            max_drawdown_close=pending.profile.drawdown_close_pips,
            signal=pending.signal,
        )
        self.timeout_controller.register_trade(
            trade_id,
            side=pending.direction,
            entry_price=entry_price,
            entry_monotonic=tick.epoch,
            features=pending.features,
            spread_pips=tick.spread_pips,
            tick_rate=pending.tick_rate,
            latency_ms=pending.latency_ms,
        )
        self.profile_counter[pending.profile.name] += 1
        self.pending_entry = None

    def _execute_exit(self, tick: ReplayTick, features: Optional[SignalFeatures]) -> None:
        if self.pending_exit is None or self.active_trade is None:
            return
        pending = self.pending_exit
        trade = self.active_trade
        expected_exit = pending.trigger_price
        atr_pips = features.atr_pips if features is not None else None
        slip_pips = self._calc_slip_pips(
            spread_pips=tick.spread_pips,
            atr_pips=atr_pips,
            latency_ms=pending.latency_ms,
        )
        raw_exit = self._expected_exit_price(tick, trade.direction)
        exit_price = self._apply_slip(raw_exit, trade.direction, is_entry=False, slip_pips=slip_pips)
        if trade.direction == "long":
            gain_pips = (exit_price - trade.entry_price) / PIP_VALUE
        else:
            gain_pips = (trade.entry_price - exit_price) / PIP_VALUE
        self._close_trade(
            tick,
            pending.reason,
            exit_price=exit_price,
            expected_exit_price=expected_exit,
            exit_slip_pips=slip_pips,
            latency_ms=pending.latency_ms,
            gain_pips=gain_pips,
            tick_rate=pending.tick_rate,
        )
        self.pending_exit = None

    def _close_trade(
        self,
        tick: ReplayTick,
        reason: str,
        *,
        exit_price: float,
        expected_exit_price: float,
        exit_slip_pips: float,
        latency_ms: float,
        gain_pips: float,
        tick_rate: float,
    ) -> None:
        if not self.active_trade:
            return
        trade = self.active_trade
        pnl_pips = gain_pips
        pnl_jpy = pnl_pips * trade.units * PIP_VALUE
        summary = self.timeout_controller.finalize(
            trade.trade_id,
            reason=reason,
            pips_gain=pnl_pips,
            tick_rate=tick_rate,
            spread_pips=tick.spread_pips,
        )
        self.results.append(
            TradeResult(
                direction=trade.direction,
                units=trade.units,
                entry_time=trade.entry_time,
                exit_time=tick.ts,
                hold_seconds=tick.epoch - trade.entry_epoch,
                entry_price=trade.entry_price,
                entry_expected=trade.entry_expected,
                entry_slip_pips=trade.entry_slip_pips,
                exit_price=exit_price,
                exit_expected=expected_exit_price,
                exit_slip_pips=exit_slip_pips,
                tp_price=trade.tp_price,
                sl_price=trade.sl_price,
                pnl_pips=round(pnl_pips, 3),
                pnl_jpy=round(pnl_jpy, 2),
                reason=reason,
                profile=trade.profile.name,
                signal=trade.signal,
                pattern_score=trade.pattern_score,
                pattern_tag=trade.pattern_tag,
                latency_ms=latency_ms,
                timeout_meta=summary if summary else None,
            )
        )
        self.active_trade = None

    def _check_exit(self, tick: ReplayTick, features: Optional[SignalFeatures]) -> None:
        if not self.active_trade:
            return
        if self.pending_exit is not None:
            return
        trade = self.active_trade
        if features is not None and features.span_seconds > 0.0:
            tick_rate = features.tick_count / max(features.span_seconds, 0.5)
        elif features is not None:
            tick_rate = float(features.tick_count)
        else:
            tick_rate = 1.0
        tick_rate = tick_rate if tick_rate > 0.0 else 0.1
        latency_ms = self.latency_ms if self.latency_ms > 0.0 else 1000.0 / max(tick_rate, 0.1)
        price = self._expected_exit_price(tick, trade.direction)
        if trade.direction == "long":
            gain_pips = (price - trade.entry_price) / PIP_VALUE
            if price >= trade.tp_price:
                self._schedule_exit(
                    reason="tp_hit",
                    tick=tick,
                    expected_exit_price=price,
                    tick_rate=tick_rate,
                    latency_ms=latency_ms,
                )
                return
            if price <= trade.sl_price:
                self._schedule_exit(
                    reason="sl_hit",
                    tick=tick,
                    expected_exit_price=price,
                    tick_rate=tick_rate,
                    latency_ms=latency_ms,
                )
                return
        else:
            gain_pips = (trade.entry_price - price) / PIP_VALUE
            if price <= trade.tp_price:
                self._schedule_exit(
                    reason="tp_hit",
                    tick=tick,
                    expected_exit_price=price,
                    tick_rate=tick_rate,
                    latency_ms=latency_ms,
                )
                return
            if price >= trade.sl_price:
                self._schedule_exit(
                    reason="sl_hit",
                    tick=tick,
                    expected_exit_price=price,
                    tick_rate=tick_rate,
                    latency_ms=latency_ms,
                )
                return

        # Hard TP/SL
        elapsed = tick.epoch - trade.entry_epoch
        reason: Optional[str] = None

        if features is None:
            return

        decision = self.timeout_controller.update(
            trade.trade_id,
            elapsed_sec=elapsed,
            pips_gain=gain_pips,
            features=features,
            tick_rate=tick_rate,
            latency_ms=latency_ms,
        )

        if features.rsi is not None and gain_pips < 0:
            if trade.direction == "long" and features.rsi < config.RSI_EXIT_LONG:
                reason = "rsi_fade"
            elif trade.direction == "short" and features.rsi > config.RSI_EXIT_SHORT:
                reason = "rsi_fade"

        if (
            reason is None
            and features.atr_pips is not None
            and gain_pips < 0
            and features.atr_pips >= config.ATR_HIGH_VOL_PIPS
        ):
            reason = "atr_spike"

        if reason is None:
            drawdown_hit = gain_pips <= -trade.max_drawdown_close
            if drawdown_hit:
                reason = "drawdown"

        if reason is None and decision.action == "close":
            reason = decision.reason or "timeout_controller"

        if reason is not None:
            self._schedule_exit(
                reason=reason,
                tick=tick,
                expected_exit_price=price,
                tick_rate=tick_rate,
                latency_ms=latency_ms,
            )

    def _is_low_quality(self, features: SignalFeatures) -> bool:
        if features.atr_pips is None or features.atr_pips < config.MIN_ENTRY_ATR_PIPS:
            return True
        if features.tick_count < config.MIN_ENTRY_TICK_COUNT:
            return True
        if features.span_seconds < config.MIN_ENTRY_TICK_SPAN_SEC:
            return True
        return False

    def on_tick(self, tick: ReplayTick) -> None:
        self._update_buffer(tick)
        features = self._extract_features(tick)

        if self.pending_exit is not None and self.active_trade and tick.epoch >= self.pending_exit.ready_epoch:
            self._execute_exit(tick, features)
            return

        if self.active_trade:
            self._check_exit(tick, features)
            if self.pending_exit is not None:
                return
            if self.active_trade:
                return

        if self.pending_entry is not None:
            if self.active_trade is None and tick.epoch >= self.pending_entry.ready_epoch:
                self._execute_entry(tick)
            return

        if tick.spread_pips > config.MAX_SPREAD_PIPS and not config.FORCE_ENTRIES:
            return

        if not features:
            return

        if self._is_low_quality(features):
            return

        action = evaluate_signal(
            features,
            m1_rsi=self.m1_rsi,
            range_active=self.range_active,
        )
        if not action:
            return

        pattern_prob: Optional[float] = None
        if features.pattern_features is not None:
            direction = "long" if action.endswith("LONG") else "short"
            pattern_prob = pattern_score(features.pattern_features, direction)
            if pattern_prob is not None and pattern_prob < config.PATTERN_MIN_PROB:
                return

        profile = select_profile(action, features, range_active=self.range_active)
        direction = "long" if action.endswith("LONG") else "short"

        spread_padding = max(tick.spread_pips, config.TP_SPREAD_BUFFER_PIPS)
        tp_margin = max(config.TP_SAFE_MARGIN_PIPS, tick.spread_pips * 0.5)
        base_tp = config.TP_BASE_PIPS + spread_padding + tp_margin
        tp_pips = max(0.2, base_tp * profile.tp_margin_multiplier + profile.tp_adjust)
        sl_pips = profile.sl_pips if profile.sl_pips is not None else config.SL_PIPS

        entry_price = self._expected_entry_price(tick, direction)
        if direction == "long":
            sl_price = entry_price - sl_pips * PIP_VALUE
            tp_price = entry_price + tp_pips * PIP_VALUE
        else:
            sl_price = entry_price + sl_pips * PIP_VALUE
            tp_price = entry_price - tp_pips * PIP_VALUE

        sl_price, tp_price = clamp_sl_tp(entry_price, sl_price, tp_price, direction == "long")
        sl_adjust = sl_pips + config.SL_POST_ADJUST_BUFFER_PIPS
        if direction == "long":
            sl_price = entry_price - sl_adjust * PIP_VALUE
        else:
            sl_price = entry_price + sl_adjust * PIP_VALUE

        entry_tick_rate = (
            features.tick_count / max(features.span_seconds, 0.5)
            if features.span_seconds > 0.0
            else float(features.tick_count)
        )
        entry_tick_rate = entry_tick_rate if entry_tick_rate > 0.0 else 0.1
        entry_latency_ms = self.latency_ms if self.latency_ms > 0.0 else 1000.0 / max(entry_tick_rate, 0.1)
        ready_epoch = tick.epoch + max(0.0, entry_latency_ms) / 1000.0
        self.pending_entry = PendingEntry(
            direction=direction,
            ready_epoch=ready_epoch,
            signal_time=tick.ts,
            signal_epoch=tick.epoch,
            signal_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            profile=profile,
            pattern_score=pattern_prob,
            pattern_tag=features.pattern_tag,
            signal=action,
            features=features,
            tick_rate=entry_tick_rate,
            latency_ms=entry_latency_ms,
        )

    def summary(self) -> dict:
        total_pips = sum(r.pnl_pips for r in self.results)
        total_jpy = sum(r.pnl_jpy for r in self.results)
        wins = [r for r in self.results if r.pnl_pips > 0]
        losses = [r for r in self.results if r.pnl_pips < 0]
        win_rate = len(wins) / len(self.results) if self.results else 0.0
        gross_win = sum(r.pnl_pips for r in wins)
        gross_loss = abs(sum(r.pnl_pips for r in losses))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else math.inf
        avg_hold = (
            sum(r.hold_seconds for r in self.results) / len(self.results)
            if self.results
            else 0.0
        )
        return {
            "trades": len(self.results),
            "total_pnl_pips": round(total_pips, 3),
            "total_pnl_jpy": round(total_jpy, 2),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else float("inf"),
            "avg_hold_seconds": round(avg_hold, 2),
            "profiles": dict(self.profile_counter),
        }


def resolve_tick_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if args.ticks:
        for item in args.ticks:
            p = Path(item)
            if p.is_dir():
                paths.extend(sorted(p.glob("*.jsonl")))
            else:
                paths.append(p)
    if args.glob:
        pattern = args.glob[0]
        paths.extend(sorted(Path(".").glob(pattern)))
    return sorted({p.resolve() for p in paths})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay FastScalp against recorded tick JSONL files.")
    parser.add_argument("--ticks", nargs="*", help="Explicit tick JSONL files or directories.")
    parser.add_argument("--glob", nargs=1, help="Glob pattern for tick files (e.g. 'logs/replay/USD_JPY/*.jsonl').")
    parser.add_argument("--instrument", default="USD_JPY", help="Instrument symbol (default: USD_JPY).")
    parser.add_argument("--units", type=int, default=config.MIN_UNITS, help="Simulation trade size (default: FAST_SCALP_MIN_UNITS).")
    parser.add_argument("--range-active", action="store_true", help="Force range mode flag during replay.")
    parser.add_argument("--m1-rsi", type=float, help="Optional constant M1 RSI bias.")
    parser.add_argument("--json-out", type=Path, help="Write detailed trade log JSON to this path.")
    parser.add_argument("--print-trades", action="store_true", help="Print each trade result to stdout.")
    parser.add_argument("--candles", nargs="*", help="M1 candle JSON (OANDA format) to synthesize ticks from.")
    parser.add_argument("--bq", action="store_true", help="Load M1 candles from BigQuery.")
    parser.add_argument("--bq-start", help="ISO timestamp (UTC) start for BigQuery candles.")
    parser.add_argument("--bq-end", help="ISO timestamp (UTC) end for BigQuery candles.")
    parser.add_argument("--bq-project", default=os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT"), help="BigQuery project override.")
    parser.add_argument("--bq-dataset", default=os.getenv("BQ_DATASET", "quantrabbit"), help="BigQuery dataset.")
    parser.add_argument("--bq-table", default=os.getenv("BQ_CANDLES_TABLE", "candles_m1"), help="BigQuery candles table.")
    parser.add_argument("--bq-instrument", default="USD_JPY", help="Instrument symbol for BigQuery candles.")
    parser.add_argument("--bq-timeframe", default="M1", help="Timeframe for BigQuery candles.")
    parser.add_argument("--bq-limit", type=int, default=None, help="Optional BigQuery row limit.")
    parser.add_argument("--bq-no-dedupe", action="store_true", help="Disable ts dedupe for BigQuery candles.")
    parser.add_argument("--bq-out", type=Path, help="Optional output path for fetched BigQuery candles JSON.")
    parser.add_argument("--latency-ms", type=float, default=150.0, help="Execution latency in ms (entry/exit).")
    parser.add_argument("--slip-base-pips", type=float, default=0.0, help="Base slippage in pips.")
    parser.add_argument("--slip-spread-coef", type=float, default=0.0, help="Slippage coefficient for spread.")
    parser.add_argument("--slip-atr-coef", type=float, default=0.0, help="Slippage coefficient for ATR.")
    parser.add_argument("--slip-latency-coef", type=float, default=0.0, help="Slippage coefficient for latency (per ms).")
    parser.add_argument("--synthetic-spread", type=float, default=0.32, help="Spread (pips) used when synthesizing ticks from candles.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tick_paths = resolve_tick_paths(args)
    ticks: List[ReplayTick] = []
    if tick_paths:
        ticks.extend(load_ticks(tick_paths, args.instrument))

    bq_candles: List[Dict[str, object]] = []
    bq_source = ""
    bq_out_path: Optional[Path] = None
    if args.bq:
        if not args.bq_start or not args.bq_end:
            raise SystemExit("--bq-start and --bq-end are required when using --bq")
        from analytics.bq_candles_loader import fetch_bq_candles, write_candles_json

        bq_candles, bq_source = fetch_bq_candles(
            instrument=args.bq_instrument,
            timeframe=args.bq_timeframe,
            start=args.bq_start,
            end=args.bq_end,
            project=args.bq_project,
            dataset=args.bq_dataset,
            table=args.bq_table,
            limit=args.bq_limit,
            dedupe=not args.bq_no_dedupe,
        )
        if args.bq_out:
            bq_out_path = args.bq_out
        else:
            try:
                start_tag = _parse_candle_time(str(args.bq_start)).strftime("%Y%m%dT%H%M%SZ")
                end_tag = _parse_candle_time(str(args.bq_end)).strftime("%Y%m%dT%H%M%SZ")
            except Exception:
                start_tag = "start"
                end_tag = "end"
            bq_out_path = Path("tmp") / f"bq_candles_{start_tag}_{end_tag}.json"
        write_candles_json(
            bq_candles,
            instrument=args.bq_instrument,
            timeframe=args.bq_timeframe,
            start=args.bq_start,
            end=args.bq_end,
            table_fqn=bq_source,
            out_path=bq_out_path,
        )
        print(f"[bq] candles={len(bq_candles)} table={bq_source} out={bq_out_path}")

    candle_paths: List[Path] = []
    if args.candles:
        for item in args.candles:
            p = Path(item)
            if p.is_dir():
                candle_paths.extend(sorted(p.glob("candles_M1_*.json")))
            else:
                candle_paths.append(p)
    if candle_paths:
        candle_payload = load_candles(candle_paths)
        synthetic = synth_ticks_from_candles(
            candle_payload,
            spread_pips=max(0.05, args.synthetic_spread),
        )
        ticks.extend(synthetic)

    if bq_candles:
        candle_payload = load_candles_payload(bq_candles)
        synthetic = synth_ticks_from_candles(
            candle_payload,
            spread_pips=max(0.05, args.synthetic_spread),
        )
        ticks.extend(synthetic)

    if not ticks:
        raise SystemExit("No valid ticks or candles found for replay.")

    replayer = FastScalpReplayer(
        units=max(config.MIN_UNITS, abs(args.units)),
        range_active=args.range_active,
        m1_rsi=args.m1_rsi,
        buffer_seconds=max(config.LONG_WINDOW_SEC * 3, 60.0),
        latency_ms=args.latency_ms,
        slip_base_pips=args.slip_base_pips,
        slip_spread_coef=args.slip_spread_coef,
        slip_atr_coef=args.slip_atr_coef,
        slip_latency_coef=args.slip_latency_coef,
    )

    ticks.sort(key=lambda t: t.epoch)
    for tick in ticks:
        replayer.on_tick(tick)

    if replayer.active_trade:
        # Force-close any open trade at final tick price for accounting purposes.
        closing_tick = ticks[-1]
        trade = replayer.active_trade
        expected_exit = replayer._expected_exit_price(closing_tick, trade.direction)
        slip_pips = replayer._calc_slip_pips(
            spread_pips=closing_tick.spread_pips,
            atr_pips=None,
            latency_ms=0.0,
        )
        exit_price = replayer._apply_slip(expected_exit, trade.direction, is_entry=False, slip_pips=slip_pips)
        if trade.direction == "long":
            final_gain = (exit_price - trade.entry_price) / PIP_VALUE
        else:
            final_gain = (trade.entry_price - exit_price) / PIP_VALUE
        replayer._close_trade(
            closing_tick,
            "end_of_replay",
            exit_price=exit_price,
            expected_exit_price=expected_exit,
            exit_slip_pips=slip_pips,
            latency_ms=0.0,
            gain_pips=final_gain,
            tick_rate=8.0,
        )

    summary = replayer.summary()
    print("----- FastScalp Tick Replay Summary -----")
    print(f"tick_files: {len(tick_paths)} candle_files: {len(candle_paths)} bq_candles: {len(bq_candles)}")
    print(f"ticks_total: {len(ticks)}")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.print_trades:
        for trade in replayer.results:
            print(
                f"{trade.entry_time.isoformat()} -> {trade.exit_time.isoformat()} "
                f"{trade.direction} pnl={trade.pnl_pips:.2f}p reason={trade.reason} profile={trade.profile}"
            )

    if args.json_out:
        payload = {
            "summary": summary,
            "trades": [
                {
                    "direction": tr.direction,
                    "units": tr.units,
                    "entry_time": tr.entry_time.isoformat(),
                    "exit_time": tr.exit_time.isoformat(),
                    "hold_seconds": tr.hold_seconds,
                    "entry_price": round(tr.entry_price, 5),
                    "entry_expected": round(tr.entry_expected, 5),
                    "entry_slip_pips": round(tr.entry_slip_pips, 4),
                    "exit_price": round(tr.exit_price, 5),
                    "exit_expected": round(tr.exit_expected, 5),
                    "exit_slip_pips": round(tr.exit_slip_pips, 4),
                    "tp_price": round(tr.tp_price, 5),
                    "sl_price": round(tr.sl_price, 5),
                    "pnl_pips": tr.pnl_pips,
                    "pnl_jpy": tr.pnl_jpy,
                    "reason": tr.reason,
                    "profile": tr.profile,
                    "signal": tr.signal,
                    "pattern_score": tr.pattern_score,
                    "pattern_tag": tr.pattern_tag,
                    "latency_ms": tr.latency_ms,
                    "timeout_meta": tr.timeout_meta,
                }
                for tr in replayer.results
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved details -> {args.json_out}")


if __name__ == "__main__":
    main()
