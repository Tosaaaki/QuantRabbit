#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight tick replay harness for the QuantRabbit worker strategies.

The goal is to approximate the live worker behaviour from recorded USD/JPY
tick streams (JSONL format) and produce a JSON report with summary metrics
and individual trade records. It is intentionally self-contained so it can
run inside the dev environment without touching OANDA or SQLite.

Supported workers:
  * fast_scalp
  * pullback_s5
  * mirror_spike_s5

Usage examples:
  python scripts/replay_workers.py --worker fast_scalp \\
      --ticks tmp/ticks_USDJPY_20250929.jsonl \\
      --out tmp/replay_fast_scalp_20250929.json

The replay model is a simplification of the live worker logic.  It focuses on
entry qualification, TP/SL handling, and timeout management, but it does not
attempt to mimic broker-side fills, stage tracking, or exit-manager overrides.
Treat the output as a comparative benchmark across code changes rather than
an exact reproduction of live trading results.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Tick:
    epoch: float
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.epoch, tz=timezone.utc)


def load_ticks(path: Path) -> List[Tick]:
    ticks: List[Tick] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            ts_raw = payload.get("timestamp") or payload.get("ts")
            if ts_raw is None:
                continue
            if isinstance(ts_raw, (int, float)):
                epoch = float(ts_raw)
            else:
                epoch = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).timestamp()
            bid = float(payload.get("bid", 0.0))
            ask = float(payload.get("ask", 0.0))
            ticks.append(Tick(epoch=epoch, bid=bid, ask=ask))
    ticks.sort(key=lambda t: t.epoch)
    return ticks


def load_s5_candles(path: Path) -> List[Tick]:
    payload = json.loads(path.read_text())
    candles = payload.get("candles", [])
    ticks: List[Tick] = []
    substeps = 10
    for candle in candles:
        time_str = candle.get("time")
        if not time_str:
            continue
        base_dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        mid = candle.get("mid", {})
        o = float(mid.get("o", 0.0))
        h = float(mid.get("h", o))
        l = float(mid.get("l", o))
        c = float(mid.get("c", o))
        segments: List[Tuple[float, float, float, float]] = [
            (o, h, 0.0, 0.35),
            (h, l, 0.35, 0.7),
            (l, c, 0.7, 1.0),
        ]
        for i in range(substeps):
            frac = i / max(substeps - 1, 1)
            price = c
            for start, end, f0, f1 in segments:
                if frac <= f1 or f1 == 1.0:
                    span = max(f1 - f0, 1e-6)
                    local = min(max((frac - f0) / span, 0.0), 1.0)
                    price = start + (end - start) * local
                    break
            dt = base_dt + timedelta(seconds=frac * 5.0)
            epoch = dt.timestamp()
            spread = 0.0004
            bid = price - spread / 2
            ask = price + spread / 2
            ticks.append(Tick(epoch=epoch, bid=bid, ask=ask))
    ticks.sort(key=lambda t: t.epoch)
    return ticks


def _z_score(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if len(vals) < 20:
        return None
    sample = vals[-20:]
    mu = sum(sample) / len(sample)
    var = sum((v - mu) ** 2 for v in sample) / max(len(sample) - 1, 1)
    if var <= 0.0:
        return 0.0
    sigma = var ** 0.5
    if sigma == 0:
        return 0.0
    return (sample[-1] - mu) / sigma


def _rsi(values: Iterable[float], period: int) -> Optional[float]:
    seq = list(values)
    if len(seq) <= 1:
        return None
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(seq)):
        diff = seq[i] - seq[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    period = max(1, min(period, len(gains)))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_from_prices(values: Iterable[float], period: int, pip_value: float) -> float:
    seq = list(values)
    if len(seq) <= 1:
        return 0.0
    trs = [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))]
    period = max(1, min(period, len(trs)))
    return (sum(trs[-period:]) / period) / pip_value


def session_tag(dt: datetime) -> str:
    hour = dt.hour
    if 22 <= hour or hour < 6:
        return "asia"
    if 6 <= hour < 12:
        return "london"
    return "newyork"


_RAW_NEWS = os.getenv("REPLAY_NEWS_TIMES", "").strip()
NEWS_EVENTS: List[datetime] = []
if _RAW_NEWS:
    for token in _RAW_NEWS.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            NEWS_EVENTS.append(datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(timezone.utc))
        except ValueError:
            continue


def is_news_block(ts: datetime, window_minutes: float) -> bool:
    if not NEWS_EVENTS or window_minutes <= 0:
        return False
    window = window_minutes * 60.0
    for event in NEWS_EVENTS:
        if abs((ts - event).total_seconds()) <= window:
            return True
    return False


def passes_quality_gate(
    ts: datetime,
    spread_pips: float,
    atr_pips: float,
    *,
    max_spread_pips: float = 0.8,
    min_atr_pips: float = 0.0,
    sessions: Optional[Iterable[str]] = None,
    news_block_min: float = 0.0,
) -> bool:
    if spread_pips > max_spread_pips:
        return False
    if min_atr_pips > 0.0 and atr_pips < min_atr_pips:
        return False
    if sessions:
        session = session_tag(ts)
        if session not in set(sessions):
            return False
    if news_block_min > 0.0 and is_news_block(ts, news_block_min):
        return False
    return True


# ---------------------------------------------------------------------------
# Fast scalp replay
# ---------------------------------------------------------------------------


def replay_fast_scalp(ticks: List[Tick]) -> Dict[str, object]:
    fs_config = __import__("workers.fast_scalp.config", fromlist=["config"])  # type: ignore[assignment]
    fs_config.REQUIRE_CONSOLIDATION = True
    signal_mod = __import__("workers.fast_scalp.signal", fromlist=["signal"])
    profiles_mod = __import__("workers.fast_scalp.profiles", fromlist=["profiles"])
    extract_features = signal_mod.extract_features
    evaluate_signal = signal_mod.evaluate_signal
    select_profile = profiles_mod.select_profile

    density_buffer: Deque[Tick] = deque()
    max_ticks_in_window = 0
    window_span = fs_config.LONG_WINDOW_SEC or 9.0
    sample_limit = min(len(ticks), 5000)
    for sample_tick in ticks[:sample_limit]:
        density_buffer.append(sample_tick)
        cutoff = sample_tick.epoch - window_span
        while density_buffer and density_buffer[0].epoch < cutoff:
            density_buffer.popleft()
        if len(density_buffer) > max_ticks_in_window:
            max_ticks_in_window = len(density_buffer)
    if max_ticks_in_window <= 0:
        max_ticks_in_window = fs_config.MIN_TICK_COUNT

    if max_ticks_in_window >= 8:
        effective_entry_ticks = min(fs_config.MIN_ENTRY_TICK_COUNT, max_ticks_in_window - 1)
        effective_min_ticks = min(fs_config.MIN_TICK_COUNT, max_ticks_in_window - 2)
    else:
        effective_entry_ticks = min(fs_config.MIN_ENTRY_TICK_COUNT, max_ticks_in_window)
        effective_min_ticks = min(fs_config.MIN_TICK_COUNT, max_ticks_in_window)

    effective_entry_ticks = max(4, int(effective_entry_ticks))
    effective_min_ticks = max(3, min(int(effective_min_ticks), effective_entry_ticks))
    fs_config.MIN_ENTRY_TICK_COUNT = effective_entry_ticks
    fs_config.MIN_TICK_COUNT = effective_min_ticks

    if max_ticks_in_window <= 6:
        span_floor = 0.0004
        fs_config.MIN_ENTRY_TICK_SPAN_SEC = span_floor

    cons_cap = max(3, max_ticks_in_window // 5)
    fs_config.CONSOLIDATION_MIN_TICKS = max(
        3, min(getattr(fs_config, "CONSOLIDATION_MIN_TICKS", 6), cons_cap)
    )
    imp_cap = max(3, max_ticks_in_window // 5)
    fs_config.IMPULSE_MIN_TICKS = max(
        3, min(getattr(fs_config, "IMPULSE_MIN_TICKS", 6), imp_cap)
    )
    fs_config.IMPULSE_LOOKBACK_SEC = max(getattr(fs_config, "IMPULSE_LOOKBACK_SEC", 2.0), 3.0)

    feature_sample: List[Dict[str, float]] = []
    atr_samples: List[float] = []
    for sample_tick in ticks[:sample_limit]:
        feature_sample.append(
            {
                "epoch": sample_tick.epoch,
                "mid": sample_tick.mid,
                "bid": sample_tick.bid,
                "ask": sample_tick.ask,
            }
        )
        cutoff = sample_tick.epoch - window_span
        while feature_sample and feature_sample[0]["epoch"] < cutoff:
            feature_sample.pop(0)
        spread = (sample_tick.ask - sample_tick.bid) / fs_config.PIP_VALUE
        sample_features = extract_features(spread, ticks=feature_sample)
        if sample_features and sample_features.atr_pips is not None:
            atr_samples.append(float(sample_features.atr_pips))
    original_consolidation_range = getattr(fs_config, "CONSOLIDATION_MAX_RANGE_PIPS", 0.55)
    if atr_samples:
        atr_samples.sort()
        median_atr = atr_samples[len(atr_samples) // 2]
        fs_config.MIN_ENTRY_ATR_PIPS = max(
            0.04, min(fs_config.MIN_ENTRY_ATR_PIPS, median_atr * 0.5)
        )
    else:
        fs_config.MIN_ENTRY_ATR_PIPS = max(0.04, min(fs_config.MIN_ENTRY_ATR_PIPS, 0.055))
    fs_config.MIN_IMPULSE_PIPS = min(
        getattr(fs_config, "MIN_IMPULSE_PIPS", 0.7),
        max(0.05, fs_config.MIN_ENTRY_ATR_PIPS * 0.5),
    )
    if max_ticks_in_window <= 6:
        fs_config.MIN_IMPULSE_PIPS = 0.0
    if hasattr(fs_config, "CONSOLIDATION_MAX_RANGE_PIPS"):
        fs_config.CONSOLIDATION_MAX_RANGE_PIPS = max(
            0.45, min(0.85, original_consolidation_range * 1.4)
        )

    loop_interval = fs_config.LOOP_INTERVAL_SEC
    min_units = max(fs_config.MIN_UNITS, 1000)
    require_consolidation = fs_config.REQUIRE_CONSOLIDATION
    min_hold_sec = max(0.6, getattr(fs_config, "MIN_HOLD_SEC", 0.6))
    max_hold_k = getattr(fs_config, "MAX_HOLD_ATR_K", 1.0)
    cost_cfg = {"slippage": 0.05, "commission": 0.0}
    gate_kwargs = {
        "max_spread_pips": 0.8,
        "min_atr_pips": max(0.01, fs_config.MIN_ENTRY_ATR_PIPS * 0.45),
        "sessions": {"london", "newyork"},
        "news_block_min": 5.0,
    }
    if max_ticks_in_window <= 6:
        gate_kwargs["sessions"] = None

    trades: List[Dict[str, object]] = []
    open_positions: List[Dict[str, object]] = []
    tick_buffer: List[Dict[str, float]] = []

    next_eval_epoch = ticks[0].epoch
    loss_streak = 0
    loss_cooldown_until = ticks[0].epoch - 1.0
    cooldown_until = { "long": ticks[0].epoch - 1.0, "short": ticks[0].epoch - 1.0 }

    def close_position(pos: Dict[str, object], reason: str, exit_tick: Tick, *, force: bool = False) -> None:
        nonlocal loss_streak, loss_cooldown_until
        hold_sec = exit_tick.epoch - pos["entry_epoch"]
        if not force and hold_sec < min_hold_sec:
            return
        entry_mid = pos["entry_price"]
        direction = pos["side"]
        exit_mid = exit_tick.mid
        gross_pips = (exit_mid - entry_mid) / fs_config.PIP_VALUE
        if direction == "short":
            gross_pips = -gross_pips
        exit_spread = (exit_tick.ask - exit_tick.bid) / fs_config.PIP_VALUE
        cost_pips = pos.get("entry_spread", 0.0) + exit_spread + cost_cfg["slippage"] + cost_cfg["commission"]
        net_pips = gross_pips - cost_pips
        trades.append(
            {
                "direction": direction,
                "units": pos["units"],
                "entry_time": pos["entry_time"].isoformat(),
                "exit_time": exit_tick.dt.isoformat(),
                "hold_seconds": hold_sec,
                "entry_price": round(entry_mid, 5),
                "exit_price": round(exit_mid, 5),
                "tp_price": round(pos["tp_price"], 5),
                "sl_price": round(pos["sl_price"], 5),
                "gross_pips": round(gross_pips, 3),
                "cost_pips": round(cost_pips, 3),
                "pnl_pips": round(net_pips, 3),
                "reason": reason,
                "profile": pos["profile"],
                "signal": pos["signal"],
                "pattern_tag": pos.get("pattern_tag"),
            }
        )
        if net_pips < 0:
            loss_streak += 1
            if fs_config.LOSS_STREAK_MAX and loss_streak >= fs_config.LOSS_STREAK_MAX:
                loss_cooldown_until = exit_tick.epoch + fs_config.LOSS_STREAK_COOLDOWN_MIN * 60.0
        else:
            loss_streak = 0
        open_positions.remove(pos)

    for tick in ticks:
        tick_buffer.append(
            {
                "epoch": tick.epoch,
                "bid": tick.bid,
                "ask": tick.ask,
                "mid": tick.mid,
            }
        )
        # Prune outdated ticks
        cutoff = tick.epoch - fs_config.LONG_WINDOW_SEC
        while tick_buffer and tick_buffer[0]["epoch"] < cutoff:
            tick_buffer.pop(0)

        # Update open positions
        for pos in list(open_positions):
            hold_age = tick.epoch - pos["entry_epoch"]
            atr_snapshot = pos.get("atr_snapshot", 0.1)
            max_hold_sec = max(30.0, atr_snapshot * 60.0 * max_hold_k) if max_hold_k > 0 else float("inf")
            if hold_age >= max_hold_sec:
                close_position(pos, "max_hold", tick, force=True)
                continue

            if pos["side"] == "long":
                gain_pips = (tick.bid - pos["entry_price"]) / fs_config.PIP_VALUE
                if gain_pips >= pos.get("tp1_trigger", 0.0) and not pos.get("tp1_done"):
                    pos["tp1_done"] = True
                if gain_pips >= pos.get("be_trigger", 0.0) and not pos.get("be_done"):
                    pos["be_done"] = True
                    pos["sl_price"] = max(pos["sl_price"], pos["entry_price"])
                if gain_pips >= pos.get("trail_trigger", 0.0):
                    target = tick.bid - pos.get("trail_step", 0.3) * fs_config.PIP_VALUE
                    pos["sl_price"] = max(pos["sl_price"], round(target, 5))
                if tick.bid <= pos["sl_price"]:
                    close_position(pos, "stop", tick, force=True)
                    continue
                if tick.bid >= pos["tp_price"]:
                    close_position(pos, "take_profit", tick, force=True)
                    continue
            else:
                gain_pips = (pos["entry_price"] - tick.ask) / fs_config.PIP_VALUE
                if gain_pips >= pos.get("tp1_trigger", 0.0) and not pos.get("tp1_done"):
                    pos["tp1_done"] = True
                if gain_pips >= pos.get("be_trigger", 0.0) and not pos.get("be_done"):
                    pos["be_done"] = True
                    pos["sl_price"] = min(pos["sl_price"], pos["entry_price"])
                if gain_pips >= pos.get("trail_trigger", 0.0):
                    target = tick.ask + pos.get("trail_step", 0.3) * fs_config.PIP_VALUE
                    pos["sl_price"] = min(pos["sl_price"], round(target, 5))
                if tick.ask >= pos["sl_price"]:
                    close_position(pos, "stop", tick, force=True)
                    continue
                if tick.ask <= pos["tp_price"]:
                    close_position(pos, "take_profit", tick, force=True)
                    continue

            if tick.epoch - pos["entry_epoch"] >= pos["timeout_sec"]:
                close_position(pos, "timeout", tick, force=True)

        if tick.epoch < next_eval_epoch:
            continue
        next_eval_epoch = tick.epoch + loop_interval

        if tick.epoch < loss_cooldown_until:
            continue

        spread_pips = (tick.ask - tick.bid) / fs_config.PIP_VALUE
        features = extract_features(spread_pips, ticks=tick_buffer)
        if features is None:
            continue
        atr_recent = float(features.atr_pips or 0.0)
        if not passes_quality_gate(tick.dt, spread_pips, atr_recent, **gate_kwargs):
            continue
        from dataclasses import replace

        momentum_sign = 0
        if features.momentum_pips > 0:
            momentum_sign = 1
        elif features.momentum_pips < 0:
            momentum_sign = -1
        if features.impulse_direction == 0 and momentum_sign != 0:
            features = replace(
                features,
                impulse_direction=momentum_sign,
                impulse_pips=max(features.impulse_pips, abs(features.momentum_pips)),
            )
        if not features.consolidation_ok and not require_consolidation:
            features = replace(features, consolidation_ok=True)
        action = evaluate_signal(
            features,
            m1_rsi=None,
            range_active=False,
        )
        if action not in {"OPEN_LONG", "OPEN_SHORT"}:
            continue

        direction = "long" if action.endswith("LONG") else "short"
        if tick.epoch < cooldown_until[direction]:
            continue

        profile = select_profile(action, features, range_active=False)
        sl_pips = profile.drawdown_close_pips
        tp_pips = (
            fs_config.TP_BASE_PIPS + profile.tp_adjust
        ) * profile.tp_margin_multiplier
        if fs_config.FIXED_UNITS and abs(fs_config.FIXED_UNITS) >= min_units:
            units = abs(int(fs_config.FIXED_UNITS))
        else:
            units = min_units

        entry_price = tick.ask if direction == "long" else tick.bid
        sl_price = (
            entry_price - sl_pips * fs_config.PIP_VALUE
            if direction == "long"
            else entry_price + sl_pips * fs_config.PIP_VALUE
        )
        tp_price = (
            entry_price + tp_pips * fs_config.PIP_VALUE
            if direction == "long"
            else entry_price - tp_pips * fs_config.PIP_VALUE
        )

        open_positions.append(
            {
                "side": direction,
                "units": units,
                "entry_price": entry_price,
                "entry_epoch": tick.epoch,
                "entry_time": tick.dt,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "entry_spread": spread_pips,
                "timeout_sec": profile.timeout_sec or fs_config.TIMEOUT_SEC_BASE,
                "profile": profile.name,
                "signal": action,
                "pattern_tag": features.pattern_tag,
                "tp1_trigger": sl_pips * 0.5,
                "be_trigger": sl_pips * 0.6,
                "trail_trigger": sl_pips * 0.8,
                "trail_step": 0.3,
                "tp1_done": False,
                "be_done": False,
                "atr_snapshot": max(0.05, atr_recent),
            }
        )
        cooldown_until[direction] = tick.epoch + fs_config.ENTRY_COOLDOWN_SEC

    # Close residual positions at final tick mid
    if open_positions:
        last_tick = ticks[-1]
        for pos in list(open_positions):
            close_position(pos, "eod", last_tick)

    summary = {
        "trades": len(trades),
        "total_pnl_pips": round(sum(t["pnl_pips"] for t in trades), 3),
        "total_pnl_jpy": round(sum(t["pnl_pips"] for t in trades) * 100, 3),
        "win_rate": round(
            sum(1 for t in trades if t["pnl_pips"] > 0) / len(trades), 4
        )
        if trades
        else 0.0,
        "profit_factor": (
            round(
                sum(t["pnl_pips"] for t in trades if t["pnl_pips"] > 0)
                / abs(sum(t["pnl_pips"] for t in trades if t["pnl_pips"] < 0)),
                4,
            )
            if any(t["pnl_pips"] < 0 for t in trades)
            else float("inf")
        )
        if trades
        else 0.0,
        "avg_hold_seconds": round(
            sum(t["hold_seconds"] for t in trades) / len(trades), 3
        )
        if trades
        else 0.0,
    }

    profiles: Dict[str, int] = {}
    for t in trades:
        profiles[t["profile"]] = profiles.get(t["profile"], 0) + 1
    summary["profiles"] = profiles

    return {
        "summary": summary,
        "trades": trades,
    }


# ---------------------------------------------------------------------------
# Pullback S5 replay (simplified)
# ---------------------------------------------------------------------------


def replay_pullback_s5(ticks: List[Tick]) -> Dict[str, object]:
    pb_config = __import__("workers.pullback_s5.config", fromlist=["config"])

    window_sec = pb_config.WINDOW_SEC
    bucket_span = pb_config.BUCKET_SECONDS

    trades: List[Dict[str, object]] = []
    open_trade: Optional[Dict[str, object]] = None
    last_entry_epoch: float = ticks[0].epoch - pb_config.COOLDOWN_SEC

    bucket: List[Dict[str, float]] = []
    bucket_end = ticks[0].epoch - (ticks[0].epoch % bucket_span) + bucket_span

    def finalize_bucket(mid: float) -> None:
        nonlocal open_trade
        if not bucket:
            return
        candle = {
            "start": bucket[0]["epoch"],
            "end": bucket[-1]["epoch"],
            "open": bucket[0]["mid"],
            "high": max(b["mid"] for b in bucket),
            "low": min(b["mid"] for b in bucket),
            "close": bucket[-1]["mid"],
        }
        bucket.clear()
        bucket.append(
            {
                "epoch": candle["end"],
                "mid": candle["close"],
            }
        )

        # simple entry condition based on z-score approximations
        closes = [candle["close"]]
        if len(closes) < pb_config.FAST_BUCKETS:
            return

    for tick in ticks:
        if tick.epoch >= bucket_end:
            bucket_end += bucket_span
        bucket.append({"epoch": tick.epoch, "mid": tick.mid})

        if open_trade:
            if open_trade["side"] == "long":
                if tick.bid <= open_trade["sl_price"]:
                    open_trade["exit"] = tick
                    open_trade["reason"] = "stop"
            else:
                if tick.ask >= open_trade["sl_price"]:
                    open_trade["exit"] = tick
                    open_trade["reason"] = "stop"
            if tick.epoch - open_trade["entry_tick"].epoch >= 180.0:
                open_trade["exit"] = tick
                open_trade["reason"] = "timeout"
            if open_trade.get("exit"):
                exit_tick = open_trade["exit"]
                entry_tick = open_trade["entry_tick"]
                entry_mid = entry_tick.mid
                exit_mid = exit_tick.mid
                pnl_pips = (exit_mid - entry_mid) / pb_config.PIP_VALUE
                if open_trade["side"] == "short":
                    pnl_pips = -pnl_pips
                trades.append(
                    {
                        "direction": open_trade["side"],
                        "entry_time": entry_tick.dt.isoformat(),
                        "exit_time": exit_tick.dt.isoformat(),
                        "entry_price": round(entry_mid, 5),
                        "exit_price": round(exit_mid, 5),
                        "tp_price": round(open_trade["tp_price"], 5),
                        "sl_price": round(open_trade["sl_price"], 5),
                        "pnl_pips": round(pnl_pips, 3),
                        "reason": open_trade["reason"],
                    }
                )
                open_trade = None
                last_entry_epoch = tick.epoch

        if open_trade is None and tick.epoch - last_entry_epoch >= pb_config.COOLDOWN_SEC:
            trend = "long" if tick.mid > bucket[0]["mid"] else "short"
            side = trend
            entry_price = tick.ask if side == "long" else tick.bid
            tp_price = (
                entry_price + pb_config.TP_PIPS * pb_config.PIP_VALUE
                if side == "long"
                else entry_price - pb_config.TP_PIPS * pb_config.PIP_VALUE
            )
            sl_price = (
                entry_price - pb_config.MIN_SL_PIPS * pb_config.PIP_VALUE
                if side == "long"
                else entry_price + pb_config.MIN_SL_PIPS * pb_config.PIP_VALUE
            )
            open_trade = {
                "side": side,
                "entry_tick": tick,
                "tp_price": tp_price,
                "sl_price": sl_price,
            }

    if open_trade:
        exit_tick = ticks[-1]
        entry_mid = open_trade["entry_tick"].mid
        pnl_pips = (exit_tick.mid - entry_mid) / pb_config.PIP_VALUE
        if open_trade["side"] == "short":
            pnl_pips = -pnl_pips
        trades.append(
            {
                "direction": open_trade["side"],
                "entry_time": open_trade["entry_tick"].dt.isoformat(),
                "exit_time": exit_tick.dt.isoformat(),
                "entry_price": round(entry_mid, 5),
                "exit_price": round(exit_tick.mid, 5),
                "tp_price": round(open_trade["tp_price"], 5),
                "sl_price": round(open_trade["sl_price"], 5),
                "pnl_pips": round(pnl_pips, 3),
                "reason": "eod",
            }
        )

    summary = {
        "trades": len(trades),
        "total_pnl_pips": round(sum(t["pnl_pips"] for t in trades), 3),
        "total_pnl_jpy": round(sum(t["pnl_pips"] for t in trades) * 100, 3),
        "win_rate": round(
            sum(1 for t in trades if t["pnl_pips"] > 0) / len(trades), 4
        )
        if trades
        else 0.0,
    }
    return {"summary": summary, "trades": trades}


# ---------------------------------------------------------------------------
# Mirror spike S5 replay (simplified)
# ---------------------------------------------------------------------------


def replay_mirror_spike_s5(ticks: List[Tick]) -> Dict[str, object]:
    ms_config = __import__("workers.mirror_spike_s5.config", fromlist=["config"])

    trades: List[Dict[str, object]] = []
    open_trade: Optional[Dict[str, object]] = None

    for tick in ticks:
        if open_trade:
            if open_trade["side"] == "long":
                if tick.bid <= open_trade["sl_price"]:
                    open_trade["reason"] = "stop"
                elif tick.bid >= open_trade["tp_price"]:
                    open_trade["reason"] = "take_profit"
            else:
                if tick.ask >= open_trade["sl_price"]:
                    open_trade["reason"] = "stop"
                elif tick.ask <= open_trade["tp_price"]:
                    open_trade["reason"] = "take_profit"
            if tick.epoch - open_trade["entry_tick"].epoch >= 300.0:
                open_trade["reason"] = "timeout"
            if open_trade.get("reason"):
                entry_mid = open_trade["entry_tick"].mid
                exit_mid = tick.mid
                pnl_pips = (exit_mid - entry_mid) / ms_config.PIP_VALUE
                if open_trade["side"] == "short":
                    pnl_pips = -pnl_pips
                trades.append(
                    {
                        "direction": open_trade["side"],
                        "entry_time": open_trade["entry_tick"].dt.isoformat(),
                        "exit_time": tick.dt.isoformat(),
                        "entry_price": round(entry_mid, 5),
                        "exit_price": round(exit_mid, 5),
                        "tp_price": round(open_trade["tp_price"], 5),
                        "sl_price": round(open_trade["sl_price"], 5),
                        "pnl_pips": round(pnl_pips, 3),
                        "reason": open_trade["reason"],
                    }
                )
                open_trade = None
            continue

        # look for simple wick reversal pattern
        body = abs(tick.ask - tick.bid)
        if body <= 0.0001:
            continue
        # We approximate spike detection by comparing to previous 5 seconds
        # For brevity we simply sample the previous tick (lagging by 1)
        # In practice, more sophisticated detection should be used.
        # Here we alternate between long and short impulses.
        candidate_side = "short" if len(trades) % 2 == 0 else "long"
        tp_pips = ms_config.TP_PIPS
        sl_pips = ms_config.SL_PIPS
        entry_price = tick.ask if candidate_side == "long" else tick.bid
        tp_price = (
            entry_price + tp_pips * ms_config.PIP_VALUE
            if candidate_side == "long"
            else entry_price - tp_pips * ms_config.PIP_VALUE
        )
        sl_price = (
            entry_price - sl_pips * ms_config.PIP_VALUE
            if candidate_side == "long"
            else entry_price + sl_pips * ms_config.PIP_VALUE
        )
        open_trade = {
            "side": candidate_side,
            "entry_tick": tick,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }

    if open_trade:
        exit_tick = ticks[-1]
        entry_mid = open_trade["entry_tick"].mid
        pnl_pips = (exit_tick.mid - entry_mid) / ms_config.PIP_VALUE
        if open_trade["side"] == "short":
            pnl_pips = -pnl_pips
        trades.append(
            {
                "direction": open_trade["side"],
                "entry_time": open_trade["entry_tick"].dt.isoformat(),
                "exit_time": exit_tick.dt.isoformat(),
                "entry_price": round(entry_mid, 5),
                "exit_price": round(exit_tick.mid, 5),
                "tp_price": round(open_trade["tp_price"], 5),
                "sl_price": round(open_trade["sl_price"], 5),
                "pnl_pips": round(pnl_pips, 3),
                "reason": "eod",
            }
        )

    summary = {
        "trades": len(trades),
        "total_pnl_pips": round(sum(t["pnl_pips"] for t in trades), 3),
        "total_pnl_jpy": round(sum(t["pnl_pips"] for t in trades) * 100, 3),
        "win_rate": round(
            sum(1 for t in trades if t["pnl_pips"] > 0) / len(trades), 4
        )
        if trades
        else 0.0,
    }
    return {"summary": summary, "trades": trades}


# ---------------------------------------------------------------------------
# Mirror spike (tick-based) replay using detection helpers
# ---------------------------------------------------------------------------


def replay_mirror_spike(ticks: List[Tick]) -> Dict[str, object]:
    ms_mod = __import__("workers.mirror_spike.worker", fromlist=["worker"])
    ms_config = ms_mod.config
    detect_signal: Callable[[List[dict], object], Optional[object]] = ms_mod._detect_signal
    extract_features = __import__("workers.fast_scalp.signal", fromlist=["signal"]).extract_features

    ms_config.SPIKE_THRESHOLD_PIPS = max(1.5, ms_config.SPIKE_THRESHOLD_PIPS * 0.7)
    ms_config.RETRACE_TRIGGER_PIPS = max(0.2, ms_config.RETRACE_TRIGGER_PIPS * 0.7)
    ms_config.MIN_RETRACE_PIPS = max(0.15, ms_config.MIN_RETRACE_PIPS * 0.5)
    ms_config.MIN_TICK_COUNT = min(ms_config.MIN_TICK_COUNT, 40)
    ms_config.MIN_TICK_RATE = min(ms_config.MIN_TICK_RATE, 0.4)

    tick_buffer: List[dict] = []
    trades: List[Dict[str, object]] = []
    open_trade: Optional[Dict[str, object]] = None
    cooldown_until = ticks[0].epoch - ms_config.COOLDOWN_SEC

    for tick in ticks:
        tick_buffer.append({"epoch": tick.epoch, "mid": tick.mid, "bid": tick.bid, "ask": tick.ask})
        cutoff = tick.epoch - ms_config.LOOKBACK_SEC
        while tick_buffer and tick_buffer[0]["epoch"] < cutoff:
            tick_buffer.pop(0)
        if len(tick_buffer) < ms_config.MIN_TICK_COUNT:
            continue

        if open_trade:
            if open_trade["side"] == "long":
                if tick.bid <= open_trade["sl_price"]:
                    open_trade["reason"] = "stop"
                elif tick.bid >= open_trade["tp_price"]:
                    open_trade["reason"] = "take_profit"
            else:
                if tick.ask >= open_trade["sl_price"]:
                    open_trade["reason"] = "stop"
                elif tick.ask <= open_trade["tp_price"]:
                    open_trade["reason"] = "take_profit"
            if tick.epoch - open_trade["entry_tick"].epoch >= ms_config.POST_EXIT_COOLDOWN_SEC:
                open_trade["reason"] = open_trade.get("reason") or "timeout"
            if open_trade.get("reason"):
                entry_tick = open_trade["entry_tick"]
                entry_mid = entry_tick.mid
                exit_mid = tick.mid
                pnl_pips = (exit_mid - entry_mid) / ms_config.PIP_VALUE
                if open_trade["side"] == "short":
                    pnl_pips = -pnl_pips
                trades.append(
                    {
                        "direction": open_trade["side"],
                        "entry_time": entry_tick.dt.isoformat(),
                        "exit_time": tick.dt.isoformat(),
                        "entry_price": round(entry_mid, 5),
                        "exit_price": round(exit_mid, 5),
                        "tp_price": round(open_trade["tp_price"], 5),
                        "sl_price": round(open_trade["sl_price"], 5),
                        "pnl_pips": round(pnl_pips, 3),
                        "reason": open_trade["reason"],
                    }
                )
                cooldown_until = tick.epoch + ms_config.POST_EXIT_COOLDOWN_SEC
                open_trade = None
            continue

        if tick.epoch < cooldown_until:
            continue

        features = extract_features((tick.ask - tick.bid) / ms_config.PIP_VALUE, ticks=tick_buffer)
        if features is None:
            continue
        signal = detect_signal(tick_buffer, features)
        if not signal and len(tick_buffer) >= 10:
            delta = tick_buffer[-1]["mid"] - tick_buffer[-10]["mid"]
            if abs(delta) >= 0.00005:
                side = "long" if delta < 0 else "short"
                signal = type("FallbackSignal", (), {"side": side})()
        if not signal:
            continue

        if signal.side == "short":
            entry_price = tick.bid
            tp_price = entry_price - ms_config.TP_PIPS * ms_config.PIP_VALUE
            sl_price = entry_price + ms_config.SL_PIPS * ms_config.PIP_VALUE
        else:
            entry_price = tick.ask
            tp_price = entry_price + ms_config.TP_PIPS * ms_config.PIP_VALUE
            sl_price = entry_price - ms_config.SL_PIPS * ms_config.PIP_VALUE

        open_trade = {
            "side": signal.side,
            "entry_tick": tick,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }

    if open_trade:
        exit_tick = ticks[-1]
        entry_mid = open_trade["entry_tick"].mid
        pnl_pips = (exit_tick.mid - entry_mid) / ms_config.PIP_VALUE
        if open_trade["side"] == "short":
            pnl_pips = -pnl_pips
        trades.append(
            {
                "direction": open_trade["side"],
                "entry_time": open_trade["entry_tick"].dt.isoformat(),
                "exit_time": exit_tick.dt.isoformat(),
                "entry_price": round(entry_mid, 5),
                "exit_price": round(exit_tick.mid, 5),
                "tp_price": round(open_trade["tp_price"], 5),
                "sl_price": round(open_trade["sl_price"], 5),
                "pnl_pips": round(pnl_pips, 3),
                "reason": "eod",
            }
        )

    summary = {
        "trades": len(trades),
        "total_pnl_pips": round(sum(t["pnl_pips"] for t in trades), 3),
        "total_pnl_jpy": round(sum(t["pnl_pips"] for t in trades) * 100, 3),
        "win_rate": round(
            sum(1 for t in trades if t["pnl_pips"] > 0) / len(trades), 4
        )
        if trades
        else 0.0,
    }
    return {"summary": summary, "trades": trades}


# ---------------------------------------------------------------------------
# Scalp exit manager replay (wraps fast scalper simulation)
# ---------------------------------------------------------------------------


def replay_scalp_exit(ticks: List[Tick], *, candle_path: Optional[Path] = None) -> Dict[str, object]:
    exit_mod = __import__("workers.scalp_exit.worker", fromlist=["worker"])
    exit_config = exit_mod.config
    ScalpExitManager = exit_mod.ScalpExitManager

    # Prepare fast scalp simulation with exit manager hooked in.
    result = replay_fast_scalp(ticks)
    trade_records = result.get("trades", [])
    if not trade_records:
        return {"summary": {"trades": 0, "exit_triggers": {}}, "trades": []}

    # Aggregate exit reasons by replaying manager decisions on the recorded trades.
    manager = ScalpExitManager()
    reasons: Dict[str, int] = {}
    exit_trades: List[Dict[str, object]] = []

    # Build simple M1 factor stream from ticks or optional candle file.
    m1_candles: List[dict] = []
    def _aggregate_candle(epoch: float, price: float) -> None:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        if not m1_candles or dt.minute != m1_candles[-1]["time"].minute:
            m1_candles.append({
                "time": dt.replace(second=0, microsecond=0),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            })
        else:
            candle = m1_candles[-1]
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price

    for tick in ticks:
        _aggregate_candle(tick.epoch, tick.mid)

    def _current_factors(price: float) -> dict:
        candles = [
            {
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
            }
            for c in m1_candles[-120:]
        ]
        closes = [c["close"] for c in m1_candles]
        rsi = _rsi(closes, 14) if len(closes) >= 15 else 50.0
        return {"candles": candles, "rsi": rsi or 50.0}

    exit_mod.all_factors = lambda: {"M1": _current_factors(last_price)}  # type: ignore[assignment]
    last_price = ticks[-1].mid

    for tr in trade_records:
        side = tr["direction"]
        entry_price = tr["entry_price"]
        exit_price = tr["exit_price"]
        pnl_pips = tr["pnl_pips"]
        trade_id = f"replay-{len(exit_trades)+1}"
        units = 2000 if side == "long" else -2000
        trade_dict = {
            "trade_id": trade_id,
            "units": units,
            "price": entry_price,
            "open_time": tr["entry_time"],
            "entry_thesis": {"strategy_tag": "fast_scalp"},
        }
        last_price = exit_price
        exit_mod._latest_mid = lambda: exit_price  # type: ignore[assignment]
        reason = manager.evaluate(trade_dict, _current_factors(exit_price), datetime.fromisoformat(tr["exit_time"].replace("Z", "+00:00")))
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
        exit_trades.append({"trade_id": trade_id, "pnl_pips": pnl_pips, "reason": reason or "original"})

    summary = {
        "trades": len(exit_trades),
        "exit_triggers": reasons,
    }
    return {"summary": summary, "trades": exit_trades}


# ---------------------------------------------------------------------------
# Pullback scalp (M1/M5) replay (simplified)
# ---------------------------------------------------------------------------


def replay_pullback_scalp(ticks: List[Tick]) -> Dict[str, object]:
    from workers.pullback_scalp import config as ps_config

    ps_config.M1_Z_MIN = min(ps_config.M1_Z_MIN, 0.1)
    ps_config.M1_Z_MAX = max(ps_config.M1_Z_MAX, 0.4)
    ps_config.M5_Z_SHORT_MAX = max(ps_config.M5_Z_SHORT_MAX, 0.35)
    ps_config.M5_Z_LONG_MIN = min(ps_config.M5_Z_LONG_MIN, -0.35)
    ps_config.MIN_ATR_PIPS = min(ps_config.MIN_ATR_PIPS, 0.0)
    ps_config.COOLDOWN_SEC = min(ps_config.COOLDOWN_SEC, 40.0)

    m1_window: Deque[Tuple[float, float]] = deque()
    m5_window: Deque[Tuple[float, float]] = deque()
    trades: List[Dict[str, object]] = []
    open_trade: Optional[Dict[str, object]] = None
    last_entry_epoch = ticks[0].epoch - ps_config.COOLDOWN_SEC

    def _append(window: Deque[Tuple[float, float]], tick: Tick, span: float) -> None:
        window.append((tick.epoch, tick.mid))
        cutoff = tick.epoch - span
        while window and window[0][0] < cutoff:
            window.popleft()

    def _vals(window: Deque[Tuple[float, float]]) -> List[float]:
        return [mid for _, mid in window]

    for tick in ticks:
        _append(m1_window, tick, ps_config.M1_WINDOW_SEC)
        _append(m5_window, tick, ps_config.M5_WINDOW_SEC)
        if len(m1_window) < 20 or len(m5_window) < 40:
            continue

        if open_trade:
            if open_trade["side"] == "long":
                if tick.bid <= open_trade["sl_price"]:
                    open_trade["reason"] = "stop"
                elif tick.bid >= open_trade["tp_price"]:
                    open_trade["reason"] = "tp"
            else:
                if tick.ask >= open_trade["sl_price"]:
                    open_trade["reason"] = "stop"
                elif tick.ask <= open_trade["tp_price"]:
                    open_trade["reason"] = "tp"
            if open_trade.get("reason"):
                entry_tick = open_trade["entry_tick"]
                entry_mid = entry_tick.mid
                exit_mid = tick.mid
                pnl_pips = (exit_mid - entry_mid) / ps_config.PIP_VALUE
                if open_trade["side"] == "short":
                    pnl_pips = -pnl_pips
                trades.append(
                    {
                        "direction": open_trade["side"],
                        "entry_time": entry_tick.dt.isoformat(),
                        "exit_time": tick.dt.isoformat(),
                        "entry_price": round(entry_mid, 5),
                        "exit_price": round(exit_mid, 5),
                        "tp_price": round(open_trade["tp_price"], 5),
                        "sl_price": round(open_trade["sl_price"], 5) if open_trade["sl_price"] is not None else None,
                        "pnl_pips": round(pnl_pips, 3),
                        "reason": open_trade["reason"],
                    }
                )
                open_trade = None
                last_entry_epoch = tick.epoch
            continue

        if tick.epoch - last_entry_epoch < ps_config.COOLDOWN_SEC:
            continue

        m1_vals = _vals(m1_window)
        m5_vals = _vals(m5_window)
        z_m1 = _z_score(m1_vals)
        z_m5 = _z_score(m5_vals)
        if z_m1 is None or z_m5 is None:
            continue
        if ps_config.M1_Z_TRIGGER > 0.0 and abs(z_m1) < ps_config.M1_Z_TRIGGER:
            continue
        rsi_m1 = _rsi(m1_vals, ps_config.RSI_PERIOD)
        atr_m1 = _atr_from_prices(m1_vals, min(12, max(6, ps_config.RSI_PERIOD)), ps_config.PIP_VALUE)
        if ps_config.MIN_ATR_PIPS > 0.0 and atr_m1 < ps_config.MIN_ATR_PIPS:
            continue

        side: Optional[str] = None
        if ps_config.M1_Z_MIN <= z_m1 <= ps_config.M1_Z_MAX and z_m5 <= ps_config.M5_Z_SHORT_MAX:
            if rsi_m1 is None or ps_config.RSI_SHORT_RANGE[0] <= rsi_m1 <= ps_config.RSI_SHORT_RANGE[1]:
                side = "short"
        elif -ps_config.M1_Z_MAX <= z_m1 <= -ps_config.M1_Z_MIN and z_m5 >= ps_config.M5_Z_LONG_MIN:
            if rsi_m1 is None or ps_config.RSI_LONG_RANGE[0] <= rsi_m1 <= ps_config.RSI_LONG_RANGE[1]:
                side = "long"
        if side is None and len(m1_vals) >= 5:
            delta = m1_vals[-1] - m1_vals[-5]
            if abs(delta) >= 0.00005:
                side = "long" if delta > 0 else "short"
        if side is None:
            continue

        entry_price = tick.ask if side == "long" else tick.bid
        tp_price = entry_price + ps_config.TP_PIPS * ps_config.PIP_VALUE if side == "long" else entry_price - ps_config.TP_PIPS * ps_config.PIP_VALUE
        sl_price = None
        if ps_config.USE_INITIAL_SL:
            sl_pips = max(ps_config.MIN_SL_PIPS, atr_m1 * ps_config.SL_ATR_MULT)
            sl_price = entry_price - sl_pips * ps_config.PIP_VALUE if side == "long" else entry_price + sl_pips * ps_config.PIP_VALUE

        open_trade = {
            "side": side,
            "entry_tick": tick,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }

    if open_trade:
        exit_tick = ticks[-1]
        entry_mid = open_trade["entry_tick"].mid
        pnl_pips = (exit_tick.mid - entry_mid) / ps_config.PIP_VALUE
        if open_trade["side"] == "short":
            pnl_pips = -pnl_pips
        trades.append(
            {
                "direction": open_trade["side"],
                "entry_time": open_trade["entry_tick"].dt.isoformat(),
                "exit_time": exit_tick.dt.isoformat(),
                "entry_price": round(entry_mid, 5),
                "exit_price": round(exit_tick.mid, 5),
                "tp_price": round(open_trade["tp_price"], 5),
                "sl_price": round(open_trade["sl_price"] or entry_mid, 5) if open_trade["sl_price"] is not None else None,
                "pnl_pips": round(pnl_pips, 3),
                "reason": "eod",
            }
        )

    summary = {
        "trades": len(trades),
        "total_pnl_pips": round(sum(t["pnl_pips"] for t in trades), 3),
        "total_pnl_jpy": round(sum(t["pnl_pips"] for t in trades) * 100, 3),
        "win_rate": round(
            sum(1 for t in trades if t["pnl_pips"] > 0) / len(trades), 4
        )
        if trades
        else 0.0,
    }
    return {"summary": summary, "trades": trades}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Simplified tick replays for worker strategies.")
    ap.add_argument(
        "--worker",
        required=True,
        choices=[
            "fast_scalp",
            "pullback_s5",
            "mirror_spike_s5",
            "pullback_scalp",
            "mirror_spike",
            "scalp_exit",
        ],
    )
    ap.add_argument("--ticks", help="Tick JSONL file (timestamp,bid,ask).")
    ap.add_argument("--candles", help="S5 candle JSON to generate synthetic ticks.")
    ap.add_argument("--out", default="", help="Optional output JSON path.")
    args = ap.parse_args()

    if not args.ticks and not args.candles:
        raise SystemExit("--ticks か --candles のいずれかを指定してください")
    if args.ticks:
        tick_path = Path(args.ticks)
        if not tick_path.exists():
            raise SystemExit(f"tick file not found: {tick_path}")
        ticks = load_ticks(tick_path)
    else:
        candle_path = Path(args.candles)
        if not candle_path.exists():
            raise SystemExit(f"candle file not found: {candle_path}")
        ticks = load_s5_candles(candle_path)
    if not ticks:
        raise SystemExit("no ticks loaded")

    if args.worker == "fast_scalp":
        result = replay_fast_scalp(ticks)
    elif args.worker == "pullback_s5":
        result = replay_pullback_s5(ticks)
    elif args.worker == "mirror_spike_s5":
        result = replay_mirror_spike_s5(ticks)
    elif args.worker == "pullback_scalp":
        result = replay_pullback_scalp(ticks)
    elif args.worker == "mirror_spike":
        result = replay_mirror_spike(ticks)
    else:
        result = replay_scalp_exit(ticks, candle_path=Path(args.candles) if args.candles else None)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
