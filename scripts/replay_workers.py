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
  * impulse_break_s5
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
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import time

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.pseudo_cfg import DensityCfg, ShapeCfg, SimCfg, SpreadCfg
from sim.pseudo_ticks import synth_from_candles


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
                if epoch > 10**11:  # assume milliseconds
                    epoch /= 1000.0
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


def _atr_from_closes(values: Sequence[float], period: int, pip_value: float) -> float:
    closes = list(values)
    if len(closes) <= 1 or period <= 0:
        return 0.0
    true_ranges = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    if not true_ranges:
        return 0.0
    period = max(1, min(period, len(true_ranges)))
    return sum(true_ranges[-period:]) / period / pip_value


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


FAST_SCALP_EXIT_SCHEMES: Dict[str, Dict[str, float]] = {
    "A": {"tp1_r": 0.40, "be_r": 0.50, "trail_start_r": 0.70, "trail_step_pips": 0.20},
    "B": {"tp1_r": 0.50, "be_r": 0.60, "trail_start_r": 0.80, "trail_step_pips": 0.30},
    "C": {"tp1_r": 0.60, "be_r": 0.70, "trail_start_r": 0.90, "trail_step_pips": 0.30},
}


# ---------------------------------------------------------------------------
# Fast scalp replay
# ---------------------------------------------------------------------------


def replay_fast_scalp(
    ticks: List[Tick],
    *,
    exit_scheme: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    fs_config = __import__("workers.fast_scalp.config", fromlist=["config"])  # type: ignore[assignment]
    signal_mod = __import__("workers.fast_scalp.signal", fromlist=["signal"])
    profiles_mod = __import__("workers.fast_scalp.profiles", fromlist=["profiles"])
    extract_features = signal_mod.extract_features
    evaluate_signal = signal_mod.evaluate_signal
    select_profile = profiles_mod.select_profile

    overrides = overrides or {}
    if "require_consolidation" in overrides:
        fs_config.REQUIRE_CONSOLIDATION = bool(overrides["require_consolidation"])
    else:
        fs_config.REQUIRE_CONSOLIDATION = True
    if overrides.get("consolidation_window_sec") is not None:
        fs_config.CONSOLIDATION_WINDOW_SEC = max(0.1, float(overrides["consolidation_window_sec"]))
    if overrides.get("consolidation_range_pips") is not None:
        fs_config.CONSOLIDATION_MAX_RANGE_PIPS = max(
            0.0, float(overrides["consolidation_range_pips"])
        )
    if overrides.get("impulse_window_sec") is not None:
        fs_config.IMPULSE_LOOKBACK_SEC = max(0.2, float(overrides["impulse_window_sec"]))
    if overrides.get("min_impulse_pips") is not None:
        fs_config.MIN_IMPULSE_PIPS = max(0.0, float(overrides["min_impulse_pips"]))
    if overrides.get("min_entry_atr_pips") is not None:
        fs_config.MIN_ENTRY_ATR_PIPS = max(0.0, float(overrides["min_entry_atr_pips"]))

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
    if overrides.get("impulse_window_sec") is None:
        fs_config.IMPULSE_LOOKBACK_SEC = max(
            getattr(fs_config, "IMPULSE_LOOKBACK_SEC", 2.0), 3.0
        )

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
    if atr_samples and "min_entry_atr_pips" not in overrides:
        atr_samples.sort()
        median_atr = atr_samples[len(atr_samples) // 2]
        fs_config.MIN_ENTRY_ATR_PIPS = max(
            0.04, min(fs_config.MIN_ENTRY_ATR_PIPS, median_atr * 0.5)
        )
    elif "min_entry_atr_pips" not in overrides:
        fs_config.MIN_ENTRY_ATR_PIPS = max(0.04, min(fs_config.MIN_ENTRY_ATR_PIPS, 0.055))
    if "min_impulse_pips" not in overrides:
        fs_config.MIN_IMPULSE_PIPS = min(
            getattr(fs_config, "MIN_IMPULSE_PIPS", 0.7),
            max(0.05, fs_config.MIN_ENTRY_ATR_PIPS * 0.5),
        )
    if max_ticks_in_window <= 6 and "min_impulse_pips" not in overrides:
        fs_config.MIN_IMPULSE_PIPS = 0.0
    if (
        hasattr(fs_config, "CONSOLIDATION_MAX_RANGE_PIPS")
        and "consolidation_range_pips" not in overrides
    ):
        fs_config.CONSOLIDATION_MAX_RANGE_PIPS = max(
            0.45, min(0.85, original_consolidation_range * 1.4)
        )

    scheme_key = (exit_scheme or os.getenv("FAST_SCALP_EXIT_SCHEME", "B")).strip().upper()
    if scheme_key not in FAST_SCALP_EXIT_SCHEMES:
        scheme_key = "B"
    exit_cfg = FAST_SCALP_EXIT_SCHEMES[scheme_key]

    loop_interval = fs_config.LOOP_INTERVAL_SEC
    min_units = max(fs_config.MIN_UNITS, 1000)
    require_consolidation = fs_config.REQUIRE_CONSOLIDATION
    if overrides.get("min_hold_ms") is not None:
        min_hold_sec = max(0.0, float(overrides["min_hold_ms"]) / 1000.0)
    else:
        min_hold_sec = max(0.6, getattr(fs_config, "MIN_HOLD_SEC", 0.6))
    max_hold_k = getattr(fs_config, "MAX_HOLD_ATR_K", 1.0)
    cost_cfg = {"slippage": 0.05, "commission": 0.0}
    gate_kwargs = {
        "max_spread_pips": 0.8,
        "min_atr_pips": max(0.01, fs_config.MIN_ENTRY_ATR_PIPS * 0.45),
        "sessions": {"london", "newyork"},
        "news_block_min": 5.0,
    }
    if overrides.get("max_spread_pips") is not None:
        gate_kwargs["max_spread_pips"] = max(0.0, float(overrides["max_spread_pips"]))
    if overrides.get("gate_min_atr_pips") is not None:
        gate_kwargs["min_atr_pips"] = max(0.0, float(overrides["gate_min_atr_pips"]))
    elif overrides.get("min_entry_atr_pips") is not None:
        gate_kwargs["min_atr_pips"] = max(0.0, float(overrides["min_entry_atr_pips"]))
    if overrides.get("news_block_min") is not None:
        gate_kwargs["news_block_min"] = max(0.0, float(overrides["news_block_min"]))
    session_override = overrides.get("sessions")
    if session_override is not None:
        tokens = [token.strip().lower() for token in str(session_override).split(",") if token.strip()]
        gate_kwargs["sessions"] = set(tokens) if tokens else None
    tickrate_window_sec = float(overrides.get("tickrate_window_sec", 5.0))
    min_tickrate = overrides.get("tickrate_min")
    if max_ticks_in_window <= 6:
        gate_kwargs["sessions"] = None

    trades: List[Dict[str, object]] = []
    open_positions: List[Dict[str, object]] = []
    tick_buffer: List[Dict[str, float]] = []

    next_eval_epoch = ticks[0].epoch
    loss_streak = 0
    loss_cooldown_until = ticks[0].epoch - 1.0
    if overrides.get("entry_cooldown_sec") is not None:
        entry_cooldown_sec = max(0.0, float(overrides["entry_cooldown_sec"]))
    else:
        entry_cooldown_sec = max(15.0, getattr(fs_config, "ENTRY_COOLDOWN_SEC", 1.0))
    cooldown_until = {"long": ticks[0].epoch - 1.0, "short": ticks[0].epoch - 1.0}

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
        if (
            min_tickrate is not None
            and min_tickrate > 0
            and tickrate_window_sec > 0
        ):
            recent_tick_count = sum(
                1 for _t in tick_buffer if tick.epoch - _t["epoch"] <= tickrate_window_sec
            )
            if recent_tick_count < min_tickrate:
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
                "tp1_trigger": sl_pips * exit_cfg["tp1_r"],
                "be_trigger": sl_pips * exit_cfg["be_r"],
                "trail_trigger": sl_pips * exit_cfg["trail_start_r"],
                "trail_step": exit_cfg["trail_step_pips"],
                "tp1_done": False,
                "be_done": False,
                "atr_snapshot": max(0.05, atr_recent),
                "exit_scheme": scheme_key,
            }
        )
        cooldown_until[direction] = tick.epoch + entry_cooldown_sec

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
    cfg = __import__("workers.pullback_s5.config", fromlist=["config"])

    bucket: List[Dict[str, float]] = []
    bucket_end = ticks[0].epoch - (ticks[0].epoch % cfg.BUCKET_SECONDS) + cfg.BUCKET_SECONDS
    trades: List[Dict[str, object]] = []
    open_positions: List[Dict[str, object]] = []
    last_entry_epoch = ticks[0].epoch - cfg.COOLDOWN_SEC

    def _stage_units(side: str) -> Optional[int]:
        same_side = [pos for pos in open_positions if pos["side"] == side]
        stage_idx = len(same_side)
        if len(open_positions) >= cfg.MAX_ACTIVE_TRADES:
            return None
        ratios = cfg.ENTRY_STAGE_RATIOS
        ratio = ratios[stage_idx] if stage_idx < len(ratios) else ratios[-1]
        units = int(round(cfg.ENTRY_UNITS * ratio))
        return max(1000, units)

    candles: List[Dict[str, float]] = []
    allowed_hours = getattr(cfg, "ALLOWED_HOURS_UTC", frozenset())
    active_hours = getattr(cfg, "ACTIVE_HOURS_UTC", frozenset())

    for tick in ticks:
        bucket.append({"epoch": tick.epoch, "mid": tick.mid})
        if tick.epoch >= bucket_end:
            high = max(b["mid"] for b in bucket)
            low = min(b["mid"] for b in bucket)
            closes = bucket[-1]["mid"]
            candle = {
                "close": closes,
                "high": high,
                "low": low,
            }
            candles.append(candle)
            if len(candles) > cfg.SLOW_BUCKETS * 3:
                candles.pop(0)
            bucket = []
            bucket_end += cfg.BUCKET_SECONDS

        if open_positions:
            remaining_positions: List[Dict[str, object]] = []
            for pos in open_positions:
                side = pos["side"]
                if side == "long":
                    stop_hit = tick.bid <= pos["sl_price"]
                    tp_hit = tick.bid >= pos["tp_price"]
                else:
                    stop_hit = tick.ask >= pos["sl_price"]
                    tp_hit = tick.ask <= pos["tp_price"]
                if stop_hit or tp_hit:
                    pnl = (
                        (tick.mid - pos["entry_price"]) / cfg.PIP_VALUE
                        if side == "long"
                        else (pos["entry_price"] - tick.mid) / cfg.PIP_VALUE
                    )
                    trades.append(
                        {
                            "direction": side,
                            "entry_time": pos["entry_tick"].dt.isoformat(),
                            "exit_time": tick.dt.isoformat(),
                            "entry_price": round(pos["entry_price"], 5),
                            "exit_price": round(tick.mid, 5),
                            "tp_price": round(pos["tp_price"], 5),
                            "sl_price": round(pos["sl_price"], 5),
                            "pnl_pips": round(pnl, 3),
                            "reason": "tp" if tp_hit else "sl",
                            "units": pos["units"],
                            "stage_index": pos["stage_index"],
                        }
                    )
                    last_entry_epoch = tick.epoch
                else:
                    remaining_positions.append(pos)
            open_positions = remaining_positions

        if tick.epoch - last_entry_epoch < cfg.COOLDOWN_SEC:
            continue
        if len(candles) < max(cfg.FAST_BUCKETS, cfg.SLOW_BUCKETS):
            continue

        tick_hour = tick.dt.hour
        if allowed_hours and tick_hour not in allowed_hours:
            continue
        if active_hours and tick_hour not in active_hours:
            continue

        closes = [c["close"] for c in candles]
        fast_series = closes[-cfg.FAST_BUCKETS :]
        slow_series = closes[-cfg.SLOW_BUCKETS :]

        def _zscore(series: Sequence[float]) -> Optional[float]:
            if len(series) < 5:
                return None
            mean_val = sum(series) / len(series)
            var = sum((x - mean_val) ** 2 for x in series) / max(len(series) - 1, 1)
            if var <= 0.0:
                return 0.0
            return (series[-1] - mean_val) / (var ** 0.5)

        z_fast = _zscore(fast_series)
        z_slow = _zscore(slow_series)
        if z_fast is None or z_slow is None:
            continue

        atr_fast = _atr_from_closes(fast_series, cfg.RSI_PERIOD, cfg.PIP_VALUE)
        if atr_fast < cfg.MIN_ATR_PIPS:
            continue

        rsi_fast = _rsi(fast_series, cfg.RSI_PERIOD)

        side: Optional[str] = None
        if cfg.FAST_Z_MIN <= z_fast <= cfg.FAST_Z_MAX and z_slow <= cfg.SLOW_Z_SHORT_MAX:
            if rsi_fast is None or cfg.RSI_SHORT_RANGE[0] <= rsi_fast <= cfg.RSI_SHORT_RANGE[1]:
                side = "short"
        elif -cfg.FAST_Z_MAX <= z_fast <= -cfg.FAST_Z_MIN and z_slow >= cfg.SLOW_Z_LONG_MIN:
            if rsi_fast is None or cfg.RSI_LONG_RANGE[0] <= rsi_fast <= cfg.RSI_LONG_RANGE[1]:
                side = "long"
        if side is None:
            continue

        if not cfg.ALLOW_DUPLICATE_ENTRIES:
            if any(pos["side"] == side for pos in open_positions):
                continue

        units = _stage_units(side)
        if units is None:
            continue

        entry_price = tick.ask if side == "long" else tick.bid
        tp_pips = cfg.TP_PIPS
        if atr_fast > 0.0:
            tp_pips = min(
                cfg.TP_ATR_MAX_PIPS,
                max(cfg.TP_ATR_MIN_PIPS, atr_fast * cfg.TP_ATR_MULT),
            )
        tp_price = (
            entry_price + tp_pips * cfg.PIP_VALUE
            if side == "long"
            else entry_price - tp_pips * cfg.PIP_VALUE
        )
        sl_pips = max(cfg.MIN_SL_PIPS, atr_fast * cfg.SL_ATR_MULT)
        sl_price = (
            entry_price - sl_pips * cfg.PIP_VALUE
            if side == "long"
            else entry_price + sl_pips * cfg.PIP_VALUE
        )
        stage_index = sum(1 for pos in open_positions if pos["side"] == side) + 1
        open_positions.append(
            {
                "side": side,
                "entry_tick": tick,
                "entry_price": entry_price,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "units": units,
                "stage_index": stage_index,
            }
        )
        last_entry_epoch = tick.epoch

    if open_positions:
        exit_tick = ticks[-1]
        for pos in open_positions:
            side = pos["side"]
            pnl = (
                (exit_tick.mid - pos["entry_price"]) / cfg.PIP_VALUE
                if side == "long"
                else (pos["entry_price"] - exit_tick.mid) / cfg.PIP_VALUE
            )
            trades.append(
                {
                    "direction": side,
                    "entry_time": pos["entry_tick"].dt.isoformat(),
                    "exit_time": exit_tick.dt.isoformat(),
                    "entry_price": round(pos["entry_price"], 5),
                    "exit_price": round(exit_tick.mid, 5),
                    "tp_price": round(pos["tp_price"], 5),
                    "sl_price": round(pos["sl_price"], 5),
                    "pnl_pips": round(pnl, 3),
                    "reason": "eod",
                    "units": pos["units"],
                    "stage_index": pos["stage_index"],
                }
            )

    total_pnl = sum(t["pnl_pips"] for t in trades)
    win_rate = (
        sum(1 for t in trades if t["pnl_pips"] > 0) / len(trades) if trades else 0.0
    )
    summary = {
        "trades": len(trades),
        "total_pnl_pips": round(total_pnl, 3),
        "total_pnl_jpy": round(total_pnl * 100, 3),
        "win_rate": round(win_rate, 4),
    }
    return {"summary": summary, "trades": trades}


# ---------------------------------------------------------------------------
# VWAP magnet S5 replay (simplified)
# ---------------------------------------------------------------------------


def replay_vwap_magnet_s5(ticks: List[Tick]) -> Dict[str, object]:
    cfg = __import__("workers.vwap_magnet_s5.config", fromlist=["config"])

    span = cfg.BUCKET_SECONDS
    bucket: List[Dict[str, float]] = []
    bucket_end = ticks[0].epoch - (ticks[0].epoch % span) + span
    close_series: List[float] = []
    weight_series: List[float] = []
    trades: List[Dict[str, object]] = []
    last_entry_epoch = ticks[0].epoch - cfg.COOLDOWN_SEC
    open_positions: List[Dict[str, object]] = []

    def _stage_units(side: str) -> Optional[int]:
        same_side = [pos for pos in open_positions if pos["side"] == side]
        stage_idx = len(same_side)
        if len(open_positions) >= cfg.MAX_ACTIVE_TRADES:
            return None
        ratios = cfg.ENTRY_STAGE_RATIOS
        ratio = ratios[stage_idx] if stage_idx < len(ratios) else ratios[-1]
        units = int(round(cfg.ENTRY_UNITS * ratio))
        return max(1000, units)

    for tick in ticks:
        bucket.append({"epoch": tick.epoch, "mid": tick.mid})

        if open_positions:
            remaining: List[Dict[str, object]] = []
            for pos in open_positions:
                side = pos["side"]
                entry_price = pos["entry_price"]
                if side == "long":
                    stop_hit = tick.bid <= pos["sl_price"]
                    tp_hit = tick.bid >= pos["tp_price"]
                else:
                    stop_hit = tick.ask >= pos["sl_price"]
                    tp_hit = tick.ask <= pos["tp_price"]
                if stop_hit or tp_hit:
                    pnl = (
                        (tick.mid - entry_price) / cfg.PIP_VALUE
                        if side == "long"
                        else (entry_price - tick.mid) / cfg.PIP_VALUE
                    )
                    trades.append(
                        {
                            "direction": side,
                            "entry_time": pos["entry_tick"].dt.isoformat(),
                            "exit_time": tick.dt.isoformat(),
                            "entry_price": round(entry_price, 5),
                            "exit_price": round(tick.mid, 5),
                            "tp_price": round(pos["tp_price"], 5),
                            "sl_price": round(pos["sl_price"], 5),
                            "pnl_pips": round(pnl, 3),
                            "reason": "tp" if tp_hit else "sl",
                            "units": pos["units"],
                            "stage_index": pos["stage_index"],
                        }
                    )
                    last_entry_epoch = tick.epoch
                else:
                    remaining.append(pos)
            open_positions = remaining

        if tick.epoch < bucket_end:
            continue

        candle = {
            "close": bucket[-1]["mid"],
            "count": float(len(bucket)),
        }
        bucket = []
        bucket_end += span

        close_series.append(candle["close"])
        weight_series.append(candle["count"])
        if len(close_series) > cfg.VWAP_WINDOW_BUCKETS * 3:
            close_series.pop(0)
            weight_series.pop(0)

        if tick.epoch - last_entry_epoch < cfg.COOLDOWN_SEC:
            continue
        if len(close_series) < cfg.VWAP_WINDOW_BUCKETS:
            continue

        window_closes = close_series[-cfg.VWAP_WINDOW_BUCKETS :]
        window_weights = weight_series[-cfg.VWAP_WINDOW_BUCKETS :]
        total_w = sum(window_weights)
        if total_w <= 0.0:
            continue
        vwap = sum(c * w for c, w in zip(window_closes, window_weights)) / total_w
        mean_val = sum(window_closes) / cfg.VWAP_WINDOW_BUCKETS
        var = sum((c - mean_val) ** 2 for c in window_closes) / max(
            cfg.VWAP_WINDOW_BUCKETS - 1, 1
        )
        sigma = var ** 0.5 if var > 0.0 else 0.0
        if sigma == 0.0:
            continue
        latest_close = window_closes[-1]
        z_dev = (latest_close - vwap) / sigma
        atr = _atr_from_closes(
            close_series, max(1, cfg.VWAP_WINDOW_BUCKETS // 2), cfg.PIP_VALUE
        )
        if atr < cfg.MIN_ATR_PIPS:
            continue
        rsi = _rsi(close_series[-cfg.VWAP_WINDOW_BUCKETS :], cfg.RSI_PERIOD)
        if rsi is None:
            continue

        prev_vwap = window_closes[-2]
        slope = latest_close - prev_vwap
        side: Optional[str] = None
        if (
            z_dev >= cfg.Z_ENTRY_SIGMA
            and slope <= 0.0
            and cfg.RSI_SHORT_RANGE[0] <= rsi <= cfg.RSI_SHORT_RANGE[1]
        ):
            side = "short"
        elif (
            z_dev <= -cfg.Z_ENTRY_SIGMA
            and slope >= 0.0
            and cfg.RSI_LONG_RANGE[0] <= rsi <= cfg.RSI_LONG_RANGE[1]
        ):
            side = "long"
        if side is None:
            continue

        units = _stage_units(side)
        if units is None:
            continue

        entry_price = tick.ask if side == "long" else tick.bid
        tp_price = (
            entry_price + cfg.TP_PIPS * cfg.PIP_VALUE
            if side == "long"
            else entry_price - cfg.TP_PIPS * cfg.PIP_VALUE
        )
        sl_pips = max(cfg.SL_MIN_PIPS, atr * cfg.SL_ATR_MULT)
        sl_price = (
            entry_price - sl_pips * cfg.PIP_VALUE
            if side == "long"
            else entry_price + sl_pips * cfg.PIP_VALUE
        )
        stage_index = (
            sum(1 for pos in open_positions if pos["side"] == side) + 1
        )
        open_positions.append(
            {
                "side": side,
                "entry_tick": tick,
                "entry_price": entry_price,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "units": units,
                "stage_index": stage_index,
            }
        )

    if open_positions:
        exit_tick = ticks[-1]
        for pos in open_positions:
            side = pos["side"]
            entry_price = pos["entry_price"]
            pnl = (
                (exit_tick.mid - entry_price) / cfg.PIP_VALUE
                if side == "long"
                else (entry_price - exit_tick.mid) / cfg.PIP_VALUE
            )
            trades.append(
                {
                    "direction": side,
                    "entry_time": pos["entry_tick"].dt.isoformat(),
                    "exit_time": exit_tick.dt.isoformat(),
                    "entry_price": round(entry_price, 5),
                    "exit_price": round(exit_tick.mid, 5),
                    "tp_price": round(pos["tp_price"], 5),
                    "sl_price": round(pos["sl_price"], 5),
                    "pnl_pips": round(pnl, 3),
                    "reason": "eod",
                    "units": pos["units"],
                    "stage_index": pos["stage_index"],
                }
            )

    total_pnl = sum(t["pnl_pips"] for t in trades)
    wins = sum(1 for t in trades if t["pnl_pips"] > 0)
    summary = {
        "trades": len(trades),
        "total_pnl_pips": round(total_pnl, 3),
        "total_pnl_jpy": round(total_pnl * 100, 3),
        "win_rate": round(wins / len(trades), 4) if trades else 0.0,
    }
    return {"summary": summary, "trades": trades}


# ---------------------------------------------------------------------------
# Squeeze break S5 replay (simplified)
# ---------------------------------------------------------------------------


def replay_squeeze_break_s5(ticks: List[Tick]) -> Dict[str, object]:
    cfg = __import__("workers.squeeze_break_s5.config", fromlist=["config"])

    span = cfg.BUCKET_SECONDS
    bucket: List[Dict[str, float]] = []
    bucket_end = ticks[0].epoch - (ticks[0].epoch % span) + span
    candles: List[Dict[str, float]] = []
    trades: List[Dict[str, object]] = []
    last_entry_epoch = ticks[0].epoch - cfg.COOLDOWN_SEC
    open_position: Optional[Dict[str, object]] = None

    for tick in ticks:
        bucket.append({"epoch": tick.epoch, "mid": tick.mid})

        if open_position:
            side = open_position["side"]
            entry_price = open_position["entry_price"]
            if side == "long":
                stop_hit = tick.bid <= open_position["sl_price"]
                tp_hit = tick.bid >= open_position["tp_price"]
            else:
                stop_hit = tick.ask >= open_position["sl_price"]
                tp_hit = tick.ask <= open_position["tp_price"]
            timeout = tick.epoch - open_position["entry_tick"].epoch >= cfg.TIMEOUT_SEC
            if stop_hit or tp_hit or timeout:
                pnl = (
                    (tick.mid - entry_price) / cfg.PIP_VALUE
                    if side == "long"
                    else (entry_price - tick.mid) / cfg.PIP_VALUE
                )
                trades.append(
                    {
                        "direction": side,
                        "entry_time": open_position["entry_tick"].dt.isoformat(),
                        "exit_time": tick.dt.isoformat(),
                        "entry_price": round(entry_price, 5),
                        "exit_price": round(tick.mid, 5),
                        "tp_price": round(open_position["tp_price"], 5),
                        "sl_price": round(open_position["sl_price"], 5),
                        "pnl_pips": round(pnl, 3),
                        "reason": "tp" if tp_hit else ("sl" if stop_hit else "timeout"),
                    }
                )
                open_position = None
                last_entry_epoch = tick.epoch

        if tick.epoch >= bucket_end:
            candle = {
                "close": bucket[-1]["mid"],
                "high": max(b["mid"] for b in bucket),
                "low": min(b["mid"] for b in bucket),
            }
            candles.append(candle)
            if len(candles) > cfg.SLOW_BUCKETS * 3:
                candles.pop(0)
            bucket = []
            bucket_end += span

        if open_position is not None:
            continue
        if tick.epoch - last_entry_epoch < cfg.COOLDOWN_SEC:
            continue
        if len(candles) < cfg.MIN_BUCKETS:
            continue

        closes = [c["close"] for c in candles]
        atr = _atr_from_closes(closes, max(4, cfg.FAST_BUCKETS // 2), cfg.PIP_VALUE)
        if atr < cfg.MIN_ATR_PIPS:
            continue
        rsi = _rsi(closes[-cfg.FAST_BUCKETS :], cfg.RSI_PERIOD)
        if rsi is None:
            continue

        if len(candles) <= cfg.SLOW_BUCKETS:
            continue
        prior_slice = candles[-(cfg.SLOW_BUCKETS + 1) : -1]
        if not prior_slice:
            continue
        recent_high = max(c["high"] for c in prior_slice)
        recent_low = min(c["low"] for c in prior_slice)

        latest_close = candles[-1]["close"]
        buffer = max(cfg.BREAK_BUFFER_PIPS * cfg.PIP_VALUE, 0.05 * cfg.PIP_VALUE)
        direction: Optional[str] = None
        if latest_close > recent_high + buffer and rsi <= cfg.RSI_LONG_MAX:
            direction = "long"
        elif latest_close < recent_low - buffer and rsi >= cfg.RSI_SHORT_MIN:
            direction = "short"
        if direction is None:
            continue

        entry_price = tick.ask if direction == "long" else tick.bid
        tp_pips = max(cfg.TP_MIN_PIPS, atr * cfg.TP_ATR_MULT)
        sl_pips = max(cfg.SL_MIN_PIPS, atr * cfg.SL_ATR_MULT)
        tp_price = (
            entry_price + tp_pips * cfg.PIP_VALUE
            if direction == "long"
            else entry_price - tp_pips * cfg.PIP_VALUE
        )
        sl_price = (
            entry_price - sl_pips * cfg.PIP_VALUE
            if direction == "long"
            else entry_price + sl_pips * cfg.PIP_VALUE
        )
        open_position = {
            "side": direction,
            "entry_tick": tick,
            "entry_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }
        last_entry_epoch = tick.epoch

    total_pnl = sum(t["pnl_pips"] for t in trades)
    wins = sum(1 for t in trades if t["pnl_pips"] > 0)
    summary = {
        "trades": len(trades),
        "total_pnl_pips": round(total_pnl, 3),
        "total_pnl_jpy": round(total_pnl * 100, 3),
        "win_rate": round(wins / len(trades), 4) if trades else 0.0,
    }
    return {"summary": summary, "trades": trades}


# ---------------------------------------------------------------------------
# Impulse break S5 replay (simplified)
# ---------------------------------------------------------------------------


def _replay_impulse_style(
    ticks: List[Tick],
    cfg_module: str,
    tag: str,
) -> Dict[str, object]:
    cfg = __import__(cfg_module, fromlist=["config"])

    bucket_span = cfg.BUCKET_SECONDS
    trades: List[Dict[str, object]] = []
    candles: List[Dict[str, float]] = []
    open_trade: Optional[Dict[str, object]] = None
    managed: Dict[str, Dict[str, float]] = {}

    bucket: List[Dict[str, float]] = []
    bucket_end = ticks[0].epoch - (ticks[0].epoch % bucket_span) + bucket_span
    last_entry_epoch = ticks[0].epoch - cfg.COOLDOWN_SEC

    def z_score(values: Sequence[float], window: int) -> Optional[float]:
        if len(values) < window or window <= 0:
            return None
        sample = values[-window:]
        mean_val = sum(sample) / len(sample)
        var = sum((v - mean_val) ** 2 for v in sample) / max(len(sample) - 1, 1)
        if var <= 0.0:
            return 0.0
        return (sample[-1] - mean_val) / (var ** 0.5)

    def atr_from_closes(values: Sequence[float], period: int) -> float:
        if len(values) <= 1:
            return 0.0
        true_ranges = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        if not true_ranges:
            return 0.0
        period = max(1, min(period, len(true_ranges)))
        return sum(true_ranges[-period:]) / period / cfg.PIP_VALUE

    def flush_bucket() -> None:
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
        candles.append(candle)
        while len(candles) > cfg.SLOW_BUCKETS * 3:
            candles.pop(0)

    for tick in ticks:
        if tick.epoch >= bucket_end:
            flush_bucket()
            bucket_end += bucket_span
        bucket.append({"epoch": tick.epoch, "mid": tick.mid})

        if open_trade:
            trade_id = open_trade["id"]
            side = open_trade["side"]
            entry_price = open_trade["entry_price"]
            profit_pips = (
                (tick.mid - entry_price) / cfg.PIP_VALUE
                if side == "long"
                else (entry_price - tick.mid) / cfg.PIP_VALUE
            )
            state = managed.setdefault(
                trade_id, {"sl": open_trade["sl_price"], "be": False}
            )
            if not state["be"] and profit_pips >= cfg.BE_TRIGGER_PIPS:
                if side == "long":
                    state["sl"] = entry_price + cfg.BE_OFFSET_PIPS * cfg.PIP_VALUE
                else:
                    state["sl"] = entry_price - cfg.BE_OFFSET_PIPS * cfg.PIP_VALUE
                state["be"] = True
            if profit_pips >= cfg.TRAIL_TRIGGER_PIPS:
                if side == "long":
                    desired = tick.mid - cfg.TRAIL_BACKOFF_PIPS * cfg.PIP_VALUE
                    if desired > state["sl"] + cfg.TRAIL_STEP_PIPS * cfg.PIP_VALUE:
                        state["sl"] = desired
                else:
                    desired = tick.mid + cfg.TRAIL_BACKOFF_PIPS * cfg.PIP_VALUE
                    if desired < state["sl"] - cfg.TRAIL_STEP_PIPS * cfg.PIP_VALUE:
                        state["sl"] = desired
            sl_price = state["sl"]
            if side == "long":
                stop_hit = tick.bid <= sl_price
                tp_hit = tick.bid >= open_trade["tp_price"]
            else:
                stop_hit = tick.ask >= sl_price
                tp_hit = tick.ask <= open_trade["tp_price"]
            if stop_hit or tp_hit:
                exit_price = tick.mid
                pnl = (exit_price - entry_price) / cfg.PIP_VALUE
                if side == "short":
                    pnl = -pnl
                trades.append(
                    {
                        "direction": side,
                        "entry_time": open_trade["entry_tick"].dt.isoformat(),
                        "exit_time": tick.dt.isoformat(),
                        "entry_price": round(entry_price, 5),
                        "exit_price": round(exit_price, 5),
                        "tp_price": round(open_trade["tp_price"], 5),
                        "sl_price": round(sl_price, 5),
                        "pnl_pips": round(pnl, 3),
                        "reason": "tp" if tp_hit else "sl",
                    }
                )
                managed.pop(trade_id, None)
                open_trade = None
                last_entry_epoch = tick.epoch

        if open_trade is None and tick.epoch - last_entry_epoch >= cfg.COOLDOWN_SEC:
            if len(candles) < cfg.MIN_BUCKETS:
                continue
            closes = [c["close"] for c in candles]
            fast_z = z_score(closes, cfg.FAST_BUCKETS)
            slow_z = z_score(closes, cfg.SLOW_BUCKETS)
            atr = atr_from_closes(closes, max(4, cfg.FAST_BUCKETS // 2))
            if fast_z is None or slow_z is None or atr < cfg.MIN_ATR_PIPS:
                continue
            latest = candles[-1]
            prev_close = candles[-2]["close"] if len(candles) >= 2 else latest["close"]
            recent_high = max(c["high"] for c in candles[-(cfg.FAST_BUCKETS + 6) : -1])
            recent_low = min(c["low"] for c in candles[-(cfg.FAST_BUCKETS + 6) : -1])
            rsi = _rsi(closes[-cfg.FAST_BUCKETS :], cfg.RSI_PERIOD)
            breakout_gap = cfg.MIN_BREAKOUT_PIPS * cfg.PIP_VALUE
            retrace_gap = cfg.MIN_RETRACE_GAP_PIPS * cfg.PIP_VALUE
            direction: Optional[str] = None
            allow_long = getattr(cfg, "ALLOW_LONG", True)
            allow_short = getattr(cfg, "ALLOW_SHORT", True)
            if (
                allow_long
                and latest["close"] >= recent_high + breakout_gap
                and fast_z >= cfg.FAST_Z_LONG_MIN
                and slow_z >= cfg.SLOW_Z_LONG_MIN
                and (rsi is None or rsi <= cfg.RSI_LONG_MAX)
                and latest["close"] - prev_close >= retrace_gap
            ):
                direction = "long"
            elif (
                allow_short
                and latest["close"] <= recent_low - breakout_gap
                and fast_z <= cfg.FAST_Z_SHORT_MAX
                and slow_z <= cfg.SLOW_Z_SHORT_MAX
                and (rsi is None or rsi >= cfg.RSI_SHORT_MIN)
                and prev_close - latest["close"] >= retrace_gap
            ):
                direction = "short"
            if direction is None:
                continue
            atr_tp = max(cfg.TP_ATR_MIN_PIPS, atr * cfg.TP_ATR_MULT)
            atr_tp = min(atr_tp, cfg.TP_ATR_MAX_PIPS)
            atr_sl = max(cfg.SL_ATR_MIN_PIPS, atr * cfg.SL_ATR_MULT)
            entry_price = tick.ask if direction == "long" else tick.bid
            tp_price = (
                entry_price + atr_tp * cfg.PIP_VALUE
                if direction == "long"
                else entry_price - atr_tp * cfg.PIP_VALUE
            )
            sl_price = (
                entry_price - atr_sl * cfg.PIP_VALUE
                if direction == "long"
                else entry_price + atr_sl * cfg.PIP_VALUE
            )
            trade_id = f"{tag}-{len(trades)}-{int(tick.epoch)}"
            open_trade = {
                "id": trade_id,
                "side": direction,
                "entry_tick": tick,
                "entry_price": entry_price,
                "tp_price": tp_price,
                "sl_price": sl_price,
            }
            managed[trade_id] = {"sl": sl_price, "be": False}

    flush_bucket()

    if open_trade:
        exit_tick = ticks[-1]
        entry_price = open_trade["entry_price"]
        pnl = (exit_tick.mid - entry_price) / cfg.PIP_VALUE
        if open_trade["side"] == "short":
            pnl = -pnl
        sl_final = managed.get(open_trade["id"], {}).get("sl", open_trade["sl_price"])
        trades.append(
            {
                "direction": open_trade["side"],
                "entry_time": open_trade["entry_tick"].dt.isoformat(),
                "exit_time": exit_tick.dt.isoformat(),
                "entry_price": round(entry_price, 5),
                "exit_price": round(exit_tick.mid, 5),
                "tp_price": round(open_trade["tp_price"], 5),
                "sl_price": round(sl_final, 5),
                "pnl_pips": round(pnl, 3),
                "reason": "eod",
            }
        )

    total_pnl = sum(t["pnl_pips"] for t in trades)
    wins = sum(1 for t in trades if t["pnl_pips"] > 0)
    summary = {
        "trades": len(trades),
        "total_pnl_pips": round(total_pnl, 3),
        "total_pnl_jpy": round(total_pnl * 100, 3),
        "win_rate": round(wins / len(trades), 4) if trades else 0.0,
    }
    return {"summary": summary, "trades": trades}


def replay_impulse_break_s5(ticks: List[Tick]) -> Dict[str, object]:
    return _replay_impulse_style(
        ticks,
        "workers.impulse_break_s5.config",
        "impulse",
    )


def replay_impulse_momentum_s5(ticks: List[Tick]) -> Dict[str, object]:
    return _replay_impulse_style(
        ticks,
        "workers.impulse_momentum_s5.config",
        "impulse-momo",
    )


# ---------------------------------------------------------------------------
# Impulse retest S5 replay (simplified)
# ---------------------------------------------------------------------------


def replay_impulse_retest_s5(ticks: List[Tick]) -> Dict[str, object]:
    cfg = __import__("workers.impulse_retest_s5.config", fromlist=["config"])

    span = cfg.BUCKET_SECONDS
    bucket: List[Dict[str, float]] = []
    bucket_end = ticks[0].epoch - (ticks[0].epoch % span) + span
    candles: List[Dict[str, float]] = []
    trades: List[Dict[str, object]] = []
    last_entry_epoch = ticks[0].epoch - cfg.COOLDOWN_SEC
    candidate: Optional[Dict[str, float]] = None
    open_trade: Optional[Dict[str, object]] = None

    for tick in ticks:
        bucket.append({"epoch": tick.epoch, "mid": tick.mid})

        if tick.epoch >= bucket_end:
            candle = {
                "close": bucket[-1]["mid"],
                "high": max(b["mid"] for b in bucket),
                "low": min(b["mid"] for b in bucket),
                "epoch": tick.epoch,
            }
            candles.append(candle)
            if len(candles) > cfg.IMPULSE_LOOKBACK * 3:
                candles.pop(0)
            bucket = []
            bucket_end += span

        if open_trade:
            side = open_trade["side"]
            entry_price = open_trade["entry_price"]
            if side == "long":
                stop_hit = tick.bid <= open_trade["sl_price"]
                tp_hit = tick.bid >= open_trade["tp_price"]
            else:
                stop_hit = tick.ask >= open_trade["sl_price"]
                tp_hit = tick.ask <= open_trade["tp_price"]
            timeout = tick.epoch - open_trade["entry_tick"].epoch >= cfg.TIMEOUT_SEC
            if stop_hit or tp_hit or timeout:
                pnl = (
                    (tick.mid - entry_price) / cfg.PIP_VALUE
                    if side == "long"
                    else (entry_price - tick.mid) / cfg.PIP_VALUE
                )
                trades.append(
                    {
                        "direction": side,
                        "entry_time": open_trade["entry_tick"].dt.isoformat(),
                        "exit_time": tick.dt.isoformat(),
                        "entry_price": round(entry_price, 5),
                        "exit_price": round(tick.mid, 5),
                        "tp_price": round(open_trade["tp_price"], 5),
                        "sl_price": round(open_trade["sl_price"], 5),
                        "pnl_pips": round(pnl, 3),
                        "reason": "tp" if tp_hit else ("sl" if stop_hit else "timeout"),
                    }
                )
                open_trade = None
                last_entry_epoch = tick.epoch
                candidate = None

        if open_trade is not None:
            continue
        if tick.epoch - last_entry_epoch < cfg.COOLDOWN_SEC:
            continue
        if len(candles) < cfg.MIN_BUCKETS:
            continue

        closes = [c["close"] for c in candles]
        atr = _atr_from_closes(closes, max(4, cfg.IMPULSE_LOOKBACK // 4), cfg.PIP_VALUE)
        if atr < cfg.MIN_ATR_PIPS:
            continue
        rsi = _rsi(closes[-cfg.IMPULSE_LOOKBACK :], cfg.RSI_PERIOD)
        if rsi is None:
            continue

        lookback = candles[-cfg.IMPULSE_LOOKBACK :]
        high_val = max(c["high"] for c in lookback)
        low_val = min(c["low"] for c in lookback)
        impulse_pips = (high_val - low_val) / cfg.PIP_VALUE

        if candidate and candles[-1]["epoch"] > candidate.get("expire", 0.0):
            candidate = None

        latest_close = candles[-1]["close"]
        if candidate is None and impulse_pips >= cfg.IMPULSE_MIN_PIPS:
            if latest_close >= high_val - 0.2 * cfg.PIP_VALUE:
                candidate = {
                    "dir": "long",
                    "start": low_val,
                    "end": high_val,
                    "expire": candles[-1]["epoch"] + cfg.RETEST_MAX_SEC,
                }
            elif latest_close <= low_val + 0.2 * cfg.PIP_VALUE:
                candidate = {
                    "dir": "short",
                    "start": high_val,
                    "end": low_val,
                    "expire": candles[-1]["epoch"] + cfg.RETEST_MAX_SEC,
                }
            continue

        if candidate is None:
            continue

        direction = candidate["dir"]
        start = candidate["start"]
        end = candidate["end"]
        fib_low = start + (end - start) * cfg.FIB_LOWER
        fib_high = start + (end - start) * cfg.FIB_UPPER

        in_zone = False
        if direction == "long" and fib_low <= latest_close <= fib_high and rsi <= cfg.RSI_LONG_MAX:
            in_zone = True
        elif direction == "short" and fib_high <= latest_close <= fib_low and rsi >= cfg.RSI_SHORT_MIN:
            in_zone = True
        if not in_zone:
            continue

        impulse_range = abs(end - start)
        zone_mid = (fib_low + fib_high) / 2.0
        prev_close = candles[-2]["close"] if len(candles) >= 2 else latest_close
        if direction == "long":
            if tick.mid < zone_mid:
                continue
            if latest_close < prev_close:
                continue
        else:
            if tick.mid > zone_mid:
                continue
            if latest_close > prev_close:
                continue

        entry_price = tick.ask if direction == "long" else tick.bid
        tp_price = (
            entry_price + impulse_range * cfg.TP_RATIO
            if direction == "long"
            else entry_price - impulse_range * cfg.TP_RATIO
        )
        sl_price = (
            fib_low - cfg.SL_BUFFER_PIPS * cfg.PIP_VALUE
            if direction == "long"
            else fib_low + cfg.SL_BUFFER_PIPS * cfg.PIP_VALUE
        )

        open_trade = {
            "side": direction,
            "entry_tick": tick,
            "entry_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }
        last_entry_epoch = tick.epoch
        candidate = None

    if open_trade:
        exit_tick = ticks[-1]
        entry_price = open_trade["entry_price"]
        pnl = (
            (exit_tick.mid - entry_price) / cfg.PIP_VALUE
            if open_trade["side"] == "long"
            else (entry_price - exit_tick.mid) / cfg.PIP_VALUE
        )
        trades.append(
            {
                "direction": open_trade["side"],
                "entry_time": open_trade["entry_tick"].dt.isoformat(),
                "exit_time": exit_tick.dt.isoformat(),
                "entry_price": round(entry_price, 5),
                "exit_price": round(exit_tick.mid, 5),
                "tp_price": round(open_trade["tp_price"], 5),
                "sl_price": round(open_trade["sl_price"], 5),
                "pnl_pips": round(pnl, 3),
                "reason": "eod",
            }
        )

    total_pnl = sum(t["pnl_pips"] for t in trades)
    wins = sum(1 for t in trades if t["pnl_pips"] > 0)
    summary = {
        "trades": len(trades),
        "total_pnl_pips": round(total_pnl, 3),
        "total_pnl_jpy": round(total_pnl * 100, 3),
        "win_rate": round(wins / len(trades), 4) if trades else 0.0,
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


def replay_mirror_spike_tight(ticks: List[Tick]) -> Dict[str, object]:
    cfg = __import__("workers.mirror_spike_tight.config", fromlist=["config"])
    trades: List[Dict[str, object]] = []
    open_trade: Optional[Dict[str, object]] = None
    window: Deque[Tick] = deque(maxlen=max(cfg.MIRROR_LOOKBACK_BUCKETS * 5, 20))
    cooldown_end = ticks[0].epoch - cfg.COOLDOWN_SEC
    timeout_sec = 240.0
    per_day_counts: Dict[str, int] = {}

    for tick in ticks:
        if cfg.ALLOWED_HOURS_UTC and tick.dt.hour not in cfg.ALLOWED_HOURS_UTC:
            continue
        window.append(tick)

        if open_trade:
            side = open_trade["side"]
            entry_tick = open_trade["entry_tick"]
            if side == "long":
                if tick.bid <= open_trade["sl_price"]:
                    reason = "sl"
                elif tick.bid >= open_trade["tp_price"]:
                    reason = "tp"
                else:
                    reason = ""
            else:
                if tick.ask >= open_trade["sl_price"]:
                    reason = "sl"
                elif tick.ask <= open_trade["tp_price"]:
                    reason = "tp"
                else:
                    reason = ""
            if not reason and tick.epoch - entry_tick.epoch >= timeout_sec:
                reason = "timeout"
            if reason:
                entry_mid = entry_tick.mid
                exit_mid = tick.mid
                pnl = (exit_mid - entry_mid) / cfg.PIP_VALUE
                if side == "short":
                    pnl = -pnl
                trades.append(
                    {
                        "direction": side,
                        "entry_time": entry_tick.dt.isoformat(),
                        "exit_time": tick.dt.isoformat(),
                        "entry_price": round(entry_mid, 5),
                        "exit_price": round(exit_mid, 5),
                        "tp_price": round(open_trade["tp_price"], 5),
                        "sl_price": round(open_trade["sl_price"], 5),
                        "pnl_pips": round(pnl, 3),
                        "reason": reason,
                    }
                )
                open_trade = None
                cooldown_end = tick.epoch
            continue

        day_key = tick.dt.date().isoformat()
        if cfg.MAX_TRADES_PER_DAY > 0 and per_day_counts.get(day_key, 0) >= cfg.MAX_TRADES_PER_DAY:
            continue

        if tick.epoch - cooldown_end < cfg.COOLDOWN_SEC:
            continue
        if len(window) < cfg.MIRROR_LOOKBACK_BUCKETS:
            continue

        highs = max(x.mid for x in window)
        lows = min(x.mid for x in window)
        range_pips = (highs - lows) / cfg.PIP_VALUE
        if range_pips < cfg.SPIKE_THRESHOLD_PIPS:
            continue

        upper_trigger = highs - cfg.CONFIRM_RANGE_PIPS * cfg.PIP_VALUE
        lower_trigger = lows + cfg.CONFIRM_RANGE_PIPS * cfg.PIP_VALUE
        side: Optional[str] = None
        if tick.mid >= upper_trigger:
            side = "short"
        elif tick.mid <= lower_trigger:
            side = "long"
        if side is None:
            continue

        slope = (window[-1].mid - window[0].mid) / cfg.PIP_VALUE
        if side == "long" and slope < cfg.TREND_SLOPE_MIN_PIPS:
            continue
        if side == "short" and -slope < cfg.TREND_SLOPE_MIN_PIPS:
            continue

        entry_price = tick.ask if side == "long" else tick.bid
        tp_price = (
            entry_price + cfg.TP_PIPS * cfg.PIP_VALUE
            if side == "long"
            else entry_price - cfg.TP_PIPS * cfg.PIP_VALUE
        )
        sl_price = (
            entry_price - cfg.SL_PIPS * cfg.PIP_VALUE
            if side == "long"
            else entry_price + cfg.SL_PIPS * cfg.PIP_VALUE
        )
        open_trade = {
            "side": side,
            "entry_tick": tick,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }
        per_day_counts[day_key] = per_day_counts.get(day_key, 0) + 1

    if open_trade:
        exit_tick = ticks[-1]
        entry_mid = open_trade["entry_tick"].mid
        pnl = (exit_tick.mid - entry_mid) / cfg.PIP_VALUE
        if open_trade["side"] == "short":
            pnl = -pnl
        trades.append(
            {
                "direction": open_trade["side"],
                "entry_time": open_trade["entry_tick"].dt.isoformat(),
                "exit_time": exit_tick.dt.isoformat(),
                "entry_price": round(entry_mid, 5),
                "exit_price": round(exit_tick.mid, 5),
                "tp_price": round(open_trade["tp_price"], 5),
                "sl_price": round(open_trade["sl_price"], 5),
                "pnl_pips": round(pnl, 3),
                "reason": "eod",
            }
        )

    total_pnl = sum(t["pnl_pips"] for t in trades)
    wins = sum(1 for t in trades if t["pnl_pips"] > 0)
    summary = {
        "trades": len(trades),
        "total_pnl_pips": round(total_pnl, 3),
        "total_pnl_jpy": round(total_pnl * 100, 3),
        "win_rate": round(wins / len(trades), 4) if trades else 0.0,
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
            "vwap_magnet_s5",
            "impulse_break_s5",
            "impulse_momentum_s5",
            "squeeze_break_s5",
            "impulse_retest_s5",
            "mirror_spike_s5",
            "mirror_spike_tight",
            "pullback_scalp",
            "mirror_spike",
        ],
    )
    ap.add_argument("--ticks", help="Tick JSONL file (timestamp,bid,ask).")
    ap.add_argument("--candles", help="S5 candle JSON to generate synthetic ticks.")
    ap.add_argument("--candles-sim", help="S5 candles JSON to synthesise pseudo ticks.")
    ap.add_argument("--sim-out", help="Optional path to write generated pseudo ticks JSONL.")
    ap.add_argument("--sim-seed", type=int, help="Random seed for pseudo tick generator.")
    ap.add_argument("--sim-tpm-london", type=float, help="Target ticks/5s for London session.")
    ap.add_argument("--sim-tpm-ny", type=float, help="Target ticks/5s for New York session.")
    ap.add_argument("--sim-tpm-asia", type=float, help="Target ticks/5s for Asia session.")
    ap.add_argument("--sim-atr-high", type=float, help="ATR multiplier (high volatility).")
    ap.add_argument("--sim-atr-low", type=float, help="ATR multiplier (low volatility).")
    ap.add_argument("--sim-target-min", type=int, help="Target minimum ticks per window.")
    ap.add_argument("--sim-target-window", type=int, help="Window seconds for target density.")
    ap.add_argument("--sim-target-coverage", type=float, help="Required coverage ratio (0-1).")
    ap.add_argument("--sim-stall-prob", type=float, help="Probability to insert consolidation block.")
    ap.add_argument("--sim-stall-range", type=float, help="Consolidation range in pips.")
    ap.add_argument("--sim-stall-ticks", type=int, help="Number of consolidation ticks.")
    ap.add_argument("--sim-impulse-prob", type=float, help="Probability to insert impulse block.")
    ap.add_argument("--sim-impulse-atr-k", type=float, help="Impulse ATR multiplier.")
    ap.add_argument("--sim-impulse-ticks", type=int, help="Impulse tick length.")
    ap.add_argument("--sim-noise-sigma", type=float, help="Gaussian noise sigma in pips.")
    ap.add_argument("--sim-spread-london", type=float, help="Mean spread in pips for London.")
    ap.add_argument("--sim-spread-ny", type=float, help="Mean spread in pips for New York.")
    ap.add_argument("--sim-spread-asia", type=float, help="Mean spread in pips for Asia.")
    ap.add_argument("--sim-spread-std", type=float, help="Spread standard deviation in pips.")
    ap.add_argument("--sim-spread-night-mul", type=float, help="Night spread multiplier for Asia.")
    ap.add_argument("--out", default="", help="Optional output JSON path.")
    ap.add_argument(
        "--exit-scheme",
        choices=sorted(FAST_SCALP_EXIT_SCHEMES.keys()),
        help="FastScalp exit scheme (A/B/C).",
    )
    ap.add_argument("--min-impulse-pips", type=float, help="Override minimum impulse size in pips.")
    ap.add_argument("--impulse-window-sec", type=float, help="Impulse lookback window seconds.")
    ap.add_argument(
        "--require-consolidation",
        action="store_true",
        dest="require_consolidation",
        help="Force consolidation requirement for entries.",
    )
    ap.add_argument(
        "--no-require-consolidation",
        action="store_false",
        dest="require_consolidation",
        help="Allow entries without consolidation.",
    )
    ap.set_defaults(require_consolidation=None)
    ap.add_argument("--consolidation-sec", type=float, help="Consolidation window seconds.")
    ap.add_argument("--consolidation-range-pips", type=float, help="Maximum consolidation range in pips.")
    ap.add_argument("--spread-max-pips", type=float, help="Maximum spread in pips for entries.")
    ap.add_argument("--tickrate-min", type=float, help="Minimum tick count within tickrate window.")
    ap.add_argument("--tickrate-window-sec", type=float, help="Tickrate measurement window seconds.")
    ap.add_argument("--min-atr-pips", type=float, help="Override minimum ATR in pips for entries.")
    ap.add_argument("--gate-min-atr-pips", type=float, help="Quality gate ATR floor in pips.")
    ap.add_argument("--news-block-min", type=float, help="News blackout window in minutes (± window).")
    ap.add_argument("--cooldown-sec", type=float, help="Entry cooldown seconds per direction.")
    ap.add_argument("--hold-ms", type=float, help="Minimum hold duration in milliseconds.")
    ap.add_argument("--sessions", type=str, help="Comma separated session tags to allow (asia,london,newyork).")
    args = ap.parse_args()

    if not args.ticks and not args.candles and not args.candles_sim:
        raise SystemExit("--ticks / --candles / --candles-sim のいずれかを指定してください")

    sim_meta: Optional[Dict[str, Any]] = None

    if args.candles_sim:
        density_kwargs: Dict[str, Any] = {}
        if args.sim_tpm_london is not None:
            density_kwargs["tpm_5s_london"] = args.sim_tpm_london
        if args.sim_tpm_ny is not None:
            density_kwargs["tpm_5s_ny"] = args.sim_tpm_ny
        if args.sim_tpm_asia is not None:
            density_kwargs["tpm_5s_asia"] = args.sim_tpm_asia
        if args.sim_atr_high is not None:
            density_kwargs["atr_k_high"] = args.sim_atr_high
        if args.sim_atr_low is not None:
            density_kwargs["atr_k_low"] = args.sim_atr_low
        if args.sim_target_min is not None:
            density_kwargs["target_tickrate_min"] = args.sim_target_min
        if args.sim_target_window is not None:
            density_kwargs["target_window_sec"] = args.sim_target_window
        if args.sim_target_coverage is not None:
            density_kwargs["target_coverage"] = args.sim_target_coverage

        shape_kwargs: Dict[str, Any] = {}
        if args.sim_stall_prob is not None:
            shape_kwargs["stall_prob"] = args.sim_stall_prob
        if args.sim_stall_range is not None:
            shape_kwargs["stall_range_pips"] = args.sim_stall_range
        if args.sim_stall_ticks is not None:
            shape_kwargs["stall_ticks"] = args.sim_stall_ticks
        if args.sim_impulse_prob is not None:
            shape_kwargs["impulse_prob"] = args.sim_impulse_prob
        if args.sim_impulse_atr_k is not None:
            shape_kwargs["impulse_atr_k"] = args.sim_impulse_atr_k
        if args.sim_impulse_ticks is not None:
            shape_kwargs["impulse_ticks"] = args.sim_impulse_ticks
        if args.sim_noise_sigma is not None:
            shape_kwargs["noise_pips_sigma"] = args.sim_noise_sigma

        spread_kwargs: Dict[str, Any] = {}
        if args.sim_spread_london is not None:
            spread_kwargs["mean_pips_london"] = args.sim_spread_london
        if args.sim_spread_ny is not None:
            spread_kwargs["mean_pips_ny"] = args.sim_spread_ny
        if args.sim_spread_asia is not None:
            spread_kwargs["mean_pips_asia"] = args.sim_spread_asia
        if args.sim_spread_std is not None:
            spread_kwargs["std_pips"] = args.sim_spread_std
        if args.sim_spread_night_mul is not None:
            spread_kwargs["night_multiplier"] = args.sim_spread_night_mul

        density_cfg = DensityCfg(**density_kwargs)
        if (
            args.sim_target_window is not None
            and args.sim_target_min is not None
        ):
            density_cfg.tickrate_checks = (
                (5, density_cfg.target_tickrate_min),
                (args.sim_target_window, args.sim_target_min),
            )

        sim_cfg = SimCfg(
            density=density_cfg,
            shape=ShapeCfg(**shape_kwargs),
            spread=SpreadCfg(**spread_kwargs),
            random_seed=args.sim_seed if args.sim_seed is not None else SimCfg().random_seed,
        )

        sim_out = Path(args.sim_out) if args.sim_out else Path("tmp") / (
            f"sim_{Path(args.candles_sim).stem}_{int(time.time())}.jsonl"
        )
        sim_out.parent.mkdir(parents=True, exist_ok=True)
        sim_path, density_info = synth_from_candles(args.candles_sim, str(sim_out), sim_cfg)
        ticks = load_ticks(sim_path)
        sim_meta = {
            "ticks_path": str(sim_path),
            "density": density_info,
            "config": {
                "density": asdict(sim_cfg.density),
                "shape": asdict(sim_cfg.shape),
                "spread": asdict(sim_cfg.spread),
                "seed": sim_cfg.random_seed,
            },
        }
    elif args.ticks:
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

    overrides: Dict[str, Any] = {}
    if args.min_impulse_pips is not None:
        overrides["min_impulse_pips"] = args.min_impulse_pips
    if args.impulse_window_sec is not None:
        overrides["impulse_window_sec"] = args.impulse_window_sec
    if args.require_consolidation is not None:
        overrides["require_consolidation"] = args.require_consolidation
    if args.consolidation_sec is not None:
        overrides["consolidation_window_sec"] = args.consolidation_sec
    if args.consolidation_range_pips is not None:
        overrides["consolidation_range_pips"] = args.consolidation_range_pips
    if args.spread_max_pips is not None:
        overrides["max_spread_pips"] = args.spread_max_pips
    if args.tickrate_min is not None:
        overrides["tickrate_min"] = args.tickrate_min
    if args.tickrate_window_sec is not None:
        overrides["tickrate_window_sec"] = args.tickrate_window_sec
    if args.min_atr_pips is not None:
        overrides["min_entry_atr_pips"] = args.min_atr_pips
    if args.gate_min_atr_pips is not None:
        overrides["gate_min_atr_pips"] = args.gate_min_atr_pips
    if args.news_block_min is not None:
        overrides["news_block_min"] = args.news_block_min
    if args.cooldown_sec is not None:
        overrides["entry_cooldown_sec"] = args.cooldown_sec
    if args.hold_ms is not None:
        overrides["min_hold_ms"] = args.hold_ms
    if args.sessions is not None:
        overrides["sessions"] = args.sessions

    if args.worker == "fast_scalp":
        result = replay_fast_scalp(ticks, exit_scheme=args.exit_scheme, overrides=overrides)
    elif args.worker == "pullback_s5":
        result = replay_pullback_s5(ticks)
    elif args.worker == "vwap_magnet_s5":
        result = replay_vwap_magnet_s5(ticks)
    elif args.worker == "impulse_break_s5":
        result = replay_impulse_break_s5(ticks)
    elif args.worker == "impulse_momentum_s5":
        result = replay_impulse_momentum_s5(ticks)
    elif args.worker == "squeeze_break_s5":
        result = replay_squeeze_break_s5(ticks)
    elif args.worker == "impulse_retest_s5":
        result = replay_impulse_retest_s5(ticks)
    elif args.worker == "mirror_spike_s5":
        result = replay_mirror_spike_s5(ticks)
    elif args.worker == "mirror_spike_tight":
        result = replay_mirror_spike_tight(ticks)
    elif args.worker == "pullback_scalp":
        result = replay_pullback_scalp(ticks)
    elif args.worker == "mirror_spike":
        result = replay_mirror_spike(ticks)
    else:
        raise ValueError(f"Unsupported worker: {args.worker}")

    if sim_meta:
        result.setdefault("meta", {})["simulation"] = sim_meta

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
