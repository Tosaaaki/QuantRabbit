#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay a day of M1 candles and evaluate macro strategy performance with
MacroState-based gating and event-window suppression.

This harness approximates the macro portion of ``main.py`` without requiring
live services.  It loads a MacroState snapshot (JSON) and applies the same bias
logic used in the trading loop to filter MovingAverageCross / Donchian55
signals.  Event windows can optionally block new entries to mimic the live
risk guard.

Example:
    python scripts/backtest_macro_gate.py \\
        --candles logs/candles_M1_20251022.json \\
        --snapshot fixtures/macro_snapshots/latest.json \\
        --gate-enabled \\
        --event-block
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.macro_state import MacroState, MacroSnapshot
from analysis.range_guard import detect_range_mode
from execution.exit_manager import ExitManager
from strategies.trend.ma_cross import MovingAverageCross
from strategies.breakout.donchian55 import Donchian55

PIP = 0.01
TARGET_INSTRUMENT = "USDJPY"


def parse_time(ts: str) -> datetime:
    ts = ts.rstrip("Z")
    if "." in ts:
        head, frac = ts.split(".", 1)
        frac = (frac + "000000")[:6]
        ts = f"{head}.{frac}+00:00"
    else:
        ts = ts + "+00:00"
    return datetime.fromisoformat(ts)


def load_candles(path: Path) -> pd.DataFrame:
    with path.open() as f:
        payload = __import__("json").load(f)
    candles = payload.get("candles", payload)
    rows: List[Dict[str, float]] = []
    for c in candles:
        mid = c.get("mid") or {}
        rows.append(
            {
                "time": parse_time(c["time"]),
                "open": float(mid.get("o", mid.get("open", 0.0))),
                "high": float(mid.get("h", mid.get("high", 0.0))),
                "low": float(mid.get("l", mid.get("low", 0.0))),
                "close": float(mid.get("c", mid.get("close", 0.0))),
            }
        )
    df = pd.DataFrame(rows)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    prices = df["close"]
    df["ema20"] = prices.ewm(span=20, adjust=False, min_periods=20).mean()
    df["ma10"] = prices.rolling(window=10, min_periods=10).mean()
    df["ma20"] = prices.rolling(window=20, min_periods=20).mean()
    std20 = prices.rolling(window=20, min_periods=20).std()
    df["bbw"] = ((std20 * 4).div(df["ma20"].abs().replace(0.0, np.nan))).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50.0)

    prev_close = prices.shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().fillna(0.0)
    df["atr"] = atr
    df["atr_pips"] = df["atr"] / PIP

    up_move = df["high"].diff()
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr_all = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr_all.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = (
        pd.Series(plus_dm, index=df.index)
        .ewm(alpha=1 / 14, adjust=False, min_periods=14)
        .mean()
    )
    minus_di = (
        pd.Series(minus_dm, index=df.index)
        .ewm(alpha=1 / 14, adjust=False, min_periods=14)
        .mean()
    )
    atr_safe = atr14.replace(0.0, np.nan)
    plus_di = 100 * (plus_di / atr_safe)
    minus_di = 100 * (minus_di / atr_safe)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    dx = dx.fillna(0.0) * 100
    adx = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["adx"] = adx.fillna(0.0)

    pip_move = prices.diff().abs() / PIP
    df["vol_5m"] = pip_move.rolling(window=5, min_periods=5).mean().fillna(0.0)
    return df


def resample_h4(df_m1: pd.DataFrame) -> pd.DataFrame:
    res = (
        df_m1.set_index("time")
        .resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    )
    res = res.dropna(subset=["open", "high", "low", "close"]).reset_index()
    res = compute_indicators(res)
    return res


def build_fac(row: pd.Series, history: pd.DataFrame) -> Dict[str, float]:
    fac: Dict[str, float] = {}
    for key in (
        "close",
        "ema20",
        "ma10",
        "ma20",
        "bbw",
        "rsi",
        "atr",
        "atr_pips",
        "adx",
        "vol_5m",
    ):
        val = row.get(key)
        if val is not None and not np.isnan(val):
            fac[key] = float(val)
    candles = [
        {"open": float(item.open), "high": float(item.high), "low": float(item.low), "close": float(item.close)}
        for item in history.itertuples(index=False)
    ]
    fac["candles"] = candles[-60:]
    return fac


def current_h4_row(df_h4: pd.DataFrame, t: datetime) -> Dict[str, float]:
    subset = df_h4[df_h4["time"] <= t]
    if subset.empty:
        return {}
    row = subset.iloc[-1]
    data: Dict[str, float] = {}
    for col in ("close", "ema20", "ma10", "ma20", "adx", "bbw", "atr", "atr_pips", "vol_5m"):
        if col in row and not pd.isna(row[col]):
            data[col] = float(row[col])
    data["time"] = row["time"]
    return data


@dataclass
class SimTrade:
    side: str
    units: int
    open_time: datetime
    open_px: float
    exit_time: Optional[datetime] = None
    exit_px: Optional[float] = None
    pnl_pips: float = 0.0


def simulate_day(
    df_m1: pd.DataFrame,
    macro_state: Optional[MacroState],
    *,
    gate_enabled: bool,
    event_block: bool,
    event_before: float,
    event_after: float,
) -> Dict[str, float]:
    df_m1 = compute_indicators(df_m1)
    df_h4 = resample_h4(df_m1)

    em = ExitManager()
    open_trades: List[SimTrade] = []
    closed: List[SimTrade] = []
    macro_blocks = 0
    event_blocks = 0
    total_signals = 0

    for idx, row in df_m1.iterrows():
        t = row["time"]
        hist_start = max(0, idx - 80)
        hist = df_m1.iloc[hist_start : idx + 1]
        fac_m1 = build_fac(row, hist)
        fac_h4 = current_h4_row(df_h4, t)
        if not fac_h4:
            continue

        range_ctx = detect_range_mode(fac_m1, fac_h4)
        range_active = bool(range_ctx.active)

        signals: List[Dict[str, object]] = []
        for Strat in (MovingAverageCross, Donchian55):
            sig = Strat.check(fac_m1)
            if not sig:
                continue
            total_signals += 1

            action = sig.get("action")
            blocked = False

            if gate_enabled and macro_state:
                bias_val = macro_state.bias(TARGET_INSTRUMENT)
                if abs(bias_val) >= macro_state.deadzone:
                    is_long = action == "OPEN_LONG"
                    if (bias_val > 0 and not is_long) or (bias_val < 0 and is_long):
                        macro_blocks += 1
                        blocked = True

            if (
                not blocked
                and event_block
                and macro_state
                and macro_state.in_event_window(
                    TARGET_INSTRUMENT,
                    before_hours=event_before,
                    after_hours=event_after,
                    now=t.replace(tzinfo=timezone.utc),
                )
            ):
                event_blocks += 1
                blocked = True

            if blocked:
                continue

            signals.append(
                {
                    "strategy": Strat.name,
                    "pocket": "macro",
                    "action": action,
                    "confidence": int(sig.get("confidence", 50) or 50),
                    "sl_pips": float(sig.get("sl_pips", 30) or 30),
                    "tp_pips": float(sig.get("tp_pips", 60) or 60),
                    "tag": sig.get("tag", Strat.name),
                }
            )

        long_units = sum(tr.units for tr in open_trades if tr.side == "long")
        short_units = sum(tr.units for tr in open_trades if tr.side == "short")
        open_info = {
            "units": long_units - short_units,
            "avg_price": float(row["close"]),
            "trades": len(open_trades),
            "long_units": long_units,
            "short_units": short_units,
            "long_avg_price": (
                np.mean([tr.open_px for tr in open_trades if tr.side == "long"])
                if long_units
                else 0.0
            ),
            "short_avg_price": (
                np.mean([tr.open_px for tr in open_trades if tr.side == "short"])
                if short_units
                else 0.0
            ),
            "open_trades": [
                {
                    "trade_id": f"sim-{i}",
                    "side": tr.side,
                    "units": tr.units,
                    "open_time": tr.open_time.isoformat().replace("+00:00", "Z"),
                }
                for i, tr in enumerate(open_trades)
            ],
        }
        open_positions = {"macro": open_info, "__net__": {"units": open_info["units"]}}

        exit_decisions = em.plan_closures(
            open_positions,
            signals,
            fac_m1,
            fac_h4,
            event_soon=False,
            range_mode=range_active,
            now=t,
        )
        for decision in exit_decisions:
            if decision.pocket != "macro":
                continue
            for tr in list(open_trades):
                if tr.side == ("long" if decision.units < 0 else "short"):
                    tr.exit_time = t
                    tr.exit_px = float(row["close"])
                    direction = 1 if tr.side == "long" else -1
                    tr.pnl_pips = (tr.exit_px - tr.open_px) * direction / PIP
                    closed.append(tr)
                    open_trades.remove(tr)

        cur_long = any(tr.side == "long" for tr in open_trades)
        cur_short = any(tr.side == "short" for tr in open_trades)
        for sig in signals:
            action = sig.get("action")
            if action == "OPEN_LONG" and not cur_long:
                open_trades.append(
                    SimTrade(side="long", units=1000, open_time=t, open_px=float(row["close"]))
                )
                cur_long = True
            elif action == "OPEN_SHORT" and not cur_short:
                open_trades.append(
                    SimTrade(side="short", units=1000, open_time=t, open_px=float(row["close"]))
                )
                cur_short = True

    if not df_m1.empty:
        last_t = df_m1.iloc[-1]["time"]
        last_px = float(df_m1.iloc[-1]["close"])
        for tr in open_trades:
            tr.exit_time = last_t
            tr.exit_px = last_px
            direction = 1 if tr.side == "long" else -1
            tr.pnl_pips = (last_px - tr.open_px) * direction / PIP
            closed.append(tr)

    total_pips = round(sum(tr.pnl_pips for tr in closed), 2)
    wins = sum(1 for tr in closed if tr.pnl_pips > 0)
    losses = sum(1 for tr in closed if tr.pnl_pips < 0)
    win_rate = wins / len(closed) if closed else 0.0
    gross_prof = sum(tr.pnl_pips for tr in closed if tr.pnl_pips > 0)
    gross_loss = -sum(tr.pnl_pips for tr in closed if tr.pnl_pips < 0)
    profit_factor = gross_prof / gross_loss if gross_loss else float("inf")
    max_dd = 0.0
    equity = 0.0
    peak = 0.0
    for tr in closed:
        equity += tr.pnl_pips
        peak = max(peak, equity)
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return {
        "signals_total": total_signals,
        "signals_allowed": total_signals - macro_blocks - event_blocks,
        "macro_blocks": macro_blocks,
        "event_blocks": event_blocks,
        "trades": len(closed),
        "profit_pips": round(total_pips, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor if profit_factor != float("inf") else 0.0, 4),
        "max_dd_pips": round(max_dd, 2),
    }


def load_macro_state(path: Optional[Path], deadzone: float) -> Optional[MacroState]:
    if not path:
        return None
    data = path.read_text(encoding="utf-8")
    snap = MacroSnapshot(**__import__("json").loads(data))
    return MacroState(snap, deadzone=deadzone)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candles", required=True, help="logs/candles_M1_YYYYMMDD.json")
    ap.add_argument("--snapshot", help="MacroState snapshot JSON file")
    ap.add_argument("--deadzone", type=float, default=0.25, help="Macro bias deadzone")
    ap.add_argument("--gate-enabled", action="store_true", help="Enable macro bias gating")
    ap.add_argument("--event-block", action="store_true", help="Block entries during event window")
    ap.add_argument("--event-before", type=float, default=2.0, help="Event window hours before")
    ap.add_argument("--event-after", type=float, default=1.0, help="Event window hours after")
    args = ap.parse_args()

    df = load_candles(Path(args.candles))
    macro_state = load_macro_state(Path(args.snapshot), args.deadzone) if args.snapshot else None

    result = simulate_day(
        df,
        macro_state,
        gate_enabled=args.gate_enabled,
        event_block=args.event_block,
        event_before=args.event_before,
        event_after=args.event_after,
    )
    print(__import__("json").dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
