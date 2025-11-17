#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest exit_manager jitter on candle data (M1 -> H4 resample).

This harness replays a single day of M1 candles, generates macro signals
using TrendMA / Donchian55, applies range_guard and the current ExitManager,
and measures how often macro trades are closed too early (e.g., <5m, <2pips).

Usage:
  python scripts/backtest_exit_manager.py --candles logs/candles_M1_20251023.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.trend.ma_cross import MovingAverageCross
from strategies.breakout.donchian55 import Donchian55
from analysis.range_guard import detect_range_mode
from execution.exit_manager import ExitManager


PIP = 0.01


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
    df["bbw"] = ((std20 * 4).div(df["ma20"].abs().replace(0.0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50.0)

    # ATR and ADX approximations
    prev_close = prices.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().fillna(0.0)
    df["atr"] = atr
    df["atr_pips"] = df["atr"] / PIP

    # ADX
    up_move = df["high"].diff()
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr_all = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr_all.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    minus_di = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    atr_safe = atr14.replace(0.0, np.nan)
    plus_di = 100 * (plus_di / atr_safe)
    minus_di = 100 * (minus_di / atr_safe)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    dx = dx.fillna(0.0) * 100
    adx = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["adx"] = adx.fillna(0.0)

    # 5-minute pip vol proxy
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


@dataclass
class SimTrade:
    side: str  # "long" or "short"
    units: int
    open_time: datetime
    open_px: float
    exit_time: Optional[datetime] = None
    exit_px: Optional[float] = None
    pnl_pips: float = 0.0


def current_h4_row(df_h4: pd.DataFrame, t: datetime) -> Dict[str, float]:
    # pick the last H4 candle whose time <= t
    subset = df_h4[df_h4["time"] <= t]
    if subset.empty:
        return {}
    row = subset.iloc[-1]
    return {k: float(row[k]) if k != "time" else row[k] for k in row.index}


def build_fac(row: pd.Series, history: pd.DataFrame) -> Dict[str, float]:
    fac: Dict[str, float] = {}
    for k in ("close", "ema20", "ma10", "ma20", "bbw", "rsi", "atr", "atr_pips", "adx", "vol_5m"):
        if k in row and not pd.isna(row[k]):
            fac[k] = float(row[k])
    # candles list for Donchian
    candles_list: List[Dict[str, float]] = []
    for item in history.itertuples(index=False):
        candles_list.append({"open": float(item.open), "high": float(item.high), "low": float(item.low), "close": float(item.close)})
    fac["candles"] = candles_list[-60:]
    return fac


def simulate_day(df_m1: pd.DataFrame) -> Dict[str, float]:
    df_m1 = compute_indicators(df_m1)
    df_h4 = resample_h4(df_m1)

    em = ExitManager()
    open_trades: List[SimTrade] = []
    closed: List[SimTrade] = []

    for idx, row in df_m1.iterrows():
        t = row["time"]
        hist_start = max(0, idx - 80)
        hist = df_m1.iloc[hist_start : idx + 1]
        fac_m1 = build_fac(row, hist)
        fac_h4 = current_h4_row(df_h4, t)
        if not fac_h4:
            continue

        # Range detection
        range_ctx = detect_range_mode(fac_m1, fac_h4)
        range_active = bool(range_ctx.active)

        # Build signals (macro only)
        signals: List[Dict[str, object]] = []
        for Strat in (MovingAverageCross, Donchian55):
            sig = Strat.check(fac_m1)
            if not sig:
                continue
            # For the purpose of exit jitter evaluation, allow entries even in range;
            # range effect is applied via ExitManager range_mode behavior.
            signals.append({
                "strategy": Strat.name,
                "pocket": "macro",
                "action": sig.get("action"),
                "confidence": int(sig.get("confidence", 50) or 50),
                "sl_pips": float(sig.get("sl_pips", 30) or 30),
                "tp_pips": float(sig.get("tp_pips", 60) or 60),
                "tag": sig.get("tag", Strat.name),
            })

        # Construct open_positions compatible with ExitManager
        long_units = sum(tr.units for tr in open_trades if tr.side == "long")
        short_units = sum(tr.units for tr in open_trades if tr.side == "short")
        open_info = {
            "units": long_units - short_units,
            "avg_price": float(row["close"]),
            "trades": len(open_trades),
            "long_units": long_units,
            "short_units": short_units,
            "long_avg_price": (np.mean([tr.open_px for tr in open_trades if tr.side == "long"]) if long_units else 0.0),
            "short_avg_price": (np.mean([tr.open_px for tr in open_trades if tr.side == "short"]) if short_units else 0.0),
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

        # Plan exits
        exit_decisions = em.plan_closures(
            open_positions,
            signals,
            fac_m1,
            fac_h4,
            event_soon=False,
            range_mode=range_active,
            now=t,
            stage_tracker=None,
        )
        for decision in exit_decisions:
            if decision.pocket != "macro":
                continue
            # Close all for simplicity
            for tr in list(open_trades):
                if tr.side == ("long" if decision.units < 0 else "short"):
                    tr.exit_time = t
                    tr.exit_px = float(row["close"])
                    direction = 1 if tr.side == "long" else -1
                    tr.pnl_pips = (tr.exit_px - tr.open_px) * direction / PIP
                    closed.append(tr)
                    open_trades.remove(tr)

        # Open entries after exits (allow entries regardless of range; ExitManager handles range behavior)
        # Simple rule: allow one open trade at a time per direction
        cur_long = any(tr.side == "long" for tr in open_trades)
        cur_short = any(tr.side == "short" for tr in open_trades)
        for sig in signals:
            action = sig.get("action")
            if action == "OPEN_LONG" and not cur_long:
                open_trades.append(SimTrade(side="long", units=1000, open_time=t, open_px=float(row["close"])) )
                cur_long = True
            elif action == "OPEN_SHORT" and not cur_short:
                open_trades.append(SimTrade(side="short", units=1000, open_time=t, open_px=float(row["close"])) )
                cur_short = True

    # Close any remaining at last price
    if not df_m1.empty:
        last_t = df_m1.iloc[-1]["time"]
        last_px = float(df_m1.iloc[-1]["close"])
        for tr in open_trades:
            tr.exit_time = last_t
            tr.exit_px = last_px
            direction = 1 if tr.side == "long" else -1
            tr.pnl_pips = (last_px - tr.open_px) * direction / PIP
            closed.append(tr)

    # Metrics
    under5 = 0
    under2p = 0
    for tr in closed:
        if tr.exit_time and tr.open_time:
            age_min = (tr.exit_time - tr.open_time).total_seconds() / 60.0
            if age_min < 5.0:
                under5 += 1
        if abs(tr.pnl_pips) < 2.0:
            under2p += 1
    return {
        "closed": len(closed),
        "under5min": under5,
        "under2pips": under2p,
        "profit_pips": round(sum(tr.pnl_pips for tr in closed), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candles", required=True, help="logs/candles_M1_YYYYMMDD.json")
    args = ap.parse_args()
    df = load_candles(Path(args.candles))
    result = simulate_day(df)
    print(result)


if __name__ == "__main__":
    main()
