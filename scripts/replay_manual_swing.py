#!/usr/bin/env python3
"""Offline replay harness for the manual_swing worker logic.

This script rebuilds the directional factors (MA/ATR/ADX) from cached M1 candle
files and emulates the staged swing strategy used by
``workers/manual_swing/worker.py``.  It is intended for parameter tuning and
profit-risk validation without touching live OANDA endpoints.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import statistics

import sys
from collections import deque

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workers.manual_swing import config as swing_config  # noqa: E402

PIP = swing_config.PIP_VALUE


@dataclass
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_candles(candle_dir: Path, start: date, end: date) -> List[Candle]:
    candles: List[Candle] = []
    current = start
    while current <= end:
        fname = candle_dir / f"candles_M1_{current:%Y%m%d}.json"
        if fname.exists():
            payload = json.loads(fname.read_text())
            for item in payload.get("candles", []):
                if not item.get("complete", False):
                    continue
                mid = item.get("mid") or {}
                candles.append(
                    Candle(
                        time=parse_iso(item["time"]),
                        open=float(mid.get("o", 0.0)),
                        high=float(mid.get("h", 0.0)),
                        low=float(mid.get("l", 0.0)),
                        close=float(mid.get("c", 0.0)),
                    )
                )
        current += timedelta(days=1)
    candles.sort(key=lambda c: c.time)
    return candles


def scaled_stage_units(equity: float) -> List[int]:
    if equity <= 0:
        equity = swing_config.REFERENCE_EQUITY
    scale = equity / swing_config.REFERENCE_EQUITY if swing_config.REFERENCE_EQUITY else 1.0
    units: List[int] = []
    for base in swing_config.STAGE_UNITS_BASE[: swing_config.MAX_ACTIVE_STAGES]:
        scaled = int(round(base * scale))
        units.append(max(swing_config.MIN_STAGE_UNITS, scaled))
    return units


def simple_moving_average(values: Sequence[float], window: int) -> Optional[float]:
    if len(values) < window or window <= 0:
        return None
    window_slice = list(values)[-window:]
    return sum(window_slice) / window


def compute_atr(candles: Sequence[Candle], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    trs: List[float] = []
    for i in range(1, len(candles)):
        cur = candles[i]
        prev = candles[i - 1]
        tr = max(
            cur.high - cur.low,
            abs(cur.high - prev.close),
            abs(cur.low - prev.close),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period / PIP


def compute_adx(candles: Sequence[Candle], period: int = 14) -> Optional[float]:
    if len(candles) <= period:
        return None
    trs: List[float] = []
    dm_plus: List[float] = []
    dm_minus: List[float] = []
    for i in range(1, len(candles)):
        cur = candles[i]
        prev = candles[i - 1]
        up = cur.high - prev.high
        down = prev.low - cur.low
        dm_plus.append(up if up > down and up > 0 else 0.0)
        dm_minus.append(down if down > up and down > 0 else 0.0)
        trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    if len(trs) < period:
        return None
    idx: List[float] = []
    for i in range(period - 1, len(trs)):
        tr_sum = sum(trs[i - period + 1 : i + 1])
        if tr_sum <= 0:
            continue
        di_plus = 100.0 * (sum(dm_plus[i - period + 1 : i + 1]) / tr_sum)
        di_minus = 100.0 * (sum(dm_minus[i - period + 1 : i + 1]) / tr_sum)
        denom = di_plus + di_minus
        if denom <= 0:
            continue
        idx.append(100.0 * abs(di_plus - di_minus) / denom)
    if len(idx) < period:
        return None
    return sum(idx[-period:]) / period


def last_features_direction(features: Dict[str, float]) -> Optional[str]:
    if (
        features.get("ma_gap_h4", 0.0) >= swing_config.H4_GAP_MIN
        and features.get("ma_gap_h1", 0.0) >= swing_config.H1_GAP_MIN
        and features.get("adx", 0.0) >= swing_config.ADX_MIN
    ):
        return "long"
    if (
        features.get("ma_gap_h4", 0.0) <= -swing_config.H4_GAP_MIN
        and features.get("ma_gap_h1", 0.0) <= -swing_config.H1_GAP_MIN
        and features.get("adx", 0.0) >= swing_config.ADX_MIN
    ):
        return "short"
    return None


@dataclass
class TradeRecord:
    direction: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    units: int
    stages: int
    hold_hours: float
    pnl_pips: float
    pnl_jpy: float
    reason: str


def replay_manual_swing(
    candles: List[Candle],
    initial_nav: float,
    margin_rate: float,
) -> Dict[str, object]:
    nav = initial_nav
    margin_used = 0.0
    trades: List[TradeRecord] = []
    equity_curve: List[Tuple[datetime, float]] = []

    position: Optional[Dict[str, object]] = None
    stage_cooldown_until: Dict[str, datetime] = {
        "long": datetime.min.replace(tzinfo=timezone.utc),
        "short": datetime.min.replace(tzinfo=timezone.utc),
    }
    cooldown_delta = timedelta(minutes=swing_config.STAGE_COOLDOWN_MINUTES)

    last_features: Dict[str, float] = {}
    h1_builder: List[Candle] = []
    h4_builder: List[Candle] = []
    h1_candles: List[Candle] = []
    h4_candles: List[Candle] = []

    for candle in candles:
        h1_builder.append(candle)
        h4_builder.append(candle)
        if len(h1_builder) == 60:
            h1_candle = Candle(
                time=h1_builder[-1].time,
                open=h1_builder[0].open,
                high=max(c.high for c in h1_builder),
                low=min(c.low for c in h1_builder),
                close=h1_builder[-1].close,
            )
            h1_candles.append(h1_candle)
            h1_builder.clear()

            if len(h4_builder) == 240:
                h4_candle = Candle(
                    time=h4_builder[-1].time,
                    open=h4_builder[0].open,
                    high=max(c.high for c in h4_builder),
                    low=min(c.low for c in h4_builder),
                    close=h4_builder[-1].close,
                )
                h4_candles.append(h4_candle)
                h4_builder.clear()

            ma10_h1 = simple_moving_average([c.close for c in h1_candles], 10)
            ma20_h1 = simple_moving_average([c.close for c in h1_candles], 20)
            ma10_h4 = simple_moving_average([c.close for c in h4_candles], 10)
            ma20_h4 = simple_moving_average([c.close for c in h4_candles], 20)
            atr = compute_atr(h1_candles, 14)
            adx = compute_adx(h1_candles, 14)

            if None not in (ma10_h1, ma20_h1, ma10_h4, ma20_h4, atr, adx):
                last_features = {
                    "ma_gap_h1": float(ma10_h1 - ma20_h1),
                    "ma_gap_h4": float(ma10_h4 - ma20_h4),
                    "atr_pips": float(atr),
                    "adx": float(adx),
                }

        if not last_features:
            continue

        price = candle.close
        now = candle.time
        stage_units = scaled_stage_units(nav)
        if not stage_units:
            continue
        max_stage_count = min(swing_config.MAX_ACTIVE_STAGES, len(stage_units))

        # Track equity curve (incl. unrealized)
        unrealized = 0.0
        if position:
            units = int(position["units"])
            if units != 0:
                avg_price = float(position["avg_price"])
                sign = 1 if units > 0 else -1
                pip_diff = (price - avg_price) / PIP * sign
                unrealized = pip_diff * abs(units) / 100.0
        equity_curve.append((now, nav + unrealized))

        # Exit logic
        if position:
            units = int(position["units"])
            side = position["side"]
            avg_price = float(position["avg_price"])
            hold_hours = (now - position["open_time"]).total_seconds() / 3600.0
            margin_available = max(nav - margin_used, 0.0)
            free_margin_ratio = margin_available / nav if nav > 0 else 0.0

            direction = last_features_direction(last_features)
            if direction == "long" and not swing_config.ALLOW_LONG:
                direction = None
            elif direction == "short" and not swing_config.ALLOW_SHORT:
                direction = None

            should_exit = False
            reason = ""

            if direction and direction != side:
                should_exit = True
                reason = "trend_flip"
            elif hold_hours >= swing_config.MAX_HOLD_HOURS:
                should_exit = True
                reason = "max_hold"
            elif free_margin_ratio < swing_config.MARGIN_HEALTH_EXIT:
                should_exit = True
                reason = "margin_health"
            else:
                pip_drawdown = (
                    (price - avg_price) / PIP if side == "long" else (avg_price - price) / PIP
                )
                if pip_drawdown < -swing_config.MAX_DRAWDOWN_PIPS:
                    should_exit = True
                    reason = "max_drawdown"
                else:
                    favourable = (
                        (price - avg_price) / PIP if side == "long" else (avg_price - price) / PIP
                    )
                    best = position.get("best_pips", 0.0)
                    if favourable > best:
                        position["best_pips"] = favourable
                        best = favourable
                    if best >= swing_config.PROFIT_TRIGGER_PIPS:
                        should_exit = True
                        reason = "profit_lock"
                    elif (
                        best >= swing_config.TRAIL_TRIGGER_PIPS
                        and favourable <= best - swing_config.TRAIL_BACKOFF_PIPS
                    ):
                        should_exit = True
                        reason = "trail_backoff"
                    elif (
                        side == "long" and last_features["ma_gap_h1"] < -swing_config.REVERSAL_GAP_EXIT
                    ) or (
                        side == "short" and last_features["ma_gap_h1"] > swing_config.REVERSAL_GAP_EXIT
                    ):
                        should_exit = True
                        reason = "gap_reversal"

            if should_exit:
                sign = 1 if units > 0 else -1
                pnl_pips = (price - avg_price) / PIP * sign
                pnl_jpy = pnl_pips * abs(units) / 100.0
                nav += pnl_jpy
                margin_used = 0.0
                trades.append(
                    TradeRecord(
                        direction=side,
                        entry_time=position["open_time"].isoformat(),
                        exit_time=now.isoformat(),
                        entry_price=avg_price,
                        exit_price=price,
                        units=abs(units),
                        stages=position["stages"],
                        hold_hours=hold_hours,
                        pnl_pips=pnl_pips,
                        pnl_jpy=pnl_jpy,
                        reason=reason,
                    )
                )
                position = None
                stage_cooldown_until["long"] = now + cooldown_delta
                stage_cooldown_until["short"] = now + cooldown_delta
                continue

        # Entry logic
        direction = last_features_direction(last_features)
        if direction == "long" and not swing_config.ALLOW_LONG:
            direction = None
        elif direction == "short" and not swing_config.ALLOW_SHORT:
            direction = None
        if direction is None:
            continue

        if now < stage_cooldown_until[direction]:
            continue

        margin_available = max(nav - margin_used, 0.0)
        if nav <= 0:
            continue
        free_margin_ratio = margin_available / nav if nav else 0.0
        if free_margin_ratio < swing_config.MIN_FREE_MARGIN_RATIO:
            continue

        atr_pips = last_features["atr_pips"]
        if atr_pips < swing_config.ATR_MIN_PIPS or last_features["adx"] < swing_config.ADX_MIN:
            continue

        current_units = int(position["units"]) if position else 0
        if current_units != 0 and (
            (current_units > 0 and direction == "short")
            or (current_units < 0 and direction == "long")
        ):
            continue

        stage_count = position["stages"] if position else 0
        stage_count = int(stage_count)
        if stage_count >= max_stage_count:
            continue

        stage_size = stage_units[stage_count]
        if stage_size < swing_config.MIN_STAGE_UNITS:
            continue

        if stage_count > 0 and position and position.get("side") == direction:
            anchor = position.get("last_stage_price") or position.get("avg_price") or price
            try:
                anchor_val = float(anchor)
                favourable = (
                    (price - anchor_val) / PIP if direction == "long" else (anchor_val - price) / PIP
                )
            except Exception:
                favourable = 0.0
            if favourable < swing_config.STAGE_ADD_TRIGGER_PIPS:
                continue
        elif current_units != 0:
            continue

        leverage_budget = margin_available * swing_config.RISK_FREE_MARGIN_FRACTION
        if margin_rate <= 0:
            continue
        units_budget = int(leverage_budget / (price * margin_rate))
        if units_budget <= 0:
            continue
        incremental_units = min(stage_size, units_budget)
        if incremental_units < swing_config.MIN_STAGE_UNITS:
            continue

        side = direction
        signed_units = incremental_units if side == "long" else -incremental_units
        margin_used += abs(signed_units) * price * margin_rate

        sl_pips = max(swing_config.MIN_SL_PIPS, atr_pips * swing_config.SL_ATR_MULT)
        tp_pips = max(swing_config.MIN_TP_PIPS, atr_pips * swing_config.TP_ATR_MULT)
        if side == "long":
            sl_price = price - sl_pips * PIP
            tp_price = price + tp_pips * PIP
        else:
            sl_price = price + sl_pips * PIP
            tp_price = price - tp_pips * PIP

        if not position:
            position = {
                "units": signed_units,
                "avg_price": price,
                "side": side,
                "stages": 1,
                "open_time": now,
                "stages_detail": [
                    {
                        "units": abs(signed_units),
                        "price": price,
                        "time": now.isoformat(),
                    }
                ],
                "sl_price": sl_price,
                "tp_price": tp_price,
                "best_pips": 0.0,
                "last_stage_price": price,
            }
        else:
            prev_units = int(position["units"])
            prev_avg = float(position["avg_price"])
            new_total = prev_units + signed_units
            if new_total == 0:
                continue
            weighted = (
                prev_avg * abs(prev_units) + price * abs(signed_units)
            ) / abs(new_total)
            position["units"] = new_total
            position["avg_price"] = weighted
            position["stages"] += 1
            position["stages_detail"].append(
                {
                    "units": abs(signed_units),
                    "price": price,
                    "time": now.isoformat(),
                }
            )
            position["sl_price"] = sl_price
            position["tp_price"] = tp_price
            position.setdefault("best_pips", 0.0)
            position["last_stage_price"] = price
            position["side"] = side

        stage_cooldown_until[direction] = now + cooldown_delta

    # Close residual position at end of replay window
    if position:
        units = int(position["units"])
        if units != 0:
            side = position["side"]
            avg_price = float(position["avg_price"])
            sign = 1 if units > 0 else -1
            pnl_pips = (candles[-1].close - avg_price) / PIP * sign
            pnl_jpy = pnl_pips * abs(units) / 100.0
            nav += pnl_jpy
            trades.append(
                TradeRecord(
                    direction=side,
                    entry_time=position["open_time"].isoformat(),
                    exit_time=candles[-1].time.isoformat(),
                    entry_price=avg_price,
                    exit_price=candles[-1].close,
                    units=abs(units),
                    stages=position["stages"],
                    hold_hours=(candles[-1].time - position["open_time"]).total_seconds() / 3600.0,
                    pnl_pips=pnl_pips,
                    pnl_jpy=pnl_jpy,
                    reason="forced_close_end_of_replay",
                )
            )

    total_pnl_jpy = sum(t.pnl_jpy for t in trades)
    total_pnl_pips = sum(t.pnl_pips for t in trades)
    wins = sum(1 for t in trades if t.pnl_jpy > 0)
    win_rate = wins / len(trades) if trades else 0.0
    equity_values = [val for _, val in equity_curve]
    max_equity = max(equity_values) if equity_values else nav
    min_equity = min(equity_values) if equity_values else nav
    max_drawdown = max_equity - min_equity

    summary = {
        "trades": len(trades),
        "total_pnl_pips": total_pnl_pips,
        "total_pnl_jpy": total_pnl_jpy,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "final_nav": nav,
    }
    timeline = [{"time": ts.isoformat(), "equity": eq} for ts, eq in equity_curve]
    return {
        "summary": summary,
        "trades": [asdict(t) for t in trades],
        "equity_curve": timeline,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candles-dir",
        type=Path,
        default=Path("logs/oanda_candles"),
        help="Directory containing candles_M1_YYYYMMDD.json files",
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--initial-nav", type=float, default=500000.0, help="Initial NAV (JPY)")
    parser.add_argument("--margin-rate", type=float, default=0.02, help="Account margin rate")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tmp/replay_manual_swing.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start_date).date()
    end = datetime.fromisoformat(args.end_date).date()
    candles = load_candles(args.candles_dir, start, end)
    if not candles:
        raise SystemExit("No candles found for the requested range.")
    result = replay_manual_swing(candles, args.initial_nav, args.margin_rate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"[manual_swing] wrote replay results -> {args.out}")


if __name__ == "__main__":
    main()
