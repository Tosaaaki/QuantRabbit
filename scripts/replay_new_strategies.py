#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.range_guard import detect_range_mode
from indicators.calc_core import IndicatorEngine
from market_data import tick_window
from strategies.micro_lowvol.compression_revert import MicroCompressionRevert
from strategies.micro.trend_retest import MicroTrendRetest

PIP = 0.01

LSR_LOOKBACK = 20
LSR_SWEEP_PIPS = 0.45
LSR_RECLAIM_PIPS = 0.10
LSR_BODY_MAX_PIPS = 1.4
LSR_RANGE_MIN_PIPS = 1.2
LSR_WICK_RATIO_MIN = 0.5
LSR_ADX_MAX = 30.0
LSR_BBW_MAX = 0.0018
LSR_RANGE_SCORE_MIN = 0.35
LSR_REQUIRE_TICK_REVERSAL = True
LSR_SIZE_MULT = 1.0
LSR_ALLOWED_REGIMES: set[str] = set()
LSR_SWEEP_ADX_BUFFER = 0.15


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _atr_pips(fac_m1: Dict[str, object]) -> float:
    if fac_m1.get("atr") is not None:
        return abs(_to_float(fac_m1.get("atr")))
    if fac_m1.get("atr_pips") is not None:
        return abs(_to_float(fac_m1.get("atr_pips")))
    candles = fac_m1.get("candles")
    if not isinstance(candles, list) or len(candles) < 4:
        return 1.8
    values: List[float] = []
    for idx in range(1, len(candles)):
        c0 = candles[idx - 1]
        c1 = candles[idx]
        h = _to_float(c1.get("high", c1.get("h")))
        l = _to_float(c1.get("low", c1.get("l")))
        c_prev = _to_float(c0.get("close", c0.get("c")))
        values.append((max(h, c_prev) - min(l, c_prev)) / PIP)
    return sum(values) / len(values) if values else 1.8


def _adx(fac_m1: Dict[str, object]) -> float:
    if fac_m1.get("adx") is not None:
        return abs(_to_float(fac_m1.get("adx")))
    if fac_m1.get("trend_strength") is not None:
        return abs(_to_float(fac_m1.get("trend_strength")))
    return 0.0


def _bbw(fac_m1: Dict[str, object]) -> float:
    if fac_m1.get("bbw") is not None:
        return abs(_to_float(fac_m1.get("bbw")))
    candles = fac_m1.get("candles")
    if not isinstance(candles, list) or not candles:
        return 0.0
    closes = [ _to_float(c.get("close", c.get("c")) for c in candles if isinstance(c, dict) ]
    if len(closes) < 2:
        return 0.0
    hi = max(closes)
    lo = min(closes)
    if hi <= 0:
        return 0.0
    return (hi - lo) / hi


def _current_regime_score(range_ctx: object) -> float:
    try:
        return float(getattr(range_ctx, "score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _signal_liquidity_sweep(
    fac_m1: Dict[str, object],
    range_ctx: object | None = None,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    if LSR_ALLOWED_REGIMES:
        regime = str(fac_m1.get("regime") or "").strip().lower()
        if regime and regime not in LSR_ALLOWED_REGIMES:
            return None

    if _adx(fac_m1) > LSR_ADX_MAX:
        return None
    if _bbw(fac_m1) > LSR_BBW_MAX:
        return None
    if LSR_RANGE_SCORE_MIN > 0.0 and _current_regime_score(range_ctx) < LSR_RANGE_SCORE_MIN:
        return None

    candles = fac_m1.get("candles")
    if not isinstance(candles, list) or len(candles) < LSR_LOOKBACK + 2:
        return None

    last = candles[-1]
    prev = candles[-2]
    o = _to_float(last.get("open", last.get("o")))
    h = _to_float(last.get("high", last.get("h")))
    l = _to_float(last.get("low", last.get("l")))
    c = _to_float(last.get("close", last.get("c")))
    if h <= 0 or l <= 0:
        return None

    rng_pips = (h - l) / PIP
    if rng_pips < LSR_RANGE_MIN_PIPS:
        return None
    body_pips = abs(c - o) / PIP
    if body_pips > LSR_BODY_MAX_PIPS:
        return None
    upper_wick = (h - max(o, c)) / PIP
    lower_wick = (min(o, c) - l) / PIP
    wick_ratio = max(upper_wick, lower_wick) / max(rng_pips, 0.01)
    if wick_ratio < LSR_WICK_RATIO_MIN:
        return None

    history = candles[-(LSR_LOOKBACK + 1):-1]
    highs: List[float] = []
    lows: List[float] = []
    for candle in history:
        if not isinstance(candle, dict):
            continue
        hi_v = _to_float(candle.get("high", candle.get("h")))
        lo_v = _to_float(candle.get("low", candle.get("l")))
        if hi_v > 0 and lo_v > 0:
            highs.append(hi_v)
            lows.append(lo_v)
    if not highs or not lows:
        return None

    level_high = max(highs)
    level_low = min(lows)
    side: Optional[str] = None
    sweep_dist = 0.0
    if h >= level_high + LSR_SWEEP_PIPS * PIP and c < level_high - LSR_RECLAIM_PIPS * PIP:
        side = "short"
        sweep_dist = (h - level_high) / PIP
        if upper_wick < max(LSR_SWEEP_PIPS, 0.4):
            return None
    elif l <= level_low - LSR_SWEEP_PIPS * PIP and c > level_low + LSR_RECLAIM_PIPS * PIP:
        side = "long"
        sweep_dist = (level_low - l) / PIP
        if lower_wick < max(LSR_SWEEP_PIPS, 0.4):
            return None
    else:
        return None

    prev_close = _to_float(prev.get("close", prev.get("c")))
    rev_ok = True
    rev_dir = "long" if c >= prev_close else "short"
    rev_strength = 1.0
    if LSR_REQUIRE_TICK_REVERSAL and (not rev_ok or rev_dir != side):
        hard_sweep_ok = sweep_dist >= (LSR_SWEEP_PIPS * 2.0) and wick_ratio >= (LSR_WICK_RATIO_MIN + 0.15)
        if not hard_sweep_ok:
            return None
        rev_strength = 1.0
        rev_dir = side

    atr = _atr_pips(fac_m1)
    sl = max(1.4, min(2.6, atr * 0.8))
    tp = max(1.8, min(3.2, atr * 1.15))
    conf = 58 + int(min(14.0, sweep_dist * 3.0 + wick_ratio * 8.0))
    if rev_ok:
        conf += int(min(6.0, max(0.0, rev_strength) * 6.0))
    conf = max(45, min(92, conf))

    return {
        "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
        "sl_pips": round(sl, 2),
        "tp_pips": round(tp, 2),
        "confidence": conf,
        "tag": tag,
        "reason": "liquidity_sweep",
        "size_mult": round(LSR_SIZE_MULT, 3),
    }


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


class _TickObj:
    def __init__(self, bid: float, ask: float, epoch: float) -> None:
        self.bid = bid
        self.ask = ask
        self.time = datetime.fromtimestamp(epoch, tz=timezone.utc)


def _parse_epoch(raw: object) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        epoch = float(raw)
        if epoch > 10**11:
            epoch /= 1000.0
        return epoch
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def load_ticks(path: Path) -> List[Tick]:
    ticks: List[Tick] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            ts_raw = payload.get("timestamp") or payload.get("ts") or payload.get("time") or payload.get("ts_ms")
            epoch = _parse_epoch(ts_raw)
            if epoch is None:
                continue
            bid = payload.get("bid")
            ask = payload.get("ask")
            if bid is None or ask is None:
                mid = payload.get("mid")
                if mid is None:
                    continue
                mid = float(mid)
                bid = mid - 0.0015
                ask = mid + 0.0015
            ticks.append(Tick(epoch=epoch, bid=float(bid), ask=float(ask)))
    ticks.sort(key=lambda t: t.epoch)
    return ticks


def build_candles(ticks: Iterable[Tick]) -> List[Dict[str, float]]:
    candles: List[Dict[str, float]] = []
    bucket = None
    current = None
    for tick in ticks:
        tick_window.record(_TickObj(tick.bid, tick.ask, tick.epoch))
        minute = int(tick.epoch // 60)
        price = tick.mid
        if bucket is None:
            bucket = minute
            current = {
                "timestamp": tick.dt.isoformat(),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            }
            continue
        if minute != bucket:
            candles.append(current)  # type: ignore[arg-type]
            bucket = minute
            current = {
                "timestamp": tick.dt.isoformat(),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            }
        else:
            current["high"] = max(current["high"], price)
            current["low"] = min(current["low"], price)
            current["close"] = price
    if current:
        candles.append(current)
    return candles


def compute_factors(candles: List[Dict[str, float]], idx: int, window: deque) -> Optional[Dict[str, object]]:
    if idx >= len(candles):
        return None
    window.append(candles[idx])
    if len(window) < 20:
        return None
    df = pd.DataFrame(list(window))
    factors = IndicatorEngine.compute(df)
    factors["candles"] = list(window)
    last = window[-1]
    factors.update(
        {
            "close": last.get("close"),
            "open": last.get("open"),
            "high": last.get("high"),
            "low": last.get("low"),
            "timestamp": last.get("timestamp"),
        }
    )
    return factors


def simulate(signals: List[Dict], candles: List[Dict[str, float]], max_hold: int) -> Dict[str, Dict[str, float]]:
    summaries: Dict[str, Dict[str, float]] = defaultdict(lambda: {"n": 0, "win": 0, "sum_pips": 0.0, "profit": 0.0, "loss": 0.0})
    for sig in signals:
        idx = sig["index"]
        side = sig["side"]
        entry = sig["entry"]
        sl_pips = sig["sl_pips"]
        tp_pips = sig["tp_pips"]
        sl_price = entry - sl_pips * PIP if side == "long" else entry + sl_pips * PIP
        tp_price = entry + tp_pips * PIP if side == "long" else entry - tp_pips * PIP
        exit_price = entry
        outcome = "timeout"
        last_idx = min(len(candles) - 1, idx + max_hold)
        for j in range(idx + 1, last_idx + 1):
            high = candles[j]["high"]
            low = candles[j]["low"]
            hit_sl = low <= sl_price if side == "long" else high >= sl_price
            hit_tp = high >= tp_price if side == "long" else low <= tp_price
            if hit_sl and hit_tp:
                outcome = "sl"
                exit_price = sl_price
                break
            if hit_sl:
                outcome = "sl"
                exit_price = sl_price
                break
            if hit_tp:
                outcome = "tp"
                exit_price = tp_price
                break
        else:
            exit_price = candles[last_idx]["close"]
        pnl_pips = (exit_price - entry) / PIP if side == "long" else (entry - exit_price) / PIP
        strat = sig["strategy"]
        summary = summaries[strat]
        summary["n"] += 1
        summary["sum_pips"] += pnl_pips
        if pnl_pips > 0:
            summary["win"] += 1
            summary["profit"] += pnl_pips
        elif pnl_pips < 0:
            summary["loss"] += abs(pnl_pips)
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay new strategies on tick data (signal + simple TP/SL).")
    ap.add_argument("--ticks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hold", type=int, default=30)
    ap.add_argument("--limit-candles", type=int, default=0)
    args = ap.parse_args()

    ticks = load_ticks(Path(args.ticks))
    candles = build_candles(ticks)
    if args.limit_candles and args.limit_candles > 0:
        candles = candles[: args.limit_candles]

    window: deque = deque(maxlen=2000)
    signals: List[Dict] = []

    for idx in range(len(candles)):
        fac = compute_factors(candles, idx, window)
        if not fac:
            continue
        range_ctx = detect_range_mode(fac, fac, env_tf="M1", macro_tf="H4")

        lsr = _signal_liquidity_sweep(fac, range_ctx, tag="LiquiditySweep")
        if lsr:
            signals.append(
                {
                    "index": idx,
                    "strategy": "LiquiditySweep",
                    "side": "long" if lsr["action"] == "OPEN_LONG" else "short",
                    "entry": float(fac.get("close") or 0.0),
                    "sl_pips": float(lsr.get("sl_pips") or 0.0),
                    "tp_pips": float(lsr.get("tp_pips") or 0.0),
                    "confidence": int(lsr.get("confidence") or 0),
                }
            )

        rcr = MicroCompressionRevert.check(fac)
        if rcr:
            signals.append(
                {
                    "index": idx,
                    "strategy": "MicroCompressionRevert",
                    "side": "long" if rcr["action"] == "OPEN_LONG" else "short",
                    "entry": float(fac.get("close") or 0.0),
                    "sl_pips": float(rcr.get("sl_pips") or 0.0),
                    "tp_pips": float(rcr.get("tp_pips") or 0.0),
                    "confidence": int(rcr.get("confidence") or 0),
                }
            )

        trc = MicroTrendRetest.check(fac)
        if trc:
            signals.append(
                {
                    "index": idx,
                    "strategy": "MicroTrendRetest",
                    "side": "long" if trc["action"] == "OPEN_LONG" else "short",
                    "entry": float(fac.get("close") or 0.0),
                    "sl_pips": float(trc.get("sl_pips") or 0.0),
                    "tp_pips": float(trc.get("tp_pips") or 0.0),
                    "confidence": int(trc.get("confidence") or 0),
                }
            )

    summary = simulate(signals, candles, max_hold=args.max_hold)
    output = {
        "ticks": len(ticks),
        "candles": len(candles),
        "signals": len(signals),
        "max_hold": args.max_hold,
        "summary": {},
    }
    for strat, data in summary.items():
        n = data["n"] or 0
        win = data["win"] or 0
        profit = data["profit"] or 0.0
        loss = data["loss"] or 0.0
        pf = profit / loss if loss > 0 else None
        output["summary"][strat] = {
            "n": n,
            "win_rate": round(win / n, 4) if n else 0.0,
            "avg_pips": round(data["sum_pips"] / n, 3) if n else 0.0,
            "pf": round(pf, 3) if pf is not None else None,
        }

    Path(args.out).write_text(json.dumps(output, ensure_ascii=True, indent=2))
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
