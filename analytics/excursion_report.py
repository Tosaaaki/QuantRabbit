"""
analytics.excursion_report
~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute trade-level MFE/MAE and post-exit favorable excursion statistics
from logs/trades.db and M1 candles in logs/candles_M1_*.json.

Outputs a concise textual summary grouped overall and by JST hour of close.

Usage (from repo root):
  python -m analytics.excursion_report --days 3 --post-min 15 --thresholds 0.6 1.0 1.6 2.0
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS = REPO_ROOT / "logs"
PIP = 0.01  # USD/JPY


def _parse_iso(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    raw = ts.strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        if "." in raw and "+" in raw and raw.rfind("+") > raw.find("."):
            # Normalize fractional seconds to <= 6 digits
            head, frac_plus = raw.split(".", 1)
            frac, plus = frac_plus.split("+", 1)
            frac_digits = "".join(ch for ch in frac if ch.isdigit())[:6]
            raw = f"{head}.{frac_digits}+{plus}"
        elif "." in raw and "+" not in raw:
            head, frac = raw.split(".", 1)
            frac_digits = "".join(ch for ch in frac if ch.isdigit())[:6]
            raw = f"{head}.{frac_digits}+00:00"
        elif "+" not in raw:
            raw = f"{raw}+00:00"
        return dt.datetime.fromisoformat(raw).astimezone(dt.timezone.utc)
    except Exception:
        try:
            trimmed = raw.split(".", 1)[0].rstrip("Z") + "+00:00"
            return dt.datetime.fromisoformat(trimmed).astimezone(dt.timezone.utc)
        except Exception:
            return None


@dataclass
class Candle:
    ts: dt.datetime  # UTC
    o: float
    h: float
    l: float
    c: float


def _load_m1_for_dates(dates: Iterable[dt.date]) -> Dict[dt.date, List[Candle]]:
    out: Dict[dt.date, List[Candle]] = {}
    for d in sorted(set(dates)):
        path = LOGS / f"candles_M1_{d.strftime('%Y%m%d')}.json"
        if not path.exists():
            # try aggregated file range
            agg = LOGS / "candles_M1_20251001_20251022.json"
            if agg.exists():
                # rudimentary filter by date later
                path = agg
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        arr = data.get("candles") or []
        entries: List[Candle] = []
        for row in arr:
            t = _parse_iso(row.get("time"))
            if not t:
                continue
            mid = row.get("mid") or {}
            try:
                cnd = Candle(
                    ts=t,
                    o=float(mid.get("o")),
                    h=float(mid.get("h")),
                    l=float(mid.get("l")),
                    c=float(mid.get("c")),
                )
                entries.append(cnd)
            except Exception:
                continue
        if path.name.startswith("candles_M1_") and path.name.endswith(".json") and len(entries) > 0:
            # If aggregated, keep all and filter later by date keys when accessing
            out[d] = entries
    return out


def _iter_m1_between(
    candles_by_day: Dict[dt.date, List[Candle]], start: dt.datetime, end: dt.datetime
) -> Iterable[Candle]:
    cur = start
    while cur <= end:
        day = cur.date()
        arr = candles_by_day.get(day)
        if arr:
            for c in arr:
                if start <= c.ts <= end:
                    yield c
        cur += dt.timedelta(days=1)


@dataclass
class TradeRow:
    id: int
    pocket: str
    units: int
    entry_price: float
    close_price: Optional[float]
    open_time: dt.datetime
    close_time: Optional[dt.datetime]


def _load_trades(days: int) -> List[TradeRow]:
    path = LOGS / "trades.db"
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    utc_now = dt.datetime.now(dt.timezone.utc)
    since = utc_now - dt.timedelta(days=max(1, days))
    rows = con.execute(
        """
        SELECT id, pocket, units, entry_price, close_price, open_time, close_time
        FROM trades
        WHERE open_time IS NOT NULL
          AND (close_time IS NULL OR close_time >= ?)
        ORDER BY id ASC
        """,
        (since.isoformat(),),
    ).fetchall()
    con.close()
    out: List[TradeRow] = []
    for r in rows:
        ot = _parse_iso(r["open_time"])
        if not ot:
            continue
        ct = _parse_iso(r["close_time"]) if r["close_time"] else None
        try:
            out.append(
                TradeRow(
                    id=int(r["id"]),
                    pocket=str(r["pocket"] or ""),
                    units=int(r["units"] or 0),
                    entry_price=float(r["entry_price"]),
                    close_price=float(r["close_price"]) if r["close_price"] is not None else None,
                    open_time=ot,
                    close_time=ct,
                )
            )
        except Exception:
            continue
    return out


def _mfe_mae_for_trade(
    trade: TradeRow,
    candles_by_day: Dict[dt.date, List[Candle]],
) -> Tuple[float, float, Optional[int]]:
    """Return (MFE_pips, MAE_pips, time_to_MFE_min) within the hold window."""
    start = trade.open_time
    end = trade.close_time or (trade.open_time + dt.timedelta(hours=2))
    entry = trade.entry_price
    long = trade.units > 0
    mfe = 0.0
    mae = 0.0
    t_mfe: Optional[int] = None
    idx = 0
    for c in _iter_m1_between(candles_by_day, start, end):
        if long:
            fav = (c.h - entry) / PIP
            adv = (entry - c.l) / PIP
        else:
            fav = (entry - c.l) / PIP  # favorable = down move
            adv = (c.h - entry) / PIP
        if fav > mfe:
            mfe = fav
            t_mfe = idx
        if adv > mae:
            mae = adv
        idx += 1
    return round(mfe, 2), round(mae, 2), t_mfe


def _post_exit_mfe(
    trade: TradeRow,
    candles_by_day: Dict[dt.date, List[Candle]],
    post_minutes: int,
) -> float:
    if not trade.close_time:
        return 0.0
    start = trade.close_time
    end = start + dt.timedelta(minutes=max(1, post_minutes))
    entry = trade.entry_price
    long = trade.units > 0
    best = 0.0
    for c in _iter_m1_between(candles_by_day, start, end):
        if long:
            fav = (c.h - entry) / PIP
        else:
            fav = (entry - c.l) / PIP
        if fav > best:
            best = fav
    return round(best, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2, help="lookback days")
    ap.add_argument("--post-min", type=int, default=15, help="post-exit window minutes")
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.6, 1.0, 1.6, 2.0],
        help="pip thresholds for success rates",
    )
    args = ap.parse_args()

    trades = _load_trades(args.days)
    if not trades:
        print("No trades found in lookback window.")
        return
    dates = set()
    for t in trades:
        dates.add(t.open_time.date())
        if t.close_time:
            dates.add(t.close_time.date())
    candles_by_day = _load_m1_for_dates(dates)

    # Aggregates
    overall = {
        "count": 0,
        "neg_trades": 0,
        "mfe_sum": 0.0,
        "mae_sum": 0.0,
        "post_mfe_sum": 0.0,
        "hit_during_hold": {thr: 0 for thr in args.thresholds},
        "hit_post_exit": {thr: 0 for thr in args.thresholds},
        "neg_then_post_hit": {thr: 0 for thr in args.thresholds},
    }
    by_hour: Dict[int, dict] = defaultdict(lambda: {
        "count": 0,
        "neg_trades": 0,
        "mfe_sum": 0.0,
        "mae_sum": 0.0,
        "post_mfe_sum": 0.0,
        "hit_during_hold": {thr: 0 for thr in args.thresholds},
        "hit_post_exit": {thr: 0 for thr in args.thresholds},
        "neg_then_post_hit": {thr: 0 for thr in args.thresholds},
    })

    for tr in trades:
        mfe, mae, _t = _mfe_mae_for_trade(tr, candles_by_day)
        post = _post_exit_mfe(tr, candles_by_day, args.post_min)
        overall["count"] += 1
        overall["mfe_sum"] += mfe
        overall["mae_sum"] += mae
        overall["post_mfe_sum"] += post

        jst_hour = tr.close_time.astimezone(dt.timezone(dt.timedelta(hours=9))).hour if tr.close_time else tr.open_time.astimezone(dt.timezone(dt.timedelta(hours=9))).hour
        agg = by_hour[jst_hour]
        agg["count"] += 1
        agg["mfe_sum"] += mfe
        agg["mae_sum"] += mae
        agg["post_mfe_sum"] += post

        realized_negative = (tr.close_price is not None and ((tr.close_price - tr.entry_price) / PIP) * (1 if tr.units > 0 else -1) < 0)
        if realized_negative:
            overall["neg_trades"] += 1
            agg["neg_trades"] += 1

        for thr in args.thresholds:
            if mfe >= thr:
                overall["hit_during_hold"][thr] += 1
                agg["hit_during_hold"][thr] += 1
            if post >= thr:
                overall["hit_post_exit"][thr] += 1
                agg["hit_post_exit"][thr] += 1
                if realized_negative:
                    overall["neg_then_post_hit"][thr] += 1
                    agg["neg_then_post_hit"][thr] += 1

    def _ratio(n: int, d: int) -> str:
        return "-" if d == 0 else f"{(100.0 * n / d):.1f}%"

    print("=== Overall (last %d days, post %d min) ===" % (args.days, args.post_min))
    print(f"trades={overall['count']} avg_MFE={overall['mfe_sum']/max(1,overall['count']):.2f}p avg_MAE={overall['mae_sum']/max(1,overall['count']):.2f}p post_MFE={overall['post_mfe_sum']/max(1,overall['count']):.2f}p")
    for thr in args.thresholds:
        during = overall["hit_during_hold"][thr]
        post = overall["hit_post_exit"][thr]
        neg_post = overall["neg_then_post_hit"][thr]
        print(f"hit≥{thr:.1f}p during_hold: {during}/{overall['count']} ({_ratio(during, overall['count'])}) | post_exit: {post}/{overall['count']} ({_ratio(post, overall['count'])}) | neg_then_post: {neg_post}/{overall['neg_trades']} ({_ratio(neg_post, overall['neg_trades'])})")

    print("\n=== By JST close hour ===")
    for h in sorted(by_hour):
        v = by_hour[h]
        avg_mfe = v['mfe_sum']/max(1,v['count'])
        avg_mae = v['mae_sum']/max(1,v['count'])
        avg_post = v['post_mfe_sum']/max(1,v['count'])
        print(f"{h:02d}: trades={v['count']:3d} avg_MFE={avg_mfe:.2f}p avg_MAE={avg_mae:.2f}p post_MFE={avg_post:.2f}p")
        parts = []
        for thr in args.thresholds:
            during = v['hit_during_hold'][thr]
            post = v['hit_post_exit'][thr]
            parts.append(f"≥{thr:.1f}p hold:{_ratio(during,v['count'])} post:{_ratio(post,v['count'])}")
        print("    " + " | ".join(parts))


if __name__ == "__main__":
    main()
