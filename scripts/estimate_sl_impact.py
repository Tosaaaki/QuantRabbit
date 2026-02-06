#!/usr/bin/env python3
"""
Estimate the impact of adding a hard stop-loss (SL) on historical closed trades.

This script:
- Reads closed trades from a trades DB (default: logs/trades.db)
- Reads tick JSONL (bid/ask) files
- Computes per-trade MAE/MFE (max adverse/favorable excursion) in pips
- Simulates a hypothetical hard SL at given pips (stop at -N pips if MAE>=N)

Notes:
- Uses bid for long exits and ask for short exits (conservative).
- This is a what-if estimate; real fills may differ due to slippage / gaps.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


PIP_DEFAULT = 0.01  # USDJPY


def _parse_iso_to_epoch(value: str) -> Optional[float]:
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _tick_epoch(obj: dict) -> Optional[float]:
    if "ts" in obj:
        ts = obj.get("ts")
        if isinstance(ts, str):
            return _parse_iso_to_epoch(ts)
    ts = obj.get("timestamp")
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        return _parse_iso_to_epoch(ts)
    return None


def _as_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_sl_list(raw: str) -> list[float]:
    out: list[float] = []
    for part in (raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
        except ValueError:
            continue
        if v > 0:
            out.append(v)
    # stable ordering, no duplicates
    return sorted(set(out))


@dataclass(frozen=True)
class Trade:
    ticket_id: str
    pocket: str
    strategy_tag: str
    units: int
    entry_price: float
    pl_pips: float
    entry_ts: float
    close_ts: float


@dataclass
class TradeMetrics:
    trade: Trade
    mae_pips: float
    mfe_pips: float


def _load_closed_trades(
    db_path: Path,
    *,
    instrument: str,
    pocket: Optional[str],
    strategy_contains: Optional[str],
) -> list[Trade]:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ticket_id, pocket, strategy_tag, strategy, units, entry_price, pl_pips, entry_time, close_time
            FROM trades
            WHERE instrument = ?
              AND close_time IS NOT NULL
              AND entry_time IS NOT NULL
              AND ticket_id IS NOT NULL
            ORDER BY entry_time ASC
            """,
            (instrument,),
        )
        out: list[Trade] = []
        for (
            ticket_id,
            pocket_val,
            strategy_tag,
            strategy,
            units,
            entry_price,
            pl_pips,
            entry_time,
            close_time,
        ) in cur.fetchall():
            ticket = str(ticket_id or "").strip()
            if not ticket:
                continue
            pocket_text = str(pocket_val or "unknown").strip().lower() or "unknown"
            if pocket and pocket_text != pocket.strip().lower():
                continue
            tag = (strategy_tag or strategy) if (strategy_tag or strategy) else "unknown"
            tag_text = str(tag).strip() or "unknown"
            if strategy_contains and strategy_contains.strip().lower() not in tag_text.lower():
                continue
            et = _parse_iso_to_epoch(str(entry_time))
            ct = _parse_iso_to_epoch(str(close_time))
            if et is None or ct is None:
                continue
            try:
                units_int = int(units or 0)
            except (TypeError, ValueError):
                continue
            ep = _as_float(entry_price)
            pp = _as_float(pl_pips)
            if ep is None or ep <= 0 or pp is None:
                continue
            if units_int == 0:
                continue
            if ct <= et:
                continue
            out.append(
                Trade(
                    ticket_id=ticket,
                    pocket=pocket_text,
                    strategy_tag=tag_text,
                    units=units_int,
                    entry_price=float(ep),
                    pl_pips=float(pp),
                    entry_ts=float(et),
                    close_ts=float(ct),
                )
            )
        return out
    finally:
        conn.close()


def _load_ticks(path: Path) -> tuple[list[float], list[float], list[float]]:
    times: list[float] = []
    bids: list[float] = []
    asks: list[float] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            ts = _tick_epoch(obj)
            if ts is None:
                continue
            bid = _as_float(obj.get("bid"))
            ask = _as_float(obj.get("ask"))
            if bid is None and ask is None:
                continue
            if bid is None:
                bid = ask
            if ask is None:
                ask = bid
            if bid is None or ask is None:
                continue
            times.append(float(ts))
            bids.append(float(bid))
            asks.append(float(ask))
    return times, bids, asks


def _quantile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    idx = int(round((len(vals) - 1) * q))
    return float(vals[idx])


def _group_key(metrics: TradeMetrics, group_by: str) -> tuple[str, ...]:
    if group_by == "pocket":
        return (metrics.trade.pocket or "unknown",)
    if group_by in {"pocket_strategy", "pocket_strategy_tag"}:
        return (metrics.trade.pocket or "unknown", metrics.trade.strategy_tag or "unknown")
    return ("ALL",)


def _iter_metrics_for_tickfile(
    *,
    tick_path: Path,
    trades: list[Trade],
    seen_ticket_ids: set[str],
    pip: float,
) -> list[TradeMetrics]:
    times, bids, asks = _load_ticks(tick_path)
    if not times:
        return []
    start = times[0]
    end = times[-1]
    selected = [
        t
        for t in trades
        if t.ticket_id not in seen_ticket_ids and t.entry_ts >= start and t.close_ts <= end
    ]
    if not selected:
        return []
    out: list[TradeMetrics] = []
    for tr in selected:
        i0 = bisect_left(times, tr.entry_ts)
        i1 = bisect_right(times, tr.close_ts)
        if i1 <= i0:
            continue
        if tr.units > 0:
            worst = min(bids[i0:i1])
            best = max(bids[i0:i1])
            mae = max(0.0, (tr.entry_price - worst) / pip)
            mfe = max(0.0, (best - tr.entry_price) / pip)
        else:
            worst = max(asks[i0:i1])
            best = min(asks[i0:i1])
            mae = max(0.0, (worst - tr.entry_price) / pip)
            mfe = max(0.0, (tr.entry_price - best) / pip)
        out.append(TradeMetrics(trade=tr, mae_pips=float(mae), mfe_pips=float(mfe)))
        seen_ticket_ids.add(tr.ticket_id)
    return out


def _format_opt(val: Optional[float]) -> str:
    if val is None:
        return "-"
    return f"{val:.2f}"


def _print_group_summary(
    *,
    title: str,
    items: list[TradeMetrics],
    sl_pips: list[float],
):
    if not items:
        return
    base_pnl = sum(m.trade.pl_pips for m in items)
    winners = [m for m in items if m.trade.pl_pips > 0]
    win_mae = [m.mae_pips for m in winners]
    print(f"\n[{title}] trades={len(items)} winners={len(winners)} pnl={base_pnl:.1f}p")
    print(
        "  winner_MAE_pips:"
        f" p50={_format_opt(_quantile(win_mae, 0.50))}"
        f" p90={_format_opt(_quantile(win_mae, 0.90))}"
        f" p95={_format_opt(_quantile(win_mae, 0.95))}"
        f" p99={_format_opt(_quantile(win_mae, 0.99))}"
    )
    for th in sl_pips:
        stopped = [m for m in items if m.mae_pips >= th - 1e-9]
        winners_stopped = [m for m in winners if m.mae_pips >= th - 1e-9]
        new_pnl = sum((-th if m.mae_pips >= th - 1e-9 else m.trade.pl_pips) for m in items)
        stop_rate = (len(stopped) / len(items)) if items else 0.0
        win_stop_rate = (len(winners_stopped) / len(winners)) if winners else 0.0
        print(
            f"  SL {th:>5.1f}p:"
            f" stop={len(stopped):>4}/{len(items):<4} ({stop_rate:>5.1%})"
            f" winners_stopped={len(winners_stopped):>4}/{len(winners):<4} ({win_stop_rate:>5.1%})"
            f" pnl={base_pnl:>8.1f}p->{new_pnl:>8.1f}p (Δ{new_pnl - base_pnl:>+7.1f})"
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trades-db",
        default="logs/trades.db",
        help="Path to trades.db (default: logs/trades.db)",
    )
    parser.add_argument(
        "--ticks",
        action="append",
        required=True,
        help="Tick JSONL path (repeatable). Supports keys: ts or timestamp + bid/ask.",
    )
    parser.add_argument("--instrument", default="USD_JPY")
    parser.add_argument("--pip", type=float, default=PIP_DEFAULT)
    parser.add_argument(
        "--sl-pips",
        default="3,4,5,6,8,10,12,15,20",
        help="Comma-separated SL thresholds in pips to simulate.",
    )
    parser.add_argument("--pocket", default=None, help="Filter pocket (e.g. scalp, macro, micro)")
    parser.add_argument("--strategy-contains", default=None, help="Filter strategy_tag contains token")
    parser.add_argument(
        "--group-by",
        default="pocket",
        choices=["all", "pocket", "pocket_strategy"],
        help="Aggregation level (default: pocket)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    tick_paths = [Path(p).expanduser() for p in args.ticks]
    for p in tick_paths:
        if not p.exists():
            raise SystemExit(f"tick file not found: {p}")

    pip = float(args.pip)
    if pip <= 0:
        raise SystemExit("--pip must be > 0")
    sl_pips = _parse_sl_list(args.sl_pips)
    if not sl_pips:
        raise SystemExit("--sl-pips must contain at least one positive value")

    trades_db = Path(args.trades_db).expanduser()
    if not trades_db.exists():
        raise SystemExit(f"trades db not found: {trades_db}")

    trades = _load_closed_trades(
        trades_db,
        instrument=str(args.instrument),
        pocket=args.pocket,
        strategy_contains=args.strategy_contains,
    )
    if not trades:
        print("no trades matched")
        return 0

    metrics_all: list[TradeMetrics] = []
    seen: set[str] = set()
    for tick_path in tick_paths:
        metrics = _iter_metrics_for_tickfile(
            tick_path=tick_path,
            trades=trades,
            seen_ticket_ids=seen,
            pip=pip,
        )
        if metrics:
            metrics_all.extend(metrics)
    if not metrics_all:
        print("no trades fell within provided tick time ranges")
        return 0

    group_by = str(args.group_by)
    if group_by == "all":
        group_by = "all"  # alias

    groups: dict[tuple[str, ...], list[TradeMetrics]] = {}
    for m in metrics_all:
        key = _group_key(m, group_by)
        groups.setdefault(key, []).append(m)

    if group_by == "all":
        _print_group_summary(title="ALL", items=metrics_all, sl_pips=sl_pips)
        return 0

    for key in sorted(groups.keys()):
        title = "/".join(key)
        _print_group_summary(title=title, items=groups[key], sl_pips=sl_pips)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

