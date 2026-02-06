#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Optional


PIP = 0.01  # USD/JPY


def _parse_iso(ts: str) -> dt.datetime:
    # Example: 2026-02-04T04:50:10.046568+00:00
    return dt.datetime.fromisoformat(ts)


def _parse_oanda_ts(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    text = str(ts).strip()
    if not text:
        return None
    # Example: 2026-02-04T05:01:48.423936297Z (nanoseconds + Z)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        head, tail = text.split(".", 1)
        frac, rest = tail.split("+", 1) if "+" in tail else (tail, "")
        frac = (frac[:6]).ljust(6, "0")  # microseconds
        text = f"{head}.{frac}"
        if rest:
            text += f"+{rest}"
    try:
        return dt.datetime.fromisoformat(text)
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _percentile(sorted_values: list[float], p: float) -> Optional[float]:
    if not sorted_values:
        return None
    if p <= 0:
        return float(sorted_values[0])
    if p >= 1:
        return float(sorted_values[-1])
    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    lo = float(sorted_values[f])
    hi = float(sorted_values[c])
    return lo + (hi - lo) * (k - f)


def _fmt(value: Optional[float], *, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.0f}ms"


@dataclass(slots=True)
class EntryRow:
    ts: dt.datetime
    client_order_id: str
    pocket: str
    instrument: str
    side: str
    units: int
    executed_price: Optional[float]
    sl_price: Optional[float]
    tp_price: Optional[float]
    strategy_tag: str
    quote_bid: Optional[float]
    quote_ask: Optional[float]
    quote_mid: Optional[float]
    quote_spread_pips: Optional[float]
    quote_ts: Optional[dt.datetime]
    latency_submit_ms: Optional[float]
    latency_preflight_ms: Optional[float]
    slip_vs_side_pips: Optional[float]
    cost_vs_mid_pips: Optional[float]
    eff_tp_pips: Optional[float]
    eff_sl_pips: Optional[float]
    thesis_tp_pips: Optional[float]
    thesis_sl_pips: Optional[float]


def _chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _load_events(con: sqlite3.Connection, client_ids: list[str]) -> dict[str, dict[str, list[dt.datetime]]]:
    out: dict[str, dict[str, list[dt.datetime]]] = defaultdict(lambda: defaultdict(list))
    if not client_ids:
        return out
    # SQLite default max variables is typically 999; stay below.
    for chunk in _chunked(client_ids, 800):
        placeholders = ",".join(["?"] * len(chunk))
        rows = con.execute(
            f"""
            SELECT client_order_id, status, ts
            FROM orders
            WHERE client_order_id IN ({placeholders})
              AND status IN ('preflight_start','submit_attempt','filled')
            """,
            chunk,
        ).fetchall()
        for cid, status, ts in rows:
            if not cid or not status or not ts:
                continue
            try:
                out[str(cid)][str(status)].append(_parse_iso(str(ts)))
            except Exception:
                continue
    return out


def _strategy_from_payload(payload: dict) -> str:
    entry_thesis = payload.get("entry_thesis")
    if isinstance(entry_thesis, dict):
        tag = entry_thesis.get("strategy_tag") or entry_thesis.get("strategy")
        if tag:
            return str(tag)
    # Fallback: some workers may only set meta.
    meta = payload.get("meta")
    if isinstance(meta, dict):
        tag = meta.get("strategy_tag") or meta.get("strategy")
        if tag:
            return str(tag)
    return "unknown"


def _pips_diff(price_a: float, price_b: float) -> float:
    return (price_a - price_b) / PIP


def _summarize(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p95": None, "max": None, "min": None}
    values_sorted = sorted(values)
    return {
        "n": float(len(values_sorted)),
        "mean": float(statistics.mean(values_sorted)),
        "p50": _percentile(values_sorted, 0.50),
        "p90": _percentile(values_sorted, 0.90),
        "p95": _percentile(values_sorted, 0.95),
        "max": float(values_sorted[-1]),
        "min": float(values_sorted[0]),
    }


def _print_summary(name: str, values: list[float], *, digits: int = 3, unit: str = "") -> None:
    s = _summarize(values)
    n = int(s["n"] or 0)
    if n <= 0:
        print(f"{name}: n=0")
        return
    print(
        f"{name}: n={n} mean={_fmt(s['mean'], digits=digits)}"
        f" p50={_fmt(s['p50'], digits=digits)} p90={_fmt(s['p90'], digits=digits)}"
        f" p95={_fmt(s['p95'], digits=digits)} max={_fmt(s['max'], digits=digits)}{unit}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze entry execution precision from logs/orders.db")
    ap.add_argument("--db", default="logs/orders.db", help="Path to orders.db")
    ap.add_argument("--limit", type=int, default=300, help="Number of latest filled entry orders to analyze")
    ap.add_argument("--pocket", default=None, help="Filter pocket (e.g., scalp)")
    ap.add_argument("--strategy", default=None, help="Filter strategy_tag (e.g., TickImbalance)")
    ap.add_argument("--show-worst", type=int, default=12, help="Show worst slippage samples")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    where = [
        "status='filled'",
        "client_order_id IS NOT NULL",
        "pocket IS NOT NULL",
        "pocket != ''",
        "instrument IS NOT NULL",
        "instrument != ''",
        "side IN ('buy','sell')",
    ]
    params: list[Any] = []
    if args.pocket:
        where.append("LOWER(pocket)=LOWER(?)")
        params.append(args.pocket)
    sql = f"""
        SELECT ts, pocket, instrument, side, units, sl_price, tp_price, client_order_id, executed_price, request_json
        FROM orders
        WHERE {' AND '.join(where)}
        ORDER BY ts DESC
        LIMIT ?
    """
    params.append(int(args.limit))
    filled = con.execute(sql, params).fetchall()
    client_ids = [str(r["client_order_id"]) for r in filled if r["client_order_id"]]
    events = _load_events(con, client_ids)

    rows: list[EntryRow] = []
    missing_quote = 0
    missing_executed = 0

    for r in filled:
        cid = str(r["client_order_id"] or "").strip()
        if not cid:
            continue
        ts_raw = str(r["ts"])
        try:
            ts = _parse_iso(ts_raw)
        except Exception:
            continue
        payload: dict = {}
        try:
            payload = json.loads(r["request_json"] or "{}") or {}
        except Exception:
            payload = {}
        quote = payload.get("quote") if isinstance(payload, dict) else None
        quote = quote if isinstance(quote, dict) else {}
        bid = _as_float(quote.get("bid"))
        ask = _as_float(quote.get("ask"))
        mid = _as_float(quote.get("mid"))
        spread_pips = _as_float(quote.get("spread_pips"))
        quote_ts = _parse_oanda_ts(quote.get("ts"))
        if bid is None or ask is None:
            missing_quote += 1
        strategy_tag = _strategy_from_payload(payload)
        if args.strategy and strategy_tag.lower() != str(args.strategy).lower():
            continue

        side = str(r["side"] or "").lower()
        executed = _as_float(r["executed_price"])
        if executed is None:
            missing_executed += 1
        sl_price = _as_float(r["sl_price"])
        tp_price = _as_float(r["tp_price"])

        thesis_tp_pips = None
        thesis_sl_pips = None
        entry_thesis = payload.get("entry_thesis")
        if isinstance(entry_thesis, dict):
            thesis_tp_pips = _as_float(entry_thesis.get("tp_pips") or entry_thesis.get("target_tp_pips"))
            thesis_sl_pips = _as_float(entry_thesis.get("sl_pips") or entry_thesis.get("loss_guard_pips"))

        preflight_ts = None
        submit_ts = None
        ev = events.get(cid) or {}
        # Use earliest preflight; use latest submit_attempt before fill.
        if ev.get("preflight_start"):
            preflight_ts = min(ev["preflight_start"])
        if ev.get("submit_attempt"):
            submit_ts = max([t for t in ev["submit_attempt"] if t <= ts] or ev["submit_attempt"])

        latency_preflight_ms = None
        latency_submit_ms = None
        if preflight_ts is not None:
            latency_preflight_ms = (ts - preflight_ts).total_seconds() * 1000.0
        if submit_ts is not None:
            latency_submit_ms = (ts - submit_ts).total_seconds() * 1000.0

        slip_vs_side = None
        cost_vs_mid = None
        if executed is not None and bid is not None and ask is not None:
            if side == "buy":
                slip_vs_side = _pips_diff(executed, ask)
            elif side == "sell":
                slip_vs_side = _pips_diff(bid, executed)
        if executed is not None and mid is not None:
            if side == "buy":
                cost_vs_mid = _pips_diff(executed, mid)
            elif side == "sell":
                cost_vs_mid = _pips_diff(mid, executed)

        eff_tp_pips = None
        if executed is not None and tp_price is not None:
            if side == "buy":
                eff_tp_pips = _pips_diff(tp_price, executed)
            elif side == "sell":
                eff_tp_pips = _pips_diff(executed, tp_price)
        eff_sl_pips = None
        if executed is not None and sl_price is not None:
            if side == "buy":
                eff_sl_pips = _pips_diff(executed, sl_price)
            elif side == "sell":
                eff_sl_pips = _pips_diff(sl_price, executed)

        rows.append(
            EntryRow(
                ts=ts,
                client_order_id=cid,
                pocket=str(r["pocket"] or ""),
                instrument=str(r["instrument"] or ""),
                side=side,
                units=int(r["units"] or 0),
                executed_price=executed,
                sl_price=sl_price,
                tp_price=tp_price,
                strategy_tag=strategy_tag,
                quote_bid=bid,
                quote_ask=ask,
                quote_mid=mid,
                quote_spread_pips=spread_pips,
                quote_ts=quote_ts,
                latency_submit_ms=latency_submit_ms,
                latency_preflight_ms=latency_preflight_ms,
                slip_vs_side_pips=slip_vs_side,
                cost_vs_mid_pips=cost_vs_mid,
                eff_tp_pips=eff_tp_pips,
                eff_sl_pips=eff_sl_pips,
                thesis_tp_pips=thesis_tp_pips,
                thesis_sl_pips=thesis_sl_pips,
            )
        )

    if not rows:
        print("No rows matched.")
        return 2

    rows_sorted = sorted(rows, key=lambda x: x.ts)
    print(f"db={args.db} sample={len(rows)} range={rows_sorted[0].ts.isoformat()}..{rows_sorted[-1].ts.isoformat()}")
    if args.pocket:
        print(f"filter pocket={args.pocket}")
    if args.strategy:
        print(f"filter strategy={args.strategy}")
    print(f"missing quote={missing_quote}/{len(rows)} missing executed_price={missing_executed}/{len(rows)}")

    slip_vals = [r.slip_vs_side_pips for r in rows if r.slip_vs_side_pips is not None]
    cost_vals = [r.cost_vs_mid_pips for r in rows if r.cost_vs_mid_pips is not None]
    spread_vals = [r.quote_spread_pips for r in rows if r.quote_spread_pips is not None]
    lat_submit_vals = [r.latency_submit_ms for r in rows if r.latency_submit_ms is not None]
    lat_pre_vals = [r.latency_preflight_ms for r in rows if r.latency_preflight_ms is not None]
    tp_vals = [r.eff_tp_pips for r in rows if r.eff_tp_pips is not None]
    sl_vals = [r.eff_sl_pips for r in rows if r.eff_sl_pips is not None]

    print("")
    _print_summary("slip_vs_side_pips", slip_vals, digits=3)
    _print_summary("cost_vs_mid_pips", cost_vals, digits=3)
    _print_summary("spread_pips", spread_vals, digits=3)
    _print_summary("latency_submit", lat_submit_vals, digits=1, unit="")
    _print_summary("latency_preflight", lat_pre_vals, digits=1, unit="")
    _print_summary("effective_tp_pips", tp_vals, digits=2)
    _print_summary("effective_sl_pips", sl_vals, digits=2)

    # Per-strategy summary
    by_strategy: dict[str, list[EntryRow]] = defaultdict(list)
    for r in rows:
        by_strategy[r.strategy_tag].append(r)
    print("")
    print("by_strategy (count desc):")
    for tag, group in sorted(by_strategy.items(), key=lambda kv: len(kv[1]), reverse=True)[:12]:
        s_slip = [x.slip_vs_side_pips for x in group if x.slip_vs_side_pips is not None]
        s_lat = [x.latency_submit_ms for x in group if x.latency_submit_ms is not None]
        s_spread = [x.quote_spread_pips for x in group if x.quote_spread_pips is not None]
        s_tp = [x.eff_tp_pips for x in group if x.eff_tp_pips is not None]
        s_sl = [x.eff_sl_pips for x in group if x.eff_sl_pips is not None]
        p95 = None
        if s_slip:
            p95 = _percentile(sorted(s_slip), 0.95)
        print(
            f"- {tag}: n={len(group)}"
            f" slip_mean={_fmt(statistics.mean(s_slip) if s_slip else None, digits=3)}"
            f" slip_p95={_fmt(p95, digits=3)}"
            f" spread_mean={_fmt(statistics.mean(s_spread) if s_spread else None, digits=3)}"
            f" lat_p50={_fmt_ms(_percentile(sorted(s_lat), 0.50) if s_lat else None)}"
            f" tp_mean={_fmt(statistics.mean(s_tp) if s_tp else None, digits=2)}"
            f" sl_mean={_fmt(statistics.mean(s_sl) if s_sl else None, digits=2)}"
        )

    # Worst slippage samples
    if args.show_worst > 0 and slip_vals:
        print("")
        print(f"worst_slip_samples (top {args.show_worst}):")
        worst = sorted(
            [r for r in rows if r.slip_vs_side_pips is not None],
            key=lambda x: float(x.slip_vs_side_pips or 0.0),
            reverse=True,
        )[: int(args.show_worst)]
        for r in worst:
            print(
                f"- {r.ts.isoformat()} {r.strategy_tag} {r.side} slip={_fmt(r.slip_vs_side_pips, digits=3)}p"
                f" spread={_fmt(r.quote_spread_pips, digits=3)}p"
                f" lat={_fmt_ms(r.latency_submit_ms)}"
                f" id={r.client_order_id}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

