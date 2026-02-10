#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


def _parse_time(text: str) -> dt.datetime:
    # trades.db uses ISO8601 with timezone (e.g. 2026-02-10T06:37:15.069697+00:00).
    t = str(text).strip()
    if not t:
        raise ValueError("empty timestamp")
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    return dt.datetime.fromisoformat(t)


def _pip_size(instrument: str) -> float:
    inst = str(instrument or "").upper()
    # FX heuristic: JPY pairs typically quote to 2 decimals.
    return 0.01 if inst.endswith("JPY") else 0.0001


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _format_dt(value: Optional[dt.datetime], tz: Optional[str]) -> str:
    if value is None:
        return ""
    if tz and ZoneInfo is not None:
        try:
            value = value.astimezone(ZoneInfo(tz))
        except Exception:
            pass
    return value.isoformat()


@dataclass(frozen=True, slots=True)
class TradeRow:
    ticket_id: str
    pocket: str
    instrument: str
    units: int
    entry_price: float
    open_time: dt.datetime
    close_time: Optional[dt.datetime]
    close_reason: Optional[str]
    pl_pips: Optional[float]
    realized_pl: Optional[float]
    strategy_tag: Optional[str]
    client_order_id: Optional[str]
    entry_thesis_raw: Optional[str]


def _load_trade(trades_db: Path, ticket_id: str) -> TradeRow:
    conn = sqlite3.connect(str(trades_db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT
          ticket_id,pocket,instrument,units,entry_price,open_time,close_time,close_reason,
          pl_pips,realized_pl,strategy_tag,client_order_id,entry_thesis
        FROM trades
        WHERE ticket_id = ?
        """,
        (str(ticket_id),),
    ).fetchone()
    if row is None:
        raise SystemExit(f"[trade_sl_perspectives] ticket not found: {ticket_id}")
    return TradeRow(
        ticket_id=str(row["ticket_id"]),
        pocket=str(row["pocket"] or ""),
        instrument=str(row["instrument"] or ""),
        units=int(row["units"] or 0),
        entry_price=float(row["entry_price"] or 0.0),
        open_time=_parse_time(str(row["open_time"])),
        close_time=_parse_time(str(row["close_time"])) if row["close_time"] else None,
        close_reason=str(row["close_reason"]) if row["close_reason"] else None,
        pl_pips=_safe_float(row["pl_pips"]),
        realized_pl=_safe_float(row["realized_pl"]),
        strategy_tag=str(row["strategy_tag"]) if row["strategy_tag"] else None,
        client_order_id=str(row["client_order_id"]) if row["client_order_id"] else None,
        entry_thesis_raw=str(row["entry_thesis"]) if row["entry_thesis"] else None,
    )


def _parse_entry_thesis(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _iter_ticks(paths: list[Path], *, instrument: str) -> Iterable[tuple[dt.datetime, float, float]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("instrument") != instrument:
                    continue
                ts = obj.get("ts")
                bid = obj.get("bid")
                ask = obj.get("ask")
                if ts is None or bid is None or ask is None:
                    continue
                try:
                    t = _parse_time(str(ts))
                    b = float(bid)
                    a = float(ask)
                except Exception:
                    continue
                yield t, b, a


@dataclass(slots=True)
class TickMetrics:
    sl_hit_time: Optional[dt.datetime] = None
    tp_touch_time: Optional[dt.datetime] = None
    post_close_tp_touch_time: Optional[dt.datetime] = None
    min_by_h: dict[int, float] = None  # type: ignore[assignment]
    max_by_h: dict[int, float] = None  # type: ignore[assignment]
    sl_crossings_by_h: dict[int, int] = None  # type: ignore[assignment]
    sl_near_ticks_by_h: dict[int, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.min_by_h = {}
        self.max_by_h = {}
        self.sl_crossings_by_h = {}
        self.sl_near_ticks_by_h = {}


def _compute_tick_metrics(
    *,
    trade: TradeRow,
    sl_pips: Optional[float],
    tp_pips: Optional[float],
    ticks: list[Path],
    horizons: list[int],
    post_close_sec: int,
    sl_near_band_pips: float,
) -> TickMetrics:
    pip = _pip_size(trade.instrument)
    is_long = trade.units > 0
    sl_price = None
    tp_price = None
    if sl_pips is not None:
        sl_price = trade.entry_price - (sl_pips * pip) if is_long else trade.entry_price + (sl_pips * pip)
    if tp_pips is not None:
        tp_price = trade.entry_price + (tp_pips * pip) if is_long else trade.entry_price - (tp_pips * pip)

    open_time = trade.open_time
    max_h = max(horizons) if horizons else 0
    open_end = open_time + dt.timedelta(seconds=max_h)
    close_end = None
    if trade.close_time is not None:
        close_end = trade.close_time + dt.timedelta(seconds=max(0, int(post_close_sec)))
    end_time = max(open_end, close_end) if close_end is not None else open_end

    m = TickMetrics()
    prev_sl_side_by_h: dict[int, Optional[int]] = {h: None for h in horizons}

    for ts, bid, ask in _iter_ticks(ticks, instrument=trade.instrument):
        if ts < open_time:
            continue
        if ts > end_time:
            break
        px = bid if is_long else ask

        # SL-hit / TP-touch within the entry horizons window.
        if ts <= open_end:
            if sl_price is not None and m.sl_hit_time is None:
                if (is_long and px <= sl_price) or ((not is_long) and px >= sl_price):
                    m.sl_hit_time = ts
            if tp_price is not None and m.tp_touch_time is None:
                if (is_long and px >= tp_price) or ((not is_long) and px <= tp_price):
                    m.tp_touch_time = ts

            sec_open = (ts - open_time).total_seconds()
            if sec_open >= 0:
                for h in horizons:
                    if sec_open <= h:
                        cur_min = m.min_by_h.get(h)
                        cur_max = m.max_by_h.get(h)
                        m.min_by_h[h] = px if cur_min is None else min(cur_min, px)
                        m.max_by_h[h] = px if cur_max is None else max(cur_max, px)

                        if sl_price is not None:
                            cur_side = 1 if px > sl_price else 0
                            prev_side = prev_sl_side_by_h.get(h)
                            if prev_side is not None and cur_side != prev_side:
                                m.sl_crossings_by_h[h] = int(m.sl_crossings_by_h.get(h, 0)) + 1
                            prev_sl_side_by_h[h] = cur_side
                            if abs(px - sl_price) / pip <= max(0.0, float(sl_near_band_pips)):
                                m.sl_near_ticks_by_h[h] = int(m.sl_near_ticks_by_h.get(h, 0)) + 1

        # Post-close TP-touch (detect "exit too early").
        if (
            trade.close_time is not None
            and m.post_close_tp_touch_time is None
            and tp_price is not None
            and close_end is not None
            and trade.close_time < ts <= close_end
        ):
            if (is_long and px >= tp_price) or ((not is_long) and px <= tp_price):
                m.post_close_tp_touch_time = ts

    return m


def _load_orders(
    orders_db: Path,
    *,
    client_order_id: str,
) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(orders_db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT ts,status,attempt,ticket_id,executed_price,error_code,error_message,request_json
        FROM orders
        WHERE client_order_id = ?
        ORDER BY ts ASC
        """,
        (client_order_id,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        rec: dict[str, Any] = {k: r[k] for k in r.keys()}
        req = rec.get("request_json")
        if isinstance(req, str) and req.strip():
            try:
                rec["request"] = json.loads(req)
            except Exception:
                rec["request"] = None
        out.append(rec)
    return out


def _load_exit_composite_blocks(
    metrics_db: Path,
    *,
    since: dt.datetime,
    until: dt.datetime,
) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(metrics_db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT ts,value,tags
        FROM metrics
        WHERE metric = 'exit_composite_block'
          AND ts >= ?
          AND ts <= ?
        ORDER BY ts ASC
        """,
        (since.isoformat(), until.isoformat()),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        rec = {"ts": r["ts"], "value": r["value"], "tags": None}
        tags = r["tags"]
        if isinstance(tags, str) and tags.strip():
            try:
                rec["tags"] = json.loads(tags)
            except Exception:
                rec["tags"] = {"raw": tags}
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-perspective SL/entry diagnostics for a single trade ticket.")
    ap.add_argument("--ticket", required=True, help="ticket_id (e.g. 316549)")
    ap.add_argument("--trades-db", type=Path, default=Path("logs/trades.db"))
    ap.add_argument("--orders-db", type=Path, default=None, help="Optional orders.db to include close/open trace")
    ap.add_argument("--metrics-db", type=Path, default=None, help="Optional metrics.db to include exit_composite blocks")
    ap.add_argument("--ticks", type=Path, action="append", default=[], help="Tick JSONL path (repeatable; chronological order)")
    ap.add_argument("--horizons-sec", default="120,300,600", help="Comma-separated horizons in seconds (default: 120,300,600)")
    ap.add_argument("--post-close-sec", type=int, default=600, help="Seconds after close to check post-close TP touch")
    ap.add_argument("--sl-near-band-pips", type=float, default=0.2, help="Band around SL line to count 'near-SL' ticks")
    ap.add_argument("--tz", default="Asia/Tokyo", help="Timezone for extra timestamp column (default: Asia/Tokyo)")
    ap.add_argument("--out-tsv", type=Path, default=None, help="Write a one-line TSV summary to this path")
    args = ap.parse_args()

    trade = _load_trade(args.trades_db, args.ticket)
    thesis = _parse_entry_thesis(trade.entry_thesis_raw)

    sl_pips = _safe_float(thesis.get("sl_pips"))
    tp_pips = _safe_float(thesis.get("tp_pips"))
    direction = "Long" if trade.units > 0 else "Short"

    print(f"ticket_id: {trade.ticket_id}")
    print(f"strategy_tag: {trade.strategy_tag or ''}")
    print(f"pocket: {trade.pocket}  instrument: {trade.instrument}  dir: {direction}  units: {trade.units}")
    print(f"entry: {trade.entry_price:.3f}")
    if trade.close_time is not None:
        print(
            f"close: {_format_dt(trade.close_time, None)}  reason: {trade.close_reason or ''}  "
            f"pl_pips: {_format_float(trade.pl_pips, 2)}  realized_pl: {_format_float(trade.realized_pl, 2)}"
        )
    else:
        print("close: (OPEN)")

    print(f"open_time_utc: {_format_dt(trade.open_time, None)}")
    print(f"open_time_{args.tz}: {_format_dt(trade.open_time, args.tz)}")
    if trade.close_time is not None:
        print(f"close_time_{args.tz}: {_format_dt(trade.close_time, args.tz)}")

    print("")
    print("entry_thesis (key fields):")
    keys = [
        "confidence",
        "reason",
        "sl_pips",
        "tp_pips",
        "range_active",
        "range_score",
        "range_mode",
        "macro_regime",
        "micro_regime",
        "rsi",
        "stoch_rsi",
        "adx",
        "atr_pips",
        "bbw",
        "macd_hist_pips",
        "vwap_gap",
        "ema_slope_10_pips",
    ]
    for k in keys:
        if k in thesis:
            print(f"  {k}: {thesis.get(k)}")

    horizons = []
    for token in str(args.horizons_sec).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            horizons.append(max(1, int(float(token))))
        except Exception:
            continue
    horizons = sorted(set(horizons))

    tick_metrics = None
    if args.ticks:
        tick_metrics = _compute_tick_metrics(
            trade=trade,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            ticks=list(args.ticks),
            horizons=horizons,
            post_close_sec=int(args.post_close_sec),
            sl_near_band_pips=float(args.sl_near_band_pips),
        )

        pip = _pip_size(trade.instrument)
        print("")
        print("ticks (trigger-side, derived):")
        print(f"  sl_pips: {_format_float(sl_pips, 2)}  tp_pips: {_format_float(tp_pips, 2)}  pip: {pip}")
        if tick_metrics.sl_hit_time is not None:
            sec = (tick_metrics.sl_hit_time - trade.open_time).total_seconds()
            print(f"  sl_hit_s: {_format_float(sec, 0)}  sl_hit_{args.tz}: {_format_dt(tick_metrics.sl_hit_time, args.tz)}")
        if tick_metrics.tp_touch_time is not None:
            sec = (tick_metrics.tp_touch_time - trade.open_time).total_seconds()
            print(f"  tp_touch_s: {_format_float(sec, 0)}  tp_touch_{args.tz}: {_format_dt(tick_metrics.tp_touch_time, args.tz)}")
        if trade.close_time is not None and tick_metrics.post_close_tp_touch_time is not None:
            sec = (tick_metrics.post_close_tp_touch_time - trade.close_time).total_seconds()
            print(f"  post_close_tp_touch_s: {_format_float(sec, 0)}")

        for h in horizons:
            min_px = tick_metrics.min_by_h.get(h)
            max_px = tick_metrics.max_by_h.get(h)
            mae = None
            mfe = None
            if min_px is not None and max_px is not None:
                if trade.units > 0:
                    mae = (trade.entry_price - min_px) / pip
                    mfe = (max_px - trade.entry_price) / pip
                else:
                    mae = (max_px - trade.entry_price) / pip
                    mfe = (trade.entry_price - min_px) / pip
            crossings = tick_metrics.sl_crossings_by_h.get(h)
            near_ticks = tick_metrics.sl_near_ticks_by_h.get(h)
            print(
                f"  horizon={h:>4}s  mae={_format_float(mae, 2):>6}  mfe={_format_float(mfe, 2):>6}  "
                f"sl_crossings={crossings if crossings is not None else ''}  "
                f"sl_near_ticks={near_ticks if near_ticks is not None else ''}"
            )

    if trade.client_order_id and args.orders_db:
        print("")
        print("orders (client_order_id trace):")
        rows = _load_orders(args.orders_db, client_order_id=trade.client_order_id)
        for r in rows:
            ts = r.get("ts")
            status = r.get("status")
            executed_price = r.get("executed_price")
            err = r.get("error_code") or ""
            req = r.get("request") if isinstance(r.get("request"), dict) else {}
            exit_reason = req.get("exit_reason") if isinstance(req, dict) else None
            note = f" exit_reason={exit_reason}" if exit_reason else ""
            print(f"  {ts}  {status}  price={executed_price or ''}  {err}{note}")

    if args.metrics_db and trade.close_time is not None:
        print("")
        print("metrics (exit_composite_block around close):")
        since = trade.close_time - dt.timedelta(seconds=120)
        until = trade.close_time + dt.timedelta(seconds=30)
        blocks = _load_exit_composite_blocks(args.metrics_db, since=since, until=until)
        for b in blocks:
            tags = b.get("tags") if isinstance(b.get("tags"), dict) else {}
            reason = tags.get("reason") if isinstance(tags, dict) else None
            side = tags.get("side") if isinstance(tags, dict) else None
            min_score = tags.get("min_score") if isinstance(tags, dict) else None
            print(
                f"  {b.get('ts')}  score={_format_float(_safe_float(b.get('value')), 1)}  "
                f"reason={reason or ''} side={side or ''} min_score={min_score or ''}"
            )

    if args.out_tsv:
        args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ticket_id": trade.ticket_id,
            "strategy_tag": trade.strategy_tag or "",
            "dir": direction,
            "units": str(trade.units),
            "instrument": trade.instrument,
            "entry_price": f"{trade.entry_price:.3f}",
            "open_time_utc": trade.open_time.isoformat(),
            "close_time_utc": trade.close_time.isoformat() if trade.close_time else "",
            "close_reason": trade.close_reason or "",
            "pl_pips": _format_float(trade.pl_pips, 2),
            "sl_pips": _format_float(sl_pips, 2),
            "tp_pips": _format_float(tp_pips, 2),
        }
        if tick_metrics is not None:
            row["sl_hit_s"] = _format_float((tick_metrics.sl_hit_time - trade.open_time).total_seconds(), 0) if tick_metrics.sl_hit_time else ""
            row["tp_touch_s"] = _format_float((tick_metrics.tp_touch_time - trade.open_time).total_seconds(), 0) if tick_metrics.tp_touch_time else ""
            row["post_close_tp_touch_s"] = _format_float((tick_metrics.post_close_tp_touch_time - trade.close_time).total_seconds(), 0) if (trade.close_time and tick_metrics.post_close_tp_touch_time) else ""
            for h in horizons:
                row[f"sl_crossings_{h}s"] = str(tick_metrics.sl_crossings_by_h.get(h, ""))
                row[f"sl_near_ticks_{h}s"] = str(tick_metrics.sl_near_ticks_by_h.get(h, ""))
        # deterministic header order (TSV)
        header = list(row.keys())
        with args.out_tsv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, delimiter="\t")
            w.writeheader()
            w.writerow(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

