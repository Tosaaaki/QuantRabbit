#!/usr/bin/env python3
"""Export an analysis-friendly trade journal from logs/trades.db (local-only).

This is intended for "post-trade journaling" and offline analysis:
- Normalizes common fields (pocket/strategy/confidence/reason/rr/hold_minutes)
- Emits JSONL so it can be grepped, diffed, and loaded into BQ/duckdb later

No LLM calls. Read-only on the source DB.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_naive_iso() -> str:
    # trades.db timestamps can be mixed (with/without timezone);
    # string comparisons work best when we keep the filter naive.
    return _utcnow().replace(tzinfo=None).isoformat()


def _parse_iso(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_json_object(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {}
    raw = str(text).strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _iter_trades(
    db_path: Path,
    *,
    since_ts: str,
    include_open: bool,
    limit: int,
) -> Iterator[sqlite3.Row]:
    if not db_path.exists():
        return iter(())
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    where = "WHERE (entry_time >= ? OR close_time >= ?)"
    params: list[Any] = [since_ts, since_ts]
    if not include_open:
        where += " AND state = 'CLOSED'"
    q = (
        "SELECT ticket_id,pocket,instrument,units,entry_price,close_price,fill_price,"
        "pl_pips,realized_pl,commission,financing,entry_time,close_time,state,strategy,strategy_tag,"
        "client_order_id,entry_thesis,macro_regime,micro_regime "
        f"FROM trades {where} ORDER BY close_time DESC"
    )
    if limit > 0:
        q += " LIMIT ?"
        params.append(int(limit))
    try:
        cur = con.execute(q, tuple(params))
        for row in cur:
            yield row
    finally:
        try:
            con.close()
        except Exception:
            pass


def _normalize_trade(row: sqlite3.Row) -> Dict[str, Any]:
    entry_thesis = _parse_json_object(row["entry_thesis"])
    strategy_tag = row["strategy_tag"] or row["strategy"] or entry_thesis.get("strategy_tag") or entry_thesis.get("strategy")
    pocket = row["pocket"] or "unknown"
    pl_pips = _safe_float(row["pl_pips"]) or 0.0
    entry_time = _parse_iso(row["entry_time"])
    close_time = _parse_iso(row["close_time"])
    hold_minutes = None
    if entry_time and close_time:
        hold_minutes = round((close_time - entry_time).total_seconds() / 60.0, 2)

    sl_pips = _safe_float(entry_thesis.get("sl_pips"))
    tp_pips = _safe_float(entry_thesis.get("tp_pips"))
    rr = None
    if sl_pips and tp_pips and sl_pips > 0:
        rr = round(tp_pips / sl_pips, 4)

    out: Dict[str, Any] = {
        "ticket_id": row["ticket_id"],
        "client_order_id": row["client_order_id"],
        "pocket": pocket,
        "instrument": row["instrument"],
        "strategy_tag": str(strategy_tag or "unknown"),
        "units": _safe_int(row["units"]),
        "entry_time": row["entry_time"],
        "close_time": row["close_time"],
        "hold_minutes": hold_minutes,
        "entry_price": _safe_float(row["entry_price"]),
        "close_price": _safe_float(row["close_price"]),
        "fill_price": _safe_float(row["fill_price"]),
        "pl_pips": round(pl_pips, 5),
        "win": bool(pl_pips > 0),
        "realized_pl": _safe_float(row["realized_pl"]),
        "commission": _safe_float(row["commission"]),
        "financing": _safe_float(row["financing"]),
        "state": row["state"],
        "macro_regime": row["macro_regime"],
        "micro_regime": row["micro_regime"],
        # normalized thesis fields (best-effort; keep raw separately)
        "thesis": {
            "confidence": _safe_int(entry_thesis.get("confidence")),
            "reason": entry_thesis.get("reason"),
            "pattern_tag": entry_thesis.get("pattern_tag"),
            "entry_type": entry_thesis.get("entry_type"),
            "intent": entry_thesis.get("intent"),
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "rr": rr,
            "range_active": entry_thesis.get("range_active"),
            "range_score": _safe_float(entry_thesis.get("range_score")),
            "range_reason": entry_thesis.get("range_reason"),
            "flags": entry_thesis.get("flags"),
        },
        "entry_thesis_raw": entry_thesis,
    }
    return out


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for item in rows:
            fh.write(json.dumps(item, ensure_ascii=True) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Export trade journal JSONL (local-only)")
    ap.add_argument("--trades-db", default=os.getenv("TRADES_DB", "logs/trades.db"))
    ap.add_argument("--out", default="logs/journal/trade_journal.jsonl")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--since", default=None, help="Override filter (ISO string)")
    ap.add_argument("--include-open", action="store_true", help="Include non-CLOSED rows")
    ap.add_argument("--limit", type=int, default=0, help="0=unlimited")
    args = ap.parse_args()

    since_ts = args.since
    if not since_ts:
        since = _utcnow() - timedelta(days=float(args.days))
        since_ts = since.replace(tzinfo=None).isoformat()
    db_path = Path(args.trades_db)
    out_path = Path(args.out)

    normalized = (_normalize_trade(row) for row in _iter_trades(db_path, since_ts=since_ts, include_open=args.include_open, limit=args.limit))
    n = _write_jsonl(out_path, normalized)
    print(
        json.dumps(
            {
                "ok": True,
                "rows": n,
                "since_ts": since_ts,
                "out": str(out_path),
                "trades_db": str(db_path),
                "generated_at": _utcnow_naive_iso(),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

