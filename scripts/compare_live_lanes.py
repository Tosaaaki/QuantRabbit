#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


@dataclass(slots=True)
class LaneSummary:
    net_jpy: float = 0.0
    trades: int = 0
    wins: int = 0
    sum_pips: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        win_rate = float(self.wins / self.trades) if self.trades > 0 else 0.0
        expectancy_pips = float(self.sum_pips / self.trades) if self.trades > 0 else 0.0
        return {
            "net_jpy": float(self.net_jpy),
            "trades": int(self.trades),
            "win_rate": win_rate,
            "expectancy_pips": expectancy_pips,
        }


def _parse_ts_epoch(raw: object) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        pass
    if "." not in text:
        return None
    try:
        head, tail = text.split(".", 1)
        sign_pos = max(tail.find("+"), tail.find("-"))
        if sign_pos >= 0:
            frac = tail[:sign_pos]
            zone = tail[sign_pos:]
            frac = (frac[:6]).ljust(6, "0")
            text2 = f"{head}.{frac}{zone}"
        else:
            frac = (tail[:6]).ljust(6, "0")
            text2 = f"{head}.{frac}"
        return datetime.fromisoformat(text2).timestamp()
    except Exception:
        return None


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_local_lane(
    *,
    local_log: Path,
    cutoff_epoch: float,
    local_prefix: str,
) -> LaneSummary:
    summary = LaneSummary()
    if not local_log.exists():
        return summary

    trade_to_client: dict[str, str] = {}
    with local_log.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text or not text.startswith("{"):
                continue
            try:
                row = json.loads(text)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            msg = str(row.get("msg") or "")
            if msg == "trade_opened":
                trade_id = str(row.get("trade_id") or "").strip()
                client_id = str(row.get("client_id") or "").strip()
                if trade_id and client_id:
                    trade_to_client[trade_id] = client_id
                continue
            if msg != "trade_closed":
                continue

            ts_epoch = _parse_ts_epoch(row.get("ts"))
            if ts_epoch is None or ts_epoch < cutoff_epoch:
                continue

            trade_id = str(row.get("trade_id") or "").strip()
            client_id = str(row.get("client_id") or "").strip()
            if not client_id and trade_id:
                client_id = trade_to_client.get(trade_id, "")
            if local_prefix and client_id and not client_id.startswith(local_prefix):
                continue

            # `realized_jpy` in codex log is cumulative, so per-trade P/L must use `pl`.
            net = _as_float(row.get("pl"), _as_float(row.get("realized_jpy"), 0.0))
            pips = _as_float(row.get("pnl_pips"), _as_float(row.get("pl_pips"), 0.0))
            summary.net_jpy += net
            summary.sum_pips += pips
            summary.trades += 1
            if net > 0.0 or (net == 0.0 and pips > 0.0):
                summary.wins += 1
    return summary


def _sqlite_immutable_uri(path: Path) -> str:
    resolved = path.expanduser().resolve()
    quoted = quote(str(resolved), safe="/")
    return f"file:{quoted}?mode=ro&immutable=1"


def _load_vm_lane(
    *,
    trades_db: Path,
    cutoff_epoch: int,
    vm_prefix: str,
) -> LaneSummary:
    summary = LaneSummary()
    if not trades_db.exists():
        return summary

    uri = _sqlite_immutable_uri(trades_db)
    con = sqlite3.connect(uri, uri=True)
    try:
        like_pattern = f"{vm_prefix}%" if vm_prefix else "%"
        rows = con.execute(
            """
            SELECT
                COALESCE(realized_pl, 0.0) AS realized_pl,
                COALESCE(pl_pips, 0.0) AS pl_pips
            FROM trades
            WHERE COALESCE(client_order_id, '') LIKE ?
              AND COALESCE(pocket, '') <> 'manual'
              AND close_time IS NOT NULL
              AND CAST(strftime('%s', close_time) AS INTEGER) >= ?
            """,
            (like_pattern, int(cutoff_epoch)),
        ).fetchall()
    finally:
        try:
            con.close()
        except Exception:
            pass

    for realized_pl, pl_pips in rows:
        net = _as_float(realized_pl, 0.0)
        pips = _as_float(pl_pips, 0.0)
        summary.net_jpy += net
        summary.sum_pips += pips
        summary.trades += 1
        if net > 0.0 or (net == 0.0 and pips > 0.0):
            summary.wins += 1
    return summary


def _winner(
    local_lane: dict[str, float | int],
    vm_lane: dict[str, float | int],
    *,
    min_trades: int,
) -> dict[str, object]:
    local_trades = _as_int(local_lane.get("trades"), 0)
    vm_trades = _as_int(vm_lane.get("trades"), 0)
    if local_trades < min_trades and vm_trades < min_trades:
        return {"lane": "insufficient_data", "metric": "min_trades"}
    if local_trades < min_trades <= vm_trades:
        return {"lane": "vm", "metric": "min_trades"}
    if vm_trades < min_trades <= local_trades:
        return {"lane": "local", "metric": "min_trades"}

    local_net = _as_float(local_lane.get("net_jpy"), 0.0)
    vm_net = _as_float(vm_lane.get("net_jpy"), 0.0)
    if local_net > vm_net:
        return {"lane": "local", "metric": "net_jpy"}
    if vm_net > local_net:
        return {"lane": "vm", "metric": "net_jpy"}

    local_exp = _as_float(local_lane.get("expectancy_pips"), 0.0)
    vm_exp = _as_float(vm_lane.get("expectancy_pips"), 0.0)
    if local_exp > vm_exp:
        return {"lane": "local", "metric": "expectancy_pips"}
    if vm_exp > local_exp:
        return {"lane": "vm", "metric": "expectancy_pips"}
    return {"lane": "tie", "metric": "expectancy_pips"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare local lane vs VM lane trade performance.")
    ap.add_argument("--local-log", default="logs/codex_long_autotrade.log")
    ap.add_argument("--vm-trades-db", default="logs/trades.db")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--vm-prefix", default="qr-")
    ap.add_argument("--local-prefix", default="codexlhf_")
    ap.add_argument("--min-trades", type=int, default=5)
    args = ap.parse_args()

    now_epoch = time.time()
    cutoff_epoch = now_epoch - (max(0.0, float(args.hours)) * 3600.0)
    cutoff_epoch_int = _as_int(cutoff_epoch, 0)

    local_summary = _load_local_lane(
        local_log=Path(args.local_log),
        cutoff_epoch=cutoff_epoch,
        local_prefix=str(args.local_prefix or ""),
    ).as_dict()
    vm_summary = _load_vm_lane(
        trades_db=Path(args.vm_trades_db),
        cutoff_epoch=cutoff_epoch_int,
        vm_prefix=str(args.vm_prefix or ""),
    ).as_dict()

    result = {
        "window_hours": max(0.0, float(args.hours)),
        "min_trades": max(1, int(args.min_trades)),
        "lanes": {
            "local": local_summary,
            "vm": vm_summary,
        },
        "winner": _winner(local_summary, vm_summary, min_trades=max(1, int(args.min_trades))),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
