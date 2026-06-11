"""Mine the operator's 2025 manual trading history from OANDA transactions.

Background (2026-06-11, operator): 「去年の今頃、1ヶ月で利益200％以上出したよ。
俺が手動トレードで。だから、AI裁量トレーダーだったらもっといけるはず。」
The current product target (5% daily floor / 10% stretch, AGENT_CONTRACT §4)
is a reproduction of that observed manual performance, so the winning
patterns from that period are first-party evidence for the trader's
strategy profile — not external lore.

Read-only: uses OandaReadOnlyClient transaction history. Never trades.

Usage:
  PYTHONPATH=src python3 tools/mine_manual_history.py \
      --from 2025-05-15T00:00:00Z --to 2025-07-15T00:00:00Z \
      --out data/manual_history_2025_mining.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.broker.oanda import OandaReadOnlyClient


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        # OANDA emits nanosecond timestamps; trim to microseconds.
        if "." in value:
            head, tail = value.split(".", 1)
            tail = tail.rstrip("Z")[:6]
            value = f"{head}.{tail}+00:00"
        else:
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def fetch_transactions(client: OandaReadOnlyClient, *, time_from: str, time_to: str) -> list[dict]:
    listing = client.get_json(
        f"/v3/accounts/{client.account_id}/transactions"
        f"?from={time_from}&to={time_to}&pageSize=1000"
    )
    out: list[dict] = []
    for page_url in listing.get("pages", []):
        # pages are absolute URLs; reuse only the path+query part.
        path = page_url.split("oanda.com", 1)[-1]
        payload = client.get_json(path)
        out.extend(payload.get("transactions", []) or [])
    return out


def reconstruct_trades(transactions: list[dict]) -> list[dict]:
    """Pair trade opens with their closes via tradeID linkage."""
    opens: dict[str, dict] = {}
    trades: list[dict] = []
    for tx in transactions:
        if tx.get("type") != "ORDER_FILL":
            continue
        opened = tx.get("tradeOpened")
        if isinstance(opened, dict) and opened.get("tradeID"):
            opens[str(opened["tradeID"])] = {
                "trade_id": str(opened["tradeID"]),
                "pair": tx.get("instrument"),
                "open_time": tx.get("time"),
                "open_price": float(opened.get("price") or tx.get("price") or 0),
                "units": float(opened.get("units") or 0),
                "open_reason": tx.get("reason"),
            }
        for closed in tx.get("tradesClosed") or []:
            trade_id = str(closed.get("tradeID") or "")
            base = opens.get(trade_id, {})
            open_at = _parse_time(str(base.get("open_time") or ""))
            close_at = _parse_time(str(tx.get("time") or ""))
            hold_hours = (
                (close_at - open_at).total_seconds() / 3600.0
                if open_at and close_at
                else None
            )
            trades.append(
                {
                    "trade_id": trade_id,
                    "pair": base.get("pair") or tx.get("instrument"),
                    "units": base.get("units"),
                    "open_time": base.get("open_time"),
                    "close_time": tx.get("time"),
                    "hold_hours": round(hold_hours, 3) if hold_hours is not None else None,
                    "open_price": base.get("open_price"),
                    "close_price": float(closed.get("price") or tx.get("price") or 0),
                    "realized_pl": float(closed.get("realizedPL") or 0),
                    "financing": float(closed.get("financing") or 0),
                    "close_reason": tx.get("reason"),
                    "open_hour_utc": open_at.hour if open_at else None,
                }
            )
    return trades


def _bucket_stats(rows: list[dict]) -> dict:
    pls = [r["realized_pl"] for r in rows]
    wins = [p for p in pls if p > 0]
    losses = [p for p in pls if p < 0]
    holds = [r["hold_hours"] for r in rows if r["hold_hours"] is not None]
    return {
        "trades": len(rows),
        "net": round(sum(pls), 1),
        "win_rate": round(len(wins) / len(pls), 3) if pls else None,
        "avg_win": round(statistics.mean(wins), 1) if wins else None,
        "avg_loss": round(statistics.mean(losses), 1) if losses else None,
        "payoff": (
            round(abs(statistics.mean(wins) / statistics.mean(losses)), 2)
            if wins and losses
            else None
        ),
        "median_hold_hours": round(statistics.median(holds), 2) if holds else None,
        "expectancy": round(statistics.mean(pls), 1) if pls else None,
    }


def analyze(trades: list[dict], transactions: list[dict]) -> dict:
    by_pair: dict[str, list[dict]] = defaultdict(list)
    by_session: dict[str, list[dict]] = defaultdict(list)
    by_side: dict[str, list[dict]] = defaultdict(list)
    by_close_reason: dict[str, list[dict]] = defaultdict(list)
    daily: dict[str, float] = defaultdict(float)
    for t in trades:
        by_pair[str(t["pair"])].append(t)
        hour = t.get("open_hour_utc")
        if hour is not None:
            # JST session buckets (UTC+9): Tokyo 9-15 JST = 0-6 UTC, etc.
            if 0 <= hour < 6:
                session = "TOKYO"
            elif 6 <= hour < 12:
                session = "LONDON_AM"
            elif 12 <= hour < 18:
                session = "NY_OVERLAP"
            else:
                session = "OFF_HOURS"
            by_session[session].append(t)
        units = t.get("units")
        if units is not None:
            by_side["LONG" if units > 0 else "SHORT"].append(t)
        by_close_reason[str(t.get("close_reason") or "UNKNOWN")].append(t)
        day = str(t.get("close_time") or "")[:10]
        if day:
            daily[day] += t["realized_pl"]

    balance_curve: list[dict] = []
    for tx in transactions:
        balance = tx.get("accountBalance")
        if balance is not None:
            balance_curve.append({"time": tx.get("time"), "balance": float(balance)})

    sorted_days = sorted(daily.items())
    win_days = [d for d in sorted_days if d[1] > 0]
    return {
        "overall": _bucket_stats(trades),
        "by_pair": {k: _bucket_stats(v) for k, v in sorted(by_pair.items())},
        "by_session_jst": {k: _bucket_stats(v) for k, v in sorted(by_session.items())},
        "by_side": {k: _bucket_stats(v) for k, v in sorted(by_side.items())},
        "by_close_reason": {k: _bucket_stats(v) for k, v in sorted(by_close_reason.items())},
        "daily_pl": dict(sorted_days),
        "daily_win_rate": round(len(win_days) / len(sorted_days), 3) if sorted_days else None,
        "best_day": max(sorted_days, key=lambda kv: kv[1]) if sorted_days else None,
        "worst_day": min(sorted_days, key=lambda kv: kv[1]) if sorted_days else None,
        "balance_start": balance_curve[0] if balance_curve else None,
        "balance_end": balance_curve[-1] if balance_curve else None,
        "balance_peak": max(balance_curve, key=lambda r: r["balance"]) if balance_curve else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="time_from", default="2025-05-15T00:00:00Z")
    parser.add_argument("--to", dest="time_to", default="2025-07-15T00:00:00Z")
    parser.add_argument("--out", default="data/manual_history_2025_mining.json")
    args = parser.parse_args()

    client = OandaReadOnlyClient()
    transactions = fetch_transactions(client, time_from=args.time_from, time_to=args.time_to)
    trades = reconstruct_trades(transactions)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window": {"from": args.time_from, "to": args.time_to},
        "transaction_count": len(transactions),
        "closed_trades": len(trades),
        "analysis": analyze(trades, transactions),
        "trades": trades,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    summary = result["analysis"]["overall"]
    print(f"transactions={len(transactions)} closed_trades={len(trades)}")
    print(f"overall: {json.dumps(summary)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
