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
from datetime import datetime, timedelta, timezone
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
    """Pair trade opens with their close/reduction exits via tradeID linkage."""
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
        reduced = tx.get("tradeReduced")
        reduced_items = [reduced] if isinstance(reduced, dict) else []
        closed_items = list(tx.get("tradesClosed") or [])
        for exit_kind, closed in [("REDUCED", item) for item in reduced_items] + [
            ("CLOSED", item) for item in closed_items
        ]:
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
                    "exit_kind": exit_kind,
                    "pair": base.get("pair") or tx.get("instrument"),
                    "units": base.get("units"),
                    "exit_units": float(closed.get("units") or 0),
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


def _transfer_adjusted_cash_flows(transactions: list[dict], balance_curve: list[dict]) -> dict:
    """Separate account funding from trading P/L.

    Raw account balance is the broker truth for equity, but it is not the same
    as strategy performance when the operator adds or withdraws cash during
    the window. The 2025 manual month had both, so the mining report must keep
    raw-balance and transfer-adjusted returns side by side.
    """
    transfers: list[dict] = []
    for tx in transactions:
        if tx.get("type") != "TRANSFER_FUNDS":
            continue
        try:
            amount = float(tx.get("amount") or 0.0)
        except (TypeError, ValueError):
            amount = 0.0
        transfers.append(
            {
                "time": tx.get("time"),
                "amount": amount,
                "balance": float(tx.get("accountBalance") or 0.0)
                if tx.get("accountBalance") is not None
                else None,
            }
        )

    if not balance_curve:
        return {
            "transfers": transfers,
            "net_additional_transfers": 0.0,
        }

    start = balance_curve[0]
    end = balance_curve[-1]
    peak = max(balance_curve, key=lambda r: r["balance"])
    start_at = _parse_time(str(start.get("time") or ""))
    peak_at = _parse_time(str(peak.get("time") or ""))
    start_balance = float(start["balance"])

    def _after_start(row: dict) -> bool:
        ts = _parse_time(str(row.get("time") or ""))
        return bool(ts and start_at and ts > start_at)

    def _before_or_at_peak(row: dict) -> bool:
        ts = _parse_time(str(row.get("time") or ""))
        return bool(ts and peak_at and ts <= peak_at)

    additional = [row for row in transfers if _after_start(row)]
    additional_to_peak = [row for row in additional if _before_or_at_peak(row)]
    net_additional = sum(float(row["amount"]) for row in additional)
    net_additional_to_peak = sum(float(row["amount"]) for row in additional_to_peak)

    adjusted_peak_balance = float(peak["balance"]) - net_additional_to_peak
    adjusted_end_balance = float(end["balance"]) - net_additional
    adjusted_peak_profit = adjusted_peak_balance - start_balance
    adjusted_end_profit = adjusted_end_balance - start_balance

    # The 30-day window is the audit lens for the operator's specific
    # "1 month / 200%+" recollection. It is reporting metadata, not a risk
    # setting or trading threshold.
    best_30d = _best_funding_adjusted_window(
        balance_curve=balance_curve,
        transfers=transfers,
        window_days=30,
    )

    return {
        "transfers": transfers,
        "initial_balance": round(start_balance, 4),
        "raw_balance_peak": peak,
        "raw_balance_end": end,
        "net_additional_transfers": round(net_additional, 4),
        "net_additional_transfers_to_peak": round(net_additional_to_peak, 4),
        "transfer_adjusted_peak_balance": round(adjusted_peak_balance, 4),
        "transfer_adjusted_peak_profit": round(adjusted_peak_profit, 4),
        "transfer_adjusted_peak_return_pct": (
            round(adjusted_peak_profit / start_balance * 100.0, 2)
            if start_balance
            else None
        ),
        "transfer_adjusted_end_balance": round(adjusted_end_balance, 4),
        "transfer_adjusted_end_profit": round(adjusted_end_profit, 4),
        "transfer_adjusted_end_return_pct": (
            round(adjusted_end_profit / start_balance * 100.0, 2)
            if start_balance
            else None
        ),
        "best_30d_funding_adjusted": best_30d,
    }


def _best_funding_adjusted_window(
    *,
    balance_curve: list[dict],
    transfers: list[dict],
    window_days: int,
) -> dict | None:
    """Return the best transfer-adjusted balance return inside a calendar window."""
    if not balance_curve:
        return None
    points: list[tuple[datetime, float]] = []
    for row in balance_curve:
        ts = _parse_time(str(row.get("time") or ""))
        if not ts:
            continue
        points.append((ts, float(row["balance"])))
    if not points:
        return None
    transfer_points: list[tuple[datetime, float]] = []
    for row in transfers:
        ts = _parse_time(str(row.get("time") or ""))
        if not ts:
            continue
        transfer_points.append((ts, float(row.get("amount") or 0.0)))

    def _net_transfers_between(start_at: datetime, end_at: datetime) -> float:
        return sum(amount for ts, amount in transfer_points if start_at < ts <= end_at)

    best: dict | None = None
    for index, (start_at, start_balance) in enumerate(points):
        if start_balance <= 0:
            continue
        limit = start_at + timedelta(days=window_days)
        for end_at, end_balance in points[index:]:
            if end_at > limit:
                break
            net_transfers = _net_transfers_between(start_at, end_at)
            profit = end_balance - start_balance - net_transfers
            return_pct = profit / start_balance * 100.0
            if best is None or return_pct > float(best["return_pct"]):
                best = {
                    "window_days": window_days,
                    "start_time": start_at.isoformat(),
                    "end_time": end_at.isoformat(),
                    "start_balance": round(start_balance, 4),
                    "end_balance": round(end_balance, 4),
                    "net_transfers": round(net_transfers, 4),
                    "profit": round(profit, 4),
                    "return_pct": round(return_pct, 2),
                }
    return best


def _realized_pl_components(transactions: list[dict]) -> dict:
    """Summarize where broker-realized P/L appeared in OANDA transactions."""
    components: dict[str, dict[str, float | int]] = defaultdict(lambda: {"count": 0, "net": 0.0})
    for tx in transactions:
        if tx.get("type") == "DAILY_FINANCING":
            try:
                components["daily_financing"]["net"] += float(tx.get("financing") or 0.0)
                components["daily_financing"]["count"] += 1
            except (TypeError, ValueError):
                pass
            continue
        if tx.get("type") != "ORDER_FILL":
            continue
        for key in ("tradeReduced",):
            item = tx.get(key)
            if isinstance(item, dict) and item.get("realizedPL") is not None:
                components[key]["net"] += float(item.get("realizedPL") or 0.0)
                components[key]["count"] += 1
        for key in ("tradesClosed",):
            for item in tx.get(key) or []:
                if isinstance(item, dict) and item.get("realizedPL") is not None:
                    components[key]["net"] += float(item.get("realizedPL") or 0.0)
                    components[key]["count"] += 1
    return {
        key: {"count": int(value["count"]), "net": round(float(value["net"]), 4)}
        for key, value in sorted(components.items())
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
        "realized_pl_components": _realized_pl_components(transactions),
        "cash_flows": _transfer_adjusted_cash_flows(transactions, balance_curve),
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
        "exit_events": len(trades),
        "closed_trades": sum(1 for trade in trades if trade.get("exit_kind") == "CLOSED"),
        "reduced_trades": sum(1 for trade in trades if trade.get("exit_kind") == "REDUCED"),
        "analysis": analyze(trades, transactions),
        "trades": trades,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    summary = result["analysis"]["overall"]
    print(
        "transactions="
        f"{len(transactions)} exit_events={result['exit_events']} "
        f"closed_trades={result['closed_trades']} reduced_trades={result['reduced_trades']}"
    )
    print(f"overall: {json.dumps(summary)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
