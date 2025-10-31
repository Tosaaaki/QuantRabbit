import argparse
from pathlib import Path
import sqlite3
import json
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"


def fetch_latest(pattern_limit: int = 80):
    orders = sqlite3.connect(LOG_DIR / "orders.db")
    trades = sqlite3.connect(LOG_DIR / "trades.db")
    orders.row_factory = sqlite3.Row
    trades.row_factory = sqlite3.Row

    rows = trades.execute(
        """
        SELECT id, ticket_id, pocket, entry_time, close_time, pl_pips, close_reason
        FROM trades
        WHERE pocket='scalp_fast'
        ORDER BY id DESC
        LIMIT ?
        """,
        (pattern_limit,),
    ).fetchall()

    records = []
    for tr in rows:
        order_row = orders.execute(
            "SELECT request_json FROM orders WHERE ticket_id=? AND status='submit_attempt' ORDER BY id DESC LIMIT 1",
            (tr["ticket_id"],),
        ).fetchone()
        thesis = {}
        if order_row and order_row["request_json"]:
            try:
                payload = json.loads(order_row["request_json"])
                thesis = payload.get("meta", {}).get("entry_thesis", {})
            except json.JSONDecodeError:
                pass
        records.append({
            "ticket_id": tr["ticket_id"],
            "close_reason": tr["close_reason"],
            "pl_pips": tr["pl_pips"],
            "entry": tr["entry_time"],
            "pattern": thesis.get("pattern_tag", "unknown"),
            "momentum": thesis.get("momentum_pips"),
            "short_momentum": thesis.get("short_momentum_pips"),
            "atr": thesis.get("tick_atr"),
            "rsi": thesis.get("tick_rsi"),
            "range": thesis.get("range_pips"),
            "tick_count": thesis.get("tick_count"),
        })

    orders.close()
    trades.close()
    return records


def aggregate(records):
    buckets = {}
    for rec in records:
        buckets.setdefault(rec["pattern"], []).append(rec)
    lines = []
    for pattern, recs in sorted(buckets.items(), key=lambda x: x[0]):
        closes = [r["pl_pips"] for r in recs if isinstance(r["pl_pips"], (int, float))]
        wins = sum(1 for r in recs if r["close_reason"] == "TAKE_PROFIT_ORDER")
        losses = len(recs) - wins
        avg_pl = mean(closes) if closes else 0.0
        lines.append(
            f"pattern={pattern:15s} trades={len(recs):3d} wins={wins:2d} losses={losses:2d} avg_pl={avg_pl:6.2f}"
        )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()
    recs = fetch_latest(args.limit)
    for line in aggregate(recs):
        print(line)
    print("--- last samples ---")
    for rec in recs[:10]:
        print(rec)


if __name__ == "__main__":
    main()
