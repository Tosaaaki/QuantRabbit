"""
Verify user market calls — compare predicted direction vs actual price movement.

Finds unverified user_calls (outcome IS NULL) that are old enough to evaluate (>2h),
fetches historical price from OANDA, and marks them correct/incorrect.

Usage:
  python3 tools/verify_user_calls.py           # verify all pending calls
  python3 tools/verify_user_calls.py --dry-run  # show what would be updated

Designed to run in daily-review, but safe to run anytime.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --- DB setup ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "collab_trade" / "memory"))
from schema import get_conn

# --- OANDA config ---
_env = Path(__file__).resolve().parent.parent / "config" / "env.toml"
_lines = _env.read_text().split("\n")
TOKEN = next(l.split("=")[1].strip().strip('"') for l in _lines if l.startswith("oanda_token"))
ACCT = next(l.split("=")[1].strip().strip('"') for l in _lines if l.startswith("oanda_account_id"))
BASE = "https://api-fxtrade.oanda.com"

# Pip size per pair
PIP = {
    "USD_JPY": 0.01, "EUR_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
}

MIN_AGE_HOURS = 2  # Don't verify calls less than 2h old — need time for the move
EVAL_WINDOW_H = 4  # Check price 4h after the call
CORRECT_THRESHOLD_PIPS = 3.0  # Must move 3+ pip in predicted direction to count as correct


def oanda_get(path: str) -> dict:
    url = f"{BASE}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def get_price_at(pair: str, time_iso: str) -> float | None:
    """Get mid close price from H1 candle covering the given time."""
    try:
        data = oanda_get(
            f"/v3/instruments/{pair}/candles?granularity=H1&from={time_iso}&count=1"
        )
        if data.get("candles"):
            return float(data["candles"][0]["mid"]["c"])
    except Exception as e:
        print(f"  [WARN] Failed to get price for {pair} at {time_iso}: {e}")
    return None


def get_price_after(pair: str, time_iso: str, hours: int = EVAL_WINDOW_H) -> float | None:
    """Get mid close price N hours after the given time."""
    try:
        dt = datetime.fromisoformat(time_iso.replace("Z", "+00:00"))
        after = dt + timedelta(hours=hours)
        after_iso = after.strftime("%Y-%m-%dT%H:%M:%SZ")
        data = oanda_get(
            f"/v3/instruments/{pair}/candles?granularity=H1&from={after_iso}&count=1"
        )
        if data.get("candles"):
            return float(data["candles"][0]["mid"]["c"])
    except Exception as e:
        print(f"  [WARN] Failed to get price after for {pair}: {e}")
    return None


def estimate_call_time(row: dict) -> str | None:
    """Build ISO timestamp from session_date + timestamp fields."""
    ts = row.get("timestamp")
    sd = row.get("session_date")
    if ts and sd:
        # timestamp is like "13:24Z" or "16:17Z"
        ts_clean = ts.replace("Z", "").strip()
        return f"{sd}T{ts_clean}:00Z"
    if sd:
        # No timestamp — use midday as rough estimate
        return f"{sd}T12:00:00Z"
    return None


def verify_call(row: dict, dry_run: bool = False) -> dict | None:
    """Verify a single user call. Returns update dict or None if can't verify."""
    pair = row["pair"]
    direction = row["direction"]  # UP or DOWN
    call_time = estimate_call_time(row)

    if not call_time or not pair or pair not in PIP:
        return None

    pip_size = PIP[pair]

    # Get price at call time (or use recorded price)
    price_at = row.get("price_at_call")
    if not price_at:
        price_at = get_price_at(pair, call_time)
        if not price_at:
            return None

    # Get price after evaluation window
    price_after = get_price_after(pair, call_time, EVAL_WINDOW_H)
    if not price_after:
        return None

    # Calculate pip movement
    move_pips = (price_after - price_at) / pip_size
    if direction == "DOWN":
        move_pips = -move_pips  # Positive = moved in predicted direction

    # Determine outcome
    if move_pips >= CORRECT_THRESHOLD_PIPS:
        outcome = "correct"
    elif move_pips <= -CORRECT_THRESHOLD_PIPS:
        outcome = "incorrect"
    else:
        outcome = "neutral"  # Didn't move enough either way

    # Also get 30m and 1h prices for the record
    price_30m = get_price_after(pair, call_time, hours=0.5)
    price_1h = get_price_after(pair, call_time, hours=1)
    pl_30m = ((price_30m - price_at) / pip_size) if price_30m else None
    pl_1h = ((price_1h - price_at) / pip_size) if price_1h else None
    if direction == "DOWN":
        pl_30m = -pl_30m if pl_30m is not None else None
        pl_1h = -pl_1h if pl_1h is not None else None

    return {
        "id": row["id"],
        "pair": pair,
        "direction": direction,
        "call_text": row["call_text"],
        "call_time": call_time,
        "price_at_call": price_at,
        "price_after": price_after,
        "move_pips": move_pips,
        "outcome": outcome,
        "price_after_30m": price_30m,
        "price_after_1h": price_1h,
        "pl_after_30m": pl_30m,
        "pl_after_1h": pl_1h,
    }


def main():
    dry_run = "--dry-run" in sys.argv
    conn = get_conn()

    # Find unverified calls older than MIN_AGE_HOURS
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=MIN_AGE_HOURS)).strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT * FROM user_calls WHERE outcome IS NULL AND session_date <= ? ORDER BY id",
        (cutoff,)
    ).fetchall()

    col_names = [d[0] for d in conn.execute("PRAGMA table_info(user_calls)").fetchall()]
    col_map = {name: i for i, (_, name, *_) in enumerate(
        conn.execute("PRAGMA table_info(user_calls)").fetchall()
    )}
    rows_dict = []
    for r in rows:
        d = {}
        for col_info in conn.execute("PRAGMA table_info(user_calls)").fetchall():
            idx, name = col_info[0], col_info[1]
            d[name] = r[idx]
        rows_dict.append(d)

    if not rows_dict:
        print("No unverified user calls to process.")
        return

    print(f"=== Verifying {len(rows_dict)} unverified user calls ===\n")

    results = {"correct": 0, "incorrect": 0, "neutral": 0, "failed": 0}

    for row in rows_dict:
        print(f"[{row['session_date']}] {row['pair']} {row['direction']} — \"{row['call_text']}\"")
        result = verify_call(row, dry_run)

        if not result:
            print(f"  -> SKIP (missing data)\n")
            results["failed"] += 1
            continue

        mark = {"correct": "OK", "incorrect": "NG", "neutral": "--"}[result["outcome"]]
        print(f"  Price at call: {result['price_at_call']:.5f}")
        print(f"  Price after {EVAL_WINDOW_H}h: {result['price_after']:.5f}")
        print(f"  Move: {result['move_pips']:+.1f} pip in predicted direction")
        print(f"  -> {mark} ({result['outcome']})")
        results[result["outcome"]] += 1

        if not dry_run:
            conn.execute(
                """UPDATE user_calls SET
                    outcome = ?, price_at_call = ?,
                    price_after_30m = ?, price_after_1h = ?,
                    pl_after_30m = ?, pl_after_1h = ?
                WHERE id = ?""",
                (
                    result["outcome"],
                    result["price_at_call"],
                    result["price_after_30m"],
                    result["price_after_1h"],
                    result["pl_after_30m"],
                    result["pl_after_1h"],
                    result["id"],
                )
            )
            print(f"  -> DB updated")
        else:
            print(f"  -> (dry-run, no DB update)")
        print()

    # Summary
    total = sum(results.values())
    verified = results["correct"] + results["incorrect"] + results["neutral"]
    print(f"=== Summary: {verified}/{total} verified ===")
    print(f"  Correct: {results['correct']}  Incorrect: {results['incorrect']}  Neutral: {results['neutral']}  Skipped: {results['failed']}")
    if verified > 0:
        accuracy = results["correct"] / verified
        print(f"  User call accuracy: {accuracy:.0%} (n={verified})")


if __name__ == "__main__":
    main()
