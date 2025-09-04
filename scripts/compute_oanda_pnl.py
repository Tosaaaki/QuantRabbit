import argparse
import json
from datetime import datetime, timezone

from execution.oanda_pnl import compute_daily_pnl

try:
    from google.cloud import firestore
except Exception:
    firestore = None


def main():
    ap = argparse.ArgumentParser(description="Compute daily PnL from OANDA")
    ap.add_argument("--date", help="UTC date YYYY-MM-DD (default: today)")
    ap.add_argument("--write-firestore", action="store_true", help="Write result to Firestore perf/daily/{date}")
    args = ap.parse_args()

    if args.date:
        dt = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)
    else:
        dt = None

    res = compute_daily_pnl(dt)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    if args.write_firestore:
        if firestore is None:
            print("Firestore client not available. Skipping write.")
            return
        db = firestore.Client()
        key = res["date"]
        db.collection("perf").document("daily").collection("days").document(key).set(res)
        # mirror the latest under perf/daily/today for dashboard convenience
        db.collection("perf").document("daily").set(res)
        print("Wrote Firestore: perf/daily/{} and perf/daily".format(key))


if __name__ == "__main__":
    main()
