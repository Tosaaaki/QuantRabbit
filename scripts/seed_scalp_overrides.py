from __future__ import annotations

import argparse
from analysis.scalp_config import store_overrides

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Firestore config/scalp overrides.")
    parser.add_argument("--share", type=float, default=0.02)
    parser.add_argument("--base-lot", type=float, default=0.01)
    parser.add_argument("--min-atr", type=float, default=2.5)
    parser.add_argument("--deviation", type=float, default=3.5)
    args = parser.parse_args()

    payload = {
        "share": round(args.share, 3),
        "base_lot": round(args.base_lot, 3),
        "min_atr_pips": round(args.min_atr, 3),
        "deviation_pips": round(args.deviation, 3),
    }
    store_overrides(payload)
    print("stored overrides:", payload)

if __name__ == "__main__":
    main()
