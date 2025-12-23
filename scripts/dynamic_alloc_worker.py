#!/usr/bin/env python
"""
Compute simple strategy scores from recent trades and emit config/dynamic_alloc.json.

Intended to run periodically (cron/systemd timer) to feed main.py with score-driven
confidence trims and pocket caps.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
TRADES_DB = BASE_DIR / "logs" / "trades.db"
OUTPUT_PATH = BASE_DIR / "config" / "dynamic_alloc.json"


def fetch_trades(limit: int) -> List[Tuple]:
    if not TRADES_DB.exists():
        return []
    # 読み取り専用で接続し、ロック影響を最小化
    uri = f"file:{TRADES_DB}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10.0, isolation_level=None)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT strategy_tag, pocket, pl_pips, entry_time
            FROM trades
            ORDER BY entry_time DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def compute_scores(rows: List[Tuple]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for strat, pocket, pl_pips, _entry in rows:
        strat = strat or "unknown"
        pocket = pocket or "unknown"
        key = strat
        s = stats.setdefault(key, {"pocket": pocket, "wins": 0, "losses": 0, "pl": 0.0})
        pl = float(pl_pips or 0.0)
        s["pl"] += pl
        if pl > 0:
            s["wins"] += 1
        elif pl < 0:
            s["losses"] += 1
    scores: Dict[str, Dict[str, float]] = {}
    for strat, s in stats.items():
        wins = s["wins"]
        losses = s["losses"]
        wr = wins / max(1, wins + losses)
        avg_pl = s["pl"] / max(1, wins + losses)
        # Simple PF approximation: total win / total loss
        total_win = wins or 1
        total_loss = losses or 1
        pf = (total_win) / max(1, total_loss)
        # Normalize
        wr_norm = min(max((wr - 0.4) / 0.25, 0.0), 1.0)
        pf_norm = min(max(pf / 1.5, 0.0), 1.0)
        avg_norm = min(max((avg_pl + 5) / 15, 0.0), 1.0)
        score = 0.45 * pf_norm + 0.35 * wr_norm + 0.2 * avg_norm
        scores[strat] = {
            "pocket": s["pocket"],
            "score": round(score, 3),
            "lot_multiplier": round(0.8 + 0.8 * score, 3),
        }
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300, help="Number of recent trades to use")
    ap.add_argument("--target-use", type=float, default=0.88, help="Target margin usage fraction")
    args = ap.parse_args()

    rows = fetch_trades(args.limit)
    scores = compute_scores(rows)

    alloc = {
        "as_of": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "target_use": args.target_use,
        "pocket_caps": {"macro": 0.35, "micro": 0.35, "scalp": 0.30},
        "strategies": scores,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(alloc, ensure_ascii=False, indent=2))
    print(f"[dynamic_alloc] wrote {OUTPUT_PATH} with {len(scores)} strategies")


if __name__ == "__main__":
    main()
