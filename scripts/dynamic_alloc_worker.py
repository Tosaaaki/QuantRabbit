#!/usr/bin/env python
"""
Compute simple strategy scores from recent trades and emit config/dynamic_alloc.json.

Intended to run periodically (cron/systemd timer) to feed score-driven allocation context
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


def fetch_trades(limit: int, lookback_days: int) -> List[Tuple]:
    if not TRADES_DB.exists():
        return []
    # 読み取り専用で接続し、ロック影響を最小化
    uri = f"file:{TRADES_DB}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10.0, isolation_level=None)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
              COALESCE(NULLIF(strategy_tag, ''), strategy) AS strategy,
              pocket,
              pl_pips,
              close_time
            FROM trades
            WHERE close_time IS NOT NULL
              AND close_time >= datetime('now', ?)
            ORDER BY close_time DESC
            LIMIT ?
            """,
            (f"-{int(lookback_days)} day", limit),
        )
        return cur.fetchall()
    finally:
        conn.close()


def compute_scores(rows: List[Tuple], *, min_trades: int, pf_cap: float) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    pockets: Dict[str, Dict[str, int]] = {}
    for strat, pocket, pl_pips, _entry in rows:
        strat = strat or "unknown"
        pocket = pocket or "unknown"
        key = strat
        s = stats.setdefault(
            key,
            {
                "wins": 0,
                "losses": 0,
                "trades": 0,
                "sum_pips": 0.0,
                "win_pips": 0.0,
                "loss_pips": 0.0,
            },
        )
        pockets.setdefault(key, {})
        pockets[key][pocket] = pockets[key].get(pocket, 0) + 1
        pl = float(pl_pips or 0.0)
        s["sum_pips"] += pl
        s["trades"] += 1
        if pl > 0:
            s["wins"] += 1
            s["win_pips"] += pl
        elif pl < 0:
            s["losses"] += 1
            s["loss_pips"] += abs(pl)
    scores: Dict[str, Dict[str, float]] = {}
    for strat, s in stats.items():
        wins = int(s["wins"])
        losses = int(s["losses"])
        trades = int(s["trades"])
        sum_pips = float(s["sum_pips"])
        win_pips = float(s["win_pips"])
        loss_pips = float(s["loss_pips"])
        wr = wins / max(1, trades)
        avg_pl = sum_pips / max(1, trades)
        pf = win_pips / loss_pips if loss_pips > 0 else (win_pips if win_pips > 0 else 0.0)
        pf = min(pf, max(0.1, pf_cap))
        # Normalize
        wr_norm = min(max((wr - 0.45) / 0.30, 0.0), 1.0)
        pf_norm = min(max(pf / max(0.1, pf_cap), 0.0), 1.0)
        avg_norm = min(max((avg_pl + 4.0) / 12.0, 0.0), 1.0)
        base_score = 0.45 * pf_norm + 0.35 * wr_norm + 0.2 * avg_norm
        sample_scale = min(1.0, trades / max(1, min_trades))
        score = base_score * (0.6 + 0.4 * sample_scale)
        pocket_counts = pockets.get(strat, {})
        pocket = max(pocket_counts, key=pocket_counts.get) if pocket_counts else "unknown"
        lot_multiplier = 0.8 + 0.8 * score
        # Guardrail: strategies with poor payoff quality should not receive size-up even when
        # short-term win-rate looks high. Keep underperformers below neutral size.
        if pf < 1.0:
            lot_multiplier = min(lot_multiplier, 0.95)
        if pf < 0.7:
            lot_multiplier = min(lot_multiplier, 0.90)
        if trades < max(1, min_trades):
            lot_multiplier = min(lot_multiplier, 1.00)
        scores[strat] = {
            "pocket": pocket,
            "score": round(score, 3),
            "lot_multiplier": round(lot_multiplier, 3),
            "trades": trades,
            "win_rate": round(wr, 3),
            "pf": round(pf, 3),
            "avg_pips": round(avg_pl, 3),
            "sum_pips": round(sum_pips, 2),
        }
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300, help="Number of recent trades to use")
    ap.add_argument("--lookback-days", type=int, default=7, help="Lookback window in days")
    ap.add_argument("--min-trades", type=int, default=12, help="Min trades for full score weight")
    ap.add_argument("--pf-cap", type=float, default=2.0, help="Profit factor cap for normalization")
    ap.add_argument("--target-use", type=float, default=0.88, help="Target margin usage fraction")
    args = ap.parse_args()

    rows = fetch_trades(args.limit, args.lookback_days)
    scores = compute_scores(rows, min_trades=args.min_trades, pf_cap=args.pf_cap)

    alloc = {
        "as_of": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "lookback_days": int(args.lookback_days),
        "min_trades": int(args.min_trades),
        "pf_cap": float(args.pf_cap),
        "target_use": args.target_use,
        "pocket_caps": {"macro": 0.35, "micro": 0.35, "scalp": 0.30},
        "strategies": scores,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(alloc, ensure_ascii=False, indent=2))
    print(f"[dynamic_alloc] wrote {OUTPUT_PATH} with {len(scores)} strategies")


if __name__ == "__main__":
    main()
