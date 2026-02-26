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
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
TRADES_DB = BASE_DIR / "logs" / "trades.db"
OUTPUT_PATH = BASE_DIR / "config" / "dynamic_alloc.json"

_EPHEMERAL_SUFFIX_PATTERNS = (
    re.compile(r"^(?P<base>.+?)-l[0-9a-f]{8,}$", re.IGNORECASE),
    re.compile(r"^(?P<base>.+?)-[0-9a-f]{8,}$", re.IGNORECASE),
)
_PREFIX_ALIAS_PATTERNS = (
    (re.compile(r"^micropul[0-9a-f]{8,}$", re.IGNORECASE), "MicroPullbackEMA"),
    (re.compile(r"^microran[0-9a-f]{8,}$", re.IGNORECASE), "MicroRangeBreak"),
    (re.compile(r"^microtre[0-9a-f]{8,}$", re.IGNORECASE), "MicroTrendRetest-long"),
    (re.compile(r"^scalpmacdrsi[0-9a-f]{8,}$", re.IGNORECASE), "scalp_macd_rsi_div_b_live"),
)


def normalize_strategy_key(raw: str | None) -> str:
    key = str(raw or "").strip() or "unknown"
    for pattern, alias in _PREFIX_ALIAS_PATTERNS:
        if pattern.match(key):
            return alias
    for pattern in _EPHEMERAL_SUFFIX_PATTERNS:
        matched = pattern.match(key)
        if matched:
            base = str(matched.group("base") or "").strip()
            if base:
                return base
    return key


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_utc_timestamp(raw: object) -> dt.datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        ts = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _recency_weight(close_time_raw: object, *, now_utc: dt.datetime, half_life_hours: float) -> float:
    ts = _parse_utc_timestamp(close_time_raw)
    if ts is None:
        return 1.0
    age_hours = max(0.0, (now_utc - ts).total_seconds() / 3600.0)
    half_life = max(1.0, float(half_life_hours))
    return 0.5 ** (age_hours / half_life)


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
              close_time,
              close_reason
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


def compute_scores(
    rows: List[Tuple],
    *,
    min_trades: int,
    pf_cap: float,
    min_lot_multiplier: float = 0.45,
    max_lot_multiplier: float = 1.65,
    half_life_hours: float = 36.0,
    allow_loser_block: bool = False,
    allow_winner_only: bool = False,
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    pockets: Dict[str, Dict[str, int]] = {}
    now_utc = dt.datetime.now(dt.timezone.utc)
    for row in rows:
        strat = row[0] if len(row) > 0 else None
        pocket = row[1] if len(row) > 1 else None
        pl_pips = row[2] if len(row) > 2 else 0.0
        close_time = row[3] if len(row) > 3 else None
        close_reason = str(row[4] if len(row) > 4 else "")

        strat = normalize_strategy_key(strat)
        pocket = pocket or "unknown"
        key = strat
        weight = _recency_weight(close_time, now_utc=now_utc, half_life_hours=half_life_hours)
        sl_like = close_reason in {"STOP_LOSS_ORDER", "MARKET_ORDER_MARGIN_CLOSEOUT"}
        s = stats.setdefault(
            key,
            {
                "wins": 0,
                "losses": 0,
                "trades": 0,
                "sum_pips": 0.0,
                "win_pips": 0.0,
                "loss_pips": 0.0,
                "w_trades": 0.0,
                "w_wins": 0.0,
                "w_sum_pips": 0.0,
                "w_win_pips": 0.0,
                "w_loss_pips": 0.0,
                "w_sl_like": 0.0,
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
        s["w_trades"] += weight
        s["w_sum_pips"] += pl * weight
        if pl > 0:
            s["w_wins"] += weight
            s["w_win_pips"] += pl * weight
        elif pl < 0:
            s["w_loss_pips"] += abs(pl) * weight
        if sl_like:
            s["w_sl_like"] += weight
    scores: Dict[str, Dict[str, float]] = {}
    for strat, s in stats.items():
        wins = int(s["wins"])
        trades = int(s["trades"])
        sum_pips = float(s["sum_pips"])
        w_trades = float(s.get("w_trades", 0.0))
        w_wins = float(s.get("w_wins", 0.0))
        w_sum_pips = float(s.get("w_sum_pips", 0.0))
        w_win_pips = float(s.get("w_win_pips", 0.0))
        w_loss_pips = float(s.get("w_loss_pips", 0.0))
        w_sl_like = float(s.get("w_sl_like", 0.0))

        wr = wins / max(1, trades)
        weighted_wr = w_wins / max(1e-9, w_trades)
        avg_pl = w_sum_pips / max(1e-9, w_trades)
        pf = w_win_pips / w_loss_pips if w_loss_pips > 0 else (w_win_pips if w_win_pips > 0 else 0.0)
        pf = min(pf, max(0.1, pf_cap))
        downside_share = w_loss_pips / max(1e-9, w_win_pips + w_loss_pips)
        sl_rate = w_sl_like / max(1e-9, w_trades)

        # Risk-adjusted normalization.
        wr_norm = _clamp((weighted_wr - 0.42) / 0.26, 0.0, 1.0)
        pf_norm = _clamp((pf - 0.60) / max(0.10, (pf_cap - 0.60)), 0.0, 1.0)
        avg_norm = _clamp((avg_pl + 1.80) / 4.20, 0.0, 1.0)
        downside_penalty = _clamp((downside_share - 0.50) / 0.35, 0.0, 1.0)
        sl_penalty = _clamp((sl_rate - 0.38) / 0.40, 0.0, 1.0)

        base_score = (
            0.42 * pf_norm
            + 0.34 * wr_norm
            + 0.24 * avg_norm
            - 0.28 * downside_penalty
            - 0.20 * sl_penalty
        )
        base_score = _clamp(base_score, 0.0, 1.0)
        sample_scale = min(1.0, trades / max(1, min_trades))
        score = _clamp(base_score * (0.55 + 0.45 * sample_scale), 0.0, 1.0)
        pocket_counts = pockets.get(strat, {})
        pocket = max(pocket_counts, key=pocket_counts.get) if pocket_counts else "unknown"

        min_mult = _clamp(float(min_lot_multiplier), 0.10, 1.00)
        max_mult = max(min_mult, float(max_lot_multiplier))
        lot_multiplier = min_mult + (max_mult - min_mult) * score

        # Guardrail: enforce soft participation and suppress aggressive size-up for underperformers.
        if pf < 1.0:
            lot_multiplier = min(lot_multiplier, 0.95)
        if pf < 0.8:
            lot_multiplier = min(lot_multiplier, 0.82)
        if pf < 0.7:
            lot_multiplier = min(lot_multiplier, 0.74)
        if pf < 0.6:
            lot_multiplier = min(lot_multiplier, 0.68)
        if trades >= max(12, min_trades) and avg_pl <= -1.0:
            lot_multiplier = min(lot_multiplier, 0.72)
        if trades >= max(24, min_trades * 2) and sum_pips <= -80.0:
            lot_multiplier = min(lot_multiplier, 0.66)
        if trades >= max(12, min_trades) and sl_rate >= 0.60:
            lot_multiplier = min(lot_multiplier, 0.66)
        if trades < max(1, min_trades):
            lot_multiplier = min(lot_multiplier, 1.00)
        lot_multiplier = _clamp(lot_multiplier, min_mult, max_mult)

        scores[strat] = {
            "pocket": pocket,
            "score": round(score, 3),
            "lot_multiplier": round(lot_multiplier, 3),
            "trades": trades,
            "win_rate": round(wr, 3),
            "weighted_win_rate": round(weighted_wr, 3),
            "pf": round(pf, 3),
            "avg_pips": round(avg_pl, 3),
            "sum_pips": round(sum_pips, 2),
            "sl_rate": round(sl_rate, 3),
            "downside_share": round(downside_share, 3),
            "allow_loser_block": bool(allow_loser_block),
            "allow_winner_only": bool(allow_winner_only),
        }
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300, help="Number of recent trades to use")
    ap.add_argument("--lookback-days", type=int, default=7, help="Lookback window in days")
    ap.add_argument("--min-trades", type=int, default=12, help="Min trades for full score weight")
    ap.add_argument("--pf-cap", type=float, default=2.0, help="Profit factor cap for normalization")
    ap.add_argument("--target-use", type=float, default=0.88, help="Target margin usage fraction")
    ap.add_argument("--half-life-hours", type=float, default=36.0, help="Recency half-life for scoring")
    ap.add_argument("--min-lot-multiplier", type=float, default=0.45, help="Lower bound of strategy lot multiplier")
    ap.add_argument("--max-lot-multiplier", type=float, default=1.65, help="Upper bound of strategy lot multiplier")
    ap.add_argument(
        "--soft-participation",
        type=int,
        choices=(0, 1),
        default=1,
        help="If 1, disable dyn alloc hard blocks and keep all strategies in reduced size",
    )
    ap.add_argument(
        "--allow-loser-block",
        type=int,
        choices=(0, 1),
        default=0,
        help="If 1, dyn alloc may hard-block low-score strategies",
    )
    ap.add_argument(
        "--allow-winner-only",
        type=int,
        choices=(0, 1),
        default=0,
        help="If 1, workers may route entries to winner-only subset",
    )
    args = ap.parse_args()

    rows = fetch_trades(args.limit, args.lookback_days)
    soft_participation = bool(int(args.soft_participation))
    allow_loser_block = bool(int(args.allow_loser_block)) and not soft_participation
    allow_winner_only = bool(int(args.allow_winner_only)) and not soft_participation
    scores = compute_scores(
        rows,
        min_trades=args.min_trades,
        pf_cap=args.pf_cap,
        min_lot_multiplier=args.min_lot_multiplier,
        max_lot_multiplier=args.max_lot_multiplier,
        half_life_hours=args.half_life_hours,
        allow_loser_block=allow_loser_block,
        allow_winner_only=allow_winner_only,
    )

    alloc = {
        "as_of": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "lookback_days": int(args.lookback_days),
        "min_trades": int(args.min_trades),
        "pf_cap": float(args.pf_cap),
        "half_life_hours": float(args.half_life_hours),
        "target_use": args.target_use,
        "pocket_caps": {"macro": 0.35, "micro": 0.35, "scalp": 0.30},
        "allocation_policy": {
            "mode": "soft_participation" if soft_participation else "classic",
            "soft_participation": soft_participation,
            "allow_loser_block": allow_loser_block,
            "allow_winner_only": allow_winner_only,
            "min_lot_multiplier": float(args.min_lot_multiplier),
            "max_lot_multiplier": float(args.max_lot_multiplier),
        },
        "strategies": scores,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(alloc, ensure_ascii=False, indent=2))
    print(f"[dynamic_alloc] wrote {OUTPUT_PATH} with {len(scores)} strategies")


if __name__ == "__main__":
    main()
