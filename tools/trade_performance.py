#!/usr/bin/env python3
"""v4 Trade Performance Tracker — parses live_trade_log.txt (v6 format).

Handles all log format variations:
- v6 (current): [ts] CLOSE PAIR DIR UNITSu @PRICE PL=±NNN円|JPY
- v6 PARTIAL:   [ts] PARTIAL_CLOSE PAIR DIR UNITSu @PRICE PL=±NNN...
- Legacy:       [ts] FAST: CLOSED PAIR DIR ±N.Npip
- Legacy:       [ts] MONITOR: CLOSED PAIR ±N.Npip

Outputs: win rate, P/L by pair/direction/session/date, trend analysis.

Usage:
    python tools/trade_performance.py [--json] [--last N] [--days N] [--date YYYY-MM-DD]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_PATH = BASE_DIR / "logs" / "live_trade_log.txt"
OUTPUT_PATH = BASE_DIR / "logs" / "strategy_feedback.json"

# ─── v6 CLOSE/PARTIAL_CLOSE patterns ───
# [2026-03-25T13:54:28Z] CLOSE GBP_JPY SHORT 1000u @212.829 PL=+155円
# [2026-03-25 14:22:34 UTC] CLOSE EUR_JPY SHORT 500u @184.121 PL=+89円
# [14:12Z CLOSE] AUD_JPY SHORT 500u @110.712 PL=-23円
# [2026-03-26 04:14 UTC] CLOSE EUR_JPY SHORT 4000u @184.453 PL=+2円
# PL formats: +155円, -45.5円, +49JPY, +95.5JPY, -31, +160.11, +0.6JPY, +1,183
V6_CLOSE = re.compile(
    r"\[([^\]]+)\]\s*"                                          # timestamp
    r"(?:CLOSE|PARTIAL_CLOSE)\]?\s+"                            # action
    r"(USD_JPY|EUR_USD|GBP_USD|AUD_USD|EUR_JPY|GBP_JPY|AUD_JPY)\s+"  # pair
    r"(LONG(?:_HEDGE)?|SHORT(?:_HEDGE)?)\s+"                    # direction
    r"(\d+)u?\s+"                                               # units
    r"@([\d.]+)\s+"                                             # price
    r".*?PL=([+-]?[\d,.]+)\s*(?:円|JPY|J)?",                    # P/L
    re.IGNORECASE,
)

# ─── Legacy FAST/SWING/MONITOR patterns ───
# [2026-03-19T10:30:00Z] FAST: CLOSED EUR_USD LONG +3.2pip
LEGACY_CLOSE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?)\]\s+"
    r"(?:FAST|SWING|SCALP|MONITOR)[:\s]+"
    r".*?(?:CLOSED?|TP_HIT|SL_HIT|CUT|TRAIL_HIT)"
    r".*?"
    r"(USD_JPY|EUR_USD|GBP_USD|AUD_USD|EUR_JPY|GBP_JPY|AUD_JPY)"
    r".*?"
    r"(LONG|SHORT|L|S)"
    r".*?"
    r"([+-]?\d+\.?\d*)\s*pip",
    re.IGNORECASE,
)

# ─── ENTRY pattern (for entry tracking) ───
V6_ENTRY = re.compile(
    r"\[([^\]]+)\]\s*"
    r"ENTRY\s+"
    r"(USD_JPY|EUR_USD|GBP_USD|AUD_USD|EUR_JPY|GBP_JPY|AUD_JPY)\s+"
    r"(LONG|SHORT)\s+"
    r"-?(\d+)u?\s+"
    r"@([\d.]+)",
    re.IGNORECASE,
)


def parse_timestamp(ts_str: str) -> datetime | None:
    """Parse various timestamp formats to datetime."""
    ts_str = ts_str.strip()
    for fmt in [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S UTC",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M UTC",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%MZ",
        "%H:%MZ",
    ]:
        try:
            dt = datetime.strptime(ts_str, fmt)
            # Handle time-only format
            if dt.year == 1900:
                dt = dt.replace(year=2026, month=3, day=25)
            return dt
        except ValueError:
            continue
    return None


def classify_session(dt: datetime) -> str:
    """Classify UTC hour into trading session."""
    h = dt.hour
    if 0 <= h < 7:
        return "tokyo"
    elif 7 <= h < 15:
        return "london"
    elif 15 <= h < 22:
        return "newyork"
    return "late"


def normalize_direction(d: str) -> str:
    d = d.upper().replace("_HEDGE", "")
    return "LONG" if d in ("L", "LONG") else "SHORT"


def parse_pl(pl_str: str) -> float:
    """Parse P/L string to float JPY value."""
    # Remove commas (e.g. +1,183)
    cleaned = pl_str.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_trades(log_path: Path, max_lines: int = 0,
                 filter_days: int = 0, filter_date: str = "") -> list[dict]:
    """Parse trade log and extract closed trades."""
    if not log_path.exists():
        print(f"WARN: {log_path} not found", file=sys.stderr)
        return []

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if max_lines > 0:
        lines = lines[-max_lines:]

    trades = []
    seen = set()

    # Date filters
    cutoff_dt = None
    if filter_days > 0:
        cutoff_dt = datetime.utcnow() - timedelta(days=filter_days)
    target_date = filter_date  # YYYY-MM-DD string

    for line in lines:
        # Try v6 format first
        m = V6_CLOSE.search(line)
        if m:
            ts_str, pair, direction, units, price, pl_str = m.groups()
            dt = parse_timestamp(ts_str)
            if dt is None:
                continue

            # Apply filters
            if cutoff_dt and dt < cutoff_dt:
                continue
            if target_date and dt.strftime("%Y-%m-%d") != target_date:
                continue

            pl_jpy = parse_pl(pl_str)

            # For non-JPY pairs, the PL might already be in USD —
            # but OANDA reports P/L in account currency (JPY), so treat as JPY
            trade = {
                "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "date": dt.strftime("%Y-%m-%d"),
                "pair": pair.upper(),
                "direction": normalize_direction(direction),
                "units": int(units),
                "price": float(price),
                "pl_jpy": pl_jpy,
                "session": classify_session(dt),
                "is_partial": "PARTIAL" in line.upper(),
                "is_hedge": "HEDGE" in line.upper() or "hedge" in line.lower() or "ヘッジ" in line,
            }
            key = f"{trade['timestamp']}_{pair}_{pl_str}_{units}"
            if key not in seen:
                seen.add(key)
                trades.append(trade)
            continue

        # Try legacy format
        m = LEGACY_CLOSE.search(line)
        if m:
            ts_str, pair, direction, pips = m.groups()
            dt = parse_timestamp(ts_str)
            if dt is None:
                continue
            if cutoff_dt and dt < cutoff_dt:
                continue
            if target_date and dt.strftime("%Y-%m-%d") != target_date:
                continue

            trade = {
                "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "date": dt.strftime("%Y-%m-%d"),
                "pair": pair.upper(),
                "direction": normalize_direction(direction),
                "units": 0,
                "price": 0.0,
                "pl_jpy": float(pips),  # legacy pip = treat as JPY proxy
                "session": classify_session(dt),
                "is_partial": False,
                "is_hedge": False,
            }
            key = f"{trade['timestamp']}_{pair}_{pips}"
            if key not in seen:
                seen.add(key)
                trades.append(trade)

    return trades


def parse_entries(log_path: Path, max_lines: int = 0) -> list[dict]:
    """Parse ENTRY lines for entry count analysis."""
    if not log_path.exists():
        return []

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if max_lines > 0:
        lines = lines[-max_lines:]

    entries = []
    for line in lines:
        m = V6_ENTRY.search(line)
        if m:
            ts_str, pair, direction, units, price = m.groups()
            dt = parse_timestamp(ts_str)
            if dt:
                entries.append({
                    "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "date": dt.strftime("%Y-%m-%d"),
                    "pair": pair.upper(),
                    "direction": normalize_direction(direction),
                    "units": int(units),
                })
    return entries


def compute_stats(trades: list[dict]) -> dict:
    """Compute performance statistics."""
    if not trades:
        return {"total_trades": 0}

    wins = [t for t in trades if t["pl_jpy"] > 0]
    losses = [t for t in trades if t["pl_jpy"] < 0]
    be = [t for t in trades if t["pl_jpy"] == 0]

    total_pl = sum(t["pl_jpy"] for t in trades)
    gross_win = sum(t["pl_jpy"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pl_jpy"] for t in losses)) if losses else 0

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(be),
        "win_rate": round(len(wins) / len(trades), 3) if trades else 0,
        "total_pl_jpy": round(total_pl, 1),
        "avg_pl_jpy": round(total_pl / len(trades), 1) if trades else 0,
        "avg_win_jpy": round(gross_win / len(wins), 1) if wins else 0,
        "avg_loss_jpy": round(gross_loss / len(losses), 1) if losses else 0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0),
        "max_win_jpy": round(max(t["pl_jpy"] for t in wins), 1) if wins else 0,
        "max_loss_jpy": round(min(t["pl_jpy"] for t in losses), 1) if losses else 0,
    }


def breakdown_by(trades: list[dict], key: str) -> dict:
    """Group and compute stats by key."""
    groups = defaultdict(list)
    for t in trades:
        groups[t[key]].append(t)
    return {k: compute_stats(v) for k, v in sorted(groups.items())}


def recent_trend(trades: list[dict], short_w: int = 10, long_w: int = 50) -> dict:
    recent = trades[-short_w:] if len(trades) >= short_w else trades
    longer = trades[-long_w:] if len(trades) >= long_w else trades
    r_stats = compute_stats(recent)
    l_stats = compute_stats(longer)

    r_wr = r_stats.get("win_rate", 0)
    l_wr = l_stats.get("win_rate", 0)
    r_avg = r_stats.get("avg_pl_jpy", 0)
    l_avg = l_stats.get("avg_pl_jpy", 0)

    if r_wr > l_wr and r_avg > l_avg:
        trend = "improving"
    elif r_avg < l_avg:
        trend = "worsening"
    else:
        trend = "stable"

    return {f"last_{short_w}": r_stats, f"last_{long_w}": l_stats, "trend": trend}


def hedge_analysis(trades: list[dict]) -> dict:
    """Analyze hedge trade performance separately."""
    hedges = [t for t in trades if t.get("is_hedge")]
    non_hedges = [t for t in trades if not t.get("is_hedge")]
    return {
        "hedge_trades": compute_stats(hedges),
        "directional_trades": compute_stats(non_hedges),
    }


def daily_summary(trades: list[dict]) -> dict:
    """Summarize P/L by date."""
    by_date = defaultdict(list)
    for t in trades:
        by_date[t["date"]].append(t)

    result = {}
    for date in sorted(by_date.keys()):
        day_trades = by_date[date]
        pl = sum(t["pl_jpy"] for t in day_trades)
        wins = sum(1 for t in day_trades if t["pl_jpy"] > 0)
        losses = sum(1 for t in day_trades if t["pl_jpy"] < 0)
        result[date] = {
            "trades": len(day_trades),
            "wins": wins,
            "losses": losses,
            "pl_jpy": round(pl, 1),
            "win_rate": round(wins / len(day_trades), 2) if day_trades else 0,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="v4 Trade Performance Tracker")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--last", type=int, default=0, help="Parse last N lines of log")
    parser.add_argument("--days", type=int, default=0, help="Filter to last N days")
    parser.add_argument("--date", type=str, default="", help="Filter to specific date (YYYY-MM-DD)")
    args = parser.parse_args()

    trades = parse_trades(LOG_PATH, max_lines=args.last,
                          filter_days=args.days, filter_date=args.date)
    entries = parse_entries(LOG_PATH, max_lines=args.last)

    if not trades:
        msg = "No closed trades found"
        if args.date:
            msg += f" for {args.date}"
        elif args.days:
            msg += f" in last {args.days} days"
        result = {"total_trades": 0, "message": msg}
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"=== Trade Performance v4 ===\n{msg}")
        return

    overall = compute_stats(trades)
    by_pair = breakdown_by(trades, "pair")
    by_session = breakdown_by(trades, "session")
    by_direction = breakdown_by(trades, "direction")
    trend = recent_trend(trades)
    hedge = hedge_analysis(trades)
    daily = daily_summary(trades)

    # Entry count analysis
    entry_count = len(entries)
    close_count = len(trades)

    result = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall": overall,
        "by_pair": by_pair,
        "by_session": by_session,
        "by_direction": by_direction,
        "trend": trend,
        "hedge_analysis": hedge,
        "daily": daily,
        "entry_count": entry_count,
        "close_count": close_count,
    }

    # Write strategy_feedback.json
    try:
        OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"WARN: Could not write {OUTPUT_PATH}: {e}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # ─── Human-readable output ───
    o = overall
    print("=" * 60)
    print("  TRADE PERFORMANCE REPORT (v4)")
    print("=" * 60)

    filter_label = ""
    if args.date:
        filter_label = f" [{args.date}]"
    elif args.days:
        filter_label = f" [last {args.days} days]"

    print(f"\n--- Overall{filter_label} ({o['total_trades']} closes, {entry_count} entries) ---")
    print(f"  Win Rate: {o['win_rate']:.1%}  |  PF: {o['profit_factor']}")
    print(f"  Total P/L: {o['total_pl_jpy']:+,.0f}円  |  Avg: {o['avg_pl_jpy']:+,.1f}円/trade")
    print(f"  Avg Win: +{o['avg_win_jpy']:,.1f}円  |  Avg Loss: -{o['avg_loss_jpy']:,.1f}円")
    print(f"  Best: {o['max_win_jpy']:+,.1f}円  |  Worst: {o['max_loss_jpy']:+,.1f}円")

    # Hedge analysis
    h = hedge
    if h["hedge_trades"]["total_trades"] > 0:
        ht = h["hedge_trades"]
        dt_ = h["directional_trades"]
        print(f"\n--- Hedge vs Directional ---")
        print(f"  Hedge:  {ht['total_trades']} trades, WR={ht['win_rate']:.0%}, P/L={ht['total_pl_jpy']:+,.0f}円")
        print(f"  Direct: {dt_['total_trades']} trades, WR={dt_['win_rate']:.0%}, P/L={dt_['total_pl_jpy']:+,.0f}円")

    print(f"\n--- By Pair ---")
    for pair, s in by_pair.items():
        emoji = "✅" if s['total_pl_jpy'] > 0 else "❌"
        print(f"  {pair:8s}: WR={s['win_rate']:.0%} PF={s['profit_factor']:.2f} "
              f"trades={s['total_trades']} P/L={s['total_pl_jpy']:+,.0f}円 {emoji}")

    print(f"\n--- By Session ---")
    for sess, s in by_session.items():
        print(f"  {sess:10s}: WR={s['win_rate']:.0%} trades={s['total_trades']} "
              f"P/L={s['total_pl_jpy']:+,.0f}円")

    print(f"\n--- By Direction ---")
    for d, s in by_direction.items():
        print(f"  {d:6s}: WR={s['win_rate']:.0%} PF={s['profit_factor']:.2f} "
              f"trades={s['total_trades']} P/L={s['total_pl_jpy']:+,.0f}円")

    print(f"\n--- Daily P/L ---")
    for date, d in daily.items():
        emoji = "📈" if d['pl_jpy'] > 0 else "📉" if d['pl_jpy'] < 0 else "➡️"
        print(f"  {date}: {d['trades']}trades WR={d['win_rate']:.0%} P/L={d['pl_jpy']:+,.0f}円 {emoji}")

    print(f"\n--- Trend ---")
    t = trend
    for k, v in t.items():
        if k == "trend":
            marker = "📈" if v == "improving" else "📉" if v == "worsening" else "➡️"
            print(f"  Direction: {v.upper()} {marker}")
        else:
            print(f"  {k}: WR={v['win_rate']:.0%} avg={v['avg_pl_jpy']:+,.1f}円 ({v['total_trades']} trades)")

    # ─── Recommendations ───
    print(f"\n--- Recommendations ---")
    if o["win_rate"] < 0.4:
        print("  ⚠️  WIN RATE < 40% — 方向判断を見直せ")
    if o["profit_factor"] < 1.0:
        print("  ⚠️  PF < 1.0 — 負けている。SL/TP比率を見直せ")
    if o.get("avg_loss_jpy", 0) > o.get("avg_win_jpy", 0) * 1.5:
        print("  ⚠️  平均損失 > 平均利益×1.5 — 損切り遅延 or 利確遅延")

    # Pair-specific warnings
    for pair, s in by_pair.items():
        if s["total_trades"] >= 3 and s["win_rate"] < 0.3:
            print(f"  ⚠️  {pair}: WR={s['win_rate']:.0%} — このペアは避けるかサイズ縮小")
        if s["total_trades"] >= 3 and s["total_pl_jpy"] < -100:
            print(f"  ⚠️  {pair}: P/L={s['total_pl_jpy']:+,.0f}円 — 損失ペア。テーゼ再検証")

    # Hedge warning
    if h["hedge_trades"]["total_trades"] > 0:
        ht = h["hedge_trades"]
        if ht["total_pl_jpy"] < -50:
            print(f"  ⚠️  ヘッジ損失={ht['total_pl_jpy']:+,.0f}円 — ヘッジが効いていない")

    # Direction imbalance
    long_n = by_direction.get("LONG", {}).get("total_trades", 0)
    short_n = by_direction.get("SHORT", {}).get("total_trades", 0)
    total_n = long_n + short_n
    if total_n > 5:
        ratio = long_n / total_n
        if ratio > 0.75:
            print(f"  ⚠️  LONG偏重 ({ratio:.0%}) — SHORT機会を探せ")
        elif ratio < 0.25:
            print(f"  ⚠️  SHORT偏重 ({1-ratio:.0%}) — LONG機会を探せ")

    print()


if __name__ == "__main__":
    main()
