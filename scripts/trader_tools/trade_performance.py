#!/usr/bin/env python3
"""v3 Trade Performance Tracker — parses live_trade_log.txt for macro-intel analysis.

Outputs performance stats:
- Win rate, profit factor, avg pip P/L by agent
- Per-pair breakdown
- Session breakdown (Tokyo/London/NY)
- Directional breakdown (LONG vs SHORT)
- Recent trend (last 10 vs last 50)

Usage:
    python scripts/trader_tools/trade_performance.py [--json] [--last N]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_PATH = BASE_DIR / "logs" / "live_trade_log.txt"
OUTPUT_PATH = BASE_DIR / "logs" / "strategy_feedback.json"
PREDICTION_TRACKER_PATH = BASE_DIR / "logs" / "prediction_tracker.json"

# Patterns to extract trade results from log lines
# Matches: [2026-03-19T10:30:00Z] FAST: CLOSED EUR_USD LONG +3.2pip
# Matches: [2026-03-19T10:30:00Z] SWING: CLOSED GBP_USD SHORT -5.1pip
# Also handles: TP_HIT, SL_HIT, PARTIAL, CLOSE, closed variants
CLOSE_PATTERN = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?)\]\s+"
    r"(FAST|SWING|SCALP|MONITOR)[:\s]+"
    r".*?(CLOSED?|TP_HIT|SL_HIT|PARTIAL|CUT|TRAIL_HIT)"
    r".*?"
    r"(USD_JPY|EUR_USD|GBP_USD|AUD_USD|EUR_JPY|GBP_JPY|AUD_JPY)"
    r".*?"
    r"(LONG|SHORT|L|S)"
    r".*?"
    r"([+-]?\d+\.?\d*)\s*pip",
    re.IGNORECASE,
)

# Also try to match pip results in different formats
RESULT_PATTERN = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?)\]\s+"
    r"(FAST|SWING|SCALP|MONITOR)[:\s]+"
    r".*?"
    r"(USD_JPY|EUR_USD|GBP_USD|AUD_USD|EUR_JPY|GBP_JPY|AUD_JPY)"
    r".*?"
    r"(LONG|SHORT|L|S)"
    r".*?"
    r"(?:result|P/L|pips?|UPL)[=:\s]*([+-]?\d+\.?\d*)",
    re.IGNORECASE,
)

# Match monitor auto-close lines
MONITOR_CLOSE_PATTERN = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?)\]\s+"
    r"(?:MONITOR|AUTO)[:\s]+"
    r".*?(CLOSED?|TP_HIT|SL_HIT|CUT|TRAIL)"
    r".*?"
    r"(USD_JPY|EUR_USD|GBP_USD|AUD_USD|EUR_JPY|GBP_JPY|AUD_JPY)"
    r".*?"
    r"([+-]?\d+\.?\d*)\s*pip",
    re.IGNORECASE,
)


def classify_session(ts_str: str) -> str:
    """Classify UTC hour into trading session."""
    try:
        ts = datetime.fromisoformat(ts_str.rstrip("Z"))
        hour = ts.hour
    except (ValueError, AttributeError):
        return "unknown"
    if 0 <= hour < 7:
        return "tokyo"
    elif 7 <= hour < 15:
        return "london"
    elif 15 <= hour < 22:
        return "newyork"
    else:
        return "late"


def normalize_direction(d: str) -> str:
    d = d.upper()
    if d in ("L", "LONG"):
        return "LONG"
    return "SHORT"


def normalize_agent(a: str) -> str:
    a = a.upper()
    if a in ("FAST", "SCALP"):
        return "scalp-fast"
    elif a == "SWING":
        return "swing-trader"
    elif a in ("MONITOR", "AUTO"):
        return "monitor"
    return a.lower()


def parse_trades(log_path: Path, max_lines: int = 0) -> list[dict]:
    """Parse trade log and extract closed trades with pip results."""
    if not log_path.exists():
        print(f"WARN: {log_path} not found", file=sys.stderr)
        return []

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if max_lines > 0:
        lines = lines[-max_lines:]

    trades = []
    seen = set()

    for line in lines:
        m = CLOSE_PATTERN.search(line)
        if not m:
            m = RESULT_PATTERN.search(line)
        if not m:
            m = MONITOR_CLOSE_PATTERN.search(line)
            if m:
                ts, action, pair, pips = m.group(1), "MONITOR", m.group(3), m.group(4)
                direction = "UNKNOWN"
                # Try to infer direction from context
                if "LONG" in line.upper() or " L " in line.upper():
                    direction = "LONG"
                elif "SHORT" in line.upper() or " S " in line.upper():
                    direction = "SHORT"
                trade = {
                    "timestamp": ts,
                    "agent": normalize_agent("MONITOR"),
                    "pair": pair,
                    "direction": direction,
                    "pips": float(pips),
                    "session": classify_session(ts),
                }
                key = f"{ts}_{pair}_{pips}"
                if key not in seen:
                    seen.add(key)
                    trades.append(trade)
                continue

        if not m:
            continue

        groups = m.groups()
        if len(groups) == 6:
            ts, agent, _, pair, direction, pips = groups
        elif len(groups) == 5:
            ts, agent, pair, direction, pips = groups
        else:
            continue

        trade = {
            "timestamp": ts,
            "agent": normalize_agent(agent),
            "pair": pair,
            "direction": normalize_direction(direction),
            "pips": float(pips),
            "session": classify_session(ts),
        }
        key = f"{ts}_{pair}_{pips}"
        if key not in seen:
            seen.add(key)
            trades.append(trade)

    return trades


def compute_stats(trades: list[dict]) -> dict:
    """Compute performance statistics from trade list."""
    if not trades:
        return {"total_trades": 0, "message": "No closed trades found in log"}

    wins = [t for t in trades if t["pips"] > 0]
    losses = [t for t in trades if t["pips"] < 0]
    breakeven = [t for t in trades if t["pips"] == 0]

    total_pips = sum(t["pips"] for t in trades)
    gross_profit = sum(t["pips"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pips"] for t in losses)) if losses else 0

    stats = {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate": len(wins) / len(trades) if trades else 0,
        "total_pips": round(total_pips, 1),
        "avg_pip": round(total_pips / len(trades), 2) if trades else 0,
        "avg_win": round(gross_profit / len(wins), 2) if wins else 0,
        "avg_loss": round(gross_loss / len(losses), 2) if losses else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0,
        "max_win": round(max(t["pips"] for t in wins), 1) if wins else 0,
        "max_loss": round(min(t["pips"] for t in losses), 1) if losses else 0,
    }
    return stats


def breakdown_by(trades: list[dict], key: str) -> dict:
    """Group trades by a key and compute stats for each group."""
    groups = defaultdict(list)
    for t in trades:
        groups[t[key]].append(t)
    return {k: compute_stats(v) for k, v in sorted(groups.items())}


def recent_trend(trades: list[dict], short_window: int = 10, long_window: int = 50) -> dict:
    """Compare recent performance to longer-term."""
    recent = trades[-short_window:] if len(trades) >= short_window else trades
    longer = trades[-long_window:] if len(trades) >= long_window else trades

    recent_stats = compute_stats(recent)
    longer_stats = compute_stats(longer)

    improving = (
        recent_stats.get("win_rate", 0) > longer_stats.get("win_rate", 0)
        and recent_stats.get("avg_pip", 0) > longer_stats.get("avg_pip", 0)
    )

    return {
        f"last_{short_window}": recent_stats,
        f"last_{long_window}": longer_stats,
        "trend": "improving" if improving else "worsening" if recent_stats.get("avg_pip", 0) < longer_stats.get("avg_pip", 0) else "stable",
    }


def parse_prediction_accuracy(log_path: Path) -> dict:
    """Parse REFLECTION entries for prediction accuracy tracking."""
    if not log_path.exists():
        return {"total_reflections": 0}

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    text = "\n".join(lines)

    # Count prediction/thesis outcomes from REFLECTION and SWING REVIEW entries
    # scalp-fast writes "Prediction was right/wrong", swing-trader writes "Thesis was right/wrong"
    right_pattern = re.compile(r"(?:Prediction|Thesis) was right", re.IGNORECASE)
    wrong_pattern = re.compile(r"(?:Prediction|Thesis) was wrong", re.IGNORECASE)

    # All-time counts
    all_right = len(right_pattern.findall(text))
    all_wrong = len(wrong_pattern.findall(text))
    all_total = all_right + all_wrong

    # Recent counts (last 200 lines — roughly last few hours of activity)
    recent_text = "\n".join(lines[-200:]) if len(lines) > 200 else text
    recent_right = len(right_pattern.findall(recent_text))
    recent_wrong = len(wrong_pattern.findall(recent_text))
    recent_total = recent_right + recent_wrong

    # Count REFLECTION entries (compliance check)
    reflection_count = len(re.findall(r"REFLECTION:", text))
    swing_review_count = len(re.findall(r"SWING REVIEW:", text))
    pattern_check_count = len(re.findall(r"PATTERN CHECK:", text))
    macro_review_count = len(re.findall(r"\[MACRO-INTEL REVIEW\]", text))

    result = {
        "prediction_right": all_right,
        "prediction_wrong": all_wrong,
        "prediction_total": all_total,
        "prediction_accuracy": round(all_right / all_total, 2) if all_total > 0 else None,
        "recent_prediction_right": recent_right,
        "recent_prediction_wrong": recent_wrong,
        "recent_prediction_total": recent_total,
        "recent_prediction_accuracy": round(recent_right / recent_total, 2) if recent_total > 0 else None,
        "reflection_count": reflection_count,
        "swing_review_count": swing_review_count,
        "pattern_check_count": pattern_check_count,
        "macro_review_count": macro_review_count,
    }

    # Trend detection: is recent accuracy worse than all-time?
    if all_total >= 10 and recent_total >= 5:
        all_acc = all_right / all_total
        recent_acc = recent_right / recent_total
        if recent_acc < all_acc - 0.15:
            result["prediction_trend"] = "deteriorating"
        elif recent_acc > all_acc + 0.15:
            result["prediction_trend"] = "improving"
        else:
            result["prediction_trend"] = "stable"

    return result


def parse_prediction_tracker() -> dict:
    """Parse prediction_tracker.json for detailed prediction accuracy analysis.

    Breaks down by: agent, pair, session, direction, score_agreed.
    This complements parse_prediction_accuracy() which only reads log text.
    """
    try:
        preds = json.loads(PREDICTION_TRACKER_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"status": "no_data", "total": 0}

    resolved = [p for p in preds if p.get("status") in ("correct", "partial", "wrong")]
    open_preds = [p for p in preds if p.get("status") == "open"]

    if not resolved:
        return {"status": "no_resolved", "total": len(preds), "open": len(open_preds)}

    def _accuracy(items: list) -> dict | None:
        if not items:
            return None
        correct = sum(1 for p in items if p["status"] in ("correct", "partial"))
        wrong = sum(1 for p in items if p["status"] == "wrong")
        total = len(items)
        avg_pips = round(sum(p.get("pips", 0) or 0 for p in items) / total, 2) if total else 0
        return {
            "correct": correct,
            "wrong": wrong,
            "total": total,
            "accuracy_pct": round(correct / total * 100) if total else 0,
            "avg_pips": avg_pips,
        }

    # Overall
    overall = _accuracy(resolved)

    # By agent
    by_agent = {}
    for agent in set(p.get("agent", "unknown") for p in resolved):
        agent_preds = [p for p in resolved if p.get("agent") == agent]
        by_agent[agent] = _accuracy(agent_preds)

    # By pair
    by_pair = {}
    for pair in set(p.get("pair", "unknown") for p in resolved):
        pair_preds = [p for p in resolved if p.get("pair") == pair]
        by_pair[pair] = _accuracy(pair_preds)

    # By session
    by_session = {}
    for session in set(p.get("session", "unknown") for p in resolved):
        session_preds = [p for p in resolved if p.get("session") == session]
        by_session[session] = _accuracy(session_preds)

    # By direction
    by_direction = {}
    for direction in set(p.get("direction", "unknown") for p in resolved):
        dir_preds = [p for p in resolved if p.get("direction") == direction]
        by_direction[direction] = _accuracy(dir_preds)

    # Score agreement analysis — the key insight
    agreed = [p for p in resolved if p.get("score_agreed") is True]
    disagreed = [p for p in resolved if p.get("score_agreed") is False]
    score_analysis = {
        "score_agreed": _accuracy(agreed),
        "score_disagreed": _accuracy(disagreed),
    }
    # Recommendation based on data
    if agreed and disagreed:
        ag_acc = (score_analysis["score_agreed"] or {}).get("accuracy_pct", 0)
        dis_acc = (score_analysis["score_disagreed"] or {}).get("accuracy_pct", 0)
        if dis_acc > ag_acc + 10:
            score_analysis["recommendation"] = "PREDICTIONS_BEAT_SCORE — trust your judgment more"
        elif ag_acc > dis_acc + 10:
            score_analysis["recommendation"] = "SCORE_MORE_RELIABLE — lean on score confirmation"
        else:
            score_analysis["recommendation"] = "BALANCED — both sources similar, use together"

    # Recent trend (last 10 vs all)
    recent_10 = resolved[-10:] if len(resolved) > 10 else resolved
    trend_info = {
        "last_10": _accuracy(recent_10),
        "all_time": overall,
    }
    if len(resolved) >= 15 and recent_10:
        all_acc = overall["accuracy_pct"]
        recent_acc = trend_info["last_10"]["accuracy_pct"]
        if recent_acc < all_acc - 15:
            trend_info["trend"] = "deteriorating"
        elif recent_acc > all_acc + 15:
            trend_info["trend"] = "improving"
        else:
            trend_info["trend"] = "stable"

    return {
        "status": "active",
        "total_predictions": len(preds),
        "resolved": len(resolved),
        "open": len(open_preds),
        "overall": overall,
        "by_agent": by_agent,
        "by_pair": by_pair,
        "by_session": by_session,
        "by_direction": by_direction,
        "score_analysis": score_analysis,
        "trend": trend_info,
    }


def main():
    parser = argparse.ArgumentParser(description="v3 Trade Performance Tracker")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--last", type=int, default=0, help="Only parse last N lines of log")
    args = parser.parse_args()

    trades = parse_trades(LOG_PATH, max_lines=args.last)

    if not trades:
        result = {"total_trades": 0, "message": "No closed trades found in live_trade_log.txt"}
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("=== Trade Performance ===")
            print("No closed trades found in logs/live_trade_log.txt")
            print("Ensure trade closes include pip result, e.g.:")
            print('  [2026-03-19T10:30:00Z] FAST: CLOSED EUR_USD LONG +3.2pip')
        return

    overall = compute_stats(trades)
    by_agent = breakdown_by(trades, "agent")
    by_pair = breakdown_by(trades, "pair")
    by_session = breakdown_by(trades, "session")
    by_direction = breakdown_by(trades, "direction")
    trend = recent_trend(trades)

    prediction = parse_prediction_accuracy(LOG_PATH)
    prediction_tracker = parse_prediction_tracker()

    result = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall": overall,
        "by_agent": by_agent,
        "by_pair": by_pair,
        "by_session": by_session,
        "by_direction": by_direction,
        "trend": trend,
        "prediction_accuracy": prediction,
        "prediction_tracker": prediction_tracker,
    }

    # Write to strategy_feedback.json for other agents to read
    try:
        OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"WARN: Could not write {OUTPUT_PATH}: {e}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Human-readable output
    print("=" * 60)
    print("  TRADE PERFORMANCE REPORT (v3)")
    print("=" * 60)

    o = overall
    print(f"\n--- Overall ({o['total_trades']} trades) ---")
    print(f"  Win Rate: {o['win_rate']:.1%}  |  Profit Factor: {o['profit_factor']}")
    print(f"  Total: {o['total_pips']:+.1f}pip  |  Avg: {o['avg_pip']:+.2f}pip/trade")
    print(f"  Avg Win: +{o['avg_win']}pip  |  Avg Loss: -{o['avg_loss']}pip")
    print(f"  Best: {o['max_win']:+.1f}pip  |  Worst: {o['max_loss']:+.1f}pip")

    print(f"\n--- By Agent ---")
    for agent, s in by_agent.items():
        print(f"  {agent:15s}: WR={s['win_rate']:.0%} PF={s['profit_factor']:.2f} "
              f"trades={s['total_trades']} total={s['total_pips']:+.1f}pip")

    print(f"\n--- By Pair ---")
    for pair, s in by_pair.items():
        wr_str = f"{s['win_rate']:.0%}"
        emoji = "+" if s['total_pips'] > 0 else "-" if s['total_pips'] < 0 else "="
        print(f"  {pair:8s}: WR={wr_str:>4s} PF={s['profit_factor']:.2f} "
              f"trades={s['total_trades']} total={s['total_pips']:+.1f}pip {emoji}")

    print(f"\n--- By Session ---")
    for sess, s in by_session.items():
        print(f"  {sess:10s}: WR={s['win_rate']:.0%} trades={s['total_trades']} "
              f"total={s['total_pips']:+.1f}pip")

    print(f"\n--- By Direction ---")
    for d, s in by_direction.items():
        print(f"  {d:6s}: WR={s['win_rate']:.0%} PF={s['profit_factor']:.2f} "
              f"trades={s['total_trades']} total={s['total_pips']:+.1f}pip")

    print(f"\n--- Trend ---")
    t = trend
    for k, v in t.items():
        if k == "trend":
            print(f"  Direction: {v.upper()}")
        else:
            print(f"  {k}: WR={v['win_rate']:.0%} avg={v['avg_pip']:+.2f}pip ({v['total_trades']} trades)")

    # Prediction accuracy
    p = prediction
    if p["prediction_total"] > 0:
        print(f"\n--- Prediction Accuracy ---")
        print(f"  All-time: {p['prediction_right']}R / {p['prediction_wrong']}W = "
              f"{p['prediction_accuracy']:.0%} ({p['prediction_total']} predictions)")
        if p.get("recent_prediction_total", 0) > 0:
            print(f"  Recent:   {p['recent_prediction_right']}R / {p['recent_prediction_wrong']}W = "
                  f"{p['recent_prediction_accuracy']:.0%} ({p['recent_prediction_total']} predictions)")
        if "prediction_trend" in p:
            trend_str = p["prediction_trend"].upper()
            if trend_str == "DETERIORATING":
                print(f"  !! PREDICTION ACCURACY DECLINING — lean heavier on score confirmation")
            elif trend_str == "IMPROVING":
                print(f"  >> Prediction accuracy improving — good sign")

    # Prediction Tracker (from prediction_tracker.json)
    pt = prediction_tracker
    if pt.get("status") == "active" and pt.get("overall"):
        print(f"\n--- Prediction Tracker (prediction_tracker.json) ---")
        ov = pt["overall"]
        print(f"  Overall: {ov['correct']}/{ov['total']} correct = {ov['accuracy_pct']}% "
              f"(avg {ov['avg_pips']:+.1f}pip)")
        # Score analysis
        sa = pt.get("score_analysis", {})
        if sa.get("score_agreed"):
            ag = sa["score_agreed"]
            print(f"  Score Agreed:    {ag['accuracy_pct']}% ({ag['total']} predictions)")
        if sa.get("score_disagreed"):
            da = sa["score_disagreed"]
            print(f"  Score Disagreed: {da['accuracy_pct']}% ({da['total']} predictions)")
        if sa.get("recommendation"):
            print(f"  >> {sa['recommendation']}")
        # By pair
        if pt.get("by_pair"):
            print(f"  By pair:")
            for pair, stats in sorted(pt["by_pair"].items()):
                if stats:
                    print(f"    {pair:8s}: {stats['accuracy_pct']}% ({stats['total']}pred, avg {stats['avg_pips']:+.1f}pip)")
        # Trend
        tr = pt.get("trend", {})
        if tr.get("trend"):
            print(f"  Trend: {tr['trend'].upper()}")
    else:
        msg = "agents not writing predictions yet" if pt.get("total", 0) == 0 else f"{pt.get('open', 0)} open, none resolved yet"
        print(f"\n--- Prediction Tracker: {msg} ---")

    print(f"\n--- Self-Improvement Compliance ---")
    print(f"  REFLECTION entries: {p['reflection_count']} | SWING REVIEW: {p['swing_review_count']}")
    print(f"  PATTERN CHECK: {p['pattern_check_count']} | MACRO-INTEL REVIEW: {p['macro_review_count']}")

    # Actionable recommendations
    print(f"\n--- Recommendations ---")
    if o["win_rate"] < 0.4:
        print("  !! LOW WIN RATE — direction may be wrong. Check bias alignment.")
    if o["profit_factor"] < 1.0:
        print("  !! NEGATIVE EDGE — losing money per trade. Review SL/TP ratio.")
    if o.get("avg_loss", 0) > o.get("avg_win", 0) * 1.5:
        print("  !! LOSSES TOO LARGE vs WINS — tighten SL or widen TP.")

    # Per-pair warnings
    for pair, s in by_pair.items():
        if s["total_trades"] >= 3 and s["win_rate"] < 0.3:
            print(f"  !! {pair}: WR={s['win_rate']:.0%} — consider avoiding or reducing size.")

    # Direction imbalance
    long_trades = by_direction.get("LONG", {}).get("total_trades", 0)
    short_trades = by_direction.get("SHORT", {}).get("total_trades", 0)
    total = long_trades + short_trades
    if total > 5:
        ratio = long_trades / total if total else 0.5
        if ratio > 0.75:
            print(f"  !! LONG-BIASED ({ratio:.0%} of trades). Consider more SHORT entries.")
        elif ratio < 0.25:
            print(f"  !! SHORT-BIASED ({1-ratio:.0%} of trades). Consider more LONG entries.")

    print()


if __name__ == "__main__":
    main()
