#!/usr/bin/env python3
"""OANDA-based Performance Analysis — ground truth from the API, not log files.

Queries OANDA Transaction API for ORDER_FILL events and computes:
- Daily realized P&L
- Win rate, average win/loss, realized expectancy
- Best/worst trades
- Best N-hour windows (streak detection)
- Per-pair breakdown

Usage:
    python3 tools/oanda_performance.py                 # last 7 days
    python3 tools/oanda_performance.py --days 14       # last 14 days
    python3 tools/oanda_performance.py --date 2026-04-07  # specific day detail
    python3 tools/oanda_performance.py --streak 12     # best 12-hour window
    python3 tools/oanda_performance.py --json          # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "env.toml"
JST = timezone(timedelta(hours=9))


def load_config():
    text = CONFIG_PATH.read_text()
    token = [l.split("=")[1].strip().strip('"') for l in text.split("\n") if l.startswith("oanda_token")][0]
    acct = [l.split("=")[1].strip().strip('"') for l in text.split("\n") if l.startswith("oanda_account_id")][0]
    return token, acct


def payoff_metrics(pls: list[float]) -> dict:
    """Realized payoff shape.

    Planned R:R is a hypothesis. These metrics show whether the closed-trade
    distribution actually has positive expectancy.
    """
    if not pls:
        return {
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "rr_ratio": 0.0,
            "expectancy": 0.0,
            "profit_factor": None,
            "break_even_win_rate": None,
        }

    wins = [pl for pl in pls if pl > 0]
    losses = [pl for pl in pls if pl < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    if avg_loss < 0:
        rr_ratio = avg_win / abs(avg_loss)
    elif avg_win > 0 and not losses:
        rr_ratio = float("inf")
    else:
        rr_ratio = 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    break_even_win_rate = None
    if avg_win > 0 and avg_loss < 0:
        break_even_win_rate = abs(avg_loss) / (avg_win + abs(avg_loss))
    elif avg_win > 0 and not losses:
        break_even_win_rate = 0.0

    profit_factor = None
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        profit_factor = float("inf")

    return {
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(pls) if pls else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "rr_ratio": rr_ratio,
        "expectancy": sum(pls) / len(pls),
        "profit_factor": profit_factor,
        "break_even_win_rate": break_even_win_rate,
    }


def fetch_fills(token: str, acct: str, from_date: str) -> list[dict]:
    """Fetch all ORDER_FILL transactions with non-zero P&L from OANDA."""
    base = "https://api-fxtrade.oanda.com"
    url = f"{base}/v3/accounts/{acct}/transactions?from={from_date}&type=ORDER_FILL"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    data = json.loads(urllib.request.urlopen(req).read())

    fills = []
    for page_url in data.get("pages", []):
        req2 = urllib.request.Request(page_url, headers={"Authorization": f"Bearer {token}"})
        page_data = json.loads(urllib.request.urlopen(req2).read())
        for tx in page_data.get("transactions", []):
            if tx.get("type") == "ORDER_FILL":
                pl = float(tx.get("pl", 0))
                if pl != 0:
                    fills.append({
                        "time": tx["time"][:19],
                        "instrument": tx.get("instrument", ""),
                        "units": int(tx.get("units", 0)),
                        "pl": pl,
                        "balance": float(tx.get("accountBalance", 0)),
                        "price": float(tx.get("price", 0)),
                        "reason": tx.get("reason", ""),
                    })
    return fills


def analyze(fills: list[dict], streak_hours: int = 12) -> dict:
    """Analyze fills and return comprehensive stats."""
    if not fills:
        return {"error": "No fills found"}

    # Daily P&L (grouped by JST date)
    daily = defaultdict(lambda: {"pl": 0, "wins": 0, "losses": 0, "trades": 0, "win_pls": [], "loss_pls": []})
    pair_stats = defaultdict(lambda: {"pl": 0, "wins": 0, "losses": 0, "trades": 0, "win_pls": [], "loss_pls": []})

    all_wins = []
    all_losses = []

    for f in fills:
        utc_dt = datetime.fromisoformat(f["time"]).replace(tzinfo=timezone.utc)
        day = utc_dt.astimezone(JST).strftime("%Y-%m-%d")
        d = daily[day]
        d["pl"] += f["pl"]
        d["trades"] += 1
        if f["pl"] > 0:
            d["wins"] += 1
            d["win_pls"].append(f["pl"])
            all_wins.append(f)
        else:
            d["losses"] += 1
            d["loss_pls"].append(f["pl"])
            all_losses.append(f)

        pair = f["instrument"]
        p = pair_stats[pair]
        p["pl"] += f["pl"]
        p["trades"] += 1
        if f["pl"] > 0:
            p["wins"] += 1
            p["win_pls"].append(f["pl"])
        else:
            p["losses"] += 1
            p["loss_pls"].append(f["pl"])

    # Overall stats
    total_pl = sum(f["pl"] for f in fills)
    total_trades = len(fills)
    overall_payoff = payoff_metrics([f["pl"] for f in fills])
    total_wins = overall_payoff["wins"]
    total_losses = overall_payoff["losses"]
    avg_win = overall_payoff["avg_win"]
    avg_loss = overall_payoff["avg_loss"]
    win_rate = overall_payoff["win_rate"] * 100 if total_trades else 0

    # Best/worst trades
    best_trades = sorted(fills, key=lambda x: x["pl"], reverse=True)[:5]
    worst_trades = sorted(fills, key=lambda x: x["pl"])[:5]

    # Best N-hour window
    best_streak = {"gain": 0}
    for i in range(len(fills)):
        start_time = datetime.fromisoformat(fills[i]["time"])
        start_bal = fills[i]["balance"] - fills[i]["pl"]  # balance before this trade
        for j in range(i + 1, len(fills)):
            end_time = datetime.fromisoformat(fills[j]["time"])
            hours = (end_time - start_time).total_seconds() / 3600
            if streak_hours - 2 <= hours <= streak_hours + 2:
                gain = fills[j]["balance"] - start_bal
                pct = gain / start_bal * 100 if start_bal > 0 else 0
                if pct > best_streak.get("pct", 0):
                    best_streak = {
                        "start": fills[i]["time"],
                        "end": fills[j]["time"],
                        "hours": round(hours, 1),
                        "start_bal": round(start_bal),
                        "end_bal": round(fills[j]["balance"]),
                        "gain": round(gain),
                        "pct": round(pct, 1),
                        "trades": j - i + 1,
                    }

    # Daily summary
    daily_summary = []
    for day in sorted(daily.keys()):
        d = daily[day]
        metrics = payoff_metrics(d["win_pls"] + d["loss_pls"])
        daily_summary.append({
            "date": day,
            "pl": round(d["pl"]),
            "trades": d["trades"],
            "wins": metrics["wins"],
            "losses": metrics["losses"],
            "win_rate": round(metrics["win_rate"] * 100, 1),
            "avg_win": round(metrics["avg_win"]),
            "avg_loss": round(metrics["avg_loss"]),
            "rr_ratio": round(metrics["rr_ratio"], 2),
            "expectancy": round(metrics["expectancy"]),
            "profit_factor": None if metrics["profit_factor"] is None else round(metrics["profit_factor"], 2),
            "break_even_win_rate": None if metrics["break_even_win_rate"] is None else round(metrics["break_even_win_rate"] * 100, 1),
        })

    # Pair summary
    pair_summary = []
    for pair in sorted(pair_stats.keys()):
        p = pair_stats[pair]
        metrics = payoff_metrics(p["win_pls"] + p["loss_pls"])
        pair_summary.append({
            "pair": pair,
            "pl": round(p["pl"]),
            "trades": p["trades"],
            "wins": metrics["wins"],
            "losses": metrics["losses"],
            "win_rate": round(metrics["win_rate"] * 100, 1),
            "avg_win": round(metrics["avg_win"]),
            "avg_loss": round(metrics["avg_loss"]),
            "rr_ratio": round(metrics["rr_ratio"], 2),
            "expectancy": round(metrics["expectancy"]),
            "profit_factor": None if metrics["profit_factor"] is None else round(metrics["profit_factor"], 2),
            "break_even_win_rate": None if metrics["break_even_win_rate"] is None else round(metrics["break_even_win_rate"] * 100, 1),
        })

    return {
        "period": f"{fills[0]['time'][:10]} to {fills[-1]['time'][:10]}",
        "total_pl": round(total_pl),
        "total_trades": total_trades,
        "wins": total_wins,
        "losses": total_losses,
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win),
        "avg_loss": round(avg_loss),
        "rr_ratio": round(overall_payoff["rr_ratio"], 2),
        "expectancy": round(overall_payoff["expectancy"]),
        "profit_factor": None if overall_payoff["profit_factor"] is None else round(overall_payoff["profit_factor"], 2),
        "break_even_win_rate": None if overall_payoff["break_even_win_rate"] is None else round(overall_payoff["break_even_win_rate"] * 100, 1),
        "best_trades": [{"time": t["time"], "pair": t["instrument"], "units": t["units"], "pl": round(t["pl"])} for t in best_trades],
        "worst_trades": [{"time": t["time"], "pair": t["instrument"], "units": t["units"], "pl": round(t["pl"])} for t in worst_trades],
        "best_streak": best_streak if best_streak.get("start") else None,
        "daily": daily_summary,
        "by_pair": pair_summary,
    }


def print_report(stats: dict, detail_date: str | None = None):
    """Print human-readable report."""
    if "error" in stats:
        print(stats["error"])
        return

    print(f"=== OANDA Performance Report ({stats['period']}) ===")
    print(f"Total P&L: {stats['total_pl']:+,} JPY | Trades: {stats['total_trades']}")
    be_wr = stats.get("break_even_win_rate")
    pf = stats.get("profit_factor")
    pf_text = "" if pf is None else "inf" if pf == float("inf") else f"{pf:.2f}"
    be_text = f" | Break-even WR: {be_wr}%" if be_wr is not None else ""
    pf_suffix = f" | PF: {pf_text}" if pf_text else ""
    print(f"Win Rate: {stats['win_rate']}% ({stats['wins']}W / {stats['losses']}L) | Expectancy: {stats['expectancy']:+,} JPY/trade{be_text}")
    print(f"Avg Win: {stats['avg_win']:+,} | Avg Loss: {stats['avg_loss']:+,} | R:R = {stats['rr_ratio']}{pf_suffix}")
    print()

    # Daily
    print("--- Daily P&L ---")
    for d in stats["daily"]:
        marker = " ★" if d["pl"] > 3000 else " ✗" if d["pl"] < -2000 else ""
        if detail_date and d["date"] != detail_date:
            continue
        be_wr = f" | BE {d['break_even_win_rate']}%" if d.get("break_even_win_rate") is not None else ""
        print(
            f"  {d['date']}: {d['pl']:+,} JPY | {d['trades']}t | WR {d['win_rate']}% "
            f"| EV {d['expectancy']:+,} | avg_W {d['avg_win']:+,} avg_L {d['avg_loss']:+,}"
            f" | R:R {d['rr_ratio']}{be_wr}{marker}"
        )
    print()

    # By pair
    print("--- By Pair ---")
    for p in sorted(stats["by_pair"], key=lambda x: x["pl"], reverse=True):
        be_wr = f" | BE {p['break_even_win_rate']}%" if p.get("break_even_win_rate") is not None else ""
        print(
            f"  {p['pair']}: {p['pl']:+,} JPY | {p['trades']}t | WR {p['win_rate']}% "
            f"| EV {p['expectancy']:+,} | R:R {p['rr_ratio']}{be_wr}"
        )
    print()

    # Best streak
    if stats.get("best_streak") and stats["best_streak"].get("start"):
        s = stats["best_streak"]
        print(f"--- Best {s['hours']}h Window ---")
        print(f"  {s['start']} → {s['end']}")
        print(f"  {s['start_bal']:,} → {s['end_bal']:,} = +{s['gain']:,} JPY ({s['pct']}%)")
        print(f"  {s['trades']} trades")
    print()

    # Top 5 / Bottom 5
    print("--- Best Trades ---")
    for t in stats["best_trades"]:
        print(f"  {t['time'][:16]} {t['pair']} {t['units']}u → {t['pl']:+,}")
    print("--- Worst Trades ---")
    for t in stats["worst_trades"]:
        print(f"  {t['time'][:16]} {t['pair']} {t['units']}u → {t['pl']:+,}")


def main():
    parser = argparse.ArgumentParser(description="OANDA-based performance analysis")
    parser.add_argument("--days", type=int, default=7, help="Number of days to analyze (default: 7)")
    parser.add_argument("--date", type=str, help="Show detail for specific date (YYYY-MM-DD)")
    parser.add_argument("--streak", type=int, default=12, help="Best N-hour window (default: 12)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    token, acct = load_config()
    from_date = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fills = fetch_fills(token, acct, from_date)

    if not fills:
        print("No trades found in the specified period.")
        return

    stats = analyze(fills, streak_hours=args.streak)

    if args.json:
        json.dump(stats, sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        print_report(stats, detail_date=args.date)


if __name__ == "__main__":
    main()
