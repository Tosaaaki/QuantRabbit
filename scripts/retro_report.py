#!/usr/bin/env python3
from __future__ import annotations

"""
scripts.retro_report
~~~~~~~~~~~~~~~~~~~~

Generate a concise retrospective report (daily/weekly) from `logs/trades.db`.

- Summaries: PnL pips, PF, win rate, avg pips
- Breakdowns: by pocket (micro/macro) and by strategy
- Highlights: best/worst strategies (min trades threshold)

Usage examples:
  python scripts/retro_report.py --days 1         # daily retro (prints + writes reports/daily/..)
  python scripts/retro_report.py --days 7         # weekly retro (prints + writes reports/weekly/..)
  python scripts/retro_report.py --days 1 --stdout-only
"""

import argparse
import datetime as dt
import pathlib
import sqlite3
from typing import Tuple

import pandas as pd


DB_PATH = pathlib.Path("logs/trades.db")


def _calc_stats(df: pd.DataFrame) -> Tuple[float, float, float, float, int]:
    if df.empty:
        return 0.0, 0.0, 0.0, 0.0, 0
    profit = df.loc[df.pl_pips > 0, "pl_pips"].sum()
    loss = abs(df.loc[df.pl_pips < 0, "pl_pips"].sum())
    pf = float(profit / loss) if loss else float("inf")
    win_rate = float((df.pl_pips > 0).mean())
    avg = float(df.pl_pips.mean())
    sd = float(df.pl_pips.std()) if len(df) > 1 else 0.0
    sharpe = float(avg / sd) if sd else 0.0
    return pf, win_rate, avg, sharpe, int(len(df))


def build_report(days: int, min_trades_strategy: int = 5) -> str:
    if not DB_PATH.exists():
        return "No trades.db found. Skipping retrospective."

    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trades", con, parse_dates=["close_time", "entry_time"])
    con.close()

    if df.empty:
        return "No trades recorded in trades.db yet."

    if "close_time" in df.columns and pd.api.types.is_datetime64tz_dtype(df["close_time"]):
        df["close_time"] = df["close_time"].dt.tz_convert("UTC").dt.tz_localize(None)
    if "entry_time" in df.columns and pd.api.types.is_datetime64tz_dtype(df["entry_time"]):
        df["entry_time"] = df["entry_time"].dt.tz_convert("UTC").dt.tz_localize(None)

    cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=days)
    df = df[df["close_time"] >= cutoff]
    if df.empty:
        return f"No trades in the last {days} day(s)."

    pf, win_rate, avg, sharpe, n = _calc_stats(df)

    # Pocket breakdown
    pocket_lines = []
    for pocket, sub in df.groupby("pocket"):
        ppf, pwr, pavg, psh, pn = _calc_stats(sub)
        pocket_lines.append(
            f"- {pocket}: PF={ppf:.2f}, Win={pwr:.2%}, Avg={pavg:.2f}p, Sharpe={psh:.2f} ({pn} trades)"
        )

    # Strategy breakdown
    strat_stats = []
    for strat, sub in df.groupby("strategy"):
        if strat is None:
            continue
        spf, swr, savg, ssh, sn = _calc_stats(sub)
        strat_stats.append((strat, sn, spf, swr, savg, ssh))
    # Rank by avg pips
    strat_stats.sort(key=lambda x: x[4], reverse=True)
    best = [s for s in strat_stats if s[1] >= min_trades_strategy][:3]
    worst = [s for s in strat_stats if s[1] >= min_trades_strategy][-3:]

    # Observations and suggestions (rule-based)
    suggestions = []
    # Pocket-level guardrails
    for pocket, sub in df.groupby("pocket"):
        ppf, pwr, pavg, _, pn = _calc_stats(sub)
        if pn >= 5 and ppf < 1.0:
            suggestions.append(
                f"- {pocket}: PF {ppf:.2f} < 1.0. Reduce lot allocation and review entry filters."
            )
    # Strategy-level nudges
    for strat, sn, spf, swr, savg, _ in strat_stats:
        if sn >= min_trades_strategy and savg < 0:
            suggestions.append(
                f"- {strat}: Avg {savg:.2f}p ({sn} trades). Consider disabling in current regimes or tightening stops."
            )
    if not suggestions:
        suggestions.append("- Keep: Current mix performing acceptably. Keep monitoring PF and DD.")

    period = "daily" if days <= 1 else f"last {days} days"
    date_key = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    lines = []
    lines.append(f"# Retro Report – {period} ({date_key})")
    lines.append("")
    lines.append("## Summary")
    lines.append(
        f"- Overall: PF={pf:.2f}, Win={win_rate:.2%}, Avg={avg:.2f}p, Sharpe={sharpe:.2f} ({n} trades)"
    )
    lines.append("## Pockets")
    lines.extend(pocket_lines or ["- (no pocket data)"])
    lines.append("## Best Strategies")
    if best:
        for s in best:
            lines.append(
                f"- {s[0]}: Avg={s[4]:.2f}p, PF={s[2]:.2f}, Win={s[3]:.2%} ({s[1]} trades)"
            )
    else:
        lines.append("- (insufficient data)")
    lines.append("## Worst Strategies")
    if worst:
        for s in worst:
            lines.append(
                f"- {s[0]}: Avg={s[4]:.2f}p, PF={s[2]:.2f}, Win={s[3]:.2%} ({s[1]} trades)"
            )
    else:
        lines.append("- (insufficient data)")
    lines.append("## Suggestions (KPT: Try)")
    lines.extend(suggestions)

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Generate a retrospective report from logs/trades.db")
    ap.add_argument("--days", type=int, default=1, help="Number of days to include (default: 1)")
    ap.add_argument(
        "--stdout-only", action="store_true", help="Print only; do not write report file"
    )
    ap.add_argument(
        "--min-trades-strategy",
        type=int,
        default=5,
        help="Minimum trades to consider for best/worst (default: 5)",
    )
    args = ap.parse_args()

    text = build_report(args.days, args.min_trades_strategy)
    print(text)

    if args.stdout_only:
        return

    # Write under reports/{daily|weekly}/
    base = pathlib.Path("reports")
    sub = "daily" if args.days <= 1 else "weekly"
    out_dir = base / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y-%m-%d")
    out = out_dir / f"{stamp}.md"
    out.write_text(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
