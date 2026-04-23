#!/usr/bin/env python3
"""Layer-aware pre-close reflection with post-close regret evidence.

Same philosophy as pretrade_check: this is not a mechanical blocker.
It forces the trader to name which thesis layer died before flattening
inventory. Most premature losses came from trigger / vehicle wobble
being treated as full market-thesis collapse.

Usage:
    python3 tools/preclose_check.py EUR_USD SHORT 12000 -612
    python3 tools/preclose_check.py EUR_USD SHORT 12000 -612 --reason m1_pulse_flip
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "live_trade_log.txt"
STATE_PATH = ROOT / "collab_trade" / "state.md"
JST = timezone(timedelta(hours=9))
UTC = timezone.utc


def _parse_log_timestamp(line: str) -> datetime | None:
    match = re.match(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\]", line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _close_family(close_reason: str | None) -> str:
    text = str(close_reason or "").upper()
    if text == "MARKET_ORDER_TRADE_CLOSE":
        return "manual"
    if text == "STOP_LOSS_ORDER":
        return "stop"
    return text.lower() or "unknown"


def _classify_layer(reason: str | None) -> str:
    text = str(reason or "").lower()
    if not text:
        return "unknown"
    if any(token in text for token in ("stale", "zombie", "recycle", "aged", "wish_distance")):
        return "aging"
    if any(token in text for token in ("spread", "slippage", "friction", "too wide", "market fill")):
        return "vehicle"
    if any(token in text for token in ("acceptance_above_entry", "accept", "body break", "structure", "shelf broke")):
        return "structure"
    if any(
        token in text
        for token in (
            "m1_pulse_flip",
            "no_first_confirmation",
            "no_reclaim",
            "shelf_fail",
            "no_rebuy",
            "failed_break",
            "failed_floor",
            "retest",
            "reclaim",
        )
    ):
        return "trigger"
    if any(token in text for token in ("usd bid", "risk-off", "market reversed", "macro flip")):
        return "market"
    return "unknown"


def _summarize_regret_rows(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    recovered = sum(1 for row in rows if row.get("recovered"))
    lags = []
    for row in rows:
        recovered_at = row.get("recovered_at")
        if not row.get("recovered") or not recovered_at:
            continue
        try:
            close_time = datetime.strptime(row["close_time_jst"], "%Y-%m-%d %H:%M JST")
            recovered_time = datetime.strptime(recovered_at, "%Y-%m-%d %H:%M JST")
        except Exception:
            continue
        lags.append((recovered_time - close_time).total_seconds() / 60.0)
    avg_loss = sum(float(row.get("loss_pips") or 0.0) for row in rows) / len(rows)
    avg_fav = sum(float(row.get("fav_pips") or 0.0) for row in rows) / len(rows)
    return {
        "count": len(rows),
        "recovered": recovered,
        "recovery_rate": recovered / len(rows) * 100.0,
        "avg_loss_pips": avg_loss,
        "avg_fav_pips": avg_fav,
        "median_lag_min": statistics.median(lags) if lags else None,
    }


def _load_recent_regret_payload(days: int = 7) -> dict:
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        from post_close_regret import build_regret_payload  # type: ignore
    except Exception:
        return {}

    session_date_from = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        return build_regret_payload(session_date_from=session_date_from, hours=6)
    except Exception:
        return {}


def _regret_views(pair: str, close_family: str) -> tuple[dict | None, dict | None, dict | None]:
    payload = _load_recent_regret_payload()
    rows = list(payload.get("results", []))
    pair_rows = [row for row in rows if row.get("pair") == pair]
    family_rows = [row for row in rows if _close_family(row.get("close_reason")) == close_family]
    pair_family_rows = [row for row in pair_rows if _close_family(row.get("close_reason")) == close_family]
    return (
        _summarize_regret_rows(pair_rows),
        _summarize_regret_rows(family_rows),
        _summarize_regret_rows(pair_family_rows),
    )


def get_consecutive_losses() -> tuple[int, float]:
    if not LOG_PATH.exists():
        return 0, 0.0
    closes: list[float] = []
    for line in LOG_PATH.read_text().splitlines():
        if "CLOSE" not in line:
            continue
        match = re.search(r"PL=([+-]?[\d,.]+)", line)
        if match:
            closes.append(float(match.group(1).replace(",", "")))
    streak = 0
    streak_sum = 0.0
    for pl in reversed(closes):
        if pl < 0:
            streak += 1
            streak_sum += pl
        else:
            break
    return streak, streak_sum


def get_pair_pl_today(pair: str) -> float:
    total = 0.0
    if not LOG_PATH.exists():
        return total
    today_jst = datetime.now(JST).date()
    for line in LOG_PATH.read_text().splitlines():
        if "CLOSE" not in line or pair not in line:
            continue
        ts = _parse_log_timestamp(line)
        if ts is None or ts.astimezone(JST).date() != today_jst:
            continue
        match = re.search(r"PL=([+-]?[\d,.]+)", line)
        if match:
            total += float(match.group(1).replace(",", ""))
    return total


def get_thesis_from_state(pair: str) -> str | None:
    if not STATE_PATH.exists():
        return None
    lines = STATE_PATH.read_text().splitlines()
    section: list[str] = []
    in_section = False
    for line in lines:
        if pair in line and ("###" in line or "##" in line):
            in_section = True
            section.append(line)
            continue
        if in_section:
            if line.startswith("### ") or line.startswith("---"):
                break
            section.append(line)
    return "\n".join(section) if section else None


def get_today_stats() -> tuple[int, int, float, float]:
    wins = 0
    losses = 0
    win_sum = 0.0
    loss_sum = 0.0
    if not LOG_PATH.exists():
        return wins, losses, win_sum, loss_sum
    today_jst = datetime.now(JST).date()
    for line in LOG_PATH.read_text().splitlines():
        if "CLOSE" not in line:
            continue
        ts = _parse_log_timestamp(line)
        if ts is None or ts.astimezone(JST).date() != today_jst:
            continue
        match = re.search(r"PL=([+-]?[\d,.]+)", line)
        if not match:
            continue
        pl = float(match.group(1).replace(",", ""))
        if pl > 0:
            wins += 1
            win_sum += pl
        else:
            losses += 1
            loss_sum += pl
    return wins, losses, win_sum, loss_sum


def _print_regret_line(label: str, stats: dict | None) -> None:
    if not stats:
        print(f"  {label}: no recent sample")
        return
    lag = stats["median_lag_min"]
    lag_text = f" | median return {lag:.0f}min" if lag is not None else ""
    print(
        f"  {label}: {stats['recovered']}/{stats['count']} recovered in 6h "
        f"({stats['recovery_rate']:.1f}%) | avg loss {stats['avg_loss_pips']:.1f}pip "
        f"-> avg later favorable {stats['avg_fav_pips']:.1f}pip{lag_text}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair")
    parser.add_argument("side")
    parser.add_argument("units", type=int)
    parser.add_argument("unrealized_pl", type=float)
    parser.add_argument("--reason", default="", help="Planned close reason, if already known.")
    parser.add_argument(
        "--close-family",
        choices=("manual", "stop"),
        default="manual",
        help="Planned close family. preclose_check is usually called before a discretionary close, so default=manual.",
    )
    args = parser.parse_args()

    pair = args.pair
    side = args.side
    units = args.units
    upl = args.unrealized_pl
    planned_layer = _classify_layer(args.reason)

    print(f"=== PRECLOSE DISCIPLINE: {pair} {side} {units}u (unrealized P&L: {upl:+,.0f} JPY) ===")
    print()

    thesis = get_thesis_from_state(pair)
    if thesis:
        print("[CURRENT THESIS SNIPPET]")
        for line in thesis.splitlines():
            compact = line.strip()
            if not compact:
                continue
            if any(token in compact.lower() for token in ("thesis", "conviction", "dead if", "upgrade", "best expression")):
                print(f"  {compact}")
        print()

    pair_regret, family_regret, pair_family_regret = _regret_views(pair, args.close_family)
    print("[POST-CLOSE REGRET EVIDENCE]")
    _print_regret_line(f"{pair} all losing closes", pair_regret)
    _print_regret_line(f"{args.close_family} closes (market-wide)", family_regret)
    _print_regret_line(f"{pair} + {args.close_family} closes", pair_family_regret)
    print()

    streak, streak_sum = get_consecutive_losses()
    pair_pl = get_pair_pl_today(pair)
    wins, losses, win_sum, loss_sum = get_today_stats()
    avg_win = (win_sum / wins) if wins else 0.0
    avg_loss = (loss_sum / losses) if losses else 0.0

    print("[CONTEXT]")
    print(f"  Last close streak: {streak} losses ({streak_sum:+,.0f} JPY)" if streak else "  Last close streak: no active loss streak")
    print(f"  {pair} realized today: {pair_pl:+,.0f} JPY")
    if wins or losses:
        print(
            f"  Today so far: {wins}W/{losses}L | avg win {avg_win:+,.0f} JPY | avg loss {avg_loss:+,.0f} JPY"
        )
    if -100 < upl < 0:
        print(f"  Noise check: {upl:+,.0f} JPY is still small. Be sure this is not a wobble masquerading as collapse.")
    print()

    print("[THESIS-LAYER TEST BEFORE FULL CLOSE]")
    print("  1. Market layer dead? [YES/NO] What macro / cross-currency contradiction killed it?")
    print("  2. Structure layer dead? [YES/NO] Which defended level accepted through, not just wicked through?")
    print("  3. Trigger layer dead? [YES/NO] Was this only the first wobble / failed print?")
    print("  4. Vehicle layer dead? [YES/NO] Is the problem spread / slippage / a bad paid entry vehicle?")
    print("  5. Aging layer dead? [YES/NO] Am I holding or cutting a stale recycled idea?")
    if args.reason:
        print(f"  Planned close reason: {args.reason} -> maps to dead layer [{planned_layer}]")
    else:
        print("  Planned close reason: [write it first] -> dead layer = [market / structure / trigger / vehicle / aging]")
    print()

    print("[DISCIPLINE BIAS]")
    if upl < 0:
        if pair_family_regret and pair_family_regret["recovery_rate"] >= 70.0:
            print(
                "  This exact pair/close family usually trades back. Full close needs market/structure/aging death, not just trigger wobble."
            )
        elif family_regret and family_regret["recovery_rate"] >= 70.0:
            print(
                "  This close family usually trades back. Treat trigger/vehicle damage as HOLD or HALF-TP territory unless structure really failed."
            )
        else:
            print("  No strong regret edge either way. You still must name the dead layer explicitly.")

        if planned_layer in {"trigger", "vehicle", "unknown"}:
            print("  Default if only trigger/vehicle died: do NOT full-close by reflex. Prefer HOLD / HALF / reload after a new print.")
        if planned_layer in {"market", "structure", "aging"}:
            print("  A full close can be honest here, but only if you can write the exact contradiction in one sentence.")
    else:
        print("  Profit-side reminder: if the trade already paid, justify why holding is better than banking inventory.")
    print()

    print("[ANSWER THESE BEFORE CLOSING]")
    print("  A. Dead layer = ___")
    print("  B. Surviving layers = ___")
    print("  C. If dead layer is trigger/vehicle only, why is full close better than hold / half / reload?")
    print("  D. If dead layer is market/structure/aging, what exact contradiction changed?")
    print()
    print("If you can answer these coherently, close — and record the reason in live_trade_log.")
    print("======================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
