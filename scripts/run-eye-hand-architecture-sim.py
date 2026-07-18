#!/usr/bin/env python3
"""Eye-hand-mechanic architecture simulation: how much is a good eye worth?

The adopted architecture: the EYE (layer-2 discretionary read) declares the
day type each day, the HANDS (protected mechanical workers) harvest inside
the cage, the MECHANIC maintains the arsenal.  Historical simulation of a
discretionary eye is impossible without hindsight, so the eye is bracketed
between three selectors over the same two protected workers:

  ALWAYS_ON     no eye: every worker armed every day (the old death mode,
                minus the deaths — protections stay on)
  MEASURED_EYE  causal proxy: today's class = yesterday's realized class
                (efficiency-ratio persistence, computable live)
  ORACLE_EYE    perfect weather forecast: today's class from today's own
                realized path (NOT causal — an upper bound only; the gap
                to MEASURED is the prize the discretionary read plays for)

Workers (both protected, both flat at day end — no overnight inventory):
  W_BURST  USD_JPY trend harvester: 24h trend side, 3-bar M5 break entry,
           TP +3 pips, 60-min time-stop, trend-flip flush, max 3 concurrent.
           Armed on TREND days.
  W_GRID   EUR_USD range harvester: 10-pip grid x 5 levels both sides
           anchored at arming, TP one step, kill-switch at 1.5x span
           escape (disarms for the day).  Armed on RANGE days.

Day class per pair from hourly closes: efficiency = |last-first| /
sum(|hourly steps|); TREND >= 0.35, RANGE <= 0.18, else UNCLEAR (nothing
armed for that pair).  Money accounting: one shared 200k JPY account,
NAV-proportional sizing (burst 4.3x per position, grid 0.9x per level),
25x margin, combined usage cap 92% blocks new entries.  Single declared
configuration, no search.  Shadow only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc
NAV_START = 200_000.0
LEVERAGE = 25.0
USAGE_CAP = 0.92
JPY_PER_USD = 150.0  # declared conversion for EUR_USD P&L marking

BURST_PAIR = "USD_JPY"
BURST_PIP = 0.01
BURST_TREND_S = 24 * 3600
BURST_BREAK_BARS = 3
BURST_TP_PIPS = 3.0
BURST_TIME_STOP_S = 3600
BURST_MAX_CONCURRENT = 3
BURST_PER_POS_LEV = 4.3

GRID_PAIR = "EUR_USD"
GRID_PIP = 0.0001
GRID_STEP_PIPS = 10.0
GRID_LEVELS = 5
GRID_PER_LEVEL_LEV = 0.9
GRID_KILL_SPAN_MULT = 1.5

TREND_EFF_MIN = 0.35
RANGE_EFF_MAX = 0.18
MIN_HOURS = 6
REPLAY_FROM = "2020-03-03"
SPLITS = {
    "TRAIN_2020_2023": ("2020-03-03", "2024-01-01"),
    "VAL_2024_2025": ("2024-01-01", "2026-01-01"),
}


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(root: Path, pair: str):
    rows = []
    for shard_file in sorted(root.glob(f"*/{pair}/{pair}_M5_BA_*.jsonl.gz")):
        with gzip.open(shard_file, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                epoch = int(datetime.fromisoformat(row["time"][:19] + "+00:00").timestamp())
                rows.append(
                    (
                        epoch,
                        float(row["bid"]["o"]), float(row["bid"]["h"]),
                        float(row["bid"]["l"]), float(row["bid"]["c"]),
                        float(row["ask"]["o"]), float(row["ask"]["h"]),
                        float(row["ask"]["l"]), float(row["ask"]["c"]),
                    )
                )
    rows.sort()
    return rows


def _day_of(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).date().isoformat()


def _day_classes(rows) -> dict[str, str]:
    """Realized day class per calendar day from hourly closes."""

    hourly: dict[str, list[float]] = {}
    last_hour: dict[str, int] = {}
    for r in rows:
        day = _day_of(r[0])
        hour = r[0] // 3600
        mid = (r[4] + r[8]) / 2.0
        bucket = hourly.setdefault(day, [])
        if last_hour.get(day) != hour:
            bucket.append(mid)
            last_hour[day] = hour
        else:
            bucket[-1] = mid
    classes: dict[str, str] = {}
    for day, closes in hourly.items():
        if len(closes) < MIN_HOURS:
            classes[day] = "UNCLEAR"
            continue
        net = abs(closes[-1] - closes[0])
        path = sum(abs(b - a) for a, b in zip(closes, closes[1:]))
        if path <= 0:
            classes[day] = "UNCLEAR"
            continue
        eff = net / path
        if eff >= TREND_EFF_MIN:
            classes[day] = "TREND"
        elif eff <= RANGE_EFF_MAX:
            classes[day] = "RANGE"
        else:
            classes[day] = "UNCLEAR"
    return classes


def simulate(burst_rows, burst_mid_close, grid_rows, eye: str,
             burst_truth: dict[str, str], grid_truth: dict[str, str]) -> dict[str, Any]:
    balance = NAV_START
    daily_pl: dict[str, float] = {}
    armed_days = {"W_BURST": 0, "W_GRID": 0}
    grid_kills = 0
    blocked_entries = 0

    def book(day: str, pl: float) -> None:
        daily_pl[day] = daily_pl.get(day, 0.0) + pl

    burst_days = sorted({_day_of(r[0]) for r in burst_rows})
    grid_days = sorted({_day_of(r[0]) for r in grid_rows})
    prev_class: dict[str, str] = {}

    def eye_class(worker: str, day: str, truth: dict[str, str], days: list[str]) -> str:
        if eye == "ALWAYS_ON":
            return "TREND" if worker == "W_BURST" else "RANGE"
        if eye == "ORACLE_EYE":
            return truth.get(day, "UNCLEAR")
        # MEASURED_EYE: previous traded day's realized class
        i = days.index(day)
        return truth.get(days[i - 1], "UNCLEAR") if i > 0 else "UNCLEAR"

    burst_by_day: dict[str, list] = {}
    for r in burst_rows:
        burst_by_day.setdefault(_day_of(r[0]), []).append(r)
    grid_by_day: dict[str, list] = {}
    for r in grid_rows:
        grid_by_day.setdefault(_day_of(r[0]), []).append(r)

    all_days = sorted(set(burst_days) | set(grid_days))
    dead_day = None
    for day in all_days:
        if day < REPLAY_FROM:
            continue
        if balance <= NAV_START * 0.02:
            dead_day = dead_day or day
            break
        burst_armed = day in burst_by_day and eye_class(
            "W_BURST", day, burst_truth, burst_days) == "TREND"
        grid_armed = day in grid_by_day and eye_class(
            "W_GRID", day, grid_truth, grid_days) == "RANGE"
        armed_days["W_BURST"] += int(burst_armed)
        armed_days["W_GRID"] += int(grid_armed)
        if not burst_armed and not grid_armed:
            continue

        # ---- burst state
        b_pos: list[tuple[str, float, int, float, float]] = []
        prev_trend = None
        b_rows = burst_by_day.get(day, [])
        b_index = {r[0]: i for i, r in enumerate(b_rows)}
        # ---- grid state
        g_rows = grid_by_day.get(day, [])
        g_longs: list[tuple[float, float]] = []
        g_shorts: list[tuple[float, float]] = []
        g_anchor = None
        g_live = grid_armed
        step = GRID_STEP_PIPS * GRID_PIP
        if grid_armed and g_rows:
            g_anchor = (g_rows[0][4] + g_rows[0][8]) / 2.0

        # Merge both pairs' bars into one intraday timeline.
        events = [("B", r) for r in b_rows if burst_armed] + [
            ("G", r) for r in g_rows if grid_armed]
        events.sort(key=lambda e: e[1][0])

        for kind, r in events:
            epoch, b_o, b_h, b_l, b_c, a_o, a_h, a_l, a_c = r
            if kind == "B":
                mid = (b_c + a_c) / 2.0
                still = []
                for side, entry, e_epoch, tp, units in b_pos:
                    if e_epoch >= epoch:
                        still.append((side, entry, e_epoch, tp, units))
                        continue
                    tp_hit = (b_h >= tp) if side == "LONG" else (a_l <= tp)
                    timed_out = epoch >= e_epoch + BURST_TIME_STOP_S
                    if tp_hit:
                        pl = ((tp - entry) if side == "LONG" else (entry - tp)) * units
                        balance += pl
                        book(day, pl)
                    elif timed_out:
                        px = b_o if side == "LONG" else a_o
                        pl = ((px - entry) if side == "LONG" else (entry - px)) * units
                        balance += pl
                        book(day, pl)
                    else:
                        still.append((side, entry, e_epoch, tp, units))
                b_pos = still

                close_epoch = epoch + 300
                now = burst_mid_close.get(close_epoch)
                past = burst_mid_close.get(close_epoch - BURST_TREND_S)
                if now is None or past is None:
                    continue
                trend = "LONG" if now > past else "SHORT"
                if prev_trend is not None and trend != prev_trend and b_pos:
                    for s, e, _, _, u in b_pos:
                        px = b_c if s == "LONG" else a_c
                        pl = ((px - e) if s == "LONG" else (e - px)) * u
                        balance += pl
                        book(day, pl)
                    b_pos = []
                prev_trend = trend
                if len(b_pos) >= BURST_MAX_CONCURRENT:
                    continue
                i = b_index[epoch]
                if i + 1 >= len(b_rows) or b_rows[i + 1][0] != epoch + 300:
                    continue
                if i < BURST_BREAK_BARS:
                    continue
                window = b_rows[i - BURST_BREAK_BARS: i]
                mids_h = [(w[2] + w[6]) / 2.0 for w in window]
                mids_l = [(w[3] + w[7]) / 2.0 for w in window]
                if not ((trend == "LONG" and mid > max(mids_h))
                        or (trend == "SHORT" and mid < min(mids_l))):
                    continue
                # combined margin usage check
                b_units = sum(u for *_, u in b_pos)
                g_long_u = sum(u for _, u in g_longs)
                g_short_u = sum(u for _, u in g_shorts)
                g_mid = g_anchor or 1.0
                margin = (b_units * mid / LEVERAGE
                          + max(g_long_u, g_short_u) * g_mid * JPY_PER_USD / LEVERAGE)
                if margin / max(balance, 1e-9) >= USAGE_CAP:
                    blocked_entries += 1
                    continue
                nxt = b_rows[i + 1]
                entry = nxt[5] if trend == "LONG" else nxt[1]
                tp = entry + BURST_TP_PIPS * BURST_PIP if trend == "LONG" else entry - BURST_TP_PIPS * BURST_PIP
                units = max(balance, 0.0) * BURST_PER_POS_LEV / entry
                b_pos.append((trend, entry, nxt[0], tp, units))
            else:
                if not g_live or g_anchor is None:
                    continue
                mid = (b_c + a_c) / 2.0
                units_per_level = max(balance, 0.0) * GRID_PER_LEVEL_LEV / max(mid * JPY_PER_USD, 1e-9)
                span = GRID_LEVELS * step
                if abs(mid - g_anchor) > GRID_KILL_SPAN_MULT * span:
                    pnl = sum((b_c - e) * u for e, u in g_longs) - sum((a_c - e) * u for e, u in g_shorts)
                    balance += pnl * JPY_PER_USD
                    book(day, pnl * JPY_PER_USD)
                    g_longs, g_shorts = [], []
                    g_live = False
                    grid_kills += 1
                    continue
                for n in range(1, GRID_LEVELS + 1):
                    level = g_anchor - n * step
                    if len(g_longs) < n and a_l <= level:
                        g_longs.append((level, units_per_level))
                for n in range(1, GRID_LEVELS + 1):
                    level = g_anchor + n * step
                    if len(g_shorts) < n and b_h >= level:
                        g_shorts.append((level, units_per_level))
                still_l = []
                for e, u in g_longs:
                    if b_h >= e + step:
                        pl = step * u * JPY_PER_USD
                        balance += pl
                        book(day, pl)
                    else:
                        still_l.append((e, u))
                g_longs = still_l
                still_s = []
                for e, u in g_shorts:
                    if a_l <= e - step:
                        pl = step * u * JPY_PER_USD
                        balance += pl
                        book(day, pl)
                    else:
                        still_s.append((e, u))
                g_shorts = still_s

        # Flat at day end (no overnight inventory) at last executable closes.
        if b_pos and b_rows:
            last = b_rows[-1]
            for s, e, _, _, u in b_pos:
                px = last[4] if s == "LONG" else last[8]
                pl = ((px - e) if s == "LONG" else (e - px)) * u
                balance += pl
                book(day, pl)
        if (g_longs or g_shorts) and g_rows:
            last = g_rows[-1]
            pnl = sum((last[4] - e) * u for e, u in g_longs) - sum((last[8] - e) * u for e, u in g_shorts)
            balance += pnl * JPY_PER_USD
            book(day, pnl * JPY_PER_USD)

    def split(from_s: str, to_s: str) -> dict[str, Any]:
        days = {d: v for d, v in daily_pl.items() if from_s <= d < to_s}
        equity = NAV_START
        compound = 1.0
        worst = best = 0.0
        negd = h5 = h10 = 0
        for d in sorted(days):
            r = days[d] / max(equity, 1e-9)
            r = max(r, -1.0)  # a day cannot lose more than the account
            equity = max(equity + days[d], 0.0)
            compound = max(compound * (1.0 + r), 0.0)
            worst = min(worst, r)
            best = max(best, r)
            negd += int(days[d] < 0)
            if r >= 0.10:
                h10 += 1
            elif r >= 0.05:
                h5 += 1
        months = max(1e-9, len(days) / 21.7)
        return {
            "net_jpy": round(sum(days.values())),
            "nav_multiple": round(compound, 4),
            "monthly_multiple": round(compound ** (1.0 / months), 4) if days and compound > 0 else 0.0,
            "active_days": len(days),
            "negative_days": negd,
            "worst_day_nav": round(worst, 4),
            "best_day_nav": round(best, 4),
            "harvest_days_5_10pct": h5,
            "harvest_days_ge_10pct": h10,
        }

    return {
        "final_balance_jpy": round(balance),
        "dead_day": dead_day,
        "armed_days": armed_days,
        "grid_kill_switches": grid_kills,
        "blocked_entries_at_usage_cap": blocked_entries,
        "splits": {name: split(f, t) for name, (f, t) in SPLITS.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be clean; refusing stale reuse")

    burst_rows = _load(args.root, BURST_PAIR)
    grid_rows = _load(args.root, GRID_PAIR)
    burst_mid_close = {r[0] + 300: (r[4] + r[8]) / 2.0 for r in burst_rows}
    burst_truth = _day_classes(burst_rows)
    grid_truth = _day_classes(grid_rows)

    results = {
        eye: simulate(burst_rows, burst_mid_close, grid_rows, eye, burst_truth, grid_truth)
        for eye in ("ALWAYS_ON", "MEASURED_EYE", "ORACLE_EYE")
    }

    body: dict[str, Any] = {
        "contract": "QR_EYE_HAND_ARCHITECTURE_SIM_V1",
        "schema_version": 1,
        "architecture": "eye (day-type selector) x hands (protected workers) x cage",
        "workers": {
            "W_BURST": f"{BURST_PAIR} trend harvester, armed on TREND days",
            "W_GRID": f"{GRID_PAIR} range harvester, armed on RANGE days",
        },
        "day_class_rule": {
            "efficiency": "|last-first| / sum|hourly steps| on the day's hourly closes",
            "trend_min": TREND_EFF_MIN, "range_max": RANGE_EFF_MAX,
        },
        "eyes": {
            "ALWAYS_ON": "no selector (both armed daily)",
            "MEASURED_EYE": "causal: yesterday's realized class",
            "ORACLE_EYE": "NON-CAUSAL upper bound: today's own realized class",
        },
        "accounting": {
            "nav_start_jpy": NAV_START, "leverage": LEVERAGE, "usage_cap": USAGE_CAP,
            "sizing": "NAV_PROPORTIONAL", "flat_at_day_end": True,
            "jpy_per_usd_marking": JPY_PER_USD,
        },
        "single_declared_config_no_search": True,
        "results": results,
        "test_2026_untouched": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "research_sha256": _canonical_sha(body)}
    payload = json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_name, args.output)
    print(json.dumps({"status": "EYE_HAND_SIM_SEALED",
                      "results": {k: v["splits"] for k, v in results.items()}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
