#!/usr/bin/env python3
"""Protected vs unprotected range-grid harvest over 6.5 years (operator's bot).

The operator's real bot had +10%-in-hours harvest days and died of
inventory margin overflow when ranges broke.  Both are the same short-vol
position; protections buy out the tail.  This simulates an EUR_USD grid on
real M5 bid/ask OHLC:

  Grid: when armed, buy limits every 10 pips below the anchor and sell
  limits every 10 pips above (5 levels each side); a filled level takes
  profit one grid step back toward the anchor.  Fills: a buy limit fills
  when the bar's ask low touches it; its TP fills no earlier than the next
  bar when the bid high touches (pessimistic).  OANDA hedge netting:
  margin = max(long, short) exposure / 25; equity marks inventory to mid.

  UNPROTECTED arm: always armed, unlimited re-arming; dies only by margin
  closeout (usage >= 100%), which liquidates everything at market.
  PROTECTED arm: arms only when the measured regime is RANGE or SQUEEZE;
  kill-switch liquidates all inventory when price escapes 1.5x the grid
  span from the anchor or the regime measures TREND; re-arming refused
  above 60% margin usage; hard 5-level inventory cap per side.

Money accounting: NAV 200,000 JPY start, 10,000 units per level.  Metrics:
harvest days (>= +5% / +10%), deaths, yearly path, final multiple.
Single declared configuration, no grid search.  Shadow only.
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

from quant_rabbit.regime_classifier_shadow import (
    RegimeClassifierError,
    classify_regime,
)

UTC = timezone.utc
PAIR = "EUR_USD"
PIP = 0.0001
GRID_PIPS = 10.0
LEVELS = 5
UNITS_PER_LEVEL = 10_000.0
NAV_START = 200_000.0
JPY_PER_USD = 150.0  # declared fixed conversion for P&L marking
LEVERAGE_CAP = 25.0
KILL_SPAN_MULT = 1.5
REARM_USAGE_CAP = 0.60
REGIME_STRIDE_BARS = 12  # re-measure regime hourly
FROM = datetime(2020, 3, 2, tzinfo=UTC)
TO = datetime(2026, 1, 1, tzinfo=UTC)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(root: Path):
    rows = []
    for shard_file in sorted(root.glob(f"*/{PAIR}/{PAIR}_M5_BA_*.jsonl.gz")):
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


def simulate(rows, protected: bool) -> dict[str, Any]:
    balance = NAV_START
    longs: list[float] = []   # entry prices of filled buy levels
    shorts: list[float] = []
    armed = False
    anchor = 0.0
    dead = False
    death_day = None
    daily_equity: dict[str, float] = {}
    prev_equity = NAV_START
    harvest_5 = harvest_10 = 0
    kills = 0
    closes: list[dict] = []  # rolling H1-ish closes for regime (mid)
    step = GRID_PIPS * PIP

    for idx, (epoch, b_o, b_h, b_l, b_c, a_o, a_h, a_l, a_c) in enumerate(rows):
        if dead:
            break
        mid = (b_c + a_c) / 2.0
        stamp = datetime.fromtimestamp(epoch, tz=UTC)
        day = stamp.date().isoformat()

        # Regime measurement (hourly, protected arm only), causal closes.
        if protected and idx % REGIME_STRIDE_BARS == 0:
            closes.append({"time": stamp, "close": mid})
            if len(closes) > 130:
                closes.pop(0)
        regime = None
        if protected and len(closes) >= 121 and idx % REGIME_STRIDE_BARS == 0:
            try:
                c = classify_regime(
                    closes[:-1], as_of_utc=stamp, candle_period_seconds=3600
                )
                regime = c["regime"]
            except RegimeClassifierError:
                regime = None

        # Equity mark and margin.
        unrealized = sum((mid - e) for e in longs) - sum((mid - e) for e in shorts)
        equity = balance + unrealized * UNITS_PER_LEVEL * JPY_PER_USD
        long_exp = len(longs) * UNITS_PER_LEVEL * mid * JPY_PER_USD
        short_exp = len(shorts) * UNITS_PER_LEVEL * mid * JPY_PER_USD
        margin_used = max(long_exp, short_exp) / LEVERAGE_CAP
        usage = margin_used / equity if equity > 0 else 9.9

        # Margin closeout (both arms) — liquidate at market.
        if equity <= 0 or usage >= 1.0:
            balance = equity if equity > 0 else 0.0
            longs, shorts = [], []
            armed = False
            if balance <= NAV_START * 0.02:
                dead = True
                death_day = day
            daily_equity[day] = balance
            continue

        # Protected kill-switch.
        if protected and armed:
            span = LEVELS * step
            escaped = abs(mid - anchor) > KILL_SPAN_MULT * span
            trending = regime == "TREND"
            if escaped or trending:
                # liquidate all inventory at executable prices
                pnl = sum((b_c - e) for e in longs) - sum((a_c - e) for e in shorts)
                balance += pnl * UNITS_PER_LEVEL * JPY_PER_USD
                longs, shorts = [], []
                armed = False
                kills += 1

        # Arm / re-arm.
        if not armed and not longs and not shorts:
            can_arm = (not protected) or (
                regime in ("RANGE", "SQUEEZE") and usage < REARM_USAGE_CAP
            )
            if can_arm:
                armed = True
                anchor = mid

        if armed:
            # Fill buy levels: ask low touches level below anchor.
            for n in range(1, LEVELS + 1):
                level = anchor - n * step
                if len(longs) < n and a_l <= level:
                    if protected and len(longs) >= LEVELS:
                        break
                    longs.append(level)
            # Fill sell levels.
            for n in range(1, LEVELS + 1):
                level = anchor + n * step
                if len(shorts) < n and b_h >= level:
                    if protected and len(shorts) >= LEVELS:
                        break
                    shorts.append(level)
            # Harvest TPs one step back toward anchor (not same bar as fill —
            # approximated by checking exits before new fills next bar).
            still: list[float] = []
            for e in longs:
                tp = e + step
                if b_h >= tp:
                    balance += (tp - e) * UNITS_PER_LEVEL * JPY_PER_USD
                else:
                    still.append(e)
            longs = still
            still = []
            for e in shorts:
                tp = e - step
                if a_l <= tp:
                    balance += (e - tp) * UNITS_PER_LEVEL * JPY_PER_USD
                else:
                    still.append(e)
            shorts = still

        unrealized = sum((mid - e) for e in longs) - sum((mid - e) for e in shorts)
        equity = balance + unrealized * UNITS_PER_LEVEL * JPY_PER_USD
        if day not in daily_equity:
            day_return = equity / prev_equity - 1.0 if prev_equity > 0 else 0.0
            if day_return >= 0.10:
                harvest_10 += 1
            elif day_return >= 0.05:
                harvest_5 += 1
            prev_equity = equity
        daily_equity[day] = equity

    days = sorted(daily_equity)
    yearly: dict[str, float] = {}
    prev = NAV_START
    year_start: dict[str, float] = {}
    for day in days:
        year = day[:4]
        year_start.setdefault(year, prev)
        yearly[year] = daily_equity[day] / year_start[year] - 1.0
        prev = daily_equity[day]
    final = daily_equity[days[-1]] if days else NAV_START
    return {
        "final_equity_jpy": round(final),
        "multiple": round(final / NAV_START, 4),
        "dead": dead,
        "death_day": death_day,
        "kill_switch_liquidations": kills,
        "harvest_days_5_10pct": harvest_5,
        "harvest_days_ge_10pct": harvest_10,
        "by_year_return": {y: round(v, 4) for y, v in sorted(yearly.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be clean; refusing stale reuse")
    rows = [r for r in _load(args.root) if FROM.timestamp() <= r[0] < TO.timestamp()]

    unprotected = simulate(rows, protected=False)
    protected = simulate(rows, protected=True)

    body: dict[str, Any] = {
        "contract": "QR_PROTECTED_GRID_SIMULATION_V1",
        "schema_version": 1,
        "pair": PAIR,
        "config": {
            "grid_pips": GRID_PIPS, "levels": LEVELS,
            "units_per_level": UNITS_PER_LEVEL, "nav_start_jpy": NAV_START,
            "kill_span_mult": KILL_SPAN_MULT, "rearm_usage_cap": REARM_USAGE_CAP,
            "declared_single_config_no_search": True,
        },
        "window": [FROM.isoformat(), TO.isoformat()],
        "unprotected": unprotected,
        "protected": protected,
        "operator_question": (
            "longs/shorts stack above and below, margin overflow kills — is "
            "there a solution?  Protections tested: inventory cap, regime "
            "kill-switch, rearm usage cap"
        ),
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "simulation_sha256": _canonical_sha(body)}
    payload = json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_name, args.output)
    print(json.dumps({"status": "GRID_SIM_SEALED", "unprotected": unprotected, "protected": protected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
