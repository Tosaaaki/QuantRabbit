#!/usr/bin/env python3
"""MomentumBurst replica: RAW (as lived) vs PROTECTED (operator's inventory fix).

The live MomentumBurst micro worker printed +24,542 JPY in 53 straight
USD_JPY wins on 2025-12-09 (trend-side burst entries, ~3-pip take-profits)
and then gave 59% back overnight because the last three entries near the
top had NO time-stop and NO trend-flip exit — the stranded-inventory
death mode in miniature.  The operator's thesis: inventory management
turns this profile into a keeper.  Both arms below share the SAME declared
entry signal; only the inventory/exit protections differ.

  Signal (declared, single config, no search): USD_JPY, closed M5 bars;
  24h mid trend filter picks the side; trigger = M5 mid close breaks the
  prior 3 bars' extreme in trend direction; enter next M5 open (ask/bid);
  take-profit +3.0 pips resolved pessimistically (no earlier than the bar
  after entry; long TP at bid high, short at ask low).

  RAW arm (as the live worker behaved): TP-only exits, unlimited stacking,
  positions held until TP whenever that takes days.  Margin closeout at
  100% usage liquidates everything (the death the operator described).
  PROTECTED arm: max 3 concurrent positions, 60-minute time-stop at
  market, and a trend-flip flush (24h trend sign change closes all).

Money accounting: 200k JPY NAV, 15,000 units per position (live sizing),
25x margin, weekend gate on entries.  TRAIN 2020-2023 / VAL 2024-2025,
2026 untouched.  Shadow only.
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

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.regime_classifier_shadow import (
    RegimeClassifierError,
    classify_regime,
)

UTC = timezone.utc
PAIR = "USD_JPY"
PIP = 0.01
TREND_LOOKBACK_S = 24 * 3600
BREAK_BARS = 3
TP_PIPS = 3.0
# Live sizing was ~15k units at ~543k NAV and ~156 price = ~4.3x exposure per
# position.  Sizing is NAV-proportional (per the NAV-percent standard), not a
# fixed unit count, so the replica is feasible at any equity level.
PER_POSITION_LEVERAGE = 4.3
NAV_START = 200_000.0
LEVERAGE = 25.0
REPLAY_FROM = "2020-03-02"
TIME_STOP_S = 60 * 60
MAX_CONCURRENT = 3
WEEKEND_MARGIN_MIN = 5
SPLITS = {
    "TRAIN_2020_2023": ("2020-03-02", "2024-01-01"),
    "VAL_2024_2025": ("2024-01-01", "2026-01-01"),
}


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


def simulate(rows, mid_close, protected: bool, regime_gated: bool = False) -> dict[str, Any]:
    balance = NAV_START
    # open positions: (side, entry_price, entry_epoch, tp_price, units)
    open_pos: list[tuple[str, float, int, float, float]] = []
    dead_day = None
    closeouts = 0
    daily_pl: dict[str, float] = {}
    trades = wins = 0

    def book(day: str, pl_jpy: float, won: bool | None) -> None:
        nonlocal trades, wins
        daily_pl[day] = daily_pl.get(day, 0.0) + pl_jpy
        if won is not None:
            trades += 1
            wins += int(won)

    prev_trend = None
    regime = None
    h1_closes: list[dict] = []
    for k in range(BREAK_BARS + 1, len(rows)):
        epoch, b_o, b_h, b_l, b_c, a_o, a_h, a_l, a_c = rows[k]
        stamp = datetime.fromtimestamp(epoch, tz=UTC)
        day = stamp.date().isoformat()
        mid = (b_c + a_c) / 2.0

        # Hourly causal regime measurement (gated arm only).
        if regime_gated and k % 12 == 0:
            h1_closes.append({"time": stamp, "close": mid})
            if len(h1_closes) > 130:
                h1_closes.pop(0)
            if len(h1_closes) >= 121:
                try:
                    verdict = classify_regime(
                        h1_closes[:-1], as_of_utc=stamp, candle_period_seconds=3600
                    )
                    regime = verdict["regime"]
                except RegimeClassifierError:
                    regime = None

        # Resolve exits on this bar (positions entered on earlier bars only).
        still: list[tuple[str, float, int, float, float]] = []
        for side, entry, entry_epoch, tp, units in open_pos:
            if entry_epoch >= epoch:
                still.append((side, entry, entry_epoch, tp, units))
                continue
            tp_hit = (b_h >= tp) if side == "LONG" else (a_l <= tp)
            timed_out = protected and epoch >= entry_epoch + TIME_STOP_S
            if tp_hit:
                pl = (tp - entry) * units if side == "LONG" else (entry - tp) * units
                balance += pl
                book(day, pl, True)
            elif timed_out:
                exit_price = b_o if side == "LONG" else a_o
                pl = (exit_price - entry) * units if side == "LONG" else (entry - exit_price) * units
                balance += pl
                book(day, pl, pl > 0)
            else:
                still.append((side, entry, entry_epoch, tp, units))
        open_pos = still

        # Mark equity / margin (hedge netting: max side).
        unreal = sum(
            (mid - e) * u if s == "LONG" else (e - mid) * u
            for s, e, _, _, u in open_pos
        )
        equity = balance + unreal
        long_u = sum(u for s, *_, u in open_pos if s == "LONG")
        short_u = sum(u for s, *_, u in open_pos if s == "SHORT")
        margin_used = max(long_u, short_u) * mid / LEVERAGE
        if equity <= 0 or (margin_used > 0 and margin_used / max(equity, 1e-9) >= 1.0):
            # margin closeout: everything at executable prices
            for s, e, _, _, u in open_pos:
                exit_price = b_c if s == "LONG" else a_c
                pl = (exit_price - e) * u if s == "LONG" else (e - exit_price) * u
                balance += pl
                book(day, pl, False)
            open_pos = []
            closeouts += 1
            if balance <= NAV_START * 0.02 and dead_day is None:
                dead_day = day
                break
            continue

        # Trend measurement at this bar's close.
        close_epoch = epoch + 300
        now = mid_close.get(close_epoch)
        past = mid_close.get(close_epoch - TREND_LOOKBACK_S)
        if now is None or past is None:
            continue
        trend = "LONG" if now > past else "SHORT"

        # Protected: trend flip flushes all inventory at market.
        if protected and prev_trend is not None and trend != prev_trend and open_pos:
            for s, e, _, _, u in open_pos:
                exit_price = b_c if s == "LONG" else a_c
                pl = (exit_price - e) * u if s == "LONG" else (e - exit_price) * u
                balance += pl
                book(day, pl, pl > 0)
            open_pos = []
        prev_trend = trend

        # Entry trigger on this closed bar -> enter next bar open.
        if k + 1 >= len(rows) or rows[k + 1][0] != epoch + 300:
            continue
        status = compute_market_status(stamp)
        if not status.is_fx_open or status.minutes_to_next_close is None or (
            status.minutes_to_next_close <= WEEKEND_MARGIN_MIN + 60
        ):
            continue
        if protected and len(open_pos) >= MAX_CONCURRENT:
            continue
        if regime_gated and regime != "TREND":
            continue
        window = rows[k - BREAK_BARS: k]
        mids_h = [ (r[2] + r[6]) / 2.0 for r in window ]
        mids_l = [ (r[3] + r[7]) / 2.0 for r in window ]
        triggered = (
            (trend == "LONG" and mid > max(mids_h))
            or (trend == "SHORT" and mid < min(mids_l))
        )
        if not triggered:
            continue
        nxt = rows[k + 1]
        if trend == "LONG":
            entry = nxt[5]  # ask open
            tp = entry + TP_PIPS * PIP
        else:
            entry = nxt[1]  # bid open
            tp = entry - TP_PIPS * PIP
        units = max(equity, 0.0) * PER_POSITION_LEVERAGE / entry
        if units <= 0:
            continue
        open_pos.append((trend, entry, nxt[0], tp, units))

    # Liquidate whatever is left at the last bar (accounting closure).
    if open_pos:
        _, b_c_last, a_c_last = rows[-1][0], rows[-1][4], rows[-1][8]
        day = datetime.fromtimestamp(rows[-1][0], tz=UTC).date().isoformat()
        for s, e, _, _, u in open_pos:
            exit_price = b_c_last if s == "LONG" else a_c_last
            pl = (exit_price - e) * u if s == "LONG" else (e - exit_price) * u
            balance += pl
            book(day, pl, None)

    def split(from_s: str, to_s: str) -> dict[str, Any]:
        days = {d: v for d, v in daily_pl.items() if from_s <= d < to_s}
        compound = 1.0
        equity_track = NAV_START
        worst = 0.0
        negd = 0
        best = 0.0
        for d in sorted(days):
            r = days[d] / max(equity_track, 1e-9)
            equity_track += days[d]
            compound *= 1.0 + r
            worst = min(worst, r)
            best = max(best, r)
            negd += int(days[d] < 0)
        return {
            "net_jpy": round(sum(days.values())),
            "nav_multiple": round(compound, 4),
            "active_days": len(days),
            "negative_days": negd,
            "worst_day_nav": round(worst, 4),
            "best_day_nav": round(best, 4),
        }

    return {
        "trades": trades,
        "win_rate": round(wins / trades, 4) if trades else None,
        "margin_closeouts": closeouts,
        "dead_day": dead_day,
        "final_balance_jpy": round(balance),
        "splits": {name: split(f, t) for name, (f, t) in SPLITS.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be clean; refusing stale reuse")
    rows = _load(args.root)
    replay_from_epoch = int(datetime.fromisoformat(REPLAY_FROM + "T00:00:00+00:00").timestamp())
    rows = [r for r in rows if r[0] >= replay_from_epoch]
    mid_close = {r[0] + 300: (r[4] + r[8]) / 2.0 for r in rows}

    raw = simulate(rows, mid_close, protected=False)
    protected = simulate(rows, mid_close, protected=True)
    gated = simulate(rows, mid_close, protected=True, regime_gated=True)

    body: dict[str, Any] = {
        "contract": "QR_MOMENTUM_BURST_REPLICA_V1",
        "schema_version": 1,
        "pair": PAIR,
        "declared_rule": (
            "24h trend side, 3-bar M5 break trigger, next-open entry, TP +3.0 "
            "pips pessimistic; RAW = TP-only unlimited stacking (as lived); "
            "PROTECTED = cap 3 + 60min time-stop + trend-flip flush"
        ),
        "derived_from": (
            "live MomentumBurst 2025-12-09 (+24,542 JPY, 53/53) and 2025-12-10 "
            "(-14,461 overnight give-back, no time-stop)"
        ),
        "single_candidate_no_selection": True,
        "accounting": {
            "nav_start_jpy": NAV_START,
            "per_position_leverage": PER_POSITION_LEVERAGE,
            "leverage": LEVERAGE,
            "sizing": "NAV_PROPORTIONAL",
        },
        "raw": raw,
        "protected": protected,
        "protected_trend_gated": gated,
        "candidate_multiplicity_note": (
            "third declared arm (protections + measured-TREND arm gate) added "
            "after the first two arms' verdicts were seen; treat any positive "
            "result as hypothesis-grade, not selection-grade"
        ),
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
    print(json.dumps({"status": "MOMENTUM_BURST_REPLICA_SEALED", "raw": raw, "protected": protected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
