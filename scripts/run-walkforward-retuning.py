#!/usr/bin/env python3
"""Walk-forward retuning: the operator's architecture premise, tested.

Every fixed rule died over six years, yet each was profitable in its own
regime — a toolbox, not a graveyard, IF the periodic retuning loop works.
This simulates the loop mechanically: a declared toolbox of eight
cross-sectional tools (orientation DIRECT/INVERSE x hold 4h/12h x lookback
8h/24h) plus FLAT; every Monday the supervisor re-selects the tool with
the best trailing 28-day NAV (FLAT when nothing trailing-positive), and
runs it for the week.  Selection uses only already-realized past days —
fully causal.  Execution: real M5 bid/ask opens, rank-2 both sides,
ready-floor 24, weekend gate, 50p intraday stop, money accounting at the
200k JPY standard (21.4x over 12 slots).  A trailing-P&L picker is a WEAK
proxy for the real AI supervisor (which also reads the board and news):
a positive result validates the premise; a negative one bounds this proxy,
not the premise.  Shadow only; 2026 untouched.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import os
import tempfile
from array import array
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, instrument_pip_factor

UTC = timezone.utc
REPLAY_FROM = datetime(2020, 3, 2, tzinfo=UTC)  # after trailing warmup
REPLAY_TO = datetime(2026, 1, 1, tzinfo=UTC)
CADENCE_S = 4 * 3600
RANK = 2
READY_FLOOR = 24
WEEKEND_MARGIN_MIN = 5
STOP_PIPS = 50.0
NAV_JPY = 200_000.0
LEVERAGE = 21.4
SLOTS = 12
PER_LEG_LEV = LEVERAGE / SLOTS
COST_STRESS_RETURN = 0.00005 * PER_LEG_LEV
TRAIL_DAYS = 28
RETUNE_WEEKDAY = 0  # Monday
TOOLS = [
    {"tool_id": f"{o}_L{lb}H{h}", "orientation": o, "lookback_s": lb * 3600, "hold_s": h * 3600}
    for o in ("DIRECT", "INVERSE")
    for lb in (8, 24)
    for h in (4, 12)
]


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_pair(root: Path, pair: str):
    epochs = array("q")
    bid_o = array("d")
    ask_o = array("d")
    close_mid: dict[int, float] = {}
    for shard_file in sorted(root.glob(f"*/{pair}/{pair}_M5_BA_*.jsonl.gz")):
        with gzip.open(shard_file, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                epoch = int(
                    datetime.fromisoformat(row["time"][:19] + "+00:00").timestamp()
                )
                epochs.append(epoch)
                bid_o.append(float(row["bid"]["o"]))
                ask_o.append(float(row["ask"]["o"]))
                close_mid[epoch + 300] = (
                    float(row["bid"]["c"]) + float(row["ask"]["c"])
                ) / 2.0
    return epochs, bid_o, ask_o, close_mid


def _tool_daily(tool, data, start: int, end: int) -> tuple[dict[str, float], dict[str, float]]:
    """Daily NAV returns and daily pips for one tool over [start, end)."""

    lookback_s = tool["lookback_s"]
    hold_s = tool["hold_s"]
    inverse = tool["orientation"] == "INVERSE"
    daily_nav: dict[str, float] = {}
    daily_pips: dict[str, float] = {}
    exits: list[tuple[int, str, float]] = []
    first_decision = ((start + CADENCE_S - 1) // CADENCE_S) * CADENCE_S
    for decision_epoch in range(first_decision, end, CADENCE_S):
        if decision_epoch + hold_s >= end:
            break
        decision = datetime.fromtimestamp(decision_epoch, tz=UTC)
        status = compute_market_status(decision)
        if not status.is_fx_open or status.minutes_to_next_close is None or (
            status.minutes_to_next_close <= hold_s / 60 + WEEKEND_MARGIN_MIN
        ):
            continue
        day = decision.date().isoformat()
        realized_today = sum(
            pips for exit_epoch, exit_day, pips in exits
            if exit_day == day and exit_epoch <= decision_epoch
        )
        if realized_today <= -STOP_PIPS:
            continue
        scored: list[tuple[float, str]] = []
        for pair, (epochs, bid_o, ask_o, close_mid) in data.items():
            now = close_mid.get(decision_epoch)
            past = close_mid.get(decision_epoch - lookback_s)
            if now is None or past is None:
                continue
            factor = float(instrument_pip_factor(pair))
            scored.append(((now - past) * factor, pair))
        if len(scored) < READY_FLOOR:
            continue
        scored.sort()
        if inverse:
            legs = [(pair, "LONG") for _, pair in scored[:RANK]] + [
                (pair, "SHORT") for _, pair in scored[-RANK:]
            ]
        else:
            legs = [(pair, "SHORT") for _, pair in scored[:RANK]] + [
                (pair, "LONG") for _, pair in scored[-RANK:]
            ]
        for pair, side in legs:
            epochs, bid_o, ask_o, _ = data[pair]
            i = bisect.bisect_left(epochs, decision_epoch)
            if i >= len(epochs) or int(epochs[i]) >= end:
                continue
            entry_epoch = int(epochs[i])
            j = bisect.bisect_left(epochs, entry_epoch + hold_s)
            if j >= len(epochs) or int(epochs[j]) >= end:
                continue
            exit_epoch = int(epochs[j])
            factor = float(instrument_pip_factor(pair))
            if side == "LONG":
                entry, exit_price = float(ask_o[i]), float(bid_o[j])
                pips = (exit_price - entry) * factor
                nav = (exit_price / entry - 1.0) * PER_LEG_LEV - COST_STRESS_RETURN
            else:
                entry, exit_price = float(bid_o[i]), float(ask_o[j])
                pips = (entry - exit_price) * factor
                nav = (entry / exit_price - 1.0) * PER_LEG_LEV - COST_STRESS_RETURN
            daily_nav[day] = daily_nav.get(day, 0.0) + nav
            daily_pips[day] = daily_pips.get(day, 0.0) + pips
            exits.append((exit_epoch, day, pips))
    return daily_nav, daily_pips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be clean; refusing stale reuse")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    body_check = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _canonical_sha(body_check):
        raise ValueError("M5 manifest digest is invalid")

    data: dict[str, tuple] = {}
    for pair in DEFAULT_TRADER_PAIRS:
        data[pair] = _load_pair(args.root, pair)

    sim_start = int((REPLAY_FROM - timedelta(days=TRAIL_DAYS + 7)).timestamp())
    end = int(REPLAY_TO.timestamp())
    tool_daily: dict[str, dict[str, float]] = {}
    for tool in TOOLS:
        nav, _pips = _tool_daily(tool, data, sim_start, end)
        tool_daily[tool["tool_id"]] = nav

    all_days = sorted({d for nav in tool_daily.values() for d in nav})
    supervised: dict[str, float] = {}
    schedule: list[dict[str, str]] = []
    current = "FLAT"
    for day in all_days:
        stamp = datetime.fromisoformat(day + "T00:00:00+00:00")
        if stamp < REPLAY_FROM:
            continue
        if stamp.weekday() == RETUNE_WEEKDAY or not schedule:
            cutoff = day
            trail_from = (stamp - timedelta(days=TRAIL_DAYS)).date().isoformat()
            best_tool, best_sum = "FLAT", 0.0
            for tool_id, nav in tool_daily.items():
                trailing = sum(
                    v for d, v in nav.items() if trail_from <= d < cutoff
                )
                if trailing > best_sum:
                    best_tool, best_sum = tool_id, trailing
            current = best_tool
            schedule.append({"from": day, "tool": current})
        if current != "FLAT":
            supervised[day] = tool_daily[current].get(day, 0.0)
        else:
            supervised[day] = 0.0

    def stats(daily: dict[str, float]) -> dict[str, Any]:
        compound = 1.0
        worst = 0.0
        negd = 0
        years: dict[str, float] = {}
        for day in sorted(daily):
            r = daily[day]
            compound *= 1.0 + r
            worst = min(worst, r)
            negd += int(r < 0)
            years[day[:4]] = years.get(day[:4], 1.0) * (1.0 + r)
        months = max(1e-9, len(daily) / 21.7)
        return {
            "nav_multiple": round(compound, 6),
            "monthly_multiple": round(compound ** (1.0 / months), 6) if daily else 1.0,
            "worst_day_nav": round(worst, 6),
            "negative_days": negd,
            "active_days": len(daily),
            "by_year": {y: round(v - 1.0, 4) for y, v in sorted(years.items())},
        }

    fixed_rows = {
        tool_id: stats({d: v for d, v in nav.items() if d >= REPLAY_FROM.date().isoformat()})
        for tool_id, nav in tool_daily.items()
    }
    supervised_stats = stats(supervised)
    flat_share = sum(1 for row in schedule if row["tool"] == "FLAT") / max(1, len(schedule))
    switches = sum(
        1 for a, b in zip(schedule, schedule[1:]) if a["tool"] != b["tool"]
    )

    body: dict[str, Any] = {
        "contract": "QR_WALKFORWARD_RETUNING_V1",
        "schema_version": 1,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "premise_under_test": (
            "operator architecture: AI periodically retunes the bot; trailing-"
            "28d weekly re-selection is the mechanical proxy for that loop"
        ),
        "toolbox": [tool["tool_id"] for tool in TOOLS] + ["FLAT"],
        "retune": {"cadence": "WEEKLY_MONDAY", "trailing_days": TRAIL_DAYS,
                   "rule": "BEST_TRAILING_NAV_ELSE_FLAT"},
        "window": [REPLAY_FROM.isoformat(), REPLAY_TO.isoformat()],
        "accounting": {"standard_nav_jpy": NAV_JPY, "leverage": LEVERAGE, "slots": SLOTS},
        "supervised_walkforward": supervised_stats,
        "flat_week_share": round(flat_share, 4),
        "tool_switches": switches,
        "fixed_tool_rows": fixed_rows,
        "selection_is_causal_trailing_only": True,
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
    print(
        json.dumps(
            {
                "status": "WALKFORWARD_RETUNING_SEALED",
                "supervised": supervised_stats,
                "flat_share": round(flat_share, 3),
                "switches": switches,
                "best_fixed": max(
                    fixed_rows.items(), key=lambda kv: kv[1]["nav_multiple"]
                )[0],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
