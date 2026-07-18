#!/usr/bin/env python3
"""Six-year backward out-of-sample replay of the adopted survivor (M5 replica).

The adopted stack's evidence is 34 days of May-June 2026.  This replays an
M5-executable replica of the SAME declared strategy — cross-sectional
RETURN_PIPS rank-2 both sides, 8h lookback signal from closes landing
exactly on the decision clock, 12h hold from entry, 4h cadence, ready
floor 24/28, pre-entry continuous-hold weekend gate, adopted 50p intraday
stop — over 2020-2025: data this strategy has NEVER seen.  One declared
candidate, no selection, no grid: a pure diagnostic of whether the family
survives six years of regimes.  Execution at real M5 bid/ask opens; money
accounting per the 200k JPY standard at the executable 21.4x over 12
slots.  2026 stays untouched (its own future test).  Shadow only.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, instrument_pip_factor

UTC = timezone.utc
REPLAY_FROM = datetime(2020, 1, 6, tzinfo=UTC)  # after first signal warmup
REPLAY_TO = datetime(2026, 1, 1, tzinfo=UTC)
CADENCE_S = 4 * 3600
LOOKBACK_S = 8 * 3600
SHORT_S = 1 * 3600
KEY_4H_S = 4 * 3600
HOLD_S = 12 * 3600
RANK = 2
READY_FLOOR = 24
WEEKEND_MARGIN_MIN = 5
STOP_PIPS = 50.0
NAV_JPY = 200_000.0
LEVERAGE = 21.4
SLOTS = 12
PER_LEG_LEV = LEVERAGE / SLOTS
COST_STRESS_RETURN = 0.00005 * PER_LEG_LEV


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("replay output must be clean; refusing stale reuse")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    body_check = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _canonical_sha(body_check):
        raise ValueError("M5 manifest digest is invalid")

    data: dict[str, tuple] = {}
    for pair in DEFAULT_TRADER_PAIRS:
        data[pair] = _load_pair(args.root, pair)

    start = int(REPLAY_FROM.timestamp())
    end = int(REPLAY_TO.timestamp())
    first_decision = ((start + CADENCE_S - 1) // CADENCE_S) * CADENCE_S

    daily_pips: dict[str, float] = {}
    daily_nav: dict[str, float] = {}
    trades = 0
    skipped_by_stop = 0
    # (day -> realized pips known so far, keyed by exit; simple causal stop)
    exits: list[tuple[int, str, float]] = []  # (exit_epoch, day, pips)

    for decision_epoch in range(first_decision, end, CADENCE_S):
        if decision_epoch + HOLD_S >= end:
            break
        decision = datetime.fromtimestamp(decision_epoch, tz=UTC)
        status = compute_market_status(decision)
        if not status.is_fx_open or status.minutes_to_next_close is None or (
            status.minutes_to_next_close <= HOLD_S / 60 + WEEKEND_MARGIN_MIN
        ):
            continue
        day = decision.date().isoformat()
        # Adopted 50p stop: realized (already exited) pips of this UTC day.
        realized_today = sum(
            pips for exit_epoch, exit_day, pips in exits
            if exit_day == day and exit_epoch <= decision_epoch
        )
        if realized_today <= -STOP_PIPS:
            skipped_by_stop += 1
            continue

        scored: list[tuple[float, str]] = []
        for pair, (epochs, bid_o, ask_o, close_mid) in data.items():
            now = close_mid.get(decision_epoch)
            past = close_mid.get(decision_epoch - LOOKBACK_S)
            short = close_mid.get(decision_epoch - SHORT_S)
            key4 = close_mid.get(decision_epoch - KEY_4H_S)
            if None in (now, past, short, key4):
                continue
            factor = float(instrument_pip_factor(pair))
            scored.append(((now - past) * factor, pair))
        if len(scored) < READY_FLOOR:
            continue
        scored.sort()
        legs = [(pair, "SHORT") for _, pair in scored[:RANK]] + [
            (pair, "LONG") for _, pair in scored[-RANK:]
        ]
        for pair, side in legs:
            epochs, bid_o, ask_o, _ = data[pair]
            i = bisect.bisect_left(epochs, decision_epoch)
            if i >= len(epochs) or int(epochs[i]) >= end:
                continue
            entry_epoch = int(epochs[i])
            j = bisect.bisect_left(epochs, entry_epoch + HOLD_S)
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
            trades += 1
            daily_pips[day] = daily_pips.get(day, 0.0) + pips
            daily_nav[day] = daily_nav.get(day, 0.0) + nav
            exits.append((exit_epoch, day, pips))

    years: dict[str, dict[str, Any]] = {}
    for day in sorted(daily_nav):
        year = day[:4]
        bucket = years.setdefault(
            year, {"nav_compound": 1.0, "pips": 0.0, "days": 0, "neg_days": 0, "worst": 0.0}
        )
        r = daily_nav[day]
        bucket["nav_compound"] *= 1.0 + r
        bucket["pips"] += daily_pips[day]
        bucket["days"] += 1
        bucket["neg_days"] += int(r < 0)
        bucket["worst"] = min(bucket["worst"], r)
    year_rows = [
        {
            "year": year,
            "nav_return": round(b["nav_compound"] - 1.0, 6),
            "net_pips": round(b["pips"], 1),
            "active_days": b["days"],
            "negative_days": b["neg_days"],
            "worst_day_nav": round(b["worst"], 6),
        }
        for year, b in sorted(years.items())
    ]
    total = 1.0
    for day in sorted(daily_nav):
        total *= 1.0 + daily_nav[day]
    months = (end - start) / (30.44 * 86400)

    body: dict[str, Any] = {
        "contract": "QR_M5_SURVIVOR_REPLICA_SIX_YEAR_REPLAY_V1",
        "schema_version": 1,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "strategy": "M5_REPLICA_OF_ADOPTED_SURVIVOR_RANK2_8H_12H_4H_WEEKEND_GATE_50P_STOP",
        "declared_single_candidate_no_selection": True,
        "window": [REPLAY_FROM.isoformat(), REPLAY_TO.isoformat()],
        "accounting": {
            "standard_nav_jpy": NAV_JPY,
            "leverage": LEVERAGE,
            "slots": SLOTS,
            "per_trade_cost_stress_nav": COST_STRESS_RETURN,
        },
        "trades": trades,
        "entries_skipped_by_stop": skipped_by_stop,
        "year_rows": year_rows,
        "total_nav_multiple": round(total, 6),
        "mean_monthly_multiple": round(total ** (1.0 / months), 6),
        "backward_out_of_sample": True,
        "test_2026_untouched": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "replay_sha256": _canonical_sha(body)}
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
                "status": "SIX_YEAR_REPLAY_SEALED",
                "total_nav_multiple": body["total_nav_multiple"],
                "mean_monthly_multiple": body["mean_monthly_multiple"],
                "years": year_rows,
                "trades": trades,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
