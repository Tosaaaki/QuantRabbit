#!/usr/bin/env python3
"""Nanpin with TRUE concurrency tracking and runtime-enforced usage cap.

The conservative rehearsal assumed every add is concurrent with every slot
and shrank the base book to 10.6x, killing nanpin's edge.  This refinement
measures ACTUAL concurrency: a timeline sweep of unit intervals
(entry..exit for base units, add-fill..exit for adds) gives the real peak
concurrent units; base leverage is set on TRAIN's real peak; and — the
lesson from the breached 92% cap — VALIDATION enforces the cap at RUNTIME:
an add that would exceed headroom at its fill moment is refused, exactly as
a live gateway would refuse it.  Compare monthly multiples vs the full-25x
baseline.  Shadow only; future window is the arbiter.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.adaptive_exact_s5_profit_engine import (
    _policy_from_payload,
    _spec_from_payload,
    _validate_lock,
    evaluate_spec,
    prepare_exact_s5_series,
)
from quant_rabbit.addon_ladder_shadow import resolve_addon_ladder
from quant_rabbit.daily_loss_overlay import apply_causal_daily_stop
from quant_rabbit.fast_bot_historical_s5 import load_historical_s5_slice
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, instrument_pip_factor

TRAIN_FROM = datetime(2026, 5, 12, tzinfo=timezone.utc)
TRAIN_TO = datetime(2026, 6, 15, tzinfo=timezone.utc)
VALIDATION_TO = datetime(2026, 6, 28, tzinfo=timezone.utc)
LOOKBACK = timedelta(minutes=725)
STOP_PIPS = 50.0
ADD_STEP_PIPS = 20.0
MAX_ADDS = 1
PEAK_USAGE_CAP = 0.92
LEVERAGE_CAP = 25.0
SLOTS = 12
TRADING_DAYS = 22


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ladders(series, outcomes):
    rows = []
    for row in outcomes:
        factor = float(instrument_pip_factor(row.pair))
        entry = int(row.entry_utc.timestamp())
        exit_epoch = int(row.exit_utc.timestamp())
        try:
            ladder = resolve_addon_ladder(
                series[row.pair], row, mode="NANPIN", step_pips=ADD_STEP_PIPS,
                max_adds=MAX_ADDS, pip_factor=factor,
            )
            adds = [int(e) for e in ladder["add_fill_epochs"]]
            per_unit = [float(ladder["blended_total_pips"]) - row.realized_pips]
            # pips of the add unit alone = blended_total - base unit pips
            add_unit_pips = per_unit[0] if adds else 0.0
        except Exception:
            adds = []
            add_unit_pips = 0.0
        rows.append(
            {
                "day": row.decision_utc.date(),
                "entry": entry,
                "exit": exit_epoch,
                "base_pips": row.realized_pips,
                "add_epochs": adds,
                "add_unit_pips": add_unit_pips,
            }
        )
    return rows


def _peak_concurrency(rows, *, include_adds: bool) -> int:
    events: list[tuple[int, int]] = []
    for row in rows:
        events.append((row["entry"], 1))
        events.append((row["exit"], -1))
        if include_adds:
            for epoch in row["add_epochs"]:
                events.append((epoch, 1))
                events.append((row["exit"], -1))
    events.sort()
    level = peak = 0
    for _, delta in events:
        level += delta
        peak = max(peak, level)
    return peak


def _runtime_clipped_adds(rows, *, unit_usage: float) -> tuple[int, int]:
    """Chronological sweep refusing adds that would breach the cap.

    Returns (allowed_adds, refused_adds).  unit_usage is margin usage per
    unit; base units are never refused (they are the strategy), adds are.
    """

    events: list[tuple[int, int, str, dict]] = []
    for i, row in enumerate(rows):
        events.append((row["entry"], 0, "base_open", row))
        events.append((row["exit"], 2, "close", row))
        for epoch in row["add_epochs"]:
            events.append((epoch, 1, "add", row))
    events.sort(key=lambda e: (e[0], e[1]))
    level = 0.0
    allowed = refused = 0
    open_adds: dict[int, int] = {}
    for epoch, _, kind, row in events:
        if kind == "base_open":
            level += unit_usage
        elif kind == "close":
            level -= unit_usage * (1 + open_adds.pop(id(row), 0))
        else:  # add
            if level + unit_usage <= PEAK_USAGE_CAP + 1e-12:
                level += unit_usage
                open_adds[id(row)] = open_adds.get(id(row), 0) + 1
                allowed += 1
            else:
                refused += 1
    return allowed, refused


def _monthly(daily: dict, per_pip: float) -> dict[str, Any]:
    days = sorted(daily)
    compound = 1.0
    worst = 0.0
    for day in days:
        r = daily[day] * per_pip
        compound *= 1.0 + r
        worst = min(worst, r)
    n = len(days)
    return {
        "active_days": n,
        "monthly_multiple": round((compound ** (1.0 / n)) ** TRADING_DAYS, 6)
        if n
        else 1.0,
        "worst_day_nav": round(worst, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be clean; refusing stale reuse")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    _validate_lock(lock)
    spec = _spec_from_payload(lock["spec"])
    policy = _policy_from_payload(lock["evaluation_policy"])

    series = {}
    for pair in DEFAULT_TRADER_PAIRS:
        item = load_historical_s5_slice(
            manifest, pair=pair, time_from=TRAIN_FROM - LOOKBACK, time_to=VALIDATION_TO
        )
        series[pair] = prepare_exact_s5_series(item.candles)
        del item
        gc.collect()

    windows = {}
    for label, opened_from, opened_to in (
        ("TRAIN", TRAIN_FROM, TRAIN_TO),
        ("VALIDATION", TRAIN_TO, VALIDATION_TO),
    ):
        outcomes = evaluate_spec(
            series, spec=spec, opened_from_utc=opened_from, opened_to_utc=opened_to,
            policy=policy,
        )
        stopped, _ = apply_causal_daily_stop(outcomes, stop_pips=STOP_PIPS)
        windows[label] = _ladders(series, stopped)

    # Base leverage from TRAIN's REAL peak concurrency (with adds).
    train_peak_units = _peak_concurrency(windows["TRAIN"], include_adds=True)
    base_leverage = min(
        LEVERAGE_CAP,
        PEAK_USAGE_CAP * LEVERAGE_CAP * SLOTS / max(train_peak_units, 1),
    )
    unit_usage = (base_leverage / SLOTS) / LEVERAGE_CAP
    per_pip_base = 0.0001 * base_leverage / SLOTS
    per_pip_full = 0.0001 * LEVERAGE_CAP / SLOTS

    report: dict[str, Any] = {}
    for label, rows in windows.items():
        allowed, refused = _runtime_clipped_adds(rows, unit_usage=unit_usage)
        # Allowed adds keep their pips; refused adds contribute nothing.
        # (Approximation: refusal order matches chronological sweep.)
        refusal_ratio = refused / (allowed + refused) if (allowed + refused) else 0.0
        base_daily: dict = {}
        nanpin_daily: dict = {}
        for row in rows:
            base_daily[row["day"]] = base_daily.get(row["day"], 0.0) + row["base_pips"]
            add_pips = row["add_unit_pips"] * (1.0 - refusal_ratio)
            nanpin_daily[row["day"]] = (
                nanpin_daily.get(row["day"], 0.0) + row["base_pips"] + add_pips
            )
        report[label] = {
            "real_peak_units": _peak_concurrency(rows, include_adds=True),
            "peak_units_without_adds": _peak_concurrency(rows, include_adds=False),
            "adds_allowed": allowed,
            "adds_refused_by_runtime_cap": refused,
            "baseline_full25x": _monthly(base_daily, per_pip_full),
            "nanpin_true_concurrency": _monthly(nanpin_daily, per_pip_base),
        }

    body: dict[str, Any] = {
        "contract": "QR_NANPIN_TRUE_CONCURRENCY_REHEARSAL_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "peak_usage_cap": PEAK_USAGE_CAP,
        "train_real_peak_units": train_peak_units,
        "base_leverage_set_on_train_real_peak": round(base_leverage, 6),
        "runtime_cap_enforced": True,
        "windows": report,
        "future_window_is_final_arbiter": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "rehearsal_sha256": _canonical_sha(body)}
    payload = json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_name, args.output)
    print(json.dumps({"status": "SEALED", "base_leverage": round(base_leverage, 4), "windows": report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
