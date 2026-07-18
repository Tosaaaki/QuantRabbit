#!/usr/bin/env python3
"""Margin-feasible nanpin: does the operator's design beat the baseline in money?

The sweep's nanpin win was per-unit; adds double a trade's exposure, so the
base book must shrink to reserve margin.  This rehearsal does the money
accounting under the operator's grant: peak margin usage capped at 92%.
Declared conservative concurrency: all 12 slots are treated as concurrent
(4h cadence x 12h hold = 3 waves x 4) and every add on a day is assumed
concurrent with them, so the day's peak exposure factor is
(12 + adds_that_day)/12 x base leverage.  The base leverage is set from the
TRAIN worst add-day so that peak usage <= 92% of the 25x cap; VALIDATION
replicates that fixed base unchanged.  Compare monthly multiples vs the
full-25x no-add baseline.  Shadow only; future window is the arbiter.
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


def _resolve_day_rows(series, outcomes):
    """Per-day: total per-unit pips with adds, and add count."""

    daily: dict[Any, dict[str, float]] = {}
    for row in outcomes:
        factor = float(instrument_pip_factor(row.pair))
        try:
            ladder = resolve_addon_ladder(
                series[row.pair], row, mode="NANPIN", step_pips=ADD_STEP_PIPS,
                max_adds=MAX_ADDS, pip_factor=factor,
            )
            total = float(ladder["blended_total_pips"])
            adds = int(ladder["units_filled"]) - 1
        except Exception:
            total = row.realized_pips
            adds = 0
        day = row.decision_utc.date()
        bucket = daily.setdefault(day, {"pips_total": 0.0, "adds": 0.0, "base_pips": 0.0})
        bucket["pips_total"] += total
        bucket["adds"] += adds
        bucket["base_pips"] += row.realized_pips
    return daily


def _monthly_multiple(daily_returns: list[float]) -> float:
    compound = 1.0
    for value in daily_returns:
        compound *= 1.0 + value
    if not daily_returns:
        return 1.0
    return (compound ** (1.0 / len(daily_returns))) ** TRADING_DAYS


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
        windows[label] = _resolve_day_rows(series, stopped)

    # TRAIN sets the base leverage: worst concurrent-add day must keep peak
    # usage <= 92% of the 25x cap.  (12 + adds)/12 * base_lev <= 0.92 * 25.
    max_daily_adds_train = max(
        (row["adds"] for row in windows["TRAIN"].values()), default=0.0
    )
    base_leverage = min(
        LEVERAGE_CAP,
        PEAK_USAGE_CAP * LEVERAGE_CAP * SLOTS / (SLOTS + max_daily_adds_train),
    )
    per_pip_base = 0.0001 * base_leverage / SLOTS
    per_pip_full = 0.0001 * LEVERAGE_CAP / SLOTS

    report: dict[str, Any] = {}
    for label, daily in windows.items():
        days = sorted(daily)
        nanpin_returns = [daily[d]["pips_total"] * per_pip_base for d in days]
        baseline_returns = [daily[d]["base_pips"] * per_pip_full for d in days]
        peak_usage = [
            (SLOTS + daily[d]["adds"]) / SLOTS * base_leverage / LEVERAGE_CAP
            for d in days
        ]
        report[label] = {
            "active_days": len(days),
            "baseline_full25x_monthly_multiple": round(
                _monthly_multiple(baseline_returns), 6
            ),
            "nanpin_92cap_monthly_multiple": round(
                _monthly_multiple(nanpin_returns), 6
            ),
            "worst_day_baseline": round(min(baseline_returns, default=0.0), 6),
            "worst_day_nanpin": round(min(nanpin_returns, default=0.0), 6),
            "max_peak_margin_usage": round(max(peak_usage, default=0.0), 6),
            "total_adds": int(sum(daily[d]["adds"] for d in days)),
        }

    body: dict[str, Any] = {
        "contract": "QR_NANPIN_MARGIN_FEASIBLE_REHEARSAL_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "peak_usage_cap": PEAK_USAGE_CAP,
        "base_leverage_set_on_train_only": round(base_leverage, 6),
        "concurrency_assumption": "ALL_SLOTS_AND_ALL_DAY_ADDS_CONCURRENT_CONSERVATIVE",
        "windows": report,
        "operator_grant": "OPERATOR_ALLOWED_UP_TO_95PCT_USED_92PCT_PEAK_CAP",
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
