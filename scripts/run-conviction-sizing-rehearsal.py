#!/usr/bin/env python3
"""Test the operator's mechanism: concentration-sized lots raise the multiple.

Uniform sizing wastes leverage on weak signals.  Arms compare, at the SAME
average exposure budget (25x cap / 12 concurrent):
  UNIFORM        w_i = 1
  SCORE_PROP     w_i proportional to |score_i| (day-normalized, capped 3x)
  SCORE_PROP_SQ  w_i proportional to score^2 (harder concentration, cap 4x)
Daily NAV return compounds w_i-weighted pips; the report seals the
22-trading-day multiple and the worst day for each arm.  The 50p SKIP stop
applies first.  Weights use only decision-time |score| — the mechanized
form of the operator's discretionary concentration.  Shadow only.
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
from quant_rabbit.daily_loss_overlay import apply_causal_daily_stop
from quant_rabbit.fast_bot_historical_s5 import load_historical_s5_slice
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS

TRAIN_FROM = datetime(2026, 5, 12, tzinfo=timezone.utc)
TRAIN_TO = datetime(2026, 6, 15, tzinfo=timezone.utc)
VALIDATION_TO = datetime(2026, 6, 28, tzinfo=timezone.utc)
LOOKBACK = timedelta(minutes=725)
STOP_PIPS = 50.0
RETURN_PER_PIP = 0.0001 * 25.0 / 12.0
TRADING_DAYS = 22


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _weights(day_rows, arm: str) -> list[float]:
    if arm == "UNIFORM":
        return [1.0] * len(day_rows)
    if arm == "SCORE_PROP":
        raw = [abs(row.score) for row in day_rows]
        cap = 3.0
    else:  # SCORE_PROP_SQ
        raw = [row.score * row.score for row in day_rows]
        cap = 4.0
    mean = sum(raw) / len(raw)
    if mean <= 0:
        return [1.0] * len(day_rows)
    return [min(value / mean, cap) for value in raw]


def _arm_report(outcomes, arm: str) -> dict[str, Any]:
    stopped, _ = apply_causal_daily_stop(outcomes, stop_pips=STOP_PIPS)
    by_day: dict[Any, list] = {}
    for row in stopped:
        by_day.setdefault(row.decision_utc.date(), []).append(row)
    daily_returns = []
    for day in sorted(by_day):
        rows = by_day[day]
        weights = _weights(rows, arm)
        norm = sum(weights) / len(weights)
        pips = sum(w * row.realized_pips for w, row in zip(weights, rows)) / norm
        daily_returns.append(pips * RETURN_PER_PIP)
    active = len(daily_returns)
    compound = 1.0
    for value in daily_returns:
        compound *= 1.0 + value
    per_day = compound ** (1.0 / active) if active else 1.0
    return {
        "arm": arm,
        "active_days": active,
        "worst_day_nav_fraction": round(min(daily_returns), 9) if active else 0.0,
        "compound_over_window": round(compound, 9),
        "monthly_multiple_22_trading_days": round(per_day**TRADING_DAYS, 9),
        "negative_days": sum(value < 0 for value in daily_returns),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("rehearsal output must be clean; refusing stale reuse")
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

    windows: dict[str, Any] = {}
    for label, opened_from, opened_to in (
        ("TRAIN", TRAIN_FROM, TRAIN_TO),
        ("VALIDATION", TRAIN_TO, VALIDATION_TO),
    ):
        outcomes = evaluate_spec(
            series,
            spec=spec,
            opened_from_utc=opened_from,
            opened_to_utc=opened_to,
            policy=policy,
        )
        windows[label] = [
            _arm_report(outcomes, arm)
            for arm in ("UNIFORM", "SCORE_PROP", "SCORE_PROP_SQ")
        ]

    body: dict[str, Any] = {
        "contract": "QR_CONVICTION_SIZING_REHEARSAL_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "exposure_model": "SAME_MEAN_BUDGET_25X_OVER_12_DAY_NORMALIZED_WEIGHTS",
        "stop_pips": STOP_PIPS,
        "windows": windows,
        "operator_mechanism_under_test": (
            "concentration-sized lots (2025 operator precedent) raise the "
            "monthly multiple at unchanged average exposure"
        ),
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
    print(json.dumps({"status": "SEALED", "windows": windows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
