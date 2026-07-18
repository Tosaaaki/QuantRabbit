#!/usr/bin/env python3
"""Monthly-multiple distribution: the honest form of "build a 3x system".

A trading system's monthly result is a distribution, not a constant.  The
operator's 2025 4x was a right-tail draw from a positive-expectancy,
concentration-sized process.  This simulator measures that distribution for
the current stack (survivor + adopted 50p stop) by week-block bootstrap:
resample the sealed daily P&L in 5-day blocks into 10,000 synthetic
22-trading-day months, compound at several leverage levels, and seal
P(>=3x), P(>=2x), P(losing month), median multiple, and P(deep drawdown).
The design objective "maximize P(3x month) subject to survival" becomes a
measured, optimizable number instead of a promise.  Shadow only; the
distribution inherits every unproven-ness of its inputs (final arbiter:
the future window).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
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
from quant_rabbit.daily_loss_overlay import apply_causal_daily_stop, daily_net_pips
from quant_rabbit.fast_bot_historical_s5 import load_historical_s5_slice
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS

TRAIN_FROM = datetime(2026, 5, 12, tzinfo=timezone.utc)
VALIDATION_TO = datetime(2026, 6, 28, tzinfo=timezone.utc)
LOOKBACK = timedelta(minutes=725)
STOP_PIPS = 50.0
RETURN_PER_PIP_AT_25X = 0.0001 * 25.0 / 12.0
LEVERAGE_GRID = (10.0, 25.0)
BLOCK_DAYS = 5
MONTH_DAYS = 22
SIMULATIONS = 10_000
SEED = 20260718
DEEP_DRAWDOWN = -0.30


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def simulate_months(
    daily_pips: list[float], *, leverage: float, rng: random.Random
) -> dict[str, Any]:
    per_pip = 0.0001 * leverage / 12.0
    blocks = [
        daily_pips[i : i + BLOCK_DAYS]
        for i in range(0, len(daily_pips) - BLOCK_DAYS + 1)
    ]
    if not blocks:
        raise ValueError("insufficient daily history for block bootstrap")
    multiples: list[float] = []
    deep = 0
    for _ in range(SIMULATIONS):
        days: list[float] = []
        while len(days) < MONTH_DAYS:
            days.extend(rng.choice(blocks))
        days = days[:MONTH_DAYS]
        nav = 1.0
        trough = 1.0
        peak = 1.0
        for pips in days:
            nav *= 1.0 + pips * per_pip
            peak = max(peak, nav)
            trough = min(trough, nav / peak)
        multiples.append(nav)
        if trough - 1.0 <= DEEP_DRAWDOWN:
            deep += 1
    multiples.sort()
    n = len(multiples)

    def q(p: float) -> float:
        return round(multiples[min(n - 1, int(p * n))], 6)

    return {
        "leverage": leverage,
        "p_ge_3x": round(sum(m >= 3.0 for m in multiples) / n, 6),
        "p_ge_2x": round(sum(m >= 2.0 for m in multiples) / n, 6),
        "p_ge_1_5x": round(sum(m >= 1.5 for m in multiples) / n, 6),
        "p_losing_month": round(sum(m < 1.0 for m in multiples) / n, 6),
        "p_deep_drawdown_30pct": round(deep / n, 6),
        "median_multiple": q(0.5),
        "p90_multiple": q(0.9),
        "p99_multiple": q(0.99),
        "p10_multiple": q(0.1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("distribution output must be clean; refusing stale reuse")
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

    outcomes = evaluate_spec(
        series, spec=spec, opened_from_utc=TRAIN_FROM, opened_to_utc=VALIDATION_TO,
        policy=policy,
    )
    stopped, _ = apply_causal_daily_stop(outcomes, stop_pips=STOP_PIPS)
    daily_map = daily_net_pips(stopped)
    daily = [daily_map[d] for d in sorted(daily_map)]

    rng = random.Random(SEED)
    rows = [simulate_months(daily, leverage=lev, rng=rng) for lev in LEVERAGE_GRID]

    body: dict[str, Any] = {
        "contract": "QR_MONTHLY_MULTIPLE_DISTRIBUTION_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "stack": "SURVIVOR_PLUS_50P_INTRADAY_STOP",
        "observed_active_days": len(daily),
        "block_days": BLOCK_DAYS,
        "month_days": MONTH_DAYS,
        "simulations": SIMULATIONS,
        "seed": SEED,
        "distribution_rows": rows,
        "design_objective": "MAXIMIZE_P_3X_MONTH_SUBJECT_TO_SURVIVAL",
        "next_levers_to_raise_p3x": [
            "add uncorrelated lanes (addition theorem W28)",
            "calibrated conviction concentration (operator 2025 mechanism)",
            "extend history so the distribution is not one regime",
        ],
        "unproven_inputs_disclosure": "daily P&L is TRAIN+VAL shadow; future window is the arbiter",
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "distribution_sha256": _canonical_sha(body)}
    payload = json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_name, args.output)
    print(json.dumps({"status": "DISTRIBUTION_SEALED", "rows": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
