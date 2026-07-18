#!/usr/bin/env python3
"""Lane addition: does momentum + range-rail raise the risk-adjusted multiple?

The all-weather theorem says only ADDING a family that wins where momentum
bleeds can improve both net and drawdown.  This measures the first
addition: at each decision, the momentum survivor trades as-is; on pairs
classified RANGE (where momentum bleeds), a two-sided passive range-rail
rotation trades independently with rails pre-committed from closed candles.
Combine daily P&L, compute the 22-trading-day multiple at the 25x cap for
momentum-alone vs momentum+range, and report the honest delta.  The range
lane's statistical edge still awaits the M5 corpus; this is a same-window
addition signal on already-sealed S5, not a proof.  Shadow only.
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
from quant_rabbit.fast_bot_historical_s5 import load_historical_s5_slice
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, instrument_pip_factor
from quant_rabbit.range_rail_shadow import resolve_range_rail_rotation
from quant_rabbit.regime_classifier_shadow import (
    VOL_HISTORY_WINDOW,
    RegimeClassifierError,
    classify_regime,
)

TRAIN_FROM = datetime(2026, 5, 12, tzinfo=timezone.utc)
VALIDATION_TO = datetime(2026, 6, 28, tzinfo=timezone.utc)
LOOKBACK = timedelta(minutes=725)
CADENCE_MIN = 240
RAIL_LOOKBACK_MIN = 60
ENTRY_TTL_SECONDS = 3600
HORIZON_SECONDS = 4 * 3600
RETURN_PER_PIP = 0.0001 * 25.0 / 12.0
TRADING_DAYS = 22


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _daily(pips_by_day: dict) -> dict:
    return {day: round(v, 9) for day, v in pips_by_day.items()}


def _multiple(daily_pips: dict) -> dict[str, Any]:
    days = sorted(daily_pips)
    if not days:
        return {"active_days": 0, "monthly_multiple": 1.0, "negative_days": 0, "worst_day_nav": 0.0, "net_pips": 0.0}
    compound = 1.0
    worst = 0.0
    negs = 0
    for day in days:
        r = daily_pips[day] * RETURN_PER_PIP
        compound *= 1.0 + r
        worst = min(worst, r)
        negs += int(daily_pips[day] < 0)
    per_day = compound ** (1.0 / len(days))
    return {
        "active_days": len(days),
        "net_pips": round(sum(daily_pips.values()), 9),
        "negative_days": negs,
        "worst_day_nav_fraction": round(worst, 9),
        "monthly_multiple_22d": round(per_day**TRADING_DAYS, 9),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("combination output must be clean; refusing stale reuse")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    _validate_lock(lock)
    spec = _spec_from_payload(lock["spec"])
    policy = _policy_from_payload(lock["evaluation_policy"])

    series = {}
    minute_index: dict[str, list] = {}
    for pair in DEFAULT_TRADER_PAIRS:
        item = load_historical_s5_slice(
            manifest, pair=pair, time_from=TRAIN_FROM - LOOKBACK, time_to=VALIDATION_TO
        )
        prepared = prepare_exact_s5_series(item.candles)
        series[pair] = prepared
        minute_index[pair] = [
            {"time": p.minute_utc, "close": p.mid_close} for p in prepared.points
        ]
        del item
        gc.collect()

    momentum = evaluate_spec(
        series, spec=spec, opened_from_utc=TRAIN_FROM, opened_to_utc=VALIDATION_TO,
        policy=policy,
    )
    momentum_daily: dict = {}
    for row in momentum:
        d = row.decision_utc.date()
        momentum_daily[d] = momentum_daily.get(d, 0.0) + row.realized_pips

    # Range lane: at each 4h decision, on RANGE-classified pairs, two-sided
    # passive rail rotation with rails pre-committed from closed candles.
    range_daily: dict = {}
    range_trades = 0
    range_fills = 0
    start_epoch = int(TRAIN_FROM.timestamp())
    end_epoch = int(VALIDATION_TO.timestamp())
    step = CADENCE_MIN * 60
    first = ((start_epoch + step - 1) // step) * step
    for decision_epoch in range(first, end_epoch, step):
        decision = datetime.fromtimestamp(decision_epoch, tz=timezone.utc)
        if decision_epoch + HORIZON_SECONDS >= end_epoch:
            break
        for pair in DEFAULT_TRADER_PAIRS:
            candles = [c for c in minute_index[pair] if c["time"] < decision]
            if len(candles) < VOL_HISTORY_WINDOW:
                continue
            try:
                cls = classify_regime(candles[-VOL_HISTORY_WINDOW - 5:], as_of_utc=decision)
            except RegimeClassifierError:
                continue
            if cls["regime"] != "RANGE":
                continue
            rail_candles = candles[-RAIL_LOOKBACK_MIN:]
            highs = [c["close"] for c in rail_candles]
            upper = max(highs)
            lower = min(highs)
            if upper <= lower:
                continue
            mid = (upper + lower) / 2.0
            half = (upper - lower) / 2.0
            buffer = 0.5 * half
            factor = float(instrument_pip_factor(pair))
            provenance = _canonical_sha([round(c["close"], 9) for c in rail_candles])
            for side, rail, stop in (
                ("SHORT", upper, upper + buffer),
                ("LONG", lower, lower - buffer),
            ):
                if not (half > 0 and buffer > 0):
                    continue
                try:
                    outcome = resolve_range_rail_rotation(
                        series[pair],
                        side=side,
                        rail_price=rail,
                        mid_price=mid,
                        stop_price=stop,
                        rail_provenance_sha256=provenance,
                        decision_utc=decision,
                        entry_ttl_seconds=ENTRY_TTL_SECONDS,
                        horizon_seconds=HORIZON_SECONDS,
                        pip_factor=factor,
                    )
                except Exception:
                    continue
                range_trades += 1
                if outcome.get("result_available"):
                    range_fills += 1
                    d = decision.date()
                    range_daily[d] = range_daily.get(d, 0.0) + float(outcome["realized_pips"])

    combined_daily = dict(momentum_daily)
    for d, v in range_daily.items():
        combined_daily[d] = combined_daily.get(d, 0.0) + v

    momentum_m = _multiple(_daily(momentum_daily))
    range_m = _multiple(_daily(range_daily))
    combined_m = _multiple(_daily(combined_daily))

    body: dict[str, Any] = {
        "contract": "QR_LANE_ADDITION_COMBINATION_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "window": [TRAIN_FROM.isoformat(), VALIDATION_TO.isoformat()],
        "range_lane_config": {
            "cadence_minutes": CADENCE_MIN,
            "rail_lookback_minutes": RAIL_LOOKBACK_MIN,
            "entry_ttl_seconds": ENTRY_TTL_SECONDS,
            "horizon_seconds": HORIZON_SECONDS,
            "only_on_measured_range": True,
        },
        "range_lane_attempts": range_trades,
        "range_lane_fills": range_fills,
        "momentum_alone": momentum_m,
        "range_lane_alone": range_m,
        "momentum_plus_range": combined_m,
        "multiple_delta_22d": round(
            combined_m["monthly_multiple_22d"] - momentum_m["monthly_multiple_22d"], 9
        ),
        "negative_day_delta": combined_m["negative_days"] - momentum_m["negative_days"],
        "range_edge_statistically_proven": False,
        "note": "same-window addition signal on sealed S5; statistical proof awaits M5 corpus",
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "combination_sha256": _canonical_sha(body)}
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
                "status": "COMBINATION_SEALED",
                "momentum_alone": momentum_m,
                "range_lane_alone": range_m,
                "momentum_plus_range": combined_m,
                "multiple_delta_22d": body["multiple_delta_22d"],
                "range_fills": range_fills,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
