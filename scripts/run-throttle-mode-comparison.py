#!/usr/bin/env python3
"""Compare SKIP vs HALF_SIZE intraday throttles on the locked survivor.

Same evaluation, three arms per window: baseline, 50p SKIP, 50p HALF_SIZE.
For each throttle the report seals the measured opportunity cost — the
summed realized pips of the trades it suppressed (SKIP) or halved
(HALF_SIZE).  Predeclared demotion rule: a throttle whose suppressed-trade
total is positive forfeited real profit and is demoted.
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
from quant_rabbit.daily_loss_overlay import (
    apply_causal_daily_stop,
    daily_net_pips,
    negative_day_stats,
)
from quant_rabbit.fast_bot_historical_s5 import load_historical_s5_slice
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS

TRAIN_FROM = datetime(2026, 5, 12, tzinfo=timezone.utc)
TRAIN_TO = datetime(2026, 6, 15, tzinfo=timezone.utc)
VALIDATION_TO = datetime(2026, 6, 28, tzinfo=timezone.utc)
LOOKBACK = timedelta(minutes=725)
STOP_PIPS = 50.0


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _arm(outcomes, mode: str | None) -> dict[str, Any]:
    if mode is None:
        kept, affected = tuple(outcomes), 0
        opportunity = 0.0
    else:
        kept, affected = apply_causal_daily_stop(
            outcomes, stop_pips=STOP_PIPS, mode=mode
        )
        base_net = sum(row.realized_pips for row in outcomes)
        kept_net = sum(row.realized_pips for row in kept)
        # What the throttle gave up: positive means real profit forfeited.
        opportunity = round(base_net - kept_net, 9)
    return {
        "mode": mode or "BASELINE",
        "affected_trades": affected,
        "suppressed_pips_total": opportunity,
        "opportunity_cost_positive": opportunity > 0,
        "stats": negative_day_stats(daily_net_pips(kept)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("comparison output must be clean; refusing stale reuse")
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
            series,
            spec=spec,
            opened_from_utc=opened_from,
            opened_to_utc=opened_to,
            policy=policy,
        )
        windows[label] = [_arm(outcomes, mode) for mode in (None, "SKIP", "HALF_SIZE")]

    body: dict[str, Any] = {
        "contract": "QR_THROTTLE_MODE_COMPARISON_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "stop_pips": STOP_PIPS,
        "demotion_rule": "SUPPRESSED_PIPS_TOTAL_POSITIVE_MEANS_FORFEITED_PROFIT_V1",
        "windows": windows,
        "zero_negative_days_guaranteed": False,
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "comparison_sha256": _canonical_sha(body)}
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
