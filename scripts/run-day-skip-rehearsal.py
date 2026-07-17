#!/usr/bin/env python3
"""Rehearse the causal previous-day skip filter on the locked survivor.

Grid is predeclared: no filter, skip after >=1p known previous-day loss,
skip after >=50p known previous-day loss — each stacked on the already
chosen 50p intraday stop.  TRAIN selects (min negative days, subject to
retaining >=50% of the stacked baseline net, then shallowest worst day,
then net); VALIDATION replicates the one chosen filter unchanged.
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
    apply_causal_day_skip,
    daily_net_pips,
    negative_day_stats,
)
from quant_rabbit.fast_bot_historical_s5 import load_historical_s5_slice
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS

TRAIN_FROM = datetime(2026, 5, 12, tzinfo=timezone.utc)
TRAIN_TO = datetime(2026, 6, 15, tzinfo=timezone.utc)
VALIDATION_TO = datetime(2026, 6, 28, tzinfo=timezone.utc)
LOOKBACK = timedelta(minutes=725)
INTRADAY_STOP_PIPS = 50.0
DAY_SKIP_GRID = (None, 1.0, 50.0)
NET_RETENTION_FLOOR = 0.5


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stacked(outcomes, day_skip):
    filtered, skipped_day_trades = (
        apply_causal_day_skip(outcomes, prev_day_known_loss_pips=day_skip)
        if day_skip is not None
        else (tuple(outcomes), 0)
    )
    stopped, skipped_intraday = apply_causal_daily_stop(
        filtered, stop_pips=INTRADAY_STOP_PIPS
    )
    stats = negative_day_stats(daily_net_pips(stopped))
    return {
        "day_skip_pips": day_skip,
        "skipped_day_trades": skipped_day_trades,
        "skipped_intraday_trades": skipped_intraday,
        "stats": stats,
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

    train = evaluate_spec(
        series, spec=spec, opened_from_utc=TRAIN_FROM, opened_to_utc=TRAIN_TO,
        policy=policy,
    )
    rows = [_stacked(train, level) for level in DAY_SKIP_GRID]
    baseline = rows[0]
    eligible = [
        row
        for row in rows
        if row["day_skip_pips"] is not None
        and baseline["stats"]["net_pips"] > 0
        and row["stats"]["net_pips"]
        >= NET_RETENTION_FLOOR * baseline["stats"]["net_pips"]
    ]
    chosen = (
        min(
            eligible,
            key=lambda row: (
                row["stats"]["negative_days"],
                -row["stats"]["worst_day_pips"],
                -row["stats"]["net_pips"],
            ),
        )
        if eligible
        else None
    )
    chosen_level = chosen["day_skip_pips"] if chosen else None

    validation = evaluate_spec(
        series, spec=spec, opened_from_utc=TRAIN_TO, opened_to_utc=VALIDATION_TO,
        policy=policy,
    )
    validation_baseline = _stacked(validation, None)
    validation_replication = (
        _stacked(validation, chosen_level) if chosen_level is not None else None
    )

    body: dict[str, Any] = {
        "contract": "QR_DAY_SKIP_REHEARSAL_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "intraday_stop_pips": INTRADAY_STOP_PIPS,
        "predeclared_day_skip_grid": [level for level in DAY_SKIP_GRID],
        "selection_objective": (
            "MIN_NEGATIVE_DAYS_SUBJECT_TO_50PCT_NET_RETENTION_"
            "THEN_SHALLOWEST_WORST_DAY_THEN_NET_V1"
        ),
        "train_rows": rows,
        "chosen_day_skip_pips": chosen_level,
        "validation_baseline_with_intraday_stop": validation_baseline,
        "validation_replication": validation_replication,
        "validation_used_for_selection": False,
        "zero_negative_days_guaranteed": False,
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
    print(
        json.dumps(
            {
                "status": "DAY_SKIP_REHEARSAL_SEALED",
                "chosen_day_skip_pips": chosen_level,
                "train_rows": [
                    {
                        "day_skip_pips": row["day_skip_pips"],
                        "negative_days": row["stats"]["negative_days"],
                        "worst_day": row["stats"]["worst_day_pips"],
                        "net": row["stats"]["net_pips"],
                    }
                    for row in rows
                ],
                "validation_baseline": validation_baseline["stats"],
                "validation_replication": (
                    validation_replication["stats"]
                    if validation_replication
                    else None
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
