#!/usr/bin/env python3
"""Entry-quality repairs from the losing-day fingerprint, on the proof ladder.

Repairs enter better instead of resting: R1 drops the structurally negative
NY-afternoon decision slots (16/20 UTC), R2 requires per-trade |score| >= 30,
R3 stacks both; every arm then applies the adopted 50p SKIP stop.  All
filters read only decision-time information.  DISCLOSURE: the fingerprint
that motivated this grid was computed on TRAIN+VALIDATION combined, so the
VALIDATION replication here is NOT clean evidence — the future window is
the only untouched arbiter for these arms.
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
DROPPED_HOURS = frozenset({16, 20})
MIN_ABS_SCORE = 30.0
ARMS = ("BASELINE", "R1_DROP_NY_AFTERNOON", "R2_MIN_ABS_SCORE", "R3_BOTH")


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _filtered(outcomes, arm: str):
    rows = list(outcomes)
    if arm in {"R1_DROP_NY_AFTERNOON", "R3_BOTH"}:
        rows = [row for row in rows if row.decision_utc.hour not in DROPPED_HOURS]
    if arm in {"R2_MIN_ABS_SCORE", "R3_BOTH"}:
        rows = [row for row in rows if abs(row.score) >= MIN_ABS_SCORE]
    stopped, affected = apply_causal_daily_stop(rows, stop_pips=STOP_PIPS)
    return stopped, len(list(outcomes)) - len(rows), affected


def _arm_report(outcomes, arm: str) -> dict[str, Any]:
    stopped, entry_filtered, stop_affected = _filtered(outcomes, arm)
    stats = negative_day_stats(daily_net_pips(stopped))
    return {
        "arm": arm,
        "entry_filtered_trades": entry_filtered,
        "stop_affected_trades": stop_affected,
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
        windows[label] = [_arm_report(outcomes, arm) for arm in ARMS]

    train_rows = {row["arm"]: row for row in windows["TRAIN"]}
    baseline_net = train_rows["BASELINE"]["stats"]["net_pips"]
    eligible = [
        row
        for arm, row in train_rows.items()
        if arm != "BASELINE" and row["stats"]["net_pips"] >= baseline_net
    ]
    chosen = (
        max(eligible, key=lambda row: row["stats"]["net_pips"])["arm"]
        if eligible
        else None
    )

    body: dict[str, Any] = {
        "contract": "QR_ENTRY_QUALITY_REPAIR_REHEARSAL_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "dropped_decision_hours_utc": sorted(DROPPED_HOURS),
        "min_abs_score": MIN_ABS_SCORE,
        "stop_pips": STOP_PIPS,
        "selection_rule": "TRAIN_NET_MUST_BEAT_BASELINE_THEN_MAX_NET_V1",
        "windows": windows,
        "chosen_arm": chosen,
        "fingerprint_used_combined_windows": True,
        "validation_replication_is_clean": False,
        "future_window_is_final_arbiter": True,
        "abstention_is_not_improvement_ack": (
            "filters select better entries every day; they do not rest on "
            "loss-triggered calendars"
        ),
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
                "status": "ENTRY_REPAIR_SEALED",
                "chosen_arm": chosen,
                "train": {
                    row["arm"]: {
                        "net": row["stats"]["net_pips"],
                        "neg_days": row["stats"]["negative_days"],
                        "worst": row["stats"]["worst_day_pips"],
                    }
                    for row in windows["TRAIN"]
                },
                "validation": {
                    row["arm"]: {
                        "net": row["stats"]["net_pips"],
                        "neg_days": row["stats"]["negative_days"],
                        "worst": row["stats"]["worst_day_pips"],
                    }
                    for row in windows["VALIDATION"]
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
