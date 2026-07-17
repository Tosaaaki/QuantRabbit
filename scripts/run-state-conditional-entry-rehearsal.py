#!/usr/bin/env python3
"""State-conditional entry repairs — no clock heuristics.

Operator principle: dividing by time-of-day is a human heuristic; the
machine conditions on measurable state.  Arms use only decision-time state:
S1 requires per-trade |score| >= 30 (signal-strength state), S2 requires
the actually-paid round-trip spread to stay under 15% of |score|
(cost-to-signal state), S3 stacks both.  Every arm then applies the adopted
50p SKIP stop.  DISCLOSURE: motivated by a fingerprint computed on combined
windows — VALIDATION here is not clean evidence; the future window decides.
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
MIN_ABS_SCORE = 30.0
MAX_SPREAD_TO_SCORE = 0.15
ARMS = ("BASELINE", "S1_MIN_SIGNAL", "S2_COST_TO_SIGNAL", "S3_BOTH")


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _entry_spread_pips(row) -> float:
    # Paid entry-side spread, knowable at decision from the quoted book.
    return abs(row.round_trip_spread_pips)


def _admit(row, arm: str) -> bool:
    if arm in {"S1_MIN_SIGNAL", "S3_BOTH"} and abs(row.score) < MIN_ABS_SCORE:
        return False
    if arm in {"S2_COST_TO_SIGNAL", "S3_BOTH"}:
        score = abs(row.score)
        if score <= 0 or _entry_spread_pips(row) / score > MAX_SPREAD_TO_SCORE:
            return False
    return True


def _arm_report(outcomes, arm: str) -> dict[str, Any]:
    rows = [row for row in outcomes if _admit(row, arm)]
    stopped, stop_affected = apply_causal_daily_stop(rows, stop_pips=STOP_PIPS)
    stats = negative_day_stats(daily_net_pips(stopped))
    return {
        "arm": arm,
        "entry_filtered_trades": len(list(outcomes)) - len(rows),
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
        "contract": "QR_STATE_CONDITIONAL_ENTRY_REHEARSAL_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "min_abs_score": MIN_ABS_SCORE,
        "max_spread_to_score": MAX_SPREAD_TO_SCORE,
        "stop_pips": STOP_PIPS,
        "time_of_day_heuristics_used": False,
        "operator_principle": (
            "condition on measurable state, never on the clock; states where "
            "this method loses route to a different family, not to rest"
        ),
        "selection_rule": "TRAIN_NET_MUST_BEAT_BASELINE_THEN_MAX_NET_V1",
        "windows": windows,
        "chosen_arm": chosen,
        "fingerprint_used_combined_windows": True,
        "validation_replication_is_clean": False,
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
    print(
        json.dumps(
            {
                "status": "STATE_ENTRY_REHEARSAL_SEALED",
                "chosen_arm": chosen,
                "train": {
                    row["arm"]: {
                        "net": row["stats"]["net_pips"],
                        "neg_days": row["stats"]["negative_days"],
                        "worst": row["stats"]["worst_day_pips"],
                        "filtered": row["entry_filtered_trades"],
                    }
                    for row in windows["TRAIN"]
                },
                "validation": {
                    row["arm"]: {
                        "net": row["stats"]["net_pips"],
                        "neg_days": row["stats"]["negative_days"],
                        "worst": row["stats"]["worst_day_pips"],
                        "filtered": row["entry_filtered_trades"],
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
