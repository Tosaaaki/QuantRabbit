#!/usr/bin/env python3
"""Replay the locked survivor with the causal daily-loss overlay.

A realistic rehearsal on already-opened windows: TRAIN picks one stop level
from the predeclared grid, VALIDATION replicates that single choice
unchanged.  The sealed report shows how many negative days the overlay
removes and at what net cost — measured, not promised.  Shadow only.
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
    OVERLAY_POLICY,
    apply_causal_daily_stop,
    choose_stop_on_train,
    daily_net_pips,
    negative_day_stats,
)
from quant_rabbit.fast_bot_historical_s5 import load_historical_s5_slice
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS

TRAIN_FROM = datetime(2026, 5, 12, tzinfo=timezone.utc)
TRAIN_TO = datetime(2026, 6, 15, tzinfo=timezone.utc)
VALIDATION_TO = datetime(2026, 6, 28, tzinfo=timezone.utc)
LOOKBACK = timedelta(minutes=725)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_series(manifest: dict, from_utc: datetime, to_utc: datetime) -> dict:
    series = {}
    for pair in DEFAULT_TRADER_PAIRS:
        item = load_historical_s5_slice(
            manifest, pair=pair, time_from=from_utc, time_to=to_utc
        )
        series[pair] = prepare_exact_s5_series(item.candles)
        del item
        gc.collect()
    return series


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

    series = _load_series(manifest, TRAIN_FROM - LOOKBACK, VALIDATION_TO)
    train_outcomes = evaluate_spec(
        series,
        spec=spec,
        opened_from_utc=TRAIN_FROM,
        opened_to_utc=TRAIN_TO,
        policy=policy,
    )
    selection = choose_stop_on_train(train_outcomes)

    validation_outcomes = evaluate_spec(
        series,
        spec=spec,
        opened_from_utc=TRAIN_TO,
        opened_to_utc=VALIDATION_TO,
        policy=policy,
    )
    validation_baseline = negative_day_stats(daily_net_pips(validation_outcomes))
    chosen = selection["chosen_stop_pips"]
    if chosen is not None:
        kept, skipped = apply_causal_daily_stop(
            validation_outcomes, stop_pips=float(chosen)
        )
        validation_overlay = {
            "stop_pips": chosen,
            "skipped_trades": skipped,
            "stats": negative_day_stats(daily_net_pips(kept)),
        }
    else:
        validation_overlay = None

    body: dict[str, Any] = {
        "contract": "QR_NEGATIVE_DAY_SUPPRESSION_REHEARSAL_V1",
        "schema_version": 1,
        "overlay_policy": OVERLAY_POLICY,
        "lock_sha256": lock["lock_sha256"],
        "spec_id": spec.spec_id,
        "train_selection": selection,
        "validation_baseline": validation_baseline,
        "validation_overlay_replication": validation_overlay,
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
                "status": "REHEARSAL_SEALED",
                "chosen_stop_pips": chosen,
                "train_baseline_negative_days": selection["train_baseline"][
                    "negative_days"
                ],
                "validation_baseline": validation_baseline,
                "validation_overlay": validation_overlay,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
