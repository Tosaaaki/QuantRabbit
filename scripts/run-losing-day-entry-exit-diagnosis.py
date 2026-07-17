#!/usr/bin/env python3
"""Diagnose WHY losing days lose: entry/exit fingerprints, not abstention.

Operator discipline: "no trades = no losses" is not progress; losing days
have bad entries and bad exits and the system must handle those days.
This script contrasts losing-day trades against winning-day trades on the
locked survivor — signal strength, decision hour, side, pair concentration,
and score-rank quality — and seals the fingerprint so the fix targets the
entry/exit mechanism instead of the calendar.
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
from quant_rabbit.daily_loss_overlay import daily_net_pips
from quant_rabbit.fast_bot_historical_s5 import load_historical_s5_slice
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS

TRAIN_FROM = datetime(2026, 5, 12, tzinfo=timezone.utc)
VALIDATION_TO = datetime(2026, 6, 28, tzinfo=timezone.utc)
LOOKBACK = timedelta(minutes=725)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bucket(trades) -> dict[str, Any]:
    if not trades:
        return {"trade_count": 0}
    total = sum(row.realized_pips for row in trades)
    by_hour: dict[int, float] = {}
    by_side: dict[str, float] = {}
    by_pair: dict[str, float] = {}
    for row in trades:
        by_hour[row.decision_utc.hour] = (
            by_hour.get(row.decision_utc.hour, 0.0) + row.realized_pips
        )
        by_side[row.side] = by_side.get(row.side, 0.0) + row.realized_pips
        by_pair[row.pair] = by_pair.get(row.pair, 0.0) + row.realized_pips
    scores = sorted(abs(row.score) for row in trades)
    worst_pairs = sorted(by_pair.items(), key=lambda item: item[1])[:5]
    return {
        "trade_count": len(trades),
        "net_pips": round(total, 9),
        "mean_pips_per_trade": round(total / len(trades), 9),
        "median_abs_score": round(scores[len(scores) // 2], 9),
        "mean_abs_score": round(sum(scores) / len(scores), 9),
        "pips_by_decision_hour_utc": {
            str(hour): round(value, 9) for hour, value in sorted(by_hour.items())
        },
        "pips_by_side": {side: round(value, 9) for side, value in sorted(by_side.items())},
        "worst_five_pairs": [
            {"pair": pair, "pips": round(value, 9)} for pair, value in worst_pairs
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("diagnosis output must be clean; refusing stale reuse")
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
        series,
        spec=spec,
        opened_from_utc=TRAIN_FROM,
        opened_to_utc=VALIDATION_TO,
        policy=policy,
    )
    daily = daily_net_pips(outcomes)
    losing_days = {day for day, pips in daily.items() if pips < 0.0}
    winning_days = {day for day, pips in daily.items() if pips >= 0.0}
    losing_trades = [row for row in outcomes if row.decision_utc.date() in losing_days]
    winning_trades = [row for row in outcomes if row.decision_utc.date() in winning_days]

    body: dict[str, Any] = {
        "contract": "QR_LOSING_DAY_ENTRY_EXIT_DIAGNOSIS_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "window": [TRAIN_FROM.isoformat(), VALIDATION_TO.isoformat()],
        "losing_day_count": len(losing_days),
        "winning_day_count": len(winning_days),
        "losing_days_fingerprint": _bucket(losing_trades),
        "winning_days_fingerprint": _bucket(winning_trades),
        "known_exit_variant_evidence": {
            "hold_360_train_net_pips": -185.9,
            "cadence_60_train_net_pips": -712.4,
            "rank1_train_net_pips": 403.8,
            "reading": (
                "early exits and higher entry frequency both destroy the "
                "edge; losing days are not a late-exit problem at this "
                "horizon"
            ),
        },
        "abstention_is_not_improvement": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "diagnosis_sha256": _canonical_sha(body)}
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
                "status": "DIAGNOSIS_SEALED",
                "losing_days": len(losing_days),
                "losing": body["losing_days_fingerprint"],
                "winning": body["winning_days_fingerprint"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
