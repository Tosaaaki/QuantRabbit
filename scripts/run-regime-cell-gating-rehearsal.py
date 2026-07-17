#!/usr/bin/env python3
"""Regime-cell gating with strict TRAIN-select / VALIDATION-replicate.

The attribution showed the survivor bleeds in specific regime x vol cells.
The naive move — drop those cells — is exactly the overfitting trap the
time-of-day filter fell into.  This rehearsal does it honestly: TRAIN alone
picks the losing cells (net < 0), VALIDATION replicates that FIXED cell set
unchanged, and we measure net and negative days on both.  Cells are
measured causally (closed candles before each decision).  The future window
remains the only clean arbiter.  Shadow only.
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
from quant_rabbit.regime_classifier_shadow import (
    VOL_HISTORY_WINDOW,
    RegimeClassifierError,
    classify_regime,
)

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


def _cell_of(row, minute_index) -> str | None:
    candles = [c for c in minute_index[row.pair] if c["time"] < row.decision_utc]
    if len(candles) < VOL_HISTORY_WINDOW:
        return None
    try:
        c = classify_regime(candles[-VOL_HISTORY_WINDOW - 5:], as_of_utc=row.decision_utc)
    except RegimeClassifierError:
        return None
    return f"{c['regime']}_{c['vol_state']}"


def _cell_net(outcomes, minute_index) -> dict[str, float]:
    nets: dict[str, float] = {}
    for row in outcomes:
        cell = _cell_of(row, minute_index)
        if cell is None:
            continue
        nets[cell] = nets.get(cell, 0.0) + row.realized_pips
    return nets


def _apply_gate(outcomes, minute_index, excluded_cells):
    kept = [
        row
        for row in outcomes
        if _cell_of(row, minute_index) not in excluded_cells
    ]
    stopped, _ = apply_causal_daily_stop(kept, stop_pips=STOP_PIPS)
    return negative_day_stats(daily_net_pips(stopped))


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

    train = evaluate_spec(
        series, spec=spec, opened_from_utc=TRAIN_FROM, opened_to_utc=TRAIN_TO,
        policy=policy,
    )
    validation = evaluate_spec(
        series, spec=spec, opened_from_utc=TRAIN_TO, opened_to_utc=VALIDATION_TO,
        policy=policy,
    )

    train_cell_net = _cell_net(train, minute_index)
    # Losing cells are chosen from TRAIN ONLY.
    excluded_cells = sorted(cell for cell, net in train_cell_net.items() if net < 0.0)

    body: dict[str, Any] = {
        "contract": "QR_REGIME_CELL_GATING_REHEARSAL_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "stop_pips": STOP_PIPS,
        "train_cell_net_pips": {k: round(v, 9) for k, v in sorted(train_cell_net.items())},
        "train_selected_excluded_cells": excluded_cells,
        "train_baseline": _apply_gate(train, minute_index, set()),
        "train_gated": _apply_gate(train, minute_index, set(excluded_cells)),
        "validation_baseline": _apply_gate(validation, minute_index, set()),
        "validation_gated_replication": _apply_gate(
            validation, minute_index, set(excluded_cells)
        ),
        "cells_selected_on_train_only": True,
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
                "status": "CELL_GATING_SEALED",
                "excluded_cells": excluded_cells,
                "train": {"baseline": body["train_baseline"], "gated": body["train_gated"]},
                "validation": {
                    "baseline": body["validation_baseline"],
                    "gated": body["validation_gated_replication"],
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
