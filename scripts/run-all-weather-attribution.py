#!/usr/bin/env python3
"""All-weather P&L attribution: which regime x vol cells does the edge live in?

Runs the locked survivor over TRAIN+VALIDATION, measures each traded pair's
regime and volatility at the trade's decision time (closed minute closes
only, no lookahead), and attributes realized pips to each regime x vol
cell.  This is the honest all-weather scorecard: it shows, with data,
which weather the current single lane actually takes money in and which
cells are empty.  Shadow/diagnostic only.
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
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS
from quant_rabbit.regime_classifier_shadow import (
    VOL_HISTORY_WINDOW,
    RegimeClassifierError,
    classify_regime,
)

TRAIN_FROM = datetime(2026, 5, 12, tzinfo=timezone.utc)
VALIDATION_TO = datetime(2026, 6, 28, tzinfo=timezone.utc)
LOOKBACK = timedelta(minutes=725)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("attribution output must be clean; refusing stale reuse")
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
            {"time": point.minute_utc, "close": point.mid_close}
            for point in prepared.points
        ]
        del item
        gc.collect()

    outcomes = evaluate_spec(
        series, spec=spec, opened_from_utc=TRAIN_FROM, opened_to_utc=VALIDATION_TO,
        policy=policy,
    )

    cells: dict[str, dict[str, Any]] = {}
    unclassified = 0
    for row in outcomes:
        candles = [c for c in minute_index[row.pair] if c["time"] < row.decision_utc]
        if len(candles) < VOL_HISTORY_WINDOW:
            unclassified += 1
            continue
        try:
            classification = classify_regime(
                candles[-VOL_HISTORY_WINDOW - 5:], as_of_utc=row.decision_utc
            )
        except RegimeClassifierError:
            unclassified += 1
            continue
        key = f"{classification['regime']}_{classification['vol_state']}"
        cell = cells.setdefault(
            key,
            {
                "regime": classification["regime"],
                "vol_state": classification["vol_state"],
                "trade_count": 0,
                "net_pips": 0.0,
                "win_count": 0,
                "loss_count": 0,
            },
        )
        cell["trade_count"] += 1
        cell["net_pips"] += row.realized_pips
        cell["win_count"] += int(row.realized_pips > 0)
        cell["loss_count"] += int(row.realized_pips < 0)

    cell_rows = []
    for key in sorted(cells):
        cell = cells[key]
        cell_rows.append(
            {
                **cell,
                "net_pips": round(cell["net_pips"], 9),
                "mean_pips_per_trade": round(
                    cell["net_pips"] / cell["trade_count"], 9
                )
                if cell["trade_count"]
                else 0.0,
                "profitable_cell": cell["net_pips"] > 0.0,
            }
        )

    positive_cells = [c for c in cell_rows if c["profitable_cell"]]
    body: dict[str, Any] = {
        "contract": "QR_ALL_WEATHER_ATTRIBUTION_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "window": [TRAIN_FROM.isoformat(), VALIDATION_TO.isoformat()],
        "total_trades": len(outcomes),
        "unclassified_trades": unclassified,
        "cell_rows": cell_rows,
        "profitable_cell_count": len(positive_cells),
        "profitable_cells": sorted(
            f"{c['regime']}_{c['vol_state']}" for c in positive_cells
        ),
        "all_weather_covered_by_this_lane": len(positive_cells) >= 6,
        "reading": (
            "the single survivor lane concentrates its edge in specific "
            "regime x vol cells; empty/negative cells are the all-weather gap "
            "that new families (range Lane F, etc.) must fill"
        ),
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "attribution_sha256": _canonical_sha(body)}
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
                "status": "ATTRIBUTION_SEALED",
                "profitable_cells": body["profitable_cells"],
                "cells": {
                    f"{c['regime']}_{c['vol_state']}": {
                        "trades": c["trade_count"],
                        "net": c["net_pips"],
                        "mean": c["mean_pips_per_trade"],
                    }
                    for c in cell_rows
                },
                "unclassified": unclassified,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
