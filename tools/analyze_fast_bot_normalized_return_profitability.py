#!/usr/bin/env python3
"""Walk-forward audit for causal normalized-return FX candidates.

At an hourly completed-M1 clock, the strategy normalizes the closed lookback
return by its realized one-minute path variation.  It then tests a fixed grid
of momentum and reversal orientations with executable next-minute bid/ask
entry and time close.  Candidate selection cannot inspect the 2026 holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "src", ROOT / "tools"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from analyze_fast_bot_session_clock_profitability import (  # noqa: E402
    EXTRA_COST_STRESS_PIPS,
    MAXIMUM_ENTRY_SPREAD_PIPS,
    TRAIN_END,
    VALIDATION_END,
    _canonical_sha,
    _holdout_admissible,
    _metrics,
    _pip_factor,
    _profit_factor_value,
)
from analyze_fast_bot_shock_guard_replay import _load  # noqa: E402


CONTRACT = "QR_FAST_BOT_NORMALIZED_RETURN_WALK_FORWARD_V1"
LOOKBACK_MINUTES = (15, 60, 240, 1440)
HOLDING_MINUTES = (15, 60, 240)
NORMALIZED_THRESHOLDS = (0.75, 1.25, 2.0)
ORIENTATIONS = ("MOMENTUM", "REVERSAL")


def _side_metrics(sides: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    return {
        side: {
            "trades": int(np.count_nonzero(sides == side)),
            "net_pips": round(float(values[sides == side].sum()), 6),
        }
        for side in ("LONG", "SHORT")
    }


def _pre_holdout_qualified(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    validation_sides: Mapping[str, Mapping[str, Any]],
) -> bool:
    return bool(
        int(train.get("trades") or 0) >= 250
        and int(validation.get("trades") or 0) >= 100
        and _profit_factor_value(train.get("profit_factor")) > 1.0
        and _profit_factor_value(validation.get("profit_factor")) > 1.0
        and float(train.get("net_pips") or 0.0) > 0.0
        and float(validation.get("net_pips") or 0.0) > 0.0
        and float(train.get("positive_year_rate") or 0.0) >= 0.75
        and float(validation.get("positive_year_rate") or 0.0) == 1.0
        and all(
            int(validation_sides[side].get("trades") or 0) >= 20
            for side in ("LONG", "SHORT")
        )
    )


def _deoverlap(
    decision_indices: np.ndarray,
    *,
    timestamps: np.ndarray,
    holding_minutes: int,
) -> np.ndarray:
    selected: list[int] = []
    occupied_until = -1
    for index in decision_indices:
        entry_time = int(timestamps[index + 1])
        if entry_time < occupied_until:
            continue
        selected.append(int(index))
        occupied_until = entry_time + holding_minutes * 60
    return np.array(selected, dtype=np.int64)


def _candidate_rows(
    pair: str,
    data: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    factor = _pip_factor(pair)
    timestamps = data["t"]
    mid = (data["bc"] + data["ac"]) / 2.0
    minute_returns = np.diff(mid, prepend=mid[0]) * factor
    squared_cumulative = np.cumsum(np.r_[0.0, minute_returns**2])
    candidates: list[dict[str, Any]] = []
    for lookback in LOOKBACK_MINUTES:
        decision_indices = np.flatnonzero(
            (timestamps % 3600) == 0
        )
        decision_indices = decision_indices[
            (decision_indices >= lookback) & (decision_indices + max(HOLDING_MINUTES) < len(timestamps))
        ]
        contiguous = (
            timestamps[decision_indices + max(HOLDING_MINUTES)]
            - timestamps[decision_indices - lookback]
            == (lookback + max(HOLDING_MINUTES)) * 60
        )
        decision_indices = decision_indices[contiguous]
        path_variation = np.sqrt(
            squared_cumulative[decision_indices + 1]
            - squared_cumulative[decision_indices - lookback + 1]
        )
        lookback_returns = (
            mid[decision_indices] - mid[decision_indices - lookback]
        ) * factor
        valid_scale = path_variation > 0.0
        decision_indices = decision_indices[valid_scale]
        lookback_returns = lookback_returns[valid_scale]
        path_variation = path_variation[valid_scale]
        normalized = np.abs(lookback_returns) / path_variation
        for threshold in NORMALIZED_THRESHOLDS:
            threshold_mask = normalized >= threshold
            threshold_indices = decision_indices[threshold_mask]
            threshold_returns = lookback_returns[threshold_mask]
            signal_by_index = {
                int(index): 1 if value > 0.0 else -1
                for index, value in zip(threshold_indices, threshold_returns)
                if value != 0.0
            }
            raw_indices = np.array(sorted(signal_by_index), dtype=np.int64)
            for holding_minutes in HOLDING_MINUTES:
                indices = _deoverlap(
                    raw_indices,
                    timestamps=timestamps,
                    holding_minutes=holding_minutes,
                )
                entries = indices + 1
                ends = entries + holding_minutes - 1
                entry_spreads = (data["ao"][entries] - data["bo"][entries]) * factor
                liquid = entry_spreads <= MAXIMUM_ENTRY_SPREAD_PIPS
                indices = indices[liquid]
                entries = entries[liquid]
                ends = ends[liquid]
                for orientation in ORIENTATIONS:
                    directions = np.array(
                        [signal_by_index[int(index)] for index in indices],
                        dtype=np.int8,
                    )
                    if orientation == "REVERSAL":
                        directions = -directions
                    values = np.where(
                        directions > 0,
                        (data["bc"][ends] - data["ao"][entries]) * factor,
                        (data["bo"][entries] - data["ac"][ends]) * factor,
                    )
                    sides = np.where(directions > 0, "LONG", "SHORT")
                    entry_times = timestamps[entries]
                    train_mask = entry_times < TRAIN_END
                    validation_mask = (entry_times >= TRAIN_END) & (
                        entry_times < VALIDATION_END
                    )
                    train = _metrics(entry_times[train_mask], values[train_mask])
                    validation = _metrics(
                        entry_times[validation_mask], values[validation_mask]
                    )
                    validation_sides = _side_metrics(
                        sides[validation_mask], values[validation_mask]
                    )
                    candidates.append(
                        {
                            "candidate_id": (
                                f"{pair}:{orientation}:LB_{lookback}M:"
                                f"Z_{threshold}:HOLD_{holding_minutes}M"
                            ),
                            "pair": pair,
                            "orientation": orientation,
                            "lookback_minutes": lookback,
                            "normalized_threshold": threshold,
                            "holding_minutes": holding_minutes,
                            "train": train,
                            "validation": validation,
                            "validation_by_side": validation_sides,
                            "pre_holdout_qualified": _pre_holdout_qualified(
                                train, validation, validation_sides
                            ),
                            "_holdout_timestamps": entry_times[
                                entry_times >= VALIDATION_END
                            ],
                            "_holdout_values": values[entry_times >= VALIDATION_END],
                            "_holdout_sides": sides[entry_times >= VALIDATION_END],
                        }
                    )
    return candidates


def analyze(inputs: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
    truth: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for pair, paths in sorted(inputs.items()):
        data = _load(list(paths))
        if not len(data["t"]):
            raise ValueError(f"no M1 truth for {pair}")
        candidates.extend(_candidate_rows(pair, data))
        truth.append(
            {
                "pair": pair,
                "rows": len(data["t"]),
                "from_utc": datetime.fromtimestamp(
                    int(data["t"][0]), tz=timezone.utc
                ).isoformat(),
                "to_utc": datetime.fromtimestamp(
                    int(data["t"][-1]), tz=timezone.utc
                ).isoformat(),
                "files": [
                    {
                        "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in paths
                ],
            }
        )
    qualified = [row for row in candidates if row["pre_holdout_qualified"]]
    chosen = (
        max(
            qualified,
            key=lambda row: (
                min(
                    _profit_factor_value(row["train"]["profit_factor"]),
                    _profit_factor_value(row["validation"]["profit_factor"]),
                ),
                float(row["validation"]["net_pips"]),
                float(row["train"]["net_pips"]),
                str(row["candidate_id"]),
            ),
        )
        if qualified
        else None
    )
    if chosen is None:
        selection: dict[str, Any] = {
            "status": "NO_PRE_HOLDOUT_CANDIDATE",
            "holdout_opened": False,
            "shadow_candidate_admitted": False,
        }
    else:
        holdout = _metrics(
            chosen["_holdout_timestamps"], chosen["_holdout_values"]
        )
        stressed = _metrics(
            chosen["_holdout_timestamps"],
            chosen["_holdout_values"],
            extra_cost_pips=EXTRA_COST_STRESS_PIPS,
        )
        holdout_sides = _side_metrics(
            chosen["_holdout_sides"], chosen["_holdout_values"]
        )
        admitted = _holdout_admissible(holdout, stressed) and all(
            int(holdout_sides[side]["trades"]) >= 10
            for side in ("LONG", "SHORT")
        )
        selection = {
            "status": "HOLDOUT_PASS" if admitted else "HOLDOUT_REJECT",
            "candidate_id": chosen["candidate_id"],
            "pair": chosen["pair"],
            "orientation": chosen["orientation"],
            "lookback_minutes": chosen["lookback_minutes"],
            "normalized_threshold": chosen["normalized_threshold"],
            "holding_minutes": chosen["holding_minutes"],
            "train": chosen["train"],
            "validation": chosen["validation"],
            "validation_by_side": chosen["validation_by_side"],
            "holdout": holdout,
            "holdout_by_side": holdout_sides,
            "holdout_extra_cost_stress": stressed,
            "holdout_opened": True,
            "selection_used_holdout": False,
            "shadow_candidate_admitted": admitted,
        }
    public_candidates = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in candidates
    ]
    body = {
        "contract": CONTRACT,
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "truth": truth,
        "protocol": {
            "candidate_family": "CAUSAL_NORMALIZED_RETURN_TIME_CLOSE",
            "decision_clock": "HOURLY_COMPLETED_M1",
            "entry_clock": "NEXT_CONTIGUOUS_M1_OPEN",
            "lookback_minutes": list(LOOKBACK_MINUTES),
            "holding_minutes": list(HOLDING_MINUTES),
            "normalized_thresholds": list(NORMALIZED_THRESHOLDS),
            "orientations": list(ORIENTATIONS),
            "maximum_entry_spread_pips": MAXIMUM_ENTRY_SPREAD_PIPS,
            "candidate_count": len(public_candidates),
            "same_candidate_overlap_policy": "FIRST_SIGNAL_UNTIL_TIME_CLOSE",
            "train_end_exclusive_utc": datetime.fromtimestamp(
                TRAIN_END, tz=timezone.utc
            ).isoformat(),
            "validation_end_exclusive_utc": datetime.fromtimestamp(
                VALIDATION_END, tz=timezone.utc
            ).isoformat(),
            "holdout_selection_use": "FORBIDDEN",
            "multiple_testing_corrected": False,
        },
        "pre_holdout_qualified_count": len(qualified),
        "pre_holdout_candidates": public_candidates,
        "selection": selection,
        "profitability_claim": "HISTORICAL_RESEARCH_ONLY_NOT_FORWARD_PROOF",
        "automatic_candidate_activation_allowed": False,
        "execution_authority": "NONE",
        "broker_http_methods_used": [],
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "live_permission": False,
        "promotion_allowed": False,
        "manual_tagless_policy": "NO_TOUCH",
    }
    return {**body, "contract_sha256": _canonical_sha(body)}


def _parse_input(value: str) -> tuple[str, Path]:
    pair, separator, raw_path = value.partition("=")
    if not separator or not pair or not raw_path:
        raise argparse.ArgumentTypeError("input must be PAIR=/absolute/path")
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("input path must be absolute")
    return pair.upper(), path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_parse_input, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs: dict[str, list[Path]] = {}
    for pair, path in args.input:
        inputs.setdefault(pair, []).append(path)
    result = analyze(inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["selection"]["status"],
                "pre_holdout_qualified_count": result[
                    "pre_holdout_qualified_count"
                ],
                "selection": result["selection"],
                "output": str(args.output),
                "contract_sha256": result["contract_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
