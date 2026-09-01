#!/usr/bin/env python3
"""Walk-forward audit for causal normalized-return passive-limit candidates.

Signals use only completed M1 history.  A fixed limit inside the observed
bid/ask spread activates on the next minute, fills only when executable M1
bid/ask truth touches it, and exits at an executable time close.  Candidate
selection cannot inspect the 2026 holdout and the tool has zero order authority.
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

from analyze_fast_bot_normalized_return_profitability import (  # noqa: E402
    NORMALIZED_THRESHOLDS,
    ORIENTATIONS,
)
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


CONTRACT = "QR_FAST_BOT_NORMALIZED_PASSIVE_WALK_FORWARD_V3"
HOLDING_MINUTES = (15, 60, 240)
ENTRY_TTL_MINUTES = (5, 15)
ENTRY_SPREAD_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
CANDIDATE_SIDES = ("LONG", "SHORT")
LOOKBACK_CONFIRMATION_PAIRS = (
    (15, 60),
    (15, 240),
    (15, 1440),
    (60, 240),
    (60, 1440),
    (240, 1440),
)


def _pre_holdout_qualified(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> bool:
    return bool(
        int(train.get("trades") or 0) >= 125
        and int(validation.get("trades") or 0) >= 50
        and _profit_factor_value(train.get("profit_factor")) > 1.0
        and _profit_factor_value(validation.get("profit_factor")) > 1.0
        and float(train.get("net_pips") or 0.0) > 0.0
        and float(validation.get("net_pips") or 0.0) > 0.0
        and float(train.get("positive_year_rate") or 0.0) >= 0.75
        and float(validation.get("positive_year_rate") or 0.0) == 1.0
    )


def _normalized_signals(
    data: Mapping[str, np.ndarray],
    *,
    pair: str,
    lookback_minutes: int,
    confirmation_lookback_minutes: int,
    normalized_threshold: float,
) -> tuple[np.ndarray, dict[int, int]]:
    factor = _pip_factor(pair)
    timestamps = data["t"]
    mid = (data["bc"] + data["ac"]) / 2.0
    minute_returns = np.diff(mid, prepend=mid[0]) * factor
    squared_cumulative = np.cumsum(np.r_[0.0, minute_returns**2])
    indices = np.flatnonzero((timestamps % 3600) == 0)
    maximum_future = max(ENTRY_TTL_MINUTES) + max(HOLDING_MINUTES)
    history_minutes = max(lookback_minutes, confirmation_lookback_minutes)
    indices = indices[
        (indices >= history_minutes)
        & (indices + maximum_future < len(timestamps))
    ]
    contiguous = (
        timestamps[indices + maximum_future] - timestamps[indices - history_minutes]
        == (history_minutes + maximum_future) * 60
    )
    indices = indices[contiguous]
    path_variation = np.sqrt(
        squared_cumulative[indices + 1]
        - squared_cumulative[indices - lookback_minutes + 1]
    )
    lookback_returns = (
        mid[indices] - mid[indices - lookback_minutes]
    ) * factor
    confirmation_returns = (
        mid[indices] - mid[indices - confirmation_lookback_minutes]
    ) * factor
    valid = (path_variation > 0.0) & (
        np.abs(lookback_returns) / np.maximum(path_variation, 1e-12)
        >= normalized_threshold
    ) & (lookback_returns * confirmation_returns > 0.0)
    indices = indices[valid]
    lookback_returns = lookback_returns[valid]
    directions = {
        int(index): 1 if value > 0.0 else -1
        for index, value in zip(indices, lookback_returns)
        if value != 0.0
    }
    return np.array(sorted(directions), dtype=np.int64), directions


def _evaluate_candidate(
    data: Mapping[str, np.ndarray],
    *,
    pair: str,
    decision_indices: np.ndarray,
    source_directions: Mapping[int, int],
    orientation: str,
    entry_spread_fraction: float,
    entry_ttl_minutes: int,
    holding_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor = _pip_factor(pair)
    timestamps = data["t"]
    result_times: list[int] = []
    result_values: list[float] = []
    result_sides: list[str] = []
    occupied_until = -1
    tick = 1.0 / (factor * 10)
    for decision in decision_indices:
        activation = int(decision) + 1
        activation_time = int(timestamps[activation])
        if activation_time < occupied_until:
            continue
        source_direction = source_directions[int(decision)]
        direction = (
            source_direction if orientation == "MOMENTUM" else -source_direction
        )
        bid = float(data["bc"][decision])
        ask = float(data["ac"][decision])
        spread_pips = (ask - bid) * factor
        if spread_pips > MAXIMUM_ENTRY_SPREAD_PIPS or not bid < ask:
            continue
        occupied_until = activation_time + (
            entry_ttl_minutes + holding_minutes
        ) * 60
        width = ask - bid
        limit = (
            bid + width * entry_spread_fraction
            if direction > 0
            else ask - width * entry_spread_fraction
        )
        limit = (
            math.floor(limit / tick + 1e-9) * tick
            if direction > 0
            else math.ceil(limit / tick - 1e-9) * tick
        )
        fill_end = activation + entry_ttl_minutes
        if fill_end > len(timestamps):
            continue
        fill_slice = slice(activation, fill_end)
        touch = (
            data["al"][fill_slice] <= limit
            if direction > 0
            else data["bh"][fill_slice] >= limit
        )
        touched = np.flatnonzero(touch)
        if not len(touched):
            continue
        fill_index = activation + int(touched[0])
        exit_index = fill_index + holding_minutes - 1
        if exit_index >= len(timestamps) or (
            int(timestamps[exit_index]) - int(timestamps[fill_index])
            != (holding_minutes - 1) * 60
        ):
            continue
        realized = (
            (float(data["bc"][exit_index]) - limit) * factor
            if direction > 0
            else (limit - float(data["ac"][exit_index])) * factor
        )
        result_times.append(activation_time)
        result_values.append(realized)
        result_sides.append("LONG" if direction > 0 else "SHORT")
    return (
        np.array(result_times, dtype=np.int64),
        np.array(result_values, dtype=np.float64),
        np.array(result_sides),
    )


def _candidate_rows(
    pair: str,
    data: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for lookback, confirmation_lookback in LOOKBACK_CONFIRMATION_PAIRS:
        for threshold in NORMALIZED_THRESHOLDS:
            indices, directions = _normalized_signals(
                data,
                pair=pair,
                lookback_minutes=lookback,
                confirmation_lookback_minutes=confirmation_lookback,
                normalized_threshold=threshold,
            )
            for orientation in ORIENTATIONS:
                for fraction in ENTRY_SPREAD_FRACTIONS:
                    for ttl in ENTRY_TTL_MINUTES:
                        for hold in HOLDING_MINUTES:
                            times, values, sides = _evaluate_candidate(
                                data,
                                pair=pair,
                                decision_indices=indices,
                                source_directions=directions,
                                orientation=orientation,
                                entry_spread_fraction=fraction,
                                entry_ttl_minutes=ttl,
                                holding_minutes=hold,
                            )
                            for candidate_side in CANDIDATE_SIDES:
                                side_mask = sides == candidate_side
                                side_times = times[side_mask]
                                side_values = values[side_mask]
                                train_mask = side_times < TRAIN_END
                                validation_mask = (side_times >= TRAIN_END) & (
                                    side_times < VALIDATION_END
                                )
                                train = _metrics(
                                    side_times[train_mask],
                                    side_values[train_mask],
                                )
                                validation = _metrics(
                                    side_times[validation_mask],
                                    side_values[validation_mask],
                                )
                                candidates.append(
                                    {
                                        "candidate_id": (
                                            f"{pair}:{candidate_side}:"
                                            f"{orientation}:LB_{lookback}M:"
                                            f"CLB_{confirmation_lookback}M:"
                                            f"Z_{threshold}:F_{fraction}:"
                                            f"TTL_{ttl}M:HOLD_{hold}M"
                                        ),
                                        "pair": pair,
                                        "candidate_side": candidate_side,
                                        "orientation": orientation,
                                        "lookback_minutes": lookback,
                                        "confirmation_lookback_minutes": (
                                            confirmation_lookback
                                        ),
                                        "normalized_threshold": threshold,
                                        "entry_spread_fraction": fraction,
                                        "entry_ttl_minutes": ttl,
                                        "holding_minutes": hold,
                                        "train": train,
                                        "validation": validation,
                                        "pre_holdout_qualified": (
                                            _pre_holdout_qualified(
                                                train,
                                                validation,
                                            )
                                        ),
                                        "_holdout_timestamps": side_times[
                                            side_times >= VALIDATION_END
                                        ],
                                        "_holdout_values": side_values[
                                            side_times >= VALIDATION_END
                                        ],
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
        admitted = _holdout_admissible(holdout, stressed)
        selection = {
            "status": "HOLDOUT_PASS" if admitted else "HOLDOUT_REJECT",
            "candidate_id": chosen["candidate_id"],
            "pair": chosen["pair"],
            "candidate_side": chosen["candidate_side"],
            "orientation": chosen["orientation"],
            "lookback_minutes": chosen["lookback_minutes"],
            "confirmation_lookback_minutes": chosen[
                "confirmation_lookback_minutes"
            ],
            "normalized_threshold": chosen["normalized_threshold"],
            "entry_spread_fraction": chosen["entry_spread_fraction"],
            "entry_ttl_minutes": chosen["entry_ttl_minutes"],
            "holding_minutes": chosen["holding_minutes"],
            "train": chosen["train"],
            "validation": chosen["validation"],
            "holdout": holdout,
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
            "candidate_family": "CAUSAL_NORMALIZED_RETURN_PASSIVE_LIMIT_TIME_CLOSE",
            "decision_clock": "HOURLY_COMPLETED_M1",
            "activation_clock": "NEXT_CONTIGUOUS_M1_OPEN",
            "lookback_confirmation_pairs": [
                list(item) for item in LOOKBACK_CONFIRMATION_PAIRS
            ],
            "confirmation_policy": "SAME_SIGN_COMPLETED_M1_RETURN",
            "normalized_thresholds": list(NORMALIZED_THRESHOLDS),
            "orientations": list(ORIENTATIONS),
            "candidate_sides": list(CANDIDATE_SIDES),
            "entry_spread_fractions": list(ENTRY_SPREAD_FRACTIONS),
            "entry_ttl_minutes": list(ENTRY_TTL_MINUTES),
            "holding_minutes": list(HOLDING_MINUTES),
            "maximum_decision_spread_pips": MAXIMUM_ENTRY_SPREAD_PIPS,
            "candidate_count": len(public_candidates),
            "same_candidate_overlap_policy": (
                "RESERVE_ACTIVATION_THROUGH_MAXIMUM_TTL_PLUS_HOLD"
            ),
            "gap_through_fill_policy": "CONSERVATIVE_LIMIT_PRICE_NO_IMPROVEMENT",
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
