#!/usr/bin/env python3
"""Walk-forward audit for fixed UTC session-clock FX candidates.

The candidate clock, sides, and holding periods are fixed before the 2026
holdout is opened.  Every return uses executable bid/ask prices and requires a
contiguous M1 path.  The tool is local research only and has no broker client,
order path, policy write, or activation authority.
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

from analyze_fast_bot_shock_guard_replay import _load  # noqa: E402


CONTRACT = "QR_FAST_BOT_SESSION_CLOCK_WALK_FORWARD_V1"
TRAIN_END = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
VALIDATION_END = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
HOLDING_MINUTES = (15, 30, 60, 120, 240)
SIDES = ("LONG", "SHORT")
MAXIMUM_ENTRY_SPREAD_PIPS = 1.5
EXTRA_COST_STRESS_PIPS = 0.2


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("_JPY") else 10_000


def _profit_factor(values: np.ndarray) -> float | str | None:
    gross_profit = float(values[values > 0.0].sum())
    gross_loss = -float(values[values < 0.0].sum())
    if gross_loss > 0.0:
        return round(gross_profit / gross_loss, 6)
    if gross_profit > 0.0:
        return "INF"
    return None


def _pessimistic_expectancy(values: np.ndarray) -> float | None:
    if not len(values):
        return None
    wins = values[values > 0.0]
    losses = -values[values < 0.0]
    observed = len(wins) / len(values)
    z = 1.96
    denominator = 1.0 + z * z / len(values)
    center = observed + z * z / (2.0 * len(values))
    margin = z * math.sqrt(
        (observed * (1.0 - observed) + z * z / (4.0 * len(values)))
        / len(values)
    )
    lower = max(0.0, (center - margin) / denominator)
    average_win = float(wins.mean()) if len(wins) else 0.0
    average_loss = float(losses.mean()) if len(losses) else 0.0
    return lower * average_win - (1.0 - lower) * average_loss


def _metrics(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    extra_cost_pips: float = 0.0,
) -> dict[str, Any]:
    adjusted = values - extra_cost_pips
    if not len(adjusted):
        return {
            "trades": 0,
            "net_pips": 0.0,
            "profit_factor": None,
            "expectancy_pips": None,
            "pessimistic_expectancy_pips": None,
            "positive_month_rate": 0.0,
            "positive_year_rate": 0.0,
            "yearly": [],
        }
    months = np.array(
        [datetime.fromtimestamp(int(item), tz=timezone.utc).strftime("%Y-%m") for item in timestamps]
    )
    years = np.array(
        [datetime.fromtimestamp(int(item), tz=timezone.utc).year for item in timestamps]
    )
    monthly = [float(adjusted[months == item].sum()) for item in np.unique(months)]
    yearly = [
        {
            "year": int(item),
            "trades": int(np.count_nonzero(years == item)),
            "net_pips": round(float(adjusted[years == item].sum()), 6),
            "profit_factor": _profit_factor(adjusted[years == item]),
        }
        for item in np.unique(years)
    ]
    pessimistic = _pessimistic_expectancy(adjusted)
    return {
        "trades": len(adjusted),
        "wins": int(np.count_nonzero(adjusted > 0.0)),
        "losses": int(np.count_nonzero(adjusted < 0.0)),
        "net_pips": round(float(adjusted.sum()), 6),
        "profit_factor": _profit_factor(adjusted),
        "expectancy_pips": round(float(adjusted.mean()), 6),
        "pessimistic_expectancy_pips": (
            round(float(pessimistic), 6) if pessimistic is not None else None
        ),
        "positive_month_rate": round(
            sum(value > 0.0 for value in monthly) / len(monthly), 6
        ),
        "positive_year_rate": round(
            sum(float(row["net_pips"]) > 0.0 for row in yearly) / len(yearly), 6
        ),
        "yearly": yearly,
        "spread_included": True,
        "additional_cost_pips": extra_cost_pips,
    }


def _profit_factor_value(value: Any) -> float:
    return math.inf if value == "INF" else float(value or 0.0)


def _pre_holdout_qualified(
    train: Mapping[str, Any], validation: Mapping[str, Any]
) -> bool:
    return bool(
        int(train.get("trades") or 0) >= 500
        and int(validation.get("trades") or 0) >= 250
        and _profit_factor_value(train.get("profit_factor")) > 1.0
        and _profit_factor_value(validation.get("profit_factor")) > 1.0
        and float(train.get("net_pips") or 0.0) > 0.0
        and float(validation.get("net_pips") or 0.0) > 0.0
        and float(train.get("positive_year_rate") or 0.0) >= 0.75
        and float(validation.get("positive_year_rate") or 0.0) == 1.0
    )


def _holdout_admissible(metrics: Mapping[str, Any], stressed: Mapping[str, Any]) -> bool:
    return bool(
        int(metrics.get("trades") or 0) >= 100
        and _profit_factor_value(metrics.get("profit_factor")) >= 1.25
        and float(metrics.get("net_pips") or 0.0) > 0.0
        and float(metrics.get("pessimistic_expectancy_pips") or 0.0) > 0.0
        and float(metrics.get("positive_month_rate") or 0.0) >= 2.0 / 3.0
        and _profit_factor_value(stressed.get("profit_factor")) > 1.0
        and float(stressed.get("net_pips") or 0.0) > 0.0
    )


def _candidate_rows(
    pair: str,
    data: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    factor = _pip_factor(pair)
    timestamps = data["t"]
    minute_clocks = (timestamps % 3600) == 0
    entry_spread = (data["ao"] - data["bo"]) * factor
    candidates: list[dict[str, Any]] = []
    for hour_utc in range(24):
        hour_mask = ((timestamps // 3600) % 24) == hour_utc
        for holding_minutes in HOLDING_MINUTES:
            starts = np.flatnonzero(
                minute_clocks
                & hour_mask
                & (entry_spread <= MAXIMUM_ENTRY_SPREAD_PIPS)
            )
            ends = starts + holding_minutes - 1
            valid = ends < len(timestamps)
            starts = starts[valid]
            ends = ends[valid]
            contiguous = timestamps[ends] - timestamps[starts] == (
                holding_minutes - 1
            ) * 60
            starts = starts[contiguous]
            ends = ends[contiguous]
            entry_times = timestamps[starts]
            for side in SIDES:
                values = (
                    (data["bc"][ends] - data["ao"][starts]) * factor
                    if side == "LONG"
                    else (data["bo"][starts] - data["ac"][ends]) * factor
                )
                train_mask = entry_times < TRAIN_END
                validation_mask = (entry_times >= TRAIN_END) & (
                    entry_times < VALIDATION_END
                )
                train = _metrics(entry_times[train_mask], values[train_mask])
                validation = _metrics(
                    entry_times[validation_mask], values[validation_mask]
                )
                candidates.append(
                    {
                        "candidate_id": (
                            f"{pair}:{side}:UTC_{hour_utc:02d}:HOLD_{holding_minutes}M"
                        ),
                        "pair": pair,
                        "side": side,
                        "hour_utc": hour_utc,
                        "holding_minutes": holding_minutes,
                        "train": train,
                        "validation": validation,
                        "pre_holdout_qualified": _pre_holdout_qualified(
                            train, validation
                        ),
                        "_holdout_timestamps": entry_times[
                            entry_times >= VALIDATION_END
                        ],
                        "_holdout_values": values[entry_times >= VALIDATION_END],
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
    selection: dict[str, Any]
    if chosen is None:
        selection = {
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
            "side": chosen["side"],
            "hour_utc": chosen["hour_utc"],
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
            "candidate_family": "FIXED_UTC_SESSION_CLOCK_TIME_CLOSE",
            "hours_utc": list(range(24)),
            "sides": list(SIDES),
            "holding_minutes": list(HOLDING_MINUTES),
            "maximum_entry_spread_pips": MAXIMUM_ENTRY_SPREAD_PIPS,
            "candidate_count": len(public_candidates),
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
