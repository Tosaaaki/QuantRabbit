#!/usr/bin/env python3
"""Causal walk-forward research for the EUR/USD shock continuation shadow.

The script consumes local M1 bid/ask truth only.  It does not expose a broker
client, does not mutate runtime policy, and cannot grant live permission.
Candidate and target selection use train plus validation only; the chronological
holdout is read exactly once after selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for item in (SRC, TOOLS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from analyze_fast_bot_shock_guard_replay import (  # noqa: E402
    _architecture_metrics,
    _catastrophe_width,
    _episodes,
    _exit_architecture_trade,
    _load,
    _m5_atr,
)
from quant_rabbit.fast_bot_shock_guard import load_config  # noqa: E402


TRAIN_END = _validation_start = int(
    datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
)
VALIDATION_END = _holdout_start = int(
    datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
)

TARGET_R_GRID = (0.25, 0.50, 0.75, 1.00)
EXTRA_COST_STRESS_PIPS = (0.0, 0.2, 0.5, 1.0)


def _split(timestamp: int) -> str:
    if timestamp < TRAIN_END:
        return "TRAIN_2020_2023"
    if timestamp < VALIDATION_END:
        return "VALIDATION_2024_2025"
    return "HOLDOUT_2026"


def _side_relative_return(mid: np.ndarray, end: int, minutes: int, direction: int) -> float:
    return float((mid[end] - mid[end - minutes]) * 10_000.0 * direction)


def _trend_efficiency(mid: np.ndarray, end: int, minutes: int) -> float:
    path = mid[end - minutes : end + 1]
    gross = float(np.sum(np.abs(np.diff(path))))
    return abs(float(path[-1] - path[0])) / gross if gross > 0.0 else 0.0


def _episode_features(
    data: dict[str, np.ndarray],
    atr: np.ndarray,
    episode_index: int,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    entry_index = episode_index + int(config["resolution"]["freeze_minutes"])
    if episode_index < 240 or entry_index + 61 >= len(data["t"]):
        return None
    if np.any(np.diff(data["t"][episode_index - 240 : entry_index + 62]) != 60):
        return None
    mid = (data["bc"] + data["ac"]) / 2.0
    spread = (data["ac"] - data["bc"]) * 10_000.0
    direction = 1 if mid[episode_index] > mid[episode_index - 15] else -1
    post = mid[episode_index + 1 : entry_index + 1]
    new_extreme = bool(
        float(np.max(post)) > float(mid[episode_index])
        if direction > 0
        else float(np.min(post)) < float(mid[episode_index])
    )
    adverse = max(
        0.0,
        -float(np.min((post - mid[episode_index]) * 10_000.0 * direction)),
    )
    confirmed = bool(
        new_extreme
        and adverse
        < float(config["resolution"]["minimum_adverse_reversal_pips"])
    )
    spread_history = spread[entry_index - 60 : entry_index]
    spread_median = float(np.median(spread_history))
    spread_ratio = (
        float(spread[entry_index]) / spread_median if spread_median > 0.0 else math.inf
    )
    return {
        "episode_index": episode_index,
        "entry_index": entry_index,
        "timestamp": int(data["t"][entry_index]),
        "split": _split(int(data["t"][entry_index])),
        "direction": direction,
        "direction_name": "UP" if direction > 0 else "DOWN",
        "continuation_confirmed": confirmed,
        "post_shock_adverse_pips": adverse,
        "h1_return_pips": _side_relative_return(mid, entry_index, 60, direction),
        "h4_return_pips": _side_relative_return(mid, entry_index, 240, direction),
        "h1_efficiency": _trend_efficiency(mid, entry_index, 60),
        "spread_pips": float(spread[entry_index]),
        "spread_ratio": spread_ratio,
        "atr_pips": float(atr[entry_index]),
        "hour_utc": datetime.fromtimestamp(
            int(data["t"][entry_index]), tz=timezone.utc
        ).hour,
    }


CandidatePredicate = Callable[[dict[str, Any]], bool]


def _candidates() -> dict[str, CandidatePredicate]:
    return {
        "CONFIRM_ONLY": lambda row: bool(row["continuation_confirmed"]),
        "CONFIRM_H1": lambda row: bool(
            row["continuation_confirmed"] and row["h1_return_pips"] > 0.0
        ),
        "CONFIRM_H1_H4": lambda row: bool(
            row["continuation_confirmed"]
            and row["h1_return_pips"] > 0.0
            and row["h4_return_pips"] > 0.0
        ),
        "CONFIRM_H1_EFF20": lambda row: bool(
            row["continuation_confirmed"]
            and row["h1_return_pips"] > 0.0
            and row["h1_efficiency"] >= 0.20
        ),
        "CONFIRM_H1_EFF20_LIQUID": lambda row: bool(
            row["continuation_confirmed"]
            and row["h1_return_pips"] > 0.0
            and row["h1_efficiency"] >= 0.20
            and row["spread_ratio"] <= 1.5
            and row["spread_pips"] <= 1.5
        ),
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trades": 0,
            "net_pips": 0.0,
            "profit_factor": None,
            "risk_scaled_net_pip_units": 0.0,
            "risk_scaled_profit_factor": None,
        }
    return _architecture_metrics(rows)


def _cost_stress(rows: list[dict[str, Any]], extra_cost_pips: float) -> dict[str, Any]:
    stressed = []
    for row in rows:
        copy = dict(row)
        copy["pnl"] = float(copy["pnl"]) - extra_cost_pips
        stressed.append(copy)
    metrics = _metrics(stressed)
    return {
        "trades": metrics["trades"],
        "net_pips": metrics["net_pips"],
        "profit_factor": metrics["profit_factor"],
        "risk_scaled_net_pip_units": metrics["risk_scaled_net_pip_units"],
        "risk_scaled_profit_factor": metrics["risk_scaled_profit_factor"],
        "risk_scaled_p05_pip_units": metrics.get("risk_scaled_p05_pip_units"),
    }


def _side_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("UP", "DOWN"):
        selected = [row for row in rows if row["direction_name"] == side]
        result[side] = _metrics(selected)
    return result


def _qualified_pre_holdout(
    train: dict[str, Any], validation: dict[str, Any]
) -> bool:
    train_pf = train.get("risk_scaled_profit_factor")
    validation_pf = validation.get("risk_scaled_profit_factor")
    return bool(
        int(train.get("trades") or 0) >= 100
        and int(validation.get("trades") or 0) >= 50
        and train_pf is not None
        and validation_pf is not None
        and float(train_pf) > 1.0
        and float(validation_pf) > 1.0
        and float(train.get("risk_scaled_net_pip_units") or 0.0) > 0.0
        and float(validation.get("risk_scaled_net_pip_units") or 0.0) > 0.0
    )


def select_without_holdout(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        cell
        for cell in cells
        if _qualified_pre_holdout(cell["train"], cell["validation"])
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda cell: (
            min(
                float(cell["train"]["risk_scaled_profit_factor"]),
                float(cell["validation"]["risk_scaled_profit_factor"]),
            ),
            float(cell["validation"]["risk_scaled_net_pip_units"]),
            -int(cell["validation"]["maximum_loss_streak"]),
            -float(cell["target_r"]),
            cell["candidate"],
        ),
    )


def _holdout_admissible(metrics: dict[str, Any], sides: dict[str, Any]) -> bool:
    pf = metrics.get("risk_scaled_profit_factor")
    stressed = metrics.get("cost_stress_0_5") or {}
    stressed_pf = stressed.get("risk_scaled_profit_factor")
    return bool(
        int(metrics.get("trades") or 0) >= 30
        and pf is not None
        and float(pf) > 1.0
        and float(metrics.get("risk_scaled_net_pip_units") or 0.0) > 0.0
        and stressed_pf is not None
        and float(stressed_pf) >= 0.9
        and all(int(sides[side].get("trades") or 0) >= 10 for side in ("UP", "DOWN"))
    )


def analyze(paths: list[Path], config_path: Path) -> dict[str, Any]:
    config, config_sha = load_config(config_path)
    data = _load(paths)
    atr = _m5_atr(data)
    detected = _episodes(data, config)
    features = []
    for index in detected:
        row = _episode_features(data, atr, index, config)
        if row is not None and math.isfinite(float(row["atr_pips"])):
            features.append(row)

    trade_rows: dict[float, list[dict[str, Any]]] = {target: [] for target in TARGET_R_GRID}
    for feature in features:
        entry_index = int(feature["entry_index"])
        direction = int(feature["direction"])
        width = _catastrophe_width(
            data,
            float(feature["atr_pips"]),
            entry_index,
            direction,
            config,
        )
        for target_r in TARGET_R_GRID:
            trade = _exit_architecture_trade(
                data,
                entry_index,
                direction,
                catastrophe_width=width,
                config=config,
                structure_enabled=True,
                take_profit=width * target_r,
            )
            trade.update(feature)
            trade["target_r"] = target_r
            trade["catastrophe_width_pips"] = width
            trade["normalized_unit_fraction"] = min(1.0, 3.2 / width)
            trade_rows[target_r].append(trade)

    cells: list[dict[str, Any]] = []
    predicates = _candidates()
    for candidate, predicate in predicates.items():
        for target_r, rows in trade_rows.items():
            selected = [row for row in rows if predicate(row)]
            split_rows = {
                split: [row for row in selected if row["split"] == split]
                for split in ("TRAIN_2020_2023", "VALIDATION_2024_2025", "HOLDOUT_2026")
            }
            train = _metrics(split_rows["TRAIN_2020_2023"])
            validation = _metrics(split_rows["VALIDATION_2024_2025"])
            cells.append(
                {
                    "candidate": candidate,
                    "target_r": target_r,
                    "train": train,
                    "validation": validation,
                    "pre_holdout_qualified": _qualified_pre_holdout(train, validation),
                }
            )

    chosen = select_without_holdout(cells)
    selection: dict[str, Any]
    if chosen is None:
        selection = {
            "status": "NO_PRE_HOLDOUT_CANDIDATE",
            "shadow_candidate_admitted": False,
            "live_promotion_allowed": False,
            "holdout_opened": False,
        }
    else:
        predicate = predicates[str(chosen["candidate"])]
        selected_holdout = [
            row
            for row in trade_rows[float(chosen["target_r"])]
            if row["split"] == "HOLDOUT_2026" and predicate(row)
        ]
        holdout = _metrics(selected_holdout)
        holdout["cost_stress"] = {
            str(cost): _cost_stress(selected_holdout, cost)
            for cost in EXTRA_COST_STRESS_PIPS
        }
        holdout["cost_stress_0_5"] = holdout["cost_stress"]["0.5"]
        sides = _side_metrics(selected_holdout)
        admitted = _holdout_admissible(holdout, sides)
        selection = {
            "status": "HOLDOUT_PASS" if admitted else "HOLDOUT_REJECT",
            "candidate": chosen["candidate"],
            "target_r": chosen["target_r"],
            "train": chosen["train"],
            "validation": chosen["validation"],
            "holdout": holdout,
            "holdout_by_direction": sides,
            "shadow_candidate_admitted": admitted,
            "live_promotion_allowed": False,
            "holdout_opened": True,
            "selection_used_holdout": False,
        }

    years = Counter(
        datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc).year
        for row in features
    )
    return {
        "contract": "QR_FAST_BOT_SHOCK_PROFITABILITY_WALK_FORWARD_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha,
        "truth": {
            "rows": len(data["t"]),
            "from_utc": datetime.fromtimestamp(int(data["t"][0]), tz=timezone.utc).isoformat(),
            "to_utc": datetime.fromtimestamp(int(data["t"][-1]), tz=timezone.utc).isoformat(),
            "files": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in paths
            ],
            "broker_http_methods_used": [],
        },
        "episodes": {
            "detected": len(detected),
            "causal_feature_complete": len(features),
            "by_year": dict(sorted(years.items())),
            "up": sum(row["direction"] > 0 for row in features),
            "down": sum(row["direction"] < 0 for row in features),
        },
        "protocol": {
            "train": "2020-01-01 through 2023-12-31",
            "validation": "2024-01-01 through 2025-12-31",
            "holdout": "2026-01-01 onward",
            "candidate_grid": list(predicates),
            "target_r_grid": list(TARGET_R_GRID),
            "extra_cost_stress_pips": list(EXTRA_COST_STRESS_PIPS),
            "entry_delay_minutes": int(config["resolution"]["freeze_minutes"]),
            "direction_policy": "ORIGINAL_SHOCK_DIRECTION_ONLY",
            "automatic_reversal_allowed": False,
        },
        "pre_holdout_cells": cells,
        "selection": selection,
        "execution_authority": "NONE",
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_policy": "NO_TOUCH",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "fast_bot_shock_guard_v1.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.input, args.config)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
