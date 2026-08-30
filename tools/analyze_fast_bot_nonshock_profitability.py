#!/usr/bin/env python3
"""Bounded walk-forward research for non-shock EUR/USD hourly setups.

This is a local, zero-authority diagnostic.  It evaluates four side-relative
families on hourly decision clocks, excludes the configured shock band, and
selects with 2020-2025 data before opening the 2026 holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "src", ROOT / "tools"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from analyze_fast_bot_shock_guard_replay import _architecture_metrics, _load, _sl_trade  # noqa: E402
from analyze_fast_bot_shock_profitability import (  # noqa: E402
    EXTRA_COST_STRESS_PIPS,
    TRAIN_END,
    VALIDATION_END,
    _cost_stress,
    _holdout_admissible,
    _metrics,
    _side_metrics,
    _split,
    select_without_holdout,
)
from quant_rabbit.fast_bot_shock_guard import load_config  # noqa: E402


GEOMETRIES = (
    {"id": "TP10_SL7", "take_profit_pips": 10.0, "stop_loss_pips": 7.0},
    {"id": "TP15_SL10", "take_profit_pips": 15.0, "stop_loss_pips": 10.0},
    {"id": "TP20_SL12", "take_profit_pips": 20.0, "stop_loss_pips": 12.0},
)


def _efficiency(mid: np.ndarray, index: int, minutes: int) -> float:
    path = mid[index - minutes : index + 1]
    gross = float(np.sum(np.abs(np.diff(path))))
    return abs(float(path[-1] - path[0])) / gross if gross > 0.0 else 0.0


def _family(feature: dict[str, Any], family: str) -> int | None:
    r60 = float(feature["return_60_pips"])
    r240 = float(feature["return_240_pips"])
    eff = float(feature["efficiency_60"])
    if family == "H1_H4_TREND":
        if abs(r60) >= 5.0 and r60 * r240 > 0.0 and eff >= 0.20:
            return 1 if r60 > 0.0 else -1
    elif family == "H1_MOMENTUM":
        if abs(r60) >= 8.0 and eff >= 0.30:
            return 1 if r60 > 0.0 else -1
    elif family == "H4_PULLBACK_RESUME":
        if abs(r60) >= 5.0 and abs(r240) >= 8.0 and r60 * r240 < 0.0 and eff <= 0.35:
            return 1 if r240 > 0.0 else -1
    elif family == "H1_EXTREME_RANGE_FADE":
        if abs(r60) >= 10.0 and abs(r240) <= 15.0 and eff <= 0.40:
            return -1 if r60 > 0.0 else 1
    else:
        raise ValueError(f"unknown family: {family}")
    return None


FAMILIES = (
    "H1_H4_TREND",
    "H1_MOMENTUM",
    "H4_PULLBACK_RESUME",
    "H1_EXTREME_RANGE_FADE",
)


def _qualified(train: dict[str, Any], validation: dict[str, Any]) -> bool:
    train_pf = train.get("risk_scaled_profit_factor")
    validation_pf = validation.get("risk_scaled_profit_factor")
    return bool(
        int(train.get("trades") or 0) >= 250
        and int(validation.get("trades") or 0) >= 100
        and train_pf is not None
        and validation_pf is not None
        and float(train_pf) > 1.0
        and float(validation_pf) > 1.0
        and float(train.get("risk_scaled_net_pip_units") or 0.0) > 0.0
        and float(validation.get("risk_scaled_net_pip_units") or 0.0) > 0.0
    )


def analyze(paths: list[Path], config_path: Path) -> dict[str, Any]:
    config, config_sha = load_config(config_path)
    data = _load(paths)
    t = data["t"]
    mid = (data["bc"] + data["ac"]) / 2.0
    spread = (data["ac"] - data["bc"]) * 10_000.0
    maximum_spread = 1.5
    shock_threshold = float(config["detection"]["minimum_impulse_pips"])
    features: list[dict[str, Any]] = []
    for index in range(300, len(t) - 61):
        if int(t[index]) % 3600 != 0:
            continue
        if np.any(np.diff(t[index - 300 : index + 62]) != 60):
            continue
        return_15 = float((mid[index] - mid[index - 15]) * 10_000.0)
        if abs(return_15) >= shock_threshold or float(spread[index]) > maximum_spread:
            continue
        features.append(
            {
                "index": index,
                "timestamp": int(t[index]),
                "split": _split(int(t[index])),
                "year": datetime.fromtimestamp(int(t[index]), tz=timezone.utc).year,
                "return_60_pips": float((mid[index] - mid[index - 60]) * 10_000.0),
                "return_240_pips": float((mid[index] - mid[index - 240]) * 10_000.0),
                "efficiency_60": _efficiency(mid, index, 60),
                "spread_pips": float(spread[index]),
            }
        )

    rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for family in FAMILIES:
        for geometry in GEOMETRIES:
            key = (family, str(geometry["id"]))
            rows[key] = []
            for feature in features:
                direction = _family(feature, family)
                if direction is None:
                    continue
                pnl, reason, slippage, _ = _sl_trade(
                    data,
                    int(feature["index"]),
                    direction,
                    float(geometry["stop_loss_pips"]),
                    take_profit=float(geometry["take_profit_pips"]),
                )
                rows[key].append(
                    {
                        **feature,
                        "direction": direction,
                        "direction_name": "UP" if direction > 0 else "DOWN",
                        "pnl": pnl,
                        "reason": reason,
                        "held": 60,
                        "mae": max(0.0, -pnl),
                        "gap_slippage": slippage,
                        "normalized_unit_fraction": min(
                            1.0, 3.2 / float(geometry["stop_loss_pips"])
                        ),
                    }
                )

    cells: list[dict[str, Any]] = []
    for family in FAMILIES:
        for geometry in GEOMETRIES:
            selected = rows[(family, str(geometry["id"]))]
            train = _metrics([row for row in selected if row["split"] == "TRAIN_2020_2023"])
            validation = _metrics(
                [row for row in selected if row["split"] == "VALIDATION_2024_2025"]
            )
            cells.append(
                {
                    "candidate": family,
                    "geometry": geometry,
                    "target_r": float(geometry["take_profit_pips"])
                    / float(geometry["stop_loss_pips"]),
                    "train": train,
                    "validation": validation,
                    "pre_holdout_qualified": _qualified(train, validation),
                }
            )

    selectable = [cell for cell in cells if cell["pre_holdout_qualified"]]
    chosen = select_without_holdout(selectable) if selectable else None
    if chosen is None:
        selection: dict[str, Any] = {
            "status": "NO_PRE_HOLDOUT_CANDIDATE",
            "holdout_opened": False,
            "shadow_candidate_admitted": False,
            "live_promotion_allowed": False,
        }
    else:
        holdout_rows = [
            row
            for row in rows[(str(chosen["candidate"]), str(chosen["geometry"]["id"]))]
            if row["split"] == "HOLDOUT_2026"
        ]
        holdout = _metrics(holdout_rows)
        holdout["cost_stress"] = {
            str(cost): _cost_stress(holdout_rows, cost)
            for cost in EXTRA_COST_STRESS_PIPS
        }
        holdout["cost_stress_0_5"] = holdout["cost_stress"]["0.5"]
        side = _side_metrics(holdout_rows)
        admitted = _holdout_admissible(holdout, side)
        selection = {
            "status": "HOLDOUT_PASS" if admitted else "HOLDOUT_REJECT",
            "candidate": chosen["candidate"],
            "geometry": chosen["geometry"],
            "train": chosen["train"],
            "validation": chosen["validation"],
            "holdout": holdout,
            "holdout_by_direction": side,
            "holdout_opened": True,
            "selection_used_holdout": False,
            "shadow_candidate_admitted": admitted,
            "live_promotion_allowed": False,
        }

    return {
        "contract": "QR_FAST_BOT_NONSHOCK_WALK_FORWARD_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha,
        "truth": {
            "rows": len(t),
            "from_utc": datetime.fromtimestamp(int(t[0]), tz=timezone.utc).isoformat(),
            "to_utc": datetime.fromtimestamp(int(t[-1]), tz=timezone.utc).isoformat(),
            "files": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in paths
            ],
            "broker_http_methods_used": [],
        },
        "protocol": {
            "decision_clock": "HOURLY_COMPLETE_M1",
            "shock_exclusion_pips": shock_threshold,
            "maximum_spread_pips": maximum_spread,
            "families": list(FAMILIES),
            "geometries": list(GEOMETRIES),
            "train_end_exclusive": datetime.fromtimestamp(TRAIN_END, tz=timezone.utc).isoformat(),
            "validation_end_exclusive": datetime.fromtimestamp(
                VALIDATION_END, tz=timezone.utc
            ).isoformat(),
            "holdout_selection_use": "FORBIDDEN",
        },
        "eligible_hourly_observations": len(features),
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
