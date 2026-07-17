#!/usr/bin/env python3
"""Build the honest 4x profit-path feasibility board from sealed artifacts.

The board converts sealed pips/day evidence into leverage-dependent 30-day
compounding multiples and computes the remaining gap to the declared monthly
4x goal.  Every row carries an evidence status; rows that are not PROVEN are
sealed as non-citable for goal claims.  The board grants no order authority,
no live permission, and proves no future profit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

CONTRACT = "QR_PROFIT_PATH_FEASIBILITY_V1"
PIP_FRACTION = 0.0001
LEVERAGE_GRID = (5.0, 10.0, 25.0)
TARGET_MULTIPLE = 4.0
ACCEPTED_FALLBACK_MULTIPLE = 3.0
CALENDAR_DAYS = 30
TRADING_DAYS = 22
# hold 12h / cadence 4h / rank 2 on both sides -> up to 3 overlapping
# decision waves of 4 positions each share the exposure budget.
DEFAULT_CONCURRENT_POSITIONS = 12
AUTHORITY = {
    "diagnostic_only": True,
    "forward_proof_eligible": False,
    "promotion_allowed": False,
    "order_authority": "NONE",
    "live_permission": False,
    "broker_mutation_allowed": False,
}
# Design targets for lanes that have no sealed evidence yet.  They exist so
# the gap arithmetic is explicit; they are never citable as evidence.
DESIGN_TARGET_LANES: tuple[dict[str, Any], ...] = (
    {
        "lane": "B_M5_CURRENCY_STRENGTH",
        "status": "DESIGN_TARGET_NO_EVIDENCE",
        "pips_per_day": 100.0,
        "stressed_pips_per_day": 80.0,
    },
    {
        "lane": "C_FAST_BOT_PASSIVE",
        "status": "DESIGN_TARGET_NO_EVIDENCE",
        "pips_per_day": 75.0,
        "stressed_pips_per_day": 50.0,
    },
)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_digest(value: Mapping[str, Any], digest_key: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("artifact must be an object")
    body = {key: item for key, item in value.items() if key != digest_key}
    if value.get(digest_key) != _canonical_sha(body):
        raise ValueError(f"{digest_key} digest is invalid")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def daily_return_fraction(
    pips_per_day: float, leverage: float, *, concurrent_positions: int
) -> float:
    """USD-quote approximation: pip value per unit is PIP_FRACTION."""

    if leverage <= 0 or concurrent_positions <= 0 or pips_per_day < 0:
        raise ValueError("feasibility inputs must be positive")
    return pips_per_day * PIP_FRACTION * (leverage / concurrent_positions)


def required_pips_per_day(
    *,
    leverage: float,
    days: int,
    concurrent_positions: int,
    target_multiple: float = TARGET_MULTIPLE,
) -> float:
    needed_daily = target_multiple ** (1.0 / days) - 1.0
    return needed_daily / (PIP_FRACTION * leverage / concurrent_positions)


def build_board(
    lock: Mapping[str, Any],
    validation: Mapping[str, Any],
    prospective: Mapping[str, Any],
    *,
    concurrent_positions: int = DEFAULT_CONCURRENT_POSITIONS,
) -> dict[str, Any]:
    _verify_digest(lock, "lock_sha256")
    _verify_digest(validation, "evaluation_sha256")
    _verify_digest(prospective, "prospective_lock_sha256")
    if validation.get("lock_sha256") != lock["lock_sha256"]:
        raise ValueError("validation replication is not bound to this lock")
    if prospective.get("shadow_lock_sha256") != lock["lock_sha256"]:
        raise ValueError("prospective final-test lock is not bound to this lock")
    if validation.get("independent_validation_claim_allowed") is not False:
        raise ValueError("validation independence claim must be explicitly false")

    metrics = validation["metrics"]
    active_days = int(metrics["active_days"])
    if active_days <= 0:
        raise ValueError("validation active_days must be positive")
    survivor_lane = {
        "lane": "A_S5_CROSS_SECTIONAL_SURVIVOR",
        "status": "UNPROVEN_NON_INDEPENDENT_AWAITING_FUTURE_TEST",
        "spec_id": lock["spec"]["spec_id"],
        "pips_per_day": round(float(metrics["net_pips"]) / active_days, 9),
        "stressed_pips_per_day": round(
            float(metrics["stressed_net_pips"]) / active_days, 9
        ),
    }
    lanes: list[dict[str, Any]] = [survivor_lane, *DESIGN_TARGET_LANES]
    proven = [row for row in lanes if row["status"] == "PROVEN_INDEPENDENT"]

    scenarios: list[dict[str, Any]] = []
    for leverage in LEVERAGE_GRID:
        cumulative: list[dict[str, Any]] = []
        for row in lanes:
            cumulative.append(row)
            for pips_key in ("pips_per_day", "stressed_pips_per_day"):
                pips = sum(float(item[pips_key]) for item in cumulative)
                daily = daily_return_fraction(
                    pips, leverage, concurrent_positions=concurrent_positions
                )
                scenarios.append(
                    {
                        "leverage": leverage,
                        "lanes": [item["lane"] for item in cumulative],
                        "pips_basis": pips_key,
                        "pips_per_day": round(pips, 9),
                        "daily_return_fraction": round(daily, 9),
                        "multiple_30_calendar_days": round(
                            (1.0 + daily) ** CALENDAR_DAYS, 9
                        ),
                        "multiple_22_trading_days": round(
                            (1.0 + daily) ** TRADING_DAYS, 9
                        ),
                        "citable_for_goal_claims": all(
                            item["status"] == "PROVEN_INDEPENDENT"
                            for item in cumulative
                        ),
                    }
                )
        scenarios.append(
            {
                "leverage": leverage,
                "lanes": ["REQUIREMENT_ONLY"],
                "pips_basis": "required_for_4x",
                "required_pips_per_day_30_calendar_days": round(
                    required_pips_per_day(
                        leverage=leverage,
                        days=CALENDAR_DAYS,
                        concurrent_positions=concurrent_positions,
                    ),
                    9,
                ),
                "required_pips_per_day_22_trading_days": round(
                    required_pips_per_day(
                        leverage=leverage,
                        days=TRADING_DAYS,
                        concurrent_positions=concurrent_positions,
                    ),
                    9,
                ),
                "citable_for_goal_claims": False,
            }
        )

    max_leverage = max(LEVERAGE_GRID)
    required_at_cap = required_pips_per_day(
        leverage=max_leverage,
        days=CALENDAR_DAYS,
        concurrent_positions=concurrent_positions,
    )
    body: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "goal": {
            "target_multiple": TARGET_MULTIPLE,
            "calendar_days": CALENDAR_DAYS,
            "trading_days": TRADING_DAYS,
            "required_daily_factor_30_calendar_days": round(
                TARGET_MULTIPLE ** (1.0 / CALENDAR_DAYS), 9
            ),
            "required_daily_factor_22_trading_days": round(
                TARGET_MULTIPLE ** (1.0 / TRADING_DAYS), 9
            ),
        },
        "sizing_model": {
            "pip_fraction": PIP_FRACTION,
            "concurrent_positions": concurrent_positions,
            "leverage_grid": list(LEVERAGE_GRID),
            "usd_quote_approximation": True,
            "execution_degradation_modeled": False,
        },
        "bound_artifacts": {
            "lock_sha256": lock["lock_sha256"],
            "validation_evaluation_sha256": validation["evaluation_sha256"],
            "prospective_lock_sha256": prospective["prospective_lock_sha256"],
        },
        "lanes": lanes,
        "scenarios": scenarios,
        "four_x_gap_pips_per_day_at_max_leverage": round(
            required_at_cap - float(survivor_lane["pips_per_day"]), 9
        ),
        "three_x_required_pips_per_day_at_max_leverage": round(
            required_pips_per_day(
                leverage=max_leverage,
                days=CALENDAR_DAYS,
                concurrent_positions=concurrent_positions,
                target_multiple=ACCEPTED_FALLBACK_MULTIPLE,
            ),
            9,
        ),
        "three_x_gap_pips_per_day_at_max_leverage": round(
            required_pips_per_day(
                leverage=max_leverage,
                days=CALENDAR_DAYS,
                concurrent_positions=concurrent_positions,
                target_multiple=ACCEPTED_FALLBACK_MULTIPLE,
            )
            - float(survivor_lane["pips_per_day"]),
            9,
        ),
        "proven_lane_count": len(proven),
        "four_x_supported_by_proven_evidence": False,
        "unproven_rows_not_citable_as_goal_evidence": True,
        "monthly_4x_claim_allowed": False,
        **AUTHORITY,
    }
    return {**body, "board_sha256": _canonical_sha(body)}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    ) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--prospective-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--concurrent-positions", type=int, default=DEFAULT_CONCURRENT_POSITIONS
    )
    args = parser.parse_args(argv)
    board = build_board(
        _load_object(args.lock),
        _load_object(args.validation),
        _load_object(args.prospective_lock),
        concurrent_positions=args.concurrent_positions,
    )
    _atomic_json(args.output, board)
    print(
        json.dumps(
            {
                "status": "FEASIBILITY_BOARD_SEALED",
                "board_sha256": board["board_sha256"],
                "four_x_gap_pips_per_day_at_max_leverage": board[
                    "four_x_gap_pips_per_day_at_max_leverage"
                ],
                "four_x_supported_by_proven_evidence": False,
                "order_authority": "NONE",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
