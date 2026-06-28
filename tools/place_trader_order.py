#!/usr/bin/env python3
"""Dry-run target-aware order guard.

Compatibility wrapper for the legacy `place_trader_order.py` workflow. This
branch does not allow independent OANDA write helpers; real staging/sending
remains exclusively in `LiveOrderGateway`. This script only validates a
proposed order, calls `tools/position_sizing.py`, and prints a dry-run receipt.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

from position_sizing import PositionSizingInput, size_position


# Legacy unit bands, when present, are caps only. This branch had no existing
# UNIT_BANDS table, so the default is intentionally empty; pass
# `--unit-band-cap GRADE=UNITS` or `--unit-cap` for an explicit cap.
UNIT_BANDS: dict[str, int] = {}


@dataclass(frozen=True)
class DryRunGuardResult:
    status: str
    dry_run_only: bool
    live_order_sent: bool
    pair: str
    side: str
    issues: tuple[dict[str, str], ...]
    sizing: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_order(args: argparse.Namespace) -> DryRunGuardResult:
    sizing = size_position(
        PositionSizingInput(
            pair=args.pair,
            side=args.side.upper(),
            entry=args.entry,
            tp=args.tp,
            sl=args.sl,
            conviction_grade=args.grade,
            allocation_band=args.allocation_band,
            day_start_nav=args.day_start_nav,
            current_nav=args.current_nav,
            remaining_to_5pct=args.remaining_to_5pct,
            remaining_to_10pct=args.remaining_to_10pct,
            mode=args.mode,
            quote_to_jpy=args.quote_to_jpy,
            risk_budget_yen=args.risk_budget_yen,
            remaining_risk_budget_yen=args.remaining_risk_budget_yen,
            open_risk_yen=args.open_risk_yen,
            margin_available_yen=args.margin_available_yen,
            margin_per_unit_yen=args.margin_per_unit_yen,
            unit_cap=args.unit_cap,
            unit_band_caps={**UNIT_BANDS, **_parse_caps(args.unit_band_cap)},
            target_path_role=args.target_path_role,
            extension_gate=args.extension_gate == "yes",
            path_board_available=args.path_board_available,
            attack_stack_available=args.attack_stack_available,
            maps_to_attack_stack=(
                None if args.maps_to_attack_stack is None else args.maps_to_attack_stack == "yes"
            ),
            same_thesis_lost_recently=args.same_thesis_lost_recently,
            vehicle_unchanged_after_loss=args.vehicle_unchanged_after_loss,
        )
    )
    issues = list(sizing.issues)
    issues.extend(_pretrade_guard_issues(args))
    status = "DRY_RUN_READY" if sizing.suggested_units > 0 and not _has_block(issues) else "DRY_RUN_BLOCKED"
    if args.send:
        status = "DRY_RUN_BLOCKED"
        issues.append(
            {
                "code": "LIVE_SEND_DISABLED",
                "message": "tools/place_trader_order.py is dry-run only; use LiveOrderGateway for any gated send",
                "severity": "BLOCK",
            }
        )
    return DryRunGuardResult(
        status=status,
        dry_run_only=True,
        live_order_sent=False,
        pair=args.pair,
        side=args.side.upper(),
        issues=tuple(issues),
        sizing=sizing.to_dict(),
    )


def _pretrade_guard_issues(args: argparse.Namespace) -> list[dict[str, str]]:
    checks = (
        ("EXACT_PRETRADE_BLOCKED", args.exact_pretrade_ok, "exact pretrade guard did not pass"),
        ("SPREAD_GUARD_BLOCKED", args.spread_ok, "spread guard did not pass"),
        ("PRICING_PROBE_BLOCKED", args.pricing_probe_ok, "pricing probe did not pass"),
        ("FILL_GUARD_BLOCKED", args.fill_guard_ok, "fill guard did not pass"),
    )
    return [
        {"code": code, "message": message, "severity": "BLOCK"}
        for code, passed, message in checks
        if not passed
    ]


def _has_block(issues: list[dict[str, str]]) -> bool:
    return any(issue.get("severity") == "BLOCK" for issue in issues)


def _parse_caps(values: list[str]) -> dict[str, int]:
    from position_sizing import normalize_grade

    caps: dict[str, int] = {}
    for raw in values:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(f"unit cap must be GRADE=UNITS, got {raw!r}")
        grade, units = raw.split("=", 1)
        parsed = int(units)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("unit cap must be positive")
        caps[normalize_grade(grade)] = parsed
    return caps


def _yes_no(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"yes", "y", "true", "1", "on"}:
        return True
    if normalized in {"no", "n", "false", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected yes/no")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run target-aware order guard.")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--side", required=True, choices=("LONG", "SHORT", "long", "short"))
    parser.add_argument("--entry", required=True, type=float)
    parser.add_argument("--tp", required=True, type=float)
    parser.add_argument("--sl", required=True, type=float)
    parser.add_argument("--grade", "--conviction-grade", dest="grade", required=True)
    parser.add_argument("--allocation-band", default="")
    parser.add_argument("--day-start-nav", required=True, type=float)
    parser.add_argument("--current-nav", required=True, type=float)
    parser.add_argument("--remaining-to-5pct", required=True, type=float)
    parser.add_argument("--remaining-to-10pct", type=float)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--quote-to-jpy", type=float)
    parser.add_argument("--risk-budget-yen", type=float)
    parser.add_argument("--remaining-risk-budget-yen", type=float)
    parser.add_argument("--open-risk-yen", type=float, default=0.0)
    parser.add_argument("--margin-available-yen", type=float)
    parser.add_argument("--margin-per-unit-yen", type=float)
    parser.add_argument("--unit-cap", type=int)
    parser.add_argument("--unit-band-cap", action="append", default=[])
    parser.add_argument("--target-path-role", default="")
    parser.add_argument("--extension-gate", choices=("yes", "no"), default="no")
    parser.add_argument("--path-board-available", action="store_true")
    parser.add_argument("--attack-stack-available", action="store_true")
    parser.add_argument("--maps-to-attack-stack", choices=("yes", "no"))
    parser.add_argument("--same-thesis-lost-recently", action="store_true")
    parser.add_argument("--vehicle-unchanged-after-loss", action="store_true")
    parser.add_argument("--exact-pretrade-ok", type=_yes_no, default=True)
    parser.add_argument("--spread-ok", type=_yes_no, default=True)
    parser.add_argument("--pricing-probe-ok", type=_yes_no, default=True)
    parser.add_argument("--fill-guard-ok", type=_yes_no, default=True)
    parser.add_argument("--send", action="store_true", help="Accepted only to block; this script is dry-run only.")
    return parser


def main() -> int:
    result = evaluate_order(_parser().parse_args())
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
