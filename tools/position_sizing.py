#!/usr/bin/env python3
"""Dry-run target-aware position sizing.

This tool never stages or sends OANDA orders. It sizes a proposed entry from
current target progress, expected reward, explicit risk capacity, optional
margin capacity, and conviction grade. Any legacy grade/unit band is treated as
an upper cap only; it is not the sizing source.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


# OANDA accepts integer units below 1,000 on this account. Keep this standalone
# tool aligned with quant_rabbit.risk.MIN_PRODUCTION_LOT_UNITS and
# AGENT_CONTRACT §3.5: risk/NAV sizing is preserved to one unit instead of
# rounding a small valid order up or down to a legacy 1,000u band.
MIN_PRODUCTION_LOT_UNITS = 1

# Conviction labels are a trader contract taxonomy. They are ordered only for
# comparing "grade < A" / "grade <= B0" style gates.
GRADE_RANK = {
    "C": 0,
    "B-": 1,
    "B0": 2,
    "B": 2,
    "B+": 3,
    "A": 4,
    "S": 5,
}
TARGET_PATH_MAIN_ROLES = {"MAIN", "HERO", "PATH_A", "5PCT_PATH", "GUARANTEE_5", "PACE_5"}
TARGET_PATH_SUPPORT_ROLES = {"SCOUT", "RELOAD", "SECOND_SHOT", "SUPPORT", "PATH_B"}


@dataclass(frozen=True)
class ExtensionGateInput:
    progress_pct: float
    protected_sa_winner_can_carry: bool = False
    hero_thesis_paying: bool = False
    theme_confirmations: int = 0
    hero_clean_trend_band_walk: bool = False
    spread_stable: bool = False
    major_whipsaw_event_next_30m: bool = False
    last_a_s_trade_state: str = ""
    reload_level_exists: bool = False


@dataclass(frozen=True)
class ExtensionGateResult:
    gate: str
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        return self.gate == "YES"


@dataclass(frozen=True)
class PositionSizingInput:
    pair: str
    side: str
    entry: float
    tp: float
    sl: float
    conviction_grade: str
    allocation_band: str = ""
    day_start_nav: float = 0.0
    current_nav: float = 0.0
    remaining_to_5pct: float = 0.0
    mode: str = "BUILD"
    remaining_to_10pct: float | None = None
    quote_to_jpy: float | None = None
    risk_budget_yen: float | None = None
    remaining_risk_budget_yen: float | None = None
    open_risk_yen: float = 0.0
    margin_available_yen: float | None = None
    margin_per_unit_yen: float | None = None
    unit_cap: int | None = None
    unit_band_caps: dict[str, int] = field(default_factory=dict)
    target_path_role: str = ""
    extension_gate: bool = False
    path_board_available: bool = False
    attack_stack_available: bool = False
    maps_to_attack_stack: bool | None = None
    same_thesis_lost_recently: bool = False
    vehicle_unchanged_after_loss: bool = False


@dataclass(frozen=True)
class PositionSizingResult:
    suggested_units: int
    risk_yen: float
    risk_pct: float
    target_yen: float
    target_pct: float
    contribution_to_5pct: float
    cap_reason: str
    valid_as_target_path: str
    grade: str
    mode: str
    target_path_role: str
    extension_gate: str
    issues: tuple[dict[str, str], ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_extension_gate(data: ExtensionGateInput) -> ExtensionGateResult:
    """Evaluate the explicit +10% extension gate.

    The +3.5% progress reference is from the operator's requested gate: strong
    progress is preferred, unless a protected S/A winner can carry beyond +5%.
    """

    reasons: list[str] = []
    blockers: list[str] = []
    if data.progress_pct >= 3.5:
        reasons.append("progress strong at +3.5% or better")
    elif data.protected_sa_winner_can_carry:
        reasons.append("protected S/A winner can carry past +5%")
    else:
        blockers.append("progress not strong and no protected S/A winner can carry")

    if data.hero_thesis_paying:
        reasons.append("hero thesis still paying")
    else:
        blockers.append("hero thesis is not still paying")

    if data.theme_confirmations >= 3:
        reasons.append("3+ pairs confirm same currency theme")
    elif data.hero_clean_trend_band_walk:
        reasons.append("hero pair has clean trend/band-walk")
    else:
        blockers.append("no 3-pair theme confirmation or clean hero trend")

    if data.spread_stable:
        reasons.append("spread stable")
    else:
        blockers.append("spread not stable")

    if data.major_whipsaw_event_next_30m:
        blockers.append("major whipsaw event inside next 30m")
    else:
        reasons.append("no major whipsaw event inside next 30m")

    last_state = data.last_a_s_trade_state.strip().upper()
    if last_state in {"GREEN", "PROTECTED", "STRUCTURALLY_ALIVE", "ALIVE"}:
        reasons.append("last A/S trade is green, protected, or structurally alive")
    else:
        blockers.append("last A/S trade is not green/protected/structurally alive")

    if data.reload_level_exists:
        reasons.append("real reload/second-shot level exists")
    else:
        blockers.append("no real reload/second-shot level; would be chase")

    return ExtensionGateResult(
        gate="YES" if not blockers else "NO",
        reasons=tuple(reasons),
        blockers=tuple(blockers),
    )


def size_position(data: PositionSizingInput) -> PositionSizingResult:
    grade = normalize_grade(data.conviction_grade or data.allocation_band)
    mode = str(data.mode or "").strip().upper() or "BUILD"
    role = str(data.target_path_role or "").strip().upper()
    progress_pct = _progress_pct(data.day_start_nav, data.current_nav)
    quote_to_jpy = _quote_to_jpy(data.pair, data.quote_to_jpy)
    pip_factor = _pip_factor(data.pair)
    issues = _target_guard_issues(data, grade, progress_pct)

    stop_pips = abs(float(data.entry) - float(data.sl)) * pip_factor
    target_pips = abs(float(data.tp) - float(data.entry)) * pip_factor
    diagnostics: dict[str, Any] = {
        "progress_pct": round(progress_pct, 4),
        "stop_pips": round(stop_pips, 4),
        "target_pips": round(target_pips, 4),
        "quote_to_jpy": quote_to_jpy,
    }
    cap_reasons: list[str] = []
    caps: list[float] = []

    if quote_to_jpy is None:
        issues.append(_issue("CONVERSION_RATE_MISSING", f"{data.pair} needs quote-to-JPY conversion"))
    if stop_pips <= 0:
        issues.append(_issue("STOP_DISTANCE_INVALID", "entry/sl do not define positive loss distance"))
    if target_pips <= 0:
        issues.append(_issue("TARGET_DISTANCE_INVALID", "entry/tp do not define positive reward distance"))

    risk_per_unit = target_per_unit = None
    if quote_to_jpy is not None and stop_pips > 0:
        risk_per_unit = stop_pips * quote_to_jpy / pip_factor
        diagnostics["risk_per_unit_yen"] = risk_per_unit
    if quote_to_jpy is not None and target_pips > 0:
        target_per_unit = target_pips * quote_to_jpy / pip_factor
        diagnostics["target_per_unit_yen"] = target_per_unit

    target_units = None
    target_gap = _target_gap_yen(data)
    if target_per_unit is not None and target_per_unit > 0 and target_gap > 0:
        target_units = math.ceil(target_gap / target_per_unit)
        caps.append(float(target_units))
        cap_reasons.append("TARGET_GAP")
    diagnostics["target_gap_yen"] = round(target_gap, 4)
    diagnostics["target_units_raw"] = target_units

    risk_cap = _risk_cap_yen(data)
    diagnostics["risk_cap_yen"] = risk_cap
    if risk_cap is None or risk_cap <= 0:
        issues.append(_issue("RISK_CAP_MISSING", "remaining/equity-derived risk budget is required for sizing"))
    elif risk_per_unit is not None and risk_per_unit > 0:
        risk_units = math.floor(risk_cap / risk_per_unit)
        caps.append(float(risk_units))
        cap_reasons.append("RISK_CAP")
        diagnostics["risk_units_raw"] = risk_units

    band_cap = _unit_cap(data, grade)
    if band_cap is not None:
        caps.append(float(band_cap))
        cap_reasons.append("UNIT_BAND_CAP")
        diagnostics["unit_band_cap"] = band_cap

    if data.margin_available_yen is not None and data.margin_per_unit_yen is not None:
        if data.margin_available_yen <= 0 or data.margin_per_unit_yen <= 0:
            caps.append(0.0)
            cap_reasons.append("MARGIN_CAP")
        else:
            margin_units = math.floor(data.margin_available_yen / data.margin_per_unit_yen)
            caps.append(float(margin_units))
            cap_reasons.append("MARGIN_CAP")
            diagnostics["margin_units_raw"] = margin_units

    if not caps:
        raw_units = 0
    elif target_units is None:
        raw_units = 0
        issues.append(_issue("TARGET_CONTRIBUTION_MISSING", "positive target gap and reward are required"))
    else:
        raw_units = max(0, math.floor(min(caps)))

    suggested_units = _round_units(raw_units)
    if 0 < raw_units < MIN_PRODUCTION_LOT_UNITS:
        cap_reasons.append("MIN_PRODUCTION_LOT")
        suggested_units = 0
    if any(issue["severity"] == "BLOCK" for issue in issues):
        suggested_units = 0

    risk_yen = (suggested_units * risk_per_unit) if risk_per_unit is not None else 0.0
    target_yen = (suggested_units * target_per_unit) if target_per_unit is not None else 0.0
    valid_as_target_path = _valid_target_path(data, grade)
    if suggested_units <= 0 and valid_as_target_path == "YES":
        valid_as_target_path = "NO"
    contribution_to_5pct = min(max(0.0, target_yen), max(0.0, data.remaining_to_5pct))

    return PositionSizingResult(
        suggested_units=int(suggested_units),
        risk_yen=round(risk_yen, 4),
        risk_pct=_pct(risk_yen, data.day_start_nav),
        target_yen=round(target_yen, 4),
        target_pct=_pct(target_yen, data.day_start_nav),
        contribution_to_5pct=round(contribution_to_5pct, 4),
        cap_reason=";".join(dict.fromkeys(cap_reasons)) or "NONE",
        valid_as_target_path=valid_as_target_path,
        grade=grade,
        mode=mode,
        target_path_role=role,
        extension_gate="YES" if data.extension_gate else "NO",
        issues=tuple(issues),
        diagnostics=diagnostics,
    )


def normalize_grade(value: str) -> str:
    text = str(value or "").strip().upper().replace("_", "").replace(" ", "")
    if text in {"S", "A", "B+", "B0", "B", "B-", "C"}:
        return "B0" if text == "B" else text
    return text or "UNKNOWN"


def grade_rank(grade: str) -> int | None:
    return GRADE_RANK.get(normalize_grade(grade))


def _target_guard_issues(data: PositionSizingInput, grade: str, progress_pct: float) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    rank = grade_rank(grade)
    mode = str(data.mode or "").strip().upper()
    role = str(data.target_path_role or "").strip().upper()
    under_5 = data.remaining_to_5pct > 0
    base_reached = not under_5 or progress_pct >= 5.0
    fresh_risk = role not in {"HEDGE", "PROTECT", "CLOSE"}

    if mode == "EXTEND" and (rank is None or rank < GRADE_RANK["A"]):
        issues.append(_issue("EXTEND_REQUIRES_A_GRADE", "EXTEND mode requires A/S grade risk"))
    if base_reached and not data.extension_gate and fresh_risk and grade.startswith("B"):
        issues.append(_issue("BASE_TARGET_REACHED_B_RISK_BLOCKED", "+5% reached and 10% Extension Gate is NO; fresh B risk is blocked"))
    if (
        under_5
        and rank is not None
        and rank <= GRADE_RANK["B0"]
        and role in (TARGET_PATH_MAIN_ROLES | TARGET_PATH_SUPPORT_ROLES)
    ):
        issues.append(_issue("TARGET_PATH_GRADE_TOO_LOW", "B0/B-/C cannot be +5% pace-path risk"))
    if under_5 and grade == "B+" and role in TARGET_PATH_MAIN_ROLES:
        issues.append(_issue("B_PLUS_NOT_MAIN_TARGET_PATH", "B+ can support scout/reload, not the main +5% pace path"))
    if data.same_thesis_lost_recently and data.vehicle_unchanged_after_loss:
        issues.append(_issue("SAME_THESIS_LOST_RECENTLY", "same thesis lost recently and vehicle is unchanged"))
    if (data.path_board_available or data.attack_stack_available) and data.maps_to_attack_stack is not True:
        issues.append(_issue("PATH_ATTACK_STACK_MAPPING_MISSING", "order must map to 5% PACE BOARD / ATTACK STACK when available"))
    return issues


def _valid_target_path(data: PositionSizingInput, grade: str) -> str:
    rank = grade_rank(grade)
    role = str(data.target_path_role or "").strip().upper()
    if rank is None:
        return "NO"
    if rank >= GRADE_RANK["A"]:
        return "YES"
    if grade == "B+" and role in TARGET_PATH_SUPPORT_ROLES:
        return "YES"
    return "NO"


def _target_gap_yen(data: PositionSizingInput) -> float:
    mode = str(data.mode or "").strip().upper()
    if mode == "EXTEND" and data.remaining_to_10pct is not None:
        return max(0.0, float(data.remaining_to_10pct))
    return max(0.0, float(data.remaining_to_5pct))


def _risk_cap_yen(data: PositionSizingInput) -> float | None:
    if data.remaining_risk_budget_yen is not None:
        return max(0.0, float(data.remaining_risk_budget_yen))
    if data.risk_budget_yen is None:
        return None
    return max(0.0, float(data.risk_budget_yen) - max(0.0, float(data.open_risk_yen)))


def _unit_cap(data: PositionSizingInput, grade: str) -> int | None:
    if data.unit_cap is not None and data.unit_cap > 0:
        return int(data.unit_cap)
    band = str(data.allocation_band or "").strip().upper()
    for key in (band, grade):
        if key in data.unit_band_caps and data.unit_band_caps[key] > 0:
            return int(data.unit_band_caps[key])
    try:
        parsed = int(float(band))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _round_units(value: int) -> int:
    if value < MIN_PRODUCTION_LOT_UNITS:
        return 0
    return int(value // MIN_PRODUCTION_LOT_UNITS) * MIN_PRODUCTION_LOT_UNITS


def _pip_factor(pair: str) -> int:
    return 100 if str(pair).upper().endswith("_JPY") else 10000


def _quote_to_jpy(pair: str, provided: float | None) -> float | None:
    if str(pair).upper().endswith("_JPY"):
        return 1.0
    if provided is not None and provided > 0:
        return float(provided)
    return None


def _progress_pct(day_start_nav: float, current_nav: float) -> float:
    if day_start_nav <= 0:
        return 0.0
    return ((current_nav - day_start_nav) / day_start_nav) * 100.0


def _pct(value: float, base: float) -> float:
    if base <= 0:
        return 0.0
    return round((value / base) * 100.0, 4)


def _issue(code: str, message: str, severity: str = "BLOCK") -> dict[str, str]:
    return {"code": code, "message": message, "severity": severity}


def _parse_caps(values: list[str]) -> dict[str, int]:
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run target-aware position sizing.")
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
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = size_position(
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
            unit_band_caps=_parse_caps(args.unit_band_cap),
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
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
