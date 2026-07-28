"""Read-only paired PAPER pilot for a direction-aligned entry admission rule."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_fresh_model_handoff import DojoFreshModelHandoffError
from quant_rabbit.dojo_paired_model_queue import canonical_sha256

PLAN_CONTRACT: Final = "QR_DOJO_DIRECTION_ALIGNMENT_PILOT_PLAN_V1"
RESULT_CONTRACT: Final = "QR_DOJO_DIRECTION_ALIGNMENT_PILOT_RESULT_V1"
REALIZED_EVENTS: Final = frozenset(
    {"CLOSE", "EXIT_TP", "EXIT_SL", "MARGIN_CLOSEOUT", "FORCED_LIQUIDATION"}
)
FORCED_EVENTS: Final = frozenset({"MARGIN_CLOSEOUT", "FORCED_LIQUIDATION"})
MAX_LEDGER_BYTES: Final = 32 * 1024 * 1024
PAPER_AUTHORITY: Final = {
    "broker_mutation_allowed": False,
    "live_permission": False,
    "order_authority": "NONE",
}


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size > 2 * 1024 * 1024:
        raise DojoFreshModelHandoffError(f"invalid pilot JSON source: {path}")
    with resolved.open("rb") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DojoFreshModelHandoffError(f"pilot JSON root must be object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise DojoFreshModelHandoffError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DojoFreshModelHandoffError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise DojoFreshModelHandoffError(f"{label} must be finite")
    return number


def _file_sha256(path: Path, *, max_bytes: int | None = None) -> str:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or path.is_symlink():
        raise DojoFreshModelHandoffError(f"pilot source must be regular file: {path}")
    if max_bytes is not None and resolved.stat().st_size > max_bytes:
        raise DojoFreshModelHandoffError(f"pilot source exceeds byte bound: {path}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_plan(path: Path) -> dict[str, Any]:
    plan = _read_json(path)
    supplied = plan.get("plan_sha256")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("contract") != PLAN_CONTRACT
        or supplied != canonical_sha256(body)
        or plan.get("authority") != {
            **PAPER_AUTHORITY,
            "paper_replay_only": True,
            "automatic_deployment_allowed": False,
        }
        or plan.get("future_quote_allowed") is not False
        or plan.get("terminal_result_allowed_in_decision") is not False
        or plan.get("intervention_event") != "FILL_LIMIT"
        or plan.get("rule") != "SKIP_ENTRY_WHEN_SIDE_OPPOSES_TREND_24H"
    ):
        raise DojoFreshModelHandoffError("direction alignment pilot plan is invalid")
    rooms = plan.get("rooms")
    if not isinstance(rooms, list) or len(rooms) != 4:
        raise DojoFreshModelHandoffError("pilot plan must bind exactly four rooms")
    implementation = plan.get("implementation")
    if not isinstance(implementation, Mapping):
        raise DojoFreshModelHandoffError("pilot implementation binding is missing")
    implementation_path = Path(str(implementation.get("path") or ""))
    if _file_sha256(implementation_path) != implementation.get("sha256"):
        raise DojoFreshModelHandoffError("pilot implementation hash mismatch")
    return plan


def _read_verified_ledger(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if _file_sha256(path, max_bytes=MAX_LEDGER_BYTES) != expected_sha256:
        raise DojoFreshModelHandoffError("pilot ledger bytes hash mismatch")
    rows: list[dict[str, Any]] = []
    previous = "0" * 64
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DojoFreshModelHandoffError("pilot ledger row must be object")
            unsigned = {key: item for key, item in value.items() if key != "sha"}
            if value.get("prev_sha") != previous:
                raise DojoFreshModelHandoffError("pilot ledger prev_sha mismatch")
            if value.get("sha") != canonical_sha256(unsigned):
                raise DojoFreshModelHandoffError("pilot ledger row hash mismatch")
            previous = str(value["sha"])
            rows.append(value)
    return rows


def _max_drawdown(
    *,
    initial_balance: float,
    realized_path: list[float],
) -> tuple[float, float, int]:
    balance = initial_balance
    peak = initial_balance
    max_drawdown = 0.0
    max_drawdown_ratio = 0.0
    ruin_events = 0
    ruined = False
    for change in realized_path:
        balance += change
        peak = max(peak, balance)
        drawdown = peak - balance
        max_drawdown = max(max_drawdown, drawdown)
        if peak > 0:
            max_drawdown_ratio = max(max_drawdown_ratio, drawdown / peak)
        if balance <= 0 and not ruined:
            ruin_events += 1
            ruined = True
        elif balance > 0:
            ruined = False
    return max_drawdown, max_drawdown_ratio, ruin_events


def _room_metrics(binding: Mapping[str, Any]) -> dict[str, Any]:
    room_dir = Path(str(binding.get("room_dir") or "")).resolve(strict=True)
    contract_path = room_dir / "session_contract.json"
    ledger_path = room_dir / "ledger.jsonl"
    if _file_sha256(contract_path) != binding.get("session_contract_file_sha256"):
        raise DojoFreshModelHandoffError("pilot session contract bytes hash mismatch")
    contract = _read_json(contract_path)
    if (
        contract.get("contract") != "QR_VIRTUAL_MARKET_SESSION_V2"
        or contract.get("authority") != PAPER_AUTHORITY
        or contract.get("proof_mode") != "diagnostic"
        or contract.get("proof_eligible") is not False
    ):
        raise DojoFreshModelHandoffError("pilot room authority is invalid")
    ledger = _read_verified_ledger(
        ledger_path,
        str(binding.get("ledger_file_sha256") or ""),
    )
    initial_balance = _finite(contract.get("initial_balance_jpy"), "initial balance")
    costs = contract.get("costs")
    if not isinstance(costs, Mapping) or costs.get("explicit") is not True:
        raise DojoFreshModelHandoffError("pilot room cost contract is invalid")
    leverage = _finite(costs.get("leverage"), "cost leverage")
    if leverage <= 0:
        raise DojoFreshModelHandoffError("cost leverage must be positive")

    fills: dict[str, dict[str, Any]] = {}
    baseline_open: dict[str, tuple[float, float]] = {}
    shadow_open: dict[str, tuple[float, float]] = {}
    baseline_margin_peak = 0.0
    shadow_margin_peak = 0.0
    baseline_gross_units_peak = 0.0
    shadow_gross_units_peak = 0.0
    baseline_path: list[float] = []
    shadow_path: list[float] = []
    exits_by_reason: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "pl_jpy": 0.0}
    )
    baseline_net = 0.0
    shadow_net = 0.0
    baseline_profit = 0.0
    shadow_profit = 0.0
    baseline_loss = 0.0
    shadow_loss = 0.0
    baseline_tp_profit = 0.0
    shadow_tp_profit = 0.0
    skipped_winner_profit = 0.0
    avoided_loss = 0.0
    skipped_trades = 0
    retained_trades = 0
    separately_itemized_cost = 0.0
    itemized_cost_complete = True

    def update_inventory_peaks() -> None:
        nonlocal baseline_margin_peak
        nonlocal shadow_margin_peak
        nonlocal baseline_gross_units_peak
        nonlocal shadow_gross_units_peak
        baseline_gross_units_peak = max(
            baseline_gross_units_peak,
            sum(units for units, _ in baseline_open.values()),
        )
        shadow_gross_units_peak = max(
            shadow_gross_units_peak,
            sum(units for units, _ in shadow_open.values()),
        )
        baseline_margin_peak = max(
            baseline_margin_peak,
            sum(units * price / leverage for units, price in baseline_open.values()),
        )
        shadow_margin_peak = max(
            shadow_margin_peak,
            sum(units * price / leverage for units, price in shadow_open.values()),
        )

    for row in ledger:
        event = str(row.get("event") or "")
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if event == "FILL_LIMIT":
            trade_id = str(payload.get("trade_id") or "")
            if not trade_id or trade_id in fills:
                raise DojoFreshModelHandoffError("pilot fill trade id is invalid")
            entry_context = payload.get("entry_context")
            if not isinstance(entry_context, Mapping):
                raise DojoFreshModelHandoffError("pilot fill entry context is missing")
            if payload.get("entry_context_sha256") != canonical_sha256(entry_context):
                raise DojoFreshModelHandoffError("pilot entry context hash mismatch")
            side = str(payload.get("side") or "")
            trend = str(entry_context.get("trend_24h") or "")
            if side not in {"LONG", "SHORT"} or trend not in {
                "LONG",
                "SHORT",
                "FLAT",
                "UNKNOWN",
            }:
                raise DojoFreshModelHandoffError("pilot side or trend is invalid")
            units = _finite(payload.get("units"), "fill units")
            price = _finite(payload.get("price"), "fill price")
            if units <= 0 or price <= 0:
                raise DojoFreshModelHandoffError("pilot fill economics are invalid")
            skipped = trend in {"LONG", "SHORT"} and side != trend
            fills[trade_id] = {
                "skipped": skipped,
                "side": side,
                "trend_24h": trend,
                "entry_context_sha256": payload["entry_context_sha256"],
            }
            baseline_open[trade_id] = (units, price)
            if skipped:
                skipped_trades += 1
            else:
                retained_trades += 1
                shadow_open[trade_id] = (units, price)
            update_inventory_peaks()
            continue
        if event not in REALIZED_EVENTS:
            continue
        trade_id = str(payload.get("trade_id") or "")
        if trade_id not in fills or trade_id not in baseline_open:
            raise DojoFreshModelHandoffError("pilot exit has no unique prior fill")
        pl = _finite(payload.get("pl_jpy"), "exit P/L")
        skipped = bool(fills[trade_id]["skipped"])
        baseline_net += pl
        baseline_path.append(pl)
        exits_by_reason[event]["count"] = int(exits_by_reason[event]["count"]) + 1
        exits_by_reason[event]["pl_jpy"] = (
            float(exits_by_reason[event]["pl_jpy"]) + pl
        )
        if pl >= 0:
            baseline_profit += pl
            if event == "EXIT_TP":
                baseline_tp_profit += pl
        else:
            baseline_loss += -pl
        if skipped:
            shadow_path.append(0.0)
            if pl >= 0:
                skipped_winner_profit += pl
            else:
                avoided_loss += -pl
        else:
            shadow_net += pl
            shadow_path.append(pl)
            if pl >= 0:
                shadow_profit += pl
                if event == "EXIT_TP":
                    shadow_tp_profit += pl
            else:
                shadow_loss += -pl
        financing = payload.get("financing_jpy")
        execution = payload.get("execution_cost_jpy")
        if financing is None and execution is None:
            itemized_cost_complete = False
        else:
            separately_itemized_cost += abs(
                _finite(financing or 0.0, "financing")
            ) + abs(_finite(execution or 0.0, "execution cost"))
        del baseline_open[trade_id]
        shadow_open.pop(trade_id, None)
        update_inventory_peaks()

    if baseline_open:
        raise DojoFreshModelHandoffError("pilot room has unclosed baseline trades")
    baseline_dd, baseline_dd_ratio, baseline_ruin = _max_drawdown(
        initial_balance=initial_balance,
        realized_path=baseline_path,
    )
    shadow_dd, shadow_dd_ratio, shadow_ruin = _max_drawdown(
        initial_balance=initial_balance,
        realized_path=shadow_path,
    )
    return {
        "room_id": contract.get("room_id"),
        "family": binding.get("family"),
        "cost_mode": binding.get("cost_mode"),
        "source_ledger_sha256": binding.get("ledger_file_sha256"),
        "trade_count": len(fills),
        "skipped_trade_count": skipped_trades,
        "retained_trade_count": retained_trades,
        "baseline": {
            "recorded_net_after_cost_jpy": round(baseline_net, 8),
            "gross_profit_jpy": round(baseline_profit, 8),
            "gross_loss_jpy": round(baseline_loss, 8),
            "tp_exit_profit_jpy": round(baseline_tp_profit, 8),
            "max_drawdown_jpy": round(baseline_dd, 8),
            "max_drawdown_ratio": round(baseline_dd_ratio, 12),
            "ruin_events": baseline_ruin,
            "peak_entry_margin_proxy_jpy": round(baseline_margin_peak, 8),
            "peak_gross_units": round(baseline_gross_units_peak, 8),
        },
        "shadow": {
            "recorded_net_after_cost_jpy": round(shadow_net, 8),
            "gross_profit_jpy": round(shadow_profit, 8),
            "gross_loss_jpy": round(shadow_loss, 8),
            "tp_exit_profit_jpy": round(shadow_tp_profit, 8),
            "max_drawdown_jpy": round(shadow_dd, 8),
            "max_drawdown_ratio": round(shadow_dd_ratio, 12),
            "ruin_events": shadow_ruin,
            "peak_entry_margin_proxy_jpy": round(shadow_margin_peak, 8),
            "peak_gross_units": round(shadow_gross_units_peak, 8),
        },
        "effect": {
            "net_delta_jpy": round(shadow_net - baseline_net, 8),
            "avoided_loss_jpy": round(avoided_loss, 8),
            "skipped_winner_profit_jpy": round(skipped_winner_profit, 8),
            "winner_profit_retention_ratio": (
                round(shadow_profit / baseline_profit, 12)
                if baseline_profit > 0
                else None
            ),
            "early_cut_winner_profit_jpy": 0.0,
        },
        "exit_decomposition": dict(sorted(exits_by_reason.items())),
        "cost_contract": dict(costs),
        "separately_itemized_cost_jpy": (
            round(separately_itemized_cost, 8)
            if itemized_cost_complete
            else None
        ),
        "itemized_cost_status": (
            "COMPLETE" if itemized_cost_complete else "NOT_SEPARATELY_EXPOSED"
        ),
        "open_position_count_at_end": 0,
    }


def evaluate_direction_alignment_plan(plan_path: Path) -> dict[str, Any]:
    """Evaluate a sealed four-room plan without mutating any PAPER artifact."""

    plan = _verified_plan(plan_path)
    rooms = [_room_metrics(binding) for binding in plan["rooms"]]
    thresholds = plan.get("adoption_thresholds")
    if not isinstance(thresholds, Mapping):
        raise DojoFreshModelHandoffError("pilot adoption thresholds are missing")
    retention_min = _finite(
        thresholds.get("winner_profit_retention_ratio_min"),
        "winner retention threshold",
    )
    families: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for room in rooms:
        by_family[str(room["family"])].append(room)
    for family, family_rooms in sorted(by_family.items()):
        modes = {str(room["cost_mode"]) for room in family_rooms}
        if modes != {"BASE", "STRESS"} or len(family_rooms) != 2:
            raise DojoFreshModelHandoffError(
                "each pilot family requires one BASE and one STRESS room"
            )
        checks = {
            "base_and_stress_net_delta_positive": all(
                room["effect"]["net_delta_jpy"]
                > _finite(thresholds.get("net_delta_jpy_min"), "net threshold")
                for room in family_rooms
            ),
            "drawdown_non_worsening": all(
                room["shadow"]["max_drawdown_jpy"]
                <= room["baseline"]["max_drawdown_jpy"]
                for room in family_rooms
            ),
            "ruin_non_worsening": all(
                room["shadow"]["ruin_events"] <= room["baseline"]["ruin_events"]
                for room in family_rooms
            ),
            "margin_proxy_non_worsening": all(
                room["shadow"]["peak_entry_margin_proxy_jpy"]
                <= room["baseline"]["peak_entry_margin_proxy_jpy"]
                for room in family_rooms
            ),
            "winner_profit_retention": all(
                room["effect"]["winner_profit_retention_ratio"] is None
                or room["effect"]["winner_profit_retention_ratio"] >= retention_min
                for room in family_rooms
            ),
            "no_winner_early_cut": all(
                room["effect"]["early_cut_winner_profit_jpy"] == 0
                for room in family_rooms
            ),
        }
        passed = all(checks.values())
        families.append(
            {
                "family": family,
                "checks": checks,
                "decision": (
                    "PASS_DIAGNOSTIC_SHADOW_ONLY"
                    if passed
                    else "REJECT_NO_PAPER_APPLICATION"
                ),
            }
        )
    passed_families = [
        family["family"]
        for family in families
        if family["decision"] == "PASS_DIAGNOSTIC_SHADOW_ONLY"
    ]
    body = {
        "contract": RESULT_CONTRACT,
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "classification": "LINEAGE_UNSEEN_DIAGNOSTIC_NOT_OOS_PROOF",
        "authority": plan["authority"],
        "rule": plan["rule"],
        "decision_information_policy": plan["decision_information_policy"],
        "rooms": sorted(rooms, key=lambda room: str(room["room_id"])),
        "family_decisions": families,
        "passed_families": passed_families,
        "paper_application": "NONE_SHADOW_EVIDENCE_ONLY",
        "promotion_eligible": False,
        "limitations": [
            "GLOBAL_UNTOUCHED_OOS_UNAVAILABLE",
            "PRIOR_RESEARCHER_AGGREGATE_OUTCOME_EXPOSURE",
            "ITEMIZED_COST_MAY_BE_UNAVAILABLE_WHILE_RECORDED_PL_USES_COST_CONTRACT",
            "ENTRY_NOTIONAL_MARGIN_IS_A_PROXY_NOT_BROKER_MARGIN_HISTORY",
        ],
    }
    return {**body, "result_sha256": canonical_sha256(body)}


__all__ = [
    "PLAN_CONTRACT",
    "RESULT_CONTRACT",
    "evaluate_direction_alignment_plan",
]
