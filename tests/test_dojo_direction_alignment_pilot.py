from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quant_rabbit.dojo_direction_alignment_pilot import (
    PLAN_CONTRACT,
    evaluate_direction_alignment_plan,
)
from quant_rabbit.dojo_fresh_model_handoff import DojoFreshModelHandoffError
from quant_rabbit.dojo_paired_model_queue import canonical_sha256


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_room(
    root: Path,
    *,
    room_id: str,
    cost_mode: str,
    opposed_pl: float,
    aligned_pl: float,
) -> dict:
    room = root / room_id
    room.mkdir()
    contract = {
        "contract": "QR_VIRTUAL_MARKET_SESSION_V2",
        "room_id": room_id,
        "proof_mode": "diagnostic",
        "proof_eligible": False,
        "authority": {
            "broker_mutation_allowed": False,
            "live_permission": False,
            "order_authority": "NONE",
        },
        "initial_balance_jpy": 200_000.0,
        "costs": {
            "explicit": True,
            "leverage": 25.0,
            "slippage_pips_per_fill": 0.0 if cost_mode == "BASE" else 0.3,
            "financing_pips_per_day": 0.0 if cost_mode == "BASE" else 0.8,
        },
    }
    contract_path = room / "session_contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
    rows: list[dict] = []
    previous = "0" * 64

    def append(event: str, payload: dict) -> None:
        nonlocal previous
        body = {
            "event": event,
            "payload": payload,
            "prev_sha": previous,
            "ts_utc": f"2026-07-01T00:0{len(rows)}:00+00:00",
        }
        row = {**body, "sha": canonical_sha256(body)}
        previous = row["sha"]
        rows.append(row)

    for ordinal, (side, trend, pl) in enumerate(
        (("SHORT", "LONG", opposed_pl), ("LONG", "LONG", aligned_pl)),
        start=1,
    ):
        context = {
            "contract": "QR_DOJO_ENTRY_CONTEXT_V1",
            "trend_24h": trend,
        }
        trade_id = f"T{ordinal}"
        append(
            "FILL_LIMIT",
            {
                "trade_id": trade_id,
                "side": side,
                "units": 1000.0,
                "price": 160.0,
                "entry_context": context,
                "entry_context_sha256": canonical_sha256(context),
            },
        )
        append(
            "EXIT_TP" if pl >= 0 else "EXIT_SL",
            {
                "trade_id": trade_id,
                "pl_jpy": pl,
            },
        )
    ledger_path = room / "ledger.jsonl"
    ledger_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "room_dir": str(room),
        "family": "FADE",
        "cost_mode": cost_mode,
        "session_contract_file_sha256": _file_sha(contract_path),
        "ledger_file_sha256": _file_sha(ledger_path),
    }


def _plan(tmp_path: Path, implementation_path: Path) -> Path:
    rooms = [
        _write_room(
            tmp_path,
            room_id=f"room-{cost_mode.lower()}-{copy}",
            cost_mode=cost_mode,
            opposed_pl=-100.0,
            aligned_pl=50.0,
        )
        for copy, cost_mode in ((1, "BASE"), (2, "STRESS"))
    ]
    rooms.extend(
        [
            _write_room(
                tmp_path,
                room_id=f"room-spike-{cost_mode.lower()}",
                cost_mode=cost_mode,
                opposed_pl=20.0,
                aligned_pl=80.0,
            )
            for cost_mode in ("BASE", "STRESS")
        ]
    )
    for room in rooms[2:]:
        room["family"] = "SPIKE"
    body = {
        "contract": PLAN_CONTRACT,
        "schema_version": 1,
        "classification": "LINEAGE_UNSEEN_DIAGNOSTIC_NOT_OOS_PROOF",
        "authority": {
            "broker_mutation_allowed": False,
            "live_permission": False,
            "order_authority": "NONE",
            "paper_replay_only": True,
            "automatic_deployment_allowed": False,
        },
        "future_quote_allowed": False,
        "terminal_result_allowed_in_decision": False,
        "intervention_event": "FILL_LIMIT",
        "rule": "SKIP_ENTRY_WHEN_SIDE_OPPOSES_TREND_24H",
        "decision_information_policy": "FILL_ENTRY_CONTEXT_ONLY",
        "implementation": {
            "path": str(implementation_path),
            "sha256": _file_sha(implementation_path),
        },
        "adoption_thresholds": {
            "net_delta_jpy_min": 0.0,
            "winner_profit_retention_ratio_min": 0.75,
        },
        "rooms": rooms,
    }
    plan = {**body, "plan_sha256": canonical_sha256(body)}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    return path


def test_paired_pilot_passes_loss_avoiding_family_and_rejects_winner_skip(
    tmp_path: Path,
) -> None:
    implementation = Path(__file__).parents[1] / (
        "src/quant_rabbit/dojo_direction_alignment_pilot.py"
    )
    result = evaluate_direction_alignment_plan(_plan(tmp_path, implementation))

    assert result["passed_families"] == ["FADE"]
    fade = next(
        item for item in result["family_decisions"] if item["family"] == "FADE"
    )
    spike = next(
        item for item in result["family_decisions"] if item["family"] == "SPIKE"
    )
    assert fade["decision"] == "PASS_DIAGNOSTIC_SHADOW_ONLY"
    assert spike["decision"] == "REJECT_NO_PAPER_APPLICATION"
    assert result["rooms"][0]["effect"]["net_delta_jpy"] == pytest.approx(100.0)
    assert result["paper_application"] == "NONE_SHADOW_EVIDENCE_ONLY"


def test_pilot_fails_closed_when_bound_ledger_bytes_change(tmp_path: Path) -> None:
    implementation = Path(__file__).parents[1] / (
        "src/quant_rabbit/dojo_direction_alignment_pilot.py"
    )
    plan = _plan(tmp_path, implementation)
    value = json.loads(plan.read_text(encoding="utf-8"))
    ledger = Path(value["rooms"][0]["room_dir"]) / "ledger.jsonl"
    ledger.write_text(ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(DojoFreshModelHandoffError, match="bytes hash mismatch"):
        evaluate_direction_alignment_plan(plan)
