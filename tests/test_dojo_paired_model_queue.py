from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from quant_rabbit.dojo_paired_inventory_counterfactual import ACTION_IDS, AUTHORITY
from quant_rabbit.dojo_paired_inventory_counterfactual import CADENCE_IDS
from quant_rabbit.dojo_paired_model_queue import (
    DojoPairedModelQueueError,
    canonical_sha256,
    current_ready_packet,
    initialize_queue,
    queue_status,
    seal_model_response,
    submit_model_response,
    verify_model_response,
    verify_queue,
)


def _state_packet(epoch: int) -> dict:
    return {
        "decision_epoch": epoch,
        "input_available_through_epoch": epoch,
        "phase": "C",
        "equity_jpy": 200_000.0,
        "balance_jpy": 200_000.0,
        "drawdown_fraction": 0.0,
        "margin_utilization_fraction": 0.0,
        "gross_exposure_jpy": 0.0,
        "net_exposure_jpy": 0.0,
        "long_gross_exposure_jpy": 0.0,
        "short_gross_exposure_jpy": 0.0,
        "hedge_buildup_fraction": 0.0,
        "directional_skew_fraction": 0.0,
        "unrealized_pnl_jpy": 0.0,
        "realized_profit_giveback_jpy": 0.0,
        "position_count": 0,
        "pending_order_count": 0,
        "stale_valuation_pair_count": 0,
        "maximum_position_age_seconds": 0,
        "consecutive_losses": 0,
        "regime_id": "UNKNOWN",
        "strategy_regime_compatible": True,
        "paused": False,
        "direction_block": None,
        "terminal_result_visible": False,
        "future_quote_visible": False,
        "append_wall_clock_visible": False,
    }


def _fixtures() -> tuple[dict, list[dict]]:
    plan = {
        "study_id": "paired-model-test",
        "plan_sha256": "a" * 64,
        "actual_model_checkpoint_call_required_for_rank": True,
        "terminal_result_allowed_in_decision": False,
        "future_quote_allowed": False,
        "append_wall_clock_allowed": False,
        "cadence_ids": list(CADENCE_IDS),
        "authority": dict(AUTHORITY),
    }
    plan["plan_sha256"] = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    results = []
    cadence_ids = list(CADENCE_IDS)
    for coordinate in range(12):
        rows = []
        for cadence_index, cadence_id in enumerate(cadence_ids):
            epoch = 1_700_000_000 + coordinate * 100 + cadence_index
            packet = _state_packet(epoch)
            rows.append(
                {
                    "cadence_id": cadence_id,
                    "intervention_audit_log": [
                        {
                            "decision_id": (
                                f"decision-{coordinate:02d}-{cadence_index:02d}"
                            ),
                            "decision_epoch": epoch,
                            "input_available_through_epoch": epoch,
                            "packet": packet,
                            "packet_sha256": canonical_sha256(packet),
                            "provider_model_called": False,
                            "future_information_used": False,
                            "post_outcome": {
                                "must_not_reach_model": 99_999_999,
                            },
                        }
                    ],
                }
            )
        result_body = {
            "plan_sha256": plan["plan_sha256"],
            "coordinate_id": f"{coordinate:064x}",
            "family_id": f"family-{coordinate}",
            "cost_scenario": "BASE" if coordinate % 2 == 0 else "STRESS",
            "classification": "EXPERIMENTAL_UNRANKED",
            "authority": dict(AUTHORITY),
            "cadence_rows": rows,
        }
        result_body["result_sha256"] = canonical_sha256(result_body)
        results.append(result_body)
    return plan, results


def test_queue_exposes_one_packet_without_outcome_or_authority(tmp_path: Path) -> None:
    source_plan, results = _fixtures()
    status = initialize_queue(
        queue_dir=tmp_path,
        source_plan=source_plan,
        result_values=results,
    )

    assert status["state"] == "WAITING_FOR_MODEL"
    assert status["accepted_model_decision_count"] == 0
    assert len(list((tmp_path / "ready").glob("*.json"))) == 1
    packet = current_ready_packet(tmp_path)
    serialized = json.dumps(packet, sort_keys=True)
    assert "must_not_reach_model" not in serialized
    assert "post_outcome" not in serialized
    assert packet["terminal_result_allowed"] is False
    assert packet["future_quote_allowed"] is False
    assert packet["authority"]["order_authority"] == "NONE"


def test_verified_response_advances_automatically_to_next_cell(
    tmp_path: Path,
) -> None:
    source_plan, results = _fixtures()
    initial = initialize_queue(
        queue_dir=tmp_path,
        source_plan=source_plan,
        result_values=results,
    )
    first_sha = initial["current_ready_packet_sha256"]
    response = seal_model_response(
        packet=current_ready_packet(tmp_path),
        action="HOLD",
        reason_ids=["NO_MATERIAL_RISK_CHANGE"],
        provider_model="codex-test-model",
        provider_execution_id="test-execution-1",
    )

    advanced = submit_model_response(
        queue_dir=tmp_path,
        response_value=response,
    )

    assert advanced["accepted_model_decision_count"] == 1
    assert advanced["current_ready_cell"]["cell_ordinal"] == 2
    assert advanced["current_ready_packet_sha256"] != first_sha
    assert advanced["remaining_model_decision_count"] == 83
    assert len(list((tmp_path / "responses").glob("*.json"))) == 1
    assert verify_queue(tmp_path) == advanced


def test_duplicate_response_is_idempotent_and_does_not_consume_next_cell(
    tmp_path: Path,
) -> None:
    source_plan, results = _fixtures()
    initialize_queue(
        queue_dir=tmp_path,
        source_plan=source_plan,
        result_values=results,
    )
    response = seal_model_response(
        packet=current_ready_packet(tmp_path),
        action="HOLD",
        reason_ids=["NO_MATERIAL_RISK_CHANGE"],
        provider_model="codex-test-model",
        provider_execution_id="test-execution-1",
    )
    once = submit_model_response(queue_dir=tmp_path, response_value=response)
    twice = submit_model_response(queue_dir=tmp_path, response_value=response)

    assert twice == once
    assert queue_status(tmp_path)["accepted_model_decision_count"] == 1


def test_response_tamper_and_future_claim_fail_closed(tmp_path: Path) -> None:
    source_plan, results = _fixtures()
    initialize_queue(
        queue_dir=tmp_path,
        source_plan=source_plan,
        result_values=results,
    )
    packet = current_ready_packet(tmp_path)
    response = seal_model_response(
        packet=packet,
        action="HOLD",
        reason_ids=["NO_MATERIAL_RISK_CHANGE"],
        provider_model="codex-test-model",
        provider_execution_id="test-execution-1",
    )
    tampered = copy.deepcopy(response)
    tampered["future_information_used"] = True
    tampered["response_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "response_sha256"}
    )

    with pytest.raises(DojoPairedModelQueueError, match="response seal"):
        verify_model_response(tampered, packet)


def test_model_cannot_choose_order_or_live_authority(tmp_path: Path) -> None:
    source_plan, results = _fixtures()
    initialize_queue(
        queue_dir=tmp_path,
        source_plan=source_plan,
        result_values=results,
    )
    packet = current_ready_packet(tmp_path)
    assert set(packet["action_allowlist"]) == set(ACTION_IDS)
    assert "MARKET" not in packet["action_allowlist"]
    assert "LIMIT" not in packet["action_allowlist"]

    with pytest.raises(DojoPairedModelQueueError, match="allowlist"):
        seal_model_response(
            packet=packet,
            action="MARKET",
            reason_ids=["TRY_TO_TRADE"],
            provider_model="codex-test-model",
            provider_execution_id="test-execution-1",
        )
