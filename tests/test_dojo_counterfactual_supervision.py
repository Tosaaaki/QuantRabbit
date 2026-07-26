from __future__ import annotations

import copy

import pytest

from quant_rabbit.dojo_counterfactual_supervision import (
    BASELINE_CADENCE_ID,
    BOT_ONLY,
    CADENCE_IDS,
    DojoCounterfactualSupervisionError,
    build_counterfactual_supervision_plan,
    evaluate_counterfactual_supervision,
    seal_counterfactual_supervision_cell,
)


def _plan(*, oos_blocks: int = 8) -> dict:
    start = 1_700_000_000
    width = 86_400
    windows = [
        {
            "window_id": "train-00",
            "partition": "TRAIN",
            "start_epoch": start,
            "end_epoch": start + width,
            "source_slice_sha256": "1" * 64,
        }
    ]
    for index in range(oos_blocks):
        window_start = start + width * (index + 1)
        windows.append(
            {
                "window_id": f"oos-{index:02d}",
                "partition": "OOS",
                "start_epoch": window_start,
                "end_epoch": window_start + width,
                "source_slice_sha256": f"{index + 2:x}" * 64,
            }
        )
    return build_counterfactual_supervision_plan(
        study_id="six-month-supervision-cadence-v1",
        sealed_before_epoch=start,
        market_windows=windows,
        source_manifest_sha256="a" * 64,
        initial_capital_jpy=200_000,
        cost_model_sha256="b" * 64,
        execution_model_sha256="c" * 64,
        supervisor_policy_sha256="d" * 64,
        strategy_compatible_resume_signal_ids=[
            "strategy-regime-compatible-v1"
        ],
        regime_ids=["RANGE", "TREND", "VOLATILE"],
    )


def _decision(cadence_id: str, *, decision_epoch: int) -> dict:
    if cadence_id.startswith("FIXED_"):
        trigger = "FIXED"
        event_ids = []
    elif cadence_id == "EVENT_DRIVEN":
        trigger = "MAJOR_EVENT"
        event_ids = ["DRAWDOWN_DETERIORATION"]
    else:
        trigger = "HEARTBEAT_60M"
        event_ids = []
    return {
        "decision_id": "decision-0001",
        "decision_epoch": decision_epoch,
        "input_available_through_epoch": decision_epoch,
        "observation_sha256": "e" * 64,
        "action": "CONTINUE",
        "trigger_kind": trigger,
        "event_signal_ids": event_ids,
        "market_open": True,
        "has_open_position_or_pending_order": False,
        "resume_signal_ids": [],
        "regime_id": "RANGE",
        "margin_usage_fraction": 0.1,
        "net_exposure_jpy": 10_000.0,
        "gross_exposure_jpy": 10_000.0,
        "drawdown_fraction": 0.01,
        "position_age_seconds": 900,
        "consecutive_losses": 0,
        "strategy_thesis_valid": True,
    }


def _cell(
    plan: dict,
    *,
    window_id: str,
    cadence_id: str | None,
    improve: bool = True,
) -> dict:
    window = next(
        row for row in plan["market_windows"] if row["window_id"] == window_id
    )
    bot = cadence_id is None
    calls = 0 if bot else 1
    metrics = {
        "net_pnl_jpy": 0.0 if bot else (100.0 if improve else -100.0),
        "max_drawdown_fraction": 0.2 if bot else (0.1 if improve else 0.3),
        "peak_margin_usage_fraction": 0.4,
        "margin_call_count": 0,
        "ruin_event_count": 0,
        "turnover_jpy": 20_000.0,
        "mean_gross_exposure_jpy": 10_000.0,
        "loss_avoidance_jpy": 0.0 if bot else 100.0,
        "opportunity_loss_jpy": 0.0,
        "ai_call_count": calls,
        "ai_token_count": calls * 100,
        "ai_latency_ms_total": calls * 250.0,
        "stop_count": 0,
        "resume_count": 0,
    }
    return {
        "window_id": window_id,
        "arm": BOT_ONLY if bot else "AI_MANAGED",
        "cadence_id": cadence_id,
        "source_slice_sha256": window["source_slice_sha256"],
        "status": "COMPLETE",
        "economic_transcript_sha256": "f" * 64,
        "orders_sha256": "1" * 64,
        "fills_sha256": "2" * 64,
        "inventory_sha256": "3" * 64,
        "evidence_counts": {
            "orders": 1,
            "fills": 1,
            "tp_exits": 0,
            "sl_exits": 0,
            "inventory_snapshots": 2,
            "margin_snapshots": 2,
            "unrealized_pnl_snapshots": 2,
            "realized_pnl_events": 1,
        },
        "metrics": metrics,
        "regime_net_pnl_jpy": {
            "RANGE": metrics["net_pnl_jpy"],
            "TREND": 0.0,
            "VOLATILE": 0.0,
        },
        "decisions": (
            []
            if bot
            else [
                _decision(
                    str(cadence_id),
                    decision_epoch=int(window["start_epoch"]) + 100,
                )
            ]
        ),
    }


def _complete_cells(plan: dict) -> list[dict]:
    cells = []
    for window in plan["market_windows"]:
        cells.append(_cell(plan, window_id=window["window_id"], cadence_id=None))
        cells.extend(
            _cell(
                plan,
                window_id=window["window_id"],
                cadence_id=cadence_id,
            )
            for cadence_id in CADENCE_IDS
        )
    return cells


def test_plan_preregisters_requested_cadences_and_adaptive_baseline() -> None:
    plan = _plan()

    assert plan["phase_b_cadence_ids"] == list(CADENCE_IDS)
    assert plan["baseline_cadence_id"] == BASELINE_CADENCE_ID
    adaptive = next(
        row
        for row in plan["cadence_policies"]
        if row["cadence_id"] == BASELINE_CADENCE_ID
    )
    assert adaptive["normal_heartbeat_seconds"] == 3_600
    assert adaptive["high_risk_interval_seconds"] == 900
    assert adaptive["major_event_immediate"] is True
    assert plan["closed_market_call_policy"].endswith(
        "OPEN_POSITION_OR_PENDING_ORDER"
    )
    assert plan["authority"]["order_authority"] == "NONE"


def test_missing_cells_and_too_few_oos_blocks_remain_unranked() -> None:
    plan = _plan(oos_blocks=1)
    result = evaluate_counterfactual_supervision(plan=plan, cells=[])

    assert result["status"] == "UNRANKED"
    assert "FIXED_DENOMINATOR_INCOMPLETE" in result["blockers"]
    assert "INSUFFICIENT_WALK_FORWARD_OOS_BLOCKS" in result["blockers"]
    assert result["selected_paper_shadow_cadence_id"] is None


def test_rejects_future_information_and_unnecessary_closed_market_call() -> None:
    plan = _plan()
    cell = _cell(plan, window_id="train-00", cadence_id="FIXED_60M")
    future = copy.deepcopy(cell)
    future["decisions"][0]["input_available_through_epoch"] += 1
    with pytest.raises(
        DojoCounterfactualSupervisionError, match="future information"
    ):
        seal_counterfactual_supervision_cell(plan=plan, cell=future)

    closed = copy.deepcopy(cell)
    closed["decisions"][0]["market_open"] = False
    with pytest.raises(
        DojoCounterfactualSupervisionError, match="position/order exception"
    ):
        seal_counterfactual_supervision_cell(plan=plan, cell=closed)

    closed["decisions"][0]["has_open_position_or_pending_order"] = True
    sealed = seal_counterfactual_supervision_cell(plan=plan, cell=closed)
    assert sealed["authority"]["live_permission"] is False


def test_resume_requires_preregistered_strategy_compatible_signal() -> None:
    plan = _plan()
    cell = _cell(plan, window_id="train-00", cadence_id="FIXED_60M")
    cell["decisions"][0]["action"] = "RESUME"
    with pytest.raises(
        DojoCounterfactualSupervisionError, match="strategy-compatible"
    ):
        seal_counterfactual_supervision_cell(plan=plan, cell=cell)

    cell["decisions"][0]["resume_signal_ids"] = [
        "strategy-regime-compatible-v1"
    ]
    assert (
        seal_counterfactual_supervision_cell(plan=plan, cell=cell)["cell_sha256"]
    )


def test_complete_paired_oos_denominator_can_select_only_robust_shadow() -> None:
    plan = _plan()
    result = evaluate_counterfactual_supervision(
        plan=plan, cells=_complete_cells(plan)
    )

    assert result["status"] == "RANKED_OOS"
    assert result["selected_paper_shadow_cadence_id"] in CADENCE_IDS
    assert result["paper_shadow_eligible"] is True
    assert result["rank_partition"] == "OOS_ONLY"
    assert result["profit_guaranteed"] is False
    assert result["live_or_broker_authority_granted"] is False
    assert all(row["oos_block_count"] == 8 for row in result["cadence_rows"])


def test_one_missing_losing_cell_cannot_be_dropped_as_a_survivor() -> None:
    plan = _plan()
    cells = _complete_cells(plan)
    cells.pop()
    result = evaluate_counterfactual_supervision(plan=plan, cells=cells)

    assert result["status"] == "UNRANKED"
    assert result["valid_cell_count"] + 1 == result["fixed_expected_cell_count"]
    assert result["blockers"] == ["FIXED_DENOMINATOR_INCOMPLETE"]
