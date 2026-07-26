from __future__ import annotations

import pytest

from quant_rabbit.dojo_paired_inventory_counterfactual import (
    ACTION_IDS,
    CADENCE_IDS,
    DojoPairedInventoryCounterfactualError,
    _calibrated_thresholds,
    _position_values,
    _profit_metrics,
    build_paired_inventory_plan,
)


def _plan() -> dict:
    start = 1_700_000_000
    width = 10_000
    return build_paired_inventory_plan(
        study_id="paired-inventory-test",
        source_job_sha256="a" * 64,
        source_job_result_sha256="b" * 64,
        transcript_sha256_by_coordinate={"coordinate-a": "c" * 64},
        calibration_start_epoch=start,
        calibration_end_epoch=start + width,
        oos_blocks=[
            {
                "block_id": "ignored",
                "start_epoch": start + width * (index + 1),
                "end_epoch": start + width * (index + 2),
            }
            for index in range(8)
        ],
        source_quote_coverage_proved=False,
        researcher_prior_aggregate_outcome_exposure=True,
    )


def test_plan_fixes_all_cadences_actions_and_zero_authority() -> None:
    plan = _plan()

    assert plan["cadence_ids"] == list(CADENCE_IDS)
    assert plan["action_ids"] == list(ACTION_IDS)
    assert plan["authority"]["live_permission"] is False
    assert plan["authority"]["broker_mutation_allowed"] is False
    assert plan["authority"]["order_authority"] == "NONE"
    assert plan["actual_model_checkpoint_call_required_for_rank"] is True


def test_plan_marks_worn_incomplete_source_as_experimental() -> None:
    plan = _plan()

    assert plan["source_quote_coverage_proved"] is False
    assert plan["researcher_prior_aggregate_outcome_exposure"] is True
    assert plan["classification"] == "EXPERIMENTAL_WORN_TRAIN"
    assert plan["future_quote_allowed"] is False
    assert plan["terminal_result_allowed_in_decision"] is False


def test_plan_requires_exactly_eight_contiguous_oos_blocks() -> None:
    start = 1_700_000_000
    with pytest.raises(
        DojoPairedInventoryCounterfactualError, match="exactly eight"
    ):
        build_paired_inventory_plan(
            study_id="bad",
            source_job_sha256="a" * 64,
            source_job_result_sha256="b" * 64,
            transcript_sha256_by_coordinate={"a": "c" * 64},
            calibration_start_epoch=start,
            calibration_end_epoch=start + 100,
            oos_blocks=[],
            source_quote_coverage_proved=False,
            researcher_prior_aggregate_outcome_exposure=True,
        )


def test_plan_rejects_oos_gap() -> None:
    start = 1_700_000_000
    blocks = [
        {
            "block_id": "",
            "start_epoch": start + 100 + index * 100,
            "end_epoch": start + 200 + index * 100,
        }
        for index in range(8)
    ]
    blocks[4]["start_epoch"] += 1
    with pytest.raises(DojoPairedInventoryCounterfactualError, match="contiguous"):
        build_paired_inventory_plan(
            study_id="bad-gap",
            source_job_sha256="a" * 64,
            source_job_result_sha256="b" * 64,
            transcript_sha256_by_coordinate={"a": "c" * 64},
            calibration_start_epoch=start,
            calibration_end_epoch=start + 100,
            oos_blocks=blocks,
            source_quote_coverage_proved=False,
            researcher_prior_aggregate_outcome_exposure=True,
        )


def test_calibration_uses_state_distribution_with_ex_ante_floors() -> None:
    samples = [
        {
            "drawdown_fraction": value,
            "margin_utilization_fraction": value / 2,
            "gross_exposure_jpy": value * 100_000,
            "realized_profit_giveback_jpy": value * 10_000,
        }
        for value in (0.0, 0.01, 0.02, 0.04, 0.08)
    ]

    thresholds = _calibrated_thresholds(samples)

    assert thresholds["pause_drawdown_fraction"] >= 0.03
    assert thresholds["close_drawdown_fraction"] >= 0.06
    assert thresholds["reduce_margin_fraction"] >= 0.08
    assert thresholds["close_margin_fraction"] >= 0.16
    assert thresholds["gross_spike_jpy"] >= 20_000
    assert thresholds["profit_giveback_jpy"] >= 4_000


def test_profit_metrics_preserve_gross_wins_losses_and_expectancy() -> None:
    result = _profit_metrics([100.0, -50.0, 25.0, -25.0])

    assert result["close_event_count"] == 4
    assert result["win_rate"] == 0.5
    assert result["profit_factor"] == pytest.approx(125 / 75)
    assert result["expectancy_jpy_per_close"] == pytest.approx(12.5)
    assert result["gross_profit_jpy"] == 125
    assert result["gross_loss_jpy"] == 75


def test_profit_metrics_do_not_invent_values_for_zero_closes() -> None:
    result = _profit_metrics([])

    assert result["win_rate"] is None
    assert result["profit_factor"] is None
    assert result["expectancy_jpy_per_close"] is None


def test_sparse_packet_valuation_uses_only_supplied_last_observed_real_mark() -> None:
    snapshot = {
        "quotes": [
            {
                "pair": "EUR_USD",
                "bid": 1.10,
                "ask": 1.11,
                "timestamp": "2025-01-01T00:00:00+00:00#C",
            }
        ],
        "positions": [
            {
                "position_id": "position-1",
                "worker_id": "worker-1",
                "owner_id": "owner-1",
                "family_id": "family-1",
                "pair": "EUR_USD",
                "side": "LONG",
                "units": 100.0,
                "entry_price": 1.0,
                "tp_price": 2.0,
                "sl_price": 0.5,
                "opened_epoch": 1,
                "hard_exit_epoch": 2,
            }
        ],
    }
    policy = {
        "conversion_routes": [
            {
                "currency": "USD",
                "pair": "USD_JPY",
                "orientation": "JPY_PER_CURRENCY",
            }
        ]
    }
    valuation_quotes = {
        "EUR_USD": snapshot["quotes"][0],
        "USD_JPY": {
            "pair": "USD_JPY",
            "bid": 150.0,
            "ask": 150.1,
            "timestamp": "2024-12-31T23:55:00+00:00#C",
        },
    }

    values = _position_values(
        snapshot,
        policy,
        valuation_quotes=valuation_quotes,
    )

    assert values[0]["unrealized_pnl_jpy"] == pytest.approx(1_500)
