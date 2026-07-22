from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

import pytest

from quant_rabbit.dojo_anomaly_admission_runtime import EVIDENCE_SUMMARY_CONTRACT
from quant_rabbit.dojo_anomaly_portfolio_counterfactual import (
    COUNTERFACTUAL_CONTRACT,
    DojoAnomalyPortfolioCounterfactualError,
    score_anomaly_arm_portfolio_counterfactual,
)
from quant_rabbit.dojo_long_horizon_economic_runner import (
    ECONOMIC_JOB_RESULT_CONTRACT,
    ECONOMIC_TRANSCRIPT_V1_JSONL,
)
from quant_rabbit.dojo_portfolio_replay_reducer import (
    MONTH_END_MTM_WITH_STATE_HANDOFF,
    PortfolioReplaySession,
    canonical_portfolio_sha256,
    quote_batch_sha256,
    seal_portfolio_policy,
)
from quant_rabbit.dojo_shared_worker_protocol import (
    seal_worker_proposal,
    seal_worker_proposal_batch,
)


COORDINATE_ID = "2020-01|OHLC|NORMAL|portfolio-a"
WORKER = {
    "worker_id": "worker-a",
    "owner_id": "owner-a",
    "family_id": "family-a",
    "config_sha256": "a" * 64,
}


def _portfolio_result() -> dict[str, Any]:
    policy = seal_portfolio_policy(
        {
            "policy_id": "counterfactual-test-policy",
            "expected_quote_pairs": ["USD_JPY"],
            "tradable_pairs": ["USD_JPY"],
            "active_worker_bindings": [WORKER],
            "leverage": 20,
            "margin_closeout_fraction": 0.9,
            "max_margin_utilization_fraction": 0.8,
            "max_portfolio_stop_risk_fraction": 0.8,
            "max_open_and_pending_total": 4,
            "max_open_and_pending_per_pair": 2,
            "max_open_and_pending_per_family": 4,
            "max_currency_gross_notional_fraction": 100.0,
            "max_cluster_gross_notional_fraction": 100.0,
            "max_lock_seconds": 86_400,
            "slippage_by_pair": [
                {
                    "pair": "USD_JPY",
                    "entry_slippage_price": 0.01,
                    "exit_slippage_price": 0.01,
                }
            ],
            "financing_by_pair": [
                {
                    "pair": "USD_JPY",
                    "long_cost_jpy_per_unit_day": 0.0,
                    "short_cost_jpy_per_unit_day": 0.0,
                }
            ],
            "conversion_routes": [],
            "correlation_bindings": [],
        }
    )
    session = PortfolioReplaySession(policy=policy, initial_balance_jpy=200_000)
    epoch = 1_577_836_800
    quotes = [
        {
            "pair": "USD_JPY",
            "bid": 145.0,
            "ask": 145.02,
            "timestamp": f"{datetime.fromtimestamp(epoch, timezone.utc).isoformat()}#O",
        }
    ]
    snapshot = session.prepare_coordinate(
        coordinate_id=COORDINATE_ID,
        epoch=epoch,
        phase="O",
        intrabar="OHLC",
        quote_watermark=1,
        quotes=quotes,
        quote_batch_sha256_value=quote_batch_sha256(
            epoch=epoch,
            phase="O",
            intrabar="OHLC",
            quote_watermark=1,
            quotes=quotes,
        ),
    )
    proposal = seal_worker_proposal(
        snapshot,
        {
            **WORKER,
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "risk_reducing_intents": [],
            "new_risk_intents": [],
        },
    )
    session.consume_proposal_batch(seal_worker_proposal_batch(snapshot, [proposal]))
    return session.finalize(terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF)


def _runtime_evidence(*, arm: str, runtime_sha: str, held: int) -> dict[str, Any]:
    body = {
        "contract": EVIDENCE_SUMMARY_CONTRACT,
        "schema_version": 1,
        "runtime_binding_sha256": runtime_sha,
        "policy_sha256": "7" * 64,
        "arm": arm,
        "counts": {
            "decisions": 1,
            "upstream_candidates": 1,
            "selected": 1 - held,
            "held": held,
            "reduced": 0,
        },
        "evidence_chain_sha256": "8" * 64,
        "counterfactual_tail": [],
        "counterfactual_tail_truncated": False,
        "runner_integration_complete": True,
        "independent_counterfactual_reexecution_complete": False,
        "official_evidence_eligible": False,
        "authority": {"research_only": True, "live_permission": False},
    }
    return {**body, "evidence_summary_sha256": canonical_portfolio_sha256(body)}


def _job_result(*, arm: str, runtime_sha: str, held: int) -> dict[str, Any]:
    fixed_body = {
        "contract": "QR_DOJO_ANOMALY_DECISION_FIXED_DENOMINATOR_ATTESTATION_V1",
        "schema_version": 1,
        "status": "VERIFIED_COMPLETE",
        "expected_coordinate_count": 1,
        "verified_coordinate_count": 1,
        "expected_coordinate_ids_sha256": canonical_portfolio_sha256(
            [COORDINATE_ID]
        ),
        "decision_reexecution_attestation_sha256_by_coordinate": {
            COORDINATE_ID: "e" * 64
        },
        "economic_reexecution_attestation_sha256_by_coordinate": {
            COORDINATE_ID: "f" * 64
        },
        "fixed_denominator_decision_reexecution_passed": True,
        "held_economic_counterfactual_reexecution_passed": False,
        "partial_decision_evidence_reported": False,
    }
    fixed = {
        **fixed_body,
        "fixed_denominator_decision_attestation_sha256": (
            canonical_portfolio_sha256(fixed_body)
        ),
    }
    body = {
        "contract": ECONOMIC_JOB_RESULT_CONTRACT,
        "schema_version": 1,
        "runner_handoff_sha256": "1" * 64,
        "plan_sha256": "2" * 64,
        "implementation_binding_sha256": "3" * 64,
        "job_sha256": "4" * 64,
        "claim_sha256": "5" * 64,
        "source_slice_receipt_sha256": "6" * 64,
        "source_row_count": 1,
        "quote_batch_count": 1,
        "batch_chain_sha256": "9" * 64,
        "coordinate_runtime_bindings_sha256": "a" * 64,
        "coordinate_results": [{"coordinate_id": COORDINATE_ID}],
        "coordinate_result_count": 1,
        "complete_coordinate_count": 1,
        "failed_coordinate_count": 0,
        "job_status": "COMPLETE",
        "portfolio_results_by_coordinate": {COORDINATE_ID: _portfolio_result()},
        "worker_runtime_binding_sha256": runtime_sha,
        "worker_runtime_mode": "SEALED_ANOMALY_ADMISSION_OVER_TUNED_STRATEGY",
        "worker_runtime_arm": arm,
        "worker_runtime_capacity_slots": 4,
        "worker_runtime_policy_sha256": "7" * 64,
        "worker_upstream_runtime_binding_sha256": "b" * 64,
        "worker_runtime_evidence_by_coordinate": {
            COORDINATE_ID: _runtime_evidence(
                arm=arm, runtime_sha=runtime_sha, held=held
            )
        },
        "strategy_decision_reexecution_passed": True,
        "fixed_denominator_decision_reexecution_attestation": fixed,
        "fixed_denominator_decision_reexecution_attestation_sha256": fixed[
            "fixed_denominator_decision_attestation_sha256"
        ],
        "partial_economics_reported": False,
        "downstream_terminal_reduction_allowed": True,
        "independent_economic_reexecution_passed": True,
        "source_quote_coverage_proved": True,
        "economic_transcript_format": ECONOMIC_TRANSCRIPT_V1_JSONL,
        "sparse_observed_epoch_union_used": False,
        "synthetic_executable_quote_count": 0,
        "carry_forward_executable_quote_count": 0,
    }
    return {**body, "economic_job_result_sha256": canonical_portfolio_sha256(body)}


def _reseal(result: dict[str, Any]) -> None:
    result["economic_job_result_sha256"] = canonical_portfolio_sha256(
        {
            key: value
            for key, value in result.items()
            if key != "economic_job_result_sha256"
        }
    )


def test_scores_only_complete_paired_accounts_without_summing_grid_profit() -> None:
    baseline = _job_result(arm="BASE_BOT", runtime_sha="c" * 64, held=0)
    candidate = _job_result(
        arm="COMBINED_ANOMALY_ADMISSION", runtime_sha="d" * 64, held=1
    )

    result = score_anomaly_arm_portfolio_counterfactual(
        baseline_result=baseline,
        candidate_result=candidate,
    )

    assert result["contract"] == COUNTERFACTUAL_CONTRACT
    assert result["held_economic_counterfactual_reexecution_passed"] is True
    assert result["coordinate_count"] == 1
    assert result["candidate_equal_coordinate_count"] == 1
    assert result["coordinate_grid_profit_sum_allowed"] is False
    assert result["monthly_three_x_proven"] is False
    assert result["official_evidence_eligible"] is False
    assert result["authority"]["live_permission"] is False


def test_rejects_resealed_source_or_denominator_drift() -> None:
    baseline = _job_result(arm="BASE_BOT", runtime_sha="c" * 64, held=0)
    candidate = _job_result(
        arm="COMBINED_ANOMALY_ADMISSION", runtime_sha="d" * 64, held=1
    )
    drifted_source = copy.deepcopy(candidate)
    drifted_source["batch_chain_sha256"] = "e" * 64
    _reseal(drifted_source)
    with pytest.raises(
        DojoAnomalyPortfolioCounterfactualError,
        match="batch_chain_sha256",
    ):
        score_anomaly_arm_portfolio_counterfactual(
            baseline_result=baseline,
            candidate_result=drifted_source,
        )

    missing = copy.deepcopy(candidate)
    missing["worker_runtime_evidence_by_coordinate"] = {}
    _reseal(missing)
    with pytest.raises(
        DojoAnomalyPortfolioCounterfactualError,
        match="fixed denominator",
    ):
        score_anomaly_arm_portfolio_counterfactual(
            baseline_result=baseline,
            candidate_result=missing,
        )
