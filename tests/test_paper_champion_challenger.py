from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from quant_rabbit.paper_champion_challenger import (
    PaperExperimentPolicyError,
    assess_candidate_admission,
    assess_continuation,
    candidate_hash,
    generate_strategy_candidate,
)


POLICY_PATH = Path("config/paper_champion_challenger_policy_v1.json")


def _policy() -> dict:
    return json.loads(POLICY_PATH.read_text())


def _candidate() -> dict:
    return {
        "strategy_template_id": "range-sibling-v1",
        "parameters": {"atr_bucket": "LOW"},
        "virtual_capital_jpy": 50_000,
        "max_drawdown_fraction": 0.05,
        "duration_days": 14,
        "shared_feed_contract_sha256": "a" * 64,
        "virtual_account_id": "account-1",
        "inventory_id": "inventory-1",
        "order_book_id": "orders-1",
        "ledger_id": "ledger-1",
        "risk_budget_id": "risk-1",
        "future_data_allowed": False,
        "authority": {
            "live_permission": False,
            "broker_mutation_allowed": False,
            "order_authority": "NONE",
        },
    }


def test_admission_is_paper_only_deterministic_and_idempotent() -> None:
    kwargs = {
        "policy": _policy(),
        "candidate": _candidate(),
        "registry": {
            "active_candidates": [],
            "candidate_hashes": [],
            "accepted_data_hashes": [],
            "admitted_at_utc": [],
        },
        "evidence_data_hash": "b" * 64,
        "observed_at_utc": "2026-07-28T07:00:00Z",
    }
    first = assess_candidate_admission(**kwargs)
    second = assess_candidate_admission(**kwargs)

    assert first == second
    assert first["status"] == "ADMIT_PAPER_SHADOW"
    assert first["authority"]["order_authority"] == "NONE"


def test_admission_rejects_duplicate_data_without_new_decision() -> None:
    result = assess_candidate_admission(
        policy=_policy(),
        candidate=_candidate(),
        registry={
            "active_candidates": [],
            "candidate_hashes": [],
            "accepted_data_hashes": ["b" * 64],
            "admitted_at_utc": [],
        },
        evidence_data_hash="b" * 64,
        observed_at_utc="2026-07-28T07:00:00Z",
    )
    assert result["status"] == "REJECT_DUPLICATE_DATA_HASH"


def test_admission_fails_closed_on_live_authority_or_hash_mismatch() -> None:
    live = _candidate()
    live["authority"]["live_permission"] = True
    with pytest.raises(PaperExperimentPolicyError, match="Paper-only"):
        assess_candidate_admission(
            policy=_policy(),
            candidate=live,
            registry={"active_candidates": []},
            evidence_data_hash="b" * 64,
            observed_at_utc="2026-07-28T07:00:00Z",
        )

    bad_hash = _candidate()
    bad_hash["candidate_hash"] = "0" * 64
    with pytest.raises(PaperExperimentPolicyError, match="hash mismatch"):
        assess_candidate_admission(
            policy=_policy(),
            candidate=bad_hash,
            registry={"active_candidates": []},
            evidence_data_hash="b" * 64,
            observed_at_utc="2026-07-28T07:00:00Z",
        )


def test_admission_requires_complete_isolation() -> None:
    candidate = _candidate()
    candidate["ledger_id"] = candidate["inventory_id"]
    with pytest.raises(PaperExperimentPolicyError, match="distinct"):
        assess_candidate_admission(
            policy=_policy(),
            candidate=candidate,
            registry={"active_candidates": []},
            evidence_data_hash="b" * 64,
            observed_at_utc="2026-07-28T07:00:00Z",
        )


def test_admission_rejects_invalid_hash_and_future_registry_time() -> None:
    with pytest.raises(PaperExperimentPolicyError, match="sha256"):
        assess_candidate_admission(
            policy=_policy(),
            candidate=_candidate(),
            registry={"active_candidates": []},
            evidence_data_hash="z" * 64,
            observed_at_utc="2026-07-28T07:00:00Z",
        )

    with pytest.raises(PaperExperimentPolicyError, match="future admission"):
        assess_candidate_admission(
            policy=_policy(),
            candidate=_candidate(),
            registry={
                "active_candidates": [],
                "admitted_at_utc": ["2026-07-28T08:00:00Z"],
            },
            evidence_data_hash="b" * 64,
            observed_at_utc="2026-07-28T07:00:00Z",
        )


def test_continuation_requires_profit_cost_dd_regime_and_same_feed() -> None:
    passing = {
        "settlements": 30,
        "profit_factor_after_cost": 1.1,
        "expectancy_after_cost_jpy": 1.0,
        "max_drawdown_fraction": 0.02,
        "champion_max_drawdown_fraction": 0.02,
        "profitable_regime_ids": ["TREND", "RANGE"],
        "base_stress_same_direction": True,
        "shared_feed_chain_match": True,
    }
    assert (
        assess_continuation(policy=_policy(), metrics=passing)["status"]
        == "CONTINUE_PAPER_SHADOW"
    )

    failing = copy.deepcopy(passing)
    failing["expectancy_after_cost_jpy"] = 0.0
    failing["shared_feed_chain_match"] = False
    stopped = assess_continuation(policy=_policy(), metrics=failing)
    assert stopped["status"] == "STOP_PAPER_SHADOW"
    assert "EXPECTANCY_NOT_POSITIVE" in stopped["reason_ids"]
    assert "SHARED_FEED_CHAIN_MISMATCH" in stopped["reason_ids"]

    missing = assess_continuation(policy=_policy(), metrics={})
    assert missing["status"] == "STOP_PAPER_SHADOW"
    assert "INSUFFICIENT_SETTLEMENTS" in missing["reason_ids"]


def test_candidate_hash_ignores_only_its_seal_field() -> None:
    candidate = _candidate()
    sealed = candidate_hash(candidate)
    candidate["candidate_hash"] = sealed
    assert candidate_hash(candidate) == sealed


def test_strategy_lab_generates_reviewed_pullback_sibling_from_new_loss_evidence() -> None:
    result = generate_strategy_candidate(
        policy=_policy(),
        evidence={
            "data_hash": "c" * 64,
            "completed_observations_only": True,
            "future_or_terminal_data_in_decision": False,
            "pairs": ["USD_JPY"],
            "ranked_causes": [
                {
                    "cause_id": "COUNTERTREND_SHORT_CONCENTRATION",
                    "settlements": 49,
                    "net_contribution_jpy": -852.88,
                    "confidence": "HIGH",
                }
            ],
        },
        registry={"reviewed_data_hashes": []},
        observed_at_utc="2026-07-28T08:00:00Z",
    )
    assert result["status"] == "CANDIDATE_PROPOSED"
    candidate = result["candidate"]
    assert candidate["bot_config"]["signal"] == "pullback_limit"
    assert candidate["bot_config"]["global_max_concurrent"] == 1
    assert candidate["authority"]["order_authority"] == "NONE"
    assert candidate_hash(candidate) == candidate["candidate_hash"]


def test_strategy_lab_rejects_duplicate_or_weak_evidence() -> None:
    evidence = {
        "data_hash": "c" * 64,
        "completed_observations_only": True,
        "future_or_terminal_data_in_decision": False,
        "ranked_causes": [
            {
                "cause_id": "COUNTERTREND_SHORT_CONCENTRATION",
                "settlements": 3,
                "net_contribution_jpy": -20,
                "confidence": "HIGH",
            }
        ],
    }
    duplicate = generate_strategy_candidate(
        policy=_policy(),
        evidence=evidence,
        registry={"reviewed_data_hashes": ["c" * 64]},
        observed_at_utc="2026-07-28T08:00:00Z",
    )
    assert duplicate["status"] == "NO_NEW_EVIDENCE"

    weak = generate_strategy_candidate(
        policy=_policy(),
        evidence=evidence,
        registry={"reviewed_data_hashes": []},
        observed_at_utc="2026-07-28T08:00:00Z",
    )
    assert weak["status"] == "INSUFFICIENT_EVIDENCE"
