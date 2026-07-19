from __future__ import annotations

from quant_rabbit.dojo_bot_catalog import AUTHORITY_INVARIANTS, validate_bot_config
from quant_rabbit.dojo_bot_trainer import seal_candidate_proposal
from quant_rabbit.dojo_strategy_worker_factory import (
    FAMILIES,
    PAIRS,
    build_baseline_worker_cohort,
)


def test_baseline_cohort_is_stable_and_has_four_distinct_families() -> None:
    first = build_baseline_worker_cohort()
    second = build_baseline_worker_cohort()
    assert first == second
    assert first["candidate_count"] == 4
    assert first["families"] == list(FAMILIES)
    assert first["pairs"] == list(PAIRS)
    proposals = first["trainer_candidate_proposals"]
    assert [row["candidate_id"] for row in proposals] == sorted(
        row["candidate_id"] for row in proposals
    )
    assert {row["family"] for row in proposals} == set(FAMILIES)


def test_every_proposal_uses_the_same_fixed_risk_and_no_authority() -> None:
    cohort = build_baseline_worker_cohort()
    for proposal in cohort["trainer_candidate_proposals"]:
        assert seal_candidate_proposal(proposal) == proposal
        config = validate_bot_config(proposal["config"])
        assert config["pairs"] == list(PAIRS)
        assert config["tp_atr"] == 3.0
        assert config["sl_pips"] == 25.0
        assert config["ceiling_min"] == 60
        assert config["per_pos_lev"] == 5.0
        assert config["global_max_concurrent"] == 4
        assert config["max_concurrent_per_pair"] == 1
        for key, expected in AUTHORITY_INVARIANTS.items():
            assert config[key] == expected


def test_factory_does_not_pretend_to_enforce_portfolio_caps_or_later_tuning() -> None:
    cohort = build_baseline_worker_cohort()
    assert (
        cohort["admission_policy"]["runtime_portfolio_caps_enforced_by_factory"]
        is False
    )
    assert (
        cohort["admission_policy"]["runtime_cap_violation_must_fail_trainer_admission"]
        is True
    )
    assert cohort["later_tuning"]["generation_enabled"] is False
    assert cohort["later_tuning"]["blocked_until"] == [
        "APPEND_ONLY_LINEAGE_REGISTRY",
        "VERIFIED_PRIOR_TRAIN_RESULT_BINDING",
        "NO_REPEAT_CANDIDATE_REGISTRY",
    ]
    assert cohort["authority"] == {
        "research_train_only": True,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
    }
