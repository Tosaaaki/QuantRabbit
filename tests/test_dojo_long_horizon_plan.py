from __future__ import annotations

import copy
import math

import pytest

from quant_rabbit.dojo_long_horizon_plan import (
    CORE5_PAIRS,
    IMPLEMENTATION_DIGEST_KEYS,
    M1_CORE5_BINDING_ID,
    M1_FULL28_BINDING_ID,
    M5_BINDING_ID,
    M5_MONTHS,
    SOURCE_BINDING_IDS,
    DojoLongHorizonPlanError,
    build_long_horizon_train_plan,
    canonical_sha256,
    validate_long_horizon_train_plan,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


FAMILIES = ("breakout", "range_fade", "spike_fade")


def _digests(keys: tuple[str, ...], *, offset: int) -> dict[str, str]:
    return {key: f"{index + offset:064x}" for index, key in enumerate(keys, start=1)}


def _plan() -> dict:
    return build_long_horizon_train_plan(
        portfolio_families=FAMILIES,
        source_digests=_digests(SOURCE_BINDING_IDS, offset=0),
        corpus_digests=_digests(SOURCE_BINDING_IDS, offset=10),
        implementation_digests=_digests(IMPLEMENTATION_DIGEST_KEYS, offset=20),
    )


def test_builds_exact_m5_and_m1_denominators_with_no_authority() -> None:
    plan = _plan()
    denominator = plan["exact_denominator"]

    assert len(M5_MONTHS) == 78
    assert M5_MONTHS[0] == "2020-01"
    assert M5_MONTHS[-1] == "2026-06"
    assert denominator["m5_pair_count"] == 28
    assert denominator["m5_pair_month_source_cell_count"] == 28 * 78 == 2_184
    assert denominator["base_month_mode_path_scenario_cell_count"] == 78 * 2 * 2 * 2

    rectangles = plan["source_bindings"]["m1_precision_rectangles"]
    assert [row["rectangle_id"] for row in rectangles] == [
        M1_CORE5_BINDING_ID,
        M1_FULL28_BINDING_ID,
    ]
    assert rectangles[0]["pairs"] == list(CORE5_PAIRS)
    assert rectangles[0]["pair_month_cell_count"] == 5 * 78 == 390
    assert rectangles[1]["pairs"] == list(DEFAULT_TRADER_PAIRS)
    assert rectangles[1]["pair_month_cell_count"] == 28 * 18 == 504
    assert denominator["m1_precision_rectangle_pair_month_cell_count"] == 894
    assert denominator["m1_precision_overlap_pair_month_cell_count"] == 90
    assert denominator["m1_precision_unique_pair_month_cell_count"] == 804
    assert denominator["m1_precision_result_cell_count"] == 894 * 2 * 2 * 2

    assert denominator["portfolio_result_cell_count"] == 24_960
    assert denominator["total_required_result_cell_count"] == 32_112
    assert plan["authority"] == {
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "diagnostic_only": True,
        "evidence_tier": "WORN_HISTORICAL_TRAIN_ONLY",
        "forward_proof_eligible": False,
        "historical_only": True,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
        "trainer_may_change_live_configuration": False,
    }
    assert validate_long_horizon_train_plan(plan) == plan


def test_lopo_stages_have_exact_pair_family_and_currency_denominators() -> None:
    plan = _plan()
    stages = {
        row["stage"]: row for row in plan["exact_denominator"]["portfolio_stages"]
    }
    base = 78 * 2 * 2 * 2
    assert stages["PORTFOLIO_MAIN"]["result_cell_count"] == base
    assert stages["PAIR_LOPO"]["labels"] == list(DEFAULT_TRADER_PAIRS)
    assert stages["PAIR_LOPO"]["result_cell_count"] == base * 28
    assert stages["FAMILY_LOPO"]["labels"] == list(FAMILIES)
    assert stages["FAMILY_LOPO"]["result_cell_count"] == base * len(FAMILIES)
    assert stages["CURRENCY_LOPO"]["label_count"] == 8
    assert stages["CURRENCY_LOPO"]["removed_pair_count_per_label"] == 7
    assert stages["CURRENCY_LOPO"]["result_cell_count"] == base * 8
    assert plan["exact_denominator"]["early_stopping_allowed"] is False
    assert (
        plan["exact_denominator"]["missing_or_failed_cell_policy"]
        == "COUNT_IN_DENOMINATOR_AND_FAIL_CLOSED"
    )


def test_monthly_3x_is_an_every_month_diagnostic_not_a_sizing_input() -> None:
    gates = _plan()["monthly_3x_diagnostic_gates"]
    assert gates["target_multiple"] == 3.0
    assert gates["required_month_count"] == 78
    assert gates["required_three_x_month_count"] == 78
    assert gates["required_three_x_hit_rate"] == 1.0
    assert gates["maximum_losing_month_count"] == 0
    assert gates["pessimistic_month_multiple_policy"] == "MIN_OHLC_OLHC_STRESS"
    assert gates["target_is_sizing_input"] is False
    assert gates["target_may_backsolve_risk_or_leverage"] is False
    assert gates["diagnostic_pass_grants_proof"] is False
    assert gates["diagnostic_pass_grants_live_permission"] is False
    assert gates["market_return_guarantee"] is False


def test_independent_and_continuous_account_modes_are_not_interchangeable() -> None:
    modes = {row["mode"]: row for row in _plan()["evaluation"]["modes"]}
    independent = modes["INDEPENDENT_MONTH"]
    continuous = modes["CONTINUOUS_ACCOUNT"]
    assert independent["state_carried_across_months"] is False
    assert independent["terminal_flat_required_each_month"] is True
    assert continuous["state_carried_across_months"] is True
    assert continuous["month_end_scoring_uses_mtm_equity"] is True
    assert continuous["terminal_flat_required_at_period_end"] is True
    assert (
        _plan()["monthly_3x_diagnostic_gates"][
            "continuous_mode_cannot_substitute_for_failed_independent_months"
        ]
        is True
    )


def test_nzd_chf_2024_is_never_imputed_or_silently_removed() -> None:
    plan = _plan()
    policy = plan["missing_data_policy"]
    gap = policy["known_gap_disclosure"]
    assert gap["pair"] == "NZD_CHF"
    assert gap["months"] == [f"2024-{month:02d}" for month in range(1, 13)]
    assert gap["handling"] == "NOT_IN_EITHER_M1_PRECISION_RECTANGLE_AND_NEVER_IMPUTED"
    assert policy["interpolation_allowed"] is False
    assert policy["forward_fill_allowed"] is False
    assert policy["synthetic_candles_allowed"] is False
    assert policy["denominator_reduction_allowed"] is False
    assert "NZD_CHF" not in CORE5_PAIRS
    assert (
        "2024-12" not in plan["source_bindings"]["m1_precision_rectangles"][1]["months"]
    )


def test_families_must_be_presealed_unique_sorted_and_bounded() -> None:
    kwargs = {
        "source_digests": _digests(SOURCE_BINDING_IDS, offset=0),
        "corpus_digests": _digests(SOURCE_BINDING_IDS, offset=10),
        "implementation_digests": _digests(IMPLEMENTATION_DIGEST_KEYS, offset=20),
    }
    with pytest.raises(DojoLongHorizonPlanError, match="sorted"):
        build_long_horizon_train_plan(
            portfolio_families=("spike_fade", "breakout"), **kwargs
        )
    with pytest.raises(DojoLongHorizonPlanError, match="between 2 and 32"):
        build_long_horizon_train_plan(portfolio_families=("only_one",), **kwargs)
    with pytest.raises(DojoLongHorizonPlanError, match="canonical identifiers"):
        build_long_horizon_train_plan(
            portfolio_families=("breakout", "Range Fade"), **kwargs
        )


def test_digest_maps_are_exact_and_lowercase_sha256() -> None:
    sources = _digests(SOURCE_BINDING_IDS, offset=0)
    corpora = _digests(SOURCE_BINDING_IDS, offset=10)
    implementations = _digests(IMPLEMENTATION_DIGEST_KEYS, offset=20)
    del sources[M5_BINDING_ID]
    with pytest.raises(DojoLongHorizonPlanError, match="exactly"):
        build_long_horizon_train_plan(
            portfolio_families=FAMILIES,
            source_digests=sources,
            corpus_digests=corpora,
            implementation_digests=implementations,
        )

    sources[M5_BINDING_ID] = "A" * 64
    with pytest.raises(DojoLongHorizonPlanError, match="lowercase SHA-256"):
        build_long_horizon_train_plan(
            portfolio_families=FAMILIES,
            source_digests=sources,
            corpus_digests=corpora,
            implementation_digests=implementations,
        )


@pytest.mark.parametrize(
    "digest_group,digest_key",
    [
        ("source_digests", M5_BINDING_ID),
        ("corpus_digests", M1_CORE5_BINDING_ID),
        ("implementation_digests", "scorer_sha256"),
    ],
)
def test_all_zero_placeholder_cannot_be_sealed_as_a_real_digest(
    digest_group: str, digest_key: str
) -> None:
    kwargs = {
        "portfolio_families": FAMILIES,
        "source_digests": _digests(SOURCE_BINDING_IDS, offset=0),
        "corpus_digests": _digests(SOURCE_BINDING_IDS, offset=10),
        "implementation_digests": _digests(IMPLEMENTATION_DIGEST_KEYS, offset=20),
    }
    kwargs[digest_group][digest_key] = "0" * 64

    with pytest.raises(DojoLongHorizonPlanError, match="non-zero lowercase SHA-256"):
        build_long_horizon_train_plan(**kwargs)


def test_rehashed_tampering_still_fails_structural_rebuild() -> None:
    plan = _plan()
    tampered = copy.deepcopy(plan)
    tampered["authority"]["live_permission"] = True
    body = {key: value for key, value in tampered.items() if key != "plan_sha256"}
    tampered["plan_sha256"] = canonical_sha256(body)
    with pytest.raises(DojoLongHorizonPlanError, match="drifted"):
        validate_long_horizon_train_plan(tampered)

    extra = copy.deepcopy(plan)
    extra["unsealed_note"] = "looks harmless"
    with pytest.raises(DojoLongHorizonPlanError, match="top-level schema"):
        validate_long_horizon_train_plan(extra)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_canonical_hash_rejects_nonfinite_numbers(bad: float) -> None:
    with pytest.raises(DojoLongHorizonPlanError, match="non-finite"):
        canonical_sha256({"value": bad})
