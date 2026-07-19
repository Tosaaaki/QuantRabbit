from __future__ import annotations

import hashlib
import json
import math

import pytest

from quant_rabbit.dojo_bot_catalog import (
    AUTHORITY_INVARIANTS,
    EXISTING_FAMILIES,
    EXIT_POLICIES,
    NEW_CANDIDATE_FAMILIES,
    RISK_POLICY_CONTRACT,
    RISK_VECTOR_CONTRACT,
    SUPPORTED_FAMILIES,
    DojoBotCatalogError,
    bot_config_risk_vector,
    bot_config_sha256,
    bot_risk_policy_manifest,
    bot_risk_policy_sha256,
    catalog_manifest,
    validate_bot_config,
)


FAMILY_PARAMETERS = {
    "burst": {},
    "pullback_limit": {"pull_atr": 0.6},
    "compression_break": {},
    "spike_fade": {},
    "prev_day_extreme_fade": {},
    "round_number_fade": {},
    "daily_break_pullback": {},
    "mean_revert_24h": {"fade_atr": 1.2},
    "fade_ladder": {"fade_atr": 1.2, "eff_max": 0.2},
    "range_fade_limit": {"fade_atr": 1.2, "eff_max": 0.2},
    "session_open_range_break": {
        "session_buffer_atr": 0.25,
        "session_tp_range": 1.5,
        "session_sl_range": 0.75,
    },
    "weekend_gap_recovery": {
        "weekend_gap_atr": 4.0,
        "weekend_sl_gap": 1.0,
        "weekend_wait_bars": 15,
        "weekend_spread_fraction": 0.35,
    },
}


def _config(family: str) -> dict:
    pair_cap = 2 if family == "fade_ladder" else 1
    config = {
        "signal": family,
        "pairs": ["USD_JPY", "CAD_JPY"],
        "sl_pips": None if family in NEW_CANDIDATE_FAMILIES else 25,
        "ceiling_min": 60,
        "max_concurrent_per_pair": pair_cap,
        "global_max_concurrent": pair_cap * 2,
        "per_pos_lev": 5.0,
        "atr_floor_pips": 0.5,
        **FAMILY_PARAMETERS[family],
    }
    if family not in NEW_CANDIDATE_FAMILIES:
        config["tp_atr"] = 3.0
    return config


def test_catalog_covers_existing_ten_and_two_new_candidate_families() -> None:
    assert len(EXISTING_FAMILIES) == 10
    assert NEW_CANDIDATE_FAMILIES == {
        "session_open_range_break",
        "weekend_gap_recovery",
    }
    assert SUPPORTED_FAMILIES == set(FAMILY_PARAMETERS)

    manifest = catalog_manifest()
    by_id = {row["family_id"]: row for row in manifest["families"]}
    assert set(by_id) == SUPPORTED_FAMILIES
    assert all(
        by_id[family]["implementation_status"] == "LAB_BOT_IMPLEMENTED"
        for family in EXISTING_FAMILIES
    )
    assert all(
        by_id[family]["implementation_status"] == "LAB_BOT_IMPLEMENTED_UNTRAINED"
        for family in NEW_CANDIDATE_FAMILIES
    )
    assert all(
        by_id[family]["common_exit_contract"] == "COMMON_TP_EXACTLY_ONE_SL_OPTIONAL"
        for family in EXISTING_FAMILIES
    )
    assert all(
        by_id[family]["common_exit_contract"]
        == "FAMILY_DYNAMIC_TP_SL_COMMON_FIELDS_NULL"
        for family in NEW_CANDIDATE_FAMILIES
    )


@pytest.mark.parametrize("family", sorted(FAMILY_PARAMETERS))
def test_every_family_accepts_only_its_complete_declarative_shape(family: str) -> None:
    normalized = validate_bot_config(_config(family))

    assert normalized["signal"] == family
    assert normalized["pairs"] == ["CAD_JPY", "USD_JPY"]
    assert normalized["tp_pips"] is None
    assert normalized["tp_atr"] == (None if family in NEW_CANDIDATE_FAMILIES else 3.0)
    assert normalized["sl_pips"] == (None if family in NEW_CANDIDATE_FAMILIES else 25.0)
    assert normalized["exit_policy"] == "FIXED"
    assert {key: normalized[key] for key in AUTHORITY_INVARIANTS} == dict(
        AUTHORITY_INVARIANTS
    )
    assert "max_concurrent" not in normalized


def test_legacy_pair_cap_alias_is_normalized_and_must_agree() -> None:
    config = _config("spike_fade")
    config["max_concurrent"] = config.pop("max_concurrent_per_pair")
    assert validate_bot_config(config)["max_concurrent_per_pair"] == 1

    config["max_concurrent_per_pair"] = 2
    with pytest.raises(DojoBotCatalogError, match="must agree"):
        validate_bot_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("order_authority", "LIVE"),
        ("live_permission", True),
        ("live_permission", 0),
        ("external_broker_mutation_allowed", True),
    ],
)
def test_authority_cannot_be_widened_or_type_coerced(field: str, value: object) -> None:
    config = _config("spike_fade")
    config[field] = value
    with pytest.raises(DojoBotCatalogError, match="cannot widen"):
        validate_bot_config(config)


def test_explicit_safe_authority_is_accepted_and_normalized() -> None:
    config = _config("spike_fade")
    config.update(AUTHORITY_INVARIANTS)
    normalized = validate_bot_config(config)
    assert {key: normalized[key] for key in AUTHORITY_INVARIANTS} == dict(
        AUTHORITY_INVARIANTS
    )


@pytest.mark.parametrize("policy", sorted(EXIT_POLICIES))
def test_exit_policy_accepts_only_its_exact_overlay(policy: str) -> None:
    config = _config("spike_fade")
    config["exit_policy"] = policy
    if policy == "BREAKEVEN":
        config.update({"be_trigger_atr": 1.0, "be_offset_pips": 0.5})
    elif policy == "ATR_TRAILING":
        config.update({"trail_trigger_atr": 1.5, "trail_distance_atr": 1.0})

    normalized = validate_bot_config(config)
    assert normalized["exit_policy"] == policy
    expected_keys = {
        "FIXED": set(),
        "BREAKEVEN": {"be_trigger_atr", "be_offset_pips"},
        "ATR_TRAILING": {"trail_trigger_atr", "trail_distance_atr"},
    }[policy]
    assert expected_keys.issubset(normalized)
    assert not (
        (
            set(normalized)
            & {
                "be_trigger_atr",
                "be_offset_pips",
                "trail_trigger_atr",
                "trail_distance_atr",
            }
        )
        - expected_keys
    )


@pytest.mark.parametrize(
    "policy,extra",
    [
        ("FIXED", {"be_trigger_atr": 1.0}),
        ("BREAKEVEN", {"trail_trigger_atr": 1.0}),
        ("ATR_TRAILING", {"be_offset_pips": 0.5}),
    ],
)
def test_exit_policy_rejects_irrelevant_overlay_parameters(
    policy: str, extra: dict[str, float]
) -> None:
    config = _config("spike_fade")
    config["exit_policy"] = policy
    if policy == "BREAKEVEN":
        config.update({"be_trigger_atr": 1.0, "be_offset_pips": 0.5})
    elif policy == "ATR_TRAILING":
        config.update({"trail_trigger_atr": 1.5, "trail_distance_atr": 1.0})
    config.update(extra)
    with pytest.raises(DojoBotCatalogError, match="exit overlay schema mismatch"):
        validate_bot_config(config)


@pytest.mark.parametrize(
    "policy,missing",
    [
        ("BREAKEVEN", "be_offset_pips"),
        ("ATR_TRAILING", "trail_distance_atr"),
    ],
)
def test_exit_policy_requires_complete_overlay(policy: str, missing: str) -> None:
    config = _config("spike_fade")
    config["exit_policy"] = policy
    if policy == "BREAKEVEN":
        config["be_trigger_atr"] = 1.0
    else:
        config["trail_trigger_atr"] = 1.0
    with pytest.raises(DojoBotCatalogError, match=f"missing=.*{missing}"):
        validate_bot_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("be_trigger_atr", 0.49),
        ("be_trigger_atr", 5.01),
        ("be_offset_pips", -0.01),
        ("be_offset_pips", 2.01),
    ],
)
def test_breakeven_overlay_ranges_are_strict(field: str, value: float) -> None:
    config = _config("spike_fade")
    config.update(
        {
            "exit_policy": "BREAKEVEN",
            "be_trigger_atr": 1.0,
            "be_offset_pips": 0.5,
            field: value,
        }
    )
    with pytest.raises(DojoBotCatalogError, match=field):
        validate_bot_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("trail_trigger_atr", 0.49),
        ("trail_trigger_atr", 5.01),
        ("trail_distance_atr", 0.49),
        ("trail_distance_atr", 5.01),
    ],
)
def test_atr_trailing_overlay_ranges_are_strict(field: str, value: float) -> None:
    config = _config("spike_fade")
    config.update(
        {
            "exit_policy": "ATR_TRAILING",
            "trail_trigger_atr": 1.0,
            "trail_distance_atr": 1.0,
            field: value,
        }
    )
    with pytest.raises(DojoBotCatalogError, match=field):
        validate_bot_config(config)


def test_unknown_and_cross_family_parameter_keys_are_rejected() -> None:
    unknown = _config("spike_fade")
    unknown["python_code"] = "open('/tmp/pwned', 'w')"
    with pytest.raises(DojoBotCatalogError, match="unknown=.*python_code"):
        validate_bot_config(unknown)

    cross_family = _config("burst")
    cross_family["fade_atr"] = 1.0
    with pytest.raises(DojoBotCatalogError, match="unknown=.*fade_atr"):
        validate_bot_config(cross_family)

    owner_claim = _config("burst")
    owner_claim["strategy_owner_id"] = "ai:self-appointed"
    with pytest.raises(DojoBotCatalogError, match="strategy_owner_id"):
        validate_bot_config(owner_claim)


@pytest.mark.parametrize(
    "family,missing",
    [
        ("pullback_limit", "pull_atr"),
        ("fade_ladder", "eff_max"),
        ("session_open_range_break", "session_buffer_atr"),
        ("session_open_range_break", "session_tp_range"),
        ("session_open_range_break", "session_sl_range"),
        ("weekend_gap_recovery", "weekend_gap_atr"),
        ("weekend_gap_recovery", "weekend_sl_gap"),
        ("weekend_gap_recovery", "weekend_wait_bars"),
        ("weekend_gap_recovery", "weekend_spread_fraction"),
    ],
)
def test_family_specific_parameters_are_required(family: str, missing: str) -> None:
    config = _config(family)
    del config[missing]
    with pytest.raises(DojoBotCatalogError, match=f"missing=.*{missing}"):
        validate_bot_config(config)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"tp_atr": None}, "exactly one"),
        ({"tp_pips": 5.0}, "exactly one"),
        ({"tp_atr": True}, "finite number"),
        ({"sl_pips": 0}, "sl_pips must be >"),
        ({"ceiling_min": 60.0}, "integer"),
        ({"max_concurrent_per_pair": False}, "integer"),
        ({"per_pos_lev": 25.1}, "per_pos_lev must be >"),
        ({"atr_floor_pips": math.inf}, "finite number"),
        ({"tp_atr": math.nan}, "finite number"),
    ],
)
def test_numeric_types_finiteness_and_ranges_fail_closed(
    mutation: dict, match: str
) -> None:
    config = _config("spike_fade")
    config.update(mutation)
    with pytest.raises(DojoBotCatalogError, match=match):
        validate_bot_config(config)


def test_take_profit_pips_and_sl_free_configuration_are_explicitly_supported() -> None:
    config = _config("burst")
    config.pop("tp_atr")
    config["tp_pips"] = 6
    config["sl_pips"] = None
    normalized = validate_bot_config(config)
    assert normalized["tp_pips"] == 6.0
    assert normalized["tp_atr"] is None
    assert normalized["sl_pips"] is None


@pytest.mark.parametrize("family", sorted(NEW_CANDIDATE_FAMILIES))
def test_dynamic_target_families_allow_only_omitted_or_null_common_tp(
    family: str,
) -> None:
    omitted = _config(family)
    normalized = validate_bot_config(omitted)
    assert normalized["tp_pips"] is None
    assert normalized["tp_atr"] is None

    explicit_null = _config(family)
    explicit_null.update({"tp_pips": None, "tp_atr": None})
    assert validate_bot_config(explicit_null)["tp_atr"] is None

    for field in ("tp_pips", "tp_atr"):
        configured = _config(family)
        configured[field] = 3.0
        with pytest.raises(DojoBotCatalogError, match="dynamic-target"):
            validate_bot_config(configured)


@pytest.mark.parametrize("family", sorted(NEW_CANDIDATE_FAMILIES))
def test_dynamic_stop_families_require_explicit_null_common_sl(family: str) -> None:
    configured = _config(family)
    assert validate_bot_config(configured)["sl_pips"] is None

    configured["sl_pips"] = 25
    with pytest.raises(DojoBotCatalogError, match="dynamic-stop"):
        validate_bot_config(configured)

    omitted = _config(family)
    del omitted["sl_pips"]
    with pytest.raises(DojoBotCatalogError, match="missing=.*sl_pips"):
        validate_bot_config(omitted)


@pytest.mark.parametrize(
    "pairs",
    [
        [],
        ["USDJPY"],
        ["USD_USD"],
        ["USD_JPY", "USD_JPY"],
        ["usd_jpy"],
        ("USD_JPY",),
    ],
)
def test_pair_set_is_strict_json_and_canonical(pairs: object) -> None:
    config = _config("spike_fade")
    config["pairs"] = pairs
    with pytest.raises(DojoBotCatalogError):
        validate_bot_config(config)


def test_global_cap_cannot_claim_capacity_beyond_pair_caps() -> None:
    config = _config("spike_fade")
    config["global_max_concurrent"] = 3
    with pytest.raises(DojoBotCatalogError, match="exceeds"):
        validate_bot_config(config)


def test_fade_ladder_requires_room_for_its_second_layer() -> None:
    config = _config("fade_ladder")
    config["max_concurrent_per_pair"] = 1
    config["global_max_concurrent"] = 2
    with pytest.raises(DojoBotCatalogError, match="requires"):
        validate_bot_config(config)


@pytest.mark.parametrize("hour_key", ["opening_end_hour_local", "trade_end_hour_local"])
def test_session_range_hours_are_fixed_and_not_tunable(hour_key: str) -> None:
    config = _config("session_open_range_break")
    config[hour_key] = 9
    with pytest.raises(DojoBotCatalogError, match=f"unknown=.*{hour_key}"):
        validate_bot_config(config)


@pytest.mark.parametrize("retired_key", ["break_buffer_atr", "weekend_gap_atr_min"])
def test_preimplementation_candidate_key_names_are_rejected(retired_key: str) -> None:
    family = (
        "session_open_range_break"
        if retired_key == "break_buffer_atr"
        else "weekend_gap_recovery"
    )
    config = _config(family)
    config[retired_key] = 1.0
    with pytest.raises(DojoBotCatalogError, match=f"unknown=.*{retired_key}"):
        validate_bot_config(config)


def test_weekend_gap_parameters_have_strict_types_and_fraction_bounds() -> None:
    config = _config("weekend_gap_recovery")
    config["weekend_wait_bars"] = 1.5
    with pytest.raises(DojoBotCatalogError, match="integer"):
        validate_bot_config(config)

    config = _config("weekend_gap_recovery")
    config["weekend_spread_fraction"] = 1.01
    with pytest.raises(DojoBotCatalogError, match="weekend_spread_fraction"):
        validate_bot_config(config)

    config = _config("weekend_gap_recovery")
    config["weekend_wait_bars"] = 0
    with pytest.raises(DojoBotCatalogError, match="weekend_wait_bars"):
        validate_bot_config(config)


def test_config_hash_is_over_normalized_canonical_json() -> None:
    left = _config("spike_fade")
    right = dict(reversed(list(left.items())))
    right["pairs"] = list(reversed(right["pairs"]))
    right["max_concurrent"] = right.pop("max_concurrent_per_pair")

    digest = bot_config_sha256(left)
    assert digest == bot_config_sha256(right)
    assert len(digest) == 64
    assert (
        digest
        == hashlib.sha256(
            json.dumps(
                validate_bot_config(left),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )


def test_validation_does_not_mutate_input() -> None:
    config = _config("session_open_range_break")
    before = json.loads(json.dumps(config))
    validate_bot_config(config)
    assert config == before


def test_repo_owned_risk_policy_is_canonical_and_bound_into_catalog() -> None:
    policy = bot_risk_policy_manifest()
    digest = bot_risk_policy_sha256()

    assert policy["contract"] == RISK_POLICY_CONTRACT
    assert policy["schema_version"] == 1
    assert policy["hard_envelope"] == {
        "max_per_position_leverage": 5.0,
        "max_concurrent_per_pair": 2,
        "max_global_concurrent": 4,
        "max_global_gross_leverage": 20.0,
        "max_initial_sl_pips": 25.0,
        "max_single_stop_risk_index": 126.5,
        "max_gross_stop_risk_index": 506.0,
    }
    assert (
        digest
        == hashlib.sha256(
            json.dumps(
                policy,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert catalog_manifest()["risk_policy"] == policy
    assert catalog_manifest()["risk_policy_sha256"] == digest

    policy["hard_envelope"]["max_per_position_leverage"] = 999
    assert (
        bot_risk_policy_manifest()["hard_envelope"]["max_per_position_leverage"] == 5.0
    )


def test_fixed_stop_risk_vector_is_mechanical_content_addressed_and_rankable() -> None:
    config = _config("spike_fade")
    vector = bot_config_risk_vector(
        config,
        stress_slippage_pips_per_fill=0.3,
    )

    assert vector["contract"] == RISK_VECTOR_CONTRACT
    assert vector["risk_policy_sha256"] == bot_risk_policy_sha256()
    assert vector["per_position_leverage"] == 5.0
    assert vector["max_concurrent_per_pair"] == 1
    assert vector["max_global_concurrent"] == 2
    assert vector["pair_gross_leverage"] == 5.0
    assert vector["global_gross_leverage"] == 10.0
    assert vector["initial_stop_bound_kind"] == "FIXED_SL_PIPS"
    assert vector["single_stop_risk_index"] == pytest.approx(126.5)
    assert vector["gross_stop_risk_index"] == pytest.approx(253.0)
    assert vector["hard_envelope_passed"] is True
    assert vector["rankable"] is True
    assert vector["blocker_codes"] == []
    assert vector["absolute_nav_loss_bound_claimed"] is False
    assert vector["runner_market_receipt_required"] is True
    body = {key: value for key, value in vector.items() if key != "risk_vector_sha256"}
    assert (
        vector["risk_vector_sha256"]
        == hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )


def test_search_valid_config_can_be_mechanically_unrankable_under_risk_policy() -> None:
    config = _config("spike_fade")
    config.update(
        {
            "per_pos_lev": 6.0,
            "max_concurrent_per_pair": 3,
            "global_max_concurrent": 6,
            "sl_pips": 10.0,
        }
    )
    assert validate_bot_config(config)["per_pos_lev"] == 6.0

    vector = bot_config_risk_vector(
        config,
        stress_slippage_pips_per_fill=0.3,
    )
    assert vector["rankable"] is False
    assert vector["hard_envelope_passed"] is False
    assert vector["blocker_codes"] == [
        "PER_POSITION_LEVERAGE_HARD_CAP_EXCEEDED",
        "PAIR_CONCURRENCY_HARD_CAP_EXCEEDED",
        "GLOBAL_CONCURRENCY_HARD_CAP_EXCEEDED",
        "GLOBAL_GROSS_LEVERAGE_HARD_CAP_EXCEEDED",
    ]


def test_hard_envelope_boundaries_allow_two_layer_family_without_widening() -> None:
    config = _config("fade_ladder")
    config["global_max_concurrent"] = 4
    vector = bot_config_risk_vector(
        config,
        stress_slippage_pips_per_fill=0.3,
    )

    assert vector["max_concurrent_per_pair"] == 2
    assert vector["max_global_concurrent"] == 4
    assert vector["global_gross_leverage"] == 20.0
    assert vector["hard_envelope_passed"] is True
    assert vector["rankable"] is True


def test_excessive_fixed_stop_and_stop_risk_indices_are_unrankable() -> None:
    config = _config("spike_fade")
    config.update(
        {
            "sl_pips": 2_000,
            "max_concurrent_per_pair": 2,
            "global_max_concurrent": 4,
        }
    )
    assert validate_bot_config(config)["sl_pips"] == 2_000.0

    vector = bot_config_risk_vector(
        config,
        stress_slippage_pips_per_fill=0.3,
    )

    assert vector["rankable"] is False
    assert vector["hard_envelope_passed"] is False
    assert vector["single_stop_risk_index"] == pytest.approx(10_001.5)
    assert vector["gross_stop_risk_index"] == pytest.approx(40_006.0)
    assert vector["blocker_codes"] == [
        "INITIAL_SL_PIPS_HARD_CAP_EXCEEDED",
        "SINGLE_STOP_RISK_INDEX_HARD_CAP_EXCEEDED",
        "GROSS_STOP_RISK_INDEX_HARD_CAP_EXCEEDED",
    ]


def test_fixed_stop_risk_caps_are_inclusive_at_policy_boundary() -> None:
    config = _config("fade_ladder")
    config["global_max_concurrent"] = 4

    vector = bot_config_risk_vector(
        config,
        stress_slippage_pips_per_fill=0.3,
    )

    assert vector["initial_sl_pips"] == 25.0
    assert vector["single_stop_risk_index"] == pytest.approx(126.5)
    assert vector["gross_stop_risk_index"] == pytest.approx(506.0)
    assert vector["rankable"] is True
    assert vector["blocker_codes"] == []


def test_sl_free_and_dynamic_stop_configs_are_valid_search_but_unrankable() -> None:
    sl_free = _config("burst")
    sl_free["sl_pips"] = None
    assert validate_bot_config(sl_free)["sl_pips"] is None
    sl_free_vector = bot_config_risk_vector(
        sl_free,
        stress_slippage_pips_per_fill=0.3,
    )
    assert sl_free_vector["rankable"] is False
    assert sl_free_vector["initial_stop_bound_kind"] == "MISSING"
    assert sl_free_vector["single_stop_risk_index"] is None
    assert sl_free_vector["gross_stop_risk_index"] is None
    assert sl_free_vector["blocker_codes"] == ["FINITE_INITIAL_SL_BOUND_MISSING"]

    dynamic = _config("session_open_range_break")
    assert validate_bot_config(dynamic)["sl_pips"] is None
    dynamic_vector = bot_config_risk_vector(
        dynamic,
        stress_slippage_pips_per_fill=0.3,
    )
    assert dynamic_vector["rankable"] is False
    assert dynamic_vector["initial_stop_bound_kind"] == ("DYNAMIC_UNBOUNDED_BY_CONFIG")
    assert dynamic_vector["blocker_codes"] == [
        "DYNAMIC_INITIAL_STOP_BOUND_NOT_IMPLEMENTED"
    ]


def test_risk_vector_is_order_independent_and_does_not_mutate_config() -> None:
    left = _config("spike_fade")
    before = json.loads(json.dumps(left))
    right = dict(reversed(list(left.items())))
    right["pairs"] = list(reversed(right["pairs"]))
    right["max_concurrent"] = right.pop("max_concurrent_per_pair")

    first = bot_config_risk_vector(left, stress_slippage_pips_per_fill=0.3)
    second = bot_config_risk_vector(right, stress_slippage_pips_per_fill=0.3)
    assert first == second
    assert left == before


@pytest.mark.parametrize("stress", [-0.1, True, math.inf, math.nan, "0.3"])
def test_risk_vector_rejects_invalid_stress_cost(stress: object) -> None:
    with pytest.raises(DojoBotCatalogError, match="stress_slippage"):
        bot_config_risk_vector(
            _config("spike_fade"),
            stress_slippage_pips_per_fill=stress,
        )
