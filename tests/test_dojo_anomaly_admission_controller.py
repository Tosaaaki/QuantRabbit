from __future__ import annotations

import copy
import math

import pytest

from quant_rabbit.dojo_anomaly_admission_controller import (
    FORMAL_G8_CURRENCIES,
    FORMAL_G8_PAIRS,
    DojoAnomalyAdmissionError,
    allocate_candidates,
    build_policy,
    build_train_plan,
    canonical_sha256,
    validate_allocation,
    validate_policy,
    validate_train_plan,
)


DECISION_EPOCH = 1_800_000_000
HISTORY_BARS = 32


def _candles(pair_index: int, *, mode: str = "NORMAL") -> list[dict]:
    base = 100.0 + pair_index
    closes = [base]
    for index in range(1, HISTORY_BARS):
        change = (
            0.00018 * math.sin((index + pair_index * 0.71) * 0.83)
            + 0.00011 * math.cos((index * 0.37) + pair_index)
            + 0.00001 * ((pair_index % 5) - 2)
        )
        if mode == "MOMENTUM" and index >= HISTORY_BARS - 6:
            change = 0.004
        elif mode == "REVERSAL" and HISTORY_BARS - 6 <= index < HISTORY_BARS - 1:
            change = 0.0015
        elif mode == "REVERSAL" and index == HISTORY_BARS - 1:
            change = -0.009
        elif mode == "HIGH_VOL" and index >= HISTORY_BARS - 4:
            change = 0.004 * (-1 if index % 2 else 1)
        closes.append(closes[-1] * math.exp(change))

    first_epoch = DECISION_EPOCH - (HISTORY_BARS - 1) * 3_600
    result = []
    for index, close in enumerate(closes):
        spread = close * 0.00002
        half_range = close * 0.0004
        result.append(
            {
                "close_epoch": first_epoch + index * 3_600,
                "complete": True,
                "bid_high": close + half_range,
                "bid_low": close - half_range,
                "bid_close": close,
                "ask_high": close + half_range + spread,
                "ask_low": close - half_range + spread,
                "ask_close": close + spread,
            }
        )
    return result


def _panel(*, special_pair: str | None = None, mode: str = "NORMAL") -> dict:
    return {
        pair: _candles(index, mode=mode if pair == special_pair else "NORMAL")
        for index, pair in enumerate(FORMAL_G8_PAIRS)
    }


def _band(half: float, quarter: float, hold: float) -> dict:
    return {
        "reduce_to_half_at": half,
        "reduce_to_quarter_at": quarter,
        "hold_at": hold,
    }


def _policy() -> dict:
    return build_policy(
        policy_id="room-meta-01-policy-test-v1",
        lookbacks={
            "momentum_bars": 6,
            "reversal_prior_bars": 5,
            "volatility_short_bars": 4,
            "volatility_long_bars": 20,
            "atr_bars": 14,
            "correlation_bars": 20,
        },
        bands={
            "momentum_z": _band(1.25, 2.0, 2.75),
            "reversal_shock_z": _band(1.25, 2.0, 2.75),
            "volatility_ratio": _band(1.75, 2.5, 4.0),
            "spread_atr_ratio": _band(0.20, 0.35, 0.55),
            "correlation_concentration": _band(0.85, 0.92, 0.99),
            "currency_gross_exposure_fraction": _band(0.30, 0.45, 0.60),
        },
        selected_pair_abs_correlation_hold_at=0.99,
    )


def _candidate(
    candidate_id: str,
    rank: int,
    pair: str,
    *,
    side: str = "LONG",
    exposure_increment: float = 0.05,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "priority_rank": rank,
        "strategy_family": "upstream-test-bot",
        "pair": pair,
        "side": side,
        "full_size_units": 4_000,
        "currency_exposure_increment_at_full_size": exposure_increment,
    }


def _exposure(value: float = 0.05) -> dict:
    return {currency: value for currency in FORMAL_G8_CURRENCIES}


def _allocate(
    *,
    panel: dict,
    arm: str,
    candidates: list[dict],
    exposure: dict | None = None,
    slots: int = 1,
    policy: dict | None = None,
) -> dict:
    return allocate_candidates(
        completed_h1_panel=panel,
        decision_epoch=DECISION_EPOCH,
        policy=policy or _policy(),
        arm=arm,
        candidates=candidates,
        currency_gross_exposure_fractions=exposure or _exposure(),
        capacity_slots=slots,
    )


def test_policy_is_exact_content_addressed_and_direction_neutral() -> None:
    policy = _policy()
    assert validate_policy(policy) == policy
    assert policy["size_multiplier_alphabet"] == [0.0, 0.25, 0.5, 1.0]
    assert policy["direction_policy"] == {
        "predict_direction": False,
        "change_upstream_direction": False,
        "change_upstream_rank": False,
    }
    tampered = copy.deepcopy(policy)
    tampered["bands"]["momentum_z"]["hold_at"] += 0.1
    with pytest.raises(DojoAnomalyAdmissionError, match="canonical V1"):
        validate_policy(tampered)


def test_extreme_momentum_hold_releases_slot_to_next_upstream_candidate() -> None:
    first_pair, second_pair = FORMAL_G8_PAIRS[:2]
    result = _allocate(
        panel=_panel(special_pair=first_pair, mode="MOMENTUM"),
        arm="EXTREME_MOMENTUM_VETO",
        candidates=[
            _candidate("first", 1, first_pair, side="SHORT"),
            _candidate("second", 2, second_pair, side="LONG"),
        ],
    )

    first, second = result["candidate_decisions"]
    assert first["admission_decision"] == "HOLD"
    assert first["size_multiplier"] == 0.0
    assert first["selected_slot"] is None
    assert second["admission_decision"] == "ENTER_OK"
    assert second["selected_slot"] == 1
    assert result["selected_candidate_ids"] == ["second"]
    assert result["capital_recycle_policy"]["hold_consumes_capacity_slot"] is False
    assert first["upstream_side_preserved"] == "SHORT"
    assert second["upstream_side_preserved"] == "LONG"
    assert result["direction_predictions_emitted"] is False
    assert result["evidence_class"] == "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC"
    assert result["runner_integration_complete"] is False

    assert (
        validate_allocation(
            result,
            completed_h1_panel=_panel(special_pair=first_pair, mode="MOMENTUM"),
            policy=_policy(),
            candidates=[
                _candidate("first", 1, first_pair, side="SHORT"),
                _candidate("second", 2, second_pair, side="LONG"),
            ],
            currency_gross_exposure_fractions=_exposure(),
        )
        == result
    )
    tampered = copy.deepcopy(result)
    tampered["candidate_decisions"][0]["size_multiplier"] = 1.0
    with pytest.raises(DojoAnomalyAdmissionError, match="canonical recomputation"):
        validate_allocation(
            tampered,
            completed_h1_panel=_panel(special_pair=first_pair, mode="MOMENTUM"),
            policy=_policy(),
            candidates=[
                _candidate("first", 1, first_pair, side="SHORT"),
                _candidate("second", 2, second_pair, side="LONG"),
            ],
            currency_gross_exposure_fractions=_exposure(),
        )


def test_base_arm_does_not_depend_on_unused_anomaly_feature_variation() -> None:
    pair = FORMAL_G8_PAIRS[0]
    panel = _panel()
    for candle in panel[pair]:
        candle["bid_close"] = 100.0
        candle["ask_close"] = 100.002
        candle["bid_high"] = 100.04
        candle["ask_high"] = 100.042
        candle["bid_low"] = 99.96
        candle["ask_low"] = 99.962
    result = _allocate(
        panel=panel,
        arm="BASE_BOT",
        candidates=[_candidate("base", 1, pair)],
    )
    row = result["candidate_decisions"][0]
    assert row["admission_decision"] == "ENTER_OK"
    assert row["anomaly_features"] == {}


def test_reversal_shock_is_an_independent_veto_arm() -> None:
    pair = FORMAL_G8_PAIRS[2]
    result = _allocate(
        panel=_panel(special_pair=pair, mode="REVERSAL"),
        arm="REVERSAL_SHOCK_VETO",
        candidates=[_candidate("reversal", 1, pair)],
    )
    row = result["candidate_decisions"][0]
    assert row["admission_decision"] == "HOLD"
    assert (
        row["anomaly_features"]["reversal_shock_z"]
        >= _policy()["bands"]["reversal_shock_z"]["hold_at"]
    )


def test_volatility_arm_reduces_size_without_changing_direction() -> None:
    pair = FORMAL_G8_PAIRS[3]
    policy = _policy()
    policy = build_policy(
        policy_id="volatility-reduction-test",
        lookbacks=policy["lookbacks"],
        bands={
            **policy["bands"],
            "volatility_ratio": _band(1.01, 10.0, 20.0),
            "correlation_concentration": _band(2.0, 3.0, 4.0),
            "spread_atr_ratio": _band(2.0, 3.0, 4.0),
        },
        selected_pair_abs_correlation_hold_at=1.0,
    )
    result = _allocate(
        panel=_panel(special_pair=pair, mode="HIGH_VOL"),
        arm="VOLATILITY_CORRELATION_SIZING",
        candidates=[_candidate("volatile", 1, pair, side="SHORT")],
        policy=policy,
    )
    row = result["candidate_decisions"][0]
    assert row["admission_decision"] == "REDUCE_SIZE"
    assert row["size_multiplier"] == 0.5
    assert row["allocated_units"] == 2_000
    assert row["upstream_side_preserved"] == "SHORT"


def test_currency_exposure_can_step_down_to_quarter_size() -> None:
    pair = FORMAL_G8_PAIRS[4]
    policy = _policy()
    policy = build_policy(
        policy_id="exposure-reduction-test",
        lookbacks=policy["lookbacks"],
        bands={
            **policy["bands"],
            "volatility_ratio": _band(10.0, 20.0, 30.0),
            "spread_atr_ratio": _band(10.0, 20.0, 30.0),
            "correlation_concentration": _band(2.0, 3.0, 4.0),
            "currency_gross_exposure_fraction": _band(0.30, 0.40, 0.50),
        },
        selected_pair_abs_correlation_hold_at=1.0,
    )
    result = _allocate(
        panel=_panel(),
        arm="VOLATILITY_CORRELATION_SIZING",
        candidates=[
            _candidate("exposure", 1, pair, exposure_increment=0.40),
        ],
        exposure=_exposure(0.25),
        policy=policy,
    )
    row = result["candidate_decisions"][0]
    assert row["admission_decision"] == "REDUCE_SIZE"
    assert row["size_multiplier"] == 0.25
    assert row["allocated_units"] == 1_000


def test_selected_pair_correlation_hold_advances_to_uncorrelated_candidate() -> None:
    first_pair, correlated_pair, fallback_pair = FORMAL_G8_PAIRS[:3]
    panel = _panel()
    # Preserve a different price scale while making the return path identical.
    scale = panel[correlated_pair][0]["bid_close"] / panel[first_pair][0]["bid_close"]
    panel[correlated_pair] = [
        {
            key: (value * scale if key.startswith(("bid_", "ask_")) else value)
            for key, value in candle.items()
        }
        for candle in panel[first_pair]
    ]
    policy = _policy()
    policy = build_policy(
        policy_id="selected-correlation-test",
        lookbacks=policy["lookbacks"],
        bands={
            **policy["bands"],
            "volatility_ratio": _band(10.0, 20.0, 30.0),
            "spread_atr_ratio": _band(10.0, 20.0, 30.0),
            "correlation_concentration": _band(2.0, 3.0, 4.0),
            "currency_gross_exposure_fraction": _band(2.0, 3.0, 4.0),
        },
        selected_pair_abs_correlation_hold_at=0.999,
    )
    result = _allocate(
        panel=panel,
        arm="VOLATILITY_CORRELATION_SIZING",
        candidates=[
            _candidate("first", 1, first_pair),
            _candidate("correlated", 2, correlated_pair),
            _candidate("fallback", 3, fallback_pair),
        ],
        slots=2,
        policy=policy,
    )
    rows = {row["candidate_id"]: row for row in result["candidate_decisions"]}
    assert result["selected_candidate_ids"] == ["first", "fallback"]
    assert rows["correlated"]["admission_decision"] == "HOLD"
    assert "SELECTED_PAIR_CORRELATION_HOLD" in rows["correlated"]["reason_codes"]


def test_exact28_completed_synchronized_panel_fails_closed() -> None:
    pair = FORMAL_G8_PAIRS[0]
    candidate = [_candidate("candidate", 1, pair)]
    missing = _panel()
    missing.pop(FORMAL_G8_PAIRS[-1])
    with pytest.raises(DojoAnomalyAdmissionError, match="schema mismatch"):
        _allocate(panel=missing, arm="BASE_BOT", candidates=candidate)

    incomplete = _panel()
    incomplete[pair][-1]["complete"] = False
    with pytest.raises(DojoAnomalyAdmissionError, match="not complete"):
        _allocate(panel=incomplete, arm="BASE_BOT", candidates=candidate)

    future = _panel()
    future[pair][-1]["close_epoch"] += 3_600
    with pytest.raises(
        DojoAnomalyAdmissionError,
        match="discontinuous|synchronized|decision_epoch",
    ):
        _allocate(panel=future, arm="BASE_BOT", candidates=candidate)


def test_ai_exit_arm_remains_outside_direction_neutral_controller() -> None:
    with pytest.raises(DojoAnomalyAdmissionError, match="room-ai-01"):
        _allocate(
            panel=_panel(),
            arm="AI_EXIT_CAPITAL_RELEASE",
            candidates=[_candidate("candidate", 1, FORMAL_G8_PAIRS[0])],
        )


def test_train_plan_preregisters_ablation_economics_and_room_separation() -> None:
    policy = _policy()
    digests = {
        name: canonical_sha256({"binding": name})
        for name in (
            "fixed_denominator",
            "source_binding",
            "evaluator_binding",
            "cost_policy",
            "risk_policy",
        )
    }
    plan = build_train_plan(
        plan_id="room-meta-01-ablation-v1",
        fixed_denominator_sha256=digests["fixed_denominator"],
        source_binding_sha256=digests["source_binding"],
        evaluator_binding_sha256=digests["evaluator_binding"],
        cost_policy_sha256=digests["cost_policy"],
        risk_policy_sha256=digests["risk_policy"],
        policy_sha256=policy["policy_sha256"],
    )
    assert [row["arm"] for row in plan["arms"]] == [
        "BASE_BOT",
        "EXTREME_MOMENTUM_VETO",
        "REVERSAL_SHOCK_VETO",
        "VOLATILITY_CORRELATION_SIZING",
        "COMBINED_ANOMALY_ADMISSION",
        "AI_EXIT_CAPITAL_RELEASE",
    ]
    assert plan["arms"][-1]["owner_room"] == "room-ai-01"
    assert plan["separation"]["room_03_generates_relative_strength_alpha"] is True
    assert plan["separation"]["room_meta_01_generates_direction"] is False
    assert {
        "net_profit_after_all_costs_jpy",
        "maximum_mark_to_market_drawdown_fraction",
        "cvar_95_loss_jpy",
        "ruin_probability",
        "margin_closeout_count",
        "missed_opportunity_after_cost_jpy",
        "capital_utilization_fraction",
    }.issubset(plan["required_metrics"])
    assert plan["authority"]["live_permission"] is False
    assert validate_train_plan(plan) == plan
    tampered = copy.deepcopy(plan)
    tampered["required_metrics"].remove("ruin_probability")
    with pytest.raises(DojoAnomalyAdmissionError, match="canonical V1"):
        validate_train_plan(tampered)
