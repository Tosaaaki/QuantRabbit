from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

import quant_rabbit.fast_bot_technical_grid_backtest as grid_module
from quant_rabbit.fast_bot_technical_grid_backtest import (
    PLANNED_CANDIDATE_COUNT,
    CausalTimeframeFeature,
    assign_fixed_split_roles_v1,
    build_causal_technical_feature_snapshot_v1,
    build_fast_bot_technical_grid_catalog_v1,
    deoverlap_same_pair_signal_specs_v1,
    freeze_fast_bot_technical_grid_signal_v1,
    resolve_executable_bidask_time_close_v1,
    select_validation_one_se_v1,
    technical_grid_arms_v1,
)
from quant_rabbit.fast_bot_technical_hypotheses import (
    build_fast_bot_technical_hypotheses,
    technical_hypothesis_shadow_valid,
)
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


UTC = timezone.utc
DECISION = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
SOURCE_SHA = "a" * 64
TRUTH_SHA = "b" * 64


def _features(
    *,
    decision: datetime = DECISION,
) -> list[CausalTimeframeFeature]:
    closes = {
        "M1": decision,
        "M5": decision,
        "M15": decision,
        "M30": decision,
        "H1": decision,
        "H4": decision - timedelta(hours=2),
        "D": decision - timedelta(hours=14),
    }
    return [
        CausalTimeframeFeature(
            timeframe=timeframe,
            complete_candle_close_utc=closes[timeframe],
            market_state={},
            indicators={"atr_pips": 5.0} if timeframe == "M5" else {},
            indicator_series={},
        )
        for timeframe in ("M1", "M5", "M15", "M30", "H1", "H4", "D")
    ]


def _signal(
    *,
    activation: datetime = DECISION,
    hypothesis_id: str = "H01",
    orientation: str = "DIRECT",
    source_side: str = "LONG",
    side: str = "LONG",
    arm_id: str = "HOLD180",
    order_type: str = "STOP",
    entry: float = 1.1002,
    target: float = 1.1020,
    stop: float = 1.0980,
    source_sha: str = SOURCE_SHA,
) -> dict[str, Any]:
    return freeze_fast_bot_technical_grid_signal_v1(
        pair="EUR_USD",
        hypothesis_id=hypothesis_id,
        orientation=orientation,
        source_predicted_side=source_side,
        side=side,
        arm_id=arm_id,
        order_type=order_type,
        activation_at_utc=activation,
        entry_price=entry,
        base_take_profit_price=target,
        base_stop_loss_price=stop,
        causal_source_sha256=source_sha,
    )


def _candle(
    seconds: int,
    *,
    bid_o: float,
    bid_h: float,
    bid_l: float,
    bid_c: float | None = None,
    spread: float = 0.0001,
) -> S5BidAskCandle:
    close = bid_o if bid_c is None else bid_c
    return S5BidAskCandle(
        timestamp_utc=DECISION + timedelta(seconds=seconds),
        bid_o=bid_o,
        bid_h=bid_h,
        bid_l=bid_l,
        bid_c=close,
        ask_o=bid_o + spread,
        ask_h=bid_h + spread,
        ask_l=bid_l + spread,
        ask_c=close + spread,
    )


def _metric(
    candidate_id: str,
    role: str,
    *,
    daily: float,
    standard_error: float = 0.2,
    p_value: float = 1e-8,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "data_role": role,
        "cluster_count": 10,
        "mean_daily_post_cost_pips": daily,
        "standard_error_daily_post_cost_pips": standard_error,
        "mean_post_cost_pips": daily / 10.0,
        "profit_factor": 2.0,
        "one_sided_p_value": p_value,
    }


def test_fixed_catalog_is_exactly_182_ordered_candidates_plus_h08_control() -> None:
    arms = technical_grid_arms_v1()
    catalog = build_fast_bot_technical_grid_catalog_v1()

    assert [arm.arm_id for arm in arms] == [
        "BASE",
        "TP050",
        "TP075",
        "TP125",
        "SL075",
        "SL125",
        "SL150",
        "HOLD180",
        "HOLD300",
        "HOLD600",
        "HOLD1800",
        "TTL45",
        "TTL180",
    ]
    assert len(catalog["candidates"]) == PLANNED_CANDIDATE_COUNT == 182
    assert catalog["multiple_testing_denominator"] == 182
    assert catalog["candidates"][0]["candidate_id"] == "H01:DIRECT:BASE"
    assert catalog["candidates"][12]["candidate_id"] == "H01:DIRECT:TTL180"
    assert catalog["candidates"][13]["candidate_id"] == "H01:INVERSE:BASE"
    assert catalog["candidates"][-1]["candidate_id"] == "H07:INVERSE:TTL180"
    assert [row["catalog_order"] for row in catalog["candidates"]] == list(range(182))
    assert catalog["h08_control"]["post_cost_realized_pips"] == 0.0
    assert catalog["h08_control"]["multiple_testing_denominator_member"] is False
    assert catalog["order_authority"] == "NONE"


def test_causal_snapshot_is_consumable_and_future_or_stale_m1_is_rejected() -> None:
    snapshot = build_causal_technical_feature_snapshot_v1(
        pair="EUR_USD",
        decision_at_utc=DECISION,
        timeframes=_features(),
    )
    shadow = build_fast_bot_technical_hypotheses(
        snapshot,
        attempt_direction="UP",
        branch_outcome="ACCEPTED",
        route_family="BREAKOUT_CONTINUATION",
        spread_pips=0.5,
        m5_atr_pips=5.0,
        spread_to_m5_atr=0.1,
    )

    assert technical_hypothesis_shadow_valid(
        shadow,
        feature_snapshot=snapshot,
        attempt_direction="UP",
        branch_outcome="ACCEPTED",
        route_family="BREAKOUT_CONTINUATION",
        spread_pips=0.5,
        m5_atr_pips=5.0,
        spread_to_m5_atr=0.1,
    )
    assert shadow["feature_snapshot_sha256"] == snapshot["contract_sha256"]

    future = _features()
    future[1] = CausalTimeframeFeature(
        timeframe="M5",
        complete_candle_close_utc=DECISION + timedelta(minutes=5),
        market_state={},
        indicators={},
        indicator_series={},
    )
    with pytest.raises(ValueError, match="exceeds"):
        build_causal_technical_feature_snapshot_v1(
            pair="EUR_USD", decision_at_utc=DECISION, timeframes=future
        )

    stale_m1 = _features()
    stale_m1[0] = CausalTimeframeFeature(
        timeframe="M1",
        complete_candle_close_utc=DECISION - timedelta(minutes=1),
        market_state={},
        indicators={},
        indicator_series={},
    )
    with pytest.raises(ValueError, match="M1 close"):
        build_causal_technical_feature_snapshot_v1(
            pair="EUR_USD", decision_at_utc=DECISION, timeframes=stale_m1
        )

    malformed = [
        {
            "timeframe": row.timeframe,
            "complete_candle_close_utc": row.complete_candle_close_utc,
            "market_state": 0 if row.timeframe == "M1" else {},
            "indicators": row.indicators,
            "indicator_series": row.indicator_series,
        }
        for row in _features()
    ]
    with pytest.raises(ValueError, match="market_state must be a mapping"):
        build_causal_technical_feature_snapshot_v1(
            pair="EUR_USD", decision_at_utc=DECISION, timeframes=malformed
        )


def test_inverse_signal_requires_actual_short_geometry_and_seals_arm_clocks() -> None:
    inverse = _signal(
        orientation="INVERSE",
        source_side="LONG",
        side="SHORT",
        arm_id="TTL45",
        entry=1.1002,
        target=1.0992,
        stop=1.1012,
    )

    assert inverse["side"] == "SHORT"
    assert inverse["take_profit_price"] < inverse["entry_price"]
    assert (
        inverse["entry_expires_at_utc"]
        == (DECISION + timedelta(seconds=45)).isoformat()
    )
    assert (
        inverse["latest_maturity_at_utc"]
        == (DECISION + timedelta(seconds=945)).isoformat()
    )
    assert inverse["order_authority"] == "NONE"

    with pytest.raises(ValueError, match="base geometry"):
        _signal(
            orientation="INVERSE",
            source_side="LONG",
            side="SHORT",
            entry=1.1002,
            target=1.1020,
            stop=1.0980,
        )
    with pytest.raises(ValueError, match="actual-side"):
        _signal(orientation="INVERSE", source_side="LONG", side="LONG")


def test_ofat_geometry_changes_only_declared_axis() -> None:
    base = _signal(arm_id="BASE", target=1.1012, stop=1.0992)
    tp_half = _signal(arm_id="TP050", target=1.1012, stop=1.0992)
    sl_wide = _signal(arm_id="SL150", target=1.1012, stop=1.0992)

    assert base["take_profit_price"] == pytest.approx(1.1012)
    assert base["stop_loss_price"] == pytest.approx(1.0992)
    assert tp_half["take_profit_price"] == pytest.approx(1.1007)
    assert tp_half["stop_loss_price"] == base["stop_loss_price"]
    assert sl_wide["take_profit_price"] == base["take_profit_price"]
    assert sl_wide["stop_loss_price"] == pytest.approx(1.0987)


def test_intrabar_fill_candle_never_uses_prefill_open_as_profit_or_stop_gap() -> None:
    signal = _signal(entry=1.1002, target=1.1005, stop=1.0998)
    candle = _candle(
        0,
        bid_o=1.0997,
        bid_h=1.1006,
        bid_l=1.0996,
        bid_c=1.1000,
    )

    outcome = resolve_executable_bidask_time_close_v1(
        signal, [candle], truth_source_receipt_sha256=TRUTH_SHA
    )

    assert outcome["fill_kind"] == "INTRABAR_TRIGGER"
    assert outcome["exit_reason"] == "STOP_LOSS_AMBIGUOUS_FILL_S5"
    assert outcome["exit_price"] == signal["stop_loss_price"]
    assert outcome["ambiguous_same_s5"] is True
    assert outcome["stop_gap_worse"] is False
    assert outcome["post_cost_realized_pips"] < 0.0


def test_later_same_s5_target_and_stop_is_stop_first() -> None:
    signal = _signal(entry=1.1002, target=1.1005, stop=1.0998)
    candles = [
        _candle(0, bid_o=1.1000, bid_h=1.10015, bid_l=1.0999),
        _candle(5, bid_o=1.1001, bid_h=1.1006, bid_l=1.0997),
    ]

    outcome = resolve_executable_bidask_time_close_v1(
        signal, candles, truth_source_receipt_sha256=TRUTH_SHA
    )

    assert outcome["exit_reason"] == "STOP_LOSS_AMBIGUOUS_SAME_S5"
    assert outcome["exit_price"] == signal["stop_loss_price"]
    assert outcome["ambiguous_same_s5"] is True


def test_stop_entry_and_stop_exit_charge_worse_real_open_gaps() -> None:
    signal = _signal(entry=1.1002, target=1.1020, stop=1.0990)
    candles = [
        _candle(0, bid_o=1.1004, bid_h=1.1006, bid_l=1.1003),
        _candle(5, bid_o=1.0985, bid_h=1.0988, bid_l=1.0982),
    ]

    outcome = resolve_executable_bidask_time_close_v1(
        signal, candles, truth_source_receipt_sha256=TRUTH_SHA
    )

    assert outcome["fill_kind"] == "OPEN_TRIGGER"
    assert outcome["fill_price"] == pytest.approx(1.1005)
    assert outcome["entry_gap_pips"] == pytest.approx(3.0)
    assert outcome["exit_reason"] == "STOP_LOSS_GAP"
    assert outcome["exit_price"] == pytest.approx(1.0985)
    assert outcome["stop_gap_worse"] is True


def test_time_close_uses_first_real_open_after_missing_s5_slots() -> None:
    signal = _signal(entry=1.1002, target=1.1020, stop=1.0980)
    candles = [
        _candle(0, bid_o=1.1000, bid_h=1.10015, bid_l=1.0999),
        _candle(190, bid_o=1.1004, bid_h=1.1005, bid_l=1.1003),
    ]

    outcome = resolve_executable_bidask_time_close_v1(
        signal, candles, truth_source_receipt_sha256=TRUTH_SHA
    )

    assert outcome["exit_reason"] == "EXECUTABLE_TIME_CLOSE"
    assert outcome["exit_at_utc"] == (DECISION + timedelta(seconds=190)).isoformat()
    assert outcome["exit_price"] == pytest.approx(1.1004)
    assert outcome["truth_no_tick_slot_count"] > 0
    assert outcome["missing_no_tick_intervals_synthesized"] is False


def test_next_real_open_after_latest_maturity_still_owns_gap_loss() -> None:
    signal = _signal(entry=1.1002, target=1.1020, stop=1.0980)
    assert datetime.fromisoformat(signal["latest_maturity_at_utc"]) == (
        DECISION + timedelta(seconds=270)
    )
    candles = [
        _candle(0, bid_o=1.1000, bid_h=1.10015, bid_l=1.0999),
        _candle(300, bid_o=1.0975, bid_h=1.0977, bid_l=1.0973),
    ]

    outcome = resolve_executable_bidask_time_close_v1(
        signal, candles, truth_source_receipt_sha256=TRUTH_SHA
    )

    assert outcome["status"] == "MATURE_FILLED_STOP_LOSS"
    assert outcome["exit_reason"] == "STOP_LOSS_GAP_AT_TIME_CLOSE"
    assert outcome["exit_at_utc"] == (DECISION + timedelta(seconds=300)).isoformat()
    assert outcome["exit_price"] == pytest.approx(1.0975)
    assert outcome["post_cost_realized_pips"] < 0.0
    assert outcome["truth_post_latest_maturity_candle_count"] == 1


def test_provenance_boundary_stops_open_exit_without_reading_boundary_price() -> None:
    signal = _signal(entry=1.1002, target=1.1020, stop=1.0980)
    fill = _candle(0, bid_o=1.1000, bid_h=1.10015, bid_l=1.0999)
    boundary = DECISION + timedelta(seconds=250)

    outcome = resolve_executable_bidask_time_close_v1(
        signal,
        [fill],
        truth_source_receipt_sha256=TRUTH_SHA,
        truth_provenance_end_utc=boundary,
    )

    assert outcome["status"] == "UNRESOLVED_EXIT_CROSSES_PROVENANCE_BOUNDARY"
    assert outcome["exit_reason"] == "EXIT_REMAINS_OPEN_AT_PROVENANCE_BOUNDARY"
    assert outcome["scorecard_result_available"] is False

    with pytest.raises(ValueError, match="exclusive provenance boundary"):
        resolve_executable_bidask_time_close_v1(
            signal,
            [fill, _candle(250, bid_o=1.0, bid_h=1.0001, bid_l=0.9999)],
            truth_source_receipt_sha256=TRUTH_SHA,
            truth_provenance_end_utc=boundary,
        )


def test_missing_real_open_after_fill_is_unresolved_not_zero_or_stop() -> None:
    signal = _signal(entry=1.1002, target=1.1020, stop=1.0980)
    fill = _candle(0, bid_o=1.1000, bid_h=1.10015, bid_l=1.0999)

    outcome = resolve_executable_bidask_time_close_v1(
        signal, [fill], truth_source_receipt_sha256=TRUTH_SHA
    )

    assert outcome["status"] == "UNRESOLVED_TIME_CLOSE_REAL_OPEN_MISSING"
    assert outcome["scorecard_result_available"] is False
    assert outcome["post_cost_realized_pips"] is None


def test_ttl_requires_s5_candle_end_not_open_to_be_within_expiry() -> None:
    signal = _signal(arm_id="TTL45", entry=1.1002, target=1.1020, stop=1.0980)
    late_touch = _candle(45, bid_o=1.1000, bid_h=1.1002, bid_l=1.0999)

    outcome = resolve_executable_bidask_time_close_v1(
        signal, [late_touch], truth_source_receipt_sha256=TRUTH_SHA
    )

    assert outcome["status"] == "MATURE_UNFILLED"
    assert outcome["filled"] is False
    assert outcome["post_cost_realized_pips"] == 0.0
    assert outcome["entry_ttl_coverage_complete"] is True
    assert (
        outcome["entry_ttl_coverage_sentinel_at_utc"]
        == late_touch.timestamp_utc.isoformat()
    )


def test_unfilled_signal_without_ttl_coverage_is_unresolved_not_zero() -> None:
    signal = _signal(arm_id="TTL45", entry=1.1002, target=1.1020, stop=1.0980)
    partial = _candle(0, bid_o=1.0998, bid_h=1.0999, bid_l=1.0997)

    missing = resolve_executable_bidask_time_close_v1(
        signal, [], truth_source_receipt_sha256=TRUTH_SHA
    )
    partial_result = resolve_executable_bidask_time_close_v1(
        signal, [partial], truth_source_receipt_sha256=TRUTH_SHA
    )

    for outcome in (missing, partial_result):
        assert outcome["status"] == "UNRESOLVED_MISSING_ENTRY_TTL_COVERAGE"
        assert outcome["scorecard_result_available"] is False
        assert outcome["post_cost_realized_pips"] is None
        assert outcome["entry_ttl_coverage_complete"] is False


def test_stop_open_gap_crossing_absolute_target_cancels_without_fill() -> None:
    signal = _signal(entry=1.1002, target=1.1005, stop=1.0998)
    crossed_target_at_open = _candle(
        0,
        bid_o=1.1006,
        bid_h=1.1007,
        bid_l=1.10055,
        bid_c=1.10065,
    )

    outcome = resolve_executable_bidask_time_close_v1(
        signal,
        [crossed_target_at_open],
        truth_source_receipt_sha256=TRUTH_SHA,
    )

    assert outcome["status"] == "CANCEL_NO_FILL_INVALID_DEPENDENT_GEOMETRY"
    assert outcome["filled"] is False
    assert outcome["fill_price"] is None
    assert outcome["post_cost_realized_pips"] == 0.0
    assert outcome["exit_reason"] == "CANCEL_NO_FILL_INVALID_DEPENDENT_GEOMETRY"


def test_resolver_rejects_signal_tampering_and_nonchronological_truth() -> None:
    signal = _signal()
    tampered = deepcopy(signal)
    tampered["take_profit_price"] = 9.0
    with pytest.raises(ValueError, match="seal"):
        resolve_executable_bidask_time_close_v1(
            tampered, [], truth_source_receipt_sha256=TRUTH_SHA
        )

    one = _candle(0, bid_o=1.1000, bid_h=1.1001, bid_l=1.0999)
    with pytest.raises(ValueError, match="chronological"):
        resolve_executable_bidask_time_close_v1(
            signal, [one, one], truth_source_receipt_sha256=TRUTH_SHA
        )


def test_deoverlap_is_only_per_exact_pair_candidate_specification() -> None:
    first = _signal(source_sha="1" * 64)
    overlap = _signal(activation=DECISION + timedelta(minutes=1), source_sha="2" * 64)
    other_h = _signal(hypothesis_id="H02", source_sha="3" * 64)
    after = _signal(activation=DECISION + timedelta(minutes=5), source_sha="4" * 64)

    receipt = deoverlap_same_pair_signal_specs_v1([overlap, other_h, after, first])

    assert receipt["source_signal_count"] == 4
    assert receipt["selected_signal_ids"] == [
        first["signal_id"],
        other_h["signal_id"],
        after["signal_id"],
    ]
    assert receipt["rejected"] == [
        {
            "signal_id": overlap["signal_id"],
            "reason": "OVERLAPPING_SAME_PAIR_SIGNAL_SPEC",
        }
    ]


def test_order_type_is_signal_identity_and_duplicate_ids_fail_closed() -> None:
    stop = _signal(order_type="STOP")
    limit = _signal(order_type="LIMIT")

    assert stop["signal_id"] != limit["signal_id"]
    assert stop["signal_sha256"] != limit["signal_sha256"]

    with pytest.raises(ValueError, match="duplicate signal_id"):
        deoverlap_same_pair_signal_specs_v1([stop, deepcopy(stop)])
    with pytest.raises(ValueError, match="duplicate signal_id"):
        assign_fixed_split_roles_v1(
            [stop, deepcopy(stop)],
            train_end_utc=DECISION + timedelta(days=1),
            validation_end_utc=DECISION + timedelta(days=2),
            holdout_end_utc=DECISION + timedelta(days=3),
        )


def test_authority_lists_are_fresh_and_nonempty_tampering_is_rejected() -> None:
    first_catalog = build_fast_bot_technical_grid_catalog_v1()
    first_catalog["order_intents"].append({"poison": True})
    first_catalog["candidates"][0]["order_intents"].append({"poison": True})

    assert first_catalog["candidates"][1]["order_intents"] == []
    second_catalog = build_fast_bot_technical_grid_catalog_v1()
    assert second_catalog["order_intents"] == []
    assert second_catalog["candidates"][0]["order_intents"] == []
    assert second_catalog["h08_control"]["order_intents"] == []

    poisoned = _signal(source_sha="8" * 64)
    poisoned["order_intents"].append({"units": 1})
    body = {key: value for key, value in poisoned.items() if key != "signal_sha256"}
    poisoned["signal_sha256"] = grid_module._canonical_sha(body)
    clean = _signal(source_sha="9" * 64)
    assert clean["order_intents"] == []
    with pytest.raises(ValueError, match="authority"):
        resolve_executable_bidask_time_close_v1(
            poisoned, [], truth_source_receipt_sha256=TRUTH_SHA
        )


def test_split_purges_cross_boundary_maturity_and_never_opens_holdout() -> None:
    train_end = datetime(2026, 1, 2, tzinfo=UTC)
    validation_end = datetime(2026, 1, 3, tzinfo=UTC)
    holdout_end = datetime(2026, 1, 4, tzinfo=UTC)
    signals = [
        _signal(activation=train_end - timedelta(hours=4), source_sha="1" * 64),
        _signal(activation=train_end - timedelta(minutes=1), source_sha="2" * 64),
        _signal(activation=validation_end - timedelta(hours=4), source_sha="3" * 64),
        _signal(activation=validation_end - timedelta(minutes=1), source_sha="4" * 64),
        _signal(activation=validation_end + timedelta(hours=4), source_sha="5" * 64),
        _signal(activation=holdout_end, source_sha="6" * 64),
    ]

    receipt = assign_fixed_split_roles_v1(
        signals,
        train_end_utc=train_end,
        validation_end_utc=validation_end,
        holdout_end_utc=holdout_end,
    )

    assert [row["role"] for row in receipt["assignments"]] == [
        "TRAIN",
        "PURGED_TRAIN_VALIDATION",
        "VALIDATION",
        "PURGED_VALIDATION_HOLDOUT",
        "RESERVED_HOLDOUT_UNOPENED",
        "OUTSIDE_FIXED_WINDOW",
    ]
    assert receipt["holdout_outcomes_opened"] is False
    assert receipt["selection_allowed_roles"] == ["TRAIN", "VALIDATION"]


def test_selection_keeps_182_denominator_and_chooses_simplest_within_one_se() -> None:
    metrics = [
        _metric("H01:DIRECT:BASE", "TRAIN", daily=1.0),
        _metric("H01:DIRECT:BASE", "VALIDATION", daily=1.0),
        _metric("H01:DIRECT:TP050", "TRAIN", daily=1.2),
        _metric("H01:DIRECT:TP050", "VALIDATION", daily=1.1),
    ]

    receipt = select_validation_one_se_v1(metrics)

    assert receipt["multiple_testing_denominator"] == 182
    assert receipt["provided_metric_row_count"] == 4
    assert receipt["selected_family_count"] == 1
    assert receipt["selected"] == [
        {
            "hypothesis_id": "H01",
            "candidate_id": "H01:DIRECT:BASE",
            "best_candidate_id": "H01:DIRECT:TP050",
            "best_mean_daily_post_cost_pips": 1.1,
            "best_standard_error_daily_post_cost_pips": 0.2,
            "one_se_floor": 0.9,
            "selected_by_simplicity_within_one_se": True,
        }
    ]
    assert receipt["holdout_metrics_consumed"] is False
    assert receipt["selection_status"] == "STRUCTURAL_DRY_RUN_ONLY"
    assert receipt["metric_provenance_binding_verified"] is False
    assert receipt["locked_portfolio_candidate_ids"] == []
    assert receipt["provisional_validation_candidate_ids"] == ["H01:DIRECT:BASE"]
    missing = next(
        row
        for row in receipt["candidate_economics"]
        if row["candidate_id"] == "H07:INVERSE:TTL180"
    )
    assert missing["holm_adjusted_p_value"] == 1.0
    assert missing["eligible"] is False


def test_selection_rejects_holdout_and_nonpositive_economics() -> None:
    with pytest.raises(ValueError, match="HOLDOUT"):
        select_validation_one_se_v1([_metric("H01:DIRECT:BASE", "HOLDOUT", daily=5.0)])

    receipt = select_validation_one_se_v1(
        [
            _metric("H01:DIRECT:BASE", "TRAIN", daily=-1.0),
            _metric("H01:DIRECT:BASE", "VALIDATION", daily=-1.0),
        ]
    )
    assert receipt["selected_family_count"] == 0
    assert receipt["locked_portfolio_candidate_ids"] == []


def test_selection_does_not_round_holm_value_into_false_alpha_pass() -> None:
    just_above_alpha_after_holm = 0.0500004 / PLANNED_CANDIDATE_COUNT
    receipt = select_validation_one_se_v1(
        [
            _metric("H01:DIRECT:BASE", "TRAIN", daily=1.0),
            _metric(
                "H01:DIRECT:BASE",
                "VALIDATION",
                daily=1.0,
                p_value=just_above_alpha_after_holm,
            ),
        ]
    )

    candidate = receipt["candidate_economics"][0]
    assert candidate["holm_adjusted_p_value"] > 0.05
    assert candidate["validation_gate_passed"] is False
    assert receipt["selected_family_count"] == 0
