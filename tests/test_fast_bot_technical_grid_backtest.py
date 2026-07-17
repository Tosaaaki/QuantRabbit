from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

import quant_rabbit.fast_bot_technical_grid_backtest as grid_module
from quant_rabbit.fast_bot_technical_grid_backtest import (
    PLANNED_CANDIDATE_COUNT,
    CausalTimeframeFeature,
    assign_fixed_split_roles_v1,
    build_fast_bot_technical_grid_historical_policy_v1,
    build_verified_fast_bot_technical_grid_metrics_v1,
    build_causal_technical_feature_snapshot_v1,
    build_fast_bot_technical_grid_catalog_v1,
    compile_fast_bot_technical_grid_base_vehicle_v1,
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
        "cluster_count": 30,
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


@pytest.mark.parametrize(
    ("side", "entry", "target", "stop", "candle"),
    [
        (
            "LONG",
            1.1002,
            1.1005,
            1.0990,
            _candle(
                0,
                bid_o=1.1000,
                bid_h=1.1006,
                bid_l=1.0999,
                bid_c=1.1004,
            ),
        ),
        (
            "SHORT",
            1.1000,
            1.0995,
            1.1010,
            _candle(
                0,
                bid_o=1.1002,
                bid_h=1.1003,
                bid_l=1.0993,
                bid_c=1.0996,
            ),
        ),
    ],
)
def test_intrabar_stop_entry_target_only_is_causally_take_profit(
    side: str,
    entry: float,
    target: float,
    stop: float,
    candle: S5BidAskCandle,
) -> None:
    signal = _signal(
        source_side=side,
        side=side,
        order_type="STOP",
        entry=entry,
        target=target,
        stop=stop,
    )

    outcome = resolve_executable_bidask_time_close_v1(
        signal, [candle], truth_source_receipt_sha256=TRUTH_SHA
    )

    assert outcome["fill_kind"] == "INTRABAR_TRIGGER"
    assert outcome["exit_reason"] == "TAKE_PROFIT_FILL_S5_AFTER_STOP_ENTRY"
    assert outcome["post_cost_realized_pips"] > 0.0
    assert outcome["ambiguous_same_s5"] is False


@pytest.mark.parametrize(
    ("side", "entry", "target", "stop", "candle"),
    [
        (
            "LONG",
            1.1002,
            1.1005,
            1.0990,
            _candle(
                0,
                bid_o=1.1004,
                bid_h=1.1006,
                bid_l=1.1000,
                bid_c=1.1003,
            ),
        ),
        (
            "SHORT",
            1.1000,
            1.0995,
            1.1010,
            _candle(
                0,
                bid_o=1.0998,
                bid_h=1.1002,
                bid_l=1.0993,
                bid_c=1.0997,
            ),
        ),
    ],
)
def test_intrabar_limit_entry_target_only_is_charged_the_full_stop_loss(
    side: str,
    entry: float,
    target: float,
    stop: float,
    candle: S5BidAskCandle,
) -> None:
    signal = _signal(
        source_side=side,
        side=side,
        order_type="LIMIT",
        entry=entry,
        target=target,
        stop=stop,
    )

    outcome = resolve_executable_bidask_time_close_v1(
        signal, [candle], truth_source_receipt_sha256=TRUTH_SHA
    )

    # The favorable extreme may precede the intrabar LIMIT fill, but the
    # contract charges every ambiguous fill-candle exit the full SL rather
    # than deleting the filled trade from the cohort as unresolvable.
    assert outcome["fill_kind"] == "INTRABAR_TRIGGER"
    assert outcome["status"] == "MATURE_FILLED_STOP_LOSS"
    assert outcome["exit_reason"] == "STOP_LOSS_AMBIGUOUS_FILL_S5"
    assert outcome["scorecard_result_available"] is True
    assert outcome["post_cost_realized_pips"] < 0.0
    assert outcome["ambiguous_same_s5"] is True


def test_open_fill_target_only_is_take_profit_not_stop() -> None:
    signal = _signal(entry=1.1002, target=1.1005, stop=1.0990)
    candle = _candle(
        0,
        bid_o=1.1003,
        bid_h=1.1006,
        bid_l=1.1002,
        bid_c=1.1005,
    )

    outcome = resolve_executable_bidask_time_close_v1(
        signal, [candle], truth_source_receipt_sha256=TRUTH_SHA
    )

    assert outcome["fill_kind"] == "OPEN_TRIGGER"
    assert outcome["exit_reason"] == "TAKE_PROFIT_FILL_S5_AFTER_OPEN_FILL"
    assert outcome["post_cost_realized_pips"] > 0.0


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


def test_split_purges_maturity_exactly_on_exclusive_truth_boundary() -> None:
    train_end = datetime(2026, 1, 2, tzinfo=UTC)
    validation_end = datetime(2026, 1, 3, tzinfo=UTC)
    holdout_end = datetime(2026, 1, 4, tzinfo=UTC)
    train_equal = _signal(
        activation=train_end - timedelta(seconds=270),
        source_sha="7" * 64,
    )
    validation_equal = _signal(
        activation=validation_end - timedelta(seconds=270),
        source_sha="8" * 64,
    )

    receipt = assign_fixed_split_roles_v1(
        [train_equal, validation_equal],
        train_end_utc=train_end,
        validation_end_utc=validation_end,
        holdout_end_utc=holdout_end,
    )

    assert [row["role"] for row in receipt["assignments"]] == [
        "PURGED_TRAIN_VALIDATION",
        "PURGED_VALIDATION_HOLDOUT",
    ]


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
            "selection_data_role": "TRAIN",
            "validation_role": "UNCHANGED_CONFIRMATION_GATE_ONLY",
            "best_mean_daily_post_cost_pips": 1.2,
            "best_standard_error_daily_post_cost_pips": 0.2,
            "one_se_floor": 1.0,
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


def test_selection_alpha_is_immutable_exact_point_zero_five() -> None:
    with pytest.raises(ValueError, match="exactly 0.05"):
        select_validation_one_se_v1([], alpha=0.049)


def test_validation_cannot_replace_the_candidate_frozen_on_train() -> None:
    receipt = select_validation_one_se_v1(
        [
            _metric("H01:DIRECT:BASE", "TRAIN", daily=1.2),
            _metric("H01:DIRECT:BASE", "VALIDATION", daily=1.2, p_value=1.0),
            _metric("H01:DIRECT:TP050", "TRAIN", daily=1.0),
            _metric("H01:DIRECT:TP050", "VALIDATION", daily=1.0),
        ]
    )

    assert receipt["selected"] == []
    assert receipt["provisional_validation_candidate_ids"] == []
    passing_alternative = next(
        row
        for row in receipt["candidate_economics"]
        if row["candidate_id"] == "H01:DIRECT:TP050"
    )
    assert passing_alternative["validation_gate_passed"] is True


def test_self_sealed_metric_receipt_cannot_grant_economic_selection() -> None:
    catalog = build_fast_bot_technical_grid_catalog_v1()
    metrics = []
    for candidate in catalog["candidates"]:
        candidate_id = candidate["candidate_id"]
        daily = 1.0 if candidate_id == "H01:DIRECT:BASE" else -1.0
        metrics.extend(
            [
                _metric(candidate_id, "TRAIN", daily=daily),
                _metric(candidate_id, "VALIDATION", daily=daily),
            ]
        )
    forged = grid_module._seal(
        {
            "contract": grid_module.ECONOMIC_METRIC_RECEIPT_CONTRACT_V1,
            "selection_metrics": metrics,
        }
    )

    receipt = grid_module.select_verified_validation_one_se_v1(forged)

    assert receipt["selection_status"] == (
        "BLOCKED_COMPLETE_CAUSAL_SIGNAL_UNIVERSE_NOT_PROVED"
    )
    assert receipt["economic_conclusion_allowed"] is False
    assert receipt["metric_provenance_binding_verified"] is False
    assert receipt["metric_outcome_reexecution_verified"] is False
    assert receipt["backtest_complete"] is False
    assert receipt["cohort_evaluation_complete"] is False
    assert receipt["selected"] == []
    assert receipt["historical_survivor_candidate_ids"] == []
    assert receipt["observed_cohort_pattern_candidate_ids"] == ["H01:DIRECT:BASE"]
    assert all(row["eligible"] is False for row in receipt["candidate_economics"])


def _fake_compiler_sources(
    *, hypothesis_id: str = "H03"
) -> tuple[dict[str, Any], dict[str, Any]]:
    pair = "EUR_USD"
    feature_rows = [
        {
            "timeframe": timeframe,
            "complete_candle_close_utc": DECISION.isoformat(),
        }
        for timeframe in ("M1", "M5", "M15", "M30", "H1", "H4", "D")
    ]
    feature = {
        "pair": pair,
        "contract_sha256": "1" * 64,
        "timeframes": feature_rows,
    }
    arm = {
        "arm_id": "BASE",
        "entry_ttl_seconds": 90,
        "max_hold_seconds": 900,
        "entry": 1.1000,
        "take_profit": 1.1010,
        "stop_loss": 1.0990,
    }
    candidate_body = {
        "candidate_id": "seat-candidate",
        "side": "LONG",
        "method": "RANGE_ROTATION",
        "arms": [arm],
    }
    candidate = {
        **candidate_body,
        "candidate_sha256": grid_module._canonical_sha(candidate_body),
    }
    seat = {
        "pair": pair,
        "seat_id": "seat-id",
        "contract_sha256": "2" * 64,
        "m1_closed_candle_utc": DECISION.isoformat(),
        "candidates": [candidate],
    }
    binding = {
        "binding_type": "EXACT_FROZEN_LEARNING_SEAT_BASE_ARM_REFERENCE",
        "learning_seat_contract_sha256": seat["contract_sha256"],
        "learning_seat_id": seat["seat_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate["candidate_sha256"],
        "side": "LONG",
        "method": "RANGE_ROTATION",
        "arm_id": "BASE",
        "arm_sha256": grid_module._canonical_sha(arm),
        "numeric_geometry_embedded": False,
        "source_resolution_required": True,
    }
    vehicle = {
        "hypothesis_id": hypothesis_id,
        "predicted_side": "LONG",
        "proxy_binding": binding,
        "execution": None,
        "scoring_vehicle_available": True,
        "vehicle_sha256": "3" * 64,
    }
    vehicle_shadow = {
        "pair": pair,
        "activation_at_utc": DECISION.isoformat(),
        "scorecard_eligible": True,
        "contract_sha256": "4" * 64,
        "vehicles": [vehicle],
    }
    sources = {
        "technical_feature_snapshot": feature,
        "technical_hypothesis_shadow": {"contract_sha256": "5" * 64},
        "episode_anchor": {"anchor": True},
        "episode_route": {"route": True},
        "learning_seat": seat,
        "confirmed_at_utc": DECISION.isoformat(),
        "technical_vehicle_shadow_v2": vehicle_shadow,
    }
    return sources, vehicle_shadow


def test_compiler_requires_canonical_proxy_and_never_economizes_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, vehicle_shadow = _fake_compiler_sources()
    monkeypatch.setattr(
        grid_module,
        "build_fast_bot_technical_hypothesis_vehicles_v2",
        lambda **_kwargs: vehicle_shadow,
    )
    monkeypatch.setattr(
        grid_module,
        "technical_hypothesis_vehicle_shadow_v2_valid",
        lambda *_args, **_kwargs: True,
    )

    direct = compile_fast_bot_technical_grid_base_vehicle_v1(
        hypothesis_id="H03", orientation="DIRECT", **sources
    )
    inverse = compile_fast_bot_technical_grid_base_vehicle_v1(
        hypothesis_id="H03", orientation="INVERSE", **sources
    )

    assert direct["economic_backtest_eligible"] is True
    assert direct["geometry_source"] == "CANONICAL_V2_EXACT_LEARNING_SEAT_BASE_ARM"
    assert direct["geometry"]["order_type"] == "LIMIT"
    assert inverse["economic_backtest_eligible"] is False
    assert inverse["geometry"] is None
    assert (
        "INVERSE_HAS_NO_CANONICAL_ACTUAL_SIDE_V2_GEOMETRY"
        in inverse["economic_ineligibility_reasons"]
    )

    broken = deepcopy(vehicle_shadow)
    broken["vehicles"][0]["proxy_binding"] = None
    monkeypatch.setattr(
        grid_module,
        "build_fast_bot_technical_hypothesis_vehicles_v2",
        lambda **_kwargs: broken,
    )
    with pytest.raises(ValueError, match="canonical V2 proxy"):
        compile_fast_bot_technical_grid_base_vehicle_v1(
            hypothesis_id="H03",
            orientation="DIRECT",
            **{**sources, "technical_vehicle_shadow_v2": broken},
        )


def test_compiler_rejects_stale_higher_timeframe_and_non_s5_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, vehicle_shadow = _fake_compiler_sources()
    monkeypatch.setattr(
        grid_module,
        "build_fast_bot_technical_hypothesis_vehicles_v2",
        lambda **_kwargs: vehicle_shadow,
    )
    monkeypatch.setattr(
        grid_module,
        "technical_hypothesis_vehicle_shadow_v2_valid",
        lambda *_args, **_kwargs: True,
    )
    stale_sources = deepcopy(sources)
    h1_row = next(
        row
        for row in stale_sources["technical_feature_snapshot"]["timeframes"]
        if row["timeframe"] == "H1"
    )
    h1_row["complete_candle_close_utc"] = (DECISION - timedelta(hours=3)).isoformat()
    with pytest.raises(ValueError, match="stale or future-dated: H1"):
        compile_fast_bot_technical_grid_base_vehicle_v1(
            hypothesis_id="H03", orientation="DIRECT", **stale_sources
        )

    non_grid_shadow = deepcopy(vehicle_shadow)
    non_grid_shadow["activation_at_utc"] = (DECISION + timedelta(seconds=1)).isoformat()
    monkeypatch.setattr(
        grid_module,
        "build_fast_bot_technical_hypothesis_vehicles_v2",
        lambda **_kwargs: non_grid_shadow,
    )
    with pytest.raises(ValueError, match="exact S5 grid"):
        compile_fast_bot_technical_grid_base_vehicle_v1(
            hypothesis_id="H03",
            orientation="DIRECT",
            **{**sources, "technical_vehicle_shadow_v2": non_grid_shadow},
        )


def _economic_signal_with_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    compiler = grid_module._seal(
        {
            "contract": grid_module.BASE_VEHICLE_COMPILER_CONTRACT_V1,
            "pair": "EUR_USD",
            "hypothesis_id": "H01",
            "orientation": "DIRECT",
            "source_predicted_side": "LONG",
            "side": "LONG",
            "activation_at_utc": DECISION.isoformat(),
            "causal_source_sha256": SOURCE_SHA,
            "geometry": {
                "order_type": "STOP",
                "entry_price": 1.1002,
                "base_take_profit_price": 1.1020,
                "base_stop_loss_price": 1.0980,
            },
            "economic_backtest_eligible": True,
        }
    )
    monkeypatch.setattr(
        grid_module, "_compile_from_input_bundle", lambda _value: compiler
    )
    diagnostic = freeze_fast_bot_technical_grid_signal_v1(
        pair="EUR_USD",
        hypothesis_id="H01",
        orientation="DIRECT",
        source_predicted_side="LONG",
        side="LONG",
        arm_id="HOLD180",
        order_type="STOP",
        activation_at_utc=DECISION,
        entry_price=1.1002,
        base_take_profit_price=1.1020,
        base_stop_loss_price=1.0980,
        causal_source_sha256=SOURCE_SHA,
        compiler_receipt=compiler,
    )
    assert diagnostic["economic_backtest_eligible"] is False
    assert diagnostic["strategy_evaluator_binding_verified"] is False
    signal = freeze_fast_bot_technical_grid_signal_v1(
        pair="EUR_USD",
        hypothesis_id="H01",
        orientation="DIRECT",
        source_predicted_side="LONG",
        side="LONG",
        arm_id="HOLD180",
        order_type="STOP",
        activation_at_utc=DECISION,
        entry_price=1.1002,
        base_take_profit_price=1.1020,
        base_stop_loss_price=1.0980,
        causal_source_sha256=SOURCE_SHA,
        compiler_receipt=compiler,
        compiler_inputs={"deep": "patched"},
    )
    return signal, compiler


def _resealed_signal_variant(
    signal: dict[str, Any], mutations: dict[str, Any]
) -> dict[str, Any]:
    body = {
        key: item
        for key, item in {**signal, **mutations}.items()
        if key not in {"signal_id", "signal_sha256"}
    }
    body["signal_id"] = grid_module._canonical_sha(
        grid_module._signal_identity_payload(body)
    )[:24]
    return {**body, "signal_sha256": grid_module._canonical_sha(body)}


def test_resealed_hindsight_geometry_or_pair_cannot_reuse_a_real_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal, _compiler = _economic_signal_with_compiler(monkeypatch)

    # Internally consistent hindsight geometry: every price shifts together,
    # so clocks/geometry stay valid and only the receipt binding can refuse.
    shift = -0.0010
    forged_geometry = _resealed_signal_variant(
        signal,
        {
            "entry_price": signal["entry_price"] + shift,
            "base_take_profit_price": signal["base_take_profit_price"] + shift,
            "base_stop_loss_price": signal["base_stop_loss_price"] + shift,
            "take_profit_price": signal["take_profit_price"] + shift,
            "stop_loss_price": signal["stop_loss_price"] + shift,
        },
    )
    with pytest.raises(ValueError, match="match its compiler receipt"):
        grid_module._validate_frozen_signal(forged_geometry)

    forged_pair = _resealed_signal_variant(signal, {"pair": "GBP_USD"})
    with pytest.raises(ValueError, match="match its compiler receipt"):
        grid_module._validate_frozen_signal(forged_pair)


def test_verified_metric_reexec_rejects_correctly_resealed_fake_pnl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal, compiler = _economic_signal_with_compiler(monkeypatch)
    slice_body = {"contract": "TEST_SLICE", "pair": "EUR_USD"}
    slice_receipt = {
        **slice_body,
        "slice_sha256": grid_module._canonical_sha(slice_body),
    }
    slice_sha = slice_receipt["slice_sha256"]
    candles = (
        _candle(0, bid_o=1.1000, bid_h=1.1003, bid_l=1.0999),
        _candle(180, bid_o=1.1004, bid_h=1.1005, bid_l=1.1003),
    )
    train_boundary = DECISION + timedelta(days=1)
    outcome = resolve_executable_bidask_time_close_v1(
        signal,
        candles,
        truth_source_receipt_sha256=slice_sha,
        truth_provenance_end_utc=train_boundary,
    )
    manifest = {
        "source_root": "/does/not/matter/in/patched-test",
        "expected_pairs": list(grid_module.DEFAULT_TRADER_PAIRS),
        "complete_pair_coverage": True,
        "all_selected_sources_acquisition_receipted": True,
        "manifest_sha256": "d" * 64,
    }
    source_slice = SimpleNamespace(
        acquisition_receipt_proved=True,
        requested_from_utc=DECISION,
        requested_to_utc=DECISION + timedelta(seconds=300),
        aligned_from_utc=DECISION,
        aligned_to_utc=DECISION + timedelta(seconds=300),
        pair="EUR_USD",
        source_manifest_sha256=manifest["manifest_sha256"],
        source_file_sha256="e" * 64,
        candles=candles,
        receipt=lambda: slice_receipt,
    )
    monkeypatch.setattr(
        grid_module, "build_historical_s5_manifest", lambda *_args, **_kwargs: manifest
    )
    monkeypatch.setattr(
        grid_module,
        "load_historical_s5_slices",
        lambda *_args, **_kwargs: (source_slice,),
    )
    monkeypatch.setattr(
        grid_module, "_compile_from_input_bundle", lambda _value: compiler
    )
    observation = {
        "signal": signal,
        "outcome": outcome,
        "compiler_inputs": {"deep": "patched"},
        "data_role": "TRAIN",
        "truth_requested_to_utc": (DECISION + timedelta(seconds=300)).isoformat(),
    }
    split = assign_fixed_split_roles_v1(
        [signal],
        train_end_utc=train_boundary,
        validation_end_utc=DECISION + timedelta(days=2),
        holdout_end_utc=DECISION + timedelta(days=3),
    )

    receipt = build_verified_fast_bot_technical_grid_metrics_v1(
        observations=[observation],
        historical_manifest=manifest,
        split_receipt=split,
    )
    assert receipt["outcome_resolver_reexecution_verified"] is True
    assert receipt["metric_row_count"] == 364

    with pytest.raises(ValueError, match="fixed split"):
        build_verified_fast_bot_technical_grid_metrics_v1(
            observations=[{**observation, "data_role": "VALIDATION"}],
            historical_manifest=manifest,
            split_receipt=split,
        )

    fake = deepcopy(outcome)
    fake["post_cost_realized_pips"] = 999.0
    fake = grid_module._seal(
        {key: value for key, value in fake.items() if key != "contract_sha256"}
    )
    with pytest.raises(ValueError, match="resolver re-execution"):
        build_verified_fast_bot_technical_grid_metrics_v1(
            observations=[{**observation, "outcome": fake}],
            historical_manifest=manifest,
            split_receipt=split,
        )


def test_historical_policy_keeps_every_current_pair_on_h08_no_trade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = {"contract": "VERIFIED_METRIC", "contract_sha256": "e" * 64}
    selection = {
        "contract": "VERIFIED_SELECTION",
        "observed_cohort_pattern_candidate_ids": ["H03:DIRECT:BASE"],
        "historical_survivor_candidate_ids": [],
        "contract_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        grid_module,
        "build_verified_fast_bot_technical_grid_metrics_v1",
        lambda **_kwargs: metric,
    )
    monkeypatch.setattr(
        grid_module,
        "select_verified_validation_one_se_v1",
        lambda _metric: selection,
    )

    policy = build_fast_bot_technical_grid_historical_policy_v1(
        observations=[],
        historical_manifest={},
        split_receipt={},
        metric_receipt=metric,
        selection_receipt=selection,
    )

    assert policy["current_route_count"] == 28
    assert policy["current_go_count"] == 0
    assert policy["all_current_routes_h08_no_trade"] is True
    assert {row["current_hypothesis_id"] for row in policy["current_routes"]} == {"H08"}
    assert {row["current_action"] for row in policy["current_routes"]} == {"NO_TRADE"}
    assert policy["future_evidence_candidate_ids"] == ["H03:DIRECT:BASE"]
    assert policy["historical_survivor_candidate_ids"] == []
