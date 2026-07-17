from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

import quant_rabbit.adaptive_story_s5_grid as grid
from quant_rabbit.adaptive_story_s5_grid import (
    UtcSplit,
    build_story_templates_v2,
    build_story_vehicle_catalog_v2,
    combine_adaptive_story_s5_grid_runs,
    run_adaptive_story_s5_grid,
)
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _candle(
    seconds: int,
    *,
    bid: float = 1.1000,
    ask: float = 1.1002,
    high_offset: float = 0.0001,
    low_offset: float = 0.0001,
) -> S5BidAskCandle:
    return S5BidAskCandle(
        timestamp_utc=BASE + timedelta(seconds=seconds),
        bid_o=bid,
        bid_h=bid + high_offset,
        bid_l=bid - low_offset,
        bid_c=bid,
        ask_o=ask,
        ask_h=ask + high_offset,
        ask_l=ask - low_offset,
        ask_c=ask,
    )


def _split(hours: int = 48) -> tuple[UtcSplit, ...]:
    return (UtcSplit("TRAIN", BASE, BASE + timedelta(hours=hours)),)


def _feature(
    *,
    timeframe: str = "M1",
    completed_at: datetime = BASE + timedelta(minutes=2),
    close: float = 1.1020,
    current_open: float = 1.1000,
    atr: float = 0.0010,
    prior_low: float | None = 1.0950,
    prior_high: float | None = 1.1050,
) -> grid._Feature:
    return grid._Feature(
        timeframe=timeframe,
        completed_at=completed_at,
        close=close,
        atr=atr,
        adx=30.0,
        plus_di=35.0,
        minus_di=15.0,
        ema_fast=1.1010,
        ema_slow=1.1000,
        ema_trend=1.0990,
        rsi=60.0,
        macd_hist=0.001,
        previous_rsi=49.0,
        previous_macd_hist=-0.001,
        previous_atr=atr * 0.9,
        previous_adx=29.0,
        prior_high=prior_high,
        prior_low=prior_low,
        current_high=1.1025,
        current_low=1.0995,
        current_open=current_open,
    )


def _vehicle(candidate_id: str = "H21:TIME_1H") -> grid.StoryVehicleV2:
    return next(
        row
        for row in build_story_vehicle_catalog_v2()
        if row.candidate_id == candidate_id
    )


def _position(
    *,
    candidate_id: str = "H21:TIME_1H",
    side: str = "LONG",
    order_mode: str = "MARKET",
    prior_low: float | None = 1.0950,
    prior_high: float | None = 1.1050,
) -> grid._StoryPosition:
    m1 = _feature(prior_low=prior_low, prior_high=prior_high)
    m5 = _feature(
        timeframe="M5",
        prior_low=prior_low,
        prior_high=prior_high,
    )
    return grid._StoryPosition(
        candidate=_vehicle(candidate_id),
        split_name="TRAIN",
        side=side,
        setup_at=BASE + timedelta(minutes=1),
        trigger_at=BASE + timedelta(minutes=2),
        direct_story_side=side,
        frozen_m1=m1,
        frozen_m5=m5,
        frozen_structural_target=1.1100 if side == "LONG" else 1.0900,
        frozen_structural_stop_anchor=(prior_low if side == "LONG" else prior_high),
        chosen_order_mode=order_mode,
        order_selection_reason="TEST",
    )


def _forced_h25_decision(
    template: grid.StoryTemplateV2,
    features: dict[str, grid._Feature],
    prior_features: dict[str, grid._Feature],
    feature_history: dict[str, tuple[grid._Feature, ...]],
    *,
    trigger_at: datetime,
) -> grid._StoryDecision | None:
    del feature_history
    if template.hypothesis_id != "H25":
        return None
    prior_m1 = prior_features["M1"]
    return grid._StoryDecision(
        hypothesis_id="H25",
        side="LONG",
        setup_at=prior_m1.completed_at,
        trigger_at=trigger_at,
        structural_target=1.2000,
        structural_stop_anchor=1.0950,
        setup_evidence={"test_setup": True},
        trigger_evidence={
            "trigger_close": features["M1"].close,
            "test_trigger": True,
        },
    )


def test_catalog_has_ten_stories_five_contextual_exits_and_control() -> None:
    templates = build_story_templates_v2()
    vehicles = build_story_vehicle_catalog_v2()

    assert [row.hypothesis_id for row in templates] == [
        *(f"H{number}" for number in range(21, 31)),
        "H31",
    ]
    assert len(vehicles) == 51
    assert sum(not row.no_trade_control for row in vehicles) == 50
    assert sum(row.no_trade_control for row in vehicles) == 1
    for template in templates[:-1]:
        rows = [row for row in vehicles if row.hypothesis_id == template.hypothesis_id]
        assert len(rows) == 5
        assert {row.exit_policy_id for row in rows} == set(grid.EXIT_POLICY_IDS)
        assert all(
            row.allowed_order_modes == template.allowed_order_modes for row in rows
        )
        assert all(":MARKET:" not in row.candidate_id for row in rows)
        assert all(":STOP:" not in row.candidate_id for row in rows)
        assert all(":LIMIT:" not in row.candidate_id for row in rows)


def test_unavailable_pair_is_explicit_and_has_no_authority() -> None:
    result = run_adaptive_story_s5_grid(
        "NZD_CHF", (), _split(), unavailable_pairs=("NZD_CHF",)
    )

    assert result["status"] == "UNAVAILABLE"
    assert result["candidate_count"] == 51
    assert result["live_permission"] is False
    assert result["broker_mutation_allowed"] is False
    assert result["order_authority"] == "NONE"


@pytest.mark.parametrize(
    "candles,error",
    [
        ((_candle(0), _candle(0)), "chronological and unique"),
        (
            (
                replace(
                    _candle(0),
                    timestamp_utc=BASE + timedelta(seconds=1),
                ),
            ),
            "five-second UTC grid",
        ),
        (
            (
                replace(
                    _candle(0),
                    ask_o=1.0999,
                    ask_h=1.1000,
                    ask_l=1.0998,
                    ask_c=1.0999,
                ),
            ),
            "crossed",
        ),
    ],
)
def test_strict_exact_s5_input_validation(
    candles: tuple[S5BidAskCandle, ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        run_adaptive_story_s5_grid("EUR_USD", candles, _split())


def test_no_tick_slots_are_counted_and_partial_m5_is_not_used() -> None:
    result = run_adaptive_story_s5_grid(
        "EUR_USD",
        (_candle(0), _candle(55), _candle(60)),
        _split(),
    )

    aggregation = result["aggregation"]
    assert aggregation["source_candle_count"] == 3
    assert aggregation["observed_missing_s5_slots"] == 10
    assert aggregation["synthetic_s5_count"] == 0
    assert aggregation["completed_bucket_counts"]["M1"] == 1
    assert aggregation["completed_bucket_counts"]["M5"] == 0
    assert aggregation["partial_buckets_used_as_features"] is False


def test_context_order_market_requires_completed_high_impulse() -> None:
    template = build_story_templates_v2()[4]

    assert grid._choose_order_mode(template, impulse_ratio=0.99) == (
        "STOP",
        "STORY_NATIVE_RESTING_ORDER",
    )
    assert grid._choose_order_mode(template, impulse_ratio=1.0) == (
        "MARKET",
        "COMPLETED_M1_HIGH_IMPULSE_MARKET",
    )


def test_cost_gate_is_direction_neutral_and_spread_stress_can_block() -> None:
    narrow_results: list[bool | None] = []
    wide_results: list[bool | None] = []
    for side in ("LONG", "SHORT"):
        narrow = _position(side=side, prior_low=None, prior_high=None)
        grid._observe_quote(narrow, _candle(125, bid=1.1000, ask=1.1002))
        narrow_results.append(narrow.cost_gate_passed)
        assert narrow.cost_gate_evidence is not None
        assert narrow.cost_gate_evidence["direction_neutral"] is True

        wide = _position(side=side, prior_low=None, prior_high=None)
        grid._observe_quote(wide, _candle(125, bid=1.0950, ask=1.1050))
        wide_results.append(wide.cost_gate_passed)

    assert narrow_results == [True, True]
    assert wide_results == [False, False]


def test_signal_observation_then_next_real_s5_is_earliest_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_feature(
        bars: tuple[grid._Bar, ...], prior: grid._Feature | None
    ) -> grid._Feature:
        bar = bars[-1]
        return _feature(
            timeframe=bar.timeframe,
            completed_at=bar.end,
            close=bar.close + 0.0020,
            current_open=bar.open,
        )

    monkeypatch.setattr(grid, "_feature", fake_feature)
    monkeypatch.setattr(grid, "_story_decision", _forced_h25_decision)
    candles = (
        _candle(0),
        _candle(60),
        _candle(120),
        _candle(180),
        _candle(240),
        _candle(300),
        _candle(305),
        _candle(310),
        _candle(3910, bid=1.1010, ask=1.1012),
    )
    result = run_adaptive_story_s5_grid("EUR_USD", candles, _split())

    signal = result["signal_audit_rows"][0]
    assert signal["setup_at_utc"] == (BASE + timedelta(minutes=4)).isoformat()
    assert signal["trigger_at_utc"] == (BASE + timedelta(minutes=5)).isoformat()
    assert signal["chosen_order_mode"] == "MARKET"
    assert signal["market_impulse_gate"]["passed"] is True
    assert signal["shadow_counterfactuals"] == [
        {
            "counterfactual_kind": "INVERSE_SIDE",
            "side": "SHORT",
            "shadow_only": True,
            "scorecard_eligible": False,
        },
        {
            "counterfactual_kind": "NEXT_M1_TIME_SHIFT",
            "shift_seconds": 60,
            "shifted_trigger_at_utc": (BASE + timedelta(minutes=6)).isoformat(),
            "shadow_only": True,
            "scorecard_eligible": False,
        },
    ]
    time_trade = next(
        row
        for row in result["trade_audit_rows"]
        if row["candidate_id"] == "H25:TIME_1H"
    )
    assert (
        time_trade["quote_observed_at_utc"]
        == (BASE + timedelta(seconds=305)).isoformat()
    )
    assert time_trade["entry_at_utc"] == (BASE + timedelta(seconds=310)).isoformat()
    assert time_trade["t_setup_lt_t_trigger_lt_t_entry"] is True


def test_current_and_future_s5_mutation_cannot_change_completed_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_feature(
        bars: tuple[grid._Bar, ...], prior: grid._Feature | None
    ) -> grid._Feature:
        bar = bars[-1]
        return _feature(
            timeframe=bar.timeframe,
            completed_at=bar.end,
            close=bar.close + 0.0020,
            current_open=bar.open,
        )

    monkeypatch.setattr(grid, "_feature", fake_feature)
    monkeypatch.setattr(grid, "_story_decision", _forced_h25_decision)
    prefix = (_candle(0), _candle(60), _candle(120), _candle(180), _candle(240))
    base = run_adaptive_story_s5_grid(
        "EUR_USD",
        (*prefix, _candle(300), _candle(305), _candle(310)),
        _split(),
    )
    mutated = run_adaptive_story_s5_grid(
        "EUR_USD",
        (
            *prefix,
            _candle(300, bid=1.1500, ask=1.1502),
            _candle(305, bid=1.1600, ask=1.1602),
            _candle(310, bid=1.1700, ask=1.1702),
        ),
        _split(),
    )

    assert base["signal_audit_rows"][0] == mutated["signal_audit_rows"][0]


def test_exact_bid_ask_separates_gross_spread_and_net() -> None:
    position = _position()
    position.quote_observed_at = BASE + timedelta(seconds=125)
    position.filled_at = BASE + timedelta(seconds=130)
    position.entry_exec = 1.1002
    position.entry_mid = 1.1001
    position.structural_stop = 1.0992
    position.initial_risk_price = 0.0010
    position.exit_due_at = BASE + timedelta(hours=1)
    candle = _candle(3600, bid=1.1008, ask=1.1010)

    trade = grid._finish(
        position,
        candle,
        exit_exec=candle.bid_o,
        reason="MANDATORY_TIME_CLOSE",
        pip_factor=10_000.0,
        exit_mid_observable=True,
    )

    assert trade["gross_mid_pips"] == pytest.approx(8.0)
    assert trade["spread_drag_pips"] == pytest.approx(2.0)
    assert trade["exact_net_pips"] == pytest.approx(6.0)
    assert trade["exact_net_r"] == pytest.approx(0.6)


def test_same_s5_target_and_structural_stop_is_stop_first() -> None:
    position = _position(candidate_id="H21:PROFIT_FIRST_24H")
    position.quote_observed_at = BASE + timedelta(seconds=125)
    position.filled_at = BASE + timedelta(seconds=130)
    position.entry_exec = 1.1002
    position.entry_mid = 1.1001
    position.structural_stop = 1.0992
    position.initial_risk_price = 0.0010
    position.take_profit = 1.1012
    position.exit_due_at = BASE + timedelta(hours=24)
    candle = _candle(
        135,
        bid=1.1000,
        ask=1.1002,
        high_offset=0.0013,
        low_offset=0.0010,
    )

    trade = grid._resolve_filled(position, candle, pip_factor=10_000.0)

    assert trade is not None
    assert trade["reason"] == "STRUCTURAL_STOP_SAME_S5"
    assert trade["exact_net_r"] < 0.0


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_open_target_gap_uses_executable_open_before_intrabar_stop_first(
    side: str,
) -> None:
    position = _position(candidate_id="H21:PROFIT_FIRST_24H", side=side)
    position.quote_observed_at = BASE + timedelta(seconds=125)
    position.filled_at = BASE + timedelta(seconds=130)
    position.entry_exec = 1.1002 if side == "LONG" else 1.1000
    position.entry_mid = 1.1001
    position.structural_stop = 1.0992 if side == "LONG" else 1.1010
    position.initial_risk_price = 0.0010
    position.take_profit = 1.1012 if side == "LONG" else 1.0990
    position.exit_due_at = BASE + timedelta(hours=24)
    candle = (
        _candle(
            135,
            bid=1.1020,
            ask=1.1022,
            high_offset=0.0002,
            low_offset=0.0030,
        )
        if side == "LONG"
        else _candle(
            135,
            bid=1.0980,
            ask=1.0982,
            high_offset=0.0030,
            low_offset=0.0002,
        )
    )

    trade = grid._resolve_filled(position, candle, pip_factor=10_000.0)

    assert trade is not None
    assert trade["reason"] == "PROFIT_FIRST_TARGET_GAP"
    assert trade["exit_exec"] == (candle.bid_o if side == "LONG" else candle.ask_o)
    assert trade["exact_net_r"] > 0.0


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_open_between_barriers_keeps_same_s5_stop_first(side: str) -> None:
    position = _position(candidate_id="H21:PROFIT_FIRST_24H", side=side)
    position.quote_observed_at = BASE + timedelta(seconds=125)
    position.filled_at = BASE + timedelta(seconds=130)
    position.entry_exec = 1.1002 if side == "LONG" else 1.1000
    position.entry_mid = 1.1001
    position.structural_stop = 1.0992 if side == "LONG" else 1.1010
    position.initial_risk_price = 0.0010
    position.take_profit = 1.1012 if side == "LONG" else 1.0990
    position.exit_due_at = BASE + timedelta(hours=24)
    candle = _candle(
        135,
        bid=1.1000,
        ask=1.1002,
        high_offset=0.0013,
        low_offset=0.0013,
    )

    trade = grid._resolve_filled(position, candle, pip_factor=10_000.0)

    assert trade is not None
    assert trade["reason"] == "STRUCTURAL_STOP_SAME_S5"
    assert trade["exact_net_r"] < 0.0


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_intrabar_limit_fill_candle_never_claims_a_target_only_print(
    side: str,
) -> None:
    position = _position(
        candidate_id="H21:PROFIT_FIRST_24H", side=side, order_mode="LIMIT"
    )
    position.quote_observed_at = BASE + timedelta(seconds=125)
    position.filled_at = BASE + timedelta(seconds=130)
    position.entry_exec = 1.1000
    position.entry_mid = None
    position.entry_fill_kind = "INTRABAR_TRIGGER"
    position.structural_stop = 1.0990 if side == "LONG" else 1.1010
    position.initial_risk_price = 0.0010
    position.take_profit = 1.1012 if side == "LONG" else 1.0988
    position.exit_due_at = BASE + timedelta(hours=24)
    candle = (
        _candle(
            130,
            bid=1.1000,
            ask=1.1002,
            high_offset=0.0013,
            low_offset=0.0002,
        )
        if side == "LONG"
        else _candle(
            130,
            bid=1.1000,
            ask=1.1002,
            high_offset=0.0002,
            low_offset=0.0013,
        )
    )

    trade = grid._resolve_intrabar_fill_candle(position, candle, pip_factor=10_000.0)

    assert trade is None


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_intrabar_fill_candle_charges_stop_conservatively_without_open_gap(
    side: str,
) -> None:
    position = _position(side=side, order_mode="STOP")
    position.quote_observed_at = BASE + timedelta(seconds=125)
    position.filled_at = BASE + timedelta(seconds=130)
    position.entry_exec = 1.1000
    position.entry_mid = None
    position.entry_fill_kind = "INTRABAR_TRIGGER"
    position.structural_stop = 1.0990 if side == "LONG" else 1.1010
    position.initial_risk_price = 0.0010
    position.take_profit = 1.1020 if side == "LONG" else 1.0980
    position.exit_due_at = BASE + timedelta(hours=1)
    candle = (
        _candle(
            130,
            bid=1.0980,
            ask=1.0982,
            high_offset=0.0022,
            low_offset=0.0001,
        )
        if side == "LONG"
        else _candle(
            130,
            bid=1.1018,
            ask=1.1020,
            high_offset=0.0001,
            low_offset=0.0022,
        )
    )

    trade = grid._resolve_intrabar_fill_candle(position, candle, pip_factor=10_000.0)

    assert trade is not None
    assert trade["reason"] == "STRUCTURAL_STOP_ON_INTRABAR_FILL_S5_CONSERVATIVE"
    assert trade["exit_exec"] == position.structural_stop
    assert trade["reason"] != "STRUCTURAL_STOP_GAP"
    assert trade["exact_net_r"] == pytest.approx(-1.0)


def test_profit_first_has_mandatory_24h_executable_loss_close() -> None:
    position = _position(candidate_id="H21:PROFIT_FIRST_24H")
    position.quote_observed_at = BASE + timedelta(seconds=125)
    position.filled_at = BASE + timedelta(seconds=130)
    position.entry_exec = 1.1002
    position.entry_mid = 1.1001
    position.structural_stop = 1.0900
    position.initial_risk_price = 0.0102
    position.take_profit = 1.1104
    position.exit_due_at = BASE + timedelta(seconds=130, hours=24)
    candle = _candle(130 + 24 * 3600, bid=1.0990, ask=1.0992)

    trade = grid._resolve_filled(position, candle, pip_factor=10_000.0)

    assert trade is not None
    assert trade["reason"] == "MANDATORY_TIME_CLOSE"
    assert trade["exit_exec"] == candle.bid_o
    assert trade["exact_net_pips"] == pytest.approx(-12.0)
    assert trade["exact_net_r"] < 0.0


def test_time_close_does_not_look_inside_candle_after_due_open() -> None:
    position = _position(candidate_id="H21:TIME_1H")
    position.quote_observed_at = BASE + timedelta(seconds=125)
    position.filled_at = BASE + timedelta(seconds=130)
    position.entry_exec = 1.1002
    position.entry_mid = 1.1001
    position.structural_stop = 1.0990
    position.initial_risk_price = 0.0012
    position.exit_due_at = BASE + timedelta(seconds=130, hours=1)
    candle = _candle(
        130 + 3600,
        bid=1.1005,
        ask=1.1007,
        low_offset=0.0020,
    )

    trade = grid._resolve_filled(position, candle, pip_factor=10_000.0)

    assert trade is not None
    assert trade["reason"] == "MANDATORY_TIME_CLOSE"
    assert trade["exit_exec"] == 1.1005


def test_split_embargo_is_entry_ttl_plus_full_24h() -> None:
    result = run_adaptive_story_s5_grid("EUR_USD", (), _split())

    assert result["split_embargo_seconds"] == (
        grid.ENTRY_TTL_SECONDS + grid.MAX_HOLD_SECONDS
    )
    assert result["profit_first_unrealized_loss_zeroed"] is False
    assert result["inverse_counterfactual_shadow_only"] is True
    assert result["time_shift_counterfactual_shadow_only"] is True


def test_candidate_whitelist_rejects_unknown_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        run_adaptive_story_s5_grid(
            "EUR_USD",
            (),
            _split(),
            candidate_ids=("H21:TIME_1H", "H21:TIME_1H"),
        )
    with pytest.raises(ValueError, match="unknown"):
        run_adaptive_story_s5_grid(
            "EUR_USD", (), _split(), candidate_ids=("H99:TIME_1H",)
        )

    result = run_adaptive_story_s5_grid(
        "EUR_USD",
        (),
        _split(),
        candidate_ids=("H21:TIME_1H",),
    )
    assert result["requested_candidate_ids"] == ["H21:TIME_1H"]
    assert result["evaluated_candidate_ids"] == ["H21:TIME_1H"]
    assert len(result["candidate_whitelist_sha256"]) == 64


def test_compression_quantile_is_prior_only_and_needs_eight_observations() -> None:
    history = [
        _feature(
            timeframe="M15",
            completed_at=BASE + timedelta(minutes=15 * index),
            atr=0.0010 + index * 0.0001,
        )
        for index in range(10)
    ]
    before = BASE + timedelta(minutes=15 * 9)
    baseline = grid._prior_atr_quantile(history, before=before)
    mutated = list(history)
    mutated[-1] = replace(mutated[-1], atr=9.0)

    assert baseline is not None
    assert grid._prior_atr_quantile(mutated, before=before) == baseline
    assert grid._prior_atr_quantile(history[:7], before=before) is None


@pytest.mark.parametrize("hypothesis_id", ["H25", "H26"])
def test_compression_story_excludes_the_setup_m15_from_prior_distribution(
    hypothesis_id: str,
) -> None:
    trigger_at = BASE + timedelta(hours=4, minutes=2)
    prior_m15 = replace(
        _feature(
            timeframe="M15",
            completed_at=trigger_at - timedelta(minutes=2),
            atr=0.0005,
        ),
        prior_low=1.0950,
        prior_high=1.1050,
    )
    prior_m1 = replace(
        _feature(completed_at=trigger_at - timedelta(minutes=1)),
        close=1.1000,
        current_open=1.1000,
        current_low=1.0990,
        current_high=1.1010,
    )
    prior = {
        "M1": prior_m1,
        "M5": _feature(
            timeframe="M5",
            completed_at=trigger_at - timedelta(minutes=5),
            atr=0.0010,
        ),
        "M15": prior_m15,
        "H1": _feature(
            timeframe="H1",
            completed_at=trigger_at - timedelta(hours=1),
            prior_low=1.0900,
            prior_high=1.1200,
        ),
    }
    current_m1 = (
        replace(
            _feature(completed_at=trigger_at),
            close=1.1060,
            current_open=1.1040,
            current_low=1.1030,
            current_high=1.1065,
        )
        if hypothesis_id == "H25"
        else replace(
            _feature(completed_at=trigger_at),
            close=1.1000,
            current_open=1.0960,
            current_low=1.0940,
            current_high=1.1010,
        )
    )
    current = {
        **prior,
        "M1": current_m1,
        "M5": replace(prior["M5"], completed_at=trigger_at, atr=0.0011),
    }
    seven_strictly_prior = tuple(
        replace(
            prior_m15,
            completed_at=prior_m15.completed_at - timedelta(minutes=15 * (index + 1)),
            atr=0.0010,
        )
        for index in range(7)
    )
    template = next(
        row for row in build_story_templates_v2() if row.hypothesis_id == hypothesis_id
    )

    assert (
        grid._story_decision(
            template,
            current,
            prior,
            {"M15": (*seven_strictly_prior, prior_m15)},
            trigger_at=trigger_at,
        )
        is None
    )

    eighth_prior = replace(
        prior_m15,
        completed_at=prior_m15.completed_at - timedelta(minutes=15 * 8),
        atr=0.0010,
    )
    assert (
        grid._story_decision(
            template,
            current,
            prior,
            {"M15": (eighth_prior, *seven_strictly_prior, prior_m15)},
            trigger_at=trigger_at,
        )
        is not None
    )


@pytest.mark.parametrize("hypothesis_id", ["H24", "H29", "H30"])
def test_declared_range_stories_require_prior_m15_adx_below_boundary(
    hypothesis_id: str,
) -> None:
    trigger_at = datetime(2026, 1, 5, 8, 5, tzinfo=UTC)
    prior_m1 = replace(
        _feature(completed_at=trigger_at - timedelta(minutes=1)),
        close=1.1060 if hypothesis_id == "H24" else 1.1000,
        current_open=1.1000,
        current_low=1.0990,
        current_high=1.1070,
    )
    current_m1 = replace(
        _feature(completed_at=trigger_at),
        close=1.1060 if hypothesis_id in {"H24", "H29"} else 1.1040,
        current_open=1.1050,
        current_low=1.1040,
        current_high=1.1060,
    )
    if hypothesis_id == "H30":
        current_m1 = replace(current_m1, current_high=1.1060, close=1.1040)
    prior_m15 = replace(
        _feature(
            timeframe="M15",
            completed_at=trigger_at - timedelta(minutes=5),
        ),
        adx=grid.RANGE_ADX_MAX,
        prior_low=1.0950,
        prior_high=1.1050,
    )
    prior = {
        "M1": prior_m1,
        "M5": _feature(timeframe="M5", completed_at=trigger_at - timedelta(minutes=5)),
        "M15": prior_m15,
        "H1": _feature(
            timeframe="H1",
            completed_at=trigger_at - timedelta(hours=1),
            prior_low=1.0900,
            prior_high=1.1200,
        ),
    }
    current = {
        **prior,
        "M1": current_m1,
        "M5": replace(prior["M5"], completed_at=trigger_at),
    }
    template = next(
        row for row in build_story_templates_v2() if row.hypothesis_id == hypothesis_id
    )

    assert (
        grid._story_decision(template, current, prior, {}, trigger_at=trigger_at)
        is None
    )
    prior["M15"] = replace(prior_m15, adx=grid.RANGE_ADX_MAX - 0.01)
    assert (
        grid._story_decision(template, current, prior, {}, trigger_at=trigger_at)
        is not None
    )


@pytest.mark.parametrize("hypothesis_id", ["H23", "H26"])
@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_sweep_stories_record_the_sweep_reaccept_m1_as_the_trigger(
    hypothesis_id: str,
    side: str,
) -> None:
    trigger_at = BASE + timedelta(hours=5)
    prior_m1 = replace(
        _feature(completed_at=trigger_at - timedelta(minutes=1)),
        close=1.1000,
        current_open=1.1000,
        current_low=1.0990,
        current_high=1.1010,
    )
    current_m1 = replace(
        _feature(completed_at=trigger_at),
        close=1.1000,
        current_open=1.1000,
        current_low=1.0940 if side == "LONG" else 1.0990,
        current_high=1.1010 if side == "LONG" else 1.1060,
    )
    prior_m15 = replace(
        _feature(
            timeframe="M15",
            completed_at=trigger_at - timedelta(minutes=15),
            atr=0.0005,
        ),
        adx=20.0,
        prior_low=1.0950,
        prior_high=1.1050,
    )
    prior = {
        "M1": prior_m1,
        "M5": _feature(timeframe="M5", completed_at=trigger_at - timedelta(minutes=5)),
        "M15": prior_m15,
        "H1": _feature(
            timeframe="H1",
            completed_at=trigger_at - timedelta(hours=1),
            prior_low=1.0900,
            prior_high=1.1200,
        ),
    }
    current = {
        **prior,
        "M1": current_m1,
        "M5": replace(prior["M5"], completed_at=trigger_at),
    }
    history = {
        "M15": tuple(
            replace(
                prior_m15,
                completed_at=prior_m15.completed_at
                - timedelta(minutes=15 * (index + 1)),
                atr=0.0010,
            )
            for index in range(8)
        )
    }
    template = next(
        row for row in build_story_templates_v2() if row.hypothesis_id == hypothesis_id
    )

    decision = grid._story_decision(
        template, current, prior, history, trigger_at=trigger_at
    )

    assert decision is not None
    assert decision.side == side
    assert decision.trigger_at == trigger_at
    if side == "LONG":
        assert decision.structural_stop_anchor == current_m1.current_low
        assert decision.trigger_evidence["reaccepted_above"] == 1.0950
    else:
        assert decision.structural_stop_anchor == current_m1.current_high
        assert decision.trigger_evidence["reaccepted_below"] == 1.1050


@pytest.mark.parametrize(
    "impulse_atr,expected", [(0.49, False), (0.50, True), (0.75, True)]
)
def test_h22_reversal_uses_half_atr_story_impulse_threshold(
    impulse_atr: float, expected: bool
) -> None:
    trigger_at = BASE + timedelta(hours=5, minutes=30)
    prior_m5 = _feature(
        timeframe="M5",
        completed_at=trigger_at - timedelta(minutes=5),
        atr=0.0010,
    )
    prior_m1 = replace(
        _feature(completed_at=trigger_at - timedelta(minutes=1)),
        current_open=1.1000,
        close=1.1000 + impulse_atr * 0.0010,
        current_low=1.0990,
        current_high=1.1010,
    )
    prior = {
        "M1": prior_m1,
        "M5": prior_m5,
        "M15": _feature(
            timeframe="M15", completed_at=trigger_at - timedelta(minutes=15)
        ),
        "H1": _feature(timeframe="H1", completed_at=trigger_at - timedelta(hours=1)),
    }
    current = {
        **prior,
        "M1": replace(
            _feature(completed_at=trigger_at),
            close=1.0980,
            current_low=1.0975,
        ),
        "M5": replace(prior_m5, completed_at=trigger_at),
    }
    template = next(
        row for row in build_story_templates_v2() if row.hypothesis_id == "H22"
    )

    decision = grid._story_decision(template, current, prior, {}, trigger_at=trigger_at)

    assert (decision is not None) is expected
    if expected:
        assert decision is not None
        assert decision.side == "SHORT"
        assert decision.setup_evidence["impulse_atr"] == pytest.approx(impulse_atr)


@pytest.mark.parametrize(
    "impulse_atr,expected", [(0.49, False), (0.50, True), (0.75, True)]
)
def test_h27_story_uses_half_atr_impulse_threshold_without_market_upgrade(
    impulse_atr: float, expected: bool
) -> None:
    trigger_at = BASE + timedelta(hours=6)
    prior_m5 = _feature(
        timeframe="M5",
        completed_at=trigger_at - timedelta(minutes=5),
        atr=0.0010,
    )
    prior_m1 = replace(
        _feature(completed_at=trigger_at - timedelta(minutes=1)),
        current_open=1.1000,
        close=1.1000 + impulse_atr * 0.0010,
        current_low=1.0999,
        current_high=1.1010,
    )
    midpoint = (prior_m1.current_open + prior_m1.close) / 2.0
    prior = {
        "M1": prior_m1,
        "M5": prior_m5,
        "M15": _feature(
            timeframe="M15", completed_at=trigger_at - timedelta(minutes=15)
        ),
        "H1": _feature(timeframe="H1", completed_at=trigger_at - timedelta(hours=1)),
    }
    current = {
        **prior,
        "M1": replace(
            _feature(completed_at=trigger_at),
            current_open=midpoint,
            current_low=midpoint,
            close=midpoint + 0.0001,
        ),
        "M5": replace(prior_m5, completed_at=trigger_at),
    }
    template = next(
        row for row in build_story_templates_v2() if row.hypothesis_id == "H27"
    )

    decision = grid._story_decision(template, current, prior, {}, trigger_at=trigger_at)

    assert (decision is not None) is expected
    if expected:
        assert grid._choose_order_mode(template, impulse_ratio=impulse_atr) == (
            "LIMIT",
            "STORY_NATIVE_RESTING_ORDER",
        )


def test_pullback_story_binds_real_prior_setup_state_and_structure() -> None:
    setup_at = BASE + timedelta(minutes=4)
    trigger_at = BASE + timedelta(minutes=5)
    prior_m1 = replace(
        _feature(completed_at=setup_at),
        close=1.1000,
        ema_fast=1.1005,
        current_open=1.1010,
        current_low=1.0990,
        current_high=1.1020,
    )
    prior = {
        "M1": prior_m1,
        "M5": _feature(timeframe="M5", completed_at=setup_at),
        "M15": _feature(timeframe="M15", completed_at=setup_at),
        "H1": _feature(timeframe="H1", completed_at=setup_at),
    }
    current = {
        **prior,
        "M1": replace(
            _feature(completed_at=trigger_at),
            close=1.1030,
            current_high=1.1035,
        ),
        "M5": _feature(timeframe="M5", completed_at=trigger_at),
    }

    decision = grid._story_decision(
        build_story_templates_v2()[0],
        current,
        prior,
        {key: tuple(value for _ in range(8)) for key, value in prior.items()},
        trigger_at=trigger_at,
    )

    assert decision is not None
    assert decision.setup_at == setup_at
    assert decision.trigger_at == trigger_at
    assert decision.side == "LONG"
    assert decision.structural_target == prior["M15"].prior_high
    assert decision.setup_evidence == {
        "pullback_side": "DOWN",
        "trend_side": "LONG",
    }
    assert decision.scorable is True

    no_pullback = dict(prior)
    no_pullback["M1"] = replace(prior_m1, close=1.1010, ema_fast=1.1005)
    assert (
        grid._story_decision(
            build_story_templates_v2()[0],
            current,
            no_pullback,
            {},
            trigger_at=trigger_at,
        )
        is None
    )


def test_dst_opening_window_tracks_london_summer_and_winter() -> None:
    winter_start = datetime(2026, 1, 5, 8, 0, tzinfo=UTC)
    winter = datetime(2026, 1, 5, 8, 5, tzinfo=UTC)
    winter_last = datetime(2026, 1, 5, 8, 15, tzinfo=UTC)
    winter_after = datetime(2026, 1, 5, 8, 16, tzinfo=UTC)
    summer = datetime(2026, 7, 6, 7, 5, tzinfo=UTC)

    assert grid._is_dst_aware_local_open(winter_start)[0] is False
    assert grid._is_dst_aware_local_open(winter) == (True, "Europe/London")
    assert grid._is_dst_aware_local_open(winter_last) == (True, "Europe/London")
    assert grid._is_dst_aware_local_open(winter_after)[0] is False
    assert grid._is_dst_aware_local_open(summer) == (True, "Europe/London")
    assert (
        grid._is_dst_aware_local_open(datetime(2026, 7, 6, 8, 20, tzinfo=UTC))[0]
        is False
    )


def test_cost_gate_uses_frozen_structure_target_without_one_r_fallback() -> None:
    position = _position(side="LONG")
    position.frozen_structural_target = 1.0990

    grid._observe_quote(position, _candle(125, bid=1.1000, ask=1.1002))

    assert position.cost_gate_passed is False
    assert position.cost_gate_evidence is not None
    assert position.cost_gate_evidence["planned_target_room_price"] == 0.0


@pytest.mark.parametrize(
    "pip_factor,base,atr", [(10_000.0, 1.1, 0.001003), (100.0, 150.0, 0.1034)]
)
@pytest.mark.parametrize("order_mode", ["LIMIT", "STOP"])
@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_broker_tick_rounding_preserves_resting_order_direction_and_geometry(
    pip_factor: float,
    base: float,
    atr: float,
    order_mode: str,
    side: str,
) -> None:
    tick = grid._broker_tick_size(pip_factor)
    bid = base + (0.000003 if pip_factor == 10_000.0 else 0.0014)
    ask = base + (0.000017 if pip_factor == 10_000.0 else 0.0036)
    position = _position(side=side, order_mode=order_mode)
    position.frozen_m1 = replace(
        position.frozen_m1,
        current_high=ask + atr * 0.137,
        current_low=bid - atr * 0.137,
    )
    position.frozen_m5 = replace(position.frozen_m5, atr=atr)
    position.frozen_structural_target = (
        base + atr * 8.007 if side == "LONG" else base - atr * 8.007
    )
    position.frozen_structural_stop_anchor = (
        base - atr * 2.003 if side == "LONG" else base + atr * 2.003
    )
    candle = replace(
        _candle(125),
        bid_o=bid,
        bid_h=bid + atr * 0.1,
        bid_l=bid - atr * 0.1,
        bid_c=bid,
        ask_o=ask,
        ask_h=ask + atr * 0.1,
        ask_l=ask - atr * 0.1,
        ask_c=ask,
    )

    grid._observe_quote(position, candle, pip_factor=pip_factor)

    assert position.entry_target is not None
    for price in (
        position.entry_target,
        position.structural_stop,
        position.frozen_structural_target,
    ):
        assert price is not None
        tick_count = Decimal(str(price)) / tick
        assert tick_count == tick_count.to_integral_value()
    if order_mode == "LIMIT":
        assert (
            position.entry_target <= ask
            if side == "LONG"
            else position.entry_target >= bid
        )
    else:
        assert (
            position.entry_target >= ask
            if side == "LONG"
            else position.entry_target <= bid
        )
    assert grid._entry_geometry_valid(position, position.entry_target)
    assert position.cost_gate_evidence is not None
    assert position.cost_gate_evidence["broker_tick_size"] == float(tick)
    assert (
        position.cost_gate_evidence["price_precision_policy"]
        == grid.PRICE_PRECISION_POLICY_V2
    )


@pytest.mark.parametrize(
    ("pip_factor", "half_tick", "long_expected", "short_expected"),
    [
        (10_000.0, 1.100005, 1.10000, 1.10001),
        (100.0, 150.0005, 150.000, 150.001),
    ],
)
def test_trailing_candidate_rounds_to_pair_tick_before_tightening(
    pip_factor: float,
    half_tick: float,
    long_expected: float,
    short_expected: float,
) -> None:
    long_tightened = grid._tighten_trailing_stop_to_tick(
        long_expected - 0.01,
        half_tick,
        side="LONG",
        pip_factor=pip_factor,
    )
    short_tightened = grid._tighten_trailing_stop_to_tick(
        short_expected + 0.01,
        half_tick,
        side="SHORT",
        pip_factor=pip_factor,
    )

    assert long_tightened == long_expected
    assert short_tightened == short_expected
    assert (
        grid._tighten_trailing_stop_to_tick(
            long_expected + 0.01,
            half_tick,
            side="LONG",
            pip_factor=pip_factor,
        )
        == long_expected + 0.01
    )
    assert (
        grid._tighten_trailing_stop_to_tick(
            short_expected - 0.01,
            half_tick,
            side="SHORT",
            pip_factor=pip_factor,
        )
        == short_expected - 0.01
    )


def test_order_creation_geometry_block_is_distinct_from_later_gap_cancel() -> None:
    position = _position(side="LONG", order_mode="MARKET")
    position.frozen_structural_target = 1.1000

    grid._observe_quote(
        position,
        _candle(125, bid=1.1000, ask=1.1002),
        pip_factor=10_000.0,
    )

    assert (
        position.order_creation_geometry_issue
        == "ORDER_CREATION_TAKE_PROFIT_GEOMETRY_INVALID"
    )
    assert position.cost_gate_passed is False


def test_entry_ttl_is_exclusive_at_exact_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_feature(
        bars: tuple[grid._Bar, ...], prior: grid._Feature | None
    ) -> grid._Feature:
        del prior
        bar = bars[-1]
        return _feature(
            timeframe=bar.timeframe,
            completed_at=bar.end,
            close=bar.close + 0.0020,
            current_open=bar.open,
        )

    monkeypatch.setattr(grid, "_feature", fake_feature)
    monkeypatch.setattr(grid, "_story_decision", _forced_h25_decision)
    result = run_adaptive_story_s5_grid(
        "EUR_USD",
        (
            _candle(0),
            _candle(60),
            _candle(120),
            _candle(180),
            _candle(240),
            _candle(300),
            _candle(305),
            _candle(1200),
        ),
        _split(),
        candidate_ids=("H25:TIME_1H",),
    )
    metric = result["all_trials"][0]["by_split"]["TRAIN"]

    assert metric["filled_count"] == 0
    assert metric["unfilled_count"] == 1
    assert metric["reason_counts"]["ENTRY_TTL_EXPIRED"] == 1


@pytest.mark.parametrize(
    ("order_mode", "side", "candidate_id", "gap_bid", "gap_ask", "reason"),
    [
        (
            "LIMIT",
            "LONG",
            "H21:TIME_1H",
            1.0938,
            1.0940,
            "BROKER_ON_FILL_STOP_LOSS_LOSS_CANCEL",
        ),
        (
            "LIMIT",
            "SHORT",
            "H21:TIME_1H",
            1.1060,
            1.1062,
            "BROKER_ON_FILL_STOP_LOSS_LOSS_CANCEL",
        ),
        (
            "STOP",
            "LONG",
            "H25:TIME_1H",
            1.2008,
            1.2010,
            "BROKER_ON_FILL_TAKE_PROFIT_LOSS_CANCEL",
        ),
        (
            "STOP",
            "SHORT",
            "H25:TIME_1H",
            0.9990,
            0.9992,
            "BROKER_ON_FILL_TAKE_PROFIT_LOSS_CANCEL",
        ),
    ],
)
def test_resting_gap_invalid_geometry_is_broker_cancel_not_a_fill_or_trade(
    monkeypatch: pytest.MonkeyPatch,
    order_mode: str,
    side: str,
    candidate_id: str,
    gap_bid: float,
    gap_ask: float,
    reason: str,
) -> None:
    def fake_feature(
        bars: tuple[grid._Bar, ...], prior: grid._Feature | None
    ) -> grid._Feature:
        del prior
        bar = bars[-1]
        return _feature(
            timeframe=bar.timeframe,
            completed_at=bar.end,
            close=bar.close + 0.0020,
            current_open=bar.open,
        )

    def forced_decision(
        template: grid.StoryTemplateV2,
        features: dict[str, grid._Feature],
        prior_features: dict[str, grid._Feature],
        feature_history: dict[str, tuple[grid._Feature, ...]],
        *,
        trigger_at: datetime,
    ) -> grid._StoryDecision:
        del feature_history
        return grid._StoryDecision(
            hypothesis_id=template.hypothesis_id,
            side=side,
            setup_at=prior_features["M1"].completed_at,
            trigger_at=trigger_at,
            structural_target=1.2000 if side == "LONG" else 1.0000,
            structural_stop_anchor=1.0950 if side == "LONG" else 1.1050,
            setup_evidence={"test": True},
            trigger_evidence={"trigger_close": features["M1"].close},
        )

    monkeypatch.setattr(grid, "_feature", fake_feature)
    monkeypatch.setattr(grid, "_story_decision", forced_decision)
    monkeypatch.setattr(
        grid,
        "_choose_order_mode",
        lambda template, *, impulse_ratio: (order_mode, "TEST_ORDER_MODE"),
    )
    result = run_adaptive_story_s5_grid(
        "EUR_USD",
        (
            _candle(0),
            _candle(60),
            _candle(120),
            _candle(180),
            _candle(240),
            _candle(300),
            _candle(305),
            _candle(310, bid=gap_bid, ask=gap_ask),
        ),
        _split(),
        candidate_ids=(candidate_id,),
    )
    metric = result["all_trials"][0]["by_split"]["TRAIN"]

    assert metric["filled_count"] == 0
    assert metric["resolved_count"] == 0
    assert metric["unfilled_count"] == 1
    assert metric["canceled_before_fill_count"] == 1
    assert metric["gap_attempt_canceled_count"] == 1
    assert metric["reason_counts"][reason] == 1
    assert result["trade_audit_rows"] == []
    gap_row = next(
        row for row in result["blocked_audit_rows"] if row["reason"] == reason
    )
    assert gap_row["entry_fill_kind"] == "EXECUTABLE_OPEN_GAP"
    assert gap_row["broker_fill_occurred"] is False
    assert gap_row["economic_outcome_recorded"] is False


@pytest.mark.parametrize(
    ("side", "stop", "target", "fill", "expected_reason"),
    [
        (
            "LONG",
            1.0990,
            1.1100,
            1.0980,
            "BROKER_ON_FILL_STOP_LOSS_LOSS_CANCEL",
        ),
        (
            "LONG",
            1.0990,
            1.1100,
            1.1110,
            "BROKER_ON_FILL_TAKE_PROFIT_LOSS_CANCEL",
        ),
        (
            "SHORT",
            1.1110,
            1.0900,
            1.1120,
            "BROKER_ON_FILL_STOP_LOSS_LOSS_CANCEL",
        ),
        (
            "SHORT",
            1.1110,
            1.0900,
            1.0890,
            "BROKER_ON_FILL_TAKE_PROFIT_LOSS_CANCEL",
        ),
    ],
)
def test_gap_fill_cannot_invert_stop_or_overshoot_target(
    side: str, stop: float, target: float, fill: float, expected_reason: str
) -> None:
    position = _position(side=side)
    position.structural_stop = stop
    position.frozen_structural_target = target

    assert grid._entry_geometry_valid(position, fill) is False
    assert grid._entry_geometry_issue(position, fill) == expected_reason


def test_intrabar_resting_fill_does_not_fabricate_mid_spread_decomposition() -> None:
    position = _position(order_mode="LIMIT")
    position.quote_observed_at = BASE + timedelta(seconds=125)
    position.filled_at = BASE + timedelta(seconds=130)
    position.entry_exec = 1.1000
    position.entry_mid = None
    position.entry_fill_kind = "INTRABAR_TRIGGER"
    position.structural_stop = 1.0990
    position.initial_risk_price = 0.0010
    position.exit_due_at = BASE + timedelta(hours=1)
    candle = _candle(3600, bid=1.1008, ask=1.1010)

    trade = grid._finish(
        position,
        candle,
        exit_exec=candle.bid_o,
        reason="MANDATORY_TIME_CLOSE",
        pip_factor=10_000.0,
        exit_mid_observable=True,
    )

    assert trade["exact_net_pips"] == pytest.approx(8.0)
    assert trade["gross_mid_pips"] is None
    assert trade["spread_drag_pips"] is None
    assert trade["gross_spread_decomposition_status"].startswith("UNAVAILABLE")


def test_combine_uses_complete_zero_filled_daily_vectors_and_sealed_whitelist() -> None:
    splits = (UtcSplit("TRAIN", BASE, BASE + timedelta(days=2)),)
    candidate_ids = ("H21:TIME_1H",)
    first = run_adaptive_story_s5_grid(
        "EUR_USD", (), splits, candidate_ids=candidate_ids
    )
    second = run_adaptive_story_s5_grid(
        "GBP_USD", (), splits, candidate_ids=candidate_ids
    )

    combined = combine_adaptive_story_s5_grid_runs(
        (first, second), splits, candidate_ids=candidate_ids
    )
    metric = combined["candidate_metrics"][0]["by_split"]["TRAIN"]

    assert metric["daily_net_r"] == [
        {"utc_date": "2026-01-01", "exact_net_r": 0.0, "resolved_count": 0},
        {"utc_date": "2026-01-02", "exact_net_r": 0.0, "resolved_count": 0},
    ]
    assert metric["average_daily_net_r"] == 0.0
    assert combined["daily_cluster_basis"] == "ENTRY_UTC_DATE"
    assert combined["candidate_whitelist_sha256"] == first["candidate_whitelist_sha256"]


def test_combine_economic_screen_uses_pooled_pairs_and_entry_days() -> None:
    splits = (UtcSplit("VALIDATION", BASE, BASE + timedelta(days=8)),)
    candidate_ids = ("H21:TIME_1H",)
    runs = []
    for pair in ("EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"):
        run = run_adaptive_story_s5_grid(pair, (), splits, candidate_ids=candidate_ids)
        trial = run["all_trials"][0]
        metric = trial["by_split"]["VALIDATION"]
        metric.update(
            {
                "filled_count": 8,
                "resolved_count": 8,
                "exact_net_r": 8.0,
                "gross_profit_r": 8.0,
                "gross_loss_r": 0.0,
            }
        )
        for day in trial["daily_aggregates_by_split"]["VALIDATION"]:
            day.update(
                {
                    "filled_count": 1,
                    "resolved_count": 1,
                    "exact_net_r": 1.0,
                    "gross_profit_r": 1.0,
                    "gross_loss_r": 0.0,
                }
            )
        run["status"] = "COMPLETE"
        run["aggregation"]["source_candle_count"] = 1
        run["result_sha256"] = grid._canonical_sha(
            {key: value for key, value in run.items() if key != "result_sha256"}
        )
        runs.append(run)

    combined = combine_adaptive_story_s5_grid_runs(
        runs, splits, candidate_ids=candidate_ids
    )
    row = combined["candidate_metrics"][0]
    metric = row["by_split"]["VALIDATION"]
    screen = row["economic_screen_by_split"]["VALIDATION"]

    assert metric["resolved_count"] == 32
    assert metric["active_entry_day_count"] == 8
    assert metric["contributing_pair_count"] == 4
    assert metric["average_daily_net_r"] == 4.0
    assert metric["gross_profit_r"] == 32.0
    assert screen["eligible"] is True
    assert screen["gates"]["no_unresolved_filled"] is True
    assert metric["unresolved_filled_count"] == 0
    assert combined["economic_survivor_ids"] == ["H21:TIME_1H"]


@pytest.mark.parametrize(
    "alias_pair",
    ["eur_usd", " EUR_USD", "EUR_USD ", "EURUSD", "EUR-USD"],
)
def test_combiner_rejects_resealed_noncanonical_pair_aliases(
    alias_pair: str,
) -> None:
    splits = (UtcSplit("TRAIN", BASE, BASE + timedelta(days=2)),)
    candidate_ids = ("H21:TIME_1H",)
    run = run_adaptive_story_s5_grid(
        "EUR_USD", (_candle(0),), splits, candidate_ids=candidate_ids
    )
    run["pair"] = alias_pair
    run["result_sha256"] = grid._canonical_sha(
        {key: value for key, value in run.items() if key != "result_sha256"}
    )

    with pytest.raises(ValueError, match="canonical uppercase AAA_BBB"):
        combine_adaptive_story_s5_grid_runs((run,), splits, candidate_ids=candidate_ids)


def test_combiner_rejects_resealed_unresolved_filled_ownership_contradiction() -> None:
    splits = (UtcSplit("TRAIN", BASE, BASE + timedelta(days=2)),)
    candidate_ids = ("H21:TIME_1H",)
    run = run_adaptive_story_s5_grid(
        "EUR_USD", (_candle(0),), splits, candidate_ids=candidate_ids
    )
    metric = run["all_trials"][0]["by_split"]["TRAIN"]
    assert metric["purged_count"] == 0
    assert metric["truth_window_unresolved_count"] == 0
    metric["unresolved_filled_count"] = 1
    run["result_sha256"] = grid._canonical_sha(
        {key: value for key, value in run.items() if key != "result_sha256"}
    )

    with pytest.raises(ValueError, match="unresolved filled count exceeds"):
        combine_adaptive_story_s5_grid_runs((run,), splits, candidate_ids=candidate_ids)


@pytest.mark.parametrize(
    "tamper",
    [
        "status",
        "story_policy",
        "truth_policy",
        "catalog_digest",
        "evaluator_digest",
        "trial_metadata",
        "daily_summary",
        "candidate_scope",
    ],
)
def test_combiner_rejects_resealed_policy_trial_and_daily_tampering(
    tamper: str,
) -> None:
    splits = (UtcSplit("TRAIN", BASE, BASE + timedelta(days=2)),)
    candidate_ids = ("H21:TIME_1H",)
    run = run_adaptive_story_s5_grid(
        "EUR_USD", (_candle(0),), splits, candidate_ids=candidate_ids
    )
    tampered = copy.deepcopy(run)
    if tamper == "status":
        tampered["status"] = "FAILED_PARTIAL"
    elif tamper == "story_policy":
        tampered["story_catalog_policy"] = "INCOMPATIBLE_POLICY"
    elif tamper == "truth_policy":
        tampered["truth_policy"] = "INCOMPATIBLE_TRUTH"
    elif tamper == "catalog_digest":
        tampered["story_catalog_sha256"] = "0" * 64
    elif tamper == "evaluator_digest":
        tampered["truth_evaluator_sha256"] = "0" * 64
    elif tamper == "trial_metadata":
        tampered["all_trials"][0]["hypothesis_id"] = "H99"
    elif tamper == "daily_summary":
        tampered["all_trials"][0]["daily_aggregates_by_split"]["TRAIN"][0][
            "exact_net_r"
        ] = 1.0
    elif tamper == "candidate_scope":
        tampered["evaluated_candidate_ids"] = []
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(tamper)
    tampered["result_sha256"] = grid._canonical_sha(
        {key: value for key, value in tampered.items() if key != "result_sha256"}
    )

    with pytest.raises(ValueError):
        combine_adaptive_story_s5_grid_runs(
            (tampered,), splits, candidate_ids=candidate_ids
        )


def test_combiner_rejects_coherently_resealed_economics_inside_no_data_run() -> None:
    splits = (UtcSplit("TRAIN", BASE, BASE + timedelta(days=2)),)
    candidate_ids = ("H21:TIME_1H",)
    run = run_adaptive_story_s5_grid("EUR_USD", (), splits, candidate_ids=candidate_ids)
    metric = run["all_trials"][0]["by_split"]["TRAIN"]
    metric.update(
        {
            "filled_count": 1,
            "resolved_count": 1,
            "exact_net_r": 1.0,
            "gross_profit_r": 1.0,
            "gross_loss_r": 0.0,
        }
    )
    day = run["all_trials"][0]["daily_aggregates_by_split"]["TRAIN"][0]
    day.update(
        {
            "filled_count": 1,
            "resolved_count": 1,
            "exact_net_r": 1.0,
            "gross_profit_r": 1.0,
            "gross_loss_r": 0.0,
        }
    )
    run["result_sha256"] = grid._canonical_sha(
        {key: value for key, value in run.items() if key != "result_sha256"}
    )

    with pytest.raises(ValueError, match="NO_DATA"):
        combine_adaptive_story_s5_grid_runs((run,), splits, candidate_ids=candidate_ids)
