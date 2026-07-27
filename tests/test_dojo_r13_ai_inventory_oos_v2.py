from __future__ import annotations

from copy import deepcopy

from quant_rabbit.dojo_portfolio_replay_reducer import canonical_portfolio_sha256
from quant_rabbit.dojo_r13_ai_inventory_oos import (
    B_INVENTORY_ONLY,
    C_FORECAST_INVENTORY,
    _inventory_packet,
    validate_worker_response,
)
from quant_rabbit.dojo_r13_ai_inventory_oos_v2 import (
    FACTORY_CONTRACT,
    WALK_FORWARD_CONTRACT,
    WORKER_RESPONSE_SCHEMA,
    _closed_tf_series,
    _combine_regime,
    _metric_projection,
    _tf_feature,
    build_factory_contract,
    deterministic_v2_worker_response,
    seal_walk_forward_contract,
)


def _frame(epoch: int, bid: float = 100.0, ask: float = 100.01) -> dict:
    return {
        "epoch": epoch,
        "phase": "C",
        "intrabar": "OHLC",
        "quote_watermark": epoch,
        "quotes": [{"pair": "USD_JPY", "bid": bid, "ask": ask}],
    }


def _coordinate() -> dict:
    return {
        "coordinate_id": "coord",
        "family_id": "mean_revert_24h",
        "cost_scenario": "BASE",
        "prepared_coordinate_sha256": "b" * 64,
        "cost_policy": {
            "leverage": 25.0,
            "margin_closeout_fraction": 0.9,
            "financing_by_pair": [],
            "slippage_by_pair": [],
        },
    }


def _trade() -> dict:
    return {
        "position_id": "p1",
        "family_id": "mean_revert_24h",
        "pair": "USD_JPY",
        "side": "LONG",
        "units": 100.0,
        "entry_price": 100.01,
        "tp_price": 101.0,
        "sl_price": 99.0,
        "opened_epoch": 3600,
        "opened_phase": "C",
        "hard_exit_epoch": 7200,
        "open_frame_index": 0,
        "first_seen_frame_index": 0,
        "close_frame_index": 4,
        "closed_epoch": 7200,
        "closed_phase": "C",
        "close_reason": "STOP_LOSS",
        "baseline_fill_price": 98.99,
        "baseline_exit_slippage_price": 0.0,
        "baseline_price_pnl_jpy": -102.0,
        "baseline_financing_jpy": 0.0,
        "baseline_net_pnl_jpy": -102.0,
        "remaining_units": 100.0,
        "mfe_jpy": 50.0,
        "mae_jpy": -51.0,
    }


def _packet(
    arm: str,
    *,
    bid: float = 99.5,
    prior_narrative_state: dict | None = None,
    positions: list[dict] | None = None,
) -> dict:
    return _inventory_packet(
        study_sha256="a" * 64,
        coordinate=_coordinate(),
        arm=arm,
        cadence_id="ADAPTIVE",
        policy_version="INVENTORY_PROTECTIVE_V2",
        prompt_version="prompt-v2",
        frame=_frame(3900, bid=bid, ask=bid + 0.01),
        active_positions=[_trade()] if positions is None else positions,
        realized_pnl_jpy=0.0,
        peak_equity_jpy=200000.0,
        equity_jpy=199949.0,
        history={"USD_JPY": [100.0, 99.8, bid]},
        narrative_state=prior_narrative_state,
        triggers=["LOSS_PROGRESS"],
        state_hash="c" * 64,
    )


def _tf(
    direction: str,
    *,
    timeframe: str,
    confidence: float = 0.8,
) -> dict:
    return {
        "direction": direction,
        "strength": 0.7,
        "vol_percentile": 0.6,
        "structure": "INNER_RANGE",
        "confidence": confidence,
        "age_seconds": 0,
        "closed_bar_count": 40,
        "spread_percentile": 0.5,
        "range_position": 0.6,
        "timeframe": timeframe,
    }


def _regime(
    *,
    d1: str = "UP",
    h4: str = "UP",
    h1: str = "DOWN",
    m5: str = "DOWN",
) -> dict:
    return _combine_regime(
        epoch=3900,
        pair="USD_JPY",
        features={
            "D1": _tf(d1, timeframe="D1"),
            "H4": _tf(h4, timeframe="H4"),
            "H1": _tf(h1, timeframe="H1"),
            "M5": _tf(m5, timeframe="M5"),
        },
    )


def test_higher_timeframe_feature_uses_only_closed_bucket() -> None:
    source = [
        (epoch, 100.0 + epoch / 100000.0, 0.01)
        for epoch in range(0, 3600, 300)
    ]
    ends, closes, spreads = _closed_tf_series(source, 3600)
    before_close = _tf_feature(
        decision_epoch=3599,
        ends=ends,
        closes=closes,
        spreads=spreads,
        timeframe="H1",
    )
    at_close = _tf_feature(
        decision_epoch=3600,
        ends=ends,
        closes=closes,
        spreads=spreads,
        timeframe="H1",
    )
    assert before_close["closed_bar_count"] == 0
    assert at_close["closed_bar_count"] == 1


def test_future_source_append_does_not_change_prior_regime_feature() -> None:
    source = [
        (epoch, 100.0 + epoch / 100000.0, 0.01)
        for epoch in range(0, 7200, 300)
    ]
    ends, closes, spreads = _closed_tf_series(source, 3600)
    original = _tf_feature(
        decision_epoch=7200,
        ends=ends,
        closes=closes,
        spreads=spreads,
        timeframe="H1",
    )
    future = source + [(7500, 500.0, 0.50), (7800, 1.0, 0.50)]
    future_ends, future_closes, future_spreads = _closed_tf_series(future, 3600)
    recomputed = _tf_feature(
        decision_epoch=7200,
        ends=future_ends,
        closes=future_closes,
        spreads=future_spreads,
        timeframe="H1",
    )
    assert recomputed == original


def test_regime_schema_records_timeframe_conflict_and_hash() -> None:
    regime = _regime()
    assert regime["macro"]["direction"] == "UP"
    assert regime["meso"]["direction"] == "DOWN"
    assert regime["tf_conflict"] is True
    assert regime["regime_state_sha256"] == canonical_portfolio_sha256(
        {
            key: value
            for key, value in regime.items()
            if key != "regime_state_sha256"
        }
    )


def test_inventory_only_v2_response_uses_risk_not_forecast() -> None:
    packet = _packet(B_INVENTORY_ONLY)
    response = deterministic_v2_worker_response(
        packet,
        policy_id="INVENTORY_PROTECTIVE_V2",
        regime_state=_regime(),
    )
    sealed = validate_worker_response(packet=packet, response=response)
    assert sealed["forecast"] is None
    assert sealed["action"]["type"] in {
        "HOLD",
        "PARTIAL_CLOSE",
        "CLOSE_RISKY",
        "PAUSE_NEW_ENTRIES",
    }


def test_forecast_confidence_is_bounded_and_cannot_increase_risk() -> None:
    packet = _packet(C_FORECAST_INVENTORY)
    response = deterministic_v2_worker_response(
        packet,
        policy_id="MTF_CONFLICT_GUARD_V2",
        regime_state=_regime(),
    )
    sealed = validate_worker_response(packet=packet, response=response)
    assert sealed["forecast"]["confidence"] <= 0.60
    assert sealed["forecast"]["horizon_min"] == 120
    assert sealed["action"]["type"] != "RESUME"


def test_paused_flat_state_resumes_on_next_registered_call() -> None:
    prior = deterministic_v2_worker_response(
        _packet(C_FORECAST_INVENTORY),
        policy_id="INVENTORY_PROTECTIVE_V2",
        regime_state=_regime(),
    )["narrative_state"]
    packet = _packet(
        C_FORECAST_INVENTORY,
        prior_narrative_state=prior,
        positions=[],
    )
    response = deterministic_v2_worker_response(
        packet,
        policy_id="INVENTORY_PROTECTIVE_V2",
        regime_state=None,
    )
    assert response["action"]["type"] == "RESUME"
    assert response["narrative_state"]["version"] == prior["version"] + 1


def test_ai_cost_is_included_in_v2_net() -> None:
    cell = {
        "initial_capital_jpy": 200000.0,
        "metrics": {
            "net_after_all_costs_jpy": 100.0,
            "ending_equity_jpy": 200100.0,
            "profit_factor": 1.2,
            "win_rate": 0.5,
            "expectancy_jpy": 10.0,
            "max_drawdown_fraction": 0.01,
            "max_margin_utilization_fraction": 0.10,
            "margin_call_count": 0,
            "ruin_event_count": 0,
            "tp_profit_retained_fraction": 0.5,
            "loss_avoided_jpy": 0.0,
            "missed_upside_jpy": 0.0,
            "turnover_jpy": 1000.0,
            "scheduled_trade_count": 10,
            "trade_count": 10,
            "skipped_trade_count": 0,
            "ai_decision_count": 2,
            "ai_call_count": 2,
            "ai_fallback_count": 0,
            "ai_estimated_input_tokens": 1000,
            "ai_estimated_output_tokens": 1000,
            "ai_notional_cost_usd": 1.0,
        },
    }
    projected = _metric_projection(cell)
    assert projected["ai_cost_jpy"] == 160.0
    assert projected["net_after_all_costs_including_ai_jpy"] == -60.0
    deterministic_screen = _metric_projection(cell, include_ai_cost=False)
    assert deterministic_screen["ai_cost_jpy"] == 0.0
    assert (
        deterministic_screen["net_after_all_costs_including_ai_jpy"]
        == 100.0
    )


def test_worker_envelope_schema_matches_fail_closed_validator_contract() -> None:
    assert WORKER_RESPONSE_SCHEMA["next_trigger"] == "string"
    assert WORKER_RESPONSE_SCHEMA["inventory_diagnosis"]["exact_keys"] == [
        "risk_level",
        "strategy_regime_fit",
        "inventory_story_mismatch",
        "tp_profit_retention_risk",
        "loss_giveback_risk",
    ]
    assert WORKER_RESPONSE_SCHEMA["authority_exact"]["order_authority"] == "NONE"


def test_walk_forward_and_factory_do_not_overclaim_january(tmp_path) -> None:
    oos = {
        "result_sha256": "d" * 64,
        "family_decisions": [
            {
                "family_id": "mean_revert_24h",
                "base_stress_both_positive_hard_gate": True,
                "mechanism_improvement_confirmed": False,
            }
        ],
    }
    walk = seal_walk_forward_contract(output_root=tmp_path, oos_result=oos)
    factory = build_factory_contract(output_root=tmp_path, oos_result=oos)
    assert walk["contract"] == WALK_FORWARD_CONTRACT
    assert walk["january_is_final_model_validation"] is False
    assert walk["minimum_non_overlapping_oos_blocks"] == 8
    assert factory["contract"] == FACTORY_CONTRACT
    assert factory["status"] == "NOT_STARTED_NO_JANUARY_BASE_STRESS_CHAMPION"
    assert factory["siblings_per_parent_max"] == 3
