from __future__ import annotations

import json

import pytest

from quant_rabbit.dojo_portfolio_replay_reducer import canonical_portfolio_sha256
from quant_rabbit.dojo_r13_ai_inventory_oos import (
    A_BOT_ONLY,
    B_INVENTORY_ONLY,
    C_FORECAST_INVENTORY,
    DojoR13AIInventoryError,
    _inventory_packet,
    deterministic_worker_response,
    simulate_partition,
    validate_worker_response,
)


def _frame(epoch: int, bid: float, ask: float) -> dict:
    return {
        "epoch": epoch,
        "phase": "C",
        "intrabar": "OHLC",
        "quote_watermark": epoch,
        "quotes": [{"pair": "USD_JPY", "bid": bid, "ask": ask}],
    }


def _trade(
    *,
    position_id: str = "p1",
    opened_epoch: int = 1000,
    closed_epoch: int = 2200,
) -> dict:
    return {
        "position_id": position_id,
        "family_id": "test_family",
        "pair": "USD_JPY",
        "side": "LONG",
        "units": 100.0,
        "entry_price": 100.01,
        "tp_price": 102.0,
        "sl_price": 99.0,
        "opened_epoch": opened_epoch,
        "opened_phase": "C",
        "hard_exit_epoch": closed_epoch + 3600,
        "open_frame_index": 0,
        "first_seen_frame_index": 0,
        "close_frame_index": 4,
        "closed_epoch": closed_epoch,
        "closed_phase": "C",
        "close_reason": "STOP_LOSS",
        "baseline_fill_price": 98.99,
        "baseline_exit_slippage_price": 0.0,
        "baseline_price_pnl_jpy": -102.0,
        "baseline_financing_jpy": 0.0,
        "baseline_net_pnl_jpy": -102.0,
    }


def _study() -> dict:
    return {
        "study_sha256": "a" * 64,
        "initial_capital_jpy_per_partition": 200000.0,
        "calibration_window": {"start_epoch": 1000, "end_epoch": 1600},
        "oos_window": {"start_epoch": 1600, "end_epoch": 2501},
    }


def _coordinate(trades: list[dict] | None = None) -> dict:
    trade_rows = trades if trades is not None else [_trade()]
    account_path = []
    for epoch in (1000, 1300, 1600, 1900, 2200, 2500):
        balance = 200000.0 + sum(
            float(trade["baseline_price_pnl_jpy"])
            for trade in trade_rows
            if int(trade["closed_epoch"]) <= epoch
        )
        account_path.append(
            {
                "epoch": epoch,
                "phase": "C",
                "balance_jpy": balance,
                "equity_jpy": balance,
                "margin_used_jpy": 0.0,
            }
        )
    return {
        "coordinate_id": "coord",
        "family_id": "test_family",
        "cost_scenario": "BASE",
        "prepared_coordinate_sha256": "b" * 64,
        "cost_policy": {
            "leverage": 25.0,
            "margin_closeout_fraction": 0.9,
            "financing_by_pair": [
                {
                    "pair": "USD_JPY",
                    "long_cost_jpy_per_unit_day": 0.0,
                    "short_cost_jpy_per_unit_day": 0.0,
                }
            ],
            "slippage_by_pair": [
                {
                    "pair": "USD_JPY",
                    "entry_slippage_price": 0.0,
                    "exit_slippage_price": 0.0,
                }
            ],
        },
        "baseline_account_path": account_path,
        "trades": trade_rows,
    }


def _packet(arm: str) -> dict:
    return _inventory_packet(
        study_sha256="a" * 64,
        coordinate=_coordinate(),
        arm=arm,
        cadence_id="FIXED_15M",
        policy_version="test-v1",
        prompt_version="prompt-v1",
        frame=_frame(1900, 99.5, 99.51),
        active_positions=[
            {
                **_trade(opened_epoch=1600),
                "remaining_units": 100.0,
                "mfe_jpy": 20.0,
                "mae_jpy": -51.0,
            }
        ],
        realized_pnl_jpy=0.0,
        peak_equity_jpy=200000.0,
        equity_jpy=199949.0,
        history={"USD_JPY": [100.0, 99.8, 99.5]},
        narrative_state=None,
        triggers=["LOSS_PROGRESS"],
        state_hash="c" * 64,
    )


def test_packet_is_causal_and_sealed() -> None:
    packet = _packet(C_FORECAST_INVENTORY)
    assert packet["packet_sha256"] == canonical_portfolio_sha256(
        {key: value for key, value in packet.items() if key != "packet_sha256"}
    )
    encoded = json.dumps(packet, sort_keys=True)
    for forbidden in (
        "future_quotes",
        "future_pnl",
        "terminal_profit",
        "append_wall_clock_at",
    ):
        assert forbidden not in encoded
    assert packet["future_quote_included"] is False
    assert packet["terminal_result_included"] is False
    assert packet["append_wall_clock_included"] is False


def test_inventory_only_rejects_forecast_and_tampered_packet() -> None:
    packet = _packet(B_INVENTORY_ONLY)
    response = deterministic_worker_response(packet, policy_id="test")
    response["forecast"] = {
        "direction": "DOWN",
        "confidence": 0.8,
        "horizon_min": 60,
        "invalidation": "Observed return changes sign.",
        "evidence_refs": ["observed_market"],
    }
    with pytest.raises(DojoR13AIInventoryError, match="must not emit forecast"):
        validate_worker_response(packet=packet, response=response)

    tampered = dict(packet)
    tampered["cutoff_epoch"] += 300
    raw = deterministic_worker_response(packet, policy_id="test")
    with pytest.raises(DojoR13AIInventoryError, match="causal sealed packet"):
        validate_worker_response(packet=tampered, response=raw)


def test_forecast_response_has_versioned_narrative_and_observed_evidence() -> None:
    packet = _packet(C_FORECAST_INVENTORY)
    response = deterministic_worker_response(packet, policy_id="test")
    sealed = validate_worker_response(packet=packet, response=response)
    assert sealed["narrative_state"]["version"] == 1
    assert sealed["forecast"]["direction"] in {"UP", "DOWN", "RANGE", "UNCERTAIN"}
    assert sealed["forecast"]["evidence_refs"]
    assert sealed["response_sha256"] == canonical_portfolio_sha256(response)


def test_oos_purges_left_boundary_trade() -> None:
    frames = [
        _frame(1600, 100.0, 100.01),
        _frame(1900, 99.7, 99.71),
        _frame(2200, 98.99, 99.0),
        _frame(2500, 99.0, 99.01),
    ]
    left_boundary = _trade(
        position_id="left",
        opened_epoch=1000,
        closed_epoch=1900,
    )
    included = _trade(
        position_id="included",
        opened_epoch=1600,
        closed_epoch=2200,
    )
    cell = simulate_partition(
        study=_study(),
        coordinate=_coordinate([left_boundary, included]),
        frames=frames,
        partition="OOS",
        arm=A_BOT_ONLY,
        cadence_id=None,
        policy_version="bot-only",
        prompt_version="none",
        worker=None,
    )
    assert cell["purged_boundary_trade_count"] == 1
    assert cell["metrics"]["scheduled_trade_count"] == 1


def test_inventory_overlay_can_close_observed_loss_without_future_input() -> None:
    frames = [
        _frame(1600, 100.0, 100.01),
        _frame(1900, 99.5, 99.51),
        _frame(2200, 98.99, 99.0),
        _frame(2500, 99.0, 99.01),
    ]
    trade = _trade(opened_epoch=1600, closed_epoch=2200)
    coordinate = _coordinate([trade])
    baseline = simulate_partition(
        study=_study(),
        coordinate=coordinate,
        frames=frames,
        partition="OOS",
        arm=A_BOT_ONLY,
        cadence_id=None,
        policy_version="bot-only",
        prompt_version="none",
        worker=None,
    )
    candidate = simulate_partition(
        study=_study(),
        coordinate=coordinate,
        frames=frames,
        partition="OOS",
        arm=B_INVENTORY_ONLY,
        cadence_id="FIXED_5M",
        policy_version="phase-a-test",
        prompt_version="prompt-test",
        worker=lambda packet: deterministic_worker_response(
            packet,
            policy_id="phase-a-test",
        ),
    )
    assert candidate["metrics"]["net_after_all_costs_jpy"] > baseline["metrics"][
        "net_after_all_costs_jpy"
    ]
    assert candidate["metrics"]["max_drawdown_fraction"] <= baseline["metrics"][
        "max_drawdown_fraction"
    ]
    assert all(
        row["cutoff_epoch"] == row["cutoff_epoch"]
        for row in candidate["intervention_audit"]
    )


def test_first_worker_packet_is_invariant_to_future_quote_changes() -> None:
    common_prefix = [
        _frame(1600, 100.0, 100.01),
        _frame(1900, 99.8, 99.81),
    ]
    future_a = [
        _frame(2200, 98.99, 99.0),
        _frame(2500, 99.0, 99.01),
    ]
    future_b = [
        _frame(2200, 105.0, 105.01),
        _frame(2500, 110.0, 110.01),
    ]
    first_packets: list[dict] = []
    for frames in (common_prefix + future_a, common_prefix + future_b):
        captured: list[dict] = []

        def worker(packet: dict) -> dict:
            captured.append(packet)
            return deterministic_worker_response(packet, policy_id="causal-test")

        simulate_partition(
            study=_study(),
            coordinate=_coordinate(
                [_trade(opened_epoch=1600, closed_epoch=2200)]
            ),
            frames=frames,
            partition="OOS",
            arm=C_FORECAST_INVENTORY,
            cadence_id="FIXED_5M",
            policy_version="causal-test",
            prompt_version="prompt-test",
            worker=worker,
            max_ai_calls=1,
        )
        first_packets.append(captured[0])

    assert first_packets[0] == first_packets[1]
    assert first_packets[0]["cutoff_epoch"] == 1600


def test_direction_restriction_vetoes_later_same_side_entry() -> None:
    first = _trade(
        position_id="first",
        opened_epoch=1600,
        closed_epoch=2200,
    )
    second = _trade(
        position_id="second",
        opened_epoch=1900,
        closed_epoch=2500,
    )

    def worker(packet: dict) -> dict:
        response = deterministic_worker_response(packet, policy_id="direction-test")
        response["action"]["direction_restriction"] = "NO_NEW_LONGS"
        return response

    cell = simulate_partition(
        study=_study(),
        coordinate=_coordinate([first, second]),
        frames=[
            _frame(1600, 100.0, 100.01),
            _frame(1900, 99.8, 99.81),
            _frame(2200, 98.99, 99.0),
            _frame(2500, 99.0, 99.01),
        ],
        partition="OOS",
        arm=C_FORECAST_INVENTORY,
        cadence_id="FIXED_5M",
        policy_version="direction-test",
        prompt_version="prompt-test",
        worker=worker,
        max_ai_calls=1,
    )
    assert cell["metrics"]["scheduled_trade_count"] == 2
    assert cell["metrics"]["skipped_trade_count"] == 1


def test_invalid_worker_response_uses_one_shot_hold_fallback() -> None:
    frames = [
        _frame(1600, 100.0, 100.01),
        _frame(1900, 99.8, 99.81),
        _frame(2200, 98.99, 99.0),
        _frame(2500, 99.0, 99.01),
    ]
    cell = simulate_partition(
        study=_study(),
        coordinate=_coordinate([_trade(opened_epoch=1600, closed_epoch=2200)]),
        frames=frames,
        partition="OOS",
        arm=C_FORECAST_INVENTORY,
        cadence_id="FIXED_5M",
        policy_version="actual-worker-test",
        prompt_version="prompt-test",
        worker=lambda _: {"invalid": True},
        max_ai_calls=1,
    )
    assert cell["metrics"]["ai_call_count"] == 1
    assert cell["metrics"]["ai_fallback_count"] == 1
    assert cell["intervention_audit"][0]["fallback"] is True
    assert cell["intervention_audit"][0]["action"]["type"] == "HOLD"
    assert [
        row["failure_class"] for row in cell["intervention_audit"]
    ] == ["DojoR13AIInventoryError"]
