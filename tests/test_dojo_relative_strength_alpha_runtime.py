from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from quant_rabbit.dojo_relative_strength_alpha_runtime import (
    ALPHA_DECISION_CONTRACT,
    DojoRelativeStrengthAlphaRuntimeError,
    FORMAL_G8_PAIRS,
    RelativeStrengthAlphaRuntime,
    build_relative_strength_alpha_runtime_seal,
    canonical_sha256,
    verify_relative_strength_alpha_runtime_seal,
)
from quant_rabbit.dojo_training_rooms import (
    INITIAL_ROOM_TAXONOMY,
    ROOM_TAXONOMY_V2,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASES = ("O", "H", "L", "C")


def _seal(*, lookback_h1_bars: int = 1) -> dict:
    return build_relative_strength_alpha_runtime_seal(
        REPO_ROOT,
        config={"lookback_h1_bars": lookback_h1_bars},
    )


def _base_prices() -> dict[str, float]:
    prices = {}
    for index, pair in enumerate(FORMAL_G8_PAIRS):
        _base, quote = pair.split("_")
        prices[pair] = (100.0 + index) if quote == "JPY" else (1.0 + index / 100.0)
    return prices


def _trend_prices(base_prices: dict[str, float]) -> dict[str, float]:
    # USD is uniquely strongest and JPY uniquely weakest.  Every cross gets a
    # consistent pair return from the two currency effects.
    effects = {
        "USD": 0.004,
        "EUR": 0.002,
        "GBP": 0.001,
        "CHF": 0.0005,
        "CAD": -0.0005,
        "AUD": -0.001,
        "NZD": -0.002,
        "JPY": -0.004,
    }
    return {
        pair: price
        * math.exp(effects[pair.split("_")[0]] - effects[pair.split("_")[1]])
        for pair, price in base_prices.items()
    }


def _snapshot(
    *,
    epoch: int,
    phase: str,
    mids: dict[str, float],
    missing_pair: str | None = None,
) -> dict:
    pairs = [pair for pair in FORMAL_G8_PAIRS if pair != missing_pair]
    spread = 0.00002
    quotes = [
        {
            "pair": pair,
            "bid": mids[pair] - spread / 2.0,
            "ask": mids[pair] + spread / 2.0,
            "timestamp": f"{epoch}#{phase}",
        }
        for pair in pairs
    ]
    identity = {
        "epoch": epoch,
        "phase": phase,
        "pairs": pairs,
        "mids": {pair: mids[pair] for pair in pairs},
    }
    return {
        "snapshot_sha256": canonical_sha256(identity),
        "epoch": epoch,
        "phase": phase,
        "intrabar": "OHLC",
        "expected_quote_pairs": list(FORMAL_G8_PAIRS),
        "quotes": quotes,
    }


def _consume_two_h1(
    runtime: RelativeStrengthAlphaRuntime,
    *,
    next_open_prices: dict[str, float] | None = None,
    missing_next_open_pair: str | None = None,
) -> tuple[list[dict], dict]:
    base = _base_prices()
    trend = _trend_prices(base)
    decisions = []
    for epoch in range(0, 7200, 300):
        mids = base if epoch < 3600 else trend
        for phase in PHASES:
            decisions.append(
                runtime.observe(_snapshot(epoch=epoch, phase=phase, mids=mids))
            )
    open_prices = next_open_prices or trend
    next_open = runtime.observe(
        _snapshot(
            epoch=7200,
            phase="O",
            mids=open_prices,
            missing_pair=missing_next_open_pair,
        )
    )
    return decisions, next_open


def test_seal_is_content_bound_and_alpha_only() -> None:
    seal = _seal(lookback_h1_bars=6)
    assert (
        verify_relative_strength_alpha_runtime_seal(seal, repo_root=REPO_ROOT) == seal
    )
    assert seal["dojo_room_id"] == "room-03"
    assert seal["strategy_family"] == "g8_relative_strength_alpha"
    assert seal["formal_pair_universe"] == list(FORMAL_G8_PAIRS)
    assert len(seal["formal_pair_universe"]) == 28
    assert seal["output_scope"] == "PAIR_SIDE_ALPHA_ONLY_NO_UNITS"
    assert seal["authority"]["portfolio_sizing_allowed"] is False
    assert seal["authority"]["anomaly_veto_allowed"] is False

    tampered = copy.deepcopy(seal)
    tampered["config"]["lookback_h1_bars"] = 7
    with pytest.raises(
        DojoRelativeStrengthAlphaRuntimeError,
        match="differs from current config or source bytes",
    ):
        verify_relative_strength_alpha_runtime_seal(tampered, repo_root=REPO_ROOT)

    with pytest.raises(
        DojoRelativeStrengthAlphaRuntimeError,
        match="content hash is invalid",
    ):
        RelativeStrengthAlphaRuntime(seal=tampered, cadence_seconds=300)


def test_completed_h1_alpha_emits_only_at_immediate_next_m5_open() -> None:
    runtime = RelativeStrengthAlphaRuntime(seal=_seal(), cadence_seconds=300)
    prior, decision = _consume_two_h1(runtime)

    assert all(row["status"] == "HOLD" for row in prior)
    assert prior[-1]["snapshot_epoch"] == 6900
    assert prior[-1]["snapshot_phase"] == "C"
    assert prior[-1]["reason_code"] == "WAIT_FOR_NEXT_M5_OPEN_HOLD"
    assert decision["contract"] == ALPHA_DECISION_CONTRACT
    assert decision["status"] == "ENTER"
    assert decision["pair"] == "USD_JPY"
    assert decision["side"] == "LONG"
    assert decision["signal_h1_close_epoch"] == 7200
    assert decision["execution_epoch"] == 7200
    assert decision["execution_timing"] == "NEXT_M5_OPEN_AFTER_COMPLETED_H1"
    assert decision["ranked_currencies"][0] == "USD"
    assert decision["ranked_currencies"][-1] == "JPY"
    assert decision["signal_h1_close_epoch"] <= decision["execution_epoch"]
    assert decision["output_scope"] == "PAIR_SIDE_ALPHA_ONLY_NO_UNITS"
    assert "units" not in decision
    assert "size_multiplier" not in decision
    assert "anomaly" not in decision


def test_next_m5_open_quote_cannot_change_completed_h1_rank() -> None:
    ordinary = RelativeStrengthAlphaRuntime(seal=_seal(), cadence_seconds=300)
    shocked = RelativeStrengthAlphaRuntime(seal=_seal(), cadence_seconds=300)
    _prior, ordinary_decision = _consume_two_h1(ordinary)

    # Change only the execution-open quotes after H1 has closed.  They may be
    # used later for execution pricing, but never for the already frozen rank.
    open_shock = {
        pair: price * (1.2 if "USD" in pair else 0.8)
        for pair, price in _trend_prices(_base_prices()).items()
    }
    _prior, shocked_decision = _consume_two_h1(shocked, next_open_prices=open_shock)

    for key in (
        "pair",
        "side",
        "signal_h1_close_epoch",
        "currency_scores",
        "ranked_currencies",
        "strength_dispersion",
    ):
        assert shocked_decision[key] == ordinary_decision[key]


def test_strongest_weakest_orientation_can_emit_short_without_pair_relabeling() -> None:
    runtime = RelativeStrengthAlphaRuntime(seal=_seal(), cadence_seconds=300)
    base = _base_prices()
    effects = {
        "JPY": 0.004,
        "NZD": 0.002,
        "AUD": 0.001,
        "CAD": 0.0005,
        "CHF": -0.0005,
        "GBP": -0.001,
        "EUR": -0.002,
        "USD": -0.004,
    }
    inverse = {
        pair: price
        * math.exp(effects[pair.split("_")[0]] - effects[pair.split("_")[1]])
        for pair, price in base.items()
    }
    for epoch in range(0, 7200, 300):
        mids = base if epoch < 3600 else inverse
        for phase in PHASES:
            runtime.observe(_snapshot(epoch=epoch, phase=phase, mids=mids))
    decision = runtime.observe(_snapshot(epoch=7200, phase="O", mids=inverse))

    assert decision["status"] == "ENTER"
    assert decision["pair"] == "USD_JPY"
    assert decision["side"] == "SHORT"
    assert decision["ranked_currencies"][0] == "JPY"
    assert decision["ranked_currencies"][-1] == "USD"


def test_missing_exact28_at_execution_open_is_hold_and_resets_warmup() -> None:
    runtime = RelativeStrengthAlphaRuntime(seal=_seal(), cadence_seconds=300)
    _prior, decision = _consume_two_h1(
        runtime, missing_next_open_pair=FORMAL_G8_PAIRS[-1]
    )

    assert decision["status"] == "HOLD"
    assert decision["reason_code"] == "EXACT28_QUOTE_BATCH_MISSING_HOLD"
    state = runtime.export_state()
    assert all(not rows for rows in state["completed_h1"].values())
    assert state["forming_h1"] is None
    assert state["enter_count"] == 0


def test_flat_completed_h1_matrix_holds_without_inventing_direction() -> None:
    runtime = RelativeStrengthAlphaRuntime(seal=_seal(), cadence_seconds=300)
    base = _base_prices()
    for epoch in range(0, 7200, 300):
        for phase in PHASES:
            runtime.observe(_snapshot(epoch=epoch, phase=phase, mids=base))
    decision = runtime.observe(_snapshot(epoch=7200, phase="O", mids=base))

    assert decision["status"] == "HOLD"
    assert decision["reason_code"] == "NO_CROSS_SECTIONAL_DISPERSION_HOLD"
    assert decision["pair"] is None
    assert decision["side"] is None


def test_carry_round_trip_preserves_causality_and_rejects_binding_drift() -> None:
    seal = _seal()
    runtime = RelativeStrengthAlphaRuntime(seal=seal, cadence_seconds=300)
    base = _base_prices()
    for epoch in range(0, 3600, 300):
        for phase in PHASES:
            runtime.observe(_snapshot(epoch=epoch, phase=phase, mids=base))
    carry = runtime.export_state()
    restored = RelativeStrengthAlphaRuntime(
        seal=seal, cadence_seconds=300, prior_state=carry
    )
    assert restored.export_state() == carry

    drifted = copy.deepcopy(carry)
    drifted["runtime_binding_sha256"] = "f" * 64
    with pytest.raises(
        DojoRelativeStrengthAlphaRuntimeError,
        match="carry binding drifted",
    ):
        RelativeStrengthAlphaRuntime(
            seal=seal, cadence_seconds=300, prior_state=drifted
        )

    future = copy.deepcopy(carry)
    future["completed_h1"][FORMAL_G8_PAIRS[0]][-1]["close_epoch"] += 3600
    with pytest.raises(
        DojoRelativeStrengthAlphaRuntimeError,
        match="histories are unsynchronized",
    ):
        RelativeStrengthAlphaRuntime(seal=seal, cadence_seconds=300, prior_state=future)


def test_noncausal_phase_and_nondivisible_cadence_fail_loud() -> None:
    seal = _seal()
    with pytest.raises(
        DojoRelativeStrengthAlphaRuntimeError,
        match="cadence must divide",
    ):
        RelativeStrengthAlphaRuntime(seal=seal, cadence_seconds=420)

    runtime = RelativeStrengthAlphaRuntime(seal=seal, cadence_seconds=300)
    runtime.observe(_snapshot(epoch=0, phase="O", mids=_base_prices()))
    with pytest.raises(
        DojoRelativeStrengthAlphaRuntimeError,
        match="phase sequence is non-causal",
    ):
        runtime.observe(_snapshot(epoch=0, phase="L", mids=_base_prices()))


def test_v1_taxonomy_remains_immutable_while_v2_room03_is_alpha_only() -> None:
    v1 = {room.room_id: room.strategy_family for room in INITIAL_ROOM_TAXONOMY}
    v2 = {room.room_id: room.strategy_family for room in ROOM_TAXONOMY_V2}
    assert v1["room-03"] == "g8_relative_strength_risk_budget"
    assert v2["room-03"] == "g8_relative_strength_alpha"
    assert v2["room-meta-01"] == "anomaly_admission_controller"
