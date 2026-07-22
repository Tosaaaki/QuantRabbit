from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pytest

from quant_rabbit.dojo_anomaly_admission_controller import (
    FORMAL_G8_CURRENCIES,
    FORMAL_G8_PAIRS,
    build_policy,
)
from quant_rabbit.dojo_anomaly_admission_runtime import (
    DojoAnomalyAdmissionRuntimeError,
    _AnomalyAdmissionRuntime,
    _CompletedH1Tracker,
    build_anomaly_admission_runtime_seal,
    verify_anomaly_admission_runtime_seal,
)
from quant_rabbit.dojo_bot_catalog import AUTHORITY_INVARIANTS, validate_bot_config
from quant_rabbit.dojo_bot_trainer import PROPOSAL_CONTRACT, seal_candidate_proposal
from quant_rabbit.dojo_tuned_strategy_runtime import (
    build_tuned_strategy_runtime_seal,
)


ROOT = Path(__file__).resolve().parents[1]
START = 1_800_000_000 - (1_800_000_000 % 3_600)
BINDING = {
    "worker_id": "worker-a",
    "owner_id": "owner-a",
    "family_id": "burst",
    "config_sha256": "a" * 64,
}


def _band(half: float, quarter: float, hold: float) -> dict[str, float]:
    return {
        "reduce_to_half_at": half,
        "reduce_to_quarter_at": quarter,
        "hold_at": hold,
    }


def _policy() -> dict[str, Any]:
    return build_policy(
        policy_id="runtime-test-policy-v1",
        lookbacks={
            "momentum_bars": 2,
            "reversal_prior_bars": 2,
            "volatility_short_bars": 2,
            "volatility_long_bars": 3,
            "atr_bars": 3,
            "correlation_bars": 3,
        },
        bands={
            name: _band(10.0, 20.0, 30.0)
            for name in (
                "momentum_z",
                "reversal_shock_z",
                "volatility_ratio",
                "spread_atr_ratio",
                "currency_gross_exposure_fraction",
            )
        }
        | {"correlation_concentration": _band(2.0, 3.0, 4.0)},
        selected_pair_abs_correlation_hold_at=1.0,
    )


def _snapshot(epoch: int, phase: str, *, jump: float = 0.0) -> dict[str, Any]:
    phase_offset = {"O": 0.0, "H": 0.0003, "L": -0.0002, "C": 0.0001}[phase]
    quotes = []
    for index, pair in enumerate(FORMAL_G8_PAIRS):
        base = 1.0 + index * 0.01
        if pair.endswith("JPY"):
            base = 100.0 + index
        wave = 0.001 * math.sin((epoch - START) / 2_700 + index * 0.37)
        mid = base * (1 + wave + phase_offset + jump)
        spread = base * 0.00002
        quotes.append(
            {
                "pair": pair,
                "bid": mid - spread / 2,
                "ask": mid + spread / 2,
                "timestamp": f"{epoch}#{phase}",
            }
        )
    return {
        "snapshot_sha256": f"{epoch * 10 + 'OHLC'.index(phase):064x}"[-64:],
        "epoch": epoch,
        "phase": phase,
        "intrabar": "OHLC",
        "quotes": quotes,
        "account": {
            "balance_jpy": 200_000.0,
            "equity_jpy": 200_000.0,
            "margin_used_jpy": 0.0,
            "accrued_financing_jpy": 0.0,
        },
        "positions": [],
        "pending_orders": [],
    }


def _intent(intent_id: str, pair: str, units: float) -> dict[str, Any]:
    return {
        "intent_id": intent_id,
        "action": "MARKET",
        "parameters": {
            "pair": pair,
            "side": "LONG",
            "units": units,
            "entry_price": 1.0,
            "tp_price": 1.1,
            "sl_price": 0.9,
            "stress_cost_pips": 0.0,
            "hard_max_holding_seconds": 3_600,
            "valid_until_epoch": START + 99_999,
            "expected_net_edge_jpy": 0.0,
        },
        "reason_code": "TEST",
    }


class _FakeUpstream:
    def __init__(self, signal_epoch: int) -> None:
        self.signal_epoch = signal_epoch

    def propose(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        intents = []
        if snapshot["epoch"] == self.signal_epoch and snapshot["phase"] == "C":
            intents = [
                _intent("i-1", FORMAL_G8_PAIRS[0], 4_000.75),
                _intent("i-2", FORMAL_G8_PAIRS[1], 5_000.25),
            ]
        return [
            {
                **BINDING,
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "risk_reducing_intents": [],
                "new_risk_intents": intents,
            }
        ]

    def export_state(self) -> dict[str, bool]:
        return {"fake": True}


class _FakeFactory:
    def __init__(self, signal_epoch: int) -> None:
        self.signal_epoch = signal_epoch

    def __call__(self, *_args: object) -> _FakeUpstream:
        return _FakeUpstream(self.signal_epoch)


def _runtime(signal_epoch: int, *, slots: int = 1) -> _AnomalyAdmissionRuntime:
    policy = _policy()
    seal = {
        "runtime_binding_sha256": "b" * 64,
        "required_completed_h1_bars": 4,
        "policy": policy,
        "policy_sha256": policy["policy_sha256"],
        "arm": "BASE_BOT",
        "capacity_slots": slots,
    }
    return _AnomalyAdmissionRuntime(
        seal=seal,
        upstream_factory=_FakeFactory(signal_epoch),
        coordinate={"bar_seconds": 300},
        bindings=[BINDING],
        prior_state=None,
    )


def _feed(runtime: _AnomalyAdmissionRuntime, *, hours: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for offset in range(0, hours * 3_600, 300):
        epoch = START + offset
        for phase in "OHLC":
            result = runtime.propose(_snapshot(epoch, phase))
    return result


def test_runtime_recycles_held_seat_and_preserves_fractional_upstream_units() -> None:
    signal_epoch = START + 4 * 3_600 - 300
    runtime = _runtime(signal_epoch, slots=1)
    proposals = _feed(runtime, hours=4)

    assert len(proposals[0]["new_risk_intents"]) == 1
    selected = proposals[0]["new_risk_intents"][0]
    assert selected["intent_id"] == "i-1"
    assert selected["parameters"]["units"] == 4_000.75
    evidence = runtime.export_admission_evidence()
    assert evidence["runner_integration_complete"] is True
    assert evidence["independent_counterfactual_reexecution_complete"] is False
    assert evidence["official_evidence_eligible"] is False
    assert evidence["counts"] == {
        "decisions": 1,
        "upstream_candidates": 2,
        "selected": 1,
        "held": 1,
        "reduced": 0,
        "warmup_held": 0,
    }
    rows = evidence["counterfactual_tail"][0]["candidate_decisions"]
    assert [row["admission_decision"] for row in rows] == ["ENTER_OK", "HOLD"]


def test_runtime_holds_all_candidates_until_exact_h1_history_is_complete() -> None:
    signal_epoch = START + 3_600 - 300
    runtime = _runtime(signal_epoch)
    proposals = _feed(runtime, hours=1)

    assert proposals[0]["new_risk_intents"] == []
    evidence = runtime.export_admission_evidence()
    assert evidence["counts"]["warmup_held"] == 2
    assert evidence["counts"]["selected"] == 0
    assert evidence["counterfactual_tail"][0]["candidate_decisions"][0][
        "reason_codes"
    ] == ["H1_WARMUP_OR_DISCONTINUITY_HOLD"]


def test_runtime_carry_rejects_counter_and_hash_chain_forgery() -> None:
    signal_epoch = START + 4 * 3_600 - 300
    runtime = _runtime(signal_epoch)
    _feed(runtime, hours=4)
    state = runtime.export_state()
    policy = _policy()
    seal = {
        "runtime_binding_sha256": "b" * 64,
        "required_completed_h1_bars": 4,
        "policy": policy,
        "policy_sha256": policy["policy_sha256"],
        "arm": "BASE_BOT",
        "capacity_slots": 1,
    }

    forged_count = copy.deepcopy(state)
    forged_count["counts"]["selected"] += 1
    with pytest.raises(
        DojoAnomalyAdmissionRuntimeError, match="evidence counters"
    ):
        _AnomalyAdmissionRuntime(
            seal=seal,
            upstream_factory=_FakeFactory(signal_epoch + 300),
            coordinate={"bar_seconds": 300},
            bindings=[BINDING],
            prior_state=forged_count,
        )

    forged_chain = copy.deepcopy(state)
    forged_chain["counterfactual_tail"][0]["candidate_decisions"][0][
        "allocated_units"
    ] += 1
    with pytest.raises(
        DojoAnomalyAdmissionRuntimeError, match="counterfactual hash"
    ):
        _AnomalyAdmissionRuntime(
            seal=seal,
            upstream_factory=_FakeFactory(signal_epoch + 300),
            coordinate={"bar_seconds": 300},
            bindings=[BINDING],
            prior_state=forged_chain,
        )


def test_h1_gap_discards_pre_gap_history_instead_of_carrying_prices() -> None:
    tracker = _CompletedH1Tracker(cadence_seconds=300, required_bars=4, state=None)
    for offset in range(0, 2 * 3_600, 300):
        for phase in "OHLC":
            tracker.consume(_snapshot(START + offset, phase))
    assert len(tracker.completed[FORMAL_G8_PAIRS[0]]) == 2

    gap_epoch = START + 2 * 3_600 + 600
    for phase in "OHLC":
        tracker.consume(_snapshot(gap_epoch, phase))
    assert tracker.ready_panel() is None
    assert tracker.completed[FORMAL_G8_PAIRS[0]] == []


def _upstream_seal() -> dict[str, Any]:
    config = validate_bot_config(
        {
            "signal": "burst",
            "pairs": ["USD_JPY"],
            "tp_atr": 3.0,
            "sl_pips": 25.0,
            "ceiling_min": 60,
            "max_concurrent_per_pair": 1,
            "global_max_concurrent": 1,
            "per_pos_lev": 5.0,
            "atr_floor_pips": 0.5,
            "exit_policy": "FIXED",
            **dict(AUTHORITY_INVARIANTS),
        }
    )
    proposal = seal_candidate_proposal(
        {
            "contract": PROPOSAL_CONTRACT,
            "schema_version": 1,
            "candidate_id": "runtime-seal-burst",
            "family": "burst",
            "hypothesis": "Test the sealed anomaly wrapper.",
            "config": config,
            "risk_increase": False,
        }
    )
    return build_tuned_strategy_runtime_seal(
        ROOT,
        candidate_proposals=[proposal],
        generation_ordinal=1,
        generation_binding_sha256="c" * 64,
    )


def test_runtime_seal_binds_upstream_policy_arm_and_current_source_bytes() -> None:
    seal = build_anomaly_admission_runtime_seal(
        ROOT,
        upstream_runtime_seal=_upstream_seal(),
        policy=_policy(),
        arm="COMBINED_ANOMALY_ADMISSION",
        capacity_slots=4,
    )
    assert verify_anomaly_admission_runtime_seal(seal, repo_root=ROOT) == seal
    assert seal["evidence"]["economic_runner_integration_available"] is True
    assert seal["evidence"]["official_evidence_eligible"] is False
    assert seal["authority"]["live_permission"] is False
    assert seal["formal_pair_universe"] == list(FORMAL_G8_PAIRS)
    assert set(FORMAL_G8_CURRENCIES) == {
        currency for pair in FORMAL_G8_PAIRS for currency in pair.split("_")
    }

    changed = copy.deepcopy(seal)
    changed["capacity_slots"] = 5
    with pytest.raises(
        DojoAnomalyAdmissionRuntimeError,
        match="differs from its closed dependency",
    ):
        verify_anomaly_admission_runtime_seal(changed, repo_root=ROOT)
