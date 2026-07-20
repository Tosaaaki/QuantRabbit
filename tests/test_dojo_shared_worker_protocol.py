from __future__ import annotations

import inspect
from copy import deepcopy

import pytest

from quant_rabbit import dojo_shared_worker_protocol as protocol


SHA_A = "a" * 64
SHA_B = "b" * 64


def raw_snapshot() -> dict:
    return {
        "coordinate_id": "1704067200:O",
        "epoch": 1_704_067_200,
        "phase": "O",
        "intrabar": "OHLC",
        "quote_batch_sha256": SHA_A,
        "quote_watermark": 42,
        "expected_quote_pairs": ["USD_JPY", "EUR_USD"],
        "active_worker_bindings": [
            {
                "worker_id": "worker-b",
                "owner_id": "owner-b",
                "family_id": "family-b",
                "config_sha256": SHA_B,
            },
            {
                "worker_id": "worker-a",
                "owner_id": "owner-a",
                "family_id": "family-a",
                "config_sha256": SHA_B,
            },
        ],
        "account": {
            "balance_jpy": 200_000,
            "equity_jpy": 201_000,
            "margin_used_jpy": 50_000,
            "accrued_financing_jpy": 125,
        },
        "quotes": [
            {
                "pair": "EUR_USD",
                "bid": 1.0998,
                "ask": 1.1,
                "timestamp": "2024-01-01T00:00:00+00:00#O",
            },
            {
                "pair": "USD_JPY",
                "bid": 145.0,
                "ask": 145.02,
                "timestamp": "2024-01-01T00:00:00+00:00#O",
            },
        ],
        "positions": [
            {
                "position_id": "position-b",
                "worker_id": "worker-b",
                "owner_id": "owner-b",
                "family_id": "family-b",
                "pair": "EUR_USD",
                "side": "SHORT",
                "units": 1_000,
                "entry_price": 1.1,
                "tp_price": 1.09,
                "sl_price": 1.11,
                "opened_epoch": 1_704_067_100,
                "hard_exit_epoch": 1_704_070_800,
            },
            {
                "position_id": "position-a",
                "worker_id": "worker-a",
                "owner_id": "owner-a",
                "family_id": "family-a",
                "pair": "USD_JPY",
                "side": "LONG",
                "units": 2_000,
                "entry_price": 144.8,
                "tp_price": 145.5,
                "sl_price": 144.5,
                "opened_epoch": 1_704_067_100,
                "hard_exit_epoch": 1_704_070_800,
            },
        ],
        "pending_orders": [
            {
                "order_id": "order-a",
                "worker_id": "worker-a",
                "owner_id": "owner-a",
                "family_id": "family-a",
                "pair": "USD_JPY",
                "side": "LONG",
                "order_kind": "LIMIT",
                "units": 1_000,
                "trigger_price": 144.5,
                "tp_price": 145.2,
                "sl_price": 144.2,
                "created_epoch": 1_704_067_100,
                "valid_until_epoch": 1_704_067_800,
            }
        ],
    }


def raw_proposal(worker: str = "worker-a") -> dict:
    if worker == "worker-b":
        return {
            "worker_id": "worker-b",
            "owner_id": "owner-b",
            "family_id": "family-b",
            "config_sha256": SHA_B,
            "snapshot_sha256": "",
            "risk_reducing_intents": [
                {
                    "intent_id": "risk-b",
                    "action": "TIGHTEN_STOP",
                    "parameters": {"position_id": "position-b", "sl_price": 1.105},
                    "reason_code": "TRAIL_PROFIT",
                }
            ],
            "new_risk_intents": [],
        }
    return {
        "worker_id": "worker-a",
        "owner_id": "owner-a",
        "family_id": "family-a",
        "config_sha256": SHA_B,
        "snapshot_sha256": "",
        "risk_reducing_intents": [
            {
                "intent_id": "risk-z",
                "action": "CANCEL_ORDER",
                "parameters": {"order_id": "order-a"},
                "reason_code": "STALE_THESIS",
            },
            {
                "intent_id": "risk-a",
                "action": "TIGHTEN_STOP",
                "parameters": {"position_id": "position-a", "sl_price": 144.7},
                "reason_code": "LOCK_PROFIT",
            },
        ],
        "new_risk_intents": [
            {
                "intent_id": "new-z",
                "action": "LIMIT",
                "parameters": {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "units": 1_000,
                    "entry_price": 1.099,
                    "tp_price": 1.103,
                    "sl_price": 1.096,
                    "stress_cost_pips": 1.5,
                    "hard_max_holding_seconds": 3_600,
                    "valid_until_epoch": 1_704_067_800,
                    "expected_net_edge_jpy": 600,
                },
                "reason_code": "MEAN_REVERSION",
            },
            {
                "intent_id": "new-a",
                "action": "MARKET",
                "parameters": {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "units": 500,
                    "entry_price": 145.0,
                    "tp_price": 144.7,
                    "sl_price": 145.2,
                    "stress_cost_pips": 1.0,
                    "hard_max_holding_seconds": 1_800,
                    "valid_until_epoch": 1_704_067_200,
                    "expected_net_edge_jpy": 350,
                },
                "reason_code": "SPIKE_FADE",
            },
        ],
    }


def sealed_snapshot() -> dict:
    return protocol.seal_post_exit_snapshot(raw_snapshot())


def sealed_proposal(worker: str = "worker-a") -> dict:
    snapshot = sealed_snapshot()
    proposal = raw_proposal(worker)
    proposal["snapshot_sha256"] = snapshot["snapshot_sha256"]
    return protocol.seal_worker_proposal(snapshot, proposal)


def test_snapshot_seal_is_input_order_independent_and_worker_view_is_recursive_read_only() -> (
    None
):
    first = raw_snapshot()
    second = deepcopy(first)
    second["quotes"].reverse()
    second["positions"].reverse()
    second["expected_quote_pairs"].reverse()
    second["active_worker_bindings"].reverse()

    sealed_first = protocol.seal_post_exit_snapshot(first)
    sealed_second = protocol.seal_post_exit_snapshot(second)
    assert sealed_first == sealed_second
    assert sealed_first["snapshot_state"] == "POST_EXIT"
    assert sealed_first["read_only"] is True
    assert sealed_first["order_authority"] == "NONE"

    readonly = protocol.readonly_post_exit_snapshot(sealed_first)
    with pytest.raises(TypeError):
        readonly["epoch"] = 0
    with pytest.raises(TypeError):
        readonly["account"]["equity_jpy"] = 0
    assert isinstance(readonly["quotes"], tuple)


def test_proposal_separates_classes_sorts_intents_and_carries_no_authority() -> None:
    snapshot = sealed_snapshot()
    proposal = raw_proposal()
    proposal["snapshot_sha256"] = snapshot["snapshot_sha256"]

    sealed = protocol.seal_worker_proposal(snapshot, proposal)

    assert [row["intent_id"] for row in sealed["risk_reducing_intents"]] == [
        "risk-a",
        "risk-z",
    ]
    assert [row["intent_id"] for row in sealed["new_risk_intents"]] == [
        "new-a",
        "new-z",
    ]
    assert sealed["new_risk_intents"][0]["parameters"]["activation_policy"] == (
        "CURRENT_COORDINATE_AFTER_ALLOCATION"
    )
    assert sealed["new_risk_intents"][1]["parameters"]["activation_policy"] == (
        "NEXT_COORDINATE_OR_LATER"
    )
    assert sealed["risk_reduction_processed_before_new_risk"] is True
    assert sealed["proposal_only"] is True
    assert sealed["allocation_allowed"] is False
    assert sealed["execution_allowed"] is False
    assert sealed["order_authority"] == "NONE"
    assert sealed["live_permission"] is False
    assert sealed["broker_mutation_allowed"] is False
    assert sealed["worker_economic_claims_authoritative"] is False
    assert sealed["reducer_recomputes_entry_and_edge"] is True
    assert protocol.verify_worker_proposal(snapshot, sealed) == sealed


def test_batch_seal_is_independent_of_worker_and_intent_input_order() -> None:
    snapshot = sealed_snapshot()
    proposal_a = raw_proposal("worker-a")
    proposal_b = raw_proposal("worker-b")
    for proposal in (proposal_a, proposal_b):
        proposal["snapshot_sha256"] = snapshot["snapshot_sha256"]
    reversed_a = deepcopy(proposal_a)
    reversed_a["risk_reducing_intents"].reverse()
    reversed_a["new_risk_intents"].reverse()

    sealed_a = protocol.seal_worker_proposal(snapshot, proposal_a)
    sealed_a_reversed = protocol.seal_worker_proposal(snapshot, reversed_a)
    sealed_b = protocol.seal_worker_proposal(snapshot, proposal_b)
    assert sealed_a == sealed_a_reversed
    assert protocol.seal_worker_proposal_batch(snapshot, [sealed_b, sealed_a]) == (
        protocol.seal_worker_proposal_batch(snapshot, [sealed_a_reversed, sealed_b])
    )


@pytest.mark.parametrize("injected_key", ["broker", "execute", "allocation"])
def test_strict_schema_rejects_authority_injection(injected_key: str) -> None:
    snapshot = sealed_snapshot()
    proposal = raw_proposal()
    proposal["snapshot_sha256"] = snapshot["snapshot_sha256"]
    proposal[injected_key] = object()

    with pytest.raises(protocol.ProtocolViolation, match="schema mismatch"):
        protocol.seal_worker_proposal(snapshot, proposal)


def test_public_entry_points_have_no_broker_argument_or_execution_primitive() -> None:
    for name in protocol.__all__:
        if inspect.isfunction(member := getattr(protocol, name)):
            assert "broker" not in inspect.signature(member).parameters
    assert not any(
        name.startswith(("execute", "allocate", "place_order"))
        for name in protocol.__all__
    )


def test_risk_reduction_is_owned_and_actually_reduces_risk() -> None:
    snapshot = sealed_snapshot()
    wrong_owner = raw_proposal()
    wrong_owner["snapshot_sha256"] = snapshot["snapshot_sha256"]
    wrong_owner["risk_reducing_intents"][0]["parameters"]["order_id"] = "unknown-order"
    with pytest.raises(protocol.ProtocolViolation, match="unknown pending order"):
        protocol.seal_worker_proposal(snapshot, wrong_owner)

    loose_stop = raw_proposal()
    loose_stop["snapshot_sha256"] = snapshot["snapshot_sha256"]
    loose_stop["risk_reducing_intents"] = [
        {
            "intent_id": "loosen",
            "action": "TIGHTEN_STOP",
            "parameters": {"position_id": "position-a", "sl_price": 144.4},
            "reason_code": "INVALID",
        }
    ]
    with pytest.raises(protocol.ProtocolViolation, match="strictly tighten"):
        protocol.seal_worker_proposal(snapshot, loose_stop)


def test_intent_classes_cannot_be_mixed_or_reuse_an_id() -> None:
    snapshot = sealed_snapshot()
    mixed = raw_proposal()
    mixed["snapshot_sha256"] = snapshot["snapshot_sha256"]
    mixed["risk_reducing_intents"][0]["action"] = "MARKET"
    with pytest.raises(protocol.ProtocolViolation, match="must be one of"):
        protocol.seal_worker_proposal(snapshot, mixed)

    duplicate = raw_proposal()
    duplicate["snapshot_sha256"] = snapshot["snapshot_sha256"]
    duplicate["new_risk_intents"][0]["intent_id"] = "risk-a"
    with pytest.raises(
        protocol.ProtocolViolation, match="reused across intent classes"
    ):
        protocol.seal_worker_proposal(snapshot, duplicate)


def test_new_risk_requires_valid_quote_geometry_stop_and_activation_window() -> None:
    snapshot = sealed_snapshot()
    proposal = raw_proposal()
    proposal["snapshot_sha256"] = snapshot["snapshot_sha256"]
    proposal["new_risk_intents"][0]["parameters"]["sl_price"] = 1.2
    with pytest.raises(protocol.ProtocolViolation, match="invalid LONG exit geometry"):
        protocol.seal_worker_proposal(snapshot, proposal)

    proposal = raw_proposal()
    proposal["snapshot_sha256"] = snapshot["snapshot_sha256"]
    proposal["new_risk_intents"][0]["parameters"]["valid_until_epoch"] = snapshot[
        "epoch"
    ]
    with pytest.raises(protocol.ProtocolViolation, match="expires before"):
        protocol.seal_worker_proposal(snapshot, proposal)


def test_tampering_and_nonfinite_or_boolean_numbers_are_rejected() -> None:
    snapshot = sealed_snapshot()
    tampered = sealed_proposal()
    tampered["new_risk_intents"][0]["parameters"]["units"] = 999
    with pytest.raises(protocol.ProtocolViolation, match="proposal_sha256"):
        protocol.verify_worker_proposal(snapshot, tampered)

    nonfinite = raw_snapshot()
    nonfinite["account"]["equity_jpy"] = float("nan")
    with pytest.raises(protocol.ProtocolViolation, match="finite"):
        protocol.seal_post_exit_snapshot(nonfinite)

    boolean_number = raw_snapshot()
    boolean_number["quote_watermark"] = True
    with pytest.raises(protocol.ProtocolViolation, match="integer"):
        protocol.seal_post_exit_snapshot(boolean_number)


def test_snapshot_hash_binding_and_exact_phase_are_enforced() -> None:
    snapshot = sealed_snapshot()
    proposal = raw_proposal()
    proposal["snapshot_sha256"] = SHA_B
    with pytest.raises(protocol.ProtocolViolation, match="does not match"):
        protocol.seal_worker_proposal(snapshot, proposal)

    wrong_phase = raw_snapshot()
    wrong_phase["quotes"][0]["timestamp"] = "2024-01-01T00:00:00+00:00#C"
    with pytest.raises(protocol.ProtocolViolation, match="must end with #O"):
        protocol.seal_post_exit_snapshot(wrong_phase)


def test_batch_rejects_duplicate_workers_and_tampering() -> None:
    snapshot = sealed_snapshot()
    proposal = sealed_proposal()
    with pytest.raises(protocol.ProtocolViolation, match="duplicate worker_id"):
        protocol.seal_worker_proposal_batch(snapshot, [proposal, proposal])

    proposal_b = sealed_proposal("worker-b")
    batch = protocol.seal_worker_proposal_batch(snapshot, [proposal, proposal_b])
    assert protocol.verify_worker_proposal_batch(snapshot, batch) == batch
    batch["proposal_count"] = 7
    with pytest.raises(protocol.ProtocolViolation, match="batch_sha256"):
        protocol.verify_worker_proposal_batch(snapshot, batch)


def test_snapshot_requires_exact_expected_quote_set() -> None:
    missing = raw_snapshot()
    missing["quotes"] = missing["quotes"][:1]
    with pytest.raises(protocol.ProtocolViolation, match="quote set is incomplete"):
        protocol.seal_post_exit_snapshot(missing)

    extra = raw_snapshot()
    extra["expected_quote_pairs"] = ["USD_JPY"]
    with pytest.raises(protocol.ProtocolViolation, match="quote set is incomplete"):
        protocol.seal_post_exit_snapshot(extra)


def test_batch_requires_one_proposal_from_every_bound_worker_even_for_hold() -> None:
    snapshot = sealed_snapshot()
    proposal_a = sealed_proposal("worker-a")
    with pytest.raises(protocol.ProtocolViolation, match="every active worker"):
        protocol.seal_worker_proposal_batch(snapshot, [proposal_a])

    hold_b = raw_proposal("worker-b")
    hold_b["snapshot_sha256"] = snapshot["snapshot_sha256"]
    hold_b["risk_reducing_intents"] = []
    hold_b["new_risk_intents"] = []
    sealed_hold_b = protocol.seal_worker_proposal(snapshot, hold_b)
    batch = protocol.seal_worker_proposal_batch(snapshot, [sealed_hold_b, proposal_a])
    assert batch["proposal_count"] == 2
    assert batch["proposals"][1]["intent_counts"] == {"new_risk": 0, "risk_reducing": 0}


@pytest.mark.parametrize(
    ("field", "value"),
    [("owner_id", "wrong-owner"), ("config_sha256", SHA_A)],
)
def test_proposal_identity_and_config_must_match_snapshot_binding(
    field: str, value: str
) -> None:
    snapshot = sealed_snapshot()
    proposal = raw_proposal("worker-a")
    proposal["snapshot_sha256"] = snapshot["snapshot_sha256"]
    proposal[field] = value
    with pytest.raises(protocol.ProtocolViolation, match="identity/config"):
        protocol.seal_worker_proposal(snapshot, proposal)


def test_extra_unbound_worker_is_rejected_before_batch_sealing() -> None:
    snapshot = sealed_snapshot()
    proposal = raw_proposal("worker-a")
    proposal["snapshot_sha256"] = snapshot["snapshot_sha256"]
    proposal["worker_id"] = "worker-extra"
    proposal["risk_reducing_intents"] = []
    with pytest.raises(protocol.ProtocolViolation, match="not active"):
        protocol.seal_worker_proposal(snapshot, proposal)
