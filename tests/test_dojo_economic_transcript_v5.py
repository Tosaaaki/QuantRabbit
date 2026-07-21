from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from quant_rabbit.dojo_economic_transcript_v4 import (
    build_account_delta,
    build_shared_source_segment,
    publish_immutable_artifact,
)
from quant_rabbit.dojo_economic_transcript_v5 import (
    DojoEconomicTranscriptV5Error,
    build_sparse_account_delta,
    verify_v5_account_files,
    verify_v5_coordinate_subset,
)
from quant_rabbit.dojo_portfolio_replay_reducer import (
    MONTH_END_MTM_WITH_STATE_HANDOFF,
    PortfolioReplaySession,
    canonical_portfolio_sha256,
    seal_portfolio_policy,
)
from quant_rabbit.dojo_shared_worker_protocol import (
    seal_worker_proposal,
    seal_worker_proposal_batch,
)


WORKERS = [
    {
        "worker_id": "worker-a",
        "owner_id": "owner-a",
        "family_id": "family-a",
        "config_sha256": "a" * 64,
    },
    {
        "worker_id": "worker-b",
        "owner_id": "owner-b",
        "family_id": "family-b",
        "config_sha256": "b" * 64,
    },
]


def _policy() -> dict[str, Any]:
    return seal_portfolio_policy(
        {
            "policy_id": "sparse-account-delta-v5-test",
            "expected_quote_pairs": ["USD_JPY"],
            "tradable_pairs": ["USD_JPY"],
            "active_worker_bindings": copy.deepcopy(WORKERS),
            "leverage": 20,
            "margin_closeout_fraction": 0.9,
            "max_margin_utilization_fraction": 0.8,
            "max_portfolio_stop_risk_fraction": 0.8,
            "max_open_and_pending_total": 16,
            "max_open_and_pending_per_pair": 16,
            "max_open_and_pending_per_family": 16,
            "max_currency_gross_notional_fraction": 100.0,
            "max_cluster_gross_notional_fraction": 100.0,
            "max_lock_seconds": 86_400,
            "slippage_by_pair": [
                {
                    "pair": "USD_JPY",
                    "entry_slippage_price": 0.01,
                    "exit_slippage_price": 0.02,
                }
            ],
            "financing_by_pair": [
                {
                    "pair": "USD_JPY",
                    "long_cost_jpy_per_unit_day": 0.0,
                    "short_cost_jpy_per_unit_day": 0.0,
                }
            ],
            "conversion_routes": [],
            "correlation_bindings": [],
        }
    )


def _source_inputs(count: int) -> list[dict[str, Any]]:
    epoch = 1_704_067_200
    phases = (("O", 0.0), ("H", 0.02), ("L", -0.02), ("C", 0.01))
    result: list[dict[str, Any]] = []
    for index in range(count):
        phase, offset = phases[index % len(phases)]
        batch_epoch = epoch + 60 * (index // len(phases))
        bid = 145.0 + offset + (index // len(phases)) * 0.0001
        timestamp = datetime.fromtimestamp(batch_epoch, timezone.utc).isoformat()
        result.append(
            {
                "epoch": batch_epoch,
                "phase": phase,
                "intrabar": "OHLC",
                "quote_watermark": index + 1,
                "quotes": [
                    {
                        "pair": "USD_JPY",
                        "bid": bid,
                        "ask": bid + 0.02,
                        "timestamp": f"{timestamp}#{phase}",
                    }
                ],
            }
        )
    return result


def _hold_proposal(snapshot: Mapping[str, Any], binding: Mapping[str, Any]) -> dict:
    return seal_worker_proposal(
        snapshot,
        {
            **binding,
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "risk_reducing_intents": [],
            "new_risk_intents": [],
        },
    )


def _market_payload(snapshot: Mapping[str, Any], *, ordinal: int) -> dict[str, Any]:
    ask = snapshot["quotes"][0]["ask"]
    return {
        **WORKERS[0],
        "risk_reducing_intents": [],
        "new_risk_intents": [
            {
                "intent_id": f"market-{ordinal}",
                "action": "MARKET",
                "parameters": {
                    "pair": "USD_JPY",
                    "side": "LONG",
                    "units": 100,
                    "entry_price": ask,
                    "tp_price": ask + 1.0,
                    "sl_price": ask - 1.0,
                    "stress_cost_pips": 1.0,
                    "hard_max_holding_seconds": 3_600,
                    "valid_until_epoch": snapshot["epoch"],
                    "expected_net_edge_jpy": 100.0,
                },
                "reason_code": "SPARSE_V5_TEST",
            }
        ],
    }


def _produce_account(
    *,
    source: Mapping[str, Any],
    coordinate_id: str,
    policy: Mapping[str, Any],
    event_ordinals: Sequence[int] = (),
) -> dict[str, Any]:
    selected = set(event_ordinals)
    session = PortfolioReplaySession(policy=policy, initial_balance_jpy=200_000.0)
    start = session.export_checkpoint()
    events: list[dict[str, Any]] = []
    v4_rows: list[dict[str, Any]] = []
    for ordinal, source_batch in enumerate(source["batches"]):
        snapshot = session.prepare_coordinate(
            coordinate_id=coordinate_id,
            epoch=source_batch["epoch"],
            phase=source_batch["phase"],
            intrabar=source_batch["intrabar"],
            quote_watermark=source_batch["quote_watermark"],
            quotes=source_batch["quotes"],
            quote_batch_sha256_value=source_batch["quote_batch_sha256"],
        )
        non_hold: list[dict[str, Any]] = []
        proposals: list[dict[str, Any]] = []
        if ordinal in selected:
            payload = _market_payload(snapshot, ordinal=ordinal)
            sealed = seal_worker_proposal(
                snapshot,
                {**payload, "snapshot_sha256": snapshot["snapshot_sha256"]},
            )
            non_hold.append(sealed)
            proposals.append(sealed)
            events.append(
                {
                    "source_ordinal": ordinal,
                    "non_hold_proposal_payloads": [payload],
                }
            )
        supplied = {proposal["worker_id"] for proposal in proposals}
        proposals.extend(
            _hold_proposal(snapshot, binding)
            for binding in WORKERS
            if binding["worker_id"] not in supplied
        )
        receipt = session.consume_proposal_batch(
            seal_worker_proposal_batch(snapshot, proposals)
        )
        v4_rows.append(
            {
                "source_ordinal": ordinal,
                "source_quote_batch_sha256": source_batch["quote_batch_sha256"],
                "source_batch_chain_sha256": source_batch["source_batch_chain_sha256"],
                "post_exit_snapshot_sha256": snapshot["snapshot_sha256"],
                "non_hold_proposals": non_hold,
                "allocation_receipt": receipt,
            }
        )
    terminal = session.export_checkpoint()
    result = session.finalize(terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF)
    return {
        "start_checkpoint": start,
        "terminal_checkpoint": terminal,
        "events": events,
        "v4_rows": v4_rows,
        "result": result,
    }


def _build_v5(
    *,
    source: Mapping[str, Any],
    policy: Mapping[str, Any],
    coordinate_id: str,
    transcript_id: str,
    event_ordinals: Sequence[int] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    produced = _produce_account(
        source=source,
        coordinate_id=coordinate_id,
        policy=policy,
        event_ordinals=event_ordinals,
    )
    delta = build_sparse_account_delta(
        transcript_id=transcript_id,
        coordinate_id=coordinate_id,
        shared_source_segment=source,
        portfolio_policy=policy,
        start_checkpoint=produced["start_checkpoint"],
        terminal_checkpoint=produced["terminal_checkpoint"],
        terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF,
        producer_portfolio_result=produced["result"],
        events=produced["events"],
    )
    return produced, delta


def _reseal_delta(delta: dict[str, Any]) -> None:
    delta["events_sha256"] = canonical_portfolio_sha256(delta["events"])
    body = {key: value for key, value in delta.items() if key != "account_delta_sha256"}
    delta["account_delta_sha256"] = canonical_portfolio_sha256(body)


def test_hold_only_delta_is_near_constant_and_source_is_shared(tmp_path: Path) -> None:
    policy = _policy()
    source_small = build_shared_source_segment(
        job_sha256="1" * 64,
        source_slice_receipt_sha256="3" * 64,
        batches=_source_inputs(4),
    )
    source_large = build_shared_source_segment(
        job_sha256="1" * 64,
        source_slice_receipt_sha256="3" * 64,
        batches=_source_inputs(256),
    )
    produced_small, delta_small_a = _build_v5(
        source=source_small,
        policy=policy,
        coordinate_id="coordinate-a",
        transcript_id="hold-small-a",
    )
    _, delta_small_b = _build_v5(
        source=source_small,
        policy=policy,
        coordinate_id="coordinate-b",
        transcript_id="hold-small-b",
    )
    _, delta_large = _build_v5(
        source=source_large,
        policy=policy,
        coordinate_id="coordinate-a",
        transcript_id="hold-large-a",
    )
    v4 = build_account_delta(
        transcript_id="hold-small-v4-a",
        coordinate_id="coordinate-a",
        shared_source_segment=source_small,
        portfolio_policy=policy,
        start_checkpoint=produced_small["start_checkpoint"],
        terminal_checkpoint=produced_small["terminal_checkpoint"],
        start_worker_state={"mode": "hold"},
        terminal_worker_state={"mode": "hold"},
        terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF,
        producer_portfolio_result=produced_small["result"],
        delta_rows=produced_small["v4_rows"],
    )
    assert (
        delta_small_a["producer_portfolio_result_sha256"]
        == v4["producer_portfolio_result_sha256"]
    )
    assert delta_small_a["events"] == []
    assert delta_large["events"] == []

    source_small_path = tmp_path / "source-small.json"
    source_large_path = tmp_path / "source-large.json"
    delta_small_a_path = tmp_path / "delta-small-a.json"
    delta_small_b_path = tmp_path / "delta-small-b.json"
    delta_large_path = tmp_path / "delta-large.json"
    for path, value in (
        (source_small_path, source_small),
        (source_large_path, source_large),
        (delta_small_a_path, delta_small_a),
        (delta_small_b_path, delta_small_b),
        (delta_large_path, delta_large),
    ):
        publish_immutable_artifact(path, value)

    small_bytes = delta_small_a_path.stat().st_size
    large_bytes = delta_large_path.stat().st_size
    assert abs(large_bytes - small_bytes) <= 128
    assert source_large_path.stat().st_size > source_small_path.stat().st_size * 40

    one = verify_v5_coordinate_subset(
        source_path=source_small_path,
        account_delta_paths=[delta_small_a_path],
        caller_declared_coordinate_ids=["coordinate-a"],
    )
    two = verify_v5_coordinate_subset(
        source_path=source_small_path,
        account_delta_paths=[delta_small_a_path, delta_small_b_path],
        caller_declared_coordinate_ids=["coordinate-a", "coordinate-b"],
    )
    assert one["source_file_bytes"] == two["source_file_bytes"]
    assert two["source_bytes_scale_with_coordinate_count"] is False
    assert two["reexecuted_coordinate_count"] == 2
    assert two["status"] == "SUBSET_REEXECUTED_UNANCHORED"
    assert two["fixed_coordinate_denominator_proven"] is False


def test_non_hold_ordinals_only_and_implicit_workers_reexecute(tmp_path: Path) -> None:
    policy = _policy()
    source = build_shared_source_segment(
        job_sha256="1" * 64,
        source_slice_receipt_sha256="3" * 64,
        batches=_source_inputs(12),
    )
    produced, delta = _build_v5(
        source=source,
        policy=policy,
        coordinate_id="coordinate-events",
        transcript_id="sparse-events",
        event_ordinals=[0, 4],
    )
    assert [event["source_ordinal"] for event in delta["events"]] == [0, 4]
    assert delta["event_count"] == 2
    assert len(delta["events"]) < source["expected_source_batch_count"]
    event_json = json.dumps(delta["events"], sort_keys=True)
    for forbidden in (
        "snapshot_sha256",
        "allocation_receipt",
        "source_batch_chain_sha256",
        "quote_batch_sha256",
        '"quotes"',
    ):
        assert forbidden not in event_json
    assert all(
        [proposal["worker_id"] for proposal in event["non_hold_proposal_payloads"]]
        == ["worker-a"]
        for event in delta["events"]
    )

    source_path = tmp_path / "source.json"
    delta_path = tmp_path / "delta.json"
    publish_immutable_artifact(source_path, source)
    publish_immutable_artifact(delta_path, delta)
    attestation = verify_v5_account_files(source_path, delta_path)
    assert attestation["portfolio_result_sha256"] == produced["result"]["result_sha256"]
    assert attestation["sparse_event_count"] == 2
    assert attestation["implicit_hold_reexecution_passed"] is True
    assert attestation["official_evidence_eligible"] is False
    assert attestation["order_authority"] == "NONE"


def test_sparse_tamper_reorder_fork_truncation_and_denominator_fail_closed(
    tmp_path: Path,
) -> None:
    policy = _policy()
    source = build_shared_source_segment(
        job_sha256="1" * 64,
        source_slice_receipt_sha256="3" * 64,
        batches=_source_inputs(12),
    )
    produced, delta = _build_v5(
        source=source,
        policy=policy,
        coordinate_id="coordinate-a",
        transcript_id="sparse-a",
        event_ordinals=[0, 4],
    )
    source_path = tmp_path / "source.json"
    delta_path = tmp_path / "delta.json"
    publish_immutable_artifact(source_path, source)
    publish_immutable_artifact(delta_path, delta)

    tampered = copy.deepcopy(delta)
    tampered["events"][0]["source_ordinal"] = 1
    event_body = {
        key: value
        for key, value in tampered["events"][0].items()
        if key != "event_sha256"
    }
    tampered["events"][0]["event_sha256"] = canonical_portfolio_sha256(event_body)
    _reseal_delta(tampered)
    tampered_path = tmp_path / "delta-tampered.json"
    publish_immutable_artifact(tampered_path, tampered)
    with pytest.raises(DojoEconomicTranscriptV5Error, match="invalid|replay"):
        verify_v5_account_files(source_path, tampered_path)

    reordered = copy.deepcopy(delta)
    reordered["events"] = list(reversed(reordered["events"]))
    _reseal_delta(reordered)
    reordered_path = tmp_path / "delta-reordered.json"
    publish_immutable_artifact(reordered_path, reordered)
    with pytest.raises(DojoEconomicTranscriptV5Error, match="strictly increasing"):
        verify_v5_account_files(source_path, reordered_path)

    fork = build_sparse_account_delta(
        transcript_id="sparse-a-valid-fork",
        coordinate_id="coordinate-a",
        shared_source_segment=source,
        portfolio_policy=policy,
        start_checkpoint=produced["start_checkpoint"],
        terminal_checkpoint=produced["terminal_checkpoint"],
        terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF,
        producer_portfolio_result=produced["result"],
        events=produced["events"],
    )
    fork_path = tmp_path / "delta-fork.json"
    publish_immutable_artifact(fork_path, fork)
    with pytest.raises(DojoEconomicTranscriptV5Error, match="duplicate|forked"):
        verify_v5_coordinate_subset(
            source_path=source_path,
            account_delta_paths=[delta_path, fork_path],
            caller_declared_coordinate_ids=["coordinate-a", "coordinate-b"],
        )

    with pytest.raises(DojoEconomicTranscriptV5Error, match="declared subset"):
        verify_v5_coordinate_subset(
            source_path=source_path,
            account_delta_paths=[delta_path],
            caller_declared_coordinate_ids=["coordinate-a", "coordinate-b"],
        )

    truncated_path = tmp_path / "delta-truncated.json"
    truncated_path.write_bytes(delta_path.read_bytes()[:-1])
    with pytest.raises(DojoEconomicTranscriptV5Error, match="truncated"):
        verify_v5_account_files(source_path, truncated_path)


def test_caller_subset_cannot_claim_complete_two_coordinate_plan(
    tmp_path: Path,
) -> None:
    policy = _policy()
    source = build_shared_source_segment(
        job_sha256="1" * 64,
        source_slice_receipt_sha256="3" * 64,
        batches=_source_inputs(4),
    )
    _, delta_a = _build_v5(
        source=source,
        policy=policy,
        coordinate_id="coordinate-a",
        transcript_id="two-coordinate-plan-a",
    )
    _, delta_b = _build_v5(
        source=source,
        policy=policy,
        coordinate_id="coordinate-b",
        transcript_id="two-coordinate-plan-b",
    )
    source_path = tmp_path / "source.json"
    delta_a_path = tmp_path / "delta-a.json"
    delta_b_path = tmp_path / "delta-b.json"
    publish_immutable_artifact(source_path, source)
    publish_immutable_artifact(delta_a_path, delta_a)
    publish_immutable_artifact(delta_b_path, delta_b)

    # Both planned coordinates exist, but a caller can declare only one.  Until a
    # sealed job/plan/handoff supplies the denominator, this is only a subset proof.
    subset = verify_v5_coordinate_subset(
        source_path=source_path,
        account_delta_paths=[delta_a_path],
        caller_declared_coordinate_ids=["coordinate-a"],
    )

    assert delta_b_path.exists()
    assert subset["status"] == "SUBSET_REEXECUTED_UNANCHORED"
    assert subset["caller_declared_coordinate_ids"] == ["coordinate-a"]
    assert subset["reexecuted_coordinate_count"] == 1
    assert subset["complete_job_coordinate_denominator_proven"] is False
    assert subset["fixed_coordinate_denominator_proven"] is False
    assert subset["job_economics_complete"] is False
    assert subset["official_evidence_eligible"] is False
