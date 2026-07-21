from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest

from quant_rabbit.dojo_economic_transcript import (
    EconomicTranscriptRecorder,
    build_economic_transcript_header,
    reexecute_economic_transcript,
)
from quant_rabbit.dojo_economic_transcript_v2 import EconomicSegmentWriter
from quant_rabbit.dojo_economic_transcript_v2_auditor import (
    reexecute_v2_economic_segment_chain,
)
from quant_rabbit.dojo_economic_transcript_v4 import (
    DojoEconomicTranscriptV4Error,
    build_account_delta,
    build_shared_source_segment,
    publish_immutable_artifact,
    verify_v4_account_files,
    verify_v4_fixed_denominator,
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
            "policy_id": "shared-source-account-delta-v4-test",
            "expected_quote_pairs": ["USD_JPY"],
            "tradable_pairs": ["USD_JPY"],
            "active_worker_bindings": copy.deepcopy(WORKERS),
            "leverage": 20,
            "margin_closeout_fraction": 0.9,
            "max_margin_utilization_fraction": 0.8,
            "max_portfolio_stop_risk_fraction": 0.8,
            "max_open_and_pending_total": 8,
            "max_open_and_pending_per_pair": 8,
            "max_open_and_pending_per_family": 8,
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


def _source_inputs() -> list[dict[str, Any]]:
    epoch = 1_704_067_200
    result: list[dict[str, Any]] = []
    for watermark, (phase, bid) in enumerate(
        (("O", 145.0), ("H", 145.2), ("L", 144.8), ("C", 145.1)), 1
    ):
        timestamp = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
        result.append(
            {
                "epoch": epoch,
                "phase": phase,
                "intrabar": "OHLC",
                "quote_watermark": watermark,
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


def _empty_proposals(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        seal_worker_proposal(
            snapshot,
            {
                **binding,
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "risk_reducing_intents": [],
                "new_risk_intents": [],
            },
        )
        for binding in WORKERS
    ]


def _produce_account(
    *,
    source: Mapping[str, Any],
    coordinate_id: str,
    policy: Mapping[str, Any],
    v1_recorder: EconomicTranscriptRecorder | None = None,
) -> dict[str, Any]:
    session = PortfolioReplaySession(policy=policy, initial_balance_jpy=200_000.0)
    start = session.export_checkpoint()
    delta_rows: list[dict[str, Any]] = []
    v3_rows: list[dict[str, Any]] = []
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
        proposals = _empty_proposals(snapshot)
        proposal_batch = seal_worker_proposal_batch(snapshot, proposals)
        receipt = session.consume_proposal_batch(proposal_batch)
        if v1_recorder is not None:
            v1_recorder.record_quote_batch(
                coordinate_id=coordinate_id,
                epoch=source_batch["epoch"],
                phase=source_batch["phase"],
                intrabar=source_batch["intrabar"],
                quote_watermark=source_batch["quote_watermark"],
                quotes=source_batch["quotes"],
                quote_batch_sha256_value=source_batch["quote_batch_sha256"],
                source_batch_chain_sha256=source_batch[
                    "source_batch_chain_sha256"
                ],
            )
            v1_recorder.record_post_exit_snapshot(snapshot)
            v1_recorder.record_worker_proposal_batch(proposal_batch)
            v1_recorder.record_allocation_receipt(receipt)
        delta_rows.append(
            {
                "source_ordinal": ordinal,
                "source_quote_batch_sha256": source_batch["quote_batch_sha256"],
                "source_batch_chain_sha256": source_batch[
                    "source_batch_chain_sha256"
                ],
                "post_exit_snapshot_sha256": snapshot["snapshot_sha256"],
                "non_hold_proposals": [],
                "allocation_receipt": receipt,
            }
        )
        v3_body: dict[str, Any] = {
            "coordinate_ordinal": ordinal,
            "source_offset_start": ordinal,
            "source_offset_end_exclusive": ordinal + 1,
            "quote": {
                "coordinate_id": coordinate_id,
                "epoch": source_batch["epoch"],
                "phase": source_batch["phase"],
                "intrabar": source_batch["intrabar"],
                "quote_watermark": source_batch["quote_watermark"],
                "quotes": source_batch["quotes"],
                "quote_batch_sha256": source_batch["quote_batch_sha256"],
                "source_batch_chain_sha256": source_batch[
                    "source_batch_chain_sha256"
                ],
            },
            "post_exit_snapshot": snapshot,
            "non_hold_proposals": [],
            "allocation_receipt": receipt,
        }
        v3_body["batch_sha256"] = canonical_portfolio_sha256(v3_body)
        v3_rows.append(v3_body)
    terminal = session.export_checkpoint()
    result = session.finalize(terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF)
    return {
        "start_checkpoint": start,
        "terminal_checkpoint": terminal,
        "delta_rows": delta_rows,
        "v3_rows": v3_rows,
        "result": result,
    }


def _v1_header(
    *, policy: Mapping[str, Any], coordinate_id: str, transcript_id: str
) -> dict[str, Any]:
    return build_economic_transcript_header(
        transcript_id=transcript_id,
        coordinate_id=coordinate_id,
        portfolio_policy=policy,
        input_bindings={
            "job_sha256": "1" * 64,
            "claim_sha256": "2" * 64,
            "source_slice_receipt_sha256": "3" * 64,
            "worker_runtime_binding_sha256": "4" * 64,
            "cost_policy_sha256": "5" * 64,
            "risk_policy_sha256": "6" * 64,
            "replay_engine_sha256": "7" * 64,
            "portfolio_policy_binding_sha256": "8" * 64,
            "predecessor_state_sha256": None,
            "predecessor_portfolio_carry_state_sha256": None,
            "predecessor_source_batch_chain_sha256": "0" * 64,
        },
        terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF,
        expected_quote_batch_count=4,
        initial_balance_jpy=200_000.0,
        predecessor_portfolio_carry_state=None,
    )


def test_v1_v3_v4_differential_and_shared_source_scaling(tmp_path: Path) -> None:
    policy = _policy()
    source = build_shared_source_segment(
        job_sha256="1" * 64,
        source_slice_receipt_sha256="3" * 64,
        batches=_source_inputs(),
    )
    coordinate_a = "account-coordinate-a"
    v1_path = tmp_path / "v1.jsonl"
    v1 = EconomicTranscriptRecorder(
        v1_path,
        _v1_header(
            policy=policy,
            coordinate_id=coordinate_a,
            transcript_id="v1-differential-a",
        ),
    )
    produced_a = _produce_account(
        source=source,
        coordinate_id=coordinate_a,
        policy=policy,
        v1_recorder=v1,
    )
    v1.seal_success(
        terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF,
        portfolio_result=produced_a["result"],
        source_batch_chain_sha256=source["terminal_source_batch_chain_sha256"],
    )
    v1_attestation = reexecute_economic_transcript(v1_path)

    v3_path_a = tmp_path / "v3-a.json"
    EconomicSegmentWriter(v3_path_a).publish(
        transcript_id="v3-differential-a",
        job_sha256="1" * 64,
        segment_index=0,
        prior_segment_sha256="0" * 64,
        portfolio_policy=policy,
        source_slice_receipt_sha256="3" * 64,
        source_offset_start=0,
        source_offset_end_exclusive=4,
        prior_source_batch_chain_sha256="0" * 64,
        expected_job_coordinate_count=4,
        segment_coordinate_start=0,
        start_checkpoint=produced_a["start_checkpoint"],
        terminal_checkpoint=produced_a["terminal_checkpoint"],
        batches=produced_a["v3_rows"],
        terminal_segment=True,
    )
    v3_attestation = reexecute_v2_economic_segment_chain(
        [v3_path_a],
        coordinate_id=coordinate_a,
        terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF,
    )

    delta_a = build_account_delta(
        transcript_id="v4-differential-a",
        coordinate_id=coordinate_a,
        shared_source_segment=source,
        portfolio_policy=policy,
        start_checkpoint=produced_a["start_checkpoint"],
        terminal_checkpoint=produced_a["terminal_checkpoint"],
        start_worker_state={"calls": 0},
        terminal_worker_state={"calls": 4},
        terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF,
        producer_portfolio_result=produced_a["result"],
        delta_rows=produced_a["delta_rows"],
    )
    source_path = tmp_path / "shared-source-v4.json"
    delta_path_a = tmp_path / "account-a-v4.json"
    publish_immutable_artifact(source_path, source)
    publish_immutable_artifact(delta_path_a, delta_a)
    v4_attestation_a = verify_v4_account_files(source_path, delta_path_a)

    result_sha = produced_a["result"]["result_sha256"]
    assert v1_attestation["portfolio_result_sha256"] == result_sha
    assert v3_attestation["portfolio_result_sha256"] == result_sha
    assert v4_attestation_a["portfolio_result_sha256"] == result_sha
    assert delta_a["quote_arrays_embedded"] is False
    assert delta_a["full_snapshots_embedded"] is False
    assert all("quotes" not in row for row in delta_a["delta_rows"])
    assert v4_attestation_a["order_authority"] == "NONE"
    assert v4_attestation_a["live_permission"] is False

    coordinate_b = "account-coordinate-b"
    produced_b = _produce_account(
        source=source,
        coordinate_id=coordinate_b,
        policy=policy,
    )
    delta_b = build_account_delta(
        transcript_id="v4-differential-b",
        coordinate_id=coordinate_b,
        shared_source_segment=source,
        portfolio_policy=policy,
        start_checkpoint=produced_b["start_checkpoint"],
        terminal_checkpoint=produced_b["terminal_checkpoint"],
        start_worker_state={"calls": 0},
        terminal_worker_state={"calls": 4},
        terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF,
        producer_portfolio_result=produced_b["result"],
        delta_rows=produced_b["delta_rows"],
    )
    delta_path_b = tmp_path / "account-b-v4.json"
    publish_immutable_artifact(delta_path_b, delta_b)
    one_account = verify_v4_fixed_denominator(
        source_path=source_path,
        account_delta_paths=[delta_path_a],
        expected_coordinate_ids=[coordinate_a],
    )
    two_accounts = verify_v4_fixed_denominator(
        source_path=source_path,
        account_delta_paths=[delta_path_a, delta_path_b],
        expected_coordinate_ids=sorted([coordinate_a, coordinate_b]),
    )
    assert one_account["source_file_bytes"] == two_accounts["source_file_bytes"]
    assert two_accounts["source_bytes_scale_with_coordinate_count"] is False
    assert two_accounts["account_delta_file_bytes"] > one_account[
        "account_delta_file_bytes"
    ]
    assert two_accounts["coordinate_count"] == 2
    assert two_accounts["official_evidence_eligible"] is False


def test_tamper_reorder_fork_and_truncation_fail_closed(tmp_path: Path) -> None:
    policy = _policy()
    source = build_shared_source_segment(
        job_sha256="1" * 64,
        source_slice_receipt_sha256="3" * 64,
        batches=_source_inputs(),
    )
    coordinate_a = "account-coordinate-a"
    produced = _produce_account(
        source=source,
        coordinate_id=coordinate_a,
        policy=policy,
    )
    delta = build_account_delta(
        transcript_id="v4-adversarial-a",
        coordinate_id=coordinate_a,
        shared_source_segment=source,
        portfolio_policy=policy,
        start_checkpoint=produced["start_checkpoint"],
        terminal_checkpoint=produced["terminal_checkpoint"],
        start_worker_state={"calls": 0},
        terminal_worker_state={"calls": 4},
        terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF,
        producer_portfolio_result=produced["result"],
        delta_rows=produced["delta_rows"],
    )
    source_path = tmp_path / "source.json"
    delta_path = tmp_path / "delta.json"
    publish_immutable_artifact(source_path, source)
    publish_immutable_artifact(delta_path, delta)

    tampered = copy.deepcopy(source)
    tampered["batches"][0]["quotes"][0]["bid"] = 999.0
    tampered["source_segment_sha256"] = canonical_portfolio_sha256(
        {key: value for key, value in tampered.items() if key != "source_segment_sha256"}
    )
    tampered_path = tmp_path / "source-tampered.json"
    publish_immutable_artifact(tampered_path, tampered)
    with pytest.raises(
        DojoEconomicTranscriptV4Error,
        match="canonical|denominator|digest",
    ):
        verify_v4_account_files(tampered_path, delta_path)

    reordered = copy.deepcopy(source)
    reordered["batches"] = list(reversed(reordered["batches"]))
    reordered["source_batches_sha256"] = canonical_portfolio_sha256(
        reordered["batches"]
    )
    reordered["source_segment_sha256"] = canonical_portfolio_sha256(
        {key: value for key, value in reordered.items() if key != "source_segment_sha256"}
    )
    reordered_path = tmp_path / "source-reordered.json"
    publish_immutable_artifact(reordered_path, reordered)
    with pytest.raises(DojoEconomicTranscriptV4Error, match="ordinal"):
        verify_v4_account_files(reordered_path, delta_path)

    fork = build_account_delta(
        transcript_id="v4-adversarial-a",
        coordinate_id=coordinate_a,
        shared_source_segment=source,
        portfolio_policy=policy,
        start_checkpoint=produced["start_checkpoint"],
        terminal_checkpoint=produced["terminal_checkpoint"],
        start_worker_state={"calls": 0},
        terminal_worker_state={"calls": 4, "fork": True},
        terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF,
        producer_portfolio_result=produced["result"],
        delta_rows=produced["delta_rows"],
    )
    fork_path = tmp_path / "delta-fork.json"
    publish_immutable_artifact(fork_path, fork)
    with pytest.raises(DojoEconomicTranscriptV4Error, match="duplicate|forked"):
        verify_v4_fixed_denominator(
            source_path=source_path,
            account_delta_paths=[delta_path, fork_path],
            expected_coordinate_ids=[coordinate_a, "account-coordinate-b"],
        )

    truncated_path = tmp_path / "delta-truncated.json"
    truncated_path.write_bytes(delta_path.read_bytes()[:-1])
    with pytest.raises(DojoEconomicTranscriptV4Error, match="truncated"):
        verify_v4_account_files(source_path, truncated_path)
