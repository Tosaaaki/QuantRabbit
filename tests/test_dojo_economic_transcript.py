from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from quant_rabbit.dojo_economic_transcript import (
    DojoEconomicTranscriptError,
    EconomicTranscriptRecorder,
    build_economic_transcript_header,
    build_fixed_denominator_reexecution_attestation,
    reexecute_economic_transcript,
)
from quant_rabbit.dojo_portfolio_replay_reducer import (
    MONTH_END_FLAT_SETTLEMENT,
    MONTH_END_MTM_WITH_STATE_HANDOFF,
    PortfolioReplaySession,
    canonical_portfolio_sha256,
    quote_batch_sha256,
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
            "policy_id": "transcript-test-policy",
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


def _bindings(
    *,
    predecessor_state_sha256: str | None = None,
    predecessor_portfolio_carry_state_sha256: str | None = None,
    predecessor_source_batch_chain_sha256: str = "0" * 64,
) -> dict[str, Any]:
    return {
        "job_sha256": "1" * 64,
        "claim_sha256": "2" * 64,
        "source_slice_receipt_sha256": "3" * 64,
        "worker_runtime_binding_sha256": "4" * 64,
        "cost_policy_sha256": "5" * 64,
        "risk_policy_sha256": "6" * 64,
        "replay_engine_sha256": "7" * 64,
        "portfolio_policy_binding_sha256": "8" * 64,
        "predecessor_state_sha256": predecessor_state_sha256,
        "predecessor_portfolio_carry_state_sha256": (
            predecessor_portfolio_carry_state_sha256
        ),
        "predecessor_source_batch_chain_sha256": (
            predecessor_source_batch_chain_sha256
        ),
    }


def _quotes(epoch: int, phase: str, bid: float) -> list[dict[str, Any]]:
    timestamp = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    return [
        {
            "pair": "USD_JPY",
            "bid": bid,
            "ask": bid + 0.02,
            "timestamp": f"{timestamp}#{phase}",
        }
    ]


def _batch(snapshot: Mapping[str, Any], *, open_market: bool) -> dict[str, Any]:
    proposals = []
    for binding in WORKERS:
        intents = []
        if open_market and binding["worker_id"] == "worker-a":
            intents = [
                {
                    "intent_id": "open-a",
                    "action": "MARKET",
                    "parameters": {
                        "pair": "USD_JPY",
                        "side": "LONG",
                        "units": 100.0,
                        "entry_price": 145.02,
                        "tp_price": 147.0,
                        "sl_price": 143.0,
                        "stress_cost_pips": 999.0,
                        "hard_max_holding_seconds": 3_600,
                        "valid_until_epoch": snapshot["epoch"] + 3_600,
                        "expected_net_edge_jpy": 999_999.0,
                    },
                    "reason_code": "TRANSCRIPT_TEST_ONLY",
                }
            ]
        proposal = seal_worker_proposal(
            snapshot,
            {
                **binding,
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "risk_reducing_intents": [],
                "new_risk_intents": intents,
            },
        )
        proposals.append(proposal)
    batch = seal_worker_proposal_batch(snapshot, proposals)
    # Both workers must be present even when one or both explicitly HOLD.
    assert batch["proposal_count"] == len(WORKERS)
    return batch


def _next_source_chain(
    previous: str, *, quote_digest: str, epoch: int, phase: str, watermark: int
) -> str:
    return canonical_portfolio_sha256(
        {
            "previous_batch_chain_sha256": previous,
            "quote_batch_sha256": quote_digest,
            "epoch": epoch,
            "phase": phase,
            "quote_watermark": watermark,
        }
    )


def _record_coordinate(
    *,
    recorder: EconomicTranscriptRecorder,
    session: PortfolioReplaySession,
    coordinate_id: str,
    epoch: int,
    phase: str,
    watermark: int,
    bid: float,
    previous_source_chain: str,
    open_market: bool,
) -> str:
    quotes = _quotes(epoch, phase, bid)
    quote_digest = quote_batch_sha256(
        epoch=epoch,
        phase=phase,
        intrabar="OHLC",
        quote_watermark=watermark,
        quotes=quotes,
    )
    source_chain = _next_source_chain(
        previous_source_chain,
        quote_digest=quote_digest,
        epoch=epoch,
        phase=phase,
        watermark=watermark,
    )
    recorder.record_quote_batch(
        coordinate_id=coordinate_id,
        epoch=epoch,
        phase=phase,
        intrabar="OHLC",
        quote_watermark=watermark,
        quotes=quotes,
        quote_batch_sha256_value=quote_digest,
        source_batch_chain_sha256=source_chain,
    )
    snapshot = session.prepare_coordinate(
        coordinate_id=coordinate_id,
        epoch=epoch,
        phase=phase,
        intrabar="OHLC",
        quote_watermark=watermark,
        quotes=quotes,
        quote_batch_sha256_value=quote_digest,
    )
    recorder.record_post_exit_snapshot(snapshot)
    proposal_batch = _batch(snapshot, open_market=open_market)
    recorder.record_worker_proposal_batch(proposal_batch)
    receipt = session.consume_proposal_batch(proposal_batch)
    recorder.record_allocation_receipt(receipt)
    return source_chain


def _successful_transcript(
    path: Path,
    *,
    coordinate_id: str,
    initial_balance_jpy: float | None = 200_000.0,
    predecessor_carry: Mapping[str, Any] | None = None,
    predecessor_state_sha256: str | None = None,
    predecessor_source_chain: str = "0" * 64,
    start_epoch: int = 1_704_067_200,
    start_watermark: int = 0,
    terminal_policy: str = MONTH_END_FLAT_SETTLEMENT,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    policy = _policy()
    header = build_economic_transcript_header(
        transcript_id=f"transcript-{coordinate_id}",
        coordinate_id=coordinate_id,
        portfolio_policy=policy,
        input_bindings=_bindings(
            predecessor_state_sha256=predecessor_state_sha256,
            predecessor_portfolio_carry_state_sha256=(
                None
                if predecessor_carry is None
                else predecessor_carry["carry_state_sha256"]
            ),
            predecessor_source_batch_chain_sha256=predecessor_source_chain,
        ),
        terminal_policy=terminal_policy,
        expected_quote_batch_count=4,
        initial_balance_jpy=initial_balance_jpy,
        predecessor_portfolio_carry_state=predecessor_carry,
    )
    session = PortfolioReplaySession(
        policy=policy,
        initial_balance_jpy=initial_balance_jpy,
        carry_state=predecessor_carry,
    )
    source_chain = predecessor_source_chain
    phases: Sequence[tuple[str, float]] = (
        ("O", 145.00),
        ("H", 145.20),
        ("L", 144.80),
        ("C", 145.10),
    )
    with EconomicTranscriptRecorder(path, header) as recorder:
        for offset, (phase, bid) in enumerate(phases, 1):
            source_chain = _record_coordinate(
                recorder=recorder,
                session=session,
                coordinate_id=coordinate_id,
                epoch=start_epoch,
                phase=phase,
                watermark=start_watermark + offset,
                bid=bid,
                previous_source_chain=source_chain,
                open_market=(offset == 1 and predecessor_carry is None),
            )
        result = session.finalize(terminal_policy=terminal_policy)
        recorder.seal_success(
            terminal_policy=terminal_policy,
            portfolio_result=result,
            source_batch_chain_sha256=source_chain,
        )
    return reexecute_economic_transcript(path), result, source_chain


def _rewrite_rechained(path: Path, mutate: Any) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    previous = "0" * 64
    for index, row in enumerate(rows):
        row["record_index"] = index
        row["previous_record_sha256"] = previous
        unsigned = {key: value for key, value in row.items() if key != "record_sha256"}
        row["record_sha256"] = canonical_portfolio_sha256(unsigned)
        previous = row["record_sha256"]
    path.write_bytes(
        b"".join(
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
            for row in rows
        )
    )


def test_independent_reexecution_matches_quotes_holds_allocation_and_carry(
    tmp_path: Path,
) -> None:
    attestation, result, _ = _successful_transcript(
        tmp_path / "economic-transcript.jsonl", coordinate_id="coord-a"
    )

    assert attestation["status"] == "VERIFIED_COMPLETE"
    assert attestation["independent_economic_reexecution_passed"] is True
    assert attestation["portfolio_result_sha256"] == result["result_sha256"]
    assert attestation["portfolio_carry_state_sha256"] == result["carry_state_sha256"]
    assert attestation["official_evidence_eligible"] is False
    assert attestation["live_permission"] is False


def test_module_entrypoint_reexecutes_in_a_separate_process(tmp_path: Path) -> None:
    path = tmp_path / "subprocess.jsonl"
    attestation, _, _ = _successful_transcript(path, coordinate_id="coord-process")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_rabbit.dojo_economic_transcript",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert json.loads(completed.stdout) == attestation


def test_carry_continuation_reexecutes_from_predecessor_state(tmp_path: Path) -> None:
    _, first_result, first_source_chain = _successful_transcript(
        tmp_path / "first.jsonl",
        coordinate_id="coord-first",
        terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF,
    )

    attestation, second_result, _ = _successful_transcript(
        tmp_path / "second.jsonl",
        coordinate_id="coord-second",
        initial_balance_jpy=None,
        predecessor_carry=first_result["carry_state"],
        predecessor_state_sha256="d" * 64,
        predecessor_source_chain=first_source_chain,
        start_epoch=1_704_067_260,
        start_watermark=4,
    )

    assert attestation["status"] == "VERIFIED_COMPLETE"
    assert second_result["start_equity_jpy"] == first_result["end_equity_jpy"]

    with pytest.raises(DojoEconomicTranscriptError, match="carry digest binding"):
        build_economic_transcript_header(
            transcript_id="transcript-bad-carry",
            coordinate_id="coord-bad-carry",
            portfolio_policy=_policy(),
            input_bindings=_bindings(
                predecessor_state_sha256="d" * 64,
                predecessor_portfolio_carry_state_sha256="c" * 64,
                predecessor_source_batch_chain_sha256=first_source_chain,
            ),
            terminal_policy=MONTH_END_FLAT_SETTLEMENT,
            expected_quote_batch_count=4,
            predecessor_portfolio_carry_state=first_result["carry_state"],
        )


def test_reexecutor_rejects_rehashed_forged_allocation_economics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forged-allocation.jsonl"
    _successful_transcript(path, coordinate_id="coord-forged")

    def mutate(rows: list[dict[str, Any]]) -> None:
        row = next(item for item in rows if item["event_type"] == "ALLOCATION_RECEIPT")
        receipt = row["payload"]["receipt"]
        receipt["ending_equity_jpy"] += 1.0
        unsigned = {
            key: value
            for key, value in receipt.items()
            if key != "coordinate_receipt_sha256"
        }
        receipt["coordinate_receipt_sha256"] = canonical_portfolio_sha256(unsigned)

    _rewrite_rechained(path, mutate)

    with pytest.raises(
        DojoEconomicTranscriptError,
        match="independent allocation/admission receipt mismatch",
    ):
        reexecute_economic_transcript(path)


@pytest.mark.parametrize("mutation", ["missing", "reordered", "doubled"])
def test_reexecutor_rejects_missing_reordered_or_double_counted_events(
    tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / f"{mutation}.jsonl"
    _successful_transcript(path, coordinate_id=f"coord-{mutation}")

    def mutate(rows: list[dict[str, Any]]) -> None:
        proposal_index = next(
            index
            for index, row in enumerate(rows)
            if row["event_type"] == "WORKER_PROPOSAL_BATCH"
        )
        allocation_index = proposal_index + 1
        if mutation == "missing":
            rows.pop(proposal_index)
        elif mutation == "reordered":
            rows[proposal_index], rows[allocation_index] = (
                rows[allocation_index],
                rows[proposal_index],
            )
        else:
            rows.insert(allocation_index + 1, copy.deepcopy(rows[allocation_index]))

    _rewrite_rechained(path, mutate)

    with pytest.raises(DojoEconomicTranscriptError, match="out of order"):
        reexecute_economic_transcript(path)


def test_failure_attestation_suppresses_partial_economics(tmp_path: Path) -> None:
    policy = _policy()
    coordinate_id = "coord-failed"
    header = build_economic_transcript_header(
        transcript_id="transcript-failed",
        coordinate_id=coordinate_id,
        portfolio_policy=policy,
        input_bindings=_bindings(),
        terminal_policy=MONTH_END_FLAT_SETTLEMENT,
        expected_quote_batch_count=4,
        initial_balance_jpy=200_000,
    )
    session = PortfolioReplaySession(policy=policy, initial_balance_jpy=200_000)
    path = tmp_path / "failed.jsonl"
    with EconomicTranscriptRecorder(path, header) as recorder:
        _record_coordinate(
            recorder=recorder,
            session=session,
            coordinate_id=coordinate_id,
            epoch=1_704_067_200,
            phase="O",
            watermark=1,
            bid=145.0,
            previous_source_chain="0" * 64,
            open_market=True,
        )
        recorder.seal_failure(
            failure_code="WORKER_PROTOCOL_FAILURE",
            failure_stage="PROPOSAL_COLLECTION",
            failure_evidence_sha256="e" * 64,
        )

    attestation = reexecute_economic_transcript(path)

    assert attestation["status"] == "VERIFIED_FAILED_TRANSCRIPT"
    assert attestation["partial_economics_reported"] is False
    forbidden_fragments = ("balance", "equity", "pnl", "result", "carry", "margin")
    assert not any(
        fragment in key.lower()
        for key in attestation
        for fragment in forbidden_fragments
    )


def test_fixed_denominator_mixed_failure_suppresses_success_hash(
    tmp_path: Path,
) -> None:
    success, _, _ = _successful_transcript(
        tmp_path / "success.jsonl", coordinate_id="coord-a"
    )
    policy = _policy()
    header = build_economic_transcript_header(
        transcript_id="transcript-coord-b",
        coordinate_id="coord-b",
        portfolio_policy=policy,
        input_bindings=_bindings(),
        terminal_policy=MONTH_END_FLAT_SETTLEMENT,
        expected_quote_batch_count=4,
        initial_balance_jpy=200_000,
    )
    failed_path = tmp_path / "failed-b.jsonl"
    with EconomicTranscriptRecorder(failed_path, header) as recorder:
        recorder.seal_failure(
            failure_code="SOURCE_STREAM_FAILURE",
            failure_stage="SOURCE_OPEN",
            failure_evidence_sha256="f" * 64,
        )
    failed = reexecute_economic_transcript(failed_path)

    combined = build_fixed_denominator_reexecution_attestation(
        expected_coordinate_ids=["coord-a", "coord-b"],
        attestations_by_coordinate={"coord-a": success, "coord-b": failed},
    )

    assert combined["status"] == "INCOMPLETE_FAILED"
    assert combined["portfolio_result_sha256_by_coordinate"] == {}
    assert combined["downstream_terminal_reduction_allowed"] is False
    assert success["portfolio_result_sha256"] not in json.dumps(combined)


def test_reexecutor_rejects_truncated_or_unterminated_transcript(
    tmp_path: Path,
) -> None:
    path = tmp_path / "truncated.jsonl"
    _successful_transcript(path, coordinate_id="coord-truncated")
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(DojoEconomicTranscriptError, match="truncated"):
        reexecute_economic_transcript(path)


def test_recorder_rejects_success_with_unacknowledged_snapshot(tmp_path: Path) -> None:
    policy = _policy()
    header = build_economic_transcript_header(
        transcript_id="transcript-unfinished",
        coordinate_id="coord-unfinished",
        portfolio_policy=policy,
        input_bindings=_bindings(),
        terminal_policy=MONTH_END_FLAT_SETTLEMENT,
        expected_quote_batch_count=1,
        initial_balance_jpy=200_000,
    )
    recorder = EconomicTranscriptRecorder(tmp_path / "unfinished.jsonl", header)
    quotes = _quotes(1_704_067_200, "O", 145.0)
    digest = quote_batch_sha256(
        epoch=1_704_067_200,
        phase="O",
        intrabar="OHLC",
        quote_watermark=1,
        quotes=quotes,
    )
    source_chain = _next_source_chain(
        "0" * 64,
        quote_digest=digest,
        epoch=1_704_067_200,
        phase="O",
        watermark=1,
    )
    recorder.record_quote_batch(
        coordinate_id="coord-unfinished",
        epoch=1_704_067_200,
        phase="O",
        intrabar="OHLC",
        quote_watermark=1,
        quotes=quotes,
        quote_batch_sha256_value=digest,
        source_batch_chain_sha256=source_chain,
    )

    with pytest.raises(
        DojoEconomicTranscriptError, match="complete coordinate transactions"
    ):
        recorder.seal_success(
            terminal_policy=MONTH_END_FLAT_SETTLEMENT,
            portfolio_result={},
            source_batch_chain_sha256=source_chain,
        )
    recorder.close()


def test_precommitted_quote_denominator_blocks_early_success(tmp_path: Path) -> None:
    policy = _policy()
    coordinate_id = "coord-short"
    header = build_economic_transcript_header(
        transcript_id="transcript-short",
        coordinate_id=coordinate_id,
        portfolio_policy=policy,
        input_bindings=_bindings(),
        terminal_policy=MONTH_END_FLAT_SETTLEMENT,
        expected_quote_batch_count=2,
        initial_balance_jpy=200_000,
    )
    session = PortfolioReplaySession(policy=policy, initial_balance_jpy=200_000)
    recorder = EconomicTranscriptRecorder(tmp_path / "short.jsonl", header)
    source_chain = _record_coordinate(
        recorder=recorder,
        session=session,
        coordinate_id=coordinate_id,
        epoch=1_704_067_200,
        phase="O",
        watermark=1,
        bid=145.0,
        previous_source_chain="0" * 64,
        open_market=True,
    )
    result = session.finalize(terminal_policy=MONTH_END_FLAT_SETTLEMENT)

    with pytest.raises(DojoEconomicTranscriptError, match="binding drifted"):
        recorder.seal_success(
            terminal_policy=MONTH_END_FLAT_SETTLEMENT,
            portfolio_result=result,
            source_batch_chain_sha256=source_chain,
        )
    recorder.seal_failure(
        failure_code="SOURCE_STREAM_TRUNCATED",
        failure_stage="SOURCE_READ",
        failure_evidence_sha256="9" * 64,
    )
