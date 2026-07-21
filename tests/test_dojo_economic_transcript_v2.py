from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest

from quant_rabbit.dojo_economic_transcript_v2 import (
    DojoEconomicSegmentError,
    EconomicSegmentWriter,
    build_economic_segment,
    verify_economic_segment,
    verify_economic_segment_chain,
)
from quant_rabbit.dojo_portfolio_replay_reducer import (
    DojoPortfolioReplayError,
    MONTH_END_MTM_WITH_STATE_HANDOFF,
    PortfolioReplaySession,
    canonical_portfolio_sha256,
    quote_batch_sha256,
    seal_portfolio_policy,
    verify_portfolio_replay_checkpoint,
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
            "policy_id": "economic-segment-v2-test-policy",
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


def _source_chain(
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


def _proposal(
    snapshot: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    open_market: bool,
) -> dict[str, Any]:
    intents: list[dict[str, Any]] = []
    if open_market and binding["worker_id"] == "worker-a":
        intents = [
            {
                "intent_id": "open-a",
                "action": "MARKET",
                "parameters": {
                    "pair": "USD_JPY",
                    "side": "LONG",
                    "units": 100.0,
                    "entry_price": snapshot["quotes"][0]["ask"],
                    "tp_price": 147.0,
                    "sl_price": 143.0,
                    "stress_cost_pips": 999.0,
                    "hard_max_holding_seconds": 3_600,
                    "valid_until_epoch": snapshot["epoch"] + 3_600,
                    "expected_net_edge_jpy": 999_999.0,
                },
                "reason_code": "SEGMENT_V2_TEST_ONLY",
            }
        ]
    return seal_worker_proposal(
        snapshot,
        {
            **binding,
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "risk_reducing_intents": [],
            "new_risk_intents": intents,
        },
    )


def _batch_row(
    *,
    session: PortfolioReplaySession,
    ordinal: int,
    epoch: int,
    phase: str,
    watermark: int,
    bid: float,
    source_offset_start: int,
    prior_source_chain: str,
    open_market: bool,
) -> tuple[dict[str, Any], str]:
    quotes = _quotes(epoch, phase, bid)
    quote_digest = quote_batch_sha256(
        epoch=epoch,
        phase=phase,
        intrabar="OHLC",
        quote_watermark=watermark,
        quotes=quotes,
    )
    source_chain = _source_chain(
        prior_source_chain,
        quote_digest=quote_digest,
        epoch=epoch,
        phase=phase,
        watermark=watermark,
    )
    quote = {
        "coordinate_id": "economic-coordinate-a",
        "epoch": epoch,
        "phase": phase,
        "intrabar": "OHLC",
        "quote_watermark": watermark,
        "quotes": quotes,
        "quote_batch_sha256": quote_digest,
        "source_batch_chain_sha256": source_chain,
    }
    snapshot = session.prepare_coordinate(
        coordinate_id=quote["coordinate_id"],
        epoch=epoch,
        phase=phase,
        intrabar="OHLC",
        quote_watermark=watermark,
        quotes=quotes,
        quote_batch_sha256_value=quote_digest,
    )
    proposals = [
        _proposal(snapshot, binding, open_market=open_market) for binding in WORKERS
    ]
    full_batch = seal_worker_proposal_batch(snapshot, proposals)
    receipt = session.consume_proposal_batch(full_batch)
    non_hold = [
        row
        for row in proposals
        if row["intent_counts"]["risk_reducing"]
        or row["intent_counts"]["new_risk"]
    ]
    # Source offsets are immutable byte ranges from the source-slice receipt;
    # the test uses a deterministic fixed-size synthetic range per coordinate.
    source_offset_end = source_offset_start + 100
    body: dict[str, Any] = {
        "coordinate_ordinal": ordinal,
        "source_offset_start": source_offset_start,
        "source_offset_end_exclusive": source_offset_end,
        "quote": quote,
        "post_exit_snapshot": snapshot,
        "non_hold_proposals": non_hold,
        "allocation_receipt": receipt,
    }
    body["batch_sha256"] = canonical_portfolio_sha256(body)
    return body, source_chain


def _make_segments(
    tmp_path: Path,
    *,
    fork_second: bool = False,
) -> tuple[Path, Path, PortfolioReplaySession, dict[str, Any], dict[str, Any]]:
    policy = _policy()
    session = PortfolioReplaySession(policy=policy, initial_balance_jpy=200_000.0)
    checkpoint_0 = session.export_checkpoint()
    epoch = 1_704_067_200
    source_chain = "0" * 64
    rows_0: list[dict[str, Any]] = []
    for ordinal, (phase, bid) in enumerate((("O", 145.0), ("H", 145.2))):
        row, source_chain = _batch_row(
            session=session,
            ordinal=ordinal,
            epoch=epoch,
            phase=phase,
            watermark=ordinal + 1,
            bid=bid,
            source_offset_start=ordinal * 100,
            prior_source_chain=source_chain,
            open_market=ordinal == 0,
        )
        rows_0.append(row)
    checkpoint_1 = session.export_checkpoint()
    path_0 = tmp_path / "segment-000000.json"
    segment_0 = EconomicSegmentWriter(path_0).publish(
        transcript_id="economic-transcript-v2-a",
        job_sha256="1" * 64,
        segment_index=0,
        prior_segment_sha256="0" * 64,
        portfolio_policy=policy,
        source_slice_receipt_sha256="2" * 64,
        source_offset_start=0,
        source_offset_end_exclusive=200,
        prior_source_batch_chain_sha256="0" * 64,
        expected_job_coordinate_count=4,
        segment_coordinate_start=0,
        start_checkpoint=checkpoint_0,
        terminal_checkpoint=checkpoint_1,
        batches=rows_0,
        terminal_segment=False,
    )

    if fork_second:
        session = PortfolioReplaySession.restore_checkpoint(
            policy=policy,
            checkpoint=checkpoint_1,
        )
    rows_1: list[dict[str, Any]] = []
    second_prices = (("L", 144.7), ("C", 145.05)) if fork_second else (
        ("L", 144.8),
        ("C", 145.1),
    )
    for offset, (phase, bid) in enumerate(second_prices):
        ordinal = offset + 2
        row, source_chain = _batch_row(
            session=session,
            ordinal=ordinal,
            epoch=epoch,
            phase=phase,
            watermark=ordinal + 1,
            bid=bid,
            source_offset_start=ordinal * 100,
            prior_source_chain=source_chain,
            open_market=False,
        )
        rows_1.append(row)
    checkpoint_2 = session.export_checkpoint()
    suffix = "fork" if fork_second else "000001"
    path_1 = tmp_path / f"segment-{suffix}.json"
    EconomicSegmentWriter(path_1).publish(
        transcript_id="economic-transcript-v2-a",
        job_sha256="1" * 64,
        segment_index=1,
        prior_segment_sha256=segment_0["segment_sha256"],
        portfolio_policy=policy,
        source_slice_receipt_sha256="2" * 64,
        source_offset_start=200,
        source_offset_end_exclusive=400,
        prior_source_batch_chain_sha256=segment_0[
            "terminal_source_batch_chain_sha256"
        ],
        expected_job_coordinate_count=4,
        segment_coordinate_start=2,
        start_checkpoint=checkpoint_1,
        terminal_checkpoint=checkpoint_2,
        batches=rows_1,
        terminal_segment=True,
    )
    return path_0, path_1, session, checkpoint_1, policy


def _rewrite(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def test_batch_major_chain_restores_exactly_and_uses_implicit_no_intent(
    tmp_path: Path,
) -> None:
    path_0, path_1, uninterrupted, _, policy = _make_segments(tmp_path)

    attestation = verify_economic_segment_chain(
        [path_0, path_1],
        require_terminal=True,
    )
    terminal_segment = verify_economic_segment(path_1)
    restored = PortfolioReplaySession.restore_checkpoint(
        policy=policy,
        checkpoint=terminal_segment["terminal_checkpoint"],
    )
    uninterrupted_result = uninterrupted.finalize(
        terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF
    )
    restored_result = restored.finalize(
        terminal_policy=MONTH_END_MTM_WITH_STATE_HANDOFF
    )

    assert attestation["status"] == "VERIFIED_TERMINAL"
    assert attestation["completed_coordinate_count"] == 4
    assert attestation["deterministic_restore_verified"] is True
    assert restored_result == uninterrupted_result
    first = verify_economic_segment(path_0)
    assert first["coordinate_denominator"]["stored_non_hold_proposal_count"] == 1
    assert (
        first["coordinate_denominator"]["expanded_active_worker_proposal_count"]
        == 4
    )
    assert first["batches"][1]["non_hold_proposals"] == []
    assert first["publication_fsyncs_per_segment"] == 1
    assert first["fork_absence_proven"] is False
    assert first["official_evidence_eligible"] is False
    assert first["live_permission"] is False


def test_writer_performs_one_data_fsync_for_whole_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_rabbit.dojo_economic_transcript_v2 as module

    calls = 0
    real_fsync = module.os.fsync

    def counted_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", counted_fsync)
    _make_segments(tmp_path)
    assert calls == 2  # one publication fsync for each of the two whole segments


def test_tamper_reorder_fork_and_truncation_fail_closed(tmp_path: Path) -> None:
    path_0, path_1, _, _, _ = _make_segments(tmp_path)

    with pytest.raises(DojoEconomicSegmentError, match="reordered|forked|gapped"):
        verify_economic_segment_chain([path_1, path_0])

    fork_root = tmp_path / "fork"
    fork_root.mkdir()
    fork_0, fork_1, _, _, _ = _make_segments(fork_root, fork_second=True)
    with pytest.raises(DojoEconomicSegmentError, match="terminal|reordered|forked"):
        verify_economic_segment_chain([path_0, path_1, fork_1])
    assert verify_economic_segment_chain([fork_0, fork_1], require_terminal=True)[
        "status"
    ] == "VERIFIED_TERMINAL"

    tampered = json.loads(path_0.read_text())
    tampered["batches"][0]["quote"]["quotes"][0]["bid"] = 999.0
    # Re-hashing only the outer segment cannot forge the nested quote/batch and
    # deterministic reducer bindings.
    unsigned = {key: value for key, value in tampered.items() if key != "segment_sha256"}
    tampered["segment_sha256"] = canonical_portfolio_sha256(unsigned)
    tampered_path = tmp_path / "tampered.json"
    _rewrite(tampered_path, tampered)
    with pytest.raises(DojoEconomicSegmentError, match="batch digest|quote digest"):
        verify_economic_segment(tampered_path)

    truncated_path = tmp_path / "truncated.json"
    truncated_path.write_bytes(path_1.read_bytes()[:-1])
    with pytest.raises(DojoEconomicSegmentError, match="truncated"):
        verify_economic_segment(truncated_path)


def test_checkpoint_tamper_and_explicit_hold_fail_closed(tmp_path: Path) -> None:
    path_0, _, _, checkpoint_1, policy = _make_segments(tmp_path)
    damaged = copy.deepcopy(checkpoint_1)
    damaged["processed_coordinate_count"] += 1
    with pytest.raises(DojoPortfolioReplayError, match="checkpoint"):
        verify_portfolio_replay_checkpoint(policy=policy, checkpoint=damaged)

    segment = verify_economic_segment(path_0)
    explicit_hold = seal_worker_proposal(
        segment["batches"][1]["post_exit_snapshot"],
        {
            **WORKERS[0],
            "snapshot_sha256": segment["batches"][1]["post_exit_snapshot"][
                "snapshot_sha256"
            ],
            "risk_reducing_intents": [],
            "new_risk_intents": [],
        },
    )
    batch = copy.deepcopy(segment["batches"][1])
    batch["non_hold_proposals"] = [explicit_hold]
    unsigned = {key: value for key, value in batch.items() if key != "batch_sha256"}
    batch["batch_sha256"] = canonical_portfolio_sha256(unsigned)
    with pytest.raises(DojoEconomicSegmentError, match="explicit HOLD"):
        build_economic_segment(
            transcript_id=segment["transcript_id"],
            job_sha256=segment["job_sha256"],
            segment_index=segment["segment_index"],
            prior_segment_sha256=segment["prior_segment_sha256"],
            portfolio_policy=segment["portfolio_policy"],
            source_slice_receipt_sha256=segment["source_range"][
                "source_slice_receipt_sha256"
            ],
            source_offset_start=segment["source_range"]["offset_start"],
            source_offset_end_exclusive=segment["source_range"][
                "offset_end_exclusive"
            ],
            prior_source_batch_chain_sha256=segment[
                "prior_source_batch_chain_sha256"
            ],
            expected_job_coordinate_count=segment["coordinate_denominator"][
                "expected_job_coordinate_count"
            ],
            segment_coordinate_start=segment["coordinate_denominator"][
                "segment_coordinate_start"
            ],
            start_checkpoint=segment["start_checkpoint"],
            terminal_checkpoint=segment["terminal_checkpoint"],
            batches=[segment["batches"][0], batch],
            terminal_segment=segment["terminal_segment"],
        )
