from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from quant_rabbit.dojo_market_calendar import expected_oanda_fx_slots
from quant_rabbit.dojo_sparse_replay import build_sparse_replay_schedule_from_rows
from quant_rabbit.dojo_sparse_source_slice_v2 import (
    GENESIS_SHA256,
    DojoSparseSourceSliceV2Error,
    SparseSourceSliceV2,
    build_parent_quote_map_binding,
    build_sparse_source_slice_v2,
    canonical_source_sha256,
    consume_sparse_source_slice_v2,
    validate_parent_quote_map_binding,
    verify_sparse_source_receipt_chain_v2,
    verify_sparse_source_slice_v2,
)


UTC = timezone.utc
PAIRS = ("EUR_USD", "USD_JPY")


@dataclass(frozen=True)
class SourceFixture:
    start: datetime
    end: datetime
    granularity: str
    quote_maps: dict[str, dict[int, dict]]
    schedule: object
    parent_bindings: dict[str, dict]


def _sha(label: str) -> str:
    return canonical_source_sha256({"fixture": label})


def _quote(pair: str, epoch: int) -> dict:
    base = 1.1 if pair == "EUR_USD" else 150.0
    drift = (epoch % 17) * (0.00001 if pair == "EUR_USD" else 0.001)
    spread = 0.0002 if pair == "EUR_USD" else 0.02
    bid = [
        base + drift,
        base + drift + spread * 2,
        base + drift - spread,
        base + drift + spread,
    ]
    return {
        "pair": pair,
        "bid": bid,
        "ask": [value + spread for value in bid],
    }


def _source_fixture(
    *,
    granularity: str = "M1",
    start: datetime | None = None,
) -> SourceFixture:
    start = start or datetime(2026, 1, 6, 12, 0, tzinfo=UTC)
    step = timedelta(minutes=1 if granularity == "M1" else 5)
    end = start + (timedelta(hours=1) if granularity == "M1" else timedelta(hours=2))
    epochs = tuple(
        int(stamp.timestamp())
        for stamp in expected_oanda_fx_slots(start, end, step=step)
    )
    missing_indices = {5} if granularity == "M1" else {4, 5}
    quote_maps = {
        "EUR_USD": {epoch: _quote("EUR_USD", epoch) for epoch in epochs},
        "USD_JPY": {
            epoch: _quote("USD_JPY", epoch)
            for index, epoch in enumerate(epochs)
            if index not in missing_indices
        },
    }
    schedule = build_sparse_replay_schedule_from_rows(
        quote_maps,
        feed_pairs=PAIRS,
        start=start,
        end=end,
        granularity=granularity,
    )
    parent_bindings = {
        pair: build_parent_quote_map_binding(
            schedule=schedule,
            pair=pair,
            quote_map=quote_maps[pair],
            authentication={
                "raw_source_id": (f"raw:{granularity}:{pair}:{int(start.timestamp())}"),
                "raw_artifact_sha256": _sha(f"raw:{granularity}:{pair}:{start}"),
                "upstream_authentication_receipt_sha256": _sha(
                    f"auth:{granularity}:{pair}:{start}"
                ),
            },
        )
        for pair in PAIRS
    }
    return SourceFixture(
        start=start,
        end=end,
        granularity=granularity,
        quote_maps=quote_maps,
        schedule=schedule,
        parent_bindings=parent_bindings,
    )


@pytest.fixture
def m1_source() -> SourceFixture:
    return _source_fixture(granularity="M1")


@pytest.fixture
def m5_source() -> SourceFixture:
    return _source_fixture(granularity="M5")


def _build_slice(
    fixture: SourceFixture,
    *,
    stream_id: str = "sparse-stream-v2",
    slice_id: str = "slice-000",
    sequence: int = 0,
    prior: str = GENESIS_SHA256,
) -> SparseSourceSliceV2:
    return build_sparse_source_slice_v2(
        stream_id=stream_id,
        slice_id=slice_id,
        sequence=sequence,
        prior_receipt_sha256=prior,
        schedule=fixture.schedule,
        pair_quote_maps=fixture.quote_maps,
        parent_bindings=fixture.parent_bindings,
    )


def _verify(
    source_slice: SparseSourceSliceV2, fixture: SourceFixture
) -> SparseSourceSliceV2:
    return verify_sparse_source_slice_v2(
        source_slice,
        expected_stream_id="sparse-stream-v2",
        expected_slice_id="slice-000",
        expected_sequence=0,
        expected_prior_receipt_sha256=GENESIS_SHA256,
        schedule=fixture.schedule,
        pair_quote_maps=fixture.quote_maps,
        parent_bindings=fixture.parent_bindings,
    )


def test_m1_builds_canonical_union_rows_with_parent_quote_identity(
    m1_source: SourceFixture,
) -> None:
    source_slice = _verify(_build_slice(m1_source), m1_source)
    receipt = source_slice.receipt

    assert receipt["granularity"] == "M1"
    assert receipt["source_row_count"] == 60
    assert receipt["observed_quote_count"] == 60 + 59
    assert (
        receipt["availability_mask_sha256"]
        == receipt["schedule_coverage_receipt"]["availability_mask_sha256"]
    )
    assert (
        receipt["pair_local_quote_age_mask_sha256"]
        == receipt["schedule_coverage_receipt"]["pair_local_quote_age_mask_sha256"]
    )
    assert receipt["calendar_coverage_validated"] is True
    assert receipt["synthetic_quote_count"] == 0
    assert receipt["carry_forward_quote_count"] == 0
    assert receipt["authority"]["live_permission"] is False
    assert receipt["authority"]["order_authority"] == "NONE"

    sparse = source_slice.rows[5]
    assert sparse["observed_pairs"] == ["EUR_USD"]
    assert sparse["fresh_executable_pairs"] == ["EUR_USD"]
    assert sparse["unavailable_pairs"] == ["USD_JPY"]
    assert sparse["pair_local_quote_age_seconds"] == {
        "EUR_USD": 0,
        "USD_JPY": 60,
    }
    assert [quote["pair"] for quote in sparse["quotes"]] == ["EUR_USD"]
    expected_parent = canonical_source_sha256(
        {
            "pair": "EUR_USD",
            "epoch": sparse["epoch"],
            "granularity": "M1",
            "bid": m1_source.quote_maps["EUR_USD"][sparse["epoch"]]["bid"],
            "ask": m1_source.quote_maps["EUR_USD"][sparse["epoch"]]["ask"],
        }
    )
    assert sparse["quotes"][0]["parent_quote_sha256"] == expected_parent


def test_consumer_exposes_only_fresh_quotes_and_rejects_stale_execution(
    m1_source: SourceFixture,
) -> None:
    source_slice = _build_slice(m1_source)
    batches = consume_sparse_source_slice_v2(
        source_slice,
        expected_stream_id="sparse-stream-v2",
        expected_slice_id="slice-000",
        expected_sequence=0,
        expected_prior_receipt_sha256=GENESIS_SHA256,
        schedule=m1_source.schedule,
        pair_quote_maps=m1_source.quote_maps,
        parent_bindings=m1_source.parent_bindings,
    )

    sparse = batches[5]
    assert sparse.fresh_executable_pairs == ("EUR_USD",)
    assert tuple(pair for pair, _quote_row in sparse.fresh_observed_quotes) == (
        "EUR_USD",
    )
    assert sparse.executable_quote("EUR_USD")["pair"] == "EUR_USD"
    with pytest.raises(DojoSparseSourceSliceV2Error, match="no fresh executable"):
        sparse.executable_quote("USD_JPY")


def test_m5_uses_the_same_sparse_source_contract(m5_source: SourceFixture) -> None:
    source_slice = _build_slice(m5_source)
    verified = _verify(source_slice, m5_source)

    assert verified.receipt["granularity"] == "M5"
    assert verified.receipt["source_row_count"] == 24
    assert verified.receipt["observed_quote_count"] == 24 + 22
    assert verified.rows[4]["pair_local_quote_age_seconds"]["USD_JPY"] == 300
    assert verified.rows[5]["pair_local_quote_age_seconds"]["USD_JPY"] == 600
    assert verified.rows[6]["fresh_executable_pairs"] == list(PAIRS)


def test_parent_binding_recomputes_quote_map_and_raw_authentication_identity(
    m1_source: SourceFixture,
) -> None:
    binding = validate_parent_quote_map_binding(
        m1_source.parent_bindings["EUR_USD"],
        schedule=m1_source.schedule,
        pair="EUR_USD",
        quote_map=m1_source.quote_maps["EUR_USD"],
    )
    assert binding["upstream_raw_authentication_required"] is True
    assert binding["raw_bytes_opened_by_this_builder"] is False
    assert binding["observed_row_count"] == 60

    changed_maps = copy.deepcopy(m1_source.quote_maps)
    epoch = next(iter(changed_maps["EUR_USD"]))
    changed_maps["EUR_USD"][epoch]["bid"][3] += 0.0001
    with pytest.raises(
        DojoSparseSourceSliceV2Error, match="differs from supplied raw quote identity"
    ):
        validate_parent_quote_map_binding(
            binding,
            schedule=m1_source.schedule,
            pair="EUR_USD",
            quote_map=changed_maps["EUR_USD"],
        )


def test_builder_rejects_missing_quote_and_synthetic_metadata(
    m1_source: SourceFixture,
) -> None:
    missing = copy.deepcopy(m1_source.quote_maps)
    missing["EUR_USD"].pop(next(iter(missing["EUR_USD"])))
    with pytest.raises(DojoSparseSourceSliceV2Error, match="scheduled observed epochs"):
        build_sparse_source_slice_v2(
            stream_id="sparse-stream-v2",
            slice_id="slice-000",
            sequence=0,
            prior_receipt_sha256=GENESIS_SHA256,
            schedule=m1_source.schedule,
            pair_quote_maps=missing,
            parent_bindings=m1_source.parent_bindings,
        )

    synthetic = copy.deepcopy(m1_source.quote_maps)
    epoch = next(iter(synthetic["EUR_USD"]))
    synthetic["EUR_USD"][epoch]["synthetic"] = True
    with pytest.raises(DojoSparseSourceSliceV2Error, match="schema is not exact"):
        build_parent_quote_map_binding(
            schedule=m1_source.schedule,
            pair="EUR_USD",
            quote_map=synthetic["EUR_USD"],
            authentication={
                "raw_source_id": "raw:synthetic",
                "raw_artifact_sha256": _sha("raw:synthetic"),
                "upstream_authentication_receipt_sha256": _sha("auth:synthetic"),
            },
        )


def test_verifier_rejects_source_row_loss_quote_tamper_and_mask_tamper(
    m1_source: SourceFixture,
) -> None:
    source_slice = _build_slice(m1_source)
    missing_row = SparseSourceSliceV2(
        rows=source_slice.rows[:-1], receipt=source_slice.receipt
    )
    with pytest.raises(DojoSparseSourceSliceV2Error, match="parent truth"):
        _verify(missing_row, m1_source)

    changed_rows = copy.deepcopy(list(source_slice.rows))
    changed_rows[0]["quotes"][0]["bid"][3] += 0.0001
    quote_tamper = SparseSourceSliceV2(
        rows=tuple(changed_rows), receipt=source_slice.receipt
    )
    with pytest.raises(DojoSparseSourceSliceV2Error, match="parent truth"):
        _verify(quote_tamper, m1_source)

    changed_receipt = copy.deepcopy(dict(source_slice.receipt))
    changed_receipt["availability_mask_sha256"] = _sha("wrong-mask")
    body = {
        key: item for key, item in changed_receipt.items() if key != "receipt_sha256"
    }
    changed_receipt["receipt_sha256"] = canonical_source_sha256(body)
    mask_tamper = SparseSourceSliceV2(rows=source_slice.rows, receipt=changed_receipt)
    with pytest.raises(DojoSparseSourceSliceV2Error, match="parent truth"):
        _verify(mask_tamper, m1_source)


def test_verifier_requires_expected_lineage_and_rejects_alternate_prior(
    m1_source: SourceFixture,
) -> None:
    prior = _sha("prior")
    branch = _build_slice(
        m1_source,
        slice_id="slice-001",
        sequence=1,
        prior=prior,
    )
    with pytest.raises(DojoSparseSourceSliceV2Error, match="parent truth"):
        verify_sparse_source_slice_v2(
            branch,
            expected_stream_id="sparse-stream-v2",
            expected_slice_id="slice-001",
            expected_sequence=1,
            expected_prior_receipt_sha256=_sha("alternate-prior"),
            schedule=m1_source.schedule,
            pair_quote_maps=m1_source.quote_maps,
            parent_bindings=m1_source.parent_bindings,
        )


def test_receipt_chain_rejects_fork_gap_and_overlap(m1_source: SourceFixture) -> None:
    first = _build_slice(m1_source)
    second_fixture = _source_fixture(
        granularity="M1",
        start=m1_source.end,
    )
    second = _build_slice(
        second_fixture,
        slice_id="slice-001",
        sequence=1,
        prior=first.receipt["receipt_sha256"],
    )
    attestation = verify_sparse_source_receipt_chain_v2(
        [first.receipt, second.receipt],
        expected_stream_id="sparse-stream-v2",
    )
    assert attestation["supplied_chain_gap_free"] is True
    assert attestation["supplied_chain_fork_free"] is True
    assert attestation["global_fork_absence_proven"] is False

    fork = _build_slice(
        second_fixture,
        slice_id="slice-001-fork",
        sequence=1,
        prior=first.receipt["receipt_sha256"],
    )
    with pytest.raises(
        DojoSparseSourceSliceV2Error, match="gap, alternate prior, or fork"
    ):
        verify_sparse_source_receipt_chain_v2(
            [first.receipt, second.receipt, fork.receipt],
            expected_stream_id="sparse-stream-v2",
        )

    overlap = copy.deepcopy(dict(second.receipt))
    overlap["from_epoch"] = first.receipt["from_epoch"]
    overlap_body = {
        key: item for key, item in overlap.items() if key != "receipt_sha256"
    }
    overlap["receipt_sha256"] = canonical_source_sha256(overlap_body)
    with pytest.raises(DojoSparseSourceSliceV2Error, match="gap or overlap"):
        verify_sparse_source_receipt_chain_v2(
            [first.receipt, overlap],
            expected_stream_id="sparse-stream-v2",
        )
