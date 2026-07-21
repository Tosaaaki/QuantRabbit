from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from quant_rabbit.dojo_market_calendar import expected_oanda_fx_slots
from quant_rabbit.dojo_sparse_replay import (
    QUOTE_POLICY,
    SparseReplayError,
    build_sparse_replay_schedule,
    build_sparse_replay_schedule_from_rows,
    canonical_sparse_sha256,
    iter_observed_epoch_batches,
)


UTC = timezone.utc
PAIRS = ("EUR_USD", "USD_JPY")


@dataclass(frozen=True)
class SparseRowsFixture:
    start: datetime
    end: datetime
    epochs: tuple[int, ...]
    pair_rows: dict[str, dict[int, object]]


@pytest.fixture
def sparse_m1_rows() -> SparseRowsFixture:
    start = datetime(2026, 1, 6, 12, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    epochs = tuple(
        int(stamp.timestamp())
        for stamp in expected_oanda_fx_slots(
            start, end, step=timedelta(minutes=1)
        )
    )
    assert len(epochs) == 60
    eur_rows = {
        epoch: {"pair": "EUR_USD", "epoch": epoch, "identity": object()}
        for epoch in epochs
    }
    usd_rows = {
        epoch: {"pair": "USD_JPY", "epoch": epoch, "identity": object()}
        for index, epoch in enumerate(epochs)
        if index != 5
    }
    return SparseRowsFixture(
        start=start,
        end=end,
        epochs=epochs,
        pair_rows={"EUR_USD": eur_rows, "USD_JPY": usd_rows},
    )


@pytest.fixture
def sparse_m5_rows() -> SparseRowsFixture:
    start = datetime(2026, 1, 6, 12, 0, tzinfo=UTC)
    end = start + timedelta(hours=2)
    epochs = tuple(
        int(stamp.timestamp())
        for stamp in expected_oanda_fx_slots(
            start, end, step=timedelta(minutes=5)
        )
    )
    assert len(epochs) == 24
    return SparseRowsFixture(
        start=start,
        end=end,
        epochs=epochs,
        pair_rows={
            pair: {
                epoch: {"pair": pair, "epoch": epoch}
                for epoch in epochs
            }
            for pair in PAIRS
        },
    )


def test_m1_schedule_preserves_existing_union_availability_contract(
    sparse_m1_rows: SparseRowsFixture,
) -> None:
    fixture = sparse_m1_rows
    schedule = build_sparse_replay_schedule_from_rows(
        fixture.pair_rows,
        feed_pairs=PAIRS,
        start=fixture.start,
        end=fixture.end,
        granularity="M1",
    )
    receipt = schedule.coverage_receipt()

    assert receipt["granularity"] == "M1"
    assert receipt["cadence_seconds"] == 60
    assert receipt["quote_policy"] == QUOTE_POLICY
    assert receipt["expected_calendar_slot_count"] == 60
    assert receipt["expected_union_epoch_count"] == 60
    assert receipt["expected_partial_epoch_count"] == 1
    assert receipt["expected_partial_phase_count"] == 4
    assert receipt["expected_observed_quote_count"] == (60 + 59) * 4
    assert receipt["pair_row_counts"] == {"EUR_USD": 60, "USD_JPY": 59}
    assert receipt["synthetic_quote_count"] == 0
    assert receipt["carry_forward_quote_count"] == 0
    assert receipt["coverage_sha256"] == canonical_sparse_sha256(
        {key: value for key, value in receipt.items() if key != "coverage_sha256"}
    )

    availability_rows = [row.availability_dict() for row in schedule.availability]
    assert receipt["availability_mask_sha256"] == canonical_sparse_sha256(
        availability_rows
    )
    sparse = schedule.availability[5]
    assert sparse.epoch == fixture.epochs[5]
    assert sparse.observed_pairs == ("EUR_USD",)
    assert sparse.unavailable_pairs == ("USD_JPY",)
    assert dict(sparse.pair_local_quote_age_seconds) == {
        "EUR_USD": 0,
        "USD_JPY": 60,
    }


def test_batcher_groups_every_fresh_pair_at_epoch_and_never_carries_missing_row(
    sparse_m1_rows: SparseRowsFixture,
) -> None:
    fixture = sparse_m1_rows
    schedule = build_sparse_replay_schedule_from_rows(
        fixture.pair_rows,
        feed_pairs=PAIRS,
        start=fixture.start,
        end=fixture.end,
        granularity="M1",
    )
    batches = list(
        iter_observed_epoch_batches(fixture.pair_rows, schedule=schedule)
    )

    assert len(batches) == 60
    assert batches[4].observed_pairs == PAIRS
    assert batches[5].observed_pairs == ("EUR_USD",)
    assert batches[5].unavailable_pairs == ("USD_JPY",)
    assert batches[5].quote_origin == "OBSERVED_INPUT_ROW"
    assert batches[5].synthetic_quote_count == 0
    assert batches[5].carry_forward_quote_count == 0
    assert all(pair != "USD_JPY" for pair, _row in batches[5].observations)
    assert batches[6].observed_pairs == PAIRS
    for batch in batches:
        assert batch.observed_pairs == tuple(
            pair for pair in PAIRS if batch.epoch in fixture.pair_rows[pair]
        )
        for pair, row in batch.observations:
            assert row is fixture.pair_rows[pair][batch.epoch]


def test_pair_local_age_is_none_until_first_real_observation() -> None:
    start = datetime(2026, 1, 6, 12, 0, tzinfo=UTC)
    end = start + timedelta(minutes=10)
    epochs = tuple(
        int(stamp.timestamp())
        for stamp in expected_oanda_fx_slots(
            start, end, step=timedelta(minutes=1)
        )
    )
    schedule = build_sparse_replay_schedule(
        {
            "EUR_USD": epochs,
            "USD_JPY": epochs[1:],
        },
        feed_pairs=PAIRS,
        start=start,
        end=end,
        granularity="M1",
    )

    first = schedule.availability[0]
    assert first.observed_pairs == ("EUR_USD",)
    assert dict(first.pair_local_quote_age_seconds) == {
        "EUR_USD": 0,
        "USD_JPY": None,
    }
    assert schedule.availability[1].observed_pairs == PAIRS


def test_m5_uses_same_calendar_and_sparse_union_semantics(
    sparse_m5_rows: SparseRowsFixture,
) -> None:
    fixture = sparse_m5_rows
    del fixture.pair_rows["USD_JPY"][fixture.epochs[4]]
    del fixture.pair_rows["USD_JPY"][fixture.epochs[5]]
    schedule = build_sparse_replay_schedule_from_rows(
        fixture.pair_rows,
        feed_pairs=PAIRS,
        start=fixture.start,
        end=fixture.end,
        granularity="M5",
    )
    receipt = schedule.coverage_receipt()

    assert receipt["granularity"] == "M5"
    assert receipt["cadence_seconds"] == 300
    assert receipt["full_day_minimum_expected_slots"] == 200
    assert receipt["expected_union_epoch_count"] == 24
    assert receipt["expected_partial_epoch_count"] == 2
    assert schedule.availability[4].observed_pairs == ("EUR_USD",)
    assert dict(schedule.availability[4].pair_local_quote_age_seconds)[
        "USD_JPY"
    ] == 300
    assert dict(schedule.availability[5].pair_local_quote_age_seconds)[
        "USD_JPY"
    ] == 600
    assert schedule.availability[6].observed_pairs == PAIRS


def test_m5_rejects_open_slot_gap_larger_than_fixed_bound(
    sparse_m5_rows: SparseRowsFixture,
) -> None:
    fixture = sparse_m5_rows
    for index in (4, 5, 6):
        del fixture.pair_rows["USD_JPY"][fixture.epochs[index]]

    with pytest.raises(SparseReplayError, match="open-slot gap exceeds 900s"):
        build_sparse_replay_schedule_from_rows(
            fixture.pair_rows,
            feed_pairs=PAIRS,
            start=fixture.start,
            end=fixture.end,
            granularity="M5",
        )


def test_full_m1_day_keeps_ninety_eight_percent_floor() -> None:
    start = datetime(2026, 1, 6, 0, 0, tzinfo=UTC)
    end = start + timedelta(days=1)
    epochs = tuple(
        int(stamp.timestamp())
        for stamp in expected_oanda_fx_slots(
            start, end, step=timedelta(minutes=1)
        )
    )
    passing = tuple(epoch for index, epoch in enumerate(epochs) if index % 60)
    schedule = build_sparse_replay_schedule(
        {"EUR_USD": passing},
        feed_pairs=("EUR_USD",),
        start=start,
        end=end,
        granularity="M1",
    )
    day = schedule.coverage_receipt()["daily_coverage"][0]
    assert day["partial_day"] is False
    assert day["coverage_floor"] == 0.98

    failing = tuple(epoch for index, epoch in enumerate(epochs) if index % 50)
    with pytest.raises(SparseReplayError, match="coverage below 98%"):
        build_sparse_replay_schedule(
            {"EUR_USD": failing},
            feed_pairs=("EUR_USD",),
            start=start,
            end=end,
            granularity="M1",
        )


def test_rejects_closed_calendar_rows_duplicates_and_unsupported_granularity(
    sparse_m1_rows: SparseRowsFixture,
) -> None:
    fixture = sparse_m1_rows
    duplicate = list(fixture.epochs)
    duplicate.append(fixture.epochs[0])
    with pytest.raises(SparseReplayError, match="duplicate observed epoch"):
        build_sparse_replay_schedule(
            {"EUR_USD": duplicate},
            feed_pairs=("EUR_USD",),
            start=fixture.start,
            end=fixture.end,
            granularity="M1",
        )

    with pytest.raises(SparseReplayError, match="exactly M1 or M5"):
        build_sparse_replay_schedule(
            {"EUR_USD": fixture.epochs},
            feed_pairs=("EUR_USD",),
            start=fixture.start,
            end=fixture.end,
            granularity="H1",
        )

    friday_start = datetime(2026, 1, 9, 20, 0, tzinfo=UTC)
    saturday_end = datetime(2026, 1, 10, 22, 0, tzinfo=UTC)
    valid = tuple(
        int(stamp.timestamp())
        for stamp in expected_oanda_fx_slots(
            friday_start, saturday_end, step=timedelta(minutes=5)
        )
    )
    saturday_epoch = int(datetime(2026, 1, 10, 12, 0, tzinfo=UTC).timestamp())
    with pytest.raises(SparseReplayError, match="outside the OANDA open calendar"):
        build_sparse_replay_schedule(
            {"EUR_USD": (*valid, saturday_epoch)},
            feed_pairs=("EUR_USD",),
            start=friday_start,
            end=saturday_end,
            granularity="M5",
        )


def test_batcher_rejects_rows_that_drift_from_sealed_observed_epochs(
    sparse_m1_rows: SparseRowsFixture,
) -> None:
    fixture = sparse_m1_rows
    schedule = build_sparse_replay_schedule_from_rows(
        fixture.pair_rows,
        feed_pairs=PAIRS,
        start=fixture.start,
        end=fixture.end,
        granularity="M1",
    )
    drifted = {
        pair: dict(rows) for pair, rows in fixture.pair_rows.items()
    }
    del drifted["EUR_USD"][fixture.epochs[0]]

    with pytest.raises(SparseReplayError, match="differ from the sealed"):
        list(iter_observed_epoch_batches(drifted, schedule=schedule))
