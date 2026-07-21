"""Pure sparse-candle scheduling for causal multi-pair DOJO replay.

OANDA does not promise that every instrument emits a candle at every open
market slot.  A causal replay therefore cannot require identical timestamp
sets and must never fill a missing pair with its previous quote.  This module
turns pair-local observed epoch sets into one deterministic union schedule and
then batches only the rows that were actually observed at each epoch.

The module deliberately performs no file I/O.  Source readers remain
responsible for authenticating raw bytes and for constructing the exact
``pair -> epoch -> observed row`` mappings passed here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Final, Generic, TypeVar

from quant_rabbit.dojo_market_calendar import (
    OANDA_FX_HOURS_POLICY,
    expected_oanda_fx_slots,
)


SPARSE_REPLAY_COVERAGE_CONTRACT: Final = "QR_DOJO_OANDA_SPARSE_REPLAY_COVERAGE_V1"
COORDINATE_SCHEDULE: Final = "SEALED_OBSERVED_PAIR_EPOCH_UNION"
QUOTE_POLICY: Final = "OBSERVED_ONLY_NO_SYNTHETIC_OR_CARRY_FORWARD_QUOTES"
SUPPORTED_GRANULARITY_SECONDS: Final = {"M1": 60, "M5": 300}
INTRABAR_PHASE_COUNT: Final = 4

_T = TypeVar("_T")


class SparseReplayError(ValueError):
    """Raised when sparse replay cannot preserve its causal source contract."""


@dataclass(frozen=True)
class SparseCoveragePolicy:
    """Fixed acceptance bounds inherited from the DOJO M1 trainer.

    ``full_day_reference_m1_slots`` scales with candle duration, preserving the
    existing 1,000-slot M1 classification and giving M5 the equivalent
    200-slot threshold.  ``max_open_slot_gap_seconds`` counts expected open
    candle slots, so a normal weekend closure is not mistaken for missing data.
    """

    full_day_coverage_floor: float = 0.98
    partial_day_coverage_floor: float = 0.80
    full_day_reference_m1_slots: int = 1_000
    max_open_slot_gap_seconds: int = 900
    minimum_rows_per_pair: int = 2

    def validate(self) -> None:
        for name, value in (
            ("full_day_coverage_floor", self.full_day_coverage_floor),
            ("partial_day_coverage_floor", self.partial_day_coverage_floor),
        ):
            if (
                value.__class__ is not float
                or not math.isfinite(value)
                or not 0.0 < value <= 1.0
            ):
                raise SparseReplayError(f"{name} must be a finite float in (0, 1]")
        for name, value in (
            ("full_day_reference_m1_slots", self.full_day_reference_m1_slots),
            ("max_open_slot_gap_seconds", self.max_open_slot_gap_seconds),
            ("minimum_rows_per_pair", self.minimum_rows_per_pair),
        ):
            if value.__class__ is not int or value <= 0:
                raise SparseReplayError(f"{name} must be a positive integer")

    def full_day_minimum_expected_slots(self, cadence_seconds: int) -> int:
        return math.ceil(
            self.full_day_reference_m1_slots * 60 / cadence_seconds
        )


DEFAULT_SPARSE_COVERAGE_POLICY: Final = SparseCoveragePolicy()


@dataclass(frozen=True)
class SparseDailyCoverage:
    utc_date: str
    partial_day: bool
    partial_window_day: bool
    expected_slot_count: int
    coverage_floor: float
    pair_row_counts: tuple[tuple[str, int], ...]
    pair_coverage_ratios: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "utc_date": self.utc_date,
            "partial_day": self.partial_day,
            "partial_window_day": self.partial_window_day,
            "expected_slot_count": self.expected_slot_count,
            "coverage_floor": self.coverage_floor,
            "pair_row_counts": dict(self.pair_row_counts),
            "pair_coverage_ratios": dict(self.pair_coverage_ratios),
        }


@dataclass(frozen=True)
class SparseEpochAvailability:
    """One union coordinate without any invented quote payload."""

    epoch: int
    observed_pairs: tuple[str, ...]
    unavailable_pairs: tuple[str, ...]
    pair_local_quote_age_seconds: tuple[tuple[str, int | None], ...]

    def availability_dict(self) -> dict[str, Any]:
        """Return the compatibility mask hashed by the existing M1 trainer."""

        return {"epoch": self.epoch, "batch_pairs": list(self.observed_pairs)}

    def age_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "pair_local_quote_age_seconds": dict(
                self.pair_local_quote_age_seconds
            ),
        }


@dataclass(frozen=True)
class SparseReplaySchedule:
    """Validated immutable schedule over the union of observed pair epochs."""

    granularity: str
    cadence_seconds: int
    start_epoch: int
    end_epoch: int
    feed_pairs: tuple[str, ...]
    expected_calendar_epochs: tuple[int, ...]
    observed_epochs_by_pair: tuple[tuple[str, tuple[int, ...]], ...]
    availability: tuple[SparseEpochAvailability, ...]
    daily_coverage: tuple[SparseDailyCoverage, ...]
    policy: SparseCoveragePolicy

    def epochs_for_pair(self, pair: str) -> tuple[int, ...]:
        for candidate, epochs in self.observed_epochs_by_pair:
            if candidate == pair:
                return epochs
        raise SparseReplayError(f"pair is outside the sealed feed: {pair}")

    def coverage_receipt(self) -> dict[str, Any]:
        availability_rows = [row.availability_dict() for row in self.availability]
        age_rows = [row.age_dict() for row in self.availability]
        partial_count = sum(
            row.observed_pairs != self.feed_pairs for row in self.availability
        )
        pair_row_counts = {
            pair: len(epochs) for pair, epochs in self.observed_epochs_by_pair
        }
        observed_ages = [
            age
            for row in self.availability
            for _pair, age in row.pair_local_quote_age_seconds
            if age is not None
        ]
        body = {
            "contract": SPARSE_REPLAY_COVERAGE_CONTRACT,
            "calendar_policy": OANDA_FX_HOURS_POLICY,
            "granularity": self.granularity,
            "cadence_seconds": self.cadence_seconds,
            "coordinate_schedule": COORDINATE_SCHEDULE,
            "quote_policy": QUOTE_POLICY,
            "feed_pairs": list(self.feed_pairs),
            "from_epoch": self.start_epoch,
            "to_epoch": self.end_epoch,
            "expected_calendar_slot_count": len(self.expected_calendar_epochs),
            "expected_union_epoch_count": len(self.availability),
            "expected_full_epoch_count": len(self.availability) - partial_count,
            "expected_partial_epoch_count": partial_count,
            "expected_phase_mark_count": len(self.availability)
            * INTRABAR_PHASE_COUNT,
            "expected_partial_phase_count": partial_count
            * INTRABAR_PHASE_COUNT,
            "pair_row_counts": pair_row_counts,
            "availability_mask_sha256": canonical_sparse_sha256(
                availability_rows
            ),
            "pair_local_quote_age_mask_sha256": canonical_sparse_sha256(age_rows),
            "expected_observed_quote_count": sum(pair_row_counts.values())
            * INTRABAR_PHASE_COUNT,
            "maximum_pair_local_quote_age_seconds": max(
                observed_ages, default=0
            ),
            "maximum_open_slot_gap_seconds": self.policy.max_open_slot_gap_seconds,
            "synthetic_quote_count": 0,
            "carry_forward_quote_count": 0,
            "full_day_coverage_floor": self.policy.full_day_coverage_floor,
            "partial_day_coverage_floor": self.policy.partial_day_coverage_floor,
            "full_day_minimum_expected_slots": (
                self.policy.full_day_minimum_expected_slots(self.cadence_seconds)
            ),
            "daily_coverage": [row.as_dict() for row in self.daily_coverage],
            "first_epoch": self.availability[0].epoch,
            "last_epoch": self.availability[-1].epoch,
        }
        return {**body, "coverage_sha256": canonical_sparse_sha256(body)}


@dataclass(frozen=True)
class SparseObservedEpochBatch(Generic[_T]):
    """All and only fresh observed rows available at one source epoch."""

    epoch: int
    observations: tuple[tuple[str, _T], ...]
    unavailable_pairs: tuple[str, ...]
    pair_local_quote_age_seconds: tuple[tuple[str, int | None], ...]
    quote_origin: str = "OBSERVED_INPUT_ROW"
    synthetic_quote_count: int = 0
    carry_forward_quote_count: int = 0

    @property
    def observed_pairs(self) -> tuple[str, ...]:
        return tuple(pair for pair, _row in self.observations)


def canonical_sparse_sha256(value: Any) -> str:
    """Hash one JSON-compatible sparse-replay value canonically."""

    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SparseReplayError("sparse replay value is not canonical JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def build_sparse_replay_schedule(
    pair_epochs: Mapping[str, Iterable[int]],
    *,
    feed_pairs: Sequence[str],
    start: datetime,
    end: datetime,
    granularity: str,
    policy: SparseCoveragePolicy = DEFAULT_SPARSE_COVERAGE_POLICY,
) -> SparseReplaySchedule:
    """Validate pair-local coverage and seal the observed epoch union.

    The function never inserts an epoch into a pair's observed set.  An epoch
    is scheduled when at least one configured pair has a real row, and its
    availability mask lists all configured pairs that have a row at that exact
    epoch in feed order.
    """

    cadence_seconds = _granularity_seconds(granularity)
    policy.validate()
    pairs = _feed_pairs(feed_pairs)
    if set(pair_epochs) != set(pairs):
        raise SparseReplayError(
            "pair epoch mappings must exactly equal the configured feed pairs"
        )
    start_utc = _aligned_utc(start, cadence_seconds, field="start")
    end_utc = _aligned_utc(end, cadence_seconds, field="end")
    if start_utc >= end_utc:
        raise SparseReplayError("sparse replay window must be positive")
    start_epoch = int(start_utc.timestamp())
    end_epoch = int(end_utc.timestamp())
    expected_slots = expected_oanda_fx_slots(
        start_utc,
        end_utc,
        step=timedelta(seconds=cadence_seconds),
    )
    expected_epochs = tuple(int(stamp.timestamp()) for stamp in expected_slots)
    if not expected_epochs:
        raise SparseReplayError("replay window has no OANDA open-market slots")
    expected_set = set(expected_epochs)

    normalized: dict[str, tuple[int, ...]] = {}
    for pair in pairs:
        raw_epochs = list(pair_epochs[pair])
        for epoch in raw_epochs:
            if epoch.__class__ is not int:
                raise SparseReplayError(f"observed epoch for {pair} must be an integer")
            if epoch % cadence_seconds:
                raise SparseReplayError(
                    f"observed epoch for {pair} is not {granularity}-aligned: {epoch}"
                )
            if not start_epoch <= epoch < end_epoch:
                raise SparseReplayError(
                    f"observed epoch for {pair} is outside the replay window: {epoch}"
                )
        if len(raw_epochs) != len(set(raw_epochs)):
            raise SparseReplayError(f"duplicate observed epoch for {pair}")
        epochs = tuple(sorted(raw_epochs))
        outside_calendar = [epoch for epoch in epochs if epoch not in expected_set]
        if outside_calendar:
            raise SparseReplayError(
                f"observed epoch for {pair} is outside the OANDA open calendar: "
                f"{outside_calendar[0]}"
            )
        normalized[pair] = epochs

    daily = _daily_coverage(
        normalized,
        pairs=pairs,
        expected_epochs=expected_epochs,
        start=start_utc,
        end=end_utc,
        cadence_seconds=cadence_seconds,
        policy=policy,
    )
    _validate_open_slot_gaps(
        normalized,
        pairs=pairs,
        expected_epochs=expected_epochs,
        cadence_seconds=cadence_seconds,
        policy=policy,
    )

    observed_sets = {pair: set(normalized[pair]) for pair in pairs}
    union_epochs = sorted(set().union(*observed_sets.values()))
    if not union_epochs:
        raise SparseReplayError("sparse replay union is empty")
    last_seen: dict[str, int | None] = {pair: None for pair in pairs}
    availability: list[SparseEpochAvailability] = []
    for epoch in union_epochs:
        observed = tuple(pair for pair in pairs if epoch in observed_sets[pair])
        unavailable = tuple(pair for pair in pairs if pair not in observed)
        ages: list[tuple[str, int | None]] = []
        for pair in pairs:
            if pair in observed:
                last_seen[pair] = epoch
                age: int | None = 0
            else:
                prior = last_seen[pair]
                age = None if prior is None else epoch - prior
            ages.append((pair, age))
        availability.append(
            SparseEpochAvailability(
                epoch=epoch,
                observed_pairs=observed,
                unavailable_pairs=unavailable,
                pair_local_quote_age_seconds=tuple(ages),
            )
        )

    return SparseReplaySchedule(
        granularity=granularity,
        cadence_seconds=cadence_seconds,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        feed_pairs=pairs,
        expected_calendar_epochs=expected_epochs,
        observed_epochs_by_pair=tuple(
            (pair, normalized[pair]) for pair in pairs
        ),
        availability=tuple(availability),
        daily_coverage=daily,
        policy=policy,
    )


def build_sparse_replay_schedule_from_rows(
    pair_rows: Mapping[str, Mapping[int, _T]],
    *,
    feed_pairs: Sequence[str],
    start: datetime,
    end: datetime,
    granularity: str,
    policy: SparseCoveragePolicy = DEFAULT_SPARSE_COVERAGE_POLICY,
) -> SparseReplaySchedule:
    """Integration helper for authenticated ``pair -> epoch -> row`` maps."""

    return build_sparse_replay_schedule(
        {pair: rows.keys() for pair, rows in pair_rows.items()},
        feed_pairs=feed_pairs,
        start=start,
        end=end,
        granularity=granularity,
        policy=policy,
    )


def iter_observed_epoch_batches(
    pair_rows: Mapping[str, Mapping[int, _T]],
    *,
    schedule: SparseReplaySchedule,
) -> Iterator[SparseObservedEpochBatch[_T]]:
    """Yield exact same-epoch observed batches without quote imputation.

    Every supplied row must be part of the sealed schedule.  Unavailable pairs
    receive age metadata but no payload, so consumers cannot accidentally use a
    carried quote as an executable fill/entry/exit quote.
    """

    if set(pair_rows) != set(schedule.feed_pairs):
        raise SparseReplayError(
            "pair row mappings must exactly equal the sealed feed pairs"
        )
    for pair in schedule.feed_pairs:
        rows = pair_rows[pair]
        if not isinstance(rows, Mapping):
            raise SparseReplayError(f"pair rows for {pair} must be an epoch mapping")
        supplied = set(rows)
        expected = set(schedule.epochs_for_pair(pair))
        if supplied != expected:
            raise SparseReplayError(
                f"pair rows for {pair} differ from the sealed observed epochs"
            )

    for coordinate in schedule.availability:
        observations = tuple(
            (pair, pair_rows[pair][coordinate.epoch])
            for pair in coordinate.observed_pairs
        )
        if not observations:
            raise SparseReplayError("sealed union coordinate has no observed rows")
        yield SparseObservedEpochBatch(
            epoch=coordinate.epoch,
            observations=observations,
            unavailable_pairs=coordinate.unavailable_pairs,
            pair_local_quote_age_seconds=coordinate.pair_local_quote_age_seconds,
        )


def _daily_coverage(
    observed: Mapping[str, tuple[int, ...]],
    *,
    pairs: tuple[str, ...],
    expected_epochs: tuple[int, ...],
    start: datetime,
    end: datetime,
    cadence_seconds: int,
    policy: SparseCoveragePolicy,
) -> tuple[SparseDailyCoverage, ...]:
    expected_by_day: dict[str, list[int]] = {}
    for epoch in expected_epochs:
        day = datetime.fromtimestamp(epoch, tz=timezone.utc).date().isoformat()
        expected_by_day.setdefault(day, []).append(epoch)
    observed_sets = {pair: set(observed[pair]) for pair in pairs}
    window_start_day = start.date()
    window_end_day = (end - timedelta(microseconds=1)).date()
    full_day_minimum = policy.full_day_minimum_expected_slots(cadence_seconds)
    receipts: list[SparseDailyCoverage] = []
    for day, day_epochs in sorted(expected_by_day.items()):
        day_value = datetime.fromisoformat(day).date()
        day_start = datetime.combine(day_value, time.min, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        partial_window_day = (day_value == window_start_day and start > day_start) or (
            day_value == window_end_day and end < day_end
        )
        partial_day = partial_window_day or len(day_epochs) < full_day_minimum
        floor = (
            policy.partial_day_coverage_floor
            if partial_day
            else policy.full_day_coverage_floor
        )
        counts: list[tuple[str, int]] = []
        ratios: list[tuple[str, float]] = []
        for pair in pairs:
            count = len(observed_sets[pair].intersection(day_epochs))
            ratio = count / len(day_epochs)
            counts.append((pair, count))
            ratios.append((pair, ratio))
            if ratio < floor:
                raise SparseReplayError(
                    f"OANDA sparse coverage below {floor:.0%} for {pair} "
                    f"on {day}: {count}/{len(day_epochs)}"
                )
        receipts.append(
            SparseDailyCoverage(
                utc_date=day,
                partial_day=partial_day,
                partial_window_day=partial_window_day,
                expected_slot_count=len(day_epochs),
                coverage_floor=floor,
                pair_row_counts=tuple(counts),
                pair_coverage_ratios=tuple(ratios),
            )
        )
    return tuple(receipts)


def _validate_open_slot_gaps(
    observed: Mapping[str, tuple[int, ...]],
    *,
    pairs: tuple[str, ...],
    expected_epochs: tuple[int, ...],
    cadence_seconds: int,
    policy: SparseCoveragePolicy,
) -> None:
    expected_index = {epoch: index for index, epoch in enumerate(expected_epochs)}
    for pair in pairs:
        epochs = observed[pair]
        if len(epochs) < policy.minimum_rows_per_pair:
            raise SparseReplayError(
                f"replay period requires at least {policy.minimum_rows_per_pair} "
                f"observed rows for {pair}"
            )
        indices = [expected_index[epoch] for epoch in epochs]
        boundary_missing_slots = max(
            indices[0], len(expected_epochs) - indices[-1] - 1
        )
        maximum_observation_age_slots = max(
            right - left for left, right in zip(indices, indices[1:], strict=False)
        )
        open_slot_gap_seconds = max(
            boundary_missing_slots * cadence_seconds,
            maximum_observation_age_slots * cadence_seconds,
        )
        if open_slot_gap_seconds > policy.max_open_slot_gap_seconds:
            raise SparseReplayError(
                "OANDA sparse boundary/open-slot gap exceeds "
                f"{policy.max_open_slot_gap_seconds}s for {pair}"
            )


def _granularity_seconds(granularity: str) -> int:
    if granularity not in SUPPORTED_GRANULARITY_SECONDS:
        raise SparseReplayError("granularity must be exactly M1 or M5")
    return SUPPORTED_GRANULARITY_SECONDS[granularity]


def _feed_pairs(feed_pairs: Sequence[str]) -> tuple[str, ...]:
    if isinstance(feed_pairs, (str, bytes, bytearray)):
        raise SparseReplayError("feed pairs must be a sequence of pair names")
    pairs = tuple(feed_pairs)
    if (
        not pairs
        or len(pairs) != len(set(pairs))
        or any(not isinstance(pair, str) or not pair for pair in pairs)
    ):
        raise SparseReplayError("feed pairs must be non-empty unique strings")
    return pairs


def _aligned_utc(value: datetime, cadence_seconds: int, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SparseReplayError(f"{field} must be a timezone-aware datetime")
    normalized = value.astimezone(timezone.utc)
    if (
        normalized.second
        or normalized.microsecond
        or int(normalized.timestamp()) % cadence_seconds
    ):
        raise SparseReplayError(f"{field} is not aligned to the candle grid")
    return normalized


__all__ = [
    "COORDINATE_SCHEDULE",
    "DEFAULT_SPARSE_COVERAGE_POLICY",
    "INTRABAR_PHASE_COUNT",
    "QUOTE_POLICY",
    "SPARSE_REPLAY_COVERAGE_CONTRACT",
    "SUPPORTED_GRANULARITY_SECONDS",
    "SparseCoveragePolicy",
    "SparseDailyCoverage",
    "SparseEpochAvailability",
    "SparseObservedEpochBatch",
    "SparseReplayError",
    "SparseReplaySchedule",
    "build_sparse_replay_schedule",
    "build_sparse_replay_schedule_from_rows",
    "canonical_sparse_sha256",
    "iter_observed_epoch_batches",
]
