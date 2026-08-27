from __future__ import annotations

import pytest

import london_session_chronology_v1 as chronology


SPEC = chronology.SessionSpec(
    name="LONDON_HIERARCHICAL_FIXTURE",
    rail_start_local_minute=9 * 60,
    rail_end_local_minute=9 * 60 + 50,
    decision_local_minute=9 * 60 + 55,
)


def test_winter_gmt_and_summer_bst_resolve_same_local_session_with_shifted_utc() -> None:
    winter = chronology.resolve_completed_m5_session(
        chronology.consecutive_utc_m5("2026-01-15T09:00:00Z", 13),
        "2026-01-15", SPEC,
    )
    summer = chronology.resolve_completed_m5_session(
        chronology.consecutive_utc_m5("2026-06-15T08:00:00Z", 13),
        "2026-06-15", SPEC,
    )
    assert winter["decision_bar_time_utc"] == "2026-01-15T09:55:00Z"
    assert winter["fill_next_m5_open_time_utc"] == "2026-01-15T10:00:00Z"
    assert winter["utc_offset_seconds"] == 0
    assert winter["timezone_name"] == "GMT"
    assert summer["decision_bar_time_utc"] == "2026-06-15T08:55:00Z"
    assert summer["fill_next_m5_open_time_utc"] == "2026-06-15T09:00:00Z"
    assert summer["utc_offset_seconds"] == 3600
    assert summer["timezone_name"] == "BST"
    assert winter["rail_latest_bar_relation"] == "RAIL_LE_T_MINUS_1"
    assert summer["fixed_utc_hour_used_as_edge_definition"] is False


def test_spring_and_fall_transition_are_derived_from_utc_without_local_ambiguity() -> None:
    spring_before = chronology.utc_to_london("2026-03-29T00:55:00.000000000Z")
    spring_after = chronology.utc_to_london("2026-03-29T01:00:00.000000000Z")
    assert spring_before.local_clock == "00:55:00"
    assert spring_before.utc_offset_seconds == 0
    assert spring_after.local_clock == "02:00:00"
    assert spring_after.utc_offset_seconds == 3600

    fall_before = chronology.utc_to_london("2026-10-25T00:55:00.000000000Z")
    fall_after = chronology.utc_to_london("2026-10-25T01:00:00.000000000Z")
    assert fall_before.local_clock == "01:55:00"
    assert fall_before.utc_offset_seconds == 3600
    assert fall_after.local_clock == "01:00:00"
    assert fall_after.utc_offset_seconds == 0
    assert fall_before.fold == 0
    assert fall_after.fold == 1


def test_local_or_offset_timestamp_input_is_rejected() -> None:
    for value in (
        "2026-10-25T01:30:00",
        "2026-10-25T01:30:00+01:00",
        "2026-10-25T01:30:00+00:00",
    ):
        with pytest.raises(chronology.ChronologyError, match="canonical UTC Z"):
            chronology.utc_to_london(value)


def test_missing_duplicate_or_reversed_m5_bars_fail_closed() -> None:
    good = chronology.consecutive_utc_m5("2026-06-15T08:00:00Z", 13)
    with pytest.raises(chronology.ChronologyError, match="missing"):
        chronology.resolve_completed_m5_session(good[:5] + good[6:], "2026-06-15", SPEC)
    duplicate = good[:5] + [dict(good[4])] + good[5:]
    with pytest.raises(chronology.ChronologyError, match="duplicate UTC"):
        chronology.resolve_completed_m5_session(duplicate, "2026-06-15", SPEC)
    reversed_rows = list(good)
    reversed_rows[4], reversed_rows[5] = reversed_rows[5], reversed_rows[4]
    with pytest.raises(chronology.ChronologyError, match="strictly increasing"):
        chronology.resolve_completed_m5_session(reversed_rows, "2026-06-15", SPEC)


def test_decision_close_cannot_retroactively_use_fill_or_later_bar() -> None:
    resolved = chronology.resolve_completed_m5_session(
        chronology.consecutive_utc_m5("2026-06-15T08:00:00Z", 13),
        "2026-06-15", SPEC,
    )
    decision = chronology.utc_epoch_seconds(resolved["decision_bar_time_utc"])
    fill = chronology.utc_epoch_seconds(resolved["fill_next_m5_open_time_utc"])
    assert all(chronology.utc_epoch_seconds(value) < decision
               for value in resolved["rail_bar_times_utc"])
    assert fill - decision == chronology.M5_SECONDS
    assert resolved["decision_close_available_at_utc"] == resolved[
        "fill_next_m5_open_time_utc"
    ]
