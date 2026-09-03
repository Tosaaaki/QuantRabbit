from __future__ import annotations

from quant_rabbit.fast_bot_entry_edge import (
    build_entry_edge_snapshot,
    entry_edge_snapshot_valid,
)


def _vote(
    direction: str,
    *,
    phase: str = "PRE_TREND",
    readiness: str = "TRIGGERED",
    location: str = "MIDDLE_THIRD",
    value_zone: str = "FAIR_VALUE",
    extension: str = "BALANCED",
) -> dict:
    return {
        "observed_direction": direction,
        "direction_score": 1,
        "phase": phase,
        "readiness": readiness,
        "trigger": "BREAKOUT_CLOSE",
        "structure": "BREAKOUT_ACTIVE",
        "location": location,
        "value_zone": value_zone,
        "extension": extension,
        "evidence_complete": True,
    }


def _row(method: str = "TREND_CONTINUATION", side: str = "LONG") -> dict:
    direction = "UP" if side == "LONG" else "DOWN"
    return {
        "method": method,
        "side": side,
        "failed_break_direction": side if method == "BREAKOUT_FAILURE" else None,
        "timeframe_votes": {
            timeframe: _vote(direction)
            for timeframe in ("M1", "M5", "M15", "M30", "H1", "H4", "D")
        },
    }


def test_trend_requires_direction_trigger_room_and_economics() -> None:
    accepted = build_entry_edge_snapshot(
        _row(),
        reward_risk=1.3,
        spread_to_m5_atr=0.1,
    )
    assert accepted["accepted"] is True
    assert accepted["blockers"] == []
    assert accepted["lookahead_used"] is False
    assert accepted["outcome_fields_used"] == []
    assert entry_edge_snapshot_valid(accepted)

    exhausted = _row(side="SHORT")
    exhausted["timeframe_votes"]["M5"].update(
        extension="OVERSOLD",
        value_zone="DEEP_DISCOUNT",
    )
    rejected = build_entry_edge_snapshot(
        exhausted,
        reward_risk=1.3,
        spread_to_m5_atr=0.1,
    )
    assert rejected["accepted"] is False
    assert "FAST_TREND_ALREADY_EXHAUSTED" in rejected["blockers"]


def test_range_rotation_waits_for_reversal_instead_of_catching_fall() -> None:
    row = _row(method="RANGE_ROTATION", side="LONG")
    for timeframe in ("M5", "M15"):
        row["timeframe_votes"][timeframe].update(
            phase="RANGE",
            location="LOWER_THIRD",
            value_zone="DISCOUNT",
        )
    accepted = build_entry_edge_snapshot(
        row,
        reward_risk=1.0,
        spread_to_m5_atr=0.2,
    )
    assert accepted["accepted"] is True

    row["timeframe_votes"]["M1"]["observed_direction"] = "DOWN"
    rejected = build_entry_edge_snapshot(
        row,
        reward_risk=1.0,
        spread_to_m5_atr=0.2,
    )
    assert rejected["accepted"] is False
    assert "M1_RANGE_REVERSAL_DIRECTION_NOT_CONFIRMED" in rejected["blockers"]


def test_snapshot_tampering_fails_closed() -> None:
    snapshot = build_entry_edge_snapshot(
        _row(),
        reward_risk=1.3,
        spread_to_m5_atr=0.1,
    )
    snapshot["accepted"] = False
    assert entry_edge_snapshot_valid(snapshot) is False
