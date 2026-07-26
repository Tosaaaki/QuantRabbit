from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run-dojo-autonomous-improvement.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_dojo_autonomous_improvement",
    SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load script: {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_assessment_gate_accepts_open_market() -> None:
    open_time = datetime(2026, 7, 22, 18, 0, tzinfo=timezone.utc)
    MODULE._assert_assessment_market_open(
        {"as_of_utc": open_time.isoformat()},
        recorded_at_utc=open_time,
    )


@pytest.mark.parametrize(
    ("as_of_utc", "recorded_at_utc"),
    [
        (
            datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 7, 24, 20, 59, tzinfo=timezone.utc),
            datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 7, 24, 20, 59, tzinfo=timezone.utc),
        ),
    ],
)
def test_assessment_gate_rejects_closed_as_of_or_record_time(
    as_of_utc: datetime,
    recorded_at_utc: datetime,
) -> None:
    with pytest.raises(ValueError, match="AI_ASSESSMENT_MARKET_CLOSED"):
        MODULE._assert_assessment_market_open(
            {"as_of_utc": as_of_utc.isoformat()},
            recorded_at_utc=recorded_at_utc,
        )


def test_assessment_gate_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone"):
        MODULE._assert_assessment_market_open(
            {"as_of_utc": "2026-07-22T18:00:00"},
            recorded_at_utc=datetime(
                2026,
                7,
                22,
                18,
                0,
                tzinfo=timezone.utc,
            ),
        )
