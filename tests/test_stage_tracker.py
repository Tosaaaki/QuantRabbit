import pathlib
import sys
from datetime import datetime, timedelta, timezone

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.stage_tracker import StageTracker


def test_stage_tracker_cooldown_with_timezone_aware_now(tmp_path):
    db_path = tmp_path / "stage.db"
    tracker = StageTracker(db_path=db_path)
    try:
        now_aware = datetime.now(timezone.utc)
        tracker.set_cooldown(
            "macro",
            "long",
            reason="test",
            seconds=120,
            now=now_aware,
        )

        before_expiry = now_aware + timedelta(seconds=30)
        cooldown = tracker.get_cooldown("macro", "long", now=before_expiry)
        assert cooldown is not None
        assert cooldown.cooldown_until.tzinfo is None

        # Calling without specifying now should not raise even though stored value included tz info.
        # (internally StageTracker uses datetime.utcnow()).
        cooldown_again = tracker.get_cooldown("macro", "long")
        assert cooldown_again is not None
    finally:
        tracker.close()


def test_stage_tracker_handles_existing_tzaware_rows(tmp_path):
    db_path = tmp_path / "stage.db"
    tracker = StageTracker(db_path=db_path)
    try:
        # Simulate historical row written with timezone offset.
        tracker._con.execute(
            """
            INSERT INTO stage_cooldown(pocket, direction, reason, cooldown_until)
            VALUES (?, ?, ?, ?)
            """,
            ("macro", "short", "loss_cluster", "2025-10-30T12:52:00+00:00"),
        )
        tracker._con.commit()

        now_aware = datetime(2025, 10, 30, 12, 51, tzinfo=timezone.utc)
        cooldown = tracker.get_cooldown("macro", "short", now=now_aware)
        assert cooldown is not None
        assert cooldown.cooldown_until.tzinfo is None
    finally:
        tracker.close()
