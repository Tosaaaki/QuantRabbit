import pathlib
import sqlite3
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


def test_stage_tracker_reentry_state_updates_when_trade_id_regresses(tmp_path):
    stage_db = tmp_path / "stage.db"
    trades_db = tmp_path / "trades.db"
    tracker = StageTracker(db_path=stage_db)
    try:
        tcon = sqlite3.connect(trades_db)
        try:
            tcon.execute(
                """
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY,
                    pocket TEXT,
                    units INTEGER,
                    pl_pips REAL,
                    realized_pl REAL,
                    strategy_tag TEXT,
                    close_time TEXT,
                    close_price REAL,
                    close_reason TEXT
                )
                """
            )
            tcon.execute(
                """
                INSERT INTO trades(
                    id, pocket, units, pl_pips, realized_pl, strategy_tag, close_time, close_price, close_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    10,
                    "scalp",
                    1000,
                    -1.2,
                    -1200.0,
                    "TickImbalance",
                    "2026-02-24T07:24:26.315457+00:00",
                    155.854,
                    "MARKET_ORDER_TRADE_CLOSE",
                ),
            )
            tcon.commit()
        finally:
            tcon.close()

        tracker._con.execute(
            """
            INSERT INTO strategy_reentry_state(
                strategy, direction, last_trade_id, last_close_time, last_close_price, last_result, last_pl_pips, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "TickImbalance",
                "long",
                99999,
                "2026-02-12T16:49:28.005847+00:00",
                152.746,
                "win",
                0.8,
                "2026-02-12T16:49:28.005847+00:00",
            ),
        )
        tracker._con.commit()

        tracker.update_loss_streaks(
            trades_db=trades_db,
            now=datetime(2026, 2, 24, 7, 25, tzinfo=timezone.utc),
            cooldown_seconds=0,
        )

        row = tracker._con.execute(
            """
            SELECT last_trade_id, last_close_time, last_close_price, last_result, last_pl_pips
            FROM strategy_reentry_state
            WHERE strategy=? AND direction=?
            """,
            ("TickImbalance", "long"),
        ).fetchone()
        assert row is not None
        assert int(row["last_trade_id"]) == 10
        assert row["last_close_time"] == "2026-02-24T07:24:26.315457+00:00"
        assert float(row["last_close_price"]) == pytest.approx(155.854)
        assert row["last_result"] == "loss"
        assert float(row["last_pl_pips"]) == pytest.approx(-1.2)
    finally:
        tracker.close()
