import pathlib
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.stage_tracker import StageTracker


def _create_trades_db(trades_db: pathlib.Path, rows: list[tuple]) -> None:
    con = sqlite3.connect(trades_db)
    try:
        con.execute(
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
        con.executemany(
            """
            INSERT INTO trades(
                id, pocket, units, pl_pips, realized_pl, strategy_tag, close_time, close_price, close_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        con.commit()
    finally:
        con.close()


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


def test_stage_tracker_is_blocked_handles_naive_public_cooldown(tmp_path):
    db_path = tmp_path / "stage.db"
    tracker = StageTracker(db_path=db_path)
    try:
        now_aware = datetime(2025, 10, 30, 12, 51, tzinfo=timezone.utc)
        tracker.set_cooldown(
            "scalp",
            "long",
            reason="loss_cluster",
            seconds=120,
            now=now_aware,
        )

        blocked, remain, reason = tracker.is_blocked(
            "scalp",
            "long",
            now=now_aware + timedelta(seconds=30),
        )

        assert blocked is True
        assert remain is not None and 1 <= remain <= 90
        assert reason == "loss_cluster"
    finally:
        tracker.close()


def test_stage_tracker_ensure_cooldown_handles_naive_public_cooldown(tmp_path):
    db_path = tmp_path / "stage.db"
    tracker = StageTracker(db_path=db_path)
    try:
        now_aware = datetime(2025, 10, 30, 12, 51, tzinfo=timezone.utc)
        tracker.set_cooldown(
            "micro",
            "short",
            reason="existing",
            seconds=180,
            now=now_aware,
        )

        extended = tracker.ensure_cooldown(
            "micro",
            "short",
            reason="new_reason",
            seconds=60,
            now=now_aware + timedelta(seconds=10),
        )

        cooldown = tracker.get_cooldown("micro", "short", now=now_aware + timedelta(seconds=10))
        assert extended is False
        assert cooldown is not None
        assert cooldown.cooldown_until.tzinfo is None
        assert cooldown.reason == "existing"
    finally:
        tracker.close()


def test_stage_tracker_reentry_state_updates_when_trade_id_regresses(tmp_path):
    stage_db = tmp_path / "stage.db"
    trades_db = tmp_path / "trades.db"
    tracker = StageTracker(db_path=stage_db)
    try:
        _create_trades_db(
            trades_db,
            [
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
            ],
        )

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


def test_stage_tracker_syncs_loss_window_to_current_trades(tmp_path):
    stage_db = tmp_path / "stage.db"
    trades_db = tmp_path / "trades.db"
    tracker = StageTracker(db_path=stage_db)
    try:
        tracker._cooldown_disabled = False
        tracker._cluster_cooldown_disabled = False
        tracker._scalp_cluster_cooldown_disabled = False
        _create_trades_db(
            trades_db,
            [
                (
                    10,
                    "scalp",
                    -3834,
                    -1.8,
                    -69.012,
                    "TickImbalance",
                    "2026-03-13T00:05:54.265570+00:00",
                    159.070,
                    "STOP_LOSS_ORDER",
                ),
            ],
        )
        tracker._con.execute(
            """
            INSERT INTO pocket_loss_window(
                pocket, trade_id, closed_at, loss_jpy, loss_pips, strategy_tag
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("scalp", 9, "2026-03-13T00:17:29.557361+00:00", 69.012, 1.8, "TickImbalance"),
        )
        tracker._con.commit()

        tracker.update_loss_streaks(
            trades_db=trades_db,
            now=datetime(2026, 3, 13, 0, 10, tzinfo=timezone.utc),
            cooldown_seconds=900,
            cooldown_map={"scalp": 900},
            atr_pips=3.1,
            vol_5m=1.2,
        )

        rows = tracker._con.execute(
            """
            SELECT pocket, trade_id, closed_at, loss_jpy, loss_pips, strategy_tag
            FROM pocket_loss_window
            ORDER BY trade_id
            """
        ).fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["pocket"] == "scalp"
        assert int(row["trade_id"]) == 10
        assert row["closed_at"] == "2026-03-13T00:05:54.265570+00:00"
        assert float(row["loss_jpy"]) == pytest.approx(69.012)
        assert float(row["loss_pips"]) == pytest.approx(1.8)
        assert row["strategy_tag"] == "TickImbalance"
    finally:
        tracker.close()


def test_stage_tracker_skips_small_single_strategy_scalp_cluster(tmp_path):
    stage_db = tmp_path / "stage.db"
    trades_db = tmp_path / "trades.db"
    tracker = StageTracker(db_path=stage_db)
    try:
        tracker._cooldown_disabled = False
        tracker._cluster_cooldown_disabled = False
        tracker._scalp_cluster_cooldown_disabled = False
        _create_trades_db(
            trades_db,
            [
                (
                    10,
                    "scalp",
                    -3834,
                    -1.8,
                    -69.012,
                    "TickImbalance",
                    "2026-03-13T00:05:54.265570+00:00",
                    159.070,
                    "STOP_LOSS_ORDER",
                ),
                (
                    11,
                    "scalp",
                    -3834,
                    -1.8,
                    -69.012,
                    "TickImbalance",
                    "2026-03-13T00:17:29.557361+00:00",
                    159.070,
                    "STOP_LOSS_ORDER",
                ),
            ],
        )

        tracker.update_loss_streaks(
            trades_db=trades_db,
            now=datetime(2026, 3, 13, 0, 20, tzinfo=timezone.utc),
            cooldown_seconds=900,
            cooldown_map={"scalp": 900},
            atr_pips=3.1,
            vol_5m=1.2,
        )

        check_now = datetime(2026, 3, 13, 0, 20, tzinfo=timezone.utc)
        assert tracker.get_cooldown("scalp", "long", now=check_now) is None
        assert tracker.get_cooldown("scalp", "short", now=check_now) is None
    finally:
        tracker.close()


def test_stage_tracker_keeps_multi_strategy_scalp_cluster_cooldown(tmp_path):
    stage_db = tmp_path / "stage.db"
    trades_db = tmp_path / "trades.db"
    tracker = StageTracker(db_path=stage_db)
    try:
        _create_trades_db(
            trades_db,
            [
                (
                    10,
                    "scalp",
                    -3834,
                    -1.8,
                    -69.012,
                    "TickImbalance",
                    "2026-03-13T00:05:54.265570+00:00",
                    159.070,
                    "STOP_LOSS_ORDER",
                ),
                (
                    11,
                    "scalp",
                    -272,
                    -1.7,
                    -69.012,
                    "PrecisionLowVol",
                    "2026-03-13T00:17:29.557361+00:00",
                    159.330,
                    "STOP_LOSS_ORDER",
                ),
            ],
        )

        tracker.update_loss_streaks(
            trades_db=trades_db,
            now=datetime(2026, 3, 13, 0, 20, tzinfo=timezone.utc),
            cooldown_seconds=900,
            cooldown_map={"scalp": 900},
            atr_pips=3.1,
            vol_5m=1.2,
        )

        check_now = datetime(2026, 3, 13, 0, 20, tzinfo=timezone.utc)
        long_cd = tracker.get_cooldown("scalp", "long", now=check_now)
        short_cd = tracker.get_cooldown("scalp", "short", now=check_now)
        assert long_cd is not None
        assert short_cd is not None
        assert long_cd.reason == "loss_cluster_2"
        assert short_cd.reason == "loss_cluster_2"
    finally:
        tracker.close()
