from __future__ import annotations

import datetime
import pathlib
import sqlite3
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.scalp_ping_5s import worker as scalp_worker


def _create_trades_db(path: pathlib.Path, rows: list[tuple[str, int, str, str, str]]) -> None:
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE trades (
          close_time TEXT,
          units INTEGER,
          close_reason TEXT,
          strategy_tag TEXT,
          pocket TEXT
        )
        """
    )
    con.executemany(
        """
        INSERT INTO trades (close_time, units, close_reason, strategy_tag, pocket)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()
    con.close()


def _sample_signal(side: str) -> scalp_worker.TickSignal:
    return scalp_worker.TickSignal(
        side=side,
        mode="momentum",
        mode_score=1.0,
        momentum_score=0.8,
        revert_score=0.0,
        confidence=75,
        momentum_pips=-0.2 if side == "short" else 0.2,
        trigger_pips=0.1,
        imbalance=0.65,
        tick_rate=5.0,
        span_sec=1.2,
        tick_age_ms=90.0,
        spread_pips=0.8,
        bid=154.5,
        ask=154.51,
        mid=154.505,
        range_pips=1.1,
        instant_range_pips=0.7,
        signal_window_sec=1.5,
    )


def _set_sl_flip_config(monkeypatch) -> None:
    monkeypatch.setattr(
        scalp_worker.config,
        "SL_STREAK_DIRECTION_FLIP_ENABLED",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "SL_STREAK_DIRECTION_FLIP_MIN_STREAK",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "SL_STREAK_DIRECTION_FLIP_LOOKBACK_TRADES",
        6,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "SL_STREAK_DIRECTION_FLIP_MAX_AGE_SEC",
        180.0,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "SL_STREAK_DIRECTION_FLIP_CONFIDENCE_ADD",
        4,
        raising=False,
    )
    monkeypatch.setattr(
        scalp_worker.config,
        "SL_STREAK_DIRECTION_FLIP_CACHE_TTL_SEC",
        0.1,
        raising=False,
    )


def test_load_stop_loss_streak_detects_same_side_streak(monkeypatch, tmp_path: pathlib.Path) -> None:
    now_utc = datetime.datetime(2026, 2, 19, 10, 0, tzinfo=datetime.timezone.utc)
    db_path = tmp_path / "trades.db"
    _create_trades_db(
        db_path,
        [
            ("2026-02-19T09:59:40+00:00", -1200, "STOP_LOSS_ORDER", "scalp_ping_5s_b_live", "scalp_fast"),
            ("2026-02-19T09:59:15+00:00", -1000, "STOP_LOSS_ORDER", "scalp_ping_5s_b_live", "scalp_fast"),
            ("2026-02-19T09:58:30+00:00", 900, "TAKE_PROFIT_ORDER", "scalp_ping_5s_b_live", "scalp_fast"),
        ],
    )
    monkeypatch.setattr(scalp_worker, "_TRADES_DB", db_path, raising=False)
    monkeypatch.setattr(scalp_worker, "_SL_STREAK_CACHE", {}, raising=False)

    streak = scalp_worker._load_stop_loss_streak(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        now_utc=now_utc,
        now_mono=10.0,
    )

    assert streak is not None
    assert streak.side == "short"
    assert streak.streak == 2
    assert streak.sample == 3
    assert streak.age_sec == 20.0


def test_sl_streak_direction_flip_applies_when_signal_is_same_side(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    _set_sl_flip_config(monkeypatch)
    now_utc = datetime.datetime(2026, 2, 19, 10, 0, tzinfo=datetime.timezone.utc)
    db_path = tmp_path / "trades.db"
    _create_trades_db(
        db_path,
        [
            ("2026-02-19T09:59:50+00:00", 1200, "STOP_LOSS_ORDER", "scalp_ping_5s_b_live", "scalp_fast"),
            ("2026-02-19T09:59:20+00:00", 900, "STOP_LOSS_ORDER", "scalp_ping_5s_b_live", "scalp_fast"),
        ],
    )
    monkeypatch.setattr(scalp_worker, "_TRADES_DB", db_path, raising=False)
    monkeypatch.setattr(scalp_worker, "_SL_STREAK_CACHE", {}, raising=False)

    signal = _sample_signal("long")
    flipped, reason, streak = scalp_worker._maybe_sl_streak_direction_flip(
        signal,
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        now_utc=now_utc,
        now_mono=10.0,
    )

    assert streak is not None
    assert flipped is not None
    assert flipped.side == "short"
    assert flipped.mode.endswith("_slflip")
    assert "long->short" in reason


def test_sl_streak_direction_flip_skips_when_signal_already_opposite(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    _set_sl_flip_config(monkeypatch)
    now_utc = datetime.datetime(2026, 2, 19, 10, 0, tzinfo=datetime.timezone.utc)
    db_path = tmp_path / "trades.db"
    _create_trades_db(
        db_path,
        [
            ("2026-02-19T09:59:50+00:00", 1200, "STOP_LOSS_ORDER", "scalp_ping_5s_b_live", "scalp_fast"),
            ("2026-02-19T09:59:20+00:00", 900, "STOP_LOSS_ORDER", "scalp_ping_5s_b_live", "scalp_fast"),
        ],
    )
    monkeypatch.setattr(scalp_worker, "_TRADES_DB", db_path, raising=False)
    monkeypatch.setattr(scalp_worker, "_SL_STREAK_CACHE", {}, raising=False)

    signal = _sample_signal("short")
    flipped, reason, streak = scalp_worker._maybe_sl_streak_direction_flip(
        signal,
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        now_utc=now_utc,
        now_mono=10.0,
    )

    assert streak is not None
    assert flipped is None
    assert reason == "already_opposite"


def test_sl_streak_direction_flip_skips_stale_streak(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    _set_sl_flip_config(monkeypatch)
    monkeypatch.setattr(
        scalp_worker.config,
        "SL_STREAK_DIRECTION_FLIP_MAX_AGE_SEC",
        30.0,
        raising=False,
    )
    now_utc = datetime.datetime(2026, 2, 19, 10, 0, tzinfo=datetime.timezone.utc)
    db_path = tmp_path / "trades.db"
    _create_trades_db(
        db_path,
        [
            ("2026-02-19T09:58:00+00:00", -1200, "STOP_LOSS_ORDER", "scalp_ping_5s_b_live", "scalp_fast"),
            ("2026-02-19T09:57:30+00:00", -900, "STOP_LOSS_ORDER", "scalp_ping_5s_b_live", "scalp_fast"),
        ],
    )
    monkeypatch.setattr(scalp_worker, "_TRADES_DB", db_path, raising=False)
    monkeypatch.setattr(scalp_worker, "_SL_STREAK_CACHE", {}, raising=False)

    signal = _sample_signal("short")
    flipped, reason, streak = scalp_worker._maybe_sl_streak_direction_flip(
        signal,
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        now_utc=now_utc,
        now_mono=10.0,
    )

    assert streak is not None
    assert flipped is None
    assert reason == "streak_stale"
