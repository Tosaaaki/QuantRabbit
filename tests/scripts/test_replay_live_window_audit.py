from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scripts.replay_live_window_audit as audit


UTC = timezone.utc


def _write_ticks(path: Path, start: datetime, prices: list[float]) -> None:
    rows = []
    for idx, price in enumerate(prices):
        ts = (start + timedelta(minutes=idx)).isoformat()
        rows.append(
            {
                "ts": ts,
                "instrument": "USD_JPY",
                "bid": price - 0.001,
                "ask": price + 0.001,
                "mid": price,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_trades_db(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE trades (
            strategy_tag TEXT,
            entry_time TEXT,
            open_time TEXT,
            close_time TEXT
        )
        """
    )
    con.executemany(
        "INSERT INTO trades(strategy_tag, entry_time, open_time, close_time) VALUES (?, ?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()


def test_build_trade_windows_merges_overlapping_ranges() -> None:
    base = datetime(2026, 3, 6, 9, 0, tzinfo=UTC)
    trades = [
        audit.LiveTrade(
            strategy_tag="TrendBreakout",
            entry_time=base,
            open_time=base,
            close_time=base + timedelta(minutes=2),
        ),
        audit.LiveTrade(
            strategy_tag="TrendBreakout",
            entry_time=base + timedelta(minutes=3),
            open_time=base + timedelta(minutes=3),
            close_time=base + timedelta(minutes=6),
        ),
    ]

    windows = audit._build_trade_windows(trades, pre_minutes=2, post_minutes=3)

    assert len(windows) == 1
    assert windows[0].start == base - timedelta(minutes=2)
    assert windows[0].end == base + timedelta(minutes=9)
    assert windows[0].trade_count == 2


def test_required_tick_basenames_spans_multiple_utc_dates() -> None:
    window = audit.ReplayWindow(
        start=datetime(2026, 3, 5, 23, 55, tzinfo=UTC),
        end=datetime(2026, 3, 6, 0, 5, tzinfo=UTC),
        trade_count=1,
    )

    assert audit._required_tick_basenames(window) == [
        "USD_JPY_ticks_20260305.jsonl",
        "USD_JPY_ticks_20260306.jsonl",
    ]


def test_build_report_marks_missing_and_covered_windows(tmp_path) -> None:
    trades_db = tmp_path / "trades.db"
    first_open = datetime(2026, 3, 6, 9, 6, tzinfo=UTC)
    second_open = datetime(2026, 3, 7, 9, 6, tzinfo=UTC)
    _write_trades_db(
        trades_db,
        [
            ("TrendBreakout", first_open.isoformat(), first_open.isoformat(), (first_open + timedelta(minutes=4)).isoformat()),
            ("TrendBreakout", second_open.isoformat(), second_open.isoformat(), (second_open + timedelta(minutes=2)).isoformat()),
        ],
    )

    covered_ticks = tmp_path / "USD_JPY_ticks_20260306.jsonl"
    _write_ticks(
        covered_ticks,
        start=datetime(2026, 3, 6, 9, 0, tzinfo=UTC),
        prices=[157.6 + (idx * 0.001) for idx in range(20)],
    )

    out_dir = tmp_path / "out"
    report = audit.build_report(
        workers=["trend_breakout"],
        trades_db=trades_db,
        tick_patterns=[str(tmp_path / "USD_JPY_ticks_*.jsonl")],
        pre_minutes=3,
        post_minutes=5,
        out_dir=out_dir,
        run_replay=False,
    )

    worker = report["workers"]["trend_breakout"]
    assert worker["live_trade_count"] == 2
    assert worker["window_count"] == 2

    first_window = worker["windows"][0]
    assert first_window["coverage"]["status"] == "covered"
    assert first_window["required_tick_basenames"] == ["USD_JPY_ticks_20260306.jsonl"]
    assert first_window["clipped_tick_count"] > 0
    assert Path(first_window["clipped_ticks_path"]).exists()
    assert first_window["replay"]["status"] == "skipped"

    second_window = worker["windows"][1]
    assert second_window["coverage"]["status"] == "missing"
    assert second_window["required_tick_basenames"] == ["USD_JPY_ticks_20260307.jsonl"]
    assert second_window["clipped_tick_count"] == 0
    assert second_window["clipped_ticks_path"] is None


def test_main_runs_standard_replay_when_requested(tmp_path, monkeypatch) -> None:
    trades_db = tmp_path / "trades.db"
    open_time = datetime(2026, 3, 6, 9, 6, tzinfo=UTC)
    _write_trades_db(
        trades_db,
        [
            ("TrendBreakout", open_time.isoformat(), open_time.isoformat(), (open_time + timedelta(minutes=2)).isoformat()),
        ],
    )
    ticks = tmp_path / "USD_JPY_ticks_20260306.jsonl"
    _write_ticks(
        ticks,
        start=datetime(2026, 3, 6, 9, 0, tzinfo=UTC),
        prices=[157.7 + (idx * 0.001) for idx in range(12)],
    )

    calls: list[tuple[str, Path, Path]] = []

    def fake_run_replay(*, worker: str, ticks_path: Path, out_dir: Path) -> dict[str, object]:
        calls.append((worker, ticks_path, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "summary_all.json"
        summary_path.write_text("{}", encoding="utf-8")
        return {
            "status": "completed",
            "command": ["python", "scripts/replay_exit_workers_groups.py"],
            "returncode": 0,
            "stdout": str(summary_path),
            "stderr": "",
            "summary_all_path": str(summary_path),
        }

    monkeypatch.setattr(audit, "_run_replay", fake_run_replay)
    out_dir = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_live_window_audit.py",
            "--workers",
            "trend_breakout",
            "--ticks-glob",
            str(tmp_path / "USD_JPY_ticks_*.jsonl"),
            "--trades-db",
            str(trades_db),
            "--out-dir",
            str(out_dir),
            "--run-replay",
        ],
    )

    audit.main()

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    worker = report["workers"]["trend_breakout"]
    assert len(calls) == 1
    assert worker["windows"][0]["replay"]["status"] == "completed"
    assert worker["windows"][0]["replay"]["summary_all_path"] is not None
