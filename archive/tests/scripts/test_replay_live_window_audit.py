from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scripts.replay_live_window_audit as audit

UTC = timezone.utc


def _write_ticks(path: Path, start: datetime, prices: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_candles(path: Path, start: datetime, prices: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candles = []
    for idx, price in enumerate(prices):
        ts = (start + timedelta(seconds=5 * idx)).strftime("%Y-%m-%dT%H:%M:%SZ")
        candles.append(
            {
                "time": ts,
                "mid": {
                    "o": f"{price:.3f}",
                    "h": f"{price + 0.002:.3f}",
                    "l": f"{price - 0.002:.3f}",
                    "c": f"{price + 0.001:.3f}",
                },
            }
        )
    payload = {
        "instrument": "USD_JPY",
        "granularity": "S5",
        "price": "M",
        "candles": candles,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_trades_db(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE trades (
            strategy_tag TEXT,
            entry_time TEXT,
            open_time TEXT,
            close_time TEXT
        )
        """)
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
            (
                "TrendBreakout",
                first_open.isoformat(),
                first_open.isoformat(),
                (first_open + timedelta(minutes=4)).isoformat(),
            ),
            (
                "TrendBreakout",
                second_open.isoformat(),
                second_open.isoformat(),
                (second_open + timedelta(minutes=2)).isoformat(),
            ),
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
        replay_warmup_minutes=0.0,
    )

    worker = report["workers"]["trend_breakout"]
    assert worker["live_trade_count"] == 2
    assert worker["window_count"] == 2

    first_window = worker["windows"][0]
    assert first_window["coverage"]["status"] == "covered"
    assert first_window["required_tick_basenames"] == ["USD_JPY_ticks_20260306.jsonl"]
    assert first_window["replay_required_tick_basenames"] == [
        "USD_JPY_ticks_20260306.jsonl"
    ]
    assert first_window["clipped_tick_count"] > 0
    assert Path(first_window["clipped_ticks_path"]).exists()
    assert first_window["replay_window_start"] == first_window["window_start"]
    assert first_window["replay_ticks_path"] == first_window["clipped_ticks_path"]
    assert first_window["replay"]["status"] == "skipped"

    second_window = worker["windows"][1]
    assert second_window["coverage"]["status"] == "missing"
    assert second_window["required_tick_basenames"] == ["USD_JPY_ticks_20260307.jsonl"]
    assert second_window["replay_required_tick_basenames"] == [
        "USD_JPY_ticks_20260307.jsonl"
    ]
    assert second_window["clipped_tick_count"] == 0
    assert second_window["clipped_ticks_path"] is None
    assert second_window["replay_ticks_path"] is None


def test_build_report_extends_replay_window_with_warmup(tmp_path) -> None:
    trades_db = tmp_path / "trades.db"
    open_time = datetime(2026, 3, 6, 9, 6, tzinfo=UTC)
    close_time = open_time + timedelta(minutes=2)
    _write_trades_db(
        trades_db,
        [
            (
                "TrendBreakout",
                open_time.isoformat(),
                open_time.isoformat(),
                close_time.isoformat(),
            ),
        ],
    )

    covered_ticks = tmp_path / "USD_JPY_ticks_20260306.jsonl"
    _write_ticks(
        covered_ticks,
        start=datetime(2026, 3, 6, 8, 58, tzinfo=UTC),
        prices=[157.6 + (idx * 0.001) for idx in range(20)],
    )

    report = audit.build_report(
        workers=["trend_breakout"],
        trades_db=trades_db,
        tick_patterns=[str(tmp_path / "USD_JPY_ticks_*.jsonl")],
        pre_minutes=3,
        post_minutes=5,
        out_dir=tmp_path / "out",
        run_replay=False,
        replay_warmup_minutes=5,
    )

    window = report["workers"]["trend_breakout"]["windows"][0]
    assert window["window_start"] == (open_time - timedelta(minutes=3)).isoformat()
    assert (
        window["replay_window_start"] == (open_time - timedelta(minutes=8)).isoformat()
    )
    assert (
        window["replay_window_end"] == (close_time + timedelta(minutes=5)).isoformat()
    )
    assert window["replay_required_tick_basenames"] == ["USD_JPY_ticks_20260306.jsonl"]
    assert window["clipped_tick_count"] < window["replay_tick_count"]
    assert window["replay_ticks_path"] != window["clipped_ticks_path"]
    assert Path(window["replay_ticks_path"]).exists()
    assert window["coverage"]["replay_tick_file_count"] == 1


def test_build_report_uses_candle_sim_fallback_with_replay_warmup(
    tmp_path, monkeypatch
) -> None:
    trades_db = tmp_path / "trades.db"
    open_time = datetime(2026, 3, 6, 9, 6, tzinfo=UTC)
    close_time = open_time + timedelta(minutes=4)
    _write_trades_db(
        trades_db,
        [
            (
                "TrendBreakout",
                open_time.isoformat(),
                open_time.isoformat(),
                close_time.isoformat(),
            ),
        ],
    )

    generated_ticks = tmp_path / "generated_ticks.jsonl"
    _write_ticks(
        generated_ticks,
        start=datetime(2026, 3, 6, 9, 1, tzinfo=UTC),
        prices=[157.6 + (idx * 0.001) for idx in range(10)],
    )

    calls: list[tuple[str, audit.ReplayWindow, Path, str]] = []

    def fake_run_candle_sim_fallback(
        *,
        instrument: str,
        window: audit.ReplayWindow,
        worker_out_dir: Path,
        label: str,
    ) -> dict[str, object]:
        calls.append((instrument, window, worker_out_dir, label))
        return {
            "enabled": True,
            "used": True,
            "status": "completed",
            "source": "s5_candles_pseudo_ticks",
            "candles_path": str(tmp_path / "candles.json"),
            "generated_ticks_path": str(generated_ticks),
            "candle_count": 12,
            "generated_tick_count": 10,
            "fetch": {"status": "completed"},
            "simulation": {"status": "completed"},
        }

    monkeypatch.setattr(audit, "_run_candle_sim_fallback", fake_run_candle_sim_fallback)

    report = audit.build_report(
        workers=["trend_breakout"],
        trades_db=trades_db,
        tick_patterns=[str(tmp_path / "missing_ticks_*.jsonl")],
        pre_minutes=5,
        post_minutes=15,
        out_dir=tmp_path / "out",
        run_replay=False,
        allow_candle_sim_fallback=True,
        replay_warmup_minutes=60,
    )

    worker = report["workers"]["trend_breakout"]
    window = worker["windows"][0]
    assert len(calls) == 1
    assert calls[0][0] == "USD_JPY"
    assert calls[0][1].start == open_time - timedelta(minutes=65)
    assert calls[0][1].end == close_time + timedelta(minutes=15)
    assert window["coverage"]["status"] == "candle_simulated"
    assert window["coverage"]["fallback_enabled"] is True
    assert window["coverage"]["fallback_used"] is True
    assert window["coverage"]["fallback"]["status"] == "completed"
    assert window["clipped_ticks_path"] is None
    assert window["clipped_tick_count"] == 0
    assert window["replay_ticks_path"] == str(generated_ticks)
    assert window["replay_tick_count"] == 10
    assert window["replay"]["status"] == "skipped"


def test_build_report_uses_default_replay_warmup_for_m1_family(tmp_path) -> None:
    trades_db = tmp_path / "trades.db"
    open_time = datetime(2026, 3, 6, 9, 6, tzinfo=UTC)
    close_time = open_time + timedelta(minutes=2)
    _write_trades_db(
        trades_db,
        [
            (
                "TrendBreakout",
                open_time.isoformat(),
                open_time.isoformat(),
                close_time.isoformat(),
            ),
        ],
    )

    covered_ticks = tmp_path / "USD_JPY_ticks_20260306.jsonl"
    _write_ticks(
        covered_ticks,
        start=datetime(2026, 3, 6, 7, 0, tzinfo=UTC),
        prices=[157.6 + (idx * 0.001) for idx in range(180)],
    )

    report = audit.build_report(
        workers=["trend_breakout"],
        trades_db=trades_db,
        tick_patterns=[str(tmp_path / "USD_JPY_ticks_*.jsonl")],
        pre_minutes=3,
        post_minutes=5,
        out_dir=tmp_path / "out",
        run_replay=False,
    )

    assert report["requested_replay_warmup_minutes"] is None
    window = report["workers"]["trend_breakout"]["windows"][0]
    assert window["replay_warmup_minutes"] == 120.0
    assert (
        window["replay_window_start"]
        == (open_time - timedelta(minutes=123)).isoformat()
    )


def test_main_runs_standard_replay_when_requested(tmp_path, monkeypatch) -> None:
    trades_db = tmp_path / "trades.db"
    open_time = datetime(2026, 3, 6, 9, 6, tzinfo=UTC)
    _write_trades_db(
        trades_db,
        [
            (
                "TrendBreakout",
                open_time.isoformat(),
                open_time.isoformat(),
                (open_time + timedelta(minutes=2)).isoformat(),
            ),
        ],
    )
    ticks = tmp_path / "USD_JPY_ticks_20260306.jsonl"
    _write_ticks(
        ticks,
        start=datetime(2026, 3, 6, 9, 0, tzinfo=UTC),
        prices=[157.7 + (idx * 0.001) for idx in range(12)],
    )

    calls: list[tuple[str, Path, Path]] = []

    def fake_run_replay(
        *, worker: str, ticks_path: Path, out_dir: Path
    ) -> dict[str, object]:
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
            "--replay-warmup-minutes",
            "0",
        ],
    )

    audit.main()

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    worker = report["workers"]["trend_breakout"]
    assert len(calls) == 1
    assert worker["windows"][0]["replay"]["status"] == "completed"
    assert worker["windows"][0]["replay"]["summary_all_path"] is not None
    assert (
        worker["windows"][0]["replay_ticks_path"]
        == worker["windows"][0]["clipped_ticks_path"]
    )


def test_build_report_runs_real_candle_sim_fallback(tmp_path, monkeypatch) -> None:
    trades_db = tmp_path / "trades.db"
    open_time = datetime(2026, 3, 7, 9, 6, tzinfo=UTC)
    _write_trades_db(
        trades_db,
        [
            (
                "TrendBreakout",
                open_time.isoformat(),
                open_time.isoformat(),
                (open_time + timedelta(minutes=2)).isoformat(),
            ),
        ],
    )

    def fake_fetch_candles(
        *, instrument: str, start: datetime, end: datetime, out_path: Path
    ) -> dict[str, object]:
        assert instrument == "USD_JPY"
        assert start <= end
        _write_candles(out_path, start=start, prices=[157.8, 157.81, 157.82])
        return {
            "status": "completed",
            "command": ["python", "scripts/fetch_candles.py"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "candles_path": str(out_path),
            "candle_count": 3,
        }

    def fake_synth_ticks(*, candles_path: Path, out_path: Path) -> dict[str, object]:
        assert candles_path.exists()
        _write_ticks(
            out_path,
            start=datetime(2026, 3, 7, 9, 0, tzinfo=UTC),
            prices=[157.8, 157.81, 157.82],
        )
        return {
            "status": "completed",
            "ticks_path": str(out_path),
            "tick_count": 3,
            "density": [
                {
                    "window_sec": 5,
                    "min_ticks": 1,
                    "samples": 1,
                    "meet": 1,
                    "coverage": 1.0,
                }
            ],
        }

    monkeypatch.setattr(audit, "_fetch_candles_to_json", fake_fetch_candles)
    monkeypatch.setattr(audit, "_synth_candles_to_ticks", fake_synth_ticks)

    report = audit.build_report(
        workers=["trend_breakout"],
        trades_db=trades_db,
        tick_patterns=[str(tmp_path / "USD_JPY_ticks_*.jsonl")],
        pre_minutes=3,
        post_minutes=5,
        out_dir=tmp_path / "out",
        run_replay=False,
        allow_candle_sim_fallback=True,
    )

    window = report["workers"]["trend_breakout"]["windows"][0]
    fallback = window["coverage"]["fallback"]
    assert window["coverage"]["status"] == "candle_simulated"
    assert window["coverage"]["fallback_enabled"] is True
    assert window["coverage"]["fallback_used"] is True
    assert fallback["status"] == "completed"
    assert Path(fallback["candles_path"]).exists()
    assert Path(fallback["generated_ticks_path"]).exists()
    assert fallback["generated_tick_count"] == 3
    assert window["clipped_ticks_path"] is None
    assert window["replay_ticks_path"] == fallback["generated_ticks_path"]
    assert window["replay_tick_count"] == 3


def test_main_runs_replay_with_candle_sim_fallback(tmp_path, monkeypatch) -> None:
    trades_db = tmp_path / "trades.db"
    open_time = datetime(2026, 3, 7, 9, 6, tzinfo=UTC)
    _write_trades_db(
        trades_db,
        [
            (
                "TrendBreakout",
                open_time.isoformat(),
                open_time.isoformat(),
                (open_time + timedelta(minutes=2)).isoformat(),
            ),
        ],
    )

    def fake_fetch_candles(
        *, instrument: str, start: datetime, end: datetime, out_path: Path
    ) -> dict[str, object]:
        _write_candles(out_path, start=start, prices=[157.9, 157.91, 157.92])
        return {
            "status": "completed",
            "command": ["python", "scripts/fetch_candles.py"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "candles_path": str(out_path),
            "candle_count": 3,
        }

    def fake_synth_ticks(*, candles_path: Path, out_path: Path) -> dict[str, object]:
        _write_ticks(
            out_path,
            start=datetime(2026, 3, 7, 9, 0, tzinfo=UTC),
            prices=[157.9, 157.91, 157.92],
        )
        return {
            "status": "completed",
            "ticks_path": str(out_path),
            "tick_count": 3,
            "density": [],
        }

    calls: list[tuple[str, Path, Path]] = []

    def fake_run_replay(
        *, worker: str, ticks_path: Path, out_dir: Path
    ) -> dict[str, object]:
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

    monkeypatch.setattr(audit, "_fetch_candles_to_json", fake_fetch_candles)
    monkeypatch.setattr(audit, "_synth_candles_to_ticks", fake_synth_ticks)
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
            "--allow-candle-sim-fallback",
        ],
    )

    audit.main()

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    window = report["workers"]["trend_breakout"]["windows"][0]
    assert len(calls) == 1
    assert calls[0][0] == "trend_breakout"
    assert calls[0][1] == Path(window["replay_ticks_path"])
    assert window["coverage"]["status"] == "candle_simulated"
    assert window["coverage"]["fallback"]["status"] == "completed"
    assert window["replay"]["status"] == "completed"
    assert window["replay"]["summary_all_path"] is not None
