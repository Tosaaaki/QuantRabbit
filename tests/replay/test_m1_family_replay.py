from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scripts.replay_exit_workers_groups as rweg
import scripts.replay_workers as rw


def _ticks_from_mids(mids: list[float]) -> list[rw.Tick]:
    start = datetime(2026, 2, 25, 0, 0, tzinfo=timezone.utc)
    ticks: list[rw.Tick] = []
    for idx, mid in enumerate(mids):
        epoch = (start + timedelta(minutes=idx)).timestamp()
        ticks.append(rw.Tick(epoch=epoch, bid=mid - 0.001, ask=mid + 0.001))
    return ticks


def _write_ticks_jsonl(path: Path, ticks: list[rw.Tick]) -> None:
    rows = []
    for tick in ticks:
        rows.append(
            {
                "ts": tick.dt.isoformat(),
                "instrument": "USD_JPY",
                "bid": tick.bid,
                "ask": tick.ask,
                "mid": tick.mid,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_trades_db(path: Path, rows: list[tuple[str, str | None, str | None, str | None]]) -> None:
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


def test_replay_trend_breakout_limit_entry_hits_tp(tmp_path, monkeypatch) -> None:
    ticks = _ticks_from_mids(
        [157.000 + (idx * 0.001) for idx in range(20)]
        + [157.050, 157.044, 157.051, 157.053]
    )
    fired = {"done": False}
    trades_db = tmp_path / "trades.db"
    _write_trades_db(
        trades_db,
        [
            (
                "TrendBreakout",
                ticks[2].dt.isoformat(),
                ticks[2].dt.isoformat(),
                ticks[4].dt.isoformat(),
            ),
            (
                "TrendBreakout",
                (ticks[0].dt - timedelta(hours=2)).isoformat(),
                (ticks[0].dt - timedelta(hours=2)).isoformat(),
                (ticks[0].dt - timedelta(hours=1, minutes=30)).isoformat(),
            ),
        ],
    )
    monkeypatch.setenv("REPLAY_LIVE_TRADES_DB", str(trades_db))

    def fake_signal(fac: dict[str, object]) -> dict[str, object] | None:
        if fired["done"] or len(fac.get("candles") or []) < 20:
            return None
        fired["done"] = True
        return {
            "action": "OPEN_LONG",
            "tag": "M1Scalper-breakout-retest-long",
            "confidence": 82,
            "entry_type": "limit",
            "entry_price": 157.045,
            "entry_tolerance_pips": 0.05,
            "limit_expiry_seconds": 180,
            "sl_pips": 0.8,
            "tp_pips": 0.5,
        }

    result = rw._replay_m1_family("trend_breakout", ticks, signal_func=fake_signal)

    assert "trend_breakout" in rw.SUPPORTED_WORKERS
    assert result["summary"]["trades"] == 1
    assert result["summary"]["pending_unfilled"] == 0
    coverage = result["summary"]["coverage"]
    assert coverage["tick_start"] == ticks[0].dt.isoformat()
    assert coverage["tick_end"] == ticks[-1].dt.isoformat()
    assert coverage["tick_count"] == len(ticks)
    assert coverage["tick_span_sec"] == round(ticks[-1].epoch - ticks[0].epoch, 3)
    live_overlap = coverage["live_trade_overlap"]
    assert live_overlap["status"] == "ok"
    assert live_overlap["strategy_tag"] == "TrendBreakout"
    assert live_overlap["overlap_count"] == 1
    assert live_overlap["total_strategy_trades"] == 2
    trade = result["trades"][0]
    assert trade["strategy_tag"] == "TrendBreakout"
    assert trade["reason"] == "tp"


def test_replay_exit_workers_groups_accepts_trend_breakout(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("M1SCALP_EXIT_MAX_HOLD_SEC", "60")

    ticks = _ticks_from_mids([157.000] * 15)
    entries = [
        rweg.EntryEvent(
            ts=ticks[0].dt,
            direction="long",
            entry_price=157.000,
            tp_pips=20.0,
            sl_pips=20.0,
            units=1000,
            strategy_tag="TrendBreakout",
        )
    ]
    out_path = tmp_path / "replay_exit_trend_breakout.json"

    summary = rweg._simulate(
        ticks_path=tmp_path / "unused_ticks.jsonl",
        ticks_cache=ticks,
        entries=entries,
        worker="trend_breakout",
        out_path=out_path,
        no_hard_sl=True,
        no_hard_tp=True,
        exclude_end_of_replay=True,
        prefeed_h4=[],
    )

    assert "trend_breakout" in rweg.WORKER_TAGS
    assert summary["trades"] == 1
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["summary"]["trades"] == 1
    assert payload["trades"][0]["reason"] == "time_stop"


def test_replay_exit_workers_groups_summary_all_keeps_entry_replay_coverage(tmp_path, monkeypatch) -> None:
    ticks = _ticks_from_mids([157.000 + (idx * 0.001) for idx in range(24)])
    ticks_path = tmp_path / "USD_JPY_ticks_20260225.jsonl"
    _write_ticks_jsonl(ticks_path, ticks)

    trades_db = tmp_path / "trades.db"
    _write_trades_db(
        trades_db,
        [
            (
                "TrendBreakout",
                ticks[1].dt.isoformat(),
                ticks[1].dt.isoformat(),
                ticks[3].dt.isoformat(),
            )
        ],
    )
    monkeypatch.setenv("REPLAY_LIVE_TRADES_DB", str(trades_db))

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_exit_workers_groups.py",
            "--ticks",
            str(ticks_path),
            "--workers",
            "trend_breakout",
            "--out-dir",
            str(out_dir),
            "--no-hard-sl",
            "--no-hard-tp",
            "--exclude-end-of-replay",
        ],
    )

    rweg.main()

    summary_all = json.loads((out_dir / "summary_all.json").read_text(encoding="utf-8"))
    entry_replay = summary_all["trend_breakout"]["entry_replay"]["summary"]
    coverage = entry_replay["coverage"]
    assert coverage["tick_start"] == ticks[0].dt.isoformat()
    assert coverage["tick_end"] == ticks[-1].dt.isoformat()
    assert coverage["tick_count"] == len(ticks)
    assert coverage["live_trade_overlap"]["overlap_count"] == 1
