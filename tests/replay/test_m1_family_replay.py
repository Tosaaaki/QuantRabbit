from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import scripts.replay_exit_workers_groups as rweg
import scripts.replay_workers as rw


def _ticks_from_mids(mids: list[float]) -> list[rw.Tick]:
    start = datetime(2026, 2, 25, 0, 0, tzinfo=timezone.utc)
    ticks: list[rw.Tick] = []
    for idx, mid in enumerate(mids):
        epoch = (start + timedelta(minutes=idx)).timestamp()
        ticks.append(rw.Tick(epoch=epoch, bid=mid - 0.001, ask=mid + 0.001))
    return ticks


def test_replay_trend_breakout_limit_entry_hits_tp() -> None:
    ticks = _ticks_from_mids(
        [157.000 + (idx * 0.001) for idx in range(20)]
        + [157.050, 157.044, 157.051, 157.053]
    )
    fired = {"done": False}

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
