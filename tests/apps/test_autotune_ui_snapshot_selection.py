from __future__ import annotations

from datetime import datetime, timedelta, timezone

import apps.autotune_ui as ui


def _snapshot(minutes_ago: int) -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    return {
        "generated_at": ts,
        "metrics": {
            "daily": {"pips": 1.0, "jpy": 100.0, "trades": 1, "wins": 1, "losses": 0},
        },
    }


def test_pick_snapshot_prefers_fresh_when_remote_is_stale(monkeypatch):
    monkeypatch.setattr(ui, "_SNAPSHOT_STALE_MAX_AGE_SEC", 120)
    candidates = [
        ("remote", _snapshot(minutes_ago=30)),
        ("local", _snapshot(minutes_ago=1)),
    ]

    picked = ui._pick_snapshot_by_preference(candidates)

    assert picked is not None
    assert picked[0] == "local"


def test_pick_snapshot_keeps_remote_preference_when_fresh(monkeypatch):
    monkeypatch.setattr(ui, "_SNAPSHOT_STALE_MAX_AGE_SEC", 3600)
    candidates = [
        ("remote", _snapshot(minutes_ago=20)),
        ("local", _snapshot(minutes_ago=1)),
    ]

    picked = ui._pick_snapshot_by_preference(candidates)

    assert picked is not None
    assert picked[0] == "remote"


def test_pick_snapshot_falls_back_to_latest_when_all_stale(monkeypatch):
    monkeypatch.setattr(ui, "_SNAPSHOT_STALE_MAX_AGE_SEC", 30)
    candidates = [
        ("remote", _snapshot(minutes_ago=30)),
        ("local", _snapshot(minutes_ago=10)),
        ("gcs", _snapshot(minutes_ago=5)),
    ]

    picked = ui._pick_snapshot_by_preference(candidates)

    assert picked is not None
    assert picked[0] == "gcs"
