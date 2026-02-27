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


def _hourly_rows(reference_now: datetime, lookback: int = 24) -> list[dict]:
    anchor = reference_now.astimezone(ui._JST).replace(minute=0, second=0, microsecond=0)
    rows: list[dict] = []
    for i in range(lookback):
        hour = anchor - timedelta(hours=i)
        rows.append(
            {
                "key": hour.isoformat(),
                "label": hour.strftime("%m/%d %H:%M"),
                "pips": 0.0,
                "jpy": 0.0,
                "trades": 1,
                "wins": 1,
                "losses": 0,
                "win_rate": 1.0,
            }
        )
    return rows


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


def test_pick_snapshot_prefers_richer_fresh_snapshot_over_remote_when_hourly_available(monkeypatch):
    monkeypatch.setattr(ui, "_SNAPSHOT_STALE_MAX_AGE_SEC", 3600)
    now = datetime.now(timezone.utc)
    remote = _snapshot(minutes_ago=1)
    gcs = {
        "generated_at": now.isoformat(),
        "metrics": {
            "daily": {"pips": 1.0, "jpy": 100.0, "trades": 1, "wins": 1, "losses": 0},
            "yesterday": {"pips": 0.0, "jpy": 0.0, "trades": 0},
            "weekly": {"pips": 1.0, "jpy": 100.0, "trades": 1},
            "total": {"pips": 1.0, "jpy": 100.0, "wins": 1, "losses": 0, "win_rate": 1.0, "trades": 1},
            "hourly_trades": {
                "timezone": "JST",
                "lookback_hours": 24,
                "exclude_manual": True,
                "hours": _hourly_rows(now),
            },
        },
    }
    candidates = [("remote", remote), ("gcs", gcs)]

    picked = ui._pick_snapshot_by_preference(candidates)

    assert picked is not None
    assert picked[0] == "gcs"


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


def test_collect_snapshot_candidates_skips_local_when_gcs_is_fresh(monkeypatch):
    monkeypatch.setattr(ui, "_SNAPSHOT_STALE_MAX_AGE_SEC", 120)
    monkeypatch.setattr(ui, "_fetch_remote_snapshot_with_status", lambda key: (None, f"{key} missing"))
    monkeypatch.setattr(ui, "_fetch_gcs_snapshot", lambda: (_snapshot(minutes_ago=0), None))

    local_called = {"count": 0}

    def _local_stub():
        local_called["count"] += 1
        return _snapshot(minutes_ago=0), None

    monkeypatch.setattr(ui, "_build_local_snapshot_with_status", _local_stub)

    candidates, attempts = ui._collect_snapshot_candidates()

    assert local_called["count"] == 0
    assert [source for source, _ in candidates] == ["gcs"]
    assert attempts[-1]["source"] == "local"
    assert attempts[-1]["status"] == "skip"


def test_collect_snapshot_candidates_uses_local_when_only_gcs_is_stale(monkeypatch):
    monkeypatch.setattr(ui, "_SNAPSHOT_STALE_MAX_AGE_SEC", 30)
    monkeypatch.setattr(ui, "_fetch_remote_snapshot_with_status", lambda key: (None, f"{key} missing"))
    monkeypatch.setattr(ui, "_fetch_gcs_snapshot", lambda: (_snapshot(minutes_ago=10), None))

    local_called = {"count": 0}

    def _local_stub():
        local_called["count"] += 1
        return _snapshot(minutes_ago=0), None

    monkeypatch.setattr(ui, "_build_local_snapshot_with_status", _local_stub)

    candidates, attempts = ui._collect_snapshot_candidates()

    assert local_called["count"] == 1
    assert [source for source, _ in candidates] == ["gcs", "local"]
    assert attempts[-1]["source"] == "local"
    assert attempts[-1]["status"] == "ok"
