from __future__ import annotations

from datetime import datetime, timezone

from scripts import run_market_playbook_cycle


def _cfg() -> run_market_playbook_cycle.CadenceConfig:
    return run_market_playbook_cycle.CadenceConfig(
        normal_interval_sec=900,
        event_interval_sec=300,
        active_interval_sec=60,
        post_interval_sec=300,
        pre_window_min=60,
        active_before_min=10,
        active_after_min=30,
        post_window_min=120,
    )


def test_select_interval_active_window() -> None:
    now = datetime(2026, 2, 27, 13, 0, tzinfo=timezone.utc)
    interval, phase, _ = run_market_playbook_cycle.select_interval_sec(
        events=[{"minutes_to_event": 5, "name": "PPI"}],
        now_utc=now,
        cfg=_cfg(),
    )
    assert interval == 60
    assert phase == "event_active"


def test_select_interval_pre_window() -> None:
    now = datetime(2026, 2, 27, 13, 0, tzinfo=timezone.utc)
    interval, phase, _ = run_market_playbook_cycle.select_interval_sec(
        events=[{"minutes_to_event": 45, "name": "PMI"}],
        now_utc=now,
        cfg=_cfg(),
    )
    assert interval == 300
    assert phase == "event_pre"


def test_select_interval_post_window() -> None:
    now = datetime(2026, 2, 27, 13, 0, tzinfo=timezone.utc)
    interval, phase, _ = run_market_playbook_cycle.select_interval_sec(
        events=[{"minutes_to_event": -40, "name": "PPI"}],
        now_utc=now,
        cfg=_cfg(),
    )
    assert interval == 300
    assert phase == "event_post"


def test_select_interval_normal_when_no_events() -> None:
    now = datetime(2026, 2, 27, 13, 0, tzinfo=timezone.utc)
    interval, phase, event = run_market_playbook_cycle.select_interval_sec(
        events=[],
        now_utc=now,
        cfg=_cfg(),
    )
    assert interval == 900
    assert phase == "normal"
    assert event is None
