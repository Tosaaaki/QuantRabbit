from __future__ import annotations

from datetime import datetime, timezone

from execution import reentry_gate


def test_empty_strategy_hours_inherit_default_block(monkeypatch) -> None:
    monkeypatch.setattr(reentry_gate, "_ENABLED", True)
    monkeypatch.setattr(
        reentry_gate,
        "_load_config",
        lambda: {
            "defaults": {"block_jst_hours": [7, 8], "allow_jst_hours": []},
            "strategies": {"TrendMA": {"block_jst_hours": []}},
        },
    )
    # 22:00 UTC == 07:00 JST
    now = datetime(2026, 2, 12, 22, 0, tzinfo=timezone.utc)
    allowed, reason, details = reentry_gate.allow_entry(
        strategy_tag="TrendMA",
        units=1000,
        price=150.0,
        now=now,
    )
    assert allowed is False
    assert reason == "time_block"
    assert details.get("jst_hour") == 7


def test_non_empty_strategy_hours_override_default(monkeypatch) -> None:
    monkeypatch.setattr(reentry_gate, "_ENABLED", True)
    monkeypatch.setattr(
        reentry_gate,
        "_load_config",
        lambda: {
            "defaults": {"block_jst_hours": [7, 8], "allow_jst_hours": []},
            "strategies": {"TrendMA": {"block_jst_hours": [9]}},
        },
    )
    # 22:00 UTC == 07:00 JST (should not be blocked by strategy override [9])
    now = datetime(2026, 2, 12, 22, 0, tzinfo=timezone.utc)
    allowed, reason, _ = reentry_gate.allow_entry(
        strategy_tag="TrendMA",
        units=1000,
        price=150.0,
        now=now,
    )
    assert allowed is True
    assert reason == "no_state"

