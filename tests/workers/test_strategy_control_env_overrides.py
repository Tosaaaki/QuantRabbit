from __future__ import annotations

import importlib


def test_strategy_control_sync_env_overrides_applies_per_strategy_flags(
    monkeypatch, tmp_path
) -> None:
    db_path = tmp_path / "strategy_control.db"

    monkeypatch.setenv("STRATEGY_CONTROL_DB_PATH", str(db_path))
    monkeypatch.setenv("STRATEGY_CONTROL_SEED_STRATEGIES", "foo,bar_live")

    monkeypatch.setenv("STRATEGY_CONTROL_ENTRY_FOO", "0")
    monkeypatch.setenv("STRATEGY_CONTROL_EXIT_BAR_LIVE", "0")
    monkeypatch.setenv("STRATEGY_CONTROL_LOCK_BAR", "1")

    from workers.common import strategy_control as sc

    sc = importlib.reload(sc)
    sc.sync_env_overrides()

    foo = sc.get_flags("foo")
    assert foo is not None
    assert foo == (False, True, False)

    bar = sc.get_flags("bar")
    assert bar is not None
    assert bar == (True, False, True)
