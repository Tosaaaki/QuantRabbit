from __future__ import annotations

import os


os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")


def test_spread_ok_falls_back_to_tick_cache(monkeypatch):
    from workers.scalp_precision import common

    monkeypatch.setattr(common.time, "time", lambda: 4.0)
    monkeypatch.setattr(common.spread_monitor, "get_state", lambda: None)
    monkeypatch.setattr(
        common.tick_window,
        "recent_ticks",
        lambda seconds=60.0, limit=None: [
            {"epoch": 2.0, "bid": 156.000, "ask": 156.010, "mid": 156.005},
            {"epoch": 3.0, "bid": 156.000, "ask": 156.008, "mid": 156.004},
            {"epoch": 4.0, "bid": 156.000, "ask": 156.006, "mid": 156.003},
        ],
    )

    ok, state = common.spread_ok(max_pips=1.2, p25_max=1.0)
    assert ok is True
    assert state is not None
    assert state.get("source") == "tick_cache"
