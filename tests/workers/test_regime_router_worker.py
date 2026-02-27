from __future__ import annotations

from workers.regime_router import worker as regime_router_worker


def test_decide_candidate_route_prefers_unknown_when_missing() -> None:
    assert regime_router_worker._decide_candidate_route(None, None) == "unknown"
    assert regime_router_worker._decide_candidate_route("", "") == "unknown"


def test_decide_candidate_route_matrix() -> None:
    assert regime_router_worker._decide_candidate_route("Trend", "Trend") == "trend"
    assert regime_router_worker._decide_candidate_route("Breakout", "Trend") == "trend"
    assert regime_router_worker._decide_candidate_route("Range", "Breakout") == "breakout"
    assert regime_router_worker._decide_candidate_route("Range", "Range") == "range"
    assert regime_router_worker._decide_candidate_route("Mixed", "Range") == "range"
    assert regime_router_worker._decide_candidate_route("Trend", "Mixed") == "mixed"
    assert regime_router_worker._decide_candidate_route("Event", "Trend") == "event"


def test_apply_dwell_blocks_fast_switch() -> None:
    route, since, switched = regime_router_worker._apply_dwell(
        active_route="trend",
        active_since_mono=100.0,
        candidate_route="range",
        now_mono=110.0,
        min_dwell_sec=30.0,
    )
    assert route == "trend"
    assert since == 100.0
    assert switched is False


def test_apply_dwell_switches_after_threshold() -> None:
    route, since, switched = regime_router_worker._apply_dwell(
        active_route="trend",
        active_since_mono=100.0,
        candidate_route="range",
        now_mono=135.0,
        min_dwell_sec=30.0,
    )
    assert route == "range"
    assert since == 135.0
    assert switched is True


def test_apply_entry_plan_updates_only_changed_rows(monkeypatch) -> None:
    state = {
        "scalp_ping_5s_b": (True, True, False),
        "scalp_ping_5s_c": (False, True, False),
    }
    calls: list[tuple[str, bool, str]] = []

    monkeypatch.setattr(
        regime_router_worker.strategy_control,
        "get_flags",
        lambda slug: state.get(slug),
    )

    def _set_flags(slug: str, *, entry=None, exit=None, lock=None, note=None) -> None:  # noqa: A002
        state[slug] = (bool(entry), True, False)
        calls.append((slug, bool(entry), str(note or "")))

    monkeypatch.setattr(regime_router_worker.strategy_control, "set_strategy_flags", _set_flags)

    changed, total = regime_router_worker._apply_entry_plan(
        {
            "scalp_ping_5s_b": True,
            "scalp_ping_5s_c": True,
            "scalp_ping_5s_d": False,
        },
        note="router-test",
    )

    assert total == 3
    assert changed == 2
    assert calls == [
        ("scalp_ping_5s_c", True, "router-test"),
        ("scalp_ping_5s_d", False, "router-test"),
    ]


def test_load_config_normalizes_routes(monkeypatch) -> None:
    monkeypatch.setenv("REGIME_ROUTER_MANAGED_STRATEGIES", "scalp_ping_5s_c_live,SCALP_PING_5S_D")
    monkeypatch.setenv("REGIME_ROUTER_TREND_ENTRY_STRATEGIES", "scalp_ping_5s_d_live")
    monkeypatch.setenv("REGIME_ROUTER_RANGE_ENTRY_STRATEGIES", "scalp_ping_5s_c_live")
    monkeypatch.setenv("REGIME_ROUTER_MIXED_ENTRY_STRATEGIES", "scalp_ping_5s_c_live")
    monkeypatch.setenv("REGIME_ROUTER_BREAKOUT_ENTRY_STRATEGIES", "scalp_ping_5s_d_live")
    monkeypatch.setenv("REGIME_ROUTER_UNKNOWN_ENTRY_STRATEGIES", "scalp_ping_5s_c_live")

    cfg = regime_router_worker._load_config()
    assert cfg.managed_strategies == ("scalp_ping_5s_c", "scalp_ping_5s_d")
    assert cfg.route_targets["trend"] == {"scalp_ping_5s_d"}
    assert cfg.route_targets["range"] == {"scalp_ping_5s_c"}

