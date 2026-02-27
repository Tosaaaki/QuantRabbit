from __future__ import annotations

import json
from pathlib import Path

from scripts import replay_regime_router_wfo as tuner


def _write_payload(path: Path, trades: list[dict]) -> None:
    payload = {"trades": trades}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_route_trades_uses_fallback_route(tmp_path: Path) -> None:
    path = tmp_path / "replay.json"
    _write_payload(
        path,
        [
            {
                "entry_time": "2026-02-10T00:00:00+00:00",
                "pnl_pips": 1.0,
                "pnl_jpy": 10.0,
                "macro_regime": "Trend",
                "micro_regime": "Trend",
                "reason": "tp",
            }
        ],
    )

    rows = tuner._load_route_trades(path, worker="C", exclude_end_of_replay=True)
    assert len(rows) == 1
    assert rows[0].route == "trend"
    assert rows[0].worker == "C"


def test_choose_route_mapping_prefers_higher_train_total_jpy() -> None:
    train_rows = [
        tuner.RouteTrade(
            worker="C",
            route="range",
            day="2026-02-10",
            entry_time="2026-02-10T00:00:00+00:00",
            pnl_pips=1.0,
            pnl_jpy=20.0,
            reason="tp",
        ),
        tuner.RouteTrade(
            worker="D",
            route="range",
            day="2026-02-10",
            entry_time="2026-02-10T00:01:00+00:00",
            pnl_pips=1.0,
            pnl_jpy=5.0,
            reason="tp",
        ),
        tuner.RouteTrade(
            worker="D",
            route="trend",
            day="2026-02-10",
            entry_time="2026-02-10T00:02:00+00:00",
            pnl_pips=1.0,
            pnl_jpy=30.0,
            reason="tp",
        ),
        tuner.RouteTrade(
            worker="C",
            route="trend",
            day="2026-02-10",
            entry_time="2026-02-10T00:03:00+00:00",
            pnl_pips=-1.0,
            pnl_jpy=-10.0,
            reason="sl",
        ),
    ]
    mapping = tuner._choose_route_mapping(
        train_rows,
        workers=("C", "D"),
        default_worker="C",
        min_train_route_trades=1,
    )
    assert mapping["range"] == "C"
    assert mapping["trend"] == "D"


def test_build_env_suggestion_majority_vote() -> None:
    env = tuner._build_env_suggestion(
        [
            {"trend": "D", "range": "C", "mixed": "C"},
            {"trend": "D", "range": "D", "mixed": "C"},
            {"trend": "C", "range": "C", "mixed": "C"},
        ],
        worker_to_strategy={"C": "scalp_ping_5s_c", "D": "scalp_ping_5s_d"},
    )
    assert env["REGIME_ROUTER_TREND_ENTRY_STRATEGIES"] == "scalp_ping_5s_d"
    assert env["REGIME_ROUTER_RANGE_ENTRY_STRATEGIES"] == "scalp_ping_5s_c"
    assert env["REGIME_ROUTER_MIXED_ENTRY_STRATEGIES"] == "scalp_ping_5s_c"

