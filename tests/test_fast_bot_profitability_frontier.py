from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "build_fast_bot_profitability_frontier.py"
)
SPEC = importlib.util.spec_from_file_location("profitability_frontier", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _cell(pf: float, net: float) -> dict:
    return {
        "candidate": "TEST",
        "geometry": {"id": "TEST"},
        "target_r": 1.0,
        "pre_holdout_qualified": False,
        "validation": {
            "trades": 100,
            "net_pips": net,
            "profit_factor": pf,
            "risk_scaled_net_pip_units": net / 2.0,
            "risk_scaled_profit_factor": pf,
            "p05_trade_pips": -7.0,
            "maximum_loss_streak": 8,
        },
    }


def _walk(contract: str, cells: list[dict]) -> dict:
    return {
        "contract": contract,
        "execution_authority": "NONE",
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "pre_holdout_cells": cells,
        "selection": {
            "status": "NO_PRE_HOLDOUT_CANDIDATE",
            "holdout_opened": False,
            "live_promotion_allowed": False,
        },
    }


def _audjpy() -> dict:
    base = {
        "sample_count": 135,
        "active_days": 6,
        "profit_factor": 1.278563,
        "net_pl_pips": 116.3,
        "expectancy_pips": 0.861481,
        "pessimistic_expectancy_pips": -0.364079,
        "positive_day_rate": 0.3333,
        "max_daily_sample_share": 0.9111,
        "spread_included": True,
        "replay_window_utc": {
            "first": "2026-05-14T15:15:47Z",
            "last": "2026-06-22T05:47:46Z",
        },
    }
    rank = {
        **base,
        "sample_count": 40,
        "active_days": 2,
        "profit_factor": 2.318235,
        "net_pl_pips": 113.5,
        "expectancy_pips": 2.8375,
        "pessimistic_expectancy_pips": 0.714694,
        "positive_day_rate": 0.5,
        "max_daily_sample_share": 0.95,
    }
    return {
        "live_side_effects": [],
        "requested_shape": {
            "lane_id": "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT",
            "pair": "AUD_JPY",
            "side": "SHORT",
            "method": "BREAKOUT_FAILURE",
            "order_type": "LIMIT",
        },
        "exact_shape_replay": base,
        "rank_only_precision_subset": rank,
    }


def test_frontier_rejects_negative_cells_and_collects_thin_positive_limit() -> None:
    value = MODULE.build_frontier(
        shock=_walk(MODULE.SHOCK_CONTRACT, [_cell(0.50, -480.0)]),
        nonshock=_walk(MODULE.NONSHOCK_CONTRACT, [_cell(0.86, -120.0)]),
        audjpy=_audjpy(),
        shock_sha256="a" * 64,
        nonshock_sha256="b" * 64,
        audjpy_sha256="c" * 64,
        generated_at_utc=datetime(2026, 8, 30, tzinfo=timezone.utc),
    )
    assert value["status"] == "CAPITAL_PRESERVATION_IMPROVED_PROFITABILITY_UNPROVEN"
    assert value["replay_loss_pips_avoided_by_rejecting_best_negative_cells"] == 600.0
    assert value["trade_eligible_candidates"] == []
    assert value["audjpy_limit_evidence"]["exact_shape"]["gate"]["status"] == (
        "REJECT_NEGATIVE_EXPECTANCY"
    )
    assert value["audjpy_limit_evidence"]["rank_only_precision"]["gate"]["status"] == (
        "COLLECT_MORE_INDEPENDENT_DAYS"
    )
    assert value["next_profit_work"] == "COLLECT_DECONCENTRATED_EXACT_LIMIT_FORWARD_TRUTH"
    assert value["live_permission"] is False
    assert value["live_order_gateway_invocation_count"] == 0
    assert value["external_order_attempts"] == 0
    assert value["external_orders"] == 0


def test_walk_forward_with_authority_fails_closed() -> None:
    shock = _walk(MODULE.SHOCK_CONTRACT, [_cell(0.5, -10.0)])
    shock["execution_authority"] = "LIVE"
    try:
        MODULE.build_frontier(
            shock=shock,
            nonshock=_walk(MODULE.NONSHOCK_CONTRACT, [_cell(0.8, -10.0)]),
            audjpy=_audjpy(),
            shock_sha256="a" * 64,
            nonshock_sha256="b" * 64,
            audjpy_sha256="c" * 64,
            generated_at_utc=datetime(2026, 8, 30, tzinfo=timezone.utc),
        )
    except ValueError as exc:
        assert "execution authority" in str(exc)
    else:
        raise AssertionError("authority-bearing artifact must fail closed")


def test_best_cell_is_ranked_only_on_validation() -> None:
    best = MODULE._best_validation_cell(
        {
            "pre_holdout_cells": [
                _cell(0.75, -50.0),
                _cell(0.90, -100.0),
                _cell(0.90, -80.0),
            ]
        }
    )
    assert best["validation_risk_scaled_profit_factor"] == 0.90
    assert best["validation_risk_scaled_net_pip_units"] == -40.0
