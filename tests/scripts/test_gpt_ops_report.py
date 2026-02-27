from __future__ import annotations

from datetime import datetime, timezone

from scripts import gpt_ops_report


def test_calc_bucket_metrics_computes_profit_factor() -> None:
    rows = [
        {"pl_pips": 2.0, "realized_pl": 120.0},
        {"pl_pips": -1.0, "realized_pl": -60.0},
        {"pl_pips": 1.5, "realized_pl": 90.0},
    ]
    metrics = gpt_ops_report._calc_bucket_metrics(rows)

    assert metrics["trade_count"] == 3
    assert metrics["wins"] == 2
    assert metrics["losses"] == 1
    assert metrics["win_rate"] == 0.6667
    assert metrics["profit_factor"] == 3.5
    assert metrics["total_pips"] == 2.5
    assert metrics["total_jpy"] == 150.0


def test_build_scenarios_probabilities_sum_to_hundred() -> None:
    scenarios = gpt_ops_report._build_scenarios(
        direction_score=0.62,
        snapshot={
            "current_price": 156.0,
            "support_price": 155.82,
            "resistance_price": 156.22,
        },
        forecast={"reference": {"p_up": 0.63, "edge": 0.24}},
        performance={"overall": {"profit_factor": 1.12}},
        order_stats={"reject_rate": 0.04},
        event_ctx={"event_soon": False},
    )

    assert len(scenarios) == 3
    total = round(sum(float(row["probability_pct"]) for row in scenarios), 1)
    assert total == 100.0
    primary = max(scenarios, key=lambda row: float(row["probability_pct"]))
    assert primary["key"] == "continuation_up"


def test_build_ops_report_handles_minimal_inputs() -> None:
    payload = gpt_ops_report.build_ops_report(
        hours=24.0,
        factors={},
        forecast={"enabled": False, "reference": None},
        performance={"window_hours": 24.0, "overall": {}, "by_pocket": {}},
        order_stats={"window_hours": 24.0, "total_orders": 0, "reject_rate": 0.0},
        policy={},
        events=[],
        now_utc=datetime(2026, 2, 27, 10, 0, tzinfo=timezone.utc),
    )

    assert payload["llm_disabled"] is True
    assert payload["playbook_version"] == 1
    assert isinstance(payload["snapshot"], dict)
    assert isinstance(payload["short_term"], dict)
    assert isinstance(payload["swing"], dict)
    assert isinstance(payload["scenarios"], list)
    assert payload["data_sources"]["factors_ready"] is False
