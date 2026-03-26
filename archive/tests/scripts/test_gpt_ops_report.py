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


def test_build_market_context_uses_external_snapshot_for_crosses() -> None:
    context = gpt_ops_report._build_market_context(
        factors={"M1": {"close": 156.12}},
        events=[],
        now_utc=datetime(2026, 2, 27, 10, 0, tzinfo=timezone.utc),
        external_snapshot={
            "pairs": {
                "EUR_USD": {"price": 1.1812, "change_pct_24h": 0.23},
                "AUD_JPY": {"price": 111.04, "change_pct_24h": 0.61},
                "EUR_JPY": {"price": 184.22, "change_pct_24h": 0.48},
            },
            "dxy": 97.72,
            "dxy_change_pct_24h": -0.21,
            "rates": {"US10Y": 4.01, "JP10Y": 2.12},
        },
        macro_snapshot={"vix": 18.4, "dxy": 0.0, "yield2y": {"USD": 0.62, "JPY": 0.41}},
    )

    assert context["pairs"]["usd_jpy"]["price"] == 156.12
    assert context["pairs"]["eur_usd"]["price"] == 1.1812
    assert context["dollar"]["dxy"] == 97.72
    assert context["rates"]["us_jp_10y_spread"] == 1.89


def test_build_driver_breakdown_detects_yen_flow_dominance() -> None:
    context = {
        "pairs": {
            "usd_jpy": {"price": 156.0, "change_pct_24h": 0.9},
            "aud_jpy": {"price": 111.0, "change_pct_24h": 1.1},
            "eur_jpy": {"price": 184.0, "change_pct_24h": 0.8},
            "eur_usd": {"price": 1.18, "change_pct_24h": 0.2},
        },
        "dollar": {"dxy": 97.7, "dxy_change_pct_24h": -0.3, "source": "external"},
        "rates": {"us_jp_10y_spread": 1.5},
        "risk": {"mode": "neutral"},
    }
    driver = gpt_ops_report._build_driver_breakdown(
        market_context=context,
        event_ctx={"event_soon": False},
    )

    assert driver["dominant_driver"] == "yen_flow"
    assert driver["net_score"] > 0.0


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
    assert payload["playbook_version"] == 2
    assert isinstance(payload["snapshot"], dict)
    assert isinstance(payload["market_context"], dict)
    assert isinstance(payload["driver_breakdown"], dict)
    assert isinstance(payload["break_points"], list)
    assert isinstance(payload["if_then_rules"], list)
    assert isinstance(payload["short_term"], dict)
    assert isinstance(payload["swing"], dict)
    assert isinstance(payload["scenarios"], list)
    assert payload["data_sources"]["factors_ready"] is False


def test_build_ops_report_falls_back_to_external_price_when_factor_is_stale() -> None:
    now_utc = datetime(2026, 2, 27, 10, 0, tzinfo=timezone.utc)
    payload = gpt_ops_report.build_ops_report(
        hours=24.0,
        factors={
            "M1": {"close": 152.727, "ts": "2025-10-29T23:58:00+00:00"},
            "H4": {"ma10": 152.90, "ma20": 152.80},
        },
        forecast={"enabled": False, "reference": None},
        performance={"window_hours": 24.0, "overall": {}, "by_pocket": {}},
        order_stats={"window_hours": 24.0, "total_orders": 0, "reject_rate": 0.0},
        policy={},
        events=[],
        market_context={
            "pairs": {
                "usd_jpy": {"price": 156.0985, "change_pct_24h": 0.1},
                "eur_usd": {"price": 1.18, "change_pct_24h": 0.0},
                "aud_jpy": {"price": 111.0, "change_pct_24h": 0.0},
                "eur_jpy": {"price": 184.0, "change_pct_24h": 0.0},
            },
            "dollar": {"dxy": 97.7, "dxy_change_pct_24h": 0.0, "source": "external"},
            "rates": {"us_jp_10y_spread": 1.9},
            "risk": {"mode": "neutral"},
        },
        now_utc=now_utc,
    )

    assert payload["snapshot"]["factor_stale"] is True
    assert payload["snapshot"]["current_price_source"] == "external_snapshot"
    assert payload["snapshot"]["current_price"] == 156.099
    assert payload["data_sources"]["factors_m1_stale"] is True


def test_build_ops_report_uses_factor_price_when_factor_is_fresh() -> None:
    now_utc = datetime(2026, 2, 27, 10, 0, tzinfo=timezone.utc)
    payload = gpt_ops_report.build_ops_report(
        hours=24.0,
        factors={
            "M1": {
                "close": 155.4321,
                "timestamp": "2026-02-27T09:59:30+00:00",
                "ma10": 155.45,
                "ma20": 155.40,
                "atr_pips": 2.2,
                "candles": [{"high": 155.45, "low": 155.40}],
            },
            "H4": {"ma10": 155.60, "ma20": 155.30, "regime": "Mixed"},
        },
        forecast={"enabled": False, "reference": None},
        performance={"window_hours": 24.0, "overall": {}, "by_pocket": {}},
        order_stats={"window_hours": 24.0, "total_orders": 0, "reject_rate": 0.0},
        policy={},
        events=[],
        market_context={
            "pairs": {
                "usd_jpy": {"price": 156.0, "change_pct_24h": 0.1},
                "eur_usd": {"price": 1.18, "change_pct_24h": 0.0},
                "aud_jpy": {"price": 111.0, "change_pct_24h": 0.0},
                "eur_jpy": {"price": 184.0, "change_pct_24h": 0.0},
            },
            "dollar": {"dxy": 97.7, "dxy_change_pct_24h": 0.0, "source": "external"},
            "rates": {"us_jp_10y_spread": 1.9},
            "risk": {"mode": "neutral"},
        },
        now_utc=now_utc,
    )

    assert payload["snapshot"]["factor_stale"] is False
    assert payload["snapshot"]["current_price_source"] == "factor_cache"
    assert payload["snapshot"]["current_price"] == 155.432
    assert payload["data_sources"]["factors_m1_stale"] is False


def test_build_policy_diff_from_report_sets_directional_bias() -> None:
    now_utc = datetime(2026, 2, 27, 10, 0, tzinfo=timezone.utc)
    report = {
        "generated_at": "2026-02-27T10:00:00+00:00",
        "playbook_version": 2,
        "direction_score": 0.64,
        "direction_confidence_pct": 72.0,
        "short_term": {
            "bias": "long_usd_jpy",
            "primary_scenario": "A. USD/JPY continuation higher",
        },
        "event_context": {
            "event_soon": False,
            "event_active_window": False,
        },
        "snapshot": {
            "factor_stale": False,
            "micro_regime": "trend",
        },
        "order_quality": {
            "reject_rate": 0.04,
        },
        "performance": {"overall": {"profit_factor": 1.08}},
        "scenarios": [
            {"key": "continuation_up", "probability_pct": 62.0},
            {"key": "reversal_down", "probability_pct": 21.0},
            {"key": "event_two_way", "probability_pct": 17.0},
        ],
    }

    diff = gpt_ops_report._build_policy_diff_from_report(
        report=report,
        current_policy={},
        now_utc=now_utc,
    )

    assert diff["no_change"] is False
    patch = diff["patch"]
    assert patch["pockets"]["scalp"]["bias"] == "long"
    assert patch["pockets"]["micro"]["bias"] == "long"
    assert patch["pockets"]["scalp"]["entry_gates"]["allow_new"] is True
    assert patch["event_lock"] is False


def test_build_policy_diff_from_report_detects_no_delta() -> None:
    now_utc = datetime(2026, 2, 27, 10, 0, tzinfo=timezone.utc)
    report = {
        "generated_at": "2026-02-27T10:00:00+00:00",
        "playbook_version": 2,
        "direction_score": 0.12,
        "direction_confidence_pct": 28.0,
        "short_term": {
            "bias": "two_way_wait_for_confirmation",
            "primary_scenario": "C. Event-driven two-way volatility",
        },
        "event_context": {
            "event_soon": True,
            "event_active_window": True,
        },
        "snapshot": {
            "factor_stale": True,
            "micro_regime": "range",
        },
        "order_quality": {
            "reject_rate": 0.32,
        },
        "performance": {"overall": {"profit_factor": 0.86}},
        "scenarios": [
            {"key": "continuation_up", "probability_pct": 20.0},
            {"key": "reversal_down", "probability_pct": 18.0},
            {"key": "event_two_way", "probability_pct": 62.0},
        ],
    }

    first = gpt_ops_report._build_policy_diff_from_report(
        report=report,
        current_policy={},
        now_utc=now_utc,
    )
    assert first["no_change"] is False
    assert first["patch"]["pockets"]["scalp"]["entry_gates"]["allow_new"] is False
    assert first["patch"]["pockets"]["scalp"]["bias"] == "neutral"

    second = gpt_ops_report._build_policy_diff_from_report(
        report=report,
        current_policy=first["patch"],
        now_utc=now_utc,
    )
    assert second["no_change"] is True
    assert "patch" not in second
