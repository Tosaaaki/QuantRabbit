from __future__ import annotations

from datetime import datetime, timezone

from quant_rabbit.fast_bot_profitability_gate import (
    assess_profitability_evidence,
    build_profitability_evidence,
)


NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)
SOURCE_SHA = "a" * 64


def _metrics(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "sample_count": 140,
        "active_days": 12,
        "profit_factor": 1.40,
        "net_pl_pips": 80.0,
        "expectancy_pips": 0.57,
        "pessimistic_expectancy_pips": 0.10,
        "positive_day_rate": 0.75,
        "max_daily_sample_share": 0.20,
        "spread_included": True,
    }
    value.update(overrides)
    return value


def _evidence(*, side: str = "SHORT", rank_only: bool = False, **metrics: object) -> dict:
    return build_profitability_evidence(
        lane_id=f"failure_trader:AUD_JPY:{side}:BREAKOUT_FAILURE:LIMIT",
        pair="AUD_JPY",
        side=side,
        method="BREAKOUT_FAILURE",
        order_type="LIMIT",
        metrics=_metrics(**metrics),
        source_artifact_sha256=SOURCE_SHA,
        generated_at_utc=NOW,
        evidence_end_utc=datetime(2026, 8, 29, tzinfo=timezone.utc),
        rank_only=rank_only,
    )


def test_stable_positive_evidence_is_shadow_observation_ready_only() -> None:
    result = assess_profitability_evidence(_evidence())
    assert result["status"] == "SHADOW_FORWARD_OBSERVATION_READY"
    assert result["shadow_observation_allowed"] is True
    assert result["primary_trading_candidate_allowed"] is True
    assert result["promotion_allowed"] is False
    assert result["live_permission"] is False
    assert result["execution_authority"] == "NONE"
    assert result["live_order_gateway_invocation_count"] == 0
    assert result["external_order_attempts"] == 0
    assert result["external_orders"] == 0
    assert result["manual_tagless_policy"] == "NO_TOUCH"


def test_positive_but_concentrated_rank_only_evidence_collects_more_days() -> None:
    result = assess_profitability_evidence(
        _evidence(
            rank_only=True,
            sample_count=40,
            active_days=2,
            profit_factor=2.318235,
            net_pl_pips=113.5,
            expectancy_pips=2.8375,
            pessimistic_expectancy_pips=0.714694,
            positive_day_rate=0.5,
            max_daily_sample_share=0.95,
        )
    )
    assert result["status"] == "COLLECT_MORE_INDEPENDENT_DAYS"
    assert result["shadow_observation_allowed"] is True
    assert result["primary_trading_candidate_allowed"] is False
    assert set(result["blockers"]) == {
        "DAILY_SAMPLE_CONCENTRATION_TOO_HIGH",
        "INSUFFICIENT_ACTIVE_DAYS",
        "INSUFFICIENT_SAMPLES",
        "POSITIVE_DAY_RATE_BELOW_FLOOR",
    }


def test_positive_thin_evidence_with_unestimable_bound_collects_more_days() -> None:
    result = assess_profitability_evidence(
        _evidence(
            sample_count=11,
            active_days=1,
            profit_factor=1.323944,
            net_pl_pips=4.6,
            expectancy_pips=0.418182,
            pessimistic_expectancy_pips=None,
            positive_day_rate=1.0,
            max_daily_sample_share=1.0,
        )
    )

    assert result["status"] == "COLLECT_MORE_INDEPENDENT_DAYS"
    assert "PESSIMISTIC_EXPECTANCY_NOT_ESTIMABLE" in result["blockers"]
    assert "INSUFFICIENT_SAMPLES" in result["blockers"]
    assert "INSUFFICIENT_ACTIVE_DAYS" in result["blockers"]
    assert result["primary_trading_candidate_allowed"] is False


def test_zero_loss_profit_factor_infinity_remains_json_safe_and_shadow_only() -> None:
    result = assess_profitability_evidence(_evidence(profit_factor="INF"))

    assert result["status"] == "SHADOW_FORWARD_OBSERVATION_READY"
    assert result["metrics"]["profit_factor"] == "INF"
    assert result["promotion_allowed"] is False
    assert result["live_permission"] is False


def test_empty_forward_cohort_collects_without_manufacturing_negative_return() -> None:
    result = assess_profitability_evidence(
        _evidence(
            sample_count=0,
            active_days=0,
            profit_factor=0.0,
            net_pl_pips=0.0,
            expectancy_pips=0.0,
            pessimistic_expectancy_pips=None,
            positive_day_rate=0.0,
            max_daily_sample_share=1.0,
        )
    )

    assert result["status"] == "COLLECT_MORE_INDEPENDENT_DAYS"
    assert "NO_RESOLVED_SAMPLES" in result["blockers"]
    assert "NET_PIPS_NOT_POSITIVE" not in result["blockers"]
    assert result["primary_trading_candidate_allowed"] is False


def test_negative_pessimistic_expectancy_is_rejected_even_when_pf_above_one() -> None:
    result = assess_profitability_evidence(
        _evidence(
            profit_factor=1.278563,
            net_pl_pips=116.3,
            expectancy_pips=0.861481,
            pessimistic_expectancy_pips=-0.364079,
        )
    )
    assert result["status"] == "REJECT_NEGATIVE_EXPECTANCY"
    assert result["shadow_observation_allowed"] is False
    assert "PESSIMISTIC_EXPECTANCY_NOT_POSITIVE" in result["blockers"]


def test_direction_is_not_a_profitability_shortcut() -> None:
    short_result = assess_profitability_evidence(_evidence(side="SHORT"))
    long_result = assess_profitability_evidence(_evidence(side="LONG"))
    assert short_result["status"] == long_result["status"]
    assert short_result["blockers"] == long_result["blockers"]


def test_tampered_evidence_fails_closed() -> None:
    evidence = _evidence()
    evidence["profit_factor"] = 99.0
    result = assess_profitability_evidence(evidence)
    assert result["status"] == "REJECT_INVALID_EVIDENCE"
    assert result["shadow_observation_allowed"] is False
    assert "EVIDENCE_SEAL_INVALID" in result["blockers"]


def test_missing_spread_cost_is_not_collectable() -> None:
    result = assess_profitability_evidence(_evidence(spread_included=False))
    assert result["status"] == "REJECT_NEGATIVE_EXPECTANCY"
    assert "SPREAD_NOT_INCLUDED" in result["blockers"]


def test_manual_or_external_authority_tampering_is_invalid() -> None:
    evidence = _evidence()
    evidence["external_orders"] = 1
    result = assess_profitability_evidence(evidence)
    assert result["status"] == "REJECT_INVALID_EVIDENCE"
    assert "EXTERNAL_ORDER_COUNT_NONZERO" in result["blockers"]
    assert "EVIDENCE_SEAL_INVALID" in result["blockers"]
