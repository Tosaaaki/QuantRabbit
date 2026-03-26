from __future__ import annotations

from datetime import datetime, timezone

from scripts import policy_guard


def _base_kwargs() -> dict:
    return {
        "max_decision_ms": 2000.0,
        "max_data_lag_ms": 3000.0,
        "max_drawdown_pct": 0.18,
        "min_order_success": 0.995,
        "max_reject_rate": 0.01,
        "max_gpt_timeout": 0.05,
        "reject_streak_threshold": 3,
        "require_slo_metrics_when_open": True,
        "slo_metrics_max_stale_sec": 900.0,
    }


def test_collect_violations_flags_missing_slo_metrics_when_market_open() -> None:
    metrics = {
        "decision_latency_count": 0,
        "data_lag_count": 0,
        "reject_streak": 0,
    }
    violations = policy_guard.collect_violations(
        metrics,
        market_open=True,
        now_utc=datetime(2026, 2, 21, 0, 0, tzinfo=timezone.utc),
        **_base_kwargs(),
    )
    assert "decision_latency_missing" in violations
    assert "data_lag_missing" in violations


def test_collect_violations_ignores_missing_slo_metrics_when_market_closed() -> None:
    metrics = {
        "decision_latency_count": 0,
        "data_lag_count": 0,
        "reject_streak": 0,
    }
    violations = policy_guard.collect_violations(
        metrics,
        market_open=False,
        now_utc=datetime(2026, 2, 21, 0, 0, tzinfo=timezone.utc),
        **_base_kwargs(),
    )
    assert "decision_latency_missing" not in violations
    assert "data_lag_missing" not in violations


def test_collect_violations_flags_stale_slo_metrics_when_market_open() -> None:
    metrics = {
        "decision_latency_count": 10,
        "data_lag_count": 12,
        "decision_latency_latest_ts": "2026-02-21T00:00:00+00:00",
        "data_lag_latest_ts": "2026-02-21T00:00:00Z",
        "reject_streak": 0,
    }
    violations = policy_guard.collect_violations(
        metrics,
        market_open=True,
        now_utc=datetime(2026, 2, 21, 0, 20, tzinfo=timezone.utc),
        **_base_kwargs(),
    )
    assert "decision_latency_stale" in violations
    assert "data_lag_stale" in violations


def test_collect_violations_keeps_threshold_checks() -> None:
    metrics = {
        "decision_latency_p95": 2500.0,
        "data_lag_p95": 5000.0,
        "drawdown_pct_max": 0.2,
        "order_success_min": 0.9,
        "reject_rate_max": 0.2,
        "gpt_timeout_rate_max": 0.2,
        "reject_streak": 5,
        "decision_latency_count": 1,
        "data_lag_count": 1,
        "decision_latency_latest_ts": "2026-02-21T00:19:30+00:00",
        "data_lag_latest_ts": "2026-02-21T00:19:30+00:00",
    }
    violations = policy_guard.collect_violations(
        metrics,
        market_open=True,
        now_utc=datetime(2026, 2, 21, 0, 20, tzinfo=timezone.utc),
        **_base_kwargs(),
    )
    assert "decision_latency_p95" in violations
    assert "data_lag_p95" in violations
    assert "drawdown_pct_max" in violations
    assert "order_success_rate" in violations
    assert "reject_rate_max" in violations
    assert "gpt_timeout_rate_max" in violations
    assert "reject_streak" in violations
