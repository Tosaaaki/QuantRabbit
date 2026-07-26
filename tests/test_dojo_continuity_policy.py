from __future__ import annotations

from quant_rabbit.dojo_continuity_policy import (
    PROBE_INTERVAL_SECONDS,
    build_continuity_policy,
    classify_continuity_reason,
)


def test_integrity_and_authority_failures_stop_the_whole_generation() -> None:
    for reason in (
        "SOURCE_OR_POLICY_SEAL_MISMATCH",
        "NON_MONOTONIC_OR_FUTURE_MARKET_DATA",
        "AUTHORITY_ESCALATION_DETECTED",
        "DISK_RECOVERY_FLOOR_VIOLATION",
    ):
        result = classify_continuity_reason(reason)
        assert result["classification"] == "GLOBAL_STOP"
        assert result["other_coordinates_may_continue"] is False
        assert result["gate_bypass_allowed"] is False


def test_coordinate_failure_is_visible_but_does_not_stop_other_coordinates() -> None:
    result = classify_continuity_reason("PORTFOLIO_PREPARE_FAILURE")

    assert result["classification"] == "COORDINATE_ISOLATE"
    assert result["other_coordinates_may_continue"] is True
    assert result["retry_scope"] == "NEW_GENERATION_AFTER_REPAIR"


def test_sparse_dependency_defers_only_to_next_causal_coordinate() -> None:
    result = classify_continuity_reason(
        "FRESH_EXECUTION_CONVERSION_QUOTE_UNAVAILABLE"
    )

    assert result["classification"] == "COORDINATE_DEFER"
    assert result["retry_scope"] == "NEXT_CAUSAL_QUOTE_COORDINATE"
    assert result["other_coordinates_may_continue"] is True


def test_lifecycle_waits_use_existing_gates_and_short_probe() -> None:
    policy = build_continuity_policy()
    waits = {row["state"]: row for row in policy["automatable_lifecycle_waits"]}

    assert PROBE_INTERVAL_SECONDS == 60
    assert policy["recovery_probe_interval_seconds"] == 60
    assert waits["WAIT_FOR_SIGNED_REMOTE_ATTESTATION"][
        "existing_authority_required"
    ] is True
    assert all(row["gate_bypass_allowed"] is False for row in waits.values())
    assert policy["failed_coordinate_dropped_from_denominator"] is False
    assert policy["authority"]["order_authority"] == "NONE"


def test_unknown_reason_fails_closed() -> None:
    result = classify_continuity_reason("NEW_UNREVIEWED_REASON")

    assert result["classification"] == "GLOBAL_STOP"
    assert result["retry_scope"] == "NONE_UNKNOWN_REASON_FAIL_CLOSED"
