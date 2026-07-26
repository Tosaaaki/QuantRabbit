"""Machine-readable stop, isolation, and lifecycle-wait policy for DOJO.

Unknown reasons fail closed as global integrity stops.  Coordinate failures
remain visible in the fixed denominator, while lifecycle transitions may be
retried only through the existing sealed supervisor and signing authorities.
"""

from __future__ import annotations

from typing import Any, Final

from quant_rabbit.dojo_portfolio_replay_reducer import canonical_portfolio_sha256


CONTRACT: Final = "QR_DOJO_CONTINUITY_POLICY_V1"
SCHEMA_VERSION: Final = 1
PROBE_INTERVAL_SECONDS: Final = 60

GLOBAL_STOP_REASONS: Final = (
    "AUTHORITY_ESCALATION_DETECTED",
    "CLAIM_OR_KERNEL_LEASE_CONFLICT",
    "CUSTODY_OR_ATTESTATION_MISMATCH",
    "DISK_RECOVERY_FLOOR_VIOLATION",
    "FIXED_DENOMINATOR_DRIFT",
    "IMPLEMENTATION_DIGEST_MISMATCH",
    "NON_MONOTONIC_OR_FUTURE_MARKET_DATA",
    "SOURCE_OR_POLICY_SEAL_MISMATCH",
    "TRANSCRIPT_CHAIN_FORK",
)
COORDINATE_ISOLATION_REASONS: Final = (
    "PORTFOLIO_CONSUME_FAILURE",
    "PORTFOLIO_FINALIZE_FAILURE",
    "PORTFOLIO_PREPARE_FAILURE",
    "STRATEGY_WORKER_FAILURE",
    "WORKER_PROTOCOL_FAILURE",
)
COORDINATE_DEFER_REASONS: Final = (
    "FRESH_EXECUTION_CONVERSION_QUOTE_UNAVAILABLE",
    "FRESH_EXECUTION_PAIR_QUOTE_UNAVAILABLE",
)
AUTOMATABLE_LIFECYCLE_WAITS: Final = (
    "ARCHIVE_NEXT",
    "WAIT_FOR_EXACT_V2_RAW_RECLAIM",
    "WAIT_FOR_SIGNED_REMOTE_ATTESTATION",
)


def build_continuity_policy() -> dict[str, Any]:
    body = {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "global_stop_reasons": list(GLOBAL_STOP_REASONS),
        "coordinate_isolation_reasons": list(COORDINATE_ISOLATION_REASONS),
        "coordinate_defer_reasons": list(COORDINATE_DEFER_REASONS),
        "automatable_lifecycle_waits": [
            {
                "state": "ARCHIVE_NEXT",
                "automatic_step": "SEALED_LOCAL_ARCHIVE_AND_READBACK",
                "existing_authority_required": False,
                "gate_bypass_allowed": False,
            },
            {
                "state": "WAIT_FOR_SIGNED_REMOTE_ATTESTATION",
                "automatic_step": (
                    "DRIVE_OBJECT_READBACK_AND_EXISTING_APPROVED_SIGNER"
                ),
                "existing_authority_required": True,
                "gate_bypass_allowed": False,
            },
            {
                "state": "WAIT_FOR_EXACT_V2_RAW_RECLAIM",
                "automatic_step": "EXACT_V2_PLAN_REVALIDATE_AND_RECLAIM",
                "existing_authority_required": False,
                "gate_bypass_allowed": False,
            },
        ],
        "recovery_probe_interval_seconds": PROBE_INTERVAL_SECONDS,
        "one_expensive_transition_per_probe": True,
        "active_job_duplicate_claim_allowed": False,
        "failed_coordinate_dropped_from_denominator": False,
        "failed_coordinate_blocks_other_coordinates": False,
        "retry_same_immutable_failed_result_allowed": False,
        "repair_requires_new_implementation_binding_and_generation": True,
        "authority": {
            "paper_replay_only": True,
            "live_permission": False,
            "broker_mutation_allowed": False,
            "order_authority": "NONE",
            "automatic_deployment_allowed": False,
        },
    }
    return {**body, "policy_sha256": canonical_portfolio_sha256(body)}


def classify_continuity_reason(reason: str) -> dict[str, Any]:
    """Classify a machine reason; an unknown value is an integrity stop."""

    if reason in GLOBAL_STOP_REASONS:
        classification = "GLOBAL_STOP"
        retry_scope = "NONE_UNTIL_INTEGRITY_RESTORED"
    elif reason in COORDINATE_ISOLATION_REASONS:
        classification = "COORDINATE_ISOLATE"
        retry_scope = "NEW_GENERATION_AFTER_REPAIR"
    elif reason in COORDINATE_DEFER_REASONS:
        classification = "COORDINATE_DEFER"
        retry_scope = "NEXT_CAUSAL_QUOTE_COORDINATE"
    elif reason in AUTOMATABLE_LIFECYCLE_WAITS:
        classification = "AUTOMATABLE_LIFECYCLE_WAIT"
        retry_scope = "SEALED_SUPERVISOR_PROBE"
    else:
        classification = "GLOBAL_STOP"
        retry_scope = "NONE_UNKNOWN_REASON_FAIL_CLOSED"
    return {
        "reason": reason,
        "classification": classification,
        "retry_scope": retry_scope,
        "other_coordinates_may_continue": classification
        in {"COORDINATE_ISOLATE", "COORDINATE_DEFER"},
        "gate_bypass_allowed": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }


__all__ = [
    "AUTOMATABLE_LIFECYCLE_WAITS",
    "CONTRACT",
    "COORDINATE_DEFER_REASONS",
    "COORDINATE_ISOLATION_REASONS",
    "GLOBAL_STOP_REASONS",
    "PROBE_INTERVAL_SECONDS",
    "build_continuity_policy",
    "classify_continuity_reason",
]
