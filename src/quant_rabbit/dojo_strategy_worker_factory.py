"""Sealed baseline strategy-worker cohort for DOJO TRAIN.

Only attempt 1 is generated here.  Later AI tuning is deliberately blocked
until it is bound to an append-only lineage registry and independently
verified prior TRAIN results; otherwise repeated calls could hide an unbounded
parameter search behind a per-call budget.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

from quant_rabbit.dojo_bot_catalog import AUTHORITY_INVARIANTS, validate_bot_config
from quant_rabbit.dojo_bot_trainer import PROPOSAL_CONTRACT, seal_candidate_proposal


CONTRACT: Final = "QR_DOJO_STRATEGY_WORKER_BASELINE_COHORT_V1"
SCHEMA_VERSION: Final = 1
PAIRS: Final = ("AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_JPY")
FAMILIES: Final = (
    "compression_break",
    "daily_break_pullback",
    "range_fade_limit",
    "spike_fade",
)
LINEAGE_TOTAL_CANDIDATE_CAP: Final = 14


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config(family: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "signal": family,
        "pairs": list(PAIRS),
        "tp_atr": 3.0,
        "sl_pips": 25.0,
        "ceiling_min": 60,
        "max_concurrent_per_pair": 1,
        "global_max_concurrent": 4,
        "per_pos_lev": 5.0,
        "atr_floor_pips": 0.5,
        "exit_policy": "FIXED",
        **dict(AUTHORITY_INVARIANTS),
    }
    if family == "range_fade_limit":
        value.update({"fade_atr": 1.2, "eff_max": 0.2})
    return validate_bot_config(value)


def build_baseline_worker_cohort() -> dict[str, Any]:
    """Return four stable, catalog-valid, sealed attempt-1 proposals."""

    proposals = []
    for ordinal, family in enumerate(FAMILIES, start=1):
        proposals.append(
            seal_candidate_proposal(
                {
                    "contract": PROPOSAL_CONTRACT,
                    "schema_version": 1,
                    "candidate_id": f"qr-a1-{ordinal:02d}-{family.replace('_', '-')}",
                    "family": family,
                    "hypothesis": (
                        f"{family} supplies a distinct market-regime worker under "
                        "the same fixed entry-risk and exit contract."
                    ),
                    "config": _config(family),
                    "risk_increase": False,
                }
            )
        )
    body = {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "attempt_ordinal": 1,
        "total_attempts_in_lineage": 3,
        "candidate_count": len(proposals),
        "lineage_total_candidate_cap": LINEAGE_TOTAL_CANDIDATE_CAP,
        "pairs": list(PAIRS),
        "families": list(FAMILIES),
        "trainer_search_budget": {
            "attempt_ordinal": 1,
            "total_attempts_in_lineage": 3,
            "max_candidates": len(proposals),
        },
        "admission_policy": {
            "broker_leverage": 25.0,
            "max_per_position_gross_leverage": 5.0,
            "max_global_gross_leverage": 20.0,
            "max_peak_margin_fraction": 0.45,
            "max_normal_mtm_drawdown_fraction": 0.10,
            "max_stress_mtm_drawdown_fraction": 0.15,
            "runtime_portfolio_caps_enforced_by_factory": False,
            "runtime_cap_violation_must_fail_trainer_admission": True,
        },
        "later_tuning": {
            "generation_enabled": False,
            "blocked_until": [
                "APPEND_ONLY_LINEAGE_REGISTRY",
                "VERIFIED_PRIOR_TRAIN_RESULT_BINDING",
                "NO_REPEAT_CANDIDATE_REGISTRY",
            ],
        },
        "trainer_candidate_proposals": proposals,
        "authority": {
            "research_train_only": True,
            "promotion_eligible": False,
            "live_permission": False,
            "order_authority": "NONE",
        },
    }
    return {**body, "cohort_sha256": _canonical_sha256(body)}


__all__ = [
    "CONTRACT",
    "FAMILIES",
    "LINEAGE_TOTAL_CANDIDATE_CAP",
    "PAIRS",
    "build_baseline_worker_cohort",
]
