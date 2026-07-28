"""Fail-closed control plane for Paper-only champion/challenger experiments.

This module does not place orders, read a broker, mutate an Automation, or
promote anything to live.  It only validates a sealed candidate admission or
continuation decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from quant_rabbit.dojo_bot_catalog import validate_bot_config


class PaperExperimentPolicyError(ValueError):
    """Raised when an experiment violates a safety or isolation invariant."""


def canonical_sha256(value: Any) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise PaperExperimentPolicyError("timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _require_authority_none(value: Mapping[str, Any]) -> None:
    authority = value.get("authority")
    if not isinstance(authority, Mapping):
        raise PaperExperimentPolicyError("authority is required")
    expected = {
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
        "auto_live_promotion": False,
    }
    for key, expected_value in expected.items():
        if authority.get(key) != expected_value:
            raise PaperExperimentPolicyError(
                f"authority invariant failed: {key}={authority.get(key)!r}"
            )


def validate_policy(policy: Mapping[str, Any]) -> None:
    _require_authority_none(policy)
    if policy.get("dojo_dependency") != "NONE":
        raise PaperExperimentPolicyError("DOJO must not be a dependency")
    budget = policy.get("candidate_budget")
    if not isinstance(budget, Mapping):
        raise PaperExperimentPolicyError("candidate_budget is required")
    if int(budget.get("max_active_challengers", 0)) not in (1, 2):
        raise PaperExperimentPolicyError("active challenger cap must be 1 or 2")
    if not 0 < float(budget.get("max_virtual_capital_jpy_per_challenger", 0)) <= 50_000:
        raise PaperExperimentPolicyError("challenger virtual capital cap is unsafe")
    if not 0 < float(budget.get("max_drawdown_fraction", 0)) <= 0.05:
        raise PaperExperimentPolicyError("challenger DD kill must be at most 5%")
    if not 0 < int(budget.get("max_duration_days", 0)) <= 14:
        raise PaperExperimentPolicyError("challenger duration must be at most 14 days")
    if int(budget.get("minimum_settlements", 0)) < 30:
        raise PaperExperimentPolicyError("minimum settlements must be at least 30")
    if int(budget.get("maximum_fresh_candidates_per_24h", 0)) != 1:
        raise PaperExperimentPolicyError("fresh candidate budget must be one per 24h")


def candidate_hash(candidate: Mapping[str, Any]) -> str:
    body = {key: value for key, value in candidate.items() if key != "candidate_hash"}
    return canonical_sha256(body)


def generate_strategy_candidate(
    *,
    policy: Mapping[str, Any],
    evidence: Mapping[str, Any],
    registry: Mapping[str, Any],
    observed_at_utc: str,
) -> dict[str, Any]:
    """Select one reviewed sibling strategy from completed Paper evidence.

    This is intentionally a bounded template selector, not an unrestricted
    code generator.  The returned configuration cannot place external orders;
    a separate shared-feed Paper runner must activate it.
    """

    validate_policy(policy)
    observed_at = _parse_utc(observed_at_utc)
    if evidence.get("completed_observations_only") is not True:
        raise PaperExperimentPolicyError("evidence must be completed observations")
    if evidence.get("future_or_terminal_data_in_decision") is not False:
        raise PaperExperimentPolicyError("future/outcome data must be excluded")
    evidence_hash = evidence.get("data_hash")
    if not _is_sha256(evidence_hash):
        raise PaperExperimentPolicyError("evidence data_hash must be sha256")
    if evidence_hash in set(registry.get("reviewed_data_hashes", [])):
        status = "NO_NEW_EVIDENCE"
        reason_ids = ["DUPLICATE_DATA_HASH"]
        candidate = None
    else:
        causes = evidence.get("ranked_causes")
        if not isinstance(causes, list) or not causes:
            raise PaperExperimentPolicyError("ranked causes are required")
        lead = causes[0]
        if not isinstance(lead, Mapping):
            raise PaperExperimentPolicyError("lead cause must be an object")
        settlements = int(_finite_float(lead.get("settlements"), default=0))
        loss_jpy = _finite_float(lead.get("net_contribution_jpy"), default=0)
        confidence = str(lead.get("confidence", "")).upper()
        minimum = int(policy["strategy_lab"]["minimum_cause_settlements"])
        minimum_loss = float(policy["strategy_lab"]["minimum_absolute_loss_jpy"])
        if (
            settlements < minimum
            or loss_jpy >= -minimum_loss
            or confidence not in {"HIGH", "VERY_HIGH"}
        ):
            status = "INSUFFICIENT_EVIDENCE"
            reason_ids = ["CAUSE_SAMPLE_OR_IMPACT_BELOW_GATE"]
            candidate = None
        elif lead.get("cause_id") != "COUNTERTREND_SHORT_CONCENTRATION":
            status = "NO_REVIEWED_SIBLING_TEMPLATE"
            reason_ids = ["UNSUPPORTED_CAUSE"]
            candidate = None
        else:
            experiment_key = canonical_sha256(
                {
                    "evidence_data_hash": evidence_hash,
                    "observed_at_utc": observed_at.isoformat(),
                    "template": "pullback_limit",
                }
            )[:16]
            config = validate_bot_config(
                {
                    "signal": "pullback_limit",
                    "pairs": list(evidence.get("pairs") or ["USD_JPY"]),
                    "tp_pips": 6.0,
                    "sl_pips": None,
                    "ceiling_min": 480,
                    "max_concurrent_per_pair": 1,
                    "global_max_concurrent": 1,
                    "per_pos_lev": 1.0,
                    "atr_floor_pips": 1.0,
                    "pull_atr": 0.6,
                    "external_broker_mutation_allowed": False,
                    "live_permission": False,
                    "order_authority": "NONE",
                }
            )
            candidate = {
                "strategy_template_id": "pullback-limit-follow-24h-v1",
                "strategy_principle": "ENTER_PULLBACK_ONLY_WITH_COMPLETED_24H_TREND",
                "source_cause_id": lead["cause_id"],
                "bot_config": config,
                "virtual_capital_jpy": 50_000,
                "max_drawdown_fraction": 0.05,
                "duration_days": 14,
                "shared_feed_contract_sha256": canonical_sha256(
                    {
                        "contract": "QR_SHARED_CAUSAL_PAPER_FEED_V1",
                        "pairs": config["pairs"],
                        "future_data_allowed": False,
                        "single_quote_batch_fanned_out": True,
                    }
                ),
                "virtual_account_id": f"paper-account-{experiment_key}",
                "inventory_id": f"paper-inventory-{experiment_key}",
                "order_book_id": f"paper-orders-{experiment_key}",
                "ledger_id": f"paper-ledger-{experiment_key}",
                "risk_budget_id": f"paper-risk-{experiment_key}",
                "future_data_allowed": False,
                "authority": {
                    "live_permission": False,
                    "broker_mutation_allowed": False,
                    "order_authority": "NONE",
                },
            }
            candidate["candidate_hash"] = candidate_hash(candidate)
            status = "CANDIDATE_PROPOSED"
            reason_ids = ["REVIEWED_SIBLING_MATCHES_LEAD_CAUSE"]

    result = {
        "contract": "QR_PAPER_STRATEGY_LAB_DECISION_V1",
        "status": status,
        "reason_ids": reason_ids,
        "evidence_data_hash": evidence_hash,
        "observed_at_utc": observed_at.isoformat().replace("+00:00", "Z"),
        "candidate": candidate,
        "authority": {
            "live_permission": False,
            "broker_mutation_allowed": False,
            "order_authority": "NONE",
        },
    }
    result["decision_sha256"] = canonical_sha256(result)
    return result


def assess_candidate_admission(
    *,
    policy: Mapping[str, Any],
    candidate: Mapping[str, Any],
    registry: Mapping[str, Any],
    evidence_data_hash: str,
    observed_at_utc: str,
) -> dict[str, Any]:
    """Return a deterministic Paper admission result without mutating state."""

    validate_policy(policy)
    if not _is_sha256(evidence_data_hash):
        raise PaperExperimentPolicyError("evidence_data_hash must be sha256")
    observed_at = _parse_utc(observed_at_utc)
    budget = policy["candidate_budget"]

    candidate_authority = candidate.get("authority")
    if candidate_authority != {
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }:
        raise PaperExperimentPolicyError("candidate authority is not Paper-only")
    if candidate.get("future_data_allowed") is not False:
        raise PaperExperimentPolicyError("future data must be explicitly forbidden")
    if not _is_sha256(candidate.get("shared_feed_contract_sha256")):
        raise PaperExperimentPolicyError("shared causal feed contract hash is required")

    capital = float(candidate.get("virtual_capital_jpy", 0))
    if not 0 < capital <= float(budget["max_virtual_capital_jpy_per_challenger"]):
        raise PaperExperimentPolicyError("candidate virtual capital exceeds budget")
    if not 0 < float(candidate.get("max_drawdown_fraction", 0)) <= float(
        budget["max_drawdown_fraction"]
    ):
        raise PaperExperimentPolicyError("candidate DD kill exceeds policy")
    if not 0 < int(candidate.get("duration_days", 0)) <= int(
        budget["max_duration_days"]
    ):
        raise PaperExperimentPolicyError("candidate duration exceeds policy")

    isolation_values = [
        candidate.get("virtual_account_id"),
        candidate.get("inventory_id"),
        candidate.get("order_book_id"),
        candidate.get("ledger_id"),
        candidate.get("risk_budget_id"),
    ]
    if any(not isinstance(value, str) or not value for value in isolation_values):
        raise PaperExperimentPolicyError("all isolation identifiers are required")
    if len(set(isolation_values)) != len(isolation_values):
        raise PaperExperimentPolicyError("isolation identifiers must be distinct")

    sealed_candidate_hash = candidate_hash(candidate)
    supplied_hash = candidate.get("candidate_hash")
    if supplied_hash is not None and supplied_hash != sealed_candidate_hash:
        raise PaperExperimentPolicyError("candidate hash mismatch")

    active = registry.get("active_candidates", [])
    if not isinstance(active, list):
        raise PaperExperimentPolicyError("active_candidates must be a list")
    if len(active) >= int(budget["max_active_challengers"]):
        status = "REJECT_CANDIDATE_CAP"
    elif sealed_candidate_hash in set(registry.get("candidate_hashes", [])):
        status = "REJECT_DUPLICATE_CANDIDATE_HASH"
    elif evidence_data_hash in set(registry.get("accepted_data_hashes", [])):
        status = "REJECT_DUPLICATE_DATA_HASH"
    else:
        cooldown = timedelta(
            hours=float(
                policy["schedule"]["strategy_lab_minimum_cooldown_hours"]
            )
        )
        admitted_at = [_parse_utc(value) for value in registry.get("admitted_at_utc", [])]
        if any(value > observed_at for value in admitted_at):
            raise PaperExperimentPolicyError("registry contains a future admission")
        recent = [
            value
            for value in admitted_at
            if observed_at - value < timedelta(hours=24)
        ]
        last_admitted_raw = registry.get("last_admitted_at_utc")
        last_admitted = (
            _parse_utc(last_admitted_raw) if last_admitted_raw else None
        )
        if last_admitted is not None and last_admitted > observed_at:
            raise PaperExperimentPolicyError("last admission is in the future")
        if len(recent) >= int(budget["maximum_fresh_candidates_per_24h"]):
            status = "REJECT_DAILY_BUDGET"
        elif last_admitted is not None and observed_at - last_admitted < cooldown:
            status = "REJECT_COOLDOWN"
        else:
            status = "ADMIT_PAPER_SHADOW"

    decision_body = {
        "contract": "QR_PAPER_CANDIDATE_ADMISSION_V1",
        "status": status,
        "candidate_hash": sealed_candidate_hash,
        "evidence_data_hash": evidence_data_hash,
        "observed_at_utc": observed_at.isoformat().replace("+00:00", "Z"),
        "authority": {
            "live_permission": False,
            "broker_mutation_allowed": False,
            "order_authority": "NONE",
        },
    }
    decision_body["idempotency_hash"] = canonical_sha256(decision_body)
    return decision_body


def assess_continuation(
    *, policy: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate Paper-only continue/stop gates from a sealed metrics record."""

    validate_policy(policy)
    gate = policy["continuation_gate"]
    reasons: list[str] = []
    if int(_finite_float(metrics.get("settlements"), default=0)) < int(
        gate["minimum_settlements"]
    ):
        reasons.append("INSUFFICIENT_SETTLEMENTS")
    if _finite_float(metrics.get("profit_factor_after_cost"), default=0) <= float(
        gate["profit_factor_after_cost_gt"]
    ):
        reasons.append("PROFIT_FACTOR_NOT_ABOVE_ONE")
    if _finite_float(metrics.get("expectancy_after_cost_jpy"), default=0) <= float(
        gate["expectancy_after_cost_gt_jpy"]
    ):
        reasons.append("EXPECTANCY_NOT_POSITIVE")
    if _finite_float(metrics.get("max_drawdown_fraction"), default=1) > _finite_float(
        metrics.get("champion_max_drawdown_fraction"), default=0
    ):
        reasons.append("DRAWDOWN_WORSE_THAN_CHAMPION")
    if len(set(metrics.get("profitable_regime_ids", []))) < int(
        gate["minimum_distinct_regimes"]
    ):
        reasons.append("INSUFFICIENT_REGIME_REPLICATION")
    if metrics.get("base_stress_same_direction") is not True:
        reasons.append("BASE_STRESS_DIRECTION_MISMATCH")
    if metrics.get("shared_feed_chain_match") is not True:
        reasons.append("SHARED_FEED_CHAIN_MISMATCH")

    result = {
        "contract": "QR_PAPER_CANDIDATE_CONTINUATION_V1",
        "status": "CONTINUE_PAPER_SHADOW" if not reasons else "STOP_PAPER_SHADOW",
        "reason_ids": reasons,
        "authority": {
            "live_permission": False,
            "broker_mutation_allowed": False,
            "order_authority": "NONE",
        },
        "metrics_sha256": canonical_sha256(metrics),
    }
    result["decision_sha256"] = canonical_sha256(result)
    return result
