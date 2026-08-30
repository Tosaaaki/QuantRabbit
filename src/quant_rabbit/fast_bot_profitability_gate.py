"""Fail-closed profitability evidence gate for shadow research candidates.

The gate is intentionally separate from live admission.  It can reject a
negative candidate or retain a positive-but-thin candidate for additional
zero-authority observation, but it cannot create an order intent, risk
approval, supervision receipt, or live permission.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Mapping


PROFITABILITY_EVIDENCE_CONTRACT = "QR_FAST_BOT_PROFITABILITY_EVIDENCE_V1"
PROFITABILITY_GATE_CONTRACT = "QR_FAST_BOT_PROFITABILITY_GATE_V1"

DEFAULT_THRESHOLDS = {
    "minimum_samples": 100,
    "minimum_active_days": 10,
    "minimum_profit_factor": 1.25,
    "minimum_pessimistic_expectancy_pips": 0.0,
    "minimum_positive_day_rate": 2.0 / 3.0,
    "maximum_daily_sample_share": 0.70,
}


def seal_profitability_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a broker-free profitability evidence receipt."""

    body = {
        key: item
        for key, item in value.items()
        if key != "evidence_sha256"
    }
    return {**body, "evidence_sha256": _canonical_sha(body)}


def build_profitability_evidence(
    *,
    lane_id: str,
    pair: str,
    side: str,
    method: str,
    order_type: str,
    metrics: Mapping[str, Any],
    source_artifact_sha256: str,
    generated_at_utc: datetime,
    evidence_end_utc: datetime,
    rank_only: bool = False,
) -> dict[str, Any]:
    """Build a structured evidence receipt without changing trading state."""

    generated = _aware_utc(generated_at_utc)
    evidence_end = _aware_utc(evidence_end_utc)
    body = {
        "contract": PROFITABILITY_EVIDENCE_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": generated.isoformat(),
        "evidence_end_utc": evidence_end.isoformat(),
        "lane_id": _bounded_text(lane_id),
        "pair": str(pair or "").upper(),
        "side": str(side or "").upper(),
        "method": str(method or "").upper(),
        "order_type": str(order_type or "").upper(),
        "rank_only": bool(rank_only),
        "sample_count": metrics.get("sample_count"),
        "active_days": metrics.get("active_days"),
        "profit_factor": metrics.get("profit_factor"),
        "net_pips": metrics.get("net_pl_pips", metrics.get("net_pips")),
        "expectancy_pips": metrics.get("expectancy_pips"),
        "pessimistic_expectancy_pips": metrics.get(
            "pessimistic_expectancy_pips"
        ),
        "positive_day_rate": metrics.get("positive_day_rate"),
        "max_daily_sample_share": metrics.get("max_daily_sample_share"),
        "spread_included": metrics.get("spread_included"),
        "source_artifact_sha256": source_artifact_sha256,
        "execution_authority": "NONE",
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_policy": "NO_TOUCH",
    }
    return seal_profitability_evidence(body)


def assess_profitability_evidence(
    evidence: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify evidence for shadow research, never for live promotion.

    ``SHADOW_FORWARD_OBSERVATION_READY`` means only that a candidate may be
    observed prospectively with zero execution authority.  It is not a live
    admission and deliberately has no path to ``LiveOrderGateway``.
    """

    limits = _normalize_thresholds(thresholds or DEFAULT_THRESHOLDS)
    blockers: list[str] = []
    invalid = _evidence_invalid_reasons(evidence)
    metrics: dict[str, float | int | bool | None] = {}
    if not invalid:
        try:
            metrics = {
                "sample_count": _nonnegative_int(evidence.get("sample_count")),
                "active_days": _nonnegative_int(evidence.get("active_days")),
                "profit_factor": _finite(evidence.get("profit_factor")),
                "net_pips": _finite(evidence.get("net_pips")),
                "expectancy_pips": _finite(evidence.get("expectancy_pips")),
                "pessimistic_expectancy_pips": _finite(
                    evidence.get("pessimistic_expectancy_pips")
                ),
                "positive_day_rate": _rate(evidence.get("positive_day_rate")),
                "max_daily_sample_share": _rate(
                    evidence.get("max_daily_sample_share")
                ),
                "spread_included": evidence.get("spread_included") is True,
            }
        except (TypeError, ValueError):
            invalid.append("METRICS_INVALID")

    if invalid:
        status = "REJECT_INVALID_EVIDENCE"
        blockers = sorted(set(invalid))
    else:
        if not metrics["spread_included"]:
            blockers.append("SPREAD_NOT_INCLUDED")
        if float(metrics["profit_factor"] or 0.0) <= 1.0:
            blockers.append("PROFIT_FACTOR_NOT_ABOVE_ONE")
        if float(metrics["net_pips"] or 0.0) <= 0.0:
            blockers.append("NET_PIPS_NOT_POSITIVE")
        if float(metrics["expectancy_pips"] or 0.0) <= 0.0:
            blockers.append("EXPECTANCY_NOT_POSITIVE")
        if float(metrics["pessimistic_expectancy_pips"] or 0.0) <= float(
            limits["minimum_pessimistic_expectancy_pips"]
        ):
            blockers.append("PESSIMISTIC_EXPECTANCY_NOT_POSITIVE")

        negative_edge = any(
            reason in blockers
            for reason in (
                "SPREAD_NOT_INCLUDED",
                "PROFIT_FACTOR_NOT_ABOVE_ONE",
                "NET_PIPS_NOT_POSITIVE",
                "EXPECTANCY_NOT_POSITIVE",
                "PESSIMISTIC_EXPECTANCY_NOT_POSITIVE",
            )
        )
        if negative_edge:
            status = "REJECT_NEGATIVE_EXPECTANCY"
        else:
            if int(metrics["sample_count"] or 0) < int(limits["minimum_samples"]):
                blockers.append("INSUFFICIENT_SAMPLES")
            if int(metrics["active_days"] or 0) < int(limits["minimum_active_days"]):
                blockers.append("INSUFFICIENT_ACTIVE_DAYS")
            if float(metrics["profit_factor"] or 0.0) < float(
                limits["minimum_profit_factor"]
            ):
                blockers.append("PROFIT_FACTOR_BELOW_FORWARD_FLOOR")
            if float(metrics["positive_day_rate"] or 0.0) < float(
                limits["minimum_positive_day_rate"]
            ):
                blockers.append("POSITIVE_DAY_RATE_BELOW_FLOOR")
            if float(metrics["max_daily_sample_share"] or 0.0) > float(
                limits["maximum_daily_sample_share"]
            ):
                blockers.append("DAILY_SAMPLE_CONCENTRATION_TOO_HIGH")
            status = (
                "COLLECT_MORE_INDEPENDENT_DAYS"
                if blockers
                else "SHADOW_FORWARD_OBSERVATION_READY"
            )

    body = {
        "contract": PROFITABILITY_GATE_CONTRACT,
        "schema_version": 1,
        "evidence_sha256": evidence.get("evidence_sha256"),
        "lane_id": evidence.get("lane_id"),
        "pair": evidence.get("pair"),
        "side": evidence.get("side"),
        "method": evidence.get("method"),
        "order_type": evidence.get("order_type"),
        "rank_only": evidence.get("rank_only") is True,
        "status": status,
        "blockers": sorted(set(blockers)),
        "thresholds": limits,
        "metrics": metrics,
        "shadow_observation_allowed": status
        in {"COLLECT_MORE_INDEPENDENT_DAYS", "SHADOW_FORWARD_OBSERVATION_READY"},
        "primary_trading_candidate_allowed": status
        == "SHADOW_FORWARD_OBSERVATION_READY",
        "promotion_allowed": False,
        "live_permission": False,
        "execution_authority": "NONE",
        "broker_mutation_allowed": False,
        "live_order_gateway_invocation_count": 0,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_policy": "NO_TOUCH",
    }
    return {**body, "gate_sha256": _canonical_sha(body)}


def _evidence_invalid_reasons(value: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if value.get("contract") != PROFITABILITY_EVIDENCE_CONTRACT:
        reasons.append("CONTRACT_INVALID")
    seal = value.get("evidence_sha256")
    body = {key: item for key, item in value.items() if key != "evidence_sha256"}
    if not isinstance(seal, str) or seal != _canonical_sha(body):
        reasons.append("EVIDENCE_SEAL_INVALID")
    if not _valid_sha(str(value.get("source_artifact_sha256") or "")):
        reasons.append("SOURCE_ARTIFACT_SHA_INVALID")
    if value.get("shadow_only") is not True or value.get("live_permission") is not False:
        reasons.append("EVIDENCE_AUTHORITY_INVALID")
    if value.get("execution_authority") != "NONE":
        reasons.append("EXECUTION_AUTHORITY_INVALID")
    if value.get("broker_mutation_allowed") is not False:
        reasons.append("BROKER_MUTATION_AUTHORITY_INVALID")
    if value.get("external_order_attempts") != 0 or value.get("external_orders") != 0:
        reasons.append("EXTERNAL_ORDER_COUNT_NONZERO")
    if value.get("manual_tagless_policy") != "NO_TOUCH":
        reasons.append("MANUAL_TAGLESS_POLICY_INVALID")
    if value.get("side") not in {"LONG", "SHORT"}:
        reasons.append("SIDE_INVALID")
    if value.get("order_type") not in {"MARKET", "LIMIT", "STOP"}:
        reasons.append("ORDER_TYPE_INVALID")
    if not _bounded_text(value.get("lane_id")):
        reasons.append("LANE_ID_INVALID")
    try:
        if _parse_utc(value.get("evidence_end_utc")) > _parse_utc(
            value.get("generated_at_utc")
        ):
            reasons.append("EVIDENCE_TIME_INVALID")
    except (TypeError, ValueError):
        reasons.append("EVIDENCE_TIME_INVALID")
    return reasons


def _normalize_thresholds(value: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        "minimum_samples": _positive_int(value.get("minimum_samples")),
        "minimum_active_days": _positive_int(value.get("minimum_active_days")),
        "minimum_profit_factor": _positive_float(value.get("minimum_profit_factor")),
        "minimum_pessimistic_expectancy_pips": _finite(
            value.get("minimum_pessimistic_expectancy_pips")
        ),
        "minimum_positive_day_rate": _rate(value.get("minimum_positive_day_rate")),
        "maximum_daily_sample_share": _rate(value.get("maximum_daily_sample_share")),
    }


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _valid_sha(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _bounded_text(value: Any) -> str:
    text = str(value or "").strip()
    return text if 0 < len(text) <= 256 else ""


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("number must be finite")
    return number


def _positive_float(value: Any) -> float:
    number = _finite(value)
    if number <= 0.0:
        raise ValueError("number must be positive")
    return number


def _positive_int(value: Any) -> int:
    number = int(value)
    if isinstance(value, bool) or number <= 0 or float(value) != number:
        raise ValueError("number must be a positive integer")
    return number


def _nonnegative_int(value: Any) -> int:
    number = int(value)
    if isinstance(value, bool) or number < 0 or float(value) != number:
        raise ValueError("number must be a nonnegative integer")
    return number


def _rate(value: Any) -> float:
    number = _finite(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError("rate must be within [0, 1]")
    return number


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone aware")
    return value.astimezone(timezone.utc)
