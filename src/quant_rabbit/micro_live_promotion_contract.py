"""T1 micro-live promotion contract (shadow -> bounded execution-data tier).

The missing bridge between shadow evidence and scaled capital: T1 grants a
bounded NAV-fraction risk pool whose purpose is execution-data collection,
never profit maximization.  This module only builds and validates sealed
contract artifacts; it grants no live permission by itself.  Activation
additionally requires a separate operator-authored approval artifact bound
to the exact contract digest, and every order path still runs through
RiskEngine and LiveOrderGateway.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping

CONTRACT = "QR_T1_MICRO_LIVE_PROMOTION_CONTRACT_V1"
APPROVAL_CONTRACT = "QR_T1_MICRO_LIVE_OPERATOR_APPROVAL_V1"
APPROVAL_STATEMENT = (
    "I approve bounded T1 micro-live execution-data collection for this "
    "exact contract digest."
)
# T0 -> T1 admission thresholds: deliberately below the T2 promotion gate
# (>=100 fills / PF>=1.25 / positive LB) because T1 exists to collect the
# execution evidence T2 needs, not to prove the edge.
MIN_VALID_FILLS = 100
MIN_FILLED_DAYS = 10
MIN_STRESSED_PROFIT_FACTOR = 1.05
NAV_RISK_POOL_FRACTION = 0.02
MAX_RISK_PER_TRADE_FRACTION = 0.0025
DAILY_STOP_FRACTION = 0.01


class MicroLivePromotionError(ValueError):
    """Raised when a T1 contract or approval fails validation."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_micro_live_promotion_contract(
    *,
    lane_id: str,
    shadow_scorecard: Mapping[str, Any],
    scorecard_sha256: str,
    declared_at_utc: datetime,
) -> dict[str, Any]:
    """Seal a T1 candidacy from a T0 shadow scorecard, without live permission."""

    if not isinstance(lane_id, str) or not lane_id.strip():
        raise MicroLivePromotionError("lane identity is required")
    if not isinstance(shadow_scorecard, Mapping):
        raise MicroLivePromotionError("shadow scorecard must be an object")
    if declared_at_utc.tzinfo is None:
        raise MicroLivePromotionError("declaration clock must be timezone-aware")
    fills = shadow_scorecard.get("valid_fill_count")
    filled_days = shadow_scorecard.get("filled_day_count")
    stressed_pf = shadow_scorecard.get("stressed_profit_factor")
    for label, value in (
        ("valid_fill_count", fills),
        ("filled_day_count", filled_days),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MicroLivePromotionError(f"scorecard {label} is invalid")
    if (
        isinstance(stressed_pf, bool)
        or not isinstance(stressed_pf, (int, float))
        or not stressed_pf == stressed_pf  # NaN guard
    ):
        raise MicroLivePromotionError("scorecard stressed_profit_factor is invalid")
    eligible = bool(
        fills >= MIN_VALID_FILLS
        and filled_days >= MIN_FILLED_DAYS
        and float(stressed_pf) >= MIN_STRESSED_PROFIT_FACTOR
    )
    if not eligible:
        raise MicroLivePromotionError(
            "T0 shadow evidence does not meet the T1 admission floor"
        )

    body: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "lane_id": lane_id,
        "declared_at_utc": declared_at_utc.astimezone(timezone.utc).isoformat(),
        "t0_scorecard_sha256": scorecard_sha256,
        "t0_evidence": {
            "valid_fill_count": int(fills),
            "filled_day_count": int(filled_days),
            "stressed_profit_factor": float(stressed_pf),
        },
        "admission_floor": {
            "min_valid_fills": MIN_VALID_FILLS,
            "min_filled_days": MIN_FILLED_DAYS,
            "min_stressed_profit_factor": MIN_STRESSED_PROFIT_FACTOR,
        },
        "risk_budget": {
            "nav_risk_pool_fraction": NAV_RISK_POOL_FRACTION,
            "max_risk_per_trade_fraction": MAX_RISK_PER_TRADE_FRACTION,
            "daily_stop_fraction": DAILY_STOP_FRACTION,
        },
        "purpose": "EXECUTION_DATA_COLLECTION_NOT_PROFIT_MAXIMIZATION",
        "risk_engine_required": True,
        "live_order_gateway_required": True,
        "operator_approval_required": True,
        "operator_approval_statement": APPROVAL_STATEMENT,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return {**body, "contract_sha256": _canonical_sha(body)}


def validate_micro_live_activation(
    contract: Mapping[str, Any], approval: Mapping[str, Any]
) -> dict[str, Any]:
    """Check a contract/approval pair; never mutates authority fields.

    Returns a sealed activation-review artifact stating whether the pair is
    coherent.  Even a coherent pair grants nothing by itself: the runtime
    gateway must independently re-run this validation before any T1 order.
    """

    if not isinstance(contract, Mapping) or not isinstance(approval, Mapping):
        raise MicroLivePromotionError("contract and approval must be objects")
    body = {k: v for k, v in contract.items() if k != "contract_sha256"}
    if contract.get("contract_sha256") != _canonical_sha(body):
        raise MicroLivePromotionError("T1 contract digest is invalid")
    if contract.get("contract") != CONTRACT:
        raise MicroLivePromotionError("T1 contract identity is invalid")
    if (
        contract.get("live_permission") is not False
        or contract.get("order_authority") != "NONE"
        or contract.get("operator_approval_required") is not True
    ):
        raise MicroLivePromotionError("T1 contract authority fields were tampered")
    approval_body = {k: v for k, v in approval.items() if k != "approval_sha256"}
    if approval.get("approval_sha256") != _canonical_sha(approval_body):
        raise MicroLivePromotionError("operator approval digest is invalid")
    if approval.get("contract") != APPROVAL_CONTRACT:
        raise MicroLivePromotionError("operator approval identity is invalid")
    if approval.get("approved_contract_sha256") != contract["contract_sha256"]:
        raise MicroLivePromotionError(
            "operator approval is not bound to this exact contract"
        )
    if approval.get("statement") != APPROVAL_STATEMENT:
        raise MicroLivePromotionError("operator approval statement is not exact")
    if not str(approval.get("operator") or "").strip():
        raise MicroLivePromotionError("operator approval author is missing")
    review_body = {
        "contract": "QR_T1_MICRO_LIVE_ACTIVATION_REVIEW_V1",
        "contract_sha256": contract["contract_sha256"],
        "approval_sha256": approval["approval_sha256"],
        "pair_coherent": True,
        "grants_live_permission": False,
        "runtime_revalidation_required": True,
    }
    return {**review_body, "review_sha256": _canonical_sha(review_body)}
