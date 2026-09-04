"""Pure, content-addressed entry decisions and risk-budget sizing.

This module deliberately has no broker, network, environment, clock, or filesystem
adapter.  It turns already-audited numeric allowances into at most one entry
proposal and validates that proposal at an execution boundary.

Numeric policy constants:

* ``MIN_ENTRY_UNITS = 1`` is the broker integer minimum.  There is intentionally
  no 1,000-unit floor or cap.
* ``MAX_DECISION_TTL_SECONDS = 3_600`` bounds how long one sealed observation may
  authorize an entry.  Callers may select any positive shorter TTL.
* ``MAX_SIZING_FACTOR = 1.0`` makes calibration/drawdown/correlation/net-edge
  factors conservative reducers, never risk amplifiers.
* ``MAX_ID_LENGTH = 256`` bounds opaque cycle, epoch, exposure, and reason text.

There is intentionally no target-trade-count divisor and no allocation
multiplier.  Sizing is based on risk remaining and loss at the proposed stop.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any


ENTRY_DECISION_CONTRACT = "QR_ENTRY_DECISION_V1"
ENTRY_SIZING_RECEIPT_CONTRACT = "QR_ENTRY_SIZING_RECEIPT_V1"
SCHEMA_VERSION = 1

ENTRY_ACTIONS = frozenset({"ENTER", "WAIT", "REQUEST_EVIDENCE"})
ENTRY_SIDES = frozenset({"LONG", "SHORT"})
MIN_ENTRY_UNITS = 1
MAX_DECISION_TTL_SECONDS = 60 * 60
MAX_SIZING_FACTOR = 1.0
MAX_ID_LENGTH = 256
DECISION_ID_PREFIX = "qre_"

_FORBIDDEN_SIZING_KEYS = frozenset(
    {
        "allocation_multiplier",
        "target_trade_count",
        "target_trade_count_divisor",
    }
)
_MANUAL_OWNERS = frozenset({"MANUAL", "OPERATOR_MANUAL", "EXTERNAL"})


class EntryDecisionError(ValueError):
    """A fail-closed entry-decision validation error with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def compute_dynamic_units(
    *,
    daily_remaining: float,
    portfolio_allowance: float,
    nav_risk_ceiling: float,
    calibration_factor: float,
    drawdown_factor: float,
    correlation_factor: float,
    net_edge_factor: float,
    loss_per_unit_at_stop: float,
    margin_max_units: float,
    correlation_max_units: float,
    broker_max_units: float,
    exposures: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return a complete sizing receipt for one proposed entry.

    ``margin_max_units`` and ``correlation_max_units`` are gross unit capacities.
    Every existing exposure consumes those capacities, including manual and
    unknown-owner exposure.  Exposure rows are accounting-only and are always
    stamped ``NO_TOUCH``; this module never proposes mutating an existing position.
    """

    budgets = {
        "daily_remaining": _positive_number(daily_remaining, "daily_remaining"),
        "portfolio_allowance": _positive_number(
            portfolio_allowance, "portfolio_allowance"
        ),
        "nav_risk_ceiling": _positive_number(nav_risk_ceiling, "nav_risk_ceiling"),
    }
    factors = {
        "calibration_factor": _sizing_factor(calibration_factor, "calibration_factor"),
        "drawdown_factor": _sizing_factor(drawdown_factor, "drawdown_factor"),
        "correlation_factor": _sizing_factor(correlation_factor, "correlation_factor"),
        "net_edge_factor": _sizing_factor(net_edge_factor, "net_edge_factor"),
    }
    loss = _positive_number(loss_per_unit_at_stop, "loss_per_unit_at_stop")
    gross_caps = {
        "margin_max_units": _positive_number(margin_max_units, "margin_max_units"),
        "correlation_max_units": _positive_number(
            correlation_max_units, "correlation_max_units"
        ),
        "broker_max_units": _positive_number(broker_max_units, "broker_max_units"),
    }
    exposure_rows, margin_used, correlation_used = _normalize_exposures(exposures)

    base_budget = min(budgets.values())
    effective_factor = min(factors.values())
    scaled_risk_budget = base_budget * effective_factor
    risk_formula_units = math.floor(scaled_risk_budget / loss)

    effective_caps = {
        "margin_max_units": math.floor(gross_caps["margin_max_units"] - margin_used),
        "correlation_max_units": math.floor(
            gross_caps["correlation_max_units"] - correlation_used
        ),
        "broker_max_units": math.floor(gross_caps["broker_max_units"]),
    }
    unit_candidates = {"risk_formula_units": risk_formula_units, **effective_caps}
    final_units = min(unit_candidates.values())
    if final_units < MIN_ENTRY_UNITS:
        raise EntryDecisionError(
            "INSUFFICIENT_CAPACITY",
            "risk formula and all post-exposure caps must permit at least one unit",
        )

    budget_limits = _minimum_keys(budgets)
    factor_limits = _minimum_keys(factors)
    unit_limits = _minimum_keys(unit_candidates)
    limiting_reasons = [
        *(f"BUDGET:{name}" for name in budget_limits),
        *(f"FACTOR:{name}" for name in factor_limits),
        *(f"UNITS:{name}" for name in unit_limits),
    ]
    return {
        "contract": ENTRY_SIZING_RECEIPT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "formula": (
            "floor(min(daily_remaining,portfolio_allowance,nav_risk_ceiling)"
            "*min(calibration_factor,drawdown_factor,correlation_factor,net_edge_factor)"
            "/loss_per_unit_at_stop), then min with post-exposure margin, correlation,"
            " and broker unit caps"
        ),
        "risk_budget_components": budgets,
        "base_risk_budget": base_budget,
        "base_risk_limiting_reasons": budget_limits,
        "factor_components": factors,
        "effective_factor": effective_factor,
        "factor_limiting_reasons": factor_limits,
        "scaled_risk_budget": scaled_risk_budget,
        "loss_per_unit_at_stop": loss,
        "risk_formula_units": risk_formula_units,
        "gross_unit_caps": gross_caps,
        "exposure_totals": {
            "margin_units_equivalent": margin_used,
            "correlation_units_equivalent": correlation_used,
        },
        "exposures": exposure_rows,
        "effective_unit_caps": effective_caps,
        "unit_candidates": unit_candidates,
        "final_units": final_units,
        "unit_limiting_reasons": unit_limits,
        "limiting_reasons": limiting_reasons,
        "numeric_policy": {
            "minimum_integer_units": MIN_ENTRY_UNITS,
            "maximum_units": None,
            "maximum_sizing_factor": MAX_SIZING_FACTOR,
        },
    }


def build_entry_decision(
    *,
    action: str,
    cycle_id: str,
    broker_epoch: str,
    evidence_observed_at_utc: datetime,
    proposal: Mapping[str, Any] | None = None,
    requested_evidence: Sequence[str] = (),
    reasons: Sequence[str] = (),
    ttl_seconds: int = 10 * 60,
    created_at_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build and content-address an entry decision with zero or one proposal."""

    normalized_action = str(action).strip().upper()
    if normalized_action not in ENTRY_ACTIONS:
        raise EntryDecisionError("ACTION_INVALID", "unsupported entry action")
    cycle = _bounded_text(cycle_id, "cycle_id")
    epoch = _bounded_text(broker_epoch, "broker_epoch")
    ttl = _ttl(ttl_seconds)
    created = _utc_datetime(created_at_utc or datetime.now(timezone.utc), "created_at_utc")
    observed = _utc_datetime(evidence_observed_at_utc, "evidence_observed_at_utc")
    if observed > created:
        raise EntryDecisionError(
            "EVIDENCE_FROM_FUTURE", "evidence cannot be newer than the decision"
        )

    normalized_reasons = _bounded_text_list(reasons, "reasons")
    normalized_requested = _bounded_text_list(
        requested_evidence, "requested_evidence"
    )
    proposals: list[dict[str, Any]] = []
    if proposal is not None:
        proposals.append(_normalize_proposal(proposal))
    if normalized_action == "ENTER" and len(proposals) != 1:
        raise EntryDecisionError(
            "PROPOSAL_COUNT_INVALID", "ENTER requires exactly one proposal"
        )
    if normalized_action != "ENTER" and proposals:
        raise EntryDecisionError(
            "PROPOSAL_COUNT_INVALID", "non-ENTER decisions require zero proposals"
        )
    if normalized_action == "REQUEST_EVIDENCE" and not normalized_requested:
        raise EntryDecisionError(
            "REQUESTED_EVIDENCE_MISSING",
            "REQUEST_EVIDENCE requires at least one requested item",
        )
    if normalized_action != "REQUEST_EVIDENCE" and normalized_requested:
        raise EntryDecisionError(
            "REQUESTED_EVIDENCE_INVALID",
            "requested evidence is allowed only for REQUEST_EVIDENCE",
        )

    material: dict[str, Any] = {
        "contract": ENTRY_DECISION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "action": normalized_action,
        "cycle_id": cycle,
        "broker_epoch": epoch,
        "created_at_utc": created.isoformat(),
        "evidence_observed_at_utc": observed.isoformat(),
        "ttl_seconds": ttl,
        "expires_at_utc": (created + timedelta(seconds=ttl)).isoformat(),
        "proposals": proposals,
        "requested_evidence": normalized_requested,
        "reasons": normalized_reasons,
    }
    return {"decision_id": decision_id_for(material), **material}


def decision_id_for(decision: Mapping[str, Any]) -> str:
    """Return the qre content address over all outer fields except decision_id."""

    if not isinstance(decision, Mapping):
        raise EntryDecisionError("DECISION_INVALID", "decision must be a mapping")
    material = {key: value for key, value in decision.items() if key != "decision_id"}
    _reject_forbidden_keys(material)
    try:
        raw = json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EntryDecisionError(
            "DECISION_NOT_CANONICAL_JSON", "decision is not finite JSON data"
        ) from exc
    return DECISION_ID_PREFIX + hashlib.sha256(raw).hexdigest()


def validate_entry_decision(
    decision: Mapping[str, Any],
    *,
    expected_cycle_id: str,
    expected_broker_epoch: str,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Revalidate content address, binding, freshness, and sizing at execution time."""

    if not isinstance(decision, Mapping):
        raise EntryDecisionError("DECISION_INVALID", "decision must be a mapping")
    normalized = dict(decision)
    if normalized.get("contract") != ENTRY_DECISION_CONTRACT:
        raise EntryDecisionError("CONTRACT_INVALID", "entry decision contract mismatch")
    if normalized.get("schema_version") != SCHEMA_VERSION:
        raise EntryDecisionError("SCHEMA_INVALID", "entry decision schema mismatch")
    supplied_id = normalized.get("decision_id")
    if not isinstance(supplied_id, str) or supplied_id != decision_id_for(normalized):
        raise EntryDecisionError("DECISION_ID_MISMATCH", "decision content address mismatch")

    cycle = _bounded_text(normalized.get("cycle_id"), "cycle_id")
    epoch = _bounded_text(normalized.get("broker_epoch"), "broker_epoch")
    if cycle != _bounded_text(expected_cycle_id, "expected_cycle_id"):
        raise EntryDecisionError("CYCLE_MISMATCH", "decision belongs to another cycle")
    if epoch != _bounded_text(expected_broker_epoch, "expected_broker_epoch"):
        raise EntryDecisionError(
            "BROKER_EPOCH_MISMATCH", "decision belongs to another broker epoch"
        )

    ttl = _ttl(normalized.get("ttl_seconds"))
    created = _parse_utc(normalized.get("created_at_utc"), "created_at_utc")
    observed = _parse_utc(
        normalized.get("evidence_observed_at_utc"), "evidence_observed_at_utc"
    )
    expires = _parse_utc(normalized.get("expires_at_utc"), "expires_at_utc")
    expected_expiry = created + timedelta(seconds=ttl)
    if expires != expected_expiry:
        raise EntryDecisionError("EXPIRY_MISMATCH", "expires_at_utc does not match TTL")
    if observed > created:
        raise EntryDecisionError(
            "EVIDENCE_FROM_FUTURE", "evidence cannot be newer than the decision"
        )
    current = _utc_datetime(now_utc or datetime.now(timezone.utc), "now_utc")
    if created > current or observed > current:
        raise EntryDecisionError("DECISION_FROM_FUTURE", "decision clock is in the future")
    if current > expires or current > observed + timedelta(seconds=ttl):
        raise EntryDecisionError("DECISION_STALE", "decision or bound evidence is stale")

    action = normalized.get("action")
    if action not in ENTRY_ACTIONS:
        raise EntryDecisionError("ACTION_INVALID", "unsupported entry action")
    proposals = normalized.get("proposals")
    if not isinstance(proposals, list) or len(proposals) > 1:
        raise EntryDecisionError(
            "PROPOSAL_COUNT_INVALID", "decision must contain zero or one proposal"
        )
    if action == "ENTER" and len(proposals) != 1:
        raise EntryDecisionError(
            "PROPOSAL_COUNT_INVALID", "ENTER requires exactly one proposal"
        )
    if action != "ENTER" and proposals:
        raise EntryDecisionError(
            "PROPOSAL_COUNT_INVALID", "non-ENTER decisions require zero proposals"
        )
    if proposals:
        _validate_proposal(proposals[0])
    requested = normalized.get("requested_evidence")
    if not isinstance(requested, list):
        raise EntryDecisionError(
            "REQUESTED_EVIDENCE_INVALID", "requested_evidence must be a list"
        )
    _bounded_text_list(requested, "requested_evidence")
    if action == "REQUEST_EVIDENCE" and not requested:
        raise EntryDecisionError(
            "REQUESTED_EVIDENCE_MISSING",
            "REQUEST_EVIDENCE requires at least one requested item",
        )
    if action != "REQUEST_EVIDENCE" and requested:
        raise EntryDecisionError(
            "REQUESTED_EVIDENCE_INVALID",
            "requested evidence is allowed only for REQUEST_EVIDENCE",
        )
    reasons = normalized.get("reasons")
    if not isinstance(reasons, list):
        raise EntryDecisionError("REASONS_INVALID", "reasons must be a list")
    _bounded_text_list(reasons, "reasons")
    return normalized


def validate_sizing_receipt(receipt: Mapping[str, Any]) -> None:
    """Recompute every material sizing component from a receipt."""

    if not isinstance(receipt, Mapping):
        raise EntryDecisionError("SIZING_RECEIPT_INVALID", "sizing receipt must be a mapping")
    if receipt.get("contract") != ENTRY_SIZING_RECEIPT_CONTRACT:
        raise EntryDecisionError("SIZING_RECEIPT_INVALID", "sizing contract mismatch")
    budgets = _mapping(receipt.get("risk_budget_components"), "risk_budget_components")
    factors = _mapping(receipt.get("factor_components"), "factor_components")
    caps = _mapping(receipt.get("gross_unit_caps"), "gross_unit_caps")
    recomputed = compute_dynamic_units(
        daily_remaining=budgets.get("daily_remaining"),
        portfolio_allowance=budgets.get("portfolio_allowance"),
        nav_risk_ceiling=budgets.get("nav_risk_ceiling"),
        calibration_factor=factors.get("calibration_factor"),
        drawdown_factor=factors.get("drawdown_factor"),
        correlation_factor=factors.get("correlation_factor"),
        net_edge_factor=factors.get("net_edge_factor"),
        loss_per_unit_at_stop=receipt.get("loss_per_unit_at_stop"),
        margin_max_units=caps.get("margin_max_units"),
        correlation_max_units=caps.get("correlation_max_units"),
        broker_max_units=caps.get("broker_max_units"),
        exposures=_sequence_of_mappings(receipt.get("exposures"), "exposures"),
    )
    if _canonical_json(recomputed) != _canonical_json(dict(receipt)):
        raise EntryDecisionError(
            "SIZING_RECEIPT_MISMATCH", "sizing receipt does not reproduce exactly"
        )


def _normalize_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(proposal, Mapping):
        raise EntryDecisionError("PROPOSAL_INVALID", "proposal must be a mapping")
    normalized = dict(proposal)
    _reject_forbidden_keys(normalized)
    _validate_proposal(normalized)
    _canonical_json(normalized)
    return normalized


def _validate_proposal(proposal: Mapping[str, Any]) -> None:
    _bounded_text(proposal.get("pair"), "proposal.pair")
    side = proposal.get("side")
    if side not in ENTRY_SIDES:
        raise EntryDecisionError("SIDE_INVALID", "proposal side must be LONG or SHORT")
    units = proposal.get("units")
    if isinstance(units, bool) or not isinstance(units, int) or units < MIN_ENTRY_UNITS:
        raise EntryDecisionError("UNITS_INVALID", "proposal units must be a positive integer")
    receipt = proposal.get("sizing_receipt")
    validate_sizing_receipt(_mapping(receipt, "proposal.sizing_receipt"))
    if receipt.get("final_units") != units:
        raise EntryDecisionError(
            "UNITS_RECEIPT_MISMATCH", "proposal units do not match sizing receipt"
        )


def _normalize_exposures(
    exposures: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], float, float]:
    if isinstance(exposures, (str, bytes)) or not isinstance(exposures, Sequence):
        raise EntryDecisionError("EXPOSURES_INVALID", "exposures must be a sequence")
    rows: list[dict[str, Any]] = []
    margin_total = 0.0
    correlation_total = 0.0
    for index, raw in enumerate(exposures):
        if not isinstance(raw, Mapping):
            raise EntryDecisionError("EXPOSURE_INVALID", f"exposure {index} is not a mapping")
        reference = _bounded_text(
            raw.get("reference", f"exposure-{index}"), f"exposures[{index}].reference"
        )
        owner_raw = str(
            raw.get("owner") or raw.get("reported_owner") or "UNKNOWN"
        ).strip().upper()
        if owner_raw == "TRADER":
            owner_class = "SYSTEM"
        elif owner_raw in _MANUAL_OWNERS:
            owner_class = "MANUAL"
        else:
            owner_class = "UNKNOWN"
        margin = _nonnegative_number(
            raw.get("margin_units_equivalent", 0.0),
            f"exposures[{index}].margin_units_equivalent",
        )
        correlation = _nonnegative_number(
            raw.get("correlation_units_equivalent", 0.0),
            f"exposures[{index}].correlation_units_equivalent",
        )
        margin_total += margin
        correlation_total += correlation
        if not math.isfinite(margin_total) or not math.isfinite(correlation_total):
            raise EntryDecisionError("NUMBER_NONFINITE", "exposure total is non-finite")
        rows.append(
            {
                "reference": reference,
                "reported_owner": owner_raw,
                "owner_class": owner_class,
                "management_action": "NO_TOUCH",
                "margin_units_equivalent": margin,
                "correlation_units_equivalent": correlation,
            }
        )
    return rows, margin_total, correlation_total


def _positive_number(value: Any, field: str) -> float:
    number = _finite_number(value, field)
    if number <= 0:
        raise EntryDecisionError("NUMBER_NONPOSITIVE", f"{field} must be positive")
    return number


def _nonnegative_number(value: Any, field: str) -> float:
    number = _finite_number(value, field)
    if number < 0:
        raise EntryDecisionError("NUMBER_NEGATIVE", f"{field} must be non-negative")
    return number


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise EntryDecisionError("NUMBER_INVALID", f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EntryDecisionError("NUMBER_INVALID", f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise EntryDecisionError("NUMBER_NONFINITE", f"{field} must be finite")
    return number


def _sizing_factor(value: Any, field: str) -> float:
    factor = _positive_number(value, field)
    if factor > MAX_SIZING_FACTOR:
        raise EntryDecisionError(
            "FACTOR_ABOVE_MAXIMUM", f"{field} must not exceed {MAX_SIZING_FACTOR}"
        )
    return factor


def _minimum_keys(values: Mapping[str, float | int]) -> list[str]:
    minimum = min(values.values())
    return [key for key, value in values.items() if value == minimum]


def _ttl(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EntryDecisionError("TTL_INVALID", "ttl_seconds must be an integer")
    if value <= 0 or value > MAX_DECISION_TTL_SECONDS:
        raise EntryDecisionError(
            "TTL_INVALID",
            f"ttl_seconds must be between 1 and {MAX_DECISION_TTL_SECONDS}",
        )
    return value


def _bounded_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EntryDecisionError("TEXT_INVALID", f"{field} must be non-empty text")
    normalized = value.strip()
    if len(normalized) > MAX_ID_LENGTH:
        raise EntryDecisionError("TEXT_TOO_LONG", f"{field} is too long")
    return normalized


def _bounded_text_list(values: Sequence[str], field: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise EntryDecisionError("TEXT_LIST_INVALID", f"{field} must be a sequence")
    return [_bounded_text(value, f"{field}[{index}]") for index, value in enumerate(values)]


def _utc_datetime(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EntryDecisionError("TIMESTAMP_INVALID", f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise EntryDecisionError("TIMESTAMP_INVALID", f"{field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EntryDecisionError("TIMESTAMP_INVALID", f"{field} is invalid") from exc
    return _utc_datetime(parsed, field)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EntryDecisionError("MAPPING_INVALID", f"{field} must be a mapping")
    return value


def _sequence_of_mappings(value: Any, field: str) -> Sequence[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EntryDecisionError("EXPOSURES_INVALID", f"{field} must be a sequence")
    if not all(isinstance(item, Mapping) for item in value):
        raise EntryDecisionError("EXPOSURES_INVALID", f"{field} contains a non-mapping")
    return value


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in _FORBIDDEN_SIZING_KEYS:
                raise EntryDecisionError(
                    "FORBIDDEN_SIZING_FIELD", f"forbidden sizing field: {key}"
                )
            _reject_forbidden_keys(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _reject_forbidden_keys(nested)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EntryDecisionError(
            "DECISION_NOT_CANONICAL_JSON", "value is not finite JSON data"
        ) from exc
