from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping


REQUIRED_STAGES = (
    "pricing",
    "candidate_order",
    "fillability",
    "slippage_fee_financing",
    "margin_exposure_concurrency",
    "exit_unwind",
)


@dataclass(frozen=True)
class RiskPolicy:
    per_trade_risk_fraction: float = 0.0025
    daily_gross_loss_fraction: float = 0.01
    drawdown_lock_fraction: float = 0.05

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True)
class DecisionInput:
    decision_id: str
    decision_time: str
    source_sha: str
    stage_coverage: Mapping[str, bool]
    equity_jpy: float | None
    peak_equity_jpy: float | None
    daily_gross_loss_spent_jpy: float | None
    candidate_loss_bound_jpy: float | None
    expected_after_cost_lcb_jpy: float | None
    existing_position: bool = False


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def evaluate(decision: DecisionInput, policy: RiskPolicy = RiskPolicy()) -> dict[str, Any]:
    """Return a deterministic research-only permission receipt.

    Realized outcome is deliberately absent from DecisionInput.  Unknown values
    never become zero and every blocked reason includes a reopening condition.
    """

    missing_stages = [stage for stage in REQUIRED_STAGES if decision.stage_coverage.get(stage) is not True]
    reasons: list[str] = []
    reopen: list[str] = []

    if decision.existing_position:
        action = "MANAGE"
        reasons.append("EXISTING_POSITION_REQUIRES_MANAGEMENT")
        reopen.append("close or explicitly manage the existing position under its bound exit policy")
    else:
        action = "TRADE"

    if missing_stages:
        action = "WAIT"
        reasons.extend(f"{stage.upper()}_EVIDENCE_MISSING" for stage in missing_stages)
        reopen.extend(f"persist causal decision-time {stage} evidence" for stage in missing_stages)

    if decision.equity_jpy is None or decision.equity_jpy <= 0:
        action = "WAIT"
        reasons.append("EQUITY_EVIDENCE_MISSING")
        reopen.append("persist positive decision-time broker equity")

    if decision.peak_equity_jpy is None or decision.peak_equity_jpy <= 0:
        action = "WAIT"
        reasons.append("PEAK_EQUITY_EVIDENCE_MISSING")
        reopen.append("persist campaign peak equity before this decision")

    if decision.daily_gross_loss_spent_jpy is None or decision.daily_gross_loss_spent_jpy < 0:
        action = "WAIT"
        reasons.append("DAILY_GROSS_LOSS_EVIDENCE_MISSING")
        reopen.append("persist prior-only non-refillable gross realized loss spend")

    if decision.candidate_loss_bound_jpy is None or decision.candidate_loss_bound_jpy < 0:
        action = "WAIT"
        reasons.append("CANDIDATE_LOSS_BOUND_MISSING")
        reopen.append("derive side-correct worst-case candidate loss from causal entry and invalidation")

    if decision.expected_after_cost_lcb_jpy is None:
        if action == "TRADE":
            action = "SKIP"
        reasons.append("AFTER_COST_LCB_MISSING")
        reopen.append("establish a TRAIN-fixed positive after-cost lower confidence bound")
    elif decision.expected_after_cost_lcb_jpy <= 0:
        if action == "TRADE":
            action = "SKIP"
        reasons.append("AFTER_COST_LCB_NONPOSITIVE")
        reopen.append("establish a TRAIN-fixed positive after-cost lower confidence bound")

    per_trade_cap = None
    daily_budget = None
    remaining_daily_budget = None
    drawdown_fraction = None
    if decision.equity_jpy is not None and decision.equity_jpy > 0:
        per_trade_cap = decision.equity_jpy * policy.per_trade_risk_fraction
        daily_budget = decision.equity_jpy * policy.daily_gross_loss_fraction
        if decision.daily_gross_loss_spent_jpy is not None and decision.daily_gross_loss_spent_jpy >= 0:
            remaining_daily_budget = max(0.0, daily_budget - decision.daily_gross_loss_spent_jpy)
            if remaining_daily_budget <= 0:
                action = "WAIT"
                reasons.append("DAILY_GROSS_LOSS_BUDGET_EXHAUSTED")
                reopen.append("wait for the next preregistered campaign day; profits do not refill this budget")
        if decision.candidate_loss_bound_jpy is not None and decision.candidate_loss_bound_jpy >= 0:
            effective_cap = per_trade_cap if remaining_daily_budget is None else min(per_trade_cap, remaining_daily_budget)
            if decision.candidate_loss_bound_jpy > effective_cap:
                action = "WAIT"
                reasons.append("CANDIDATE_LOSS_EXCEEDS_AVAILABLE_CAP")
                reopen.append("reduce units or tighten causal geometry without exceeding the evidence-bound cap")

    if (
        decision.equity_jpy is not None
        and decision.equity_jpy > 0
        and decision.peak_equity_jpy is not None
        and decision.peak_equity_jpy > 0
    ):
        drawdown_fraction = max(0.0, 1.0 - decision.equity_jpy / decision.peak_equity_jpy)
        if drawdown_fraction >= policy.drawdown_lock_fraction:
            action = "WAIT"
            reasons.append("DRAWDOWN_LOCK_REACHED")
            reopen.append("operator review plus new forward evidence; do not auto-reset from a later winner")

    decision_payload = {
        "decision_id": decision.decision_id,
        "decision_time": decision.decision_time,
        "source_sha": decision.source_sha,
        "stage_coverage": {stage: decision.stage_coverage.get(stage) is True for stage in REQUIRED_STAGES},
        "equity_jpy": decision.equity_jpy,
        "peak_equity_jpy": decision.peak_equity_jpy,
        "daily_gross_loss_spent_jpy": decision.daily_gross_loss_spent_jpy,
        "candidate_loss_bound_jpy": decision.candidate_loss_bound_jpy,
        "expected_after_cost_lcb_jpy": decision.expected_after_cost_lcb_jpy,
        "existing_position": decision.existing_position,
        "policy": asdict(policy),
    }
    receipt = {
        "contract": "CAPITAL_PRESERVATION_GATE_V1",
        "decision_id": decision.decision_id,
        "decision_time": decision.decision_time,
        "action": action,
        "new_exposure_permitted": action == "TRADE",
        "reason_codes": sorted(set(reasons)),
        "reopen_conditions": sorted(set(reopen)),
        "missing_stages": missing_stages,
        "risk": {
            "per_trade_cap_jpy": per_trade_cap,
            "daily_budget_jpy": daily_budget,
            "remaining_daily_budget_jpy": remaining_daily_budget,
            "drawdown_fraction": drawdown_fraction,
        },
        "decision_input_sha256": _canonical_sha(decision_payload),
        "realized_outcome_used": False,
        "live_permission_granted": False,
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    return receipt
