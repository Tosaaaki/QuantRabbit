from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_POST_TRADE_LEARNING,
    DEFAULT_POST_TRADE_LEARNING_REPORT,
    DEFAULT_POSITION_EXECUTION,
    DEFAULT_TRADER_DECISION,
)
from quant_rabbit.risk import RiskPolicy


@dataclass(frozen=True)
class LearningCandidate:
    source_ref: str
    lane_id: str | None
    pair: str | None
    direction: str | None
    realized_pl_jpy: float | None
    recommendation: str
    confidence: str
    reason: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class PostTradeLearningSummary:
    output_path: Path
    report_path: Path
    status: str
    candidates: int
    profile_update_candidates: int
    blockers: int


class PostTradeLearner:
    """Create receipt-backed learning candidates without silently mutating strategy."""

    def __init__(
        self,
        *,
        outcome_path: Path | None = None,
        live_order_path: Path = DEFAULT_LIVE_ORDER_REQUEST,
        position_execution_path: Path = DEFAULT_POSITION_EXECUTION,
        trader_decision_path: Path = DEFAULT_TRADER_DECISION,
        gpt_decision_path: Path = DEFAULT_GPT_TRADER_DECISION,
        output_path: Path = DEFAULT_POST_TRADE_LEARNING,
        report_path: Path = DEFAULT_POST_TRADE_LEARNING_REPORT,
        max_loss_jpy: float = RiskPolicy().max_loss_jpy,
    ) -> None:
        self.outcome_path = outcome_path
        self.live_order_path = live_order_path
        self.position_execution_path = position_execution_path
        self.trader_decision_path = trader_decision_path
        self.gpt_decision_path = gpt_decision_path
        self.output_path = output_path
        self.report_path = report_path
        self.max_loss_jpy = max_loss_jpy

    def run(self) -> PostTradeLearningSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        live_order = _load_json(self.live_order_path)
        position_execution = _load_json(self.position_execution_path)
        trader_decision = _load_json(self.trader_decision_path)
        gpt_decision = _load_json(self.gpt_decision_path)
        outcomes = _load_outcomes(self.outcome_path)
        candidates = tuple(
            self._candidate_for_outcome(index, outcome, live_order, position_execution, trader_decision, gpt_decision)
            for index, outcome in enumerate(outcomes, 1)
        )
        if not candidates:
            candidates = tuple(_observational_candidates(live_order, position_execution, trader_decision, gpt_decision))
        blockers = tuple(_blockers(candidates))
        payload = {
            "generated_at_utc": generated_at,
            "status": "BLOCKED" if blockers else "READY_FOR_REVIEW",
            "outcome_path": str(self.outcome_path) if self.outcome_path else None,
            "live_order_path": str(self.live_order_path) if self.live_order_path.exists() else None,
            "position_execution_path": str(self.position_execution_path) if self.position_execution_path.exists() else None,
            "trader_decision_path": str(self.trader_decision_path) if self.trader_decision_path.exists() else None,
            "gpt_decision_path": str(self.gpt_decision_path) if self.gpt_decision_path.exists() else None,
            "candidates": [asdict(candidate) for candidate in candidates],
            "profile_update_candidates": [
                asdict(candidate) for candidate in candidates if candidate.recommendation != "NO_PROFILE_CHANGE"
            ],
            "blockers": list(blockers),
        }
        self._write_output(payload)
        self._write_report(payload)
        return PostTradeLearningSummary(
            output_path=self.output_path,
            report_path=self.report_path,
            status=payload["status"],
            candidates=len(candidates),
            profile_update_candidates=len(payload["profile_update_candidates"]),
            blockers=len(blockers),
        )

    def _candidate_for_outcome(
        self,
        index: int,
        outcome: dict[str, Any],
        live_order: dict[str, Any],
        position_execution: dict[str, Any],
        trader_decision: dict[str, Any],
        gpt_decision: dict[str, Any],
    ) -> LearningCandidate:
        lane_id = _first_text(outcome.get("lane_id"), live_order.get("lane_id"), trader_decision.get("selected_lane_id"))
        pair = _first_text(outcome.get("pair"), _order_pair(live_order), _lane_pair(lane_id))
        direction = _first_text(outcome.get("direction"), _lane_direction(lane_id))
        realized = _optional_float(outcome.get("realized_pl_jpy") if "realized_pl_jpy" in outcome else outcome.get("pl_jpy"))
        recommendation, confidence, reason = _recommendation(
            realized=realized,
            close_reason=str(outcome.get("close_reason") or outcome.get("reason") or ""),
            max_loss_jpy=self.max_loss_jpy,
        )
        refs = _evidence_refs(index, lane_id, live_order, position_execution, trader_decision, gpt_decision)
        return LearningCandidate(
            source_ref=f"outcome:{index}",
            lane_id=lane_id,
            pair=pair,
            direction=direction,
            realized_pl_jpy=realized,
            recommendation=recommendation,
            confidence=confidence,
            reason=reason,
            evidence_refs=refs,
        )

    def _write_output(self, payload: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, payload: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Post-Trade Learning Report",
            "",
            f"- Generated at UTC: `{payload['generated_at_utc']}`",
            f"- Status: `{payload['status']}`",
            f"- Candidates: `{len(payload['candidates'])}`",
            f"- Profile update candidates: `{len(payload['profile_update_candidates'])}`",
            "",
            "## Blockers",
            "",
        ]
        if payload["blockers"]:
            lines.extend(f"- {item}" for item in payload["blockers"])
        else:
            lines.append("- none")
        lines.extend(["", "## Candidates", ""])
        for item in payload["candidates"]:
            lines.append(
                f"- `{item['source_ref']}` lane=`{item['lane_id']}` pair=`{item['pair']} {item['direction']}` "
                f"pl=`{item['realized_pl_jpy']}` recommendation=`{item['recommendation']}`"
            )
            lines.append(f"  - reason: {item['reason']}")
            lines.append(f"  - refs: {', '.join(item['evidence_refs']) if item['evidence_refs'] else 'none'}")
        lines.extend(
            [
                "",
                "## Learning Contract",
                "",
                "- Learning memory is advisory and cannot force entries, suppress exits, or resize trades.",
                "- Profile changes are candidates only until backed by receipts and validated by live risk gates.",
                "- Losses beyond the current cap become blockers, not prompt-only lessons.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _observational_candidates(
    live_order: dict[str, Any],
    position_execution: dict[str, Any],
    trader_decision: dict[str, Any],
    gpt_decision: dict[str, Any],
) -> list[LearningCandidate]:
    candidates: list[LearningCandidate] = []
    if live_order:
        candidates.append(
            LearningCandidate(
                source_ref="live_order",
                lane_id=_first_text(live_order.get("lane_id"), trader_decision.get("selected_lane_id")),
                pair=_order_pair(live_order),
                direction=_lane_direction(_first_text(live_order.get("lane_id"), trader_decision.get("selected_lane_id"))),
                realized_pl_jpy=None,
                recommendation="NO_PROFILE_CHANGE",
                confidence="LOW",
                reason="entry receipt exists but no close/fill outcome was supplied",
                evidence_refs=_evidence_refs(0, live_order.get("lane_id"), live_order, position_execution, trader_decision, gpt_decision),
            )
        )
    for action in position_execution.get("actions", []) or []:
        if not isinstance(action, dict) or action.get("request") is None:
            continue
        candidates.append(
            LearningCandidate(
                source_ref=f"position_execution:{action.get('trade_id')}",
                lane_id=None,
                pair=str(action.get("pair") or ""),
                direction=None,
                realized_pl_jpy=None,
                recommendation="NO_PROFILE_CHANGE",
                confidence="LOW",
                reason=f"position protection action `{action.get('management_action')}` was receipted",
                evidence_refs=_evidence_refs(0, None, live_order, position_execution, trader_decision, gpt_decision),
            )
        )
    return candidates


def _recommendation(*, realized: float | None, close_reason: str, max_loss_jpy: float) -> tuple[str, str, str]:
    reason_text = close_reason.upper()
    if realized is None:
        return "NO_PROFILE_CHANGE", "LOW", "no realized outcome supplied"
    if realized < -max_loss_jpy:
        return (
            "BLOCK_UNTIL_NEW_EVIDENCE",
            "HIGH",
            f"realized loss {realized:.0f} JPY breached current {max_loss_jpy:.0f} JPY cap",
        )
    if realized < 0:
        return "RISK_REPAIR_CANDIDATE", "MEDIUM", f"loss {realized:.0f} JPY needs geometry or timing repair"
    if "MISSED" in reason_text or "PREMATURE" in reason_text:
        return "MINE_MISSED_EDGE", "MEDIUM", "profitable or avoided move still contains missed-edge evidence"
    if realized > 0:
        return "REINFORCE_PROFILE", "MEDIUM", f"positive realized outcome {realized:.0f} JPY is receipt-backed"
    return "NO_PROFILE_CHANGE", "LOW", "flat outcome has no profile edge"


def _blockers(candidates: tuple[LearningCandidate, ...]) -> list[str]:
    blockers: list[str] = []
    for candidate in candidates:
        if candidate.recommendation == "BLOCK_UNTIL_NEW_EVIDENCE":
            blockers.append(
                f"{candidate.pair} {candidate.direction} lane {candidate.lane_id} needs blocker review: {candidate.reason}"
            )
    if not candidates:
        blockers.append("no execution, protection, or outcome receipts were available")
    return blockers


def _evidence_refs(
    index: int,
    lane_id: str | None,
    live_order: dict[str, Any],
    position_execution: dict[str, Any],
    trader_decision: dict[str, Any],
    gpt_decision: dict[str, Any],
) -> tuple[str, ...]:
    refs: list[str] = []
    if index:
        refs.append(f"outcome:{index}")
    if lane_id:
        refs.append(f"lane:{lane_id}")
    if live_order:
        refs.append("live_order_request")
    if position_execution:
        refs.append("position_execution")
    if trader_decision:
        refs.append("trader_decision")
    if gpt_decision:
        refs.append("gpt_trader_decision")
    return tuple(dict.fromkeys(refs))


def _load_outcomes(path: Path | None) -> tuple[dict[str, Any], ...]:
    if path is None or not path.exists():
        return ()
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return tuple(item for item in payload if isinstance(item, dict))
    if isinstance(payload.get("outcomes"), list):
        return tuple(item for item in payload["outcomes"] if isinstance(item, dict))
    return (payload,)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _order_pair(payload: dict[str, Any]) -> str | None:
    order = payload.get("order_request") if isinstance(payload.get("order_request"), dict) else {}
    return str(order.get("instrument") or "") or None


def _lane_pair(lane_id: str | None) -> str | None:
    parts = (lane_id or "").split(":")
    return parts[1] if len(parts) >= 4 else None


def _lane_direction(lane_id: str | None) -> str | None:
    parts = (lane_id or "").split(":")
    return parts[2] if len(parts) >= 4 else None


def _first_text(*values: object) -> str | None:
    for value in values:
        if value is None or value == "":
            continue
        return str(value)
    return None


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
