from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_rabbit.models import OrderIntent, RiskIssue


@dataclass(frozen=True)
class StrategyProfileEntry:
    pair: str
    direction: str
    status: str
    required_fix: str


class StrategyProfile:
    def __init__(self, entries: dict[tuple[str, str], StrategyProfileEntry]) -> None:
        self.entries = entries

    @classmethod
    def load(cls, path: Path) -> "StrategyProfile":
        payload = json.loads(path.read_text())
        entries: dict[tuple[str, str], StrategyProfileEntry] = {}
        for item in payload.get("profiles", []):
            if not isinstance(item, dict):
                continue
            pair = str(item.get("pair") or "")
            direction = str(item.get("direction") or "")
            if not pair or not direction:
                continue
            entries[(pair, direction)] = StrategyProfileEntry(
                pair=pair,
                direction=direction,
                status=str(item.get("status") or "WATCH_ONLY"),
                required_fix=str(item.get("required_fix") or ""),
            )
        return cls(entries)

    def validate(self, intent: OrderIntent, *, for_live_send: bool = False) -> tuple[RiskIssue, ...]:
        entry = self.entries.get((intent.pair, intent.side.value))
        if entry is None:
            severity = "BLOCK" if for_live_send else "WARN"
            return (
                RiskIssue(
                    "STRATEGY_PROFILE_MISSING",
                    f"{intent.pair} {intent.side.value} is absent from the mined strategy profile",
                    severity=severity,
                ),
            )
        if entry.status == "CANDIDATE":
            return ()
        if entry.status == "RISK_REPAIR_CANDIDATE":
            severity = "BLOCK" if for_live_send else "WARN"
            return (
                RiskIssue(
                    "STRATEGY_RISK_REPAIR_REQUIRED",
                    f"{entry.pair} {entry.direction} needs risk-repair evidence before live use: {entry.required_fix}",
                    severity=severity,
                ),
            )
        if entry.status == "MINE_MISSED_EDGE":
            severity = "BLOCK" if for_live_send else "WARN"
            return (
                RiskIssue(
                    "STRATEGY_TRIGGER_RECEIPT_REQUIRED",
                    f"{entry.pair} {entry.direction} requires trigger/pending-entry receipts before live use: {entry.required_fix}",
                    severity=severity,
                ),
            )
        return (
            RiskIssue(
                "STRATEGY_NOT_ELIGIBLE",
                f"{entry.pair} {entry.direction} is {entry.status}: {entry.required_fix}",
            ),
        )


def issues_to_dicts(issues: tuple[RiskIssue, ...] | list[RiskIssue]) -> list[dict[str, Any]]:
    return [issue.__dict__ for issue in issues]
