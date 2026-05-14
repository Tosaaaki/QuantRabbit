from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_rabbit.models import OrderIntent, OrderType, RiskIssue


def _sl_free_active() -> bool:
    """SL-free runtime gate (`QR_TRADER_DISABLE_SL_REPAIR=1`).

    User 2026-05-08「短期SHORTでも長期LONGでも両方取れる」+
    `feedback_no_direction_bias_rules.md`: under SL-free the AI trader
    reads market direction in real time, so historical strategy_profile
    direction filters that BLOCK the opposite side become advisory only.
    The mirror lane synthesised by `intent_generator._mirror_lane` would
    otherwise be killed by STRATEGY_PROFILE_MISSING / NOT_ELIGIBLE before
    scoring ever sees it. Under SL-free we downgrade those to WARN so
    the lane reaches LIVE_READY and `trader_brain` can compare LONG vs
    SHORT on its own merits.
    """
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }


@dataclass(frozen=True)
class StrategyProfileEntry:
    pair: str
    direction: str
    method: str | None
    status: str
    required_fix: str


class StrategyProfile:
    def __init__(self, entries: dict[tuple[str, str, str | None], StrategyProfileEntry]) -> None:
        self.entries = entries

    @classmethod
    def load(cls, path: Path) -> "StrategyProfile":
        payload = json.loads(path.read_text())
        entries: dict[tuple[str, str, str | None], StrategyProfileEntry] = {}
        for item in payload.get("profiles", []):
            if not isinstance(item, dict):
                continue
            pair = str(item.get("pair") or "")
            direction = str(item.get("direction") or "")
            if not pair or not direction:
                continue
            method = str(item.get("method") or "").strip().upper() or None
            entries[(pair, direction, method)] = StrategyProfileEntry(
                pair=pair,
                direction=direction,
                method=method,
                status=str(item.get("status") or "WATCH_ONLY"),
                required_fix=str(item.get("required_fix") or ""),
            )
        return cls(entries)

    def validate(self, intent: OrderIntent, *, for_live_send: bool = False) -> tuple[RiskIssue, ...]:
        # Under SL-free the AI trader is the discretionary direction picker;
        # missing profile entries remain advisory for candidate discovery, but
        # strategy statuses that explicitly require fresh evidence must still
        # block live sends. Otherwise MARKET lanes marked MINE_MISSED_EDGE
        # are sent even though the mined fix says "trigger/pending only".
        sl_free = _sl_free_active()
        live_block_severity = "WARN" if sl_free else ("BLOCK" if for_live_send else "WARN")

        method = _intent_method(intent)
        entry = self.entries.get((intent.pair, intent.side.value, method))
        if entry is None and method is not None and _has_method_specific_entry(
            self.entries,
            intent.pair,
            intent.side.value,
        ):
            return (
                RiskIssue(
                    "STRATEGY_METHOD_PROFILE_MISSING",
                    (
                        f"{intent.pair} {intent.side.value} {method} has no method-specific mined profile; "
                        "do not reuse another strategy method's evidence for this lane"
                    ),
                    severity=live_block_severity,
                ),
            )
        if entry is None:
            entry = self.entries.get((intent.pair, intent.side.value, None))
        if entry is None:
            return (
                RiskIssue(
                    "STRATEGY_PROFILE_MISSING",
                    f"{intent.pair} {intent.side.value} is absent from the mined strategy profile",
                    severity=live_block_severity,
                ),
            )
        if entry.status == "CANDIDATE":
            return ()
        if entry.status == "RISK_REPAIR_CANDIDATE":
            return (
                RiskIssue(
                    "STRATEGY_RISK_REPAIR_REQUIRED",
                    f"{entry.pair} {entry.direction} needs risk-repair evidence before live use: {entry.required_fix}",
                    severity=live_block_severity,
                ),
            )
        if entry.status == "MINE_MISSED_EDGE":
            severity = live_block_severity
            if for_live_send and intent.order_type == OrderType.MARKET:
                severity = "BLOCK"
            return (
                RiskIssue(
                    "STRATEGY_TRIGGER_RECEIPT_REQUIRED",
                    f"{entry.pair} {entry.direction} requires trigger/pending-entry receipts before live use: {entry.required_fix}",
                    severity=severity,
                ),
            )
        if entry.status == "BLOCK_UNTIL_NEW_EVIDENCE" and for_live_send:
            return (
                RiskIssue(
                    "STRATEGY_NOT_ELIGIBLE",
                    f"{entry.pair} {entry.direction}{_method_suffix(entry)} is {entry.status}: {entry.required_fix}",
                    severity="BLOCK",
                ),
            )
        return (
            RiskIssue(
                "STRATEGY_NOT_ELIGIBLE",
                f"{entry.pair} {entry.direction}{_method_suffix(entry)} is {entry.status}: {entry.required_fix}",
                severity="WARN" if sl_free else "BLOCK",
            ),
        )


def issues_to_dicts(issues: tuple[RiskIssue, ...] | list[RiskIssue]) -> list[dict[str, Any]]:
    return [issue.__dict__ for issue in issues]


def _intent_method(intent: OrderIntent) -> str | None:
    if intent.market_context is None:
        return None
    return intent.market_context.method.value


def _has_method_specific_entry(
    entries: dict[tuple[str, str, str | None], StrategyProfileEntry],
    pair: str,
    direction: str,
) -> bool:
    return any(
        entry_pair == pair and entry_direction == direction and method is not None
        for entry_pair, entry_direction, method in entries
    )


def _method_suffix(entry: StrategyProfileEntry) -> str:
    return f" {entry.method}" if entry.method else ""
