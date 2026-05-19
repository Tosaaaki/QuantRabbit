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
    `feedback_no_direction_bias_rules.md`: under SL-free the AI trader can
    compare current LONG/SHORT receipts, but a WATCH_ONLY or missing mined
    profile is still not live permission. Pending trigger / risk-repair
    receipts may downgrade to advisory when they carry fresh executable
    geometry; observation-only lanes must stay out of LIVE_READY.
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
        # fresh trigger / risk-repair receipts can be compared by the trader,
        # but missing or WATCH_ONLY mined evidence is still not executable.
        # Otherwise an observation lane can become LIVE_READY just because the
        # current risk geometry happens to pass.
        sl_free = _sl_free_active()
        evidence_repair_severity = "WARN" if sl_free else ("BLOCK" if for_live_send else "WARN")
        strict_live_severity = "BLOCK" if for_live_send else "WARN"

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
                    severity=strict_live_severity,
                ),
            )
        if entry is None:
            entry = self.entries.get((intent.pair, intent.side.value, None))
        if entry is None:
            synthetic_severity = _synthetic_missing_profile_severity(
                intent,
                entries=self.entries,
                method=method,
                sl_free=sl_free,
                for_live_send=for_live_send,
            )
            return (
                RiskIssue(
                    "STRATEGY_PROFILE_MISSING",
                    f"{intent.pair} {intent.side.value} is absent from the mined strategy profile",
                    severity=synthetic_severity or strict_live_severity,
                ),
            )
        if entry.status == "CANDIDATE":
            return ()
        if entry.status == "RISK_REPAIR_CANDIDATE":
            return (
                RiskIssue(
                    "STRATEGY_RISK_REPAIR_REQUIRED",
                    f"{entry.pair} {entry.direction} needs risk-repair evidence before live use: {entry.required_fix}",
                    severity=evidence_repair_severity,
                ),
            )
        if entry.status == "MINE_MISSED_EDGE":
            severity = evidence_repair_severity
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
        if entry.status == "WATCH_ONLY":
            severity = _watch_only_severity(intent, sl_free=sl_free, for_live_send=for_live_send)
            return (
                RiskIssue(
                    "STRATEGY_NOT_ELIGIBLE",
                    f"{entry.pair} {entry.direction}{_method_suffix(entry)} is {entry.status}: {entry.required_fix}",
                    severity=severity,
                ),
            )
        return (
            RiskIssue(
                "STRATEGY_NOT_ELIGIBLE",
                f"{entry.pair} {entry.direction}{_method_suffix(entry)} is {entry.status}: {entry.required_fix}",
                severity=strict_live_severity if for_live_send else ("WARN" if sl_free else "BLOCK"),
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


def _synthetic_missing_profile_severity(
    intent: OrderIntent,
    *,
    entries: dict[tuple[str, str, str | None], StrategyProfileEntry],
    method: str | None,
    sl_free: bool,
    for_live_send: bool,
) -> str | None:
    if not (sl_free and for_live_send):
        return None
    metadata = intent.metadata or {}
    if metadata.get("forecast_seed"):
        return "WARN"
    mirror_of = str(metadata.get("mirror_of") or "").strip().upper()
    if mirror_of not in {"LONG", "SHORT"}:
        return None
    source = entries.get((intent.pair, mirror_of, method))
    if source is None:
        source = entries.get((intent.pair, mirror_of, None))
    if source is None:
        return None
    if source.status in {"CANDIDATE", "RISK_REPAIR_CANDIDATE", "MINE_MISSED_EDGE"}:
        return "WARN"
    return None


def _watch_only_severity(intent: OrderIntent, *, sl_free: bool, for_live_send: bool) -> str:
    """Return profile severity for a WATCH_ONLY mined lane.

    WATCH_ONLY remains a hard live blocker for ordinary campaign/mirror lanes:
    observation-only history is not executable permission. The narrow exception
    is a fresh forecast-seed pending trigger under SL-free runtime. That path is
    not reusing WATCH_ONLY history; it is current prediction + structure
    evidence asking the gateway to place a non-market trigger. MARKET entries
    still require stronger mined evidence, because otherwise WATCH_ONLY would
    again become "buy/sell now" permission.
    """
    if not for_live_send:
        return "WARN" if sl_free else "BLOCK"
    if not sl_free:
        return "BLOCK"
    metadata = intent.metadata or {}
    if metadata.get("forecast_seed") and intent.order_type != OrderType.MARKET:
        return "WARN"
    return "BLOCK"


def _method_suffix(entry: StrategyProfileEntry) -> str:
    return f" {entry.method}" if entry.method else ""
