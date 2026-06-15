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
    target_reward_risk: float | None = None
    positive_best_jpy: float | None = None
    positive_tail_jpy: float | None = None
    positive_evidence_n: int | None = None
    live_net_jpy: float | None = None
    live_avg_jpy: float | None = None
    live_n: int | None = None
    pretrade_net_jpy: float | None = None
    pretrade_avg_jpy: float | None = None
    pretrade_n: int | None = None
    seat_net_jpy: float | None = None
    seat_win_rate_pct: float | None = None
    seat_pl_n: int | None = None
    top_block_reasons: tuple[str, ...] = ()


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
                target_reward_risk=_optional_float(item.get("target_reward_risk")),
                positive_best_jpy=_optional_float(item.get("positive_best_jpy")),
                positive_tail_jpy=_optional_float(item.get("positive_tail_jpy")),
                positive_evidence_n=_optional_int(item.get("positive_evidence_n")),
                live_net_jpy=_optional_float(item.get("live_net_jpy")),
                live_avg_jpy=_optional_float(item.get("live_avg_jpy")),
                live_n=_optional_int(item.get("live_n")),
                pretrade_net_jpy=_optional_float(item.get("pretrade_net_jpy")),
                pretrade_avg_jpy=_optional_float(item.get("pretrade_avg_jpy")),
                pretrade_n=_optional_int(item.get("pretrade_n")),
                seat_net_jpy=_optional_float(item.get("seat_net_jpy")),
                seat_win_rate_pct=_optional_float(item.get("seat_win_rate_pct")),
                seat_pl_n=_optional_int(item.get("seat_pl_n")),
                top_block_reasons=_string_tuple(item.get("top_block_reasons")),
            )
        return cls(entries)

    def issue_evidence(self, intent: OrderIntent) -> dict[str, Any]:
        method = _intent_method(intent)
        exact = self.entries.get((intent.pair, intent.side.value, method))
        if exact is not None:
            return _entry_evidence(
                exact,
                profile_match="method_specific" if method is not None else "pair_side",
                requested_method=method,
            )

        fallback = self.entries.get((intent.pair, intent.side.value, None))
        method_entries = _method_specific_entries(self.entries, intent.pair, intent.side.value)
        if method is not None and method_entries:
            evidence: dict[str, Any] = {
                "profile_match": "method_specific_missing",
                "profile_pair": intent.pair,
                "profile_direction": intent.side.value,
                "requested_method": method,
                "available_methods": sorted(entry.method for entry in method_entries if entry.method),
            }
            if fallback is not None:
                evidence["fallback_pair_side_profile"] = _entry_evidence(
                    fallback,
                    profile_match="pair_side_fallback_not_used",
                    requested_method=method,
                )
            return evidence

        if fallback is not None:
            return _entry_evidence(
                fallback,
                profile_match="pair_side_fallback",
                requested_method=method,
            )

        return {
            "profile_match": "missing",
            "profile_pair": intent.pair,
            "profile_direction": intent.side.value,
            "requested_method": method,
        }

    def validate(self, intent: OrderIntent, *, for_live_send: bool = False) -> tuple[RiskIssue, ...]:
        # Under SL-free the AI trader is the discretionary direction picker;
        # fresh trigger / risk-repair receipts can be compared by the trader,
        # but missing or WATCH_ONLY mined evidence is still not executable.
        # Otherwise an observation lane can become LIVE_READY just because the
        # current risk geometry happens to pass.
        sl_free = _sl_free_active()
        evidence_repair_severity = "WARN" if sl_free else ("BLOCK" if for_live_send else "WARN")
        strict_live_severity = "BLOCK" if for_live_send else "WARN"

        if not self.entries:
            return (
                RiskIssue(
                    "STRATEGY_PROFILE_EMPTY",
                    "mined strategy profile has zero usable entries; run import-legacy and mine-strategy before live send",
                    severity=strict_live_severity,
                ),
            )

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


def issues_to_dicts(
    issues: tuple[RiskIssue, ...] | list[RiskIssue],
    *,
    strategy_profile_evidence: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for issue in issues:
        payload = dict(issue.__dict__)
        if strategy_profile_evidence:
            payload["strategy_profile_evidence"] = strategy_profile_evidence
        out.append(payload)
    return out


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if str(item))


def _entry_evidence(
    entry: StrategyProfileEntry,
    *,
    profile_match: str,
    requested_method: str | None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "profile_match": profile_match,
        "profile_pair": entry.pair,
        "profile_direction": entry.direction,
        "profile_method": entry.method,
        "requested_method": requested_method,
        "profile_status": entry.status,
        "required_fix": entry.required_fix,
    }
    for key in (
        "target_reward_risk",
        "positive_best_jpy",
        "positive_tail_jpy",
        "positive_evidence_n",
        "live_net_jpy",
        "live_avg_jpy",
        "live_n",
        "pretrade_net_jpy",
        "pretrade_avg_jpy",
        "pretrade_n",
        "seat_net_jpy",
        "seat_win_rate_pct",
        "seat_pl_n",
    ):
        value = getattr(entry, key)
        if value is not None:
            evidence[key] = value
    if entry.top_block_reasons:
        evidence["top_block_reasons"] = list(entry.top_block_reasons[:3])
    return evidence


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


def _method_specific_entries(
    entries: dict[tuple[str, str, str | None], StrategyProfileEntry],
    pair: str,
    direction: str,
) -> list[StrategyProfileEntry]:
    return [
        entry
        for (entry_pair, entry_direction, method), entry in entries.items()
        if entry_pair == pair and entry_direction == direction and method is not None
    ]


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
    if _is_reversal_recovery_hedge(metadata):
        return "WARN"
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


def _is_reversal_recovery_hedge(metadata: dict[str, Any]) -> bool:
    """Return True for a same-pair recovery hedge backed by reversal evidence.

    Missing mined strategy history remains a live blocker for ordinary fresh
    entries. A REVERSAL recovery hedge is different: it is an executable risk
    response to already-open trapped exposure, and its live permission comes
    from broker-truth hedge metadata plus risk validation rather than archived
    pair/side strategy promotion.
    """
    return (
        str(metadata.get("position_intent") or "").upper() == "HEDGE"
        and bool(metadata.get("hedge_recovery"))
        and str(metadata.get("hedge_timing_class") or "").upper() == "REVERSAL"
    )


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
