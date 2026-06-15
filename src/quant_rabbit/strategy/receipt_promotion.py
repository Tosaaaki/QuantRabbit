from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_ORDER_INTENTS,
    DEFAULT_RECEIPT_PROMOTION_REPORT,
    DEFAULT_STRATEGY_PROFILE,
)


PROMOTABLE_STATUSES = {"DRY_RUN_PASSED", "LIVE_READY"}
PENDING_ENTRY_TYPES = {"LIMIT", "STOP-ENTRY"}
NON_PROMOTABLE_RISK_CODES = {
    "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
    "FORECAST_CONTEXT_REQUIRED_FOR_LIVE",
    "FORECAST_DIRECTIONAL_HIT_RATE_WEAK_FOR_LIVE",
    "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
    "FORECAST_TREND_CONTINUATION_HIGHER_TF_REQUIRED_FOR_LIVE",
    "FORECAST_TREND_CONTINUATION_REWARD_RISK_TOO_LOW",
    "FORECAST_WATCH_ONLY",
    "FRESH_ENTRY_REWARD_RISK_NOT_POSITIVE",
    "TELEMETRY_FORECAST_CONTEXT_REQUIRED_FOR_LIVE",
    "TELEMETRY_FORECAST_HISTORY_REQUIRED_FOR_LIVE",
    "TELEMETRY_FORECAST_NOT_EXECUTABLE_FOR_LIVE",
    "TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE",
}


@dataclass(frozen=True)
class Promotion:
    pair: str
    direction: str
    method: str | None
    old_status: str
    new_status: str
    lane_id: str
    reason: str


@dataclass(frozen=True)
class ReceiptPromotionSummary:
    profile_path: Path
    intents_path: Path
    report_path: Path
    profiles_seen: int
    receipts_seen: int
    promoted: int
    still_blocked: int


class ReceiptPromoter:
    """Promote mined strategy profiles only from concrete dry-run receipts."""

    def __init__(
        self,
        *,
        profile_path: Path = DEFAULT_STRATEGY_PROFILE,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        output_profile: Path | None = None,
        report_path: Path = DEFAULT_RECEIPT_PROMOTION_REPORT,
    ) -> None:
        self.profile_path = profile_path
        self.intents_path = intents_path
        self.output_profile = output_profile or profile_path
        self.report_path = report_path

    def run(self) -> ReceiptPromotionSummary:
        profile_payload = json.loads(self.profile_path.read_text())
        intents_payload = json.loads(self.intents_path.read_text())
        receipts = _eligible_receipts(intents_payload)
        generated_at = datetime.now(timezone.utc).isoformat()
        profiles = profile_payload.get("profiles")
        if not isinstance(profiles, list):
            profiles = []
            profile_payload["profiles"] = profiles
        else:
            profiles = _deduped_profiles(profiles)
            profile_payload["profiles"] = profiles
        method_specific_pairs = _method_specific_pairs(profiles)
        profile_keys = _profile_keys(profiles)

        promotions: list[Promotion] = []
        still_blocked = 0
        for item in profiles:
            if not isinstance(item, dict):
                continue
            pair = str(item.get("pair") or "")
            direction = str(item.get("direction") or "")
            method = str(item.get("method") or "").strip().upper() or None
            status = str(item.get("status") or "")
            receipt = _receipt_for(receipts, pair, direction, method)
            if _should_create_method_profile_instead(item, receipt, method_specific_pairs):
                if status in {"RISK_REPAIR_CANDIDATE", "MINE_MISSED_EDGE", "BLOCK_UNTIL_NEW_EVIDENCE"}:
                    still_blocked += 1
                continue
            promotion = _promotion_for(item, receipt)
            if promotion is None:
                if status in {"RISK_REPAIR_CANDIDATE", "MINE_MISSED_EDGE", "BLOCK_UNTIL_NEW_EVIDENCE"}:
                    still_blocked += 1
                continue
            if _would_duplicate_method_profile(item, promotion, profile_keys):
                if status in {"RISK_REPAIR_CANDIDATE", "MINE_MISSED_EDGE", "BLOCK_UNTIL_NEW_EVIDENCE"}:
                    still_blocked += 1
                continue
            item["status"] = "CANDIDATE"
            if promotion.method:
                item["method"] = promotion.method
                profile_keys.discard((pair, direction, method))
                profile_keys.add((pair, direction, promotion.method))
            item["required_fix"] = (
                f"promoted from {promotion.old_status} by {promotion.reason}; "
                f"source lane {promotion.lane_id}; receipt_at_utc={generated_at}"
            )
            item["receipt_promotion"] = {
                "promoted_at_utc": generated_at,
                "from_status": promotion.old_status,
                "lane_id": promotion.lane_id,
                "method": promotion.method,
                "reason": promotion.reason,
            }
            promotions.append(promotion)

        for receipt in receipts.values():
            method_profile = _method_profile_from_missing_issue(receipt, profile_keys, generated_at)
            if method_profile is None:
                continue
            item, promotion = method_profile
            profiles.append(item)
            profile_keys.add((promotion.pair, promotion.direction, promotion.method))
            promotions.append(promotion)

        profile_payload["last_receipt_promotion_at_utc"] = generated_at
        profile_payload["receipt_promotions"] = [promotion.__dict__ for promotion in promotions]
        self.output_profile.parent.mkdir(parents=True, exist_ok=True)
        self.output_profile.write_text(json.dumps(profile_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self._write_report(profile_payload, intents_payload, promotions, still_blocked, generated_at)
        return ReceiptPromotionSummary(
            profile_path=self.output_profile,
            intents_path=self.intents_path,
            report_path=self.report_path,
            profiles_seen=len(profile_payload.get("profiles", []) or []),
            receipts_seen=len(receipts),
            promoted=len(promotions),
            still_blocked=still_blocked,
        )

    def _write_report(
        self,
        profile_payload: dict[str, Any],
        intents_payload: dict[str, Any],
        promotions: list[Promotion],
        still_blocked: int,
        generated_at: str,
    ) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Receipt Promotion Report",
            "",
            f"- Generated at UTC: `{generated_at}`",
            f"- Strategy profile: `{self.output_profile}`",
            f"- Order intents: `{self.intents_path}`",
            f"- Intent snapshot: `{intents_payload.get('snapshot_path')}`",
            f"- Profiles seen: `{len(profile_payload.get('profiles', []) or [])}`",
            f"- Promoted: `{len(promotions)}`",
            f"- Still blocked: `{still_blocked}`",
            "",
            "## Promotions",
            "",
        ]
        if promotions:
            for item in promotions:
                lines.append(
                    f"- `{item.pair} {item.direction}{_method_suffix(item.method)}` {item.old_status} -> {item.new_status} "
                    f"via `{item.lane_id}`"
                )
                lines.append(f"  - reason: {item.reason}")
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "## Promotion Contract",
                "",
                "- `RISK_REPAIR_CANDIDATE` can promote only from a risk-allowed dry-run receipt with no blocking risk issue.",
                "- `MINE_MISSED_EDGE` can promote only from a risk-allowed LIMIT or STOP-ENTRY receipt.",
                "- `BLOCK_UNTIL_NEW_EVIDENCE` is never auto-promoted by this command.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _eligible_receipts(intents_payload: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    receipts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for result in intents_payload.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        if result.get("status") not in PROMOTABLE_STATUSES or result.get("risk_allowed") is not True:
            continue
        if _has_blocking_risk(result):
            continue
        if _has_non_promotable_risk(result):
            continue
        intent = result.get("intent")
        if not isinstance(intent, dict):
            continue
        pair = str(intent.get("pair") or "")
        direction = str(intent.get("side") or "")
        method = _receipt_method(result)
        if not pair or not direction or not method:
            continue
        key = (pair, direction, method)
        current = receipts.get(key)
        if current is None or _receipt_rank(result) > _receipt_rank(current):
            receipts[key] = result
    return receipts


def _receipt_for(
    receipts: dict[tuple[str, str, str], dict[str, Any]],
    pair: str,
    direction: str,
    method: str | None,
) -> dict[str, Any] | None:
    if method:
        return receipts.get((pair, direction, method))
    candidates = [
        receipt
        for (receipt_pair, receipt_direction, _), receipt in receipts.items()
        if receipt_pair == pair and receipt_direction == direction
    ]
    if not candidates:
        return None
    return max(candidates, key=_receipt_rank)


def _has_blocking_risk(result: dict[str, Any]) -> bool:
    for issue in result.get("risk_issues", []) or []:
        if isinstance(issue, dict) and issue.get("severity") == "BLOCK":
            return True
    return False


def _has_non_promotable_risk(result: dict[str, Any]) -> bool:
    """Return True when the receipt is blocked by non-strategy live evidence.

    Receipt promotion fixes mined strategy-profile gaps only. A DRY_RUN_PASSED
    lane can still be deliberately non-fillable because the current forecast is
    watch-only, stale, or below the live confidence/telemetry floor. Promoting
    those receipts would convert diagnostic geometry into strategy permission
    and bypass the forecast live gates.
    """

    for issue in result.get("risk_issues", []) or []:
        if not isinstance(issue, dict):
            continue
        if str(issue.get("code") or "") in NON_PROMOTABLE_RISK_CODES:
            return True
    return False


def _receipt_rank(result: dict[str, Any]) -> tuple[int, int]:
    status_rank = 1 if result.get("status") == "LIVE_READY" else 0
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    order_type = str(intent.get("order_type") or "")
    pending_rank = 1 if order_type in PENDING_ENTRY_TYPES else 0
    return status_rank, pending_rank


def _promotion_for(item: dict[str, Any], receipt: dict[str, Any] | None) -> Promotion | None:
    if receipt is None:
        return None
    status = str(item.get("status") or "")
    lane_id = str(receipt.get("lane_id") or "")
    intent = receipt.get("intent") if isinstance(receipt.get("intent"), dict) else {}
    pair = str(item.get("pair") or "")
    direction = str(item.get("direction") or "")
    method = _receipt_method(receipt)
    order_type = str(intent.get("order_type") or "")
    if status == "RISK_REPAIR_CANDIDATE":
        return Promotion(
            pair=pair,
            direction=direction,
            method=method,
            old_status=status,
            new_status="CANDIDATE",
            lane_id=lane_id,
            reason="loss-cap geometry repaired by current dry-run receipt",
        )
    if status == "MINE_MISSED_EDGE" and order_type in PENDING_ENTRY_TYPES:
        return Promotion(
            pair=pair,
            direction=direction,
            method=method,
            old_status=status,
            new_status="CANDIDATE",
            lane_id=lane_id,
            reason=f"missed edge converted into {order_type} trigger receipt",
        )
    return None


def _should_create_method_profile_instead(
    item: dict[str, Any],
    receipt: dict[str, Any] | None,
    method_specific_pairs: set[tuple[str, str]],
) -> bool:
    """Keep pair-side fallback memory intact when a concrete method is missing."""

    if receipt is None:
        return False
    if str(item.get("method") or "").strip():
        return False
    pair = str(item.get("pair") or "")
    direction = str(item.get("direction") or "")
    if (pair, direction) not in method_specific_pairs:
        return False
    method = _receipt_method(receipt)
    if method is None:
        return False
    return _method_missing_issue(receipt, pair=pair, direction=direction, method=method) is not None


def _method_profile_from_missing_issue(
    receipt: dict[str, Any],
    existing_keys: set[tuple[str, str, str | None]],
    generated_at: str,
) -> tuple[dict[str, Any], Promotion] | None:
    intent = receipt.get("intent") if isinstance(receipt.get("intent"), dict) else {}
    pair = str(intent.get("pair") or "")
    direction = str(intent.get("side") or "")
    method = _receipt_method(receipt)
    if not pair or not direction or method is None:
        return None
    if (pair, direction, method) in existing_keys:
        return None

    issue = _method_missing_issue(receipt, pair=pair, direction=direction, method=method)
    if issue is None:
        return None
    evidence = issue.get("strategy_profile_evidence") if isinstance(issue.get("strategy_profile_evidence"), dict) else {}
    fallback = evidence.get("fallback_pair_side_profile") if isinstance(evidence.get("fallback_pair_side_profile"), dict) else None
    if fallback is None:
        return None
    if str(fallback.get("profile_pair") or "") != pair or str(fallback.get("profile_direction") or "") != direction:
        return None

    old_status = str(fallback.get("profile_status") or "")
    order_type = str(intent.get("order_type") or "")
    if old_status == "RISK_REPAIR_CANDIDATE":
        reason = "method-specific profile repaired by current dry-run receipt from pair-side risk-repair fallback"
    elif old_status == "MINE_MISSED_EDGE" and order_type in PENDING_ENTRY_TYPES:
        reason = f"method-specific profile created from pair-side missed-edge {order_type} trigger receipt"
    else:
        return None

    lane_id = str(receipt.get("lane_id") or "")
    promotion = Promotion(
        pair=pair,
        direction=direction,
        method=method,
        old_status=old_status,
        new_status="CANDIDATE",
        lane_id=lane_id,
        reason=reason,
    )
    item = _profile_from_fallback(fallback, promotion, generated_at)
    return item, promotion


def _method_missing_issue(
    receipt: dict[str, Any],
    *,
    pair: str,
    direction: str,
    method: str,
) -> dict[str, Any] | None:
    for key in ("live_strategy_issues", "strategy_issues"):
        for issue in receipt.get(key, []) or []:
            if not isinstance(issue, dict) or issue.get("code") != "STRATEGY_METHOD_PROFILE_MISSING":
                continue
            evidence = issue.get("strategy_profile_evidence")
            if not isinstance(evidence, dict):
                continue
            if str(evidence.get("profile_pair") or "") != pair:
                continue
            if str(evidence.get("profile_direction") or "") != direction:
                continue
            if str(evidence.get("requested_method") or "").strip().upper() != method:
                continue
            if evidence.get("profile_match") != "method_specific_missing":
                continue
            return issue
    return None


def _profile_from_fallback(
    fallback: dict[str, Any],
    promotion: Promotion,
    generated_at: str,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "pair": promotion.pair,
        "direction": promotion.direction,
        "method": promotion.method,
        "status": "CANDIDATE",
        "required_fix": (
            f"promoted from {promotion.old_status} by {promotion.reason}; "
            f"source lane {promotion.lane_id}; receipt_at_utc={generated_at}"
        ),
        "receipt_promotion": {
            "promoted_at_utc": generated_at,
            "from_status": promotion.old_status,
            "lane_id": promotion.lane_id,
            "method": promotion.method,
            "reason": promotion.reason,
        },
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
        "top_block_reasons",
    ):
        if key in fallback:
            item[key] = fallback[key]
    return item


def _profile_keys(profiles: list[Any]) -> set[tuple[str, str, str | None]]:
    keys: set[tuple[str, str, str | None]] = set()
    for item in profiles:
        if not isinstance(item, dict):
            continue
        pair = str(item.get("pair") or "")
        direction = str(item.get("direction") or "")
        method = str(item.get("method") or "").strip().upper() or None
        if pair and direction:
            keys.add((pair, direction, method))
    return keys


def _method_specific_pairs(profiles: list[Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for item in profiles:
        if not isinstance(item, dict):
            continue
        pair = str(item.get("pair") or "")
        direction = str(item.get("direction") or "")
        method = str(item.get("method") or "").strip()
        if pair and direction and method:
            pairs.add((pair, direction))
    return pairs


def _would_duplicate_method_profile(
    item: dict[str, Any],
    promotion: Promotion,
    profile_keys: set[tuple[str, str, str | None]],
) -> bool:
    current_method = str(item.get("method") or "").strip().upper() or None
    if current_method is not None or promotion.method is None:
        return False
    return (promotion.pair, promotion.direction, promotion.method) in profile_keys


def _deduped_profiles(profiles: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    indexes: dict[tuple[str, str, str | None], int] = {}
    for item in profiles:
        if not isinstance(item, dict):
            deduped.append(item)
            continue
        pair = str(item.get("pair") or "")
        direction = str(item.get("direction") or "")
        method = str(item.get("method") or "").strip().upper() or None
        if not pair or not direction:
            deduped.append(item)
            continue
        key = (pair, direction, method)
        existing_index = indexes.get(key)
        if existing_index is None:
            indexes[key] = len(deduped)
            deduped.append(item)
            continue
        existing = deduped[existing_index]
        if isinstance(existing, dict) and _duplicate_profile_rank(item) > _duplicate_profile_rank(existing):
            deduped[existing_index] = item
    return deduped


def _duplicate_profile_rank(item: dict[str, Any]) -> tuple[int, int]:
    status = str(item.get("status") or "")
    strictness = {
        "BLOCK_UNTIL_NEW_EVIDENCE": 4,
        "WATCH_ONLY": 3,
        "RISK_REPAIR_CANDIDATE": 2,
        "MINE_MISSED_EDGE": 2,
        "CANDIDATE": 1,
    }.get(status, 0)
    has_receipt = 1 if isinstance(item.get("receipt_promotion"), dict) else 0
    return strictness, has_receipt


def _receipt_method(result: dict[str, Any]) -> str | None:
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
    method = str(context.get("method") or "").strip().upper()
    if method:
        return method
    lane_id = str(result.get("lane_id") or "")
    parts = lane_id.split(":")
    if len(parts) >= 4:
        return parts[3].strip().upper() or None
    return None


def _method_suffix(method: str | None) -> str:
    return f" {method}" if method else ""
