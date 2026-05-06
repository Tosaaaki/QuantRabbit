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

        promotions: list[Promotion] = []
        still_blocked = 0
        for item in profile_payload.get("profiles", []) or []:
            if not isinstance(item, dict):
                continue
            pair = str(item.get("pair") or "")
            direction = str(item.get("direction") or "")
            method = str(item.get("method") or "").strip().upper() or None
            status = str(item.get("status") or "")
            receipt = _receipt_for(receipts, pair, direction, method)
            promotion = _promotion_for(item, receipt)
            if promotion is None:
                if status in {"RISK_REPAIR_CANDIDATE", "MINE_MISSED_EDGE", "BLOCK_UNTIL_NEW_EVIDENCE"}:
                    still_blocked += 1
                continue
            item["status"] = "CANDIDATE"
            if promotion.method:
                item["method"] = promotion.method
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
