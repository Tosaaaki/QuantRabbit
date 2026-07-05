from __future__ import annotations

from typing import Any


MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCK_CODE = "MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCKED"
MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCK_CODE = "MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED"


def month_scale_residual_metadata_issue(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    metadata = _metadata_from_payload(payload)
    if not metadata:
        return None
    if not metadata.get("month_scale_residual_loss_repair_blocked"):
        group = metadata.get("month_scale_residual_loss_group")
        if not isinstance(group, dict):
            return None
    else:
        group = metadata.get("month_scale_residual_loss_group")
    if not isinstance(group, dict):
        group = {}
    residual_scope = str(group.get("residual_scope") or "").strip().upper()
    code = (
        MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCK_CODE
        if residual_scope == "ENTRY_QUALITY_OR_CLOSE_RESIDUAL"
        else MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCK_CODE
    )
    pair = str(group.get("pair") or payload.get("pair") or "").strip() or "UNKNOWN"
    side = str(group.get("side") or payload.get("side") or payload.get("direction") or "").strip() or "UNKNOWN"
    method = str(group.get("method") or payload.get("method") or "").strip() or "UNKNOWN"
    repair_pl = group.get("repair_replay_pl_jpy")
    loss_closes = group.get("loss_closes")
    return {
        "code": code,
        "severity": "BLOCK",
        "message": (
            "month-scale residual family gate blocks this live send until the "
            "matching 744h replay residual disappears or the filtered replay is "
            "non-negative "
            f"(pair={pair}, side={side}, method={method}, "
            f"repair_replay_pl_jpy={repair_pl}, loss_closes={loss_closes})"
        ),
        "blocked_profitability_segment": {
            "pair": pair,
            "side": side,
            "method": method,
            "residual_scope": residual_scope or None,
            "repair_replay_pl_jpy": repair_pl,
            "loss_closes": loss_closes,
            "trade_ids": group.get("trade_ids") or [],
        },
    }


def _metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    direct = payload.get("metadata")
    if isinstance(direct, dict):
        return direct
    intent = payload.get("intent")
    if isinstance(intent, dict) and isinstance(intent.get("metadata"), dict):
        return intent["metadata"]
    self_improvement = payload.get("self_improvement")
    if isinstance(self_improvement, dict):
        metadata: dict[str, Any] = {}
        for key in (
            "month_scale_residual_loss_repair_blocked",
            "month_scale_residual_loss_group",
        ):
            if key in self_improvement:
                metadata[key] = self_improvement[key]
        return metadata
    return {}
