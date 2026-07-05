from __future__ import annotations

from typing import Any

from .models import OrderIntent, Owner, Side, TradeMethod


MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE = "MARKET_CLOSE_LEAK_FAMILY_BLOCKED"
MARKET_CLOSE_LEAK_FAMILY_PAIR = "EUR_USD"
MARKET_CLOSE_LEAK_FAMILY_SIDE = "LONG"
MARKET_CLOSE_LEAK_FAMILY_METHOD = "BREAKOUT_FAILURE"
MARKET_CLOSE_LEAK_FAMILY_TRADE_IDS = (
    "470356",
    "470353",
    "470730",
    "471174",
    "471089",
    "471255",
    "472280",
)
MARKET_CLOSE_LEAK_FAMILY_MANUAL_EXCLUDED_TRADE_IDS = ("472987",)

_TRUE_EVIDENCE_TOKENS = {
    "1",
    "TRUE",
    "YES",
    "Y",
    "PASS",
    "PASSED",
    "PROVEN",
    "VERIFIED",
    "CLEARED",
    "OK",
}
_FALSE_EVIDENCE_TOKENS = {
    "",
    "0",
    "FALSE",
    "NO",
    "N",
    "FAIL",
    "FAILED",
    "BLOCK",
    "BLOCKED",
    "MISSING",
    "NONE",
}

CLOSE_GATE_PROOF_KEYS = (
    "market_close_leak_family_close_gate_proof",
    "market_close_leak_close_gate_proof",
    "loss_side_market_close_gate_proof",
    "close_gate_proof",
    "close_gate_evidence_proven",
    "close_gate_evidence_passed",
)
CONTAINED_RISK_TIMING_EVIDENCE_KEYS = (
    "market_close_leak_family_contained_risk_timing_evidence",
    "contained_risk_timing_evidence",
    "loss_close_contained_risk_timing_evidence",
    "market_close_contained_risk_timing_evidence",
)
TP_PROVEN_EXCEPTION_KEYS = (
    "market_close_leak_family_tp_proven_exception",
    "market_close_leak_tp_proven_exception",
    "tp_proven_exception_evidence",
    "tp_proven_market_close_exception",
)


def market_close_leak_family_proof_requirements() -> dict[str, Any]:
    return {
        "close_gate_proof_required": True,
        "contained_risk_timing_evidence_required": True,
        "tp_proven_exception_evidence_required": True,
        "required_close_reason": "MARKET_ORDER_TRADE_CLOSE",
        "attribution_scope": "SYSTEM_GATEWAY_ATTRIBUTED_ONLY",
    }


def market_close_leak_family_proof_status(metadata: dict[str, Any] | None) -> dict[str, bool]:
    payload = metadata if isinstance(metadata, dict) else {}
    close_gate = _any_truthy_evidence(payload, CLOSE_GATE_PROOF_KEYS)
    contained_risk_timing = _any_truthy_evidence(payload, CONTAINED_RISK_TIMING_EVIDENCE_KEYS)
    tp_proven_exception = _any_truthy_evidence(payload, TP_PROVEN_EXCEPTION_KEYS)
    return {
        "close_gate_proof": close_gate,
        "contained_risk_timing_evidence": contained_risk_timing,
        "tp_proven_exception_evidence": tp_proven_exception,
        "exception_proven": close_gate and contained_risk_timing and tp_proven_exception,
    }


def market_close_leak_family_matches_intent(intent: OrderIntent) -> bool:
    metadata = intent.metadata if isinstance(intent.metadata, dict) else {}
    if _manual_excluded(metadata, owner=intent.owner):
        return False
    method = intent.market_context.method if intent.market_context is not None else None
    method_value = method.value if isinstance(method, TradeMethod) else str(method or metadata.get("method") or "")
    return _family_fields_match(
        pair=intent.pair,
        side=intent.side.value if isinstance(intent.side, Side) else str(intent.side),
        method=method_value,
    )


def market_close_leak_family_matches_payload(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else payload
    metadata = _payload_metadata(payload, intent)
    owner = str(intent.get("owner") or payload.get("owner") or "").strip().lower()
    if _manual_excluded(metadata, owner=owner):
        return False
    market_context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
    method = (
        payload.get("method")
        or market_context.get("method")
        or metadata.get("method")
        or _method_from_lane_id(str(payload.get("lane_id") or ""))
    )
    return _family_fields_match(
        pair=str(intent.get("pair") or payload.get("pair") or ""),
        side=str(intent.get("side") or payload.get("side") or payload.get("direction") or ""),
        method=str(method or ""),
    )


def market_close_leak_family_exception_proven(metadata: dict[str, Any] | None) -> bool:
    return market_close_leak_family_proof_status(metadata).get("exception_proven", False)


def market_close_leak_family_block_issue(intent: OrderIntent) -> dict[str, Any] | None:
    metadata = intent.metadata if isinstance(intent.metadata, dict) else {}
    if not market_close_leak_family_matches_intent(intent):
        return None
    proof_status = market_close_leak_family_proof_status(metadata)
    if proof_status["exception_proven"]:
        return None
    return market_close_leak_family_issue(
        lane_id=str(metadata.get("lane_id") or metadata.get("candidate_id") or ""),
        proof_status=proof_status,
    )


def market_close_leak_family_payload_issue(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not market_close_leak_family_matches_payload(payload):
        return None
    intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else payload
    metadata = _payload_metadata(payload, intent)
    proof_status = market_close_leak_family_proof_status(metadata)
    if proof_status["exception_proven"]:
        return None
    return market_close_leak_family_issue(
        lane_id=str(payload.get("lane_id") or metadata.get("lane_id") or ""),
        proof_status=proof_status,
    )


def market_close_leak_family_issue(
    *,
    lane_id: str = "",
    proof_status: dict[str, bool] | None = None,
) -> dict[str, Any]:
    status = proof_status or market_close_leak_family_proof_status({})
    missing = [
        name
        for name in (
            "close_gate_proof",
            "contained_risk_timing_evidence",
            "tp_proven_exception_evidence",
        )
        if not status.get(name)
    ]
    return {
        "code": MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE,
        "severity": "BLOCK",
        "message": (
            "EUR_USD LONG BREAKOUT_FAILURE system-gateway market-close leak family is blocked "
            "until close-gate proof, contained-risk timing evidence, and TP-proven exception "
            "evidence are all present."
        ),
        "evidence": {
            "lane_id": lane_id or None,
            "pair": MARKET_CLOSE_LEAK_FAMILY_PAIR,
            "side": MARKET_CLOSE_LEAK_FAMILY_SIDE,
            "method": MARKET_CLOSE_LEAK_FAMILY_METHOD,
            "close_reason": "MARKET_ORDER_TRADE_CLOSE",
            "system_gateway_loss_trade_ids": list(MARKET_CLOSE_LEAK_FAMILY_TRADE_IDS),
            "operator_manual_excluded_trade_ids": list(MARKET_CLOSE_LEAK_FAMILY_MANUAL_EXCLUDED_TRADE_IDS),
            "proof_status": dict(status),
            "missing_proofs": missing,
            "requirements": market_close_leak_family_proof_requirements(),
        },
    }


def _family_fields_match(*, pair: str, side: str, method: str) -> bool:
    return (
        _normal(pair) == MARKET_CLOSE_LEAK_FAMILY_PAIR
        and _normal(side) == MARKET_CLOSE_LEAK_FAMILY_SIDE
        and _normal(method) == MARKET_CLOSE_LEAK_FAMILY_METHOD
    )


def _method_from_lane_id(lane_id: str) -> str:
    parts = [part for part in str(lane_id or "").split(":") if part]
    return parts[-1] if parts else ""


def _manual_excluded(metadata: dict[str, Any], *, owner: Owner | str | None) -> bool:
    owner_value = owner.value if isinstance(owner, Owner) else str(owner or "")
    if owner_value.strip().lower() in {Owner.OPERATOR_MANUAL.value, Owner.MANUAL.value}:
        return True
    trade_id = str(metadata.get("trade_id") or metadata.get("broker_trade_id") or "").strip()
    if trade_id in MARKET_CLOSE_LEAK_FAMILY_MANUAL_EXCLUDED_TRADE_IDS:
        return True
    operator_packet = metadata.get("operator_manual_position")
    if isinstance(operator_packet, dict):
        if str(operator_packet.get("classification") or "").upper() == "OPERATOR_MANUAL":
            return True
    classification = str(metadata.get("classification") or metadata.get("operator_classification") or "").upper()
    return classification == "OPERATOR_MANUAL"


def _payload_metadata(payload: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if isinstance(intent.get("metadata"), dict):
        metadata.update(intent["metadata"])
    if isinstance(payload.get("market_close_leak_family"), dict):
        metadata.update(payload["market_close_leak_family"])
    return metadata


def _any_truthy_evidence(payload: dict[str, Any], keys: tuple[str, ...]) -> bool:
    nested = payload.get("market_close_leak_family")
    if isinstance(nested, dict) and _any_truthy_evidence(nested, keys):
        return True
    return any(_truthy_evidence(payload.get(key)) for key in keys)


def _truthy_evidence(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().upper()
        if token in _FALSE_EVIDENCE_TOKENS:
            return False
        if token in _TRUE_EVIDENCE_TOKENS:
            return True
        return False
    if isinstance(value, dict):
        for key in ("passed", "proven", "verified", "cleared", "allowed"):
            if key in value:
                return _truthy_evidence(value.get(key))
        status = value.get("status") or value.get("result") or value.get("verdict")
        if status is not None:
            return _truthy_evidence(status)
    return False


def _normal(value: Any) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
