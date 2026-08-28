from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

from quant_rabbit.fast_bot import SIGNAL_CONTRACT
from quant_rabbit.inventory_controller import InventoryController, InventoryState, LotIdentity


PROMOTION_DECISION_CONTRACT = "QR_FAST_BOT_PROMOTION_DECISION_V1"
FORWARD_ADMISSION_CONTRACT = "QR_FAST_BOT_FORWARD_ADMISSION_V1"
RISK_CONTRACT = "QR_FAST_BOT_LIVE_RISK_CONTRACT_V1"
SUPERVISION_RECEIPT_CONTRACT = "QR_FAST_BOT_SUPERVISION_RECEIPT_V1"
SIZING_RECEIPT_CONTRACT = "QR_FAST_BOT_SIZING_RECEIPT_V1"
DISPATCH_LEDGER_CONTRACT = "QR_FAST_BOT_PROMOTION_DISPATCH_LEDGER_V1"
EXTERNAL_MUTATION_GATEWAY = "LiveOrderGateway"
# These are the preregistered forward-admission floors already published by
# the fast-bot shadow contract.  They are constants because changing them
# after seeing outcomes would invalidate the experiment; a later experiment
# must replace the entire versioned admission contract rather than tune them.
MINIMUM_RESOLVED_FORWARD_FILLS = 100
MINIMUM_ACTIVE_FORWARD_DAYS = 10
MINIMUM_FORWARD_PROFIT_FACTOR = 1.25
MAXIMUM_FORWARD_SPREAD_ANOMALY_RATE = 0.02
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_LLM_ORDER_KEYS = frozenset(
    {
        "pair",
        "side",
        "units",
        "order_type",
        "entry",
        "take_profit",
        "stop_loss",
        "tp",
        "sl",
        "price",
    }
)


class ExistingLiveOrderGateway(Protocol):
    """The already-existing broker boundary; this module never implements one."""

    def run(
        self,
        *,
        intents_path: Path,
        lane_id: str | None = None,
        size_multiple: float = 1.0,
        send: bool = False,
        confirm_live: bool = False,
    ) -> Any: ...


def build_fast_bot_promotion(
    *,
    signal: Mapping[str, Any],
    supervision_receipt: Mapping[str, Any],
    sizing_receipt: Mapping[str, Any],
    forward_admission: Mapping[str, Any],
    risk_contract: Mapping[str, Any],
    software_version_sha256: str,
    expected_feature_snapshot_sha256: str,
    inventory: InventoryController,
    now_utc: datetime,
) -> dict[str, Any]:
    """Validate an exact fast-bot proposal and emit one standard gateway intent.

    The function is deterministic and broker-free.  LLM output can constrain a
    regime, strategy set, risk cap, and position cap, but order fields and units
    come only from the sealed fast-bot signal and deterministic sizing receipt.
    """

    now = _aware_utc(now_utc)
    blockers: list[str] = []
    software_sha = str(software_version_sha256 or "")
    feature_sha = str(expected_feature_snapshot_sha256 or "")
    signal_sha = str(signal.get("signal_sha256") or "")
    proof_sha = str(forward_admission.get("admission_sha256") or "")
    risk_sha = str(risk_contract.get("risk_contract_sha256") or "")

    if not _sealed(signal, seal_key="signal_sha256", contract=SIGNAL_CONTRACT):
        blockers.append("SIGNAL_SEAL_INVALID")
    if signal.get("shadow_only") is not True or signal.get("live_permission") is not False:
        blockers.append("SIGNAL_NOT_SHADOW_ORIGIN")
    if signal.get("broker_mutation_allowed") is not False:
        blockers.append("SIGNAL_SOURCE_MUTATION_AUTHORITY_INVALID")
    if not _sha(software_sha) or not _sha(feature_sha):
        blockers.append("SOFTWARE_OR_FEATURE_BINDING_INVALID")

    signal_expires_at_utc: str | None = None
    try:
        generated_at = _parse_utc(signal.get("generated_at_utc"))
        quote_at = _parse_utc(signal.get("quote_timestamp_utc"))
        ttl_seconds = _positive_int(signal.get("entry_ttl_seconds"))
    except (TypeError, ValueError):
        blockers.append("SIGNAL_TIME_CONTRACT_INVALID")
    else:
        signal_expires_at_utc = datetime.fromtimestamp(
            quote_at.timestamp() + ttl_seconds,
            tz=timezone.utc,
        ).isoformat()
        if generated_at > now or quote_at > now or now.timestamp() > quote_at.timestamp() + ttl_seconds:
            blockers.append("SIGNAL_STALE_OR_FUTURE")

    source_identity: LotIdentity | None = None
    try:
        source_identity = LotIdentity.from_metadata(signal)
    except ValueError:
        blockers.append("SIGNAL_OWNERSHIP_INVALID")
    if source_identity is not None and not source_identity.campaign_id.startswith("paper-fb-"):
        blockers.append("SIGNAL_NOT_PAPER_CAMPAIGN")

    proof_body = _validate_forward_admission(
        forward_admission,
        software_sha=software_sha,
        signal=signal,
    )
    if proof_body is None:
        blockers.append("FORWARD_ADMISSION_INVALID_OR_UNPROVEN")
    risk_body = _validate_risk_contract(
        risk_contract,
        software_sha=software_sha,
        forward_admission_sha256=proof_sha,
        now=now,
    )
    if risk_body is None:
        blockers.append("RISK_CONTRACT_INVALID_OR_UNACCEPTED")

    source_strategy_id = str(signal.get("strategy_id") or "")
    live_strategy_id = f"live-{source_strategy_id}"
    live_identity: LotIdentity | None = None
    try:
        live_identity = LotIdentity(
            campaign_id=str(risk_contract.get("live_campaign_id") or ""),
            strategy_id=live_strategy_id,
            lot_id=f"live-{str(signal.get('signal_id') or '')}",
        )
    except ValueError:
        blockers.append("LIVE_OWNERSHIP_IDENTITY_INVALID")
    if live_identity is not None:
        if not live_identity.campaign_id.startswith("live-fb-"):
            blockers.append("LIVE_CAMPAIGN_NAMESPACE_INVALID")
        if source_identity is not None and live_identity.campaign_id == source_identity.campaign_id:
            blockers.append("LIVE_PAPER_CAMPAIGN_NOT_SEPARATED")
        if live_identity.strategy_id == source_strategy_id:
            blockers.append("LIVE_PAPER_STRATEGY_NOT_SEPARATED")
        if live_identity.campaign_id != inventory.campaign_id:
            blockers.append("INVENTORY_CAMPAIGN_MISMATCH")
    if _contains_forbidden_order_keys(supervision_receipt):
        blockers.append("LLM_RECEIPT_CONTAINS_ORDER_FIELDS")
    supervision_is_valid = _supervision_valid(
        supervision_receipt,
        now=now,
        feature_sha=feature_sha,
        signal_sha=signal_sha,
        regime_sha=str(signal.get("regime_contract_sha256") or ""),
        strategy_id=live_strategy_id,
    )
    if not supervision_is_valid:
        blockers.append("SUPERVISION_RECEIPT_INVALID_OR_STALE")

    units = (
        _deterministic_sized_units(
            sizing_receipt,
            signal_sha=signal_sha,
            proof_sha=proof_sha,
            risk_sha=risk_sha,
            software_sha=software_sha,
            risk_contract=risk_contract,
            supervision_receipt=supervision_receipt,
            now=now,
        )
        if proof_body is not None and risk_body is not None and supervision_is_valid
        else None
    )
    if units is None:
        blockers.append("SIZING_RECEIPT_INVALID_OR_BLOCKED")

    if inventory.state is not InventoryState.RUNNING or not inventory.can_enter(now):
        blockers.append("INVENTORY_NOT_RUNNING_OR_COOLDOWN_ACTIVE")
    if str(supervision_receipt.get("receipt_id") or "") not in inventory.applied_receipt_ids:
        blockers.append("SUPERVISION_NOT_APPLIED_TO_DURABLE_INVENTORY")
    if live_strategy_id not in inventory.allowed_strategy_ids:
        blockers.append("STRATEGY_NOT_ALLOWED_BY_DURABLE_INVENTORY")
    if (
        inventory.supervision_regime != str(supervision_receipt.get("regime") or "")
        or inventory.allowed_strategy_ids != list(supervision_receipt.get("allowed_strategy_ids") or [])
        or inventory.supervision_risk_budget_cap_jpy
        != _safe_float(supervision_receipt.get("risk_budget_cap_jpy"))
        or inventory.supervision_max_positions_cap
        != _safe_int(supervision_receipt.get("max_positions_cap"))
    ):
        blockers.append("DURABLE_SUPERVISION_CONTENT_MISMATCH")
    if inventory.supervision_expires_at_utc:
        try:
            if now > _parse_utc(inventory.supervision_expires_at_utc):
                blockers.append("DURABLE_SUPERVISION_EXPIRED")
        except (TypeError, ValueError):
            blockers.append("DURABLE_SUPERVISION_TIME_INVALID")
    active_lots = [lot for lot in inventory.lots.values() if lot.remaining_units > 0]
    if inventory.pending_entry_ids:
        blockers.append("BOT_PENDING_OR_UNRESOLVED_ENTRY_PRESENT")
    try:
        position_cap = min(
            _nonnegative_int(supervision_receipt.get("max_positions_cap")),
            _positive_int(risk_contract.get("max_bot_positions")),
        ) if supervision_is_valid and risk_body is not None else 0
    except (TypeError, ValueError):
        position_cap = 0
    if position_cap <= 0 or len(active_lots) >= position_cap:
        blockers.append("BOT_POSITION_CAP_REACHED")
    if any(
        lot.identity.strategy_id == live_strategy_id
        and lot.pair == str(signal.get("pair") or "").upper()
        and lot.reduction_started
        for lot in inventory.lots.values()
    ):
        blockers.append("REDUCED_INVENTORY_READD_FORBIDDEN")

    lane_id = (
        f"fast_bot:{str(signal.get('pair') or '')}:"
        f"{str(signal.get('side') or '')}:{str(signal.get('method') or '')}:"
        f"{str(signal.get('order_type') or '')}"
    )
    binding = {
        "signal_sha256": signal_sha,
        "signal_quote_timestamp_utc": signal.get("quote_timestamp_utc"),
        "signal_entry_ttl_seconds": signal.get("entry_ttl_seconds"),
        "forward_admission_sha256": proof_sha,
        "risk_contract_sha256": risk_sha,
        "software_version_sha256": software_sha,
        "feature_snapshot_sha256": feature_sha,
        "supervision_receipt_id": str(supervision_receipt.get("receipt_id") or ""),
        "sizing_receipt_sha256": _canonical_sha(sizing_receipt),
        "inventory_revision": inventory.revision,
        "source_shadow_campaign_id": (
            source_identity.campaign_id if source_identity is not None else None
        ),
        "source_shadow_strategy_id": source_strategy_id,
        "live_campaign_id": live_identity.campaign_id if live_identity is not None else None,
        "live_strategy_id": live_strategy_id,
        "lane_id": lane_id,
    }
    promotion_id = f"fbp:{_canonical_sha(binding)}"
    status = "ADMITTED" if not blockers else "BLOCKED"
    result: dict[str, Any] = {
        "contract": PROMOTION_DECISION_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "expires_at_utc": signal_expires_at_utc,
        "status": status,
        "promotion_id": promotion_id,
        "lane_id": lane_id,
        "live_permission": status == "ADMITTED",
        "broker_mutation_allowed": status == "ADMITTED",
        "external_mutation_gateway": EXTERNAL_MUTATION_GATEWAY,
        "blocking_reasons": sorted(set(blockers)),
        "bindings": binding,
        "intents_payload": None,
    }
    if status == "ADMITTED" and live_identity is not None and units is not None:
        intent = {
            "pair": str(signal["pair"]).upper(),
            "side": str(signal["side"]).upper(),
            "order_type": str(signal["order_type"]).upper(),
            "units": units,
            "entry": float(signal["entry"]),
            "tp": float(signal["take_profit"]),
            "sl": float(signal["stop_loss"]),
            "thesis": "content-addressed fast-bot proposal under bounded supervision",
            "owner": "trader",
            "market_context": {
                "regime": str(supervision_receipt.get("regime") or ""),
                "narrative": "deterministic fast-bot signal under sealed supervisor limits",
                "chart_story": str(signal.get("feature_reason") or signal.get("method") or ""),
                "method": str(signal.get("method") or ""),
                "invalidation": "attached deterministic stop-loss geometry",
                "event_risk": "bounded by account-wide progressive-live gates",
                "session": str(signal.get("session") or ""),
            },
            "metadata": {
                **live_identity.to_metadata(),
                "lane_id": lane_id,
                "desk": "fast_bot",
                "campaign_role": "NOW",
                "fast_bot_promotion_id": promotion_id,
                "source_shadow_campaign_id": source_identity.campaign_id,
                "source_shadow_strategy_id": source_strategy_id,
                "source_shadow_lot_id": source_identity.lot_id,
                **binding,
            },
        }
        result["intents_payload"] = {
            "contract": "QR_ORDER_INTENTS_V1",
            "results": [
                {
                    "lane_id": lane_id,
                    "status": "LIVE_READY",
                    "risk_allowed": True,
                    "intent": intent,
                }
            ],
        }
    return _seal_result(result)


def dispatch_promotion_once(
    *,
    promotion: Mapping[str, Any],
    gateway: ExistingLiveOrderGateway,
    intents_path: Path,
    dispatch_ledger_path: Path,
    inventory_state_path: Path,
    now_utc: datetime,
    send: bool = False,
    confirm_live: bool = False,
) -> dict[str, Any]:
    """Reserve once, then call the supplied existing LiveOrderGateway once.

    Reservation is persisted before invoking the gateway.  An exception or
    ambiguous gateway outcome consumes the reservation and is never retried.
    """

    if not _sealed(
        promotion,
        seal_key="promotion_sha256",
        contract=PROMOTION_DECISION_CONTRACT,
    ):
        return _dispatch_result("BLOCKED_INVALID_PROMOTION", promotion, 0, False, None)
    if promotion.get("status") != "ADMITTED" or not isinstance(
        promotion.get("intents_payload"), Mapping
    ):
        return _dispatch_result("BLOCKED_NOT_ADMITTED", promotion, 0, False, None)
    if send and (promotion.get("broker_mutation_allowed") is not True or not confirm_live):
        return _dispatch_result("BLOCKED_LIVE_CONFIRMATION_REQUIRED", promotion, 0, False, None)

    bindings = promotion.get("bindings")
    if not isinstance(bindings, Mapping):
        return _dispatch_result("BLOCKED_INVALID_PROMOTION_BINDINGS", promotion, 0, False, None)
    intent_rows = promotion.get("intents_payload", {}).get("results", [])
    try:
        campaign_id = str(intent_rows[0]["intent"]["metadata"]["campaign_id"])
        inventory = InventoryController.open(
            inventory_state_path,
            campaign_id=campaign_id,
            now_utc=_aware_utc(now_utc),
        )
    except (IndexError, KeyError, TypeError, ValueError, RuntimeError):
        return _dispatch_result("BLOCKED_INVENTORY_READBACK_INVALID", promotion, 0, False, None)
    if (
        inventory.revision != bindings.get("inventory_revision")
        or inventory.state is not InventoryState.RUNNING
        or not inventory.can_enter(_aware_utc(now_utc))
    ):
        return _dispatch_result("BLOCKED_INVENTORY_CHANGED_AFTER_PROMOTION", promotion, 0, False, None)

    promotion_id = str(promotion.get("promotion_id") or "")
    reservation = _reserve_dispatch_once(
        dispatch_ledger_path,
        promotion_id=promotion_id,
        promotion_sha256=str(promotion.get("promotion_sha256") or ""),
        send=send,
    )
    if not reservation:
        return _dispatch_result("DUPLICATE_BLOCKED", promotion, 0, False, None)
    _atomic_json_write(intents_path, promotion["intents_payload"])
    try:
        summary = gateway.run(
            intents_path=intents_path,
            lane_id=str(promotion.get("lane_id") or ""),
            size_multiple=1.0,
            send=send,
            confirm_live=confirm_live,
        )
    except Exception as exc:
        _finish_dispatch(
            dispatch_ledger_path,
            promotion_id=promotion_id,
            outcome="UNKNOWN_GATEWAY_RESULT_NO_RETRY",
        )
        return _dispatch_result(
            "UNKNOWN_GATEWAY_RESULT_NO_RETRY",
            promotion,
            1,
            False,
            {"error_type": type(exc).__name__},
        )
    sent = bool(
        summary.get("sent", False)
        if isinstance(summary, Mapping)
        else getattr(summary, "sent", False)
    )
    payload = _summary_payload(summary)
    _finish_dispatch(
        dispatch_ledger_path,
        promotion_id=promotion_id,
        outcome="GATEWAY_RETURNED",
    )
    return _dispatch_result("GATEWAY_RETURNED", promotion, 1, sent, payload)


def _validate_forward_admission(
    value: Mapping[str, Any], *, software_sha: str, signal: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    if not _sealed(value, seal_key="admission_sha256", contract=FORWARD_ADMISSION_CONTRACT):
        return None
    if (
        value.get("status") != "ADMITTED"
        or value.get("promotion_allowed") is not True
        or value.get("live_permission") is not True
        or value.get("external_mutation_gateway") != EXTERNAL_MUTATION_GATEWAY
        or value.get("software_version_sha256") != software_sha
        or str(signal.get("strategy_id") or "") not in (value.get("allowed_strategy_ids") or [])
        or str(signal.get("pair") or "") not in (value.get("allowed_pairs") or [])
    ):
        return None
    progressive = value.get("admission_mode") == "PROGRESSIVE_MICRO_LIVE"
    if progressive:
        return value if (
            value.get("progressive_live_user_authorized") is True
            and value.get("authorization_source") == "EXPLICIT_USER_DECISION"
            and bool(str(value.get("authorization_id") or "").strip())
            and value.get("resident_shadow_required") is True
            and value.get("resident_shadow_status") == "RUNNING"
            and value.get("resident_shadow_execution_authority") == "NONE"
            and value.get("resident_shadow_broker_mutation_count") == 0
            and value.get("resident_shadow_external_order_attempts") == 0
            and value.get("resident_shadow_external_orders") == 0
            and value.get("scorecard_monitoring_active") is True
            and value.get("scorecard_can_force_demotion") is True
            and value.get("fixed_sample_wait_required_for_micro_live") is False
            and value.get("micro_live_only") is True
            and value.get("independent_readback_verified") is True
        ) else None
    try:
        return value if (
            _positive_int(value.get("resolved_fills")) >= MINIMUM_RESOLVED_FORWARD_FILLS
            and _positive_int(value.get("active_days")) >= MINIMUM_ACTIVE_FORWARD_DAYS
            and _finite(value.get("profit_factor")) >= MINIMUM_FORWARD_PROFIT_FACTOR
            and _finite(value.get("one_sided_95_expectancy_lower_pips")) > 0.0
            and 0.0 <= _finite(value.get("spread_anomaly_rate")) <= MAXIMUM_FORWARD_SPREAD_ANOMALY_RATE
            and _finite(value.get("after_cost_net_pips")) > 0.0
            and _nonnegative_int(value.get("leftover_inventory_units")) == 0
            and _nonnegative_int(value.get("paper_broker_mutation_count")) == 0
            and value.get("maximum_drawdown_within_predeclared_limit") is True
            and value.get("tail_loss_within_predeclared_limit") is True
            and value.get("margin_stress_passed") is True
            and value.get("independent_readback_verified") is True
        ) else None
    except (TypeError, ValueError):
        return None


def _validate_risk_contract(
    value: Mapping[str, Any], *, software_sha: str, forward_admission_sha256: str,
    now: datetime
) -> Mapping[str, Any] | None:
    if not _sealed(value, seal_key="risk_contract_sha256", contract=RISK_CONTRACT):
        return None
    if (
        value.get("status") != "ACCEPTED"
        or value.get("accepted_by_user") is not True
        or value.get("acceptance_source") != "EXPLICIT_USER_DECISION"
        or not str(value.get("acceptance_id") or "").strip()
        or value.get("software_version_sha256") != software_sha
        or value.get("forward_admission_sha256") != forward_admission_sha256
        or not str(value.get("live_campaign_id") or "").startswith("live-fb-")
    ):
        return None
    try:
        current = _finite(value.get("max_post_entry_current_mcp"))
        stress = _finite(value.get("max_post_entry_stress_mcp"))
        accepted_at = _parse_utc(value.get("accepted_at_utc"))
        return value if (
            accepted_at <= now
            and _finite(value.get("max_loss_per_order_jpy")) > 0.0
            and _finite(value.get("stop_drawdown_jpy")) > 0.0
            and _finite(value.get("minimum_margin_buffer_jpy")) > 0.0
            and 0.0 < current < stress < 1.0
            and _finite(value.get("max_currency_factor_nav_multiple")) > 0.0
            and _positive_int(value.get("max_bot_positions")) > 0
            and 0.0 < _finite(value.get("mode_hysteresis_mcp")) < current
            and _finite(value.get("stress_pips")) > 0.0
            and _finite(value.get("max_account_snapshot_age_seconds")) > 0.0
        ) else None
    except (TypeError, ValueError):
        return None


def _supervision_valid(
    value: Mapping[str, Any], *, now: datetime, feature_sha: str, signal_sha: str,
    regime_sha: str, strategy_id: str
) -> bool:
    if not _sealed(
        value,
        seal_key="receipt_sha256",
        contract=SUPERVISION_RECEIPT_CONTRACT,
    ):
        return False
    try:
        return bool(
            value.get("decision") == "ALLOW"
            and _parse_utc(value.get("generated_at_utc")) <= now <= _parse_utc(value.get("expires_at_utc"))
            and value.get("feature_snapshot_sha256") == feature_sha
            and value.get("signal_sha256") == signal_sha
            and value.get("regime_contract_sha256") == regime_sha
            and strategy_id in (value.get("allowed_strategy_ids") or [])
            and _finite(value.get("risk_budget_cap_jpy")) > 0.0
            and _positive_int(value.get("max_positions_cap")) > 0
        )
    except (TypeError, ValueError):
        return False


def _deterministic_sized_units(
    value: Mapping[str, Any], *, signal_sha: str, proof_sha: str, risk_sha: str,
    software_sha: str, risk_contract: Mapping[str, Any], supervision_receipt: Mapping[str, Any],
    now: datetime
) -> int | None:
    if not _sealed(
        value,
        seal_key="sizing_receipt_sha256",
        contract=SIZING_RECEIPT_CONTRACT,
    ):
        return None
    try:
        units = _positive_int(value.get("calculated_units"))
        minimum = _positive_int(value.get("broker_minimum_units"))
        planned_loss = _finite(value.get("planned_loss_jpy"))
        current_mcp = _finite(value.get("post_entry_current_mcp"))
        stress_mcp = _finite(value.get("post_entry_stress_mcp"))
        post_margin = _finite(value.get("post_entry_margin_available_jpy"))
        post_factor_multiple = _finite(
            value.get("post_entry_max_currency_factor_nav_multiple")
        )
        campaign_drawdown = _finite(value.get("campaign_drawdown_jpy"))
        account_age = _finite(value.get("account_snapshot_age_seconds"))
        quote_age = _finite(value.get("quote_age_seconds"))
        calculated_at = _parse_utc(value.get("calculated_at_utc"))
        manual_mutations = _nonnegative_int(value.get("manual_tagless_mutation_count"))
    except (TypeError, ValueError):
        return None
    bindings_ok = all(
        value.get(key) == expected
        for key, expected in (
            ("signal_sha256", signal_sha),
            ("forward_admission_sha256", proof_sha),
            ("risk_contract_sha256", risk_sha),
            ("software_version_sha256", software_sha),
        )
    )
    return units if (
        bindings_ok
        and value.get("mode") == "THROTTLED_LIVE"
        and value.get("mutation_allowed") is True
        and value.get("account_scope_includes_manual_and_tagless_positions") is True
        and manual_mutations == 0
        and value.get("spread_gate_passed") is True
        and _sha(str(value.get("account_snapshot_sha256") or ""))
        and _sha(str(value.get("quote_snapshot_sha256") or ""))
        and calculated_at <= now
        and units >= minimum
        and planned_loss >= 0.0
        and planned_loss <= _finite(risk_contract.get("max_loss_per_order_jpy"))
        and planned_loss <= _finite(supervision_receipt.get("risk_budget_cap_jpy"))
        and 0.0 <= campaign_drawdown < _finite(risk_contract.get("stop_drawdown_jpy"))
        and post_margin >= _finite(risk_contract.get("minimum_margin_buffer_jpy"))
        and 0.0 <= post_factor_multiple <= _finite(
            risk_contract.get("max_currency_factor_nav_multiple")
        )
        and 0.0 <= account_age <= _finite(
            risk_contract.get("max_account_snapshot_age_seconds")
        )
        and 0.0 <= quote_age <= _finite(
            risk_contract.get("max_account_snapshot_age_seconds")
        )
        and 0.0 <= current_mcp <= stress_mcp < 1.0
        and current_mcp <= _finite(risk_contract.get("max_post_entry_current_mcp"))
        and stress_mcp <= _finite(risk_contract.get("max_post_entry_stress_mcp"))
    ) else None


def _contains_forbidden_order_keys(value: Any) -> bool:
    if isinstance(value, Mapping):
        normalized_keys = {str(key).strip().lower() for key in value}
        return bool(_FORBIDDEN_LLM_ORDER_KEYS.intersection(normalized_keys)) or any(
            _contains_forbidden_order_keys(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_order_keys(item) for item in value)
    return False


def _reserve_dispatch_once(
    path: Path, *, promotion_id: str, promotion_sha256: str, send: bool
) -> bool:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        payload = _load_dispatch_ledger(path)
        if any(item.get("promotion_id") == promotion_id for item in payload["dispatches"]):
            return False
        payload["dispatches"].append(
            {
                "promotion_id": promotion_id,
                "promotion_sha256": promotion_sha256,
                "send_requested": send,
                "state": "DISPATCH_RESERVED",
            }
        )
        _atomic_json_write(path, payload)
        return True


def _finish_dispatch(path: Path, *, promotion_id: str, outcome: str) -> None:
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        payload = _load_dispatch_ledger(path)
        for item in payload["dispatches"]:
            if item.get("promotion_id") == promotion_id:
                item["state"] = outcome
                break
        _atomic_json_write(path, payload)


def _load_dispatch_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"contract": DISPATCH_LEDGER_CONTRACT, "dispatches": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("contract") != DISPATCH_LEDGER_CONTRACT or not isinstance(value.get("dispatches"), list):
        raise RuntimeError("promotion dispatch ledger is invalid")
    return value


def _dispatch_result(
    status: str, promotion: Mapping[str, Any], invocation_count: int,
    sent: bool, gateway_summary: Mapping[str, Any] | None
) -> dict[str, Any]:
    return {
        "contract": "QR_FAST_BOT_GATEWAY_DISPATCH_READBACK_V1",
        "status": status,
        "promotion_id": promotion.get("promotion_id"),
        "external_mutation_gateway": EXTERNAL_MUTATION_GATEWAY,
        "live_order_gateway_invocation_count": invocation_count,
        "broker_mutation_performed": sent,
        "gateway_summary": dict(gateway_summary) if gateway_summary is not None else None,
    }


def _summary_payload(summary: Any) -> dict[str, Any]:
    if is_dataclass(summary):
        value = asdict(summary)
    elif isinstance(summary, Mapping):
        value = dict(summary)
    else:
        value = {"status": str(getattr(summary, "status", "UNKNOWN"))}
    return {
        key: str(item) if isinstance(item, Path) else item
        for key, item in value.items()
    }


def _seal_result(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "promotion_sha256"}
    return {**body, "promotion_sha256": _canonical_sha(body)}


def seal_forward_admission(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "admission_sha256"}
    return {**body, "admission_sha256": _canonical_sha(body)}


def seal_risk_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "risk_contract_sha256"}
    return {**body, "risk_contract_sha256": _canonical_sha(body)}


def seal_supervision_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return {**body, "receipt_sha256": _canonical_sha(body)}


def seal_sizing_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sizing_receipt_sha256"}
    return {**body, "sizing_receipt_sha256": _canonical_sha(body)}


def build_sizing_receipt(
    *,
    mode_receipt: Mapping[str, Any],
    signal_sha256: str,
    forward_admission_sha256: str,
    risk_contract_sha256: str,
    software_version_sha256: str,
    account_snapshot_sha256: str,
    quote_snapshot_sha256: str,
    campaign_drawdown_jpy: float,
    account_snapshot_age_seconds: float,
    quote_age_seconds: float,
    calculated_at_utc: datetime,
    spread_gate_passed: bool,
) -> dict[str, Any]:
    """Bind account-wide mode sizing to the promotion contract.

    This adapter never mutates the broker.  Manual/tagless exposure is part of
    the upstream account-wide calculation and its mutation count is fixed at
    zero here.
    """

    return seal_sizing_receipt(
        {
            "contract": SIZING_RECEIPT_CONTRACT,
            "signal_sha256": signal_sha256,
            "forward_admission_sha256": forward_admission_sha256,
            "risk_contract_sha256": risk_contract_sha256,
            "software_version_sha256": software_version_sha256,
            "mode": mode_receipt.get("mode"),
            "mutation_allowed": mode_receipt.get("mutation_allowed") is True,
            "calculated_units": mode_receipt.get("calculated_units"),
            "safe_unit_capacity": mode_receipt.get("safe_unit_capacity"),
            "broker_minimum_units": mode_receipt.get("broker_minimum_units"),
            "planned_loss_jpy": mode_receipt.get("planned_loss_jpy"),
            "post_entry_current_mcp": mode_receipt.get("post_entry_current_mcp"),
            "post_entry_stress_mcp": mode_receipt.get("post_entry_stress_mcp"),
            "post_entry_margin_available_jpy": mode_receipt.get(
                "post_entry_margin_available_jpy"
            ),
            "post_entry_max_currency_factor_nav_multiple": mode_receipt.get(
                "post_entry_max_currency_factor_nav_multiple"
            ),
            "campaign_drawdown_jpy": campaign_drawdown_jpy,
            "account_snapshot_age_seconds": account_snapshot_age_seconds,
            "quote_age_seconds": quote_age_seconds,
            "account_snapshot_sha256": account_snapshot_sha256,
            "quote_snapshot_sha256": quote_snapshot_sha256,
            "calculated_at_utc": _aware_utc(calculated_at_utc).isoformat(),
            "account_scope_includes_manual_and_tagless_positions": True,
            "manual_tagless_mutation_count": 0,
            "spread_gate_passed": spread_gate_passed,
        }
    )


def _sealed(value: Mapping[str, Any], *, seal_key: str, contract: str) -> bool:
    if not isinstance(value, Mapping) or value.get("contract") != contract:
        return False
    body = {key: item for key, item in value.items() if key != seal_key}
    return str(value.get(seal_key) or "") == _canonical_sha(body)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha(value: str) -> bool:
    return _SHA256_RE.fullmatch(value) is not None


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timezone-aware UTC timestamp required")
    return value.astimezone(timezone.utc)


def _parse_utc(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    return _aware_utc(parsed)


def _finite(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("finite number required")
    return number


def _positive_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("positive integer required")
    number = int(value)
    if number <= 0:
        raise ValueError("positive integer required")
    return number


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("nonnegative integer required")
    number = int(value)
    if number < 0:
        raise ValueError("nonnegative integer required")
    return number


def _safe_float(value: object) -> float | None:
    try:
        return _finite(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        return _nonnegative_int(value)
    except (TypeError, ValueError):
        return None
