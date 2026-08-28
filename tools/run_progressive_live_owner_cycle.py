#!/usr/bin/env python3
"""One owner cycle: GET-only preflight, promotion, one Gateway call, readback.

Before an explicit release receipt and THROTTLED_LIVE sizing exist this tool
never constructs a write-capable broker client and exits with broker mutation
and order attempts fixed at zero.  Once admitted, it consumes one sealed
supervision receipt, durably reserves the promotion, invokes LiveOrderGateway
once, and treats every ambiguous result as consumed/no-retry.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.broker.execution import LiveOrderGateway
from quant_rabbit.broker.oanda import OandaExecutionClient
from quant_rabbit.broker.position_execution import PositionProtectionGateway
from quant_rabbit.fast_bot_promotion import (
    build_fast_bot_promotion,
    dispatch_promotion_once,
)
from quant_rabbit.inventory_controller import (
    InventoryController,
    InventoryState,
    broker_order_identity,
    broker_position_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_PREFLIGHT_SPEC = importlib.util.spec_from_file_location(
    "qr_progressive_live_preflight",
    REPO_ROOT / "tools" / "run_progressive_live_preflight.py",
)
if _PREFLIGHT_SPEC is None or _PREFLIGHT_SPEC.loader is None:
    raise RuntimeError("PREFLIGHT_IMPORT_FAILED")
_PREFLIGHT = importlib.util.module_from_spec(_PREFLIGHT_SPEC)
_PREFLIGHT_SPEC.loader.exec_module(_PREFLIGHT)
load_sealer_module = _PREFLIGHT.load_sealer_module
read_json = _PREFLIGHT.read_json
run_preflight = _PREFLIGHT.run_preflight
verify_release_receipt = _PREFLIGHT.verify_release_receipt


OWNER_CYCLE_CONTRACT = "QR_PROGRESSIVE_LIVE_OWNER_CYCLE_V1"
DEFAULT_STATE_ROOT = (
    Path.home() / ".codex" / "state" / "quantrabbit" / "progressive-live-owner-v1"
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def _mode_event(preflight: Mapping[str, Any]) -> dict[str, Any]:
    event_id = str(preflight.get("last_event_id") or "")
    ledger = Path(str(preflight.get("ledger_path") or ""))
    if not event_id or not ledger.is_file():
        raise RuntimeError("PREFLIGHT_MODE_EVENT_MISSING")
    for line in reversed(ledger.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        if row.get("event_id") == event_id:
            return row
    raise RuntimeError("PREFLIGHT_MODE_EVENT_NOT_FOUND")


def _apply_supervision_to_inventory(
    inventory: InventoryController,
    *,
    supervision: Mapping[str, Any],
    mode_event: Mapping[str, Any],
    now_utc: datetime,
) -> str:
    """Apply supervision only against the actual current preflight identity."""

    event_id = str(mode_event.get("event_id") or "")
    if not event_id:
        raise RuntimeError("PREFLIGHT_MODE_EVENT_ID_MISSING")
    return inventory.apply_supervision_receipt(
        event={"event_id": event_id, "dedupe_key": event_id},
        receipt=supervision,
        now_utc=now_utc,
    )


def _no_send_result(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract": OWNER_CYCLE_CONTRACT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "NO_LIVE_DISPATCH",
        "mode": preflight.get("mode"),
        "transition_reason": preflight.get("transition_reason"),
        "needs_user_decision": preflight.get("needs_user_decision"),
        "waiting_external_state": preflight.get("waiting_external_state"),
        "live_order_gateway_invocation_count": 0,
        "external_order_attempts": 0,
        "external_orders": 0,
        "broker_mutation_performed": False,
        "manual_tagless_policy": "NO_TOUCH",
        "existing_tp_sl_policy": "NO_TOUCH",
    }


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _mark_and_price_inventory(
    controller: InventoryController,
    *,
    snapshot: Any,
    now_utc: datetime,
) -> None:
    for position in snapshot.positions:
        identity = broker_position_identity(position)
        if identity is None or identity.campaign_id != controller.campaign_id:
            continue
        quote = snapshot.quotes.get(position.pair)
        if quote is not None and identity.lot_id in controller.lots:
            executable = quote.bid if position.side.value == "LONG" else quote.ask
            controller.mark_lot(
                identity.lot_id,
                executable_price=float(executable),
                now_utc=now_utc,
            )
        raw = position.raw if isinstance(position.raw, Mapping) else {}
        margin_relief = float(raw.get("initialMargin") or raw.get("marginUsed") or 0.0)
        controller.update_unwind_economics(
            identity.lot_id,
            estimated_margin_relief_jpy=max(0.0, margin_relief),
            estimated_close_loss_and_cost_jpy=max(0.0, -float(position.unrealized_pl_jpy)),
            currency_factor=str(position.pair).split("_", 1)[0],
            now_utc=now_utc,
        )


def _drain_gateway_actions(
    controller: InventoryController,
    *,
    snapshot: Any,
    now_utc: datetime,
    hard_terminal: bool,
) -> tuple[dict[str, Any], ...]:
    positions_by_lot = {
        identity.lot_id: position
        for position in snapshot.positions
        for identity in [broker_position_identity(position)]
        if identity is not None and identity.campaign_id == controller.campaign_id
    }
    orders_by_id = {
        str(order.order_id): (identity, order)
        for order in snapshot.orders
        for identity in [broker_order_identity(order)]
        if identity is not None and identity.campaign_id == controller.campaign_id
    }
    deadline = now_utc if hard_terminal else now_utc + timedelta(minutes=30)
    planned: list[dict[str, Any]] = []
    for action in controller.unwind_actions(
        now_utc=now_utc,
        terminal_deadline_utc=deadline,
    ):
        if action.action == "REDUCE_BOT_LOT" and action.lot_id in positions_by_lot:
            lot = controller.lots[str(action.lot_id)]
            position = positions_by_lot[str(action.lot_id)]
            body = {
                "action": action.action,
                "campaign_id": lot.identity.campaign_id,
                "strategy_id": lot.identity.strategy_id,
                "lot_id": lot.identity.lot_id,
                "trade_id": position.trade_id,
                "pair": position.pair,
                "units": int(action.units or 0),
                "remaining_units_before": int(position.units),
                "reason": action.reason,
            }
        elif action.action == "CANCEL_PENDING_ENTRY" and action.pending_order_id in orders_by_id:
            identity, order = orders_by_id[str(action.pending_order_id)]
            body = {
                "action": action.action,
                "campaign_id": identity.campaign_id,
                "strategy_id": identity.strategy_id,
                "lot_id": identity.lot_id,
                "order_id": order.order_id,
                "pair": order.pair,
                "reason": action.reason,
            }
        else:
            continue
        planned.append({**body, "action_id": f"qria:{_canonical_sha(body)}"})
    return tuple(planned)


def _manage_owned_inventory(
    *,
    controller: InventoryController,
    client: OandaExecutionClient,
    pairs: tuple[str, ...],
    preflight: Mapping[str, Any],
    owner_state_root: Path,
    now_utc: datetime,
) -> tuple[dict[str, Any] | None, Any]:
    snapshot = client.snapshot(pairs)
    reconcile = controller.reconcile_broker_truth(
        positions=snapshot.positions,
        orders=snapshot.orders,
        now_utc=now_utc,
    )
    _mark_and_price_inventory(controller, snapshot=snapshot, now_utc=now_utc)
    event = _mode_event(preflight)
    nav = float((event.get("account") or {}).get("nav_jpy") or 0.0)
    if controller.cycle_start_nav_jpy is None and controller.state is InventoryState.RUNNING:
        controller.configure_profit_lock(cycle_start_nav_jpy=nav, now_utc=now_utc)
    if controller.state is InventoryState.STOPPED:
        if (
            preflight.get("promotion_ready") is True
            and controller.cooldown_elapsed(now_utc)
            and reconcile["bot_position_count"] == 0
            and reconcile["bot_pending_entry_count"] == 0
        ):
            controller.restart_cycle(cycle_start_nav_jpy=nav, now_utc=now_utc)
        else:
            return {
                **_no_send_result(preflight),
                "status": "STOPPED_COOLDOWN_OR_FRESH_GATE_WAIT",
                "inventory_state": controller.state.value,
                "inventory_reconcile": reconcile,
            }, snapshot

    has_owned = bool(
        reconcile["bot_position_count"]
        or reconcile["bot_pending_entry_count"]
        or controller.pending_entry_ids
        or any(lot.remaining_units > 0 for lot in controller.lots.values())
    )
    hard_reason = None
    profit_lock_drain = (
        controller.state is InventoryState.DRAINING
        and controller.profit_lock_triggered
        and preflight.get("transition_reason") == "INVENTORY_DRAINING"
    )
    if has_owned and preflight.get("mode") != "THROTTLED_LIVE" and not profit_lock_drain:
        hard_reason = str(preflight.get("transition_reason") or "ACCOUNT_GATE_DEGRADED")
    profit_action = controller.evaluate_profit_lock(
        current_nav_jpy=nav,
        now_utc=now_utc,
        hard_limit_reason=hard_reason,
    )
    if controller.state is InventoryState.FREEZE_NEW:
        controller.begin_draining(now_utc=now_utc)
    if controller.state is InventoryState.FLAT:
        controller.stop(now_utc=now_utc, cooldown=timedelta(minutes=30))
        return {
            **_no_send_result(preflight),
            "status": "BOT_INVENTORY_FLAT_STOPPED",
            "inventory_state": controller.state.value,
            "profit_lock_action": profit_action,
            "inventory_reconcile": reconcile,
        }, snapshot
    if controller.state is not InventoryState.DRAINING:
        return None, snapshot

    planned = _drain_gateway_actions(
        controller,
        snapshot=snapshot,
        now_utc=now_utc,
        hard_terminal=hard_reason is not None,
    )
    gateway = PositionProtectionGateway(
        client=client,
        output_path=owner_state_root / "inventory_gateway_result.json",
        report_path=owner_state_root / "inventory_gateway_report.md",
        live_enabled=True,
    )
    summary = gateway.run_inventory_drain(
        actions=planned,
        snapshot=snapshot,
        reservation_path=owner_state_root / "inventory_dispatch_ledger.json",
        send=bool(planned),
    )
    after = client.snapshot(pairs)
    post_reconcile = controller.reconcile_broker_truth(
        positions=after.positions,
        orders=after.orders,
        now_utc=datetime.now(timezone.utc),
    )
    if controller.state is InventoryState.FLAT:
        controller.stop(
            now_utc=datetime.now(timezone.utc),
            cooldown=timedelta(minutes=30),
        )
    payload = json.loads((owner_state_root / "inventory_gateway_result.json").read_text())
    attempts = sum(1 for item in payload.get("actions", []) if item.get("broker_post_attempted"))
    result = {
        **_no_send_result(preflight),
        "status": "INVENTORY_DRAIN_CYCLE",
        "mode": "DRAINING",
        "inventory_state": controller.state.value,
        "profit_lock_action": profit_action,
        "inventory_reconcile": reconcile,
        "post_drain_reconcile": post_reconcile,
        "inventory_gateway_status": summary.status,
        "inventory_gateway_invocation_count": 1,
        "external_order_attempts": attempts,
        "external_orders": 0,
        "external_inventory_mutation_attempts": attempts,
        "external_inventory_mutations_acknowledged": sum(
            1 for item in payload.get("actions", []) if item.get("sent") is True
        ),
        "external_entry_orders": 0,
        "broker_mutation_performed": summary.sent,
        "no_resend_on_unchanged_broker_truth": True,
    }
    return result, after


def run_owner_cycle(
    *,
    env_file: Path,
    approval_packet_path: Path,
    expected_packet_sha256: str,
    resident_status_path: Path,
    release_receipt_path: Path | None,
    supervision_receipt_path: Path | None,
    inventory_state_path: Path,
    preflight_state_root: Path,
    owner_state_root: Path,
    strategy_profile_path: Path,
    execution_ledger_path: Path,
    target_state_path: Path,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    preflight = run_preflight(
        env_file=env_file,
        approval_packet_path=approval_packet_path,
        expected_packet_sha256=expected_packet_sha256,
        resident_status_path=resident_status_path,
        release_receipt_path=release_receipt_path,
        inventory_state_path=inventory_state_path,
        state_root=preflight_state_root,
        now_utc=now_utc,
    )
    # Network reads happen inside preflight.  In production, advance the
    # promotion clock after those reads so a quote/signal created during the
    # cycle is not misclassified as future relative to the cycle-start clock.
    if now_utc is None:
        now = datetime.now(timezone.utc)
    if release_receipt_path is None or not release_receipt_path.is_file():
        result = _no_send_result(preflight)
        _atomic_json(owner_state_root / "state.json", result)
        return result
    sealer = load_sealer_module()
    packet = sealer.verify_approval_packet(
        read_json(approval_packet_path),
        expected_packet_sha256=expected_packet_sha256,
    )
    manifest = sealer.software_manifest()
    admission: dict[str, Any] | None = None
    risk: dict[str, Any] | None = None
    inventory: InventoryController | None = None
    client: OandaExecutionClient | None = None
    if release_receipt_path.is_file():
        admission, risk = verify_release_receipt(
            read_json(release_receipt_path),
            approval_packet_sha256=expected_packet_sha256,
            software_manifest=manifest,
        )
        inventory = InventoryController.open(
            inventory_state_path,
            campaign_id=str(risk["live_campaign_id"]),
            now_utc=now,
        )
        event = _mode_event(preflight)
        account_has_bot_inventory = int(
            ((event.get("account") or {}).get("system_owned_position_count") or 0)
        ) > 0
        needs_inventory_readback = bool(
            preflight.get("promotion_ready") is True
            or account_has_bot_inventory
            or inventory.pending_entry_ids
            or any(lot.remaining_units > 0 for lot in inventory.lots.values())
            or inventory.state is not InventoryState.RUNNING
        )
        if needs_inventory_readback:
            client = OandaExecutionClient(env_file=env_file)
            inventory_result, _ = _manage_owned_inventory(
                controller=inventory,
                client=client,
                pairs=tuple(packet["initial_pairs"]),
                preflight=preflight,
                owner_state_root=owner_state_root,
                now_utc=now,
            )
            if inventory_result is not None:
                _atomic_json(owner_state_root / "state.json", inventory_result)
                return inventory_result
    if preflight.get("promotion_ready") is not True:
        result = _no_send_result(preflight)
        _atomic_json(owner_state_root / "state.json", result)
        return result
    if (
        release_receipt_path is None
        or supervision_receipt_path is None
        or admission is None
        or risk is None
        or inventory is None
    ):
        raise RuntimeError("PROMOTION_EVIDENCE_PATH_REQUIRED")

    event = _mode_event(preflight)
    receipts = [
        row
        for row in event.get("signal_receipts", [])
        if isinstance(row, Mapping)
        and (row.get("mode_receipt") or {}).get("mode") == "THROTTLED_LIVE"
        and int((row.get("mode_receipt") or {}).get("calculated_units") or 0) > 0
    ]
    if not receipts:
        raise RuntimeError("THROTTLED_SIGNAL_RECEIPT_MISSING")
    selected = max(
        receipts,
        key=lambda row: int((row.get("mode_receipt") or {}).get("calculated_units") or 0),
    )
    signal = selected.get("signal")
    sizing = selected.get("sizing_receipt")
    if not isinstance(signal, Mapping) or not isinstance(sizing, Mapping):
        raise RuntimeError("SEALED_SIGNAL_OR_SIZING_MISSING")

    supervision = read_json(supervision_receipt_path)
    apply_status = _apply_supervision_to_inventory(
        inventory,
        supervision=supervision,
        mode_event=event,
        now_utc=now,
    )
    if apply_status in {"APPLIED_FREEZE_NEW", "APPLIED_UNWIND"}:
        client = client or OandaExecutionClient(env_file=env_file)
        inventory_result, _ = _manage_owned_inventory(
            controller=inventory,
            client=client,
            pairs=tuple(packet["initial_pairs"]),
            preflight=preflight,
            owner_state_root=owner_state_root,
            now_utc=now,
        )
        result = inventory_result or {
            **_no_send_result(preflight),
            "status": "SUPERVISION_ENTRY_FREEZE_APPLIED",
            "inventory_state": inventory.state.value,
            "supervision_apply_status": apply_status,
        }
        _atomic_json(owner_state_root / "state.json", result)
        return result
    if apply_status not in {"APPLIED_ALLOW", "DUPLICATE_IGNORED"}:
        raise RuntimeError(f"SUPERVISION_NOT_APPLIED:{apply_status}")
    promotion = build_fast_bot_promotion(
        signal=signal,
        supervision_receipt=supervision,
        sizing_receipt=sizing,
        forward_admission=admission,
        risk_contract=risk,
        software_version_sha256=str(manifest["software_version_sha256"]),
        expected_feature_snapshot_sha256=str(supervision.get("feature_snapshot_sha256") or ""),
        inventory=inventory,
        now_utc=now,
    )
    if promotion.get("status") != "ADMITTED":
        result = {
            **_no_send_result(preflight),
            "status": "PROMOTION_BLOCKED",
            "blocking_reasons": promotion.get("blocking_reasons"),
        }
        _atomic_json(owner_state_root / "state.json", result)
        return result

    promotion_path = owner_state_root / "promotion.json"
    intents_path = owner_state_root / "gateway_intents.json"
    _atomic_json(promotion_path, promotion)
    client = client or OandaExecutionClient(env_file=env_file)
    gateway = LiveOrderGateway(
        client=client,
        strategy_profile=strategy_profile_path,
        output_path=owner_state_root / "gateway_result.json",
        report_path=owner_state_root / "gateway_report.md",
        live_enabled=True,
        max_loss_jpy=float(risk["max_loss_per_order_jpy"]),
        portfolio_loss_cap_jpy=float(risk["stop_drawdown_jpy"]),
        target_state_path=target_state_path,
        verified_decision_path=None,
        progressive_promotion_path=promotion_path,
        execution_ledger_db_path=execution_ledger_path,
        execution_ledger_report_path=execution_ledger_path.with_suffix(".md"),
    )
    dispatch = dispatch_promotion_once(
        promotion=promotion,
        gateway=gateway,
        intents_path=intents_path,
        dispatch_ledger_path=owner_state_root / "dispatch_ledger.json",
        inventory_state_path=inventory_state_path,
        now_utc=now,
        send=True,
        confirm_live=True,
    )
    sent = dispatch.get("broker_mutation_performed") is True
    readback: dict[str, Any] | None = None
    if sent:
        snapshot = client.snapshot(tuple(packet["initial_pairs"]))
        inventory = InventoryController.open(
            inventory_state_path,
            campaign_id=str(risk["live_campaign_id"]),
            now_utc=datetime.now(timezone.utc),
        )
        inventory_reconcile = inventory.reconcile_broker_truth(
            positions=snapshot.positions,
            orders=snapshot.orders,
            now_utc=datetime.now(timezone.utc),
        )
        intent = promotion.get("intents_payload", {}).get("results", [])[0].get("intent", {})
        metadata = intent.get("metadata") if isinstance(intent, Mapping) else {}
        lot_id = str((metadata or {}).get("lot_id") or "")
        matching_truth = bool(
            lot_id in inventory.lots and inventory.lots[lot_id].remaining_units > 0
        ) or any(
            (identity := broker_order_identity(order)) is not None
            and identity.campaign_id == inventory.campaign_id
            and identity.lot_id == lot_id
            for order in snapshot.orders
        )
        if lot_id and not matching_truth:
            inventory.register_unresolved_entry(
                lot_id,
                now_utc=datetime.now(timezone.utc),
            )
        readback = {
            "fetched_at_utc": snapshot.fetched_at_utc.isoformat(),
            "bot_positions": [
                {
                    "trade_id": row.trade_id,
                    "pair": row.pair,
                    "side": row.side.value,
                    "units": row.units,
                    "owner": row.owner.value,
                }
                for row in snapshot.positions
                if row.owner.value == "trader"
            ],
            "pending_bot_orders": [
                {
                    "order_id": row.order_id,
                    "pair": row.pair,
                    "type": row.order_type,
                    "owner": row.owner.value,
                }
                for row in snapshot.orders
                if row.owner.value == "trader"
            ],
            "inventory_reconcile": inventory_reconcile,
            "matching_bot_truth_readback": matching_truth,
        }
    status = str(dispatch.get("status") or "")
    attempts = 1 if status == "UNKNOWN_GATEWAY_RESULT_NO_RETRY" or sent else 0
    if status == "UNKNOWN_GATEWAY_RESULT_NO_RETRY":
        inventory = InventoryController.open(
            inventory_state_path,
            campaign_id=str(risk["live_campaign_id"]),
            now_utc=datetime.now(timezone.utc),
        )
        intent = promotion.get("intents_payload", {}).get("results", [])[0].get("intent", {})
        metadata = intent.get("metadata") if isinstance(intent, Mapping) else {}
        lot_id = str((metadata or {}).get("lot_id") or "")
        if lot_id:
            inventory.register_unresolved_entry(
                lot_id,
                now_utc=datetime.now(timezone.utc),
            )
    result = {
        "contract": OWNER_CYCLE_CONTRACT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "mode": preflight.get("mode"),
        "promotion_id": promotion.get("promotion_id"),
        "promotion_sha256": promotion.get("promotion_sha256"),
        "live_order_gateway_invocation_count": dispatch.get("live_order_gateway_invocation_count"),
        "external_order_attempts": attempts,
        "external_orders": 1 if sent else 0,
        "broker_mutation_performed": sent,
        "dispatch": dispatch,
        "broker_readback": readback,
        "manual_tagless_policy": "NO_TOUCH",
        "existing_tp_sl_policy": "NO_TOUCH",
        "no_resend_on_unknown": True,
    }
    _atomic_json(owner_state_root / "state.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--approval-packet", type=Path, required=True)
    parser.add_argument("--expected-packet-sha256", required=True)
    parser.add_argument("--resident-status", type=Path, required=True)
    parser.add_argument("--release-receipt", type=Path)
    parser.add_argument("--supervision-receipt", type=Path)
    parser.add_argument("--inventory-state", type=Path, required=True)
    parser.add_argument("--preflight-state-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument(
        "--strategy-profile",
        type=Path,
        default=REPO_ROOT / "config" / "fast_bot_progressive_strategy_profile_v1.json",
    )
    parser.add_argument("--execution-ledger", type=Path, required=True)
    parser.add_argument("--target-state", type=Path, required=True)
    args = parser.parse_args()
    result = run_owner_cycle(
        env_file=args.env_file,
        approval_packet_path=args.approval_packet,
        expected_packet_sha256=args.expected_packet_sha256,
        resident_status_path=args.resident_status,
        release_receipt_path=args.release_receipt,
        supervision_receipt_path=args.supervision_receipt,
        inventory_state_path=args.inventory_state,
        preflight_state_root=args.preflight_state_root,
        owner_state_root=args.state_root,
        strategy_profile_path=args.strategy_profile,
        execution_ledger_path=args.execution_ledger,
        target_state_path=args.target_state,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
