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
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.broker.execution import LiveOrderGateway
from quant_rabbit.broker.oanda import OandaExecutionClient
from quant_rabbit.fast_bot_promotion import (
    build_fast_bot_promotion,
    dispatch_promotion_once,
)
from quant_rabbit.inventory_controller import InventoryController

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
    if preflight.get("promotion_ready") is not True:
        result = _no_send_result(preflight)
        _atomic_json(owner_state_root / "state.json", result)
        return result
    if release_receipt_path is None or supervision_receipt_path is None:
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

    sealer = load_sealer_module()
    packet = sealer.verify_approval_packet(
        read_json(approval_packet_path),
        expected_packet_sha256=expected_packet_sha256,
    )
    manifest = sealer.software_manifest()
    admission, risk = verify_release_receipt(
        read_json(release_receipt_path),
        approval_packet_sha256=expected_packet_sha256,
        software_manifest=manifest,
    )
    supervision = read_json(supervision_receipt_path)
    inventory = InventoryController.open(
        inventory_state_path,
        campaign_id=str(risk["live_campaign_id"]),
        now_utc=now,
    )
    apply_status = inventory.apply_supervision_receipt(
        event={
            "event_id": supervision.get("event_id"),
            "dedupe_key": supervision.get("dedupe_key"),
        },
        receipt=supervision,
        now_utc=now,
    )
    if apply_status not in {"APPLIED_ALLOW", "DUPLICATE_RECEIPT"}:
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
    client = OandaExecutionClient(env_file=env_file)
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
        }
    status = str(dispatch.get("status") or "")
    attempts = 1 if status == "UNKNOWN_GATEWAY_RESULT_NO_RETRY" or sent else 0
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
