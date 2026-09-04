from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.capture_economics import (
    exact_vehicle_metrics_from_surface,
    execution_cost_floor_from_surface,
    read_exact_vehicle_allocation_surface,
)
from quant_rabbit.decision_execution_lineage import _expected_decision_receipt_id
from quant_rabbit.market_read_overlay import (
    GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT,
    canonical_json_sha256,
    guardian_action_receipt_scope_material,
)


class AILiveGatewayError(RuntimeError):
    pass


_EVIDENCE_PREFIXES = ("capture_", "loss_asymmetry_")
_EVIDENCE_KEYS = {
    "attach_stop_loss_on_fill",
    "attach_take_profit_on_fill",
    "tp_execution_mode",
    "sl_execution_mode",
}


def execute_ai_trade_candidate(
    *, repo_root: Path, state_root: Path, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Send one AI-authored entry only through the existing live gateway."""

    decision = receipt.get("decision")
    if not isinstance(decision, Mapping):
        raise AILiveGatewayError("accepted AI receipt has no decision object")
    action = str(decision.get("action") or "").strip().upper()
    if action != "TRADE":
        return {
            "status": "NO_BROKER_ACTION",
            "sink": "live_gateway",
            "broker_mutation_allowed": True,
            "broker_order_posts": 0,
            "sent": False,
            "reason": f"{action or 'UNKNOWN'} does not authorize a fresh entry",
        }
    orders = decision.get("orders")
    if not isinstance(orders, list) or len(orders) != 1:
        raise AILiveGatewayError("live TRADE requires exactly one AI-authored order")
    order = orders[0]
    if not isinstance(order, Mapping):
        raise AILiveGatewayError("AI order must be an object")
    run_id = str(receipt.get("run_id") or "").strip()
    if not run_id:
        raise AILiveGatewayError("accepted AI receipt has no run_id")
    run_dir = state_root / "live" / "runs" / run_id
    artifacts = build_live_gateway_artifacts(
        repo_root=repo_root,
        order=order,
        candidate=decision,
        run_id=run_id,
        generated_at=datetime.now(timezone.utc),
    )
    intents_path = run_dir / "intents.json"
    verified_path = run_dir / "verified_decision.json"
    output_path = run_dir / "gateway_receipt.json"
    report_path = run_dir / "gateway_report.md"
    _atomic_write_json(intents_path, artifacts["intents"])
    _atomic_write_json(verified_path, artifacts["verified_decision"])

    command = [
        sys.executable,
        "-m",
        "quant_rabbit.cli",
        "stage-live-order",
        "--intents",
        str(intents_path),
        "--strategy-profile",
        str(repo_root / "data" / "strategy_profile.json"),
        "--lane-id",
        artifacts["lane_id"],
        "--output",
        str(output_path),
        "--report",
        str(report_path),
        "--verified-decision",
        str(verified_path),
        "--execution-ledger-db",
        str(repo_root / "data" / "execution_ledger.db"),
        "--execution-ledger-report",
        str(repo_root / "docs" / "execution_ledger_report.md"),
        "--target-state",
        str(repo_root / "data" / "daily_target_state.json"),
        "--target-report",
        str(repo_root / "docs" / "daily_target_report.md"),
        "--send",
        "--confirm-live",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root / "src")
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
        check=False,
    )
    if completed.returncode != 0 and not output_path.exists():
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise AILiveGatewayError(
            "live gateway failed before producing a receipt: " + detail[:1000]
        )
    gateway = _load_json(output_path)
    sent = gateway.get("sent") is True
    return {
        "status": str(gateway.get("status") or "GATEWAY_FAILED"),
        "sink": "live_gateway",
        "broker_mutation_allowed": True,
        "broker_order_posts": 1 if sent else 0,
        "sent": sent,
        "lane_id": artifacts["lane_id"],
        "gateway_receipt_path": str(output_path),
        "gateway_report_path": str(report_path),
        "risk_issues": gateway.get("risk_issues", []),
        "strategy_issues": gateway.get("strategy_issues", []),
        "command_exit_code": completed.returncode,
    }


def build_live_gateway_artifacts(
    *,
    repo_root: Path,
    order: Mapping[str, Any],
    candidate: Mapping[str, Any],
    run_id: str,
    generated_at: datetime,
) -> dict[str, Any]:
    pair = str(order["pair"]).strip().upper()
    side = str(order["side"]).strip().upper()
    method = str(order["method"]).strip().upper().replace("-", "_")
    order_type = str(order["order_type"]).strip().upper()
    vehicle = "STOP" if order_type in {"STOP", "STOP-ENTRY", "STOP_ENTRY"} else order_type
    if str(order.get("vehicle") or "").strip().upper() != vehicle:
        raise AILiveGatewayError("AI vehicle must match order_type transport semantics")
    multiplier = float(order["allocation_multiplier"])
    if multiplier not in {0.5, 0.75, 1.0}:
        raise AILiveGatewayError("live allocation_multiplier must be 0.5, 0.75, or 1.0")
    selected_units = int(order["units"])
    base_units = _base_units_for_selected(selected_units, multiplier)
    lane_id = "ai:" + hashlib.sha256(
        f"{run_id}\0{order['decision_id']}\0{pair}\0{side}".encode("utf-8")
    ).hexdigest()[:32] + f":{pair}:{side}"
    metadata = _audited_execution_metadata(
        repo_root / "data" / "order_intents.json",
        pair=pair,
        side=side,
        method=method,
        vehicle=vehicle,
    )
    metadata.update(
        {
            "desk": "ai_trader",
            "campaign_role": "NOW",
            "parent_lane_id": lane_id,
            "ai_decision_id": str(order["decision_id"]),
            "ai_run_id": run_id,
        }
    )
    extensions = order.get("extensions")
    extensions = extensions if isinstance(extensions, Mapping) else {}
    intent = {
        "pair": pair,
        "side": side,
        "order_type": "STOP-ENTRY" if vehicle == "STOP" else vehicle,
        "units": base_units,
        "entry": float(order["entry"]),
        "tp": float(order["take_profit"]),
        "sl": float(order["stop_loss"]),
        "thesis": str(candidate.get("thesis") or order["rationale"]),
        "reason": str(order["rationale"]),
        "owner": "trader",
        "market_context": {
            "regime": str(extensions.get("regime") or "AI_DISCRETIONARY"),
            "narrative": str(candidate.get("thesis") or order["rationale"]),
            "chart_story": str(order["rationale"]),
            "method": method,
            "invalidation": f"AI stop_loss {float(order['stop_loss'])}",
        },
        "metadata": metadata,
    }
    timestamp = generated_at.astimezone(timezone.utc).isoformat()
    intents = {
        "generated_at_utc": timestamp,
        "producer": "AI_PRIMARY_DECISION_RUNTIME",
        "results": [
            {
                "lane_id": lane_id,
                "status": "LIVE_READY",
                "risk_allowed": True,
                "intent": intent,
            }
        ],
    }
    surface = read_exact_vehicle_allocation_surface(
        repo_root / "data" / "execution_ledger.db"
    )
    exact_key = (pair, side, method, vehicle)
    cost_floor = execution_cost_floor_from_surface(
        surface, exact_key=exact_key, as_of=generated_at
    )
    board_material = {
        "run_id": run_id,
        "source_digest": candidate.get("source_digest"),
        "decision_id": order["decision_id"],
        "pair": pair,
        "side": side,
        "method": method,
        "vehicle": vehicle,
        "entry": float(order["entry"]),
        "take_profit": float(order["take_profit"]),
        "stop_loss": float(order["stop_loss"]),
        "selected_units": selected_units,
        "allocation_multiplier": multiplier,
    }
    board_sha = canonical_json_sha256(board_material)
    allocation = {
        "decision": "ALLOCATE",
        "lane_id": lane_id,
        "size_multiple": multiplier,
        "selected_units": selected_units,
        "allocation_board_sha256": board_sha,
        "rationale": str(order["rationale"]),
    }
    guardian_material = guardian_action_receipt_scope_material(
        repo_root / "data" / "guardian_action_receipt.json",
        baseline_pairs=[pair],
        as_of=generated_at,
    )
    decision = {
        "generated_at_utc": timestamp,
        "action": "TRADE",
        "selected_lane_id": lane_id,
        "selected_lane_ids": [lane_id],
        "confidence": candidate.get("confidence"),
        "thesis": candidate.get("thesis"),
        "method": method,
        "evidence_refs": list(candidate.get("evidence_refs") or []),
        "market_read_first": {
            "next_30m_prediction": {"pair": pair, "direction": side},
            "best_trade_if_forced": {
                "pair": pair,
                "direction": side,
                "vehicle": vehicle,
                "entry": str(order["entry"]),
                "tp": str(order["take_profit"]),
                "sl": str(order["stop_loss"]),
                "why_this_pays": str(order["rationale"]),
            },
        },
        "capital_allocation": allocation,
        "decision_provenance": {
            "schema_version": 2,
            "author_kind": "CODEX_MARKET_READ",
            "model": candidate.get("model"),
            "reasoning_effort": candidate.get("reasoning_effort"),
            "capital_allocation_edge_basis": _edge_basis(metadata, surface, exact_key),
            "capital_allocation_sha256": canonical_json_sha256(allocation),
            "capital_allocation_board_sha256": board_sha,
            "authorized_size_multiple": multiplier,
            "authorized_units": selected_units,
            "execution_cost_floor_sha256": cost_floor.get("proof_sha256"),
            "guardian_action_receipt_material_contract": GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT,
            "guardian_action_receipt_baseline_pairs": [pair],
            "guardian_action_receipt_scope_state_sha256": canonical_json_sha256(
                guardian_material
            ),
        },
    }
    broker_snapshot = _load_json(repo_root / "data" / "broker_snapshot.json")
    verified = {
        "generated_at_utc": timestamp,
        "status": "ACCEPTED",
        "decision": decision,
        "verification_issues": [],
        "input_packet": {
            "broker_snapshot": {
                "fetched_at_utc": broker_snapshot.get("fetched_at_utc")
            }
        },
    }
    receipt_id = _expected_decision_receipt_id(verified)
    if receipt_id is None:
        raise AILiveGatewayError("failed to create content-addressed decision receipt")
    verified["market_read_prediction"] = {
        "status": "RECORDED",
        "schema_version": 2,
        "prediction_id": "mr2:" + canonical_json_sha256(board_material),
        "decision_receipt_id": receipt_id,
        "read_only": True,
        "live_permission": False,
    }
    return {"lane_id": lane_id, "intents": intents, "verified_decision": verified}


def _audited_execution_metadata(
    path: Path, *, pair: str, side: str, method: str, vehicle: str
) -> dict[str, Any]:
    payload = _load_json(path)
    for row in payload.get("results", []):
        if not isinstance(row, Mapping) or not isinstance(row.get("intent"), Mapping):
            continue
        intent = row["intent"]
        context = intent.get("market_context")
        context = context if isinstance(context, Mapping) else {}
        transport = str(intent.get("order_type") or "").upper()
        transport = "STOP" if transport in {"STOP", "STOP-ENTRY", "STOP_ENTRY"} else transport
        identity = (
            str(intent.get("pair") or "").upper(),
            str(intent.get("side") or "").upper(),
            str(context.get("method") or "").upper(),
            transport,
        )
        if identity != (pair, side, method, vehicle):
            continue
        source = intent.get("metadata")
        source = source if isinstance(source, Mapping) else {}
        return {
            str(key): value
            for key, value in source.items()
            if str(key).startswith(_EVIDENCE_PREFIXES) or key in _EVIDENCE_KEYS
        }
    raise AILiveGatewayError(
        f"no current audited execution evidence for {pair}|{side}|{method}|{vehicle}"
    )


def _edge_basis(
    metadata: Mapping[str, Any],
    surface: Mapping[str, Any],
    exact_key: tuple[str, str, str, str],
) -> str:
    net = exact_vehicle_metrics_from_surface(surface, field="exact_vehicle_net") or {}
    tp = exact_vehicle_metrics_from_surface(surface, field="exact_vehicle_take_profit") or {}
    if _positive_edge(net.get(exact_key) or {}):
        return "EXACT_VEHICLE_ALL_EXIT_NET"
    if _positive_edge(tp.get(exact_key) or {}) and metadata.get("attach_take_profit_on_fill") is True:
        return "EXACT_VEHICLE_TAKE_PROFIT"
    return "EXACT_VEHICLE_ALL_EXIT_NET"


def _positive_edge(row: Mapping[str, Any]) -> bool:
    try:
        return (
            int(row.get("trades") or 0) >= 20
            and int(row.get("unresolved_realized_trades") or 0) == 0
            and math.isfinite(float(row.get("net_jpy") or 0.0))
            and float(row.get("net_jpy") or 0.0) > 0.0
            and float(row.get("expectancy_jpy_per_trade") or 0.0) > 0.0
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _base_units_for_selected(selected_units: int, multiplier: float) -> int:
    if multiplier == 1.0:
        return selected_units
    numerator, denominator = (1, 2) if multiplier == 0.5 else (3, 4)
    base = math.ceil(selected_units * denominator / numerator)
    if base * numerator // denominator != selected_units:
        raise AILiveGatewayError(
            "selected units cannot be represented by allocation multiplier"
        )
    return base


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AILiveGatewayError(
            f"required live evidence is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AILiveGatewayError(f"required live evidence is not an object: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
