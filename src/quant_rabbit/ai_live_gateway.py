from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_rabbit.ai_evidence_adapter import CONTRACT as AI_EVIDENCE_CONTRACT
from quant_rabbit.capture_economics import (
    evaluate_exact_vehicle_net_edge,
    exact_vehicle_metrics_from_surface,
    execution_cost_floor_from_surface,
    read_exact_vehicle_allocation_surface,
)
from quant_rabbit.decision_execution_lineage import _expected_decision_receipt_id
from quant_rabbit.entry_decision import (
    ENTRY_DECISION_CONTRACT,
    EntryDecisionError,
    validate_entry_decision,
)
from quant_rabbit.market_read_overlay import (
    GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT,
    canonical_json_sha256,
    guardian_action_receipt_scope_material,
)
from quant_rabbit.policy_snapshot import PolicyBinding, PolicySnapshotError, verify_policy_snapshot


class AILiveGatewayError(RuntimeError):
    pass


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NET_EDGE_FIELDS = (
    "trades", "wins", "losses", "net_jpy", "expectancy_jpy_per_trade",
    "avg_win_jpy", "avg_loss_jpy", "unresolved_realized_trades",
    "unresolved_realized_net_jpy",
)


def execute_ai_trade_candidate(
    *, repo_root: Path, state_root: Path, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Send one sealed qre entry through the existing one-shot gateway."""

    entry_decision = receipt.get("entry_decision")
    if not isinstance(entry_decision, Mapping):
        raise AILiveGatewayError(
            "live gateway requires receipt.entry_decision with a sealed qre decision"
        )
    if entry_decision.get("contract") != ENTRY_DECISION_CONTRACT:
        raise AILiveGatewayError("legacy AI/order-intent candidates are not executable")
    evidence_packet = receipt.get("evidence_packet")
    if not isinstance(evidence_packet, Mapping):
        raise AILiveGatewayError("live gateway requires the bound compact evidence packet")
    run_id = str(receipt.get("run_id") or "").strip()
    if not run_id:
        raise AILiveGatewayError("accepted AI receipt has no run_id")
    expected_cycle = str(receipt.get("cycle_id") or run_id).strip()
    now = datetime.now(timezone.utc)
    broker_epoch = broker_epoch_from_evidence_packet(evidence_packet)
    try:
        validated = validate_entry_decision(
            entry_decision,
            expected_cycle_id=expected_cycle,
            expected_broker_epoch=broker_epoch,
            now_utc=now,
        )
    except EntryDecisionError as exc:
        raise AILiveGatewayError(f"sealed entry decision rejected: {exc.code}: {exc}") from exc

    action = str(validated.get("action") or "").strip().upper()
    if action != "ENTER":
        return {
            "status": "NO_BROKER_ACTION",
            "sink": "live_gateway",
            "broker_mutation_allowed": False,
            "broker_order_posts": 0,
            "sent": False,
            "reason": f"{action or 'UNKNOWN'} does not authorize a fresh entry",
            "decision_id": validated.get("decision_id"),
        }
    _validate_active_hotpath_lease(
        state_root=state_root,
        receipt_lease=receipt.get("hotpath_lease"),
        run_id=run_id,
        now=now,
    )
    _validate_live_policy(receipt.get("policy_snapshot"), now=now)
    proposals = validated.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != 1:
        raise AILiveGatewayError("live ENTER requires exactly one sealed proposal")

    artifacts = build_live_gateway_artifacts(
        repo_root=repo_root,
        entry_decision=validated,
        evidence_packet=evidence_packet,
        run_id=run_id,
        generated_at=now,
        candidate_context=receipt,
    )
    run_dir = state_root / "live" / "runs" / run_id
    intents_path = run_dir / "intents.json"
    verified_path = run_dir / "verified_decision.json"
    output_path = run_dir / "gateway_receipt.json"
    report_path = run_dir / "gateway_report.md"
    _atomic_write_json(intents_path, artifacts["intents"])
    _atomic_write_json(verified_path, artifacts["verified_decision"])

    command = [
        sys.executable, "-m", "quant_rabbit.cli", "stage-live-order",
        "--intents", str(intents_path),
        "--strategy-profile", str(repo_root / "data" / "strategy_profile.json"),
        "--lane-id", artifacts["lane_id"],
        "--output", str(output_path),
        "--report", str(report_path),
        "--verified-decision", str(verified_path),
        "--execution-ledger-db", str(repo_root / "data" / "execution_ledger.db"),
        "--execution-ledger-report", str(repo_root / "docs" / "execution_ledger_report.md"),
        "--target-state", str(repo_root / "data" / "daily_target_state.json"),
        "--target-report", str(repo_root / "docs" / "daily_target_report.md"),
        "--send", "--confirm-live",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root / "src")
    completed = subprocess.run(
        command, cwd=repo_root, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240, check=False,
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
        "decision_id": validated.get("decision_id"),
        "gateway_receipt_path": str(output_path),
        "gateway_report_path": str(report_path),
        "risk_issues": gateway.get("risk_issues", []),
        "strategy_issues": gateway.get("strategy_issues", []),
        "command_exit_code": completed.returncode,
    }


def build_live_gateway_artifacts(
    *,
    repo_root: Path,
    entry_decision: Mapping[str, Any],
    evidence_packet: Mapping[str, Any],
    run_id: str,
    generated_at: datetime,
    candidate_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build transport artifacts without changing the AI-selected units."""

    now = _aware_utc(generated_at, "generated_at")
    packet = _validate_evidence_packet(evidence_packet, now=now)
    broker_epoch = broker_epoch_from_evidence_packet(packet)
    cycle_id = str(entry_decision.get("cycle_id") or "").strip()
    try:
        decision = validate_entry_decision(
            entry_decision,
            expected_cycle_id=cycle_id,
            expected_broker_epoch=broker_epoch,
            now_utc=now,
        )
    except EntryDecisionError as exc:
        raise AILiveGatewayError(f"sealed entry decision rejected: {exc.code}: {exc}") from exc
    if decision.get("action") != "ENTER":
        raise AILiveGatewayError("artifact construction requires an ENTER decision")
    proposals = decision.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != 1 or not isinstance(proposals[0], Mapping):
        raise AILiveGatewayError("live ENTER requires exactly one sealed proposal")
    proposal = proposals[0]
    _reject_legacy_fields(proposal)

    pair = _required_upper(proposal, "pair")
    side = _required_upper(proposal, "side")
    method = _required_upper(proposal, "method").replace("-", "_")
    raw_vehicle = _required_upper(proposal, "vehicle").replace("_", "-")
    vehicle = "STOP" if raw_vehicle in {"STOP", "STOP-ENTRY"} else raw_vehicle
    if vehicle not in {"MARKET", "LIMIT", "STOP"}:
        raise AILiveGatewayError("entry proposal vehicle must be MARKET, LIMIT, or STOP-ENTRY")
    if proposal.get("order_type") is not None:
        raw_order_type = str(proposal.get("order_type") or "").strip().upper().replace("_", "-")
        order_vehicle = "STOP" if raw_order_type in {"STOP", "STOP-ENTRY"} else raw_order_type
        if order_vehicle != vehicle:
            raise AILiveGatewayError("entry proposal vehicle and order_type disagree")
    selected_units = proposal.get("units")
    if isinstance(selected_units, bool) or not isinstance(selected_units, int) or selected_units <= 0:
        raise AILiveGatewayError("entry proposal units must be a positive integer")
    sizing = proposal.get("sizing_receipt")
    if not isinstance(sizing, Mapping) or sizing.get("final_units") != selected_units:
        raise AILiveGatewayError("AI units must equal sizing_receipt.final_units")
    _validate_sizing_evidence_binding(sizing=sizing, packet=packet, proposal=proposal)

    evidence_binding = proposal.get("evidence_binding")
    if not isinstance(evidence_binding, Mapping):
        raise AILiveGatewayError("entry proposal has no evidence_binding")
    expected_binding = {
        "packet_sha256": packet["packet_sha256"],
        "source_set_sha256": packet["source_set_sha256"],
        "broker_epoch": broker_epoch,
    }
    if dict(evidence_binding) != expected_binding:
        raise AILiveGatewayError("entry evidence binding does not match the sealed packet")

    exact_key = (pair, side, method, vehicle)
    net_proof = proposal.get("net_edge_proof")
    cost_proof = proposal.get("cost_proof")
    if not isinstance(net_proof, Mapping) or not isinstance(cost_proof, Mapping):
        raise AILiveGatewayError("entry proposal requires explicit net_edge_proof and cost_proof")
    surface = read_exact_vehicle_allocation_surface(repo_root / "data" / "execution_ledger.db")
    fresh_net, metadata = _validate_net_edge_proof(
        packet=packet, proof=net_proof, surface=surface, exact_key=exact_key,
    )
    fresh_cost = _validate_cost_proof(
        packet=packet, proof=cost_proof, surface=surface,
        exact_key=exact_key, as_of=now,
    )

    if "entry_price" in proposal and "entry" in proposal and proposal.get("entry_price") != proposal.get("entry"):
        raise AILiveGatewayError("entry and entry_price disagree")
    entry = _positive_price(proposal.get("entry_price", proposal.get("entry")), "entry_price")
    take_profit = _positive_price(proposal.get("take_profit"), "take_profit")
    stop_loss = _positive_price(proposal.get("stop_loss"), "stop_loss")
    _validate_geometry(side=side, entry=entry, take_profit=take_profit, stop_loss=stop_loss)
    decision_id = str(decision["decision_id"])
    lane_id = "ai:" + hashlib.sha256(
        f"{run_id}\0{decision_id}\0{pair}\0{side}".encode("utf-8")
    ).hexdigest()[:32] + f":{pair}:{side}"
    metadata.update(
        {
            "desk": "ai_trader",
            "campaign_role": "NOW",
            "parent_lane_id": lane_id,
            "ai_decision_id": decision_id,
            "ai_run_id": run_id,
            "ai_evidence_packet_sha256": packet["packet_sha256"],
            "ai_evidence_source_set_sha256": packet["source_set_sha256"],
            "ai_broker_epoch": broker_epoch,
            "forecast_cycle_id": (
                f"pre-entry-forecast-refresh:{decision['created_at_utc']}:"
                f"{decision_id[:20]}"
            ),
            "forecast_target_price": take_profit,
            "forecast_invalidation_price": stop_loss,
            "forecast_direction": side,
            "forecast_confidence": proposal.get("confidence"),
            "forecast_horizon_min": 10,
        }
    )
    rationale = _proposal_rationale(proposal, decision)
    intent = {
        "pair": pair, "side": side,
        "order_type": "STOP-ENTRY" if vehicle == "STOP" else vehicle,
        "units": selected_units, "entry": entry, "tp": take_profit, "sl": stop_loss,
        "thesis": rationale, "reason": rationale, "owner": "trader",
        "market_context": {
            "regime": str(proposal.get("regime") or "AI_DISCRETIONARY"),
            "narrative": rationale, "chart_story": rationale, "method": method,
            "invalidation": f"AI stop_loss {stop_loss}",
        },
        "metadata": metadata,
    }
    timestamp = now.isoformat()
    intents = {
        "generated_at_utc": timestamp,
        "producer": "AI_PRIMARY_DECISION_RUNTIME",
        "results": [{"lane_id": lane_id, "status": "LIVE_READY", "risk_allowed": True, "intent": intent}],
    }
    board_material = {
        "run_id": run_id, "cycle_id": decision["cycle_id"],
        "broker_epoch": broker_epoch, "decision_id": decision_id,
        "evidence_packet_sha256": packet["packet_sha256"],
        "source_set_sha256": packet["source_set_sha256"],
        "pair": pair, "side": side, "method": method, "vehicle": vehicle,
        "entry": entry, "take_profit": take_profit, "stop_loss": stop_loss,
        "selected_units": selected_units,
        "sizing_receipt_sha256": canonical_json_sha256(sizing),
        "net_edge_proof_sha256": canonical_json_sha256(net_proof),
        "cost_proof_sha256": canonical_json_sha256(cost_proof),
    }
    board_sha = canonical_json_sha256(board_material)
    allocation = {
        "decision": "ALLOCATE", "lane_id": lane_id,
        # Existing RiskEngine calls an exact, already-sized order a 1.0 multiple.
        # This is transport metadata, not an AI input or a unit reconstruction.
        "size_multiple": 1.0, "selected_units": selected_units,
        "allocation_board_sha256": board_sha, "rationale": rationale,
    }
    guardian_material = guardian_action_receipt_scope_material(
        repo_root / "data" / "guardian_action_receipt.json",
        baseline_pairs=[pair], as_of=now,
    )
    context = candidate_context if isinstance(candidate_context, Mapping) else {}
    verified_decision = {
        "generated_at_utc": timestamp, "action": "TRADE",
        "selected_lane_id": lane_id, "selected_lane_ids": [lane_id],
        "confidence": proposal.get("confidence"), "thesis": rationale,
        "method": method,
        "evidence_refs": [f"ai-evidence:{packet['packet_sha256']}", f"capture:{net_proof.get('source_sha256')}"],
        "market_read_first": {
            "next_30m_prediction": {"pair": pair, "direction": side},
            "best_trade_if_forced": {
                "pair": pair, "direction": side, "vehicle": vehicle,
                "entry": str(entry), "tp": str(take_profit), "sl": str(stop_loss),
                "why_this_pays": rationale,
            },
        },
        "capital_allocation": allocation,
        "decision_provenance": {
            "schema_version": 2, "author_kind": "CODEX_MARKET_READ",
            "model": context.get("model"), "reasoning_effort": context.get("reasoning_effort"),
            "entry_decision_id": decision_id,
            "entry_decision_contract": ENTRY_DECISION_CONTRACT,
            "entry_sizing_receipt_sha256": canonical_json_sha256(sizing),
            "evidence_packet_sha256": packet["packet_sha256"],
            "evidence_source_set_sha256": packet["source_set_sha256"],
            "broker_epoch": broker_epoch,
            "capital_allocation_edge_basis": "EXACT_VEHICLE_ALL_EXIT_NET",
            "capital_allocation_sha256": canonical_json_sha256(allocation),
            "capital_allocation_board_sha256": board_sha,
            "authorized_size_multiple": 1.0, "authorized_units": selected_units,
            "execution_cost_floor_sha256": fresh_cost["proof_sha256"],
            "net_edge_recheck_sha256": canonical_json_sha256(fresh_net),
            "guardian_action_receipt_material_contract": GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT,
            "guardian_action_receipt_baseline_pairs": [pair],
            "guardian_action_receipt_scope_state_sha256": canonical_json_sha256(guardian_material),
        },
    }
    verified = {
        "generated_at_utc": timestamp, "status": "ACCEPTED",
        "decision": verified_decision, "verification_issues": [],
        "input_packet": {"broker_snapshot": {"fetched_at_utc": packet["broker_epoch"].get("as_of_utc")}},
    }
    receipt_id = _expected_decision_receipt_id(verified)
    if receipt_id is None:
        raise AILiveGatewayError("failed to create content-addressed decision receipt")
    verified["market_read_prediction"] = {
        "status": "RECORDED", "schema_version": 2,
        "prediction_id": "mr2:" + canonical_json_sha256(board_material),
        "decision_receipt_id": receipt_id, "read_only": True, "live_permission": False,
    }
    return {"lane_id": lane_id, "intents": intents, "verified_decision": verified}


def broker_epoch_from_evidence_packet(packet: Mapping[str, Any]) -> str:
    """Return the broker transaction epoch carried by the sealed packet.

    The mutation path separately verifies that ``source_sha256`` equals the
    broker source descriptor.  Keeping the public epoch as the broker's exact
    transaction id also lets Entry, Exit, and adjudication share one value.
    """

    if not isinstance(packet, Mapping):
        raise AILiveGatewayError("evidence packet must be an object")
    epoch = packet.get("broker_epoch")
    if not isinstance(epoch, Mapping):
        raise AILiveGatewayError("evidence packet has no broker_epoch")
    transaction_id = str(epoch.get("last_transaction_id") or "").strip()
    as_of = str(epoch.get("as_of_utc") or "").strip()
    if not transaction_id or not as_of:
        raise AILiveGatewayError("evidence packet broker_epoch is incomplete")
    _parse_utc(as_of, "broker_epoch.as_of_utc")
    return transaction_id


def _validate_live_policy(value: Any, *, now: datetime) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AILiveGatewayError("live ENTER requires the verified sealed policy snapshot")
    required_env = {
        "project_key": os.environ.get("QR_AI_PROJECT_KEY"),
        "broker_account_id": os.environ.get("QR_AI_BROKER_ACCOUNT_ID"),
        "environment": os.environ.get("QR_AI_ENVIRONMENT"),
        "revocation_epoch": os.environ.get("QR_AI_POLICY_REVOCATION_EPOCH"),
    }
    missing = [name for name, raw in required_env.items() if raw is None or not str(raw).strip()]
    if missing:
        raise AILiveGatewayError(
            "live policy binding environment is incomplete: " + ", ".join(missing)
        )
    try:
        revocation_epoch = int(str(required_env["revocation_epoch"]))
    except ValueError as exc:
        raise AILiveGatewayError("live policy revocation epoch is invalid") from exc
    required_pages = tuple(
        page.strip()
        for page in os.environ.get("QR_AI_REQUIRED_POLICY_SOURCE_PAGES", "").split(",")
        if page.strip()
    )
    if not required_pages:
        raise AILiveGatewayError("live ENTER requires explicit policy source-page bindings")
    try:
        return verify_policy_snapshot(
            value,
            binding=PolicyBinding(
                project_key=str(required_env["project_key"]).strip(),
                broker_account_id=str(required_env["broker_account_id"]).strip(),
                environment=str(required_env["environment"]).strip(),
                revocation_epoch=revocation_epoch,
            ),
            now=now,
            required_source_pages=required_pages,
        )
    except PolicySnapshotError as exc:
        raise AILiveGatewayError(f"sealed live policy rejected: {exc.code}: {exc}") from exc


def _validate_active_hotpath_lease(
    *,
    state_root: Path,
    receipt_lease: Any,
    run_id: str,
    now: datetime,
) -> None:
    if not isinstance(receipt_lease, Mapping):
        raise AILiveGatewayError("live ENTER requires the active hot-path lease")
    path = state_root / "hotpath_lease.json"
    current = _load_json(path)
    if dict(receipt_lease) != current:
        raise AILiveGatewayError("hot-path lease changed before broker mutation")
    supplied = current.get("lease_sha256")
    material = {key: value for key, value in current.items() if key != "lease_sha256"}
    if not isinstance(supplied, str) or supplied != canonical_json_sha256(material):
        raise AILiveGatewayError("hot-path lease seal mismatch")
    if current.get("status") != "ACTIVE" or current.get("run_id") != run_id:
        raise AILiveGatewayError("hot-path lease is not active for this run")
    if now >= _parse_utc(current.get("expires_at_utc"), "hotpath_lease.expires_at_utc"):
        raise AILiveGatewayError("hot-path lease expired before broker mutation")


def _validate_sizing_evidence_binding(
    *,
    sizing: Mapping[str, Any],
    packet: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> None:
    portfolio = packet.get("portfolio")
    daily = portfolio.get("daily_target") if isinstance(portfolio, Mapping) else None
    if not isinstance(daily, Mapping):
        raise AILiveGatewayError("sealed packet has no daily risk state for sizing")
    remaining = _nonnegative_number(
        daily.get("remaining_risk_budget_jpy"),
        "portfolio.daily_target.remaining_risk_budget_jpy",
    )
    if remaining <= 0.0:
        raise AILiveGatewayError("sealed daily risk capacity is exhausted")
    budgets = sizing.get("risk_budget_components")
    if not isinstance(budgets, Mapping):
        raise AILiveGatewayError("sizing receipt has no risk budget components")
    claimed_daily = _nonnegative_number(budgets.get("daily_remaining"), "daily_remaining")
    if not math.isclose(claimed_daily, remaining, rel_tol=0.0, abs_tol=1e-9):
        raise AILiveGatewayError("sizing daily_remaining does not match sealed portfolio truth")
    if _positive_number(budgets.get("portfolio_allowance"), "portfolio_allowance") > remaining:
        raise AILiveGatewayError("sizing portfolio allowance exceeds sealed daily capacity")
    margin = portfolio.get("margin")
    nav = _positive_number(
        margin.get("nav_jpy") if isinstance(margin, Mapping) else None,
        "portfolio.margin.nav_jpy",
    )
    if _positive_number(budgets.get("nav_risk_ceiling"), "nav_risk_ceiling") > nav:
        raise AILiveGatewayError("sizing NAV risk ceiling exceeds sealed NAV")

    pair = _required_upper(proposal, "pair")
    entry = _positive_price(proposal.get("entry_price", proposal.get("entry")), "entry_price")
    stop_loss = _positive_price(proposal.get("stop_loss"), "stop_loss")
    quote_currency = pair.split("_", 1)[1] if "_" in pair else ""
    broker = packet.get("broker")
    conversions = broker.get("home_conversions") if isinstance(broker, Mapping) else None
    conversion = (
        1.0
        if quote_currency == "JPY"
        else _positive_number(
            conversions.get(quote_currency) if isinstance(conversions, Mapping) else None,
            f"broker.home_conversions.{quote_currency}",
        )
    )
    expected_loss_per_unit = abs(entry - stop_loss) * conversion
    claimed_loss_per_unit = _positive_number(
        sizing.get("loss_per_unit_at_stop"), "loss_per_unit_at_stop"
    )
    if not math.isclose(
        claimed_loss_per_unit,
        expected_loss_per_unit,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise AILiveGatewayError(
            "sizing loss_per_unit_at_stop does not match sealed conversion and entry-stop geometry"
        )

    exposure = broker.get("exposure") if isinstance(broker, Mapping) else None
    if not isinstance(exposure, Mapping):
        raise AILiveGatewayError("sealed packet has no broker exposure for sizing")
    expected: dict[str, str] = {}
    for kind, rows, identity in (
        ("position", exposure.get("positions"), "trade_id"),
        ("order", exposure.get("pending_orders"), "order_id"),
    ):
        if not isinstance(rows, list):
            raise AILiveGatewayError("sealed broker exposure rows are malformed")
        for row in rows:
            if not isinstance(row, Mapping) or not str(row.get(identity) or "").strip():
                raise AILiveGatewayError("sealed broker exposure identity is incomplete")
            owner = str(row.get("ownership") or "UNKNOWN").strip().upper()
            expected[f"{kind}:{str(row[identity]).strip()}"] = owner
    actual_rows = sizing.get("exposures")
    if not isinstance(actual_rows, list):
        raise AILiveGatewayError("sizing exposure rows are missing")
    actual = {
        str(row.get("reference") or "").strip(): str(row.get("reported_owner") or "UNKNOWN").strip().upper()
        for row in actual_rows
        if isinstance(row, Mapping)
    }
    normalized_expected = {
        ref: ("MANUAL" if owner == "OPERATOR_MANUAL" else owner)
        for ref, owner in expected.items()
    }
    if actual != normalized_expected:
        raise AILiveGatewayError("sizing exposures do not exactly cover sealed broker exposure")


def _validate_evidence_packet(packet: Mapping[str, Any], *, now: datetime) -> dict[str, Any]:
    normalized = dict(packet)
    if normalized.get("contract") != AI_EVIDENCE_CONTRACT or normalized.get("schema_version") != 1:
        raise AILiveGatewayError("unsupported compact evidence packet contract")
    if normalized.get("status") != "READY":
        raise AILiveGatewayError("compact evidence packet is not READY")
    supplied_sha = str(normalized.get("packet_sha256") or "").strip().lower()
    material = {key: value for key, value in normalized.items() if key != "packet_sha256"}
    if not _SHA256_RE.fullmatch(supplied_sha) or canonical_json_sha256(material) != supplied_sha:
        raise AILiveGatewayError("compact evidence packet seal mismatch")
    sources = normalized.get("sources")
    if not isinstance(sources, Mapping):
        raise AILiveGatewayError("compact evidence packet sources are missing")
    source_set_sha = str(normalized.get("source_set_sha256") or "").strip().lower()
    if not _SHA256_RE.fullmatch(source_set_sha) or canonical_json_sha256(sources) != source_set_sha:
        raise AILiveGatewayError("compact evidence source-set seal mismatch")
    for name, descriptor in sources.items():
        if not isinstance(descriptor, Mapping):
            raise AILiveGatewayError(f"evidence source {name} is malformed")
        if descriptor.get("required") is True and descriptor.get("status") != "READY":
            raise AILiveGatewayError(f"required evidence source {name} is not READY")
        if descriptor.get("status") == "READY":
            stale_after = _parse_utc(descriptor.get("stale_after_utc"), f"sources.{name}.stale_after_utc")
            if now > stale_after:
                raise AILiveGatewayError(f"evidence source {name} is stale at gateway entry")
    broker_source = sources.get("broker_snapshot")
    epoch = normalized.get("broker_epoch")
    if not isinstance(broker_source, Mapping) or not isinstance(epoch, Mapping):
        raise AILiveGatewayError("broker evidence is missing")
    broker_sha = str(broker_source.get("sha256") or "").strip().lower()
    if (
        broker_source.get("status") != "READY"
        or not _SHA256_RE.fullmatch(broker_sha)
        or epoch.get("source_sha256") != broker_sha
    ):
        raise AILiveGatewayError("broker epoch source does not match broker evidence")
    broker_epoch_from_evidence_packet(normalized)
    return normalized


def _validate_net_edge_proof(
    *, packet: Mapping[str, Any], proof: Mapping[str, Any], surface: Mapping[str, Any],
    exact_key: tuple[str, str, str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pair, side, method, vehicle = exact_key
    identity = tuple(
        str(proof.get(key) or "").strip().upper().replace("-ENTRY", "")
        for key in ("pair", "side", "method", "vehicle")
    )
    if identity != exact_key:
        raise AILiveGatewayError("net-edge proof does not match proposal identity")
    sources = packet["sources"]
    capture_source = sources.get("capture_economics")
    if not isinstance(capture_source, Mapping) or capture_source.get("status") != "READY":
        raise AILiveGatewayError("capture-economics source is unavailable")
    capture_sha = str(capture_source.get("sha256") or "").strip().lower()
    if (
        not _SHA256_RE.fullmatch(capture_sha)
        or str(proof.get("source_sha256") or "").strip().lower() != capture_sha
    ):
        raise AILiveGatewayError("net-edge proof source does not match the evidence packet")
    packet_net = packet.get("net_edge_inputs")
    segments = packet_net.get("segments") if isinstance(packet_net, Mapping) else None
    if not isinstance(segments, list):
        raise AILiveGatewayError("compact evidence packet has no net-edge segments")
    comparable_keys = (
        "pair", "side", "method", "trades", "wins", "losses", "net_jpy",
        "expectancy_jpy_per_trade", "avg_win_jpy", "avg_loss_jpy",
    )
    if not any(
        isinstance(row, Mapping) and all(row.get(key) == proof.get(key) for key in comparable_keys)
        for row in segments
    ):
        raise AILiveGatewayError("net-edge proof is not present in the sealed evidence packet")
    metrics = {field: proof.get(field) for field in _NET_EDGE_FIELDS}
    if evaluate_exact_vehicle_net_edge(metrics).get("proven") is not True:
        raise AILiveGatewayError("net-edge proof is nonpositive, incomplete, or unreconciled")
    after_cost = proof.get("net_edge_after_cost_jpy")
    if isinstance(after_cost, bool):
        raise AILiveGatewayError("net edge after cost must be numeric")
    try:
        after_cost_number = float(after_cost)
    except (TypeError, ValueError) as exc:
        raise AILiveGatewayError("net edge after cost is missing") from exc
    if not math.isfinite(after_cost_number) or after_cost_number <= 0:
        raise AILiveGatewayError("net edge after spread/slippage/swap/latency must be positive")
    fresh_rows = exact_vehicle_metrics_from_surface(surface, field="exact_vehicle_net")
    if not isinstance(fresh_rows, Mapping) or not isinstance(fresh_rows.get(exact_key), Mapping):
        raise AILiveGatewayError("current exact-vehicle net-edge row is missing")
    fresh = fresh_rows[exact_key]
    if evaluate_exact_vehicle_net_edge(fresh).get("proven") is not True:
        raise AILiveGatewayError("current exact-vehicle net edge is not positive")
    for field in _NET_EDGE_FIELDS:
        expected = metrics.get(field)
        actual = fresh.get(field)
        if expected is None and field in {"unresolved_realized_trades", "unresolved_realized_net_jpy"}:
            expected = 0 if field.endswith("trades") else 0.0
        if actual is None and field in {"unresolved_realized_trades", "unresolved_realized_net_jpy"}:
            actual = 0 if field.endswith("trades") else 0.0
        if expected != actual:
            raise AILiveGatewayError(f"net-edge proof no longer matches current ledger field {field}")
    fresh_expectancy = _positive_number(
        fresh.get("expectancy_jpy_per_trade"),
        "current exact-vehicle net expectancy",
    )
    if not math.isclose(after_cost_number, fresh_expectancy, rel_tol=0.0, abs_tol=1e-9):
        raise AILiveGatewayError(
            "declared after-cost edge does not equal current audited net expectancy"
        )
    metadata = {
        "capture_exact_vehicle_net_scope": "PAIR_SIDE_METHOD_VEHICLE",
        "capture_exact_vehicle_net_scope_key": f"{pair}|{side}|{method}|{vehicle}|ALL_AUDITED_EXITS",
        "capture_exact_vehicle_net_vehicle": vehicle,
        "capture_exact_vehicle_net_metrics_source": "data/execution_ledger.db:exact_vehicle_net",
        "capture_exact_vehicle_net_exit_scope": "ALL_AUDITED_EXITS",
        "capture_exact_vehicle_net_trades": metrics["trades"],
        "capture_exact_vehicle_net_wins": metrics["wins"],
        "capture_exact_vehicle_net_losses": metrics["losses"],
        "capture_exact_vehicle_net_jpy": metrics["net_jpy"],
        "capture_exact_vehicle_net_expectancy_jpy": metrics["expectancy_jpy_per_trade"],
        "capture_exact_vehicle_net_avg_win_jpy": metrics["avg_win_jpy"],
        "capture_exact_vehicle_net_avg_loss_jpy": metrics["avg_loss_jpy"],
        "capture_exact_vehicle_net_unresolved_realized_trades": metrics.get("unresolved_realized_trades", 0),
        "capture_exact_vehicle_net_unresolved_realized_net_jpy": metrics.get("unresolved_realized_net_jpy", 0.0),
        "attach_stop_loss_on_fill": True, "attach_take_profit_on_fill": True,
        "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
    }
    return dict(fresh), metadata


def _validate_cost_proof(
    *, packet: Mapping[str, Any], proof: Mapping[str, Any], surface: Mapping[str, Any],
    exact_key: tuple[str, str, str, str], as_of: datetime,
) -> dict[str, Any]:
    costs = packet.get("costs")
    facts = costs.get("spread_slippage_latency_swap_facts") if isinstance(costs, Mapping) else None
    if not isinstance(facts, list) or not facts:
        raise AILiveGatewayError("compact evidence packet has no execution-cost facts")
    if str(proof.get("packet_costs_sha256") or "").strip().lower() != canonical_json_sha256(costs):
        raise AILiveGatewayError("cost proof does not match compact evidence costs")
    fresh = execution_cost_floor_from_surface(surface, exact_key=exact_key, as_of=as_of)
    if not isinstance(fresh, Mapping) or fresh.get("status") != "PASSED":
        raise AILiveGatewayError("current execution-cost floor is unavailable or failed")
    expected_sha = str(proof.get("execution_cost_floor_sha256") or "").strip().lower()
    actual_sha = str(fresh.get("proof_sha256") or "").strip().lower()
    if not _SHA256_RE.fullmatch(expected_sha) or expected_sha != actual_sha:
        raise AILiveGatewayError("execution-cost proof is stale or mismatched")
    return dict(fresh)


def _reject_legacy_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).strip().lower() in {"allocation_multiplier", "order_intents", "order_intent"}:
                raise AILiveGatewayError(f"legacy field is not executable: {key}")
            _reject_legacy_fields(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _reject_legacy_fields(nested)


def _proposal_rationale(proposal: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    direct = str(proposal.get("rationale") or "").strip()
    if direct:
        return direct
    reasons = decision.get("reasons")
    if isinstance(reasons, list):
        joined = "; ".join(str(reason).strip() for reason in reasons if str(reason).strip())
        if joined:
            return joined
    return "sealed AI entry decision"


def _validate_geometry(*, side: str, entry: float, take_profit: float, stop_loss: float) -> None:
    if side == "LONG" and not (stop_loss < entry < take_profit):
        raise AILiveGatewayError("LONG requires stop_loss < entry < take_profit")
    if side == "SHORT" and not (take_profit < entry < stop_loss):
        raise AILiveGatewayError("SHORT requires take_profit < entry < stop_loss")


def _required_upper(value: Mapping[str, Any], key: str) -> str:
    result = str(value.get(key) or "").strip().upper()
    if not result:
        raise AILiveGatewayError(f"entry proposal has no {key}")
    return result


def _positive_price(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise AILiveGatewayError(f"{key} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AILiveGatewayError(f"{key} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise AILiveGatewayError(f"{key} must be a positive finite number")
    return result


def _positive_number(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise AILiveGatewayError(f"{key} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AILiveGatewayError(f"{key} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise AILiveGatewayError(f"{key} must be a positive finite number")
    return result


def _nonnegative_number(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise AILiveGatewayError(f"{key} must be a non-negative finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AILiveGatewayError(f"{key} must be a non-negative finite number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise AILiveGatewayError(f"{key} must be a non-negative finite number")
    return result


def _parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise AILiveGatewayError(f"{field} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AILiveGatewayError(f"{field} is invalid") from exc
    return _aware_utc(parsed, field)


def _aware_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AILiveGatewayError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AILiveGatewayError(f"required live evidence is unreadable: {path}: {exc}") from exc
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
